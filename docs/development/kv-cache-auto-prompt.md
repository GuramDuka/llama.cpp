# KV Cache Auto-Save/Restore -- Design Specification

## Overview

`llama-server` persists KV cache state across a **unified 3-tier hierarchy**: L1 (GPU slots), L2 (RAM prompt cache), and L3 (disk cache). When a request arrives, all matching entries from all three tiers are collected into a single sorted pool. The best candidate is selected by a composite sort (similarity DESC, freshness DESC, tier priority) and promoted according to tier rules.

Saves to L2 and L3 occur **only on successful completion**; interrupted or cancelled requests skip all persistence.

---

## 3-Tier Architecture

```
  Request arrives
       |
       v
  +---------------------------------------------+
  | Build combined candidate pool               |
  |                                             |
  | L1 (slots):   get_common_prefix for idle    |
  | L2 (RAM):     find_all_matching()           |
  | L3 (disk):    find_all_matching_entries()   |
  +---------------------+-----------------------+
                        |
                        v
  +---------------------------------------------+
  | Sort pool: similarity DESC                  |
  |            freshness DESC                   |
  |            tier priority (L1 < L2 < L3)     |
  +---------------------+-----------------------+
                        |
            +-----------+-----------+
            |                       |
            v                       v
       best.tier == L1        best.tier == L2
       Use slot directly      Load L2 -> L1
                                  |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
       best.tier == L3                                  No candidates
       Restore L3 -> L1 + promote L3 -> L2              LRU slot
            |
            v
  +---------------------------------------------+
  | Process request (generate)                  |
  +---------------------+-----------------------+
                        |
            +-----------+-----------+
            |                       |
            v                       v
     completed_successfully     interrupted/error
            |                       |
            v                       v
     Save L1->L2+L3            Skip all saves
            |
            v
       slot.reset()
```

---

## Data Structures

### cache_candidate (temporary, built per request)

```cpp
struct cache_candidate {
    enum tier_t { TIER_L1_SLOT, TIER_L2_RAM, TIER_L3_DISK };

    tier_t        tier;         // origin tier
    float         similarity;   // LCP ratio to request tokens
    int64_t       freshness;    // timestamp_us for tie-breaking
    server_slot * slot;         // L1 only
    std::string   filepath;     // L3 only
};
```

### L3 -- kv_cache_trie (Radix Tree)

Each node maps a token ID to child nodes. Entry indices (pointing into the LRU ring) are stored at the deepest matching node, enabling O(m log k) prefix matching.

- **Insert**: O(m log k) -- m = token sequence length, k = vocabulary size
- **Search**: traverses the trie, collects entry indices from the deepest matching node
- **Remove**: walks all nodes and removes stale entry indices

### L3 -- kv_cache_disk_manager

Central class for disk cache. Owns:

- `lru_ring_` -- ring buffer of `ring_buffer_entry` (metadata + access order)
- `filepath_index_` -- O(1) map from filepath to ring index
- `trie_` -- Radix Tree for prefix matching
- `metrics_` -- `kv_cache_metrics` counters (hits, misses, saves, evictions)

### L3 -- disk_cache_entry

```cpp
struct disk_cache_entry {
    std::string          filepath;
    int64_t              created_at_us;
    int64_t              last_accessed_us;
    size_t               file_size_bytes;
    uint32_t             seq_id;
    std::vector<int32_t> tokens;   // first N tokens for LCP matching
};
```

### L2 -- server_prompt_cache

In-memory ring buffer of `server_prompt` states with:

- `find_all_matching(tokens_new, threshold)` -- returns ALL matching entries
- `load(prompt, tokens, ctx, slot)` -- restores KV from RAM to GPU
- `alloc()` -- allocates space for new entry
- `update()` -- applies pending changes

### L2 -- prompt_cache_match

```cpp
struct prompt_cache_match {
    float   similarity;      // LCP ratio
    int64_t timestamp_us;    // created_us for freshness
};
```

### L1 -- server_slot

Each slot carries:

- `prompt.tokens` -- currently loaded prompt (for LCP matching)
- `task_tokens_original` -- prompt tokens captured at task start (for disk save)
- `t_last_used` -- last use timestamp (freshness + LRU fallback)
- `completed_successfully` -- gate for conditional save
- `callback_save_kv_cache_to_disk` -- L3 save function
- `callback_save_kv_cache_to_ram` -- L2 save function

---

## Pool Building & Sorting

### Candidate Collection (in `get_available_slot()`)

1. **L1**: iterate idle slots, compute `get_common_prefix / task.tokens.size()`, push if above `slot_prompt_similarity`
2. **L2**: `prompt_cache->find_all_matching(tokens, threshold)` -- linear scan of all cached states
3. **L3**: `kv_cache_disk_mgr->find_all_matching_entries(tokens, threshold)` -- Radix Tree pre-filter + per-candidate LCP

### Sort Comparator

```cpp
std::sort(candidates.begin(), candidates.end(),
    [](const cache_candidate & a, const cache_candidate & b) {
        if (std::abs(a.similarity - b.similarity) > 1e-6f)
            return a.similarity > b.similarity;       // 1. highest LCP first
        if (a.freshness != b.freshness)
            return a.freshness > b.freshness;          // 2. most recent first
        return a.tier < b.tier;                         // 3. L1 < L2 < L3
    });
```

### Selection Rules

| Best Tier | Action |
|---|---|
| **L1** (slot) | Use directly -- KV already in GPU |
| **L2** (RAM) | `slot.prompt_load(*prompt_cache, tokens)` -- restore from RAM to slot |
| **L3** (disk) | 1. `kv_cache_disk_mgr->restore_from_disk(filepath, slot.id, ctx)` 2. `slot.prompt_save(*prompt_cache) + prompt_cache->update()` -- promote to L2 |
| None | LRU fallback (oldest `t_last_used`) |

---

## Conditional Save Protocol

### The `completed_successfully` Gate

A boolean flag on `server_slot` controls all persistence:

- **Set `true`** on: normal generation stop, EOS token, `max_tokens` reached, embedding/rerank completion
- **Cleared `false`** on: `slot.reset()`, any interrupt, cancel, or error
- **Checked** in `slot.release()` before calling save callbacks

### Release Flow

```
slot.release()
  |
  +-- completed_successfully == true?
  |     YES: callback_save_kv_cache_to_disk()  -> L3
  |          callback_save_kv_cache_to_ram()    -> L2
  |     NO:  log "skipping cache save"
  |
  +-- reset()  (clears completed_successfully = false, tokens, etc.)
  +-- callback_on_release(id_slot)
```

### Save Callbacks

Installed per slot during `load_model()`:

**L3 (disk):**
```cpp
kv_cache_disk_mgr->save_to_disk(slot.id, slot.ctx_tgt, slot.ctx_dft,
                                slot.task_tokens_original.data(),
                                slot.task_tokens_original.size());
```

**L2 (RAM):**
```cpp
slot.prompt_save(*prompt_cache);
prompt_cache->update();
```

Saves happen **before** `reset()`, so `task_tokens_original` and KV state are still valid.

---

## Eviction

### L3 (Disk)

- **TTL**: `evict_expired_entries()` called before every search and save
- **LRU**: when `current_size_bytes_` exceeds `max_size_bytes_`, the entry with lowest `access_order` is evicted

### L2 (RAM)

- **Size limit**: managed by `server_prompt_cache` ring buffer with configurable size (`--cache-ram` MiB limit)

---

## CLI Parameters

| Flag | Type | Default | Tiers | Description |
|---|---|---|---|---|
| `--kv-cache-auto` | bool | false | L3 | Enable automatic KV cache disk management |
| `--slot-save-path <dir>` | string | "" | L3 | Cache directory (required for L3) |
| `--cache-ram <mib>` | int | 0 | L2 | RAM prompt cache size (MiB); 0 = disabled |
| `--slot-prompt-similarity` | float | varies | L1+L2+L3 | LCP threshold for all 3 tiers |
| `--max-cache-size <gb>` | float | 8.0 | L3 | Maximum disk cache size in GB (0 = unlimited) |
| `--cache-ttl <seconds>` | int64 | 3600 | L3 | TTL for disk entries (0 = no expiration) |

---

## File Format (L3)

Cache files use the `llama_state_seq_save_file` / `llama_state_seq_load_file` format:

- Header: magic (`LLAMA_STATE_SEQ_MAGIC`, 4 bytes) + version (4 bytes) + `n_token_count` (4 bytes)
- Token data: `n_token_count` int32 values
- KV cache data: variable length

---

## Cross-Model Safety (Known Gaps)

### Current Validation on Load

When restoring from disk (`llama_state_seq_load_file` / `llama_kv_cache::state_read`),
the following are validated and will cause load to fail gracefully (return 0):

| Check | Detection | Outcome |
|-------|-----------|---------|
| **Magic + version** (header) | `LLAMA_STATE_SEQ_MAGIC` / version field | Load rejected |
| **n_stream mismatch** | `n_stream_cur != n_stream` | `std::runtime_error` caught, load returns 0 |
| **Layer count mismatch** | `n_layer != layers.size()` | Load returns 0 |
| **K type mismatch** (per layer) | `k_type_i != k_type_i_ref` | Load returns 0 |
| **V type mismatch** (per layer) | `v_type_i != v_type_i_ref` | Load returns 0 |
| **V-transposition mismatch** | `v_trans != saved_v_trans` | Load returns 0 |
| **Row size mismatch** | key/value row size per layer | Load returns 0 |
| **Cell count overflow** | `cell_count > cells.size()` | Load returns 0 |
| **n_token_count > capacity** | token buffer overflow | Load returns 0 |

These checks catch most accidental cross-model reuse when the models differ
in architecture, layer count, or KV quantization type.

### Silent Corruption Risk

The following scenario is **NOT detected**:

- Two models with the **same architecture** (same layer count, same n_stream,
  same v_trans flag) and **same KV quantization type** but **different weights**
  (e.g., two fine-tuned versions of the same base model, or the same model
  with different GGUF quantization levels).

In this case:
1. The same prompt text produces the **same token IDs** (same tokenizer)
2. The trie search finds a **match** (tokens match)
3. `llama_state_seq_load_file` passes all structural checks (same layers,
   same K/V types, same n_stream)
4. The **raw K/V tensor data from the wrong model is loaded** into the
   context, producing garbage output or silent corruption

### Root Cause

The `state_seq_save_file` / `state_seq_load_file` path (used by the disk
cache) does **not** write or verify any model identity information. The
full-state path (`state_write_data` / `state_read_data`) does write the
model architecture string, but the sequence-level path used for per-
sequence disk cache skips this step.

Both paths have a TODO comment acknowledging this gap:
```
// TODO: add more model-specific info which should prevent loading the
//       session file if not identical
```

No model hash, fingerprint, or checksum is embedded in cache files.

### Mitigations (Current)

- **Different tokenizers** (most common cross-model case): same text
  produces different token IDs, so the trie won't match -> clean miss.
- **Different KV quant types** (e.g., `f16` vs `q8_0`): caught by
  per-layer type validation.
- **Different architectures** (different layer counts): caught by
  n_layer validation.
- **n_stream changes** (e.g., dual-stream vs single-stream): caught
  by n_stream mismatch check.

### Recommended Future Work

1. **Embed model fingerprint in cache files**: Write a hash of the model
   file (or the model's `llama_model_desc` + parameter count) into the
   sequence file header, and verify it before loading.
2. **Per-model subdirectories**: Isolate cache files by model hash in
   the cache directory to prevent accidental cross-contamination.
3. **File-level re-validation**: After a successful `llama_state_seq_load_file`,
   run a lightweight checksum on a portion of the restored KV data to
   verify it matches expected values for the current model.

---

## Thread Safety

- `kv_cache_disk_manager`: all public methods guarded by `std::recursive_mutex mutex_`
- `server_prompt_cache`: single-threaded access (server main loop)
- `server_slot`: owned by main loop, no concurrent access

---

## Security

1. Feature is **disabled by default** -- requires explicit `--kv-cache-auto`
2. Size limits prevent disk exhaustion with `--max-cache-size`
3. TTL enforcement automatically cleans up stale disk entries
4. `completed_successfully` gate prevents partial state persistence on interrupts
5. Orphaned files are reconciled on startup
6. All operations are synchronous and bounded

---

## Metrics

Logged periodically (every 300 seconds):

- `cache_hits` / `cache_misses`
- `saves_completed` / `saves_skipped`
- `restores_completed` / `total_restore_bytes`
- `evictions_ttl` / `evictions_lru`
