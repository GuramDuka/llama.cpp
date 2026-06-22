# KV Cache Auto-Save/Restore — Design Specification

## Overview

`llama-server` can persist KV cache state to disk so that subsequent requests with the same or similar prompts can restore the cache instead of re-decoding the prompt. The feature is disabled by default and gated behind the `--kv-cache-auto` flag.

## Architecture

```
  Request arrives
       |
       v
  ┌──────────────────────┐
  │  find_cache_entry()  │  Radix Tree (Trie) pre-filter, LCP similarity filter
  │  (disk cache lookup) │
  └────────┬─────────────┘
           │  cache_file (path, may be empty)
           v
  ┌──────────────────────┐
  │  Compare disk LCP    │  vs best RAM LCP across free slots
  │  vs RAM LCP          │
  └────────┬─────────────┘
           │
    disk LCP > RAM LCP ?
           │
       +---+---+
      yes     no
       |       |
       v       v
  restore_from_disk()  Continue to RAM-level LCP / LRU selection
       |
       v
  Return restored slot
```

## Data Structures

### kv_cache_trie (Radix Tree)

Each node maps a token ID to child nodes. Entry indices (pointing into the LRU ring) are stored at the deepest matching node, enabling fast prefix matching.

- **Insert**: O(m log k) — m = token sequence length, k = vocabulary size
- **Search**: traverses the trie, collects entry indices from the deepest matching node
- **Remove**: walks all nodes and removes stale entry indices

### kv_cache_disk_manager

The central class. Owns:

- `lru_ring_` — ring buffer of `ring_buffer_entry` (metadata + access order)
- `filepath_index_` — O(1) map from filepath to ring index
- `trie_` — Radix Tree for prefix matching
- `metrics_` — `kv_cache_metrics` counters (hits, misses, saves, evictions)

### disk_cache_entry

Per-entry metadata stored in memory:

```cpp
struct disk_cache_entry {
    std::string          filepath;          // Path to serialized KV state file
    int64_t              created_at_us;     // Creation timestamp (microseconds)
    int64_t              last_accessed_us;  // Last access timestamp
    size_t               file_size_bytes;   // Size of the cached data
    uint32_t             seq_id;            // Sequence ID associated with this entry
    std::vector<int32_t> tokens;            // Token sequence for LCP matching (first N tokens)
};
```

## Public API

```cpp
class kv_cache_disk_manager {
  public:
    bool initialize(const std::string & cache_dir, float max_size_gb, int64_t ttl_seconds);
    std::string find_cache_entry(const llama_tokens & tokens, float lcp_threshold);
    bool restore_from_disk(const std::string & filepath, int32_t slot_id, llama_context * ctx_tgt);
    bool save_to_disk(int32_t slot_id, llama_context * ctx_tgt, llama_context * ctx_dft,
                      const int32_t * tokens, size_t token_count);
    float get_disk_lcp(const std::string & filepath, const std::vector<int32_t> & tokens) const;
    void  set_prompt_similarity_threshold(float threshold);
    float get_prompt_similarity_threshold() const;
    kv_cache_metrics get_metrics() const;
    void  reset_metrics();
    void  reconcile_orphaned_files();
    void  purge_all_cache_files();
    float calculate_lcp_ratio(const std::vector<int32_t> & a, const std::vector<int32_t> & b) const;
    float calculate_lcp_ratio(const server_tokens & a, const std::vector<int32_t> & b) const;
};
```

## Slot Allocation Logic (server-context.cpp)

In `get_available_slot()`:

1. If `--kv-cache-auto` and disk manager is initialized:
   - Single `find_cache_entry()` search (not per-slot)
   - Compute `disk_lcp` from the returned filepath
   - Iterate free slots to find `best_ram_lcp` (common prefix with GPU KV cache)
   - If `disk_lcp > best_ram_lcp` → `restore_from_disk()` to the first free slot
   - If `best_ram_lcp > disk_lcp` → log "RAM cache preferred", fall through
2. Prompt-similarity matching (RAM only)
3. LRU fallback

## Save Callback

A callback (`slot.callback_save_kv_cache_to_disk`) is installed per slot. On invocation:

1. Guard: skip if streaming task still active
2. Prefer `slot.kv_cache_original_tokens` (captured at task start); fall back to `slot.prompt.tokens`; then save with `nullptr/0`
3. Call `save_to_disk()` which handles eviction, writes the file via `llama_state_seq_save_file()`, and inserts the entry into the LRU ring + trie

## Eviction

- **TTL**: `evict_expired_entries()` is called before every search and save
- **LRU**: when `current_size_bytes_` exceeds `max_size_bytes_`, the LRU entry (lowest `access_order`) is evicted

## CLI Parameters

| Flag                         | Type     | Default | Description                               |
| ---------------------------- | -------- | ------- | ----------------------------------------- |
| `--kv-cache-auto`            | bool     | false   | Enable automatic KV cache disk management |
| `--max-cache-size <gb>`      | float    | 8.0     | Maximum total cache size in GB            |
| `--cache-ttl <seconds>`      | int64_t  | 3600    | TTL for cache entries (0 = disabled)      |
| `--kv-cache-dir <path>`      | string   | (auto)  | Directory for cache files                 |

When `--kv-cache-dir` is not set, the default is `--slot-save-path + "kv-meta"`.

## File Format

Cache files use the `llama_state_seq_save_file` / `llama_state_seq_load_file` format:

- Header: magic (4 bytes, `LLAMA_STATE_SEQ_MAGIC`) + version (4 bytes) + n_token_count (4 bytes)
- Token data: `n_token_count` int32 values
- KV cache data: variable length

## Thread Safety

All public methods are guarded by `std::recursive_mutex mutex_`.

## Security

1. Feature is **disabled by default** — requires explicit `--kv-cache-auto`
2. Size limits prevent disk exhaustion with `--max-cache-size`
3. TTL enforcement automatically cleans up stale entries
4. Orphaned files are reconciled on startup
5. All operations are synchronous and bounded

## Metrics

Logged periodically (every 300 seconds) and available via `get_metrics()`:

- `cache_hits` / `cache_misses`
- `saves_completed` / `saves_skipped`
- `restores_completed` / `total_restore_bytes`
- `evictions_ttl` / `evictions_lru`
