# Automatic KV Cache Save/Restore — Implementation Reference

## Summary

The feature is implemented and integrated into `llama-server`. KV cache state is persisted to disk using a Radix Tree (Trie) for O(m log k) prefix matching. Slot allocation compares disk LCP vs RAM LCP and restores from disk only when the disk match is strictly better.

---

## Files

### Core implementation
- `tools/server/kv-cache-disk-manager.h` — Header: trie, ring buffer, metrics, public API
- `tools/server/kv-cache-disk-manager.cpp` — Implementation: trie operations, save/restore, eviction, LCP

### Integration
- `tools/server/server-context.cpp` — Integration into `server_context_impl`
  - `#include "kv-cache-disk-manager.h"` (line 6)
  - `std::unique_ptr<kv_cache_disk_manager> kv_cache_disk_mgr` member (line 820)
  - Initialization in `load_model()` (lines 1288-1355)
  - Save callback installed per slot (lines 1313-1348)
  - Slot allocation in `get_available_slot()` (lines 1454-1507)
  - Metrics logging in main loop (lines 3814-3820)

### CLI parameters
- `common/common.h` — `kv_cache_auto`, `max_cache_size_gb`, `cache_ttl_seconds` (lines 625-627)
- `common/arg.cpp` — Argument handlers for each parameter (lines 1321-1348)

### Tests
- `tests/test-kv-cache-disk.cpp` — C++ unit tests (13 tests)
- `tools/server/tests/unit/test_kv_cache_disk.py` — Python integration tests (10 tests)

---

## Radix Tree Implementation

### Architecture

```
kv_cache_trie (root)
  └── kv_cache_trie_node (token_id=1)
       └── kv_cache_trie_node (token_id=2)
            └── kv_cache_trie_node (token_id=3, entry_indices=[0, 5])
```

- **O(m log k)** prefix matching vs O(n x m) linear search
- Memory efficient — shared prefixes don't duplicate nodes
- Fast pre-filtering before expensive LCP similarity calculation

### Performance Comparison

| Cache Size  | Linear Search | Radix Tree | Speedup |
|-------------|---------------|------------|---------|
| 100 entries | 5,000 ops     | ~50 ops    | **100x** |
| 1,000 entries | 100,000 ops | ~100 ops   | **1,000x** |
| 10,000 entries | 2,000,000 ops | ~200 ops | **10,000x** |

---

## Integration Flow

### 1. Initialization (server-context.cpp, load_model())

```cpp
// lines 1288-1355
if (params_base.kv_cache_auto) {
    if (params_base.slot_save_path.empty()) {
        SRV_WRN("KV cache auto enabled but slot-save-path is empty\n");
    } else {
        std::string cache_dir = params_base.slot_save_path;
        std::filesystem::create_directories(cache_dir);

        kv_cache_disk_mgr = std::make_unique<kv_cache_disk_manager>();
        if (!kv_cache_disk_mgr->initialize(cache_dir, params_base.max_cache_size_gb,
                                            params_base.cache_ttl_seconds)) {
            SRV_WRN("Failed to initialize KV cache disk manager\n");
        } else {
            kv_cache_disk_mgr->set_prompt_similarity_threshold(params_base.slot_prompt_similarity);

            // Install save callback for each slot
            for (int i = 0; i < params_base.n_parallel; i++) {
                server_slot & slot = slots[i];
                slot.callback_save_kv_cache_to_disk = [this, &slot]() {
                    if (!kv_cache_disk_mgr || !slot.ctx_tgt) return;
                    if (slot.task && slot.task->params.stream) return;

                    if (!slot.kv_cache_original_tokens.empty()) {
                        kv_cache_disk_mgr->save_to_disk(slot.id, slot.ctx_tgt, slot.ctx_dft,
                                                        slot.kv_cache_original_tokens.data(),
                                                        slot.kv_cache_original_tokens.size());
                    } else if (slot.prompt.tokens.size() > 0) {
                        kv_cache_disk_mgr->save_to_disk(slot.id, slot.ctx_tgt, slot.ctx_dft,
                                                        slot.prompt.tokens.get_tokens().data(),
                                                        slot.prompt.tokens.get_tokens().size());
                    } else {
                        kv_cache_disk_mgr->save_to_disk(slot.id, slot.ctx_tgt, slot.ctx_dft, nullptr, 0);
                    }
                };
            }

            LOG_INF("KV cache auto enabled: dir='%s', max=%.1f GB, ttl=%ld s, sim_threshold=%.3f\n",
                    cache_dir.c_str(), params_base.max_cache_size_gb,
                    params_base.cache_ttl_seconds, params_base.slot_prompt_similarity);
        }
    }
}
```

### 2. Slot Allocation (server-context.cpp, get_available_slot())

```cpp
// lines 1454-1507
if (params_base.kv_cache_auto && kv_cache_disk_mgr) {
    float lcp_threshold = slot_prompt_similarity;

    // Single disk cache search (not per-slot)
    std::string cache_file = kv_cache_disk_mgr->find_cache_entry(task.tokens.get_tokens(), lcp_threshold);

    // Compute disk LCP if a match was found
    float disk_lcp = 0.0f;
    if (!cache_file.empty()) {
        disk_lcp = kv_cache_disk_mgr->get_disk_lcp(cache_file, task.tokens.get_tokens());
    }

    // Find best RAM LCP among free slots
    float best_ram_lcp = 0.0f;
    for (server_slot & slot : slots) {
        if (slot.is_processing()) continue;
        if (!slot.prompt.tokens.empty()) {
            float ram_lcp = float(slot.prompt.tokens.get_common_prefix(task.tokens)) / task.tokens.size();
            if (ram_lcp > best_ram_lcp) best_ram_lcp = ram_lcp;
        }
    }

    // Disk wins only if strictly greater than RAM
    if (disk_lcp > best_ram_lcp && !cache_file.empty()) {
        for (server_slot & slot : slots) {
            if (slot.is_processing()) continue;
            if (kv_cache_disk_mgr->restore_from_disk(cache_file, slot.id, ctx_tgt)) {
                ret = &slot;
                SLT_INF(*ret, "restored slot from disk cache (disk LCP=%.3f > ram LCP=%.3f)\n",
                        disk_lcp, best_ram_lcp);
                break;
            }
        }
        if (ret) return ret;
    } else if (best_ram_lcp > disk_lcp) {
        LOG_INF("RAM cache preferred (ram LCP=%.3f >= disk LCP=%.3f, skip disk restore)\n",
                best_ram_lcp, disk_lcp);
    }
}
```

### 3. Metrics Logging (server-context.cpp, main loop)

```cpp
// lines 3814-3820
if (params_base.kv_cache_auto && kv_cache_disk_mgr) {
    // Logged every 300 seconds
    auto metrics = kv_cache_disk_mgr->get_metrics();
    // ... log hits, misses, saves, evictions ...
}
```

---

## CLI Usage

```bash
# Enable KV cache auto (requires --slot-save-path)
./build/bin/llama-server \
    -m model.gguf \
    --port 8080 \
    --slot-save-path /data/slots \
    --kv-cache-auto \
    --max-cache-size 2 \
    --cache-ttl 7200
```

---

## Security & Constraints

1. **Disabled by default** — requires explicit `--kv-cache-auto` flag
2. **Directory validation** — ensures path is a valid directory
3. **Size limits** — prevents disk exhaustion with `--max-cache-size`
4. **TTL enforcement** — automatically cleans up stale entries
5. **Orphaned file reconciliation** — removed on startup
6. **No external file I/O loops** — all operations are synchronous and bounded

---

## Future Enhancements (Not Implemented)

1. Async save/restore with background threads
2. Compression for KV state before saving
3. Network replication across servers
4. Hot reload to detect new cache entries
5. HTTP metrics endpoint for cache stats
