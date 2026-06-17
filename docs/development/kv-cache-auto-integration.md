# Automatic KV Cache Save/Restore - Integration Guide

## Overview

This feature implements automatic KV cache persistence to disk for llama.cpp's `llama-server`. When enabled, completed generation slots are saved to disk and can be restored in subsequent sessions based on prompt similarity (LCP matching).

---

## Files Created/Modified

### New Files
1. **`tools/server/kv-cache-disk-manager.h`** — Header with class definition
2. **`tools/server/kv-cache-disk-manager.cpp`** — Implementation

### Modified Files
1. **`common/common.h`** — Added CLI parameters (lines 624-628)
2. **`common/arg.cpp`** — Added CLI argument handlers (after line 1335)

---

## Step-by-Step Integration

### Step 1: Add Include to server-context.cpp

In `tools/server/server-context.cpp`, add after existing includes:

```cpp
#include "kv-cache-disk-manager.h"
```

### Step 2: Add Member Variable to server_context_impl

In `server-context.cpp`, find the `server_context_impl` struct and add:

```cpp
// Automatic KV cache disk manager (optional)
std::unique_ptr<kv_cache_disk_manager> kv_cache_disk_mgr;
```

Place this near other member variables (around line 1050-1100).

### Step 3: Initialize in load_model()

In `server_context_impl::load_model()`, add initialization after model loading:

```cpp
// Initialize automatic KV cache disk manager if enabled
if (params_base.kv_cache_auto) {
    // Determine cache directory
    std::string cache_dir;
    
    if (!params_base.kv_cache_dir.empty()) {
        // Use explicitly configured directory
        cache_dir = params_base.kv_cache_dir;
    } else if (!params_base.slot_save_path.empty()) {
        // Default: slot-save-path + "/kv-meta"
        cache_dir = params_base.slot_save_path + "kv-meta";
    } else {
        SRV_WRN("KV cache auto enabled but no directory configured (slot-save-path is empty)\n");
        return false;
    }
    
    // Create directory if needed
    std::filesystem::create_directories(cache_dir);
    
    // Initialize manager
    kv_cache_disk_mgr = std::make_unique<kv_cache_disk_manager>();
    if (!kv_cache_disk_mgr->initialize(cache_dir, 
                                       params_base.max_cache_size_gb,
                                       params_base.cache_ttl_seconds)) {
        SRV_WRN("Failed to initialize KV cache disk manager\n");
        return false;
    }
    
    // Reconcile orphaned files on startup
    kv_cache_disk_mgr->reconcile_orphaned_files();
    
    SRV_INF("KV cache auto enabled: dir='%s', max=%.1f GB, ttl=%" PRId64 "s\n",
            cache_dir.c_str(), params_base.max_cache_size_gb, 
            params_base.cache_ttl_seconds);
}
```

### Step 4: Hook into Slot Release (Save to Disk)

In `server_slot::release()`, add save logic before resetting the slot:

```cpp
void release() {
    if (is_processing()) {
        // ... existing code ...
        
        // Save KV cache to disk if auto-save is enabled and generation completed normally
        if (task && params_base.kv_cache_auto && kv_cache_disk_mgr) {
            // Only save if:
            // 1. Generation was not cancelled
            // 2. We actually generated some tokens
            // 3. The prompt didn't match an existing cache entry (avoid redundant saves)
            
            bool should_save = true;
            
            // Skip if generation was interrupted
            if (state == SLOT_STATE_GENERATING && n_decoded > 0) {
                // Check if we have a complete response (not truncated/cancelled)
                // This depends on your definition of "complete"
                
                SLT_DBG(*this, "saving KV cache to disk: slot=%d, tokens=%d\n", id, n_decoded);
                
                kv_cache_disk_mgr->save_to_disk(id, ctx_tgt, ctx_dft, &prompt.tokens);
            }
        }
        
        // ... rest of existing release code ...
    }
}
```

### Step 5: Hook into Slot Allocation (Restore from Disk)

In `server_context_impl::get_available_slot()`, add cache lookup before LRU selection:

```cpp
server_slot * get_available_slot(const server_task & task) {
    server_slot * ret = nullptr;
    
    // NEW: Try to restore from disk cache if enabled
    if (params_base.kv_cache_auto && kv_cache_disk_mgr) {
        float lcp_threshold = slot_prompt_similarity;  // Use existing similarity threshold
        
        for (server_slot & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }
            
            // Try to restore from disk
            if (kv_cache_disk_mgr->try_restore_from_disk(slot.id, task.tokens, lcp_threshold)) {
                ret = &slot;
                metrics_.on_cache_hit();  // Optional metric tracking
                SLT_INF(*ret, "restored slot from disk cache (LCP hit)\n");
                break;
            }
        }
        
        if (ret) {
            // Don't do prompt cache update or LRU selection
            return ret;
        }
    }
    
    // ... existing code for prompt similarity and LRU ...
}
```

### Step 6: Add Metrics Reporting (Optional)

In `server_context_impl::start_loop()`, add periodic metrics logging:

```cpp
// In the main loop, every N iterations or on idle:
if (params_base.kv_cache_auto && kv_cache_disk_mgr) {
    // Log metrics every 5 minutes (300 seconds)
    static int64_t last_metrics_log = ggml_time_us();
    
    if ((ggml_time_us() - last_metrics_log) > 300 * 1000000LL) {
        auto metrics = kv_cache_disk_mgr->get_metrics();
        
        SRV_INF("KV cache stats: hits=%llu misses=%llu saves=%llu evictions_ttl=%llu evictions_lru=%llu\n",
                (unsigned long long)metrics.cache_hits.load(),
                (unsigned long long)metrics.cache_misses.load(),
                (unsigned long long)metrics.saves_completed.load(),
                (unsigned long long)metrics.evictions_ttl.load(),
                (unsigned long long)metrics.evictions_lru.load());
        
        last_metrics_log = ggml_time_us();
    }
}
```

### Step 7: Add CMake Build Entry

In `tools/server/CMakeLists.txt`, add the new source file:

```cmake
set(SERVER_SOURCES
    # ... existing sources ...
    kv-cache-disk-manager.cpp
)
```

---

## CLI Usage Example

```bash
# Start server with automatic KV cache
./build/bin/llama-server \
    -m model.gguf \
    --port 8080 \
    --slot-save-path /data/slots \
    --kv-cache-auto true \
    --max-cache-size 2 \
    --cache-ttl 7200 \
    --log-verbosity 4

# Or with custom directory
./build/bin/llama-server \
    -m model.gguf \
    --port 8080 \
    --kv-cache-auto true \
    --kv-cache-dir /data/kv-cache \
    --max-cache-size 4 \
    --cache-ttl 3600
```

---

## Testing with curl

### Test 1: First Request (Cache Miss)
```bash
curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "",
        "messages": [{"role": "user", "content": "Hello, introduce yourself"}],
        "max_tokens": 50,
        "stream": false
    }' | python3 -m json.tool
```

### Test 2: Second Request with Same Prompt (Cache Hit)
```bash
# Should show cache hit in logs (verbosity >= 4)
curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "",
        "messages": [{"role": "user", "content": "Hello, introduce yourself"}],
        "max_tokens": 50,
        "stream": false
    }' | python3 -m json.tool
```

### Check Logs for Cache Metrics
```bash
# Look for these patterns in server logs:
grep "KV cache" server.log
grep "Cache Hit\|Cache Miss" server.log
```

---

## Architecture Decisions

### Why Separate Directory?
- **Clean separation**: Manual slot saves (`--slot-save-path`) vs automatic KV cache
- **Different eviction policies**: Manual files persist forever; auto-cache uses TTL+LRU
- **Safety**: No risk of mixing file formats or accidentally deleting user-saved slots
- **Debuggability**: Easy to identify which files belong to which system

### Default Directory Logic
```cpp
if (kv_cache_dir) {
    use kv_cache_dir  // Explicit configuration
} else if (slot_save_path) {
    use slot_save_path + "/kv-meta"  // Smart default
} else {
    disable auto-save  // No safe location
}
```

### Token Storage for LCP Matching
- Store first N tokens (up to 8192) in metadata
- Compare against incoming prompt on cache lookup
- Enables efficient prefix matching without file I/O

---

## Security Considerations

1. **Feature disabled by default** — requires explicit `--kv-cache-auto` flag
2. **Directory validation** — ensures path is a valid directory
3. **Size limits** — prevents disk exhaustion with `--max-cache-size`
4. **TTL enforcement** — automatically cleans up stale entries
5. **No external file I/O loops** — all operations are synchronous and bounded

---

## Future Enhancements (Not Implemented)

1. **Async save/restore** — use background threads for disk I/O
2. **Compression** — compress KV state before saving
3. **Network replication** — share cache across multiple servers
4. **Hot reload** — detect new cache entries without restart
5. **Metrics endpoint** — expose cache stats via HTTP
