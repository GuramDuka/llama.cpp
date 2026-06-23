# Automatic KV Cache Save/Restore — Integration Guide

## Overview

KV cache state is persisted to disk so that subsequent requests with the same or similar prompts can restore the cache instead of re-decoding the prompt. The feature is disabled by default and gated behind `--kv-cache-auto`.

Disk cache matches are compared against RAM (GPU) cache matches. A disk restore is performed only when disk LCP strictly exceeds RAM LCP.

---

## File Layout

```
tools/server/kv-cache-disk-manager.h       — Header: trie, ring buffer, metrics, public API
tools/server/kv-cache-disk-manager.cpp     — Implementation
tools/server/server-context.cpp            — Integration (see below)
common/common.h                            — CLI parameters (lines 625-628)
common/arg.cpp                             — CLI argument handlers (lines 1321-1360)
```

---

## Integration Points

### 1. Include and Member

```cpp
// tools/server/server-context.cpp, line 6
#include "kv-cache-disk-manager.h"

// server_context_impl struct, line 820
std::unique_ptr<kv_cache_disk_manager> kv_cache_disk_mgr;
```

### 2. Initialization in load_model()

Location: `server_context_impl::load_model()`, lines 1288-1355.

The cache directory is `--slot-save-path`. If it is empty, the feature is disabled with a warning.

After initialization, the prompt similarity threshold is set from `--slot-prompt-similarity` and a save callback is installed per slot.

### 3. Save Callback (per slot)

The callback `slot.callback_save_kv_cache_to_disk` is installed for each slot during initialization. On invocation it:

1. Guards against streaming tasks (`slot.task && slot.task->params.stream`)
2. Prefers `slot.kv_cache_original_tokens` (captured at task start)
3. Falls back to `slot.prompt.tokens`
4. Falls back to `nullptr/0` if no tokens available
5. Calls `kv_cache_disk_mgr->save_to_disk()`

The save path handles eviction internally: TTL eviction first, then LRU eviction if the cache exceeds `--max-cache-size`.

### 4. Slot Allocation in get_available_slot()

Location: `server_context_impl::get_available_slot()`, lines 1454-1507.

```
┌──────────────────────────────┐
│  find_cache_entry(tokens)    │  Single trie search (not per-slot)
└─────────────┬────────────────┘
              │
              v
┌──────────────────────────────┐
│  get_disk_lcp(filepath)      │  Compute LCP of disk match
│  vs                          │  against incoming tokens
│  best_ram_lcp (all slots)    │  Common prefix / prompt length
└─────────────┬────────────────┘
              │
         disk_lcp > best_ram_lcp ?
              │
          +---+---+
         yes      no
          |       |
          v       v
  restore_from_disk()  "RAM cache preferred" (fall through
    to first free slot    to prompt-similarity / LRU)
```

### 5. Metrics Logging

Location: main loop, lines 3814-3820. Metrics are logged every 300 seconds when the feature is enabled.

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

# Check parameters in common.h
#   kv_cache_auto        = false   (disabled by default)
#   max_cache_size_gb    = 8.0f
#   cache_ttl_seconds    = 3600
```

---

## Testing with curl

### First Request (Cache Miss)

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

### Second Request with Same Prompt (Cache Hit)

```bash
# Should log: "KV cache HIT" and "restored slot from disk cache"
curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "",
        "messages": [{"role": "user", "content": "Hello, introduce yourself"}],
        "max_tokens": 50,
        "stream": false
    }' | python3 -m json.tool
```

### Check Logs

```bash
grep "KV cache" server.log
grep "KV cache HIT\|KV cache MISS\|restored slot from disk" server.log
```

---

## Architecture Decisions

### Why Separate Directory?

- **Clean separation**: Manual slot saves (`--slot-save-path`) vs automatic KV cache
- **Different eviction policies**: Manual files persist forever; auto-cache uses TTL+LRU
- **Safety**: No risk of mixing file formats or accidentally deleting user-saved slots

### Default Directory Logic

```
if (--slot-save-path is set)
    use --slot-save-path
else
    disable auto-save (warning logged)
```

### Disk vs RAM Comparison

Disk restore is performed **only** when `disk_lcp > best_ram_lcp`. When RAM and disk have equal LCP, the RAM cache is preferred (no disk restore needed). When RAM LCP exceeds disk LCP, a log message is emitted and slot selection falls through to prompt-similarity and LRU logic.

### Token Storage for LCP Matching

- First N tokens (up to 8192, defined by `KV_CACHE_MAX_TOKENS`) are stored in metadata
- LCP is computed as: `common_prefix_length / incoming_prompt_length`
- The trie pre-filters candidates before full LCP computation

---

## Security Considerations

1. **Feature disabled by default** — requires explicit `--kv-cache-auto` flag
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
