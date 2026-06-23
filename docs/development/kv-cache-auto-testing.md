# KV Cache Auto Feature — Testing Guide

## Test Infrastructure

Three layers of tests cover the KV cache auto-save/restore feature, from low-level API to cross-model smoke tests.

### C++ Unit Tests (`tests/test-kv-cache-disk.cpp`)

Low-level tests using the `testing.h` framework. Tests run with a real GGUF model.

| Test | Description |
|------|-------------|
| `lcp_computation` | LCP ratio calculation with identical, partial, and no-match sequences |
| `file_header_format` | File header contains correct magic, version, and token count |
| `save_restore` | Save and restore KV state in the same context |
| `restart_restore` | Save in context A, restore in a freshly initialized context B |
| `ttl_expiration` | File with backdated mtime is still readable (TTL logic is in disk manager) |
| `trie_operations` | Related prompts have LCP > 0; unrelated prompts have LCP ~ 0 |
| `multiple_entries_lcp` | LCP between multiple prompt pairs with varying overlap |
| `save_multiple_entries` | Save 3 files, verify each has a valid header |
| `restore_different_ctx` | Save with n_ctx=256, restore with n_ctx=512 |

Run:
```bash
cd build
ctest -R kv-cache-disk
```

### Python Integration Tests (`tools/server/tests/unit/test_kv_cache_disk.py`)

Integration tests using a real `llama-server` instance with `ServerPreset.tinyllama2`. Each test starts/stops the server independently.

| Test | Description |
|------|-------------|
| `test_kv_cache_initialization` | Server starts with KV cache auto enabled, disk manager initialized, cache directory exists |
| `test_first_request_save` | First request saves KV cache to disk (log + file verification) |
| `test_disk_lcp_wins_after_restart` | After restart (RAM empty), disk should win the LCP comparison |
| `test_ram_cache_preferred` | When RAM GPU cache has better LCP than disk, skip disk restore |
| `test_both_caches_empty_lru_fallback` | When both caches are empty, LRU slot is selected |
| `test_disk_miss_ram_partial_match` | Disk MISS, but RAM slot has partial match |
| `test_trie_rebuild_from_disk` | Restart server: trie rebuild from disk, cache HIT for same request |
| `test_ttl_eviction` | Cache entries expire after TTL seconds (3s TTL, verified with 4s wait) |
| `test_callback_invocation_and_save` | Callback is invoked and KV cache is saved on request completion |

Run:
```bash
cd tools/server/tests
pip install -r requirements.txt
pytest unit/test_kv_cache_disk.py -v
```

### Shell Smoke Tests

The shell smoke tests have been replaced by the Python integration tests above. All scenarios are covered in Python and C++.

---

## Quick Test

### 1. Start Server with KV Cache Auto

```bash
build/bin/llama-server \
    --model path/to/model.gguf \
    --port 8080 \
    --slot-save-path /tmp/kv-cache-test/ \
    --kv-cache-auto \
    --max-cache-size 1 \
    --cache-ttl 3600 \
    -lv 4
```

### 2. Verify Server Initialization

Check server logs for:
```
KV cache auto enabled: dir='/tmp/kv-cache-test/kv-meta', max=1.0 GB, ttl=3600 s, sim_threshold=...
KV cache disk manager initialized: dir='...', max_size=... GB, ttl=... s, entries=...
```

### 3. Make Test Requests

```bash
# First request — will save KV cache after generation
curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Tell me a joke"}]
    }'

# Wait for save callback (~3 seconds)
sleep 3

# Second request with same prompt — should restore from cache
curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Tell me a joke"}]
    }'
```

### 4. Check Logs for Cache Hit/Miss

```
KV cache HIT: LCP=... (threshold=...), file='slot_...'
KV cache MISS: no matching entry found
restored slot from disk cache (disk LCP=... > ram LCP=...)
RAM cache preferred (ram LCP=... >= disk LCP=..., skip disk restore)
```

### 5. Verify Cache Files

```bash
ls -la /tmp/kv-cache-test/kv-meta/
```

Expected: one or more `slot_*.bin` files.

---

## Advanced Testing

### Test Cache Eviction (TTL)

```bash
# Start with short TTL (60 seconds)
build/bin/llama-server --kv-cache-auto --cache-ttl 60 ...

# Make a request, wait for TTL to expire
sleep 65

# Make same request — should be cache miss (file expired)
curl http://localhost:8080/v1/chat/completions \
    -d '{"model":"test","messages":[{"role":"user","content":"test"}]}'
```

### Test Cache Eviction (LRU)

```bash
# Start with small cache size (0.5 GB)
build/bin/llama-server --kv-cache-auto --max-cache-size 0.5 ...

# Make multiple requests to fill cache
for i in {1..20}; do
    curl -s "http://localhost:8080/v1/chat/completions" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"request $i\"}]}" > /dev/null
done

# Check logs for eviction messages
grep "evict" /tmp/server.log
```

---

## Verification Checklist

- [ ] Server starts with `--kv-cache-auto` flag
- [ ] Cache directory is created automatically
- [ ] KV cache is saved after generation completes
- [ ] KV cache is restored when similar prompt detected
- [ ] Cache files have correct format (.bin extension)
- [ ] Metrics are logged every 5 minutes
- [ ] TTL-based eviction works correctly
- [ ] LRU-based eviction works when cache exceeds max size
- [ ] Orphaned file reconciliation works on startup
- [ ] C++ unit tests pass (`ctest -R kv-cache-disk`)
- [ ] Python integration tests pass (`pytest unit/test_kv_cache_disk.py`)
- [ ] Python integration tests pass (`pytest unit/test_kv_cache_disk.py`)

---

## Troubleshooting

### Issue: "KV cache auto enabled but no directory configured"

**Solution**: Ensure `--slot-save-path` is set.

### Issue: "Failed to restore KV state for slot X"

**Possible causes**:
- Cache file corrupted
- Model changed between saves
- llama.cpp version mismatch

**Solution**: Delete cache files and restart server.

### Issue: No cache hits despite same prompts

**Possible causes**:
- `--slot-prompt-similarity` threshold too high
- Streaming task still active (save callback skipped)
- TTL too short (entry expired before second request)

**Solution**: Check logs at `-lv 4` for disk/ram LCP comparison values.

---

## Performance Expectations

### With KV Cache Auto Enabled:
- First request with new prompt: ~normal latency
- Subsequent request with similar prompt: ~50-90% faster (depends on prompt overlap)

### Memory Usage:
- Cache files stored on disk, minimal RAM impact
- Max disk usage: `--max-cache-size` GB (default: 8.0 GB)

## Cleanup

```bash
# Remove all cache files
rm -rf /tmp/kv-cache-test/*
```
