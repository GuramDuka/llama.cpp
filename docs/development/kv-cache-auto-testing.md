# KV Cache Auto Feature - Testing Guide

## Overview
This document provides instructions for testing the automatic KV cache save/restore feature.

## Prerequisites
1. llama.cpp built with KV cache auto support (commit f4c963c89 or later)
2. A small model file (e.g., SmolLM-135M or LFM2.5-350M in GGUF format)
3. Linux environment with curl installed

## Quick Test

### 1. Start Server with KV Cache Auto
```bash
build/bin/llama-server \
    --model path/to/model.gguf \
    --port 8080 \
    --kv-cache-auto \
    --max-cache-size 1 \
    --cache-ttl 3600 \
    --kv-cache-dir /tmp/kv-cache-test
```

### 2. Verify Server Initialization
Check server logs for:
```
KV cache auto enabled: dir='/path/to/cache', max=1.0 GB, ttl=3600s, sim_threshold=0.75
```

### 3. Make Test Requests
```bash
# First request - will save KV cache after generation
curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Tell me a joke"}]
    }'

# Second request with similar prompt - should restore from cache
curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Tell me another joke"}]
    }'
```

### 4. Check Logs for Cache Hit/Miss
Enable verbose logging with `-lv 4` to see:
```
KV cache HIT: LCP=0.850 (threshold=0.750), file='slot_1234567890.state'
KV cache restored: slot=0, file='/path/to/cache/slot_1234567890.state', bytes=123456
```

### 5. Verify Cache Files
Check that cache files are created:
```bash
ls -la /tmp/kv-cache-test/
```

Expected output:
```
-rw-r--r-- 1 user user 123K Jun 17 15:45 slot_1234567890.state
```

### 6. Check Metrics Logging
Every 5 minutes, the server logs KV cache statistics:
```
KV cache stats: hits=1 misses=1 saves=1 restores=1 evictions_ttl=0 evictions_lru=0
```

## Advanced Testing

### Test Cache Eviction (TTL)
```bash
# Start with short TTL (60 seconds)
build/bin/llama-server --kv-cache-auto --cache-ttl 60 ...

# Wait for TTL to expire
sleep 120

# Make request - should be cache miss (file expired)
curl http://localhost:8080/v1/chat/completions -d '{"model":"test","messages":[{"role":"user","content":"test"}]}'
```

### Test Cache Eviction (LRU)
```bash
# Start with small cache size (0.5 GB)
build/bin/llama-server --kv-cache-auto --max-cache-size 0.5 ...

# Make multiple requests to fill cache
for i in {1..20}; do
    curl http://localhost:8080/v1/chat/completions -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"request $i\"}]}" > /dev/null
done

# Check logs for eviction messages
grep "evict" /tmp/server.log
```

### Test Cache Directory Migration
```bash
# Move cache directory while server is running
mv /tmp/kv-cache-test /tmp/kv-cache-test-old

# Server should handle missing directory gracefully
curl http://localhost:8080/v1/chat/completions -d '{"model":"test","messages":[{"role":"user","content":"test"}]}'
```

## Verification Checklist

- [ ] Server starts with `--kv-cache-auto` flag
- [ ] Cache directory is created automatically
- [ ] KV cache is saved after generation completes
- [ ] KV cache is restored when similar prompt detected
- [ ] Cache files have correct format (.state extension)
- [ ] Metrics are logged every 5 minutes
- [ ] TTL-based eviction works correctly
- [ ] LRU-based eviction works when cache exceeds max size
- [ ] Orphaned file reconciliation works on startup
- [ ] No memory leaks or crashes during extended operation

## Troubleshooting

### Issue: "KV cache auto enabled but no directory configured"
**Solution**: Ensure `--slot-save-path` is set, as it's used to determine the default cache directory.

### Issue: "Failed to restore KV state for slot X"
**Possible causes**:
- Cache file corrupted
- Model changed between saves
- llama.cpp version mismatch

**Solution**: Delete cache files and restart server.

### Issue: No cache hits despite similar prompts
**Possible causes**:
- `--slot-prompt-similarity` threshold too high
- Prompts not similar enough for LCP matching

**Solution**: Lower `--slot-prompt-similarity` value (default: 0.75).

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

# Or use server endpoint (if implemented)
curl http://localhost:8080/v1/cache/purge -X POST
```
