#!/bin/bash
# Test script for KV cache auto feature - tests models sequentially
# Downloads models from HuggingFace if not present in models/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
MODEL_DIR="$PROJECT_DIR/models"
PORT_BASE=8100

# Model names and download URLs
declare -a MODELS=(
    "SmolLM-135M-Instruct.i1-Q4_K_M.gguf"
    "LFM2.5-350M.i1-Q4_K_M.gguf"
    "Qwen3.5-0.8B.i1-Q4_K_M.gguf"
)

declare -a DOWNLOAD_URLS=(
    "https://huggingface.co/mradermacher/SmolLM-135M-Instruct-i1-GGUF/resolve/main/SmolLM-135M-Instruct.i1-Q4_K_M.gguf"
    "https://huggingface.co/mradermacher/LFM2.5-350M-i1-GGUF/resolve/main/LFM2.5-350M.i1-Q4_K_M.gguf"
    "https://huggingface.co/mradermacher/Qwen3.5-0.8B-i1-GGUF/resolve/main/Qwen3.5-0.8B.i1-Q4_K_M.gguf"
)

PASS_COUNT=0
FAIL_COUNT=0

run_check() {
    local description="$1"
    local result="$2"
    if [ "$result" = "true" ]; then
        echo "  ✓ $description"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  ✗ $description"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

cleanup_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

download_model() {
    local model_file="$1"
    local download_url="$2"
    local model_path="$3"

    if [ -f "$model_path" ]; then
        echo "  Model already exists: $model_file"
        return 0
    fi

    mkdir -p "$MODEL_DIR"
    echo "  Downloading $model_file..."
    if command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$model_path" "$download_url"
    elif command -v wget &> /dev/null; then
        wget --show-progress -O "$model_path" "$download_url"
    else
        echo "  ERROR: Neither curl nor wget found"
        return 1
    fi

    if [ -f "$model_path" ]; then
        echo "  Download complete: $(du -h "$model_path" | cut -f1)"
    else
        echo "  ERROR: Download failed for $model_file"
        return 1
    fi
}

for i in "${!MODELS[@]}"; do
    MODEL_FILE="${MODELS[$i]}"
    DOWNLOAD_URL="${DOWNLOAD_URLS[$i]}"
    MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

    echo ""
    echo "Checking model: $MODEL_FILE"
    download_model "$MODEL_FILE" "$DOWNLOAD_URL" "$MODEL_PATH" || exit 1
    
    PORT=$((PORT_BASE + RANDOM % 100))
    CACHE_DIR="/tmp/kv-cache-test-$(echo $MODEL_FILE | sed 's/\.gguf//g')"
    SERVER_LOG="/tmp/server-$MODEL_FILE.log"
    
    echo ""
    echo "=============================================="
    echo "Testing model: $MODEL_FILE"
    echo "Port: $PORT, Cache dir: $CACHE_DIR"
    echo "=============================================="
    
    # Create cache directory before starting server
    mkdir -p "$CACHE_DIR"
    rm -f "$SERVER_LOG"
    
    # Start server (without keep-alive - it stops when idle)
    echo ""
    echo "Starting server..."
    "$BUILD_DIR/bin/llama-server" \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --kv-cache-auto \
        --max-cache-size 1 \
        --cache-ttl 3600 \
        --kv-cache-dir "$CACHE_DIR" \
        -lv 4 \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 15
    
    # Check if server is running
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server failed to start!"
        tail -30 "$SERVER_LOG"
        cleanup_server "$SERVER_PID"
        exit 1
    fi
    
    echo "✓ Server started (PID: $SERVER_PID)"
    
    # Test 1: Check KV cache initialization in logs
    echo ""
    echo "--- Test 1: KV Cache Initialization ---"
    grep -q "KV cache auto enabled" "$SERVER_LOG" && run_check "KV cache auto enabled message found" "true" || run_check "KV cache auto enabled message found" "false"
    grep -q "KV cache disk manager initialized" "$SERVER_LOG" && run_check "KV cache disk manager initialized" "true" || run_check "KV cache disk manager initialized" "false"
    
    # Test 2: Check cache directory was created
    echo ""
    echo "--- Test 2: Cache Directory ---"
    [ -d "$CACHE_DIR" ] && run_check "Cache directory exists" "true" || run_check "Cache directory exists" "false"
    
    # Test 3: Make first test request (will save KV cache)
    echo ""
    echo "--- Test 3: First Request (Save KV Cache) ---"
    echo "Making first test request..."
    RESPONSE1=$(curl -s --max-time 60 http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "test",
            "messages": [{"role": "user", "content": "Tell me a short joke"}]
        }' 2>/dev/null || echo "")
    
    if [ -n "$RESPONSE1" ]; then
        run_check "First request response received" "true"
        TOKEN_COUNT=$(echo "$RESPONSE1" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*$' || echo "unknown")
        echo "  Response tokens: $TOKEN_COUNT"
    else
        run_check "First request response received" "false"
    fi
    
    # Wait for request to complete and callback to execute
    sleep 5
    
    # Test 4: Check if KV cache was saved
    echo ""
    echo "--- Test 4: KV Cache Save ---"
    CACHE_FILES=$(ls "$CACHE_DIR"/* 2>/dev/null | wc -l)
    if [ "$CACHE_FILES" -gt 0 ]; then
        run_check "KV cache files created ($CACHE_FILES file(s))" "true"
        ls -lh "$CACHE_DIR"/* 2>/dev/null | head -3
        
        CACHE_FILE=$(ls "$CACHE_DIR"/* 2>/dev/null | head -1)
        echo "  Cache file: $CACHE_FILE"
    else
        run_check "KV cache files created" "false"
        echo "  No files found in $CACHE_DIR"
    fi
    
    # Test 5: Restart server and verify trie rebuild from disk
    echo ""
    echo "--- Test 5: Server Restart (Trie Rebuild) ---"

    # Stop server
    cleanup_server "$SERVER_PID"
    sleep 2

    # Start server again with same cache dir
    echo "Restarting server with same cache dir..."
    "$BUILD_DIR/bin/llama-server" \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --kv-cache-auto \
        --max-cache-size 1 \
        --cache-ttl 3600 \
        --kv-cache-dir "$CACHE_DIR" \
        -lv 4 \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    sleep 15

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server failed to restart!"
        tail -30 "$SERVER_LOG"
        cleanup_server "$SERVER_PID"
        exit 1
    fi

    echo "✓ Server restarted (PID: $SERVER_PID)"

    # Check trie rebuild log
    REBUILD_LOG=$(grep -c "KV cache rebuild" "$SERVER_LOG" 2>/dev/null || echo "0")
    if [ "$REBUILD_LOG" -gt 0 ]; then
        run_check "Trie rebuild from disk confirmed ($REBUILD_LOG message(s))" "true"
        echo "  Rebuild log: $(grep 'KV cache rebuild' "$SERVER_LOG" | head -1)"
    else
        run_check "Trie rebuild from disk confirmed" "false"
    fi

    # Check that cache files survived (not deleted by reconcile_orphaned_files)
    CACHE_FILES_AFTER_RESTART=$(ls "$CACHE_DIR"/* 2>/dev/null | wc -l)
    if [ "$CACHE_FILES_AFTER_RESTART" -gt 0 ]; then
        run_check "Cache files survived restart ($CACHE_FILES_AFTER_RESTART file(s))" "true"
    else
        run_check "Cache files survived restart" "false"
    fi

    # Make same request again -- should HIT from rebuilt trie
    RESPONSE2=$(curl -s --max-time 60 http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "test",
            "messages": [{"role": "user", "content": "Tell me a short joke"}]
        }' 2>/dev/null || echo "")

    if [ -n "$RESPONSE2" ]; then
        run_check "Request after restart response received" "true"
    else
        run_check "Request after restart response received" "false"
    fi

    # Check for cache hit after restart
    if grep -qi "KV cache.*HIT\|KV cache HIT" "$SERVER_LOG" 2>/dev/null; then
        run_check "Cache HIT after restart (trie restored from disk)" "true"
    else
        run_check "Cache HIT after restart (trie restored from disk)" "false"
        echo "  Note: cache MISS after restart may indicate trie rebuild bug"
    fi

    # Stop server again (will be restarted for next tests)
    cleanup_server "$SERVER_PID"
    sleep 2

    # Start server again for remaining tests
    echo "Restarting server for remaining tests..."
    rm -f "$SERVER_LOG"
    "$BUILD_DIR/bin/llama-server" \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --kv-cache-auto \
        --max-cache-size 1 \
        --cache-ttl 3600 \
        --kv-cache-dir "$CACHE_DIR" \
        -lv 4 \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    sleep 15

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server failed to restart for remaining tests!"
        tail -30 "$SERVER_LOG"
        cleanup_server "$SERVER_PID"
        exit 1
    fi

    # Test 6: Check for KV cache metrics logging
    echo ""
    echo "--- Test 5: Metrics Logging ---"
    grep -q "KV cache stats" "$SERVER_LOG" && run_check "KV cache metrics logged" "true" || run_check "KV cache metrics logged (may take 5 min)" "false"
    
    METRICS=$(grep "KV cache stats" "$SERVER_LOG" | tail -1)
    if [ -n "$METRICS" ]; then
        echo "  Metrics: $METRICS"
    fi
    
    # Test 7: Check callback invocation logs
    echo ""
    echo "--- Test 6: Callback Invocation ---"
    CALLBACK_INVOKED=$(grep -c "KV cache callback invoked" "$SERVER_LOG" 2>/dev/null || echo "0")
    if [ "$CALLBACK_INVOKED" -gt 0 ]; then
        run_check "Callback was invoked ($CALLBACK_INVOKED time(s))" "true"
        echo "  Last callback log: $(grep 'KV cache callback invoked' $SERVER_LOG | tail -1)"
    else
        run_check "Callback was invoked" "false"
        echo "  Callback may not have been called or logged at this verbosity level"
    fi
    
    # Test 8: Check KV cache save logs
    echo ""
    echo "--- Test 7: KV Cache Save Logs ---"
    SAVE_LOGS=$(grep -c "KV cache saved" "$SERVER_LOG" 2>/dev/null || echo "0")
    if [ "$SAVE_LOGS" -gt 0 ]; then
        run_check "KV cache save confirmed in logs ($SAVE_LOGS time(s))" "true"
        echo "  Save log: $(grep 'KV cache saved' $SERVER_LOG | tail -1)"
    else
        run_check "KV cache save confirmed in logs" "false"
        echo "  No save confirmation found in logs"
    fi
    
    # Stop server
    cleanup_server "$SERVER_PID"
    
    echo ""
    echo "Server stopped."
done

echo ""
echo "=============================================="
echo "Test Summary"
echo "=============================================="
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "Total:  $((PASS_COUNT + FAIL_COUNT))"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "All tests passed! ✓"
    exit 0
else
    echo "Some tests failed. Check logs for details."
    exit 1
fi
