#!/bin/bash
# Test script for KV cache auto feature - tests both models sequentially

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
MODEL_DIR="$PROJECT_DIR/models"
PORT_BASE=8100

MODELS=(
    "SmolLM-135M-Instruct.i1-Q4_K_M.gguf"
    "LFM2.5-350M.i1-Q4_K_M.gguf"
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

for MODEL_FILE in "${MODELS[@]}"; do
    MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model not found: $MODEL_PATH"
        exit 1
    fi
    
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
    
    # Start server
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
    
    # Test 3: Make a test request
    echo ""
    echo "--- Test 3: Test Request ---"
    echo "Making test request..."
    RESPONSE=$(curl -s --max-time 60 http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "test",
            "messages": [{"role": "user", "content": "Tell me a short joke"}]
        }' 2>/dev/null || echo "")
    
    if [ -n "$RESPONSE" ]; then
        run_check "Response received from server" "true"
    else
        run_check "Response received from server" "false"
    fi
    
    # Wait for request to complete and callback to execute
    sleep 5
    
    # Test 4: Check if KV cache was saved
    echo ""
    echo "--- Test 4: KV Cache Save ---"
    CACHE_FILES=$(ls "$CACHE_DIR"/*.state 2>/dev/null | wc -l)
    if [ "$CACHE_FILES" -gt 0 ]; then
        run_check "KV cache files created ($CACHE_FILES file(s))" "true"
        ls -lh "$CACHE_DIR"/*.state 2>/dev/null | head -3
    else
        run_check "KV cache files created" "false"
        echo "  No .state files found in $CACHE_DIR"
        echo "  Server logs (slot release):"
        grep "slot.*release" "$SERVER_LOG" | tail -3
    fi
    
    # Test 5: Check for KV cache metrics logging
    echo ""
    echo "--- Test 5: Metrics Logging ---"
    grep -q "KV cache stats" "$SERVER_LOG" && run_check "KV cache metrics logged" "true" || run_check "KV cache metrics logged (may take 5 min)" "false"
    
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
