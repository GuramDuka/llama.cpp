#!/bin/bash
# Test script for automatic KV cache save/restore feature
# This script tests persistence across sessions by running two sequential requests

set -e

echo "=========================================="
echo "  KV Cache Auto-Save/Restore Test Script"
echo "=========================================="

# Configuration
MODEL_URL="https://huggingface.co/mradermacher/SmolLM-135M-Instruct-i1-GGUF/resolve/main/SmolLM-135M-Instruct.i1-Q4_K_M.gguf"
MODEL_NAME=$(basename "$MODEL_URL")
CACHE_DIR="./kv-cache-test"
SERVER_PORT=8080

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Download model if not present
echo ""
echo "Step 1: Checking for model..."
if [ ! -f "$MODEL_NAME" ]; then
    log_info "Downloading model: $MODEL_URL"
    wget -q --show-progress "$MODEL_URL" || curl -L -o "$MODEL_NAME" "$MODEL_URL"
else
    log_info "Model already exists: $MODEL_NAME"
fi

# Step 2: Clean up any previous test data
echo ""
echo "Step 2: Cleaning up previous test data..."
rm -rf "$CACHE_DIR"
pkill -f "llama-server.*$SERVER_PORT" 2>/dev/null || true
sleep 1

# Step 2b: Create cache directory (required by --kv-cache-dir validation)
mkdir -p "$CACHE_DIR"

# Step 3: Start server with KV cache auto-save enabled
echo ""
echo "Step 3: Starting llama-server with KV cache auto-save..."
log_info "Cache directory: $CACHE_DIR"
log_info "Max cache size: 1 GB"
log_info "Cache TTL: 3600 seconds (1 hour)"

./build/bin/llama-server \
    -m "$MODEL_NAME" \
    --port "$SERVER_PORT" \
    --kv-cache-auto \
    --kv-cache-dir "$CACHE_DIR" \
    --max-cache-size 1 \
    --cache-ttl 3600 \
    --log-verbosity 4 \
    --n-predict 50 \
    > server.log 2>&1 &

SERVER_PID=$!
log_info "Server started with PID: $SERVER_PID"

# Wait for server to be ready
echo ""
echo "Step 4: Waiting for server to be ready..."
MAX_WAIT=60
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
        log_info "Server is ready!"
        # Wait extra time for model to be fully loaded and slots initialized
        sleep 3
        break
    fi
    WAIT_COUNT=$((WAIT_COUNT + 1))
    sleep 1
done

if [ $WAIT_COUNT -eq $MAX_WAIT ]; then
    log_error "Server failed to start within ${MAX_WAIT} seconds"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Step 5: First request - should NOT hit cache (cache miss)
echo ""
echo "Step 5: Making FIRST request (expected: Cache Miss)..."
FIRST_RESPONSE=$(curl -s http://localhost:$SERVER_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "",
        "messages": [
            {"role": "user", "content": "Hello, can you introduce yourself in one sentence?"}
        ],
        "max_tokens": 50,
        "stream": false
    }')

echo "$FIRST_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$FIRST_RESPONSE"
echo ""

# Check for cache miss in logs (case-insensitive)
if grep -qi "KV cache.*miss\|Cache Miss" server.log 2>/dev/null; then
    log_info "✓ Confirmed: Cache MISS detected in logs"
else
    log_warn "Could not confirm cache miss in logs"
fi

# Wait a moment for save to complete
sleep 2

# Step 6: Check if cache was saved
echo ""
echo "Step 6: Checking if cache was saved..."
if [ -d "$CACHE_DIR" ] && [ "$(ls -A $CACHE_DIR 2>/dev/null)" ]; then
    log_info "✓ Cache directory contains files:"
    ls -lh "$CACHE_DIR/"
else
    log_warn "Cache directory is empty or doesn't exist"
fi

# Step 7: Second request with SAME prompt - should HIT cache
echo ""
echo "Step 7: Making SECOND request with SAME prompt (expected: Cache Hit)..."
SECOND_RESPONSE=$(curl -s http://localhost:$SERVER_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "",
        "messages": [
            {"role": "user", "content": "Hello, can you introduce yourself in one sentence?"}
        ],
        "max_tokens": 50,
        "stream": false
    }')

echo "$SECOND_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$SECOND_RESPONSE"
echo ""

# Check for cache hit in logs (case-insensitive)
if grep -qi "KV cache.*hit\|Cache Hit" server.log 2>/dev/null; then
    log_info "✓ Confirmed: Cache HIT detected in logs"
else
    log_warn "Could not confirm cache hit in logs"
fi

# Step 8: Display metrics from server
echo ""
echo "Step 8: Retrieving server metrics..."
METRICS=$(curl -s http://localhost:$SERVER_PORT/metrics)
if [ -n "$METRICS" ]; then
    echo "$METRICS" | head -20
fi

# Step 9: Summary
echo ""
echo "=========================================="
echo "  Test Summary"
echo "=========================================="
echo ""
echo "Server logs saved to: server.log"
echo "Cache directory: $CACHE_DIR"
echo ""
echo "To verify KV cache operations, check server.log for:"
echo "  - 'KV cache' messages (verbosity >= 4)"
echo "  - Cache hit/miss indicators"
echo "  - Save/eviction statistics"
echo ""

# Cleanup
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

log_info "Test completed!"
