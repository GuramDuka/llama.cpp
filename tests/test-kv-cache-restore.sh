#!/bin/bash
# Test KV cache restore functionality with rapid requests

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
MODEL_DIR="$PROJECT_DIR/models"

MODELS=(
    "SmolLM-135M-Instruct.i1-Q4_K_M.gguf"
    "LFM2.5-350M.i1-Q4_K_M.gguf"
)

for MODEL_FILE in "${MODELS[@]}"; do
    MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model not found: $MODEL_PATH"
        exit 1
    fi
    
    PORT=$((8200 + RANDOM % 100))
    CACHE_DIR="/tmp/kv-cache-restore-$(echo $MODEL_FILE | sed 's/\.gguf//g')"
    SERVER_LOG="/tmp/server-restore-$MODEL_FILE.log"
    
    echo ""
    echo "=============================================="
    echo "Testing KV Cache Restore: $MODEL_FILE"
    echo "Port: $PORT, Cache dir: $CACHE_DIR"
    echo "=============================================="
    
    # Create cache directory and clean it
    mkdir -p "$CACHE_DIR"
    rm -rf "$CACHE_DIR"/*
    rm -f "$SERVER_LOG"
    
    # Start server with higher similarity threshold for better matching
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
    
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server failed to start!"
        tail -30 "$SERVER_LOG"
        exit 1
    fi
    
    echo "✓ Server started (PID: $SERVER_PID)"
    
    # Make first request with similar prompt
    echo ""
    echo "--- Request 1: 'Tell me a joke about cats' ---"
    RESPONSE1=$(curl -s --max-time 60 http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "test",
            "messages": [{"role": "user", "content": "Tell me a joke about cats"}]
        }' 2>/dev/null || echo "")
    
    if [ -n "$RESPONSE1" ]; then
        echo "✓ Response received"
        TOKENS=$(echo "$RESPONSE1" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*$')
        echo "  Tokens: $TOKENS"
    else
        echo "✗ No response"
    fi
    
    # Wait for cache save
    sleep 5
    
    # Check cache file was created
    CACHE_FILE=$(ls "$CACHE_DIR"/* 2>/dev/null | head -1)
    if [ -n "$CACHE_FILE" ]; then
        echo "✓ Cache file created: $(basename $CACHE_FILE)"
        ls -lh "$CACHE_FILE"
    else
        echo "✗ No cache file created"
    fi
    
    # Make second request with VERY similar prompt (should trigger restore)
    echo ""
    echo "--- Request 2: 'Tell me a joke about cats please' ---"
    echo "(Very similar to first request - should restore from cache)"
    
    RESPONSE2=$(curl -s --max-time 60 http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "test",
            "messages": [{"role": "user", "content": "Tell me a joke about cats please"}]
        }' 2>/dev/null || echo "")
    
    if [ -n "$RESPONSE2" ]; then
        echo "✓ Response received"
        TOKENS=$(echo "$RESPONSE2" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*$')
        echo "  Tokens: $TOKENS"
    else
        echo "✗ No response"
    fi
    
    # Wait for any restore to happen
    sleep 3
    
    # Check logs for cache hit/restore
    echo ""
    echo "--- Checking for KV Cache Hit/Restore ---"
    
    CACHE_HIT=$(grep -c "KV cache HIT" "$SERVER_LOG" 2>/dev/null || echo "0")
    RESTORED=$(grep -c "restored slot from disk cache" "$SERVER_LOG" 2>/dev/null || echo "0")
    
    if [ "$CACHE_HIT" -gt 0 ]; then
        echo "✓ KV cache HIT detected: $CACHE_HIT time(s)"
        grep "KV cache HIT" "$SERVER_LOG" | tail -1
    else
        echo "✗ No KV cache HIT detected"
        echo "  (prompts may not be similar enough for LCP matching)"
    fi
    
    if [ "$RESTORED" -gt 0 ]; then
        echo "✓ Slot restored from disk: $RESTORED time(s)"
        grep "restored slot" "$SERVER_LOG" | tail -1
    else
        echo "✗ No slot restoration detected"
    fi
    
    # Show all KV cache related logs
    echo ""
    echo "--- All KV Cache Logs ---"
    grep "KV cache\|restored slot" "$SERVER_LOG" 2>/dev/null | tail -10
    
    # Cleanup
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "Restore Test Complete"
echo "=============================================="
