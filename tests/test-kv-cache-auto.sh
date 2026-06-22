#!/bin/bash
# Test script for KV cache auto feature - tests models sequentially
# Downloads models from HuggingFace if not present in models/
#
# Detailed tests (SmolLM): disk LCP vs RAM LCP priority, restart/rebuild, eviction
# Lightweight tests (LFM2.5, Qwen3.5): basic save/restore/smoke

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

###############################################################################
# Start/stop helpers
###############################################################################

start_server() {
    local model_path="$1" port="$2" cache_dir="$3" server_log="$4" n_parallel="$5" ttl="$6"
    n_parallel="${n_parallel:-1}"
    ttl="${ttl:-3600}"
    "$BUILD_DIR/bin/llama-server" \
        --model "$model_path" \
        --port "$port" \
        --kv-cache-auto \
        --max-cache-size 1 \
        --cache-ttl "$ttl" \
        --kv-cache-dir "$cache_dir" \
        --parallel "$n_parallel" \
        -lv 4 \
        >> "$server_log" 2>&1 &
    SERVER_PID=$!
    # Wait for server to be ready
    for i in $(seq 1 30); do
        curl -sf http://localhost:$port/health > /dev/null 2>&1 && break
        sleep 1
    done
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server failed to start!"
        tail -30 "$server_log"
        exit 1
    fi
    echo "  Server started (PID $SERVER_PID, port $port, parallel $n_parallel)"
}

send_request() {
    local port="$1" content="$2" max_tokens="$3"
    max_tokens="${max_tokens:-10}"
    curl -s --max-time 60 http://localhost:$port/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$content\"}],\"max_tokens\":$max_tokens}" \
        > /dev/null 2>&1
}

send_request_get() {
    local port="$1" content="$2" max_tokens="$3"
    max_tokens="${max_tokens:-10}"
    curl -s --max-time 60 http://localhost:$port/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$content\"}],\"max_tokens\":$max_tokens}" \
        2>/dev/null || echo ""
}

###############################################################################
# Detailed tests: SmolLM (model 0)
###############################################################################

# Helper: count log lines matching pattern since marker
count_log_since() {
    local pattern="$1" marker="$2" logfile="$3"
    local count=0
    local found_marker=0
    while IFS= read -r line; do
        if echo "$line" | grep -q "$marker"; then
            found_marker=1
        fi
        if [ "$found_marker" -eq 1 ] && echo "$line" | grep -q "$pattern"; then
            count=$((count + 1))
        fi
    done < "$logfile"
    echo "$count"
}

run_detailed_tests() {
    local model_path="$1" port="$2" cache_dir="$3" server_log="$4"

    echo ""
    echo "=============================================="
    echo "DETAILED TESTS: SmolLM-135M"
    echo "Port: $port, Cache: $cache_dir, Parallel: 4"
    echo "=============================================="

    # =====================================================================
    # Phase A: Baseline — init, first save
    # =====================================================================
    start_server "$model_path" "$port" "$cache_dir" "$server_log" 4 3600
    echo ">>> MARK: phase_a_start" >> "$server_log"

    # --- Test 1: Initialization ---
    echo ""
    echo "--- Test 1: KV Cache Initialization ---"
    grep -q "KV cache auto enabled" "$server_log" && run_check "KV cache auto enabled" "true" || run_check "KV cache auto enabled" "false"
    grep -q "KV cache disk manager initialized" "$server_log" && run_check "Disk manager initialized" "true" || run_check "Disk manager initialized" "false"
    [ -d "$cache_dir" ] && run_check "Cache directory exists" "true" || run_check "Cache directory exists" "false"

    # --- Test 2: First request (save to disk) ---
    echo ""
    echo "--- Test 2: First Request - Save KV Cache ---"
    RESPONSE=$(send_request_get "$port" "Tell me a short joke" 10)
    [ -n "$RESPONSE" ] && run_check "Request 1 response received" "true" || run_check "Request 1 response received" "false"

    sleep 5  # wait for save callback

    CACHE_FILES=$(ls "$cache_dir"/* 2>/dev/null | wc -l)
    [ "$CACHE_FILES" -gt 0 ] && run_check "Cache files created ($CACHE_FILES)" "true" || run_check "Cache files created" "false"

    cleanup_server "$SERVER_PID"
    sleep 2

    # =====================================================================
    # Phase B: Disk LCP > RAM LCP — after restart, RAM empty, disk has entry
    # =====================================================================
    rm -rf "$cache_dir"
    mkdir -p "$cache_dir"
    start_server "$model_path" "$port" "$cache_dir" "$server_log" 4 3600
    echo ">>> MARK: phase_b_start" >> "$server_log"

    # Save entry to disk
    send_request "$port" "What is 2+2? Briefly." 5 > /dev/null
    sleep 5  # wait for save

    # Restart: RAM empty, disk has entry
    cleanup_server "$SERVER_PID"
    sleep 2
    start_server "$model_path" "$port" "$cache_dir" "$server_log" 4 3600

    # Same request: disk LCP=1.0 > RAM LCP=0.0 => disk should win
    send_request "$port" "What is 2+2? Briefly." 5 > /dev/null
    sleep 3

    # --- Test 3: Disk LCP > RAM LCP ---
    echo ""
    echo "--- Test 3: Disk LCP > RAM LCP (disk wins after restart) ---"
    DISK_RESTORE_B=$(count_log_since "restored slot from disk cache" "phase_b_start" "$server_log")
    [ "$DISK_RESTORE_B" -gt 0 ] && run_check "Disk restore after restart (disk LCP > RAM LCP)" "true" || run_check "Disk restore after restart (disk LCP > RAM LCP)" "false"

    DISK_LCP_LOGGED_B=$(count_log_since "disk LCP=" "phase_b_start" "$server_log")
    [ "$DISK_LCP_LOGGED_B" -gt 0 ] && run_check "Disk LCP value logged" "true" || run_check "Disk LCP value logged" "false"
    echo "  Log: $(grep 'disk LCP=' "$server_log" | tail -1)"

    cleanup_server "$SERVER_PID"
    sleep 2

    # =====================================================================
    # Phase C: RAM LCP > Disk LCP — RAM has recent entry, disk has old one
    # =====================================================================
    # Clear cache so disk has NO entry
    rm -rf "$cache_dir"
    mkdir -p "$cache_dir"
    start_server "$model_path" "$port" "$cache_dir" "$server_log" 4 3600
    echo ">>> MARK: phase_c_start" >> "$server_log"

    # Request A: saves to disk
    send_request "$port" "What is 2+2? Briefly." 5
    sleep 5

    # Request B: partially overlaps with A, saves to disk
    send_request "$port" "What is 3+3? Briefly." 5
    sleep 5

    # Now request A again: slot 0 has A on GPU (RAM LCP=1.0),
    # disk also has A (disk LCP=1.0). Equal => neither branch fires,
    # falls through to RAM level 2. But to test RAM PREFERRED,
    # we need RAM LCP > disk LCP. Use request C that matches A partially
    # on GPU but disk has only B with low LCP.
    send_request "$port" "What is 2+3? Briefly." 5
    sleep 3

    # --- Test 4: RAM LCP > Disk LCP ---
    echo ""
    echo "--- Test 4: RAM LCP > Disk LCP (RAM wins, skip disk restore) ---"
    RAM_PREFERRED_C=$(count_log_since "RAM cache preferred" "phase_c_start" "$server_log")
    [ "$RAM_PREFERRED_C" -gt 0 ] && run_check "RAM cache preferred (skip disk restore)" "true" || run_check "RAM cache preferred (skip disk restore)" "false"
    if [ "$RAM_PREFERRED_C" -gt 0 ]; then
        echo "  Log: $(grep 'RAM cache preferred' "$server_log" | tail -1)"
    fi

    cleanup_server "$SERVER_PID"
    sleep 2

    # =====================================================================
    # Phase D: Both caches empty (LRU fallback)
    # =====================================================================
    rm -rf "$cache_dir"
    mkdir -p "$cache_dir"
    start_server "$model_path" "$port" "$cache_dir" "$server_log" 4 3600
    echo ">>> MARK: phase_d_start" >> "$server_log"

    # Totally new prompt, no disk entry, no RAM entry
    send_request "$port" "What is the meaning of life?" 5
    sleep 3

    # --- Test 5: Both empty (LRU slot) ---
    echo ""
    echo "--- Test 5: Both caches empty (LRU slot) ---"
    LRU_D=$(count_log_since "selected slot by LRU" "phase_d_start" "$server_log")
    [ "$LRU_D" -gt 0 ] && run_check "LRU slot selected (both caches empty)" "true" || run_check "LRU slot selected (both caches empty)" "false"

    cleanup_server "$SERVER_PID"
    sleep 2

    # =====================================================================
    # Phase E: Disk MISS + RAM partial match
    # =====================================================================
    start_server "$model_path" "$port" "$cache_dir" "$server_log" 4 3600
    echo ">>> MARK: phase_e_start" >> "$server_log"

    # Save A to disk and GPU
    send_request "$port" "What is 2+2? Briefly." 5
    sleep 5

    # Request B: disk MISS, but RAM slot has A (partial match)
    send_request "$port" "What is 3+3? Briefly." 5
    sleep 3

    # --- Test 6: Disk MISS, RAM partial ---
    echo ""
    echo "--- Test 6: Disk MISS, RAM has partial match ---"
    MISS_E=$(count_log_since "KV cache.*MISS" "phase_e_start" "$server_log")
    [ "$MISS_E" -gt 0 ] && run_check "Disk MISS for new prompt" "true" || run_check "Disk MISS for new prompt" "false"

    cleanup_server "$SERVER_PID"
    sleep 2

    # =====================================================================
    # Phase F: Trie rebuild from disk + HIT
    # =====================================================================
    start_server "$model_path" "$port" "$cache_dir" "$server_log" 4 3600
    echo ">>> MARK: phase_f_start" >> "$server_log"

    # Save entry
    send_request "$port" "What is 2+2? Briefly." 5
    sleep 5

    # Restart and check rebuild
    cleanup_server "$SERVER_PID"
    sleep 2

    CACHE_FILES_BEFORE=$(ls "$cache_dir"/* 2>/dev/null | wc -l)
    run_check "Cache files before restart: $CACHE_FILES_BEFORE" "$([ "$CACHE_FILES_BEFORE" -gt 0 ] && echo true || echo false)"

    start_server "$model_path" "$port" "$cache_dir" "$server_log" 4 3600

    # --- Test 7: Trie rebuild ---
    echo ""
    echo "--- Test 7: Trie Rebuild from Disk ---"
    if grep -q "KV cache rebuild" "$server_log"; then
        run_check "Trie rebuild from disk confirmed" "true"
        echo "  Log: $(grep 'KV cache rebuild' "$server_log" | tail -1)"
    else
        run_check "Trie rebuild from disk confirmed" "false"
    fi

    CACHE_FILES_AFTER=$(ls "$cache_dir"/* 2>/dev/null | wc -l)
    [ "$CACHE_FILES_AFTER" -gt 0 ] && run_check "Cache files survived restart ($CACHE_FILES_AFTER)" "true" || run_check "Cache files survived restart" "false"

    # Same request should HIT
    send_request "$port" "What is 2+2? Briefly." 5
    sleep 3

    if grep -q "KV cache.*HIT" "$server_log"; then
        run_check "Cache HIT after rebuild" "true"
    else
        run_check "Cache HIT after rebuild" "false"
    fi

    cleanup_server "$SERVER_PID"
    sleep 2

    # =====================================================================
    # Phase G: TTL eviction
    # =====================================================================
    start_server "$model_path" "$port" "$cache_dir" "$server_log" 4 5
    echo ">>> MARK: phase_g_start" >> "$server_log"

    send_request "$port" "Quick answer: 1+1?" 5
    sleep 3

    echo "  Waiting for TTL expiration (7 seconds)..."
    sleep 7

    send_request "$port" "Quick answer: 1+1?" 5
    sleep 3

    # --- Test 8: TTL eviction ---
    echo ""
    echo "--- Test 8: TTL Eviction ---"
    MISS_G=$(count_log_since "KV cache.*MISS" "phase_g_start" "$server_log")
    [ "$MISS_G" -gt 0 ] && run_check "Cache MISS after TTL expiration" "true" || run_check "Cache MISS after TTL expiration" "false"

    # --- Test 9: Callback and save logs ---
    echo ""
    echo "--- Test 9: Callback Invocation and Save ---"
    CALLBACK_COUNT=$(grep -c "KV cache callback invoked" "$server_log" 2>/dev/null || echo "0")
    [ "$CALLBACK_COUNT" -gt 0 ] && run_check "Callback invoked ($CALLBACK_COUNT times)" "true" || run_check "Callback invoked" "false"

    SAVE_COUNT=$(grep -c "KV cache saved" "$server_log" 2>/dev/null || echo "0")
    [ "$SAVE_COUNT" -gt 0 ] && run_check "KV cache saved ($SAVE_COUNT times)" "true" || run_check "KV cache saved" "false"

    # Stop server
    cleanup_server "$SERVER_PID"
    sleep 2
}

###############################################################################
# Lightweight tests: LFM2.5, Qwen3.5
###############################################################################

run_lightweight_tests() {
    local model_file="$1" model_path="$2" port="$3" cache_dir="$4" server_log="$5"

    echo ""
    echo "=============================================="
    echo "LIGHTWEIGHT TESTS: $model_file"
    echo "Port: $port, Cache: $cache_dir"
    echo "=============================================="

    start_server "$model_path" "$port" "$cache_dir" "$server_log" 1 3600

    # Basic: init
    echo ""
    echo "--- Init ---"
    grep -q "KV cache auto enabled" "$server_log" && run_check "KV cache auto enabled" "true" || run_check "KV cache auto enabled" "false"

    # Request
    echo ""
    echo "--- Request ---"
    RESPONSE=$(send_request_get "$port" "Tell me a short joke" 10)
    [ -n "$RESPONSE" ] && run_check "Response received" "true" || run_check "Response received" "false"
    sleep 5

    # Save
    CACHE_FILES=$(ls "$cache_dir"/* 2>/dev/null | wc -l)
    [ "$CACHE_FILES" -gt 0 ] && run_check "Cache files created ($CACHE_FILES)" "true" || run_check "Cache files created" "false"

    # Restart + rebuild
    echo ""
    echo "--- Restart + Rebuild ---"
    cleanup_server "$SERVER_PID"
    sleep 2

    start_server "$model_path" "$port" "$cache_dir" "$server_log" 1 3600

    if grep -q "KV cache rebuild" "$server_log"; then
        run_check "Trie rebuild confirmed" "true"
    else
        run_check "Trie rebuild confirmed" "false"
    fi

    # Request after restart
    RESPONSE2=$(send_request_get "$port" "Tell me a short joke" 10)
    [ -n "$RESPONSE2" ] && run_check "Response after restart" "true" || run_check "Response after restart" "false"

    if grep -q "KV cache.*HIT" "$server_log"; then
        run_check "Cache HIT after restart" "true"
    else
        run_check "Cache HIT after restart" "false"
    fi

    # Stop
    cleanup_server "$SERVER_PID"
    sleep 2
}

###############################################################################
# Main
###############################################################################

for i in "${!MODELS[@]}"; do
    MODEL_FILE="${MODELS[$i]}"
    DOWNLOAD_URL="${DOWNLOAD_URLS[$i]}"
    MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

    echo ""
    echo "Checking model: $MODEL_FILE"
    download_model "$MODEL_FILE" "$DOWNLOAD_URL" "$MODEL_PATH" || exit 1

    PORT=$((PORT_BASE + RANDOM % 100))
    CACHE_DIR="/tmp/kv-cache-test-$(echo "$MODEL_FILE" | sed 's/\.gguf//g')"
    SERVER_LOG="/tmp/server-$MODEL_FILE.log"

    mkdir -p "$CACHE_DIR"

    if [ "$i" -eq 0 ]; then
        # SmolLM: detailed tests with 4 parallel slots
        run_detailed_tests "$MODEL_PATH" "$PORT" "$CACHE_DIR" "$SERVER_LOG"
    else
        # Other models: lightweight tests
        run_lightweight_tests "$MODEL_FILE" "$MODEL_PATH" "$PORT" "$CACHE_DIR" "$SERVER_LOG"
    fi

    echo ""
    echo "Server stopped for $MODEL_FILE"
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
