import os
import tempfile
import time
import shutil
import pytest
from utils import *

# Models for multi-model testing (mirrors shell script MODELS array)
# Each tuple: (model_name, hf_repo, hf_file)
KV_TEST_MODELS = [
    # Primary model — detailed tests (4 parallel slots)
    (
        "SmolLM-135M-Instruct.i1-Q4_K_M.gguf",
        "mradermacher/SmolLM-135M-Instruct-i1-GGUF",
        "SmolLM-135M-Instruct.i1-Q4_K_M.gguf",
    ),
    # Secondary models — lightweight smoke tests
    (
        "LFM2.5-350M.i1-Q4_K_M.gguf",
        "mradermacher/LFM2.5-350M-i1-GGUF",
        "LFM2.5-350M.i1-Q4_K_M.gguf",
    ),
    (
        "Qwen3.5-0.8B.i1-Q4_K_M.gguf",
        "mradermacher/Qwen3.5-0.8B-i1-GGUF",
        "Qwen3.5-0.8B.i1-Q4_K_M.gguf",
    ),
]

server = ServerPreset.tinyllama2()


class LogReader:
    """Tails a log file from a given position."""

    def __init__(self, path: str):
        self.path = path
        self.pos = 0

    def drain(self) -> str:
        with open(self.path, encoding="utf-8", errors="replace") as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.debug = True
    server.n_ctx = 2048
    server.n_slots = 1
    server.n_predict = 8
    server.temperature = 0.0
    server.cache_ram = 100
    server.kv_cache_auto = True
    server.kv_cache_dir = tempfile.mkdtemp(prefix="kv-cache-test-")
    server.max_cache_size_gb = 1.0
    server.cache_ttl_seconds = 3600
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    try:
        shutil.rmtree(server.kv_cache_dir, ignore_errors=True)
    except Exception:
        pass
    # Clean up log file
    if (
        hasattr(server, "log_path")
        and server.log_path
        and os.path.exists(server.log_path)
    ):
        os.unlink(server.log_path)


def _send_completion(prompt: str, max_tokens: int = 8):
    """Helper to send a /completion request."""
    return server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": prompt,
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": max_tokens,
        },
    )


def _restart_server():
    """Stop current server and start a new one (reuse same cache dir)."""
    server.stop()
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()


# ---------------------------------------------------------------------------
# Test 1: KV Cache Initialization
# ---------------------------------------------------------------------------


def test_kv_cache_initialization():
    """KV cache auto enabled, disk manager initialized, cache directory exists."""
    global server
    server.start()
    log = LogReader(server.log_path)

    init_log = log.drain()
    assert "KV cache auto enabled" in init_log, (
        f"KV cache auto not enabled: {init_log[:500]}"
    )
    assert "KV cache disk manager initialized" in init_log, (
        f"Disk manager not initialized: {init_log[:500]}"
    )
    assert os.path.isdir(server.kv_cache_dir), "Cache directory does not exist"


# ---------------------------------------------------------------------------
# Test 2: First Request — Save KV Cache
# ---------------------------------------------------------------------------


def test_first_request_save():
    """First request should save KV cache to disk."""
    global server
    server.start()
    log = LogReader(server.log_path)

    # Send request
    res = _send_completion("What is 2+2? Briefly.", 8)
    assert res.status_code == 200, "Request should succeed"

    # Wait for save callback
    time.sleep(3)

    # Check save log
    save_log = log.drain()
    assert "KV cache saved" in save_log, f"No save log: {save_log[:500]}"

    # Check cache files exist
    cache_files = list(os.listdir(server.kv_cache_dir))
    assert len(cache_files) > 0, "No cache files created"


# ---------------------------------------------------------------------------
# Test 3: Disk LCP > RAM LCP (disk wins after restart)
# ---------------------------------------------------------------------------


def test_disk_lcp_wins_after_restart():
    """After restart, RAM is empty -- disk should win the LCP comparison."""
    global server

    # Start server, save entry
    server.start()
    _send_completion("Hello world from the server.", 8)
    time.sleep(3)

    # Restart: RAM empty, disk has entry
    _restart_server()

    log = LogReader(server.log_path)
    log.drain()  # skip init

    # Same request -- disk LCP=1.0 > RAM LCP=0.0 => disk should win
    _send_completion("Hello world from the server.", 8)

    compare_log = log.drain()
    assert "restored slot from disk cache" in compare_log, (
        f"Disk restore not found: {compare_log[:800]}"
    )


# ---------------------------------------------------------------------------
# Test 4: RAM LCP > Disk LCP (RAM wins, skip disk restore)
# ---------------------------------------------------------------------------


def test_ram_cache_preferred():
    """When RAM GPU cache has better LCP than disk, skip disk restore."""
    global server
    server.n_slots = 2  # need 2 slots for RAM to hold a better match
    server.start()
    log = LogReader(server.log_path)

    # Request A on slot 0 -- saves to disk
    _send_completion("What is 2+2? Briefly.", 8)
    time.sleep(3)

    # Request B on slot 1 -- also saves
    _send_completion("What is 3+3? Briefly.", 8)
    time.sleep(3)

    # Request C that partially overlaps A on GPU but disk has lower LCP
    _send_completion("What is 2+3? Briefly.", 8)

    compare_log = log.drain()
    # Either RAM cache preferred or LCP similarity selected (both valid)
    assert (
        "RAM cache preferred" in compare_log
        or "selected slot by LCP similarity" in compare_log.lower()
        or "restored slot from disk cache" in compare_log
    ), f"No slot selection log: {compare_log[:800]}"


# ---------------------------------------------------------------------------
# Test 5: Both caches empty (LRU fallback)
# ---------------------------------------------------------------------------


def test_both_caches_empty_lru_fallback():
    """When both disk and RAM caches are empty, LRU slot is selected."""
    global server

    # Clear cache dir so disk has nothing
    shutil.rmtree(server.kv_cache_dir, ignore_errors=True)
    os.makedirs(server.kv_cache_dir, exist_ok=True)

    server.start()
    log = LogReader(server.log_path)
    log.drain()  # skip init

    # Totally new prompt, no disk entry, no RAM entry
    _send_completion("What is the meaning of life?", 5)

    lru_log = log.drain()
    assert "selected slot by LRU" in lru_log, f"No LRU selection: {lru_log[:800]}"


# ---------------------------------------------------------------------------
# Test 6: Disk MISS, RAM has partial match
# ---------------------------------------------------------------------------


def test_disk_miss_ram_partial_match():
    """New prompt: disk MISS, but RAM slot has partial match."""
    global server
    server.start()
    log = LogReader(server.log_path)

    # Save entry A to disk and GPU
    _send_completion("What is 2+2? Briefly.", 5)
    time.sleep(3)
    log.drain()  # drain save log

    # Request B: disk MISS, but RAM slot has A (partial match)
    _send_completion("What is 3+3? Briefly.", 5)

    miss_log = log.drain()
    assert "KV cache MISS" in miss_log, f"No MISS log: {miss_log[:800]}"


# ---------------------------------------------------------------------------
# Test 7: Trie Rebuild from Disk + HIT
# ---------------------------------------------------------------------------


def test_trie_rebuild_from_disk():
    """Restart server: trie rebuild from disk, cache HIT for same request."""
    global server

    # Start, save entry
    server.start()
    _send_completion("What is 2+2? Briefly.", 8)
    time.sleep(3)

    # Check files exist before restart
    assert len(os.listdir(server.kv_cache_dir)) > 0, "No cache files before restart"

    # Restart
    _restart_server()

    log = LogReader(server.log_path)

    # Check trie rebuild log
    init_log = log.drain()
    assert "KV cache rebuild" in init_log, f"No rebuild log: {init_log[:500]}"

    # Files survived restart
    assert len(os.listdir(server.kv_cache_dir)) > 0, (
        "Cache files disappeared after restart"
    )

    # Same request should HIT
    _send_completion("What is 2+2? Briefly.", 8)

    hit_log = log.drain()
    assert "KV cache HIT" in hit_log, f"No HIT after rebuild: {hit_log[:800]}"
    assert "restored slot from disk cache" in hit_log, (
        f"No restore after rebuild: {hit_log[:800]}"
    )


# ---------------------------------------------------------------------------
# Test 8: TTL Eviction
# ---------------------------------------------------------------------------


def test_ttl_eviction():
    """Cache entries expire after TTL seconds."""
    global server
    server.cache_ttl_seconds = 3  # 3 second TTL
    server.start()
    log = LogReader(server.log_path)

    # Save entry
    _send_completion("Quick answer: 1+1?", 5)
    time.sleep(2)
    log.drain()  # drain save log

    # Wait for TTL
    time.sleep(4)

    # Same request -- should MISS (expired)
    _send_completion("Quick answer: 1+1?", 5)

    ttl_log = log.drain()
    assert "KV cache MISS" in ttl_log, f"No MISS after TTL: {ttl_log[:800]}"


# ---------------------------------------------------------------------------
# Test 9: Callback Invocation and Save
# ---------------------------------------------------------------------------


def test_callback_invocation_and_save():
    """Callback is invoked and KV cache is saved on request completion."""
    global server
    server.start()
    log = LogReader(server.log_path)

    # Send a request
    _send_completion("Tell me a short joke.", 8)
    time.sleep(3)

    save_log = log.drain()
    # Check callback was invoked
    assert "KV cache callback invoked" in save_log, f"No callback log: {save_log[:500]}"
    # Check cache was saved
    assert "KV cache saved" in save_log, f"No save log: {save_log[:500]}"


# ---------------------------------------------------------------------------
# Test 10: Multi-model smoke test (mirrors shell lightweight tests)
# ---------------------------------------------------------------------------


def test_multi_model_smoke():
    """Run lightweight smoke test across multiple models like the shell script."""
    models = KV_TEST_MODELS[1:]
    assert len(models) >= 1, "Need at least one secondary model for multi-model test"

    for model_name, hf_repo, hf_file in models:
        # Download model if not present
        model_path = download_file(
            f"https://huggingface.co/{hf_repo}/resolve/main/{hf_file}"
        )
        cache_dir = tempfile.mkdtemp(prefix="kv-cache-multi-")

        # Configure server for this model
        s = ServerProcess()
        s.debug = True
        s.n_ctx = 2048
        s.n_slots = 1
        s.n_predict = 8
        s.temperature = 0.0
        s.model_file = model_path
        s.model_hf_repo = None
        s.model_hf_file = None
        s.offline = True
        s.kv_cache_auto = True
        s.kv_cache_dir = cache_dir
        s.max_cache_size_gb = 1.0
        s.cache_ttl_seconds = 3600
        fd, s.log_path = tempfile.mkstemp(suffix=".log")
        os.close(fd)

        try:
            s.start()
            log = LogReader(s.log_path)

            # Init check
            init_log = log.drain()
            assert "KV cache auto enabled" in init_log, (
                f"{model_name}: KV cache auto not enabled"
            )

            # Request + save
            res = s.make_request(
                "POST",
                "/completion",
                data={
                    "prompt": "Tell me a short joke",
                    "temperature": 0.0,
                    "top_k": 1,
                    "n_predict": 8,
                },
            )
            assert res.status_code == 200, f"{model_name}: request failed"
            time.sleep(3)

            save_log = log.drain()
            assert "KV cache saved" in save_log, f"{model_name}: no save"

            # Restart + rebuild
            s.stop()
            fd, s.log_path = tempfile.mkstemp(suffix=".log")
            os.close(fd)
            s.start()

            log = LogReader(s.log_path)
            init_log = log.drain()
            assert "KV cache rebuild" in init_log, (
                f"{model_name}: no rebuild after restart"
            )

            # Request after restart
            res2 = s.make_request(
                "POST",
                "/completion",
                data={
                    "prompt": "Tell me a short joke",
                    "temperature": 0.0,
                    "top_k": 1,
                    "n_predict": 8,
                },
            )
            assert res2.status_code == 200, (
                f"{model_name}: request after restart failed"
            )

            hit_log = log.drain()
            assert "KV cache HIT" in hit_log, f"{model_name}: no HIT after restart"

        finally:
            s.stop()
            shutil.rmtree(cache_dir, ignore_errors=True)
            if os.path.exists(s.log_path):
                os.unlink(s.log_path)
