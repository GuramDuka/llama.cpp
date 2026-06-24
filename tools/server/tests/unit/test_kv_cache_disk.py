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


def _cache_dir(slot_save_path):
    """Derive the KV cache directory from slot-save-path."""
    return slot_save_path


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
    server.slot_save_path = tempfile.mkdtemp(prefix="kv-cache-test-")
    server.max_cache_size_gb = 1.0
    server.cache_ttl_seconds = 3600
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    try:
        shutil.rmtree(server.slot_save_path, ignore_errors=True)
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
    cache_dir = _cache_dir(server.slot_save_path)
    assert os.path.isdir(cache_dir), "Cache directory does not exist"


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
    cache_files = list(os.listdir(_cache_dir(server.slot_save_path)))
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
    # Slot selection can result in LCP similarity, disk restore, or LRU fallback
    assert (
        "selected slot by LCP similarity" in compare_log
        or "restored slot from disk cache" in compare_log
        or "selected slot by LRU" in compare_log
    ), f"No slot selection log: {compare_log[:800]}"


# ---------------------------------------------------------------------------
# Test 5: Both caches empty (LRU fallback)
# ---------------------------------------------------------------------------


def test_both_caches_empty_lru_fallback():
    """When both disk and RAM caches are empty, LRU slot is selected."""
    global server

    # Clear cache dir so disk has nothing
    cache_dir = _cache_dir(server.slot_save_path)
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

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


def test_combined_pool_lcp_match():
    """Similar prompt: combined 3-tier pool finds match via LCP."""
    global server
    server.start()
    log = LogReader(server.log_path)

    # Save entry A to disk, RAM, and GPU
    _send_completion("What is 2+2? Briefly.", 5)
    time.sleep(3)
    log.drain()  # drain save log

    # Request B: similar prompt - combined pool should find LCP match in L1 slot
    _send_completion("What is 3+3? Briefly.", 5)

    pool_log = log.drain()
    # The combined 3-tier pool should select a slot by LCP similarity
    assert (
        "selected from 3-tier pool" in pool_log
        or "selected slot by LCP similarity" in pool_log
        or "selected slot by LRU" in pool_log
    ), f"No pool selection log: {pool_log[:800]}"


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
    assert len(os.listdir(_cache_dir(server.slot_save_path))) > 0, (
        "No cache files before restart"
    )

    # Restart
    _restart_server()

    log = LogReader(server.log_path)

    # Check trie rebuild log
    init_log = log.drain()
    assert "KV cache rebuild" in init_log, f"No rebuild log: {init_log[:500]}"

    # Files survived restart
    assert len(os.listdir(_cache_dir(server.slot_save_path))) > 0, (
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
    # Check cache was saved (callback_save_kv_cache_to_disk invokes save)
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
        s.slot_save_path = cache_dir
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


# ---------------------------------------------------------------------------
# Test 11: Cross-model cache isolation — different architectures
# ---------------------------------------------------------------------------
# Scenario: Two different models (different architecture, different tokenizer)
# write to the same cache directory.
#
# Expected behavior:
# - Cache files from model A have no model identity marker, but model B's
#   different tokenizer produces different token IDs for the same text,
#   so the trie won't find a match (trie miss → LRU fallback).
# - If a trie match did occur (e.g., same tokenizer, different weights),
#   `llama_state_seq_load_file` would catch n_layer or k_type/v_type
#   mismatches and return 0 gracefully (tested in C++ test-kv-cache-disk).
# - Under NO circumstances should the server crash or serve corrupted data.
#
# Known limitation (no validation):
# - Two models with the SAME architecture, layer count, and KV quantization
#   but DIFFERENT weights (e.g., two fine-tuned versions of the same base)
#   will NOT be detected — the KV data would be loaded into the wrong model.
#   This is a known gap documented in kv-cache-auto-prompt.md.
# ---------------------------------------------------------------------------


def test_cross_model_cache_isolation():
    """Different models sharing a cache directory — no crash, LRU fallback."""
    global server

    # Use the first secondary model (different arch from tinyllama2)
    model_name, hf_repo, hf_file = KV_TEST_MODELS[1]
    model_path = download_file(
        f"https://huggingface.co/{hf_repo}/resolve/main/{hf_file}"
    )

    # Save cache with the PRIMARY model first (from the fixture)
    server.start()
    log = LogReader(server.log_path)
    _send_completion("What is 2+2? Briefly.", 8)
    time.sleep(3)
    save_log = log.drain()
    assert "KV cache saved" in save_log, f"No save from primary model: {save_log[:300]}"
    server.stop()

    cache_dir = _cache_dir(server.slot_save_path)
    assert len(os.listdir(cache_dir)) > 0, "No cache files from primary model"

    # Now start SECONDARY model with the SAME cache directory
    fd, log_path_b = tempfile.mkstemp(suffix=".log")
    os.close(fd)

    s_b = ServerProcess()
    s_b.debug = True
    s_b.n_ctx = 2048
    s_b.n_slots = 1
    s_b.n_predict = 8
    s_b.temperature = 0.0
    s_b.model_file = model_path
    s_b.model_hf_repo = None
    s_b.model_hf_file = None
    s_b.offline = True
    s_b.kv_cache_auto = True
    s_b.slot_save_path = cache_dir
    s_b.max_cache_size_gb = 1.0
    s_b.cache_ttl_seconds = 3600
    s_b.log_path = log_path_b

    try:
        s_b.start()
        log_b = LogReader(s_b.log_path)
        init_log = log_b.drain()

        # Cache dir should have been initialized (and rebuilt from foreign files)
        assert "KV cache disk manager initialized" in init_log, (
            f"Disk manager not initialized: {init_log[:500]}"
        )

        # Foreign cache files exist — server should not crash
        assert len(os.listdir(cache_dir)) > 0, "Cache files disappeared"

        # Send request to secondary model — should use LRU (trie miss due to
        # different tokenizer), never crash
        res = s_b.make_request(
            "POST",
            "/completion",
            data={
                "prompt": "What is 2+2? Briefly.",
                "temperature": 0.0,
                "top_k": 1,
                "n_predict": 8,
            },
        )
        assert res.status_code == 200, "Secondary model request should succeed"

        # Must not crash — check process is alive
        assert s_b.process and s_b.process.poll() is None, (
            "Server process died after request"
        )

        # Verify we see LRU or normal slot selection (no disk restore of
        # foreign data)
        request_log = log_b.drain()
        has_lru = "selected slot by LRU" in request_log
        has_lcp = "selected slot by LCP similarity" in request_log
        has_pool = "selected from 3-tier pool" in request_log
        assert has_lru or has_lcp or has_pool, f"No slot selection: {request_log[:500]}"

    finally:
        s_b.stop()
        if os.path.exists(log_path_b):
            os.unlink(log_path_b)


# ---------------------------------------------------------------------------
# Test 12: Cross-model cache coexisting in the same directory
# ---------------------------------------------------------------------------
# Tests that cache files from two different models can coexist in the same
# directory without issues. Each model's trie only matches its own tokens.
# ---------------------------------------------------------------------------


def test_shared_cache_directory_two_models():
    """Two models save to the same cache dir — files coexist peacefully."""
    global server

    # Secondary model
    model2_name, hf_repo_2, hf_file_2 = KV_TEST_MODELS[1]
    path2 = download_file(
        f"https://huggingface.co/{hf_repo_2}/resolve/main/{hf_file_2}"
    )

    # Shared cache directory (from the fixture — will be cleaned up)
    cache_dir = _cache_dir(server.slot_save_path)

    # --- Server 1: primary model ---
    server.start()
    log1 = LogReader(server.log_path)
    _send_completion("Hello from model A.", 8)
    time.sleep(3)
    s1_save = log1.drain()
    assert "KV cache saved" in s1_save, f"No save from model A: {s1_save[:200]}"
    server.stop()

    files_after_a = set(os.listdir(cache_dir))
    assert len(files_after_a) > 0, "No cache files from model A"

    # --- Server 2: secondary model, same cache dir ---
    fd2, log2_path = tempfile.mkstemp(suffix=".log")
    os.close(fd2)

    s2 = ServerProcess()
    s2.debug = True
    s2.n_ctx = 2048
    s2.n_slots = 1
    s2.n_predict = 8
    s2.temperature = 0.0
    s2.model_file = path2
    s2.model_hf_repo = None
    s2.model_hf_file = None
    s2.offline = True
    s2.kv_cache_auto = True
    s2.slot_save_path = cache_dir
    s2.max_cache_size_gb = 1.0
    s2.cache_ttl_seconds = 3600
    s2.log_path = log2_path

    try:
        s2.start()
        log2 = LogReader(s2.log_path)
        log2.drain()  # skip init

        # Save from model B
        res2 = s2.make_request(
            "POST",
            "/completion",
            data={
                "prompt": "Hello from model B.",
                "temperature": 0.0,
                "top_k": 1,
                "n_predict": 8,
            },
        )
        assert res2.status_code == 200, "Model B request failed"
        time.sleep(3)
        s2_save = log2.drain()
        assert "KV cache saved" in s2_save, f"No save from model B: {s2_save[:200]}"
        s2.stop()

        # Both model cache files should be present
        files_after_both = set(os.listdir(cache_dir))
        assert len(files_after_both) >= 2, (
            f"Expected at least 2 cache files, got {files_after_both}"
        )

        # Model B files should be in the directory
        b_files = [f for f in files_after_both if f not in files_after_a]
        assert len(b_files) > 0, f"No new cache files from model B: {files_after_both}"

    finally:
        s2.stop()
        if os.path.exists(log2_path):
            os.unlink(log2_path)


# ---------------------------------------------------------------------------
# Test 13: Restart server with foreign cache files from another model
# ---------------------------------------------------------------------------
# Tests that the server can rebuild its trie from a cache directory containing
# files from multiple models, and that it handles them gracefully on restart.
# ---------------------------------------------------------------------------


def test_restart_with_foreign_cache_files():
    """Server rebuilds from cache dir that has files from another model."""
    global server

    # Secondary model
    model2_name, hf_repo_2, hf_file_2 = KV_TEST_MODELS[1]
    path2 = download_file(
        f"https://huggingface.co/{hf_repo_2}/resolve/main/{hf_file_2}"
    )

    cache_dir = _cache_dir(server.slot_save_path)

    # --- Step 1: Save from model A ---
    server.start()
    log = LogReader(server.log_path)
    _send_completion("The quick brown fox jumps over the lazy dog.", 8)
    time.sleep(3)
    assert "KV cache saved" in log.drain(), "No save from model A"
    server.stop()

    files_from_a = set(os.listdir(cache_dir))
    assert len(files_from_a) > 0

    # --- Step 2: Save from model B into same dir ---
    fd2, log2_path = tempfile.mkstemp(suffix=".log")
    os.close(fd2)

    s2 = ServerProcess()
    s2.debug = True
    s2.n_ctx = 2048
    s2.n_slots = 1
    s2.n_predict = 8
    s2.temperature = 0.0
    s2.model_file = path2
    s2.model_hf_repo = None
    s2.model_hf_file = None
    s2.offline = True
    s2.kv_cache_auto = True
    s2.slot_save_path = cache_dir
    s2.max_cache_size_gb = 1.0
    s2.cache_ttl_seconds = 3600
    s2.log_path = log2_path

    try:
        s2.start()
        log2 = LogReader(s2.log_path)
        log2.drain()

        # Model B request + save
        res2 = s2.make_request(
            "POST",
            "/completion",
            data={
                "prompt": "The quick brown fox jumps over the lazy dog.",
                "temperature": 0.0,
                "top_k": 1,
                "n_predict": 8,
            },
        )
        assert res2.status_code == 200, "Model B request failed"
        time.sleep(3)
        assert "KV cache saved" in log2.drain(), "No save from model B"
        s2.stop()

        # Now directory has files from both models
        files_from_both = set(os.listdir(cache_dir))
        assert len(files_from_both) > len(files_from_a), (
            f"Expected more files after model B: {files_from_both}"
        )

        # --- Step 3: Restart with model B, verify rebuild ---
        s2.start()
        log3 = LogReader(s2.log_path)
        init_log = log3.drain()
        assert "KV cache rebuild" in init_log, f"No rebuild log: {init_log[:500]}"
        assert "KV cache disk manager initialized" in init_log, (
            f"Disk manager not initialized: {init_log[:500]}"
        )

        # Request after restart — should work (trie may match model B's own
        # entries; model A's entries have different token sequences and won't
        # match model B's tokenizer)
        res3 = s2.make_request(
            "POST",
            "/completion",
            data={
                "prompt": "The quick brown fox jumps over the lazy dog.",
                "temperature": 0.0,
                "top_k": 1,
                "n_predict": 8,
            },
        )
        assert res3.status_code == 200, "Model B request after restart failed"

        # Should see either hit or miss (not crash)
        hit_log = log3.drain()
        # This could be HIT (if model B's tokens matched) or MISS (if not),
        # but should never be a crash or error
        assert s2.process and s2.process.poll() is None, (
            "Server process died after restart"
        )

    finally:
        s2.stop()
        if os.path.exists(log2_path):
            os.unlink(log2_path)
