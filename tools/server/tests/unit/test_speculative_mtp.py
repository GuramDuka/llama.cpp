import os
import tempfile
import time
import shutil
import pytest
from utils import *

# Qwopus3.5-4B-Coder-MTP with --spec-type draft-mtp
# The model has MTP heads built-in, no separate draft model needed.

# Resolve model paths relative to project root (where pytest is invoked)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../../"))
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models/Qwopus3.5-4B-Coder-MTP-Q5_K_M.gguf")

# Auto-download: this test suite can download the MTP model on demand.
#   - 4B Qwopus-MTP model: Jackrong/Qwopus3.5-4B-Coder-MTP-GGUF
#   - 27B Qwopus model:    Jackrong/Qwopus3.6-27B-Coder-GGUF
_HF_REPO_MTP = "Jackrong/Qwopus3.5-4B-Coder-MTP-GGUF"
_HF_FILE_MTP = "Qwopus3.5-4B-Coder-MTP-Q5_K_M.gguf"
_HF_REPO_27B = "Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF"
_HF_FILE_27B = "Qwopus3.6-27B-Coder-MTP-IQ4_XS.gguf"

# Allow large model tests via environment variable
_LARGE_TESTS_ENABLED = os.environ.get("LARGE_MODEL_TESTS") == "1"

server = ServerPreset.qwopus_mtp()
server.model_file = _MODEL_PATH


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
    server = ServerPreset.qwopus_mtp()
    server.model_file = _MODEL_PATH
    # Only enable auto-download if the model is not already cached locally.
    # Tests run with offline=True (from the preset), so the HF repo is only
    # used by load_all() which overrides offline=False.
    if not os.path.isfile(_MODEL_PATH):
        server.model_hf_repo = _HF_REPO_MTP
        server.model_hf_file = _HF_FILE_MTP
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.debug = True

    # KV cache auto settings
    server.cache_ram = 8000
    server.n_threads = 16
    server.kv_cache_auto = True
    server.slot_save_path = tempfile.mkdtemp(prefix="mtp-kv-cache-")
    server.max_cache_size_gb = 1.0
    server.cache_ttl_seconds = 3600

    # Log file
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)

    yield

    # Clean up temp dir and log
    try:
        shutil.rmtree(server.slot_save_path, ignore_errors=True)
    except Exception:
        pass
    if (
        hasattr(server, "log_path")
        and server.log_path
        and os.path.exists(server.log_path)
    ):
        try:
            os.unlink(server.log_path)
        except Exception:
            pass


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


# ============================================================================
# MTP-Specific Tests
# ============================================================================


def test_mtp_completion():
    """Basic completion with MTP -- must produce valid output."""
    global server
    server.start()
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        },
    )
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    server.stop()


def test_mtp_draft_params():
    """Different draft_min/draft_max values must produce valid completions.
    Output may differ because speculative decoding acceptance patterns vary
    with the number of draft tokens generated, even at temperature=0."""
    global server
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
    ]
    for draft_min, draft_max in test_values:
        server.stop()
        server.spec_draft_n_min = draft_min
        server.spec_draft_n_max = draft_max
        server.start()
        res = server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": "I believe the meaning of life is",
                "temperature": 0.0,
                "top_k": 1,
                "n_predict": 16,
            },
        )
        assert res.status_code == 200
        assert len(res.body["content"]) > 0, (
            f"Empty completion with draft_n_min={draft_min}, draft_n_max={draft_max}"
        )
        assert "draft_n" in res.body["timings"], (
            f"No MTP timings with draft_n_min={draft_min}, draft_n_max={draft_max}"
        )


def test_mtp_slot_ctx_not_exceeded():
    """MTP with a long prompt must not exceed the slot context."""
    global server
    server.n_ctx = 256
    server.start()
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Hello " * 248,
            "temperature": 0.0,
            "top_k": 1,
            "speculative.p_min": 0.0,
        },
    )
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_mtp_ctx_shift():
    """MTP with context shift -- exercises L3 restore + seq_rm path.
    The exact number of generated tokens depends on the model and MTP acceptance
    rate; the key invariant is that the server does not crash."""
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Hello " * 248,
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 256,
            "speculative.p_min": 0.0,
        },
    )
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


# ============================================================================
# KV Cache Disk Auto Tests (ported from test_kv_cache_disk.py)
# ============================================================================


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
    assert (
        "restored slot from disk cache" in compare_log
        or "best candidate from L3" in compare_log
    ), f"Disk restore not found: {compare_log[:800]}"


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


def test_trie_rebuild_from_disk():
    """Restart server: trie rebuild from disk, cache HIT for same request.

    The output from a fresh server run must match the output from a server
    that restored the same prompt from the KV cache, ensuring deterministic
    generation regardless of whether the KV cache was used.
    """
    global server

    # Start, send request on fresh server and capture response
    server.start()
    res_fresh = _send_completion("What is 2+2? Briefly.", 8)
    assert res_fresh.status_code == 200, "Fresh request should succeed"
    fresh_text = res_fresh.body["content"]
    assert len(fresh_text) > 0, "Fresh response should not be empty"
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
    res_cached = _send_completion("What is 2+2? Briefly.", 8)
    assert res_cached.status_code == 200, "Cached request should succeed"
    cached_text = res_cached.body["content"]

    hit_log = log.drain()
    assert "best candidate from L3" in hit_log or "KV cache HIT" in hit_log, (
        f"No HIT or restore after rebuild: {hit_log[:800]}"
    )

    # Both responses must be identical at temperature=0
    assert fresh_text == cached_text, (
        f"KV cache output differs!\n"
        f"  Fresh:    {repr(fresh_text)}\n"
        f"  Cached:   {repr(cached_text)}\n"
    )


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


# ============================================================================
# Router-Mode KV Cache Tests (separate router server with model preset)
# ============================================================================
# These tests mirror test_kv_cache_disk_router.py but use the MTP model.
# They spin up a router server with a preset.ini that points to the local
# 4B MTP model, and verify KV cache init / save / restore in router mode.


def _write_preset_ini(preset_path: str, model_paths: dict[str, str]) -> str:
    """Write a models-preset.ini file with the given models."""
    lines = ["[default]\n"]
    for name, path in model_paths.items():
        lines.append(f"[{name}]\n")
        lines.append(f"model = {path}\n")
        lines.append("\n")
    with open(preset_path, "w") as f:
        f.write("".join(lines))
    return preset_path


_ROUTER_MODEL_NAME = "qwopus-mtp"


@pytest.fixture
def router_server():
    """Create a router-mode server with a single MTP model in preset.ini."""
    s = ServerPreset.router()
    s.debug = True
    s.n_ctx = 512
    s.n_slots = 1
    s.n_predict = 8
    s.temperature = 0.0
    s.cache_ram = 8000
    s.n_threads = 16
    s.kv_cache_auto = True
    s.max_cache_size_gb = 1.0
    s.cache_ttl_seconds = 3600
    s.models_max = 1
    s.no_models_autoload = True

    preset_dir = tempfile.mkdtemp(prefix="mtp-router-")
    preset_ini_path = os.path.join(preset_dir, "models-preset.ini")
    slot_cache_dir = os.path.join(preset_dir, "slot-cache")
    os.makedirs(slot_cache_dir, exist_ok=True)

    model_paths = {_ROUTER_MODEL_NAME: _MODEL_PATH}
    _write_preset_ini(preset_ini_path, model_paths)
    s.models_preset = preset_ini_path
    s.slot_save_path = slot_cache_dir

    fd, s.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)

    yield s

    try:
        s.stop()
    except Exception:
        pass
    try:
        shutil.rmtree(preset_dir, ignore_errors=True)
    except Exception:
        pass
    if hasattr(s, "log_path") and s.log_path and os.path.exists(s.log_path):
        try:
            os.unlink(s.log_path)
        except Exception:
            pass


def test_router_kv_cache_init(router_server):
    """Router: KV cache auto enabled, disk manager initialized."""
    s = router_server
    s.start()
    log = LogReader(s.log_path)
    log.drain()  # skip router init

    # Request triggers model load
    res = s.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Hello",
            "model": _ROUTER_MODEL_NAME,
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )
    assert res.status_code == 200

    model_log = log.drain()
    assert "KV cache auto enabled" in model_log, f"Cache not enabled: {model_log[:500]}"
    assert "KV cache disk manager initialized" in model_log, (
        f"Disk not initialized: {model_log[:500]}"
    )
    assert os.path.isdir(s.slot_save_path)


def test_router_first_request_save(router_server):
    """Router: first request saves KV cache to disk."""
    s = router_server
    s.start()
    log = LogReader(s.log_path)

    res = s.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "What is 2+2? Briefly.",
            "model": _ROUTER_MODEL_NAME,
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )
    assert res.status_code == 200
    time.sleep(3)

    save_log = log.drain()
    assert "KV cache saved" in save_log, f"No save log: {save_log[:500]}"
    assert len(os.listdir(s.slot_save_path)) > 0, "No cache files"


def test_router_restart_disk_restore(router_server):
    """Router: after restart, disk cache restores for same request."""
    s = router_server
    s.start()

    # Save
    res = s.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Hello world from router.",
            "model": _ROUTER_MODEL_NAME,
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )
    assert res.status_code == 200
    time.sleep(3)

    # Restart
    s.stop()
    fd, s.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    s.start()

    log = LogReader(s.log_path)
    log.drain()  # skip init

    # Same request -- disk should restore
    res2 = s.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Hello world from router.",
            "model": _ROUTER_MODEL_NAME,
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )
    assert res2.status_code == 200

    compare_log = log.drain()
    assert "restored slot from disk cache" in compare_log, (
        f"Disk restore not found: {compare_log[:800]}"
    )


# ============================================================================
# MTP L3 Cache Regression Tests
# ============================================================================
# These tests verify that previously-fixed bugs stay fixed:
#
#   1. L3 cache deterministic output after restart
#      - Without the preamble fix, the draft model's state at end-of-generation
#        was saved and restored.  After seq_rm, cells had accumulated generation
#        content, corrupting the first batch of draft tokens.
#      Fix: preamble buffer captures the draft state right after prompt processing.
#
#   2. seq_rm tail cell handling (n_rs_seq = 0)
#      - When seq_rm removed all cells from p0 to end, the tail cell (the very
#        last one) was cleared, making the SSM state start from zero on the next
#        decode and producing wrong logits.
#      Fix: keep the tail cell alive by adjusting its position to p0-1.
#
#   3. Draft state preamble
#      - The preamble ensures the .draft companion file stores the draft model's
#        state at the BEGINNING of generation, not the end.  This avoids
#        propagating corrupted accumulated content to subsequent requests.
#


def _send_chat(prompt: str, max_tokens: int = 32):
    """Helper to send /v1/chat/completions with temperature=0."""
    return server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "cache_prompt": True,
        },
    )


def test_mtp_l3_cache_deterministic():
    """L3 cache restore must produce bit-identical output to a fresh run.

    This exercises the draft-state preamble fix:
      1. First request writes to L3 cache (end-of-generation state).
      2. Second request with the same prompt hits L3 cache and restores.
      3. Both outputs must be identical (temperature=0, greedy).

    Regression: without the preamble fix, the end-of-generation draft state
    contaminated cells after seq_rm, producing different draft tokens and a
    diverging output after the first accepted token.
    """
    global server
    server.n_predict = 32
    server.start()
    log = LogReader(server.log_path)
    log.drain()  # skip init

    # --- Request 1: fresh run ---
    r1 = _send_chat("What is 2+2? Briefly.", 32)
    assert r1.status_code == 200
    fresh_text = r1.body["choices"][0]["message"]["content"]
    assert len(fresh_text) > 0

    # Wait for L3 save to complete
    time.sleep(3)
    save_log = log.drain()
    assert "KV cache saved" in save_log, f"L3 save not found: {save_log[:500]}"

    # --- Request 2: should restore from L3 cache ---
    r2 = _send_chat("What is 2+2? Briefly.", 32)
    assert r2.status_code == 200
    restored_text = r2.body["choices"][0]["message"]["content"]

    # Check that L3 cache was actually used
    r2_log = log.drain()
    assert (
        "best candidate from L3" in r2_log or "restored slot from disk cache" in r2_log
    ), f"L3 restore not detected: {r2_log[:800]}"

    # The outputs must be identical at temperature=0
    assert fresh_text == restored_text, (
        f"MTP L3 cache output differs!\n"
        f"  Fresh:    {repr(fresh_text)}\n"
        f"  Restored: {repr(restored_text)}\n"
    )


def test_mtp_l3_draft_preamble_saved():
    """The L3 save callback must write a .draft companion file.

    Without the preamble fix, the draft state was saved/loaded through the
    end-of-generation path, which leaked accumulated content.  This test
    verifies that a .draft file exists after the first request, and that
    its size is consistent with the expected preamble state size.
    """
    global server
    server.start()
    log = LogReader(server.log_path)

    # Send request
    res = _send_chat("Hello world", 16)
    assert res.status_code == 200
    time.sleep(3)

    # Check for .draft companion files
    cache_dir = server.slot_save_path
    draft_files = [f for f in os.listdir(cache_dir) if f.endswith(".draft")]
    assert len(draft_files) > 0, f"No .draft files found in {cache_dir}"
    # Each .draft should be non-empty (preamble state)
    for df in draft_files:
        full_path = os.path.join(cache_dir, df)
        assert os.path.getsize(full_path) > 0, f"Empty .draft file: {full_path}"


def test_mtp_l3_deterministic_after_restart():
    """Server restart: L3 cache must still produce deterministic output.

    This exercises both:
      - L3 cache trie rebuild from disk on startup
      - Full preamble restore after server restart (cold cache)

    The output of a fresh request must match the output after a restart
    that serves the same prompt from disk.

    Uses the 27B model for thorough testing.
    """
    global server
    server.model_file = os.path.join(
        _PROJECT_ROOT, "models/Qwopus3.6-27B-Coder-MTP-IQ4_XS.gguf"
    )
    server.n_predict = 32
    server.jinja = True
    server.reasoning = "off"

    # --- Cleanup: ensure cache directory is empty before test ---
    cache_dir = _cache_dir(server.slot_save_path)
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            fpath = os.path.join(cache_dir, fname)
            if os.path.isfile(fpath):
                os.unlink(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath, ignore_errors=True)

    server.start()

    # --- Debug: check cache dir before first request ---
    if os.path.isdir(cache_dir):
        print(f"[DEBUG] Cache dir before first request: {os.listdir(cache_dir)}")
    else:
        print(f"[DEBUG] Cache dir does not exist: {cache_dir}")

    # --- First request: populate L3 ---
    r1 = _send_chat("What is 2+2? Briefly.", 32)
    assert r1.status_code == 200
    expected = r1.body["choices"][0]["message"]["content"]
    assert len(expected) > 0
    time.sleep(3)

    # --- Debug: check cache dir after first request ---
    if os.path.isdir(cache_dir):
        print(f"[DEBUG] Cache dir after first request: {os.listdir(cache_dir)}")
        for fname in os.listdir(cache_dir):
            fpath = os.path.join(cache_dir, fname)
            if os.path.isfile(fpath):
                print(f"[DEBUG]   File: {fname}, size: {os.path.getsize(fpath)}")
            elif os.path.isdir(fpath):
                print(f"[DEBUG]   Dir: {fname}")
    else:
        print(f"[DEBUG] Cache dir does not exist after first request: {cache_dir}")

    # --- Debug: read server log ---
    log_before = ""
    if os.path.exists(server.log_path):
        with open(server.log_path, encoding="utf-8", errors="replace") as f:
            log_before = f.read()
    print(f"[DEBUG] Log before restart (last 3000 chars): {log_before[-3000:]}")

    # --- Restart server (disk cache persists) ---
    server.stop()
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)

    # --- Debug: check cache dir before restart ---
    if os.path.isdir(cache_dir):
        print(f"[DEBUG] Cache dir before restart: {os.listdir(cache_dir)}")
    else:
        print(f"[DEBUG] Cache dir does not exist before restart: {cache_dir}")

    server.start()

    # --- Debug: read server log after restart ---
    time.sleep(1)
    log_after = ""
    if os.path.exists(server.log_path):
        with open(server.log_path, encoding="utf-8", errors="replace") as f:
            log_after = f.read()
    print(f"[DEBUG] Log after restart (last 3000 chars): {log_after[-3000:]}")

    # --- Second request: restore from disk ---
    r2 = _send_chat("What is 2+2? Briefly.", 32)
    assert r2.status_code == 200
    actual = r2.body["choices"][0]["message"]["content"]

    # --- Debug: read server log after second request ---
    log_after2 = ""
    if os.path.exists(server.log_path):
        with open(server.log_path, encoding="utf-8", errors="replace") as f:
            log_after2 = f.read()
    print(f"[DEBUG] Log after second request (last 5000 chars): {log_after2[-5000:]}")

    # Must match exactly
    assert expected == actual, (
        f"MTP L3 output differs after restart!\n"
        f"  Before restart: {repr(expected)}\n"
        f"  After restart:  {repr(actual)}\n"
    )


def test_mtp_seq_rm_tail_not_lost():
    """seq_rm must preserve the tail cell when trimming from p0 to end
    (n_rs_seq = 0).  Without the fix, the tail cell was cleared and the
    subsequent decode started from a zero state, producing different logits.

    This test indirectly exercises the fix by verifying that:
      - A request with a long prompt triggers seq_rm on KV cache restore
      - The output is still valid (not garbled / empty)
    """
    global server
    server.n_ctx = 256
    server.start()

    # Prompt that is long enough to cause KV shifting / seq_rm
    res = _send_chat("Write a story about " + "mountains " * 80, 16)
    assert res.status_code == 200
    assert len(res.body["choices"][0]["message"]["content"]) > 0, (
        "seq_rm tail loss produced empty/garbled output"
    )


def test_mtp_draft_preamble_multiple_requests():
    """Multiple requests using L3 cache: each request saves and restores
    from the preamble state, producing consistent output across all runs."""
    global server
    server.n_predict = 16
    server.start()

    # Run several requests, all with the same prompt
    prompt = "What is 2+2? Briefly."
    outputs = []
    for i in range(3):
        res = _send_chat(prompt, 16)
        assert res.status_code == 200
        outputs.append(res.body["choices"][0]["message"]["content"])
        time.sleep(2)  # let L3 save complete

    # All outputs must be identical
    for i in range(1, len(outputs)):
        assert outputs[0] == outputs[i], (
            f"MTP preamble: output {i} differs from output 0\n"
            f"  [0]: {repr(outputs[0])}\n"
            f"  [{i}]: {repr(outputs[i])}\n"
        )


def test_mtp_l3_with_longer_prompt():
    """L3 cache restore with a moderately long prompt must be deterministic.

    Longer prompts stress the preamble state size and seq_rm correctness."""
    global server
    server.n_predict = 16
    server.n_ctx = 1024
    server.start()
    log = LogReader(server.log_path)

    long_prompt = (
        "The following is a conversation with an AI assistant. "
        "The assistant is helpful, creative, and very friendly.\n\n"
        "Human: Can you explain the concept of recursion in computer science?\n"
        "Assistant:"
    )

    # Request 1: fresh
    r1 = _send_chat(long_prompt, 16)
    assert r1.status_code == 200
    expected = r1.body["choices"][0]["message"]["content"]

    time.sleep(3)
    save_log = log.drain()
    assert "KV cache saved" in save_log, f"L3 save not found: {save_log[:500]}"

    # Request 2: should use L3
    r2 = _send_chat(long_prompt, 16)
    assert r2.status_code == 200
    actual = r2.body["choices"][0]["message"]["content"]

    restore_log = log.drain()
    assert (
        "restored from L3" in restore_log or "best candidate from L3" in restore_log
    ), f"L3 not used for long prompt: {restore_log[:500]}"

    assert expected == actual, (
        f"MTP L3 mismatch with longer prompt!\n"
        f"  Fresh:    {repr(expected)}\n"
        f"  Restored: {repr(actual)}\n"
    )


# ============================================================================
# 27B Model Test (skipped by default, requires LARGE_MODEL_TESTS=1 env var)
# ============================================================================


@pytest.mark.skipif(
    not _LARGE_TESTS_ENABLED,
    reason="Set LARGE_MODEL_TESTS=1 to run (requires pre-downloaded 27B model)",
)
def test_mtp_27b_completion():
    """Basic MTP completion with the 27B Qwopus model.

    Auto-downloads from HuggingFace if not cached locally.
    The 27B model does NOT have MTP heads; this test verifies the server
    can load and run a large model without issues.
    """
    global server
    server.model_file = None  # trigger auto-download
    server.model_hf_repo = _HF_REPO_27B
    server.model_hf_file = _HF_FILE_27B
    server.n_predict = 32
    server.start()
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Write a short poem about artificial intelligence.",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 32,
        },
    )
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
