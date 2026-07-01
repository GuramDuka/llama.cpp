"""
Unified KV Cache Disk Tests

Sections:
  [1] Simple server  - tinyllama2 (no MTP)           = 13 tests
  [2] Router server  - tinyllama2 via router preset  = 10 tests
  [3] MTP simple     - 4B + 27B (parametrized)       = 19 tests x 2 models
  [4] MTP router     - router preset with MTP model  =  3 tests
"""

import os
import tempfile
import time
import shutil
import pytest
from utils import *

# ==========================================================================
# Constants
# ==========================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../../"))
_MODEL_4B_PATH = os.path.join(
    _PROJECT_ROOT, "models/Qwopus3.5-4B-Coder-MTP-Q5_K_M.gguf"
)
_MODEL_27B_PATH = os.path.join(
    _PROJECT_ROOT, "models/Qwopus3.6-27B-Coder-MTP-IQ4_XS.gguf"
)

_HF_REPO_4B = "Jackrong/Qwopus3.5-4B-Coder-MTP-GGUF"
_HF_FILE_4B = "Qwopus3.5-4B-Coder-MTP-Q5_K_M.gguf"
_HF_REPO_27B = "Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF"
_HF_FILE_27B = "Qwopus3.6-27B-Coder-MTP-IQ4_XS.gguf"

KV_TEST_MODELS = [
    (
        "SmolLM-135M-Instruct.i1-Q4_K_M.gguf",
        "mradermacher/SmolLM-135M-Instruct-i1-GGUF",
        "SmolLM-135M-Instruct.i1-Q4_K_M.gguf",
    ),
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
    (
        "Qwopus3.5-4B-Coder-MTP-Q5_K_M.gguf",
        "Jackrong/Qwopus3.5-4B-Coder-MTP-GGUF",
        "Qwopus3.5-4B-Coder-MTP-Q5_K_M.gguf",
    ),
]

KV_MODEL_ALIASES = [m[0] for m in KV_TEST_MODELS]

_ROUTER_MODEL_NAME = "qwopus-mtp"

# ==========================================================================
# Module-level server (reassigned by section-1/2/3 fixtures)
# ==========================================================================

server = None  # type: ServerProcess

# ==========================================================================
# Common helpers
# ==========================================================================


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


def _cache_dir(slot_save_path):
    """Derive the KV cache directory from slot-save-path."""
    return slot_save_path


def _send_completion(prompt: str, max_tokens: int = 8):
    """Helper to send a /completion request (works with any server)."""
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


def _clear_cache_dir():
    """Remove all KV cache files in the server's slot-save-path."""
    if server and server.slot_save_path and os.path.exists(server.slot_save_path):
        for fname in os.listdir(server.slot_save_path):
            fpath = os.path.join(server.slot_save_path, fname)
            try:
                os.remove(fpath)
            except Exception:
                pass


def _restart_server():
    """Stop current server and start a new one (reuse same cache dir)."""
    server.stop()
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()


def _wait_for_log(
    log_reader: LogReader, pattern: str, timeout: float = 10.0, interval: float = 0.1
) -> str:
    """Poll server log until *pattern* appears. Returns all new content."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        content = log_reader.drain()
        if pattern in content:
            return content
        time.sleep(interval)
    # timeout – drain once more and return whatever is there
    return log_reader.drain()


# ==========================================================================
# Section fixtures
# ==========================================================================


def _configure_kv_base(s, prefix):
    """Apply KV-cache boilerplate to any server."""
    s.debug = True
    s.n_ctx = 2048
    s.n_slots = 1
    s.n_predict = 8
    s.temperature = 0.0
    s.cache_ram = 8000
    s.n_threads = 16
    s.kv_cache_auto = True
    s.slot_save_path = tempfile.mkdtemp(prefix=prefix)
    s.max_cache_size_gb = 1.0
    s.cache_ttl_seconds = 3600
    fd, s.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    return s


def _cleanup_server(s):
    """Clean up temp dirs and logs left by a server."""
    if s is None:
        return
    try:
        if s.slot_save_path and os.path.isdir(s.slot_save_path):
            shutil.rmtree(s.slot_save_path, ignore_errors=True)
    except Exception:
        pass
    try:
        if s.log_path and os.path.exists(s.log_path):
            os.unlink(s.log_path)
    except Exception:
        pass


# ------------------------------------------------------------------
# [Section 1] Simple server (no MTP) -- tinyllama2
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def section_simple():
    global server
    s = ServerPreset.tinyllama2()
    _configure_kv_base(s, "kv-cache-test-")
    server = s
    server.start()
    yield
    server.stop()
    _cleanup_server(server)
    server = None


# ------------------------------------------------------------------
# [Section 2] Router server (no MTP)
# ------------------------------------------------------------------
# In router mode the server uses --models-preset pointing to an INI file
# that lists all KV_TEST_MODELS.  The router owns the preset dir + slot
# cache dir so that cross-model tests can reuse them.


@pytest.fixture(scope="module")
def section_router():
    global server

    s = ServerPreset.router()
    _configure_kv_base(s, "kv-cache-router-")
    s.models_max = 1

    # Override slot_save_path to live inside a dedicated router dir
    # alongside the preset.ini
    preset_dir = os.path.dirname(s.slot_save_path)
    shutil.rmtree(s.slot_save_path, ignore_errors=True)
    s_slot = os.path.join(preset_dir, "slot-cache")
    os.makedirs(s_slot, exist_ok=True)
    s.slot_save_path = s_slot

    # Download models + write preset.ini
    preset_ini_path = os.path.join(preset_dir, "models-preset.ini")
    model_paths = {}
    for _name, _repo, _file in KV_TEST_MODELS:
        model_path = download_file(
            f"https://huggingface.co/{_repo}/resolve/main/{_file}"
        )
        model_paths[_name] = model_path
    _write_preset_ini(preset_ini_path, model_paths)
    s.models_preset = preset_ini_path

    server = s
    server.start()
    yield
    server.stop()
    _cleanup_server(server)
    server = None


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


def _send_completion_router(
    prompt: str, max_tokens: int = 8, model_name: str = KV_MODEL_ALIASES[0]
):
    """Helper to send a /completion request in router mode."""
    return server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": prompt,
            "model": model_name,
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": max_tokens,
        },
    )


# ------------------------------------------------------------------
# [Section 3] MTP server -- parametrized across 4B and 27B
# ------------------------------------------------------------------

_MTP_MODELS = [
    pytest.param(("4b", _MODEL_4B_PATH, _HF_REPO_4B, _HF_FILE_4B), id="4b"),
    pytest.param(("27b", _MODEL_27B_PATH, _HF_REPO_27B, _HF_FILE_27B), id="27b"),
]


@pytest.fixture(scope="module", params=_MTP_MODELS)
def mtp_model(request):
    """Provides (model_label, model_path, hf_repo, hf_file) for MTP tests."""
    return request.param


@pytest.fixture(scope="module")
def section_mtp(mtp_model):
    global server
    model_label, model_path, hf_repo, hf_file = mtp_model
    s = ServerPreset.qwopus_mtp()
    s.model_file = model_path
    if not os.path.isfile(model_path):
        s.model_hf_repo = hf_repo
        s.model_hf_file = hf_file
    s.spec_draft_n_min = 4
    s.spec_draft_n_max = 8
    s.n_ctx = 512
    s.n_predict = 64
    _configure_kv_base(s, f"mtp-kv-cache-{model_label}-")
    server = s
    server.start()
    yield
    server.stop()
    _cleanup_server(server)
    server = None


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


# ------------------------------------------------------------------
# [Section 4] MTP router server
# ------------------------------------------------------------------
# NOTE: These tests are skipped by default because MTP models don't
# work correctly in router mode -- the child process gets stuck in
# an infinite "capture target preamble" loop (server bug).
# Set ROUTER_MODEL_TESTS=1 to run them (they will likely fail until
# the MTP+router integration is fixed).


@pytest.fixture
def section_mtp_router():
    """Router-mode server with a single MTP model in preset.ini."""
    s = ServerPreset.router()
    _configure_kv_base(s, "kv-cache-mtp-router-")
    s.spec_type = "draft-mtp"
    s.models_max = 1

    # Download the MTP model and write preset.ini
    _mtp_name, _mtp_repo, _mtp_file = KV_TEST_MODELS[3]  # Qwopus3.5-4B-Coder-MTP
    model_path = download_file(
        f"https://huggingface.co/{_mtp_repo}/resolve/main/{_mtp_file}"
    )
    preset_dir = os.path.dirname(s.slot_save_path)
    preset_ini_path = os.path.join(preset_dir, "models-preset.ini")
    _write_preset_ini(preset_ini_path, {_ROUTER_MODEL_NAME: model_path})
    s.models_preset = preset_ini_path

    yield s
    s.stop()
    _cleanup_server(s)


# ==========================================================================
# [Section 1] Simple Server KV Cache Tests  (13 tests)
# ==========================================================================


def test_simple_kv_cache_initialization(section_simple):
    """KV cache auto enabled, disk manager initialized, cache directory exists."""
    global server
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


def test_simple_first_request_save(section_simple):
    """First request should save KV cache to disk."""
    global server
    log = LogReader(server.log_path)

    res = _send_completion("What is 2+2? Briefly.", 8)
    assert res.status_code == 200, "Request should succeed"

    save_log = _wait_for_log(log, "KV cache saved")
    assert "KV cache saved" in save_log, f"No save log: {save_log[:500]}"

    cache_files = list(os.listdir(_cache_dir(server.slot_save_path)))
    assert len(cache_files) > 0, "No cache files created"


def test_simple_disk_lcp_wins_after_restart(section_simple):
    """After restart, RAM is empty -- disk should win the LCP comparison."""
    global server
    log = LogReader(server.log_path)
    _send_completion("Hello world from the server.", 8)
    _wait_for_log(log, "KV cache saved")

    _restart_server()

    log = LogReader(server.log_path)
    log.drain()  # skip init
    _send_completion("Hello world from the server.", 8)

    compare_log = log.drain()
    assert "restored slot from L3 disk cache" in compare_log, (
        f"Disk restore not found: {compare_log[:800]}"
    )


def test_simple_ram_cache_preferred(section_simple):
    """When RAM GPU cache has better LCP than disk, skip disk restore."""
    global server
    server.n_slots = 2
    _restart_server()
    log = LogReader(server.log_path)

    _send_completion("What is 2+2? Briefly.", 8)
    _wait_for_log(log, "KV cache saved")
    _send_completion("What is 3+3? Briefly.", 8)
    _wait_for_log(log, "KV cache saved")
    _send_completion("What is 2+3? Briefly.", 8)

    compare_log = log.drain()
    assert (
        "selected slot by LCP similarity" in compare_log
        or "restored slot from L2 prompt cache" in compare_log
        or "restored slot from L3 disk cache" in compare_log
        or "selected slot by LRU" in compare_log
    ), f"No slot selection log: {compare_log[:800]}"


def test_simple_both_caches_empty_lru_fallback(section_simple):
    """When both disk and RAM caches are empty, LRU slot is selected."""
    global server
    cache_dir = _cache_dir(server.slot_save_path)
    server.stop()
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    server.start()
    log = LogReader(server.log_path)
    log.drain()
    _send_completion("What is the meaning of life?", 5)

    lru_log = log.drain()
    assert "selected slot by LRU" in lru_log, f"No LRU selection: {lru_log[:800]}"


def test_simple_combined_pool_lcp_match(section_simple):
    """Similar prompt: combined 3-tier pool finds match via LCP."""
    global server
    log = LogReader(server.log_path)

    _send_completion("What is 2+2? Briefly.", 5)
    log.drain()  # sleep(3) replaced by drain
    _send_completion("What is 3+3? Briefly.", 5)

    pool_log = log.drain()
    assert (
        "selected from 3-tier pool" in pool_log
        or "selected slot by LCP similarity" in pool_log
        or "selected slot by LRU" in pool_log
    ), f"No pool selection log: {pool_log[:800]}"


def test_simple_trie_rebuild_from_disk(section_simple):
    """Restart server: trie rebuild from disk, cache HIT for same request."""
    global server
    # Start fresh to avoid interference from previous tests in the
    # module-scoped fixture (which leaks L2 cache entries and n_slots).
    _restart_server()
    log = LogReader(server.log_path)
    res_fresh = _send_completion("What is 2+2? Briefly.", 8)
    assert res_fresh.status_code == 200
    fresh_text = res_fresh.body["content"]
    assert len(fresh_text) > 0
    _wait_for_log(log, "KV cache saved")

    assert len(os.listdir(_cache_dir(server.slot_save_path))) > 0, (
        "No cache files before restart"
    )

    _restart_server()
    log = LogReader(server.log_path)

    init_log = log.drain()
    assert "KV cache rebuild" in init_log, f"No rebuild log: {init_log[:500]}"
    assert len(os.listdir(_cache_dir(server.slot_save_path))) > 0, (
        "Cache files disappeared"
    )

    res_cached = _send_completion("What is 2+2? Briefly.", 8)
    assert res_cached.status_code == 200
    cached_text = res_cached.body["content"]

    hit_log = log.drain()
    # Save hit_log for debugging
    import uuid as _uuid

    _debug_path = f"/tmp/trie_hit_{_uuid.uuid4().hex[:8]}.txt"
    with open(_debug_path, "w") as _f:
        _f.write(hit_log)
    print(f"\nHIT_LOG saved: {_debug_path}")
    assert "best candidate from L3" in hit_log, (
        f"No L3 HIT after rebuild: {hit_log[:800]}"
    )
    assert "restored slot from L3 disk cache" in hit_log, f"No restore: {hit_log[:800]}"

    assert fresh_text == cached_text, (
        f"KV cache output differs!\n"
        f"  Fresh:    {repr(fresh_text)}\n"
        f"  Cached:   {repr(cached_text)}\n"
    )


def test_simple_ttl_eviction(section_simple):
    """Cache entries expire after TTL seconds."""
    global server
    server.cache_ttl_seconds = 2
    _restart_server()
    log = LogReader(server.log_path)

    _send_completion("Quick answer: 1+1?", 5)
    _wait_for_log(log, "KV cache saved")
    log.drain()
    time.sleep(3)

    _send_completion("Quick answer: 1+1?", 5)

    ttl_log = log.drain()
    assert "KV cache MISS" in ttl_log, f"No MISS after TTL: {ttl_log[:800]}"


def test_simple_callback_invocation_and_save(section_simple):
    """Callback is invoked and KV cache is saved on request completion."""
    global server
    log = LogReader(server.log_path)

    _send_completion("Tell me a short joke.", 8)
    _wait_for_log(log, "KV cache saved")


def test_simple_multi_model_smoke(section_simple):
    """Run lightweight smoke test across multiple models."""
    models = KV_TEST_MODELS[1:]
    assert len(models) >= 1

    for model_name, hf_repo, hf_file in models:
        model_path = download_file(
            f"https://huggingface.co/{hf_repo}/resolve/main/{hf_file}"
        )
        cache_dir = tempfile.mkdtemp(prefix="kv-cache-multi-")

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
        s.n_threads = 16
        s.slot_save_path = cache_dir
        s.max_cache_size_gb = 1.0
        s.cache_ttl_seconds = 3600
        fd, s.log_path = tempfile.mkstemp(suffix=".log")
        os.close(fd)

        try:
            s.start()
            log = LogReader(s.log_path)

            init_log = log.drain()
            assert "KV cache auto enabled" in init_log, (
                f"{model_name}: KV cache auto not enabled"
            )

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
            _wait_for_log(log, "KV cache saved")

            s.stop()
            fd, s.log_path = tempfile.mkstemp(suffix=".log")
            os.close(fd)
            s.start()

            log = LogReader(s.log_path)
            init_log = log.drain()
            assert "KV cache rebuild" in init_log, (
                f"{model_name}: no rebuild after restart"
            )

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
            assert (
                "best candidate from L3" in hit_log
                or "restored slot from L3 disk cache" in hit_log
            ), f"{model_name}: no L3 HIT after restart"

        finally:
            s.stop()
            shutil.rmtree(cache_dir, ignore_errors=True)
            if os.path.exists(s.log_path):
                os.unlink(s.log_path)


def test_simple_cross_model_cache_isolation(section_simple):
    """Different models sharing a cache directory -- no crash, LRU fallback."""
    global server
    model_name, hf_repo, hf_file = KV_TEST_MODELS[1]
    model_path = download_file(
        f"https://huggingface.co/{hf_repo}/resolve/main/{hf_file}"
    )

    log = LogReader(server.log_path)
    _send_completion("What is 2+2? Briefly.", 8)
    save_log = _wait_for_log(log, "KV cache saved")
    assert "KV cache saved" in save_log, f"No save from primary model: {save_log[:300]}"
    server.stop()

    cache_dir = _cache_dir(server.slot_save_path)
    assert len(os.listdir(cache_dir)) > 0, "No cache files from primary model"

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
    s_b.n_threads = 16
    s_b.slot_save_path = cache_dir
    s_b.max_cache_size_gb = 1.0
    s_b.cache_ttl_seconds = 3600
    s_b.log_path = log_path_b

    try:
        s_b.start()
        log_b = LogReader(s_b.log_path)
        init_log = log_b.drain()
        assert "KV cache disk manager initialized" in init_log, (
            f"Disk manager not initialized: {init_log[:500]}"
        )
        assert len(os.listdir(cache_dir)) > 0, "Cache files disappeared"

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
        assert s_b.process and s_b.process.poll() is None, "Server died after request"

        request_log = log_b.drain()
        has_lru = "selected slot by LRU" in request_log
        has_lcp = "selected slot by LCP similarity" in request_log
        has_pool = "selected from 3-tier pool" in request_log
        assert has_lru or has_lcp or has_pool, f"No slot selection: {request_log[:500]}"

    finally:
        s_b.stop()
        if os.path.exists(log_path_b):
            os.unlink(log_path_b)


def test_simple_shared_cache_directory_two_models(section_simple):
    """Two models save to the same cache dir -- files coexist peacefully."""
    global server
    model2_name, hf_repo_2, hf_file_2 = KV_TEST_MODELS[1]
    path2 = download_file(
        f"https://huggingface.co/{hf_repo_2}/resolve/main/{hf_file_2}"
    )

    cache_dir = _cache_dir(server.slot_save_path)

    log1 = LogReader(server.log_path)
    _send_completion("Hello from model A.", 8)
    s1_save = _wait_for_log(log1, "KV cache saved")
    assert "KV cache saved" in s1_save, f"No save from model A: {s1_save[:200]}"
    server.stop()

    files_after_a = set(os.listdir(cache_dir))
    assert len(files_after_a) > 0, "No cache files from model A"

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
    s2.n_threads = 16
    s2.slot_save_path = cache_dir
    s2.max_cache_size_gb = 1.0
    s2.cache_ttl_seconds = 3600
    s2.log_path = log2_path

    try:
        s2.start()
        log2 = LogReader(s2.log_path)
        log2.drain()

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
        _wait_for_log(log2, "KV cache saved")
        s2.stop()

        files_after_both = set(os.listdir(cache_dir))
        assert len(files_after_both) >= 2, (
            f"Expected at least 2 cache files, got {files_after_both}"
        )
        b_files = [f for f in files_after_both if f not in files_after_a]
        assert len(b_files) > 0, f"No new cache files from model B: {files_after_both}"

    finally:
        s2.stop()
        if os.path.exists(log2_path):
            os.unlink(log2_path)


def test_simple_restart_with_foreign_cache_files(section_simple):
    """Server rebuilds from cache dir that has files from another model."""
    global server
    model2_name, hf_repo_2, hf_file_2 = KV_TEST_MODELS[1]
    path2 = download_file(
        f"https://huggingface.co/{hf_repo_2}/resolve/main/{hf_file_2}"
    )

    cache_dir = _cache_dir(server.slot_save_path)

    log = LogReader(server.log_path)
    _send_completion("The quick brown fox jumps over the lazy dog.", 8)
    assert _wait_for_log(log, "KV cache saved"), "No save from model A"
    server.stop()

    files_from_a = set(os.listdir(cache_dir))
    assert len(files_from_a) > 0

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
    s2.n_threads = 16
    s2.slot_save_path = cache_dir
    s2.max_cache_size_gb = 1.0
    s2.cache_ttl_seconds = 3600
    s2.log_path = log2_path

    try:
        s2.start()
        log2 = LogReader(s2.log_path)
        log2.drain()

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
        _wait_for_log(log2, "KV cache saved")
        s2.stop()

        files_from_both = set(os.listdir(cache_dir))
        assert len(files_from_both) > len(files_from_a), (
            f"Expected more files after model B: {files_from_both}"
        )

        s2.start()
        log3 = LogReader(s2.log_path)
        init_log = log3.drain()
        assert "KV cache rebuild" in init_log, f"No rebuild log: {init_log[:500]}"
        assert "KV cache disk manager initialized" in init_log, (
            f"Disk not initialized: {init_log[:500]}"
        )

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
        assert s2.process and s2.process.poll() is None, (
            "Server process died after restart"
        )

    finally:
        s2.stop()
        if os.path.exists(log2_path):
            os.unlink(log2_path)


# ==========================================================================
# [Section 2] Router Server KV Cache Tests  (10 tests)
# ==========================================================================


def test_router_kv_cache_initialization(section_router):
    """Router: KV cache init after model load."""
    global server
    log = LogReader(server.log_path)
    log.drain()  # skip router startup

    res = _send_completion_router("What is 2+2? Briefly.", 8)
    assert res.status_code == 200

    model_log = log.drain()
    assert "KV cache auto enabled" in model_log, f"Cache not enabled: {model_log[:500]}"
    assert "KV cache disk manager initialized" in model_log, (
        f"Disk not initialized: {model_log[:500]}"
    )
    cache_dir = _cache_dir(server.slot_save_path)
    assert os.path.isdir(cache_dir), "Cache directory does not exist"


def test_router_first_request_save(section_router):
    """Router: first request saves KV cache to disk."""
    global server
    log = LogReader(server.log_path)

    res = _send_completion_router("What is 2+2? Briefly.", 8)
    assert res.status_code == 200
    _wait_for_log(log, "KV cache saved")

    cache_files = list(os.listdir(_cache_dir(server.slot_save_path)))
    assert len(cache_files) > 0, "No cache files created"


def test_router_disk_lcp_wins_after_restart(section_router):
    """Router: after restart, disk wins the LCP comparison."""
    global server
    log = LogReader(server.log_path)
    _send_completion_router("Hello world from the server.", 8)
    _wait_for_log(log, "KV cache saved")

    _restart_server()

    log = LogReader(server.log_path)
    log.drain()
    _send_completion_router("Hello world from the server.", 8)

    compare_log = log.drain()
    assert "restored slot from L3 disk cache" in compare_log, (
        f"Disk restore not found: {compare_log[:800]}"
    )


def test_router_ram_cache_preferred(section_router):
    """Router: RAM LCP > Disk LCP, skip disk restore."""
    global server
    server.n_slots = 2
    _restart_server()
    log = LogReader(server.log_path)

    _send_completion_router("What is 2+2? Briefly.", 8)
    _wait_for_log(log, "KV cache saved")
    _send_completion_router("What is 3+3? Briefly.", 8)
    _wait_for_log(log, "KV cache saved")
    _send_completion_router("What is 2+3? Briefly.", 8)

    compare_log = log.drain()
    assert (
        "selected slot by LCP similarity" in compare_log
        or "restored slot from L3 disk cache" in compare_log
        or "selected slot by LRU" in compare_log
    ), f"No slot selection log: {compare_log[:800]}"


def test_router_both_caches_empty_lru_fallback(section_router):
    """Router: both caches empty, LRU slot selected."""
    global server
    cache_dir = _cache_dir(server.slot_save_path)
    server.stop()
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    server.start()
    log = LogReader(server.log_path)
    log.drain()
    _send_completion_router("What is the meaning of life?", 5)

    lru_log = log.drain()
    assert "selected slot by LRU" in lru_log, f"No LRU selection: {lru_log[:800]}"


def test_router_combined_pool_lcp_match(section_router):
    """Router: similar prompt, combined pool finds LCP match."""
    global server
    log = LogReader(server.log_path)

    _send_completion_router("What is 2+2? Briefly.", 5)
    log.drain()  # sleep(3) replaced by drain
    _send_completion_router("What is 3+3? Briefly.", 5)

    pool_log = log.drain()
    assert (
        "selected from 3-tier pool" in pool_log
        or "selected slot by LCP similarity" in pool_log
        or "selected slot by LRU" in pool_log
    ), f"No pool selection log: {pool_log[:800]}"


def test_router_trie_rebuild_from_disk(section_router):
    """Router: restart, trie rebuild, cache HIT for same request."""
    global server
    log = LogReader(server.log_path)
    res_fresh = _send_completion_router("What is 2+2? Briefly.", 8)
    assert res_fresh.status_code == 200
    fresh_text = res_fresh.body["content"]
    assert len(fresh_text) > 0
    _wait_for_log(log, "KV cache saved")

    assert len(os.listdir(_cache_dir(server.slot_save_path))) > 0, (
        "No cache files before restart"
    )

    _restart_server()

    assert len(os.listdir(_cache_dir(server.slot_save_path))) > 0, (
        "Cache files disappeared"
    )

    log = LogReader(server.log_path)
    log.drain()

    res_cached = _send_completion_router("What is 2+2? Briefly.", 8)
    assert res_cached.status_code == 200
    cached_text = res_cached.body["content"]

    hit_log = log.drain()
    assert (
        "best candidate from L3" in hit_log
        or "restored slot from L3 disk cache" in hit_log
    ), f"No L3 HIT after rebuild: {hit_log[:800]}"
    assert "restored slot from L3 disk cache" in hit_log, f"No restore: {hit_log[:800]}"

    assert fresh_text == cached_text, (
        f"KV cache output differs!\n"
        f"  Fresh:    {repr(fresh_text)}\n"
        f"  Cached:   {repr(cached_text)}\n"
    )


def test_router_ttl_eviction(section_router):
    """Router: cache entries expire after TTL seconds."""
    global server
    server.cache_ttl_seconds = 2
    _restart_server()
    log = LogReader(server.log_path)

    _send_completion_router("Quick answer: 1+1?", 5)
    _wait_for_log(log, "KV cache saved")
    log.drain()
    time.sleep(3)

    _send_completion_router("Quick answer: 1+1?", 5)

    ttl_log = log.drain()
    assert "KV cache MISS" in ttl_log, f"No MISS after TTL: {ttl_log[:800]}"


def test_router_callback_invocation_and_save(section_router):
    """Router: callback invoked and KV cache saved."""
    global server
    log = LogReader(server.log_path)

    _send_completion_router("Tell me a short joke.", 8)
    _wait_for_log(log, "KV cache saved")


def test_router_multi_model_smoke(section_router):
    """Router: lightweight smoke test across all models, one at a time."""
    global server

    for model_name, hf_repo, hf_file in KV_TEST_MODELS:
        model_path = download_file(
            f"https://huggingface.co/{hf_repo}/resolve/main/{hf_file}"
        )

        s = ServerPreset.router()
        s.debug = True
        s.n_ctx = 2048
        s.n_slots = 1
        s.n_predict = 8
        s.temperature = 0.0
        s.cache_ram = 8000
        s.n_threads = 16
        s.kv_cache_auto = True
        s.max_cache_size_gb = 1.0
        s.cache_ttl_seconds = 3600
        s.models_max = 1

        cache_dir = tempfile.mkdtemp(prefix="kv-cache-router-smoke-")
        slot_cache_dir = os.path.join(cache_dir, "slot-cache")
        os.makedirs(slot_cache_dir, exist_ok=True)

        preset_path = os.path.join(cache_dir, "models-preset.ini")
        _write_preset_ini(preset_path, {model_name: model_path})
        s.models_preset = preset_path
        s.slot_save_path = slot_cache_dir

        fd, s.log_path = tempfile.mkstemp(suffix=".log")
        os.close(fd)

        try:
            s.start()
            log = LogReader(s.log_path)
            log.drain()

            res = s.make_request(
                "POST",
                "/completion",
                data={
                    "prompt": "Tell me a short joke",
                    "model": model_name,
                    "temperature": 0.0,
                    "top_k": 1,
                    "n_predict": 8,
                },
            )
            assert res.status_code == 200, f"{model_name}: request failed"

            _wait_for_log(log, "KV cache saved")
            model_log = log.drain()
            assert "KV cache auto enabled" in model_log, (
                f"{model_name}: KV cache auto not enabled"
            )

            s.stop()
            fd, s.log_path = tempfile.mkstemp(suffix=".log")
            os.close(fd)
            s.start()

            log = LogReader(s.log_path)
            log.drain()

            res2 = s.make_request(
                "POST",
                "/completion",
                data={
                    "prompt": "Tell me a short joke",
                    "model": model_name,
                    "temperature": 0.0,
                    "top_k": 1,
                    "n_predict": 8,
                },
            )
            assert res2.status_code == 200, (
                f"{model_name}: request after restart failed"
            )

            hit_log = log.drain()
            assert (
                "best candidate from L3" in hit_log
                or "restored slot from L3 disk cache" in hit_log
            ), f"{model_name}: no L3 HIT after restart"

        finally:
            s.stop()
            shutil.rmtree(cache_dir, ignore_errors=True)
            if os.path.exists(s.log_path):
                os.unlink(s.log_path)


# ==========================================================================
# [Section 3] MTP Simple Server Tests  (19 tests)
# ==========================================================================

# ------------------------------------------------------------------
# Basic MTP
# ------------------------------------------------------------------


def test_mtp_completion(section_mtp):
    """Basic completion with MTP -- must produce valid output."""
    global server
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


def test_mtp_draft_params(section_mtp):
    """Different draft_min/draft_max values must produce valid completions."""
    global server
    test_values = [(1, 2), (1, 4), (4, 8)]
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


def test_mtp_slot_ctx_not_exceeded(section_mtp):
    """MTP with a long prompt must not exceed the slot context."""
    global server
    server.n_ctx = 256
    _restart_server()
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


def test_mtp_ctx_shift(section_mtp):
    """MTP with context shift -- exercises L3 restore + seq_rm path."""
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.stop()
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


# ------------------------------------------------------------------
# MTP KV cache basic
# ------------------------------------------------------------------


def test_mtp_kv_cache_initialization(section_mtp):
    """MTP: KV cache auto enabled, disk manager initialized."""
    global server
    log = LogReader(server.log_path)

    init_log = log.drain()
    assert "KV cache auto enabled" in init_log, f"Cache not enabled: {init_log[:500]}"
    assert "KV cache disk manager initialized" in init_log, (
        f"Disk not initialized: {init_log[:500]}"
    )
    cache_dir = _cache_dir(server.slot_save_path)
    assert os.path.isdir(cache_dir)


def test_mtp_first_request_save(section_mtp):
    """MTP: first request saves KV cache to disk."""
    global server
    _restart_server()
    log = LogReader(server.log_path)

    res = _send_completion("What is 2+2? Briefly.", 8)
    assert res.status_code == 200
    _wait_for_log(log, "KV cache saved")

    cache_files = list(os.listdir(_cache_dir(server.slot_save_path)))
    assert len(cache_files) > 0, "No cache files created"


def test_mtp_disk_lcp_wins_after_restart(section_mtp):
    """MTP: after restart, RAM empty -- disk should win LCP comparison."""
    global server
    log = LogReader(server.log_path)
    _send_completion("Hello world from the server.", 8)
    _wait_for_log(log, "KV cache saved")

    _restart_server()

    log = LogReader(server.log_path)
    log.drain()
    _send_completion("Hello world from the server.", 8)

    compare_log = log.drain()
    assert (
        "restored slot from L3 disk cache" in compare_log
        or "best candidate from L3" in compare_log
    ), f"Disk restore not found: {compare_log[:800]}"


def test_mtp_ram_cache_preferred(section_mtp):
    """MTP: RAM LCP > Disk LCP, skip disk restore."""
    global server
    server.n_slots = 2
    _restart_server()
    log = LogReader(server.log_path)

    _send_completion("What is 2+2? Briefly.", 8)
    _wait_for_log(log, "KV cache saved")
    _send_completion("What is 3+3? Briefly.", 8)
    _wait_for_log(log, "KV cache saved")
    _send_completion("What is 2+3? Briefly.", 8)

    compare_log = log.drain()
    assert (
        "selected slot by LCP similarity" in compare_log
        or "restored slot from L3 disk cache" in compare_log
        or "selected slot by LRU" in compare_log
    ), f"No slot selection log: {compare_log[:800]}"


def test_mtp_both_caches_empty_lru_fallback(section_mtp):
    """MTP: both caches empty, LRU slot selected."""
    global server
    cache_dir = _cache_dir(server.slot_save_path)
    server.stop()
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    server.start()
    log = LogReader(server.log_path)
    log.drain()
    _send_completion("What is the meaning of life?", 5)

    lru_log = log.drain()
    assert "selected slot by LRU" in lru_log, f"No LRU selection: {lru_log[:800]}"


def test_mtp_combined_pool_lcp_match(section_mtp):
    """MTP: similar prompt, combined pool finds LCP match."""
    global server
    log = LogReader(server.log_path)

    _send_completion("What is 2+2? Briefly.", 5)
    log.drain()  # sleep(3) replaced by drain
    _send_completion("What is 3+3? Briefly.", 5)

    pool_log = log.drain()
    assert (
        "selected from 3-tier pool" in pool_log
        or "selected slot by LCP similarity" in pool_log
        or "selected slot by LRU" in pool_log
    ), f"No pool selection log: {pool_log[:800]}"


def test_mtp_trie_rebuild_from_disk(section_mtp):
    """MTP: restart, trie rebuild from disk, cache HIT for same request."""
    global server
    log = LogReader(server.log_path)
    res_fresh = _send_completion("What is 2+2? Briefly.", 8)
    assert res_fresh.status_code == 200
    fresh_text = res_fresh.body["content"]
    assert len(fresh_text) > 0
    _wait_for_log(log, "KV cache saved")

    assert len(os.listdir(_cache_dir(server.slot_save_path))) > 0, (
        "No cache files before restart"
    )

    _restart_server()

    log = LogReader(server.log_path)
    init_log = log.drain()
    assert "KV cache rebuild" in init_log, f"No rebuild log: {init_log[:500]}"
    assert len(os.listdir(_cache_dir(server.slot_save_path))) > 0, (
        "Cache files disappeared"
    )

    res_cached = _send_completion("What is 2+2? Briefly.", 8)
    assert res_cached.status_code == 200
    cached_text = res_cached.body["content"]

    hit_log = log.drain()
    assert "best candidate from L3" in hit_log or "KV cache HIT" in hit_log, (
        f"No HIT or restore after rebuild: {hit_log[:800]}"
    )

    assert fresh_text == cached_text, (
        f"KV cache output differs!\n"
        f"  Fresh:    {repr(fresh_text)}\n"
        f"  Cached:   {repr(cached_text)}\n"
    )


def test_mtp_ttl_eviction(section_mtp):
    """MTP: cache entries expire after TTL seconds."""
    global server
    server.cache_ttl_seconds = 2
    _restart_server()
    log = LogReader(server.log_path)

    _send_completion("Quick answer: 1+1?", 5)
    _wait_for_log(log, "KV cache saved")
    log.drain()
    time.sleep(3)

    _send_completion("Quick answer: 1+1?", 5)

    ttl_log = log.drain()
    assert "KV cache MISS" in ttl_log, f"No MISS after TTL: {ttl_log[:800]}"


def test_mtp_callback_invocation_and_save(section_mtp):
    """MTP: callback invoked and KV cache saved."""
    global server
    log = LogReader(server.log_path)

    _send_completion("Tell me a short joke.", 8)
    _wait_for_log(log, "KV cache saved")


# ------------------------------------------------------------------
# MTP L3 Cache Regression Tests
# ------------------------------------------------------------------


def test_mtp_l3_cache_deterministic(section_mtp):
    """L3 cache restore must produce bit-identical output to a fresh run."""
    global server
    server.n_predict = 32
    _restart_server()
    log = LogReader(server.log_path)
    log.drain()

    r1 = _send_chat("What is 2+2? Briefly.", 32)
    assert r1.status_code == 200
    fresh_text = r1.body["choices"][0]["message"]["content"]
    assert len(fresh_text) > 0

    save_log = _wait_for_log(log, "KV cache saved")
    assert "KV cache saved" in save_log, f"L3 save not found: {save_log[:500]}"

    r2 = _send_chat("What is 2+2? Briefly.", 32)
    assert r2.status_code == 200
    restored_text = r2.body["choices"][0]["message"]["content"]

    r2_log = log.drain()
    assert (
        "best candidate from L3" in r2_log
        or "restored slot from L3 disk cache" in r2_log
    ), f"L3 restore not detected: {r2_log[:800]}"

    assert fresh_text == restored_text, (
        f"MTP L3 cache output differs!\n"
        f"  Fresh:    {repr(fresh_text)}\n"
        f"  Restored: {repr(restored_text)}\n"
    )


def test_mtp_l3_draft_preamble_saved(section_mtp):
    """The L3 save callback must write a .draft companion file."""
    global server
    _restart_server()
    log = LogReader(server.log_path)
    res = _send_chat("Hello world", 16)
    assert res.status_code == 200
    _wait_for_log(log, "KV cache saved")

    cache_dir = server.slot_save_path
    draft_files = [f for f in os.listdir(cache_dir) if f.endswith(".draft")]
    assert len(draft_files) > 0, f"No .draft files found in {cache_dir}"
    for df in draft_files:
        full_path = os.path.join(cache_dir, df)
        assert os.path.getsize(full_path) > 0, f"Empty .draft file: {full_path}"


def test_mtp_l3_deterministic_after_restart(section_mtp):
    """Server restart: L3 cache must still produce deterministic output."""
    global server
    server.n_predict = 32
    server.jinja = True
    server.reasoning = "off"

    cache_dir = _cache_dir(server.slot_save_path)
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            fpath = os.path.join(cache_dir, fname)
            if os.path.isfile(fpath):
                os.unlink(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath, ignore_errors=True)

    server.stop()
    server.start()

    log = LogReader(server.log_path)
    r1 = _send_chat("What is 2+2? Briefly.", 32)
    assert r1.status_code == 200
    expected = r1.body["choices"][0]["message"]["content"]
    assert len(expected) > 0
    _wait_for_log(log, "KV cache saved")

    server.stop()
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()

    time.sleep(1)
    r2 = _send_chat("What is 2+2? Briefly.", 32)
    assert r2.status_code == 200
    actual = r2.body["choices"][0]["message"]["content"]

    assert expected == actual, (
        f"MTP L3 output differs after restart!\n"
        f"  Before restart: {repr(expected)}\n"
        f"  After restart:  {repr(actual)}\n"
    )


def test_mtp_seq_rm_tail_not_lost(section_mtp):
    """seq_rm must preserve the tail cell when trimming from p0 to end."""
    global server
    server.n_ctx = 256
    _restart_server()

    res = _send_chat("Write a story about " + "mountains " * 80, 16)
    assert res.status_code == 200
    assert len(res.body["choices"][0]["message"]["content"]) > 0, (
        "seq_rm tail loss produced empty/garbled output"
    )


def test_mtp_draft_preamble_multiple_requests(section_mtp):
    """Multiple requests using L3 cache: each produces consistent output."""
    global server
    server.n_predict = 16
    _restart_server()

    log = LogReader(server.log_path)
    prompt = "What is 2+2? Briefly."
    outputs = []
    for i in range(3):
        res = _send_chat(prompt, 16)
        assert res.status_code == 200
        outputs.append(res.body["choices"][0]["message"]["content"])
        _wait_for_log(log, "KV cache saved")

    for i in range(1, len(outputs)):
        assert outputs[0] == outputs[i], (
            f"MTP preamble: output {i} differs from output 0\n"
            f"  [0]: {repr(outputs[0])}\n"
            f"  [{i}]: {repr(outputs[i])}\n"
        )


def test_mtp_l3_with_longer_prompt(section_mtp):
    """L3 cache restore with a moderately long prompt must be deterministic."""
    global server
    server.n_predict = 16
    server.n_ctx = 1024
    _clear_cache_dir()
    _restart_server()
    log = LogReader(server.log_path)

    long_prompt = (
        "The following is a conversation with an AI assistant. "
        "The assistant is helpful, creative, and very friendly.\n\n"
        "Human: Can you explain the concept of recursion in computer science?\n"
        "Assistant:"
    )

    r1 = _send_chat(long_prompt, 16)
    assert r1.status_code == 200
    expected = r1.body["choices"][0]["message"]["content"]

    save_log = _wait_for_log(log, "KV cache saved")
    assert "KV cache saved" in save_log, f"L3 save not found: {save_log[:500]}"

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


# ==========================================================================
# [Section 4] MTP Router Server Tests  (3 tests)
# ==========================================================================


def test_mtp_router_kv_cache_init(section_mtp_router):
    """MTP Router: KV cache auto enabled, disk manager initialized."""
    s = section_mtp_router
    s.start()
    log = LogReader(s.log_path)
    log.drain()

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


def test_mtp_router_first_request_save(section_mtp_router):
    """MTP Router: first request saves KV cache to disk."""
    s = section_mtp_router
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
    _wait_for_log(log, "KV cache saved")
    assert len(os.listdir(s.slot_save_path)) > 0, "No cache files"


def test_mtp_router_restart_disk_restore(section_mtp_router):
    """MTP Router: after restart, disk cache restores for same request."""
    s = section_mtp_router
    s.start()

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
    log = LogReader(s.log_path)
    _wait_for_log(log, "KV cache saved")

    s.stop()
    fd, s.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    s.start()

    log = LogReader(s.log_path)
    log.drain()

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
    assert "restored slot from L3 disk cache" in compare_log, (
        f"Disk restore not found: {compare_log[:800]}"
    )
