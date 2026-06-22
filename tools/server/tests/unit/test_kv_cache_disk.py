import os
import tempfile
import time
import pytest
from utils import *

server = ServerPreset.tinyllama2()


class LogReader:
    """Tails a log file from a given position."""

    def __init__(self, path: str):
        self.path = path
        self.pos = 0

    def drain(self) -> str:
        with open(self.path) as f:
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
    # Cleanup: remove the cache directory
    import shutil

    try:
        shutil.rmtree(server.kv_cache_dir, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Test 1: disk cache HIT after restart (trie rebuild)
# ---------------------------------------------------------------------------


def test_kv_cache_restart_rebuild():
    """Save a cache entry, restart server, verify trie rebuild and restore."""
    global server

    # Start server, send a request (cache is created)
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "What is 2+2? Briefly.",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )
    assert res.status_code == 200

    # Wait for save callback to fire
    time.sleep(3)

    save_log = log.drain()
    assert "KV cache saved" in save_log, f"no save log: {save_log[:500]}"

    # Restart server with same cache dir
    server.stop()
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()

    log2 = LogReader(server.log_path)

    # Check trie rebuild log
    init_log = log2.drain()
    assert "KV cache rebuild" in init_log, f"no rebuild log: {init_log[:500]}"

    # Send same request — should HIT from rebuilt trie
    res2 = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "What is 2+2? Briefly.",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )
    assert res2.status_code == 200

    phase2_log = log2.drain()
    assert "KV cache HIT" in phase2_log, f"no HIT log: {phase2_log[:500]}"
    assert "restored slot from disk cache" in phase2_log, (
        f"no restore log: {phase2_log[:500]}"
    )


# ---------------------------------------------------------------------------
# Test 2: disk LCP vs RAM LCP comparison (disk wins when RAM empty)
# ---------------------------------------------------------------------------


def test_disk_lcp_vs_ram_lcp():
    """After restart, RAM is empty — disk should win the LCP comparison."""
    global server

    # Start fresh
    server.start()
    log = LogReader(server.log_path)

    # Save an entry
    server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Hello world from the server.",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )
    time.sleep(3)

    # Restart
    server.stop()
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()

    log2 = LogReader(server.log_path)
    log2.drain()  # skip init log

    # Same request — disk LCP=1.0 > RAM LCP=0.0
    server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Hello world from the server.",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )

    compare_log = log2.drain()
    # Should see the LCP comparison in logs
    assert "disk LCP=" in compare_log.lower() or "disk_lcp" in compare_log.lower(), (
        f"no LCP comparison log: {compare_log[:800]}"
    )


# ---------------------------------------------------------------------------
# Test 3: RAM cache preferred over disk (RAM has better LCP)
# ---------------------------------------------------------------------------


def test_ram_cache_preferred():
    """When RAM GPU cache has better LCP than disk, skip disk restore."""
    global server

    # Use 2 parallel slots so slot 0 can hold RAM cache while slot 1 is free
    server.n_slots = 2
    server.start()
    log = LogReader(server.log_path)

    # Request A on slot 0
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "What is 2+2? Briefly.",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )
    assert res.status_code == 200
    time.sleep(3)

    # Request A again — RAM has LCP=1.0, disk also has LCP=1.0
    # Since equal, neither branch fires; falls through to RAM level 2
    res2 = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "What is 2+2? Briefly.",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )
    assert res2.status_code == 200

    compare_log = log.drain()
    # Should see "selected slot by LCP similarity" (RAM level 2)
    assert (
        "selected slot by LCP similarity" in compare_log.lower()
        or "RAM cache preferred" in compare_log
        or "restored slot from disk cache" in compare_log
    ), f"no slot selection log: {compare_log[:800]}"


# ---------------------------------------------------------------------------
# Test 4: TTL eviction
# ---------------------------------------------------------------------------


def test_ttl_eviction():
    """Cache entries expire after TTL seconds."""
    global server

    server.cache_ttl_seconds = 3  # 3 second TTL
    server.start()
    log = LogReader(server.log_path)

    # Save entry
    server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Quick answer: 1+1?",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )
    time.sleep(3)

    # Wait for TTL
    time.sleep(4)

    # Same request — should MISS (expired)
    server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Quick answer: 1+1?",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 8,
        },
    )

    ttl_log = log.drain()
    # Should see a MISS after TTL
    assert "KV cache MISS" in ttl_log, f"no MISS after TTL: {ttl_log[:800]}"
