#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// Per-tier timing statistics
struct kv_cache_timing {
    int64_t  min = INT64_MAX;
    int64_t  max = 0;
    int64_t  sum = 0;
    uint64_t n   = 0;

    void record(int64_t us) {
        if (us < min) {
            min = us;
        }
        if (us > max) {
            max = us;
        }
        sum += us;
        ++n;
    }

    void reset() {
        min = INT64_MAX;
        max = 0;
        sum = 0;
        n   = 0;
    }

    double avg_us() const { return n > 0 ? (double) sum / n : 0.0; }
};

// Combined metrics collector for L1/L2/L3 KV cache + zstd compression
//
// Usage:
//   1. Call on_* methods at each cache operation site.
//   2. Call try_log(now_us) periodically — it logs and resets when the
//      interval elapses.
struct kv_cache_metrics_collector {
    // --- Per-tier counters ---
    struct {
        uint64_t        hits   = 0;
        uint64_t        misses = 0;
        kv_cache_timing find;  // time to find matching entries
        kv_cache_timing load;  // time to load/restore (L2/L3 only)
        kv_cache_timing save;  // time to save to tier (L2/L3 only)
    } l1, l2, l3;

    // --- Disk-specific ---
    uint64_t evictions_ttl   = 0;
    uint64_t evictions_lru   = 0;
    uint64_t saves_skipped   = 0;
    uint64_t restores_failed = 0;

    // --- Compression ---
    struct {
        uint64_t        uncompressed_bytes = 0;
        uint64_t        compressed_bytes   = 0;
        kv_cache_timing compress;    // time to compress
        kv_cache_timing decompress;  // time to decompress

        double ratio() const { return compressed_bytes > 0 ? (double) uncompressed_bytes / compressed_bytes : 0.0; }
    } compression;

    // --- Control ---
    int64_t log_interval_us  = 300 * 1000000;  // 300 seconds
    int64_t last_log_time_us = 0;

    // ---- Event receivers ----

    void on_l1_find(int64_t duration_us, bool hit) {
        if (hit) {
            l1.hits++;
        } else {
            l1.misses++;
        }
        l1.find.record(duration_us);
    }

    void on_l2_find(int64_t duration_us, bool hit) {
        if (hit) {
            l2.hits++;
        } else {
            l2.misses++;
        }
        l2.find.record(duration_us);
    }

    void on_l2_load(int64_t duration_us, bool success) {
        l2.load.record(duration_us);
        if (!success) {
            l2.misses++;
        }
    }

    void on_l2_save(int64_t duration_us, bool success) {
        l2.save.record(duration_us);
        if (!success) {
            saves_skipped++;
        }
    }

    void on_l3_find(int64_t duration_us, bool hit) {
        if (hit) {
            l3.hits++;
        } else {
            l3.misses++;
        }
        l3.find.record(duration_us);
    }

    void on_l3_restore(int64_t duration_us, bool success) {
        l3.load.record(duration_us);
        if (!success) {
            restores_failed++;
        }
    }

    void on_l3_save(int64_t duration_us, size_t uncompressed_size, size_t compressed_size, bool success) {
        l3.save.record(duration_us);
        if (uncompressed_size > 0 && compressed_size > 0) {
            compression.uncompressed_bytes += uncompressed_size;
            compression.compressed_bytes += compressed_size;
        }
        if (!success) {
            saves_skipped++;
        }
    }

    void on_compress(int64_t duration_us) { compression.compress.record(duration_us); }

    void on_decompress(int64_t duration_us) { compression.decompress.record(duration_us); }

    void on_l3_eviction(bool ttl) {
        if (ttl) {
            evictions_ttl++;
        } else {
            evictions_lru++;
        }
    }

    // ---- Periodic logging ----

    // Call on every slot allocation attempt. Logs a summary line when
    // `log_interval_us` has elapsed since the last log, then resets counters.
    bool try_log(int64_t now_us);

    // Build a log line without side effects
    std::string format_summary() const;

    // Reset all counters (does not touch log interval / last-log timestamp)
    void reset_counters();
};
