#include "kv-cache-metrics-collector.h"

#include "log.h"

#include <cinttypes>
#include <string>

static void append_timing(std::string & out, const char * label, const kv_cache_timing & t) {
    if (t.n == 0) {
        return;
    }
    char buf[128];
    snprintf(buf, sizeof(buf), " %s=[%.1f/%.1f/%.1f]ms", label, t.min / 1000.0, t.avg_us() / 1000.0, t.max / 1000.0);
    out += buf;
}

bool kv_cache_metrics_collector::try_log(int64_t now_us) {
    if (last_log_time_us == 0) {
        last_log_time_us = now_us;
        return false;
    }
    if (now_us - last_log_time_us < log_interval_us) {
        return false;
    }

    LOG_INF("%s", format_summary().c_str());

    last_log_time_us = now_us;
    reset_counters();
    llama_kv_cache_compression_stats_reset();
    return true;
}

std::string kv_cache_metrics_collector::format_summary() const {
    std::string out = "kv-cache metrics [collect]:";

    // L1
    {
        char buf[128];
        snprintf(buf, sizeof(buf), " L1: hits=%" PRIu64 " misses=%" PRIu64, l1.hits, l1.misses);
        out += buf;
        append_timing(out, "find", l1.find);
    }

    // L2
    {
        char buf[128];
        snprintf(buf, sizeof(buf), " | L2: hits=%" PRIu64 " misses=%" PRIu64, l2.hits, l2.misses);
        out += buf;
        append_timing(out, "find", l2.find);
        append_timing(out, "load", l2.load);
        append_timing(out, "save", l2.save);
    }

    // L3
    {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 " | L3: hits=%" PRIu64 " misses=%" PRIu64 " evict_ttl=%" PRIu64 " evict_lru=%" PRIu64 " skip=%" PRIu64
                 " fail=%" PRIu64,
                 l3.hits, l3.misses, evictions_ttl, evictions_lru, saves_skipped, restores_failed);
        out += buf;
        append_timing(out, "find", l3.find);
        append_timing(out, "restore", l3.load);
        append_timing(out, "save", l3.save);
    }

    // Compression (zstd global stats)
    {
        int64_t  comp_us = 0, decom_us = 0;
        uint64_t unc_bytes = 0, com_bytes = 0, comp_n = 0, decom_n = 0;
        llama_kv_cache_compression_stats_get(&comp_us, &decom_us, &unc_bytes, &com_bytes, &comp_n, &decom_n);
        if (comp_n > 0 || decom_n > 0) {
            char         buf[256];
            const double ratio = com_bytes > 0 ? (double) unc_bytes / com_bytes : 0.0;
            const double c_avg = comp_n > 0 ? (double) comp_us / comp_n / 1000.0 : 0.0;
            const double d_avg = decom_n > 0 ? (double) decom_us / decom_n / 1000.0 : 0.0;
            snprintf(buf, sizeof(buf),
                     " | comp: ratio=%.2fx unc=%" PRIu64 " comp=%" PRIu64
                     " enc=[avg=%.1f]ms dec=[avg=%.1f]ms cnt=%" PRIu64 "/%" PRIu64,
                     ratio, unc_bytes, com_bytes, c_avg, d_avg, comp_n, decom_n);
            out += buf;
        }
    }

    out += '\n';
    return out;
}

void kv_cache_metrics_collector::reset_counters() {
    l1              = {};
    l2              = {};
    l3              = {};
    evictions_ttl   = 0;
    evictions_lru   = 0;
    saves_skipped   = 0;
    restores_failed = 0;
    compression     = {};
}
