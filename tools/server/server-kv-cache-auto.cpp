#include "server-kv-cache-auto.h"

#include "common.h"
#include "kv-cache-disk-manager.h"
#include "kv-cache-metrics-collector.h"
#include "log.h"
#include "server-common.h"

#include <cinttypes>
#include <filesystem>

std::unique_ptr<kv_cache_disk_manager> init_kv_cache_disk_manager(bool                kv_cache_auto,
                                                                  float               max_cache_size_gb,
                                                                  int64_t             cache_ttl_seconds,
                                                                  float               slot_prompt_similarity,
                                                                  const std::string & slot_save_path,
                                                                  bool                compress_kv_cache,
                                                                  int                 compress_kv_cache_level,
                                                                  bool                compress_kv_cache_learn) {
    if (!kv_cache_auto) {
        return nullptr;
    }

    if (slot_save_path.empty()) {
        SRV_WRN("%s\n", "KV cache auto enabled but slot-save-path is empty");
        return nullptr;
    }

    std::string cache_dir = slot_save_path;
    std::filesystem::create_directories(cache_dir);

    auto mgr = std::make_unique<kv_cache_disk_manager>();
    if (!mgr->initialize(cache_dir, max_cache_size_gb, cache_ttl_seconds)) {
        SRV_WRN("%s\n", "Failed to initialize KV cache disk manager");
        return nullptr;
    }

    mgr->set_prompt_similarity_threshold(slot_prompt_similarity);

    if (compress_kv_cache) {
        LOG_INF(
            "KV cache auto enabled: dir='%s', max=%.1f GB, ttl=%lld s, sim_threshold=%.3f, compress=zstd level %d, "
            "learn=%d\n",
            cache_dir.c_str(), max_cache_size_gb, (long long) cache_ttl_seconds, slot_prompt_similarity,
            compress_kv_cache_level, compress_kv_cache_learn);
    } else {
        LOG_INF("KV cache auto enabled: dir='%s', max=%.1f GB, ttl=%lld s, sim_threshold=%.3f\n", cache_dir.c_str(),
                max_cache_size_gb, (long long) cache_ttl_seconds, slot_prompt_similarity);
    }

    return mgr;
}

bool try_log_kv_cache_metrics(kv_cache_metrics_collector & cache_metrics) {
    return cache_metrics.try_log(ggml_time_us());
}
