#pragma once

#include <cstdint>

#ifdef LLAMA_HAS_ZSTD

// Global compression statistics (exposed via llama_kv_cache_compression_stats_*)
struct llm_compress_stats {
    int64_t  total_compress_us   = 0;
    int64_t  total_decompress_us = 0;
    uint64_t total_uncompressed  = 0;
    uint64_t total_compressed    = 0;
    uint64_t compress_count      = 0;
    uint64_t decompress_count    = 0;
};

extern llm_compress_stats g_comp_stats;

void llama_kv_cache_compression_stats_reset(void);
void llama_kv_cache_compression_stats_get(int64_t *  total_compress_us,
                                          int64_t *  total_decompress_us,
                                          uint64_t * total_uncompressed,
                                          uint64_t * total_compressed,
                                          uint64_t * compress_count,
                                          uint64_t * decompress_count);

#endif  // LLAMA_HAS_ZSTD
