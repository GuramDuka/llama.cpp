#include "llama-compress-stats.h"

#ifdef LLAMA_HAS_ZSTD

llm_compress_stats g_comp_stats;

void llama_kv_cache_compression_stats_reset() {
    g_comp_stats = {};
}

void llama_kv_cache_compression_stats_get(int64_t *  total_compress_us,
                                          int64_t *  total_decompress_us,
                                          uint64_t * total_uncompressed,
                                          uint64_t * total_compressed,
                                          uint64_t * compress_count,
                                          uint64_t * decompress_count) {
    *total_compress_us   = g_comp_stats.total_compress_us;
    *total_decompress_us = g_comp_stats.total_decompress_us;
    *total_uncompressed  = g_comp_stats.total_uncompressed;
    *total_compressed    = g_comp_stats.total_compressed;
    *compress_count      = g_comp_stats.compress_count;
    *decompress_count    = g_comp_stats.decompress_count;
}

#endif  // LLAMA_HAS_ZSTD
