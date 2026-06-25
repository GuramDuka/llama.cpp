#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward declarations
struct llama_context;
struct kv_cache_disk_manager;
struct kv_cache_metrics_collector;

// ============================================================================
// Initialize KV cache disk manager on server start
// ============================================================================
// Returns the manager if initialized, nullptr otherwise
std::unique_ptr<kv_cache_disk_manager> init_kv_cache_disk_manager(bool                kv_cache_auto,
                                                                  float               max_cache_size_gb,
                                                                  int64_t             cache_ttl_seconds,
                                                                  float               slot_prompt_similarity,
                                                                  const std::string & slot_save_path,
                                                                  bool                compress_kv_cache,
                                                                  int                 compress_kv_cache_level,
                                                                  bool                compress_kv_cache_learn);

// ============================================================================
// Periodic metrics logging
// ============================================================================
// Call periodically; returns true if a log was emitted
bool try_log_kv_cache_metrics(kv_cache_metrics_collector & cache_metrics);
