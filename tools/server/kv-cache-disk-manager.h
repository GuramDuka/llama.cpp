#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
struct llama_context;
struct server_tokens;

// Maximum tokens stored per cache entry (for LCP matching)
#define KV_CACHE_MAX_TOKENS 8192

// Metrics for cache operations tracking
struct kv_cache_metrics {
    std::atomic<uint64_t> cache_hits      = 0;
    std::atomic<uint64_t> cache_misses    = 0;
    std::atomic<uint64_t> saves_completed = 0;
    std::atomic<uint64_t> saves_skipped   = 0;
    std::atomic<uint64_t> evictions_ttl   = 0;
    std::atomic<uint64_t> evictions_lru   = 0;

    void reset() {
        cache_hits      = 0;
        cache_misses    = 0;
        saves_completed = 0;
        saves_skipped   = 0;
        evictions_ttl   = 0;
        evictions_lru   = 0;
    }
};

// Disk cache entry metadata
struct disk_cache_entry {
    std::string          filepath;          // Path to serialized KV state file
    int64_t              created_at_us;     // Creation timestamp (microseconds)
    int64_t              last_accessed_us;  // Last access timestamp
    size_t               file_size_bytes;   // Size of the cached data
    uint32_t             seq_id;            // Sequence ID associated with this entry
    std::vector<int32_t> tokens;            // Token sequence for LCP matching (first N tokens)

    bool is_expired(int64_t ttl_seconds) const {
        if (ttl_seconds <= 0) {
            return false;
        }
        int64_t age_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - std::chrono::time_point<std::chrono::steady_clock>(
                                                                    std::chrono::nanoseconds(created_at_us)))
                             .count();
        return (age_us / 1000000) > ttl_seconds;
    }
};

// Ring buffer entry for LRU eviction
struct ring_buffer_entry {
    disk_cache_entry metadata;

    // For efficient ordering in LRU structure
    int64_t access_order;
};

// ============================================================================
// Radix Tree (Trie) for Token Prefix Matching
// ============================================================================
// Provides O(m log k) prefix matching instead of O(n * m) linear search
// where m = token sequence length, k = unique tokens in vocabulary
// ============================================================================

class kv_cache_trie_node {
  public:
    uint32_t token_id;                  // Token ID at this node (0 for root)
    std::unique_ptr<std::unordered_map<uint32_t, std::unique_ptr<kv_cache_trie_node>>> children;
    std::vector<size_t> entry_indices;  // Indices of cache entries sharing this prefix

    kv_cache_trie_node() :
        token_id(0),
        children(std::make_unique<std::unordered_map<uint32_t, std::unique_ptr<kv_cache_trie_node>>>()) {}

    bool is_leaf() const { return children->empty(); }

    ~kv_cache_trie_node() = default;
};

class kv_cache_trie {
  public:
    kv_cache_trie();
    ~kv_cache_trie();

    // Insert token sequence into trie, associate with entry index
    void insert(const std::vector<int32_t> & tokens, size_t entry_index);

    // Search for matching entries by prefix similarity
    // Returns vector of entry indices that match the threshold
    std::vector<size_t> search_prefix(const std::vector<int32_t> & tokens, float min_similarity) const;

    // Remove all references to an entry index from trie
    void remove_entry(size_t entry_index);

    // Get trie statistics for debugging
    struct stats {
        size_t total_nodes   = 0;
        size_t max_depth     = 0;
        size_t root_branches = 0;
    };

    stats get_stats() const;

  private:
    std::unique_ptr<kv_cache_trie_node> root_;
    size_t                              max_depth_ = KV_CACHE_MAX_TOKENS;

    // Helper to traverse and find matching node
    kv_cache_trie_node * find_matching_node(const std::vector<int32_t> & tokens) const;

    // Count nodes recursively
    size_t count_nodes(kv_cache_trie_node * node) const;

    // Get max depth recursively
    size_t get_max_depth(kv_cache_trie_node * node) const;
};

class kv_cache_disk_manager {
  public:
    kv_cache_disk_manager();
    ~kv_cache_disk_manager();

    // Initialize the disk cache manager
    // cache_dir: directory path for storing cached KV states
    // max_size_gb: maximum total cache size in GB (0 = unlimited)
    // ttl_seconds: time-to-live for cache entries in seconds (0 = no expiration)
    bool initialize(const std::string & cache_dir, float max_size_gb, int64_t ttl_seconds);

    // Find matching KV cache entry on disk for given token sequence
    // Returns: filepath string if match found, empty string otherwise
    // Note: Caller must restore using llama_state_seq_load_file API
    std::string find_cache_entry(const std::vector<int32_t> & tokens, float lcp_threshold);

    // Restore KV cache state from disk file to slot context
    // filepath: path to saved state file
    // ctx_tgt: target context for restoration
    // Returns true on success
    bool restore_from_disk(const std::string & filepath, int32_t slot_id, llama_context * ctx_tgt);

    // Set the slot prompt similarity threshold for cache matching
    // This allows tuning sensitivity based on --slot-prompt-similarity parameter
    void set_prompt_similarity_threshold(float threshold);

    // Get current prompt similarity threshold
    float get_prompt_similarity_threshold() const;

    // Save the current KV cache state to disk after successful generation
    // tokens: token sequence for LCP matching (first N tokens stored)
    // Returns true on success
    bool save_to_disk(int32_t                      slot_id,
                      llama_context *              ctx_tgt,
                      llama_context *              ctx_dft = nullptr,
                      const std::vector<int32_t> * tokens  = nullptr);

    // Get current metrics (thread-safe)
    kv_cache_metrics get_metrics() const;

    // Reset metrics counter
    void reset_metrics();

    // Cleanup orphaned files on startup
    void reconcile_orphaned_files();

    // Cleanup all cache files (for manual purge)
    void purge_all_cache_files();

    // Calculate LCP ratio between two token sequences (public for use in callbacks)
    float calculate_lcp_ratio(const std::vector<int32_t> & tokens_a, const std::vector<int32_t> & tokens_b) const;

  private:
    // Generate unique filename for a slot's KV cache
    std::string generate_cache_filename(int32_t seq_id, int64_t timestamp_us);

    // Find matching cache entry by prompt similarity using Radix Tree
    disk_cache_entry * find_matching_entry(const std::vector<int32_t> & tokens, float threshold);

    // Evict expired entries based on TTL
    void evict_expired_entries();

    // Evict LRU entry when cache exceeds max size
    bool evict_lru_entry();

    // Remove a cache entry from disk and index
    void remove_entry(const std::string & filepath);

    // Update total cache size accounting
    void update_cache_size(int64_t delta_bytes);

    // Thread-safe access to cache index
    std::mutex mutex_;

    // Configuration
    std::string cache_dir_;
    size_t      max_size_bytes_ = 0;  // 0 means unlimited
    int64_t     ttl_seconds_    = 0;

    // Cache state tracking
    size_t           current_size_bytes_ = 0;
    kv_cache_metrics metrics_;

    // LRU ring buffer: ordered by last access time (oldest first)
    std::vector<ring_buffer_entry> lru_ring_;

    // Index for O(1) lookups by filepath
    std::unordered_map<std::string, size_t> filepath_index_;

    // Access order counter for LRU tracking
    int64_t access_counter_ = 0;

    // Radix Tree for efficient prefix matching
    std::unique_ptr<kv_cache_trie> trie_;

    // Prompt similarity threshold for cache matching (from --slot-prompt-similarity)
    float prompt_similarity_threshold_ = 0.0f;

    // Cleanup flag
    std::atomic<bool> shutdown_requested_ = false;
};
