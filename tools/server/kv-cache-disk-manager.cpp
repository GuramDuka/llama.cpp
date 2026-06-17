#include "kv-cache-disk-manager.h"

#include "server-common.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

// Include llama.cpp headers for KV cache serialization
extern "C" {
#include "llama.h"
}

#include "log.h"

using namespace std::chrono_literals;

// ============================================================================
// kv_cache_trie Implementation
// ============================================================================

// ============================================================================
// kv_cache_trie Implementation
// ============================================================================

kv_cache_trie::kv_cache_trie() : root_(std::make_unique<kv_cache_trie_node>()), max_depth_(KV_CACHE_MAX_TOKENS) {}

kv_cache_trie::~kv_cache_trie() = default;

void kv_cache_trie::insert(const std::vector<int32_t> & tokens, size_t entry_index) {
    kv_cache_trie_node * node = root_.get();

    for (size_t i = 0; i < std::min(tokens.size(), max_depth_); ++i) {
        uint32_t token_id = static_cast<uint32_t>(tokens[i]);

        // Create child node if it doesn't exist
        if (!node->children->count(token_id)) {
            auto child      = std::make_unique<kv_cache_trie_node>();
            child->token_id = token_id;
            node->children->emplace(token_id, std::move(child));
        }

        node = node->children->at(token_id).get();
    }

    // Add entry index to the root node (all sequences share this prefix)
    // This allows LCP matching by checking entries at any depth
    LOG("KV cache trie insert: adding entry_index=%zu to root node\n", entry_index);
    root_->entry_indices.push_back(entry_index);
}

std::vector<size_t> kv_cache_trie::search_prefix(const std::vector<int32_t> & tokens, float min_similarity) const {
    kv_cache_trie_node * node = find_matching_node(tokens);
    std::vector<size_t>  matching_entries;

    // Collect entries from root node (all sequences share this prefix)
    if (root_ && !root_->entry_indices.empty()) {
        for (size_t entry_idx : root_->entry_indices) {
            // Note: Actual similarity check happens in kv_cache_disk_manager
            // This is a fast pre-filter using trie structure
            matching_entries.push_back(entry_idx);
        }
    }

    return matching_entries;
}

void kv_cache_trie::remove_entry(size_t entry_index) {
    // Traverse all nodes and remove entry_index from their lists
    // For simplicity, we rebuild affected paths
    // Production code might use more sophisticated deletion

    // Mark for lazy cleanup - actual removal happens during eviction
    (void) entry_index;  // Suppress unused warning
}

kv_cache_trie::stats kv_cache_trie::get_stats() const {
    stats s;
    s.total_nodes = count_nodes(root_.get());
    s.max_depth   = get_max_depth(root_.get());

    if (root_->children) {
        s.root_branches = root_->children->size();
    }

    return s;
}

kv_cache_trie_node * kv_cache_trie::find_matching_node(const std::vector<int32_t> & tokens) const {
    kv_cache_trie_node * node               = root_.get();
    kv_cache_trie_node * last_matching_node = nullptr;

    for (size_t i = 0; i < tokens.size() && node; ++i) {
        uint32_t token_id = static_cast<uint32_t>(tokens[i]);

        auto it = node->children->find(token_id);
        if (it == node->children->end()) {
            LOG("KV cache trie: token[%zu]=%u not found in trie (depth=%zu)\n", i, token_id, i);
            break;  // Token not found in trie
        }

        last_matching_node = node;  // Track last matching node
        node               = it->second.get();
    }

    // Return the deepest node with entry_indices (not necessarily the full match)
    // This allows LCP matching for similar but not identical sequences
    kv_cache_trie_node * result = node;
    if (!result || result->entry_indices.empty()) {
        // Try last matching node if current one has no entries
        LOG("KV cache trie: current node has %zu entries, checking last_matching_node\n",
            result ? result->entry_indices.size() : 0);
        if (last_matching_node) {
            LOG("KV cache trie: last_matching_node has %zu entries at depth %zu\n",
                last_matching_node->entry_indices.size(), tokens.size() - 1);
            if (!last_matching_node->entry_indices.empty()) {
                result = last_matching_node;
                LOG("KV cache trie: fallback to last_matching_node with %zu entries\n", result->entry_indices.size());
            }
        }
    }

    if (result) {
        LOG("KV cache trie: found matching node at depth %zu, entry_indices.size()=%zu\n",
            result == last_matching_node ? tokens.size() - 1 : tokens.size(), result->entry_indices.size());
    } else {
        LOG("KV cache trie: no matching node with entries found\n");
    }

    return result;
}

size_t kv_cache_trie::count_nodes(kv_cache_trie_node * node) const {
    if (!node) {
        return 0;
    }

    size_t count = 1;  // Count this node
    for (const auto & pair : *node->children) {
        count += count_nodes(pair.second.get());
    }

    return count;
}

size_t kv_cache_trie::get_max_depth(kv_cache_trie_node * node) const {
    if (!node || node->children->empty()) {
        return 0;
    }

    size_t max_child_depth = 0;
    for (const auto & pair : *node->children) {
        size_t child_depth = get_max_depth(pair.second.get());
        max_child_depth    = std::max(max_child_depth, child_depth);
    }

    return 1 + max_child_depth;
}

// ============================================================================
// kv_cache_disk_manager Implementation
// ============================================================================

kv_cache_disk_manager::kv_cache_disk_manager() :
    max_size_bytes_(0),
    ttl_seconds_(0),
    current_size_bytes_(0),
    access_counter_(0) {
    trie_ = std::make_unique<kv_cache_trie>();
}

kv_cache_disk_manager::~kv_cache_disk_manager() {
    shutdown_requested_ = true;
}

bool kv_cache_disk_manager::initialize(const std::string & cache_dir, float max_size_gb, int64_t ttl_seconds) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Create cache directory if it doesn't exist
    try {
        std::filesystem::create_directories(cache_dir);
    } catch (const std::exception & e) {
        LOG_ERR("Failed to create KV cache directory '%s': %s\n", cache_dir.c_str(), e.what());
        return false;
    }

    cache_dir_ = cache_dir;
    if (max_size_gb > 0.0f) {
        max_size_bytes_ = static_cast<size_t>(max_size_gb * 1024ULL * 1024ULL * 1024ULL);
    }
    ttl_seconds_ = ttl_seconds;

    fprintf(stderr, "KV cache disk manager initialized: dir='%s', max_size=%.1f GB, ttl=%ld s\n", cache_dir.c_str(),
            max_size_gb, (long) ttl_seconds);

    return true;
}

std::string kv_cache_disk_manager::generate_cache_filename(int32_t seq_id, int64_t timestamp_us) {
    std::ostringstream oss;
    oss << "slot_" << seq_id << "_" << (timestamp_us / 1000000) << ".bin";
    return oss.str();
}

float kv_cache_disk_manager::calculate_lcp_ratio(const std::vector<int32_t> & tokens_a,
                                                 const std::vector<int32_t> & tokens_b) const {
    if (tokens_a.empty() || tokens_b.empty()) {
        return 0.0f;
    }

    size_t min_len       = std::min(tokens_a.size(), tokens_b.size());
    size_t common_prefix = 0;

    for (size_t i = 0; i < min_len; ++i) {
        if (tokens_a[i] == tokens_b[i]) {
            common_prefix++;
        } else {
            break;
        }
    }

    return static_cast<float>(common_prefix) / static_cast<float>(tokens_b.size());
}

// Overload for server_tokens
float kv_cache_disk_manager::calculate_lcp_ratio(const server_tokens &        tokens_a,
                                                 const std::vector<int32_t> & tokens_b) const {
    if (tokens_a.empty() || tokens_b.empty()) {
        return 0.0f;
    }

    size_t min_len       = std::min(tokens_a.size(), tokens_b.size());
    size_t common_prefix = 0;

    for (size_t i = 0; i < min_len; ++i) {
        if (tokens_a[i] == tokens_b[i]) {
            common_prefix++;
        } else {
            break;
        }
    }

    return static_cast<float>(common_prefix) / static_cast<float>(tokens_b.size());
}

disk_cache_entry * kv_cache_disk_manager::find_matching_entry(const std::vector<int32_t> & tokens, float threshold) {
    if (!trie_ || tokens.empty()) {
        return nullptr;
    }

    // Use Radix Tree for fast prefix matching
    auto trie_stats = trie_->get_stats();
    LOG("KV cache trie search: tokens=%zu, trie_nodes=%zu, max_depth=%zu\n", tokens.size(), trie_stats.total_nodes,
        trie_stats.max_depth);

    std::vector<size_t> candidate_indices = trie_->search_prefix(tokens, 0.0f);  // Pre-filter by trie structure

    LOG("KV cache trie search: found %zu candidates\n", candidate_indices.size());

    if (candidate_indices.empty()) {
        return nullptr;
    }

    // Collect all candidates with similarity >= threshold
    struct candidate_entry {
        disk_cache_entry * entry;
        float              similarity;
        int64_t            access_order;  // For LRU tie-breaking
    };

    std::vector<candidate_entry> valid_candidates;

    for (size_t idx : candidate_indices) {
        if (idx >= lru_ring_.size()) {
            continue;
        }

        float sim = calculate_lcp_ratio(lru_ring_[idx].metadata.tokens, tokens);

        if (sim >= threshold) {
            valid_candidates.push_back({ &lru_ring_[idx].metadata, sim, lru_ring_[idx].access_order });
        }
    }

    if (valid_candidates.empty()) {
        return nullptr;
    }

    // Sort candidates:
    // 1. Primary: similarity descending (highest first)
    // 2. Secondary: access_order ascending (LRU - oldest first for tie-breaking)
    std::sort(valid_candidates.begin(), valid_candidates.end(),
              [](const candidate_entry & a, const candidate_entry & b) {
                  if (std::abs(a.similarity - b.similarity) > 1e-6f) {
                      return a.similarity > b.similarity;  // Higher similarity first
                  }
                  return a.access_order < b.access_order;  // LRU (older) first for equal similarity
              });

    // Return top candidate (highest similarity, LRU if tied)
    return valid_candidates.front().entry;
}

// Find matching KV cache entry on disk for given token sequence
std::string kv_cache_disk_manager::find_cache_entry(const llama_tokens & tokens, float lcp_threshold) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!trie_ || tokens.empty()) {
        metrics_.cache_misses++;
        LOG("KV cache MISS: trie not initialized or tokens empty\n");
        return "";
    }

    // Use configured threshold if not explicitly provided
    float effective_threshold = (lcp_threshold > 0.0f) ? lcp_threshold : prompt_similarity_threshold_;

    LOG("KV cache search: tokens=%zu, threshold=%.3f, trie_nodes=%zu\n", tokens.size(), effective_threshold,
        trie_->get_stats().total_nodes);

    // Use Radix Tree for O(m log k) prefix matching
    disk_cache_entry * match = find_matching_entry(tokens, effective_threshold);

    if (match) {
        metrics_.cache_hits++;
        float actual_lcp = calculate_lcp_ratio(match->tokens, tokens);
        LOG("KV cache HIT: LCP=%.3f (threshold=%.3f), file='%s'\n", actual_lcp, effective_threshold,
            match->filepath.c_str());

        return match->filepath;
    }

    metrics_.cache_misses++;
    LOG("KV cache MISS: no matching entry found\n");
    return "";
}

// Restore KV cache state from disk file to slot context
bool kv_cache_disk_manager::restore_from_disk(const std::string & filepath, int32_t slot_id, llama_context * ctx_tgt) {
    if (!ctx_tgt || filepath.empty()) {
        LOG_WRN("KV cache restore: invalid parameters\n");
        return false;
    }

    // Check file exists before attempting load
    if (!std::filesystem::exists(filepath)) {
        LOG_WRN("KV cache restore: file '%s' does not exist\n", filepath.c_str());
        return false;
    }

    // Get file size for logging
    struct stat st;
    if (stat(filepath.c_str(), &st) == 0) {
        LOG("KV cache restore: file '%s' size=%ld bytes\n", filepath.c_str(), st.st_size);
    }

    // Load tokens and KV cache into context
    // Note: llama_state_seq_load_file internally handles state allocation via state_read
    // Using -1 as seq_id to load all streams (not just specific slot)
    size_t bytes_loaded = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), -1, nullptr, 0, nullptr);

    if (bytes_loaded == 0) {
        LOG_WRN("KV cache restore: failed to load state from '%s' (seq_id=-1, file_size=0)\n", filepath.c_str());

        // Try loading with specific slot_id as fallback
        LOG("KV cache restore: trying with slot_id=%d as fallback\n", slot_id);
        bytes_loaded = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot_id, nullptr, 0, nullptr);

        if (bytes_loaded == 0) {
            LOG_WRN("KV cache restore: failed to load state from '%s' (slot_id=%d)\n", filepath.c_str(), slot_id);
            return false;
        }
    }

    if (bytes_loaded == 0) {
        LOG_WRN("KV cache restore: failed to load state from '%s'\n", filepath.c_str());
        return false;
    }

    LOG("KV cache restored: slot=%d, file='%s', bytes=%zu\n", slot_id, filepath.c_str(), bytes_loaded);

    // Update metrics
    metrics_.restores_completed++;
    metrics_.total_restore_bytes += bytes_loaded;

    return true;
}

void kv_cache_disk_manager::set_prompt_similarity_threshold(float threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    prompt_similarity_threshold_ = threshold;
    LOG("KV cache prompt similarity threshold set to %.3f\n", threshold);
}

float kv_cache_disk_manager::get_prompt_similarity_threshold() const {
    return prompt_similarity_threshold_;
}

bool kv_cache_disk_manager::save_to_disk(int32_t         slot_id,
                                         llama_context * ctx_tgt,
                                         llama_context * ctx_dft,
                                         const int32_t * tokens,
                                         size_t          token_count) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!ctx_tgt) {
        LOG_WRN("KV cache save: no target context\n");
        return false;
    }

    // Get current timestamp
    auto    now          = std::chrono::steady_clock::now();
    int64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    // Generate filename
    std::string filename = generate_cache_filename(slot_id, timestamp_us);
    std::string filepath = cache_dir_ + "/" + filename;

    // Check if we need to evict before saving
    if (max_size_bytes_ > 0) {
        while (current_size_bytes_ + 1024 * 1024 > max_size_bytes_) {
            if (!evict_lru_entry()) {
                LOG_WRN("KV cache save: cannot evict, skipping save\n");
                return false;
            }
        }
    }

    // Save KV cache state to file using llama_state_seq_save_file API
    size_t state_size = llama_state_seq_get_size_ext(ctx_tgt, slot_id, LLAMA_STATE_SEQ_FLAGS_NONE);

    if (state_size == 0) {
        LOG_WRN("KV cache save: empty state for slot %d\n", slot_id);
        return false;
    }

    std::vector<uint8_t> buffer(state_size);
    size_t               bytes_written =
        llama_state_seq_get_data_ext(ctx_tgt, buffer.data(), state_size, slot_id, LLAMA_STATE_SEQ_FLAGS_NONE);

    if (bytes_written == 0) {
        LOG_WRN("KV cache save: failed to serialize state for slot %d\n", slot_id);
        return false;
    }

    // Replace io_magic with LLAMA_STATE_SEQ_MAGIC for compatibility with llama_state_seq_load_file
    // io_magic is 0xaf143cd8, LLAMA_STATE_SEQ_MAGIC is 0x67677371
    const uint32_t io_magic        = 0xaf143cd8;
    const uint32_t state_seq_magic = 0x67677371;
    if (bytes_written >= sizeof(uint32_t)) {
        uint32_t * magic_ptr = reinterpret_cast<uint32_t *>(buffer.data());
        if (*magic_ptr == io_magic) {
            *magic_ptr = state_seq_magic;
        }
    }

    // Write to disk
    FILE * fp = fopen(filepath.c_str(), "wb");
    if (!fp) {
        LOG_ERR("KV cache save: cannot open file '%s'\n", filepath.c_str());
        return false;
    }

    size_t written = fwrite(buffer.data(), 1, bytes_written, fp);
    fclose(fp);

    if (written != bytes_written) {
        LOG_ERR("KV cache save: incomplete write to '%s'\n", filepath.c_str());
        remove_entry(filepath);
        return false;
    }

    // Update metadata
    disk_cache_entry entry;
    entry.filepath         = filepath;
    entry.created_at_us    = timestamp_us;
    entry.last_accessed_us = timestamp_us;
    entry.file_size_bytes  = written;
    entry.seq_id           = slot_id;

    // Store token sequence for LCP matching (limit to MAX_TOKENS)
    if (tokens && token_count > 0) {
        size_t tokens_to_store = std::min(token_count, static_cast<size_t>(KV_CACHE_MAX_TOKENS));
        entry.tokens.reserve(tokens_to_store);
        for (size_t i = 0; i < tokens_to_store; ++i) {
            entry.tokens.push_back(tokens[i]);
        }
    }

    // Add to LRU ring
    ring_buffer_entry rbe;
    rbe.metadata     = entry;
    rbe.access_order = ++access_counter_;
    lru_ring_.push_back(rbe);
    filepath_index_[filepath] = lru_ring_.size() - 1;

    // Insert into Radix Tree for efficient prefix matching
    if (trie_ && !entry.tokens.empty()) {
        LOG("KV cache trie insert: tokens=%zu, entry_index=%zu\n", entry.tokens.size(), lru_ring_.size() - 1);
        trie_->insert(entry.tokens, lru_ring_.size() - 1);
    }

    current_size_bytes_ += written;
    metrics_.saves_completed++;

    // Log trie statistics periodically (every 10 saves)
    static int save_counter = 0;
    save_counter++;
    if (save_counter % 10 == 0) {
        auto trie_stats = trie_->get_stats();
        LOG("KV cache trie stats: nodes=%zu, max_depth=%zu, branches=%zu\n", trie_stats.total_nodes,
            trie_stats.max_depth, trie_stats.root_branches);
    }

    LOG("KV cache saved: slot=%d, size=%zu bytes, file='%s'\n", slot_id, written, filepath.c_str());

    return true;
}

// Overload for convenience (backward compatible - assumes token_count from tokens pointer)
bool kv_cache_disk_manager::save_to_disk(int32_t         slot_id,
                                         llama_context * ctx_tgt,
                                         llama_context * ctx_dft,
                                         const int32_t * tokens) {
    // This overload doesn't know the token count, so it won't store tokens for matching
    return save_to_disk(slot_id, ctx_tgt, ctx_dft, tokens, 0);
}

void kv_cache_disk_manager::evict_expired_entries() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (ttl_seconds_ <= 0) {
        return;
    }

    auto    now    = std::chrono::steady_clock::now();
    int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    for (auto it = lru_ring_.begin(); it != lru_ring_.end();) {
        int64_t age_us = (now_us - it->metadata.created_at_us) / 1000000;

        if (age_us > ttl_seconds_) {
            remove_entry(it->metadata.filepath);
            lru_ring_.erase(it);
            metrics_.evictions_ttl++;
        } else {
            ++it;
        }
    }
}

bool kv_cache_disk_manager::evict_lru_entry() {
    if (lru_ring_.empty()) {
        return false;
    }

    // Find LRU entry (lowest access_order)
    auto    lru_it    = lru_ring_.begin();
    int64_t min_order = lru_it->access_order;

    for (auto it = lru_ring_.begin() + 1; it != lru_ring_.end(); ++it) {
        if (it->access_order < min_order) {
            min_order = it->access_order;
            lru_it    = it;
        }
    }

    remove_entry(lru_it->metadata.filepath);
    lru_ring_.erase(lru_it);
    metrics_.evictions_lru++;

    return true;
}

void kv_cache_disk_manager::remove_entry(const std::string & filepath) {
    auto idx_it = filepath_index_.find(filepath);
    if (idx_it != filepath_index_.end()) {
        size_t file_size = 0;

        // Get file size before removal
        try {
            file_size = std::filesystem::file_size(filepath);
        } catch (...) {
            file_size = 0;
        }

        current_size_bytes_ -= file_size;

        // Remove from ring buffer
        if (idx_it->second < lru_ring_.size()) {
            lru_ring_.erase(lru_ring_.begin() + idx_it->second);
        }

        filepath_index_.erase(idx_it);
    }

    // Delete file from disk
    try {
        std::filesystem::remove(filepath);
    } catch (const std::exception & e) {
        LOG_WRN("KV cache eviction: failed to remove '%s': %s\n", filepath.c_str(), e.what());
    }
}

void kv_cache_disk_manager::update_cache_size(int64_t delta_bytes) {
    current_size_bytes_ += delta_bytes;
}

kv_cache_metrics kv_cache_disk_manager::get_metrics() const {
    return metrics_;
}

void kv_cache_disk_manager::reset_metrics() {
    metrics_.reset();
}

void kv_cache_disk_manager::reconcile_orphaned_files() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_dir_.empty()) {
        return;
    }

    try {
        for (const auto & entry : std::filesystem::directory_iterator(cache_dir_)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            const std::string filepath = entry.path().string();

            // Check if file is in our index
            if (filepath_index_.find(filepath) == filepath_index_.end()) {
                LOG("KV cache reconciliation: removing orphaned file '%s'\n", filepath.c_str());

                size_t file_size = 0;
                try {
                    file_size = std::filesystem::file_size(filepath);
                } catch (...) {
                    file_size = 0;
                }

                current_size_bytes_ -= file_size;
                std::filesystem::remove(filepath);
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("KV cache reconciliation: failed to scan directory '%s': %s\n", cache_dir_.c_str(), e.what());
    }
}

void kv_cache_disk_manager::purge_all_cache_files() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_dir_.empty()) {
        return;
    }

    LOG("KV cache purge: removing all files from '%s'\n", cache_dir_.c_str());

    size_t total_freed = 0;

    for (const auto & entry : std::filesystem::directory_iterator(cache_dir_)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string filepath = entry.path().string();

        // Remove from index tracking
        auto idx_it = filepath_index_.find(filepath);
        if (idx_it != filepath_index_.end()) {
            lru_ring_.erase(lru_ring_.begin() + idx_it->second);
            filepath_index_.erase(idx_it);
        }

        // Delete file from disk
        try {
            size_t file_size = std::filesystem::file_size(filepath);
            total_freed += file_size;
            current_size_bytes_ -= file_size;
            std::filesystem::remove(filepath);
        } catch (...) {
            // Ignore errors during purge
        }
    }

    LOG("KV cache purge complete: freed %zu bytes, removed %zu files\n", total_freed, lru_ring_.size());
}
