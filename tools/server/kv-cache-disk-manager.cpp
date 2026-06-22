#include "kv-cache-disk-manager.h"

#include "server-common.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

// Include llama.cpp headers for KV cache serialization
extern "C" {
#include "llama.h"
}

#include "log.h"

using namespace std::chrono_literals;

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

    // Add entry index to the deepest matching node (end of token sequence)
    // This enables efficient prefix matching: search traverses the trie and
    // collects entries from the deepest matching node, which have the longest
    // common prefix with the query.
    node->entry_indices.push_back(entry_index);
    LOG("KV cache trie insert done: root=%p, depth_node=%p, tokens=%zu, entry_index=%zu, "
        "node.entry_indices.size()=%zu\n",
        (void *) root_.get(), (void *) node, tokens.size(), entry_index, node->entry_indices.size());
}

std::vector<size_t> kv_cache_trie::search_prefix(const std::vector<int32_t> & tokens,
                                                 float /* min_similarity */) const {
    // Find the deepest node matching the token prefix
    kv_cache_trie_node * node = find_matching_node(tokens);
    std::vector<size_t>  matching_entries;

    LOG("KV cache search_prefix: root=%p, node=%p, node->entry_indices.size()=%zu\n", (void *) root_.get(),
        (void *) node, node ? node->entry_indices.size() : 0);
    if (node) {
        for (size_t idx : node->entry_indices) {
            LOG("KV cache search_prefix:   entry_index=%zu\n", idx);
        }
    }

    // Collect entries from the deepest matching node
    // This returns entries that share the longest common prefix with the query
    if (node && !node->entry_indices.empty()) {
        for (size_t entry_idx : node->entry_indices) {
            matching_entries.push_back(entry_idx);
        }
    } else if (root_ && !root_->entry_indices.empty()) {
        // Fallback to root if no matching node found
        for (size_t entry_idx : root_->entry_indices) {
            matching_entries.push_back(entry_idx);
        }
    }

    return matching_entries;
}

void kv_cache_trie::remove_entry(size_t entry_index) {
    // Traverse all nodes and remove entry_index from their entry_indices lists
    remove_entry_recursive(root_.get(), entry_index);
}

void kv_cache_trie::remove_entry_recursive(kv_cache_trie_node * node, size_t entry_index) {
    if (!node) {
        return;
    }

    // Remove entry_index from this node's list
    auto & indices = node->entry_indices;
    indices.erase(std::remove(indices.begin(), indices.end(), entry_index), indices.end());

    // Recurse into children
    if (node->children) {
        for (auto & pair : *node->children) {
            remove_entry_recursive(pair.second.get(), entry_index);
        }
    }
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // Create cache directory if it doesn't exist
    try {
        std::filesystem::create_directories(cache_dir);
    } catch (const std::exception & e) {
        LOG_ERR("Failed to create KV cache directory '%s': %s\n", cache_dir.c_str(), e.what());
        return false;
    }

    cache_dir_ = cache_dir;
    // Ensure cache_dir_ ends with a separator for consistent path construction
    if (!cache_dir_.empty() && cache_dir_.back() != '/') {
        cache_dir_ += '/';
    }
    if (max_size_gb > 0.0f) {
        max_size_bytes_ = static_cast<size_t>(max_size_gb * 1024ULL * 1024ULL * 1024ULL);
    }
    ttl_seconds_ = ttl_seconds;

    // Rebuild in-memory index from existing cache files on disk
    try {
        int rebuild_count = 0;
        for (const auto & file_entry : std::filesystem::directory_iterator(cache_dir_)) {
            if (!file_entry.is_regular_file()) {
                continue;
            }

            std::string filepath = file_entry.path().string();

            // Read file header (magic 4 + version 4 + n_token_count 4 = 12 bytes)
            std::ifstream file(filepath, std::ios::binary);
            if (!file) {
                continue;
            }

            uint8_t header[12];
            file.read(reinterpret_cast<char *>(header), 12);
            if (file.gcount() != 12) {
                continue;
            }

            uint32_t magic         = *(uint32_t *) (header + 0);
            uint32_t n_token_count = *(uint32_t *) (header + 8);

            // Validate magic
            if (magic != LLAMA_STATE_SEQ_MAGIC) {
                LOG("KV cache rebuild: invalid magic in '%s', skipping\n", filepath.c_str());
                continue;
            }

            // Read tokens
            std::vector<int32_t> tokens(n_token_count);
            file.read(reinterpret_cast<char *>(tokens.data()), sizeof(int32_t) * n_token_count);
            file.close();

            // Build cache entry
            disk_cache_entry entry;
            entry.filepath        = filepath;
            entry.file_size_bytes = static_cast<size_t>(file_entry.file_size());
            // Use stat to get file mtime in seconds since Unix epoch.
            // file_time_type has a different epoch from system_clock,
            // so time_since_epoch() cannot be compared with system_clock.
            struct stat st;
            if (stat(filepath.c_str(), &st) != 0) {
                LOG("KV cache rebuild: stat failed on '%s', skipping\n", filepath.c_str());
                continue;
            }
            entry.created_at_us = st.st_mtime * 1000000LL;
            LOG("KV cache rebuild: file='%s', created_at_us=%ld, now_us=%ld, age_s=%ld\n", filepath.c_str(),
                (long) entry.created_at_us,
                (long) std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count(),
                (long) ((std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count() -
                         entry.created_at_us) /
                        1000000));
            entry.last_accessed_us = entry.created_at_us;
            entry.seq_id           = 0;
            entry.tokens           = std::move(tokens);

            // Add to LRU ring
            ring_buffer_entry rbe;
            rbe.metadata     = entry;
            rbe.access_order = ++access_counter_;
            lru_ring_.push_back(rbe);
            filepath_index_[filepath] = lru_ring_.size() - 1;

            // Insert into Radix Tree for efficient prefix matching
            if (trie_ && !entry.tokens.empty()) {
                LOG("KV cache rebuild: inserting %zu tokens at entry_index=%zu\n", entry.tokens.size(),
                    lru_ring_.size() - 1);
                trie_->insert(entry.tokens, lru_ring_.size() - 1);
            }

            current_size_bytes_ += entry.file_size_bytes;
            rebuild_count++;
        }

        if (rebuild_count > 0) {
            LOG("KV cache rebuild: restored %d entries from disk (total %zu bytes)\n", rebuild_count,
                current_size_bytes_);
        }
    } catch (const std::exception & e) {
        LOG_ERR("KV cache: failed to rebuild index from disk: %s\n", e.what());
    }

    // Now reconcile is safe -- all valid files are in the index
    reconcile_orphaned_files();

    fprintf(stderr, "KV cache disk manager initialized: dir='%s', max_size=%.1f GB, ttl=%ld s, entries=%zu\n",
            cache_dir.c_str(), max_size_gb, (long) ttl_seconds, lru_ring_.size());

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
    disk_cache_entry * result = valid_candidates.front().entry;

    // Update LRU position for the matched entry (most recently accessed)
    auto idx_it = filepath_index_.find(result->filepath);
    if (idx_it != filepath_index_.end() && idx_it->second < lru_ring_.size()) {
        lru_ring_[idx_it->second].access_order = ++access_counter_;
    }

    return result;
}

// Find matching KV cache entry on disk for given token sequence
std::string kv_cache_disk_manager::find_cache_entry(const llama_tokens & tokens, float lcp_threshold) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // Evict expired entries before searching
    evict_expired_entries();

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

    // Read file header to extract n_token_count (magic 4 + version 4 + n_token_count 4 = 12 bytes)
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        LOG_WRN("KV cache restore: cannot open file '%s'\n", filepath.c_str());
        return false;
    }

    uint8_t header[12];
    file.read(reinterpret_cast<char *>(header), 12);
    if (file.gcount() != 12) {
        LOG_WRN("KV cache restore: invalid file header size in '%s'\n", filepath.c_str());
        return false;
    }

    uint32_t magic         = *(uint32_t *) (header + 0);
    uint32_t version       = *(uint32_t *) (header + 4);
    uint32_t n_token_count = *(uint32_t *) (header + 8);

    if (magic != LLAMA_STATE_SEQ_MAGIC) {
        LOG_WRN("KV cache restore: invalid magic in '%s' (expected 0x%08x, got 0x%08x)\n", filepath.c_str(),
                LLAMA_STATE_SEQ_MAGIC, magic);
        return false;
    }

    LOG("KV cache restore: file version=%u, token_count=%u\n", version, n_token_count);

    // Allocate buffer for token data so llama_state_seq_load_file can copy safely
    std::vector<llama_token> tokens(n_token_count);
    size_t                   n_token_count_out = 0;

    // Try restoring with -1 (null seq_id) first
    LOG("KV cache restore: attempting load with seq_id=-1, n_token_count=%u for file '%s'\n", n_token_count,
        filepath.c_str());
    size_t bytes_loaded =
        llama_state_seq_load_file(ctx_tgt, filepath.c_str(), -1, tokens.data(), n_token_count, &n_token_count_out);

    if (bytes_loaded == 0) {
        LOG_WRN("KV cache restore: failed to load state from '%s' with seq_id=-1\n", filepath.c_str());

        // Try loading with specific slot_id as fallback
        LOG("KV cache restore: trying with slot_id=%d as fallback\n", slot_id);
        bytes_loaded = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot_id, tokens.data(), n_token_count,
                                                 &n_token_count_out);

        if (bytes_loaded == 0) {
            LOG_WRN("KV cache restore: failed to load state from '%s' with slot_id=%d\n", filepath.c_str(), slot_id);
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    prompt_similarity_threshold_ = threshold;
    LOG("KV cache prompt similarity threshold set to %.3f\n", threshold);
}

float kv_cache_disk_manager::get_prompt_similarity_threshold() const {
    return prompt_similarity_threshold_;
}

float kv_cache_disk_manager::get_disk_lcp(const std::string & filepath, const std::vector<int32_t> & tokens) const {
    auto it = filepath_index_.find(filepath);
    if (it == filepath_index_.end() || it->second >= lru_ring_.size()) {
        return 0.0f;
    }
    return calculate_lcp_ratio(lru_ring_[it->second].metadata.tokens, tokens);
}

bool kv_cache_disk_manager::save_to_disk(int32_t         slot_id,
                                         llama_context * ctx_tgt,
                                         llama_context * ctx_dft,
                                         const int32_t * tokens,
                                         size_t          token_count) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    (void) ctx_dft;  // Reserved for future use (e.g., dual-context save)

    if (!ctx_tgt) {
        LOG_WRN("KV cache save: no target context\n");
        return false;
    }

    // Evict expired entries before saving
    evict_expired_entries();

    // Get current timestamp (system_clock for TTL comparison consistency)
    auto    now          = std::chrono::system_clock::now();
    int64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    // Generate filename
    std::string filename = generate_cache_filename(slot_id, timestamp_us);
    std::string filepath = cache_dir_ + filename;

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
    // This ensures compatibility with llama_state_seq_load_file for restore
    size_t bytes_written = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot_id, tokens, token_count);

    if (bytes_written == 0) {
        LOG_WRN("KV cache save: failed to save state to '%s'\n", filepath.c_str());
        return false;
    }

    // Verify write completed successfully
    FILE * fp_check = fopen(filepath.c_str(), "rb");
    if (fp_check) {
        fseek(fp_check, 0, SEEK_END);
        long file_size = ftell(fp_check);
        fclose(fp_check);

        if (static_cast<size_t>(file_size) != bytes_written) {
            LOG_WRN("KV cache save: file size mismatch for '%s' (expected %zu, got %ld)\n", filepath.c_str(),
                    bytes_written, file_size);
            remove_entry(filepath);
            return false;
        }
    }

    // Update metadata
    disk_cache_entry entry;
    entry.filepath         = filepath;
    entry.created_at_us    = timestamp_us;
    entry.last_accessed_us = timestamp_us;
    entry.file_size_bytes  = bytes_written;
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

    current_size_bytes_ += bytes_written;
    metrics_.saves_completed++;

    // Log trie statistics periodically (every 10 saves)
    static int save_counter = 0;
    save_counter++;
    if (save_counter % 10 == 0) {
        auto trie_stats = trie_->get_stats();
        LOG("KV cache trie stats: nodes=%zu, max_depth=%zu, branches=%zu\n", trie_stats.total_nodes,
            trie_stats.max_depth, trie_stats.root_branches);
    }

    LOG("KV cache saved: slot=%d, size=%zu bytes, file='%s'\n", slot_id, bytes_written, filepath.c_str());

    return true;
}

void kv_cache_disk_manager::evict_expired_entries() {
    if (ttl_seconds_ <= 0) {
        return;
    }

    auto    now    = std::chrono::system_clock::now();
    int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    LOG("KV cache evict_expired_entries: checking %zu entries\n", lru_ring_.size());

    // Collect expired filepaths first, then remove (to avoid iterator invalidation)
    std::vector<std::string> expired;
    for (const auto & entry : lru_ring_) {
        int64_t age_us = (now_us - entry.metadata.created_at_us) / 1000000;
        LOG("KV cache evict_expired_entries: file='%s', age=%ld s, ttl=%ld s\n", entry.metadata.filepath.c_str(),
            (long) age_us, (long) ttl_seconds_);

        if (age_us > ttl_seconds_) {
            expired.push_back(entry.metadata.filepath);
        }
    }

    for (const auto & filepath : expired) {
        remove_entry(filepath);
        metrics_.evictions_ttl++;
    }
    LOG("KV cache evict_expired_entries: expired=%zu, lru_ring_ size=%zu\n", expired.size(), lru_ring_.size());
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
    // Note: remove_entry already erases from lru_ring_ and updates indices
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
        size_t erased_idx = idx_it->second;

        // Remove from trie BEFORE modifying lru_ring_ (indices become stale)
        if (trie_) {
            trie_->remove_entry(erased_idx);
        }

        if (erased_idx < lru_ring_.size()) {
            lru_ring_.erase(lru_ring_.begin() + erased_idx);
        }

        // Update all indices in filepath_index_ that were after the erased entry
        for (auto & pair : filepath_index_) {
            if (pair.second > erased_idx) {
                --pair.second;
            }
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);

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
                std::filesystem::remove(filepath);
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("KV cache reconciliation: failed to scan directory '%s': %s\n", cache_dir_.c_str(), e.what());
    }
}

void kv_cache_disk_manager::purge_all_cache_files() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (cache_dir_.empty()) {
        return;
    }

    LOG("KV cache purge: removing all files from '%s'\n", cache_dir_.c_str());

    size_t total_freed   = 0;
    size_t files_removed = 0;

    // Collect filepaths first to avoid modifying lru_ring_ while iterating
    std::vector<std::string> filepaths;
    for (const auto & entry : std::filesystem::directory_iterator(cache_dir_)) {
        if (entry.is_regular_file()) {
            filepaths.push_back(entry.path().string());
        }
    }

    for (const auto & filepath : filepaths) {
        remove_entry(filepath);
        files_removed++;
    }

    LOG("KV cache purge complete: freed %zu bytes, removed %zu files\n", total_freed, files_removed);
}
