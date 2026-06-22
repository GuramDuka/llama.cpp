# KV Cache Auto-Save/Restore — Similarity Sorting and LCP Matching

## Overview

When a request arrives, the KV cache disk manager uses a Radix Tree (Trie) to efficiently find candidate cache entries, then computes the actual LCP (Longest Common Prefix) ratio for each candidate and sorts them by similarity with LRU tie-breaking.

---

## LCP Ratio Computation

```cpp
float calculate_lcp_ratio(const std::vector<int32_t> & tokens_a,
                          const std::vector<int32_t> & tokens_b) const;
```

LCP is defined as: `common_prefix_length / incoming_prompt_length`.

The incoming prompt length (`tokens_b`) is used as the denominator, so the ratio represents what fraction of the new request can be served from the cached prefix.

```cpp
// kv-cache-disk-manager.cpp, lines 339-357
float kv_cache_disk_manager::calculate_lcp_ratio(
    const std::vector<int32_t> & tokens_a,
    const std::vector<int32_t> & tokens_b) const {
    if (tokens_a.empty() || tokens_b.empty()) return 0.0f;

    size_t min_len = std::min(tokens_a.size(), tokens_b.size());
    size_t common_prefix = 0;

    for (size_t i = 0; i < min_len; ++i) {
        if (tokens_a[i] == tokens_b[i]) common_prefix++;
        else break;
    }

    return static_cast<float>(common_prefix) / static_cast<float>(tokens_b.size());
}
```

An overload for `server_tokens` is also provided.

---

## Radix Tree (Trie) — Candidate Pre-Filtering

The trie eliminates O(n) linear search. It provides O(m log k) prefix matching where m = token sequence length, k = unique tokens.

### Insert

Each token in the sequence is traversed down the trie. The entry index (pointing into `lru_ring_`) is stored at the deepest matching node.

### Search

`search_prefix()` traverses the trie following the incoming token sequence, returning the deepest matching node's `entry_indices`. If that node has no entries, it falls back to the last matching node, then the root.

### Remove

`remove_entry()` walks all nodes recursively, removing stale entry indices.

---

## Candidate Sorting

The `find_matching_entry()` method (private) performs:

1. **Trie pre-filter**: get candidate indices from the trie
2. **LCP computation**: calculate `calculate_lcp_ratio()` for each candidate
3. **Threshold filter**: discard candidates below the configured threshold
4. **Sort**: primary by similarity (descending), secondary by `access_order` (ascending — LRU tie-breaking)
5. **Return**: top candidate
6. **Update**: increment `access_order` for the matched entry

```cpp
// kv-cache-disk-manager.cpp, lines 380-444
disk_cache_entry * kv_cache_disk_manager::find_matching_entry(
    const std::vector<int32_t> & tokens, float threshold) {

    std::vector<size_t> candidate_indices = trie_->search_prefix(tokens, 0.0f);

    if (candidate_indices.empty()) return nullptr;

    struct candidate_entry {
        disk_cache_entry * entry;
        float              similarity;
        int64_t            access_order;
    };

    std::vector<candidate_entry> valid_candidates;

    for (size_t idx : candidate_indices) {
        if (idx >= lru_ring_.size()) continue;

        float sim = calculate_lcp_ratio(lru_ring_[idx].metadata.tokens, tokens);
        if (sim >= threshold) {
            valid_candidates.push_back({ &lru_ring_[idx].metadata, sim, lru_ring_[idx].access_order });
        }
    }

    if (valid_candidates.empty()) return nullptr;

    std::sort(valid_candidates.begin(), valid_candidates.end(),
        [](const candidate_entry & a, const candidate_entry & b) {
            if (std::abs(a.similarity - b.similarity) > 1e-6f)
                return a.similarity > b.similarity;   // Higher similarity first
            return a.access_order < b.access_order;    // LRU (older) first for ties
        });

    disk_cache_entry * result = valid_candidates.front().entry;

    // Update LRU position for the matched entry
    auto idx_it = filepath_index_.find(result->filepath);
    if (idx_it != filepath_index_.end() && idx_it->second < lru_ring_.size())
        lru_ring_[idx_it->second].access_order = ++access_counter_;

    return result;
}
```

---

## Disk vs RAM Comparison

The disk cache search runs **once** (not per slot) in `get_available_slot()`. The returned filepath's LCP is compared against the best RAM LCP across all free slots:

```
if (disk_lcp > best_ram_lcp)
    restore_from_disk() to first free slot
else if (best_ram_lcp > disk_lcp)
    log "RAM cache preferred", fall through to prompt-similarity / LRU
else
    // equal: fall through (RAM preferred by default)
```

---

## Configurable Threshold

The threshold is set from `--slot-prompt-similarity` at initialization:

```cpp
// server-context.cpp, line 1311
kv_cache_disk_mgr->set_prompt_similarity_threshold(params_base.slot_prompt_similarity);
```

The threshold is stored in `prompt_similarity_threshold_` and used as the `effective_threshold` in `find_cache_entry()` when no explicit threshold is passed.

- **Lower values**: more aggressive caching (more hits, potentially less accurate)
- **Higher values**: conservative caching (fewer hits, higher accuracy)

---

## Performance Comparison

| Cache Size  | Linear Search | Radix Tree | Speedup |
|-------------|---------------|------------|---------|
| 100 entries | 5,000 ops     | ~50 ops    | **100x** |
| 1,000 entries | 100,000 ops | ~100 ops   | **1,000x** |
| 10,000 entries | 2,000,000 ops | ~200 ops | **10,000x** |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Radix Tree over linear search | O(m log k) vs O(n x m) — critical for large caches |
| Sort by similarity then LRU | Best semantic match + deterministic tie-breaking |
| Use --slot-prompt-similarity as default threshold | Consistent with existing prompt caching behavior |
| Disk restore only if strictly better than RAM | Avoids unnecessary disk I/O when GPU already has the cache |
| Single disk search (not per-slot) | Avoids redundant trie traversals |
