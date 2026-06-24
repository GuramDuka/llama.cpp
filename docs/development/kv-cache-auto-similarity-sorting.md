# KV Cache Auto-Save/Restore -- Similarity Sorting and LCP Matching

## Overview

KV cache matching across all 3 tiers (L1 slots, L2 RAM, L3 disk) uses **Longest Common Prefix (LCP)** ratio as the similarity metric. All matching entries are collected into a single sorted pool and the best candidate is selected by a composite sort order.

---

## LCP Ratio Computation

```cpp
float calculate_lcp_ratio(const std::vector<int32_t> & tokens_a,
                          const std::vector<int32_t> & tokens_b) const;
```

LCP is defined as: `common_prefix_length / incoming_prompt_length`.

The incoming prompt length (`tokens_b`) is used as the denominator, so the ratio represents what fraction of the new request can be served from the cached prefix.

```cpp
// kv-cache-disk-manager.cpp
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

The same LCP logic applies to all 3 tiers:
- **L1**: `slot.prompt.tokens.get_common_prefix(task.tokens) / task.tokens.size()`
- **L2**: `state.tokens.get_common_prefix(tokens_new) / tokens_new.size()`
- **L3**: `calculate_lcp_ratio(entry.tokens, request.tokens)`

---

## Radix Tree (Trie) -- L3 Pre-Filtering

The L3 disk cache uses a Radix Tree for O(m log k) prefix matching where m = token sequence length, k = unique tokens. This eliminates O(n x m) linear search over all cache entries.

### Insert

Each token in the sequence is traversed down the trie. The entry index (pointing into `lru_ring_`) is stored at the deepest matching node.

### Search

`search_prefix()` traverses the trie following the incoming token sequence, returning the deepest matching node's `entry_indices`. If that node has no entries, it falls back to the last matching node, then the root.

### Remove

`remove_entry()` walks all nodes recursively, removing stale entry indices.

**Note**: L2 (RAM prompt cache) and L1 (slots) use linear scan over their respective arrays, which is acceptable at smaller scale.

---

## Candidate Collection per Tier

### L3 -- `find_all_matching_entries()` (kv_cache_disk_manager)

Returns ALL matching entries above threshold, not just the best one, for use in the combined pool:

```cpp
// kv-cache-disk-manager.cpp
std::vector<kv_cache_disk_manager::disk_cache_match> kv_cache_disk_manager::find_all_matching_entries(
    const llama_tokens & tokens, float lcp_threshold) {

    evict_expired_entries();
    if (!trie_ || tokens.empty()) return {};

    // Radix Tree pre-filter: O(m log k)
    std::vector<size_t> candidate_indices = trie_->search_prefix(tokens, 0.0f);
    if (candidate_indices.empty()) return {};

    std::vector<disk_cache_match> results;
    for (size_t idx : candidate_indices) {
        if (idx >= lru_ring_.size()) continue;
        float sim = calculate_lcp_ratio(lru_ring_[idx].metadata.tokens, tokens);
        if (sim >= effective_threshold) {
            results.push_back({ lru_ring_[idx].metadata.filepath, sim,
                                lru_ring_[idx].metadata.created_at_us });
        }
    }
    return results;
}
```

### L2 -- `find_all_matching()` (server_prompt_cache)

Linear scan of all cached prompt states:

```cpp
std::vector<server_prompt_cache::prompt_cache_match> server_prompt_cache::find_all_matching(
    const server_tokens & tokens_new, float threshold) const {
    std::vector<prompt_cache_match> results;
    for (const auto & state : states) {
        const int   lcp_cur = state.tokens.get_common_prefix(tokens_new);
        const float sim_cur = float(lcp_cur) / tokens_new.size();
        if (sim_cur >= threshold) {
            results.push_back({ sim_cur, state.created_us });
        }
    }
    return results;
}
```

### L1 -- Slots

Iterates idle slots, computes common prefix ratio directly:

```cpp
for (server_slot & slot : slots) {
    if (slot.is_processing()) continue;
    const auto & tokens = slot.prompt.tokens;
    if (tokens.empty()) continue;
    const float sim = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();
    if (sim > slot_prompt_similarity) {
        candidates.push_back({ TIER_L1_SLOT, sim, slot.t_last_used, &slot, "" });
    }
}
```

---

## Combined 3-Tier Pool Sorting

All candidates are combined into `vector<cache_candidate>` and sorted with a 3-key comparator:

```cpp
struct cache_candidate {
    enum tier_t { TIER_L1_SLOT, TIER_L2_RAM, TIER_L3_DISK };
    tier_t        tier;
    float         similarity;     // LCP ratio (higher = better match)
    int64_t       freshness;      // timestamp_us (higher = more recent)
    server_slot * slot;           // L1 only
    std::string   filepath;       // L3 only
};

std::sort(candidates.begin(), candidates.end(),
    [](const cache_candidate & a, const cache_candidate & b) {
        // 1. Similarity DESC -- best semantic match first
        if (std::abs(a.similarity - b.similarity) > 1e-6f)
            return a.similarity > b.similarity;
        // 2. Freshness DESC -- most recent first for ties
        if (a.freshness != b.freshness)
            return a.freshness > b.freshness;
        // 3. Tier priority -- L1 < L2 < L3 for equal similarity+freshness
        return a.tier < b.tier;
    });
```

### Selection Outcomes

| Best Tier | Action |
|---|---|
| **L1** (slot) | Use directly (KV already in GPU) |
| **L2** (RAM) | Load from RAM to slot via `prompt_load()` |
| **L3** (disk) | Restore from disk + promote L3->L2 via `prompt_save()` |
| None | LRU fallback (oldest `t_last_used`) |

### Example Scenarios

```
Incoming prompt: "The quick brown fox jumps over the lazy dog"

Candidates:
  L3: "The quick brown fox"       sim=0.44, created=10:00
  L2: "The quick brown fox jumps" sim=0.56, created=09:55
  L1: "The quick brown"           sim=0.33, t_last=09:58

Combined pool (sorted):
  1. L2, sim=0.56, fresh=09:55   ← BEST
  2. L3, sim=0.44, fresh=10:00
  3. L1, sim=0.33, fresh=09:58

Best: L2 (RAM) -- load to slot
```

```
Incoming prompt: "The quick brown fox jumps over the lazy dog"

Candidates:
  L3: "The quick brown fox jumps over the lazy"  sim=0.78, created=10:05
  L2: "The quick brown"                          sim=0.33, created=09:50

Combined pool (sorted):
  1. L3, sim=0.78, fresh=10:05   ← BEST
  2. L2, sim=0.33, fresh=09:50

Best: L3 (disk) -- restore to slot + promote to L2
```

---

## Configurable Threshold

The single `--slot-prompt-similarity` threshold applies to all 3 tiers:

```cpp
// server-context.cpp
kv_cache_disk_mgr->set_prompt_similarity_threshold(params_base.slot_prompt_similarity);
```

The threshold is used as:
- **L1**: `sim > slot_prompt_similarity` (strictly greater)
- **L2**: `sim >= threshold` (greater or equal)
- **L3**: `sim >= effective_threshold` (greater or equal)

- **Lower values**: more aggressive caching (more hits, potentially less accurate)
- **Higher values**: conservative caching (fewer hits, higher accuracy)

---

## Performance Comparison (L3 only)

| Cache Size | Linear Search | Radix Tree | Speedup |
|---|---|---|---|
| 100 entries | 5,000 ops | ~50 ops | **100x** |
| 1,000 entries | 100,000 ops | ~100 ops | **1,000x** |
| 10,000 entries | 2,000,000 ops | ~200 ops | **10,000x** |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Radix Tree over linear search for L3 | O(m log k) vs O(n x m) -- critical for large disk caches |
| Combined pool over sequential tier checks | Enables true cross-tier comparison (L3 with sim 0.9 beats L2 with 0.7) |
| Sort: similarity > freshness > tier | Best semantic match first; recency breaks ties; L1 preferred when equal |
| L3 -> L2 promotion on restore | Avoids re-reading disk if same prompt arrives again |
| Single `--slot-prompt-similarity` for all tiers | Consistent configuration, predictable behavior |
| `completed_successfully` gate | Prevents saving partial/corrupted KV state on interrupts |
