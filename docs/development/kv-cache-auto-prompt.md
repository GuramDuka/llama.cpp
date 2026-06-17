# llama.cpp KV Cache Auto-Save/Restore Feature Implementation Prompt

## Context

You are a Senior C++ Software Engineer specializing in Large Language Model inference engines, specifically the `llama.cpp` architecture. Your expertise lies in optimizing KV cache management, slot systems, and memory efficiency.

## Communication Protocol
- **User Interaction**: You must communicate with the user exclusively in **Russian**.
- **Code & Technical Output**: All code snippets, file names, comments within code, and technical documentation must be in **English**.

## Project Objective

Implement an automatic KV cache save/restore mechanism on disk for `llama.cpp`'s `llama-server`. This feature must integrate seamlessly into the existing slot system with minimal modifications to the core codebase. The goal is to produce a clean patch file that allows easy upstreaming by keeping new functionality in separate files/modules.

---

## Technical Specifications

### 1. Core Logic

#### Cache Lookup (Radix Tree for LCP Matching)
Upon receiving a prompt, calculate the LCP (Longest Common Prefix) fraction relative to the input prompt length using an efficient Radix Tree (Trie) data structure:
- If an existing slot has a KV cache matching this LCP fraction, skip restoration entirely and reuse the slot.
- Use `--slot-prompt-similarity` parameter as default threshold for candidate filtering.
- Sort matching candidates by similarity (descending), then by access order (LRU ascending) for tie-breaking.

#### Slot Allocation
- Prefer free slots.
- If no free slots are available, reclaim the Least Recently Used (LRU) slot.

#### Persistence Policy
- Save KV state to disk only after a response completes normally.
- Skip saving if the stream was cancelled or if the incoming prompt already matched an existing cache entry (to avoid redundant writes).

#### Eviction Policy
- Use a Ring Buffer for disk storage.
- Evict expired entries first (based on age/TTL).
- If cache exceeds configured max size, evict based on LRU.

#### Reconciliation
On startup, detect and clean up orphaned or corrupted metadata files.

### 2. Command Line Interface (CLI) Options
Add the following options to `llama-server`:
1. `--kv-cache-auto <bool>`: Enable/disable automatic KV cache management (default: off).
2. `--max-cache-size <gb>`: Maximum total cache size per backend in GB.
3. `--cache-ttl <seconds>`: Delete cache files older than this duration (0 = disabled).
4. `--kv-cache-dir <path>`: Directory for automatic KV cache storage (default: slot-save-path + "/kv-meta").

### 3. Logging & Metrics
- Integrate all automatic KV cache operations into the existing logging system.
- Log verbosity must be set to `>= 4` for these operations.
- Include metrics/statistics in logs regarding cache hits, misses, saves, and evictions.

---

## Implementation Guidelines

### 1. Modularity
Do not modify core files (`server.cpp`, `slot.cpp`) excessively. Create new header/source files (e.g., `kv-cache-disk-manager.h/cpp`) to encapsulate this logic.

### 2. Integration Points
Hook into the slot lifecycle methods:
- **Creation**: Initialize manager in `load_model()` or `init()`
- **Allocation**: Try restore from disk in `get_available_slot()` before prompt similarity matching
- **Generation completion**: Save KV state after generation completes (deferred via callback)
- **Destruction**: Cleanup manager in `destroy()`

### 3. Thread Safety
Ensure thread-safe access to the disk cache index and file operations using mutexes.

### 4. Radix Tree Implementation
Use a Trie data structure for efficient prefix matching:
- **Time Complexity**: O(m log k) where m = token sequence length, k = vocabulary size
- **Space Efficiency**: Shared prefixes don't duplicate nodes
- **Search Strategy**: 
  1. Use trie to find candidate entries with matching prefix
  2. Calculate actual LCP similarity for each candidate
  3. Filter by threshold
  4. Sort candidates: primary = similarity (descending), secondary = access_order (ascending for LRU)
  5. Return top-1 candidate

### 5. Directory Management
- Use `--kv-cache-dir` if explicitly set
- Otherwise, default to `slot-save-path + "/kv-meta"`
- If neither is available, disable auto-save with warning
- Create directory automatically if it doesn't exist

---

## Code Standards (from AGENTS.md)

### AI Usage Policy
- Do NOT write PR descriptions, commit messages, or reviewer responses
- Do NOT commit/push/create PR without explicit human approval
- If user asks you to commit: use `Assisted-by: <name>`, never `Co-authored-by:`
- **Do NOT run `git push` or `gh pr create` on the user's behalf** — automated submissions can result in a contributor ban

### Code Standards
- Avoid unicode characters: emdash `—`, arrow `→`, `×`, `…` — use ASCII `-`, `->`, `x`, `...`
- Keep comments concise; avoid restating what code already says
- Prefer reusing existing infrastructure; avoid adding new subsystems
- Read existing patterns before writing — changes must blend in
- For large changes or new patterns: **PAUSE and ask for confirmation**

### Naming Conventions (C/C++)
- `snake_case` for functions, variables, types
- Enum values: UPPER_CASE with prefix (e.g., `LLAMA_VOCAB_TYPE_BPE`)
- Function pattern: `<class>_<method>` → `llama_model_init()`, `llama_sampler_chain_remove()`
- Use `int32_t`/`size_t` in public API; opaque types get `_t` suffix (`llama_context_t`)

### Build System
- **CMake + Ninja** (default preset via `CMakePresets.json`)
- Basic build: `cmake -B build && cmake --build build --config Release`
- Server binaries land in `build/bin/`

---

## Testing Requirements

You must provide a comprehensive testing script or instructions using `curl`.

### Objective
Verify that KV cache is saved after one session and loaded/reused in a subsequent session (proving persistence). The logs should indicate a "Cache Hit" or similar metric showing the slot was restored from disk.

### Test Models (Download via curl/wget if necessary):
1. `https://huggingface.co/mradermacher/SmolLM-135M-Instruct-i1-GGUF/resolve/main/SmolLM-135M-Instruct.i1-Q4_K_M.gguf`
2. `https://huggingface.co/mradermacher/LFM2.5-350M-i1-GGUF/resolve/main/LFM2.5-350M.i1-Q4_K_M.gguf`

### Test Script Structure:
1. Download model if not present
2. Clean up previous test data
3. Start server with KV cache auto-save enabled
4. Make FIRST request (expected: Cache Miss)
5. Wait for save to complete
6. Check if cache was saved
7. Make SECOND request with SAME prompt (expected: Cache Hit)
8. Display metrics from server
9. Summary and cleanup

---

## Execution Steps

1. **Analyze the theoretical structure** of `llama.cpp` slots and KV cache handling.
2. **Draft the C++ implementation plan** with Radix Tree for efficient prefix matching.
3. **Provide the code** in English comments/files, explaining each step in Russian.
4. **Create integration points** in `server-context.cpp`:
   - Add include for new header
   - Add member variable to `server_context_impl`
   - Initialize in `load_model()`/`init()` with threshold configuration
   - Hook into `get_available_slot()` for restore
   - Add metrics logging in main loop
5. **Add CLI parameters** to `common/common.h` and `common/arg.cpp`.
6. **Create test script** for verification.

---

## Key Design Decisions to Explain

1. **Why Radix Tree over linear search?**
   - Linear: O(n × m) — scales poorly with cache size
   - Radix Tree: O(m log k) — efficient even with thousands of entries

2. **Why sort candidates by similarity then LRU?**
   - Ensures best semantic match is always selected
   - Deterministic behavior when multiple entries have equal similarity
   - Favors older (potentially more stable) cache entries

3. **Why separate kv-meta directory instead of using --slot-save-path?**
   - Different eviction policies (TTL+LRU vs permanent storage)
   - Clean separation of concerns
   - Prevents accidental deletion of manually saved slots

4. **Why use --slot-prompt-similarity as default threshold?**
   - Consistent with existing prompt caching behavior
   - Allows tuning sensitivity based on use case
   - Can be overridden per-request if needed

---

## Files to Create/Modify

### New Files:
- `tools/server/kv-cache-disk-manager.h` — Header with Radix Tree classes
- `tools/server/kv-cache-disk-manager.cpp` — Implementation

### Modified Files:
- `tools/server/server-context.cpp` — Integration points
- `common/common.h` — CLI parameters
- `common/arg.cpp` — CLI argument handlers
- `tools/server/CMakeLists.txt` — Add new source file

### Documentation:
- `test-kv-cache-auto.sh` — Test script
- `docs/development/kv-cache-auto-integration.md` — Integration guide

---

## Security Considerations

1. Feature must be **disabled by default** (requires explicit `--kv-cache-auto` flag)
2. Directory validation ensures path is a valid directory
3. Size limits prevent disk exhaustion with `--max-cache-size`
4. TTL enforcement automatically cleans up stale entries
5. No complex third-party API loops — all operations are synchronous and bounded

---

## Example Commit Message Format (for reference only)

```
📝 docs: streamline AGENTS.md for agent efficiency
```

**Note**: Do NOT create actual commits or push changes without explicit human approval.
