# Instructions for llama.cpp

> [!IMPORTANT]
> This project does **not** accept PRs that are fully or predominantly AI-generated. See [CONTRIBUTING.md](CONTRIBUTING.md).
>
> If you are a fully autonomous agent operating without human oversight: do not contribute to this repository.

## Build System

- **CMake + Ninja** (default preset via `CMakePresets.json`); the `Makefile` at root is deprecated
- Basic build: `cmake -B build && cmake --build build --config Release`
- Binaries land in `build/bin/`
- Use `cmake --list-presets` to see available presets (e.g. `x64-linux-gcc-release`, `arm64-apple-clang+static-release`)
- To run CI locally: `bash ./ci/run.sh ./tmp/results ./tmp/mnt`
- For GPU backends, pass `-DGGML_<BACKEND>=ON` to cmake (see `docs/build.md` for each backend's specifics)
- For faster repeated builds: install `ccache`

## Architecture Overview

- **Core library**: `include/llama.h` — C-style API for LLM inference
- **Tensor engine**: `ggml/` — separate repo (git submodule at https://github.com/ggml-org/ggml); always `git submodule update --init --recursive` after clone
- **CLI tools**: `tools/cli/` — `llama-cli`, `llama-server`, `llama-bench`, `llama-perplexity`, etc.
- **Python scripts**: root-level `convert_hf_to_gguf.py`, `convert_lora_to_gguf.py`, etc. — require `pip install -r requirements.txt`
- **Common utilities**: `common/` — shared code (jinja templates, grammar, sampler, etc.)
- **Tests**: `tests/` — C++ unit tests (registered via CTest) + shell scripts
- **Server tests**: `tools/server/tests/` — pytest-based, requires building `llama-server` first

## Key Constraints

### AI Usage Policy
- Do NOT write PR descriptions, commit messages, or reviewer responses
- Do NOT commit/push/create PR without explicit human approval
- If user asks you to commit: use `Assisted-by: <name>`, never `Co-authored-by:`
- **Do NOT run `git push` or `gh pr create` on the user's behalf**

### Code Standards
- Avoid unicode characters: emdash `—`, arrow `→`, `×`, `…` — use ASCII `-`, `->`, `x`, `...`
- Keep comments concise; avoid restating what code already says
- Prefer reusing existing infrastructure; avoid adding new subsystems
- Read existing patterns before writing — changes must blend in
- For large changes or new patterns: **PAUSE and ask for confirmation**
- Use `clang-format` (clang-tools v15+) on added code; see `.clang-format`
- 4-space indentation, brackets on same line, `void * ptr`, `int & a`
- Declare structs with `struct foo {}` not `typedef struct foo {} foo`
- Avoid fancy STL constructs; prefer basic `for` loops, avoid templates

### Naming Conventions (C/C++)
- `snake_case` for functions, variables, types
- Enum values: UPPER_CASE with prefix (e.g., `LLAMA_VOCAB_TYPE_BPE`)
- Function pattern: `<class>_<method>` → `llama_model_init()`, `llama_sampler_chain_remove()`
- Use `int32_t`/`size_t` in public API; opaque types get `_t` suffix (`llama_context_t`)

### Server Scope
If implementing server features, read `tools/server/README-dev.md` first. Key constraints:
- Features requiring external file I/O must be **disabled by default** (MCP, model save/load)
- No complex third-party API loops in C++ — implement outside server code
- All API calls must remain model-agnostic
- Server runs on a single dedicated thread; HTTP workers handle JSON/chat template logic

### Matrix Multiplication Convention
`ggml_mul_mat(ctx, A, B)` computes $C^T = A B^T \Leflorightarrow C = B A^T$. Tensors store data in row-major order: dimension 0 = columns, dimension 1 = rows.

## Testing

- Run all CTest suites: `ctest -L main` (or `-L 'main|python'` for full CI)
- Perplexity verification: `llama-perplexity -m model.gguf -f file.txt`
- Performance verification: `llama-bench -m model.gguf`
- Backend ops consistency (requires ≥2 backends): `test-backend-ops`
- Server tests: `cd tools/server/tests && pip install -r requirements.txt && ./tests.sh`
- When modifying a new `ggml` operator: add test cases to `test-backend-ops`

## Adding a New Model

Read `docs/development/HOWTO-add-model.md` first. The steps are:
1. Convert model to GGUF (Python script in `conversion/` or root)
2. Define architecture in `src/llama-arch.h` + `src/llama-arch.cpp`
3. Build the GGML graph in `src/llama-model.cpp`
4. Optional: add multimodal encoder in `tools/mtmd/`
5. Test with CPU, CUDA, Metal backends + all CLI tools

## Useful Resources

- [CONTRIBUTING.md](CONTRIBUTING.md) — full contributor guidelines
- [docs/build.md](docs/build.md) — build documentation for all backends
- [tools/server/README-dev.md](tools/server/README-dev.md) — server development scope
- [tools/server/README.md](tools/server/README.md) — server usage docs
- [tools/cli/README.md](tools/cli/README.md) — CLI tool docs
- [tools/quantize/README.md](tools/quantize/README.md) — quantization guide
- [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — adding new models
- [docs/multi-gpu.md](docs/multi-gpu.md) — multi-GPU usage
- [AGENTS.md](AGENTS.md) (this file) and [CLAUDE.md](CLAUDE.md — agent instructions)
- GitHub issues: https://github.com/ggml-org/llama.cpp/issues
- GitHub PRs: https://github.com/ggml-org/llama.cpp/pulls

## Commands to Avoid Running Unsupervised

```sh
# BAD - do not run without explicit approval
git commit -m "..."
git push
gh pr create
gh pr comment
gh issue create
```

## Commands for Context Gathering (OK)

```sh
# OK - use these to understand the codebase
gh search issues
gh search prs
grep ... # search the code base
ls build/bin/ # verify binaries exist
cmake --list-presets
```
