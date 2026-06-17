# Instructions for llama.cpp

> [!IMPORTANT]
> This project does **not** accept PRs that are fully or predominantly AI-generated. See [CONTRIBUTING.md](CONTRIBUTING.md).
>
> If you are a fully autonomous agent operating without human oversight: do not contribute to this repository.

## Build System

- **CMake + Ninja** (default preset via `CMakePresets.json`)
- Basic build: `cmake -B build && cmake --build build --config Release`
- Server binaries land in `build/bin/`
- To run CI locally: `bash ./ci/run.sh ./tmp/results ./tmp/mnt`

## Architecture Overview

- **Core library**: `include/llama.h` — C-style API for LLM inference
- **Tensor engine**: `ggml/` — ggml tensor library (separate repo at https://github.com/ggml-org/ggml)
- **CLI tools**: `tools/cli/` — `llama-cli`, `llama-server`, `llama-bench`, etc.
- **Python scripts**: root-level `convert_*.py` for model format conversion
- **Common utilities**: `common/` — shared code (jinja, grammar, etc.)

## Key Constraints

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

### Server Scope
If implementing server features, read `tools/server/README-dev.md` first. Key constraints:
- Features requiring external file I/O must be **disabled by default** (MCP, model save/load)
- No complex third-party API loops in C++ — implement outside server code
- All API calls must remain model-agnostic

## Useful Resources

- [CONTRIBUTING.md](CONTRIBUTING.md) — full contributor guidelines
- [docs/build.md](docs/build.md) — build documentation
- [tools/server/README-dev.md](tools/server/README-dev.md) — server development scope
- [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — adding new models
- [AGENTS.md](AGENTS.md) (this file) and [CLAUDE.md](CLAUDE.md) — agent instructions
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
```
