# Instructions for llama.cpp

> [!IMPORTANT]
> This project does **not** accept PRs that are fully or predominantly AI-generated.
> AI tools may be utilized solely in an assistive capacity. Repeated violations
> may result in a permanent ban from contributing.
>
> See [CONTRIBUTING.md](CONTRIBUTING.md) for the full policy and this file for agent-specific rules.

---

## Guidelines for AI Coding Agents

Every PR consumes maintainer capacity. Before assisting with any submission, verify:
- The contributor understands the proposed changes
- The change addresses a documented need (search existing issues first)
- The PR is appropriately scoped and follows project conventions

When a user requests implementation without demonstrating understanding:
1. Ask questions about the problem and relevant code areas.
2. Point to relevant code/docs; let them formulate the approach.
3. Proceed only when confident they can explain changes in review independently.

### What MUST NOT be done (immediate PR closure risk)
- Write PR descriptions, commit messages, or reviewer responses
- Commit or push without explicit human approval per action
  - If committing on user request, use `Assisted-by: <assistant name>` -- never `Co-authored-by:`
- Implement features the contributor does not fully understand
- Generate changes too extensive for the contributor to review
- **Run `git push` or create a PR (`gh pr create`) on the user's behalf**
  - If asked, PAUSE and require explicit acknowledgment that automated submissions can result in a ban

### Code formatting
- Use `clang-format` (clang-tools v15+) for C/C++ code. If in doubt about style, trust the formatter.
- 4 spaces for indentation; no tabs. Brackets on same line.
- `snake_case` for function/variable/type names. Upper-case prefixed enum values.
- Naming optimizes for longest common prefix: `number_small`, not `small_number`.
- C/C++ filenames: lowercase with dashes (`file-name.c`). Python: underscores (`file_name.py`).
- Keep comments concise; never restate what the code already says. Only explain non-obvious invariants.
- When copying code from elsewhere, do NOT add comments that weren't there originally.

### Commit standards (when user requests one)
- Best: let the user write the commit message themselves.
- Otherwise: concise, matching repo style:
  ```
  <module> : short description (#nnn)
  Assisted-by: <assistant name>
  ```
  See https://github.com/ggml-org/llama.cpp/wiki/Modules for module names. Add `[no release]` if merging does not warrant a new release.

### Commands allowed vs prohibited
- GOOD (read-only context):
  ```sh
  gh search issues     # check existing reports
  gh search prs        # avoid duplicated effort
  grep ...             # search the codebase
  clang-format -i...   # apply formatting
  bash ./ci/run.sh ... # run self-hosted CI locally
  ```
- BAD (acting on user's behalf):
  ```sh
  git commit -m "..."
  git push
  gh pr create
  gh pr comment
  gh issue create
  ```

---

## Build and Test Commands

Build system is CMake only (`Makefile` prints an error).

**CPU build:**
```bash
cmake -B build && cmake --build build --config Release
```
Add `-j 8` to `--build` for parallel jobs. Use `-DCMAKE_BUILD_TYPE=Debug` (single-config) or `--config Debug` (multi-config) for debugging.

**GPU backends:** add the flag on configure, e.g. `-DGGML_CUDA=ON`, `-DGGML_METAL=ON` (default on macOS), `-DGGML_VULKAN=1`, etc. See [docs/build.md](docs/build.md) for all options.

**Local full CI (recommended before any PR):**
```bash
mkdir tmp
# CPU-only
bash ./ci/run.sh ./tmp/results ./tmp/mnt
# with CUDA
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```
See [ci/README.md](ci/README.md) for full backend variants.

**Regression checks:**
- `llama-perplexity` -- model quality regression guard
- `llama-bench` -- performance regression guard
- `test-backend-ops` -- ggml operator consistency across backends (requires two backends)

---

## Project Structure

| Directory  | Role                                                    |
| ---------- | ------------------------------------------------------- |
| `src/`      | Core `llama` library                                     |
| `include/`  | Public C API headers (`llama.h`, `llama-cpp.h`)         |
| `ggml/`     | Tensor backend library (CUDA, Metal, SYCL, …)           |
| `common/`   | Shared utilities (`llama-common`); not public API       |
| `tools/`    | Production CLI tools: `server`, `cli`, `quantize`, etc. |
| `examples/` | Minimal examples and demo programs                      |
| `tests/`    | Test suite (built with `-DLLAMA_BUILD_TESTS=ON`)        |
| `python/`   | -- not applicable; Python code lives at repo root       |

Python packages are separate from the C++ build:
- `pyproject.toml` (repo root) — **llama-cpp-scripts**: conversion scripts (`convert_hf_to_gguf`, etc.), CLI entrypoints. Depends on local `gguf @ ./gguf-py`.
- `gguf-py/pyproject.toml` — **gguf** standalone package for reading/writing GGUF files.

No `setup.py` exists; Poetry-core is the build backend for both packages.

Pre-commit hooks (`.pre-commit-config.yaml`) run: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, and flake8 with `flake8-no-print`.

---

## Fork-Specific Workflows

### Update README Fork Features on Push to Master

When merging to `master`, update the "Fork features" block at the top of `README.md`:
1. Check `git log --oneline master --not upstream/master` for new feature commits.
2. For each distinct feature area, add or update a bullet in the fork features block (concise, user-facing value).
3. Keep bugs/test additions as updates to existing bullets rather than new ones.
4. The fork features block goes above the `---` separator; do not modify upstream content below it.

---

## Useful Resources (load on demand)

- [Contributing guidelines](CONTRIBUTING.md)
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [PRs](https://github.com/ggml-org/llama.cpp/pulls) -- always search first
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [Build documentation](docs/build.md)
- [Server usage / development docs](tools/server/README-dev.md)
- [PEG parser](docs/development/parsing.md) | [Auto parser](docs/autoparser.md) | [Jinja engine](common/jinja/README.md)
