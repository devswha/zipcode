# AGENTS.md - zipcode

**Generated:** 2026-04-03  
**Project Type:** Rust workspace (4 crates) - Local-only AI coding agent  
**Parent:** Root

---

## Project Purpose

zipcode is a Rust-based **local-only AI coding agent** designed for air-gapped environments. It runs Gemma 4 models via GGUF inference using either candle or llama-cpp backends, providing Claude Code-like functionality (file editing, shell execution, code search) offline by default. The explicit exception is user-requested GitHub repository fetching via `fetch_repo`. Deployable as a single binary or packaged ZIP for USB-based transfer.

**Key Characteristics:**
- No API keys. No cloud dependency; GitHub repo fetch is an explicit user-requested network action.
- Supports CUDA acceleration for GPU inference.
- Single static binary with embedded models.
- 11 built-in tools with permission-based access control.
- Agentic conversation loop with streaming generation.

---

## Workspace Structure

```
zipcode/
 ├── crates/              (workspace members: inference, tools, runtime, cli)
 ├── docs/                (design specs, implementation plans)
 ├── scripts/             (download_model.sh, package.sh, install.sh)
 ├── models/              (GGUF model storage, gitignored)
 ├── Cargo.toml           (workspace definition)
 ├── CLAUDE.md            (project conventions + stack)
 ├── README.md            (user guide + features)
 ├── .zipcode.md          (project-specific instructions, injected into system prompt)
 ├── .zipcode.json        (project config, overrides ~/.zipcode/config.json)
 └── .gitignore           (ignores: /target, *.gguf, models/, .env)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `Cargo.toml` | Workspace root; defines members, shared dependencies, workspace lints |
| `CLAUDE.md` | Stack, build/test commands, architecture diagram, key patterns, known limitations |
| `README.md` | User guide, quick start, air-gapped deployment workflow, CLI reference, configuration |
| `.gitignore` | Excludes /target, *.gguf files, .env, model binaries |
| `Cargo.lock` | Locked dependency versions (committed) |

---

## Subdirectories

| Directory | Contents | Notes |
|-----------|----------|-------|
| `crates/` | 4 Rust crate members (inference, tools, runtime, cli) | See crates/AGENTS.md |
| `docs/` | Design specifications, implementation plans | Architecture diagrams, feature proposals |
| `scripts/` | Helper scripts (download_model.sh, package.sh, install.sh) | Build automation, model setup |
| `models/` | GGUF model files (gitignored) | Downloaded at ~/.zipcode/models/ or ./models/ |

---

## Workspace Dependencies

**Workspace-managed (shared across all crates):**
```toml
serde, serde_json, anyhow, thiserror, tokio, tracing, tracing-subscriber
```

**Workspace lints:**
- `unsafe_code = "forbid"` — No unsafe code allowed
- `clippy::all = "warn"` — All clippy lints (pedantic, module_name_repetitions allowed)

**Crate-specific notable dependencies:**
- `zipcode-inference`: candle (0.8), llama-cpp-2 (0.1.141) - feature-gated
- `zipcode-tools`: glob, regex, grep-regex, grep-searcher, wait-timeout
- `zipcode-runtime`: uuid, chrono, dirs (config/session storage)
- `zipcode-cli`: clap, rustyline, termimad (TUI/CLI)

---

## Build Commands

```bash
# Build CLI with candle backend (default)
cargo build --release

# Build with llama-cpp backend (requires cmake + libclang-dev)
cargo build --release -p zipcode-inference --features llama-cpp

# Build CLI after enabling llama-cpp in crates/cli/Cargo.toml
cargo build --release -p zipcode

# Test all crates
cargo test --workspace

# Lint (must pass with zero warnings)
cargo clippy --workspace --all-targets -- -D warnings

# Format check
cargo fmt --all -- --check

# Format files
cargo fmt --all
```

---

## Test Commands

```bash
# Run all tests (workspace-wide)
cargo test --workspace

# Run tests for a single crate
cargo test -p zipcode-inference
cargo test -p zipcode-tools
cargo test -p zipcode-runtime
cargo test -p zipcode

# Run with logging (RUST_LOG env var)
RUST_LOG=debug cargo test --workspace -- --nocapture

# Run single test by name
cargo test --workspace test_name_substring
```

---

## Lint & Format

```bash
# Clippy (must show zero warnings)
cargo clippy --workspace --all-targets -- -D warnings

# Check format
cargo fmt --all -- --check

# Auto-format
cargo fmt --all
```

---

## Key Patterns & Conventions

### Inference Abstraction
- **Trait:** `InferenceProvider` abstracts candle vs llama-cpp vs mock backends
- **Generic:** `ConversationLoop<T: InferenceProvider>` for type-safe backend switching
- **Testing:** `MockInferenceProvider` queues predetermined responses

### Tool System
- **Trait:** `Tool` with `execute()` method, `ToolRegistry` router
- **Permissions:** `PermissionMode` (read-only, workspace-write, full-access) gates tool execution
- **Truncation:** Tool output auto-truncated at 8 KB (`MAX_TOOL_OUTPUT_BYTES`)
- **Safety:** All file tools use `resolve_and_validate_path()` for path traversal prevention

### Conversation Loop
- **Cap:** Max 25 tool iterations per conversation (`MAX_TOOL_ITERATIONS`)
- **Chat Template:** Hardcoded Gemma format (`<start_of_turn>/<end_of_turn>`, `<tool_call>` tags)
- **Streaming:** `generate_stream()` queues events, returns async receiver

### Storage Layout
- **Config:** `~/.zipcode/config.json` (global) + `.zipcode.json` (project override)
- **Models:** `~/.zipcode/models/` or `./models/`
- **Sessions:** `~/.zipcode/sessions/<uuid>.json`
- **Instructions:** `.zipcode.md` injected into system prompt

### Dependency Graph
```
cli
 ├── runtime
 │   ├── inference
 │   └── tools
 └── tools
```

---

## Known Limitations

1. **Gemma 4 GGUF Unsupported (temporarily)**
   - `llama-cpp-rs` 0.1.141 bundles old llama.cpp lacking Gemma 4 support
   - Workaround: Wait for llama-cpp-2 0.1.142+ or use llama-server subprocess backend

2. **Candle Backend Incomplete**
   - candle 0.8 has no `quantized_gemma` module
   - Currently using `quantized_llama` as stand-in (compiles but doesn't load Gemma GGUF)

3. **Chat Template Gemma-Only**
   - Other models (Qwen, Llama) use different tool-calling formats
   - Models not trained on `<tool_call>` tags won't trigger tool execution

4. **No True Streaming**
   - `generate_stream()` is synchronous, queues all events, returns receiver
   - No first-token latency improvement yet

5. **Agent Tool Stub**
   - Returns "not yet implemented"
   - Planned for multi-agent task delegation

---

## Development Workflow

### Before Committing
1. Run `cargo test --workspace` — all tests must pass
2. Run `cargo clippy --workspace --all-targets -- -D warnings` — zero warnings
3. Run `cargo fmt --all` — auto-format code
4. Verify no `.gguf`, `.env`, or `/target` in git

### Commit Message Format
```
feat: add new feature
fix: fix bug
test: add tests
docs: update documentation
chore: internal cleanup
```

### AI Agent Instructions

When working with this codebase:

1. **Understand the trait-based architecture** — All backends implement `InferenceProvider`, all tools implement `Tool`. Look for trait definitions first.

2. **Check feature flags** — candle vs llama-cpp are mutually exclusive. Verify which is enabled in Cargo.toml before suggesting compilation options.

3. **Respect permission boundaries** — Tool execution is gated by `PermissionMode`. Some tools are blocked in read-only mode; verify before implementing features.

4. **Path safety is critical** — All file operations must go through `resolve_and_validate_path()`. Never allow untrusted paths.

5. **Tool output size limits** — All tool results are truncated at 8 KB. Plan response summarization accordingly.

6. **Conversation loop bounds** — Max 25 iterations per conversation. Monitor loop counters.

7. **Test with MockInferenceProvider** — When writing integration tests, use MockInferenceProvider to avoid model dependencies.

8. **Check known limitations** — Gemma 4 GGUF, candle backend, chat template are partially broken. Document workarounds.

9. **Build targets matter** — `cargo build -p zipcode` builds the CLI binary; `-p zipcode-inference` tests inference independently.

10. **Documentation injection** — Project instructions from `.zipcode.md` are injected into the system prompt. Keep that file current.

---

## Project-Specific Instructions (.zipcode.md)

The `.zipcode.md` file in the project root contains project-specific instructions automatically injected into the system prompt. Keep it updated with:
- Build commands
- Testing strategy
- File conventions
- Known gotchas
- Contribution guidelines

---

<!-- MANUAL -->

## Manual Maintenance

- **Dependency updates:** Check `Cargo.toml` workspace.dependencies for llama-cpp-2, candle versions
- **Known issue tracking:** See CLAUDE.md "Current Limitations" section for real-time blockers
- **Chat template changes:** If supporting new models, update hardcoded Gemma template in inference crate
- **Permission gates:** When adding new tools, add PermissionMode checks in tool executor

<!-- /MANUAL -->
