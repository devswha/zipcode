# cli — clap, REPL, fullscreen TUI, doctor, setup

The `zipcode` (cli) crate — the user-facing entrypoint. Depends on `runtime` and re-wires `tools` at startup.

**Crate path:** [`crates/cli/`](../../crates/cli/)

---

## Module layout

**EXTRACTED** `crates/cli/src/`

| File | Owns |
|------|------|
| `main.rs` | clap parser + subcommand dispatch |
| `commands.rs` | `doctor`, `setup`, `run_default` |
| `repl.rs` | `run_interactive_with_ui()`, `run_oneshot()`, model resolution, engine creation |
| `tui.rs` | Fullscreen TUI with `termimad` + `rustyline` |
| `tui_composer.rs` | Message composition UI |
| `render.rs` | Terminal rendering helpers |

---

## Global CLI flags

**EXTRACTED** `main.rs:12-58`

| Flag | Purpose |
|------|---------|
| `--model <PATH>` | Override model path (takes precedence over config) |
| `--permission-mode <MODE>` | `read-only` / `workspace-write` / `full-access` |
| `--backend <BACKEND>` | `llama-cpp` (default compile flag) / `llama-server` / `candle` |
| `--ui <UI_MODE>` | `plain` / `fullscreen` (default: fullscreen, per recent commit `e2d22bd`) |

---

## Subcommands

**EXTRACTED** `main.rs`

| Subcommand | Behavior |
|-----------|---------|
| `repl` | Interactive REPL (default if no subcommand given) |
| `prompt <TEXT>` | One-shot: send prompt, stream reply, exit |
| `doctor` | System health report (models, tokenizer, llama-server, config) |
| `setup [--skip-smoke]` | Guided setup: find models, find llama-server, write config, smoke-test |
| `update [--check] [--rebuild]` | Inspect git update status, or fast-forward + rebuild when safe |

**INFERRED** — if no args are passed, recent commit `29c436a` made bare paths not be misread as REPL commands, and `59610ad` restored helper-backed setup flows. Run `zipcode --help` for the live help.

---

## `doctor`

**EXTRACTED** `commands.rs`

Classifies state as:

| State | Meaning |
|-------|---------|
| `Ready` | All prerequisites present |
| `NeedsSetup` | No config / no models — user should run `zipcode setup` |
| `NeedsRepair` | Partial state — some files missing |

Reports:
- Model file existence (from config or scan of `~/.zipcode/models`)
- Tokenizer file (if `candle` or `llama-cpp` backend)
- `llama-server` binary location (env var → PATH)
- Current config values

**Current readiness rules that matter for Gemma 4:**
- bare `zipcode`, explicit `zipcode repl`, and `zipcode prompt ...` all route through the same startup readiness guidance instead of surfacing raw missing-model-directory errors
- bare `zipcode` startup and `zipcode doctor` use the same helper discovery/fallback logic
- a stale saved `llama_server_bin` no longer blocks a valid bundled/PATH helper fallback
- explicit `--backend candle` / `--backend llama-cpp` will not be reported as Gemma 4 ready just because a helper exists elsewhere
- helper-backed GPU configs get a lightweight preflight (`--list-devices` when supported) so obviously unrunnable setups are surfaced before startup says `Ready`
- a helper probe that hangs is timed out and reported as a backend-readiness problem instead of blocking `doctor`/startup indefinitely

---

## `setup`

**EXTRACTED** `commands.rs`

1. Discover models in `~/.zipcode/models`.
2. Locate `llama-server`:
   - `ZIPCODE_LLAMA_SERVER_BIN` env var (absolute path)
   - Fallback: `which llama-server` on PATH
3. Optional smoke prompt (`--skip-smoke` to bypass).
4. Write `~/.zipcode/config.json`.
5. Create `~/.zipcode/bin/zipcode-local` wrapper script (shell shim for convenience).

Used by `install.sh` to bootstrap a fresh machine.

**INFERRED:** the root installer now only writes aggressive Gemma 4 helper GPU defaults when the helper can answer a quick capability probe, which keeps install-time defaults aligned with later readiness checks. `scripts/lib/install_common.sh:99`, `scripts/lib/install_common.sh:106`, `scripts/lib/install_common.sh:111`

---

## `update`

**EXTRACTED** `commands.rs`

- `zipcode update --check` prints the local checkout state and reports whether an update can be applied.
- `zipcode update` fast-forwards with `git pull --ff-only`, optionally rebuilds, then runs `zipcode doctor`.
- A **dirty working tree blocks both commands before any `git fetch`** so local modifications are reported as the blocker even if the network or remote would fail afterward.

---

## REPL initialization

**EXTRACTED** `repl.rs` (file layout; exact line numbers vary)

1. Parse CLI args.
2. Load `ZipcodeConfig::load(cwd)` — see [config](config.md).
3. Resolve model path (flag > project config > global config > scan `model_dir`).
4. Detect Gemma 4 by filename pattern.
5. Build `ServerOptions` from env vars + config (`ZIPCODE_GPU_LAYERS`, `ZIPCODE_FLASH_ATTENTION`, `DEFAULT_CONTEXT_SIZE = 8192`).
6. Call `create_engine(backend, model_path, tokenizer_path, config, server_options)` — returns `Box<dyn InferenceProvider>`.
7. Build `ToolRegistry` with all 10 tools from `zipcode-tools`.
8. Build system prompt via `prompt::build(...)` — see [conversation-loop › run_turn](conversation-loop.md#run_turn-flow) for how it's consumed.
9. Create `Session::new()` and persist it immediately so `/session` can resume a fresh startup session without waiting for the first model turn.
10. Construct [`ConversationLoop`](conversation-loop.md).
11. Enter TUI (`tui.rs`) or plain REPL depending on `--ui`.

**Session / slash-command details that matter in practice:**
- `/clear` creates a brand-new session **and saves it immediately**, so the printed id is resumable right away.
- `/session <id>` trims surrounding whitespace before loading.
- Failed `/session <id>` loads report an inline error and keep the interactive REPL alive.
- Known slash commands reject unexpected trailing arguments (`/status extra`, `/clear now`, etc.) instead of falling through to model input.

---

## Fullscreen TUI

**EXTRACTED** `tui.rs`

- `termimad` for markdown rendering of assistant output.
- `rustyline` for input line editing + history.
- Slash commands: `/help`, `/status`, `/clear`, `/quit`, `/doctor`, etc.
- Transcript navigation supports both `PgUp` / `PgDn` page jumps and mouse-wheel line scrolling in fullscreen mode.
- `/compact` may legitimately be a no-op; the status line now says `Compaction skipped` instead of claiming success when there is not enough history to compact.
- Automation hook: `ZIPCODE_TUI_AUTOMATION_SCRIPT` env var lets integration tests drive the TUI non-interactively (commit `52a7717`, `e2d22bd`).

---

## Tests

**EXTRACTED** — `crates/cli/tests/smoke.rs` currently exposes 40 smoke tests (`cargo test -p zipcode --test smoke -- --list` on 2026-04-14).

Notable newer regression guards added since the earlier snapshot include:

| Area | Example tests |
|------|---------------|
| Dirty-tree update gating before any fetch | `crates/cli/tests/smoke.rs:521`, `crates/cli/tests/smoke.rs:558` |
| Missing-model startup guidance for explicit `prompt` / `repl` subcommands | `crates/cli/tests/smoke.rs:628`, `crates/cli/tests/smoke.rs:712` |
| Stale helper-path fallback discovery | `crates/cli/tests/smoke.rs:905` |
| Unrunnable helper GPU-offload config surfaced in `doctor` and bare startup | `crates/cli/tests/smoke.rs:1018`, `crates/cli/tests/smoke.rs:1064` |
| Project-local config parse errors point to the right file | `crates/cli/tests/smoke.rs:1208` |
| Root installer remains `Ready` when multiple existing models are available | `crates/cli/tests/smoke.rs:1840` |
| Plain REPL session-load failures stay inline instead of exiting | `crates/cli/tests/smoke.rs:2070` |
| `/clear` persists the new session immediately | `crates/cli/tests/smoke.rs:2113` |
| Fullscreen `/compact` reports skipped/no-op honestly | `crates/cli/tests/smoke.rs:2194` |

All tests use temp `HOME` directories to avoid side effects.

---

## Related pages

- [config](config.md) — loaded here at startup
- [conversation-loop](conversation-loop.md) — constructed here, driven by the TUI
- [tools](tools.md) — registry built here
- [llama-server](llama-server.md) — consumes the `ServerOptions` constructed here
