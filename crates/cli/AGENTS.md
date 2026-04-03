# AGENTS.md - zipcode-cli

**Generated:** 2026-04-03  
**Crate Type:** Binary (REPL + one-shot CLI interface)  
**Parent:** ../AGENTS.md

---

## Purpose

The CLI crate is the **user-facing binary entry point** for zipcode. It provides three interaction modes:

1. **Interactive REPL** (default) — Multi-turn conversation with slash commands for session management
2. **One-shot mode** — Single prompt execution via `zipcode prompt "text"`
3. **Doctor mode** — System health check for model files, CUDA, and dependencies

The CLI handles:
- Command-line argument parsing (clap)
- REPL loop with line editing (rustyline)
- Tool registration and streaming callback dispatch
- Markdown rendering with ANSI styling (termimad)
- Session initialization and permission setup

---

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/main.rs` | 72 | CLI entry point, clap argument parsing, command routing |
| `src/repl.rs` | 261 | Interactive REPL loop, one-shot mode, tool registry construction |
| `src/render.rs` | 64 | ANSI markdown rendering, tool invocation display |
| `src/commands.rs` | 136 | Doctor command: version check, CUDA detection, model file discovery |
| `tests/smoke.rs` | 72 | 4 smoke tests: doctor, help, version, prompt graceful error handling |

---

## CLI Flags & Arguments

### Global Flags (all modes)

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--model <PATH>` | PathBuf | None | Override model file location; searches ~/.zipcode/models if unset |
| `--permission-mode <MODE>` | String | From config | read-only, workspace-write, or full-access |
| `--backend <BACKEND>` | String | llama-cpp | Inference engine: llama-cpp or candle |

### Subcommands

| Command | Argument | Purpose |
|---------|----------|---------|
| `prompt` | `<TEXT>` | Run single prompt and exit (one-shot mode) |
| `doctor` | (none) | Check system health: version, CUDA, model files |
| (default) | (none) | Start interactive REPL |

### Help & Version

| Flag | Purpose |
|------|---------|
| `--help` | Show usage information |
| `--version` | Show binary version |

---

## Subcommands & Modes

### 1. Interactive REPL (Default)

**Entry:** `zipcode`

**Behavior:**
- Prints welcome message with version
- Loads `.zipcode.json` project config or uses defaults
- Builds tool registry (9 standard tools + ToolSearchTool)
- Initializes ConversationLoop with inference engine
- Loops on user input with line editing

**Exit:** Ctrl+D (EOF) or `/quit` command

### 2. One-Shot Prompt

**Entry:** `zipcode prompt "your prompt text"`

**Behavior:**
- Runs single inference turn
- Prints streamed output to stdout
- Tool invocations logged to stderr
- Exits immediately after response

**Use case:** Shell scripts, batch processing, integration testing

### 3. Doctor Command

**Entry:** `zipcode doctor`

**Behavior:**
- Prints binary version
- Checks CUDA availability (checks CUDA_PATH env, libcuda.so, ldconfig)
- Scans default model directories (~/.zipcode/models, ./models)
- Lists all .gguf files found
- Provides helpful error messages if models missing

**Exit codes:** 0 (success, may have warnings)

---

## Slash Commands (REPL Only)

Entered as `/command` in interactive mode. All slash commands are processed before sending to the model.

| Command | Purpose |
|---------|---------|
| `/help` | Show available commands |
| `/status` | Display session ID, message count, tool count, working directory |
| `/clear` | Clear conversation history (creates new Session) |
| `/quit` or `/exit` | Exit the REPL gracefully |

**Keyboard controls:**
- `Ctrl+D` — Exit (same as `/quit`)
- `Ctrl+C` — Cancel current input (continue REPL)
- Arrow keys — Navigate history (rustyline)

---

## Tool Registration

**Function:** `repl::build_registry()` (repl.rs, lines 56-75)

Registers **10 tools** in order:

1. BashTool — Shell execution
2. ReadFileTool — File reading
3. WriteFileTool — File creation/overwrite
4. EditFileTool — Line-based file editing
5. GlobSearchTool — Filename pattern matching
6. GrepSearchTool — Content text search
7. TodoWriteTool — Task/todo tracking
8. ReplTool — Python REPL for data analysis
9. AgentTool — Multi-agent task delegation (stub)
10. ToolSearchTool — Tool discovery (built from registry)

ToolSearchTool is constructed last using specs from the first 9 tools.

---

## Streaming Callback System

**Trait:** `StreamCallback` (implemented as `CliCallback` in repl.rs)

Routes inference events to terminal rendering:

| Callback | Source | Behavior |
|----------|--------|----------|
| `on_token(text)` | Streamed generation | Print to stdout, flush immediately |
| `on_tool_start(name, args)` | Model tool invocation | Print yellow ">" indicator + tool name + args preview |
| `on_tool_result(name, result)` | Tool completion | Print green "<" indicator + tool name + result preview (200 chars) |
| `on_permission_prompt(msg)` | Permission gating | Print yellow "[permission]" + message, prompt user [Y/n] |
| `on_error(error)` | Runtime error | Print red "error:" + message to stderr |

**ANSI colors used:**
- Green (\x1b[32m) — Tool results
- Yellow (\x1b[33m) — Tool starts, permission prompts
- Red (\x1b[31m) — Errors
- Cyan (\x1b[36m) — Markdown headers (create_skin)

---

## Configuration Loading

**Function:** `repl::create_loop()` (repl.rs, lines 87-156)

1. Load config from `.zipcode.json` (project) or `~/.zipcode/config.json` (global)
2. If load fails, warn and use ZipcodeConfig::default()
3. Resolve model file:
   - Use `--model <PATH>` if provided
   - Else use config.model_file if set
   - Else scan config.model_dir for first .gguf
   - Error if no model found
4. Load tokenizer from model directory (tokenizer.json)
5. Build GenerationConfig from config (temperature, top_p, max_tokens)
6. Create inference engine via create_engine(backend)
7. Build tool registry
8. Build system prompt with tool specs
9. Return ConversationLoop ready to use

---

## Rendering & Display

### Markdown Rendering

**File:** `src/render.rs`

- `create_skin()` — Builds termimad MadSkin with colored headers/bold/italic
- `render_markdown(text)` — Print markdown block to terminal with styling

**Colors:**
- L1 headers (Cyan), L2 headers (Blue), L3 headers (Green)
- Bold (Yellow), Italic (Magenta)

### Tool Display

**Functions:**
- `print_tool_start(name, args)` — Yellow ">" + bold tool name + args preview (truncated at 60 chars per arg)
- `print_tool_result(name, result)` — Green "<" + bold tool name + result preview (truncated at 200 chars)

**Example output:**
```
> Read(path="/home/user/file.rs")
< Read: use std::fs; fn main() { println!("hello"); }...
```

---

## Session Management

**In REPL:**
- Session object maintains message history and session ID
- `/clear` creates new Session::new(), preserving config
- `/status` prints session.id, message count, tool count, cwd

**Message flow:**
```
User input → conv.run_turn(input, &mut cb)
  → Inference engine streams tokens via on_token
  → Model outputs tool calls parsed as JsonValue
  → Tool invoked, on_tool_start/result called
  → Response added to session.messages
  → Loop for next user input
```

---

## Smoke Tests

**File:** `tests/smoke.rs` (4 tests)

| Test | Command | Validates |
|------|---------|-----------|
| `doctor_runs` | `zipcode doctor` | Exit 0, output contains "zipcode" |
| `help_flag` | `zipcode --help` | Exit 0, output contains "Usage" |
| `version_flag` | `zipcode --version` | Exit 0, output contains "0.1.0" or "zipcode" |
| `prompt_no_model_graceful_error` | `zipcode prompt "hello"` (no model) | No panic/backtrace, graceful error message |

All tests use the compiled binary via `env!("CARGO_BIN_EXE_zipcode")`.

---

## Error Handling

**Config loading failure:**
- Warns user with yellow "[Warning]" message
- Falls back to ZipcodeConfig::default()
- Continues execution

**Model not found:**
- Error message lists searched directories
- Suggests running `./scripts/download_model.sh`
- Exits with non-zero code

**REPL input errors:**
- Model inference errors printed to stderr with red "error:" prefix
- REPL loop continues, does not exit

**Permission prompts:**
- Yellow "[permission]" prompt
- User types Y/n
- Default is yes (empty input)

---

## Binary Output

**Binary name:** `zipcode` (defined in Cargo.toml)

**Release build:** `cargo build --release -p zipcode`

**Location:** `target/release/zipcode`

**Size:** ~5-10 MB (depends on model embedding and backend choice)

---

## AI Agent Instructions

### Adding a New Slash Command

1. Add pattern match in `repl::run_interactive()` around line 201
2. Implement handler function (e.g., `fn handle_new_command()`)
3. Add help text in `print_help()` function
4. Test with REPL manually and add smoke test if applicable

Example:
```rust
"/newcmd" => {
    println!("Output here");
}
```

### Adding a New Subcommand

1. Add variant to `Commands` enum in main.rs (lines 29-38)
2. Add routing in `cli.command` match (lines 57-68)
3. Implement handler function (e.g., `commands::new_subcommand()`)
4. Update help text in clap command attribute
5. Test with `zipcode newcmd [args]`

### Registering a New Tool

1. Implement `Tool` trait in zipcode-tools crate
2. Add `registry.register(Box::new(MyNewTool))` in `build_registry()` (repl.rs, line 56-75)
3. Verify tool spec appears in system prompt via `/status` session info
4. Add permission checks if needed in tool execute method

### Modifying Inference Behavior

1. Check feature flags in Cargo.toml (llama-cpp vs candle)
2. Update `Backend::from_name()` if adding new backend option
3. Update `--backend` flag documentation
4. Test with `zipcode --backend newbackend prompt "test"`

### Updating CLI Flags

1. Edit `Cli` struct in main.rs (lines 10-27)
2. Use clap derives for documentation
3. Update help text in `#[arg]` attributes
4. Test with `zipcode --help`

### Debugging REPL Issues

Use `RUST_LOG=debug` environment variable:
```bash
RUST_LOG=debug zipcode
```

Logs are printed to stderr. Check for:
- Config loading errors
- Model file resolution
- Tool registration steps
- Permission policy decisions

---

## Dependencies

**Direct dependencies:**
- `clap` (4.x) — CLI argument parsing
- `rustyline` (15.x) — Line editing with history
- `termimad` (0.30.x) — Markdown ANSI rendering
- `dirs` (6.x) — Config/session directory discovery
- `anyhow`, `tokio`, `tracing`, `serde_json` — (workspace)

**Via zipcode-runtime:**
- uuid, chrono, dirs

**Via zipcode-inference:**
- candle or llama-cpp (feature-gated)

---

## Testing Strategy

**Unit tests:** None in CLI crate (thin wrapper)

**Integration tests:** 4 smoke tests in tests/smoke.rs
- Test binary invocation, not library functions
- Run against compiled artifact

**Manual testing:**
1. `cargo build -p zipcode`
2. `./target/debug/zipcode --help`
3. `./target/debug/zipcode doctor`
4. Download model: `./scripts/download_model.sh`
5. `./target/debug/zipcode` (interactive REPL)
6. Type `/help`, `/status`, `/clear`, then a prompt
7. Test one-shot: `./target/debug/zipcode prompt "hello"`

**To run tests:**
```bash
cargo test -p zipcode
RUST_LOG=debug cargo test -p zipcode -- --nocapture
```

---

## Known Limitations & TODOs

1. **Agent Tool Stub** — AgentTool returns "not yet implemented"; needs multi-agent delegation
2. **No True Async Streaming** — Line editor blocks during inference; consider async refactor
3. **Limited CUDA Detection** — Checks common paths; may fail on unusual CUDA installs
4. **No Config Validation** — Invalid permission_mode silently becomes full-access fallback
5. **Model Search Order** — First .gguf found in model_dir is used (non-deterministic if multiple models exist)

---

## Dependency Diagram

```
cli (binary)
 ├── runtime (ConversationLoop, PermissionPolicy, build_system_prompt)
 │   ├── inference (InferenceProvider, GenerationConfig)
 │   └── tools (Tool trait, ToolRegistry)
 ├── tools (9 tool implementations)
 ├── clap (CLI parsing)
 ├── rustyline (REPL editing)
 ├── termimad (Markdown rendering)
 └── dirs (Config paths)
```

---

## Build & Release

**Debug build:**
```bash
cargo build -p zipcode
```

**Release build (optimized):**
```bash
cargo build --release -p zipcode
```

**With llama-cpp backend:**
```bash
cargo build --release -p zipcode-inference --features llama-cpp
cargo build --release -p zipcode
```

**Binary location after build:**
- Debug: `target/debug/zipcode`
- Release: `target/release/zipcode`

---

## Manual Maintenance Notes

- **Help text:** Update `print_help()` if adding slash commands
- **Version:** Managed by workspace Cargo.toml
- **Status output:** Update `print_status()` if adding new Session fields
- **Smoke tests:** Add test if adding new CLI flag or subcommand
- **Error messages:** Keep user-friendly (avoid panic backtraces in main path)
