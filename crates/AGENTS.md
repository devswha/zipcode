# AGENTS.md - crates/

**Generated:** 2026-04-03  
**Project Type:** Rust workspace members (4 crates)  
**Parent:** ../AGENTS.md

---

## Container Purpose

The `crates/` directory is the workspace root containing **4 interdependent Rust crates** that together implement the zipcode local AI agent. Each crate is independently testable and buildable via `cargo test -p <name>` and `cargo build -p <name>`.

**Dependency Graph:**
```
cli
 ├── runtime ──→ inference
 │         └─→ tools
 └─→ tools
```

---

## Crate Summary Table

| Crate | Path | Responsibility | Key Types |
|-------|------|-----------------|-----------|
| **inference** | `inference/` | GGUF model loading, tokenization, streaming generation, KV cache, sampler | `InferenceProvider` trait, `CandeInferenceProvider`, `LlamaCppProvider`, `MockInferenceProvider` |
| **tools** | `tools/` | 10 tool implementations, `Tool` trait, `ToolRegistry`, result truncation | `Tool`, `ToolRegistry`, `ToolResult`, `Bash`, `ReadFile`, `WriteFile`, `EditFile`, `GlobSearch`, `GrepSearch`, `TodoWrite`, `REPL`, `Agent`, `ToolSearch` |
| **runtime** | `runtime/` | Agentic conversation loop, config hierarchy, permission policy, session persistence | `ConversationLoop`, `ConversationConfig`, `PermissionMode`, `SessionManager` |
| **cli** | `cli/` | Binary entry point, REPL (rustyline), slash commands, doctor command, ANSI rendering (termimad) | `main()`, `CliArgs`, `ReplHandler`, `DoctorCommand` |

---

## Crate Details

### 1. inference/

**Purpose:** Pure inference engine — GGUF model loading, tokenization, streaming token generation, KV cache management, temperature/top-p sampling.

**Key Files:**
- `src/lib.rs` — `InferenceProvider` trait definition
- `src/candle.rs` — candle backend implementation
- `src/llama_cpp.rs` — llama-cpp backend (feature-gated)
- `src/mock.rs` — `MockInferenceProvider` for testing
- `src/tokenizer.rs` — Tokenizer wrapper
- `src/sampler.rs` — Temperature, top-p sampling
- `src/chat.rs` — Gemma chat template formatting

**Key Types:**
```rust
pub trait InferenceProvider: Send + Sync {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    fn generate_stream(&self, tokens: Vec<u32>, config: GenerationConfig) 
        -> Result<tokio::sync::mpsc::Receiver<String>>;
}

pub struct CandeInferenceProvider { ... }
pub struct LlamaCppProvider { ... }
pub struct MockInferenceProvider { ... }
```

**Features:**
- Default: `candle` (HuggingFace candle 0.8)
- Optional: `llama-cpp` (llama-cpp-2 0.1.141)

**Dependencies:**
- candle: candle-core, candle-nn, candle-transformers (optional)
- llama-cpp: llama-cpp-2 (optional)
- Common: serde, anyhow, thiserror, tracing, encoding_rs, fastrand

**Test Commands:**
```bash
cargo test -p zipcode-inference
RUST_LOG=debug cargo test -p zipcode-inference -- --nocapture
```

**Known Issues:**
- Gemma 4 GGUF not loadable (llama-cpp-rs too old)
- candle has no quantized_gemma module (uses quantized_llama placeholder)

---

### 2. tools/

**Purpose:** Tool implementations — 10 autonomous tools (Bash, ReadFile, WriteFile, EditFile, GlobSearch, GrepSearch, TodoWrite, REPL, Agent, ToolSearch) with permission gating and output truncation.

**Key Files:**
- `src/lib.rs` — `Tool` trait, `ToolRegistry` router, `ToolResult`
- `src/bash.rs` — Shell execution via `bash -c`
- `src/file_ops.rs` — ReadFile, WriteFile, EditFile with path validation
- `src/search.rs` — GlobSearch, GrepSearch
- `src/todo.rs` — TodoWrite (JSON persistence)
- `src/repl.rs` — Python/Node.js REPL (subprocess)
- `src/agent.rs` — Agent stub (not yet implemented)
- `src/tool_search.rs` — ToolSearch by keyword
- `src/permissions.rs` — `PermissionMode` enforcement

**Key Types:**
```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&mut self, input: ToolInput) -> Result<ToolResult>;
    fn permissions(&self) -> PermissionLevel;
}

pub struct ToolRegistry { tools: HashMap<String, Box<dyn Tool>> }
pub struct ToolResult { 
    output: String,
    bytes: usize,  // truncated at 8 KB
}

pub enum PermissionMode { ReadOnly, WorkspaceWrite, FullAccess }
```

**Tool Specifications:**

| Tool | Permission | Purpose |
|------|-----------|---------|
| Bash | workspace-write approval | Execute shell commands |
| ReadFile | Always | Read file with offset/limit |
| WriteFile | workspace-write | Create/overwrite files |
| EditFile | workspace-write | Targeted string replacement |
| GlobSearch | Always | Find files by pattern |
| GrepSearch | Always | Search contents (regex) |
| TodoWrite | workspace-write | JSON todo persistence |
| REPL | workspace-write | Python/Node.js snippets |
| Agent | workspace-write | Delegated tasks (stub) |
| ToolSearch | Always | Search available tools |

**Safety Features:**
- Path traversal prevention via `resolve_and_validate_path()` in all file tools
- Output truncation at 8 KB max
- Permission gating on write/execution tools
- Subprocess timeout on Bash/REPL

**Dependencies:**
- glob, regex, grep-regex, grep-searcher (search)
- wait-timeout (subprocess timeout)
- Common: serde, anyhow, thiserror, tracing

**Test Commands:**
```bash
cargo test -p zipcode-tools
RUST_LOG=debug cargo test -p zipcode-tools -- --nocapture
cargo test -p zipcode-tools test_bash_execution
```

---

### 3. runtime/

**Purpose:** Agentic conversation loop — orchestrates inference provider, tool registry, config loading, permission policy, session persistence.

**Key Files:**
- `src/lib.rs` — `ConversationLoop` orchestrator
- `src/config.rs` — `ConversationConfig`, hierarchy (.zipcode.json > ~/.zipcode/config.json)
- `src/permissions.rs` — `PermissionMode` policy enforcement
- `src/session.rs` — `SessionManager`, conversation history persistence
- `src/loop.rs` — Main event loop (prompt → tokenize → generate → tool dispatch → repeat)

**Key Types:**
```rust
pub struct ConversationLoop<T: InferenceProvider> {
    inference: T,
    tools: ToolRegistry,
    config: ConversationConfig,
    session: SessionManager,
}

pub enum PermissionMode { ReadOnly, WorkspaceWrite, FullAccess }

pub struct ConversationConfig {
    permission_mode: PermissionMode,
    model_dir: PathBuf,
    generation: GenerationConfig,
}

pub struct SessionManager {
    session_id: Uuid,
    history: Vec<Message>,
}
```

**Conversation Loop Bounds:**
- Max 25 tool iterations per conversation (`MAX_TOOL_ITERATIONS`)
- Tool output truncated at 8 KB
- Chat template hardcoded for Gemma format
- Session history persisted at `~/.zipcode/sessions/<uuid>.json`

**Configuration Hierarchy:**
1. `.zipcode.json` (project root, highest priority)
2. `~/.zipcode/config.json` (global)
3. Compiled defaults

**Dependencies:**
- zipcode-inference, zipcode-tools (internal)
- uuid, chrono (session tracking)
- dirs (config/session paths)
- Common: serde, anyhow, thiserror, tokio, tracing

**Test Commands:**
```bash
cargo test -p zipcode-runtime
cargo test -p zipcode-runtime test_conversation_loop
RUST_LOG=debug cargo test -p zipcode-runtime -- --nocapture
```

---

### 4. cli/

**Purpose:** Binary entry point — REPL (rustyline), one-shot mode, doctor command, ANSI markdown rendering (termimad), CLI argument parsing (clap).

**Key Files:**
- `src/main.rs` — Entry point, argument parsing, mode dispatch
- `src/repl.rs` — Interactive REPL loop
- `src/one_shot.rs` — Single-prompt mode
- `src/doctor.rs` — Environment check (CUDA, model, binary version)
- `src/commands.rs` — Slash commands (/help, /status, /clear, /quit)

**Key Types:**
```rust
#[derive(Parser)]
struct CliArgs {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(long)]
    model: Option<PathBuf>,

    #[arg(long, default_value = "workspace-write")]
    permission_mode: PermissionMode,
}

enum Commands {
    Prompt { text: String },
    Doctor,
}
```

**CLI Interface:**
```bash
zipcode                              # Start REPL
zipcode prompt "explain this code"   # One-shot
zipcode doctor                       # Check environment
zipcode --model ./model.gguf         # Custom model
zipcode --permission-mode read-only  # Read-only mode
```

**Slash Commands (REPL):**
```
/help           Show available commands
/status         Session ID, message count, cwd
/clear          Reset conversation
/quit           Exit
```

**Doctor Command Output:**
```
zipcode doctor
--------------
  Binary:   zipcode v0.1.0 (linux-x86_64)
  CUDA:     Available
  Model:    gemma-4-27b-it-Q8_0.gguf (28.3 GB)
  Ready:    All checks passed
```

**Dependencies:**
- zipcode-runtime, zipcode-tools, zipcode-inference (internal)
- clap (CLI parsing)
- rustyline (REPL line editing)
- termimad (markdown to ANSI)
- Common: tokio, serde_json, anyhow, tracing

**Build & Run:**
```bash
# Build CLI (enables llama-cpp feature for zipcode-inference)
cargo build --release -p zipcode

# Run
./target/release/zipcode
./target/release/zipcode prompt "explain main.rs"
./target/release/zipcode doctor
```

**Test Commands:**
```bash
cargo test -p zipcode
cargo test -p zipcode test_cli_smoke
```

---

## Per-Crate Testing

Each crate has its own test suite. Run selectively:

```bash
# Test one crate
cargo test -p zipcode-inference
cargo test -p zipcode-tools
cargo test -p zipcode-runtime
cargo test -p zipcode

# Test all
cargo test --workspace

# Test with logging
RUST_LOG=debug cargo test -p zipcode-tools -- --nocapture

# Test single function
cargo test -p zipcode-tools test_bash_execution
```

---

## AI Agent Instructions

When working with crates:

1. **Understand trait boundaries** — Each crate exports core traits (InferenceProvider, Tool, ToolRegistry). Implementations extend these traits.

2. **Test crate-by-crate** — Each crate is independently buildable. Use `cargo test -p <name>` to isolate failures.

3. **Feature flags are critical** — candle and llama-cpp are mutually exclusive. Check Cargo.toml before suggesting features.

4. **Tool permissions matter** — Before implementing new tools, check PermissionMode enforcement in runtime crate.

5. **Path safety in file tools** — All file operations must use `resolve_and_validate_path()`. Never bypass this.

6. **Inference provider abstraction** — New backends should implement the InferenceProvider trait, not directly use candle/llama-cpp in runtime/cli.

7. **Output truncation is enforced** — Tools return ToolResult with 8 KB truncation. Plan response summarization.

8. **Conversation loop bounds** — Max 25 iterations. Monitor loop state.

9. **Session persistence** — ConversationLoop auto-saves history to ~/.zipcode/sessions/. Don't break this.

10. **CLI is minimal** — Keep main.rs light. Complex logic belongs in runtime/tools crates.

---

## Dependency Graph (Detailed)

```
zipcode-cli
  ├─ zipcode-runtime
  │   ├─ zipcode-inference (core computation)
  │   └─ zipcode-tools (tool execution)
  ├─ zipcode-tools (re-exported)
  └─ (CLI-specific: clap, rustyline, termimad)

zipcode-runtime
  ├─ zipcode-inference (inference provider)
  ├─ zipcode-tools (tool registry)
  └─ (common: tokio, tracing, serde, dirs)

zipcode-inference
  ├─ candle-* (feature-gated, default)
  ├─ llama-cpp-2 (feature-gated, optional)
  └─ (common: serde, anyhow, thiserror, tracing)

zipcode-tools
  ├─ glob, regex, grep-* (search)
  ├─ wait-timeout (subprocess)
  └─ (common: serde, anyhow, thiserror, tracing)
```

---

<!-- MANUAL -->

## Manual Maintenance

- **Add new crate:** Update workspace members in root Cargo.toml
- **Add new tool:** Implement Tool trait in tools/src/, register in ToolRegistry
- **Add new inference backend:** Implement InferenceProvider trait in inference/src/
- **Update chat template:** Edit inference/src/chat.rs for new model formats
- **Feature changes:** Update feature flags in crates/*/Cargo.toml and root Cargo.toml

<!-- /MANUAL -->
