# AGENTS.md - crates/runtime/

**Generated:** 2026-04-03  
**Parent:** ../AGENTS.md

---

## Crate Purpose

The **runtime** crate implements the **agentic conversation loop** — the core orchestrator that combines inference generation with tool execution, permission checking, config hierarchy, and session persistence. It is the glue between the inference engine and tool registry, managing the multi-turn conversation flow where the model generates responses, parses tool calls, checks permissions, executes tools, and loops until completion.

**Tagline:** Inference + Tools + Config + Permissions + Sessions = Agentic Conversation Loop

---

## Key Responsibilities

| Responsibility | Files | Key Type |
|---|---|---|
| Main conversation loop orchestration | `conversation.rs` | `ConversationLoop` |
| Streaming token/tool callback interface | `conversation.rs` | `StreamCallback` trait |
| Config loading and hierarchy | `config.rs` | `ZipcodeConfig` |
| Permission checking and modes | `permission.rs` | `PermissionPolicy`, `PermissionCheck` |
| Session persistence (~/.zipcode/sessions/) | `session.rs` | `Session` |
| System prompt building with context | `prompt.rs` | `build_system_prompt()` |

---

## Files Overview

| File | Lines | Purpose |
|------|-------|---------|
| `src/lib.rs` | 11 | Public API exports |
| `src/conversation.rs` | 144 | `ConversationLoop::run_turn()` implementation, `StreamCallback` trait |
| `src/config.rs` | 134 | Config hierarchy loader, `ZipcodeConfig`, `GenerationOverrides` |
| `src/permission.rs` | 125 | `PermissionPolicy`, `PermissionCheck` enum, permission modes |
| `src/session.rs` | 96 | Session model, load/save to ~/.zipcode/sessions/ |
| `src/prompt.rs` | 87 | System prompt builder with .zipcode.md injection |
| `tests/integration.rs` | 316 | 6 integration tests with `MockInferenceProvider` |

**Total:** ~903 lines of code + tests

---

## Core Data Structures

### ConversationLoop

```rust
pub struct ConversationLoop {
    pub engine: Box<dyn InferenceProvider>,
    pub tools: ToolRegistry,
    pub session: Session,
    pub permission: PermissionPolicy,
    pub system_prompt: String,
    pub tool_specs: Vec<ToolSpec>,
    pub cwd: std::path::PathBuf,
}
```

**Responsibility:** Orchestrate a single conversation turn (prompt → generate → tool execution → repeat).

**Main Method:**
```rust
pub fn run_turn(&mut self, user_input: &str, callback: &mut dyn StreamCallback) -> Result<()>
```

---

### StreamCallback Trait

```rust
pub trait StreamCallback: Send {
    fn on_token(&mut self, text: &str);
    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value);
    fn on_tool_result(&mut self, name: &str, result: &str);
    fn on_permission_prompt(&mut self, message: &str) -> bool;  // Returns true if approved
    fn on_error(&mut self, error: &str);
}
```

**Purpose:** Allow UI/CLI layer to react to streaming tokens, tool invocations, permission prompts, and errors in real time.

**Implementations:**
- CLI: rustyline-based callback (stdout streaming)
- Tests: `TestCallback` in integration tests (collects tokens/tool calls for assertions)

---

### ConversationLoop Flow: `run_turn()`

```
1. SETUP
   └─ If first turn: inject system prompt into session

2. INPUT
   └─ Add user message to session

3. LOOP (MAX_TOOL_ITERATIONS = 25):
   
   A. GENERATE
      └─ engine.generate_stream(messages, tool_specs)
      └─ Collect tokens, tool_calls, finish_reason
      └─ Stream tokens via callback.on_token()
   
   B. PARSE
      └─ If no tool calls: break loop (turn complete)
      └─ Store assistant response + tool calls in session
   
   C. PERMISSION CHECK (for each tool call)
      ├─ Allowed              → execute immediately
      ├─ NeedsApproval        → prompt via callback.on_permission_prompt()
      │                          if denied: store denial in session, continue
      └─ Denied               → store denial in session, continue
   
   D. EXECUTE (approved tools only)
      ├─ Create ToolContext with cwd, permission mode, session_id
      ├─ execute_tool(registry, name, args, context)
      ├─ Format result as ToolResult
      └─ Store tool_result in session
   
   E. REPEAT
      └─ Loop back to step A (model sees tool results, decides next action)

4. PERSIST
   └─ session.save() to ~/.zipcode/sessions/{id}.json
```

**Iteration Bound:** MAX_TOOL_ITERATIONS = 25 (prevents infinite tool loops)

---

## Config Hierarchy

The runtime loads config from **two sources**, with project-level overrides:

```
Priority (high → low):
1. .zipcode.json          (project root)
2. ~/.zipcode/config.json (user global)
3. Compiled defaults
```

### ZipcodeConfig Structure

```rust
pub struct ZipcodeConfig {
    pub model_dir: PathBuf,              // Default: ~/.zipcode/models
    pub model_file: Option<String>,      // e.g., "gemma-2-27b-it-Q8_0.gguf"
    pub permission_mode: String,         // Default: "workspace-write"
    pub generation: GenerationOverrides,
}

pub struct GenerationOverrides {
    pub temperature: Option<f64>,        // Sampling temperature
    pub top_p: Option<f64>,              // Nucleus sampling
    pub max_tokens: Option<usize>,       // Output token limit
}
```

### Load Order Example

```bash
# 1. Load ~/.zipcode/config.json
{
  "permission_mode": "full-access",
  "generation": { "temperature": 0.7 }
}

# 2. Apply .zipcode.json override
{
  "permission_mode": "read-only",  # Overrides global
  "generation": { "temperature": 0.5 }
}

# Result
permission_mode = "read-only"      # Project override
temperature = 0.5                  # Project override
top_p = null                       # Not set, uses inference default
model_dir = ~/.zipcode/models      # Compiled default
```

**Loading Method:**
```rust
pub fn load(cwd: &Path) -> Result<Self>
```

---

## Permission Policy

### PermissionMode (from zipcode-tools)

Three permission levels:

| Mode | Allowed | Requires Approval | Blocked |
|------|---------|-------------------|---------|
| **FullAccess** | All tools | None | None |
| **WorkspaceWrite** | Read + write ops, fetch_repo | bash, repl | agent, unknown tools |
| **ReadOnly** | read_file, glob_search, grep_search, tool_search | None (denied instead) | bash, write_file, edit_file, fetch_repo, repl |

### PermissionCheck Enum

```rust
pub enum PermissionCheck {
    Allowed,                      // Proceed immediately
    NeedsApproval(String),        // Ask callback.on_permission_prompt()
    Denied(String),               // Block and send error to session
}
```

### PermissionPolicy::check()

```rust
pub fn check(&self, tool_name: &str, _args: &serde_json::Value) -> PermissionCheck
```

**Decision Tree:**
1. FullAccess mode? → Always Allowed
2. ReadOnly mode? → Only allow read/search tools, deny all writes
3. WorkspaceWrite mode? → Allow writes, require approval for bash/repl

**Denied Result:** Tool result message stored in session, tool does not execute.

**Approval Denied Result:** User declined in callback → deny tool, continue conversation.

---

## Session Persistence

### Session Structure

```rust
pub struct Session {
    pub id: String,                          // UUIDv4
    pub messages: Vec<ChatMessage>,          // Conversation history
    pub created_at: String,                  // RFC3339 timestamp
    pub updated_at: String,                  // Updated on push_message()
}
```

### Methods

| Method | Purpose |
|--------|---------|
| `Session::new()` | Create new session with UUID, timestamps |
| `session.save()` | Write to ~/.zipcode/sessions/{id}.json |
| `Session::load(id)` | Load from ~/.zipcode/sessions/{id}.json |
| `session.push_message(msg)` | Append message, update timestamp |

### Persistence Location

```
~/.zipcode/sessions/{session_id}.json
```

**Example:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "messages": [
    { "role": "system", "content": "You are zipcode..." },
    { "role": "user", "content": "explain this code" },
    { "role": "assistant", "content": "...", "tool_calls": [...] },
    { "role": "tool", "tool_use_id": "...", "content": "..." }
  ],
  "created_at": "2026-04-03T14:23:01Z",
  "updated_at": "2026-04-03T14:23:15Z"
}
```

**Auto-created directories:** `~/.zipcode/sessions/` is created on first save.

---

## System Prompt Building

### build_system_prompt() Function

```rust
pub fn build_system_prompt(
    cwd: &Path,
    registry: &ToolRegistry,
    permission_mode: &str,
) -> (String, Vec<ToolSpec>)
```

**Builds a dynamic system prompt with:**

1. **Base Prompt** — Hardcoded instruction (code assistant, rules, etc.)
2. **Permission Mode** — Current restriction level
3. **Working Directory** — Current cwd for context
4. **Project Instructions** — Injects `.zipcode.md` if present
5. **Git Status** — Runs `git status --short`, includes if non-empty
6. **Tool Specs** — All available tools from registry

**Example Output:**
```
You are zipcode, an AI coding assistant running locally on the user's machine.
You help with software engineering tasks: writing code, debugging, refactoring, and explaining code.

You have access to tools for file operations, shell commands, and code search. Use them to help the user.

Key rules:
- Read files before modifying them
- Prefer editing existing files over creating new ones
- Run tests after making changes
- Be concise and direct

Permission mode: workspace-write
Working directory: /home/user/my-project

# Project Instructions
Use Rust for everything. Tests are mandatory.

# Git Status
 M src/main.rs
 M Cargo.toml
?? new-feature.rs
```

---

## Integration Tests (6 tests)

All tests use `MockInferenceProvider` to simulate inference responses.

### Test 1: text_only_response()

**Scenario:** Model responds with text only (no tool calls).

**Assertion:**
- Tokens streamed and concatenated correctly
- Session contains 3 messages (system + user + assistant)
- No errors

---

### Test 2: single_tool_call()

**Scenario:** Model calls `read_file` once.

**Setup:**
- Create temp file with content "hello"
- Mock: ToolCall(read_file) → Text("I read the file.")

**Assertion:**
- 1 tool call executed
- Tool result contains "hello"
- Callback received on_tool_start() and on_tool_result()

---

### Test 3: multi_tool_turn()

**Scenario:** Model calls two tools (bash) in a single turn.

**Mock Responses:**
1. ToolCall(bash, "echo test1")
2. ToolCall(bash, "echo test2")
3. Text("Both done.")

**Assertion:**
- 2 tool calls executed
- Both results contain test1 and test2
- Turn completes after model final response

---

### Test 4: permission_denied()

**Scenario:** ReadOnly mode blocks write_file.

**Mock:** ToolCall(write_file) with ReadOnly permission mode.

**Assertion:**
- Tool does NOT execute
- Denial message stored in session messages
- Callback does not receive on_tool_result()

---

### Test 5: tool_call_loop_cap()

**Scenario:** Model generates 30 tool calls (exceeds MAX_TOOL_ITERATIONS = 25).

**Mock:** 30 ToolCall(bash) → Text("done")

**Assertion:**
- Loop breaks at iteration 25
- No more than 25 tool calls executed
- Session remains valid

---

### Test 6: path_traversal_blocked()

**Scenario:** Model attempts to read `../../etc/passwd` (traversal attack).

**Assertion:**
- read_file tool validates path, denies traversal
- Error message appears in session or callback results
- Contains "outside the workspace"

---

## Key Constants and Defaults

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_TOOL_ITERATIONS` | 25 | Max tool calls per turn (prevents infinite loops) |
| `BASE_SYSTEM_PROMPT` | (hardcoded) | Foundation instruction text |
| Default permission mode | "workspace-write" | Balanced safety + functionality |
| Default model_dir | `~/.zipcode/models` | Model storage location |
| Session save path | `~/.zipcode/sessions/` | Conversation history storage |

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `zipcode-inference` | (workspace) | `InferenceProvider` trait, `ChatMessage`, `ToolSpec` |
| `zipcode-tools` | (workspace) | `ToolRegistry`, `ToolContext`, `ToolResult`, `PermissionMode` |
| `anyhow` | ^1.0 | Error handling |
| `serde` | ^1.0 | JSON serialization (config, session) |
| `serde_json` | ^1.0 | JSON handling |
| `chrono` | ^0.4 | Timestamps (UTC RFC3339) |
| `uuid` | ^1.0 | Session IDs |
| `dirs` | ^5.0 | Home dir, config paths |
| `tracing` | ^0.1 | Logging (tool execution) |

---

## Test Commands

```bash
# All runtime tests
cargo test -p zipcode-runtime

# Single test
cargo test -p zipcode-runtime text_only_response

# With logging
RUST_LOG=debug cargo test -p zipcode-runtime -- --nocapture

# Integration tests only
cargo test -p zipcode-runtime --test integration
```

---

## Public API (lib.rs Exports)

```rust
pub use config::ZipcodeConfig;
pub use conversation::{ConversationLoop, StreamCallback};
pub use permission::{permission_mode_from_str, PermissionCheck, PermissionPolicy};
pub use session::Session;
```

**Modules (private):**
- `config` — Config loading
- `conversation` — Main loop
- `permission` — Permission checking
- `prompt` — Prompt building
- `session` — Session persistence

---

## AI Agent Instructions

### When Modifying the Conversation Loop

1. **Understand the iteration:** Each `run_turn()` call is one full conversation turn. A turn may include multiple tool calls and loops.

2. **MAX_TOOL_ITERATIONS is a safety bound:** Set to 25 to prevent infinite tool loops. Increase only if you understand the consequences.

3. **StreamCallback is the UI contract:** Changes to callback methods require updating all implementations (CLI, tests, any custom UIs).

4. **Permission checks happen before execution:** A tool call passes through `PermissionPolicy::check()` before `execute_tool()`. This is intentional.

5. **Session is single-threaded:** Session history is append-only. Do not mutate or replay messages.

6. **Tool specs must match registry:** The `tool_specs` passed to inference must match tools in the registry. Use `build_system_prompt()` to sync them.

### When Adding New Permission Rules

1. **Modify `PermissionPolicy::check()`** in `permission.rs` to recognize new tool names.

2. **Return appropriate `PermissionCheck`:**
   - `Allowed` — proceed with execution
   - `NeedsApproval(msg)` — prompt user (only in WorkspaceWrite mode)
   - `Denied(msg)` — block silently (only in ReadOnly mode)

3. **Add test in permission.rs** covering the new rule in all three modes.

4. **Document the rule** in this file under "Permission Policy" section.

### When Implementing a Custom StreamCallback

1. **Implement all 5 methods:**
   - `on_token()` — Called for every token (frequent)
   - `on_tool_start()` — Called before tool execution
   - `on_tool_result()` — Called after tool execution
   - `on_permission_prompt()` — Must return bool (true = approved)
   - `on_error()` — Called on inference errors

2. **Callbacks are synchronous:** Keep implementations lightweight (no blocking I/O).

3. **on_permission_prompt() determines approval:** Return true to proceed, false to deny.

### When Modifying Config Loading

1. **Order matters:** Project override > Global > Defaults. Respect this hierarchy.

2. **Use `ZipcodeConfig::load(cwd)`** to get the merged config.

3. **test_load_project_override** demonstrates override mechanics.

### When Changing Session Persistence

1. **Sessions auto-save in run_turn():** Changes to session structure require migration.

2. **~/.zipcode/sessions/ is the canonical path:** Don't change without strong reason.

3. **UUIDs are immutable:** session.id never changes. Use it as the persistence key.

### When Extending System Prompt

1. **Use `build_system_prompt()` in prompt.rs:** This is the single source of truth.

2. **Inject .zipcode.md for project-specific rules:** Users can customize per-project behavior.

3. **Git status is optional:** Only included if `git status --short` succeeds and is non-empty.

### When Adding Integration Tests

1. **Use `MockInferenceProvider`** to control inference responses.

2. **Use `TestCallback`** to capture tokens, tool calls, and errors.

3. **Follow the pattern in tests/integration.rs:**
   - Setup (temp dir, mock responses)
   - Build test loop
   - Run turn
   - Assert on callback/session state

4. **Test both happy path and edge cases:**
   - Permission denied
   - Invalid tool name
   - Path traversal
   - Loop overflow

---

## Debugging Tips

### Enable Logging

```bash
RUST_LOG=debug cargo test -p zipcode-runtime -- --nocapture
RUST_LOG=zipcode_runtime=trace cargo test -p zipcode-runtime -- --nocapture
```

**Key log sites:**
- `info!(tool = %call.name, "executing tool")` in conversation.rs line 118

### Inspect Session History

```bash
cat ~/.zipcode/sessions/{session-id}.json | jq .
```

**Fields to check:**
- `messages` array (should grow after each turn)
- `updated_at` (should match latest interaction)
- Tool call and tool_result messages

### Test Config Loading

```rust
// In code:
let config = ZipcodeConfig::load(Path::new("."))?;
println!("{:?}", config);
```

### Mock Inference Responses

```rust
let mock = MockInferenceProvider::new(vec![
    MockResponse::ToolCall { name: "bash".into(), args: json!({"command": "ls"}) },
    MockResponse::Text("Done.".into()),
]);
```

---

## Known Issues & Limitations

1. **MAX_TOOL_ITERATIONS = 25** may be too aggressive for complex tasks. Monitor in production.

2. **Chat template is hardcoded for Gemma format** (in inference crate). Multi-model support requires template selection logic.

3. **Permission prompts are synchronous** — No async approval workflow. UI/tests must implement blocking approval.

4. **Session files are JSON-only** — Large conversations may become slow to serialize/deserialize (no streaming).

5. **Path traversal check in tools crate** — Runtime trusts the tools crate validation. Never bypass `resolve_and_validate_path()`.

---

## Future Enhancements

- [ ] Async StreamCallback for non-blocking prompts
- [ ] Session compression/archiving for long conversations
- [ ] Dynamic tool spec generation from registry (less duplication)
- [ ] Permission revocation mid-conversation
- [ ] Conversation branching (replay from earlier turn)
- [ ] Tool result caching (avoid re-execution of identical calls)
