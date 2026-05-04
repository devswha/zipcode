# zipcode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust-based local-only coding agent powered by Gemma 4 via candle, deployable as a ZIP to air-gapped environments.

**Architecture:** 4-crate Rust workspace — `inference` (candle GGUF + Gemma 4), `tools` (11 built-in tools), `runtime` (agentic conversation loop), `cli` (REPL + one-shot). No cloud calls by default; explicit GitHub repo fetch uses `git clone`. Models loaded from local disk.

**Tech Stack:** Rust 2021, candle (candle-core, candle-nn, candle-transformers), tokio, serde/serde_json, clap, rustyline, termimad, glob, grep-regex

**Parallelism Map:** Tasks marked with the same `[PARALLEL GROUP]` can be executed simultaneously by different agents.

---

## File Structure

```
zipcode/
├── Cargo.toml                          # workspace root
├── .gitignore
├── crates/
│   ├── inference/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # re-exports
│   │       ├── types.rs                # ChatMessage, TokenEvent, GenerationConfig
│   │       ├── chat_template.rs        # Gemma 4 turn format + tool call parsing
│   │       ├── sampler.rs              # top-k, top-p, temperature sampling
│   │       ├── engine.rs               # GGUF load, KV cache, streaming generation
│   │       └── device.rs              # CUDA/CPU device detection
│   │
│   ├── tools/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # Tool trait, ToolContext, ToolResult, ToolRegistry
│   │       ├── bash.rs                 # shell command execution
│   │       ├── read_file.rs            # file reading with offset/limit
│   │       ├── write_file.rs           # file create/overwrite
│   │       ├── edit_file.rs            # string replacement editing
│   │       ├── glob_search.rs          # glob pattern file search
│   │       ├── grep_search.rs          # content search with regex
│   │       ├── todo_write.rs           # JSON-based todo tracking
│   │       ├── repl.rs                 # subprocess REPL execution
│   │       ├── agent.rs                # sub-agent delegation (stub for MVP)
│   │       └── tool_search.rs          # registry name/description search
│   │
│   ├── runtime/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # re-exports
│   │       ├── config.rs               # config hierarchy loader
│   │       ├── permission.rs           # permission policy + user prompt
│   │       ├── prompt.rs               # system prompt assembly
│   │       ├── session.rs              # session save/restore
│   │       └── conversation.rs         # agentic loop: generate → parse → execute → repeat
│   │
│   └── cli/
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs                 # entrypoint + clap args
│           ├── repl.rs                 # interactive REPL with rustyline
│           ├── render.rs               # markdown → ANSI rendering
│           └── commands.rs             # slash command dispatch
│
├── models/
│   └── .gitkeep
├── scripts/
│   ├── download_model.sh
│   ├── install.sh
│   └── package.sh
└── README.md
```

---

## Task 1: Workspace Scaffolding

**[PARALLEL GROUP: none — must complete first]**

**Files:**
- Create: `Cargo.toml`, `.gitignore`
- Create: `crates/inference/Cargo.toml`, `crates/inference/src/lib.rs`
- Create: `crates/tools/Cargo.toml`, `crates/tools/src/lib.rs`
- Create: `crates/runtime/Cargo.toml`, `crates/runtime/src/lib.rs`
- Create: `crates/cli/Cargo.toml`, `crates/cli/src/main.rs`
- Create: `models/.gitkeep`

- [ ] **Step 1: Create workspace root Cargo.toml**

```toml
# Cargo.toml
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
publish = false

[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
thiserror = "2"
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[workspace.lints.rust]
unsafe_code = "forbid"

[workspace.lints.clippy]
all = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
module_name_repetitions = "allow"
missing_panics_doc = "allow"
missing_errors_doc = "allow"
```

- [ ] **Step 2: Create .gitignore**

```gitignore
# .gitignore
/target
*.gguf
*.bin
models/*.gguf
models/*.bin
.env
```

- [ ] **Step 3: Create inference crate stub**

```toml
# crates/inference/Cargo.toml
[package]
name = "zipcode-inference"
version.workspace = true
edition.workspace = true

[dependencies]
candle-core = { git = "https://github.com/huggingface/candle.git", features = ["cuda"] }
candle-nn = { git = "https://github.com/huggingface/candle.git" }
candle-transformers = { git = "https://github.com/huggingface/candle.git" }
tokenizers = "0.21"
serde.workspace = true
serde_json.workspace = true
anyhow.workspace = true
thiserror.workspace = true
tracing.workspace = true

[dev-dependencies]
tokio = { workspace = true, features = ["test-util"] }
```

```rust
// crates/inference/src/lib.rs
pub mod types;

pub use types::*;
```

```rust
// crates/inference/src/types.rs
// placeholder — filled in Task 3
```

- [ ] **Step 4: Create tools crate stub**

```toml
# crates/tools/Cargo.toml
[package]
name = "zipcode-tools"
version.workspace = true
edition.workspace = true

[dependencies]
serde.workspace = true
serde_json.workspace = true
anyhow.workspace = true
thiserror.workspace = true
tracing.workspace = true
glob = "0.3"
regex = "1"
grep-regex = "0.1"
grep-searcher = "0.1"

[dev-dependencies]
tempfile = "3"
tokio = { workspace = true, features = ["test-util"] }
```

```rust
// crates/tools/src/lib.rs
// placeholder — filled in Task 2
```

- [ ] **Step 5: Create runtime crate stub**

```toml
# crates/runtime/Cargo.toml
[package]
name = "zipcode-runtime"
version.workspace = true
edition.workspace = true

[dependencies]
zipcode-inference = { path = "../inference" }
zipcode-tools = { path = "../tools" }
serde.workspace = true
serde_json.workspace = true
anyhow.workspace = true
thiserror.workspace = true
tokio.workspace = true
tracing.workspace = true
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
dirs = "6"

[dev-dependencies]
tempfile = "3"
tokio = { workspace = true, features = ["test-util"] }
```

```rust
// crates/runtime/src/lib.rs
// placeholder — filled in Task 9
```

- [ ] **Step 6: Create cli crate stub**

```toml
# crates/cli/Cargo.toml
[package]
name = "zipcode"
version.workspace = true
edition.workspace = true

[[bin]]
name = "zipcode"
path = "src/main.rs"

[dependencies]
zipcode-runtime = { path = "../runtime" }
zipcode-tools = { path = "../tools" }
zipcode-inference = { path = "../inference" }
clap = { version = "4", features = ["derive"] }
rustyline = "15"
termimad = "0.30"
tokio.workspace = true
anyhow.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
```

```rust
// crates/cli/src/main.rs
fn main() {
    println!("zipcode v0.1.0");
}
```

- [ ] **Step 7: Create models/.gitkeep**

```bash
mkdir -p models && touch models/.gitkeep
```

- [ ] **Step 8: Verify workspace compiles**

Run: `cargo check --workspace`
Expected: Compiles with no errors (warnings OK at this stage).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: scaffold zipcode workspace with 4 crates"
```

---

## Task 2: Tools Crate — Core Trait + Registry

**[PARALLEL GROUP A — with Task 3]**

**Files:**
- Create: `crates/tools/src/lib.rs`
- Test: `crates/tools/src/lib.rs` (inline tests)

- [ ] **Step 1: Write the failing test**

Add to `crates/tools/src/lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &str { "echo" }
        fn description(&self) -> &str { "Echoes input" }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                },
                "required": ["text"]
            })
        }
        fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
            let text = args["text"].as_str().unwrap_or("");
            Ok(ToolResult::new(text.to_string()))
        }
    }

    #[test]
    fn test_registry_add_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        assert!(registry.get("echo").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_specs() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let specs = registry.specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "echo");
    }

    #[test]
    fn test_tool_result_truncation() {
        let long_content = "x".repeat(10_000);
        let result = ToolResult::new(long_content);
        let truncated = result.truncate(8192);
        assert!(truncated.content.len() <= 8192 + 100); // margin for suffix
        assert!(truncated.truncated);
    }

    #[test]
    fn test_tool_execution() {
        let tool = EchoTool;
        let ctx = ToolContext {
            cwd: std::path::PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        };
        let args = serde_json::json!({"text": "hello"});
        let result = tool.execute(args, &ctx).unwrap();
        assert_eq!(result.content, "hello");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p zipcode-tools`
Expected: FAIL — types not defined yet.

- [ ] **Step 3: Write the implementation**

Replace `crates/tools/src/lib.rs` with:

```rust
use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod bash;
pub mod read_file;
pub mod write_file;
pub mod edit_file;
pub mod glob_search;
pub mod grep_search;
pub mod todo_write;
pub mod repl;
pub mod agent;
pub mod tool_search;

/// Permission modes for tool execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    ReadOnly,
    WorkspaceWrite,
    FullAccess,
}

/// Context passed to every tool execution
pub struct ToolContext {
    pub cwd: PathBuf,
    pub permission: PermissionMode,
    pub session_id: String,
}

/// Result from a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: String,
    pub truncated: bool,
}

impl ToolResult {
    pub fn new(content: String) -> Self {
        Self { content, truncated: false }
    }

    pub fn error(msg: String) -> Self {
        Self { content: format!("Error: {msg}"), truncated: false }
    }

    pub fn truncate(self, max_bytes: usize) -> Self {
        if self.content.len() <= max_bytes {
            return self;
        }
        let truncated_content = format!(
            "{}\n\n[truncated: showing first {} bytes of {}]",
            &self.content[..max_bytes],
            max_bytes,
            self.content.len()
        );
        Self {
            content: truncated_content,
            truncated: true,
        }
    }
}

/// Spec for a single tool — used to inject tool schemas into the model prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Trait every tool must implement
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult>;
}

/// Registry holding all available tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(AsRef::as_ref)
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| ToolSpec {
            name: t.name().to_string(),
            description: t.description().to_string(),
            parameters: t.parameters_schema(),
        }).collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_TOOL_OUTPUT_BYTES: usize = 8192;

/// Execute a tool by name with automatic truncation
pub fn execute_tool(
    registry: &ToolRegistry,
    name: &str,
    args: serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolResult> {
    let tool = registry
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown tool: {name}"))?;
    let result = tool.execute(args, ctx)?;
    Ok(result.truncate(MAX_TOOL_OUTPUT_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &str { "echo" }
        fn description(&self) -> &str { "Echoes input" }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                },
                "required": ["text"]
            })
        }
        fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
            let text = args["text"].as_str().unwrap_or("");
            Ok(ToolResult::new(text.to_string()))
        }
    }

    fn test_ctx() -> ToolContext {
        ToolContext {
            cwd: std::path::PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        }
    }

    #[test]
    fn test_registry_add_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        assert!(registry.get("echo").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_specs() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let specs = registry.specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "echo");
    }

    #[test]
    fn test_tool_result_truncation() {
        let long_content = "x".repeat(10_000);
        let result = ToolResult::new(long_content);
        let truncated = result.truncate(8192);
        assert!(truncated.content.len() <= 8192 + 100);
        assert!(truncated.truncated);
    }

    #[test]
    fn test_tool_execution() {
        let tool = EchoTool;
        let ctx = test_ctx();
        let args = serde_json::json!({"text": "hello"});
        let result = tool.execute(args, &ctx).unwrap();
        assert_eq!(result.content, "hello");
    }

    #[test]
    fn test_execute_unknown_tool() {
        let registry = ToolRegistry::new();
        let ctx = test_ctx();
        let result = execute_tool(&registry, "unknown", serde_json::json!({}), &ctx);
        assert!(result.is_err());
    }
}
```

Create empty module files so the crate compiles:

```bash
for f in bash read_file write_file edit_file glob_search grep_search todo_write repl agent tool_search; do
  echo "// TODO: implement" > "crates/tools/src/${f}.rs"
done
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p zipcode-tools`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/tools/
git commit -m "feat(tools): add Tool trait, ToolRegistry, and ToolResult with truncation"
```

---

## Task 3: Inference Crate — Types + Chat Template

**[PARALLEL GROUP A — with Task 2]**

**Files:**
- Create: `crates/inference/src/types.rs`
- Create: `crates/inference/src/chat_template.rs`
- Modify: `crates/inference/src/lib.rs`

- [ ] **Step 1: Write the failing test for types**

Add to `crates/inference/src/types.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_user() {
        let msg = ChatMessage::user("hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn test_chat_message_tool_result() {
        let msg = ChatMessage::tool_result("call_1", "file contents here");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert!((config.temperature - 0.7).abs() < f64::EPSILON);
        assert_eq!(config.max_tokens, 4096);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p zipcode-inference`
Expected: FAIL — types not defined.

- [ ] **Step 3: Implement types.rs**

```rust
// crates/inference/src/types.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Model,
    Tool,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallParsed>>,
}

impl ChatMessage {
    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Model,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant_with_tool_calls(content: &str, calls: Vec<ToolCallParsed>) -> Self {
        Self {
            role: Role::Model,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: Some(calls),
        }
    }

    pub fn tool_result(call_id: &str, content: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
            tool_call_id: Some(call_id.to_string()),
            tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallParsed {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum TokenEvent {
    Token(String),
    ToolCall(ToolCallParsed),
    Done(FinishReason),
    Error(InferenceError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    MaxTokens,
    ToolUse,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum InferenceError {
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("CUDA out of memory")]
    OutOfMemory,
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    #[error("Generation error: {0}")]
    GenerationError(String),
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 4096,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_user() {
        let msg = ChatMessage::user("hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn test_chat_message_tool_result() {
        let msg = ChatMessage::tool_result("call_1", "file contents here");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert!((config.temperature - 0.7).abs() < f64::EPSILON);
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn test_assistant_with_tool_calls() {
        let call = ToolCallParsed {
            id: "1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        };
        let msg = ChatMessage::assistant_with_tool_calls("", vec![call]);
        assert_eq!(msg.tool_calls.unwrap().len(), 1);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p zipcode-inference`
Expected: All 4 tests PASS.

- [ ] **Step 5: Write the failing test for chat_template**

Create `crates/inference/src/chat_template.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn test_format_user_turn() {
        let msg = ChatMessage::user("Hello");
        let formatted = format_message(&msg, &[]);
        assert!(formatted.contains("<start_of_turn>user"));
        assert!(formatted.contains("Hello"));
        assert!(formatted.contains("<end_of_turn>"));
    }

    #[test]
    fn test_format_with_tools_in_system() {
        let tools = vec![ToolSpec {
            name: "bash".to_string(),
            description: "Execute shell commands".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
        }];
        let msg = ChatMessage::user("run ls");
        let formatted = format_message(&msg, &tools);
        assert!(formatted.contains("bash"));
        assert!(formatted.contains("Execute shell commands"));
    }

    #[test]
    fn test_parse_tool_call_from_output() {
        let output = r#"<tool_call>
{"name": "read_file", "arguments": {"file_path": "src/main.rs"}}
</tool_call>"#;
        let parsed = parse_tool_calls(output);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "read_file");
        assert_eq!(parsed[0].arguments["file_path"], "src/main.rs");
    }

    #[test]
    fn test_parse_no_tool_call() {
        let output = "Here is the file content.";
        let parsed = parse_tool_calls(output);
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_format_tool_result_turn() {
        let msg = ChatMessage::tool_result("call_1", "{\"content\": \"hello\"}");
        let formatted = format_message(&msg, &[]);
        assert!(formatted.contains("<start_of_turn>tool"));
        assert!(formatted.contains("hello"));
    }

    #[test]
    fn test_format_conversation() {
        let messages = vec![
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello!"),
            ChatMessage::user("read main.rs"),
        ];
        let formatted = format_conversation(&messages, &[]);
        assert_eq!(formatted.matches("<start_of_turn>").count(), 3);
    }
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test -p zipcode-inference`
Expected: FAIL — functions not defined.

- [ ] **Step 7: Implement chat_template.rs**

```rust
// crates/inference/src/chat_template.rs

use crate::types::{ChatMessage, Role, ToolCallParsed};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Format a single message in Gemma 4 turn format
pub fn format_message(msg: &ChatMessage, tools: &[ToolSpec]) -> String {
    match msg.role {
        Role::System => {
            format!("<start_of_turn>user\n{}<end_of_turn>\n", msg.content)
        }
        Role::User => {
            let mut parts = String::new();
            if !tools.is_empty() {
                let tools_json = serde_json::to_string_pretty(tools).unwrap_or_default();
                parts.push_str(&format!(
                    "You have access to the following tools:\n{tools_json}\n\n"
                ));
            }
            parts.push_str(&msg.content);
            format!("<start_of_turn>user\n{parts}<end_of_turn>\n")
        }
        Role::Model => {
            format!("<start_of_turn>model\n{}<end_of_turn>\n", msg.content)
        }
        Role::Tool => {
            format!("<start_of_turn>tool\n{}<end_of_turn>\n", msg.content)
        }
    }
}

/// Format an entire conversation history into a single prompt string
pub fn format_conversation(messages: &[ChatMessage], tools: &[ToolSpec]) -> String {
    let mut prompt = String::new();
    let mut tools_injected = false;

    for msg in messages {
        if msg.role == Role::User && !tools_injected && !tools.is_empty() {
            prompt.push_str(&format_message(msg, tools));
            tools_injected = true;
        } else {
            prompt.push_str(&format_message(msg, &[]));
        }
    }

    // Add model turn prefix to prompt generation
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Parse tool calls from model output text
pub fn parse_tool_calls(output: &str) -> Vec<ToolCallParsed> {
    let mut calls = Vec::new();
    let mut search_from = 0;

    while let Some(start) = output[search_from..].find("<tool_call>") {
        let start = search_from + start + "<tool_call>".len();
        if let Some(end) = output[start..].find("</tool_call>") {
            let json_str = output[start..start + end].trim();
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                let name = parsed["name"].as_str().unwrap_or("").to_string();
                let arguments = parsed["arguments"].clone();
                calls.push(ToolCallParsed {
                    id: format!("call_{}", calls.len()),
                    name,
                    arguments,
                });
            }
            search_from = start + end + "</tool_call>".len();
        } else {
            break;
        }
    }

    calls
}

/// Extract the text content from model output, removing tool_call blocks
pub fn extract_text_content(output: &str) -> String {
    let mut result = output.to_string();
    while let Some(start) = result.find("<tool_call>") {
        if let Some(end) = result[start..].find("</tool_call>") {
            result = format!(
                "{}{}",
                &result[..start],
                &result[start + end + "</tool_call>".len()..]
            );
        } else {
            break;
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn test_format_user_turn() {
        let msg = ChatMessage::user("Hello");
        let formatted = format_message(&msg, &[]);
        assert!(formatted.contains("<start_of_turn>user"));
        assert!(formatted.contains("Hello"));
        assert!(formatted.contains("<end_of_turn>"));
    }

    #[test]
    fn test_format_with_tools_in_system() {
        let tools = vec![ToolSpec {
            name: "bash".to_string(),
            description: "Execute shell commands".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
        }];
        let msg = ChatMessage::user("run ls");
        let formatted = format_message(&msg, &tools);
        assert!(formatted.contains("bash"));
        assert!(formatted.contains("Execute shell commands"));
    }

    #[test]
    fn test_parse_tool_call_from_output() {
        let output = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"src/main.rs\"}}\n</tool_call>";
        let parsed = parse_tool_calls(output);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "read_file");
        assert_eq!(parsed[0].arguments["file_path"], "src/main.rs");
    }

    #[test]
    fn test_parse_no_tool_call() {
        let output = "Here is the file content.";
        let parsed = parse_tool_calls(output);
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_format_tool_result_turn() {
        let msg = ChatMessage::tool_result("call_1", "{\"content\": \"hello\"}");
        let formatted = format_message(&msg, &[]);
        assert!(formatted.contains("<start_of_turn>tool"));
        assert!(formatted.contains("hello"));
    }

    #[test]
    fn test_format_conversation() {
        let messages = vec![
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello!"),
            ChatMessage::user("read main.rs"),
        ];
        let formatted = format_conversation(&messages, &[]);
        assert_eq!(formatted.matches("<start_of_turn>").count(), 4); // 3 messages + 1 model prefix
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let output = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>\nsome text\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"a.rs\"}}\n</tool_call>";
        let parsed = parse_tool_calls(output);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "bash");
        assert_eq!(parsed[1].name, "read_file");
    }

    #[test]
    fn test_extract_text_content() {
        let output = "Hello <tool_call>{\"name\": \"bash\", \"arguments\": {}}</tool_call> world";
        let text = extract_text_content(output);
        assert_eq!(text, "Hello  world");
    }
}
```

Update `crates/inference/src/lib.rs`:

```rust
pub mod types;
pub mod chat_template;

pub use types::*;
pub use chat_template::{ToolSpec, format_conversation, format_message, parse_tool_calls, extract_text_content};
```

- [ ] **Step 8: Run all inference tests**

Run: `cargo test -p zipcode-inference`
Expected: All tests PASS.

- [ ] **Step 9: Commit**

```bash
git add crates/inference/
git commit -m "feat(inference): add ChatMessage types and Gemma 4 chat template parser"
```

---

## Task 4: Tools — File Operations (read, write, edit)

**[PARALLEL GROUP B — with Task 5, 6, 7]**

**Files:**
- Create: `crates/tools/src/read_file.rs`
- Create: `crates/tools/src/write_file.rs`
- Create: `crates/tools/src/edit_file.rs`

- [ ] **Step 1: Write failing tests for read_file**

```rust
// crates/tools/src/read_file.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolContext, PermissionMode};
    use tempfile::NamedTempFile;
    use std::io::Write;

    fn ctx() -> ToolContext {
        ToolContext {
            cwd: std::env::temp_dir(),
            permission: PermissionMode::FullAccess,
            session_id: "test".into(),
        }
    }

    #[test]
    fn test_read_existing_file() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "line1\nline2\nline3").unwrap();
        let args = serde_json::json!({"file_path": f.path().to_str().unwrap()});
        let result = ReadFileTool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("line1"));
    }

    #[test]
    fn test_read_with_offset_limit() {
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(f, "line{i}").unwrap();
        }
        let args = serde_json::json!({
            "file_path": f.path().to_str().unwrap(),
            "offset": 3,
            "limit": 2
        });
        let result = ReadFileTool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("line4"));
        assert!(result.content.contains("line5"));
        assert!(!result.content.contains("line6"));
    }

    #[test]
    fn test_read_nonexistent_file() {
        let args = serde_json::json!({"file_path": "/tmp/nonexistent_zipcode_test_file"});
        let result = ReadFileTool.execute(args, &ctx());
        assert!(result.is_err() || result.unwrap().content.contains("Error"));
    }
}
```

- [ ] **Step 2: Implement read_file.rs**

```rust
// crates/tools/src/read_file.rs
use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &str { "read_file" }

    fn description(&self) -> &str {
        "Read the contents of a file. Supports offset and limit for partial reads."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["file_path"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let file_path = args["file_path"]
            .as_str()
            .context("file_path is required")?;
        let path = resolve_path(file_path, &ctx.cwd);

        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        let lines: Vec<&str> = content.lines().collect();
        let offset = args["offset"].as_u64().unwrap_or(0) as usize;
        let limit = args["limit"].as_u64().map(|l| l as usize);

        let end = match limit {
            Some(l) => (offset + l).min(lines.len()),
            None => lines.len(),
        };

        let selected: Vec<String> = lines
            .get(offset..end)
            .unwrap_or(&[])
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{}\t{line}", offset + i + 1))
            .collect();

        Ok(ToolResult::new(selected.join("\n")))
    }
}

fn resolve_path(file_path: &str, cwd: &Path) -> std::path::PathBuf {
    let p = Path::new(file_path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use tempfile::NamedTempFile;
    use std::io::Write;

    fn ctx() -> ToolContext {
        ToolContext {
            cwd: std::env::temp_dir(),
            permission: PermissionMode::FullAccess,
            session_id: "test".into(),
        }
    }

    #[test]
    fn test_read_existing_file() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "line1\nline2\nline3").unwrap();
        let args = serde_json::json!({"file_path": f.path().to_str().unwrap()});
        let result = ReadFileTool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("line1"));
    }

    #[test]
    fn test_read_with_offset_limit() {
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(f, "line{i}").unwrap();
        }
        let args = serde_json::json!({
            "file_path": f.path().to_str().unwrap(),
            "offset": 3,
            "limit": 2
        });
        let result = ReadFileTool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("line4"));
        assert!(result.content.contains("line5"));
        assert!(!result.content.contains("line6"));
    }

    #[test]
    fn test_read_nonexistent_file() {
        let args = serde_json::json!({"file_path": "/tmp/nonexistent_zipcode_test_file_xyz"});
        let result = ReadFileTool.execute(args, &ctx());
        assert!(result.is_err());
    }
}
```

- [ ] **Step 3: Implement write_file.rs**

```rust
// crates/tools/src/write_file.rs
use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub struct WriteFileTool;

impl Tool for WriteFileTool {
    fn name(&self) -> &str { "write_file" }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let file_path = args["file_path"].as_str().context("file_path is required")?;
        let content = args["content"].as_str().context("content is required")?;
        let path = resolve_path(file_path, &ctx.cwd);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directories for: {}", path.display()))?;
        }

        fs::write(&path, content)
            .with_context(|| format!("Failed to write file: {}", path.display()))?;

        Ok(ToolResult::new(format!(
            "Successfully wrote {} bytes to {}",
            content.len(),
            path.display()
        )))
    }
}

fn resolve_path(file_path: &str, cwd: &Path) -> std::path::PathBuf {
    let p = Path::new(file_path);
    if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use tempfile::TempDir;

    fn ctx(dir: &TempDir) -> ToolContext {
        ToolContext {
            cwd: dir.path().to_path_buf(),
            permission: PermissionMode::FullAccess,
            session_id: "test".into(),
        }
    }

    #[test]
    fn test_write_new_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        let args = serde_json::json!({
            "file_path": path.to_str().unwrap(),
            "content": "hello world"
        });
        let result = WriteFileTool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("Successfully wrote"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello world");
    }

    #[test]
    fn test_write_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sub/dir/test.txt");
        let args = serde_json::json!({
            "file_path": path.to_str().unwrap(),
            "content": "nested"
        });
        WriteFileTool.execute(args, &ctx(&dir)).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "nested");
    }
}
```

- [ ] **Step 4: Implement edit_file.rs**

```rust
// crates/tools/src/edit_file.rs
use crate::{Tool, ToolContext, ToolResult};
use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;

pub struct EditFileTool;

impl Tool for EditFileTool {
    fn name(&self) -> &str { "edit_file" }

    fn description(&self) -> &str {
        "Edit a file by replacing an exact string match with new content."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string"
                }
            },
            "required": ["file_path", "old_string", "new_string"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let file_path = args["file_path"].as_str().context("file_path is required")?;
        let old_string = args["old_string"].as_str().context("old_string is required")?;
        let new_string = args["new_string"].as_str().context("new_string is required")?;
        let path = resolve_path(file_path, &ctx.cwd);

        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        let count = content.matches(old_string).count();
        if count == 0 {
            bail!("old_string not found in {}", path.display());
        }
        if count > 1 {
            bail!(
                "old_string found {} times in {} — must be unique. Provide more context.",
                count,
                path.display()
            );
        }

        let new_content = content.replacen(old_string, new_string, 1);
        fs::write(&path, &new_content)
            .with_context(|| format!("Failed to write file: {}", path.display()))?;

        Ok(ToolResult::new(format!(
            "Successfully edited {}",
            path.display()
        )))
    }
}

fn resolve_path(file_path: &str, cwd: &Path) -> std::path::PathBuf {
    let p = Path::new(file_path);
    if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use tempfile::TempDir;
    use std::fs;

    fn ctx(dir: &TempDir) -> ToolContext {
        ToolContext {
            cwd: dir.path().to_path_buf(),
            permission: PermissionMode::FullAccess,
            session_id: "test".into(),
        }
    }

    #[test]
    fn test_edit_replaces_unique_string() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        fs::write(&path, "fn main() {\n    println!(\"hello\");\n}").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_str().unwrap(),
            "old_string": "println!(\"hello\")",
            "new_string": "println!(\"world\")"
        });
        EditFileTool.execute(args, &ctx(&dir)).unwrap();
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("println!(\"world\")"));
        assert!(!content.contains("println!(\"hello\")"));
    }

    #[test]
    fn test_edit_fails_on_not_found() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        fs::write(&path, "hello").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_str().unwrap(),
            "old_string": "nonexistent",
            "new_string": "replacement"
        });
        let result = EditFileTool.execute(args, &ctx(&dir));
        assert!(result.is_err());
    }

    #[test]
    fn test_edit_fails_on_duplicate() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        fs::write(&path, "aaa bbb aaa").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_str().unwrap(),
            "old_string": "aaa",
            "new_string": "ccc"
        });
        let result = EditFileTool.execute(args, &ctx(&dir));
        assert!(result.is_err());
    }
}
```

- [ ] **Step 5: Run all file tool tests**

Run: `cargo test -p zipcode-tools -- read_file write_file edit_file`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/tools/src/read_file.rs crates/tools/src/write_file.rs crates/tools/src/edit_file.rs
git commit -m "feat(tools): implement read_file, write_file, and edit_file tools"
```

---

## Task 5: Tools — Search Operations (glob, grep)

**[PARALLEL GROUP B — with Task 4, 6, 7]**

**Files:**
- Create: `crates/tools/src/glob_search.rs`
- Create: `crates/tools/src/grep_search.rs`

- [ ] **Step 1: Implement glob_search.rs with tests**

```rust
// crates/tools/src/glob_search.rs
use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Context, Result};
use std::path::Path;

pub struct GlobSearchTool;

impl Tool for GlobSearchTool {
    fn name(&self) -> &str { "glob_search" }

    fn description(&self) -> &str {
        "Find files matching a glob pattern (e.g., '**/*.rs', 'src/**/*.ts')."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (defaults to cwd)"
                }
            },
            "required": ["pattern"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let pattern = args["pattern"].as_str().context("pattern is required")?;
        let base = match args["path"].as_str() {
            Some(p) => resolve_path(p, &ctx.cwd),
            None => ctx.cwd.clone(),
        };

        let full_pattern = base.join(pattern);
        let pattern_str = full_pattern.to_str().context("Invalid pattern path")?;

        let mut matches: Vec<String> = glob::glob(pattern_str)
            .with_context(|| format!("Invalid glob pattern: {pattern}"))?
            .filter_map(|entry| entry.ok())
            .map(|path| path.display().to_string())
            .collect();

        matches.sort();

        if matches.is_empty() {
            Ok(ToolResult::new("No files matched the pattern.".to_string()))
        } else {
            Ok(ToolResult::new(matches.join("\n")))
        }
    }
}

fn resolve_path(file_path: &str, cwd: &Path) -> std::path::PathBuf {
    let p = Path::new(file_path);
    if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use tempfile::TempDir;
    use std::fs;

    fn ctx(dir: &TempDir) -> ToolContext {
        ToolContext {
            cwd: dir.path().to_path_buf(),
            permission: PermissionMode::FullAccess,
            session_id: "test".into(),
        }
    }

    #[test]
    fn test_glob_finds_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "").unwrap();
        fs::write(dir.path().join("b.rs"), "").unwrap();
        fs::write(dir.path().join("c.txt"), "").unwrap();

        let args = serde_json::json!({"pattern": "*.rs"});
        let result = GlobSearchTool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("a.rs"));
        assert!(result.content.contains("b.rs"));
        assert!(!result.content.contains("c.txt"));
    }

    #[test]
    fn test_glob_no_matches() {
        let dir = TempDir::new().unwrap();
        let args = serde_json::json!({"pattern": "*.xyz"});
        let result = GlobSearchTool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("No files matched"));
    }
}
```

- [ ] **Step 2: Implement grep_search.rs with tests**

```rust
// crates/tools/src/grep_search.rs
use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub struct GrepSearchTool;

impl Tool for GrepSearchTool {
    fn name(&self) -> &str { "grep_search" }

    fn description(&self) -> &str {
        "Search file contents for a regex pattern. Returns matching lines with file paths and line numbers."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (defaults to cwd)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob filter for files (e.g., '*.rs')"
                }
            },
            "required": ["pattern"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let pattern_str = args["pattern"].as_str().context("pattern is required")?;
        let regex = regex::Regex::new(pattern_str)
            .with_context(|| format!("Invalid regex: {pattern_str}"))?;

        let base = match args["path"].as_str() {
            Some(p) => resolve_path(p, &ctx.cwd),
            None => ctx.cwd.clone(),
        };

        let file_glob = args["glob"].as_str();
        let mut results = Vec::new();

        if base.is_file() {
            search_file(&base, &regex, &mut results)?;
        } else {
            let glob_pattern = match file_glob {
                Some(g) => base.join("**").join(g),
                None => base.join("**").join("*"),
            };
            let pattern = glob_pattern.to_str().context("Invalid path")?;

            for entry in glob::glob(pattern).into_iter().flatten().flatten() {
                if entry.is_file() {
                    search_file(&entry, &regex, &mut results)?;
                }
            }
        }

        if results.is_empty() {
            Ok(ToolResult::new("No matches found.".to_string()))
        } else {
            Ok(ToolResult::new(results.join("\n")))
        }
    }
}

fn search_file(path: &Path, regex: &regex::Regex, results: &mut Vec<String>) -> Result<()> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()), // skip binary/unreadable files
    };

    for (line_num, line) in content.lines().enumerate() {
        if regex.is_match(line) {
            results.push(format!("{}:{}: {}", path.display(), line_num + 1, line));
        }
    }
    Ok(())
}

fn resolve_path(file_path: &str, cwd: &Path) -> std::path::PathBuf {
    let p = Path::new(file_path);
    if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use tempfile::TempDir;
    use std::fs;

    fn ctx(dir: &TempDir) -> ToolContext {
        ToolContext {
            cwd: dir.path().to_path_buf(),
            permission: PermissionMode::FullAccess,
            session_id: "test".into(),
        }
    }

    #[test]
    fn test_grep_finds_matches() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "fn main() {}\nfn helper() {}").unwrap();
        fs::write(dir.path().join("b.rs"), "let x = 42;").unwrap();

        let args = serde_json::json!({"pattern": "fn \\w+", "glob": "*.rs"});
        let result = GrepSearchTool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("fn main"));
        assert!(result.content.contains("fn helper"));
        assert!(!result.content.contains("let x"));
    }

    #[test]
    fn test_grep_no_matches() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "hello world").unwrap();

        let args = serde_json::json!({"pattern": "nonexistent_pattern"});
        let result = GrepSearchTool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("No matches"));
    }

    #[test]
    fn test_grep_single_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "line1 foo\nline2 bar\nline3 foo").unwrap();

        let args = serde_json::json!({"pattern": "foo", "path": file.to_str().unwrap()});
        let result = GrepSearchTool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains(":1:"));
        assert!(result.content.contains(":3:"));
    }
}
```

- [ ] **Step 3: Run search tool tests**

Run: `cargo test -p zipcode-tools -- glob_search grep_search`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/tools/src/glob_search.rs crates/tools/src/grep_search.rs
git commit -m "feat(tools): implement glob_search and grep_search tools"
```

---

## Task 6: Tools — Bash

**[PARALLEL GROUP B — with Task 4, 5, 7]**

**Files:**
- Create: `crates/tools/src/bash.rs`

- [ ] **Step 1: Implement bash.rs with tests**

```rust
// crates/tools/src/bash.rs
use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Context, Result};
use std::process::Command;
use std::time::Duration;

pub struct BashTool;

const DEFAULT_TIMEOUT_MS: u64 = 120_000;

impl Tool for BashTool {
    fn name(&self) -> &str { "bash" }

    fn description(&self) -> &str {
        "Execute a bash command and return its stdout and stderr."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default 120000)"
                }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let command = args["command"].as_str().context("command is required")?;
        let timeout_ms = args["timeout"].as_u64().unwrap_or(DEFAULT_TIMEOUT_MS);

        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&ctx.cwd)
            .output()
            .with_context(|| format!("Failed to execute: {command}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("STDERR:\n");
            result.push_str(&stderr);
        }

        if !output.status.success() {
            result.push_str(&format!("\nExit code: {}", output.status.code().unwrap_or(-1)));
        }

        Ok(ToolResult::new(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;

    fn ctx() -> ToolContext {
        ToolContext {
            cwd: std::env::temp_dir(),
            permission: PermissionMode::FullAccess,
            session_id: "test".into(),
        }
    }

    #[test]
    fn test_bash_echo() {
        let args = serde_json::json!({"command": "echo hello"});
        let result = BashTool.execute(args, &ctx()).unwrap();
        assert!(result.content.trim().contains("hello"));
    }

    #[test]
    fn test_bash_captures_stderr() {
        let args = serde_json::json!({"command": "echo err >&2"});
        let result = BashTool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("err"));
    }

    #[test]
    fn test_bash_nonzero_exit() {
        let args = serde_json::json!({"command": "exit 1"});
        let result = BashTool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("Exit code: 1"));
    }

    #[test]
    fn test_bash_uses_cwd() {
        let args = serde_json::json!({"command": "pwd"});
        let result = BashTool.execute(args, &ctx()).unwrap();
        assert!(!result.content.is_empty());
    }
}
```

- [ ] **Step 2: Run bash tool tests**

Run: `cargo test -p zipcode-tools -- bash`
Expected: All 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/tools/src/bash.rs
git commit -m "feat(tools): implement bash tool with subprocess execution"
```

---

## Task 7: Tools — Utility Tools (todo, repl, agent, tool_search)

**[PARALLEL GROUP B — with Task 4, 5, 6]**

**Files:**
- Create: `crates/tools/src/todo_write.rs`
- Create: `crates/tools/src/repl.rs`
- Create: `crates/tools/src/agent.rs`
- Create: `crates/tools/src/tool_search.rs`

- [ ] **Step 1: Implement todo_write.rs with tests**

```rust
// crates/tools/src/todo_write.rs
use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

pub struct TodoWriteTool;

#[derive(Debug, Serialize, Deserialize)]
struct TodoList {
    todos: Vec<TodoItem>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TodoItem {
    id: usize,
    content: String,
    status: String,
}

impl Tool for TodoWriteTool {
    fn name(&self) -> &str { "todo_write" }

    fn description(&self) -> &str {
        "Write or update a todo list. Provide the full list of todos each time."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                        }
                    }
                }
            },
            "required": ["todos"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let todos: Vec<TodoItem> = serde_json::from_value(args["todos"].clone())
            .context("Invalid todos format")?;

        let todo_path = todo_file_path(&ctx.cwd);
        let list = TodoList { todos };
        let json = serde_json::to_string_pretty(&list)?;
        fs::write(&todo_path, &json)?;

        Ok(ToolResult::new(format!(
            "Updated {} todos in {}",
            list.todos.len(),
            todo_path.display()
        )))
    }
}

fn todo_file_path(cwd: &std::path::Path) -> PathBuf {
    cwd.join(".zipcode-todos.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use tempfile::TempDir;

    fn ctx(dir: &TempDir) -> ToolContext {
        ToolContext {
            cwd: dir.path().to_path_buf(),
            permission: PermissionMode::FullAccess,
            session_id: "test".into(),
        }
    }

    #[test]
    fn test_todo_write_and_persist() {
        let dir = TempDir::new().unwrap();
        let args = serde_json::json!({
            "todos": [
                {"id": 1, "content": "fix bug", "status": "pending"},
                {"id": 2, "content": "add tests", "status": "completed"}
            ]
        });
        let result = TodoWriteTool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("Updated 2 todos"));

        let saved = fs::read_to_string(dir.path().join(".zipcode-todos.json")).unwrap();
        let list: TodoList = serde_json::from_str(&saved).unwrap();
        assert_eq!(list.todos.len(), 2);
    }
}
```

- [ ] **Step 2: Implement tool_search.rs with tests**

```rust
// crates/tools/src/tool_search.rs
use crate::{Tool, ToolContext, ToolResult, ToolRegistry};
use anyhow::{Context, Result};

pub struct ToolSearchTool {
    tool_specs: Vec<(String, String)>, // (name, description)
}

impl ToolSearchTool {
    pub fn new(registry: &ToolRegistry) -> Self {
        let tool_specs = registry.specs().into_iter()
            .map(|s| (s.name, s.description))
            .collect();
        Self { tool_specs }
    }

    pub fn from_specs(specs: Vec<(String, String)>) -> Self {
        Self { tool_specs: specs }
    }
}

impl Tool for ToolSearchTool {
    fn name(&self) -> &str { "tool_search" }

    fn description(&self) -> &str {
        "Search available tools by name or description keyword."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to match against tool names and descriptions"
                }
            },
            "required": ["query"]
        })
    }

    fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
        let query = args["query"].as_str().context("query is required")?;
        let query_lower = query.to_lowercase();

        let matches: Vec<String> = self.tool_specs.iter()
            .filter(|(name, desc)| {
                name.to_lowercase().contains(&query_lower)
                    || desc.to_lowercase().contains(&query_lower)
            })
            .map(|(name, desc)| format!("- {name}: {desc}"))
            .collect();

        if matches.is_empty() {
            Ok(ToolResult::new(format!("No tools matching '{query}'.")))
        } else {
            Ok(ToolResult::new(matches.join("\n")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;

    fn ctx() -> ToolContext {
        ToolContext {
            cwd: std::env::temp_dir(),
            permission: PermissionMode::FullAccess,
            session_id: "test".into(),
        }
    }

    #[test]
    fn test_search_by_name() {
        let tool = ToolSearchTool::from_specs(vec![
            ("bash".into(), "Execute shell commands".into()),
            ("read_file".into(), "Read file contents".into()),
        ]);
        let args = serde_json::json!({"query": "bash"});
        let result = tool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("bash"));
        assert!(!result.content.contains("read_file"));
    }

    #[test]
    fn test_search_by_description() {
        let tool = ToolSearchTool::from_specs(vec![
            ("bash".into(), "Execute shell commands".into()),
            ("read_file".into(), "Read file contents".into()),
        ]);
        let args = serde_json::json!({"query": "file"});
        let result = tool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("read_file"));
    }
}
```

- [ ] **Step 3: Implement repl.rs stub and agent.rs stub**

```rust
// crates/tools/src/repl.rs
use crate::{Tool, ToolContext, ToolResult};
use anyhow::{Context, Result};
use std::process::Command;

pub struct ReplTool;

impl Tool for ReplTool {
    fn name(&self) -> &str { "repl" }

    fn description(&self) -> &str {
        "Execute code in a subprocess REPL (Python or Node.js)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["python", "node"],
                    "description": "Language runtime to use"
                },
                "code": {
                    "type": "string",
                    "description": "Code to execute"
                }
            },
            "required": ["language", "code"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let language = args["language"].as_str().context("language is required")?;
        let code = args["code"].as_str().context("code is required")?;

        let (cmd, flag) = match language {
            "python" => ("python3", "-c"),
            "node" => ("node", "-e"),
            other => return Ok(ToolResult::error(format!("Unsupported language: {other}"))),
        };

        let output = Command::new(cmd)
            .arg(flag)
            .arg(code)
            .current_dir(&ctx.cwd)
            .output()
            .with_context(|| format!("Failed to run {cmd}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut result = stdout.to_string();
        if !stderr.is_empty() {
            result.push_str("\nSTDERR:\n");
            result.push_str(&stderr);
        }

        Ok(ToolResult::new(result))
    }
}
```

```rust
// crates/tools/src/agent.rs
use crate::{Tool, ToolContext, ToolResult};
use anyhow::Result;

pub struct AgentTool;

impl Tool for AgentTool {
    fn name(&self) -> &str { "agent" }

    fn description(&self) -> &str {
        "Delegate a sub-task to another agent instance. (MVP: not yet implemented)"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Task description for the sub-agent"
                }
            },
            "required": ["prompt"]
        })
    }

    fn execute(&self, _args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
        Ok(ToolResult::new(
            "Agent delegation is not yet implemented in this version.".to_string()
        ))
    }
}
```

- [ ] **Step 4: Run all utility tool tests**

Run: `cargo test -p zipcode-tools -- todo_write tool_search`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/tools/src/todo_write.rs crates/tools/src/repl.rs crates/tools/src/agent.rs crates/tools/src/tool_search.rs
git commit -m "feat(tools): implement todo_write, repl, agent (stub), and tool_search"
```

---

## Task 8: Inference Engine — candle GGUF + Gemma 4

**[PARALLEL GROUP C — after Task 3]**

**Files:**
- Create: `crates/inference/src/device.rs`
- Create: `crates/inference/src/sampler.rs`
- Create: `crates/inference/src/engine.rs`
- Modify: `crates/inference/src/lib.rs`

- [ ] **Step 1: Implement device.rs — CUDA/CPU detection**

```rust
// crates/inference/src/device.rs
use candle_core::Device;
use tracing::info;

pub fn select_device() -> Device {
    match Device::new_cuda(0) {
        Ok(device) => {
            info!("Using CUDA device 0");
            device
        }
        Err(e) => {
            info!("CUDA not available ({e}), falling back to CPU");
            Device::Cpu
        }
    }
}

pub fn device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "CUDA GPU".to_string(),
        _ => "Unknown device".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_device_returns_valid() {
        let device = select_device();
        let info = device_info(&device);
        assert!(!info.is_empty());
    }
}
```

- [ ] **Step 2: Implement sampler.rs — token sampling**

```rust
// crates/inference/src/sampler.rs
use candle_core::{Result, Tensor};
use crate::types::GenerationConfig;

pub struct Sampler {
    temperature: f64,
    top_p: f64,
    top_k: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    rng: fastrand::Rng,
}

impl Sampler {
    pub fn new(config: &GenerationConfig) -> Self {
        Self {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            repeat_penalty: config.repeat_penalty,
            repeat_last_n: config.repeat_last_n,
            rng: fastrand::Rng::new(),
        }
    }

    /// Sample a token index from logits tensor
    pub fn sample(&mut self, logits: &Tensor, past_tokens: &[u32]) -> Result<u32> {
        let logits = logits.to_dtype(candle_core::DType::F32)?.squeeze(0)?;
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;

        // Apply repeat penalty
        if self.repeat_penalty != 1.0 {
            let start = past_tokens.len().saturating_sub(self.repeat_last_n);
            for &token in &past_tokens[start..] {
                let idx = token as usize;
                if idx < logits_vec.len() {
                    if logits_vec[idx] > 0.0 {
                        logits_vec[idx] /= self.repeat_penalty;
                    } else {
                        logits_vec[idx] *= self.repeat_penalty;
                    }
                }
            }
        }

        // Apply temperature
        if self.temperature > 0.0 && self.temperature != 1.0 {
            for l in &mut logits_vec {
                *l /= self.temperature as f32;
            }
        }

        // If temperature is 0, use greedy
        if self.temperature == 0.0 {
            return Ok(logits_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0));
        }

        // Softmax
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits_vec.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Top-k filtering
        if self.top_k > 0 && self.top_k < probs.len() {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let threshold = indexed[self.top_k - 1].1;
            for p in &mut probs {
                if *p < threshold {
                    *p = 0.0;
                }
            }
        }

        // Top-p filtering
        if self.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cumulative = 0.0;
            let mut cutoff = 0.0;
            for (_, p) in &indexed {
                cumulative += p;
                if cumulative > self.top_p as f32 {
                    cutoff = *p;
                    break;
                }
            }
            for p in &mut probs {
                if *p < cutoff {
                    *p = 0.0;
                }
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum == 0.0 {
            return Ok(0);
        }
        for p in &mut probs {
            *p /= sum;
        }

        // Sample
        let r: f32 = self.rng.f32();
        let mut cumulative = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return Ok(i as u32);
            }
        }

        Ok(probs.len() as u32 - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let config = GenerationConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        // Create a simple logits tensor: [0.1, 0.9, 0.5]
        let logits = Tensor::new(&[0.1f32, 0.9, 0.5], &candle_core::Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let token = sampler.sample(&logits, &[]).unwrap();
        assert_eq!(token, 1); // highest logit
    }
}
```

Note: Add `fastrand = "2"` to `crates/inference/Cargo.toml` dependencies.

- [ ] **Step 3: Implement engine.rs — GGUF model loading and generation**

```rust
// crates/inference/src/engine.rs
use std::path::Path;
use std::sync::mpsc;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_gemma2 as gemma;
use tokenizers::Tokenizer;
use tracing::info;

use crate::chat_template::{self, ToolSpec};
use crate::sampler::Sampler;
use crate::types::*;

pub struct InferenceEngine {
    model: gemma::ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    config: GenerationConfig,
}

impl InferenceEngine {
    /// Load a GGUF model from disk
    pub fn load(model_path: &Path, tokenizer_path: &Path, device: Device) -> Result<Self> {
        info!("Loading model from {}", model_path.display());

        let mut file = std::fs::File::open(model_path)
            .with_context(|| format!("Model file not found: {}", model_path.display()))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .context("Failed to parse GGUF file")?;

        let model = gemma::ModelWeights::from_gguf(content, &mut file, &device)
            .context("Failed to load Gemma model weights")?;

        info!("Loading tokenizer from {}", tokenizer_path.display());
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config: GenerationConfig::default(),
        })
    }

    pub fn set_config(&mut self, config: GenerationConfig) {
        self.config = config;
    }

    /// Generate tokens streaming via channel
    pub fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel();

        let prompt = chat_template::format_conversation(messages, tools);

        let tokens = match self.tokenizer.encode(prompt.as_str(), true) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                let _ = tx.send(TokenEvent::Error(InferenceError::TokenizerError(e.to_string())));
                return rx;
            }
        };

        let mut sampler = Sampler::new(&self.config);
        let mut all_tokens = tokens.clone();
        let mut generated_text = String::new();

        // Feed prompt through model
        let input = match Tensor::new(tokens.as_slice(), &self.device) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(e.to_string())));
                return rx;
            }
        };

        let input = match input.unsqueeze(0) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(e.to_string())));
                return rx;
            }
        };

        let mut logits = match self.model.forward(&input, 0) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(e.to_string())));
                return rx;
            }
        };

        let eos_token = self.tokenizer.token_to_id("<eos>").unwrap_or(1);
        let end_of_turn = self.tokenizer.token_to_id("<end_of_turn>").unwrap_or(eos_token);

        // Autoregressive generation loop
        for i in 0..self.config.max_tokens {
            let next_logits = if logits.dims().len() == 3 {
                logits.squeeze(0).and_then(|t| {
                    let seq_len = t.dim(0).unwrap_or(1);
                    t.narrow(0, seq_len - 1, 1)
                })
            } else {
                let seq_len = logits.dim(0).unwrap_or(1);
                logits.narrow(0, seq_len - 1, 1)
            };

            let next_logits = match next_logits {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(e.to_string())));
                    break;
                }
            };

            let token = match sampler.sample(&next_logits, &all_tokens.iter().map(|&t| t).collect::<Vec<_>>()) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(e.to_string())));
                    break;
                }
            };

            // Check for stop tokens
            if token == eos_token || token == end_of_turn {
                // Parse for tool calls before finishing
                let tool_calls = chat_template::parse_tool_calls(&generated_text);
                if !tool_calls.is_empty() {
                    for call in tool_calls {
                        let _ = tx.send(TokenEvent::ToolCall(call));
                    }
                    let _ = tx.send(TokenEvent::Done(FinishReason::ToolUse));
                } else {
                    let _ = tx.send(TokenEvent::Done(FinishReason::Stop));
                }
                break;
            }

            all_tokens.push(token);

            // Decode token to text
            if let Ok(text) = self.tokenizer.decode(&[token], false) {
                generated_text.push_str(&text);
                let _ = tx.send(TokenEvent::Token(text));
            }

            // Prepare next input
            let next_input = match Tensor::new(&[token], &self.device) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(e.to_string())));
                    break;
                }
            };
            let next_input = match next_input.unsqueeze(0) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(e.to_string())));
                    break;
                }
            };

            logits = match self.model.forward(&next_input, tokens.len() + i) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(e.to_string())));
                    break;
                }
            };
        }

        rx
    }
}
```

- [ ] **Step 4: Update lib.rs**

```rust
// crates/inference/src/lib.rs
pub mod types;
pub mod chat_template;
pub mod sampler;
pub mod device;
pub mod engine;

pub use types::*;
pub use chat_template::{ToolSpec, format_conversation, format_message, parse_tool_calls, extract_text_content};
pub use device::select_device;
pub use engine::InferenceEngine;
```

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p zipcode-inference`
Expected: Compiles (full tests require a model file, so we test at integration level later).

- [ ] **Step 6: Commit**

```bash
git add crates/inference/
git commit -m "feat(inference): implement candle GGUF engine with Gemma 4 streaming generation"
```

---

## Task 9: Runtime — Config + Permission + Session

**[PARALLEL GROUP C — after Task 1]**

**Files:**
- Create: `crates/runtime/src/config.rs`
- Create: `crates/runtime/src/permission.rs`
- Create: `crates/runtime/src/session.rs`
- Modify: `crates/runtime/src/lib.rs`

- [ ] **Step 1: Implement config.rs with tests**

```rust
// crates/runtime/src/config.rs
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipcodeConfig {
    #[serde(default = "default_model_dir")]
    pub model_dir: PathBuf,
    #[serde(default)]
    pub model_file: Option<String>,
    #[serde(default = "default_permission")]
    pub permission_mode: String,
    #[serde(default)]
    pub generation: GenerationOverrides,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationOverrides {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<usize>,
}

fn default_model_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".zipcode/models")
}

fn default_permission() -> String {
    "workspace-write".to_string()
}

impl Default for ZipcodeConfig {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            model_file: None,
            permission_mode: default_permission(),
            generation: GenerationOverrides::default(),
        }
    }
}

impl ZipcodeConfig {
    /// Load config with hierarchy: global (~/.zipcode/config.json) < project (.zipcode.json)
    pub fn load(cwd: &Path) -> Result<Self> {
        let mut config = Self::default();

        // Global config
        let global_path = dirs::home_dir()
            .map(|h| h.join(".zipcode/config.json"));
        if let Some(path) = global_path {
            if path.exists() {
                let content = std::fs::read_to_string(&path)?;
                config = serde_json::from_str(&content)?;
            }
        }

        // Project config (overrides)
        let project_path = cwd.join(".zipcode.json");
        if project_path.exists() {
            let content = std::fs::read_to_string(&project_path)?;
            let project: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(perm) = project["permission_mode"].as_str() {
                config.permission_mode = perm.to_string();
            }
            if let Some(gen) = project.get("generation") {
                if let Some(t) = gen["temperature"].as_f64() {
                    config.generation.temperature = Some(t);
                }
                if let Some(t) = gen["top_p"].as_f64() {
                    config.generation.top_p = Some(t);
                }
                if let Some(t) = gen["max_tokens"].as_u64() {
                    config.generation.max_tokens = Some(t as usize);
                }
            }
        }

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ZipcodeConfig::default();
        assert_eq!(config.permission_mode, "workspace-write");
        assert!(config.model_dir.to_str().unwrap().contains(".zipcode"));
    }

    #[test]
    fn test_load_from_empty_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.permission_mode, "workspace-write");
    }

    #[test]
    fn test_load_project_override() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"permission_mode": "full-access", "generation": {"temperature": 0.5}}"#,
        ).unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.permission_mode, "full-access");
        assert_eq!(config.generation.temperature, Some(0.5));
    }
}
```

- [ ] **Step 2: Implement permission.rs with tests**

```rust
// crates/runtime/src/permission.rs
use zipcode_tools::PermissionMode;
use std::io::{self, Write};

pub struct PermissionPolicy {
    mode: PermissionMode,
}

#[derive(Debug, PartialEq)]
pub enum PermissionCheck {
    Allowed,
    NeedsApproval(String),
    Denied(String),
}

impl PermissionPolicy {
    pub fn new(mode: PermissionMode) -> Self {
        Self { mode }
    }

    pub fn check(&self, tool_name: &str, _args: &serde_json::Value) -> PermissionCheck {
        match self.mode {
            PermissionMode::FullAccess => PermissionCheck::Allowed,
            PermissionMode::ReadOnly => {
                match tool_name {
                    "read_file" | "glob_search" | "grep_search" | "tool_search" => {
                        PermissionCheck::Allowed
                    }
                    _ => PermissionCheck::Denied(format!(
                        "Tool '{tool_name}' is not allowed in read-only mode"
                    )),
                }
            }
            PermissionMode::WorkspaceWrite => {
                match tool_name {
                    "bash" => PermissionCheck::NeedsApproval(
                        "Bash execution requires approval in workspace-write mode".to_string()
                    ),
                    _ => PermissionCheck::Allowed,
                }
            }
        }
    }

    /// Prompt user for Y/N approval. Returns true if approved.
    pub fn prompt_user(message: &str) -> bool {
        print!("{message} [Y/n] ");
        io::stdout().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        let trimmed = input.trim().to_lowercase();
        trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
    }
}

impl From<&str> for PermissionMode {
    fn from(s: &str) -> Self {
        match s {
            "read-only" => PermissionMode::ReadOnly,
            "full-access" | "danger-full-access" => PermissionMode::FullAccess,
            _ => PermissionMode::WorkspaceWrite,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_access_allows_everything() {
        let policy = PermissionPolicy::new(PermissionMode::FullAccess);
        assert_eq!(policy.check("bash", &serde_json::json!({})), PermissionCheck::Allowed);
        assert_eq!(policy.check("write_file", &serde_json::json!({})), PermissionCheck::Allowed);
    }

    #[test]
    fn test_read_only_blocks_writes() {
        let policy = PermissionPolicy::new(PermissionMode::ReadOnly);
        assert_eq!(policy.check("read_file", &serde_json::json!({})), PermissionCheck::Allowed);
        match policy.check("bash", &serde_json::json!({})) {
            PermissionCheck::Denied(_) => {}
            other => panic!("Expected Denied, got {other:?}"),
        }
    }

    #[test]
    fn test_workspace_write_needs_approval_for_bash() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
        match policy.check("bash", &serde_json::json!({})) {
            PermissionCheck::NeedsApproval(_) => {}
            other => panic!("Expected NeedsApproval, got {other:?}"),
        }
        assert_eq!(policy.check("write_file", &serde_json::json!({})), PermissionCheck::Allowed);
    }
}
```

- [ ] **Step 3: Implement session.rs with tests**

```rust
// crates/runtime/src/session.rs
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use zipcode_inference::ChatMessage;

#[derive(Debug, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: String,
    pub updated_at: String,
}

impl Session {
    pub fn new() -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id,
            messages: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = session_path(&self.id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)
            .with_context(|| format!("Failed to save session to {}", path.display()))?;
        Ok(())
    }

    pub fn load(id: &str) -> Result<Self> {
        let path = session_path(id);
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Session not found: {id}"))?;
        let session: Self = serde_json::from_str(&content)?;
        Ok(session)
    }

    pub fn push_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
}

fn session_path(id: &str) -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(format!(".zipcode/sessions/{id}.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_session() {
        let session = Session::new();
        assert!(!session.id.is_empty());
        assert!(session.messages.is_empty());
    }

    #[test]
    fn test_push_message() {
        let mut session = Session::new();
        session.push_message(ChatMessage::user("hello"));
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn test_session_roundtrip() {
        let mut session = Session::new();
        session.push_message(ChatMessage::user("test"));

        // Save and reload
        session.save().unwrap();
        let loaded = Session::load(&session.id).unwrap();
        assert_eq!(loaded.messages.len(), 1);

        // Cleanup
        let path = session_path(&session.id);
        std::fs::remove_file(path).ok();
    }
}
```

- [ ] **Step 4: Update runtime lib.rs**

```rust
// crates/runtime/src/lib.rs
pub mod config;
pub mod permission;
pub mod session;

pub use config::ZipcodeConfig;
pub use permission::{PermissionPolicy, PermissionCheck};
pub use session::Session;
```

- [ ] **Step 5: Run all runtime tests**

Run: `cargo test -p zipcode-runtime`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/runtime/
git commit -m "feat(runtime): implement config loader, permission policy, and session management"
```

---

## Task 10: Runtime — System Prompt + Conversation Loop

**[PARALLEL GROUP D — after Tasks 2-9]**

**Files:**
- Create: `crates/runtime/src/prompt.rs`
- Create: `crates/runtime/src/conversation.rs`
- Modify: `crates/runtime/src/lib.rs`

- [ ] **Step 1: Implement prompt.rs with tests**

```rust
// crates/runtime/src/prompt.rs
use zipcode_tools::ToolRegistry;
use zipcode_inference::chat_template::ToolSpec;
use std::path::Path;

const BASE_SYSTEM_PROMPT: &str = r#"You are zipcode, an AI coding assistant running locally on the user's machine.
You help with software engineering tasks: writing code, debugging, refactoring, and explaining code.

You have access to tools for file operations, shell commands, and code search. Use them to help the user.

Key rules:
- Read files before modifying them
- Prefer editing existing files over creating new ones
- Run tests after making changes
- Be concise and direct"#;

pub fn build_system_prompt(
    cwd: &Path,
    registry: &ToolRegistry,
    permission_mode: &str,
) -> (String, Vec<ToolSpec>) {
    let mut prompt = BASE_SYSTEM_PROMPT.to_string();

    // Add permission context
    prompt.push_str(&format!("\n\nPermission mode: {permission_mode}"));
    prompt.push_str(&format!("\nWorking directory: {}", cwd.display()));

    // Load .zipcode.md if present
    let memory_path = cwd.join(".zipcode.md");
    if memory_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&memory_path) {
            prompt.push_str("\n\n# Project Instructions\n");
            prompt.push_str(&content);
        }
    }

    // Git status
    if let Ok(output) = std::process::Command::new("git")
        .args(["status", "--short"])
        .current_dir(cwd)
        .output()
    {
        if output.status.success() {
            let status = String::from_utf8_lossy(&output.stdout);
            if !status.is_empty() {
                prompt.push_str("\n\n# Git Status\n```\n");
                prompt.push_str(&status);
                prompt.push_str("```");
            }
        }
    }

    // Build tool specs
    let tool_specs: Vec<ToolSpec> = registry.specs().into_iter().map(|s| ToolSpec {
        name: s.name,
        description: s.description,
        parameters: s.parameters,
    }).collect();

    (prompt, tool_specs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_prompt_content() {
        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(Path::new("/tmp"), &registry, "workspace-write");
        assert!(prompt.contains("zipcode"));
        assert!(prompt.contains("workspace-write"));
    }

    #[test]
    fn test_prompt_includes_zipcode_md() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".zipcode.md"), "Use Rust for everything").unwrap();
        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "full-access");
        assert!(prompt.contains("Use Rust for everything"));
    }
}
```

- [ ] **Step 2: Implement conversation.rs**

```rust
// crates/runtime/src/conversation.rs
use anyhow::Result;
use tracing::info;

use zipcode_inference::{ChatMessage, InferenceEngine, TokenEvent, FinishReason};
use zipcode_inference::chat_template::ToolSpec;
use zipcode_tools::{ToolRegistry, ToolContext, execute_tool};

use crate::permission::{PermissionPolicy, PermissionCheck};
use crate::session::Session;

pub struct ConversationLoop {
    pub engine: InferenceEngine,
    pub tools: ToolRegistry,
    pub session: Session,
    pub permission: PermissionPolicy,
    pub system_prompt: String,
    pub tool_specs: Vec<ToolSpec>,
    pub cwd: std::path::PathBuf,
}

/// Callback for streaming tokens to the UI
pub trait StreamCallback: Send {
    fn on_token(&mut self, text: &str);
    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value);
    fn on_tool_result(&mut self, name: &str, result: &str);
    fn on_permission_prompt(&mut self, message: &str) -> bool;
    fn on_error(&mut self, error: &str);
}

impl ConversationLoop {
    pub fn run_turn(
        &mut self,
        user_input: &str,
        callback: &mut dyn StreamCallback,
    ) -> Result<()> {
        // Add system prompt on first turn
        if self.session.messages.is_empty() {
            self.session.push_message(ChatMessage::system(&self.system_prompt));
        }

        self.session.push_message(ChatMessage::user(user_input));

        loop {
            // Generate response
            let rx = self.engine.generate_stream(
                &self.session.messages,
                &self.tool_specs,
            );

            let mut full_text = String::new();
            let mut tool_calls = Vec::new();
            let mut finish_reason = FinishReason::Stop;

            for event in rx {
                match event {
                    TokenEvent::Token(text) => {
                        callback.on_token(&text);
                        full_text.push_str(&text);
                    }
                    TokenEvent::ToolCall(call) => {
                        tool_calls.push(call);
                    }
                    TokenEvent::Done(reason) => {
                        finish_reason = reason;
                        break;
                    }
                    TokenEvent::Error(e) => {
                        callback.on_error(&e.to_string());
                        return Err(anyhow::anyhow!("Inference error: {e}"));
                    }
                }
            }

            // Store assistant response
            if tool_calls.is_empty() {
                self.session.push_message(ChatMessage::assistant(&full_text));
            } else {
                self.session.push_message(
                    ChatMessage::assistant_with_tool_calls(&full_text, tool_calls.clone())
                );
            }

            // If no tool calls, turn is done
            if tool_calls.is_empty() {
                break;
            }

            // Execute tool calls
            for call in &tool_calls {
                // Permission check
                let check = self.permission.check(&call.name, &call.arguments);
                match check {
                    PermissionCheck::Allowed => {}
                    PermissionCheck::NeedsApproval(msg) => {
                        if !callback.on_permission_prompt(&msg) {
                            self.session.push_message(ChatMessage::tool_result(
                                &call.id,
                                "User denied permission for this action.",
                            ));
                            continue;
                        }
                    }
                    PermissionCheck::Denied(msg) => {
                        self.session.push_message(ChatMessage::tool_result(&call.id, &msg));
                        continue;
                    }
                }

                callback.on_tool_start(&call.name, &call.arguments);

                let ctx = ToolContext {
                    cwd: self.cwd.clone(),
                    permission: zipcode_tools::PermissionMode::FullAccess,
                    session_id: self.session.id.clone(),
                };

                let result = execute_tool(&self.tools, &call.name, call.arguments.clone(), &ctx);
                let result_text = match &result {
                    Ok(r) => r.content.clone(),
                    Err(e) => format!("Tool error: {e}"),
                };

                callback.on_tool_result(&call.name, &result_text);
                self.session.push_message(ChatMessage::tool_result(&call.id, &result_text));
            }

            // Continue loop — model will see tool results and decide next action
        }

        self.session.save()?;
        Ok(())
    }
}
```

- [ ] **Step 3: Update runtime lib.rs**

```rust
// crates/runtime/src/lib.rs
pub mod config;
pub mod permission;
pub mod session;
pub mod prompt;
pub mod conversation;

pub use config::ZipcodeConfig;
pub use permission::{PermissionPolicy, PermissionCheck};
pub use session::Session;
pub use conversation::{ConversationLoop, StreamCallback};
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p zipcode-runtime`
Expected: Compiles successfully.

- [ ] **Step 5: Commit**

```bash
git add crates/runtime/
git commit -m "feat(runtime): implement system prompt builder and conversation loop"
```

---

## Task 11: CLI — REPL + One-Shot + Doctor

**[PARALLEL GROUP E — after Task 10]**

**Files:**
- Create: `crates/cli/src/main.rs`
- Create: `crates/cli/src/repl.rs`
- Create: `crates/cli/src/render.rs`
- Create: `crates/cli/src/commands.rs`

- [ ] **Step 1: Implement main.rs with clap**

```rust
// crates/cli/src/main.rs
mod repl;
mod render;
mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "zipcode", version, about = "Local-only AI coding agent")]
struct Cli {
    /// Path to GGUF model file
    #[arg(long)]
    model: Option<PathBuf>,

    /// Permission mode
    #[arg(long, default_value = "workspace-write")]
    permission_mode: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// One-shot prompt (non-interactive)
    Prompt {
        /// The prompt text
        text: String,
    },
    /// Check environment health
    Doctor,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Doctor) => commands::doctor(&cli.model),
        Some(Commands::Prompt { text }) => {
            repl::run_oneshot(&text, cli.model.as_deref(), &cli.permission_mode)
        }
        None => {
            repl::run_interactive(cli.model.as_deref(), &cli.permission_mode)
        }
    }
}
```

- [ ] **Step 2: Implement render.rs**

```rust
// crates/cli/src/render.rs
use termimad::MadSkin;

pub fn create_skin() -> MadSkin {
    let mut skin = MadSkin::default();
    skin.set_headers_fg(termimad::crossterm::style::Color::Cyan);
    skin.bold.set_fg(termimad::crossterm::style::Color::White);
    skin.italic.set_fg(termimad::crossterm::style::Color::Grey);
    skin
}

pub fn render_markdown(text: &str) {
    let skin = create_skin();
    skin.print_text(text);
}

pub fn print_tool_start(name: &str, args: &serde_json::Value) {
    let args_str = serde_json::to_string(args).unwrap_or_default();
    eprintln!("\x1b[33m> {name}\x1b[0m {args_str}");
}

pub fn print_tool_result(name: &str, result: &str) {
    let preview = if result.len() > 200 {
        format!("{}...", &result[..200])
    } else {
        result.to_string()
    };
    eprintln!("\x1b[32m< {name}\x1b[0m {preview}");
}
```

- [ ] **Step 3: Implement commands.rs (doctor)**

```rust
// crates/cli/src/commands.rs
use anyhow::Result;
use std::path::{Path, PathBuf};

pub fn doctor(model_path: &Option<PathBuf>) -> Result<()> {
    println!("zipcode doctor");
    println!("──────────────");

    // Binary info
    println!("  Binary:   zipcode v{} (linux-x86_64)", env!("CARGO_PKG_VERSION"));

    // CUDA check
    let cuda_available = check_cuda();
    if cuda_available {
        println!("  CUDA:     \x1b[32m✅\x1b[0m Available");
    } else {
        println!("  CUDA:     \x1b[33m⚠️\x1b[0m  Not available (CPU mode)");
    }

    // Model check
    let model_dir = model_path
        .clone()
        .or_else(|| {
            dirs::home_dir().map(|h| h.join(".zipcode/models"))
        })
        .unwrap_or_else(|| PathBuf::from("models"));

    if model_dir.is_file() {
        let size = std::fs::metadata(&model_dir)
            .map(|m| m.len())
            .unwrap_or(0);
        println!(
            "  Model:    \x1b[32m✅\x1b[0m {} ({:.1} GB)",
            model_dir.display(),
            size as f64 / 1_073_741_824.0
        );
    } else if model_dir.is_dir() {
        let gguf_files: Vec<_> = std::fs::read_dir(&model_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext == "gguf")
            })
            .collect();

        if gguf_files.is_empty() {
            println!("  Model:    \x1b[31m❌\x1b[0m No .gguf files in {}", model_dir.display());
        } else {
            for f in &gguf_files {
                let size = f.metadata().map(|m| m.len()).unwrap_or(0);
                println!(
                    "  Model:    \x1b[32m✅\x1b[0m {} ({:.1} GB)",
                    f.file_name().to_string_lossy(),
                    size as f64 / 1_073_741_824.0
                );
            }
        }
    } else {
        println!("  Model:    \x1b[31m❌\x1b[0m Model directory not found: {}", model_dir.display());
    }

    println!();
    Ok(())
}

fn check_cuda() -> bool {
    // Simple check: try to find libcuda
    Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so").exists()
        || Path::new("/usr/local/cuda/lib64/libcudart.so").exists()
        || std::env::var("CUDA_PATH").is_ok()
}
```

- [ ] **Step 4: Implement repl.rs**

```rust
// crates/cli/src/repl.rs
use anyhow::Result;
use rustyline::DefaultEditor;
use std::path::Path;

use zipcode_inference::{InferenceEngine, select_device};
use zipcode_runtime::*;
use zipcode_tools::*;

use crate::render;

struct CliCallback;

impl StreamCallback for CliCallback {
    fn on_token(&mut self, text: &str) {
        print!("{text}");
        use std::io::Write;
        std::io::stdout().flush().ok();
    }

    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value) {
        render::print_tool_start(name, args);
    }

    fn on_tool_result(&mut self, name: &str, result: &str) {
        render::print_tool_result(name, result);
    }

    fn on_permission_prompt(&mut self, message: &str) -> bool {
        PermissionPolicy::prompt_user(message)
    }

    fn on_error(&mut self, error: &str) {
        eprintln!("\x1b[31mError: {error}\x1b[0m");
    }
}

fn build_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(bash::BashTool));
    registry.register(Box::new(read_file::ReadFileTool));
    registry.register(Box::new(write_file::WriteFileTool));
    registry.register(Box::new(edit_file::EditFileTool));
    registry.register(Box::new(glob_search::GlobSearchTool));
    registry.register(Box::new(grep_search::GrepSearchTool));
    registry.register(Box::new(todo_write::TodoWriteTool));
    registry.register(Box::new(repl::ReplTool));
    registry.register(Box::new(agent::AgentTool));
    // tool_search added after registry is built
    registry
}

fn create_loop(
    model_path: Option<&Path>,
    permission_mode: &str,
) -> Result<ConversationLoop> {
    let cwd = std::env::current_dir()?;
    let config = ZipcodeConfig::load(&cwd)?;

    let model = match model_path {
        Some(p) => p.to_path_buf(),
        None => find_model(&config.model_dir)?,
    };

    let tokenizer_path = model.with_extension("tokenizer.json");
    let tokenizer_path = if tokenizer_path.exists() {
        tokenizer_path
    } else {
        model.parent().unwrap_or(Path::new(".")).join("tokenizer.json")
    };

    let device = select_device();
    let engine = InferenceEngine::load(&model, &tokenizer_path, device)?;

    let registry = build_registry();
    let permission = PermissionPolicy::new(permission_mode.into());
    let session = Session::new();

    let (system_prompt, tool_specs) = zipcode_runtime::prompt::build_system_prompt(
        &cwd,
        &registry,
        permission_mode,
    );

    Ok(ConversationLoop {
        engine,
        tools: registry,
        session,
        permission,
        system_prompt,
        tool_specs,
        cwd,
    })
}

pub fn run_oneshot(text: &str, model_path: Option<&Path>, permission_mode: &str) -> Result<()> {
    let mut conv = create_loop(model_path, permission_mode)?;
    let mut callback = CliCallback;
    conv.run_turn(text, &mut callback)?;
    println!();
    Ok(())
}

pub fn run_interactive(model_path: Option<&Path>, permission_mode: &str) -> Result<()> {
    println!("zipcode v{} — local AI coding agent", env!("CARGO_PKG_VERSION"));
    println!("Type /help for commands, Ctrl+D to exit\n");

    let mut conv = create_loop(model_path, permission_mode)?;
    let mut rl = DefaultEditor::new()?;
    let mut callback = CliCallback;

    loop {
        let readline = rl.readline("\x1b[36m> \x1b[0m");
        match readline {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                rl.add_history_entry(trimmed)?;

                // Slash commands
                if trimmed.starts_with('/') {
                    match trimmed {
                        "/help" => print_help(),
                        "/status" => print_status(&conv),
                        "/clear" => {
                            conv.session = Session::new();
                            println!("Conversation cleared.");
                        }
                        "/quit" | "/exit" => break,
                        _ => println!("Unknown command: {trimmed}. Type /help for help."),
                    }
                    continue;
                }

                if let Err(e) = conv.run_turn(trimmed, &mut callback) {
                    eprintln!("\x1b[31mError: {e}\x1b[0m");
                }
                println!();
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("^C");
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("Goodbye!");
                break;
            }
            Err(e) => {
                eprintln!("Input error: {e}");
                break;
            }
        }
    }

    Ok(())
}

fn find_model(model_dir: &Path) -> Result<std::path::PathBuf> {
    if model_dir.is_file() {
        return Ok(model_dir.to_path_buf());
    }

    let entries: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
        .collect();

    match entries.len() {
        0 => anyhow::bail!("No .gguf model found in {}. Run 'zipcode doctor' for help.", model_dir.display()),
        1 => Ok(entries[0].path()),
        _ => {
            eprintln!("Multiple models found, using first: {}", entries[0].file_name().to_string_lossy());
            Ok(entries[0].path())
        }
    }
}

fn print_help() {
    println!("Commands:");
    println!("  /help    — Show this help");
    println!("  /status  — Show session status");
    println!("  /clear   — Clear conversation");
    println!("  /quit    — Exit");
}

fn print_status(conv: &ConversationLoop) {
    println!("Session: {}", conv.session.id);
    println!("Messages: {}", conv.session.messages.len());
    println!("CWD: {}", conv.cwd.display());
}
```

- [ ] **Step 5: Verify full workspace compiles**

Run: `cargo check --workspace`
Expected: Compiles successfully.

- [ ] **Step 6: Commit**

```bash
git add crates/cli/
git commit -m "feat(cli): implement REPL, one-shot mode, doctor, and slash commands"
```

---

## Task 12: Packaging Scripts

**[PARALLEL GROUP E — with Task 11]**

**Files:**
- Create: `scripts/install.sh`
- Create: `scripts/download_model.sh`
- Create: `scripts/package.sh`

- [ ] **Step 1: Create install.sh**

```bash
#!/bin/bash
# scripts/install.sh — Install zipcode from extracted ZIP
set -e

INSTALL_DIR="${HOME}/.zipcode"
BIN_DIR="/usr/local/bin"

echo "Installing zipcode..."

mkdir -p "${INSTALL_DIR}/models" "${INSTALL_DIR}/sessions"

if [ -f "./zipcode" ]; then
    cp ./zipcode "${INSTALL_DIR}/"
    chmod +x "${INSTALL_DIR}/zipcode"
else
    echo "Error: zipcode binary not found in current directory"
    exit 1
fi

# Try to symlink to PATH
if [ -w "${BIN_DIR}" ]; then
    ln -sf "${INSTALL_DIR}/zipcode" "${BIN_DIR}/zipcode"
    echo "Installed to ${BIN_DIR}/zipcode"
else
    echo "Cannot write to ${BIN_DIR}. Add ${INSTALL_DIR} to your PATH:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
fi

echo ""
echo "Done! Next steps:"
echo "  1. Place your .gguf model file in ${INSTALL_DIR}/models/"
echo "  2. Place the matching tokenizer.json alongside it"
echo "  3. Run: zipcode doctor"
```

- [ ] **Step 2: Create download_model.sh**

```bash
#!/bin/bash
# scripts/download_model.sh — Download Gemma 4 GGUF model
set -e

MODEL_DIR="${HOME}/.zipcode/models"
mkdir -p "${MODEL_DIR}"

echo "This script helps you download a Gemma 4 GGUF model."
echo ""
echo "For air-gapped environments, download these files on an internet-connected machine:"
echo ""
echo "  1. Model:     https://huggingface.co/google/gemma-4-27b-it-GGUF"
echo "  2. Tokenizer: https://huggingface.co/google/gemma-4-27b-it/raw/main/tokenizer.json"
echo ""
echo "Then copy both files to: ${MODEL_DIR}/"
echo ""

read -p "Download now? (requires internet + huggingface-cli) [y/N] " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v huggingface-cli &> /dev/null; then
        echo "Downloading model..."
        huggingface-cli download google/gemma-4-27b-it-GGUF --local-dir "${MODEL_DIR}"
        echo "Done! Model saved to ${MODEL_DIR}"
    else
        echo "huggingface-cli not found. Install with: pip install huggingface-hub"
        exit 1
    fi
fi
```

- [ ] **Step 3: Create package.sh**

```bash
#!/bin/bash
# scripts/package.sh — Build release binary and create ZIP archive
set -e

VERSION=$(cargo metadata --no-deps --format-version 1 | python3 -c "import sys,json;print(json.load(sys.stdin)['packages'][0]['version'])")
ARCHIVE="zipcode-v${VERSION}-linux-x86_64-cuda"

echo "Building release binary..."
cargo build --release -p zipcode

echo "Creating archive..."
mkdir -p "dist/${ARCHIVE}"

cp target/release/zipcode "dist/${ARCHIVE}/"
cp scripts/install.sh "dist/${ARCHIVE}/"
cp scripts/download_model.sh "dist/${ARCHIVE}/"
cp README.md "dist/${ARCHIVE}/" 2>/dev/null || true

mkdir -p "dist/${ARCHIVE}/models"
echo "Place your .gguf model file here." > "dist/${ARCHIVE}/models/PLACE_MODEL_HERE.txt"

cd dist
zip -r "${ARCHIVE}.zip" "${ARCHIVE}/"
echo ""
echo "Archive created: dist/${ARCHIVE}.zip"
echo "Size: $(du -h "${ARCHIVE}.zip" | cut -f1)"
```

- [ ] **Step 4: Make scripts executable**

```bash
chmod +x scripts/install.sh scripts/download_model.sh scripts/package.sh
```

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "feat: add install, download_model, and packaging scripts"
```

---

## Task 13: Integration — Full Build + Smoke Test

**[SEQUENTIAL — after all previous tasks]**

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All unit tests pass.

- [ ] **Step 2: Build release binary**

Run: `cargo build --release -p zipcode`
Expected: Builds successfully, binary at `target/release/zipcode`.

- [ ] **Step 3: Run doctor command**

Run: `./target/release/zipcode doctor`
Expected: Shows binary version, CUDA status, model status.

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: No warnings.

- [ ] **Step 5: Run fmt check**

Run: `cargo fmt --all -- --check`
Expected: All files formatted.

- [ ] **Step 6: Fix any issues from steps 1-5**

Address any test failures, clippy warnings, or formatting issues.

- [ ] **Step 7: Commit final state**

```bash
git add -A
git commit -m "chore: fix clippy warnings and formatting"
```

- [ ] **Step 8: Create packaging ZIP**

Run: `./scripts/package.sh`
Expected: Creates `dist/zipcode-v0.1.0-linux-x86_64-cuda.zip`

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "release: zipcode v0.1.0 — local-only AI coding agent"
```
