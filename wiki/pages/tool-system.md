---
title: Tool System
tags: [modules]
sources: [session-2026-04-08]
updated: 2026-04-08
---

# Tool System

The `zipcode-tools` crate implements 10 tools behind the `Tool` trait.

## Tool Trait

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult>;
}
```

## Tools

| Tool | File | Description |
|------|------|-------------|
| read_file | `read_file.rs` | Read with line numbers, binary detection, 10MB limit, metadata header |
| write_file | `write_file.rs` | Create/overwrite, auto-creates parent dirs |
| edit_file | `edit_file.rs` | Single-occurrence string replacement |
| glob_search | `glob_search.rs` | Pattern matching, sorted output |
| grep_search | `grep_search.rs` | Regex search with optional glob filter |
| bash | `bash.rs` | Shell execution, 120s timeout, stderr capture |
| todo_write | `todo_write.rs` | Serialize todos to `.zipcode-todos.json` |
| tool_search | `tool_search.rs` | Keyword search over tool specs |
| repl | `repl.rs` | Python/Node subprocess execution |
| agent | `agent.rs` | Stub — returns "not yet implemented" |

## Key Patterns

- **Path traversal prevention**: `resolve_and_validate_path()` canonicalizes and validates paths stay within workspace
- **Output truncation**: All output capped at 8,192 bytes (`MAX_TOOL_OUTPUT_BYTES`)
- **ToolRegistry**: HashMap-based, with spec extraction for model injection
- **ToolSearchTool**: Built after all others so it can describe them

## See Also
- [[runtime-loop]]
- [[cli-entrypoints]]
