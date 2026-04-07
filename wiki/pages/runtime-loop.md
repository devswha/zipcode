---
title: Runtime Loop
tags: [modules]
sources: [session-2026-04-08]
updated: 2026-04-08
---

# Runtime Loop

The `zipcode-runtime` crate orchestrates the agentic conversation cycle.

## ConversationLoop

```rust
pub struct ConversationLoop {
    pub engine: Box<dyn InferenceProvider>,
    pub tools: ToolRegistry,
    pub session: Session,
    pub permission: PermissionPolicy,
    pub system_prompt: String,
    pub tool_specs: Vec<ToolSpec>,
    pub cwd: PathBuf,
}
```

### run_turn()

1. Add system prompt on first turn
2. Add user message
3. Loop (max 25 iterations):
   - Generate response via `engine.generate_stream()`
   - Parse tokens and tool calls from `mpsc::Receiver<TokenEvent>`
   - If no tool calls → turn complete
   - For each tool call: check permissions → execute → feed result back
4. Save session

## Config Hierarchy

`~/.zipcode/config.json` (global) < `.zipcode.json` (project override)

Fields: `model_dir`, `model_file`, `llama_server_bin`, `permission_mode`, `gpu_layers`, `flash_attention`, `generation` (temperature, top_p, max_tokens)

## Permission Modes

| Mode | Read | Write | Bash/REPL |
|------|------|-------|-----------|
| read-only | Allowed | Denied | Denied |
| workspace-write | Allowed | Allowed | Needs approval |
| full-access | Allowed | Allowed | Allowed |

## Session Persistence

UUID v4 session IDs. Messages serialized to `~/.zipcode/sessions/{id}.json`.

## See Also
- [[inference-backends]]
- [[tool-system]]
- [[cli-entrypoints]]
