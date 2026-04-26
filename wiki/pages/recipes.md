# recipes — "How do I...?" playbook

Dev-facing task recipes. Each recipe lists the files you'll touch and the invariants you have to preserve.

---

## Add a new tool

**Goal:** register a new `Tool` implementation that the agent can call.

1. **Create the impl.** `crates/tools/src/my_tool.rs`:
   ```rust
   use crate::{Tool, ToolContext, ToolResult};
   use anyhow::Result;

   pub struct MyTool;

   impl Tool for MyTool {
       fn name(&self) -> &str { "my_tool" }
       fn description(&self) -> &str { "Does the thing." }
       fn parameters_schema(&self) -> serde_json::Value {
           serde_json::json!({
               "type": "object",
               "properties": {"arg": {"type": "string"}},
               "required": ["arg"]
           })
       }
       fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
           // if touching files, call resolve_and_validate_path(&ctx.cwd, path)
           Ok(ToolResult { content: "ok".into(), truncated: false })
       }
   }
   ```
2. **Export in `crates/tools/src/lib.rs`:** add `pub mod my_tool;` and `pub use my_tool::MyTool;`.
3. **Register in CLI:** in `crates/cli/src/repl.rs`, wherever the `ToolRegistry` is built, add `registry.register(Box::new(MyTool))`.
4. **Slot into permission matrix:** [`crates/runtime/src/permission.rs:5`](../../crates/runtime/src/permission.rs). Decide which tier allows it. **GOTCHA:** if you skip this and it touches the filesystem, read-only mode will silently deny it — which may or may not be what you want.
5. **Path safety:** if the tool reads/writes files, route every path through `resolve_and_validate_path()` (`tools/lib.rs:14`). See [tools › path safety](tools.md#path-safety-resolve_and_validate_path).
6. **Write tests** next to `my_tool.rs` covering:
   - Happy path
   - Permission-denied path (via a `ToolContext` built with `PermissionMode::ReadOnly`)
   - Path traversal rejection (if applicable)
7. Run `cargo test -p zipcode-tools`.

**Invariants preserved:**
- Output is auto-truncated at 8 KB by `execute_tool()` — you don't need to truncate yourself.
- `Send + Sync` trait bounds are enforced by the trait definition — your struct must be both.

---

## Add a new inference backend

**Goal:** let zipcode run on a new inference runtime (e.g., MLX, ONNX).

1. **Add a feature flag** in `crates/inference/Cargo.toml`:
   ```toml
   [features]
   mlx = ["dep:mlx-rs"]
   ```
2. **Create the impl.** `crates/inference/src/mlx_backend.rs`:
   ```rust
   use crate::{InferenceProvider, ChatMessage, TokenEvent};
   use crate::chat_template::ToolSpec;

   pub struct MlxProvider { /* ... */ }

   impl InferenceProvider for MlxProvider {
       fn generate_stream(
           &mut self,
           messages: &[ChatMessage],
           tools: &[ToolSpec],
       ) -> std::sync::mpsc::Receiver<TokenEvent> {
           let (tx, rx) = std::sync::mpsc::channel();
           // spawn a thread or run synchronously; emit Token / ToolCall / Done events
           rx
       }
   }
   ```
3. **Register in the `Backend` enum:** `crates/inference/src/lib.rs:82-89` — add `Mlx` variant.
4. **Register in the factory:** `crates/inference/src/lib.rs:82-200` — add a dispatch arm, feature-gated with `#[cfg(feature = "mlx")]`.
5. **Parse from CLI:** `Backend::parse()` (same file) — add the kebab-case name.
6. **Contract test** using a tiny prompt and `cargo test --features mlx`. Ideally run the same tests against `MockInferenceProvider` as a baseline.

**Gotcha:** the chat template is still hardcoded Gemma. See next recipe.

---

## Swap or add a chat template

**Goal:** support a model family that doesn't use Gemma's `<start_of_turn>` / `<tool_call>` format.

This one is an actual refactor, not a drop-in.

1. **Define a `ChatTemplate` trait** in `crates/inference/src/chat_template.rs`:
   ```rust
   pub trait ChatTemplate: Send + Sync {
       fn format_conversation(&self, messages: &[ChatMessage], tools: &[ToolSpec]) -> String;
       fn parse_tool_calls(&self, raw: &str) -> Vec<ToolCallParsed>;
       fn extract_text_content(&self, raw: &str) -> String;
   }
   ```
2. **Move Gemma functions behind a `GemmaTemplate` struct** that implements the trait.
3. **Add template field** to every `InferenceProvider` impl so each backend can own the right template per model.
4. **Teach `create_engine()`** (`lib.rs:71`) to pick the template based on model filename heuristics or an explicit config field.
5. **Add tool schema injection hook** to the trait — Gemma injects into the first user turn; other families inject elsewhere.
6. **Tests** for each new template: format round-trip, parse round-trip on recorded outputs.

**Trap:** the llama-server backend delegates template handling to llama-server's own `--jinja` flag (the server formats prompts before tokenizing). Swapping templates for that backend requires controlling the server's template config, not the Rust side.

---

## Change tool loop cap or output size

**Goal:** let longer tool chains run, or let tools return more data.

- **Tool iterations:** change `MAX_TOOL_ITERATIONS` in `crates/runtime/src/conversation.rs:110`. Tests in `crates/runtime/tests/integration.rs` assert the cap; update or parameterize.
- **Tool output size:** change `MAX_TOOL_OUTPUT_BYTES` in `crates/tools/src/lib.rs:502`. Tests in `crates/tools/src/lib.rs` assert truncation at 8 KB; update.

**Gotcha:** raising the output size directly inflates the model's working context. On 8192-token models this will cause context overflow sooner.

---

## Tune GPU offload

**Fastest path:** env vars at invocation time.
```bash
ZIPCODE_GPU_LAYERS=99 ZIPCODE_FLASH_ATTENTION=1 zipcode repl --backend llama-server
```
**Gotcha:** case-sensitive env var names. See [gotchas #9](gotchas.md#9--gpu-env-vars-are-case-sensitive).

**Persistent path:** edit `~/.zipcode/config.json` or `.zipcode.json`:
```json
{
  "gpu_layers": 99,
  "flash_attention": true
}
```

See [llama-server › GPU offload](llama-server.md#gpu-offload) and [config](config.md).

---

## Add a config field

1. Add the field to the `ZipcodeConfig` struct in `crates/runtime/src/config.rs` with `#[serde(default)]` for backward compatibility.
2. Add a default constant if needed.
3. Add a selective-merge line in the project-override code (if the field should layer).
4. Wire it into CLI startup (`crates/cli/src/repl.rs`) — decide precedence against env vars.
5. Add a test in `config.rs` covering: default, global override, project override.

---

## Add a slash command to the TUI

**Files:** `crates/cli/src/tui.rs` + `tui_composer.rs`.

1. Find the slash-command dispatcher (look for `/help` or `/clear`).
2. Add a match arm for the new command.
3. Test via `ZIPCODE_TUI_AUTOMATION_SCRIPT` env var in a smoke test under `crates/cli/tests/smoke.rs`.

---

## Change what goes into the system prompt

**File:** `crates/runtime/src/prompt.rs`

1. `BASE_SYSTEM_PROMPT` constant at lines 7-23 is the hardcoded identity + tool guidance block.
2. Injections added after the base:
   - Permission mode (line 34)
   - Working directory (line 35)
   - `.zipcode.md` content if present (lines 37-44)
   - `.zipcode.md` content from the detected project root if present (`crates/runtime/src/prompt.rs:40-46`)
3. Update the matching test in `prompt.rs` — there are 3.

---

## Debug "the model doesn't call tools"

1. Check `--backend` — is it `llama-server`? If not, switch.
2. Check the model — is it actually Gemma 4? Other families won't emit `<tool_call>` blocks against the hardcoded template.
3. Add a temporary `eprintln!` in `chat_template::parse_tool_calls()` (`chat_template.rs:60`) to print the raw model output and see whether `<tool_call>` blocks are actually present.
4. See [chat-template › gotcha: single-model coupling](chat-template.md#gotcha-single-model-coupling).

---

## Debug "llama-server won't start"

1. `zipcode doctor` — shows readiness state and where it's looking for the binary.
2. Check `ZIPCODE_LLAMA_SERVER_BIN` env var is set and the path exists.
3. Try running the binary directly with the same args zipcode would pass:
   ```bash
   llama-server -m <model.gguf> --host 127.0.0.1 --port 8080 --alias zipcode --jinja -c 8192
   ```
4. If the direct run works but zipcode times out, the health check timeout is too short for your hardware — see [gotchas #6](gotchas.md#6--llama-server-subprocess-lifecycle).

---

## Related pages

- [tools](tools.md) — when adding tools
- [inference](inference.md) — when adding backends
- [chat-template](chat-template.md) — when supporting new model families
- [gotchas](gotchas.md) — things to watch for in every recipe
