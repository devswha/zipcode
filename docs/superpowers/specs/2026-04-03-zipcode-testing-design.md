# zipcode Testing Design Spec — Mock Inference + Integration + CLI Smoke

Date: 2026-04-03

## Overview

Add a mock inference engine and two test suites (integration + CLI smoke) to the zipcode project. The mock engine returns predetermined responses, enabling full agentic loop testing without a real GGUF model.

## Approach

Introduce an `InferenceProvider` trait that both the real `InferenceEngine` and a new `MockInferenceProvider` implement. The `ConversationLoop` becomes generic over this trait, allowing tests to inject mock responses.

## MockInferenceProvider

```rust
pub struct MockInferenceProvider {
    responses: VecDeque<MockResponse>,
}

pub enum MockResponse {
    Text(String),
    ToolCall { name: String, args: serde_json::Value },
    Error(InferenceError),
}
```

- Pops one `MockResponse` per `generate_stream()` call
- `Text` → sends tokens then `Done(Stop)`
- `ToolCall` → sends `TokenEvent::ToolCall` then `Done(ToolUse)`
- `Error` → sends `TokenEvent::Error`

## InferenceProvider Trait

```rust
pub trait InferenceProvider: Send {
    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> std::sync::mpsc::Receiver<TokenEvent>;
}
```

- `InferenceEngine` implements this (delegates to existing method)
- `ConversationLoop.engine` becomes `Box<dyn InferenceProvider>`

## Integration Tests (6 scenarios)

File: `crates/runtime/tests/integration.rs`

| Scenario | Mock Response | Verifies |
|----------|---------------|----------|
| `text_only_response` | `Text("Hello!")` | History contains user + assistant messages |
| `single_tool_call` | `ToolCall(read_file)` then `Text("Done")` | Tool executed, result in history, final text returned |
| `multi_tool_turn` | `ToolCall(bash)` + `ToolCall(read_file)` then `Text("Done")` | Both results in history |
| `permission_denied` | `ToolCall(write_file)` then `Text("Denied")` | ReadOnly mode blocks write, denial message in history |
| `tool_call_loop_cap` | 30x `ToolCall(bash echo)` | Loop breaks at 25 iterations |
| `path_traversal_blocked` | `ToolCall(read_file ../../etc/passwd)` then `Text("Error")` | Error result in history |

## CLI Smoke Tests (4 scenarios)

File: `crates/cli/tests/smoke.rs`

| Test | Command | Verifies |
|------|---------|----------|
| `doctor_runs` | `zipcode doctor` | Exit 0, output contains "zipcode" |
| `help_flag` | `zipcode --help` | Exit 0, output contains "Usage" |
| `version_flag` | `zipcode --version` | Exit 0, contains version |
| `prompt_no_model` | `zipcode prompt "hi"` | No panic, graceful error about missing model |

## Files Changed

| File | Change |
|------|--------|
| `crates/inference/src/lib.rs` | Add `InferenceProvider` trait, export `mock` module |
| `crates/inference/src/engine.rs` | `impl InferenceProvider for InferenceEngine` |
| `crates/inference/src/mock.rs` | New — `MockInferenceProvider` |
| `crates/runtime/src/conversation.rs` | `engine: Box<dyn InferenceProvider>` |
| `crates/cli/src/repl.rs` | `Box::new(engine)` wrapping |
| `crates/runtime/tests/integration.rs` | New — 6 integration tests |
| `crates/cli/tests/smoke.rs` | New — 4 CLI smoke tests |

## Non-Goals

- Testing actual model inference quality
- Benchmarking inference performance
- Testing CUDA-specific codepaths
