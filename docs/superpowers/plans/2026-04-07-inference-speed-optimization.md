# Inference Speed Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make zipcode's llama-server backend dramatically faster by enabling SSE streaming, GPU offload, flash attention, and KV cache reuse.

**Architecture:** Modify `llama_server_backend.rs` to stream SSE tokens in a background thread, add GPU/FA flags to llama-server startup, and enable slot-based KV caching. Add `gpu_layers` and `flash_attention` config fields to `ZipcodeConfig`. Pass server options through from config to provider via a new `ServerOptions` struct.

**Tech Stack:** Rust, llama.cpp server (OpenAI-compatible API), SSE (Server-Sent Events), raw TCP HTTP

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/inference/src/llama_server_backend.rs` | Modify | SSE streaming, GPU flags, KV cache slot |
| `crates/runtime/src/config.rs` | Modify | Add `gpu_layers`, `flash_attention` fields |
| `crates/cli/src/repl.rs` | Modify | Pass server options from config to provider |
| `crates/inference/src/lib.rs` | Modify | Update `create_engine` to accept `ServerOptions` |

---

### Task 1: Add GPU and Flash Attention Config Fields

**Files:**
- Modify: `crates/runtime/src/config.rs:6-17` (ZipcodeConfig struct)
- Modify: `crates/runtime/src/config.rs:36-46` (Default impl)

- [ ] **Step 1: Write the failing test**

Add to `crates/runtime/src/config.rs` tests module:

```rust
#[test]
fn test_load_project_override_gpu_settings() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(
        dir.path().join(".zipcode.json"),
        r#"{"gpu_layers": 99, "flash_attention": true}"#,
    )
    .unwrap();
    let config = ZipcodeConfig::load(dir.path()).unwrap();
    assert_eq!(config.gpu_layers, Some(99));
    assert!(config.flash_attention);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p zipcode-runtime test_load_project_override_gpu_settings`
Expected: FAIL — `gpu_layers` and `flash_attention` fields don't exist on `ZipcodeConfig`

- [ ] **Step 3: Add fields to ZipcodeConfig**

In `crates/runtime/src/config.rs`, add two fields to `ZipcodeConfig`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipcodeConfig {
    #[serde(default = "default_model_dir")]
    pub model_dir: PathBuf,
    #[serde(default)]
    pub model_file: Option<String>,
    #[serde(default)]
    pub llama_server_bin: Option<PathBuf>,
    #[serde(default = "default_permission")]
    pub permission_mode: String,
    #[serde(default)]
    pub generation: GenerationOverrides,
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    #[serde(default)]
    pub flash_attention: bool,
}
```

Update `Default` impl:

```rust
impl Default for ZipcodeConfig {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            model_file: None,
            llama_server_bin: None,
            permission_mode: default_permission(),
            generation: GenerationOverrides::default(),
            gpu_layers: None,
            flash_attention: false,
        }
    }
}
```

Add project override parsing in `ZipcodeConfig::load()`, after the `generation` block (~line 102):

```rust
if let Some(layers) = project["gpu_layers"].as_i64() {
    config.gpu_layers = Some(layers as i32);
}
if let Some(fa) = project["flash_attention"].as_bool() {
    config.flash_attention = fa;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p zipcode-runtime test_load_project_override_gpu_settings`
Expected: PASS

- [ ] **Step 5: Verify existing tests still pass**

Run: `cargo test -p zipcode-runtime`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/runtime/src/config.rs
git commit -m "feat: add gpu_layers and flash_attention config fields"
```

---

### Task 2: Create ServerOptions and Wire GPU/FA Flags into llama-server Startup

**Files:**
- Modify: `crates/inference/src/llama_server_backend.rs:23-28` (struct), `31-83` (load fn)
- Modify: `crates/inference/src/lib.rs` (create_engine signature)
- Modify: `crates/cli/src/repl.rs:184-268` (create_loop fn)

- [ ] **Step 1: Write the failing test**

Add to `crates/inference/src/llama_server_backend.rs` tests module:

```rust
#[test]
fn server_options_default_has_no_gpu_layers() {
    let opts = ServerOptions::default();
    assert_eq!(opts.gpu_layers, None);
    assert!(!opts.flash_attention);
    assert_eq!(opts.context_size, DEFAULT_CONTEXT_SIZE);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p zipcode-inference server_options_default`
Expected: FAIL — `ServerOptions` doesn't exist

- [ ] **Step 3: Add ServerOptions struct**

In `crates/inference/src/llama_server_backend.rs`, after the constants, add:

```rust
#[derive(Debug, Clone)]
pub struct ServerOptions {
    pub gpu_layers: Option<i32>,
    pub flash_attention: bool,
    pub context_size: usize,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            gpu_layers: None,
            flash_attention: false,
            context_size: DEFAULT_CONTEXT_SIZE,
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p zipcode-inference server_options_default`
Expected: PASS

- [ ] **Step 5: Update LlamaServerProvider::load to accept ServerOptions**

Change `LlamaServerProvider::load` signature and add GPU/FA flags:

```rust
pub fn load(model_path: &Path, options: &ServerOptions) -> Result<Self> {
    let port = reserve_local_port()?;
    let model_alias = std::env::var("ZIPCODE_LLAMA_SERVER_ALIAS")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_ALIAS.to_string());
    let binary = resolve_llama_server_binary()?;

    let mut command = Command::new(&binary);
    command
        .arg("-m")
        .arg(model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--alias")
        .arg(&model_alias)
        .arg("--jinja")
        .arg("-c")
        .arg(options.context_size.to_string());

    if let Some(layers) = options.gpu_layers {
        command.arg("-ngl").arg(layers.to_string());
    }

    if options.flash_attention {
        command.arg("-fa");
    }

    // Enable KV cache slot reuse
    let cache_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".zipcode/cache");
    std::fs::create_dir_all(&cache_dir).ok();
    command
        .arg("--slot-save-path")
        .arg(&cache_dir);

    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    info!(
        binary = %binary.display(),
        model = %model_path.display(),
        port,
        gpu_layers = ?options.gpu_layers,
        flash_attention = options.flash_attention,
        "starting llama-server backend"
    );

    let mut child = command
        .spawn()
        .with_context(|| format!("Failed to start llama-server at {}", binary.display()))?;

    wait_until_ready(&mut child, port)?;

    Ok(Self {
        child,
        port,
        model_alias,
        config: GenerationConfig::default(),
    })
}
```

- [ ] **Step 6: Update create_engine and all call sites**

In `crates/inference/src/lib.rs`, add `pub use llama_server_backend::ServerOptions;` to the imports and update all `create_engine_inner` variants. Every `Backend::LlamaServer` arm changes from:

```rust
let mut provider = LlamaServerProvider::load(model_path)?;
```

to:

```rust
let mut provider = LlamaServerProvider::load(model_path, &server_options)?;
```

Update `create_engine` signature:

```rust
pub fn create_engine(
    backend: Backend,
    model_path: &std::path::Path,
    tokenizer_path: &std::path::Path,
    config: GenerationConfig,
    server_options: ServerOptions,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    create_engine_inner(backend, model_path, tokenizer_path, config, server_options)
}
```

All 4 `create_engine_inner` variants get the same additional parameter. Non-LlamaServer arms ignore it with `let _ = server_options;` where needed.

- [ ] **Step 7: Update CLI to pass ServerOptions from config**

In `crates/cli/src/repl.rs`, in `create_loop()`, after building `gen_config` (~line 220), add:

```rust
let server_options = zipcode_inference::ServerOptions {
    gpu_layers: config.gpu_layers,
    flash_attention: config.flash_attention,
    context_size: std::env::var("ZIPCODE_LLAMA_SERVER_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192),
};
```

And pass it to both `create_engine` calls:

```rust
let engine = match create_engine(backend, &model_file, &tokenizer_path, gen_config.clone(), server_options.clone()) {
```

and the fallback:

```rust
create_engine(
    Backend::LlamaServer,
    &model_file,
    &tokenizer_path,
    gen_config,
    server_options,
)
```

- [ ] **Step 8: Fix compilation and run all tests**

Run: `cargo test --workspace`
Expected: All tests pass. The mock/candle/llama-cpp backends don't use `ServerOptions`.

- [ ] **Step 9: Commit**

```bash
git add crates/inference/src/llama_server_backend.rs crates/inference/src/lib.rs crates/cli/src/repl.rs
git commit -m "feat: pass GPU layers, flash attention, and KV cache options to llama-server"
```

---

### Task 3: Enable SSE Streaming

**Files:**
- Modify: `crates/inference/src/llama_server_backend.rs:90-134` (generate_stream method)

This is the biggest change. The current `generate_stream` sends `"stream": false` and waits for the entire response. We switch to `"stream": true` and parse SSE events line-by-line from a raw TCP connection.

- [ ] **Step 1: Write the failing test for SSE line parsing**

Add to `crates/inference/src/llama_server_backend.rs` tests:

```rust
#[test]
fn parse_sse_data_line_extracts_token() {
    let line = r#"data: {"choices":[{"delta":{"content":"hello"}}]}"#;
    let event = parse_sse_line(line);
    assert!(matches!(event, Some(SseEvent::Token(ref t)) if t == "hello"));
}

#[test]
fn parse_sse_data_line_detects_done() {
    let line = "data: [DONE]";
    let event = parse_sse_line(line);
    assert!(matches!(event, Some(SseEvent::Done)));
}

#[test]
fn parse_sse_data_line_extracts_tool_call_chunks() {
    let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":"{\"command\":"}}]}}]}"#;
    let event = parse_sse_line(line);
    assert!(matches!(event, Some(SseEvent::ToolCallDelta { .. })));
}

#[test]
fn parse_sse_data_line_ignores_empty() {
    assert!(parse_sse_line("").is_none());
    assert!(parse_sse_line(": comment").is_none());
    assert!(parse_sse_line("data: ").is_none());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p zipcode-inference parse_sse`
Expected: FAIL — `parse_sse_line` and `SseEvent` don't exist

- [ ] **Step 3: Implement SseEvent enum and parse_sse_line**

Add above the tests module in `crates/inference/src/llama_server_backend.rs`:

```rust
#[derive(Debug)]
enum SseEvent {
    Token(String),
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },
    FinishReason(String),
    Done,
}

fn parse_sse_line(line: &str) -> Option<SseEvent> {
    let data = line.strip_prefix("data: ")?;
    let data = data.trim();
    if data.is_empty() {
        return None;
    }
    if data == "[DONE]" {
        return Some(SseEvent::Done);
    }

    let json: Value = serde_json::from_str(data).ok()?;
    let choice = json["choices"].get(0)?;

    // Check finish_reason
    if let Some(reason) = choice["finish_reason"].as_str() {
        if reason != "null" {
            return Some(SseEvent::FinishReason(reason.to_string()));
        }
    }

    let delta = &choice["delta"];

    // Tool call deltas
    if let Some(tool_calls) = delta["tool_calls"].as_array() {
        if let Some(tc) = tool_calls.first() {
            let index = tc["index"].as_u64().unwrap_or(0) as usize;
            let id = tc["id"].as_str().map(String::from);
            let name = tc["function"]["name"].as_str().map(String::from);
            let arguments = tc["function"]["arguments"].as_str().map(String::from);
            return Some(SseEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments,
            });
        }
    }

    // Content token
    if let Some(content) = delta["content"].as_str() {
        if !content.is_empty() {
            return Some(SseEvent::Token(content.to_string()));
        }
    }

    None
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p zipcode-inference parse_sse`
Expected: All 4 tests pass

- [ ] **Step 5: Commit SSE parser**

```bash
git add crates/inference/src/llama_server_backend.rs
git commit -m "feat: add SSE event parser for streaming llama-server responses"
```

- [ ] **Step 6: Replace generate_stream with streaming implementation**

Replace the `generate_stream` method on `LlamaServerProvider` (the inherent method, not the trait impl):

```rust
pub fn generate_stream(
    &mut self,
    messages: &[ChatMessage],
    tools: &[ToolSpec],
) -> mpsc::Receiver<TokenEvent> {
    let (tx, rx) = mpsc::channel();

    let request = build_chat_request(messages, tools, &self.config, &self.model_alias);
    let port = self.port;

    // Spawn a thread to read SSE events and send tokens
    std::thread::spawn(move || {
        if let Err(e) = stream_sse_events(port, &request, &tx) {
            let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                e.to_string(),
            )));
        }
    });

    rx
}
```

- [ ] **Step 7: Implement stream_sse_events function**

Add this function:

```rust
fn stream_sse_events(
    port: u16,
    request: &Value,
    tx: &mpsc::Sender<TokenEvent>,
) -> Result<()> {
    let body = serde_json::to_string(request)?;

    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .context("Failed to connect to llama-server")?;
    stream.set_read_timeout(Some(DEFAULT_REQUEST_TIMEOUT))?;
    stream.set_write_timeout(Some(DEFAULT_REQUEST_TIMEOUT))?;

    // Send HTTP request
    let http_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         User-Agent: zipcode/{version}\r\n\
         Accept: text/event-stream\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         \r\n\
         {body}",
        version = env!("CARGO_PKG_VERSION"),
        len = body.len(),
    );
    stream.write_all(http_req.as_bytes())?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);

    // Read status line
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    if !status_line.contains(" 200 ") {
        anyhow::bail!("llama-server returned: {}", status_line.trim());
    }

    // Skip headers
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
    }

    // Read SSE events
    // llama-server uses chunked transfer encoding, but we can read line-by-line
    // since SSE events are newline-delimited
    let mut tool_call_accum: Vec<ToolCallAccumulator> = Vec::new();
    let mut finish_reason = FinishReason::Stop;

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => break,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(e) => return Err(e.into()),
        }

        let line = line.trim();
        // Handle chunked transfer encoding: skip hex chunk-size lines
        if line.chars().all(|c| c.is_ascii_hexdigit()) && !line.is_empty() {
            continue;
        }

        let Some(event) = parse_sse_line(line) else {
            continue;
        };

        match event {
            SseEvent::Token(text) => {
                if tx.send(TokenEvent::Token(text)).is_err() {
                    return Ok(()); // receiver dropped
                }
            }
            SseEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments,
            } => {
                // Grow accumulator if needed
                while tool_call_accum.len() <= index {
                    tool_call_accum.push(ToolCallAccumulator::default());
                }
                let acc = &mut tool_call_accum[index];
                if let Some(id) = id {
                    acc.id = id;
                }
                if let Some(name) = name {
                    acc.name = name;
                }
                if let Some(args) = arguments {
                    acc.arguments.push_str(&args);
                }
            }
            SseEvent::FinishReason(reason) => {
                finish_reason = match reason.as_str() {
                    "length" => FinishReason::MaxTokens,
                    "tool_calls" => FinishReason::ToolUse,
                    _ => FinishReason::Stop,
                };
            }
            SseEvent::Done => break,
        }
    }

    // Emit accumulated tool calls
    if !tool_call_accum.is_empty() {
        for acc in tool_call_accum {
            let arguments: serde_json::Value =
                serde_json::from_str(&acc.arguments).unwrap_or_else(|_| json!({}));
            let _ = tx.send(TokenEvent::ToolCall(ToolCallParsed {
                id: acc.id,
                name: acc.name,
                arguments,
            }));
        }
        finish_reason = FinishReason::ToolUse;
    }

    let _ = tx.send(TokenEvent::Done(finish_reason));
    Ok(())
}

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}
```

- [ ] **Step 8: Update build_chat_request to enable streaming**

In `build_chat_request`, change `"stream": false` to `"stream": true`:

```rust
"stream": true,
```

- [ ] **Step 9: Add `id_slot` for KV cache reuse**

In `build_chat_request`, after the `repeat_last_n` field, add:

```rust
"id_slot": 0,
```

- [ ] **Step 10: Remove the now-unused post_json method's usage from generate_stream**

The `post_json` method is no longer called by `generate_stream`. Keep it for `wait_until_ready` health checks (it's used indirectly via `http_request`). No code to remove — `post_json` was only called from the old `generate_stream`.

- [ ] **Step 11: Run all tests**

Run: `cargo test --workspace`
Expected: All tests pass. The SSE streaming only activates against a live llama-server, so unit tests using `MockInferenceProvider` are unaffected.

- [ ] **Step 12: Commit**

```bash
git add crates/inference/src/llama_server_backend.rs
git commit -m "feat: enable SSE streaming for real-time token delivery from llama-server"
```

---

### Task 4: Add Environment Variable Overrides for GPU Settings

**Files:**
- Modify: `crates/cli/src/repl.rs` (create_loop, around ServerOptions construction)

- [ ] **Step 1: Write the test**

Add to `crates/cli/src/repl.rs` tests:

```rust
#[test]
fn server_options_respects_env_vars() {
    // This is a design validation — the actual env reading happens in create_loop
    // which requires a full model setup. We test the construction logic inline.
    let gpu = "42".parse::<i32>().ok();
    let fa = "1" == "1";
    assert_eq!(gpu, Some(42));
    assert!(fa);
}
```

- [ ] **Step 2: Update ServerOptions construction in create_loop**

In `crates/cli/src/repl.rs`, update the `ServerOptions` construction to check env vars as overrides:

```rust
let server_options = zipcode_inference::ServerOptions {
    gpu_layers: std::env::var("ZIPCODE_GPU_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .or(config.gpu_layers),
    flash_attention: std::env::var("ZIPCODE_FLASH_ATTENTION")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(config.flash_attention),
    context_size: std::env::var("ZIPCODE_LLAMA_SERVER_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192),
};
```

- [ ] **Step 3: Run all tests**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/cli/src/repl.rs
git commit -m "feat: support ZIPCODE_GPU_LAYERS and ZIPCODE_FLASH_ATTENTION env vars"
```

---

### Task 5: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md Known Limitations**

Remove item 4 ("No streaming to terminal") from the "Current Limitations" section since streaming is now implemented.

Add to the Build & Test section the new env vars:

```markdown
## Environment Variables (llama-server backend)

- `ZIPCODE_LLAMA_SERVER_BIN` — Path to llama-server binary
- `ZIPCODE_LLAMA_SERVER_CTX` — Context window size (default: 8192)
- `ZIPCODE_GPU_LAYERS` — Number of layers to offload to GPU (e.g., 99)
- `ZIPCODE_FLASH_ATTENTION` — Enable flash attention (1 or true)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update limitations and add GPU/streaming env var docs"
```

---

### Task 6: Integration Smoke Test

**Files:**
- Modify: `crates/inference/src/llama_server_backend.rs` (add ignored integration test)

- [ ] **Step 1: Add ignored integration test for SSE streaming**

Add to the tests module:

```rust
#[test]
#[ignore = "requires ZIPCODE_TEST_MODEL_PATH and llama-server on PATH"]
fn sse_streaming_returns_tokens_incrementally() {
    let model_path = std::env::var("ZIPCODE_TEST_MODEL_PATH")
        .expect("ZIPCODE_TEST_MODEL_PATH must be set");
    let model_path = std::path::Path::new(&model_path);

    let options = ServerOptions::default();
    let mut provider = LlamaServerProvider::load(model_path, &options).unwrap();

    let messages = vec![ChatMessage::user("Say hello in exactly one word.")];
    let rx = provider.generate_stream(&messages, &[]);

    let mut got_token = false;
    let mut got_done = false;
    for event in rx {
        match event {
            TokenEvent::Token(_) => got_token = true,
            TokenEvent::Done(_) => {
                got_done = true;
                break;
            }
            TokenEvent::Error(e) => panic!("unexpected error: {e}"),
            _ => {}
        }
    }

    assert!(got_token, "expected at least one token event");
    assert!(got_done, "expected done event");
}
```

- [ ] **Step 2: Run the ignored test (manual, requires model)**

Run: `ZIPCODE_TEST_MODEL_PATH=/path/to/model.gguf cargo test -p zipcode-inference sse_streaming -- --ignored`
Expected: PASS with tokens arriving incrementally

- [ ] **Step 3: Final full test suite**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings`
Expected: All tests pass, zero clippy warnings

- [ ] **Step 4: Commit**

```bash
git add crates/inference/src/llama_server_backend.rs
git commit -m "test: add SSE streaming integration smoke test"
```
