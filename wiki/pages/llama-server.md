# llama-server — The primary working inference backend

**File:** [`crates/inference/src/llama_server_backend.rs`](../../crates/inference/src/llama_server_backend.rs)
**Trait:** [`InferenceProvider`](inference.md#inferenceprovider-trait)

This is the backend that actually works for Gemma 4 today. Candle uses `quantized_llama` as a placeholder and `llama-cpp-rs` 0.1.141 lacks Gemma 4 arch support — see [gotchas › backend reality check](gotchas.md#backend-reality-check).

---

## Architecture

```
LlamaServerProvider (Rust struct)
        │
        │ spawns
        ▼
  llama-server subprocess
        │
        │ exposes
        ▼
  HTTP on 127.0.0.1:<reserved-port>
        │
        │ OpenAI-compatible /v1/chat/completions + SSE
        ▼
  stream_sse_events() parses deltas into TokenEvent
```

---

## Struct & lifecycle

**EXTRACTED** `llama_server_backend.rs:40-149`

```rust
pub struct LlamaServerProvider {
    child: Child,             // spawned subprocess
    port: u16,                // reserved local port
    client: reqwest::Client,  // HTTP client
    // ...
}
```

### Spawn path

**EXTRACTED** `llama_server_backend.rs:81-106`

1. `reserve_local_port()` — bind `0.0.0.0:0`, read the assigned port, drop the listener. Race-prone but typically fine on localhost.
2. Build arg list:
   ```
   llama-server \
       -m <model.gguf> \
       --host 127.0.0.1 \
       --port <port> \
       --alias zipcode \
       --jinja \
       -c 8192 \
       -ngl <gpu_layers>          (if set)
       --flash-attn on            (if flash_attention=true)
       --slot-save-path ~/.zipcode/cache
   ```
3. `Command::new(binary).args(...).spawn()` → `Child`.
4. `wait_until_ready()` — poll `/health` endpoint with timeout.

### Drop

**EXTRACTED** `llama_server_backend.rs:143-148`

```rust
impl Drop for LlamaServerProvider {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
```

**GOTCHA:** if the process is already dead (crashed), `kill()` returns an error that's discarded. `wait()` still reaps to avoid zombie processes.

---

## Health check

**EXTRACTED** `llama_server_backend.rs:106` + `wait_until_ready` helper.

Uses **`/health`** (not `/v1/models`) because `/health` reports `status: "ok"` only after the model is fully loaded into memory. Documented rationale in previous wiki ADR `adr-health-check` (removed with old wiki rewrite on 2026-04-09).

**GOTCHA:** health check timeout is hardcoded. If the model is huge and the machine is slow, initialization can exceed it and zipcode will report the server as unhealthy even though it will eventually come up.

---

## SSE streaming

**EXTRACTED** `stream_sse_events()` in `llama_server_backend.rs`.

- POST to `/v1/chat/completions` with `stream: true`.
- Parse Server-Sent Events: each `data: {json}` line.
- Extract `choices[0].delta.content` → `TokenEvent::Token(text)`.
- Extract `choices[0].delta.tool_calls` → passed through the chat template parser (see [chat-template › tool-call parsing](chat-template.md#tool-call-parsing)).
- On `data: [DONE]` → `TokenEvent::Done(FinishReason)`.

The receiver channel is `std::sync::mpsc::Receiver<TokenEvent>` so the caller doesn't need tokio.

---

## GPU offload

**EXTRACTED** from `crates/cli/src/repl.rs` + `crates/runtime/src/config.rs`.

| Control | Env var | Config field | llama-server flag |
|---------|---------|--------------|-------------------|
| Layer offload | `ZIPCODE_GPU_LAYERS` | `gpu_layers` | `-ngl <n>` |
| Flash attention | `ZIPCODE_FLASH_ATTENTION` | `flash_attention` | `--flash-attn on` |

**Precedence:** env > config > unset.

**EXTRACTED** performance reference (from previous wiki `overview.md`): ~20× speedup on RTX 2070 SUPER (7.6 s vs 150 s for a representative prompt).

**GOTCHA:** env var names are case-sensitive. `zipcode_gpu_layers=99` does nothing and zipcode silently runs CPU-only.

---

## Binary discovery

**EXTRACTED** from `commands.rs` (setup) and `repl.rs`:

Precedence for the `llama-server` binary:

1. `ZIPCODE_LLAMA_SERVER_BIN` env var (absolute path)
2. `config.llama_server_bin` field from `~/.zipcode/config.json`
3. `which llama-server` on PATH

If none resolve, `doctor` reports `NeedsRepair`. See [cli › doctor](cli.md#doctor).

---

## Subprocess directories

**EXTRACTED** `llama_server_backend.rs:81-86`

- `--slot-save-path` points at `~/.zipcode/cache` — used by llama-server for KV-cache reuse across requests. Recent work (see CLAUDE.md "Production-ready") enables KV cache reuse for warmer subsequent prompts.

---

## Related pages

- [inference](inference.md) — the trait this implements
- [config](config.md) — `gpu_layers`, `flash_attention`, `llama_server_bin`
- [cli](cli.md) — resolves binary and builds `ServerOptions` before construction
- [gotchas › backend reality check](gotchas.md#backend-reality-check) — why this backend is the only working path
