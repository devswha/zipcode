use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tracing::{debug, info};

use crate::chat_template::ToolSpec;
use crate::types::{
    ChatMessage, FinishReason, GenerationConfig, InferenceError, TokenEvent, ToolCallParsed,
};
use crate::{InferenceProvider, Role};

const DEFAULT_ALIAS: &str = "zipcode";
const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(180);
/// Default context size for llama-server.
///
/// Gemma 4 E2B/E4B support up to 128K native context; 26B/31B support
/// 256K. On an RTX 2070 SUPER 8 GB with E4B `Q4_K_M`, 128K context uses
/// ~5.8 GB VRAM — still within budget.
///
/// The previous 8K default caused context overflows during agentic loops
/// where the model reads back files it just wrote. Users with tighter
/// VRAM can override via `ZIPCODE_LLAMA_SERVER_CTX`.
pub const DEFAULT_CONTEXT_SIZE: usize = 131_072;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlashAttentionMode {
    LegacyFlagOnly,
    ExplicitOnValue,
}

fn flash_attention_mode_from_help(help: &str) -> FlashAttentionMode {
    if help.contains("[on|off|auto]") || help.contains("set Flash Attention use") {
        FlashAttentionMode::ExplicitOnValue
    } else {
        FlashAttentionMode::LegacyFlagOnly
    }
}

fn flash_attention_args(binary: &Path, enabled: bool) -> Vec<&'static str> {
    if !enabled {
        return Vec::new();
    }

    let help_output = Command::new(binary).arg("--help").output().ok();
    let help_text = help_output
        .as_ref()
        .map(|output| {
            format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .unwrap_or_default();

    match flash_attention_mode_from_help(&help_text) {
        FlashAttentionMode::LegacyFlagOnly => vec!["-fa"],
        FlashAttentionMode::ExplicitOnValue => vec!["-fa", "on"],
    }
}

pub struct LlamaServerProvider {
    /// Subprocess handle. `None` when connected to a pre-existing remote
    /// server via `ZIPCODE_LLAMA_SERVER_URL` — in that mode zipcode is a
    /// thin client that does not own the inference process lifecycle.
    child: Option<Child>,
    /// HTTP host for the server. `"127.0.0.1"` for locally-spawned
    /// processes, may be a LAN IP / hostname when pointing at a remote
    /// llama-server (e.g. a beefy Windows / Mac box on the same network).
    host: String,
    port: u16,
    model_alias: String,
    config: GenerationConfig,
}

impl LlamaServerProvider {
    /// Launch a local llama.cpp server for the given GGUF model, OR
    /// connect to a pre-existing remote llama-server if
    /// `ZIPCODE_LLAMA_SERVER_URL` is set.
    ///
    /// Remote mode: when `ZIPCODE_LLAMA_SERVER_URL=http://HOST:PORT` is
    /// exported, subprocess spawning is skipped entirely — `model_path`
    /// and `options` are ignored because the model is already loaded on
    /// the remote end. This lets the main dev machine run zipcode as a
    /// thin client while a bigger GPU elsewhere serves the model.
    ///
    /// Local mode: the binary is resolved from `ZIPCODE_LLAMA_SERVER_BIN`,
    /// `LLAMA_SERVER_BIN`, or `llama-server` on PATH.
    ///
    /// # Errors
    ///
    /// Returns an error if the server binary cannot be found, the process
    /// fails to spawn, or the server does not become ready within the
    /// startup timeout.
    pub fn load(model_path: &Path, options: &ServerOptions) -> Result<Self> {
        if let Ok(url) = std::env::var("ZIPCODE_LLAMA_SERVER_URL") {
            if !url.trim().is_empty() {
                return Self::connect_remote(url.trim());
            }
        }
        Self::spawn_local(model_path, options)
    }

    /// Connect to an already-running llama-server at the given URL.
    /// Accepts `http://host:port`, `host:port`, or bare `host` (defaults
    /// to port 8080). Probes `/health` once to verify the server is
    /// reachable before returning.
    fn connect_remote(url: &str) -> Result<Self> {
        let (host, port) = parse_server_url(url)?;
        let model_alias = std::env::var("ZIPCODE_LLAMA_SERVER_ALIAS")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_ALIAS.to_string());

        info!(
            host = %host,
            port,
            alias = %model_alias,
            "connecting to remote llama-server"
        );

        wait_until_remote_ready(&host, port)?;

        Ok(Self {
            child: None,
            host,
            port,
            model_alias,
            config: GenerationConfig::default(),
        })
    }

    fn spawn_local(model_path: &Path, options: &ServerOptions) -> Result<Self> {
        let port = reserve_local_port()?;
        let host = "127.0.0.1".to_string();
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
        for arg in flash_attention_args(&binary, options.flash_attention) {
            command.arg(arg);
        }

        let cache_dir = std::env::var_os("HOME")
            .map_or_else(|| PathBuf::from("."), PathBuf::from)
            .join(".zipcode/cache");
        std::fs::create_dir_all(&cache_dir).ok();
        command.arg("--slot-save-path").arg(&cache_dir);

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

        wait_until_ready(&mut child, &host, port)?;

        Ok(Self {
            child: Some(child),
            host,
            port,
            model_alias,
            config: GenerationConfig::default(),
        })
    }

    pub const fn set_config(&mut self, config: GenerationConfig) {
        self.config = config;
    }

    /// Call the local server using the OpenAI-compatible chat completions API with SSE streaming.
    pub fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel();

        let request = build_chat_request(messages, tools, &self.config, &self.model_alias);
        let host = self.host.clone();
        let port = self.port;

        std::thread::spawn(move || {
            if let Err(e) = stream_sse_events(&host, port, &request, &tx) {
                let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                    e.to_string(),
                )));
            }
        });

        rx
    }
}

impl Drop for LlamaServerProvider {
    fn drop(&mut self) {
        // Remote mode: no subprocess to clean up.
        let Some(mut child) = self.child.take() else {
            return;
        };
        if matches!(child.try_wait(), Ok(None)) {
            let _ = child.kill();
        }
        let _ = child.wait();
    }
}

impl InferenceProvider for LlamaServerProvider {
    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent> {
        self.generate_stream(messages, tools)
    }

    fn manages_own_context(&self) -> bool {
        // llama-server retains conversation state via slot-save-path when
        // `--slot-save-path` is configured, and more importantly, it keeps
        // the KV cache across requests to the same slot. Combined with
        // server-side Jinja chat-template rendering (`--jinja`), this
        // provider does not need the caller to re-send full history.
        true
    }
}

fn resolve_llama_server_binary() -> Result<PathBuf> {
    let env_candidates = ["ZIPCODE_LLAMA_SERVER_BIN", "LLAMA_SERVER_BIN"];
    for key in env_candidates {
        if let Ok(value) = std::env::var(key) {
            let path = PathBuf::from(value);
            if path.exists() {
                return Ok(path);
            }
        }
    }

    which_in_path("llama-server").ok_or_else(|| {
        anyhow::anyhow!(
            "Gemma 4 requires a recent llama.cpp server. Set ZIPCODE_LLAMA_SERVER_BIN or install `llama-server` on PATH."
        )
    })
}

fn which_in_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn reserve_local_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("Failed to reserve local port")?;
    let port = listener
        .local_addr()
        .context("Failed to inspect reserved port")?
        .port();
    drop(listener);
    Ok(port)
}

fn wait_until_ready(child: &mut Child, host: &str, port: u16) -> Result<()> {
    let deadline = Instant::now() + DEFAULT_STARTUP_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait()? {
            anyhow::bail!("llama-server exited before becoming ready (status: {status})");
        }

        if Instant::now() >= deadline {
            anyhow::bail!("Timed out waiting for llama-server on {host}:{port}");
        }

        match health_check(host, port) {
            HealthStatus::Ready => return Ok(()),
            HealthStatus::Loading => {
                debug!(host, port, "llama-server model still loading");
                std::thread::sleep(Duration::from_millis(500));
            }
            HealthStatus::Unreachable(error) => {
                debug!(host, port, error = %error, "waiting for llama-server to accept requests");
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
}

/// Remote-mode readiness probe — no child process to monitor, just poll
/// `/health` until the server is ready or the timeout expires. A hard
/// error on `Unreachable` because if the remote endpoint isn't even
/// accepting TCP connections the user likely mis-typed the URL.
fn wait_until_remote_ready(host: &str, port: u16) -> Result<()> {
    let deadline = Instant::now() + DEFAULT_STARTUP_TIMEOUT;
    loop {
        if Instant::now() >= deadline {
            anyhow::bail!("Remote llama-server at {host}:{port} never became ready");
        }

        match health_check(host, port) {
            HealthStatus::Ready => return Ok(()),
            HealthStatus::Loading => {
                debug!(host, port, "remote llama-server model still loading");
                std::thread::sleep(Duration::from_millis(500));
            }
            HealthStatus::Unreachable(error) => {
                anyhow::bail!(
                    "Remote llama-server at {host}:{port} is unreachable: {error}. \
                    Confirm the server is running and `ZIPCODE_LLAMA_SERVER_URL` points at it."
                );
            }
        }
    }
}

#[derive(Debug)]
enum HealthStatus {
    Ready,
    Loading,
    Unreachable(String),
}

fn health_check(host: &str, port: u16) -> HealthStatus {
    let Ok(mut stream) = TcpStream::connect((host, port)) else {
        return HealthStatus::Unreachable("connection refused".to_string());
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));

    let request =
        format!("GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() || stream.flush().is_err() {
        return HealthStatus::Unreachable("write failed".to_string());
    }

    let mut buf = Vec::new();
    if let Err(e) = stream.read_to_end(&mut buf) {
        tracing::debug!(error = %e, "health_check: failed to read response from llama-server");
        return HealthStatus::Unreachable(format!("read failed: {e}"));
    }
    let response = String::from_utf8_lossy(&buf);

    // Parse the HTTP status line to detect server errors early.
    // Without this check, a 500 Internal Server Error response would be
    // silently treated as HealthStatus::Loading, causing the startup polling
    // loop to wait the full DEFAULT_STARTUP_TIMEOUT (180s) before failing.
    // This is inconsistent with stream_sse_events() which checks for 200.
    if let Some(status_code) = parse_http_status_code(&response) {
        if !(200..300).contains(&status_code) {
            let status_line = response
                .lines()
                .next()
                .unwrap_or("unknown")
                .trim()
                .to_string();
            return HealthStatus::Unreachable(format!(
                "server returned HTTP {status_code}: {status_line}"
            ));
        }
    }

    // Parse the JSON body (after the blank line separating headers from body)
    let body = response
        .split("\r\n\r\n")
        .nth(1)
        .or_else(|| response.split("\n\n").nth(1))
        .unwrap_or("");

    if let Ok(json) = serde_json::from_str::<Value>(body.trim()) {
        if json["status"].as_str() == Some("ok") {
            return HealthStatus::Ready;
        }
    }

    HealthStatus::Loading
}

/// Extract the numeric HTTP status code from an HTTP response's status line.
/// Returns `None` if the response doesn't start with a valid HTTP status line.
fn parse_http_status_code(response: &str) -> Option<u16> {
    let status_line = response.lines().next()?;
    // HTTP status line format: "HTTP/1.x NNN ..."
    let parts: Vec<&str> = status_line.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    if !parts[0].starts_with("HTTP/") {
        return None;
    }
    parts[1].parse::<u16>().ok()
}

/// Parse a server URL for remote mode. Accepts `http://host:port`,
/// `https://host:port`, `host:port`, or bare `host` (defaults to port
/// 8080). Returns an error only if an explicit `:port` suffix fails to
/// parse as a `u16`.
fn parse_server_url(url: &str) -> Result<(String, u16)> {
    let stripped = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let stripped = stripped.trim_end_matches('/');

    // Handle IPv6 bracket notation: [::1]:8080 or [::1]
    if let Some(bracketed) = stripped.strip_prefix('[') {
        if let Some(bracket_end) = bracketed.find(']') {
            let host = format!("[{}]", &bracketed[..bracket_end]);
            let rest = &bracketed[bracket_end + 1..];
            if let Some(port_str) = rest.strip_prefix(':') {
                let port: u16 = port_str
                    .parse()
                    .with_context(|| format!("invalid port in llama-server URL: {url}"))?;
                return Ok((host, port));
            }
            return Ok((host, 8080));
        }
    }

    if let Some((host, port)) = stripped.rsplit_once(':') {
        let port: u16 = port
            .parse()
            .with_context(|| format!("invalid port in llama-server URL: {url}"))?;
        return Ok((host.to_string(), port));
    }

    Ok((stripped.to_string(), 8080))
}

fn build_chat_request(
    messages: &[ChatMessage],
    tools: &[ToolSpec],
    config: &GenerationConfig,
    model_alias: &str,
) -> Value {
    let mut request = json!({
        "model": model_alias,
        "messages": messages.iter().map(openai_message).collect::<Vec<_>>(),
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "stream": true,
        "repeat_penalty": config.repeat_penalty,
        "repeat_last_n": config.repeat_last_n,
        "id_slot": 0,
        // Gemma 4's thinking channel is gated at the Jinja template via the
        // `enable_thinking` kwarg. When enabled, llama-server emits the thought
        // process on `delta.reasoning_content`, which `parse_sse_events` routes
        // to `TokenEvent::Thinking` so UIs can render it distinctly. When
        // disabled, the template skips `<|think|>` and the model produces
        // direct answers only. See `wiki/pages/gemma4-format-spec.md`.
        "chat_template_kwargs": { "enable_thinking": config.enable_thinking },
    });

    if !tools.is_empty() {
        request["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                    })
                })
                .collect(),
        );
        request["tool_choice"] = Value::String("auto".to_string());
        request["parse_tool_calls"] = Value::Bool(true);
    }

    request
}

fn openai_message(message: &ChatMessage) -> Value {
    match message.role {
        Role::System => json!({
            "role": "system",
            "content": message.content,
        }),
        Role::User => json!({
            "role": "user",
            "content": message.content,
        }),
        Role::Tool => json!({
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }),
        Role::Model => {
            let mut base = json!({
                "role": "assistant",
                "content": message.content,
            });

            if let Some(tool_calls) = &message.tool_calls {
                base["tool_calls"] = Value::Array(
                    tool_calls
                        .iter()
                        .map(|call| {
                            json!({
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": serde_json::to_string(&call.arguments)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                }
                            })
                        })
                        .collect(),
                );
            }

            base
        }
    }
}

#[derive(Debug)]
enum SseEvent {
    Token(String),
    /// Gemma 4's private reasoning channel streamed by llama-server via
    /// `delta.reasoning_content`. Distinct from `Token` so the runtime
    /// can route it through a separate UI lane and avoid mixing it into
    /// the stored assistant history.
    Thinking(String),
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },
    FinishReason(String),
    Done,
}

fn parse_sse_events(line: &str) -> Vec<SseEvent> {
    let Some(data) = line.strip_prefix("data: ") else {
        return Vec::new();
    };
    let data = data.trim();
    if data.is_empty() {
        return Vec::new();
    }
    if data == "[DONE]" {
        return vec![SseEvent::Done];
    }

    let Ok(json) = serde_json::from_str::<Value>(data) else {
        return Vec::new();
    };
    let Some(choice) = json["choices"].get(0) else {
        return Vec::new();
    };

    let mut events: Vec<SseEvent> = Vec::new();

    if let Some(reason) = choice["finish_reason"].as_str() {
        if reason != "null" {
            events.push(SseEvent::FinishReason(reason.to_string()));
        }
    }

    let delta = &choice["delta"];

    if let Some(tool_calls) = delta["tool_calls"].as_array() {
        for tc in tool_calls {
            events.push(SseEvent::ToolCallDelta {
                index: usize::try_from(tc["index"].as_u64().unwrap_or(0)).unwrap_or(0),
                id: tc["id"].as_str().map(String::from),
                name: tc["function"]["name"].as_str().map(String::from),
                arguments: tc["function"]["arguments"].as_str().map(String::from),
            });
        }
    }

    // Gemma 4 thinking channel — llama-server emits this as `reasoning_content`
    // when `chat_template_kwargs.enable_thinking` is true. Kept separate from
    // `content` deltas so the runtime can render it in a distinct UI lane.
    if let Some(reasoning) = delta["reasoning_content"].as_str() {
        if !reasoning.is_empty() {
            events.push(SseEvent::Thinking(reasoning.to_string()));
        }
    }

    if let Some(content) = delta["content"].as_str() {
        if !content.is_empty() {
            events.push(SseEvent::Token(content.to_string()));
        }
    }

    events
}

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

/// Open a TCP connection, send an HTTP POST request, validate the 200
/// status line, and skip the response headers, returning a `BufReader`
/// positioned at the start of the SSE body.
fn open_sse_stream(host: &str, port: u16, request_body: &str) -> Result<BufReader<TcpStream>> {
    let mut stream =
        TcpStream::connect((host, port)).context("Failed to connect to llama-server")?;
    stream.set_read_timeout(Some(DEFAULT_REQUEST_TIMEOUT))?;
    stream.set_write_timeout(Some(DEFAULT_REQUEST_TIMEOUT))?;

    let http_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         User-Agent: zipcode/{version}\r\n\
         Accept: text/event-stream\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         \r\n\
         {request_body}",
        version = env!("CARGO_PKG_VERSION"),
        len = request_body.len(),
    );
    stream.write_all(http_req.as_bytes())?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);

    // Read status line
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    if !status_line.contains(" 200 ") {
        // Read body for error details
        let mut error_body = String::new();
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => error_body.push_str(&line),
            }
            if error_body.len() > 1024 {
                break;
            }
        }
        anyhow::bail!(
            "llama-server returned {}: {}",
            status_line.trim(),
            error_body.trim()
        );
    }

    // Skip headers
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
    }

    Ok(reader)
}

fn stream_sse_events(
    host: &str,
    port: u16,
    request: &Value,
    tx: &mpsc::Sender<TokenEvent>,
) -> Result<()> {
    let body = serde_json::to_string(request)?;
    let mut reader = open_sse_stream(host, port, &body)?;

    let mut tool_call_accum: Vec<ToolCallAccumulator> = Vec::new();
    let mut finish_reason = FinishReason::Stop;
    let (mut saw_done, mut stream_error) = (false, None);

    'sse: loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                stream_error = Some("llama-server SSE stream ended before [DONE]".to_string());
                break;
            }
            Ok(_) => {}
            Err(e)
                if e.kind() == std::io::ErrorKind::TimedOut
                    || e.kind() == std::io::ErrorKind::WouldBlock =>
            {
                stream_error =
                    Some("llama-server SSE stream timed out or stalled before [DONE]".to_string());
                break;
            }
            Err(e) => return Err(e.into()),
        }

        let line = line.trim();
        if line.chars().all(|c| c.is_ascii_hexdigit()) && !line.is_empty() {
            continue;
        }

        for event in parse_sse_events(line) {
            match event {
                SseEvent::Token(text) => {
                    if tx.send(TokenEvent::Token(text)).is_err() {
                        return Ok(());
                    }
                }
                SseEvent::Thinking(text) => {
                    if tx.send(TokenEvent::Thinking(text)).is_err() {
                        return Ok(());
                    }
                }
                SseEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments,
                } => {
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
                SseEvent::Done => {
                    saw_done = true;
                    break 'sse;
                }
            }
        }
    }

    if let Some(error) = stream_error {
        return Err(anyhow::anyhow!(error));
    }

    if !saw_done {
        return Err(anyhow::anyhow!(
            "llama-server SSE stream ended before a terminal event"
        ));
    }

    if !tool_call_accum.is_empty() {
        for acc in tool_call_accum {
            let arguments: serde_json::Value =
                serde_json::from_str(&acc.arguments).with_context(|| {
                    format!(
                        "llama-server returned an incomplete tool call for `{}`",
                        acc.name
                    )
                })?;
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

#[allow(dead_code)]
fn extract_message_content(message: &Value) -> String {
    if let Some(text) = message["content"].as_str() {
        return text.to_string();
    }

    if let Some(parts) = message["content"].as_array() {
        return parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect::<String>();
    }

    String::new()
}

#[allow(dead_code)]
fn parse_response_tool_calls(message: &Value) -> Vec<ToolCallParsed> {
    message["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|tool_call| {
            let id = tool_call["id"].as_str().unwrap_or("call_0").to_string();
            let name = tool_call["function"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let arguments = tool_call["function"]["arguments"]
                .as_str()
                .and_then(|text| serde_json::from_str(text).ok())
                .unwrap_or_else(|| json!({}));
            ToolCallParsed {
                id,
                name,
                arguments,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_template::ToolSpec;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::time::Duration;

    #[test]
    fn parse_sse_data_line_extracts_token() {
        let line = r#"data: {"choices":[{"delta":{"content":"hello"}}]}"#;
        let events = parse_sse_events(line);
        assert!(matches!(events.as_slice(), [SseEvent::Token(t)] if t == "hello"));
    }

    #[test]
    fn parse_sse_data_line_detects_done() {
        let line = "data: [DONE]";
        let events = parse_sse_events(line);
        assert!(matches!(events.as_slice(), [SseEvent::Done]));
    }

    /// Regression test for issue #46: when a chunk carries both a non-null
    /// `finish_reason` *and* `delta.content`, the parser must emit both
    /// `Token` and `FinishReason` events — previously it short-circuited on
    /// `finish_reason` and silently dropped the final assistant content.
    #[test]
    fn parse_sse_data_line_emits_token_and_finish_reason() {
        let line = r#"data: {"choices":[{"finish_reason":"stop","delta":{"content":"world"}}]}"#;
        let events = parse_sse_events(line);
        assert!(matches!(
            events.as_slice(),
            [SseEvent::FinishReason(r), SseEvent::Token(t)] if r == "stop" && t == "world"
        ));
    }

    /// Also ensure a finish_reason-only chunk (no content) still works.
    #[test]
    fn parse_sse_data_line_finish_reason_without_content() {
        let line = r#"data: {"choices":[{"finish_reason":"stop","delta":{}}]}"#;
        let events = parse_sse_events(line);
        assert!(matches!(
            events.as_slice(),
            [SseEvent::FinishReason(r)] if r == "stop"
        ));
    }

    #[test]
    fn parse_sse_data_line_extracts_tool_call_chunks() {
        let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":"{\"command\":"}}]}}]}"#;
        let events = parse_sse_events(line);
        assert!(matches!(
            events.as_slice(),
            [SseEvent::ToolCallDelta { .. }]
        ));
    }

    #[test]
    fn parse_sse_data_line_extracts_multiple_tool_call_chunks() {
        let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","function":{"name":"read_file","arguments":"{\"path\":"}},{"index":1,"id":"call_1","function":{"name":"tool_search","arguments":"{\"query\":"}}]}}]}"#;
        let events = parse_sse_events(line);
        assert!(matches!(
            events.as_slice(),
            [
                SseEvent::ToolCallDelta { index: 0, id: Some(id0), name: Some(name0), arguments: Some(args0) },
                SseEvent::ToolCallDelta { index: 1, id: Some(id1), name: Some(name1), arguments: Some(args1) },
            ] if id0 == "call_0"
                && name0 == "read_file"
                && args0 == "{\"path\":"
                && id1 == "call_1"
                && name1 == "tool_search"
                && args1 == "{\"query\":"
        ));
    }

    #[test]
    fn parse_sse_data_line_ignores_empty() {
        assert!(parse_sse_events("").is_empty());
        assert!(parse_sse_events(": comment").is_empty());
        assert!(parse_sse_events("data: ").is_empty());
    }

    /// Spawn a one-shot TCP server that responds with the given raw HTTP response.
    /// Returns the port number the server is listening on.
    fn serve_http_response(response: &'static str) -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(1)))
                .unwrap();
            let mut buf = [0_u8; 4096];
            let _ = stream.read(&mut buf);
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });
        port
    }

    /// Convenience wrapper: serve an SSE response with the standard event-stream headers.
    fn serve_sse_response(body: &'static str) -> u16 {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
        );
        // We leak the formatted string to get a &'static str for serve_http_response.
        // This is acceptable in test code where the number of calls is bounded.
        let response_static: &'static str = Box::leak(response.into_boxed_str());
        serve_http_response(response_static)
    }

    #[test]
    fn stream_sse_events_emits_multiple_tool_calls_from_one_chunk() {
        let port = serve_sse_response(
            concat!(
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[",
                "{\"index\":0,\"id\":\"call_0\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}},",
                "{\"index\":1,\"id\":\"call_1\",\"function\":{\"name\":\"tool_search\",\"arguments\":\"{\\\"query\\\":\\\"read_file\\\"}\"}}",
                "]}}]}\n\n",
                "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n",
                "data: [DONE]\n\n"
            ),
        );
        let request = json!({"stream": true});
        let (tx, rx) = mpsc::channel();

        stream_sse_events("127.0.0.1", port, &request, &tx).unwrap();
        drop(tx);

        let events: Vec<_> = rx.into_iter().collect();
        let tool_calls: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                TokenEvent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();

        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].id, "call_0");
        assert_eq!(tool_calls[0].name, "read_file");
        assert_eq!(tool_calls[0].arguments["path"], "README.md");
        assert_eq!(tool_calls[1].id, "call_1");
        assert_eq!(tool_calls[1].name, "tool_search");
        assert_eq!(tool_calls[1].arguments["query"], "read_file");
        assert!(matches!(
            events.last(),
            Some(TokenEvent::Done(FinishReason::ToolUse))
        ));
    }

    #[test]
    fn stream_sse_events_stops_after_done() {
        let port = serve_sse_response(concat!(
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n"
        ));
        let request = json!({"stream": true});
        let (tx, rx) = mpsc::channel();

        stream_sse_events("127.0.0.1", port, &request, &tx).unwrap();
        drop(tx);

        let events: Vec<_> = rx.into_iter().collect();
        assert!(matches!(
            events.as_slice(),
            [TokenEvent::Done(FinishReason::Stop)]
        ));
    }

    #[test]
    fn stream_sse_events_errors_on_truncated_tool_call_stream() {
        let port = serve_sse_response(concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[",
            "{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\"}}",
            "]}}]}\n\n"
        ));
        let request = json!({"stream": true});
        let (tx, rx) = mpsc::channel();

        let error = stream_sse_events("127.0.0.1", port, &request, &tx)
            .unwrap_err()
            .to_string();
        drop(tx);

        let events: Vec<_> = rx.into_iter().collect();
        assert!(
            events.is_empty(),
            "truncated stream should not emit tool calls"
        );
        assert!(
            error.contains("before [DONE]"),
            "unexpected error for truncated SSE stream: {error}"
        );
    }

    #[test]
    fn build_chat_request_includes_tools() {
        let request = build_chat_request(
            &[ChatMessage::system("sys"), ChatMessage::user("hi")],
            &[ToolSpec {
                name: "read_file".to_string(),
                description: "Read a file".to_string(),
                parameters: json!({"type": "object"}),
            }],
            &GenerationConfig::default(),
            DEFAULT_ALIAS,
        );

        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(request["messages"][1]["role"], "user");
        assert_eq!(request["tools"][0]["function"]["name"], "read_file");
        assert_eq!(request["tool_choice"], "auto");
        assert_eq!(request["parse_tool_calls"], true);
    }

    /// Default `GenerationConfig` has `enable_thinking: true`, and the kwarg
    /// is forwarded to llama-server so Gemma 4 streams its reasoning channel.
    /// The runtime routes `delta.reasoning_content` through `TokenEvent::Thinking`.
    /// See `wiki/pages/gemma4-format-spec.md` § "Empirical wire contract".
    #[test]
    fn build_chat_request_enables_gemma4_thinking_mode_by_default() {
        let request = build_chat_request(
            &[ChatMessage::user("hi")],
            &[],
            &GenerationConfig::default(),
            DEFAULT_ALIAS,
        );

        assert_eq!(
            request["chat_template_kwargs"]["enable_thinking"],
            serde_json::Value::Bool(true),
            "default config should request thinking so the UI can render reasoning deltas",
        );
    }

    /// Callers that explicitly want direct answers (benchmarks, latency-sensitive
    /// automation) can set `config.enable_thinking = false` and the request
    /// must propagate that choice to the template kwarg.
    #[test]
    fn build_chat_request_honors_disabled_thinking_config() {
        let config = GenerationConfig {
            enable_thinking: false,
            ..GenerationConfig::default()
        };

        let request = build_chat_request(&[ChatMessage::user("hi")], &[], &config, DEFAULT_ALIAS);

        assert_eq!(
            request["chat_template_kwargs"]["enable_thinking"],
            serde_json::Value::Bool(false),
            "explicit opt-out must suppress the template thinking token",
        );
    }

    #[test]
    fn parse_sse_data_line_extracts_reasoning_as_thinking() {
        let line =
            r#"data: {"choices":[{"delta":{"reasoning_content":"I should call the tool."}}]}"#;
        let events = parse_sse_events(line);
        assert!(
            matches!(events.as_slice(), [SseEvent::Thinking(t)] if t == "I should call the tool."),
            "reasoning_content delta should produce SseEvent::Thinking, got: {events:?}"
        );
    }

    #[test]
    fn parse_sse_data_line_ignores_empty_reasoning() {
        let line = r#"data: {"choices":[{"delta":{"reasoning_content":""}}]}"#;
        assert!(
            parse_sse_events(line).is_empty(),
            "empty reasoning delta should not produce a Thinking event"
        );
    }

    /// End-to-end proof that a mixed stream — reasoning chunks followed by
    /// tool_call chunks — surfaces both as distinct TokenEvent variants and
    /// does not leak reasoning into the regular token lane.
    #[test]
    fn stream_sse_events_routes_reasoning_and_tool_calls_separately() {
        let port = serve_sse_response(concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"The user wants \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"the readme.\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_r\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        ));
        let request = json!({"stream": true});
        let (tx, rx) = mpsc::channel();

        stream_sse_events("127.0.0.1", port, &request, &tx).unwrap();
        drop(tx);

        let events: Vec<_> = rx.into_iter().collect();

        let thinking: String = events
            .iter()
            .filter_map(|event| match event {
                TokenEvent::Thinking(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking, "The user wants the readme.");

        let tokens: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                TokenEvent::Token(text) => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert!(
            tokens.is_empty(),
            "reasoning chunks must not leak into the regular token lane: {tokens:?}"
        );

        let tool_calls: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                TokenEvent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "read_file");
        assert_eq!(tool_calls[0].arguments["path"], "README.md");

        assert!(matches!(
            events.last(),
            Some(TokenEvent::Done(FinishReason::ToolUse))
        ));
    }

    #[test]
    fn parse_response_tool_calls_reads_openai_shape() {
        let message = json!({
            "content": null,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": "{\"command\":\"ls\"}"
                    }
                }
            ]
        });

        let calls = parse_response_tool_calls(&message);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn extract_message_content_supports_block_arrays() {
        let message = json!({
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": " world"}
            ]
        });

        assert_eq!(extract_message_content(&message), "hello world");
    }

    #[test]
    fn server_options_default_has_no_gpu_layers() {
        let opts = ServerOptions::default();
        assert_eq!(opts.gpu_layers, None);
        assert!(!opts.flash_attention);
        assert_eq!(opts.context_size, DEFAULT_CONTEXT_SIZE);
    }

    /// `ZIPCODE_LLAMA_SERVER_URL` accepts several convenient shapes so
    /// users can paste whatever form their remote host documentation
    /// suggests without having to remember a strict format.
    #[test]
    fn parse_server_url_accepts_common_shapes() {
        assert_eq!(
            parse_server_url("http://192.168.1.10:8080").unwrap(),
            ("192.168.1.10".to_string(), 8080)
        );
        assert_eq!(
            parse_server_url("https://gpu-box.lan:5555").unwrap(),
            ("gpu-box.lan".to_string(), 5555)
        );
        assert_eq!(
            parse_server_url("127.0.0.1:9090").unwrap(),
            ("127.0.0.1".to_string(), 9090)
        );
        // bare host defaults to port 8080
        assert_eq!(
            parse_server_url("llamahost").unwrap(),
            ("llamahost".to_string(), 8080)
        );
        // trailing slash is tolerated
        assert_eq!(
            parse_server_url("http://10.0.0.1:44444/").unwrap(),
            ("10.0.0.1".to_string(), 44444)
        );
    }

    #[test]
    fn parse_server_url_rejects_nonnumeric_port() {
        let err = parse_server_url("http://host:notaport")
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid port"));
    }

    #[test]
    fn flash_attention_uses_legacy_compatible_flag() {
        assert_eq!(
            flash_attention_mode_from_help(
                "-fa, --flash-attn                     enable Flash Attention (default: disabled)"
            ),
            FlashAttentionMode::LegacyFlagOnly
        );
        assert_eq!(
            flash_attention_mode_from_help(
                "-fa, --flash-attn [on|off|auto]       set Flash Attention use"
            ),
            FlashAttentionMode::ExplicitOnValue
        );
    }

    #[test]
    #[ignore = "requires ZIPCODE_TEST_MODEL_PATH and llama-server on PATH"]
    fn sse_streaming_returns_tokens_incrementally() {
        let model_path =
            std::env::var("ZIPCODE_TEST_MODEL_PATH").expect("ZIPCODE_TEST_MODEL_PATH must be set");
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

    // ── openai_message tests ────────────────────────────────────────

    #[test]
    fn openai_message_system_role() {
        let msg = ChatMessage::system("you are helpful");
        let value = openai_message(&msg);
        assert_eq!(value["role"], "system");
        assert_eq!(value["content"], "you are helpful");
    }

    #[test]
    fn openai_message_user_role() {
        let msg = ChatMessage::user("hello");
        let value = openai_message(&msg);
        assert_eq!(value["role"], "user");
        assert_eq!(value["content"], "hello");
    }

    #[test]
    fn openai_message_tool_result_role() {
        let msg = ChatMessage::tool_result("call_42", "file contents");
        let value = openai_message(&msg);
        assert_eq!(value["role"], "tool");
        assert_eq!(value["tool_call_id"], "call_42");
        assert_eq!(value["content"], "file contents");
    }

    #[test]
    fn openai_message_model_with_tool_calls() {
        let call = ToolCallParsed {
            id: "call_1".to_string(),
            name: "bash".to_string(),
            arguments: json!({"command": "ls"}),
        };
        let msg = ChatMessage::assistant_with_tool_calls("thinking", vec![call]);
        let value = openai_message(&msg);
        assert_eq!(value["role"], "assistant");
        assert_eq!(value["content"], "thinking");
        let tc = &value["tool_calls"][0];
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["function"]["name"], "bash");
        assert_eq!(tc["type"], "function");
        // arguments should be a JSON string, not an object
        let args: Value =
            serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["command"], "ls");
    }

    #[test]
    fn openai_message_model_without_tool_calls() {
        let msg = ChatMessage::assistant("here is the answer");
        let value = openai_message(&msg);
        assert_eq!(value["role"], "assistant");
        assert_eq!(value["content"], "here is the answer");
        assert!(value.get("tool_calls").is_none());
    }

    // ── parse_server_url IPv6 tests ─────────────────────────────────

    #[test]
    fn parse_server_url_handles_ipv6_with_port() {
        assert_eq!(
            parse_server_url("http://[::1]:8080").unwrap(),
            ("[::1]".to_string(), 8080)
        );
    }

    #[test]
    fn parse_server_url_handles_ipv6_without_port() {
        assert_eq!(
            parse_server_url("[::1]").unwrap(),
            ("[::1]".to_string(), 8080)
        );
    }

    #[test]
    fn parse_server_url_handles_ipv6_full_address_with_port() {
        assert_eq!(
            parse_server_url("http://[2001:db8::1]:5555").unwrap(),
            ("[2001:db8::1]".to_string(), 5555)
        );
    }

    #[test]
    fn parse_server_url_handles_ipv6_https() {
        assert_eq!(
            parse_server_url("https://[::1]:443").unwrap(),
            ("[::1]".to_string(), 443)
        );
    }

    // ── health_check JSON parsing test ──────────────────────────────

    #[test]
    fn health_check_parses_json_body_not_headers() {
        // Verify that a response where "ok" appears in a header but
        // the body says "loading" correctly reports Loading.
        let response = "HTTP/1.1 200 OK\r\nX-Status: \"ok\"\r\nContent-Type: application/json\r\n\r\n{\"status\":\"loading\"}";
        let port = serve_http_response(response);

        let result = health_check("127.0.0.1", port);
        assert!(
            matches!(result, HealthStatus::Loading),
            "health_check should parse JSON body, not match headers — got {result:?}"
        );
    }

    // ── parse_http_status_code tests ──────────────────────────────────

    #[test]
    fn parse_http_status_code_extracts_200() {
        let response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(parse_http_status_code(response), Some(200));
    }

    #[test]
    fn parse_http_status_code_extracts_500() {
        let response = "HTTP/1.1 500 Internal Server Error\r\n\r\n";
        assert_eq!(parse_http_status_code(response), Some(500));
    }

    #[test]
    fn parse_http_status_code_extracts_503() {
        let response = "HTTP/1.0 503 Service Unavailable\r\n\r\n";
        assert_eq!(parse_http_status_code(response), Some(503));
    }

    #[test]
    fn parse_http_status_code_returns_none_for_garbage() {
        assert_eq!(parse_http_status_code("not http"), None);
        assert_eq!(parse_http_status_code(""), None);
    }

    #[test]
    fn parse_http_status_code_returns_none_for_missing_code() {
        assert_eq!(parse_http_status_code("HTTP/1.1\r\n"), None);
    }

    // ── health_check HTTP status code validation tests ────────────────

    #[test]
    fn health_check_returns_unreachable_on_http_500() {
        // A 500 response should NOT be treated as Loading — it must
        // surface immediately so the startup loop can report the error
        // instead of polling for 180 seconds.
        let response = "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n\r\n{\"error\":\"model failed to load\"}";
        let port = serve_http_response(response);

        let result = health_check("127.0.0.1", port);
        match result {
            HealthStatus::Unreachable(msg) => {
                assert!(
                    msg.contains("HTTP 500"),
                    "error message should mention HTTP 500, got: {msg}"
                );
            }
            other => panic!("expected Unreachable for HTTP 500, got {other:?}"),
        }
    }

    #[test]
    fn health_check_returns_unreachable_on_http_503() {
        let response = "HTTP/1.1 503 Service Unavailable\r\n\r\ntry later";
        let port = serve_http_response(response);

        let result = health_check("127.0.0.1", port);
        assert!(
            matches!(result, HealthStatus::Unreachable(_)),
            "HTTP 503 should be Unreachable, got {result:?}"
        );
    }

    #[test]
    fn health_check_returns_ready_on_http_200_with_status_ok() {
        let response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ok\"}";
        let port = serve_http_response(response);

        let result = health_check("127.0.0.1", port);
        assert!(
            matches!(result, HealthStatus::Ready),
            "HTTP 200 with status:ok should be Ready, got {result:?}"
        );
    }

    #[test]
    fn health_check_returns_loading_on_http_200_without_status_ok() {
        let response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"loading\"}";
        let port = serve_http_response(response);

        let result = health_check("127.0.0.1", port);
        assert!(
            matches!(result, HealthStatus::Loading),
            "HTTP 200 with status:loading should still be Loading, got {result:?}"
        );
    }

    #[test]
    fn health_check_returns_unreachable_on_http_400() {
        let response = "HTTP/1.1 400 Bad Request\r\n\r\ninvalid";
        let port = serve_http_response(response);

        let result = health_check("127.0.0.1", port);
        assert!(
            matches!(result, HealthStatus::Unreachable(_)),
            "HTTP 400 should be Unreachable, got {result:?}"
        );
    }

    // ── Additional edge-case tests (cron audit 2026-04-23) ──────────

    // ── extract_message_content edge cases ──────────────────────────

    #[test]
    fn extract_message_content_returns_empty_for_null_content() {
        let message = json!({"content": null});
        assert_eq!(extract_message_content(&message), "");
    }

    #[test]
    fn extract_message_content_returns_empty_for_missing_content() {
        let message = json!({"role": "assistant"});
        assert_eq!(extract_message_content(&message), "");
    }

    #[test]
    fn extract_message_content_returns_empty_for_empty_array() {
        let message = json!({"content": []});
        assert_eq!(extract_message_content(&message), "");
    }

    #[test]
    fn extract_message_content_filters_non_text_parts() {
        let message = json!({
            "content": [
                {"type": "image", "url": "http://example.com/img.png"},
                {"type": "text", "text": "only this"},
                {"type": "image_url", "image_url": {"url": "http://example.com"}},
            ]
        });
        assert_eq!(extract_message_content(&message), "only this");
    }

    #[test]
    fn extract_message_content_handles_unicode() {
        let message = json!({"content": "안녕하세요 🌍"});
        assert_eq!(extract_message_content(&message), "안녕하세요 🌍");
    }

    // ── parse_response_tool_calls edge cases ────────────────────────

    #[test]
    fn parse_response_tool_calls_returns_empty_when_no_tool_calls() {
        let message = json!({"content": "hello", "role": "assistant"});
        assert!(parse_response_tool_calls(&message).is_empty());
    }

    #[test]
    fn parse_response_tool_calls_handles_null_tool_calls() {
        let message = json!({"content": null, "tool_calls": null});
        assert!(parse_response_tool_calls(&message).is_empty());
    }

    #[test]
    fn parse_response_tool_calls_handles_empty_array() {
        let message = json!({"tool_calls": []});
        assert!(parse_response_tool_calls(&message).is_empty());
    }

    #[test]
    fn parse_response_tool_calls_multiple_calls() {
        let message = json!({
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"a.rs\"}"}
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
                },
                {
                    "id": "call_c",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{\"path\":\"out.txt\",\"content\":\"hi\"}"}
                }
            ]
        });
        let calls = parse_response_tool_calls(&message);
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].id, "call_a");
        assert_eq!(calls[1].name, "bash");
        assert_eq!(calls[2].name, "write_file");
        assert_eq!(calls[2].arguments["content"], "hi");
    }

    #[test]
    fn parse_response_tool_calls_defaults_on_missing_fields() {
        let message = json!({
            "tool_calls": [
                {"function": {"name": "grep"}}
            ]
        });
        let calls = parse_response_tool_calls(&message);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].name, "grep");
        // Missing arguments string defaults to empty JSON object
        assert_eq!(calls[0].arguments, json!({}));
    }

    #[test]
    fn parse_response_tool_calls_handles_invalid_arguments_json() {
        let message = json!({
            "tool_calls": [
                {
                    "id": "call_x",
                    "function": {"name": "bash", "arguments": "not valid json{{{"}
                }
            ]
        });
        let calls = parse_response_tool_calls(&message);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        // Invalid arguments JSON falls back to empty object
        assert_eq!(calls[0].arguments, json!({}));
    }

    // ── openai_message edge cases ───────────────────────────────────

    #[test]
    fn openai_message_model_plain_no_tool_calls_key() {
        let msg = ChatMessage::assistant("plain text answer");
        let value = openai_message(&msg);
        assert_eq!(value["role"], "assistant");
        assert_eq!(value["content"], "plain text answer");
        // Must not have tool_calls key at all (not null, not empty array)
        assert!(
            value.get("tool_calls").is_none(),
            "assistant without tool_calls should not have the key"
        );
    }

    #[test]
    fn openai_message_model_with_multiple_tool_calls() {
        let calls = vec![
            ToolCallParsed {
                id: "c1".to_string(),
                name: "bash".to_string(),
                arguments: json!({"command": "ls"}),
            },
            ToolCallParsed {
                id: "c2".to_string(),
                name: "read_file".to_string(),
                arguments: json!({"path": "src/main.rs"}),
            },
            ToolCallParsed {
                id: "c3".to_string(),
                name: "glob_search".to_string(),
                arguments: json!({"pattern": "*.rs"}),
            },
        ];
        let msg = ChatMessage::assistant_with_tool_calls("", calls);
        let value = openai_message(&msg);
        let tc_array = value["tool_calls"].as_array().unwrap();
        assert_eq!(tc_array.len(), 3);
        assert_eq!(tc_array[0]["function"]["name"], "bash");
        assert_eq!(tc_array[1]["function"]["name"], "read_file");
        assert_eq!(tc_array[2]["function"]["name"], "glob_search");
    }

    #[test]
    fn openai_message_empty_content() {
        let msg = ChatMessage::user("");
        let value = openai_message(&msg);
        assert_eq!(value["content"], "");
    }

    #[test]
    fn openai_message_unicode_content() {
        let msg = ChatMessage::user("파일을 읽어줘 📂");
        let value = openai_message(&msg);
        assert_eq!(value["content"], "파일을 읽어줘 📂");
    }

    // ── parse_sse_events edge cases ─────────────────────────────────

    #[test]
    fn parse_sse_events_ignores_invalid_json() {
        let line = "data: {not valid json at all";
        assert!(parse_sse_events(line).is_empty());
    }

    #[test]
    fn parse_sse_events_ignores_missing_choices() {
        let line = r#"data: {"id":"chatcmpl-1"}"#;
        assert!(parse_sse_events(line).is_empty());
    }

    #[test]
    fn parse_sse_events_ignores_empty_choices_array() {
        let line = r#"data: {"choices":[]}"#;
        assert!(parse_sse_events(line).is_empty());
    }

    #[test]
    fn parse_sse_events_handles_content_and_tool_calls_in_same_delta() {
        let line = r#"data: {"choices":[{"delta":{"content":"thinking","tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":""}}]}}]}"#;
        let events = parse_sse_events(line);
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], SseEvent::ToolCallDelta { .. }));
        assert!(matches!(events[1], SseEvent::Token(ref t) if t == "thinking"));
    }

    #[test]
    fn parse_sse_events_ignores_null_finish_reason() {
        let line = r#"data: {"choices":[{"finish_reason":null,"delta":{"content":"hi"}}]}"#;
        let events = parse_sse_events(line);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], SseEvent::Token(ref t) if t == "hi"));
    }

    #[test]
    fn parse_sse_events_handles_reasoning_and_content_together() {
        let line = r#"data: {"choices":[{"delta":{"reasoning_content":"I think","content":"therefore"}}]}"#;
        let events = parse_sse_events(line);
        assert_eq!(events.len(), 2);
        // Thinking comes first in the event order
        assert!(matches!(events[0], SseEvent::Thinking(ref t) if t == "I think"));
        assert!(matches!(events[1], SseEvent::Token(ref t) if t == "therefore"));
    }

    #[test]
    fn parse_sse_events_tool_call_delta_with_missing_fields() {
        let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":2}]}}]}"#;
        let events = parse_sse_events(line);
        assert!(matches!(
            events.as_slice(),
            [SseEvent::ToolCallDelta {
                index: 2,
                id: None,
                name: None,
                arguments: None
            }]
        ));
    }

    #[test]
    fn parse_sse_events_finish_reason_length() {
        let line = r#"data: {"choices":[{"finish_reason":"length","delta":{}}]}"#;
        let events = parse_sse_events(line);
        assert!(matches!(
            events.as_slice(),
            [SseEvent::FinishReason(ref r)] if r == "length"
        ));
    }

    #[test]
    fn parse_sse_events_finish_reason_tool_calls() {
        let line = r#"data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}"#;
        let events = parse_sse_events(line);
        assert!(matches!(
            events.as_slice(),
            [SseEvent::FinishReason(ref r)] if r == "tool_calls"
        ));
    }

    // ── build_chat_request edge cases ───────────────────────────────

    #[test]
    fn build_chat_request_no_tools_omits_tool_fields() {
        let request = build_chat_request(
            &[ChatMessage::user("hi")],
            &[],
            &GenerationConfig::default(),
            "test-model",
        );
        assert_eq!(request["model"], "test-model");
        assert!(request.get("tools").is_none());
        assert!(request.get("tool_choice").is_none());
        assert!(request.get("parse_tool_calls").is_none());
    }

    #[test]
    fn build_chat_request_custom_config_values() {
        let config = GenerationConfig {
            max_tokens: 512,
            temperature: 0.5,
            top_p: 0.8,
            top_k: 20,
            repeat_penalty: 1.2,
            repeat_last_n: 32,
            enable_thinking: false,
        };
        let request =
            build_chat_request(&[ChatMessage::user("test")], &[], &config, "custom-alias");
        assert_eq!(request["max_tokens"], 512);
        assert_eq!(request["temperature"], 0.5);
        assert_eq!(request["top_p"], 0.8);
        assert_eq!(request["top_k"], 20);
        // f32 serializes with extra precision digits
        let rp = request["repeat_penalty"].as_f64().unwrap();
        assert!(
            (rp - 1.2).abs() < 0.01,
            "repeat_penalty should be ~1.2, got {rp}"
        );
        assert_eq!(request["repeat_last_n"], 32);
        assert_eq!(request["model"], "custom-alias");
        assert_eq!(request["stream"], true);
    }

    #[test]
    fn build_chat_request_multiple_tools() {
        let tools = vec![
            ToolSpec {
                name: "bash".to_string(),
                description: "Run a command".to_string(),
                parameters: json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            },
            ToolSpec {
                name: "read_file".to_string(),
                description: "Read file contents".to_string(),
                parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            },
        ];
        let request = build_chat_request(
            &[ChatMessage::user("do stuff")],
            &tools,
            &GenerationConfig::default(),
            DEFAULT_ALIAS,
        );
        let tools_array = request["tools"].as_array().unwrap();
        assert_eq!(tools_array.len(), 2);
        assert_eq!(tools_array[0]["type"], "function");
        assert_eq!(tools_array[0]["function"]["name"], "bash");
        assert_eq!(tools_array[1]["function"]["name"], "read_file");
    }

    // ── flash_attention_mode_from_help edge cases ───────────────────

    #[test]
    fn flash_attention_mode_from_help_empty_string() {
        assert_eq!(
            flash_attention_mode_from_help(""),
            FlashAttentionMode::LegacyFlagOnly
        );
    }

    #[test]
    fn flash_attention_mode_from_help_unrelated_help_text() {
        assert_eq!(
            flash_attention_mode_from_help("some other help text without flash attention keywords"),
            FlashAttentionMode::LegacyFlagOnly
        );
    }

    #[test]
    fn flash_attention_mode_from_help_set_usage_phrase() {
        assert_eq!(
            flash_attention_mode_from_help(
                "-fa, --flash-attn [on|off|auto]  set Flash Attention use"
            ),
            FlashAttentionMode::ExplicitOnValue
        );
    }

    // ── stream_sse_events finish_reason mapping ─────────────────────

    #[test]
    fn stream_sse_maps_length_to_max_tokens() {
        let port = serve_sse_response(concat!(
            "data: {\"choices\":[{\"finish_reason\":\"length\",\"delta\":{}}]}\n\n",
            "data: [DONE]\n\n"
        ));
        let request = json!({"stream": true});
        let (tx, rx) = mpsc::channel();

        stream_sse_events("127.0.0.1", port, &request, &tx).unwrap();
        drop(tx);

        let events: Vec<_> = rx.into_iter().collect();
        assert!(matches!(
            events.last(),
            Some(TokenEvent::Done(FinishReason::MaxTokens))
        ));
    }

    #[test]
    fn stream_sse_maps_tool_calls_finish_to_tool_use() {
        let port = serve_sse_response(concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"tc1\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"tool_calls\",\"delta\":{}}]}\n\n",
            "data: [DONE]\n\n"
        ));
        let request = json!({"stream": true});
        let (tx, rx) = mpsc::channel();

        stream_sse_events("127.0.0.1", port, &request, &tx).unwrap();
        drop(tx);

        let events: Vec<_> = rx.into_iter().collect();
        assert!(matches!(
            events.last(),
            Some(TokenEvent::Done(FinishReason::ToolUse))
        ));
    }
}
