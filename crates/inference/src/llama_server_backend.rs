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
const DEFAULT_CONTEXT_SIZE: usize = 8192;

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
    child: Child,
    port: u16,
    model_alias: String,
    config: GenerationConfig,
}

impl LlamaServerProvider {
    /// Launch a local llama.cpp server for the given GGUF model.
    ///
    /// The binary is resolved from `ZIPCODE_LLAMA_SERVER_BIN`, `LLAMA_SERVER_BIN`,
    /// or `llama-server` on PATH.
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
        for arg in flash_attention_args(&binary, options.flash_attention) {
            command.arg(arg);
        }

        let cache_dir = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
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

        wait_until_ready(&mut child, port)?;

        Ok(Self {
            child,
            port,
            model_alias,
            config: GenerationConfig::default(),
        })
    }

    pub fn set_config(&mut self, config: GenerationConfig) {
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
        let port = self.port;

        std::thread::spawn(move || {
            if let Err(e) = stream_sse_events(port, &request, &tx) {
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
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
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

fn wait_until_ready(child: &mut Child, port: u16) -> Result<()> {
    let deadline = Instant::now() + DEFAULT_STARTUP_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait()? {
            anyhow::bail!("llama-server exited before becoming ready (status: {status})");
        }

        if Instant::now() >= deadline {
            anyhow::bail!("Timed out waiting for llama-server on port {port}");
        }

        match health_check(port) {
            HealthStatus::Ready => return Ok(()),
            HealthStatus::Loading => {
                debug!(port, "llama-server model still loading");
                std::thread::sleep(Duration::from_millis(500));
            }
            HealthStatus::Unreachable(error) => {
                debug!(port, error = %error, "waiting for llama-server to accept requests");
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
}

enum HealthStatus {
    Ready,
    Loading,
    Unreachable(String),
}

fn health_check(port: u16) -> HealthStatus {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return HealthStatus::Unreachable("connection refused".to_string());
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));

    let request =
        format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() || stream.flush().is_err() {
        return HealthStatus::Unreachable("write failed".to_string());
    }

    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf);
    let response = String::from_utf8_lossy(&buf);

    if response.contains("\"ok\"") {
        HealthStatus::Ready
    } else {
        HealthStatus::Loading
    }
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

    if let Some(reason) = choice["finish_reason"].as_str() {
        if reason != "null" {
            return vec![SseEvent::FinishReason(reason.to_string())];
        }
    }

    let delta = &choice["delta"];

    if let Some(tool_calls) = delta["tool_calls"].as_array() {
        let events = tool_calls
            .iter()
            .map(|tc| SseEvent::ToolCallDelta {
                index: tc["index"].as_u64().unwrap_or(0) as usize,
                id: tc["id"].as_str().map(String::from),
                name: tc["function"]["name"].as_str().map(String::from),
                arguments: tc["function"]["arguments"].as_str().map(String::from),
            })
            .collect::<Vec<_>>();
        if !events.is_empty() {
            return events;
        }
    }

    // Gemma 4 thinking channel — llama-server emits this as `reasoning_content`
    // when `chat_template_kwargs.enable_thinking` is true. Kept separate from
    // `content` deltas so the runtime can render it in a distinct UI lane.
    if let Some(reasoning) = delta["reasoning_content"].as_str() {
        if !reasoning.is_empty() {
            return vec![SseEvent::Thinking(reasoning.to_string())];
        }
    }

    if let Some(content) = delta["content"].as_str() {
        if !content.is_empty() {
            return vec![SseEvent::Token(content.to_string())];
        }
    }

    Vec::new()
}

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

fn stream_sse_events(port: u16, request: &Value, tx: &mpsc::Sender<TokenEvent>) -> Result<()> {
    let body = serde_json::to_string(request)?;

    let mut stream =
        TcpStream::connect(("127.0.0.1", port)).context("Failed to connect to llama-server")?;
    stream.set_read_timeout(Some(DEFAULT_REQUEST_TIMEOUT))?;
    stream.set_write_timeout(Some(DEFAULT_REQUEST_TIMEOUT))?;

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

    // Read SSE events
    let mut tool_call_accum: Vec<ToolCallAccumulator> = Vec::new();
    let mut finish_reason = FinishReason::Stop;
    let mut saw_done = false;
    let mut stream_error = None;

    'sse: loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                stream_error = Some("llama-server SSE stream ended before [DONE]".to_string());
                break;
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                stream_error = Some("llama-server SSE stream timed out before [DONE]".to_string());
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                stream_error = Some("llama-server SSE stream stalled before [DONE]".to_string());
                break;
            }
            Err(e) => return Err(e.into()),
        }

        let line = line.trim();
        // Skip chunked transfer encoding hex size lines
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

    // Emit accumulated tool calls
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

    fn serve_sse_response(body: &'static str) -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(1)))
                .unwrap();
            let mut buf = [0_u8; 4096];
            let _ = stream.read(&mut buf);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });
        port
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

        stream_sse_events(port, &request, &tx).unwrap();
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

        stream_sse_events(port, &request, &tx).unwrap();
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

        let error = stream_sse_events(port, &request, &tx)
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

        stream_sse_events(port, &request, &tx).unwrap();
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
}
