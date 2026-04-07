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
        if options.flash_attention {
            command.arg("-fa");
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

        match http_request(
            port,
            "GET",
            "/v1/models",
            None,
            None,
            Duration::from_secs(2),
        ) {
            Ok(_) => {
                std::thread::sleep(Duration::from_secs(1));
                return Ok(());
            }
            Err(error) if Instant::now() < deadline => {
                debug!(port, error = %error, "waiting for llama-server to accept requests");
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(error) => {
                anyhow::bail!("Timed out waiting for llama-server on port {port}: {error}");
            }
        }
    }
}

fn http_request(
    port: u16,
    method: &str,
    path: &str,
    content_type: Option<&str>,
    body: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .with_context(|| format!("Failed to connect to llama-server on port {port}"))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let body = body.unwrap_or("");
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUser-Agent: zipcode/{version}\r\nAccept: application/json\r\n",
        version = env!("CARGO_PKG_VERSION"),
    );
    if let Some(content_type) = content_type {
        request.push_str(&format!("Content-Type: {content_type}\r\n"));
    }
    if !body.is_empty() {
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    request.push_str("\r\n");
    request.push_str(body);

    stream.write_all(request.as_bytes())?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    if status_line.trim().is_empty() {
        anyhow::bail!("Empty HTTP response from llama-server");
    }

    let mut headers = Vec::new();
    let mut content_length = None;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }

        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse::<usize>().ok();
            }
        }
        headers.push(line);
    }

    let mut body_bytes = Vec::new();
    if let Some(length) = content_length {
        body_bytes.resize(length, 0);
        reader.read_exact(&mut body_bytes)?;
    } else {
        reader.read_to_end(&mut body_bytes)?;
    }

    let body =
        String::from_utf8(body_bytes).context("llama-server returned a non-UTF-8 response body")?;
    if !status_line.contains(" 200 ") {
        anyhow::bail!("llama-server returned {status_line}: {body}");
    }

    Ok(body)
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

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

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
        anyhow::bail!("llama-server returned {}: {}", status_line.trim(), error_body.trim());
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

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => break,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(e) => return Err(e.into()),
        }

        let line = line.trim();
        // Skip chunked transfer encoding hex size lines
        if line.chars().all(|c| c.is_ascii_hexdigit()) && !line.is_empty() {
            continue;
        }

        let Some(event) = parse_sse_line(line) else {
            continue;
        };

        match event {
            SseEvent::Token(text) => {
                if tx.send(TokenEvent::Token(text)).is_err() {
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
}
