use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use zipcode_inference::{
    create_engine, Backend, GenerationConfig, ServerOptions, DEFAULT_CONTEXT_SIZE,
};
use zipcode_runtime::config::{expand_user_path, find_project_root, resolve_project_path};
use zipcode_runtime::prompt::build_system_prompt;
use zipcode_runtime::{
    parse_permission_mode, CompactPolicy, CompactResult, ConversationLoop, PermissionPolicy,
    Session, ZipcodeConfig,
};
use zipcode_tools::{
    agent::AgentTool, bash::BashTool, edit_file::EditFileTool, glob_search::GlobSearchTool,
    grep_search::GrepSearchTool, read_file::ReadFileTool, repl::ReplTool,
    todo_write::TodoWriteTool, tool_search::ToolSearchTool, write_file::WriteFileTool,
    ToolRegistry,
};

use crate::render::{print_tool_result, print_tool_start, Spinner};

pub struct LoopLaunch {
    pub conv: ConversationLoop,
    pub effective_backend: Backend,
    pub startup_notices: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct HelperDiscovery {
    pub path: Option<PathBuf>,
    pub issue: Option<String>,
    pub issue_is_blocking: bool,
}

/// CLI callback that renders streaming tokens and tool events.
///
/// Manages a background spinner during model inference and between
/// tool execution rounds to indicate activity.
pub struct CliCallback {
    spinner: Option<Spinner>,
    /// Set while the model is streaming a thinking chunk so the next
    /// transition (token, tool, error) can close the ANSI dim/italic
    /// attributes cleanly and emit a visual separator.
    thinking_active: bool,
    /// True until the first visible token is printed in this turn.
    /// Used to emit a `Zip: ` role prefix before the first token,
    /// matching the TUI's `Zip:` transcript entries.
    needs_role_prefix: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlashCommand {
    Help,
    Quit,
    Clear,
    Status,
    Compact,
    SessionShow,
    SessionLoad,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParsedSlashCommand {
    Command(SlashCommand, Option<String>),
    NotCommand,
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompactFeedback {
    Compacted(String),
    Skipped(String),
}

impl CompactFeedback {
    pub(crate) fn message(&self) -> &str {
        match self {
            Self::Compacted(message) | Self::Skipped(message) => message,
        }
    }
}

impl CliCallback {
    pub fn new() -> Self {
        Self {
            spinner: None,
            thinking_active: false,
            needs_role_prefix: true,
        }
    }

    /// Start the thinking spinner. Call before `run_turn`.
    pub fn start_turn(&mut self) {
        self.spinner = Some(Spinner::start("thinking"));
    }

    /// Ensure the spinner is stopped. Call after `run_turn` returns.
    pub fn stop_turn(&mut self) {
        if let Some(mut s) = self.spinner.take() {
            s.stop();
        }
        self.end_thinking_block();
        // Print a turn separator and reset role prefix for the next turn
        if !self.needs_role_prefix {
            // Only print separator if we actually emitted tokens this turn
            eprintln!("\x1b[90m─── ─── ───\x1b[0m");
        }
        self.needs_role_prefix = true;
    }

    fn stop_spinner(&mut self) {
        if let Some(mut s) = self.spinner.take() {
            s.stop();
        }
    }

    /// Close an open thinking block: reset ANSI attributes and emit a blank
    /// line so subsequent output starts on a clean line. Called on every
    /// callback transition (token, tool, error, end-of-turn) so reasoning
    /// never bleeds into other output lanes.
    fn end_thinking_block(&mut self) {
        if self.thinking_active {
            use std::io::Write;
            print!("\x1b[0m\n\n");
            let _ = std::io::stdout().flush();
            self.thinking_active = false;
        }
    }
}

fn permission_prompt_allowed(interactive: bool, bytes_read: usize, input: &str) -> bool {
    if !interactive || bytes_read == 0 {
        return false;
    }

    let trimmed = input.trim().to_lowercase();
    trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
}

impl zipcode_runtime::StreamCallback for CliCallback {
    fn on_token(&mut self, text: &str) {
        self.stop_spinner();
        self.end_thinking_block();
        if self.needs_role_prefix {
            print!("\x1b[1mZip:\x1b[0m ");
            self.needs_role_prefix = false;
        }
        print!("{text}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    fn on_thinking(&mut self, text: &str) {
        // Silent by default: keep the spinner alive and forward the chunk
        // size to its live counter so the user still gets visible progress
        // feedback ("thinking (N chars)") without drowning the terminal in
        // reasoning tokens. Only in verbose mode do we stream the raw
        // reasoning as dim+italic inline — useful for prompt-engineering
        // debugging but noisy for day-to-day use. This matches Claude Code
        // default UX where "Thinking…" is a header, not a content lane.
        if let Some(spinner) = &self.spinner {
            spinner.add_thinking_bytes(text.len());
        }

        if !crate::render::is_verbose() {
            return;
        }

        self.stop_spinner();
        use std::io::Write;
        if !self.thinking_active {
            print!("\x1b[2m\x1b[3m");
            self.thinking_active = true;
        }
        print!("{text}");
        let _ = std::io::stdout().flush();
    }

    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value) {
        self.stop_spinner();
        self.end_thinking_block();
        println!(); // newline after streamed tokens
        print_tool_start(name, args);
    }

    fn on_tool_result(&mut self, name: &str, result: &str) {
        self.end_thinking_block();
        print_tool_result(name, result);
        // Restart spinner — model will think again after processing results
        self.spinner = Some(Spinner::start("thinking"));
    }

    fn on_permission_prompt(&mut self, message: &str) -> bool {
        self.stop_spinner();
        self.end_thinking_block();
        use std::io::{self, IsTerminal, Write};
        print!("\x1b[33m[permission]\x1b[0m {message} [Y/n] ");
        io::stdout().flush().ok();
        let mut input = String::new();
        let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
        let bytes_read = io::stdin().read_line(&mut input).unwrap_or(0);
        permission_prompt_allowed(interactive, bytes_read, &input)
    }

    fn on_error(&mut self, error: &str) {
        self.stop_spinner();
        self.end_thinking_block();
        eprintln!("\x1b[31merror:\x1b[0m {error}");
    }
}

/// Build a ToolRegistry with all 10 tools registered.
pub fn build_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // Register the 9 tools that don't need special construction
    registry.register(Box::new(BashTool));
    registry.register(Box::new(ReadFileTool));
    registry.register(Box::new(WriteFileTool));
    registry.register(Box::new(EditFileTool));
    registry.register(Box::new(GlobSearchTool));
    registry.register(Box::new(GrepSearchTool));
    registry.register(Box::new(TodoWriteTool));
    registry.register(Box::new(ReplTool));
    registry.register(Box::new(AgentTool));

    // ToolSearchTool needs the other specs — build from the registry so far
    let tool_search = ToolSearchTool::from_registry(&registry);
    registry.register(Box::new(tool_search));

    registry
}

fn is_gguf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

pub(crate) fn is_probably_gemma4_model(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("gemma-4") || lower.contains("gemma4")
        })
        .unwrap_or(false)
}

pub(crate) fn list_models(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut models: Vec<_> = std::fs::read_dir(model_dir)
        .with_context(|| format!("Failed to read model directory: {}", model_dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|path| path.is_file() && is_gguf_path(path))
        .collect();

    models.sort_by_cached_key(|path| {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let priority = if file_name.contains("gemma-4") || file_name.contains("gemma4") {
            0
        } else {
            1
        };
        (priority, file_name)
    });
    Ok(models)
}

/// Find a single .gguf file in the given directory.
pub fn find_model(model_dir: &Path) -> Result<PathBuf> {
    list_models(model_dir)?.into_iter().next().ok_or_else(|| {
        anyhow::anyhow!(
            "No .gguf model file found in directory: {}",
            model_dir.display()
        )
    })
}

pub(crate) fn resolve_model_path(
    explicit_model_path: Option<&Path>,
    config: &ZipcodeConfig,
    cwd: &Path,
    project_root: &Path,
) -> Result<PathBuf> {
    if let Some(path) = explicit_model_path {
        let expanded = expand_user_path(path);
        let resolved = if expanded.is_absolute() {
            expanded
        } else {
            cwd.join(expanded)
        };

        return if resolved.is_dir() {
            find_model(&resolved)
        } else if resolved.exists() {
            Ok(resolved)
        } else {
            anyhow::bail!("Model path not found: {}", resolved.display())
        };
    }

    let model_dir = resolve_project_path(&config.model_dir, project_root);

    if let Some(ref model_file) = config.model_file {
        let configured = expand_user_path(Path::new(model_file));
        let resolved = if configured.is_absolute() {
            configured
        } else if has_nonempty_parent(&configured) {
            project_root.join(configured)
        } else {
            model_dir.join(configured)
        };

        return if resolved.is_dir() {
            find_model(&resolved)
        } else if resolved.exists() {
            Ok(resolved)
        } else {
            anyhow::bail!("Configured model path not found: {}", resolved.display())
        };
    }

    find_model(&model_dir)
}

pub(crate) fn has_nonempty_parent(path: &Path) -> bool {
    path.parent()
        .is_some_and(|parent| !parent.as_os_str().is_empty())
}

pub(crate) fn resolve_requested_backend(backend_override: Option<&str>) -> Result<Option<Backend>> {
    backend_override.map(Backend::parse).transpose()
}

pub(crate) fn resolve_effective_backend(
    requested_backend: Option<Backend>,
    model_path: &Path,
    helper_path: Option<&Path>,
) -> Backend {
    match requested_backend {
        Some(backend) => backend,
        None if is_probably_gemma4_model(model_path) && helper_path.is_some() => {
            Backend::LlamaServer
        }
        None => Backend::LlamaCpp,
    }
}

fn is_runnable_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .map(|meta| meta.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }

    #[cfg(not(unix))]
    {
        true
    }
}

fn which_runnable_in_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(binary);
        if is_runnable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

pub(crate) fn discover_helper(config: &ZipcodeConfig) -> HelperDiscovery {
    let mut issue = None;

    if let Some(path) = &config.llama_server_bin {
        if is_runnable_file(path) {
            return HelperDiscovery {
                path: Some(path.clone()),
                issue: None,
                issue_is_blocking: false,
            };
        }

        issue = Some(format!(
            "saved helper path could not be used at {}",
            path.display()
        ));
    }

    for key in ["ZIPCODE_LLAMA_SERVER_BIN", "LLAMA_SERVER_BIN"] {
        if let Ok(value) = std::env::var(key) {
            if value.trim().is_empty() {
                continue;
            }
            let path = PathBuf::from(value);
            if is_runnable_file(&path) {
                return HelperDiscovery {
                    path: Some(path),
                    issue,
                    issue_is_blocking: false,
                };
            }

            if issue.is_none() {
                issue = Some(format!(
                    "{key} points to {} but that file could not be used",
                    path.display()
                ));
            }
        }
    }

    if let Some(home) = dirs::home_dir() {
        let bundled = home.join(".zipcode/bin/llama-server");
        if is_runnable_file(&bundled) {
            return HelperDiscovery {
                path: Some(bundled),
                issue,
                issue_is_blocking: false,
            };
        }
    }

    if let Some(path) = which_runnable_in_path("llama-server") {
        return HelperDiscovery {
            path: Some(path),
            issue,
            issue_is_blocking: false,
        };
    }

    HelperDiscovery {
        path: None,
        issue_is_blocking: issue.is_some(),
        issue,
    }
}

fn helper_devices_support_acceleration(output: &str) -> bool {
    let lowered = output.to_ascii_lowercase();
    ["cuda", "metal", "vulkan", "rocm", "gpu"]
        .iter()
        .any(|needle| lowered.contains(needle))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HelperDeviceProbe {
    Devices(String),
    Unsupported,
    TimedOut,
}

fn wait_for_exit(child: &mut Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            Ok(None) | Err(_) => return false,
        }
    }
}

fn read_child_pids(pid: u32) -> Vec<u32> {
    let path = format!("/proc/{pid}/task/{pid}/children");
    let Ok(children) = std::fs::read_to_string(path) else {
        return Vec::new();
    };

    children
        .split_whitespace()
        .filter_map(|value| value.parse::<u32>().ok())
        .collect()
}

fn process_tree_pids(root_pid: u32) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut stack = vec![root_pid];
    let mut ordered = Vec::new();

    while let Some(pid) = stack.pop() {
        if !seen.insert(pid) {
            continue;
        }
        ordered.push(pid);
        stack.extend(read_child_pids(pid));
    }

    ordered.reverse();
    ordered
}

fn send_signal_to_process_tree(root_pid: u32, signal: &str) {
    let pids = process_tree_pids(root_pid);
    if pids.is_empty() {
        return;
    }

    let kill_bin = ["/bin/kill", "/usr/bin/kill"]
        .into_iter()
        .map(Path::new)
        .find(|path| path.is_file());

    if let Some(kill_bin) = kill_bin {
        let mut args = Vec::with_capacity(pids.len() + 1);
        args.push(signal.to_string());
        args.extend(pids.iter().map(u32::to_string));
        let _ = Command::new(kill_bin).args(&args).status();
    }
}

fn terminate_child_tree(child: &mut Child) {
    let root_pid = child.id();

    for signal in ["-TERM", "-KILL"] {
        send_signal_to_process_tree(root_pid, signal);

        if wait_for_exit(child, Duration::from_millis(250)) {
            return;
        }
    }

    let _ = child.kill();
    let _ = child.wait();
}

fn probe_helper_devices(helper_path: &Path) -> HelperDeviceProbe {
    const HELPER_DEVICE_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

    let mut child = match Command::new(helper_path)
        .arg("--list-devices")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return HelperDeviceProbe::Unsupported,
    };

    if !wait_for_exit(&mut child, HELPER_DEVICE_PROBE_TIMEOUT) {
        terminate_child_tree(&mut child);
        return HelperDeviceProbe::TimedOut;
    }

    let Ok(output) = child.wait_with_output() else {
        return HelperDeviceProbe::Unsupported;
    };
    if !output.status.success() {
        return HelperDeviceProbe::Unsupported;
    }

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let lowered = combined.to_ascii_lowercase();
    if !(lowered.contains("device")
        || lowered.contains("cuda")
        || lowered.contains("metal")
        || lowered.contains("vulkan")
        || lowered.contains("rocm")
        || lowered.contains("cpu"))
    {
        return HelperDeviceProbe::Unsupported;
    }

    HelperDeviceProbe::Devices(combined)
}

pub(crate) fn server_options_from_config(config: &ZipcodeConfig) -> ServerOptions {
    ServerOptions {
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
            .unwrap_or(DEFAULT_CONTEXT_SIZE),
    }
}

pub(crate) fn validate_backend_configuration(
    requested_backend: Option<Backend>,
    effective_backend: Backend,
    model_path: &Path,
    helper_path: Option<&Path>,
    server_options: &ServerOptions,
) -> Option<String> {
    if is_probably_gemma4_model(model_path) {
        match requested_backend.unwrap_or(effective_backend) {
            Backend::Candle => {
                return Some(
                    "Gemma 4 is not ready with the candle backend yet. Use `--backend llama-server` or remove the backend override so zipcode can choose the helper automatically.".to_string(),
                )
            }
            Backend::LlamaCpp if requested_backend.is_some() => {
                return Some(
                    "Gemma 4 is not supported by the native llama-cpp backend in this build. Use `--backend llama-server` or remove the backend override so zipcode can choose the helper automatically.".to_string(),
                )
            }
            _ => {}
        }
    }

    if matches!(effective_backend, Backend::LlamaServer) {
        let gpu_layers = server_options.gpu_layers.unwrap_or(0);
        if gpu_layers > 0 {
            if let Some(path) = helper_path {
                match probe_helper_devices(path) {
                    HelperDeviceProbe::Devices(devices) => {
                        if !helper_devices_support_acceleration(&devices) {
                            return Some(format!(
                                "gpu_layers={gpu_layers} requests GPU offload, but the compatibility helper at {} reported no GPU devices.",
                                path.display()
                            ));
                        }
                    }
                    HelperDeviceProbe::TimedOut => {
                        return Some(format!(
                            "gpu_layers={gpu_layers} requests GPU offload, but the compatibility helper probe at {} timed out before reporting available devices.",
                            path.display()
                        ));
                    }
                    HelperDeviceProbe::Unsupported => {}
                }
            }
        }
    }

    None
}

fn build_startup_notices(
    requested_backend: Option<Backend>,
    effective_backend: Backend,
    model_path: &Path,
    server_options: &ServerOptions,
) -> Vec<String> {
    let mut notices = Vec::new();

    if requested_backend.is_none()
        && matches!(effective_backend, Backend::LlamaServer)
        && is_probably_gemma4_model(model_path)
    {
        notices.push(
            "Using llama-server directly for Gemma 4 to avoid slow native fallback.".to_string(),
        );
    }

    if matches!(effective_backend, Backend::LlamaServer) {
        if server_options.gpu_layers.unwrap_or(0) <= 0 {
            notices.push(
                "Slow setting: GPU layer offload is not configured; set ZIPCODE_GPU_LAYERS or gpu_layers for faster replies."
                    .to_string(),
            );
        }
        if !server_options.flash_attention {
            notices.push(
                "Slow setting: flash attention is off; enable ZIPCODE_FLASH_ATTENTION=1 or flash_attention=true if your helper supports it."
                    .to_string(),
            );
        }
    }

    notices
}

/// Build and return a ConversationLoop plus launch metadata.
pub fn prepare_loop(
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_override: Option<&str>,
    session_id: Option<&str>,
) -> Result<LoopLaunch> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let project_root = find_project_root(&cwd);

    // Load config
    let config = match ZipcodeConfig::load(&cwd) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\x1b[33mWarning: Failed to load config: {e}\x1b[0m");
            ZipcodeConfig::default()
        }
    };

    let mut startup_notices = Vec::new();
    let session = match session_id {
        Some(id) => {
            let session =
                Session::load(id).with_context(|| format!("Failed to load session `{id}`"))?;
            startup_notices.push(format!(
                "Resumed session {} ({} messages).",
                session.id,
                session.messages.len()
            ));
            session
        }
        None => Session::new(),
    };

    // Resolve model file
    let model_file = resolve_model_path(model_path, &config, &cwd, &project_root)?;

    // Tokenizer lives next to the model or in the same dir
    let tokenizer_path = model_file
        .parent()
        .unwrap_or(Path::new("."))
        .join("tokenizer.json");

    let mut gen_config = GenerationConfig::default();
    if let Some(t) = config.generation.temperature {
        gen_config.temperature = t;
    }
    if let Some(p) = config.generation.top_p {
        gen_config.top_p = p;
    }
    if let Some(m) = config.generation.max_tokens {
        gen_config.max_tokens = m;
    }

    let helper = discover_helper(&config);
    let helper_issue = helper.issue.clone();
    let helper_path = helper.path;
    if let Some(path) = helper_path.as_deref() {
        std::env::set_var("ZIPCODE_LLAMA_SERVER_BIN", path);
    }
    if helper_path.is_some() {
        if let Some(issue) = helper_issue {
            startup_notices.push(format!("Compatibility helper note: {issue}"));
        }
    }

    let server_options = server_options_from_config(&config);

    let requested_backend = resolve_requested_backend(backend_override)?;
    let effective_backend =
        resolve_effective_backend(requested_backend, &model_file, helper_path.as_deref());
    startup_notices.extend(build_startup_notices(
        requested_backend,
        effective_backend,
        &model_file,
        &server_options,
    ));

    if let Some(issue) = validate_backend_configuration(
        requested_backend,
        effective_backend,
        &model_file,
        helper_path.as_deref(),
        &server_options,
    ) {
        anyhow::bail!("{issue}");
    }

    let engine = match create_engine(
        effective_backend,
        &model_file,
        &tokenizer_path,
        gen_config.clone(),
        server_options.clone(),
    ) {
        Ok(engine) => engine,
        Err(error) => return Err(error).context("Failed to load inference engine"),
    };

    // Build tools and system prompt
    let registry = build_registry();
    let resolved_permission_mode = permission_mode.unwrap_or(&config.permission_mode);
    let effective_permission = parse_permission_mode(resolved_permission_mode)?;
    let permission_str = resolved_permission_mode.to_string();
    let (system_prompt, tool_specs) = build_system_prompt(&cwd, &registry, &permission_str);

    let permission = PermissionPolicy::new(effective_permission);

    Ok(LoopLaunch {
        conv: ConversationLoop {
            engine,
            tools: registry,
            session,
            permission,
            system_prompt,
            tool_specs,
            cwd,
            last_sent_idx: 0,
        },
        effective_backend,
        startup_notices,
    })
}

/// Run a single turn (one-shot mode) then exit.
pub fn run_oneshot(
    text: &str,
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_override: Option<&str>,
    session_id: Option<&str>,
) -> Result<()> {
    let launch = prepare_loop(model_path, permission_mode, backend_override, session_id)?;
    for notice in &launch.startup_notices {
        eprintln!("\x1b[33m[notice]\x1b[0m {notice}");
    }
    let mut conv = launch.conv;
    let mut cb = CliCallback::new();
    cb.start_turn();
    conv.run_turn(text, &mut cb)?;
    cb.stop_turn();
    println!(); // final newline
    Ok(())
}

pub(crate) fn parse_slash_command(input: &str) -> ParsedSlashCommand {
    let trimmed = input.trim();
    let mut parts = trimmed.split_whitespace();
    let Some(command) = parts.next() else {
        return ParsedSlashCommand::NotCommand;
    };
    let args = parts.collect::<Vec<_>>();

    let reject_extra_args =
        || ParsedSlashCommand::Error(format!("Unknown slash command usage: {trimmed}"));

    match command {
        "/help" => {
            if args.is_empty() {
                ParsedSlashCommand::Command(SlashCommand::Help, None)
            } else {
                reject_extra_args()
            }
        }
        "/quit" | "/exit" => {
            if args.is_empty() {
                ParsedSlashCommand::Command(SlashCommand::Quit, None)
            } else {
                reject_extra_args()
            }
        }
        "/clear" => {
            if args.is_empty() {
                ParsedSlashCommand::Command(SlashCommand::Clear, None)
            } else {
                reject_extra_args()
            }
        }
        "/status" => {
            if args.is_empty() {
                ParsedSlashCommand::Command(SlashCommand::Status, None)
            } else {
                reject_extra_args()
            }
        }
        "/compact" => {
            if args.is_empty() {
                ParsedSlashCommand::Command(SlashCommand::Compact, None)
            } else {
                reject_extra_args()
            }
        }
        "/session" => match args.as_slice() {
            [] => ParsedSlashCommand::Command(SlashCommand::SessionShow, None),
            [session_id] => ParsedSlashCommand::Command(
                SlashCommand::SessionLoad,
                Some(session_id.trim().to_string()),
            ),
            _ => ParsedSlashCommand::Error("Usage: /session [SESSION_ID]".to_string()),
        },
        _ => ParsedSlashCommand::NotCommand,
    }
}

/// Run an interactive REPL loop.
pub fn run_interactive(
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_override: Option<&str>,
    session_id: Option<&str>,
) -> Result<()> {
    let launch = prepare_loop(model_path, permission_mode, backend_override, session_id)?;
    println!(
        "zipcode v{} — type /help for commands, Ctrl+D to exit",
        env!("CARGO_PKG_VERSION")
    );
    for notice in &launch.startup_notices {
        println!("\x1b[33m[notice]\x1b[0m {notice}");
    }
    let mut conv = launch.conv;
    let mut cb = CliCallback::new();

    let mut rl = DefaultEditor::new().context("Failed to initialize line editor")?;

    loop {
        let prompt = "\x1b[32m> \x1b[0m";
        match rl.readline(prompt) {
            Ok(line) => {
                let input = line.trim().to_string();
                if input.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(&input);

                // Handle slash commands
                match parse_slash_command(&input) {
                    ParsedSlashCommand::Command(command, argument) => {
                        match command {
                            SlashCommand::Help => print_help(),
                            SlashCommand::Quit => {
                                println!("Goodbye.");
                                break;
                            }
                            SlashCommand::Clear => println!("{}", clear_session(&mut conv)?),
                            SlashCommand::Status => print_status(&conv),
                            SlashCommand::Compact => {
                                println!("{}", compact_session(&mut conv)?.message())
                            }
                            SlashCommand::SessionShow => print_session_status(&conv),
                            SlashCommand::SessionLoad => match load_session_into_loop(
                                &mut conv,
                                argument.as_deref().unwrap_or_else(|| {
                                    eprintln!(
                                        "\\x1b[33m/session load requires a session id\\x1b[0m"
                                    );
                                    ""
                                }),
                            ) {
                                Ok(message) => println!("{message}"),
                                Err(error) => println!("{error:#}"),
                            },
                        }
                        continue;
                    }
                    ParsedSlashCommand::Error(message) => {
                        println!("{message}");
                        continue;
                    }
                    ParsedSlashCommand::NotCommand => {}
                }

                // Regular user input — run a turn
                cb.start_turn();
                if let Err(e) = conv.run_turn(&input, &mut cb) {
                    eprintln!("\x1b[31merror:\x1b[0m {e}");
                }
                cb.stop_turn();
                println!(); // newline after assistant response
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl+C — continue
                println!("(Ctrl+C — press Ctrl+D to exit)");
            }
            Err(ReadlineError::Eof) => {
                // Ctrl+D — exit
                println!("Goodbye.");
                break;
            }
            Err(e) => {
                eprintln!("\x1b[31mREPL error:\x1b[0m {e}");
                break;
            }
        }
    }

    Ok(())
}

fn print_help() {
    println!("{}", help_text());
}

fn print_status(conv: &ConversationLoop) {
    println!("Session ID:  {}", conv.session.id);
    println!("Messages:    {}", conv.session.messages.len());
    println!("Tools:       {}", conv.tools.names().len());
    println!("Working dir: {}", conv.cwd.display());
}

pub(crate) fn session_status_lines(conv: &ConversationLoop) -> Vec<String> {
    vec![
        format!("Session ID:   {}", conv.session.id),
        format!("Session path: {}", conv.session.path().display()),
        format!("Messages:     {}", conv.session.messages.len()),
        format!("Created at:   {}", conv.session.created_at),
        format!("Updated at:   {}", conv.session.updated_at),
    ]
}

fn print_session_status(conv: &ConversationLoop) {
    for line in session_status_lines(conv) {
        println!("{line}");
    }
}

pub(crate) fn clear_session(conv: &mut ConversationLoop) -> Result<String> {
    let session = Session::new();
    session.save()?;
    let session_id = session.id.clone();
    conv.session = session;
    // Reset so a context-owning provider (e.g. llama-server) sees the
    // fresh session from turn 1 instead of slicing past the old cursor
    // into an empty Vec (which would panic on `messages[idx..]`).
    conv.last_sent_idx = 0;
    Ok(format!(
        "Conversation cleared. New session started: {session_id}"
    ))
}

pub(crate) fn compact_session(conv: &mut ConversationLoop) -> Result<CompactFeedback> {
    let result = conv.session.compact(CompactPolicy::default());
    if result.changed {
        // Compaction rewrites `messages` in place and the vec shrinks.
        // Clamp the cursor so the next slice never indexes past the end.
        // Resetting to 0 also forces the provider to re-prime with the
        // compacted history on the next turn, which is the correct
        // behaviour because the provider-side KV may reference pruned
        // messages that no longer exist locally.
        conv.last_sent_idx = 0;
        conv.session.save()?;
        Ok(CompactFeedback::Compacted(format_compact_result(
            &conv.session,
            result,
        )))
    } else {
        Ok(CompactFeedback::Skipped(format!(
            "Compaction skipped for session {}: not enough history to compact.",
            conv.session.id
        )))
    }
}

pub(crate) fn load_session_into_loop(
    conv: &mut ConversationLoop,
    session_id: &str,
) -> Result<String> {
    let session_id = session_id.trim();
    let loaded = Session::load(session_id)
        .with_context(|| format!("Failed to load session `{session_id}`"))?;
    let loaded_id = loaded.id.clone();
    let message_count = loaded.messages.len();
    let path = loaded.path();
    conv.session = loaded;
    // Restored session is new state for the current provider instance,
    // so the provider has no cached prefix matching these messages.
    // Send the full restored history on the next turn.
    conv.last_sent_idx = 0;
    Ok(format!(
        "Loaded session {loaded_id} ({message_count} messages) from {}",
        path.display()
    ))
}

fn format_compact_result(session: &Session, result: CompactResult) -> String {
    format!(
        "Compacted session {}: {} -> {} messages (pruned {}, retained {}).",
        session.id,
        result.before_messages,
        result.after_messages,
        result.pruned_messages,
        result.retained_messages
    )
}

pub(crate) fn help_text() -> &'static str {
    "Available commands:\n  /help              — show this help\n  /status            — show session info and model\n  /session           — show active session metadata\n  /session <id>      — load an existing session by id\n  /compact           — compact older session history\n  /clear             — clear conversation history\n  /quit, /exit       — exit zipcode\n\n  Ctrl+C             — cancel current input (continues)\n  Ctrl+D             — exit zipcode"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use zipcode_runtime::config::GenerationOverrides;

    fn temp_dir(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("zipcode-{prefix}-{unique}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn list_models_sorts_gemma4_first() {
        let dir = temp_dir("find-model");
        std::fs::write(dir.join("z-last.gguf"), "").unwrap();
        std::fs::write(dir.join("gemma-4-e2b-it-q8_0.gguf"), "").unwrap();
        std::fs::write(dir.join("a-first.gguf"), "").unwrap();

        let models = list_models(&dir).unwrap();
        assert_eq!(models[0], dir.join("gemma-4-e2b-it-q8_0.gguf"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn resolve_model_path_accepts_directory_argument() {
        let dir = temp_dir("dir-arg");
        let model_dir = dir.join("models");
        std::fs::create_dir_all(&model_dir).unwrap();
        let model_file = model_dir.join("gemma-4-e2b-it-q8_0.gguf");
        std::fs::write(&model_file, "").unwrap();

        let config = ZipcodeConfig::default();
        let resolved = resolve_model_path(Some(&model_dir), &config, &dir, &dir).unwrap();
        assert_eq!(resolved, model_file);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn resolve_model_path_uses_project_root_for_relative_config_paths() {
        let dir = temp_dir("project-root");
        let nested = dir.join("src/bin");
        let model_dir = dir.join("models");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir_all(&model_dir).unwrap();
        let model_file = model_dir.join("gemma-4-e2b-it-q8_0.gguf");
        std::fs::write(&model_file, "").unwrap();

        let config = ZipcodeConfig {
            model_dir: PathBuf::from("models"),
            model_file: None,
            llama_server_bin: None,
            permission_mode: "workspace-write".to_string(),
            generation: GenerationOverrides::default(),
            gpu_layers: None,
            flash_attention: false,
        };

        let resolved = resolve_model_path(None, &config, &nested, &dir).unwrap();
        assert_eq!(resolved, model_file);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn resolve_model_path_expands_tilde_model_file() {
        let dir = temp_dir("tilde-model-file");
        let home = dirs::home_dir().unwrap();
        let model_dir = home.join(".zipcode/models");
        std::fs::create_dir_all(&model_dir).unwrap();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let file_name = format!("tilde-model-{unique}.gguf");
        let model_file = model_dir.join(&file_name);
        std::fs::write(&model_file, "").unwrap();

        let config = ZipcodeConfig {
            model_dir: PathBuf::from("unused"),
            model_file: Some(format!("~/.zipcode/models/{file_name}")),
            llama_server_bin: None,
            permission_mode: "workspace-write".to_string(),
            generation: GenerationOverrides::default(),
            gpu_layers: None,
            flash_attention: false,
        };

        let resolved = resolve_model_path(None, &config, &dir, &dir).unwrap();
        assert_eq!(resolved, model_file);
        let _ = std::fs::remove_file(model_file);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn parse_slash_command_accepts_known_commands_only() {
        assert_eq!(
            parse_slash_command("/help"),
            ParsedSlashCommand::Command(SlashCommand::Help, None)
        );
        assert_eq!(
            parse_slash_command("/status"),
            ParsedSlashCommand::Command(SlashCommand::Status, None)
        );
        assert_eq!(
            parse_slash_command("/clear"),
            ParsedSlashCommand::Command(SlashCommand::Clear, None)
        );
        assert_eq!(
            parse_slash_command("/quit"),
            ParsedSlashCommand::Command(SlashCommand::Quit, None)
        );
        assert_eq!(
            parse_slash_command("/exit"),
            ParsedSlashCommand::Command(SlashCommand::Quit, None)
        );
        assert_eq!(
            parse_slash_command("/compact"),
            ParsedSlashCommand::Command(SlashCommand::Compact, None)
        );
        assert_eq!(
            parse_slash_command("/session"),
            ParsedSlashCommand::Command(SlashCommand::SessionShow, None)
        );
        assert_eq!(
            parse_slash_command("/session abc"),
            ParsedSlashCommand::Command(SlashCommand::SessionLoad, Some("abc".to_string()))
        );
    }

    #[test]
    fn parse_slash_command_treats_paths_and_unknown_slashes_as_regular_input() {
        assert_eq!(
            parse_slash_command("/home/devswha/workspace/test_zipcode"),
            ParsedSlashCommand::NotCommand
        );
        assert_eq!(
            parse_slash_command("/unknown"),
            ParsedSlashCommand::NotCommand
        );
        assert_eq!(
            parse_slash_command("\"/home/devswha/workspace/test_zipcode\" 레포 분석해봐"),
            ParsedSlashCommand::NotCommand
        );
    }

    #[test]
    fn parse_slash_command_rejects_malformed_session_usage() {
        assert_eq!(
            parse_slash_command("/session abc def"),
            ParsedSlashCommand::Error("Usage: /session [SESSION_ID]".to_string())
        );
    }

    #[test]
    fn parse_slash_command_trims_session_id_whitespace() {
        assert_eq!(
            parse_slash_command("/session   abc-123   "),
            ParsedSlashCommand::Command(SlashCommand::SessionLoad, Some("abc-123".to_string()))
        );
    }

    #[test]
    fn parse_slash_command_rejects_extra_arguments_for_known_commands() {
        for input in [
            "/help extra",
            "/status extra",
            "/clear extra",
            "/compact extra",
            "/quit now",
            "/exit now",
        ] {
            assert_eq!(
                parse_slash_command(input),
                ParsedSlashCommand::Error(format!("Unknown slash command usage: {input}"))
            );
        }
    }

    #[test]
    fn parse_slash_command_keeps_session_prefix_variants_as_regular_input() {
        assert_eq!(
            parse_slash_command("/sessionfoo"),
            ParsedSlashCommand::NotCommand
        );
        assert_eq!(
            parse_slash_command("/session/abc"),
            ParsedSlashCommand::NotCommand
        );
        assert_eq!(
            parse_slash_command("/session-status"),
            ParsedSlashCommand::NotCommand
        );
    }

    #[test]
    fn help_text_documents_session_and_compact_commands() {
        let help = help_text();
        assert!(help.contains("/session"));
        assert!(help.contains("/compact"));
        assert!(help.contains("/quit, /exit"));
    }

    #[test]
    fn auto_backend_prefers_llama_server_for_gemma4_when_helper_exists() {
        let backend = resolve_effective_backend(
            None,
            Path::new("/tmp/gemma-4-e2b-it-q8_0.gguf"),
            Some(Path::new("/tmp/llama-server")),
        );
        assert!(matches!(backend, Backend::LlamaServer));
    }

    #[test]
    fn auto_backend_stays_on_llama_cpp_without_helper() {
        let backend = resolve_effective_backend(None, Path::new("/tmp/codeqwen.gguf"), None);
        assert!(matches!(backend, Backend::LlamaCpp));
    }

    #[test]
    fn explicit_backend_is_respected_for_gemma4() {
        let backend = resolve_effective_backend(
            Some(Backend::LlamaCpp),
            Path::new("/tmp/gemma-4-e2b-it-q8_0.gguf"),
            Some(Path::new("/tmp/llama-server")),
        );
        assert!(matches!(backend, Backend::LlamaCpp));
    }

    #[test]
    fn startup_notices_flag_slow_llama_server_settings() {
        let notices = build_startup_notices(
            None,
            Backend::LlamaServer,
            Path::new("/tmp/gemma-4-e2b-it-q8_0.gguf"),
            &ServerOptions::default(),
        );
        assert!(notices
            .iter()
            .any(|line| line.contains("avoid slow native fallback")));
        assert!(notices
            .iter()
            .any(|line| line.contains("GPU layer offload")));
        assert!(notices.iter().any(|line| line.contains("flash attention")));
    }

    #[test]
    fn permission_prompt_denies_eof_even_if_interactive() {
        assert!(!permission_prompt_allowed(true, 0, ""));
    }

    #[test]
    fn permission_prompt_denies_noninteractive_input_even_if_yes() {
        assert!(!permission_prompt_allowed(false, 4, "yes\n"));
        assert!(!permission_prompt_allowed(false, 1, "\n"));
    }

    #[test]
    fn permission_prompt_accepts_only_interactive_enter_or_yes() {
        assert!(permission_prompt_allowed(true, 1, "\n"));
        assert!(permission_prompt_allowed(true, 2, "y\n"));
        assert!(permission_prompt_allowed(true, 4, "yes\n"));
        assert!(!permission_prompt_allowed(true, 2, "n\n"));
    }

    // ── is_gguf_path ──────────────────────────────────────────────

    #[test]
    fn gguf_path_accepts_gguf_extension() {
        assert!(is_gguf_path(Path::new("model.gguf")));
    }

    #[test]
    fn gguf_path_is_case_insensitive() {
        assert!(is_gguf_path(Path::new("model.GGUF")));
        assert!(is_gguf_path(Path::new("model.Gguf")));
    }

    #[test]
    fn gguf_path_rejects_other_extensions() {
        assert!(!is_gguf_path(Path::new("model.bin")));
        assert!(!is_gguf_path(Path::new("model.safetensors")));
    }

    #[test]
    fn gguf_path_rejects_no_extension() {
        assert!(!is_gguf_path(Path::new("model")));
        assert!(!is_gguf_path(Path::new("models/")));
    }

    #[test]
    fn gguf_path_rejects_double_extension() {
        assert!(!is_gguf_path(Path::new("model.gguf.bin")));
    }

    // ── is_probably_gemma4_model ──────────────────────────────────

    #[test]
    fn gemma4_detects_gemma_4_hyphen() {
        assert!(is_probably_gemma4_model(Path::new(
            "/models/gemma-4-e2b-it-q8_0.gguf"
        )));
    }

    #[test]
    fn gemma4_detects_gemma4_no_hyphen() {
        assert!(is_probably_gemma4_model(Path::new(
            "/models/gemma4-it-q8_0.gguf"
        )));
    }

    #[test]
    fn gemma4_is_case_insensitive() {
        assert!(is_probably_gemma4_model(Path::new(
            "/models/Gemma-4-it.gguf"
        )));
        assert!(is_probably_gemma4_model(Path::new("/models/GEMMA4.gguf")));
    }

    #[test]
    fn gemma4_rejects_non_gemma() {
        assert!(!is_probably_gemma4_model(Path::new(
            "/models/codeqwen-7b.gguf"
        )));
        assert!(!is_probably_gemma4_model(Path::new("/models/llama-3.gguf")));
    }

    #[test]
    fn gemma4_rejects_no_filename() {
        assert!(!is_probably_gemma4_model(Path::new("/models/")));
        assert!(!is_probably_gemma4_model(Path::new("/")));
    }

    // ── has_nonempty_parent ───────────────────────────────────────

    #[test]
    fn nonempty_parent_with_directory() {
        assert!(has_nonempty_parent(Path::new("dir/file")));
        assert!(has_nonempty_parent(Path::new("a/b/c")));
    }

    #[test]
    fn nonempty_parent_without_directory() {
        assert!(!has_nonempty_parent(Path::new("file.txt")));
    }

    #[test]
    fn nonempty_parent_with_dot_slash() {
        // "./file" has parent "." which is not empty
        assert!(has_nonempty_parent(Path::new("./file")));
    }

    #[test]
    fn nonempty_parent_with_empty_parent() {
        // A bare filename at root has no parent
        assert!(!has_nonempty_parent(Path::new("readme.md")));
    }

    // ── helper_devices_support_acceleration ────────────────────────

    #[test]
    fn acceleration_detects_cuda() {
        assert!(helper_devices_support_acceleration("Found CUDA device 0"));
    }

    #[test]
    fn acceleration_detects_metal_case_insensitive() {
        assert!(helper_devices_support_acceleration("METAL GPU available"));
    }

    #[test]
    fn acceleration_detects_vulkan() {
        assert!(helper_devices_support_acceleration("vulkan renderer"));
    }

    #[test]
    fn acceleration_detects_rocm() {
        assert!(helper_devices_support_acceleration("ROCm device found"));
    }

    #[test]
    fn acceleration_detects_gpu() {
        assert!(helper_devices_support_acceleration("GPU: NVIDIA A100"));
    }

    #[test]
    fn acceleration_rejects_cpu_only() {
        assert!(!helper_devices_support_acceleration("cpu only mode"));
    }

    #[test]
    fn acceleration_rejects_empty_string() {
        assert!(!helper_devices_support_acceleration(""));
    }

    #[test]
    fn acceleration_rejects_unrelated_text() {
        assert!(!helper_devices_support_acceleration(
            "memory: 16GB, disk: 512GB"
        ));
    }

    // ── format_compact_result ─────────────────────────────────────

    #[test]
    fn compact_result_formats_all_fields() {
        let session = Session::new();
        let result = CompactResult {
            changed: true,
            before_messages: 20,
            after_messages: 10,
            pruned_messages: 10,
            retained_messages: 10,
        };
        let text = format_compact_result(&session, result);
        assert!(text.contains(&session.id));
        assert!(text.contains("20 -> 10 messages"));
        assert!(text.contains("pruned 10"));
        assert!(text.contains("retained 10"));
    }

    #[test]
    fn compact_result_with_zero_pruned() {
        let session = Session::new();
        let result = CompactResult {
            changed: false,
            before_messages: 5,
            after_messages: 5,
            pruned_messages: 0,
            retained_messages: 5,
        };
        let text = format_compact_result(&session, result);
        assert!(text.contains(&session.id));
        assert!(text.contains("5 -> 5 messages"));
        assert!(text.contains("pruned 0"));
    }

    // ── resolve_requested_backend ─────────────────────────────────

    #[test]
    fn requested_backend_none_returns_none() {
        assert!(resolve_requested_backend(None).unwrap().is_none());
    }

    #[test]
    fn requested_backend_parses_llama_server() {
        let backend = resolve_requested_backend(Some("llama-server")).unwrap();
        assert!(matches!(backend, Some(Backend::LlamaServer)));
    }

    #[test]
    fn requested_backend_parses_llama_cpp() {
        let backend = resolve_requested_backend(Some("llama-cpp")).unwrap();
        assert!(matches!(backend, Some(Backend::LlamaCpp)));
    }

    #[test]
    fn requested_backend_parses_candle() {
        let backend = resolve_requested_backend(Some("candle")).unwrap();
        assert!(matches!(backend, Some(Backend::Candle)));
    }

    #[test]
    fn requested_backend_rejects_invalid() {
        let err = resolve_requested_backend(Some("invalid")).unwrap_err();
        assert!(err.to_string().contains("unsupported backend"));
    }

    // ── server_options_from_config ────────────────────────────────

    #[test]
    fn server_options_defaults_match_config() {
        let config = ZipcodeConfig::default();
        let opts = server_options_from_config(&config);
        // Default config has gpu_layers: None, flash_attention: false
        assert_eq!(opts.gpu_layers, config.gpu_layers);
        assert_eq!(opts.flash_attention, config.flash_attention);
    }

    #[test]
    fn server_options_env_overrides_gpu_layers() {
        let config = ZipcodeConfig {
            gpu_layers: Some(10),
            ..ZipcodeConfig::default()
        };

        // Set env override
        std::env::set_var("ZIPCODE_GPU_LAYERS", "99");
        let opts = server_options_from_config(&config);
        std::env::remove_var("ZIPCODE_GPU_LAYERS");

        assert_eq!(opts.gpu_layers, Some(99), "env var should override config");
    }

    #[test]
    fn server_options_env_overrides_flash_attention() {
        let config = ZipcodeConfig {
            flash_attention: false,
            ..ZipcodeConfig::default()
        };

        std::env::set_var("ZIPCODE_FLASH_ATTENTION", "1");
        let opts = server_options_from_config(&config);
        std::env::remove_var("ZIPCODE_FLASH_ATTENTION");

        assert!(
            opts.flash_attention,
            "env var should override config flash_attention"
        );
    }

    #[test]
    fn server_options_env_overrides_context_size() {
        let config = ZipcodeConfig::default();

        std::env::set_var("ZIPCODE_LLAMA_SERVER_CTX", "16384");
        let opts = server_options_from_config(&config);
        std::env::remove_var("ZIPCODE_LLAMA_SERVER_CTX");

        assert_eq!(opts.context_size, 16384);
    }

    #[test]
    fn server_options_config_values_when_no_env() {
        // Ensure no leftover env vars
        std::env::remove_var("ZIPCODE_GPU_LAYERS");
        std::env::remove_var("ZIPCODE_FLASH_ATTENTION");
        std::env::remove_var("ZIPCODE_LLAMA_SERVER_CTX");

        let config = ZipcodeConfig {
            gpu_layers: Some(42),
            flash_attention: true,
            ..ZipcodeConfig::default()
        };
        let opts = server_options_from_config(&config);

        assert_eq!(opts.gpu_layers, Some(42));
        assert!(opts.flash_attention);
    }
}
