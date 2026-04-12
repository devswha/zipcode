use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use zipcode_inference::{create_engine, Backend, GenerationConfig, ServerOptions};
use zipcode_runtime::config::{expand_user_path, find_project_root, resolve_project_path};
use zipcode_runtime::prompt::build_system_prompt;
use zipcode_runtime::{
    parse_permission_mode, ConversationLoop, PermissionPolicy, Session, ZipcodeConfig,
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

/// CLI callback that renders streaming tokens and tool events.
///
/// Manages a background spinner during model inference and between
/// tool execution rounds to indicate activity.
pub struct CliCallback {
    spinner: Option<Spinner>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlashCommand {
    Help,
    Quit,
    Clear,
    Status,
}

impl CliCallback {
    pub fn new() -> Self {
        Self { spinner: None }
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
    }

    fn stop_spinner(&mut self) {
        if let Some(mut s) = self.spinner.take() {
            s.stop();
        }
    }
}

impl zipcode_runtime::StreamCallback for CliCallback {
    fn on_token(&mut self, text: &str) {
        self.stop_spinner();
        print!("{text}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value) {
        self.stop_spinner();
        println!(); // newline after streamed tokens
        print_tool_start(name, args);
    }

    fn on_tool_result(&mut self, name: &str, result: &str) {
        print_tool_result(name, result);
        // Restart spinner — model will think again after processing results
        self.spinner = Some(Spinner::start("thinking"));
    }

    fn on_permission_prompt(&mut self, message: &str) -> bool {
        self.stop_spinner();
        use std::io::{self, Write};
        print!("\x1b[33m[permission]\x1b[0m {message} [Y/n] ");
        io::stdout().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        let trimmed = input.trim().to_lowercase();
        trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
    }

    fn on_error(&mut self, error: &str) {
        self.stop_spinner();
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

pub(crate) fn discover_helper_path(config: &ZipcodeConfig) -> Option<PathBuf> {
    if let Some(path) = &config.llama_server_bin {
        if path.is_file() {
            return Some(path.clone());
        }
    }

    for key in ["ZIPCODE_LLAMA_SERVER_BIN", "LLAMA_SERVER_BIN"] {
        if let Ok(value) = std::env::var(key) {
            let path = PathBuf::from(value);
            if path.is_file() {
                return Some(path);
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

    let helper_path = discover_helper_path(&config);
    if let Some(path) = helper_path.as_deref() {
        std::env::set_var("ZIPCODE_LLAMA_SERVER_BIN", path);
    }

    let server_options = ServerOptions {
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

    let requested_backend = resolve_requested_backend(backend_override)?;
    let effective_backend =
        resolve_effective_backend(requested_backend, &model_file, helper_path.as_deref());
    let startup_notices = build_startup_notices(
        requested_backend,
        effective_backend,
        &model_file,
        &server_options,
    );

    if matches!(effective_backend, Backend::LlamaCpp) && is_probably_gemma4_model(&model_file) {
        anyhow::bail!(
            "Gemma 4 is not supported by the native llama-cpp backend in this build. Run with `--backend llama-server` or configure a helper and let zipcode choose it automatically."
        );
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

    let session = Session::new();
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
) -> Result<()> {
    let launch = prepare_loop(model_path, permission_mode, backend_override)?;
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

pub(crate) fn parse_slash_command(input: &str) -> Option<SlashCommand> {
    match input {
        "/help" => Some(SlashCommand::Help),
        "/quit" | "/exit" => Some(SlashCommand::Quit),
        "/clear" => Some(SlashCommand::Clear),
        "/status" => Some(SlashCommand::Status),
        _ => None,
    }
}

/// Run an interactive REPL loop.
pub fn run_interactive(
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_override: Option<&str>,
) -> Result<()> {
    println!(
        "zipcode v{} — type /help for commands, Ctrl+D to exit",
        env!("CARGO_PKG_VERSION")
    );

    let launch = prepare_loop(model_path, permission_mode, backend_override)?;
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
                if let Some(command) = parse_slash_command(&input) {
                    match command {
                        SlashCommand::Help => print_help(),
                        SlashCommand::Quit => {
                            println!("Goodbye.");
                            break;
                        }
                        SlashCommand::Clear => {
                            conv.session = Session::new();
                            println!("Conversation cleared.");
                        }
                        SlashCommand::Status => print_status(&conv),
                    }
                    continue;
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

pub(crate) fn help_text() -> &'static str {
    "Available commands:\n  /help    — show this help\n  /status  — show session info and model\n  /clear   — clear conversation history\n  /quit    — exit zipcode\n\n  Ctrl+C   — cancel current input (continues)\n  Ctrl+D   — exit zipcode"
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
        assert_eq!(parse_slash_command("/help"), Some(SlashCommand::Help));
        assert_eq!(parse_slash_command("/status"), Some(SlashCommand::Status));
        assert_eq!(parse_slash_command("/clear"), Some(SlashCommand::Clear));
        assert_eq!(parse_slash_command("/quit"), Some(SlashCommand::Quit));
        assert_eq!(parse_slash_command("/exit"), Some(SlashCommand::Quit));
    }

    #[test]
    fn parse_slash_command_treats_paths_and_unknown_slashes_as_regular_input() {
        assert_eq!(
            parse_slash_command("/home/devswha/workspace/test_zipcode"),
            None
        );
        assert_eq!(parse_slash_command("/unknown"), None);
        assert_eq!(
            parse_slash_command("\"/home/devswha/workspace/test_zipcode\" 레포 분석해봐"),
            None
        );
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
}
