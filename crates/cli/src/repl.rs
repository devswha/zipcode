use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use zipcode_inference::{select_device, GenerationConfig, InferenceEngine};
use zipcode_runtime::prompt::build_system_prompt;
use zipcode_runtime::{
    permission_mode_from_str, ConversationLoop, PermissionPolicy, Session, ZipcodeConfig,
};
use zipcode_tools::{
    agent::AgentTool, bash::BashTool, edit_file::EditFileTool, glob_search::GlobSearchTool,
    grep_search::GrepSearchTool, read_file::ReadFileTool, repl::ReplTool,
    todo_write::TodoWriteTool, tool_search::ToolSearchTool, write_file::WriteFileTool,
    ToolRegistry,
};

use crate::render::{print_tool_result, print_tool_start};

/// CLI callback that renders streaming tokens and tool events.
pub struct CliCallback;

impl zipcode_runtime::StreamCallback for CliCallback {
    fn on_token(&mut self, text: &str) {
        print!("{text}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value) {
        println!(); // newline after streamed tokens
        print_tool_start(name, args);
    }

    fn on_tool_result(&mut self, name: &str, result: &str) {
        print_tool_result(name, result);
    }

    fn on_permission_prompt(&mut self, message: &str) -> bool {
        use std::io::{self, Write};
        print!("\x1b[33m[permission]\x1b[0m {message} [Y/n] ");
        io::stdout().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        let trimmed = input.trim().to_lowercase();
        trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
    }

    fn on_error(&mut self, error: &str) {
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

/// Find a .gguf file in the given directory.
pub fn find_model(model_dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(model_dir).ok()?;
    entries
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
}

/// Build and return a ConversationLoop ready for use.
pub fn create_loop(
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
) -> Result<ConversationLoop> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;

    // Load config
    let config = match ZipcodeConfig::load(&cwd) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\x1b[33mWarning: Failed to load config: {e}\x1b[0m");
            ZipcodeConfig::default()
        }
    };

    // Resolve model file
    let model_file = if let Some(p) = model_path {
        p.to_path_buf()
    } else {
        // Try config model_file, then scan model_dir
        if let Some(ref mf) = config.model_file {
            config.model_dir.join(mf)
        } else {
            find_model(&config.model_dir)
                .context("No .gguf model file found. Use --model to specify one.")?
        }
    };

    // Tokenizer lives next to the model or in the same dir
    let tokenizer_path = model_file
        .parent()
        .unwrap_or(Path::new("."))
        .join("tokenizer.json");

    let device = select_device();

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

    let mut engine = InferenceEngine::load(&model_file, &tokenizer_path, device)
        .context("Failed to load inference engine")?;
    engine.set_config(gen_config);

    // Build tools and system prompt
    let registry = build_registry();
    let resolved_permission_mode = permission_mode.unwrap_or(&config.permission_mode);
    let effective_permission = permission_mode_from_str(resolved_permission_mode);
    let permission_str = resolved_permission_mode.to_string();
    let (system_prompt, tool_specs) = build_system_prompt(&cwd, &registry, &permission_str);

    let session = Session::new();
    let permission = PermissionPolicy::new(effective_permission);

    Ok(ConversationLoop {
        engine,
        tools: registry,
        session,
        permission,
        system_prompt,
        tool_specs,
        cwd,
    })
}

/// Run a single turn (one-shot mode) then exit.
pub fn run_oneshot(
    text: &str,
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
) -> Result<()> {
    let mut conv = create_loop(model_path, permission_mode)?;
    let mut cb = CliCallback;
    conv.run_turn(text, &mut cb)?;
    println!(); // final newline
    Ok(())
}

/// Run an interactive REPL loop.
pub fn run_interactive(model_path: Option<&Path>, permission_mode: Option<&str>) -> Result<()> {
    println!(
        "zipcode v{} — type /help for commands, Ctrl+D to exit",
        env!("CARGO_PKG_VERSION")
    );

    let mut conv = create_loop(model_path, permission_mode)?;
    let mut cb = CliCallback;

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
                if input.starts_with('/') {
                    match input.as_str() {
                        "/help" => print_help(),
                        "/quit" | "/exit" => {
                            println!("Goodbye.");
                            break;
                        }
                        "/clear" => {
                            conv.session = Session::new();
                            println!("Conversation cleared.");
                        }
                        "/status" => print_status(&conv),
                        _ => {
                            println!("Unknown command: {input}. Type /help for available commands.")
                        }
                    }
                    continue;
                }

                // Regular user input — run a turn
                if let Err(e) = conv.run_turn(&input, &mut cb) {
                    eprintln!("\x1b[31merror:\x1b[0m {e}");
                }
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
    println!("Available commands:");
    println!("  /help    — show this help");
    println!("  /status  — show session info and model");
    println!("  /clear   — clear conversation history");
    println!("  /quit    — exit zipcode");
    println!();
    println!("  Ctrl+C   — cancel current input (continues)");
    println!("  Ctrl+D   — exit zipcode");
}

fn print_status(conv: &ConversationLoop) {
    println!("Session ID:  {}", conv.session.id);
    println!("Messages:    {}", conv.session.messages.len());
    println!("Tools:       {}", conv.tools.names().len());
    println!("Working dir: {}", conv.cwd.display());
}
