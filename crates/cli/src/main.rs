use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

mod commands;
mod render;
mod repl;
mod tui;
mod tui_composer;
mod width;

#[derive(Parser)]
#[command(name = "zipcode", version, about = "Local AI coding assistant")]
struct Cli {
    /// Path to the model directory or .gguf file
    #[arg(long, value_name = "PATH", global = true)]
    model: Option<PathBuf>,

    /// Permission mode: read-only, workspace-write, full-access
    #[arg(long, value_name = "MODE", global = true)]
    permission_mode: Option<String>,

    /// Inference backend override: llama-cpp, llama-server, or candle (auto-selects when omitted)
    #[arg(long, value_name = "BACKEND", global = true)]
    backend: Option<String>,

    /// UI mode for interactive sessions
    #[arg(long, value_enum, default_value_t = UiMode::Fullscreen, global = true)]
    ui: UiMode,

    /// Resume a saved session by id
    #[arg(long, value_name = "SESSION_ID", global = true)]
    session: Option<String>,

    /// Stream the model's reasoning channel inline as dimmed text.
    /// By default the reasoning stays behind a "Thinking… (N chars)"
    /// progress indicator — matches Claude Code's default UX. Enable
    /// this for prompt-engineering debugging or when you want to see
    /// how a local model is thinking through a problem. Can also be
    /// set via `ZIPCODE_VERBOSE=1`.
    #[arg(long, short = 'v', global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum UiMode {
    Plain,
    Fullscreen,
}

#[derive(Subcommand)]
enum Commands {
    /// Open the interactive REPL directly
    Repl,
    /// Run a single prompt and exit
    Prompt {
        /// The text to send to the model
        text: String,
    },
    /// Check system health: model files, CUDA, version
    Doctor,
    /// Discover local prerequisites, write config, and optionally run a smoke prompt
    Setup {
        /// Write config/wrapper only and skip the live smoke prompt
        #[arg(long)]
        skip_smoke: bool,
    },
    /// Check for updates or fast-forward this checkout and rebuild zipcode
    Update {
        /// Only inspect update status; do not pull or rebuild
        #[arg(long)]
        check: bool,
        /// Rebuild even when the checkout is already up to date
        #[arg(long)]
        rebuild: bool,
    },
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(false)
        .compact()
        .init();

    let cli = Cli::parse();

    // Wire the process-wide verbose flag before any UI callbacks spin
    // up: the flag is read during streaming, not consulted again at
    // construction time. `ZIPCODE_VERBOSE` acts as a fallback so shell
    // wrappers can flip it without editing argv.
    let verbose_env = std::env::var("ZIPCODE_VERBOSE")
        .ok()
        .filter(|value| !matches!(value.as_str(), "" | "0" | "false"))
        .is_some();
    render::set_verbose(cli.verbose || verbose_env);

    let model_path = cli.model.as_deref();
    let permission_mode = cli.permission_mode.as_deref();
    let backend = cli.backend.as_deref();
    let session_id = cli.session.as_deref();

    match cli.command {
        Some(Commands::Repl) => {
            commands::run_repl_command(model_path, permission_mode, backend, session_id, cli.ui)?;
        }
        Some(Commands::Doctor) => {
            commands::doctor(model_path, backend)?;
        }
        Some(Commands::Setup { skip_smoke }) => {
            commands::setup(model_path, backend, skip_smoke)?;
        }
        Some(Commands::Update { check, rebuild }) => {
            commands::update(check, rebuild)?;
        }
        Some(Commands::Prompt { text }) => {
            commands::run_prompt_command(&text, model_path, permission_mode, backend, session_id)?;
        }
        None => {
            commands::run_default(model_path, permission_mode, backend, session_id, cli.ui)?;
        }
    }

    Ok(())
}
