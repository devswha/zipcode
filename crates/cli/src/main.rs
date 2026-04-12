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

    let model_path = cli.model.as_deref();
    let permission_mode = cli.permission_mode.as_deref();
    let backend = cli.backend.as_deref();

    match cli.command {
        Some(Commands::Repl) => {
            tui::run_interactive_with_ui(model_path, permission_mode, backend, cli.ui)?;
        }
        Some(Commands::Doctor) => {
            commands::doctor(model_path, backend)?;
        }
        Some(Commands::Setup { skip_smoke }) => {
            commands::setup(model_path, backend, skip_smoke)?;
        }
        Some(Commands::Prompt { text }) => {
            repl::run_oneshot(&text, model_path, permission_mode, backend)?;
        }
        None => {
            commands::run_default(model_path, permission_mode, backend, cli.ui)?;
        }
    }

    Ok(())
}
