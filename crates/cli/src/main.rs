use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

mod commands;
mod render;
mod repl;

#[derive(Parser)]
#[command(name = "zipcode", version, about = "Local AI coding assistant")]
struct Cli {
    /// Path to the model directory or .gguf file
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Permission mode: read-only, workspace-write, full-access
    #[arg(long, value_name = "MODE")]
    permission_mode: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a single prompt and exit
    Prompt {
        /// The text to send to the model
        text: String,
    },
    /// Check system health: model files, CUDA, version
    Doctor,
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
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

    match cli.command {
        Some(Commands::Doctor) => {
            commands::doctor(model_path);
        }
        Some(Commands::Prompt { text }) => {
            repl::run_oneshot(&text, model_path, permission_mode)?;
        }
        None => {
            // Interactive REPL mode
            repl::run_interactive(model_path, permission_mode)?;
        }
    }

    Ok(())
}
