use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

mod commands;
mod render;
mod repl;
mod tui;
mod tui_composer;
mod width;

/// CLI-validated permission mode.
///
/// Uses clap's `ValueEnum` to reject invalid values at parse time,
/// consistent with `--backend` and `--ui`.  Internally converted to
/// the kebab-case string that the rest of the codebase expects.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum CliPermissionMode {
    ReadOnly,
    WorkspaceWrite,
    FullAccess,
}

impl CliPermissionMode {
    /// Return the kebab-case string accepted by `parse_permission_mode()`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::FullAccess => "full-access",
        }
    }
}

impl std::fmt::Display for CliPermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Parser)]
#[command(name = "zipcode", version, about = "Local AI coding assistant")]
struct Cli {
    /// Path to the model directory or .gguf file
    #[arg(long, value_name = "PATH", global = true)]
    model: Option<PathBuf>,

    /// Permission mode: read-only, workspace-write, full-access
    #[arg(long, value_name = "MODE", global = true)]
    permission_mode: Option<CliPermissionMode>,

    /// Inference backend override: llama-cpp, llama-server, or candle (auto-selects when omitted)
    #[arg(long, value_name = "BACKEND", global = true)]
    backend: Option<String>,

    /// UI mode for interactive sessions (only `repl` / `prompt` / no-subcommand
    /// honor this flag; `doctor`/`setup`/`update` reject it).
    #[arg(long, value_enum, global = true)]
    ui: Option<UiMode>,

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

    /// Skip every approval prompt for the duration of this run by
    /// forcing `permission_mode = full-access`. Equivalent to
    /// `--permission-mode full-access`, useful as a one-shot override
    /// inside a workspace where you trust the agent. Conflicts with
    /// an explicit `--permission-mode`.
    #[arg(long, global = true)]
    yolo: bool,

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
    /// Invoke a named skill as a one-shot task
    Skill {
        /// Skill name to invoke
        name: String,
        /// Parameter substitutions in key=value form (repeatable)
        #[arg(long = "param", value_name = "KEY=VALUE", num_args = 1)]
        params: Vec<String>,
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

/// Resolve the effective `permission_mode` string from the two CLI
/// surfaces that can set it: `--yolo` (shorthand for `full-access`) and
/// `--permission-mode <MODE>`. Mutually exclusive — pass one or neither.
fn resolve_permission_mode_str(
    yolo: bool,
    permission_mode: Option<CliPermissionMode>,
) -> Result<Option<&'static str>> {
    if yolo && permission_mode.is_some() {
        anyhow::bail!(
            "--yolo and --permission-mode are mutually exclusive (use one or the other)"
        );
    }
    Ok(if yolo {
        Some("full-access")
    } else {
        permission_mode.map(CliPermissionMode::as_str)
    })
}

/// Error out when global flags `--ui` / `--session` are passed to a
/// subcommand that wouldn't honor them. Previously such flags were
/// silently swallowed, which masked typos in shell scripts and made
/// `--help` output noisy with irrelevant options. Tracked as #59 / #72.
fn reject_irrelevant_global_flags(
    command: Option<&Commands>,
    ui_set: bool,
    session_set: bool,
) -> Result<()> {
    let (sub_name, accepts_ui, accepts_session) = match command {
        Some(Commands::Doctor) => ("doctor", false, false),
        Some(Commands::Setup { .. }) => ("setup", false, false),
        Some(Commands::Update { .. }) => ("update", false, false),
        Some(Commands::Skill { .. }) => ("skill", false, false),
        // Repl, Prompt, and the no-subcommand default both honor --ui
        // and --session.
        _ => return Ok(()),
    };
    if ui_set && !accepts_ui {
        anyhow::bail!("--ui is not applicable to `{sub_name}`");
    }
    if session_set && !accepts_session {
        anyhow::bail!("--session is not applicable to `{sub_name}`");
    }
    Ok(())
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
    let permission_mode = resolve_permission_mode_str(cli.yolo, cli.permission_mode)?;
    let backend = cli.backend.as_deref();
    let session_id = cli.session.as_deref();

    // Reject global flags whose effect would be silently dropped on
    // non-interactive subcommands first, before any session validation
    // tries to load the file. Tracked as #59 / #72.
    reject_irrelevant_global_flags(cli.command.as_ref(), cli.ui.is_some(), session_id.is_some())?;

    // Fail fast on a typo'd `--session` regardless of subcommand. Without
    // this, `zipcode --session does-not-exist doctor` (and any other
    // subcommand that doesn't actually consume `session_id`) silently
    // ignored the flag and exited 0, hiding shell-script bugs.
    if let Some(id) = session_id {
        commands::validate_session_id_or_exit(id)?;
    }

    let resolved_ui = cli.ui.unwrap_or(UiMode::Fullscreen);

    match cli.command {
        Some(Commands::Repl) => {
            commands::run_repl_command(
                model_path,
                permission_mode,
                backend,
                session_id,
                resolved_ui,
            )?;
        }
        Some(Commands::Doctor) => {
            commands::doctor(model_path, backend)?;
        }
        Some(Commands::Setup { skip_smoke }) => {
            commands::setup(model_path, backend, permission_mode, skip_smoke)?;
        }
        Some(Commands::Update { check, rebuild }) => {
            commands::update(check, rebuild)?;
        }
        Some(Commands::Prompt { text }) => {
            commands::run_prompt_command(&text, model_path, permission_mode, backend, session_id)?;
        }
        Some(Commands::Skill { name, params }) => {
            commands::run_skill_command(&name, &params, model_path, permission_mode, backend)?;
        }
        None => {
            commands::run_default(
                model_path,
                permission_mode,
                backend,
                session_id,
                resolved_ui,
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;

    // --- CliPermissionMode unit tests ---

    #[test]
    fn cli_permission_mode_as_str() {
        assert_eq!(CliPermissionMode::ReadOnly.as_str(), "read-only");
        assert_eq!(
            CliPermissionMode::WorkspaceWrite.as_str(),
            "workspace-write"
        );
        assert_eq!(CliPermissionMode::FullAccess.as_str(), "full-access");
    }

    #[test]
    fn cli_permission_mode_display() {
        assert_eq!(CliPermissionMode::ReadOnly.to_string(), "read-only");
        assert_eq!(
            CliPermissionMode::WorkspaceWrite.to_string(),
            "workspace-write"
        );
        assert_eq!(CliPermissionMode::FullAccess.to_string(), "full-access");
    }

    // --- --yolo flag tests ---

    #[test]
    fn yolo_alone_yields_full_access() {
        assert_eq!(
            resolve_permission_mode_str(true, None).unwrap(),
            Some("full-access")
        );
    }

    #[test]
    fn yolo_plus_permission_mode_is_rejected() {
        let err = resolve_permission_mode_str(true, Some(CliPermissionMode::ReadOnly))
            .expect_err("yolo + permission-mode must conflict");
        assert!(
            err.to_string().contains("mutually exclusive"),
            "error should mention mutual exclusion, got: {err}"
        );
    }

    #[test]
    fn no_yolo_no_permission_mode_yields_none() {
        assert_eq!(resolve_permission_mode_str(false, None).unwrap(), None);
    }

    #[test]
    fn permission_mode_alone_passes_through() {
        assert_eq!(
            resolve_permission_mode_str(false, Some(CliPermissionMode::WorkspaceWrite)).unwrap(),
            Some("workspace-write")
        );
    }

    #[test]
    fn clap_accepts_yolo_flag() {
        let cli = Cli::try_parse_from(["zipcode", "--yolo", "doctor"])
            .expect("--yolo should parse");
        assert!(cli.yolo);
        assert!(cli.permission_mode.is_none());
    }

    #[test]
    fn cli_permission_mode_all_variants_roundtrip() {
        // Verify every ValueEnum variant maps to a string that parse_permission_mode accepts
        let variants = [
            CliPermissionMode::ReadOnly,
            CliPermissionMode::WorkspaceWrite,
            CliPermissionMode::FullAccess,
        ];
        for v in variants {
            let s = v.as_str();
            let parsed = zipcode_runtime::parse_permission_mode(s);
            assert!(
                parsed.is_ok(),
                "parse_permission_mode({s:?}) should succeed"
            );
        }
    }

    // --- Clap parse-time rejection tests ---

    fn extract_clap_error(result: Result<Cli, clap::Error>) -> clap::Error {
        match result {
            Err(e) => e,
            Ok(_) => panic!("expected clap error but parsing succeeded"),
        }
    }

    #[test]
    fn clap_rejects_invalid_permission_mode() {
        let err = extract_clap_error(Cli::try_parse_from([
            "zipcode",
            "--permission-mode",
            "invalid",
        ]));
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }

    #[test]
    fn clap_rejects_typo_permission_mode() {
        let err = extract_clap_error(Cli::try_parse_from([
            "zipcode",
            "--permission-mode",
            "workspae-write",
        ]));
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }

    #[test]
    fn clap_rejects_underscore_permission_mode() {
        let err = extract_clap_error(Cli::try_parse_from([
            "zipcode",
            "--permission-mode",
            "full_access",
        ]));
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }

    #[test]
    fn clap_accepts_valid_permission_modes() {
        for mode in &["read-only", "workspace-write", "full-access"] {
            let result = Cli::try_parse_from(["zipcode", "--permission-mode", mode, "doctor"]);
            assert!(result.is_ok(), "should accept {mode}");
            let cli = result.unwrap();
            let pm = cli.permission_mode.expect("should be set");
            assert_eq!(pm.as_str(), *mode);
        }
    }

    #[test]
    fn clap_permission_mode_unset_by_default() {
        let result = Cli::try_parse_from(["zipcode", "doctor"]).unwrap();
        assert!(result.permission_mode.is_none());
    }
}
