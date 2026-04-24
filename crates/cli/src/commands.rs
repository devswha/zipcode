use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use zipcode_inference::Backend;
use zipcode_runtime::config::{
    expand_user_path, find_project_root, global_config_path, resolve_project_path,
};
use zipcode_runtime::ZipcodeConfig;

use crate::repl::{
    discover_helper, has_nonempty_parent, is_probably_gemma4_model, prepare_loop,
    resolve_effective_backend, resolve_model_path, resolve_requested_backend, run_oneshot,
    server_options_from_config, validate_backend_configuration, CliCallback,
};
use crate::tui::run_interactive_with_ui;
use crate::UiMode;

const SETUP_WRAPPER_NAME: &str = "zipcode-local";
const TOKENIZER_FILE_NAME: &str = "tokenizer.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadinessStatus {
    NativeOk,
    FallbackRequired,
    MissingServer,
    MissingModel,
    MissingTokenizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UserReadiness {
    Ready,
    NeedsSetup,
    NeedsRepair,
}

impl UserReadiness {
    const fn headline(self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::NeedsSetup => "Setup needed",
            Self::NeedsRepair => "Repair needed",
        }
    }

    const fn startup_title(self) -> &'static str {
        match self {
            Self::Ready => "zipcode is ready.",
            Self::NeedsSetup => "Setup needed before zipcode can start.",
            Self::NeedsRepair => "Repair needed before zipcode can start.",
        }
    }
}

#[derive(Debug, Clone)]
struct ConfigLoad {
    config: ZipcodeConfig,
    warning: Option<String>,
    warning_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct ReadinessReport {
    status: ReadinessStatus,
    backend: Backend,
    model: Option<PathBuf>,
    model_search: Vec<PathBuf>,
    model_issue: Option<String>,
    model_issue_is_misconfigured: bool,
    tokenizer: Option<PathBuf>,
    tokenizer_issue: Option<String>,
    llama_server_bin: Option<PathBuf>,
    llama_server_issue: Option<String>,
    llama_server_issue_is_misconfigured: bool,
    backend_issue: Option<String>,
    backend_issue_is_misconfigured: bool,
}

#[derive(Debug, Clone)]
struct UpdateStatus {
    repo_root: PathBuf,
    branch: String,
    local_head: String,
    remote_ref: String,
    remote_head: Option<String>,
    ahead: Option<usize>,
    behind: Option<usize>,
    dirty: bool,
}

/// Default zipcode entrypoint: start the REPL when ready, otherwise guide setup/repair.
pub fn run_default(
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_override: Option<&str>,
    session_id: Option<&str>,
    ui_mode: UiMode,
) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let config_load = load_config_with_warning(&cwd);
    let report = build_readiness_report(&cwd, &config_load.config, model_path, backend_override)?;
    let readiness = classify_user_readiness(&report, config_load.warning.as_deref());

    match readiness {
        UserReadiness::Ready => run_interactive_with_ui(
            model_path,
            permission_mode,
            backend_override,
            session_id,
            ui_mode,
        ),
        state => {
            print_startup_guidance(
                state,
                &report,
                config_load.warning.as_deref(),
                config_load.warning_path.as_deref(),
            );
            Ok(())
        }
    }
}

pub fn run_repl_command(
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_override: Option<&str>,
    session_id: Option<&str>,
    ui_mode: UiMode,
) -> Result<()> {
    run_default(
        model_path,
        permission_mode,
        backend_override,
        session_id,
        ui_mode,
    )
}

pub fn run_prompt_command(
    text: &str,
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_override: Option<&str>,
    session_id: Option<&str>,
) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let config_load = load_config_with_warning(&cwd);
    let report = build_readiness_report(&cwd, &config_load.config, model_path, backend_override)?;
    let readiness = classify_user_readiness(&report, config_load.warning.as_deref());

    match readiness {
        UserReadiness::Ready => run_oneshot(
            text,
            model_path,
            permission_mode,
            backend_override,
            session_id,
        ),
        state => {
            print_startup_guidance(
                state,
                &report,
                config_load.warning.as_deref(),
                config_load.warning_path.as_deref(),
            );
            Ok(())
        }
    }
}

/// Parse `key=value` strings into a `HashMap<String, String>`.
/// Returns an error if any entry is missing the `=` separator.
fn parse_params(raw: &[String]) -> Result<std::collections::HashMap<String, String>> {
    let mut map = std::collections::HashMap::new();
    for entry in raw {
        let (k, v) = entry
            .split_once('=')
            .ok_or_else(|| anyhow!("invalid --param '{entry}': expected key=value"))?;
        map.insert(k.to_string(), v.to_string());
    }
    Ok(map)
}

/// Invoke a named skill as a one-shot task and print the child summary to stdout.
pub fn run_skill_command(
    skill_name: &str,
    raw_params: &[String],
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_override: Option<&str>,
) -> Result<()> {
    use zipcode_runtime::config::find_project_root;
    use zipcode_runtime::SkillRegistry;

    let params = parse_params(raw_params)?;

    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let project_root = find_project_root(&cwd);
    let skills_dir = project_root.join(".zipcode/skills");

    let registry = SkillRegistry::load_from(&skills_dir).unwrap_or_default();

    let skill = match registry.get(skill_name) {
        Some(s) => s,
        None => {
            let mut available = registry.names();
            available.sort();
            if available.is_empty() {
                eprintln!("skill '{skill_name}' not found. No skills are available.");
                eprintln!("Add skill files to .zipcode/skills/");
            } else {
                eprintln!(
                    "skill '{skill_name}' not found. Available skills: {}",
                    available.join(", ")
                );
            }
            std::process::exit(1);
        }
    };

    let task_prompt = skill.render(&params);

    let launch = prepare_loop(model_path, permission_mode, backend_override, None)?;
    for notice in &launch.startup_notices {
        eprintln!("\x1b[33m[notice]\x1b[0m {notice}");
    }
    let mut conv = launch.conv;
    let mut cb = CliCallback::new();
    cb.start_turn();
    conv.run_turn(&task_prompt, &mut cb)?;
    cb.stop_turn();
    println!();
    Ok(())
}

/// Run the doctor command: check version, GPU support, local AI files, and fallback readiness.
pub fn doctor(model_path: Option<&Path>, backend_override: Option<&str>) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let config_load = load_config_with_warning(&cwd);
    let report = build_readiness_report(&cwd, &config_load.config, model_path, backend_override)?;
    let readiness = classify_user_readiness(&report, config_load.warning.as_deref());

    println!("zipcode doctor\n");
    println!("Status: {}", readiness.headline());
    println!("Engine: {}", friendly_engine_name(report.backend));
    println!();

    for line in doctor_check_lines(&report, config_load.warning.as_deref()) {
        println!("{line}");
    }

    println!();
    println!("Next:");
    for line in next_steps(
        readiness,
        &report,
        config_load.warning.as_deref(),
        config_load.warning_path.as_deref(),
    ) {
        println!("  • {line}");
    }

    println!();
    println!(
        "Tip: run `zipcode` with no arguments. It will start the chat REPL when ready and show setup help otherwise."
    );
    Ok(())
}

/// Discover local prerequisites, write global config + launcher, and optionally run a smoke prompt.
pub fn setup(
    model_path: Option<&Path>,
    backend_override: Option<&str>,
    permission_mode: Option<&str>,
    skip_smoke: bool,
) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let config_load = load_config_with_warning(&cwd);
    let report = build_readiness_report(&cwd, &config_load.config, model_path, backend_override)?;
    let readiness = classify_user_readiness(&report, None);

    let model = report
        .model
        .clone()
        .ok_or_else(|| anyhow!(missing_model_message(&report)))?;
    let mut global_config = ZipcodeConfig::load_global().unwrap_or_default();
    global_config.model_dir = model
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    global_config.model_file = model
        .file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned);
    global_config
        .llama_server_bin
        .clone_from(&report.llama_server_bin);
    if let Some(mode) = permission_mode {
        global_config.permission_mode = mode.to_string();
    }
    let perf_defaults = apply_recommended_performance_defaults(
        &mut global_config,
        &model,
        report.llama_server_bin.as_deref(),
    );

    let config_path = global_config.save_global()?;
    let wrapper_path = write_setup_wrapper()?;

    println!("zipcode setup\n");
    println!("Saved settings: {}", config_path.display());
    println!("Launcher:       {}", wrapper_path.display());
    println!("AI model:       {}", model.display());
    println!("Engine:         {}", friendly_engine_name(report.backend));
    println!("Status:         {}", readiness.headline());
    if let Some(path) = &report.tokenizer {
        println!("Tokenizer:      {}", path.display());
    }
    if let Some(path) = &report.llama_server_bin {
        println!("Helper:         {}", path.display());
    }
    if !perf_defaults.is_empty() {
        println!("Performance:    {}", perf_defaults.join(", "));
    }
    println!();

    for line in next_steps(readiness, &report, None, None) {
        println!("{line}");
    }

    if !matches!(readiness, UserReadiness::Ready) {
        if skip_smoke {
            println!("Smoke: skipped until setup is complete");
            return Ok(());
        }

        anyhow::bail!(
            "setup incomplete: add the missing files, then rerun `zipcode setup --skip-smoke`"
        );
    }

    if skip_smoke {
        println!("Smoke: skipped (--skip-smoke)");
        return Ok(());
    }

    println!("Smoke: running prompt...\n");
    run_oneshot(
        "Reply with READY only.",
        Some(&model),
        Some("read-only"),
        Some(backend_name(report.backend)),
        None,
    )?;
    println!("Smoke: PASS");
    Ok(())
}

/// Check for updates or fast-forward the current checkout and rebuild zipcode.
pub fn update(check_only: bool, rebuild: bool) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let status = inspect_update_status(&cwd)?;

    println!("zipcode update\n");
    println!("Repo:         {}", status.repo_root.display());
    println!("Branch:       {}", status.branch);
    println!("Local HEAD:   {}", status.local_head);
    println!(
        "Remote HEAD:  {} ({})",
        status
            .remote_head
            .as_deref()
            .unwrap_or("not checked (working tree dirty)"),
        status.remote_ref
    );
    match (status.ahead, status.behind) {
        (Some(ahead), Some(behind)) => println!("Ahead/behind: {ahead}/{behind}"),
        _ => println!("Ahead/behind: not checked (working tree dirty)"),
    }
    println!(
        "Working tree: {}",
        if status.dirty { "dirty" } else { "clean" }
    );
    println!();

    if check_only {
        print_update_check_summary(&status);
        return Ok(());
    }

    ensure_update_is_safe(&status)?;

    let changed = if status.behind.unwrap_or(0) > 0 {
        run_checked(
            &status.repo_root,
            "git",
            &["pull", "--ff-only", "origin", &status.branch],
            "fast-forward the checkout",
        )?;
        true
    } else {
        println!("Already up to date.");
        false
    };

    if changed || rebuild {
        println!();
        println!("Rebuilding zipcode...");
        run_checked(
            &status.repo_root,
            "cargo",
            &["build", "-p", "zipcode"],
            "rebuild zipcode",
        )?;
    }

    println!();
    println!("Verifying with doctor...");
    run_checked(
        &status.repo_root,
        "cargo",
        &["run", "--quiet", "-p", "zipcode", "--", "doctor"],
        "run doctor after update",
    )?;

    println!();
    println!("Update complete.");
    Ok(())
}

/// Check if CUDA is available by looking for libcuda.so or `CUDA_PATH` env var.
pub fn check_cuda() -> bool {
    if std::env::var("CUDA_PATH").is_ok() {
        return true;
    }
    if std::env::var("CUDA_HOME").is_ok() {
        return true;
    }

    let cuda_libs = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/libcuda.so",
        "/usr/local/cuda/lib64/libcuda.so",
        "/usr/local/cuda/lib/libcuda.so",
    ];

    for lib in &cuda_libs {
        if Path::new(lib).exists() {
            return true;
        }
    }

    if let Ok(output) = std::process::Command::new("ldconfig").arg("-p").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains("libcuda.so") {
            return true;
        }
    }

    false
}

fn read_global_config_json() -> Option<serde_json::Value> {
    let path = global_config_path();
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<serde_json::Value>(&content).ok()
}

fn config_field_missing_or_null(raw: Option<&serde_json::Value>, field: &str) -> bool {
    raw.and_then(|json| json.get(field))
        .is_none_or(serde_json::Value::is_null)
}

fn should_upgrade_flash_attention(config: &ZipcodeConfig, raw: Option<&serde_json::Value>) -> bool {
    if config.flash_attention {
        return false;
    }

    match raw.and_then(|json| json.get("flash_attention")) {
        None => true,
        Some(value) if value.is_null() => true,
        Some(value) if value.as_bool() == Some(false) => {
            config_field_missing_or_null(raw, "gpu_layers")
        }
        _ => false,
    }
}

fn apply_recommended_performance_defaults(
    config: &mut ZipcodeConfig,
    model: &Path,
    helper_path: Option<&Path>,
) -> Vec<String> {
    let mut applied = Vec::new();
    if !is_probably_gemma4_model(model) || helper_path.is_none() || !check_cuda() {
        return applied;
    }

    let raw_config = read_global_config_json();
    let raw_config = raw_config.as_ref();

    if config.gpu_layers.is_none() && config_field_missing_or_null(raw_config, "gpu_layers") {
        config.gpu_layers = Some(999);
        applied.push("gpu_layers=999".to_string());
    }

    if should_upgrade_flash_attention(config, raw_config) {
        config.flash_attention = true;
        applied.push("flash_attention=true".to_string());
    }

    applied
}

fn load_config_with_warning(cwd: &Path) -> ConfigLoad {
    match ZipcodeConfig::load(cwd) {
        Ok(config) => ConfigLoad {
            config,
            warning: None,
            warning_path: None,
        },
        Err(error) => {
            let warning_path = detect_config_warning_path(cwd);
            ConfigLoad {
                config: ZipcodeConfig::default(),
                warning: Some(error.to_string()),
                warning_path,
            }
        }
    }
}

fn detect_config_warning_path(cwd: &Path) -> Option<PathBuf> {
    let global_path = global_config_path();
    let project_path = find_project_root(cwd).join(".zipcode.json");
    detect_config_warning_path_from_paths(&global_path, &project_path)
}

fn detect_config_warning_path_from_paths(
    global_path: &Path,
    project_path: &Path,
) -> Option<PathBuf> {
    fn validate_json_file(path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(path)?;
        let _: serde_json::Value = serde_json::from_str(&content)?;
        Ok(())
    }

    if global_path.exists() && validate_json_file(global_path).is_err() {
        return Some(global_path.to_path_buf());
    }

    if project_path.exists() && validate_json_file(project_path).is_err() {
        return Some(project_path.to_path_buf());
    }

    None
}

fn inspect_update_status(cwd: &Path) -> Result<UpdateStatus> {
    let repo_root = git_stdout(cwd, &["rev-parse", "--show-toplevel"], "find git repo root")?;
    let repo_root = PathBuf::from(repo_root.trim());
    let branch = git_stdout(
        &repo_root,
        &["branch", "--show-current"],
        "read current branch",
    )?;
    let branch = branch.trim().to_string();
    if branch.is_empty() {
        anyhow::bail!("zipcode update requires a named branch checkout");
    }

    let dirty = !git_stdout(&repo_root, &["status", "--short"], "read git status")?
        .trim()
        .is_empty();
    let local_head = git_stdout(
        &repo_root,
        &["rev-parse", "--short", "HEAD"],
        "read local HEAD",
    )?;
    let local_head = local_head.trim().to_string();
    let remote_ref = format!("origin/{branch}");

    if dirty {
        return Ok(UpdateStatus {
            repo_root,
            branch,
            local_head,
            remote_ref,
            remote_head: None,
            ahead: None,
            behind: None,
            dirty,
        });
    }

    run_checked(&repo_root, "git", &["fetch", "origin"], "fetch origin")?;
    let remote_head = git_stdout(
        &repo_root,
        &["rev-parse", "--short", &remote_ref],
        "read remote HEAD",
    )?;
    let remote_head = remote_head.trim().to_string();
    let counts = git_stdout(
        &repo_root,
        &[
            "rev-list",
            "--left-right",
            "--count",
            &format!("{remote_ref}...HEAD"),
        ],
        "compare local and remote history",
    )?;
    let mut parts = counts.split_whitespace();
    let behind = parts
        .next()
        .ok_or_else(|| anyhow!("missing behind count from git rev-list output"))?
        .parse::<usize>()
        .context("parse behind count")?;
    let ahead = parts
        .next()
        .ok_or_else(|| anyhow!("missing ahead count from git rev-list output"))?
        .parse::<usize>()
        .context("parse ahead count")?;

    Ok(UpdateStatus {
        repo_root,
        branch,
        local_head,
        remote_ref,
        remote_head: Some(remote_head),
        ahead: Some(ahead),
        behind: Some(behind),
        dirty,
    })
}

fn print_update_check_summary(status: &UpdateStatus) {
    if status.dirty {
        println!("Update check: blocked by local modifications.");
        println!("Clean, commit, or stash changes before applying updates.");
        return;
    }

    match (status.ahead, status.behind) {
        (Some(0), Some(0)) => println!("Update check: already up to date."),
        (Some(0), Some(behind)) => println!("Update check: {behind} commit(s) available to pull."),
        (Some(ahead), Some(0)) => println!(
            "Update check: local checkout is {ahead} commit(s) ahead of {}.",
            status.remote_ref
        ),
        (Some(ahead), Some(behind)) => println!(
            "Update check: local and remote have diverged ({ahead} ahead, {behind} behind)."
        ),
        _ => println!("Update check: remote status not checked."),
    }
}

fn ensure_update_is_safe(status: &UpdateStatus) -> Result<()> {
    if status.dirty {
        anyhow::bail!(
            "Refusing to update with local modifications present. Commit, stash, or clean the working tree first."
        );
    }

    let ahead = status.ahead.unwrap_or(0);
    let behind = status.behind.unwrap_or(0);

    if ahead > 0 && behind > 0 {
        anyhow::bail!(
            "Refusing to auto-update because this checkout has diverged from {} (ahead {}, behind {}).",
            status.remote_ref,
            ahead,
            behind
        );
    }

    if ahead > 0 {
        anyhow::bail!(
            "Refusing to auto-update because this checkout is {} commit(s) ahead of {}.",
            ahead,
            status.remote_ref
        );
    }

    Ok(())
}

fn git_stdout(cwd: &Path, args: &[&str], action: &str) -> Result<String> {
    command_stdout(cwd, "git", args, action)
}

fn command_stdout(cwd: &Path, program: &str, args: &[&str], action: &str) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .output()
        .with_context(|| format!("Failed to {action}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to {action}: {}", stderr.trim());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn run_checked(cwd: &Path, program: &str, args: &[&str], action: &str) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .status()
        .with_context(|| format!("Failed to {action}"))?;
    if !status.success() {
        anyhow::bail!("Failed to {action} (exit status: {status})");
    }
    Ok(())
}

fn build_readiness_report(
    cwd: &Path,
    config: &ZipcodeConfig,
    explicit_model_path: Option<&Path>,
    backend_override: Option<&str>,
) -> Result<ReadinessReport> {
    let project_root = find_project_root(cwd);
    let model_resolution = resolve_model_path(explicit_model_path, config, cwd, &project_root);
    let (model, model_issue) = match model_resolution {
        Ok(path) => match validate_gguf_file(&path) {
            Ok(()) => (Some(path), None),
            Err(validation_err) => (Some(path), Some(validation_err.to_string())),
        },
        Err(error) => (None, Some(error.to_string())),
    };
    let model_search = model_search_locations(explicit_model_path, config, cwd, &project_root);
    let model_issue_is_misconfigured = explicit_model_path.is_some() || config.model_file.is_some();
    let llama_server = discover_helper(config);
    let requested_backend = resolve_requested_backend(backend_override)?;
    let backend = model.as_deref().map_or_else(
        || requested_backend.unwrap_or(Backend::LlamaCpp),
        |model_path| {
            resolve_effective_backend(requested_backend, model_path, llama_server.path.as_deref())
        },
    );
    let server_options = server_options_from_config(config);
    let tokenizer_required = tokenizer_is_required(backend, model.as_deref());

    let tokenizer = tokenizer_required
        .then(|| {
            model
                .as_deref()
                .map(tokenizer_path_for_model)
                .filter(|path| path.is_file())
        })
        .flatten();
    let tokenizer_issue = tokenizer_required
        .then(|| {
            model.as_deref().and_then(|model_path| {
                let tokenizer_path = tokenizer_path_for_model(model_path);
                (!tokenizer_path.is_file())
                    .then(|| format!("tokenizer.json is missing next to {}", model_path.display()))
            })
        })
        .flatten();

    let status = classify_readiness(
        backend,
        model.as_deref(),
        tokenizer.as_deref(),
        llama_server.path.as_deref(),
    );
    let backend_issue = model.as_deref().and_then(|model_path| {
        validate_backend_configuration(
            requested_backend,
            backend,
            model_path,
            llama_server.path.as_deref(),
            &server_options,
        )
    });

    Ok(ReadinessReport {
        status,
        backend,
        model,
        model_search,
        model_issue,
        model_issue_is_misconfigured,
        tokenizer,
        tokenizer_issue,
        llama_server_bin: llama_server.path,
        llama_server_issue: llama_server.issue,
        llama_server_issue_is_misconfigured: llama_server.issue_is_blocking,
        backend_issue_is_misconfigured: backend_issue.is_some(),
        backend_issue,
    })
}

const fn classify_user_readiness(
    report: &ReadinessReport,
    config_warning: Option<&str>,
) -> UserReadiness {
    if config_warning.is_some()
        || (report.model_issue.is_some() && report.model_issue_is_misconfigured)
        || (report.llama_server_issue.is_some() && report.llama_server_issue_is_misconfigured)
        || (report.backend_issue.is_some() && report.backend_issue_is_misconfigured)
    {
        return UserReadiness::NeedsRepair;
    }

    match report.status {
        ReadinessStatus::NativeOk | ReadinessStatus::FallbackRequired => UserReadiness::Ready,
        ReadinessStatus::MissingModel
        | ReadinessStatus::MissingTokenizer
        | ReadinessStatus::MissingServer => UserReadiness::NeedsSetup,
    }
}

fn classify_readiness(
    backend: Backend,
    model: Option<&Path>,
    tokenizer: Option<&Path>,
    llama_server_bin: Option<&Path>,
) -> ReadinessStatus {
    let Some(model) = model else {
        return ReadinessStatus::MissingModel;
    };

    if helper_is_required(backend, Some(model)) && llama_server_bin.is_none() {
        return ReadinessStatus::MissingServer;
    }

    if tokenizer_is_required(backend, Some(model)) && tokenizer.is_none() {
        return ReadinessStatus::MissingTokenizer;
    }

    if matches!(backend, Backend::LlamaCpp) && is_probably_gemma4_model(model) {
        if llama_server_bin.is_some() {
            return ReadinessStatus::FallbackRequired;
        }
        return ReadinessStatus::MissingServer;
    }

    ReadinessStatus::NativeOk
}

const fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::LlamaCpp => "llama-cpp",
        Backend::LlamaServer => "llama-server",
        Backend::Candle => "candle",
    }
}

const fn friendly_engine_name(backend: Backend) -> &'static str {
    match backend {
        Backend::LlamaCpp => "local engine (llama-cpp)",
        Backend::LlamaServer => "compatibility helper (llama-server)",
        Backend::Candle => "pure Rust engine (candle)",
    }
}

/// GGUF magic bytes: the ASCII string "GGUF" (0x47 0x47 0x55 0x46).
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Validate that a file looks like a legitimate GGUF model file.
///
/// Checks that the file is non-empty and starts with the GGUF magic header.
///
/// # Errors
///
/// Returns an error if the file is empty, cannot be read, or does not start
/// with the GGUF magic bytes.
fn validate_gguf_file(path: &Path) -> Result<()> {
    let metadata =
        std::fs::metadata(path).with_context(|| format!("Cannot stat {}", path.display()))?;
    let size = metadata.len();
    if size == 0 {
        anyhow::bail!("Model file is empty (0 bytes): {}", path.display());
    }
    if size < 4 {
        anyhow::bail!(
            "Model file is too small to be valid ({size} bytes): {}",
            path.display()
        );
    }

    let mut handle =
        std::fs::File::open(path).with_context(|| format!("Cannot open {}", path.display()))?;
    let mut header = [0u8; 4];
    std::io::Read::read_exact(&mut handle, &mut header)
        .with_context(|| format!("Cannot read header from {}", path.display()))?;

    if header != GGUF_MAGIC {
        anyhow::bail!(
            "Model file does not have a valid GGUF header (found {:02X?}, expected {:02X?}): {}",
            header,
            GGUF_MAGIC,
            path.display()
        );
    }

    Ok(())
}

fn format_model(path: &Path) -> String {
    let size = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    #[allow(clippy::cast_precision_loss)]
    let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
    format!("{} ({size_gb:.1} GB)", path.display())
}

fn print_startup_guidance(
    readiness: UserReadiness,
    report: &ReadinessReport,
    config_warning: Option<&str>,
    config_warning_path: Option<&Path>,
) {
    println!("zipcode\n");
    println!("{}", readiness.startup_title());
    println!();

    let details = startup_details(report, config_warning);
    if !details.is_empty() {
        println!("What zipcode found:");
        for detail in details {
            println!("  • {detail}");
        }
        println!();
    }

    println!("Next:");
    for (index, step) in next_steps(readiness, report, config_warning, config_warning_path)
        .iter()
        .enumerate()
    {
        println!("  {}. {step}", index + 1);
    }
}

fn startup_details(report: &ReadinessReport, config_warning: Option<&str>) -> Vec<String> {
    let mut lines = Vec::new();
    let tokenizer_required = tokenizer_is_required(report.backend, report.model.as_deref());

    if let Some(warning) = config_warning {
        lines.push(format!("Saved settings could not be read ({warning})."));
    }

    if report.model_issue_is_misconfigured {
        if let Some(issue) = &report.model_issue {
            lines.push(issue.clone());
        }
    } else if report.model.is_none() {
        lines.push("No local AI model was found yet.".to_string());
    }

    if tokenizer_required {
        if let Some(issue) = &report.tokenizer_issue {
            lines.push(issue.clone());
        }
    }

    if let Some(issue) = &report.backend_issue {
        lines.push(issue.clone());
    }

    if report.llama_server_bin.is_some() {
        if let Some(issue) = &report.llama_server_issue {
            lines.push(issue.clone());
        }
    }

    if helper_is_required(report.backend, report.model.as_deref())
        && report.llama_server_bin.is_none()
    {
        if let Some(issue) = &report.llama_server_issue {
            lines.push(issue.clone());
        } else {
            lines.push(
                "The compatibility helper needed for this model is not installed yet.".to_string(),
            );
        }
    }

    lines
}

fn doctor_check_lines(report: &ReadinessReport, config_warning: Option<&str>) -> Vec<String> {
    let mut lines = Vec::new();
    let tokenizer_required = tokenizer_is_required(report.backend, report.model.as_deref());

    if check_cuda() {
        lines.push("  ✅ GPU support: available".to_string());
    } else {
        lines.push("  ❌ GPU support: not found (zipcode can still run on CPU)".to_string());
    }

    if let Some(warning) = config_warning {
        lines.push(format!("  ⚠️ Saved settings: {warning}"));
    }

    match &report.model {
        Some(path) => lines.push(format!("  ✅ AI model: {}", format_model(path))),
        None if report.model_issue_is_misconfigured => lines.push(format!(
            "  ⚠️ AI model: {}",
            report
                .model_issue
                .as_deref()
                .unwrap_or("saved model path could not be used")
        )),
        None => {
            lines.push("  ❌ AI model: not found".to_string());
            lines.push(format!(
                "     Looked in: {}",
                format_search_paths(&report.model_search)
            ));
        }
    }

    if tokenizer_required && report.model.is_some() {
        match &report.tokenizer {
            Some(path) => lines.push(format!("  ✅ Tokenizer: {}", path.display())),
            None => lines.push(format!(
                "  ❌ Tokenizer: {}",
                report
                    .tokenizer_issue
                    .as_deref()
                    .unwrap_or("tokenizer.json not found next to the model")
            )),
        }
    }

    if let Some(issue) = &report.backend_issue {
        lines.push(format!("  ⚠️ Backend readiness: {issue}"));
    }

    if helper_is_required(report.backend, report.model.as_deref()) {
        match &report.llama_server_bin {
            Some(path) => lines.push(format!("  ✅ Compatibility helper: {}", path.display())),
            None if report.llama_server_issue_is_misconfigured => lines.push(format!(
                "  ⚠️ Compatibility helper: {}",
                report
                    .llama_server_issue
                    .as_deref()
                    .unwrap_or("saved compatibility helper path could not be used")
            )),
            None => lines.push("  ❌ Compatibility helper: not found".to_string()),
        }

        if report.llama_server_bin.is_some() {
            if let Some(issue) = &report.llama_server_issue {
                lines.push(format!("  ⚠️ Compatibility helper note: {issue}"));
            }
        }
    } else {
        lines.push("  ℹ️ Compatibility helper: not needed for this setup".to_string());
    }

    lines
}

fn next_steps(
    readiness: UserReadiness,
    report: &ReadinessReport,
    config_warning: Option<&str>,
    config_warning_path: Option<&Path>,
) -> Vec<String> {
    match readiness {
        UserReadiness::Ready => {
            let mut lines = vec!["Run `zipcode` to start the chat REPL.".to_string()];
            if matches!(report.status, ReadinessStatus::FallbackRequired) {
                let helper = report
                    .llama_server_bin
                    .as_deref()
                    .unwrap_or_else(|| Path::new("llama-server"));
                lines.push(format!(
                    "zipcode will use the compatibility helper at {} automatically for this model.",
                    helper.display()
                ));
            }
            lines
        }
        UserReadiness::NeedsSetup => setup_steps(report),
        UserReadiness::NeedsRepair => repair_steps(report, config_warning, config_warning_path),
    }
}

fn setup_steps(report: &ReadinessReport) -> Vec<String> {
    let tokenizer_required = tokenizer_is_required(report.backend, report.model.as_deref());

    match report.status {
        ReadinessStatus::MissingModel => {
            let mut lines = vec![format!(
                "Copy a .gguf AI model into {} or rerun with `--model <PATH>`.",
                model_location_hint(report)
            )];
            if tokenizer_required {
                lines.push("Copy the matching tokenizer.json next to that model file.".to_string());
            }
            lines.push("Run `zipcode setup --skip-smoke`, then run `zipcode` again.".to_string());
            lines
        }
        ReadinessStatus::MissingTokenizer => vec![
            format!(
                "Copy tokenizer.json next to {}.",
                report
                    .model
                    .as_deref()
                    .unwrap_or_else(|| Path::new("your-model.gguf"))
                    .display()
            ),
            "Run `zipcode setup --skip-smoke`, then run `zipcode` again.".to_string(),
        ],
        ReadinessStatus::MissingServer => vec![
            "Install the compatibility helper needed for this model.".to_string(),
            missing_server_hint(),
            "Run `zipcode setup --skip-smoke`, then run `zipcode` again.".to_string(),
        ],
        ReadinessStatus::NativeOk | ReadinessStatus::FallbackRequired => {
            vec!["Run `zipcode` to start the chat REPL.".to_string()]
        }
    }
}

fn repair_steps(
    report: &ReadinessReport,
    config_warning: Option<&str>,
    config_warning_path: Option<&Path>,
) -> Vec<String> {
    let mut lines = Vec::new();
    let tokenizer_required = tokenizer_is_required(report.backend, report.model.as_deref());

    if let Some(warning) = config_warning {
        let global_path = global_config_path();
        let warning_path = config_warning_path.unwrap_or(global_path.as_path());
        lines.push(format!(
            "Fix or replace {} (current error: {warning}).",
            warning_path.display()
        ));
    }

    if report.model_issue_is_misconfigured {
        if let Some(issue) = &report.model_issue {
            lines.push(format!("Update the saved AI model path ({issue})."));
        }
    }

    if tokenizer_required {
        if let Some(issue) = &report.tokenizer_issue {
            lines.push(format!("Restore the missing tokenizer file ({issue})."));
        }
    }

    if report.llama_server_issue_is_misconfigured {
        if let Some(issue) = &report.llama_server_issue {
            lines.push(format!(
                "Update the saved compatibility helper path ({issue})."
            ));
        }
    }

    if report.backend_issue_is_misconfigured {
        if let Some(issue) = &report.backend_issue {
            lines.push(issue.clone());
        }
    }

    if lines.is_empty() {
        lines.push("Run `zipcode doctor` to review the saved paths.".to_string());
    }

    lines.push("Run `zipcode setup --skip-smoke` after fixing the files.".to_string());
    lines.push("Run `zipcode` again.".to_string());
    lines
}

fn helper_is_required(backend: Backend, model: Option<&Path>) -> bool {
    matches!(backend, Backend::LlamaServer) || model.is_some_and(is_probably_gemma4_model)
}

fn tokenizer_is_required(backend: Backend, model: Option<&Path>) -> bool {
    !helper_is_required(backend, model)
}

fn model_location_hint(report: &ReadinessReport) -> String {
    report.model_search.first().map_or_else(
        || {
            dirs::home_dir().map_or_else(
                || "~/.zipcode/models".to_string(),
                |home| home.join(".zipcode/models").display().to_string(),
            )
        },
        |path| path.display().to_string(),
    )
}

fn missing_model_message(report: &ReadinessReport) -> String {
    if report.model_issue_is_misconfigured {
        if let Some(issue) = &report.model_issue {
            return issue.clone();
        }
    }

    format!(
        "No .gguf AI model found. Looked in: {}",
        format_search_paths(&report.model_search)
    )
}

fn missing_server_hint() -> String {
    let install_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".zipcode");
    let helper = install_dir.join("install_llama_server.sh");
    if helper.is_file() {
        format!(
            "Run `{} /path/to/llama-server {}` to install it.",
            helper.display(),
            install_dir.display()
        )
    } else {
        "Set `ZIPCODE_LLAMA_SERVER_BIN=/path/to/llama-server` after installing the helper."
            .to_string()
    }
}

fn format_search_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn tokenizer_path_for_model(model: &Path) -> PathBuf {
    model
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(TOKENIZER_FILE_NAME)
}

fn model_search_locations(
    explicit_model_path: Option<&Path>,
    config: &ZipcodeConfig,
    cwd: &Path,
    project_root: &Path,
) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Some(path) = explicit_model_path {
        paths.push(resolve_path_from_cwd(path, cwd));
        return paths;
    }

    let model_dir = resolve_model_dir(config, project_root);
    if let Some(model_file) = &config.model_file {
        let configured = expand_user_path(Path::new(model_file));
        let resolved = if configured.is_absolute() {
            configured
        } else if has_nonempty_parent(&configured) {
            project_root.join(configured)
        } else {
            model_dir.join(configured)
        };
        paths.push(resolved);
    } else {
        paths.push(model_dir);
    }

    for default_dir in default_model_dirs() {
        if !paths.contains(&default_dir) {
            paths.push(default_dir);
        }
    }

    paths
}

fn resolve_model_dir(config: &ZipcodeConfig, project_root: &Path) -> PathBuf {
    resolve_project_path(&config.model_dir, project_root)
}

fn resolve_path_from_cwd(path: &Path, cwd: &Path) -> PathBuf {
    let expanded = expand_user_path(path);
    if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    }
}

fn default_model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".zipcode/models"));
    }

    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("models"));
    }

    dirs
}

fn write_setup_wrapper() -> Result<PathBuf> {
    let home = dirs::home_dir().context("HOME is not set")?;
    let bin_dir = home.join(".zipcode/bin");
    std::fs::create_dir_all(&bin_dir)?;
    let wrapper_path = bin_dir.join(SETUP_WRAPPER_NAME);
    let script = r#"#!/bin/sh
SETUP_ENV="${HOME}/.zipcode/setup.env"
if [ -f "${SETUP_ENV}" ]; then
  . "${SETUP_ENV}"
fi
ZIPCODE_BIN="${HOME}/.zipcode/bin/zipcode"
if [ -x "${ZIPCODE_BIN}" ]; then
  exec "${ZIPCODE_BIN}" "$@"
fi
exec zipcode "$@"
"#;
    std::fs::write(&wrapper_path, script)?;
    let mut permissions = std::fs::metadata(&wrapper_path)?.permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&wrapper_path, permissions)?;
    Ok(wrapper_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report(status: ReadinessStatus) -> ReadinessReport {
        ReadinessReport {
            status,
            backend: Backend::LlamaCpp,
            model: Some(PathBuf::from("/tmp/model.gguf")),
            model_search: Vec::new(),
            model_issue: None,
            model_issue_is_misconfigured: false,
            tokenizer: Some(PathBuf::from("/tmp/tokenizer.json")),
            tokenizer_issue: None,
            llama_server_bin: None,
            llama_server_issue: None,
            llama_server_issue_is_misconfigured: false,
            backend_issue: None,
            backend_issue_is_misconfigured: false,
        }
    }

    fn sample_update_status() -> UpdateStatus {
        UpdateStatus {
            repo_root: PathBuf::from("/tmp/zipcode"),
            branch: "main".to_string(),
            local_head: "abc1234".to_string(),
            remote_ref: "origin/main".to_string(),
            remote_head: Some("def5678".to_string()),
            ahead: Some(0),
            behind: Some(0),
            dirty: false,
        }
    }

    #[test]
    fn classify_missing_model_when_no_model_is_found() {
        let status = classify_readiness(Backend::LlamaCpp, None, None, None);
        assert_eq!(status, ReadinessStatus::MissingModel);
    }

    #[test]
    fn classify_missing_tokenizer_when_model_has_no_tokenizer() {
        let status = classify_readiness(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/codeqwen.gguf")),
            None,
            Some(Path::new("/tmp/llama-server")),
        );
        assert_eq!(status, ReadinessStatus::MissingTokenizer);
    }

    #[test]
    fn classify_llama_server_is_ready_without_tokenizer() {
        let status = classify_readiness(
            Backend::LlamaServer,
            Some(Path::new("/tmp/gemma-4-test.gguf")),
            None,
            Some(Path::new("/tmp/llama-server")),
        );
        assert_eq!(status, ReadinessStatus::NativeOk);
    }

    #[test]
    fn classify_missing_server_for_gemma4_without_server() {
        let status = classify_readiness(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/gemma-4-test.gguf")),
            None,
            None,
        );
        assert_eq!(status, ReadinessStatus::MissingServer);
    }

    #[test]
    fn classify_fallback_required_for_gemma4_with_server() {
        let status = classify_readiness(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/gemma-4-test.gguf")),
            Some(Path::new("/tmp/tokenizer.json")),
            Some(Path::new("/tmp/llama-server")),
        );
        assert_eq!(status, ReadinessStatus::FallbackRequired);
    }

    #[test]
    fn classify_fallback_required_for_gemma4_without_tokenizer_when_server_exists() {
        let status = classify_readiness(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/gemma-4-test.gguf")),
            None,
            Some(Path::new("/tmp/llama-server")),
        );
        assert_eq!(status, ReadinessStatus::FallbackRequired);
    }

    #[test]
    fn classify_native_ok_for_non_gemma4_model() {
        let status = classify_readiness(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/codeqwen.gguf")),
            Some(Path::new("/tmp/tokenizer.json")),
            None,
        );
        assert_eq!(status, ReadinessStatus::NativeOk);
    }

    #[test]
    fn user_readiness_prefers_repair_for_broken_saved_paths() {
        let mut report = sample_report(ReadinessStatus::MissingModel);
        report.model = None;
        report.model_issue = Some("configured model path not found".to_string());
        report.model_issue_is_misconfigured = true;

        assert_eq!(
            classify_user_readiness(&report, None),
            UserReadiness::NeedsRepair
        );
    }

    #[test]
    fn user_readiness_treats_missing_model_as_setup() {
        let mut report = sample_report(ReadinessStatus::MissingModel);
        report.model = None;
        report.tokenizer = None;
        report.status = ReadinessStatus::MissingModel;

        assert_eq!(
            classify_user_readiness(&report, None),
            UserReadiness::NeedsSetup
        );
    }

    #[test]
    fn user_readiness_prefers_repair_when_config_is_invalid() {
        let report = sample_report(ReadinessStatus::NativeOk);
        assert_eq!(
            classify_user_readiness(&report, Some("invalid json")),
            UserReadiness::NeedsRepair
        );
    }

    #[test]
    fn detect_config_warning_path_prefers_global_config_when_both_are_invalid() {
        let unique = format!(
            "zipcode-config-warning-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&root).unwrap();
        let global_path = root.join("global.json");
        let project_path = root.join("project/.zipcode.json");
        std::fs::create_dir_all(project_path.parent().unwrap()).unwrap();
        std::fs::write(&global_path, "{ invalid json\n").unwrap();
        std::fs::write(&project_path, "{ invalid json\n").unwrap();

        let detected = detect_config_warning_path_from_paths(&global_path, &project_path);

        assert_eq!(detected, Some(global_path));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn flash_attention_upgrade_accepts_stale_default_config() {
        let config = ZipcodeConfig {
            flash_attention: false,
            ..ZipcodeConfig::default()
        };
        let raw = serde_json::json!({
            "gpu_layers": null,
            "flash_attention": false
        });
        assert!(should_upgrade_flash_attention(&config, Some(&raw)));
    }

    #[test]
    fn flash_attention_upgrade_preserves_explicit_false_when_gpu_layers_are_set() {
        let config = ZipcodeConfig {
            gpu_layers: Some(64),
            flash_attention: false,
            ..ZipcodeConfig::default()
        };
        let raw = serde_json::json!({
            "gpu_layers": 64,
            "flash_attention": false
        });
        assert!(!should_upgrade_flash_attention(&config, Some(&raw)));
    }

    #[test]
    fn update_refuses_dirty_checkout() {
        let mut status = sample_update_status();
        status.dirty = true;
        let error = ensure_update_is_safe(&status).unwrap_err().to_string();
        assert!(error.contains("local modifications"));
    }

    #[test]
    fn update_refuses_diverged_checkout() {
        let mut status = sample_update_status();
        status.ahead = Some(2);
        status.behind = Some(3);
        let error = ensure_update_is_safe(&status).unwrap_err().to_string();
        assert!(error.contains("diverged"));
    }

    #[test]
    fn update_refuses_ahead_checkout() {
        let mut status = sample_update_status();
        status.ahead = Some(1);
        let error = ensure_update_is_safe(&status).unwrap_err().to_string();
        assert!(error.contains("ahead"));
    }

    // --- Tests for previously-untested pure utility functions ---

    #[test]
    fn backend_name_maps_all_variants() {
        assert_eq!(backend_name(Backend::LlamaCpp), "llama-cpp");
        assert_eq!(backend_name(Backend::LlamaServer), "llama-server");
        assert_eq!(backend_name(Backend::Candle), "candle");
    }

    #[test]
    fn friendly_engine_name_maps_all_variants() {
        assert_eq!(
            friendly_engine_name(Backend::LlamaCpp),
            "local engine (llama-cpp)"
        );
        assert_eq!(
            friendly_engine_name(Backend::LlamaServer),
            "compatibility helper (llama-server)"
        );
        assert_eq!(
            friendly_engine_name(Backend::Candle),
            "pure Rust engine (candle)"
        );
    }

    #[test]
    fn format_search_paths_joins_multiple_paths() {
        let paths = vec![
            PathBuf::from("/home/user/.zipcode/models"),
            PathBuf::from("/project/models"),
        ];
        assert_eq!(
            format_search_paths(&paths),
            "/home/user/.zipcode/models, /project/models"
        );
    }

    #[test]
    fn format_search_paths_returns_empty_string_for_empty_slice() {
        let paths: Vec<PathBuf> = vec![];
        assert_eq!(format_search_paths(&paths), "");
    }

    #[test]
    fn format_search_paths_handles_single_path() {
        let paths = vec![PathBuf::from("/only/path")];
        assert_eq!(format_search_paths(&paths), "/only/path");
    }

    #[test]
    fn tokenizer_path_for_model_returns_parent_dir_tokenizer() {
        let model = Path::new("/home/user/.zipcode/models/gemma-4.gguf");
        assert_eq!(
            tokenizer_path_for_model(model),
            PathBuf::from("/home/user/.zipcode/models/tokenizer.json")
        );
    }

    #[test]
    fn tokenizer_path_for_model_handles_root_path() {
        // Model at root like /model.gguf → parent is /
        let model = Path::new("/model.gguf");
        assert_eq!(
            tokenizer_path_for_model(model),
            PathBuf::from("/tokenizer.json")
        );
    }

    #[test]
    fn tokenizer_path_for_model_handles_relative_path() {
        let model = Path::new("models/test.gguf");
        assert_eq!(
            tokenizer_path_for_model(model),
            PathBuf::from("models/tokenizer.json")
        );
    }

    #[test]
    fn helper_is_required_true_for_llama_server() {
        assert!(helper_is_required(Backend::LlamaServer, None));
    }

    #[test]
    fn helper_is_required_true_for_gemma4_model() {
        assert!(helper_is_required(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/gemma-4-it.gguf"))
        ));
    }

    #[test]
    fn helper_is_required_true_for_gemma4_model_case_insensitive() {
        assert!(helper_is_required(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/Gemma-4-it.gguf"))
        ));
    }

    #[test]
    fn helper_is_required_false_for_non_gemma4_with_llama_cpp() {
        assert!(!helper_is_required(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/codeqwen.gguf"))
        ));
    }

    #[test]
    fn helper_is_required_false_for_candle_without_model() {
        assert!(!helper_is_required(Backend::Candle, None));
    }

    #[test]
    fn tokenizer_is_required_is_inverse_of_helper_is_required() {
        // LlamaServer backend → helper required → tokenizer NOT required
        assert!(!tokenizer_is_required(Backend::LlamaServer, None));
        // Candle backend without model → helper NOT required → tokenizer required
        assert!(tokenizer_is_required(Backend::Candle, None));
        // Gemma4 model → helper required → tokenizer NOT required
        assert!(!tokenizer_is_required(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/gemma-4.gguf"))
        ));
        // Non-gemma model → helper NOT required → tokenizer required
        assert!(tokenizer_is_required(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/phi.gguf"))
        ));
    }

    #[test]
    fn config_field_missing_or_null_returns_true_when_none() {
        assert!(config_field_missing_or_null(None, "gpu_layers"));
    }

    #[test]
    fn config_field_missing_or_null_returns_true_when_field_absent() {
        let json = serde_json::json!({"other_field": 42});
        assert!(config_field_missing_or_null(Some(&json), "gpu_layers"));
    }

    #[test]
    fn config_field_missing_or_null_returns_true_when_null() {
        let json = serde_json::json!({"gpu_layers": null});
        assert!(config_field_missing_or_null(Some(&json), "gpu_layers"));
    }

    #[test]
    fn config_field_missing_or_null_returns_false_when_present() {
        let json = serde_json::json!({"gpu_layers": 999});
        assert!(!config_field_missing_or_null(Some(&json), "gpu_layers"));
    }

    #[test]
    fn config_field_missing_or_null_returns_false_for_false_bool() {
        let json = serde_json::json!({"flash_attention": false});
        assert!(!config_field_missing_or_null(
            Some(&json),
            "flash_attention"
        ));
    }

    #[test]
    fn should_upgrade_flash_attention_when_no_raw_config() {
        let config = ZipcodeConfig {
            flash_attention: false,
            ..ZipcodeConfig::default()
        };
        // No raw config → treat as missing → upgrade
        assert!(should_upgrade_flash_attention(&config, None));
    }

    #[test]
    fn should_upgrade_flash_attention_when_already_enabled() {
        let config = ZipcodeConfig {
            flash_attention: true,
            ..ZipcodeConfig::default()
        };
        let raw = serde_json::json!({});
        assert!(!should_upgrade_flash_attention(&config, Some(&raw)));
    }

    #[test]
    fn should_upgrade_flash_attention_when_field_null() {
        let config = ZipcodeConfig {
            flash_attention: false,
            ..ZipcodeConfig::default()
        };
        let raw = serde_json::json!({"flash_attention": null});
        assert!(should_upgrade_flash_attention(&config, Some(&raw)));
    }

    #[test]
    fn should_upgrade_flash_attention_when_explicit_false_without_gpu_layers() {
        let config = ZipcodeConfig {
            flash_attention: false,
            ..ZipcodeConfig::default()
        };
        let raw = serde_json::json!({
            "flash_attention": false,
            "gpu_layers": null
        });
        // gpu_layers is null (missing from user's config) → upgrade
        assert!(should_upgrade_flash_attention(&config, Some(&raw)));
    }

    #[test]
    fn should_upgrade_flash_attention_when_explicit_false_with_gpu_layers_set() {
        let config = ZipcodeConfig {
            flash_attention: false,
            ..ZipcodeConfig::default()
        };
        let raw = serde_json::json!({
            "flash_attention": false,
            "gpu_layers": 64
        });
        // gpu_layers is explicitly set by user → don't override their FA preference
        assert!(!should_upgrade_flash_attention(&config, Some(&raw)));
    }

    #[test]
    fn should_upgrade_flash_attention_when_explicit_true() {
        let config = ZipcodeConfig {
            flash_attention: false,
            ..ZipcodeConfig::default()
        };
        let raw = serde_json::json!({"flash_attention": true});
        // User explicitly set FA=true in JSON but config has false? Should not upgrade
        // (config.flash_attention is already handled by the first check)
        assert!(!should_upgrade_flash_attention(&config, Some(&raw)));
    }

    #[test]
    fn missing_model_message_shows_misconfigured_issue_when_flag_set() {
        let mut report = sample_report(ReadinessStatus::MissingModel);
        report.model = None;
        report.model_issue = Some("configured model path does not exist".to_string());
        report.model_issue_is_misconfigured = true;

        let msg = missing_model_message(&report);
        assert_eq!(msg, "configured model path does not exist");
    }

    #[test]
    fn missing_model_message_falls_through_to_search_paths() {
        let mut report = sample_report(ReadinessStatus::MissingModel);
        report.model = None;
        report.model_issue = None;
        report.model_issue_is_misconfigured = false;
        report.model_search = vec![
            PathBuf::from("/home/user/.zipcode/models"),
            PathBuf::from("/project/models"),
        ];

        let msg = missing_model_message(&report);
        assert!(msg.starts_with("No .gguf AI model found."));
        assert!(msg.contains("/home/user/.zipcode/models"));
        assert!(msg.contains("/project/models"));
    }

    #[test]
    fn missing_model_message_with_empty_search_paths() {
        let mut report = sample_report(ReadinessStatus::MissingModel);
        report.model = None;
        report.model_search = vec![];

        let msg = missing_model_message(&report);
        assert!(msg.starts_with("No .gguf AI model found."));
    }

    #[test]
    fn resolve_path_from_cwd_keeps_absolute_paths() {
        let path = Path::new("/absolute/path/model.gguf");
        let cwd = Path::new("/some/cwd");
        assert_eq!(
            resolve_path_from_cwd(path, cwd),
            PathBuf::from("/absolute/path/model.gguf")
        );
    }

    #[test]
    fn resolve_path_from_cwd_joins_relative_paths() {
        let path = Path::new("relative/model.gguf");
        let cwd = Path::new("/working/dir");
        assert_eq!(
            resolve_path_from_cwd(path, cwd),
            PathBuf::from("/working/dir/relative/model.gguf")
        );
    }

    #[test]
    fn resolve_path_from_cwd_resolves_tilde() {
        let path = Path::new("~/models/model.gguf");
        let cwd = Path::new("/some/cwd");
        let result = resolve_path_from_cwd(path, cwd);
        // Tilde should be expanded and result should be absolute
        assert!(result.is_absolute());
        assert!(result.to_string_lossy().contains("models/model.gguf"));
    }

    #[test]
    fn resolve_path_from_cwd_bare_filename() {
        let path = Path::new("model.gguf");
        let cwd = Path::new("/working/dir");
        assert_eq!(
            resolve_path_from_cwd(path, cwd),
            PathBuf::from("/working/dir/model.gguf")
        );
    }

    // ── validate_gguf_file tests ──────────────────────────────────

    #[test]
    fn validate_gguf_file_accepts_valid_header() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("model.gguf");
        // Write GGUF magic + some extra bytes
        let mut data = vec![0x47, 0x47, 0x55, 0x46]; // "GGUF"
        data.extend_from_slice(&[0u8; 100]); // padding
        std::fs::write(&path, &data).unwrap();
        assert!(
            validate_gguf_file(&path).is_ok(),
            "valid GGUF header should pass validation"
        );
    }

    #[test]
    fn validate_gguf_file_rejects_empty_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("empty.gguf");
        std::fs::write(&path, "").unwrap();
        let err = validate_gguf_file(&path).unwrap_err().to_string();
        assert!(
            err.contains("empty"),
            "empty file should report 'empty', got: {err}"
        );
        assert!(
            err.contains("0 bytes"),
            "empty file should mention '0 bytes', got: {err}"
        );
    }

    #[test]
    fn validate_gguf_file_rejects_wrong_magic() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("bad.gguf");
        // Write some bytes that are NOT the GGUF magic
        std::fs::write(&path, b"NOT_A_GGUF_FILE_AT_ALL!!!!").unwrap();
        let err = validate_gguf_file(&path).unwrap_err().to_string();
        assert!(
            err.contains("valid GGUF header"),
            "wrong magic should mention invalid header, got: {err}"
        );
    }

    #[test]
    fn validate_gguf_file_rejects_truncated_header() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("tiny.gguf");
        // Only 2 bytes — too small to contain a 4-byte header
        std::fs::write(&path, b"GG").unwrap();
        let err = validate_gguf_file(&path).unwrap_err().to_string();
        assert!(
            err.contains("too small"),
            "truncated file should report 'too small', got: {err}"
        );
    }

    #[test]
    fn validate_gguf_file_accepts_exact_4_byte_header() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("minimal.gguf");
        // Exactly 4 bytes: the GGUF magic — should pass (header present)
        std::fs::write(&path, [0x47, 0x47, 0x55, 0x46]).unwrap();
        assert!(
            validate_gguf_file(&path).is_ok(),
            "exact 4-byte GGUF header should pass validation"
        );
    }

    #[test]
    fn validate_gguf_file_rejects_nonexistent_file() {
        let path = Path::new("/tmp/this_file_definitely_does_not_exist_12345.gguf");
        let err = validate_gguf_file(path).unwrap_err().to_string();
        assert!(
            err.contains("Cannot stat"),
            "nonexistent file should report stat error, got: {err}"
        );
    }
}
