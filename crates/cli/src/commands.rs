use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use zipcode_inference::Backend;
use zipcode_runtime::config::{find_project_root, global_config_path};
use zipcode_runtime::ZipcodeConfig;

use crate::repl::{is_probably_gemma4_model, resolve_model_path, run_interactive, run_oneshot};

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
    fn headline(self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::NeedsSetup => "Setup needed",
            Self::NeedsRepair => "Repair needed",
        }
    }

    fn startup_title(self) -> &'static str {
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
}

#[derive(Debug, Clone)]
struct LlamaServerDiscovery {
    path: Option<PathBuf>,
    issue: Option<String>,
    issue_is_misconfigured: bool,
}

/// Default zipcode entrypoint: start the REPL when ready, otherwise guide setup/repair.
pub fn run_default(
    model_path: Option<&Path>,
    permission_mode: Option<&str>,
    backend_str: &str,
) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let config_load = load_config_with_warning(&cwd);
    let backend = Backend::parse(backend_str)?;
    let report = build_readiness_report(&cwd, &config_load.config, model_path, backend);
    let readiness = classify_user_readiness(&report, config_load.warning.as_deref());

    match readiness {
        UserReadiness::Ready => run_interactive(model_path, permission_mode, backend_str),
        state => {
            print_startup_guidance(state, &report, config_load.warning.as_deref());
            Ok(())
        }
    }
}

/// Run the doctor command: check version, GPU support, local AI files, and fallback readiness.
pub fn doctor(model_path: Option<&Path>, backend_str: &str) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let config_load = load_config_with_warning(&cwd);
    let backend = Backend::parse(backend_str)?;
    let report = build_readiness_report(&cwd, &config_load.config, model_path, backend);
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
    for line in next_steps(readiness, &report, config_load.warning.as_deref()) {
        println!("  • {line}");
    }

    println!();
    println!(
        "Tip: run `zipcode` with no arguments. It will start the chat REPL when ready and show setup help otherwise."
    );
    Ok(())
}

/// Discover local prerequisites, write global config + launcher, and optionally run a smoke prompt.
pub fn setup(model_path: Option<&Path>, backend_str: &str, skip_smoke: bool) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let config_load = load_config_with_warning(&cwd);
    let backend = Backend::parse(backend_str)?;
    let report = build_readiness_report(&cwd, &config_load.config, model_path, backend);
    let readiness = classify_user_readiness(&report, None);

    let model = report
        .model
        .clone()
        .ok_or_else(|| anyhow!(missing_model_message(&report)))?;
    let mut global_config = ZipcodeConfig::load_global().unwrap_or_default();
    global_config.model_dir = model.parent().unwrap_or(Path::new(".")).to_path_buf();
    global_config.model_file = model
        .file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned);
    global_config.llama_server_bin = report.llama_server_bin.clone();

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
    println!();

    for line in next_steps(readiness, &report, None) {
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
        backend_name(report.backend),
    )?;
    println!("Smoke: PASS");
    Ok(())
}

/// Check if CUDA is available by looking for libcuda.so or CUDA_PATH env var.
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

fn load_config_with_warning(cwd: &Path) -> ConfigLoad {
    match ZipcodeConfig::load(cwd) {
        Ok(config) => ConfigLoad {
            config,
            warning: None,
        },
        Err(error) => ConfigLoad {
            config: ZipcodeConfig::default(),
            warning: Some(error.to_string()),
        },
    }
}

fn build_readiness_report(
    cwd: &Path,
    config: &ZipcodeConfig,
    explicit_model_path: Option<&Path>,
    backend: Backend,
) -> ReadinessReport {
    let project_root = find_project_root(cwd);
    let model_resolution = resolve_model_path(explicit_model_path, config, cwd, &project_root);
    let (model, model_issue) = match model_resolution {
        Ok(path) => (Some(path), None),
        Err(error) => (None, Some(error.to_string())),
    };
    let model_search = model_search_locations(explicit_model_path, config, cwd, &project_root);
    let model_issue_is_misconfigured = explicit_model_path.is_some() || config.model_file.is_some();

    let tokenizer = model
        .as_deref()
        .map(tokenizer_path_for_model)
        .filter(|path| path.is_file());
    let tokenizer_issue = model.as_deref().and_then(|model_path| {
        let tokenizer_path = tokenizer_path_for_model(model_path);
        (!tokenizer_path.is_file())
            .then(|| format!("tokenizer.json is missing next to {}", model_path.display()))
    });

    let llama_server = discover_llama_server_bin(config);
    let status = classify_readiness(
        backend,
        model.as_deref(),
        tokenizer.as_deref(),
        llama_server.path.as_deref(),
    );

    ReadinessReport {
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
        llama_server_issue_is_misconfigured: llama_server.issue_is_misconfigured,
    }
}

fn classify_user_readiness(
    report: &ReadinessReport,
    config_warning: Option<&str>,
) -> UserReadiness {
    if config_warning.is_some()
        || (report.model_issue.is_some() && report.model_issue_is_misconfigured)
        || (report.llama_server_issue.is_some() && report.llama_server_issue_is_misconfigured)
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

    if tokenizer.is_none() {
        return ReadinessStatus::MissingTokenizer;
    }

    if matches!(backend, Backend::LlamaServer) && llama_server_bin.is_none() {
        return ReadinessStatus::MissingServer;
    }

    if matches!(backend, Backend::LlamaCpp) && is_probably_gemma4_model(model) {
        if llama_server_bin.is_some() {
            return ReadinessStatus::FallbackRequired;
        }
        return ReadinessStatus::MissingServer;
    }

    ReadinessStatus::NativeOk
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::LlamaCpp => "llama-cpp",
        Backend::LlamaServer => "llama-server",
        Backend::Candle => "candle",
    }
}

fn friendly_engine_name(backend: Backend) -> &'static str {
    match backend {
        Backend::LlamaCpp => "local engine (llama-cpp)",
        Backend::LlamaServer => "compatibility helper (llama-server)",
        Backend::Candle => "pure Rust engine (candle)",
    }
}

fn format_model(path: &Path) -> String {
    let size = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
    format!("{} ({size_gb:.1} GB)", path.display())
}

fn print_startup_guidance(
    readiness: UserReadiness,
    report: &ReadinessReport,
    config_warning: Option<&str>,
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
    for (index, step) in next_steps(readiness, report, config_warning)
        .iter()
        .enumerate()
    {
        println!("  {}. {step}", index + 1);
    }
}

fn startup_details(report: &ReadinessReport, config_warning: Option<&str>) -> Vec<String> {
    let mut lines = Vec::new();

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

    if let Some(issue) = &report.tokenizer_issue {
        lines.push(issue.clone());
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

    if report.model.is_some() {
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
    } else {
        lines.push("  ℹ️ Compatibility helper: not needed for this setup".to_string());
    }

    lines
}

fn next_steps(
    readiness: UserReadiness,
    report: &ReadinessReport,
    config_warning: Option<&str>,
) -> Vec<String> {
    match readiness {
        UserReadiness::Ready => {
            let mut lines = vec!["Run `zipcode` to start the chat REPL.".to_string()];
            if matches!(report.status, ReadinessStatus::FallbackRequired) {
                let helper = report
                    .llama_server_bin
                    .as_deref()
                    .unwrap_or(Path::new("llama-server"));
                lines.push(format!(
                    "zipcode will use the compatibility helper at {} automatically for this model.",
                    helper.display()
                ));
            }
            lines
        }
        UserReadiness::NeedsSetup => setup_steps(report),
        UserReadiness::NeedsRepair => repair_steps(report, config_warning),
    }
}

fn setup_steps(report: &ReadinessReport) -> Vec<String> {
    match report.status {
        ReadinessStatus::MissingModel => vec![
            format!(
                "Copy a .gguf AI model into {} or rerun with `--model <PATH>`.",
                model_location_hint(report)
            ),
            "Copy the matching tokenizer.json next to that model file.".to_string(),
            "Run `zipcode setup --skip-smoke`, then run `zipcode` again.".to_string(),
        ],
        ReadinessStatus::MissingTokenizer => vec![
            format!(
                "Copy tokenizer.json next to {}.",
                report
                    .model
                    .as_deref()
                    .unwrap_or(Path::new("your-model.gguf"))
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

fn repair_steps(report: &ReadinessReport, config_warning: Option<&str>) -> Vec<String> {
    let mut lines = Vec::new();

    if let Some(warning) = config_warning {
        lines.push(format!(
            "Fix or replace {} (current error: {warning}).",
            global_config_path().display()
        ));
    }

    if report.model_issue_is_misconfigured {
        if let Some(issue) = &report.model_issue {
            lines.push(format!("Update the saved AI model path ({issue})."));
        }
    }

    if let Some(issue) = &report.tokenizer_issue {
        lines.push(format!("Restore the missing tokenizer file ({issue})."));
    }

    if report.llama_server_issue_is_misconfigured {
        if let Some(issue) = &report.llama_server_issue {
            lines.push(format!(
                "Update the saved compatibility helper path ({issue})."
            ));
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

fn model_location_hint(report: &ReadinessReport) -> String {
    report
        .model_search
        .first()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| {
            dirs::home_dir()
                .map(|home| home.join(".zipcode/models").display().to_string())
                .unwrap_or_else(|| "~/.zipcode/models".to_string())
        })
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
        .unwrap_or(Path::new("."))
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
        let configured = PathBuf::from(model_file);
        let resolved = if configured.is_absolute() {
            configured
        } else if configured.parent().is_some() {
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
    if config.model_dir.is_absolute() {
        config.model_dir.clone()
    } else {
        project_root.join(&config.model_dir)
    }
}

fn resolve_path_from_cwd(path: &Path, cwd: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
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

fn discover_llama_server_bin(config: &ZipcodeConfig) -> LlamaServerDiscovery {
    if let Some(path) = &config.llama_server_bin {
        if path.is_file() {
            return LlamaServerDiscovery {
                path: Some(path.clone()),
                issue: None,
                issue_is_misconfigured: false,
            };
        }

        return LlamaServerDiscovery {
            path: None,
            issue: Some(format!("saved helper path not found at {}", path.display())),
            issue_is_misconfigured: true,
        };
    }

    for key in ["ZIPCODE_LLAMA_SERVER_BIN", "LLAMA_SERVER_BIN"] {
        if let Ok(value) = std::env::var(key) {
            let path = PathBuf::from(&value);
            if path.is_file() {
                return LlamaServerDiscovery {
                    path: Some(path),
                    issue: None,
                    issue_is_misconfigured: false,
                };
            }

            return LlamaServerDiscovery {
                path: None,
                issue: Some(format!(
                    "{key} points to {} but that file was not found",
                    path.display()
                )),
                issue_is_misconfigured: true,
            };
        }
    }

    if let Some(home) = dirs::home_dir() {
        let bundled = home.join(".zipcode/bin/llama-server");
        if bundled.is_file() {
            return LlamaServerDiscovery {
                path: Some(bundled),
                issue: None,
                issue_is_misconfigured: false,
            };
        }
    }

    LlamaServerDiscovery {
        path: which_in_path("llama-server"),
        issue: None,
        issue_is_misconfigured: false,
    }
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
            Some(Path::new("/tmp/gemma-4-test.gguf")),
            None,
            Some(Path::new("/tmp/llama-server")),
        );
        assert_eq!(status, ReadinessStatus::MissingTokenizer);
    }

    #[test]
    fn classify_missing_server_for_gemma4_without_server() {
        let status = classify_readiness(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/gemma-4-test.gguf")),
            Some(Path::new("/tmp/tokenizer.json")),
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
}
