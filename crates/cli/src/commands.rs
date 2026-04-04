use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use zipcode_inference::Backend;
use zipcode_runtime::config::find_project_root;
use zipcode_runtime::ZipcodeConfig;

use crate::repl::{is_probably_gemma4_model, resolve_model_path, run_oneshot};

const SETUP_WRAPPER_NAME: &str = "zipcode-local";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadinessStatus {
    NativeOk,
    FallbackRequired,
    MissingServer,
    MissingModel,
}

impl ReadinessStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::NativeOk => "native-ok",
            Self::FallbackRequired => "fallback-required",
            Self::MissingServer => "missing-server",
            Self::MissingModel => "missing-model",
        }
    }
}

#[derive(Debug, Clone)]
struct ReadinessReport {
    status: ReadinessStatus,
    backend: Backend,
    model: Option<PathBuf>,
    model_search: Vec<PathBuf>,
    llama_server_bin: Option<PathBuf>,
}

/// Run the doctor command: check binary version, CUDA availability, model files, and fallback readiness.
pub fn doctor(model_path: Option<&Path>, backend_str: &str) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let config = load_config_with_warning(&cwd);
    let backend = Backend::parse(backend_str)?;
    let report = build_readiness_report(&cwd, &config, model_path, backend);

    println!("zipcode doctor\n");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!("Backend: {}", backend_name(report.backend));
    println!();

    if check_cuda() {
        println!("  ✅ CUDA available");
    } else {
        println!("  ❌ CUDA not available (CPU inference only)");
    }

    match &report.model {
        Some(path) => println!("  ✅ Model:    {}", format_model(path)),
        None => {
            println!("  ❌ Model:    not found");
            println!(
                "     Searched: {}",
                format_search_paths(&report.model_search)
            );
        }
    }

    match &report.llama_server_bin {
        Some(path) => println!("  ✅ Server:   {}", path.display()),
        None => println!("  ❌ Server:   llama-server not found"),
    }

    println!("  ℹ️  Status:   {}", report.status.as_str());
    for line in remediation_lines(&report) {
        println!("     {line}");
    }

    println!();
    println!("Done.");
    Ok(())
}

/// Discover a model + llama-server, persist global config, create a helper wrapper, and optionally run a smoke prompt.
pub fn setup(model_path: Option<&Path>, backend_str: &str, skip_smoke: bool) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let config = load_config_with_warning(&cwd);
    let backend = Backend::parse(backend_str)?;
    let report = build_readiness_report(&cwd, &config, model_path, backend);

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
    println!("Config:  {}", config_path.display());
    println!("Wrapper: {}", wrapper_path.display());
    println!("Model:   {}", model.display());
    println!("Backend: {}", backend_name(report.backend));
    println!("Status:  {}", report.status.as_str());
    if let Some(path) = &report.llama_server_bin {
        println!("Server:  {}", path.display());
    }
    println!();

    if matches!(report.status, ReadinessStatus::MissingServer) {
        for line in remediation_lines(&report) {
            println!("{line}");
        }
        anyhow::bail!(
            "setup incomplete: Gemma 4 needs llama-server fallback before zipcode can run prompts"
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

fn load_config_with_warning(cwd: &Path) -> ZipcodeConfig {
    match ZipcodeConfig::load(cwd) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("\x1b[33mWarning: Failed to load config: {error}\x1b[0m");
            ZipcodeConfig::default()
        }
    }
}

fn build_readiness_report(
    cwd: &Path,
    config: &ZipcodeConfig,
    explicit_model_path: Option<&Path>,
    backend: Backend,
) -> ReadinessReport {
    let project_root = find_project_root(cwd);
    let model = resolve_model_path(explicit_model_path, config, cwd, &project_root).ok();
    let model_search = model_search_locations(explicit_model_path, config, cwd, &project_root);
    let llama_server_bin = discover_llama_server_bin(config);
    let status = classify_readiness(backend, model.as_deref(), llama_server_bin.as_deref());

    ReadinessReport {
        status,
        backend,
        model,
        model_search,
        llama_server_bin,
    }
}

fn classify_readiness(
    backend: Backend,
    model: Option<&Path>,
    llama_server_bin: Option<&Path>,
) -> ReadinessStatus {
    let Some(model) = model else {
        return ReadinessStatus::MissingModel;
    };

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

fn format_model(path: &Path) -> String {
    let size = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
    format!("{} ({size_gb:.1} GB)", path.display())
}

fn remediation_lines(report: &ReadinessReport) -> Vec<String> {
    match report.status {
        ReadinessStatus::NativeOk => vec![
            "Ready: the selected backend can use the discovered model without fallback."
                .to_string(),
        ],
        ReadinessStatus::FallbackRequired => vec![format!(
            "Ready with fallback: Gemma 4 will use llama-server at {}.",
            report
                .llama_server_bin
                .as_deref()
                .unwrap_or(Path::new("llama-server"))
                .display()
        )],
        ReadinessStatus::MissingServer => vec![
            "Gemma 4 needs a recent llama-server fallback on this build.".to_string(),
            missing_server_hint(),
        ],
        ReadinessStatus::MissingModel => vec![
            format!("Searched: {}", format_search_paths(&report.model_search)),
            "Copy a .gguf model into ~/.zipcode/models or rerun with --model <PATH>.".to_string(),
        ],
    }
}

fn missing_model_message(report: &ReadinessReport) -> String {
    format!(
        "No .gguf model found. Searched: {}",
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
            "Run: \"{}\" /path/to/llama-server \"{}\"",
            helper.display(),
            install_dir.display()
        )
    } else {
        "Install a recent llama-server and set ZIPCODE_LLAMA_SERVER_BIN=/path/to/llama-server."
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

fn discover_llama_server_bin(config: &ZipcodeConfig) -> Option<PathBuf> {
    if let Some(path) = config
        .llama_server_bin
        .as_ref()
        .filter(|path| path.is_file())
    {
        return Some(path.clone());
    }

    for key in ["ZIPCODE_LLAMA_SERVER_BIN", "LLAMA_SERVER_BIN"] {
        if let Some(path) = env_path(key) {
            return Some(path);
        }
    }

    if let Some(home) = dirs::home_dir() {
        let bundled = home.join(".zipcode/bin/llama-server");
        if bundled.is_file() {
            return Some(bundled);
        }
    }

    which_in_path("llama-server")
}

fn env_path(key: &str) -> Option<PathBuf> {
    let value = std::env::var(key).ok()?;
    let path = PathBuf::from(value);
    path.is_file().then_some(path)
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

    #[test]
    fn classify_missing_model_when_no_model_is_found() {
        let status = classify_readiness(Backend::LlamaCpp, None, None);
        assert_eq!(status, ReadinessStatus::MissingModel);
    }

    #[test]
    fn classify_missing_server_for_gemma4_without_server() {
        let status = classify_readiness(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/gemma-4-test.gguf")),
            None,
        );
        assert_eq!(status, ReadinessStatus::MissingServer);
    }

    #[test]
    fn classify_fallback_required_for_gemma4_with_server() {
        let status = classify_readiness(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/gemma-4-test.gguf")),
            Some(Path::new("/tmp/llama-server")),
        );
        assert_eq!(status, ReadinessStatus::FallbackRequired);
    }

    #[test]
    fn classify_native_ok_for_non_gemma4_model() {
        let status = classify_readiness(
            Backend::LlamaCpp,
            Some(Path::new("/tmp/codeqwen.gguf")),
            None,
        );
        assert_eq!(status, ReadinessStatus::NativeOk);
    }
}
