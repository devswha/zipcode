use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn zipcode_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zipcode"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("cli crate should have workspace parent")
        .parent()
        .expect("workspace root should exist")
        .to_path_buf()
}

fn make_temp_dir(name: &str) -> PathBuf {
    let unique = format!(
        "zipcode-smoke-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos()
    );
    let dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_executable(path: &std::path::Path, body: &str) {
    std::fs::write(path, body).expect("write executable");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(path)
            .expect("stat executable")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(path, permissions).expect("chmod executable");
    }
}

fn write_file(path: &std::path::Path, body: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create parent dir");
    }
    std::fs::write(path, body).expect("write file");
}

fn run_bash_script_with_input(
    script: &std::path::Path,
    args: &[&str],
    input: &str,
    home: &std::path::Path,
) -> Output {
    let mut child = Command::new("bash")
        .arg(script)
        .args(args)
        .env("HOME", home)
        .env("ZIPCODE_INSTALL_SKIP_SYSTEM_BIN", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bash script");

    child
        .stdin
        .as_mut()
        .expect("script stdin should exist")
        .write_all(input.as_bytes())
        .expect("write script input");

    child.wait_with_output().expect("wait for bash script")
}

#[test]
fn doctor_runs() {
    let output = zipcode_bin()
        .arg("doctor")
        .output()
        .expect("failed to run zipcode doctor");

    assert!(output.status.success(), "doctor should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("zipcode"),
        "output should mention zipcode, got: {stdout}"
    );
}

#[test]
fn doctor_accepts_global_model_flag_after_subcommand() {
    let dir = make_temp_dir("doctor-model-dir");
    let model = dir.join("test.gguf");
    std::fs::write(&model, b"gguf").expect("write fake gguf");

    let output = zipcode_bin()
        .args(["doctor", "--model"])
        .arg(&dir)
        .output()
        .expect("failed to run zipcode doctor --model <dir>");

    assert!(output.status.success(), "doctor should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("test.gguf") && stdout.contains("Status:"),
        "doctor should report models from explicit directory, got: {stdout}"
    );

    std::fs::remove_dir_all(dir).expect("cleanup temp dir");
}

#[test]
fn help_flag() {
    let output = zipcode_bin()
        .arg("--help")
        .output()
        .expect("failed to run zipcode --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Usage") || stdout.contains("usage"),
        "help should show usage info, got: {stdout}"
    );
}

#[test]
fn help_lists_explicit_power_user_flows() {
    let output = zipcode_bin()
        .arg("--help")
        .output()
        .expect("failed to run zipcode --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for command in ["repl", "prompt", "doctor", "setup"] {
        assert!(
            stdout.contains(command),
            "root help should list `{command}`, got: {stdout}"
        );
    }
}

#[test]
fn version_flag() {
    let output = zipcode_bin()
        .arg("--version")
        .output()
        .expect("failed to run zipcode --version");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("0.1.0") || stdout.contains("zipcode"),
        "version should show version number or name, got: {stdout}"
    );
}

#[test]
fn prompt_no_model_graceful_error() {
    let output = zipcode_bin()
        .args(["prompt", "hello"])
        .env("HOME", "/tmp/zipcode-test-nonexistent")
        .output()
        .expect("failed to run zipcode prompt");

    // Should NOT panic (exit code should not be signal-terminated)
    // It's OK to exit with non-zero since there's no model
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}{stderr}");

    // Should show a user-friendly error, not a panic backtrace
    assert!(
        !combined.contains("panicked at") && !combined.contains("RUST_BACKTRACE"),
        "should not panic, got: {combined}"
    );
}

#[test]
fn bare_zipcode_routes_to_setup_guidance_when_not_ready() {
    let home = make_temp_dir("startup-setup");

    let output = zipcode_bin()
        .env("HOME", &home)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run bare zipcode");

    assert!(output.status.success(), "bare zipcode should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Setup needed before zipcode can start.")
            && stdout.contains("Copy a .gguf AI model"),
        "bare zipcode should guide setup when not ready, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn bare_zipcode_routes_to_repair_guidance_when_saved_model_path_is_broken() {
    let home = make_temp_dir("startup-repair");
    write_file(
        &home.join(".zipcode/config.json"),
        r#"{
  "model_dir": "/tmp/zipcode-missing-model-dir",
  "model_file": "missing.gguf",
  "permission_mode": "workspace-write"
}"#,
    );

    let output = zipcode_bin()
        .env("HOME", &home)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run bare zipcode");

    assert!(output.status.success(), "bare zipcode should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Repair needed before zipcode can start.")
            && stdout.contains("Update the saved AI model path"),
        "bare zipcode should guide repair for broken saved paths, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn repl_no_model_graceful_error() {
    let output = zipcode_bin()
        .arg("repl")
        .env("HOME", "/tmp/zipcode-test-nonexistent")
        .output()
        .expect("failed to run zipcode repl");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}{stderr}");

    assert!(
        !combined.contains("panicked at") && !combined.contains("RUST_BACKTRACE"),
        "repl should not panic, got: {combined}"
    );
}

#[test]
fn invalid_backend_is_rejected() {
    let output = zipcode_bin()
        .args(["--backend", "bad-backend", "doctor"])
        .output()
        .expect("failed to run zipcode doctor");

    assert!(!output.status.success(), "invalid backend should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}{stderr}");

    assert!(
        combined.contains("unsupported backend"),
        "invalid backend should produce a clear error, got: {combined}"
    );
}

#[test]
fn doctor_reports_missing_server_for_gemma4_without_llama_server() {
    let home = make_temp_dir("doctor-missing-server");
    let model_dir = home.join(".zipcode/models");
    let isolated_bin = home.join("bin");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&isolated_bin).expect("create isolated bin dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"gguf").expect("write fake gguf");
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("write fake tokenizer");

    let output = zipcode_bin()
        .arg("doctor")
        .env("HOME", &home)
        .env("PATH", &isolated_bin)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor");

    assert!(output.status.success(), "doctor should still exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Status: Setup needed")
            && stdout.contains("Compatibility helper: not found"),
        "doctor should use setup-friendly helper wording, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn setup_writes_config_and_wrapper_when_smoke_skipped() {
    let home = make_temp_dir("setup-skip-smoke");
    let model_dir = home.join(".zipcode/models");
    let zipcode_bin_dir = home.join(".zipcode/bin");
    let isolated_path = home.join("empty-path");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&zipcode_bin_dir).expect("create zipcode bin dir");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"gguf").expect("write fake gguf");
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("write fake tokenizer");
    write_executable(
        &zipcode_bin_dir.join("llama-server"),
        "#!/bin/sh\necho fake llama-server\n",
    );

    let output = zipcode_bin()
        .args(["setup", "--skip-smoke"])
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode setup --skip-smoke");

    assert!(
        output.status.success(),
        "setup should succeed, got: {output:?}"
    );

    let config_path = home.join(".zipcode/config.json");
    let wrapper_path = home.join(".zipcode/bin/zipcode-local");
    let config = std::fs::read_to_string(&config_path).expect("read written config");
    assert!(
        config.contains("gemma-4-test.gguf"),
        "setup should persist the discovered model file, got: {config}"
    );
    assert!(
        config.contains("llama-server"),
        "setup should persist the discovered llama-server path, got: {config}"
    );
    assert!(wrapper_path.is_file(), "setup should create wrapper shim");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Smoke: skipped (--skip-smoke)"),
        "setup should report skipped smoke, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn doctor_stays_ready_after_setup_skip_smoke() {
    let home = make_temp_dir("setup-then-doctor");
    let model_dir = home.join(".zipcode/models");
    let zipcode_bin_dir = home.join(".zipcode/bin");
    let isolated_path = home.join("empty-path");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&zipcode_bin_dir).expect("create zipcode bin dir");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"gguf").expect("write fake gguf");
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("write fake tokenizer");
    write_executable(
        &zipcode_bin_dir.join("llama-server"),
        "#!/bin/sh\necho fake llama-server\n",
    );

    let setup = zipcode_bin()
        .args(["setup", "--skip-smoke"])
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode setup --skip-smoke");
    assert!(
        setup.status.success(),
        "setup should succeed, got: {setup:?}"
    );

    let doctor = zipcode_bin()
        .arg("doctor")
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor after setup");
    assert!(
        doctor.status.success(),
        "doctor should succeed, got: {doctor:?}"
    );

    let stdout = String::from_utf8_lossy(&doctor.stdout);
    assert!(
        stdout.contains("Status: Ready"),
        "doctor should stay Ready after setup, got: {stdout}"
    );
    assert!(
        stdout.contains(&model_dir.join("gemma-4-test.gguf").display().to_string()),
        "doctor should resolve the configured model path, got: {stdout}"
    );
    assert!(
        stdout.contains(&zipcode_bin_dir.join("llama-server").display().to_string()),
        "doctor should resolve the configured helper path, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn empty_helper_env_vars_do_not_force_repair_mode() {
    let home = make_temp_dir("empty-helper-env");

    let output = zipcode_bin()
        .env("HOME", &home)
        .env("ZIPCODE_LLAMA_SERVER_BIN", "")
        .env("LLAMA_SERVER_BIN", "")
        .output()
        .expect("failed to run bare zipcode with empty helper env vars");

    assert!(output.status.success(), "bare zipcode should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Setup needed before zipcode can start."),
        "empty helper env vars should still route to setup, got: {stdout}"
    );
    assert!(
        !stdout.contains("Repair needed before zipcode can start."),
        "empty helper env vars should not force repair mode, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn root_install_script_bootstraps_clone_users_into_ready_state() {
    let home = make_temp_dir("root-install");
    let asset_dir = make_temp_dir("root-install-assets");
    let model = asset_dir.join("gemma-4-test.gguf");
    let tokenizer = asset_dir.join("tokenizer.json");
    let helper = asset_dir.join("llama-server");
    let install_script = repo_root().join("install.sh");

    std::fs::write(&model, b"gguf").expect("write fake gguf");
    std::fs::write(&tokenizer, b"{}").expect("write fake tokenizer");
    write_executable(&helper, "#!/bin/sh\necho fake llama-server\n");

    let output = Command::new("bash")
        .arg(&install_script)
        .args(["--binary", env!("CARGO_BIN_EXE_zipcode")])
        .arg("--model")
        .arg(&model)
        .arg("--tokenizer")
        .arg(&tokenizer)
        .arg("--llama-server")
        .arg(&helper)
        .env("HOME", &home)
        .env("ZIPCODE_INSTALL_SKIP_SYSTEM_BIN", "1")
        .output()
        .expect("failed to run root install.sh");

    assert!(
        output.status.success(),
        "root install.sh should succeed, got: {output:?}"
    );

    let launcher = home.join(".local/bin/zipcode");
    assert!(
        launcher.is_file(),
        "install should create ~/.local/bin/zipcode"
    );

    let doctor = Command::new(&launcher)
        .arg("doctor")
        .env("HOME", &home)
        .output()
        .expect("failed to run installed zipcode doctor");

    assert!(
        doctor.status.success(),
        "doctor should succeed, got: {doctor:?}"
    );
    let stdout = String::from_utf8_lossy(&doctor.stdout);
    assert!(
        stdout.contains("Status: Ready"),
        "installed zipcode should report Ready, got: {stdout}"
    );
    assert!(
        stdout.contains(
            &home
                .join(".zipcode/models/gemma-4-test.gguf")
                .display()
                .to_string()
        ),
        "doctor should report installed model path, got: {stdout}"
    );
    assert!(
        stdout.contains(&home.join(".zipcode/bin/llama-server").display().to_string()),
        "doctor should report installed helper path, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
    std::fs::remove_dir_all(asset_dir).expect("cleanup asset dir");
}

#[test]
fn root_install_script_interviews_users_with_links_then_paths() {
    let home = make_temp_dir("root-install-interview");
    let asset_dir = make_temp_dir("root-install-interview-assets");
    let model = asset_dir.join("gemma-4-test.gguf");
    let tokenizer = asset_dir.join("tokenizer.json");
    let helper = asset_dir.join("llama-server");
    let install_script = repo_root().join("install.sh");

    std::fs::write(&model, b"gguf").expect("write fake gguf");
    std::fs::write(&tokenizer, b"{}").expect("write fake tokenizer");
    write_executable(&helper, "#!/bin/sh\necho fake llama-server\n");

    let scripted_input = format!(
        "1\n\n{}\n{}\n1\n{}\n",
        model.display(),
        tokenizer.display(),
        helper.display()
    );
    let output = run_bash_script_with_input(
        &install_script,
        &["--binary", env!("CARGO_BIN_EXE_zipcode")],
        &scripted_input,
        &home,
    );

    assert!(
        output.status.success(),
        "interactive install should succeed, got: {output:?}"
    );

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF"),
        "interactive install should show the model download link, got: {combined}"
    );
    assert!(
        combined.contains("Local path to the .gguf model:"),
        "interactive install should ask for the model path, got: {combined}"
    );
    assert!(
        combined.contains("Local path to tokenizer.json:"),
        "interactive install should ask for the tokenizer path, got: {combined}"
    );
    assert!(
        combined.contains("Local path to llama-server:"),
        "interactive install should ask for the helper path, got: {combined}"
    );

    let doctor = Command::new(home.join(".local/bin/zipcode"))
        .arg("doctor")
        .env("HOME", &home)
        .output()
        .expect("run installed zipcode doctor");
    assert!(
        doctor.status.success(),
        "doctor should succeed, got: {doctor:?}"
    );
    let stdout = String::from_utf8_lossy(&doctor.stdout);
    assert!(
        stdout.contains("Status: Ready"),
        "interactive install should still produce a Ready doctor state, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
    std::fs::remove_dir_all(asset_dir).expect("cleanup asset dir");
}
