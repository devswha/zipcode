use std::path::PathBuf;
use std::process::Command;

fn zipcode_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zipcode"))
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
        stdout.contains("Status:   missing-server"),
        "doctor should report missing-server, got: {stdout}"
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
