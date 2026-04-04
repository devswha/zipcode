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
        stdout.contains("Model found:") && stdout.contains("test.gguf"),
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
