use std::process::Command;

fn zipcode_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zipcode"))
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
