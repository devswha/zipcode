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
    assert!(
        stdout.contains("--ui"),
        "help should document the fullscreen UI flag, got: {stdout}"
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
fn help_lists_ui_flag() {
    let output = zipcode_bin()
        .arg("--help")
        .output()
        .expect("failed to run zipcode --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--ui"),
        "root help should document --ui, got: {stdout}"
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
fn doctor_allows_llama_server_without_tokenizer() {
    let home = make_temp_dir("doctor-llama-server-no-tokenizer");
    let model_dir = home.join(".zipcode/models");
    let zipcode_bin_dir = home.join(".zipcode/bin");
    let isolated_path = home.join("empty-path");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&zipcode_bin_dir).expect("create helper dir");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"gguf").expect("write fake gguf");
    write_executable(
        &zipcode_bin_dir.join("llama-server"),
        "#!/bin/sh\necho fake llama-server\n",
    );

    let output = zipcode_bin()
        .args(["doctor", "--backend", "llama-server"])
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor");

    assert!(output.status.success(), "doctor should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Status: Ready") && !stdout.contains("Tokenizer:"),
        "llama-server flow should not require tokenizer, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn doctor_resolves_tilde_project_model_dir() {
    let home = make_temp_dir("doctor-tilde-home");
    let project = make_temp_dir("doctor-tilde-project");
    let model_dir = home.join(".zipcode/models");
    let zipcode_bin_dir = home.join(".zipcode/bin");
    let isolated_path = home.join("empty-path");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&zipcode_bin_dir).expect("create helper dir");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");
    std::fs::write(model_dir.join("demo.gguf"), b"gguf").expect("write fake gguf");
    write_executable(
        &zipcode_bin_dir.join("llama-server"),
        "#!/bin/sh\necho fake llama-server\n",
    );
    write_file(
        &project.join(".zipcode.json"),
        r#"{
  "model_dir": "~/.zipcode/models"
}"#,
    );

    let output = zipcode_bin()
        .args(["doctor", "--backend", "llama-server"])
        .current_dir(&project)
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor");

    assert!(output.status.success(), "doctor should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Status: Ready")
            && stdout.contains(&model_dir.join("demo.gguf").display().to_string()),
        "doctor should resolve tilde project paths via HOME, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp home");
    std::fs::remove_dir_all(project).expect("cleanup temp project");
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
        "2\n1\n\n{}\n{}\n2\n{}\n",
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

#[test]
fn root_install_script_can_download_in_terminal_then_finish_setup() {
    let home = make_temp_dir("root-install-terminal-download");
    let asset_dir = make_temp_dir("root-install-terminal-download-assets");
    let fake_downloader = asset_dir.join("fake-download-model.sh");
    let fake_helper_builder = asset_dir.join("fake-build-llama-server.sh");
    let install_script = repo_root().join("install.sh");

    write_executable(
        &fake_downloader,
        r#"#!/bin/sh
set -eu
dir=""
preset="e2b"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dir)
      dir="$2"
      shift
      ;;
    --preset)
      preset="$2"
      shift
      ;;
  esac
  shift
done
mkdir -p "$dir"
if [ "$preset" = "31b" ]; then
  printf 'gguf' > "$dir/gemma-4-31b-it-q8_0.gguf"
  printf 'gguf' > "$dir/gemma-4-31b-it-f16.gguf"
else
  printf 'gguf' > "$dir/gemma-4-e2b-it-q8_0.gguf"
fi
printf 'gguf' > "$dir/qwen2.5-0.5b-instruct-q4_k_m.gguf"
printf 'gguf' > "$dir/mmproj-gemma-4-31b-it-f16.gguf"
printf '{}' > "$dir/tokenizer.json"
echo "fake downloader wrote $preset assets to $dir"
"#,
    );
    write_executable(
        &fake_helper_builder,
        r#"#!/bin/sh
set -eu
install_dir="${1:-${HOME}/.zipcode}"
mkdir -p "$install_dir/bin"
cat > "$install_dir/bin/llama-server" <<'EOF'
#!/bin/sh
echo fake llama-server
EOF
chmod +x "$install_dir/bin/llama-server"
echo "fake helper builder installed llama-server into $install_dir/bin/llama-server"
"#,
    );

    let scripted_input = "1\n2\n1\n".to_string();
    let output = Command::new("bash")
        .arg(&install_script)
        .args(["--binary", env!("CARGO_BIN_EXE_zipcode")])
        .env("HOME", &home)
        .env("ZIPCODE_INSTALL_SKIP_SYSTEM_BIN", "1")
        .env("ZIPCODE_DOWNLOAD_MODEL_SCRIPT", &fake_downloader)
        .env("ZIPCODE_LLAMA_SERVER_BUILD_SCRIPT", &fake_helper_builder)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            child
                .stdin
                .as_mut()
                .expect("stdin")
                .write_all(scripted_input.as_bytes())?;
            child.wait_with_output()
        })
        .expect("run interactive install with fake downloader");

    assert!(
        output.status.success(),
        "terminal download install should succeed, got: {output:?}"
    );

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("Downloading in this terminal now:"),
        "install should advertise the terminal-download path, got: {combined}"
    );
    assert!(
        combined.contains("Gemma 4 31B IT"),
        "install should show the selected 31B preset, got: {combined}"
    );
    assert!(
        combined.contains("fake downloader wrote 31b assets"),
        "install should run the configured downloader, got: {combined}"
    );
    assert!(
        combined.contains("fake helper builder installed llama-server"),
        "install should run the configured helper builder, got: {combined}"
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
        stdout.contains("Status: Ready") && stdout.contains("gemma-4-31b-it-q8_0.gguf"),
        "terminal download flow should leave a Ready install, got: {stdout}"
    );

    let bare = Command::new(home.join(".local/bin/zipcode"))
        .env("HOME", &home)
        .output()
        .expect("run installed zipcode");
    let bare_stdout = String::from_utf8_lossy(&bare.stdout);
    assert!(
        bare_stdout.contains("type /help")
            && !bare_stdout.contains("Setup needed")
            && !bare_stdout.contains("Repair needed"),
        "bare installed zipcode should reach the REPL path once ready, got: {bare:?}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
    std::fs::remove_dir_all(asset_dir).expect("cleanup asset dir");
}

#[test]
fn root_install_script_handles_multiple_existing_models_and_still_reaches_ready_setup() {
    let home = make_temp_dir("root-install-multi-existing");
    let model_dir = home.join(".zipcode/models");
    let asset_dir = make_temp_dir("root-install-multi-existing-assets");
    let fake_helper_builder = asset_dir.join("fake-build-llama-server.sh");
    let install_script = repo_root().join("install.sh");

    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::write(model_dir.join("gemma-4-e2b-it-q8_0.gguf"), b"gguf").expect("write model 1");
    std::fs::write(model_dir.join("gemma-4-e2b-it-f16.gguf"), b"gguf").expect("write model 2");
    std::fs::write(model_dir.join("mmproj-gemma-4-e2b-it-f16.gguf"), b"gguf")
        .expect("write mmproj");
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("write tokenizer");

    write_executable(
        &fake_helper_builder,
        r#"#!/bin/sh
set -eu
install_dir="${1:-${HOME}/.zipcode}"
mkdir -p "$install_dir/bin"
cat > "$install_dir/bin/llama-server" <<'EOF'
#!/bin/sh
echo fake llama-server
EOF
chmod +x "$install_dir/bin/llama-server"
echo "fake helper builder installed llama-server into $install_dir/bin/llama-server"
"#,
    );

    let scripted_input = "3\n\n\n1\n1\n".to_string();
    let output = Command::new("bash")
        .arg(&install_script)
        .args(["--binary", env!("CARGO_BIN_EXE_zipcode")])
        .env("HOME", &home)
        .env("ZIPCODE_INSTALL_SKIP_SYSTEM_BIN", "1")
        .env("ZIPCODE_LLAMA_SERVER_BUILD_SCRIPT", &fake_helper_builder)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            child
                .stdin
                .as_mut()
                .expect("stdin")
                .write_all(scripted_input.as_bytes())?;
            child.wait_with_output()
        })
        .expect("run install with multiple existing models");

    assert!(
        output.status.success(),
        "install should succeed, got: {output:?}"
    );

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("Multiple AI models are available.")
            && combined.contains("fake helper builder installed llama-server")
            && combined.contains("Status:         Ready"),
        "installer should choose a model and complete setup, got: {combined}"
    );
    assert!(
        !combined.contains("Skipping zipcode setup for now"),
        "installer should not skip setup when existing models are selectable, got: {combined}"
    );
}

#[test]
fn fullscreen_repl_e2e_accepts_status_and_quit() {
    let python = if Command::new("python3").arg("--version").output().is_ok() {
        "python3"
    } else {
        "python"
    };

    let home = make_temp_dir("fullscreen-e2e-home");
    let model_dir = home.join(".zipcode/models");
    let helper_path = home.join(".zipcode/bin/fake-llama-server");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(helper_path.parent().expect("helper parent"))
        .expect("create helper dir");
    let model_path = model_dir.join("fake.gguf");
    std::fs::write(&model_path, b"gguf").expect("write fake model");
    write_executable(
        &helper_path,
        r#"#!/usr/bin/python3
import signal
import socket
import sys

args = sys.argv[1:]
port = int(args[args.index("--port") + 1]) if "--port" in args else 8080
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", port))
server.listen()

def shutdown(*_args):
    server.close()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

while True:
    conn, _ = server.accept()
    request = conn.recv(4096)
    if b"GET /health " in request:
        body = b'{"status":"ok"}'
    else:
        body = b'ok'
    response = (
        b"HTTP/1.1 200 OK\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"Connection: close\r\n\r\n"
        + body
    )
    conn.sendall(response)
    conn.close()
"#,
    );

    let script = format!(
        r#"
import os, pty, select, subprocess, sys, time
cmd = [{cmd:?}, "repl", "--ui", "fullscreen", "--backend", "llama-server", "--model", {model:?}]
env = os.environ.copy()
env["ZIPCODE_LLAMA_SERVER_BIN"] = {helper:?}
env["ZIPCODE_TUI_AUTOMATION_SCRIPT"] = "/status\n/quit\n"
env["HOME"] = {home:?}
master, slave = pty.openpty()
proc = subprocess.Popen(
    cmd,
    stdin=slave,
    stdout=slave,
    stderr=slave,
    cwd={cwd:?},
    env=env,
    text=False,
)
os.close(slave)
deadline = time.time() + 20
chunks = []
while time.time() < deadline:
    if proc.poll() is not None:
        break
    try:
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            chunk = os.read(master, 8192)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError:
        break
if proc.poll() is None:
    proc.terminate()
    proc.wait(timeout=5)
while True:
    try:
        chunk = os.read(master, 8192)
        if not chunk:
            break
        chunks.append(chunk)
    except OSError:
        break
output = b"".join(chunks).decode("utf-8", "replace")
print(output)
sys.exit(proc.returncode or 0)
"#,
        cmd = env!("CARGO_BIN_EXE_zipcode"),
        model = model_path.display().to_string(),
        helper = helper_path.display().to_string(),
        home = home.display().to_string(),
        cwd = repo_root().display().to_string(),
    );

    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .output()
        .expect("failed to run fullscreen TUI E2E harness");

    assert!(
        output.status.success(),
        "fullscreen TUI harness should exit 0, got: {output:?}"
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("Session Status")
            && combined.contains("Session ID:")
            && !combined.contains("Unknown command"),
        "fullscreen e2e should process /status and /quit cleanly, got: {combined}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}
