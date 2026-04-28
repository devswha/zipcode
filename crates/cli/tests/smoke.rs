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

fn prepend_path(dir: &std::path::Path) -> String {
    std::env::var_os("PATH").map_or_else(
        || dir.display().to_string(),
        |existing| {
            let mut paths = vec![dir.to_path_buf()];
            paths.extend(std::env::split_paths(&existing));
            std::env::join_paths(paths)
                .expect("join PATH")
                .to_string_lossy()
                .into_owned()
        },
    )
}

fn write_fake_git(path: &std::path::Path) {
    write_executable(
        path,
        r#"#!/usr/bin/python3
import os
import pathlib
import sys

args = sys.argv[1:]
log_path = pathlib.Path(os.environ["ZIPCODE_FAKE_GIT_LOG"])
with log_path.open("a", encoding="utf-8") as log:
    log.write(" ".join(args) + "\n")

repo = os.environ["ZIPCODE_FAKE_GIT_REPO"]

if args == ["rev-parse", "--show-toplevel"]:
    print(repo)
    sys.exit(0)
if args == ["branch", "--show-current"]:
    print("main")
    sys.exit(0)
if args == ["status", "--short"]:
    print(" M src/lib.rs")
    sys.exit(0)
if args == ["rev-parse", "--short", "HEAD"]:
    print("abc123")
    sys.exit(0)
if args == ["fetch", "origin"]:
    print("simulated fetch failure", file=sys.stderr)
    sys.exit(2)

print(f"unexpected fake git invocation: {args}", file=sys.stderr)
sys.exit(99)
"#,
    );
}

fn write_fake_llama_server(path: &std::path::Path) {
    write_executable(
        path,
        r#"#!/usr/bin/python3
import signal
import socket
import sys

args = sys.argv[1:]

if "--help" in args:
    print("fake llama-server help")
    sys.exit(0)

if "--list-devices" in args:
    print("Available devices:\n  CUDA0")
    sys.exit(0)

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
}

fn write_cpu_only_llama_server(path: &std::path::Path) {
    write_executable(
        path,
        r#"#!/usr/bin/python3
import signal
import socket
import sys

args = sys.argv[1:]

if "--help" in args:
    print("fake llama-server help")
    sys.exit(0)

if "--list-devices" in args:
    print("Available devices:\n  CPU")
    sys.exit(0)

if "-ngl" in args:
    layers = args[args.index("-ngl") + 1]
    if layers not in ("0", "-1"):
        print(f"error: GPU offload requested with -ngl {layers}, but this helper only supports CPU", file=sys.stderr)
        sys.exit(1)

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
}

fn write_hanging_list_devices_llama_server(path: &std::path::Path) {
    write_executable(
        path,
        r#"#!/usr/bin/python3
import signal
import socket
import sys
import time

args = sys.argv[1:]

if "--help" in args:
    print("fake llama-server help")
    sys.exit(0)

if "--list-devices" in args:
    signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_args: sys.exit(0))
    time.sleep(30)
    sys.exit(0)

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
}

fn run_zipcode_in_pty(
    args: &[String],
    home: &std::path::Path,
    cwd: &std::path::Path,
    input_script: Option<&str>,
    automation_script: Option<&str>,
    extra_env: &[(&str, String)],
) -> Output {
    let python = if Command::new("python3").arg("--version").output().is_ok() {
        "python3"
    } else {
        "python"
    };

    let rendered_args = args
        .iter()
        .map(|arg| format!("{arg:?}"))
        .collect::<Vec<_>>()
        .join(", ");

    let env_lines: String = extra_env
        .iter()
        .fold(String::new(), |mut acc, (key, value)| {
            use std::fmt::Write;
            let _ = writeln!(acc, "env[{key:?}] = {value:?}");
            acc
        });
    let automation_script =
        automation_script.map_or_else(|| "None".to_string(), |script| format!("{script:?}"));
    let input_script =
        input_script.map_or_else(|| "None".to_string(), |script| format!("{script:?}"));
    let has_automation_script = if automation_script == "None" {
        "False"
    } else {
        "True"
    };
    let has_input_script = if input_script == "None" {
        "False"
    } else {
        "True"
    };

    let script = format!(
        r#"
import fcntl, os, pty, select, struct, subprocess, sys, termios, time
cmd = [{cmd:?}, {rendered_args}]
env = os.environ.copy()
env["HOME"] = {home:?}
{env_lines}if {has_automation_script}:
    env["ZIPCODE_TUI_AUTOMATION_SCRIPT"] = {automation_script}
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
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
if {has_input_script}:
    time.sleep(0.3)
    for line in {input_script}.splitlines(True):
        os.write(master, line.encode("utf-8"))
        time.sleep(0.5)
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
        rendered_args = rendered_args,
        home = home.display().to_string(),
        cwd = cwd.display().to_string(),
        env_lines = env_lines,
        automation_script = automation_script,
        input_script = input_script,
        has_automation_script = has_automation_script,
        has_input_script = has_input_script,
    );

    Command::new(python)
        .arg("-c")
        .arg(script)
        .output()
        .expect("failed to run zipcode PTY harness")
}

fn run_plain_zipcode_with_input(
    args: &[String],
    home: &std::path::Path,
    cwd: &std::path::Path,
    input_script: &str,
    extra_env: &[(&str, String)],
) -> Output {
    run_zipcode_in_pty(args, home, cwd, Some(input_script), None, extra_env)
}

fn run_fullscreen_zipcode_with_script(
    args: &[String],
    home: &std::path::Path,
    cwd: &std::path::Path,
    automation_script: &str,
    extra_env: &[(&str, String)],
) -> Output {
    run_zipcode_in_pty(args, home, cwd, None, Some(automation_script), extra_env)
}

fn setup_repl_fixture(name: &str) -> (PathBuf, PathBuf, PathBuf) {
    let home = make_temp_dir(name);
    let model_dir = home.join(".zipcode/models");
    let helper_path = home.join(".zipcode/bin/fake-llama-server");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(helper_path.parent().expect("helper parent"))
        .expect("create helper dir");
    let model_path = model_dir.join("fake.gguf");
    std::fs::write(&model_path, b"GGUF").expect("write fake model");
    write_fake_llama_server(&helper_path);
    (home, model_path, helper_path)
}

fn parse_cleared_session_id(output: &str) -> Option<String> {
    output
        .lines()
        .find_map(|line| {
            line.split("Conversation cleared. New session started: ")
                .nth(1)
                .map(str::trim)
        })
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
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
    // Isolate from the developer's real ~/.zipcode/config.json so the
    // assertion holds regardless of whether the dev box has a working
    // local model. Without this, the test fails on dogfood machines
    // because doctor correctly reports Ready and exits 0.
    let home = make_temp_dir("doctor-runs-isolated-home");
    let output = zipcode_bin()
        .arg("doctor")
        .env("HOME", &home)
        .env(
            "ZIPCODE_GLOBAL_CONFIG",
            home.join(".zipcode/absent-config.json"),
        )
        .env_remove("ZIPCODE_LLAMA_SERVER_URL")
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor");

    assert!(
        !output.status.success(),
        "doctor should exit non-zero when not ready, got status {:?} stdout {:?} stderr {:?}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn doctor_accepts_global_model_flag_after_subcommand() {
    let dir = make_temp_dir("doctor-model-dir");
    let model = dir.join("test.gguf");
    std::fs::write(&model, b"GGUF").expect("write fake gguf");

    let output = zipcode_bin()
        .args(["doctor", "--model"])
        .arg(&dir)
        .output()
        .expect("failed to run zipcode doctor --model <dir>");

    assert!(
        !output.status.success(),
        "doctor should exit non-zero when model has no helper"
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
    for command in ["repl", "prompt", "doctor", "setup", "update"] {
        assert!(
            stdout.contains(command),
            "root help should list `{command}`, got: {stdout}"
        );
    }
}

#[test]
fn update_check_reports_dirty_tree_before_fetch_failure() {
    let repo = make_temp_dir("update-check-dirty");
    let fake_bin = repo.join("fake-bin");
    std::fs::create_dir_all(&fake_bin).expect("create fake bin dir");
    let git_log = repo.join("git.log");
    write_fake_git(&fake_bin.join("git"));

    let output = zipcode_bin()
        .args(["update", "--check"])
        .current_dir(&repo)
        .env("PATH", prepend_path(&fake_bin))
        .env("ZIPCODE_FAKE_GIT_REPO", &repo)
        .env("ZIPCODE_FAKE_GIT_LOG", &git_log)
        .output()
        .expect("failed to run zipcode update --check");

    assert!(
        output.status.success(),
        "update --check should report the dirty tree instead of failing the fetch: {output:?}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let git_log = std::fs::read_to_string(&git_log).expect("read fake git log");
    assert!(
        stdout.contains("Working tree: dirty")
            && stdout.contains("Update check: blocked by local modifications."),
        "expected dirty-tree blocker output, got stdout: {stdout}"
    );
    assert!(
        !stderr.contains("fetch origin") && !git_log.contains("fetch origin"),
        "dirty check should stop before fetch/network work, stderr: {stderr}, git log: {git_log}"
    );

    std::fs::remove_dir_all(repo).expect("cleanup temp dir");
}

#[test]
fn update_reports_dirty_tree_before_fetch_failure() {
    let repo = make_temp_dir("update-dirty");
    let fake_bin = repo.join("fake-bin");
    std::fs::create_dir_all(&fake_bin).expect("create fake bin dir");
    let git_log = repo.join("git.log");
    write_fake_git(&fake_bin.join("git"));

    let output = zipcode_bin()
        .arg("update")
        .current_dir(&repo)
        .env("PATH", prepend_path(&fake_bin))
        .env("ZIPCODE_FAKE_GIT_REPO", &repo)
        .env("ZIPCODE_FAKE_GIT_LOG", &git_log)
        .output()
        .expect("failed to run zipcode update");

    assert!(
        !output.status.success(),
        "update should fail with a local-modifications blocker"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let git_log = std::fs::read_to_string(&git_log).expect("read fake git log");
    assert!(
        stdout.contains("Working tree: dirty"),
        "expected dirty working tree summary, got stdout: {stdout}"
    );
    assert!(
        stderr.contains("local modifications"),
        "expected local modifications error, got stderr: {stderr}"
    );
    assert!(
        !stderr.contains("fetch origin") && !git_log.contains("fetch origin"),
        "dirty update should stop before fetch/network work, stderr: {stderr}, git log: {git_log}"
    );

    std::fs::remove_dir_all(repo).expect("cleanup temp dir");
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
    let home = make_temp_dir("prompt-no-model-guidance");
    let output = zipcode_bin()
        .args(["prompt", "hello"])
        .env("HOME", &home)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode prompt");

    assert!(
        !output.status.success(),
        "prompt should exit non-zero when not ready"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}{stderr}");

    assert!(
        !combined.contains("panicked at") && !combined.contains("RUST_BACKTRACE"),
        "should not panic, got: {combined}"
    );
    assert!(
        stdout.contains("Setup needed before zipcode can start.")
            && stdout.contains("Copy a .gguf AI model"),
        "prompt should show setup guidance instead of a raw missing-model error, got: {combined}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
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

    assert!(
        !output.status.success(),
        "bare zipcode should exit non-zero when not ready"
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

    assert!(
        !output.status.success(),
        "bare zipcode should exit non-zero when repair needed"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn repl_no_model_graceful_error() {
    let home = make_temp_dir("repl-no-model-guidance");
    let output = zipcode_bin()
        .args(["repl", "--ui", "plain"])
        .env("HOME", &home)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode repl");

    assert!(
        !output.status.success(),
        "repl should exit non-zero when not ready"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}{stderr}");

    assert!(
        !combined.contains("panicked at") && !combined.contains("RUST_BACKTRACE"),
        "repl should not panic, got: {combined}"
    );
    assert!(
        stdout.contains("Setup needed before zipcode can start.")
            && stdout.contains("Copy a .gguf AI model"),
        "repl should show setup guidance instead of a raw missing-model error, got: {combined}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
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
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("write fake tokenizer");

    let output = zipcode_bin()
        .arg("doctor")
        .env("HOME", &home)
        .env("PATH", &isolated_bin)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor");

    assert!(
        !output.status.success(),
        "doctor should exit non-zero when server missing"
    );
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
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
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
fn doctor_auto_selects_llama_server_for_gemma4_when_helper_exists() {
    let home = make_temp_dir("doctor-auto-helper");
    let model_dir = home.join(".zipcode/models");
    let zipcode_bin_dir = home.join(".zipcode/bin");
    let isolated_path = home.join("empty-path");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&zipcode_bin_dir).expect("create helper dir");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    write_fake_llama_server(&zipcode_bin_dir.join("llama-server"));

    let output = zipcode_bin()
        .arg("doctor")
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor");

    assert!(output.status.success(), "doctor should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Engine: compatibility helper (llama-server)")
            && stdout.contains("Status: Ready"),
        "doctor should auto-select llama-server for Gemma 4 when helper exists, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn bare_zipcode_and_doctor_agree_when_helper_is_found_on_path_for_gemma4() {
    let home = make_temp_dir("startup-doctor-path-helper");
    let model_dir = home.join(".zipcode/models");
    let path_dir = home.join("path-bin");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&path_dir).expect("create PATH dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    write_fake_llama_server(&path_dir.join("llama-server"));

    let doctor = zipcode_bin()
        .arg("doctor")
        .env("HOME", &home)
        .env("PATH", &path_dir)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor");
    assert!(doctor.status.success(), "doctor should exit 0");
    let doctor_stdout = String::from_utf8_lossy(&doctor.stdout);
    assert!(
        doctor_stdout.contains("Status: Ready")
            && doctor_stdout.contains("Engine: compatibility helper (llama-server)")
            && doctor_stdout.contains(&path_dir.join("llama-server").display().to_string()),
        "doctor should be ready via PATH helper, got: {doctor_stdout}"
    );

    let startup = run_fullscreen_zipcode_with_script(
        &[],
        &home,
        &repo_root(),
        "/quit\n",
        &[("PATH", path_dir.display().to_string())],
    );
    assert!(startup.status.success(), "bare zipcode should exit 0");
    let startup_output = format!(
        "{}{}",
        String::from_utf8_lossy(&startup.stdout),
        String::from_utf8_lossy(&startup.stderr)
    );
    assert!(
        !startup_output.contains("Setup needed before zipcode can start.")
            && !startup_output.contains("Repair needed before zipcode can start.")
            && !startup_output.contains("Gemma 4 is not supported by the native llama-cpp backend"),
        "bare zipcode should agree with doctor and start instead of showing readiness errors, got: {startup_output}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn stale_saved_helper_path_does_not_block_valid_path_fallback_discovery() {
    let home = make_temp_dir("stale-helper-fallback");
    let model_dir = home.join(".zipcode/models");
    let path_dir = home.join("path-bin");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&path_dir).expect("create PATH dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    write_fake_llama_server(&path_dir.join("llama-server"));
    write_file(
        &home.join(".zipcode/config.json"),
        r#"{
  "model_dir": "~/.zipcode/models",
  "model_file": "gemma-4-test.gguf",
  "llama_server_bin": "~/.zipcode/bin/missing-llama-server"
}"#,
    );

    let output = zipcode_bin()
        .arg("doctor")
        .env("HOME", &home)
        .env("PATH", &path_dir)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor");

    assert!(output.status.success(), "doctor should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Status: Ready")
            && stdout.contains(&path_dir.join("llama-server").display().to_string())
            && !stdout.contains("Status: Repair needed"),
        "doctor should fall back to a working helper instead of blocking on the stale saved path, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn doctor_resolves_tilde_global_llama_server_bin() {
    let home = make_temp_dir("doctor-tilde-global-helper");
    let model_dir = home.join(".zipcode/models");
    let helper_dir = home.join("custom/bin");
    let isolated_path = home.join("empty-path");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&helper_dir).expect("create helper dir");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    write_fake_llama_server(&helper_dir.join("my-helper"));
    write_file(
        &home.join(".zipcode/config.json"),
        r#"{
  "model_dir": "~/.zipcode/models",
  "model_file": "gemma-4-test.gguf",
  "llama_server_bin": "~/custom/bin/my-helper"
}"#,
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
        stdout.contains("Status: Ready")
            && stdout.contains(&helper_dir.join("my-helper").display().to_string())
            && !stdout.contains("saved helper path could not be used"),
        "doctor should expand a tilde helper path from global config, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn startup_surfaces_stale_saved_helper_path_when_fallback_is_used() {
    let home = make_temp_dir("startup-stale-helper-fallback");
    let model_dir = home.join(".zipcode/models");
    let path_dir = home.join("path-bin");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&path_dir).expect("create PATH dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    write_fake_llama_server(&path_dir.join("llama-server"));
    write_file(
        &home.join(".zipcode/config.json"),
        r#"{
  "model_dir": "~/.zipcode/models",
  "model_file": "gemma-4-test.gguf",
  "llama_server_bin": "~/.zipcode/bin/missing-llama-server"
}"#,
    );

    let output = run_fullscreen_zipcode_with_script(
        &[],
        &home,
        &repo_root(),
        "/quit\n",
        &[("PATH", path_dir.display().to_string())],
    );
    assert!(output.status.success(), "bare zipcode should exit 0");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("Compatibility helper note: saved helper path could not be used")
            && !combined.contains("Repair needed before zipcode can start."),
        "startup should surface the stale saved helper warning while still starting, got: {combined}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn doctor_backend_candle_does_not_claim_gemma4_is_ready() {
    let home = make_temp_dir("doctor-candle-gemma4");
    let model_dir = home.join(".zipcode/models");
    let zipcode_bin_dir = home.join(".zipcode/bin");
    let isolated_path = home.join("empty-path");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&zipcode_bin_dir).expect("create helper dir");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("write fake tokenizer");
    write_fake_llama_server(&zipcode_bin_dir.join("llama-server"));

    let output = zipcode_bin()
        .args(["doctor", "--backend", "candle"])
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor --backend candle");

    assert!(
        !output.status.success(),
        "doctor should exit non-zero when not ready"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("Status: Ready")
            && stdout.contains("Gemma 4")
            && stdout.contains("candle"),
        "doctor should not report candle as Gemma 4 ready, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn prompt_explicit_llama_cpp_is_rejected_for_gemma4() {
    let home = make_temp_dir("prompt-gemma4-explicit-llama-cpp");
    let model_path = home.join("gemma-4-test.gguf");
    let helper_path = home.join("path-bin/llama-server");
    std::fs::create_dir_all(helper_path.parent().expect("helper parent"))
        .expect("create helper dir");
    std::fs::write(&model_path, b"GGUF").expect("write fake gguf");
    write_fake_llama_server(&helper_path);

    let output = zipcode_bin()
        .args([
            "--backend",
            "llama-cpp",
            "--model",
            model_path.to_str().expect("utf-8 model path"),
            "prompt",
            "hello",
        ])
        .env("HOME", &home)
        .env("ZIPCODE_LLAMA_SERVER_BIN", &helper_path)
        .output()
        .expect("failed to run zipcode prompt");

    assert!(
        !output.status.success(),
        "prompt should exit non-zero when backend rejected"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Repair needed before zipcode can start.")
            && stdout.contains("Gemma 4 is not supported by the native llama-cpp backend"),
        "prompt should reject explicit native llama-cpp for Gemma 4, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn doctor_reports_unrunnable_helper_gpu_offload_config() {
    let home = make_temp_dir("doctor-unrunnable-helper-config");
    let model_dir = home.join(".zipcode/models");
    let zipcode_bin_dir = home.join(".zipcode/bin");
    let isolated_path = home.join("empty-path");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&zipcode_bin_dir).expect("create helper dir");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    write_cpu_only_llama_server(&zipcode_bin_dir.join("llama-server"));
    write_file(
        &home.join(".zipcode/config.json"),
        &format!(
            r#"{{
  "model_dir": "{}",
  "model_file": "gemma-4-test.gguf",
  "llama_server_bin": "{}",
  "gpu_layers": 999
}}"#,
            model_dir.display(),
            zipcode_bin_dir.join("llama-server").display()
        ),
    );

    let output = zipcode_bin()
        .arg("doctor")
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor");

    assert!(
        !output.status.success(),
        "doctor should exit non-zero when not ready"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("Status: Ready")
            && stdout.contains("gpu_layers")
            && stdout.contains("Compatibility helper"),
        "doctor should surface unrunnable helper-backed GPU config instead of reporting Ready, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn bare_zipcode_surfaces_unrunnable_helper_gpu_offload_config() {
    let home = make_temp_dir("startup-unrunnable-helper-config");
    let model_dir = home.join(".zipcode/models");
    let zipcode_bin_dir = home.join(".zipcode/bin");
    let path_dir = home.join("path-bin");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&zipcode_bin_dir).expect("create helper dir");
    std::fs::create_dir_all(&path_dir).expect("create PATH dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    write_cpu_only_llama_server(&zipcode_bin_dir.join("llama-server"));
    write_file(
        &home.join(".zipcode/config.json"),
        &format!(
            r#"{{
  "model_dir": "{}",
  "model_file": "gemma-4-test.gguf",
  "llama_server_bin": "{}",
  "gpu_layers": 999
}}"#,
            model_dir.display(),
            zipcode_bin_dir.join("llama-server").display()
        ),
    );

    let output = run_fullscreen_zipcode_with_script(
        &[],
        &home,
        &repo_root(),
        "/quit\n",
        &[("PATH", path_dir.display().to_string())],
    );
    assert!(
        !output.status.success(),
        "bare zipcode should exit non-zero when helper unrunnable"
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("Repair needed before zipcode can start.")
            && combined.contains("gpu_layers"),
        "startup should surface the unrunnable helper-backed config, got: {combined}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn doctor_times_out_hanging_helper_device_probe() {
    let home = make_temp_dir("doctor-hanging-helper-probe");
    let model_dir = home.join(".zipcode/models");
    let zipcode_bin_dir = home.join(".zipcode/bin");
    let isolated_path = home.join("empty-path");
    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&zipcode_bin_dir).expect("create helper dir");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    write_hanging_list_devices_llama_server(&zipcode_bin_dir.join("llama-server"));
    write_file(
        &home.join(".zipcode/config.json"),
        &format!(
            r#"{{
  "model_dir": "{}",
  "model_file": "gemma-4-test.gguf",
  "llama_server_bin": "{}",
  "gpu_layers": 999
}}"#,
            model_dir.display(),
            zipcode_bin_dir.join("llama-server").display()
        ),
    );

    let started = std::time::Instant::now();
    let output = zipcode_bin()
        .arg("doctor")
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode doctor");
    let elapsed = started.elapsed();

    assert!(
        !output.status.success(),
        "doctor should exit non-zero when not ready"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "doctor should time out the helper probe instead of hanging for {elapsed:?}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("Status: Ready")
            && stdout.contains("timed out")
            && stdout.contains("gpu_layers"),
        "doctor should surface the timed-out helper probe, got: {stdout}"
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
    std::fs::write(model_dir.join("demo.gguf"), b"GGUF").expect("write fake gguf");
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
fn doctor_reports_project_config_path_when_project_json_is_invalid() {
    let home = make_temp_dir("doctor-project-config-warning-home");
    let project = make_temp_dir("doctor-project-config-warning-project");
    std::fs::write(project.join(".zipcode.json"), "{ invalid json\n")
        .expect("write invalid project config");

    let output = zipcode_bin()
        .arg("doctor")
        .current_dir(&project)
        .env("HOME", &home)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to run zipcode doctor");

    assert!(
        !output.status.success(),
        "doctor should exit non-zero when not ready"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let project_config = project.join(".zipcode.json").display().to_string();
    let global_config = home.join(".zipcode/config.json").display().to_string();

    assert!(
        stdout.contains(&format!("Fix or replace {project_config}")),
        "doctor should point at the broken project config, got: {stdout}"
    );
    assert!(
        !stdout.contains(&format!("Fix or replace {global_config}")),
        "doctor should not blame the unrelated global config, got: {stdout}"
    );
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
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("write fake tokenizer");
    write_executable(
        &zipcode_bin_dir.join("llama-server"),
        "#!/bin/sh\necho fake llama-server\n",
    );

    let output = zipcode_bin()
        .args(["setup", "--skip-smoke"])
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env("CUDA_PATH", "/tmp/fake-cuda")
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
    assert!(
        config.contains("\"gpu_layers\": 999"),
        "setup should persist the recommended gpu_layers default, got: {config}"
    );
    assert!(
        config.contains("\"flash_attention\": true"),
        "setup should persist the recommended flash_attention default, got: {config}"
    );
    assert!(wrapper_path.is_file(), "setup should create wrapper shim");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Smoke: skipped (--skip-smoke)"),
        "setup should report skipped smoke, got: {stdout}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

/// Regression for #49: `zipcode setup --skip-smoke` with no local model must
/// surface the friendly first-run guidance (model location, next steps) and
/// exit cleanly under `--skip-smoke`, instead of bailing with the terse
/// `No .gguf AI model found` error.
#[test]
fn setup_skip_smoke_guides_first_run_when_model_missing() {
    let home = make_temp_dir("setup-skip-smoke-no-model");
    let isolated_path = home.join("empty-path");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");

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
        "setup --skip-smoke should exit 0 in no-model first-run state, got: {output:?}"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stdout.contains("Setup needed before zipcode can start."),
        "setup should print friendly first-run guidance, got stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("Copy a .gguf AI model into"),
        "setup should explain where to put the model, got: {stdout}"
    );
    assert!(
        stdout.contains("Smoke: skipped until setup is complete"),
        "setup should report skipped smoke, got: {stdout}"
    );
    assert!(
        !stderr.contains("Error: No .gguf AI model found"),
        "setup should NOT bail with the terse missing-model error, got stderr: {stderr}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

/// Regression for #51: `zipcode setup --skip-smoke` from a directory with a
/// malformed project-local `.zipcode.json` must surface the parse failure
/// (matching `doctor`'s behavior) instead of silently ignoring it and
/// reporting only a generic missing-model error.
#[test]
fn setup_skip_smoke_surfaces_malformed_project_config() {
    let home = make_temp_dir("setup-skip-smoke-bad-cfg");
    let isolated_path = home.join("empty-path");
    let project_dir = home.join("project");
    std::fs::create_dir_all(&isolated_path).expect("create isolated path dir");
    std::fs::create_dir_all(&project_dir).expect("create project dir");
    std::fs::write(project_dir.join(".zipcode.json"), b"{ invalid json\n")
        .expect("write malformed project config");

    let output = zipcode_bin()
        .args(["setup", "--skip-smoke"])
        .current_dir(&project_dir)
        .env("HOME", &home)
        .env("PATH", &isolated_path)
        .env_remove("ZIPCODE_LLAMA_SERVER_BIN")
        .env_remove("LLAMA_SERVER_BIN")
        .output()
        .expect("failed to run zipcode setup --skip-smoke");

    assert!(
        output.status.success(),
        "setup --skip-smoke should exit 0 after surfacing the parse error, got: {output:?}"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let bad_path = project_dir.join(".zipcode.json");
    assert!(
        stdout.contains("Repair needed before zipcode can start."),
        "setup should escalate to repair guidance, got stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("Saved settings could not be read"),
        "setup should mention the parse failure in the findings, got: {stdout}"
    );
    assert!(
        stdout.contains(&bad_path.display().to_string()),
        "setup should reference the broken project config path, got: {stdout}"
    );
    assert!(
        stdout.contains("Smoke: skipped until setup is complete"),
        "setup should report skipped smoke, got: {stdout}"
    );
    assert!(
        !stderr.contains("Error: No .gguf AI model found"),
        "setup should NOT hide the parse error behind a missing-model error, got stderr: {stderr}"
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
    std::fs::write(model_dir.join("gemma-4-test.gguf"), b"GGUF").expect("write fake gguf");
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

    assert!(
        !output.status.success(),
        "should exit non-zero when not ready"
    );
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

    std::fs::write(&model, b"GGUF").expect("write fake gguf");
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
        .env("CUDA_PATH", "/tmp/fake-cuda")
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

    let config =
        std::fs::read_to_string(home.join(".zipcode/config.json")).expect("read installed config");
    assert!(
        config.contains("\"gpu_layers\": 999"),
        "install should persist the recommended gpu_layers default, got: {config}"
    );
    assert!(
        config.contains("\"flash_attention\": true"),
        "install should persist the recommended flash_attention default, got: {config}"
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
fn root_install_script_reuses_existing_model_and_helper_without_prompt() {
    let home = make_temp_dir("root-install-reuse-existing");
    let model_dir = home.join(".zipcode/models");
    let helper_dir = home.join(".zipcode/bin");
    let install_script = repo_root().join("install.sh");

    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::create_dir_all(&helper_dir).expect("create helper dir");
    std::fs::write(model_dir.join("gemma-4-e2b-it-Q8_0.gguf"), b"GGUF")
        .expect("write gemma 4 model");
    std::fs::write(model_dir.join("qwen2.5-0.5b-instruct-q4_k_m.gguf"), b"GGUF")
        .expect("write secondary model");
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("write tokenizer");
    write_executable(
        &helper_dir.join("llama-server"),
        "#!/bin/sh
echo fake llama-server
",
    );
    write_file(
        &home.join(".zipcode/config.json"),
        &format!(
            r#"{{
  "model_dir": "{}",
  "model_file": "gemma-4-e2b-it-Q8_0.gguf",
  "llama_server_bin": "{}",
  "permission_mode": "workspace-write",
  "gpu_layers": null,
  "flash_attention": false
}}"#,
            model_dir.display(),
            helper_dir.join("llama-server").display(),
        ),
    );

    let output = Command::new("bash")
        .arg(&install_script)
        .args(["--binary", env!("CARGO_BIN_EXE_zipcode")])
        .env("HOME", &home)
        .env("CUDA_PATH", "/tmp/fake-cuda")
        .env("ZIPCODE_INSTALL_SKIP_SYSTEM_BIN", "1")
        .output()
        .expect("failed to rerun root install.sh");

    assert!(
        output.status.success(),
        "rerun install should succeed without prompts, got: {output:?}"
    );

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !combined.contains("Model assets are still needed before zipcode can run the full setup."),
        "existing model selection should suppress model prompts, got: {combined}"
    );
    assert!(
        !combined
            .contains("Gemma 4 compatibility helper is required for zipcode to run this model:"),
        "existing helper should suppress helper prompts, got: {combined}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn root_install_script_reuses_installed_helper_wrapper_safely() {
    let home = make_temp_dir("root-install-reuse-wrapper");
    let asset_dir = make_temp_dir("root-install-reuse-wrapper-assets");
    let model = asset_dir.join("gemma-4-test.gguf");
    let tokenizer = asset_dir.join("tokenizer.json");
    let helper = asset_dir.join("llama-server");
    let install_script = repo_root().join("install.sh");

    std::fs::write(&model, b"GGUF").expect("write fake gguf");
    std::fs::write(&tokenizer, b"{}").expect("write fake tokenizer");
    write_executable(
        &helper,
        "#!/bin/sh
echo fake llama-server
",
    );

    let first = Command::new("bash")
        .arg(&install_script)
        .args(["--binary", env!("CARGO_BIN_EXE_zipcode")])
        .arg("--model")
        .arg(&model)
        .arg("--tokenizer")
        .arg(&tokenizer)
        .arg("--llama-server")
        .arg(&helper)
        .env("HOME", &home)
        .env("CUDA_PATH", "/tmp/fake-cuda")
        .env("ZIPCODE_INSTALL_SKIP_SYSTEM_BIN", "1")
        .output()
        .expect("failed to run first install.sh");
    assert!(
        first.status.success(),
        "first install should succeed, got: {first:?}"
    );

    let installed_wrapper = home.join(".zipcode/bin/llama-server");
    let second = Command::new("bash")
        .arg(&install_script)
        .args(["--binary", env!("CARGO_BIN_EXE_zipcode")])
        .arg("--model")
        .arg(&model)
        .arg("--tokenizer")
        .arg(&tokenizer)
        .arg("--llama-server")
        .arg(&installed_wrapper)
        .env("HOME", &home)
        .env("CUDA_PATH", "/tmp/fake-cuda")
        .env("ZIPCODE_INSTALL_SKIP_SYSTEM_BIN", "1")
        .output()
        .expect("failed to run second install.sh");
    assert!(
        second.status.success(),
        "second install should succeed, got: {second:?}"
    );

    let helper_run = Command::new(&installed_wrapper)
        .arg("--help")
        .env("HOME", &home)
        .output()
        .expect("run installed helper wrapper");
    assert!(
        helper_run.status.success(),
        "reused helper wrapper should stay executable, got: {helper_run:?}"
    );
    let stdout = String::from_utf8_lossy(&helper_run.stdout);
    assert!(
        stdout.contains("fake llama-server"),
        "helper wrapper should still delegate to the real helper, got: {stdout}"
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

    std::fs::write(&model, b"GGUF").expect("write fake gguf");
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
#[allow(clippy::too_many_lines)]
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
  printf 'GGUF' > "$dir/gemma-4-31b-it-q8_0.gguf"
  printf 'GGUF' > "$dir/gemma-4-31b-it-f16.gguf"
else
  printf 'GGUF' > "$dir/gemma-4-e2b-it-q8_0.gguf"
fi
printf 'GGUF' > "$dir/qwen2.5-0.5b-instruct-q4_k_m.gguf"
printf 'GGUF' > "$dir/mmproj-gemma-4-31b-it-f16.gguf"
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
#!/usr/bin/python3
import http.server
import json
import socketserver
import sys

args = sys.argv[1:]
if "--list-devices" in args:
    print("Available devices:\n  CUDA0")
    sys.exit(0)
if "--help" in args or "-h" in args:
    # Match the modern llama-server help so flash_attention probing
    # picks the [on|off|auto] flag form (production parity).
    print("usage: llama-server [options]\n  -fa, --flash-attn [on|off|auto]   set Flash Attention use")
    sys.exit(0)
port = int(args[args.index("--port") + 1]) if "--port" in args else 8080

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok"}' if self.path == "/health" else b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = json.dumps({
            "choices": [
                {
                    "delta": {"content": "READY"},
                    "finish_reason": "stop",
                }
            ]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format, *args):
        return

class Server(socketserver.TCPServer):
    allow_reuse_address = True

server = Server(("127.0.0.1", port), Handler)
server.serve_forever()
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
#[allow(clippy::too_many_lines)]
fn root_install_script_handles_multiple_existing_models_and_still_reaches_ready_setup() {
    let home = make_temp_dir("root-install-multi-existing");
    let model_dir = home.join(".zipcode/models");
    let asset_dir = make_temp_dir("root-install-multi-existing-assets");
    let fake_helper_builder = asset_dir.join("fake-build-llama-server.sh");
    let install_script = repo_root().join("install.sh");

    std::fs::create_dir_all(&model_dir).expect("create model dir");
    std::fs::write(model_dir.join("gemma-4-e2b-it-q8_0.gguf"), b"GGUF").expect("write model 1");
    std::fs::write(model_dir.join("gemma-4-e2b-it-f16.gguf"), b"GGUF").expect("write model 2");
    std::fs::write(model_dir.join("mmproj-gemma-4-e2b-it-f16.gguf"), b"GGUF")
        .expect("write mmproj");
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("write tokenizer");

    write_executable(
        &fake_helper_builder,
        r#"#!/bin/sh
set -eu
install_dir="${1:-${HOME}/.zipcode}"
mkdir -p "$install_dir/bin"
cat > "$install_dir/bin/llama-server" <<'EOF'
#!/usr/bin/python3
import http.server
import json
import socketserver
import sys

args = sys.argv[1:]
if "--list-devices" in args:
    print("Available devices:\n  CUDA0")
    sys.exit(0)
if "--help" in args or "-h" in args:
    # Match the modern llama-server help so flash_attention probing
    # picks the [on|off|auto] flag form (production parity).
    print("usage: llama-server [options]\n  -fa, --flash-attn [on|off|auto]   set Flash Attention use")
    sys.exit(0)
port = int(args[args.index("--port") + 1]) if "--port" in args else 8080

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok"}' if self.path == "/health" else b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = json.dumps({
            "choices": [
                {
                    "delta": {"content": "READY"},
                    "finish_reason": "stop",
                }
            ]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format, *args):
        return

class Server(socketserver.TCPServer):
    allow_reuse_address = True

server = Server(("127.0.0.1", port), Handler)
server.serve_forever()
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
#[allow(clippy::too_many_lines)]
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
    std::fs::write(&model_path, b"GGUF").expect("write fake model");
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

#[test]
fn plain_repl_session_load_failure_stays_alive() {
    let (home, model_path, helper_path) = setup_repl_fixture("plain-session-load-failure");
    let args = vec![
        "repl".to_string(),
        "--ui".to_string(),
        "plain".to_string(),
        "--backend".to_string(),
        "llama-server".to_string(),
        "--model".to_string(),
        model_path.display().to_string(),
    ];

    let output = run_plain_zipcode_with_input(
        &args,
        &home,
        &repo_root(),
        "/session missing-session\n/status\n/quit\n",
        &[(
            "ZIPCODE_LLAMA_SERVER_BIN",
            helper_path.display().to_string(),
        )],
    );

    assert!(
        output.status.success(),
        "plain REPL should stay alive after a failed /session load, got: {output:?}"
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("Failed to load session `missing-session`")
            && combined.contains("Session ID:")
            && combined.contains("Goodbye."),
        "expected inline session-load error and continued REPL interaction, got: {combined}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn plain_repl_missing_session_flag_fails_before_printing_banner() {
    let (home, model_path, helper_path) = setup_repl_fixture("plain-missing-session-flag");
    let output = zipcode_bin()
        .args([
            "--ui",
            "plain",
            "--session",
            "missing-session",
            "repl",
            "--backend",
            "llama-server",
            "--model",
            model_path.to_str().expect("utf-8 model path"),
        ])
        .env("HOME", &home)
        .env("ZIPCODE_LLAMA_SERVER_BIN", &helper_path)
        .output()
        .expect("failed to run plain REPL with missing session");

    assert!(
        !output.status.success(),
        "repl with a missing startup session should fail"
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("Failed to load session `missing-session`")
            && !combined.contains("type /help for commands"),
        "startup session failure should not print the REPL banner first, got: {combined}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn clear_persists_new_session_id_immediately() {
    let (home, model_path, helper_path) = setup_repl_fixture("plain-clear-persists-session");
    let args = vec![
        "repl".to_string(),
        "--ui".to_string(),
        "plain".to_string(),
        "--backend".to_string(),
        "llama-server".to_string(),
        "--model".to_string(),
        model_path.display().to_string(),
    ];

    let clear_output = run_plain_zipcode_with_input(
        &args,
        &home,
        &repo_root(),
        "/clear\n/quit\n",
        &[(
            "ZIPCODE_LLAMA_SERVER_BIN",
            helper_path.display().to_string(),
        )],
    );
    assert!(
        clear_output.status.success(),
        "plain REPL clear flow should exit 0, got: {clear_output:?}"
    );
    let clear_combined = format!(
        "{}{}",
        String::from_utf8_lossy(&clear_output.stdout),
        String::from_utf8_lossy(&clear_output.stderr)
    );
    let session_id = parse_cleared_session_id(&clear_combined)
        .expect("clear output should print the new resumable session id");
    let session_path = home.join(format!(".zipcode/sessions/{session_id}.json"));
    assert!(
        session_path.is_file(),
        "cleared session should be saved immediately at {}, output: {}",
        session_path.display(),
        clear_combined
    );

    let resumed_args = vec![
        "--session".to_string(),
        session_id.clone(),
        "repl".to_string(),
        "--ui".to_string(),
        "plain".to_string(),
        "--backend".to_string(),
        "llama-server".to_string(),
        "--model".to_string(),
        model_path.display().to_string(),
    ];
    let resumed = run_plain_zipcode_with_input(
        &resumed_args,
        &home,
        &repo_root(),
        "/session\n/quit\n",
        &[(
            "ZIPCODE_LLAMA_SERVER_BIN",
            helper_path.display().to_string(),
        )],
    );
    assert!(
        resumed.status.success(),
        "resuming the freshly cleared session should work, got: {resumed:?}"
    );
    let resumed_combined = format!(
        "{}{}",
        String::from_utf8_lossy(&resumed.stdout),
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert!(
        resumed_combined.contains(&format!("Resumed session {session_id}"))
            && resumed_combined.contains(&format!("Session ID:   {session_id}")),
        "expected the cleared session id to be immediately resumable, got: {resumed_combined}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
fn fullscreen_compact_noop_is_reported_as_skipped() {
    let (home, model_path, helper_path) = setup_repl_fixture("fullscreen-compact-noop");
    let args = vec![
        "repl".to_string(),
        "--ui".to_string(),
        "fullscreen".to_string(),
        "--backend".to_string(),
        "llama-server".to_string(),
        "--model".to_string(),
        model_path.display().to_string(),
    ];

    let output = run_fullscreen_zipcode_with_script(
        &args,
        &home,
        &repo_root(),
        "/compact\n/quit\n",
        &[(
            "ZIPCODE_LLAMA_SERVER_BIN",
            helper_path.display().to_string(),
        )],
    );

    assert!(output.status.success(), "fullscreen TUI should exit 0");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("Compaction skipped") && !combined.contains("Compaction complete"),
        "fullscreen /compact should report skipped/no-op accurately, got: {combined}"
    );

    std::fs::remove_dir_all(home).expect("cleanup temp dir");
}

#[test]
#[allow(clippy::too_many_lines, clippy::literal_string_with_formatting_args)]
fn build_llama_server_retries_cpu_only_after_cuda_failure() {
    let temp = make_temp_dir("build-llama-server-cpu-fallback");
    let fake_bin = temp.join("bin");
    let workdir = temp.join("work");
    let install_dir = temp.join("install");
    let cmake_log = temp.join("cmake.log");
    std::fs::create_dir_all(&fake_bin).expect("create fake bin");
    std::fs::create_dir_all(&workdir).expect("create workdir");

    write_executable(
        &fake_bin.join("git"),
        r#"#!/bin/sh
set -eu
dest=""
for arg in "$@"; do
  dest="$arg"
done
mkdir -p "$dest"
"#,
    );
    write_executable(
        &fake_bin.join("nvcc"),
        r#"#!/bin/sh
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release 11.5, V11.5.119"
"#,
    );
    write_executable(
        &fake_bin.join("nvidia-smi"),
        r#"#!/bin/sh
if [ "${1:-}" = "--query-gpu=compute_cap" ]; then
  echo "7.5"
else
  echo "NVIDIA-SMI fake"
fi
"#,
    );
    write_executable(
        &fake_bin.join("c++"),
        r#"#!/bin/sh
if [ "${1:-}" = "-dumpfullversion" ] || [ "${1:-}" = "-dumpversion" ]; then
  echo "11.4.0"
else
  exit 0
fi
"#,
    );
    write_executable(
        &fake_bin.join("cmake"),
        format!(
            r#"#!/bin/sh
set -eu
printf '%s\n' "$*" >> "{log}"
if [ "${{1:-}}" = "-S" ]; then
  build=""
  prev=""
  for arg in "$@"; do
    if [ "$prev" = "-B" ]; then
      build="$arg"
    fi
    prev="$arg"
  done
  mkdir -p "$build"
  printf '%s\n' "$@" > "$build/config-args.txt"
  exit 0
fi
if [ "${{1:-}}" = "--build" ]; then
  build="${{2:?}}"
  if grep -q -- '-DGGML_CUDA=ON' "$build/config-args.txt" && [ ! -f "{sentinel}" ]; then
    touch "{sentinel}"
    echo "fake cuda build failure" >&2
    exit 1
  fi
  mkdir -p "$build/bin"
  cat > "$build/bin/llama-server" <<'EOF'
#!/bin/sh
exit 0
EOF
  chmod +x "$build/bin/llama-server"
  exit 0
fi
exit 0
"#,
            log = cmake_log.display(),
            sentinel = temp.join("gpu-failed-once").display(),
        )
        .as_str(),
    );

    let output = Command::new("bash")
        .arg(repo_root().join("scripts/build_llama_server.sh"))
        .arg(&install_dir)
        .env(
            "PATH",
            format!("{}:{}", fake_bin.display(), std::env::var("PATH").unwrap()),
        )
        .env("CUDACXX", fake_bin.join("nvcc"))
        .env("ZIPCODE_LLAMA_SERVER_WORKDIR", &workdir)
        .output()
        .expect("run build_llama_server.sh");

    assert!(
        output.status.success(),
        "build script should recover via CPU fallback, got: {output:?}"
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("Retrying CPU-only build"),
        "build script should announce CPU fallback, got: {combined}"
    );
    assert!(
        install_dir.join("bin/llama-server").is_file(),
        "install helper should still install llama-server wrapper"
    );

    let log = std::fs::read_to_string(cmake_log).expect("read cmake log");
    assert!(
        log.contains("-DGGML_CUDA=ON"),
        "initial configure should try CUDA, got: {log}"
    );
}

#[test]
#[allow(clippy::too_many_lines, clippy::literal_string_with_formatting_args)]
fn build_llama_server_uses_gcc10_host_compiler_when_available() {
    let temp = make_temp_dir("build-llama-server-gcc10");
    let fake_bin = temp.join("bin");
    let workdir = temp.join("work");
    let install_dir = temp.join("install");
    let cmake_log = temp.join("cmake.log");
    std::fs::create_dir_all(&fake_bin).expect("create fake bin");
    std::fs::create_dir_all(&workdir).expect("create workdir");

    write_executable(
        &fake_bin.join("git"),
        r#"#!/bin/sh
set -eu
dest=""
for arg in "$@"; do
  dest="$arg"
done
mkdir -p "$dest"
"#,
    );
    write_executable(
        &fake_bin.join("nvcc"),
        r#"#!/bin/sh
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release 11.5, V11.5.119"
"#,
    );
    write_executable(
        &fake_bin.join("nvidia-smi"),
        r#"#!/bin/sh
if [ "${1:-}" = "--query-gpu=compute_cap" ]; then
  echo "7.5"
else
  echo "NVIDIA-SMI fake"
fi
"#,
    );
    write_executable(
        &fake_bin.join("c++"),
        r#"#!/bin/sh
if [ "${1:-}" = "-dumpfullversion" ] || [ "${1:-}" = "-dumpversion" ]; then
  echo "11.4.0"
else
  exit 0
fi
"#,
    );
    write_executable(
        &fake_bin.join("gcc-10"),
        r#"#!/bin/sh
if [ "${1:-}" = "-dumpfullversion" ] || [ "${1:-}" = "-dumpversion" ]; then
  echo "10.5.0"
else
  exit 0
fi
"#,
    );
    write_executable(
        &fake_bin.join("g++-10"),
        r#"#!/bin/sh
if [ "${1:-}" = "-dumpfullversion" ] || [ "${1:-}" = "-dumpversion" ]; then
  echo "10.5.0"
else
  exit 0
fi
"#,
    );
    write_executable(
        &fake_bin.join("cmake"),
        format!(
            r#"#!/bin/sh
set -eu
printf 'ARGS:%s\n' "$*" >> "{log}"
printf 'CC=%s CXX=%s\n' "${{CC:-}}" "${{CXX:-}}" >> "{log}"
if [ "${{1:-}}" = "-S" ]; then
  build=""
  prev=""
  for arg in "$@"; do
    if [ "$prev" = "-B" ]; then
      build="$arg"
    fi
    prev="$arg"
  done
  mkdir -p "$build"
  printf '%s\n' "$@" > "$build/config-args.txt"
  exit 0
fi
if [ "${{1:-}}" = "--build" ]; then
  build="${{2:?}}"
  mkdir -p "$build/bin"
  cat > "$build/bin/llama-server" <<'EOF'
#!/bin/sh
exit 0
EOF
  chmod +x "$build/bin/llama-server"
  exit 0
fi
exit 0
"#,
            log = cmake_log.display(),
        )
        .as_str(),
    );

    let output = Command::new("bash")
        .arg(repo_root().join("scripts/build_llama_server.sh"))
        .arg(&install_dir)
        .env(
            "PATH",
            format!("{}:{}", fake_bin.display(), std::env::var("PATH").unwrap()),
        )
        .env("CUDACXX", fake_bin.join("nvcc"))
        .env("ZIPCODE_LLAMA_SERVER_WORKDIR", &workdir)
        .output()
        .expect("run build_llama_server.sh");

    assert!(
        output.status.success(),
        "build script should succeed, got: {output:?}"
    );
    let log = std::fs::read_to_string(cmake_log).expect("read cmake log");
    assert!(
        log.contains("-DCMAKE_CUDA_HOST_COMPILER="),
        "configure should set a CUDA host compiler, got: {log}"
    );
    assert!(
        log.contains("g++-10"),
        "gcc-10/g++-10 should be selected, got: {log}"
    );
    assert!(
        log.contains("-DCMAKE_CUDA_ARCHITECTURES=75"),
        "compute capability should be narrowed to detected arch, got: {log}"
    );
}
