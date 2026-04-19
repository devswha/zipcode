use anyhow::{Context, Result};

use crate::{wait_with_output_timeout, Tool, ToolContext, ToolResult};

/// Default timeout for REPL execution: 30 seconds.
const REPL_DEFAULT_TIMEOUT_MS: u64 = 30_000;
/// Minimum allowed timeout in milliseconds.
const REPL_MIN_TIMEOUT_MS: u64 = 1_000;
/// Maximum allowed timeout in milliseconds.
const REPL_MAX_TIMEOUT_MS: u64 = 300_000;

pub struct ReplTool;

impl Tool for ReplTool {
    fn name(&self) -> &str {
        "repl"
    }

    fn description(&self) -> &str {
        "Execute code in a REPL environment (python or node)"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["python", "node"],
                    "description": "The language runtime to use"
                },
                "code": {
                    "type": "string",
                    "description": "The code to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 30000)"
                }
            },
            "required": ["language", "code"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let language = args["language"].as_str().unwrap_or("").to_string();
        let code = args["code"].as_str().unwrap_or("").to_string();
        let timeout_ms = args["timeout"].as_u64().unwrap_or(REPL_DEFAULT_TIMEOUT_MS);
        let timeout_ms = timeout_ms.clamp(REPL_MIN_TIMEOUT_MS, REPL_MAX_TIMEOUT_MS);
        if timeout_ms != args["timeout"].as_u64().unwrap_or(REPL_DEFAULT_TIMEOUT_MS) {
            tracing::debug!(
                requested = args["timeout"].as_u64().unwrap_or(REPL_DEFAULT_TIMEOUT_MS),
                clamped = timeout_ms,
                "repl timeout clamped to valid range [{REPL_MIN_TIMEOUT_MS}, {REPL_MAX_TIMEOUT_MS}]"
            );
        }

        let (program, flag) = match language.as_str() {
            "python" => ("python3", "-c"),
            "node" => ("node", "-e"),
            other => {
                return Ok(ToolResult::error(format!(
                    "Unsupported language: '{other}'. Use 'python' or 'node'."
                )));
            }
        };

        let child = std::process::Command::new(program)
            .arg(flag)
            .arg(&code)
            .current_dir(&ctx.cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to execute {program}"))?;

        let timeout = std::time::Duration::from_millis(timeout_ms);
        match wait_with_output_timeout(child, timeout)? {
            Some(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                let mut result = String::new();
                if !stdout.is_empty() {
                    result.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str("[stderr]\n");
                    result.push_str(&stderr);
                }
                if !output.status.success() {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str(&format!(
                        "[exit code: {}]",
                        output.status.code().unwrap_or(-1)
                    ));
                }
                if result.is_empty() {
                    result = "(no output)".to_string();
                }
                Ok(ToolResult::new(result))
            }
            None => Ok(ToolResult::error(format!(
                "REPL execution timed out after {timeout_ms}ms"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use std::{path::PathBuf, thread, time::Duration};
    use tempfile::TempDir;

    fn test_ctx() -> ToolContext {
        ToolContext {
            cwd: std::env::temp_dir(),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        }
    }

    #[test]
    fn test_repl_python_basic() {
        let tool = ReplTool;
        let result = tool
            .execute(
                serde_json::json!({"language": "python", "code": "print('hello')"}),
                &test_ctx(),
            )
            .unwrap();
        assert!(result.content.contains("hello"));
    }

    #[test]
    fn test_repl_unsupported_language() {
        let tool = ReplTool;
        let result = tool
            .execute(
                serde_json::json!({"language": "ruby", "code": "puts 'hi'"}),
                &test_ctx(),
            )
            .unwrap();
        assert!(result.content.contains("Unsupported language"));
    }

    #[test]
    fn test_repl_python_stderr() {
        let tool = ReplTool;
        let result = tool
            .execute(
                serde_json::json!({"language": "python", "code": "import sys; sys.stderr.write('err\\n')"}),
                &test_ctx(),
            )
            .unwrap();
        assert!(result.content.contains("[stderr]"));
        assert!(result.content.contains("err"));
    }

    #[test]
    fn test_repl_timeout() {
        let tool = ReplTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "language": "python",
                    "code": "import time; time.sleep(60)",
                    "timeout": 500
                }),
                &test_ctx(),
            )
            .unwrap();
        assert!(result.content.contains("timed out"));
    }

    #[test]
    fn test_repl_uses_cwd() {
        let tool = ReplTool;
        let tmpdir = std::env::temp_dir();
        let ctx = ToolContext {
            cwd: tmpdir.clone(),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        };
        let result = tool
            .execute(
                serde_json::json!({"language": "python", "code": "import os; print(os.getcwd())"}),
                &ctx,
            )
            .unwrap();
        // Canonicalize both to handle /tmp -> /private/tmp on macOS
        let expected = tmpdir.canonicalize().unwrap_or(tmpdir);
        let actual = PathBuf::from(result.content.trim());
        let actual_canon = actual.canonicalize().unwrap_or(actual);
        assert_eq!(actual_canon, expected);
    }

    #[test]
    fn test_repl_nonzero_exit() {
        let tool = ReplTool;
        let result = tool
            .execute(
                serde_json::json!({"language": "python", "code": "exit(42)"}),
                &test_ctx(),
            )
            .unwrap();
        assert!(result.content.contains("[exit code: 42]"));
    }

    #[test]
    fn test_repl_timeout_kills_descendants() {
        let tool = ReplTool;
        let dir = TempDir::new().unwrap();
        let flag = dir.path().join("leaked.txt");
        let code = format!(
            "import pathlib, subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import pathlib, time; time.sleep(1); pathlib.Path(r\"{}\").write_text(\"leaked\")']); time.sleep(60)",
            flag.display()
        );
        let ctx = ToolContext {
            cwd: dir.path().to_path_buf(),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        };

        let result = tool
            .execute(
                serde_json::json!({
                    "language": "python",
                    "code": code,
                    "timeout": 200
                }),
                &ctx,
            )
            .unwrap();

        assert!(result.content.contains("timed out"));
        thread::sleep(Duration::from_millis(1500));
        assert!(
            !flag.exists(),
            "timeout should kill descendant interpreters before they outlive the REPL tool"
        );
    }

    #[test]
    fn test_repl_large_stdout_does_not_false_timeout() {
        let tool = ReplTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "language": "python",
                    "code": "import sys; sys.stdout.write('x' * 200000)",
                    "timeout": 2_000
                }),
                &test_ctx(),
            )
            .unwrap();

        assert!(
            !result.content.contains("timed out"),
            "fast large-output repl should not false-timeout: {}",
            result.content
        );
        assert!(
            result.content.len() >= 200_000,
            "expected captured stdout, got {} bytes",
            result.content.len()
        );
    }

    #[test]
    fn test_repl_timeout_clamped_to_minimum() {
        let tool = ReplTool;
        // 0ms timeout — should be clamped to REPL_MIN_TIMEOUT_MS (100ms)
        let result = tool
            .execute(
                serde_json::json!({
                    "language": "python",
                    "code": "print('clamped_repl')",
                    "timeout": 0
                }),
                &test_ctx(),
            )
            .unwrap();
        assert!(
            result.content.contains("clamped_repl"),
            "repl should succeed with clamped timeout, got: {}",
            result.content
        );
    }

    #[test]
    fn test_repl_timeout_clamped_to_maximum() {
        let tool = ReplTool;
        let result = tool
            .execute(
                serde_json::json!({
                    "language": "python",
                    "code": "print('huge_repl_timeout')",
                    "timeout": u64::MAX
                }),
                &test_ctx(),
            )
            .unwrap();
        assert!(
            result.content.contains("huge_repl_timeout"),
            "repl should succeed even with absurdly large timeout, got: {}",
            result.content
        );
    }
}
