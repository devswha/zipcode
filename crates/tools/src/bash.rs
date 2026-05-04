use std::fmt::Write;
use std::process::Command;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::{wait_with_output_timeout, Tool, ToolContext, ToolResult};

/// Minimum allowed timeout in milliseconds. Values below this are clamped up.
const BASH_MIN_TIMEOUT_MS: u64 = 100;
/// Maximum allowed timeout in milliseconds. Values above this are clamped down.
const BASH_MAX_TIMEOUT_MS: u64 = 300_000;
const BASH_DEFAULT_TIMEOUT_MS: u64 = 120_000;

pub struct BashTool;

impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "Execute an approved shell command in the workspace and return its output. Use for tests, builds, and shell-based inspection; use fetch_repo instead of bash when analyzing GitHub repository URLs."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 120000, min: 100, max: 300000)"
                }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let command = args["command"]
            .as_str()
            .context("missing 'command' argument")?;

        if command.trim().is_empty() {
            anyhow::bail!("command must not be empty");
        }

        let timeout_ms = args["timeout"].as_u64().unwrap_or(BASH_DEFAULT_TIMEOUT_MS);
        let timeout_ms = timeout_ms.clamp(BASH_MIN_TIMEOUT_MS, BASH_MAX_TIMEOUT_MS);
        if timeout_ms != args["timeout"].as_u64().unwrap_or(BASH_DEFAULT_TIMEOUT_MS) {
            tracing::debug!(
                requested = args["timeout"].as_u64().unwrap_or(BASH_DEFAULT_TIMEOUT_MS),
                clamped = timeout_ms,
                "bash timeout clamped to valid range [{BASH_MIN_TIMEOUT_MS}, {BASH_MAX_TIMEOUT_MS}]"
            );
        }
        let timeout = std::time::Duration::from_millis(timeout_ms);

        let child = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&ctx.cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to execute: {command}"))?;

        let Some(output) = wait_with_output_timeout(child, timeout)? else {
            return Ok(ToolResult::new(format!(
                "Error: command timed out after {timeout_ms}ms"
            )));
        };

        let mut result = String::from_utf8_lossy(&output.stdout).into_owned();

        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("STDERR:\n");
            result.push_str(&stderr);
        }

        if !output.status.success() {
            let code = output.status.code().unwrap_or(-1);
            if !result.is_empty() {
                result.push('\n');
            }
            let _ = write!(result, "Exit code: {code}");
        }

        Ok(ToolResult::new(result))
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
            cwd: PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
            parent_session_id: None,
            depth: 0,
            budget_tokens: None,
            spawn_child: None,
        }
    }

    #[test]
    fn test_bash_echo() {
        let tool = BashTool;
        let ctx = test_ctx();
        let args = serde_json::json!({"command": "echo hello"});
        let result = tool.execute(args, &ctx).unwrap();
        assert!(result.content.contains("hello"));
    }

    #[test]
    fn test_bash_captures_stderr() {
        let tool = BashTool;
        let ctx = test_ctx();
        let args = serde_json::json!({"command": "echo err >&2"});
        let result = tool.execute(args, &ctx).unwrap();
        assert!(result.content.contains("err"));
    }

    #[test]
    fn test_bash_nonzero_exit() {
        let tool = BashTool;
        let ctx = test_ctx();
        let args = serde_json::json!({"command": "exit 1"});
        let result = tool.execute(args, &ctx).unwrap();
        assert!(result.content.contains("Exit code: 1"));
    }

    #[test]
    fn test_bash_uses_cwd() {
        let tool = BashTool;
        let ctx = test_ctx();
        let args = serde_json::json!({"command": "pwd"});
        let result = tool.execute(args, &ctx).unwrap();
        assert!(!result.content.is_empty());
    }

    #[test]
    fn test_bash_timeout_kills_descendants() {
        let dir = TempDir::new().unwrap();
        let flag = dir.path().join("leaked.txt");
        let tool = BashTool;
        let ctx = ToolContext {
            cwd: dir.path().to_path_buf(),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
            parent_session_id: None,
            depth: 0,
            budget_tokens: None,
            spawn_child: None,
        };
        let command = format!(
            "python3 -c 'import pathlib, time; time.sleep(1); pathlib.Path(r\"{}\").write_text(\"leaked\")' & sleep 60",
            flag.display()
        );

        let result = tool
            .execute(
                serde_json::json!({"command": command, "timeout": 200}),
                &ctx,
            )
            .unwrap();

        assert!(result.content.contains("timed out"));
        thread::sleep(Duration::from_millis(1500));
        assert!(
            !flag.exists(),
            "timeout should kill descendant processes before they can outlive the tool"
        );
    }

    #[test]
    fn test_bash_large_stdout_does_not_false_timeout() {
        let tool = BashTool;
        let ctx = test_ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "command": "python3 -c 'import sys; sys.stdout.write(\"x\" * 200000)'",
                    "timeout": 2_000
                }),
                &ctx,
            )
            .unwrap();

        assert!(
            !result.content.contains("timed out"),
            "fast large-output command should not false-timeout: {}",
            result.content
        );
        assert!(
            result.content.len() >= 200_000,
            "expected captured stdout, got {} bytes",
            result.content.len()
        );
    }

    #[test]
    fn test_bash_timeout_clamped_to_minimum() {
        let tool = BashTool;
        let ctx = test_ctx();
        // Request a 0ms timeout — should be clamped to BASH_MIN_TIMEOUT_MS (100ms)
        // so the command actually gets a chance to run.
        let result = tool
            .execute(
                serde_json::json!({
                    "command": "echo clamped",
                    "timeout": 0
                }),
                &ctx,
            )
            .unwrap();
        assert!(
            result.content.contains("clamped"),
            "command should succeed with clamped timeout, got: {}",
            result.content
        );
    }

    #[test]
    fn test_bash_timeout_clamped_to_maximum() {
        // u64::MAX would effectively disable the timeout — clamp to 300_000ms.
        // We can't wait 300 seconds, so just verify the command runs normally
        // when given an absurdly large timeout value.
        let tool = BashTool;
        let ctx = test_ctx();
        let result = tool
            .execute(
                serde_json::json!({
                    "command": "echo huge_timeout",
                    "timeout": u64::MAX
                }),
                &ctx,
            )
            .unwrap();
        assert!(
            result.content.contains("huge_timeout"),
            "command should succeed even with absurdly large timeout, got: {}",
            result.content
        );
    }

    #[test]
    fn test_bash_schema_includes_timeout_parameter() {
        let tool = BashTool;
        let schema = tool.parameters_schema();
        let props = schema["properties"].as_object().unwrap();
        assert!(
            props.contains_key("timeout"),
            "bash schema should include 'timeout' property, got: {props:?}"
        );
        let timeout_schema = &props["timeout"];
        assert_eq!(timeout_schema["type"], "integer");
        assert!(
            timeout_schema["description"]
                .as_str()
                .unwrap()
                .contains("Timeout"),
            "timeout description should mention 'Timeout', got: {timeout_schema:?}"
        );
        // timeout should NOT be required — it has a default
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(
            !required.contains(&"timeout"),
            "timeout should be optional (has default), got required: {required:?}"
        );
    }

    #[test]
    fn test_bash_description_points_github_repo_urls_to_fetch_repo() {
        let description = BashTool.description();
        assert!(description.contains("fetch_repo"));
        assert!(description.contains("repository URLs"));
        assert!(!description.contains("git clone"));
    }

    #[test]
    fn test_bash_empty_command_rejected() {
        let tool = BashTool;
        let ctx = test_ctx();
        let args = serde_json::json!({"command": ""});
        let result = tool.execute(args, &ctx);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("must not be empty"),
            "expected empty command error, got: {err}"
        );
    }

    #[test]
    fn test_bash_whitespace_command_rejected() {
        let tool = BashTool;
        let ctx = test_ctx();
        let args = serde_json::json!({"command": "   "});
        let result = tool.execute(args, &ctx);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("must not be empty"),
            "expected whitespace command error, got: {err}"
        );
    }
}
