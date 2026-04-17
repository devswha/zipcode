use std::process::Command;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::{wait_with_output_timeout, Tool, ToolContext, ToolResult};

const BASH_DEFAULT_TIMEOUT_MS: u64 = 120_000;

pub struct BashTool;

impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its output"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let command = args["command"]
            .as_str()
            .context("missing 'command' argument")?;

        let timeout_ms = args["timeout"].as_u64().unwrap_or(BASH_DEFAULT_TIMEOUT_MS);
        let timeout = std::time::Duration::from_millis(timeout_ms);

        let child = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&ctx.cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to execute: {command}"))?;

        let output = match wait_with_output_timeout(child, timeout)? {
            Some(output) => output,
            None => {
                return Ok(ToolResult::new(format!(
                    "Error: command timed out after {timeout_ms}ms"
                )));
            }
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
            result.push_str(&format!("Exit code: {code}"));
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
}
