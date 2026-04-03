use std::process::Command;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::{Tool, ToolContext, ToolResult};

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

        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&ctx.cwd)
            .output()
            .context("failed to spawn bash")?;

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
    use std::path::PathBuf;

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
}
