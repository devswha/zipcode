use anyhow::{Context, Result};

use crate::{Tool, ToolContext, ToolResult};

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
                }
            },
            "required": ["language", "code"]
        })
    }

    fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
        let language = args["language"].as_str().unwrap_or("").to_string();
        let code = args["code"].as_str().unwrap_or("").to_string();

        let (program, flag) = match language.as_str() {
            "python" => ("python3", "-c"),
            "node" => ("node", "-e"),
            other => {
                return Ok(ToolResult::error(format!(
                    "Unsupported language: '{other}'. Use 'python' or 'node'."
                )));
            }
        };

        let output = std::process::Command::new(program)
            .arg(flag)
            .arg(&code)
            .output()
            .with_context(|| format!("Failed to execute {program}"))?;

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

        if result.is_empty() {
            result = "(no output)".to_string();
        }

        Ok(ToolResult::new(result))
    }
}
