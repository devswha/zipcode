use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;

use crate::{Tool, ToolContext, ToolResult};

pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a file with line numbers. Supports offset and limit for partial reads."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative paths resolved against cwd)"
                },
                "offset": {
                    "type": "integer",
                    "description": "0-based line number to start reading from (default: 0)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: all)"
                }
            },
            "required": ["path"]
        })
    }

    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: path"))?;

        let path = crate::resolve_and_validate_path(path_str, &ctx.cwd)?;

        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read file: {}", path.display()))?;

        let offset = args["offset"].as_u64().unwrap_or(0) as usize;
        let limit = args["limit"].as_u64().map(|v| v as usize);

        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();

        let start = offset.min(total);
        let end = match limit {
            Some(n) => (start + n).min(total),
            None => total,
        };

        let output: String = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{}\t{}", start + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ToolResult::new(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn ctx() -> ToolContext {
        ToolContext {
            cwd: std::path::PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        }
    }

    #[test]
    fn test_read_existing_file() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "line one").unwrap();
        writeln!(f, "line two").unwrap();
        writeln!(f, "line three").unwrap();

        let tool = ReadFileTool;
        let args = serde_json::json!({ "path": f.path().to_str().unwrap() });
        let result = tool.execute(args, &ctx()).unwrap();

        assert!(result.content.contains("1\tline one"));
        assert!(result.content.contains("2\tline two"));
        assert!(result.content.contains("3\tline three"));
    }

    #[test]
    fn test_read_with_offset_and_limit() {
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=5 {
            writeln!(f, "line {i}").unwrap();
        }

        let tool = ReadFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "offset": 1,
            "limit": 2
        });
        let result = tool.execute(args, &ctx()).unwrap();

        // offset=1 means skip first line; lines 2 and 3 returned
        assert!(!result.content.contains("1\tline 1"));
        assert!(result.content.contains("2\tline 2"));
        assert!(result.content.contains("3\tline 3"));
        assert!(!result.content.contains("4\tline 4"));
    }

    #[test]
    fn test_read_nonexistent_file() {
        let tool = ReadFileTool;
        let args = serde_json::json!({ "path": "/tmp/does_not_exist_zipcode_test.txt" });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err());
    }
}
