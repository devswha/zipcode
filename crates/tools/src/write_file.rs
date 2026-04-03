use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;

use crate::{Tool, ToolContext, ToolResult};

pub struct WriteFileTool;

impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file, creating parent directories if needed."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative paths resolved against cwd)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: path"))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: content"))?;

        let path = crate::resolve_and_validate_path(path_str, &ctx.cwd)?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directories for: {}", path.display()))?;
        }

        fs::write(&path, content)
            .with_context(|| format!("failed to write file: {}", path.display()))?;

        Ok(ToolResult::new(format!("wrote {}", path.display())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use tempfile::TempDir;

    fn ctx(dir: &TempDir) -> ToolContext {
        ToolContext {
            cwd: dir.path().to_path_buf(),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        }
    }

    #[test]
    fn test_write_new_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;
        let path = dir.path().join("hello.txt");

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "hello world"
        });
        let result = tool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("wrote"));

        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, "hello world");
    }

    #[test]
    fn test_write_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;
        let path = dir.path().join("a").join("b").join("c.txt");

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "nested"
        });
        let result = tool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("wrote"));

        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, "nested");
    }
}
