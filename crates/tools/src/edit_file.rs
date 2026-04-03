use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;

use crate::{Tool, ToolContext, ToolResult};

pub struct EditFileTool;

impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace a unique occurrence of old_string with new_string in a file. Fails if old_string is not found or appears more than once."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative paths resolved against cwd)"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace (must appear exactly once)"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string"
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: path"))?;
        let old_string = args["old_string"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: old_string"))?;
        let new_string = args["new_string"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: new_string"))?;

        let path = if std::path::Path::new(path_str).is_absolute() {
            std::path::PathBuf::from(path_str)
        } else {
            ctx.cwd.join(path_str)
        };

        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read file: {}", path.display()))?;

        let matches = content.matches(old_string).count();
        if matches == 0 {
            anyhow::bail!("old_string not found in {}", path.display());
        }
        if matches > 1 {
            anyhow::bail!(
                "old_string found {} times in {} — must be unique",
                matches,
                path.display()
            );
        }

        let new_content = content.replacen(old_string, new_string, 1);
        fs::write(&path, &new_content)
            .with_context(|| format!("failed to write file: {}", path.display()))?;

        Ok(ToolResult::new(format!("edited {}", path.display())))
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
    fn test_replace_unique_string() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "hello world").unwrap();
        writeln!(f, "foo bar").unwrap();

        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "hello world",
            "new_string": "goodbye world"
        });
        let result = tool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("edited"));

        let content = std::fs::read_to_string(f.path()).unwrap();
        assert!(content.contains("goodbye world"));
        assert!(!content.contains("hello world"));
    }

    #[test]
    fn test_fail_on_not_found() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "some content").unwrap();

        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "nonexistent string",
            "new_string": "replacement"
        });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_fail_on_duplicate() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "dup dup").unwrap();

        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "dup",
            "new_string": "unique"
        });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("2 times") || msg.contains("unique"));
    }
}
