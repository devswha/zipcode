use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;

use crate::{Tool, ToolContext, ToolResult};

pub struct EditFileTool;

impl Tool for EditFileTool {
    fn name(&self) -> &'static str {
        "edit_file"
    }

    fn description(&self) -> &'static str {
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

        let path = crate::resolve_and_validate_path(path_str, &ctx.cwd)?;

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
            parent_session_id: None,
            depth: 0,
            budget_tokens: None,
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

    #[test]
    fn test_empty_old_string_matches_everywhere() {
        // Empty old_string matches at every position in the content.
        // `content.matches("")` returns count = len + 1 (including trailing
        // boundary), so this should always exceed 1 and be rejected as
        // non-unique, even for a one-character file.
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "x").unwrap();

        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "",
            "new_string": "y"
        });
        let result = tool.execute(args, &ctx());
        assert!(
            result.is_err(),
            "empty old_string should fail because it matches multiple times"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("times") || msg.contains("unique"),
            "error message should mention multiple matches, got: {msg}"
        );
    }

    #[test]
    fn test_identical_old_and_new_string_is_noop() {
        let mut f = NamedTempFile::new().unwrap();
        let original = "unchanged content\n";
        write!(f, "{original}").unwrap();

        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "unchanged content",
            "new_string": "unchanged content"
        });
        let result = tool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("edited"));

        let after = std::fs::read_to_string(f.path()).unwrap();
        assert_eq!(
            after, original,
            "file content should be identical after same-string replacement"
        );
    }

    #[test]
    fn test_multiline_replacement() {
        let mut f = NamedTempFile::new().unwrap();
        let original = "fn main() {\n    println!(\"hello\");\n}\n";
        write!(f, "{original}").unwrap();

        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "    println!(\"hello\");",
            "new_string": "    println!(\"world\");\n    println!(\"done\");"
        });
        let result = tool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("edited"));

        let after = std::fs::read_to_string(f.path()).unwrap();
        assert!(after.contains("world"));
        assert!(after.contains("done"));
        assert!(!after.contains("hello"));
    }

    #[test]
    fn test_nonexistent_file_path() {
        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": "/tmp/zipcode_test_nonexistent_abcdef123.rs",
            "old_string": "foo",
            "new_string": "bar"
        });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err(), "editing a nonexistent file should fail");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("failed to read file") || msg.contains("No such file"),
            "error message should mention file read failure, got: {msg}"
        );
    }

    #[test]
    fn test_path_traversal_rejected() {
        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": "../../etc/passwd",
            "old_string": "root",
            "new_string": "blocked"
        });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err(), "path traversal attempt should be rejected");
    }

    #[test]
    fn test_missing_path_parameter() {
        let tool = EditFileTool;
        let args = serde_json::json!({
            "old_string": "foo",
            "new_string": "bar"
        });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("missing required parameter: path"),
            "expected missing path error, got: {msg}"
        );
    }

    #[test]
    fn test_missing_old_string_parameter() {
        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": "/tmp/dummy.txt",
            "new_string": "bar"
        });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("missing required parameter: old_string"),
            "expected missing old_string error, got: {msg}"
        );
    }

    #[test]
    fn test_missing_new_string_parameter() {
        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": "/tmp/dummy.txt",
            "old_string": "foo"
        });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("missing required parameter: new_string"),
            "expected missing new_string error, got: {msg}"
        );
    }

    #[test]
    fn test_delete_content_via_empty_new_string() {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "keep this\nremove this\nkeep this too\n").unwrap();

        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "remove this\n",
            "new_string": ""
        });
        let result = tool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("edited"));

        let content = std::fs::read_to_string(f.path()).unwrap();
        assert_eq!(content, "keep this\nkeep this too\n");
    }

    #[test]
    fn test_edit_whitespace_only_file() {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "   \n   \n").unwrap();

        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "   \n",
            "new_string": "content\n"
        });
        // This should fail because "   \n" appears twice
        let result = tool.execute(args, &ctx());
        assert!(
            result.is_err(),
            "whitespace-only old_string matching multiple times should fail"
        );
    }

    #[test]
    fn test_edit_single_character_replacement() {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "x + y = z").unwrap();

        let tool = EditFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "+",
            "new_string": "-"
        });
        let result = tool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("edited"));

        let content = std::fs::read_to_string(f.path()).unwrap();
        assert_eq!(content, "x - y = z");
    }

    #[test]
    fn test_tool_trait_interface() {
        let tool = EditFileTool;
        assert_eq!(tool.name(), "edit_file");
        assert!(!tool.description().is_empty());

        let schema = tool.parameters_schema();
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"path"));
        assert!(required.contains(&"old_string"));
        assert!(required.contains(&"new_string"));
    }
}
