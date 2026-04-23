use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;

use crate::{Tool, ToolContext, ToolResult};

pub struct WriteFileTool;

impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
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
            parent_session_id: None,
            depth: 0,
            budget_tokens: None,
            spawn_child: None,
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

    #[test]
    fn test_overwrite_existing_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;
        let path = dir.path().join("replace.txt");

        // Write initial content
        std::fs::write(&path, "old content").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "old content");

        // Overwrite with new content
        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "new content"
        });
        let result = tool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("wrote"));

        // Verify old content is fully replaced
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, "new content");
        assert!(!written.contains("old"));
    }

    #[test]
    fn test_write_empty_content() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;
        let path = dir.path().join("empty.txt");

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": ""
        });
        let result = tool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("wrote"));

        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, "");
    }

    #[test]
    fn test_missing_path_parameter() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;

        let args = serde_json::json!({
            "content": "hello"
        });
        let err = tool.execute(args, &ctx(&dir)).unwrap_err();
        assert!(
            err.to_string().contains("missing required parameter: path"),
            "expected missing path error, got: {err}"
        );
    }

    #[test]
    fn test_missing_content_parameter() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;

        let args = serde_json::json!({
            "path": "test.txt"
        });
        let err = tool.execute(args, &ctx(&dir)).unwrap_err();
        assert!(
            err.to_string()
                .contains("missing required parameter: content"),
            "expected missing content error, got: {err}"
        );
    }

    #[test]
    fn test_path_traversal_blocked() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;

        let args = serde_json::json!({
            "path": "../escape.txt",
            "content": "escaping"
        });
        let err = tool.execute(args, &ctx(&dir)).unwrap_err();
        assert!(
            err.to_string().contains("outside the workspace"),
            "expected path traversal error, got: {err}"
        );
    }

    #[test]
    fn test_tool_trait_interface() {
        let tool = WriteFileTool;
        assert_eq!(tool.name(), "write_file");
        assert!(!tool.description().is_empty());

        let schema = tool.parameters_schema();
        let props = schema["properties"].as_object().unwrap();
        assert!(props.contains_key("path"), "schema missing 'path' property");
        assert!(
            props.contains_key("content"),
            "schema missing 'content' property"
        );

        let required = schema["required"].as_array().unwrap();
        let required_names: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(required_names.contains(&"path"));
        assert!(required_names.contains(&"content"));
    }

    #[test]
    fn test_write_unicode_content() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;
        let path = dir.path().join("unicode.txt");

        let unicode_content = "안녕하세요 🌍 Привет мир こんにちは";
        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": unicode_content
        });
        let result = tool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("wrote"));

        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, unicode_content);
    }

    #[test]
    fn test_write_large_content_succeeds() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;

        // Verify the tool handles content larger than the 8KB truncation limit
        // The truncation is applied to tool *output*, not to what's written to disk.
        let large_content = "x".repeat(10_000);
        let path = dir.path().join("large.txt");

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": large_content
        });
        let result = tool.execute(args, &ctx(&dir)).unwrap();
        // The file-system write succeeds; truncation only applies at the registry layer
        assert!(!result.truncated);
        assert!(result.content.contains("wrote"));

        // Full content is on disk
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written.len(), 10_000);
    }

    #[test]
    fn test_write_relative_path() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;

        // Create the subdirectory first so resolve_and_validate_path can canonicalize
        std::fs::create_dir_all(dir.path().join("subdir")).unwrap();

        let args = serde_json::json!({
            "path": "subdir/relative.txt",
            "content": "relative path content"
        });
        let result = tool.execute(args, &ctx(&dir)).unwrap();
        assert!(result.content.contains("wrote"));

        let written = std::fs::read_to_string(dir.path().join("subdir/relative.txt")).unwrap();
        assert_eq!(written, "relative path content");
    }
}
