use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{resolve_and_validate_path, Tool, ToolContext, ToolResult};

#[derive(Debug, Serialize, Deserialize)]
struct TodoItem {
    id: String,
    content: String,
    status: String,
}

pub struct TodoWriteTool;

impl Tool for TodoWriteTool {
    fn name(&self) -> &str {
        "todo_write"
    }

    fn description(&self) -> &str {
        "Write a list of todos to .zipcode-todos.json in the current working directory"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string" },
                            "content": { "type": "string" },
                            "status": { "type": "string" }
                        },
                        "required": ["id", "content", "status"]
                    }
                }
            },
            "required": ["todos"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let todos: Vec<TodoItem> =
            serde_json::from_value(args["todos"].clone()).context("Failed to parse todos")?;

        let path = resolve_and_validate_path(".zipcode-todos.json", &ctx.cwd)?;
        let json = serde_json::to_string_pretty(&todos).context("Failed to serialize todos")?;
        std::fs::write(&path, &json)
            .with_context(|| format!("Failed to write todos to {}", path.display()))?;

        Ok(ToolResult::new(format!(
            "Wrote {} todo(s) to {}",
            todos.len(),
            path.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use tempfile::tempdir;

    fn ctx(dir: &std::path::Path) -> ToolContext {
        ToolContext {
            cwd: dir.to_path_buf(),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        }
    }

    #[test]
    fn test_write_todos_creates_file() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        let args = serde_json::json!({
            "todos": [
                { "id": "1", "content": "Do something", "status": "pending" },
                { "id": "2", "content": "Do another thing", "status": "completed" }
            ]
        });
        let result = tool.execute(args, &ctx(dir.path())).unwrap();
        assert!(result.content.contains("2 todo(s)"));

        let file_path = dir.path().join(".zipcode-todos.json");
        assert!(file_path.exists());
    }

    #[test]
    fn test_write_todos_correct_contents() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        let args = serde_json::json!({
            "todos": [
                { "id": "42", "content": "Test content", "status": "in_progress" }
            ]
        });
        tool.execute(args, &ctx(dir.path())).unwrap();

        let file_path = dir.path().join(".zipcode-todos.json");
        let contents = std::fs::read_to_string(file_path).unwrap();
        let parsed: Vec<TodoItem> = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].id, "42");
        assert_eq!(parsed[0].content, "Test content");
        assert_eq!(parsed[0].status, "in_progress");
    }

    #[test]
    fn test_write_empty_todos() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        let args = serde_json::json!({ "todos": [] });
        let result = tool.execute(args, &ctx(dir.path())).unwrap();
        assert!(result.content.contains("0 todo(s)"));

        let file_path = dir.path().join(".zipcode-todos.json");
        let contents = std::fs::read_to_string(file_path).unwrap();
        let parsed: Vec<TodoItem> = serde_json::from_str(&contents).unwrap();
        assert!(parsed.is_empty());
    }
}
