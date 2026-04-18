use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{resolve_and_validate_path, Tool, ToolContext, ToolResult};

/// Allowed status values for todo items.
const VALID_STATUSES: &[&str] = &["pending", "in_progress", "completed", "cancelled"];

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

        // Validate each todo item
        for (i, todo) in todos.iter().enumerate() {
            validate_todo_item(todo, i)?;
        }

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

/// Validate a single todo item, returning a descriptive error on failure.
///
/// The `index` parameter is used in error messages to help the caller
/// identify which item in the batch failed validation.
fn validate_todo_item(item: &TodoItem, index: usize) -> Result<()> {
    if item.id.trim().is_empty() {
        anyhow::bail!("todo item at index {index}: 'id' must be a non-empty string");
    }

    if item.content.trim().is_empty() {
        anyhow::bail!("todo item at index {index}: 'content' must be a non-empty string");
    }

    if !VALID_STATUSES.contains(&item.status.as_str()) {
        anyhow::bail!(
            "todo item at index {index}: 'status' must be one of {:?}, got {:?}",
            VALID_STATUSES,
            item.status
        );
    }

    Ok(())
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

    // --- Original tests (preserved) ---

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

    // --- New validation tests ---

    #[test]
    fn test_write_todos_empty_id_rejected() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        let args = serde_json::json!({
            "todos": [
                { "id": "", "content": "Do something", "status": "pending" }
            ]
        });
        let err = tool.execute(args, &ctx(dir.path())).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("index 0"), "expected index in error: {msg}");
        assert!(
            msg.contains("'id' must be a non-empty string"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn test_write_todos_whitespace_id_rejected() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        let args = serde_json::json!({
            "todos": [
                { "id": "  \t ", "content": "Do something", "status": "pending" }
            ]
        });
        let err = tool.execute(args, &ctx(dir.path())).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("'id' must be a non-empty string"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn test_write_todos_invalid_status_rejected() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        let args = serde_json::json!({
            "todos": [
                { "id": "1", "content": "Do something", "status": "unknown_status" }
            ]
        });
        let err = tool.execute(args, &ctx(dir.path())).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("'status' must be one of"), "unexpected: {msg}");
        assert!(
            msg.contains("unknown_status"),
            "expected offending value in error: {msg}"
        );
    }

    #[test]
    fn test_write_todos_empty_content_rejected() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        let args = serde_json::json!({
            "todos": [
                { "id": "1", "content": "", "status": "pending" }
            ]
        });
        let err = tool.execute(args, &ctx(dir.path())).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("'content' must be a non-empty string"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn test_write_todos_whitespace_content_rejected() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        let args = serde_json::json!({
            "todos": [
                { "id": "1", "content": "  \n  ", "status": "pending" }
            ]
        });
        let err = tool.execute(args, &ctx(dir.path())).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("'content' must be a non-empty string"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn test_write_todos_valid_statuses_accepted() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        let args = serde_json::json!({
            "todos": [
                { "id": "a", "content": "Task A", "status": "pending" },
                { "id": "b", "content": "Task B", "status": "in_progress" },
                { "id": "c", "content": "Task C", "status": "completed" },
                { "id": "d", "content": "Task D", "status": "cancelled" }
            ]
        });
        let result = tool.execute(args, &ctx(dir.path())).unwrap();
        assert!(result.content.contains("4 todo(s)"));

        let file_path = dir.path().join(".zipcode-todos.json");
        let contents = std::fs::read_to_string(file_path).unwrap();
        let parsed: Vec<TodoItem> = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed[0].status, "pending");
        assert_eq!(parsed[1].status, "in_progress");
        assert_eq!(parsed[2].status, "completed");
        assert_eq!(parsed[3].status, "cancelled");
    }

    #[test]
    fn test_write_todos_mixed_valid_invalid_rejected() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        // Second item has invalid status — entire batch should fail
        let args = serde_json::json!({
            "todos": [
                { "id": "1", "content": "Valid task", "status": "pending" },
                { "id": "2", "content": "Bad task", "status": "done" }
            ]
        });
        let err = tool.execute(args, &ctx(dir.path())).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("index 1"), "expected index 1 in error: {msg}");
        assert!(msg.contains("'status' must be one of"), "unexpected: {msg}");

        // File should NOT have been written
        let file_path = dir.path().join(".zipcode-todos.json");
        assert!(
            !file_path.exists(),
            "file must not be created on validation failure"
        );
    }

    #[test]
    fn test_write_todos_missing_required_field() {
        let dir = tempdir().unwrap();
        let tool = TodoWriteTool;
        // Missing 'status' field — serde deserialization should fail
        let args = serde_json::json!({
            "todos": [
                { "id": "1", "content": "No status" }
            ]
        });
        let err = tool.execute(args, &ctx(dir.path())).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Failed to parse todos"), "unexpected: {msg}");
    }

    #[test]
    fn test_valid_statuses_constant_matches_schema() {
        // Ensure VALID_STATUSES covers exactly the values the tool expects
        assert_eq!(
            VALID_STATUSES,
            &["pending", "in_progress", "completed", "cancelled"]
        );
    }
}
