use anyhow::Result;

use crate::{Tool, ToolContext, ToolResult};

pub struct ToolSearchTool {
    specs: Vec<(String, String)>,
}

impl ToolSearchTool {
    #[must_use]
    pub const fn from_specs(specs: Vec<(String, String)>) -> Self {
        Self { specs }
    }

    #[must_use]
    pub fn from_registry(registry: &crate::ToolRegistry) -> Self {
        let specs = registry
            .specs()
            .into_iter()
            .map(|s| (s.name, s.description))
            .collect();
        Self { specs }
    }
}

impl Tool for ToolSearchTool {
    fn name(&self) -> &'static str {
        "tool_search"
    }

    fn description(&self) -> &'static str {
        "Search available tools by name or description keyword"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Keyword to search in tool names and descriptions" }
            },
            "required": ["query"]
        })
    }

    fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: query"))?;

        if query.trim().is_empty() {
            anyhow::bail!("query must not be empty — provide a keyword to search for");
        }

        let query = query.trim().to_lowercase();

        let matches: Vec<String> = self
            .specs
            .iter()
            .filter(|(name, desc)| {
                name.to_lowercase().contains(&query) || desc.to_lowercase().contains(&query)
            })
            .map(|(name, desc)| format!("- {name}: {desc}"))
            .collect();

        if matches.is_empty() {
            Ok(ToolResult::new(format!(
                "No tools found matching '{query}'"
            )))
        } else {
            Ok(ToolResult::new(matches.join("\n")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use std::path::PathBuf;

    fn ctx() -> ToolContext {
        ToolContext {
            cwd: PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
            parent_session_id: None,
            depth: 0,
            budget_tokens: None,
            spawn_child: None,
        }
    }

    fn make_tool() -> ToolSearchTool {
        ToolSearchTool::from_specs(vec![
            (
                "read_file".to_string(),
                "Read a file from the filesystem".to_string(),
            ),
            (
                "write_file".to_string(),
                "Write content to a file".to_string(),
            ),
            ("bash".to_string(), "Execute a shell command".to_string()),
            (
                "grep_search".to_string(),
                "Search file contents with regex".to_string(),
            ),
        ])
    }

    #[test]
    fn test_search_by_name() {
        let tool = make_tool();
        let result = tool
            .execute(serde_json::json!({ "query": "bash" }), &ctx())
            .unwrap();
        assert!(result.content.contains("- bash:"));
        assert!(!result.content.contains("read_file"));
    }

    #[test]
    fn test_search_by_description() {
        let tool = make_tool();
        let result = tool
            .execute(serde_json::json!({ "query": "regex" }), &ctx())
            .unwrap();
        assert!(result.content.contains("- grep_search:"));
        assert!(!result.content.contains("bash"));
    }

    #[test]
    fn test_search_case_insensitive() {
        let tool = make_tool();
        let result = tool
            .execute(serde_json::json!({ "query": "FILE" }), &ctx())
            .unwrap();
        assert!(result.content.contains("read_file"));
        assert!(result.content.contains("write_file"));
    }

    #[test]
    fn test_search_no_results() {
        let tool = make_tool();
        let result = tool
            .execute(serde_json::json!({ "query": "nonexistent_xyz" }), &ctx())
            .unwrap();
        assert!(result.content.contains("No tools found"));
    }

    #[test]
    fn test_search_empty_query_rejected() {
        let tool = make_tool();
        let err = tool
            .execute(serde_json::json!({ "query": "" }), &ctx())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("empty"),
            "empty query should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_search_whitespace_only_query_rejected() {
        let tool = make_tool();
        let err = tool
            .execute(serde_json::json!({ "query": "   " }), &ctx())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("empty"),
            "whitespace-only query should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_search_missing_query_parameter_rejected() {
        let tool = make_tool();
        let err = tool
            .execute(serde_json::json!({}), &ctx())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("missing required parameter"),
            "missing query param should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_search_trimmed_query_works() {
        let tool = make_tool();
        let result = tool
            .execute(serde_json::json!({ "query": "  bash  " }), &ctx())
            .unwrap();
        assert!(result.content.contains("- bash:"));
    }
}
