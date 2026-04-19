use anyhow::Result;

use crate::{Tool, ToolContext, ToolResult};

pub struct AgentTool;

impl Tool for AgentTool {
    fn name(&self) -> &str {
        "agent"
    }

    fn description(&self) -> &str {
        "Delegate a task to a sub-agent (not yet implemented)"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to delegate to the agent"
                }
            },
            "required": ["task"]
        })
    }

    fn execute(&self, _args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
        Ok(ToolResult::new(
            "Agent delegation is not yet implemented in this version.".to_string(),
        ))
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
        }
    }

    #[test]
    fn test_name() {
        let tool = AgentTool;
        assert_eq!(tool.name(), "agent");
    }

    #[test]
    fn test_description_is_non_empty() {
        let tool = AgentTool;
        assert!(!tool.description().is_empty());
        assert!(tool.description().contains("sub-agent"));
    }

    #[test]
    fn test_parameters_schema_is_valid_json() {
        let tool = AgentTool;
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["task"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "task"));
    }

    #[test]
    fn test_execute_returns_stub_message() {
        let tool = AgentTool;
        let args = serde_json::json!({ "task": "do something" });
        let result = tool.execute(args, &ctx()).unwrap();
        assert!(!result.truncated);
        assert!(result.content.contains("not yet implemented"));
    }

    #[test]
    fn test_execute_ignores_args() {
        let tool = AgentTool;
        // Execute with empty args — should still succeed with the stub message
        let result = tool.execute(serde_json::json!({}), &ctx()).unwrap();
        assert!(result.content.contains("not yet implemented"));
    }
}
