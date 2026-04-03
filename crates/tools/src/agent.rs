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
