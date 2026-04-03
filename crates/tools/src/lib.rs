use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod agent;
pub mod bash;
pub mod edit_file;
pub mod glob_search;
pub mod grep_search;
pub mod read_file;
pub mod repl;
pub mod todo_write;
pub mod tool_search;
pub mod write_file;

/// Permission modes for tool execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    ReadOnly,
    WorkspaceWrite,
    FullAccess,
}

/// Context passed to every tool execution
pub struct ToolContext {
    pub cwd: PathBuf,
    pub permission: PermissionMode,
    pub session_id: String,
}

/// Result from a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: String,
    pub truncated: bool,
}

impl ToolResult {
    pub fn new(content: String) -> Self {
        Self {
            content,
            truncated: false,
        }
    }

    pub fn error(msg: String) -> Self {
        Self {
            content: format!("Error: {msg}"),
            truncated: false,
        }
    }

    pub fn truncate(self, max_bytes: usize) -> Self {
        if self.content.len() <= max_bytes {
            return self;
        }
        // Find a safe char boundary at or before max_bytes
        let mut end = max_bytes;
        while end > 0 && !self.content.is_char_boundary(end) {
            end -= 1;
        }
        let truncated_content = format!(
            "{}\n\n[truncated: showing first {} bytes of {}]",
            &self.content[..end],
            end,
            self.content.len()
        );
        Self {
            content: truncated_content,
            truncated: true,
        }
    }
}

/// Spec for a single tool — used to inject tool schemas into the model prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Trait every tool must implement
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult>;
}

/// Registry holding all available tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(AsRef::as_ref)
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_TOOL_OUTPUT_BYTES: usize = 8192;

/// Execute a tool by name with automatic truncation
pub fn execute_tool(
    registry: &ToolRegistry,
    name: &str,
    args: serde_json::Value,
    ctx: &ToolContext,
) -> Result<ToolResult> {
    let tool = registry
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown tool: {name}"))?;
    let result = tool.execute(args, ctx)?;
    Ok(result.truncate(MAX_TOOL_OUTPUT_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes input"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                },
                "required": ["text"]
            })
        }
        fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<ToolResult> {
            let text = args["text"].as_str().unwrap_or("");
            Ok(ToolResult::new(text.to_string()))
        }
    }

    fn test_ctx() -> ToolContext {
        ToolContext {
            cwd: std::path::PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        }
    }

    #[test]
    fn test_registry_add_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        assert!(registry.get("echo").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_specs() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let specs = registry.specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "echo");
    }

    #[test]
    fn test_tool_result_truncation() {
        let long_content = "x".repeat(10_000);
        let result = ToolResult::new(long_content);
        let truncated = result.truncate(8192);
        assert!(truncated.content.len() <= 8192 + 100);
        assert!(truncated.truncated);
    }

    #[test]
    fn test_tool_execution() {
        let tool = EchoTool;
        let ctx = test_ctx();
        let args = serde_json::json!({"text": "hello"});
        let result = tool.execute(args, &ctx).unwrap();
        assert_eq!(result.content, "hello");
    }

    #[test]
    fn test_execute_unknown_tool() {
        let registry = ToolRegistry::new();
        let ctx = test_ctx();
        let result = execute_tool(&registry, "unknown", serde_json::json!({}), &ctx);
        assert!(result.is_err());
    }
}
