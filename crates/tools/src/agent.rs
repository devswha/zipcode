use anyhow::Result;

use crate::{ChildResult, Tool, ToolContext, ToolResult};

pub struct AgentTool;

impl Tool for AgentTool {
    fn name(&self) -> &'static str {
        "agent"
    }

    fn description(&self) -> &'static str {
        "Delegate a sub-task to a child agent. The child runs with a filtered tool set and downgraded permissions, then returns a summary."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task prompt to send to the child agent"
                },
                "skill": {
                    "type": "string",
                    "description": "Optional skill name to apply"
                },
                "tool_allowlist": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional list of tool names the child may use. 'agent' is always removed."
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Optional token budget hint for the child agent"
                }
            },
            "required": ["task"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let task = args
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required field 'task'"))?
            .to_string();

        if ctx.depth >= 2 {
            return Ok(ToolResult::error(
                "max agent depth reached; sub-agent delegation is not allowed here",
            ));
        }

        let Some(spawn_fn) = ctx.spawn_child.as_ref() else {
            return Ok(ToolResult::error(
                "spawn callback unavailable; agent delegation requires a full runtime context",
            ));
        };

        // Build allowlist, stripping "agent" to prevent circular delegation.
        let raw_allowlist: Option<Vec<String>> = args
            .get("tool_allowlist")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .filter(|name| *name != "agent")
                    .map(String::from)
                    .collect()
            });

        let allowlist_ref: Option<&[String]> = raw_allowlist.as_deref();

        // False positive: token counts always fit in usize.
        #[allow(clippy::cast_possible_truncation)]
        let max_tokens: Option<usize> = match args.get("max_tokens") {
            Some(v) => match v.as_u64() {
                Some(n) => Some(n as usize),
                None => {
                    return Ok(ToolResult::error(
                        "'max_tokens' must be a non-negative integer",
                    ));
                }
            },
            None => None,
        };

        let result: ChildResult = spawn_fn(&task, allowlist_ref, None, max_tokens)?;

        Ok(ToolResult::new(format!(
            "Child agent complete.\nSummary: {}\nTool calls: {}",
            result.summary, result.tool_call_count
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{PermissionMode, SpawnChildFn};
    use std::path::PathBuf;

    fn ctx_no_callback(depth: u32) -> ToolContext {
        ToolContext {
            cwd: PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
            parent_session_id: None,
            depth,
            budget_tokens: None,
            spawn_child: None,
        }
    }

    fn ctx_with_callback(depth: u32, cb: Arc<SpawnChildFn>) -> ToolContext {
        ToolContext {
            cwd: PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
            parent_session_id: None,
            depth,
            budget_tokens: None,
            spawn_child: Some(cb),
        }
    }

    fn ok_callback() -> Arc<SpawnChildFn> {
        Arc::new(|task, _allowlist, _perm, _tokens| {
            Ok(ChildResult {
                summary: format!("did: {task}"),
                tool_call_count: 3,
                child_session_id: "child-session-abc".to_string(),
            })
        })
    }

    #[test]
    fn test_depth_reject() {
        let tool = AgentTool;
        let ctx = ctx_no_callback(2);
        let result = tool
            .execute(serde_json::json!({ "task": "do something" }), &ctx)
            .unwrap();
        assert!(
            result.content.contains("max agent depth"),
            "expected depth error, got: {}",
            result.content
        );
    }

    #[test]
    fn test_no_callback_reject() {
        let tool = AgentTool;
        let ctx = ctx_no_callback(0);
        let result = tool
            .execute(serde_json::json!({ "task": "do something" }), &ctx)
            .unwrap();
        assert!(
            result.content.contains("spawn callback unavailable"),
            "expected callback error, got: {}",
            result.content
        );
    }

    #[test]
    fn test_strips_agent_from_allowlist() {
        let received_allowlist: Arc<std::sync::Mutex<Option<Vec<String>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let received_clone = Arc::clone(&received_allowlist);

        let cb: Arc<SpawnChildFn> = Arc::new(move |_task, allowlist, _perm, _tokens| {
            *received_clone.lock().unwrap() = allowlist.map(<[std::string::String]>::to_vec);
            Ok(ChildResult {
                summary: "ok".to_string(),
                tool_call_count: 0,
                child_session_id: "s".to_string(),
            })
        });

        let tool = AgentTool;
        let ctx = ctx_with_callback(0, cb);
        tool.execute(
            serde_json::json!({
                "task": "test",
                "tool_allowlist": ["read_file", "agent", "grep_search"]
            }),
            &ctx,
        )
        .unwrap();

        let list = received_allowlist.lock().unwrap();
        let list = list.as_ref().unwrap();
        assert!(
            !list.contains(&"agent".to_string()),
            "agent should be stripped"
        );
        assert!(list.contains(&"read_file".to_string()));
        assert!(list.contains(&"grep_search".to_string()));
    }

    #[test]
    fn test_forwards_task() {
        let received_task: Arc<std::sync::Mutex<String>> =
            Arc::new(std::sync::Mutex::new(String::new()));
        let received_clone = Arc::clone(&received_task);

        let cb: Arc<SpawnChildFn> = Arc::new(move |task, _al, _perm, _tok| {
            *received_clone.lock().unwrap() = task.to_string();
            Ok(ChildResult {
                summary: "done".to_string(),
                tool_call_count: 0,
                child_session_id: "s".to_string(),
            })
        });

        let tool = AgentTool;
        let ctx = ctx_with_callback(0, cb);
        tool.execute(serde_json::json!({ "task": "write some tests" }), &ctx)
            .unwrap();

        assert_eq!(*received_task.lock().unwrap(), "write some tests");
    }

    #[test]
    fn test_formats_result() {
        let tool = AgentTool;
        let ctx = ctx_with_callback(0, ok_callback());
        let result = tool
            .execute(serde_json::json!({ "task": "review code" }), &ctx)
            .unwrap();

        assert!(result.content.contains("Child agent complete."));
        assert!(result.content.contains("did: review code"));
        assert!(result.content.contains("Tool calls: 3"));
    }

    #[test]
    fn test_missing_task_error() {
        let tool = AgentTool;
        let ctx = ctx_with_callback(0, ok_callback());
        let result = tool.execute(serde_json::json!({}), &ctx);
        assert!(result.is_err() || result.unwrap().content.contains("missing"));
    }

    #[test]
    fn test_invalid_max_tokens_error() {
        let tool = AgentTool;
        let ctx = ctx_with_callback(0, ok_callback());
        let result = tool
            .execute(
                serde_json::json!({ "task": "something", "max_tokens": -1 }),
                &ctx,
            )
            .unwrap();
        assert!(
            result.content.contains("max_tokens"),
            "expected max_tokens error, got: {}",
            result.content
        );
    }
}
