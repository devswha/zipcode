use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use zipcode_tools::{Tool, ToolContext, ToolResult};

use crate::skills::{Skill, SkillRegistry};

pub struct SkillTool {
    pub registry: Arc<SkillRegistry>,
}

impl Tool for SkillTool {
    fn name(&self) -> &'static str {
        "skill"
    }

    fn description(&self) -> &'static str {
        "Invoke a named skill as a child agent with rendered instructions"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name from the catalog"
                },
                "params": {
                    "type": "object",
                    "description": "Parameter substitutions for {{ key }} placeholders",
                    "additionalProperties": { "type": "string" }
                }
            },
            "required": ["name"]
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolResult> {
        let name = match args.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => return Ok(ToolResult::error("missing required field 'name'")),
        };

        let skill: &Skill = if let Some(s) = self.registry.get(&name) {
            s
        } else {
            let mut available = self.registry.names();
            available.sort_unstable();
            let list = available.join(", ");
            return Ok(ToolResult::error(&format!(
                "skill not found: '{name}'. available: [{list}]"
            )));
        };

        let params: HashMap<String, String> = args.get("params").map_or_else(HashMap::new, |p| {
            p.as_object().map_or_else(HashMap::new, |obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
        });

        // A parameter without a default is required: missing it would silently
        // render an empty string into the child's task prompt, which produces a
        // broken instruction ("Focus on .") rather than a clear failure.
        let missing: Vec<&str> = skill
            .parameters
            .iter()
            .filter(|p| p.default.is_none() && !params.contains_key(&p.name))
            .map(|p| p.name.as_str())
            .collect();
        if !missing.is_empty() {
            return Ok(ToolResult::error(&format!(
                "skill '{}' missing required parameters: [{}]",
                name,
                missing.join(", ")
            )));
        }

        let rendered_task = skill.render(&params);

        let Some(spawn_fn) = ctx.spawn_child.as_ref() else {
            return Ok(ToolResult::error(
                "spawn callback unavailable; skill invocation requires a full runtime context",
            ));
        };

        let allowlist: Option<&[String]> = if skill.tool_allowlist.is_empty() {
            None
        } else {
            Some(&skill.tool_allowlist)
        };

        let result = spawn_fn(&rendered_task, allowlist, None, None)?;

        Ok(ToolResult::new(format!(
            "Skill '{}' complete.\nSummary: {}\nTool calls: {}",
            name, result.summary, result.tool_call_count
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use zipcode_tools::{ChildResult, PermissionMode, SpawnChildFn};

    fn make_registry_with_skill(
        name: &str,
        body: &str,
        allowlist: Vec<String>,
    ) -> Arc<SkillRegistry> {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let al_yaml: String = if allowlist.is_empty() {
            String::new()
        } else {
            format!(
                "tool_allowlist:\n{}\n",
                allowlist
                    .iter()
                    .map(|t| format!("  - {t}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        let content = format!("---\nname: {name}\ndescription: Test skill\n{al_yaml}---\n{body}\n");
        let path = dir.path().join(format!("{name}.md"));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();

        let registry = SkillRegistry::load_from(dir.path()).unwrap();
        std::mem::forget(dir);
        Arc::new(registry)
    }

    fn ctx_no_callback() -> ToolContext {
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

    fn ctx_with_callback(cb: Arc<SpawnChildFn>) -> ToolContext {
        ToolContext {
            cwd: PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
            parent_session_id: None,
            depth: 0,
            budget_tokens: None,
            spawn_child: Some(cb),
        }
    }

    #[allow(dead_code)]
    fn ok_callback() -> Arc<SpawnChildFn> {
        Arc::new(|task, _allowlist, _perm, _tokens| {
            Ok(ChildResult {
                summary: format!("did: {task}"),
                tool_call_count: 2,
                child_session_id: "child-abc".to_string(),
            })
        })
    }

    #[test]
    fn test_skill_tool_name_and_description() {
        let registry = Arc::new(SkillRegistry::new());
        let tool = SkillTool { registry };
        assert_eq!(tool.name(), "skill");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_skill_tool_schema_valid_json() {
        let registry = Arc::new(SkillRegistry::new());
        let tool = SkillTool { registry };
        let schema = tool.parameters_schema();
        assert!(schema.is_object());
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("name")));
    }

    #[test]
    fn test_skill_tool_missing_name_errors() {
        let registry = Arc::new(SkillRegistry::new());
        let tool = SkillTool { registry };
        let result = tool
            .execute(serde_json::json!({}), &ctx_no_callback())
            .unwrap();
        assert!(result.content.contains("missing required field 'name'"));
    }

    #[test]
    fn test_skill_tool_unknown_skill_errors_with_available_list() {
        let registry = make_registry_with_skill("review", "Review code", vec![]);
        let tool = SkillTool { registry };
        let result = tool
            .execute(
                serde_json::json!({"name": "nonexistent"}),
                &ctx_no_callback(),
            )
            .unwrap();
        assert!(result.content.contains("skill not found: 'nonexistent'"));
        assert!(result.content.contains("review"));
    }

    #[test]
    fn test_skill_tool_renders_and_calls_spawn_child() {
        let registry = make_registry_with_skill("greet", "Hello {{ who }}!", vec![]);
        let tool = SkillTool { registry };

        let received_task: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let received_clone = Arc::clone(&received_task);

        let cb: Arc<SpawnChildFn> = Arc::new(move |task, _al, _perm, _tok| {
            *received_clone.lock().unwrap() = task.to_string();
            Ok(ChildResult {
                summary: "done".to_string(),
                tool_call_count: 1,
                child_session_id: "s".to_string(),
            })
        });

        let result = tool
            .execute(
                serde_json::json!({"name": "greet", "params": {"who": "world"}}),
                &ctx_with_callback(cb),
            )
            .unwrap();

        assert!(result.content.contains("Skill 'greet' complete."));
        assert_eq!(received_task.lock().unwrap().trim_end(), "Hello world!");
    }

    #[test]
    fn test_skill_tool_passes_tool_allowlist_to_spawn_child() {
        let registry = make_registry_with_skill(
            "limited",
            "Do something",
            vec!["read_file".to_string(), "grep_search".to_string()],
        );
        let tool = SkillTool { registry };

        let received_allowlist: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
        let received_clone = Arc::clone(&received_allowlist);

        let cb: Arc<SpawnChildFn> = Arc::new(move |_task, allowlist, _perm, _tok| {
            *received_clone.lock().unwrap() = allowlist.map(<[std::string::String]>::to_vec);
            Ok(ChildResult {
                summary: "ok".to_string(),
                tool_call_count: 0,
                child_session_id: "s".to_string(),
            })
        });

        tool.execute(
            serde_json::json!({"name": "limited"}),
            &ctx_with_callback(cb),
        )
        .unwrap();

        let list = received_allowlist.lock().unwrap();
        let list = list.as_ref().unwrap();
        assert!(list.contains(&"read_file".to_string()));
        assert!(list.contains(&"grep_search".to_string()));
    }

    #[test]
    fn test_skill_tool_rejects_missing_required_param() {
        use std::io::Write;
        use tempfile::TempDir;

        // Skill declares `scope` with no default → required.
        let dir = TempDir::new().unwrap();
        let content = "---\nname: scoped\ndescription: scope-required skill\nparameters:\n  - name: scope\n---\nFocus on {{ scope }}.\n";
        let path = dir.path().join("scoped.md");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        let registry = Arc::new(SkillRegistry::load_from(dir.path()).unwrap());
        std::mem::forget(dir);

        let tool = SkillTool { registry };

        // Call without providing `scope` → must error, must NOT invoke callback.
        let invoked: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
        let invoked_clone = Arc::clone(&invoked);
        let cb: Arc<SpawnChildFn> = Arc::new(move |_task, _al, _perm, _tok| {
            *invoked_clone.lock().unwrap() = true;
            Ok(ChildResult {
                summary: String::new(),
                tool_call_count: 0,
                child_session_id: "s".to_string(),
            })
        });

        let result = tool
            .execute(
                serde_json::json!({"name": "scoped"}),
                &ctx_with_callback(cb),
            )
            .unwrap();

        assert!(
            result.content.contains("missing required parameters"),
            "expected missing-param error, got: {}",
            result.content
        );
        assert!(result.content.contains("scope"));
        assert!(
            !*invoked.lock().unwrap(),
            "spawn callback must not be invoked when required params are missing"
        );
    }

    #[test]
    fn test_skill_tool_without_spawn_callback_errors() {
        let registry = make_registry_with_skill("review", "Review code", vec![]);
        let tool = SkillTool { registry };
        let result = tool
            .execute(serde_json::json!({"name": "review"}), &ctx_no_callback())
            .unwrap();
        assert!(result.content.contains("spawn callback unavailable"));
    }

    #[test]
    fn test_skill_tool_empty_allowlist_passes_none_to_spawn_child() {
        let registry = make_registry_with_skill("open", "Open task", vec![]);
        let tool = SkillTool { registry };

        let received_allowlist: Arc<Mutex<Option<Vec<String>>>> =
            Arc::new(Mutex::new(Some(vec!["sentinel".to_string()])));
        let received_clone = Arc::clone(&received_allowlist);

        let cb: Arc<SpawnChildFn> = Arc::new(move |_task, allowlist, _perm, _tok| {
            *received_clone.lock().unwrap() = allowlist.map(<[std::string::String]>::to_vec);
            Ok(ChildResult {
                summary: "ok".to_string(),
                tool_call_count: 0,
                child_session_id: "s".to_string(),
            })
        });

        tool.execute(serde_json::json!({"name": "open"}), &ctx_with_callback(cb))
            .unwrap();

        assert!(received_allowlist.lock().unwrap().is_none());
    }

    // ── Edge-case tests ──────────────────────────────────────────────

    /// Empty string name should not match any skill → "skill not found"
    #[test]
    fn test_skill_tool_empty_name_rejected() {
        let registry = make_registry_with_skill("review", "Review code", vec![]);
        let tool = SkillTool { registry };
        let result = tool
            .execute(serde_json::json!({"name": ""}), &ctx_no_callback())
            .unwrap();
        assert!(
            result.content.contains("skill not found"),
            "empty name should produce skill-not-found error, got: {}",
            result.content
        );
    }

    /// Non-string name (integer) → `as_str()` returns None → "missing required field 'name'"
    #[test]
    fn test_skill_tool_name_non_string_rejected() {
        let registry = Arc::new(SkillRegistry::new());
        let tool = SkillTool { registry };
        let result = tool
            .execute(serde_json::json!({"name": 42}), &ctx_no_callback())
            .unwrap();
        assert!(
            result.content.contains("missing required field 'name'"),
            "non-string name should error, got: {}",
            result.content
        );
    }

    /// params as a string (not object) → `as_object()` returns None → empty `HashMap` (no crash)
    #[test]
    fn test_skill_tool_params_non_object_no_crash() {
        let registry = make_registry_with_skill("greet", "Hello {{ who }}!", vec![]);
        let tool = SkillTool { registry };

        let received_task: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let received_clone = Arc::clone(&received_task);

        let cb: Arc<SpawnChildFn> = Arc::new(move |task, _al, _perm, _tok| {
            *received_clone.lock().unwrap() = task.to_string();
            Ok(ChildResult {
                summary: "ok".to_string(),
                tool_call_count: 0,
                child_session_id: "s".to_string(),
            })
        });

        let result = tool
            .execute(
                serde_json::json!({"name": "greet", "params": "not_an_object"}),
                &ctx_with_callback(cb),
            )
            .unwrap();
        // Should succeed — empty params means {{ who }} substituted as empty string
        assert!(result.content.contains("Skill 'greet' complete."));
        assert_eq!(received_task.lock().unwrap().trim_end(), "Hello !");
    }

    /// params with integer values → `filter_map` skips non-string values
    #[test]
    fn test_skill_tool_params_values_not_strings() {
        let registry =
            make_registry_with_skill("greet", "Hello {{ who }}! Count: {{ count }}.", vec![]);
        let tool = SkillTool { registry };

        let received_task: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let received_clone = Arc::clone(&received_task);

        let cb: Arc<SpawnChildFn> = Arc::new(move |task, _al, _perm, _tok| {
            *received_clone.lock().unwrap() = task.to_string();
            Ok(ChildResult {
                summary: "ok".to_string(),
                tool_call_count: 0,
                child_session_id: "s".to_string(),
            })
        });

        let result = tool
            .execute(
                serde_json::json!({"name": "greet", "params": {"who": "world", "count": 42}}),
                &ctx_with_callback(cb),
            )
            .unwrap();
        assert!(result.content.contains("Skill 'greet' complete."));
        // "who" substituted, "count" skipped (integer value)
        assert_eq!(
            received_task.lock().unwrap().trim_end(),
            "Hello world! Count: ."
        );
    }

    /// Multiple placeholders in skill template — all substituted correctly
    #[test]
    fn test_skill_tool_multiple_placeholders() {
        let registry =
            make_registry_with_skill("multi", "Hello {{ a }} and {{ b }} from {{ c }}!", vec![]);
        let tool = SkillTool { registry };

        let received_task: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let received_clone = Arc::clone(&received_task);

        let cb: Arc<SpawnChildFn> = Arc::new(move |task, _al, _perm, _tok| {
            *received_clone.lock().unwrap() = task.to_string();
            Ok(ChildResult {
                summary: "ok".to_string(),
                tool_call_count: 0,
                child_session_id: "s".to_string(),
            })
        });

        tool.execute(
            serde_json::json!({"name": "multi", "params": {"a": "Alice", "b": "Bob", "c": "Charlie"}}),
            &ctx_with_callback(cb),
        )
        .unwrap();

        assert_eq!(
            received_task.lock().unwrap().trim_end(),
            "Hello Alice and Bob from Charlie!"
        );
    }

    /// `spawn_child` callback returns Err — should propagate as `anyhow::Error`
    #[test]
    fn test_skill_tool_spawn_child_error_propagated() {
        let registry = make_registry_with_skill("fail", "Fail task", vec![]);
        let tool = SkillTool { registry };

        let cb: Arc<SpawnChildFn> =
            Arc::new(|_task, _al, _perm, _tok| Err(anyhow::anyhow!("child agent exploded")));

        let result = tool.execute(serde_json::json!({"name": "fail"}), &ctx_with_callback(cb));
        assert!(result.is_err(), "spawn_child error should propagate as Err");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("child agent exploded"),
            "expected spawn error message, got: {err}"
        );
    }

    /// Result message includes `tool_call_count` from `ChildResult`
    #[test]
    fn test_skill_tool_result_includes_tool_call_count() {
        let registry = make_registry_with_skill("work", "Do work", vec![]);
        let tool = SkillTool { registry };

        let cb: Arc<SpawnChildFn> = Arc::new(|_task, _al, _perm, _tok| {
            Ok(ChildResult {
                summary: "completed all tasks".to_string(),
                tool_call_count: 7,
                child_session_id: "child-xyz".to_string(),
            })
        });

        let result = tool
            .execute(serde_json::json!({"name": "work"}), &ctx_with_callback(cb))
            .unwrap();

        assert!(result.content.contains("Skill 'work' complete."));
        assert!(result.content.contains("Summary: completed all tasks"));
        assert!(
            result.content.contains("Tool calls: 7"),
            "result should include tool_call_count, got: {}",
            result.content
        );
    }

    /// Unknown skill with multiple skills registered — error lists all sorted names
    #[test]
    fn test_skill_tool_multiple_skills_available_list() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        for name in &["zebra", "alpha", "mango"] {
            let content = format!("---\nname: {name}\ndescription: Skill {name}\n---\nBody\n");
            let path = dir.path().join(format!("{name}.md"));
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(content.as_bytes()).unwrap();
        }
        let registry = Arc::new(SkillRegistry::load_from(dir.path()).unwrap());
        let tool = SkillTool { registry };

        let result = tool
            .execute(
                serde_json::json!({"name": "nonexistent"}),
                &ctx_no_callback(),
            )
            .unwrap();

        assert!(result.content.contains("skill not found: 'nonexistent'"));
        // The available list should be sorted alphabetically
        let list_part = result.content.split("available: [").nth(1).unwrap();
        let list_part = list_part.trim_end_matches(']');
        let names: Vec<&str> = list_part.split(", ").collect();
        assert_eq!(
            names,
            vec!["alpha", "mango", "zebra"],
            "available skills should be sorted, got: {names:?}"
        );
    }

    /// No params key at all — should work fine with empty `HashMap`
    #[test]
    fn test_skill_tool_no_params_key() {
        let registry = make_registry_with_skill("plain", "No placeholders here", vec![]);
        let tool = SkillTool { registry };

        let received_task: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let received_clone = Arc::clone(&received_task);

        let cb: Arc<SpawnChildFn> = Arc::new(move |task, _al, _perm, _tok| {
            *received_clone.lock().unwrap() = task.to_string();
            Ok(ChildResult {
                summary: "ok".to_string(),
                tool_call_count: 0,
                child_session_id: "s".to_string(),
            })
        });

        let result = tool
            .execute(serde_json::json!({"name": "plain"}), &ctx_with_callback(cb))
            .unwrap();

        assert!(result.content.contains("Skill 'plain' complete."));
        assert_eq!(
            received_task.lock().unwrap().trim_end(),
            "No placeholders here"
        );
    }

    /// params field is JSON null — treated as missing → empty `HashMap`
    #[test]
    fn test_skill_tool_params_null_value() {
        let registry = make_registry_with_skill("greet", "Hello {{ who }}!", vec![]);
        let tool = SkillTool { registry };

        let received_task: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let received_clone = Arc::clone(&received_task);

        let cb: Arc<SpawnChildFn> = Arc::new(move |task, _al, _perm, _tok| {
            *received_clone.lock().unwrap() = task.to_string();
            Ok(ChildResult {
                summary: "ok".to_string(),
                tool_call_count: 0,
                child_session_id: "s".to_string(),
            })
        });

        let result = tool
            .execute(
                serde_json::json!({"name": "greet", "params": null}),
                &ctx_with_callback(cb),
            )
            .unwrap();

        assert!(result.content.contains("Skill 'greet' complete."));
        // null → not an object → empty HashMap → {{ who }} substituted as empty string
        assert_eq!(received_task.lock().unwrap().trim_end(), "Hello !");
    }
}
