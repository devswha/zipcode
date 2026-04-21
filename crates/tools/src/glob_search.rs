use anyhow::{Context, Result};
use serde_json::Value;

use crate::{validate_glob_pattern, Tool, ToolContext, ToolResult};

pub struct GlobSearchTool;

impl Tool for GlobSearchTool {
    fn name(&self) -> &'static str {
        "glob_search"
    }

    fn description(&self) -> &'static str {
        "Find files matching a glob pattern. Returns a sorted list of matching file paths."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g. '*.rs', '**/*.ts')"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in. Defaults to the current working directory."
                }
            },
            "required": ["pattern"]
        })
    }

    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: pattern"))?;
        validate_glob_pattern(pattern)?;

        let base = if let Some(p) = args["path"].as_str() {
            crate::resolve_and_validate_path(p, &ctx.cwd)?
        } else {
            ctx.cwd.clone()
        };

        // Build full glob pattern: base + pattern
        let full_pattern = base.join(pattern);
        let full_pattern_str = full_pattern.to_str().context("Invalid path encoding")?;

        let mut matches: Vec<String> = glob::glob(full_pattern_str)
            .context("Invalid glob pattern")?
            .filter_map(std::result::Result::ok)
            .filter(|p| p.is_file())
            .filter_map(|p| {
                let display = p.to_string_lossy().into_owned();
                crate::resolve_and_validate_path(&display, &ctx.cwd)
                    .ok()
                    .map(|_| crate::make_relative_path(&p, &ctx.cwd))
            })
            .collect();

        if matches.is_empty() {
            return Ok(ToolResult::new("No files matched the pattern.".to_string()));
        }

        matches.sort();
        Ok(ToolResult::new(matches.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use std::fs;
    use tempfile::TempDir;

    fn make_ctx(dir: &TempDir) -> ToolContext {
        ToolContext {
            cwd: dir.path().to_path_buf(),
            permission: PermissionMode::ReadOnly,
            session_id: "test".to_string(),
            parent_session_id: None,
            depth: 0,
            budget_tokens: None,
        }
    }

    #[test]
    fn test_find_rs_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("lib.rs"), "pub fn foo() {}").unwrap();
        fs::write(dir.path().join("notes.txt"), "hello").unwrap();

        let tool = GlobSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "*.rs" });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(!result.truncated);
        assert!(result.content.contains("main.rs"));
        assert!(result.content.contains("lib.rs"));
        assert!(!result.content.contains("notes.txt"));
    }

    #[test]
    fn test_no_matches() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();

        let tool = GlobSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "*.ts" });
        let result = tool.execute(args, &ctx).unwrap();

        assert_eq!(result.content, "No files matched the pattern.");
    }

    #[test]
    fn test_recursive_pattern() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("src");
        fs::create_dir(&subdir).unwrap();
        fs::write(subdir.join("app.ts"), "export {}").unwrap();
        fs::write(dir.path().join("index.ts"), "export {}").unwrap();
        fs::write(dir.path().join("readme.md"), "# hello").unwrap();

        let tool = GlobSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "**/*.ts" });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(result.content.contains("app.ts"));
        assert!(result.content.contains("index.ts"));
        assert!(!result.content.contains("readme.md"));
    }

    #[test]
    fn test_explicit_path_param() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("sub");
        fs::create_dir(&subdir).unwrap();
        fs::write(subdir.join("foo.rs"), "").unwrap();

        let tool = GlobSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": "*.rs",
            "path": subdir.to_str().unwrap()
        });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(result.content.contains("foo.rs"));
    }

    #[test]
    fn test_results_are_sorted() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("z.rs"), "").unwrap();
        fs::write(dir.path().join("a.rs"), "").unwrap();
        fs::write(dir.path().join("m.rs"), "").unwrap();

        let tool = GlobSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "*.rs" });
        let result = tool.execute(args, &ctx).unwrap();

        let lines: Vec<&str> = result.content.lines().collect();
        let mut sorted = lines.clone();
        sorted.sort();
        assert_eq!(lines, sorted);
    }

    #[test]
    fn test_absolute_pattern_cannot_escape_workspace() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        fs::write(outside.path().join("escape.rs"), "").unwrap();

        let tool = GlobSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": format!("{}/*.rs", outside.path().display())
        });

        let error = tool.execute(args, &ctx).unwrap_err().to_string();
        assert!(error.contains("workspace") || error.contains("escape"));
    }

    #[test]
    fn test_parent_dir_pattern_cannot_escape_workspace() {
        let dir = TempDir::new().unwrap();
        let outside_parent = dir
            .path()
            .parent()
            .unwrap()
            .join("glob-search-escape-parent");
        fs::create_dir_all(&outside_parent).unwrap();
        fs::write(outside_parent.join("escape.rs"), "").unwrap();

        let tool = GlobSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": "../glob-search-escape-parent/*.rs"
        });

        let error = tool.execute(args, &ctx).unwrap_err().to_string();
        assert!(error.contains("workspace") || error.contains("escape"));

        fs::remove_dir_all(&outside_parent).unwrap();
    }

    #[test]
    fn test_results_are_relative_to_cwd() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("alpha.rs"), "").unwrap();
        let subdir = dir.path().join("src");
        fs::create_dir(&subdir).unwrap();
        fs::write(subdir.join("beta.rs"), "").unwrap();

        let tool = GlobSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "**/*.rs" });
        let result = tool.execute(args, &ctx).unwrap();

        // Every result line should be a relative path — no leading '/'
        for line in result.content.lines() {
            assert!(
                !line.starts_with('/'),
                "glob result should be relative, got absolute: {line}"
            );
        }
        assert!(result.content.contains("alpha.rs"));
        // Recursive match inside src/
        assert!(
            result.content.contains("src/beta.rs"),
            "expected 'src/beta.rs' in output, got: {}",
            result.content
        );
    }
}
