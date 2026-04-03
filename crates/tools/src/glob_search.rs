use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::{Tool, ToolContext, ToolResult};

fn resolve_path(file_path: &str, cwd: &Path) -> std::path::PathBuf {
    let p = Path::new(file_path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

pub struct GlobSearchTool;

impl Tool for GlobSearchTool {
    fn name(&self) -> &str {
        "glob_search"
    }

    fn description(&self) -> &str {
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

        let base = if let Some(p) = args["path"].as_str() {
            resolve_path(p, &ctx.cwd)
        } else {
            ctx.cwd.clone()
        };

        // Build full glob pattern: base + pattern
        let full_pattern = base.join(pattern);
        let full_pattern_str = full_pattern.to_str().context("Invalid path encoding")?;

        let mut matches: Vec<String> = glob::glob(full_pattern_str)
            .context("Invalid glob pattern")?
            .filter_map(|entry| entry.ok())
            .filter(|p| p.is_file())
            .map(|p| p.to_string_lossy().into_owned())
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
}
