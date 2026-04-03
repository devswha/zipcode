use anyhow::{Context, Result};
use regex::Regex;
use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::{Tool, ToolContext, ToolResult};

pub struct GrepSearchTool;

impl Tool for GrepSearchTool {
    fn name(&self) -> &str {
        "grep_search"
    }

    fn description(&self) -> &str {
        "Search file contents using a regex pattern. Returns matches in 'filepath:line_num: line_content' format."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to the current working directory."
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files when path is a directory (e.g. '*.rs')"
                }
            },
            "required": ["pattern"]
        })
    }

    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: pattern"))?;

        let regex = Regex::new(pattern).context("Invalid regex pattern")?;

        let search_path = if let Some(p) = args["path"].as_str() {
            crate::resolve_and_validate_path(p, &ctx.cwd)?
        } else {
            ctx.cwd.clone()
        };

        let glob_filter = args["glob"].as_str();

        let files = collect_files(&search_path, glob_filter)?;

        let mut output_lines: Vec<String> = Vec::new();

        for file_path in &files {
            search_file(file_path, &regex, &mut output_lines);
        }

        if output_lines.is_empty() {
            return Ok(ToolResult::new("No matches found.".to_string()));
        }

        Ok(ToolResult::new(output_lines.join("\n")))
    }
}

/// Collect files to search. If path is a file, return just that file.
/// If path is a directory, walk recursively applying optional glob filter.
fn collect_files(path: &Path, glob_filter: Option<&str>) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    let pattern = if let Some(g) = glob_filter {
        format!("{}/**/{}", path.to_str().unwrap_or("."), g)
    } else {
        format!("{}/**/*", path.to_str().unwrap_or("."))
    };

    let files: Vec<PathBuf> = glob::glob(&pattern)
        .context("Invalid glob pattern")?
        .filter_map(|e| e.ok())
        .filter(|p| p.is_file())
        .collect();

    Ok(files)
}

/// Search a single file for regex matches, appending results to output_lines.
/// Silently skips binary or unreadable files.
fn search_file(path: &Path, regex: &Regex, output_lines: &mut Vec<String>) {
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return,
    };

    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return;
    }

    // Skip binary files: check for null bytes in the first 8KB
    let check_len = bytes.len().min(8192);
    if bytes[..check_len].contains(&0u8) {
        return;
    }

    let content = match std::str::from_utf8(&bytes) {
        Ok(s) => s,
        Err(_) => return,
    };

    let path_str = path.to_string_lossy();
    for (idx, line) in content.lines().enumerate() {
        if regex.is_match(line) {
            output_lines.push(format!("{}:{}: {}", path_str, idx + 1, line));
        }
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
    fn test_find_regex_matches_in_directory() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("foo.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .unwrap();
        fs::write(dir.path().join("bar.rs"), "fn other() {}\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "fn main" });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(result.content.contains("foo.rs"));
        assert!(result.content.contains("fn main"));
        assert!(!result.content.contains("bar.rs"));
    }

    #[test]
    fn test_no_matches() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("hello.rs"), "fn greet() {}\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "nonexistent_xyz_pattern" });
        let result = tool.execute(args, &ctx).unwrap();

        assert_eq!(result.content, "No matches found.");
    }

    #[test]
    fn test_search_single_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("target.txt");
        fs::write(&file_path, "line one\nfind me here\nline three\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": "find me",
            "path": file_path.to_str().unwrap()
        });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(result.content.contains("target.txt:2: find me here"));
    }

    #[test]
    fn test_glob_filter() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("code.rs"), "let x = 42;\n").unwrap();
        fs::write(dir.path().join("notes.txt"), "let y = 42;\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": "let .* = 42",
            "glob": "*.rs"
        });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(result.content.contains("code.rs"));
        assert!(!result.content.contains("notes.txt"));
    }

    #[test]
    fn test_output_format() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("sample.rs");
        fs::write(&file_path, "// first line\nTODO: fix this\n// third line\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": "TODO",
            "path": file_path.to_str().unwrap()
        });
        let result = tool.execute(args, &ctx).unwrap();

        // Format: filepath:line_num: line_content
        assert!(result.content.contains(":2: TODO: fix this"));
    }

    #[test]
    fn test_skips_binary_files() {
        let dir = TempDir::new().unwrap();
        // Write a binary file with null bytes
        let binary_data: Vec<u8> = vec![0x7f, 0x45, 0x4c, 0x46, 0x00, 0x00, b'T', b'O', b'D', b'O'];
        fs::write(dir.path().join("binary.bin"), &binary_data).unwrap();
        fs::write(dir.path().join("text.rs"), "TODO: real match\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "TODO" });
        let result = tool.execute(args, &ctx).unwrap();

        // Should find match in text.rs but not crash on binary
        assert!(result.content.contains("text.rs"));
        assert!(!result.content.contains("binary.bin"));
    }
}
