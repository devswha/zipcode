use anyhow::{Context, Result};
use regex::Regex;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Seek};
use std::path::Path;

use crate::{validate_glob_pattern, Tool, ToolContext, ToolResult};

pub struct GrepSearchTool;

const MAX_SEARCH_OUTPUT_BYTES: usize = 8_192;
const BINARY_HEADER_SCAN_SIZE: usize = 8192;
const SEARCH_STOP_MESSAGE: &str =
    "[stopped after collecting enough matches; refine the pattern/path for more]";

impl Tool for GrepSearchTool {
    fn name(&self) -> &'static str {
        "grep_search"
    }

    fn description(&self) -> &'static str {
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
        if let Some(glob) = glob_filter {
            validate_glob_pattern(glob)?;
        }

        let mut output_lines: Vec<String> = Vec::new();
        let mut output_bytes = 0usize;

        let stopped_early = if search_path.is_file() {
            search_file(
                &search_path,
                &regex,
                &mut output_lines,
                &mut output_bytes,
                &ctx.cwd,
            )
        } else {
            search_directory(
                &search_path,
                &ctx.cwd,
                glob_filter,
                &regex,
                &mut output_lines,
                &mut output_bytes,
            )?
        };

        if output_lines.is_empty() {
            return Ok(ToolResult::new("No matches found.".to_string()));
        }

        if stopped_early {
            output_lines.push(SEARCH_STOP_MESSAGE.to_string());
        }

        let mut result = ToolResult::new(output_lines.join("\n"));
        result.truncated = stopped_early;
        Ok(result)
    }
}

/// Search a directory lazily, stopping as soon as the output budget is exhausted.
fn search_directory(
    path: &Path,
    workspace_root: &Path,
    glob_filter: Option<&str>,
    regex: &Regex,
    output_lines: &mut Vec<String>,
    output_bytes: &mut usize,
) -> Result<bool> {
    let pattern = glob_filter.map_or_else(
        || format!("{}/**/*", path.to_str().unwrap_or(".")),
        |g| format!("{}/**/{}", path.to_str().unwrap_or("."), g),
    );

    for entry in glob::glob(&pattern).context("Invalid glob pattern")? {
        let Ok(file_path) = entry else {
            continue;
        };
        if !file_path.is_file() {
            continue;
        }
        let display = file_path.to_string_lossy().into_owned();
        if crate::resolve_and_validate_path(&display, workspace_root).is_err() {
            continue;
        }
        if search_file(
            &file_path,
            regex,
            output_lines,
            output_bytes,
            workspace_root,
        ) {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Search a single file for regex matches, appending results to `output_lines`.
/// Returns true when the output budget has been exhausted and the caller should stop.
/// Silently skips binary or unreadable files.
fn search_file(
    path: &Path,
    regex: &Regex,
    output_lines: &mut Vec<String>,
    output_bytes: &mut usize,
    workspace_root: &Path,
) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };

    let mut header = [0_u8; BINARY_HEADER_SCAN_SIZE];
    let Ok(header_len) = file.read(&mut header) else {
        return false;
    };

    if header[..header_len].contains(&0u8) {
        return false;
    }

    if file.rewind().is_err() {
        return false;
    }

    let reader = BufReader::new(file);
    let path_str = crate::make_relative_path(path, workspace_root);

    for (idx, line) in reader.lines().enumerate() {
        let Ok(line) = line else {
            return false;
        };
        if regex.is_match(&line) {
            let rendered = format!("{path_str}:{}: {line}", idx + 1);
            let projected = output_bytes
                .saturating_add(rendered.len())
                .saturating_add(SEARCH_STOP_MESSAGE.len())
                .saturating_add(1);
            if projected > MAX_SEARCH_OUTPUT_BYTES && !output_lines.is_empty() {
                return true;
            }
            *output_bytes = output_bytes.saturating_add(rendered.len() + 1);
            output_lines.push(rendered);
            if *output_bytes >= MAX_SEARCH_OUTPUT_BYTES {
                return true;
            }
        }
    }

    false
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

    #[test]
    fn test_stops_after_output_budget() {
        let dir = TempDir::new().unwrap();
        let repeated = (0..500)
            .map(|idx| format!("match line number {idx:04} with some extra text to grow output"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(dir.path().join("large.txt"), repeated).unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "match line number" });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(result.truncated);
        assert!(result.content.contains(SEARCH_STOP_MESSAGE));
        assert!(result.content.len() <= MAX_SEARCH_OUTPUT_BYTES + SEARCH_STOP_MESSAGE.len() + 32);
    }

    #[test]
    fn test_stops_before_scanning_later_files_once_budget_is_full() {
        let dir = TempDir::new().unwrap();
        let repeated = (0..500)
            .map(|idx| format!("match line number {idx:04} with some extra text to grow output"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(dir.path().join("a-large.txt"), repeated).unwrap();
        fs::write(dir.path().join("z-late.txt"), "match line number late\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "match line number" });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(result.truncated);
        assert!(!result.content.contains("z-late.txt"));
    }

    #[test]
    fn test_absolute_glob_cannot_escape_workspace() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        fs::write(outside.path().join("escape.txt"), "SECRET_MATCH\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": "SECRET_MATCH",
            "glob": format!("{}/*.txt", outside.path().display())
        });

        let error = tool.execute(args, &ctx).unwrap_err().to_string();
        assert!(error.contains("workspace") || error.contains("escape"));
    }

    #[test]
    fn test_parent_dir_glob_cannot_escape_workspace() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("nested");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("inside.txt"), "inside\n").unwrap();

        let outside_dir = dir
            .path()
            .parent()
            .unwrap()
            .join(format!("grep-search-escape-parent-{}", std::process::id()));
        fs::create_dir_all(&outside_dir).unwrap();
        let outside = outside_dir.join("escape.txt");
        fs::write(&outside, "SECRET_MATCH\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": "SECRET_MATCH",
            "glob": "nested/../../*.txt"
        });

        let error = tool.execute(args, &ctx).unwrap_err().to_string();
        assert!(error.contains("workspace") || error.contains("escape"));

        fs::remove_dir_all(&outside_dir).unwrap();
    }

    #[test]
    fn test_invalid_regex_pattern_rejected() {
        let dir = TempDir::new().unwrap();
        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "[unclosed bracket" });
        let result = tool.execute(args, &ctx);
        assert!(result.is_err(), "invalid regex should be rejected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Invalid regex"),
            "error should mention invalid regex, got: {err}"
        );
    }

    #[test]
    fn test_missing_pattern_parameter_rejected() {
        let dir = TempDir::new().unwrap();
        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({});
        let result = tool.execute(args, &ctx);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Missing required parameter"),
            "error should mention missing parameter, got: {err}"
        );
    }

    #[test]
    fn test_empty_directory_returns_no_matches() {
        let dir = TempDir::new().unwrap();
        // Directory exists but has no files at all
        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "anything" });
        let result = tool.execute(args, &ctx).unwrap();
        assert_eq!(result.content, "No matches found.");
    }

    #[test]
    fn test_search_file_with_no_newline_at_end() {
        let dir = TempDir::new().unwrap();
        // Write a file without trailing newline
        fs::write(dir.path().join("noeol.rs"), "fn main() {}").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "fn main" });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(result.content.contains("noeol.rs"));
        assert!(result.content.contains("fn main"));
    }

    #[test]
    fn test_output_paths_are_relative_to_cwd() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("src");
        fs::create_dir(&subdir).unwrap();
        fs::write(subdir.join("app.rs"), "fn greet() {}\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "fn greet" });
        let result = tool.execute(args, &ctx).unwrap();

        // Output format should be relative_path:line_num: content
        // NOT /tmp/.../src/app.rs:1: content
        for line in result.content.lines() {
            assert!(
                !line.starts_with('/'),
                "grep result should be relative, got absolute: {line}"
            );
        }
        assert!(
            result.content.contains("src/app.rs:1: fn greet()"),
            "expected relative path in output, got: {}",
            result.content
        );
    }

    // --- New edge-case tests (10 added) ---

    #[test]
    fn test_unicode_content_in_matches() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("unicode.txt"),
            "안녕하세요 hello 🦀\n普通文本\nmore text\n",
        )
        .unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "안녕" });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(
            result.content.contains("안녕하세요"),
            "should find Korean text in output, got: {}",
            result.content
        );
        assert!(
            result.content.contains("🦀"),
            "should preserve emoji in output, got: {}",
            result.content
        );
    }

    #[test]
    fn test_path_traversal_via_path_param() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        fs::write(outside.path().join("secret.txt"), "SECRET_DATA\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": "SECRET_DATA",
            "path": "../../*.txt"
        });

        let result = tool.execute(args, &ctx);
        assert!(
            result.is_err(),
            "path traversal should be rejected, got: {:?}",
            result
        );
    }

    #[test]
    fn test_nonexistent_path_param() {
        let dir = TempDir::new().unwrap();
        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": "anything",
            "path": "/tmp/absolutely_nonexistent_dir_for_zipcode_test_12345"
        });
        let result = tool.execute(args, &ctx);
        assert!(result.is_err(), "nonexistent path should return error");
    }

    #[test]
    fn test_case_insensitive_regex() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("mixed.rs"),
            "fn Hello() {}\nfn hello() {}\nfn HELLO() {}\n",
        )
        .unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "(?i)hello" });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(
            result.content.contains("Hello"),
            "should match Hello with (?i), got: {}",
            result.content
        );
        assert!(
            result.content.contains("hello"),
            "should match hello with (?i), got: {}",
            result.content
        );
        assert!(
            result.content.contains("HELLO"),
            "should match HELLO with (?i), got: {}",
            result.content
        );
    }

    #[test]
    fn test_multiple_matches_per_line() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("multi.txt"), "aaa bbb aaa\nbbb\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "aaa" });
        let result = tool.execute(args, &ctx).unwrap();

        // grep_search reports the line once when regex matches it
        assert!(
            result.content.contains("multi.txt:1: aaa bbb aaa"),
            "should report the line containing matches, got: {}",
            result.content
        );
        // Line 2 does not contain 'aaa' so should not appear
        assert!(
            !result.content.contains("multi.txt:2:"),
            "line 2 should not match, got: {}",
            result.content
        );
    }

    #[test]
    fn test_nested_directory_search() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src");
        let lib = src.join("lib");
        fs::create_dir_all(&lib).unwrap();
        fs::write(src.join("main.rs"), "fn main() {}\n").unwrap();
        fs::write(lib.join("mod.rs"), "fn helper() {}\n").unwrap();
        fs::write(
            lib.join("utils.rs"),
            "fn util() -> fn helper() { helper() }\n",
        )
        .unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "fn helper" });
        let result = tool.execute(args, &ctx).unwrap();

        // mod.rs defines `fn helper()`, utils.rs mentions `fn helper` in a return type
        assert!(
            result.content.contains("src/lib/mod.rs"),
            "should find match in nested src/lib/mod.rs, got: {}",
            result.content
        );
        assert!(
            result.content.contains("src/lib/utils.rs"),
            "should find match in deeply nested src/lib/utils.rs, got: {}",
            result.content
        );
        // main.rs only has `fn main`, not `fn helper`
        assert!(
            !result.content.contains("src/main.rs"),
            "src/main.rs should not match, got: {}",
            result.content
        );
    }

    #[test]
    fn test_glob_filter_with_nested_dirs() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("subdir");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("code.rs"), "let x = 42;\n").unwrap();
        fs::write(sub.join("notes.txt"), "let y = 42;\n").unwrap();
        fs::write(dir.path().join("top.rs"), "let z = 42;\n").unwrap();
        fs::write(dir.path().join("top.txt"), "let w = 42;\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({
            "pattern": "let .* = 42",
            "glob": "*.rs"
        });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(
            result.content.contains("top.rs"),
            "should find .rs at top level, got: {}",
            result.content
        );
        assert!(
            result.content.contains("subdir/code.rs"),
            "should find .rs in nested dir, got: {}",
            result.content
        );
        assert!(
            !result.content.contains(".txt"),
            "should not include .txt files, got: {}",
            result.content
        );
    }

    #[test]
    fn test_special_chars_in_filename() {
        let dir = TempDir::new().unwrap();
        let filename = "my [test] file.rs";
        fs::write(dir.path().join(filename), "TODO: fix this later\n").unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "TODO" });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(
            result.content.contains("TODO"),
            "should find match in file with special chars name, got: {}",
            result.content
        );
    }

    #[test]
    fn test_empty_pattern_matches_everything() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("data.txt"),
            "line one\nline two\nline three\n",
        )
        .unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "" });
        let result = tool.execute(args, &ctx).unwrap();

        // Empty regex matches every line
        assert!(
            result.content.contains("line one"),
            "empty pattern should match all lines, got: {}",
            result.content
        );
        assert!(
            result.content.contains("line two"),
            "empty pattern should match line two, got: {}",
            result.content
        );
        assert!(
            result.content.contains("line three"),
            "empty pattern should match line three, got: {}",
            result.content
        );
    }

    #[test]
    fn test_single_match_near_budget_boundary() {
        let dir = TempDir::new().unwrap();
        // Create a single line that is just under the budget, then a few more lines
        // to verify truncation kicks in when the budget is actually exceeded
        let long_line: String = "x".repeat(MAX_SEARCH_OUTPUT_BYTES - 100);
        let content =
            format!("match_line: {long_line}\nmatch_line: short one\nmatch_line: another\n");
        fs::write(dir.path().join("budget.txt"), content).unwrap();

        let tool = GrepSearchTool;
        let ctx = make_ctx(&dir);
        let args = serde_json::json!({ "pattern": "match_line" });
        let result = tool.execute(args, &ctx).unwrap();

        // The first line alone is near the budget, so subsequent lines may or may not
        // be truncated. The key assertion is that we don't panic and the output is bounded.
        assert!(
            result.content.contains("match_line"),
            "should find at least the first match, got: {}",
            result.content
        );
        assert!(
            result.content.len() <= MAX_SEARCH_OUTPUT_BYTES + SEARCH_STOP_MESSAGE.len() + 200,
            "output should not wildly exceed budget, got {} bytes",
            result.content.len()
        );
    }
}
