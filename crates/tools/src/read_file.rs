use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;

use crate::{Tool, ToolContext, ToolResult};

const MAX_READ_SIZE: u64 = 10 * 1024 * 1024; // 10MB

pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a file with line numbers. Supports offset and limit for partial reads."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative paths resolved against cwd)"
                },
                "offset": {
                    "type": "integer",
                    "description": "0-based line number to start reading from (default: 0)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: all)"
                }
            },
            "required": ["path"]
        })
    }

    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: path"))?;

        let requested_path = crate::resolve_and_validate_path(path_str, &ctx.cwd)
            .with_context(|| format!("failed to resolve path: {path_str}"))?;
        let mut corrected_from = None;
        let (path, metadata) = match fs::metadata(&requested_path) {
            Ok(metadata) => (requested_path, metadata),
            Err(original_error) => {
                if let Some((repaired_relative, repaired_absolute)) =
                    crate::recover_duplicated_workspace_prefix(path_str, &ctx.cwd)
                {
                    let metadata = fs::metadata(&repaired_absolute).with_context(|| {
                        format!(
                            "failed to stat file after path recovery: {}",
                            repaired_absolute.display()
                        )
                    })?;
                    corrected_from = Some((path_str.to_string(), repaired_relative));
                    (repaired_absolute, metadata)
                } else {
                    return Err(original_error).with_context(|| {
                        format!("failed to stat file: {}", requested_path.display())
                    });
                }
            }
        };

        let file_size = metadata.len();
        if file_size > MAX_READ_SIZE {
            anyhow::bail!(
                "file too large to read: {} ({} bytes, limit is {} bytes)",
                path.display(),
                file_size,
                MAX_READ_SIZE
            );
        }

        let bytes =
            fs::read(&path).with_context(|| format!("failed to read file: {}", path.display()))?;

        if bytes[..bytes.len().min(8192)].contains(&0u8) {
            anyhow::bail!("file appears to be binary: {}", path.display());
        }

        let content = String::from_utf8(bytes)
            .with_context(|| format!("file is not valid UTF-8: {}", path.display()))?;

        let offset = args["offset"].as_u64().unwrap_or(0) as usize;
        let limit = args["limit"].as_u64().map(|v| v as usize);

        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();

        let start = offset.min(total);
        let end = match limit {
            Some(n) => (start + n).min(total),
            None => total,
        };

        let auto_correct_note = corrected_from
            .as_ref()
            .map(|(from, to)| format!("[auto-corrected path: {from} -> {to}]"));

        let header = if total == 0 {
            format!("[file: {}, empty file]", path.display())
        } else if start >= total {
            format!(
                "[file: {}, offset {} exceeds {} total lines]",
                path.display(),
                offset,
                total
            )
        } else {
            format!(
                "[file: {}, lines: {}-{} of {} total]",
                path.display(),
                start + 1,
                end,
                total
            )
        };
        let header = match auto_correct_note {
            Some(note) => format!("{note}\n{header}"),
            None => header,
        };

        let numbered: String = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{}\t{}", start + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

        let output = if numbered.is_empty() {
            header
        } else {
            format!("{}\n{}", header, numbered)
        };

        Ok(ToolResult::new(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn ctx() -> ToolContext {
        ToolContext {
            cwd: std::path::PathBuf::from("/tmp"),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        }
    }

    #[test]
    fn test_read_existing_file() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "line one").unwrap();
        writeln!(f, "line two").unwrap();
        writeln!(f, "line three").unwrap();

        let tool = ReadFileTool;
        let args = serde_json::json!({ "path": f.path().to_str().unwrap() });
        let result = tool.execute(args, &ctx()).unwrap();

        assert!(result.content.contains("1\tline one"));
        assert!(result.content.contains("2\tline two"));
        assert!(result.content.contains("3\tline three"));
    }

    #[test]
    fn test_read_with_offset_and_limit() {
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=5 {
            writeln!(f, "line {i}").unwrap();
        }

        let tool = ReadFileTool;
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "offset": 1,
            "limit": 2
        });
        let result = tool.execute(args, &ctx()).unwrap();

        // offset=1 means skip first line; lines 2 and 3 returned
        assert!(!result.content.contains("1\tline 1"));
        assert!(result.content.contains("2\tline 2"));
        assert!(result.content.contains("3\tline 3"));
        assert!(!result.content.contains("4\tline 4"));
    }

    #[test]
    fn test_read_nonexistent_file() {
        let tool = ReadFileTool;
        let args = serde_json::json!({ "path": "/tmp/does_not_exist_zipcode_test.txt" });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err());
    }

    #[test]
    fn test_read_binary_file_rejected() {
        let mut f = NamedTempFile::new().unwrap();
        // Write some null bytes to make the file appear binary
        f.write_all(b"some text\x00more text").unwrap();

        let tool = ReadFileTool;
        let args = serde_json::json!({ "path": f.path().to_str().unwrap() });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("binary"), "error should mention binary: {err}");
    }

    #[test]
    fn test_read_large_file_rejected() {
        let f = NamedTempFile::new().unwrap();
        f.as_file().set_len(MAX_READ_SIZE + 1).unwrap();

        let tool = ReadFileTool;
        let args = serde_json::json!({ "path": f.path().to_str().unwrap() });
        let result = tool.execute(args, &ctx());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("too large"),
            "error should mention size: {err}"
        );
    }

    #[test]
    fn test_read_empty_file() {
        let f = NamedTempFile::new().unwrap();
        let tool = ReadFileTool;
        let args = serde_json::json!({ "path": f.path().to_str().unwrap() });
        let result = tool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("empty file"));
    }

    #[test]
    fn test_read_offset_beyond_eof() {
        let mut f = NamedTempFile::new().unwrap();
        use std::io::Write;
        writeln!(f, "line one").unwrap();
        writeln!(f, "line two").unwrap();

        let tool = ReadFileTool;
        let args = serde_json::json!({ "path": f.path().to_str().unwrap(), "offset": 500 });
        let result = tool.execute(args, &ctx()).unwrap();
        assert!(result.content.contains("exceeds"));
    }

    #[test]
    fn test_read_auto_corrects_duplicated_workspace_prefix() {
        let dir = tempfile::TempDir::new().unwrap();
        let workspace = dir.path().join("zipcode");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::write(workspace.join("README.md"), "hello from readme").unwrap();

        let ctx = ToolContext {
            cwd: workspace.clone(),
            permission: PermissionMode::FullAccess,
            session_id: "test".to_string(),
        };

        let tool = ReadFileTool;
        let args = serde_json::json!({ "path": "zipcode/README.md" });
        let result = tool.execute(args, &ctx).unwrap();

        assert!(result.content.contains("auto-corrected path"));
        assert!(result.content.contains("hello from readme"));
    }

    #[test]
    fn test_read_output_includes_metadata() {
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(f, "line {i}").unwrap();
        }

        let tool = ReadFileTool;
        // Read lines 3-5 (offset=2, limit=3)
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "offset": 2,
            "limit": 3
        });
        let result = tool.execute(args, &ctx()).unwrap();

        assert!(
            result.content.starts_with("[file:"),
            "output should start with metadata header: {}",
            result.content
        );
        assert!(
            result.content.contains("lines: 3-5 of 10 total"),
            "header should include line range and total: {}",
            result.content
        );
    }
}
