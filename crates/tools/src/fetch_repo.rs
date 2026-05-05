use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::{make_relative_path, wait_with_output_timeout, Tool, ToolContext, ToolResult};

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const SCRATCH_DIR: &str = ".zipcode-remote";

pub struct FetchRepoTool;

impl Tool for FetchRepoTool {
    fn name(&self) -> &'static str {
        "fetch_repo"
    }

    fn description(&self) -> &'static str {
        "Fetch a public GitHub repository URL into a workspace-local .zipcode-remote directory for analysis without executing repository code. Use this before read_file/grep_search/glob_search when the user asks to analyze a GitHub URL."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "GitHub repository URL, e.g. https://github.com/owner/repo"
                },
                "dest": {
                    "type": "string",
                    "description": "Optional destination under .zipcode-remote/. Defaults to .zipcode-remote/<owner>__<repo>."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 120000)"
                }
            },
            "required": ["url"]
        })
    }

    fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let url = args["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: url"))?;
        let repo = parse_github_repo_url(url)?;
        let dest = destination_path(args["dest"].as_str(), &repo, &ctx.cwd)?;

        if dest.exists() {
            if !dest.is_dir() {
                anyhow::bail!(
                    "destination already exists and is not a directory: {}",
                    make_relative_path(&dest, &ctx.cwd)
                );
            }
            return Ok(ToolResult::new(repo_result_message(
                &repo, &dest, &ctx.cwd, true,
            )));
        }

        let scratch = ctx.cwd.join(SCRATCH_DIR);
        std::fs::create_dir_all(&scratch).with_context(|| {
            format!("failed to create scratch directory: {}", scratch.display())
        })?;

        let timeout_ms = args["timeout"].as_u64().unwrap_or(DEFAULT_TIMEOUT_MS);
        let timeout = std::time::Duration::from_millis(timeout_ms.clamp(100, 300_000));
        let child = Command::new("git")
            .args(["clone", "--depth", "1", "--", url])
            .arg(&dest)
            .current_dir(&ctx.cwd)
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("failed to start git clone; ensure git is installed")?;

        let Some(output) = wait_with_output_timeout(child, timeout)? else {
            let _ = std::fs::remove_dir_all(&dest);
            return Ok(ToolResult::error(&format!(
                "git clone timed out after {}ms",
                timeout.as_millis()
            )));
        };

        if !output.status.success() {
            let _ = std::fs::remove_dir_all(&dest);
            let mut message = String::from("git clone failed");
            if let Some(code) = output.status.code() {
                let _ = write!(message, " with exit code {code}");
            }
            append_output(&mut message, &output.stdout, "STDOUT");
            append_output(&mut message, &output.stderr, "STDERR");
            return Ok(ToolResult::error(&message));
        }

        Ok(ToolResult::new(repo_result_message(
            &repo, &dest, &ctx.cwd, false,
        )))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GithubRepo {
    owner: String,
    name: String,
}

fn parse_github_repo_url(url: &str) -> Result<GithubRepo> {
    let rest = url.strip_prefix("https://github.com/").ok_or_else(|| {
        anyhow::anyhow!("fetch_repo only accepts https://github.com/<owner>/<repo> URLs")
    })?;
    if rest.contains('?') || rest.contains('#') {
        anyhow::bail!("GitHub repository URL must not contain query strings or fragments");
    }

    let trimmed = rest.trim_end_matches('/');
    let parts = trimmed.split('/').collect::<Vec<_>>();
    if parts.len() != 2 {
        anyhow::bail!("GitHub repository URL must be exactly https://github.com/<owner>/<repo>");
    }

    let owner = parts[0];
    let repo = parts[1].strip_suffix(".git").unwrap_or(parts[1]);
    validate_github_segment(owner, "owner")?;
    validate_github_segment(repo, "repo")?;

    Ok(GithubRepo {
        owner: owner.to_string(),
        name: repo.to_string(),
    })
}

fn validate_github_segment(value: &str, label: &str) -> Result<()> {
    if value.is_empty() {
        anyhow::bail!("GitHub {label} must not be empty");
    }
    if !value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        anyhow::bail!("GitHub {label} contains unsupported characters");
    }
    Ok(())
}

fn destination_path(dest_arg: Option<&str>, repo: &GithubRepo, cwd: &Path) -> Result<PathBuf> {
    let relative = dest_arg.map_or_else(
        || format!("{SCRATCH_DIR}/{}__{}", repo.owner, repo.name),
        ToString::to_string,
    );
    if !relative.starts_with(&format!("{SCRATCH_DIR}/")) {
        anyhow::bail!("dest must be under {SCRATCH_DIR}/");
    }
    if Path::new(&relative).components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        anyhow::bail!("dest must stay under {SCRATCH_DIR}/");
    }

    let scratch = crate::resolve_and_validate_path(SCRATCH_DIR, cwd)?;
    let dest = crate::resolve_and_validate_path(&relative, cwd)?;
    if dest == scratch || !dest.starts_with(&scratch) {
        anyhow::bail!("dest must stay under {SCRATCH_DIR}/");
    }
    Ok(dest)
}

fn primary_files(dest: &Path, cwd: &Path) -> Vec<String> {
    [
        "README.md",
        "readme.md",
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
    ]
    .into_iter()
    .map(|name| dest.join(name))
    .filter(|path| path.is_file())
    .map(|path| make_relative_path(&path, cwd))
    .collect()
}

fn repo_result_message(repo: &GithubRepo, dest: &Path, cwd: &Path, reused: bool) -> String {
    let relative_dest = make_relative_path(dest, cwd);
    let mut message = if reused {
        format!(
            "GitHub repository {}/{} is already available at {relative_dest}.",
            repo.owner, repo.name
        )
    } else {
        format!(
            "Fetched GitHub repository {}/{} to {relative_dest}.",
            repo.owner, repo.name
        )
    };
    let candidates = primary_files(dest, cwd);
    if !candidates.is_empty() {
        message.push_str("\nPrimary files to inspect next:");
        for file in candidates {
            let _ = write!(message, "\n- {file}");
        }
    }
    message
}

fn append_output(message: &mut String, bytes: &[u8], label: &str) {
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim();
    if !trimmed.is_empty() {
        let _ = write!(message, "\n{label}:\n{trimmed}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PermissionMode;
    use tempfile::TempDir;

    fn ctx(dir: &TempDir) -> ToolContext {
        ToolContext {
            cwd: dir.path().to_path_buf(),
            permission: PermissionMode::WorkspaceWrite,
            session_id: "test".to_string(),
            parent_session_id: None,
            depth: 0,
            budget_tokens: None,
            spawn_child: None,
        }
    }

    #[test]
    fn parses_github_repo_urls() {
        assert_eq!(
            parse_github_repo_url("https://github.com/devswha/patina").unwrap(),
            GithubRepo {
                owner: "devswha".to_string(),
                name: "patina".to_string(),
            }
        );
        assert_eq!(
            parse_github_repo_url("https://github.com/devswha/patina.git")
                .unwrap()
                .name,
            "patina"
        );
    }

    #[test]
    fn rejects_non_github_or_nested_urls() {
        assert!(parse_github_repo_url("https://example.com/devswha/patina").is_err());
        assert!(parse_github_repo_url("https://github.com/devswha/patina/tree/main").is_err());
        assert!(parse_github_repo_url("https://github.com/devswha/patina?tab=readme").is_err());
    }

    #[test]
    fn destination_defaults_under_scratch_dir() {
        let dir = TempDir::new().unwrap();
        let repo = parse_github_repo_url("https://github.com/devswha/patina").unwrap();
        let dest = destination_path(None, &repo, dir.path()).unwrap();
        assert_eq!(
            dest.strip_prefix(dir.path()).unwrap().to_string_lossy(),
            ".zipcode-remote/devswha__patina"
        );
    }

    #[test]
    fn destination_must_stay_under_scratch_dir() {
        let dir = TempDir::new().unwrap();
        let repo = parse_github_repo_url("https://github.com/devswha/patina").unwrap();
        assert!(destination_path(Some("outside"), &repo, dir.path()).is_err());
        assert!(destination_path(Some(".zipcode-remote/../outside"), &repo, dir.path()).is_err());
        assert!(
            destination_path(Some(".zipcode-remote/../../outside"), &repo, dir.path()).is_err()
        );
        assert!(destination_path(Some(".zipcode-remote/"), &repo, dir.path()).is_err());
    }

    #[test]
    fn existing_directory_is_reused_before_clone() {
        let dir = TempDir::new().unwrap();
        let existing = dir.path().join(".zipcode-remote/devswha__patina");
        std::fs::create_dir_all(&existing).unwrap();
        std::fs::write(existing.join("README.md"), "# Patina").unwrap();
        let result = FetchRepoTool
            .execute(
                serde_json::json!({ "url": "https://github.com/devswha/patina" }),
                &ctx(&dir),
            )
            .unwrap();
        assert!(result.content.contains("already available"));
        assert!(result.content.contains(".zipcode-remote/devswha__patina"));
        assert!(result.content.contains("README.md"));
    }

    #[test]
    fn existing_file_destination_is_rejected_before_clone() {
        let dir = TempDir::new().unwrap();
        let existing = dir.path().join(".zipcode-remote/devswha__patina");
        std::fs::create_dir_all(existing.parent().unwrap()).unwrap();
        std::fs::write(&existing, "not a directory").unwrap();
        let result = FetchRepoTool.execute(
            serde_json::json!({ "url": "https://github.com/devswha/patina" }),
            &ctx(&dir),
        );
        let error = result.unwrap_err().to_string();
        assert!(error.contains("not a directory"));
    }

    // --- New tests added for comprehensive coverage ---

    #[test]
    fn validate_github_segment_rejects_empty() {
        assert!(super::validate_github_segment("", "owner").is_err());
        assert!(super::validate_github_segment("", "repo").is_err());
    }

    #[test]
    fn validate_github_segment_rejects_special_chars() {
        // @, !, space, :, / are not allowed
        assert!(super::validate_github_segment("user@name", "owner").is_err());
        assert!(super::validate_github_segment("my repo", "repo").is_err());
        assert!(super::validate_github_segment("owner!", "owner").is_err());
        assert!(super::validate_github_segment("na:me", "owner").is_err());
    }

    #[test]
    fn validate_github_segment_accepts_valid() {
        // alphanumeric, hyphens, underscores, dots
        assert!(super::validate_github_segment("devswha", "owner").is_ok());
        assert!(super::validate_github_segment("my-repo", "repo").is_ok());
        assert!(super::validate_github_segment("repo_name", "repo").is_ok());
        assert!(super::validate_github_segment("repo.name", "repo").is_ok());
        assert!(super::validate_github_segment("A1-b2.c3_d4", "owner").is_ok());
    }

    #[test]
    fn parse_url_trailing_slash() {
        let repo = parse_github_repo_url("https://github.com/devswha/patina/").unwrap();
        assert_eq!(repo.owner, "devswha");
        assert_eq!(repo.name, "patina");
    }

    #[test]
    fn primary_files_finds_existing() {
        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("repo");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("README.md"), "# readme").unwrap();
        std::fs::write(dest.join("Cargo.toml"), "[package]").unwrap();
        // A file that is NOT in the candidates list
        std::fs::write(dest.join("random.txt"), "ignore").unwrap();

        let files = super::primary_files(&dest, dir.path());
        assert_eq!(files.len(), 2);
        assert!(files[0].contains("README.md"));
        assert!(files[1].contains("Cargo.toml"));
    }

    #[test]
    fn primary_files_empty_dir() {
        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("repo");
        std::fs::create_dir_all(&dest).unwrap();
        // No recognized files
        std::fs::write(dest.join("random.txt"), "ignore").unwrap();

        let files = super::primary_files(&dest, dir.path());
        assert!(files.is_empty());
    }

    #[test]
    fn primary_files_case_insensitive_readme() {
        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("repo");
        std::fs::create_dir_all(&dest).unwrap();
        // lowercase readme.md should also be found
        std::fs::write(dest.join("readme.md"), "# readme").unwrap();

        let files = super::primary_files(&dest, dir.path());
        assert_eq!(files.len(), 1);
        assert!(files[0].contains("readme.md"));
    }

    #[test]
    fn repo_result_message_fresh() {
        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("repo");
        std::fs::create_dir_all(&dest).unwrap();
        let repo = GithubRepo {
            owner: "devswha".to_string(),
            name: "patina".to_string(),
        };
        let msg = super::repo_result_message(&repo, &dest, dir.path(), false);
        assert!(
            msg.contains("Fetched"),
            "fresh message should say 'Fetched', got: {msg}"
        );
        assert!(
            msg.contains("devswha/patina"),
            "message should contain repo slug, got: {msg}"
        );
    }

    #[test]
    fn repo_result_message_reused() {
        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("repo");
        std::fs::create_dir_all(&dest).unwrap();
        let repo = GithubRepo {
            owner: "devswha".to_string(),
            name: "patina".to_string(),
        };
        let msg = super::repo_result_message(&repo, &dest, dir.path(), true);
        assert!(
            msg.contains("already available"),
            "reused message should say 'already available', got: {msg}"
        );
        assert!(
            msg.contains("devswha/patina"),
            "message should contain repo slug, got: {msg}"
        );
    }

    #[test]
    fn append_output_appends_nonempty() {
        let mut message = String::from("prefix");
        super::append_output(&mut message, b"hello world", "STDOUT");
        assert!(
            message.contains("STDOUT"),
            "should contain label STDOUT, got: {message}"
        );
        assert!(
            message.contains("hello world"),
            "should contain content, got: {message}"
        );
    }

    #[test]
    fn append_output_skips_empty() {
        let mut message = String::from("prefix");
        super::append_output(&mut message, b"", "STDOUT");
        assert_eq!(message, "prefix", "empty bytes should not modify message");
    }

    #[test]
    fn append_output_skips_whitespace_only() {
        let mut message = String::from("prefix");
        super::append_output(&mut message, b"   \n  \t ", "STDOUT");
        assert_eq!(
            message, "prefix",
            "whitespace-only bytes should not modify message"
        );
    }

    #[test]
    fn execute_missing_url_param() {
        let dir = TempDir::new().unwrap();
        let result = FetchRepoTool.execute(serde_json::json!({}), &ctx(&dir));
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("missing required parameter: url"),
            "should mention missing url, got: {err}"
        );
    }

    #[test]
    fn destination_with_valid_custom_dest() {
        let dir = TempDir::new().unwrap();
        // Create the scratch dir so resolve_and_validate_path works
        std::fs::create_dir_all(dir.path().join(".zipcode-remote")).unwrap();
        let repo = parse_github_repo_url("https://github.com/devswha/patina").unwrap();
        let dest =
            destination_path(Some(".zipcode-remote/custom-name"), &repo, dir.path()).unwrap();
        assert_eq!(
            dest.strip_prefix(dir.path()).unwrap().to_string_lossy(),
            ".zipcode-remote/custom-name"
        );
    }

    #[test]
    fn parse_url_rejects_query_string() {
        let err = parse_github_repo_url("https://github.com/devswha/patina?tab=readme")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("query strings"),
            "should mention query strings, got: {err}"
        );
    }

    #[test]
    fn parse_url_rejects_fragment() {
        let err = parse_github_repo_url("https://github.com/devswha/patina#readme")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("query strings or fragments"),
            "should mention fragments, got: {err}"
        );
    }

    #[test]
    fn parse_url_rejects_too_many_path_segments() {
        let err = parse_github_repo_url("https://github.com/devswha/patina/tree/main")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("exactly") || err.contains("must be exactly"),
            "should reject nested paths, got: {err}"
        );
    }

    #[test]
    fn parse_url_with_dotgit_suffix() {
        let repo = parse_github_repo_url("https://github.com/devswha/patina.git").unwrap();
        assert_eq!(repo.name, "patina", ".git suffix should be stripped");
    }
}
