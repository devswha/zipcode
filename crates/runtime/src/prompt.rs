use std::fmt::Write;
use std::path::Path;

use crate::config::find_project_root;
use zipcode_inference::chat_template::ToolSpec;
use zipcode_tools::ToolRegistry;

const BASE_SYSTEM_PROMPT: &str = r"You are zipcode, an AI coding assistant running locally on the user's machine.
You help with software engineering tasks: writing code, debugging, refactoring, and explaining code.

You have access to tools for file operations, shell commands, and code search. Use them to help the user.

Key rules:
- Read files before modifying them
- Prefer editing existing files over creating new ones
- Run tests after making changes
- Be concise and direct
- Reply in the same natural language as the user's latest message. If the user writes Korean, keep the whole response in Korean unless they explicitly ask for another language or you are quoting code, commands, file contents, or error output.

## Tool Usage Guidelines

- **Prefer edit_file over write_file** when modifying existing files — edit_file does targeted replacement and is safer for large files; write_file overwrites the entire file and risks losing content if the rewrite is incomplete
- **Use `**/*` glob patterns** for project-wide searches — `*` only matches files in the current directory and will miss files in subdirectories (e.g. use `**/*.rs` not `*.rs` to find all Rust files)
- **When asked to analyze a repository or project path**, start by inspecting its README and primary manifest/build files (for example `Cargo.toml`, `package.json`, `pyproject.toml`) before asking follow-up questions, unless the user already asked for a narrower focus
- **When asked to analyze a GitHub repository URL**, do not treat the URL as a local file path. First use `fetch_repo` to clone it into `.zipcode-remote/<owner>__<repo>`, then inspect the README and primary manifest/build files. Use `bash` for manual `git clone --depth 1 ...` only if `fetch_repo` is unavailable or reports a recoverable git error.
- **Prefer paths relative to the current working directory** — if the working directory is already the repo root, use `README.md` not `repo-name/README.md`
- **If a file read or search fails because the path redundantly prefixes the current workspace name**, retry once without that leading repo-name segment before giving up
- **If a search fails**, retry with a broader pattern or inspect likely files directly before asking the user for the file location";

pub fn build_system_prompt(
    cwd: &Path,
    registry: &ToolRegistry,
    permission_mode: &str,
    skill_catalog: Option<&str>,
) -> (String, Vec<ToolSpec>) {
    let mut prompt = BASE_SYSTEM_PROMPT.to_string();
    let project_root = find_project_root(cwd);

    // Add permission context
    let _ = write!(prompt, "\n\nPermission mode: {permission_mode}");
    let _ = write!(prompt, "\nWorking directory: {}", cwd.display());

    // Inject skill catalog when available
    if let Some(catalog) = skill_catalog {
        if !catalog.is_empty() {
            prompt.push_str("\n\n## Available Skills\nInvoke with the `skill` tool using {name: \"...\", params: {...}}.\n\n");
            prompt.push_str(catalog);
        }
    }

    // Load .zipcode.md if present
    let memory_path = project_root.join(".zipcode.md");
    if memory_path.exists() {
        match std::fs::read_to_string(&memory_path) {
            Ok(content) => {
                prompt.push_str("\n\n# Project Instructions\n");
                prompt.push_str(&content);
            }
            Err(e) => {
                tracing::warn!(
                    path = %memory_path.display(),
                    error = %e,
                    ".zipcode.md exists but could not be read — project instructions will be missing from the system prompt"
                );
            }
        }
    }

    // Build tool specs
    let tool_specs: Vec<ToolSpec> = registry
        .specs()
        .into_iter()
        .map(|s| ToolSpec {
            name: s.name,
            description: s.description,
            parameters: s.parameters,
        })
        .collect();

    (prompt, tool_specs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_prompt_content() {
        let registry = ToolRegistry::new();
        let (prompt, _) =
            build_system_prompt(Path::new("/tmp"), &registry, "workspace-write", None);
        assert!(prompt.contains("zipcode"));
        assert!(prompt.contains("workspace-write"));
        assert!(prompt.contains("analyze a repository or project path"));
        assert!(prompt.contains("Reply in the same natural language"));
        assert!(prompt.contains("GitHub repository URL"));
        assert!(prompt.contains("fetch_repo"));
        assert!(prompt.contains("git clone --depth 1"));
        assert!(prompt.contains("Prefer paths relative to the current working directory"));
    }

    #[test]
    fn test_prompt_includes_zipcode_md() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".zipcode.md"), "Use Rust for everything").unwrap();
        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "full-access", None);
        assert!(prompt.contains("Use Rust for everything"));
    }

    #[test]
    fn test_prompt_includes_project_root_zipcode_md_from_subdir() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("src/bin");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.path().join(".zipcode.md"), "Root instructions").unwrap();
        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(&nested, &registry, "workspace-write", None);
        assert!(prompt.contains("Root instructions"));
        assert!(prompt.contains(&nested.display().to_string()));
    }

    #[test]
    fn test_prompt_omits_git_status_by_default() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join("dirty.txt"), "dirty").unwrap();

        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "workspace-write", None);
        assert!(!prompt.contains("# Git Status"));
        assert!(!prompt.contains("dirty.txt"));
    }

    #[test]
    fn test_tool_specs_populated_from_registry() {
        use zipcode_tools::{bash::BashTool, read_file::ReadFileTool, write_file::WriteFileTool};

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(BashTool));
        registry.register(Box::new(ReadFileTool));
        registry.register(Box::new(WriteFileTool));

        let dir = tempfile::TempDir::new().unwrap();
        let (_, specs) = build_system_prompt(dir.path(), &registry, "full-access", None);

        assert!(
            !specs.is_empty(),
            "tool specs should not be empty when tools are registered"
        );
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"bash"),
            "expected 'bash' in tool specs, got: {names:?}"
        );
        assert!(
            names.contains(&"read_file"),
            "expected 'read_file' in tool specs, got: {names:?}"
        );
        assert!(
            names.contains(&"write_file"),
            "expected 'write_file' in tool specs, got: {names:?}"
        );

        // Each spec should have a non-empty description and valid parameters schema
        for spec in &specs {
            assert!(!spec.name.is_empty(), "tool spec name should not be empty");
            assert!(
                !spec.description.is_empty(),
                "tool spec description should not be empty for {}",
                spec.name
            );
            assert!(
                spec.parameters.is_object(),
                "tool spec parameters should be a JSON object for {}",
                spec.name
            );
        }
    }

    #[test]
    fn test_empty_zipcode_md_does_not_corrupt_prompt() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".zipcode.md"), "").unwrap();

        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "workspace-write", None);

        // Should still have the "# Project Instructions" header even with empty file
        assert!(
            prompt.contains("# Project Instructions"),
            "empty .zipcode.md should still add the Project Instructions header"
        );
        // The prompt should be well-formed overall
        assert!(prompt.contains("zipcode"));
        assert!(prompt.contains("workspace-write"));
    }

    #[test]
    fn test_cwd_displayed_in_prompt() {
        let dir = tempfile::TempDir::new().unwrap();
        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "full-access", None);

        let expected_path = dir.path().display().to_string();
        assert!(
            prompt.contains(&expected_path),
            "prompt should contain the CWD path '{}', got prompt starting with: {}",
            expected_path,
            &prompt[..prompt.len().min(200)]
        );
    }

    #[test]
    fn test_no_zipcode_md_does_not_add_project_instructions_section() {
        let dir = tempfile::TempDir::new().unwrap();
        // Do NOT create .zipcode.md
        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "workspace-write", None);

        assert!(
            !prompt.contains("# Project Instructions"),
            "prompt should not contain Project Instructions section when .zipcode.md is absent"
        );
    }

    #[test]
    fn test_empty_registry_produces_empty_tool_specs() {
        let registry = ToolRegistry::new();
        let dir = tempfile::TempDir::new().unwrap();
        let (_, specs) = build_system_prompt(dir.path(), &registry, "read-only", None);

        assert!(
            specs.is_empty(),
            "empty registry should produce zero tool specs"
        );
    }

    #[test]
    fn test_unreadable_zipcode_md_does_not_panic() {
        let dir = tempfile::TempDir::new().unwrap();
        let md_path = dir.path().join(".zipcode.md");

        // Create a file and then make it unreadable.
        // On Linux, removing read permission from a file we own still allows
        // root to read it, so this test verifies that the code handles the
        // error path gracefully rather than panicking.
        std::fs::write(&md_path, "should not be read").unwrap();

        // Make the file unreadable (best-effort; may not work if running as root)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&md_path, std::fs::Permissions::from_mode(0o000)).unwrap();
        }

        let registry = ToolRegistry::new();
        // The function must not panic regardless of whether the file is readable
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "workspace-write", None);

        // On non-root Unix, the file is unreadable so content should NOT appear.
        // On root or non-Unix, the content WILL appear. Either way, no panic.
        let _ = prompt; // just verify we got here

        // Restore permissions so tempfile cleanup works
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&md_path, std::fs::Permissions::from_mode(0o644));
        }
    }

    #[test]
    fn test_prompt_injects_skill_catalog_when_present() {
        let registry = ToolRegistry::new();
        let dir = tempfile::TempDir::new().unwrap();
        let catalog = "review: Review recently changed code\nsearch: Search the codebase";
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "full-access", Some(catalog));

        assert!(
            prompt.contains("## Available Skills"),
            "prompt should contain skill catalog header"
        );
        assert!(
            prompt.contains("review: Review recently changed code"),
            "prompt should contain skill entry"
        );
        assert!(
            prompt.contains("search: Search the codebase"),
            "prompt should contain second skill entry"
        );
    }

    #[test]
    fn test_prompt_omits_skill_section_when_catalog_none_or_empty() {
        let registry = ToolRegistry::new();
        let dir = tempfile::TempDir::new().unwrap();

        let (prompt_none, _) = build_system_prompt(dir.path(), &registry, "full-access", None);
        assert!(
            !prompt_none.contains("## Available Skills"),
            "None catalog should not add skill section"
        );

        let (prompt_empty, _) = build_system_prompt(dir.path(), &registry, "full-access", Some(""));
        assert!(
            !prompt_empty.contains("## Available Skills"),
            "empty catalog should not add skill section"
        );
    }
}
