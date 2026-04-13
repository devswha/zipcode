use std::path::Path;

use crate::config::find_project_root;
use zipcode_inference::chat_template::ToolSpec;
use zipcode_tools::ToolRegistry;

const BASE_SYSTEM_PROMPT: &str = r#"You are zipcode, an AI coding assistant running locally on the user's machine.
You help with software engineering tasks: writing code, debugging, refactoring, and explaining code.

You have access to tools for file operations, shell commands, and code search. Use them to help the user.

Key rules:
- Read files before modifying them
- Prefer editing existing files over creating new ones
- Run tests after making changes
- Be concise and direct

## Tool Usage Guidelines

- **Prefer edit_file over write_file** when modifying existing files — edit_file does targeted replacement and is safer for large files; write_file overwrites the entire file and risks losing content if the rewrite is incomplete
- **Use `**/*` glob patterns** for project-wide searches — `*` only matches files in the current directory and will miss files in subdirectories (e.g. use `**/*.rs` not `*.rs` to find all Rust files)
- **When asked to analyze a repository or project path**, start by inspecting its README and primary manifest/build files (for example `Cargo.toml`, `package.json`, `pyproject.toml`) before asking follow-up questions, unless the user already asked for a narrower focus
- **Prefer paths relative to the current working directory** — if the working directory is already the repo root, use `README.md` not `repo-name/README.md`
- **If a file read or search fails because the path redundantly prefixes the current workspace name**, retry once without that leading repo-name segment before giving up
- **If a search fails**, retry with a broader pattern or inspect likely files directly before asking the user for the file location"#;

pub fn build_system_prompt(
    cwd: &Path,
    registry: &ToolRegistry,
    permission_mode: &str,
) -> (String, Vec<ToolSpec>) {
    let mut prompt = BASE_SYSTEM_PROMPT.to_string();
    let project_root = find_project_root(cwd);

    // Add permission context
    prompt.push_str(&format!("\n\nPermission mode: {permission_mode}"));
    prompt.push_str(&format!("\nWorking directory: {}", cwd.display()));

    // Load .zipcode.md if present
    let memory_path = project_root.join(".zipcode.md");
    if memory_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&memory_path) {
            prompt.push_str("\n\n# Project Instructions\n");
            prompt.push_str(&content);
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
        let (prompt, _) = build_system_prompt(Path::new("/tmp"), &registry, "workspace-write");
        assert!(prompt.contains("zipcode"));
        assert!(prompt.contains("workspace-write"));
        assert!(prompt.contains("analyze a repository or project path"));
        assert!(prompt.contains("Prefer paths relative to the current working directory"));
    }

    #[test]
    fn test_prompt_includes_zipcode_md() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".zipcode.md"), "Use Rust for everything").unwrap();
        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "full-access");
        assert!(prompt.contains("Use Rust for everything"));
    }

    #[test]
    fn test_prompt_includes_project_root_zipcode_md_from_subdir() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("src/bin");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.path().join(".zipcode.md"), "Root instructions").unwrap();
        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(&nested, &registry, "workspace-write");
        assert!(prompt.contains("Root instructions"));
        assert!(prompt.contains(&nested.display().to_string()));
    }

    #[test]
    fn test_prompt_omits_git_status_by_default() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join("dirty.txt"), "dirty").unwrap();

        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "workspace-write");
        assert!(!prompt.contains("# Git Status"));
        assert!(!prompt.contains("dirty.txt"));
    }
}
