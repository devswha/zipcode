use std::path::Path;

use zipcode_inference::chat_template::ToolSpec;
use zipcode_tools::ToolRegistry;

const BASE_SYSTEM_PROMPT: &str = r#"You are zipcode, an AI coding assistant running locally on the user's machine.
You help with software engineering tasks: writing code, debugging, refactoring, and explaining code.

You have access to tools for file operations, shell commands, and code search. Use them to help the user.

Key rules:
- Read files before modifying them
- Prefer editing existing files over creating new ones
- Run tests after making changes
- Be concise and direct"#;

pub fn build_system_prompt(
    cwd: &Path,
    registry: &ToolRegistry,
    permission_mode: &str,
) -> (String, Vec<ToolSpec>) {
    let mut prompt = BASE_SYSTEM_PROMPT.to_string();

    // Add permission context
    prompt.push_str(&format!("\n\nPermission mode: {permission_mode}"));
    prompt.push_str(&format!("\nWorking directory: {}", cwd.display()));

    // Load .zipcode.md if present
    let memory_path = cwd.join(".zipcode.md");
    if memory_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&memory_path) {
            prompt.push_str("\n\n# Project Instructions\n");
            prompt.push_str(&content);
        }
    }

    // Git status
    if let Ok(output) = std::process::Command::new("git")
        .args(["status", "--short"])
        .current_dir(cwd)
        .output()
    {
        if output.status.success() {
            let status = String::from_utf8_lossy(&output.stdout);
            if !status.is_empty() {
                prompt.push_str("\n\n# Git Status\n```\n");
                prompt.push_str(&status);
                prompt.push_str("```");
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
        let (prompt, _) = build_system_prompt(Path::new("/tmp"), &registry, "workspace-write");
        assert!(prompt.contains("zipcode"));
        assert!(prompt.contains("workspace-write"));
    }

    #[test]
    fn test_prompt_includes_zipcode_md() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".zipcode.md"), "Use Rust for everything").unwrap();
        let registry = ToolRegistry::new();
        let (prompt, _) = build_system_prompt(dir.path(), &registry, "full-access");
        assert!(prompt.contains("Use Rust for everything"));
    }
}
