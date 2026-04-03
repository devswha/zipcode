use std::io::{self, Write};
use zipcode_tools::PermissionMode;

pub struct PermissionPolicy {
    mode: PermissionMode,
}

#[derive(Debug, PartialEq)]
pub enum PermissionCheck {
    Allowed,
    NeedsApproval(String),
    Denied(String),
}

impl PermissionPolicy {
    pub fn new(mode: PermissionMode) -> Self {
        Self { mode }
    }

    pub fn mode(&self) -> PermissionMode {
        self.mode
    }

    pub fn check(&self, tool_name: &str, _args: &serde_json::Value) -> PermissionCheck {
        match self.mode {
            PermissionMode::FullAccess => PermissionCheck::Allowed,
            PermissionMode::ReadOnly => match tool_name {
                "read_file" | "glob_search" | "grep_search" | "tool_search" => {
                    PermissionCheck::Allowed
                }
                _ => PermissionCheck::Denied(format!(
                    "Tool '{tool_name}' is not allowed in read-only mode"
                )),
            },
            PermissionMode::WorkspaceWrite => match tool_name {
                "bash" | "repl" => PermissionCheck::NeedsApproval(
                    "Bash execution requires approval in workspace-write mode".to_string(),
                ),
                _ => PermissionCheck::Allowed,
            },
        }
    }

    /// Prompt user for Y/N approval. Returns true if approved.
    pub fn prompt_user(message: &str) -> bool {
        print!("{message} [Y/n] ");
        io::stdout().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        let trimmed = input.trim().to_lowercase();
        trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
    }
}

/// Convert a permission mode string to a `PermissionMode`.
pub fn permission_mode_from_str(s: &str) -> PermissionMode {
    match s {
        "read-only" => PermissionMode::ReadOnly,
        "full-access" | "danger-full-access" => PermissionMode::FullAccess,
        _ => PermissionMode::WorkspaceWrite,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_access_allows_everything() {
        let policy = PermissionPolicy::new(PermissionMode::FullAccess);
        assert_eq!(
            policy.check("bash", &serde_json::json!({})),
            PermissionCheck::Allowed
        );
        assert_eq!(
            policy.check("write_file", &serde_json::json!({})),
            PermissionCheck::Allowed
        );
    }

    #[test]
    fn test_read_only_blocks_writes() {
        let policy = PermissionPolicy::new(PermissionMode::ReadOnly);
        assert_eq!(
            policy.check("read_file", &serde_json::json!({})),
            PermissionCheck::Allowed
        );
        match policy.check("bash", &serde_json::json!({})) {
            PermissionCheck::Denied(_) => {}
            other => panic!("Expected Denied, got {other:?}"),
        }
    }

    #[test]
    fn test_workspace_write_needs_approval_for_bash() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
        match policy.check("bash", &serde_json::json!({})) {
            PermissionCheck::NeedsApproval(_) => {}
            other => panic!("Expected NeedsApproval, got {other:?}"),
        }
        assert_eq!(
            policy.check("write_file", &serde_json::json!({})),
            PermissionCheck::Allowed
        );
    }

    #[test]
    fn test_workspace_write_needs_approval_for_repl() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
        match policy.check("repl", &serde_json::json!({})) {
            PermissionCheck::NeedsApproval(_) => {}
            other => panic!("Expected NeedsApproval, got {other:?}"),
        }
    }

    #[test]
    fn test_read_only_blocks_repl() {
        let policy = PermissionPolicy::new(PermissionMode::ReadOnly);
        match policy.check("repl", &serde_json::json!({})) {
            PermissionCheck::Denied(_) => {}
            other => panic!("Expected Denied, got {other:?}"),
        }
    }
}
