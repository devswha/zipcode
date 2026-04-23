use std::io::{self, Write};
use zipcode_tools::PermissionMode;

pub struct PermissionPolicy {
    mode: PermissionMode,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PermissionCheck {
    Allowed,
    NeedsApproval(String),
    Denied(String),
}

impl PermissionPolicy {
    #[must_use]
    pub const fn new(mode: PermissionMode) -> Self {
        Self { mode }
    }

    #[must_use]
    pub const fn mode(&self) -> PermissionMode {
        self.mode
    }

    pub const fn set_mode(&mut self, mode: PermissionMode) {
        self.mode = mode;
    }

    #[must_use]
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
                "read_file" | "glob_search" | "grep_search" | "tool_search" | "write_file"
                | "edit_file" | "todo_write" => PermissionCheck::Allowed,
                "bash" | "repl" => PermissionCheck::NeedsApproval(format!(
                    "Tool '{tool_name}' requires approval in workspace-write mode"
                )),
                _ => PermissionCheck::Denied(format!(
                    "Tool '{tool_name}' is not allowed in workspace-write mode"
                )),
            },
        }
    }

    /// Create a child `PermissionPolicy` from this parent.
    ///
    /// - `None` → clone the parent's mode verbatim.
    /// - `Some(m)` where `m` ≤ parent → adopt `m` (downgrade allowed).
    /// - `Some(m)` where `m` > parent → keep the parent's mode (escalation refused).
    #[must_use]
    pub fn inherit_for_child(&self, override_mode: Option<PermissionMode>) -> Self {
        let child_mode = match override_mode {
            None => self.mode,
            Some(m) => {
                if mode_rank(m) <= mode_rank(self.mode) {
                    m
                } else {
                    self.mode
                }
            }
        };
        Self { mode: child_mode }
    }

    /// Prompt user for Y/N approval. Returns true if approved.
    #[must_use]
    pub fn prompt_user(message: &str) -> bool {
        print!("{message} [Y/n] ");
        io::stdout().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        let trimmed = input.trim().to_lowercase();
        trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
    }
}

/// Numeric rank for permission level comparison. Higher = more permissive.
const fn mode_rank(mode: PermissionMode) -> u8 {
    match mode {
        PermissionMode::ReadOnly => 0,
        PermissionMode::WorkspaceWrite => 1,
        PermissionMode::FullAccess => 2,
    }
}

/// Convert a permission mode string to a `PermissionMode`.
///
/// # Errors
///
/// Returns an error when the permission mode is unsupported.
pub fn parse_permission_mode(s: &str) -> anyhow::Result<PermissionMode> {
    match s {
        "read-only" => Ok(PermissionMode::ReadOnly),
        "workspace-write" => Ok(PermissionMode::WorkspaceWrite),
        "full-access" | "danger-full-access" => Ok(PermissionMode::FullAccess),
        other => anyhow::bail!(
            "unsupported permission mode '{other}'. Expected one of: read-only, workspace-write, full-access"
        ),
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
    fn test_workspace_write_denies_agent() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
        match policy.check("agent", &serde_json::json!({})) {
            PermissionCheck::Denied(_) => {}
            other => panic!("Expected Denied, got {other:?}"),
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

    #[test]
    fn test_workspace_write_denies_unknown_tools() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
        match policy.check("future_exec_tool", &serde_json::json!({})) {
            PermissionCheck::Denied(_) => {}
            other => panic!("Expected Denied, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_permission_mode_accepts_known_values() {
        assert!(matches!(
            parse_permission_mode("read-only").unwrap(),
            PermissionMode::ReadOnly
        ));
        assert!(matches!(
            parse_permission_mode("workspace-write").unwrap(),
            PermissionMode::WorkspaceWrite
        ));
        assert!(matches!(
            parse_permission_mode("full-access").unwrap(),
            PermissionMode::FullAccess
        ));
    }

    #[test]
    fn test_parse_permission_mode_rejects_unknown_values() {
        let error = parse_permission_mode("workspace").unwrap_err().to_string();
        assert!(error.contains("unsupported permission mode"));
    }

    #[test]
    fn test_parse_permission_mode_accepts_danger_full_access() {
        assert!(matches!(
            parse_permission_mode("danger-full-access").unwrap(),
            PermissionMode::FullAccess
        ));
    }

    #[test]
    fn test_read_only_allows_all_read_tools() {
        let policy = PermissionPolicy::new(PermissionMode::ReadOnly);
        for tool in &["read_file", "glob_search", "grep_search", "tool_search"] {
            assert_eq!(
                policy.check(tool, &serde_json::json!({})),
                PermissionCheck::Allowed,
                "ReadOnly should allow '{tool}'"
            );
        }
    }

    #[test]
    fn test_workspace_write_allows_all_write_tools() {
        let policy = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
        for tool in &["write_file", "edit_file", "todo_write"] {
            assert_eq!(
                policy.check(tool, &serde_json::json!({})),
                PermissionCheck::Allowed,
                "WorkspaceWrite should allow '{tool}'"
            );
        }
    }

    #[test]
    fn test_full_access_check_returns_allowed_for_unknown() {
        let policy = PermissionPolicy::new(PermissionMode::FullAccess);
        assert_eq!(
            policy.check("nonexistent_future_tool", &serde_json::json!({})),
            PermissionCheck::Allowed,
            "FullAccess should allow unknown tool names"
        );
    }

    // ── inherit_for_child tests ───────────────────────────────────

    #[test]
    fn test_inherit_for_child_none_clones_parent() {
        let parent = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
        let child = parent.inherit_for_child(None);
        assert_eq!(child.mode(), PermissionMode::WorkspaceWrite);
    }

    #[test]
    fn test_inherit_for_child_downgrade_allowed() {
        let parent = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
        let child = parent.inherit_for_child(Some(PermissionMode::ReadOnly));
        assert_eq!(child.mode(), PermissionMode::ReadOnly);
    }

    #[test]
    fn test_inherit_for_child_same_mode_allowed() {
        let parent = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
        let child = parent.inherit_for_child(Some(PermissionMode::WorkspaceWrite));
        assert_eq!(child.mode(), PermissionMode::WorkspaceWrite);
    }

    #[test]
    fn test_inherit_for_child_escalation_refused() {
        let parent = PermissionPolicy::new(PermissionMode::ReadOnly);
        let child = parent.inherit_for_child(Some(PermissionMode::FullAccess));
        assert_eq!(child.mode(), PermissionMode::ReadOnly);
    }

    #[test]
    fn test_inherit_for_child_full_access_parent_allows_downgrade() {
        let parent = PermissionPolicy::new(PermissionMode::FullAccess);
        let child = parent.inherit_for_child(Some(PermissionMode::ReadOnly));
        assert_eq!(child.mode(), PermissionMode::ReadOnly);
    }

    #[test]
    fn test_inherit_for_child_workspace_write_escalation_to_full_access_refused() {
        let parent = PermissionPolicy::new(PermissionMode::WorkspaceWrite);
        let child = parent.inherit_for_child(Some(PermissionMode::FullAccess));
        assert_eq!(child.mode(), PermissionMode::WorkspaceWrite);
    }

    #[test]
    fn test_permission_policy_set_mode_changes_behavior() {
        let mut policy = PermissionPolicy::new(PermissionMode::FullAccess);
        // Bash is allowed under FullAccess
        assert_eq!(
            policy.check("bash", &serde_json::json!({})),
            PermissionCheck::Allowed
        );
        // Switch to ReadOnly — bash should now be denied
        policy.set_mode(PermissionMode::ReadOnly);
        match policy.check("bash", &serde_json::json!({})) {
            PermissionCheck::Denied(_) => {}
            other => panic!("Expected Denied after set_mode(ReadOnly), got {other:?}"),
        }
    }
}
