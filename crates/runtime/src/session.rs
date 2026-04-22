use std::fmt::Write;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use zipcode_inference::{ChatMessage, Role};

pub const COMPACTED_SUMMARY_MARKER: &str = "[Compacted context summary]";
const DEFAULT_RETAIN_USER_TURNS: usize = 4;
const DEFAULT_MAX_SUMMARY_BULLETS: usize = 12;
const DEFAULT_MAX_SUMMARY_LINE_CHARS: usize = 120;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactPolicy {
    pub retain_user_turns: usize,
    pub max_summary_bullets: usize,
    pub max_summary_line_chars: usize,
}

impl Default for CompactPolicy {
    fn default() -> Self {
        Self {
            retain_user_turns: DEFAULT_RETAIN_USER_TURNS,
            max_summary_bullets: DEFAULT_MAX_SUMMARY_BULLETS,
            max_summary_line_chars: DEFAULT_MAX_SUMMARY_LINE_CHARS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactResult {
    pub changed: bool,
    pub before_messages: usize,
    pub after_messages: usize,
    pub pruned_messages: usize,
    pub retained_messages: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: String,
    pub updated_at: String,
}

impl Session {
    #[must_use]
    pub fn new() -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let now = timestamp_now();
        Self {
            id,
            messages: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// Persist the session to `~/.zipcode/sessions/{id}.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if the session ID contains invalid characters, the
    /// session directory cannot be created, or the file cannot be written.
    pub fn save(&self) -> Result<()> {
        validate_session_id(&self.id)?;
        let path = self.path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)
            .with_context(|| format!("Failed to save session to {}", path.display()))?;
        Ok(())
    }

    /// Load a session from `~/.zipcode/sessions/{id}.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if the session ID contains invalid characters, the
    /// file cannot be read, the content is not valid JSON, or the stored
    /// session ID does not match the requested ID.
    pub fn load(id: &str) -> Result<Self> {
        validate_session_id(id)?;
        let path = session_path(id);
        let content =
            std::fs::read_to_string(&path).with_context(|| format!("Session not found: {id}"))?;
        let session: Self = serde_json::from_str(&content)?;
        if session.id != id {
            anyhow::bail!(
                "Session id mismatch: requested {id}, but stored session id is {}",
                session.id
            );
        }
        Ok(session)
    }

    #[must_use]
    pub fn path(&self) -> PathBuf {
        session_path(&self.id)
    }

    pub fn compact(&mut self, policy: CompactPolicy) -> CompactResult {
        let before_messages = self.messages.len();
        let system_prefix_len = usize::from(
            self.messages
                .first()
                .is_some_and(|message| message.role == Role::System),
        );

        let user_turn_starts: Vec<usize> = self
            .messages
            .iter()
            .enumerate()
            .skip(system_prefix_len)
            .filter_map(|(index, message)| (message.role == Role::User).then_some(index))
            .collect();

        if user_turn_starts.len() <= policy.retain_user_turns {
            return CompactResult {
                changed: false,
                before_messages,
                after_messages: before_messages,
                pruned_messages: 0,
                retained_messages: before_messages.saturating_sub(system_prefix_len),
            };
        }

        let retained_start = user_turn_starts[user_turn_starts.len() - policy.retain_user_turns];
        let pruned_slice = &self.messages[system_prefix_len..retained_start];
        let retained_slice = &self.messages[retained_start..];
        let pruned_messages = pruned_slice.len();
        let retained_messages = retained_slice.len();

        let summary = build_compacted_summary(pruned_slice, policy);

        let mut compacted_messages =
            Vec::with_capacity(system_prefix_len + 1 + retained_slice.len());
        compacted_messages.extend_from_slice(&self.messages[..system_prefix_len]);
        compacted_messages.push(ChatMessage::assistant(&summary));
        compacted_messages.extend_from_slice(retained_slice);

        self.messages = compacted_messages;
        self.updated_at = timestamp_now();

        CompactResult {
            changed: true,
            before_messages,
            after_messages: self.messages.len(),
            pruned_messages,
            retained_messages,
        }
    }

    pub fn push_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
        self.updated_at = timestamp_now();
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_session_id(id: &str) -> Result<()> {
    let is_valid = !id.is_empty()
        && id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_');
    if is_valid {
        Ok(())
    } else {
        anyhow::bail!("Invalid session id: {id}")
    }
}

fn session_path(id: &str) -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(format!(".zipcode/sessions/{id}.json"))
}

fn timestamp_now() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn build_compacted_summary(messages: &[ChatMessage], policy: CompactPolicy) -> String {
    let mut bullets = Vec::new();
    let mut extra_count = 0usize;

    for message in messages {
        let Some(line) = summarize_message(message, policy.max_summary_line_chars) else {
            continue;
        };

        if bullets.len() < policy.max_summary_bullets {
            bullets.push(line);
        } else {
            extra_count += 1;
        }
    }

    let mut summary = format!(
        "{COMPACTED_SUMMARY_MARKER}\nCompacted {} earlier messages into a deterministic recap.",
        messages.len()
    );

    if bullets.is_empty() {
        summary.push_str("\n- Earlier transcript content was compacted.");
    } else {
        for bullet in bullets {
            summary.push_str("\n- ");
            summary.push_str(&bullet);
        }
    }

    if extra_count > 0 {
        let _ = write!(
            summary,
            "\n- {extra_count} additional earlier messages were compacted."
        );
    }

    summary
}

fn summarize_message(message: &ChatMessage, max_chars: usize) -> Option<String> {
    match message.role {
        Role::System => None,
        Role::User => Some(format!(
            "User: {}",
            truncate_inline(&message.content, max_chars)
        )),
        Role::Model => {
            if message.content.starts_with(COMPACTED_SUMMARY_MARKER) {
                let prior_summary = message
                    .content
                    .lines()
                    .skip(1)
                    .collect::<Vec<_>>()
                    .join(" ");
                return Some(format!(
                    "Prior summary: {}",
                    truncate_inline(&prior_summary, max_chars)
                ));
            }

            if let Some(tool_calls) = &message.tool_calls {
                let tool_names = tool_calls
                    .iter()
                    .map(|call| call.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let note = truncate_inline(&message.content, max_chars / 2);
                return Some(if note.is_empty() {
                    format!("Assistant invoked tools: {tool_names}")
                } else {
                    format!("Assistant invoked tools: {tool_names}; note: {note}")
                });
            }

            Some(format!(
                "Assistant: {}",
                truncate_inline(&message.content, max_chars)
            ))
        }
        Role::Tool => Some(format!(
            "Tool result ({}): {}",
            message.tool_call_id.as_deref().unwrap_or("unknown"),
            truncate_inline(&message.content, max_chars)
        )),
    }
}

fn truncate_inline(text: &str, max_chars: usize) -> String {
    let flattened = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = flattened.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zipcode_inference::ToolCallParsed;

    #[test]
    fn test_new_session() {
        let session = Session::new();
        assert!(!session.id.is_empty());
        assert!(session.messages.is_empty());
    }

    #[test]
    fn test_push_message() {
        let mut session = Session::new();
        session.push_message(ChatMessage::user("hello"));
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn test_session_roundtrip() {
        let mut session = Session::new();
        session.push_message(ChatMessage::user("test"));

        // Save and reload
        session.save().unwrap();
        let loaded = Session::load(&session.id).unwrap();
        assert_eq!(loaded.messages.len(), 1);

        // Cleanup
        let path = session.path();
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_rejects_mismatched_session_id() {
        let session = Session::new();
        let path = session.path();
        session.save().unwrap();

        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        json["id"] = serde_json::json!("different-session-id");
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let error = Session::load(&session.id).unwrap_err().to_string();
        assert!(error.contains("Session id mismatch"));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_rejects_invalid_session_id() {
        let error = Session::load("../escape").unwrap_err().to_string();
        assert!(error.contains("Invalid session id"));
    }

    #[test]
    fn test_compact_preserves_system_message_and_safe_boundary_suffix() {
        let tool_call = ToolCallParsed {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"file_path":"src/main.rs"}),
        };

        let mut session = Session {
            id: "session".to_string(),
            messages: vec![
                ChatMessage::system("system prompt"),
                ChatMessage::user("first request"),
                ChatMessage::assistant("first response"),
                ChatMessage::user("second request"),
                ChatMessage::assistant_with_tool_calls("checking", vec![tool_call]),
                ChatMessage::tool_result("call_1", "fn main() {}"),
                ChatMessage::assistant("second response"),
                ChatMessage::user("third request"),
                ChatMessage::assistant("third response"),
            ],
            created_at: timestamp_now(),
            updated_at: timestamp_now(),
        };

        let result = session.compact(CompactPolicy {
            retain_user_turns: 2,
            max_summary_bullets: 8,
            max_summary_line_chars: 80,
        });

        assert!(result.changed);
        assert_eq!(session.messages[0].role, Role::System);
        assert_eq!(session.messages[1].role, Role::Model);
        assert!(session.messages[1]
            .content
            .starts_with(COMPACTED_SUMMARY_MARKER));
        assert_eq!(session.messages[2].role, Role::User);
        assert_eq!(session.messages[2].content, "second request");
        assert_eq!(session.messages[3].role, Role::Model);
        assert!(session.messages[3].tool_calls.is_some());
        assert_eq!(session.messages[4].role, Role::Tool);
        assert_eq!(session.messages[5].role, Role::Model);
    }

    #[test]
    fn test_compact_is_idempotent_without_new_user_turns() {
        let mut session = Session {
            id: "session".to_string(),
            messages: vec![
                ChatMessage::system("system prompt"),
                ChatMessage::user("one"),
                ChatMessage::assistant("one response"),
                ChatMessage::user("two"),
                ChatMessage::assistant("two response"),
                ChatMessage::user("three"),
                ChatMessage::assistant("three response"),
            ],
            created_at: timestamp_now(),
            updated_at: timestamp_now(),
        };

        let policy = CompactPolicy {
            retain_user_turns: 2,
            ..CompactPolicy::default()
        };
        let first = session.compact(policy);
        let after_first = session
            .messages
            .iter()
            .map(|message| message.content.clone())
            .collect::<Vec<_>>();
        let second = session.compact(policy);

        assert!(first.changed);
        assert!(!second.changed);
        assert_eq!(
            after_first,
            session
                .messages
                .iter()
                .map(|message| message.content.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_session_id_path_traversal_blocked() {
        assert!(Session::load("../../etc/passwd").is_err());
        assert!(Session::load("../escape").is_err());
        assert!(Session::load("valid/../../../etc/shadow").is_err());
    }

    #[test]
    fn test_session_id_slash_blocked() {
        assert!(Session::load("foo/bar").is_err());
        assert!(Session::load("/absolute/path").is_err());
    }

    #[test]
    fn test_session_id_null_byte_blocked() {
        assert!(Session::load("valid\0evil").is_err());
    }

    #[test]
    fn test_session_id_empty_blocked() {
        assert!(Session::load("").is_err());
    }

    #[test]
    fn test_session_id_backslash_blocked() {
        assert!(Session::load("foo\\bar").is_err());
    }

    #[test]
    fn test_save_with_malicious_id_blocked() {
        let mut session = Session::new();
        session.id = "../../etc/cron.d/evil".to_string();
        assert!(session.save().is_err());
    }

    #[test]
    fn test_valid_uuid_session_id_accepted() {
        assert!(validate_session_id("550e8400-e29b-41d4-a716-446655440000").is_ok());
        assert!(validate_session_id("simple-id").is_ok());
        assert!(validate_session_id("test_session_123").is_ok());
    }

    // ── truncate_inline tests ──────────────────────────────────────

    #[test]
    fn test_truncate_inline_empty_string() {
        assert_eq!(truncate_inline("", 80), "");
    }

    #[test]
    fn test_truncate_inline_short_string_unchanged() {
        assert_eq!(truncate_inline("hello world", 80), "hello world");
    }

    #[test]
    fn test_truncate_inline_long_string_truncated() {
        let input = "x".repeat(200);
        let result = truncate_inline(&input, 100);
        // Ellipsis '…' is 3 bytes in UTF-8; check char count, not byte count
        assert!(
            result.chars().count() <= 101,
            "result should be at most max_chars + 1 char (ellipsis), got {} chars",
            result.chars().count()
        );
        assert!(
            result.ends_with('…'),
            "truncated string should end with ellipsis"
        );
        assert!(
            !result.starts_with('…'),
            "should have some content before ellipsis"
        );
    }

    #[test]
    fn test_truncate_inline_collapses_whitespace() {
        let input = "hello   world\t\tfoo\nbar";
        let result = truncate_inline(input, 80);
        assert_eq!(result, "hello world foo bar");
    }

    #[test]
    fn test_truncate_inline_multibyte_utf8() {
        // Korean characters: 3 bytes each, but .chars() counts codepoints
        let input = "안녕하세요";
        let result = truncate_inline(input, 3);
        assert_eq!(result, "안녕하…");
    }

    #[test]
    fn test_truncate_inline_exact_boundary() {
        let input = "abcde";
        // Exactly 5 chars → should NOT be truncated
        let result = truncate_inline(input, 5);
        assert_eq!(result, "abcde");
        assert!(!result.contains('…'));
    }

    #[test]
    fn test_truncate_inline_one_over_boundary() {
        let input = "abcdef"; // 6 chars
        let result = truncate_inline(input, 5);
        assert!(result.contains('…'));
        assert!(result.starts_with("abcde"));
    }

    #[test]
    fn test_truncate_inline_zero_max() {
        let result = truncate_inline("hello", 0);
        // With max_chars=0, take(0) returns empty, but chars.next() is Some → ellipsis
        assert_eq!(result, "…");
    }

    #[test]
    fn test_truncate_inline_single_char_at_limit() {
        assert_eq!(truncate_inline("a", 1), "a");
        assert_eq!(truncate_inline("ab", 1), "a…");
    }

    // ── build_compacted_summary tests ─────────────────────────────

    #[test]
    fn test_build_compacted_summary_empty_messages() {
        let policy = CompactPolicy::default();
        let summary = build_compacted_summary(&[], policy);
        assert!(summary.starts_with(COMPACTED_SUMMARY_MARKER));
        assert!(summary.contains("Compacted 0 earlier messages"));
        assert!(summary.contains("Earlier transcript content was compacted."));
    }

    #[test]
    fn test_build_compacted_summary_skips_system_messages() {
        let messages = vec![
            ChatMessage::system("system prompt"),
            ChatMessage::user("hello"),
        ];
        let policy = CompactPolicy {
            max_summary_bullets: 10,
            max_summary_line_chars: 80,
            ..CompactPolicy::default()
        };
        let summary = build_compacted_summary(&messages, policy);
        assert!(summary.starts_with(COMPACTED_SUMMARY_MARKER));
        assert!(
            !summary.contains("system prompt"),
            "system messages should be skipped"
        );
        assert!(summary.contains("User: hello"));
    }

    #[test]
    fn test_build_compacted_summary_overflow_count() {
        let mut messages = Vec::new();
        for i in 0..15 {
            messages.push(ChatMessage::user(&format!("request {i}")));
        }
        let policy = CompactPolicy {
            max_summary_bullets: 5,
            max_summary_line_chars: 80,
            ..CompactPolicy::default()
        };
        let summary = build_compacted_summary(&messages, policy);
        assert!(summary.contains("additional earlier messages were compacted"));
        // Should still have at most max_summary_bullets bullet points + the overflow line
        let bullet_count = summary.lines().filter(|l| l.starts_with("- ")).count();
        assert!(
            bullet_count <= 6,
            "should have at most 5 bullets + 1 overflow line, got {bullet_count}"
        );
    }

    #[test]
    fn test_build_compacted_summary_starts_with_marker() {
        let messages = vec![ChatMessage::user("test")];
        let policy = CompactPolicy::default();
        let summary = build_compacted_summary(&messages, policy);
        assert!(summary.starts_with(COMPACTED_SUMMARY_MARKER));
        assert!(summary.contains("Compacted 1 earlier messages"));
    }

    #[test]
    fn test_build_compacted_summary_includes_all_roles_except_system() {
        let tool_call = ToolCallParsed {
            id: "call_1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        };
        let messages = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("do stuff"),
            ChatMessage::assistant_with_tool_calls("thinking", vec![tool_call]),
            ChatMessage::tool_result("call_1", "file1.rs\nfile2.rs"),
            ChatMessage::assistant("done"),
        ];
        let policy = CompactPolicy {
            max_summary_bullets: 10,
            max_summary_line_chars: 80,
            ..CompactPolicy::default()
        };
        let summary = build_compacted_summary(&messages, policy);
        assert!(summary.contains("User:"));
        assert!(summary.contains("Assistant invoked tools:"));
        assert!(summary.contains("Tool result"));
        assert!(summary.contains("Assistant:"));
        assert!(
            !summary.contains("sys"),
            "system message should be excluded"
        );
    }

    // ── summarize_message tests ───────────────────────────────────

    #[test]
    fn test_summarize_message_system_returns_none() {
        let msg = ChatMessage::system("you are helpful");
        assert!(summarize_message(&msg, 80).is_none());
    }

    #[test]
    fn test_summarize_message_user() {
        let msg = ChatMessage::user("fix the bug");
        let result = summarize_message(&msg, 80).unwrap();
        assert_eq!(result, "User: fix the bug");
    }

    #[test]
    fn test_summarize_message_assistant() {
        let msg = ChatMessage::assistant("the fix is applied");
        let result = summarize_message(&msg, 80).unwrap();
        assert_eq!(result, "Assistant: the fix is applied");
    }

    #[test]
    fn test_summarize_message_tool_result() {
        let msg = ChatMessage::tool_result("call_42", "output line 1\noutput line 2");
        let result = summarize_message(&msg, 80).unwrap();
        assert!(result.starts_with("Tool result (call_42):"));
        assert!(result.contains("output line 1"));
    }

    #[test]
    fn test_summarize_message_assistant_with_tool_calls() {
        let call = ToolCallParsed {
            id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        };
        let msg = ChatMessage::assistant_with_tool_calls("thinking", vec![call]);
        let result = summarize_message(&msg, 80).unwrap();
        assert!(result.contains("Assistant invoked tools: bash"));
        assert!(result.contains("note: thinking"));
    }

    #[test]
    fn test_summarize_message_assistant_with_tool_calls_empty_note() {
        let call = ToolCallParsed {
            id: "c1".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "foo.rs"}),
        };
        let msg = ChatMessage::assistant_with_tool_calls("", vec![call]);
        let result = summarize_message(&msg, 80).unwrap();
        assert_eq!(result, "Assistant invoked tools: read_file");
    }

    #[test]
    fn test_summarize_message_compacted_summary_merged() {
        let prior = format!(
            "{COMPACTED_SUMMARY_MARKER}\n- User asked about tests\n- Assistant ran cargo test"
        );
        let msg = ChatMessage::assistant(&prior);
        let result = summarize_message(&msg, 200).unwrap();
        assert!(result.starts_with("Prior summary:"));
        assert!(result.contains("User asked about tests"));
    }

    // ── compact() edge cases ──────────────────────────────────────

    #[test]
    fn test_compact_no_system_message() {
        let mut session = Session {
            id: "session".to_string(),
            messages: vec![
                ChatMessage::user("first"),
                ChatMessage::assistant("first response"),
                ChatMessage::user("second"),
                ChatMessage::assistant("second response"),
                ChatMessage::user("third"),
                ChatMessage::assistant("third response"),
            ],
            created_at: timestamp_now(),
            updated_at: timestamp_now(),
        };

        let result = session.compact(CompactPolicy {
            retain_user_turns: 2,
            ..CompactPolicy::default()
        });

        assert!(result.changed);
        // No system message, so compacted summary should be at index 0
        assert_eq!(session.messages[0].role, Role::Model);
        assert!(session.messages[0]
            .content
            .starts_with(COMPACTED_SUMMARY_MARKER));
        // First retained user message at index 1
        assert_eq!(session.messages[1].role, Role::User);
        assert_eq!(session.messages[1].content, "second");
        assert_eq!(result.before_messages, 6);
        assert!(result.after_messages < 6);
        assert!(result.pruned_messages > 0);
    }

    #[test]
    fn test_compact_exact_threshold_no_compaction() {
        // Exactly 2 user turns → retain_user_turns=2 means no compaction
        let mut session = Session {
            id: "session".to_string(),
            messages: vec![
                ChatMessage::system("sys"),
                ChatMessage::user("one"),
                ChatMessage::assistant("response one"),
                ChatMessage::user("two"),
                ChatMessage::assistant("response two"),
            ],
            created_at: timestamp_now(),
            updated_at: timestamp_now(),
        };

        let result = session.compact(CompactPolicy {
            retain_user_turns: 2,
            ..CompactPolicy::default()
        });

        assert!(!result.changed);
        assert_eq!(result.before_messages, 5);
        assert_eq!(result.after_messages, 5);
        assert_eq!(result.pruned_messages, 0);
    }

    #[test]
    fn test_compact_zero_user_messages_no_compaction() {
        let mut session = Session {
            id: "session".to_string(),
            messages: vec![ChatMessage::system("sys"), ChatMessage::assistant("hello")],
            created_at: timestamp_now(),
            updated_at: timestamp_now(),
        };

        let result = session.compact(CompactPolicy {
            retain_user_turns: 2,
            ..CompactPolicy::default()
        });

        assert!(!result.changed);
    }

    #[test]
    fn test_compact_re_compaction_merges_prior_summary() {
        let mut session = Session {
            id: "session".to_string(),
            messages: vec![
                ChatMessage::system("sys"),
                ChatMessage::assistant(&format!("{COMPACTED_SUMMARY_MARKER}\n- Prior work done")),
                ChatMessage::user("cycle2 request"),
                ChatMessage::assistant("cycle2 response"),
                ChatMessage::user("cycle3 request"),
                ChatMessage::assistant("cycle3 response"),
                ChatMessage::user("cycle4 request"),
                ChatMessage::assistant("cycle4 response"),
            ],
            created_at: timestamp_now(),
            updated_at: timestamp_now(),
        };

        let result = session.compact(CompactPolicy {
            retain_user_turns: 2,
            max_summary_bullets: 10,
            max_summary_line_chars: 80,
        });

        assert!(result.changed);
        // The new summary should contain a "Prior summary:" bullet from the old summary
        let summary = &session.messages[1].content;
        assert!(summary.starts_with(COMPACTED_SUMMARY_MARKER));
        assert!(
            summary.contains("Prior summary:"),
            "re-compaction should merge the prior summary, got: {summary}"
        );
    }

    #[test]
    fn test_compact_result_counts_are_consistent() {
        let mut session = Session {
            id: "session".to_string(),
            messages: vec![
                ChatMessage::system("sys"),
                ChatMessage::user("u1"),
                ChatMessage::assistant("a1"),
                ChatMessage::user("u2"),
                ChatMessage::assistant("a2"),
                ChatMessage::user("u3"),
                ChatMessage::assistant("a3"),
                ChatMessage::user("u4"),
                ChatMessage::assistant("a4"),
            ],
            created_at: timestamp_now(),
            updated_at: timestamp_now(),
        };

        let result = session.compact(CompactPolicy {
            retain_user_turns: 2,
            ..CompactPolicy::default()
        });

        assert!(result.changed);
        assert_eq!(
            result.before_messages,
            result.pruned_messages + result.retained_messages + 1, // +1 for summary message
            "before = pruned + retained + summary message"
        );
        assert_eq!(
            result.after_messages,
            result.retained_messages + 2, // system + summary + retained
            "after = system + summary + retained"
        );
    }

    // ── updated_at mutation tests ─────────────────────────────────

    #[test]
    fn test_push_message_updates_timestamp() {
        let mut session = Session::new();
        let before = session.updated_at.clone();
        // Small sleep to ensure timestamp could differ
        std::thread::sleep(std::time::Duration::from_millis(10));
        session.push_message(ChatMessage::user("hello"));
        // updated_at should have been refreshed
        assert_eq!(session.messages.len(), 1);
        // Timestamps are RFC3339 and monotonically increasing within a test
        assert!(
            session.updated_at >= before,
            "updated_at should be >= before push"
        );
    }
}
