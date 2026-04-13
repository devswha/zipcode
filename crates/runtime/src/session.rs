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

fn session_path(id: &str) -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(format!(".zipcode/sessions/{id}.json"))
}

fn validate_session_id(id: &str) -> Result<()> {
    let is_valid = !id.is_empty() && id.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '-');
    if is_valid {
        Ok(())
    } else {
        anyhow::bail!("Invalid session id: {id}");
    }
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
        summary.push_str(&format!(
            "\n- {extra_count} additional earlier messages were compacted."
        ));
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
}
