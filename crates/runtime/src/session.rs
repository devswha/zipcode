use std::fmt::Write;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use zipcode_inference::{ChatMessage, Role};

pub const COMPACTED_SUMMARY_MARKER: &str = "[Compacted context summary]";
pub const TOOL_PAIR_SUMMARY_MARKER: &str = "[compacted tool calls]";

const DEFAULT_RETAIN_USER_TURNS: usize = 4;
const DEFAULT_MAX_SUMMARY_BULLETS: usize = 12;
const DEFAULT_MAX_SUMMARY_LINE_CHARS: usize = 120;
const DEFAULT_TIER1_PAIR_CUTOFF: usize = 10;
const DEFAULT_TIER1_BATCH_SIZE: usize = 10;
const DEFAULT_TIER2_USAGE_THRESHOLD: f64 = 0.8;
/// Matches `zipcode_inference::DEFAULT_CONTEXT_SIZE` (Gemma 4's 128K native
/// window, also the default for `ZIPCODE_LLAMA_SERVER_CTX`). With the prior
/// 32_768 default, tier-2 was triggering at 26_214 effective tokens — about
/// 20 % of the actual window — and evicting tool responses prematurely on
/// any deployment that didn't manually trim `CompactPolicy`.
const DEFAULT_CONTEXT_WINDOW_TOKENS: usize = 131_072;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactPolicy {
    pub retain_user_turns: usize,
    pub max_summary_bullets: usize,
    pub max_summary_line_chars: usize,
    /// Number of visible tool pairs above which tier-1 compaction triggers.
    pub tier1_pair_cutoff: usize,
    /// Number of tool pairs to summarise in a single tier-1 compaction pass.
    pub tier1_batch_size: usize,
    /// Context-usage fraction (0.0–1.0) at which tier-2 eviction triggers.
    pub tier2_usage_threshold: f64,
    /// Estimated total context window size in tokens (used for tier-2 ratio).
    pub context_window_tokens: usize,
}

impl Default for CompactPolicy {
    fn default() -> Self {
        Self {
            retain_user_turns: DEFAULT_RETAIN_USER_TURNS,
            max_summary_bullets: DEFAULT_MAX_SUMMARY_BULLETS,
            max_summary_line_chars: DEFAULT_MAX_SUMMARY_LINE_CHARS,
            tier1_pair_cutoff: DEFAULT_TIER1_PAIR_CUTOFF,
            tier1_batch_size: DEFAULT_TIER1_BATCH_SIZE,
            tier2_usage_threshold: DEFAULT_TIER2_USAGE_THRESHOLD,
            context_window_tokens: DEFAULT_CONTEXT_WINDOW_TOKENS,
        }
    }
}

impl CompactPolicy {
    /// Resolve the tier-2 eviction threshold in tokens, guarding against NaN,
    /// infinity, and out-of-range threshold values by clamping to `[0, 1]`.
    /// Unsafe callers would otherwise compute `threshold = 0` from NaN via
    /// `as usize`, evicting the entire session on every turn.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn tier2_threshold_tokens(&self) -> usize {
        let pct = if self.tier2_usage_threshold.is_finite() {
            self.tier2_usage_threshold.clamp(0.0, 1.0)
        } else {
            DEFAULT_TIER2_USAGE_THRESHOLD
        };
        let window = self.context_window_tokens as f64;
        (window * pct).floor() as usize
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
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parent_id: Option<String>,
    /// Resolved save path, captured at construction so parallel tests that
    /// mutate `ZIPCODE_SESSIONS_DIR` don't cause I/O to land in the wrong dir.
    #[serde(skip)]
    resolved_path: PathBuf,
}

impl Session {
    #[must_use]
    pub fn new() -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let now = timestamp_now();
        let resolved_path = session_path(&id);
        Self {
            id,
            messages: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
            parent_id: None,
            resolved_path,
        }
    }

    /// Create a new child session that records its parent's session ID.
    #[must_use]
    pub fn new_child(parent_id: String) -> Self {
        let mut child = Self::new();
        child.parent_id = Some(parent_id);
        child
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
        let mut session: Self = serde_json::from_str(&content)?;
        if session.id != id {
            anyhow::bail!(
                "Session id mismatch: requested {id}, but stored session id is {}",
                session.id
            );
        }
        if let Some(ref parent_id) = session.parent_id {
            validate_session_id(parent_id)
                .map_err(|_| anyhow::anyhow!("Invalid parent_id in session {id}: {parent_id}"))?;
        }
        session.resolved_path = path;
        Ok(session)
    }

    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.resolved_path.clone()
    }

    /// Return the filesystem path for a session by ID, without loading it.
    ///
    /// # Errors
    ///
    /// Returns an error if the ID fails path-traversal validation.
    pub fn path_for_id(id: &str) -> anyhow::Result<PathBuf> {
        validate_session_id(id)?;
        Ok(session_path(id))
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

    /// Estimate the number of tokens in the current conversation using a
    /// chars-per-4 heuristic.  Only agent-visible messages are counted.
    #[must_use]
    pub fn estimated_tokens(&self) -> usize {
        self.messages
            .iter()
            .filter(|m| !m.agent_invisible)
            .map(|m| m.content.chars().count() / 4)
            .sum()
    }

    /// Return the number of visible `Role::Tool` messages (tool results).
    #[must_use]
    pub fn tool_pair_count(&self) -> usize {
        self.messages
            .iter()
            .filter(|m| m.role == Role::Tool && !m.agent_invisible)
            .count()
    }

    /// **Tier 1** — Summarise the oldest `batch_size` (assistant-with-tool-calls,
    /// tool-result) pairs into a single assistant message prefixed with
    /// `TOOL_PAIR_SUMMARY_MARKER`, and mark the originals `agent_invisible`.
    ///
    /// Returns the number of pairs compacted, or 0 if there are fewer than
    /// `batch_size` visible pairs (no-op in that case).
    pub fn compact_tool_pairs(&mut self, batch_size: usize, max_line_chars: usize) -> usize {
        // batch_size == 0 is a misconfiguration — never useful. Return 0 rather
        // than panicking on the empty `batch[0]` indexing that would follow.
        if batch_size == 0 {
            return 0;
        }

        let pairs = collect_tool_pairs(&self.messages);
        if pairs.len() < batch_size {
            return 0;
        }

        let batch = &pairs[..batch_size];
        let summary_text = build_tool_pair_summary(batch, &self.messages, max_line_chars);
        let insert_before = batch[0].0;

        // Atomicity: insert the summary FIRST, then mark originals invisible.
        // If a panic occurs after `insert` but before the marking loop runs,
        // the session has a spurious summary but still sees the originals — a
        // recoverable state. The reverse order would leave originals hidden
        // with no summary, silently losing information.
        let summary_msg = ChatMessage {
            role: Role::Model,
            content: summary_text,
            tool_call_id: None,
            tool_calls: None,
            agent_invisible: false,
        };
        self.messages.insert(insert_before, summary_msg);

        // All batch indices were >= insert_before; inserting shifted them by 1.
        for &(asst_idx, tool_idx) in batch {
            self.messages[asst_idx + 1].agent_invisible = true;
            self.messages[tool_idx + 1].agent_invisible = true;
        }

        self.updated_at = timestamp_now();
        batch_size
    }

    /// **Tier 2** — Progressively evict tool-result messages when the estimated
    /// token count exceeds `threshold_tokens`.
    ///
    /// Eviction proceeds in steps (10 % → 20 % → 50 % → 100 % of the
    /// original visible tool-result count), stopping as soon as
    /// `estimated_tokens()` drops below `threshold_tokens`.
    ///
    /// Indices collected before the first eviction step are reused across all
    /// steps; messages are never removed, only flagged `agent_invisible = true`,
    /// so indices remain stable.
    ///
    /// Returns `true` if any eviction occurred, `false` if already under
    /// threshold.
    pub fn evict_tool_responses_progressive(&mut self, threshold_tokens: usize) -> bool {
        if self.estimated_tokens() <= threshold_tokens {
            return false;
        }

        // Collect indices once; reused across all eviction steps.
        let tool_indices: Vec<usize> = self
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == Role::Tool && !m.agent_invisible)
            .map(|(i, _)| i)
            .collect();

        let total = tool_indices.len();
        if total == 0 {
            return false;
        }

        for pct in [10usize, 20, 50, 100] {
            // Ceiling division: evict at least 1 message per step.
            let cutoff = (total * pct).div_ceil(100);
            for &idx in &tool_indices[..cutoff] {
                self.messages[idx].agent_invisible = true;
            }
            if self.estimated_tokens() <= threshold_tokens {
                break;
            }
        }

        self.updated_at = timestamp_now();
        true
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

/// Collect (`assistant_index`, `tool_result_index`) pairs from `messages`.
///
/// A pair is defined as an agent-visible assistant message that carries at
/// least one tool call immediately followed (no gap) by an agent-visible
/// `Role::Tool` message.
fn collect_tool_pairs(messages: &[ChatMessage]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut i = 0;
    while i + 1 < messages.len() {
        let asst = &messages[i];
        let tool = &messages[i + 1];
        if asst.role == Role::Model
            && asst.tool_calls.is_some()
            && !asst.agent_invisible
            && tool.role == Role::Tool
            && !tool.agent_invisible
        {
            pairs.push((i, i + 1));
            i += 2;
        } else {
            i += 1;
        }
    }
    pairs
}

/// Build a multi-line summary string for the given tool-call pairs.
///
/// Each line has the form `tool_name(args_preview) → result_preview`.
fn build_tool_pair_summary(
    pairs: &[(usize, usize)],
    messages: &[ChatMessage],
    max_line_chars: usize,
) -> String {
    let mut lines = Vec::with_capacity(pairs.len());
    for &(asst_idx, tool_idx) in pairs {
        let asst = &messages[asst_idx];
        let tool = &messages[tool_idx];

        let tool_name = asst
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .map_or("unknown", |c| c.name.as_str());

        let args_preview = asst
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .map(|c| truncate_inline(&c.arguments.to_string(), max_line_chars / 4))
            .unwrap_or_default();

        let result_preview = truncate_inline(&tool.content, max_line_chars / 2);

        let line = format!("{tool_name}({args_preview}) → {result_preview}");
        lines.push(truncate_inline(&line, max_line_chars));
    }

    format!("{TOOL_PAIR_SUMMARY_MARKER}\n{}", lines.join("\n"))
}

fn session_path(id: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("ZIPCODE_SESSIONS_DIR") {
        return PathBuf::from(dir).join(format!("{id}.json"));
    }
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
        let (_dir, _guard) = crate::test_support::with_test_session_dir();
        let mut session = Session::new();
        session.push_message(ChatMessage::user("test"));

        // Save and reload
        session.save().unwrap();
        let loaded = Session::load(&session.id).unwrap();
        assert_eq!(loaded.messages.len(), 1);
    }

    #[test]
    fn test_load_rejects_mismatched_session_id() {
        let (_dir, _guard) = crate::test_support::with_test_session_dir();
        let session = Session::new();
        let path = session.path();
        session.save().unwrap();

        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        json["id"] = serde_json::json!("different-session-id");
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let error = Session::load(&session.id).unwrap_err().to_string();
        assert!(error.contains("Session id mismatch"));
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
            parent_id: None,
            resolved_path: std::path::PathBuf::new(),
        };

        let result = session.compact(CompactPolicy {
            retain_user_turns: 2,
            max_summary_bullets: 8,
            max_summary_line_chars: 80,
            ..CompactPolicy::default()
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
            parent_id: None,
            resolved_path: std::path::PathBuf::new(),
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
            parent_id: None,
            resolved_path: std::path::PathBuf::new(),
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
            parent_id: None,
            resolved_path: std::path::PathBuf::new(),
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
            parent_id: None,
            resolved_path: std::path::PathBuf::new(),
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
            parent_id: None,
            resolved_path: std::path::PathBuf::new(),
        };

        let result = session.compact(CompactPolicy {
            retain_user_turns: 2,
            max_summary_bullets: 10,
            max_summary_line_chars: 80,
            ..CompactPolicy::default()
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
            parent_id: None,
            resolved_path: std::path::PathBuf::new(),
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

    // ── agent_invisible / tier primitives tests ──────────────────

    #[test]
    fn test_chat_message_agent_invisible_serde_roundtrip() {
        let mut msg = ChatMessage::user("hello");
        msg.agent_invisible = true;
        let json = serde_json::to_string(&msg).unwrap();
        // agent_invisible=true must appear in JSON
        assert!(
            json.contains("agent_invisible"),
            "agent_invisible should be serialized when true: {json}"
        );
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert!(
            back.agent_invisible,
            "agent_invisible should round-trip as true"
        );
    }

    #[test]
    fn test_chat_message_agent_invisible_default_false_when_deserializing_old_format() {
        // JSON without agent_invisible field (legacy format)
        let json = r#"{"role":"user","content":"hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert!(
            !msg.agent_invisible,
            "missing field should default to false"
        );
        // And agent_invisible=false must NOT appear in serialized output
        let serialized = serde_json::to_string(&ChatMessage::user("hi")).unwrap();
        assert!(
            !serialized.contains("agent_invisible"),
            "agent_invisible=false should be skipped in serialization: {serialized}"
        );
    }

    #[test]
    fn test_compact_policy_new_tier_fields_defaults() {
        let policy = CompactPolicy::default();
        assert_eq!(policy.tier1_pair_cutoff, 10);
        assert_eq!(policy.tier1_batch_size, 10);
        assert!(
            (policy.tier2_usage_threshold - 0.8).abs() < f64::EPSILON,
            "tier2_usage_threshold should default to 0.8"
        );
        assert_eq!(policy.context_window_tokens, 131_072);
    }

    #[test]
    fn test_estimated_tokens_counts_all_messages() {
        let mut session = Session::new();
        // "hello" = 5 chars / 4 = 1; "world foo" = 9 chars / 4 = 2
        session.push_message(ChatMessage::user("hello"));
        session.push_message(ChatMessage::assistant("world foo"));
        let tokens = session.estimated_tokens();
        assert_eq!(
            tokens,
            1 + 2,
            "chars/4 heuristic across all visible messages"
        );
    }

    #[test]
    fn test_estimated_tokens_excludes_agent_invisible() {
        let mut session = Session::new();
        let mut invisible = ChatMessage::user("invisible content here big");
        invisible.agent_invisible = true;
        session.messages.push(invisible);
        session.push_message(ChatMessage::user("hi")); // 2 chars / 4 = 0
        let tokens = session.estimated_tokens();
        // Only "hi" (0 tokens) is visible; the big message is excluded
        assert_eq!(
            tokens, 0,
            "agent_invisible message should not count toward tokens"
        );
    }

    #[test]
    fn test_tool_pair_count_counts_tool_messages() {
        let mut session = Session::new();
        session.push_message(ChatMessage::tool_result("c1", "output1"));
        session.push_message(ChatMessage::tool_result("c2", "output2"));
        session.push_message(ChatMessage::user("question"));
        assert_eq!(session.tool_pair_count(), 2);
    }

    #[test]
    fn test_tool_pair_count_excludes_agent_invisible() {
        let mut session = Session::new();
        session.push_message(ChatMessage::tool_result("c1", "visible"));
        let mut invisible_tool = ChatMessage::tool_result("c2", "invisible");
        invisible_tool.agent_invisible = true;
        session.messages.push(invisible_tool);
        assert_eq!(session.tool_pair_count(), 1);
    }

    #[test]
    fn test_compact_tool_pairs_no_op_below_batch_size() {
        let call = ToolCallParsed {
            id: "c1".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "foo.rs"}),
        };
        let mut session = Session::new();
        session.push_message(ChatMessage::assistant_with_tool_calls(
            "thinking",
            vec![call],
        ));
        session.push_message(ChatMessage::tool_result("c1", "contents"));
        // 1 pair < batch_size=5 → no-op
        let count = session.compact_tool_pairs(5, 120);
        assert_eq!(count, 0, "should be no-op when pairs < batch_size");
        assert!(
            !session.messages[0].agent_invisible,
            "original should remain visible"
        );
    }

    #[test]
    fn test_compact_tool_pairs_creates_summary_and_marks_invisible() {
        let call = ToolCallParsed {
            id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        };
        let mut session = Session::new();
        // Two pairs → batch_size=2 triggers compaction
        let call2 = ToolCallParsed {
            id: "c2".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "main.rs"}),
        };
        session.push_message(ChatMessage::assistant_with_tool_calls("step1", vec![call]));
        session.push_message(ChatMessage::tool_result("c1", "file1.rs\nfile2.rs"));
        session.push_message(ChatMessage::assistant_with_tool_calls("step2", vec![call2]));
        session.push_message(ChatMessage::tool_result("c2", "fn main() {}"));

        let count = session.compact_tool_pairs(2, 120);
        assert_eq!(count, 2, "should report 2 pairs compacted");

        // Find the summary message (the inserted one)
        let summary_msg = session
            .messages
            .iter()
            .find(|m| m.content.starts_with(TOOL_PAIR_SUMMARY_MARKER))
            .expect("summary message should exist");
        assert!(!summary_msg.agent_invisible, "summary must be visible");
        assert!(
            summary_msg.content.contains("bash"),
            "summary should mention tool name"
        );

        // Originals must be invisible
        let invisible_count = session
            .messages
            .iter()
            .filter(|m| m.agent_invisible)
            .count();
        assert_eq!(
            invisible_count, 4,
            "all 4 original messages should be invisible"
        );
    }

    #[test]
    fn test_compact_tool_pairs_preserves_order() {
        let mk_call = |id: &str, name: &str| ToolCallParsed {
            id: id.to_string(),
            name: name.to_string(),
            arguments: serde_json::json!({}),
        };
        let mut session = Session::new();
        session.push_message(ChatMessage::user("start"));
        session.push_message(ChatMessage::assistant_with_tool_calls(
            "t1",
            vec![mk_call("c1", "tool_a")],
        ));
        session.push_message(ChatMessage::tool_result("c1", "result_a"));
        session.push_message(ChatMessage::assistant_with_tool_calls(
            "t2",
            vec![mk_call("c2", "tool_b")],
        ));
        session.push_message(ChatMessage::tool_result("c2", "result_b"));
        session.push_message(ChatMessage::user("end"));

        session.compact_tool_pairs(2, 120);

        // User "start" should still be first, user "end" last
        assert_eq!(session.messages[0].content, "start");
        let last = session.messages.last().unwrap();
        assert_eq!(last.content, "end");

        // Summary message must exist and come before the trailing user message
        let summary_pos = session
            .messages
            .iter()
            .position(|m| m.content.starts_with(TOOL_PAIR_SUMMARY_MARKER))
            .expect("summary must exist");
        let end_pos = session.messages.len() - 1;
        assert!(
            summary_pos < end_pos,
            "summary must precede trailing user turn"
        );
    }

    #[test]
    fn test_evict_tool_responses_no_op_under_threshold() {
        let mut session = Session::new();
        session.push_message(ChatMessage::tool_result("c1", "tiny"));
        // With "tiny" = 4 chars, estimated_tokens = 1
        // threshold = 100 → already under → false
        let evicted = session.evict_tool_responses_progressive(100);
        assert!(!evicted, "should return false when already under threshold");
        assert!(
            !session.messages[0].agent_invisible,
            "no messages should be evicted"
        );
    }

    #[test]
    fn test_evict_tool_responses_progressive_10_percent() {
        let mut session = Session::new();
        // 10 tool-result messages, each exactly 40 chars → 10 tokens each (chars/4).
        // Total: 100 tokens. Threshold: 95. After evicting 10% (1 msg) → 90 < 95 → stop.
        for i in 0..10u8 {
            let content = "x".repeat(40); // 40 chars / 4 = 10 tokens
            session.push_message(ChatMessage::tool_result(&format!("c{i}"), &content));
        }
        let evicted = session.evict_tool_responses_progressive(95);
        assert!(evicted, "should return true when eviction happened");

        let invisible_count = session
            .messages
            .iter()
            .filter(|m| m.agent_invisible)
            .count();
        // After evicting 1 message (10%), 90 tokens < 95 threshold → stops at first step.
        assert_eq!(
            invisible_count, 1,
            "only 10% (1 of 10) should be evicted to satisfy threshold"
        );
    }

    #[test]
    fn test_evict_tool_responses_progressive_escalates_to_100_percent() {
        let mut session = Session::new();
        // 10 tool results, each 400 chars → 100 tokens each → 1000 total
        for i in 0..10 {
            let content = "y".repeat(400);
            session.push_message(ChatMessage::tool_result(&format!("c{i}"), &content));
        }
        // threshold=0 → never satisfied until all are evicted
        let evicted = session.evict_tool_responses_progressive(0);
        assert!(evicted);
        let invisible_count = session
            .messages
            .iter()
            .filter(|m| m.agent_invisible)
            .count();
        assert_eq!(
            invisible_count, 10,
            "all 10 should be evicted when threshold=0"
        );
    }

    #[test]
    fn test_evict_tool_responses_indices_stable_across_steps() {
        // Regression: indices collected before the first eviction step must
        // remain valid in subsequent steps (messages are not removed, only
        // flagged, so indices are inherently stable).
        let mut session = Session::new();
        for i in 0..10 {
            let content = "z".repeat(200); // 50 tokens each; total 500
            session.push_message(ChatMessage::tool_result(&format!("c{i}"), &content));
        }
        // threshold=1 → almost everything must be evicted
        let evicted = session.evict_tool_responses_progressive(1);
        assert!(evicted);

        // All messages must still be present (no deletions)
        assert_eq!(
            session.messages.len(),
            10,
            "no messages should be deleted, only flagged"
        );

        // At threshold=1, at most 1 token remains visible (chars/4 = 0 for short).
        // Verify the invisible ones are the oldest (lowest indices).
        let invisible: Vec<usize> = session
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.agent_invisible)
            .map(|(i, _)| i)
            .collect();
        // Invisible set must be a prefix (oldest-first eviction)
        for (pos, &idx) in invisible.iter().enumerate() {
            assert_eq!(
                idx, pos,
                "eviction must proceed oldest-first (index {idx} at position {pos})"
            );
        }
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

    // ── new_child tests ───────────────────────────────────────────

    #[test]
    fn test_new_child_records_parent_id() {
        let parent = Session::new();
        let child = Session::new_child(parent.id.clone());
        assert_eq!(child.parent_id.as_deref(), Some(parent.id.as_str()));
    }

    #[test]
    fn test_new_session_has_no_parent_id() {
        let session = Session::new();
        assert!(session.parent_id.is_none());
    }

    // ── tier2_threshold_tokens direct tests ──────────────────────

    #[test]
    fn test_tier2_threshold_default() {
        let policy = CompactPolicy::default();
        // 131_072 * 0.8 = 104_857.6 → floor = 104_857
        assert_eq!(policy.tier2_threshold_tokens(), 104_857);
    }

    #[test]
    fn test_tier2_threshold_nan_falls_back() {
        let policy = CompactPolicy {
            tier2_usage_threshold: f64::NAN,
            ..CompactPolicy::default()
        };
        // NaN is not finite → falls back to DEFAULT_TIER2_USAGE_THRESHOLD (0.8)
        assert_eq!(policy.tier2_threshold_tokens(), 104_857);
    }

    #[test]
    fn test_tier2_threshold_infinity_clamped() {
        let policy = CompactPolicy {
            tier2_usage_threshold: f64::INFINITY,
            ..CompactPolicy::default()
        };
        // Infinity is not finite → falls back to 0.8
        assert_eq!(policy.tier2_threshold_tokens(), 104_857);
    }

    #[test]
    fn test_tier2_threshold_negative_clamped() {
        let policy = CompactPolicy {
            tier2_usage_threshold: -1.0,
            ..CompactPolicy::default()
        };
        // Negative clamped to 0 → 131_072 * 0 = 0
        assert_eq!(policy.tier2_threshold_tokens(), 0);
    }

    #[test]
    fn test_tier2_threshold_above_one_clamped() {
        let policy = CompactPolicy {
            tier2_usage_threshold: 5.0,
            ..CompactPolicy::default()
        };
        // 5.0 clamped to 1.0 → 131_072 * 1.0 = 131_072
        assert_eq!(policy.tier2_threshold_tokens(), 131_072);
    }

    // ── collect_tool_pairs direct tests ──────────────────────────

    #[test]
    fn test_collect_tool_pairs_basic() {
        let call = ToolCallParsed {
            id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({}),
        };
        let messages = vec![
            ChatMessage::assistant_with_tool_calls("thinking", vec![call]),
            ChatMessage::tool_result("c1", "output"),
        ];
        let pairs = collect_tool_pairs(&messages);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn test_collect_tool_pairs_non_adjacent() {
        let call = ToolCallParsed {
            id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({}),
        };
        // Gap: user message between assistant+tool_calls and tool result
        let messages = vec![
            ChatMessage::assistant_with_tool_calls("thinking", vec![call]),
            ChatMessage::user("interrupt"),
            ChatMessage::tool_result("c1", "output"),
        ];
        let pairs = collect_tool_pairs(&messages);
        assert!(pairs.is_empty(), "non-adjacent pair should not match");
    }

    #[test]
    fn test_collect_tool_pairs_agent_invisible_asst() {
        let call = ToolCallParsed {
            id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({}),
        };
        let mut asst = ChatMessage::assistant_with_tool_calls("thinking", vec![call]);
        asst.agent_invisible = true;
        let tool = ChatMessage::tool_result("c1", "output");
        let messages = vec![asst, tool];
        let pairs = collect_tool_pairs(&messages);
        assert!(pairs.is_empty(), "invisible assistant should be skipped");
    }

    #[test]
    fn test_collect_tool_pairs_agent_invisible_tool() {
        let call = ToolCallParsed {
            id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({}),
        };
        let asst = ChatMessage::assistant_with_tool_calls("thinking", vec![call]);
        let mut tool = ChatMessage::tool_result("c1", "output");
        tool.agent_invisible = true;
        let messages = vec![asst, tool];
        let pairs = collect_tool_pairs(&messages);
        assert!(pairs.is_empty(), "invisible tool result should be skipped");
    }

    #[test]
    fn test_collect_tool_pairs_empty() {
        let messages: Vec<ChatMessage> = vec![];
        let pairs = collect_tool_pairs(&messages);
        assert!(pairs.is_empty());
    }

    // ── build_tool_pair_summary direct tests ─────────────────────

    #[test]
    fn test_build_tool_pair_summary_basic() {
        let call = ToolCallParsed {
            id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        };
        let messages = vec![
            ChatMessage::assistant_with_tool_calls("", vec![call]),
            ChatMessage::tool_result("c1", "file1.rs\nfile2.rs"),
        ];
        let summary = build_tool_pair_summary(&[(0, 1)], &messages, 120);
        assert!(summary.starts_with(TOOL_PAIR_SUMMARY_MARKER));
        assert!(summary.contains("bash("));
        assert!(summary.contains("file1.rs"));
    }

    #[test]
    fn test_build_tool_pair_summary_truncation() {
        let call = ToolCallParsed {
            id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        };
        let long_result = "x".repeat(500);
        let messages = vec![
            ChatMessage::assistant_with_tool_calls("", vec![call]),
            ChatMessage::tool_result("c1", &long_result),
        ];
        let max_chars = 80;
        let summary = build_tool_pair_summary(&[(0, 1)], &messages, max_chars);
        // The summary line should be truncated to max_chars + 1 (for ellipsis)
        for line in summary.lines().skip(1) {
            assert!(
                line.chars().count() <= max_chars + 1,
                "line should be truncated: got {} chars",
                line.chars().count()
            );
        }
    }

    #[test]
    fn test_build_tool_pair_summary_unknown_tool() {
        // Assistant with tool_calls = None → tool name should be "unknown"
        let mut asst = ChatMessage::assistant("thinking");
        asst.tool_calls = None;
        let tool = ChatMessage::tool_result("c1", "some result");
        // Force role to be Model for the assistant to match collect criteria
        // Actually build_tool_pair_summary doesn't check roles, just reads tool_calls
        let messages = vec![asst, tool];
        let summary = build_tool_pair_summary(&[(0, 1)], &messages, 120);
        assert!(
            summary.contains("unknown("),
            "no tool_calls should yield 'unknown'"
        );
    }

    // ── session_path direct tests ────────────────────────────────
    #[test]
    fn test_session_path_default_format() {
        // Acquire the shared SESSION_DIR_LOCK so removing the env var here
        // cannot race with sibling tests that set it.
        let _guard = crate::test_support::SESSION_DIR_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("ZIPCODE_SESSIONS_DIR");
        let path = session_path("my-session-id");
        // Default path should end with .zipcode/sessions/my-session-id.json
        assert!(
            path.to_string_lossy()
                .contains(".zipcode/sessions/my-session-id.json"),
            "default session path should contain .zipcode/sessions/{{id}}.json, got: {}",
            path.display()
        );
    }
}
