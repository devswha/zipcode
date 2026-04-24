use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tracing::{info, warn};

use zipcode_inference::chat_template::ToolSpec;
use zipcode_inference::{
    extract_text_content, ChatMessage, FinishReason, InferenceProvider, Role, TokenEvent,
    ToolCallParsed,
};
use zipcode_tools::{execute_tool, ChildResult, PermissionMode, ToolContext, ToolRegistry};

use crate::permission::{PermissionCheck, PermissionPolicy};
use crate::session::Session;

/// Maximum sub-agent nesting depth. Depth 0 = top-level user session.
pub const MAX_AGENT_DEPTH: u32 = 2;

/// Compute the token budget for a child agent.
/// Child gets at most half the parent's remaining budget, floored at 4096 and
/// capped at 32768.
#[must_use]
pub fn compute_child_budget(parent_remaining: usize) -> usize {
    (parent_remaining / 2).clamp(4096, 32768)
}

/// Callback for streaming tokens and events to the UI layer.
pub trait StreamCallback: Send {
    fn on_token(&mut self, text: &str);
    /// A chunk of the model's private reasoning channel (Gemma 4's thinking
    /// mode). UIs may render this dimmed/collapsed to keep the main token
    /// lane clean. Defaults to dropping the text so existing implementers
    /// remain correct without code changes — the reasoning is already
    /// excluded from stored assistant history by the run loop.
    fn on_thinking(&mut self, _text: &str) {}
    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value);
    fn on_tool_result(&mut self, name: &str, result: &str);
    /// Returns true if the user approves the action.
    fn on_permission_prompt(&mut self, message: &str) -> bool;
    fn on_error(&mut self, error: &str);
}

/// A no-op StreamCallback used by child agents to discard UI events.
struct DevNullCallback;

impl StreamCallback for DevNullCallback {
    fn on_token(&mut self, _text: &str) {}
    fn on_tool_start(&mut self, _name: &str, _args: &serde_json::Value) {}
    fn on_tool_result(&mut self, _name: &str, _result: &str) {}
    fn on_permission_prompt(&mut self, _message: &str) -> bool {
        true
    }
    fn on_error(&mut self, _error: &str) {}
}

pub struct ConversationLoop {
    pub engine: Box<dyn InferenceProvider>,
    pub tools: ToolRegistry,
    pub session: Session,
    pub permission: PermissionPolicy,
    pub system_prompt: String,
    pub tool_specs: Vec<ToolSpec>,
    pub cwd: std::path::PathBuf,
    /// Nesting depth: 0 = top-level user session, 1 = sub-agent, etc.
    pub depth: u32,
    /// Index into `session.messages` of the first message not yet sent to the
    /// provider. Only meaningful when `engine.manages_own_context()` is true;
    /// advances after each `generate_stream` call so subsequent calls send
    /// only the new slice rather than the full accumulated history.
    pub last_sent_idx: usize,
    /// Child agents spawned by this loop: (session_id, saved_path) for orphan cleanup.
    /// Path captured at spawn time so Drop doesn't re-derive from env var.
    pub child_session_ids: Arc<Mutex<Vec<(String, std::path::PathBuf)>>>,
}

impl ConversationLoop {
    /// Run a single conversation turn: generate a response, execute any tool
    /// calls, and repeat until the model finishes or the iteration cap is hit.
    ///
    /// # Errors
    ///
    /// Returns an error if the model exceeds the maximum tool iterations, an
    /// inference error occurs during generation, or the session cannot be
    /// persisted.
    pub fn run_turn(&mut self, user_input: &str, callback: &mut dyn StreamCallback) -> Result<()> {
        const MAX_TOOL_ITERATIONS: usize = 25;
        const MAX_EMPTY_RETRIES: usize = 2;

        // Add system prompt on the very first turn
        if self.session.messages.is_empty() {
            self.session
                .push_message(ChatMessage::system(&self.system_prompt));
        }

        self.session.push_message(ChatMessage::user(user_input));

        let mut iterations = 0;
        let mut empty_retries = 0usize;
        loop {
            if iterations >= MAX_TOOL_ITERATIONS {
                let message = format!(
                    "Stopped after {MAX_TOOL_ITERATIONS} tool iterations to avoid an infinite loop"
                );
                callback.on_error(&message);
                self.session.push_message(ChatMessage::assistant(&message));
                self.session.save()?;
                return Err(anyhow::anyhow!(message));
            }
            iterations += 1;

            let msgs_for_provider: &[ChatMessage] = if self.engine.manages_own_context() {
                &self.session.messages[self.last_sent_idx..]
            } else {
                &self.session.messages
            };
            let rx = self
                .engine
                .generate_stream(msgs_for_provider, &self.tool_specs);
            self.last_sent_idx = self.session.messages.len();

            let (mut full_text, tool_calls, finish_reason) =
                self.collect_generation(rx, callback)?;

            if finish_reason == FinishReason::MaxTokens {
                warn!("model output truncated due to token limit");
                full_text.push_str("\n[...output truncated due to token limit]");
            }

            if tool_calls.is_empty() {
                self.session
                    .push_message(ChatMessage::assistant(&full_text));
            } else {
                let assistant_text = extract_text_content(&full_text);
                self.session
                    .push_message(ChatMessage::assistant_with_tool_calls(
                        &assistant_text,
                        tool_calls.clone(),
                    ));
            }

            // No tool calls → turn is complete, unless auto-retry applies.
            if tool_calls.is_empty() {
                if full_text.trim().is_empty()
                    && empty_retries < MAX_EMPTY_RETRIES
                    && recent_tool_results_contain_errors(&self.session.messages)
                {
                    empty_retries += 1;
                    if let Some(last) = self.session.messages.last_mut() {
                        if last.role == Role::Model && last.content.is_empty() {
                            last.content =
                                "Let me re-read the error output and try a different approach."
                                    .to_string();
                        }
                    }
                    info!(
                        retry = empty_retries,
                        "auto-retry: empty model turn after error"
                    );
                    continue;
                }
                break;
            }

            self.execute_tool_calls(&tool_calls, callback);
        }

        self.session.save()?;
        Ok(())
    }

    fn collect_generation(
        &self,
        rx: Receiver<TokenEvent>,
        callback: &mut dyn StreamCallback,
    ) -> Result<(String, Vec<ToolCallParsed>, FinishReason)> {
        let mut full_text = String::new();
        let mut tool_calls = Vec::new();
        let mut finish_reason = FinishReason::Stop;

        for event in rx {
            match event {
                TokenEvent::Token(text) => {
                    callback.on_token(&text);
                    full_text.push_str(&text);
                }
                TokenEvent::Thinking(text) => {
                    callback.on_thinking(&text);
                }
                TokenEvent::ToolCall(call) => {
                    tool_calls.push(call);
                }
                TokenEvent::Done(reason) => {
                    finish_reason = reason;
                    break;
                }
                TokenEvent::Error(e) => {
                    callback.on_error(&e.to_string());
                    if let Err(save_err) = self.session.save() {
                        warn!(
                            error = %save_err,
                            "failed to save session after inference error (inference error takes priority)"
                        );
                    }
                    return Err(anyhow::anyhow!("Inference error: {e}"));
                }
            }
        }

        Ok((full_text, tool_calls, finish_reason))
    }

    fn execute_tool_calls(
        &mut self,
        tool_calls: &[ToolCallParsed],
        callback: &mut dyn StreamCallback,
    ) {
        for call in tool_calls {
            let check = self.permission.check(&call.name, &call.arguments);

            match check {
                PermissionCheck::Allowed => {}
                PermissionCheck::NeedsApproval(msg) => {
                    if !callback.on_permission_prompt(&msg) {
                        self.session.push_message(ChatMessage::tool_result(
                            &call.id,
                            "User denied permission for this action.",
                        ));
                        continue;
                    }
                }
                PermissionCheck::Denied(msg) => {
                    self.session
                        .push_message(ChatMessage::tool_result(&call.id, &msg));
                    continue;
                }
            }

            callback.on_tool_start(&call.name, &call.arguments);
            info!(tool = %call.name, "executing tool");

            let ctx = ToolContext {
                cwd: self.cwd.clone(),
                permission: self.permission.mode(),
                session_id: self.session.id.clone(),
                parent_session_id: self.session.parent_id.clone(),
                depth: self.depth,
                budget_tokens: None,
                spawn_child: self.make_spawn_child_callback(),
            };

            let result = execute_tool(&self.tools, &call.name, call.arguments.clone(), &ctx);
            let result_text = match &result {
                Ok(r) => r.content.clone(),
                Err(e) => format!("Tool error: {e}"),
            };

            callback.on_tool_result(&call.name, &result_text);
            self.session
                .push_message(ChatMessage::tool_result(&call.id, &result_text));
        }
    }

    /// Build the `Arc<SpawnChildFn>` callback that the Agent tool uses to
    /// delegate work back into the runtime.
    ///
    /// Captures enough state to build a real child `ConversationLoop` and run
    /// a full turn without needing `&mut self` inside the closure.
    fn make_spawn_child_callback(&self) -> Option<Arc<zipcode_tools::SpawnChildFn>> {
        let child_engine = self.engine.clone_for_child()?;

        let session_id = self.session.id.clone();
        let current_depth: u32 = self.depth;
        let permission_policy = self.permission.clone();
        let child_ids = Arc::clone(&self.child_session_ids);
        let all_tool_names: Vec<String> =
            self.tools.names().into_iter().map(String::from).collect();
        let tool_registry_snapshot = self.tools.create_filtered(&all_tool_names);
        let system_prompt = self.system_prompt.clone();
        let cwd = self.cwd.clone();

        // Wrap the engine in Arc<Mutex> so the closure (which is Fn, not FnMut)
        // can take ownership each call. In practice the closure is called once.
        let engine_cell = Arc::new(std::sync::Mutex::new(Some(child_engine)));

        Some(Arc::new(
            move |task_prompt, allowlist, permission_override, max_tokens| {
                if current_depth >= MAX_AGENT_DEPTH {
                    anyhow::bail!(
                        "spawn_child refused: already at maximum agent depth ({MAX_AGENT_DEPTH})"
                    );
                }

                let engine = engine_cell
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .take()
                    .ok_or_else(|| {
                        anyhow::anyhow!("spawn callback already consumed (called more than once)")
                    })?;

                let child_permission = permission_policy.inherit_for_child(permission_override);
                let child_registry = match allowlist {
                    Some(names) => tool_registry_snapshot.create_filtered(names),
                    None => tool_registry_snapshot.create_filtered(&all_tool_names),
                };
                let child_tool_specs = convert_tool_specs(child_registry.specs());
                let child_session = crate::session::Session::new_child(session_id.clone());
                let child_session_id = child_session.id.clone();
                let child_session_path = child_session.path();

                child_ids
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push((child_session_id.clone(), child_session_path));

                let mut child_loop = ConversationLoop {
                    engine,
                    tools: child_registry,
                    session: child_session,
                    permission: child_permission,
                    system_prompt: system_prompt.clone(),
                    tool_specs: child_tool_specs,
                    cwd: cwd.clone(),
                    depth: current_depth + 1,
                    last_sent_idx: 0,
                    child_session_ids: Arc::new(Mutex::new(Vec::new())),
                };

                let mut sink = DevNullCallback;
                child_loop.run_turn(task_prompt, &mut sink)?;

                let tool_call_count = child_loop
                    .session
                    .messages
                    .iter()
                    .filter(|m| m.role == zipcode_inference::Role::Tool)
                    .count();

                let summary = child_loop
                    .session
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == zipcode_inference::Role::Model && !m.content.is_empty())
                    .map(|m| m.content.clone())
                    .unwrap_or_else(|| "(no response)".to_string());

                let _ = max_tokens;

                Ok(ChildResult {
                    summary,
                    tool_call_count,
                    child_session_id,
                })
            },
        ))
    }

    /// Spawn a child conversation loop for a sub-agent task.
    ///
    /// # Errors
    ///
    /// Returns an error if the current depth is already at `MAX_AGENT_DEPTH`,
    /// if the backend cannot be cloned for child use, or if session I/O fails.
    pub fn spawn_child(
        &mut self,
        task_prompt: &str,
        allowlist: Option<&[String]>,
        permission_override: Option<PermissionMode>,
        max_tokens: Option<usize>,
    ) -> Result<ChildResult> {
        if self.depth >= MAX_AGENT_DEPTH {
            anyhow::bail!(
                "spawn_child refused: already at maximum agent depth ({MAX_AGENT_DEPTH})"
            );
        }

        let child_engine = self
            .engine
            .clone_for_child()
            .ok_or_else(|| anyhow::anyhow!("inference backend does not support child agents"))?;

        let child_permission = self.permission.inherit_for_child(permission_override);
        let child_registry = match allowlist {
            Some(names) => self.tools.create_filtered(names),
            None => {
                let all: Vec<String> = self.tools.names().into_iter().map(String::from).collect();
                self.tools.create_filtered(&all)
            }
        };
        let child_tool_specs = convert_tool_specs(child_registry.specs());
        let child_session = Session::new_child(self.session.id.clone());
        let child_session_id = child_session.id.clone();
        let child_session_path = child_session.path();

        self.child_session_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((child_session_id.clone(), child_session_path));

        let mut child_loop = ConversationLoop {
            engine: child_engine,
            tools: child_registry,
            session: child_session,
            permission: child_permission,
            system_prompt: self.system_prompt.clone(),
            tool_specs: child_tool_specs,
            cwd: self.cwd.clone(),
            depth: self.depth + 1,
            last_sent_idx: 0,
            child_session_ids: Arc::new(Mutex::new(Vec::new())),
        };

        let mut sink = DevNullCallback;
        child_loop.run_turn(task_prompt, &mut sink)?;

        let tool_call_count = child_loop
            .session
            .messages
            .iter()
            .filter(|m| m.role == zipcode_inference::Role::Tool)
            .count();

        let summary = child_loop
            .session
            .messages
            .iter()
            .rev()
            .find(|m| m.role == zipcode_inference::Role::Model && !m.content.is_empty())
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "(no response)".to_string());

        let _ = max_tokens;

        Ok(ChildResult {
            summary,
            tool_call_count,
            child_session_id,
        })
    }
}

impl Drop for ConversationLoop {
    fn drop(&mut self) {
        let entries = self
            .child_session_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (id, path) in entries.iter() {
            if let Err(e) = std::fs::remove_file(path) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(session_id = %id, error = %e, "failed to remove orphan child session");
                }
            }
        }
    }
}

fn convert_tool_specs(specs: Vec<zipcode_tools::ToolSpec>) -> Vec<ToolSpec> {
    specs
        .into_iter()
        .map(|s| ToolSpec {
            name: s.name,
            description: s.description,
            parameters: s.parameters,
        })
        .collect()
}

/// Walk backwards through recent messages looking for tool results that
/// contain error indicators. Stops at the first user or system message
/// so we only consider tool output from the current agentic cycle.
/// Returns true if `text` looks like it contains a *genuine* error indicator
/// rather than a benign mention (e.g. "no errors found", "0 errors").
fn content_contains_error(text: &str) -> bool {
    // "FAILED" and "panicked" are strong signals — keep them as-is.
    if text.contains("FAILED") || text.contains("panicked") {
        return true;
    }

    // Check "error" / "Error" word by word, skipping benign contexts.
    for line in text.lines() {
        let line_lower = line.to_ascii_lowercase();
        let line_trimmed = line.trim();

        // "error:" at the start of a line is a strong signal (common in
        // compilers and test runners).  But "error: 0" or similar zero-count
        // patterns are benign.
        if line_trimmed.starts_with("error:") || line_trimmed.starts_with("Error:") {
            // Check for zero-count after the prefix — e.g. "error: 0 issues"
            let after = &line_trimmed[6..]; // skip "error:" / "Error:"
            if after.trim().starts_with('0') {
                continue; // benign: "error: 0 ..."
            }
            return true;
        }

        // Scan for the word "error" / "errors" anywhere in the line.
        // We look for word boundaries to avoid matching inside other words.
        if contains_error_word(&line_lower) && !is_benign_error_line(&line_lower) {
            return true;
        }
    }
    false
}

/// Returns true if the lowered line contains the word "error" or "errors"
/// as a standalone token (word boundary check).
fn contains_error_word(line_lower: &str) -> bool {
    // Simple word-boundary scan: look for "error" preceded/followed by a
    // non-alphanumeric character (or at string boundaries).
    let bytes = line_lower.as_bytes();
    let pattern = b"error";
    let pat_len = pattern.len();

    for i in 0..=bytes.len().saturating_sub(pat_len) {
        if &bytes[i..i + pat_len] == pattern {
            // Check preceding character
            let prev_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            // Check following character — allow 's' (errors) but nothing else
            let after_end = i + pat_len;
            let next_ok = if after_end >= bytes.len() {
                true
            } else {
                let next = bytes[after_end];
                next == b's' // "errors"
                    || !next.is_ascii_alphanumeric()
            };
            if prev_ok && next_ok {
                return true;
            }
        }
    }
    false
}

/// Returns true if the lowered line contains the word "fix" as a standalone
/// token (word boundary check). Prevents false benign matches on "prefix",
/// "suffix", "affix", "fixture", etc.
fn contains_fix_word(line_lower: &str) -> bool {
    let bytes = line_lower.as_bytes();
    let pattern = b"fix";
    let pat_len = pattern.len();

    for i in 0..=bytes.len().saturating_sub(pat_len) {
        if &bytes[i..i + pat_len] == pattern {
            let prev_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let after_end = i + pat_len;
            let next_ok = if after_end >= bytes.len() {
                true
            } else {
                !bytes[after_end].is_ascii_alphanumeric()
            };
            if prev_ok && next_ok {
                return true;
            }
        }
    }
    false
}

/// Benign patterns that mention "error" but do NOT indicate a failure.
fn is_benign_error_line(line_lower: &str) -> bool {
    // Order matters: check more specific patterns first.

    // "no error", "no errors"
    if line_lower.contains("no error") || line_lower.contains("no errors") {
        return true;
    }
    // "0 error", "0 errors"
    if line_lower.contains("0 error") || line_lower.contains("0 errors") {
        return true;
    }
    // "without error", "without errors"
    if line_lower.contains("without error") || line_lower.contains("without errors") {
        return true;
    }
    // "fixed ... error", "fix ... error" (resolved errors)
    if line_lower.contains("fixed") && line_lower.contains("error") {
        return true;
    }
    // Use word-boundary matching for "fix" to avoid false matches with
    // "prefix", "suffix", "affix", "fixture", etc. where "fix" is a
    // substring of an unrelated word rather than the verb "to fix".
    if contains_fix_word(line_lower) && line_lower.contains("error") {
        return true;
    }
    // "error handling" (discussing error handling, not reporting one)
    if line_lower.contains("error handling") {
        return true;
    }
    // "error recovery"
    if line_lower.contains("error recovery") {
        return true;
    }
    // "resolved ... error", "error ... resolved"
    if line_lower.contains("resolved") && line_lower.contains("error") {
        return true;
    }
    // "cleared ... error", "error ... cleared"
    if line_lower.contains("cleared") && line_lower.contains("error") {
        return true;
    }
    // "successfully ... error" (e.g. "successfully fixed the error")
    if line_lower.contains("successfully") && line_lower.contains("error") {
        return true;
    }

    false
}

fn recent_tool_results_contain_errors(messages: &[ChatMessage]) -> bool {
    for msg in messages.iter().rev() {
        match msg.role {
            Role::Model => {}
            Role::Tool => {
                if content_contains_error(&msg.content) {
                    return true;
                }
            }
            _ => break,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_true_when_recent_tool_result_contains_error() {
        let messages = vec![
            ChatMessage::user("do something"),
            ChatMessage::tool_result("c1", "compilation error in main.rs"),
        ];
        assert!(recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn returns_true_when_recent_tool_result_contains_uppercase_error() {
        let messages = vec![
            ChatMessage::user("do something"),
            ChatMessage::tool_result("c1", "Error: file not found"),
        ];
        assert!(recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn returns_true_when_recent_tool_result_contains_failed_keyword() {
        let messages = vec![
            ChatMessage::user("run tests"),
            ChatMessage::tool_result("c1", "FAILED test_foo_bar"),
        ];
        assert!(recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn returns_true_when_recent_tool_result_contains_panicked() {
        let messages = vec![
            ChatMessage::user("run tests"),
            ChatMessage::tool_result("c1", "thread 'main' panicked at 'assertion failed'"),
        ];
        assert!(recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn returns_false_when_no_tool_results() {
        let messages = vec![
            ChatMessage::system("you are a helper"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi there"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn returns_false_when_tool_results_have_no_errors() {
        let messages = vec![
            ChatMessage::user("read file"),
            ChatMessage::tool_result("c1", "file contents: all good here"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn stops_scanning_at_user_message() {
        // Tool errors before a user message should be ignored — they belong
        // to a previous agentic cycle.
        let messages = vec![
            ChatMessage::tool_result("c_old", "error from previous cycle"),
            ChatMessage::user("new request"),
            ChatMessage::tool_result("c_new", "all clear"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn skips_model_messages_while_scanning() {
        // Model reasoning messages between tool results are transparent to
        // the scan — only tool-role messages are checked for error keywords.
        let messages = vec![
            ChatMessage::user("run tests"),
            ChatMessage::tool_result("c1", "error: build failed"),
            ChatMessage::assistant("I see the build failed, let me fix it."),
            ChatMessage::tool_result("c2", "fixed"),
        ];
        // The most recent tool result is clean, so the function should
        // continue scanning past the model message and find the error in c1.
        assert!(recent_tool_results_contain_errors(&messages));
    }

    // ── False-positive edge cases (should NOT trigger auto-retry) ──────

    #[test]
    fn false_positive_no_errors_found() {
        let messages = vec![
            ChatMessage::user("run tests"),
            ChatMessage::tool_result("c1", "no errors found"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn false_positive_zero_errors_zero_warnings() {
        let messages = vec![
            ChatMessage::user("run linter"),
            ChatMessage::tool_result("c1", "0 errors, 0 warnings"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn false_positive_ran_without_error() {
        let messages = vec![
            ChatMessage::user("run tests"),
            ChatMessage::tool_result("c1", "Ran without error"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn false_positive_error_handling_improved() {
        let messages = vec![
            ChatMessage::user("refactor"),
            ChatMessage::tool_result("c1", "error handling improved"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn false_positive_successfully_fixed_the_error() {
        let messages = vec![
            ChatMessage::user("fix the bug"),
            ChatMessage::tool_result("c1", "Successfully fixed the error in main.rs"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn false_positive_error_prefix_zero_count() {
        let messages = vec![
            ChatMessage::user("run lint"),
            ChatMessage::tool_result("c1", "error: 0 issues detected"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn false_positive_error_recovery_message() {
        let messages = vec![
            ChatMessage::user("deploy"),
            ChatMessage::tool_result("c1", "error recovery completed successfully"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn false_positive_no_errors_plural() {
        let messages = vec![
            ChatMessage::user("run check"),
            ChatMessage::tool_result("c1", "Checking module... no errors, all clear"),
        ];
        assert!(!recent_tool_results_contain_errors(&messages));
    }

    // ── True-positive edge cases (SHOULD still trigger auto-retry) ─────

    #[test]
    fn true_positive_compilation_error() {
        let messages = vec![
            ChatMessage::user("build"),
            ChatMessage::tool_result("c1", "compilation error: expected `;`"),
        ];
        assert!(recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn true_positive_error_prefix_with_nonzero_count() {
        let messages = vec![
            ChatMessage::user("run lint"),
            ChatMessage::tool_result("c1", "error: 3 issues detected"),
        ];
        assert!(recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn true_positive_error_mid_sentence() {
        let messages = vec![
            ChatMessage::user("build"),
            ChatMessage::tool_result("c1", "the build returned an error: linking failed"),
        ];
        assert!(recent_tool_results_contain_errors(&messages));
    }

    #[test]
    fn true_positive_one_error() {
        let messages = vec![
            ChatMessage::user("run tests"),
            ChatMessage::tool_result("c1", "1 error found in test suite"),
        ];
        assert!(recent_tool_results_contain_errors(&messages));
    }

    // ── Regression: "fix" substring in "prefix"/"suffix" must not mask errors ──

    #[test]
    fn prefix_error_is_detected_as_real_error() {
        // "prefix" contains "fix" as substring — the old contains("fix")
        // heuristic would incorrectly treat this as benign.
        let messages = vec![
            ChatMessage::user("compile"),
            ChatMessage::tool_result("c1", "prefix error in compilation"),
        ];
        assert!(
            recent_tool_results_contain_errors(&messages),
            "'prefix error' should be detected as a real error, not benign"
        );
    }

    #[test]
    fn suffix_error_is_detected_as_real_error() {
        let messages = vec![
            ChatMessage::user("compile"),
            ChatMessage::tool_result("c1", "suffix error: unexpected token"),
        ];
        assert!(
            recent_tool_results_contain_errors(&messages),
            "'suffix error' should be detected as a real error, not benign"
        );
    }

    #[test]
    fn affix_error_is_detected_as_real_error() {
        let messages = vec![
            ChatMessage::user("parse"),
            ChatMessage::tool_result("c1", "affix error: invalid format"),
        ];
        assert!(
            recent_tool_results_contain_errors(&messages),
            "'affix error' should be detected as a real error, not benign"
        );
    }

    #[test]
    fn fix_the_error_is_still_benign() {
        // "fix" as a standalone word + "error" → benign (intention to fix)
        let messages = vec![
            ChatMessage::user("refactor"),
            ChatMessage::tool_result("c1", "I will fix the error in main.rs"),
        ];
        assert!(
            !recent_tool_results_contain_errors(&messages),
            "'fix the error' should still be benign"
        );
    }

    #[test]
    fn fixture_error_is_detected_as_real_error() {
        // "fixture" contains "fix" as substring — must not mask real errors.
        let messages = vec![
            ChatMessage::user("run tests"),
            ChatMessage::tool_result("c1", "fixture error: test setup failed"),
        ];
        assert!(
            recent_tool_results_contain_errors(&messages),
            "'fixture error' should be detected as a real error, not benign"
        );
    }

    // ── compute_child_budget tests ────────────────────────────────

    #[test]
    fn test_compute_child_budget_zero_gives_floor() {
        assert_eq!(compute_child_budget(0), 4096);
    }

    #[test]
    fn test_compute_child_budget_small_gives_floor() {
        assert_eq!(compute_child_budget(2000), 4096);
    }

    #[test]
    fn test_compute_child_budget_mid_halved() {
        assert_eq!(compute_child_budget(10000), 5000);
    }

    #[test]
    fn test_compute_child_budget_large_capped() {
        assert_eq!(compute_child_budget(70000), 32768);
    }

    // ── spawn_child tests ─────────────────────────────────────────

    #[test]
    fn test_spawn_child_rejects_at_max_depth() {
        use crate::permission::PermissionPolicy;
        use std::sync::{Arc, Mutex};
        use zipcode_inference::mock::{MockInferenceProvider, MockResponse};
        use zipcode_tools::PermissionMode;
        use zipcode_tools::ToolRegistry;

        let provider = MockInferenceProvider::new(vec![MockResponse::Text("ok".to_string())]);
        let session = crate::session::Session::new();

        // Set depth = MAX_AGENT_DEPTH so spawn_child guard fires immediately.
        let mut conv = ConversationLoop {
            engine: Box::new(provider),
            tools: ToolRegistry::default(),
            session,
            permission: PermissionPolicy::new(PermissionMode::ReadOnly),
            system_prompt: String::new(),
            tool_specs: Vec::new(),
            cwd: std::path::PathBuf::from("/tmp"),
            depth: MAX_AGENT_DEPTH,
            last_sent_idx: 0,
            child_session_ids: Arc::new(Mutex::new(Vec::new())),
        };

        let result = conv.spawn_child("do something", None, None, None);
        assert!(result.is_err(), "depth=MAX_AGENT_DEPTH should be rejected");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("maximum agent depth"),
            "error should mention max depth"
        );
    }

    #[test]
    fn test_child_session_ids_tracked() {
        use crate::permission::PermissionPolicy;
        use std::sync::{Arc, Mutex};
        use zipcode_inference::mock::{MockInferenceProvider, MockResponse};
        use zipcode_tools::PermissionMode;
        use zipcode_tools::ToolRegistry;

        let provider = MockInferenceProvider::new(vec![MockResponse::Text("ok".to_string())]);
        let session = crate::session::Session::new();
        let mut conv = ConversationLoop {
            engine: Box::new(provider),
            tools: ToolRegistry::default(),
            session,
            permission: PermissionPolicy::new(PermissionMode::FullAccess),
            system_prompt: String::new(),
            tool_specs: Vec::new(),
            cwd: std::path::PathBuf::from("/tmp"),
            depth: 0,
            last_sent_idx: 0,
            child_session_ids: Arc::new(Mutex::new(Vec::new())),
        };

        let result = conv.spawn_child("task 1", None, None, None).unwrap();
        let ids = conv.child_session_ids.lock().unwrap();
        assert!(
            ids.iter().any(|(id, _)| id == &result.child_session_id),
            "child session id should be tracked"
        );
    }

    #[test]
    fn test_dropped_loop_removes_child_session_file() {
        use crate::permission::PermissionPolicy;
        use std::sync::{Arc, Mutex};
        use zipcode_inference::mock::{MockInferenceProvider, MockResponse};
        use zipcode_tools::PermissionMode;
        use zipcode_tools::ToolRegistry;

        let provider = MockInferenceProvider::new(vec![MockResponse::Text("ok".to_string())]);
        let session = crate::session::Session::new();
        let child_ids_shared: Arc<Mutex<Vec<(String, std::path::PathBuf)>>> =
            Arc::new(Mutex::new(Vec::new()));

        let child_session_id = {
            let mut conv = ConversationLoop {
                engine: Box::new(provider),
                tools: ToolRegistry::default(),
                session,
                permission: PermissionPolicy::new(PermissionMode::FullAccess),
                system_prompt: String::new(),
                tool_specs: Vec::new(),
                cwd: std::path::PathBuf::from("/tmp"),
                depth: 0,
                last_sent_idx: 0,
                child_session_ids: Arc::clone(&child_ids_shared),
            };

            let result = conv.spawn_child("cleanup test", None, None, None).unwrap();
            let child_path =
                crate::session::Session::path_for_id(&result.child_session_id).unwrap();
            assert!(
                child_path.exists(),
                "child session file should exist before drop"
            );
            result.child_session_id
        };
        // conv is now dropped; Drop impl should have removed the file
        let child_path = crate::session::Session::path_for_id(&child_session_id).unwrap();
        assert!(
            !child_path.exists(),
            "child session file should be removed after parent drop"
        );
    }
}
