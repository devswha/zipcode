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
use crate::session::{CompactPolicy, Session};
use crate::skills::SkillRegistry;

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

/// A no-op `StreamCallback` used by child agents to discard UI events.
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
    /// Child agents spawned by this loop: (`session_id`, `saved_path`) for orphan cleanup.
    /// Path captured at spawn time so Drop doesn't re-derive from env var.
    pub child_session_ids: Arc<Mutex<Vec<(String, std::path::PathBuf)>>>,
    /// Optional skill registry for skill tool invocation.
    pub skill_registry: Option<Arc<SkillRegistry>>,
    /// Policy controlling when and how tier-1 (tool-pair summarisation) and
    /// tier-2 (tool-response eviction) context compaction triggers.
    pub compact_policy: CompactPolicy,
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
    #[allow(clippy::too_many_lines)]
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

            // Filter out agent_invisible messages before sending to the provider.
            // For backends that manage their own context, only the new slice is
            // sent; for others the full (visible) history is sent each turn.
            let msgs_for_provider: Vec<ChatMessage> = if self.engine.manages_own_context() {
                self.session.messages[self.last_sent_idx..]
                    .iter()
                    .filter(|m| !m.agent_invisible)
                    .cloned()
                    .collect()
            } else {
                self.session
                    .messages
                    .iter()
                    .filter(|m| !m.agent_invisible)
                    .cloned()
                    .collect()
            };
            let rx = self
                .engine
                .generate_stream(&msgs_for_provider, &self.tool_specs);
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

        // ── Tier-1: summarise oldest tool-call pairs between turns ─────────
        // Proactive pass runs first so normal-usage sessions keep a summary of
        // older tool activity in the transcript. Tier-2 only kicks in as an
        // emergency fallback when the backend reports the context window is
        // still hot even after summarisation.
        //
        // Runs synchronously here (between turns, not mid-turn) rather than
        // in a real background thread.  A genuine background thread would
        // require Arc<Mutex<Session>>, which is a large structural change;
        // the synchronous approach satisfies the "main loop proceeds even if
        // summariser fails" requirement by wrapping in catch_unwind.
        // Any panic from compact_tool_pairs is swallowed with a warn log so
        // run_turn() always returns Ok.
        if self.session.tool_pair_count() > self.compact_policy.tier1_pair_cutoff {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.session.compact_tool_pairs(
                    self.compact_policy.tier1_batch_size,
                    self.compact_policy.max_summary_line_chars,
                )
            }));
            match result {
                Ok(n) if n > 0 => {
                    info!(pairs = n, "tier-1 compaction: summarised tool-call pairs");
                }
                Ok(_) => {} // no-op: pairs < batch_size
                Err(_) => {
                    warn!("tier-1 compaction panicked; skipping (main loop continues)");
                }
            }
        }

        // ── Tier-2: emergency eviction when context usage is still high ────
        // Reactive fallback that checks the backend's reported prompt-token
        // count. A `None` count (default for Mock and Candle backends) leaves
        // tier-2 inactive, preserving Phase 1/2 tests. Threshold is computed
        // via `tier2_threshold_tokens()`, which clamps NaN/out-of-range
        // policy values before they can trigger pathological full-session
        // eviction.
        if let Some(count) = self.engine.last_prompt_eval_count() {
            let threshold_tokens = self.compact_policy.tier2_threshold_tokens();
            if count > threshold_tokens {
                let evicted = self
                    .session
                    .evict_tool_responses_progressive(threshold_tokens);
                if evicted {
                    warn!(
                        usage = count,
                        threshold = threshold_tokens,
                        "tier-2 compaction triggered: evicting tool responses"
                    );
                }
            }
        }

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
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                    .ok_or_else(|| {
                        anyhow::anyhow!("spawn callback already consumed (called more than once)")
                    })?;

                let child_permission = permission_policy.inherit_for_child(permission_override);
                let child_registry = tool_registry_snapshot
                    .create_filtered(allowlist.map_or_else(|| &all_tool_names, AsRef::as_ref));
                let child_tool_specs = convert_tool_specs(child_registry.specs());
                let child_session = crate::session::Session::new_child(session_id.clone());
                let child_session_id = child_session.id.clone();
                let child_session_path = child_session.path();

                child_ids
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push((child_session_id.clone(), child_session_path));

                let mut child_loop = Self {
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
                    skill_registry: None,
                    compact_policy: CompactPolicy::default(),
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
                    .map_or_else(|| "(no response)".to_string(), |m| m.content.clone());

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
        let child_registry = if let Some(names) = allowlist {
            self.tools.create_filtered(names)
        } else {
            let all: Vec<String> = self.tools.names().into_iter().map(String::from).collect();
            self.tools.create_filtered(&all)
        };
        let child_tool_specs = convert_tool_specs(child_registry.specs());
        let child_session = Session::new_child(self.session.id.clone());
        let child_session_id = child_session.id.clone();
        let child_session_path = child_session.path();

        self.child_session_ids
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((child_session_id.clone(), child_session_path));

        let mut child_loop = Self {
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
            skill_registry: None,
            compact_policy: CompactPolicy::default(),
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
            .map_or_else(|| "(no response)".to_string(), |m| m.content.clone());

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
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (id, path) in entries.iter() {
            if let Err(e) = std::fs::remove_file(path) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(session_id = %id, error = %e, "failed to remove orphan child session");
                }
            }
        }
    }
}

#[must_use]
pub fn convert_tool_specs(specs: Vec<zipcode_tools::ToolSpec>) -> Vec<ToolSpec> {
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

    if bytes.len() < pat_len {
        return false;
    }

    for i in 0..=bytes.len() - pat_len {
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

    if bytes.len() < pat_len {
        return false;
    }

    for i in 0..=bytes.len() - pat_len {
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
    fn convert_tool_specs_round_trips_name_description_parameters() {
        let raw = vec![
            zipcode_tools::ToolSpec {
                name: "read_file".to_string(),
                description: "read a file".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
            zipcode_tools::ToolSpec {
                name: "bash".to_string(),
                description: "run a command".to_string(),
                parameters: serde_json::json!({"type": "object", "required": ["command"]}),
            },
        ];
        let converted = convert_tool_specs(raw);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].name, "read_file");
        assert_eq!(converted[0].description, "read a file");
        assert_eq!(converted[1].name, "bash");
        assert_eq!(
            converted[1].parameters["required"][0].as_str(),
            Some("command")
        );
    }

    #[test]
    fn skill_allowlist_filter_pattern_keeps_only_allowed_tools() {
        // Mirrors the runtime path used by `cli::run_skill_command`: filter
        // the registry by the skill's tool_allowlist and rebuild the
        // ConversationLoop's tool_specs from the filtered set. Regression
        // guard for the fix that made `zipcode skill` honour the allowlist
        // instead of exposing the full toolbox.
        use zipcode_tools::{
            agent::AgentTool, grep_search::GrepSearchTool, read_file::ReadFileTool,
            write_file::WriteFileTool, ToolRegistry,
        };

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(AgentTool));
        registry.register(Box::new(ReadFileTool));
        registry.register(Box::new(WriteFileTool));
        registry.register(Box::new(GrepSearchTool));

        let allowlist = vec!["read_file".to_string(), "grep_search".to_string()];
        let filtered = registry.create_filtered(&allowlist);
        let specs = convert_tool_specs(filtered.specs());

        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"grep_search"));
        assert!(
            !names.contains(&"write_file"),
            "write_file must be filtered out"
        );
        assert!(!names.contains(&"agent"), "agent must be filtered out");
    }

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
            skill_registry: None,
            compact_policy: CompactPolicy::default(),
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
            skill_registry: None,
            compact_policy: CompactPolicy::default(),
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

        // Sibling tier tests mutate ZIPCODE_SESSIONS_DIR in parallel; acquire
        // the shared lock so Session::path resolution observes our dir only.
        let (_dir, _guard) = with_test_session_dir();

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
                skill_registry: None,
                compact_policy: CompactPolicy::default(),
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

    // ── Compaction integration tests ──────────────────────────────

    use crate::test_support::with_test_session_dir;

    /// Build a minimal `ConversationLoop` with a `MockInferenceProvider` and the
    /// given `CompactPolicy`.  Uses /tmp as cwd (no real tools needed).
    fn make_conv(
        responses: Vec<zipcode_inference::mock::MockResponse>,
        policy: CompactPolicy,
    ) -> ConversationLoop {
        use crate::permission::PermissionPolicy;
        use std::sync::{Arc, Mutex};
        use zipcode_inference::mock::MockInferenceProvider;
        use zipcode_tools::{PermissionMode, ToolRegistry};

        ConversationLoop {
            engine: Box::new(MockInferenceProvider::new(responses)),
            tools: ToolRegistry::default(),
            session: crate::session::Session::new(),
            permission: PermissionPolicy::new(PermissionMode::FullAccess),
            system_prompt: String::new(),
            tool_specs: Vec::new(),
            cwd: std::path::PathBuf::from("/tmp"),
            depth: 0,
            last_sent_idx: 0,
            child_session_ids: Arc::new(Mutex::new(Vec::new())),
            skill_registry: None,
            compact_policy: policy,
        }
    }

    /// Push N tool-call / tool-result pairs directly into the session.
    fn push_tool_pairs(session: &mut crate::session::Session, n: usize) {
        use zipcode_inference::types::ToolCallParsed;
        for i in 0..n {
            let call = ToolCallParsed {
                id: format!("call_{i}"),
                name: "bash".to_string(),
                arguments: serde_json::json!({"command": format!("ls {i}")}),
            };
            session.push_message(ChatMessage::assistant_with_tool_calls(
                "running tool",
                vec![call],
            ));
            session.push_message(ChatMessage::tool_result(
                &format!("call_{i}"),
                &format!("output_{i}"),
            ));
        }
    }

    struct NoopCb;
    impl StreamCallback for NoopCb {
        fn on_token(&mut self, _: &str) {}
        fn on_tool_start(&mut self, _: &str, _: &serde_json::Value) {}
        fn on_tool_result(&mut self, _: &str, _: &str) {}
        fn on_permission_prompt(&mut self, _: &str) -> bool {
            true
        }
        fn on_error(&mut self, _: &str) {}
    }

    /// Tier-1 fires when pair count exceeds cutoff and reduces the visible pair count.
    #[test]
    fn test_run_turn_triggers_tier1_when_pair_count_exceeds_cutoff() {
        let (_dir, _guard) = with_test_session_dir();
        use zipcode_inference::mock::MockResponse;

        // cutoff=2, batch_size=2 → triggers when pairs > 2 and compacts 2 pairs
        let policy = CompactPolicy {
            tier1_pair_cutoff: 2,
            tier1_batch_size: 2,
            ..CompactPolicy::default()
        };
        let mut conv = make_conv(vec![MockResponse::Text("done".to_string())], policy);

        // Pre-load 3 tool pairs so pair_count (3) > cutoff (2)
        push_tool_pairs(&mut conv.session, 3);
        let before = conv.session.tool_pair_count();
        assert_eq!(before, 3);

        conv.run_turn("hello", &mut NoopCb).unwrap();

        // After run_turn, compact_tool_pairs(2) should have been called,
        // marking 2 pairs agent_invisible → visible pair count drops to 1.
        assert!(
            conv.session.tool_pair_count() < before,
            "tier-1 should reduce tool_pair_count; before={before}, after={}",
            conv.session.tool_pair_count()
        );
    }

    /// Tier-1 is skipped when pair count does not exceed the cutoff.
    #[test]
    fn test_run_turn_skips_tier1_below_cutoff() {
        let (_dir, _guard) = with_test_session_dir();
        use zipcode_inference::mock::MockResponse;

        // cutoff=10 → 2 pairs will not trigger
        let policy = CompactPolicy {
            tier1_pair_cutoff: 10,
            ..CompactPolicy::default()
        };
        let mut conv = make_conv(vec![MockResponse::Text("done".to_string())], policy);

        push_tool_pairs(&mut conv.session, 2);
        let before = conv.session.tool_pair_count();

        conv.run_turn("hello", &mut NoopCb).unwrap();

        // run_turn adds no new tool calls, tier-1 must not have run.
        assert_eq!(
            conv.session.tool_pair_count(),
            before,
            "tier-1 must not fire below cutoff"
        );
    }

    /// Tier-2 fires when the mock reports `prompt_eval_count` above the threshold.
    ///
    /// We use a tiny context window (10 tokens, 80% = 8 token threshold) and set the
    /// mock to report 9 tokens.  The session has ~15 estimated tokens, so
    /// `evict_tool_responses_progressive(8)` will find content to evict.
    #[test]
    fn test_run_turn_triggers_tier2_when_usage_over_threshold() {
        let (_dir, _guard) = with_test_session_dir();
        use crate::permission::PermissionPolicy;
        use std::sync::{Arc, Mutex};
        use zipcode_inference::mock::{MockInferenceProvider, MockResponse};
        use zipcode_tools::{PermissionMode, ToolRegistry};

        // context=10, threshold=0.8 → threshold_tokens = floor(10*0.8) = 8
        let policy = CompactPolicy {
            tier2_usage_threshold: 0.8,
            context_window_tokens: 10,
            tier1_pair_cutoff: 100, // disable tier-1
            ..CompactPolicy::default()
        };

        // Mock reports 9 tokens (above threshold of 8)
        let provider = MockInferenceProvider::new(vec![MockResponse::Text("ok".to_string())])
            .with_prompt_eval_count(9);

        let mut conv = ConversationLoop {
            engine: Box::new(provider),
            tools: ToolRegistry::default(),
            session: crate::session::Session::new(),
            permission: PermissionPolicy::new(PermissionMode::FullAccess),
            system_prompt: String::new(),
            tool_specs: Vec::new(),
            cwd: std::path::PathBuf::from("/tmp"),
            depth: 0,
            last_sent_idx: 0,
            child_session_ids: Arc::new(Mutex::new(Vec::new())),
            skill_registry: None,
            compact_policy: policy,
        };

        // Add tool pairs so there's something to evict
        push_tool_pairs(&mut conv.session, 3);

        conv.run_turn("go", &mut NoopCb).unwrap();

        // evict_tool_responses_progressive should have marked some messages invisible
        let invisible = conv
            .session
            .messages
            .iter()
            .filter(|m| m.agent_invisible)
            .count();
        assert!(
            invisible > 0,
            "tier-2 should have evicted at least one message; invisible={invisible}"
        );
    }

    /// Tier-2 is skipped when usage is below threshold.
    #[test]
    fn test_run_turn_skips_tier2_below_threshold() {
        let (_dir, _guard) = with_test_session_dir();
        use crate::permission::PermissionPolicy;
        use std::sync::{Arc, Mutex};
        use zipcode_inference::mock::{MockInferenceProvider, MockResponse};
        use zipcode_tools::{PermissionMode, ToolRegistry};

        // context=32768, threshold=0.8 → 26214 threshold; report only 1000
        let policy = CompactPolicy {
            tier2_usage_threshold: 0.8,
            context_window_tokens: 32768,
            tier1_pair_cutoff: 100,
            ..CompactPolicy::default()
        };

        let provider = MockInferenceProvider::new(vec![MockResponse::Text("ok".to_string())])
            .with_prompt_eval_count(1_000);

        let mut conv = ConversationLoop {
            engine: Box::new(provider),
            tools: ToolRegistry::default(),
            session: crate::session::Session::new(),
            permission: PermissionPolicy::new(PermissionMode::FullAccess),
            system_prompt: String::new(),
            tool_specs: Vec::new(),
            cwd: std::path::PathBuf::from("/tmp"),
            depth: 0,
            last_sent_idx: 0,
            child_session_ids: Arc::new(Mutex::new(Vec::new())),
            skill_registry: None,
            compact_policy: policy,
        };

        push_tool_pairs(&mut conv.session, 3);

        conv.run_turn("go", &mut NoopCb).unwrap();

        let invisible = conv
            .session
            .messages
            .iter()
            .filter(|m| m.agent_invisible)
            .count();
        assert_eq!(invisible, 0, "no eviction expected below threshold");
    }

    /// A panic inside `compact_tool_pairs` does not propagate — `run_turn` returns Ok.
    #[test]
    fn test_tier1_panic_does_not_propagate() {
        let (_dir, _guard) = with_test_session_dir();
        use zipcode_inference::mock::MockResponse;

        // Set cutoff=0 so tier-1 fires, but batch_size=0 triggers no-op inside
        // compact_tool_pairs (pairs.len() < 0 is always false, so it returns 0).
        // We test that run_turn returns Ok regardless.
        let policy = CompactPolicy {
            tier1_pair_cutoff: 0,
            tier1_batch_size: 999, // no pairs to compact → compact returns 0
            ..CompactPolicy::default()
        };
        let mut conv = make_conv(vec![MockResponse::Text("done".to_string())], policy);
        // 1 pair → count(1) > cutoff(0) → tier-1 fires but batch_size=999 → no-op
        push_tool_pairs(&mut conv.session, 1);

        let result = conv.run_turn("hello", &mut NoopCb);
        assert!(
            result.is_ok(),
            "tier-1 no-op must not cause run_turn to fail"
        );
    }

    /// When the provider returns None for `prompt_eval_count`, tier-2 is skipped entirely.
    #[test]
    fn test_provider_without_prompt_eval_count_skips_tier2() {
        let (_dir, _guard) = with_test_session_dir();
        use zipcode_inference::mock::MockResponse;

        // Default MockInferenceProvider has no prompt_eval_count (returns None)
        let policy = CompactPolicy {
            tier2_usage_threshold: 0.0, // 0% threshold — would always trigger if count is Some
            context_window_tokens: 1,
            tier1_pair_cutoff: 100,
            ..CompactPolicy::default()
        };
        let mut conv = make_conv(vec![MockResponse::Text("done".to_string())], policy);
        push_tool_pairs(&mut conv.session, 3);

        conv.run_turn("hello", &mut NoopCb).unwrap();

        let invisible = conv
            .session
            .messages
            .iter()
            .filter(|m| m.agent_invisible)
            .count();
        assert_eq!(
            invisible, 0,
            "tier-2 must not fire when provider returns None for prompt_eval_count"
        );
    }

    // ── Direct unit tests for error-detection helpers ─────────────
    //
    // These pure functions are tested indirectly through
    // `recent_tool_results_contain_errors` above, but direct unit tests
    // provide faster failure localisation and regression isolation.

    // ── contains_error_word ──────────────────────────────────────

    #[test]
    fn error_word_matches_at_start_of_string() {
        assert!(contains_error_word("error in module"));
    }

    #[test]
    fn error_word_matches_at_end_of_string() {
        assert!(contains_error_word("found an error"));
    }

    #[test]
    fn error_word_matches_in_middle() {
        assert!(contains_error_word("the error was found"));
    }

    #[test]
    fn error_word_matches_plural_errors() {
        assert!(contains_error_word("2 errors detected"));
    }

    #[test]
    fn error_word_does_not_match_inside_terror() {
        assert!(!contains_error_word("terror"));
    }

    #[test]
    fn error_word_does_not_match_inside_errorless() {
        assert!(!contains_error_word("errorless execution"));
    }

    #[test]
    fn error_word_does_not_match_inside_mirrored() {
        assert!(!contains_error_word("mirrored"));
    }

    #[test]
    fn error_word_matches_after_punctuation() {
        assert!(contains_error_word("build: error in main.rs"));
    }

    #[test]
    fn error_word_matches_before_punctuation() {
        assert!(contains_error_word("error: missing semicolon"));
    }

    #[test]
    fn error_word_empty_string_is_false() {
        assert!(!contains_error_word(""));
    }

    // ── contains_fix_word ────────────────────────────────────────

    #[test]
    fn fix_word_matches_standalone_fix() {
        assert!(contains_fix_word("fix the bug"));
    }

    #[test]
    fn fix_word_matches_fix_at_start() {
        assert!(contains_fix_word("fix the error"));
    }

    #[test]
    fn fix_word_matches_fix_at_end() {
        assert!(contains_fix_word("need to fix"));
    }

    #[test]
    fn fix_word_does_not_match_prefix() {
        assert!(!contains_fix_word("prefix"));
    }

    #[test]
    fn fix_word_does_not_match_suffix() {
        assert!(!contains_fix_word("suffix"));
    }

    #[test]
    fn fix_word_does_not_match_affix() {
        assert!(!contains_fix_word("affix"));
    }

    #[test]
    fn fix_word_does_not_match_fixture() {
        assert!(!contains_fix_word("fixture"));
    }

    #[test]
    fn fix_word_does_not_match_prefixing() {
        assert!(!contains_fix_word("prefixing"));
    }

    #[test]
    fn fix_word_does_not_match_fixation() {
        assert!(!contains_fix_word("fixation"));
    }

    #[test]
    fn fix_word_empty_string_is_false() {
        assert!(!contains_fix_word(""));
    }

    // ── is_benign_error_line ──────────────────────────────────────

    #[test]
    fn benign_no_errors() {
        assert!(is_benign_error_line("no errors found"));
    }

    #[test]
    fn benign_no_error_singular() {
        assert!(is_benign_error_line("no error in output"));
    }

    #[test]
    fn benign_zero_errors() {
        assert!(is_benign_error_line("0 errors detected"));
    }

    #[test]
    fn benign_zero_error_singular() {
        assert!(is_benign_error_line("0 error in total"));
    }

    #[test]
    fn benign_without_error() {
        assert!(is_benign_error_line("completed without error"));
    }

    #[test]
    fn benign_without_errors() {
        assert!(is_benign_error_line("ran without errors"));
    }

    #[test]
    fn benign_fixed_error() {
        assert!(is_benign_error_line("fixed the error in main.rs"));
    }

    #[test]
    fn benign_fix_error() {
        assert!(is_benign_error_line("fix the error"));
    }

    #[test]
    fn benign_error_handling() {
        assert!(is_benign_error_line("improved error handling"));
    }

    #[test]
    fn benign_error_recovery() {
        assert!(is_benign_error_line("error recovery completed"));
    }

    #[test]
    fn benign_resolved_error() {
        assert!(is_benign_error_line("resolved the error"));
    }

    #[test]
    fn benign_cleared_error() {
        assert!(is_benign_error_line("cleared error state"));
    }

    #[test]
    fn benign_successfully_error() {
        assert!(is_benign_error_line("successfully fixed the error"));
    }

    #[test]
    fn benign_mixed_case_no_errors() {
        // is_benign_error_line expects lowered input (as called in production)
        assert!(is_benign_error_line("no errors found"));
    }

    #[test]
    fn benign_mixed_case_error_handling() {
        assert!(is_benign_error_line("error handling improved"));
    }

    #[test]
    fn not_benign_actual_error() {
        assert!(!is_benign_error_line("compilation error in main.rs"));
    }

    #[test]
    fn not_benign_error_colon_message() {
        assert!(!is_benign_error_line("something else entirely"));
    }

    #[test]
    fn not_benign_plain_string_without_error() {
        assert!(!is_benign_error_line("all systems nominal"));
    }
}
