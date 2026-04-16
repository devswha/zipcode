use anyhow::Result;
use tracing::info;

use zipcode_inference::chat_template::ToolSpec;
use zipcode_inference::{
    extract_text_content, ChatMessage, FinishReason, InferenceProvider, Role, TokenEvent,
};
use zipcode_tools::{execute_tool, ToolContext, ToolRegistry};

use crate::permission::{PermissionCheck, PermissionPolicy};
use crate::session::Session;

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

pub struct ConversationLoop {
    pub engine: Box<dyn InferenceProvider>,
    pub tools: ToolRegistry,
    pub session: Session,
    pub permission: PermissionPolicy,
    pub system_prompt: String,
    pub tool_specs: Vec<ToolSpec>,
    pub cwd: std::path::PathBuf,
}

impl ConversationLoop {
    pub fn run_turn(&mut self, user_input: &str, callback: &mut dyn StreamCallback) -> Result<()> {
        // Add system prompt on the very first turn
        if self.session.messages.is_empty() {
            self.session
                .push_message(ChatMessage::system(&self.system_prompt));
        }

        self.session.push_message(ChatMessage::user(user_input));

        const MAX_TOOL_ITERATIONS: usize = 25;
        const MAX_EMPTY_RETRIES: usize = 2;
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
            // Generate next response
            let rx = self
                .engine
                .generate_stream(&self.session.messages, &self.tool_specs);

            let mut full_text = String::new();
            let mut tool_calls = Vec::new();
            let mut _finish_reason = FinishReason::Stop;

            for event in rx {
                match event {
                    TokenEvent::Token(text) => {
                        callback.on_token(&text);
                        full_text.push_str(&text);
                    }
                    TokenEvent::Thinking(text) => {
                        // Route to the UI but do not accumulate into full_text —
                        // reasoning must not land in the stored assistant content
                        // because Gemma 4 strips it on re-injection
                        // (`supports_preserve_reasoning = false`). Keeping it out
                        // of history also prevents drift when the session is
                        // resumed or compacted.
                        callback.on_thinking(&text);
                    }
                    TokenEvent::ToolCall(call) => {
                        tool_calls.push(call);
                    }
                    TokenEvent::Done(reason) => {
                        _finish_reason = reason;
                        break;
                    }
                    TokenEvent::Error(e) => {
                        callback.on_error(&e.to_string());
                        self.session.save()?;
                        return Err(anyhow::anyhow!("Inference error: {e}"));
                    }
                }
            }

            // Store the assistant's response
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
            //
            // When a small model (E4B-class) sees error output from a tool
            // result, it sometimes produces an empty turn — thousands of
            // chars of reasoning followed by zero tool calls and no visible
            // text. This is the "give-up" pattern observed empirically in
            // docs/experiments/2026-04-16-gemma4-e4b-gamedev-probe.md. Rather
            // than terminating the turn immediately, we replace the silent
            // exit with a brief self-nudge ("Let me re-read the error…") and
            // give the model one more generation attempt with a fresh token
            // budget. Capped at MAX_EMPTY_RETRIES to avoid burning iterations
            // on a genuinely stuck model.
            if tool_calls.is_empty() {
                if full_text.trim().is_empty()
                    && empty_retries < MAX_EMPTY_RETRIES
                    && recent_tool_results_contain_errors(&self.session.messages)
                {
                    empty_retries += 1;
                    // Replace the empty assistant message (pushed above) with
                    // a self-nudge so the model sees stated intent rather than
                    // silence in its history.
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

            // Execute each tool call
            for call in &tool_calls {
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

            // Loop: model sees tool results and decides next action
        }

        self.session.save()?;
        Ok(())
    }
}

/// Walk backwards through recent messages looking for tool results that
/// contain error indicators. Stops at the first user or system message
/// so we only consider tool output from the current agentic cycle.
fn recent_tool_results_contain_errors(messages: &[ChatMessage]) -> bool {
    for msg in messages.iter().rev() {
        match msg.role {
            Role::Model => continue,
            Role::Tool => {
                let c = &msg.content;
                if c.contains("error")
                    || c.contains("Error")
                    || c.contains("FAILED")
                    || c.contains("panicked")
                {
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
}
