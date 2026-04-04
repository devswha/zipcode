use anyhow::Result;
use tracing::info;

use zipcode_inference::chat_template::ToolSpec;
use zipcode_inference::{ChatMessage, FinishReason, InferenceProvider, TokenEvent};
use zipcode_tools::{execute_tool, ToolContext, ToolRegistry};

use crate::permission::{PermissionCheck, PermissionPolicy};
use crate::session::Session;

/// Callback for streaming tokens and events to the UI layer.
pub trait StreamCallback: Send {
    fn on_token(&mut self, text: &str);
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
        let mut iterations = 0;
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
                    TokenEvent::ToolCall(call) => {
                        tool_calls.push(call);
                    }
                    TokenEvent::Done(reason) => {
                        _finish_reason = reason;
                        break;
                    }
                    TokenEvent::Error(e) => {
                        callback.on_error(&e.to_string());
                        return Err(anyhow::anyhow!("Inference error: {e}"));
                    }
                }
            }

            // Store the assistant's response
            if tool_calls.is_empty() {
                self.session
                    .push_message(ChatMessage::assistant(&full_text));
            } else {
                self.session
                    .push_message(ChatMessage::assistant_with_tool_calls(
                        &full_text,
                        tool_calls.clone(),
                    ));
            }

            // No tool calls → turn is complete
            if tool_calls.is_empty() {
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
