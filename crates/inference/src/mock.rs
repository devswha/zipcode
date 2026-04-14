use std::collections::VecDeque;
use std::sync::mpsc;

use crate::chat_template::ToolSpec;
use crate::types::{ChatMessage, FinishReason, InferenceError, TokenEvent, ToolCallParsed};
use crate::InferenceProvider;

pub enum MockResponse {
    /// Return plain text
    Text(String),
    /// Return a tool call
    ToolCall {
        name: String,
        args: serde_json::Value,
    },
    /// Return an explicit event stream
    Events(Vec<TokenEvent>),
    /// Return an error
    Error(InferenceError),
}

pub struct MockInferenceProvider {
    responses: VecDeque<MockResponse>,
}

impl MockInferenceProvider {
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: VecDeque::from(responses),
        }
    }
}

impl InferenceProvider for MockInferenceProvider {
    fn generate_stream(
        &mut self,
        _messages: &[ChatMessage],
        _tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel();

        match self.responses.pop_front() {
            Some(MockResponse::Text(text)) => {
                let _ = tx.send(TokenEvent::Token(text));
                let _ = tx.send(TokenEvent::Done(FinishReason::Stop));
            }
            Some(MockResponse::ToolCall { name, args }) => {
                let call = ToolCallParsed {
                    id: format!("mock_call_{}", self.responses.len()),
                    name,
                    arguments: args,
                };
                let _ = tx.send(TokenEvent::ToolCall(call));
                let _ = tx.send(TokenEvent::Done(FinishReason::ToolUse));
            }
            Some(MockResponse::Events(events)) => {
                for event in events {
                    let _ = tx.send(event);
                }
            }
            Some(MockResponse::Error(e)) => {
                let _ = tx.send(TokenEvent::Error(e));
            }
            None => {
                let _ = tx.send(TokenEvent::Token("(no more mock responses)".to_string()));
                let _ = tx.send(TokenEvent::Done(FinishReason::Stop));
            }
        }

        rx
    }
}
