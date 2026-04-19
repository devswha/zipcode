use std::collections::VecDeque;
use std::sync::{mpsc, Arc, Mutex};

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
    manages_own_context_flag: bool,
    pub captured_messages: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
}

impl MockInferenceProvider {
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: VecDeque::from(responses),
            manages_own_context_flag: false,
            captured_messages: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Set whether this mock reports that it manages its own context.
    pub fn with_manages_own_context(mut self, v: bool) -> Self {
        self.manages_own_context_flag = v;
        self
    }

    /// Returns a clone of the message slices captured by each `generate_stream` call.
    pub fn captured_messages(&self) -> Vec<Vec<ChatMessage>> {
        self.captured_messages.lock().unwrap().clone()
    }
}

impl InferenceProvider for MockInferenceProvider {
    fn manages_own_context(&self) -> bool {
        self.manages_own_context_flag
    }

    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        _tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent> {
        self.captured_messages
            .lock()
            .unwrap()
            .push(messages.to_vec());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferenceProvider;

    /// Helper: drain all events from a receiver into a Vec.
    fn collect_events(rx: mpsc::Receiver<TokenEvent>) -> Vec<TokenEvent> {
        rx.iter().collect()
    }

    #[test]
    fn test_mock_text_response() {
        let mut mock = MockInferenceProvider::new(vec![MockResponse::Text("hello".to_string())]);
        let rx = mock.generate_stream(&[], &[]);
        let events = collect_events(rx);

        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], TokenEvent::Token(t) if t == "hello"));
        assert!(matches!(&events[1], TokenEvent::Done(FinishReason::Stop)));
    }

    #[test]
    fn test_mock_tool_call_response() {
        let args = serde_json::json!({"command": "ls -la"});
        let mut mock = MockInferenceProvider::new(vec![MockResponse::ToolCall {
            name: "bash".to_string(),
            args: args.clone(),
        }]);
        let rx = mock.generate_stream(&[], &[]);
        let events = collect_events(rx);

        assert_eq!(events.len(), 2);
        match &events[0] {
            TokenEvent::ToolCall(call) => {
                assert_eq!(call.name, "bash");
                assert_eq!(call.arguments, args);
            }
            other => panic!("expected ToolCall, got {:?}", other),
        }
        assert!(matches!(
            &events[1],
            TokenEvent::Done(FinishReason::ToolUse)
        ));
    }

    #[test]
    fn test_mock_events_response() {
        let custom_events = vec![
            TokenEvent::Token("part1 ".to_string()),
            TokenEvent::Token("part2".to_string()),
            TokenEvent::Done(FinishReason::Stop),
        ];
        let mut mock = MockInferenceProvider::new(vec![MockResponse::Events(custom_events)]);
        let rx = mock.generate_stream(&[], &[]);
        let events = collect_events(rx);

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], TokenEvent::Token(t) if t == "part1 "));
        assert!(matches!(&events[1], TokenEvent::Token(t) if t == "part2"));
        assert!(matches!(&events[2], TokenEvent::Done(FinishReason::Stop)));
    }

    #[test]
    fn test_mock_error_response() {
        let mut mock = MockInferenceProvider::new(vec![MockResponse::Error(
            InferenceError::TokenizerError("bad token".to_string()),
        )]);
        let rx = mock.generate_stream(&[], &[]);
        let events = collect_events(rx);

        assert_eq!(events.len(), 1);
        match &events[0] {
            TokenEvent::Error(InferenceError::TokenizerError(msg)) => {
                assert_eq!(msg, "bad token");
            }
            other => panic!("expected TokenError, got {:?}", other),
        }
    }

    #[test]
    fn test_mock_exhausted_responses() {
        let mut mock: MockInferenceProvider = MockInferenceProvider::new(vec![]);
        let rx = mock.generate_stream(&[], &[]);
        let events = collect_events(rx);

        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], TokenEvent::Token(t) if t == "(no more mock responses)"));
        assert!(matches!(&events[1], TokenEvent::Done(FinishReason::Stop)));
    }

    #[test]
    fn test_mock_sequential_responses() {
        let mut mock = MockInferenceProvider::new(vec![
            MockResponse::Text("first".to_string()),
            MockResponse::Text("second".to_string()),
            MockResponse::Text("third".to_string()),
        ]);

        // First call returns "first"
        let rx1 = mock.generate_stream(&[], &[]);
        let events1 = collect_events(rx1);
        assert!(matches!(&events1[0], TokenEvent::Token(t) if t == "first"));

        // Second call returns "second"
        let rx2 = mock.generate_stream(&[], &[]);
        let events2 = collect_events(rx2);
        assert!(matches!(&events2[0], TokenEvent::Token(t) if t == "second"));

        // Third call returns "third"
        let rx3 = mock.generate_stream(&[], &[]);
        let events3 = collect_events(rx3);
        assert!(matches!(&events3[0], TokenEvent::Token(t) if t == "third"));

        // Fourth call falls through to exhausted
        let rx4 = mock.generate_stream(&[], &[]);
        let events4 = collect_events(rx4);
        assert!(matches!(&events4[0], TokenEvent::Token(t) if t == "(no more mock responses)"));
    }

    #[test]
    fn test_mock_tool_call_has_correct_id() {
        // After popping the first response, the VecDeque has 2 items left,
        // so the ID should be "mock_call_2".
        let mut mock = MockInferenceProvider::new(vec![
            MockResponse::ToolCall {
                name: "read_file".to_string(),
                args: serde_json::json!({"path": "foo.rs"}),
            },
            MockResponse::Text("filler 1".to_string()),
            MockResponse::Text("filler 2".to_string()),
        ]);

        let rx = mock.generate_stream(&[], &[]);
        let events = collect_events(rx);

        match &events[0] {
            TokenEvent::ToolCall(call) => {
                // VecDeque had 3 items, pop_front removes one → 2 remain
                assert_eq!(call.id, "mock_call_2");
                assert_eq!(call.name, "read_file");
            }
            other => panic!("expected ToolCall, got {:?}", other),
        }
    }

    #[test]
    fn mock_provider_does_not_manage_own_context() {
        let provider = MockInferenceProvider::new(vec![]);
        assert!(!provider.manages_own_context());
    }

    #[test]
    fn test_mock_ignores_messages_and_tools() {
        // The mock ignores its inputs — verify it still produces the
        // configured response regardless of what's passed.
        let mut mock = MockInferenceProvider::new(vec![MockResponse::Text("ok".to_string())]);

        let messages = vec![
            ChatMessage::system("you are helpful"),
            ChatMessage::user("do a thing"),
        ];
        let tools = vec![ToolSpec {
            name: "bash".to_string(),
            description: "run commands".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }];

        let rx = mock.generate_stream(&messages, &tools);
        let events = collect_events(rx);
        assert!(matches!(&events[0], TokenEvent::Token(t) if t == "ok"));
    }
}
