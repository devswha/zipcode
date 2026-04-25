use std::collections::VecDeque;
use std::sync::{mpsc, Arc, Mutex};

use crate::chat_template::{ChatTemplate, GemmaTemplate, ToolSpec};
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
    /// Shared response queue — parent and cloned children draw from the same
    /// deque so tests can pre-load responses for both in one place.
    responses: Arc<Mutex<VecDeque<MockResponse>>>,
    manages_own_context_flag: bool,
    pub captured_messages: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
    /// Fixed value returned by `last_prompt_eval_count()`, set via
    /// `with_prompt_eval_count()`.  `None` by default so tests that do not
    /// opt in leave tier-2 logic inactive.
    prompt_eval_count_mock: Option<usize>,
    /// Chat template associated with this mock provider.
    /// Defaults to [`GemmaTemplate`]; override with [`Self::with_template`].
    /// Stored as `Arc` so [`clone_for_child`] can share it without cloning the inner value.
    template: Arc<dyn ChatTemplate>,
}

impl MockInferenceProvider {
    #[must_use]
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
            manages_own_context_flag: false,
            captured_messages: Arc::new(Mutex::new(Vec::new())),
            prompt_eval_count_mock: None,
            template: Arc::new(GemmaTemplate),
        }
    }

    /// Set whether this mock reports that it manages its own context.
    #[must_use]
    pub fn with_manages_own_context(self, v: bool) -> Self {
        Self {
            manages_own_context_flag: v,
            ..self
        }
    }

    /// Fix the value returned by `last_prompt_eval_count()`.
    /// Used by tier-2 tests to simulate context usage without a real server.
    #[must_use]
    pub fn with_prompt_eval_count(self, count: usize) -> Self {
        Self {
            prompt_eval_count_mock: Some(count),
            ..self
        }
    }

    /// Override the chat template reported by this mock provider.
    ///
    /// The mock does not use the template for actual generation, but stores it
    /// so tests can assert that the correct template is selected and threaded
    /// through the provider construction path.
    #[must_use]
    pub fn with_template(self, template: Box<dyn ChatTemplate>) -> Self {
        Self {
            template: Arc::from(template),
            ..self
        }
    }

    /// Return the chat template currently associated with this provider.
    #[must_use]
    pub fn template(&self) -> &dyn ChatTemplate {
        self.template.as_ref()
    }

    /// Returns a clone of the message slices captured by each `generate_stream` call.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned (only possible if a previous
    /// `lock()` call panicked, which cannot happen in normal usage).
    #[must_use]
    pub fn captured_messages(&self) -> Vec<Vec<ChatMessage>> {
        self.captured_messages
            .lock()
            .expect("mock captured_messages mutex should never be poisoned")
            .clone()
    }
}

impl InferenceProvider for MockInferenceProvider {
    fn manages_own_context(&self) -> bool {
        self.manages_own_context_flag
    }

    /// Clone shares the same response queue so pre-loaded child responses are
    /// consumed in order alongside parent responses from a single `new()` call.
    fn clone_for_child(&self) -> Option<Box<dyn crate::InferenceProvider>> {
        Some(Box::new(Self {
            responses: Arc::clone(&self.responses),
            manages_own_context_flag: self.manages_own_context_flag,
            captured_messages: Arc::clone(&self.captured_messages),
            prompt_eval_count_mock: self.prompt_eval_count_mock,
            template: Arc::clone(&self.template),
        }))
    }

    fn last_prompt_eval_count(&self) -> Option<usize> {
        self.prompt_eval_count_mock
    }

    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        _tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent> {
        self.captured_messages
            .lock()
            .expect("mock captured_messages mutex should never be poisoned")
            .push(messages.to_vec());
        let (tx, rx) = mpsc::channel();

        let next = self
            .responses
            .lock()
            .expect("mock responses mutex should never be poisoned")
            .pop_front();

        match next {
            Some(MockResponse::Text(text)) => {
                let _ = tx.send(TokenEvent::Token(text));
                let _ = tx.send(TokenEvent::Done(FinishReason::Stop));
            }
            Some(MockResponse::ToolCall { name, args }) => {
                let remaining = self
                    .responses
                    .lock()
                    .expect("mock responses mutex should never be poisoned")
                    .len();
                let call = ToolCallParsed {
                    id: format!("mock_call_{remaining}"),
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

    #[test]
    fn test_captured_messages_records_all_calls() {
        let mut mock = MockInferenceProvider::new(vec![
            MockResponse::Text("first".to_string()),
            MockResponse::Text("second".to_string()),
        ]);

        let msgs1 = vec![ChatMessage::user("hello")];
        let msgs2 = vec![
            ChatMessage::user("world"),
            ChatMessage::assistant("response"),
        ];

        let _ = mock.generate_stream(&msgs1, &[]);
        let _ = mock.generate_stream(&msgs2, &[]);

        let captured = mock.captured_messages();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].len(), 1);
        assert_eq!(captured[0][0].content, "hello");
        assert_eq!(captured[1].len(), 2);
        assert_eq!(captured[1][0].content, "world");
        assert_eq!(captured[1][1].content, "response");
    }

    #[test]
    fn test_with_manages_own_context_true() {
        let provider = MockInferenceProvider::new(vec![]).with_manages_own_context(true);
        assert!(provider.manages_own_context());
    }

    #[test]
    fn test_mock_multiple_tool_calls_sequential() {
        let mut mock = MockInferenceProvider::new(vec![
            MockResponse::ToolCall {
                name: "bash".to_string(),
                args: serde_json::json!({"command": "ls"}),
            },
            MockResponse::ToolCall {
                name: "read_file".to_string(),
                args: serde_json::json!({"path": "foo.rs"}),
            },
        ]);

        let rx1 = mock.generate_stream(&[], &[]);
        let events1 = collect_events(rx1);
        match &events1[0] {
            TokenEvent::ToolCall(call) => {
                assert_eq!(call.name, "bash");
            }
            other => panic!("expected ToolCall, got {:?}", other),
        }

        let rx2 = mock.generate_stream(&[], &[]);
        let events2 = collect_events(rx2);
        match &events2[0] {
            TokenEvent::ToolCall(call) => {
                assert_eq!(call.name, "read_file");
            }
            other => panic!("expected ToolCall, got {:?}", other),
        }
    }
}
