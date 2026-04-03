pub mod chat_template;
pub mod device;
pub mod engine;
pub mod mock;
pub mod sampler;
pub mod types;

pub use chat_template::{
    extract_text_content, format_conversation, format_message, parse_tool_calls, ToolSpec,
};
pub use device::select_device;
pub use engine::InferenceEngine;
pub use mock::{MockInferenceProvider, MockResponse};
pub use types::*;

/// Abstraction over inference backends — real or mock.
pub trait InferenceProvider: Send {
    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[chat_template::ToolSpec],
    ) -> std::sync::mpsc::Receiver<TokenEvent>;
}
