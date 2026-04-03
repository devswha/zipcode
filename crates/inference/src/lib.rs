pub mod chat_template;
#[cfg(feature = "candle")]
pub mod device;
#[cfg(feature = "candle")]
pub mod engine;
#[cfg(feature = "llama-cpp")]
pub mod llama_cpp_backend;
pub mod mock;
#[cfg(feature = "candle")]
pub mod sampler;
pub mod types;

pub use chat_template::{
    extract_text_content, format_conversation, format_message, parse_tool_calls, ToolSpec,
};
#[cfg(feature = "candle")]
pub use device::select_device;
#[cfg(feature = "candle")]
pub use engine::InferenceEngine;
#[cfg(feature = "llama-cpp")]
pub use llama_cpp_backend::LlamaCppProvider;
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

/// Which inference backend to use.
#[derive(Debug, Clone, Copy)]
pub enum Backend {
    /// llama.cpp via llama-cpp-2 bindings — supports GGUF including Gemma 4.
    LlamaCpp,
    /// Candle (pure-Rust) backend.
    Candle,
}

impl Backend {
    /// Parse a backend name string. Defaults to `LlamaCpp` for unknown values.
    #[must_use]
    pub fn from_name(s: &str) -> Self {
        match s {
            "candle" => Backend::Candle,
            _ => Backend::LlamaCpp,
        }
    }
}

/// Create an inference engine for the given backend and model files.
///
/// # Errors
///
/// Returns an error if the model cannot be loaded or a required feature flag is
/// not enabled.
pub fn create_engine(
    backend: Backend,
    model_path: &std::path::Path,
    tokenizer_path: &std::path::Path,
    config: GenerationConfig,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    create_engine_inner(backend, model_path, tokenizer_path, config)
}

#[cfg(all(feature = "llama-cpp", feature = "candle"))]
fn create_engine_inner(
    backend: Backend,
    model_path: &std::path::Path,
    tokenizer_path: &std::path::Path,
    config: GenerationConfig,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    match backend {
        Backend::LlamaCpp => {
            let mut provider = LlamaCppProvider::load(model_path)?;
            provider.set_config(config);
            Ok(Box::new(provider))
        }
        Backend::Candle => {
            let device = select_device();
            let mut engine = InferenceEngine::load(model_path, tokenizer_path, device)?;
            engine.set_config(config);
            Ok(Box::new(engine))
        }
    }
}

#[cfg(all(feature = "llama-cpp", not(feature = "candle")))]
fn create_engine_inner(
    backend: Backend,
    model_path: &std::path::Path,
    _tokenizer_path: &std::path::Path,
    config: GenerationConfig,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    match backend {
        Backend::LlamaCpp => {
            let mut provider = LlamaCppProvider::load(model_path)?;
            provider.set_config(config);
            Ok(Box::new(provider))
        }
        Backend::Candle => {
            let _ = config;
            anyhow::bail!(
                "candle backend requested but the `candle` feature is not enabled. \
                 Rebuild with: cargo build --features candle"
            );
        }
    }
}

#[cfg(all(not(feature = "llama-cpp"), feature = "candle"))]
fn create_engine_inner(
    backend: Backend,
    model_path: &std::path::Path,
    tokenizer_path: &std::path::Path,
    config: GenerationConfig,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    match backend {
        Backend::Candle => {
            let device = select_device();
            let mut engine = InferenceEngine::load(model_path, tokenizer_path, device)?;
            engine.set_config(config);
            Ok(Box::new(engine))
        }
        Backend::LlamaCpp => {
            let _ = config;
            anyhow::bail!(
                "llama-cpp backend requested but the `llama-cpp` feature is not enabled. \
                 Rebuild with: cargo build --features llama-cpp"
            );
        }
    }
}

#[cfg(all(not(feature = "llama-cpp"), not(feature = "candle")))]
fn create_engine_inner(
    _backend: Backend,
    _model_path: &std::path::Path,
    _tokenizer_path: &std::path::Path,
    _config: GenerationConfig,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    anyhow::bail!(
        "No inference backend enabled. Enable at least one of the `candle` or `llama-cpp` features."
    )
}
