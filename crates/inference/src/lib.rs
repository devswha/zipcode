pub mod chat_template;
#[cfg(feature = "candle")]
pub mod device;
#[cfg(feature = "candle")]
pub mod engine;
#[cfg(feature = "llama-cpp")]
pub mod llama_cpp_backend;
pub mod llama_server_backend;
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
pub use llama_server_backend::{LlamaServerProvider, ServerOptions, DEFAULT_CONTEXT_SIZE};
pub use mock::{MockInferenceProvider, MockResponse};
pub use types::*;

/// Abstraction over inference backends — real or mock.
pub trait InferenceProvider: Send {
    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[chat_template::ToolSpec],
    ) -> std::sync::mpsc::Receiver<TokenEvent>;

    /// Returns true if this provider manages its own conversation context
    /// (KV cache) on the server side and does not need the caller to pass
    /// the full accumulated message history on every turn.
    ///
    /// Default: false. Callers that see true should send only the newest
    /// turn (latest user message + new tool results), not the full history,
    /// to avoid double-history accumulation.
    fn manages_own_context(&self) -> bool {
        false
    }
}

/// Which inference backend to use.
#[derive(Debug, Clone, Copy)]
pub enum Backend {
    /// llama.cpp via llama-cpp-2 bindings — supports GGUF including Gemma 4.
    LlamaCpp,
    /// llama.cpp server subprocess using the OpenAI-compatible HTTP API.
    LlamaServer,
    /// Candle (pure-Rust) backend.
    Candle,
}

impl Backend {
    /// Parse a backend name string.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend name is unsupported.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s {
            "llama-cpp" => Ok(Self::LlamaCpp),
            "llama-server" => Ok(Self::LlamaServer),
            "candle" => Ok(Self::Candle),
            other => anyhow::bail!(
                "unsupported backend '{other}'. Expected one of: llama-cpp, llama-server, candle"
            ),
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
    server_options: ServerOptions,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    match backend {
        Backend::LlamaServer => {
            let mut provider = LlamaServerProvider::load(model_path, &server_options)?;
            provider.set_config(config);
            Ok(Box::new(provider))
        }
        Backend::LlamaCpp => create_llama_cpp(model_path, config, server_options),
        Backend::Candle => create_candle(model_path, tokenizer_path, config, server_options),
    }
}

/// Load a llama-cpp backend provider.
///
/// Returns an error if the `llama-cpp` feature is not enabled.
#[cfg(feature = "llama-cpp")]
fn create_llama_cpp(
    model_path: &std::path::Path,
    config: GenerationConfig,
    _server_options: ServerOptions,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    let mut provider = LlamaCppProvider::load(model_path)?;
    provider.set_config(config);
    Ok(Box::new(provider))
}

#[cfg(not(feature = "llama-cpp"))]
fn create_llama_cpp(
    _model_path: &std::path::Path,
    _config: GenerationConfig,
    _server_options: ServerOptions,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    anyhow::bail!(
        "llama-cpp backend requested but the `llama-cpp` feature is not enabled. \
         Rebuild with: cargo build --features llama-cpp"
    )
}

/// Load a Candle (pure-Rust) backend.
///
/// Returns an error if the `candle` feature is not enabled.
#[cfg(feature = "candle")]
fn create_candle(
    model_path: &std::path::Path,
    tokenizer_path: &std::path::Path,
    config: GenerationConfig,
    _server_options: ServerOptions,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    let device = select_device();
    let mut engine = InferenceEngine::load(model_path, tokenizer_path, device)?;
    engine.set_config(config);
    Ok(Box::new(engine))
}

#[cfg(not(feature = "candle"))]
fn create_candle(
    _model_path: &std::path::Path,
    _tokenizer_path: &std::path::Path,
    _config: GenerationConfig,
    _server_options: ServerOptions,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    anyhow::bail!(
        "candle backend requested but the `candle` feature is not enabled. \
         Rebuild with: cargo build --features candle"
    )
}

#[cfg(test)]
mod tests {
    use super::Backend;
    #[cfg(feature = "llama-cpp")]
    use super::{create_engine, GenerationConfig, ServerOptions};
    #[cfg(feature = "llama-cpp")]
    use std::path::Path;

    #[test]
    fn parse_accepts_known_backends() {
        assert!(matches!(
            Backend::parse("llama-cpp").unwrap(),
            Backend::LlamaCpp
        ));
        assert!(matches!(
            Backend::parse("llama-server").unwrap(),
            Backend::LlamaServer
        ));
        assert!(matches!(Backend::parse("candle").unwrap(), Backend::Candle));
    }

    #[test]
    fn parse_rejects_unknown_backend() {
        let error = Backend::parse("llama").unwrap_err().to_string();
        assert!(error.contains("unsupported backend"));
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    #[ignore = "requires ZIPCODE_TEST_MODEL_PATH and ZIPCODE_LLAMA_SERVER_BIN for a local Gemma 4 GGUF"]
    fn local_gemma4_model_loads_via_llama_server() {
        let model_path = std::env::var("ZIPCODE_TEST_MODEL_PATH")
            .expect("ZIPCODE_TEST_MODEL_PATH must point to a local Gemma 4 GGUF");
        let model_path = Path::new(&model_path);

        let engine = create_engine(
            Backend::LlamaServer,
            model_path,
            Path::new("unused-tokenizer.json"),
            GenerationConfig::default(),
            ServerOptions::default(),
        );

        if let Err(error) = engine {
            panic!("expected Gemma 4 GGUF to load via llama-server, got: {error}");
        }
    }
}
