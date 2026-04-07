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
pub use llama_server_backend::{LlamaServerProvider, ServerOptions};
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
            "llama-cpp" => Ok(Backend::LlamaCpp),
            "llama-server" => Ok(Backend::LlamaServer),
            "candle" => Ok(Backend::Candle),
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
    create_engine_inner(backend, model_path, tokenizer_path, config, server_options)
}

#[cfg(all(feature = "llama-cpp", feature = "candle"))]
fn create_engine_inner(
    backend: Backend,
    model_path: &std::path::Path,
    tokenizer_path: &std::path::Path,
    config: GenerationConfig,
    server_options: ServerOptions,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    match backend {
        Backend::LlamaCpp => {
            let _ = &server_options;
            let mut provider = LlamaCppProvider::load(model_path)?;
            provider.set_config(config);
            Ok(Box::new(provider))
        }
        Backend::LlamaServer => {
            let mut provider = LlamaServerProvider::load(model_path, &server_options)?;
            provider.set_config(config);
            Ok(Box::new(provider))
        }
        Backend::Candle => {
            let _ = &server_options;
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
    server_options: ServerOptions,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    match backend {
        Backend::LlamaCpp => {
            let _ = &server_options;
            let mut provider = LlamaCppProvider::load(model_path)?;
            provider.set_config(config);
            Ok(Box::new(provider))
        }
        Backend::LlamaServer => {
            let mut provider = LlamaServerProvider::load(model_path, &server_options)?;
            provider.set_config(config);
            Ok(Box::new(provider))
        }
        Backend::Candle => {
            let _ = config;
            let _ = &server_options;
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
    server_options: ServerOptions,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    match backend {
        Backend::LlamaServer => {
            let mut provider = LlamaServerProvider::load(model_path, &server_options)?;
            provider.set_config(config);
            Ok(Box::new(provider))
        }
        Backend::Candle => {
            let _ = &server_options;
            let device = select_device();
            let mut engine = InferenceEngine::load(model_path, tokenizer_path, device)?;
            engine.set_config(config);
            Ok(Box::new(engine))
        }
        Backend::LlamaCpp => {
            let _ = config;
            let _ = &server_options;
            anyhow::bail!(
                "llama-cpp backend requested but the `llama-cpp` feature is not enabled. \
                 Rebuild with: cargo build --features llama-cpp"
            );
        }
    }
}

#[cfg(all(not(feature = "llama-cpp"), not(feature = "candle")))]
fn create_engine_inner(
    backend: Backend,
    model_path: &std::path::Path,
    _tokenizer_path: &std::path::Path,
    config: GenerationConfig,
    server_options: ServerOptions,
) -> anyhow::Result<Box<dyn InferenceProvider>> {
    match backend {
        Backend::LlamaServer => {
            let mut provider = LlamaServerProvider::load(model_path, &server_options)?;
            provider.set_config(config);
            Ok(Box::new(provider))
        }
        Backend::LlamaCpp | Backend::Candle => {
            let _ = config;
            let _ = &server_options;
            anyhow::bail!(
                "No native inference backend enabled. Enable at least one of the `candle` or `llama-cpp` features, or use `llama-server`."
            )
        }
    }
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
