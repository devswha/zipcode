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

    /// Return a fresh provider instance suitable for a child agent, or `None`
    /// if this backend cannot be cloned (e.g. it holds exclusive OS resources).
    /// Child agents call this to get their own independent provider.
    fn clone_for_child(&self) -> Option<Box<dyn InferenceProvider>> {
        None
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
    use super::{
        create_engine, ChatMessage, GenerationConfig, InferenceProvider, MockInferenceProvider,
        MockResponse, ServerOptions,
    };
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

    // ── Backend::parse edge-case tests ──────────────────────────────

    #[test]
    fn parse_rejects_empty_string() {
        let err = Backend::parse("").unwrap_err().to_string();
        assert!(
            err.contains("unsupported backend"),
            "empty string should be rejected, got: {err}"
        );
        assert!(
            err.contains("Expected one of"),
            "error should list valid options, got: {err}"
        );
    }

    #[test]
    fn parse_rejects_whitespace_strings() {
        for input in &[" ", "  ", "\t", "\n", " llama-server "] {
            let err = Backend::parse(input).unwrap_err().to_string();
            assert!(
                err.contains("unsupported backend"),
                "whitespace-only or padded '{input}' should be rejected, got: {err}"
            );
        }
    }

    #[test]
    fn parse_is_case_sensitive() {
        // Uppercase variants should all be rejected
        for input in &["LLAMA-CPP", "LLAMA-SERVER", "CANDLE", "Llama-Cpp", "Candle"] {
            let err = Backend::parse(input).unwrap_err().to_string();
            assert!(
                err.contains("unsupported backend"),
                "case-variant '{input}' should be rejected, got: {err}"
            );
        }
    }

    #[test]
    fn parse_rejects_common_typos() {
        let typos = [
            ("llamacpp", "missing hyphen"),
            ("llama_server", "underscore instead of hyphen"),
            ("llama-server-cpp", "extra suffix"),
            ("cpp", "bare name"),
            ("llama.cpp", "dotted name"),
            ("candle-rs", "suffixed name"),
            ("vllm", "unrelated backend"),
            ("ollama", "unrelated backend"),
            ("1", "numeric"),
            ("123", "numeric"),
        ];
        for (input, reason) in &typos {
            let err = Backend::parse(input).unwrap_err().to_string();
            assert!(
                err.contains("unsupported backend"),
                "typo '{input}' ({reason}) should be rejected, got: {err}"
            );
        }
    }

    #[test]
    fn parse_error_includes_all_valid_options() {
        let err = Backend::parse("nope").unwrap_err().to_string();
        assert!(err.contains("llama-cpp"), "error should list llama-cpp");
        assert!(
            err.contains("llama-server"),
            "error should list llama-server"
        );
        assert!(err.contains("candle"), "error should list candle");
    }

    #[test]
    fn parse_error_echoes_invalid_input() {
        let err = Backend::parse("my-backend").unwrap_err().to_string();
        assert!(
            err.contains("my-backend"),
            "error should echo the invalid input, got: {err}"
        );
    }

    // ── Backend enum derive verification ─────────────────────────────

    #[test]
    fn backend_variants_clone_correctly() {
        let a = Backend::LlamaCpp;
        let b = a;
        assert!(matches!(b, Backend::LlamaCpp));

        let c = Backend::LlamaServer;
        let d = c;
        assert!(matches!(d, Backend::LlamaServer));

        let e = Backend::Candle;
        let f = e;
        assert!(matches!(f, Backend::Candle));
    }

    #[test]
    fn backend_debug_format_is_readable() {
        assert_eq!(format!("{:?}", Backend::LlamaCpp), "LlamaCpp");
        assert_eq!(format!("{:?}", Backend::LlamaServer), "LlamaServer");
        assert_eq!(format!("{:?}", Backend::Candle), "Candle");
    }

    // ── InferenceProvider trait default method tests ──────────────────

    #[test]
    fn mock_provider_default_manages_own_context_is_false() {
        let mock = MockInferenceProvider::new(vec![]);
        assert!(
            !mock.manages_own_context(),
            "default manages_own_context should be false"
        );
    }

    #[test]
    fn mock_provider_with_manages_own_context_true() {
        let mock = MockInferenceProvider::new(vec![]).with_manages_own_context(true);
        assert!(
            mock.manages_own_context(),
            "explicitly set manages_own_context should be true"
        );
    }

    #[test]
    fn mock_provider_with_manages_own_context_toggle() {
        // Setting true then verifying, then creating one with false
        let mock_true = MockInferenceProvider::new(vec![]).with_manages_own_context(true);
        let mock_false = MockInferenceProvider::new(vec![]).with_manages_own_context(false);
        assert!(mock_true.manages_own_context());
        assert!(!mock_false.manages_own_context());
    }

    #[test]
    fn mock_provider_captures_messages_across_calls() {
        let mut mock = MockInferenceProvider::new(vec![
            MockResponse::Text("first".to_string()),
            MockResponse::Text("second".to_string()),
        ]);
        let msgs1 = vec![ChatMessage::user("hello")];
        let msgs2 = vec![ChatMessage::user("world")];

        let _ = mock.generate_stream(&msgs1, &[]);
        let _ = mock.generate_stream(&msgs2, &[]);

        let captured = mock.captured_messages();
        assert_eq!(captured.len(), 2, "should have captured 2 calls");
        assert_eq!(captured[0][0].content, "hello");
        assert_eq!(captured[1][0].content, "world");
    }

    // ── create_engine error path (feature-not-enabled) ────────────────

    #[cfg(not(feature = "llama-cpp"))]
    #[test]
    fn create_engine_llama_cpp_rejected_without_feature() {
        let result = create_engine(
            Backend::LlamaCpp,
            Path::new("/dev/null"),
            Path::new("/dev/null"),
            GenerationConfig::default(),
            ServerOptions::default(),
        );
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error when llama-cpp feature is not enabled"),
        };
        assert!(
            err.contains("llama-cpp backend requested"),
            "should mention llama-cpp, got: {err}"
        );
        assert!(
            err.contains("feature is not enabled"),
            "should say feature not enabled, got: {err}"
        );
        assert!(
            err.contains("cargo build --features llama-cpp"),
            "should suggest rebuild command, got: {err}"
        );
    }

    #[cfg(not(feature = "candle"))]
    #[test]
    fn create_engine_candle_rejected_without_feature() {
        let result = create_engine(
            Backend::Candle,
            Path::new("/dev/null"),
            Path::new("/dev/null"),
            GenerationConfig::default(),
            ServerOptions::default(),
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("candle backend requested"),
            "should mention candle, got: {err}"
        );
        assert!(
            err.contains("feature is not enabled"),
            "should say feature not enabled, got: {err}"
        );
    }

    // ── create_engine with invalid model file ─────────────────────────

    #[test]
    fn create_engine_llama_server_fails_with_nonexistent_model() {
        let result = create_engine(
            Backend::LlamaServer,
            Path::new("/tmp/zipcode_test_nonexistent_model.gguf"),
            Path::new("/dev/null"),
            GenerationConfig::default(),
            ServerOptions::default(),
        );
        assert!(result.is_err(), "loading a nonexistent model should fail");
    }

    #[cfg(feature = "candle")]
    #[test]
    fn create_engine_candle_fails_with_nonexistent_model() {
        let result = create_engine(
            Backend::Candle,
            Path::new("/tmp/zipcode_test_nonexistent_model.gguf"),
            Path::new("/tmp/zipcode_test_nonexistent_tokenizer.json"),
            GenerationConfig::default(),
            ServerOptions::default(),
        );
        assert!(
            result.is_err(),
            "loading a nonexistent model should fail for candle backend"
        );
    }

    // ── GenerationConfig default sanity ──────────────────────────────

    #[test]
    fn generation_config_defaults_are_reasonable() {
        let cfg = GenerationConfig::default();
        assert!(cfg.max_tokens > 0, "default max_tokens should be positive");
        assert!(cfg.temperature >= 0.0, "temperature should be non-negative");
        assert!(
            cfg.top_p > 0.0 && cfg.top_p <= 1.0,
            "top_p should be in (0, 1]"
        );
    }

    // ── ServerOptions default sanity ─────────────────────────────────

    #[test]
    fn server_options_defaults_are_valid() {
        let opts = ServerOptions::default();
        assert!(
            opts.context_size > 0,
            "default context_size should be positive"
        );
    }
}
