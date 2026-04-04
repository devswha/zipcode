//! llama-cpp-2 inference backend for GGUF models (including Gemma 4).
//!
//! This backend requires the `llama-cpp` feature flag and cmake/a C++ compiler
//! to build llama.cpp's native library. Enable it with:
//!
//!   cargo build --features llama-cpp --no-default-features
//!
//! or to use alongside candle:
//!
//!   cargo build --features candle,llama-cpp

use std::path::Path;
use std::sync::mpsc;

use anyhow::{Context, Result};
use tracing::info;

use crate::chat_template::{self, ToolSpec};
use crate::types::{ChatMessage, FinishReason, GenerationConfig, InferenceError, TokenEvent};
use crate::InferenceProvider;

use llama_cpp::context::params::LlamaContextParams;
use llama_cpp::llama_backend::LlamaBackend;
use llama_cpp::llama_batch::LlamaBatch;
use llama_cpp::model::params::LlamaModelParams;
use llama_cpp::model::{AddBos, LlamaModel};
use llama_cpp::sampling::LlamaSampler;

pub struct LlamaCppProvider {
    model: LlamaModel,
    backend: LlamaBackend,
    config: GenerationConfig,
}

impl LlamaCppProvider {
    /// Load a GGUF model file using llama.cpp.
    pub fn load(model_path: &Path) -> Result<Self> {
        info!("Loading llama-cpp model from {}", model_path.display());

        let backend = LlamaBackend::init().map_err(|e| {
            if matches!(e, llama_cpp::LlamaCppError::BackendAlreadyInitialized) {
                anyhow::anyhow!("llama backend already initialized in this process")
            } else {
                anyhow::anyhow!("Failed to init llama backend: {e}")
            }
        })?;

        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| format!("Failed to load GGUF model: {}", model_path.display()))?;

        Ok(Self {
            model,
            backend,
            config: GenerationConfig::default(),
        })
    }

    pub fn set_config(&mut self, config: GenerationConfig) {
        self.config = config;
    }

    /// Run autoregressive generation, streaming tokens via the returned receiver.
    pub fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel();

        // Format the conversation into a single prompt string using our Gemma 4 template
        let prompt = chat_template::format_conversation(messages, tools);

        // Tokenize
        let tokens = match self.model.str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.send(TokenEvent::Error(InferenceError::TokenizerError(
                    e.to_string(),
                )));
                return rx;
            }
        };

        let n_prompt = tokens.len();

        // Create context
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(
            u32::try_from(n_prompt + self.config.max_tokens + 64).unwrap_or(4096),
        ));

        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                    e.to_string(),
                )));
                return rx;
            }
        };

        // Build sampler chain: top_k -> top_p -> temp -> dist
        let top_k = i32::try_from(self.config.top_k).unwrap_or(40);
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(top_k),
            LlamaSampler::top_p(self.config.top_p as f32, 1),
            LlamaSampler::temp(self.config.temperature as f32),
            LlamaSampler::dist(fastrand::u32(..)),
        ]);

        // Feed the prompt in one batch
        let batch_size = n_prompt + self.config.max_tokens + 64;
        let mut batch = LlamaBatch::new(batch_size, 1);

        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == n_prompt - 1;
            if let Err(e) = batch.add(*token, i32::try_from(i).unwrap_or(0), &[0], is_last) {
                let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                    e.to_string(),
                )));
                return rx;
            }
        }

        if let Err(e) = ctx.decode(&mut batch) {
            let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                e.to_string(),
            )));
            return rx;
        }

        // Autoregressive generation loop
        let mut generated_text = String::new();
        let mut n_cur = n_prompt;
        let mut finished = false;
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        for _ in 0..self.config.max_tokens {
            // Sample next token
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            // Accept into sampler state
            sampler.accept(token);

            // Check for end-of-generation
            if self.model.is_eog_token(token) {
                let tool_calls = chat_template::parse_tool_calls(&generated_text);
                if !tool_calls.is_empty() {
                    for call in tool_calls {
                        let _ = tx.send(TokenEvent::ToolCall(call));
                    }
                    let _ = tx.send(TokenEvent::Done(FinishReason::ToolUse));
                } else {
                    let _ = tx.send(TokenEvent::Done(FinishReason::Stop));
                }
                finished = true;
                break;
            }

            // Decode token to text piece
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .unwrap_or_default();
            if !piece.is_empty() {
                generated_text.push_str(&piece);
                let _ = tx.send(TokenEvent::Token(piece));
            }

            // Prepare next batch with the single new token
            batch.clear();
            if let Err(e) = batch.add(token, i32::try_from(n_cur).unwrap_or(i32::MAX), &[0], true) {
                let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                    e.to_string(),
                )));
                break;
            }

            if let Err(e) = ctx.decode(&mut batch) {
                let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                    e.to_string(),
                )));
                break;
            }

            n_cur += 1;
        }

        if !finished {
            let _ = tx.send(TokenEvent::Done(FinishReason::MaxTokens));
        }

        rx
    }
}

impl InferenceProvider for LlamaCppProvider {
    fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent> {
        self.generate_stream(messages, tools)
    }
}
