// TODO: candle-transformers 0.8 does not include quantized_gemma2.
// Using quantized_llama as a stand-in — both share the same GGUF ModelWeights
// interface (from_gguf + forward). Swap this import when a Gemma 4 GGUF
// backend is available in candle-transformers.
use std::path::Path;
use std::sync::mpsc;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama as gemma;
use tokenizers::Tokenizer;
use tracing::info;

use crate::chat_template::{self, ToolSpec};
use crate::sampler::Sampler;
use crate::types::{ChatMessage, FinishReason, GenerationConfig, InferenceError, TokenEvent};

pub struct InferenceEngine {
    model: gemma::ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    config: GenerationConfig,
}

impl InferenceEngine {
    /// Load a GGUF model from disk.
    pub fn load(model_path: &Path, tokenizer_path: &Path, device: Device) -> Result<Self> {
        info!("Loading model from {}", model_path.display());

        let mut file = std::fs::File::open(model_path)
            .with_context(|| format!("Model file not found: {}", model_path.display()))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .context("Failed to parse GGUF file")?;

        let model = gemma::ModelWeights::from_gguf(content, &mut file, &device)
            .context("Failed to load model weights from GGUF")?;

        info!("Loading tokenizer from {}", tokenizer_path.display());
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config: GenerationConfig::default(),
        })
    }

    pub fn set_config(&mut self, config: GenerationConfig) {
        self.config = config;
    }

    /// Generate tokens, streaming them via an `mpsc::Receiver<TokenEvent>`.
    ///
    /// The returned receiver yields `Token` events for each decoded piece,
    /// followed by either `ToolCall` + `Done(ToolUse)` when the model emits
    /// tool calls, or `Done(Stop)` / `Done(MaxTokens)` otherwise.
    pub fn generate_stream(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
    ) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel();

        let prompt = chat_template::format_conversation(messages, tools);

        // Tokenize prompt
        let tokens = match self.tokenizer.encode(prompt.as_str(), true) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                let _ = tx.send(TokenEvent::Error(InferenceError::TokenizerError(
                    e.to_string(),
                )));
                return rx;
            }
        };

        let mut sampler = Sampler::new(&self.config);
        let mut all_tokens: Vec<u32> = tokens.clone();
        let mut generated_text = String::new();

        // Feed the full prompt through the model to build KV cache
        let input = match Tensor::new(tokens.as_slice(), &self.device).and_then(|t| t.unsqueeze(0))
        {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                    e.to_string(),
                )));
                return rx;
            }
        };

        let mut logits = match self.model.forward(&input, 0) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                    e.to_string(),
                )));
                return rx;
            }
        };

        // Resolve stop token IDs
        let eos_token = self.tokenizer.token_to_id("<eos>").unwrap_or(1);
        let end_of_turn = self
            .tokenizer
            .token_to_id("<end_of_turn>")
            .unwrap_or(eos_token);

        // Autoregressive generation loop
        let mut finished = false;
        for i in 0..self.config.max_tokens {
            // Extract logits for the last position
            let next_logits = if logits.dims().len() == 3 {
                logits.squeeze(0).and_then(|t| {
                    let seq_len = t.dim(0).unwrap_or(1);
                    t.narrow(0, seq_len - 1, 1)
                })
            } else {
                let seq_len = logits.dim(0).unwrap_or(1);
                logits.narrow(0, seq_len - 1, 1)
            };

            let next_logits = match next_logits {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                        e.to_string(),
                    )));
                    break;
                }
            };

            let token = match sampler.sample(&next_logits, &all_tokens) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                        e.to_string(),
                    )));
                    break;
                }
            };

            // Check for stop tokens (EOS or <end_of_turn>)
            if token == eos_token || token == end_of_turn {
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

            all_tokens.push(token);

            // Decode token to text and stream
            if let Ok(text) = self.tokenizer.decode(&[token], false) {
                generated_text.push_str(&text);
                let _ = tx.send(TokenEvent::Token(text));
            }

            // Prepare single-token input for next step
            let next_input = match Tensor::new(&[token], &self.device).and_then(|t| t.unsqueeze(0))
            {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                        e.to_string(),
                    )));
                    break;
                }
            };

            logits = match self.model.forward(&next_input, tokens.len() + i) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(TokenEvent::Error(InferenceError::GenerationError(
                        e.to_string(),
                    )));
                    break;
                }
            };
        }

        // If we exhausted max_tokens without hitting a stop token, send MaxTokens.
        if !finished {
            let _ = tx.send(TokenEvent::Done(FinishReason::MaxTokens));
        }

        rx
    }
}
