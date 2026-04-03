use candle_core::{Result, Tensor};

use crate::types::GenerationConfig;

pub struct Sampler {
    temperature: f64,
    top_p: f64,
    top_k: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    rng: fastrand::Rng,
}

impl Sampler {
    pub fn new(config: &GenerationConfig) -> Self {
        Self {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            repeat_penalty: config.repeat_penalty,
            repeat_last_n: config.repeat_last_n,
            rng: fastrand::Rng::new(),
        }
    }

    /// Sample a token index from logits tensor
    pub fn sample(&mut self, logits: &Tensor, past_tokens: &[u32]) -> Result<u32> {
        let logits = logits.to_dtype(candle_core::DType::F32)?.squeeze(0)?;
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;

        // Apply repeat penalty
        if self.repeat_penalty != 1.0 {
            let start = past_tokens.len().saturating_sub(self.repeat_last_n);
            for &token in &past_tokens[start..] {
                let idx = token as usize;
                if idx < logits_vec.len() {
                    if logits_vec[idx] > 0.0 {
                        logits_vec[idx] /= self.repeat_penalty;
                    } else {
                        logits_vec[idx] *= self.repeat_penalty;
                    }
                }
            }
        }

        // Apply temperature scaling (before greedy check)
        if self.temperature > 0.0 && self.temperature != 1.0 {
            for l in &mut logits_vec {
                *l /= self.temperature as f32;
            }
        }

        // If temperature is 0, use greedy
        if self.temperature == 0.0 {
            return Ok(logits_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0));
        }

        // Softmax
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits_vec.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Top-k filtering
        if self.top_k > 0 && self.top_k < probs.len() {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = indexed[self.top_k - 1].1;
            for p in &mut probs {
                if *p < threshold {
                    *p = 0.0;
                }
            }
        }

        // Top-p filtering
        if self.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cumulative = 0.0_f32;
            let mut cutoff = 0.0_f32;
            for (_, p) in &indexed {
                cumulative += p;
                if cumulative > self.top_p as f32 {
                    cutoff = *p;
                    break;
                }
            }
            for p in &mut probs {
                if *p < cutoff {
                    *p = 0.0;
                }
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum == 0.0 {
            return Ok(0);
        }
        for p in &mut probs {
            *p /= sum;
        }

        // Random sampling
        let r: f32 = self.rng.f32();
        let mut cumulative = 0.0_f32;
        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return Ok(i as u32);
            }
        }

        Ok(probs.len() as u32 - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let config = GenerationConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        // logits: [0.1, 0.9, 0.5] — index 1 is highest
        let logits = Tensor::new(&[0.1_f32, 0.9, 0.5], &candle_core::Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let token = sampler.sample(&logits, &[]).unwrap();
        assert_eq!(token, 1);
    }
}
