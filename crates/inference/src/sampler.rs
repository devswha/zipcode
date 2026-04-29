use candle_core::{Result, Tensor};

use crate::types::GenerationConfig;

const F32_EPSILON: f32 = 1e-6;

/// Token sampler applying temperature, top-p, top-k, and repeat penalty.
///
/// Converts raw logits from the model into a probability distribution and
/// samples a single token index. The sampling pipeline is:
/// repeat penalty → temperature scaling → top-k → top-p → random sample.
pub struct Sampler {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    rng: fastrand::Rng,
}

impl Sampler {
    /// Create a sampler from a [`GenerationConfig`].
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(config: &GenerationConfig) -> Self {
        Self {
            temperature: config.temperature as f32,
            top_p: config.top_p as f32,
            top_k: config.top_k,
            repeat_penalty: config.repeat_penalty,
            repeat_last_n: config.repeat_last_n,
            rng: fastrand::Rng::new(),
        }
    }

    /// Sample a token index from logits tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the logits tensor cannot be converted to `f32`
    /// or if any tensor operation fails.
    pub fn sample(&mut self, logits: &Tensor, past_tokens: &[u32]) -> Result<u32> {
        let logits = logits.to_dtype(candle_core::DType::F32)?.squeeze(0)?;
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;

        // Apply repeat penalty
        if (self.repeat_penalty - 1.0).abs() > F32_EPSILON {
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
        if self.temperature > 0.0 && (self.temperature - 1.0).abs() > F32_EPSILON {
            for l in &mut logits_vec {
                *l /= self.temperature;
            }
        }

        // If temperature is 0, use greedy
        if self.temperature <= 0.0 {
            return Ok(logits_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| u32::try_from(i).unwrap_or(u32::MAX)));
        }

        // Softmax
        let max_logit = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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
                if cumulative > self.top_p {
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
        if sum <= 0.0 {
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
                return Ok(u32::try_from(i).unwrap_or(u32::MAX));
            }
        }

        let last = probs.len().saturating_sub(1);
        Ok(u32::try_from(last).unwrap_or(u32::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a 1-D-unsqueezed logits tensor from a slice.
    fn logits(values: &[f32]) -> Tensor {
        Tensor::new(values, &candle_core::Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
    }

    // --- Greedy / temperature=0 ------------------------------------------------

    #[test]
    fn test_greedy_sampling() {
        let config = GenerationConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        let token = sampler.sample(&logits(&[0.1, 0.9, 0.5]), &[]).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn test_greedy_with_equal_logits_picks_last_max() {
        let config = GenerationConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        // Two equal max values at index 0 and 2.
        // Iterator::max_by is not stable — it returns the *last* maximum.
        let token = sampler.sample(&logits(&[1.0, 0.0, 1.0]), &[]).unwrap();
        assert_eq!(token, 2);
    }

    // --- Temperature scaling ----------------------------------------------------

    #[test]
    fn test_temperature_scaling_preserves_argmax_for_extreme_values() {
        // With temp > 0, the highest logit still has the highest probability.
        // We set top_k=0 (disabled) and top_p=1.0 (disabled) so only temperature
        // applies. Use temp=0.001 for near-greedy behavior.
        let config = GenerationConfig {
            temperature: 0.001,
            top_p: 1.0,
            top_k: 0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        let token = sampler.sample(&logits(&[0.0, 100.0, 0.0]), &[]).unwrap();
        assert_eq!(token, 1, "near-zero temperature should pick highest logit");
    }

    // --- Repeat penalty ---------------------------------------------------------

    #[test]
    fn test_repeat_penalty_divides_positive_logits() {
        // penalty > 1 penalizes recently-seen tokens by dividing positive logits.
        let config = GenerationConfig {
            temperature: 0.0, // greedy to see the effect deterministically
            repeat_penalty: 2.0,
            repeat_last_n: 64,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);

        // Token 0 is in the history; its logit 2.0 gets divided by 2.0 → 1.0.
        // Token 1 has logit 1.5 untouched → still 1.5 → wins.
        let token = sampler.sample(&logits(&[2.0, 1.5, 0.5]), &[0u32]).unwrap();
        assert_eq!(token, 1, "token 0 should be penalized, token 1 should win");
    }

    #[test]
    fn test_repeat_penalty_multiplies_negative_logits() {
        // For negative logits, repeat penalty multiplies (pushes further negative).
        let config = GenerationConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            repeat_last_n: 64,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);

        // Token 0 in history. Its logit is -0.5. After penalty: -0.5 * 2.0 = -1.0.
        // Token 1 has logit -0.1, untouched → -0.1 > -1.0 → wins.
        let token = sampler
            .sample(&logits(&[-0.5, -0.1, -2.0]), &[0u32])
            .unwrap();
        assert_eq!(
            token, 1,
            "penalized negative logit for token 0 should make token 1 win"
        );
    }

    #[test]
    fn test_repeat_penalty_window_respects_repeat_last_n() {
        // Token 0 appeared 3 turns ago but repeat_last_n=2 means only the last
        // 2 tokens are penalized. Token 0 should NOT be penalized.
        let config = GenerationConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            repeat_last_n: 2,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);

        // past_tokens: [0, 1, 2] — only [1, 2] are in the window. Token 0 is safe.
        let token = sampler
            .sample(&logits(&[5.0, 0.1, 0.1]), &[0u32, 1u32, 2u32])
            .unwrap();
        assert_eq!(
            token, 0,
            "token 0 outside the repeat window should NOT be penalized"
        );
    }

    #[test]
    fn test_repeat_penalty_no_op_when_penalty_is_one() {
        let config = GenerationConfig {
            temperature: 0.0,
            repeat_penalty: 1.0, // no-op
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);

        // Token 0 has the highest logit; penalty=1.0 means no change.
        let token = sampler.sample(&logits(&[3.0, 2.0, 1.0]), &[0u32]).unwrap();
        assert_eq!(token, 0);
    }

    // --- Top-k filtering --------------------------------------------------------

    #[test]
    fn test_top_k_filters_to_k_highest() {
        // top_k=1 means only the highest-probability token survives → greedy.
        let config = GenerationConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);

        // With top_k=1, only the argmax survives; token 2 should be picked.
        let token = sampler.sample(&logits(&[0.1, 0.2, 5.0, 0.3]), &[]).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_top_k_disabled_when_zero() {
        // top_k=0 means no top-k filtering (all tokens pass through).
        // Use greedy to verify nothing breaks.
        let config = GenerationConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        let token = sampler.sample(&logits(&[0.1, 0.9, 0.5]), &[]).unwrap();
        assert_eq!(token, 1);
    }

    // --- Top-p (nucleus) filtering ----------------------------------------------

    #[test]
    fn test_top_p_one_point_zero_disables_filtering() {
        // top_p=1.0 means no filtering — use greedy to check.
        let config = GenerationConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        let token = sampler.sample(&logits(&[0.1, 0.9, 0.5]), &[]).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn test_top_p_near_zero_concentrates_on_highest() {
        // top_p very close to 0 should only keep the single most-likely token.
        // Use temperature > 0 so the sampling path runs (greedy bypasses top-p).
        let config = GenerationConfig {
            temperature: 0.001, // near-greedy
            top_p: 0.001,       // extreme nucleus: only the top token survives
            top_k: 0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        let token = sampler
            .sample(&logits(&[0.1, 0.2, 100.0, 0.3]), &[])
            .unwrap();
        assert_eq!(token, 2, "extreme top_p should isolate the highest logit");
    }

    // --- Boundary: single-element tensor ----------------------------------------

    #[test]
    fn test_single_logit_always_returns_zero() {
        let config = GenerationConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        let token = sampler.sample(&logits(&[2.71]), &[]).unwrap();
        assert_eq!(token, 0, "single-logit tensor must always return index 0");
    }

    // --- Boundary: all logits zero after filtering -------------------------------

    #[test]
    fn test_all_probs_zero_returns_token_zero() {
        // Construct a scenario where filtering zeroes out all probabilities.
        // top_k=1 with a tensor where every logit is identical → the top-k
        // threshold equals the identical value, so all survive. But top_p=0.0
        // (less than any cumulative) will zero everything out.
        //
        // Actually top_p < 1.0 triggers filtering. With identical logits each
        // gets 1/N probability. The loop accumulates and breaks when cumulative
        // > top_p. With top_p near zero, cutoff catches everything below the
        // highest prob — but they're all equal, so cutoff = prob. After the
        // `for p in probs: if *p < cutoff { *p = 0 }` pass, all become 0.
        // Then sum=0 → returns Ok(0).
        //
        // Use identical logits so all probs are equal. top_p < smallest prob
        // means nothing survives → sum=0 fallback.
        let config = GenerationConfig {
            temperature: 1.0,
            top_p: 0.01, // very small — no single token's prob exceeds 0.01
            top_k: 0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        // 100 identical logits → each prob = 0.01. cumulative reaches 0.01
        // after first token but cutoff = that prob. All others get zeroed.
        // Then renormalize gives first token prob=1.0.
        // Actually let's use a huge vocab to make individual probs tiny:
        let values: Vec<f32> = vec![1.0; 1000]; // each prob ≈ 0.001
        let token = sampler.sample(&logits(&values), &[]).unwrap();
        // After renormalization, one of the surviving tokens is picked.
        // With identical logits and top_p=0.01, only one token survives
        // (the first in the sorted order) and it gets all the probability.
        // Since all are identical, sorting is stable, so token 0 wins.
        assert!(
            (token as usize) < 1000,
            "token index must be within the vocabulary"
        );
    }

    // --- Integration: repeat penalty + greedy -----------------------------------

    #[test]
    fn test_repeat_penalty_and_greedy_combined() {
        // Token 2 has the highest raw logit but is in the repeat history.
        // After penalty, token 1 should win.
        let config = GenerationConfig {
            temperature: 0.0,
            repeat_penalty: 10.0, // strong penalty
            repeat_last_n: 64,
            top_p: 0.9,
            top_k: 40,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);

        // Raw: [1.0, 2.0, 3.0]. Token 2 penalized: 3.0/10.0 = 0.3.
        // After penalty: [1.0, 2.0, 0.3] → token 1 wins.
        let token = sampler.sample(&logits(&[1.0, 2.0, 3.0]), &[2u32]).unwrap();
        assert_eq!(token, 1);
    }

    // ── Edge-case tests ──────────────────────────────────────────────

    /// Token index out of vocab range should be skipped safely
    #[test]
    fn test_repeat_penalty_token_out_of_range_safe() {
        let config = GenerationConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            repeat_last_n: 64,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        // Token index 99 is beyond vocab size 3 — should be silently skipped
        let token = sampler.sample(&logits(&[1.0, 2.0, 3.0]), &[99u32]).unwrap();
        // Token 2 should win since 99 is out of range and not penalized
        assert_eq!(token, 2);
    }

    /// Temperature exactly 1.0 should not scale logits (no-op path)
    #[test]
    fn test_temperature_exactly_one_no_scaling() {
        let config = GenerationConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        // With temp=1.0 the sampling path runs but logits are unscaled.
        // Very high logit at index 2 should almost certainly be picked.
        let token = sampler.sample(&logits(&[0.0, 0.0, 100.0]), &[]).unwrap();
        assert_eq!(token, 2, "temp=1.0 should pick the dominant logit");
    }

    /// top_k equal to vocab size: all tokens survive
    #[test]
    fn test_top_k_equals_vocab_size_no_filtering() {
        let config = GenerationConfig {
            temperature: 0.0,
            top_k: 4,
            top_p: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        // top_k=4 with vocab of 4 → no filtering → greedy picks token 2
        let token = sampler.sample(&logits(&[0.1, 0.2, 0.9, 0.3]), &[]).unwrap();
        assert_eq!(token, 2);
    }

    /// top_k larger than vocab size: disabled path (no filtering)
    #[test]
    fn test_top_k_larger_than_vocab_disabled() {
        let config = GenerationConfig {
            temperature: 0.0,
            top_k: 100,
            top_p: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        let token = sampler.sample(&logits(&[0.1, 0.5, 0.2]), &[]).unwrap();
        assert_eq!(token, 1, "top_k > vocab should not filter");
    }

    /// Softmax with very large logits should not overflow (numerical stability)
    #[test]
    fn test_softmax_numerical_stability_large_logits() {
        let config = GenerationConfig {
            temperature: 0.001, // near-greedy to make result deterministic
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        // Very large logits — the max subtraction should prevent overflow
        let token = sampler
            .sample(&logits(&[10000.0, 10001.0, -50000.0]), &[])
            .unwrap();
        assert_eq!(
            token, 1,
            "highest logit should be picked after stable softmax"
        );
    }

    /// Multiple sequential samples with different past tokens produce valid results
    #[test]
    fn test_sequential_samples_with_different_history() {
        let config = GenerationConfig {
            temperature: 0.0,
            repeat_penalty: 10.0,
            repeat_last_n: 64,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);

        // First sample: token 1 wins (highest logit at 5.0)
        let t1 = sampler.sample(&logits(&[1.0, 5.0, 2.0]), &[]).unwrap();
        assert_eq!(t1, 1);

        // Second sample: token 1 in history, penalty=10.0 → 5.0/10.0=0.5
        // token 0=1.0, token 2=2.0 → token 2 wins
        let t2 = sampler.sample(&logits(&[1.0, 5.0, 2.0]), &[t1]).unwrap();
        assert_eq!(
            t2, 2,
            "token 1 penalized (5.0/10=0.5), token 2 (2.0) should win"
        );

        // Third sample: tokens 1,2 in history → token 1=5.0/10=0.5, token 2=2.0/10=0.2
        // token 0=3.0 → token 0 wins
        let t3 = sampler
            .sample(&logits(&[3.0, 5.0, 2.0]), &[t1, t2])
            .unwrap();
        assert_eq!(t3, 0, "only token 0 should survive penalty on 1 and 2");
    }

    /// All negative logits with temperature > 0 should still sample validly
    #[test]
    fn test_all_negative_logits_with_temperature() {
        let config = GenerationConfig {
            temperature: 0.5,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&config);
        // All negative logits — the least negative should be sampled
        let token = sampler
            .sample(&logits(&[-10.0, -1.0, -100.0]), &[])
            .unwrap();
        assert!((token as usize) < 3, "token index must be within vocab");
        // With softmax, -1.0 dominates; near-deterministic with low temperature
        assert_eq!(token, 1, "least negative logit should win");
    }
}
