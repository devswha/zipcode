//! Model-name → chat-template registry.
//!
//! Loaded from `~/.zipcode/models/registry.json` (glob-keyed JSON object).
//! Missing or malformed file falls back to [`default_registry`].

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use crate::chat_template::{
    ChatMLTemplate, ChatTemplate, EmulatorTemplate, GemmaTemplate, Llama31Template,
};

/// Per-model registry entry: which template format to use and whether the
/// backend emits native JSON tool calls.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    /// Template format identifier: `"gemma_native"`, `"chatml"`, `"llama31_json"`, or `"emulator"`.
    pub tool_format: String,
    /// Whether the backend emits native, structured tool-call JSON.
    #[serde(default)]
    pub native_tool_calling: bool,
}

/// Maps glob patterns (e.g. `"qwen2.5-*"`) to [`ModelEntry`] values.
///
/// Loaded from `~/.zipcode/models/registry.json`; missing or malformed file
/// falls back to [`default_registry`].
pub struct TemplateRegistry {
    entries: HashMap<String, ModelEntry>,
}

impl TemplateRegistry {
    /// Load a registry from `path`, merging over the default entries.
    ///
    /// - **Missing file** → `default_registry()`, no error.
    /// - **Malformed JSON** → warn log + `default_registry()`.
    /// - **Valid JSON** → default entries overridden / extended by file entries.
    ///
    /// # Errors
    ///
    /// Returns an error only if the file exists but cannot be read (I/O failure).
    pub fn load_from(path: &Path) -> Result<Self> {
        let mut registry = default_registry();

        if !path.exists() {
            tracing::debug!(
                "Template registry not found at {}, using defaults",
                path.display()
            );
            return Ok(registry);
        }

        let content = std::fs::read_to_string(path)?;
        match serde_json::from_str::<HashMap<String, ModelEntry>>(&content) {
            Ok(custom) => {
                for (k, v) in custom {
                    registry.entries.insert(k, v);
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to parse template registry at {}: {e}. Using defaults.",
                    path.display()
                );
            }
        }

        Ok(registry)
    }

    /// Find the registry entry whose glob pattern matches `model_name`.
    ///
    /// Tries an exact key lookup first, then falls through to glob matching.
    /// Returns `None` if nothing matches.
    ///
    /// When multiple patterns match (e.g. `"qwen*"` and `"qwen2.5-*"` both match
    /// `"qwen2.5-7b"`), the most specific pattern wins — measured as the count
    /// of non-wildcard characters. Ties break lexicographically for stable,
    /// deterministic resolution across runs.
    pub fn resolve_for_model(&self, model_name: &str) -> Option<&ModelEntry> {
        // Exact key match
        if let Some(entry) = self.entries.get(model_name) {
            return Some(entry);
        }

        // Glob pattern match — collect all matches, then pick the most specific.
        // HashMap iteration order is non-deterministic, so we sort the matches
        // before returning to ensure the same pattern wins across runs.
        let mut hits: Vec<(&String, &ModelEntry)> = self
            .entries
            .iter()
            .filter(|(pattern, _)| match glob::Pattern::new(pattern) {
                Ok(pat) => pat.matches(model_name),
                Err(e) => {
                    tracing::debug!("Skipping malformed glob pattern '{pattern}': {e}");
                    false
                }
            })
            .collect();

        hits.sort_by(|(a, _), (b, _)| specificity(b).cmp(&specificity(a)).then_with(|| a.cmp(b)));

        hits.into_iter().next().map(|(_, entry)| entry)
    }

    /// Return the [`ChatTemplate`] implementation for `model_name`.
    ///
    /// Falls back to [`GemmaTemplate`] when no pattern matches.
    #[must_use]
    pub fn template_for_model(&self, model_name: &str) -> Box<dyn ChatTemplate> {
        let format = self
            .resolve_for_model(model_name)
            .map(|e| e.tool_format.as_str());

        match format {
            Some("chatml") => Box::new(ChatMLTemplate),
            Some("llama31_json") => Box::new(Llama31Template),
            Some("emulator") => Box::new(EmulatorTemplate),
            _ => Box::new(GemmaTemplate),
        }
    }
}

/// Specificity score for a glob pattern: count of non-wildcard characters.
///
/// Higher score → more specific. Used to disambiguate overlapping patterns
/// in [`TemplateRegistry::resolve_for_model`].
fn specificity(pattern: &str) -> usize {
    pattern
        .chars()
        .filter(|c| !matches!(c, '*' | '?' | '[' | ']'))
        .count()
}

/// Build the default [`TemplateRegistry`] with baseline model-family mappings.
#[must_use]
pub fn default_registry() -> TemplateRegistry {
    let mut entries = HashMap::new();

    for (pattern, format, native) in [
        ("gemma-4-*", "gemma_native", true),
        ("gemma-3-*", "gemma_native", true),
        ("qwen2.5-*", "chatml", true),
        ("qwen2-*", "chatml", true),
        ("llama-3.1-*", "llama31_json", true),
        ("*-emulator", "emulator", false),
    ] {
        entries.insert(
            pattern.to_string(),
            ModelEntry {
                tool_format: format.to_string(),
                native_tool_calling: native,
            },
        );
    }

    TemplateRegistry { entries }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_template_registry_load_from_missing_file_returns_default() {
        let path = std::path::Path::new("/tmp/zipcode_nonexistent_registry_xyz987.json");
        assert!(!path.exists(), "test path must not exist for this test");
        let registry = TemplateRegistry::load_from(path).unwrap();
        // Default entries should be present
        let entry = registry.resolve_for_model("gemma-4-e2b-q4.gguf");
        assert!(entry.is_some(), "default gemma-4-* entry should exist");
        assert_eq!(entry.unwrap().tool_format, "gemma_native");
    }

    #[test]
    fn test_template_registry_glob_match_specific_pattern() {
        let registry = default_registry();

        let entry = registry.resolve_for_model("qwen2.5-0.5b-instruct.gguf");
        assert!(entry.is_some(), "qwen2.5-* should match");
        assert_eq!(entry.unwrap().tool_format, "chatml");

        let entry = registry.resolve_for_model("llama-3.1-8b-instruct-q4.gguf");
        assert!(entry.is_some(), "llama-3.1-* should match");
        assert_eq!(entry.unwrap().tool_format, "llama31_json");

        let entry = registry.resolve_for_model("my-model-emulator");
        assert!(entry.is_some(), "*-emulator should match");
        assert_eq!(entry.unwrap().tool_format, "emulator");
        assert!(!entry.unwrap().native_tool_calling);
    }

    #[test]
    fn test_template_registry_fallback_to_gemma_when_no_match() {
        let registry = default_registry();
        let entry = registry.resolve_for_model("unknown-model-xyz.gguf");
        assert!(entry.is_none(), "unknown model should have no match");

        let template = registry.template_for_model("unknown-model-xyz.gguf");
        assert_eq!(
            template.name(),
            "gemma_native",
            "should fall back to GemmaTemplate"
        );
    }

    #[test]
    fn test_template_registry_custom_file_overrides_default() {
        let custom_json =
            r#"{"custom-model-*": {"tool_format": "chatml", "native_tool_calling": true}}"#;
        let path = std::env::temp_dir().join("zipcode_test_registry_override_abc.json");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(custom_json.as_bytes()).unwrap();
        }

        let registry = TemplateRegistry::load_from(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        // Custom entry should be present
        let entry = registry.resolve_for_model("custom-model-v1.gguf");
        assert!(entry.is_some(), "custom entry should be loaded");
        assert_eq!(entry.unwrap().tool_format, "chatml");

        // Default entries should still be present
        let entry = registry.resolve_for_model("gemma-4-e2b.gguf");
        assert!(entry.is_some(), "default gemma entry should survive merge");
        assert_eq!(entry.unwrap().tool_format, "gemma_native");
    }

    #[test]
    fn test_template_registry_malformed_json_falls_back_to_default() {
        let path = std::env::temp_dir().join("zipcode_test_registry_malformed_abc.json");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"this is { not: valid json {{{{").unwrap();
        }

        let registry = TemplateRegistry::load_from(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        // Should fall back to defaults
        let entry = registry.resolve_for_model("qwen2.5-0.5b-instruct.gguf");
        assert!(
            entry.is_some(),
            "defaults should be present after malformed JSON"
        );
        assert_eq!(entry.unwrap().tool_format, "chatml");
    }

    // ── Edge-case tests: resolve_for_model ─────────────────────────────

    #[test]
    fn test_resolve_exact_match_takes_precedence_over_glob() {
        // Insert an exact key that also matches a glob pattern.
        // Exact match should win because resolve_for_model checks HashMap first.
        let mut registry = default_registry();
        registry.entries.insert(
            "gemma-4-e2b-q4.gguf".to_string(),
            ModelEntry {
                tool_format: "chatml".to_string(),
                native_tool_calling: true,
            },
        );

        let entry = registry.resolve_for_model("gemma-4-e2b-q4.gguf");
        assert!(entry.is_some());
        // Exact match returns "chatml", not the glob "gemma-4-*" → "gemma_native"
        assert_eq!(entry.unwrap().tool_format, "chatml");
    }

    #[test]
    fn test_resolve_empty_string_model_name_returns_none() {
        let registry = default_registry();
        let entry = registry.resolve_for_model("");
        assert!(entry.is_none(), "empty string should not match any pattern");
    }

    #[test]
    fn test_resolve_special_characters_in_model_name() {
        let registry = default_registry();

        // Dots and dashes are common — verify they don't confuse glob matching
        let entry = registry.resolve_for_model("gemma-4-e2b.q4_k_m.gguf");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().tool_format, "gemma_native");

        // Spaces should NOT match the glob patterns
        let entry = registry.resolve_for_model("gemma 4 something");
        assert!(entry.is_none(), "spaces should not match gemma-4-* glob");
    }

    #[test]
    fn test_resolve_suffix_glob_matches_correctly() {
        let registry = default_registry();

        // "*-emulator" is a suffix glob
        let entry = registry.resolve_for_model("test-emulator");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().tool_format, "emulator");
        assert!(!entry.unwrap().native_tool_calling);

        // Partial suffix should not match
        let entry = registry.resolve_for_model("emulator");
        assert!(
            entry.is_none(),
            "'emulator' alone should not match '*-emulator'"
        );

        // Prefix "emulator-" should not match "*-emulator"
        let entry = registry.resolve_for_model("emulator-test");
        assert!(entry.is_none(), "prefix should not match suffix glob");
    }

    #[test]
    fn test_resolve_gemma3_matches_glob() {
        let registry = default_registry();
        let entry = registry.resolve_for_model("gemma-3-4b-it.gguf");
        assert!(entry.is_some(), "gemma-3-* should match");
        assert_eq!(entry.unwrap().tool_format, "gemma_native");
        assert!(entry.unwrap().native_tool_calling);
    }

    #[test]
    fn test_resolve_qwen2_without_dot5_matches() {
        let registry = default_registry();
        // "qwen2-*" pattern (without .5)
        let entry = registry.resolve_for_model("qwen2-72b-instruct.gguf");
        assert!(entry.is_some(), "qwen2-* should match");
        assert_eq!(entry.unwrap().tool_format, "chatml");
    }

    // ── Edge-case tests: load_from ────────────────────────────────────

    #[test]
    fn test_load_from_empty_json_object() {
        // Empty JSON object should parse but add zero custom entries.
        // Defaults should survive.
        let path = std::env::temp_dir().join("zipcode_test_registry_empty_obj.json");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"{}").unwrap();
        }

        let registry = TemplateRegistry::load_from(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let entry = registry.resolve_for_model("gemma-4-e2b.gguf");
        assert!(entry.is_some(), "defaults should survive empty JSON");
        assert_eq!(entry.unwrap().tool_format, "gemma_native");
    }

    #[test]
    fn test_load_from_valid_json_partial_deserialization() {
        // JSON with an entry missing the optional `native_tool_calling` field.
        // serde(default) should set it to false.
        let json = r#"{"my-model": {"tool_format": "chatml"}}"#;
        let path = std::env::temp_dir().join("zipcode_test_registry_partial.json");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(json.as_bytes()).unwrap();
        }

        let registry = TemplateRegistry::load_from(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let entry = registry.resolve_for_model("my-model");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().tool_format, "chatml");
        assert!(
            !entry.unwrap().native_tool_calling,
            "missing native_tool_calling should default to false"
        );
    }

    #[test]
    fn test_load_from_overlapping_keys_override_defaults() {
        // A custom file that specifies a key overlapping with a default key
        // should override the default value.
        let json = r#"{"gemma-4-*": {"tool_format": "chatml", "native_tool_calling": false}}"#;
        let path = std::env::temp_dir().join("zipcode_test_registry_overlap.json");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(json.as_bytes()).unwrap();
        }

        let registry = TemplateRegistry::load_from(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let entry = registry.resolve_for_model("gemma-4-e2b.gguf");
        assert!(entry.is_some());
        assert_eq!(
            entry.unwrap().tool_format,
            "chatml",
            "custom entry should override default gemma-4-*"
        );
        assert!(
            !entry.unwrap().native_tool_calling,
            "override should set native_tool_calling to false"
        );
    }

    #[test]
    fn test_load_from_unreadable_file_returns_error() {
        // A path that exists but is a directory (not a file) should error.
        let dir = std::env::temp_dir().join("zipcode_test_registry_dir_not_file");
        let _ = std::fs::create_dir_all(&dir);

        let result = TemplateRegistry::load_from(&dir);
        assert!(
            result.is_err(),
            "reading a directory should return an error"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_from_multiple_custom_entries() {
        // Multiple entries in one file should all be loaded.
        let json = r#"{
            "model-a": {"tool_format": "chatml", "native_tool_calling": true},
            "model-b": {"tool_format": "emulator", "native_tool_calling": false}
        }"#;
        let path = std::env::temp_dir().join("zipcode_test_registry_multi.json");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(json.as_bytes()).unwrap();
        }

        let registry = TemplateRegistry::load_from(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let entry_a = registry.resolve_for_model("model-a");
        assert!(entry_a.is_some());
        assert_eq!(entry_a.unwrap().tool_format, "chatml");

        let entry_b = registry.resolve_for_model("model-b");
        assert!(entry_b.is_some());
        assert_eq!(entry_b.unwrap().tool_format, "emulator");
        assert!(!entry_b.unwrap().native_tool_calling);

        // Defaults should still exist
        let entry_default = registry.resolve_for_model("qwen2.5-test.gguf");
        assert!(
            entry_default.is_some(),
            "defaults should survive multi-entry load"
        );
    }

    // ── Edge-case tests: template_for_model ───────────────────────────

    #[test]
    fn test_template_for_model_returns_chatml_for_qwen() {
        let registry = default_registry();
        let template = registry.template_for_model("qwen2.5-7b.gguf");
        assert_eq!(template.name(), "chatml");
        assert!(template.native_tool_calling());
    }

    #[test]
    fn test_template_for_model_returns_llama31_for_llama() {
        let registry = default_registry();
        let template = registry.template_for_model("llama-3.1-70b.gguf");
        assert_eq!(template.name(), "llama31_json");
        assert!(template.native_tool_calling());
    }

    #[test]
    fn test_template_for_model_returns_emulator_for_suffix_match() {
        let registry = default_registry();
        let template = registry.template_for_model("anything-emulator");
        assert_eq!(template.name(), "emulator");
        assert!(!template.native_tool_calling());
    }

    // ── Edge-case tests: default_registry ─────────────────────────────

    #[test]
    fn test_default_registry_contains_six_entries() {
        let registry = default_registry();
        assert_eq!(
            registry.entries.len(),
            6,
            "default registry should have exactly 6 entries"
        );
    }

    // ── Regression test: #140 deterministic glob resolution ────────────

    #[test]
    fn test_overlapping_globs_resolve_to_most_specific_pattern() {
        // Regression for #140: when two glob patterns both match a model name,
        // the more specific pattern (more non-wildcard chars) must win on
        // every run. HashMap iteration order is otherwise non-deterministic.
        let custom_json = r#"{
            "qwen*": {"tool_format": "chatml", "native_tool_calling": false},
            "qwen2.5-*": {"tool_format": "llama31_json", "native_tool_calling": true}
        }"#;
        let path = std::env::temp_dir().join("zipcode_test_registry_overlap_140.json");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(custom_json.as_bytes()).unwrap();
        }
        let registry = TemplateRegistry::load_from(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        // Run resolution many times — every call must return the more
        // specific pattern's entry, regardless of HashMap bucket ordering.
        for _ in 0..50 {
            let entry = registry.resolve_for_model("qwen2.5-7b").unwrap();
            assert_eq!(
                entry.tool_format, "llama31_json",
                "more specific 'qwen2.5-*' must win over 'qwen*'"
            );
            assert!(entry.native_tool_calling);
        }
    }

    #[test]
    fn test_specificity_score_counts_non_wildcard_chars() {
        assert_eq!(specificity("qwen*"), 4);
        assert_eq!(specificity("qwen2.5-*"), 8);
        assert_eq!(specificity("*-emulator"), 9);
        // `[`/`]` themselves are stripped, but their contents still count.
        assert_eq!(specificity("[abc]xyz"), 6);
        assert_eq!(specificity("?"), 0);
    }

    #[test]
    fn test_default_registry_all_native_calling_except_emulator() {
        let registry = default_registry();

        for (pattern, entry) in &registry.entries {
            if pattern == "*-emulator" {
                assert!(
                    !entry.native_tool_calling,
                    "emulator entry should have native_tool_calling=false"
                );
            } else {
                assert!(
                    entry.native_tool_calling,
                    "non-emulator entry '{pattern}' should have native_tool_calling=true"
                );
            }
        }
    }
}
