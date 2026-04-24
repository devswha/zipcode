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
    pub fn resolve_for_model(&self, model_name: &str) -> Option<&ModelEntry> {
        // Exact key match
        if let Some(entry) = self.entries.get(model_name) {
            return Some(entry);
        }

        // Glob pattern match
        for (pattern, entry) in &self.entries {
            match glob::Pattern::new(pattern) {
                Ok(pat) if pat.matches(model_name) => return Some(entry),
                Ok(_) => {}
                Err(e) => {
                    tracing::debug!("Skipping malformed glob pattern '{pattern}': {e}");
                }
            }
        }

        None
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
}
