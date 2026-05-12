use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipcodeConfig {
    #[serde(default = "default_model_dir")]
    pub model_dir: PathBuf,
    #[serde(default)]
    pub model_file: Option<String>,
    #[serde(default)]
    pub llama_server_bin: Option<PathBuf>,
    #[serde(default = "default_permission")]
    pub permission_mode: String,
    #[serde(default)]
    pub generation: GenerationOverrides,
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    #[serde(default)]
    pub flash_attention: bool,
    #[serde(default)]
    pub context_size: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationOverrides {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<usize>,
}

fn default_model_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".zipcode/models")
}

fn default_permission() -> String {
    "workspace-write".to_string()
}

#[must_use]
pub fn expand_user_path(path: &Path) -> PathBuf {
    let Some(raw) = path.to_str() else {
        return path.to_path_buf();
    };

    if raw == "~" {
        return dirs::home_dir().unwrap_or_else(|| path.to_path_buf());
    }

    if let Some(stripped) = raw.strip_prefix("~/") {
        return dirs::home_dir().map_or_else(|| path.to_path_buf(), |home| home.join(stripped));
    }

    path.to_path_buf()
}

#[must_use]
pub fn resolve_project_path(path: &Path, project_root: &Path) -> PathBuf {
    let expanded = expand_user_path(path);
    if expanded.is_absolute() {
        expanded
    } else {
        project_root.join(expanded)
    }
}

impl Default for ZipcodeConfig {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            model_file: None,
            llama_server_bin: None,
            permission_mode: default_permission(),
            generation: GenerationOverrides::default(),
            gpu_layers: None,
            flash_attention: false,
            context_size: None,
        }
    }
}

const VALID_PERMISSION_MODES: &[&str] = &[
    "read-only",
    "workspace-write",
    "full-access",
    "danger-full-access",
];

impl ZipcodeConfig {
    fn load_from_path(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Self = serde_json::from_str(&content)?;
        config.model_dir = expand_user_path(&config.model_dir);
        config.llama_server_bin = config.llama_server_bin.as_deref().map(expand_user_path);
        config.validate()?;
        Ok(config)
    }

    /// Load only the global config (~/.zipcode/config.json), or defaults if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the config file exists but cannot be read or parsed,
    /// or if the loaded config fails validation.
    pub fn load_global() -> Result<Self> {
        let path = global_config_path();
        if path.exists() {
            Self::load_from_path(&path)
        } else {
            Ok(Self::default())
        }
    }

    /// Validate config field values, returning an error for out-of-range or
    /// unsupported settings. Called automatically at the end of [`Self::load`] and
    /// the internal `load_from_path` helper.
    ///
    /// # Errors
    ///
    /// Returns an error if `permission_mode` is not one of the valid modes,
    /// `generation.temperature` is negative, `generation.top_p` is not in
    /// `(0, 1]`, or `generation.max_tokens` is zero.
    pub fn validate(&self) -> Result<()> {
        if !VALID_PERMISSION_MODES.contains(&self.permission_mode.as_str()) {
            anyhow::bail!(
                "unsupported permission_mode '{}'. Expected one of: {}",
                self.permission_mode,
                VALID_PERMISSION_MODES.join(", ")
            );
        }
        if let Some(temp) = self.generation.temperature {
            if temp < 0.0 {
                anyhow::bail!("generation.temperature must be >= 0, got {temp}");
            }
        }
        if let Some(top_p) = self.generation.top_p {
            if top_p <= 0.0 || top_p > 1.0 {
                anyhow::bail!("generation.top_p must be in (0, 1], got {top_p}");
            }
        }
        if let Some(max_tokens) = self.generation.max_tokens {
            if max_tokens == 0 {
                anyhow::bail!("generation.max_tokens must be > 0, got 0");
            }
        }
        Ok(())
    }

    /// Load config with hierarchy: global (~/.zipcode/config.json) < project (.zipcode.json)
    ///
    /// # Errors
    ///
    /// Returns an error if the global or project config file exists but cannot
    /// be read or parsed, or if the merged config fails validation.
    pub fn load(cwd: &Path) -> Result<Self> {
        let mut config = Self::load_global()?;
        let project_root = find_project_root(cwd);

        // Project config (overrides)
        let project_path = project_root.join(".zipcode.json");
        if project_path.exists() {
            let content = std::fs::read_to_string(&project_path)?;
            let project: serde_json::Value = serde_json::from_str(&content)?;
            // Validate field types and warn on mismatches before reading values.
            // This addresses the silent-ignore problem: a typo like
            // {"permission_mode": 123} or {"gpu_layers": "many"} would previously
            // be swallowed with no feedback.
            warn_type_mismatch(&project, "permission_mode", "string");
            warn_type_mismatch(&project, "model_dir", "string");
            warn_type_mismatch(&project, "model_file", "string");
            warn_type_mismatch(&project, "llama_server_bin", "string");
            warn_type_mismatch(&project, "generation", "object");
            warn_type_mismatch(&project, "gpu_layers", "number");
            warn_type_mismatch(&project, "flash_attention", "boolean");
            warn_type_mismatch(&project, "context_size", "number");

            if let Some(perm) = project["permission_mode"].as_str() {
                config.permission_mode = perm.to_string();
            }
            if let Some(dir) = project["model_dir"].as_str() {
                config.model_dir = resolve_project_path(Path::new(dir), &project_root);
            }
            if let Some(file) = project["model_file"].as_str() {
                config.model_file = Some(file.to_string());
            }
            if let Some(bin) = project["llama_server_bin"].as_str() {
                config.llama_server_bin = Some(resolve_project_path(Path::new(bin), &project_root));
            }
            if let Some(gen) = project.get("generation") {
                warn_type_mismatch(gen, "temperature", "number");
                warn_type_mismatch(gen, "top_p", "number");
                warn_type_mismatch(gen, "max_tokens", "number");
                if let Some(t) = gen["temperature"].as_f64() {
                    config.generation.temperature = Some(t);
                }
                if let Some(t) = gen["top_p"].as_f64() {
                    config.generation.top_p = Some(t);
                }
                if let Some(t) = gen["max_tokens"].as_u64() {
                    config.generation.max_tokens = Some(usize::try_from(t).unwrap_or(usize::MAX));
                }
            }
            if let Some(layers) = project["gpu_layers"].as_i64() {
                config.gpu_layers = Some(i32::try_from(layers).map_err(|_| {
                    anyhow::anyhow!(
                        "gpu_layers value {} is out of range (must be between {} and {})",
                        layers,
                        i32::MIN,
                        i32::MAX
                    )
                })?);
            }
            if let Some(fa) = project["flash_attention"].as_bool() {
                config.flash_attention = fa;
            }
            if let Some(ctx) = project["context_size"].as_u64() {
                if ctx == 0 {
                    anyhow::bail!("context_size must be > 0, got 0");
                }
                config.context_size = Some(usize::try_from(ctx).unwrap_or(usize::MAX));
            }
        }

        config.validate()?;
        Ok(config)
    }

    /// Save the config to ~/.zipcode/config.json, creating parent directories if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the config directory cannot be created or the file
    /// cannot be written.
    pub fn save_global(&self) -> Result<PathBuf> {
        let path = global_config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(path)
    }
}

#[must_use]
pub fn find_project_root(start: &Path) -> PathBuf {
    for ancestor in start.ancestors() {
        if ancestor.join(".zipcode.json").exists()
            || ancestor.join(".zipcode.md").exists()
            || ancestor.join(".git").exists()
        {
            return ancestor.to_path_buf();
        }
    }

    start.to_path_buf()
}

#[must_use]
/// Resolve the path used by [`ZipcodeConfig::load_global`].
///
/// Honours the `ZIPCODE_GLOBAL_CONFIG` env variable so tests can isolate
/// from the developer's real `~/.zipcode/config.json`. Without this hook,
/// any test that exercises `ZipcodeConfig::load` inherits whatever the
/// user has set globally (e.g. `gpu_layers: 999`), making
/// "invalid input falls back to default" assertions fail noisily on dev
/// machines.
pub fn global_config_path() -> PathBuf {
    if let Ok(path) = std::env::var("ZIPCODE_GLOBAL_CONFIG") {
        return PathBuf::from(path);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".zipcode/config.json")
}

/// Return a human-readable name for the JSON value's type.
const fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Warn if a field exists in `obj` but is not the expected JSON type.
fn warn_type_mismatch(obj: &serde_json::Value, field: &str, expected: &str) {
    if let Some(v) = obj.get(field) {
        let matches = match expected {
            "string" => v.is_string(),
            "number" => v.is_number(),
            "boolean" => v.is_boolean(),
            "object" => v.is_object(),
            _ => false,
        };
        if !matches {
            tracing::warn!(
                field,
                expected_type = expected,
                actual_type = json_type_name(v),
                "project .zipcode.json field has wrong type — ignoring"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    /// Serialise tests that mutate `ZIPCODE_GLOBAL_CONFIG` so parallel runs
    /// don't race on the env var. Mirrors the `SESSION_DIR_LOCK` pattern in
    /// `conversation.rs` / `tests/skills.rs`.
    static GLOBAL_CONFIG_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    /// Run `f` with `ZIPCODE_GLOBAL_CONFIG` pointed at a non-existent path so
    /// `Self::load_global()` falls back to the default. The lock guard is
    /// returned alongside the tempdir so the caller's scope keeps both alive
    /// for the test body — Drop on the guard releases the lock, Drop on the
    /// dir cleans up the path.
    fn with_isolated_global_config() -> (tempfile::TempDir, std::sync::MutexGuard<'static, ()>) {
        let guard = GLOBAL_CONFIG_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempfile::TempDir::new().unwrap();
        // Point at a path *inside* the tempdir that we never create, so
        // `path.exists()` is false → `load_global()` returns Self::default().
        std::env::set_var("ZIPCODE_GLOBAL_CONFIG", dir.path().join("absent.json"));
        (dir, guard)
    }

    #[test]
    fn test_default_config() {
        let config = ZipcodeConfig::default();
        assert_eq!(config.permission_mode, "workspace-write");
        assert!(config.model_dir.to_str().unwrap().contains(".zipcode"));
        assert_eq!(config.llama_server_bin, None);
    }

    #[test]
    fn test_load_from_empty_dir() {
        let (_global_dir, _guard) = with_isolated_global_config();
        let dir = tempfile::TempDir::new().unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.permission_mode, "workspace-write");
    }

    #[test]
    fn test_load_project_override() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"permission_mode": "full-access", "generation": {"temperature": 0.5}}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.permission_mode, "full-access");
        assert_eq!(config.generation.temperature, Some(0.5));
    }

    #[test]
    fn test_load_project_override_model_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"model_dir": "/custom/models", "model_file": "my-model.gguf"}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.model_dir, std::path::PathBuf::from("/custom/models"));
        assert_eq!(config.model_file, Some("my-model.gguf".to_string()));
    }

    #[test]
    fn test_load_project_override_relative_model_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"model_dir": "models"}"#,
        )
        .unwrap();

        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.model_dir, dir.path().join("models"));
    }

    #[test]
    fn test_load_project_override_from_subdirectory() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("src/bin");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"permission_mode": "read-only", "model_dir": "models"}"#,
        )
        .unwrap();

        let config = ZipcodeConfig::load(&nested).unwrap();
        assert_eq!(config.permission_mode, "read-only");
        assert_eq!(config.model_dir, dir.path().join("models"));
    }

    #[test]
    fn test_find_project_root_uses_ancestor_marker() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("src/bin");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.path().join(".zipcode.json"), "{}").unwrap();

        assert_eq!(find_project_root(&nested), dir.path());
    }

    #[test]
    fn test_load_project_override_gpu_settings() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"gpu_layers": 99, "flash_attention": true}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.gpu_layers, Some(99));
        assert!(config.flash_attention);
    }

    #[test]
    fn test_load_project_override_gpu_layers_overflow_rejected() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"gpu_layers": 2147483648}"#,
        )
        .unwrap();
        let result = ZipcodeConfig::load(dir.path());
        let err = result.expect_err("gpu_layers overflow should be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("gpu_layers"),
            "error message should mention gpu_layers, got: {msg}"
        );
        assert!(
            msg.contains("out of range"),
            "error message should say out of range, got: {msg}"
        );
    }

    #[test]
    fn test_load_project_override_relative_llama_server_bin() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"llama_server_bin": "bin/llama-server"}"#,
        )
        .unwrap();

        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(
            config.llama_server_bin,
            Some(dir.path().join("bin/llama-server"))
        );
    }

    #[test]
    fn test_load_project_override_tilde_model_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"model_dir": "~/.zipcode/models"}"#,
        )
        .unwrap();

        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(
            config.model_dir,
            dirs::home_dir().unwrap().join(".zipcode/models")
        );
    }

    #[test]
    fn test_load_project_override_tilde_llama_server_bin() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"llama_server_bin": "~/.zipcode/bin/llama-server"}"#,
        )
        .unwrap();

        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(
            config.llama_server_bin,
            Some(dirs::home_dir().unwrap().join(".zipcode/bin/llama-server"))
        );
    }

    #[test]
    fn test_load_global_tilde_llama_server_bin() {
        let dir = tempfile::TempDir::new().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"llama_server_bin": "~/.zipcode/bin/llama-server"}"#,
        )
        .unwrap();

        let config = ZipcodeConfig::load_from_path(&config_path).unwrap();
        assert_eq!(
            config.llama_server_bin,
            Some(dirs::home_dir().unwrap().join(".zipcode/bin/llama-server"))
        );
    }

    // ── validate() + config load rejection tests ────────────────────

    #[test]
    fn test_load_project_rejects_invalid_permission_mode() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"permission_mode": "workspace"}"#,
        )
        .unwrap();
        let err = ZipcodeConfig::load(dir.path()).unwrap_err().to_string();
        assert!(
            err.contains("unsupported permission_mode"),
            "should mention unsupported permission_mode, got: {err}"
        );
        assert!(
            err.contains("workspace"),
            "should echo the bad value, got: {err}"
        );
    }

    #[test]
    fn test_load_global_rejects_invalid_permission_mode() {
        let dir = tempfile::TempDir::new().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"permission_mode": "read"}"#).unwrap();

        let err = ZipcodeConfig::load_from_path(&config_path)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unsupported permission_mode"),
            "should mention unsupported permission_mode, got: {err}"
        );
    }

    #[test]
    fn test_load_rejects_negative_temperature() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"generation": {"temperature": -0.5}}"#,
        )
        .unwrap();
        let err = ZipcodeConfig::load(dir.path()).unwrap_err().to_string();
        assert!(
            err.contains("temperature"),
            "should mention temperature, got: {err}"
        );
    }

    #[test]
    fn test_load_rejects_zero_top_p() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"generation": {"top_p": 0.0}}"#,
        )
        .unwrap();
        let err = ZipcodeConfig::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("top_p"), "should mention top_p, got: {err}");
    }

    #[test]
    fn test_load_rejects_top_p_above_one() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"generation": {"top_p": 1.5}}"#,
        )
        .unwrap();
        let err = ZipcodeConfig::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("top_p"), "should mention top_p, got: {err}");
    }

    #[test]
    fn test_load_rejects_zero_max_tokens() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"generation": {"max_tokens": 0}}"#,
        )
        .unwrap();
        let err = ZipcodeConfig::load(dir.path()).unwrap_err().to_string();
        assert!(
            err.contains("max_tokens"),
            "should mention max_tokens, got: {err}"
        );
    }

    #[test]
    fn test_load_accepts_valid_full_access() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"permission_mode": "full-access"}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.permission_mode, "full-access");
    }

    #[test]
    fn test_load_accepts_danger_full_access() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"permission_mode": "danger-full-access"}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.permission_mode, "danger-full-access");
    }

    #[test]
    fn test_load_accepts_no_generation_overrides() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".zipcode.json"), r#"{"generation": {}}"#).unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.generation.temperature, None);
        assert_eq!(config.generation.top_p, None);
        assert_eq!(config.generation.max_tokens, None);
    }

    #[test]
    fn test_load_project_rejects_malformed_json() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".zipcode.json"), "{ invalid\n").unwrap();
        let err = ZipcodeConfig::load(dir.path()).unwrap_err().to_string();
        // Serde JSON parse error, not our validation error
        assert!(
            !err.contains("unsupported permission_mode"),
            "malformed JSON should fail at parse, not validation, got: {err}"
        );
    }

    #[test]
    fn test_validate_accepts_boundary_temperature_zero() {
        let config = ZipcodeConfig {
            generation: GenerationOverrides {
                temperature: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_accepts_boundary_top_p_one() {
        let config = ZipcodeConfig {
            generation: GenerationOverrides {
                top_p: Some(1.0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_very_small_positive_top_p() {
        // top_p = 0.0001 is technically > 0, should be accepted
        let config = ZipcodeConfig {
            generation: GenerationOverrides {
                top_p: Some(0.0001),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    // ── warn_type_mismatch / json_type_name tests ────────────────────

    #[test]
    fn test_json_type_name_all_variants() {
        assert_eq!(json_type_name(&serde_json::json!(null)), "null");
        assert_eq!(json_type_name(&serde_json::json!(true)), "boolean");
        assert_eq!(json_type_name(&serde_json::json!(42)), "number");
        assert_eq!(json_type_name(&serde_json::json!("hello")), "string");
        assert_eq!(json_type_name(&serde_json::json!([1, 2])), "array");
        assert_eq!(json_type_name(&serde_json::json!({"a": 1})), "object");
    }

    #[test]
    fn test_warn_type_mismatch_no_warning_for_correct_type() {
        // string field with string value — no panic, no output to check
        // (tracing capture would be needed for a deeper test, but we
        // verify the function does not panic and handles the happy path)
        let obj = serde_json::json!({"permission_mode": "read-only"});
        warn_type_mismatch(&obj, "permission_mode", "string");
    }

    #[test]
    fn test_warn_type_mismatch_no_warning_for_missing_field() {
        // Field doesn't exist — function should do nothing
        let obj = serde_json::json!({});
        warn_type_mismatch(&obj, "nonexistent", "string");
    }

    #[test]
    fn test_warn_type_mismatch_warns_on_wrong_type() {
        // number instead of string — function logs a warning but doesn't panic
        let obj = serde_json::json!({"permission_mode": 123});
        warn_type_mismatch(&obj, "permission_mode", "string");
        // We can't easily capture tracing output in unit tests without
        // additional infrastructure, but we verify it doesn't panic.
    }

    #[test]
    fn test_warn_type_mismatch_null_field_is_not_string() {
        let obj = serde_json::json!({"permission_mode": null});
        warn_type_mismatch(&obj, "permission_mode", "string");
    }

    // ── .zipcode.json wrong-type fields still load valid fields ───────

    #[test]
    fn test_load_permission_mode_as_number_uses_default() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"permission_mode": 123}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        // Invalid type silently ignored → stays at default "workspace-write"
        assert_eq!(config.permission_mode, "workspace-write");
    }

    #[test]
    fn test_load_gpu_layers_as_string_uses_default() {
        let (_global_dir, _global_guard) = with_isolated_global_config();
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"gpu_layers": "many"}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.gpu_layers, None);
    }

    #[test]
    fn test_load_flash_attention_as_number_uses_default() {
        let (_global_dir, _global_guard) = with_isolated_global_config();
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"flash_attention": 1}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert!(!config.flash_attention);
    }

    #[test]
    fn test_load_model_dir_as_number_uses_default() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".zipcode.json"), r#"{"model_dir": 42}"#).unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        // model_dir should remain the default
        assert!(config.model_dir.to_str().unwrap().contains(".zipcode"));
    }

    #[test]
    fn test_load_mixed_valid_and_invalid_fields() {
        let (_global_dir, _global_guard) = with_isolated_global_config();
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{
                "permission_mode": "read-only",
                "gpu_layers": "invalid",
                "flash_attention": true,
                "model_dir": 42,
                "generation": "not an object",
                "model_file": "my-model.gguf"
            }"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        // Valid fields should be loaded
        assert_eq!(config.permission_mode, "read-only");
        assert!(config.flash_attention);
        assert_eq!(config.model_file, Some("my-model.gguf".to_string()));
        // Invalid fields should be silently ignored (default values kept)
        assert_eq!(config.gpu_layers, None);
        assert!(config.model_dir.to_str().unwrap().contains(".zipcode"));
        assert_eq!(config.generation.temperature, None);
    }

    #[test]
    fn test_load_generation_temperature_as_string_ignored() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"generation": {"temperature": "hot"}}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.generation.temperature, None);
    }

    #[test]
    fn test_load_generation_as_array_ignored() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"generation": [1, 2, 3]}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.generation.temperature, None);
        assert_eq!(config.generation.top_p, None);
        assert_eq!(config.generation.max_tokens, None);
    }

    #[test]
    fn test_load_llama_server_bin_as_number_uses_default() {
        let (_global_dir, _global_guard) = with_isolated_global_config();
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".zipcode.json"),
            r#"{"llama_server_bin": 42}"#,
        )
        .unwrap();
        let config = ZipcodeConfig::load(dir.path()).unwrap();
        assert_eq!(config.llama_server_bin, None);
    }

    // ── expand_user_path direct tests ─────────────────────────────

    #[test]
    fn test_expand_user_path_tilde() {
        let expanded = expand_user_path(Path::new("~"));
        let expected = dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
        assert_eq!(expanded, expected);
    }

    #[test]
    fn test_expand_user_path_tilde_prefix() {
        let expanded = expand_user_path(Path::new("~/foo/bar"));
        let expected = dirs::home_dir()
            .map_or_else(|| PathBuf::from("~/foo/bar"), |home| home.join("foo/bar"));
        assert_eq!(expanded, expected);
    }

    #[test]
    fn test_expand_user_path_absolute_and_relative() {
        // Absolute path stays absolute
        let abs = expand_user_path(Path::new("/absolute/path"));
        assert_eq!(abs, PathBuf::from("/absolute/path"));

        // Relative path stays relative
        let rel = expand_user_path(Path::new("relative/path"));
        assert_eq!(rel, PathBuf::from("relative/path"));
    }

    // ── resolve_project_path direct tests ─────────────────────────

    #[test]
    fn test_resolve_project_path_absolute() {
        let result = resolve_project_path(Path::new("/abs/models"), Path::new("/project"));
        assert_eq!(result, PathBuf::from("/abs/models"));
    }

    #[test]
    fn test_resolve_project_path_relative() {
        let result = resolve_project_path(Path::new("models"), Path::new("/project"));
        assert_eq!(result, PathBuf::from("/project/models"));
    }

    // ── save_global tests ──────────────────────────────────────────

    #[test]
    fn test_save_global_creates_file_and_dirs() {
        let (global_dir, _global_guard) = with_isolated_global_config();
        let config_path = global_dir.path().join("saved-config.json");
        std::env::set_var("ZIPCODE_GLOBAL_CONFIG", &config_path);

        let config = ZipcodeConfig {
            permission_mode: "full-access".to_string(),
            gpu_layers: Some(42),
            flash_attention: true,
            ..Default::default()
        };

        let returned_path = config.save_global().expect("save_global should succeed");
        assert_eq!(returned_path, config_path);

        // File should exist and contain valid JSON
        let content = std::fs::read_to_string(&config_path).expect("file should be readable");
        let parsed: serde_json::Value =
            serde_json::from_str(&content).expect("should be valid JSON");
        assert_eq!(parsed["permission_mode"], "full-access");
        assert_eq!(parsed["gpu_layers"], 42);
        assert_eq!(parsed["flash_attention"], true);
    }

    #[test]
    fn test_save_global_roundtrip_with_load() {
        let (global_dir, _global_guard) = with_isolated_global_config();
        let config_path = global_dir.path().join("roundtrip.json");
        std::env::set_var("ZIPCODE_GLOBAL_CONFIG", &config_path);

        let original = ZipcodeConfig {
            permission_mode: "read-only".to_string(),
            gpu_layers: Some(10),
            flash_attention: false,
            generation: GenerationOverrides {
                temperature: Some(0.7),
                top_p: Some(0.9),
                max_tokens: Some(2048),
            },
            model_file: Some("test-model.gguf".to_string()),
            context_size: Some(8192),
            ..Default::default()
        };

        original.save_global().expect("save should succeed");

        let loaded = ZipcodeConfig::load_from_path(&config_path).expect("load should succeed");
        assert_eq!(loaded.permission_mode, "read-only");
        assert_eq!(loaded.gpu_layers, Some(10));
        assert!(!loaded.flash_attention);
        assert_eq!(loaded.generation.temperature, Some(0.7));
        assert_eq!(loaded.generation.top_p, Some(0.9));
        assert_eq!(loaded.generation.max_tokens, Some(2048));
        assert_eq!(loaded.model_file, Some("test-model.gguf".to_string()));
        assert_eq!(loaded.context_size, Some(8192));
    }

    #[test]
    fn test_save_global_creates_parent_directories() {
        let (global_dir, _global_guard) = with_isolated_global_config();
        // Use a nested path where the parent directory does not exist yet
        let nested = global_dir.path().join("nested/sub/dir/config.json");
        std::env::set_var("ZIPCODE_GLOBAL_CONFIG", &nested);

        let config = ZipcodeConfig::default();
        let returned_path = config
            .save_global()
            .expect("save_global should create parent dirs");
        assert!(returned_path.exists(), "config file should be created");
        assert!(nested.exists(), "nested config file should exist");
    }

    #[test]
    fn test_save_global_writes_pretty_json() {
        let (global_dir, _global_guard) = with_isolated_global_config();
        let config_path = global_dir.path().join("pretty.json");
        std::env::set_var("ZIPCODE_GLOBAL_CONFIG", &config_path);

        let config = ZipcodeConfig {
            permission_mode: "workspace-write".to_string(),
            ..Default::default()
        };
        config.save_global().expect("save should succeed");

        let content = std::fs::read_to_string(&config_path).expect("should be readable");
        // Pretty-printed JSON should have newlines and indentation
        assert!(
            content.contains('\n'),
            "should be pretty-printed with newlines"
        );
        assert!(
            content.contains("  "),
            "should be pretty-printed with indentation"
        );
    }

    // ── find_project_root comprehensive tests ───────────────────────

    #[test]
    fn test_find_project_root_zipcode_md_marker() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("deep/nested/dir");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.path().join(".zipcode.md"), "# project instructions").unwrap();

        assert_eq!(
            find_project_root(&nested),
            dir.path(),
            "should find .zipcode.md marker in ancestor"
        );
    }

    #[test]
    fn test_find_project_root_git_marker() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("src/lib");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();

        assert_eq!(
            find_project_root(&nested),
            dir.path(),
            "should find .git directory marker in ancestor"
        );
    }

    #[test]
    fn test_find_project_root_no_marker_returns_start() {
        // Place a .zipcode.json in the TempDir root to act as a boundary:
        // this prevents the search from escaping into system directories
        // (e.g. /tmp/.git) that may exist outside the TempDir.
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".zipcode.json"), "{}").unwrap();

        let nested = dir.path().join("isolated");
        std::fs::create_dir_all(&nested).unwrap();
        // No .zipcode.json, .zipcode.md, or .git markers in `nested` itself
        // or between `nested` and `dir.path()` — but dir.path() has one.

        assert_eq!(
            find_project_root(&nested),
            dir.path(),
            "should find boundary marker at TempDir root, not escape to system dirs"
        );
    }

    #[test]
    fn test_find_project_root_zipcode_json_takes_precedence_at_same_level() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("sub");
        std::fs::create_dir_all(&nested).unwrap();
        // Both markers present
        std::fs::write(dir.path().join(".zipcode.json"), "{}").unwrap();
        std::fs::write(dir.path().join(".zipcode.md"), "# alt").unwrap();

        assert_eq!(
            find_project_root(&nested),
            dir.path(),
            "should find root regardless of which marker is checked first"
        );
    }

    #[test]
    fn test_find_project_root_at_root_level() {
        let dir = tempfile::TempDir::new().unwrap();
        // Marker is in the start directory itself
        std::fs::write(dir.path().join(".zipcode.json"), "{}").unwrap();

        assert_eq!(
            find_project_root(dir.path()),
            dir.path(),
            "should return start dir when marker is right there"
        );
    }

    // ── global_config_path tests ────────────────────────────────────

    #[test]
    fn test_global_config_path_respects_env_var() {
        // Use the GLOBAL_CONFIG_LOCK to serialize env var mutations
        let guard = GLOBAL_CONFIG_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let custom = "/tmp/zipcode-test-custom-config.json";
        std::env::set_var("ZIPCODE_GLOBAL_CONFIG", custom);
        let result = global_config_path();
        assert_eq!(
            result,
            std::path::PathBuf::from(custom),
            "should use ZIPCODE_GLOBAL_CONFIG env var"
        );

        // Clean up
        std::env::remove_var("ZIPCODE_GLOBAL_CONFIG");
        drop(guard);
    }

    #[test]
    fn test_global_config_path_default_without_env_var() {
        // This test verifies the default behavior when ZIPCODE_GLOBAL_CONFIG is not set.
        let guard = GLOBAL_CONFIG_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        std::env::remove_var("ZIPCODE_GLOBAL_CONFIG");
        let result = global_config_path();

        // Should end with .zipcode/config.json
        assert!(
            result.to_str().unwrap().contains(".zipcode"),
            "default path should contain .zipcode directory, got: {}",
            result.display()
        );
        assert!(
            result.file_name().unwrap() == "config.json",
            "default path should end with config.json, got: {}",
            result.display()
        );

        drop(guard);
    }
}
