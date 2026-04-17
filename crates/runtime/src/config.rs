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
        return dirs::home_dir()
            .map(|home| home.join(stripped))
            .unwrap_or_else(|| path.to_path_buf());
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
    pub fn load_global() -> Result<Self> {
        let path = global_config_path();
        if path.exists() {
            Self::load_from_path(&path)
        } else {
            Ok(Self::default())
        }
    }

    /// Validate config field values, returning an error for out-of-range or
    /// unsupported settings. Called automatically at the end of [`load`] and
    /// [`load_from_path`].
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
    pub fn load(cwd: &Path) -> Result<Self> {
        let mut config = Self::load_global()?;
        let project_root = find_project_root(cwd);

        // Project config (overrides)
        let project_path = project_root.join(".zipcode.json");
        if project_path.exists() {
            let content = std::fs::read_to_string(&project_path)?;
            let project: serde_json::Value = serde_json::from_str(&content)?;
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
                if let Some(t) = gen["temperature"].as_f64() {
                    config.generation.temperature = Some(t);
                }
                if let Some(t) = gen["top_p"].as_f64() {
                    config.generation.top_p = Some(t);
                }
                if let Some(t) = gen["max_tokens"].as_u64() {
                    config.generation.max_tokens = Some(t as usize);
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
        }

        config.validate()?;
        Ok(config)
    }

    /// Save the config to ~/.zipcode/config.json, creating parent directories if needed.
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
pub fn global_config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".zipcode/config.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ZipcodeConfig::default();
        assert_eq!(config.permission_mode, "workspace-write");
        assert!(config.model_dir.to_str().unwrap().contains(".zipcode"));
        assert_eq!(config.llama_server_bin, None);
    }

    #[test]
    fn test_load_from_empty_dir() {
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
}
