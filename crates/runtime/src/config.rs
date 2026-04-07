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

impl ZipcodeConfig {
    /// Load only the global config (~/.zipcode/config.json), or defaults if it does not exist.
    pub fn load_global() -> Result<Self> {
        let path = global_config_path();
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(Self::default())
        }
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
                let path = PathBuf::from(dir);
                config.model_dir = if path.is_absolute() {
                    path
                } else {
                    project_root.join(path)
                };
            }
            if let Some(file) = project["model_file"].as_str() {
                config.model_file = Some(file.to_string());
            }
            if let Some(bin) = project["llama_server_bin"].as_str() {
                let path = PathBuf::from(bin);
                config.llama_server_bin = Some(if path.is_absolute() {
                    path
                } else {
                    project_root.join(path)
                });
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
                config.gpu_layers = Some(layers as i32);
            }
            if let Some(fa) = project["flash_attention"].as_bool() {
                config.flash_attention = fa;
            }
        }

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
}
