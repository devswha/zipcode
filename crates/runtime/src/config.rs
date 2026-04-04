use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipcodeConfig {
    #[serde(default = "default_model_dir")]
    pub model_dir: PathBuf,
    #[serde(default)]
    pub model_file: Option<String>,
    #[serde(default = "default_permission")]
    pub permission_mode: String,
    #[serde(default)]
    pub generation: GenerationOverrides,
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
            permission_mode: default_permission(),
            generation: GenerationOverrides::default(),
        }
    }
}

impl ZipcodeConfig {
    /// Load config with hierarchy: global (~/.zipcode/config.json) < project (.zipcode.json)
    pub fn load(cwd: &Path) -> Result<Self> {
        let mut config = Self::default();
        let project_root = find_project_root(cwd);

        // Global config
        let global_path = dirs::home_dir().map(|h| h.join(".zipcode/config.json"));
        if let Some(path) = global_path {
            if path.exists() {
                let content = std::fs::read_to_string(&path)?;
                config = serde_json::from_str(&content)?;
            }
        }

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
        }

        Ok(config)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ZipcodeConfig::default();
        assert_eq!(config.permission_mode, "workspace-write");
        assert!(config.model_dir.to_str().unwrap().contains(".zipcode"));
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
}
