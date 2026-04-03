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

        // Global config
        let global_path = dirs::home_dir().map(|h| h.join(".zipcode/config.json"));
        if let Some(path) = global_path {
            if path.exists() {
                let content = std::fs::read_to_string(&path)?;
                config = serde_json::from_str(&content)?;
            }
        }

        // Project config (overrides)
        let project_path = cwd.join(".zipcode.json");
        if project_path.exists() {
            let content = std::fs::read_to_string(&project_path)?;
            let project: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(perm) = project["permission_mode"].as_str() {
                config.permission_mode = perm.to_string();
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
}
