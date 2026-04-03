use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use zipcode_inference::ChatMessage;

#[derive(Debug, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: String,
    pub updated_at: String,
}

impl Session {
    pub fn new() -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id,
            messages: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = session_path(&self.id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)
            .with_context(|| format!("Failed to save session to {}", path.display()))?;
        Ok(())
    }

    pub fn load(id: &str) -> Result<Self> {
        let path = session_path(id);
        let content =
            std::fs::read_to_string(&path).with_context(|| format!("Session not found: {id}"))?;
        let session: Self = serde_json::from_str(&content)?;
        Ok(session)
    }

    pub fn push_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

fn session_path(id: &str) -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(format!(".zipcode/sessions/{id}.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_session() {
        let session = Session::new();
        assert!(!session.id.is_empty());
        assert!(session.messages.is_empty());
    }

    #[test]
    fn test_push_message() {
        let mut session = Session::new();
        session.push_message(ChatMessage::user("hello"));
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn test_session_roundtrip() {
        let mut session = Session::new();
        session.push_message(ChatMessage::user("test"));

        // Save and reload
        session.save().unwrap();
        let loaded = Session::load(&session.id).unwrap();
        assert_eq!(loaded.messages.len(), 1);

        // Cleanup
        let path = session_path(&session.id);
        std::fs::remove_file(path).ok();
    }
}
