//! Event triggers.

use crate::error::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trigger {
    pub name: String,
    pub event: String,
    pub action: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool { true }

pub fn load_triggers(dir: &str) -> Result<Vec<Trigger>> {
    let mut triggers = Vec::new();
    let path = std::path::Path::new(dir);
    if !path.is_dir() {
        return Ok(triggers);
    }
    for entry in std::fs::read_dir(path).into_iter().flatten().flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("yaml")
            || p.extension().and_then(|e| e.to_str()) == Some("yml")
        {
            if let Ok(content) = std::fs::read_to_string(&p) {
                if let Ok(t) = serde_yaml::from_str::<Trigger>(&content) {
                    triggers.push(t);
                }
            }
        }
    }
    Ok(triggers)
}

pub fn check_trigger(trigger: &Trigger, event: &str) -> bool {
    trigger.enabled && trigger.event == event
}
