use crate::npc_compiler::Jinx;
use crate::npc_compiler::Npc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Team {
    pub npcs: HashMap<String, Npc>,

    pub forenpc: Option<String>,

    pub jinxes: HashMap<String, Jinx>,

    pub context: Option<String>,

    pub model: Option<String>,

    pub provider: Option<String>,

    pub shared_context: HashMap<String, serde_json::Value>,

    pub databases: Vec<String>,

    pub mcp_servers: Vec<crate::npc_compiler::McpServerSpec>,

    pub source_dir: Option<String>,
}

impl Default for Team {
    fn default() -> Self {
        Self {
            npcs: HashMap::new(),
            forenpc: None,
            jinxes: HashMap::new(),
            context: None,
            model: None,
            provider: None,
            shared_context: HashMap::new(),
            databases: Vec::new(),
            mcp_servers: Vec::new(),
            source_dir: None,
        }
    }
}

impl Team {
    pub fn get_npc(&self, name: &str) -> Option<&Npc> {
        self.npcs.get(name)
    }

    pub fn get_npc_mut(&mut self, name: &str) -> Option<&mut Npc> {
        self.npcs.get_mut(name)
    }

    pub fn lead_npc(&self) -> Option<&Npc> {
        self.forenpc
            .as_ref()
            .and_then(|name| self.npcs.get(name))
            .or_else(|| self.npcs.values().next())
    }

    pub fn npc_names(&self) -> Vec<&str> {
        self.npcs.keys().map(|s| s.as_str()).collect()
    }

    pub fn jinx_names(&self) -> Vec<&str> {
        self.jinxes.keys().map(|s| s.as_str()).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamCtx {
    #[serde(default)]
    pub context: Option<String>,

    #[serde(default)]
    pub model: Option<String>,

    #[serde(default)]
    pub provider: Option<String>,

    #[serde(default)]
    pub api_url: Option<String>,

    #[serde(default)]
    pub api_key: Option<String>,

    #[serde(default)]
    pub forenpc: Option<String>,

    #[serde(default)]
    pub databases: Vec<String>,

    #[serde(default)]
    pub mcp_servers: Vec<crate::npc_compiler::McpServerSpec>,

    #[serde(default)]
    pub use_global_jinxes: bool,

    #[serde(default)]
    pub preferences: Vec<String>,
}
