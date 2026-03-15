use crate::npc_compiler::Jinx;
use crate::npc_compiler::Npc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A team of NPCs with shared jinxes and context.
#[derive(Debug, Clone)]
pub struct Team {
    /// All NPCs in the team, keyed by name.
    pub npcs: HashMap<String, Npc>,

    /// The lead NPC (forenpc / orchestrator).
    pub forenpc: Option<String>,

    /// All available jinxes (team-level + global), keyed by name.
    pub jinxes: HashMap<String, Jinx>,

    /// Team context from .ctx file.
    pub context: Option<String>,

    /// Default model for the team.
    pub model: Option<String>,

    /// Default provider for the team.
    pub provider: Option<String>,

    /// Shared mutable state across all NPCs.
    pub shared_context: HashMap<String, serde_json::Value>,

    /// Database paths from .ctx file.
    pub databases: Vec<String>,

    /// MCP server specs from .ctx file.
    pub mcp_servers: Vec<crate::npc_compiler::McpServerSpec>,

    /// Directory this team was loaded from.
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
    /// Get an NPC by name.
    pub fn get_npc(&self, name: &str) -> Option<&Npc> {
        self.npcs.get(name)
    }

    /// Get a mutable reference to an NPC by name.
    pub fn get_npc_mut(&mut self, name: &str) -> Option<&mut Npc> {
        self.npcs.get_mut(name)
    }

    /// Get the forenpc (lead NPC), or the first NPC if none is designated.
    pub fn lead_npc(&self) -> Option<&Npc> {
        self.forenpc
            .as_ref()
            .and_then(|name| self.npcs.get(name))
            .or_else(|| self.npcs.values().next())
    }

    /// List all NPC names.
    pub fn npc_names(&self) -> Vec<&str> {
        self.npcs.keys().map(|s| s.as_str()).collect()
    }

    /// List all jinx names.
    pub fn jinx_names(&self) -> Vec<&str> {
        self.jinxes.keys().map(|s| s.as_str()).collect()
    }
}

/// The .ctx file format (team context configuration).
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
