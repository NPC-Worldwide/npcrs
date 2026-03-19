use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Npc {
    pub name: String,

    #[serde(default)]
    pub primary_directive: Option<String>,

    #[serde(default)]
    pub model: Option<String>,

    #[serde(default)]
    pub provider: Option<String>,

    #[serde(default)]
    pub api_url: Option<String>,

    #[serde(default)]
    pub api_key: Option<String>,

    #[serde(default)]
    pub ascii_art: Option<String>,

    #[serde(default)]
    pub colors: Option<NpcColors>,

    #[serde(default, alias = "jinxes")]
    pub jinx_names: Vec<String>,

    #[serde(default)]
    pub mcp_servers: Vec<McpServerSpec>,

    #[serde(default)]
    pub use_global_jinxes: bool,

    #[serde(skip)]
    pub memory: Option<String>,

    #[serde(skip)]
    pub shared_context: HashMap<String, serde_json::Value>,

    #[serde(skip)]
    pub source_path: Option<String>,
}

impl Default for Npc {
    fn default() -> Self {
        Self {
            name: "assistant".to_string(),
            primary_directive: None,
            model: None,
            provider: None,
            api_url: None,
            api_key: None,
            ascii_art: None,
            colors: None,
            jinx_names: Vec::new(),
            mcp_servers: Vec::new(),
            use_global_jinxes: false,
            memory: None,
            shared_context: HashMap::new(),
            source_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcColors {
    pub top: Option<String>,
    pub bottom: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct McpServerSpec {
    pub path: String,

    pub command: Option<String>,

    pub tools: Vec<String>,
}

impl<'de> Deserialize<'de> for McpServerSpec {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum McpSpec {
            Path(String),
            Full {
                path: String,
                #[serde(default)]
                command: Option<String>,
                #[serde(default)]
                tools: Vec<String>,
            },
        }

        match McpSpec::deserialize(deserializer)? {
            McpSpec::Path(path) => Ok(McpServerSpec {
                path,
                command: None,
                tools: Vec::new(),
            }),
            McpSpec::Full { path, command, tools } => Ok(McpServerSpec {
                path,
                command,
                tools,
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ToolExecutor {
    Jinx(String),
    Mcp(McpServerSpec),
    Native(String),
    Python(String),
}
