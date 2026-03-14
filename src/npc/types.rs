use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An NPC agent with personality, tools, and LLM configuration.
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

    /// Jinx names this NPC has access to (resolved from .npc file).
    #[serde(default, alias = "jinxes")]
    pub jinx_names: Vec<String>,

    /// MCP server configurations.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerSpec>,

    /// Whether to load jinxes from the global directory.
    #[serde(default)]
    pub use_global_jinxes: bool,

    /// Runtime memory context (populated at runtime, not from file).
    #[serde(skip)]
    pub memory: Option<String>,

    /// Shared mutable context (populated at runtime).
    #[serde(skip)]
    pub shared_context: HashMap<String, serde_json::Value>,

    /// Source file path (set during loading).
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

/// RGB gradient colors for NPC display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcColors {
    pub top: Option<String>,
    pub bottom: Option<String>,
}

/// MCP server connection specification.
///
/// Deserializes from either:
/// - A plain string: `"~/.npcsh/mcp_server.py"` → path only
/// - A struct: `{path: "...", command: "...", tools: [...]}`
#[derive(Debug, Clone, Serialize)]
pub struct McpServerSpec {
    /// Path to MCP server script or binary.
    pub path: String,

    /// Command to run (alternative to path, e.g. "npx @something").
    pub command: Option<String>,

    /// Whitelisted tool names (empty = all tools).
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

/// How a tool is executed when the LLM calls it.
#[derive(Debug, Clone)]
pub enum ToolExecutor {
    /// Execute a Jinx by name.
    Jinx(String),
    /// Call an MCP server tool.
    Mcp(McpServerSpec),
    /// Call a registered native function.
    Native(String),
    /// Execute via embedded Python (future).
    Python(String),
}
