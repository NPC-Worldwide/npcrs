//! Model Context Protocol (MCP) client and server.
//!
//! MCP enables NPCs to connect to external tool servers using a standard protocol.
//! This module provides:
//! - Client: connect to MCP servers (stdio or HTTP), fetch tools, call tools
//! - Server: expose NPC jinxes as MCP tools for external consumers

mod client;

pub use client::*;

use crate::llm::ToolDef;
use serde::{Deserialize, Serialize};

/// An MCP tool discovered from a server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
    /// Which server this tool came from.
    pub server_path: String,
}

impl McpTool {
    /// Convert to an OpenAI-compatible tool definition.
    pub fn to_tool_def(&self) -> ToolDef {
        ToolDef {
            r#type: "function".to_string(),
            function: crate::llm::FunctionDef {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.input_schema.clone(),
            },
        }
    }
}
