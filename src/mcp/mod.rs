
mod client;

pub use client::*;

use crate::r#gen::ToolDef;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
    pub server_path: String,
}

impl McpTool {
    pub fn to_tool_def(&self) -> ToolDef {
        ToolDef {
            r#type: "function".to_string(),
            function: crate::r#gen::FunctionDef {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.input_schema.clone(),
            },
        }
    }
}
