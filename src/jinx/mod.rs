//! Jinx workflow templates - the tool/skill system.
//!
//! A Jinx is a YAML-defined workflow with typed inputs and executable steps.
//! Steps can run bash commands, execute via Tera templates, or delegate to
//! other jinxes. The Rust runtime handles template rendering and step dispatch,
//! deferring to Python only for `engine: python` steps.

mod loader;
mod executor;
mod types;

pub use loader::*;
pub use executor::*;
pub use types::*;

use crate::error::Result;
use crate::llm::ToolDef;

impl Jinx {
    /// Load a Jinx from a .jinx YAML file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        loader::load_jinx_from_file(path)
    }

    /// Convert this Jinx to an OpenAI-compatible tool definition for LLM tool calling.
    pub fn to_tool_def(&self) -> Option<ToolDef> {
        if self.name.is_empty() || self.description.is_empty() {
            return None;
        }

        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for input in &self.inputs {
            let mut prop = serde_json::Map::new();
            prop.insert("type".into(), serde_json::Value::String("string".into()));

            if let Some(ref desc) = input.description {
                prop.insert(
                    "description".into(),
                    serde_json::Value::String(desc.clone()),
                );
            }

            properties.insert(
                input.name.clone(),
                serde_json::Value::Object(prop),
            );

            if input.default.is_none() {
                required.push(serde_json::Value::String(input.name.clone()));
            }
        }

        Some(ToolDef {
            r#type: "function".to_string(),
            function: crate::llm::FunctionDef {
                name: self.name.clone(),
                description: Some(self.description.clone()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }),
            },
        })
    }
}
