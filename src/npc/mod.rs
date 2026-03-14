//! NPC agent definition, loading, and tool resolution.

mod loader;
mod types;

pub use loader::load_npc_from_file;
pub use types::*;

use crate::error::Result;
use crate::jinx::Jinx;
use crate::llm::{self, LlmClient, Message, ToolDef};
use std::collections::HashMap;
use std::path::Path;

impl Npc {
    /// Load an NPC from a .npc YAML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        load_npc_from_file(path)
    }

    /// Create a minimal NPC with just a name and directive.
    pub fn new(name: impl Into<String>, primary_directive: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            primary_directive: Some(primary_directive.into()),
            ..Default::default()
        }
    }

    /// Build the full system prompt for this NPC.
    pub fn system_prompt(&self, team_context: Option<&str>) -> String {
        let mut parts = Vec::new();

        if let Some(ctx) = team_context {
            parts.push(ctx.to_string());
        }

        if let Some(ref directive) = self.primary_directive {
            parts.push(format!("Your name is {}.\n{}", self.name, directive));
        } else {
            parts.push(format!(
                "Your name is {}. You are a helpful assistant.",
                self.name
            ));
        }

        if let Some(ref memory) = self.memory {
            parts.push(format!("## Your Memory\n{}", memory));
        }

        parts.join("\n\n")
    }

    /// Resolve all available tools: jinxes + MCP + registered functions.
    /// Returns (tool_defs_for_llm, executor_map).
    pub fn resolve_tools(&self, jinxes: &HashMap<String, Jinx>) -> (Vec<ToolDef>, HashMap<String, ToolExecutor>) {
        let mut defs = Vec::new();
        let mut executors = HashMap::new();

        // Jinx tools
        for jinx_name in &self.jinx_names {
            if let Some(jinx) = jinxes.get(jinx_name) {
                if let Some(tool_def) = jinx.to_tool_def() {
                    executors.insert(
                        jinx.name.clone(),
                        ToolExecutor::Jinx(jinx.name.clone()),
                    );
                    defs.push(tool_def);
                }
            }
        }

        // MCP tools would be resolved at runtime via async MCP client
        for mcp in &self.mcp_servers {
            executors.insert(
                format!("mcp:{}", mcp.path),
                ToolExecutor::Mcp(mcp.clone()),
            );
        }

        (defs, executors)
    }

    /// Get an LLM response for a prompt, with tool calling support.
    pub async fn get_response(
        &self,
        client: &LlmClient,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
    ) -> Result<llm::LlmResponse> {
        let model = self.resolved_model();
        let provider = self.resolved_provider();

        client
            .chat_completion(&provider, &model, messages, tools, self.api_url.as_deref())
            .await
    }

    /// Model to use, falling back to defaults.
    pub fn resolved_model(&self) -> String {
        self.model
            .clone()
            .unwrap_or_else(|| "gpt-4o-mini".to_string())
    }

    /// Provider to use, falling back to defaults.
    pub fn resolved_provider(&self) -> String {
        self.provider
            .clone()
            .unwrap_or_else(|| "openai".to_string())
    }
}
