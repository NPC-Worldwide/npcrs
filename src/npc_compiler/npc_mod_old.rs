//! NPC impl block — system prompt, tool resolution, model resolution.

use crate::error::Result;
use crate::npc_compiler::{Jinx, Npc, ToolExecutor, McpServerSpec};
use crate::r#gen::{Message, ToolDef, LlmResponse};
use std::collections::HashMap;
use std::path::Path;

impl Npc {
    /// Load an NPC from a .npc YAML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        super::npc_loader::load_npc_from_file(path)
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
    /// Uses the global standalone chat_completion function (no client needed).
    pub async fn get_response(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
    ) -> Result<LlmResponse> {
        let model = self.resolved_model();
        let provider = self.resolved_provider();

        crate::r#gen::get_genai_response(&provider, &model, messages, tools, self.api_url.as_deref()).await
    }

    /// Model to use, falling back to env vars then defaults.
    pub fn resolved_model(&self) -> String {
        self.model
            .clone()
            .or_else(|| std::env::var("NPCSH_CHAT_MODEL").ok())
            .unwrap_or_else(|| "qwen3.5:2b".to_string())
    }

    /// Provider to use, falling back to env vars then defaults.
    pub fn resolved_provider(&self) -> String {
        self.provider
            .clone()
            .or_else(|| std::env::var("NPCSH_CHAT_PROVIDER").ok())
            .unwrap_or_else(|| "ollama".to_string())
    }
}
