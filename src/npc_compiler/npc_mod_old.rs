
use crate::error::Result;
use crate::npc_compiler::{Jinx, Npc, ToolExecutor, McpServerSpec};
use crate::r#gen::{Message, ToolDef, LlmResponse};
use std::collections::HashMap;
use std::path::Path;

impl Npc {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        super::npc_loader::load_npc_from_file(path)
    }

    pub fn new(name: impl Into<String>, primary_directive: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            primary_directive: Some(primary_directive.into()),
            ..Default::default()
        }
    }

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

    pub fn resolve_tools(&self, jinxes: &HashMap<String, Jinx>) -> (Vec<ToolDef>, HashMap<String, ToolExecutor>) {
        let mut defs = Vec::new();
        let mut executors = HashMap::new();

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

        for mcp in &self.mcp_servers {
            executors.insert(
                format!("mcp:{}", mcp.path),
                ToolExecutor::Mcp(mcp.clone()),
            );
        }

        (defs, executors)
    }

    pub async fn get_response(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
    ) -> Result<LlmResponse> {
        let model = self.resolved_model();
        let provider = self.resolved_provider();

        crate::r#gen::get_genai_response(&provider, &model, messages, tools, self.api_url.as_deref()).await
    }

    pub fn resolved_model(&self) -> String {
        self.model
            .clone()
            .or_else(|| std::env::var("NPCSH_CHAT_MODEL").ok())
            .unwrap_or_else(|| "qwen3.5:2b".to_string())
    }

    pub fn resolved_provider(&self) -> String {
        self.provider
            .clone()
            .or_else(|| std::env::var("NPCSH_CHAT_PROVIDER").ok())
            .unwrap_or_else(|| "ollama".to_string())
    }
}
