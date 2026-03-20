use crate::error::{NpcError, Result};
use crate::r#gen::Message;
use std::collections::HashMap;

pub struct StreamConfig {
    pub npc_name: Option<String>,
    pub model: String,
    pub provider: String,
    pub messages: Vec<Message>,
    pub command: String,
    pub temperature: f64,
    pub attachments: Vec<String>,
    pub images: Vec<String>,
}

pub struct StreamEvent {
    pub event_type: String,
    pub content: String,
    pub model: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<serde_json::Value>,
    pub done: bool,
}

pub fn clean_messages_for_llm(messages: &[Message]) -> Vec<Message> {
    crate::r#gen::sanitize::sanitize_messages(messages.to_vec())
}

pub fn ensure_system_prompt(messages: &mut Vec<Message>, system_prompt: Option<&str>) {
    let has_system = messages.first().map(|m| m.role == "system").unwrap_or(false);
    if !has_system {
        let prompt = system_prompt.unwrap_or("You are a helpful assistant.");
        messages.insert(0, Message::system(prompt));
    }
}

pub fn parse_stream_chunk(chunk: &serde_json::Value, _model: &str, _provider: &str) -> (String, String, Vec<serde_json::Value>) {
    let content = chunk.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()).unwrap_or("").to_string();
    let reasoning = chunk.get("message").and_then(|m| m.get("reasoning_content")).and_then(|c| c.as_str()).unwrap_or("").to_string();
    let tool_calls = chunk.get("message").and_then(|m| m.get("tool_calls")).and_then(|t| t.as_array()).cloned().unwrap_or_default();
    (content, reasoning, tool_calls)
}

pub fn format_sse_event(event: &StreamEvent) -> String {
    let data = serde_json::json!({
        "type": event.event_type,
        "content": event.content,
        "model": event.model,
        "reasoning": event.reasoning,
        "done": event.done,
    });
    format!("data: {}\n\n", data)
}

pub fn format_sse_raw(data: &serde_json::Value) -> String {
    format!("data: {}\n\n", data)
}

pub fn resolve_npc_tools(npc: &crate::npc_compiler::NPC, jinxes: &HashMap<String, crate::npc_compiler::Jinx>) -> (Vec<crate::r#gen::ToolDef>, HashMap<String, crate::npc_compiler::ToolExecutor>) {
    npc.resolve_tools(jinxes)
}

pub async fn execute_tool(tool_name: &str, tool_args: &serde_json::Value, _tool_id: &str, jinxes: &HashMap<String, crate::npc_compiler::Jinx>) -> Result<String> {
    if let Some(jinx) = jinxes.get(tool_name) {
        let mut inputs = HashMap::new();
        if let Some(obj) = tool_args.as_object() {
            for (k, v) in obj {
                inputs.insert(k.clone(), v.as_str().unwrap_or(&v.to_string()).to_string());
            }
        }
        let result = jinx.execute(&inputs);
        Ok(result.output)
    } else {
        match tool_name {
            "sh" => {
                let cmd = tool_args.get("command").and_then(|v| v.as_str()).unwrap_or("");
                let output = std::process::Command::new("sh").args(["-c", cmd]).output()
                    .map_err(|e| NpcError::Shell(format!("sh: {}", e)))?;
                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            }
            "python" => {
                let code = tool_args.get("code").and_then(|v| v.as_str()).unwrap_or("");
                let output = std::process::Command::new("python3").args(["-c", code]).output()
                    .map_err(|e| NpcError::Shell(format!("python: {}", e)))?;
                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            }
            "web_search" => {
                let query = tool_args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let results = crate::data::web::search_web(query, 5, "duckduckgo", None).await?;
                Ok(results.iter().map(|r| format!("{}: {}\n{}", r.title, r.url, r.snippet)).collect::<Vec<_>>().join("\n\n"))
            }
            _ => Ok(format!("Unknown tool: {}", tool_name)),
        }
    }
}

pub fn flatten_tool_messages(messages: &[Message]) -> Vec<Message> {
    let mut flat = Vec::new();
    for msg in messages {
        if let Some(ref tcs) = msg.tool_calls {
            let parts: Vec<String> = tcs.iter().map(|tc| {
                format!("Called {} with: {}", tc.function.name, tc.function.arguments)
            }).collect();
            flat.push(Message { role: "assistant".into(), content: Some(parts.join("\n")), tool_calls: None, tool_call_id: None, name: None });
        } else if msg.role == "tool" {
            let name = msg.name.as_deref().unwrap_or("tool");
            let content = msg.content.as_deref().unwrap_or("");
            flat.push(Message { role: "user".into(), content: Some(format!("Result of {}: {}", name, content)), tool_calls: None, tool_call_id: None, name: None });
        } else {
            flat.push(msg.clone());
        }
    }
    flat
}
