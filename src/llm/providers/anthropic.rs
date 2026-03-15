//! Anthropic Messages API client.
//!
//! Anthropic uses a different format from OpenAI:
//! - System message is a top-level field, not in the messages array
//! - Tool definitions use `input_schema` instead of `parameters`
//! - Tool results use `tool_result` content blocks

use crate::error::{NpcError, Result};
use crate::llm::types::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: serde_json::Value,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: serde_json::Value,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    usage: Option<AnthropicUsage>,
    model: Option<String>,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

pub async fn chat_completion(
    client: &Client,
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    messages: &[Message],
    tools: Option<&[ToolDef]>,
) -> Result<LlmResponse> {
    let url = format!("{}/v1/messages", base_url);

    // Extract system message
    let system = messages
        .iter()
        .find(|m| m.role == "system")
        .and_then(|m| m.content.clone());

    // Convert messages (skip system, convert tool messages)
    let anthropic_messages: Vec<AnthropicMessage> = messages
        .iter()
        .filter(|m| m.role != "system")
        .map(|m| {
            if m.role == "tool" {
                // Anthropic uses tool_result content blocks
                AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::json!([{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }]),
                }
            } else if let Some(ref tool_calls) = m.tool_calls {
                // Assistant message with tool calls
                let mut blocks: Vec<serde_json::Value> = Vec::new();
                if let Some(ref text) = m.content {
                    if !text.is_empty() {
                        blocks.push(serde_json::json!({"type": "text", "text": text}));
                    }
                }
                for tc in tool_calls {
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": input,
                    }));
                }
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: serde_json::Value::Array(blocks),
                }
            } else {
                AnthropicMessage {
                    role: m.role.clone(),
                    content: serde_json::Value::String(
                        m.content.clone().unwrap_or_default(),
                    ),
                }
            }
        })
        .collect();

    // Convert tools
    let anthropic_tools = tools.map(|ts| {
        ts.iter()
            .map(|t| AnthropicTool {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                input_schema: t.function.parameters.clone(),
            })
            .collect::<Vec<_>>()
    });

    let body = AnthropicRequest {
        model,
        max_tokens: 4096,
        system,
        messages: anthropic_messages,
        tools: if anthropic_tools
            .as_ref()
            .is_some_and(|t| !t.is_empty())
        {
            anthropic_tools
        } else {
            None
        },
    };

    let key = api_key.ok_or_else(|| {
        NpcError::LlmRequest("ANTHROPIC_API_KEY not set".to_string())
    })?;

    let resp = client
        .post(&url)
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(NpcError::LlmRequest(format!(
            "Anthropic HTTP {}: {}",
            status, body
        )));
    }

    let anthropic_resp: AnthropicResponse = resp.json().await?;

    // Extract text and tool calls from content blocks
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in anthropic_resp.content {
        match block {
            ContentBlock::Text { text } => text_parts.push(text),
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id,
                    r#type: "function".to_string(),
                    function: ToolCallFunction {
                        name,
                        arguments: serde_json::to_string(&input)
                            .unwrap_or_default(),
                    },
                });
            }
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let usage = anthropic_resp.usage.map(|u| Usage {
        prompt_tokens: u.input_tokens.unwrap_or(0),
        completion_tokens: u.output_tokens.unwrap_or(0),
        total_tokens: u.input_tokens.unwrap_or(0)
            + u.output_tokens.unwrap_or(0),
    });

    Ok(LlmResponse {
        message: Message {
            role: "assistant".to_string(),
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
            name: None,
        },
        usage,
        model: anthropic_resp
            .model
            .unwrap_or_else(|| model.to_string()),
        finish_reason: anthropic_resp.stop_reason,
        cost_usd: None,
    })
}
