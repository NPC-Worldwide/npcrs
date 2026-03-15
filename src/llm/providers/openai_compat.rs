//! OpenAI-compatible chat completion API.
//!
//! Works with: OpenAI, Ollama (/v1), vLLM, Together, Groq,
//! and any provider that implements the OpenAI chat completions format.

use crate::error::{NpcError, Result};
use crate::llm::types::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDef]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<UsageResponse>,
    model: Option<String>,
}

#[derive(Deserialize)]
struct Choice {
    message: MessageResponse,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct MessageResponse {
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallResponse>>,
}

#[derive(Deserialize)]
struct ToolCallResponse {
    id: String,
    r#type: String,
    function: ToolCallFunctionResponse,
}

#[derive(Deserialize)]
struct ToolCallFunctionResponse {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct UsageResponse {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

pub async fn chat_completion(
    client: &Client,
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    messages: &[Message],
    tools: Option<&[ToolDef]>,
) -> Result<LlmResponse> {
    let url = format!("{}/chat/completions", base_url);

    let body = ChatRequest {
        model,
        messages,
        tools: if tools.is_some_and(|t| !t.is_empty()) {
            tools
        } else {
            None
        },
        temperature: None,
    };

    let mut req = client.post(&url).json(&body);

    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let resp = req.send().await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(NpcError::LlmRequest(format!(
            "HTTP {}: {}",
            status, body
        )));
    }

    let chat_resp: ChatResponse = resp.json().await?;

    let choice = chat_resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| NpcError::LlmRequest("No choices in response".to_string()))?;

    let tool_calls = choice.message.tool_calls.map(|tcs| {
        tcs.into_iter()
            .map(|tc| ToolCall {
                id: tc.id,
                r#type: tc.r#type,
                function: ToolCallFunction {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
            })
            .collect()
    });

    let usage = chat_resp.usage.map(|u| Usage {
        prompt_tokens: u.prompt_tokens.unwrap_or(0),
        completion_tokens: u.completion_tokens.unwrap_or(0),
        total_tokens: u.total_tokens.unwrap_or(0),
    });

    Ok(LlmResponse {
        message: Message {
            role: choice
                .message
                .role
                .unwrap_or_else(|| "assistant".to_string()),
            content: choice.message.content,
            tool_calls,
            tool_call_id: None,
            name: None,
        },
        usage,
        model: chat_resp.model.unwrap_or_else(|| model.to_string()),
        finish_reason: choice.finish_reason,
        cost_usd: None,
    })
}
