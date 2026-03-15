//! Unified response generation — provider dispatch, sanitization, cost tracking.

use crate::error::Result;
use crate::llm::cost::calculate_cost;
use crate::llm::sanitize::sanitize_messages;
use crate::llm::{LlmClient, LlmResponse, Message, ToolDef};

/// Unified response generation entry point mirroring npcpy's `get_litellm_response()`.
///
/// Sanitizes messages, dispatches to the LLM, and computes cost.
pub async fn get_response(
    client: &LlmClient,
    provider: &str,
    model: &str,
    messages: Vec<Message>,
    tools: Option<&[ToolDef]>,
    api_url: Option<&str>,
) -> Result<LlmResponse> {
    // 1. Sanitize messages
    let clean_messages = sanitize_messages(messages);

    // 2. Call LLM
    let mut response = client
        .chat_completion(provider, model, &clean_messages, tools, api_url)
        .await?;

    // 3. Calculate cost
    if let Some(ref usage) = response.usage {
        response.cost_usd =
            Some(calculate_cost(model, usage.prompt_tokens, usage.completion_tokens));
    }

    Ok(response)
}
