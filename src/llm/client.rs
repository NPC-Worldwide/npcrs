//! LLM client powered by the `genai` crate (v0.5).
//!
//! genai handles provider routing, API keys (from env), and protocol
//! differences (OpenAI, Anthropic, Gemini, Ollama, Groq, etc.) automatically.

use crate::error::{NpcError, Result};
use crate::llm::types::*;

use genai::chat::{
    ChatMessage, ChatRequest, ChatResponse as GenaiChatResponse, ContentPart,
    MessageContent as GenaiContent, Tool as GenaiTool, ToolCall as GenaiToolCall,
    ToolResponse as GenaiToolResponse,
};
use genai::Client as GenaiClient;

/// Multi-provider LLM client backed by `genai`.
pub struct LlmClient {
    client: GenaiClient,
}

impl LlmClient {
    /// Create from environment (genai auto-discovers API keys).
    pub fn from_env() -> Self {
        Self {
            client: GenaiClient::default(),
        }
    }

    /// Create with a pre-built genai client.
    pub fn new_with_genai(client: GenaiClient) -> Self {
        Self { client }
    }

    /// Send a chat completion request.
    ///
    /// `provider` is ignored — genai infers the provider from the model name.
    /// `api_url_override` is currently unused (genai manages endpoints).
    pub async fn chat_completion(
        &self,
        _provider: &str,
        model: &str,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
        _api_url_override: Option<&str>,
    ) -> Result<LlmResponse> {
        // Build genai ChatRequest from our Message types
        let mut req = ChatRequest::new(Vec::new());

        for msg in messages {
            let content_str = msg.content.as_deref().unwrap_or("");

            match msg.role.as_str() {
                "system" => {
                    req = req.with_system(content_str);
                }
                "user" => {
                    req = req.append_message(ChatMessage::user(content_str));
                }
                "assistant" => {
                    if let Some(ref tcs) = msg.tool_calls {
                        // Assistant message with tool calls
                        let genai_tcs: Vec<GenaiToolCall> = tcs
                            .iter()
                            .map(|tc| GenaiToolCall {
                                call_id: tc.id.clone(),
                                fn_name: tc.function.name.clone(),
                                fn_arguments: serde_json::from_str(&tc.function.arguments)
                                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                                thought_signatures: None,
                            })
                            .collect();
                        req = req.append_message(ChatMessage::assistant(
                            GenaiContent::from_tool_calls(genai_tcs),
                        ));
                    } else {
                        req = req.append_message(ChatMessage::assistant(content_str));
                    }
                }
                "tool" => {
                    let call_id = msg.tool_call_id.as_deref().unwrap_or("");
                    let tool_resp = GenaiToolResponse::new(call_id, content_str);
                    req = req.append_message(ChatMessage::from(tool_resp));
                }
                _ => {
                    req = req.append_message(ChatMessage::user(content_str));
                }
            }
        }

        // Add tools if present
        if let Some(tool_defs) = tools {
            let genai_tools: Vec<GenaiTool> = tool_defs
                .iter()
                .map(|td| {
                    let mut t = GenaiTool::new(&td.function.name);
                    if let Some(ref desc) = td.function.description {
                        t = t.with_description(desc);
                    }
                    t = t.with_schema(td.function.parameters.clone());
                    t
                })
                .collect();
            req = req.with_tools(genai_tools);
        }

        // Execute via genai
        let genai_resp = self
            .client
            .exec_chat(model, req, None)
            .await
            .map_err(|e| NpcError::LlmRequest(format!("{}", e)))?;

        // Convert genai response back to our types
        convert_genai_response(genai_resp, model)
    }
}

/// Convert a genai ChatResponse into our internal LlmResponse.
fn convert_genai_response(resp: GenaiChatResponse, model: &str) -> Result<LlmResponse> {
    let mut content_text: Option<String> = None;
    let mut tool_calls: Option<Vec<ToolCall>> = None;

    let genai_content = &resp.content;

    // Check for tool calls
    let tcs = genai_content.tool_calls();
    if !tcs.is_empty() {
        tool_calls = Some(
            tcs.iter()
                .map(|tc| ToolCall {
                    id: tc.call_id.clone(),
                    r#type: "function".to_string(),
                    function: ToolCallFunction {
                        name: tc.fn_name.clone(),
                        arguments: serde_json::to_string(&tc.fn_arguments)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                })
                .collect(),
        );
    }

    // Check for text content
    let text: Option<String> = genai_content.joined_texts();
    if let Some(ref t) = text {
        if !t.is_empty() {
            content_text = text;
        }
    }

    let usage = {
        let u = &resp.usage;
        Some(Usage {
            prompt_tokens: u.prompt_tokens.unwrap_or(0) as u64,
            completion_tokens: u.completion_tokens.unwrap_or(0) as u64,
            total_tokens: u.total_tokens.unwrap_or(0) as u64,
        })
    };

    Ok(LlmResponse {
        message: Message {
            role: "assistant".to_string(),
            content: content_text,
            tool_calls,
            tool_call_id: None,
            name: None,
        },
        usage,
        model: model.to_string(),
        finish_reason: None,
        cost_usd: None,
    })
}
