use crate::error::{NpcError, Result};
use crate::r#gen::response_types::*;

use genai::chat::{
    ChatMessage, ChatRequest, ChatResponse as GenaiChatResponse, MessageContent as GenaiContent,
    Tool as GenaiTool, ToolCall as GenaiToolCall, ToolResponse as GenaiToolResponse,
};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client as GenaiClient, Headers, ServiceTarget};

use std::sync::OnceLock;

static GENAI_CLIENT: OnceLock<GenaiClient> = OnceLock::new();

fn get_client() -> &'static GenaiClient {
    GENAI_CLIENT.get_or_init(GenaiClient::default)
}

/// Maps OpenAI-compatible provider aliases to GenAI's OpenAI adapter.
fn normalize_genai_provider(provider: &str) -> &str {
    match provider {
        "openai-compatible" | "openai-like" => "openai",
        other => other,
    }
}

/// Returns a trimmed API base URL with a trailing slash for stable URL joining.
fn normalized_api_base_url(api_url: &str) -> String {
    let trimmed = api_url.trim();
    if trimmed.ends_with('/') {
        trimmed.to_string()
    } else {
        format!("{trimmed}/")
    }
}

/// Builds the OpenAI chat-completions URL used for no-auth local endpoints.
fn openai_chat_completions_url(api_url: &str) -> Result<String> {
    let base = normalized_api_base_url(api_url);
    let parsed = reqwest::Url::parse(&base)
        .map_err(|err| NpcError::LlmRequest(format!("invalid api_url '{api_url}': {err}")))?;
    let original_query_params = parsed.query().map(str::to_owned);
    let mut full_url = parsed
        .join("chat/completions")
        .map_err(|err| NpcError::LlmRequest(format!("invalid api_url '{api_url}': {err}")))?;
    full_url.set_query(original_query_params.as_deref());
    Ok(full_url.to_string())
}

/// Creates a per-call GenAI client when an OpenAI-compatible API URL is set.
fn custom_openai_client(
    api_url_override: Option<&str>,
    api_key_override: Option<&str>,
) -> Result<Option<GenaiClient>> {
    let Some(api_url_override) = api_url_override else {
        return Ok(None);
    };

    let endpoint_base = normalized_api_base_url(api_url_override);
    let no_auth_chat_url = openai_chat_completions_url(api_url_override)?;
    let api_key = api_key_override.map(str::to_owned);

    let target_resolver = ServiceTargetResolver::from_resolver_fn(
        move |service_target: ServiceTarget| -> genai::resolver::Result<ServiceTarget> {
            let ServiceTarget { model, .. } = service_target;
            let endpoint = Endpoint::from_owned(endpoint_base.clone());
            let auth = match api_key.clone() {
                Some(key) => AuthData::from_single(key),
                None => AuthData::RequestOverride {
                    url: no_auth_chat_url.clone(),
                    headers: Headers::default(),
                },
            };
            Ok(ServiceTarget {
                endpoint,
                auth,
                model,
            })
        },
    );

    Ok(Some(
        GenaiClient::builder()
            .with_service_target_resolver(target_resolver)
            .build(),
    ))
}

/// Main entry point — routes to ollama direct API or genai based on provider.
/// Mirrors npcpy's get_litellm_response dispatch.
pub async fn get_genai_response(
    provider: &str,
    model: &str,
    messages: &[Message],
    tools: Option<&[ToolDef]>,
    api_url_override: Option<&str>,
    api_key_override: Option<&str>,
    format: Option<&str>,
    images: Option<&[String]>,
    stream: bool,
    think: Option<bool>,
) -> Result<LlmResponse> {
    if provider == "ollama" {
        return get_ollama_response(
            model,
            messages,
            tools,
            api_url_override,
            format,
            images,
            stream,
            think,
        )
        .await;
    }

    let provider = normalize_genai_provider(provider);

    // Non-ollama providers go through genai
    let custom_client = if provider == "openai" {
        custom_openai_client(api_url_override, api_key_override)?
    } else {
        None
    };

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

    // genai expects provider::model format (e.g., "moonshot::kimi-k2.5")
    let genai_model = format!("{}::{}", provider, model);
    let genai_resp = match custom_client {
        Some(client) => client.exec_chat(&genai_model, req, None).await,
        None => get_client().exec_chat(&genai_model, req, None).await,
    }
    .map_err(|e| NpcError::LlmRequest(format!("{}", e)))?;

    convert_genai_response(genai_resp, model)
}

// ---------------------------------------------------------------------------
// Direct Ollama API — mirrors npcpy's get_ollama_response
// ---------------------------------------------------------------------------

/// Ollama chat API request body
#[derive(serde::Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

/// Ollama chat API response body
#[derive(serde::Deserialize, Debug)]
struct OllamaChatResponse {
    message: Option<OllamaResponseMessage>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
}

#[derive(serde::Deserialize, Debug)]
struct OllamaResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
struct OllamaToolCall {
    function: OllamaToolCallFunction,
}

#[derive(serde::Deserialize, Debug)]
struct OllamaToolCallFunction {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

/// Direct call to Ollama REST API at /api/chat, same as npcpy's ollama.chat().
/// Mirrors npcpy/gen/response.py get_ollama_response().
async fn get_ollama_response(
    model: &str,
    messages: &[Message],
    tools: Option<&[ToolDef]>,
    api_url_override: Option<&str>,
    format: Option<&str>,
    images: Option<&[String]>,
    stream: bool,
    think: Option<bool>,
) -> Result<LlmResponse> {
    let base_url = api_url_override
        .map(|s| s.to_string())
        .or_else(|| std::env::var("OLLAMA_HOST").ok())
        .or_else(|| std::env::var("OLLAMA_API_URL").ok())
        .unwrap_or_else(|| "http://localhost:11434".to_string());
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));

    // Build options — mirrors npcpy's options dict
    let num_ctx: u64 = std::env::var("NPCSH_OLLAMA_NUM_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32768);
    let options = serde_json::json!({ "num_ctx": num_ctx });

    // Convert messages — npcpy normalizes tool_call arguments from strings to
    // objects before sending (lines 483-490 in npcpy/gen/response.py)
    let mut ollama_msgs: Vec<OllamaMessage> = messages
        .iter()
        .map(|m| {
            let tool_calls_json = m.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| {
                        // Normalize arguments: if stored as JSON string, parse to object
                        let args =
                            serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        serde_json::json!({
                            "function": {
                                "name": tc.function.name,
                                "arguments": args
                            }
                        })
                    })
                    .collect()
            });
            OllamaMessage {
                role: m.role.clone(),
                content: m.content.clone().unwrap_or_default(),
                tool_calls: tool_calls_json,
                tool_call_id: m.tool_call_id.clone(),
                images: None,
                thinking: None,
                reasoning_content: None,
            }
        })
        .collect();

    // Attach images to last user message — mirrors npcpy lines 473-481
    if let Some(imgs) = images {
        if !imgs.is_empty() {
            if let Some(last_user) = ollama_msgs.iter_mut().rev().find(|m| m.role == "user") {
                last_user.images = Some(imgs.to_vec());
            }
        }
    }

    // Convert tools — same structure as npcpy's tools list
    let ollama_tools = tools.map(|tds| {
        tds.iter()
            .map(|td| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": td.function.name,
                        "description": td.function.description,
                        "parameters": td.function.parameters,
                    }
                })
            })
            .collect::<Vec<_>>()
    });

    // format param — npcpy passes "json" string or pydantic schema to ollama
    let format_value = match format {
        Some("json") if !stream => Some(serde_json::json!("json")),
        _ => None,
    };

    // json format instruction — mirrors npcpy lines 440-454
    if format == Some("json") && !stream {
        let json_instruction = "If you are returning a json object, begin directly with the opening {.\n\
            If you are returning a json array, begin directly with the opening [.\n\
            Do not include any additional markdown formatting or leading ```json tags in your response. \
            The item keys should be based on the ones provided by the user. Do not invent new ones.";
        if let Some(last_user) = ollama_msgs.iter_mut().rev().find(|m| m.role == "user") {
            last_user.content.push('\n');
            last_user.content.push_str(json_instruction);
        }
    }

    // yaml format instruction — mirrors npcpy lines 456-471
    if format == Some("yaml") && !stream {
        let yaml_instruction = "Return your response as valid YAML. Do not include ```yaml markdown tags.\n\
            For multi-line strings like code, use the literal block scalar (|) syntax:\n\
            code: |\n  your code here\n  more lines here\n\
            The keys should be based on the ones requested by the user. Do not invent new ones.";
        if let Some(last_user) = ollama_msgs.iter_mut().rev().find(|m| m.role == "user") {
            last_user.content.push('\n');
            last_user.content.push_str(yaml_instruction);
        }
    }

    // think param — use caller's value, or auto-detect for reasoning models
    let think_val = think.or_else(|| {
        if model.contains("deepseek-r1") || model.contains("qwq") {
            Some(true)
        } else {
            None
        }
    });

    let body = OllamaChatRequest {
        model,
        messages: ollama_msgs,
        stream,
        tools: ollama_tools,
        options: Some(options),
        format: format_value,
        think: think_val,
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await
        .map_err(|e| NpcError::LlmRequest(format!("Ollama request to {} failed: {}", url, e)))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        return Err(NpcError::LlmRequest(format!(
            "Ollama API returned {}: {}",
            status, body_text
        )));
    }

    let ollama_resp: OllamaChatResponse = resp
        .json()
        .await
        .map_err(|e| NpcError::LlmRequest(format!("Failed to parse Ollama response: {}", e)))?;

    // Convert to LlmResponse
    let msg = ollama_resp.message.unwrap_or(OllamaResponseMessage {
        content: String::new(),
        tool_calls: None,
        thinking: None,
        reasoning_content: None,
    });

    let content_text = if msg.content.is_empty() {
        None
    } else {
        Some(msg.content)
    };

    let tool_calls = msg.tool_calls.map(|tcs| {
        tcs.into_iter()
            .enumerate()
            .map(|(i, tc)| ToolCall {
                id: format!("call_{}", i),
                r#type: "function".to_string(),
                function: ToolCallFunction {
                    name: tc.function.name,
                    arguments: serde_json::to_string(&tc.function.arguments)
                        .unwrap_or_else(|_| "{}".to_string()),
                },
            })
            .collect()
    });

    let usage = Some(Usage {
        prompt_tokens: ollama_resp.prompt_eval_count.unwrap_or(0),
        completion_tokens: ollama_resp.eval_count.unwrap_or(0),
        total_tokens: ollama_resp.prompt_eval_count.unwrap_or(0)
            + ollama_resp.eval_count.unwrap_or(0),
    });

    Ok(LlmResponse {
        message: Message {
            role: "assistant".to_string(),
            content: content_text,
            tool_calls,
            tool_call_id: None,
            name: None,
            thinking: None,
            reasoning_content: None,
        },
        usage,
        model: model.to_string(),
        finish_reason: Some("stop".to_string()),
        cost_usd: None,
    })
}

// ---------------------------------------------------------------------------
// genai response conversion (for non-ollama providers)
// ---------------------------------------------------------------------------

fn convert_genai_response(resp: GenaiChatResponse, model: &str) -> Result<LlmResponse> {
    let mut content_text: Option<String> = None;
    let mut tool_calls: Option<Vec<ToolCall>> = None;

    let genai_content = &resp.content;

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
            thinking: None,
            reasoning_content: None,
        },
        usage,
        model: model.to_string(),
        finish_reason: None,
        cost_usd: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    #[derive(Debug)]
    struct ObservedRequest {
        path: String,
        authorization: Option<String>,
        body: Value,
    }

    async fn spawn_openai_compatible_server() -> (String, oneshot::Receiver<ObservedRequest>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = Vec::new();
            let mut chunk = [0_u8; 1024];
            let headers_end = loop {
                let read = socket.read(&mut chunk).await.unwrap();
                if read == 0 {
                    panic!("client closed before sending request headers");
                }
                buffer.extend_from_slice(&chunk[..read]);
                if let Some(pos) = find_headers_end(&buffer) {
                    break pos;
                }
            };

            let headers_text = String::from_utf8_lossy(&buffer[..headers_end]).to_string();
            let content_length = content_length(&headers_text);
            let body_start = headers_end + 4;
            while buffer.len() < body_start + content_length {
                let read = socket.read(&mut chunk).await.unwrap();
                if read == 0 {
                    break;
                }
                buffer.extend_from_slice(&chunk[..read]);
            }

            let request_line = headers_text.lines().next().unwrap_or_default();
            let path = request_line
                .split_whitespace()
                .nth(1)
                .unwrap_or_default()
                .to_string();
            let authorization = headers_text.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("authorization") {
                    Some(value.trim().to_string())
                } else {
                    None
                }
            });
            let body: Value =
                serde_json::from_slice(&buffer[body_start..body_start + content_length]).unwrap();

            let _ = tx.send(ObservedRequest {
                path,
                authorization,
                body,
            });

            let response_body = serde_json::json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "patched npcrs"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3
                }
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        (format!("http://{addr}/v1"), rx)
    }

    fn find_headers_end(buffer: &[u8]) -> Option<usize> {
        buffer.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn content_length(headers: &str) -> usize {
        headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    value.trim().parse().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0)
    }

    #[test]
    fn normalized_api_base_url_trims_and_adds_trailing_slash() {
        assert_eq!(normalized_api_base_url("http://host/v1"), "http://host/v1/");
        assert_eq!(
            normalized_api_base_url("http://host/v1/"),
            "http://host/v1/"
        );
        assert_eq!(
            normalized_api_base_url("  http://host/v1  "),
            "http://host/v1/"
        );
    }

    #[tokio::test]
    async fn openai_compatible_uses_api_url_without_auth() {
        let (api_url, observed) = spawn_openai_compatible_server().await;
        let response = get_genai_response(
            "openai-compatible",
            "test-model",
            &[Message::user("hello")],
            None,
            Some(&api_url),
            None,
            None,
            None,
            false,
            None,
        )
        .await
        .unwrap();

        let request = observed.await.unwrap();
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.authorization, None);
        assert_eq!(request.body["model"], "test-model");
        assert_eq!(
            request.body["messages"][0]["content"],
            serde_json::json!("hello")
        );
        assert_eq!(response.message.content.as_deref(), Some("patched npcrs"));
    }

    #[tokio::test]
    async fn openai_like_uses_api_url_and_api_key() {
        let (api_url, observed) = spawn_openai_compatible_server().await;
        let response = get_genai_response(
            "openai-like",
            "test-model",
            &[Message::user("hello")],
            None,
            Some(&api_url),
            Some("test-key"),
            None,
            None,
            false,
            None,
        )
        .await
        .unwrap();

        let request = observed.await.unwrap();
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.authorization.as_deref(), Some("Bearer test-key"));
        assert_eq!(request.body["model"], "test-model");
        assert_eq!(response.message.content.as_deref(), Some("patched npcrs"));
    }
}
