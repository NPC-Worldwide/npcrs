//! High-level LLM functions — response matrix, command checking, model resolution.
//!
//! This module provides the primary interface for getting LLM responses with
//! full NPC context, command type detection, and cost tracking.

use crate::error::{NpcError, Result};
#[allow(unused_imports)]
use crate::r#gen::{LlmResponse, Message, ToolDef, ToolCall};
use crate::npc_compiler::Npc;
use std::collections::HashMap;

/// Get an LLM response with full NPC context.
///
/// This is the primary interface for getting responses — handles system prompt,
/// tool resolution, message history, and cost tracking.
/// No client parameter needed — uses the global standalone chat_completion.
pub async fn get_llm_response(
    input: &str,
    npc: Option<&Npc>,
    model: Option<&str>,
    provider: Option<&str>,
    tools: Option<&[ToolDef]>,
    messages: &[Message],
    team_context: Option<&str>,
) -> Result<LlmResponseResult> {
    // Resolve model/provider from NPC or defaults
    let (resolved_model, resolved_provider) = if let Some(npc) = npc {
        (
            model
                .map(String::from)
                .unwrap_or_else(|| npc.resolved_model()),
            provider
                .map(String::from)
                .unwrap_or_else(|| npc.resolved_provider()),
        )
    } else {
        let m = model.unwrap_or("qwen3.5:2b");
        let p = provider.unwrap_or("ollama");
        (m.to_string(), p.to_string())
    };

    // Build system prompt
    let system_prompt = if let Some(npc) = npc {
        npc.system_prompt(team_context)
    } else {
        "You are a helpful assistant.".to_string()
    };

    // Build messages
    let mut full_messages = vec![Message::system(&system_prompt)];
    full_messages.extend_from_slice(messages);
    full_messages.push(Message::user(input));

    // Sanitize
    let clean = crate::r#gen::sanitize::sanitize_messages(full_messages);

    // Dispatch: local GGUF or remote API
    let response = {
        #[cfg(feature = "llamacpp")]
        {
            if resolved_provider == "llamacpp" || resolved_model.ends_with(".gguf") {
                let model_path = resolved_model.clone();
                let msgs = clean.clone();
                tokio::task::spawn_blocking(move || {
                    crate::r#gen::get_llamacpp_response(&model_path, &msgs, 512, 0.7, 4096, -1)
                })
                .await
                .map_err(|e| crate::error::NpcError::LlmRequest(format!("spawn_blocking: {}", e)))??
            } else {
                crate::r#gen::get_genai_response(
                    &resolved_provider, &resolved_model, &clean, tools,
                    npc.and_then(|n| n.api_url.as_deref()),
                ).await?
            }
        }
        #[cfg(not(feature = "llamacpp"))]
        {
            if resolved_model.ends_with(".gguf") {
                return Err(crate::error::NpcError::LlmRequest(
                    "Local GGUF inference requires the 'llamacpp' feature. Build with: cargo build --features llamacpp".into()
                ));
            }
            crate::r#gen::get_genai_response(
                &resolved_provider, &resolved_model, &clean, tools,
                npc.and_then(|n| n.api_url.as_deref()),
            ).await?
        }
    };

    // Build result matching npcpy's return dict
    let usage_info = response.usage.as_ref().map(|u| UsageInfo {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
    });

    let cost = response.usage.as_ref().map(|u| {
        crate::r#gen::cost::calculate_cost(&resolved_model, u.prompt_tokens, u.completion_tokens)
    }).unwrap_or(0.0);

    let response_text = response.message.content.clone();
    let tool_calls = response.message.tool_calls.clone().unwrap_or_default();

    // Append assistant message to messages (like npcpy does)
    let mut updated_messages = clean;
    updated_messages.push(response.message);

    Ok(LlmResponseResult {
        response: response_text,
        messages: updated_messages,
        tool_calls,
        tool_results: Vec::new(),
        usage: usage_info,
        model: resolved_model,
        provider: resolved_provider,
        cost_usd: cost,
        error: None,
    })
}

/// Result from get_llm_response — mirrors npcpy's return dict exactly.
///
/// npcpy returns: {"response", "messages", "tool_calls", "tool_results", "usage", "raw_response", "error"}
pub struct LlmResponseResult {
    /// Text response content (None if streaming or error).
    pub response: Option<String>,
    /// Updated message list with assistant response appended.
    pub messages: Vec<Message>,
    /// Tool calls from the model response.
    pub tool_calls: Vec<ToolCall>,
    /// Tool execution results.
    pub tool_results: Vec<String>,
    /// Token usage: input_tokens, output_tokens.
    pub usage: Option<UsageInfo>,
    /// The model that was used.
    pub model: String,
    /// The provider that was used.
    pub provider: String,
    /// Estimated cost in USD.
    pub cost_usd: f64,
    /// Error message if failed.
    pub error: Option<String>,
}

/// Token usage info — mirrors npcpy's usage dict.
pub struct UsageInfo {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Check if user input should be handled as a command, jinx, or LLM query.
/// Returns the command type and any extracted data.
pub fn check_llm_command(input: &str) -> CommandType {
    let trimmed = input.trim();

    if trimmed.is_empty() {
        return CommandType::Empty;
    }

    // Slash commands -> jinx execution
    if trimmed.starts_with('/') {
        let parts: Vec<&str> = trimmed[1..].splitn(2, ' ').collect();
        return CommandType::Jinx {
            name: parts[0].to_string(),
            args: parts.get(1).unwrap_or(&"").to_string(),
        };
    }

    // @npc delegation
    if trimmed.starts_with('@') {
        let parts: Vec<&str> = trimmed[1..].splitn(2, ' ').collect();
        return CommandType::Delegate {
            npc_name: parts[0].to_string(),
            message: parts.get(1).unwrap_or(&"").to_string(),
        };
    }

    // Bash command detection
    if is_likely_bash(trimmed) {
        return CommandType::Bash(trimmed.to_string());
    }

    // Default: LLM query
    CommandType::LlmQuery(trimmed.to_string())
}

/// Types of commands the shell can receive.
#[derive(Debug, Clone, PartialEq)]
pub enum CommandType {
    /// Empty input (no-op).
    Empty,
    /// Slash command to execute a jinx.
    Jinx { name: String, args: String },
    /// Delegation to another NPC via @name.
    Delegate { npc_name: String, message: String },
    /// Likely a bash/shell command.
    Bash(String),
    /// Free-form text to send to the LLM.
    LlmQuery(String),
}

/// Simple heuristic to detect likely bash commands.
fn is_likely_bash(input: &str) -> bool {
    let first_word = input.split_whitespace().next().unwrap_or("");
    matches!(
        first_word,
        "ls" | "cd"
            | "pwd"
            | "cat"
            | "grep"
            | "find"
            | "mkdir"
            | "rm"
            | "cp"
            | "mv"
            | "echo"
            | "touch"
            | "chmod"
            | "head"
            | "tail"
            | "wc"
            | "sort"
            | "curl"
            | "wget"
            | "git"
            | "docker"
            | "make"
            | "cargo"
            | "npm"
            | "pip"
            | "python"
            | "python3"
            | "node"
    )
}

// ── Internal LLM call helpers ──

async fn llm_call(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>) -> Result<String> {
    let result = get_llm_response(prompt, npc, model, provider, None, &[], None).await?;
    Ok(result.response.unwrap_or_default())
}

async fn llm_call_json(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>) -> Result<serde_json::Value> {
    let text = llm_call(prompt, model, provider, npc).await?;
    let clean = text.replace("```json", "").replace("```", "").trim().to_string();
    serde_json::from_str(&clean).map_err(|e| NpcError::Shell(format!("JSON parse error: {}", e)))
}

// ── execute_llm_command ──

pub async fn execute_llm_command(command: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, messages: &mut Vec<Message>) -> Result<LlmResponseResult> {
    for _attempt in 0..5 {
        let prompt = format!("A user submitted this query: {}.\nGenerate a bash command to accomplish the user's intent.\nRespond ONLY with the bash command. No markdown.", command);
        let result = get_llm_response(&prompt, npc, model, provider, None, messages, None).await?;
        let bash_command = result.response.clone().unwrap_or_default();
        let run = std::process::Command::new("sh").args(["-c", &bash_command]).output();
        match run {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let explain = format!("Output of {} (ran {}):\n{}\nExplain what was done.", command, bash_command, stdout);
                messages.push(Message::user(&explain));
                return get_llm_response(&explain, npc, model, provider, None, messages, None).await;
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let err_prompt = format!("Command '{}' failed: {}\nSuggest a fix as JSON: {{\"bash_command\": \"...\"}}", bash_command, stderr);
                let _ = llm_call_json(&err_prompt, model, provider, npc).await;
            }
            Err(_) => {}
        }
    }
    Ok(LlmResponseResult { response: Some("Max attempts reached.".into()), messages: messages.clone(), tool_calls: vec![], tool_results: vec![], usage: None, model: model.unwrap_or("").into(), provider: provider.unwrap_or("").into(), cost_usd: 0.0, error: None })
}

// ── handle_request_input ──

pub async fn handle_request_input(context: &str, model: &str, provider: &str) -> Result<serde_json::Value> {
    let prompt = format!("Analyze the text:\n{}\nDetermine what additional input is needed.\nReturn JSON: {{\"input_needed\": bool, \"request_reason\": str, \"request_prompt\": str}}", context);
    llm_call_json(&prompt, Some(model), Some(provider), None).await
}

// ── handle_jinx_call ──

pub async fn handle_jinx_call(command: &str, jinx_name: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, messages: &[Message], context: Option<&str>) -> Result<HashMap<String, serde_json::Value>> {
    let prompt = format!("The user wants jinx '{}' with request: '{}'\nDetermine required inputs as JSON.", jinx_name, command);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    let mut output = HashMap::new();
    output.insert("jinx_name".into(), serde_json::Value::String(jinx_name.to_string()));
    output.insert("inputs".into(), result);
    output.insert("command".into(), serde_json::Value::String(command.to_string()));
    Ok(output)
}

// ── handle_action_choice ──

pub async fn handle_action_choice(command: &str, action_data: &serde_json::Value, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, messages: &[Message], context: Option<&str>) -> Result<HashMap<String, serde_json::Value>> {
    let action_name = action_data.get("action").and_then(|v| v.as_str()).unwrap_or("answer");
    let mut result = HashMap::new();
    if action_name == "invoke_jinx" || action_data.get("jinx_name").is_some() {
        let jname = action_data.get("jinx_name").and_then(|v| v.as_str()).unwrap_or("");
        let jr = handle_jinx_call(command, jname, model, provider, npc, messages, context).await?;
        let map: serde_json::Map<String, serde_json::Value> = jr.into_iter().collect();
        result.insert("output".into(), serde_json::Value::Object(map));
    } else if action_name == "answer" {
        let prompt = format!("The user asked: {}\nProvide a direct answer.", command);
        let response = llm_call(&prompt, model, provider, npc).await?;
        result.insert("output".into(), serde_json::Value::String(response));
    } else {
        result.insert("output".into(), serde_json::Value::String("INVALID_ACTION".into()));
    }
    Ok(result)
}

// ── gen_image / gen_video ──

pub async fn gen_image(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, width: u32, height: u32, api_key: Option<&str>) -> Result<crate::r#gen::GeneratedImage> {
    let (m, p) = resolve_model_provider_for_gen(model, provider, npc);
    crate::r#gen::generate_image(prompt, &m, &p, api_key, width, height).await
}

pub async fn gen_video(prompt: &str, model: Option<&str>, provider: Option<&str>, _npc: Option<&Npc>, output_path: &str) -> Result<HashMap<String, String>> {
    let model_str = model.unwrap_or("veo-3.1-fast-generate-preview");
    let provider_str = provider.unwrap_or("gemini");
    let mut result = HashMap::new();

    if provider_str == "gemini" {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .map_err(|_| NpcError::LlmRequest("GOOGLE_API_KEY not set for video gen".into()))?;
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model_str, api_key
        );
        let body = serde_json::json!({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["video"]}
        });
        let client = reqwest::Client::new();
        let resp = client.post(&url).json(&body).send().await?;
        if resp.status().is_success() {
            let data: serde_json::Value = resp.json().await?;
            // Extract video data from response
            if let Some(b64) = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"].as_str() {
                use base64::Engine;
                let bytes = base64::engine::general_purpose::STANDARD.decode(b64)
                    .map_err(|e| NpcError::Generation(format!("Base64 decode: {}", e)))?;
                std::fs::write(output_path, &bytes)
                    .map_err(|e| NpcError::Generation(format!("Write video: {}", e)))?;
                result.insert("output".into(), format!("Video generated at {}", output_path));
            } else {
                result.insert("output".into(), "No video data in response".into());
            }
        } else {
            let text = resp.text().await.unwrap_or_default();
            result.insert("output".into(), format!("Video gen failed: {}", &text[..text.len().min(200)]));
        }
    } else {
        result.insert("output".into(), format!("Video generation not supported for provider '{}' in Rust. Use gemini.", provider_str));
    }

    Ok(result)
}

fn resolve_model_provider_for_gen(model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>) -> (String, String) {
    if let (Some(m), Some(p)) = (model, provider) { return (m.to_string(), p.to_string()); }
    if let Some(npc) = npc { return (model.map(String::from).unwrap_or_else(|| npc.resolved_model()), provider.map(String::from).unwrap_or_else(|| npc.resolved_provider())); }
    (model.unwrap_or("dall-e-3").to_string(), provider.unwrap_or("openai").to_string())
}

// ── KG Functions ──

pub async fn breathe(messages: &[Message], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>) -> Result<HashMap<String, serde_json::Value>> {
    if messages.is_empty() { let mut r = HashMap::new(); r.insert("output".into(), serde_json::json!({})); r.insert("messages".into(), serde_json::json!([])); return Ok(r); }
    let conv: String = messages.iter().filter_map(|m| m.content.as_ref().map(|c| format!("{}: {}", m.role, c))).collect::<Vec<_>>().join("\n");
    let prompt = format!("Read this conversation:\n{}\n\nIdentify: 1) high level objective 2) most recent task 3) accomplishments 4) failures\nReturn JSON: {{\"high_level_objective\": str, \"most_recent_task\": str, \"accomplishments\": [str], \"failures\": [str]}}", conv);
    let res = llm_call_json(&prompt, model, provider, npc).await?;
    let fmt = format!("Summary: objective={}, accomplishments={}, failures={}, recent_task={}", res.get("high_level_objective").and_then(|v| v.as_str()).unwrap_or("?"), res.get("accomplishments").unwrap_or(&serde_json::json!([])), res.get("failures").unwrap_or(&serde_json::json!([])), res.get("most_recent_task").and_then(|v| v.as_str()).unwrap_or("?"));
    let mut r = HashMap::new();
    r.insert("output".into(), serde_json::Value::String(fmt.clone()));
    r.insert("summary".into(), res);
    r.insert("messages".into(), serde_json::json!([{"role": "assistant", "content": fmt}]));
    Ok(r)
}

pub async fn orchestrate(prompt: &str, items: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, workflow: &str, context: Option<&str>) -> Result<String> {
    let items_text = items.iter().enumerate().map(|(i, s)| format!("{}. {}", i + 1, s)).collect::<Vec<_>>().join("\n");
    llm_call(&format!("Orchestrate using {}:\nTask: {}\nItems: {}\nContext: {}", workflow, prompt, items_text, context.unwrap_or("none")), model, provider, npc).await
}

pub async fn bootstrap(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, n_samples: usize, context: Option<&str>) -> Result<String> {
    let mut results = Vec::new();
    for i in 0..n_samples { results.push(llm_call(&format!("Sample {}: {}\nContext: {}", i + 1, prompt, context.unwrap_or("none")), model, provider, npc).await?); }
    synthesize(&results.join("\n---\n"), model, provider, npc, context).await
}

pub async fn harmonize(prompt: &str, items: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, harmony_rules: Option<&[String]>, context: Option<&str>) -> Result<String> {
    let items_text = items.iter().enumerate().map(|(i, s)| format!("{}. {}", i + 1, s)).collect::<Vec<_>>().join("\n");
    let rules = harmony_rules.map(|r| r.join(", ")).unwrap_or_else(|| "maintain_consistency".into());
    llm_call(&format!("Harmonize: {}\nTask: {}\nRules: {}\nContext: {}", items_text, prompt, rules, context.unwrap_or("none")), model, provider, npc).await
}

pub async fn spread_and_sync(prompt: &str, variations: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, sync_strategy: &str, context: Option<&str>) -> Result<String> {
    let mut results = Vec::new();
    for v in variations { results.push(llm_call(&format!("Analyze from {} perspective:\nTask: {}\nContext: {}", v, prompt, context.unwrap_or("none")), model, provider, npc).await?); }
    let combined = results.iter().enumerate().map(|(i, r)| format!("Response {}: {}", i + 1, r)).collect::<Vec<_>>().join("\n\n");
    llm_call(&format!("Synthesize perspectives:\n{}\nStrategy: {}\nContext: {}", combined, sync_strategy, context.unwrap_or("none")), model, provider, npc).await
}

pub async fn criticize(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<String> {
    llm_call(&format!("Critical analysis of:\n{}\nIdentify weaknesses, improvements, alternatives.\nContext: {}", prompt, context.unwrap_or("none")), model, provider, npc).await
}

pub async fn synthesize(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<String> {
    llm_call(&format!("Synthesize:\n{}\nContext: {}\nCreate a clear, concise synthesis.", prompt, context.unwrap_or("none")), model, provider, npc).await
}

pub async fn get_facts(content_text: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<serde_json::Value>> {
    let prompt = format!("Extract facts from text: \"{}\"\nRespond JSON: {{\"facts\": [{{\"statement\": str, \"source_text\": str, \"type\": \"explicit or inferred\"}}]}}", content_text);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("facts").and_then(|f| f.as_array()).cloned().unwrap_or_default())
}

pub async fn zoom_in(facts: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<serde_json::Value>> {
    let lines: Vec<String> = facts.iter().filter_map(|f| f.get("statement").and_then(|s| s.as_str())).map(|s| format!("- {}", s)).collect();
    if lines.is_empty() { return Ok(vec![]); }
    let prompt = format!("Look at these facts and infer new implied facts:\n{}\nRespond JSON: {{\"implied_facts\": [{{\"statement\": str, \"inferred_from\": [str]}}]}}", lines.join("\n"));
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("implied_facts").and_then(|f| f.as_array()).cloned().unwrap_or_default())
}

pub async fn generate_groups(facts: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<serde_json::Value>> {
    let text: String = facts.iter().filter_map(|f| f.get("statement").and_then(|s| s.as_str())).map(|s| format!("- {}", s)).collect::<Vec<_>>().join("\n");
    let prompt = format!("Generate conceptual groups for facts:\n{}\nRespond JSON: {{\"groups\": [{{\"name\": str}}]}}", text);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("groups").and_then(|g| g.as_array()).cloned().unwrap_or_default())
}

pub async fn r#abstract(groups: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<serde_json::Value>> {
    let text: String = groups.iter().filter_map(|g| g.get("name").and_then(|n| n.as_str())).map(|n| format!("- \"{}\"", n)).collect::<Vec<_>>().join("\n");
    let prompt = format!("Create abstract categories from groups:\n{}\nRespond JSON: {{\"groups\": [{{\"name\": str}}]}}", text);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("groups").and_then(|g| g.as_array()).cloned().unwrap_or_default())
}

pub async fn remove_redundant_groups(groups: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<serde_json::Value>> {
    let text: String = groups.iter().filter_map(|g| g.get("name").and_then(|n| n.as_str())).map(|n| format!("- {}", n)).collect::<Vec<_>>().join("\n");
    let prompt = format!("Remove redundant groups:\n{}\nRespond JSON: {{\"groups\": [{{\"name\": str}}]}}", text);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("groups").and_then(|g| g.as_array()).cloned().unwrap_or_default())
}

pub async fn prune_fact_subset_llm(fact_subset: &[serde_json::Value], concept_name: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<serde_json::Value>> {
    let facts_json = serde_json::to_string_pretty(fact_subset).unwrap_or_default();
    let prompt = format!("Facts related to \"{}\":\n{}\nReturn semantically distinct facts as JSON: {{\"refined_facts\": [...]}}", concept_name, facts_json);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("refined_facts").and_then(|f| f.as_array()).cloned().unwrap_or_default())
}

pub async fn consolidate_facts_llm(new_fact: &serde_json::Value, existing_facts: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<serde_json::Value> {
    let new_stmt = new_fact.get("statement").and_then(|s| s.as_str()).unwrap_or("");
    let existing: Vec<&str> = existing_facts.iter().filter_map(|f| f.get("statement").and_then(|s| s.as_str())).collect();
    let prompt = format!("New Fact: \"{}\"\nExisting: {:?}\nDecide: novel or redundant.\nJSON: {{\"decision\": str, \"reason\": str}}", new_stmt, existing);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    llm_call_json(&full, model, provider, npc).await
}

pub async fn get_related_facts_llm(new_fact_statement: &str, existing_fact_statements: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let prompt = format!("New fact: \"{}\"\nWhich existing facts are related?\nExisting: {:?}\nJSON: {{\"related_facts\": [str]}}", new_fact_statement, existing_fact_statements);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("related_facts").and_then(|f| f.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default())
}

pub async fn find_best_link_concept_llm(candidate: &str, existing: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Option<String>> {
    let prompt = format!("New concept: \"{}\"\nExisting: {:?}\nBest link? JSON: {{\"best_link_concept\": \"name OR none\"}}", candidate, existing);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("best_link_concept").and_then(|v| v.as_str()).map(String::from).filter(|v| v.to_lowercase() != "none"))
}

pub async fn asymptotic_freedom(parent_concept_name: &str, supporting_facts: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let stmts: Vec<&str> = supporting_facts.iter().filter_map(|f| f.get("statement").and_then(|s| s.as_str())).collect();
    let prompt = format!("Concept \"{}\" has many facts. Propose 2-4 sub-concepts.\nFacts: {:?}\nJSON: {{\"new_sub_concepts\": [str]}}", parent_concept_name, stmts);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("new_sub_concepts").and_then(|f| f.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default())
}

pub async fn identify_groups(facts: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let prompt = format!("What groups could these facts be organized into?\nFacts: {:?}\nJSON: {{\"groups\": [str]}}", facts);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("groups").and_then(|g| g.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default())
}

pub async fn get_related_concepts_multi(node_name: &str, node_type: &str, all_concept_names: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let prompt = format!("Which concepts relate to {} \"{}\"?\nAvailable: {:?}\nJSON: {{\"related_concepts\": [str]}}", node_type, node_name, all_concept_names);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("related_concepts").and_then(|c| c.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default())
}

pub async fn assign_groups_to_fact(fact: &str, groups: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let prompt = format!("Assign fact to groups.\nFact: {}\nGroups: {:?}\nJSON: {{\"groups\": [str]}}", fact, groups);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("groups").and_then(|g| g.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default())
}

pub async fn generate_group_candidates(items: &[String], item_type: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>, n_passes: usize, subset_size: usize) -> Result<Vec<String>> {
    let mut all = Vec::new();
    for _pass in 0..n_passes {
        let subset: Vec<&String> = if items.len() > subset_size { items.iter().take(subset_size).collect() } else { items.iter().collect() };
        let prompt = format!("From {} items, identify conceptual groups:\n{:?}\nJSON: {{\"groups\": [str]}}", item_type, subset);
        let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
        if let Ok(result) = llm_call_json(&full, model, provider, npc).await {
            if let Some(groups) = result.get("groups").and_then(|g| g.as_array()) {
                for g in groups { if let Some(s) = g.as_str() { if !all.contains(&s.to_string()) { all.push(s.to_string()); } } }
            }
        }
    }
    Ok(all)
}

pub async fn remove_idempotent_groups(group_candidates: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let prompt = format!("Keep only conceptually distinct groups:\n{:?}\nJSON: {{\"distinct_groups\": [str]}}", group_candidates);
    let full = if let Some(ctx) = context { format!("{}\nContext: {}", prompt, ctx) } else { prompt };
    let result = llm_call_json(&full, model, provider, npc).await?;
    Ok(result.get("distinct_groups").and_then(|g| g.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_command_type_empty() {
        assert_eq!(check_llm_command(""), CommandType::Empty);
        assert_eq!(check_llm_command("  "), CommandType::Empty);
    }

    #[test]
    fn test_check_command_type_jinx() {
        match check_llm_command("/search hello world") {
            CommandType::Jinx { name, args } => {
                assert_eq!(name, "search");
                assert_eq!(args, "hello world");
            }
            other => panic!("Expected Jinx, got {:?}", other),
        }
    }

    #[test]
    fn test_check_command_type_jinx_no_args() {
        match check_llm_command("/help") {
            CommandType::Jinx { name, args } => {
                assert_eq!(name, "help");
                assert_eq!(args, "");
            }
            other => panic!("Expected Jinx, got {:?}", other),
        }
    }

    #[test]
    fn test_check_command_type_delegate() {
        match check_llm_command("@corca what is the weather") {
            CommandType::Delegate { npc_name, message } => {
                assert_eq!(npc_name, "corca");
                assert_eq!(message, "what is the weather");
            }
            other => panic!("Expected Delegate, got {:?}", other),
        }
    }

    #[test]
    fn test_check_command_type_bash() {
        assert!(matches!(
            check_llm_command("ls -la"),
            CommandType::Bash(_)
        ));
        assert!(matches!(
            check_llm_command("git status"),
            CommandType::Bash(_)
        ));
        assert!(matches!(
            check_llm_command("cargo build"),
            CommandType::Bash(_)
        ));
    }

    #[test]
    fn test_check_command_type_llm_query() {
        assert!(matches!(
            check_llm_command("what is the meaning of life"),
            CommandType::LlmQuery(_)
        ));
        assert!(matches!(
            check_llm_command("explain quantum computing"),
            CommandType::LlmQuery(_)
        ));
    }
}
