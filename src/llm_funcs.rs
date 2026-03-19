
use crate::error::{NpcError, Result};
#[allow(unused_imports)]
use crate::r#gen::{LlmResponse, Message, ToolDef, ToolCall};
use crate::npc_compiler::{Npc, Jinx, JinxInput};
use std::collections::HashMap;

pub struct LlmResponseResult {
    pub response: Option<String>,
    pub response_json: Option<serde_json::Value>,
    pub messages: Vec<Message>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<String>,
    pub usage: Option<UsageInfo>,
    pub model: String,
    pub provider: String,
    pub cost_usd: f64,
    pub error: Option<String>,
}

pub struct UsageInfo {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

pub fn resolve_model_provider(
    npc: Option<&Npc>,
    model: Option<&str>,
    provider: Option<&str>,
) -> (String, String) {
    if let (Some(m), Some(p)) = (model, provider) {
        return (m.to_string(), p.to_string());
    }
    if provider.is_none() && model.is_some() {
        let m = model.unwrap();
        let p = lookup_provider(m);
        return (m.to_string(), p);
    }
    if let Some(npc) = npc {
        return (
            model.map(String::from).unwrap_or_else(|| npc.resolved_model()),
            provider.map(String::from).unwrap_or_else(|| npc.resolved_provider()),
        );
    }
    ("llama3.2".to_string(), "ollama".to_string())
}

fn lookup_provider(model: &str) -> String {
    let m = model.to_lowercase();
    if m.starts_with("gpt-") || m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4") || m.contains("dall-e") || m.starts_with("gpt-image") {
        "openai".into()
    } else if m.starts_with("claude") {
        "anthropic".into()
    } else if m.starts_with("gemini") || m.starts_with("gemma") || m.starts_with("veo") {
        "gemini".into()
    } else if m.starts_with("deepseek") {
        "deepseek".into()
    } else if m.contains(":") || m.starts_with("llama") || m.starts_with("qwen") || m.starts_with("mistral") || m.starts_with("phi") || m.starts_with("llava") {
        "ollama".into()
    } else {
        "ollama".into()
    }
}

pub async fn get_llm_response(
    input: &str,
    npc: Option<&Npc>,
    model: Option<&str>,
    provider: Option<&str>,
    tools: Option<&[ToolDef]>,
    messages: &[Message],
    team_context: Option<&str>,
) -> Result<LlmResponseResult> {
    get_llm_response_ext(
        input, npc, model, provider, tools, messages,
        team_context, None, None, false,
    ).await
}

pub async fn get_llm_response_ext(
    input: &str,
    npc: Option<&Npc>,
    model: Option<&str>,
    provider: Option<&str>,
    tools: Option<&[ToolDef]>,
    messages: &[Message],
    team_context: Option<&str>,
    format: Option<&str>,
    context: Option<&str>,
    _stream: bool,
) -> Result<LlmResponseResult> {
    let (resolved_model, resolved_provider) = resolve_model_provider(npc, model, provider);

    let system_prompt = if let Some(npc) = npc {
        npc.system_prompt(team_context)
    } else {
        "You are a helpful assistant.".to_string()
    };

    let full_text = match (input.is_empty(), context) {
        (false, Some(ctx)) => format!("{}\n\n\nUser Provided Context: {}", input, ctx),
        (false, None) => input.to_string(),
        (true, Some(ctx)) => format!("User Provided Context: {}", ctx),
        (true, None) => String::new(),
    };

    let mut full_messages = vec![Message::system(&system_prompt)];
    full_messages.extend_from_slice(messages);

    if !full_text.is_empty() {
        if full_messages.last().map(|m| m.role.as_str()) == Some("user") {
            if let Some(last) = full_messages.last_mut() {
                if let Some(ref mut c) = last.content {
                    c.push('\n');
                    c.push_str(&full_text);
                }
            }
        } else {
            full_messages.push(Message::user(&full_text));
        }
    }

    if format == Some("json") {
        let json_instruction = "If you are returning a json object, begin directly with the opening {.\n\
            If you are returning a json array, begin directly with the opening [.\n\
            Do not include any additional markdown formatting or leading ```json tags in your response. \
            The item keys should be based on the ones provided by the user. Do not invent new ones.";
        if let Some(last) = full_messages.iter_mut().rev().find(|m| m.role == "user") {
            if let Some(ref mut c) = last.content {
                c.push('\n');
                c.push_str(json_instruction);
            }
        }
    }

    let clean = crate::r#gen::sanitize::sanitize_messages(full_messages);

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
                .map_err(|e| NpcError::LlmRequest(format!("spawn_blocking: {}", e)))??
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
                return Err(NpcError::LlmRequest(
                    "Local GGUF inference requires the 'llamacpp' feature. Build with: cargo build --features llamacpp".into()
                ));
            }
            crate::r#gen::get_genai_response(
                &resolved_provider, &resolved_model, &clean, tools,
                npc.and_then(|n| n.api_url.as_deref()),
            ).await?
        }
    };

    let usage_info = response.usage.as_ref().map(|u| UsageInfo {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
    });
    let cost = response.usage.as_ref().map(|u| {
        crate::r#gen::cost::calculate_cost(&resolved_model, u.prompt_tokens, u.completion_tokens)
    }).unwrap_or(0.0);

    let response_text = response.message.content.clone();
    let tool_calls = response.message.tool_calls.clone().unwrap_or_default();

    let response_json = if format == Some("json") {
        if let Some(ref text) = response_text {
            let cleaned = text.trim()
                .strip_prefix("```json").unwrap_or(text.trim())
                .strip_suffix("```").unwrap_or(text.trim())
                .trim();
            serde_json::from_str::<serde_json::Value>(cleaned).ok()
        } else {
            None
        }
    } else {
        None
    };

    let mut updated_messages = clean;
    updated_messages.push(response.message);

    Ok(LlmResponseResult {
        response: response_text,
        response_json,
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

async fn llm_call(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<String> {
    let result = get_llm_response_ext(prompt, npc, model, provider, None, &[], None, None, context, false).await?;
    Ok(result.response.unwrap_or_default())
}

async fn llm_call_json(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<serde_json::Value> {
    let result = get_llm_response_ext(prompt, npc, model, provider, None, &[], None, Some("json"), context, false).await?;
    if let Some(json) = result.response_json {
        Ok(json)
    } else {
        let text = result.response.unwrap_or_default();
        let clean = text.trim()
            .strip_prefix("```json").unwrap_or(text.trim())
            .strip_suffix("```").unwrap_or(text.trim())
            .trim();
        serde_json::from_str(clean).map_err(|e| NpcError::Shell(format!("JSON parse error: {}", e)))
    }
}

fn make_result(response: Option<String>, response_json: Option<serde_json::Value>, messages: Vec<Message>, model: &str, provider: &str) -> LlmResponseResult {
    LlmResponseResult {
        response, response_json, messages,
        tool_calls: vec![], tool_results: vec![],
        usage: None, model: model.into(), provider: provider.into(),
        cost_usd: 0.0, error: None,
    }
}

pub async fn execute_llm_command(
    command: &str,
    model: Option<&str>,
    provider: Option<&str>,
    npc: Option<&Npc>,
    messages: &mut Vec<Message>,
) -> Result<LlmResponseResult> {
    for _attempt in 0..5 {
        let prompt = format!(
            "A user submitted this query: {}.\n\
            You need to generate a bash command that will accomplish the user's intent.\n\
            Respond ONLY with the bash command that should be executed.\n\
            Do not include markdown formatting", command
        );
        let result = get_llm_response(&prompt, npc, model, provider, None, messages, None).await?;
        let bash_command = result.response.clone().unwrap_or_default();

        let run = std::process::Command::new("sh").args(["-c", &bash_command]).output();
        match run {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let explain = format!(
                    "Here was the output of the result for the {} inquiry \
                    which ran this bash command {}:\n\n{}\n\n\
                    Provide a simple response to the user that explains to them \
                    what you did and how it accomplishes what they asked for.",
                    command, bash_command, stdout
                );
                messages.push(Message::user(&explain));
                return get_llm_response(&explain, npc, model, provider, None, messages, None).await;
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let err_prompt = format!(
                    "The command '{}' failed with the following error:\n{}\n\
                    Please suggest a fix or an alternative command.\n\
                    Respond with a JSON object containing the key \"bash_command\" with the suggested command.\n\
                    Do not include any additional markdown formatting.",
                    bash_command, stderr
                );
                if let Ok(fix) = llm_call_json(&err_prompt, model, provider, npc, None).await {
                    if let Some(new_cmd) = fix.get("bash_command").and_then(|v| v.as_str()) {
                        let _ = new_cmd; // npcpy updates command, but we just retry
                    }
                }
            }
            Err(_) => {}
        }
    }
    Ok(make_result(
        Some("Max attempts reached. Unable to execute the command successfully.".into()),
        None, messages.clone(), model.unwrap_or(""), provider.unwrap_or(""),
    ))
}

pub async fn handle_request_input(context: &str, model: &str, provider: &str) -> Result<serde_json::Value> {
    let prompt = format!(
        "Analyze the text:\n{}\n\
        and determine what additional input is needed.\n\
        Return a JSON object with:\n\
        {{\n\
            \"input_needed\": boolean,\n\
            \"request_reason\": string explaining why input is needed,\n\
            \"request_prompt\": string to show user if input needed\n\
        }}\n\
        Do not include any additional markdown formatting or leading ```json tags. \
        Your response must be a valid JSON object.",
        context
    );
    llm_call_json(&prompt, Some(model), Some(provider), None, None).await
}

fn get_jinxes_from_npc<'a>(npc: Option<&'a Npc>, team_jinxes: &'a HashMap<String, Jinx>) -> HashMap<String, &'a Jinx> {
    let mut result = HashMap::new();
    if let Some(npc) = npc {
        for name in &npc.jinx_names {
            if let Some(jinx) = team_jinxes.get(name) {
                result.insert(name.clone(), jinx);
            }
        }
    }
    result
}

fn build_jinx_schema(jinx: &Jinx) -> (String, String) {
    let desc = &jinx.description;
    let mut params = Vec::new();
    let mut has_primary = false;

    for inp in &jinx.inputs {
        if let Some(ref def) = inp.default {
            if let Some(ref d) = inp.description {
                params.push(format!("\"{}\": \"...({})\"", inp.name, d));
                has_primary = true;
            } else if !def.is_empty() {
                params.push(format!("\"{}\": \"...(default: {})\"", inp.name, def));
            } else if !has_primary {
                params.push(format!("\"{}\": \"...\"", inp.name));
                has_primary = true;
            }
        } else {
            params.push(format!("\"{}\": \"...\"", inp.name));
            has_primary = true;
        }
    }

    let schema_str = format!("{{{}}}", params.join(", "));
    (desc.clone(), schema_str)
}

pub async fn handle_jinx_call(
    command: &str,
    jinx_name: &str,
    jinxes: &HashMap<String, Jinx>,
    model: Option<&str>,
    provider: Option<&str>,
    npc: Option<&Npc>,
    messages: &[Message],
    context: Option<&str>,
    n_attempts: usize,
    attempt: usize,
) -> Result<HashMap<String, serde_json::Value>> {
    let jinx = match jinxes.get(jinx_name) {
        Some(j) => j,
        None => {
            if attempt < n_attempts {
                let available: Vec<&str> = jinxes.keys().map(|s| s.as_str()).collect();
                let retry_prompt = format!(
                    "In the previous attempt, the jinx name was: {}.\n\
                    That jinx was not available. Only select from: {}.\n\
                    Original request: {}",
                    jinx_name, available.join(", "), command
                );
                let resp = llm_call_json(&retry_prompt, model, provider, npc, context).await?;
                let new_name = resp.get("jinx_name").and_then(|v| v.as_str()).unwrap_or("");
                if !new_name.is_empty() && new_name != jinx_name {
                    return handle_jinx_call(command, new_name, jinxes, model, provider, npc, messages, context, n_attempts, attempt + 1).await;
                }
            }
            let mut r = HashMap::new();
            r.insert("output".into(), serde_json::json!(format!("Jinx '{}' not found after {} attempts.", jinx_name, n_attempts)));
            r.insert("messages".into(), serde_json::json!(messages));
            return Ok(r);
        }
    };

    tracing::info!("[JINX] {}", jinx.name);

    let mut example_format = serde_json::Map::new();
    for inp in &jinx.inputs {
        example_format.insert(inp.name.clone(), serde_json::Value::String("...".into()));
    }
    let json_format_str = serde_json::to_string_pretty(&serde_json::Value::Object(example_format)).unwrap_or_default();

    let recent: Vec<&Message> = messages.iter().rev().take(5).collect();
    let prompt = format!(
        "The user wants to use the jinx '{}' with the following request:\n\
        '{}'\n\n\
        Here were the previous 5 messages in the conversation: {:?}\n\n\
        Here is the jinx description: {}\n\
        Inputs: {:?}\n\n\
        Please determine the required inputs for the jinx as a JSON object.\n\
        They must be exactly as they are named in the jinx.\n\
        If the jinx requires a file path, you must include an absolute path with extension.\n\
        If the jinx requires code, generate it exactly according to the instructions.\n\n\
        Return only the JSON object without any markdown formatting.\n\
        The format of the JSON object is:\n{}",
        jinx.name, command,
        recent.iter().map(|m| format!("{}: {}", m.role, m.content.as_deref().unwrap_or(""))).collect::<Vec<_>>(),
        jinx.description,
        jinx.inputs.iter().map(|i| &i.name).collect::<Vec<_>>(),
        json_format_str,
    );

    let resp = llm_call_json(&prompt, model, provider, npc, context).await;

    let input_values = match resp {
        Ok(v) if v.is_object() => v,
        Ok(v) => v,
        Err(e) => {
            if attempt < n_attempts {
                let ctx = format!("Previous attempt failed to parse JSON: {}.", e);
                return handle_jinx_call(command, jinx_name, jinxes, model, provider, npc, messages, Some(&ctx), n_attempts, attempt + 1).await;
            }
            let mut r = HashMap::new();
            r.insert("output".into(), serde_json::json!(format!("Error extracting inputs for jinx '{}'", jinx_name)));
            r.insert("messages".into(), serde_json::json!(messages));
            return Ok(r);
        }
    };

    let missing: Vec<&str> = jinx.inputs.iter()
        .filter(|inp| inp.default.is_none())
        .filter(|inp| {
            input_values.get(&inp.name).map(|v| v.as_str() == Some("") || v.is_null()).unwrap_or(true)
        })
        .map(|inp| inp.name.as_str())
        .collect();

    if !missing.is_empty() && attempt < n_attempts {
        let ctx = format!("Previous attempt missing inputs: {:?}. Values were: {}", missing, input_values);
        return handle_jinx_call(
            &format!("{}. {}", command, ctx),
            jinx_name, jinxes, model, provider, npc, messages, Some(&ctx), n_attempts, attempt + 1,
        ).await;
    }

    tracing::info!("[INPUTS] {}", input_values);

    let mut exec_inputs: HashMap<String, String> = HashMap::new();
    for inp in &jinx.inputs {
        if let Some(ref def) = inp.default {
            exec_inputs.insert(inp.name.clone(), shellexpand::tilde(def).to_string());
        }
    }
    if let Some(obj) = input_values.as_object() {
        for (k, v) in obj {
            let val = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            exec_inputs.insert(k.clone(), val);
        }
    }

    let mut output = String::new();
    for step in &jinx.steps {
        let mut rendered = step.code.clone();
        for (k, v) in &exec_inputs {
            rendered = rendered.replace(&format!("{{{{ {} }}}}", k), v);
            rendered = rendered.replace(&format!("{{{{{}}}}}", k), v);
        }

        match step.engine.as_str() {
            "bash" | "sh" => {
                match std::process::Command::new("sh").args(["-c", &rendered]).output() {
                    Ok(o) => {
                        let stdout = String::from_utf8_lossy(&o.stdout);
                        let stderr = String::from_utf8_lossy(&o.stderr);
                        output.push_str(&stdout);
                        if !stderr.is_empty() {
                            output.push_str(&stderr);
                        }
                    }
                    Err(e) => output.push_str(&format!("Error: {}", e)),
                }
            }
            "python" | "python3" => {
                match std::process::Command::new("python3").args(["-c", &rendered]).output() {
                    Ok(o) => {
                        let stdout = String::from_utf8_lossy(&o.stdout);
                        let stderr = String::from_utf8_lossy(&o.stderr);
                        output.push_str(&stdout);
                        if !stderr.is_empty() {
                            output.push_str(&stderr);
                        }
                    }
                    Err(e) => output.push_str(&format!("Error: {}", e)),
                }
            }
            _ => {
                output.push_str(&format!("Unknown engine: {}", step.engine));
            }
        }
    }

    if output.is_empty() {
        output = "Executed with no output.".to_string();
    }

    tracing::info!("[RESULT] {}", &output[..output.len().min(300)]);

    if output.starts_with("Error:") && attempt < n_attempts {
        let ctx = format!("Jinx failed: {}. Previous inputs: {}", output, input_values);
        return handle_jinx_call(command, jinx_name, jinxes, model, provider, npc, messages, Some(&ctx), n_attempts, attempt + 1).await;
    }

    let mut r = HashMap::new();
    r.insert("output".into(), serde_json::json!(output));
    r.insert("messages".into(), serde_json::json!(messages));
    r.insert("jinx_calls".into(), serde_json::json!([{
        "name": jinx_name,
        "arguments": input_values,
        "result": output,
    }]));
    Ok(r)
}

pub async fn handle_action_choice(
    command: &str,
    action_data: &serde_json::Value,
    jinxes: &HashMap<String, Jinx>,
    model: Option<&str>,
    provider: Option<&str>,
    npc: Option<&Npc>,
    messages: &[Message],
    context: Option<&str>,
    last_jinx_output: Option<&str>,
    step_outputs: &[String],
) -> Result<HashMap<String, serde_json::Value>> {
    let action_name = action_data.get("action").and_then(|v| v.as_str()).unwrap_or("answer");

    if action_name == "invoke_jinx" || action_data.get("jinx_name").is_some() {
        let jname = action_data.get("jinx_name").and_then(|v| v.as_str()).unwrap_or("");
        let mut step_context = context.unwrap_or("").to_string();
        if !step_outputs.is_empty() {
            step_context += &format!("\nContext from previous steps: {:?}", step_outputs);
        }

        let result = handle_jinx_call(
            command, jname, jinxes, model, provider, npc, messages,
            Some(&step_context), 3, 0,
        ).await?;

        let output = result.get("output").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let jinx_calls = result.get("jinx_calls").cloned().unwrap_or(serde_json::json!([]));

        let mut r = HashMap::new();
        r.insert("output".into(), serde_json::json!(output));
        r.insert("messages".into(), result.get("messages").cloned().unwrap_or(serde_json::json!(messages)));
        r.insert("jinx_calls".into(), jinx_calls);
        Ok(r)
    } else if action_name == "answer" {
        let prompt = format!("The user asked: {}\n\nProvide a direct answer. Do not reference tools or jinxes.", command);
        let response = llm_call(&prompt, model, provider, npc, context).await?;
        let mut r = HashMap::new();
        r.insert("output".into(), serde_json::json!(response));
        r.insert("messages".into(), serde_json::json!(messages));
        r.insert("jinx_calls".into(), serde_json::json!([]));
        Ok(r)
    } else {
        let mut r = HashMap::new();
        r.insert("output".into(), serde_json::json!("INVALID_ACTION"));
        r.insert("messages".into(), serde_json::json!(messages));
        r.insert("jinx_calls".into(), serde_json::json!([]));
        Ok(r)
    }
}

pub async fn check_llm_command(
    command: &str,
    model: Option<&str>,
    provider: Option<&str>,
    npc: Option<&Npc>,
    messages: &mut Vec<Message>,
    context: Option<&str>,
    jinxes: &HashMap<String, Jinx>,
    max_iterations: usize,
) -> Result<HashMap<String, serde_json::Value>> {
    if jinxes.is_empty() {
        let response = get_llm_response_ext(
            command, npc, model, provider, None,
            &messages[messages.len().saturating_sub(10)..],
            None, None, context, false,
        ).await?;
        messages.push(Message::user(command));
        let out = response.response.unwrap_or_default();
        if !out.is_empty() {
            messages.push(Message::assistant(&out));
        }
        let mut r = HashMap::new();
        r.insert("messages".into(), serde_json::json!(messages.clone()));
        r.insert("output".into(), serde_json::json!(out));
        return Ok(r);
    }

    let jinx_list: String = jinxes.iter().map(|(name, jinx)| {
        let (desc, schema) = build_jinx_schema(jinx);
        format!("- {}: {} (inputs: {})", name, desc, schema)
    }).collect::<Vec<_>>().join("\n");

    let prompt = format!(
        "A user submitted this request: {}\n\n\
        Determine the nature of the user's request:\n\n\
        1. Should a jinx be invoked to fulfill the request? A jinx is a jinja-template execution script.\n\
        2. Is it a general question that requires an informative answer?\n\n\
        Use jinxes when the answer needs to be up-to-date, or the user wants to read/edit a file, \
        search the web, take a screenshot, run code, or perform an action.\n\n\
        If a user asks to explain something like the plot of the aeneid, or where mount everest is, \
        that can be answered without a jinx.\n\n\
        Available jinxes:\n{}\n\n\
        Return a JSON array of actions. Each action is either:\n\
        - {{\"action\": \"answer\"}} for a direct answer\n\
        - {{\"action\": \"invoke_jinx\", \"jinx_name\": \"name\"}} to invoke a jinx\n\n\
        Return only the action sequence as JSON. Do not include leading ```json or other markdown tags.",
        command, jinx_list
    );

    let recent: Vec<&Message> = messages.iter().rev().take(5).collect();
    let full_prompt = if !recent.is_empty() {
        format!("{}\n\nRecent conversation: {:?}", prompt,
            recent.iter().map(|m| format!("{}: {}", m.role, m.content.as_deref().unwrap_or(""))).collect::<Vec<_>>())
    } else {
        prompt
    };

    let response = llm_call_json(&full_prompt, model, provider, npc, context).await?;

    let actions: Vec<serde_json::Value> = if let Some(arr) = response.as_array() {
        arr.clone()
    } else if response.is_object() {
        vec![response]
    } else {
        vec![serde_json::json!({"action": "answer"})]
    };

    let mut step_outputs: Vec<String> = Vec::new();
    let mut all_jinx_calls: Vec<serde_json::Value> = Vec::new();
    let mut current_messages = messages.clone();
    let mut last_jinx_output: Option<String> = None;

    for action_data in &actions {
        let action_result = handle_action_choice(
            command, action_data, jinxes, model, provider, npc,
            &current_messages, context,
            last_jinx_output.as_deref(), &step_outputs,
        ).await?;

        if let Some(msgs) = action_result.get("messages") {
            if let Ok(m) = serde_json::from_value::<Vec<Message>>(msgs.clone()) {
                current_messages = m;
            }
        }
        let output = action_result.get("output").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if let Some(jc) = action_result.get("jinx_calls").and_then(|v| v.as_array()) {
            all_jinx_calls.extend(jc.clone());
        }

        if output == "INVALID_ACTION" {
            let retry_prompt = format!(
                "In the previous attempt, the correct action name was not provided. \
                Only select from available jinxes.\nOriginal request: {}", command
            );
            return check_llm_command(
                &retry_prompt, model, provider, npc, messages, context, jinxes,
                max_iterations.saturating_sub(1),
            ).await;
        }

        step_outputs.push(output.clone());
        last_jinx_output = Some(output);
    }

    if step_outputs.len() == 1 {
        let mut r = HashMap::new();
        r.insert("messages".into(), serde_json::json!(current_messages));
        r.insert("output".into(), serde_json::json!(step_outputs[0]));
        r.insert("jinx_calls".into(), serde_json::json!(all_jinx_calls));
        return Ok(r);
    }

    let synthesis_prompt = format!(
        "The user asked: \"{}\"\n\n\
        The following information was gathered:\n{}\n\n\
        Provide a single, coherent response answering the user's question directly.\n\
        Do not mention the steps taken.",
        command, serde_json::to_string_pretty(&step_outputs).unwrap_or_default()
    );
    let synthesis = llm_call(&synthesis_prompt, model, provider, npc, context).await?;

    let mut r = HashMap::new();
    r.insert("messages".into(), serde_json::json!(current_messages));
    r.insert("output".into(), serde_json::json!(synthesis));
    r.insert("jinx_calls".into(), serde_json::json!(all_jinx_calls));
    Ok(r)
}

pub async fn gen_image(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, width: u32, height: u32, api_key: Option<&str>) -> Result<crate::r#gen::GeneratedImage> {
    let (m, p) = if let (Some(m), Some(p)) = (model, provider) {
        (m.to_string(), p.to_string())
    } else if let Some(npc) = npc {
        (model.map(String::from).unwrap_or_else(|| npc.resolved_model()),
         provider.map(String::from).unwrap_or_else(|| npc.resolved_provider()))
    } else {
        (model.unwrap_or("dall-e-3").to_string(), provider.unwrap_or("openai").to_string())
    };
    crate::r#gen::generate_image(prompt, &m, &p, api_key, width, height).await
}

pub async fn gen_video(prompt: &str, model: Option<&str>, provider: Option<&str>, _npc: Option<&Npc>, output_path: &str) -> Result<HashMap<String, String>> {
    let model_str = model.unwrap_or("veo-3.1-fast-generate-preview");
    let provider_str = provider.unwrap_or("gemini");
    let mut result = HashMap::new();

    if provider_str == "gemini" {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .map_err(|_| NpcError::LlmRequest("GOOGLE_API_KEY or GEMINI_API_KEY not set for video gen".into()))?;
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

pub async fn breathe(messages: &[Message], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<HashMap<String, serde_json::Value>> {
    if messages.is_empty() {
        let mut r = HashMap::new();
        r.insert("output".into(), serde_json::json!({}));
        r.insert("messages".into(), serde_json::json!([]));
        return Ok(r);
    }
    let conversation_text: String = messages.iter()
        .filter_map(|m| m.content.as_ref().map(|c| format!("{}: {}", m.role, c)))
        .collect::<Vec<_>>().join("\n");

    let prompt = format!(
        "Read the following conversation:\n\n{}\n\n\
        Now identify the following items:\n\n\
        1. The high level objective\n\
        2. The most recent task\n\
        3. The accomplishments thus far\n\
        4. The failures thus far\n\n\
        Return a JSON like so:\n\n\
        {{\n\
            \"high_level_objective\": \"the overall goal so far for the user\",\n\
            \"most_recent_task\": \"The currently ongoing task\",\n\
            \"accomplishments\": [\"accomplishment1\", \"accomplishment2\"],\n\
            \"failures\": [\"failure1\", \"failure2\"]\n\
        }}",
        conversation_text
    );
    let res = llm_call_json(&prompt, model, provider, npc, context).await?;
    let fmt = format!(
        "Here is a summary of the previous session. \
        The high level objective was: {} \n The accomplishments were: {}, \
        the failures were: {} and the most recent task was: {}",
        res.get("high_level_objective").and_then(|v| v.as_str()).unwrap_or("?"),
        res.get("accomplishments").unwrap_or(&serde_json::json!([])),
        res.get("failures").unwrap_or(&serde_json::json!([])),
        res.get("most_recent_task").and_then(|v| v.as_str()).unwrap_or("?"),
    );
    let mut r = HashMap::new();
    r.insert("output".into(), serde_json::Value::String(fmt.clone()));
    r.insert("summary".into(), res);
    r.insert("messages".into(), serde_json::json!([{"role": "assistant", "content": fmt}]));
    Ok(r)
}

pub async fn get_facts(content_text: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>, attempt_number: usize, n_attempts: usize) -> Result<Vec<serde_json::Value>> {
    let prompt = format!(
        "Extract facts from this text. A fact is a specific statement that can be sourced from the text.\n\n\
        Example: if text says \"the moon is the earth's only currently known satellite\", extract:\n\
        - \"The moon is a satellite of earth\"\n\
        - \"The moon is the only current satellite of earth\"\n\
        - \"There may have been other satellites of earth\" (inferred from \"only currently known\")\n\n\
        A fact is a piece of information that makes a statement about the world.\n\
        Facts may be simple or complex. They can also be conflicting with each other.\n\n\
        Here is the text:\n\
        Text: \"{}\"\n\n\
        Facts should never be more than one or two sentences. They must be explicitly \
        derived or inferred from the source text. Do not simply repeat the source text verbatim.\n\n\
        No two facts should share substantially similar claims. They should be conceptually distinct.\n\
        Respond with JSON:\n\
        {{\"facts\": [{{\"statement\": \"fact statement\", \"source_text\": \"relevant source text\", \"type\": \"explicit or inferred\"}}]}}",
        content_text
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    let facts = result.get("facts").and_then(|f| f.as_array()).cloned().unwrap_or_default();

    if facts.is_empty() && attempt_number < n_attempts {
        tracing::info!("Attempt {} to extract facts yielded no results. Retrying...", attempt_number);
        return get_facts(content_text, model, provider, npc, context, attempt_number + 1, n_attempts).await;
    }
    Ok(facts)
}

pub async fn zoom_in(facts: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>, attempt_number: usize, n_attempts: usize) -> Result<Vec<serde_json::Value>> {
    let valid_facts: Vec<&serde_json::Value> = facts.iter()
        .filter(|f| f.get("statement").and_then(|s| s.as_str()).is_some())
        .collect();
    if valid_facts.is_empty() { return Ok(vec![]); }

    let facts_text: String = valid_facts.iter()
        .filter_map(|f| f.get("statement").and_then(|s| s.as_str()))
        .map(|s| format!("- {}", s))
        .collect::<Vec<_>>().join("\n");

    let prompt = format!(
        "Look at these facts and infer new implied facts:\n\n\
        {}\n\n\
        What other facts can be reasonably inferred from these?\n\
        Respond with JSON:\n\
        {{\n\
            \"implied_facts\": [\n\
                {{\n\
                    \"statement\": \"new implied fact\",\n\
                    \"inferred_from\": [\"which facts this comes from\"]\n\
                }}\n\
            ]\n\
        }}",
        facts_text
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    let implied = result.get("implied_facts").and_then(|f| f.as_array()).cloned().unwrap_or_default();

    if implied.is_empty() && attempt_number < n_attempts {
        return zoom_in(facts, model, provider, npc, context, attempt_number + 1, n_attempts).await;
    }
    Ok(implied)
}

pub async fn identify_groups(facts: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let prompt = format!(
        "What are the main groups these facts could be organized into?\n\
        Express these groups in plain, natural language.\n\n\
        For example, given:\n\
            - User enjoys programming in Python\n\
            - User works on machine learning projects\n\
            - User likes to play piano\n\
            - User practices meditation daily\n\n\
        You might identify groups like:\n\
            - Programming\n\
            - Machine Learning\n\
            - Musical Interests\n\
            - Daily Practices\n\n\
        Return a JSON object with the following structure:\n\
        {{\"groups\": [\"list of group names\"]}}\n\n\
        Return only the JSON object.\n\n\
        Facts: {}",
        serde_json::to_string(facts).unwrap_or_default()
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    Ok(result.get("groups").and_then(|g| g.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default())
}

pub async fn get_related_concepts_multi(node_name: &str, node_type: &str, all_concept_names: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let prompt = format!(
        "Which of the following concepts from the entire ontology relate to the given {}?\n\
        Select all that apply, from the most specific to the most abstract.\n\n\
        {}: \"{}\"\n\n\
        Available Concepts:\n{}\n\n\
        Respond with JSON: {{\"related_concepts\": [\"Concept A\", \"Concept B\", ...]}}",
        node_type, node_type, node_name,
        serde_json::to_string_pretty(all_concept_names).unwrap_or_default()
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    Ok(result.get("related_concepts").and_then(|c| c.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default())
}

pub async fn assign_groups_to_fact(fact: &str, groups: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let prompt = format!(
        "Given this fact, assign it to any relevant groups.\n\
        A fact can belong to multiple groups if it fits.\n\n\
        Here is the fact: {}\n\n\
        Here are the groups: {:?}\n\n\
        Return a JSON object: {{\"groups\": [\"list of group names\"]}}\n\
        Do not include any additional markdown formatting.",
        fact, groups
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    Ok(result.get("groups").and_then(|g| g.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default())
}

pub async fn generate_group_candidates(items: &[String], item_type: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>, n_passes: usize, subset_size: usize) -> Result<Vec<String>> {
    let mut all_candidates: Vec<String> = Vec::new();

    for _pass in 0..n_passes {
        let item_subset: Vec<&String> = if items.len() > subset_size {
            items.iter().take(subset_size).collect()
        } else {
            items.iter().collect()
        };

        let prompt = format!(
            "From the following {}, identify specific and relevant conceptual groups.\n\
            Think about the core subject or entity being discussed.\n\n\
            GUIDELINES FOR GROUP NAMES:\n\
            1. Prioritize Specificity: Names should be precise and directly reflect the content.\n\
            2. Favor Nouns and Noun Phrases.\n\
            3. AVOID: Gerunds, overly generic terms (\"Concepts\", \"Processes\", \"Dynamics\").\n\
            4. Direct Naming: specific entities can be group names themselves.\n\n\
            {}: {:?}\n\n\
            Return a JSON object:\n\
            {{\"groups\": [\"list of specific group names\"]}}",
            item_type, item_type, item_subset
        );
        if let Ok(result) = llm_call_json(&prompt, model, provider, npc, context).await {
            if let Some(groups) = result.get("groups").and_then(|g| g.as_array()) {
                for g in groups {
                    if let Some(s) = g.as_str() {
                        if !all_candidates.contains(&s.to_string()) {
                            all_candidates.push(s.to_string());
                        }
                    }
                }
            }
        }
    }
    Ok(all_candidates)
}

pub async fn remove_idempotent_groups(group_candidates: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let prompt = format!(
        "Compare these group names. Identify and list ONLY the groups that are conceptually distinct and specific.\n\n\
        GUIDELINES:\n\
        1. Prioritize Specificity and Direct Naming.\n\
        2. Prefer Concrete Entities/Actions.\n\
        3. Rephrase Gerunds to noun phrases.\n\
        4. AVOID overly generic terms.\n\
        5. If two groups are very similar, keep the more descriptive one.\n\n\
        Groups: {:?}\n\n\
        Return JSON:\n\
        {{\"distinct_groups\": [\"list of distinct group names to keep\"]}}",
        group_candidates
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    Ok(result.get("distinct_groups").and_then(|g| g.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default())
}

pub async fn generate_groups(facts: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<serde_json::Value>> {
    let facts_text: String = facts.iter()
        .filter_map(|f| f.get("statement").and_then(|s| s.as_str()))
        .map(|s| format!("- {}", s))
        .collect::<Vec<_>>().join("\n");

    let prompt = format!(
        "Generate conceptual groups for this group of facts:\n\n\
        {}\n\n\
        Create categories that encompass multiple related facts, but do not unnecessarily combine facts with conjunctions.\n\
        Your aim is to generalize commonly occurring ideas into groups.\n\
        Group names should never be more than two words. No gerunds. No conjunctions.\n\
        Respond with JSON:\n\
        {{\"groups\": [{{\"name\": \"group name\"}}]}}",
        facts_text
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    Ok(result.get("groups").and_then(|g| g.as_array()).cloned().unwrap_or_default())
}

pub async fn r#abstract(groups: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<serde_json::Value>> {
    let groups_text: String = groups.iter()
        .filter_map(|g| g.get("name").and_then(|n| n.as_str()))
        .map(|n| format!("- \"{}\"", n))
        .collect::<Vec<_>>().join("\n");

    let prompt = format!(
        "Create more abstract categories from this list of groups.\n\n\
        Groups:\n{}\n\n\
        You will create higher-level concepts that interrelate between the given groups.\n\
        Create abstract categories that encompass multiple related facts, but do not \
        unnecessarily combine facts with conjunctions.\n\
        Your aim is to abstract, not to just arbitrarily generate associations.\n\n\
        Group names should never be more than two words. No gerunds. No conjunctions.\n\
        Generate no more than 5 new concepts and no fewer than 2.\n\n\
        Respond with JSON:\n\
        {{\"groups\": [{{\"name\": \"abstract category name\"}}]}}",
        groups_text
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    Ok(result.get("groups").and_then(|g| g.as_array()).cloned().unwrap_or_default())
}

pub async fn remove_redundant_groups(groups: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<serde_json::Value>> {
    let groups_text: String = groups.iter()
        .filter_map(|g| g.get("name").and_then(|n| n.as_str()))
        .map(|n| format!("- {}", n))
        .collect::<Vec<_>>().join("\n");

    let prompt = format!(
        "Remove redundant groups from this list:\n\n\
        {}\n\n\
        Merge similar groups and keep only distinct concepts.\n\
        Your aim is to abstract, not to just arbitrarily generate associations.\n\
        Group names should never be more than two words. No gerunds. No conjunctions.\n\n\
        Respond with JSON:\n\
        {{\"groups\": [{{\"name\": \"final group name\"}}]}}",
        groups_text
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    Ok(result.get("groups").and_then(|g| g.as_array()).cloned().unwrap_or_default())
}

pub async fn prune_fact_subset_llm(fact_subset: &[serde_json::Value], concept_name: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<serde_json::Value>> {
    let facts_json = serde_json::to_string_pretty(fact_subset).unwrap_or_default();
    let prompt = format!(
        "The following facts are all related to the concept \"{}\".\n\
        Review ONLY this subset and identify groups of facts that are semantically identical.\n\
        Return only the set of facts that are semantically distinct, and archive the rest.\n\n\
        Fact Subset: {}\n\n\
        Return a json list:\n\
        {{\"refined_facts\": [fact1, fact2, fact3, ...]}}",
        concept_name, facts_json
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    Ok(result.get("refined_facts").and_then(|f| f.as_array()).cloned().unwrap_or_default())
}

pub async fn consolidate_facts_llm(new_fact: &serde_json::Value, existing_facts: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<serde_json::Value> {
    let new_stmt = new_fact.get("statement").and_then(|s| s.as_str()).unwrap_or("");
    let existing: Vec<&str> = existing_facts.iter()
        .filter_map(|f| f.get("statement").and_then(|s| s.as_str())).collect();
    let prompt = format!(
        "Analyze the \"New Fact\" in the context of the \"Existing Facts\" list.\n\
        Determine if the new fact provides genuinely new information or is a repeat.\n\n\
        New Fact: \"{}\"\n\n\
        Existing Facts:\n{}\n\n\
        Possible decisions:\n\
        - 'novel': The fact introduces new, distinct information.\n\
        - 'redundant': The fact repeats information already present.\n\n\
        Respond with JSON:\n\
        {{\"decision\": \"novel or redundant\", \"reason\": \"A brief explanation.\"}}",
        new_stmt, serde_json::to_string_pretty(&existing).unwrap_or_default()
    );
    llm_call_json(&prompt, model, provider, npc, context).await
}

pub async fn get_related_facts_llm(new_fact_statement: &str, existing_fact_statements: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>, attempt_number: usize, n_attempts: usize) -> Result<Vec<String>> {
    let prompt = format!(
        "A new fact has been learned: \"{}\"\n\n\
        Which of the following existing facts are directly related to it \
        (causally, sequentially, or thematically)?\n\
        Select only the most direct and meaningful connections.\n\n\
        Existing Facts:\n{}\n\n\
        Respond with JSON:\n\
        {{\"related_facts\": [\"statement of a related fact\", ...]}}",
        new_fact_statement, serde_json::to_string_pretty(existing_fact_statements).unwrap_or_default()
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    let related = result.get("related_facts").and_then(|f| f.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>())
        .unwrap_or_default();

    if related.is_empty() && attempt_number < n_attempts {
        tracing::info!("Attempt {} to find related facts yielded no results. Retrying...", attempt_number);
        return get_related_facts_llm(new_fact_statement, existing_fact_statements, model, provider, npc, context, attempt_number + 1, n_attempts).await;
    }
    Ok(related)
}

pub async fn find_best_link_concept_llm(candidate_concept_name: &str, existing_concept_names: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Option<String>> {
    let prompt = format!(
        "Here is a new candidate concept: \"{}\"\n\n\
        Which of the following existing concepts is it most closely related to?\n\
        The relationship could be as a sub-category, a similar idea, or a related domain.\n\n\
        Existing Concepts:\n{}\n\n\
        Respond with the single best-fit concept to link to, or \"none\" if it is a genuinely new root idea.\n\
        {{\"best_link_concept\": \"The single best concept name OR none\"}}",
        candidate_concept_name, serde_json::to_string_pretty(existing_concept_names).unwrap_or_default()
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    Ok(result.get("best_link_concept").and_then(|v| v.as_str())
        .map(String::from).filter(|v| v.to_lowercase() != "none"))
}

pub async fn asymptotic_freedom(parent_concept_name: &str, supporting_facts: &[serde_json::Value], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<Vec<String>> {
    let fact_statements: Vec<&str> = supporting_facts.iter()
        .filter_map(|f| f.get("statement").and_then(|s| s.as_str())).collect();
    let prompt = format!(
        "The concept \"{}\" is supported by many diverse facts.\n\
        Propose a layer of 2-4 more specific sub-concepts to better organize these facts.\n\
        These new concepts will exist as nodes that link to \"{}\".\n\n\
        Supporting Facts: {}\n\
        Respond with JSON:\n\
        {{\"new_sub_concepts\": [\"sub_layer1\", \"sub_layer2\"]}}",
        parent_concept_name, parent_concept_name,
        serde_json::to_string_pretty(&fact_statements).unwrap_or_default()
    );
    let result = llm_call_json(&prompt, model, provider, npc, context).await?;
    Ok(result.get("new_sub_concepts").and_then(|f| f.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default())
}

pub async fn bootstrap(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, n_samples: usize, context: Option<&str>) -> Result<String> {
    let mut results = Vec::new();
    for i in 0..n_samples {
        results.push(llm_call(
            &format!("Sample {}: {}", i + 1, prompt),
            model, provider, npc, context,
        ).await?);
    }
    let combined = results.iter().enumerate()
        .map(|(i, r)| format!("Response {}: {}", i + 1, r))
        .collect::<Vec<_>>().join("\n\n");
    synthesize(&combined, model, provider, npc, context).await
}

pub async fn harmonize(prompt: &str, items: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, harmony_rules: Option<&[String]>, context: Option<&str>) -> Result<String> {
    let items_text = items.iter().enumerate()
        .map(|(i, s)| format!("{}. {}", i + 1, s)).collect::<Vec<_>>().join("\n");
    let rules = harmony_rules.map(|r| r.join(", ")).unwrap_or_else(|| "maintain_consistency".into());
    llm_call(
        &format!("Harmonize these items: {}\nTask: {}\nRules: {}", items_text, prompt, rules),
        model, provider, npc, context,
    ).await
}

pub async fn orchestrate(prompt: &str, items: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, workflow: &str, context: Option<&str>) -> Result<String> {
    let items_text = items.iter().enumerate()
        .map(|(i, s)| format!("{}. {}", i + 1, s)).collect::<Vec<_>>().join("\n");
    llm_call(
        &format!("Orchestrate using {}:\nTask: {}\nItems: {}", workflow, prompt, items_text),
        model, provider, npc, context,
    ).await
}

pub async fn spread_and_sync(prompt: &str, variations: &[String], model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, sync_strategy: &str, context: Option<&str>) -> Result<String> {
    let mut results = Vec::new();
    for v in variations {
        results.push(llm_call(
            &format!("Analyze from {} perspective:\nTask: {}", v, prompt),
            model, provider, npc, context,
        ).await?);
    }
    let combined = results.iter().enumerate()
        .map(|(i, r)| format!("Response {}: {}", i + 1, r))
        .collect::<Vec<_>>().join("\n\n");
    llm_call(
        &format!("Synthesize these multiple perspectives:\n{}\n\nSynthesis strategy: {}", combined, sync_strategy),
        model, provider, npc, context,
    ).await
}

pub async fn criticize(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<String> {
    llm_call(
        &format!(
            "Provide a critical analysis and constructive criticism of the following:\n{}\n\n\
            Focus on identifying weaknesses, potential improvements, and alternative approaches.\n\
            Be specific and provide actionable feedback.",
            prompt
        ),
        model, provider, npc, context,
    ).await
}

pub async fn synthesize(prompt: &str, model: Option<&str>, provider: Option<&str>, npc: Option<&Npc>, context: Option<&str>) -> Result<String> {
    llm_call(
        &format!(
            "Synthesize this content:\n{}\n\n\
            Create a clear, concise synthesis that captures the essence of the content.",
            prompt
        ),
        model, provider, npc, context,
    ).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_model_provider_defaults() {
        let (m, p) = resolve_model_provider(None, None, None);
        assert_eq!(m, "llama3.2");
        assert_eq!(p, "ollama");
    }

    #[test]
    fn test_resolve_model_provider_explicit() {
        let (m, p) = resolve_model_provider(None, Some("gpt-4o"), Some("openai"));
        assert_eq!(m, "gpt-4o");
        assert_eq!(p, "openai");
    }

    #[test]
    fn test_lookup_provider() {
        assert_eq!(lookup_provider("gpt-4o"), "openai");
        assert_eq!(lookup_provider("claude-3-opus"), "anthropic");
        assert_eq!(lookup_provider("gemini-2.5-flash"), "gemini");
        assert_eq!(lookup_provider("qwen3:8b"), "ollama");
        assert_eq!(lookup_provider("llama3.2"), "ollama");
    }
}
