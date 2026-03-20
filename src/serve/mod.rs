use crate::error::{NpcError, Result};
use crate::npc_compiler;
use crate::r#gen::Message;
use crate::npc_compiler::Team;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use axum::{
    extract::{Json, State, Path as AxumPath},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};

pub struct ServerConfig {
    pub http_port: u16,
    pub mcp_enabled: bool,
    pub host: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            http_port: 5337,
            mcp_enabled: true,
            host: "0.0.0.0".to_string(),
        }
    }
}

pub struct ServerState {
    pub team: Team,
    pub active_npc_name: String,
    pub conversations: HashMap<String, Vec<Message>>,
}

type AppState = Arc<Mutex<ServerState>>;

fn json_response(value: serde_json::Value) -> impl IntoResponse {
    (StatusCode::OK, Json(value))
}

fn error_response(msg: &str) -> impl IntoResponse {
    (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": msg})))
}

pub fn create_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/team", get(get_team))
        .route("/api/npcs", get(get_npcs))
        .route("/api/npc/switch", post(switch_npc))
        .route("/api/jinxes", get(get_jinxes))
        .route("/api/chat", post(chat))
        .route("/api/check_command", post(check_command))
        .route("/api/jinx", post(execute_jinx))
        .route("/api/models", get(get_models))
        .route("/api/conversations", get(list_conversations))
        .route("/api/conversations", post(create_conversation).get(list_conversations))
        .route("/api/search", post(search))
        .route("/api/image", post(generate_image))
        .route("/api/tts", post(tts))
        .route("/api/stt", post(stt))
        .route("/api/settings", get(get_settings))
        .route("/api/settings", post(save_settings).get(get_settings))
        .route("/api/kg/stats", get(kg_stats))
        .route("/api/kg/search", post(kg_search))
        .route("/api/kg/facts", get(get_kg_facts))
        .route("/api/kg/concepts", get(get_kg_concepts))
        .route("/api/kg/node", post(add_kg_node))
        .route("/api/kg/node/:node_id", post(update_kg_node).delete(delete_kg_node))
        .route("/api/kg/edge", post(add_kg_edge))
        .route("/api/kg/process", post(trigger_kg_process))
        .route("/api/kg/ingest", post(ingest_to_kg))
        .route("/api/kg/query", post(query_kg))
        .route("/api/jinx/save", post(save_jinx))
        .route("/api/jinx/delete", post(delete_jinx))
        .route("/api/jinx/test", post(test_jinx))
        .route("/api/attachments/:message_id", get(get_attachments))
        .route("/api/settings/global", get(get_global_settings).post(save_global_settings))
        .route("/api/settings/project", get(get_project_settings).post(save_project_settings))
        .route("/api/memories/extract", post(extract_memories))
        .route("/api/command/:command", post(api_command))
        .route("/api/capture", post(capture))
        .route("/api/memories/pending", get(get_pending_memories_route))
        .route("/api/memories/approve", post(approve_memories))
        .route("/api/memories/search", post(search_memories))
        .route("/api/conversations/search", post(search_conversations_route))
        .route("/api/conversations/:id/messages", get(get_conversation_messages))
        .route("/api/conversations/:id/messages/:msg_id", axum::routing::delete(delete_message))
        .route("/api/npc/save", post(save_npc))
        .route("/api/team/sync/status", get(team_sync_status))
        .route("/api/team/sync/init", post(team_sync_init))
        .route("/api/team/sync/pull", post(team_sync_pull))
        .route("/api/team/sync/commit", post(team_sync_commit))
        .route("/api/team/sync/diff", get(team_sync_diff))
        .route("/api/ollama/status", get(ollama_status))
        .route("/api/ollama/models", get(get_ollama_models))
        .route("/api/video", post(generate_video_api))
        .route("/api/npc/executions", get(get_npc_executions_route))
        .route("/api/jinx/executions", get(get_jinx_executions_route))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn get_team(State(state): State<AppState>) -> impl IntoResponse {
    let state = state.lock().await;
    Json(serde_json::json!({
        "npcs": state.team.npc_names(),
        "jinxes": state.team.jinx_names(),
        "active_npc": state.active_npc_name,
        "context": state.team.context,
    }))
}

async fn get_npcs(State(state): State<AppState>) -> impl IntoResponse {
    let state = state.lock().await;
    let npcs: Vec<serde_json::Value> = state.team.npcs.values().map(|n| n.to_dict()).collect();
    Json(serde_json::json!({"npcs": npcs}))
}

async fn switch_npc(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let name = body["npc"].as_str().unwrap_or("");
    let mut state = state.lock().await;
    if state.team.get_npc(name).is_some() {
        state.active_npc_name = name.to_string();
        Json(serde_json::json!({"status": "switched", "npc": name}))
    } else {
        Json(serde_json::json!({"error": format!("NPC '{}' not found", name)}))
    }
}

async fn get_jinxes(State(state): State<AppState>) -> impl IntoResponse {
    let state = state.lock().await;
    let jinxes: Vec<serde_json::Value> = state.team.jinxes.values().map(|j| j.to_dict()).collect();
    Json(serde_json::json!({"jinxes": jinxes}))
}

async fn chat(State(state_arc): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let input = body["message"].as_str().unwrap_or("").to_string();
    let npc_name = body["npc"].as_str().map(String::from);
    let conversation_id = body["conversation_id"].as_str().map(String::from);

    let (npc, prev_messages, conv_id, team_context) = {
        let state = state_arc.lock().await;
        let name = npc_name.as_deref().unwrap_or(&state.active_npc_name);
        let npc = state.team.get_npc(name).cloned();
        let conv_id = conversation_id.unwrap_or_else(|| "default".to_string());
        let prev = state.conversations.get(&conv_id).cloned().unwrap_or_default();
        let ctx = state.team.context.clone();
        (npc, prev, conv_id, ctx)
    };

    if let Some(npc) = npc {
        let system = npc.system_prompt(team_context.as_deref());
        let mut messages = vec![Message::system(system)];
        messages.extend(prev_messages);
        messages.push(Message::user(&input));

        match crate::r#gen::get_genai_response(
            &npc.resolved_provider(), &npc.resolved_model(),
            &messages, None, npc.api_url.as_deref(),
        ).await {
            Ok(resp) => {
                let output = resp.message.content.clone().unwrap_or_default();
                let mut state = state_arc.lock().await;
                let existing = state.conversations.entry(conv_id.clone()).or_default();
                existing.push(Message::user(&input));
                existing.push(Message { role: "assistant".into(), content: Some(output.clone()), tool_calls: None, tool_call_id: None, name: None });
                let usage = resp.usage.as_ref().map(|u| serde_json::json!({"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}));
                Json(serde_json::json!({"response": output, "conversation_id": conv_id, "usage": usage}))
            }
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        }
    } else {
        Json(serde_json::json!({"error": "NPC not found"}))
    }
}

async fn check_command(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let command = body["command"].as_str().unwrap_or("").to_string();
    let (npc, jinxes) = {
        let s = state.lock().await;
        (s.team.get_npc(&s.active_npc_name).cloned(), s.team.jinxes.clone())
    };

    if let Some(npc) = npc {
        let model = npc.resolved_model();
        let provider = npc.resolved_provider();
        match crate::llm_funcs::get_llm_response(&command, Some(&npc), Some(&model), Some(&provider), None, &[], None).await {
            Ok(result) => Json(serde_json::json!({"output": result.response, "model": result.model, "provider": result.provider})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        }
    } else {
        Json(serde_json::json!({"error": "No active NPC"}))
    }
}

async fn execute_jinx(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let jinx_name = body["jinx"].as_str().unwrap_or("");
    let args: HashMap<String, String> = body["args"].as_object()
        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect())
        .unwrap_or_default();

    let state = state.lock().await;
    if let Some(j) = state.team.jinxes.get(jinx_name) {
        let result = j.execute(&args);
        Json(serde_json::json!({"output": result.output, "success": result.success}))
    } else {
        Json(serde_json::json!({"error": format!("Jinx '{}' not found", jinx_name)}))
    }
}

async fn get_models() -> impl IntoResponse {
    let mut models: HashMap<&str, Vec<&str>> = HashMap::new();
    if std::env::var("OPENAI_API_KEY").is_ok() { models.insert("openai", vec!["gpt-4o", "gpt-4o-mini"]); }
    if std::env::var("ANTHROPIC_API_KEY").is_ok() { models.insert("anthropic", vec!["claude-sonnet-4", "claude-haiku-4"]); }
    if std::env::var("GEMINI_API_KEY").is_ok() || std::env::var("GOOGLE_API_KEY").is_ok() { models.insert("gemini", vec!["gemini-2.5-flash", "gemini-2.5-pro"]); }
    models.insert("ollama", vec!["llama3.2", "qwen3.5:2b"]);
    Json(serde_json::json!({"models": models}))
}

async fn list_conversations(State(state): State<AppState>) -> impl IntoResponse {
    let state = state.lock().await;
    let ids: Vec<&str> = state.conversations.keys().map(|s| s.as_str()).collect();
    Json(serde_json::json!({"conversations": ids}))
}

async fn create_conversation(State(state): State<AppState>) -> impl IntoResponse {
    let conv_id = crate::memory::start_new_conversation();
    let mut state = state.lock().await;
    state.conversations.insert(conv_id.clone(), Vec::new());
    Json(serde_json::json!({"conversation_id": conv_id}))
}

async fn search(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let query = body["query"].as_str().unwrap_or("");
    let provider = body["provider"].as_str().unwrap_or("duckduckgo");
    let num = body["num_results"].as_u64().unwrap_or(5) as usize;
    match crate::data::web::search_web(query, num, provider, None).await {
        Ok(results) => {
            let items: Vec<serde_json::Value> = results.iter().map(|r| serde_json::json!({"title": r.title, "url": r.url, "snippet": r.snippet})).collect();
            Json(serde_json::json!({"results": items}))
        }
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn generate_image(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let prompt = body["prompt"].as_str().unwrap_or("");
    let model = body["model"].as_str().unwrap_or("dall-e-3");
    let provider = body["provider"].as_str().unwrap_or("openai");
    match crate::llm_funcs::gen_image(prompt, Some(model), Some(provider), None, 1024, 1024, None).await {
        Ok(img) => {
            use base64::Engine;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&img.data);
            Json(serde_json::json!({"image": b64, "format": img.format}))
        }
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn tts(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let text = body["text"].as_str().unwrap_or("");
    let engine = body["engine"].as_str().unwrap_or("openai");
    let voice = body["voice"].as_str();
    match crate::r#gen::audio_gen::text_to_speech(text, engine, voice, None).await {
        Ok(audio) => {
            use base64::Engine;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&audio);
            Json(serde_json::json!({"audio": b64}))
        }
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn stt(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let engine = body["engine"].as_str().unwrap_or("openai");
    if let Some(b64) = body["audio"].as_str() {
        use base64::Engine;
        match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(data) => match crate::data::audio::speech_to_text(&data, engine, None).await {
                Ok(result) => Json(serde_json::json!(result)),
                Err(e) => Json(serde_json::json!({"error": e.to_string()})),
            },
            Err(e) => Json(serde_json::json!({"error": format!("Base64: {}", e)})),
        }
    } else {
        Json(serde_json::json!({"error": "Missing audio"}))
    }
}

async fn get_settings(State(state): State<AppState>) -> impl IntoResponse {
    let state = state.lock().await;
    Json(serde_json::json!({
        "active_npc": state.active_npc_name,
        "model": state.team.model,
        "provider": state.team.provider,
    }))
}

async fn save_settings(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let mut state = state.lock().await;
    if let Some(npc) = body["active_npc"].as_str() { state.active_npc_name = npc.to_string(); }
    if let Some(model) = body["model"].as_str() { state.team.model = Some(model.to_string()); }
    if let Some(provider) = body["provider"].as_str() { state.team.provider = Some(provider.to_string()); }
    Json(serde_json::json!({"status": "updated"}))
}

async fn kg_stats() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn kg_search(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let _query = body["query"].as_str().unwrap_or("");
    Json(serde_json::json!({"results": []}))
}

async fn capture() -> impl IntoResponse {
    match crate::data::image::capture_screenshot(true) {
        Ok(result) => Json(serde_json::json!(result)),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn get_kg_facts() -> impl IntoResponse {
    Json(serde_json::json!({"facts": []}))
}

async fn get_kg_concepts() -> impl IntoResponse {
    Json(serde_json::json!({"concepts": []}))
}

async fn add_kg_node(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let name = body["name"].as_str().unwrap_or("");
    let node_type = body["type"].as_str().unwrap_or("Entity");
    let content = body["content"].as_str().unwrap_or("");
    Json(serde_json::json!({"status": "added", "name": name, "type": node_type, "content": content}))
}

async fn update_kg_node(AxumPath(node_id): AxumPath<String>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    Json(serde_json::json!({"status": "updated", "node_id": node_id}))
}

async fn delete_kg_node(AxumPath(node_id): AxumPath<String>) -> impl IntoResponse {
    Json(serde_json::json!({"status": "deleted", "node_id": node_id}))
}

async fn add_kg_edge(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let source = body["source"].as_str().unwrap_or("");
    let target = body["target"].as_str().unwrap_or("");
    let relation = body["relation"].as_str().unwrap_or("related_to");
    Json(serde_json::json!({"status": "added", "source": source, "target": target, "relation": relation}))
}

async fn trigger_kg_process(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let process_type = body["process"].as_str().unwrap_or("sleep");
    Json(serde_json::json!({"status": "triggered", "process": process_type}))
}

async fn ingest_to_kg(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let text = body["text"].as_str().unwrap_or("");
    Json(serde_json::json!({"status": "ingested", "text_length": text.len()}))
}

async fn query_kg(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let query = body["query"].as_str().unwrap_or("");
    let mode = body["mode"].as_str().unwrap_or("keyword");
    Json(serde_json::json!({"query": query, "mode": mode, "results": []}))
}

async fn save_jinx(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let name = body["name"].as_str().unwrap_or("");
    let description = body["description"].as_str().unwrap_or("");
    let state = state.lock().await;
    if let Some(dir) = &state.team.source_dir {
        let jinx_dir = std::path::Path::new(dir).join("jinxes");
        let _ = std::fs::create_dir_all(&jinx_dir);
        let path = jinx_dir.join(format!("{}.jinx", name));
        let content = serde_yaml::to_string(&body).unwrap_or_default();
        match std::fs::write(&path, content) {
            Ok(_) => Json(serde_json::json!({"status": "saved", "path": path.to_string_lossy()})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        }
    } else {
        Json(serde_json::json!({"error": "No team source directory"}))
    }
}

async fn delete_jinx(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let name = body["name"].as_str().unwrap_or("");
    let state = state.lock().await;
    if let Some(dir) = &state.team.source_dir {
        let path = std::path::Path::new(dir).join("jinxes").join(format!("{}.jinx", name));
        match std::fs::remove_file(&path) {
            Ok(_) => Json(serde_json::json!({"status": "deleted"})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        }
    } else {
        Json(serde_json::json!({"error": "No team source directory"}))
    }
}

async fn test_jinx(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let jinx_name = body["jinx"].as_str().unwrap_or("");
    let args: HashMap<String, String> = body["args"].as_object()
        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect())
        .unwrap_or_default();
    let state = state.lock().await;
    if let Some(j) = state.team.jinxes.get(jinx_name) {
        let result = j.execute(&args);
        Json(serde_json::json!({"output": result.output, "success": result.success, "error": result.error}))
    } else {
        Json(serde_json::json!({"error": format!("Jinx '{}' not found", jinx_name)}))
    }
}

async fn get_attachments(AxumPath(message_id): AxumPath<String>) -> impl IntoResponse {
    let db_path = crate::npc_sysenv::get_history_db_path();
    match crate::memory::CommandHistory::open(&db_path) {
        Ok(history) => match history.get_message_attachments(&message_id) {
            Ok(attachments) => Json(serde_json::json!({"attachments": attachments})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn get_global_settings() -> impl IntoResponse {
    let rc_path = crate::npc_sysenv::get_npcshrc_path();
    match std::fs::read_to_string(&rc_path) {
        Ok(content) => Json(serde_json::json!({"settings": content})),
        Err(_) => Json(serde_json::json!({"settings": ""})),
    }
}

async fn save_global_settings(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let content = body["settings"].as_str().unwrap_or("");
    let rc_path = crate::npc_sysenv::get_npcshrc_path();
    if let Some(parent) = rc_path.parent() { let _ = std::fs::create_dir_all(parent); }
    match std::fs::write(&rc_path, content) {
        Ok(_) => Json(serde_json::json!({"status": "saved"})),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn get_project_settings(State(state): State<AppState>) -> impl IntoResponse {
    let state = state.lock().await;
    Json(serde_json::json!({
        "model": state.team.model,
        "provider": state.team.provider,
        "context": state.team.context,
        "forenpc": state.team.forenpc,
        "npcs": state.team.npc_names(),
        "jinxes": state.team.jinx_names(),
    }))
}

async fn save_project_settings(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let mut state = state.lock().await;
    if let Some(model) = body["model"].as_str() { state.team.model = Some(model.to_string()); }
    if let Some(provider) = body["provider"].as_str() { state.team.provider = Some(provider.to_string()); }
    if let Some(context) = body["context"].as_str() { state.team.context = Some(context.to_string()); }
    Json(serde_json::json!({"status": "saved"}))
}

async fn extract_memories(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let messages = body["messages"].as_array();
    let npc_name = body["npc"].as_str().unwrap_or("assistant");
    let model = body["model"].as_str();
    let provider = body["provider"].as_str();

    if let Some(msgs) = messages {
        let conversation: String = msgs.iter()
            .filter_map(|m| {
                let role = m["role"].as_str().unwrap_or("");
                let content = m["content"].as_str().unwrap_or("");
                if content.is_empty() { None } else { Some(format!("{}: {}", role, content)) }
            })
            .collect::<Vec<_>>().join("\n");

        let m = model.unwrap_or("qwen3.5:2b");
        let p = provider.unwrap_or("ollama");
        let prompt = format!(
            "Extract memories from this conversation that would be useful to remember about the user.\n\n{}\n\nReturn JSON: {{\"memories\": [\"memory1\", \"memory2\"]}}",
            conversation
        );
        match crate::llm_funcs::get_llm_response_ext(&prompt, None, Some(m), Some(p), None, &[], None, Some("json"), None, false).await {
            Ok(result) => {
                let memories = result.response_json.and_then(|j| j.get("memories").and_then(|m| m.as_array()).cloned()).unwrap_or_default();
                Json(serde_json::json!({"memories": memories, "npc": npc_name}))
            }
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        }
    } else {
        Json(serde_json::json!({"error": "Missing messages"}))
    }
}

async fn api_command(State(state): State<AppState>, AxumPath(command): AxumPath<String>) -> impl IntoResponse {
    let (npc, model, provider) = {
        let s = state.lock().await;
        let npc = s.team.get_npc(&s.active_npc_name).cloned();
        let m = npc.as_ref().map(|n| n.resolved_model()).unwrap_or_else(|| "qwen3.5:2b".into());
        let p = npc.as_ref().map(|n| n.resolved_provider()).unwrap_or_else(|| "ollama".into());
        (npc, m, p)
    };
    match crate::llm_funcs::get_llm_response(&command, npc.as_ref(), Some(&model), Some(&provider), None, &[], None).await {
        Ok(result) => Json(serde_json::json!({"response": result.response, "model": result.model})),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn get_pending_memories_route() -> impl IntoResponse {
    let db_path = crate::npc_sysenv::get_history_db_path();
    match crate::memory::CommandHistory::open(&db_path) {
        Ok(h) => match h.get_pending_memories() {
            Ok(mems) => Json(serde_json::json!({"memories": mems.iter().map(|(id, npc, content)| serde_json::json!({"id": id, "npc": npc, "content": content})).collect::<Vec<_>>()})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn approve_memories(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let db_path = crate::npc_sysenv::get_history_db_path();
    let memory_ids = body["ids"].as_array();
    let status = body["status"].as_str().unwrap_or("approved");
    match crate::memory::CommandHistory::open(&db_path) {
        Ok(h) => {
            if let Some(ids) = memory_ids {
                for id in ids {
                    if let Some(id) = id.as_i64() {
                        let _ = h.update_memory_status(id, status, None);
                    }
                }
            }
            Json(serde_json::json!({"status": "updated"}))
        }
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn search_memories(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let query = body["query"].as_str().unwrap_or("");
    let npc = body["npc"].as_str();
    let limit = body["limit"].as_u64().unwrap_or(20) as usize;
    let db_path = crate::npc_sysenv::get_history_db_path();
    match crate::memory::CommandHistory::open(&db_path) {
        Ok(h) => match h.search_memory(query, npc, None, limit) {
            Ok(results) => Json(serde_json::json!({"results": results})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn search_conversations_route(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let query = body["query"].as_str().unwrap_or("");
    let db_path = crate::npc_sysenv::get_history_db_path();
    match crate::memory::CommandHistory::open(&db_path) {
        Ok(h) => match h.search_conversations(query) {
            Ok(results) => {
                let items: Vec<serde_json::Value> = results.iter().map(|m| serde_json::json!({"message_id": m.message_id, "role": m.role, "content": m.content})).collect();
                Json(serde_json::json!({"results": items}))
            }
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn get_conversation_messages(AxumPath(id): AxumPath<String>) -> impl IntoResponse {
    let db_path = crate::npc_sysenv::get_history_db_path();
    match crate::memory::CommandHistory::open(&db_path) {
        Ok(h) => match h.load_conversation_messages(&id) {
            Ok(msgs) => {
                let items: Vec<serde_json::Value> = msgs.iter().map(|m| serde_json::json!({"message_id": m.message_id, "role": m.role, "content": m.content, "model": m.model, "npc": m.npc})).collect();
                Json(serde_json::json!({"messages": items}))
            }
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn delete_message(AxumPath((conv_id, msg_id)): AxumPath<(String, String)>) -> impl IntoResponse {
    let db_path = crate::npc_sysenv::get_history_db_path();
    match crate::memory::CommandHistory::open(&db_path) {
        Ok(h) => match h.delete_message(&conv_id, &msg_id) {
            Ok(_) => Json(serde_json::json!({"status": "deleted"})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn save_npc(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let name = body["name"].as_str().unwrap_or("");
    let state = state.lock().await;
    if let Some(npc) = state.team.get_npc(name) {
        match npc.save(state.team.source_dir.as_deref()) {
            Ok(_) => Json(serde_json::json!({"status": "saved"})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        }
    } else {
        Json(serde_json::json!({"error": format!("NPC '{}' not found", name)}))
    }
}

async fn team_sync_status() -> impl IntoResponse {
    match crate::npc_sysenv::team_sync_status(None) {
        Ok(status) => Json(serde_json::json!(status)),
        Err(e) => Json(serde_json::json!({"error": e})),
    }
}

async fn team_sync_init() -> impl IntoResponse {
    match crate::npc_sysenv::team_sync_init(None) {
        Ok(msg) => Json(serde_json::json!({"status": msg})),
        Err(e) => Json(serde_json::json!({"error": e})),
    }
}

async fn team_sync_pull() -> impl IntoResponse {
    match crate::npc_sysenv::team_sync_pull(None) {
        Ok(msg) => Json(serde_json::json!({"status": msg})),
        Err(e) => Json(serde_json::json!({"error": e})),
    }
}

async fn team_sync_commit(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let message = body["message"].as_str().unwrap_or("Update NPC team");
    match crate::npc_sysenv::team_sync_commit(None, message) {
        Ok(msg) => Json(serde_json::json!({"status": msg})),
        Err(e) => Json(serde_json::json!({"error": e})),
    }
}

async fn team_sync_diff() -> impl IntoResponse {
    match crate::npc_sysenv::team_sync_diff(None, None) {
        Ok(diff) => Json(serde_json::json!({"diff": diff})),
        Err(e) => Json(serde_json::json!({"error": e})),
    }
}

async fn ollama_status() -> impl IntoResponse {
    match reqwest::get("http://localhost:11434/api/tags").await {
        Ok(resp) if resp.status().is_success() => Json(serde_json::json!({"status": "running"})),
        _ => Json(serde_json::json!({"status": "not_running"})),
    }
}

async fn get_ollama_models() -> impl IntoResponse {
    match reqwest::get("http://localhost:11434/api/tags").await {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(data) => Json(data),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn generate_video_api(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let prompt = body["prompt"].as_str().unwrap_or("");
    let model = body["model"].as_str();
    let provider = body["provider"].as_str();
    let output = body["output_path"].as_str().unwrap_or("/tmp/npc_video.mp4");
    match crate::llm_funcs::gen_video(prompt, model, provider, None, output).await {
        Ok(result) => Json(serde_json::json!(result)),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn get_npc_executions_route(State(state): State<AppState>) -> impl IntoResponse {
    let npc_name = { state.lock().await.active_npc_name.clone() };
    let db_path = crate::npc_sysenv::get_history_db_path();
    match crate::memory::CommandHistory::open(&db_path) {
        Ok(h) => match h.get_npc_executions(&npc_name, 100) {
            Ok(execs) => Json(serde_json::json!({"executions": execs})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn get_jinx_executions_route(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let jinx_name = body["jinx_name"].as_str();
    let db_path = crate::npc_sysenv::get_history_db_path();
    match crate::memory::CommandHistory::open(&db_path) {
        Ok(h) => match h.get_jinx_executions(jinx_name, 100) {
            Ok(execs) => Json(serde_json::json!({"executions": execs})),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})),
        },
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

pub async fn start_http_server(state: AppState, config: &ServerConfig) -> Result<()> {
    let app = create_app(state);
    let addr = format!("{}:{}", config.host, config.http_port);
    let listener = tokio::net::TcpListener::bind(&addr).await
        .map_err(|e| NpcError::Other(format!("Bind {}: {}", addr, e)))?;
    tracing::info!("NPC server on {}", addr);
    axum::serve(listener, app).await
        .map_err(|e| NpcError::Other(format!("Server: {}", e)))?;
    Ok(())
}

pub async fn start_mcp_server(state: AppState) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);

    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await.map_err(|e| NpcError::Other(e.to_string()))?;
        if n == 0 { break; }

        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        let request: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let id = request.get("id").cloned();
        let method = request["method"].as_str().unwrap_or("");
        let params = request.get("params").cloned().unwrap_or_default();

        let result = match method {
            "initialize" => serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": true}, "prompts": {"listChanged": true}},
                "serverInfo": {"name": "npcrs-mcp", "version": env!("CARGO_PKG_VERSION")}
            }),
            "tools/list" => {
                let state = state.lock().await;
                let active_npc = state.team.get_npc(&state.active_npc_name);
                let jinx_names: Vec<&str> = if let Some(npc) = active_npc {
                    npc.jinx_names.iter().map(|s| s.as_str()).collect()
                } else {
                    state.team.jinx_names()
                };
                let tools: Vec<serde_json::Value> = jinx_names.iter()
                    .filter_map(|name| state.team.jinxes.get(*name))
                    .filter_map(|jinx| jinx.to_tool_def())
                    .map(|td| serde_json::to_value(&td).unwrap_or_default())
                    .collect();
                serde_json::json!({"tools": tools})
            }
            "tools/call" => {
                let tool_name = params["name"].as_str().unwrap_or("");
                let args: HashMap<String, String> = params["arguments"].as_object()
                    .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect())
                    .unwrap_or_default();
                let state = state.lock().await;
                if let Some(j) = state.team.jinxes.get(tool_name) {
                    let result = j.execute(&args);
                    serde_json::json!({"content": [{"type": "text", "text": result.output}], "isError": !result.success})
                } else {
                    serde_json::json!({"content": [{"type": "text", "text": format!("Tool '{}' not found", tool_name)}], "isError": true})
                }
            }
            "prompts/list" => {
                let state = state.lock().await;
                let prompts: Vec<serde_json::Value> = state.team.npc_names().iter()
                    .map(|name| serde_json::json!({"name": format!("npc_{}", name), "description": format!("Switch to {}", name)}))
                    .collect();
                serde_json::json!({"prompts": prompts})
            }
            "notifications/initialized" => continue,
            _ => serde_json::json!({"error": {"code": -32601, "message": format!("Method not found: {}", method)}}),
        };

        if let Some(id) = id {
            let response = serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result});
            let s = serde_json::to_string(&response).unwrap_or_default();
            let header = format!("Content-Length: {}\r\n\r\n", s.len());
            stdout.write_all(header.as_bytes()).await.map_err(|e| NpcError::Other(e.to_string()))?;
            stdout.write_all(s.as_bytes()).await.map_err(|e| NpcError::Other(e.to_string()))?;
            stdout.flush().await.map_err(|e| NpcError::Other(e.to_string()))?;
        }
    }
    Ok(())
}

pub async fn start_servers(team: Team, config: ServerConfig) -> Result<()> {
    let active_npc = team.lead_npc().map(|n| n.name.clone()).unwrap_or_else(|| "assistant".to_string());
    let state = Arc::new(Mutex::new(ServerState { team, active_npc_name: active_npc, conversations: HashMap::new() }));

    if config.mcp_enabled {
        let http_state = Arc::clone(&state);
        let mcp_state = Arc::clone(&state);
        tokio::select! {
            result = start_http_server(http_state, &config) => { if let Err(e) = result { tracing::error!("HTTP: {}", e); } }
            result = start_mcp_server(mcp_state) => { if let Err(e) = result { tracing::error!("MCP: {}", e); } }
        }
    } else {
        start_http_server(state, &config).await?;
    }
    Ok(())
}
