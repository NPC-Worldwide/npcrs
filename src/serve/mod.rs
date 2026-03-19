
use crate::error::{NpcError, Result};
use crate::npc_compiler;
use crate::r#gen::Message;
use crate::npc_compiler::Team;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

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

pub async fn start_http_server(
    state: Arc<Mutex<ServerState>>,
    config: &ServerConfig,
) -> Result<()> {
    let addr = format!("{}:{}", config.host, config.http_port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| NpcError::Other(format!("Failed to bind {}: {}", addr, e)))?;

    tracing::info!("NPC HTTP server listening on {}", addr);

    loop {
        let (stream, _) = listener
            .accept()
            .await
            .map_err(|e| NpcError::Other(format!("Accept failed: {}", e)))?;

        let state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, state).await {
                tracing::error!("Connection error: {}", e);
            }
        });
    }
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    state: Arc<Mutex<ServerState>>,
) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut buf = vec![0u8; 65536];
    let n = stream
        .read(&mut buf)
        .await
        .map_err(|e| NpcError::Other(e.to_string()))?;
    let request = String::from_utf8_lossy(&buf[..n]);

    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let (method, path) = if parts.len() >= 2 {
        (parts[0], parts[1])
    } else {
        ("GET", "/")
    };

    let body = request.split("\r\n\r\n").nth(1).unwrap_or("");

    let response_body = match (method, path) {
        ("GET", "/health") => serde_json::json!({"status": "ok"}).to_string(),
        ("GET", "/api/team") => {
            let state = state.lock().await;
            let npc_names: Vec<&str> = state.team.npc_names();
            let jinx_names: Vec<&str> = state.team.jinx_names();
            serde_json::json!({
                "npcs": npc_names,
                "jinxes": jinx_names,
                "active_npc": state.active_npc_name,
            })
            .to_string()
        }
        ("POST", "/api/chat") => {
            let parsed: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
            let input = parsed["message"].as_str().unwrap_or("");
            let npc_name = parsed["npc"].as_str();

            let mut state = state.lock().await;
            let npc = if let Some(name) = npc_name {
                state.team.get_npc(name).cloned()
            } else {
                state.team.get_npc(&state.active_npc_name).cloned()
            };

            if let Some(npc) = npc {
                let system = npc.system_prompt(state.team.context.as_deref());
                let messages = vec![Message::system(system), Message::user(input)];

                match crate::r#gen::get_genai_response(
                        &npc.resolved_provider(),
                        &npc.resolved_model(),
                        &messages,
                        None,
                        npc.api_url.as_deref(),
                    )
                    .await
                {
                    Ok(resp) => {
                        let output = resp.message.content.unwrap_or_default();
                        serde_json::json!({"response": output}).to_string()
                    }
                    Err(e) => serde_json::json!({"error": e.to_string()}).to_string(),
                }
            } else {
                serde_json::json!({"error": "NPC not found"}).to_string()
            }
        }
        ("POST", "/api/jinx") => {
            let parsed: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
            let jinx_name = parsed["jinx"].as_str().unwrap_or("");
            let args: HashMap<String, String> = parsed["args"]
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                        .collect()
                })
                .unwrap_or_default();

            let state = state.lock().await;
            if let Some(j) = state.team.jinxes.get(jinx_name) {
                match npc_compiler::execute_jinx(j, &args, &state.team.jinxes).await {
                    Ok(result) => {
                        serde_json::json!({"output": result.output, "success": result.success})
                            .to_string()
                    }
                    Err(e) => serde_json::json!({"error": e.to_string()}).to_string(),
                }
            } else {
                serde_json::json!({"error": format!("Jinx '{}' not found", jinx_name)}).to_string()
            }
        }
        _ => serde_json::json!({"error": "Not found"}).to_string(),
    };

    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    );

    stream
        .write_all(http_response.as_bytes())
        .await
        .map_err(|e| NpcError::Other(e.to_string()))?;

    Ok(())
}

pub async fn start_mcp_server(state: Arc<Mutex<ServerState>>) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);

    tracing::info!("NPC MCP server started on stdio");

    let mut line = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .await
            .map_err(|e| NpcError::Other(e.to_string()))?;
        if n == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

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
                "capabilities": {
                    "tools": { "listChanged": true },
                    "prompts": { "listChanged": true },
                    "resources": {},
                },
                "serverInfo": {
                    "name": "npcrs-mcp",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            }),
            "tools/list" => {
                let state = state.lock().await;
                let active_npc = state.team.get_npc(&state.active_npc_name);
                let jinx_names: Vec<&str> = if let Some(npc) = active_npc {
                    npc.jinx_names.iter().map(|s| s.as_str()).collect()
                } else {
                    state.team.jinx_names()
                };

                let tools: Vec<serde_json::Value> = jinx_names
                    .iter()
                    .filter_map(|name| state.team.jinxes.get(*name))
                    .filter_map(|jinx| jinx.to_tool_def())
                    .map(|td| serde_json::to_value(&td).unwrap_or_default())
                    .collect();

                serde_json::json!({"tools": tools})
            }
            "tools/call" => {
                let tool_name = params["name"].as_str().unwrap_or("");
                let args: HashMap<String, String> = params["arguments"]
                    .as_object()
                    .map(|obj| {
                        obj.iter()
                            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                            .collect()
                    })
                    .unwrap_or_default();

                let state = state.lock().await;
                if let Some(j) = state.team.jinxes.get(tool_name) {
                    match npc_compiler::execute_jinx(j, &args, &state.team.jinxes).await {
                        Ok(result) => serde_json::json!({
                            "content": [{"type": "text", "text": result.output}],
                            "isError": !result.success,
                        }),
                        Err(e) => serde_json::json!({
                            "content": [{"type": "text", "text": format!("Error: {}", e)}],
                            "isError": true,
                        }),
                    }
                } else {
                    serde_json::json!({
                        "content": [{"type": "text", "text": format!("Tool '{}' not found", tool_name)}],
                        "isError": true,
                    })
                }
            }
            "prompts/list" => {
                let state = state.lock().await;
                let prompts: Vec<serde_json::Value> = state
                    .team
                    .npc_names()
                    .iter()
                    .map(|name| {
                        serde_json::json!({
                            "name": format!("npc_{}", name),
                            "description": format!("Switch to {} NPC", name),
                        })
                    })
                    .collect();
                serde_json::json!({"prompts": prompts})
            }
            "notifications/initialized" => continue,
            _ => serde_json::json!({"error": {"code": -32601, "message": format!("Method not found: {}", method)}}),
        };

        let response = if let Some(id) = id {
            serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result})
        } else {
            continue;
        };

        let response_str = serde_json::to_string(&response).unwrap_or_default();
        let header = format!("Content-Length: {}\r\n\r\n", response_str.len());

        stdout
            .write_all(header.as_bytes())
            .await
            .map_err(|e| NpcError::Other(e.to_string()))?;
        stdout
            .write_all(response_str.as_bytes())
            .await
            .map_err(|e| NpcError::Other(e.to_string()))?;
        stdout
            .flush()
            .await
            .map_err(|e| NpcError::Other(e.to_string()))?;
    }

    Ok(())
}

pub async fn start_servers(team: Team, config: ServerConfig) -> Result<()> {
    let active_npc = team
        .lead_npc()
        .map(|n| n.name.clone())
        .unwrap_or_else(|| "assistant".to_string());

    let state = Arc::new(Mutex::new(ServerState {
        team,
        active_npc_name: active_npc,
        conversations: HashMap::new(),
    }));

    if config.mcp_enabled {
        let http_state = Arc::clone(&state);
        let mcp_state = Arc::clone(&state);

        tokio::select! {
            result = start_http_server(http_state, &config) => {
                if let Err(e) = result {
                    tracing::error!("HTTP server error: {}", e);
                }
            }
            result = start_mcp_server(mcp_state) => {
                if let Err(e) = result {
                    tracing::error!("MCP server error: {}", e);
                }
            }
        }
    } else {
        start_http_server(state, &config).await?;
    }

    Ok(())
}
