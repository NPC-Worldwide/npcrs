use crate::error::{NpcError, Result};
use crate::mcp::McpTool;
use crate::npc_compiler::McpServerSpec;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};

pub struct McpClient {
    child: Child,
    request_id: u64,
}

#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    id: Option<u64>,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

impl McpClient {
    pub async fn connect(spec: &McpServerSpec) -> Result<Self> {
        let child = if let Some(ref command) = spec.command {
            let parts: Vec<&str> = command.split_whitespace().collect();
            let (cmd, args) = parts.split_first().ok_or_else(|| {
                NpcError::Mcp("Empty command".to_string())
            })?;

            Command::new(cmd)
                .args(args)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
                .map_err(|e| NpcError::Mcp(format!("Failed to spawn MCP server: {}", e)))?
        } else {
            let path = &spec.path;
            let (cmd, args): (&str, Vec<&str>) = if path.ends_with(".py") {
                ("python3", vec![path.as_str()])
            } else if path.ends_with(".js") {
                ("node", vec![path.as_str()])
            } else {
                (path.as_str(), vec![])
            };

            Command::new(cmd)
                .args(&args)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
                .map_err(|e| NpcError::Mcp(format!("Failed to spawn MCP server: {}", e)))?
        };

        let mut client = Self {
            child,
            request_id: 0,
        };

        client.initialize().await?;

        Ok(client)
    }

    async fn initialize(&mut self) -> Result<()> {
        let _resp = self
            .send_request(
                "initialize",
                Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "npcrs",
                        "version": env!("CARGO_PKG_VERSION"),
                    }
                })),
            )
            .await?;

        self.send_notification("notifications/initialized", None)
            .await?;

        Ok(())
    }

    pub async fn list_tools(&mut self) -> Result<Vec<McpTool>> {
        let resp = self
            .send_request("tools/list", None)
            .await?;

        let tools_value = resp
            .get("tools")
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![]));

        #[derive(Deserialize)]
        struct ToolEntry {
            name: String,
            description: Option<String>,
            #[serde(rename = "inputSchema")]
            input_schema: Option<serde_json::Value>,
        }

        let entries: Vec<ToolEntry> =
            serde_json::from_value(tools_value).map_err(|e| {
                NpcError::Mcp(format!("Failed to parse tools: {}", e))
            })?;

        Ok(entries
            .into_iter()
            .map(|e| McpTool {
                name: e.name,
                description: e.description,
                input_schema: e.input_schema.unwrap_or(serde_json::json!({"type": "object"})),
                server_path: String::new(),
            })
            .collect())
    }

    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String> {
        let resp = self
            .send_request(
                "tools/call",
                Some(serde_json::json!({
                    "name": name,
                    "arguments": arguments,
                })),
            )
            .await?;

        if let Some(content) = resp.get("content") {
            if let Some(arr) = content.as_array() {
                let texts: Vec<&str> = arr
                    .iter()
                    .filter_map(|block| {
                        if block.get("type")?.as_str()? == "text" {
                            block.get("text")?.as_str()
                        } else {
                            None
                        }
                    })
                    .collect();
                return Ok(texts.join("\n"));
            }
        }

        Ok(serde_json::to_string_pretty(&resp).unwrap_or_default())
    }

    async fn send_request(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        self.request_id += 1;
        let id = self.request_id;

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        };

        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or_else(|| NpcError::Mcp("No stdin".to_string()))?;
        let mut payload = serde_json::to_string(&request)?;
        payload.push('\n');
        stdin
            .write_all(payload.as_bytes())
            .await
            .map_err(|e| NpcError::Mcp(format!("Write failed: {}", e)))?;
        stdin
            .flush()
            .await
            .map_err(|e| NpcError::Mcp(format!("Flush failed: {}", e)))?;

        let stdout = self
            .child
            .stdout
            .as_mut()
            .ok_or_else(|| NpcError::Mcp("No stdout".to_string()))?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .await
            .map_err(|e| NpcError::Mcp(format!("Read failed: {}", e)))?;

        let resp: JsonRpcResponse = serde_json::from_str(&line)
            .map_err(|e| NpcError::Mcp(format!("Parse response failed: {}", e)))?;

        if let Some(error) = resp.error {
            return Err(NpcError::Mcp(format!(
                "MCP error {}: {}",
                error.code, error.message
            )));
        }

        Ok(resp.result.unwrap_or(serde_json::Value::Null))
    }

    async fn send_notification(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<()> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params.unwrap_or(serde_json::Value::Object(Default::default())),
        });

        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or_else(|| NpcError::Mcp("No stdin".to_string()))?;
        let mut payload = serde_json::to_string(&notification)?;
        payload.push('\n');
        stdin
            .write_all(payload.as_bytes())
            .await
            .map_err(|e| NpcError::Mcp(format!("Write failed: {}", e)))?;
        stdin
            .flush()
            .await
            .map_err(|e| NpcError::Mcp(format!("Flush failed: {}", e)))?;

        Ok(())
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}
