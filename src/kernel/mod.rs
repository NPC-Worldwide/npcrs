mod boot;
mod syscall;

pub use boot::*;
pub use syscall::*;

use crate::drivers::DriverManager;
use crate::error::{NpcError, Result};
use crate::r#gen::Message;
use crate::ipc::IpcBus;
use crate::memory::CommandHistory;
use crate::npc_compiler::NPC;
use crate::npc_compiler::Team;
use crate::npc_compiler::{self, Jinx};
use crate::process::{Capabilities, Pid, Process, ProcessState};
use crate::scheduler::Scheduler;
use crate::vfs::Vfs;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub struct Kernel {
    processes: HashMap<Pid, Process>,

    next_pid: AtomicU32,

    pub team: Team,

    pub jinxes: HashMap<String, Jinx>,

    pub drivers: DriverManager,

    pub vfs: Vfs,

    pub ipc: IpcBus,

    pub scheduler: Scheduler,

    pub history: CommandHistory,

    pub boot_time: chrono::DateTime<chrono::Utc>,

    pub python_daemon: Option<PythonDaemon>,
}

/// Client for the npcsh LLM daemon.
///
/// Supports two modes:
///   1. **Socket mode** — connects to a persistent Unix-domain socket (e.g.
///      `~/.npcsh/daemon.sock`).  The daemon runs as a background service
///      (`brew services` on macOS, systemd on Linux) and handles multiple
///      concurrent clients via threads.
///   2. **Subprocess mode** — spawns `python3 npcsh/daemon.py` as a child
///      process (fallback when the socket is not available).
pub struct PythonDaemon {
    mode: DaemonMode,
}

enum DaemonMode {
    Http {
        base_url: String,
        client: reqwest::Client,
    },
    #[cfg(unix)]
    Socket {
        writer: tokio::net::unix::OwnedWriteHalf,
        reader: tokio::io::BufReader<tokio::net::unix::OwnedReadHalf>,
    },
    Subprocess {
        child: Box<tokio::process::Child>,
        stdin: tokio::process::ChildStdin,
        stdout: tokio::io::BufReader<tokio::process::ChildStdout>,
        _stderr_task: tokio::task::JoinHandle<()>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    #[serde(rename = "type")]
    pub req_type: String,
    pub messages: Vec<crate::r#gen::Message>,
    pub model: String,
    pub provider: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<crate::r#gen::ToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attachments: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registered_teams: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub npc: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::r#gen::ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<crate::r#gen::Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub streamed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn parse_sse_event_data(event_text: &str) -> Option<String> {
    let mut data_lines: Vec<&str> = Vec::new();
    for line in event_text.lines() {
        let line = line.trim_start();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data_lines.is_empty() {
        None
    } else {
        Some(data_lines.join("\n"))
    }
}

impl PythonDaemon {
    /// Connect to the generic npcpy HTTP server.
    pub fn connect_http(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        tracing::info!("Connected to npcpy HTTP server at {}", base_url);
        Self {
            mode: DaemonMode::Http {
                base_url,
                client: reqwest::Client::new(),
            },
        }
    }

    /// Default local npcpy server URL.
    pub fn default_url() -> String {
        std::env::var("NPCPY_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:5337".to_string())
    }

    /// Connect to a persistent Unix socket daemon.
    /// Returns `Ok(Self)` if the socket is live, otherwise returns an error
    /// so the caller can decide whether to start a daemon.
    #[cfg(unix)]
    pub async fn connect() -> Result<Self> {
        use tokio::io::AsyncBufReadExt;

        let socket_path = Self::socket_path();
        let stream = tokio::net::UnixStream::connect(&socket_path)
            .await
            .map_err(|e| {
                NpcError::Other(format!(
                    "Failed to connect to daemon socket at {}: {}",
                    socket_path.display(),
                    e
                ))
            })?;
        let (read_half, write_half) = stream.into_split();
        let mut reader = tokio::io::BufReader::new(read_half);

        let mut ready_line = String::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            reader.read_line(&mut ready_line),
        )
        .await
        .map_err(|_| {
            NpcError::Other(format!(
                "Daemon on {} never sent ready signal",
                socket_path.display()
            ))
        })??;

        if !ready_line.contains("ready") {
            return Err(NpcError::Other(format!(
                "Daemon on {} sent unexpected ready line: {}",
                socket_path.display(),
                ready_line.trim()
            )));
        }

        tracing::info!("Connected to npcsh daemon on {}", socket_path.display());
        Ok(Self {
            mode: DaemonMode::Socket {
                writer: write_half,
                reader,
            },
        })
    }

    /// On non-Unix platforms, socket mode is not supported.
    #[cfg(not(unix))]
    pub async fn connect() -> Result<Self> {
        Err(NpcError::Other(
            "Unix socket daemon mode is not supported on this platform".into(),
        ))
    }

    /// Try to connect to the HTTP server first, then fall back to Unix socket,
    /// then subprocess.
    pub async fn spawn(_team_dir: &str, _db_path: &str) -> Result<Self> {
        let url = Self::default_url();
        if Self::http_alive(&url).await {
            return Ok(Self::connect_http(url));
        }
        if let Ok(d) = Self::connect().await {
            return Ok(d);
        }
        Self::spawn_subprocess(_team_dir, _db_path).await
    }

    async fn http_alive(url: &str) -> bool {
        reqwest::get(format!("{}/api/health", url))
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    pub fn socket_path() -> std::path::PathBuf {
        // Single canonical path so the Python daemon and Rust client agree.
        // Can be overridden with NPCSH_DAEMON_SOCKET for custom setups.
        if let Some(path) = std::env::var_os("NPCSH_DAEMON_SOCKET") {
            return std::path::PathBuf::from(path);
        }
        let mut p = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        p.push(".npcsh/daemon.sock");
        p
    }

    async fn spawn_subprocess(team_dir: &str, db_path: &str) -> Result<Self> {
        use tokio::io::{AsyncBufReadExt, BufReader};
        use tokio::process::Command;

        let daemon_script = Self::find_daemon_script();

        let mut child = if let Some(script) = daemon_script {
            Command::new("python3")
                .arg(&script)
                .env("NPCSH_DB_PATH", db_path)
                .env("NPCSH_TEAM_DIR", team_dir)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| {
                    NpcError::Other(format!(
                        "Failed to spawn Python daemon ({}): {}",
                        script.display(),
                        e
                    ))
                })?
        } else {
            // Fallback inline script (legacy, rarely used)
            Command::new("python3")
                .arg("-c")
                .arg(format!(
                    r#"
import sys, json, os
os.environ.setdefault('NPCSH_DB_PATH', '{}')
sys.path.insert(0, os.getcwd())
from npcsh._state import setup_shell, execute_slash_command, ShellState, initial_state
from npcsh.routes import router, CommandRouter
command_history, team, npc = setup_shell()
from npcsh._state import initialize_router_with_jinxes
initialize_router_with_jinxes(team, router)
state = initial_state
state.team = team
state.npc = npc
state.command_history = command_history
sys.stderr.write('npcsh-daemon: ready\n')
sys.stderr.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        cmd = req.get('command', '')
        stdin_input = req.get('stdin_input')
        state, result = execute_slash_command(cmd, stdin_input, state, False, router)
        if isinstance(result, dict):
            output = result.get('output', '')
        else:
            output = str(result) if result else ''
        resp = json.dumps({{"output": str(output), "ok": True}})
    except Exception as e:
        resp = json.dumps({{"output": f"Error: {{e}}", "ok": False}})
    print(resp, flush=True)
"#,
                    db_path
                ))
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| NpcError::Other(format!("Failed to spawn Python daemon: {}", e)))?
        };

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| NpcError::Other("No stdin on daemon".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| NpcError::Other("No stdout on daemon".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| NpcError::Other("No stderr on daemon".into()))?;

        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let mut stderr_reader = BufReader::new(stderr);

        let stderr_task = tokio::spawn(async move {
            let mut line = String::new();
            let mut found_ready = false;
            let mut ready_tx = Some(ready_tx);
            loop {
                line.clear();
                match stderr_reader.read_line(&mut line).await {
                    Ok(0) => break,
                    Ok(_) => {
                        if !found_ready && line.contains("ready") {
                            found_ready = true;
                            if let Some(tx) = ready_tx.take() {
                                let _ = tx.send(());
                            }
                            continue;
                        }
                        if found_ready {
                            // Forward [STREAM] content and [THINK] reasoning
                            let trimmed = line.trim_end_matches('\n');
                            if let Some(stripped) = trimmed.strip_prefix("[STREAM]") {
                                let unescaped = stripped.replace('\x01', "\n");
                                eprint!("{}", unescaped);
                            } else if let Some(stripped) = trimmed.strip_prefix("[THINK]") {
                                let unescaped = stripped.replace('\x01', "\n");
                                eprint!("\x1b[90m{}\x1b[0m", unescaped);
                            }
                            // Silently drop everything else
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        match tokio::time::timeout(std::time::Duration::from_secs(30), ready_rx).await {
            Ok(Ok(())) => {}
            _ => {
                return Err(NpcError::Other(
                    "Daemon failed to start: never sent ready signal".into(),
                ));
            }
        }

        Ok(Self {
            mode: DaemonMode::Subprocess {
                child: Box::new(child),
                stdin,
                stdout: BufReader::new(stdout),
                _stderr_task: stderr_task,
            },
        })
    }

    fn find_daemon_script() -> Option<std::path::PathBuf> {
        let candidates = ["npcsh/daemon.py", "../npcsh/daemon.py"];
        if let Ok(cwd) = std::env::current_dir() {
            for c in &candidates {
                let p = cwd.join(c);
                if p.exists() {
                    return Some(p);
                }
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let p = dir.join("npcsh/daemon.py");
                if p.exists() {
                    return Some(p);
                }
            }
        }
        if let Ok(output) = std::process::Command::new("python3")
            .args([
                "-c",
                "import npcsh, os; print(os.path.dirname(os.path.abspath(npcsh.__file__)))",
            ])
            .output()
        {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                let p = std::path::PathBuf::from(&path).join("daemon.py");
                if p.exists() {
                    return Some(p);
                }
            }
        }
        None
    }

    pub async fn llm(&mut self, request: &LlmRequest) -> Result<LlmResponse> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

        let mut line = serde_json::to_string(request).unwrap_or_default();
        line.push('\n');

        if let DaemonMode::Http { base_url, client } = &self.mode {
            let base_url = base_url.clone();
            let client = client.clone();
            return self.llm_http(&client, &base_url, request).await;
        }

        match &mut self.mode {
            #[cfg(unix)]
            DaemonMode::Socket { writer, reader } => {
                writer
                    .write_all(line.as_bytes())
                    .await
                    .map_err(|e| NpcError::Other(format!("Socket write: {}", e)))?;
                writer
                    .flush()
                    .await
                    .map_err(|e| NpcError::Other(format!("Socket flush: {}", e)))?;

                // Read lines from the socket until we get a JSON response.
                // Streaming chunks ([STREAM], [THINK]) are forwarded to the
                // terminal immediately.
                let mut resp_line = String::new();
                loop {
                    resp_line.clear();
                    match reader.read_line(&mut resp_line).await {
                        Ok(0) => {
                            return Err(NpcError::Other(
                                "Daemon closed socket before sending response".into(),
                            ));
                        }
                        Ok(_) => {
                            let trimmed = resp_line.trim_end_matches('\n');
                            if let Some(stripped) = trimmed.strip_prefix("[STREAM]") {
                                let unescaped = stripped.replace('\x01', "\n");
                                eprint!("{}", unescaped);
                                let _ = std::io::Write::flush(&mut std::io::stderr());
                            } else if let Some(stripped) = trimmed.strip_prefix("[THINK]") {
                                let unescaped = stripped.replace('\x01', "\n");
                                eprint!("\x1b[90m{}\x1b[0m", unescaped);
                                let _ = std::io::Write::flush(&mut std::io::stderr());
                            } else if trimmed.starts_with('{') {
                                // JSON response line
                                let resp: LlmResponse =
                                    serde_json::from_str(trimmed).map_err(|e| {
                                        NpcError::Other(format!(
                                            "Socket LLM parse: {} (raw: {})",
                                            e, trimmed
                                        ))
                                    })?;
                                return Ok(resp);
                            }
                            // Silently drop other noise
                        }
                        Err(e) => {
                            return Err(NpcError::Other(format!("Socket read: {}", e)));
                        }
                    }
                }
            }
            DaemonMode::Subprocess { stdin, stdout, .. } => {
                stdin
                    .write_all(line.as_bytes())
                    .await
                    .map_err(|e| NpcError::Other(format!("Daemon LLM write: {}", e)))?;
                stdin
                    .flush()
                    .await
                    .map_err(|e| NpcError::Other(format!("Daemon LLM flush: {}", e)))?;

                let mut resp_line = String::new();
                stdout
                    .read_line(&mut resp_line)
                    .await
                    .map_err(|e| NpcError::Other(format!("Daemon LLM read: {}", e)))?;

                let resp: LlmResponse = serde_json::from_str(&resp_line).map_err(|e| {
                    NpcError::Other(format!(
                        "Daemon LLM parse: {} (raw: {})",
                        e,
                        resp_line.trim()
                    ))
                })?;

                Ok(resp)
            }
            DaemonMode::Http { .. } => {
                unreachable!("HTTP mode is handled before match")
            }
        }
    }

    /// Send a raw command request to the daemon (used for legacy Python
    /// slash-command execution).  Only supported in subprocess mode for now.
    pub async fn execute(&mut self, command: &str, stdin_input: Option<&str>) -> Result<String> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

        let req = serde_json::json!({
            "command": command,
            "stdin_input": stdin_input,
        });
        let mut line = serde_json::to_string(&req).unwrap_or_default();
        line.push('\n');

        match &mut self.mode {
            DaemonMode::Http { base_url, client } => {
                let url = format!("{}/api/jinx/execute", base_url);
                let body = serde_json::json!({
                    "jinxName": command,
                    "stdin_input": stdin_input,
                });
                let resp =
                    client.post(&url).json(&body).send().await.map_err(|e| {
                        NpcError::Other(format!("HTTP execute request failed: {}", e))
                    })?;
                let status = resp.status();
                let text = resp
                    .text()
                    .await
                    .map_err(|e| NpcError::Other(format!("HTTP execute body failed: {}", e)))?;
                if !status.is_success() {
                    return Err(NpcError::Other(format!(
                        "HTTP execute returned {}: {}",
                        status, text
                    )));
                }
                let json: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
                Ok(json
                    .get("output")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&text)
                    .to_string())
            }
            #[cfg(unix)]
            DaemonMode::Socket { writer, reader } => {
                writer
                    .write_all(line.as_bytes())
                    .await
                    .map_err(|e| NpcError::Other(format!("Socket write: {}", e)))?;
                writer
                    .flush()
                    .await
                    .map_err(|e| NpcError::Other(format!("Socket flush: {}", e)))?;

                let mut resp_line = String::new();
                // Read lines until we hit JSON (ignoring any streaming noise)
                loop {
                    resp_line.clear();
                    reader
                        .read_line(&mut resp_line)
                        .await
                        .map_err(|e| NpcError::Other(format!("Socket read: {}", e)))?;
                    let trimmed = resp_line.trim_end_matches('\n');
                    if trimmed.starts_with('{') {
                        let resp: serde_json::Value =
                            serde_json::from_str(trimmed).map_err(|e| {
                                NpcError::Other(format!("Socket parse: {} (raw: {})", e, trimmed))
                            })?;
                        return Ok(resp
                            .get("output")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string());
                    }
                }
            }
            DaemonMode::Subprocess { stdin, stdout, .. } => {
                stdin
                    .write_all(line.as_bytes())
                    .await
                    .map_err(|e| NpcError::Other(format!("Daemon write: {}", e)))?;
                stdin
                    .flush()
                    .await
                    .map_err(|e| NpcError::Other(format!("Daemon flush: {}", e)))?;

                let mut resp_line = String::new();
                stdout
                    .read_line(&mut resp_line)
                    .await
                    .map_err(|e| NpcError::Other(format!("Daemon read: {}", e)))?;

                let resp: serde_json::Value = serde_json::from_str(&resp_line).map_err(|e| {
                    NpcError::Other(format!("Daemon parse: {} (raw: {})", e, resp_line.trim()))
                })?;

                Ok(resp
                    .get("output")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string())
            }
        }
    }

    async fn llm_http(
        &self,
        client: &reqwest::Client,
        base_url: &str,
        request: &LlmRequest,
    ) -> Result<LlmResponse> {
        use futures::StreamExt;

        let url = format!("{}/api/stream", base_url);
        let body = serde_json::json!({
            "model": request.model,
            "provider": request.provider,
            "messages": request.messages,
            "tools": request.tools,
            "tool_choice": request.tool_choice,
            "commandstr": request.prompt,
            "npc": request.npc,
            "registered_teams": request.registered_teams,
            "conversationId": request.conversation_id,
            "currentPath": request.current_path,
            "executionMode": if request.tools.is_some() { "tool_agent" } else { "chat" },
        });

        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| NpcError::Other(format!("HTTP stream request failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(NpcError::Other(format!(
                "HTTP stream returned {}: {}",
                status, text
            )));
        }

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut thinking = String::new();
        let mut tool_calls: Vec<crate::r#gen::ToolCall> = Vec::new();
        let mut usage: Option<crate::r#gen::Usage> = None;
        let mut done = false;
        let mut saw_output = false;

        let mut stream = resp.bytes_stream();
        let mut pending = String::new();

        while !done {
            let chunk = match stream.next().await {
                Some(Ok(bytes)) => bytes,
                Some(Err(e)) => return Err(NpcError::Other(format!("HTTP stream chunk: {}", e))),
                None => break,
            };
            pending.push_str(&String::from_utf8_lossy(&chunk));

            // Extract complete SSE events (separated by a blank line).
            while let Some(sep_pos) = pending.find("\n\n").or_else(|| pending.find("\r\n\r\n")) {
                let event_text = pending[..sep_pos].to_string();
                let newline_len = if pending[sep_pos..].starts_with("\r\n\r\n") {
                    4
                } else {
                    2
                };
                pending.replace_range(..sep_pos + newline_len, "");

                let data = parse_sse_event_data(&event_text);
                let data = match data {
                    Some(d) => d,
                    None => continue,
                };
                if data.trim() == "[DONE]" {
                    done = true;
                    break;
                }

                let json: serde_json::Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                Self::apply_sse_event(
                    json,
                    &mut content,
                    &mut reasoning,
                    &mut thinking,
                    &mut tool_calls,
                    &mut usage,
                    &mut done,
                    &mut saw_output,
                );
            }
        }

        // Handle any trailing bytes that never got terminated by a blank line.
        if !done && !pending.trim().is_empty() {
            if let Some(data) = parse_sse_event_data(&pending) {
                if data.trim() != "[DONE]" {
                    if let Ok(json) = serde_json::from_str(&data) {
                        Self::apply_sse_event(
                            json,
                            &mut content,
                            &mut reasoning,
                            &mut thinking,
                            &mut tool_calls,
                            &mut usage,
                            &mut done,
                            &mut saw_output,
                        );
                    }
                }
            }
        }

        if saw_output {
            eprintln!();
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }

        // Remove empty trailing tool call slots.
        tool_calls.retain(|tc| !tc.function.name.is_empty());

        Ok(LlmResponse {
            ok: true,
            response: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            usage,
            thinking: if thinking.is_empty() {
                None
            } else {
                Some(thinking)
            },
            reasoning: if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            },
            streamed: Some(saw_output),
            error: None,
        })
    }

    fn apply_sse_event(
        json: serde_json::Value,
        content: &mut String,
        reasoning: &mut String,
        thinking: &mut String,
        tool_calls: &mut Vec<crate::r#gen::ToolCall>,
        usage: &mut Option<crate::r#gen::Usage>,
        done: &mut bool,
        saw_output: &mut bool,
    ) {
        if let Some(typ) = json.get("type").and_then(|v| v.as_str()) {
            match typ {
                "usage" => {
                    *usage = Some(crate::r#gen::Usage {
                        prompt_tokens: json
                            .get("input_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                        completion_tokens: json
                            .get("output_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                        total_tokens: json
                            .get("total_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0),
                    });
                }
                "message_stop" | "stop" => {
                    *done = true;
                }
                "error" => {
                    if let Some(msg) = json.get("message").and_then(|v| v.as_str()) {
                        eprintln!("\x1b[31mstream error: {}\x1b[0m", msg);
                    }
                }
                "tool_call" | "tool_execution_start" => {
                    if let Some(tc) = json.get("tool_call").or_else(|| json.get("tool_calls")) {
                        Self::append_tool_call_json(tc, tool_calls, saw_output);
                    }
                }
                _ => {}
            }
            return;
        }

        if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
            for choice in choices {
                if let Some(delta) = choice.get("delta") {
                    if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                        content.push_str(text);
                        *saw_output = true;
                        eprint!("{}", text);
                        let _ = std::io::Write::flush(&mut std::io::stderr());
                    }
                    if let Some(t) = delta.get("thinking").and_then(|v| v.as_str()) {
                        thinking.push_str(t);
                        *saw_output = true;
                        eprint!("\x1b[90m{}\x1b[0m", t);
                        let _ = std::io::Write::flush(&mut std::io::stderr());
                    }
                    if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                        reasoning.push_str(r);
                        *saw_output = true;
                    }
                    if let Some(deltas) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for (i, d) in deltas.iter().enumerate() {
                            let idx = d
                                .get("index")
                                .and_then(|v| v.as_u64())
                                .map(|n| n as usize)
                                .unwrap_or(i);
                            while tool_calls.len() <= idx {
                                tool_calls.push(crate::r#gen::ToolCall {
                                    id: String::new(),
                                    r#type: "function".to_string(),
                                    function: crate::r#gen::ToolCallFunction {
                                        name: String::new(),
                                        arguments: String::new(),
                                    },
                                });
                            }
                            *saw_output = true;
                            if let Some(id) = d.get("id").and_then(|v| v.as_str()) {
                                if !id.is_empty() {
                                    tool_calls[idx].id = id.to_string();
                                }
                            }
                            if let Some(tc_type) = d.get("type").and_then(|v| v.as_str()) {
                                if !tc_type.is_empty() {
                                    tool_calls[idx].r#type = tc_type.to_string();
                                }
                            }
                            if let Some(func) = d.get("function") {
                                if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                    if !name.is_empty() {
                                        tool_calls[idx].function.name = name.to_string();
                                    }
                                }
                                if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                                    tool_calls[idx].function.arguments.push_str(args);
                                }
                            }
                        }
                    }
                }
                if let Some(finish) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    if finish == "stop" || finish == "length" {
                        *done = true;
                    }
                }
            }
        }
    }

    fn append_tool_call_json(
        tc: &serde_json::Value,
        tool_calls: &mut Vec<crate::r#gen::ToolCall>,
        saw_output: &mut bool,
    ) {
        let id = tc
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let name = tc
            .get("name")
            .or_else(|| tc.get("function_name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let args = tc
            .get("arguments")
            .or_else(|| tc.get("function").and_then(|f| f.get("arguments")))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if !name.is_empty() {
            tool_calls.push(crate::r#gen::ToolCall {
                id,
                r#type: "function".to_string(),
                function: crate::r#gen::ToolCallFunction {
                    name,
                    arguments: args,
                },
            });
            *saw_output = true;
        }
    }
}

impl Kernel {
    pub fn boot(team_dir: &str, db_path: &str) -> Result<Self> {
        boot::boot_kernel(team_dir, db_path)
    }

    pub fn spawn(&mut self, npc: NPC, ppid: Pid, capabilities: Capabilities) -> Pid {
        let pid = self.next_pid.fetch_add(1, Ordering::Relaxed);
        let process = Process::spawn(pid, ppid, npc, capabilities);

        tracing::info!(
            "kernel: spawned pid:{} ({}) ppid:{}",
            pid,
            process.npc.name,
            ppid
        );

        self.processes.insert(pid, process);
        self.scheduler.enqueue(pid);
        pid
    }

    pub fn spawn_init(&mut self, npc: NPC) -> Pid {
        let pid = 0;
        self.next_pid.store(1, Ordering::Relaxed);
        let mut process = Process::spawn(pid, 0, npc, Capabilities::root());
        process.state = ProcessState::Running;
        self.processes.insert(pid, process);
        pid
    }

    pub fn get_process(&self, pid: Pid) -> Option<&Process> {
        self.processes.get(&pid)
    }

    pub fn get_process_mut(&mut self, pid: Pid) -> Option<&mut Process> {
        self.processes.get_mut(&pid)
    }

    pub fn find_by_name(&self, name: &str) -> Option<&Process> {
        self.processes.values().find(|p| p.npc.name == name)
    }

    pub fn kill(&mut self, pid: Pid, exit_code: i32) -> Result<()> {
        let process = self
            .processes
            .get_mut(&pid)
            .ok_or_else(|| NpcError::Other(format!("No process with pid {}", pid)))?;
        process.kill(exit_code);
        tracing::info!("kernel: killed pid:{} exit_code:{}", pid, exit_code);
        Ok(())
    }

    pub fn ps(&self) -> Vec<&Process> {
        self.processes
            .values()
            .filter(|p| p.state != ProcessState::Dead)
            .collect()
    }

    pub fn jinx_names(&self) -> Vec<&str> {
        self.jinxes.keys().map(|s| s.as_str()).collect()
    }

    pub async fn exec_chat(&mut self, pid: Pid, input: &str) -> Result<String> {
        let process = self
            .processes
            .get_mut(&pid)
            .ok_or_else(|| NpcError::Other(format!("No process with pid {}", pid)))?;

        process.state = ProcessState::Running;
        process.new_turn();

        let system = process.npc.system_prompt(self.team.context.as_deref());

        let mut messages = vec![Message::system(system)];
        for m in &process.messages {
            if m.role != "tool" && m.tool_calls.is_none() {
                messages.push(m.clone());
            }
        }
        messages.push(Message::user(input));

        let response = crate::r#gen::get_genai_response(
            &process.npc.resolved_provider(),
            &process.npc.resolved_model(),
            &messages,
            None,
            process.npc.api_url.as_deref(),
            process.npc.api_key.as_deref(),
            None,
            None,
            false,
            None,
        )
        .await?;

        if let Some(ref usage) = response.usage {
            process.record_usage(usage.prompt_tokens, usage.completion_tokens, 0.0);
        }

        let output = response.message.content.clone().unwrap_or_default();
        process.messages.push(Message::user(input));
        process.messages.push(response.message);
        process.state = ProcessState::Blocked;

        Ok(output)
    }

    pub async fn syscall(
        &mut self,
        pid: Pid,
        jinx_name: &str,
        args: &HashMap<String, String>,
    ) -> Result<String> {
        syscall::execute_syscall(self, pid, jinx_name, args).await
    }

    pub async fn exec(&mut self, pid: Pid, input: &str) -> Result<String> {
        use crate::r#gen::cost::calculate_cost;
        use crate::r#gen::sanitize::sanitize_messages;

        let (
            model,
            provider,
            system,
            api_url,
            api_key,
            npc_name,
            active_npc,
            tool_defs,
            executors,
            think_mode,
            conv_id,
        ) = {
            let process = self
                .processes
                .get_mut(&pid)
                .ok_or_else(|| NpcError::Other(format!("No process with pid {}", pid)))?;

            if let Some(reason) = process.usage.exceeds(&process.limits) {
                process.kill(137);
                return Err(NpcError::Other(format!(
                    "Process {} killed: {}",
                    pid, reason
                )));
            }

            process.state = ProcessState::Running;
            process.new_turn();

            let (td, ex) = process.npc.resolve_tools(&self.jinxes);
            let model = process.npc.resolved_model();
            let provider = process.npc.resolved_provider();
            let system = process.npc.system_prompt(self.team.context.as_deref());
            let api_url = process.npc.api_url.clone();
            let api_key = process.npc.api_key.clone();
            let npc_name = process.npc.name.clone();
            let active_npc = process.npc.clone();
            let think_mode = process.think;
            let conv_id = process.conversation_id.clone();

            if !process.capabilities.is_superuser && !process.capabilities.allowed_jinxes.is_empty()
            {
                let mut td = td;
                td.retain(|t| {
                    process
                        .capabilities
                        .allowed_jinxes
                        .contains(&t.function.name)
                });
                (
                    model, provider, system, api_url, api_key, npc_name, active_npc, td, ex,
                    think_mode, conv_id,
                )
            } else {
                (
                    model, provider, system, api_url, api_key, npc_name, active_npc, td, ex,
                    think_mode, conv_id,
                )
            }
        };

        let tools = if tool_defs.is_empty() {
            None
        } else {
            Some(tool_defs.clone())
        };

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());
        let path_cmd = format!("The current working directory is: {}", cwd);
        let ls_files = if let Ok(entries) = std::fs::read_dir(&cwd) {
            let files: Vec<String> = entries
                .flatten()
                .take(100)
                .map(|e| e.path().to_string_lossy().to_string())
                .collect();
            let total = std::fs::read_dir(&cwd).map(|d| d.count()).unwrap_or(0);
            let mut listing = format!(
                "Files in the current directory (full paths):\n{}",
                files.join("\n")
            );
            if total > 100 {
                listing.push_str(&format!("\n... and {} more files", total - 100));
            }
            listing
        } else {
            "No files found in the current directory.".to_string()
        };
        let platform_info = format!(
            "Platform: {} {} ({})",
            std::env::consts::OS,
            "",
            std::env::consts::ARCH
        );
        let context_info = format!("{}\n{}\n{}", path_cmd, ls_files, platform_info);

        let tool_guidance = if tools.is_some() {
            let tool_names: Vec<&str> =
                tool_defs.iter().map(|t| t.function.name.as_str()).collect();
            format!(
                "\nYou have access to these tools: {}. Call tools via the function calling interface.\n\n\
Use tools when you need to take action (run commands, search, edit files, etc.). Use chat to respond to the user.\n\
IMPORTANT: After at most 3-5 tool calls, you MUST call stop to finish. Do not keep reading files or running commands indefinitely — gather what you need, respond, and stop.\n\
Do not call stop without first calling chat to deliver a response to the user.\n\
The user can see tool outputs directly. Do not re-write or repeat them in your chat response — just reference the relevant parts.",
                tool_names.join(", ")
            )
        } else {
            String::new()
        };

        let max_iterations = 12;
        let mut total_input_tokens: u64 = 0;
        let mut total_output_tokens: u64 = 0;
        let mut final_output = String::new();
        let mut tool_calls_count = 0;
        let mut stop_requested = false;

        for iteration in 0..max_iterations {
            if stop_requested {
                break;
            }

            {
                let process = self.processes.get_mut(&pid).unwrap();
                process.messages = sanitize_messages(std::mem::take(&mut process.messages));
            }

            let mut messages = vec![Message::system(&system)];
            {
                let process = self.processes.get(&pid).unwrap();
                messages.extend(process.messages.clone());
            }

            let iter_prompt = if iteration == 0 {
                format!("{}\n{}{}", input, context_info, tool_guidance)
            } else {
                "Continue. Call stop when done.".to_string()
            };
            messages.push(Message::user(&iter_prompt));

            eprintln!(
                "\x1b[90m  [iter {}] {} msgs\x1b[0m",
                iteration + 1,
                messages.len(),
            );

            let registered_teams = self
                .team
                .source_dir
                .as_ref()
                .map(|d| vec![d.clone()])
                .or_else(|| {
                    std::env::current_dir()
                        .ok()
                        .map(|d| d.to_string_lossy().to_string())
                        .map(|d| vec![d])
                });

            let has_daemon = self.python_daemon.is_some();
            let response: crate::r#gen::LlmResponse = if has_daemon {
                let daemon = self.python_daemon.as_mut().unwrap();
                let req = LlmRequest {
                    req_type: "llm".to_string(),
                    messages,
                    model: model.clone(),
                    provider: provider.clone(),
                    prompt: iter_prompt.clone(),
                    context: Some(context_info.clone()),
                    tools: tools.clone(),
                    tool_choice: Some("auto".to_string()),
                    api_url: api_url.clone(),
                    api_key: api_key.clone(),
                    think: think_mode,
                    attachments: None,
                    registered_teams,
                    conversation_id: Some(conv_id.clone()),
                    current_path: Some(cwd.clone()),
                    npc: Some(npc_name.clone()),
                };
                match tokio::time::timeout(std::time::Duration::from_secs(120), daemon.llm(&req))
                    .await
                {
                    Ok(Ok(llm_resp)) => {
                        if !llm_resp.ok {
                            let err = llm_resp
                                .error
                                .clone()
                                .unwrap_or_else(|| "unknown daemon error".to_string());
                            return Err(NpcError::Other(format!("Daemon LLM error: {}", err)));
                        }
                        let tc = llm_resp.tool_calls.as_ref().map(|tcs| {
                            tcs.iter()
                                .map(|tc| crate::r#gen::ToolCall {
                                    id: tc.id.clone(),
                                    r#type: tc.r#type.clone(),
                                    function: crate::r#gen::ToolCallFunction {
                                        name: tc.function.name.clone(),
                                        arguments: tc.function.arguments.clone(),
                                    },
                                })
                                .collect::<Vec<_>>()
                        });
                        // Track streamed state from daemon
                        {
                            let process = self.processes.get_mut(&pid).unwrap();
                            process.last_streamed = llm_resp.streamed.unwrap_or(false);
                            process.last_thinking = llm_resp.thinking.clone();
                        }
                        crate::r#gen::LlmResponse {
                            message: Message {
                                role: "assistant".to_string(),
                                content: llm_resp.response.clone(),
                                tool_calls: tc,
                                tool_call_id: None,
                                name: None,
                                thinking: llm_resp.thinking.clone(),
                                reasoning_content: llm_resp.reasoning.clone(),
                            },
                            usage: llm_resp.usage.clone(),
                            model: model.clone(),
                            finish_reason: None,
                            cost_usd: None,
                        }
                    }
                    Ok(Err(e)) => {
                        return Err(NpcError::Other(format!("Daemon LLM call failed: {}", e)));
                    }
                    Err(_) => {
                        return Err(NpcError::Other(
                            "Daemon LLM call timed out after 120s".into(),
                        ));
                    }
                }
            } else {
                crate::r#gen::get_genai_response(
                    &provider,
                    &model,
                    &messages,
                    tools.as_deref(),
                    api_url.as_deref(),
                    api_key.as_deref(),
                    None,
                    None,
                    false,
                    think_mode,
                )
                .await?
            };

            if let Some(ref usage) = response.usage {
                total_input_tokens += usage.prompt_tokens;
                total_output_tokens += usage.completion_tokens;
                let cost = calculate_cost(&model, usage.prompt_tokens, usage.completion_tokens);
                let process = self.processes.get_mut(&pid).unwrap();
                process.record_usage(usage.prompt_tokens, usage.completion_tokens, cost);
            }

            if iteration == 0 {
                let process = self.processes.get_mut(&pid).unwrap();
                process.messages.push(Message::user(input));
                let _ = self.history.save_conversation_message(
                    &conv_id,
                    "user",
                    input,
                    &cwd,
                    Some(&model),
                    Some(&provider),
                    Some(&npc_name),
                    Some(&self.team.name),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                );
            }

            // Save assistant response to DB
            let tool_calls_json = if let Some(ref tc) = response.message.tool_calls {
                serde_json::to_string(tc).ok()
            } else {
                None
            };
            let _ = self.history.save_conversation_message(
                &conv_id,
                "assistant",
                response.message.content.as_deref().unwrap_or(""),
                &cwd,
                Some(&model),
                Some(&provider),
                Some(&npc_name),
                Some(&self.team.name),
                tool_calls_json.as_deref(),
                None,
                None,
                response.usage.as_ref().map(|u| u.prompt_tokens),
                response.usage.as_ref().map(|u| u.completion_tokens),
                response
                    .usage
                    .as_ref()
                    .map(|u| calculate_cost(&model, u.prompt_tokens, u.completion_tokens)),
            );

            // Print actual thinking / reasoning content when present.
            // If the response was already streamed, thinking was shown in
            // real-time via [THINK] chunks — don't print it again.
            {
                let process = self.processes.get(&pid).unwrap();
                if !process.last_streamed {
                    if let Some(ref t) = response.message.thinking {
                        if !t.is_empty() {
                            eprintln!("\x1b[90m  [iter {}] thinking:\x1b[0m {}", iteration + 1, t);
                        }
                    }
                    if let Some(ref r) = response.message.reasoning_content {
                        if !r.is_empty() {
                            eprintln!("\x1b[90m  [iter {}] reasoning:\x1b[0m {}", iteration + 1, r);
                        }
                    }
                }
            }

            if let Some(ref tool_calls) = response.message.tool_calls {
                tool_calls_count += 1;

                {
                    let process = self.processes.get_mut(&pid).unwrap();
                    process.messages.push(response.message.clone());
                }

                let called: Vec<String> = tool_calls
                    .iter()
                    .map(|tc| {
                        let schema_params: Vec<String> = tool_defs
                            .iter()
                            .find(|td| td.function.name == tc.function.name)
                            .and_then(|td| td.function.parameters.get("properties"))
                            .and_then(|p: &serde_json::Value| p.as_object())
                            .map(|obj: &serde_json::Map<String, serde_json::Value>| {
                                obj.keys().cloned().collect()
                            })
                            .unwrap_or_default();
                        let filtered = if let Ok(parsed) =
                            serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                        {
                            if let Some(obj) = parsed.as_object() {
                                let clean: serde_json::Map<String, serde_json::Value> =
                                    if schema_params.is_empty() {
                                        obj.clone()
                                    } else {
                                        obj.iter()
                                            .filter(|(k, _)| schema_params.contains(k))
                                            .map(|(k, v)| (k.clone(), v.clone()))
                                            .collect()
                                    };
                                serde_json::to_string(&clean).unwrap_or_default()
                            } else {
                                tc.function.arguments.clone()
                            }
                        } else {
                            tc.function.arguments.clone()
                        };
                        let preview = if filtered.len() > 200 {
                            format!("{}...", &filtered[..200])
                        } else {
                            filtered
                        };
                        format!("{}({})", tc.function.name, preview)
                    })
                    .collect();
                eprintln!(
                    "\x1b[90m  [iter {}] tools: {}\x1b[0m",
                    iteration + 1,
                    called.join(", ")
                );

                let tc_info: Vec<(String, String, String)> = tool_calls
                    .iter()
                    .map(|tc| {
                        (
                            tc.id.clone(),
                            tc.function.name.clone(),
                            tc.function.arguments.clone(),
                        )
                    })
                    .collect();

                let can_run: Vec<bool> = {
                    let process = self.processes.get(&pid).unwrap();
                    tc_info
                        .iter()
                        .map(|(_, name, _)| process.capabilities.can_run_jinx(name))
                        .collect()
                };

                for (i, (tc_id, tc_name, tc_args_str)) in tc_info.iter().enumerate() {
                    if !can_run[i] {
                        let process = self.processes.get_mut(&pid).unwrap();
                        process.messages.push(Message::tool_result(
                            tc_id,
                            &format!("EPERM: lacks capability for '{}'", tc_name),
                        ));
                        continue;
                    }

                    {
                        let process = self.processes.get_mut(&pid).unwrap();
                        process.usage.tool_calls_this_turn += 1;
                    }

                    let args: HashMap<String, String> =
                        serde_json::from_str(tc_args_str).unwrap_or_default();

                    let tool_result = self
                        .execute_tool(tc_name, &args, &executors, &active_npc)
                        .await;

                    if tc_name == "chat" {
                        final_output = args
                            .get("message")
                            .or_else(|| args.get("query"))
                            .cloned()
                            .unwrap_or_default();
                    } else {
                        eprintln!("\x1b[36m\n⚡ {} [{}|{}]:\x1b[0m", tc_name, model, provider);
                        let preview = if tool_result.len() > 500 {
                            format!(
                                "{}...\n[{} chars total]",
                                &tool_result[..500],
                                tool_result.len()
                            )
                        } else {
                            tool_result.clone()
                        };
                        eprintln!("{}", preview);
                    }

                    if tc_name == "stop" {
                        stop_requested = true;
                    }

                    let process = self.processes.get_mut(&pid).unwrap();
                    process
                        .messages
                        .push(Message::tool_result(tc_id, &tool_result));

                    // Save tool result to DB
                    let _ = self.history.save_conversation_message(
                        &conv_id,
                        "tool",
                        &tool_result,
                        &cwd,
                        Some(&model),
                        Some(&provider),
                        Some(&npc_name),
                        Some(&self.team.name),
                        None,
                        None,
                        Some(tc_id),
                        None,
                        None,
                        None,
                    );
                }
            } else {
                final_output = response.message.content.clone().unwrap_or_default();
                let process = self.processes.get_mut(&pid).unwrap();
                process.messages.push(response.message);
                break;
            }
        }

        eprintln!(
            "\x1b[90m  [{} iterations, {} tool call rounds]\x1b[0m",
            std::cmp::min(max_iterations, tool_calls_count + 1),
            tool_calls_count,
        );

        let process = self.processes.get_mut(&pid).unwrap();
        process.state = ProcessState::Blocked;
        Ok(final_output)
    }

    async fn execute_tool(
        &self,
        name: &str,
        args: &HashMap<String, String>,
        executors: &HashMap<String, crate::npc_compiler::ToolExecutor>,
        active_npc: &crate::npc_compiler::NPC,
    ) -> String {
        match name {
            "sh" | "shell" => {
                let cmd = args.get("bash_command").cloned().unwrap_or_default();
                if cmd.is_empty() {
                    return "(no command provided)".to_string();
                }
                match tokio::process::Command::new("bash")
                    .arg("-c")
                    .arg(&cmd)
                    .output()
                    .await
                {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        if !out.status.success() && !stderr.is_empty() {
                            format!(
                                "Error (exit {}):\n{}",
                                out.status.code().unwrap_or(-1),
                                stderr
                            )
                        } else if stdout.trim().is_empty() {
                            "(no output)".to_string()
                        } else {
                            stdout.to_string()
                        }
                    }
                    Err(e) => format!("Failed: {}", e),
                }
            }
            "python" | "py" => {
                let code = args.get("code").cloned().unwrap_or_default();
                if code.is_empty() {
                    return "(no code provided)".to_string();
                }
                match tokio::process::Command::new("python3")
                    .arg("-c")
                    .arg(&code)
                    .output()
                    .await
                {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        if stdout.trim().is_empty() && !stderr.is_empty() {
                            format!("Python error:\n{}", stderr)
                        } else {
                            stdout.to_string()
                        }
                    }
                    Err(e) => format!("Failed: {}", e),
                }
            }
            "web_search" => {
                let query = args
                    .get("query")
                    .or_else(|| args.get("search_query"))
                    .cloned()
                    .unwrap_or_default();
                if query.is_empty() {
                    return "(no query)".to_string();
                }
                let provider = args
                    .get("provider")
                    .map(|s| s.as_str())
                    .unwrap_or("duckduckgo");
                match crate::data::web::search_web(&query, 5, provider, None).await {
                    Ok(results) if !results.is_empty() => {
                        let mut out = format!("Web search results for '{}':\n\n", query);
                        for (i, r) in results.iter().enumerate() {
                            out.push_str(&format!(
                                "{}. {}\n   {}\n   {}\n\n",
                                i + 1,
                                r.title,
                                r.url,
                                r.snippet
                            ));
                        }
                        out
                    }
                    Ok(_) => format!("No results for '{}'", query),
                    Err(e) => format!("Search failed: {}", e),
                }
            }
            "stop" => "STOP".to_string(),
            "chat" => args
                .get("message")
                .or_else(|| args.get("query"))
                .cloned()
                .unwrap_or_default(),
            "edit_file" | "edit" => {
                let path = shellexpand::tilde(
                    args.get("path")
                        .or_else(|| args.get("file_path"))
                        .map(|s| s.as_str())
                        .unwrap_or(""),
                )
                .to_string();
                let action = args.get("action").map(|s| s.as_str()).unwrap_or("create");
                let new_text = args
                    .get("new_text")
                    .or_else(|| args.get("content"))
                    .or_else(|| args.get("text"))
                    .cloned()
                    .unwrap_or_default();
                let old_text = args.get("old_text").cloned().unwrap_or_default();
                match action {
                    "create" | "write" => std::fs::write(&path, &new_text)
                        .map(|_| format!("Wrote {} ({} bytes)", path, new_text.len()))
                        .unwrap_or_else(|e| format!("Error: {}", e)),
                    "append" => {
                        use std::io::Write;
                        std::fs::OpenOptions::new()
                            .append(true)
                            .create(true)
                            .open(&path)
                            .and_then(|mut f| f.write_all(new_text.as_bytes()))
                            .map(|_| format!("Appended to {}", path))
                            .unwrap_or_else(|e| format!("Error: {}", e))
                    }
                    "replace" => std::fs::read_to_string(&path)
                        .and_then(|c| std::fs::write(&path, c.replace(&old_text, &new_text)))
                        .map(|_| format!("Replaced in {}", path))
                        .unwrap_or_else(|e| format!("Error: {}", e)),
                    _ => format!("Unknown action: {}", action),
                }
            }
            "load_file" => {
                let path = shellexpand::tilde(
                    args.get("path")
                        .or_else(|| args.get("file_path"))
                        .map(|s| s.as_str())
                        .unwrap_or(""),
                )
                .to_string();
                match std::fs::read_to_string(&path) {
                    Ok(c) => {
                        let l = c.lines().count();
                        if c.len() > 10000 {
                            format!(
                                "File: {} ({} lines)\n---\n{}...[truncated]",
                                path,
                                l,
                                &c[..10000]
                            )
                        } else {
                            format!("File: {} ({} lines)\n---\n{}", path, l, c)
                        }
                    }
                    Err(e) => format!("Error: {}", e),
                }
            }
            "file_search" => {
                let query = args
                    .get("query")
                    .or_else(|| args.get("pattern"))
                    .cloned()
                    .unwrap_or_default();
                let path = shellexpand::tilde(
                    args.get("path")
                        .or_else(|| args.get("directory"))
                        .map(|s| s.as_str())
                        .unwrap_or("."),
                )
                .to_string();
                let cmd = format!(
                    "grep -rn --include='*.{{py,rs,js,ts,md,txt,yaml,yml,toml,json,sh}}' -l '{}' '{}' 2>/dev/null | head -20",
                    query.replace('\'', ""),
                    path
                );
                match tokio::process::Command::new("bash")
                    .arg("-c")
                    .arg(&cmd)
                    .output()
                    .await
                {
                    Ok(out) => {
                        let s = String::from_utf8_lossy(&out.stdout);
                        if s.trim().is_empty() {
                            format!("No files matching '{}' in {}", query, path)
                        } else {
                            s.to_string()
                        }
                    }
                    Err(e) => format!("Error: {}", e),
                }
            }
            "delegate" | "convene" => {
                let target = args
                    .get("npc_name")
                    .or_else(|| args.get("target"))
                    .cloned()
                    .unwrap_or_default();
                let msg = args
                    .get("message")
                    .or_else(|| args.get("query"))
                    .cloned()
                    .unwrap_or_default();
                if let Some(target_npc) = self.team.get_npc(&target).cloned() {
                    match crate::llm_funcs::get_llm_response(
                        &msg,
                        Some(&target_npc),
                        None,
                        None,
                        None,
                        &[],
                        self.team.context.as_deref(),
                    )
                    .await
                    {
                        Ok(result) => format!(
                            "@{} responded: {}",
                            target,
                            result.response.unwrap_or_default()
                        ),
                        Err(e) => format!("Delegation to @{} failed: {}", target, e),
                    }
                } else {
                    format!(
                        "NPC '{}' not found in team. Available: {:?}",
                        target,
                        self.team.npc_names()
                    )
                }
            }
            _ => match executors.get(name) {
                Some(crate::npc_compiler::ToolExecutor::Jinx(jname)) => {
                    if let Some(j) = self.jinxes.get(jname) {
                        match npc_compiler::execute_jinx_with_npc(
                            j,
                            args,
                            &self.jinxes,
                            Some(active_npc),
                        )
                        .await
                        {
                            Ok(r) => r.output,
                            Err(e) => format!("Jinx error: {}", e),
                        }
                    } else {
                        format!("Jinx '{}' not found", jname)
                    }
                }
                _ => format!("Tool '{}' not implemented", name),
            },
        }
    }

    pub fn fork(&mut self, parent_pid: Pid) -> Result<Pid> {
        let parent = self
            .processes
            .get(&parent_pid)
            .ok_or_else(|| NpcError::Other(format!("No process with pid {}", parent_pid)))?;

        if !parent.capabilities.can_spawn {
            return Err(NpcError::Other(format!(
                "Process {} lacks CAP_SPAWN",
                parent_pid
            )));
        }

        let child_npc = parent.npc.clone();
        let child_caps = if parent.capabilities.is_superuser {
            Capabilities::root()
        } else {
            parent.capabilities.clone()
        };

        Ok(self.spawn(child_npc, parent_pid, child_caps))
    }

    pub async fn delegate(
        &mut self,
        from_pid: Pid,
        target_npc_name: &str,
        input: &str,
    ) -> Result<String> {
        {
            let from = self
                .processes
                .get(&from_pid)
                .ok_or_else(|| NpcError::Other(format!("No process with pid {}", from_pid)))?;
            if !from.capabilities.can_delegate {
                return Err(NpcError::Other(format!(
                    "Process {} lacks CAP_DELEGATE",
                    from_pid
                )));
            }
        }

        let target_pid = if let Some(p) = self.find_by_name(target_npc_name) {
            p.pid
        } else {
            let npc = self
                .team
                .get_npc(target_npc_name)
                .ok_or_else(|| NpcError::NpcNotFound {
                    name: target_npc_name.to_string(),
                })?
                .clone();
            self.spawn(npc, from_pid, Capabilities::root())
        };

        self.exec(target_pid, input).await
    }

    pub fn stats(&self) -> KernelStats {
        let processes = &self.processes;
        let running = processes
            .values()
            .filter(|p| p.state == ProcessState::Running)
            .count();
        let blocked = processes
            .values()
            .filter(|p| p.state == ProcessState::Blocked)
            .count();
        let total_tokens: u64 = processes
            .values()
            .map(|p| p.usage.total_input_tokens + p.usage.total_output_tokens)
            .sum();
        let total_cost: f64 = processes.values().map(|p| p.usage.total_cost_usd).sum();

        KernelStats {
            uptime_secs: (chrono::Utc::now() - self.boot_time).num_seconds() as u64,
            total_processes: processes.len(),
            running,
            blocked,
            dead: processes
                .values()
                .filter(|p| p.state == ProcessState::Dead)
                .count(),
            total_tokens,
            total_cost_usd: total_cost,
            jinx_count: self.jinxes.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelStats {
    pub uptime_secs: u64,
    pub total_processes: usize,
    pub running: usize,
    pub blocked: usize,
    pub dead: usize,
    pub total_tokens: u64,
    pub total_cost_usd: f64,
    pub jinx_count: usize,
}

impl std::fmt::Display for KernelStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "uptime: {}s | procs: {} (run:{} blk:{} dead:{}) | tokens: {} | cost: ${:.4} | jinxes: {}",
            self.uptime_secs,
            self.total_processes,
            self.running,
            self.blocked,
            self.dead,
            self.total_tokens,
            self.total_cost_usd,
            self.jinx_count,
        )
    }
}
