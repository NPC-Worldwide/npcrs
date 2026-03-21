
mod boot;
mod syscall;

pub use boot::*;
pub use syscall::*;

use crate::drivers::DriverManager;
use crate::error::{NpcError, Result};
use crate::ipc::IpcBus;
use crate::npc_compiler::{self, Jinx};
use crate::r#gen::Message;
use crate::memory::CommandHistory;
use crate::npc_compiler::NPC;
use crate::process::{Capabilities, Pid, Process, ProcessState};
use crate::scheduler::Scheduler;
use crate::npc_compiler::Team;
use crate::vfs::Vfs;
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

pub struct PythonDaemon {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    stdout: tokio::io::BufReader<tokio::process::ChildStdout>,
}

impl PythonDaemon {
    pub async fn spawn(team_dir: &str, db_path: &str) -> Result<Self> {
        use tokio::process::Command;
        use tokio::io::BufReader;

        let mut child = Command::new("python3")
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
            .map_err(|e| NpcError::Other(format!("Failed to spawn Python daemon: {}", e)))?;

        let stdin = child.stdin.take().ok_or_else(|| NpcError::Other("No stdin on daemon".into()))?;
        let stdout = child.stdout.take().ok_or_else(|| NpcError::Other("No stdout on daemon".into()))?;
        let mut stderr = child.stderr.take().ok_or_else(|| NpcError::Other("No stderr on daemon".into()))?;

        use tokio::io::AsyncBufReadExt;
        let mut stderr_reader = BufReader::new(stderr);
        let mut found_ready = false;
        for _ in 0..50 {
            let mut line = String::new();
            match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                stderr_reader.read_line(&mut line)
            ).await {
                Ok(Ok(0)) => break,
                Ok(Ok(_)) => {
                    if line.contains("ready") {
                        found_ready = true;
                        break;
                    }
                }
                _ => break,
            }
        }
        if !found_ready {
            return Err(NpcError::Other("Daemon failed to start: never sent ready signal".into()));
        }

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    pub async fn execute(&mut self, command: &str, stdin_input: Option<&str>) -> Result<String> {
        use tokio::io::{AsyncWriteExt, AsyncBufReadExt};

        let req = serde_json::json!({
            "command": command,
            "stdin_input": stdin_input,
        });
        let mut line = serde_json::to_string(&req).unwrap_or_default();
        line.push('\n');

        self.stdin.write_all(line.as_bytes()).await
            .map_err(|e| NpcError::Other(format!("Daemon write: {}", e)))?;
        self.stdin.flush().await
            .map_err(|e| NpcError::Other(format!("Daemon flush: {}", e)))?;

        let mut resp_line = String::new();
        self.stdout.read_line(&mut resp_line).await
            .map_err(|e| NpcError::Other(format!("Daemon read: {}", e)))?;

        let resp: serde_json::Value = serde_json::from_str(&resp_line)
            .map_err(|e| NpcError::Other(format!("Daemon parse: {} (raw: {})", e, resp_line.trim())))?;

        Ok(resp.get("output").and_then(|v| v.as_str()).unwrap_or("").to_string())
    }
}

impl Kernel {
    pub fn boot(team_dir: &str, db_path: &str) -> Result<Self> {
        boot::boot_kernel(team_dir, db_path)
    }

    pub fn spawn(&mut self, npc: NPC, ppid: Pid, capabilities: Capabilities) -> Pid {
        let pid = self.next_pid.fetch_add(1, Ordering::Relaxed);
        let mut process = Process::spawn(pid, ppid, npc, capabilities);

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
        let process = self.processes.get_mut(&pid).ok_or_else(|| {
            NpcError::Other(format!("No process with pid {}", pid))
        })?;
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

    pub async fn exec_chat(
        &mut self,
        pid: Pid,
        input: &str,
    ) -> Result<String> {
        let process = self.processes.get_mut(&pid).ok_or_else(|| {
            NpcError::Other(format!("No process with pid {}", pid))
        })?;

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

    pub async fn exec(
        &mut self,
        pid: Pid,
        input: &str,
    ) -> Result<String> {
        use crate::r#gen::sanitize::sanitize_messages;
        use crate::r#gen::cost::calculate_cost;

        let (model, provider, system, api_url, npc_name, active_npc, mut tool_defs, executors) = {
            let process = self.processes.get_mut(&pid).ok_or_else(|| {
                NpcError::Other(format!("No process with pid {}", pid))
            })?;

            if let Some(reason) = process.usage.exceeds(&process.limits) {
                process.kill(137);
                return Err(NpcError::Other(format!("Process {} killed: {}", pid, reason)));
            }

            process.state = ProcessState::Running;
            process.new_turn();

            let (td, ex) = process.npc.resolve_tools(&self.jinxes);
            let model = process.npc.resolved_model();
            let provider = process.npc.resolved_provider();
            let system = process.npc.system_prompt(self.team.context.as_deref());
            let api_url = process.npc.api_url.clone();
            let npc_name = process.npc.name.clone();
            let active_npc = process.npc.clone();

            if !process.capabilities.is_superuser && !process.capabilities.allowed_jinxes.is_empty() {
                let mut td = td;
                td.retain(|t| process.capabilities.allowed_jinxes.contains(&t.function.name));
                (model, provider, system, api_url, npc_name, active_npc, td, ex)
            } else {
                (model, provider, system, api_url, npc_name, active_npc, td, ex)
            }
        };

        let tools = if tool_defs.is_empty() { None } else { Some(tool_defs.as_slice()) };

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());
        let path_cmd = format!("The current working directory is: {}", cwd);
        let ls_files = if let Ok(entries) = std::fs::read_dir(&cwd) {
            let files: Vec<String> = entries.flatten().take(100)
                .map(|e| e.path().to_string_lossy().to_string())
                .collect();
            let total = std::fs::read_dir(&cwd).map(|d| d.count()).unwrap_or(0);
            let mut listing = format!("Files in the current directory (full paths):\n{}", files.join("\n"));
            if total > 100 {
                listing.push_str(&format!("\n... and {} more files", total - 100));
            }
            listing
        } else {
            "No files found in the current directory.".to_string()
        };
        let platform_info = format!("Platform: {} {} ({})", std::env::consts::OS, "", std::env::consts::ARCH);
        let context_info = format!("{}\n{}\n{}", path_cmd, ls_files, platform_info);

        let tool_guidance = if tools.is_some() {
            let tool_names: Vec<&str> = tool_defs.iter().map(|t| t.function.name.as_str()).collect();
            format!(
                "\nYou have access to these tools: {}. Call tools via the function calling interface.\n\n\
Use tools when you need to take action (run commands, search, edit files, etc.). Use chat to respond to the user. Use stop when you are done. Do not call the same tool twice with the same arguments.\n\
Do not call stop without first calling chat to deliver a response to the user.\n\
The user can see tool outputs directly. Do not re-write or repeat them in your chat response — just reference the relevant parts.",
                tool_names.join(", ")
            )
        } else {
            String::new()
        };

        let max_iterations = 50;
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

            let response = crate::r#gen::get_genai_response(
                    &provider, &model, &messages, tools, api_url.as_deref(),
                )
                .await?;

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
            }

            if let Some(ref tool_calls) = response.message.tool_calls {
                tool_calls_count += 1;

                {
                    let process = self.processes.get_mut(&pid).unwrap();
                    process.messages.push(response.message.clone());
                }

                if let Some(ref text) = response.message.content {
                    if !text.is_empty() {
                        eprintln!("\x1b[90m  [iter {}] thinking:\x1b[0m {}", iteration + 1, text);
                    }
                }
                let called: Vec<String> = tool_calls.iter().map(|tc| {
                    let schema_params: Vec<String> = tool_defs.iter()
                        .find(|td| td.function.name == tc.function.name)
                        .and_then(|td| td.function.parameters.get("properties"))
                        .and_then(|p| p.as_object())
                        .map(|obj| obj.keys().cloned().collect())
                        .unwrap_or_default();
                    let filtered = if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&tc.function.arguments) {
                        if let Some(obj) = parsed.as_object() {
                            let clean: serde_json::Map<String, serde_json::Value> = if schema_params.is_empty() {
                                obj.clone()
                            } else {
                                obj.iter().filter(|(k, _)| schema_params.contains(k)).map(|(k, v)| (k.clone(), v.clone())).collect()
                            };
                            serde_json::to_string(&clean).unwrap_or_default()
                        } else {
                            tc.function.arguments.clone()
                        }
                    } else {
                        tc.function.arguments.clone()
                    };
                    let preview = if filtered.len() > 200 { format!("{}...", &filtered[..200]) } else { filtered };
                    format!("{}({})", tc.function.name, preview)
                }).collect();
                eprintln!("\x1b[90m  [iter {}] tools: {}\x1b[0m", iteration + 1, called.join(", "));

                let tc_info: Vec<(String, String, String)> = tool_calls.iter()
                    .map(|tc| (tc.id.clone(), tc.function.name.clone(), tc.function.arguments.clone()))
                    .collect();

                let can_run: Vec<bool> = {
                    let process = self.processes.get(&pid).unwrap();
                    tc_info.iter()
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

                    let tool_result = self.execute_tool(tc_name, &args, &executors, &active_npc).await;

                    eprintln!("\x1b[36m\n⚡ {}:\x1b[0m", tc_name);
                    let preview = if tool_result.len() > 500 {
                        format!("{}...\n[{} chars total]", &tool_result[..500], tool_result.len())
                    } else {
                        tool_result.clone()
                    };
                    eprintln!("{}", preview);

                    if tc_name == "stop" {
                        stop_requested = true;
                    }

                    if tc_name == "chat" {
                        final_output = args.get("message")
                            .or_else(|| args.get("query"))
                            .cloned()
                            .unwrap_or_default();
                    }

                    let process = self.processes.get_mut(&pid).unwrap();
                    process.messages.push(Message::tool_result(tc_id, &tool_result));
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
            "sh" => {
                let cmd = args.get("bash_command").cloned().unwrap_or_default();
                if cmd.is_empty() { return "(no command provided)".to_string(); }
                match tokio::process::Command::new("bash").arg("-c").arg(&cmd).output().await {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        if !out.status.success() && !stderr.is_empty() {
                            format!("Error (exit {}):\n{}", out.status.code().unwrap_or(-1), stderr)
                        } else if stdout.trim().is_empty() { "(no output)".to_string() }
                        else { stdout.to_string() }
                    }
                    Err(e) => format!("Failed: {}", e),
                }
            }
            "python" => {
                let code = args.get("code").cloned().unwrap_or_default();
                if code.is_empty() { return "(no code provided)".to_string(); }
                match tokio::process::Command::new("python3").arg("-c").arg(&code).output().await {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        if stdout.trim().is_empty() && !stderr.is_empty() { format!("Python error:\n{}", stderr) }
                        else { stdout.to_string() }
                    }
                    Err(e) => format!("Failed: {}", e),
                }
            }
            "web_search" => {
                let query = args.get("query").or_else(|| args.get("search_query")).cloned().unwrap_or_default();
                if query.is_empty() { return "(no query)".to_string(); }
                let provider = args.get("provider").map(|s| s.as_str()).unwrap_or("duckduckgo");
                match crate::data::web::search_web(&query, 5, provider, None).await {
                    Ok(results) if !results.is_empty() => {
                        let mut out = format!("Web search results for '{}':\n\n", query);
                        for (i, r) in results.iter().enumerate() {
                            out.push_str(&format!("{}. {}\n   {}\n   {}\n\n", i + 1, r.title, r.url, r.snippet));
                        }
                        out
                    }
                    Ok(_) => format!("No results for '{}'", query),
                    Err(e) => format!("Search failed: {}", e),
                }
            }
            "stop" => "STOP".to_string(),
            "chat" => args.get("message").or_else(|| args.get("query")).cloned().unwrap_or_default(),
            "edit_file" | "edit" => {
                let path = shellexpand::tilde(args.get("path").or_else(|| args.get("file_path")).map(|s| s.as_str()).unwrap_or("")).to_string();
                let action = args.get("action").map(|s| s.as_str()).unwrap_or("create");
                let new_text = args.get("new_text").or_else(|| args.get("content")).or_else(|| args.get("text")).cloned().unwrap_or_default();
                let old_text = args.get("old_text").cloned().unwrap_or_default();
                match action {
                    "create" | "write" => std::fs::write(&path, &new_text).map(|_| format!("Wrote {} ({} bytes)", path, new_text.len())).unwrap_or_else(|e| format!("Error: {}", e)),
                    "append" => { use std::io::Write; std::fs::OpenOptions::new().append(true).create(true).open(&path).and_then(|mut f| f.write_all(new_text.as_bytes())).map(|_| format!("Appended to {}", path)).unwrap_or_else(|e| format!("Error: {}", e)) }
                    "replace" => std::fs::read_to_string(&path).and_then(|c| std::fs::write(&path, c.replace(&old_text, &new_text))).map(|_| format!("Replaced in {}", path)).unwrap_or_else(|e| format!("Error: {}", e)),
                    _ => format!("Unknown action: {}", action),
                }
            }
            "load_file" => {
                let path = shellexpand::tilde(args.get("path").or_else(|| args.get("file_path")).map(|s| s.as_str()).unwrap_or("")).to_string();
                match std::fs::read_to_string(&path) {
                    Ok(c) => { let l = c.lines().count(); if c.len() > 10000 { format!("File: {} ({} lines)\n---\n{}...[truncated]", path, l, &c[..10000]) } else { format!("File: {} ({} lines)\n---\n{}", path, l, c) } }
                    Err(e) => format!("Error: {}", e),
                }
            }
            "file_search" => {
                let query = args.get("query").or_else(|| args.get("pattern")).cloned().unwrap_or_default();
                let path = shellexpand::tilde(args.get("path").or_else(|| args.get("directory")).map(|s| s.as_str()).unwrap_or(".")).to_string();
                let cmd = format!("grep -rn --include='*.{{py,rs,js,ts,md,txt,yaml,yml,toml,json,sh}}' -l '{}' '{}' 2>/dev/null | head -20", query.replace('\'', ""), path);
                match tokio::process::Command::new("bash").arg("-c").arg(&cmd).output().await {
                    Ok(out) => { let s = String::from_utf8_lossy(&out.stdout); if s.trim().is_empty() { format!("No files matching '{}' in {}", query, path) } else { s.to_string() } }
                    Err(e) => format!("Error: {}", e),
                }
            }
            "delegate" | "convene" => {
                let target = args.get("npc_name").or_else(|| args.get("target")).cloned().unwrap_or_default();
                let msg = args.get("message").or_else(|| args.get("query")).cloned().unwrap_or_default();
                if let Some(target_npc) = self.team.get_npc(&target).cloned() {
                    match crate::llm_funcs::get_llm_response(
                        &msg, Some(&target_npc), None, None, None, &[], self.team.context.as_deref(),
                    ).await {
                        Ok(result) => format!("@{} responded: {}", target, result.response.unwrap_or_default()),
                        Err(e) => format!("Delegation to @{} failed: {}", target, e),
                    }
                } else {
                    format!("NPC '{}' not found in team. Available: {:?}", target, self.team.npc_names())
                }
            }
            _ => {
                match executors.get(name) {
                    Some(crate::npc_compiler::ToolExecutor::Jinx(jname)) => {
                        if let Some(j) = self.jinxes.get(jname) {
                            match npc_compiler::execute_jinx_with_npc(j, args, &self.jinxes, Some(active_npc)).await {
                                Ok(r) => r.output,
                                Err(e) => format!("Jinx error: {}", e),
                            }
                        } else { format!("Jinx '{}' not found", jname) }
                    }
                    _ => format!("Tool '{}' not implemented", name),
                }
            }
        }
    }

    pub fn fork(&mut self, parent_pid: Pid) -> Result<Pid> {
        let parent = self.processes.get(&parent_pid).ok_or_else(|| {
            NpcError::Other(format!("No process with pid {}", parent_pid))
        })?;

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
            let from = self.processes.get(&from_pid).ok_or_else(|| {
                NpcError::Other(format!("No process with pid {}", from_pid))
            })?;
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
        let total_cost: f64 = processes
            .values()
            .map(|p| p.usage.total_cost_usd)
            .sum();

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

use serde::Serialize;
