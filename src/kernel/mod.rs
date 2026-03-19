//! Kernel — boot, process table, and syscall dispatch.
//!
//! The kernel is the central coordinator. It:
//! 1. Boots from a team directory (like an OS booting from disk)
//! 2. Manages the process table (NPC processes)
//! 3. Dispatches syscalls (jinx invocations) with capability checks
//! 4. Manages drivers (LLM providers, MCP servers)
//! 5. Coordinates IPC between processes

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
use crate::npc_compiler::Npc;
use crate::process::{Capabilities, Pid, Process, ProcessState};
use crate::scheduler::Scheduler;
use crate::npc_compiler::Team;
use crate::vfs::Vfs;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// The NPC OS Kernel.
pub struct Kernel {
    /// Process table: pid → process.
    processes: HashMap<Pid, Process>,

    /// Next PID to assign.
    next_pid: AtomicU32,

    /// Loaded team (boot image).
    pub team: Team,

    /// All available jinxes (syscall table).
    pub jinxes: HashMap<String, Jinx>,

    /// LLM driver manager.
    pub drivers: DriverManager,

    /// Virtual filesystem.
    pub vfs: Vfs,

    /// IPC bus.
    pub ipc: IpcBus,

    /// Process scheduler.
    pub scheduler: Scheduler,

    /// Conversation history database.
    pub history: CommandHistory,

    /// Kernel uptime.
    pub boot_time: chrono::DateTime<chrono::Utc>,
}

impl Kernel {
    /// Boot the kernel from a team directory.
    pub fn boot(team_dir: &str, db_path: &str) -> Result<Self> {
        boot::boot_kernel(team_dir, db_path)
    }

    /// Spawn a new process from an NPC.
    pub fn spawn(&mut self, npc: Npc, ppid: Pid, capabilities: Capabilities) -> Pid {
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

    /// Spawn the init process (forenpc, pid 0).
    pub fn spawn_init(&mut self, npc: Npc) -> Pid {
        let pid = 0;
        self.next_pid.store(1, Ordering::Relaxed);
        let mut process = Process::spawn(pid, 0, npc, Capabilities::root());
        process.state = ProcessState::Running;
        self.processes.insert(pid, process);
        pid
    }

    /// Get a process by PID.
    pub fn get_process(&self, pid: Pid) -> Option<&Process> {
        self.processes.get(&pid)
    }

    /// Get a mutable process by PID.
    pub fn get_process_mut(&mut self, pid: Pid) -> Option<&mut Process> {
        self.processes.get_mut(&pid)
    }

    /// Find a process by NPC name.
    pub fn find_by_name(&self, name: &str) -> Option<&Process> {
        self.processes.values().find(|p| p.npc.name == name)
    }

    /// Kill a process.
    pub fn kill(&mut self, pid: Pid, exit_code: i32) -> Result<()> {
        let process = self.processes.get_mut(&pid).ok_or_else(|| {
            NpcError::Other(format!("No process with pid {}", pid))
        })?;
        process.kill(exit_code);
        tracing::info!("kernel: killed pid:{} exit_code:{}", pid, exit_code);
        Ok(())
    }

    /// List all living processes.
    pub fn ps(&self) -> Vec<&Process> {
        self.processes
            .values()
            .filter(|p| p.state != ProcessState::Dead)
            .collect()
    }

    /// List all jinx names.
    pub fn jinx_names(&self) -> Vec<&str> {
        self.jinxes.keys().map(|s| s.as_str()).collect()
    }

    /// Chat-only exec — sends to LLM without any tools.
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

        // Build messages without tool messages (like Python chat mode)
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

    /// Execute a syscall (jinx invocation) on behalf of a process.
    pub async fn syscall(
        &mut self,
        pid: Pid,
        jinx_name: &str,
        args: &HashMap<String, String>,
    ) -> Result<String> {
        syscall::execute_syscall(self, pid, jinx_name, args).await
    }

    /// Execute the agent loop — mirrors Python's process_pipeline_command.
    ///
    /// Flow (matching npcpy exactly):
    /// 1. Sanitize messages before each iteration
    /// 2. First iteration sends user input, subsequent send "Continue. Call stop when done."
    /// 3. Execute tool calls, print results, append to messages
    /// 4. Loop until no tool calls or max iterations (50)
    /// 5. Accumulate usage across iterations
    pub async fn exec(
        &mut self,
        pid: Pid,
        input: &str,
    ) -> Result<String> {
        use crate::r#gen::sanitize::sanitize_messages;
        use crate::r#gen::cost::calculate_cost;

        // Extract what we need from the process, then drop the borrow
        let (model, provider, system, api_url, npc_name, mut tool_defs, executors) = {
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

            if !process.capabilities.is_superuser && !process.capabilities.allowed_jinxes.is_empty() {
                let mut td = td;
                td.retain(|t| process.capabilities.allowed_jinxes.contains(&t.function.name));
                (model, provider, system, api_url, npc_name, td, ex)
            } else {
                (model, provider, system, api_url, npc_name, td, ex)
            }
        };

        let tools = if tool_defs.is_empty() { None } else { Some(tool_defs.as_slice()) };

        // Add context info like Python does
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());
        let platform_info = format!("Platform: {} ({})", std::env::consts::OS, std::env::consts::ARCH);
        let context_info = format!("The current working directory is: {}\n{}", cwd, platform_info);

        // Tool guidance (matches Python's tool prompt injection)
        let tool_guidance = if tools.is_some() {
            let tool_names: Vec<&str> = tool_defs.iter().map(|t| t.function.name.as_str()).collect();
            format!(
                "\nYou have access to these tools: {}. Call tools via the function calling interface.\n\
                 Use tools when you need to take action (run commands, search, edit files, etc.). \
                 Use chat to respond to the user. Use stop when you are done. \
                 Do not call the same tool twice with the same arguments.\n\
                 Do not call stop without first calling chat to deliver a response to the user.\n\
                 The user can see tool outputs directly. Do not re-write or repeat them in your chat response.",
                tool_names.join(", ")
            )
        } else {
            String::new()
        };

        // Agent loop — mirrors Python's while iteration < max_iterations
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

            // Sanitize messages before EACH iteration (matches Python)
            {
                let process = self.processes.get_mut(&pid).unwrap();
                process.messages = sanitize_messages(std::mem::take(&mut process.messages));
            }

            // Build message list fresh each iteration
            let mut messages = vec![Message::system(&system)];
            {
                let process = self.processes.get(&pid).unwrap();
                messages.extend(process.messages.clone());
            }

            // First iteration: user input + context. Subsequent: "Continue."
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

            // LLM call
            let response = crate::r#gen::get_genai_response(
                    &provider, &model, &messages, tools, api_url.as_deref(),
                )
                .await?;

            // Accumulate usage
            if let Some(ref usage) = response.usage {
                total_input_tokens += usage.prompt_tokens;
                total_output_tokens += usage.completion_tokens;
                let cost = calculate_cost(&model, usage.prompt_tokens, usage.completion_tokens);
                let process = self.processes.get_mut(&pid).unwrap();
                process.record_usage(usage.prompt_tokens, usage.completion_tokens, cost);
            }

            // Add user message to process history (only on first iteration)
            if iteration == 0 {
                let process = self.processes.get_mut(&pid).unwrap();
                process.messages.push(Message::user(input));
            }

            // Check for tool calls
            if let Some(ref tool_calls) = response.message.tool_calls {
                tool_calls_count += 1;

                // Add assistant message with tool_calls to history
                {
                    let process = self.processes.get_mut(&pid).unwrap();
                    process.messages.push(response.message.clone());
                }

                // Log tool names
                let called: Vec<&str> = tool_calls.iter().map(|tc| tc.function.name.as_str()).collect();
                eprintln!("\x1b[90m  [iter {}] tools: {}\x1b[0m", iteration + 1, called.join(", "));

                // Collect tool call info to avoid borrow conflict
                let tc_info: Vec<(String, String, String)> = tool_calls.iter()
                    .map(|tc| (tc.id.clone(), tc.function.name.clone(), tc.function.arguments.clone()))
                    .collect();

                // Check capabilities
                let can_run: Vec<bool> = {
                    let process = self.processes.get(&pid).unwrap();
                    tc_info.iter()
                        .map(|(_, name, _)| process.capabilities.can_run_jinx(name))
                        .collect()
                };

                // Execute each tool call
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

                    let tool_result = self.execute_tool(tc_name, &args, &executors).await;

                    // Print tool result immediately (like Python does)
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
                // No tool calls — final response
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

    /// Execute a single tool by name — matches Python's tool_exec_map dispatch.
    async fn execute_tool(
        &self,
        name: &str,
        args: &HashMap<String, String>,
        executors: &HashMap<String, crate::npc_compiler::ToolExecutor>,
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
                // Delegate via llm_funcs::get_llm_response on the target NPC
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
            // Fallback: jinx engine
            _ => {
                match executors.get(name) {
                    Some(crate::npc_compiler::ToolExecutor::Jinx(jname)) => {
                        if let Some(j) = self.jinxes.get(jname) {
                            match npc_compiler::execute_jinx(j, args, &self.jinxes).await {
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

    /// Fork a process — create a child with the same NPC but fresh state.
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
            // Children get same or fewer capabilities
            parent.capabilities.clone()
        };

        Ok(self.spawn(child_npc, parent_pid, child_caps))
    }

    /// Delegate: parent process sends work to a named NPC process.
    /// If no process exists for that NPC, spawns one.
    pub async fn delegate(
        &mut self,
        from_pid: Pid,
        target_npc_name: &str,
        input: &str,
    ) -> Result<String> {
        // Check delegation capability
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

        // Find or spawn target process
        let target_pid = if let Some(p) = self.find_by_name(target_npc_name) {
            p.pid
        } else {
            // Spawn from team NPCs
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

    /// Get kernel stats.
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

/// Kernel status summary.
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
