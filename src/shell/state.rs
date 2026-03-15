use crate::jinx::{self};
use crate::llm::{LlmClient, LlmResponse, Message};
use crate::memory::CommandHistory;
use crate::npc::{Npc, ToolExecutor};
use crate::team::Team;
use crate::error::{NpcError, Result};
use std::collections::HashMap;

/// The runtime state of the shell.
pub struct ShellState {
    /// Current active NPC.
    pub npc: Npc,

    /// Loaded team.
    pub team: Team,

    /// LLM client for API calls.
    pub llm_client: LlmClient,

    /// Conversation history database.
    pub history: CommandHistory,

    /// Current conversation messages.
    pub messages: Vec<Message>,

    /// Current conversation ID in the database.
    pub conversation_id: String,

    /// Current shell mode (agent, chat, cmd, etc.).
    pub current_mode: ShellMode,

    /// Current working directory.
    pub current_path: String,

    /// Whether to stream LLM output.
    pub stream_output: bool,

    /// Session token counters.
    pub session_input_tokens: u64,
    pub session_output_tokens: u64,
    pub session_cost_usd: f64,
    pub turn_count: u64,

    /// Config overrides.
    pub config: ShellConfig,
}

/// Shell operating mode.
#[derive(Debug, Clone, PartialEq)]
pub enum ShellMode {
    /// Full agent mode — can run tools, bash, and LLM.
    Agent,
    /// Chat mode — pure LLM conversation, no tools.
    Chat,
    /// Command mode — shell commands with LLM assistance.
    Cmd,
    /// Custom mode backed by a jinx name.
    Custom(String),
}

impl std::fmt::Display for ShellMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellMode::Agent => write!(f, "agent"),
            ShellMode::Chat => write!(f, "chat"),
            ShellMode::Cmd => write!(f, "cmd"),
            ShellMode::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Shell configuration (from env vars / .npcshrc).
#[derive(Debug, Clone)]
pub struct ShellConfig {
    pub chat_model: String,
    pub chat_provider: String,
    pub vision_model: Option<String>,
    pub vision_provider: Option<String>,
    pub embedding_model: Option<String>,
    pub embedding_provider: Option<String>,
    pub reasoning_model: Option<String>,
    pub reasoning_provider: Option<String>,
    pub search_provider: String,
    pub build_kg: bool,
    pub edit_approval: EditApproval,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EditApproval {
    Off,
    Interactive,
    Auto,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            chat_model: std::env::var("NPCSH_CHAT_MODEL")
                .unwrap_or_else(|_| "gpt-4o-mini".to_string()),
            chat_provider: std::env::var("NPCSH_CHAT_PROVIDER")
                .unwrap_or_else(|_| "openai".to_string()),
            vision_model: std::env::var("NPCSH_VISION_MODEL").ok(),
            vision_provider: std::env::var("NPCSH_VISION_PROVIDER").ok(),
            embedding_model: std::env::var("NPCSH_EMBEDDING_MODEL").ok(),
            embedding_provider: std::env::var("NPCSH_EMBEDDING_PROVIDER").ok(),
            reasoning_model: std::env::var("NPCSH_REASONING_MODEL").ok(),
            reasoning_provider: std::env::var("NPCSH_REASONING_PROVIDER").ok(),
            search_provider: std::env::var("NPCSH_SEARCH_PROVIDER")
                .unwrap_or_else(|_| "duckduckgo".to_string()),
            build_kg: std::env::var("NPCSH_BUILD_KG")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            edit_approval: match std::env::var("NPCSH_EDIT_APPROVAL")
                .unwrap_or_default()
                .as_str()
            {
                "interactive" => EditApproval::Interactive,
                "auto" => EditApproval::Auto,
                _ => EditApproval::Off,
            },
        }
    }
}

impl ShellState {
    /// Process a user command through the full dispatch pipeline.
    ///
    /// This is the main entry point — equivalent to npcsh's `execute_command`.
    ///
    /// 1. Check for mode switch commands (/agent, /chat, /cmd)
    /// 2. Check for @npc delegation
    /// 3. In agent mode: check bash → check slash → LLM with tools
    /// 4. In other modes: route through mode jinx
    pub async fn process_command(&mut self, input: &str) -> Result<CommandResult> {
        let input = input.trim();
        if input.is_empty() {
            return Ok(CommandResult::empty());
        }

        self.turn_count += 1;

        // Mode switch
        match input {
            "/agent" => {
                self.current_mode = ShellMode::Agent;
                return Ok(CommandResult::info("Switched to agent mode"));
            }
            "/chat" => {
                self.current_mode = ShellMode::Chat;
                return Ok(CommandResult::info("Switched to chat mode"));
            }
            "/cmd" => {
                self.current_mode = ShellMode::Cmd;
                return Ok(CommandResult::info("Switched to cmd mode"));
            }
            _ => {}
        }

        // @npc delegation
        if input.starts_with('@') {
            return self.handle_delegation(input).await;
        }

        match &self.current_mode {
            ShellMode::Agent => self.process_agent_command(input).await,
            ShellMode::Chat => self.process_chat_command(input).await,
            ShellMode::Cmd => self.process_cmd_command(input).await,
            ShellMode::Custom(mode_name) => {
                let mode_name = mode_name.clone();
                self.process_custom_mode(input, &mode_name).await
            }
        }
    }

    /// Agent mode: full pipeline with tools and bash.
    async fn process_agent_command(&mut self, input: &str) -> Result<CommandResult> {
        // Slash command?
        if input.starts_with('/') {
            return self.process_slash_command(input).await;
        }

        // Bash command check (simple heuristic: known command prefixes)
        if is_likely_bash(input) {
            return self.execute_bash(input).await;
        }

        // LLM with tools
        self.query_llm_with_tools(input).await
    }

    /// Chat mode: pure LLM, no tools.
    async fn process_chat_command(&mut self, input: &str) -> Result<CommandResult> {
        let system = self
            .npc
            .system_prompt(self.team.context.as_deref());

        let mut messages = vec![Message::system(system)];
        // Add conversation history (skip tool messages for chat mode)
        for m in &self.messages {
            if m.role != "tool" && m.tool_calls.is_none() {
                messages.push(m.clone());
            }
        }
        messages.push(Message::user(input));

        let response = self
            .npc
            .get_response(&self.llm_client, &messages, None)
            .await?;

        self.track_usage(&response);

        let output = response
            .message
            .content
            .clone()
            .unwrap_or_default();

        self.messages.push(Message::user(input));
        self.messages.push(response.message);

        Ok(CommandResult {
            output,
            ..Default::default()
        })
    }

    /// Cmd mode: execute as shell command with LLM fallback.
    async fn process_cmd_command(&mut self, input: &str) -> Result<CommandResult> {
        self.execute_bash(input).await
    }

    /// Custom mode: route through the corresponding mode jinx.
    async fn process_custom_mode(
        &mut self,
        input: &str,
        mode_name: &str,
    ) -> Result<CommandResult> {
        if let Some(jinx) = self.team.jinxes.get(mode_name) {
            let mut inputs = HashMap::new();
            inputs.insert("query".to_string(), input.to_string());

            let result =
                jinx::execute_jinx(jinx, &inputs, &self.team.jinxes).await?;

            Ok(CommandResult {
                output: result.output,
                ..Default::default()
            })
        } else {
            // Fallback to agent mode for this command
            self.process_agent_command(input).await
        }
    }

    /// Handle @npc delegation.
    async fn handle_delegation(&mut self, input: &str) -> Result<CommandResult> {
        let parts: Vec<&str> = input.splitn(2, ' ').collect();
        let npc_name = parts[0].trim_start_matches('@');
        let command = parts.get(1).unwrap_or(&"").to_string();

        let npc = self
            .team
            .get_npc(npc_name)
            .ok_or_else(|| NpcError::NpcNotFound {
                name: npc_name.to_string(),
            })?
            .clone();

        // Temporarily switch NPC, run command, switch back
        let original_npc = std::mem::replace(&mut self.npc, npc);
        let result = self.process_agent_command(&command).await;
        self.npc = original_npc;

        result
    }

    /// Process a /slash command.
    async fn process_slash_command(&mut self, input: &str) -> Result<CommandResult> {
        let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
        let cmd_name = parts[0];
        let args = parts.get(1).unwrap_or(&"").to_string();

        // Built-in commands
        match cmd_name {
            "help" => {
                return Ok(CommandResult::info(self.help_text()));
            }
            "set" => {
                return self.handle_set_command(&args);
            }
            "quit" | "exit" => {
                return Ok(CommandResult {
                    should_exit: true,
                    ..Default::default()
                });
            }
            "clear" => {
                self.messages.clear();
                return Ok(CommandResult::info("Conversation cleared"));
            }
            "jinxes" => {
                let names: Vec<&str> = self.team.jinx_names();
                return Ok(CommandResult::info(format!(
                    "Available jinxes:\n{}",
                    names.join("\n")
                )));
            }
            _ => {}
        }

        // Try to find and execute a jinx with this name
        if let Some(jinx) = self.team.jinxes.get(cmd_name).cloned() {
            let mut inputs = HashMap::new();
            if !args.is_empty() {
                // Simple arg parsing: first arg goes to first input
                if let Some(first_input) = jinx.inputs.first() {
                    inputs.insert(first_input.name.clone(), args);
                }
            }

            let result =
                jinx::execute_jinx(&jinx, &inputs, &self.team.jinxes).await?;

            // Record execution
            let conv_id = &self.conversation_id;
            let _ = self.history.save_jinx_execution(
                conv_id,
                cmd_name,
                &serde_json::to_string(&inputs).unwrap_or_default(),
                &result.output,
                if result.success { "success" } else { "error" },
                None, None,
                result.error.as_deref(),
                None,
            );

            return Ok(CommandResult {
                output: result.output,
                ..Default::default()
            });
        }

        // Check if it's a mode switch
        if self.team.jinxes.contains_key(cmd_name) {
            self.current_mode = ShellMode::Custom(cmd_name.to_string());
            return Ok(CommandResult::info(format!("Switched to {} mode", cmd_name)));
        }

        Ok(CommandResult::info(format!(
            "Unknown command: /{}",
            cmd_name
        )))
    }

    /// Query the LLM with tool calling enabled.
    async fn query_llm_with_tools(&mut self, input: &str) -> Result<CommandResult> {
        let system = self
            .npc
            .system_prompt(self.team.context.as_deref());

        // Resolve available tools
        let (tool_defs, executors) = self.npc.resolve_tools(&self.team.jinxes);

        let mut messages = vec![Message::system(system)];
        messages.extend(self.messages.clone());
        messages.push(Message::user(input));

        let tools = if tool_defs.is_empty() {
            None
        } else {
            Some(tool_defs.as_slice())
        };

        // Tool call loop (max 10 rounds to prevent infinite loops)
        let mut final_output = String::new();
        for _ in 0..10 {
            let response = self
                .npc
                .get_response(&self.llm_client, &messages, tools)
                .await?;

            self.track_usage(&response);

            if let Some(ref tool_calls) = response.message.tool_calls {
                // Execute tool calls
                messages.push(response.message.clone());

                for tc in tool_calls {
                    let tool_result =
                        self.execute_tool_call(&tc.function.name, &tc.function.arguments, &executors)
                            .await?;

                    messages.push(Message::tool_result(&tc.id, &tool_result));
                }
                // Continue the loop for the LLM to process tool results
            } else {
                // No tool calls — we have the final response
                final_output = response
                    .message
                    .content
                    .clone()
                    .unwrap_or_default();
                messages.push(response.message);
                break;
            }
        }

        // Update conversation history
        self.messages.push(Message::user(input));
        self.messages
            .push(Message::assistant(&final_output));

        Ok(CommandResult {
            output: final_output,
            ..Default::default()
        })
    }

    /// Execute a tool call by name.
    async fn execute_tool_call(
        &self,
        name: &str,
        arguments: &str,
        executors: &HashMap<String, ToolExecutor>,
    ) -> Result<String> {
        let executor = executors.get(name).ok_or_else(|| NpcError::ToolNotFound {
            name: name.to_string(),
        })?;

        match executor {
            ToolExecutor::Jinx(jinx_name) => {
                if let Some(jinx) = self.team.jinxes.get(jinx_name) {
                    let inputs: HashMap<String, String> =
                        serde_json::from_str(arguments).unwrap_or_default();
                    let result =
                        jinx::execute_jinx(jinx, &inputs, &self.team.jinxes).await?;
                    Ok(result.output)
                } else {
                    Err(NpcError::JinxNotFound {
                        name: jinx_name.clone(),
                    })
                }
            }
            ToolExecutor::Native(_func_name) => {
                // Future: registered native Rust functions
                Ok("Native function execution not yet implemented".to_string())
            }
            ToolExecutor::Mcp(_spec) => {
                // Future: MCP client tool call
                Ok("MCP tool execution not yet implemented".to_string())
            }
            ToolExecutor::Python(code) => {
                // Delegate to Python
                let output = tokio::process::Command::new("python3")
                    .arg("-c")
                    .arg(code)
                    .output()
                    .await
                    .map_err(|e| NpcError::JinxExecution {
                        step: "python".to_string(),
                        reason: e.to_string(),
                    })?;
                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            }
        }
    }

    /// Execute a bash command.
    async fn execute_bash(&self, command: &str) -> Result<CommandResult> {
        let output = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&self.current_path)
            .output()
            .await
            .map_err(|e| NpcError::Shell(e.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        Ok(CommandResult {
            output: if stderr.is_empty() {
                stdout.to_string()
            } else {
                format!("{}\n{}", stdout, stderr)
            },
            exit_code: output.status.code(),
            ..Default::default()
        })
    }

    /// Track token usage from an LLM response.
    fn track_usage(&mut self, response: &LlmResponse) {
        if let Some(ref usage) = response.usage {
            self.session_input_tokens += usage.prompt_tokens;
            self.session_output_tokens += usage.completion_tokens;
        }
    }

    /// Handle the /set command.
    fn handle_set_command(&mut self, args: &str) -> Result<CommandResult> {
        let parts: Vec<&str> = args.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Ok(CommandResult::info(
                "Usage: /set key=value (e.g., /set model=gpt-4)",
            ));
        }

        let key = parts[0].trim();
        let value = parts[1].trim();

        match key {
            "model" => {
                self.npc.model = Some(value.to_string());
                self.config.chat_model = value.to_string();
            }
            "provider" => {
                self.npc.provider = Some(value.to_string());
                self.config.chat_provider = value.to_string();
            }
            "stream" => {
                self.stream_output = value == "1" || value == "true";
            }
            "mode" => match value {
                "agent" => self.current_mode = ShellMode::Agent,
                "chat" => self.current_mode = ShellMode::Chat,
                "cmd" => self.current_mode = ShellMode::Cmd,
                other => self.current_mode = ShellMode::Custom(other.to_string()),
            },
            _ => {
                return Ok(CommandResult::info(format!("Unknown setting: {}", key)));
            }
        }

        Ok(CommandResult::info(format!("Set {} = {}", key, value)))
    }

    fn help_text(&self) -> String {
        format!(
            "npcsh-rs v{}\n\n\
             Commands:\n\
             /agent        Switch to agent mode (tools + bash + LLM)\n\
             /chat         Switch to chat mode (LLM only)\n\
             /cmd          Switch to cmd mode (bash first)\n\
             /set k=v      Set config (model, provider, stream, mode)\n\
             /jinxes       List available jinxes\n\
             /clear        Clear conversation\n\
             /quit         Exit shell\n\
             @npc <cmd>    Delegate to another NPC\n\
             /<jinx> args  Execute a jinx directly\n\n\
             Current: mode={}, npc={}, model={}",
            env!("CARGO_PKG_VERSION"),
            self.current_mode,
            self.npc.name,
            self.npc.resolved_model(),
        )
    }
}

/// Result of processing a command.
#[derive(Debug, Default)]
pub struct CommandResult {
    pub output: String,
    pub exit_code: Option<i32>,
    pub should_exit: bool,
}

impl CommandResult {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn info(msg: impl Into<String>) -> Self {
        Self {
            output: msg.into(),
            ..Default::default()
        }
    }
}

/// Simple heuristic to detect likely bash commands.
fn is_likely_bash(input: &str) -> bool {
    let first_word = input.split_whitespace().next().unwrap_or("");
    let bash_commands = [
        "ls", "cd", "pwd", "cat", "grep", "find", "mkdir", "rm", "cp", "mv",
        "echo", "touch", "chmod", "chown", "head", "tail", "wc", "sort", "uniq",
        "tar", "zip", "unzip", "curl", "wget", "git", "docker", "make", "cargo",
        "npm", "pip", "python", "python3", "node", "rustc", "gcc", "g++",
    ];
    bash_commands.contains(&first_word)
}
