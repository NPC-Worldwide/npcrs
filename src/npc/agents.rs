//! Agent subclasses mirroring npcpy's Agent/ToolAgent/CodingAgent hierarchy.
//!
//! ```text
//! NPC (base) — raw agent with name, directive, model/provider
//!   └─ Agent — NPC + default tool set (sh, python, edit_file, load_file, web_search, file_search, stop, chat)
//!        └─ ToolAgent — Agent + user-provided tool functions and/or MCP servers
//!        └─ CodingAgent — Agent + language setting, auto-detects + executes code blocks
//! ```

use crate::error::Result;
use crate::llm::{LlmClient, Message, ToolDef};
use crate::npc::Npc;
use crate::tools::{RegisteredTool, ToolBuilder, ToolRegistry};

/// Agent — NPC + default tool set.
pub struct Agent {
    pub npc: Npc,
    pub messages: Vec<Message>,
    pub tool_registry: ToolRegistry,
}

impl Agent {
    pub fn new(npc: Npc) -> Self {
        let mut registry = ToolRegistry::new();
        register_default_tools(&mut registry);
        Self {
            npc,
            messages: Vec::new(),
            tool_registry: registry,
        }
    }

    pub fn with_name_and_directive(name: &str, directive: &str) -> Self {
        Self::new(Npc::new(name, directive))
    }

    pub async fn run(&mut self, client: &LlmClient, input: &str) -> Result<String> {
        let system = self.npc.system_prompt(None);
        let mut msgs = vec![Message::system(system)];
        msgs.extend(self.messages.clone());
        msgs.push(Message::user(input));

        let tool_defs = self.tool_registry.tool_defs();
        let tools = if tool_defs.is_empty() {
            None
        } else {
            Some(tool_defs.as_slice())
        };

        let model = self.npc.resolved_model();
        let provider = self.npc.resolved_provider();

        let mut final_output = String::new();
        for _ in 0..10 {
            let response = client
                .chat_completion(
                    &provider,
                    &model,
                    &msgs,
                    tools,
                    self.npc.api_url.as_deref(),
                )
                .await?;

            if let Some(ref tool_calls) = response.message.tool_calls {
                msgs.push(response.message.clone());
                let results = self.tool_registry.process_tool_calls(tool_calls).await;
                msgs.extend(results);
            } else {
                final_output = response.message.content.clone().unwrap_or_default();
                break;
            }
        }

        self.messages.push(Message::user(input));
        self.messages.push(Message::assistant(&final_output));
        Ok(final_output)
    }
}

/// ToolAgent — Agent + user-provided tool functions and/or MCP servers.
pub struct ToolAgent {
    pub agent: Agent,
}

impl ToolAgent {
    pub fn new(npc: Npc, extra_tools: Vec<RegisteredTool>) -> Self {
        let mut agent = Agent::new(npc);
        for tool in extra_tools {
            agent.tool_registry.register(tool);
        }
        Self { agent }
    }

    pub async fn run(&mut self, client: &LlmClient, input: &str) -> Result<String> {
        self.agent.run(client, input).await
    }
}

/// CodingAgent — Agent + language setting, auto-executes code blocks in responses.
pub struct CodingAgent {
    pub agent: Agent,
    pub language: String,
    pub auto_execute: bool,
}

impl CodingAgent {
    pub fn new(npc: Npc, language: impl Into<String>) -> Self {
        Self {
            agent: Agent::new(npc),
            language: language.into(),
            auto_execute: true,
        }
    }

    /// Extract fenced code blocks matching this agent's language.
    pub fn extract_code_blocks(&self, text: &str) -> Vec<String> {
        let pattern = format!(r"```(?i:{})\s*\n([\s\S]*?)```", regex::escape(&self.language));
        let re = regex::Regex::new(&pattern).unwrap_or_else(|_| {
            regex::Regex::new(r"```\w*\s*\n([\s\S]*?)```").unwrap()
        });
        re.captures_iter(text)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().trim().to_string()))
            .collect()
    }

    /// Execute a code block.
    pub async fn execute_code(&self, code: &str) -> String {
        let (cmd, args): (&str, Vec<&str>) = match self.language.as_str() {
            "python" => ("python3", vec!["-c", code]),
            "bash" | "sh" => ("bash", vec!["-c", code]),
            "javascript" | "js" => ("node", vec!["-e", code]),
            _ => return format!("Execution not supported for: {}", self.language),
        };

        match tokio::process::Command::new(cmd)
            .args(&args)
            .output()
            .await
        {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                if out.status.success() {
                    stdout.to_string()
                } else {
                    format!("{}\nSTDERR: {}", stdout, stderr)
                }
            }
            Err(e) => format!("Execution error: {}", e),
        }
    }

    pub async fn run(&mut self, client: &LlmClient, input: &str) -> Result<String> {
        let mut current_input = input.to_string();
        let mut last_response = String::new();

        for _ in 0..5 {
            last_response = self.agent.run(client, &current_input).await?;

            if !self.auto_execute {
                return Ok(last_response);
            }

            let blocks = self.extract_code_blocks(&last_response);
            if blocks.is_empty() {
                return Ok(last_response);
            }

            let mut results = Vec::new();
            for (i, code) in blocks.iter().enumerate() {
                let output = self.execute_code(code).await;
                results.push(format!("[Block {} output]:\n{}", i + 1, output));
            }

            current_input = format!("Code execution results:\n{}", results.join("\n\n"));
        }

        Ok(last_response)
    }
}

/// Register the default tools (sh, python, edit_file, load_file, web_search, file_search, stop, chat).
fn register_default_tools(registry: &mut ToolRegistry) {
    // sh
    registry.register(
        ToolBuilder::new("sh")
            .description("Execute a bash/shell command and return output")
            .param("bash_command", "string", "The command to execute", true)
            .build(Box::new(|args| {
                Box::pin(async move {
                    let cmd = args
                        .get("bash_command")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if cmd.is_empty() {
                        return Ok("(no command provided)".to_string());
                    }
                    match tokio::process::Command::new("bash")
                        .arg("-c")
                        .arg(cmd)
                        .output()
                        .await
                    {
                        Ok(out) => {
                            let stdout = String::from_utf8_lossy(&out.stdout);
                            let stderr = String::from_utf8_lossy(&out.stderr);
                            if !out.status.success() && !stderr.is_empty() {
                                Ok(format!(
                                    "Error (exit {}):\n{}",
                                    out.status.code().unwrap_or(-1),
                                    stderr
                                ))
                            } else if stdout.trim().is_empty() {
                                Ok("(no output)".to_string())
                            } else {
                                Ok(stdout.to_string())
                            }
                        }
                        Err(e) => Ok(format!("Failed: {}", e)),
                    }
                })
            })),
    );

    // python
    registry.register(
        ToolBuilder::new("python")
            .description("Execute Python code and return output")
            .param("code", "string", "Python code to execute", true)
            .build(Box::new(|args| {
                Box::pin(async move {
                    let code = args.get("code").and_then(|v| v.as_str()).unwrap_or("");
                    if code.is_empty() {
                        return Ok("(no code provided)".to_string());
                    }
                    match tokio::process::Command::new("python3")
                        .arg("-c")
                        .arg(code)
                        .output()
                        .await
                    {
                        Ok(out) => {
                            let stdout = String::from_utf8_lossy(&out.stdout);
                            let stderr = String::from_utf8_lossy(&out.stderr);
                            Ok(if stdout.trim().is_empty() && !stderr.is_empty() {
                                format!("Python error:\n{}", stderr)
                            } else {
                                stdout.to_string()
                            })
                        }
                        Err(e) => Ok(format!("Failed: {}", e)),
                    }
                })
            })),
    );

    // edit_file
    registry.register(
        ToolBuilder::new("edit_file")
            .description("Edit a file: create, append, or replace text")
            .param("path", "string", "File path", true)
            .param("action", "string", "Action: create, write, append, replace", false)
            .param("new_text", "string", "Text to write/append/replace with", false)
            .param("old_text", "string", "Text to find (for replace)", false)
            .build(Box::new(|args| {
                Box::pin(async move {
                    let path = args
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let path = shellexpand::tilde(path).to_string();
                    let action = args
                        .get("action")
                        .and_then(|v| v.as_str())
                        .unwrap_or("create");
                    let new_text = args
                        .get("new_text")
                        .or(args.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let old_text = args
                        .get("old_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    match action {
                        "create" | "write" => match std::fs::write(&path, new_text) {
                            Ok(_) => Ok(format!("Created {} ({} bytes)", path, new_text.len())),
                            Err(e) => Ok(format!("Error: {}", e)),
                        },
                        "append" => {
                            use std::io::Write;
                            match std::fs::OpenOptions::new()
                                .append(true)
                                .create(true)
                                .open(&path)
                            {
                                Ok(mut f) => {
                                    let _ = f.write_all(new_text.as_bytes());
                                    Ok(format!("Appended to {}", path))
                                }
                                Err(e) => Ok(format!("Error: {}", e)),
                            }
                        }
                        "replace" => match std::fs::read_to_string(&path) {
                            Ok(content) => {
                                let updated = content.replace(old_text, new_text);
                                match std::fs::write(&path, &updated) {
                                    Ok(_) => Ok(format!("Replaced in {}", path)),
                                    Err(e) => Ok(format!("Error: {}", e)),
                                }
                            }
                            Err(e) => Ok(format!("Error: {}", e)),
                        },
                        _ => Ok(format!("Unknown action: {}", action)),
                    }
                })
            })),
    );

    // load_file
    registry.register(
        ToolBuilder::new("load_file")
            .description("Read and return file contents")
            .param("path", "string", "File path to read", true)
            .build(Box::new(|args| {
                Box::pin(async move {
                    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
                    let path = shellexpand::tilde(path).to_string();
                    match std::fs::read_to_string(&path) {
                        Ok(content) => {
                            let lines = content.lines().count();
                            if content.len() > 10000 {
                                Ok(format!(
                                    "File: {} ({} lines)\n---\n{}...[truncated]",
                                    path,
                                    lines,
                                    &content[..10000]
                                ))
                            } else {
                                Ok(format!("File: {} ({} lines)\n---\n{}", path, lines, content))
                            }
                        }
                        Err(e) => Ok(format!("Error: {}", e)),
                    }
                })
            })),
    );

    // web_search
    registry.register(
        ToolBuilder::new("web_search")
            .description("Search the web")
            .param("query", "string", "Search query", true)
            .build(Box::new(|args| {
                Box::pin(async move {
                    let query = args
                        .get("query")
                        .or(args.get("search_query"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let cmd = format!(
                        "curl -sL 'https://lite.duckduckgo.com/lite/?q={}' | head -100",
                        query.replace(' ', "+")
                    );
                    match tokio::process::Command::new("bash")
                        .arg("-c")
                        .arg(&cmd)
                        .output()
                        .await
                    {
                        Ok(out) => Ok(String::from_utf8_lossy(&out.stdout).to_string()),
                        Err(e) => Ok(format!("Search failed: {}", e)),
                    }
                })
            })),
    );

    // file_search
    registry.register(
        ToolBuilder::new("file_search")
            .description("Search for files containing a pattern")
            .param("query", "string", "Text to search for", true)
            .param("path", "string", "Directory to search in", false)
            .build(Box::new(|args| {
                Box::pin(async move {
                    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                    let cmd = format!(
                        "grep -rn --include='*.{{py,rs,js,ts,md,txt,yaml,yml,toml,json}}' -l '{}' '{}' | head -20",
                        query.replace('\'', ""), path
                    );
                    match tokio::process::Command::new("bash")
                        .arg("-c")
                        .arg(&cmd)
                        .output()
                        .await
                    {
                        Ok(out) => Ok(String::from_utf8_lossy(&out.stdout).to_string()),
                        Err(e) => Ok(format!("Error: {}", e)),
                    }
                })
            })),
    );

    // stop
    registry.register(
        ToolBuilder::new("stop")
            .description("Signal that the task is complete")
            .param("reason", "string", "Reason for stopping", false)
            .build(Box::new(|args| {
                Box::pin(async move {
                    let reason = args.get("reason").and_then(|v| v.as_str()).unwrap_or("");
                    Ok(if reason.is_empty() {
                        "STOP".to_string()
                    } else {
                        format!("STOP: {}", reason)
                    })
                })
            })),
    );

    // chat
    registry.register(
        ToolBuilder::new("chat")
            .description("Respond directly to the user")
            .param("message", "string", "Message to send", true)
            .build(Box::new(|args| {
                Box::pin(async move {
                    Ok(args
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string())
                })
            })),
    );
}
