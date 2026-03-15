//! High-level LLM functions — response matrix, command checking, model resolution.
//!
//! This module provides the primary interface for getting LLM responses with
//! full NPC context, command type detection, and cost tracking.

use crate::error::Result;
use crate::r#gen::{LlmResponse, Message, ToolDef, ToolCall};
use crate::npc_compiler::Npc;

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

    // Call LLM via gen::response (internal genai dispatch)
    let response = crate::r#gen::get_genai_response(
        &resolved_provider,
        &resolved_model,
        &clean,
        tools,
        npc.and_then(|n| n.api_url.as_deref()),
    )
    .await?;

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
