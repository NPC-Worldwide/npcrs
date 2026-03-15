use crate::error::{NpcError, Result};
use crate::r#gen::{FunctionDef, Message, ToolCall, ToolDef};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// A registered tool handler -- takes JSON args, returns result string.
pub type ToolHandler = Box<
    dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
>;

/// Registry of tools available for LLM calling.
pub struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>,
}

/// A tool with its definition and handler.
pub struct RegisteredTool {
    pub def: ToolDef,
    pub handler: ToolHandler,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.
    pub fn register(&mut self, tool: RegisteredTool) {
        let name = tool.def.function.name.clone();
        self.tools.insert(name, tool);
    }

    /// Get all tool definitions for sending to the LLM.
    pub fn tool_defs(&self) -> Vec<ToolDef> {
        self.tools.values().map(|t| t.def.clone()).collect()
    }

    /// Execute a tool call by name with the given JSON arguments.
    pub async fn execute(&self, name: &str, args: serde_json::Value) -> Result<String> {
        let tool = self.tools.get(name).ok_or_else(|| NpcError::ToolNotFound {
            name: name.to_string(),
        })?;
        (tool.handler)(args).await
    }

    /// Process a list of tool calls from an LLM response, returning tool result messages.
    pub async fn process_tool_calls(&self, tool_calls: &[ToolCall]) -> Vec<Message> {
        let mut results = Vec::with_capacity(tool_calls.len());

        for tc in tool_calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

            let result_text = match self.execute(&tc.function.name, args).await {
                Ok(output) => output,
                Err(e) => format!("Error executing tool '{}': {}", tc.function.name, e),
            };

            results.push(Message::tool_result(&tc.id, result_text));
        }

        results
    }

    /// Check if a tool is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating tools without runtime introspection.
pub struct ToolBuilder {
    name: String,
    description: String,
    parameters: serde_json::Value,
    required: Vec<String>,
}

impl ToolBuilder {
    /// Start building a new tool with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
            }),
            required: Vec::new(),
        }
    }

    /// Set the tool description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a parameter to the tool.
    pub fn param(
        mut self,
        name: &str,
        type_str: &str,
        description: &str,
        required: bool,
    ) -> Self {
        if let Some(props) = self.parameters.get_mut("properties") {
            props[name] = serde_json::json!({
                "type": type_str,
                "description": description,
            });
        }
        if required {
            self.required.push(name.to_string());
        }
        self
    }

    /// Finish building and attach a handler.
    pub fn build(mut self, handler: ToolHandler) -> RegisteredTool {
        // Set the required array on the parameters object.
        if !self.required.is_empty() {
            self.parameters["required"] = serde_json::json!(self.required);
        }

        RegisteredTool {
            def: ToolDef {
                r#type: "function".to_string(),
                function: FunctionDef {
                    name: self.name,
                    description: if self.description.is_empty() {
                        None
                    } else {
                        Some(self.description)
                    },
                    parameters: self.parameters,
                },
            },
            handler,
        }
    }
}

/// Flatten tool_calls and tool-result messages into plain text for models
/// that don't support tool calling. Mirrors npcpy's tools.py:flatten_tool_messages.
///
/// - Assistant messages with tool_calls are converted to text describing the calls.
/// - Tool result messages are converted to user messages with the result content.
/// - All other messages pass through unchanged.
pub fn flatten_tool_messages(messages: &[Message]) -> Vec<Message> {
    let mut out = Vec::with_capacity(messages.len());

    for msg in messages {
        if msg.role == "assistant" {
            if let Some(ref tool_calls) = msg.tool_calls {
                // Build a text representation of what the assistant wanted to do.
                let mut parts = Vec::new();
                if let Some(ref content) = msg.content {
                    if !content.is_empty() {
                        parts.push(content.clone());
                    }
                }
                for tc in tool_calls {
                    parts.push(format!(
                        "[Tool Call: {}({})]",
                        tc.function.name, tc.function.arguments
                    ));
                }
                out.push(Message::assistant(parts.join("\n")));
            } else {
                out.push(msg.clone());
            }
        } else if msg.role == "tool" {
            // Convert tool results into user messages so non-tool models can understand.
            let content = msg
                .content
                .as_deref()
                .unwrap_or("[no output]")
                .to_string();
            let label = if let Some(ref id) = msg.tool_call_id {
                format!("[Tool Result ({})]:\n{}", id, content)
            } else {
                format!("[Tool Result]:\n{}", content)
            };
            out.push(Message::user(label));
        } else {
            out.push(msg.clone());
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#gen::{ToolCall, ToolCallFunction};

    #[test]
    fn test_tool_builder() {
        let tool = ToolBuilder::new("test_tool")
            .description("A test tool")
            .param("query", "string", "The search query", true)
            .param("limit", "integer", "Max results", false)
            .build(Box::new(|_args| {
                Box::pin(async { Ok("result".to_string()) })
            }));

        assert_eq!(tool.def.function.name, "test_tool");
        assert_eq!(
            tool.def.function.description.as_deref(),
            Some("A test tool")
        );
        assert_eq!(tool.def.r#type, "function");

        let params = &tool.def.function.parameters;
        assert!(params["properties"]["query"].is_object());
        assert!(params["properties"]["limit"].is_object());
        assert_eq!(params["required"], serde_json::json!(["query"]));
    }

    #[tokio::test]
    async fn test_registry_execute() {
        let mut registry = ToolRegistry::new();

        let tool = ToolBuilder::new("echo")
            .description("Echo the input")
            .param("text", "string", "Text to echo", true)
            .build(Box::new(|args| {
                Box::pin(async move {
                    let text = args
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("nothing");
                    Ok(format!("echo: {}", text))
                })
            }));

        registry.register(tool);

        assert_eq!(registry.len(), 1);
        assert!(registry.has_tool("echo"));

        let result = registry
            .execute("echo", serde_json::json!({"text": "hello"}))
            .await
            .unwrap();
        assert_eq!(result, "echo: hello");

        // Unknown tool should error.
        let err = registry
            .execute("unknown", serde_json::Value::Null)
            .await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn test_process_tool_calls() {
        let mut registry = ToolRegistry::new();

        let tool = ToolBuilder::new("greet")
            .description("Greet someone")
            .param("name", "string", "Name to greet", true)
            .build(Box::new(|args| {
                Box::pin(async move {
                    let name = args
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("world");
                    Ok(format!("Hello, {}!", name))
                })
            }));

        registry.register(tool);

        let tool_calls = vec![ToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            function: ToolCallFunction {
                name: "greet".to_string(),
                arguments: r#"{"name":"Rust"}"#.to_string(),
            },
        }];

        let results = registry.process_tool_calls(&tool_calls).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].role, "tool");
        assert_eq!(results[0].content.as_deref(), Some("Hello, Rust!"));
        assert_eq!(results[0].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn test_flatten_tool_messages() {
        let messages = vec![
            Message::user("What's the weather?"),
            Message {
                role: "assistant".to_string(),
                content: Some("Let me check.".to_string()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_42".to_string(),
                    r#type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "get_weather".to_string(),
                        arguments: r#"{"city":"Portland"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            Message::tool_result("call_42", "72F and sunny"),
            Message::assistant("It's 72F and sunny in Portland!"),
        ];

        let flat = flatten_tool_messages(&messages);
        assert_eq!(flat.len(), 4);

        // First message unchanged
        assert_eq!(flat[0].role, "user");

        // Assistant with tool calls flattened to text
        assert_eq!(flat[1].role, "assistant");
        assert!(flat[1].tool_calls.is_none());
        let content = flat[1].content.as_ref().unwrap();
        assert!(content.contains("Let me check."));
        assert!(content.contains("[Tool Call: get_weather"));

        // Tool result becomes user message
        assert_eq!(flat[2].role, "user");
        assert!(flat[2].content.as_ref().unwrap().contains("72F and sunny"));

        // Final assistant unchanged
        assert_eq!(flat[3].role, "assistant");
    }
}
