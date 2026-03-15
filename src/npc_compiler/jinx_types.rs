use serde::{Deserialize, Serialize};

/// A Jinx workflow template — the fundamental tool/skill unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Jinx {
    /// Unique name (from `jinx_name` field in YAML).
    #[serde(alias = "jinx_name")]
    pub name: String,

    /// Human-readable description (shown to LLM for tool selection).
    #[serde(default)]
    pub description: String,

    /// Input parameters with optional defaults and descriptions.
    #[serde(default, deserialize_with = "deserialize_inputs")]
    pub inputs: Vec<JinxInput>,

    /// Ordered execution steps.
    #[serde(default)]
    pub steps: Vec<JinxStep>,

    /// Glob patterns for files to include as context.
    #[serde(default)]
    pub file_context: Vec<String>,

    /// Optional NPC name this jinx is associated with.
    #[serde(default)]
    pub npc: Option<String>,

    /// Source file path (set during loading).
    #[serde(skip)]
    pub source_path: Option<String>,
}

/// An input parameter for a Jinx.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinxInput {
    pub name: String,
    pub default: Option<String>,
    pub description: Option<String>,
}

/// A single execution step within a Jinx.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinxStep {
    /// Step identifier (optional — auto-generated if missing).
    #[serde(default = "default_step_name")]
    pub name: String,

    /// Execution engine: "python", "bash", or another jinx name.
    #[serde(default = "default_engine")]
    pub engine: String,

    /// Code/command template (Tera/Jinja2 syntax).
    #[serde(default)]
    pub code: String,
}

fn default_engine() -> String {
    "bash".to_string()
}

fn default_step_name() -> String {
    "step".to_string()
}

/// The result of executing a Jinx.
#[derive(Debug, Clone, Default)]
pub struct JinxResult {
    /// Final output text.
    pub output: String,
    /// All accumulated context from step execution.
    pub context: std::collections::HashMap<String, serde_json::Value>,
    /// Whether execution succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

/// Custom deserializer for the inputs field, which has multiple formats:
/// - `"name"` → required input, no default
/// - `{name: "default"}` → input with default value
/// - `{name: {description: "..."}}` → input with description
fn deserialize_inputs<'de, D>(deserializer: D) -> std::result::Result<Vec<JinxInput>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum InputItem {
        /// Plain string: just a name, required
        Simple(String),
        /// Map with one key: name → default_or_description
        Map(std::collections::HashMap<String, serde_yaml::Value>),
    }

    let items: Vec<InputItem> = Vec::deserialize(deserializer)?;
    let mut inputs = Vec::with_capacity(items.len());

    for item in items {
        match item {
            InputItem::Simple(name) => {
                inputs.push(JinxInput {
                    name,
                    default: None,
                    description: None,
                });
            }
            InputItem::Map(map) => {
                for (name, value) in map {
                    match value {
                        serde_yaml::Value::String(s) => {
                            inputs.push(JinxInput {
                                name,
                                default: Some(s),
                                description: None,
                            });
                        }
                        serde_yaml::Value::Mapping(m) => {
                            let desc = m
                                .get(&serde_yaml::Value::String("description".to_string()))
                                .and_then(|v| v.as_str())
                                .map(String::from);
                            let default = m
                                .get(&serde_yaml::Value::String("default".to_string()))
                                .and_then(|v| v.as_str())
                                .map(String::from);
                            inputs.push(JinxInput {
                                name,
                                default,
                                description: desc,
                            });
                        }
                        serde_yaml::Value::Null => {
                            inputs.push(JinxInput {
                                name,
                                default: None,
                                description: None,
                            });
                        }
                        other => {
                            inputs.push(JinxInput {
                                name,
                                default: Some(format!("{:?}", other)),
                                description: None,
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(inputs)
}
