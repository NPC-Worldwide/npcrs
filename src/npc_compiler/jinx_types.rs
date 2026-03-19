use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Jinx {
    #[serde(alias = "jinx_name")]
    pub name: String,

    #[serde(default)]
    pub description: String,

    #[serde(default, deserialize_with = "deserialize_inputs")]
    pub inputs: Vec<JinxInput>,

    #[serde(default)]
    pub steps: Vec<JinxStep>,

    #[serde(default)]
    pub file_context: Vec<String>,

    #[serde(default)]
    pub npc: Option<String>,

    #[serde(skip)]
    pub source_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinxInput {
    pub name: String,
    pub default: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinxStep {
    #[serde(default = "default_step_name")]
    pub name: String,

    #[serde(default = "default_engine")]
    pub engine: String,

    #[serde(default)]
    pub code: String,
}

fn default_engine() -> String {
    "bash".to_string()
}

fn default_step_name() -> String {
    "step".to_string()
}

#[derive(Debug, Clone, Default)]
pub struct JinxResult {
    pub output: String,
    pub context: std::collections::HashMap<String, serde_json::Value>,
    pub success: bool,
    pub error: Option<String>,
}

fn deserialize_inputs<'de, D>(deserializer: D) -> std::result::Result<Vec<JinxInput>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum InputItem {
        Simple(String),
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
