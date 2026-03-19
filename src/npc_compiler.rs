use crate::error::Result;
use crate::error::{NpcError, Result};
use crate::npc_compiler::*;
use crate::npc_compiler::Jinx;
use crate::npc_compiler::Npc;
use crate::r#gen::ToolDef;
use crate::r#gen::{Message, ToolDef, LlmResponse};
use crate::r#gen::{Message, ToolDef};
use crate::tools::{RegisteredTool, ToolBuilder, ToolRegistry};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use tempfile::NamedTempFile;
use tera::{Context, Tera};
use tokio::process::Command;
use walkdir::WalkDir;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Npc {
    pub name: String,

    #[serde(default)]
    pub primary_directive: Option<String>,

    #[serde(default)]
    pub model: Option<String>,

    #[serde(default)]
    pub provider: Option<String>,

    #[serde(default)]
    pub api_url: Option<String>,

    #[serde(default)]
    pub api_key: Option<String>,

    #[serde(default)]
    pub ascii_art: Option<String>,

    #[serde(default)]
    pub colors: Option<NpcColors>,

    #[serde(default, alias = "jinxes")]
    pub jinx_names: Vec<String>,

    #[serde(default)]
    pub mcp_servers: Vec<McpServerSpec>,

    #[serde(default)]
    pub use_global_jinxes: bool,

    #[serde(skip)]
    pub memory: Option<String>,

    #[serde(skip)]
    pub shared_context: HashMap<String, serde_json::Value>,

    #[serde(skip)]
    pub source_path: Option<String>,
}

impl Default for Npc {
    fn default() -> Self {
        Self {
            name: "assistant".to_string(),
            primary_directive: None,
            model: None,
            provider: None,
            api_url: None,
            api_key: None,
            ascii_art: None,
            colors: None,
            jinx_names: Vec::new(),
            mcp_servers: Vec::new(),
            use_global_jinxes: false,
            memory: None,
            shared_context: HashMap::new(),
            source_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcColors {
    pub top: Option<String>,
    pub bottom: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct McpServerSpec {
    pub path: String,

    pub command: Option<String>,

    pub tools: Vec<String>,
}

impl<'de> Deserialize<'de> for McpServerSpec {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum McpSpec {
            Path(String),
            Full {
                path: String,
                #[serde(default)]
                command: Option<String>,
                #[serde(default)]
                tools: Vec<String>,
            },
        }

        match McpSpec::deserialize(deserializer)? {
            McpSpec::Path(path) => Ok(McpServerSpec {
                path,
                command: None,
                tools: Vec::new(),
            }),
            McpSpec::Full { path, command, tools } => Ok(McpServerSpec {
                path,
                command,
                tools,
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ToolExecutor {
    Jinx(String),
    Mcp(McpServerSpec),
    Native(String),
    Python(String),
}

impl Npc {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        load_npc_from_file(path)
    }

    pub fn new(name: impl Into<String>, primary_directive: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            primary_directive: Some(primary_directive.into()),
            ..Default::default()
        }
    }

    pub fn system_prompt(&self, team_context: Option<&str>) -> String {
        let mut parts = Vec::new();

        if let Some(ctx) = team_context {
            parts.push(ctx.to_string());
        }

        if let Some(ref directive) = self.primary_directive {
            parts.push(format!("Your name is {}.\n{}", self.name, directive));
        } else {
            parts.push(format!(
                "Your name is {}. You are a helpful assistant.",
                self.name
            ));
        }

        if let Some(ref memory) = self.memory {
            parts.push(format!("## Your Memory\n{}", memory));
        }

        parts.join("\n\n")
    }

    pub fn resolve_tools(&self, jinxes: &HashMap<String, Jinx>) -> (Vec<ToolDef>, HashMap<String, ToolExecutor>) {
        let mut defs = Vec::new();
        let mut executors = HashMap::new();

        for jinx_name in &self.jinx_names {
            if let Some(jinx) = jinxes.get(jinx_name) {
                if let Some(tool_def) = jinx.to_tool_def() {
                    executors.insert(
                        jinx.name.clone(),
                        ToolExecutor::Jinx(jinx.name.clone()),
                    );
                    defs.push(tool_def);
                }
            }
        }

        for mcp in &self.mcp_servers {
            executors.insert(
                format!("mcp:{}", mcp.path),
                ToolExecutor::Mcp(mcp.clone()),
            );
        }

        (defs, executors)
    }

    pub async fn get_response(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
    ) -> Result<LlmResponse> {
        let model = self.resolved_model();
        let provider = self.resolved_provider();

        crate::r#gen::get_genai_response(&provider, &model, messages, tools, self.api_url.as_deref()).await
    }

    pub fn resolved_model(&self) -> String {
        self.model
            .clone()
            .or_else(|| std::env::var("NPCSH_CHAT_MODEL").ok())
            .unwrap_or_else(|| "qwen3.5:2b".to_string())
    }

    pub fn resolved_provider(&self) -> String {
        self.provider
            .clone()
            .or_else(|| std::env::var("NPCSH_CHAT_PROVIDER").ok())
            .unwrap_or_else(|| "ollama".to_string())
    }

    pub async fn get_llm_response(
        &self,
        prompt: &str,
        messages: Option<&[Message]>,
        context: Option<&str>,
        format: Option<&str>,
        stream: bool,
    ) -> Result<crate::llm_funcs::LlmResponseResult> {
        crate::llm_funcs::get_llm_response_ext(
            prompt, Some(self), None, None, None,
            messages.unwrap_or(&[]), None,
            format, context, stream,
        ).await
    }

    pub async fn check_llm_command(
        &self,
        command: &str,
        messages: &mut Vec<Message>,
        jinxes: &HashMap<String, Jinx>,
        context: Option<&str>,
    ) -> Result<HashMap<String, serde_json::Value>> {
        crate::llm_funcs::check_llm_command(
            command,
            Some(&self.resolved_model()).map(|s| s.as_str()),
            Some(&self.resolved_provider()).map(|s| s.as_str()),
            Some(self),
            messages,
            context,
            jinxes,
            5,
        ).await
    }

    pub async fn execute_jinx(
        &self,
        jinx_name: &str,
        jinxes: &HashMap<String, Jinx>,
        command: &str,
        messages: &[Message],
        context: Option<&str>,
    ) -> Result<HashMap<String, serde_json::Value>> {
        crate::llm_funcs::handle_jinx_call(
            command, jinx_name, jinxes,
            Some(&self.resolved_model()).map(|s| s.as_str()),
            Some(&self.resolved_provider()).map(|s| s.as_str()),
            Some(self), messages, context, 3, 0,
        ).await
    }

    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "primary_directive": self.primary_directive,
            "model": self.model,
            "provider": self.provider,
            "api_url": self.api_url,
            "jinxes": self.jinx_names,
            "mcp_servers": self.mcp_servers.iter().map(|m| &m.path).collect::<Vec<_>>(),
        })
    }

    pub fn save(&self, directory: Option<&str>) -> Result<()> {
        let dir = directory
            .map(|d| std::path::PathBuf::from(d))
            .or_else(|| self.source_path.as_ref().and_then(|p| Path::new(p).parent().map(|p| p.to_path_buf())))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default().join("npc_team"));
        let _ = std::fs::create_dir_all(&dir);
        let filename = format!("{}.npc", self.name);
        let path = dir.join(&filename);
        let yaml = serde_yaml::to_string(self).map_err(|e| crate::error::NpcError::Shell(format!("YAML serialize: {}", e)))?;
        std::fs::write(&path, yaml).map_err(|e| crate::error::NpcError::Shell(format!("Write NPC: {}", e)))?;
        Ok(())
    }

    pub async fn search_my_conversations(&self, query: &str, limit: usize) -> Result<String> {
        let db_path = crate::npc_sysenv::get_history_db_path();
        let conn = rusqlite::Connection::open(&db_path)?;
        let pattern = format!("%{}%", query);
        let mut stmt = conn.prepare(
            "SELECT content, role, timestamp FROM conversation_history WHERE npc = ?1 AND content LIKE ?2 ORDER BY timestamp DESC LIMIT ?3"
        )?;
        let results: Vec<String> = stmt.query_map(
            rusqlite::params![self.name, pattern, limit as i64],
            |row| {
                let content: String = row.get(0)?;
                let role: String = row.get(1)?;
                Ok(format!("[{}] {}", role, content))
            }
        )?.filter_map(|r| r.ok()).collect();
        Ok(results.join("\n"))
    }

    pub async fn search_my_memories(&self, query: &str, limit: usize) -> Result<String> {
        let db_path = crate::npc_sysenv::get_history_db_path();
        let conn = rusqlite::Connection::open(&db_path)?;
        let pattern = format!("%{}%", query);
        let mut stmt = conn.prepare(
            "SELECT content FROM memory_lifecycle WHERE npc = ?1 AND content LIKE ?2 AND status IN ('approved', 'human-approved') ORDER BY created_at DESC LIMIT ?3"
        )?;
        let results: Vec<String> = stmt.query_map(
            rusqlite::params![self.name, pattern, limit as i64],
            |row| row.get::<_, String>(0)
        )?.filter_map(|r| r.ok()).collect();
        Ok(results.join("\n"))
    }

    pub async fn think_step_by_step(&self, problem: &str) -> Result<String> {
        let prompt = format!(
            "Think through this problem step by step:\n\n{}\n\nProvide a clear, numbered breakdown of your reasoning.",
            problem
        );
        let result = self.get_llm_response(&prompt, None, None, None, false).await?;
        Ok(result.response.unwrap_or_default())
    }

    pub async fn write_code(&self, task: &str, language: &str) -> Result<String> {
        let prompt = format!(
            "Write {} code for the following task:\n\n{}\n\nRespond with only the code, no markdown formatting.",
            language, task
        );
        let result = self.get_llm_response(&prompt, None, None, None, false).await?;
        Ok(result.response.unwrap_or_default())
    }

    pub fn create_memory(&self, content: &str, memory_type: &str) -> Result<Option<i64>> {
        let db_path = crate::npc_sysenv::get_history_db_path();
        let conn = rusqlite::Connection::open(&db_path)?;
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO memory_lifecycle (npc, memory_text, memory_type, status, created_at) VALUES (?1, ?2, ?3, 'pending', ?4)",
            rusqlite::params![self.name, content, memory_type, now],
        )?;
        Ok(Some(conn.last_insert_rowid()))
    }

    pub fn read_memory(&self, memory_id: i64) -> Result<Option<HashMap<String, serde_json::Value>>> {
        let db_path = crate::npc_sysenv::get_history_db_path();
        let conn = rusqlite::Connection::open(&db_path)?;
        let result = conn.query_row(
            "SELECT id, memory_text, memory_type, status, created_at FROM memory_lifecycle WHERE id = ?1 AND npc = ?2",
            rusqlite::params![memory_id, self.name],
            |row| {
                let mut m = HashMap::new();
                m.insert("id".into(), serde_json::json!(row.get::<_, i64>(0)?));
                m.insert("content".into(), serde_json::json!(row.get::<_, String>(1)?));
                m.insert("type".into(), serde_json::json!(row.get::<_, String>(2)?));
                m.insert("status".into(), serde_json::json!(row.get::<_, String>(3)?));
                m.insert("created_at".into(), serde_json::json!(row.get::<_, String>(4)?));
                Ok(m)
            }
        );
        match result {
            Ok(m) => Ok(Some(m)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn delete_memory(&self, memory_id: i64) -> Result<bool> {
        let db_path = crate::npc_sysenv::get_history_db_path();
        let conn = rusqlite::Connection::open(&db_path)?;
        let rows = conn.execute(
            "DELETE FROM memory_lifecycle WHERE id = ?1 AND npc = ?2",
            rusqlite::params![memory_id, self.name],
        )?;
        Ok(rows > 0)
    }

    pub fn search_memories(&self, query: &str, limit: usize, status_filter: Option<&str>) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let db_path = crate::npc_sysenv::get_history_db_path();
        let conn = rusqlite::Connection::open(&db_path)?;
        let pattern = format!("%{}%", query);
        let sql = if let Some(status) = status_filter {
            format!("SELECT id, memory_text, memory_type, status, created_at FROM memory_lifecycle WHERE npc = ?1 AND memory_text LIKE ?2 AND status = '{}' ORDER BY created_at DESC LIMIT ?3", status)
        } else {
            "SELECT id, memory_text, memory_type, status, created_at FROM memory_lifecycle WHERE npc = ?1 AND memory_text LIKE ?2 ORDER BY created_at DESC LIMIT ?3".to_string()
        };
        let mut stmt = conn.prepare(&sql)?;
        let results = stmt.query_map(
            rusqlite::params![self.name, pattern, limit as i64],
            |row| {
                let mut m = HashMap::new();
                m.insert("id".into(), serde_json::json!(row.get::<_, i64>(0)?));
                m.insert("content".into(), serde_json::json!(row.get::<_, String>(1)?));
                m.insert("type".into(), serde_json::json!(row.get::<_, String>(2)?));
                m.insert("status".into(), serde_json::json!(row.get::<_, String>(3)?));
                m.insert("created_at".into(), serde_json::json!(row.get::<_, String>(4)?));
                Ok(m)
            }
        )?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn get_all_memories(&self, limit: usize, status_filter: Option<&str>) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        self.search_memories("", limit, status_filter)
    }

    pub fn get_memory_stats(&self) -> Result<HashMap<String, i64>> {
        let db_path = crate::npc_sysenv::get_history_db_path();
        let conn = rusqlite::Connection::open(&db_path)?;
        let mut stats = HashMap::new();
        let total: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memory_lifecycle WHERE npc = ?1",
            rusqlite::params![self.name], |row| row.get(0)
        ).unwrap_or(0);
        let pending: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memory_lifecycle WHERE npc = ?1 AND status = 'pending'",
            rusqlite::params![self.name], |row| row.get(0)
        ).unwrap_or(0);
        let approved: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memory_lifecycle WHERE npc = ?1 AND status IN ('approved', 'human-approved')",
            rusqlite::params![self.name], |row| row.get(0)
        ).unwrap_or(0);
        stats.insert("total".into(), total);
        stats.insert("pending".into(), pending);
        stats.insert("approved".into(), approved);
        Ok(stats)
    }

    pub fn get_memory_context(&self) -> Option<String> {
        let db_path = crate::npc_sysenv::get_history_db_path();
        let conn = rusqlite::Connection::open(&db_path).ok()?;
        let mut stmt = conn.prepare(
            "SELECT memory_text FROM memory_lifecycle WHERE npc = ?1 AND status IN ('approved', 'human-approved') ORDER BY created_at DESC LIMIT 20"
        ).ok()?;
        let memories: Vec<String> = stmt.query_map(
            rusqlite::params![self.name],
            |row| row.get::<_, String>(0)
        ).ok()?.filter_map(|r| r.ok()).collect();
        if memories.is_empty() { return None; }
        Some(memories.iter().map(|m| format!("- {}", m)).collect::<Vec<_>>().join("\n"))
    }
}

pub fn load_npc_from_file(path: impl AsRef<Path>) -> Result<Npc> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path).map_err(|e| NpcError::FileLoad {
        path: path.display().to_string(),
        source: e,
    })?;

    let raw = if raw.starts_with("#!") {
        raw.splitn(2, '\n').nth(1).unwrap_or("").to_string()
    } else {
        raw
    };

    let processed = preprocess_npc_yaml(&raw);

    let mut npc: Npc =
        serde_yaml::from_str(&processed).map_err(|e| NpcError::YamlParse {
            path: path.display().to_string(),
            source: e,
        })?;

    npc.source_path = Some(path.display().to_string());

    for mcp in &mut npc.mcp_servers {
        mcp.path = shellexpand::tilde(&mcp.path).to_string();
    }

    Ok(npc)
}

fn preprocess_npc_yaml(raw: &str) -> String {
    let mut output = String::with_capacity(raw.len());
    let mut in_for_block = false;
    let mut for_glob_pattern: Option<String> = None;

    for line in raw.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("{%") && trimmed.contains("jinxes_list") {
            if let Some(pattern) = extract_jinxes_list_pattern(trimmed) {
                for_glob_pattern = Some(pattern);
                in_for_block = true;
            }
            continue;
        }

        if trimmed.starts_with("{%") && trimmed.contains("endfor") {
            if in_for_block {
                if let Some(ref pattern) = for_glob_pattern {
                    let prefix = pattern.trim_end_matches('*').trim_end_matches('_');
                    let names = expand_jinx_glob(pattern);
                    for name in names {
                        output.push_str(&format!("  - {}\n", name));
                    }
                }
                in_for_block = false;
                for_glob_pattern = None;
            }
            continue;
        }

        if in_for_block {
            continue;
        }

        if trimmed.starts_with("{%") {
            continue;
        }

        if let Some(extracted) = extract_jinx_call(trimmed) {
            let indent = &line[..line.len() - line.trim_start().len()];
            output.push_str(&format!("{indent}- {extracted}\n"));
        } else {
            output.push_str(line);
            output.push('\n');
        }
    }

    output
}

fn extract_jinxes_list_pattern(line: &str) -> Option<String> {
    let start = line.find("jinxes_list(")?;
    let rest = &line[start + "jinxes_list(".len()..];
    let end = rest.find(')')?;
    let pattern = rest[..end].trim().trim_matches('\'').trim_matches('"');
    Some(pattern.to_string())
}

fn expand_jinx_glob(pattern: &str) -> Vec<String> {
    let locations = [
        shellexpand::tilde("~/.npcsh/npc_team/jinxes").to_string(),
        "./npc_team/jinxes".to_string(),
    ];

    for base in &locations {
        let glob_pattern = format!("{}/{}.jinx", base, pattern);
        if let Ok(paths) = glob::glob(&glob_pattern) {
            let names: Vec<String> = paths
                .filter_map(|p| p.ok())
                .filter_map(|p| {
                    p.file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                })
                .collect();
            if !names.is_empty() {
                return names;
            }
        }
    }

    Vec::new()
}

fn extract_jinx_call(line: &str) -> Option<String> {
    let line = line.trim().trim_start_matches('-').trim();

    if !line.starts_with("{{") || !line.ends_with("}}") {
        return None;
    }

    let inner = line
        .trim_start_matches("{{")
        .trim_end_matches("}}")
        .trim();

    if !inner.starts_with("Jinx(") {
        return None;
    }

    let name_part = inner
        .trim_start_matches("Jinx(")
        .trim_end_matches(')')
        .trim()
        .trim_matches('\'')
        .trim_matches('"');

    if name_part.is_empty() {
        return None;
    }

    Some(name_part.to_string())
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_extract_jinx_call() {
        assert_eq!(
            extract_jinx_call("- {{ Jinx('edit_file') }}"),
            Some("edit_file".to_string())
        );
        assert_eq!(
            extract_jinx_call("  - {{ Jinx(\"web_search\") }}"),
            Some("web_search".to_string())
        );
        assert_eq!(extract_jinx_call("- plain_string"), None);
        assert_eq!(extract_jinx_call("name: value"), None);
    }

    #[test]
    fn test_preprocess_strips_jinja() {
        let input = r#"
name: test_npc
primary_directive: Do things
jinxes:
  - {{ Jinx('edit_file') }}
  - {{ Jinx('sh') }}
  - plain_jinx
"#;
        let output = preprocess_npc_yaml(input);
        assert!(output.contains("- edit_file"));
        assert!(output.contains("- sh"));
        assert!(output.contains("- plain_jinx"));
        assert!(!output.contains("Jinx("));
    }
}

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

impl Jinx {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        load_jinx_from_file(path)
    }

    pub fn to_tool_def(&self) -> Option<ToolDef> {
        if self.name.is_empty() || self.description.is_empty() {
            return None;
        }

        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for input in &self.inputs {
            let mut prop = serde_json::Map::new();
            prop.insert("type".into(), serde_json::Value::String("string".into()));

            if let Some(ref desc) = input.description {
                prop.insert(
                    "description".into(),
                    serde_json::Value::String(desc.clone()),
                );
            }

            properties.insert(
                input.name.clone(),
                serde_json::Value::Object(prop),
            );

            if input.default.is_none() {
                required.push(serde_json::Value::String(input.name.clone()));
            }
        }

        Some(ToolDef {
            r#type: "function".to_string(),
            function: crate::r#gen::FunctionDef {
                name: self.name.clone(),
                description: Some(self.description.clone()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }),
            },
        })
    }

    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "jinx_name": self.name,
            "description": self.description,
            "inputs": self.inputs.iter().map(|i| {
                if let Some(ref def) = i.default {
                    serde_json::json!({i.name.clone(): def})
                } else {
                    serde_json::json!(i.name.clone())
                }
            }).collect::<Vec<_>>(),
            "steps": self.steps.iter().map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "engine": s.engine,
                    "code": s.code,
                })
            }).collect::<Vec<_>>(),
        })
    }

    pub fn save(&self, directory: &str) -> Result<()> {
        let dir = std::path::Path::new(directory);
        let _ = std::fs::create_dir_all(dir);
        let filename = format!("{}.jinx", self.name);
        let path = dir.join(&filename);
        let yaml = serde_yaml::to_string(self).map_err(|e| crate::error::NpcError::Shell(format!("YAML serialize: {}", e)))?;
        std::fs::write(&path, yaml).map_err(|e| crate::error::NpcError::Shell(format!("Write jinx: {}", e)))?;
        Ok(())
    }

    pub fn execute(
        &self,
        input_values: &std::collections::HashMap<String, String>,
    ) -> crate::npc_compiler::JinxResult {
        let mut output = String::new();
        let mut context = std::collections::HashMap::new();
        let mut success = true;

        for step in &self.steps {
            let mut rendered = step.code.clone();
            for (k, v) in input_values {
                rendered = rendered.replace(&format!("{{{{ {} }}}}", k), v);
                rendered = rendered.replace(&format!("{{{{{}}}}}", k), v);
            }
            for (k, v) in &context {
                if let Some(s) = v.as_str() {
                    rendered = rendered.replace(&format!("{{{{ {} }}}}", k), s);
                }
            }

            let result = match step.engine.as_str() {
                "bash" | "sh" => {
                    std::process::Command::new("sh").args(["-c", &rendered]).output()
                }
                "python" | "python3" => {
                    std::process::Command::new("python3").args(["-c", &rendered]).output()
                }
                _ => {
                    output.push_str(&format!("Unknown engine: {}\n", step.engine));
                    success = false;
                    continue;
                }
            };

            match result {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
                    output.push_str(&stdout);
                    if !stderr.is_empty() {
                        output.push_str(&stderr);
                    }
                    context.insert(step.name.clone(), serde_json::json!(stdout.trim()));
                    if !o.status.success() {
                        success = false;
                    }
                }
                Err(e) => {
                    output.push_str(&format!("Error: {}\n", e));
                    success = false;
                }
            }
        }

        crate::npc_compiler::JinxResult {
            output,
            context,
            success,
            error: if success { None } else { Some("Step failed".into()) },
        }
    }

    pub fn render_first_pass(
        &self,
        input_values: &std::collections::HashMap<String, String>,
    ) -> String {
        let mut rendered_steps = Vec::new();
        for step in &self.steps {
            let mut code = step.code.clone();
            for (k, v) in input_values {
                code = code.replace(&format!("{{{{ {} }}}}", k), v);
                code = code.replace(&format!("{{{{{}}}}}", k), v);
            }
            rendered_steps.push(format!("[{}:{}] {}", step.name, step.engine, code));
        }
        rendered_steps.join("\n")
    }
}

pub fn load_jinx_from_file(path: impl AsRef<Path>) -> Result<Jinx> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path).map_err(|e| NpcError::FileLoad {
        path: path.display().to_string(),
        source: e,
    })?;

    let raw = if raw.starts_with("#!") {
        raw.splitn(2, '\n').nth(1).unwrap_or("").to_string()
    } else {
        raw
    };

    let cleaned = strip_jinja2_specifics(&raw);

    let mut jinx: Jinx =
        serde_yaml::from_str(&cleaned).map_err(|e| NpcError::YamlParse {
            path: path.display().to_string(),
            source: e,
        })?;

    jinx.source_path = Some(path.display().to_string());

    Ok(jinx)
}

pub fn load_jinxes_from_directory(dir: impl AsRef<Path>) -> Result<HashMap<String, Jinx>> {
    let dir = dir.as_ref();
    let mut jinxes = HashMap::new();

    if !dir.exists() {
        return Ok(jinxes);
    }

    for entry in WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "jinx") {
            match load_jinx_from_file(path) {
                Ok(jinx) => {
                    let name = if jinx.name.is_empty() {
                        path.file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string()
                    } else {
                        jinx.name.clone()
                    };
                    jinxes.insert(name, jinx);
                }
                Err(e) => {
                    tracing::warn!("Failed to load jinx {}: {}", path.display(), e);
                }
            }
        }
    }

    Ok(jinxes)
}

fn strip_jinja2_specifics(raw: &str) -> String {
    raw.to_string()
}

#[cfg(test)]
mod tests {

    fn write_temp_jinx(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::with_suffix(".jinx").unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_load_simple_jinx() {
        let f = write_temp_jinx(
            r#"
jinx_name: test_jinx
description: A test jinx
inputs:
  - query
steps:
  - name: run
    engine: bash
    code: echo "hello {{ query }}"
"#,
        );
        let jinx = load_jinx_from_file(f.path()).unwrap();
        assert_eq!(jinx.name, "test_jinx");
        assert_eq!(jinx.inputs.len(), 1);
        assert_eq!(jinx.inputs[0].name, "query");
        assert!(jinx.inputs[0].default.is_none());
        assert_eq!(jinx.steps.len(), 1);
        assert_eq!(jinx.steps[0].engine, "bash");
    }

    #[test]
    fn test_load_jinx_with_defaults() {
        let f = write_temp_jinx(
            r#"
jinx_name: edit_file
description: Edit files
inputs:
  - path:
      description: "File path"
  - action: "create"
  - content
steps:
  - name: edit
    engine: python
    code: |
      print("editing")
"#,
        );
        let jinx = load_jinx_from_file(f.path()).unwrap();
        assert_eq!(jinx.inputs.len(), 3);
        assert_eq!(jinx.inputs[0].name, "path");
        assert!(jinx.inputs[0].description.is_some());
        assert_eq!(jinx.inputs[1].name, "action");
        assert_eq!(jinx.inputs[1].default.as_deref(), Some("create"));
        assert_eq!(jinx.inputs[2].name, "content");
        assert!(jinx.inputs[2].default.is_none());
    }
}

pub async fn execute_jinx(
    jinx: &Jinx,
    input_values: &HashMap<String, String>,
    available_jinxes: &HashMap<String, Jinx>,
) -> Result<JinxResult> {
    let mut context: HashMap<String, serde_json::Value> = HashMap::new();
    let mut output = String::new();

    for input in &jinx.inputs {
        let value = input_values
            .get(&input.name)
            .cloned()
            .or_else(|| input.default.clone())
            .unwrap_or_default();
        context.insert(
            input.name.clone(),
            serde_json::Value::String(value),
        );
    }

    let needs_tty = jinx_needs_tty(jinx);

    for step in &jinx.steps {
        let result = if needs_tty {
            execute_step_interactive(step, &context, available_jinxes).await
        } else {
            execute_step(step, &context, available_jinxes).await
        };

        match result {
            Ok(step_output) => {
                output = step_output.clone();
                context.insert(
                    step.name.clone(),
                    serde_json::Value::String(step_output.clone()),
                );
                context.insert(
                    "output".to_string(),
                    serde_json::Value::String(step_output),
                );
            }
            Err(e) => {
                return Ok(JinxResult {
                    output: format!("Error in step '{}': {}", step.name, e),
                    context,
                    success: false,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    Ok(JinxResult {
        output,
        context,
        success: true,
        error: None,
    })
}

fn jinx_needs_tty(jinx: &Jinx) -> bool {
    for step in &jinx.steps {
        let code = &step.code;
        if code.contains("termios")
            || code.contains("tty.setraw")
            || code.contains("curses")
            || code.contains("sys.stdin.isatty")
            || code.contains("select.select")
            || code.contains("getch")
        {
            return true;
        }
    }
    false
}

async fn execute_step(
    step: &JinxStep,
    context: &HashMap<String, serde_json::Value>,
    available_jinxes: &HashMap<String, Jinx>,
) -> Result<String> {
    match step.engine.as_str() {
        "python" => {
            let rendered = render_python_template(&step.code, context);
            execute_python(&rendered, context).await
        }
        "bash" => {
            let rendered = render_step_template(&step.code, context)?;
            execute_bash(&rendered).await
        }
        engine_name => execute_sub_jinx(engine_name, step, context, available_jinxes).await,
    }
}

async fn execute_step_interactive(
    step: &JinxStep,
    context: &HashMap<String, serde_json::Value>,
    available_jinxes: &HashMap<String, Jinx>,
) -> Result<String> {
    match step.engine.as_str() {
        "python" => {
            let rendered = render_python_template(&step.code, context);
            execute_python_interactive(&rendered, context).await
        }
        "bash" => {
            let rendered = render_step_template(&step.code, context)?;
            execute_bash_interactive(&rendered).await
        }
        engine_name => execute_sub_jinx(engine_name, step, context, available_jinxes).await,
    }
}

fn render_python_template(code: &str, context: &HashMap<String, serde_json::Value>) -> String {
    let re = regex::Regex::new(r"\{\{(.*?)\}\}").unwrap();

    re.replace_all(code, |caps: &regex::Captures| {
        let expr = caps[1].trim();
        resolve_template_expr(expr, context)
    })
    .to_string()
}

fn resolve_template_expr(expr: &str, context: &HashMap<String, serde_json::Value>) -> String {
    let parts: Vec<&str> = expr.split('|').map(|s| s.trim()).collect();
    if parts.is_empty() {
        return String::new();
    }

    let var_name = parts[0];

    let mut value = context.get(var_name).cloned();

    let mut use_tojson = false;
    for filter in &parts[1..] {
        if filter.starts_with("default(") {
            if value.is_none() || value.as_ref().is_some_and(|v| v.as_str() == Some("")) {
                let default_str = filter
                    .trim_start_matches("default(")
                    .trim_end_matches(')')
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'');
                value = Some(serde_json::Value::String(default_str.to_string()));
            }
        } else if *filter == "tojson" {
            use_tojson = true;
        }
    }

    match value {
        Some(v) => {
            if use_tojson {
                serde_json::to_string(&v).unwrap_or_else(|_| "null".to_string())
            } else {
                v.as_str().unwrap_or(&v.to_string()).to_string()
            }
        }
        None => {
            if use_tojson {
                "null".to_string()
            } else {
                String::new()
            }
        }
    }
}

async fn execute_sub_jinx(
    engine_name: &str,
    step: &JinxStep,
    context: &HashMap<String, serde_json::Value>,
    available_jinxes: &HashMap<String, Jinx>,
) -> Result<String> {
    if let Some(sub_jinx) = available_jinxes.get(engine_name) {
        let inputs: HashMap<String, String> = context
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str().unwrap_or(&v.to_string()).to_string(),
                )
            })
            .collect();
        let result = Box::pin(execute_jinx(sub_jinx, &inputs, available_jinxes)).await?;
        if result.success {
            Ok(result.output)
        } else {
            Err(NpcError::JinxExecution {
                step: step.name.clone(),
                reason: result.error.unwrap_or_default(),
            })
        }
    } else {
        Err(NpcError::JinxNotFound {
            name: engine_name.to_string(),
        })
    }
}

fn render_step_template(
    template: &str,
    context: &HashMap<String, serde_json::Value>,
) -> Result<String> {
    let mut tera = Tera::default();
    tera.add_raw_template("step", template)?;

    let mut ctx = Context::new();
    for (key, value) in context {
        ctx.insert(key, value);
    }

    Ok(tera.render("step", &ctx)?)
}

fn wrap_python_with_context(code: &str, context: &HashMap<String, serde_json::Value>) -> String {
    let context_json = serde_json::to_string(context).unwrap_or_else(|_| "{}".to_string());

    let indented_code = code
        .lines()
        .map(|l| format!("    {}", l))
        .collect::<Vec<_>>()
        .join("\n");

    let escaped_json = context_json
        .replace('\\', "\\\\")
        .replace('\'', "\\'");

    let mut wrapper = String::new();
    wrapper.push_str("import json, sys, os\n");
    wrapper.push_str(&format!("context = json.loads('{}')\n", escaped_json));
    wrapper.push_str("output = \"\"\n");
    wrapper.push_str("class _State:\n");
    wrapper.push_str("    current_path = os.getcwd()\n");
    wrapper.push_str("    chat_model = os.environ.get('NPCSH_CHAT_MODEL', 'gpt-4o-mini')\n");
    wrapper.push_str("    chat_provider = os.environ.get('NPCSH_CHAT_PROVIDER', 'openai')\n");
    wrapper.push_str("    stream_output = False\n");
    wrapper.push_str("state = _State()\n");
    wrapper.push_str("class _NPC:\n");
    wrapper.push_str("    name = \"assistant\"\n");
    wrapper.push_str("npc = _NPC()\n");
    wrapper.push_str("try:\n");
    wrapper.push_str(&indented_code);
    wrapper.push('\n');
    wrapper.push_str("except Exception as e:\n");
    wrapper.push_str("    context['output'] = f'Error: {e}'\n");
    wrapper.push_str("    output = str(e)\n");
    wrapper.push_str("result = context.get('output', output)\n");
    wrapper.push_str("if result:\n");
    wrapper.push_str("    print(result, end='')\n");

    wrapper
}

async fn execute_bash(code: &str) -> Result<String> {
    let output = Command::new("bash")
        .arg("-c")
        .arg(code)
        .output()
        .await
        .map_err(|e| NpcError::JinxExecution {
            step: "bash".to_string(),
            reason: e.to_string(),
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() {
        Ok(stdout.to_string())
    } else {
        Ok(format!(
            "{}{}[exit code: {}]",
            stdout,
            if stderr.is_empty() {
                String::new()
            } else {
                format!("\nSTDERR: {}", stderr)
            },
            output.status.code().unwrap_or(-1)
        ))
    }
}

async fn execute_bash_interactive(code: &str) -> Result<String> {
    let status = Command::new("bash")
        .arg("-c")
        .arg(code)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .await
        .map_err(|e| NpcError::JinxExecution {
            step: "bash".to_string(),
            reason: e.to_string(),
        })?;

    Ok(if status.success() {
        String::new()
    } else {
        format!("[exit code: {}]", status.code().unwrap_or(-1))
    })
}

async fn execute_python(code: &str, context: &HashMap<String, serde_json::Value>) -> Result<String> {
    let wrapped = wrap_python_with_context(code, context);

    let output = Command::new("python3")
        .arg("-c")
        .arg(&wrapped)
        .output()
        .await
        .map_err(|e| NpcError::JinxExecution {
            step: "python".to_string(),
            reason: format!("Failed to run Python: {}", e),
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() {
        Ok(stdout.to_string())
    } else {
        Ok(format!(
            "{}{}[python exit code: {}]",
            stdout,
            if stderr.is_empty() {
                String::new()
            } else {
                format!("\nSTDERR: {}", stderr)
            },
            output.status.code().unwrap_or(-1)
        ))
    }
}

async fn execute_python_interactive(
    code: &str,
    context: &HashMap<String, serde_json::Value>,
) -> Result<String> {
    let wrapped = wrap_python_with_context(code, context);

    let status = Command::new("python3")
        .arg("-c")
        .arg(&wrapped)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .await
        .map_err(|e| NpcError::JinxExecution {
            step: "python".to_string(),
            reason: format!("Failed to run Python: {}", e),
        })?;

    Ok(if status.success() {
        String::new()
    } else {
        format!("[python exit code: {}]", status.code().unwrap_or(-1))
    })
}

#[derive(Debug, Clone)]
pub struct Team {
    pub npcs: HashMap<String, Npc>,

    pub forenpc: Option<String>,

    pub jinxes: HashMap<String, Jinx>,

    pub context: Option<String>,

    pub model: Option<String>,

    pub provider: Option<String>,

    pub shared_context: HashMap<String, serde_json::Value>,

    pub databases: Vec<String>,

    pub mcp_servers: Vec<crate::npc_compiler::McpServerSpec>,

    pub source_dir: Option<String>,
}

impl Default for Team {
    fn default() -> Self {
        Self {
            npcs: HashMap::new(),
            forenpc: None,
            jinxes: HashMap::new(),
            context: None,
            model: None,
            provider: None,
            shared_context: HashMap::new(),
            databases: Vec::new(),
            mcp_servers: Vec::new(),
            source_dir: None,
        }
    }
}

impl Team {
    pub fn get_npc(&self, name: &str) -> Option<&Npc> {
        self.npcs.get(name)
    }

    pub fn get_npc_mut(&mut self, name: &str) -> Option<&mut Npc> {
        self.npcs.get_mut(name)
    }

    pub fn lead_npc(&self) -> Option<&Npc> {
        self.forenpc
            .as_ref()
            .and_then(|name| self.npcs.get(name))
            .or_else(|| self.npcs.values().next())
    }

    pub fn npc_names(&self) -> Vec<&str> {
        self.npcs.keys().map(|s| s.as_str()).collect()
    }

    pub fn jinx_names(&self) -> Vec<&str> {
        self.jinxes.keys().map(|s| s.as_str()).collect()
    }

    pub fn get_forenpc(&self) -> Option<&Npc> {
        self.lead_npc()
    }

    pub async fn orchestrate(&self, request: &str) -> crate::error::Result<HashMap<String, serde_json::Value>> {
        let forenpc = self.get_forenpc().ok_or_else(|| crate::error::NpcError::Shell("No forenpc set".into()))?;
        let model = forenpc.resolved_model();
        let provider = forenpc.resolved_provider();

        let team_members: Vec<String> = self.npcs.keys()
            .filter(|n| Some(n.as_str()) != self.forenpc.as_deref())
            .map(|n| format!("@{}", n))
            .collect();

        let prompt = format!(
            "You are the team coordinator. Team members: {}\n\nRequest: {}\n\nDecide how to handle this. Delegate to team members using @name if needed, or answer directly.",
            team_members.join(", "), request
        );

        let mut messages = Vec::new();
        let result = crate::llm_funcs::check_llm_command(
            &prompt,
            Some(model.as_str()),
            Some(provider.as_str()),
            Some(forenpc),
            &mut messages,
            self.context.as_deref(),
            &self.jinxes,
            3,
        ).await?;

        Ok(result)
    }

    pub fn update_context(&mut self, messages: &[crate::r#gen::Message]) {
        let recent: String = messages.iter()
            .rev().take(5)
            .filter_map(|m| m.content.as_ref().map(|c| format!("{}: {}", m.role, c)))
            .collect::<Vec<_>>().join("\n");
        self.shared_context.insert("recent_messages".into(), serde_json::json!(recent));
    }

    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "forenpc": self.forenpc,
            "model": self.model,
            "provider": self.provider,
            "context": self.context,
            "npcs": self.npcs.keys().collect::<Vec<_>>(),
            "jinxes": self.jinxes.keys().collect::<Vec<_>>(),
        })
    }

    pub fn save(&self, directory: Option<&str>) -> crate::error::Result<()> {
        let dir = directory
            .map(std::path::PathBuf::from)
            .or_else(|| self.source_dir.as_ref().map(std::path::PathBuf::from))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default().join("npc_team"));
        let _ = std::fs::create_dir_all(&dir);

        for npc in self.npcs.values() {
            npc.save(Some(dir.to_str().unwrap_or(".")))?;
        }

        for jinx in self.jinxes.values() {
            let jinx_dir = dir.join("jinxes");
            jinx.save(jinx_dir.to_str().unwrap_or("."))?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamCtx {
    #[serde(default)]
    pub context: Option<String>,

    #[serde(default)]
    pub model: Option<String>,

    #[serde(default)]
    pub provider: Option<String>,

    #[serde(default)]
    pub api_url: Option<String>,

    #[serde(default)]
    pub api_key: Option<String>,

    #[serde(default)]
    pub forenpc: Option<String>,

    #[serde(default)]
    pub databases: Vec<String>,

    #[serde(default)]
    pub mcp_servers: Vec<crate::npc_compiler::McpServerSpec>,

    #[serde(default)]
    pub use_global_jinxes: bool,

    #[serde(default)]
    pub preferences: Vec<String>,
}

pub fn load_team_from_directory(dir: impl AsRef<Path>) -> Result<Team> {
    let dir = dir.as_ref();
    let mut team = Team {
        source_dir: Some(dir.display().to_string()),
        ..Default::default()
    };

    if !dir.exists() {
        return Ok(team);
    }

    if let Some(ctx) = find_and_load_ctx(dir)? {
        team.context = ctx.context;
        team.model = ctx.model;
        team.provider = ctx.provider;
        team.forenpc = ctx.forenpc;
        team.databases = ctx
            .databases
            .into_iter()
            .map(|d| shellexpand::tilde(&d).to_string())
            .collect();
        team.mcp_servers = ctx.mcp_servers;
    }

    for entry in WalkDir::new(dir)
        .max_depth(1)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "npc") {
            match super::load_npc_from_file(path) {
                Ok(mut loaded_npc) => {
                    if loaded_npc.model.is_none() {
                        loaded_npc.model = team.model.clone();
                    }
                    if loaded_npc.provider.is_none() {
                        loaded_npc.provider = team.provider.clone();
                    }
                    team.npcs.insert(loaded_npc.name.clone(), loaded_npc);
                }
                Err(e) => {
                    tracing::warn!("Failed to load NPC {}: {}", path.display(), e);
                }
            }
        }
    }

    let jinxes_dir = dir.join("jinxes");
    if jinxes_dir.exists() {
        team.jinxes = super::load_jinxes_from_directory(&jinxes_dir)?;
    }

    let legacy_dir = dir.join("jinxs");
    if legacy_dir.exists() && !jinxes_dir.exists() {
        team.jinxes = super::load_jinxes_from_directory(&legacy_dir)?;
    }

    let project_root = dir.parent().unwrap_or(dir);
    let agents_md = project_root.join("agents.md");
    if agents_md.exists() {
        if let Ok(content) = std::fs::read_to_string(&agents_md) {
            load_agents_from_md(&content, &team.model, &team.provider, &mut team.npcs);
        }
    }
    let agents_dir = project_root.join("agents");
    if agents_dir.is_dir() {
        load_agents_from_dir(&agents_dir, &team.model, &team.provider, &mut team.npcs);
    }

    for npc in team.npcs.values_mut() {
        if npc.jinx_names.iter().any(|n| n == "*") {
            npc.jinx_names = team.jinxes.keys().cloned().collect();
        }
    }

    tracing::info!(
        "Loaded team from {}: {} NPCs, {} jinxes, forenpc={:?}",
        dir.display(),
        team.npcs.len(),
        team.jinxes.len(),
        team.forenpc
    );

    Ok(team)
}

fn find_and_load_ctx(dir: &Path) -> Result<Option<TeamCtx>> {
    let candidates = ["team.ctx", "npcsh.ctx"];

    for name in &candidates {
        let path = dir.join(name);
        if path.exists() {
            return load_ctx_file(&path).map(Some);
        }
    }

    for entry in std::fs::read_dir(dir).map_err(|e| NpcError::FileLoad {
        path: dir.display().to_string(),
        source: e,
    })? {
        let entry = entry.map_err(|e| NpcError::FileLoad {
            path: dir.display().to_string(),
            source: e,
        })?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "ctx") {
            return load_ctx_file(&path).map(Some);
        }
    }

    Ok(None)
}

fn load_agents_from_md(
    content: &str,
    team_model: &Option<String>,
    team_provider: &Option<String>,
    npcs: &mut std::collections::HashMap<String, crate::npc_compiler::Npc>,
) {
    let mut current_name: Option<String> = None;
    let mut current_body: Vec<String> = Vec::new();

    for line in content.lines() {
        if let Some(name) = line.strip_prefix("## ") {
            if let Some(prev_name) = current_name.take() {
                if !npcs.contains_key(&prev_name) {
                    let mut npc = crate::npc_compiler::Npc::new(&prev_name, current_body.join("\n").trim());
                    npc.model = team_model.clone();
                    npc.provider = team_provider.clone();
                    npcs.insert(prev_name, npc);
                }
            }
            current_name = Some(name.trim().to_string());
            current_body.clear();
        } else if current_name.is_some() {
            current_body.push(line.to_string());
        }
    }

    if let Some(name) = current_name {
        if !npcs.contains_key(&name) {
            let mut npc = crate::npc_compiler::Npc::new(&name, current_body.join("\n").trim());
            npc.model = team_model.clone();
            npc.provider = team_provider.clone();
            npcs.insert(name, npc);
        }
    }
}

fn load_agents_from_dir(
    dir: &Path,
    team_model: &Option<String>,
    team_provider: &Option<String>,
    npcs: &mut std::collections::HashMap<String, crate::npc_compiler::Npc>,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        if name.is_empty() || npcs.contains_key(&name) {
            continue;
        }

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut model = team_model.clone();
        let mut provider = team_provider.clone();
        let mut agent_name = name.clone();
        let directive;

        if content.starts_with("---") {
            let parts: Vec<&str> = content.splitn(3, "---").collect();
            if parts.len() >= 3 {
                if let Ok(fm) = serde_yaml::from_str::<serde_yaml::Value>(parts[1]) {
                    if let Some(m) = fm.get("model").and_then(|v| v.as_str()) {
                        model = Some(m.to_string());
                    }
                    if let Some(p) = fm.get("provider").and_then(|v| v.as_str()) {
                        provider = Some(p.to_string());
                    }
                    if let Some(n) = fm.get("name").and_then(|v| v.as_str()) {
                        agent_name = n.to_string();
                    }
                }
                directive = parts[2].trim().to_string();
            } else {
                directive = content;
            }
        } else {
            directive = content;
        }

        let mut npc = crate::npc_compiler::Npc::new(&agent_name, &directive);
        npc.model = model;
        npc.provider = provider;
        npcs.insert(agent_name, npc);
    }
}

fn load_ctx_file(path: &Path) -> Result<TeamCtx> {
    let raw = std::fs::read_to_string(path).map_err(|e| NpcError::FileLoad {
        path: path.display().to_string(),
        source: e,
    })?;

    serde_yaml::from_str(&raw).map_err(|e| NpcError::YamlParse {
        path: path.display().to_string(),
        source: e,
    })
}

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

    pub async fn run(&mut self, input: &str) -> Result<String> {
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
            let response = crate::r#gen::get_genai_response(
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

    pub async fn run(&mut self, input: &str) -> Result<String> {
        self.agent.run(input).await
    }
}

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

    pub fn extract_code_blocks(&self, text: &str) -> Vec<String> {
        let pattern = format!(r"```(?i:{})\s*\n([\s\S]*?)```", regex::escape(&self.language));
        let re = regex::Regex::new(&pattern).unwrap_or_else(|_| {
            regex::Regex::new(r"```\w*\s*\n([\s\S]*?)```").unwrap()
        });
        re.captures_iter(text)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().trim().to_string()))
            .collect()
    }

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

    pub async fn run(&mut self, input: &str) -> Result<String> {
        let mut current_input = input.to_string();
        let mut last_response = String::new();

        for _ in 0..5 {
            last_response = self.agent.run(&current_input).await?;

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

fn register_default_tools(registry: &mut ToolRegistry) {
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
