use thiserror::Error;

#[derive(Error, Debug)]
pub enum NpcError {
    #[error("Failed to load file: {path}: {source}")]
    FileLoad {
        path: String,
        source: std::io::Error,
    },

    #[error("YAML parse error in {path}: {source}")]
    YamlParse {
        path: String,
        source: serde_yaml::Error,
    },

    #[error("Template error: {0}")]
    Template(#[from] tera::Error),

    #[error("LLM request failed: {0}")]
    LlmRequest(String),

    #[error("LLM provider '{provider}' not supported")]
    UnsupportedProvider { provider: String },

    #[error("Tool '{name}' not found")]
    ToolNotFound { name: String },

    #[error("NPC '{name}' not found in team")]
    NpcNotFound { name: String },

    #[error("Jinx '{name}' not found")]
    JinxNotFound { name: String },

    #[error("Jinx execution failed in step '{step}': {reason}")]
    JinxExecution { step: String, reason: String },

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("Shell error: {0}")]
    Shell(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("Server error: {0}")]
    Server(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, NpcError>;
