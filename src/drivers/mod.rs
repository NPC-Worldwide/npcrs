//! Drivers — LLM providers and external tools as kernel device drivers.
//!
//! Just like Linux has device drivers for hardware, npcrs has drivers for:
//! - LLM providers (OpenAI, Anthropic, Ollama) → "compute devices"
//! - MCP servers → "peripheral devices"
//! - Python runtime → "coprocessor"
//! - Web search → "network device"

use crate::llm::LlmClient;

/// Manages all kernel drivers.
pub struct DriverManager {
    /// LLM provider driver (the main "compute device").
    llm_client: LlmClient,
    // Future: MCP driver pool, Python runtime driver, etc.
}

impl DriverManager {
    /// Initialize drivers from environment variables.
    pub fn from_env() -> Self {
        Self {
            llm_client: LlmClient::from_env(),
        }
    }

    /// Get the LLM driver.
    pub fn llm(&self) -> &LlmClient {
        &self.llm_client
    }

    /// Get a mutable reference to the LLM driver.
    pub fn llm_mut(&mut self) -> &mut LlmClient {
        &mut self.llm_client
    }
}
