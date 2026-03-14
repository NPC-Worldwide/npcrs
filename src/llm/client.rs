use crate::error::{NpcError, Result};
use crate::llm::providers;
use crate::llm::types::*;
use reqwest::Client;
use std::collections::HashMap;

/// Multi-provider LLM client.
///
/// Dispatches to the right API format based on provider name.
/// Supports: openai, anthropic, ollama, and any OpenAI-compatible endpoint.
pub struct LlmClient {
    http: Client,
    /// Provider name → config (base_url, api_key).
    providers: HashMap<String, ProviderConfig>,
}

impl LlmClient {
    /// Create a new LLM client, auto-detecting configured providers from env.
    pub fn from_env() -> Self {
        let http = Client::new();
        let mut providers_map = HashMap::new();

        // OpenAI
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            providers_map.insert(
                "openai".to_string(),
                ProviderConfig {
                    base_url: std::env::var("OPENAI_API_BASE")
                        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
                    api_key: Some(key),
                },
            );
        }

        // Anthropic
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            providers_map.insert(
                "anthropic".to_string(),
                ProviderConfig {
                    base_url: "https://api.anthropic.com".to_string(),
                    api_key: Some(key),
                },
            );
        }

        // Ollama (no key needed)
        providers_map.insert(
            "ollama".to_string(),
            ProviderConfig {
                base_url: std::env::var("OLLAMA_HOST")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string()),
                api_key: None,
            },
        );

        // Google / Gemini
        if let Ok(key) = std::env::var("GOOGLE_API_KEY") {
            providers_map.insert(
                "google".to_string(),
                ProviderConfig {
                    base_url: "https://generativelanguage.googleapis.com".to_string(),
                    api_key: Some(key),
                },
            );
        }

        Self {
            http,
            providers: providers_map,
        }
    }

    /// Create with explicit provider configs.
    pub fn new(providers: HashMap<String, ProviderConfig>) -> Self {
        Self {
            http: Client::new(),
            providers,
        }
    }

    /// Register or update a provider config.
    pub fn set_provider(&mut self, name: impl Into<String>, config: ProviderConfig) {
        self.providers.insert(name.into(), config);
    }

    /// Send a chat completion request.
    pub async fn chat_completion(
        &self,
        provider: &str,
        model: &str,
        messages: &[Message],
        tools: Option<&[ToolDef]>,
        api_url_override: Option<&str>,
    ) -> Result<LlmResponse> {
        let config = self.providers.get(provider).ok_or_else(|| {
            NpcError::UnsupportedProvider {
                provider: provider.to_string(),
            }
        })?;

        let base_url = api_url_override.unwrap_or(&config.base_url);

        match provider {
            "anthropic" => {
                providers::anthropic::chat_completion(
                    &self.http,
                    base_url,
                    config.api_key.as_deref(),
                    model,
                    messages,
                    tools,
                )
                .await
            }
            "ollama" => {
                providers::openai_compat::chat_completion(
                    &self.http,
                    &format!("{}/v1", base_url),
                    None,
                    model,
                    messages,
                    tools,
                )
                .await
            }
            // openai and any openai-compatible provider
            _ => {
                providers::openai_compat::chat_completion(
                    &self.http,
                    base_url,
                    config.api_key.as_deref(),
                    model,
                    messages,
                    tools,
                )
                .await
            }
        }
    }
}
