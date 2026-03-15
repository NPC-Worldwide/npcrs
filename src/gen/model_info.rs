//! Model-to-provider resolution.
//!
//! Given a model name, infers which LLM provider to use.
//! Checks NPCSH_CHAT_MODEL / NPCSH_CHAT_PROVIDER env vars as defaults.

/// Default model when nothing is configured.
pub const DEFAULT_MODEL: &str = "llama3.2";
/// Default provider when nothing is configured.
pub const DEFAULT_PROVIDER: &str = "ollama";

/// Given a model name string, resolve the (model, provider) pair.
///
/// Rules:
/// - starts with "gpt-", "o1", "o3", "o4", "dall-e" -> "openai"
/// - starts with "claude-" -> "anthropic"
/// - starts with "gemini-" -> "google"
/// - starts with "llama", "mixtral", "deepseek", "mistral", "phi", "qwen" -> "ollama"
/// - starts with "grok" -> "xai"
/// - Falls back to env vars NPCSH_CHAT_MODEL / NPCSH_CHAT_PROVIDER
/// - Ultimate default: ("llama3.2", "ollama")
pub fn resolve_model_provider(model: &str) -> (String, String) {
    let provider = infer_provider(model);
    (model.to_string(), provider)
}

/// Infer provider from model name alone.
pub fn infer_provider(model: &str) -> String {
    let m = model.to_lowercase();

    if m.starts_with("gpt-")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.starts_with("dall-e")
    {
        return "openai".to_string();
    }

    if m.starts_with("claude-") {
        return "anthropic".to_string();
    }

    if m.starts_with("gemini-") {
        return "google".to_string();
    }

    if m.starts_with("grok") {
        return "xai".to_string();
    }

    // Open-source models typically served by Ollama
    if m.starts_with("llama")
        || m.starts_with("mixtral")
        || m.starts_with("deepseek")
        || m.starts_with("mistral")
        || m.starts_with("phi")
        || m.starts_with("qwen")
        || m.starts_with("codellama")
        || m.starts_with("vicuna")
        || m.starts_with("solar")
        || m.starts_with("yi-")
        || m.starts_with("command-r")
    {
        return "ollama".to_string();
    }

    // Check env var fallback
    if let Ok(provider) = std::env::var("NPCSH_CHAT_PROVIDER") {
        if !provider.is_empty() {
            return provider;
        }
    }

    DEFAULT_PROVIDER.to_string()
}

/// Get the default model from env or hardcoded default.
pub fn default_model() -> String {
    std::env::var("NPCSH_CHAT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

/// Get the default provider from env or hardcoded default.
pub fn default_provider() -> String {
    std::env::var("NPCSH_CHAT_PROVIDER").unwrap_or_else(|_| DEFAULT_PROVIDER.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_models() {
        assert_eq!(infer_provider("gpt-4o"), "openai");
        assert_eq!(infer_provider("gpt-4o-mini"), "openai");
        assert_eq!(infer_provider("o1"), "openai");
        assert_eq!(infer_provider("o3-mini"), "openai");
        assert_eq!(infer_provider("o4-mini"), "openai");
    }

    #[test]
    fn anthropic_models() {
        assert_eq!(infer_provider("claude-3-5-sonnet"), "anthropic");
        assert_eq!(infer_provider("claude-sonnet-4"), "anthropic");
        assert_eq!(infer_provider("claude-opus-4"), "anthropic");
    }

    #[test]
    fn google_models() {
        assert_eq!(infer_provider("gemini-2.0-flash"), "google");
        assert_eq!(infer_provider("gemini-2.5-pro"), "google");
    }

    #[test]
    fn ollama_models() {
        assert_eq!(infer_provider("llama3.2"), "ollama");
        assert_eq!(infer_provider("mixtral"), "ollama");
        assert_eq!(infer_provider("deepseek-r1"), "ollama");
        assert_eq!(infer_provider("mistral-small"), "ollama");
        assert_eq!(infer_provider("phi"), "ollama");
        assert_eq!(infer_provider("qwen2"), "ollama");
    }

    #[test]
    fn resolve_pair() {
        let (model, provider) = resolve_model_provider("gpt-4o");
        assert_eq!(model, "gpt-4o");
        assert_eq!(provider, "openai");
    }
}
