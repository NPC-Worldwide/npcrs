//! Multi-provider LLM client with tool calling support.
//!
//! Supports OpenAI, Anthropic, Ollama, and any OpenAI-compatible API.
//! This replaces litellm — we talk directly to each provider's HTTP API.

mod client;
mod providers;
mod types;
pub mod cost;
pub mod sanitize;

pub use client::*;
pub use types::*;
