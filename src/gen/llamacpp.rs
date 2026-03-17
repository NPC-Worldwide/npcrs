//! Local GGUF inference via llama.cpp — mirrors npcpy's get_llamacpp_response().

use crate::error::{NpcError, Result};
use crate::r#gen::response_types::*;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::sync::OnceLock;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn get_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| LlamaBackend::init().expect("Failed to init llama backend"))
}

/// Load a GGUF model and run chat completion — mirrors npcpy's get_llamacpp_response().
pub fn get_llamacpp_response(
    model_path: &str,
    messages: &[Message],
    max_tokens: u32,
    temperature: f32,
    n_ctx: u32,
    n_gpu_layers: i32,
) -> Result<LlmResponse> {
    let backend = get_backend();

    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(n_gpu_layers as u32);

    let model = LlamaModel::load_from_file(backend, model_path, &model_params)
        .map_err(|e| NpcError::LlmRequest(format!("Failed to load GGUF {}: {:?}", model_path, e)))?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(n_ctx));

    let mut ctx = model.new_context(backend, ctx_params)
        .map_err(|e| NpcError::LlmRequest(format!("Context error: {:?}", e)))?;

    // Build prompt from messages (ChatML format)
    let prompt = format_chatml(messages);

    // Tokenize
    let tokens = model.str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
        .map_err(|e| NpcError::LlmRequest(format!("Tokenize error: {:?}", e)))?;

    let prompt_tokens = tokens.len() as u64;

    // Create batch and decode prompt
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    for (i, &token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch.add(token, i as i32, &[0], is_last)
            .map_err(|_| NpcError::LlmRequest("Batch add failed".into()))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| NpcError::LlmRequest(format!("Decode error: {:?}", e)))?;

    // Set up sampler with temperature
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(temperature),
        LlamaSampler::dist(42),
    ]);

    // Generate tokens
    let mut output_tokens = Vec::new();
    let mut n_cur = tokens.len() as i32;

    for _ in 0..max_tokens {
        let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);

        if model.is_eog_token(new_token) {
            break;
        }

        output_tokens.push(new_token);

        batch.clear();
        batch.add(new_token, n_cur, &[0], true)
            .map_err(|_| NpcError::LlmRequest("Batch add failed".into()))?;
        n_cur += 1;

        ctx.decode(&mut batch)
            .map_err(|e| NpcError::LlmRequest(format!("Decode error: {:?}", e)))?;
    }

    // Detokenize
    let output_text: String = output_tokens.iter()
        .filter_map(|t| model.token_to_str(*t, llama_cpp_2::model::Special::Tokenize).ok())
        .collect();

    let completion_tokens = output_tokens.len() as u64;

    Ok(LlmResponse {
        message: Message::assistant(output_text.trim()),
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }),
        model: model_path.to_string(),
        finish_reason: Some("stop".to_string()),
        cost_usd: Some(0.0),
    })
}

fn format_chatml(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let content = msg.content.as_deref().unwrap_or("");
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, content));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}
