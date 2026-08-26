use crate::error::{NpcError, Result};
use crate::r#gen::response_types::*;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{
    AddBos, ChatTemplateResult, GrammarTriggerType, LlamaChatMessage, LlamaModel,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;

use crate::r#gen::INFERENCE_CANCELLED;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn get_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| LlamaBackend::init().expect("Failed to init llama backend"))
}

pub fn get_llamacpp_response(
    model_path: &str,
    messages: &[Message],
    tools: Option<&[ToolDef]>,
    max_tokens: u32,
    temperature: f32,
    n_ctx: u32,
    n_gpu_layers: i32,
) -> Result<LlmResponse> {
    let backend = get_backend();

    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers as u32);

    let model = LlamaModel::load_from_file(backend, model_path, &model_params).map_err(|e| {
        NpcError::LlmRequest(format!("Failed to load GGUF {}: {:?}", model_path, e))
    })?;

    let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(n_ctx));

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| NpcError::LlmRequest(format!("Context error: {:?}", e)))?;

    // Native path: the model's embedded chat template renders the prompt and,
    // when tool definitions are supplied, produces the tool-call grammar and
    // parser used to constrain and decode the output.
    let (prompt, chat_result) = match render_prompt_with_template(&model, messages, tools) {
        Ok(rendered) => rendered,
        Err(e) => {
            tracing::warn!("chat template render failed ({}), falling back to ChatML", e);
            (format_chatml(messages), None)
        }
    };

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| NpcError::LlmRequest(format!("Tokenize error: {:?}", e)))?;

    let prompt_tokens = tokens.len() as u64;

    // Prompt eval runs in chunks so a host cancel is honored mid-prompt —
    // one monolithic decode of a long prompt would ignore the cancel flag
    // until generation started, which on mobile can be minutes.
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let mut prompt_cancelled = false;
    for (chunk_idx, chunk) in tokens.chunks(64).enumerate() {
        if INFERENCE_CANCELLED.load(Ordering::Relaxed) {
            eprintln!("npcrs llamacpp: inference cancelled during prompt eval");
            prompt_cancelled = true;
            break;
        }
        batch.clear();
        let start = chunk_idx * 64;
        for (j, &token) in chunk.iter().enumerate() {
            let is_last = start + j == tokens.len() - 1;
            batch
                .add(token, (start + j) as i32, &[0], is_last)
                .map_err(|_| NpcError::LlmRequest("Batch add failed".into()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| NpcError::LlmRequest(format!("Decode error: {:?}", e)))?;
    }
    // The generation loop checks the flag before sampling, so a cancelled
    // prompt eval simply yields an empty completion instead of garbage.

    let mut samplers: Vec<LlamaSampler> = Vec::new();
    if let Some(res) = &chat_result {
        if let Some(grammar) = res.grammar.as_deref() {
            if let Some(gs) = build_grammar_sampler(&model, res, grammar) {
                samplers.push(gs);
            }
        }
    }
    samplers.push(LlamaSampler::temp(temperature));
    samplers.push(LlamaSampler::dist(42));
    let mut sampler = LlamaSampler::chain_simple(samplers);

    let mut output_tokens = Vec::new();

    if prompt_cancelled {
        INFERENCE_CANCELLED.store(true, Ordering::Relaxed);
    }

    for n_cur in (tokens.len() as i32..).zip(0..max_tokens).map(|(i, _)| i) {
        if INFERENCE_CANCELLED.load(Ordering::Relaxed) {
            eprintln!("npcrs llamacpp: inference cancelled by host");
            break;
        }
        let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);

        if model.is_eog_token(new_token) {
            break;
        }

        output_tokens.push(new_token);

        batch.clear();
        batch
            .add(new_token, n_cur, &[0], true)
            .map_err(|_| NpcError::LlmRequest("Batch add failed".into()))?;

        ctx.decode(&mut batch)
            .map_err(|e| NpcError::LlmRequest(format!("Decode error: {:?}", e)))?;
    }

    #[allow(deprecated)]
    let mut output_text: String = output_tokens
        .iter()
        .filter_map(|t| {
            model
                .token_to_str(*t, llama_cpp_2::model::Special::Tokenize)
                .ok()
        })
        .collect();

    if let Some(res) = &chat_result {
        for stop in &res.additional_stops {
            if let Some(idx) = output_text.find(stop) {
                output_text.truncate(idx);
            }
        }
    }
    let output_text = output_text.trim().to_string();

    let completion_tokens = output_tokens.len() as u64;

    let message = match &chat_result {
        Some(res) if res.parse_tool_calls => res
            .parse_response_oaicompat(&output_text, false)
            .ok()
            .and_then(|json| serde_json::from_str::<Message>(&json).ok())
            .map(|mut m| {
                if m.role.is_empty() {
                    m.role = "assistant".into();
                }
                m
            })
            .unwrap_or_else(|| Message::assistant(output_text.clone())),
        _ => Message::assistant(output_text.clone()),
    };

    Ok(LlmResponse {
        message,
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cost_usd: 0.0,
        }),
        model: model_path.to_string(),
        finish_reason: Some("stop".to_string()),
        cost_usd: Some(0.0),
    })
}

/// Render messages (and optional tool definitions) with the model's embedded
/// chat template. Returns the prompt plus the template result carrying the
/// tool grammar/parser for sampling and response parsing.
fn render_prompt_with_template(
    model: &LlamaModel,
    messages: &[Message],
    tools: Option<&[ToolDef]>,
) -> std::result::Result<(String, Option<ChatTemplateResult>), String> {
    let tmpl = model
        .chat_template(None)
        .map_err(|e| format!("no embedded chat template: {:?}", e))?;

    let chat: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| {
            let content = if m.role == "assistant" && m.tool_calls.is_some() {
                // Re-encode native tool calls so templates see the tool-call turn.
                serde_json::to_string(&m.tool_calls).unwrap_or_default()
            } else {
                m.content.clone().unwrap_or_default()
            };
            LlamaChatMessage::new(m.role.clone(), content)
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| format!("invalid chat message: {:?}", e))?;

    let tools_json = match tools {
        Some(t) if !t.is_empty() => {
            Some(serde_json::to_string(t).map_err(|e| format!("tools json: {}", e))?)
        }
        _ => None,
    };

    let result = model
        .apply_chat_template_with_tools_oaicompat(
            &tmpl,
            &chat,
            tools_json.as_deref(),
            None,
            true,
        )
        .map_err(|e| format!("apply_chat_template: {:?}", e))?;

    Ok((result.prompt.clone(), Some(result)))
}

/// Build the grammar sampler that constrains tool-call output, honoring the
/// template's lazy-grammar triggers when present.
fn build_grammar_sampler(
    model: &LlamaModel,
    res: &ChatTemplateResult,
    grammar: &str,
) -> Option<LlamaSampler> {
    let words: Vec<String> = res
        .grammar_triggers
        .iter()
        .filter(|t| t.trigger_type == GrammarTriggerType::Word)
        .map(|t| t.value.clone())
        .collect();
    let token_triggers: Vec<LlamaToken> =
        res.grammar_triggers.iter().filter_map(|t| t.token).collect();
    let patterns: Vec<String> = res
        .grammar_triggers
        .iter()
        .filter(|t| {
            matches!(
                t.trigger_type,
                GrammarTriggerType::Pattern | GrammarTriggerType::PatternFull
            )
        })
        .map(|t| t.value.clone())
        .collect();

    let sampler = if !patterns.is_empty() {
        LlamaSampler::grammar_lazy_patterns(model, grammar, "root", &patterns, &token_triggers)
    } else if res.grammar_lazy || !words.is_empty() || !token_triggers.is_empty() {
        LlamaSampler::grammar_lazy(model, grammar, "root", words, &token_triggers)
    } else {
        LlamaSampler::grammar(model, grammar, "root")
    };
    sampler.ok()
}

/// Legacy fallback for models without an embedded chat template.
fn format_chatml(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let content = msg.content.as_deref().unwrap_or("");
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}
