//! Vectorized operations over model populations.
//!
//! Mirrors `npcpy.npc_array` — run the same prompt across multiple models
//! and collect/aggregate results. Useful for ensembling, benchmarking, and
//! model comparison.

use crate::error::Result;
use crate::r#gen::Message;
use std::collections::HashMap;
use std::time::Instant;

/// Result from a single model inference within an infer_matrix call.
#[derive(Debug, Clone)]
pub struct InferResult {
    /// Model name that was used.
    pub model: String,
    /// Provider that was used.
    pub provider: String,
    /// The text response from the model.
    pub response: String,
    /// Total tokens used (prompt + completion).
    pub tokens: u64,
    /// Estimated cost in USD.
    pub cost: f64,
    /// Latency in milliseconds.
    pub latency_ms: u64,
}

/// Run the same prompt across multiple models and collect results.
///
/// Each `(model, provider)` pair is called sequentially (the LlmClient is
/// borrowed, not cloneable). Failed calls are captured in the response field
/// rather than aborting the whole matrix.
///
/// # Arguments
/// * `client` — The LLM client to use.
/// * `prompt` — The user prompt to send to each model.
/// * `models` — Slice of (model, provider) pairs.
/// * `system_prompt` — Optional system prompt to prepend.
pub async fn infer_matrix(
    
    prompt: &str,
    models: &[(String, String)],
    system_prompt: Option<&str>,
) -> Result<Vec<InferResult>> {
    let mut results = Vec::with_capacity(models.len());

    for (model, provider) in models {
        let start = Instant::now();

        let mut messages = Vec::new();
        if let Some(s) = system_prompt {
            messages.push(Message::system(s));
        }
        messages.push(Message::user(prompt));

        let result = client
            .crate::llm_funcs::get_llm_response(provider, model, &messages, None, None)
            .await;

        let elapsed = start.elapsed().as_millis() as u64;

        match result {
            Ok(resp) => {
                let text = resp.message.content.unwrap_or_default();
                let (tokens, cost) = if let Some(ref usage) = resp.usage {
                    let total = usage.total_tokens;
                    let c = crate::r#gen::cost::calculate_cost(
                        model,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                    );
                    (total, c)
                } else {
                    (0, 0.0)
                };

                results.push(InferResult {
                    model: model.clone(),
                    provider: provider.clone(),
                    response: text,
                    tokens,
                    cost,
                    latency_ms: elapsed,
                });
            }
            Err(e) => {
                results.push(InferResult {
                    model: model.clone(),
                    provider: provider.clone(),
                    response: format!("[ERROR] {e}"),
                    tokens: 0,
                    cost: 0.0,
                    latency_ms: elapsed,
                });
            }
        }
    }

    Ok(results)
}

/// Simple majority vote across inference results.
///
/// Counts the frequency of each response (trimmed, case-insensitive) and
/// returns the most common one. Ties are broken by first occurrence.
/// Error responses (starting with "[ERROR]") are skipped.
pub fn ensemble_vote(results: &[InferResult]) -> String {
    if results.is_empty() {
        return String::new();
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    for r in results {
        if r.response.starts_with("[ERROR]") {
            continue;
        }

        let key = r.response.trim().to_lowercase();
        let count = counts.entry(key.clone()).or_insert(0);
        if *count == 0 {
            order.push(key);
        }
        *count += 1;
    }

    if order.is_empty() {
        return results[0].response.clone();
    }

    let best = order
        .into_iter()
        .max_by_key(|k| *counts.get(k).unwrap_or(&0))
        .unwrap_or_default();

    // Return the original (non-lowercased) response that matches
    results
        .iter()
        .find(|r| r.response.trim().to_lowercase() == best)
        .map(|r| r.response.clone())
        .unwrap_or_default()
}

/// Summary statistics for an infer_matrix run.
#[derive(Debug, Clone)]
pub struct MatrixStats {
    pub total_models: usize,
    pub successful: usize,
    pub failed: usize,
    pub total_tokens: u64,
    pub total_cost: f64,
    pub avg_latency_ms: f64,
    pub min_latency_ms: u64,
    pub max_latency_ms: u64,
}

/// Compute summary statistics over a set of inference results.
pub fn matrix_stats(results: &[InferResult]) -> MatrixStats {
    let successful = results
        .iter()
        .filter(|r| !r.response.starts_with("[ERROR]"))
        .count();
    let failed = results.len() - successful;
    let total_tokens: u64 = results.iter().map(|r| r.tokens).sum();
    let total_cost: f64 = results.iter().map(|r| r.cost).sum();
    let latencies: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
    let avg_latency = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<u64>() as f64 / latencies.len() as f64
    };

    MatrixStats {
        total_models: results.len(),
        successful,
        failed,
        total_tokens,
        total_cost,
        avg_latency_ms: avg_latency,
        min_latency_ms: latencies.iter().copied().min().unwrap_or(0),
        max_latency_ms: latencies.iter().copied().max().unwrap_or(0),
    }
}
