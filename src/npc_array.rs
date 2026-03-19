
use crate::error::Result;
use crate::r#gen::Message;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct InferResult {
    pub model: String,
    pub provider: String,
    pub response: String,
    pub tokens: u64,
    pub cost: f64,
    pub latency_ms: u64,
}

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

        let result = crate::r#gen::get_genai_response(provider, model, &messages, None, None)
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

    results
        .iter()
        .find(|r| r.response.trim().to_lowercase() == best)
        .map(|r| r.response.clone())
        .unwrap_or_default()
}

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
