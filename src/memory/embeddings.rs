
use crate::error::{NpcError, Result};
use reqwest::Client;

pub async fn get_embeddings(
    text: &str,
    model: &str,
    provider: &str,
    api_key: Option<&str>,
) -> Result<Vec<f64>> {
    let client = Client::new();

    match provider {
        "ollama" => get_embeddings_ollama(&client, text, model).await,
        "openai" => {
            let key = api_key
                .or_else(|| std::env::var("OPENAI_API_KEY").ok().as_deref().map(|_| ""))
                .ok_or_else(|| {
                    NpcError::LlmRequest("OPENAI_API_KEY not set for embeddings".to_string())
                })?;
            let key = api_key
                .map(|s| s.to_string())
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .unwrap_or_default();
            get_embeddings_openai(&client, text, model, &key).await
        }
        other => Err(NpcError::UnsupportedProvider {
            provider: other.to_string(),
        }),
    }
}

async fn get_embeddings_ollama(client: &Client, text: &str, model: &str) -> Result<Vec<f64>> {
    let base_url =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());

    let url = format!("{}/api/embeddings", base_url);

    let body = serde_json::json!({
        "model": model,
        "prompt": text,
    });

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()
        .map_err(|e| NpcError::LlmRequest(format!("Ollama embeddings request failed: {}", e)))?;

    let json: serde_json::Value = resp.json().await?;

    let embedding = json
        .get("embedding")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            NpcError::LlmRequest("Ollama embeddings response missing 'embedding' field".into())
        })?;

    let vec: Vec<f64> = embedding
        .iter()
        .filter_map(|v| v.as_f64())
        .collect();

    if vec.is_empty() {
        return Err(NpcError::LlmRequest(
            "Ollama returned empty embedding".into(),
        ));
    }

    Ok(vec)
}

async fn get_embeddings_openai(
    client: &Client,
    text: &str,
    model: &str,
    api_key: &str,
) -> Result<Vec<f64>> {
    let url = "https://api.openai.com/v1/embeddings";

    let body = serde_json::json!({
        "model": model,
        "input": text,
    });

    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&body)
        .send()
        .await?
        .error_for_status()
        .map_err(|e| NpcError::LlmRequest(format!("OpenAI embeddings request failed: {}", e)))?;

    let json: serde_json::Value = resp.json().await?;

    let embedding = json
        .get("data")
        .and_then(|d| d.get(0))
        .and_then(|d| d.get("embedding"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            NpcError::LlmRequest(
                "OpenAI embeddings response missing 'data[0].embedding' field".into(),
            )
        })?;

    let vec: Vec<f64> = embedding
        .iter()
        .filter_map(|v| v.as_f64())
        .collect();

    if vec.is_empty() {
        return Err(NpcError::LlmRequest(
            "OpenAI returned empty embedding".into(),
        ));
    }

    Ok(vec)
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }
}
