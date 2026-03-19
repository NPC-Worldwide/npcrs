
use crate::error::{NpcError, Result};

pub async fn get_ollama_embeddings(text: &str, model: &str) -> Result<Vec<f32>> {
    let api_url = std::env::var("OLLAMA_API_URL")
        .unwrap_or_else(|_| "http://localhost:11434".into());
    let url = format!("{}/api/embeddings", api_url);

    let body = serde_json::json!({
        "model": model,
        "prompt": text,
    });

    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&body).send().await?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(NpcError::LlmRequest(format!("Ollama embeddings failed: {}", text)));
    }

    let json: serde_json::Value = resp.json().await?;
    let embedding = json.get("embedding")
        .and_then(|e| e.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
        .unwrap_or_default();

    Ok(embedding)
}

pub async fn get_openai_embeddings(text: &str, model: &str, api_key: Option<&str>) -> Result<Vec<f32>> {
    let key = api_key
        .map(String::from)
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or_else(|| NpcError::LlmRequest("OPENAI_API_KEY not set".into()))?;

    let body = serde_json::json!({
        "model": model,
        "input": text,
    });

    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.openai.com/v1/embeddings")
        .header("Authorization", format!("Bearer {}", key))
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(NpcError::LlmRequest(format!("OpenAI embeddings failed: {}", text)));
    }

    let json: serde_json::Value = resp.json().await?;
    let embedding = json["data"][0]["embedding"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
        .unwrap_or_default();

    Ok(embedding)
}

pub async fn get_embeddings(text: &str, model: &str, provider: &str, api_key: Option<&str>) -> Result<Vec<f32>> {
    match provider {
        "ollama" => get_ollama_embeddings(text, model).await,
        "openai" => get_openai_embeddings(text, model, api_key).await,
        _ => Err(NpcError::UnsupportedProvider { provider: provider.to_string() }),
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}
