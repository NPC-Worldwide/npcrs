//! Image generation dispatching to different providers.

use crate::error::{NpcError, Result};

/// A generated image.
pub struct GeneratedImage {
    pub data: Vec<u8>,
    pub format: String, // "png", "jpeg"
    pub revised_prompt: Option<String>,
}

/// Generate an image using the specified provider.
pub async fn generate_image(
    prompt: &str,
    model: &str,
    provider: &str,
    api_key: Option<&str>,
    width: u32,
    height: u32,
) -> Result<GeneratedImage> {
    match provider {
        "openai" => generate_image_openai(prompt, model, api_key, width, height).await,
        "google" | "gemini" => generate_image_gemini(prompt, model, api_key).await,
        _ => Err(NpcError::UnsupportedProvider {
            provider: provider.to_string(),
        }),
    }
}

async fn generate_image_openai(
    prompt: &str,
    model: &str,
    api_key: Option<&str>,
    width: u32,
    height: u32,
) -> Result<GeneratedImage> {
    let client = reqwest::Client::new();
    let key = api_key
        .map(String::from)
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or_else(|| NpcError::LlmRequest("OPENAI_API_KEY not set".into()))?;

    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": format!("{}x{}", width, height),
        "response_format": "b64_json",
    });

    let resp = client
        .post("https://api.openai.com/v1/images/generations")
        .header("Authorization", format!("Bearer {}", key))
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(NpcError::LlmRequest(format!("Image gen failed: {}", text)));
    }

    let json: serde_json::Value = resp.json().await?;
    let b64 = json["data"][0]["b64_json"]
        .as_str()
        .ok_or_else(|| NpcError::LlmRequest("No image data".into()))?;

    use base64::Engine;
    let data = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| NpcError::LlmRequest(format!("Base64 decode: {}", e)))?;

    Ok(GeneratedImage {
        data,
        format: "png".into(),
        revised_prompt: json["data"][0]["revised_prompt"]
            .as_str()
            .map(String::from),
    })
}

async fn generate_image_gemini(
    prompt: &str,
    model: &str,
    api_key: Option<&str>,
) -> Result<GeneratedImage> {
    let key = api_key
        .map(String::from)
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .ok_or_else(|| NpcError::LlmRequest("GOOGLE_API_KEY not set".into()))?;

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, key
    );

    let body = serde_json::json!({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["image"]}
    });

    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&body).send().await?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(NpcError::LlmRequest(format!(
            "Gemini image gen failed: {}",
            text
        )));
    }

    let json: serde_json::Value = resp.json().await?;
    let parts = &json["candidates"][0]["content"]["parts"];
    if let Some(arr) = parts.as_array() {
        for part in arr {
            if let Some(inline) = part.get("inlineData") {
                if let Some(b64) = inline["data"].as_str() {
                    use base64::Engine;
                    let data = base64::engine::general_purpose::STANDARD
                        .decode(b64)
                        .map_err(|e| NpcError::LlmRequest(format!("Base64: {}", e)))?;
                    let mime = inline["mimeType"].as_str().unwrap_or("image/png");
                    let format = if mime.contains("jpeg") {
                        "jpeg"
                    } else {
                        "png"
                    };
                    return Ok(GeneratedImage {
                        data,
                        format: format.into(),
                        revised_prompt: None,
                    });
                }
            }
        }
    }

    Err(NpcError::LlmRequest("No image in Gemini response".into()))
}
