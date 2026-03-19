
use crate::error::{NpcError, Result};

pub struct GeneratedImage {
    pub data: Vec<u8>,
    pub format: String, // "png", "jpeg"
    pub revised_prompt: Option<String>,
}

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
        "ollama" => generate_image_ollama(prompt, model, width, height, None).await,
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

pub async fn generate_image_ollama(
    prompt: &str,
    model: &str,
    width: u32,
    height: u32,
    api_url: Option<&str>,
) -> Result<GeneratedImage> {
    let base_url = api_url
        .map(String::from)
        .or_else(|| std::env::var("OLLAMA_API_URL").ok())
        .unwrap_or_else(|| "http://localhost:11434".into());
    let endpoint = format!("{}/api/generate", base_url);

    let mut options = serde_json::Map::new();
    options.insert("width".into(), serde_json::json!(width));
    options.insert("height".into(), serde_json::json!(height));

    let payload = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": options,
    });

    let client = reqwest::Client::new();
    let resp = client.post(&endpoint).json(&payload).send().await?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(NpcError::LlmRequest(format!(
            "Ollama image gen failed: {}. Make sure '{}' is pulled (`ollama pull {}`)",
            text, model, model
        )));
    }

    let json: serde_json::Value = resp.json().await?;

    let b64 = json.get("image").and_then(|v| v.as_str())
        .or_else(|| json.get("images").and_then(|v| v.as_array()).and_then(|a| a.first()).and_then(|v| v.as_str()));

    if let Some(b64) = b64 {
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| NpcError::LlmRequest(format!("Base64 decode: {}", e)))?;
        Ok(GeneratedImage { data, format: "png".into(), revised_prompt: None })
    } else {
        Err(NpcError::LlmRequest(format!(
            "No images returned from Ollama. Make sure '{}' is an image generation model.",
            model
        )))
    }
}

pub async fn edit_image(
    prompt: &str,
    image_path: &str,
    provider: &str,
    model: Option<&str>,
    width: u32,
    height: u32,
    api_key: Option<&str>,
) -> Result<GeneratedImage> {
    let model = model.unwrap_or(match provider {
        "openai" => "gpt-image-1",
        "gemini" => "gemini-2.5-flash-image",
        _ => "gpt-image-1",
    });

    let image_bytes = std::fs::read(image_path)
        .map_err(|e| NpcError::Generation(format!("Failed to read image {}: {}", image_path, e)))?;

    use base64::Engine;
    let b64_image = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

    match provider {
        "openai" => {
            let key = api_key
                .map(String::from)
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .ok_or_else(|| NpcError::LlmRequest("OPENAI_API_KEY not set".into()))?;

            let client = reqwest::Client::new();
            let form = reqwest::multipart::Form::new()
                .text("model", model.to_string())
                .text("prompt", prompt.to_string())
                .text("size", format!("{}x{}", width, height))
                .text("response_format", "b64_json")
                .part("image", reqwest::multipart::Part::bytes(image_bytes).file_name("image.png").mime_str("image/png").unwrap());

            let resp = client
                .post("https://api.openai.com/v1/images/edits")
                .header("Authorization", format!("Bearer {}", key))
                .multipart(form)
                .send()
                .await?;

            if !resp.status().is_success() {
                let text = resp.text().await.unwrap_or_default();
                return Err(NpcError::LlmRequest(format!("Image edit failed: {}", text)));
            }

            let json: serde_json::Value = resp.json().await?;
            let b64 = json["data"][0]["b64_json"].as_str()
                .ok_or_else(|| NpcError::LlmRequest("No image data in edit response".into()))?;
            let data = base64::engine::general_purpose::STANDARD.decode(b64)
                .map_err(|e| NpcError::LlmRequest(format!("Base64 decode: {}", e)))?;
            Ok(GeneratedImage { data, format: "png".into(), revised_prompt: None })
        }
        "gemini" => {
            let key = api_key
                .map(String::from)
                .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
                .ok_or_else(|| NpcError::LlmRequest("GOOGLE_API_KEY not set".into()))?;

            let mime = if image_path.ends_with(".jpg") || image_path.ends_with(".jpeg") {
                "image/jpeg"
            } else {
                "image/png"
            };

            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                model, key
            );
            let body = serde_json::json!({
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime, "data": b64_image}}
                    ]
                }],
                "generationConfig": {"responseModalities": ["image"]}
            });

            let client = reqwest::Client::new();
            let resp = client.post(&url).json(&body).send().await?;
            if !resp.status().is_success() {
                let text = resp.text().await.unwrap_or_default();
                return Err(NpcError::LlmRequest(format!("Gemini image edit failed: {}", text)));
            }

            let json: serde_json::Value = resp.json().await?;
            if let Some(b64) = json["candidates"][0]["content"]["parts"].as_array()
                .and_then(|parts| parts.iter().find(|p| p.get("inlineData").is_some()))
                .and_then(|p| p["inlineData"]["data"].as_str())
            {
                let data = base64::engine::general_purpose::STANDARD.decode(b64)
                    .map_err(|e| NpcError::LlmRequest(format!("Base64: {}", e)))?;
                Ok(GeneratedImage { data, format: "png".into(), revised_prompt: None })
            } else {
                Err(NpcError::LlmRequest("No image in Gemini edit response".into()))
            }
        }
        _ => Err(NpcError::UnsupportedProvider { provider: provider.to_string() }),
    }
}
