//! Audio utilities — mirrors npcpy.data.audio

use crate::error::{NpcError, Result};
use std::collections::HashMap;

pub async fn speech_to_text(audio_data: &[u8], engine: &str, language: Option<&str>) -> Result<HashMap<String, serde_json::Value>> {
    match engine.to_lowercase().as_str() {
        "whisper" | "faster-whisper" => stt_whisper(audio_data, "base", language),
        "openai" => stt_openai(audio_data, None, "whisper-1", language).await,
        "gemini" => stt_gemini(audio_data, None, "gemini-1.5-flash", language).await,
        "groq" => stt_groq(audio_data, None, "whisper-large-v3", language).await,
        other => Err(NpcError::Shell(format!("Unknown STT engine: {}", other))),
    }
}

/// Local whisper STT — tries Groq API (free whisper endpoint) as a pure-Rust fallback.
/// For true local inference, use the llamacpp feature with a whisper GGUF model.
pub fn stt_whisper(audio_data: &[u8], _model_size: &str, language: Option<&str>) -> Result<HashMap<String, serde_json::Value>> {
    // Use tokio runtime to call the async Groq/OpenAI API
    let rt = tokio::runtime::Handle::try_current()
        .or_else(|_| {
            tokio::runtime::Runtime::new().map(|rt| rt.handle().clone())
        })
        .map_err(|e| NpcError::Other(format!("No tokio runtime: {}", e)))?;

    let data = audio_data.to_vec();
    let lang = language.map(String::from);

    // Try Groq first (free whisper), then OpenAI
    rt.block_on(async {
        if std::env::var("GROQ_API_KEY").is_ok() {
            return stt_groq(&data, None, "whisper-large-v3", lang.as_deref()).await;
        }
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return stt_openai(&data, None, "whisper-1", lang.as_deref()).await;
        }
        Err(NpcError::LlmRequest("No STT API key available. Set GROQ_API_KEY or OPENAI_API_KEY.".into()))
    })
}

pub async fn stt_openai(audio_data: &[u8], api_key: Option<&str>, model: &str, language: Option<&str>) -> Result<HashMap<String, serde_json::Value>> {
    let key = api_key.map(String::from).or_else(|| std::env::var("OPENAI_API_KEY").ok()).ok_or_else(|| NpcError::LlmRequest("OPENAI_API_KEY not set".into()))?;
    let file_part = reqwest::multipart::Part::bytes(audio_data.to_vec()).file_name("audio.wav").mime_str("audio/wav").map_err(|e| NpcError::LlmRequest(format!("MIME: {}", e)))?;
    let mut form = reqwest::multipart::Form::new().part("file", file_part).text("model", model.to_string()).text("response_format", "verbose_json".to_string());
    if let Some(l) = language { form = form.text("language", l.to_string()); }
    let resp = reqwest::Client::new().post("https://api.openai.com/v1/audio/transcriptions").header("Authorization", format!("Bearer {}", key)).multipart(form).send().await?;
    if !resp.status().is_success() { return Err(NpcError::LlmRequest(format!("OpenAI STT: {}", resp.text().await.unwrap_or_default()))); }
    let json: serde_json::Value = resp.json().await?;
    let mut r = HashMap::new();
    r.insert("text".into(), json.get("text").cloned().unwrap_or(serde_json::Value::String(String::new())));
    r.insert("language".into(), json.get("language").cloned().unwrap_or(serde_json::Value::String("en".into())));
    Ok(r)
}

pub async fn stt_gemini(audio_data: &[u8], api_key: Option<&str>, model: &str, language: Option<&str>) -> Result<HashMap<String, serde_json::Value>> {
    let key = api_key.map(String::from).or_else(|| std::env::var("GOOGLE_API_KEY").ok()).or_else(|| std::env::var("GEMINI_API_KEY").ok()).ok_or_else(|| NpcError::LlmRequest("GOOGLE_API_KEY not set".into()))?;
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(audio_data);
    let prompt = language.map(|l| format!("Transcribe in {}. Output only transcription.", l)).unwrap_or_else(|| "Transcribe exactly. Output only transcription.".into());
    let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", model, key);
    let body = serde_json::json!({"contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "audio/wav", "data": b64}}]}]});
    let resp = reqwest::Client::new().post(&url).json(&body).send().await?;
    if !resp.status().is_success() { return Err(NpcError::LlmRequest(format!("Gemini STT: {}", resp.text().await.unwrap_or_default()))); }
    let json: serde_json::Value = resp.json().await?;
    let text = json["candidates"][0]["content"]["parts"][0]["text"].as_str().unwrap_or("").trim().to_string();
    let mut r = HashMap::new(); r.insert("text".into(), serde_json::Value::String(text)); Ok(r)
}

pub async fn stt_groq(audio_data: &[u8], api_key: Option<&str>, model: &str, language: Option<&str>) -> Result<HashMap<String, serde_json::Value>> {
    let key = api_key.map(String::from).or_else(|| std::env::var("GROQ_API_KEY").ok()).ok_or_else(|| NpcError::LlmRequest("GROQ_API_KEY not set".into()))?;
    let file_part = reqwest::multipart::Part::bytes(audio_data.to_vec()).file_name("audio.wav").mime_str("audio/wav").map_err(|e| NpcError::LlmRequest(format!("MIME: {}", e)))?;
    let mut form = reqwest::multipart::Form::new().part("file", file_part).text("model", model.to_string());
    if let Some(l) = language { form = form.text("language", l.to_string()); }
    let resp = reqwest::Client::new().post("https://api.groq.com/openai/v1/audio/transcriptions").header("Authorization", format!("Bearer {}", key)).multipart(form).send().await?;
    if !resp.status().is_success() { return Err(NpcError::LlmRequest(format!("Groq STT: {}", resp.text().await.unwrap_or_default()))); }
    let json: serde_json::Value = resp.json().await?;
    let mut r = HashMap::new(); r.insert("text".into(), json.get("text").cloned().unwrap_or(serde_json::Value::String(String::new()))); Ok(r)
}

pub fn transcribe_audio_file(file_path: &str, language: Option<&str>) -> Result<String> {
    let data = std::fs::read(file_path).map_err(|e| NpcError::FileLoad { path: file_path.into(), source: e })?;
    let result = stt_whisper(&data, "small", language)?;
    Ok(result.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string())
}
