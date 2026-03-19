use crate::error::{NpcError, Result};
use std::collections::HashMap;

pub async fn tts_elevenlabs(text: &str, voice: &str, api_key: Option<&str>, model: Option<&str>) -> Result<Vec<u8>> {
    let key = api_key
        .map(String::from)
        .or_else(|| std::env::var("ELEVENLABS_API_KEY").ok())
        .ok_or_else(|| NpcError::LlmRequest("ELEVENLABS_API_KEY not set".into()))?;
    let model_id = model.unwrap_or("eleven_monolingual_v1");
    let url = format!("https://api.elevenlabs.io/v1/text-to-speech/{}", voice);
    let body = serde_json::json!({
        "text": text,
        "model_id": model_id,
    });
    let client = reqwest::Client::new();
    let resp = client.post(&url)
        .header("xi-api-key", &key)
        .header("Content-Type", "application/json")
        .json(&body)
        .send().await?;
    if !resp.status().is_success() {
        let err = resp.text().await.unwrap_or_default();
        return Err(NpcError::Generation(format!("ElevenLabs TTS failed: {}", err)));
    }
    Ok(resp.bytes().await?.to_vec())
}

pub async fn get_elevenlabs_voices(api_key: Option<&str>) -> Result<Vec<String>> {
    let key = api_key
        .map(String::from)
        .or_else(|| std::env::var("ELEVENLABS_API_KEY").ok())
        .ok_or_else(|| NpcError::LlmRequest("ELEVENLABS_API_KEY not set".into()))?;
    let client = reqwest::Client::new();
    let resp = client.get("https://api.elevenlabs.io/v1/voices")
        .header("xi-api-key", &key)
        .send().await?;
    let json: serde_json::Value = resp.json().await?;
    Ok(json.get("voices").and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.get("name").and_then(|n| n.as_str()).map(String::from)).collect())
        .unwrap_or_default())
}

pub async fn tts_openai(text: &str, voice: &str, api_key: Option<&str>, model: Option<&str>) -> Result<Vec<u8>> {
    let key = api_key
        .map(String::from)
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or_else(|| NpcError::LlmRequest("OPENAI_API_KEY not set".into()))?;
    let model = model.unwrap_or("tts-1");
    let body = serde_json::json!({
        "model": model,
        "input": text,
        "voice": voice,
    });
    let client = reqwest::Client::new();
    let resp = client.post("https://api.openai.com/v1/audio/speech")
        .header("Authorization", format!("Bearer {}", key))
        .json(&body)
        .send().await?;
    if !resp.status().is_success() {
        let err = resp.text().await.unwrap_or_default();
        return Err(NpcError::Generation(format!("OpenAI TTS failed: {}", err)));
    }
    Ok(resp.bytes().await?.to_vec())
}

pub fn get_openai_voices() -> Vec<String> {
    vec!["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        .into_iter().map(String::from).collect()
}

pub async fn tts_gemini(text: &str, voice: &str, api_key: Option<&str>) -> Result<Vec<u8>> {
    let key = api_key
        .map(String::from)
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .or_else(|| std::env::var("GEMINI_API_KEY").ok())
        .ok_or_else(|| NpcError::LlmRequest("GOOGLE_API_KEY not set".into()))?;
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}",
        key
    );
    let body = serde_json::json!({
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["audio"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}
        }
    });
    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        let err = resp.text().await.unwrap_or_default();
        return Err(NpcError::Generation(format!("Gemini TTS failed: {}", err)));
    }
    let json: serde_json::Value = resp.json().await?;
    if let Some(b64) = json["candidates"][0]["content"]["parts"][0]["inlineData"]["data"].as_str() {
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD.decode(b64)
            .map_err(|e| NpcError::Generation(format!("Base64 decode: {}", e)))?;
        Ok(data)
    } else {
        Err(NpcError::Generation("No audio in Gemini TTS response".into()))
    }
}

pub fn get_gemini_voices() -> Vec<String> {
    vec!["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
        .into_iter().map(String::from).collect()
}

pub async fn text_to_speech(text: &str, engine: &str, voice: Option<&str>, api_key: Option<&str>) -> Result<Vec<u8>> {
    match engine {
        "openai" => {
            let v = voice.unwrap_or("alloy");
            tts_openai(text, v, api_key, None).await
        }
        "elevenlabs" => {
            let v = voice.unwrap_or("Rachel");
            tts_elevenlabs(text, v, api_key, None).await
        }
        "gemini" => {
            let v = voice.unwrap_or("Puck");
            tts_gemini(text, v, api_key).await
        }
        _ => Err(NpcError::UnsupportedProvider { provider: engine.to_string() }),
    }
}

pub fn get_available_voices(engine: &str) -> Vec<String> {
    match engine {
        "openai" => get_openai_voices(),
        "gemini" => get_gemini_voices(),
        _ => vec![],
    }
}

pub fn get_available_engines() -> HashMap<String, bool> {
    let mut engines = HashMap::new();
    engines.insert("openai".into(), std::env::var("OPENAI_API_KEY").is_ok());
    engines.insert("elevenlabs".into(), std::env::var("ELEVENLABS_API_KEY").is_ok());
    engines.insert("gemini".into(), std::env::var("GOOGLE_API_KEY").is_ok() || std::env::var("GEMINI_API_KEY").is_ok());
    engines
}

pub fn pcm16_to_wav(pcm_data: &[u8], sample_rate: u32, channels: u16) -> Vec<u8> {
    let data_size = pcm_data.len() as u32;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + pcm_data.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(pcm_data);
    wav
}

pub fn wav_to_pcm16(wav_data: &[u8]) -> (Vec<u8>, u32, u16) {
    if wav_data.len() < 44 {
        return (wav_data.to_vec(), 24000, 1);
    }
    let sample_rate = u32::from_le_bytes([wav_data[24], wav_data[25], wav_data[26], wav_data[27]]);
    let channels = u16::from_le_bytes([wav_data[22], wav_data[23]]);
    let pcm = wav_data[44..].to_vec();
    (pcm, sample_rate, channels)
}

pub fn audio_to_base64(audio_data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(audio_data)
}

pub fn base64_to_audio(b64_string: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(b64_string)
        .map_err(|e| NpcError::Generation(format!("Base64 decode: {}", e)))
}
