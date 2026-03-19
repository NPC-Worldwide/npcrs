//! Video utilities — mirrors npcpy.data.video

use crate::error::Result;

pub fn process_video(file_path: &str, _table_name: &str) -> Result<(Vec<String>, Vec<String>)> {
    let output = std::process::Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-show_entries", "format=duration", "-of", "json", file_path]).output();
    let mut texts = Vec::new();
    match output {
        Ok(out) if out.status.success() => {
            let json_str = String::from_utf8_lossy(&out.stdout);
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&json_str) {
                texts.push(format!("Video: {}x{}, {} frames", json["streams"][0]["width"].as_u64().unwrap_or(0), json["streams"][0]["height"].as_u64().unwrap_or(0), json["streams"][0]["nb_frames"].as_str().unwrap_or("?")));
            }
        }
        _ => { texts.push(format!("Video file: {}", std::path::Path::new(file_path).file_name().and_then(|n| n.to_str()).unwrap_or(file_path))); }
    }
    Ok((Vec::new(), texts))
}

pub fn summarize_video_file(file_path: &str, language: Option<&str>, max_audio_seconds: u32) -> Result<String> {
    let mut meta = Vec::new();
    let output = std::process::Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-show_entries", "format=duration", "-of", "json", file_path]).output();
    match output {
        Ok(out) if out.status.success() => {
            let json_str = String::from_utf8_lossy(&out.stdout);
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&json_str) {
                let basename = std::path::Path::new(file_path).file_name().and_then(|n| n.to_str()).unwrap_or(file_path);
                meta.push(format!("Video file: {} | {}x{} | {} fps | {} frames | ~{}s", basename, json["streams"][0]["width"].as_u64().unwrap_or(0), json["streams"][0]["height"].as_u64().unwrap_or(0), json["streams"][0]["r_frame_rate"].as_str().unwrap_or("?"), json["streams"][0]["nb_frames"].as_str().unwrap_or("?"), json["format"]["duration"].as_str().unwrap_or("?")));
            }
        }
        _ => { meta.push(format!("Video file: {}", std::path::Path::new(file_path).file_name().and_then(|n| n.to_str()).unwrap_or(file_path))); }
    }
    let tmp_audio = std::env::temp_dir().join(format!("npc_va_{}.wav", std::process::id()));
    let extract = std::process::Command::new("ffmpeg").args(["-y", "-i", file_path, "-vn", "-ac", "1", "-ar", "16000", "-t", &max_audio_seconds.to_string(), tmp_audio.to_str().unwrap()]).output();
    let audio_ok = extract.map(|o| o.status.success()).unwrap_or(false) && tmp_audio.exists();
    let mut transcript = String::new();
    if audio_ok {
        if let Ok(t) = super::audio::transcribe_audio_file(tmp_audio.to_str().unwrap(), language) { if !t.is_empty() { transcript = t; } }
        let _ = std::fs::remove_file(&tmp_audio);
    }
    if !transcript.is_empty() { meta.push("Audio transcript:".into()); meta.push(transcript); }
    else { meta.push("[No transcript; ensure ffmpeg and transcription backend installed]".into()); }
    Ok(meta.join("\n"))
}
