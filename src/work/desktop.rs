//! Desktop automation — mirrors npcpy.work.desktop

use crate::error::Result;
use std::collections::HashMap;

pub fn action_space() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("click", r#"{"x": "int (0-100)", "y": "int (0-100)"}"#);
    m.insert("type", r#"{"text": "string"}"#);
    m.insert("key", r#"{"keys": "list of key names"}"#);
    m.insert("shell", r#"{"command": "string"}"#);
    m.insert("wait", r#"{"duration": "float seconds"}"#);
    m.insert("hotkey", r#"{"keys": "list of keys"}"#);
    m.insert("scroll", r#"{"direction": "up|down", "amount": "int"}"#);
    m.insert("quit", r#"{"description": "goal complete"}"#);
    m
}

pub fn perform_action(action: &serde_json::Value) -> Result<HashMap<String, String>> {
    let action_type = action.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let mut result = HashMap::new();
    match action_type {
        "click" => {
            let x = action.get("x").and_then(|v| v.as_f64()).unwrap_or(50.0).max(0.0).min(100.0);
            let y = action.get("y").and_then(|v| v.as_f64()).unwrap_or(50.0).max(0.0).min(100.0);
            let (w, h) = std::process::Command::new("xdotool").arg("getdisplaygeometry").output()
                .ok().filter(|o| o.status.success()).map(|o| { let s = String::from_utf8_lossy(&o.stdout); let p: Vec<&str> = s.trim().split_whitespace().collect(); (p.first().and_then(|p| p.parse::<f64>().ok()).unwrap_or(1920.0), p.get(1).and_then(|p| p.parse::<f64>().ok()).unwrap_or(1080.0)) }).unwrap_or((1920.0, 1080.0));
            let _ = std::process::Command::new("xdotool").args(["mousemove", &((x * w / 100.0) as i64).to_string(), &((y * h / 100.0) as i64).to_string(), "click", "1"]).output();
            result.insert("status".into(), "success".into()); result.insert("output".into(), format!("Clicked at ({}, {}).", x, y));
        }
        "type" => {
            let text = action.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let _ = std::process::Command::new("xdotool").args(["type", "--clearmodifiers", "--delay", "12", "--", text]).output();
            result.insert("status".into(), "success".into()); result.insert("output".into(), format!("Typed '{}'.", text));
        }
        "key" => {
            let keys: Vec<String> = action.get("keys").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).or_else(|| action.get("keys").and_then(|v| v.as_str()).map(|s| vec![s.into()])).unwrap_or_default();
            for k in &keys { let _ = std::process::Command::new("xdotool").args(["key", k]).output(); }
            result.insert("status".into(), "success".into()); result.insert("output".into(), "Pressed key(s).".into());
        }
        "hotkey" => {
            let keys = action.get("keys").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join("+")).unwrap_or_default();
            if !keys.is_empty() { let _ = std::process::Command::new("xdotool").args(["key", &keys]).output(); }
            result.insert("status".into(), "success".into()); result.insert("output".into(), "Pressed hotkey.".into());
        }
        "shell" | "bash" => {
            let cmd = action.get("command").and_then(|v| v.as_str()).unwrap_or("");
            let _ = std::process::Command::new("sh").args(["-c", cmd]).spawn();
            result.insert("status".into(), "success".into()); result.insert("output".into(), format!("Launched '{}'.", cmd));
        }
        "wait" => {
            let dur = action.get("duration").and_then(|v| v.as_f64()).unwrap_or(1.0);
            std::thread::sleep(std::time::Duration::from_secs_f64(dur));
            result.insert("status".into(), "success".into()); result.insert("output".into(), format!("Waited {}s.", dur));
        }
        "scroll" => {
            let dir = action.get("direction").and_then(|v| v.as_str()).unwrap_or("down");
            let amt = action.get("amount").and_then(|v| v.as_i64()).unwrap_or(3);
            let btn = if dir == "up" { "4" } else { "5" };
            for _ in 0..amt.unsigned_abs() { let _ = std::process::Command::new("xdotool").args(["click", btn]).output(); }
            result.insert("status".into(), "success".into()); result.insert("output".into(), format!("Scrolled {} by {}.", dir, amt));
        }
        other => { result.insert("status".into(), "error".into()); result.insert("message".into(), format!("Unknown action: {}", other)); }
    }
    Ok(result)
}
