
use crate::error::{NpcError, Result};
use std::collections::HashMap;
use std::path::Path;

pub fn capture_screenshot(full: bool) -> Result<HashMap<String, String>> {
    let dir = shellexpand::tilde("~/.npcsh/screenshots").to_string();
    std::fs::create_dir_all(&dir).map_err(|e| NpcError::FileLoad { path: dir.clone(), source: e })?;
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let filename = format!("screenshot_{}.png", timestamp);
    let file_path = format!("{}/{}", dir, filename);
    let os = std::env::consts::OS;
    if full {
        match os {
            "macos" => { let _ = std::process::Command::new("screencapture").arg(&file_path).output(); }
            "linux" => {
                let tools: &[(&str, &[&str])] = &[("grim", &[]), ("scrot", &[]), ("import", &["-window", "root"]), ("gnome-screenshot", &["-f"])];
                let mut took = false;
                for (cmd, extra) in tools {
                    if std::process::Command::new("which").arg(cmd).output().map(|o| o.status.success()).unwrap_or(false) {
                        let mut args: Vec<&str> = extra.to_vec(); args.push(&file_path);
                        let _ = std::process::Command::new(cmd).args(&args).output();
                        if Path::new(&file_path).exists() { took = true; break; }
                    }
                }
                if !took { return Err(NpcError::Shell("No screenshot tool found".into())); }
            }
            _ => return Err(NpcError::Shell(format!("Unsupported OS: {}", os))),
        }
    } else {
        match os {
            "macos" => { let _ = std::process::Command::new("screencapture").args(["-i", &file_path]).output(); }
            "linux" => {
                if std::process::Command::new("which").arg("scrot").output().map(|o| o.status.success()).unwrap_or(false) {
                    let _ = std::process::Command::new("scrot").args(["-s", &file_path]).output();
                } else { return Err(NpcError::Shell("No interactive screenshot tool".into())); }
            }
            _ => return Err(NpcError::Shell(format!("Unsupported OS: {}", os))),
        }
    }
    if Path::new(&file_path).exists() {
        let mut r = HashMap::new(); r.insert("filename".into(), filename); r.insert("file_path".into(), file_path); Ok(r)
    } else { Err(NpcError::Shell("Screenshot failed".into())) }
}

pub fn compress_image(image_bytes: &[u8], max_width: u32, max_height: u32) -> Result<Vec<u8>> {
    let tmp_in = std::env::temp_dir().join(format!("npc_ci_{}.png", std::process::id()));
    let tmp_out = std::env::temp_dir().join(format!("npc_co_{}.jpg", std::process::id()));
    std::fs::write(&tmp_in, image_bytes).map_err(|e| NpcError::FileLoad { path: tmp_in.display().to_string(), source: e })?;
    let resize = format!("{}x{}>", max_width, max_height);
    let result = std::process::Command::new("convert").args([tmp_in.to_str().unwrap(), "-resize", &resize, "-quality", "95", tmp_out.to_str().unwrap()]).output();
    let out = match result {
        Ok(o) if o.status.success() && tmp_out.exists() => std::fs::read(&tmp_out).unwrap_or_else(|_| image_bytes.to_vec()),
        _ => image_bytes.to_vec(),
    };
    let _ = std::fs::remove_file(&tmp_in); let _ = std::fs::remove_file(&tmp_out);
    Ok(out)
}
