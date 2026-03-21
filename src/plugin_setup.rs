use crate::error::{NpcError, Result};
use std::path::{Path, PathBuf};

fn write_file(path: &Path, content: &str, executable: bool) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| NpcError::Shell(format!("mkdir: {}", e)))?;
    }
    std::fs::write(path, content).map_err(|e| NpcError::Shell(format!("write: {}", e)))?;
    #[cfg(unix)]
    if executable {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755));
    }
    Ok(())
}

pub fn setup_claude(uninstall: bool) -> Result<String> {
    let home = dirs::home_dir().unwrap_or_default();
    let plugins_dir = home.join(".claude").join("plugins");

    if uninstall {
        let _ = std::fs::remove_dir_all(&plugins_dir);
        return Ok("Claude plugin uninstalled".into());
    }

    std::fs::create_dir_all(&plugins_dir).map_err(|e| NpcError::Shell(format!("mkdir: {}", e)))?;

    let manifest = serde_json::json!({
        "name": "npc",
        "description": "NPC team integration",
        "mcpServers": {
            "npc": {
                "command": "python3",
                "args": ["-m", "npcpy.mcp_server"]
            }
        }
    });

    let manifest_path = plugins_dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap_or_default())
        .map_err(|e| NpcError::Shell(format!("write manifest: {}", e)))?;

    Ok("Claude plugin installed".into())
}

pub fn setup_codex(uninstall: bool) -> Result<String> {
    let home = dirs::home_dir().unwrap_or_default();
    let mcp_path = home.join(".codex").join(".mcp.json");

    if uninstall {
        let _ = std::fs::remove_file(&mcp_path);
        return Ok("Codex plugin uninstalled".into());
    }

    let mut existing: serde_json::Value = if mcp_path.exists() {
        let content = std::fs::read_to_string(&mcp_path).unwrap_or_default();
        serde_json::from_str(&content).unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    let servers = existing.as_object_mut().unwrap().entry("mcpServers").or_insert(serde_json::json!({}));
    servers.as_object_mut().unwrap().insert("npc".into(), serde_json::json!({
        "command": "python3",
        "args": ["-m", "npcpy.mcp_server"]
    }));

    if let Some(parent) = mcp_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&mcp_path, serde_json::to_string_pretty(&existing).unwrap_or_default())
        .map_err(|e| NpcError::Shell(format!("write: {}", e)))?;

    Ok("Codex plugin installed".into())
}

pub fn setup_gemini(uninstall: bool) -> Result<String> {
    let home = dirs::home_dir().unwrap_or_default();
    let mcp_path = home.join(".gemini").join(".mcp.json");

    if uninstall {
        let _ = std::fs::remove_file(&mcp_path);
        return Ok("Gemini plugin uninstalled".into());
    }

    let mut existing: serde_json::Value = if mcp_path.exists() {
        let content = std::fs::read_to_string(&mcp_path).unwrap_or_default();
        serde_json::from_str(&content).unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    let servers = existing.as_object_mut().unwrap().entry("mcpServers").or_insert(serde_json::json!({}));
    servers.as_object_mut().unwrap().insert("npc".into(), serde_json::json!({
        "command": "python3",
        "args": ["-m", "npcpy.mcp_server"]
    }));

    if let Some(parent) = mcp_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&mcp_path, serde_json::to_string_pretty(&existing).unwrap_or_default())
        .map_err(|e| NpcError::Shell(format!("write: {}", e)))?;

    Ok("Gemini plugin installed".into())
}
