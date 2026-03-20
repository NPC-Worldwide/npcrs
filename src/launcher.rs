use crate::error::{NpcError, Result};
use std::collections::HashMap;
use std::path::Path;

pub fn discover_team_path(explicit: Option<&str>) -> String {
    if let Some(p) = explicit {
        return p.to_string();
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let candidate = cwd.join("npc_team");
    if candidate.exists() {
        return candidate.to_string_lossy().to_string();
    }
    let global = crate::npc_sysenv::get_data_dir().join("npc_team");
    global.to_string_lossy().to_string()
}

pub fn load_team(team_path: &str) -> Result<crate::npc_compiler::Team> {
    crate::npc_compiler::load_team_from_directory(team_path)
}

pub fn pick_npc(npcs: &HashMap<String, crate::npc_compiler::Npc>) -> String {
    npcs.keys().next().map(|s| s.clone()).unwrap_or_else(|| "assistant".to_string())
}

pub fn build_system_prompt(npc_name: &str, npcs: &HashMap<String, crate::npc_compiler::Npc>) -> String {
    if let Some(npc) = npcs.get(npc_name) {
        npc.system_prompt(None)
    } else {
        format!("You are {}. You are a helpful assistant.", npc_name)
    }
}

pub fn launch_claude(npc_name: &str, npcs: &HashMap<String, crate::npc_compiler::Npc>, extra_args: &[String]) -> Result<()> {
    let prompt = build_system_prompt(npc_name, npcs);
    let mut args = vec!["--system-prompt".to_string(), prompt];
    args.extend(extra_args.iter().cloned());
    let status = std::process::Command::new("claude").args(&args).status()
        .map_err(|e| NpcError::Shell(format!("Failed to launch claude: {}", e)))?;
    if !status.success() { return Err(NpcError::Shell("claude exited with error".into())); }
    Ok(())
}

pub fn launch_codex(npc_name: &str, npcs: &HashMap<String, crate::npc_compiler::Npc>, extra_args: &[String]) -> Result<()> {
    let prompt = build_system_prompt(npc_name, npcs);
    let mut args = vec!["--full-context".to_string(), "--instructions".to_string(), prompt];
    args.extend(extra_args.iter().cloned());
    let status = std::process::Command::new("codex").args(&args).status()
        .map_err(|e| NpcError::Shell(format!("Failed to launch codex: {}", e)))?;
    if !status.success() { return Err(NpcError::Shell("codex exited with error".into())); }
    Ok(())
}

pub fn launch_gemini(npc_name: &str, npcs: &HashMap<String, crate::npc_compiler::Npc>, extra_args: &[String]) -> Result<()> {
    let prompt = build_system_prompt(npc_name, npcs);
    let mut args = vec!["--system-instruction".to_string(), prompt];
    args.extend(extra_args.iter().cloned());
    let status = std::process::Command::new("gemini").args(&args).status()
        .map_err(|e| NpcError::Shell(format!("Failed to launch gemini: {}", e)))?;
    if !status.success() { return Err(NpcError::Shell("gemini exited with error".into())); }
    Ok(())
}

pub fn launch_opencode(npc_name: &str, npcs: &HashMap<String, crate::npc_compiler::Npc>, extra_args: &[String]) -> Result<()> {
    let prompt = build_system_prompt(npc_name, npcs);
    let mut args = vec!["--system-prompt".to_string(), prompt];
    args.extend(extra_args.iter().cloned());
    let status = std::process::Command::new("opencode").args(&args).status()
        .map_err(|e| NpcError::Shell(format!("Failed to launch opencode: {}", e)))?;
    if !status.success() { return Err(NpcError::Shell("opencode exited with error".into())); }
    Ok(())
}

pub fn launch_aider(npc_name: &str, npcs: &HashMap<String, crate::npc_compiler::Npc>, extra_args: &[String]) -> Result<()> {
    let prompt = build_system_prompt(npc_name, npcs);
    let mut args = vec!["--system-prompt".to_string(), prompt];
    args.extend(extra_args.iter().cloned());
    let status = std::process::Command::new("aider").args(&args).status()
        .map_err(|e| NpcError::Shell(format!("Failed to launch aider: {}", e)))?;
    if !status.success() { return Err(NpcError::Shell("aider exited with error".into())); }
    Ok(())
}

pub fn launch_amp(npc_name: &str, npcs: &HashMap<String, crate::npc_compiler::Npc>, extra_args: &[String]) -> Result<()> {
    let prompt = build_system_prompt(npc_name, npcs);
    let mut args = vec!["--system-prompt".to_string(), prompt];
    args.extend(extra_args.iter().cloned());
    let status = std::process::Command::new("amp").args(&args).status()
        .map_err(|e| NpcError::Shell(format!("Failed to launch amp: {}", e)))?;
    if !status.success() { return Err(NpcError::Shell("amp exited with error".into())); }
    Ok(())
}

pub fn launch(tool: &str, team_path: Option<&str>, npc_name: Option<&str>, extra_args: &[String]) -> Result<()> {
    let tp = discover_team_path(team_path);
    let team = load_team(&tp)?;
    let name = npc_name.map(String::from).unwrap_or_else(|| pick_npc(&team.npcs));
    match tool {
        "claude" => launch_claude(&name, &team.npcs, extra_args),
        "codex" => launch_codex(&name, &team.npcs, extra_args),
        "gemini" => launch_gemini(&name, &team.npcs, extra_args),
        "opencode" => launch_opencode(&name, &team.npcs, extra_args),
        "aider" => launch_aider(&name, &team.npcs, extra_args),
        "amp" => launch_amp(&name, &team.npcs, extra_args),
        _ => Err(NpcError::Shell(format!("Unknown tool: {}", tool))),
    }
}
