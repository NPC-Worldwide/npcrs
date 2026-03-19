use crate::error::{NpcError, Result};

use crate::npc_compiler::{Team, TeamCtx};
use std::path::Path;
use walkdir::WalkDir;

pub fn load_team_from_directory(dir: impl AsRef<Path>) -> Result<Team> {
    let dir = dir.as_ref();
    let mut team = Team {
        source_dir: Some(dir.display().to_string()),
        ..Default::default()
    };

    if !dir.exists() {
        return Ok(team);
    }

    if let Some(ctx) = find_and_load_ctx(dir)? {
        team.context = ctx.context;
        team.model = ctx.model;
        team.provider = ctx.provider;
        team.forenpc = ctx.forenpc;
        team.databases = ctx
            .databases
            .into_iter()
            .map(|d| shellexpand::tilde(&d).to_string())
            .collect();
        team.mcp_servers = ctx.mcp_servers;
    }

    for entry in WalkDir::new(dir)
        .max_depth(1)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "npc") {
            match super::load_npc_from_file(path) {
                Ok(mut loaded_npc) => {
                    if loaded_npc.model.is_none() {
                        loaded_npc.model = team.model.clone();
                    }
                    if loaded_npc.provider.is_none() {
                        loaded_npc.provider = team.provider.clone();
                    }
                    team.npcs.insert(loaded_npc.name.clone(), loaded_npc);
                }
                Err(e) => {
                    tracing::warn!("Failed to load NPC {}: {}", path.display(), e);
                }
            }
        }
    }

    let jinxes_dir = dir.join("jinxes");
    if jinxes_dir.exists() {
        team.jinxes = super::load_jinxes_from_directory(&jinxes_dir)?;
    }

    let legacy_dir = dir.join("jinxs");
    if legacy_dir.exists() && !jinxes_dir.exists() {
        team.jinxes = super::load_jinxes_from_directory(&legacy_dir)?;
    }

    let project_root = dir.parent().unwrap_or(dir);
    let agents_md = project_root.join("agents.md");
    if agents_md.exists() {
        if let Ok(content) = std::fs::read_to_string(&agents_md) {
            load_agents_from_md(&content, &team.model, &team.provider, &mut team.npcs);
        }
    }
    let agents_dir = project_root.join("agents");
    if agents_dir.is_dir() {
        load_agents_from_dir(&agents_dir, &team.model, &team.provider, &mut team.npcs);
    }

    for npc in team.npcs.values_mut() {
        if npc.jinx_names.iter().any(|n| n == "*") {
            npc.jinx_names = team.jinxes.keys().cloned().collect();
        }
    }

    tracing::info!(
        "Loaded team from {}: {} NPCs, {} jinxes, forenpc={:?}",
        dir.display(),
        team.npcs.len(),
        team.jinxes.len(),
        team.forenpc
    );

    Ok(team)
}

fn find_and_load_ctx(dir: &Path) -> Result<Option<TeamCtx>> {
    let candidates = ["team.ctx", "npcsh.ctx"];

    for name in &candidates {
        let path = dir.join(name);
        if path.exists() {
            return load_ctx_file(&path).map(Some);
        }
    }

    for entry in std::fs::read_dir(dir).map_err(|e| NpcError::FileLoad {
        path: dir.display().to_string(),
        source: e,
    })? {
        let entry = entry.map_err(|e| NpcError::FileLoad {
            path: dir.display().to_string(),
            source: e,
        })?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "ctx") {
            return load_ctx_file(&path).map(Some);
        }
    }

    Ok(None)
}

fn load_agents_from_md(
    content: &str,
    team_model: &Option<String>,
    team_provider: &Option<String>,
    npcs: &mut std::collections::HashMap<String, crate::npc_compiler::Npc>,
) {
    let mut current_name: Option<String> = None;
    let mut current_body: Vec<String> = Vec::new();

    for line in content.lines() {
        if let Some(name) = line.strip_prefix("## ") {
            if let Some(prev_name) = current_name.take() {
                if !npcs.contains_key(&prev_name) {
                    let mut npc = crate::npc_compiler::Npc::new(&prev_name, current_body.join("\n").trim());
                    npc.model = team_model.clone();
                    npc.provider = team_provider.clone();
                    npcs.insert(prev_name, npc);
                }
            }
            current_name = Some(name.trim().to_string());
            current_body.clear();
        } else if current_name.is_some() {
            current_body.push(line.to_string());
        }
    }

    if let Some(name) = current_name {
        if !npcs.contains_key(&name) {
            let mut npc = crate::npc_compiler::Npc::new(&name, current_body.join("\n").trim());
            npc.model = team_model.clone();
            npc.provider = team_provider.clone();
            npcs.insert(name, npc);
        }
    }
}

fn load_agents_from_dir(
    dir: &Path,
    team_model: &Option<String>,
    team_provider: &Option<String>,
    npcs: &mut std::collections::HashMap<String, crate::npc_compiler::Npc>,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        if name.is_empty() || npcs.contains_key(&name) {
            continue;
        }

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut model = team_model.clone();
        let mut provider = team_provider.clone();
        let mut agent_name = name.clone();
        let directive;

        if content.starts_with("---") {
            let parts: Vec<&str> = content.splitn(3, "---").collect();
            if parts.len() >= 3 {
                if let Ok(fm) = serde_yaml::from_str::<serde_yaml::Value>(parts[1]) {
                    if let Some(m) = fm.get("model").and_then(|v| v.as_str()) {
                        model = Some(m.to_string());
                    }
                    if let Some(p) = fm.get("provider").and_then(|v| v.as_str()) {
                        provider = Some(p.to_string());
                    }
                    if let Some(n) = fm.get("name").and_then(|v| v.as_str()) {
                        agent_name = n.to_string();
                    }
                }
                directive = parts[2].trim().to_string();
            } else {
                directive = content;
            }
        } else {
            directive = content;
        }

        let mut npc = crate::npc_compiler::Npc::new(&agent_name, &directive);
        npc.model = model;
        npc.provider = provider;
        npcs.insert(agent_name, npc);
    }
}

fn load_ctx_file(path: &Path) -> Result<TeamCtx> {
    let raw = std::fs::read_to_string(path).map_err(|e| NpcError::FileLoad {
        path: path.display().to_string(),
        source: e,
    })?;

    serde_yaml::from_str(&raw).map_err(|e| NpcError::YamlParse {
        path: path.display().to_string(),
        source: e,
    })
}
