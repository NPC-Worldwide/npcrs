use crate::error::{NpcError, Result};
use crate::jinx;
use crate::npc;
use crate::team::{Team, TeamCtx};
use std::path::Path;
use walkdir::WalkDir;

/// Load a Team from a directory.
///
/// Expected structure:
/// ```text
/// npc_team/
///   ├── team.ctx (or <name>.ctx)
///   ├── *.npc files
///   └── jinxes/
///       ├── bin/
///       ├── lib/
///       ├── modes/
///       └── ...
/// ```
pub fn load_team_from_directory(dir: impl AsRef<Path>) -> Result<Team> {
    let dir = dir.as_ref();
    let mut team = Team {
        source_dir: Some(dir.display().to_string()),
        ..Default::default()
    };

    if !dir.exists() {
        return Ok(team);
    }

    // 1. Load .ctx file (team context)
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

    // 2. Load all .npc files
    for entry in WalkDir::new(dir)
        .max_depth(1)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "npc") {
            match npc::load_npc_from_file(path) {
                Ok(mut loaded_npc) => {
                    // Inherit team defaults if NPC doesn't specify
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

    // 3. Load jinxes from jinxes/ directory
    let jinxes_dir = dir.join("jinxes");
    if jinxes_dir.exists() {
        team.jinxes = jinx::load_jinxes_from_directory(&jinxes_dir)?;
    }

    // Also check legacy "jinxs/" directory
    let legacy_dir = dir.join("jinxs");
    if legacy_dir.exists() && !jinxes_dir.exists() {
        team.jinxes = jinx::load_jinxes_from_directory(&legacy_dir)?;
    }

    // 4. Wire up NPC jinx references
    // If an NPC has jinx_names: ["*"], give it all team jinxes
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

/// Find and load the .ctx file from a team directory.
fn find_and_load_ctx(dir: &Path) -> Result<Option<TeamCtx>> {
    // Look for team.ctx first, then any .ctx file
    let candidates = ["team.ctx", "npcsh.ctx"];

    for name in &candidates {
        let path = dir.join(name);
        if path.exists() {
            return load_ctx_file(&path).map(Some);
        }
    }

    // Fallback: find any .ctx file
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

/// Load a .ctx YAML file.
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
