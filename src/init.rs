use crate::error::Result;
use std::path::Path;

pub fn initialize_npc_project(directory: Option<&str>) -> Result<String> {
    let dir = directory.unwrap_or(".");
    let npc_team = Path::new(dir).join("npc_team");
    let jinxes = npc_team.join("jinxes");

    std::fs::create_dir_all(&jinxes).map_err(|e| crate::error::NpcError::Shell(format!("mkdir: {}", e)))?;

    let team_ctx = npc_team.join("team.ctx");
    if !team_ctx.exists() {
        std::fs::write(&team_ctx, "context: |\n  Default NPC team.\nforenpc: assistant\nmodel: qwen3.5:2b\nprovider: ollama\n")
            .map_err(|e| crate::error::NpcError::Shell(format!("write team.ctx: {}", e)))?;
    }

    let assistant_npc = npc_team.join("assistant.npc");
    if !assistant_npc.exists() {
        std::fs::write(&assistant_npc, "#!/usr/bin/env npc\nname: assistant\nprimary_directive: |\n  You are a helpful assistant.\njinxes:\n  - sh\n  - python\n  - web_search\n")
            .map_err(|e| crate::error::NpcError::Shell(format!("write assistant.npc: {}", e)))?;
    }

    Ok(format!("Initialized NPC project at {}", npc_team.display()))
}
