use crate::error::{NpcError, Result};
use std::path::Path;
use std::collections::HashMap;

pub fn get_team_name(team_path: &str) -> String {
    Path::new(team_path).file_name().and_then(|n| n.to_str()).unwrap_or("npc_team").to_string()
}

pub fn build_dockerfile(config: &HashMap<String, String>) -> Result<String> {
    let team_name = config.get("team_name").map(|s| s.as_str()).unwrap_or("npc_team");
    let port = config.get("port").map(|s| s.as_str()).unwrap_or("5000");
    Ok(format!(
        "FROM python:3.11-slim\n\
        WORKDIR /app\n\
        COPY requirements.txt .\n\
        RUN pip install --no-cache-dir -r requirements.txt\n\
        COPY . .\n\
        EXPOSE {}\n\
        CMD [\"python\", \"-m\", \"npcpy.serve\", \"--team\", \"{}\", \"--port\", \"{}\"]\n",
        port, team_name, port
    ))
}

pub fn build_docker_compose(config: &HashMap<String, String>) -> Result<String> {
    let team_name = config.get("team_name").map(|s| s.as_str()).unwrap_or("npc_team");
    let port = config.get("port").map(|s| s.as_str()).unwrap_or("5000");
    Ok(format!(
        "version: '3.8'\nservices:\n  npc-server:\n    build: .\n    ports:\n      - \"{}:{}\"\n    volumes:\n      - ./{}:/app/{}\n    environment:\n      - NPCSH_TEAM_PATH=/app/{}\n",
        port, port, team_name, team_name, team_name
    ))
}

pub fn build_flask_server(config: &HashMap<String, String>) -> Result<String> {
    let team_path = config.get("team_path").map(|s| s.as_str()).unwrap_or("./npc_team");
    let port = config.get("port").map(|s| s.as_str()).unwrap_or("5000");
    Ok(format!(
        "from npcpy.serve import create_app\napp = create_app(team_path='{}')\nif __name__ == '__main__':\n    app.run(port={})\n",
        team_path, port
    ))
}

pub fn build_cli_executable(config: &HashMap<String, String>) -> Result<String> {
    let team_path = config.get("team_path").map(|s| s.as_str()).unwrap_or("./npc_team");
    Ok(format!(
        "#!/usr/bin/env python3\nfrom npcpy.npc_compiler import Team\nteam = Team(team_path='{}')\nresult = team.orchestrate(input('> '))\nprint(result.get('output', ''))\n",
        team_path
    ))
}

pub fn build_static_site(config: &HashMap<String, String>) -> Result<String> {
    let team_name = config.get("team_name").map(|s| s.as_str()).unwrap_or("npc_team");
    Ok(format!(
        "<!DOCTYPE html>\n<html>\n<head><title>{}</title></head>\n<body>\n<h1>{} API</h1>\n<p>Connect to the NPC team via the REST API.</p>\n</body>\n</html>\n",
        team_name, team_name
    ))
}
