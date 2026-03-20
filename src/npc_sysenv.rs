
use std::path::{Path, PathBuf};
use std::collections::HashMap;

pub fn get_data_dir() -> PathBuf {
    if let Ok(home) = std::env::var("INCOGNIDE_HOME") {
        return shellexpand::tilde(&home).to_string().into();
    }

    #[cfg(target_os = "windows")]
    {
        let base = std::env::var("LOCALAPPDATA")
            .unwrap_or_else(|_| dirs::data_local_dir().map(|d| d.to_string_lossy().to_string()).unwrap_or_else(|| "~\\AppData\\Local".into()));
        let new_path = PathBuf::from(&base).join("npcsh");
        let old_path = dirs::home_dir().unwrap_or_default().join(".npcsh");
        if old_path.exists() && !new_path.exists() { return old_path; }
        return new_path;
    }

    #[cfg(target_os = "macos")]
    {
        let new_path = dirs::home_dir().unwrap_or_default().join("Library/Application Support/npcsh");
        let old_path = dirs::home_dir().unwrap_or_default().join(".npcsh");
        if old_path.exists() && !new_path.exists() { return old_path; }
        return new_path;
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let xdg_data = std::env::var("XDG_DATA_HOME")
            .unwrap_or_else(|_| {
                dirs::home_dir().unwrap_or_default().join(".local/share").to_string_lossy().to_string()
            });
        let new_path = PathBuf::from(&xdg_data).join("npcsh");
        let old_path = dirs::home_dir().unwrap_or_default().join(".npcsh");
        if old_path.exists() && !new_path.exists() { return old_path; }
        new_path
    }
}

pub fn get_config_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var("APPDATA")
            .unwrap_or_else(|_| dirs::config_dir().map(|d| d.to_string_lossy().to_string()).unwrap_or_default());
        let new_path = PathBuf::from(&base).join("npcsh");
        let old_path = dirs::home_dir().unwrap_or_default().join(".npcsh");
        if old_path.exists() && !new_path.exists() { return old_path; }
        return new_path;
    }

    #[cfg(target_os = "macos")]
    {
        let new_path = dirs::home_dir().unwrap_or_default().join("Library/Application Support/npcsh");
        let old_path = dirs::home_dir().unwrap_or_default().join(".npcsh");
        if old_path.exists() && !new_path.exists() { return old_path; }
        return new_path;
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let xdg_config = std::env::var("XDG_CONFIG_HOME")
            .unwrap_or_else(|_| {
                dirs::home_dir().unwrap_or_default().join(".config").to_string_lossy().to_string()
            });
        let new_path = PathBuf::from(&xdg_config).join("npcsh");
        let old_path = dirs::home_dir().unwrap_or_default().join(".npcsh");
        if old_path.exists() && !new_path.exists() { return old_path; }
        new_path
    }
}

pub fn get_cache_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var("LOCALAPPDATA")
            .unwrap_or_else(|_| dirs::data_local_dir().map(|d| d.to_string_lossy().to_string()).unwrap_or_default());
        return PathBuf::from(&base).join("npcsh").join("cache");
    }

    #[cfg(target_os = "macos")]
    {
        return dirs::home_dir().unwrap_or_default().join("Library/Caches/npcsh");
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let xdg_cache = std::env::var("XDG_CACHE_HOME")
            .unwrap_or_else(|_| {
                dirs::home_dir().unwrap_or_default().join(".cache").to_string_lossy().to_string()
            });
        PathBuf::from(&xdg_cache).join("npcsh")
    }
}

pub fn get_npcshrc_path() -> PathBuf {
    let old_path = dirs::home_dir().unwrap_or_default().join(".npcshrc");
    if old_path.exists() { return old_path; }
    get_config_dir().join("npcshrc")
}

pub fn get_history_db_path() -> PathBuf {
    let old_path = dirs::home_dir().unwrap_or_default().join("npcsh_history.db");
    if old_path.exists() { return old_path; }
    get_data_dir().join("history.db")
}

pub fn get_models_dir() -> PathBuf { get_data_dir().join("npc_team").join("models") }
pub fn get_images_dir() -> PathBuf { get_data_dir().join("npc_team").join("images") }
pub fn get_jobs_dir() -> PathBuf { get_data_dir().join("npc_team").join("jobs") }
pub fn get_triggers_dir() -> PathBuf { get_data_dir().join("npc_team").join("triggers") }
pub fn get_videos_dir() -> PathBuf { get_data_dir().join("npc_team").join("videos") }
pub fn get_attachments_dir() -> PathBuf { get_data_dir().join("npc_team").join("attachments") }
pub fn get_logs_dir() -> PathBuf { get_data_dir().join("npc_team").join("logs") }

pub fn ensure_npcsh_dirs() {
    for dir in &[
        get_data_dir(), get_config_dir(), get_cache_dir(),
        get_models_dir(), get_images_dir(), get_jobs_dir(),
        get_triggers_dir(), get_videos_dir(), get_attachments_dir(),
        get_logs_dir(),
    ] {
        let _ = std::fs::create_dir_all(dir);
    }
}

pub fn check_internet_connection(timeout_secs: u64) -> bool {
    use std::net::{TcpStream, SocketAddr};
    let addr: SocketAddr = "8.8.8.8:53".parse().unwrap();
    TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(timeout_secs)).is_ok()
}

pub fn load_env_from_execution_dir() {
    let cwd = std::env::current_dir().unwrap_or_default();
    let env_path = cwd.join(".env");
    if env_path.exists() {
        if let Ok(contents) = std::fs::read_to_string(&env_path) {
            for line in contents.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') { continue; }
                let line = line.strip_prefix("export ").unwrap_or(line);
                if let Some((key, value)) = line.split_once('=') {
                    let key = key.trim();
                    let value = value.trim().trim_matches('"').trim_matches('\'');
                    unsafe { std::env::set_var(key, value); }
                }
            }
        }
    }
}

pub fn lookup_provider(model: &str) -> Option<String> {
    if model.is_empty() { return None; }

    let expanded = shellexpand::tilde(model).to_string();
    if Path::new(&expanded).is_dir() {
        let adapter_config = Path::new(&expanded).join("adapter_config.json");
        if adapter_config.exists() {
            return Some("lora".into());
        }
    }

    let custom = load_custom_providers();
    for (provider_name, _config) in &custom {
        if model.starts_with(&format!("{}-", provider_name)) {
            return Some(provider_name.clone());
        }
    }

    if model == "deepseek-chat" || model == "deepseek-reasoner" {
        return Some("deepseek".into());
    }

    if model.starts_with("airllm-") {
        return Some("airllm".into());
    }

    let ollama_prefixes = ["llama", "deepseek", "qwen", "llava", "phi", "mistral", "mixtral", "dolphin", "codellama", "gemma"];
    if ollama_prefixes.iter().any(|p| model.starts_with(p)) {
        return Some("ollama".into());
    }

    let openai_prefixes = ["gpt-", "dall-e-", "whisper-", "o1", "o3", "o4", "gpt-image"];
    if openai_prefixes.iter().any(|p| model.starts_with(p)) {
        return Some("openai".into());
    }

    if model.starts_with("claude") { return Some("anthropic".into()); }
    if model.starts_with("gemini") || model.starts_with("veo") { return Some("gemini".into()); }
    if model.contains("diffusion") { return Some("diffusers".into()); }

    None
}

pub fn load_custom_providers() -> HashMap<String, serde_json::Value> {
    let mut providers = HashMap::new();
    let rc_path = get_npcshrc_path();
    if !rc_path.exists() { return providers; }

    if let Ok(contents) = std::fs::read_to_string(&rc_path) {
        for line in contents.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if !line.contains("CUSTOM_PROVIDER_") || !line.contains('=') { continue; }
            let line = line.strip_prefix("export ").unwrap_or(line);
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"').trim_matches('\'');
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(value) {
                    let provider_name = key.replace("CUSTOM_PROVIDER_", "").to_lowercase();
                    providers.insert(provider_name, config);
                }
            }
        }
    }
    providers
}

use crate::npc_compiler::{NPC, Jinx};

pub fn get_system_message(npc: Option<&NPC>, tool_capable: bool, team_context: Option<&str>, team_members: Option<&[(String, String)]>, jinxes: &HashMap<String, Jinx>) -> String {
    let npc = match npc {
        Some(n) => n,
        None => return "You are a helpful assistant".into(),
    };

    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let cwd = std::env::current_dir().unwrap_or_default().to_string_lossy().to_string();

    let mut msg = format!(
        ".\n..\n...\n....\n.....\n......\n.......\n........\n.........\n..........\n\
        Hello!\nWelcome to the team.\n\
        You are the {} NPC with the following primary directive: {}.\n\
        Users may refer to you by your assistant name, {} and you should \
        consider this to be your core identity.\n\
        The current working directory is {}.\n\
        The current date and time are : {}\n",
        npc.name,
        npc.primary_directive.as_deref().unwrap_or(""),
        npc.name,
        cwd,
        now,
    );

    if let Some(ctx) = team_context {
        msg.push_str(&format!("\nTeam context: {}\n", ctx));
    }

    if let Some(members) = team_members {
        if !members.is_empty() {
            msg.push_str("\nTeam members available for delegation:\n");
            for (name, directive) in members {
                if name != &npc.name {
                    let desc: String = directive.chars().take(50).collect();
                    msg.push_str(&format!("  - @{}: {}\n", name, desc.trim()));
                }
            }
        }
    }

    if !jinxes.is_empty() {
        msg.push_str("\nYou have access to the following jinxes:\n");
        for (jname, jinx) in jinxes {
            msg.push_str(&format!("  - {}: {}\n", jname, jinx.description.trim()));
        }

        if tool_capable {
            msg.push_str("\nUse the provided function calling interface to invoke tools when they are relevant to the request. For multi-step tasks, call one tool at a time and use its result to inform the next step.\n");
        } else {
            let jinx_names: Vec<&str> = jinxes.keys().map(|s| s.as_str()).collect();
            msg.push_str(&format!(
                "\nif you are in the [ReAct loop] and you are asked to use jinxes, refer to these guidelines:\n\
                [BEGIN GUIDELINES FOR JINX EXECUTION]\n\
                Use jinxes when appropriate. For example:\n\
                - If you are asked about something up-to-date or dynamic\n\
                - If the user asks you to read or edit a file\n\
                - If the user asks for code that should be executed\n\
                - If the user requests to open, search, download or scrape\n\
                - If they request a screenshot, audio, or image manipulation\n\
                - Situations requiring file parsing\n\
                - Scripted workflows or pipelines\n\n\
                You do not need to use a jinx if:\n\
                - the user asks a simple question that only requires general knowledge\n\
                - The user asks you to write them a story (unless they specify saving to a file)\n\n\
                To invoke a jinx, return:\n\
                {{\n\
                    \"action\": \"invoke_jinx\",\n\
                    \"jinx_name\": \"jinx_name_here\",\n\
                    \"explanation\": \"detailed explanation\"\n\
                }}\n\n\
                Do not invent jinx names. Use only: [{}]\n\
                [END GUIDELINES FOR JINX EXECUTION]\n",
                jinx_names.join(", ")
            ));
        }
    }

    msg
}

pub fn log_action(action: &str, detail: &str) {
    let logs_dir = get_logs_dir();
    let _ = std::fs::create_dir_all(&logs_dir);
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let entry = format!("[{}] {}: {}\n", now, action, detail);
    let log_file = logs_dir.join("npcsh.log");
    let _ = std::fs::OpenOptions::new()
        .create(true).append(true).open(&log_file)
        .and_then(|mut f| std::io::Write::write_all(&mut f, entry.as_bytes()));
}

pub fn get_directory_npcs(directory: Option<&str>) -> Vec<String> {
    let dir = directory
        .map(PathBuf::from)
        .unwrap_or_else(|| get_data_dir().join("npc_team"));
    let mut npcs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "npc").unwrap_or(false) {
                if let Some(stem) = path.file_stem() {
                    npcs.push(stem.to_string_lossy().to_string());
                }
            }
        }
    }
    npcs
}

pub fn guess_mime_type(filename: &str) -> &'static str {
    let ext = Path::new(filename).extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext.to_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "pdf" => "application/pdf",
        "csv" => "text/csv",
        "json" => "application/json",
        "html" | "htm" => "text/html",
        "txt" | "md" | "rst" => "text/plain",
        "py" => "text/x-python",
        "rs" => "text/x-rust",
        "js" => "text/javascript",
        "ts" => "text/typescript",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "xlsx" | "xls" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        _ => "application/octet-stream",
    }
}

pub fn ensure_dirs_exist(dirs: &[&Path]) {
    for dir in dirs {
        let _ = std::fs::create_dir_all(dir);
    }
}

pub fn resolve_team_dir(team_path: Option<&str>) -> PathBuf {
    let base = get_data_dir();
    match team_path {
        None | Some("incognide") => base.join("incognide").join("npc_team"),
        Some("npcsh") => base.join("npc_team"),
        Some(path) => PathBuf::from(path),
    }
}

fn git(args: &[&str], cwd: &Path) -> std::result::Result<String, String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("git failed: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(stderr.is_empty().then(|| format!("git {} failed", args[0])).unwrap_or(stderr));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn team_sync_status(team_path: Option<&str>) -> std::result::Result<HashMap<String, serde_json::Value>, String> {
    let team_dir = resolve_team_dir(team_path);
    if !team_dir.join(".git").exists() {
        return Ok({
            let mut m = HashMap::new();
            m.insert("initialized".into(), serde_json::json!(false));
            m
        });
    }

    let branch = git(&["rev-parse", "--abbrev-ref", "HEAD"], &team_dir)?;
    let status = git(&["status", "--porcelain"], &team_dir)?;
    let remote = git(&["remote"], &team_dir).unwrap_or_default();
    let has_remote = !remote.is_empty();

    let mut behind = 0u64;
    let mut ahead = 0u64;
    if has_remote {
        let _ = git(&["fetch", "--quiet"], &team_dir);
        if let Ok(counts) = git(&["rev-list", "--left-right", "--count", &format!("HEAD...origin/{}", branch)], &team_dir) {
            let parts: Vec<&str> = counts.split_whitespace().collect();
            if parts.len() == 2 {
                ahead = parts[0].parse().unwrap_or(0);
                behind = parts[1].parse().unwrap_or(0);
            }
        }
    }

    let modified: Vec<&str> = status.lines()
        .filter(|l| l.starts_with(" M") || l.starts_with("M "))
        .map(|l| l[3..].trim())
        .collect();

    let mut result = HashMap::new();
    result.insert("initialized".into(), serde_json::json!(true));
    result.insert("branch".into(), serde_json::json!(branch));
    result.insert("has_remote".into(), serde_json::json!(has_remote));
    result.insert("behind".into(), serde_json::json!(behind));
    result.insert("ahead".into(), serde_json::json!(ahead));
    result.insert("modified_files".into(), serde_json::json!(modified));
    result.insert("clean".into(), serde_json::json!(status.is_empty()));
    Ok(result)
}

pub fn team_sync_init(team_path: Option<&str>) -> std::result::Result<String, String> {
    let team_dir = resolve_team_dir(team_path);
    let _ = std::fs::create_dir_all(&team_dir);
    if !team_dir.join(".git").exists() {
        git(&["init"], &team_dir)?;
    }
    git(&["add", "."], &team_dir)?;
    let _ = git(&["commit", "-m", "Initial NPC team commit"], &team_dir);
    Ok("Team sync initialized".into())
}

pub fn team_sync_pull(team_path: Option<&str>) -> std::result::Result<String, String> {
    let team_dir = resolve_team_dir(team_path);
    if !team_dir.join(".git").exists() {
        return Err("Team directory is not a git repository".into());
    }
    let remote = git(&["remote"], &team_dir)?;
    if remote.is_empty() {
        return Err("No remote configured".into());
    }
    let branch = git(&["rev-parse", "--abbrev-ref", "HEAD"], &team_dir)?;
    git(&["pull", "origin", &branch], &team_dir)
}

pub fn team_sync_commit(team_path: Option<&str>, message: &str) -> std::result::Result<String, String> {
    let team_dir = resolve_team_dir(team_path);
    git(&["add", "."], &team_dir)?;
    git(&["commit", "-m", message], &team_dir)?;
    let remote = git(&["remote"], &team_dir).unwrap_or_default();
    if !remote.is_empty() {
        let branch = git(&["rev-parse", "--abbrev-ref", "HEAD"], &team_dir)?;
        git(&["push", "origin", &branch], &team_dir)?;
    }
    Ok("Changes committed".into())
}

pub fn team_sync_diff(team_path: Option<&str>, file_path: Option<&str>) -> std::result::Result<String, String> {
    let team_dir = resolve_team_dir(team_path);
    let mut args = vec!["diff"];
    if let Some(fp) = file_path {
        args.push("--");
        args.push(fp);
    }
    git(&args, &team_dir)
}

pub fn init_db_tables(db_path: Option<&str>) -> std::result::Result<(), String> {
    let path = db_path
        .map(|p| shellexpand::tilde(p).to_string())
        .unwrap_or_else(|| get_history_db_path().to_string_lossy().to_string());

    let conn = rusqlite::Connection::open(&path)
        .map_err(|e| format!("Failed to open DB: {}", e))?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            message_id TEXT UNIQUE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            npc TEXT,
            team TEXT,
            model TEXT,
            provider TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            session_cost REAL DEFAULT 0.0
        );
        CREATE TABLE IF NOT EXISTS jinx_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            jinx_name TEXT NOT NULL,
            input_values TEXT,
            output TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS npc_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            npc_name TEXT NOT NULL,
            command TEXT,
            output TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS memory_lifecycle (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            memory_text TEXT NOT NULL,
            memory_type TEXT DEFAULT 'auto',
            status TEXT DEFAULT 'pending',
            embedding TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            approved_at DATETIME
        );
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            label TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            node_type TEXT NOT NULL,
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER REFERENCES knowledge_graph_nodes(id),
            target_id INTEGER REFERENCES knowledge_graph_nodes(id),
            edge_type TEXT,
            weight REAL DEFAULT 1.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS npc_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            npc_name TEXT NOT NULL,
            version INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(npc_name, version)
        );"
    ).map_err(|e| format!("Failed to create tables: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_data_dir_returns_path() {
        let dir = get_data_dir();
        assert!(!dir.to_string_lossy().is_empty());
    }

    #[test]
    fn test_lookup_provider_known() {
        assert_eq!(lookup_provider("gpt-4o"), Some("openai".into()));
        assert_eq!(lookup_provider("claude-3-opus"), Some("anthropic".into()));
        assert_eq!(lookup_provider("gemini-2.5-flash"), Some("gemini".into()));
        assert_eq!(lookup_provider("llama3.2"), Some("ollama".into()));
        assert_eq!(lookup_provider("qwen3:8b"), Some("ollama".into()));
        assert_eq!(lookup_provider("deepseek-chat"), Some("deepseek".into()));
    }

    #[test]
    fn test_guess_mime_type() {
        assert_eq!(guess_mime_type("photo.jpg"), "image/jpeg");
        assert_eq!(guess_mime_type("data.csv"), "text/csv");
        assert_eq!(guess_mime_type("doc.pdf"), "application/pdf");
        assert_eq!(guess_mime_type("unknown.xyz"), "application/octet-stream");
    }

    #[test]
    fn test_resolve_team_dir() {
        let dir = resolve_team_dir(Some("npcsh"));
        assert!(dir.to_string_lossy().contains("npc_team"));
    }
}
