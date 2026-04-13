use crate::error::Result;
use crate::r#gen::Message;
use crate::memory::CommandHistory;
use crate::npc_compiler::{NPC, Team};
use crate::shell::permissions::{
    build_command_key, is_safe_tool, load_permission_file, match_permission,
};
use std::collections::HashMap;

pub struct ShellState {
    pub npc: NPC,
    pub team: Team,
    pub history: CommandHistory,
    pub messages: Vec<Message>,
    pub conversation_id: String,
    pub current_mode: ShellMode,
    pub current_path: String,
    pub stream_output: bool,

    // Permission system — hierarchical prefix matching.
    // Keys:   "tool_name" or "tool_name:subcommand" (e.g. "sh:git commit")
    // Values: "auto" | "ask" | "deny" | "session"
    // Scoped: session > workspace (./npc_team/) > global (~/.npcsh/npc_team/)
    permission_rules: HashMap<String, String>,
    session_grants: HashMap<String, String>,
    permissions_loaded: bool,
}

#[derive(Debug, Clone)]
pub enum ShellMode {
    Agent,
    Chat,
    Cmd,
    Custom(String),
}

impl ShellState {
    pub fn new(
        npc: NPC,
        team: Team,
        history: CommandHistory,
        conversation_id: String,
        current_path: String,
    ) -> Self {
        Self {
            npc,
            team,
            history,
            messages: Vec::new(),
            conversation_id,
            current_mode: ShellMode::Agent,
            current_path,
            stream_output: false,
            permission_rules: HashMap::new(),
            session_grants: HashMap::new(),
            permissions_loaded: false,
        }
    }

    // ── permission helpers ────────────────────────────────────────────────────

    fn load_permissions(&mut self) {
        if self.permissions_loaded {
            return;
        }
        self.permissions_loaded = true;

        // Global defaults
        let global_path = dirs_next_or_home(".npcsh/npc_team/permissions.yaml");
        self.permission_rules = load_permission_file(&global_path);

        // Workspace overrides
        let workspace_path = format!("{}/npc_team/permissions.yaml", self.current_path);
        let workspace_rules = load_permission_file(&workspace_path);
        self.permission_rules.extend(workspace_rules);
    }

    /// Check permission for a tool call.
    /// Returns "allow", "deny", or "ask".
    pub fn check_tool_permission(&mut self, tool_name: &str, args: &serde_json::Value) -> &'static str {
        if is_safe_tool(tool_name) {
            return "allow";
        }

        self.load_permissions();

        let cmd_key = build_command_key(tool_name, args);

        // Session grants take precedence
        if let Some(decision) = match_permission(&cmd_key, &self.session_grants) {
            return decision_str(decision);
        }

        // Persistent rules
        if let Some(decision) = match_permission(&cmd_key, &self.permission_rules) {
            return decision_str(decision);
        }

        // Default: ask
        "ask"
    }

    /// Grant permission for the rest of this conversation only.
    pub fn grant_session(&mut self, tool_name: &str, args: &serde_json::Value) {
        let key = build_command_key(tool_name, args);
        self.session_grants.insert(key, "auto".to_string());
    }

    /// Grant or deny permission persistently and save to permissions.yaml.
    pub fn save_permission(&mut self, tool_name: &str, args: &serde_json::Value, level: &str, scope: &str) {
        let key = build_command_key(tool_name, args);
        self.permission_rules.insert(key.clone(), level.to_string());

        let dir_path = if scope == "global" {
            dirs_next_or_home(".npcsh/npc_team")
        } else {
            format!("{}/npc_team", self.current_path)
        };

        let perm_path = format!("{}/permissions.yaml", dir_path);
        let mut existing = load_permission_file(&perm_path);
        existing.insert(key, level.to_string());

        let _ = std::fs::create_dir_all(&dir_path);
        if let Ok(yaml) = serde_yaml::to_string(&existing) {
            let _ = std::fs::write(&perm_path, yaml);
        }
    }

    pub fn grant_forever(&mut self, tool_name: &str, args: &serde_json::Value, scope: &str) {
        self.save_permission(tool_name, args, "auto", scope);
    }

    pub fn deny_forever(&mut self, tool_name: &str, args: &serde_json::Value, scope: &str) {
        self.save_permission(tool_name, args, "deny", scope);
    }
}

fn decision_str(s: String) -> &'static str {
    match s.as_str() {
        "auto" | "allow" | "session" => "allow",
        "deny" => "deny",
        _ => "ask",
    }
}

fn dirs_next_or_home(rel: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    format!("{}/{}", home, rel)
}
