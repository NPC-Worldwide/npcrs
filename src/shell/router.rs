use crate::jinx::Jinx;
use std::collections::HashMap;

/// Command router that maps slash commands to jinx executors.
///
/// Jinxes are registered by name and can be looked up for dispatch.
pub struct CommandRouter {
    /// Registered jinx routes.
    routes: HashMap<String, String>,
}

impl CommandRouter {
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }

    /// Register a jinx as a routable slash command.
    pub fn register_jinx(&mut self, jinx: &Jinx) {
        if !jinx.name.is_empty() {
            self.routes
                .insert(jinx.name.clone(), jinx.name.clone());
        }
    }

    /// Register all jinxes from a map.
    pub fn register_all(&mut self, jinxes: &HashMap<String, Jinx>) {
        for jinx in jinxes.values() {
            self.register_jinx(jinx);
        }
    }

    /// Look up a command name to find the jinx to execute.
    pub fn resolve(&self, command: &str) -> Option<&str> {
        self.routes.get(command).map(|s| s.as_str())
    }

    /// List all registered commands.
    pub fn commands(&self) -> Vec<&str> {
        let mut cmds: Vec<&str> = self.routes.keys().map(|s| s.as_str()).collect();
        cmds.sort();
        cmds
    }
}

impl Default for CommandRouter {
    fn default() -> Self {
        Self::new()
    }
}
