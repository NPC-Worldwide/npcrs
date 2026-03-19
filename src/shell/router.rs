use crate::npc_compiler::Jinx;
use std::collections::HashMap;

pub struct CommandRouter {
    routes: HashMap<String, String>,
}

impl CommandRouter {
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }

    pub fn register_jinx(&mut self, jinx: &Jinx) {
        if !jinx.name.is_empty() {
            self.routes
                .insert(jinx.name.clone(), jinx.name.clone());
        }
    }

    pub fn register_all(&mut self, jinxes: &HashMap<String, Jinx>) {
        for jinx in jinxes.values() {
            self.register_jinx(jinx);
        }
    }

    pub fn resolve(&self, command: &str) -> Option<&str> {
        self.routes.get(command).map(|s| s.as_str())
    }

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
