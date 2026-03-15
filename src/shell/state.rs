//! Shell runtime state — used by the FFI/Flutter shell path.
//! The main npcsh binary (rust/src/main.rs) has its own REPL loop.
//! TODO: Rewrite to match npcpy _state.py properly.

use crate::r#gen::Message;
use crate::memory::CommandHistory;
use crate::npc_compiler::{Npc, Team};
use crate::error::Result;

/// The runtime state of the shell.
pub struct ShellState {
    pub npc: Npc,
    pub team: Team,
    pub history: CommandHistory,
    pub messages: Vec<Message>,
    pub conversation_id: String,
    pub current_mode: ShellMode,
    pub current_path: String,
    pub stream_output: bool,
}

/// Shell mode.
#[derive(Debug, Clone)]
pub enum ShellMode {
    Agent,
    Chat,
    Cmd,
    Custom(String),
}
