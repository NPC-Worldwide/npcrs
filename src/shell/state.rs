
use crate::r#gen::Message;
use crate::memory::CommandHistory;
use crate::npc_compiler::{Npc, Team};
use crate::error::Result;

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

#[derive(Debug, Clone)]
pub enum ShellMode {
    Agent,
    Chat,
    Cmd,
    Custom(String),
}
