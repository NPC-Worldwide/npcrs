//! NPC compiler — NPCs, Teams, Jinxes, and Agent subclasses.
//! Mirrors npcpy.npc_compiler.

// Type definitions
mod npc_types;
mod team_types;
mod jinx_types;

// Loaders
mod npc_loader;
mod team_loader;
mod jinx_loader;

// Executor
mod jinx_executor;

// Impl blocks (extend types with methods)
mod npc_mod_old;
mod jinx_mod_old;
mod team_mod_old;

// Agent subclasses
pub mod agents;

// Re-export everything flat — matches npcpy's npc_compiler exports
pub use npc_types::*;
pub use npc_loader::*;
pub use team_types::*;
pub use team_loader::*;
pub use jinx_types::*;
pub use jinx_loader::*;
pub use jinx_executor::*;
pub use agents::{Agent, ToolAgent, CodingAgent};
