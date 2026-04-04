pub mod drivers;
pub mod ipc;
pub mod kernel;
pub mod process;
pub mod scheduler;
pub mod vfs;

pub mod data; // Web, file loading, text — mirrors npcpy.data
pub mod ft; // Fine-tuning — mirrors npcpy.ft
pub mod r#gen; // LLM response, cost, sanitize, image — mirrors npcpy.gen
pub mod llm_funcs; // High-level LLM functions — mirrors npcpy.llm_funcs
pub mod memory; // History, KG, embeddings, search — mirrors npcpy.memory
pub mod mix; // Multi-agent debate — mirrors npcpy.mix
pub mod ml_funcs; // ML utilities — mirrors npcpy.ml_funcs
pub mod npc_array; // Vectorized inference — mirrors npcpy.npc_array
pub mod npc_compiler; // NPC, Team, Jinx, Agent — mirrors npcpy.npc_compiler
pub mod npc_sysenv;
pub mod tools; // Tool registry — mirrors npcpy.tools
pub mod work; // Job scheduling, triggers — mirrors npcpy.work // System environment — mirrors npcpy.npc_sysenv

pub mod build_funcs;
pub mod init;
pub mod launcher;
pub mod mcp;
pub mod plugin_setup;
pub mod serve;
pub mod shell;
pub mod streaming;
pub mod template;

#[cfg(feature = "ffi")]
pub mod ffi;

pub mod db;
pub mod error;

pub use error::{NpcError, Result};
pub use r#gen::{LlmResponse, Message, ToolCall, ToolDef, Usage};
pub use r#gen::{calculate_cost, sanitize_messages};
pub use kernel::Kernel;
pub use npc_compiler::{Agent, CodingAgent, Jinx, NPC, Team, ToolAgent};
pub use process::Process;
pub use shell::ShellState;
pub use tools::ToolRegistry;
