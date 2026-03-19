
pub mod kernel;
pub mod process;
pub mod scheduler;
pub mod ipc;
pub mod vfs;
pub mod drivers;

pub mod npc_compiler;    // NPC, Team, Jinx, Agent — mirrors npcpy.npc_compiler
pub mod r#gen;           // LLM response, cost, sanitize, image — mirrors npcpy.gen
pub mod memory;          // History, KG, embeddings, search — mirrors npcpy.memory
pub mod tools;           // Tool registry — mirrors npcpy.tools
pub mod data;            // Web, file loading, text — mirrors npcpy.data
pub mod work;            // Job scheduling, triggers — mirrors npcpy.work
pub mod mix;             // Multi-agent debate — mirrors npcpy.mix
pub mod ft;              // Fine-tuning — mirrors npcpy.ft
pub mod ml_funcs;        // ML utilities — mirrors npcpy.ml_funcs
pub mod npc_array;       // Vectorized inference — mirrors npcpy.npc_array
pub mod llm_funcs;       // High-level LLM functions — mirrors npcpy.llm_funcs
pub mod npc_sysenv;      // System environment — mirrors npcpy.npc_sysenv

pub mod mcp;
pub mod template;
pub mod serve;
pub mod shell;

#[cfg(feature = "ffi")]
pub mod ffi;

pub mod error;

pub use npc_compiler::{Npc, Team, Jinx, Agent, ToolAgent, CodingAgent};
pub use r#gen::{Message, ToolCall, ToolDef, LlmResponse, Usage};
pub use r#gen::{calculate_cost, sanitize_messages};
pub use process::Process;
pub use kernel::Kernel;
pub use tools::ToolRegistry;
pub use shell::ShellState;
pub use error::{NpcError, Result};
