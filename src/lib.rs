#![allow(
    dead_code,
    unused_variables,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::needless_borrow,
    clippy::map_unwrap_or,
    clippy::unwrap_used,
    clippy::manual_unwrap_or,
    clippy::should_implement_trait,
    clippy::regex_creation_in_loops,
    clippy::needless_range_loop,
    clippy::option_map_or_none,
    clippy::if_same_then_else,
    clippy::redundant_closure,
    clippy::manual_split_once,
    clippy::derivable_impls,
    clippy::map_entry,
    clippy::needless_pass_by_ref_mut,
    clippy::unnecessary_mut_passed,
    clippy::unwrap_or_default,
    clippy::duplicated_attributes,
    clippy::for_kv_map,
    clippy::let_and_return,
    clippy::manual_clamp,
    clippy::map_clone,
    clippy::needless_borrows_for_generic_args,
    clippy::obfuscated_if_else,
    clippy::trim_split_whitespace,
    clippy::unnecessary_map_or,
    clippy::unnecessary_unwrap,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::needless_option_as_deref,
    clippy::mut_from_ref,
    clippy::needless_return
)]

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
