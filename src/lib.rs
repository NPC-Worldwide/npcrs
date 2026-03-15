//! npcrs — NPC Operating System kernel
//!
//! A microkernel architecture where AI agents are first-class processes.
//!
//! ## OS Model
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │  User Space                                         │
//! │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
//! │  │ npcsh-rs │ │ Flutter  │ │ MCP CLI  │  (shells)  │
//! │  └────┬─────┘ └────┬─────┘ └────┬─────┘            │
//! ├───────┼─────────────┼───────────┼───────────────────┤
//! │  Kernel                                             │
//! │  ┌─────────────────────────────────────────────┐    │
//! │  │ Process Table (NPC processes)               │    │
//! │  │  pid:1 sibiji (init)                        │    │
//! │  │  pid:2 alicanto (research daemon)           │    │
//! │  │  pid:3 corca (mcp bridge)                   │    │
//! │  └─────────────────────────────────────────────┘    │
//! │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
//! │  │Scheduler │ │   IPC    │ │   VFS    │            │
//! │  └──────────┘ └──────────┘ └──────────┘            │
//! │  ┌──────────────────────────────────────────────┐   │
//! │  │ Syscalls (Jinxes)                            │   │
//! │  │  sh, edit_file, web_search, python, ...      │   │
//! │  └──────────────────────────────────────────────┘   │
//! ├─────────────────────────────────────────────────────┤
//! │  Drivers                                            │
//! │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
//! │  │ OpenAI   │ │Anthropic │ │  Ollama  │ (LLM)     │
//! │  └──────────┘ └──────────┘ └──────────┘            │
//! │  ┌──────────┐ ┌──────────┐                         │
//! │  │ MCP srv  │ │ Python   │              (tools)    │
//! │  └──────────┘ └──────────┘                         │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Module Map
//!
//! - `kernel/`        — Boot, process table, capability system
//! - `process/`       — NPC process lifecycle, resource limits
//! - `scheduler/`     — Process scheduling, priority, token budgets
//! - `ipc/`           — Inter-process communication (pipes, signals, shared memory)
//! - `vfs/`           — Virtual filesystem (real FS + memory + KG)
//! - `drivers/`       — LLM providers, MCP servers, Python runtime as devices
//! - `npc_compiler/`  — NPC, Team, Jinx definitions and loading (mirrors npcpy.npc_compiler)
//! - `gen/`           — LLM response, cost, sanitization, image gen (mirrors npcpy.gen)
//! - `memory/`        — SQLite history, knowledge graph, embeddings
//! - `data/`          — Web search, file loading, text processing (mirrors npcpy.data)
//! - `tools/`         — Tool registry for LLM function calling (mirrors npcpy.tools)
//! - `work/`          — Job scheduling, triggers (mirrors npcpy.work)
//! - `mcp/`           — MCP protocol client
//! - `shell/`         — User-space shell interface
//! - `template/`      — Tera template engine
//! - `ffi/`           — C-ABI for Flutter/Dart

// ── Kernel layer ──
pub mod kernel;
pub mod process;
pub mod scheduler;
pub mod ipc;
pub mod vfs;
pub mod drivers;

// ── Core types (mirrors npcpy) ──
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

// ── Infrastructure ──
pub mod mcp;
pub mod template;
pub mod serve;
pub mod shell;

#[cfg(feature = "ffi")]
pub mod ffi;

pub mod error;

// Re-exports (matching npcpy's top-level imports)
pub use npc_compiler::{Npc, Team, Jinx, Agent, ToolAgent, CodingAgent};
pub use r#gen::{Message, ToolCall, ToolDef, LlmResponse, Usage};
pub use r#gen::{calculate_cost, sanitize_messages};
pub use process::Process;
pub use kernel::Kernel;
pub use tools::ToolRegistry;
pub use shell::ShellState;
pub use error::{NpcError, Result};
