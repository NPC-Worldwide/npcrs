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
//! - `kernel/`    — Boot, process table, capability system
//! - `process/`   — NPC process lifecycle, resource limits
//! - `scheduler/` — Process scheduling, priority, token budgets
//! - `ipc/`       — Inter-process communication (pipes, signals, shared memory)
//! - `vfs/`       — Virtual filesystem (real FS + memory + KG)
//! - `drivers/`   — LLM providers, MCP servers, Python runtime as devices
//! - `npc/`       — NPC definition, loading
//! - `jinx/`      — Jinx (syscall) definitions, step execution
//! - `team/`      — Team loading (boot image)
//! - `llm/`       — Raw LLM HTTP client
//! - `memory/`    — SQLite history, knowledge graph
//! - `mcp/`       — MCP protocol client
//! - `shell/`     — User-space shell interface
//! - `template/`  — Tera template engine
//! - `ffi/`       — C-ABI for Flutter/Dart

// ── Kernel layer ──
pub mod kernel;
pub mod process;
pub mod scheduler;
pub mod ipc;
pub mod vfs;
pub mod drivers;

// ── Core types ──
pub mod npc;
pub mod jinx;
pub mod team;
pub mod llm;
pub mod memory;
pub mod mcp;
pub mod template;
pub mod tools;

// ── Generation & high-level functions ──
pub mod r#gen;
pub mod llm_funcs;

// ── Server mode ──
pub mod serve;

// ── User-space ──
pub mod shell;

#[cfg(feature = "ffi")]
pub mod ffi;

pub mod error;

// Re-exports
pub use kernel::Kernel;
pub use npc::Npc;
pub use npc::{Agent, ToolAgent, CodingAgent};
pub use jinx::Jinx;
pub use team::Team;
pub use process::Process;
pub use llm::LlmClient;
pub use tools::ToolRegistry;
pub use shell::ShellState;
pub use error::{NpcError, Result};
