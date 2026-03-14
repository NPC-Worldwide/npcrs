//! Shell REPL state, command routing, and mode dispatch.
//!
//! This is the npcsh-rs runtime — the Rust equivalent of npcsh's _state.py.
//! It manages the REPL loop, parses commands, dispatches to tools or LLM,
//! and tracks session state.

mod router;
mod state;

pub use router::*;
pub use state::*;
