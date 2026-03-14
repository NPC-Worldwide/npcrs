//! Team loading and NPC orchestration.
//!
//! A Team is a directory containing .npc files, .jinx files, and a .ctx config.
//! The Team loader scans the directory, compiles NPCs, resolves jinxes,
//! and sets up the shared context.

mod loader;
mod types;

pub use loader::*;
pub use types::*;
