//! C-ABI FFI exports for Flutter/Dart integration.
//!
//! This module exposes npcrs functionality through a C-compatible interface
//! that can be called from Dart via `dart:ffi`.
//!
//! ## Usage from Dart:
//! ```dart
//! final npcrs = DynamicLibrary.open('libnpcrs.so');
//! final initTeam = npcrs.lookupFunction<...>('npcrs_team_load');
//! ```
//!
//! ## Memory model:
//! - Strings are returned as heap-allocated null-terminated C strings.
//! - The caller must free them with `npcrs_free_string`.
//! - Opaque handles are `Box::into_raw` pointers, freed with type-specific free functions.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;


use crate::memory::CommandHistory;
use crate::npc_compiler::Npc;
use crate::shell::{ShellMode, ShellState};
use crate::npc_compiler::Team;

// ─── Helpers ───

fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

unsafe fn from_c_str(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string()
}

/// Free a string returned by npcrs.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

// ─── Team ───

/// Load a team from a directory path. Returns an opaque handle.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_load(path: *const c_char) -> *mut Team {
    let path = unsafe { from_c_str(path) };
    match crate::npc_compiler::load_team_from_directory(&path) {
        Ok(team) => Box::into_raw(Box::new(team)),
        Err(e) => {
            eprintln!("npcrs_team_load error: {}", e);
            ptr::null_mut()
        }
    }
}

/// Free a team handle.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_free(team: *mut Team) {
    if !team.is_null() {
        unsafe {
            drop(Box::from_raw(team));
        }
    }
}

/// Get the number of NPCs in a team.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_npc_count(team: *const Team) -> u32 {
    if team.is_null() {
        return 0;
    }
    unsafe { (*team).npcs.len() as u32 }
}

/// Get NPC names as a JSON array string.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_npc_names(team: *const Team) -> *mut c_char {
    if team.is_null() {
        return to_c_string("[]");
    }
    let names: Vec<&str> = unsafe { (*team).npc_names() };
    to_c_string(&serde_json::to_string(&names).unwrap_or_else(|_| "[]".to_string()))
}

/// Get jinx names as a JSON array string.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_jinx_names(team: *const Team) -> *mut c_char {
    if team.is_null() {
        return to_c_string("[]");
    }
    let names: Vec<&str> = unsafe { (*team).jinx_names() };
    to_c_string(&serde_json::to_string(&names).unwrap_or_else(|_| "[]".to_string()))
}

/// Get the team context string.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_context(team: *const Team) -> *mut c_char {
    if team.is_null() {
        return to_c_string("");
    }
    let ctx = unsafe { &(*team).context };
    to_c_string(ctx.as_deref().unwrap_or(""))
}

// ─── NPC ───

/// Load an NPC from a .npc file. Returns an opaque handle.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_load(path: *const c_char) -> *mut Npc {
    let path = unsafe { from_c_str(path) };
    match Npc::from_file(&path) {
        Ok(npc) => Box::into_raw(Box::new(npc)),
        Err(e) => {
            eprintln!("npcrs_npc_load error: {}", e);
            ptr::null_mut()
        }
    }
}

/// Free an NPC handle.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_free(npc: *mut Npc) {
    if !npc.is_null() {
        unsafe {
            drop(Box::from_raw(npc));
        }
    }
}

/// Get NPC name.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_name(npc: *const Npc) -> *mut c_char {
    if npc.is_null() {
        return to_c_string("");
    }
    to_c_string(&unsafe { &*npc }.name)
}

/// Get NPC system prompt.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_system_prompt(
    npc: *const Npc,
    team_context: *const c_char,
) -> *mut c_char {
    if npc.is_null() {
        return to_c_string("");
    }
    let team_ctx = if team_context.is_null() {
        None
    } else {
        Some(unsafe { from_c_str(team_context) })
    };
    let prompt =
        unsafe { &*npc }.system_prompt(team_ctx.as_deref());
    to_c_string(&prompt)
}

/// Get NPC as JSON.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_to_json(npc: *const Npc) -> *mut c_char {
    if npc.is_null() {
        return to_c_string("{}");
    }
    let json = serde_json::to_string(unsafe { &*npc }).unwrap_or_else(|_| "{}".to_string());
    to_c_string(&json)
}

// ─── Shell State (async, requires runtime) ───

/// Create a new shell state. Returns an opaque handle.
/// The caller must provide a team handle and a database path.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_shell_create(
    team: *mut Team,
    db_path: *const c_char,
) -> *mut ShellState {
    if team.is_null() {
        return ptr::null_mut();
    }

    let db_path = unsafe { from_c_str(db_path) };
    let team = unsafe { &*team }.clone();

    let history = match CommandHistory::open(&db_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("npcrs_shell_create db error: {}", e);
            return ptr::null_mut();
        }
    };

    let npc = team
        .lead_npc()
        .cloned()
        .unwrap_or_else(|| Npc::new("assistant", "You are a helpful assistant."));

    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());

    let state = ShellState {
        npc,
        team,
        history,
        messages: Vec::new(),
        conversation_id: crate::memory::start_new_conversation(),
        current_mode: ShellMode::Agent,
        current_path: cwd,
        stream_output: false,
    };

    Box::into_raw(Box::new(state))
}

/// Free a shell state handle.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_shell_free(state: *mut ShellState) {
    if !state.is_null() {
        unsafe {
            drop(Box::from_raw(state));
        }
    }
}

/// Process a command asynchronously.
/// Returns the output as a C string (must be freed with npcrs_free_string).
///
/// NOTE: This blocks on the tokio runtime. For Flutter, prefer calling
/// through a Dart isolate to avoid blocking the UI thread.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_shell_process_command(
    state: *mut ShellState,
    input: *const c_char,
) -> *mut c_char {
    if state.is_null() || input.is_null() {
        return to_c_string("");
    }

    let state = unsafe { &mut *state };
    let input = unsafe { from_c_str(input) };

    // Add user message
    state.messages.push(crate::r#gen::Message::user(&input));

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(crate::llm_funcs::get_llm_response(
        &input,
        Some(&state.npc),
        None,
        None,
        None,
        &state.messages,
        None,
    )) {
        Ok(result) => {
            let output = result.response.as_deref().unwrap_or("");
            // Update state messages
            state.messages = result.messages;
            to_c_string(output)
        }
        Err(e) => to_c_string(&format!("Error: {}", e)),
    }
}

/// Set the model and provider for a shell state.
/// For local GGUF: set model to the file path, provider to "llamacpp".
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_shell_set_model(
    state: *mut ShellState,
    model: *const c_char,
    provider: *const c_char,
) {
    if state.is_null() {
        return;
    }
    let state = unsafe { &mut *state };
    if !model.is_null() {
        state.npc.model = Some(unsafe { from_c_str(model) });
    }
    if !provider.is_null() {
        state.npc.provider = Some(unsafe { from_c_str(provider) });
    }
}

/// Set an API key as an environment variable.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_set_api_key(
    key_name: *const c_char,
    key_value: *const c_char,
) {
    if key_name.is_null() || key_value.is_null() {
        return;
    }
    let name = unsafe { from_c_str(key_name) };
    let value = unsafe { from_c_str(key_value) };
    unsafe { std::env::set_var(&name, &value) };
}
