
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::memory::CommandHistory;
use crate::npc_compiler::Npc;
use crate::shell::{ShellMode, ShellState};
use crate::npc_compiler::Team;

fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

unsafe fn from_c_str(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string()
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

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

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_free(team: *mut Team) {
    if !team.is_null() {
        unsafe {
            drop(Box::from_raw(team));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_npc_count(team: *const Team) -> u32 {
    if team.is_null() {
        return 0;
    }
    unsafe { (*team).npcs.len() as u32 }
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_npc_names(team: *const Team) -> *mut c_char {
    if team.is_null() {
        return to_c_string("[]");
    }
    let names: Vec<&str> = unsafe { (*team).npc_names() };
    to_c_string(&serde_json::to_string(&names).unwrap_or_else(|_| "[]".to_string()))
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_jinx_names(team: *const Team) -> *mut c_char {
    if team.is_null() {
        return to_c_string("[]");
    }
    let names: Vec<&str> = unsafe { (*team).jinx_names() };
    to_c_string(&serde_json::to_string(&names).unwrap_or_else(|_| "[]".to_string()))
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_team_context(team: *const Team) -> *mut c_char {
    if team.is_null() {
        return to_c_string("");
    }
    let ctx = unsafe { &(*team).context };
    to_c_string(ctx.as_deref().unwrap_or(""))
}

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

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_free(npc: *mut Npc) {
    if !npc.is_null() {
        unsafe {
            drop(Box::from_raw(npc));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_name(npc: *const Npc) -> *mut c_char {
    if npc.is_null() {
        return to_c_string("");
    }
    to_c_string(&unsafe { &*npc }.name)
}

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

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_to_json(npc: *const Npc) -> *mut c_char {
    if npc.is_null() {
        return to_c_string("{}");
    }
    let json = serde_json::to_string(unsafe { &*npc }).unwrap_or_else(|_| "{}".to_string());
    to_c_string(&json)
}

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

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_shell_free(state: *mut ShellState) {
    if !state.is_null() {
        unsafe {
            drop(Box::from_raw(state));
        }
    }
}

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
            state.messages = result.messages;
            to_c_string(output)
        }
        Err(e) => to_c_string(&format!("Error: {}", e)),
    }
}

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
