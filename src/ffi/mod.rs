use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::atomic::Ordering;

use crate::memory::CommandHistory;
use crate::npc_compiler::NPC;
use crate::npc_compiler::Team;
use crate::shell::ShellState;

fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s).unwrap_or_default().into_raw()
}

unsafe fn from_c_str(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string()
}

/// Ask any in-flight local inference to stop at the next token. The running
/// call returns normally with whatever partial output it had; the flag is
/// cleared automatically when the next turn starts.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_cancel_inference() {
    crate::r#gen::INFERENCE_CANCELLED.store(true, Ordering::Relaxed);
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
pub extern "C" fn npcrs_npc_load(path: *const c_char) -> *mut NPC {
    let path = unsafe { from_c_str(path) };
    match NPC::from_file(&path) {
        Ok(npc) => Box::into_raw(Box::new(npc)),
        Err(e) => {
            eprintln!("npcrs_npc_load error: {}", e);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_free(npc: *mut NPC) {
    if !npc.is_null() {
        unsafe {
            drop(Box::from_raw(npc));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_name(npc: *const NPC) -> *mut c_char {
    if npc.is_null() {
        return to_c_string("");
    }
    to_c_string(&unsafe { &*npc }.name)
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_system_prompt(
    npc: *const NPC,
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
    let prompt = unsafe { &*npc }.system_prompt(team_ctx.as_deref());
    to_c_string(&prompt)
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_npc_to_json(npc: *const NPC) -> *mut c_char {
    if npc.is_null() {
        return to_c_string("{}");
    }
    let json = serde_json::to_string(unsafe { &*npc }).unwrap_or_else(|_| "{}".to_string());
    to_c_string(&json)
}

#[unsafe(no_mangle)]
pub extern "C" fn npcrs_shell_create(team: *mut Team, db_path: *const c_char) -> *mut ShellState {
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
        .unwrap_or_else(|| NPC::new("assistant", "You are a helpful assistant."));

    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());

    let mut state = ShellState::new(
        npc,
        team,
        history,
        crate::memory::start_new_conversation(),
        cwd,
    );
    state.stream_output = false;

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
pub extern "C" fn npcrs_set_api_key(key_name: *const c_char, key_value: *const c_char) {
    if key_name.is_null() || key_value.is_null() {
        return;
    }
    let name = unsafe { from_c_str(key_name) };
    let value = unsafe { from_c_str(key_value) };
    unsafe { std::env::set_var(&name, &value) };
}

// ── Native tool-calling turn loop ──
//
// Jinxes from the loaded team are compiled into OpenAI-style tool
// definitions and handed to the model's native tool-calling interface. The
// model's emitted tool calls are returned to the host, which executes them
// (via npcd's JinxExecutor) and pushes the results back with
// `npcrs_shell_push_tool_results` to continue the turn.

/// Build the tool definitions offered to the model from the team's jinxes,
/// filtered to `enabled_names` when that set is non-empty.
fn build_tool_defs(state: &ShellState, enabled_names: &[String]) -> Vec<crate::r#gen::ToolDef> {
    let mut defs: Vec<crate::r#gen::ToolDef> = state
        .team
        .jinxes
        .values()
        .filter(|jinx| enabled_names.is_empty() || enabled_names.contains(&jinx.name))
        .filter_map(|jinx| jinx.to_tool_def())
        .collect();
    defs.sort_by(|a, b| a.function.name.cmp(&b.function.name));
    defs
}

/// Serialize one inference round for the host: content plus native tool calls.
fn response_to_turn_json(result: &crate::llm_funcs::LlmResponseResult) -> String {
    let tool_calls: Vec<serde_json::Value> = result
        .tool_calls
        .iter()
        .map(|tc| {
            let arguments = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            serde_json::json!({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": arguments,
            })
        })
        .collect();

    serde_json::json!({
        "content": result.response,
        "tool_calls": tool_calls,
        "done": tool_calls.is_empty(),
    })
    .to_string()
}

/// Store the conversation history from an inference round, dropping the
/// system prompt so it is not accumulated across turns (it is prepended by
/// `get_llm_response` on each call).
fn store_round_messages(state: &mut ShellState, result: crate::llm_funcs::LlmResponseResult) {
    state.messages = result
        .messages
        .into_iter()
        .filter(|m| m.role != "system")
        .collect();
}

/// Run one inference round with the team's jinxes compiled into tools.
fn run_tool_turn(
    state: &mut ShellState,
    input: &str,
    enabled_names: &[String],
) -> crate::llm_funcs::LlmResponseResult {
    let tool_defs = build_tool_defs(state, enabled_names);
    eprintln!(
        "npcrs tool turn: input=\"{}\" tools={:?}",
        input,
        tool_defs
            .iter()
            .map(|t| t.function.name.as_str())
            .collect::<Vec<_>>()
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(crate::llm_funcs::get_llm_response(
        input,
        Some(&state.npc),
        None,
        None,
        Some(&tool_defs),
        &state.messages,
        state.team.context.as_deref(),
    ))
    .unwrap_or_else(|e| crate::llm_funcs::LlmResponseResult {
        response: None,
        response_json: None,
        messages: state.messages.clone(),
        tool_calls: Vec::new(),
        tool_results: Vec::new(),
        usage: None,
        model: String::new(),
        provider: String::new(),
        cost_usd: 0.0,
        error: Some(format!("{}", e)),
        session_id: None,
    })
}

/// Start a tool-calling turn: run inference with jinx tools and return the
/// model's content and/or tool calls as JSON.
///
/// `enabled_jinxes_json` is a JSON array of jinx names; null or `[]` enables
/// every jinx in the team.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_shell_process_command_tools(
    state: *mut ShellState,
    input: *const c_char,
    enabled_jinxes_json: *const c_char,
) -> *mut c_char {
    if state.is_null() || input.is_null() {
        return to_c_string("");
    }
    crate::r#gen::INFERENCE_CANCELLED.store(false, Ordering::Relaxed);

    let state = unsafe { &mut *state };
    let input = unsafe { from_c_str(input) };
    let enabled: Vec<String> = if enabled_jinxes_json.is_null() {
        Vec::new()
    } else {
        serde_json::from_str(&unsafe { from_c_str(enabled_jinxes_json) }).unwrap_or_default()
    };

    state.messages.push(crate::r#gen::Message::user(&input));

    // History already carries the user turn; pass empty input so it is not
    // merged in twice by get_llm_response.
    let result = run_tool_turn(state, "", &enabled);
    if let Some(ref err) = result.error {
        return to_c_string(&serde_json::json!({"error": err}).to_string());
    }
    let json = response_to_turn_json(&result);
    store_round_messages(state, result);
    to_c_string(&json)
}

/// Continue a tool-calling turn after the host has executed the model's tool
/// calls. `results_json` is a JSON array of
/// `{"id": "<tool_call_id>", "name": "<jinx>", "result": "<output>"}`.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_shell_push_tool_results(
    state: *mut ShellState,
    results_json: *const c_char,
    enabled_jinxes_json: *const c_char,
) -> *mut c_char {
    if state.is_null() || results_json.is_null() {
        return to_c_string("");
    }
    crate::r#gen::INFERENCE_CANCELLED.store(false, Ordering::Relaxed);

    let state = unsafe { &mut *state };
    let results_raw = unsafe { from_c_str(results_json) };
    let enabled: Vec<String> = if enabled_jinxes_json.is_null() {
        Vec::new()
    } else {
        serde_json::from_str(&unsafe { from_c_str(enabled_jinxes_json) }).unwrap_or_default()
    };

    let results: Vec<serde_json::Value> = match serde_json::from_str(&results_raw) {
        Ok(v) => v,
        Err(e) => return to_c_string(&serde_json::json!({"error": format!("{}", e)}).to_string()),
    };

    for r in results {
        let id = r.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let name = r.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let content = r.get("result").and_then(|v| v.as_str()).unwrap_or("");
        eprintln!("npcrs tool result: {} -> {}", name, content);
        let mut msg = crate::r#gen::Message::tool_result(id, content);
        msg.name = Some(name.to_string());
        state.messages.push(msg);
    }

    let result = run_tool_turn(state, "", &enabled);
    if let Some(ref err) = result.error {
        return to_c_string(&serde_json::json!({"error": err}).to_string());
    }
    let json = response_to_turn_json(&result);
    store_round_messages(state, result);
    to_c_string(&json)
}

/// Reset the tool-calling conversation history.
#[unsafe(no_mangle)]
pub extern "C" fn npcrs_shell_clear_messages(state: *mut ShellState) {
    if state.is_null() {
        return;
    }
    let state = unsafe { &mut *state };
    state.messages.clear();
}
