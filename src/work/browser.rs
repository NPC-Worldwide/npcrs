//! Browser session management — mirrors npcpy.work.browser

use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
pub struct BrowserSession {
    pub session_id: String,
    pub url: Option<String>,
}

struct BrowserState {
    sessions: HashMap<String, BrowserSession>,
    current: Option<String>,
}

static SESSIONS: Lazy<Mutex<BrowserState>> = Lazy::new(|| Mutex::new(BrowserState { sessions: HashMap::new(), current: None }));

pub fn get_sessions() -> Vec<String> { SESSIONS.lock().unwrap().sessions.keys().cloned().collect() }

pub fn get_current_driver() -> Option<BrowserSession> {
    let state = SESSIONS.lock().unwrap();
    state.current.as_ref().and_then(|id| state.sessions.get(id)).cloned()
}

pub fn set_driver(session_id: &str, session: BrowserSession) {
    let mut state = SESSIONS.lock().unwrap();
    state.sessions.insert(session_id.to_string(), session);
    state.current = Some(session_id.to_string());
}

pub fn close_current() -> bool {
    let mut state = SESSIONS.lock().unwrap();
    if let Some(id) = state.current.take() { state.sessions.remove(&id); true } else { false }
}
