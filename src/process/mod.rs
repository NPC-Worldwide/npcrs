
use crate::npc_compiler::NPC;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

pub type Pid = u32;

pub struct Process {
    pub pid: Pid,

    pub ppid: Pid,

    pub npc: NPC,

    pub state: ProcessState,

    pub capabilities: Capabilities,

    pub limits: ResourceLimits,

    pub usage: ResourceUsage,

    pub env: HashMap<String, String>,

    pub cwd: String,

    pub messages: Vec<crate::r#gen::Message>,

    pub fds: HashMap<u32, FileDescriptor>,

    pub spawned_at: DateTime<Utc>,

    pub last_active: DateTime<Utc>,

    pub exit_code: Option<i32>,

    pub conversation_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessState {
    Spawned,
    Running,
    Blocked,
    Sleeping,
    Zombie,
    Dead,
}

#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    pub allowed_jinxes: HashSet<String>,

    pub can_spawn: bool,

    pub can_fs: bool,

    pub can_bash: bool,

    pub can_network: bool,

    pub can_delegate: bool,

    pub max_delegation_depth: u32,

    pub is_superuser: bool,
}

impl Capabilities {
    pub fn root() -> Self {
        Self {
            allowed_jinxes: HashSet::new(), // empty = all
            can_spawn: true,
            can_fs: true,
            can_bash: true,
            can_network: true,
            can_delegate: true,
            max_delegation_depth: 10,
            is_superuser: true,
        }
    }

    pub fn sandboxed() -> Self {
        Self {
            allowed_jinxes: HashSet::new(),
            can_spawn: false,
            can_fs: false,
            can_bash: false,
            can_network: true, // needs LLM access
            can_delegate: false,
            max_delegation_depth: 0,
            is_superuser: false,
        }
    }

    pub fn can_run_jinx(&self, name: &str) -> bool {
        self.is_superuser || self.allowed_jinxes.is_empty() || self.allowed_jinxes.contains(name)
    }
}

#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_input_tokens_per_turn: Option<u64>,

    pub max_output_tokens_per_turn: Option<u64>,

    pub total_token_budget: Option<u64>,

    pub max_cost_usd: Option<f64>,

    pub max_tool_calls_per_turn: Option<u32>,

    pub max_turns: Option<u64>,

    pub max_runtime_secs: Option<u64>,

    pub max_children: Option<u32>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_input_tokens_per_turn: None,
            max_output_tokens_per_turn: None,
            total_token_budget: None,
            max_cost_usd: None,
            max_tool_calls_per_turn: None,
            max_turns: None,
            max_runtime_secs: None,
            max_children: Some(10),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cost_usd: f64,
    pub total_tool_calls: u64,
    pub total_turns: u64,
    pub tool_calls_this_turn: u32,
}

impl ResourceUsage {
    pub fn exceeds(&self, limits: &ResourceLimits) -> Option<String> {
        if let Some(budget) = limits.total_token_budget {
            let total = self.total_input_tokens + self.total_output_tokens;
            if total >= budget {
                return Some(format!(
                    "Token budget exhausted: {} / {}",
                    total, budget
                ));
            }
        }
        if let Some(max_cost) = limits.max_cost_usd {
            if self.total_cost_usd >= max_cost {
                return Some(format!(
                    "Cost limit reached: ${:.4} / ${:.4}",
                    self.total_cost_usd, max_cost
                ));
            }
        }
        if let Some(max_turns) = limits.max_turns {
            if self.total_turns >= max_turns {
                return Some(format!(
                    "Turn limit reached: {} / {}",
                    self.total_turns, max_turns
                ));
            }
        }
        if let Some(max_tc) = limits.max_tool_calls_per_turn {
            if self.tool_calls_this_turn >= max_tc {
                return Some(format!(
                    "Tool call limit per turn: {} / {}",
                    self.tool_calls_this_turn, max_tc
                ));
            }
        }
        None
    }
}

pub enum FileDescriptor {
    Pipe(Arc<Mutex<mpsc::Sender<String>>>),
    Null,
    File { path: String, writable: bool },
}

impl Process {
    pub fn spawn(pid: Pid, ppid: Pid, npc: NPC, capabilities: Capabilities) -> Self {
        let now = Utc::now();
        Self {
            pid,
            ppid,
            npc,
            state: ProcessState::Spawned,
            capabilities,
            limits: ResourceLimits::default(),
            usage: ResourceUsage::default(),
            env: HashMap::new(),
            cwd: "/".to_string(),
            messages: Vec::new(),
            fds: HashMap::new(),
            spawned_at: now,
            last_active: now,
            exit_code: None,
            conversation_id: crate::memory::start_new_conversation(),
        }
    }

    pub fn can_invoke(&self, jinx_name: &str) -> bool {
        if self.state != ProcessState::Running {
            return false;
        }
        self.capabilities.can_run_jinx(jinx_name)
    }

    pub fn record_usage(&mut self, input_tokens: u64, output_tokens: u64, cost: f64) {
        self.usage.total_input_tokens += input_tokens;
        self.usage.total_output_tokens += output_tokens;
        self.usage.total_cost_usd += cost;
        self.last_active = Utc::now();
    }

    pub fn new_turn(&mut self) {
        self.usage.total_turns += 1;
        self.usage.tool_calls_this_turn = 0;
    }

    pub fn kill(&mut self, exit_code: i32) {
        self.state = ProcessState::Dead;
        self.exit_code = Some(exit_code);
    }

    pub fn status_line(&self) -> String {
        format!(
            "[pid:{} {} {:?}] tokens:{}/{} cost:${:.4} turns:{}",
            self.pid,
            self.npc.name,
            self.state,
            self.usage.total_input_tokens,
            self.usage.total_output_tokens,
            self.usage.total_cost_usd,
            self.usage.total_turns,
        )
    }
}
