//! NPC Process — an agent as an OS process.
//!
//! Each NPC runs as a process with:
//! - A PID (unique identifier)
//! - Its own memory space (conversation history, knowledge)
//! - Resource limits (token budget, cost cap, max tool calls)
//! - A capability set (which syscalls/jinxes it can invoke)
//! - Lifecycle state (spawned → running → blocked → zombie → dead)
//! - File descriptors (stdin/stdout/stderr as message channels)

use crate::npc::Npc;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

/// Process ID.
pub type Pid = u32;

/// An NPC process in the kernel.
pub struct Process {
    /// Unique process ID.
    pub pid: Pid,

    /// Parent PID (0 = init).
    pub ppid: Pid,

    /// The underlying NPC agent.
    pub npc: Npc,

    /// Current state.
    pub state: ProcessState,

    /// Capabilities — which jinxes/syscalls this process can invoke.
    pub capabilities: Capabilities,

    /// Resource limits.
    pub limits: ResourceLimits,

    /// Resource usage counters.
    pub usage: ResourceUsage,

    /// Process environment variables.
    pub env: HashMap<String, String>,

    /// Working directory (in the VFS).
    pub cwd: String,

    /// Message history (process-local memory).
    pub messages: Vec<crate::llm::Message>,

    /// File descriptors: fd → channel.
    pub fds: HashMap<u32, FileDescriptor>,

    /// When this process was spawned.
    pub spawned_at: DateTime<Utc>,

    /// When this process last ran.
    pub last_active: DateTime<Utc>,

    /// Exit code (set when state = Dead).
    pub exit_code: Option<i32>,

    /// Conversation ID (UUID string, matches npcpy).
    pub conversation_id: String,
}

/// Process lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessState {
    /// Created but not yet scheduled.
    Spawned,
    /// Actively running (has the "CPU" = LLM attention).
    Running,
    /// Waiting for I/O (LLM response, tool result, user input).
    Blocked,
    /// Sleeping (cron/scheduled, will wake at a specific time).
    Sleeping,
    /// Finished but parent hasn't collected exit status.
    Zombie,
    /// Fully terminated.
    Dead,
}

/// Capability set — controls what a process can do.
#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    /// Allowed jinx names (empty = all allowed).
    pub allowed_jinxes: HashSet<String>,

    /// Whether this process can spawn child processes.
    pub can_spawn: bool,

    /// Whether this process can access the real filesystem.
    pub can_fs: bool,

    /// Whether this process can execute bash commands.
    pub can_bash: bool,

    /// Whether this process can make network requests.
    pub can_network: bool,

    /// Whether this process can delegate to other NPCs.
    pub can_delegate: bool,

    /// Maximum delegation depth.
    pub max_delegation_depth: u32,

    /// Whether this process can modify its own capabilities (root).
    pub is_superuser: bool,
}

impl Capabilities {
    /// Full capabilities (superuser / init process).
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

    /// Restricted capabilities (sandboxed process).
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

    /// Check if a jinx is allowed.
    pub fn can_run_jinx(&self, name: &str) -> bool {
        self.is_superuser || self.allowed_jinxes.is_empty() || self.allowed_jinxes.contains(name)
    }
}

/// Resource limits for a process.
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Max input tokens per turn.
    pub max_input_tokens_per_turn: Option<u64>,

    /// Max output tokens per turn.
    pub max_output_tokens_per_turn: Option<u64>,

    /// Total token budget for this process's lifetime.
    pub total_token_budget: Option<u64>,

    /// Max cost in USD for this process.
    pub max_cost_usd: Option<f64>,

    /// Max number of tool calls per turn.
    pub max_tool_calls_per_turn: Option<u32>,

    /// Max number of turns before forced termination.
    pub max_turns: Option<u64>,

    /// Max wall-clock time before kill (seconds).
    pub max_runtime_secs: Option<u64>,

    /// Max concurrent child processes.
    pub max_children: Option<u32>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_input_tokens_per_turn: None,
            max_output_tokens_per_turn: None,
            total_token_budget: None,
            max_cost_usd: None,
            max_tool_calls_per_turn: Some(20),
            max_turns: None,
            max_runtime_secs: None,
            max_children: Some(10),
        }
    }
}

/// Tracked resource usage.
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
    /// Check if any limit has been exceeded.
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

/// A file descriptor for IPC.
pub enum FileDescriptor {
    /// Standard channel (stdin=0, stdout=1, stderr=2).
    Pipe(Arc<Mutex<mpsc::Sender<String>>>),
    /// Null device (/dev/null equivalent).
    Null,
    /// File on the VFS.
    File { path: String, writable: bool },
}

impl Process {
    /// Spawn a new process from an NPC.
    pub fn spawn(pid: Pid, ppid: Pid, npc: Npc, capabilities: Capabilities) -> Self {
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

    /// Check if this process can invoke a jinx.
    pub fn can_invoke(&self, jinx_name: &str) -> bool {
        if self.state != ProcessState::Running {
            return false;
        }
        self.capabilities.can_run_jinx(jinx_name)
    }

    /// Record token usage for this turn.
    pub fn record_usage(&mut self, input_tokens: u64, output_tokens: u64, cost: f64) {
        self.usage.total_input_tokens += input_tokens;
        self.usage.total_output_tokens += output_tokens;
        self.usage.total_cost_usd += cost;
        self.last_active = Utc::now();
    }

    /// Start a new turn.
    pub fn new_turn(&mut self) {
        self.usage.total_turns += 1;
        self.usage.tool_calls_this_turn = 0;
    }

    /// Kill this process.
    pub fn kill(&mut self, exit_code: i32) {
        self.state = ProcessState::Dead;
        self.exit_code = Some(exit_code);
    }

    /// Signal summary for display.
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
