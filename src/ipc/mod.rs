//! Inter-Process Communication — how NPC processes talk to each other.
//!
//! Supported IPC mechanisms:
//! - **Signals**: Simple notifications (wake, kill, pause)
//! - **Pipes**: Stream data between processes (stdout of one → stdin of another)
//! - **Shared memory**: The team's shared_context (read/write with locking)
//! - **Message passing**: Typed messages between PIDs

use crate::process::Pid;
use std::collections::HashMap;
use tokio::sync::{broadcast, mpsc};

/// IPC bus managing all inter-process communication.
pub struct IpcBus {
    /// Per-process message inboxes.
    inboxes: HashMap<Pid, mpsc::Sender<IpcMessage>>,

    /// Broadcast channel for signals (all processes receive).
    signal_tx: broadcast::Sender<Signal>,

    /// Named pipes: name → (sender, receiver).
    pipes: HashMap<String, mpsc::Sender<Vec<u8>>>,

    /// Shared memory segments.
    shared: HashMap<String, serde_json::Value>,
}

/// A typed IPC message between processes.
#[derive(Debug, Clone)]
pub struct IpcMessage {
    pub from: Pid,
    pub to: Pid,
    pub kind: MessageKind,
    pub payload: String,
}

/// Message types.
#[derive(Debug, Clone)]
pub enum MessageKind {
    /// Free-form text (like a chat message between NPCs).
    Text,
    /// Delegation request: "please handle this task".
    Delegate,
    /// Delegation response: "here's the result".
    DelegateResult,
    /// Data transfer (structured JSON).
    Data,
    /// Agent pass: hand off the conversation.
    AgentPass,
}

/// Signals that can be sent to processes.
#[derive(Debug, Clone)]
pub struct Signal {
    pub target: SignalTarget,
    pub kind: SignalKind,
    pub from: Pid,
}

#[derive(Debug, Clone)]
pub enum SignalTarget {
    Process(Pid),
    All,
}

#[derive(Debug, Clone, Copy)]
pub enum SignalKind {
    /// Wake a sleeping process.
    Wake,
    /// Interrupt current operation.
    Interrupt,
    /// Terminate gracefully.
    Term,
    /// Kill immediately.
    Kill,
    /// Pause execution.
    Stop,
    /// Resume execution.
    Continue,
}

impl IpcBus {
    pub fn new() -> Self {
        let (signal_tx, _) = broadcast::channel(64);
        Self {
            inboxes: HashMap::new(),
            signal_tx,
            pipes: HashMap::new(),
            shared: HashMap::new(),
        }
    }

    /// Register a process's inbox.
    pub fn register(&mut self, pid: Pid) -> mpsc::Receiver<IpcMessage> {
        let (tx, rx) = mpsc::channel(32);
        self.inboxes.insert(pid, tx);
        rx
    }

    /// Send a message to a process.
    pub async fn send(&self, msg: IpcMessage) -> bool {
        if let Some(tx) = self.inboxes.get(&msg.to) {
            tx.send(msg).await.is_ok()
        } else {
            false
        }
    }

    /// Broadcast a signal.
    pub fn signal(&self, signal: Signal) {
        let _ = self.signal_tx.send(signal);
    }

    /// Subscribe to signals.
    pub fn subscribe_signals(&self) -> broadcast::Receiver<Signal> {
        self.signal_tx.subscribe()
    }

    /// Write to shared memory.
    pub fn shm_write(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.shared.insert(key.into(), value);
    }

    /// Read from shared memory.
    pub fn shm_read(&self, key: &str) -> Option<&serde_json::Value> {
        self.shared.get(key)
    }

    /// Create a named pipe.
    pub fn create_pipe(&mut self, name: impl Into<String>) -> mpsc::Receiver<Vec<u8>> {
        let (tx, rx) = mpsc::channel(64);
        self.pipes.insert(name.into(), tx);
        rx
    }

    /// Write to a named pipe.
    pub async fn pipe_write(&self, name: &str, data: Vec<u8>) -> bool {
        if let Some(tx) = self.pipes.get(name) {
            tx.send(data).await.is_ok()
        } else {
            false
        }
    }

    /// Unregister a process (cleanup on death).
    pub fn unregister(&mut self, pid: Pid) {
        self.inboxes.remove(&pid);
    }
}

impl Default for IpcBus {
    fn default() -> Self {
        Self::new()
    }
}
