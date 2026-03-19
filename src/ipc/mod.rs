
use crate::process::Pid;
use std::collections::HashMap;
use tokio::sync::{broadcast, mpsc};

pub struct IpcBus {
    inboxes: HashMap<Pid, mpsc::Sender<IpcMessage>>,

    signal_tx: broadcast::Sender<Signal>,

    pipes: HashMap<String, mpsc::Sender<Vec<u8>>>,

    shared: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct IpcMessage {
    pub from: Pid,
    pub to: Pid,
    pub kind: MessageKind,
    pub payload: String,
}

#[derive(Debug, Clone)]
pub enum MessageKind {
    Text,
    Delegate,
    DelegateResult,
    Data,
    AgentPass,
}

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
    Wake,
    Interrupt,
    Term,
    Kill,
    Stop,
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

    pub fn register(&mut self, pid: Pid) -> mpsc::Receiver<IpcMessage> {
        let (tx, rx) = mpsc::channel(32);
        self.inboxes.insert(pid, tx);
        rx
    }

    pub async fn send(&self, msg: IpcMessage) -> bool {
        if let Some(tx) = self.inboxes.get(&msg.to) {
            tx.send(msg).await.is_ok()
        } else {
            false
        }
    }

    pub fn signal(&self, signal: Signal) {
        let _ = self.signal_tx.send(signal);
    }

    pub fn subscribe_signals(&self) -> broadcast::Receiver<Signal> {
        self.signal_tx.subscribe()
    }

    pub fn shm_write(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.shared.insert(key.into(), value);
    }

    pub fn shm_read(&self, key: &str) -> Option<&serde_json::Value> {
        self.shared.get(key)
    }

    pub fn create_pipe(&mut self, name: impl Into<String>) -> mpsc::Receiver<Vec<u8>> {
        let (tx, rx) = mpsc::channel(64);
        self.pipes.insert(name.into(), tx);
        rx
    }

    pub async fn pipe_write(&self, name: &str, data: Vec<u8>) -> bool {
        if let Some(tx) = self.pipes.get(name) {
            tx.send(data).await.is_ok()
        } else {
            false
        }
    }

    pub fn unregister(&mut self, pid: Pid) {
        self.inboxes.remove(&pid);
    }
}

impl Default for IpcBus {
    fn default() -> Self {
        Self::new()
    }
}
