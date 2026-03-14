//! Process scheduler — manages which NPC gets "CPU time" (LLM attention).
//!
//! In a traditional OS, the scheduler decides which process runs next.
//! Here, the scarce resource is LLM inference — so the scheduler manages:
//! - Priority queues (which NPC gets to query the LLM next)
//! - Cron jobs (scheduled NPC activations)
//! - Rate limiting (fair sharing of token budgets)

use crate::process::Pid;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;

/// Process scheduler.
#[derive(Debug)]
pub struct Scheduler {
    /// Ready queue: processes waiting to run (FIFO with priority).
    ready_queue: VecDeque<SchedulerEntry>,

    /// Cron table: scheduled process activations.
    cron_table: Vec<CronEntry>,
}

#[derive(Debug, Clone)]
struct SchedulerEntry {
    pid: Pid,
    priority: Priority,
    enqueued_at: DateTime<Utc>,
}

/// Process priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Background work (knowledge graph updates, indexing).
    Low = 0,
    /// Normal interactive requests.
    Normal = 1,
    /// High priority (user-facing, latency-sensitive).
    High = 2,
    /// Real-time (system-critical, init process).
    Realtime = 3,
}

/// A scheduled (cron) job.
#[derive(Debug, Clone)]
pub struct CronEntry {
    pub pid: Pid,
    pub schedule: CronSchedule,
    pub command: String,
    pub last_run: Option<DateTime<Utc>>,
    pub next_run: DateTime<Utc>,
    pub enabled: bool,
}

/// Simple cron schedule (interval-based for now).
#[derive(Debug, Clone)]
pub enum CronSchedule {
    /// Run every N seconds.
    Interval(u64),
    /// Run once at a specific time.
    Once(DateTime<Utc>),
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            ready_queue: VecDeque::new(),
            cron_table: Vec::new(),
        }
    }

    /// Add a process to the ready queue with normal priority.
    pub fn enqueue(&mut self, pid: Pid) {
        self.enqueue_with_priority(pid, Priority::Normal);
    }

    /// Add a process to the ready queue with a specific priority.
    pub fn enqueue_with_priority(&mut self, pid: Pid, priority: Priority) {
        // Remove duplicate if already in queue
        self.ready_queue.retain(|e| e.pid != pid);

        let entry = SchedulerEntry {
            pid,
            priority,
            enqueued_at: Utc::now(),
        };

        // Insert by priority (higher priority closer to front)
        let pos = self
            .ready_queue
            .iter()
            .position(|e| e.priority < priority)
            .unwrap_or(self.ready_queue.len());

        self.ready_queue.insert(pos, entry);
    }

    /// Dequeue the next process to run.
    pub fn next(&mut self) -> Option<Pid> {
        self.ready_queue.pop_front().map(|e| e.pid)
    }

    /// Peek at the next process without removing it.
    pub fn peek(&self) -> Option<Pid> {
        self.ready_queue.front().map(|e| e.pid)
    }

    /// Number of processes in the ready queue.
    pub fn queue_len(&self) -> usize {
        self.ready_queue.len()
    }

    /// Add a cron job.
    pub fn add_cron(
        &mut self,
        pid: Pid,
        schedule: CronSchedule,
        command: String,
    ) {
        let next_run = match &schedule {
            CronSchedule::Interval(secs) => {
                Utc::now() + chrono::Duration::seconds(*secs as i64)
            }
            CronSchedule::Once(at) => *at,
        };

        self.cron_table.push(CronEntry {
            pid,
            schedule,
            command,
            last_run: None,
            next_run,
            enabled: true,
        });
    }

    /// Check for due cron jobs and return them.
    pub fn check_cron(&mut self) -> Vec<(Pid, String)> {
        let now = Utc::now();
        let mut due = Vec::new();

        for entry in &mut self.cron_table {
            if !entry.enabled {
                continue;
            }
            if now >= entry.next_run {
                due.push((entry.pid, entry.command.clone()));
                entry.last_run = Some(now);

                // Schedule next run
                match &entry.schedule {
                    CronSchedule::Interval(secs) => {
                        entry.next_run = now + chrono::Duration::seconds(*secs as i64);
                    }
                    CronSchedule::Once(_) => {
                        entry.enabled = false;
                    }
                }
            }
        }

        due
    }

    /// List all cron entries.
    pub fn cron_entries(&self) -> &[CronEntry] {
        &self.cron_table
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}
