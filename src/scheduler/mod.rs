
use crate::process::Pid;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;

#[derive(Debug)]
pub struct Scheduler {
    ready_queue: VecDeque<SchedulerEntry>,

    cron_table: Vec<CronEntry>,
}

#[derive(Debug, Clone)]
struct SchedulerEntry {
    pid: Pid,
    priority: Priority,
    enqueued_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Realtime = 3,
}

#[derive(Debug, Clone)]
pub struct CronEntry {
    pub pid: Pid,
    pub schedule: CronSchedule,
    pub command: String,
    pub last_run: Option<DateTime<Utc>>,
    pub next_run: DateTime<Utc>,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub enum CronSchedule {
    Interval(u64),
    Once(DateTime<Utc>),
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            ready_queue: VecDeque::new(),
            cron_table: Vec::new(),
        }
    }

    pub fn enqueue(&mut self, pid: Pid) {
        self.enqueue_with_priority(pid, Priority::Normal);
    }

    pub fn enqueue_with_priority(&mut self, pid: Pid, priority: Priority) {
        self.ready_queue.retain(|e| e.pid != pid);

        let entry = SchedulerEntry {
            pid,
            priority,
            enqueued_at: Utc::now(),
        };

        let pos = self
            .ready_queue
            .iter()
            .position(|e| e.priority < priority)
            .unwrap_or(self.ready_queue.len());

        self.ready_queue.insert(pos, entry);
    }

    pub fn next(&mut self) -> Option<Pid> {
        self.ready_queue.pop_front().map(|e| e.pid)
    }

    pub fn peek(&self) -> Option<Pid> {
        self.ready_queue.front().map(|e| e.pid)
    }

    pub fn queue_len(&self) -> usize {
        self.ready_queue.len()
    }

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

    pub fn cron_entries(&self) -> &[CronEntry] {
        &self.cron_table
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}
