//! Job scheduling — cron-like job management stored in SQLite.

use crate::error::Result;
use rusqlite::{params, Connection};

#[derive(Debug, Clone)]
pub struct Job {
    pub name: String,
    pub cron_expr: String,
    pub command: String,
    pub last_run: Option<String>,
    pub next_run: Option<String>,
    pub status: String,
}

pub fn init_jobs_table(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS npc_jobs (
            name TEXT PRIMARY KEY,
            cron_expr TEXT NOT NULL,
            command TEXT NOT NULL,
            last_run TEXT,
            next_run TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL
        );"
    )?;
    Ok(())
}

pub fn schedule_job(db_path: &str, name: &str, cron_expr: &str, command: &str) -> Result<()> {
    let conn = Connection::open(db_path)?;
    init_jobs_table(&conn)?;
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT OR REPLACE INTO npc_jobs (name, cron_expr, command, status, created_at) VALUES (?1, ?2, ?3, 'active', ?4)",
        params![name, cron_expr, command, now],
    )?;
    Ok(())
}

pub fn unschedule_job(db_path: &str, name: &str) -> Result<()> {
    let conn = Connection::open(db_path)?;
    conn.execute("DELETE FROM npc_jobs WHERE name = ?1", params![name])?;
    Ok(())
}

pub fn list_jobs(db_path: &str) -> Result<Vec<Job>> {
    let conn = Connection::open(db_path)?;
    init_jobs_table(&conn)?;
    let mut stmt = conn.prepare("SELECT name, cron_expr, command, last_run, next_run, status FROM npc_jobs ORDER BY name")?;
    let jobs = stmt.query_map([], |row| {
        Ok(Job {
            name: row.get(0)?,
            cron_expr: row.get(1)?,
            command: row.get(2)?,
            last_run: row.get(3)?,
            next_run: row.get(4)?,
            status: row.get(5)?,
        })
    })?.filter_map(|r| r.ok()).collect();
    Ok(jobs)
}
