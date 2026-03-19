//! Job scheduling — mirrors npcpy.work.plan
//!
//! SQLite-backed storage plus platform-specific crontab/launchd scheduling
//! and LLM-driven plan command.

use crate::error::{NpcError, Result};
use rusqlite::{params, Connection};
use std::collections::HashMap;

// ── Helpers ──

fn jobs_dir() -> String {
    shellexpand::tilde("~/.npcsh/jobs").to_string()
}

fn logs_dir() -> String {
    shellexpand::tilde("~/.npcsh/logs").to_string()
}

fn npc_bin_path() -> String {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let candidate = parent.join("npc");
            if candidate.exists() {
                return candidate.to_string_lossy().to_string();
            }
        }
    }
    let output = std::process::Command::new("which")
        .arg("npc")
        .output();
    match output {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        }
        _ => "npc".to_string(),
    }
}

fn plist_path(job_name: &str) -> String {
    shellexpand::tilde(&format!(
        "~/Library/LaunchAgents/com.npcsh.job.{}.plist",
        job_name
    ))
    .to_string()
}

fn cron_tag(job_name: &str) -> String {
    format!("# npcsh:{}", job_name)
}

// ── Data model ──

#[derive(Debug, Clone)]
pub struct Job {
    pub name: String,
    pub cron_expr: String,
    pub command: String,
    pub last_run: Option<String>,
    pub next_run: Option<String>,
    pub status: String,
}

// ── SQLite-backed storage ──

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

pub fn list_jobs_db(db_path: &str) -> Result<Vec<Job>> {
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

// ── OS-level scheduling ──

/// Compile a command into a self-contained executable bash script.
pub fn compile_job_script(command: &str, job_name: &str) -> Result<String> {
    let dir = jobs_dir();
    std::fs::create_dir_all(&dir).map_err(|e| NpcError::FileLoad {
        path: dir.clone(),
        source: e,
    })?;
    let script_path = format!("{}/{}.sh", dir, job_name);
    let npc = npc_bin_path();
    let content = format!(
        "#!/bin/bash\n# npcsh job: {}\nset -euo pipefail\n\n{} {}\n",
        job_name,
        npc,
        command.trim_start_matches('/')
    );
    std::fs::write(&script_path, &content).map_err(|e| NpcError::FileLoad {
        path: script_path.clone(),
        source: e,
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755));
    }
    Ok(script_path)
}

/// Schedule a job using the OS scheduler.
pub fn schedule_job_os(schedule: &str, command: &str, job_name: &str) -> (bool, String) {
    let log_dir = logs_dir();
    let _ = std::fs::create_dir_all(&log_dir);
    let script_path = match compile_job_script(command, job_name) {
        Ok(p) => p,
        Err(e) => return (false, format!("Failed to compile script: {}", e)),
    };
    let log_path = format!("{}/{}.log", log_dir, job_name);

    let os = std::env::consts::OS;
    match os {
        "macos" => _schedule_launchd(&script_path, schedule, job_name, &log_path),
        _ => _schedule_crontab(&script_path, schedule, job_name, &log_path),
    }
}

/// Unschedule a job from the OS scheduler.
pub fn unschedule_job_os(job_name: &str) -> (bool, String) {
    let os = std::env::consts::OS;
    match os {
        "macos" => _unschedule_launchd(job_name),
        _ => _unschedule_crontab(job_name),
    }
}

/// List all scheduled jobs from the OS scheduler.
pub fn list_jobs() -> Vec<HashMap<String, serde_json::Value>> {
    let os = std::env::consts::OS;
    let mut jobs = Vec::new();

    match os {
        "macos" => {
            let agents = shellexpand::tilde("~/Library/LaunchAgents/").to_string();
            if let Ok(entries) = std::fs::read_dir(&agents) {
                for entry in entries.flatten() {
                    let fname = entry.file_name().to_string_lossy().to_string();
                    if fname.starts_with("com.npcsh.job.") && fname.ends_with(".plist") {
                        let name = fname
                            .replace("com.npcsh.job.", "")
                            .replace(".plist", "");
                        let label = format!("com.npcsh.job.{}", name);
                        let active = std::process::Command::new("launchctl")
                            .args(["list", &label])
                            .output()
                            .map(|o| o.status.success())
                            .unwrap_or(false);
                        let mut job = HashMap::new();
                        job.insert("name".into(), serde_json::Value::String(name));
                        job.insert("active".into(), serde_json::Value::Bool(active));
                        jobs.push(job);
                    }
                }
            }
        }
        _ => {
            let output = std::process::Command::new("crontab")
                .arg("-l")
                .output();
            if let Ok(out) = output {
                if out.status.success() {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    for line in stdout.lines() {
                        if let Some(pos) = line.find("# npcsh:") {
                            let name = line[pos + 8..].trim().to_string();
                            let mut job = HashMap::new();
                            job.insert("name".into(), serde_json::Value::String(name));
                            job.insert("active".into(), serde_json::Value::Bool(true));
                            jobs.push(job);
                        }
                    }
                }
            }
        }
    }

    jobs
}

/// Quick check whether a job is currently scheduled.
pub fn job_is_active(job_name: &str) -> bool {
    let os = std::env::consts::OS;
    match os {
        "macos" => std::path::Path::new(&plist_path(job_name)).exists(),
        _ => {
            let output = std::process::Command::new("crontab")
                .arg("-l")
                .output();
            match output {
                Ok(out) if out.status.success() => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let tag = cron_tag(job_name);
                    stdout.lines().any(|l| l.contains(&tag))
                }
                _ => false,
            }
        }
    }
}

/// Detailed status dict for a job.
pub fn job_status(job_name: &str) -> HashMap<String, serde_json::Value> {
    let log_path = format!("{}/{}.log", logs_dir(), job_name);
    let mut info = HashMap::new();
    info.insert("name".into(), serde_json::Value::String(job_name.to_string()));
    info.insert("active".into(), serde_json::Value::Bool(job_is_active(job_name)));
    info.insert("log".into(), serde_json::Value::String(log_path.clone()));

    let recent_log = if std::path::Path::new(&log_path).exists() {
        std::fs::read_to_string(&log_path)
            .ok()
            .map(|content| {
                let lines: Vec<&str> = content.lines().collect();
                let start = lines.len().saturating_sub(10);
                lines[start..].join("\n")
            })
            .unwrap_or_default()
    } else {
        String::new()
    };
    info.insert("recent_log".into(), serde_json::Value::String(recent_log));
    info
}

// ── Platform-specific ──

pub fn _schedule_crontab(
    script_path: &str,
    schedule: &str,
    job_name: &str,
    log_path: &str,
) -> (bool, String) {
    let output = std::process::Command::new("crontab")
        .arg("-l")
        .output();
    let existing = match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).to_string(),
        _ => String::new(),
    };

    let tag = cron_tag(job_name);
    let entry = format!("{} {} >> {} 2>&1 {}", schedule, script_path, log_path, tag);
    let new_crontab = format!("{}\n{}\n", existing.trim_end(), entry);

    let result = std::process::Command::new("crontab")
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                stdin.write_all(new_crontab.as_bytes())?;
            }
            child.wait_with_output()
        });

    match result {
        Ok(out) if out.status.success() => {
            (true, format!("Scheduled \"{}\": {}", job_name, schedule))
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            (false, format!("Failed: {}", stderr))
        }
        Err(e) => (false, format!("Failed: {}", e)),
    }
}

pub fn _unschedule_crontab(job_name: &str) -> (bool, String) {
    let output = std::process::Command::new("crontab")
        .arg("-l")
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let tag = cron_tag(job_name);
            let filtered: Vec<&str> = stdout.lines().filter(|l| !l.contains(&tag)).collect();
            let new_crontab = format!("{}\n", filtered.join("\n"));

            let result = std::process::Command::new("crontab")
                .arg("-")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .and_then(|mut child| {
                    use std::io::Write;
                    if let Some(ref mut stdin) = child.stdin {
                        stdin.write_all(new_crontab.as_bytes())?;
                    }
                    child.wait_with_output()
                });

            match result {
                Ok(out2) if out2.status.success() => (true, format!("Removed \"{}\"", job_name)),
                Ok(out2) => {
                    let stderr = String::from_utf8_lossy(&out2.stderr);
                    (false, format!("Failed: {}", stderr))
                }
                Err(e) => (false, format!("Failed: {}", e)),
            }
        }
        _ => (false, "No crontab found.".to_string()),
    }
}

pub fn _schedule_launchd(
    script_path: &str,
    schedule: &str,
    job_name: &str,
    log_path: &str,
) -> (bool, String) {
    let ppath = plist_path(job_name);
    if let Some(parent) = std::path::Path::new(&ppath).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let parts: Vec<&str> = schedule.split_whitespace().collect();
    let mut plist = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\">\n<dict>\n\
         <key>Label</key>\n<string>com.npcsh.job.{}</string>\n\
         <key>ProgramArguments</key>\n<array>\n<string>{}</string>\n</array>\n",
        job_name, script_path
    );

    if parts.len() == 5 {
        plist.push_str("<key>StartCalendarInterval</key>\n<dict>\n");
        let keys = ["Minute", "Hour", "Day", "Month", "Weekday"];
        for (i, key) in keys.iter().enumerate() {
            if parts[i] != "*" {
                plist.push_str(&format!(
                    "<key>{}</key>\n<integer>{}</integer>\n",
                    key, parts[i]
                ));
            }
        }
        plist.push_str("</dict>\n");
    } else if let Ok(interval) = schedule.parse::<u64>() {
        plist.push_str(&format!(
            "<key>StartInterval</key>\n<integer>{}</integer>\n",
            interval
        ));
    }

    plist.push_str(&format!(
        "<key>StandardOutPath</key>\n<string>{}</string>\n\
         <key>StandardErrorPath</key>\n<string>{}</string>\n\
         </dict>\n</plist>\n",
        log_path, log_path
    ));

    if let Err(e) = std::fs::write(&ppath, &plist) {
        return (false, format!("Failed to write plist: {}", e));
    }

    let _ = std::process::Command::new("launchctl")
        .args(["load", &ppath])
        .output();

    (true, format!("Scheduled \"{}\": {}", job_name, schedule))
}

pub fn _unschedule_launchd(job_name: &str) -> (bool, String) {
    let ppath = plist_path(job_name);
    if std::path::Path::new(&ppath).exists() {
        let _ = std::process::Command::new("launchctl")
            .args(["unload", &ppath])
            .output();
        let _ = std::fs::remove_file(&ppath);
        (true, format!("Removed \"{}\"", job_name))
    } else {
        (false, format!("Job \"{}\" not found.", job_name))
    }
}

// ── LLM-driven /plan command ──

pub async fn execute_plan_command(
    command: &str,
    model: &str,
    provider: &str,
    _messages: &[crate::r#gen::Message],
) -> Result<HashMap<String, String>> {
    let parts: Vec<&str> = command.splitn(2, ' ').collect();
    if parts.len() < 2 {
        let mut result = HashMap::new();
        result.insert("output".into(), "Usage: /plan <command and schedule description>".into());
        return Ok(result);
    }

    let request = parts[1];
    let os = std::env::consts::OS;

    let prompt = if os == "macos" {
        format!(
            "Convert this scheduling request into a launchd-compatible script:\n\
             Request: {}\n\n\
             Your response must be valid json with keys: script, schedule (interval in seconds), description, name.\n\
             Do not include any markdown formatting or leading ```json tags.",
            request
        )
    } else {
        format!(
            "Convert this scheduling request into a crontab-based script:\n\
             Request: {}\n\n\
             Your response must be valid json with keys: script, schedule (crontab 5 fields), description, name.\n\
             Do not include any markdown formatting or leading ```json tags.",
            request
        )
    };

    let msgs = vec![
        crate::r#gen::Message::system("You are a helpful scheduling assistant."),
        crate::r#gen::Message::user(&prompt),
    ];

    let response = crate::r#gen::get_genai_response(provider, model, &msgs, None, None).await?;

    let response_text = response.message.content.unwrap_or_default();
    let clean_text = response_text.replace("```json", "").replace("```", "").trim().to_string();

    let schedule_info: serde_json::Value = serde_json::from_str(&clean_text).map_err(|e| {
        NpcError::Shell(format!("Failed to parse plan response as JSON: {}", e))
    })?;

    let job_name = format!("job_{}", schedule_info["name"].as_str().unwrap_or("unnamed"));
    let sched_str = schedule_info["schedule"].as_str()
        .map(String::from)
        .unwrap_or_else(|| schedule_info["schedule"].to_string());

    let jdir = jobs_dir();
    let ldir = logs_dir();
    let _ = std::fs::create_dir_all(&jdir);
    let _ = std::fs::create_dir_all(&ldir);

    let script_path = format!("{}/{}.sh", jdir, job_name);
    let log_path = format!("{}/{}.log", ldir, job_name);

    let script_content = schedule_info["script"].as_str().unwrap_or("#!/bin/bash\necho 'no script'");
    std::fs::write(&script_path, script_content).map_err(|e| NpcError::FileLoad {
        path: script_path.clone(),
        source: e,
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755));
    }

    let (_ok, _msg) = if os == "macos" {
        _schedule_launchd(&script_path, &sched_str, &job_name, &log_path)
    } else {
        _schedule_crontab(&script_path, &sched_str, &job_name, &log_path)
    };

    let description = schedule_info["description"].as_str().unwrap_or("(no description)");
    let output = format!(
        "Job created successfully:\n- Description: {}\n- Schedule: {}\n- Script: {}\n- Log: {}",
        description, sched_str, script_path, log_path
    );

    let mut result = HashMap::new();
    result.insert("output".into(), output);
    Ok(result)
}
