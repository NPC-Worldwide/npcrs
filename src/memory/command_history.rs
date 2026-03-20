
use crate::error::Result;
use chrono::Utc;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::Path;

pub fn generate_message_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

pub fn start_new_conversation() -> String {
    uuid::Uuid::new_v4().to_string()
}

pub struct CommandHistory {
    conn: Connection,
    pool: Option<sqlx::AnyPool>,
    pub db_path: String,
}

impl CommandHistory {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref().to_string_lossy().to_string();
        let conn = Connection::open(path.as_ref())?;
        let history = Self { conn, pool: None, db_path: p };
        history.init_tables()?;
        Ok(history)
    }

    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let history = Self { conn, pool: None, db_path: ":memory:".to_string() };
        history.init_tables()?;
        Ok(history)
    }

    pub async fn open_async(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        let url = if path == ":memory:" {
            "sqlite::memory:".to_string()
        } else {
            format!("sqlite://{}?mode=rwc", path)
        };
        let pool = sqlx::AnyPool::connect(&url).await
            .map_err(|e| crate::error::NpcError::Other(format!("sqlx connect: {}", e)))?;
        let history = Self { conn, pool: Some(pool), db_path: path.to_string() };
        history.init_tables()?;
        Ok(history)
    }

    pub fn pool(&self) -> Option<&sqlx::AnyPool> {
        self.pool.as_ref()
    }

    fn init_tables(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            -- Legacy command_history table
            CREATE TABLE IF NOT EXISTS command_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp VARCHAR(50),
                command TEXT,
                subcommands TEXT,
                output TEXT,
                location TEXT
            );

            -- Main message store (matches npcpy exactly)
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id VARCHAR(50) UNIQUE NOT NULL,
                timestamp VARCHAR(50),
                role VARCHAR(20),
                content TEXT,
                conversation_id VARCHAR(100),
                directory_path TEXT,
                model VARCHAR(100),
                provider VARCHAR(100),
                npc VARCHAR(100),
                team VARCHAR(100),
                reasoning_content TEXT,
                tool_calls TEXT,
                tool_results TEXT,
                parent_message_id VARCHAR(50),
                device_id VARCHAR(255),
                device_name VARCHAR(255),
                params TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost VARCHAR(50)
            );

            -- Jinx execution tracking
            CREATE TABLE IF NOT EXISTS jinx_executions (
                message_id VARCHAR(50) PRIMARY KEY,
                jinx_name VARCHAR(100),
                input TEXT,
                timestamp VARCHAR(50),
                npc VARCHAR(100),
                team VARCHAR(100),
                conversation_id VARCHAR(100),
                output TEXT,
                status VARCHAR(50),
                error_message TEXT,
                duration_ms INTEGER
            );

            -- NPC execution tracking
            CREATE TABLE IF NOT EXISTS npc_executions (
                message_id VARCHAR(50) PRIMARY KEY,
                input TEXT,
                timestamp VARCHAR(50),
                npc VARCHAR(100),
                team VARCHAR(100),
                conversation_id VARCHAR(100),
                model VARCHAR(100),
                provider VARCHAR(100)
            );

            -- Message attachments
            CREATE TABLE IF NOT EXISTS message_attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id VARCHAR(50) NOT NULL,
                attachment_name VARCHAR(255),
                attachment_type VARCHAR(100),
                attachment_data BLOB,
                attachment_size INTEGER,
                upload_timestamp VARCHAR(50),
                file_path TEXT
            );

            -- Compiled NPC cache
            CREATE TABLE IF NOT EXISTS compiled_npcs (
                name TEXT PRIMARY KEY,
                source_path TEXT,
                compiled_content TEXT,
                compiled_at TEXT
            );

            -- Memory lifecycle
            CREATE TABLE IF NOT EXISTS memory_lifecycle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id VARCHAR(50) NOT NULL,
                conversation_id VARCHAR(100) NOT NULL,
                npc VARCHAR(100) NOT NULL,
                team VARCHAR(100) NOT NULL,
                directory_path TEXT NOT NULL,
                timestamp VARCHAR(50) NOT NULL,
                initial_memory TEXT NOT NULL,
                final_memory TEXT,
                status VARCHAR(50) NOT NULL,
                model VARCHAR(100),
                provider VARCHAR(100),
                created_at TEXT
            );

            -- Labels
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type VARCHAR(50) NOT NULL,
                entity_id VARCHAR(100) NOT NULL,
                label VARCHAR(100) NOT NULL,
                metadata TEXT,
                created_at TEXT
            );

            -- NPC memories
            CREATE TABLE IF NOT EXISTS npc_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                npc_name TEXT NOT NULL,
                team_name TEXT,
                content TEXT NOT NULL,
                embedding BLOB,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT
            );

            -- Knowledge graphs
            CREATE TABLE IF NOT EXISTS knowledge_graphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                npc_name TEXT,
                team_name TEXT,
                kg_data TEXT NOT NULL,
                generation INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_conv_hist_conv_id ON conversation_history(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_conv_hist_role ON conversation_history(role);
            CREATE INDEX IF NOT EXISTS idx_conv_hist_npc ON conversation_history(npc);
            CREATE INDEX IF NOT EXISTS idx_conv_hist_msg_id ON conversation_history(message_id);
            CREATE INDEX IF NOT EXISTS idx_jinx_exec_name ON jinx_executions(jinx_name);
            CREATE INDEX IF NOT EXISTS idx_npc_memories_npc ON npc_memories(npc_name);
            CREATE INDEX IF NOT EXISTS idx_npc_memories_status ON npc_memories(status);
            CREATE INDEX IF NOT EXISTS idx_kg_npc ON knowledge_graphs(npc_name);
            ",
        )?;
        Ok(())
    }

    pub fn save_conversation_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
        directory_path: &str,
        model: Option<&str>,
        provider: Option<&str>,
        npc: Option<&str>,
        team: Option<&str>,
        tool_calls_json: Option<&str>,
        tool_results_json: Option<&str>,
        parent_message_id: Option<&str>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        cost: Option<f64>,
    ) -> Result<String> {
        let message_id = generate_message_id();
        let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let cost_str = cost.map(|c| format!("{:.6}", c));

        self.conn.execute(
            "INSERT INTO conversation_history
             (message_id, timestamp, role, content, conversation_id, directory_path,
              model, provider, npc, team, tool_calls, tool_results,
              parent_message_id, input_tokens, output_tokens, cost)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            params![
                message_id,
                timestamp,
                role,
                content,
                conversation_id,
                directory_path,
                model,
                provider,
                npc,
                team,
                tool_calls_json,
                tool_results_json,
                parent_message_id,
                input_tokens.map(|t| t as i64),
                output_tokens.map(|t| t as i64),
                cost_str,
            ],
        )?;
        Ok(message_id)
    }

    pub fn save_jinx_execution(
        &self,
        conversation_id: &str,
        jinx_name: &str,
        input: &str,
        output: &str,
        status: &str,
        npc: Option<&str>,
        team: Option<&str>,
        error_message: Option<&str>,
        duration_ms: Option<i64>,
    ) -> Result<()> {
        let message_id = generate_message_id();
        let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();

        self.conn.execute(
            "INSERT INTO jinx_executions
             (message_id, jinx_name, input, timestamp, npc, team, conversation_id,
              output, status, error_message, duration_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                message_id,
                jinx_name,
                input,
                timestamp,
                npc,
                team,
                conversation_id,
                output,
                status,
                error_message,
                duration_ms,
            ],
        )?;
        Ok(())
    }

    pub fn load_conversation_messages(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<ConversationMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT message_id, role, content, model, provider, npc, team,
                    tool_calls, input_tokens, output_tokens, cost
             FROM conversation_history
             WHERE conversation_id = ?1 ORDER BY id ASC",
        )?;

        let messages = stmt
            .query_map(params![conversation_id], |row| {
                Ok(ConversationMessage {
                    message_id: row.get(0)?,
                    role: row.get(1)?,
                    content: row.get(2)?,
                    model: row.get(3)?,
                    provider: row.get(4)?,
                    npc: row.get(5)?,
                    team: row.get(6)?,
                    tool_calls: row.get(7)?,
                    input_tokens: row.get(8)?,
                    output_tokens: row.get(9)?,
                    cost: row.get(10)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(messages)
    }

    pub fn get_last_message_id(&self, conversation_id: &str) -> Result<Option<String>> {
        let result = self.conn.query_row(
            "SELECT message_id FROM conversation_history WHERE conversation_id = ?1 ORDER BY id DESC LIMIT 1",
            params![conversation_id],
            |row| row.get(0),
        );
        match result {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn total_usage(&self) -> Result<(u64, u64)> {
        let mut stmt = self.conn.prepare(
            "SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM conversation_history",
        )?;
        let (input, output) = stmt.query_row([], |row| {
            Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
        })?;
        Ok((input, output))
    }

    pub fn save_memory(&self, npc_name: &str, content: &str) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO npc_memories (npc_name, content, status, created_at) VALUES (?1, ?2, 'pending', ?3)",
            params![npc_name, content, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn get_pending_memories(&self) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, npc_name, content FROM npc_memories WHERE status = 'pending' ORDER BY id ASC",
        )?;
        let memories = stmt
            .query_map(params![], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        Ok(memories)
    }

    pub fn save_kg_to_db(&self, npc_name: &str, kg_json: &str, generation: i32) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let existing: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM knowledge_graphs WHERE npc_name = ?1",
                params![npc_name],
                |row| row.get(0),
            )
            .ok();

        if let Some(id) = existing {
            self.conn.execute(
                "UPDATE knowledge_graphs SET kg_data = ?1, generation = ?2, updated_at = ?3 WHERE id = ?4",
                params![kg_json, generation, now, id],
            )?;
        } else {
            self.conn.execute(
                "INSERT INTO knowledge_graphs (npc_name, kg_data, generation, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![npc_name, kg_json, generation, now],
            )?;
        }
        Ok(())
    }

    pub fn load_kg_from_db(&self, npc_name: &str) -> Result<Option<(String, i32)>> {
        let result = self.conn.query_row(
            "SELECT kg_data, generation FROM knowledge_graphs WHERE npc_name = ?1 ORDER BY id DESC LIMIT 1",
            params![npc_name],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i32>(1)?)),
        );
        match result {
            Ok(data) => Ok(Some(data)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn log_entry(&self, entity_id: &str, entry_type: &str, content: &str, metadata: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO labels (entity_type, entity_id, label, metadata, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![entry_type, entity_id, content, metadata, now],
        )?;
        Ok(())
    }

    pub fn retrieve_last_conversation(&self) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT conversation_id FROM conversation_history ORDER BY timestamp DESC LIMIT 1"
        )?;
        let result = stmt.query_row(params![], |row| row.get::<_, String>(0));
        match result {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn save_npc_version(&self, npc_name: &str, content: &str) -> Result<i64> {
        let version: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM npc_versions WHERE npc_name = ?1",
            params![npc_name],
            |row| row.get(0),
        ).unwrap_or(1);
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO npc_versions (npc_name, version, content, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![npc_name, version, content, now],
        )?;
        Ok(version)
    }

    pub fn get_npc_versions(&self, npc_name: &str) -> Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT version, created_at FROM npc_versions WHERE npc_name = ?1 ORDER BY version DESC"
        )?;
        let results = stmt.query_map(params![npc_name], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn get_npc_version_content(&self, npc_name: &str, version: Option<i64>) -> Result<Option<String>> {
        let query = if let Some(v) = version {
            self.conn.query_row(
                "SELECT content FROM npc_versions WHERE npc_name = ?1 AND version = ?2",
                params![npc_name, v],
                |row| row.get::<_, String>(0),
            )
        } else {
            self.conn.query_row(
                "SELECT content FROM npc_versions WHERE npc_name = ?1 ORDER BY version DESC LIMIT 1",
                params![npc_name],
                |row| row.get::<_, String>(0),
            )
        };
        match query {
            Ok(content) => Ok(Some(content)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn rollback_npc_to_version(&self, npc_name: &str, version: i64) -> Result<Option<String>> {
        self.get_npc_version_content(npc_name, Some(version))
    }

    pub fn save_attachment_to_message(&self, message_id: &str, attachment_type: &str, data: &[u8], filename: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT OR IGNORE INTO message_attachments (message_id, attachment_type, attachment_data, attachment_name, upload_timestamp) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![message_id, attachment_type, data, filename, now],
        )?;
        Ok(())
    }

    pub fn add_command(&self, command: &str, subcommands: &str, output: &str, location: &str) -> Result<()> {
        let now = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        self.conn.execute(
            "INSERT INTO command_history (timestamp, command, subcommands, output, location) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![now, command, subcommands, output, location],
        )?;
        Ok(())
    }

    pub fn add_conversation(&self, conversation_id: &str, role: &str, content: &str, npc: Option<&str>, team: Option<&str>, model: Option<&str>, provider: Option<&str>) -> Result<String> {
        let dir = std::env::current_dir().unwrap_or_default().to_string_lossy().to_string();
        self.save_conversation_message(conversation_id, role, content, &dir, model, provider, npc, team, None, None, None, None, None, None)
    }

    pub fn add_memory_to_database(&self, message_id: &str, conversation_id: &str, npc: &str, team: &str, directory_path: &str, initial_memory: &str, model: Option<&str>, provider: Option<&str>) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO memory_lifecycle (message_id, conversation_id, npc, team, directory_path, timestamp, initial_memory, status, model, provider, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pending', ?8, ?9, ?10)",
            params![message_id, conversation_id, npc, team, directory_path, now, initial_memory, model, provider, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn get_memories_for_scope(&self, npc: &str, team: &str, directory_path: &str, limit: usize) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, initial_memory, final_memory, status, created_at FROM memory_lifecycle WHERE npc = ?1 AND team = ?2 AND directory_path = ?3 AND status IN ('approved', 'human-approved', 'human-edited') ORDER BY created_at DESC LIMIT ?4"
        )?;
        let results = stmt.query_map(params![npc, team, directory_path, limit as i64], |row| {
            let mut m = HashMap::new();
            m.insert("id".into(), serde_json::json!(row.get::<_, i64>(0)?));
            m.insert("initial_memory".into(), serde_json::json!(row.get::<_, String>(1)?));
            m.insert("final_memory".into(), serde_json::json!(row.get::<_, Option<String>>(2)?));
            m.insert("status".into(), serde_json::json!(row.get::<_, String>(3)?));
            m.insert("created_at".into(), serde_json::json!(row.get::<_, String>(4)?));
            Ok(m)
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn search_memory(&self, query: &str, npc: Option<&str>, team: Option<&str>, limit: usize) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let pattern = format!("%{}%", query);
        let sql = match (npc, team) {
            (Some(n), Some(t)) => format!("SELECT id, initial_memory, final_memory, status, npc, team FROM memory_lifecycle WHERE (initial_memory LIKE ?1 OR final_memory LIKE ?1) AND npc = '{}' AND team = '{}' ORDER BY created_at DESC LIMIT ?2", n, t),
            (Some(n), None) => format!("SELECT id, initial_memory, final_memory, status, npc, team FROM memory_lifecycle WHERE (initial_memory LIKE ?1 OR final_memory LIKE ?1) AND npc = '{}' ORDER BY created_at DESC LIMIT ?2", n),
            _ => "SELECT id, initial_memory, final_memory, status, npc, team FROM memory_lifecycle WHERE (initial_memory LIKE ?1 OR final_memory LIKE ?1) ORDER BY created_at DESC LIMIT ?2".to_string(),
        };
        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map(params![pattern, limit as i64], |row| {
            let mut m = HashMap::new();
            m.insert("id".into(), serde_json::json!(row.get::<_, i64>(0)?));
            m.insert("initial_memory".into(), serde_json::json!(row.get::<_, String>(1)?));
            m.insert("final_memory".into(), serde_json::json!(row.get::<_, Option<String>>(2)?));
            m.insert("status".into(), serde_json::json!(row.get::<_, String>(3)?));
            m.insert("npc".into(), serde_json::json!(row.get::<_, String>(4)?));
            m.insert("team".into(), serde_json::json!(row.get::<_, String>(5)?));
            Ok(m)
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn get_memory_examples_for_context(&self, npc: &str, team: &str, directory_path: &str, limit: usize) -> Result<Vec<String>> {
        let memories = self.get_memories_for_scope(npc, team, directory_path, limit)?;
        Ok(memories.iter().map(|m| {
            m.get("final_memory").and_then(|v| v.as_str()).or_else(|| m.get("initial_memory").and_then(|v| v.as_str())).unwrap_or("").to_string()
        }).filter(|s| !s.is_empty()).collect())
    }

    pub fn update_memory_status(&self, memory_id: i64, new_status: &str, final_memory: Option<&str>) -> Result<()> {
        if let Some(fm) = final_memory {
            self.conn.execute("UPDATE memory_lifecycle SET status = ?1, final_memory = ?2 WHERE id = ?3", params![new_status, fm, memory_id])?;
        } else {
            self.conn.execute("UPDATE memory_lifecycle SET status = ?1 WHERE id = ?2", params![new_status, memory_id])?;
        }
        Ok(())
    }

    pub fn get_approved_memories_by_scope(&self) -> Result<HashMap<String, Vec<String>>> {
        let mut stmt = self.conn.prepare(
            "SELECT npc, COALESCE(final_memory, initial_memory) FROM memory_lifecycle WHERE status IN ('approved', 'human-approved', 'human-edited') ORDER BY npc"
        )?;
        let mut result: HashMap<String, Vec<String>> = HashMap::new();
        let rows = stmt.query_map(params![], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
        for row in rows.flatten() {
            result.entry(row.0).or_default().push(row.1);
        }
        Ok(result)
    }

    pub fn get_jinx_executions(&self, jinx_name: Option<&str>, limit: usize) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let sql = if let Some(name) = jinx_name {
            format!("SELECT message_id, jinx_name, input, output, status, timestamp FROM jinx_executions WHERE jinx_name = '{}' ORDER BY timestamp DESC LIMIT {}", name, limit)
        } else {
            format!("SELECT message_id, jinx_name, input, output, status, timestamp FROM jinx_executions ORDER BY timestamp DESC LIMIT {}", limit)
        };
        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map(params![], |row| {
            let mut m = HashMap::new();
            m.insert("message_id".into(), serde_json::json!(row.get::<_, String>(0)?));
            m.insert("jinx_name".into(), serde_json::json!(row.get::<_, String>(1)?));
            m.insert("input".into(), serde_json::json!(row.get::<_, String>(2)?));
            m.insert("output".into(), serde_json::json!(row.get::<_, String>(3)?));
            m.insert("status".into(), serde_json::json!(row.get::<_, String>(4)?));
            m.insert("timestamp".into(), serde_json::json!(row.get::<_, String>(5)?));
            Ok(m)
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn get_npc_executions(&self, npc_name: &str, limit: usize) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut stmt = self.conn.prepare(
            "SELECT message_id, input, npc, team, model, provider, timestamp FROM npc_executions WHERE npc = ?1 ORDER BY timestamp DESC LIMIT ?2"
        )?;
        let results = stmt.query_map(params![npc_name, limit as i64], |row| {
            let mut m = HashMap::new();
            m.insert("message_id".into(), serde_json::json!(row.get::<_, String>(0)?));
            m.insert("input".into(), serde_json::json!(row.get::<_, String>(1)?));
            m.insert("npc".into(), serde_json::json!(row.get::<_, String>(2)?));
            m.insert("team".into(), serde_json::json!(row.get::<_, String>(3)?));
            m.insert("model".into(), serde_json::json!(row.get::<_, String>(4)?));
            m.insert("provider".into(), serde_json::json!(row.get::<_, String>(5)?));
            m.insert("timestamp".into(), serde_json::json!(row.get::<_, String>(6)?));
            Ok(m)
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn label_execution(&self, message_id: &str, label: &str) -> Result<()> {
        self.add_label("execution", message_id, label, None)
    }

    pub fn add_label(&self, entity_type: &str, entity_id: &str, label: &str, metadata: Option<&str>) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO labels (entity_type, entity_id, label, metadata, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![entity_type, entity_id, label, metadata, now],
        )?;
        Ok(())
    }

    pub fn get_labels(&self, entity_type: Option<&str>, label: Option<&str>) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let sql = match (entity_type, label) {
            (Some(et), Some(l)) => format!("SELECT id, entity_type, entity_id, label, metadata, created_at FROM labels WHERE entity_type = '{}' AND label = '{}'", et, l),
            (Some(et), None) => format!("SELECT id, entity_type, entity_id, label, metadata, created_at FROM labels WHERE entity_type = '{}'", et),
            (None, Some(l)) => format!("SELECT id, entity_type, entity_id, label, metadata, created_at FROM labels WHERE label = '{}'", l),
            _ => "SELECT id, entity_type, entity_id, label, metadata, created_at FROM labels".to_string(),
        };
        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt.query_map(params![], |row| {
            let mut m = HashMap::new();
            m.insert("id".into(), serde_json::json!(row.get::<_, i64>(0)?));
            m.insert("entity_type".into(), serde_json::json!(row.get::<_, String>(1)?));
            m.insert("entity_id".into(), serde_json::json!(row.get::<_, String>(2)?));
            m.insert("label".into(), serde_json::json!(row.get::<_, String>(3)?));
            m.insert("metadata".into(), serde_json::json!(row.get::<_, Option<String>>(4)?));
            m.insert("created_at".into(), serde_json::json!(row.get::<_, String>(5)?));
            Ok(m)
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn get_training_data_by_label(&self, label: &str) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut stmt = self.conn.prepare(
            "SELECT ch.role, ch.content, ch.model, ch.npc FROM conversation_history ch INNER JOIN labels l ON l.entity_id = ch.message_id WHERE l.label = ?1"
        )?;
        let results = stmt.query_map(params![label], |row| {
            let mut m = HashMap::new();
            m.insert("role".into(), serde_json::json!(row.get::<_, String>(0)?));
            m.insert("content".into(), serde_json::json!(row.get::<_, String>(1)?));
            m.insert("model".into(), serde_json::json!(row.get::<_, Option<String>>(2)?));
            m.insert("npc".into(), serde_json::json!(row.get::<_, Option<String>>(3)?));
            Ok(m)
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn get_message_by_id(&self, message_id: &str) -> Result<Option<ConversationMessage>> {
        let result = self.conn.query_row(
            "SELECT message_id, role, content, model, provider, npc, team, tool_calls, input_tokens, output_tokens, cost FROM conversation_history WHERE message_id = ?1",
            params![message_id],
            |row| Ok(ConversationMessage { message_id: row.get(0)?, role: row.get(1)?, content: row.get(2)?, model: row.get(3)?, provider: row.get(4)?, npc: row.get(5)?, team: row.get(6)?, tool_calls: row.get(7)?, input_tokens: row.get(8)?, output_tokens: row.get(9)?, cost: row.get(10)? }),
        );
        match result { Ok(m) => Ok(Some(m)), Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None), Err(e) => Err(e.into()) }
    }

    pub fn get_messages_by_npc(&self, npc: &str, n_last: usize) -> Result<Vec<ConversationMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT message_id, role, content, model, provider, npc, team, tool_calls, input_tokens, output_tokens, cost FROM conversation_history WHERE npc = ?1 ORDER BY id DESC LIMIT ?2"
        )?;
        let results = stmt.query_map(params![npc, n_last as i64], |row| {
            Ok(ConversationMessage { message_id: row.get(0)?, role: row.get(1)?, content: row.get(2)?, model: row.get(3)?, provider: row.get(4)?, npc: row.get(5)?, team: row.get(6)?, tool_calls: row.get(7)?, input_tokens: row.get(8)?, output_tokens: row.get(9)?, cost: row.get(10)? })
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn get_messages_by_team(&self, team: &str, n_last: usize) -> Result<Vec<ConversationMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT message_id, role, content, model, provider, npc, team, tool_calls, input_tokens, output_tokens, cost FROM conversation_history WHERE team = ?1 ORDER BY id DESC LIMIT ?2"
        )?;
        let results = stmt.query_map(params![team, n_last as i64], |row| {
            Ok(ConversationMessage { message_id: row.get(0)?, role: row.get(1)?, content: row.get(2)?, model: row.get(3)?, provider: row.get(4)?, npc: row.get(5)?, team: row.get(6)?, tool_calls: row.get(7)?, input_tokens: row.get(8)?, output_tokens: row.get(9)?, cost: row.get(10)? })
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn get_most_recent_conversation_id(&self) -> Result<Option<String>> {
        self.retrieve_last_conversation()
    }

    pub fn get_last_conversation(&self, conversation_id: &str) -> Result<Vec<ConversationMessage>> {
        self.load_conversation_messages(conversation_id)
    }

    pub fn get_conversations_by_id(&self, conversation_id: &str) -> Result<Vec<ConversationMessage>> {
        self.load_conversation_messages(conversation_id)
    }

    pub fn get_last_command(&self) -> Result<Option<HashMap<String, String>>> {
        let result = self.conn.query_row(
            "SELECT command, output, location, timestamp FROM command_history ORDER BY id DESC LIMIT 1",
            params![],
            |row| {
                let mut m = HashMap::new();
                m.insert("command".into(), row.get::<_, String>(0)?);
                m.insert("output".into(), row.get::<_, String>(1)?);
                m.insert("location".into(), row.get::<_, String>(2)?);
                m.insert("timestamp".into(), row.get::<_, String>(3)?);
                Ok(m)
            }
        );
        match result { Ok(m) => Ok(Some(m)), Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None), Err(e) => Err(e.into()) }
    }

    pub fn search_commands(&self, search_term: &str) -> Result<Vec<HashMap<String, String>>> {
        let pattern = format!("%{}%", search_term);
        let mut stmt = self.conn.prepare("SELECT command, output, location, timestamp FROM command_history WHERE command LIKE ?1 ORDER BY id DESC LIMIT 100")?;
        let results = stmt.query_map(params![pattern], |row| {
            let mut m = HashMap::new();
            m.insert("command".into(), row.get::<_, String>(0)?);
            m.insert("output".into(), row.get::<_, String>(1)?);
            m.insert("location".into(), row.get::<_, String>(2)?);
            m.insert("timestamp".into(), row.get::<_, String>(3)?);
            Ok(m)
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn search_conversations(&self, search_term: &str) -> Result<Vec<ConversationMessage>> {
        let pattern = format!("%{}%", search_term);
        let mut stmt = self.conn.prepare(
            "SELECT message_id, role, content, model, provider, npc, team, tool_calls, input_tokens, output_tokens, cost FROM conversation_history WHERE content LIKE ?1 ORDER BY id DESC LIMIT 100"
        )?;
        let results = stmt.query_map(params![pattern], |row| {
            Ok(ConversationMessage { message_id: row.get(0)?, role: row.get(1)?, content: row.get(2)?, model: row.get(3)?, provider: row.get(4)?, npc: row.get(5)?, team: row.get(6)?, tool_calls: row.get(7)?, input_tokens: row.get(8)?, output_tokens: row.get(9)?, cost: row.get(10)? })
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn get_all_commands(&self, limit: usize) -> Result<Vec<HashMap<String, String>>> {
        let mut stmt = self.conn.prepare("SELECT command, output, location, timestamp FROM command_history ORDER BY id DESC LIMIT ?1")?;
        let results = stmt.query_map(params![limit as i64], |row| {
            let mut m = HashMap::new();
            m.insert("command".into(), row.get::<_, String>(0)?);
            m.insert("output".into(), row.get::<_, String>(1)?);
            m.insert("location".into(), row.get::<_, String>(2)?);
            m.insert("timestamp".into(), row.get::<_, String>(3)?);
            Ok(m)
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn delete_message(&self, conversation_id: &str, message_id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM conversation_history WHERE conversation_id = ?1 AND message_id = ?2", params![conversation_id, message_id])?;
        Ok(())
    }

    pub fn get_message_attachments(&self, message_id: &str) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut stmt = self.conn.prepare("SELECT id, attachment_name, attachment_type, attachment_size, file_path FROM message_attachments WHERE message_id = ?1")?;
        let results = stmt.query_map(params![message_id], |row| {
            let mut m = HashMap::new();
            m.insert("id".into(), serde_json::json!(row.get::<_, i64>(0)?));
            m.insert("name".into(), serde_json::json!(row.get::<_, Option<String>>(1)?));
            m.insert("type".into(), serde_json::json!(row.get::<_, Option<String>>(2)?));
            m.insert("size".into(), serde_json::json!(row.get::<_, Option<i64>>(3)?));
            m.insert("file_path".into(), serde_json::json!(row.get::<_, Option<String>>(4)?));
            Ok(m)
        })?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn get_available_tables(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")?;
        let results = stmt.query_map(params![], |row| row.get::<_, String>(0))?.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    pub fn close(self) {
        drop(self.conn);
    }
}

pub fn normalize_path_for_db(path: &str) -> String {
    let expanded = shellexpand::tilde(path).to_string();
    std::path::Path::new(&expanded).canonicalize().map(|p| p.to_string_lossy().to_string()).unwrap_or(expanded)
}

pub fn flush_messages(n: usize, messages: &[HashMap<String, String>]) -> HashMap<String, serde_json::Value> {
    let kept: Vec<&HashMap<String, String>> = if messages.len() > n { &messages[messages.len()-n..] } else { messages }.iter().collect();
    let mut result = HashMap::new();
    result.insert("messages".into(), serde_json::json!(kept));
    result.insert("flushed".into(), serde_json::json!(messages.len().saturating_sub(n)));
    result
}

pub fn format_memory_context(memory_examples: &[String]) -> String {
    if memory_examples.is_empty() { return String::new(); }
    let mut ctx = String::from("Here are some things I remember about you:\n");
    for mem in memory_examples {
        ctx.push_str(&format!("- {}\n", mem));
    }
    ctx
}

#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub message_id: String,
    pub role: String,
    pub content: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub npc: Option<String>,
    pub team: Option<String>,
    pub tool_calls: Option<String>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cost: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_lifecycle() {
        let history = CommandHistory::in_memory().unwrap();
        let conv_id = start_new_conversation();

        let msg_id = history
            .save_conversation_message(
                &conv_id, "user", "hello", "/tmp",
                Some("qwen3.5:2b"), Some("ollama"), Some("sibiji"), Some("npc_team"),
                None, None, None, Some(10), None, None,
            )
            .unwrap();
        assert!(!msg_id.is_empty());

        history
            .save_conversation_message(
                &conv_id, "assistant", "hi there", "/tmp",
                Some("qwen3.5:2b"), Some("ollama"), Some("sibiji"), Some("npc_team"),
                None, None, None, None, Some(20), Some(0.001),
            )
            .unwrap();

        let messages = history.load_conversation_messages(&conv_id).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].content.as_deref(), Some("hi there"));
    }
}
