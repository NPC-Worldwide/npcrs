//! Conversation history — mirrors npcpy's conversation_history schema exactly.
//!
//! Uses the same table names, column names, and UUID-based IDs as the Python version
//! so both can share the same SQLite database.

use crate::error::Result;
use chrono::Utc;
use rusqlite::{params, Connection};
use std::path::Path;

/// Generate a UUID message ID (matches npcpy's generate_message_id).
pub fn generate_message_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Start a new conversation (matches npcpy's start_new_conversation).
pub fn start_new_conversation() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// SQLite-backed conversation history matching npcpy's CommandHistory.
pub struct CommandHistory {
    conn: Connection,
}

impl CommandHistory {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        let history = Self { conn };
        history.init_tables()?;
        Ok(history)
    }

    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let history = Self { conn };
        history.init_tables()?;
        Ok(history)
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

    /// Save a conversation message (mirrors npcpy's save_conversation_message).
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

    /// Save a jinx execution record.
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

    /// Load messages for a conversation (ordered by id).
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

    /// Get the last message_id in a conversation (for linking).
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

    /// Get total token usage.
    pub fn total_usage(&self) -> Result<(u64, u64)> {
        let mut stmt = self.conn.prepare(
            "SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM conversation_history",
        )?;
        let (input, output) = stmt.query_row([], |row| {
            Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
        })?;
        Ok((input, output))
    }

    // ── Memory management ──

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
}

/// A message from conversation_history.
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
