use crate::error::Result;
use crate::llm::Message;
use chrono::Utc;
use rusqlite::{params, Connection};
use std::path::Path;

/// SQLite-backed conversation history.
pub struct CommandHistory {
    conn: Connection,
}

impl CommandHistory {
    /// Open or create a history database.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        let history = Self { conn };
        history.init_tables()?;
        Ok(history)
    }

    /// Open an in-memory database (for testing).
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let history = Self { conn };
        history.init_tables()?;
        Ok(history)
    }

    fn init_tables(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                npc_name TEXT,
                started_at TEXT NOT NULL,
                parent_id INTEGER,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                created_at TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS jinx_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                jinx_name TEXT NOT NULL,
                inputs TEXT,
                output TEXT,
                success INTEGER NOT NULL DEFAULT 1,
                executed_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS compiled_npcs (
                name TEXT PRIMARY KEY,
                source_path TEXT,
                compiled_content TEXT,
                compiled_at TEXT
            );

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

            CREATE TABLE IF NOT EXISTS knowledge_graphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                npc_name TEXT,
                team_name TEXT,
                kg_data TEXT NOT NULL,
                generation INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS npc_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL,
                entry_type TEXT NOT NULL,
                content TEXT,
                metadata TEXT,
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
            CREATE INDEX IF NOT EXISTS idx_jinx_exec_name ON jinx_executions(jinx_name);
            CREATE INDEX IF NOT EXISTS idx_npc_memories_npc ON npc_memories(npc_name);
            CREATE INDEX IF NOT EXISTS idx_npc_memories_status ON npc_memories(status);
            CREATE INDEX IF NOT EXISTS idx_kg_npc ON knowledge_graphs(npc_name);
            CREATE INDEX IF NOT EXISTS idx_npc_log_entity ON npc_log(entity_id);
            ",
        )?;

        // Migration-safe: add npc_name and team_name columns to messages if they don't exist.
        // SQLite doesn't have IF NOT EXISTS for ALTER TABLE, so we try and ignore errors.
        let _ = self
            .conn
            .execute("ALTER TABLE messages ADD COLUMN npc_name TEXT", []);
        let _ = self
            .conn
            .execute("ALTER TABLE messages ADD COLUMN team_name TEXT", []);

        Ok(())
    }

    /// Start a new conversation.
    pub fn new_conversation(
        &self,
        npc_name: &str,
        parent_id: Option<i64>,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO conversations (npc_name, started_at, parent_id) VALUES (?1, ?2, ?3)",
            params![npc_name, now, parent_id],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Save a message to a conversation.
    pub fn save_message(
        &self,
        conversation_id: i64,
        message: &Message,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let tool_calls_json = message
            .tool_calls
            .as_ref()
            .map(|tc| serde_json::to_string(tc).unwrap_or_default());

        self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, tool_calls, tool_call_id, created_at, input_tokens, output_tokens)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                conversation_id,
                message.role,
                message.content,
                tool_calls_json,
                message.tool_call_id,
                now,
                input_tokens,
                output_tokens,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Load all messages for a conversation.
    pub fn load_messages(&self, conversation_id: i64) -> Result<Vec<Message>> {
        let mut stmt = self.conn.prepare(
            "SELECT role, content, tool_calls, tool_call_id FROM messages
             WHERE conversation_id = ?1 ORDER BY id ASC",
        )?;

        let messages = stmt
            .query_map(params![conversation_id], |row| {
                let tool_calls_str: Option<String> = row.get(2)?;
                let tool_calls = tool_calls_str
                    .and_then(|s| serde_json::from_str(&s).ok());
                Ok(Message {
                    role: row.get(0)?,
                    content: row.get(1)?,
                    tool_calls,
                    tool_call_id: row.get(3)?,
                    name: None,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(messages)
    }

    /// Record a jinx execution.
    pub fn record_jinx_execution(
        &self,
        conversation_id: Option<i64>,
        jinx_name: &str,
        inputs: &str,
        output: &str,
        success: bool,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO jinx_executions (conversation_id, jinx_name, inputs, output, success, executed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![conversation_id, jinx_name, inputs, output, success as i32, now],
        )?;
        Ok(())
    }

    /// List recent conversations.
    pub fn recent_conversations(&self, limit: u32) -> Result<Vec<ConversationInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, npc_name, started_at, summary FROM conversations
             ORDER BY id DESC LIMIT ?1",
        )?;

        let convos = stmt
            .query_map(params![limit], |row| {
                Ok(ConversationInfo {
                    id: row.get(0)?,
                    npc_name: row.get(1)?,
                    started_at: row.get(2)?,
                    summary: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(convos)
    }

    /// Get total token usage across all conversations.
    pub fn total_usage(&self) -> Result<(u64, u64)> {
        let mut stmt = self.conn.prepare(
            "SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM messages",
        )?;

        let (input, output) = stmt.query_row([], |row| {
            Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
        })?;

        Ok((input, output))
    }

    // ── Memory management ──

    /// Save a new pending memory for an NPC.
    pub fn save_memory(&self, npc_name: &str, content: &str) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO npc_memories (npc_name, content, status, created_at) VALUES (?1, ?2, 'pending', ?3)",
            params![npc_name, content, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get all pending memories for review.
    /// Returns tuples of (id, npc_name, content).
    pub fn get_pending_memories(&self) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, npc_name, content FROM npc_memories WHERE status = 'pending' ORDER BY id ASC",
        )?;

        let memories = stmt
            .query_map(params![], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(memories)
    }

    // ── Knowledge graph persistence ──

    /// Save a knowledge graph to the database.
    pub fn save_kg_to_db(
        &self,
        npc_name: &str,
        kg_json: &str,
        generation: i32,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();

        // Upsert: if a KG exists for this NPC, update it; otherwise insert.
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

    /// Load a knowledge graph from the database.
    /// Returns (kg_json, generation) if found.
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

    // ── General logging ──

    /// Write an entry to the npc_log table.
    pub fn log_entry(
        &self,
        entity_id: &str,
        entry_type: &str,
        content: &str,
        metadata: &str,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO npc_log (entity_id, entry_type, content, metadata, timestamp) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![entity_id, entry_type, content, metadata, now],
        )?;
        Ok(())
    }
}

/// Summary info about a conversation.
#[derive(Debug, Clone)]
pub struct ConversationInfo {
    pub id: i64,
    pub npc_name: Option<String>,
    pub started_at: String,
    pub summary: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_lifecycle() {
        let history = CommandHistory::in_memory().unwrap();

        let conv_id = history.new_conversation("test_npc", None).unwrap();
        assert!(conv_id > 0);

        history
            .save_message(conv_id, &Message::user("hello"), 10, 0)
            .unwrap();
        history
            .save_message(
                conv_id,
                &Message::assistant("hi there"),
                0,
                20,
            )
            .unwrap();

        let messages = history.load_messages(conv_id).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].content.as_deref(), Some("hi there"));

        let (input, output) = history.total_usage().unwrap();
        assert_eq!(input, 10);
        assert_eq!(output, 20);
    }
}
