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

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
            CREATE INDEX IF NOT EXISTS idx_jinx_exec_name ON jinx_executions(jinx_name);
            ",
        )?;
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
