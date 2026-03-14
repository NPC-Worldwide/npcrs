//! Memory lifecycle management: save, review, approve/reject.

use crate::error::Result;
use chrono::Utc;
use rusqlite::{params, Connection};

/// Status of a memory in the review pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryStatus {
    Pending,
    Approved,
    Rejected,
}

impl MemoryStatus {
    /// Convert to database string.
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryStatus::Pending => "pending",
            MemoryStatus::Approved => "approved",
            MemoryStatus::Rejected => "rejected",
        }
    }

    /// Parse from database string.
    pub fn from_str(s: &str) -> Self {
        match s {
            "approved" => MemoryStatus::Approved,
            "rejected" => MemoryStatus::Rejected,
            _ => MemoryStatus::Pending,
        }
    }
}

/// A memory record with full metadata.
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: i64,
    pub npc_name: String,
    pub content: String,
    pub status: MemoryStatus,
    pub embedding: Option<Vec<f64>>,
    pub created_at: String,
}

/// Save a new pending memory.
pub fn save_memory(conn: &Connection, npc_name: &str, content: &str) -> Result<i64> {
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO npc_memories (npc_name, content, status, created_at) VALUES (?1, ?2, ?3, ?4)",
        params![npc_name, content, "pending", now],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Get all pending memories for review.
pub fn get_pending_memories(conn: &Connection) -> Result<Vec<Memory>> {
    let mut stmt = conn.prepare(
        "SELECT id, npc_name, content, status, embedding, created_at
         FROM npc_memories WHERE status = 'pending' ORDER BY id ASC",
    )?;

    let memories = stmt
        .query_map(params![], |row| {
            let embedding_blob: Option<Vec<u8>> = row.get(4)?;
            let embedding = embedding_blob
                .and_then(|blob| serde_json::from_slice::<Vec<f64>>(&blob).ok());
            let status_str: String = row.get(3)?;
            Ok(Memory {
                id: row.get(0)?,
                npc_name: row.get(1)?,
                content: row.get(2)?,
                status: MemoryStatus::from_str(&status_str),
                embedding,
                created_at: row.get(5)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(memories)
}

/// Approve or reject a memory.
pub fn update_memory_status(conn: &Connection, id: i64, status: MemoryStatus) -> Result<()> {
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE npc_memories SET status = ?1, updated_at = ?2 WHERE id = ?3",
        params![status.as_str(), now, id],
    )?;
    Ok(())
}

/// Store an embedding for a memory.
pub fn set_memory_embedding(conn: &Connection, id: i64, embedding: &[f64]) -> Result<()> {
    let blob = serde_json::to_vec(embedding)
        .map_err(|e| crate::error::NpcError::Other(format!("Failed to serialize embedding: {}", e)))?;
    conn.execute(
        "UPDATE npc_memories SET embedding = ?1 WHERE id = ?2",
        params![blob, id],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS npc_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                npc_name TEXT NOT NULL,
                team_name TEXT,
                content TEXT NOT NULL,
                embedding BLOB,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT
            );"
        ).unwrap();
        conn
    }

    #[test]
    fn test_save_and_get_pending() {
        let conn = setup_test_db();

        let id1 = save_memory(&conn, "sibiji", "Rust is fast").unwrap();
        let id2 = save_memory(&conn, "sibiji", "NPC systems are cool").unwrap();
        assert!(id1 > 0);
        assert!(id2 > id1);

        let pending = get_pending_memories(&conn).unwrap();
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].npc_name, "sibiji");
        assert_eq!(pending[0].content, "Rust is fast");
        assert_eq!(pending[0].status, MemoryStatus::Pending);
    }

    #[test]
    fn test_approve_reject() {
        let conn = setup_test_db();

        let id1 = save_memory(&conn, "alicanto", "fact one").unwrap();
        let id2 = save_memory(&conn, "alicanto", "fact two").unwrap();

        update_memory_status(&conn, id1, MemoryStatus::Approved).unwrap();
        update_memory_status(&conn, id2, MemoryStatus::Rejected).unwrap();

        // No pending memories left.
        let pending = get_pending_memories(&conn).unwrap();
        assert_eq!(pending.len(), 0);
    }

    #[test]
    fn test_set_embedding() {
        let conn = setup_test_db();
        let id = save_memory(&conn, "test", "embedding test").unwrap();

        let emb = vec![0.1, 0.2, 0.3, 0.4];
        set_memory_embedding(&conn, id, &emb).unwrap();

        // Read back and verify.
        let blob: Vec<u8> = conn
            .query_row(
                "SELECT embedding FROM npc_memories WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .unwrap();
        let decoded: Vec<f64> = serde_json::from_slice(&blob).unwrap();
        assert_eq!(decoded, emb);
    }
}
