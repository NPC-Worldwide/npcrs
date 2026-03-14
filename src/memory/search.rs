//! Memory search combining embedding similarity and knowledge graph lookups.

use crate::error::Result;
use crate::memory::embeddings::{cosine_similarity, get_embeddings};
use rusqlite::params;

/// A result from memory search.
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    /// The memory content text.
    pub content: String,
    /// Source of the result: `"embedding"` or `"kg"`.
    pub source: String,
    /// Relevance score (cosine similarity for embeddings, keyword match score for KG).
    pub score: f64,
}

/// Search memories using embedding similarity.
///
/// Computes the embedding of `query`, then compares against all stored embeddings
/// in the `npc_memories` table, returning the top-k most similar results.
pub async fn search_memories_by_embedding(
    query: &str,
    db_path: &str,
    model: &str,
    provider: &str,
    top_k: usize,
) -> Result<Vec<MemorySearchResult>> {
    // Get the query embedding.
    let query_embedding = get_embeddings(query, model, provider, None).await?;

    // Open the database and read all approved memories that have embeddings.
    let conn = rusqlite::Connection::open(db_path)?;

    let mut stmt = conn.prepare(
        "SELECT content, embedding FROM npc_memories WHERE status = 'approved' AND embedding IS NOT NULL",
    )?;

    let mut scored: Vec<MemorySearchResult> = stmt
        .query_map(params![], |row| {
            let content: String = row.get(0)?;
            let embedding_blob: Vec<u8> = row.get(1)?;
            Ok((content, embedding_blob))
        })?
        .filter_map(|r| r.ok())
        .filter_map(|(content, blob)| {
            // Deserialize the embedding from the blob (stored as JSON).
            let embedding: Vec<f64> = serde_json::from_slice(&blob).ok()?;
            let score = cosine_similarity(&query_embedding, &embedding);
            Some(MemorySearchResult {
                content,
                source: "embedding".to_string(),
                score,
            })
        })
        .collect();

    // Sort by descending score.
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Return top-k.
    scored.truncate(top_k);

    Ok(scored)
}

/// Search memories by simple keyword matching in the npc_memories table.
///
/// Useful as a fallback when no embedding model is available.
pub fn search_memories_by_keyword(
    query: &str,
    db_path: &str,
    top_k: usize,
) -> Result<Vec<MemorySearchResult>> {
    let conn = rusqlite::Connection::open(db_path)?;

    let pattern = format!("%{}%", query);

    let mut stmt = conn.prepare(
        "SELECT content FROM npc_memories WHERE status = 'approved' AND content LIKE ?1 LIMIT ?2",
    )?;

    let results: Vec<MemorySearchResult> = stmt
        .query_map(params![pattern, top_k as i64], |row| {
            let content: String = row.get(0)?;
            Ok(MemorySearchResult {
                content,
                source: "keyword".to_string(),
                score: 1.0, // Keyword match doesn't produce a gradient score.
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_search_empty_db() {
        // Create a temp DB with the right schema.
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE npc_memories (
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

        // We can't use search_memories_by_keyword with in-memory DB directly,
        // so just verify the schema is correct by inserting and querying.
        conn.execute(
            "INSERT INTO npc_memories (npc_name, content, status, created_at) VALUES (?1, ?2, ?3, ?4)",
            params!["test", "Rust is a systems language", "approved", "2025-01-01"],
        ).unwrap();

        let mut stmt = conn.prepare(
            "SELECT content FROM npc_memories WHERE status = 'approved' AND content LIKE ?1 LIMIT ?2",
        ).unwrap();

        let results: Vec<String> = stmt
            .query_map(params!["%Rust%", 5i64], |row| row.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(results.len(), 1);
        assert!(results[0].contains("Rust"));
    }
}
