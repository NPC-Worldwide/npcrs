
use crate::error::Result;
use crate::memory::embeddings::{cosine_similarity, get_embeddings};
use rusqlite::params;

#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    pub content: String,
    pub source: String,
    pub score: f64,
}

pub async fn search_similar_texts(
    query: &str,
    db_path: &str,
    model: &str,
    provider: &str,
    top_k: usize,
) -> Result<Vec<MemorySearchResult>> {
    let query_embedding = get_embeddings(query, model, provider, None).await?;

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
            let embedding: Vec<f64> = serde_json::from_slice(&blob).ok()?;
            let score = cosine_similarity(&query_embedding, &embedding);
            Some(MemorySearchResult {
                content,
                source: "embedding".to_string(),
                score,
            })
        })
        .collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    scored.truncate(top_k);

    Ok(scored)
}

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

pub async fn execute_search_command(command: &str, provider: Option<&str>, num_results: usize) -> Result<std::collections::HashMap<String, serde_json::Value>> {
    let results = crate::data::web::search_web(command, num_results, provider.unwrap_or("duckduckgo"), None).await?;
    let output: String = results.iter().enumerate().map(|(i, r)| {
        format!("{}. {} - {}\n   {}", i + 1, r.title, r.url, r.snippet)
    }).collect::<Vec<_>>().join("\n\n");
    let citations: Vec<String> = results.iter().map(|r| r.url.clone()).collect();
    let mut r = std::collections::HashMap::new();
    r.insert("output".into(), serde_json::json!(format!("{}\n\nCitation Links: {:?}", output, citations)));
    r.insert("messages".into(), serde_json::json!([
        {"role": "user", "content": command},
        {"role": "assistant", "content": output}
    ]));
    Ok(r)
}

pub async fn execute_rag_command(
    command: &str,
    db_path: &str,
    embedding_model: &str,
    embedding_provider: &str,
    top_k: usize,
) -> Result<std::collections::HashMap<String, serde_json::Value>> {
    let similar = search_similar_texts(command, db_path, embedding_model, embedding_provider, top_k).await?;
    let context: String = similar.iter().map(|r| r.content.clone()).collect::<Vec<_>>().join("\n\n---\n\n");
    let mut r = std::collections::HashMap::new();
    r.insert("context".into(), serde_json::json!(context));
    r.insert("results".into(), serde_json::json!(similar.len()));
    Ok(r)
}

pub async fn execute_brainblast_command(
    command: &str,
    db_path: &str,
    embedding_model: &str,
    embedding_provider: &str,
    model: &str,
    provider: &str,
    top_k: usize,
) -> Result<std::collections::HashMap<String, serde_json::Value>> {
    let similar = search_similar_texts(command, db_path, embedding_model, embedding_provider, top_k).await?;
    let context: String = similar.iter().map(|r| r.content.clone()).collect::<Vec<_>>().join("\n\n");
    let prompt = format!(
        "Using the following context from memory, answer the question.\n\nContext:\n{}\n\nQuestion: {}",
        context, command
    );
    let result = crate::llm_funcs::get_llm_response(&prompt, None, Some(model), Some(provider), None, &[], None).await?;
    let mut r = std::collections::HashMap::new();
    r.insert("output".into(), serde_json::json!(result.response.unwrap_or_default()));
    r.insert("context_used".into(), serde_json::json!(similar.len()));
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_search_empty_db() {
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
