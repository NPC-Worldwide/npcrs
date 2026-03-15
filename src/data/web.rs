//! Web search and URL fetching.

use crate::error::{NpcError, Result};

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Search the web via DuckDuckGo.
pub async fn search_web(query: &str) -> Result<Vec<SearchResult>> {
    let client = reqwest::Client::new();
    let url = format!(
        "https://html.duckduckgo.com/html/?q={}",
        query.replace(' ', "+")
    );
    let resp = client
        .get(&url)
        .header("User-Agent", "npcsh/1.0")
        .send()
        .await?;
    let html = resp.text().await?;

    // Parse results from DDG HTML
    let mut results = Vec::new();
    for cap in regex::Regex::new(r#"<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>"#)
        .unwrap()
        .captures_iter(&html)
    {
        if results.len() >= 5 {
            break;
        }
        results.push(SearchResult {
            url: cap[1].to_string(),
            title: cap[2].to_string(),
            snippet: String::new(),
        });
    }
    Ok(results)
}

/// Fetch a URL and return text content.
pub async fn fetch_url(url: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .header("User-Agent", "npcsh/1.0")
        .send()
        .await?;
    Ok(resp.text().await?)
}
