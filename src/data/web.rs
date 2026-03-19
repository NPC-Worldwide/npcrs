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

/// Search Startpage via scraping.
pub async fn search_startpage(query: &str, num_results: usize) -> Result<Vec<SearchResult>> {
    let client = reqwest::Client::new();
    let resp = client.post("https://www.startpage.com/sp/search").form(&[("query", query), ("cat", "web")]).header("User-Agent", "Mozilla/5.0").header("Accept", "text/html").send().await?;
    if !resp.status().is_success() { return Err(NpcError::LlmRequest("Startpage search failed".into())); }
    let html = resp.text().await?;
    let mut results = Vec::new();
    let link_re = regex::Regex::new(r#"<a[^>]*href="(https?://[^"]*)"[^>]*>"#).unwrap();
    for cap in link_re.captures_iter(&html) { if results.len() >= num_results { break; } results.push(SearchResult { title: String::new(), url: cap[1].to_string(), snippet: String::new() }); }
    Ok(results)
}

/// Perplexity search via API. Returns (answer, citations).
pub async fn search_perplexity(query: &str, api_key: Option<&str>, max_tokens: Option<u32>, temperature: Option<f64>, top_p: Option<f64>) -> Result<(String, Vec<String>)> {
    let key = api_key.map(String::from).or_else(|| std::env::var("PERPLEXITY_API_KEY").ok()).ok_or_else(|| NpcError::LlmRequest("PERPLEXITY_API_KEY not set".into()))?;
    let body = serde_json::json!({"model": "sonar", "messages": [{"role": "system", "content": "Be precise and concise."}, {"role": "user", "content": query}], "max_tokens": max_tokens.unwrap_or(400), "temperature": temperature.unwrap_or(0.2), "top_p": top_p.unwrap_or(0.9), "stream": false});
    let client = reqwest::Client::new();
    let resp = client.post("https://api.perplexity.ai/chat/completions").header("Authorization", format!("Bearer {}", key)).header("Content-Type", "application/json").json(&body).send().await?;
    if !resp.status().is_success() { let t = resp.text().await.unwrap_or_default(); return Err(NpcError::LlmRequest(format!("Perplexity error: {}", &t[..t.len().min(200)]))); }
    let data: serde_json::Value = resp.json().await?;
    let answer = data["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
    let citations = data.get("citations").and_then(|c| c.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
    Ok((answer, citations))
}

/// Exa search via API.
pub async fn search_exa(query: &str, api_key: Option<&str>, top_k: usize) -> Result<Vec<SearchResult>> {
    let key = api_key.map(String::from).or_else(|| std::env::var("EXA_API_KEY").ok()).ok_or_else(|| NpcError::LlmRequest("EXA_API_KEY not set".into()))?;
    let body = serde_json::json!({"query": query, "contents": {"text": true}, "numResults": top_k});
    let client = reqwest::Client::new();
    let resp = client.post("https://api.exa.ai/search").header("x-api-key", &key).header("Content-Type", "application/json").json(&body).send().await?;
    if !resp.status().is_success() { let t = resp.text().await.unwrap_or_default(); return Err(NpcError::LlmRequest(format!("Exa error: {}", t))); }
    let data: serde_json::Value = resp.json().await?;
    let mut results = Vec::new();
    if let Some(arr) = data.get("results").and_then(|r| r.as_array()) { for item in arr.iter().take(top_k) { results.push(SearchResult { title: item["title"].as_str().unwrap_or("").to_string(), url: item["url"].as_str().unwrap_or("").to_string(), snippet: item["text"].as_str().unwrap_or("").chars().take(500).collect() }); } }
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
