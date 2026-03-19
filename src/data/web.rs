//! Web search and URL fetching.

use crate::error::{NpcError, Result};

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Search the web — mirrors npcpy's search_web(). Dispatches to provider.
pub async fn search_web(query: &str, num_results: usize, provider: &str, api_key: Option<&str>) -> Result<Vec<SearchResult>> {
    match provider {
        "brave" => search_brave(query, num_results, api_key).await,
        "searxng" => search_searxng(query, num_results, None).await,
        "startpage" => search_startpage(query, num_results).await,
        "perplexity" => {
            let (answer, _citations) = search_perplexity(query, api_key, None, None, None).await?;
            Ok(vec![SearchResult { title: "Perplexity Answer".into(), url: String::new(), snippet: answer }])
        }
        "exa" => search_exa(query, api_key, num_results).await,
        _ => search_duckduckgo(query, num_results).await,
    }
}

/// DuckDuckGo search via HTML lite endpoint.
pub async fn search_duckduckgo(query: &str, num_results: usize) -> Result<Vec<SearchResult>> {
    let client = reqwest::Client::new();
    let resp = client
        .get("https://html.duckduckgo.com/html/")
        .query(&[("q", query)])
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0")
        .send()
        .await?;
    let html = resp.text().await?;

    let mut results = Vec::new();
    let link_re = regex::Regex::new(r#"<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>"#).unwrap();
    let snippet_re = regex::Regex::new(r#"<a[^>]*class="result__snippet"[^>]*>(.*?)</a>"#).unwrap();
    let tag_strip = regex::Regex::new(r"<[^>]+>").unwrap();

    let links: Vec<(String, String)> = link_re.captures_iter(&html)
        .map(|cap| (cap[1].to_string(), tag_strip.replace_all(&cap[2], "").to_string()))
        .collect();
    let snippets: Vec<String> = snippet_re.captures_iter(&html)
        .map(|cap| tag_strip.replace_all(&cap[1], "").to_string())
        .collect();

    for (i, (url, title)) in links.iter().enumerate() {
        if results.len() >= num_results { break; }
        let actual_url = if url.contains("uddg=") {
            url.split("uddg=").nth(1)
                .and_then(|u| urlencoding::decode(u).ok())
                .map(|u| u.into_owned())
                .unwrap_or_else(|| url.clone())
        } else {
            url.clone()
        };
        results.push(SearchResult {
            title: title.clone(),
            url: actual_url,
            snippet: snippets.get(i).cloned().unwrap_or_default(),
        });
    }
    Ok(results)
}

/// Brave search via API.
pub async fn search_brave(query: &str, num_results: usize, api_key: Option<&str>) -> Result<Vec<SearchResult>> {
    let key = api_key.map(String::from).or_else(|| std::env::var("BRAVE_API_KEY").ok())
        .ok_or_else(|| NpcError::LlmRequest("BRAVE_API_KEY not set".into()))?;
    let client = reqwest::Client::new();
    let resp = client.get("https://api.search.brave.com/res/v1/web/search")
        .query(&[("q", query), ("count", &num_results.to_string())])
        .header("X-Subscription-Token", &key)
        .header("Accept", "application/json")
        .send().await?;
    let json: serde_json::Value = resp.json().await?;
    let mut results = Vec::new();
    if let Some(web) = json.get("web").and_then(|w| w.get("results")).and_then(|r| r.as_array()) {
        for item in web.iter().take(num_results) {
            results.push(SearchResult {
                title: item["title"].as_str().unwrap_or("").to_string(),
                url: item["url"].as_str().unwrap_or("").to_string(),
                snippet: item["description"].as_str().unwrap_or("").to_string(),
            });
        }
    }
    Ok(results)
}

/// SearxNG search via public instances.
pub async fn search_searxng(query: &str, num_results: usize, instance_url: Option<&str>) -> Result<Vec<SearchResult>> {
    let instances = if let Some(url) = instance_url { vec![url.to_string()] }
    else if let Ok(url) = std::env::var("SEARXNG_URL") { vec![url] }
    else { vec!["https://search.sapti.me".into(), "https://searx.work".into()] };
    let client = reqwest::Client::new();
    for instance in &instances {
        let url = format!("{}/search", instance);
        if let Ok(resp) = client.get(&url).query(&[("q", query), ("format", "json"), ("categories", "general")])
            .header("User-Agent", "npcsh/1.0").send().await {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                let mut results = Vec::new();
                if let Some(arr) = json.get("results").and_then(|r| r.as_array()) {
                    for item in arr.iter().take(num_results) {
                        results.push(SearchResult {
                            title: item["title"].as_str().unwrap_or("").to_string(),
                            url: item["url"].as_str().unwrap_or("").to_string(),
                            snippet: item["content"].as_str().unwrap_or("").to_string(),
                        });
                    }
                }
                if !results.is_empty() { return Ok(results); }
            }
        }
    }
    Err(NpcError::LlmRequest("All SearxNG instances failed".into()))
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
