//! Text processing utilities.

/// Chunk text into overlapping segments.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() || chunk_size == 0 {
        return vec![];
    }
    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        chunks.push(chars[start..end].iter().collect());
        start += step;
        if end == chars.len() {
            break;
        }
    }
    chunks
}

/// Extract URLs from text.
pub fn extract_urls(text: &str) -> Vec<String> {
    regex::Regex::new(r#"https?://[^\s<>"')\]]+"#)
        .unwrap()
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Strip HTML tags from text.
pub fn strip_html(html: &str) -> String {
    regex::Regex::new(r"<[^>]+>")
        .unwrap()
        .replace_all(html, "")
        .to_string()
}

/// Keyword-based RAG search over a file map. Returns (filename, snippet) pairs.
pub fn rag_search(query: &str, text_data: &std::collections::HashMap<String, String>, similarity_threshold: f64) -> Vec<(String, String)> {
    let ql = query.to_lowercase();
    let qw: std::collections::HashSet<&str> = ql.split_whitespace().collect();
    let mut results = Vec::new();
    for (filename, content) in text_data {
        let lines: Vec<&str> = content.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let ll = line.to_lowercase();
            let lw: std::collections::HashSet<&str> = ll.split_whitespace().collect();
            if lw.is_empty() { continue; }
            let inter = qw.intersection(&lw).count();
            let score = inter as f64 / qw.len().max(1) as f64;
            if score >= similarity_threshold {
                let s = idx.saturating_sub(10);
                let e = (idx + 11).min(lines.len());
                results.push((filename.clone(), lines[s..e].join("\n")));
            }
        }
    }
    results
}

/// Keyword-based RAG search on a single text string.
pub fn rag_search_text(query: &str, text: &str, similarity_threshold: f64) -> Vec<String> {
    let ql = query.to_lowercase();
    let qw: std::collections::HashSet<&str> = ql.split_whitespace().collect();
    let sentences: Vec<&str> = text.split('.').collect();
    let mut results = Vec::new();
    for (idx, sentence) in sentences.iter().enumerate() {
        let sl = sentence.to_lowercase();
        let sw: std::collections::HashSet<&str> = sl.split_whitespace().collect();
        if sw.is_empty() { continue; }
        let inter = qw.intersection(&sw).count();
        let score = inter as f64 / qw.len().max(1) as f64;
        if score >= similarity_threshold {
            let s = idx.saturating_sub(10);
            let e = (idx + 11).min(sentences.len());
            results.push(sentences[s..e].join(". "));
        }
    }
    results
}

/// Load all text files in a directory recursively.
pub fn load_all_files(directory: &str, extensions: Option<&[&str]>, depth: usize) -> std::collections::HashMap<String, String> {
    let default_exts = [".txt", ".md", ".py", ".java", ".c", ".cpp", ".html", ".css", ".js", ".ts", ".tsx", ".npc"];
    let exts = extensions.unwrap_or(&default_exts);
    let mut text_data = std::collections::HashMap::new();
    if depth < 1 { return text_data; }
    let entries = match std::fs::read_dir(directory) { Ok(e) => e, Err(_) => return text_data };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            let ps = path.to_string_lossy().to_string();
            if exts.iter().any(|ext| ps.ends_with(ext)) {
                if let Ok(content) = std::fs::read_to_string(&path) { text_data.insert(ps, content); }
            }
        } else if path.is_dir() {
            text_data.extend(load_all_files(&path.to_string_lossy(), extensions, depth - 1));
        }
    }
    text_data
}
