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
