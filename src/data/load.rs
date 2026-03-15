//! File loading with type detection.

use crate::error::{NpcError, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FileContent {
    pub content: String,
    pub file_type: String,
    pub path: String,
    pub size: usize,
}

/// Load file contents with type detection.
pub fn load_file_contents(path: &str) -> Result<FileContent> {
    let path_obj = Path::new(path);
    let ext = path_obj
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let raw = std::fs::read(path).map_err(|e| NpcError::FileLoad {
        path: path.to_string(),
        source: e,
    })?;
    let size = raw.len();

    let content = match ext.as_str() {
        "pdf" => extract_pdf_text(path),
        "html" | "htm" => {
            let html = String::from_utf8_lossy(&raw).to_string();
            super::text::strip_html(&html)
        }
        _ => String::from_utf8_lossy(&raw).to_string(),
    };

    Ok(FileContent {
        content,
        file_type: ext,
        path: path.to_string(),
        size,
    })
}

fn extract_pdf_text(path: &str) -> String {
    std::process::Command::new("pdftotext")
        .args(["-nopgbrk", path, "-"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_else(|| format!("[PDF extraction failed for {}]", path))
}
