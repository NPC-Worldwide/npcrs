
use crate::error::{NpcError, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FileContent {
    pub content: String,
    pub file_type: String,
    pub path: String,
    pub size: usize,
}

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

pub fn load_csv(path: &str) -> Result<String> { std::fs::read_to_string(path).map_err(|e| NpcError::FileLoad { path: path.into(), source: e }) }

pub fn load_json(path: &str) -> Result<String> {
    let data = std::fs::read_to_string(path).map_err(|e| NpcError::FileLoad { path: path.into(), source: e })?;
    match serde_json::from_str::<serde_json::Value>(&data) { Ok(val) => Ok(serde_json::to_string_pretty(&val).unwrap_or(data)), Err(_) => Ok(data) }
}

pub fn load_excel(path: &str) -> Result<String> {
    use calamine::{Reader, open_workbook_auto};
    let mut workbook = open_workbook_auto(path)
        .map_err(|e| NpcError::Other(format!("Excel open failed: {}", e)))?;
    let mut output = String::new();
    for sheet_name in workbook.sheet_names().to_vec() {
        if let Ok(range) = workbook.worksheet_range(&sheet_name) {
            output.push_str(&format!("--- {} ---\n", sheet_name));
            for row in range.rows() {
                let cells: Vec<String> = row.iter().map(|c| format!("{}", c)).collect();
                output.push_str(&cells.join("\t"));
                output.push('\n');
            }
            output.push('\n');
        }
    }
    Ok(output)
}

pub fn load_image(path: &str) -> Result<String> {
    let raw = std::fs::read(path).map_err(|e| NpcError::FileLoad { path: path.into(), source: e })?;
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
    let ext = Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("png");
    Ok(format!("[Image: {} ({} bytes)]\ndata:image/{};base64,{}", path, raw.len(), ext, b64))
}

pub fn load_pdf(path: &str) -> String { extract_pdf_text(path) }

pub fn load_docx(path: &str) -> Result<String> {
    let file = std::fs::File::open(path).map_err(|e| NpcError::FileLoad { path: path.into(), source: e })?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| NpcError::Other(format!("DOCX zip open failed: {}", e)))?;
    let mut text = String::new();
    if let Ok(mut doc_xml) = archive.by_name("word/document.xml") {
        let mut xml = String::new();
        std::io::Read::read_to_string(&mut doc_xml, &mut xml).ok();
        for cap in regex::Regex::new(r"<w:t[^>]*>(.*?)</w:t>").unwrap().captures_iter(&xml) {
            text.push_str(&cap[1]);
        }
        text = regex::Regex::new(r"</w:p>").unwrap().replace_all(&text, "\n").to_string();
    }
    Ok(text)
}

pub fn load_pptx(path: &str) -> Result<String> {
    let file = std::fs::File::open(path).map_err(|e| NpcError::FileLoad { path: path.into(), source: e })?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| NpcError::Other(format!("PPTX zip open failed: {}", e)))?;
    let mut text = String::new();
    let tag_strip = regex::Regex::new(r"<[^>]+>").unwrap();
    for i in 0..archive.len() {
        if let Ok(entry) = archive.by_index(i) {
            let name = entry.name().to_string();
            if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
                let mut xml = String::new();
                let mut reader = std::io::BufReader::new(entry);
                std::io::Read::read_to_string(&mut reader, &mut xml).ok();
                for cap in regex::Regex::new(r"<a:t>(.*?)</a:t>").unwrap().captures_iter(&xml) {
                    text.push_str(&cap[1]);
                    text.push(' ');
                }
                text.push('\n');
            }
        }
    }
    Ok(text)
}

pub fn load_html(path: &str) -> Result<String> {
    let raw = std::fs::read_to_string(path).map_err(|e| NpcError::FileLoad { path: path.into(), source: e })?;
    Ok(super::text::strip_html(&raw))
}

pub fn load_audio(path: &str) -> Result<String> {
    match super::audio::transcribe_audio_file(path, None) { Ok(t) if !t.is_empty() => Ok(t), _ => Ok(format!("[Audio file at {}; no transcript]", path)) }
}

pub fn load_video(path: &str) -> Result<String> {
    match super::video::summarize_video_file(path, None, 600) { Ok(s) => Ok(s), Err(_) => Ok(format!("[Video file at {}]", path)) }
}

pub fn chunk_text_simple(content: &str, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < content.len() {
        let mut end = (start + chunk_size).min(content.len());
        while end > start && !content.is_char_boundary(end) { end -= 1; }
        let chunk = content[start..end].trim();
        if !chunk.is_empty() { chunks.push(chunk.to_string()); }
        start = end;
    }
    chunks
}

pub fn load_file_contents_chunked(path: &str, chunk_size: Option<usize>) -> Vec<String> {
    let cs = chunk_size.unwrap_or(8000);
    match load_file_contents(path) {
        Ok(fc) => if fc.content.is_empty() { vec![] } else { chunk_text_simple(&fc.content, cs) },
        Err(e) => vec![format!("Error loading {}: {}", path, e)],
    }
}

pub fn extension_category(ext: &str) -> &'static str {
    match ext.to_uppercase().as_str() {
        "PNG" | "JPG" | "JPEG" | "GIF" | "SVG" | "WEBP" | "BMP" | "TIFF" => "images",
        "MP4" | "AVI" | "MOV" | "WMV" | "MPG" | "MPEG" | "WEBM" | "MKV" => "videos",
        "DOCX" | "PPTX" | "PDF" | "XLSX" | "XLS" | "TXT" | "CSV" | "MD" | "HTML" | "HTM" => "documents",
        "MP3" | "WAV" | "M4A" | "AAC" | "FLAC" | "OGG" => "audio",
        "ZIP" | "RAR" | "7Z" | "TAR" | "GZ" => "archives",
        _ => "unknown",
    }
}
