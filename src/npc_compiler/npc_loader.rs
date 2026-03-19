use crate::error::{NpcError, Result};
use crate::npc_compiler::Npc;
use std::path::Path;

pub fn load_npc_from_file(path: impl AsRef<Path>) -> Result<Npc> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path).map_err(|e| NpcError::FileLoad {
        path: path.display().to_string(),
        source: e,
    })?;

    let raw = if raw.starts_with("#!") {
        raw.splitn(2, '\n').nth(1).unwrap_or("").to_string()
    } else {
        raw
    };

    let processed = preprocess_npc_yaml(&raw);

    let mut npc: Npc =
        serde_yaml::from_str(&processed).map_err(|e| NpcError::YamlParse {
            path: path.display().to_string(),
            source: e,
        })?;

    npc.source_path = Some(path.display().to_string());

    for mcp in &mut npc.mcp_servers {
        mcp.path = shellexpand::tilde(&mcp.path).to_string();
    }

    Ok(npc)
}

fn preprocess_npc_yaml(raw: &str) -> String {
    let mut output = String::with_capacity(raw.len());
    let mut in_for_block = false;
    let mut for_glob_pattern: Option<String> = None;

    for line in raw.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("{%") && trimmed.contains("jinxes_list") {
            if let Some(pattern) = extract_jinxes_list_pattern(trimmed) {
                for_glob_pattern = Some(pattern);
                in_for_block = true;
            }
            continue;
        }

        if trimmed.starts_with("{%") && trimmed.contains("endfor") {
            if in_for_block {
                if let Some(ref pattern) = for_glob_pattern {
                    let prefix = pattern.trim_end_matches('*').trim_end_matches('_');
                    let names = expand_jinx_glob(pattern);
                    for name in names {
                        output.push_str(&format!("  - {}\n", name));
                    }
                }
                in_for_block = false;
                for_glob_pattern = None;
            }
            continue;
        }

        if in_for_block {
            continue;
        }

        if trimmed.starts_with("{%") {
            continue;
        }

        if let Some(extracted) = extract_jinx_call(trimmed) {
            let indent = &line[..line.len() - line.trim_start().len()];
            output.push_str(&format!("{indent}- {extracted}\n"));
        } else {
            output.push_str(line);
            output.push('\n');
        }
    }

    output
}

fn extract_jinxes_list_pattern(line: &str) -> Option<String> {
    let start = line.find("jinxes_list(")?;
    let rest = &line[start + "jinxes_list(".len()..];
    let end = rest.find(')')?;
    let pattern = rest[..end].trim().trim_matches('\'').trim_matches('"');
    Some(pattern.to_string())
}

fn expand_jinx_glob(pattern: &str) -> Vec<String> {
    let locations = [
        shellexpand::tilde("~/.npcsh/npc_team/jinxes").to_string(),
        "./npc_team/jinxes".to_string(),
    ];

    for base in &locations {
        let glob_pattern = format!("{}/{}.jinx", base, pattern);
        if let Ok(paths) = glob::glob(&glob_pattern) {
            let names: Vec<String> = paths
                .filter_map(|p| p.ok())
                .filter_map(|p| {
                    p.file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                })
                .collect();
            if !names.is_empty() {
                return names;
            }
        }
    }

    Vec::new()
}

fn extract_jinx_call(line: &str) -> Option<String> {
    let line = line.trim().trim_start_matches('-').trim();

    if !line.starts_with("{{") || !line.ends_with("}}") {
        return None;
    }

    let inner = line
        .trim_start_matches("{{")
        .trim_end_matches("}}")
        .trim();

    if !inner.starts_with("Jinx(") {
        return None;
    }

    let name_part = inner
        .trim_start_matches("Jinx(")
        .trim_end_matches(')')
        .trim()
        .trim_matches('\'')
        .trim_matches('"');

    if name_part.is_empty() {
        return None;
    }

    Some(name_part.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_jinx_call() {
        assert_eq!(
            extract_jinx_call("- {{ Jinx('edit_file') }}"),
            Some("edit_file".to_string())
        );
        assert_eq!(
            extract_jinx_call("  - {{ Jinx(\"web_search\") }}"),
            Some("web_search".to_string())
        );
        assert_eq!(extract_jinx_call("- plain_string"), None);
        assert_eq!(extract_jinx_call("name: value"), None);
    }

    #[test]
    fn test_preprocess_strips_jinja() {
        let input = r#"
name: test_npc
primary_directive: Do things
jinxes:
  - {{ Jinx('edit_file') }}
  - {{ Jinx('sh') }}
  - plain_jinx
"#;
        let output = preprocess_npc_yaml(input);
        assert!(output.contains("- edit_file"));
        assert!(output.contains("- sh"));
        assert!(output.contains("- plain_jinx"));
        assert!(!output.contains("Jinx("));
    }
}
