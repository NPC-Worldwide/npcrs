use crate::error::{NpcError, Result};
use crate::npc::Npc;
use std::path::Path;

/// Load an NPC from a .npc YAML file.
///
/// Handles the quirks of the .npc format:
/// - Jinja2 template expressions in the jinxes list (stripped to plain names)
/// - Optional fields with various default behaviors
/// - Shell expansion for paths (~/)
pub fn load_npc_from_file(path: impl AsRef<Path>) -> Result<Npc> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path).map_err(|e| NpcError::FileLoad {
        path: path.display().to_string(),
        source: e,
    })?;

    // Pre-process: strip Jinja2 template calls like {{ Jinx('name') }}
    // and convert them to plain string names for serde
    let processed = preprocess_npc_yaml(&raw);

    let mut npc: Npc =
        serde_yaml::from_str(&processed).map_err(|e| NpcError::YamlParse {
            path: path.display().to_string(),
            source: e,
        })?;

    npc.source_path = Some(path.display().to_string());

    // Expand ~ in MCP server paths
    for mcp in &mut npc.mcp_servers {
        mcp.path = shellexpand::tilde(&mcp.path).to_string();
    }

    Ok(npc)
}

/// Pre-process .npc YAML to handle Jinja2 template expressions.
///
/// Converts `{{ Jinx('edit_file') }}` → `edit_file`
/// Expands `{% for j in jinxes_list('pattern') %}` blocks by globbing.
/// Strips other `{% %}` control flow.
fn preprocess_npc_yaml(raw: &str) -> String {
    let mut output = String::with_capacity(raw.len());
    let mut in_for_block = false;
    let mut for_glob_pattern: Option<String> = None;

    // Determine jinxes directory from the NPC file's sibling
    // (will be resolved at team load time, but we try to expand here)

    for line in raw.lines() {
        let trimmed = line.trim();

        // Handle {% for j in jinxes_list('pattern') %} blocks
        if trimmed.starts_with("{%") && trimmed.contains("jinxes_list") {
            // Extract the glob pattern
            if let Some(pattern) = extract_jinxes_list_pattern(trimmed) {
                for_glob_pattern = Some(pattern);
                in_for_block = true;
            }
            continue;
        }

        // Handle {% endfor %}
        if trimmed.starts_with("{%") && trimmed.contains("endfor") {
            if in_for_block {
                // Expand the glob pattern into jinx names
                if let Some(ref pattern) = for_glob_pattern {
                    // The pattern is like 'lib/browser_*' — extract the prefix
                    let prefix = pattern.trim_end_matches('*').trim_end_matches('_');
                    // Generate likely jinx names from common patterns
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

        // Skip lines inside for blocks (they reference {{ j }})
        if in_for_block {
            continue;
        }

        // Skip other Jinja2 control flow lines
        if trimmed.starts_with("{%") {
            continue;
        }

        // Replace {{ Jinx('name') }} with just 'name'
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

/// Extract glob pattern from `{% for j in jinxes_list('lib/browser_*') %}`.
fn extract_jinxes_list_pattern(line: &str) -> Option<String> {
    let start = line.find("jinxes_list(")?;
    let rest = &line[start + "jinxes_list(".len()..];
    let end = rest.find(')')?;
    let pattern = rest[..end].trim().trim_matches('\'').trim_matches('"');
    Some(pattern.to_string())
}

/// Expand a jinx glob pattern into names.
/// For example: "lib/browser_*" → ["browser_action", "browser_screenshot", ...]
fn expand_jinx_glob(pattern: &str) -> Vec<String> {
    // Try to find the actual jinx files by globbing common locations
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

/// Extract jinx name from a Jinja2 template call.
/// `- {{ Jinx('edit_file') }}` → Some("edit_file")
/// `- {{ Jinx("web_search") }}` → Some("web_search")
fn extract_jinx_call(line: &str) -> Option<String> {
    let line = line.trim().trim_start_matches('-').trim();

    // Match {{ Jinx('name') }} or {{ Jinx("name") }}
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
