use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Build a hierarchical command key from tool name + arguments.
///
/// Examples:
///   sh + {bash_command: "git commit -m foo"} -> "sh:git commit"
///   edit_file + {filepath: "src/main.rs"}    -> "edit_file:main.rs"
///   delegate + {target: "researcher"}         -> "delegate:researcher"
pub fn build_command_key(tool_name: &str, args: &serde_json::Value) -> String {
    match tool_name {
        "sh" => {
            if let Some(cmd) = args.get("bash_command").and_then(|v| v.as_str()) {
                let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
                if parts.is_empty() {
                    return "sh".to_string();
                }
                let base = format!("sh:{}", parts[0]);
                if parts.len() > 1 && !parts[1].starts_with('-') {
                    return format!("{} {}", base, parts[1]);
                }
                return base;
            }
            "sh".to_string()
        }
        "python" => "python".to_string(),
        "edit_file" => {
            if let Some(fp) = args.get("filepath").and_then(|v| v.as_str()) {
                let basename = Path::new(fp)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(fp);
                return format!("edit_file:{}", basename);
            }
            "edit_file".to_string()
        }
        "delegate" => {
            if let Some(target) = args.get("target").and_then(|v| v.as_str()) {
                return format!("delegate:{}", target);
            }
            "delegate".to_string()
        }
        other => other.to_string(),
    }
}

/// Hierarchical prefix match — longest matching prefix wins.
///
/// "sh:git commit -m foo" matches:
///   "sh:git commit" (wins — most specific)
///   "sh:git"
///   "sh"
pub fn match_permission(cmd_key: &str, rules: &HashMap<String, String>) -> Option<String> {
    if let Some(v) = rules.get(cmd_key) {
        return Some(v.clone());
    }

    let mut best: Option<&str> = None;
    let mut best_len = 0usize;

    for rule_key in rules.keys() {
        if cmd_key.starts_with(rule_key.as_str()) {
            let next = cmd_key.as_bytes().get(rule_key.len()).copied();
            let at_boundary = matches!(next, None | Some(b':') | Some(b' '));
            if at_boundary && rule_key.len() > best_len {
                best_len = rule_key.len();
                best = Some(rule_key.as_str());
            }
        }
    }

    best.and_then(|k| rules.get(k)).cloned()
}

/// Load permission rules from a YAML file.
/// Supports both flat `{key: value}` and nested `{rules: {key: value}}` structure.
pub fn load_permission_file(path: &str) -> HashMap<String, String> {
    let content = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return HashMap::new(),
    };
    let parsed: serde_yaml::Value = match serde_yaml::from_str(&content) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    let map = match &parsed {
        serde_yaml::Value::Mapping(m) => {
            // Check for nested "rules" key
            let inner = m
                .get("rules")
                .and_then(|v| v.as_mapping())
                .unwrap_or(m);
            inner
        }
        _ => return HashMap::new(),
    };

    map.iter()
        .filter_map(|(k, v)| {
            let key = k.as_str()?.to_string();
            let val = match v {
                serde_yaml::Value::String(s) => s.clone(),
                serde_yaml::Value::Bool(b) => b.to_string(),
                _ => v.as_str()?.to_string(),
            };
            Some((key, val))
        })
        .collect()
}

/// Tools that never need a permission prompt.
pub fn is_safe_tool(name: &str) -> bool {
    matches!(
        name,
        "chat" | "help" | "stop" | "screenshot" | "ask_form"
            | "config" | "switches" | "verbose" | "shh"
            | "usage" | "lookback" | "reload" | "init"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_command_key_sh() {
        let args = serde_json::json!({"bash_command": "git commit -m 'foo'"});
        assert_eq!(build_command_key("sh", &args), "sh:git commit");
    }

    #[test]
    fn test_build_command_key_sh_single() {
        let args = serde_json::json!({"bash_command": "ls -la"});
        assert_eq!(build_command_key("sh", &args), "sh:ls");
    }

    #[test]
    fn test_build_command_key_edit_file() {
        let args = serde_json::json!({"filepath": "src/main.rs"});
        assert_eq!(build_command_key("edit_file", &args), "edit_file:main.rs");
    }

    #[test]
    fn test_match_permission_exact() {
        let mut rules = HashMap::new();
        rules.insert("sh:git commit".to_string(), "auto".to_string());
        assert_eq!(
            match_permission("sh:git commit", &rules).as_deref(),
            Some("auto")
        );
    }

    #[test]
    fn test_match_permission_prefix() {
        let mut rules = HashMap::new();
        rules.insert("sh:git".to_string(), "ask".to_string());
        rules.insert("sh".to_string(), "deny".to_string());
        // "sh:git commit -m foo" should match "sh:git" (longer prefix wins)
        assert_eq!(
            match_permission("sh:git commit -m foo", &rules).as_deref(),
            Some("ask")
        );
    }

    #[test]
    fn test_match_permission_no_false_prefix() {
        let mut rules = HashMap::new();
        rules.insert("sh:git".to_string(), "auto".to_string());
        // "sh:gitstatus" should NOT match "sh:git" (not a boundary)
        assert_eq!(match_permission("sh:gitstatus", &rules), None);
    }
}
