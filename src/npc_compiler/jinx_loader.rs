use crate::error::{NpcError, Result};
use crate::npc_compiler::Jinx;
use std::collections::HashMap;
use std::path::Path;
use walkdir::WalkDir;

/// Load a single Jinx from a .jinx YAML file.
pub fn load_jinx_from_file(path: impl AsRef<Path>) -> Result<Jinx> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path).map_err(|e| NpcError::FileLoad {
        path: path.display().to_string(),
        source: e,
    })?;

    // Strip shebang if present (for executable .jinx files)
    let raw = if raw.starts_with("#!") {
        raw.splitn(2, '\n').nth(1).unwrap_or("").to_string()
    } else {
        raw
    };

    // Strip Jinja2 template syntax that Tera won't understand
    let cleaned = strip_jinja2_specifics(&raw);

    let mut jinx: Jinx =
        serde_yaml::from_str(&cleaned).map_err(|e| NpcError::YamlParse {
            path: path.display().to_string(),
            source: e,
        })?;

    jinx.source_path = Some(path.display().to_string());

    Ok(jinx)
}

/// Load all jinxes from a directory tree, returning name → Jinx map.
///
/// Walks the directory recursively, loading all .jinx files.
/// The jinx name in the map is the `jinx_name` from the YAML,
/// falling back to the filename stem.
pub fn load_jinxes_from_directory(dir: impl AsRef<Path>) -> Result<HashMap<String, Jinx>> {
    let dir = dir.as_ref();
    let mut jinxes = HashMap::new();

    if !dir.exists() {
        return Ok(jinxes);
    }

    for entry in WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "jinx") {
            match load_jinx_from_file(path) {
                Ok(jinx) => {
                    let name = if jinx.name.is_empty() {
                        path.file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string()
                    } else {
                        jinx.name.clone()
                    };
                    jinxes.insert(name, jinx);
                }
                Err(e) => {
                    tracing::warn!("Failed to load jinx {}: {}", path.display(), e);
                }
            }
        }
    }

    Ok(jinxes)
}

/// Strip Jinja2-specific syntax that doesn't translate to Tera.
///
/// Tera is Jinja2-compatible for most things, but some Python-specific
/// constructs in the npcpy jinx files need preprocessing.
fn strip_jinja2_specifics(raw: &str) -> String {
    // For now, pass through. Tera handles most Jinja2 syntax.
    // We'll add specific transformations as needed.
    raw.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp_jinx(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::with_suffix(".jinx").unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_load_simple_jinx() {
        let f = write_temp_jinx(
            r#"
jinx_name: test_jinx
description: A test jinx
inputs:
  - query
steps:
  - name: run
    engine: bash
    code: echo "hello {{ query }}"
"#,
        );
        let jinx = load_jinx_from_file(f.path()).unwrap();
        assert_eq!(jinx.name, "test_jinx");
        assert_eq!(jinx.inputs.len(), 1);
        assert_eq!(jinx.inputs[0].name, "query");
        assert!(jinx.inputs[0].default.is_none());
        assert_eq!(jinx.steps.len(), 1);
        assert_eq!(jinx.steps[0].engine, "bash");
    }

    #[test]
    fn test_load_jinx_with_defaults() {
        let f = write_temp_jinx(
            r#"
jinx_name: edit_file
description: Edit files
inputs:
  - path:
      description: "File path"
  - action: "create"
  - content
steps:
  - name: edit
    engine: python
    code: |
      print("editing")
"#,
        );
        let jinx = load_jinx_from_file(f.path()).unwrap();
        assert_eq!(jinx.inputs.len(), 3);
        assert_eq!(jinx.inputs[0].name, "path");
        assert!(jinx.inputs[0].description.is_some());
        assert_eq!(jinx.inputs[1].name, "action");
        assert_eq!(jinx.inputs[1].default.as_deref(), Some("create"));
        assert_eq!(jinx.inputs[2].name, "content");
        assert!(jinx.inputs[2].default.is_none());
    }
}
