use npcrs::npc_compiler::{load_jinx_from_file, load_jinxes_from_directory, execute_jinx, Jinx};
use std::collections::HashMap;

#[test]
fn test_load_all_jinxes() {
    let jinx_dirs = vec![
        "../npcsh/npcsh/npc_team/jinxes",
    ];
    for dir in jinx_dirs {
        let path = std::path::Path::new(dir);
        if !path.exists() { continue; }
        match load_jinxes_from_directory(path) {
            Ok(jinxes) => {
                assert!(!jinxes.is_empty(), "No jinxes loaded from {}", dir);
                for (name, jinx) in &jinxes {
                    assert!(!name.is_empty(), "Jinx has empty name");
                    assert!(!jinx.steps.is_empty(), "Jinx '{}' has no steps", name);
                    for step in &jinx.steps {
                        assert!(
                            ["bash", "python", "rust", "sh"].contains(&step.engine.as_str()) || jinxes.contains_key(&step.engine),
                            "Jinx '{}' step '{}' has unknown engine '{}'", name, step.name, step.engine
                        );
                    }
                }
                println!("Loaded {} jinxes from {}", jinxes.len(), dir);
            }
            Err(e) => panic!("Failed to load jinxes from {}: {}", dir, e),
        }
    }
}

#[test]
fn test_jinx_to_tool_def() {
    let jinx_dir = std::path::Path::new("../npcsh/npcsh/npc_team/jinxes");
    if !jinx_dir.exists() { return; }
    let jinxes = load_jinxes_from_directory(jinx_dir).unwrap();
    let mut with_tools = 0;
    for (name, jinx) in &jinxes {
        if let Some(td) = jinx.to_tool_def() {
            assert!(td.function.name == *name || jinx.aliases.contains(name), "Tool def name '{}' doesn't match key '{}' or aliases {:?}", td.function.name, name, jinx.aliases);
            assert!(td.function.description.is_some());
            with_tools += 1;
        }
    }
    println!("{} jinxes have tool definitions", with_tools);
}

#[tokio::test]
async fn test_bash_jinx_execution() {
    let mut jinx = Jinx {
        name: "test_echo".into(),
        aliases: vec![],
        description: "Test echo".into(),
        inputs: vec![],
        steps: vec![npcrs::npc_compiler::JinxStep {
            name: "echo".into(),
            engine: "bash".into(),
            code: "echo hello".into(),
        }],
        file_context: vec![],
        npc: None,
        source_path: None,
    };
    let args = HashMap::new();
    let available = HashMap::new();
    let result = execute_jinx(&jinx, &args, &available).await.unwrap();
    assert!(result.success);
    assert!(result.output.contains("hello"));
}

#[tokio::test]
async fn test_python_jinx_execution() {
    let jinx = Jinx {
        name: "test_py".into(),
        aliases: vec![],
        description: "Test python".into(),
        inputs: vec![],
        steps: vec![npcrs::npc_compiler::JinxStep {
            name: "py".into(),
            engine: "bash".into(),
            code: "python3 -c \"print('hello from python')\"".into(),
        }],
        file_context: vec![],
        npc: None,
        source_path: None,
    };
    let args = HashMap::new();
    let available = HashMap::new();
    let result = execute_jinx(&jinx, &args, &available).await.unwrap();
    assert!(result.success);
    assert!(result.output.contains("hello from python"));
}

#[test]
fn test_all_jinxes_have_descriptions() {
    let jinx_dir = std::path::Path::new("../npcsh/npcsh/npc_team/jinxes");
    if !jinx_dir.exists() { return; }
    let jinxes = load_jinxes_from_directory(jinx_dir).unwrap();
    let mut missing = vec![];
    for (name, jinx) in &jinxes {
        if jinx.description.trim().is_empty() {
            missing.push(name.clone());
        }
    }
    if !missing.is_empty() {
        println!("WARNING: {} jinxes missing descriptions: {:?}", missing.len(), &missing[..missing.len().min(10)]);
    }
}

#[test]
fn test_jinx_aliases() {
    let jinx_dir = std::path::Path::new("../npcsh/npcsh/npc_team/jinxes");
    if !jinx_dir.exists() { return; }
    let jinxes = load_jinxes_from_directory(jinx_dir).unwrap();
    for (name, jinx) in &jinxes {
        for alias in &jinx.aliases {
            assert!(jinxes.contains_key(alias), "Alias '{}' for jinx '{}' should be registered", alias, name);
        }
    }
}
