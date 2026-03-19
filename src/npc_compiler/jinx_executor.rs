use crate::error::{NpcError, Result};
use crate::npc_compiler::{Jinx, JinxResult, JinxStep};
use std::collections::HashMap;
use tera::{Context, Tera};
use tokio::process::Command;

pub async fn execute_jinx(
    jinx: &Jinx,
    input_values: &HashMap<String, String>,
    available_jinxes: &HashMap<String, Jinx>,
) -> Result<JinxResult> {
    let mut context: HashMap<String, serde_json::Value> = HashMap::new();
    let mut output = String::new();

    for input in &jinx.inputs {
        let value = input_values
            .get(&input.name)
            .cloned()
            .or_else(|| input.default.clone())
            .unwrap_or_default();
        context.insert(
            input.name.clone(),
            serde_json::Value::String(value),
        );
    }

    let needs_tty = jinx_needs_tty(jinx);

    for step in &jinx.steps {
        let result = if needs_tty {
            execute_step_interactive(step, &context, available_jinxes).await
        } else {
            execute_step(step, &context, available_jinxes).await
        };

        match result {
            Ok(step_output) => {
                output = step_output.clone();
                context.insert(
                    step.name.clone(),
                    serde_json::Value::String(step_output.clone()),
                );
                context.insert(
                    "output".to_string(),
                    serde_json::Value::String(step_output),
                );
            }
            Err(e) => {
                return Ok(JinxResult {
                    output: format!("Error in step '{}': {}", step.name, e),
                    context,
                    success: false,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    Ok(JinxResult {
        output,
        context,
        success: true,
        error: None,
    })
}

fn jinx_needs_tty(jinx: &Jinx) -> bool {
    for step in &jinx.steps {
        let code = &step.code;
        if code.contains("termios")
            || code.contains("tty.setraw")
            || code.contains("curses")
            || code.contains("sys.stdin.isatty")
            || code.contains("select.select")
            || code.contains("getch")
        {
            return true;
        }
    }
    false
}

async fn execute_step(
    step: &JinxStep,
    context: &HashMap<String, serde_json::Value>,
    available_jinxes: &HashMap<String, Jinx>,
) -> Result<String> {
    match step.engine.as_str() {
        "python" => {
            let rendered = render_python_template(&step.code, context);
            execute_python(&rendered, context).await
        }
        "bash" => {
            let rendered = render_step_template(&step.code, context)?;
            execute_bash(&rendered).await
        }
        engine_name => execute_sub_jinx(engine_name, step, context, available_jinxes).await,
    }
}

async fn execute_step_interactive(
    step: &JinxStep,
    context: &HashMap<String, serde_json::Value>,
    available_jinxes: &HashMap<String, Jinx>,
) -> Result<String> {
    match step.engine.as_str() {
        "python" => {
            let rendered = render_python_template(&step.code, context);
            execute_python_interactive(&rendered, context).await
        }
        "bash" => {
            let rendered = render_step_template(&step.code, context)?;
            execute_bash_interactive(&rendered).await
        }
        engine_name => execute_sub_jinx(engine_name, step, context, available_jinxes).await,
    }
}

fn render_python_template(code: &str, context: &HashMap<String, serde_json::Value>) -> String {
    let re = regex::Regex::new(r"\{\{(.*?)\}\}").unwrap();

    re.replace_all(code, |caps: &regex::Captures| {
        let expr = caps[1].trim();
        resolve_template_expr(expr, context)
    })
    .to_string()
}

fn resolve_template_expr(expr: &str, context: &HashMap<String, serde_json::Value>) -> String {
    let parts: Vec<&str> = expr.split('|').map(|s| s.trim()).collect();
    if parts.is_empty() {
        return String::new();
    }

    let var_name = parts[0];

    let mut value = context.get(var_name).cloned();

    let mut use_tojson = false;
    for filter in &parts[1..] {
        if filter.starts_with("default(") {
            if value.is_none() || value.as_ref().is_some_and(|v| v.as_str() == Some("")) {
                let default_str = filter
                    .trim_start_matches("default(")
                    .trim_end_matches(')')
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'');
                value = Some(serde_json::Value::String(default_str.to_string()));
            }
        } else if *filter == "tojson" {
            use_tojson = true;
        }
    }

    match value {
        Some(v) => {
            if use_tojson {
                serde_json::to_string(&v).unwrap_or_else(|_| "null".to_string())
            } else {
                v.as_str().unwrap_or(&v.to_string()).to_string()
            }
        }
        None => {
            if use_tojson {
                "null".to_string()
            } else {
                String::new()
            }
        }
    }
}

async fn execute_sub_jinx(
    engine_name: &str,
    step: &JinxStep,
    context: &HashMap<String, serde_json::Value>,
    available_jinxes: &HashMap<String, Jinx>,
) -> Result<String> {
    if let Some(sub_jinx) = available_jinxes.get(engine_name) {
        let inputs: HashMap<String, String> = context
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str().unwrap_or(&v.to_string()).to_string(),
                )
            })
            .collect();
        let result = Box::pin(execute_jinx(sub_jinx, &inputs, available_jinxes)).await?;
        if result.success {
            Ok(result.output)
        } else {
            Err(NpcError::JinxExecution {
                step: step.name.clone(),
                reason: result.error.unwrap_or_default(),
            })
        }
    } else {
        Err(NpcError::JinxNotFound {
            name: engine_name.to_string(),
        })
    }
}

fn render_step_template(
    template: &str,
    context: &HashMap<String, serde_json::Value>,
) -> Result<String> {
    let mut tera = Tera::default();
    tera.add_raw_template("step", template)?;

    let mut ctx = Context::new();
    for (key, value) in context {
        ctx.insert(key, value);
    }

    Ok(tera.render("step", &ctx)?)
}

fn wrap_python_with_context(code: &str, context: &HashMap<String, serde_json::Value>) -> String {
    let context_json = serde_json::to_string(context).unwrap_or_else(|_| "{}".to_string());

    let indented_code = code
        .lines()
        .map(|l| format!("    {}", l))
        .collect::<Vec<_>>()
        .join("\n");

    let escaped_json = context_json
        .replace('\\', "\\\\")
        .replace('\'', "\\'");

    let mut wrapper = String::new();
    wrapper.push_str("import json, sys, os\n");
    wrapper.push_str(&format!("context = json.loads('{}')\n", escaped_json));
    wrapper.push_str("output = \"\"\n");
    wrapper.push_str("class _State:\n");
    wrapper.push_str("    current_path = os.getcwd()\n");
    wrapper.push_str("    chat_model = os.environ.get('NPCSH_CHAT_MODEL', 'gpt-4o-mini')\n");
    wrapper.push_str("    chat_provider = os.environ.get('NPCSH_CHAT_PROVIDER', 'openai')\n");
    wrapper.push_str("    stream_output = False\n");
    wrapper.push_str("state = _State()\n");
    wrapper.push_str("class _NPC:\n");
    wrapper.push_str("    name = \"assistant\"\n");
    wrapper.push_str("npc = _NPC()\n");
    wrapper.push_str("try:\n");
    wrapper.push_str(&indented_code);
    wrapper.push('\n');
    wrapper.push_str("except Exception as e:\n");
    wrapper.push_str("    context['output'] = f'Error: {e}'\n");
    wrapper.push_str("    output = str(e)\n");
    wrapper.push_str("result = context.get('output', output)\n");
    wrapper.push_str("if result:\n");
    wrapper.push_str("    print(result, end='')\n");

    wrapper
}

async fn execute_bash(code: &str) -> Result<String> {
    let output = Command::new("bash")
        .arg("-c")
        .arg(code)
        .output()
        .await
        .map_err(|e| NpcError::JinxExecution {
            step: "bash".to_string(),
            reason: e.to_string(),
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() {
        Ok(stdout.to_string())
    } else {
        Ok(format!(
            "{}{}[exit code: {}]",
            stdout,
            if stderr.is_empty() {
                String::new()
            } else {
                format!("\nSTDERR: {}", stderr)
            },
            output.status.code().unwrap_or(-1)
        ))
    }
}

async fn execute_bash_interactive(code: &str) -> Result<String> {
    let status = Command::new("bash")
        .arg("-c")
        .arg(code)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .await
        .map_err(|e| NpcError::JinxExecution {
            step: "bash".to_string(),
            reason: e.to_string(),
        })?;

    Ok(if status.success() {
        String::new()
    } else {
        format!("[exit code: {}]", status.code().unwrap_or(-1))
    })
}

async fn execute_python(code: &str, context: &HashMap<String, serde_json::Value>) -> Result<String> {
    let wrapped = wrap_python_with_context(code, context);

    let output = Command::new("python3")
        .arg("-c")
        .arg(&wrapped)
        .output()
        .await
        .map_err(|e| NpcError::JinxExecution {
            step: "python".to_string(),
            reason: format!("Failed to run Python: {}", e),
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() {
        Ok(stdout.to_string())
    } else {
        Ok(format!(
            "{}{}[python exit code: {}]",
            stdout,
            if stderr.is_empty() {
                String::new()
            } else {
                format!("\nSTDERR: {}", stderr)
            },
            output.status.code().unwrap_or(-1)
        ))
    }
}

async fn execute_python_interactive(
    code: &str,
    context: &HashMap<String, serde_json::Value>,
) -> Result<String> {
    let wrapped = wrap_python_with_context(code, context);

    let status = Command::new("python3")
        .arg("-c")
        .arg(&wrapped)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .await
        .map_err(|e| NpcError::JinxExecution {
            step: "python".to_string(),
            reason: format!("Failed to run Python: {}", e),
        })?;

    Ok(if status.success() {
        String::new()
    } else {
        format!("[python exit code: {}]", status.code().unwrap_or(-1))
    })
}
