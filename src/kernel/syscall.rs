
use crate::error::{NpcError, Result};
use crate::npc_compiler;
use crate::kernel::Kernel;
use crate::process::Pid;
use std::collections::HashMap;

pub async fn execute_syscall(
    kernel: &mut Kernel,
    pid: Pid,
    jinx_name: &str,
    args: &HashMap<String, String>,
) -> Result<String> {
    let process = kernel.processes.get(&pid).ok_or_else(|| {
        NpcError::Other(format!("ESRCH: no process with pid {}", pid))
    })?;

    if process.state == crate::process::ProcessState::Dead {
        return Err(NpcError::Other(format!(
            "ESRCH: process {} is dead",
            pid
        )));
    }

    if !process.capabilities.can_run_jinx(jinx_name) {
        return Err(NpcError::Other(format!(
            "EPERM: process {} (npc:{}) cannot invoke jinx '{}'",
            pid, process.npc.name, jinx_name
        )));
    }

    if jinx_name == "sh" && !process.capabilities.can_bash {
        return Err(NpcError::Other(format!(
            "EPERM: process {} lacks CAP_BASH for jinx 'sh'",
            pid
        )));
    }

    if let Some(reason) = process.usage.exceeds(&process.limits) {
        return Err(NpcError::Other(format!(
            "ENOMEM: process {} — {}",
            pid, reason
        )));
    }

    let jinx = kernel.jinxes.get(jinx_name).ok_or_else(|| {
        NpcError::JinxNotFound {
            name: jinx_name.to_string(),
        }
    })?;

    tracing::debug!(
        "syscall: pid:{} invoking jinx '{}' with {} args",
        pid,
        jinx_name,
        args.len()
    );

    let has_python_steps = jinx.steps.iter().any(|s| s.engine == "python");

    let result = if has_python_steps {
        if let Some(ref mut daemon) = kernel.python_daemon {
            let cmd = if args.is_empty() {
                format!("/{}", jinx_name)
            } else {
                let args_str: Vec<String> = args.iter().map(|(k, v)| {
                    if v.is_empty() { k.clone() } else { format!("{}={}", k, v) }
                }).collect();
                format!("/{} {}", jinx_name, args_str.join(" "))
            };
            let output = daemon.execute(&cmd, None).await?;
            crate::npc_compiler::JinxResult {
                output,
                context: HashMap::new(),
                success: true,
                error: None,
            }
        } else {
            let active_npc = kernel.processes.get(&pid).map(|p| &p.npc);
            npc_compiler::execute_jinx_with_npc(jinx, args, &kernel.jinxes, active_npc).await?
        }
    } else {
        let active_npc = kernel.processes.get(&pid).map(|p| &p.npc);
        npc_compiler::execute_jinx_with_npc(jinx, args, &kernel.jinxes, active_npc).await?
    };

    let conv_id = kernel.processes.get(&pid)
        .map(|p| p.conversation_id.clone())
        .unwrap_or_default();
    let npc_name = kernel.processes.get(&pid).map(|p| p.npc.name.as_str());
    let _ = kernel.history.save_jinx_execution(
        &conv_id,
        jinx_name,
        &serde_json::to_string(args).unwrap_or_default(),
        &result.output,
        if result.success { "success" } else { "error" },
        npc_name,
        None,
        result.error.as_deref(),
        None,
    );

    if let Some(process) = kernel.processes.get_mut(&pid) {
        process.usage.total_tool_calls += 1;
        process.usage.tool_calls_this_turn += 1;
    }

    if result.success {
        Ok(result.output)
    } else {
        Err(NpcError::JinxExecution {
            step: jinx_name.to_string(),
            reason: result.error.unwrap_or_default(),
        })
    }
}
