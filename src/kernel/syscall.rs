//! Syscall dispatch — jinx invocation with capability checking.
//!
//! Every jinx invocation goes through the syscall layer, which:
//! 1. Checks the calling process has the right capability
//! 2. Checks resource limits
//! 3. Executes the jinx
//! 4. Records the execution in history
//! 5. Returns the result

use crate::error::{NpcError, Result};
use crate::jinx;
use crate::kernel::Kernel;
use crate::process::Pid;
use std::collections::HashMap;

/// Execute a syscall (jinx invocation) on behalf of a process.
pub async fn execute_syscall(
    kernel: &mut Kernel,
    pid: Pid,
    jinx_name: &str,
    args: &HashMap<String, String>,
) -> Result<String> {
    // 1. Get process and check state
    let process = kernel.processes.get(&pid).ok_or_else(|| {
        NpcError::Other(format!("ESRCH: no process with pid {}", pid))
    })?;

    if process.state == crate::process::ProcessState::Dead {
        return Err(NpcError::Other(format!(
            "ESRCH: process {} is dead",
            pid
        )));
    }

    // 2. Capability check
    if !process.capabilities.can_run_jinx(jinx_name) {
        return Err(NpcError::Other(format!(
            "EPERM: process {} (npc:{}) cannot invoke jinx '{}'",
            pid, process.npc.name, jinx_name
        )));
    }

    // Special capability checks for dangerous jinxes
    if jinx_name == "sh" && !process.capabilities.can_bash {
        return Err(NpcError::Other(format!(
            "EPERM: process {} lacks CAP_BASH for jinx 'sh'",
            pid
        )));
    }

    // 3. Resource limit check
    if let Some(reason) = process.usage.exceeds(&process.limits) {
        return Err(NpcError::Other(format!(
            "ENOMEM: process {} — {}",
            pid, reason
        )));
    }

    // 4. Find the jinx
    let jinx = kernel.jinxes.get(jinx_name).ok_or_else(|| {
        NpcError::JinxNotFound {
            name: jinx_name.to_string(),
        }
    })?;

    // 5. Execute
    tracing::debug!(
        "syscall: pid:{} invoking jinx '{}' with {} args",
        pid,
        jinx_name,
        args.len()
    );

    let result = jinx::execute_jinx(jinx, args, &kernel.jinxes).await?;

    // 6. Record execution
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

    // 7. Update process usage
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
