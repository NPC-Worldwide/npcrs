
use crate::drivers::DriverManager;
use crate::error::Result;
use crate::ipc::IpcBus;
use crate::kernel::Kernel;
use crate::memory::CommandHistory;
use crate::process::Capabilities;
use crate::scheduler::Scheduler;
use crate::npc_compiler;
use crate::vfs::Vfs;
use std::collections::HashMap;
use std::sync::atomic::AtomicU32;

pub fn boot_kernel(team_dir: &str, db_path: &str) -> Result<Kernel> {
    tracing::info!("kernel: booting from {}", team_dir);

    let team = npc_compiler::load_team_from_directory(team_dir)?;
    let jinxes = team.jinxes.clone();

    tracing::info!(
        "kernel: loaded {} NPCs, {} jinxes",
        team.npcs.len(),
        jinxes.len()
    );

    let drivers = DriverManager::from_env();

    let db_path_expanded = shellexpand::tilde(db_path).to_string();
    let history = CommandHistory::open(&db_path_expanded)?;

    let vfs = Vfs::new(team_dir);

    let mut kernel = Kernel {
        processes: HashMap::new(),
        next_pid: AtomicU32::new(0),
        team,
        jinxes,
        drivers,
        vfs,
        ipc: IpcBus::new(),
        scheduler: Scheduler::new(),
        history,
        boot_time: chrono::Utc::now(),
    };

    let init_npc = kernel
        .team
        .lead_npc()
        .cloned()
        .unwrap_or_else(|| crate::npc_compiler::NPC::new("init", "You are the init process. Coordinate the system."));

    kernel.spawn_init(init_npc);
    tracing::info!("kernel: init process spawned (pid 0)");

    let other_npcs: Vec<_> = kernel
        .team
        .npcs
        .values()
        .filter(|n| Some(&n.name) != kernel.team.forenpc.as_ref())
        .cloned()
        .collect();

    for npc in other_npcs {
        let name = npc.name.clone();
        let pid = kernel.spawn(npc, 0, Capabilities::root());
        tracing::info!("kernel: spawned daemon {} (pid {})", name, pid);
    }

    tracing::info!("kernel: boot complete");
    Ok(kernel)
}
