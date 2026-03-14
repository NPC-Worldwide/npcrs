//! Boot sequence — loads team, initializes kernel, spawns init process.

use crate::drivers::DriverManager;
use crate::error::Result;
use crate::ipc::IpcBus;
use crate::kernel::Kernel;
use crate::memory::CommandHistory;
use crate::process::Capabilities;
use crate::scheduler::Scheduler;
use crate::team;
use crate::vfs::Vfs;
use std::collections::HashMap;
use std::sync::atomic::AtomicU32;

/// Boot the kernel from a team directory.
///
/// Boot sequence:
/// 1. Load team from directory (boot image)
/// 2. Initialize drivers (LLM providers from env)
/// 3. Open history database
/// 4. Mount virtual filesystem
/// 5. Spawn init process (forenpc)
/// 6. Spawn daemon processes for other NPCs
pub fn boot_kernel(team_dir: &str, db_path: &str) -> Result<Kernel> {
    tracing::info!("kernel: booting from {}", team_dir);

    // 1. Load the team (our "disk image")
    let team = team::load_team_from_directory(team_dir)?;
    let jinxes = team.jinxes.clone();

    tracing::info!(
        "kernel: loaded {} NPCs, {} jinxes",
        team.npcs.len(),
        jinxes.len()
    );

    // 2. Initialize drivers
    let drivers = DriverManager::from_env();

    // 3. Open history database
    let db_path_expanded = shellexpand::tilde(db_path).to_string();
    let history = CommandHistory::open(&db_path_expanded)?;

    // 4. Create VFS
    let vfs = Vfs::new(team_dir);

    // 5. Create kernel
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

    // 6. Spawn init process
    let init_npc = kernel
        .team
        .lead_npc()
        .cloned()
        .unwrap_or_else(|| crate::npc::Npc::new("init", "You are the init process. Coordinate the system."));

    kernel.spawn_init(init_npc);
    tracing::info!("kernel: init process spawned (pid 0)");

    // 7. Optionally spawn other NPCs as daemons
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
