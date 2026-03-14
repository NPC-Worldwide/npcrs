//! Virtual Filesystem — unified view of real FS, memory, and knowledge.
//!
//! The VFS presents a single namespace:
//!
//! ```text
//! /                          (root)
//! ├── fs/                    (real filesystem, passthrough)
//! ├── proc/                  (process info, like /proc in Linux)
//! │   ├── 0/                 (pid 0 = init)
//! │   │   ├── status         (process state, usage)
//! │   │   ├── messages       (conversation history)
//! │   │   └── env            (environment vars)
//! │   └── 1/
//! ├── dev/                   (devices = LLM providers)
//! │   ├── openai             (OpenAI driver)
//! │   ├── anthropic          (Anthropic driver)
//! │   └── ollama             (Ollama driver)
//! ├── sys/                   (kernel info)
//! │   ├── stats              (kernel stats)
//! │   ├── jinxes/            (available syscalls)
//! │   └── team/              (team context)
//! ├── mem/                   (knowledge graph + memories)
//! │   ├── kg/                (knowledge graph entities)
//! │   └── conversations/     (conversation history)
//! └── tmp/                   (scratch space)
//! ```
//!
//! This lets NPCs navigate the system with familiar path semantics.

use crate::error::Result;
use std::path::{Path, PathBuf};

/// Virtual Filesystem.
pub struct Vfs {
    /// Root of the real filesystem mount (usually the team directory).
    fs_root: PathBuf,

    /// Temp directory for scratch space.
    tmp_dir: PathBuf,
}

impl Vfs {
    pub fn new(fs_root: impl Into<PathBuf>) -> Self {
        let fs_root = fs_root.into();
        let tmp_dir = std::env::temp_dir().join("npcrs");
        let _ = std::fs::create_dir_all(&tmp_dir);

        Self { fs_root, tmp_dir }
    }

    /// Resolve a VFS path to a real action.
    pub fn resolve(&self, vfs_path: &str) -> VfsResolution {
        let path = vfs_path.trim_start_matches('/');

        if path.is_empty() || path == "/" {
            return VfsResolution::Directory(vec![
                "fs".into(),
                "proc".into(),
                "dev".into(),
                "sys".into(),
                "mem".into(),
                "tmp".into(),
            ]);
        }

        let parts: Vec<&str> = path.splitn(2, '/').collect();
        let mount = parts[0];
        let rest = parts.get(1).unwrap_or(&"");

        match mount {
            "fs" => {
                let real_path = if rest.is_empty() {
                    self.fs_root.clone()
                } else {
                    self.fs_root.join(rest)
                };
                VfsResolution::RealPath(real_path)
            }
            "tmp" => {
                let real_path = if rest.is_empty() {
                    self.tmp_dir.clone()
                } else {
                    self.tmp_dir.join(rest)
                };
                VfsResolution::RealPath(real_path)
            }
            "proc" => VfsResolution::Virtual(VirtualNode::Proc(rest.to_string())),
            "dev" => VfsResolution::Virtual(VirtualNode::Dev(rest.to_string())),
            "sys" => VfsResolution::Virtual(VirtualNode::Sys(rest.to_string())),
            "mem" => VfsResolution::Virtual(VirtualNode::Mem(rest.to_string())),
            _ => VfsResolution::NotFound,
        }
    }

    /// Read a file from the VFS.
    pub fn read_file(&self, vfs_path: &str) -> Result<String> {
        match self.resolve(vfs_path) {
            VfsResolution::RealPath(path) => {
                std::fs::read_to_string(&path).map_err(|e| crate::error::NpcError::FileLoad {
                    path: path.display().to_string(),
                    source: e,
                })
            }
            VfsResolution::Virtual(node) => Ok(format!("[virtual: {:?}]", node)),
            VfsResolution::Directory(entries) => Ok(entries.join("\n")),
            VfsResolution::NotFound => {
                Err(crate::error::NpcError::Other(format!(
                    "ENOENT: {} not found",
                    vfs_path
                )))
            }
        }
    }

    /// Write a file to the VFS.
    pub fn write_file(&self, vfs_path: &str, content: &str) -> Result<()> {
        match self.resolve(vfs_path) {
            VfsResolution::RealPath(path) => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        crate::error::NpcError::FileLoad {
                            path: parent.display().to_string(),
                            source: e,
                        }
                    })?;
                }
                std::fs::write(&path, content).map_err(|e| {
                    crate::error::NpcError::FileLoad {
                        path: path.display().to_string(),
                        source: e,
                    }
                })
            }
            VfsResolution::Virtual(_) => Err(crate::error::NpcError::Other(
                "EROFS: cannot write to virtual node".to_string(),
            )),
            _ => Err(crate::error::NpcError::Other(format!(
                "ENOENT: {} not found",
                vfs_path
            ))),
        }
    }

    /// List directory contents.
    pub fn list_dir(&self, vfs_path: &str) -> Result<Vec<String>> {
        match self.resolve(vfs_path) {
            VfsResolution::RealPath(path) => {
                let entries = std::fs::read_dir(&path)
                    .map_err(|e| crate::error::NpcError::FileLoad {
                        path: path.display().to_string(),
                        source: e,
                    })?
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect();
                Ok(entries)
            }
            VfsResolution::Directory(entries) => Ok(entries),
            _ => Ok(Vec::new()),
        }
    }

    /// Get the real FS root.
    pub fn fs_root(&self) -> &Path {
        &self.fs_root
    }
}

/// Result of resolving a VFS path.
#[derive(Debug)]
pub enum VfsResolution {
    /// Maps to a real filesystem path.
    RealPath(PathBuf),
    /// A virtual (in-kernel) node.
    Virtual(VirtualNode),
    /// A directory listing.
    Directory(Vec<String>),
    /// Path doesn't exist.
    NotFound,
}

/// Virtual filesystem nodes (generated on-the-fly from kernel state).
#[derive(Debug)]
pub enum VirtualNode {
    /// /proc/<pid>/... — process info.
    Proc(String),
    /// /dev/<driver> — device/driver info.
    Dev(String),
    /// /sys/... — kernel info.
    Sys(String),
    /// /mem/... — knowledge graph / memory.
    Mem(String),
}
