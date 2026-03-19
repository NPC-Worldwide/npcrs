
use crate::error::Result;
use std::path::{Path, PathBuf};

pub struct Vfs {
    fs_root: PathBuf,

    tmp_dir: PathBuf,
}

impl Vfs {
    pub fn new(fs_root: impl Into<PathBuf>) -> Self {
        let fs_root = fs_root.into();
        let tmp_dir = std::env::temp_dir().join("npcrs");
        let _ = std::fs::create_dir_all(&tmp_dir);

        Self { fs_root, tmp_dir }
    }

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

    pub fn fs_root(&self) -> &Path {
        &self.fs_root
    }
}

#[derive(Debug)]
pub enum VfsResolution {
    RealPath(PathBuf),
    Virtual(VirtualNode),
    Directory(Vec<String>),
    NotFound,
}

#[derive(Debug)]
pub enum VirtualNode {
    Proc(String),
    Dev(String),
    Sys(String),
    Mem(String),
}
