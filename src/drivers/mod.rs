//! Drivers — external tool providers managed by the kernel.

pub struct DriverManager;

impl DriverManager {
    pub fn from_env() -> Self { Self }
}
