use crate::error::{NpcError, Result};
use sqlx::AnyPool;

pub struct DbPool {
    pool: AnyPool,
}

impl DbPool {
    pub async fn connect(url: &str) -> Result<Self> {
        let pool = AnyPool::connect(url).await
            .map_err(|e| NpcError::Other(format!("DB connect: {}", e)))?;
        Ok(Self { pool })
    }

    pub async fn connect_sqlite(path: &str) -> Result<Self> {
        let url = if path == ":memory:" {
            "sqlite::memory:".to_string()
        } else {
            format!("sqlite://{}?mode=rwc", path)
        };
        Self::connect(&url).await
    }

    pub fn pool(&self) -> &AnyPool {
        &self.pool
    }

    pub async fn execute(&self, sql: &str) -> Result<u64> {
        let result = sqlx::query(sql).execute(&self.pool).await
            .map_err(|e| NpcError::Other(format!("DB execute: {}", e)))?;
        Ok(result.rows_affected())
    }

    pub async fn execute_batch(&self, sql: &str) -> Result<()> {
        for statement in sql.split(';') {
            let trimmed = statement.trim();
            if trimmed.is_empty() { continue; }
            sqlx::query(trimmed).execute(&self.pool).await
                .map_err(|e| NpcError::Other(format!("DB batch: {}", e)))?;
        }
        Ok(())
    }
}
