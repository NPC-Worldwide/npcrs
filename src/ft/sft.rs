
use crate::error::{NpcError, Result};

pub struct SftConfig {
    pub model: String,
    pub dataset: String,
    pub output_dir: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub lora_r: u32,
    pub lora_alpha: u32,
}

impl Default for SftConfig {
    fn default() -> Self {
        Self {
            model: "qwen3.5:2b".into(),
            dataset: String::new(),
            output_dir: "./sft_output".into(),
            epochs: 3,
            batch_size: 4,
            learning_rate: 2e-5,
            lora_r: 16,
            lora_alpha: 32,
        }
    }
}

pub async fn train_sft(config: &SftConfig) -> Result<String> {
    let _ = config;
    Err(NpcError::Other(
        "SFT training requires Python transformers/trl runtime. Use npcpy.ft.sft.".into()
    ))
}

pub async fn train_sft_simple(model: &str, dataset_path: &str, output_dir: &str) -> Result<String> {
    let config = SftConfig {
        model: model.into(),
        dataset: dataset_path.into(),
        output_dir: output_dir.into(),
        ..Default::default()
    };
    train_sft(&config).await
}
