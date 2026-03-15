//! Supervised Fine-Tuning (SFT) via Python subprocess.
//!
//! Mirrors `npcpy.ft.sft`. Shells out to Python since SFT requires
//! PyTorch, transformers, and trl.

use crate::error::{NpcError, Result};
use tokio::process::Command;

/// Configuration for an SFT training run.
#[derive(Debug, Clone)]
pub struct SftConfig {
    /// Base model name or path (e.g. "meta-llama/Llama-3-8B").
    pub model_name: String,
    /// Path to the training dataset (JSON, JSONL, or CSV).
    pub dataset_path: String,
    /// Directory to save the fine-tuned model.
    pub output_dir: String,
    /// Number of training epochs.
    pub epochs: u32,
    /// Batch size per device.
    pub batch_size: u32,
    /// Learning rate.
    pub learning_rate: f64,
    /// Maximum sequence length.
    pub max_seq_length: u32,
    /// Whether to use LoRA (recommended for large models).
    pub use_lora: bool,
    /// LoRA rank (only used if use_lora is true).
    pub lora_rank: u32,
}

impl Default for SftConfig {
    fn default() -> Self {
        Self {
            model_name: "meta-llama/Llama-3-8B".to_string(),
            dataset_path: String::new(),
            output_dir: "./sft_output".to_string(),
            epochs: 3,
            batch_size: 4,
            learning_rate: 2e-5,
            max_seq_length: 2048,
            use_lora: true,
            lora_rank: 16,
        }
    }
}

/// Run SFT training via Python subprocess.
///
/// Attempts to use npcpy's SFT implementation first, falls back to
/// a direct transformers/trl script.
///
/// # Arguments
/// * `config` — Full SFT configuration.
///
/// # Returns
/// JSON string with training results (output_dir, loss, etc.).
pub async fn train_sft(config: &SftConfig) -> Result<String> {
    let script = format!(
        r#"
import json, sys

config = {{
    "model_name": "{model_name}",
    "dataset_path": "{dataset_path}",
    "output_dir": "{output_dir}",
    "epochs": {epochs},
    "batch_size": {batch_size},
    "learning_rate": {learning_rate},
    "max_seq_length": {max_seq_length},
    "use_lora": {use_lora},
    "lora_rank": {lora_rank},
}}

try:
    from npcpy.ft.sft import train_sft
    result = train_sft(**config)
    print(json.dumps(result))
except ImportError:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=config["dataset_path"])["train"]

        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(config["model_name"])

        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            save_strategy="epoch",
            logging_steps=10,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=config["max_seq_length"],
        )

        result = trainer.train()
        trainer.save_model(config["output_dir"])

        print(json.dumps({{
            "output_dir": config["output_dir"],
            "train_loss": result.training_loss,
            "epochs": config["epochs"],
            "status": "complete",
        }}))
    except ImportError as e:
        print(json.dumps({{"error": f"Missing dependencies: {{e}}. Install with: pip install transformers trl datasets torch"}}))
        sys.exit(1)
"#,
        model_name = config.model_name,
        dataset_path = config.dataset_path,
        output_dir = config.output_dir,
        epochs = config.epochs,
        batch_size = config.batch_size,
        learning_rate = config.learning_rate,
        max_seq_length = config.max_seq_length,
        use_lora = if config.use_lora { "True" } else { "False" },
        lora_rank = config.lora_rank,
    );

    let output = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| NpcError::Other(format!("Failed to spawn python3 for SFT: {e}")))?
        .wait_with_output()
        .await
        .map_err(|e| NpcError::Other(format!("Failed to wait for SFT process: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(NpcError::Other(format!(
            "SFT training failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Convenience wrapper: train SFT with just the essential parameters.
pub async fn train_sft_simple(
    model_name: &str,
    dataset_path: &str,
    output_dir: &str,
    epochs: u32,
) -> Result<String> {
    let config = SftConfig {
        model_name: model_name.to_string(),
        dataset_path: dataset_path.to_string(),
        output_dir: output_dir.to_string(),
        epochs,
        ..Default::default()
    };
    train_sft(&config).await
}
