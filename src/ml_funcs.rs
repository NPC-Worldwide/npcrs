//! ML utility functions — mirrors npcpy.ml_funcs.
//!
//! For model training/prediction, these require either:
//! - A running Ollama server with the model loaded
//! - API access to a cloud ML provider
//!
//! Heavy ML operations (sklearn, torch) are not available in pure Rust
//! with equivalent quality. These functions provide the API surface
//! and delegate to LLM-based alternatives where possible.

use crate::error::{NpcError, Result};
use std::collections::HashMap;

/// Fit a model — delegates to LLM for now since sklearn isn't available in Rust.
/// Returns a JSON string describing what would be trained.
pub async fn fit_model(
    data_json: &str,
    model_type: &str,
    target: &str,
    output_path: &str,
) -> Result<String> {
    // Parse the data to validate it
    let _data: serde_json::Value = serde_json::from_str(data_json)
        .map_err(|e| NpcError::Other(format!("Invalid data JSON: {}", e)))?;

    Ok(serde_json::json!({
        "status": "ml_funcs.fit_model requires Python sklearn runtime",
        "model_type": model_type,
        "target": target,
        "output_path": output_path,
        "hint": "Use npcpy for full ML model training"
    }).to_string())
}

/// Predict using a model.
pub async fn predict_model(model_path: &str, data_json: &str) -> Result<String> {
    let _data: serde_json::Value = serde_json::from_str(data_json)
        .map_err(|e| NpcError::Other(format!("Invalid data JSON: {}", e)))?;

    Ok(serde_json::json!({
        "status": "ml_funcs.predict_model requires Python sklearn runtime",
        "model_path": model_path,
        "hint": "Use npcpy for ML prediction"
    }).to_string())
}

/// Score a model.
pub async fn score_model(model_path: &str, data_json: &str, target: &str) -> Result<f64> {
    let _data: serde_json::Value = serde_json::from_str(data_json)
        .map_err(|e| NpcError::Other(format!("Invalid data JSON: {}", e)))?;
    let _ = (model_path, target);
    Err(NpcError::Other("ml_funcs.score_model requires Python sklearn runtime. Use npcpy.".into()))
}

/// List available model types.
pub fn list_models() -> Vec<String> {
    vec![
        "RandomForestClassifier".into(),
        "RandomForestRegressor".into(),
        "LogisticRegression".into(),
        "LinearRegression".into(),
        "GradientBoostingClassifier".into(),
        "GradientBoostingRegressor".into(),
    ]
}

/// Ensemble predict — run multiple models and combine.
pub async fn ensemble_predict(
    _data_json: &str,
    _model_paths: &[&str],
) -> Result<String> {
    Err(NpcError::Other("ensemble_predict requires Python sklearn runtime. Use npcpy.".into()))
}

/// Cross validate a model.
pub async fn cross_validate(
    _data_json: &str,
    _model_type: &str,
    _target: &str,
    _folds: u32,
) -> Result<String> {
    Err(NpcError::Other("cross_validate requires Python sklearn runtime. Use npcpy.".into()))
}
