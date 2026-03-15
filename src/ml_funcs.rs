//! ML utility functions — thin wrappers over Python sklearn via subprocess.
//!
//! Mirrors `npcpy.ml_funcs`. These functions shell out to `python3` and
//! call into npcpy or sklearn directly.

use crate::error::{NpcError, Result};
use tokio::process::Command;

/// Fit a model using Python sklearn via subprocess.
///
/// Passes `data_json` via stdin to avoid shell-escaping issues.
/// Returns JSON string with fit results (score, model_path, model_type).
pub async fn fit_model(
    data_json: &str,
    model_type: &str,
    target: &str,
    output_path: &str,
) -> Result<String> {
    let script = format!(
        r#"
import json, sys
try:
    from npcpy.ml_funcs import fit_model
    data = json.loads(sys.stdin.read())
    result = fit_model(data, '{model_type}', '{target}', '{output_path}')
    print(json.dumps(result))
except ImportError:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import pickle
    data = json.loads(sys.stdin.read())
    df = pd.DataFrame(data)
    X = df.drop(columns=['{target}'])
    y = df['{target}']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    models = {{
        'RandomForestClassifier': RandomForestClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'LogisticRegression': LogisticRegression,
        'LinearRegression': LinearRegression,
    }}
    cls = models.get('{model_type}', RandomForestClassifier)
    model = cls()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    with open('{output_path}', 'wb') as f:
        pickle.dump(model, f)
    print(json.dumps({{"score": score, "model_path": "{output_path}", "model_type": "{model_type}"}}))
"#,
        model_type = model_type,
        target = target,
        output_path = output_path,
    );

    run_python_script_with_stdin(&script, data_json).await
}

/// Predict using a fitted model (pickle file).
/// Returns JSON string with predictions array.
pub async fn predict_model(model_path: &str, data_json: &str) -> Result<String> {
    let script = format!(
        r#"
import json, sys, pickle
import pandas as pd
with open('{model_path}', 'rb') as f:
    model = pickle.load(f)
data = json.loads(sys.stdin.read())
df = pd.DataFrame(data)
preds = model.predict(df).tolist()
print(json.dumps({{"predictions": preds}}))
"#,
        model_path = model_path,
    );

    run_python_script_with_stdin(&script, data_json).await
}

/// Score a model against labelled data.
/// Returns the model score (accuracy for classifiers, R^2 for regressors).
pub async fn score_model(model_path: &str, data_json: &str, target: &str) -> Result<f64> {
    let script = format!(
        r#"
import json, sys, pickle
import pandas as pd
with open('{model_path}', 'rb') as f:
    model = pickle.load(f)
data = json.loads(sys.stdin.read())
df = pd.DataFrame(data)
X = df.drop(columns=['{target}'])
y = df['{target}']
score = model.score(X, y)
print(json.dumps({{"score": score}}))
"#,
        model_path = model_path,
        target = target,
    );

    let output = run_python_script_with_stdin(&script, data_json).await?;
    let parsed: serde_json::Value =
        serde_json::from_str(&output).map_err(|e| NpcError::Other(format!("JSON parse: {e}")))?;
    parsed["score"]
        .as_f64()
        .ok_or_else(|| NpcError::Other("score not a number in Python output".into()))
}

/// List available sklearn model types (static).
pub fn list_models() -> Vec<String> {
    vec![
        "RandomForestClassifier".into(),
        "RandomForestRegressor".into(),
        "LogisticRegression".into(),
        "LinearRegression".into(),
        "GradientBoostingClassifier".into(),
        "GradientBoostingRegressor".into(),
        "SVR".into(),
        "SVC".into(),
        "KNeighborsClassifier".into(),
        "KNeighborsRegressor".into(),
        "DecisionTreeClassifier".into(),
        "DecisionTreeRegressor".into(),
    ]
}

/// Run a Python script, piping `stdin_data` via stdin, and return stdout.
async fn run_python_script_with_stdin(script: &str, stdin_data: &str) -> Result<String> {
    use tokio::io::AsyncWriteExt;

    let mut child = Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| NpcError::Other(format!("Failed to spawn python3: {e}")))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(stdin_data.as_bytes())
            .await
            .map_err(|e| NpcError::Other(format!("Failed to write to python3 stdin: {e}")))?;
    }

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| NpcError::Other(format!("Failed to wait for python3: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(NpcError::Other(format!(
            "Python script failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
