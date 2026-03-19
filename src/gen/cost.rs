
use once_cell::sync::Lazy;
use std::collections::HashMap;

static TOKEN_COSTS: Lazy<HashMap<&'static str, (f64, f64)>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("gpt-4o", (2.50, 10.00));
    m.insert("gpt-4o-mini", (0.15, 0.60));
    m.insert("gpt-4-turbo", (10.00, 30.00));
    m.insert("gpt-3.5-turbo", (0.50, 1.50));
    m.insert("gpt-5", (1.25, 10.00));
    m.insert("gpt-5-mini", (0.25, 2.00));
    m.insert("o1", (15.00, 60.00));
    m.insert("o1-mini", (3.00, 12.00));
    m.insert("o3", (10.00, 40.00));
    m.insert("o3-mini", (1.10, 4.40));
    m.insert("o4-mini", (1.10, 4.40));
    m.insert("claude-3-5-sonnet", (3.00, 15.00));
    m.insert("claude-3-opus", (15.00, 75.00));
    m.insert("claude-3-haiku", (0.25, 1.25));
    m.insert("claude-sonnet-4", (3.00, 15.00));
    m.insert("claude-opus-4", (15.00, 75.00));
    m.insert("claude-opus-4-5", (5.00, 25.00));
    m.insert("claude-sonnet-4-5", (3.00, 15.00));
    m.insert("claude-haiku-4", (0.80, 4.00));
    m.insert("gemini-1.5-pro", (1.25, 5.00));
    m.insert("gemini-1.5-flash", (0.075, 0.30));
    m.insert("gemini-2.0-flash", (0.10, 0.40));
    m.insert("gemini-2.5-pro", (1.25, 10.00));
    m.insert("gemini-2.5-flash", (0.15, 0.60));
    m.insert("gemini-3.1-pro", (2.00, 12.00));
    m.insert("llama-3", (0.05, 0.08));
    m.insert("llama-3.1", (0.05, 0.08));
    m.insert("llama-3.2", (0.05, 0.08));
    m.insert("llama-4", (0.05, 0.10));
    m.insert("mixtral", (0.24, 0.24));
    m.insert("deepseek-v3", (0.27, 1.10));
    m.insert("deepseek-r1", (0.55, 2.19));
    m.insert("mistral-large", (2.00, 6.00));
    m.insert("mistral-small", (0.20, 0.60));
    m.insert("grok-2", (2.00, 10.00));
    m.insert("grok-3", (3.00, 15.00));
    m
});

pub fn calculate_cost(model: &str, input_tokens: u64, output_tokens: u64) -> f64 {
    let (input_per_m, output_per_m) = lookup_cost(model);
    let input_cost = (input_tokens as f64 / 1_000_000.0) * input_per_m;
    let output_cost = (output_tokens as f64 / 1_000_000.0) * output_per_m;
    input_cost + output_cost
}

pub fn lookup_cost(model: &str) -> (f64, f64) {
    let model_lower = model.to_lowercase();

    if let Some(&costs) = TOKEN_COSTS.get(model_lower.as_str()) {
        return costs;
    }

    let mut candidate = model_lower.as_str();
    loop {
        if let Some(&costs) = TOKEN_COSTS.get(candidate) {
            return costs;
        }
        match candidate.rfind('-') {
            Some(pos) => candidate = &candidate[..pos],
            None => break,
        }
    }

    let mut best_match: Option<(&str, (f64, f64))> = None;
    for (&key, &costs) in TOKEN_COSTS.iter() {
        if model_lower.starts_with(key) {
            match best_match {
                Some((prev_key, _)) if key.len() > prev_key.len() => {
                    best_match = Some((key, costs));
                }
                None => {
                    best_match = Some((key, costs));
                }
                _ => {}
            }
        }
    }
    if let Some((_, costs)) = best_match {
        return costs;
    }

    let mut best_contains: Option<(&str, (f64, f64))> = None;
    for (&key, &costs) in TOKEN_COSTS.iter() {
        if model_lower.contains(key) {
            match best_contains {
                Some((prev_key, _)) if key.len() > prev_key.len() => {
                    best_contains = Some((key, costs));
                }
                None => {
                    best_contains = Some((key, costs));
                }
                _ => {}
            }
        }
    }
    if let Some((_, costs)) = best_contains {
        return costs;
    }

    (0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let cost = calculate_cost("gpt-4o", 1_000_000, 1_000_000);
        assert!((cost - 12.50).abs() < 0.01); // 2.50 + 10.00
    }

    #[test]
    fn fuzzy_match_with_date_suffix() {
        let (inp, out) = lookup_cost("gpt-4o-2024-08-06");
        assert!((inp - 2.50).abs() < 0.01);
        assert!((out - 10.00).abs() < 0.01);
    }

    #[test]
    fn fuzzy_match_claude_versioned() {
        let (inp, out) = lookup_cost("claude-3-5-sonnet-20241022");
        assert!((inp - 3.00).abs() < 0.01);
        assert!((out - 15.00).abs() < 0.01);
    }

    #[test]
    fn unknown_model_is_free() {
        let cost = calculate_cost("my-custom-local-model", 1_000_000, 1_000_000);
        assert!((cost - 0.0).abs() < 0.001);
    }

    #[test]
    fn ollama_llama_model() {
        let cost = calculate_cost("llama3.2", 1000, 1000);
        assert!(cost < 0.001);
    }

    #[test]
    fn prefix_match() {
        let (inp, out) = lookup_cost("gpt-4o-mini-some-variant");
        assert!((inp - 0.15).abs() < 0.01);
        assert!((out - 0.60).abs() < 0.01);
    }
}
