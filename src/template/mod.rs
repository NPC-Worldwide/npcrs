//! Tera-based template engine for Jinx rendering.
//!
//! Tera is a Jinja2-compatible template engine for Rust.
//! This module wraps Tera with NPC-specific functions and filters.

use tera::{Context, Tera};
use std::collections::HashMap;

/// Create a Tera instance with NPC-specific functions registered.
pub fn create_engine() -> Tera {
    let mut tera = Tera::default();

    // Register custom filters
    tera.register_filter("tojson", tojson_filter);

    tera
}

/// Render a template string with the given context variables.
pub fn render(
    template: &str,
    context: &HashMap<String, serde_json::Value>,
) -> Result<String, tera::Error> {
    let mut tera = create_engine();
    tera.add_raw_template("__inline__", template)?;

    let mut ctx = Context::new();
    for (key, value) in context {
        ctx.insert(key, value);
    }

    tera.render("__inline__", &ctx)
}

/// Custom `tojson` filter — serializes a value to JSON string.
fn tojson_filter(
    value: &tera::Value,
    _args: &HashMap<String, tera::Value>,
) -> Result<tera::Value, tera::Error> {
    Ok(tera::Value::String(
        serde_json::to_string(value).unwrap_or_else(|_| value.to_string()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_simple() {
        let mut ctx = HashMap::new();
        ctx.insert(
            "name".to_string(),
            serde_json::Value::String("world".to_string()),
        );

        let result = render("Hello {{ name }}!", &ctx).unwrap();
        assert_eq!(result, "Hello world!");
    }

    #[test]
    fn test_render_tojson_filter() {
        let mut ctx = HashMap::new();
        ctx.insert(
            "data".to_string(),
            serde_json::json!({"key": "value"}),
        );

        let result = render("{{ data | tojson }}", &ctx).unwrap();
        assert!(result.contains("key"));
        assert!(result.contains("value"));
    }
}
