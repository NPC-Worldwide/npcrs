pub fn calculate_cost(_model: &str, _input_tokens: u64, _output_tokens: u64) -> f64 {
    0.0
}

pub fn lookup_cost(_model: &str) -> (f64, f64) {
    (0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_is_always_zero() {
        assert!((calculate_cost("gpt-4o", 1_000_000, 1_000_000)).abs() < 0.001);
        assert_eq!(lookup_cost("anything"), (0.0, 0.0));
    }
}
