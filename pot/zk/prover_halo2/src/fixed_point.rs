/// Fixed-point arithmetic utilities for handling decimal numbers in field elements
/// This module provides functions to convert between floating point and fixed-point representation

/// Fixed-point arithmetic with scale 2^16
const SCALE: u64 = 65536; // 2^16

/// Utility functions for fixed-point arithmetic outside circuits
pub mod primitives {
    use super::*;

    /// Convert float to fixed-point representation
    pub fn float_to_fixed_u64(value: f64) -> u64 {
        (value * (SCALE as f64)).round() as u64
    }

    /// Convert integer to fixed-point representation
    pub fn int_to_fixed_u64(value: i64) -> u64 {
        (value * (SCALE as i64)) as u64
    }

    /// Convert fixed-point back to float (for verification)
    pub fn fixed_to_float(value: u64) -> f64 {
        (value as f64) / (SCALE as f64)
    }

    /// Fixed-point addition
    pub fn add_fixed_u64(a: u64, b: u64) -> u64 {
        a.saturating_add(b)
    }

    /// Fixed-point subtraction
    pub fn sub_fixed_u64(a: u64, b: u64) -> u64 {
        a.saturating_sub(b)
    }

    /// Fixed-point multiplication
    pub fn mul_fixed_u64(a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) / SCALE as u128) as u64
    }

    /// Fixed-point division
    pub fn div_fixed_u64(a: u64, b: u64) -> u64 {
        if b == 0 {
            0
        } else {
            ((a as u128 * SCALE as u128) / b as u128) as u64
        }
    }

    /// Check if value represents a reasonable gradient (not too large)
    pub fn is_reasonable_gradient(value: f64) -> bool {
        value.abs() < 100.0 // Reasonable threshold for gradients
    }

    /// Check if value represents a reasonable weight
    pub fn is_reasonable_weight(value: f64) -> bool {
        value.abs() < 1000.0 // Reasonable threshold for weights
    }

    /// Validate SGD update: w_new = w_old - lr * grad
    pub fn validate_sgd_update(w_old: f64, grad: f64, lr: f64, w_new: f64) -> bool {
        let expected = w_old - lr * grad;
        (w_new - expected).abs() < 1e-6 // Small tolerance for floating point errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_conversion() {
        let value = 3.14159;
        let fixed = primitives::float_to_fixed_u64(value);
        let recovered = primitives::fixed_to_float(fixed);
        assert!((value - recovered).abs() < 0.01, "Fixed-point conversion should be accurate");
    }

    #[test]
    fn test_primitive_arithmetic() {
        let a = primitives::float_to_fixed_u64(2.5);
        let b = primitives::float_to_fixed_u64(1.5);
        
        let sum = primitives::add_fixed_u64(a, b);
        let diff = primitives::sub_fixed_u64(a, b);
        let product = primitives::mul_fixed_u64(a, b);
        let quotient = primitives::div_fixed_u64(a, b);
        
        assert!((primitives::fixed_to_float(sum) - 4.0).abs() < 0.01);
        assert!((primitives::fixed_to_float(diff) - 1.0).abs() < 0.01);
        assert!((primitives::fixed_to_float(product) - 3.75).abs() < 0.01);
        assert!((primitives::fixed_to_float(quotient) - (2.5/1.5)).abs() < 0.01);
    }

    #[test]
    fn test_sgd_validation() {
        let w_old = 1.0;
        let grad = 0.1;
        let lr = 0.01;
        let w_new = w_old - lr * grad;
        
        assert!(primitives::validate_sgd_update(w_old, grad, lr, w_new));
        assert!(!primitives::validate_sgd_update(w_old, grad, lr, w_new + 0.1));
    }

    #[test]
    fn test_reasonable_values() {
        assert!(primitives::is_reasonable_gradient(0.1));
        assert!(!primitives::is_reasonable_gradient(1000.0));
        
        assert!(primitives::is_reasonable_weight(10.0));
        assert!(!primitives::is_reasonable_weight(10000.0));
    }
}