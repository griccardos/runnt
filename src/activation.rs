use std::{fmt::Display, str::FromStr};

#[derive(Clone, Copy)]
pub enum ActivationType {
    Relu,    // max(0,val)
    Sigmoid, // 0 to 1
    Linear,  // val
    Tanh,    //-1 to 1
    Swish,   //x * sigmoid(x)
}

pub fn activate(val: f32, ltype: ActivationType) -> f32 {
    match ltype {
        ActivationType::Relu => val.max(0.),
        ActivationType::Sigmoid => 1. / (1. + (-val).exp()),
        ActivationType::Linear => val,
        ActivationType::Tanh => val.tanh(),
        ActivationType::Swish => val * (1. / (1. + (-val).exp())), // x * sigmoid(x)
    }
}
/// `value` is the pre-activation, `activated_value` is the activated value
pub fn activate_der(value: f32, activated_value: f32, ltype: ActivationType) -> f32 {
    match ltype {
        ActivationType::Relu => {
            if activated_value > 0. {
                1.
            } else {
                0.
            }
        }
        ActivationType::Sigmoid => activated_value * (1. - activated_value),
        ActivationType::Linear => 1.,
        ActivationType::Tanh => 1. - activated_value * activated_value, //1-tanh(x)^2 (since we have val=tanh(x), we just use val)
        ActivationType::Swish => {
            // swish derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            let sigmoid = 1. / (1. + (-value).exp());
            sigmoid + value * sigmoid * (1. - sigmoid)
        }
    }
}

impl Display for ActivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationType::Relu => write!(f, "Relu"),
            ActivationType::Sigmoid => write!(f, "Sigmoid"),
            ActivationType::Linear => write!(f, "Linear"),
            ActivationType::Tanh => write!(f, "Tanh"),
            ActivationType::Swish => write!(f, "Swish"),
        }
    }
}

impl FromStr for ActivationType {
    type Err = std::io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "relu" => Ok(ActivationType::Relu),
            "sigmoid" => Ok(ActivationType::Sigmoid),
            "linear" => Ok(ActivationType::Linear),
            "tanh" => Ok(ActivationType::Tanh),
            "swish" | "silu" => Ok(ActivationType::Swish),
            _ => Err(std::io::Error::other("Unknown activation type")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that ReLU activation returns 0 for negative input and the input itself for positive input
    #[test]
    fn test_relu() {
        assert_eq!(activate(-1.0, ActivationType::Relu), 0.0);
        assert_eq!(activate(2.5, ActivationType::Relu), 2.5);
    }

    // Test that Sigmoid activation returns values between 0 and 1, and matches known values for 0 and large positive/negative inputs
    #[test]
    fn test_sigmoid() {
        let val = activate(0.0, ActivationType::Sigmoid);
        assert!((val - 0.5).abs() < 1e-6);
        let val = activate(100.0, ActivationType::Sigmoid);
        assert!((val - 1.0).abs() < 1e-6);
        let val = activate(-100.0, ActivationType::Sigmoid);
        assert!((val - 0.0).abs() < 1e-6);
    }

    // Test that Linear activation returns the input value unchanged
    #[test]
    fn test_linear() {
        assert_eq!(activate(-5.0, ActivationType::Linear), -5.0);
        assert_eq!(activate(3.55, ActivationType::Linear), 3.55);
    }

    // Test that Tanh activation returns values between -1 and 1, and matches known values for 0 and large positive/negative inputs
    #[test]
    fn test_tanh() {
        let val = activate(0.0, ActivationType::Tanh);
        assert!((val - 0.0).abs() < 1e-6);
        let val = activate(100.0, ActivationType::Tanh);
        assert!((val - 1.0).abs() < 1e-6);
        let val = activate(-100.0, ActivationType::Tanh);
        assert!((val + 1.0).abs() < 1e-6);
    }

    // Test that Swish activation returns x * sigmoid(x), and matches known values for 0 and positive/negative inputs
    #[test]
    fn test_swish() {
        let val = activate(0.0, ActivationType::Swish);
        assert!((val - 0.0).abs() < 1e-6);
        let val = activate(1.0, ActivationType::Swish);
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((val - 1.0 * expected).abs() < 1e-6);
        let val = activate(-1.0, ActivationType::Swish);
        let expected = 1.0 / (1.0 + (1.0f32).exp());
        assert!((val - (-expected)).abs() < 1e-6);
    }

    #[test]
    fn swish_derivative() {
        let z = 0.3f32;
        let a = activate(z, ActivationType::Swish);
        // swish(x) = x * sigmoid(x)
        // derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        let sigmoid = 1.0f32 / (1.0f32 + (-z).exp());
        let expected = sigmoid + z * sigmoid * (1.0 - sigmoid);
        let derivative = activate_der(z, a, ActivationType::Swish);
        assert!((derivative - expected).abs() < 1e-6);

        let val = 100.;
        let a = activate(val, ActivationType::Swish);
        let der = activate_der(val, a, ActivationType::Swish);
        assert!(
            (der - 1.0).abs() < 1e-6,
            "Swish derivative at large value should be close to 1, got {}",
            der
        );

        let val = -100.;
        let a = activate(val, ActivationType::Swish);
        let der = activate_der(val, a, ActivationType::Swish);
        assert!(
            (der + 0.0).abs() < 1e-6,
            "Swish derivative at large negative value should be close to 0, got {}",
            der
        );

        let val = 0.;
        let a = activate(val, ActivationType::Swish);
        let der = activate_der(val, a, ActivationType::Swish);
        assert!(
            (der - 0.5).abs() < 1e-6,
            "Swish derivative at zero should be close to 0.5, got {}",
            der
        );
    }
}
