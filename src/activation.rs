use crate::{error::Error, sede::Sede};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    Relu,    // max(0,val)
    Sigmoid, // 0 to 1
    Linear,  // val
    Tanh,    //-1 to 1
    Swish,   //x * sigmoid(x)
}

pub fn activate(val: f32, ltype: Activation) -> f32 {
    match ltype {
        Activation::Relu => val.max(0.),
        Activation::Sigmoid => 1. / (1. + (-val).exp()),
        Activation::Linear => val,
        Activation::Tanh => val.tanh(),
        Activation::Swish => val * (1. / (1. + (-val).exp())), // x * sigmoid(x)
    }
}
/// `value` is the pre-activation, `activated_value` is the activated value
pub fn activate_der(value: f32, activated_value: f32, ltype: Activation) -> f32 {
    match ltype {
        Activation::Relu => {
            if activated_value > 0. {
                1.
            } else {
                0.
            }
        }
        Activation::Sigmoid => activated_value * (1. - activated_value),
        Activation::Linear => 1.,
        Activation::Tanh => 1. - activated_value * activated_value, //1-tanh(x)^2 (since we have val=tanh(x), we just use val)
        Activation::Swish => {
            // swish derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            let sigmoid = 1. / (1. + (-value).exp());
            sigmoid + value * sigmoid * (1. - sigmoid)
        }
    }
}

impl Sede for Activation {
    fn serialize(&self) -> String {
        match self {
            Activation::Relu => "Relu",
            Activation::Sigmoid => "Sigmoid",
            Activation::Linear => "Linear",
            Activation::Tanh => "Tanh",
            Activation::Swish => "Swish",
        }
        .to_string()
    }

    fn deserialize(s: &str) -> Result<Self, Error>
    where
        Self: Sized,
    {
        match s.to_lowercase().as_str() {
            "relu" => Ok(Activation::Relu),
            "sigmoid" => Ok(Activation::Sigmoid),
            "linear" => Ok(Activation::Linear),
            "tanh" => Ok(Activation::Tanh),
            "swish" | "silu" => Ok(Activation::Swish),
            _ => Err(std::io::Error::other("Unknown activation type").into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that ReLU activation returns 0 for negative input and the input itself for positive input
    #[test]
    fn test_relu() {
        assert_eq!(activate(-1.0, Activation::Relu), 0.0);
        assert_eq!(activate(2.5, Activation::Relu), 2.5);
    }

    // Test that Sigmoid activation returns values between 0 and 1, and matches known values for 0 and large positive/negative inputs
    #[test]
    fn test_sigmoid() {
        let val = activate(0.0, Activation::Sigmoid);
        assert!((val - 0.5).abs() < 1e-6);
        let val = activate(100.0, Activation::Sigmoid);
        assert!((val - 1.0).abs() < 1e-6);
        let val = activate(-100.0, Activation::Sigmoid);
        assert!((val - 0.0).abs() < 1e-6);
    }

    // Test that Linear activation returns the input value unchanged
    #[test]
    fn test_linear() {
        assert_eq!(activate(-5.0, Activation::Linear), -5.0);
        assert_eq!(activate(3.55, Activation::Linear), 3.55);
    }

    // Test that Tanh activation returns values between -1 and 1, and matches known values for 0 and large positive/negative inputs
    #[test]
    fn test_tanh() {
        let val = activate(0.0, Activation::Tanh);
        assert!((val - 0.0).abs() < 1e-6);
        let val = activate(100.0, Activation::Tanh);
        assert!((val - 1.0).abs() < 1e-6);
        let val = activate(-100.0, Activation::Tanh);
        assert!((val + 1.0).abs() < 1e-6);
    }

    // Test that Swish activation returns x * sigmoid(x), and matches known values for 0 and positive/negative inputs
    #[test]
    fn test_swish() {
        let val = activate(0.0, Activation::Swish);
        assert!((val - 0.0).abs() < 1e-6);
        let val = activate(1.0, Activation::Swish);
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((val - 1.0 * expected).abs() < 1e-6);
        let val = activate(-1.0, Activation::Swish);
        let expected = 1.0 / (1.0 + (1.0f32).exp());
        assert!((val - (-expected)).abs() < 1e-6);
    }

    #[test]
    fn swish_derivative() {
        let z = 0.3f32;
        let a = activate(z, Activation::Swish);
        // swish(x) = x * sigmoid(x)
        // derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        let sigmoid = 1.0f32 / (1.0f32 + (-z).exp());
        let expected = sigmoid + z * sigmoid * (1.0 - sigmoid);
        let derivative = activate_der(z, a, Activation::Swish);
        assert!((derivative - expected).abs() < 1e-6);

        let val = 100.;
        let a = activate(val, Activation::Swish);
        let der = activate_der(val, a, Activation::Swish);
        assert!(
            (der - 1.0).abs() < 1e-6,
            "Swish derivative at large value should be close to 1, got {}",
            der
        );

        let val = -100.;
        let a = activate(val, Activation::Swish);
        let der = activate_der(val, a, Activation::Swish);
        assert!(
            (der + 0.0).abs() < 1e-6,
            "Swish derivative at large negative value should be close to 0, got {}",
            der
        );

        let val = 0.;
        let a = activate(val, Activation::Swish);
        let der = activate_der(val, a, Activation::Swish);
        assert!(
            (der - 0.5).abs() < 1e-6,
            "Swish derivative at zero should be close to 0.5, got {}",
            der
        );
    }
}
