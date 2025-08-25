use std::fmt::Display;

use crate::{error::Error, sede::Sede};

#[derive(Clone, Copy)]
pub enum InitializationType {
    /// Best for tanh, sigmoid (default)
    Xavier,
    /// Best for Relu, swish
    He,
    /// set all weights the same
    Fixed(f32),
    ///-1 to 1
    Random,
}

pub fn calc_initialization(
    typ: InitializationType,
    prev_layer_size: usize,
    next_layer_size: usize,
) -> f32 {
    match typ {
        InitializationType::Random => fastrand::f32() * 2. - 1.,
        InitializationType::He => {
            (fastrand::f32() * 2. - 1.) * (6.0 / prev_layer_size as f32).sqrt()
        }
        InitializationType::Xavier => {
            (fastrand::f32() * 2. - 1.) * (6.0 / (prev_layer_size + next_layer_size) as f32).sqrt()
        }
        InitializationType::Fixed(val) => val,
    }
}

impl Display for InitializationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitializationType::Random => write!(f, "Random"),
            InitializationType::He => write!(f, "He"),
            InitializationType::Xavier => write!(f, "Xavier"),
            InitializationType::Fixed(val) => write!(f, "Fixed({})", val),
        }
    }
}

impl Sede for InitializationType {
    fn serialize(&self) -> String {
        format!("{}", self)
    }

    fn deserialize(s: &str) -> Result<Self, Error> {
        if s == "Random" {
            Ok(InitializationType::Random)
        } else if s == "He" {
            Ok(InitializationType::He)
        } else if s == "Xavier" {
            Ok(InitializationType::Xavier)
        } else if let Some(val) = s.strip_prefix("Fixed(").and_then(|s| s.strip_suffix(')')) {
            val.parse::<f32>()
                .map(InitializationType::Fixed)
                .map_err(|_| Error::SerializationError(format!("Invalid Fixed value: {}", val)))
        } else {
            Err(Error::SerializationError(format!(
                "Unknown initialization type: {}",
                s
            )))
        }
    }
}

// -------------------- Tests for initialization --------------------

#[cfg(test)]
mod tests {
    use crate::initialization::InitializationType;
    use crate::nn::NN;
    use fastrand;

    #[test]
    fn test_fixed_initialization() {
        let fixed = 0.5f32;
        let nn = NN::new(&[4, 3, 2]).with_initialization(InitializationType::Fixed(fixed));
        let weights = nn.get_weights();
        assert!(weights.iter().all(|&w| (w - fixed).abs() < 1e-6));
    }

    #[test]
    fn test_random_distribution_stats() {
        use std::f32;
        // create reasonably large network to sample many weights
        fastrand::seed(12345);
        let nn = NN::new(&[100, 50, 20]).with_initialization(InitializationType::Random);
        let vals: Vec<f32> = nn.get_weights();
        // mean should be near 0 for symmetric Random in [-1,1]
        let mean: f32 = vals.iter().copied().sum::<f32>() / vals.len() as f32;
        // variance approx 1/3 for uniform[-1,1]
        let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;

        assert!(mean.abs() < 0.05, "mean too far from 0: {}", mean);
        assert!((var - 0.3333).abs() < 0.05, "variance off: {}", var);

        // ensure values fall within [-1,1]
        assert!(vals
            .iter()
            .all(|&v| v >= -1.0 - f32::EPSILON && v <= 1.0 + f32::EPSILON));
    }
}
