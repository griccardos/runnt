use std::fmt::Display;

use crate::{error::Error, sede::Sede};

#[derive(Clone, Copy)]
pub enum InitializationType {
    ///-1 to 1
    Random,
    Xavier,
    Fixed(f32),
}

pub fn calc_initialization(typ: InitializationType, prev_layer_size: usize) -> f32 {
    match typ {
        InitializationType::Random => fastrand::f32() * 2. - 1.,
        InitializationType::Xavier => {
            (fastrand::f32() * 2. - 1.) * (1.0 / prev_layer_size as f32).sqrt()
        }
        InitializationType::Fixed(val) => val,
    }
}

impl Display for InitializationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitializationType::Random => write!(f, "Random"),
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
