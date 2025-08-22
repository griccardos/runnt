use std::fmt::{Display, Formatter};

use crate::{error::Error, sede::Sede};

pub enum Loss {
    ///Mean Squared Error, uses outputs as declared (Sigmoid, Tanh, Linear etc.)
    MSE,
    ///Will apply Softmax, then compute multi-class cross-entropy
    SoftmaxAndCrossEntropy, // softmax + multi-class cross-entropy
    ///Will apply Sigmoid, then compute binary cross-entropy
    BinaryCrossEntropy, // sigmoid + binary cross-entropy
}
impl Display for Loss {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Loss::MSE => write!(f, "Mean Squared Error"),
            Loss::SoftmaxAndCrossEntropy => write!(f, "Softmax and Cross-Entropy"),
            Loss::BinaryCrossEntropy => write!(f, "Binary Cross-Entropy"),
        }
    }
}
impl Sede for Loss {
    fn serialize(&self) -> String {
        match self {
            Loss::MSE => "MSE".to_string(),
            Loss::SoftmaxAndCrossEntropy => "SoftmaxAndCrossEntropy".to_string(),
            Loss::BinaryCrossEntropy => "BinaryCrossEntropy".to_string(),
        }
    }
    fn deserialize(s: &str) -> Result<Loss, Error> {
        let l = match s {
            "MSE" => Loss::MSE,
            "SoftmaxAndCrossEntropy" => Loss::SoftmaxAndCrossEntropy,
            "BinaryCrossEntropy" => Loss::BinaryCrossEntropy,
            _ => panic!("Unknown loss function: {}", s),
        };
        Ok(l)
    }
}
