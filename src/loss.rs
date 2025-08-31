use std::fmt::{Display, Formatter};

use ndarray::Array2;

use crate::{error::Error, sede::Sede};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    ///Mean Squared Error,
    /// defined as `0.5*(y_true - y_pred)^2`,
    /// with the derivative being `(y_pred - y_true)`.
    /// This is the textbook definition, but is different to keras and other frameworks
    /// which has derivative as `2(y_pred - y_true)`.
    MSE,
    ///Will apply Softmax, then compute multi-class cross-entropy
    /// with the derivative wrt to logits being `y_pred - y_true`
    SoftmaxAndCrossEntropy, // softmax + multi-class cross-entropy
    ///Will apply Sigmoid, then compute binary cross-entropy
    /// with the derivative wrt to logits being `y_pred - y_true`
    BinaryCrossEntropy, // sigmoid + binary cross-entropy
}
impl Loss {
    /// calc gradient for each example
    pub fn gradient(&self, outputs: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
        //MSE LOSS
        //E = error / loss
        //a = value after activation
        //t = target value

        // E = 0.5* (t-a)^2
        // error gradient: dE/da = -1*2*0.5*(t-a) = -(t-a)=a-t

        //SOFTMAX+CROSS ENTROPY LOSS
        // softmax and together with crossentropy derivative
        // conveniently is also output-target (same as mse loss)

        outputs - target
    }
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
        let l = match s.to_lowercase().as_str() {
            "mse" => Loss::MSE,
            "softmaxandcrossentropy" => Loss::SoftmaxAndCrossEntropy,
            "binarycrossentropy" | "binary_crossentropy" => Loss::BinaryCrossEntropy,
            _ => panic!("Unknown loss function: {}", s),
        };
        Ok(l)
    }
}
