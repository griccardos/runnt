use std::fmt::Display;

use ndarray::Array2;

use crate::{
    activation::Activation,
    dropout::Dropout,
    initialization::{Initialization, calc_initialization},
    regularization::Regularization,
    sede::Sede,
};
#[derive(Debug, PartialEq)]
pub struct Dense {
    pub(crate) weights: Array2<f32>, //  2D matrix e.g. [2x2]
    pub(crate) bias: Array2<f32>,    // 2D matrix
    pub(crate) initialization: Initialization,
    pub(crate) activation: Activation,
    pub(crate) regularization: Regularization,
    pub(crate) dropout: Option<Dropout>,
}
impl Dense {
    pub(crate) fn new(
        inputs: usize,
        outputs: usize,
        initialization: Initialization,
        activation: Activation,
        regularization: Regularization,
        dropout: Option<f32>,
    ) -> Self {
        let weights = Array2::zeros((inputs, outputs));
        let bias = Array2::zeros((1, outputs));
        let mut s = Self {
            weights,
            bias,
            initialization,
            activation,
            regularization,
            dropout: dropout.map(|a| Dropout::new(a, outputs)),
        };
        s.reinitialize();
        s
    }
    pub(crate) fn reinitialize(&mut self) {
        let inputs = self.weights.shape()[0];
        let outputs = self.weights.shape()[1];

        self.weights
            .mapv_inplace(|_| calc_initialization(self.initialization, inputs, outputs));
    }
}

impl Display for Dense {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Weights:{}x{} Bias:{} Initialization:{}",
            self.weights.shape()[0],
            self.weights.shape()[1],
            self.bias.shape()[1],
            self.initialization,
        )
    }
}

#[derive(Clone)]
pub struct DenseBuilder {
    pub(crate) size: usize,
    pub(crate) initialization: Initialization,
    pub(crate) activation: Activation,
    pub(crate) regularization: Regularization,
    pub(crate) dropout: Option<f32>,
}
impl Into<DenseBuilder> for usize {
    fn into(self) -> DenseBuilder {
        dense(self)
    }
}
pub fn dense(size: usize) -> DenseBuilder {
    DenseBuilder {
        size,
        initialization: Initialization::Xavier,
        activation: Activation::Sigmoid,
        regularization: Regularization::None,
        dropout: None,
    }
}

impl DenseBuilder {
    pub fn initializer(mut self, init_type: Initialization) -> Self {
        self.initialization = init_type;
        self
    }
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
    pub fn regularization(mut self, reg: Regularization) -> Self {
        self.regularization = reg.clone();
        self
    }
    /// Dropout rate in range (0,1)
    /// A good starting point is 0.2
    pub fn dropout(mut self, rate: f32) -> Self {
        assert!(
            rate > 0. && rate < 1.,
            "Dropout rate must be in range (0,1)"
        );
        self.dropout = Some(rate);
        self
    }

    //we need input size to build
    pub(crate) fn build(&self, input: usize) -> Dense {
        Dense::new(
            input,
            self.size,
            self.initialization,
            self.activation,
            self.regularization.clone(),
            self.dropout,
        )
    }
}

impl Sede for Dense {
    fn serialize(&self) -> String {
        format!(
            "Weights={}_Bias={}_Initialization={}_Activation={}_Regularization={}_Dropout={}",
            self.weights.serialize(),
            self.bias.serialize(),
            self.initialization,
            self.activation.serialize(),
            self.regularization.serialize(),
            self.dropout.serialize()
        )
    }

    fn deserialize(s: &str) -> Result<Self, crate::error::Error>
    where
        Self: Sized,
    {
        let parts: Vec<&str> = s.split('_').collect();
        let mut weights = Array2::zeros((0, 0));
        let mut bias = Array2::zeros((0, 0));
        let mut initialization = Initialization::Xavier;
        let mut activation = Activation::Sigmoid; // Default value 
        let mut regularization = Regularization::None;
        let mut dropout = None;
        for p in parts {
            if let Some((name, value)) = p.split_once("=") {
                if name == "Weights" {
                    weights = Array2::deserialize(value)?;
                } else if name == "Bias" {
                    bias = Array2::deserialize(value)?;
                } else if name == "Initialization" {
                    initialization = Initialization::deserialize(value)?;
                } else if name == "Activation" {
                    activation = Activation::deserialize(value)?;
                } else if name == "Regularization" {
                    regularization = Regularization::deserialize(value)?;
                } else if name == "Dropout" {
                    dropout = Option::<Dropout>::deserialize(value)?;
                }
            }
        }
        Ok(Dense {
            weights,
            bias,
            initialization,
            activation,
            regularization,
            dropout,
        })
    }
}
mod tests {
    #[allow(unused_imports)]
    use crate::{layer::Dense, sede::Sede};

    #[test]
    fn sede() {
        let d = Dense::new(
            2,
            3,
            crate::initialization::Initialization::He,
            crate::activation::Activation::Relu,
            crate::regularization::Regularization::l1(),
            Some(0.3),
        );
        let ser = d.serialize();
        let des: Dense = Dense::deserialize(&ser).unwrap();
        assert_eq!(d, des);
    }
}
