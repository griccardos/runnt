use std::fmt::Display;

use ndarray::Array2;

//used for selection
#[derive(Clone, Copy)]
pub enum OptimizerType {
    None,
    Momentum { beta: f32 },
    Adam { beta1: f32, beta2: f32 },
}
impl OptimizerType {
    pub fn momentum() -> Self {
        OptimizerType::Momentum { beta: 0.9 }
    }

    pub fn adam() -> Self {
        OptimizerType::Adam {
            beta1: 0.9,
            beta2: 0.999,
        }
    }
}

pub(crate) struct Optimizer {
    optimizer_type: OptimizerType,
    optimizer: OptimizerInternal,
}

impl Optimizer {
    pub(crate) fn none() -> Self {
        Self {
            optimizer_type: OptimizerType::None,
            optimizer: OptimizerInternal::None,
        }
    }

    pub(crate) fn new(typ: OptimizerType, weights: &[Array2<f32>], bias: &[Array2<f32>]) -> Self {
        match typ {
            OptimizerType::None => Optimizer {
                optimizer_type: typ,
                optimizer: OptimizerInternal::None,
            },
            OptimizerType::Momentum { beta } => Optimizer {
                optimizer_type: typ,
                optimizer: OptimizerInternal::Momentum(Momentum::new(beta, weights, bias)),
            },
            OptimizerType::Adam { beta1, beta2 } => Optimizer {
                optimizer_type: typ,
                optimizer: OptimizerInternal::Adam(Adam::new(beta1, beta2, weights, bias)),
            },
        }
    }
}

impl Display for OptimizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerType::None => write!(f, "None"),
            OptimizerType::Momentum { beta } => write!(f, "Momentum({})", beta),
            OptimizerType::Adam { beta1, beta2 } => write!(f, "Adam({},{})", beta1, beta2),
        }
    }
}

//used internally to hold data
pub(crate) enum OptimizerInternal {
    None,
    Momentum(Momentum),
    Adam(Adam),
}

pub(crate) struct Momentum {
    weight_velocity: Vec<Array2<f32>>,
    bias_velocity: Vec<Array2<f32>>,
    beta: f32,
}

impl Momentum {
    ///Beta is the amount of the old velocity we will keep (default=0.9)
    pub fn new(beta: f32, weights: &[Array2<f32>], bias: &[Array2<f32>]) -> Self {
        Momentum {
            weight_velocity: weights.iter().map(|a| Array2::zeros(a.dim())).collect(),
            bias_velocity: bias.iter().map(|a| Array2::zeros(a.dim())).collect(),

            beta,
        }
    }
}

pub(crate) struct Adam {
    beta1: f32,
    beta2: f32,
    weight_velocity1: Vec<Array2<f32>>,
    bias_velocity1: Vec<Array2<f32>>,
    weight_velocity2: Vec<Array2<f32>>,
    bias_velocity2: Vec<Array2<f32>>,
}

impl Adam {
    ///Beta1 is the amount of the old velocity we will keep (default=0.9)
    ///Beta2 is the amount of the old squared velocity we will keep (default=0.999)
    pub fn new(beta1: f32, beta2: f32, weights: &[Array2<f32>], bias: &[Array2<f32>]) -> Self {
        Adam {
            beta1,
            beta2,
            weight_velocity1: weights.iter().map(|a| Array2::zeros(a.dim())).collect(),
            bias_velocity1: bias.iter().map(|a| Array2::zeros(a.dim())).collect(),
            weight_velocity2: weights.iter().map(|a| Array2::zeros(a.dim())).collect(),
            bias_velocity2: bias.iter().map(|a| Array2::zeros(a.dim())).collect(),
        }
    }
}

impl Optimizer {
    ///this updates the gradients with the optimizer, and learning rate
    ///e.g. for None, it returns `-learning_rate*gradient`
    ///this returns (`weights_grad`,`bias_grad`)
    pub fn calc_gradient_update(
        &mut self,
        mut weight_gradients: Vec<Array2<f32>>,
        mut bias_gradients: Vec<Array2<f32>>,
        learning_rate: f32,
        step: usize,
    ) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        for l in 0..weight_gradients.len() {
            match &mut self.optimizer {
                OptimizerInternal::None => {
                    weight_gradients[l].mapv_inplace(|a| a * -learning_rate);
                    bias_gradients[l].mapv_inplace(|a| a * -learning_rate);
                }
                OptimizerInternal::Momentum(momentum) => {
                    momentum.weight_velocity[l] = &momentum.weight_velocity[l] * momentum.beta
                        + &weight_gradients[l] * -learning_rate;

                    weight_gradients[l] = momentum.weight_velocity[l].clone();

                    momentum.bias_velocity[l] = &momentum.bias_velocity[l] * momentum.beta
                        + &bias_gradients[l] * -learning_rate;

                    bias_gradients[l] = momentum.bias_velocity[l].clone();
                }
                OptimizerInternal::Adam(adam) => {
                    // Update first moment estimate
                    adam.weight_velocity1[l] = &adam.weight_velocity1[l] * adam.beta1
                        + &weight_gradients[l] * (1.0 - adam.beta1);
                    adam.bias_velocity1[l] = &adam.bias_velocity1[l] * adam.beta1
                        + &bias_gradients[l] * (1.0 - adam.beta1);

                    // Update second moment estimate
                    adam.weight_velocity2[l] = &adam.weight_velocity2[l] * adam.beta2
                        + &weight_gradients[l].mapv(|x| x.powi(2)) * (1.0 - adam.beta2);
                    adam.bias_velocity2[l] = &adam.bias_velocity2[l] * adam.beta2
                        + &bias_gradients[l].mapv(|x| x.powi(2)) * (1.0 - adam.beta2);

                    // Compute bias-corrected first moment estimate
                    let weight_m = &adam.weight_velocity1[l] / (1.0 - adam.beta1.powi(step as i32)); // assuming t=2 for simplicity
                    let bias_m = &adam.bias_velocity1[l] / (1.0 - adam.beta1.powi(step as i32));

                    // Compute bias-corrected second moment estimate
                    let weight_v = &adam.weight_velocity2[l] / (1.0 - adam.beta2.powi(step as i32));
                    let bias_v = &adam.bias_velocity2[l] / (1.0 - adam.beta2.powi(step as i32));

                    // Update weights and biases
                    weight_gradients[l] =
                        weight_m / (weight_v.mapv(|x| x.sqrt()) + 1e-8) * -learning_rate;
                    bias_gradients[l] =
                        bias_m / (bias_v.mapv(|x| x.sqrt()) + 1e-8) * -learning_rate;
                }
            }
        }
        (weight_gradients, bias_gradients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn approx_eq(a: &Array2<f32>, b: &Array2<f32>, eps: f32) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
    }

    #[test]
    fn test_none_optimizer() {
        let weights = vec![array![[1.0, 2.0], [3.0, 4.0]]];
        let bias = vec![array![[1.0, 2.0]]];
        let mut optimizer = Optimizer::new(OptimizerType::None, &weights, &bias);

        let weight_grad = vec![array![[0.5, -0.5], [1.0, -1.0]]];
        let bias_grad = vec![array![[0.1, -0.1]]];
        let learning_rate = 0.1;

        let (updated_weights, updated_bias) = optimizer.calc_gradient_update(
            weight_grad.clone(),
            bias_grad.clone(),
            learning_rate,
            1,
        );

        let expected_weights = array![[0.5 * -0.1, -0.5 * -0.1], [1.0 * -0.1, -1.0 * -0.1]];
        let expected_bias = array![[0.1 * -0.1, -0.1 * -0.1]];

        assert!(approx_eq(&updated_weights[0], &expected_weights, 1e-6));
        assert!(approx_eq(&updated_bias[0], &expected_bias, 1e-6));
    }

    #[test]
    fn test_momentum_optimizer() {
        let weights = vec![array![[1.0, 2.0], [3.0, 4.0]]];
        let bias = vec![array![[1.0, 2.0]]];
        let beta = 0.9;
        let mut optimizer = Optimizer::new(OptimizerType::Momentum { beta }, &weights, &bias);

        let weight_grad = vec![array![[0.5, -0.5], [1.0, -1.0]]];
        let bias_grad = vec![array![[0.1, -0.1]]];
        let learning_rate = 0.1;

        // First update: velocity should be just -learning_rate * gradient
        let (updated_weights, updated_bias) = optimizer.calc_gradient_update(
            weight_grad.clone(),
            bias_grad.clone(),
            learning_rate,
            1,
        );

        let expected_weights = array![[0.5 * -0.1, -0.5 * -0.1], [1.0 * -0.1, -1.0 * -0.1]];
        let expected_bias = array![[0.1 * -0.1, -0.1 * -0.1]];

        assert!(approx_eq(&updated_weights[0], &expected_weights, 1e-6));
        assert!(approx_eq(&updated_bias[0], &expected_bias, 1e-6));

        // Second update: velocity should accumulate
        let (updated_weights2, updated_bias2) = optimizer.calc_gradient_update(
            weight_grad.clone(),
            bias_grad.clone(),
            learning_rate,
            2,
        );

        let expected_weights2 = &expected_weights * beta + &expected_weights;
        let expected_bias2 = &expected_bias * beta + &expected_bias;

        assert!(approx_eq(&updated_weights2[0], &expected_weights2, 1e-6));
        assert!(approx_eq(&updated_bias2[0], &expected_bias2, 1e-6));
    }

    #[test]
    fn test_adam_optimizer() {
        let weights = vec![array![[1.0, 2.0], [3.0, 4.0]]];
        let bias = vec![array![[1.0, 2.0]]];
        let beta1 = 0.9;
        let beta2 = 0.999;
        let mut optimizer = Optimizer::new(OptimizerType::Adam { beta1, beta2 }, &weights, &bias);

        let weight_grad = vec![array![[0.5, -0.5], [1.0, -1.0]]];
        let bias_grad = vec![array![[0.1, -0.1]]];
        let learning_rate = 0.1;

        // First update: with the current implementation and bias-correction,
        // the first-step produces update ~= -learning_rate * sign(gradient)
        let (updated_weights, updated_bias) = optimizer.calc_gradient_update(
            weight_grad.clone(),
            bias_grad.clone(),
            learning_rate,
            1,
        );

        let expected_weights = array![[-0.1, 0.1], [-0.1, 0.1]];
        let expected_bias = array![[-0.1, 0.1]];

        assert!(approx_eq(&updated_weights[0], &expected_weights, 1e-6));
        assert!(approx_eq(&updated_bias[0], &expected_bias, 1e-6));
    }
}
