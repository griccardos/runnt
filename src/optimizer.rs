use ndarray::Array2;

//used for selection
pub enum OptimizerType {
    None,
    Momentum { beta: f32 },
}

pub(crate) struct Optimizer {
    optimizer: OptimizerInternal,
}

impl Optimizer {
    pub(crate) fn none() -> Self {
        Self {
            optimizer: OptimizerInternal::None,
        }
    }

    pub(crate) fn new(typ: OptimizerType, weights: &[Array2<f32>], bias: &[Array2<f32>]) -> Self {
        match typ {
            OptimizerType::None => Optimizer {
                optimizer: OptimizerInternal::None,
            },
            OptimizerType::Momentum { beta } => Optimizer {
                optimizer: OptimizerInternal::Momentum(Momentum::new(beta, weights, bias)),
            },
        }
    }
}

//used internally to hold data
pub(crate) enum OptimizerInternal {
    None,
    Momentum(Momentum),
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

impl Optimizer {
    ///this updates the gradients with the optimizer, and learning rate
    ///e.g. for None, it returns `-learning_rate*gradient`
    ///this returns (`weights_grad`,`bias_grad`)
    pub fn calc_gradient_update(
        &mut self,
        mut weight_gradients: Vec<Array2<f32>>,
        mut bias_gradients: Vec<Array2<f32>>,
        learning_rate: f32,
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

        let (updated_weights, updated_bias) =
            optimizer.calc_gradient_update(weight_grad.clone(), bias_grad.clone(), learning_rate);

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
        let (updated_weights, updated_bias) =
            optimizer.calc_gradient_update(weight_grad.clone(), bias_grad.clone(), learning_rate);

        let expected_weights = array![[0.5 * -0.1, -0.5 * -0.1], [1.0 * -0.1, -1.0 * -0.1]];
        let expected_bias = array![[0.1 * -0.1, -0.1 * -0.1]];

        assert!(approx_eq(&updated_weights[0], &expected_weights, 1e-6));
        assert!(approx_eq(&updated_bias[0], &expected_bias, 1e-6));

        // Second update: velocity should accumulate
        let (updated_weights2, updated_bias2) =
            optimizer.calc_gradient_update(weight_grad.clone(), bias_grad.clone(), learning_rate);

        let expected_weights2 = &expected_weights * beta + &expected_weights;
        let expected_bias2 = &expected_bias * beta + &expected_bias;

        assert!(approx_eq(&updated_weights2[0], &expected_weights2, 1e-6));
        assert!(approx_eq(&updated_bias2[0], &expected_bias2, 1e-6));
    }
}
