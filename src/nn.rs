use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;

use ndarray::prelude::*;
use ndarray::Array;

use crate::activation::{activate, activate_der, ActivationType};
use crate::initialization::{calc_initialization, InitializationType};

pub struct NN {
    weights: Vec<Array2<f32>>, // layer * 2D matrix e.g. [2x2], [2x1]
    bias: Vec<Array2<f32>>,    //layer * 2D matrix
    shape: Vec<usize>,
    learning_rate: f32,
    error: f32, //currently MSE, may change
    hidden_type: ActivationType,
    output_type: ActivationType,
}

struct ErrorAndGradient {
    error: f32,
    gradient: Vec<f32>,
}

impl NN {
    ///```rust
    ///   //XOR
    ///    use runnt::{nn::NN,activation::ActivationType};
    ///    let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    ///    let outputs = [[0.], [1.], [1.], [0.]];
    ///
    ///        let mut nn = NN::new(&[2, 8, 1])
    ///        .with_learning_rate(0.2)
    ///        .with_hidden_type(ActivationType::Tanh)
    ///        .with_output_type(ActivationType::Linear);
    ///
    ///
    ///    for i in 0..5000 {
    ///        nn.fit_one(&inputs[i % 4], &outputs[i % 4]);
    ///    }
    ///    assert_eq!(nn.forward(&[0., 0.]).first().unwrap().round(), 0.);
    ///    assert_eq!(nn.forward(&[0., 1.]).first().unwrap().round(), 1.);
    ///    assert_eq!(nn.forward(&[1., 0.]).first().unwrap().round(), 1.);
    ///    assert_eq!(nn.forward(&[1., 1.]).first().unwrap().round(), 0.);
    ///```
    pub fn new(network_shape: &[usize]) -> Self {
        let mut weights = vec![];
        let mut bias = vec![];
        let mut values = vec![];

        let layers = network_shape.len();
        assert!(layers >= 2);

        for l1 in 0..layers {
            let this_size = network_shape[l1];

            values.push(Array::from_shape_fn([1, this_size], |_| 0.));

            //dont need for last layer
            if l1 < layers - 1 {
                let next_size = network_shape[l1 + 1];
                bias.push(Array::from_shape_fn([1, next_size], |_| 0.));
                weights.push(Array::from_shape_fn([this_size, next_size], |_| 0.));
            }
        }

        let s = Self {
            shape: Vec::from(network_shape),
            weights,
            bias,
            learning_rate: 0.01,
            error: 0.,
            hidden_type: ActivationType::Sigmoid,
            output_type: ActivationType::Linear,
        };
        s.with_initialization(InitializationType::Random)
    }

    pub fn with_learning_rate(mut self, rate: f32) -> NN {
        self.learning_rate = rate;
        self
    }

    pub fn with_hidden_type(mut self, types: ActivationType) -> Self {
        self.hidden_type = types;
        self
    }
    pub fn with_output_type(mut self, types: ActivationType) -> Self {
        self.output_type = types;
        self
    }
    pub fn with_initialization(mut self, typ: InitializationType) -> Self {
        self.reset_weights(typ);
        self
    }

    pub fn reset_weights(&mut self, typ: InitializationType) {
        let layers = self.shape.len();
        for l in 0..layers - 1 {
            let this_size = self.shape[l];
            self.bias[l].mapv_inplace(|_| calc_initialization(typ, this_size));
            self.weights[l].mapv_inplace(|_| calc_initialization(typ, this_size));
        }
    }

    ///Also known as Stochastic Gradient Descent i.e. Gradient descent with batch size = 1
    pub fn fit_one(&mut self, input: &[f32], targets: &[f32]) {
        self.fit(&[input], &[targets]);
    }

    /// Perform mini batch gradient descent on `batch_size`.
    /// If batch is smaller than data, will perform fit multiple times
    pub fn fit_batch_size(&mut self, inputs: &[&[f32]], targets: &[&[f32]], batch_size: usize) {
        inputs
            .chunks(batch_size)
            .zip(targets.chunks(batch_size))
            .for_each(|(inps, outs)| self.fit(inps, outs));
    }

    /// Gradient descent on entire batch.
    pub fn fit(&mut self, inputs: &[&[f32]], targets: &[&[f32]]) {
        //forward inputs and calculate error gradients
        //we add up the weights of all gradients, then we divide by count to get average
        //we then apply the gradients backwards to update the weights
        let mut weight_gradient_sum: Vec<Array2<f32>> = vec![];
        let mut bias_gradient_sum: Vec<Array2<f32>> = vec![];

        let mut errors = 0.;
        for (input, target) in inputs.iter().zip(targets) {
            let values = self.internal_forward(input);
            let outputs = values.last().unwrap().row(0).as_slice().unwrap().to_vec();
            let err_and_grad = Self::output_error_and_gradient(&outputs, target);
            errors += err_and_grad.error;

            let (weight_gradient, bias_gradient) = self.backward(values, err_and_grad.gradient);

            if weight_gradient_sum.is_empty() {
                weight_gradient_sum = weight_gradient;
            } else {
                for (s, w) in weight_gradient_sum.iter_mut().zip(&weight_gradient) {
                    *s += w;
                }
            }
            if bias_gradient_sum.is_empty() {
                bias_gradient_sum = bias_gradient;
            } else {
                for (s, w) in bias_gradient_sum.iter_mut().zip(&bias_gradient) {
                    *s += w;
                }
            }
        }

        let count = inputs.len() as f32;
        //calc MSE
        self.error = errors / count;

        //now get average of error gradient
        for layer in &mut weight_gradient_sum {
            layer.mapv_inplace(|a| a / count);
        }
        for layer in &mut bias_gradient_sum {
            layer.mapv_inplace(|a| a / count);
        }

        //propagate error gradients backwards
        self.apply_gradients(&weight_gradient_sum, &bias_gradient_sum);
    }

    /// Forward inputs into the network, and returns output result i.e. `prediction`
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let values = self.internal_forward(input);
        values.last().unwrap().row(0).as_slice().unwrap().to_vec()
    }

    /// convert to array, and forward, returning values for each layer
    fn internal_forward(&self, input: &[f32]) -> Vec<Array2<f32>> {
        //set inputs
        assert!(input.len() == self.shape[0]);
        let vec = input.to_vec();
        let len = vec.len();

        let mut values = self
            .shape
            .iter()
            .map(|size| Array::zeros([1, *size]))
            .collect::<Vec<_>>();

        values[0] = Array::from(vec).into_shape([1, len]).unwrap();

        let layers = self.shape.len();
        for l in 0..layers - 1 {
            let vals = &values[l];
            let weights = &self.weights[l];
            let bias = &self.bias[l];

            let ltype = if l == layers - 2 {
                self.output_type
            } else {
                self.hidden_type
            };

            let mut sum = vals.dot(weights) + bias;

            //apply activation
            sum.mapv_inplace(|a| activate(a, ltype));

            values[l + 1] = sum;
        }

        values
    }

    ///We calc current error, and the error gradient for output layer
    fn output_error_and_gradient(outputs: &[f32], target: &[f32]) -> ErrorAndGradient {
        //E = error / loss
        //a = value after activation
        //t = target value

        // E = 0.5* (t-a)^2
        // error gradient: dE/da = -1*2*0.5*(t-a) = -(t-a)=a-t

        //we save the MSE which is the sum of the errors / N
        let mut errors = vec![];
        for (act, tar) in outputs.iter().zip(target) {
            errors.push(0.5 * (act - tar).powi(2));
        }

        //get error gradient
        let mut gradient = vec![];
        for (act, tar) in outputs.iter().zip(target) {
            let deda = act - tar; //dE/da = act - target; NOTE: this is the same for cross entropy with softmax
            gradient.push(deda);
        }

        ErrorAndGradient {
            error: errors.iter().sum::<f32>(), //sse
            gradient,
        }
    }

    ///Calculates the gradients for all layers
    /// Returns the weight and bias gradients
    fn backward(
        &self,
        values: Vec<Array2<f32>>,
        output_gradient: Vec<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let layers = self.shape.len();

        //calc error by looking at last layer and comparing target vs actual

        //E = error i.e. loss
        //t = target value
        //a = value after activation
        //z = value before activation
        //i = input
        //derAct = activation derivative
        //w = weight

        // E = 0.5* (t-a)^2
        // output error gradient: dE/da = -1*2*0.5*(t-a) = -(t-a)=a-t
        // dE/dw = dE/da * da/dz * dz/dw = (a-t) * derAct * di
        // dE/db = dE/da * da/dz * 1 = (a-t) * derAct * 1
        // grad : dE/da * da/dz * dz/di = (a-t) * derAct * w

        //with matrices, sometimes we use dot and sometimes we use a row multiply

        let mut weight_gradients = vec![];
        let mut bias_gradients = vec![];

        let last_size = *self.shape.last().unwrap();
        let mut next_layer_error_deriv = Array::from_iter(output_gradient)
            .into_shape([1, last_size])
            .unwrap();

        //backprop error
        for l in (0..layers - 1).rev() {
            let next_values = &values[l + 1];
            let this_values = &values[l];
            let this_weights = &self.weights[l];
            let ltype = if l == layers - 2 {
                self.output_type
            } else {
                self.hidden_type
            };

            let next_dadz = next_values.map(|&a| activate_der(a, ltype));

            let error_grad_bias = next_layer_error_deriv.clone()
                *&next_dadz//dE/dA * dA/dZ
                //*dZ/db (=1 so ignore)
              ;

            let error_grad_weights = next_layer_error_deriv.clone() //dE/dA
                *next_dadz // * dA/dz
                .t()//get correct dimensions
                .dot(this_values) // *dz/dw ( =weights)
                .t()//to get correct dimensions
                ;

            //now change layer error to current: from E to E * actder * w, to be passed down

            next_layer_error_deriv = next_layer_error_deriv.clone() * &next_dadz; //dA/dz

            next_layer_error_deriv = next_layer_error_deriv.dot(&this_weights.t());
            //dz/din

            weight_gradients.insert(0, error_grad_weights);
            bias_gradients.insert(0, error_grad_bias);
        }

        (weight_gradients, bias_gradients)
    }

    ///Apply gradients to network
    fn apply_gradients(
        &mut self,
        weight_gradients: &[Array2<f32>],
        bias_gradients: &[Array2<f32>],
    ) {
        let layers = self.shape.len();
        for l in 0..layers - 1 {
            self.bias[l] = &self.bias[l] - &bias_gradients[l] * self.learning_rate;
            self.weights[l] = &self.weights[l] - &weight_gradients[l] * self.learning_rate;
        }
    }

    pub fn error(&self) -> f32 {
        self.error
    }

    ///Save to path
    pub fn save<P: AsRef<Path>>(&self, path: P) {
        let mut vec = vec![];
        vec.push(format!("learning_rate={}", self.learning_rate));
        vec.push(format!("hidden_type={}", self.hidden_type));
        vec.push(format!("output_type={}", self.output_type));
        let layers = self.shape.len();
        for l in 0..layers - 1 {
            let bias = &self.bias[l]
                .as_slice()
                .unwrap()
                .iter()
                .map(|x| format!("{x}"))
                .collect::<Vec<_>>()
                .join(";");

            vec.push(format!("bias={bias}"));

            let weights = &self.weights[l]
                .to_string()
                .split('\n')
                .collect::<Vec<_>>()
                .join(";")
                .replace(['[', ']', ' '], "");

            vec.push(format!("weight={weights}"));

            std::fs::write(path.as_ref(), vec.join("\n")).unwrap();
        }
    }
    ///Load from file
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        let data: Vec<Vec<String>> = std::fs::read_to_string(path)
            .unwrap()
            .split('\n')
            .map(|x| {
                x.split('=')
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<String>>()
            })
            .collect();

        let mut lr = 0.5f32;
        let mut ht = ActivationType::Sigmoid;
        let mut ot = ActivationType::Relu;
        let mut weights: Vec<Array<f32, Ix2>> = vec![];
        let mut biases: Vec<Array<f32, Ix2>> = vec![];

        for line in data {
            if line[0] == "learning_rate" {
                lr = line[1].parse::<f32>().unwrap_or(0.01);
            } else if line[0] == "hidden_type" {
                ht = ActivationType::from_str(&line[1]).unwrap_or(ActivationType::Sigmoid);
            } else if line[0] == "output_type" {
                ot = ActivationType::from_str(&line[1]).unwrap_or(ActivationType::Linear);
            } else if line[0] == "weight" {
                let ww: Vec<Vec<f32>> = line[1]
                    .split(';')
                    .map(|x| {
                        x.split(',')
                            .filter_map(|f| f.parse::<f32>().ok())
                            .collect::<Vec<f32>>()
                    })
                    .collect();
                let r = dbg!(ww.len());
                let c = dbg!(ww[0].len());
                let ww = ww.iter().flatten().copied().collect::<Vec<f32>>();
                let ww = Array2::from_shape_vec([r, c], ww).unwrap();
                weights.push(ww);
            } else if line[0] == "bias" {
                let bb = line[1]
                    .split(';')
                    .map(|f| f.parse::<f32>().unwrap_or_default())
                    .collect::<Vec<f32>>();
                let bb = bb;
                let bb = Array2::from_shape_vec([1, bb.len()], bb).unwrap();

                biases.push(bb);
            }
        }

        let mut network_shape = vec![];

        for l in &weights {
            network_shape.push(l.shape()[0]);
        }
        network_shape.push(weights.last().unwrap().shape()[1]);

        let mut s = Self::new(network_shape.as_slice())
            .with_hidden_type(ht)
            .with_output_type(ot)
            .with_learning_rate(lr);

        s.weights = weights;
        s.bias = biases;

        s
    }

    ///Returns weights (including biases)
    pub fn get_weights(&self) -> Vec<f32> {
        let mut weights = vec![];

        for l in 0..self.weights.len() {
            for i in &self.weights[l] {
                weights.push(*i);
            }
            for i in &self.bias[l] {
                weights.push(*i);
            }
        }
        weights
    }

    ///Sets weights (including biases)
    pub fn set_weights(&mut self, weights: &[f32]) {
        let mut counter = 0;

        for l in 0..self.weights.len() {
            for i in self.weights[l].iter_mut() {
                *i = weights[counter];
                counter += 1;
            }
            for i in self.bias[l].iter_mut() {
                *i = weights[counter];
                counter += 1;
            }
        }
    }
    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

impl Display for NN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut str = String::new();

        str.push_str(
            self.shape
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<String>>()
                .join(", ")
                .as_str(),
        );

        str.push_str("\nweights: \n");
        let wei = &self.weights;
        for l in wei {
            str.push_str(l.to_string().as_str());
            str.push('\n');
        }
        str.push('\n');

        str.push_str("\nbiases: \n");
        let wei = &self.bias;
        for l in wei {
            str.push_str(l.to_string().as_str());
            str.push('\n');
        }
        str.push('\n');

        write!(f, "{str}")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use ndarray::arr2;
    use tempfile::NamedTempFile;

    use crate::activation::ActivationType;
    use crate::initialization::InitializationType;
    use crate::nn::NN;

    #[test]
    /// Test xor nn against known outputs
    fn test_xor() {
        //setup initial  weights
        let mut nn = NN::new(&[2, 2, 1])
            .with_learning_rate(0.5)
            .with_hidden_type(ActivationType::Sigmoid)
            .with_output_type(ActivationType::Sigmoid);

        nn.weights = [arr2(&[[0.15, 0.2], [0.25, 0.3]]), arr2(&[[0.4], [0.45]])].to_vec();
        nn.bias = [arr2(&[[0.35, 0.35]]), arr2(&[[0.6]])].to_vec();

        //check forward
        let vals = nn.forward(&[0., 0.]);
        assert_eq!(vals, [0.750002384]);

        //check first 4
        let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let outputs = [[0.], [1.], [1.], [0.]];

        let known_errors = [0.2812518, 0.034677744, 0.033219155, 0.28904837];
        let known_biases = [
            //1
            [arr2(&[[0.35, 0.35]]), arr2(&[[0.6]])].to_vec(),
            //2
            [arr2(&[[0.343179762, 0.342327237]]), arr2(&[[0.529687762]])].to_vec(),
            //3
            [arr2(&[[0.3452806, 0.34468588]]), arr2(&[[0.555233]])].to_vec(),
            //4
            [arr2(&[[0.3474572, 0.34712338]]), arr2(&[[0.5798897]])].to_vec(),
        ]
        .to_vec();

        for i in 0..4 {
            //check error
            assert_eq!(nn.bias, known_biases[i]);
            println!("{i}: biases ok");
            nn.fit_one(&inputs[i], &outputs[i]);
            assert_eq!(nn.error, known_errors[i]);
            println!("{i}: errors ok");
        }
    }

    #[test]
    fn xor_sgd() {
        fastrand::seed(1);
        let mut nn = crate::nn::NN::new(&[2, 8, 1])
            .with_learning_rate(0.2)
            .with_hidden_type(ActivationType::Sigmoid)
            .with_output_type(ActivationType::Sigmoid);

        let mut inp_out = [
            ([0., 0.], [0.1]),
            ([0., 1.], [1.]),
            ([1., 0.], [1.]),
            ([1., 1.], [0.]),
        ];

        let mut completed_steps_matrix = vec![];

        //run a few times, and get average completion epoch
        for _ in 0..5 {
            let mut results: VecDeque<f32> = VecDeque::new();

            results.clear();
            nn.reset_weights(InitializationType::Random);
            for steps in 0..20_000 {
                fastrand::shuffle(&mut inp_out);
                nn.fit_one(&inp_out[0].0, &inp_out[0].1);

                results.push_back(nn.error());
                if results.len() > 100 {
                    results.pop_front();
                    if results.iter().sum::<f32>() / 100. < 0.02 {
                        completed_steps_matrix.push(steps);
                        break;
                    }
                }
            }
        }

        let avg_matrix =
            completed_steps_matrix.iter().sum::<usize>() / completed_steps_matrix.len();

        println!("len:{} avg:{avg_matrix}", completed_steps_matrix.len());
        assert!(completed_steps_matrix.len() == 5);
        assert!(avg_matrix == 5535);
    }

    #[test]
    fn xor_gd() {
        //do mini batch gradient descent.
        //we take a few at a time
        fastrand::seed(1);
        let mut nn = crate::nn::NN::new(&[2, 8, 1])
            .with_learning_rate(0.8)
            .with_hidden_type(ActivationType::Sigmoid)
            .with_output_type(ActivationType::Sigmoid);

        let mut inp_out = [
            ([0f32, 0.], [0.1]),
            ([0., 1.], [1.]),
            ([1., 0.], [1.]),
            ([1., 1.], [0.]),
        ];

        let mut completed_steps_matrix = vec![];

        //run a few times, and get average completion epoch
        for _ in 0..5 {
            let mut results: VecDeque<f32> = VecDeque::new();

            results.clear();
            nn.reset_weights(InitializationType::Random);
            for steps in 0..20_000 {
                fastrand::shuffle(&mut inp_out);
                let ins = vec![&inp_out[0].0[..], &inp_out[1].0[..]];
                let outs = vec![&inp_out[0].1[..], &inp_out[1].1[..]];

                nn.fit(&ins, &outs);
                results.push_back(nn.error());
                if results.len() > 100 {
                    results.pop_front();
                    if results.iter().sum::<f32>() / 100. < 0.02 {
                        completed_steps_matrix.push(steps);
                        break;
                    }
                }
            }
        }

        let avg_matrix =
            completed_steps_matrix.iter().sum::<usize>() / completed_steps_matrix.len();

        println!("len:{} avg:{avg_matrix}", completed_steps_matrix.len());

        assert!(completed_steps_matrix.len() == 5);
        assert!(avg_matrix == 1674);
    }

    #[test]
    fn test_save_load() {
        let nn = NN::new(&[10, 10, 10])
            .with_learning_rate(0.5)
            .with_hidden_type(ActivationType::Sigmoid)
            .with_output_type(ActivationType::Sigmoid);

        let input = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let result1 = nn.forward(&input);
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();

        //test looks the same
        let orig = nn.to_string();
        println!("{orig}");
        nn.save(&path);
        let nn2 = NN::load(path);
        let new = nn2.to_string();
        println!("{new}");
        assert_eq!(orig, new);

        //test result the same
        let result2 = nn2.forward(&input);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_get_set_weights() {
        //the result before and after should be the same if set with the same weights
        let mut nn = NN::new(&[12, 12, 12]);

        let test = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let res1 = nn.forward(&test);
        let old = nn.get_weights();
        nn.set_weights(&old);

        let res2 = nn.forward(&test);
        println!("before {res1:?} after: {res2:?}");

        assert_eq!(res1, res2);
    }
}
