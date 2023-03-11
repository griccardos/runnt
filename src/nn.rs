use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use ndarray::prelude::*;
use ndarray::Array;

use crate::activation::{activate, activate_der, ActivationType};
use crate::dataset::Dataset;
use crate::initialization::{calc_initialization, InitializationType};
use crate::regularization::Regularization;

/// Struct holding Neural Net functionality
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
pub struct NN {
    weights: Vec<Array2<f32>>, // layer * 2D matrix e.g. [2x2], [2x1]
    bias: Vec<Array2<f32>>,    //layer * 2D matrix
    shape: Vec<usize>,
    pub learning_rate: f32,
    hidden_type: ActivationType,
    output_type: ActivationType,
    regularization: Regularization,
}

impl NN {
    /// Initialize a neural net with the shape , including input and output layer sizes.
    /// For network with sizes: input 7, hidden layer 1 of 10, hidden layer 2 of 20, output of 2
    /// we use: `&[7,10,20,2]`.
    ///
    /// Uses the defaults: <br/>
    ///  learning rate = 0.01<br/>
    ///  hidden layer type: Sigmoid  <br/>
    ///  output layer type: Linear  <br/>
    ///  weight initialization: Random  <br/>
    ///
    /// #Panics
    /// If network shape does not have at least 2 layers

    pub fn new(network_shape: &[usize]) -> NN {
        let mut weights = vec![];
        let mut bias = vec![];
        let mut values = vec![];

        let layers = network_shape.len();
        assert!(
            layers >= 2,
            "Network must have at least 2 layers: input and output"
        );

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
            hidden_type: ActivationType::Sigmoid,
            output_type: ActivationType::Linear,
            regularization: Regularization::None,
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

    pub fn with_regularization(mut self, reg: Regularization) -> Self {
        self.regularization = reg;
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
        self.fit(&[&input.to_vec()], &[&targets.to_vec()]);
    }

    /// Perform mini batch gradient descent on `batch_size`.
    /// If batch is smaller than data, will perform fit multiple times
    pub fn fit_batch_size(
        &mut self,
        inputs: &[&Vec<f32>],
        targets: &[&Vec<f32>],
        batch_size: usize,
    ) {
        //perform fit on chunks of batch size
        //collect errors
        inputs
            .chunks(batch_size)
            .zip(targets.chunks(batch_size))
            .for_each(|(inps, outs)| self.fit(inps, outs));
    }

    /// Gradient descent on entire batch <br>
    /// This processes every example, and adjusts gradients once
    /// 1. Forwards data
    /// 2. Calculates output loss
    /// 3. Backprop to calculate gradient
    /// 4. Regularize
    /// 5. Applies gradients to weights and biases
    pub fn fit(&mut self, inputs: &[&Vec<f32>], targets: &[&Vec<f32>]) {
        let inputs_matrix = self.to_matrix(inputs);
        let targets_matrix = self.to_matrix(targets);

        let values = self.internal_forward(&inputs_matrix);
        let outputs = values.last().expect("There should be outputs").clone();
        let loss = self.output_loss(&outputs, &targets_matrix);
        let (mut weights, biases) = self.backwards(values, loss);

        self.regularize(&mut weights);

        self.apply_gradients(&weights, &biases);
    }

    /// convert this into 2x2 Matrix
    /// Each column is an input into the neural network
    /// Each row is an example
    fn to_matrix(&self, vec: &[&Vec<f32>]) -> Array2<f32> {
        Array2::from_shape_vec(
            (vec.len(), vec[0].len()),
            vec.iter().flat_map(|x| x.to_owned()).copied().collect(),
        )
        .expect("shape not allowed by size of vec")
    }

    /// Forward inputs into the network, and returns output result i.e. `prediction`
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let inputs = self.to_matrix(&[&input.to_vec()]);
        let values = self.internal_forward(&inputs);
        values
            .last()
            .expect("There should be outputs")
            .clone()
            .into_raw_vec()
    }

    /// Forward inputs to get error
    ///
    pub fn forward_error(&self, input: &[f32], target: &[f32]) -> f32 {
        self.calc_error(&self.forward(input), target)
    }

    /// Forward batch, and get mean error
    pub fn forward_errors(&self, inputs: &[&Vec<f32>], targets: &[&Vec<f32>]) -> f32 {
        let inputs = self.to_matrix(inputs);
        let targets = self.to_matrix(targets);
        self.internal_forward_errors(&inputs, &targets)
    }

    fn internal_forward_errors(&self, inputs: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let count = inputs.shape()[0];
        let vals = self.internal_forward(inputs);
        let outs = vals.last().expect("There should be outputs");
        let errs = self.internal_calc_errors(outs, targets);
        errs.iter().sum::<f32>() / count as f32
    }

    ///calcs error based on outputs and target
    pub fn calc_error(&self, outputs: &[f32], target: &[f32]) -> f32 {
        let outputs =
            Array2::from_shape_vec((1, outputs.len()), outputs.to_vec()).expect("Shape is correct");
        let target =
            Array2::from_shape_vec((1, target.len()), target.to_vec()).expect("Shape is correct");

        self.internal_calc_errors(&outputs, &target)[0]
    }

    ///calcs error based on outputs and target
    fn internal_calc_errors(&self, outputs: &Array2<f32>, target: &Array2<f32>) -> Vec<f32> {
        //E = error or loss
        //a = forwarded value after activation
        //t = target value
        // E = 0.5* (t-a)^2

        let mut diff = target - outputs;
        diff.mapv_inplace(|x| 0.5 * x.powi(2));
        //now have a matrix of [example_count x output_size]
        //we want sum of all output size so we get [example_count x 1]

        let sums = diff.sum_axis(Axis(1));
        sums.to_vec()
    }

    /// calc gradient for each example
    fn output_loss(&self, outputs: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
        //E = error / loss
        //a = value after activation
        //t = target value

        // E = 0.5* (t-a)^2
        // error gradient: dE/da = -1*2*0.5*(t-a) = -(t-a)=a-t

        //get error gradient
        outputs - target
    }

    /// convert to array, and forward, returning batch values for each layer
    /// each input example is in a different row
    /// each value is in a different column
    /// so for 32 examples in a 10 node layer we have 32x10 matrix
    fn internal_forward(&self, input: &Array2<f32>) -> Vec<Array2<f32>> {
        assert_eq!(
            input.shape()[1],
            self.shape[0],
            "Input size does not equal first layer size"
        ); //columns = number of nodes in first layer

        let example_count = input.shape()[0];

        let mut values = self
            .shape
            .iter()
            .map(|size| Array::zeros([example_count, *size]))
            .collect::<Vec<_>>();

        values[0] = input.clone();

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

            //here bias only is 1xsize, but ndarray adds to each row example
            let mut sum = vals.dot(weights) + bias;

            //apply activation
            sum.mapv_inplace(|a| activate(a, ltype));

            values[l + 1] = sum;
        }

        values
    }

    /// Calculates the gradients for all layers.
    /// Returns the weight and bias gradients for each layer
    fn backwards(
        &self,
        values: Vec<Array2<f32>>,
        output_gradient: Array2<f32>,
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
        // partial gradient : dE/dz = dE/dA * dA/dZ
        // weight gradient = dE/dw = dE/da * da/dz * dz/dw = (a-t) * derAct * di
        // bias gradient = dE/db = dE/da * da/dz * 1 = (a-t) * derAct * 1
        // error gradient : dE/da * da/dz * dz/di = (a-t) * derAct * w

        let mut weight_gradients = vec![];
        let mut bias_gradients = vec![];
        let example_count = output_gradient.shape()[0];

        let mut next_layer_error_deriv = output_gradient; //output error gradient dE/dA

        //backprop to each layer
        for l in (0..layers - 1).rev() {
            let next_values = &values[l + 1]; //A
            let this_values = &values[l]; //In
            let this_weights = &self.weights[l]; //W
            let ltype = if l == layers - 2 {
                self.output_type
            } else {
                self.hidden_type
            };

            let da_dz = next_values.map(|&a| activate_der(a, ltype)); //dA/dZ

            let de_dz = &next_layer_error_deriv * &da_dz; //dE/dA * dA/dZ = dE/dZ

            //here error_grad_bias is numberexamples x size. But we only keep the sum
            let ones: Array2<f32> = Array2::ones((1, example_count));
            let error_grad_bias = ones.dot(&de_dz); //dE/dZ * 1

            let error_grad_weights = this_values.t().dot(&de_dz); //de/dZ * dZ/dw (inputs)

            //now change layer error to current: from E to E * actder * w, to be passed down
            next_layer_error_deriv = de_dz.dot(&this_weights.t()); //dE/dA = dE/dZ * dz/din

            weight_gradients.insert(0, error_grad_weights);
            bias_gradients.insert(0, error_grad_bias.to_owned());
        }

        //we have sum, now get average of error gradient
        for layer in &mut weight_gradients {
            layer.mapv_inplace(|a| a / example_count as f32);
        }
        for layer in &mut bias_gradients {
            layer.mapv_inplace(|a| a / example_count as f32);
        }

        (weight_gradients, bias_gradients)
    }

    ///Apply regularization to gradients
    fn regularize(&self, weight_gradients: &mut [Array2<f32>]) {
        match self.regularization {
            Regularization::None => {}
            Regularization::L1(lambda) => {
                // L1 =  0.5*|w|*lambda
                //thus dL1/dw = sign(w) * lambda
                //we adjust current gradients effectively reducing value of weight
                for (wlayer, gradlayer) in self.weights.iter().zip(weight_gradients.iter_mut()) {
                    *gradlayer += &(wlayer.mapv(f32::signum) * lambda);
                }
            }
            Regularization::L2(lambda) => {
                // L2 =  0.5*w^2*lambda
                //thus dL2/dw = w * lambda
                //we adjust current gradients effectively reducing value of weight
                for (wlayer, gradlayer) in self.weights.iter().zip(weight_gradients.iter_mut()) {
                    *gradlayer += &(wlayer * lambda);
                }
            }
            Regularization::L1L2(l1lambda, l2lambda) => {
                for (wlayer, gradlayer) in self.weights.iter().zip(weight_gradients.iter_mut()) {
                    let l1: Array2<f32> = wlayer.mapv(f32::signum) * l1lambda;
                    let l2: Array2<f32> = wlayer * l2lambda;
                    *gradlayer += &(l1 + l2);
                }
            }
        }
    }

    ///Apply gradients to network using learning rate
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

    ///Save to path
    pub fn save<P: AsRef<Path>>(&self, path: P) {
        let mut vec = vec![];
        vec.push(format!("learning_rate={}", self.learning_rate));
        vec.push(format!("hidden_type={}", self.hidden_type));
        vec.push(format!("output_type={}", self.output_type));
        vec.push(format!("regularization={}", self.regularization));
        vec.push(format!(
            "shape={}",
            self.shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ));
        let layers = self.shape.len();
        for l in 0..layers - 1 {
            let bias = &self.bias[l]
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(";");

            vec.push(format!("bias={bias}"));

            let weights = &self.weights[l]
                .rows()
                .into_iter()
                .map(|x| {
                    x.iter()
                        .map(ToString::to_string)
                        .collect::<Vec<String>>()
                        .join(",")
                })
                .collect::<Vec<_>>()
                .join(";");

            vec.push(format!("weight={weights}"));

            let result = std::fs::write(path.as_ref(), vec.join("\n"));
            if let Err(err) = result {
                println!(
                    "Could not write file {:?}: {}",
                    path.as_ref().as_os_str(),
                    err
                );
            }
        }
    }
    ///Load from file
    /// # Panics
    /// If cannot load file
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        let data: Vec<Vec<String>> = std::fs::read_to_string(path)
            .expect("Could not load from file")
            .split('\n')
            .map(|x| {
                x.split('=')
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<String>>()
            })
            .collect();

        let mut lr = 0.01f32;
        let mut ht = ActivationType::Sigmoid;
        let mut ot = ActivationType::Linear;
        let mut reg = Regularization::None;
        let mut weights: Vec<Array<f32, Ix2>> = vec![];
        let mut biases: Vec<Array<f32, Ix2>> = vec![];
        let mut network_shape = vec![];
        for line in data {
            if line[0] == "learning_rate" {
                lr = line[1].parse::<f32>().unwrap_or(0.01);
            } else if line[0] == "hidden_type" {
                ht = ActivationType::from_str(&line[1]).unwrap_or(ActivationType::Sigmoid);
            } else if line[0] == "output_type" {
                ot = ActivationType::from_str(&line[1]).unwrap_or(ActivationType::Linear);
            } else if line[0] == "regularization" {
                reg = Regularization::from_str(&line[1]);
            } else if line[0] == "weight" {
                let ww: Vec<Vec<f32>> = line[1]
                    .split(';')
                    .map(|x| {
                        x.split(',')
                            .filter_map(|f| f.parse::<f32>().ok())
                            .collect::<Vec<f32>>()
                    })
                    .collect();
                let r = ww.len();
                let c = ww[0].len();
                let ww = ww.iter().flatten().copied().collect::<Vec<f32>>();
                let ww = Array2::from_shape_vec([r, c], ww).expect("Shape is wrong for vec");
                weights.push(ww);
            } else if line[0] == "bias" {
                let bb = line[1]
                    .split(';')
                    .map(|f| f.parse::<f32>().unwrap_or_default())
                    .collect::<Vec<f32>>();
                let bb = bb;
                let bb = Array2::from_shape_vec([1, bb.len()], bb).expect("Shape is wrong for vec");

                biases.push(bb);
            } else if line[0] == "shape" {
                network_shape = line[1]
                    .split(',')
                    .map(|f| f.parse::<usize>().unwrap_or_default())
                    .collect::<Vec<usize>>();
            }
        }

        let mut weight_shape = vec![];
        for l in &weights {
            weight_shape.push(l.shape()[0]);
        }
        weight_shape.push(weights.last().expect("There should be weights").shape()[1]);

        assert_eq!(
            network_shape, weight_shape,
            "Weight shape does not equal to shape"
        );

        let mut s = Self::new(&network_shape)
            .with_hidden_type(ht)
            .with_output_type(ot)
            .with_regularization(reg)
            .with_learning_rate(lr);

        s.weights = weights;
        s.bias = biases;

        s
    }

    ///Returns weights (including biases)
    pub fn get_weights(&self) -> Vec<f32> {
        let mut weights = vec![];

        //for each layer...
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
    /// Uses output of `get_weights`
    /// Format: layer1weights,layer1biases,layer2weights,layer2biases etc...
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

//TOOLS which help

/// This checks the index of the maximum value in each vec are equal
/// Used to compare one hot encoding predicted with actual
/// Typically the actual will be 1 for a value and zero for else,
/// whereas the predicted may not be exactly one
/// So instead we compare the index of the maximum value, to determine equality
/// # Panics
/// If target and predicted lengths differ
pub fn max_index_equal(target: &[f32], predicted: &[f32]) -> bool {
    assert_eq!(
        target.len(),
        predicted.len(),
        "Target and predicted should have same length"
    );
    let pred = max_index(predicted);
    let tar = max_index(target);

    pred == tar
}

/// Returns the index of the maximum value, panics if empty
pub fn max_index(vec: &[f32]) -> usize {
    vec.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("max_index expects a vec with non zero size")
        .0
}

/// Basic Runner that loops through:
/// 1. getting shuffled data and test data from Dataset
/// 2. fitting data with batch size
/// 3. reporting train and test mse
/// 4. (optionally) report accuracy (assumes target and output is one hot encoded, and chooses highest value)
/// Currently a convenience function. May be removed in the future.
pub fn run_and_report(
    set: &Dataset,
    net: &mut NN,
    epochs: usize,
    batch_size: usize,
    report_epoch: Option<usize>,
    report_accuracy: bool,
) {
    let start = Instant::now();
    let acc = if report_accuracy {
        " train_acc test_acc"
    } else {
        ""
    };
    println!("epoch train_mse test_mse{acc} duration(s)");
    for e in 1..=epochs {
        let (inp, tar) = set.get_data();
        net.fit_batch_size(&inp, &tar, batch_size);
        if let Some(re) = report_epoch {
            if re > 0 && e % re == 0 {
                //get mse
                let train_err = net.forward_errors(&inp, &tar);
                let (inp_test, tar_test) = set.get_test_data();
                let test_err = net.forward_errors(&inp_test, &tar_test);

                //get accuracy
                let mut acc = "".to_string();
                if report_accuracy {
                    let mut train_count = 0;
                    for (inp, tar) in inp.iter().zip(tar) {
                        let pred = net.forward(inp);
                        if max_index_equal(tar, &pred) {
                            train_count += 1;
                        }
                    }

                    let mut test_count = 0;
                    for (inp, tar) in inp_test.iter().zip(tar_test) {
                        let pred = net.forward(inp);
                        if max_index_equal(tar, &pred) {
                            test_count += 1;
                        }
                    }
                    let train_acc = train_count as f32 / inp.len() as f32 * 100.;
                    let test_acc = test_count as f32 / inp_test.len() as f32 * 100.;
                    acc = format!(" {train_acc:.1}% {test_acc:.1}%");
                }

                println!(
                    "{e} {train_err} {test_err}{acc} {:.1}",
                    start.elapsed().as_secs_f32()
                );
            }
        }
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
    use crate::regularization::Regularization;

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
            let error = nn.calc_error(&nn.forward(&inputs[i]), &outputs[i]);
            nn.fit_one(&inputs[i], &outputs[i]);
            assert_eq!(error, known_errors[i]);
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

        let mut completed_steps = vec![];

        //run a few times, and get average completion epoch
        for _ in 0..5 {
            let mut results: VecDeque<f32> = VecDeque::new();

            results.clear();
            nn.reset_weights(InitializationType::Random);
            for steps in 0..20_000 {
                fastrand::shuffle(&mut inp_out);
                nn.fit_one(&inp_out[0].0, &inp_out[0].1);
                let err = nn.forward_error(&inp_out[0].0, &inp_out[0].1);

                results.push_back(err);
                if results.len() > 100 {
                    results.pop_front();
                    if results.iter().sum::<f32>() / 100. < 0.02 {
                        completed_steps.push(steps);
                        break;
                    }
                }
            }
        }

        let avg = completed_steps.iter().sum::<usize>() / completed_steps.len();

        println!("len:{} avg:{avg}", completed_steps.len());
        assert!(completed_steps.len() == 5);
        assert!(avg < 8000);
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
            (vec![0f32, 0.], vec![0.1]),
            (vec![0., 1.], vec![1.]),
            (vec![1., 0.], vec![1.]),
            (vec![1., 1.], vec![0.]),
        ];

        let mut completed_steps = vec![];

        //run a few times, and get average completion epoch
        for _ in 0..5 {
            let mut results: VecDeque<f32> = VecDeque::new();

            results.clear();
            nn.reset_weights(InitializationType::Random);
            for steps in 0..20_000 {
                fastrand::shuffle(&mut inp_out);
                let ins = vec![&inp_out[0].0, &inp_out[1].0];
                let outs = vec![&inp_out[0].1, &inp_out[1].1];

                nn.fit(&ins, &outs);

                let err: f32 = nn.forward_errors(&ins, &outs);

                results.push_back(err / ins.len() as f32);
                if results.len() > 100 {
                    results.pop_front();
                    if results.iter().sum::<f32>() / 100. < 0.02 {
                        completed_steps.push(steps);
                        break;
                    }
                }
            }
        }

        let avg = completed_steps.iter().sum::<usize>() / completed_steps.len();

        println!("len:{} avg:{avg}", completed_steps.len());

        assert!(completed_steps.len() == 5);
        assert!(avg < 2000);
    }
    #[test]
    ///use this in documentation
    fn readme() {
        let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let outputs = [[0.], [1.], [1.], [0.]];

        let mut nn = NN::new(&[2, 8, 1])
            .with_learning_rate(0.2)
            .with_hidden_type(ActivationType::Tanh)
            .with_output_type(ActivationType::Linear);

        for i in 0..5000 {
            nn.fit_one(&inputs[i % 4], &outputs[i % 4]);
        }
        assert_eq!(nn.forward(&[0., 0.]).first().unwrap().round(), 0.);
        assert_eq!(nn.forward(&[0., 1.]).first().unwrap().round(), 1.);
        assert_eq!(nn.forward(&[1., 0.]).first().unwrap().round(), 1.);
        assert_eq!(nn.forward(&[1., 1.]).first().unwrap().round(), 0.);
    }
    #[test]
    fn test_save_load() {
        let nn = NN::new(&[10, 100, 10])
            .with_learning_rate(0.5)
            .with_regularization(Regularization::L1L2(0.1, 0.2))
            .with_hidden_type(ActivationType::Tanh)
            .with_output_type(ActivationType::Relu);

        let input = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let result1 = nn.forward(&input);
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();

        //test looks the same
        let orig = nn.get_weights();
        let orig_shape = nn.get_shape();
        println!("shape:{orig_shape:?} weights:{orig:?}");
        nn.save(&path);
        let nn2 = NN::load(path);
        let new = nn2.get_weights();
        let new_shape = nn2.get_shape();
        println!("shape:{new_shape:?} weights:{new:?}");
        assert_eq!(orig_shape, new_shape);
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
