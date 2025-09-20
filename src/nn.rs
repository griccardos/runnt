use ndarray::prelude::*;
use std::fmt::Display;
use std::path::Path;
use std::time::Instant;

use crate::activation::{Activation, activate, activate_der};
use crate::dataset::Dataset;
use crate::dropout::Dropout;
use crate::error::Error;
use crate::initialization::Initialization;
use crate::layer::{Dense, DenseBuilder};
use crate::learning::{LearningRate, Rate};
use crate::loss::Loss;
use crate::optimizer::{Optimizer, OptimizerInternal};
use crate::regularization::Regularization;
use crate::sede::Sede;

/// Struct holding Neural Net functionality
///```rust
///   //XOR
///    use runnt::{nn::NN,activation::Activation};
///    let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
///    let outputs = [[0.], [1.], [1.], [0.]];
///
///        let mut nn = NN::new(&[2, 8, 1])
///        .with_learning_rate(0.25)
///        .with_activation_hidden(Activation::Tanh)
///        .with_activation_output(Activation::Sigmoid);
///        //OR

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
    pub(crate) layers: Vec<Dense>,
    pub(crate) learning_rate: LearningRate,
    pub(crate) loss: Loss,
    pub(crate) optimizer: OptimizerInternal,
    pub(crate) label_smoothing: Option<f32>,
    //only for checking on adding first layer
    pub(crate) input: usize,
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
    ///  weight initialization: Xavier  <br/>
    ///
    /// # Panics
    /// If network shape does not have at least 2 layers (one input and one output)
    pub fn new(network_shape: &[usize]) -> NN {
        let layer_count = network_shape.len();
        assert!(
            layer_count >= 2,
            "Network must have at least 2 layers: input and output"
        );

        let mut b = NN::new_input(network_shape[0]);

        for i in 1..layer_count {
            b = b.layer(network_shape[i]);
        }
        b
    }

    pub fn new_input(input: usize) -> NN {
        NN {
            layers: vec![],
            learning_rate: 0.01.into(),
            loss: Loss::MSE,
            optimizer: Optimizer::sgd().into(),
            label_smoothing: None,
            input,
        }
    }

    pub fn layer(mut self, dense: impl Into<DenseBuilder>) -> NN {
        let input = if self.layers.is_empty() {
            self.input
        } else {
            self.layers.last().unwrap().weights.shape()[1]
        };
        let dense: DenseBuilder = dense.into();
        self.layers.push(dense.build(input));
        self
    }

    /// The learning rate to use for training.
    ///
    /// For constant: `.with_learning_rate(0.01)`<br/>
    /// For cosine decay with warmup: `.with_learning_rate(Rate::Cosine { start_rate: 0.0, warmup_target_rate: 0.1, warmup_steps: 100, total_steps: 10000, min_rate: 0.001 })`
    pub fn with_learning_rate(mut self, rate: impl Into<LearningRate>) -> NN {
        self.learning_rate = rate.into();
        self
    }
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> NN {
        self.optimizer = OptimizerInternal::from(optimizer);
        self
    }

    /// Set the loss function.
    ///
    /// Typically it will be `MSE` for regression problems,
    /// `SoftmaxAndCrossEntropy` for multi-class classification problems,
    /// `BinaryCrossEntropy` for binary classification problems.
    ///
    /// Overrides `with_output_type` for both CrossEntropy types
    pub fn with_loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }
    /// Smooths labels for one hot encoded values.
    ///
    /// Typically One hot has multiple outputs with one set to 1 and rest to zero.
    /// Smoothing decreases the 1 and allocates to the other labels, "smoothing" the targets  
    /// e.g. `[1,0,0]` becomes `[0.8,0.1,0.1]`.
    /// This helps avoiding huge gradient spikes.  
    ///
    /// A good default is 0.1
    pub fn with_label_smoothing(mut self, rate: f32) -> Self {
        assert!(
            rate >= 0. && rate < 1.,
            "Label smoothing rate must be in range [0,1)"
        );
        self.label_smoothing = Some(rate);
        self
    }

    pub fn shape(&self) -> Vec<usize> {
        let mut shape = vec![self.layers[0].weights.shape()[0]]; //input layer size
        for lay in &self.layers {
            shape.push(lay.weights.shape()[1]); //next layer size
        }
        shape
    }

    ///Also known as Stochastic Gradient Descent i.e. Gradient descent with batch size = 1
    pub fn fit_one(&mut self, input: &[f32], targets: &[f32]) {
        self.fit_batch(&[&input.to_vec()], &[&targets.to_vec()]);
    }

    /// Perform multiple mini batch gradient descents on `batch_size`.  
    ///
    /// If `batch_size` is smaller than data, will perform fit multiple times
    ///
    /// If `batch_size` == number of examples, will do one gradient descent. (same as `fit_batch`)
    ///
    /// If `batch_size` == 1, will perform gradient descent for each example. (same as `fit_one`)
    ///
    /// For example if data has 100 examples and batch size is 20, it will adjust gradients 5 times
    pub fn fit(&mut self, inputs: &[&Vec<f32>], targets: &[&Vec<f32>], batch_size: usize) {
        //perform fit on chunks of batch size
        inputs
            .chunks(batch_size)
            .zip(targets.chunks(batch_size))
            .for_each(|(inps, outs)| self.fit_batch(inps, outs));
    }

    /// Gradient descent on entire batch <br>
    /// This processes every example, and adjusts gradients once
    /// 1. Forwards data
    /// 2. Calculates output loss
    /// 3. Backprop to calculate gradient
    /// 4. Regularize
    /// 5. Applies gradients to weights and biases using specified optimizer or learning rate if none
    /// # Panics
    /// If target's length does not match that of the network
    pub fn fit_batch(&mut self, inputs: &[&Vec<f32>], targets: &[&Vec<f32>]) {
        assert_eq!(
            &targets[0].len(),
            self.shape().last().unwrap(),
            "Target size does not match network output size"
        );

        //smoothing labels if set
        let new_targets = match self.label_smoothing {
            Some(rate) => smooth_labels(targets, rate),
            None => targets.iter().map(|a| a.to_vec()).collect(),
        };
        let targets = &new_targets.iter().map(|a| a).collect::<Vec<&Vec<f32>>>();

        let inputs_matrix = to_matrix(inputs);
        let targets_matrix = to_matrix(targets);
        self.update_dropout();
        self.learning_rate.step();
        let values = self.internal_forward(&inputs_matrix, true);
        let outputs = values.activated.last().expect("There should be outputs");

        let loss = self.loss.gradient(outputs, &targets_matrix);
        let (mut weight_gradient, bias_gradient) = self.backwards(values, loss);
        self.regularize(&mut weight_gradient);
        self.apply_gradients(weight_gradient, bias_gradient);
    }

    fn update_dropout(&mut self) {
        self.layers.iter_mut().for_each(|l| {
            if let Some(dropout) = &mut l.dropout {
                dropout.recalc();
            }
        });
    }

    /// Forward inputs into the network, and returns output result i.e. `prediction`
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let inputs = to_matrix(&[&input.to_vec()]);
        let values = self.internal_forward(&inputs, false);
        let vals = values.activated.last().expect("There should be outputs");
        //now return the values as a vec
        vals.flatten().to_vec()
    }

    /// Forward inputs to get error
    ///
    pub fn forward_error(&self, input: &[f32], target: &[f32]) -> f32 {
        self.calc_error(&self.forward(input), target)
    }

    /// Forward batch, and get mean error
    /// # Panics
    /// if inputs or targets are  empty
    pub fn forward_errors(&self, inputs: &[&Vec<f32>], targets: &[&Vec<f32>]) -> f32 {
        assert!(!inputs.is_empty(), "No inputs");
        assert!(!targets.is_empty(), "No targets");
        let inputs = to_matrix(inputs);
        let targets = to_matrix(targets);
        self.internal_forward_errors(&inputs, &targets)
    }

    fn internal_forward_errors(&self, inputs: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let count = inputs.shape()[0];
        let vals = self.internal_forward(inputs, false);
        let outs = vals.activated.last().expect("There should be outputs");
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
        let eps = 1e-7f32; //prevents log(0)

        match self.loss {
            Loss::MSE => {
                //MSE
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
            Loss::SoftmaxAndCrossEntropy => {
                //crossentopy
                // E= - Sum(Ti Log(Si))
                //Ti target value
                //Si softmax value
                let mut err = outputs.mapv(|a| (a + eps).ln());
                err *= -1.;
                err *= target;
                let sums = err.sum_axis(Axis(1));
                sums.to_vec()
            }
            Loss::BinaryCrossEntropy => {
                // Binary cross-entropy: -Σ [ t*ln(a) + (1-t)*ln(1-a) ]
                // where:
                // - t is the target 0 or 1 for pure binary
                // - a is the predicted probability after sigmoid.
                let mut a = outputs.clone();
                // clamp to [eps, 1-eps]
                a.mapv_inplace(|v| v.max(eps).min(1.0 - eps));
                let term1 = target * &a.mapv(|x| x.ln());
                let term2 = (1.0 - target) * &a.mapv(|x| (1.0 - x).ln());
                let sum = -(term1 + term2);
                let sums = sum.sum_axis(Axis(1));
                sums.to_vec()
            }
        }
    }

    /// Forward inputs into the network, returning both values before activation and after activation
    /// each input example is in a different row
    /// each input value is in a different column
    /// so for 32 examples in a 10 node layer we have 32x10 matrix
    /// if softmax, we convert last layer
    fn internal_forward(&self, input: &Array2<f32>, is_training: bool) -> ValueAndActivated {
        assert_eq!(
            input.shape()[1],
            self.shape()[0],
            "Input size does not equal first layer size"
        ); //columns = number of nodes in first layer

        let mut activated = Vec::new();
        let mut values = Vec::new();

        activated.push(input.clone());
        values.push(input.clone());

        for l in 0..self.layers.len() {
            let mut act_sum = &activated[l].dot(&self.layers[l].weights) + &self.layers[l].bias;
            let sum = act_sum.clone();

            //apply activation
            let ltype = self.get_layer_type(l);

            let is_last_layer = l == self.layers.len() - 1;

            if !is_last_layer {
                //hidden layers
                act_sum.mapv_inplace(|a| activate(a, ltype));
                //apply dropout if set
                if is_training && let Some(dropout) = &self.layers[l].dropout {
                    act_sum *= &dropout.mask();
                }
            } else {
                match self.loss {
                    Loss::MSE => {
                        act_sum.mapv_inplace(|a| activate(a, ltype));
                    }
                    Loss::SoftmaxAndCrossEntropy => {
                        //avoid overflow by subtracting the max from each row
                        for mut r in act_sum.rows_mut() {
                            let max = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max); //find max
                            r.mapv_inplace(|a| a - max);
                        }
                        act_sum.mapv_inplace(f32::exp); //calc e^val
                        let sums = act_sum.sum_axis(Axis(1)); //sum all rows
                        for (ri, mut r) in act_sum.rows_mut().into_iter().enumerate() {
                            r.mapv_inplace(|a| a / sums[ri]);
                        }
                    }
                    Loss::BinaryCrossEntropy => {
                        //apply sigmoid
                        act_sum.mapv_inplace(|a| activate(a, Activation::Sigmoid));
                    }
                }
            }

            activated.push(act_sum);
            values.push(sum);
        }

        ValueAndActivated { values, activated }
    }

    fn get_layer_type(&self, layer_number: usize) -> Activation {
        let is_last_layer = layer_number == self.layers.len() - 1;
        if !is_last_layer {
            return self.layers[layer_number].activation;
        }
        match self.loss {
            Loss::MSE => self.layers[layer_number].activation,
            // Last layer: choose linear for these because for backward we don't want to activation derivative. We manually calculate the correct activation in forward
            Loss::SoftmaxAndCrossEntropy | Loss::BinaryCrossEntropy => Activation::Linear,
        }
    }

    /// Calculates the gradients for all layers.
    /// Returns the weight and bias gradients for each layer
    fn backwards(
        &self,
        values_and_activations: ValueAndActivated,
        output_gradient: Array2<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let layers = self.shape().len();

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
            let next_activations = &values_and_activations.activated[l + 1]; //A
            let this_values = &values_and_activations.activated[l]; //In
            let this_weights = &self.layers[l].weights; //W
            let next_values = &values_and_activations.values[l + 1]; //Z - needed for derivative sometimes
            let ltype = self.get_layer_type(l);

            //we calc derivative of activation function for each example in the batch
            let example_count = next_activations.shape()[0];
            let number_count = next_activations.shape()[1];
            let mut da_dz: Array2<f32> = next_activations.clone();

            let dropout_mask = self.layers[l].dropout.as_ref().map(|a| a.mask().clone());
            for e in 0..example_count {
                for n in 0..number_count {
                    if dropout_mask.is_none() {
                        //use already calculated activations (speeds up because we dont recalc activation)
                        da_dz[[e, n]] =
                            activate_der(next_values[(e, n)], next_activations[(e, n)], ltype);
                    } else {
                        //we need to recalculate activation from Z otherwise we will be using the scales/dropped out values which is incorrect
                        let unmasked_activation = activate(next_values[(e, n)], ltype);
                        da_dz[[e, n]] =
                            activate_der(next_values[(e, n)], unmasked_activation, ltype);
                    }
                }
            }
            //dA/dZ
            let mut de_dz = &next_layer_error_deriv * &da_dz; //dE/dA * dA/dZ = dE/dZ
            //apply dropout if set
            if let Some(mask) = dropout_mask {
                if l < mask.len() {
                    de_dz *= &mask;
                }
            }

            //here error_grad_bias is numberexamples x size. But we only keep the sum so make 1xsize
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
        for (gradlayer, layer) in weight_gradients.iter_mut().zip(&self.layers) {
            let wlayer = &layer.weights;
            match layer.regularization {
                Regularization::None => {}
                Regularization::L1(lambda) => {
                    // L1 =  0.5*|w|*lambda
                    //thus dL1/dw = sign(w) * lambda
                    //we adjust current gradients effectively reducing value of weight
                    *gradlayer += &(wlayer.mapv(f32::signum) * lambda);
                }
                Regularization::L2(lambda) => {
                    // L2 =  0.5*w^2*lambda
                    //thus dL2/dw = w * lambda
                    //we adjust current gradients effectively reducing value of weight
                    *gradlayer += &(wlayer * lambda);
                }
                Regularization::L1L2(l1lambda, l2lambda) => {
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
        weight_gradients: Vec<Array2<f32>>,
        bias_gradients: Vec<Array2<f32>>,
    ) {
        let (dw, db) = self.optimizer.calc_gradient_update(
            weight_gradients,
            bias_gradients,
            self.learning_rate.get(),
        );

        let layers = self.shape().len();
        for l in 0..layers - 1 {
            self.layers[l].bias = &self.layers[l].bias + &db[l];
            self.layers[l].weights = &self.layers[l].weights + &dw[l];
        }
    }

    ///Save to path
    pub fn save<P: AsRef<Path>>(&self, path: P) {
        let str = self.serialize();
        let result = std::fs::write(path.as_ref(), str);
        if let Err(err) = result {
            println!(
                "Could not write file {:?}: {}",
                path.as_ref().as_os_str(),
                err
            );
        }
    }
    ///Load from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let data = std::fs::read_to_string(path)?;
        Self::deserialize(&data)
    }
    ///Load from string
    pub fn load_str(str: &str) -> Result<Self, Error> {
        Self::deserialize(str)
    }

    ///Returns weights
    pub fn get_weights(&self) -> Vec<f32> {
        let mut weights = vec![];

        //for each layer...
        for l in 0..self.layers.len() {
            for i in &self.layers[l].weights {
                weights.push(*i);
            }
        }
        weights
    }

    ///Returns biases
    pub fn get_biases(&self) -> Vec<f32> {
        let mut biases = vec![];

        //for each layer...
        for l in 0..self.layers.len() {
            for i in &self.layers[l].bias {
                biases.push(*i);
            }
        }
        biases
    }

    ///Sets weights
    /// Uses output of `get_weights`
    /// Format: layer1weights,layer2weights etc...
    pub fn set_weights(&mut self, weights: &[f32]) {
        assert!(
            weights.len() == self.layers.iter().map(|a| a.weights.len()).sum::<usize>(),
            "Weights length does not match network size"
        );
        let mut counter = 0;

        for l in 0..self.layers.len() {
            for i in self.layers[l].weights.iter_mut() {
                *i = weights[counter];
                counter += 1;
            }
        }
    }

    ///Sets biases
    /// Uses output of `get_biases`
    /// Format: layer1biases,layer2biases etc...
    pub fn set_biases(&mut self, biases: &[f32]) {
        let mut counter = 0;

        for l in 0..self.layers.len() {
            for i in self.layers[l].bias.iter_mut() {
                *i = biases[counter];
                counter += 1;
            }
        }
    }

    /// Train with dataset
    /// This is the same as running `fit` for a number of epochs
    /// and reporting on each epoch
    /// 1. getting shuffled data and test data from Dataset
    /// 2. fitting data with batch size
    /// 3. reporting train and test error each N epochs. 0 for no reporting
    /// 4. Report metric applied to both train and test data
    pub fn train(
        &mut self,
        set: &Dataset,
        epochs: usize,
        batch_size: usize,
        report_epoch: usize,
        metric: ReportMetric,
    ) {
        let start = Instant::now();
        let acc = match metric {
            ReportMetric::None => "",
            ReportMetric::CorrectClassification | ReportMetric::Custom(_) => " train_acc test_acc",
            ReportMetric::RSquared => " train_R² test R²",
        };

        if report_epoch > 0 {
            println!("epoch train_error test_error{acc} duration(s)");
        }
        for e in 1..=epochs {
            let (inp, tar) = set.get_data();
            self.fit(&inp, &tar, batch_size);

            if report_epoch > 0 && (e % report_epoch == 0 || e == epochs) {
                //get error
                let train_err = self.forward_errors(&inp, &tar);
                let (inp_test, tar_test) = set.get_test_data();
                let test_err = if inp_test.is_empty() {
                    0.
                } else {
                    self.forward_errors(&inp_test, &tar_test)
                };

                //get accuracy
                let train_acc = self.report(&metric, &inp, &tar);
                let test_acc = if inp_test.is_empty() {
                    String::new()
                } else {
                    self.report(&metric, &inp_test, &tar_test)
                };

                let acc = format!(" {train_acc} {test_acc}");

                println!(
                    "{e} {train_err} {test_err}{acc} {:.1}",
                    start.elapsed().as_secs_f32()
                );
            }
        }
    }

    ///Returns `ReportMetric` for given inputs and targets
    pub fn report(
        &self,
        metric: &ReportMetric,
        inp: &Vec<&Vec<f32>>,
        tar: &Vec<&Vec<f32>>,
    ) -> String {
        let mut acc_string = String::new();

        match *metric {
            ReportMetric::None => {}
            ReportMetric::CorrectClassification => {
                let mut count = 0;

                match self.loss {
                    Loss::MSE => {} //not valid
                    Loss::SoftmaxAndCrossEntropy => {
                        for (inp, tar) in inp.iter().zip(tar) {
                            let pred = self.forward(inp);
                            if max_index_equal(tar, &pred) {
                                count += 1;
                            }
                        }
                    }
                    Loss::BinaryCrossEntropy => {
                        //if value>0.5 = 1 else 0 for binary cross entropy
                        //add 1 if all outputs match
                        for (inp, tar) in inp.iter().zip(tar) {
                            let pred = self.forward(inp);
                            let mut all_correct = true;
                            for (p, t) in pred.iter().zip(tar.iter()) {
                                let p = if *p > 0.5 { 1. } else { 0. };
                                if (p - *t).abs() > f32::EPSILON {
                                    all_correct = false;
                                    break;
                                }
                            }
                            if all_correct {
                                count += 1;
                            }
                        }
                    }
                }
                let t_acc = count as f32 / inp.len() as f32 * 100.;
                acc_string = format!("{t_acc:.2}%");
            }
            ReportMetric::RSquared => {
                // Calc 1 - SSR/SST = 1 - Σ(y-ŷ)/Σ(y-ȳ)
                //get ȳ
                //Rsquared is usually just one regression output, however if there are multiple,
                //we calculated  the r2 for each output

                //TRAIN,TEST
                let pred: Vec<Vec<f32>> = inp.iter().map(|inp| self.forward(inp)).collect();
                let mut r2s = vec![];
                for i in 0..tar[0].len() {
                    let tar = tar.iter().map(|x| x[i]).collect::<Vec<_>>();
                    let pred = pred.iter().map(|x| x[i]).collect::<Vec<_>>();
                    let avg: f32 = tar.iter().sum::<f32>() / tar.len() as f32;
                    let sst = tar.iter().map(|x| (x - avg).powi(2)).sum::<f32>();
                    let ssr = tar
                        .into_iter()
                        .zip(pred)
                        .map(|(tar, pred)| (tar - pred).powi(2))
                        .sum::<f32>();
                    let r2 = 1. - ssr / sst;
                    r2s.push(r2);
                }
                let r2 = r2s.iter().sum::<f32>() / r2s.len() as f32;

                acc_string = format!("{r2}");
            }
            ReportMetric::Custom(fun) => {
                let mut trains = vec![];
                for (inp, tar) in inp.iter().zip(tar) {
                    let pred = self.forward(inp);
                    trains.push(
                        tar.iter()
                            .zip(pred)
                            .map(|a| TargetPredicted {
                                target: *a.0,
                                predicted: a.1,
                            })
                            .collect::<Vec<_>>(),
                    );
                }

                acc_string = fun(trains);
            }
        }
        acc_string
    }
    /// Overrides initialization on each hidden layer, and reinitializes weights
    pub fn with_initialization(mut self, initialization: Initialization) -> Self {
        for layer in &mut self.layers {
            layer.initialization = initialization;
            layer.reinitialize();
        }
        self
    }
    /// Overrides regularization on each hidden layer.
    pub fn with_regularization(mut self, reg: Regularization) -> Self {
        for layer in &mut self.layers {
            layer.regularization = reg.clone();
        }
        self
    }
    /// Overrides dropout on each hidden layer.
    pub fn with_dropout(mut self, rate: f32) -> NN {
        for lay in &mut self.layers {
            lay.dropout = Some(Dropout::new(rate, lay.weights.shape()[1]));
        }
        self
    }
    /// Overrides activation on each hidden layer.
    pub fn with_activation_hidden(mut self, atype: Activation) -> Self {
        for l in 0..self.layers.len() - 1 {
            self.layers[l].activation = atype;
        }
        self
    }
    /// Overrides activation on output layer.
    pub fn with_activation_output(mut self, atype: Activation) -> Self {
        let last = self.layers.len() - 1;
        self.layers[last].activation = atype;
        self
    }
    /// Reinitializes weights on all layers
    pub fn reset_weights(&mut self) {
        for l in self.layers.iter_mut() {
            l.reinitialize();
        }
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate.get()
    }
}

fn smooth_labels(targets: &[&Vec<f32>], rate: f32) -> Vec<Vec<f32>> {
    targets
        .iter()
        .map(|a| {
            a.iter()
                .map(|v| v * (1. - rate) + rate / (targets[0].len() as f32))
                .collect()
        })
        .collect::<Vec<Vec<f32>>>()
}
impl Sede for NN {
    fn serialize(&self) -> String {
        let mut vec = vec![];
        vec.push(format!("learning_rate={}", self.learning_rate.serialize()));
        vec.push(format!("loss={}", self.loss.serialize()));
        vec.push(format!("optimizer={}", self.optimizer.serialize()));
        for l in &self.layers {
            vec.push(format!("layer={}", l.serialize()));
        }
        vec.push(format!(
            "label_smoothing={}",
            match self.label_smoothing {
                Some(a) => format!("{a}"),
                None => "None".to_string(),
            }
        ));
        vec.join("\n")
    }

    fn deserialize(s: &str) -> Result<Self, Error> {
        let data = s
            .split('\n')
            .filter_map(|x| {
                x.split_once('=')
                    .map(|(k, v)| vec![k.to_string(), v.to_string()])
            })
            .collect::<Vec<Vec<String>>>();

        let mut learning_rate = LearningRate::new(Rate::Constant(0.01));
        let mut loss = Loss::MSE;
        let mut label_smoothing = None;
        let mut optimizer = Optimizer::sgd().into();
        let mut layers: Vec<Dense> = vec![];
        for line in data {
            if line[0] == "learning_rate" {
                learning_rate = LearningRate::deserialize(&line[1])?;
            } else if line[0] == "label_smoothing" {
                label_smoothing = match line[1].as_str() {
                    "None" => None,
                    _ => Some(line[1].parse()?),
                };
            } else if line[0] == "loss" {
                loss = Loss::deserialize(&line[1])?;
            } else if line[0] == "optimizer" {
                println!("opti");
                optimizer = OptimizerInternal::deserialize(&line[1])?;
                println!("opti2");
            } else if line[0] == "layer" {
                layers.push(Dense::deserialize(&line[1])?);
            }
        }
        Ok(NN {
            input: layers.first().unwrap().weights.shape()[0],
            layers,
            learning_rate,
            loss,
            optimizer,
            label_smoothing,
        })
    }
}

impl Display for NN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(write!(
            f,
            r"
Shape: {:?}
Layers: {:?}
Learning Rate: {}
Loss: {}
Optimizer: {}
",
            self.shape(),
            self.layers
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>(),
            self.learning_rate,
            self.loss,
            self.optimizer,
        )?)
    }
}

//TOOLS which help

/// This checks the index of the maximum value in each vec are equal
/// Used to compare one hot encoding predicted with actual
/// Typically the actual will be 1 for a value and zero for else,
/// whereas the predicted may not be exactly one
/// So instead we compare the index of the maximum value, to determine equality
/// # Panics
/// If target and predicted lengths differ or if zero length
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

/// Returns the index of the maximum value, panics if empty vec passed in
/// also known as argmax
pub fn max_index(vec: &[f32]) -> usize {
    vec.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("max_index expects a vec with non zero size")
        .0
}

/// convert this into Matrix
/// Each column is an input into the neural network
/// Each row is an example
fn to_matrix(vec: &[&Vec<f32>]) -> Array2<f32> {
    assert!(!vec.is_empty(), "Input vec is empty");

    let rows = vec.len();
    let cols = vec[0].len();
    let mut data = Vec::with_capacity(rows * cols);
    for r in vec {
        data.extend_from_slice(r);
    }
    Array2::from_shape_vec((vec.len(), vec[0].len()), data)
        .expect("shape not allowed by size of vec")
}

pub struct TargetPredicted {
    pub target: f32,
    pub predicted: f32,
}

pub type CustomMetric = fn(Vec<Vec<TargetPredicted>>) -> String;
///Reporting metric
pub enum ReportMetric {
    ///No metric reported
    None,
    ///Assumes target is one hot encoding, chooses highest category,
    /// and calculates the % that are correctly classified
    CorrectClassification,
    ///Assumes target is regression, calculates 1-SSR/SST
    /// Assumes target Vec size is 1, else it will calculate the average RSquared over each output
    RSquared,
    ///Custom accuracy function.
    /// We pass in a Vec of examples.<br>
    /// For each example, there is a Vec of (Target,Predicted)<br>
    /// We Return a String representing the metric e.g. accuracy <br>
    /// Example: <br>
    /// We want to count the %of examples that is within 1% of target.  
    /// If predicted is within 1% of target, then it is correct, else it is incorrect
    ///```rust
    /// let fun = |ex: Vec<Vec<runnt::nn::TargetPredicted>>| {
    ///     let mut count = 0;
    ///     for eg in &ex {
    ///         if (eg[0].target - eg[0].predicted).abs() / eg[0].target < 0.01 {
    ///             count += 1
    ///         }
    ///     }
    ///     format!("{}%", count as f32 / ex.len() as f32 * 100.)
    ///};
    ///```
    Custom(CustomMetric),
}

#[derive(Clone, Debug)]
struct ValueAndActivated {
    ///holds the output values Z=(WxI)
    values: Vec<Array2<f32>>,
    ///hold the activated values A=f(Z)
    activated: Vec<Array2<f32>>,
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
