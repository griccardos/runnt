use std::collections::VecDeque;
use std::str::FromStr;

use ndarray::{Array2, arr2};
use tempfile::NamedTempFile;

use crate::prelude::*;
use crate::sede::*;

#[test]
fn builder_patterns() {
    //Simple builder
    let nn1 = NN::new(&[1, 2, 3, 4]) //give full network shape up front
        .with_activation_hidden(Activation::Sigmoid) //apply to all hidden layers
        .with_activation_output(Activation::Sigmoid) //apply to output only
        .with_regularization(Regularization::None) //apply to all layers
        .with_initialization(Initialization::Fixed(0.)); //apply to all layers

    //Complex builder
    //allows us to specify each layer individually
    let nn2 = NN::new_input(1) //start with input size
        .layer(2) //add layer with number assuming type = dense
        .layer(dense(3)) //or specify dense
        .layer(
            dense(4)
                .activation(Activation::Sigmoid) //configure activation for this layer
                .regularization(Regularization::None) //configure regularization for this layer
                .initializer(Initialization::Fixed(0.)), //configure initialization for this layer
        )
        .with_initialization(Initialization::Fixed(0.)); //can still override initialization for all layers
    assert_eq!(nn1.serialize(), nn2.serialize());
}
#[test]
fn test_shape() {
    let nn = NN::new(&[2, 3, 4, 5]);
    let shape = nn.shape();
    assert_eq!(shape, vec![2, 3, 4, 5]);
}

#[test]
/// Test xor nn against known outputs
fn test_xor() {
    //setup initial  weights
    let mut nn = NN::new(&[2, 2, 1])
        .with_learning_rate(0.5)
        .with_activation_hidden(Activation::Sigmoid)
        .with_activation_output(Activation::Sigmoid);

    nn.set_weights(&[0.15, 0.2, 0.25, 0.3, 0.4, 0.45]);
    nn.set_biases(&[0.35, 0.35, 0.6]);

    //check forward
    let vals = nn.forward(&[0., 0.]);
    assert_eq!(vals, [0.7500024]);

    //check first 4
    let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let outputs = [[0.], [1.], [1.], [0.]];

    let known_errors = [0.2812518, 0.034677744, 0.033219155, 0.28904837];
    let known_biases = [
        //1
        [arr2(&[[0.35, 0.35]]), arr2(&[[0.6]])].to_vec(),
        //2
        [arr2(&[[0.34317976, 0.34232724]]), arr2(&[[0.52968776]])].to_vec(),
        //3
        [arr2(&[[0.3452806, 0.34468588]]), arr2(&[[0.555233]])].to_vec(),
        //4
        [arr2(&[[0.3474572, 0.34712338]]), arr2(&[[0.5798897]])].to_vec(),
    ]
    .to_vec();

    for i in 0..4 {
        //check error
        assert_eq!(
            nn.layers.iter().map(|a| a.bias.clone()).collect::<Vec<_>>(),
            known_biases[i]
        );
        println!("{i}: biases ok");
        let error = nn.calc_error(&nn.forward(&inputs[i]), &outputs[i]);
        nn.fit_one(&inputs[i], &outputs[i]);
        assert_eq!(error, known_errors[i]);
        println!("{i}: errors ok");
    }
}

#[test]
fn xor_bce() {
    // Train XOR using sigmoid output + Binary Cross-Entropy (BCE)
    fastrand::seed(1);

    let mut nn = NN::new(&[2, 8, 1])
        .with_learning_rate(0.1)
        .with_activation_hidden(Activation::Sigmoid)
        .with_loss(Loss::BinaryCrossEntropy);

    let mut inp_out = [
        ([0., 0.], [0.]),
        ([0., 1.], [1.]),
        ([1., 0.], [1.]),
        ([1., 1.], [0.]),
    ];

    let mut solved = false;
    let mut step = 0;
    for _ in 0..10000 {
        fastrand::shuffle(&mut inp_out);
        nn.fit_one(&inp_out[0].0, &inp_out[0].1);

        let mut all_ok = true;
        for (inp, out) in &inp_out {
            let pred = nn.forward(inp);
            if (pred[0] - out[0]).abs() > 0.4 {
                all_ok = false;
            }
        }
        if all_ok {
            solved = true;
            break;
        }
        step += 1;
    }
    println!(
        "step {step} pred: 0,0 = {} 0,1 = {} 1,0 = {} 1,1 = {} ",
        nn.forward(&[0., 0.])[0],
        nn.forward(&[0., 1.])[0],
        nn.forward(&[1., 0.])[0],
        nn.forward(&[1., 1.])[0]
    );
    assert!(
        solved,
        "Network failed to learn XOR with BinaryCrossEntropy"
    );
}

#[test]
fn xor_gd() {
    //do mini batch gradient descent.
    //we take a few at a time
    fastrand::seed(1);
    let mut nn = crate::nn::NN::new(&[2, 8, 1])
        .with_learning_rate(0.8)
        .with_activation_hidden(Activation::Sigmoid)
        .with_activation_output(Activation::Sigmoid);

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
        nn.reset_weights();
        for steps in 0..20_000 {
            fastrand::shuffle(&mut inp_out);
            let ins = vec![&inp_out[0].0, &inp_out[1].0];
            let outs = vec![&inp_out[0].1, &inp_out[1].1];

            nn.fit_batch(&ins, &outs);

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
    assert!(avg < 3000);
}
#[test]
///use this in documentation
fn readme() {
    use crate::dataset::*;
    use crate::nn::*;
    //XOR

    let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let outputs = [[0.], [1.], [1.], [0.]];

    let mut nn = NN::new(&[2, 8, 1])
        .with_learning_rate(0.2)
        .with_activation_hidden(Activation::Tanh)
        .with_activation_output(Activation::Linear);

    for i in 0..5000 {
        nn.fit_one(&inputs[i % 4], &outputs[i % 4]);
    }

    //iris

    let set = Dataset::builder()
        .read_csv("examples/data/iris.csv")
        .add_input_columns(&[0, 1, 2, 3], Conversion::NormaliseMean)
        .add_target_columns(&[4], Conversion::OneHot)
        .allocate_to_test_data(0.4)
        .build();

    let mut net = NN::new(&[set.input_size(), 32, set.target_size()]).with_learning_rate(0.15);
    net.train(&set, 100, 8, 10, ReportMetric::CorrectClassification);

    //run only if have diamonds dataset

    if !Path::new(r"/temp/diamonds.csv").exists() {
        return;
    }

    //run only on release
    if cfg!(debug_assertions) {
        return;
    }

    let set = Dataset::builder()
        .read_csv(r"/temp/diamonds.csv")
        .allocate_to_test_data(0.2)
        .add_input_columns(&[0, 4, 5, 7, 8, 9], Conversion::NormaliseMean)
        .add_input_columns(&[1, 2, 3], Conversion::OneHot)
        .add_target_columns(
            &[6],
            Conversion::Function(|f| f.parse::<f32>().unwrap_or_default() / 1_000.),
        )
        .build();

    let save_path = r"/temp/network.txt";
    let mut net = if std::path::PathBuf::from_str(save_path).unwrap().exists() {
        NN::load(save_path).unwrap()
    } else {
        NN::new(&[set.input_size(), 32, set.target_size()])
    };

    //run for 100 epochs, with batch size 32 and report mse every 10 epochs
    net.train(&set, 100, 32, 10, ReportMetric::RSquared);
    net.save(save_path);
}
#[test]
fn test_save_load() {
    let nn = NN::new(&[10, 100, 10])
        .with_learning_rate(0.5)
        .with_regularization(Regularization::L1L2(0.1, 0.2))
        .with_activation_hidden(Activation::Tanh)
        .with_activation_output(Activation::Relu)
        .with_initialization(Initialization::Fixed(0.5))
        .with_optimizer(Optimizer::adam());

    let input = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    nn.forward(&input);
    let result1 = nn.forward(&input); //forward twice
    let temp = NamedTempFile::new().unwrap();
    let path = temp.path();

    //test looks the same
    let orig = nn.get_weights();
    let orig_shape = nn.shape();
    println!("nn:{nn} ");
    nn.save(path);
    let nn2 = NN::load(path).unwrap();
    let new = nn2.get_weights();
    let new_shape = nn2.shape();
    println!("nn loaded:{nn2} ");
    assert_eq!(orig_shape, new_shape);
    assert_eq!(orig, new);
    assert_eq!(nn.serialize(), nn2.serialize());
    assert_eq!(nn.get_weights(), nn2.get_weights());
    assert_eq!(nn.get_biases(), nn2.get_biases());

    //test result the same
    nn2.forward(&input);
    let result2 = nn2.forward(&input);
    assert_eq!(result1, result2);
}

#[test]
fn test_get_set_weights() {
    //the result before and after should be the same if set with the same weights
    let mut nn = NN::new(&[2, 2, 2]);
    let weights = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    nn.set_weights(&weights);
    assert_eq!(&weights, &nn.get_weights());
}
#[test]
fn test_get_set_bias() {
    //the result before and after should be the same if set with the same weights
    let mut nn = NN::new(&[2, 2, 2]);
    let bias = vec![1., 2., 3., 4.];
    nn.set_biases(&bias);
    assert_eq!(&bias, &nn.get_biases());
}

#[test]
fn test_softmax_splits_even() {
    let nn = NN::new(&[1, 4])
        .with_initialization(Initialization::Fixed(1.))
        .with_loss(crate::loss::Loss::SoftmaxAndCrossEntropy);
    let test = arr2(&[[5.], [3.]]); //they should all have the same weight so 0.25
    let out = nn.internal_forward(&test, false);
    let out = out.activated.last().unwrap();
    for row in out.rows() {
        assert_eq!(row.to_vec(), [0.25, 0.25, 0.25, 0.25]);
    }
}

#[test]
fn test_softmax_sums_to_one() {
    let nn = NN::new(&[1, 10, 8]).with_loss(crate::loss::Loss::SoftmaxAndCrossEntropy);
    let test = arr2(&[[5.], [3.]]);
    let out = nn.internal_forward(&test, false);
    let out = out.activated.last().unwrap();
    for row in out.rows() {
        let sum = row.sum();
        assert!(sum.abs() - 1.0 < 0.00001);
    }
}

//test against known weights,
// manually calculated weights
#[test]
fn known_weights() {
    let mut nn = NN::new(&[2, 3, 2])
        .with_activation_output(Activation::Sigmoid)
        .with_learning_rate(0.1);
    println!("nn: {:?}, {:?}", nn.get_weights(), nn.get_biases());
    nn.set_weights(&[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
    assert_eq!(nn.get_biases(), &[0., 0., 0., 0., 0.]);

    assert_eq!(nn.forward(&[-0.5, 0.7]), [0.5378027, 0.5378027]);
    nn.fit_one(&[-0.5, 0.7], &[0., 0.]);

    assert_eq!(
        nn.get_weights(),
        vec![
            0.100334175,
            0.100334175,
            0.100334175,
            0.09953216,
            0.09953216,
            0.09953216,
            0.09324905,
            0.09324905,
            0.09324905,
            0.09324905,
            0.09324905,
            0.09324905,
        ]
    );
}

fn approx_eq_mat(a: &Array2<f32>, b: &Array2<f32>, eps: f32) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
}

#[test]
fn test_loss_and_backprop_mse() {
    // Network: 2 inputs -> 1 output, linear output + MSE
    let mut nn = NN::new(&[2, 1])
        .with_activation_output(Activation::Linear)
        .with_loss(Loss::MSE);

    // set weights and bias to zero (2 weights)
    nn.set_weights(&[0.0, 0.0]);

    // single example input and target
    let input = arr2(&[[1.0, 2.0]]); // shape (1 x 2)
    let target = arr2(&[[1.0]]); // shape (1 x 1)

    // forward
    let vals = nn.internal_forward(&input, false);
    let outputs = vals.activated.last().unwrap().clone(); // (1 x 1) -> a = 0 because weights/bias = 0

    // check internal_calc_errors (MSE) = 0.5 * (t - a)^2 = 0.5
    let errs = nn.internal_calc_errors(&outputs, &target);
    assert!((errs[0] - 0.5).abs() < 1e-6);

    // check output_loss (dE/dA) and backwards gradient:
    let loss_grad = nn.loss.gradient(&outputs, &target); // should be a - t = -1
    let (wgrads, _bgrads) = nn.backwards(vals, loss_grad);

    // expected weight gradient for single example:
    // de/dz = a - t = -1
    // weight_grad = input^T dot de_dz => a 2x1 matrix: [[1 * -1], [2 * -1]]
    let expected = arr2(&[[-1.0], [-2.0]]);
    assert!(approx_eq_mat(&wgrads[0], &expected, 1e-6));
}

#[test]
fn test_loss_and_backprop_softmax_crossentropy() {
    // Network: 3 inputs -> 3 outputs, use Softmax + Cross-Entropy
    let mut nn = NN::new(&[3, 3]).with_loss(Loss::SoftmaxAndCrossEntropy);

    // set weights and bias to zero (3*3 weights )
    nn.set_weights(&vec![0.0f32; 9]);

    // single example: input selects first feature, target is one-hot class 0
    let input = arr2(&[[1.0, 0.0, 0.0]]); // (1 x 3)
    let target = arr2(&[[1.0, 0.0, 0.0]]); // (1 x 3)

    // forward: with zero pre-activations, softmax -> uniform probabilities 1/3
    let vals = nn.internal_forward(&input, false);
    let outputs = vals.activated.last().unwrap().clone();
    for &p in outputs.row(0).iter() {
        // each probability must be approx 1/3
        assert!((p - (1.0f32 / 3.0)).abs() < 1e-6);
    }

    // internal_calc_errors should equal -ln(1/3) for the one-hot class
    let errs = nn.internal_calc_errors(&outputs, &target);
    let expected_loss = (3.0f32).ln(); // -ln(1/3) = ln(3)
    assert!((errs[0] - expected_loss).abs() < 1e-6);

    // Now check gradients:
    let loss_grad = nn.loss.gradient(&outputs, &target); // a - t = [1/3 - 1, 1/3, 1/3]
    let (wgrads, _bgrads) = nn.backwards(vals, loss_grad);

    // For input [1,0,0], expected weight gradient matrix shape (3 x 3):
    // first row should be (a - t), other rows zero
    let row0 = outputs.row(0).to_vec();
    let de = vec![row0[0] - 1.0, row0[1] - 0.0, row0[2] - 0.0];
    let mut expected = Array2::zeros((3, 3));
    for (j, &val) in de.iter().enumerate() {
        expected[[0, j]] = val;
    }
    assert!(approx_eq_mat(&wgrads[0], &expected, 1e-6));
}

#[test]
fn test_loss_and_backprop_binary_crossentropy() {
    // Network: 2 inputs -> 1 output, use sigmoid + binary cross-entropy
    let mut nn = NN::new(&[2, 1]).with_loss(Loss::BinaryCrossEntropy);

    // set weights  (2 weights )
    nn.set_weights(&[0.0, 0.0]);
    nn.set_biases(&[0.]);

    // single example input and binary target = 1
    let input = arr2(&[[1.0, 2.0]]); // (1 x 2)
    let target = arr2(&[[1.0]]); // (1 x 1)

    // forward: z = 0 -> sigmoid(z) = 0.5
    let vals = nn.internal_forward(&input, false);
    let outputs = vals.activated.last().unwrap().clone();
    assert!((outputs[[0, 0]] - 0.5).abs() < 1e-6);

    // internal_calc_errors -> -[t*ln(a) + (1-t)*ln(1-a)] for t=1 -> -ln(0.5) = ln(2)
    let errs = nn.internal_calc_errors(&outputs, &target);
    let expected_loss = 2f32.ln();
    assert!((errs[0] - expected_loss).abs() < 1e-6);

    // gradients: output_loss = a - t = -0.5
    let loss_grad = nn.loss.gradient(&outputs, &target);
    let (wgrads, _bgrads) = nn.backwards(vals, loss_grad);

    // expected weight gradient = input^T * (a - t) => [[-0.5], [-1.0]]
    let expected = arr2(&[[-0.5], [-1.0]]);
    assert!(approx_eq_mat(&wgrads[0], &expected, 1e-6));
}

#[test]
fn compare_mse_vs_bce_xor() {
    let seeds = 1u64..10;
    let max_steps = 20000;
    let hidden = 2;
    let mut results = vec![];

    for seed in seeds.into_iter() {
        fastrand::seed(seed);
        // MSE run
        let mut nn_mse = NN::new(&[2, hidden, 1])
            .with_learning_rate(0.25)
            .with_activation_hidden(Activation::Sigmoid)
            .with_activation_output(Activation::Sigmoid)
            .with_loss(Loss::MSE);

        // BCE run
        fastrand::seed(seed); // same init for fair comparison
        let mut nn_bce = NN::new(&[2, hidden, 1])
            .with_learning_rate(0.25)
            .with_activation_hidden(Activation::Sigmoid)
            .with_loss(Loss::BinaryCrossEntropy);

        let mut inp_out = [
            ([0., 0.], [0.]),
            ([0., 1.], [1.]),
            ([1., 0.], [1.]),
            ([1., 1.], [0.]),
        ];

        let mut run = |nn: &mut NN| {
            let mut solved_at: Option<usize> = None;
            for step in 0..max_steps {
                fastrand::shuffle(&mut inp_out);
                nn.fit_one(&inp_out[0].0, &inp_out[0].1);

                if step % 50 == 0 {
                    let mut all_ok = true;
                    for (inp, out) in &inp_out {
                        let pred = nn.forward(inp)[0];
                        // accept rounding to nearest class
                        if pred.round() as i32 != out[0] as i32 {
                            all_ok = false;
                            break;
                        }
                    }
                    if all_ok {
                        solved_at = Some(step);
                        break;
                    }
                }
            }
            solved_at.unwrap_or(max_steps)
        };

        let t_mse = run(&mut nn_mse);
        let t_bce = run(&mut nn_bce);
        results.push((seed, t_mse, t_bce));
    }

    for (seed, t_mse, t_bce) in &results {
        println!("seed {}: MSE steps={}  BCE steps={}", seed, t_mse, t_bce);
    }
    let mse_avg = results.iter().map(|r| r.1).sum::<usize>() as f32 / results.len() as f32;
    let bce_avg = results.iter().map(|r| r.2).sum::<usize>() as f32 / results.len() as f32;
    println!("Avg MSE steps: {}", mse_avg);
    println!("Avg BCE steps: {}", bce_avg);
    assert!(bce_avg < mse_avg);
}

#[test]
fn test_label_smoothing_scales_gradients() {
    // smoothing should scale gradients by (1 - r).
    let k = 3usize;
    let rate = 0.1f32;
    let mut nn = NN::new(&[1, k]).with_loss(Loss::SoftmaxAndCrossEntropy);

    let zeros = vec![0f32; 1 * k];
    nn.set_weights(&zeros);

    let input = arr2(&[[1.0f32]]);
    let vals = nn.internal_forward(&input, false);
    let outputs = vals.activated.last().unwrap().clone();

    let mut target_vec = vec![0f32; k];
    target_vec[0] = 1.0;
    let target = Array2::from_shape_vec((1, k), target_vec.clone()).unwrap();

    let mut smoothed_vec = vec![rate / (k as f32); k];
    smoothed_vec[0] = 1.0 - rate + rate / (k as f32);
    let smoothed = Array2::from_shape_vec((1, k), smoothed_vec).unwrap();

    // compute gradients (dE/dA) for both targets
    let grad_no = nn.loss.gradient(&outputs, &target);
    let grad_smooth = nn.loss.gradient(&outputs, &smoothed);

    // compute weight gradients via backwards
    let (wgrads_no, _b_no) = nn.backwards(vals.clone(), grad_no);
    let (wgrads_s, _b_s) = nn.backwards(vals, grad_smooth);

    let expected_factor = 1.0 - rate;

    // check that each weight gradient equals scaled version
    // e.g. -0.66 in non smooth will be -0.66*0.9=-0.6 in smoothed
    let eps = 1e-6;
    let gno = &wgrads_no[0];
    let gs = &wgrads_s[0];
    for (x, y) in gno.iter().zip(gs.iter()) {
        assert!((y - x * expected_factor).abs() < eps);
    }
}

#[test]
fn test_label_smoothing_changes_training_update() {
    let mut nn_no_smooth = NN::new(&[1, 3]).with_loss(Loss::SoftmaxAndCrossEntropy);
    let mut nn_smooth = NN::new(&[1, 3])
        .with_loss(Loss::SoftmaxAndCrossEntropy)
        .with_label_smoothing(0.1);

    let initial_weights = vec![0.0f32; 3]; // 1*3 weights
    nn_no_smooth.set_weights(&initial_weights);
    nn_smooth.set_weights(&initial_weights);

    let input = [1.0f32];
    let target = [1.0f32, 0.0f32, 0.0f32];

    nn_no_smooth.fit_one(&input, &target);
    nn_smooth.fit_one(&input, &target);

    // weights should differ because smoothing changes the effective target
    assert_ne!(nn_no_smooth.get_weights(), nn_smooth.get_weights());
}
#[test]
fn test_dropout_masks() {
    fastrand::seed(2);

    let dropout_rate = 0.2;
    let mut nn = NN::new_input(2)
        .layer(dense(20).dropout(dropout_rate))
        .layer(3);
    nn.update_dropout();
    let mask_opt = nn.layers[0].dropout.as_ref().map(|a| a.mask());
    assert!(mask_opt.is_some());
    let first_mask = mask_opt.unwrap();

    for &v in first_mask.iter() {
        assert!(v == 0.0f32 || v == 1.0f32 / (1.0 - dropout_rate));
    }

    // we expect at least one 0 and one scaled
    let zero_count = first_mask.iter().filter(|&&x| x == 0.0f32).count();
    let non_zero_count = first_mask.iter().filter(|&&x| x != 0.0f32).count();
    println!("masks: {:?}", first_mask);
    assert!(zero_count > 0);
    assert!(non_zero_count > 0);
    //last layer  has no dropout:
    assert_eq!(nn.layers[1].dropout, None)
}
#[test]
fn test_dropout_forward_and_backward_masking() {
    fastrand::seed(2);

    let dropout_rate = 0.5f32;
    let mut nn = NN::new_input(2)
        .layer(dense(4).dropout(dropout_rate))
        .layer(1);
    nn.update_dropout();
    // set all weights to 1.0 so pre-activation is deterministic (z = sum(inputs*1))
    let total_w: usize = nn.layers.iter().map(|w| w.weights.len()).sum();
    nn.set_weights(&vec![1.0f32; total_w]);

    // single example with inputs [1, 1] -> z_hidden = 1*1 + 1*1 = 2.0 for each hidden unit
    let input = arr2(&[[1.0f32, 1.0f32]]);

    let vals = nn.internal_forward(&input, true);
    let hidden_activ = vals.activated.get(1).expect("hidden layer exists");
    let mask_vec = &nn.layers[0].dropout.as_ref().map(|a| a.mask()).unwrap();
    println!("mask: {:?}", mask_vec);
    println!("activations: {:?}", vals.activated);
    println!("hidden: {:?}", hidden_activ);

    // compute expected unmasked activation for sigmoid(2.0)
    let z = 2.0f32;
    let expected_unmasked = 1.0f32 / (1.0f32 + (-z).exp());
    let eps = 1e-6f32;

    // check forward masking/scaling
    for j in 0..mask_vec.len() {
        let actual = hidden_activ[[0, j]];
        if mask_vec[j] == 0.0f32 {
            assert!(actual == 0., "node {j} should be dropped, got {actual}");
        } else {
            let expected = expected_unmasked * mask_vec[j];
            assert!(
                (actual - expected).abs() < eps,
                "node {j} expected {expected} got {actual}"
            );
        }
    }

    // Now test backward: gradients for dropped units should be zero.
    // Use a target of zero for simple gradient (output - target)
    let outputs = vals.activated.last().unwrap().clone();
    println!("outputs: {:?}", outputs);
    let targets = arr2(&[[0.0f32]]);
    let loss_grad = nn.loss.gradient(&outputs, &targets);
    println!("loss grad: {:?}", loss_grad);

    let (wgrads, bgrads) = nn.backwards(vals.clone(), loss_grad);
    println!("weight grads: {:?}", wgrads);

    // First layer gradients correspond to input->hidden (shape: inputs x hidden)
    let w0 = &wgrads[0];
    let b0 = &bgrads[0]; // shape 1 x hidden

    for j in 0..mask_vec.len() {
        // check column j of weight gradient (all input rows) and bias gradient
        let col_zero = w0.column(j).iter().all(|&v| v.abs() < eps);
        let bias_zero = (b0[[0, j]]).abs() < eps;

        if mask_vec[j] == 0.0f32 {
            assert!(
                col_zero,
                "weight gradients for dropped unit {j} should be zero"
            );
            assert!(
                bias_zero,
                "bias gradient for dropped unit {j} should be zero"
            );
        } else {
            let any_nonzero =
                w0.column(j).iter().any(|&v| v.abs() > eps) || (b0[[0, j]].abs() > eps);
            assert!(any_nonzero, "active unit {j} should have non-zero gradient");
        }
    }
}
