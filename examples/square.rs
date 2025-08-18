use runnt::activation::ActivationType;

//Regression and regularization example
pub fn main() {
    fastrand::seed(1);

    //Create Neural Network
    let mut nn = runnt::nn::NN::new(&[6, 8, 8, 1])
        .with_hidden_type(ActivationType::Sigmoid)
        .with_output_type(ActivationType::Linear)
        .with_regularization(runnt::regularization::Regularization::None)
        .with_learning_rate(0.2);

    let mut nnl1 = runnt::nn::NN::new(&[6, 8, 8, 1])
        .with_hidden_type(ActivationType::Sigmoid)
        .with_output_type(ActivationType::Linear)
        .with_regularization(runnt::regularization::Regularization::L1(0.001))
        .with_learning_rate(0.2);

    let mut nnl2 = runnt::nn::NN::new(&[6, 8, 8, 1])
        .with_hidden_type(ActivationType::Sigmoid)
        .with_output_type(ActivationType::Linear)
        .with_regularization(runnt::regularization::Regularization::L2(0.001))
        .with_learning_rate(0.4);

    let mut nnl1l2 = runnt::nn::NN::new(&[6, 8, 8, 1])
        .with_hidden_type(ActivationType::Sigmoid)
        .with_output_type(ActivationType::Linear)
        .with_regularization(runnt::regularization::Regularization::L1L2(0.001, 0.001))
        .with_learning_rate(0.4);

    //generate function which we want to predict
    //returns ( observed y,true y)
    //This has a bit of randomness, because we want to test regularisation
    let squarefn = |x: f32| {
        let y = x * x + x;
        let noise_y = y + (fastrand::f32() - 0.5) / 3.;
        (noise_y, y)
    };

    //get some observations
    let inp_out = (0..20)
        .map(|i| {
            let x1 = -1. + i as f32 / 10.;
            let x2 = x1.powi(2);
            let x3 = x1.powi(3);
            let x4 = x1.powi(4);
            let x5 = x1.powi(5);
            let x6 = x1.powi(6);

            let obsy = squarefn(x1).0;
            let truey = squarefn(x1).1;
            (vec![x1, x2, x3, x4, x5, x6], vec![obsy], vec![truey])
        })
        .collect::<Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>>();

    let mut mse_sum = 0.;
    //high iterations because we want to overfit...
    for step in 1..2_000_000 {
        let rand_index = fastrand::usize(0..inp_out.len());

        //Stochastic gradient descent
        nn.fit_one(&inp_out[rand_index].0, &inp_out[rand_index].1);
        let err = nn.forward_error(&inp_out[rand_index].0, &inp_out[rand_index].1);

        nnl1.fit_one(&inp_out[rand_index].0, &inp_out[rand_index].1);
        nnl2.fit_one(&inp_out[rand_index].0, &inp_out[rand_index].1);
        nnl1l2.fit_one(&inp_out[rand_index].0, &inp_out[rand_index].1);

        mse_sum += err;
        let avg_mse = mse_sum / step as f32;

        if step % 100000 == 0 {
            println!("step {step}: mse={avg_mse}");
        }
    }

    println!("x,noisey,truey,predy,predyl1,predyl2,predl1l2");
    inp_out
        .iter()
        .enumerate()
        .filter(|i| i.0 % (inp_out.len() / 20) == 0)
        .map(|x| x.1)
        .for_each(|(x, y, truey)| {
            let pred = nn.forward(x);
            let predl2 = nnl2.forward(x);
            let predl1 = nnl1.forward(x);
            let predl1l2 = nnl1l2.forward(x);
            let x = x[0];
            let noisey = y[0];
            let truey = truey[0];
            let pred = pred[0];
            let predl2 = predl2[0];
            let predl1 = predl1[0];
            let predl1l2 = predl1l2[0];
            println!(
                "{x:.3},{noisey:.3},{truey:.3},{pred:.3},{predl1:.3},{predl2:.3},{predl1l2:.3}",
            );
        });

    println!("plot above on graph to observe difference");
}
