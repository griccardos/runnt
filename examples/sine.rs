use runnt::activation::ActivationType;

pub fn main() {
    fastrand::seed(1);

    //Create Neural Network
    let mut nn = runnt::nn::NN::new(&[1, 8, 8, 1])
        .with_hidden_type(ActivationType::Sigmoid) //Non linear Activation function
        .with_output_type(ActivationType::Sigmoid) //Expected output function
        .with_learning_rate(0.5); //Learning rate

    //generate function which we want to predict
    let sinefn = |x: f32| x * x * (5. * 1.14 * x).sin().powi(2);

    //get some observations
    let inp_out = (0..1000)
        .map(|i| {
            let x = i as f32 / 1000.;
            let y = sinefn(x);
            (vec![x], vec![y])
        })
        .collect::<Vec<(Vec<f32>, Vec<f32>)>>();

    let mut mse_sum = 0.;
    let mut avg_mse;
    for step in 1..1_000_000 {
        let rand_index = fastrand::usize(0..inp_out.len());

        //Stochastic gradient descent
        nn.fit_one(&inp_out[rand_index].0, &inp_out[rand_index].1);

        mse_sum += nn.error();
        avg_mse = mse_sum / step as f32;

        if step % 100000 == 0 {
            println!("step {step}: mse={avg_mse}");
        }
    }

    inp_out
        .iter()
        .enumerate()
        .filter(|i| i.0 % (inp_out.len() / 10) == 0)
        .map(|x| x.1)
        .for_each(|(x, y)| {
            let pred = nn.forward(&x);
            let x = x[0];
            let y = y[0];
            println!(
                "x={x:.3} actual={y:.3} pred={:.3} diff= {:.3}",
                pred[0],
                y - pred[0]
            );
        })
}
