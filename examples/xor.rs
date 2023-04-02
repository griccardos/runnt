use runnt::activation::ActivationType;

//Classification, non linear example
pub fn main() {
    fastrand::seed(1);

    //create network
    let mut nn = runnt::nn::NN::new(&[2, 8, 1])
        .with_learning_rate(0.2)
        .with_hidden_type(ActivationType::Tanh)
        .with_output_type(ActivationType::Sigmoid); //we expect it to be 0 or 1 so sigmoid is ok

    let mut inp_out = [
        (vec![0., 0.], vec![0.1]),
        (vec![0., 1.], vec![1.]),
        (vec![1., 0.], vec![1.]),
        (vec![1., 1.], vec![0.]),
    ];

    let mut mse_sum = 0.;

    for _ in 1..1000 {
        fastrand::shuffle(&mut inp_out);

        let ins = inp_out.iter().map(|x| &x.0).collect::<Vec<_>>();
        let outs = inp_out.iter().map(|x| &x.1).collect::<Vec<_>>();

        nn.fit_batch(&ins, &outs); //we do batch descent, passing in all observations each time
        let mse: f32 = nn.forward_errors(&ins, &outs);

        mse_sum += mse;
    }
    let avg_mse = mse_sum / 1000.;

    println!("avg mse: {avg_mse}");
    println!("We want 0,1,1,0:");
    println!(
        "Prediction for: [0,0]: {}",
        nn.forward(&[0., 0.]).first().unwrap().round(),
    );
    println!(
        "Prediction for: [0,1]: {}",
        nn.forward(&[0., 1.]).first().unwrap().round(),
    );
    println!(
        "Prediction for: [1,0]: {}",
        nn.forward(&[1., 0.]).first().unwrap().round(),
    );
    println!(
        "Prediction for: [1,1]: {}",
        nn.forward(&[1., 1.]).first().unwrap().round(),
    );
}
