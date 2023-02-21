use runnt::activation::ActivationType;

pub fn main() {
    fastrand::seed(1);
    let mut inp_out = [
        ([0., 0.], [0.1]),
        ([0., 1.], [1.]),
        ([1., 0.], [1.]),
        ([1., 1.], [0.]),
    ];
    let mut nn = runnt::nn::NN::new(&[2, 8, 1])
        .with_learning_rate(0.2)
        .with_hidden_type(ActivationType::Tanh)
        .with_output_type(ActivationType::Linear);

    let mut mse_sum = 0.;
    let mut avg_mse = 0.;

    for e in 1..500 {
        fastrand::shuffle(&mut inp_out);

        let ins = inp_out.iter().map(|x| &x.0[..]).collect::<Vec<&[f32]>>();
        let outs = inp_out.iter().map(|x| &x.1[..]).collect::<Vec<&[f32]>>();
        nn.fit(&ins, &outs);

        mse_sum += nn.error();
        avg_mse = mse_sum / e as f32;
    }

    println!("avg mse: {avg_mse}");

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
