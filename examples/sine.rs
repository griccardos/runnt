use runnt::activation::ActivationType;

pub fn main() {
    fastrand::seed(1);

    //generate random function which we want to predict
    let sinefn = |x: f32| x * x * (5. * 3.14159 * x).sin().powi(6);

    let mut inp_out = (0..1000)
        .map(|_| {
            let x = fastrand::f32();
            let y = sinefn(x);
            (vec![x], vec![y])
        })
        .collect::<Vec<(Vec<f32>, Vec<f32>)>>();
    let mut nn = runnt::nn::NN::new(&[1, 16, 16, 1])
        .with_hidden_type(ActivationType::Sigmoid)
        .with_output_type(ActivationType::Sigmoid)
        .with_learning_rate(0.2);

    let mut mse_sum = 0.;
    let mut avg_mse;
    for e in 1..20_000 {
        fastrand::shuffle(&mut inp_out);
        let ins = inp_out
            .iter()
            .map(|x| x.0.as_slice())
            .collect::<Vec<&[f32]>>();
        let outs = inp_out
            .iter()
            .map(|x| x.1.as_slice())
            .collect::<Vec<&[f32]>>();

        nn.fit_batch_size(&ins, &outs, 1);

        mse_sum += nn.error();
        avg_mse = mse_sum / e as f32;

        if e % 1000 == 0 {
            println!("epoch {e}: mse={avg_mse}");
        }
    }

    (0..10).for_each(|_| {
        let x = fastrand::f32();
        let y = sinefn(x);
        let pred = nn.forward(&[x]);
        println!(
            "x={x:.3} actual={y:.3} pred={:.3} diff= {:.3}",
            pred[0],
            y - pred[0]
        );
    })
}
