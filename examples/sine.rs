use runnt::prelude::*;
//Regression example
pub fn main() {
    fastrand::seed(1);

    //Create Neural Network
    let mut nn = NN::new_input(1)
        .layer(8)
        .layer(dense(8).activation(Activation::Tanh))
        .layer(dense(1).activation(Activation::Sigmoid))
        .with_learning_rate(0.5); //Learning rate

    //generate function which we want to predict
    let sinefn = |x: f32| x * x * (5. * 1.14 * x).sin().powi(2);

    //get some observations
    let inp_out = (0..1000)
        .map(|i| {
            let x = i as f32 / 1000.;
            let y = sinefn(x);
            vec![x, y]
        })
        .collect::<Vec<_>>();

    let set = Dataset::builder()
        .add_data(&inp_out)
        .allocate_to_test_data(0.2)
        .add_input_columns(&[0], runnt::dataset::Conversion::F32)
        .add_target_columns(&[1], runnt::dataset::Conversion::F32)
        .build();
    nn.train(&set, 1000, 1, 100, ReportMetric::None);

    inp_out
        .iter()
        .enumerate()
        .filter(|i| i.0 % (inp_out.len() / 10) == 0)
        .map(|x| x.1)
        .for_each(|xy| {
            let pred = nn.forward(&[xy[0]]);
            let x = xy[0];
            let y = xy[1];
            println!(
                "x={x:.3} actual={y:.3} pred={:.3} diff= {:.3}",
                pred[0],
                y - pred[0]
            );
        })
}
