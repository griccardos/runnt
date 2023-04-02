use runnt::{activation::ActivationType, dataset::Dataset, nn};

//Regression example
pub fn main() {
    fastrand::seed(1);

    //Create Neural Network
    let mut nn = runnt::nn::NN::new(&[1, 8, 8, 1])
        .with_hidden_type(ActivationType::Sigmoid) //Non linear Activation function
        .with_output_type(ActivationType::Sigmoid) //Expected output between 0 and 1
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
    nn::run_and_report(&set, &mut nn, 1000, 1, 100, nn::ReportMetric::None);

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
