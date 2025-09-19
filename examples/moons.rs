use std::f32::consts::PI;

use runnt::{activation::Activation, dataset::Dataset, nn::ReportMetric};

// Example: Classification, dataset, train
pub fn main() {
    fastrand::seed(1);

    let inp_out = generate_moons();

    let set = Dataset::builder()
        .add_data(&inp_out)
        .allocate_to_test_data(0.2)
        .add_input_columns(&[0, 1], runnt::dataset::Conversion::F32)
        .add_target_columns(&[2], runnt::dataset::Conversion::F32)
        .build();

    //Create Neural Network
    let mut nn = runnt::nn::NN::new(&[set.input_size(), 8, set.target_size()])
        .with_activation_hidden(Activation::Sigmoid)
        .with_activation_output(Activation::Linear)
        .with_loss(runnt::loss::Loss::BinaryCrossEntropy)
        .with_learning_rate(0.1); //Learning rate

    nn.train(&set, 20, 1, 1, ReportMetric::CorrectClassification);
}

pub fn generate_moons() -> Vec<Vec<f32>> {
    //generate function which we want to predict
    let sine1 = |x: f32| (x * 2. + PI * 2. / 4.).sin() + fastrand::f32() / 2. - 0.2;
    let sine2 = |x: f32| (x * 2. + PI * 4. / 4.).sin() + fastrand::f32() / 2. - 0.2;

    //get some observations
    let mut inp_out = vec![];
    //moon1 from -1 to 1
    (0..1000).for_each(|_| {
        let x = fastrand::f32() * 2. - 1.;
        let y = sine1(x);
        let cat = 0.;
        inp_out.push(vec![x, y, cat]);
    });
    //moon2 from 0 to 2
    (0..1000).for_each(|_| {
        let x = fastrand::f32() * 2.;
        let y = sine2(x);
        let cat = 1.;
        inp_out.push(vec![x, y, cat]);
    });
    inp_out
}
