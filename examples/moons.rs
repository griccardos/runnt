use std::f32::consts::PI;

use runnt::{activation::ActivationType, dataset::Dataset, nn::max_index};

// Example: Classification, dataset, run_and_report
pub fn main() {
    fastrand::seed(1);

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

    let set = Dataset::builder()
        .add_data(&inp_out)
        .allocate_to_test_data(0.2)
        .add_input_columns(&[0, 1], runnt::dataset::Conversion::F32)
        .add_target_columns(&[2], runnt::dataset::Conversion::OneHot)
        .build();

    //Create Neural Network
    let mut nn = runnt::nn::NN::new(&[set.input_size(), 8, set.target_size()])
        .with_hidden_type(ActivationType::Sigmoid)
        .with_output_type(ActivationType::Linear)
        .with_learning_rate(0.5); //Learning rate

    runnt::nn::run_and_report(
        &set,
        &mut nn,
        100,
        1,
        10,
        runnt::nn::ReportMetric::CorrectClassification,
    );

    println!("x,y,tar,predicted");
    set.get_test_data_zip().iter().take(400).for_each(|x| {
        let pred = nn.forward(&x.0);
        println!(
            "{},{},{},{}",
            x.0[0],
            x.0[1],
            max_index(x.1),
            max_index(&pred),
        );
    });
}
