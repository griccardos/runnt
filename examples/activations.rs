use runnt::{
    activation::ActivationType, dataset::Dataset, initialization::InitializationType, nn::NN,
};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

pub fn main() {
    fastrand::seed(1);

    let sinefn = |x: f32| x * x * (5. * 1.14 * x).sin().powi(2);

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

    const EPOCHS: usize = 1000;
    const BATCH_SIZE: usize = 1;
    const LEARNING_RATE: f32 = 0.05;

    let mut per_epoch = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("activation_epochs.csv")
        .expect("Could not create per-epoch CSV");

    let mut summary = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("activation_summary.csv")
        .expect("Could not create summary CSV");

    let activations = [
        ActivationType::Relu,
        ActivationType::Sigmoid,
        ActivationType::Linear,
        ActivationType::Tanh,
        ActivationType::Swish,
    ];

    let mut all_mses: Vec<Vec<f32>> = Vec::with_capacity(activations.len());

    writeln!(
        summary,
        "activation,epochs,final_train_mse,final_test_mse,time_seconds"
    )
    .unwrap();

    for &act in &activations {
        let mut net = NN::new(&[1, 8, 8, 1])
            .with_hidden_type(act)
            .with_output_type(ActivationType::Sigmoid)
            .with_initialization(InitializationType::Xavier)
            .with_learning_rate(LEARNING_RATE);

        let mut mse_series: Vec<f32> = Vec::with_capacity(EPOCHS);
        let start = Instant::now();
        let mut final_train_mse = 0.;
        let mut final_test_mse = 0.;

        for _epoch in 1..=EPOCHS {
            let (inp, tar) = set.get_data();
            net.fit(&inp, &tar, BATCH_SIZE);

            final_train_mse = net.forward_errors(&inp, &tar);
            let (inp_test, tar_test) = set.get_test_data();
            let test_mse = if inp_test.is_empty() {
                0.
            } else {
                net.forward_errors(&inp_test, &tar_test)
            };
            final_test_mse = test_mse;
            mse_series.push(test_mse);
        }

        let elapsed = start.elapsed().as_secs_f32();
        all_mses.push(mse_series);

        writeln!(
            summary,
            "{},{},{:.8},{:.8},{:.4}",
            act, EPOCHS, final_train_mse, final_test_mse, elapsed
        )
        .unwrap();
        println!(
            "Activation {} completed: epochs={} final_train_mse={:.8} final_test_mse={:.8} time(s)={:.4}",
            act,
            EPOCHS,
            final_train_mse,
            final_test_mse,
            elapsed
        );
    }

    let header = std::iter::once("epoch".to_string())
        .chain(activations.iter().map(|a| a.to_string()))
        .collect::<Vec<_>>()
        .join(",");
    writeln!(per_epoch, "{}", header).unwrap();

    for epoch in 0..EPOCHS {
        let mut row = vec![(epoch + 1).to_string()];
        for series in &all_mses {
            row.push(format!("{:.8}", series[epoch]));
        }
        writeln!(per_epoch, "{}", row.join(",")).unwrap();
    }

    println!("Finished. Saved files: activation_epochs.csv; activation_summary.csv");
}
