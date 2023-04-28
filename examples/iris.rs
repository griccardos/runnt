//Data from UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

use runnt::{
    dataset::{Conversion, Dataset},
    nn::{get_report, ReportMetric, NN},
};

// Example of how to use `Dataset` and classification
pub fn main() {
    fastrand::seed(1);
    let set = Dataset::builder()
        .read_csv("examples/data/iris.csv")
        .add_input_columns(&[0, 1, 2, 3], Conversion::NormaliseMean)
        .add_target_columns(&[4], Conversion::OneHot)
        .allocate_to_test_data(0.4)
        .build();

    println!(
        "inputs: {:?} targets: {:?}",
        set.input_labels(),
        set.target_labels()
    );

    let mut net = NN::new(&[set.input_size(), 32, set.target_size()]).with_learning_rate(0.15);
    for e in 1..=12000 {
        let (ins, tars) = set.get_data();
        net.fit(&ins, &tars, 10);

        if e % 1000 == 0 {
            let err = net.forward_errors(&ins, &tars);

            let (test_ins, test_tars) = set.get_test_data();
            let test_err = net.forward_errors(&test_ins, &test_tars);

            let correct = get_report(&ReportMetric::CorrectClassification, &mut net, &ins, &tars);

            let test_correct = get_report(
                &ReportMetric::CorrectClassification,
                &mut net,
                &test_ins,
                &test_tars,
            );

            println!("{e} train mse: {err} test mse: {test_err} train correct:{correct} test correct:{test_correct}");
        }
    }
}
