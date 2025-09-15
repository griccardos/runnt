//Data from UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

use runnt::{
    dataset::{Conversion, Dataset},
    nn::{NN, ReportMetric},
};

// Example of how to use `Dataset` and classification
pub fn main() {
    fastrand::seed(1);
    let set = Dataset::builder()
        .read_csv("examples/data/iris.csv")
        .add_input_columns(&[0, 1, 2, 3], Conversion::NormaliseMean)
        .add_target_columns(&[4], Conversion::OneHot)
        .allocate_to_test_data(0.2)
        .build();

    println!(
        "inputs: {:?} targets: {:?}",
        set.input_labels(),
        set.target_labels()
    );

    let mut net = NN::new(&[set.input_size(), 32, set.target_size()])
        .with_loss(runnt::loss::Loss::SoftmaxAndCrossEntropy)
        .with_learning_rate(0.01);

    net.train(&set, 100, 1, 10, ReportMetric::CorrectClassification);
}
