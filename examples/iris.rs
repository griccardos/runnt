//Data from UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

use runnt::{
    dataset::{Adjustment, Dataset},
    nn::{self, NN},
};

// Example of how to use `Dataset` and classification
pub fn main() {
    fastrand::seed(1);
    let set = Dataset::builder()
        .read_csv("examples/data/iris.csv")
        .add_input_columns(&[0, 1, 2, 3], Adjustment::NormaliseMean)
        .add_target_columns(&[4], Adjustment::OneHot)
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
        net.fit_batch_size(&ins, &tars, 10);

        if e % 1000 == 0 {
            let err = net.forward_errors(&ins, &tars);

            let (test_ins, test_tars) = set.get_test_data();
            let test_err = net.forward_errors(&test_ins, &test_tars);

            let correct = acc(&net, &ins, &tars);
            let test_correct = acc(&net, &test_ins, &test_tars);

            println!("{e} train mse: {err} test mse: {test_err} train correct:{correct}% test correct:{test_correct}%");
        }
    }
}

fn acc(net: &NN, ins: &Vec<&Vec<f32>>, tars: &Vec<&Vec<f32>>) -> f32 {
    let mut correct = 0;
    for (ins, tars) in ins.iter().zip(tars) {
        let pred = net.forward(&ins);
        if nn::max_index_equal(&tars, &pred) {
            correct += 1;
        }
    }
    let correct = 100. * correct as f32 / ins.len() as f32;
    correct
}
