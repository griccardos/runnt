use std::time::Duration;

use runnt::{
    dataset::{Conversion, Dataset},
    nn::{self, ReportMetric, NN},
};

//Example of: Classification, Dataset, train, report
pub fn main() {
    //generate circular data, everything outside the circle is 0, and inside is 1
    //circle has radius 5
    let classify_circle = |x: f32, y: f32| {
        if (x.powi(2) + y.powi(2)).sqrt() < 5. {
            1.
        } else {
            0.
        }
    };

    //get some random points
    let data: Vec<Vec<f32>> = (0..1000)
        .map(|_| {
            let x = fastrand::f32() * 10. - 5.;
            let y = fastrand::f32() * 10. - 5.;
            let is_in_circle = classify_circle(x, y);
            vec![x, y, is_in_circle]
        })
        .collect();
    //convert to dataset
    let set = Dataset::builder()
        .add_data(&data)
        .allocate_to_test_data(0.2)
        .add_input_columns_range(0..=1, Conversion::F32)
        .add_target_columns(&[2], Conversion::OneHot)
        .build();

    //create network
    let mut net = NN::new(&[set.input_size(), 16, set.target_size()]).with_learning_rate(0.03);

    //run reporting the mse
    net.train(&set, 500, 1, 50, ReportMetric::CorrectClassification);
    println!("Generating test data to plot...");
    std::thread::sleep(Duration::from_secs(2));
    println!("x,y,tar,predicted");
    set.get_test_data_zip().iter().for_each(|x| {
        let pred = net.forward(x.0);
        println!(
            "{},{},{},{}",
            x.0[0],
            x.0[1],
            nn::max_index(x.1),
            nn::max_index(&pred),
        );
    });
}
