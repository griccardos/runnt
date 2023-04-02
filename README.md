# runnt (*ru*st *n*eural *n*e*t*)
Very simple fully connected neural network.  
For when you just want to throw something together with minimal dependencies, and few lines of code.  
Aim is to create a fully connected network, run it on data, and get results in about 10 lines of code  
This library was created due to being unable to find a nice rust library which didn't have external dependencies, and was easy to use.

You are welcome to raise an issue or PR if you identify any errors or optimisations.

### Functionality:
- [X] fully connected neural network
- [X] minimal dependencies
- [X] no external static libraries/dlls required
- [X] regression and classfication
- [X] able to define layers sizes
- [X] able to define activation types
- [X] can save/load model
- [X] Stochastic, mini batch, gradient descent
- [X] Regularisation
- [X] Dataset manager 
    - [X] csv
    - [X] onehot encoding
    - [X] normalization 
- [X] Reporting

## How to use
### Simple example
All you need is NN and data
```rust
   //XOR
    use runnt::{nn::NN,activation::ActivationType};
    let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let outputs = [[0.], [1.], [1.], [0.]];

    let mut nn = NN::new(&[2, 8, 1])
        .with_learning_rate(0.2)
        .with_hidden_type(ActivationType::Tanh)
        .with_output_type(ActivationType::Linear);

    for i in 0..5000 {
        nn.fit_one(&inputs[i % 4], &outputs[i % 4]);
    }
```
### Simple example with Dataset and reporting
`Dataset` makes loading and transforming data a bit easier  
`run_and_report` makes running epochs and reporting easy  
Complete neural net with reporting in < 10 lines   
```rust
let set = Dataset::builder()
    .read_csv("examples/data/iris.csv")
    .add_input_columns(&[0, 1, 2, 3], Conversion::NormaliseMean)
    .add_target_columns(&[4], Conversion::OneHot)
    .allocate_to_test_data(0.2)
    .build();

    let mut net = NN::new(&[set.input_size(), 32, set.target_size()]).with_learning_rate(0.15);
    run_and_report(&set, &mut net, 1000, 8, 100, ReportAccuracy::CorrectClassification);
```

### With Dataset and reporting and save:
```rust
let set = Dataset::builder()
        .read_csv(r"/temp/diamonds.csv")
        .allocate_to_test_data(0.2)
        .add_input_columns(&[0, 4, 5, 7, 8, 9], Conversion::NormaliseMean)
        .add_input_columns(&[1, 2, 3], Conversion::OneHot)
        .add_target_columns(
            &[6],
            Conversion::Function(|f| f.parse::<f32>().unwrap_or_default() / 1_000.),
        )
        .build();

    let save_path = r"network.txt";
    let mut net = if std::path::PathBuf::from_str(save_path).unwrap().exists() {
        NN::load(save_path)
    } else {
        NN::new(&[set.input_size(), 32, set.target_size()])
    };
    //run for 100 epochs, with batch size 32 and report every 10 epochs
    run_and_report(&set, &mut net, 100, 32, 10, ReportAccuracy::RSquared);
    net.save(save_path);
```
