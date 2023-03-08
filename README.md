# runnt (*ru*st *n*eural *n*e*t*)
Very simple fully connected neural network.

For when you just want to throw something together with minimal dependencies, and a few lines of code.

Created this since I struggled to find an nice rust library which didn't have external dependencies, and was easy to use.

You are welcome to raise an issue or PR if you identify any errors or optimisations.

### Functionality:
- [X] fully connected neural network
- [X] minimal dependencies
- [X] no external static libraries/dlls required
- [X] able to define layers sizes
- [X] able to define activation types
- [X] can save/load model
- [X] Stochastic, mini batch, gradient descent
- [X] Dataset manager 
    - [X] csv
    - [X] onehot encoding
    - [X] normalization 
    
## How to use
### Simple example
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
    assert_eq!(nn.forward(&[0., 0.]).first().unwrap().round(), 0.);
    assert_eq!(nn.forward(&[0., 1.]).first().unwrap().round(), 1.);
    assert_eq!(nn.forward(&[1., 0.]).first().unwrap().round(), 1.);
    assert_eq!(nn.forward(&[1., 1.]).first().unwrap().round(), 0.);
```
### With Dataset and runner:
```rust
 let set = Dataset::builder()
        .read_csv(r"diamonds.csv")
        .allocate_to_test_data(0.2)
        .add_input_columns(&[0, 4, 5, 7, 8, 9], Adjustment::NormaliseMean)
        .add_input_columns(&[1, 2, 3], Adjustment::OneHot)
        .add_target_columns(
            &[6],
            Adjustment::Function(|f| f.parse::<f32>().unwrap_or_default() / 1_000.),
        )
        .build();

    let save_path = r"network.txt";
    let mut net = if PathBuf::from_str(save_path).unwrap().exists() {
        NN::load(save_path)
    } else {
        NN::new(&[set.input_size(), 32, set.target_size()])
    };
    //run for 1000 epochs, with batch size 32 and report mse every 10 epochs
    nn::run_and_report(&set, &mut net, 1000, 32, Some(10),false);
    net.save(save_path);
```
