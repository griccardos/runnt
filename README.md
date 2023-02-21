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

## How to use
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

