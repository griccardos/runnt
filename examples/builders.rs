use runnt::prelude::*;
use runnt::sede::Sede;
pub fn main() {
    //Simple builder
    let nn1 = NN::new(&[1, 2, 3, 4]) //give full network shape up front
        .with_activation_hidden(Activation::Sigmoid) //apply to all hidden layers
        .with_activation_output(Activation::Sigmoid) //apply to output only
        .with_regularization(Regularization::None) //apply to all layers
        .with_initialization(Initialization::Fixed(0.)); //apply to all layers

    //Complex builder
    //allows us to specify each layer individually
    let nn2 = NN::new_input(1) //start with input size
        .layer(2) //add layer with number assuming type = dense
        .layer(dense(3)) //or specify dense
        .layer(
            dense(4)
                .activation(Activation::Sigmoid) //configure activation for this layer
                .regularization(Regularization::None) //configure regularization for this layer
                .initializer(Initialization::Fixed(0.)), //configure initialization for this layer
        )
        .with_initialization(Initialization::Fixed(0.)); //can still override initialization for all layers
    assert_eq!(nn1.serialize(), nn2.serialize());
    println!("{nn1}\nis equal to\n{nn2}");
}
