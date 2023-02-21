#[derive(Clone, Copy)]
pub enum InitializationType {
    ///-1 to 1
    Random,
    Xavier,
    Fixed(f32),
}

pub fn calc_initialization(typ: InitializationType, prev_layer_size: usize) -> f32 {
    match typ {
        InitializationType::Random => fastrand::f32() * 2. - 1.,
        InitializationType::Xavier => {
            (fastrand::f32() * 2. - 1.) * (1.0 / prev_layer_size as f32).sqrt()
        }
        InitializationType::Fixed(val) => val,
    }
}
