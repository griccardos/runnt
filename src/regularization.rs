pub enum Regularization {
    None,
    L1(f32),
    L2(f32),
    L1L2(f32, f32),
}
