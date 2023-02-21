use criterion::{criterion_group, criterion_main, Bencher, Criterion};

criterion_group!(benches, forward, forward_and_backward);
criterion_main!(benches);

fn forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward and back");
    group.throughput(criterion::Throughput::Elements(1));
    group.bench_function("matrix", matrix_iterations_per_second);
}

fn forward_and_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward");
    group.throughput(criterion::Throughput::Elements(1));
    group.bench_function("matrix", matrix_iterations_per_second_forward);
}

fn matrix_iterations_per_second(b: &mut Bencher) {
    let mut nn = runnt::nn::NN::new(&[10, 100, 50, 10])
        .with_learning_rate(0.01)
        .with_hidden_type(runnt::activation::ActivationType::Sigmoid)
        .with_output_type(runnt::activation::ActivationType::Linear);

    b.iter(|| {
        nn.fit_one(
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
            &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
        )
    });
}

fn matrix_iterations_per_second_forward(b: &mut Bencher) {
    let nn = runnt::nn::NN::new(&[10, 100, 50, 10])
        .with_learning_rate(0.01)
        .with_hidden_type(runnt::activation::ActivationType::Sigmoid)
        .with_output_type(runnt::activation::ActivationType::Linear);

    b.iter(|| nn.forward(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]));
}
