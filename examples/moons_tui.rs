use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    symbols,
    widgets::{Axis, Block, Borders, Chart, Dataset as RtDataset, Paragraph, Wrap},
};
use runnt::{activation::Activation, dataset::Dataset as RnDataset, nn::ReportMetric};
use std::{error::Error, io, time::Duration, time::Instant};

// copy of generate_moons from examples/moons.rs
pub fn generate_moons() -> Vec<Vec<f32>> {
    let sine1 =
        |x: f32| (x * 2. + std::f32::consts::PI * 2. / 4.).sin() + fastrand::f32() / 2. - 0.2;
    let sine2 =
        |x: f32| (x * 2. + std::f32::consts::PI * 4. / 4.).sin() + fastrand::f32() / 2. - 0.2;

    let mut inp_out = vec![];
    (0..1000).for_each(|_| {
        let x = fastrand::f32() * 2. - 1.;
        let y = sine1(x);
        let cat = 0.;
        inp_out.push(vec![x, y, cat]);
    });
    (0..1000).for_each(|_| {
        let x = fastrand::f32() * 2.;
        let y = sine2(x);
        let cat = 1.;
        inp_out.push(vec![x, y, cat]);
    });
    inp_out
}

fn build_and_run() -> (Vec<(f64, f64, i32, i32)>, String) {
    fastrand::seed(Instant::now().elapsed().subsec_nanos() as u64);
    let inp_out = generate_moons();
    let set = RnDataset::builder()
        .add_data(&inp_out)
        .allocate_to_test_data(0.2)
        .add_input_columns(&[0, 1], runnt::dataset::Conversion::F32)
        .add_target_columns(&[2], runnt::dataset::Conversion::F32)
        .build();

    let mut nn = runnt::nn::NN::new(&[set.input_size(), 8, set.target_size()])
        .with_activation_hidden(Activation::Sigmoid)
        .with_activation_output(Activation::Linear)
        .with_loss(runnt::loss::Loss::BinaryCrossEntropy)
        .with_learning_rate(0.05);
    let (inp_test, tar_test) = set.get_test_data();
    nn.train(&set, 20, 1, 0, ReportMetric::CorrectClassification);
    let log = format!(
        "Correct classifiction: {} (Try rerunning to see changes)",
        nn.report(&ReportMetric::CorrectClassification, &inp_test, &tar_test)
    );

    let mut pts = vec![];
    for x in set.get_test_data_zip().iter().take(400) {
        let pred = nn.forward(x.0);
        let tx = x.0[0] as f64;
        let ty = x.0[1] as f64;
        let tar = if x.1[0] as i32 > 0 { 1 } else { 0 };
        let pr = if pred[0] > 0.5 { 1 } else { 0 };
        pts.push((tx, ty, tar, pr));
    }

    (pts, log)
}

fn main() -> Result<(), Box<dyn Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (mut points, mut log) = build_and_run();

    loop {
        terminal.draw(|f| {
            let size = f.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(10)].as_ref())
                .split(size);

            let top = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(30), Constraint::Min(10)].as_ref())
                .split(chunks[0]);

            let controls = Paragraph::new("Press (r)erun  (q)uit")
                .block(Block::default().borders(Borders::ALL).title("Controls"));
            f.render_widget(controls, top[0]);

            let output = Paragraph::new(log.clone())
                .block(Block::default().borders(Borders::ALL).title("Output"))
                .wrap(Wrap { trim: true });
            f.render_widget(output, top[1]);

            let (s0, s1, s_err): (Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<(f64, f64)>) = {
                let mut s0: Vec<(f64, f64)> = vec![];
                let mut s1: Vec<(f64, f64)> = vec![];
                let mut s_err: Vec<(f64, f64)> = vec![];
                for (x, y, tar, pr) in &points {
                    if *tar == 0 && *tar == *pr {
                        s0.push((*x, *y))
                    } else if *tar == 1 && *tar == *pr {
                        s1.push((*x, *y))
                    } else {
                        s_err.push((*x, *y))
                    }
                }
                (s0, s1, s_err)
            };
            let datasets = vec![
                RtDataset::default()
                    .name("class0")
                    .marker(symbols::Marker::Dot)
                    .style(Style::default().fg(Color::Green))
                    .data(&s0),
                RtDataset::default()
                    .name("class1")
                    .marker(symbols::Marker::Dot)
                    .style(Style::default().fg(Color::Blue))
                    .data(&s1),
                RtDataset::default()
                    .name("mismatch")
                    .marker(symbols::Marker::Braille)
                    .style(Style::default().fg(Color::Red))
                    .data(&s_err),
            ];

            let chart = Chart::new(datasets)
                .block(Block::default().borders(Borders::ALL).title("Scatter"))
                .x_axis(Axis::default().bounds([-1.5, 3.5]).labels(vec![
                    "-1.5".into(),
                    "0".into(),
                    "1.0".into(),
                    "2.5".into(),
                ]))
                .y_axis(Axis::default().bounds([-2.0, 2.0]).labels(vec![
                    "-2.0".into(),
                    "-1.0".into(),
                    "0".into(),
                    "1.0".into(),
                    "2.0".into(),
                ]));
            f.render_widget(chart, chunks[1]);
        })?;

        if event::poll(Duration::from_millis(200))? {
            match event::read()? {
                Event::Key(key) => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        break;
                    }
                    KeyCode::Char('r') => {
                        let (pts, lg) = build_and_run();
                        points = pts;
                        log = lg;
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}
