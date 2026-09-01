use std::{env, process::ExitCode, time::Instant};

use sim_engine::{Color, DrawCommand, Layer, Scene, SceneBudget, ShapeStyle, Vec2};

const DEFAULT_COMMANDS: usize = 100_000;
const DEFAULT_ITERATIONS: usize = 5;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("scene construction benchmark failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut command_count = DEFAULT_COMMANDS;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--commands" => command_count = parse_value(args.next(), "--commands")?,
            "--iterations" => iterations = parse_value(args.next(), "--iterations")?,
            "--help" | "-h" => {
                println!("Usage: scene_construction_benchmark [--commands N] [--iterations N]");
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if command_count == 0 || iterations == 0 {
        return Err("command and iteration counts must be non-zero".into());
    }

    let template = DrawCommand::circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE))
        .map_err(debug_error)?;
    let mut probe = Scene::new(Color::BLACK).map_err(debug_error)?;
    probe.try_push(template.clone()).map_err(debug_error)?;
    let per_command = probe.statistics();
    let vertices = command_count.saturating_mul(per_command.estimated_tessellated_vertices());
    let budget = SceneBudget::new(
        command_count,
        0,
        vertices,
        command_count.saturating_mul(per_command.retained_bytes()),
        command_count
            .saturating_mul(per_command.retained_bytes())
            .saturating_mul(2),
        command_count.saturating_mul(per_command.estimated_upload_bytes()),
        command_count.saturating_mul(per_command.estimated_draw_batches()),
    );
    let mut samples = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let mut scene = Scene::with_budget(Color::BLACK, budget).map_err(debug_error)?;
        let started = Instant::now();
        let commands = (0..command_count).map(|index| {
            let layer = Layer::new(if index.is_multiple_of(2) {
                Layer::FOREGROUND.order()
            } else {
                Layer::BACKGROUND.order()
            });
            (layer, template.clone())
        });
        scene.try_extend_to_layers(commands).map_err(debug_error)?;
        let elapsed = started.elapsed();
        if scene.command_count() != command_count {
            return Err("accepted command count differs from the requested count".into());
        }
        println!(
            "iteration {}: {} mixed-layer commands in {:.3} ms",
            iteration + 1,
            command_count,
            elapsed.as_secs_f64() * 1_000.0
        );
        samples.push(elapsed.as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    println!(
        "median: {:.3} ms ({:.0} commands/s)",
        median * 1_000.0,
        command_count as f64 / median
    );
    Ok(())
}

fn parse_value(value: Option<String>, flag: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse()
        .map_err(|_| format!("{flag} requires a positive integer"))
}

fn debug_error(error: impl std::fmt::Debug) -> String {
    format!("{error:?}")
}
