//! Deterministic CPU fallback measurement for particle-instance preparation.
//!
//! This intentionally does not create a GPU device or window. It measures the
//! host-side cost of generating and validating `ParticleInstance2d` values, so
//! results remain useful on CI and machines without a presentation surface.

use std::{hint::black_box, time::Instant};

use sim_engine::{Color, ParticleInstance2d, Vec2};

const DEFAULT_PARTICLE_COUNT: usize = 10_000;
const DEFAULT_FRAME_COUNT: usize = 300;
const MAX_PARTICLE_COUNT: usize = 1_000_000;

fn main() {
    let particle_count = bounded_env_usize(
        "SIM_ENGINE_PARTICLE_CPU_BENCHMARK_COUNT",
        DEFAULT_PARTICLE_COUNT,
    );
    let frame_count = bounded_env_usize(
        "SIM_ENGINE_PARTICLE_CPU_BENCHMARK_FRAMES",
        DEFAULT_FRAME_COUNT,
    );
    let started_at = Instant::now();
    let mut checksum = 0.0_f32;

    for frame in 0..frame_count {
        let time_seconds = frame as f32 * (1.0 / 60.0);
        for index in 0..particle_count {
            let particle = particle_instance(index, time_seconds);
            checksum += particle.world_position().x()
                + particle.world_position().y()
                + particle.radius()
                + particle.depth()
                + particle.color().alpha();
            black_box(particle);
        }
    }

    let elapsed = started_at.elapsed();
    let instance_count = particle_count * frame_count;
    let nanoseconds_per_instance = elapsed.as_nanos() as f64 / instance_count as f64;
    let instances_per_second = instance_count as f64 / elapsed.as_secs_f64();
    println!(
        "particle CPU benchmark: {particle_count} particles x {frame_count} frames = {instance_count} validated instances; {nanoseconds_per_instance:.1} ns/instance, {instances_per_second:.0} instances/s, checksum={checksum:.3}"
    );
}

fn bounded_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=MAX_PARTICLE_COUNT).contains(value))
        .unwrap_or(default)
}

fn particle_instance(index: usize, time_seconds: f32) -> ParticleInstance2d {
    let phase = index as f32 * 0.618_034;
    let ring = 24.0 + (index % 80) as f32 * 2.7;
    let angle = phase + time_seconds * (0.35 + (index % 7) as f32 * 0.03);
    ParticleInstance2d::new(
        Vec2::new(
            angle.cos() * ring + (time_seconds * 0.7 + phase).sin() * 18.0,
            angle.sin() * ring * 0.56 + (time_seconds * 0.45 + phase).cos() * 12.0,
        ),
        1.5 + (index % 5) as f32 * 0.35,
        Color::rgba(0.2, 0.6, 0.9, 0.75),
        (index % 9) as f32 * 0.15,
    )
    .expect("deterministic benchmark particle is finite")
}
