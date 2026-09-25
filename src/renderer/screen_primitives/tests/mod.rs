use super::*;
use crate::{Color, LogicalPixels, LogicalScreenPosition, LogicalScreenVector};

mod certificate;
mod gpu;
mod metadata;
mod targets;
pub(in crate::renderer) use gpu::verify_analytic_primitives;

#[test]
fn conversion_preserves_order_clips_and_rejects_unsupported_commands() {
    use crate::{ScreenScene, ShapeStyle};
    let mut scene = ScreenScene::new(Color::BLACK).unwrap();
    scene
        .try_circle(
            LogicalScreenPosition::new(30.25, 20.5),
            LogicalPixels::new(4.0).unwrap(),
            ShapeStyle::filled(Color::WHITE.with_alpha(0.5)),
        )
        .unwrap();
    scene
        .try_square_rect(
            LogicalScreenPosition::new(17.0, 41.0),
            LogicalScreenVector::new(8.0, 9.0),
            ShapeStyle::filled(Color::WHITE),
        )
        .unwrap();
    let primitives = ScreenPrimitive2d::from_screen_scene(&scene).unwrap();
    assert_eq!(primitives.len(), 2);
    assert_eq!(primitives[0].gpu.anchor, [30.25, -20.5]);
    assert_eq!(primitives[0].gpu.color[3], 0.5);
    assert_eq!(primitives[1].gpu.anchor, [21.0, -45.5]);
    scene
        .try_circle(
            LogicalScreenPosition::new(30.25, 20.5),
            LogicalPixels::new(4.0).unwrap(),
            ShapeStyle::stroked(1.0, Color::WHITE),
        )
        .unwrap();
    assert!(matches!(
        ScreenPrimitive2d::from_screen_scene(&scene),
        Err(ScreenPrimitiveError::UnsupportedCommand { index: 2 })
    ));
}

#[test]
fn analytic_quad_proof_is_viewport_specific_and_preserves_extreme_rejections() {
    use crate::renderer::{CameraUniform, LogicalViewport, screen_camera};
    let primitive = ScreenPrimitive2d::circle(
        LogicalScreenPosition::new(30.25, 20.5),
        LogicalPixels::new(4.0).unwrap(),
        Color::WHITE,
    )
    .unwrap();
    for size in [(1280.0, 800.0), (853.3333, 533.3333), (3840.0, 2160.0)] {
        let viewport = LogicalViewport::new(size.0, size.1).unwrap();
        let uniform = CameraUniform::new(screen_camera(viewport).unwrap(), viewport).unwrap();
        assert!(validation::instances_are_safe(&[primitive.gpu], uniform));
        let mut overflow = uniform;
        overflow.screen_to_clip[0] = f32::MAX;
        assert!(!validation::instances_are_safe(&[primitive.gpu], overflow));
        let mut unsupported_camera = uniform;
        unsupported_camera.world_to_screen_x[0] = 2.0;
        assert!(!validation::instances_are_safe(
            &[primitive.gpu],
            unsupported_camera
        ));
    }
}

#[test]
#[ignore = "release CPU analytic proof diagnostic; excludes GPU and presentation"]
fn analytic_primitives_100_500_2000_cpu_diagnostic() {
    use crate::renderer::{CameraUniform, LogicalViewport, screen_camera};
    use std::{hint::black_box, time::Instant};
    if cfg!(debug_assertions) {
        panic!("diagnostic requires --release");
    }
    let viewport = LogicalViewport::new(1280.0, 800.0).unwrap();
    let uniform = CameraUniform::new(screen_camera(viewport).unwrap(), viewport).unwrap();
    for fixture in ["two_primitives", "math_mixed", "math_ellipse"] {
        for count in [100, 500, 2000] {
            let mut instances = Vec::new();
            for index in 0..count {
                let x = (index % 50) as f32 * 20.0 + 20.0;
                let y = (index / 50) as f32 * 14.0 + 20.0;
                if fixture == "math_ellipse" {
                    for segment in 0..13 {
                        let angle = segment as f32 * std::f32::consts::TAU / 13.0;
                        let next = (segment + 1) as f32 * std::f32::consts::TAU / 13.0;
                        instances.push(
                            ScreenPrimitive2d::round_line(
                                LogicalScreenPosition::new(
                                    x + angle.cos() * 17.0,
                                    y + angle.sin() * 13.0,
                                ),
                                LogicalScreenPosition::new(
                                    x + next.cos() * 17.0,
                                    y + next.sin() * 13.0,
                                ),
                                LogicalPixels::new(1.25).unwrap(),
                                Color::WHITE,
                            )
                            .unwrap()
                            .gpu,
                        );
                    }
                    for offset in [-9.0, 0.0, 9.0] {
                        instances.push(
                            ScreenPrimitive2d::circle(
                                LogicalScreenPosition::new(x + offset, y + 3.5),
                                LogicalPixels::new(4.0).unwrap(),
                                Color::WHITE,
                            )
                            .unwrap()
                            .gpu,
                        );
                        instances.push(
                            ScreenPrimitive2d::rect(
                                LogicalScreenPosition::new(x + offset - 3.125, y - 9.375),
                                LogicalScreenVector::new(6.25, 3.75),
                                Color::WHITE,
                            )
                            .unwrap()
                            .gpu,
                        );
                    }
                    continue;
                }
                instances.push(
                    ScreenPrimitive2d::circle(
                        LogicalScreenPosition::new(x, y),
                        LogicalPixels::new(2.0).unwrap(),
                        Color::WHITE,
                    )
                    .unwrap()
                    .gpu,
                );
                instances.push(
                    ScreenPrimitive2d::round_line(
                        LogicalScreenPosition::new(x, y),
                        LogicalScreenPosition::new(x + 13.0, y + 5.0),
                        LogicalPixels::new(2.0).unwrap(),
                        Color::WHITE,
                    )
                    .unwrap()
                    .gpu,
                );
                if fixture == "math_mixed" {
                    for offset in 1..=12 {
                        let offset = offset as f32 * 0.25;
                        instances.push(
                            ScreenPrimitive2d::round_line(
                                LogicalScreenPosition::new(x + offset, y + offset),
                                LogicalScreenPosition::new(x + 13.0 + offset, y + 5.0 + offset),
                                LogicalPixels::new(2.0).unwrap(),
                                Color::WHITE,
                            )
                            .unwrap()
                            .gpu,
                        );
                    }
                    for offset in [3.0, 6.0] {
                        instances.push(
                            ScreenPrimitive2d::circle(
                                LogicalScreenPosition::new(x + offset, y),
                                LogicalPixels::new(2.0).unwrap(),
                                Color::WHITE,
                            )
                            .unwrap()
                            .gpu,
                        );
                    }
                    for offset in [0.0, 3.0, 6.0] {
                        instances.push(
                            ScreenPrimitive2d::rect(
                                LogicalScreenPosition::new(x + offset, y + 7.0),
                                LogicalScreenVector::new(2.0, 3.0),
                                Color::WHITE,
                            )
                            .unwrap()
                            .gpu,
                        );
                    }
                }
            }
            let certificate = super::certificate::Certificate::new(&instances);
            let fast = certificate.accepts(uniform);
            let primitive_fast_count = instances
                .iter()
                .filter(|instance| {
                    super::certificate::Certificate::new(std::slice::from_ref(*instance))
                        .accepts(uniform)
                })
                .count();
            let operation = || {
                validation::instances_are_safe_accelerated(
                    black_box(&instances),
                    black_box(certificate),
                    black_box(uniform),
                )
            };
            assert!(operation());
            let (_, allocations) = crate::test_allocations::count(operation);
            assert_eq!(allocations, 0);
            let mut samples = Vec::new();
            let mut rebuild_samples = Vec::new();
            for index in 0..12 {
                let start = Instant::now();
                assert!(black_box(operation()));
                if index >= 2 {
                    samples.push(start.elapsed().as_secs_f64() * 1000.0);
                }
                let start = Instant::now();
                black_box(super::certificate::Certificate::new(black_box(&instances)));
                if index >= 2 {
                    rebuild_samples.push(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
            samples.sort_by(f64::total_cmp);
            rebuild_samples.sort_by(f64::total_cmp);
            println!(
                "analytic_primitives fixture={fixture} constructions={count} primitives={} vertices={} upload_bytes={} samples=10 certificate_accepted={fast} primitive_fast_count={primitive_fast_count} validation_p50_p95_p99_ms={:.6}/{:.6}/{:.6} certificate_rebuild_p50_p95_p99_ms={:.6}/{:.6}/{:.6} allocations={allocations}",
                instances.len(),
                instances.len() * 6,
                instances.len() * 64,
                samples[5],
                samples[9],
                samples[9],
                rebuild_samples[5],
                rebuild_samples[9],
                rebuild_samples[9]
            );
        }
    }
}

#[test]
fn compact_layout_and_construction_rejections() {
    assert_eq!(std::mem::size_of::<ScreenPrimitiveGpu>(), 64);
    let point = LogicalScreenPosition::new(12.5, 3.25);
    let length = LogicalPixels::new(2.0).unwrap();
    assert!(ScreenPrimitive2d::circle(point, length, Color::WHITE).is_ok());
    assert!(ScreenPrimitive2d::round_line(point, point, length, Color::WHITE).is_err());
    assert!(
        ScreenPrimitive2d::rect(point, LogicalScreenVector::new(-1.0, 2.0), Color::WHITE).is_err()
    );
    assert!(
        ScreenPrimitive2d::circle(point, length, Color::rgba(f32::NAN, 0.0, 0.0, 1.0)).is_err()
    );
}
