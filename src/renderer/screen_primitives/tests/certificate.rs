use super::super::certificate::Certificate;
use super::*;
use crate::renderer::{CameraUniform, LogicalViewport, screen_camera};

fn screen_uniform(width: f32, height: f32) -> CameraUniform {
    let viewport = LogicalViewport::new(width, height).unwrap();
    CameraUniform::new(screen_camera(viewport).unwrap(), viewport).unwrap()
}

#[test]
fn certificate_acceptance_implies_independent_per_corner_interval_acceptance() {
    let mut seed = 0x92ec_6754_992b_f731_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 32) as u32
    };
    let mut accepted = 0;
    let mut fallback = 0;
    for index in 0..24_000 {
        let magnitudes = [0.00001, 0.01, 1.0, 100.0, 1_000.0, 10_000.0, 1_000_000.0];
        let magnitude = magnitudes[index % magnitudes.len()];
        let x = (next() as f64 / f64::from(u32::MAX) * 2.0 - 1.0) as f32 * magnitude;
        let y = (next() as f64 / f64::from(u32::MAX) * 2.0 - 1.0) as f32 * magnitude;
        let width = [0.0001, 0.1, 0.5, 1.0, 2.0, 4.0, 20.0, 100.0][index % 8];
        let primitive = match index % 3 {
            0 => ScreenPrimitive2d::circle(
                LogicalScreenPosition::new(x, y),
                LogicalPixels::new(width).unwrap(),
                Color::WHITE,
            ),
            1 => ScreenPrimitive2d::round_line(
                LogicalScreenPosition::new(x, y),
                LogicalScreenPosition::new(x + 17.25, y - 6.75),
                LogicalPixels::new(width).unwrap(),
                Color::WHITE,
            ),
            _ => ScreenPrimitive2d::rect(
                LogicalScreenPosition::new(x, y),
                LogicalScreenVector::new(width, width * 3.0),
                Color::WHITE,
            ),
        };
        let Ok(primitive) = primitive else { continue };
        let certificate = Certificate::new(&[primitive.gpu]);
        let viewport = [
            (1280.0, 800.0),
            (853.3333, 533.3333),
            (3840.0, 2160.0),
            (16.0, 16.0),
            (131072.0, 131072.0),
        ][index % 5];
        let mut uniform = screen_uniform(viewport.0, viewport.1);
        if index % 11 == 0 {
            uniform.screen_to_clip[2] = 0.5;
            uniform.screen_to_clip[3] = -0.5;
        }
        if certificate.accepts(uniform) {
            accepted += 1;
            assert!(
                validation::instances_are_safe(&[primitive.gpu], uniform),
                "index={index}, primitive={primitive:?}"
            );
        } else {
            fallback += 1;
        }
    }
    assert!(accepted > 1_000, "accepted={accepted}");
    assert!(fallback > 1_000, "fallback={fallback}");
}

#[test]
fn certificate_falls_back_for_noncanonical_and_extreme_uniforms() {
    let primitive = ScreenPrimitive2d::circle(
        LogicalScreenPosition::new(-0.0, 0.0),
        LogicalPixels::new(4.0).unwrap(),
        Color::WHITE,
    )
    .unwrap();
    let certificate = Certificate::new(&[primitive.gpu]);
    let uniform = screen_uniform(1280.0, 800.0);
    assert!(certificate.accepts(uniform));
    for value in [f32::MAX, f32::MIN_POSITIVE, -0.0, f32::NAN, f32::INFINITY] {
        let mut changed = uniform;
        changed.screen_to_clip[0] = value;
        assert!(!certificate.accepts(changed));
    }
    let mut changed = uniform;
    changed.world_to_screen_x[0] = 1.00001;
    assert!(!certificate.accepts(changed));
}

#[test]
fn every_uniform_field_participates_in_certificate_or_independent_fallback() {
    let primitive = ScreenPrimitive2d::circle(
        LogicalScreenPosition::new(75.25, 48.5),
        LogicalPixels::new(4.0).unwrap(),
        Color::WHITE,
    )
    .unwrap();
    let certificate = Certificate::new(&[primitive.gpu]);
    let original = screen_uniform(1280.0, 800.0);
    for field in 0..16 {
        for value in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ] {
            let mut changed = original;
            let values: &mut [f32] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut changed));
            values[field] = value;
            if certificate.accepts(changed) {
                assert!(
                    validation::instances_are_safe(&[primitive.gpu], changed),
                    "field={field} value={value}"
                );
            }
            assert_eq!(
                validation::instances_are_safe_accelerated(&[primitive.gpu], certificate, changed),
                validation::instances_are_safe(&[primitive.gpu], changed),
                "field={field} value={value}",
            );
        }
    }
}

#[test]
fn newly_certified_two_to_eight_error_edge_separations_match_exact_interval_proof() {
    let mut newly_admitted = 0;
    for size in [128.0, 256.0, 640.0, 1280.0, 2560.0, 8192.0] {
        let uniform = screen_uniform(size, size * 0.75);
        for fraction in [-0.125, 0.125, 0.75] {
            for width in [0.5, 1.25, 2.0, 8.0] {
                for slope in 1..=32 {
                    let x = size * fraction + 0.375;
                    let y = size * fraction + 0.625;
                    let primitive = ScreenPrimitive2d::round_line(
                        LogicalScreenPosition::new(x, y),
                        LogicalScreenPosition::new(x + 10.0, y + slope as f32 * 0.0625),
                        LogicalPixels::new(width).unwrap(),
                        Color::WHITE,
                    )
                    .unwrap();
                    let certificate = Certificate::new(&[primitive.gpu]);
                    let ratio = certificate.edge_separation_ratio(uniform);
                    if ratio > 2.0 && ratio <= 8.0 && certificate.accepts(uniform) {
                        newly_admitted += 1;
                        assert!(
                            validation::instances_are_safe(&[primitive.gpu], uniform),
                            "size={size} fraction={fraction} width={width} slope={slope} separation={ratio}"
                        );
                    }
                }
            }
        }
    }
    assert!(newly_admitted > 300, "newly admitted={newly_admitted}");
}

#[test]
fn math_ellipse_fixture_certificate_hits_all_terminal_directions() {
    let uniform = screen_uniform(1280.0, 800.0);
    let mut accepted = 0;
    let mut fallback = 0;
    for center in [[100.375_f32, 100.625_f32], [1190.375, 730.625]] {
        for zoom in [0.92, 1.0, 1.08] {
            for segment in 0..13 {
                let angle = segment as f32 * std::f32::consts::TAU / 13.0;
                let next = (segment + 1) as f32 * std::f32::consts::TAU / 13.0;
                let primitive = ScreenPrimitive2d::round_line(
                    LogicalScreenPosition::new(
                        center[0] + angle.cos() * 17.0 * zoom,
                        center[1] + angle.sin() * 13.0 * zoom,
                    ),
                    LogicalScreenPosition::new(
                        center[0] + next.cos() * 17.0 * zoom,
                        center[1] + next.sin() * 13.0 * zoom,
                    ),
                    LogicalPixels::new(1.25).unwrap(),
                    Color::WHITE,
                )
                .unwrap();
                let certificate = Certificate::new(&[primitive.gpu]);
                if certificate.accepts(uniform) {
                    accepted += 1;
                } else {
                    fallback += 1;
                }
                assert!(validation::instances_are_safe_accelerated(
                    &[primitive.gpu],
                    certificate,
                    uniform
                ));
                assert!(validation::instances_are_safe(&[primitive.gpu], uniform));
            }
        }
    }
    assert_eq!((accepted, fallback), (78, 0));
    eprintln!("math ellipse certificate accepted={accepted}, fallback={fallback}");
}

#[test]
fn source_and_destination_precision_reject_visible_large_shape_distortion() {
    assert!(matches!(
        ScreenPrimitive2d::rect(
            LogicalScreenPosition::new(100.0, 100.0),
            LogicalScreenVector::new(1e10, 600.0),
            Color::WHITE
        ),
        Err(ScreenPrimitiveError::PrecisionLimit)
    ));
    assert!(matches!(
        ScreenPrimitive2d::round_line(
            LogicalScreenPosition::new(-1e10, 400.0),
            LogicalScreenPosition::new(100.0, 400.0),
            LogicalPixels::new(600.0).unwrap(),
            Color::WHITE
        ),
        Err(ScreenPrimitiveError::PrecisionLimit | ScreenPrimitiveError::InvalidGeometry)
    ));
    assert!(matches!(
        ScreenPrimitive2d::circle(
            LogicalScreenPosition::new(-1e8, -1e8),
            LogicalPixels::new(141_421_776.0).unwrap(),
            Color::WHITE
        ),
        Err(ScreenPrimitiveError::PrecisionLimit)
    ));
    let primitive = ScreenPrimitive2d::circle(
        LogicalScreenPosition::new(300.25, 250.5),
        LogicalPixels::new(2.5).unwrap(),
        Color::WHITE,
    )
    .unwrap();
    let quality = quality::Quality::new(&[primitive]);
    let uniform = screen_uniform(1280.0, 800.0);
    for scale in [1.0, 1.25, 1.5, 2.0, 4.0] {
        assert!(quality.accepts(uniform, scale));
    }
    for scale in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, 100.0] {
        assert!(!quality.accepts(uniform, scale));
    }
}

#[test]
fn short_capsules_keep_nonintegral_ratios_inside_the_precision_envelope() {
    for index in 1..=2000 {
        let length = 0.0001 + (index as f32 / 2000.0);
        let width = 1.0 + (index % 20) as f32;
        ScreenPrimitive2d::round_line(
            LogicalScreenPosition::new(10.0, 10.0),
            LogicalScreenPosition::new(10.0 + length, 10.0),
            LogicalPixels::new(width).unwrap(),
            Color::WHITE,
        )
        .unwrap();
    }
}
