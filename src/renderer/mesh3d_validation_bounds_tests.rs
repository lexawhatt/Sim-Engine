//! Independent per-vertex comparisons for the sufficient Native bounds proof.

use super::*;
use bounds::native_transform_is_proven;

const IDENTITY_MODEL: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];
const IDENTITY_CAMERA: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn source(extra: impl IntoIterator<Item = Vec3>) -> Mesh3d {
    let mut vertices = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
    vertices.extend(extra);
    Mesh3d::new(vertices, vec![0, 1, 2]).unwrap()
}

// Preserve the original source-order point loop as the independent oracle.
// It does not call the aggregate proof or consume its cached metadata.
fn reference(
    mesh: &Mesh3d,
    model: [[f32; 4]; 3],
    camera: [[f32; 4]; 4],
) -> Result<(), (usize, Mesh3dSurfaceError)> {
    for (index, point) in mesh.vertices().iter().enumerate() {
        validate_point_for_policy(*point, model, camera, SurfaceRasterization3d::Native)
            .map_err(|error| (index, error))?;
    }
    Ok(())
}

fn checked_proof(mesh: &Mesh3d, model: [[f32; 4]; 3], camera: [[f32; 4]; 4]) -> bool {
    let proven = native_transform_is_proven(mesh, model, camera);
    let expected = reference(mesh, model, camera);
    if proven {
        assert_eq!(
            expected,
            Ok(()),
            "unsafe bounds acceptance: model={model:?}, camera={camera:?}, vertices={:?}",
            mesh.vertices()
        );
    }
    proven
}

#[test]
fn native_bounds_proof_detects_interior_and_unused_subnormal_sources() {
    for tiny in [f32::from_bits(1), f32::from_bits(0x007f_ffff)] {
        let mesh = source([
            Vec3::new(-1.0, -1.0, -1.0).unwrap(),
            Vec3::new(1.0, 1.0, 1.0).unwrap(),
            Vec3::new(tiny, 0.0, 0.0).unwrap(),
        ]);
        assert_eq!(mesh.bounds_min().x(), -1.0);
        assert_eq!(mesh.bounds_max().x(), 1.0);
        assert_eq!(mesh.minimum_nonzero_components()[0], tiny);
        assert!(!checked_proof(&mesh, IDENTITY_MODEL, IDENTITY_CAMERA));
        assert_eq!(
            reference(&mesh, IDENTITY_MODEL, IDENTITY_CAMERA),
            Err((5, Mesh3dSurfaceError::TransformArithmetic))
        );
    }
}

#[test]
fn native_bounds_proof_preserves_zero_normal_gap_and_exact_selectors() {
    let mesh = source([
        Vec3::new(-1.0, 1.0, 0.0).unwrap(),
        Vec3::new(f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 0.0).unwrap(),
    ]);
    assert!(checked_proof(&mesh, IDENTITY_MODEL, IDENTITY_CAMERA));
    let permutation = [
        [0.0, -1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -4.0],
    ];
    assert!(checked_proof(&mesh, permutation, IDENTITY_CAMERA));
    let mut underflow = IDENTITY_MODEL;
    underflow[0][0] = 0.5;
    assert!(!checked_proof(&mesh, underflow, IDENTITY_CAMERA));
    assert!(reference(&mesh, underflow, IDENTITY_CAMERA).is_err());
}

#[test]
fn native_bounds_proof_preserves_underflow_absorbed_by_model_or_camera_translation() {
    let mesh = source([Vec3::new(f32::MIN_POSITIVE, 0.0, 0.0).unwrap()]);
    let mut model = IDENTITY_MODEL;
    model[0] = [0.5, 0.0, 0.0, 1.0];
    assert!(!checked_proof(&mesh, model, IDENTITY_CAMERA));
    assert_eq!(
        reference(&mesh, model, IDENTITY_CAMERA),
        Err((3, Mesh3dSurfaceError::TransformArithmetic))
    );
    let mut camera = IDENTITY_CAMERA;
    camera[0] = [0.5, 0.0, 0.0, 1.0];
    assert!(!checked_proof(&mesh, IDENTITY_MODEL, camera));
    assert_eq!(
        reference(&mesh, IDENTITY_MODEL, camera),
        Err((3, Mesh3dSurfaceError::TransformArithmetic))
    );
}

#[test]
fn native_bounds_fallback_preserves_original_source_error_attribution() {
    let mesh = source([
        Vec3::new(-1.0, -1.0, 0.0).unwrap(),
        Vec3::new(1.0, 1.0, 0.0).unwrap(),
        Vec3::new(f32::from_bits(1), 0.5, 0.0).unwrap(),
    ]);
    let (index, reason) = reference(&mesh, IDENTITY_MODEL, IDENTITY_CAMERA).unwrap_err();
    assert!(!native_transform_is_proven(
        &mesh,
        IDENTITY_MODEL,
        IDENTITY_CAMERA
    ));
    assert_eq!(
        source_point_error(&mesh, index, reason),
        Mesh3dObjectError::Vertex {
            vertex_index: 5,
            reason: Mesh3dSurfaceError::TransformArithmetic,
        }
    );
    let referenced = Mesh3d::new(mesh.vertices().to_vec(), vec![0, 1, 2, 3, 4, 5]).unwrap();
    assert_eq!(
        source_point_error(&referenced, index, reason),
        Mesh3dObjectError::SurfaceTriangle {
            triangle_index: 1,
            reason: Mesh3dSurfaceError::TransformArithmetic,
        }
    );
}

#[test]
fn native_bounds_proof_does_not_hide_invalid_zero_row_coefficients() {
    let mesh = source([]);
    for coefficient in [
        f32::from_bits(1),
        -f32::from_bits(0x007f_ffff),
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ] {
        let mut model = IDENTITY_MODEL;
        model[0][2] = coefficient;
        assert!(!checked_proof(&mesh, model, IDENTITY_CAMERA));
        assert!(reference(&mesh, model, IDENTITY_CAMERA).is_err());
        let mut camera = IDENTITY_CAMERA;
        camera[0][2] = coefficient;
        assert!(!checked_proof(&mesh, IDENTITY_MODEL, camera));
        assert!(reference(&mesh, IDENTITY_MODEL, camera).is_err());
    }
}

#[test]
fn native_bounds_proof_falls_back_on_model_cancellation_and_amplification() {
    let mesh = source([
        Vec3::new(1.0, -1.0, 0.0).unwrap(),
        Vec3::new(
            f32::MIN_POSITIVE * 2.0,
            -f32::from_bits((f32::MIN_POSITIVE * 2.0).to_bits() + 1),
            0.0,
        )
        .unwrap(),
    ]);
    let model = [
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    assert!(!checked_proof(&mesh, model, IDENTITY_CAMERA));
    assert_eq!(
        reference(&mesh, model, IDENTITY_CAMERA),
        Err((4, Mesh3dSurfaceError::TransformArithmetic))
    );

    let huge = 2.0_f32.powi(80);
    let amplified = source([Vec3::new(huge, -huge, 0.0).unwrap()]);
    let mut camera = IDENTITY_CAMERA;
    camera[0][0] = huge;
    assert!(!checked_proof(&amplified, model, camera));
    assert!(reference(&amplified, model, camera).is_err());
}

fn grid(side: usize) -> Mesh3d {
    let mut vertices = Vec::new();
    for y in 0..=side {
        for x in 0..=side {
            vertices.push(Vec3::new(x as f32 / side as f32, y as f32 / side as f32, 0.0).unwrap());
        }
    }
    let mut indices = Vec::new();
    for y in 0..side {
        for x in 0..side {
            let first = (y * (side + 1) + x) as u32;
            let next = first + side as u32 + 1;
            indices.extend([first, first + 1, next + 1, first, next + 1, next]);
        }
    }
    Mesh3d::new(vertices, indices).unwrap()
}

fn chunk_model(index: usize) -> [[f32; 4]; 3] {
    [
        [0.9, 0.0, 0.0, (index % 8) as f32 - 4.0],
        [0.0, 0.9, 0.0, (index / 8) as f32 - 4.0],
        [0.0, 0.0, 0.9, 0.0],
    ]
}

fn moving_camera(frame: usize) -> [[f32; 4]; 4] {
    let mut camera = IDENTITY_CAMERA;
    camera[0] = [0.125, 0.0, 0.0, (frame as f32 * 0.01).sin() * 0.01];
    camera[1][1] = 0.125;
    camera[2] = [0.0, 0.0, -0.01, 0.5];
    camera
}

#[test]
fn native_bounds_proof_accepts_translated_scaled_chunks_and_rotated_boxes() {
    for side in [16, 32] {
        let mesh = grid(side);
        for index in 0..64 {
            assert!(checked_proof(
                &mesh,
                chunk_model(index),
                moving_camera(index)
            ));
        }
    }
    let mesh = grid(8);
    let rotation = [
        [0.6, -0.8, 0.0, 2.0],
        [0.8, 0.6, 0.0, -3.0],
        [0.0, 0.0, 0.9, 0.0],
    ];
    assert!(checked_proof(&mesh, rotation, moving_camera(0)));
}

fn random_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    (*state >> 32) as u32
}

#[test]
fn native_bounds_proof_matches_reference_across_numeric_boundary_corpus() {
    let values = [
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        f32::from_bits(f32::MIN_POSITIVE.to_bits() + 1),
        f32::MIN_POSITIVE * 8.0,
        2.0_f32.powi(-100),
        2.0_f32.powi(-60),
        0.1,
        0.5,
        0.9,
        1.0,
        f32::from_bits(1.0_f32.to_bits() + 1),
        2.0,
        2.0_f32.powi(60),
        2.0_f32.powi(100),
        2.0_f32.powi(119),
        MAX_PORTABLE_SHADER_VALUE,
        f32::from_bits(MAX_PORTABLE_SHADER_VALUE.to_bits() + 1),
        f32::MAX,
    ];
    let mut proven = 0;
    for value in values {
        for sign in [-1.0, 1.0] {
            let mesh = source([Vec3::new(value * sign, value, -value).unwrap()]);
            for coefficient in values {
                let mut model = IDENTITY_MODEL;
                model[0][0] = coefficient;
                proven += usize::from(checked_proof(&mesh, model, IDENTITY_CAMERA));
                let mut camera = IDENTITY_CAMERA;
                camera[1][1] = coefficient;
                proven += usize::from(checked_proof(&mesh, IDENTITY_MODEL, camera));
            }
        }
    }
    let mut random = 0x8d1b_74a9_92f6_3c05;
    for iteration in 0..1024 {
        let mesh = source((0..12).map(|_| {
            let components: [f32; 3] = std::array::from_fn(|_| {
                let bits = random_u32(&mut random);
                let exponent = if iteration % 2 == 0 {
                    115 + bits % 25
                } else {
                    1 + bits % 247
                };
                f32::from_bits((bits & 0x807f_ffff) | (exponent << 23))
            });
            Vec3::new(components[0], components[1], components[2]).unwrap()
        }));
        let mut model = IDENTITY_MODEL;
        let mut camera = IDENTITY_CAMERA;
        for axis in 0..3 {
            if iteration % 2 == 0 {
                model[axis][axis] = 0.9;
                model[axis][3] = 16.0;
                camera[axis][axis] = 0.125;
            } else {
                model[axis][axis] = values[random_u32(&mut random) as usize % values.len()];
                model[axis][3] = values[random_u32(&mut random) as usize % values.len()];
                camera[axis][axis] = values[random_u32(&mut random) as usize % values.len()];
            }
        }
        proven += usize::from(checked_proof(&mesh, model, camera));
    }
    assert!(
        proven > 500,
        "boundary corpus did not exercise enough successful proofs"
    );
}

#[test]
fn native_bounds_proof_matches_reference_for_full_mixed_sign_dot_rows() {
    let mut random = 0x28da_719b_c604_85f3;
    let mut proven = 0;
    for iteration in 0..1024 {
        let mesh = source((0..12).map(|_| {
            let components: [f32; 3] =
                std::array::from_fn(|_| (random_u32(&mut random) % 2049) as f32 / 1024.0 - 1.0);
            Vec3::new(components[0], components[1], components[2]).unwrap()
        }));
        let mut model = IDENTITY_MODEL;
        let mut camera = IDENTITY_CAMERA;
        if iteration < 512 {
            for row in &mut model {
                for coefficient in &mut row[..3] {
                    *coefficient = (random_u32(&mut random) % 1793) as f32 / 1024.0 - 0.875;
                }
                row[3] = if iteration % 2 == 0 { 16.0 } else { -16.0 };
            }
            for row in &mut camera {
                for coefficient in row {
                    *coefficient = (random_u32(&mut random) % 1793) as f32 / 1024.0 - 0.875;
                }
            }
        } else {
            for row in model.iter_mut().chain(camera.iter_mut()) {
                for coefficient in row {
                    let bits = random_u32(&mut random);
                    let exponent = bits % 248;
                    *coefficient = f32::from_bits((bits & 0x807f_ffff) | (exponent << 23));
                }
            }
        }
        proven += usize::from(checked_proof(&mesh, model, camera));
    }
    assert!(
        proven >= 512,
        "ordinary mixed-sign non-cancelling transforms must use the proof"
    );
}

#[test]
#[ignore = "Paired CPU diagnostic; run serially in release mode on an otherwise idle host"]
fn native_bounds_paired_cpu_diagnostic() {
    use std::{hint::black_box, time::Instant};

    const FRAMES: usize = 40;
    for side in [16, 32] {
        let mesh = grid(side);
        let models: [_; 64] = std::array::from_fn(chunk_model);
        for model in models {
            assert!(checked_proof(&mesh, model, moving_camera(0)));
        }
        for trial in 0..4 {
            // Alternate order so warmup/cache/clock effects do not always favor
            // one implementation. Both paths receive identical moving inputs.
            for optimized in if trial % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let started = Instant::now();
                for frame in 0..FRAMES {
                    let camera = black_box(moving_camera(frame));
                    for model in models {
                        let model = black_box(model);
                        if !optimized
                            || !native_transform_is_proven(black_box(&mesh), model, camera)
                        {
                            black_box(reference(&mesh, model, camera)).unwrap();
                        }
                    }
                }
                eprintln!(
                    "native_bounds_cpu trial={trial} optimized={optimized} objects=64 vertices_per_mesh={} frames={FRAMES} cpu_ms_per_frame={:.6}",
                    mesh.vertices().len(),
                    started.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64,
                );
            }
        }
    }
}
