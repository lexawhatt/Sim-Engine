//! Ordered differential controls for fog proofs and repeated-normal validation.

use super::*;
use crate::{Mesh3dAttributes, Rotation3d};

const IDENTITY: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

fn mesh(points: Vec<Vec3>, normals: Vec<Vec3>) -> Mesh3d {
    Mesh3d::with_attributes(
        points,
        vec![0, 1, 2],
        vec![],
        Mesh3dAttributes::new().with_normals(normals).unwrap(),
    )
    .unwrap()
}

fn triangle() -> Vec<Vec3> {
    vec![Vec3::ZERO, Vec3::X, Vec3::Y]
}

// Frozen pre-optimization call order. Keep this independent of both the
// aggregate fog proof and the cache, including normal-before-fog per vertex.
fn reference(
    transport: SurfaceTransport,
    mesh: &Mesh3d,
    model: [[f32; 4]; 3],
) -> Result<(), (usize, Mesh3dSurfaceError)> {
    if !transport.lit && transport.depth_row.is_none() {
        return Ok(());
    }
    for index in 0..mesh.vertices().len() {
        transport
            .vertex(mesh, index, model)
            .map_err(|reason| (index, reason))?;
    }
    Ok(())
}

fn compare(transport: SurfaceTransport, mesh: &Mesh3d, model: [[f32; 4]; 3]) {
    assert_eq!(
        transport.validate_mesh(mesh, model),
        reference(transport, mesh, model),
        "model={model:?}, transport={transport:?}, vertices={:?}, normals={:?}",
        mesh.vertices(),
        mesh.normals(),
    );
}

#[test]
fn lighting_proof_keeps_earlier_fog_failure_before_later_normal_failure() {
    let transform = Transform3d::new(
        Vec3::ZERO,
        Rotation3d::IDENTITY,
        Vec3::new(1e-20, 1e20, 1.0).unwrap(),
    )
    .unwrap();
    let transport = SurfaceTransport {
        lit: true,
        normal_rows: normal_rows(transform).unwrap(),
        depth_row: Some([1.0, 0.0, 0.0, 1.0]),
        ..SurfaceTransport::default()
    };
    let points = vec![
        Vec3::new(f32::MIN_POSITIVE, 0.0, 0.0).unwrap(),
        Vec3::X,
        Vec3::Y,
    ];
    let model = transform.model_rows().unwrap();
    let early_fog = mesh(points.clone(), vec![Vec3::X, Vec3::Y, Vec3::X]);
    compare(transport, &early_fog, model);
    assert_eq!(
        transport.validate_mesh(&early_fog, model),
        Err((0, Mesh3dSurfaceError::FogArithmetic))
    );
    let same_vertex = mesh(points, vec![Vec3::Y, Vec3::Y, Vec3::X]);
    compare(transport, &same_vertex, model);
    assert_eq!(
        transport.validate_mesh(&same_vertex, model),
        Err((0, Mesh3dSurfaceError::NormalTransform))
    );
}

#[test]
fn lighting_proof_checks_unused_and_missing_normals_after_cached_successes() {
    let mut points = triangle();
    points.extend([Vec3::X; 20]);
    let mut normals = vec![Vec3::X; points.len()];
    normals[19] = Vec3::Y;
    let mesh = mesh(points, normals);
    let transport = SurfaceTransport {
        lit: true,
        normal_rows: [[1.0, 0.0, 0.0, 0.0], [0.0; 4], [0.0, 0.0, 1.0, 0.0]],
        depth_row: Some([0.0, 0.0, 1.0, 1.0]),
        ..SurfaceTransport::default()
    };
    assert!(validation::fog_transform_is_proven(
        &mesh,
        IDENTITY,
        transport.depth_row.unwrap()
    ));
    compare(transport, &mesh, IDENTITY);
    assert_eq!(
        transport.validate_mesh(&mesh, IDENTITY),
        Err((19, Mesh3dSurfaceError::NormalTransform))
    );

    let missing = Mesh3d::new(triangle(), vec![0, 1, 2]).unwrap();
    compare(transport, &missing, IDENTITY);
    assert_eq!(
        transport.validate_mesh(&missing, IDENTITY),
        Err((0, Mesh3dSurfaceError::NormalTransform))
    );

    let mut points = triangle();
    points.extend([Vec3::X; 20]);
    let mut normals: Vec<_> = (0..points.len())
        .map(|index| Vec3::new(1.0, index as f32 + 1.0, 0.0).unwrap())
        .collect();
    normals[19] = Vec3::Y;
    let varied = self::mesh(points, normals);
    // The ninth distinct success disables memoization; a later unused source
    // normal must still be validated, rather than mistaken for cache overflow.
    compare(transport, &varied, IDENTITY);
    assert_eq!(
        transport.validate_mesh(&varied, IDENTITY),
        Err((19, Mesh3dSurfaceError::NormalTransform))
    );
}

#[test]
fn lighting_proof_does_not_cache_success_across_transform_or_activation_changes() {
    let mesh = mesh(triangle(), vec![Vec3::Y; 3]);
    let ordinary = SurfaceTransport {
        lit: true,
        normal_rows: IDENTITY,
        ..SurfaceTransport::default()
    };
    assert_eq!(ordinary.validate_mesh(&mesh, IDENTITY), Ok(()));
    let lost = SurfaceTransport {
        normal_rows: [[1.0, 0.0, 0.0, 0.0], [0.0; 4], [0.0, 0.0, 1.0, 0.0]],
        ..ordinary
    };
    assert_eq!(
        lost.validate_mesh(&mesh, IDENTITY),
        Err((0, Mesh3dSurfaceError::NormalTransform))
    );
    assert_eq!(
        SurfaceTransport { lit: false, ..lost }.validate_mesh(&mesh, IDENTITY),
        Ok(())
    );
    assert_eq!(ordinary.validate_mesh(&mesh, IDENTITY), Ok(()));

    let fog = SurfaceTransport {
        depth_row: Some([f32::NAN, 0.0, 1.0, 0.0]),
        ..ordinary
    };
    compare(fog, &mesh, IDENTITY);
    assert_eq!(
        fog.validate_mesh(&mesh, IDENTITY),
        Err((0, Mesh3dSurfaceError::FogArithmetic))
    );
    assert_eq!(
        SurfaceTransport {
            depth_row: None,
            ..fog
        }
        .validate_mesh(&mesh, IDENTITY),
        Ok(())
    );
}

#[test]
fn lighting_proof_rejects_subnormal_products_hidden_by_model_or_depth_translation() {
    let mut points = triangle();
    points.push(Vec3::new(f32::MIN_POSITIVE, 0.0, 0.0).unwrap());
    let mesh = mesh(points, vec![Vec3::Z; 4]);
    for lit in [false, true] {
        let transport = SurfaceTransport {
            lit,
            normal_rows: IDENTITY,
            depth_row: Some([1.0, 0.0, 0.0, 1.0]),
            ..SurfaceTransport::default()
        };
        let mut model = IDENTITY;
        model[0] = [0.5, 0.0, 0.0, 1.0];
        compare(transport, &mesh, model);
        assert_eq!(
            transport.validate_mesh(&mesh, model),
            Err((3, Mesh3dSurfaceError::FogArithmetic))
        );
        let transport = SurfaceTransport {
            depth_row: Some([0.5, 0.0, 0.0, 1.0]),
            ..transport
        };
        compare(transport, &mesh, IDENTITY);
        assert_eq!(
            transport.validate_mesh(&mesh, IDENTITY),
            Err((3, Mesh3dSurfaceError::FogArithmetic))
        );
    }
}

#[test]
fn lighting_proof_preserves_fog_cancellation_and_invalid_zero_operands() {
    let huge = 2.0_f32.powi(80);
    let mut points = triangle();
    points.push(Vec3::new(huge, -huge, 0.0).unwrap());
    let mesh = mesh(points, vec![Vec3::Z; 4]);
    let model = [[1.0, 1.0, 0.0, 0.0], IDENTITY[1], IDENTITY[2]];
    let fog = SurfaceTransport {
        depth_row: Some([huge, 0.0, 0.0, 0.0]),
        ..SurfaceTransport::default()
    };
    compare(fog, &mesh, model);
    assert_eq!(
        fog.validate_mesh(&mesh, model),
        Err((3, Mesh3dSurfaceError::FogArithmetic))
    );
    for invalid in [
        f32::from_bits(1),
        f32::from_bits(0x007f_ffff),
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ] {
        let fog = SurfaceTransport {
            depth_row: Some([0.0, 0.0, invalid, 1.0]),
            ..fog
        };
        compare(fog, &mesh, IDENTITY);
        assert_eq!(
            fog.validate_mesh(&mesh, IDENTITY),
            Err((0, Mesh3dSurfaceError::FogArithmetic))
        );
    }
}

#[test]
fn lighting_proof_keeps_generated_attribute_bits_for_subnormal_final_depth() {
    let tiny = f32::MIN_POSITIVE * 2.0;
    let mut points = triangle();
    points.push(Vec3::new(tiny, -f32::from_bits(tiny.to_bits() + 1), 0.0).unwrap());
    let mesh = mesh(points, vec![Vec3::Z; 4]);
    let transport = SurfaceTransport {
        lit: true,
        normal_rows: IDENTITY,
        depth_row: Some([1.0, 1.0, 0.0, 0.0]),
        ..SurfaceTransport::default()
    };
    assert!(validation::fog_transform_is_proven(
        &mesh,
        IDENTITY,
        transport.depth_row.unwrap()
    ));
    compare(transport, &mesh, IDENTITY);
    assert_eq!(transport.validate_mesh(&mesh, IDENTITY), Ok(()));
    let attributes = transport.vertex(&mesh, 3, IDENTITY).unwrap();
    assert_eq!(attributes[..3], [0.0, 0.0, 1.0]);
    assert_eq!(attributes[3].to_bits(), (-f32::from_bits(2)).to_bits());
}

fn random_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    (*state >> 32) as u32
}

#[test]
fn lighting_proof_differential_corpus_covers_dense_rows_and_cache_misses() {
    let mut random = 0x9ab2_537e_149c_6021;
    let boundary = [
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE * 8.0,
        0.5,
        0.9,
        1.0,
        2.0,
        2.0_f32.powi(100),
        MAX_PORTABLE_SHADER_VALUE,
        f32::MAX,
    ];
    let mut successes = 0;
    for iteration in 0..1024 {
        let mut points = triangle();
        let mut normals = vec![Vec3::Z; 3];
        for vertex in 0..24 {
            let point = if iteration < 512 {
                let components: [f32; 3] =
                    std::array::from_fn(|_| (random_u32(&mut random) % 2049) as f32 / 1024.0 - 1.0);
                Vec3::new(components[0], components[1], components[2]).unwrap()
            } else {
                let components: [f32; 3] = std::array::from_fn(|_| {
                    boundary[random_u32(&mut random) as usize % boundary.len()]
                });
                Vec3::new(components[0], components[1], components[2]).unwrap()
            };
            points.push(point);
            normals.push(if iteration % 2 == 0 {
                Vec3::Z
            } else {
                Vec3::new(vertex as f32 + 1.0, 3.0, 7.0).unwrap()
            });
        }
        let mesh = mesh(points, normals);
        let mut model = IDENTITY;
        let mut depth = [0.0, 0.0, 1.0, 0.5];
        for row in &mut model {
            for coefficient in row.iter_mut() {
                *coefficient = if iteration < 512 {
                    (random_u32(&mut random) % 1793) as f32 / 1024.0 - 0.875
                } else {
                    boundary[random_u32(&mut random) as usize % boundary.len()]
                };
            }
            if iteration < 512 {
                row[3] = 8.0;
            }
        }
        for coefficient in &mut depth {
            *coefficient = if iteration < 512 {
                (random_u32(&mut random) % 1793) as f32 / 1024.0 - 0.875
            } else {
                boundary[random_u32(&mut random) as usize % boundary.len()]
            };
        }
        let transport = SurfaceTransport {
            lit: true,
            normal_rows: IDENTITY,
            depth_row: Some(depth),
            ..SurfaceTransport::default()
        };
        successes += usize::from(validation::fog_transform_is_proven(&mesh, model, depth));
        compare(transport, &mesh, model);
    }
    assert!(
        successes >= 512,
        "ordinary dense-row samples should prove safe"
    );
}

fn diagnostic_mesh(side: usize, varied: bool) -> Mesh3d {
    let mut points = Vec::new();
    let mut normals = Vec::new();
    for y in 0..=side {
        for x in 0..=side {
            points.push(Vec3::new(x as f32 / side as f32, y as f32 / side as f32, 0.0).unwrap());
            normals.push(if varied {
                Vec3::new(x as f32 + 1.0, y as f32 + 1.0, 1.0).unwrap()
            } else {
                Vec3::Z
            });
        }
    }
    Mesh3d::with_attributes(
        points,
        vec![0, 1, side as u32 + 1],
        vec![],
        Mesh3dAttributes::new().with_normals(normals).unwrap(),
    )
    .unwrap()
}

#[test]
#[ignore = "Paired CPU diagnostic; run serially in release mode on an idle host"]
fn lighting_proof_paired_cpu_diagnostic() {
    use std::{hint::black_box, time::Instant};
    const FRAMES: usize = 20;
    for side in [16, 32] {
        for varied in [false, true] {
            let mesh = diagnostic_mesh(side, varied);
            for (case, lit, fog) in [
                ("lit", true, false),
                ("fog", false, true),
                ("lit_fog", true, true),
            ] {
                for trial in 0..4 {
                    for optimized in if trial % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let started = Instant::now();
                        for frame in 0..FRAMES {
                            for object in 0..64 {
                                let model = black_box([
                                    [0.9, 0.0, 0.0, (object % 8) as f32 - 4.0],
                                    [0.0, 0.9, 0.0, (object / 8) as f32 - 4.0],
                                    [0.0, 0.0, 0.9, 0.0],
                                ]);
                                let transport = black_box(SurfaceTransport {
                                    lit,
                                    normal_rows: IDENTITY,
                                    depth_row: fog.then_some([
                                        0.0,
                                        0.0,
                                        -1.0,
                                        12.0 + frame as f32 * 0.01,
                                    ]),
                                    ..SurfaceTransport::default()
                                });
                                let result = if optimized {
                                    transport.validate_mesh(black_box(&mesh), model)
                                } else {
                                    reference(transport, black_box(&mesh), model)
                                };
                                black_box(result).unwrap();
                            }
                        }
                        eprintln!(
                            "lighting_cpu case={case} varied={varied} trial={trial} optimized={optimized} objects=64 vertices={} frames={FRAMES} cpu_ms_per_frame={:.6}",
                            mesh.vertices().len(),
                            started.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64
                        );
                    }
                }
            }
        }
    }
}
