//! Independent source-point checks for conservative Native surface exclusion.

use super::*;

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

fn source(points: impl IntoIterator<Item = [f32; 3]>) -> Mesh3d {
    Mesh3d::new(
        points
            .into_iter()
            .map(|[x, y, z]| Vec3::new(x, y, z).unwrap())
            .collect(),
        vec![0, 1, 2],
    )
    .unwrap()
}

fn box_source() -> Mesh3d {
    source([
        [-0.25, -0.25, 0.25],
        [0.25, -0.25, 0.25],
        [0.25, 0.25, 0.25],
        [-0.25, 0.25, 0.25],
        [-0.25, -0.25, 0.75],
        [0.25, -0.25, 0.75],
        [0.25, 0.25, 0.75],
        [-0.25, 0.25, 0.75],
    ])
}

fn translated(offset: [f32; 3]) -> [[f32; 4]; 3] {
    let mut model = IDENTITY_MODEL;
    for (row, offset) in model.iter_mut().zip(offset) {
        row[3] = offset;
    }
    model
}

// This scans actual source coordinates through the original per-point interval
// evaluator. It neither reads mesh bounds nor invokes the aggregate proof.
fn point_interval_mask(
    mesh: &Mesh3d,
    model: [[f32; 4]; 3],
    camera: [[f32; 4]; 4],
) -> Result<u8, (usize, Mesh3dRenderError)> {
    let mut common_planes = 0b11_1111;
    for (index, point) in mesh.vertices().iter().enumerate() {
        let clip =
            shader_clip_point_ranges(*point, model, camera).map_err(|error| (index, error))?;
        let planes = clip_plane_ranges(clip).map_err(|error| (index, error))?;
        let mut outside = 0;
        for (plane, (_, maximum)) in planes.into_iter().enumerate() {
            if maximum < 0.0 {
                outside |= 1 << plane;
            }
        }
        common_planes &= outside;
    }
    Ok(common_planes)
}

fn rounded_dot(row: [f32; 4], point: [f32; 4], association: usize) -> f32 {
    let terms: [f32; 4] = std::array::from_fn(|axis| row[axis] * point[axis]);
    match association {
        0 => terms.into_iter().fold(0.0, |sum, term| sum + term),
        1 => (terms[0] + terms[1]) + (terms[2] + terms[3]),
        2 => terms[0] + (terms[1] + (terms[2] + terms[3])),
        _ => row[0].mul_add(
            point[0],
            row[1].mul_add(point[1], row[2].mul_add(point[2], terms[3])),
        ),
    }
}

fn rounded_point_mask(mesh: &Mesh3d, model: [[f32; 4]; 3], camera: [[f32; 4]; 4]) -> u8 {
    let mut common_planes = 0b11_1111;
    for point in mesh.vertices() {
        let point = [point.x(), point.y(), point.z(), 1.0];
        for model_association in 0..4 {
            let world = [
                rounded_dot(model[0], point, model_association),
                rounded_dot(model[1], point, model_association),
                rounded_dot(model[2], point, model_association),
                1.0,
            ];
            for camera_association in 0..4 {
                let [x, y, z, w] = camera
                    .map(|row| rounded_dot(row, world, camera_association))
                    .map(f64::from);
                assert!([x, y, z, w].into_iter().all(f64::is_finite));
                // Compare homogeneous coordinates directly, independently of
                // the production plane-sum construction and its outward step.
                let outside = [x < -w, x > w, y < -w, y > w, z < 0.0, z > w];
                let mut mask = 0;
                for (plane, outside) in outside.into_iter().enumerate() {
                    if outside {
                        mask |= 1 << plane;
                    }
                }
                common_planes &= mask;
            }
        }
    }
    common_planes
}

fn checked_exclusion(mesh: &Mesh3d, model: [[f32; 4]; 3], camera: [[f32; 4]; 4]) -> bool {
    let excluded = native_mesh_is_outside(mesh, model, camera);
    if excluded {
        assert!(bounds::native_transform_is_proven(mesh, model, camera));
        let interval_mask = point_interval_mask(mesh, model, camera).unwrap();
        assert_ne!(
            interval_mask,
            0,
            "no shared outside plane: model={model:?}, camera={camera:?}, vertices={:?}",
            mesh.vertices()
        );
        assert_ne!(interval_mask & rounded_point_mask(mesh, model, camera), 0);
    }
    excluded
}

#[test]
fn native_culling_proves_each_of_the_six_common_outside_planes() {
    let mesh = box_source();
    let offsets = [
        [-2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, -2.0],
        [0.0, 0.0, 2.0],
    ];
    for (plane, offset) in offsets.into_iter().enumerate() {
        let model = translated(offset);
        assert!(checked_exclusion(&mesh, model, IDENTITY_CAMERA));
        assert_eq!(
            point_interval_mask(&mesh, model, IDENTITY_CAMERA),
            Ok(1 << plane)
        );
    }
    assert!(!checked_exclusion(&mesh, IDENTITY_MODEL, IDENTITY_CAMERA));
}

#[test]
fn native_culling_keeps_boundaries_and_arithmetic_uncertainty() {
    let mesh = box_source();
    for (axis, boundary, adjacent) in [
        (0, -1.0, -f32::from_bits(1.0_f32.to_bits() + 1)),
        (0, 1.0, f32::from_bits(1.0_f32.to_bits() + 1)),
        (1, -1.0, -f32::from_bits(1.0_f32.to_bits() + 1)),
        (1, 1.0, f32::from_bits(1.0_f32.to_bits() + 1)),
        (2, 0.0, -f32::MIN_POSITIVE),
        (2, 1.0, f32::from_bits(1.0_f32.to_bits() + 1)),
    ] {
        for coordinate in [boundary, adjacent] {
            let mut model = IDENTITY_MODEL;
            model[axis] = [0.0, 0.0, 0.0, coordinate];
            assert!(bounds::native_transform_is_proven(
                &mesh,
                model,
                IDENTITY_CAMERA
            ));
            assert!(!checked_exclusion(&mesh, model, IDENTITY_CAMERA));
        }
    }
}

#[test]
fn native_culling_requires_one_common_plane_including_unused_vertices() {
    let crossing = source([[-2.0, 0.0, 0.5], [2.0, 0.0, 0.5], [0.0, 2.0, 0.5]]);
    assert_eq!(
        point_interval_mask(&crossing, IDENTITY_MODEL, IDENTITY_CAMERA),
        Ok(0)
    );
    assert!(!checked_exclusion(
        &crossing,
        IDENTITY_MODEL,
        IDENTITY_CAMERA
    ));

    let unused_inside = source([
        [2.0, 0.0, 0.25],
        [2.0, 0.25, 0.25],
        [2.0, 0.0, 0.75],
        [0.0, 0.0, 0.5],
    ]);
    assert!(!checked_exclusion(
        &unused_inside,
        IDENTITY_MODEL,
        IDENTITY_CAMERA
    ));

    let camera_crossing = source([[-0.5, 0.0, -0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]);
    let mut camera = IDENTITY_CAMERA;
    camera[2] = [0.0; 4];
    camera[3] = [0.0, 0.0, 1.0, 0.0];
    assert!(!checked_exclusion(&camera_crossing, IDENTITY_MODEL, camera));
}

#[test]
fn native_culling_handles_rotations_reflections_and_perspective_without_division() {
    let mesh = box_source();
    let rotation = [
        [0.6, -0.8, 0.0, 4.0],
        [0.8, 0.6, 0.0, 4.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let reflection = [
        [-1.0, 0.0, 0.0, 3.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, -0.5, 0.5],
    ];
    for model in [rotation, reflection] {
        assert!(checked_exclusion(&mesh, model, IDENTITY_CAMERA));
    }
    let mut perspective = IDENTITY_CAMERA;
    perspective[2] = [0.0, 0.0, 0.5, 0.0];
    perspective[3] = [0.0, 0.0, 1.0, 0.0];
    assert!(checked_exclusion(
        &mesh,
        translated([3.0, 0.0, 0.0]),
        perspective
    ));
    assert!(checked_exclusion(
        &mesh,
        translated([0.0, 0.0, -2.0]),
        perspective
    ));
}

#[test]
fn native_culling_fails_open_on_subnormals_invalid_operands_and_overflow() {
    let outside = translated([4.0, 0.0, 0.0]);
    for tiny in [f32::from_bits(1), f32::from_bits(0x007f_ffff)] {
        let mesh = source([
            [-0.25, -0.25, 0.25],
            [0.25, -0.25, 0.25],
            [0.25, 0.25, 0.75],
            [tiny, 0.0, 0.5],
        ]);
        assert!(!checked_exclusion(&mesh, outside, IDENTITY_CAMERA));
        assert_eq!(
            point_interval_mask(&mesh, outside, IDENTITY_CAMERA),
            Err((3, Mesh3dRenderError::InvalidGeometryTransform))
        );
    }

    let flat = source([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0]]);
    for coefficient in [f32::from_bits(1), f32::MAX, f32::INFINITY, f32::NAN] {
        let mut model = outside;
        model[0][2] = coefficient;
        assert!(!checked_exclusion(&flat, model, IDENTITY_CAMERA));
        assert!(point_interval_mask(&flat, model, IDENTITY_CAMERA).is_err());
        let mut camera = IDENTITY_CAMERA;
        camera[0][2] = coefficient;
        assert!(!checked_exclusion(&flat, outside, camera));
        assert!(point_interval_mask(&flat, outside, camera).is_err());
    }

    let mesh = source([[2.0, 0.0, 0.25], [3.0, 0.0, 0.25], [2.0, 0.25, 0.75]]);
    let mut model = IDENTITY_MODEL;
    model[0][0] = MAX_PORTABLE_SHADER_VALUE;
    assert!(!checked_exclusion(&mesh, model, IDENTITY_CAMERA));
    assert!(point_interval_mask(&mesh, model, IDENTITY_CAMERA).is_err());
    let mut camera = IDENTITY_CAMERA;
    camera[0][0] = MAX_PORTABLE_SHADER_VALUE;
    assert!(!checked_exclusion(&mesh, IDENTITY_MODEL, camera));
    assert!(point_interval_mask(&mesh, IDENTITY_MODEL, camera).is_err());
}

#[test]
fn native_culling_does_not_use_outside_translation_to_hide_underflow_or_cancellation() {
    let mesh = source([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [f32::MIN_POSITIVE, 0.0, 0.0],
    ]);
    let mut model = translated([4.0, 0.0, 0.0]);
    model[0][0] = 0.5;
    assert!(!checked_exclusion(&mesh, model, IDENTITY_CAMERA));
    assert!(point_interval_mask(&mesh, model, IDENTITY_CAMERA).is_err());
    let mut camera = IDENTITY_CAMERA;
    camera[0] = [0.5, 0.0, 0.0, 4.0];
    assert!(!checked_exclusion(&mesh, IDENTITY_MODEL, camera));
    assert!(point_interval_mask(&mesh, IDENTITY_MODEL, camera).is_err());

    let mesh = box_source();
    let cancellation = [
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 4.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    assert!(!bounds::native_transform_is_proven(
        &mesh,
        cancellation,
        IDENTITY_CAMERA
    ));
    assert!(!checked_exclusion(&mesh, cancellation, IDENTITY_CAMERA));
    // Falling back is required even if the original point validator can prove
    // that this particular source has safe cancellation and is offscreen.
    assert_ne!(
        point_interval_mask(&mesh, cancellation, IDENTITY_CAMERA).unwrap(),
        0
    );
}

fn random_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    (*state >> 32) as u32
}

#[test]
fn native_culling_matches_independent_intervals_and_rounded_points_across_a_numeric_corpus() {
    let mut random = 0x7ed1_951a_829c_23e7;
    let mut excluded = 0;
    let mut retained = 0;
    for iteration in 0..768 {
        let mesh = source((0..12).map(|_| {
            std::array::from_fn(|_| (random_u32(&mut random) % 513) as f32 / 1024.0 - 0.25)
        }));
        let mut model = IDENTITY_MODEL;
        let mut camera = IDENTITY_CAMERA;
        if iteration < 512 {
            for row in &mut model {
                for coefficient in &mut row[..3] {
                    *coefficient = (random_u32(&mut random) % 1025) as f32 / 1024.0 - 0.5;
                }
                row[3] = if iteration % 2 == 0 { 8.0 } else { -8.0 };
            }
            if iteration % 4 == 0 {
                camera[0] = [0.6, -0.8, 0.0, 0.0];
                camera[1] = [0.8, 0.6, 0.0, 0.0];
            }
        } else {
            for row in model.iter_mut().chain(camera.iter_mut()) {
                for coefficient in row {
                    let bits = random_u32(&mut random);
                    *coefficient = f32::from_bits((bits & 0x807f_ffff) | ((bits % 248) << 23));
                }
            }
        }
        if checked_exclusion(&mesh, model, camera) {
            excluded += 1;
        } else {
            retained += 1;
        }
    }
    assert!(
        excluded >= 512,
        "ordinary distant objects should exercise exclusion"
    );
    assert!(
        retained > 0,
        "the numeric corpus must exercise fail-open fallback"
    );
}
