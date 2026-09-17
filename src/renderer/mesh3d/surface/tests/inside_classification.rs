//! Frozen clipping control flow and bitwise inside-fast-path regressions.

use super::*;

type Clipper = fn(
    [[ShaderValueRange; 4]; 3],
    [[f32; 2]; 3],
    [[f32; 4]; 3],
    [[f32; 4]; 3],
) -> Result<ClippedTriangle, Mesh3dSurfaceError>;

// The pre-dev.3 clipping order is intentionally independent of inside::accepts.
// Shared numerical primitives are unchanged by this shortcut and have their
// own arithmetic oracles. Keep this control flow frozen when optimizing it.
fn legacy(
    clips: [[ShaderValueRange; 4]; 3],
    uv: [[f32; 2]; 3],
    colors: [[f32; 4]; 3],
    lighting: [[f32; 4]; 3],
) -> Result<ClippedTriangle, Mesh3dSurfaceError> {
    let empty = ClipVertex {
        ranges: [ShaderValueRange::exact(0.0); 4],
        planes: 0,
        provenance: 0,
        uv: [0.0; 2],
        color: [1.0; 4],
        lighting: [0.0; 4],
    };
    let mut result = ClippedTriangle {
        vertices: [empty; MAX_POLYGON_VERTICES],
        count: 3,
        crossing: false,
    };
    for (provenance, (destination, ranges)) in result.vertices.iter_mut().zip(clips).enumerate() {
        let plane_ranges =
            clip_plane_ranges(ranges).map_err(|_| Mesh3dSurfaceError::TransformArithmetic)?;
        let planes = plane_ranges
            .iter()
            .enumerate()
            .fold(0, |mask, (plane, range)| {
                mask | if *range == (0.0, 0.0) { 1 << plane } else { 0 }
            });
        *destination = ClipVertex {
            ranges,
            planes,
            provenance: provenance as u8,
            uv: uv[provenance],
            color: colors[provenance],
            lighting: lighting[provenance],
        };
    }
    let mut next_provenance = 3u8;
    for plane in 0..6 {
        if result.vertices[..3]
            .iter()
            .all(|vertex| vertex.plane_range(plane).is_ok_and(|range| range.1 < 0.0))
        {
            result.count = 0;
            return Ok(result);
        }
    }
    for plane in 0..6 {
        if result.count == 0 {
            break;
        }
        let input = result;
        result.count = 0;
        let mut previous = input.vertices[input.count - 1];
        let mut previous_inside = inside(previous, plane)?;
        for current in input.vertices[..input.count].iter().copied() {
            let current_inside = inside(current, plane)?;
            let intersection = if previous_inside != current_inside {
                let intersection = intersection(previous, current, plane, next_provenance)?;
                next_provenance += 1;
                Some(intersection)
            } else {
                None
            };
            result.crossing |= !current_inside;
            for vertex in intersection
                .into_iter()
                .chain(current_inside.then_some(current))
            {
                if result.count > 0
                    && result.vertices[result.count - 1]
                        .ranges
                        .map(|value| value.fixed)
                        == vertex.ranges.map(|value| value.fixed)
                {
                    if !result.vertices[result.count - 1].same_proven_vertex(vertex) {
                        return Err(Mesh3dSurfaceError::ClippedVertexIdentity);
                    }
                    continue;
                }
                if result.count == MAX_POLYGON_VERTICES {
                    return Err(Mesh3dSurfaceError::ClippedPolygonCapacity);
                }
                result.vertices[result.count] = vertex;
                result.count += 1;
            }
            previous = current;
            previous_inside = current_inside;
        }
        if result.count > 1
            && result.vertices[0].ranges.map(|value| value.fixed)
                == result.vertices[result.count - 1]
                    .ranges
                    .map(|value| value.fixed)
        {
            if !result.vertices[0].same_proven_vertex(result.vertices[result.count - 1]) {
                return Err(Mesh3dSurfaceError::ClippedVertexIdentity);
            }
            result.count -= 1;
        }
    }
    if result.count > 0 && result.count < 3 {
        return Err(Mesh3dSurfaceError::ClippedPolygonDegenerate);
    }
    for index in 1..result.count.saturating_sub(1) {
        let vertices = [
            result.vertices[0],
            result.vertices[index],
            result.vertices[index + 1],
        ];
        for vertex in vertices {
            if vertex.ranges[3].minimum < f64::from(f32::MIN_POSITIVE) {
                return Err(Mesh3dSurfaceError::PerspectiveDivide);
            }
            for plane in 0..6 {
                if !inside(vertex, plane)? || vertex.plane_value(plane) < 0.0 {
                    return Err(Mesh3dSurfaceError::ClippedVertexOutside);
                }
            }
        }
        validate_projected_triangle_orientation(vertices.map(|vertex| vertex.ranges))
            .map_err(|_| Mesh3dSurfaceError::ProjectedOrientation)?;
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct Input {
    clips: [[ShaderValueRange; 4]; 3],
    uv: [[f32; 2]; 3],
    colors: [[f32; 4]; 3],
    lighting: [[f32; 4]; 3],
}

impl Input {
    fn exact(points: [[f32; 4]; 3]) -> Self {
        Self {
            clips: points.map(|point| point.map(ShaderValueRange::exact)),
            uv: [[0.125, 0.25], [0.75, 0.375], [0.5, 0.875]],
            colors: [
                [0.25, 0.5, 0.75, 1.0],
                [1.0, 0.25, 0.5, 0.25],
                [0.0, 1.0, 0.125, 0.75],
            ],
            lighting: [
                [0.0, 0.0, 1.0, 0.25],
                [0.5, 0.0, 0.5, 0.5],
                [-0.25, 0.25, 0.75, 0.125],
            ],
        }
    }

    fn ordinary() -> Self {
        Self::exact([
            [-0.5, -0.25, 0.25, 1.0],
            [0.5, -0.25, 0.5, 1.0],
            [0.125, 0.5, 0.75, 1.0],
        ])
    }

    fn run(self, clipper: Clipper) -> Result<ClippedTriangle, Mesh3dSurfaceError> {
        clipper(self.clips, self.uv, self.colors, self.lighting)
    }
}

#[derive(Debug, PartialEq)]
struct VertexSnapshot {
    ranges: [[u64; 3]; 4],
    planes: u8,
    provenance: u8,
    uv: [u32; 2],
    colors: [u32; 4],
    lighting: [u32; 4],
}

fn snapshot(result: ClippedTriangle) -> (usize, bool, [VertexSnapshot; MAX_POLYGON_VERTICES]) {
    (
        result.count,
        result.crossing,
        result.vertices.map(|vertex| VertexSnapshot {
            ranges: vertex.ranges.map(|range| {
                [
                    u64::from(range.fixed.to_bits()),
                    range.minimum.to_bits(),
                    range.maximum.to_bits(),
                ]
            }),
            planes: vertex.planes,
            provenance: vertex.provenance,
            uv: vertex.uv.map(f32::to_bits),
            colors: vertex.color.map(f32::to_bits),
            lighting: vertex.lighting.map(f32::to_bits),
        }),
    )
}

fn compare(input: Input) -> Result<ClippedTriangle, Mesh3dSurfaceError> {
    let actual = input.run(clipped_triangle_with_attributes);
    let expected = input.run(legacy);
    assert_eq!(actual.map(snapshot), expected.map(snapshot));
    actual
}

#[test]
fn inside_fast_path_preserves_full_vertex_bits_and_boundary_masks() {
    for reversed in [false, true] {
        let mut input = Input::ordinary();
        if reversed {
            input.clips.swap(0, 1);
        }
        let result = compare(input).unwrap();
        assert!(inside::accepts(&result.vertices, true));
        assert_eq!(result.count, 3);
        assert!(!result.crossing);
    }
    let mut boundary = Input::ordinary();
    boundary.clips[0][0] = ShaderValueRange::exact(-1.0);
    // The exact interval-derived boundary mask overrides this raw fixed
    // equation in the legacy path. The fast predicate must do so as well.
    boundary.clips[0][0].fixed = -1.0001;
    let result = compare(boundary).unwrap();
    assert_ne!(result.vertices[0].planes & 1, 0);
    assert!(inside::accepts(&result.vertices, true));
}

#[test]
fn inside_fast_path_keeps_duplicate_classification_and_orientation_error_order() {
    let mut duplicate = Input::ordinary();
    duplicate.clips[1] = duplicate.clips[0];
    duplicate.clips[1][0].minimum -= 0.000001;
    assert!(matches!(
        compare(duplicate),
        Err(Mesh3dSurfaceError::ClippedVertexIdentity)
    ));
    let mut signed_zero = Input::exact([
        [0.0, 0.0, 0.5, 1.0],
        [-0.0, -0.0, 0.5, 1.0],
        [0.5, 0.5, 0.5, 1.0],
    ]);
    assert!(matches!(
        compare(signed_zero),
        Err(Mesh3dSurfaceError::ClippedVertexIdentity)
    ));
    signed_zero.clips[2][0].minimum = -2.0;
    // Vertex2 is the initial previous vertex of the first clipping plane.
    // Its ambiguous classification precedes the duplicate0/1 check.
    assert!(matches!(
        compare(signed_zero),
        Err(Mesh3dSurfaceError::ClipPlaneClassification)
    ));
    let collinear = Input::exact([
        [-0.5, 0.0, 0.25, 1.0],
        [0.0, 0.0, 0.5, 1.0],
        [0.5, 0.0, 0.75, 1.0],
    ]);
    assert!(matches!(
        compare(collinear),
        Err(Mesh3dSurfaceError::ProjectedOrientation)
    ));
}

#[test]
fn inside_fast_path_preserves_camera_plane_and_subnormal_common_outside() {
    let camera_plane = Input::exact([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, -0.25, 0.5, 1.0],
        [-0.25, 0.5, 0.5, 1.0],
    ]);
    assert!(matches!(
        compare(camera_plane),
        Err(Mesh3dSurfaceError::PerspectiveDivide)
    ));
    let tiny = f32::MIN_POSITIVE;
    let outside = Input::exact([
        [-tiny * 1.5, 0.0, 0.0, tiny],
        [-tiny * 1.5, tiny * 0.5, 0.0, tiny],
        [-tiny * 1.5, -tiny * 0.5, 0.0, tiny],
    ]);
    let result = compare(outside).unwrap();
    assert_eq!(result.count, 0);
    assert!(!result.crossing);
}

fn random_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    (*state >> 32) as u32
}

fn corpus() -> Vec<Input> {
    let mut random = 0x295c_841d_74ba_03ef;
    (0..2048)
        .map(|index| {
            let mut input = Input::ordinary();
            let scale = 0.25 + (random_u32(&mut random) % 1024) as f32 / 4096.0;
            let offset = (random_u32(&mut random) % 2049) as f32 / 4096.0 - 0.25;
            for point in &mut input.clips {
                point[0] = ShaderValueRange::exact(point[0].fixed * scale + offset);
                point[1] = ShaderValueRange::exact(point[1].fixed * scale - offset);
                if index % 3 == 0 {
                    for value in &mut point[..3] {
                        value.minimum -= 0.0000001;
                        value.maximum += 0.0000001;
                    }
                }
            }
            if index % 2 == 0 {
                input.clips.swap(1, 2);
            }
            if index % 4 == 0 {
                match (index / 4) % 6 {
                    0 => input.clips[0][0] = ShaderValueRange::exact(-1.5),
                    1 => input.clips[0][0] = ShaderValueRange::exact(1.5),
                    2 => input.clips[0][1] = ShaderValueRange::exact(-1.5),
                    3 => input.clips[0][1] = ShaderValueRange::exact(1.5),
                    4 => input.clips[0][2] = ShaderValueRange::exact(-0.25),
                    _ => input.clips[0][2] = ShaderValueRange::exact(1.25),
                }
            }
            input
        })
        .collect()
}

#[test]
fn inside_fast_path_matches_frozen_clipper_for_varied_inside_and_crossing_triangles() {
    let mut accepted_inside = 0;
    let mut accepted_crossing = 0;
    for input in corpus() {
        if let Ok(result) = compare(input) {
            if result.crossing {
                accepted_crossing += 1;
            } else if result.count == 3 {
                accepted_inside += 1;
            }
        }
    }
    assert!(accepted_inside > 1000);
    assert!(accepted_crossing > 200);
}

#[test]
fn inside_fast_path_matches_perspective_w_and_camera_crossings() {
    let mut accepted = 0;
    for (index, mut input) in corpus().into_iter().enumerate() {
        for (vertex, point) in input.clips.iter_mut().enumerate() {
            let exponent = ((index + vertex * 13) % 41) as i32 - 20;
            let scale = 2.0_f32.powi(exponent);
            for component in point {
                component.fixed *= scale;
                component.minimum *= f64::from(scale);
                component.maximum *= f64::from(scale);
            }
        }
        accepted += usize::from(compare(input).is_ok());
        if index % 8 == 0 {
            input.clips[0][3] = ShaderValueRange::exact(0.0);
            let _ = compare(input);
            input.clips[0][3] = ShaderValueRange::exact(-1.0);
            let _ = compare(input);
        }
    }
    assert!(accepted > 1000);
}

#[test]
#[ignore = "Paired CPU diagnostic; run serially in release mode on an idle host"]
fn inside_fast_path_paired_cpu_diagnostic() {
    use std::{hint::black_box, time::Instant};
    let inputs = corpus();
    let inside: Vec<_> = inputs
        .iter()
        .copied()
        .filter(|input| {
            input
                .run(legacy)
                .is_ok_and(|result| result.count == 3 && !result.crossing)
        })
        .collect();
    let crossing: Vec<_> = inputs
        .iter()
        .copied()
        .filter(|input| {
            input
                .run(legacy)
                .is_ok_and(|result| result.count >= 3 && result.crossing)
        })
        .collect();
    for (case, inputs) in [("inside", inside), ("crossing", crossing)] {
        assert!(!inputs.is_empty());
        eprintln!("inside_clip_cpu case={case} valid_inputs={}", inputs.len());
        for trial in 0..4 {
            for optimized in if trial % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let clipper = black_box(if optimized {
                    clipped_triangle_with_attributes as Clipper
                } else {
                    legacy as Clipper
                });
                let started = Instant::now();
                for _ in 0..64 {
                    for input in &inputs {
                        let _ = black_box(black_box(*input).run(clipper));
                    }
                }
                eprintln!(
                    "inside_clip_cpu case={case} trial={trial} optimized={optimized} triangles={} ns_per_triangle={:.3}",
                    inputs.len() * 64,
                    started.elapsed().as_secs_f64() * 1e9 / (inputs.len() * 64) as f64
                );
            }
        }
    }
}
