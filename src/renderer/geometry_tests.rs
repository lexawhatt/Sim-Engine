use super::*;
use std::hint::black_box;

fn tessellate(scene: &Scene) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    tessellate_scene(scene, &mut vertices, &mut Vec::new()).unwrap();
    vertices
}

fn circle_vertices(count: usize) -> Vec<Vertex> {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    for index in 0..count {
        scene
            .try_circle(
                Vec2::new((index % 10) as f32 * 12.0, (index / 10) as f32 * 12.0),
                3.0,
                ShapeStyle::filled(Color::WHITE.with_alpha(0.5)),
            )
            .unwrap();
    }
    tessellate(&scene)
}

fn uniforms() -> [CameraUniform; 5] {
    let viewport = LogicalViewport::new(1280.0, 720.0).unwrap();
    let mut rotated = Camera2d::new(Vec2::new(3.0, 7.0), 1.3).unwrap();
    rotated.set_rotation(0.35).unwrap();
    rotated.set_projection(crate::Projection2d::new(0.5, 1.0).unwrap());
    let base = CameraUniform::new(Camera2d::default(), viewport).unwrap();
    let mut overflow = base;
    overflow.world_to_screen_x[0] = MAX_PORTABLE_SHADER_VALUE;
    overflow.screen_to_clip[0] = MAX_PORTABLE_SHADER_VALUE;
    [
        base,
        CameraUniform::new(rotated, viewport).unwrap(),
        CameraUniform::new(Camera2d::new(Vec2::ZERO, 10_000.0).unwrap(), viewport).unwrap(),
        CameraUniform::new(Camera2d::new(Vec2::splat(1e10), 1.0).unwrap(), viewport).unwrap(),
        overflow,
    ]
}

fn compare(vertices: &[Vertex], uniform: CameraUniform) -> bool {
    let extents = GeometryExtents::from_vertices(vertices);
    let source = GeometryValidationSource::Tessellated(vertices);
    let old = legacy_is_safe_for(extents, source, uniform);
    assert_eq!(geometry_is_safe_for(extents, source, uniform), old);
    old
}

#[test]
fn position_memo_returns_bit_identical_intervals_and_is_bounded() {
    let original = circle_vertices(1)[0];
    let values = [
        0.0,
        -0.0,
        3.25,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        1e30,
        f32::MAX,
        f32::NAN,
    ];
    for uniform in uniforms() {
        let mut proofs = PositionProofs::new(uniform);
        for field in 0..7 {
            for value in values {
                let mut vertex = original;
                match field {
                    0 => vertex.world_position[0] = value,
                    1 => vertex.world_position[1] = value,
                    2 => vertex.world_offset[0] = value,
                    3 => vertex.world_offset[1] = value,
                    4 => vertex.screen_offset[0] = value,
                    5 => vertex.screen_offset[1] = value,
                    _ => vertex.depth = value,
                }
                let expected = tessellated_vertex_clip_ranges(vertex, uniform)
                    .map(|ranges| ranges.map(|(low, high)| (low.to_bits(), high.to_bits())));
                for _ in 0..2 {
                    let actual = proofs
                        .clip(vertex)
                        .map(|ranges| ranges.map(|(low, high)| (low.to_bits(), high.to_bits())));
                    assert_eq!(actual, expected, "field={field} value={value:?}");
                }
                assert!(proofs.entries.iter().flatten().count() <= 8);
            }
        }
    }
}

#[test]
fn optimized_validation_matches_old_proofs_for_caps_joins_widths_and_markers() {
    let mut accepted = 0;
    let mut rejected = 0;
    for world in [false, true] {
        for cap in [
            crate::StrokeCap2d::Butt,
            crate::StrokeCap2d::Square,
            crate::StrokeCap2d::Round,
        ] {
            for join in [
                crate::StrokeJoin2d::Miter,
                crate::StrokeJoin2d::Bevel,
                crate::StrokeJoin2d::Round,
            ] {
                for mirrored in [false, true] {
                    let color = Color::WHITE.with_alpha(0.5);
                    let style = if world {
                        crate::StrokeStyle2d::world(crate::WorldLength::new(2.0).unwrap(), color)
                    } else {
                        crate::StrokeStyle2d::logical(
                            crate::LogicalPixels::new(2.0).unwrap(),
                            color,
                        )
                    }
                    .with_cap(cap)
                    .with_join(join);
                    let mut scene = Scene::new(Color::BLACK).unwrap();
                    let sign = if mirrored { -1.0 } else { 1.0 };
                    scene
                        .try_styled_polyline(
                            vec![
                                Vec2::new(-20.0, -8.0 * sign),
                                Vec2::new(0.0, 12.0 * sign),
                                Vec2::new(24.0, -6.0 * sign),
                            ],
                            style,
                        )
                        .unwrap();
                    for anchor in [
                        crate::StrokeMarkerAnchor2d::BaseAtEndpoint,
                        crate::StrokeMarkerAnchor2d::TipAtEndpoint,
                    ] {
                        let marker = crate::StrokeMarker2d::arrow(
                            crate::LogicalPixels::new(3.0).unwrap(),
                            crate::LogicalPixels::new(4.0).unwrap(),
                        )
                        .with_anchor(anchor);
                        scene
                            .try_styled_line(
                                Vec2::new(0.0, 24.0),
                                Vec2::new(4.0, 24.0),
                                style.with_start_marker(marker).with_end_marker(marker),
                            )
                            .unwrap();
                    }
                    let vertices = tessellate(&scene);
                    for uniform in uniforms() {
                        if compare(&vertices, uniform) {
                            accepted += 1;
                        } else {
                            rejected += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(accepted > 0 && rejected > 0);
}

#[test]
fn optimized_validation_preserves_extreme_and_malformed_rejections() {
    let original = circle_vertices(1);
    for uniform in uniforms() {
        compare(&[], uniform);
        compare(&original[..2], uniform);
        compare(&original, uniform);
        for value in [
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1e30,
            f32::MAX,
            f32::NAN,
            f32::INFINITY,
        ] {
            let mut vertices = original.clone();
            vertices[2].world_offset[0] = value;
            compare(&vertices, uniform);
        }
        let mut collapsed = original.clone();
        collapsed[1] = collapsed[0];
        compare(&collapsed, uniform);
    }
}

#[test]
#[ignore = "paired warmed release benchmark; CPU geometry validation only"]
fn paired_geometry_validation_benchmark() {
    if cfg!(debug_assertions) {
        panic!("benchmark requires --release");
    }
    let uniform = uniforms()[0];
    for count in [1, 100, 1000] {
        let vertices = circle_vertices(count);
        let extents = GeometryExtents::from_vertices(&vertices);
        let source = GeometryValidationSource::Tessellated(&vertices);
        let old = || legacy_is_safe_for(black_box(extents), black_box(source), black_box(uniform));
        let new =
            || geometry_is_safe_for(black_box(extents), black_box(source), black_box(uniform));
        assert!(old() && new());
        let (_, old_allocations) = crate::test_allocations::count(old);
        let (_, new_allocations) = crate::test_allocations::count(new);
        assert_eq!((old_allocations, new_allocations), (0, 0));
        let mut previous_samples = Vec::with_capacity(15);
        let mut optimized_samples = Vec::with_capacity(15);
        let repetitions = if count == 1 { 16 } else { 1 };
        for round in 0..19 {
            let mut measure = |optimized| {
                let start = Instant::now();
                for _ in 0..repetitions {
                    assert!(black_box(if optimized { new() } else { old() }));
                }
                let millis = start.elapsed().as_secs_f64() / repetitions as f64 * 1000.0;
                if round >= 4 {
                    if optimized {
                        optimized_samples.push(millis);
                    } else {
                        previous_samples.push(millis);
                    }
                }
            };
            measure(round % 2 == 0);
            measure(round % 2 != 0);
        }
        previous_samples.sort_by(f64::total_cmp);
        optimized_samples.sort_by(f64::total_cmp);
        println!(
            "paired_geometry circles={count} vertices={} old_ms={:.6} optimized_ms={:.6} old_p10_p90_ms={:.6}/{:.6} optimized_p10_p90_ms={:.6}/{:.6} allocations={old_allocations}/{new_allocations}",
            vertices.len(),
            previous_samples[7],
            optimized_samples[7],
            previous_samples[1],
            previous_samples[13],
            optimized_samples[1],
            optimized_samples[13]
        );
    }
}
