use super::*;

fn scene(count: usize) -> Scene {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    for index in 0..count {
        let center = Vec2::new(
            (index % 50) as f32 * 20.0 - 490.0,
            (index / 50) as f32 * 14.0 - 280.0,
        );
        scene
            .try_circle(center, 2.0, ShapeStyle::filled(Color::WHITE))
            .unwrap();
        scene
            .try_line(center, center + Vec2::new(13.0, 5.0), 2.0, Color::WHITE)
            .unwrap();
    }
    scene
}

#[test]
fn mixed_dense_primitives_match_frozen_041_proofs_after_changes() {
    let mut vertices = tessellate(&scene(100));
    for uniform in uniforms() {
        compare(&vertices, uniform);
    }
    for value in [f32::MIN_POSITIVE, f32::from_bits(1), f32::MAX, f32::NAN] {
        vertices[196].previous_direction[0] = value;
        for uniform in uniforms() {
            compare(&vertices, uniform);
        }
    }
}

#[test]
#[ignore = "paired warmed release CPU diagnostic; no GPU or presentation work"]
fn mixed_dense_primitives_100_500_2000_cpu_diagnostic() {
    if cfg!(debug_assertions) {
        panic!("diagnostic requires --release");
    }
    let uniform = uniforms()[0];
    for count in [100, 500, 2000] {
        let scene = scene(count);
        let mut vertices = tessellate(&scene);
        let extents = GeometryExtents::from_vertices(&vertices);
        let source = GeometryValidationSource::Tessellated(&vertices);
        let baseline =
            || reference::is_safe_for(black_box(extents), black_box(source), black_box(uniform));
        let optimized =
            || geometry_is_safe_for(black_box(extents), black_box(source), black_box(uniform));
        assert!(baseline() && optimized());
        let (_, previous_allocations) = crate::test_allocations::count(baseline);
        let (_, current_allocations) = crate::test_allocations::count(optimized);
        assert_eq!((previous_allocations, current_allocations), (0, 0));
        let mut previous_samples = Vec::new();
        let mut current_samples = Vec::new();
        for trial in 0..12 {
            for is_current in [trial % 2 == 0, trial % 2 != 0] {
                let start = Instant::now();
                assert!(black_box(if is_current { optimized() } else { baseline() }));
                if trial >= 2 {
                    let destination = if is_current {
                        &mut current_samples
                    } else {
                        &mut previous_samples
                    };
                    destination.push(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }
        previous_samples.sort_by(f64::total_cmp);
        current_samples.sort_by(f64::total_cmp);
        let mut batches = Vec::new();
        let start = Instant::now();
        vertices.clear();
        tessellate_scene(&scene, &mut vertices, &mut batches).unwrap();
        let tessellation_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "dense_primitives constructions={count} commands={} vertices={} upload_bytes={} tessellation_ms={tessellation_ms:.6} paired_samples=10 validation_041_p50_p95_p99_ms={:.6}/{:.6}/{:.6} validation_current_p50_p95_p99_ms={:.6}/{:.6}/{:.6} validation_allocations={previous_allocations}/{current_allocations}",
            scene.command_count(),
            vertices.len(),
            vertices.len() * std::mem::size_of::<Vertex>(),
            previous_samples[5],
            previous_samples[9],
            previous_samples[9],
            current_samples[5],
            current_samples[9],
            current_samples[9],
        );
    }
}
