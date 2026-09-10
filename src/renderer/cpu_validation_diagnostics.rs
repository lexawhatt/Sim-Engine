//! CPU-only diagnostic candidates. No production path calls these functions.

use super::*;
use std::{hint::black_box, time::Instant};

type ClipRanges = [(f64, f64); 2];

#[derive(Clone, Copy, PartialEq, Eq)]
struct PositionKey([u32; 7]);

impl PositionKey {
    fn new(vertex: Vertex) -> Self {
        Self([
            vertex.world_position[0].to_bits(),
            vertex.world_position[1].to_bits(),
            vertex.world_offset[0].to_bits(),
            vertex.world_offset[1].to_bits(),
            vertex.screen_offset[0].to_bits(),
            vertex.screen_offset[1].to_bits(),
            vertex.depth.to_bits(),
        ])
    }
}

fn candidate_plain_validation<const CACHE: bool>(
    vertices: &[Vertex],
    extents: GeometryExtents,
    uniform: CameraUniform,
) -> bool {
    assert!(vertices.iter().all(|vertex| vertex.normal_distance == 0.0
        && vertex.tangent_distance == 0.0
        && vertex.stroke_role >= 0.0));
    if !geometry_sources_are_portable(GeometryValidationSource::Tessellated(vertices))
        || !vertices.len().is_multiple_of(3)
    {
        return false;
    }
    // This bounded exact-key cache stores only the existing interval helper's
    // result. Uniforms are fixed throughout this invocation. No approximate
    // equality, shortened interval or changed triangle proof is introduced.
    let mut cached: [Option<(PositionKey, ClipRanges)>; 8] = [None; 8];
    let mut next = 0;
    for triangle in vertices.chunks_exact(3) {
        let mut clip = [[(0.0, 0.0); 2]; 3];
        for (output, vertex) in clip.iter_mut().zip(triangle) {
            let key = PositionKey::new(*vertex);
            let hit = if CACHE {
                cached
                    .iter()
                    .flatten()
                    .find(|(previous, _)| *previous == key)
            } else {
                None
            };
            if let Some((_, bounds)) = hit {
                *output = *bounds;
                continue;
            }
            let Some(bounds) = tessellated_vertex_clip_ranges(*vertex, uniform) else {
                return false;
            };
            *output = bounds;
            if CACHE {
                cached[next] = Some((key, bounds));
                next = (next + 1) % cached.len();
            }
        }
        if !clip_triangle_is_wholly_outside(clip)
            && !clip_triangle_has_stable_signed_area(clip, f64::from(f32::MIN_POSITIVE))
        {
            return false;
        }
    }
    vertices
        .iter()
        .all(|vertex| logical_stroke_branches_are_stable(*vertex, uniform))
        && uniform.sources_are_portable()
        && extents.is_safe_for(uniform)
}

fn circle_scene(count: usize) -> Scene {
    let mut scene = Scene::new(Color::BLACK).unwrap();
    for index in 0..count {
        scene
            .try_circle(
                Vec2::new((index % 10) as f32 * 12.0, (index / 10) as f32 * 12.0),
                3.0,
                ShapeStyle::filled(Color::rgb(0.2, 0.3, 0.4)),
            )
            .unwrap();
    }
    scene
}

#[test]
fn plain_validation_diagnostic_candidates_match_existing_proofs() {
    let (vertices, _) = tessellate_for_test(&circle_scene(10));
    for angle in [0.0, 0.13, 0.71] {
        for zoom in [0.01, 1.0, 32.0] {
            let mut camera = Camera2d::new(Vec2::new(3.0, 7.0), zoom).unwrap();
            camera.set_rotation(angle).unwrap();
            let uniform =
                CameraUniform::new(camera, LogicalViewport::new(1280.0, 720.0).unwrap()).unwrap();
            for invalid in [false, true] {
                let mut vertices = vertices.clone();
                if invalid {
                    vertices[1].world_position[0] = f32::from_bits(1);
                }
                let extents = GeometryExtents::from_vertices(&vertices);
                let baseline = geometry::legacy_is_safe_for(
                    extents,
                    GeometryValidationSource::Tessellated(&vertices),
                    uniform,
                );
                assert_eq!(
                    candidate_plain_validation::<false>(&vertices, extents, uniform),
                    baseline
                );
                assert_eq!(
                    candidate_plain_validation::<true>(&vertices, extents, uniform),
                    baseline
                );
            }
        }
    }
}

#[test]
#[ignore = "opt-in release CPU diagnostic; excludes GPU work and frame presentation"]
fn streaming_validation_cpu_diagnostic() {
    if cfg!(debug_assertions) {
        panic!("diagnostic requires --release");
    }
    let uniform = CameraUniform::new(
        Camera2d::default(),
        LogicalViewport::new(1280.0, 720.0).unwrap(),
    )
    .unwrap();
    for count in [1, 100, 1000] {
        let scene = circle_scene(count);
        let (mut vertices, mut batches) = tessellate_for_test(&scene);
        let extents = GeometryExtents::from_vertices(&vertices);
        assert!(geometry_is_safe_for(
            extents,
            GeometryValidationSource::Tessellated(&vertices),
            uniform
        ));
        for stage in [
            "tessellate",
            "extents",
            "sources",
            "centers",
            "topology",
            "branches",
            "full",
            "omit_duplicate",
            "exact_cache8",
        ] {
            let mut operation = || {
                let uniform = black_box(uniform);
                match stage {
                    "tessellate" => {
                        vertices.clear();
                        batches.clear();
                        black_box(tessellate_scene(&scene, &mut vertices, &mut batches).unwrap());
                    }
                    "extents" => {
                        black_box(GeometryExtents::from_vertices(black_box(&vertices)));
                    }
                    "sources" => assert!(black_box(geometry_sources_are_portable(
                        GeometryValidationSource::Tessellated(black_box(&vertices))
                    ))),
                    "centers" => assert!(black_box(geometry_vertex_centers_are_portable(
                        GeometryValidationSource::Tessellated(black_box(&vertices)),
                        uniform
                    ))),
                    "topology" => assert!(black_box(tessellated_triangle_topology_is_portable(
                        black_box(&vertices),
                        uniform
                    ))),
                    "branches" => {
                        assert!(black_box(vertices.iter().all(|vertex| {
                            logical_stroke_branches_are_stable(*vertex, uniform)
                        })))
                    }
                    "full" => assert!(black_box(geometry::legacy_is_safe_for(
                        extents,
                        GeometryValidationSource::Tessellated(black_box(&vertices)),
                        uniform
                    ))),
                    "omit_duplicate" => assert!(black_box(candidate_plain_validation::<false>(
                        black_box(&vertices),
                        extents,
                        uniform
                    ))),
                    _ => assert!(black_box(candidate_plain_validation::<true>(
                        black_box(&vertices),
                        extents,
                        uniform
                    ))),
                }
            };
            for _ in 0..4 {
                operation();
            }
            let (_, allocations) = crate::test_allocations::count(&mut operation);
            let mut samples = Vec::with_capacity(15);
            let repetitions = if count == 1 { 16 } else { 1 };
            for _ in 0..15 {
                let start = Instant::now();
                for _ in 0..repetitions {
                    operation();
                }
                samples.push(start.elapsed().as_secs_f64() / repetitions as f64 * 1000.0);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "streaming_validation circles={count} vertices={} stage={stage} median_ms={:.6} p10_p90_ms={:.6}/{:.6} allocations={allocations}",
                vertices.len(),
                samples[7],
                samples[1],
                samples[13]
            );
        }
    }
}
