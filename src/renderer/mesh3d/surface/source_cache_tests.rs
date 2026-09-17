//! Indexed memoization versus zero-storage and direct numerical references.

use super::*;
use crate::{Mesh3dAttributes, TextureCoordinate2d};
use source_cache::SourceCache;
use std::cell::RefCell;

const MODEL: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];
const CAMERA: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn grid(side: usize) -> Mesh3d {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut coordinates = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    for row in 0..=side {
        for column in 0..=side {
            let x = column as f32 / side as f32;
            let y = row as f32 / side as f32;
            vertices.push(Vec3::new(-0.75 + x * 1.5, -0.75 + y * 1.5, 0.25).unwrap());
            normals.push(Vec3::new(x * 0.25, y * 0.5, 1.0).unwrap());
            coordinates.push(TextureCoordinate2d::new(x, y).unwrap());
            colors.push(Color::rgba(x, y, 0.5, 0.25 + x * 0.5));
        }
    }
    for row in 0..side {
        for column in 0..side {
            let first = (row * (side + 1) + column) as u32;
            let next = first + side as u32 + 1;
            indices.extend([first, first + 1, next + 1, first, next + 1, next]);
        }
    }
    Mesh3d::with_attributes(
        vertices,
        indices,
        vec![],
        Mesh3dAttributes::new()
            .with_texture_coordinates(coordinates)
            .with_vertex_colors(colors)
            .unwrap()
            .with_normals(normals)
            .unwrap(),
    )
    .unwrap()
}

fn with_indices(source: &Mesh3d, indices: Vec<u32>) -> Mesh3d {
    Mesh3d::with_attributes(
        source.vertices().to_vec(),
        indices,
        vec![],
        Mesh3dAttributes::new()
            .with_texture_coordinates(source.texture_coordinates().to_vec())
            .with_vertex_colors(source.vertex_colors().to_vec())
            .unwrap()
            .with_normals(source.normals().to_vec())
            .unwrap(),
    )
    .unwrap()
}

fn clip_bits(values: [ShaderValueRange; 4]) -> [(u32, u64, u64); 4] {
    values.map(|value| {
        (
            value.fixed.to_bits(),
            value.minimum.to_bits(),
            value.maximum.to_bits(),
        )
    })
}

#[derive(Debug, PartialEq, Eq)]
struct Classification {
    result: Result<(), Mesh3dRenderError>,
    failures: Vec<(usize, Mesh3dSurfaceError)>,
    visits: Vec<(usize, bool, Vec<u32>)>,
}

fn classify(
    cached: bool,
    mesh: &Mesh3d,
    model: [[f32; 4]; 3],
    transport: SurfaceTransport,
    stop_at: Option<usize>,
) -> Classification {
    let failures = RefCell::new(Vec::new());
    let mut visits = Vec::new();
    let failure = |index, reason: Mesh3dSurfaceError| {
        failures.borrow_mut().push((index, reason));
        reason.legacy()
    };
    let visit = |index,
                 vertices: &[SurfaceClipVertex],
                 colors: &[MeshColorGpu],
                 lighting: &[SurfaceLightingVertex],
                 crossing| {
        let mut bits = Vec::new();
        for ((vertex, color), light) in vertices.iter().zip(colors).zip(lighting) {
            bits.extend(vertex.clip.map(f32::to_bits));
            bits.extend(vertex.uv.map(f32::to_bits));
            bits.extend(color.color.map(f32::to_bits));
            bits.extend(light.normal_depth.map(f32::to_bits));
        }
        visits.push((index, crossing, bits));
        if stop_at == Some(index) {
            Err(Mesh3dRenderError::GeneratedGeometryCapacityTooLarge)
        } else {
            Ok(())
        }
    };
    let result = if cached {
        classify_surface_impl(mesh, model, CAMERA, transport, failure, visit)
    } else {
        classify_surface_with_cache::<0>(mesh, model, CAMERA, transport, failure, visit)
    };
    Classification {
        result,
        failures: failures.into_inner(),
        visits,
    }
}

#[test]
fn indexed_source_cache_preserves_full_bits_and_recomputes_collisions() {
    let mesh = grid(16);
    let transport = SurfaceTransport::test_transport(MODEL, Some([0.0, 0.0, 1.0, 0.5]));
    let mut cache = SourceCache::<64>::new(&mesh, MODEL, CAMERA, transport);
    for index in [0, 1, 0, 1] {
        assert_eq!(
            clip_bits(cache.clip(index).unwrap()),
            clip_bits(
                shader_clip_point_ranges(mesh.vertices()[index as usize], MODEL, CAMERA).unwrap()
            )
        );
        assert_eq!(
            cache.auxiliary(index).unwrap().map(f32::to_bits),
            transport
                .vertex(&mesh, index as usize, MODEL)
                .unwrap()
                .map(f32::to_bits)
        );
    }
    assert_eq!(cache.evaluations(), [2, 2]);
    let mut cache = SourceCache::<64>::new(&mesh, MODEL, CAMERA, transport);
    for index in [0, 64, 128, 0, 64, 128] {
        assert_eq!(
            clip_bits(cache.clip(index).unwrap()),
            clip_bits(
                shader_clip_point_ranges(mesh.vertices()[index as usize], MODEL, CAMERA).unwrap()
            )
        );
        assert_eq!(
            cache.auxiliary(index).unwrap().map(f32::to_bits),
            transport
                .vertex(&mesh, index as usize, MODEL)
                .unwrap()
                .map(f32::to_bits)
        );
    }
    assert_eq!(cache.evaluations(), [6, 6]);
}

#[test]
fn classifier_cache_matches_uncached_attributes_clipping_and_callback_stop() {
    let mesh = grid(8);
    assert!(source_cache::worthwhile(&mesh));
    for transport in [
        SurfaceTransport::default(),
        SurfaceTransport::test_transport(MODEL, Some([0.0, 0.0, 1.0, 0.5])),
    ] {
        for offset in [-0.5, 0.0, 0.5] {
            let mut model = MODEL;
            model[0][3] = offset;
            for stop_at in [None, Some(2)] {
                let expected = classify(false, &mesh, model, transport, stop_at);
                assert_eq!(classify(true, &mesh, model, transport, stop_at), expected);
                if stop_at.is_none() {
                    assert!(expected.result.is_ok());
                    assert_eq!(expected.visits.len(), mesh.triangle_count());
                } else {
                    assert_eq!(
                        expected.result,
                        Err(Mesh3dRenderError::GeneratedGeometryCapacityTooLarge)
                    );
                    assert_eq!(expected.visits.len(), 3);
                }
            }
        }
    }
}

#[test]
fn source_index_not_position_identifies_normal_uv_and_color_seams() {
    let vertices = [
        Vec3::new(-0.5, -0.5, 0.25).unwrap(),
        Vec3::new(0.5, -0.5, 0.25).unwrap(),
        Vec3::new(0.0, 0.5, 0.25).unwrap(),
    ];
    let mut indices = Vec::new();
    for _ in 0..8 {
        indices.extend([0, 1, 2, 3, 4, 5]);
    }
    let mesh = Mesh3d::with_attributes(
        vertices.into_iter().chain(vertices).collect(),
        indices,
        vec![],
        Mesh3dAttributes::new()
            .with_texture_coordinates(
                vec![TextureCoordinate2d::new(0.25, 0.5).unwrap(); 3]
                    .into_iter()
                    .chain(vec![TextureCoordinate2d::new(0.75, 0.5).unwrap(); 3])
                    .collect(),
            )
            .with_vertex_colors(
                vec![Color::WHITE; 3]
                    .into_iter()
                    .chain(vec![Color::rgb(0.2, 0.3, 0.4); 3])
                    .collect(),
            )
            .unwrap()
            .with_normals(
                vec![Vec3::Z; 3]
                    .into_iter()
                    .chain(vec![Vec3::X; 3])
                    .collect(),
            )
            .unwrap(),
    )
    .unwrap();
    let transport = SurfaceTransport::test_transport(MODEL, None);
    let expected = classify(false, &mesh, MODEL, transport, None);
    assert!(expected.result.is_ok());
    assert_ne!(expected.visits[0].2, expected.visits[1].2);
    assert_eq!(classify(true, &mesh, MODEL, transport, None), expected);
}

#[test]
fn classifier_finishes_three_clips_before_any_auxiliary_failure() {
    let make_mesh = |extreme| {
        Mesh3d::with_attributes(
            vec![
                Vec3::new(0.25, 0.0, 0.25).unwrap(),
                Vec3::new(0.5, 0.0, 0.25).unwrap(),
                Vec3::new(if extreme { f32::MAX } else { 0.5 }, 0.5, 0.25).unwrap(),
            ],
            [0, 1, 2].repeat(16),
            vec![],
            Mesh3dAttributes::new()
                .with_normals(vec![Vec3::Z; 3])
                .unwrap(),
        )
        .unwrap()
    };
    let mut model = MODEL;
    model[0][0] = 2.0;
    for transport in [
        SurfaceTransport::test_transport([[0.0; 4]; 3], None),
        SurfaceTransport::test_transport(MODEL, Some([f32::MAX, 0.0, 0.0, 0.0])),
    ] {
        for extreme in [true, false] {
            let mesh = make_mesh(extreme);
            let expected = classify(false, &mesh, model, transport, None);
            assert_eq!(classify(true, &mesh, model, transport, None), expected);
            assert!(expected.result.is_err());
            assert!(expected.visits.is_empty());
            assert_eq!(expected.failures.len(), 1);
            assert_eq!(expected.failures[0].0, 0);
            if extreme {
                assert_eq!(
                    expected.failures[0].1,
                    Mesh3dSurfaceError::TransformArithmetic
                );
            } else {
                assert!(matches!(
                    expected.failures[0].1,
                    Mesh3dSurfaceError::NormalTransform | Mesh3dSurfaceError::FogArithmetic
                ));
            }
        }
    }
}

#[test]
fn source_cache_is_bounded_resets_context_and_does_not_retain_failures() {
    let mesh = grid(8);
    let transport = SurfaceTransport::test_transport(MODEL, None);
    let mut original = SourceCache::<64>::new(&mesh, MODEL, CAMERA, transport);
    let old_clip = clip_bits(original.clip(1).unwrap());
    let old_normal = original.auxiliary(1).unwrap().map(f32::to_bits);
    let mut model = MODEL;
    model[0][3] = 0.125;
    let mut camera = CAMERA;
    camera[1][1] = 0.5;
    let rotation = [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], MODEL[2]];
    let mut changed = SourceCache::<64>::new(
        &mesh,
        model,
        camera,
        SurfaceTransport::test_transport(rotation, None),
    );
    assert_ne!(clip_bits(changed.clip(1).unwrap()), old_clip);
    assert_ne!(changed.auxiliary(1).unwrap().map(f32::to_bits), old_normal);
    let mut failed_camera = CAMERA;
    failed_camera[0][0] = f32::MAX;
    let mut failed = SourceCache::<64>::new(
        &mesh,
        MODEL,
        failed_camera,
        SurfaceTransport::test_transport([[0.0; 4]; 3], None),
    );
    for _ in 0..2 {
        assert_eq!(
            failed.clip(1).unwrap_err(),
            Mesh3dSurfaceError::TransformArithmetic
        );
        assert_eq!(
            failed.auxiliary(1).unwrap_err(),
            Mesh3dSurfaceError::NormalTransform
        );
    }
    assert_eq!(failed.evaluations(), [2, 2]);
    assert!(std::mem::size_of::<SourceCache<'_, 0>>() < 512);
    assert!(std::mem::size_of::<SourceCache<'_, 64>>() < 12 * 1024);
    assert!(!source_cache::worthwhile(&grid(1)));
    let unique = Mesh3d::new(
        mesh.triangle_indices()
            .iter()
            .map(|index| mesh.vertices()[*index as usize])
            .collect(),
        (0..mesh.triangle_indices().len() as u32).collect(),
    )
    .unwrap();
    assert!(!source_cache::worthwhile(&unique));
}

#[test]
fn classifier_cache_preserves_shuffled_reversed_and_colliding_triangle_visits() {
    let source = grid(16);
    let triangles: Vec<_> = source.triangle_indices().chunks_exact(3).collect();
    let reversed: Vec<_> = triangles
        .iter()
        .rev()
        .flat_map(|value| value.iter().copied())
        .collect();
    // An odd multiplier permutes all 512 triangle indices, not merely a subset.
    let shuffled: Vec<_> = (0..triangles.len())
        .flat_map(|index| {
            triangles[(index * 197 + 83) % triangles.len()]
                .iter()
                .copied()
        })
        .collect();
    let mut colliding = Vec::new();
    for _ in 0..16 {
        for base in [0, 64, 128, 192, 256, 192, 128, 64] {
            // Each triangle evicts all three entries of the preceding one.
            colliding.extend([base, base + 1, base + 18]);
        }
    }
    let transport = SurfaceTransport::test_transport(MODEL, Some([0.0, 0.0, 1.0, 0.5]));
    for indices in [reversed, shuffled, colliding] {
        for reverse_winding in [false, true] {
            let mut indices = indices.clone();
            if reverse_winding {
                for triangle in indices.chunks_exact_mut(3) {
                    triangle.swap(1, 2);
                }
            }
            let mesh = with_indices(&source, indices);
            assert!(source_cache::worthwhile(&mesh));
            for offset in [0.0, 0.5] {
                let mut model = MODEL;
                model[0][3] = offset;
                for stop_at in [None, Some(17)] {
                    let expected = classify(false, &mesh, model, transport, stop_at);
                    assert_eq!(classify(true, &mesh, model, transport, stop_at), expected);
                    assert!(expected.failures.is_empty());
                    assert_eq!(
                        expected.visits.len(),
                        stop_at.map_or(mesh.triangle_count(), |i| i + 1)
                    );
                    assert!(
                        expected
                            .visits
                            .iter()
                            .enumerate()
                            .all(|(index, visit)| visit.0 == index)
                    );
                    assert_eq!(expected.result.is_ok(), stop_at.is_none());
                }
            }
        }
    }
}

#[test]
fn clip_and_auxiliary_collisions_have_independent_success_tags() {
    let mesh = grid(16);
    let transport = SurfaceTransport::test_transport(MODEL, Some([0.0, 0.0, 1.0, 0.5]));
    let mut cache = SourceCache::<64>::new(&mesh, MODEL, CAMERA, transport);
    for index in [0, 64, 128, 192, 256, 192, 128, 64, 0].repeat(4) {
        let expected_clip = clip_bits(
            shader_clip_point_ranges(mesh.vertices()[index as usize], MODEL, CAMERA).unwrap(),
        );
        let expected_auxiliary = transport
            .vertex(&mesh, index as usize, MODEL)
            .unwrap()
            .map(f32::to_bits);
        assert_eq!(clip_bits(cache.clip(index).unwrap()), expected_clip);
        assert_eq!(
            cache.auxiliary(index).unwrap().map(f32::to_bits),
            expected_auxiliary
        );
        // Only the clip cache changes; the auxiliary hit must remain valid.
        let collision = if index == 256 { 0 } else { index + 64 };
        cache.clip(collision).unwrap();
        let before = cache.evaluations();
        assert_eq!(
            cache.auxiliary(index).unwrap().map(f32::to_bits),
            expected_auxiliary
        );
        assert_eq!(cache.evaluations(), before);
        assert_eq!(clip_bits(cache.clip(index).unwrap()), expected_clip);
        // Now auxiliary eviction must not invalidate the independently saved clip.
        cache.auxiliary(collision).unwrap();
        let before = cache.evaluations();
        assert_eq!(clip_bits(cache.clip(index).unwrap()), expected_clip);
        assert_eq!(cache.evaluations(), before);
        assert_eq!(
            cache.auxiliary(index).unwrap().map(f32::to_bits),
            expected_auxiliary
        );
    }
}

#[test]
fn source_cache_selection_keeps_tiny_and_unshared_meshes_uncached() {
    let source = grid(1);
    let transport = SurfaceTransport::test_transport(MODEL, None);
    for count in [1, 15, 16, 17] {
        let mesh = with_indices(&source, [0, 1, 3].repeat(count));
        assert_eq!(source_cache::worthwhile(&mesh), count >= 16);
        assert_eq!(
            classify(true, &mesh, MODEL, transport, None),
            classify(false, &mesh, MODEL, transport, None)
        );
    }
    let mut uncached = SourceCache::<0>::new(&source, MODEL, CAMERA, transport);
    for _ in 0..4 {
        uncached.clip(0).unwrap();
        uncached.auxiliary(0).unwrap();
    }
    assert_eq!(uncached.evaluations(), [4, 4]);
    // Source storage exceeds the index count despite repeated references: the
    // cheap selection heuristic deliberately avoids initializing the cache.
    let sparse = with_indices(&grid(8), [0, 1, 10].repeat(16));
    assert!(!source_cache::worthwhile(&sparse));
    assert_eq!(
        classify(true, &sparse, MODEL, transport, None),
        classify(false, &sparse, MODEL, transport, None)
    );
}

#[test]
fn warmed_classifier_uses_no_heap_with_allocation_free_callback() {
    fn run(mesh: &Mesh3d) -> Result<(usize, u64), Mesh3dRenderError> {
        let mut visits = 0;
        let mut digest = 0u64;
        let mut model = MODEL;
        model[0][3] = 0.5;
        classify_surface_impl(
            std::hint::black_box(mesh),
            model,
            CAMERA,
            SurfaceTransport::test_transport(MODEL, Some([0.0, 0.0, 1.0, 0.5])),
            |_, reason| reason.legacy(),
            |index, vertices, colors, lighting, crossing| {
                visits += 1;
                digest = digest.wrapping_add(index as u64 + u64::from(crossing));
                for ((vertex, color), light) in vertices.iter().zip(colors).zip(lighting) {
                    for value in vertex
                        .clip
                        .into_iter()
                        .chain(vertex.uv)
                        .chain(color.color)
                        .chain(light.normal_depth)
                    {
                        digest = digest.rotate_left(7) ^ u64::from(value.to_bits());
                    }
                }
                Ok(())
            },
        )?;
        Ok((visits, digest))
    }
    for mesh in [grid(1), grid(16)] {
        let expected = run(&mesh).unwrap();
        for _ in 0..3 {
            let (result, allocations) = crate::test_allocations::count(|| run(&mesh));
            assert_eq!(result.unwrap(), expected);
            assert_eq!(expected.0, mesh.triangle_count());
            assert_eq!(allocations, 0);
        }
    }
}
