//! Pixel equivalence of cached indexed and uncached expanded source geometry.

use super::*;
use crate::{
    AmbientLight3d, DirectionalLight3d, Fog3d, Lighting3d, Mesh3dAttributes, TextureCoordinate2d,
};

const SIZE: u32 = 64;
const SIDE: usize = 4;

fn indexed_grid() -> Mesh3d {
    let mut vertices = Vec::new();
    let mut coordinates = Vec::new();
    let mut colors = Vec::new();
    let mut normals = Vec::new();
    for row in 0..=SIDE {
        for column in 0..=SIDE {
            let u = column as f32 / SIDE as f32;
            let v = row as f32 / SIDE as f32;
            vertices.push(Vec3::new(-0.625 + u * 1.25, -0.625 + v * 1.25, 0.0).unwrap());
            coordinates.push(TextureCoordinate2d::new(u, v).unwrap());
            colors.push(Color::rgb(
                0.25 + u * 0.5,
                0.125 + v * 0.75,
                0.75 - u * 0.25,
            ));
            // Unit axes remain bit-identical through both constructors. Varying
            // normals make an incorrect auxiliary cache entry visible in pixels.
            normals.push([Vec3::Z, Vec3::X, Vec3::Y][(column + row) % 3]);
        }
    }
    let mut indices = Vec::new();
    for row in 0..SIDE {
        for column in 0..SIDE {
            let first = (row * (SIDE + 1) + column) as u32;
            let next = first + SIDE as u32 + 1;
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

fn expanded(source: &Mesh3d) -> Mesh3d {
    let indices = source.triangle_indices();
    let result = Mesh3d::with_attributes(
        indices
            .iter()
            .map(|index| source.vertices()[*index as usize])
            .collect(),
        (0..indices.len() as u32).collect(),
        vec![],
        Mesh3dAttributes::new()
            .with_texture_coordinates(
                indices
                    .iter()
                    .map(|index| source.texture_coordinates()[*index as usize])
                    .collect(),
            )
            .with_vertex_colors(
                indices
                    .iter()
                    .map(|index| source.vertex_colors()[*index as usize])
                    .collect(),
            )
            .unwrap()
            .with_normals(
                indices
                    .iter()
                    .map(|index| source.normals()[*index as usize])
                    .collect(),
            )
            .unwrap(),
    )
    .unwrap();
    for (destination, source_index) in indices.iter().copied().enumerate() {
        let source_index = source_index as usize;
        assert_eq!(
            result.vertices()[destination],
            source.vertices()[source_index]
        );
        assert_eq!(
            result.texture_coordinates()[destination],
            source.texture_coordinates()[source_index]
        );
        assert_eq!(
            result.vertex_colors()[destination]
                .to_array()
                .map(f32::to_bits),
            source.vertex_colors()[source_index]
                .to_array()
                .map(f32::to_bits)
        );
        let bits = |normal: Vec3| [normal.x(), normal.y(), normal.z()].map(f32::to_bits);
        assert_eq!(
            bits(result.normals()[destination]),
            bits(source.normals()[source_index])
        );
    }
    result
}

fn camera(perspective: bool) -> Camera3d {
    let projection = if perspective {
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(0.5), world(8.0))
    } else {
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(8.0))
    }
    .unwrap();
    Camera3d::look_at(
        Vec3::new(0.0, 0.0, 3.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        projection,
    )
    .unwrap()
}

fn scene(mesh: &RetainedMesh3d, perspective: bool, crossing: bool) -> Scene3d {
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene.set_lighting(
        Lighting3d::new(AmbientLight3d::new(Color::rgb(0.75, 0.5, 1.0), 0.25).unwrap())
            .with_directional(Some(
                DirectionalLight3d::new(Vec3::Z, Color::rgb(1.0, 0.75, 0.5), 0.75).unwrap(),
            )),
    );
    scene.set_fog(Some(
        Fog3d::new(Color::rgb(0.125, 0.25, 0.375), 0.5, 0.2).unwrap(),
    ));
    let extent = if perspective { 3.0 } else { 1.0 };
    let transform = Transform3d::new(
        Vec3::new(if crossing { 0.8 * extent } else { 0.0 }, 0.0, 0.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(extent, extent, 1.0).unwrap(),
    )
    .unwrap();
    scene
        .try_push(
            mesh,
            transform,
            MeshStyle3d::surface(
                SurfaceStyle3d::opaque(Color::rgb(0.75, 1.0, 0.875))
                    .unwrap()
                    .with_lighting(SurfaceLighting3d::Lambert)
                    .with_fog(true),
            ),
        )
        .unwrap();
    scene
}

pub(in crate::renderer::mesh3d) fn assert_gpu_source_cache_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let source = indexed_grid();
    let control = expanded(&source);
    assert!(
        source_cache::worthwhile(&source),
        "indexed control must enable the production cache"
    );
    assert!(
        !source_cache::worthwhile(&control),
        "expanded control must disable the production cache"
    );
    assert_eq!(source.triangle_count(), 32);
    assert_eq!(control.triangle_count(), source.triangle_count());

    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = test_target(device, &identity, format, SIZE, SIZE);
    let texture = texture::create_texture(
        device,
        queue,
        &identity,
        &renderer.textures.layout,
        2,
        2,
        vec![
            255, 255, 255, 255, 255, 128, 64, 255, 64, 192, 255, 255, 128, 255, 64, 255,
        ],
        ImageBudget::default(),
    )
    .unwrap();
    let material = TextureMaterial3d::new(&texture, ImageSampling::Linear, Color::WHITE).unwrap();
    let meshes = [source, control].map(|source| {
        let retained = create_retained_mesh(device, queue, Arc::clone(&identity), source).unwrap();
        texture::attach_material(&identity, &retained, &material).unwrap()
    });
    for perspective in [false, true] {
        for crossing in [false, true] {
            let scenes = meshes
                .each_ref()
                .map(|mesh| scene(mesh, perspective, crossing));
            let mut controls = Vec::new();
            for scene in &scenes {
                let report = renderer.render_scene3d(
                    device, queue, &identity, &target, scene, camera(perspective),
                    Mesh3dRenderBudget::default(),
                ).unwrap_or_else(|error| panic!("source cache GPU control perspective={perspective} crossing={crossing}: {error}"));
                let pixels = test_read_pixels(device, queue, &target);
                let colored = pixels
                    .chunks_exact(4)
                    .filter(|pixel| pixel[..3].iter().any(|channel| *channel != 0))
                    .count();
                assert!(
                    colored > 256,
                    "source cache control must rasterize a nontrivial surface: {colored} pixels"
                );
                let distinct: std::collections::BTreeSet<_> = pixels
                    .chunks_exact(4)
                    .map(|pixel| [pixel[0], pixel[1], pixel[2]])
                    .collect();
                assert!(
                    distinct.len() > 32,
                    "normal/color/UV attributes must visibly vary across the surface"
                );
                assert_eq!(report.object_count(), 1);
                assert_eq!(report.draw_call_count(), 1);
                assert_eq!(report.render_pass_count(), 1);
                if crossing {
                    assert!(report.preflight().clipped_source_triangle_count().unwrap() > 0);
                    assert!(report.preflight().generated_vertex_count() > 0);
                } else {
                    assert_eq!(report.preflight().clipped_source_triangle_count(), Some(0));
                    assert_eq!(report.preflight().generated_vertex_count(), 0);
                }
                controls.push((
                    pixels,
                    report.preflight(),
                    report.triangle_count(),
                    report.edge_count(),
                ));
            }
            // The expanded control has exactly the same triangle and attribute
            // order. No raster tolerance is needed or permitted for memoization.
            assert_eq!(
                controls[0], controls[1],
                "cached/uncached GPU mismatch: perspective={perspective} crossing={crossing}"
            );
        }
    }
}
