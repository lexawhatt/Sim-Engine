//! Shared-buffer draws must match independent-buffer, unbatched controls exactly.

use super::*;
use crate::{
    Fog3d, Mesh3dAttributes, TextureAddressMode3d, TextureCoordinate2d, TextureUvTransform3d,
};

pub(super) fn assert_gpu_batch_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = surface::test_target(device, &identity, format, 128, 128);
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 3.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(8.0)).unwrap(),
    )
    .unwrap();
    let texture = texture::create_texture_with_alpha(
        device,
        queue,
        &identity,
        &renderer.textures.layout,
        2,
        2,
        vec![
            255, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255, 128, 255, 255, 255, 255,
        ],
        ImageBudget::default(),
    )
    .unwrap();
    for policy in [
        SurfaceRasterization3d::Native,
        SurfaceRasterization3d::StrictPortable,
    ] {
        for (case, name) in [
            "colored untextured",
            "alternating sidedness",
            "compatible Lambert",
            "texture tint and UV per instance",
            "mask cutoff per instance",
            "alternating texture filters",
            "opaque and mask share pipeline",
            "overlapping Blend remains separate",
            "alternating lighting pipelines",
            "same texture with alternating addressing",
            "fog participation per instance",
            "uncolored untextured",
            "uncolored textured",
            "coplanar opaque insertion ties",
            "Blend sorting leaves visible-index gaps",
        ]
        .into_iter()
        .enumerate()
        {
            let mut attributes = Mesh3dAttributes::new()
                .with_texture_coordinates(
                    [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                        .map(|[u, v]| TextureCoordinate2d::new(u, v).unwrap())
                        .to_vec(),
                )
                .with_normals(vec![Vec3::Z; 4])
                .unwrap();
            if ![11, 12].contains(&case) {
                attributes = attributes
                    .with_vertex_colors(vec![Color::WHITE; 4])
                    .unwrap();
            }
            let source = Mesh3d::with_attributes(
                vec![
                    Vec3::new(-0.19, -0.6, 0.0).unwrap(),
                    Vec3::new(0.19, -0.6, 0.0).unwrap(),
                    Vec3::new(0.19, 0.6, 0.0).unwrap(),
                    Vec3::new(-0.19, 0.6, 0.0).unwrap(),
                ],
                vec![0, 1, 2, 0, 2, 3],
                vec![],
                attributes,
            )
            .unwrap();
            let shared =
                create_retained_mesh(device, queue, Arc::clone(&identity), source.clone()).unwrap();
            let mut batched = Scene3d::new(Color::BLACK).unwrap();
            let mut separate = Scene3d::new(Color::BLACK).unwrap();
            if case == 10 {
                let fog = Fog3d::new(Color::rgb(0.1, 0.2, 0.5), 1.0, 0.4).unwrap();
                batched.set_fog(Some(fog));
                separate.set_fog(Some(fog));
            }
            for index in 0..4 {
                let independent =
                    create_retained_mesh(device, queue, Arc::clone(&identity), source.clone())
                        .unwrap();
                let (shared, independent) = if (3..=9).contains(&case) || case == 12 {
                    let sampling = if case == 5 && index % 2 == 1 {
                        ImageSampling::Linear
                    } else {
                        ImageSampling::Nearest
                    };
                    let material = TextureMaterial3d::with_alpha(
                        &texture,
                        sampling,
                        Color::rgba(1.0, 1.0, 1.0, 0.5 + index as f32 * 0.125),
                    )
                    .unwrap()
                    .with_address_mode(if case == 9 && index % 2 == 1 {
                        TextureAddressMode3d::Repeat
                    } else {
                        TextureAddressMode3d::Clamp
                    })
                    .with_uv_transform(
                        TextureUvTransform3d::new(
                            Vec2::splat(if case == 9 { 2.0 } else { 1.0 }),
                            Vec2::new(index as f32 * 0.05, 0.0),
                        )
                        .unwrap(),
                    );
                    (
                        texture::attach_material(&identity, &shared, &material).unwrap(),
                        texture::attach_material(&identity, &independent, &material).unwrap(),
                    )
                } else {
                    (shared.clone(), independent)
                };
                let color = [
                    Color::WHITE,
                    Color::rgb(1.0, 0.5, 0.5),
                    Color::rgb(0.5, 1.0, 0.5),
                    Color::rgb(0.5, 0.5, 1.0),
                ][index];
                let mut style = if case == 4 || (case == 6 && index % 2 == 1) {
                    SurfaceStyle3d::mask(color, 0.25 + index as f32 * 0.125).unwrap()
                } else if case == 7 || (case == 14 && index == 1) {
                    SurfaceStyle3d::blend(Color::rgba(1.0, 1.0, 1.0, 0.5)).unwrap()
                } else {
                    SurfaceStyle3d::opaque(color).unwrap()
                };
                if case == 1 && index % 2 == 1 {
                    style = style.with_sidedness(SurfaceSidedness3d::FrontOnly);
                }
                if case == 2 || (case == 8 && index % 2 == 1) {
                    style = style.with_lighting(SurfaceLighting3d::Lambert);
                }
                if case == 10 {
                    style = style.with_fog(index % 2 == 1);
                }
                let transform = Transform3d::new(
                    Vec3::new(
                        if [7, 13].contains(&case) {
                            0.0
                        } else {
                            -0.6 + index as f32 * 0.4
                        },
                        0.0,
                        if case == 7 { index as f32 * 0.1 } else { 0.0 },
                    )
                    .unwrap(),
                    Rotation3d::IDENTITY,
                    Vec3::new(1.0, 1.0, 1.0).unwrap(),
                )
                .unwrap();
                batched
                    .try_push(&shared, transform, MeshStyle3d::surface(style))
                    .unwrap();
                separate
                    .try_push(&independent, transform, MeshStyle3d::surface(style))
                    .unwrap();
                // Invisible insertion slots must not interrupt contiguous GPU instances.
                let hidden = batched
                    .try_push(&shared, transform, MeshStyle3d::surface(style))
                    .unwrap();
                batched.set_visible(hidden, false).unwrap();
            }
            let budget = Mesh3dRenderBudget::default().with_surface_policy(policy);
            let shared_report = renderer
                .render_scene3d(device, queue, &identity, &target, &batched, camera, budget)
                .unwrap();
            let shared_pixels = surface::test_read_pixels(device, queue, &target);
            let separate_report = renderer
                .render_scene3d(device, queue, &identity, &target, &separate, camera, budget)
                .unwrap();
            let separate_pixels = surface::test_read_pixels(device, queue, &target);
            assert_eq!(
                shared_pixels, separate_pixels,
                "policy={policy:?}, case={name}"
            );
            assert!(
                shared_pixels
                    .chunks_exact(4)
                    .any(|pixel| pixel[..3] != [0, 0, 0])
            );
            assert_eq!(shared_report.object_count(), 4);
            assert_eq!(shared_report.triangle_count(), 8);
            assert_eq!(separate_report.draw_call_count(), 4);
            let expected_draws = if [1, 5, 7, 8, 9].contains(&case) {
                4
            } else if case == 14 {
                3
            } else {
                1
            };
            assert_eq!(
                shared_report.draw_call_count(),
                expected_draws,
                "policy={policy:?}, case={name}"
            );
            assert_eq!(shared_report.preflight(), separate_report.preflight());
            if case == 13 {
                // Equal-depth opaque fragments use Less: the first white
                // instance must win over all later colored instances.
                let offset = (64 * 128 + 64) * 4;
                assert_eq!(&shared_pixels[offset..offset + 4], &[255, 255, 255, 255]);
            }
            if case == 14 {
                let order = material::prepare_order(&batched, camera, budget).unwrap();
                assert_eq!(
                    order
                        .iter()
                        .map(|draw| draw.visible_index)
                        .collect::<Vec<_>>(),
                    [0, 2, 3, 1]
                );
            }

            // CPU-clipped objects cannot be consumed by a retained-instance run.
            if case == 0 && policy == SurfaceRasterization3d::StrictPortable {
                let id = batched.instances()[2].id();
                batched
                    .set_transform(
                        id,
                        Transform3d::new(
                            Vec3::new(-1.0, 0.0, 0.0).unwrap(),
                            Rotation3d::IDENTITY,
                            Vec3::new(1.0, 1.0, 1.0).unwrap(),
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let report = renderer
                    .render_scene3d(device, queue, &identity, &target, &batched, camera, budget)
                    .unwrap();
                assert_eq!(report.preflight().generated_object_count(), 1);
                assert_eq!(report.draw_call_count(), 3);
                let shared_pixels = surface::test_read_pixels(device, queue, &target);
                let id = separate.instances()[1].id();
                separate
                    .set_transform(
                        id,
                        Transform3d::new(
                            Vec3::new(-1.0, 0.0, 0.0).unwrap(),
                            Rotation3d::IDENTITY,
                            Vec3::new(1.0, 1.0, 1.0).unwrap(),
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let separate_report = renderer
                    .render_scene3d(device, queue, &identity, &target, &separate, camera, budget)
                    .unwrap();
                assert_eq!(separate_report.draw_call_count(), 4);
                assert_eq!(report.preflight(), separate_report.preflight());
                assert_eq!(
                    shared_pixels,
                    surface::test_read_pixels(device, queue, &target)
                );
            }
        }
    }
}
