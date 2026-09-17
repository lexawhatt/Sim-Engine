//! Sparse edge-uniform uploads must preserve visible indices and retained capacity.

use super::*;

pub(super) fn assert_gpu_edge_upload_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    for policy in [
        SurfaceRasterization3d::Native,
        SurfaceRasterization3d::StrictPortable,
    ] {
        let identity = Arc::new(());
        let target = surface::test_target(device, &identity, format, 64, 64);
        let mut renderer = Mesh3dRenderer::new(device, format);
        let camera = Camera3d::look_at(
            Vec3::new(0.0, 0.0, 3.0).unwrap(),
            Vec3::ZERO,
            Vec3::Y,
            Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(8.0)).unwrap(),
        )
        .unwrap();
        let render = |renderer: &mut Mesh3dRenderer, scene: &Scene3d| {
            renderer
                .render_scene3d(
                    device,
                    queue,
                    &identity,
                    &target,
                    scene,
                    camera,
                    Mesh3dRenderBudget::default().with_surface_policy(policy),
                )
                .unwrap()
        };
        let mesh = create_retained_mesh(
            device,
            queue,
            Arc::clone(&identity),
            Mesh3d::new(
                vec![
                    Vec3::new(-0.6, -0.6, 0.0).unwrap(),
                    Vec3::new(0.6, -0.6, 0.0).unwrap(),
                    Vec3::new(0.0, 0.35, 0.0).unwrap(),
                ],
                vec![0, 1, 2],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(mesh.edge_count, 0);
        let edge_mesh = create_retained_mesh(
            device,
            queue,
            Arc::clone(&identity),
            Mesh3d::with_display_edges(
                vec![
                    Vec3::new(-0.75, 0.625, 0.0).unwrap(),
                    Vec3::new(0.75, 0.625, 0.0).unwrap(),
                ],
                vec![],
                vec![MeshEdge3d::new(0, 1).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
        let surface_style =
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::rgb(0.0, 1.0, 0.0)).unwrap());
        let edge_style = WireframeStyle3d::visible(Color::WHITE, logical(3.0)).unwrap();
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        for index in 0..17 {
            scene
                .try_push(
                    &mesh,
                    Transform3d::IDENTITY,
                    if index == 16 {
                        surface_style.with_wireframe(edge_style)
                    } else {
                        surface_style
                    },
                )
                .unwrap();
            if index % 4 == 0 {
                let hidden = scene
                    .try_push(
                        &edge_mesh,
                        Transform3d::new(
                            Vec3::new(0.0, -1.375, 0.0).unwrap(),
                            Rotation3d::IDENTITY,
                            Vec3::new(1.0, 1.0, 1.0).unwrap(),
                        )
                        .unwrap(),
                        MeshStyle3d::wireframe(edge_style),
                    )
                    .unwrap();
                scene.set_visible(hidden, false).unwrap();
            }
        }
        let late_edge = scene
            .try_push(
                &edge_mesh,
                Transform3d::IDENTITY,
                MeshStyle3d::wireframe(edge_style),
            )
            .unwrap();
        scene.set_visible(late_edge, false).unwrap();
        assert_eq!(scene.visible_object_count(), 17);
        assert!(scene.object_count() > 18);

        let camera_bytes = std::mem::size_of::<Camera3dUniform>();
        let instance_bytes = std::mem::size_of::<MeshInstanceGpu>();
        let initial_edge_gpu_bytes = renderer.edge_object_buffer.size();
        let first = render(&mut renderer, &scene);
        assert_eq!(first.upload_calls(), 2);
        assert_eq!(first.uploaded_bytes(), camera_bytes + 17 * instance_bytes);
        assert_eq!(first.buffer_allocation_count(), 1);
        assert_eq!(first.edge_count(), 0);
        assert_eq!(renderer.instance_capacity, 32);
        assert_eq!(renderer.edge_object_capacity, INITIAL_INSTANCE_CAPACITY);
        assert_eq!(renderer.edge_object_buffer.size(), initial_edge_gpu_bytes);
        assert_eq!(renderer.edge_object_bytes.capacity(), 0);
        let surface_pixels = surface::test_read_pixels(device, queue, &target);
        expect_pixel(&surface_pixels, 32, 40, [0, 255, 0, 255]);
        expect_pixel(&surface_pixels, 32, 12, [0, 0, 0, 255]);
        expect_pixel(&surface_pixels, 32, 56, [0, 0, 0, 255]);

        // The only active edge uses visible index 17, not its scene index or
        // compact edge index zero. Hidden interleaved objects distinguish them.
        scene.set_visible(late_edge, true).unwrap();
        let edged = render(&mut renderer, &scene);
        assert_eq!(edged.object_count(), 18);
        assert_eq!(edged.edge_count(), 1);
        assert_eq!(edged.upload_calls(), 2);
        assert_eq!(
            edged.uploaded_bytes(),
            18 * instance_bytes + 18 * renderer.edge_object_stride
        );
        assert_eq!(edged.buffer_allocation_count(), 1);
        assert_eq!(renderer.instance_capacity, 32);
        assert_eq!(renderer.edge_object_capacity, 32);
        assert_eq!(
            renderer.edge_object_bytes.len(),
            18 * renderer.edge_object_stride
        );
        let edge_pixels = surface::test_read_pixels(device, queue, &target);
        expect_pixel(&edge_pixels, 32, 12, [255, 255, 255, 255]);
        expect_pixel(&edge_pixels, 32, 40, [0, 255, 0, 255]);
        expect_pixel(&edge_pixels, 32, 56, [0, 0, 0, 255]);
        let edge_capacity = renderer.edge_object_bytes.capacity();
        let edge_gpu_bytes = renderer.edge_object_buffer.size();
        let edge_frame_gpu_bytes = renderer.retained_frame_gpu_bytes();

        scene.set_visible(late_edge, false).unwrap();
        let restored = render(&mut renderer, &scene);
        assert_eq!(restored.upload_calls(), 0);
        assert_eq!(restored.uploaded_bytes(), 0);
        assert_eq!(restored.buffer_allocation_count(), 0);
        assert_eq!(restored.edge_count(), 0);
        assert_eq!(renderer.edge_object_bytes.len(), 0);
        assert_eq!(renderer.edge_object_bytes.capacity(), edge_capacity);
        assert_eq!(renderer.edge_object_buffer.size(), edge_gpu_bytes);
        assert_eq!(restored.retained_frame_gpu_bytes(), edge_frame_gpu_bytes);
        assert_eq!(
            surface::test_read_pixels(device, queue, &target),
            surface_pixels
        );

        for _ in 17..33 {
            scene
                .try_push(&mesh, Transform3d::IDENTITY, surface_style)
                .unwrap();
        }
        let grown = render(&mut renderer, &scene);
        assert_eq!(grown.object_count(), 33);
        assert_eq!(grown.upload_calls(), 1);
        assert_eq!(grown.uploaded_bytes(), 33 * instance_bytes);
        assert_eq!(grown.buffer_allocation_count(), 1);
        assert_eq!(renderer.instance_capacity, 64);
        assert_eq!(renderer.edge_object_capacity, 32);
        assert_eq!(renderer.edge_object_buffer.size(), edge_gpu_bytes);
        assert_eq!(renderer.edge_object_bytes.capacity(), edge_capacity);
        assert_eq!(
            surface::test_read_pixels(device, queue, &target),
            surface_pixels
        );

        let instance_capacity = renderer.instances.capacity();
        let gpu_bytes = renderer.retained_frame_gpu_bytes();
        let empty = render(&mut renderer, &Scene3d::new(Color::BLACK).unwrap());
        assert_eq!(empty.object_count(), 0);
        assert_eq!(empty.edge_count(), 0);
        assert_eq!(empty.triangle_count(), 0);
        assert_eq!(empty.draw_call_count(), 0);
        assert_eq!(empty.upload_calls(), 0);
        assert_eq!(empty.uploaded_bytes(), 0);
        assert_eq!(empty.buffer_allocation_count(), 0);
        assert_eq!(renderer.instance_capacity, 64);
        assert_eq!(renderer.instances.capacity(), instance_capacity);
        assert_eq!(renderer.edge_object_capacity, 32);
        assert_eq!(renderer.edge_object_bytes.capacity(), edge_capacity);
        assert_eq!(renderer.retained_frame_gpu_bytes(), gpu_bytes);
        assert!(
            surface::test_read_pixels(device, queue, &target)
                .chunks_exact(4)
                .all(|pixel| pixel == [0, 0, 0, 255])
        );
    }
}

fn expect_pixel(pixels: &[u8], x: usize, y: usize, expected: [u8; 4]) {
    let offset = (y * 64 + x) * 4;
    assert_eq!(&pixels[offset..offset + 4], expected, "pixel ({x}, {y})");
}
