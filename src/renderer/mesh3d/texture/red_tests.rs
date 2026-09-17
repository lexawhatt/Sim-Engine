//! Independent adversarial coverage for texture revisions and mixed pipelines.

use super::*;

fn quad() -> Mesh3d {
    Mesh3d::textured(
        vec![
            Vec3::new(-0.8, 0.8, 0.0).unwrap(),
            Vec3::new(0.8, 0.8, 0.0).unwrap(),
            Vec3::new(0.8, -0.8, 0.0).unwrap(),
            Vec3::new(-0.8, -0.8, 0.0).unwrap(),
        ],
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .map(|(u, v)| TextureCoordinate2d::new(u, v).unwrap())
            .to_vec(),
        vec![0, 1, 2, 0, 2, 3],
        Vec::new(),
    )
    .unwrap()
}

fn style(color: Color) -> MeshStyle3d {
    MeshStyle3d::surface(SurfaceStyle3d::opaque(color).unwrap())
}

fn transform(x: f32) -> Transform3d {
    Transform3d::new(
        Vec3::new(x, 0.0, 0.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(0.3, 0.6, 1.0).unwrap(),
    )
    .unwrap()
}

pub(super) fn assert_texture_atomicity_and_rebinding(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let source = quad();
    let mesh = create_retained_mesh(device, queue, Arc::clone(&identity), source.clone()).unwrap();
    assert_upload_limits_precede_staging(device, &source);

    let textures = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]].map(|pixels| {
        create_texture(
            device,
            queue,
            &identity,
            &renderer.textures.layout,
            1,
            1,
            pixels.to_vec(),
            ImageBudget::new(1, 1, 4).unwrap(),
        )
        .unwrap()
    });
    let [red_mesh, green_mesh, blue_mesh] = textures.each_ref().map(|texture| {
        attach_material(
            &identity,
            &mesh,
            &TextureMaterial3d::new(texture, ImageSampling::Nearest, Color::WHITE).unwrap(),
        )
        .unwrap()
    });
    assert!(Arc::ptr_eq(
        &red_mesh.vertex_buffer,
        &green_mesh.vertex_buffer
    ));
    assert!(Arc::ptr_eq(
        red_mesh.texture_coordinate_buffer.as_ref().unwrap(),
        green_mesh.texture_coordinate_buffer.as_ref().unwrap()
    ));

    let mut scene = Scene3d::with_budget(
        Color::BLACK,
        Scene3dBudget::default().with_texture_limits(8, 8),
    )
    .unwrap();
    let left = scene
        .try_push(&red_mesh, transform(-0.65), style(Color::WHITE))
        .unwrap();
    let center = scene
        .try_push(&red_mesh, transform(0.0), style(Color::WHITE))
        .unwrap();
    let right = scene
        .try_push(&green_mesh, transform(0.65), style(Color::WHITE))
        .unwrap();
    assert_eq!(scene.statistics().texture_count(), 2);
    assert_eq!(
        scene.statistics().mesh_gpu_bytes(),
        mesh.gpu_allocation_bytes()
    );
    let before = scene.statistics();
    let (error, allocations) = crate::test_allocations::count(|| scene.set_mesh(left, &blue_mesh));
    assert!(matches!(
        error,
        Err(Scene3dError::BudgetExceeded {
            resource: Scene3dBudgetResource::TextureCpuBytes,
            limit: 8,
            actual: 12
        })
    ));
    assert_eq!(
        allocations, 0,
        "rejected texture rebind allocated bookkeeping"
    );
    assert_eq!(scene.statistics(), before);
    assert_eq!(
        scene.instances()[0]
            .mesh
            .material()
            .unwrap()
            .texture()
            .identity_key(),
        textures[0].identity_key()
    );

    // A successful last-reference retirement frees exactly one texture slot;
    // the topology remains one allocation across all material variants.
    scene.remove(center).unwrap();
    let (result, allocations) = crate::test_allocations::count(|| scene.set_mesh(left, &blue_mesh));
    result.unwrap();
    assert_eq!(allocations, 0);
    assert_eq!(scene.statistics().texture_count(), 2);
    assert_eq!(scene.statistics().texture_cpu_bytes(), 8);
    assert_eq!(scene.statistics().texture_gpu_bytes(), 8);
    assert_eq!(
        scene.statistics().mesh_gpu_bytes(),
        mesh.gpu_allocation_bytes()
    );

    // Interleave textured and untextured revisions. Pipeline vertex slot one
    // changes meaning between UV and instance data and must be rebound.
    scene.remove(right).unwrap();
    scene
        .try_push(&mesh, transform(0.0), style(Color::rgb(1.0, 0.0, 0.0)))
        .unwrap();
    scene
        .try_push(&green_mesh, transform(0.65), style(Color::WHITE))
        .unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(4.0)).unwrap(),
    )
    .unwrap();
    let target = surface::test_target(device, &identity, format, 64, 64);
    renderer
        .render_scene3d(
            device,
            queue,
            &identity,
            &target,
            &scene,
            camera,
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    let pixels = surface::test_read_pixels(device, queue, &target);
    let red_channel = usize::from(matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    )) * 2;
    let blue_channel = 2 - red_channel;
    for (x, channel) in [(11, blue_channel), (32, red_channel), (52, 1)] {
        let pixel = &pixels[(32 * 64 + x) * 4..][..4];
        assert!(
            pixel[channel] > 250 && pixel[3] == 255,
            "mixed pipeline pixel at{x}: {pixel:?}"
        );
        for other in 0..3 {
            if other != channel {
                assert!(pixel[other] < 3, "mixed pipeline channel leak: {pixel:?}");
            }
        }
    }

    let mut retained = blue_mesh.clone();
    let untextured = Mesh3d::new(
        source.vertices().to_vec(),
        source.triangle_indices().to_vec(),
    )
    .unwrap();
    let (result, allocations) = crate::test_allocations::count(|| {
        upload::replace_resources(
            device,
            queue,
            &identity,
            &mut retained,
            untextured,
            Mesh3dUploadBudget::default(),
        )
    });
    assert!(matches!(
        result,
        Err(Mesh3dResourceError::Texture(
            Texture3dError::MissingTextureCoordinates
        ))
    ));
    assert_eq!(allocations, 0);
    assert!(Arc::ptr_eq(
        &retained.vertex_buffer,
        &blue_mesh.vertex_buffer
    ));

    let foreign_identity = Arc::new(());
    let material =
        TextureMaterial3d::new(&textures[0], ImageSampling::Linear, Color::WHITE).unwrap();
    let (result, allocations) =
        crate::test_allocations::count(|| attach_material(&foreign_identity, &mesh, &material));
    assert!(matches!(result, Err(Texture3dError::RendererMismatch)));
    assert_eq!(allocations, 0);
    renderer
        .render_scene3d(
            device,
            queue,
            &identity,
            &target,
            &scene,
            camera,
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    assert_eq!(surface::test_read_pixels(device, queue, &target), pixels);

    // Rejection must happen before image/GPU staging, even when the payload is
    // deliberately too short to correspond to the over-limit dimensions.
    let pixels = vec![255; 4];
    let (result, allocations) = crate::test_allocations::count(|| {
        create_texture(
            device,
            queue,
            &identity,
            &renderer.textures.layout,
            device.limits().max_texture_dimension_2d + 1,
            1,
            pixels,
            ImageBudget::default(),
        )
    });
    assert!(matches!(
        result,
        Err(Texture3dError::Image(ImageError::DimensionsTooLarge))
    ));
    assert_eq!(allocations, 0);
    assert_asymmetric_uv_clipping(device, queue, &identity, &mut renderer, &target);
}

fn assert_asymmetric_uv_clipping(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    renderer: &mut Mesh3dRenderer,
    target: &RenderTarget3d,
) {
    // Both texture axes vary independently. A row-only field cannot detect
    // incorrect U interpolation while its V interpolation remains correct.
    let pixels = (0u8..4)
        .flat_map(|y| {
            (0u8..4).flat_map(move |x| [20 + 70 * x, 10 + 75 * y, 15 + 17 * x + 13 * y, 255])
        })
        .collect();
    let texture = create_texture(
        device,
        queue,
        identity,
        &renderer.textures.layout,
        4,
        4,
        pixels,
        ImageBudget::default(),
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(0.5), world(4.0))
            .unwrap(),
    )
    .unwrap();
    for reverse in [false, true] {
        let source = quad();
        let mut indices = source.triangle_indices().to_vec();
        if reverse {
            for triangle in indices.chunks_exact_mut(3) {
                triangle.swap(1, 2);
            }
        }
        let source = Mesh3d::textured(
            source.vertices().to_vec(),
            source.texture_coordinates().to_vec(),
            indices,
            Vec::new(),
        )
        .unwrap();
        let mesh = create_retained_mesh(device, queue, Arc::clone(identity), source).unwrap();
        for sampling in [ImageSampling::Nearest, ImageSampling::Linear] {
            let mesh = attach_material(
                identity,
                &mesh,
                &TextureMaterial3d::new(&texture, sampling, Color::WHITE).unwrap(),
            )
            .unwrap();
            for (position, rotation) in [
                ([1.8, 0.0, 0.0], [0.15, 0.2, 0.07]),
                ([0.0, 0.0, 1.45], [0.2, 0.6, 0.1]),
                ([0.0, 0.0, -2.0], [0.2, 0.6, 0.1]),
            ] {
                let transform = Transform3d::new(
                    Vec3::new(position[0], position[1], position[2]).unwrap(),
                    Rotation3d::from_euler_xyz(rotation[0], rotation[1], rotation[2]).unwrap(),
                    Vec3::new(1.0, 1.0, 1.0).unwrap(),
                )
                .unwrap();
                let mut scene = Scene3d::new(Color::BLACK).unwrap();
                scene
                    .try_push(&mesh, transform, style(Color::WHITE))
                    .unwrap();
                let report = renderer
                    .render_scene3d(
                        device,
                        queue,
                        identity,
                        target,
                        &scene,
                        camera,
                        Mesh3dRenderBudget::default(),
                    )
                    .unwrap();
                assert!(report.preflight.generated_triangles > 0);
                let actual = surface::test_read_pixels(device, queue, target);
                // Bypass only generated CPU topology for the independent
                // reference; run the retained WGSL path and hardware clipping.
                renderer.clipped_surface_objects.clear();
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                encode_scene_pass(
                    &mut encoder,
                    renderer,
                    &target.color.view,
                    &target.depth_view,
                    scene.background(),
                    scene.instances(),
                );
                queue.submit([encoder.finish()]);
                let reference = surface::test_read_pixels(device, queue, target);
                let mismatches = actual
                    .chunks_exact(4)
                    .zip(reference.chunks_exact(4))
                    .filter(|(actual, reference)| {
                        (0..3).any(|channel| actual[channel].abs_diff(reference[channel]) > 8)
                    })
                    .count();
                assert!(
                    mismatches <= 4,
                    "asymmetric UV mismatch={mismatches} reverse={reverse} sampling={sampling:?} position={position:?}"
                );
                assert!(
                    actual
                        .chunks_exact(4)
                        .any(|pixel| pixel[0] > 0 && pixel[1] > 0 && pixel[2] > 0),
                    "empty asymmetric UV oracle"
                );
            }
        }
    }
}

fn assert_upload_limits_precede_staging(device: &wgpu::Device, source: &Mesh3d) {
    let layout = preflight_mesh3d_source(source, device.limits().max_buffer_size).unwrap();
    let staging =
        (layout.vertex_bytes + layout.edge_bytes + layout.texture_coordinate_bytes) as usize;
    assert!(layout.texture_coordinate_bytes > 0);
    for (resource, gpu_bytes, staging_bytes) in [
        (
            Mesh3dUploadBudgetResource::GpuBytes,
            layout.total_bytes as usize - 1,
            staging,
        ),
        (
            Mesh3dUploadBudgetResource::StagingBytes,
            layout.total_bytes as usize,
            staging - 1,
        ),
    ] {
        let budget =
            Mesh3dUploadBudget::new(source.recovery_memory_bytes(), gpu_bytes, staging_bytes)
                .unwrap();
        let (result, allocations) = crate::test_allocations::count(|| {
            upload::prepare_with_budget(device, source.clone(), budget)
        });
        assert!(
            matches!(result, Err(Mesh3dResourceError::BudgetExceeded { resource: actual, .. }) if actual == resource)
        );
        assert_eq!(
            allocations, 0,
            "UV-inclusive capacity rejection allocated staging"
        );
    }
    let prepared = upload::prepare_with_budget(
        device,
        source.clone(),
        Mesh3dUploadBudget::new(
            source.recovery_memory_bytes(),
            layout.total_bytes as usize,
            staging,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(prepared.texture_coordinates.len(), source.vertices().len());
    assert_eq!(prepared.layout.total_bytes, layout.total_bytes);
}
