use super::*;

fn quad(uv: [TextureCoordinate2d; 4]) -> Mesh3d {
    Mesh3d::textured(
        vec![
            Vec3::new(-0.8, 0.8, 0.0).unwrap(),
            Vec3::new(0.8, 0.8, 0.0).unwrap(),
            Vec3::new(0.8, -0.8, 0.0).unwrap(),
            Vec3::new(-0.8, -0.8, 0.0).unwrap(),
        ],
        uv.to_vec(),
        vec![0, 1, 2, 0, 2, 3],
        Vec::new(),
    )
    .unwrap()
}

fn camera(perspective: bool) -> Camera3d {
    let projection = if perspective {
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(0.5), world(4.0)).unwrap()
    } else {
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(4.0)).unwrap()
    };
    Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        projection,
    )
    .unwrap()
}

fn style() -> MeshStyle3d {
    MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap())
}

pub(in crate::renderer::mesh3d) fn assert_gpu_texture_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = surface::test_target(device, &identity, format, 64, 64);
    let [red, green, blue] = if matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        [2, 1, 0]
    } else {
        [0, 1, 2]
    };
    let pixels = vec![
        255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255,
    ];
    let texture = create_texture(
        device,
        queue,
        &identity,
        &renderer.textures.layout,
        2,
        2,
        pixels,
        ImageBudget::default(),
    )
    .unwrap();
    let coordinates = texture
        .region_coordinates(ImageTexelRect::new(0, 0, 2, 2).unwrap())
        .unwrap();
    let mesh =
        create_retained_mesh(device, queue, Arc::clone(&identity), quad(coordinates)).unwrap();
    assert_eq!(
        mesh.texture_coordinate_buffer.as_ref().unwrap().size(),
        4 * 8
    );
    for sampling in [ImageSampling::Nearest, ImageSampling::Linear] {
        let material = TextureMaterial3d::new(&texture, sampling, Color::WHITE).unwrap();
        let clone_allocations = crate::test_allocations::count(|| material.clone()).1;
        assert_eq!(
            clone_allocations, 0,
            "shared material clone allocated recovery pixels"
        );
        let textured = attach_material(&identity, &mesh, &material).unwrap();
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        let object = scene
            .try_push(&textured, Transform3d::IDENTITY, style())
            .unwrap();
        renderer
            .render_scene3d(
                device,
                queue,
                &identity,
                &target,
                &scene,
                camera(false),
                Mesh3dRenderBudget::default(),
            )
            .unwrap();
        let bytes = surface::test_read_pixels(device, queue, &target);
        let top = &bytes[(9 * 64 + 32) * 4..][..4];
        let bottom = &bytes[(54 * 64 + 32) * 4..][..4];
        assert!(
            top[red] > 240 && top[blue] < 85,
            "texture top row inverted: {top:?}"
        );
        assert!(
            bottom[blue] > 240 && bottom[red] < 85,
            "texture bottom row inverted: {bottom:?}"
        );
        assert_eq!(scene.statistics().texture_count(), 1);
        assert_eq!(scene.statistics().texture_gpu_bytes(), 16);

        // Perspective clipping must interpolate UV with homogeneous edge t;
        // comparison includes both texture rows and a depth-writing opaque face.
        for (position, rotate) in [
            ([1.8, 0.0, 0.0], [0.15, 0.2, 0.07]),
            ([0.0, 0.0, 1.45], [0.2, 0.6, 0.1]),
            ([0.0, 0.0, -2.0], [0.2, 0.6, 0.1]),
        ] {
            scene
                .set_transform(
                    object,
                    Transform3d::new(
                        Vec3::new(position[0], position[1], position[2]).unwrap(),
                        Rotation3d::from_euler_xyz(rotate[0], rotate[1], rotate[2]).unwrap(),
                        Vec3::new(1.0, 1.0, 1.0).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap();
            let report = renderer
                .render_scene3d(
                    device,
                    queue,
                    &identity,
                    &target,
                    &scene,
                    camera(true),
                    Mesh3dRenderBudget::default(),
                )
                .unwrap();
            assert!(report.preflight.generated_triangles > 0);
            let actual = surface::test_read_pixels(device, queue, &target);
            renderer.clipped_surface_objects.clear();
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encode_scene_pass(
                &mut encoder,
                &renderer,
                &target.color.view,
                &target.depth_view,
                scene.background(),
                scene.instances(),
            );
            queue.submit([encoder.finish()]);
            let reference = surface::test_read_pixels(device, queue, &target);
            let mismatch = actual
                .chunks_exact(4)
                .zip(reference.chunks_exact(4))
                .filter(|(a, b)| (0..3).any(|channel| a[channel].abs_diff(b[channel]) > 8))
                .count();
            assert!(
                mismatch <= 4,
                "perspective UV/depth mismatch={mismatch} at{position:?} sampling={sampling:?}"
            );
        }
    }

    // At these exact centers neither filter may pick the neighboring cell.
    let atlas_pixels: Vec<_> = (0..8)
        .flat_map(|index| {
            if index % 4 < 2 {
                [0, 255, 0, 255]
            } else {
                [255, 0, 0, 255]
            }
        })
        .collect();
    let atlas = create_texture(
        device,
        queue,
        &identity,
        &renderer.textures.layout,
        4,
        2,
        atlas_pixels,
        ImageBudget::default(),
    )
    .unwrap();
    let coordinates = atlas
        .region_coordinates(ImageTexelRect::new(0, 0, 2, 2).unwrap())
        .unwrap();
    let base =
        create_retained_mesh(device, queue, Arc::clone(&identity), quad(coordinates)).unwrap();
    for sampling in [ImageSampling::Nearest, ImageSampling::Linear] {
        let material = TextureMaterial3d::new(&atlas, sampling, Color::rgb(1.0, 0.5, 1.0)).unwrap();
        let textured = attach_material(&identity, &base, &material).unwrap();
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        scene
            .try_push(&textured, Transform3d::IDENTITY, style())
            .unwrap();
        renderer
            .render_scene3d(
                device,
                queue,
                &identity,
                &target,
                &scene,
                camera(false),
                Mesh3dRenderBudget::default(),
            )
            .unwrap();
        let bytes = surface::test_read_pixels(device, queue, &target);
        for y in 7..57 {
            for x in 7..57 {
                let pixel = &bytes[(y * 64 + x) * 4..][..4];
                let expected = if format.is_srgb() { 188 } else { 128 };
                assert!(
                    pixel[red] < 2 && pixel[blue] < 2 && pixel[green].abs_diff(expected) <= 2,
                    "atlas bleed/tint error {pixel:?} at{x},{y}"
                );
            }
        }
    }
    assert_validation_and_accounting(device, queue, &identity, &renderer, &texture, &base);
    red_tests::assert_texture_atomicity_and_rebinding(device, queue, format);
}

fn assert_validation_and_accounting(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    renderer: &Mesh3dRenderer,
    texture: &Texture3d,
    mesh: &RetainedMesh3d,
) {
    assert!(matches!(
        create_texture(
            device,
            queue,
            identity,
            &renderer.textures.layout,
            1,
            1,
            vec![1, 2, 3, 254],
            ImageBudget::default()
        ),
        Err(Texture3dError::NonOpaquePixel { texel: 0 })
    ));
    assert!(matches!(
        TextureMaterial3d::new(
            texture,
            ImageSampling::Nearest,
            Color::rgba(1.0, 1.0, 1.0, 0.5)
        ),
        Err(Texture3dError::InvalidTint)
    ));
    let material = TextureMaterial3d::new(texture, ImageSampling::Nearest, Color::WHITE).unwrap();
    assert!(matches!(
        attach_material(&Arc::new(()), mesh, &material),
        Err(Texture3dError::RendererMismatch)
    ));
    let textured = attach_material(identity, mesh, &material).unwrap();
    let bounded = Scene3dBudget::default().with_texture_limits(
        texture.recovery_memory_bytes(),
        texture.gpu_allocation_bytes(),
    );
    let mut scene = Scene3d::with_budget(Color::BLACK, bounded).unwrap();
    for _ in 0..16 {
        scene
            .try_push(&textured, Transform3d::IDENTITY, style())
            .unwrap();
    }
    assert_eq!(scene.statistics().texture_count(), 1);
    assert_eq!(
        scene.statistics().texture_cpu_bytes(),
        texture.recovery_memory_bytes()
    );
    assert_eq!(
        scene.statistics().texture_gpu_bytes(),
        texture.gpu_allocation_bytes()
    );
    let other = create_texture(
        device,
        queue,
        identity,
        &renderer.textures.layout,
        2,
        2,
        vec![255; 16],
        ImageBudget::default(),
    )
    .unwrap();
    let other_mesh = attach_material(
        identity,
        mesh,
        &TextureMaterial3d::new(&other, ImageSampling::Nearest, Color::WHITE).unwrap(),
    )
    .unwrap();
    let original = scene.statistics();
    assert!(matches!(
        scene.set_mesh(scene.instances()[0].id(), &other_mesh),
        Err(Scene3dError::BudgetExceeded {
            resource: Scene3dBudgetResource::TextureCpuBytes,
            ..
        })
    ));
    assert_eq!(scene.statistics(), original);
    let mut forbidden = Scene3d::with_budget(
        Color::BLACK,
        Scene3dBudget::default().with_texture_limits(0, 0),
    )
    .unwrap();
    assert!(matches!(
        forbidden.try_push(&textured, Transform3d::IDENTITY, style()),
        Err(Scene3dError::BudgetExceeded {
            resource: Scene3dBudgetResource::TextureCpuBytes,
            ..
        })
    ));
    for id in scene
        .instances()
        .iter()
        .map(Mesh3dInstance::id)
        .collect::<Vec<_>>()
    {
        scene.remove(id).unwrap();
    }
    assert_eq!(scene.statistics().texture_count(), 0);
    assert_eq!(scene.statistics().texture_gpu_bytes(), 0);
}

pub(in crate::renderer::mesh3d) fn assert_gpu_textured_recovery(
    source_device: &wgpu::Device,
    source_queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
) {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let source_identity = Arc::new(());
    let recovery_identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(source_device, format);
    let mut restored_renderer = Mesh3dRenderer::new(recovery_device, format);
    let target = surface::test_target(source_device, &source_identity, format, 64, 64);
    let restored_target = surface::test_target(recovery_device, &recovery_identity, format, 64, 64);
    let texture = create_texture(
        source_device,
        source_queue,
        &source_identity,
        &renderer.textures.layout,
        2,
        2,
        vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ],
        ImageBudget::default(),
    )
    .unwrap();
    let source = quad(
        texture
            .region_coordinates(ImageTexelRect::new(0, 0, 2, 2).unwrap())
            .unwrap(),
    );
    let mesh = create_retained_mesh(
        source_device,
        source_queue,
        Arc::clone(&source_identity),
        source.clone(),
    )
    .unwrap();
    // A second upload shares CPU source but not GPU topology. Other variants
    // below share GPU topology while carrying distinct textures/filter/tints.
    let second_mesh = create_retained_mesh(
        source_device,
        source_queue,
        Arc::clone(&source_identity),
        source,
    )
    .unwrap();
    let second_texture = create_texture(
        source_device,
        source_queue,
        &source_identity,
        &renderer.textures.layout,
        2,
        2,
        vec![
            255, 255, 0, 255, 0, 255, 255, 255, 255, 0, 255, 255, 30, 20, 80, 255,
        ],
        ImageBudget::default(),
    )
    .unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    for index in 0..4 {
        let sampling = if index == 0 {
            ImageSampling::Nearest
        } else {
            ImageSampling::Linear
        };
        let material = TextureMaterial3d::new(
            if index == 1 {
                &second_texture
            } else {
                &texture
            },
            sampling,
            if index == 1 {
                Color::rgb(1.0, 0.5, 1.0)
            } else {
                Color::WHITE
            },
        )
        .unwrap();
        let textured = attach_material(
            &source_identity,
            if index == 2 { &second_mesh } else { &mesh },
            &material,
        )
        .unwrap();
        let transform = Transform3d::new(
            Vec3::new(
                if index % 2 == 0 { -0.5 } else { 0.5 },
                if index < 2 { 0.5 } else { -0.5 },
                0.0,
            )
            .unwrap(),
            Rotation3d::IDENTITY,
            Vec3::new(0.5, 0.5, 0.5).unwrap(),
        )
        .unwrap();
        scene.try_push(&textured, transform, style()).unwrap();
    }
    let ids: Vec<_> = scene.instances().iter().map(Mesh3dInstance::id).collect();
    renderer
        .render_scene3d(
            source_device,
            source_queue,
            &source_identity,
            &target,
            &scene,
            camera(false),
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    let before = surface::test_read_pixels(source_device, source_queue, &target);
    let report = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        &restored_renderer.textures.layout,
        Arc::clone(&recovery_identity),
        &mut scene,
    )
    .unwrap();
    assert_eq!(report.restored_mesh_count(), 2);
    assert_eq!(report.restored_texture_count(), 2);
    assert_eq!(report.restored_texture_bytes(), 32);
    assert_eq!(
        scene
            .instances()
            .iter()
            .map(Mesh3dInstance::id)
            .collect::<Vec<_>>(),
        ids
    );
    assert_eq!(
        scene.instances()[0].mesh.material().unwrap().sampling(),
        ImageSampling::Nearest
    );
    assert_eq!(
        scene.instances()[1].mesh.material().unwrap().sampling(),
        ImageSampling::Linear
    );
    assert_eq!(scene.statistics().texture_count(), 2);
    assert_eq!(
        scene.instances()[0]
            .mesh
            .material()
            .unwrap()
            .texture()
            .pixels(),
        texture.pixels()
    );
    assert_eq!(
        scene.instances()[1]
            .mesh
            .material()
            .unwrap()
            .texture()
            .pixels(),
        second_texture.pixels()
    );
    assert_eq!(
        scene.instances()[1].mesh.material().unwrap().tint(),
        Color::rgb(1.0, 0.5, 1.0)
    );
    assert!(Arc::ptr_eq(
        &scene.instances()[0].mesh.vertex_buffer,
        &scene.instances()[1].mesh.vertex_buffer
    ));
    assert!(!Arc::ptr_eq(
        &scene.instances()[0].mesh.vertex_buffer,
        &scene.instances()[2].mesh.vertex_buffer
    ));
    assert_eq!(
        scene.instances()[0].mesh.source.vertices().as_ptr(),
        scene.instances()[2].mesh.source.vertices().as_ptr()
    );
    assert_eq!(
        scene.instances()[0]
            .mesh
            .material()
            .unwrap()
            .texture()
            .identity_key(),
        scene.instances()[2]
            .mesh
            .material()
            .unwrap()
            .texture()
            .identity_key()
    );
    restored_renderer
        .render_scene3d(
            recovery_device,
            recovery_queue,
            &recovery_identity,
            &restored_target,
            &scene,
            camera(false),
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    assert_eq!(
        surface::test_read_pixels(recovery_device, recovery_queue, &restored_target),
        before
    );
}
