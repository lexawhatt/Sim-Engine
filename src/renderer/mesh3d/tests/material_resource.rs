use super::*;

fn camera() -> Camera3d {
    Camera3d::look_at(
        Vec3::new(0.0, 0.0, 3.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(8.0)).unwrap(),
    )
    .unwrap()
}

fn textured_source() -> Mesh3d {
    Mesh3d::textured(
        vec![
            Vec3::new(-0.8, -0.5, 0.0).unwrap(),
            Vec3::new(0.8, -0.5, 0.0).unwrap(),
            Vec3::new(0.0, 0.8, 0.0).unwrap(),
        ],
        vec![crate::TextureCoordinate2d::new(0.25, 0.5).unwrap(); 3],
        vec![0, 1, 2],
        vec![],
    )
    .unwrap()
}

pub(super) fn assert_gpu_sort_budget(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = surface::test_target(device, &identity, format, 64, 64);
    let background = Scene3d::with_alpha_background(Color::rgba(0.25, 0.5, 0.75, 0.5)).unwrap();
    renderer
        .render_scene3d(
            device,
            queue,
            &identity,
            &target,
            &background,
            camera(),
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    let before = surface::test_read_pixels(device, queue, &target);
    let mesh =
        create_retained_mesh(device, queue, Arc::clone(&identity), textured_source()).unwrap();
    let mut scene = Scene3d::with_alpha_background(Color::rgba(0.0, 0.0, 0.0, 0.0)).unwrap();
    let id = scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::blend(Color::rgba(1.0, 0.0, 0.0, 0.5)).unwrap()),
        )
        .unwrap();
    let exact = std::mem::size_of::<material::SurfaceDraw>();
    for limit in [0, exact - 1] {
        assert_eq!(
            renderer.render_scene3d(
                device,
                queue,
                &identity,
                &target,
                &scene,
                camera(),
                Mesh3dRenderBudget::default().with_max_sorting_bytes(limit)
            ),
            Err(Mesh3dRenderError::SortingBudgetExceeded {
                limit,
                actual: exact
            })
        );
        assert_eq!(surface::test_read_pixels(device, queue, &target), before);
    }
    let report = renderer
        .render_scene3d(
            device,
            queue,
            &identity,
            &target,
            &scene,
            camera(),
            Mesh3dRenderBudget::default().with_max_sorting_bytes(exact),
        )
        .unwrap();
    assert_eq!(report.preflight().sorting_capacity_bytes(), exact);
    scene
        .set_style(
            id,
            MeshStyle3d::surface(SurfaceStyle3d::mask(Color::WHITE, 0.5).unwrap()),
        )
        .unwrap();
    let report = renderer
        .render_scene3d(
            device,
            queue,
            &identity,
            &target,
            &scene,
            camera(),
            Mesh3dRenderBudget::default().with_max_sorting_bytes(0),
        )
        .unwrap();
    assert_eq!(report.preflight().sorting_capacity_bytes(), 0);
}

pub(super) fn assert_gpu_alpha_recovery(
    source_device: &wgpu::Device,
    source_queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
) {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let identity = Arc::new(());
    let recovered_identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(source_device, format);
    let mut recovered_renderer = Mesh3dRenderer::new(recovery_device, format);
    let source_target = surface::test_target(source_device, &identity, format, 64, 64);
    let recovery_target =
        surface::test_target(recovery_device, &recovered_identity, format, 64, 64);
    let pixels = vec![255, 128, 64, 128, 0, 255, 0, 0];
    assert!(matches!(
        texture::create_texture(
            source_device,
            source_queue,
            &identity,
            &renderer.textures.layout,
            2,
            1,
            pixels.clone(),
            ImageBudget::default()
        ),
        Err(Texture3dError::NonOpaquePixel { texel: 0 })
    ));
    let texture = texture::create_texture_with_alpha(
        source_device,
        source_queue,
        &identity,
        &renderer.textures.layout,
        2,
        1,
        pixels.clone(),
        ImageBudget::default(),
    )
    .unwrap();
    assert!(texture.preserves_alpha());
    assert!(matches!(
        TextureMaterial3d::new(&texture, ImageSampling::Nearest, Color::WHITE),
        Err(Texture3dError::NonOpaquePixel { texel: 0 })
    ));
    let first_material = TextureMaterial3d::with_alpha(
        &texture,
        ImageSampling::Nearest,
        Color::rgba(1.0, 1.0, 1.0, 0.75),
    )
    .unwrap();
    let second_material = TextureMaterial3d::with_alpha(
        &texture,
        ImageSampling::Linear,
        Color::rgba(0.5, 1.0, 1.0, 1.0),
    )
    .unwrap();
    let mesh = create_retained_mesh(
        source_device,
        source_queue,
        Arc::clone(&identity),
        textured_source(),
    )
    .unwrap();
    let first = texture::attach_material(&identity, &mesh, &first_material).unwrap();
    let second = texture::attach_material(&identity, &mesh, &second_material).unwrap();
    let mut scene = Scene3d::with_alpha_background(Color::rgba(0.25, 0.5, 0.75, 0.25)).unwrap();
    let transform = |x| {
        Transform3d::new(
            Vec3::new(x, 0.0, 0.0).unwrap(),
            Rotation3d::IDENTITY,
            Vec3::new(0.5, 0.5, 0.5).unwrap(),
        )
        .unwrap()
    };
    let first_id = scene
        .try_push(
            &first,
            transform(-0.5),
            MeshStyle3d::surface(SurfaceStyle3d::blend(Color::rgba(1.0, 1.0, 1.0, 0.5)).unwrap()),
        )
        .unwrap();
    let second_id = scene
        .try_push(
            &second,
            transform(0.5),
            MeshStyle3d::surface(
                SurfaceStyle3d::mask(Color::WHITE, 0.25)
                    .unwrap()
                    .with_sidedness(SurfaceSidedness3d::FrontOnly),
            ),
        )
        .unwrap();
    let styles = [
        scene.instance(first_id).unwrap().style(),
        scene.instance(second_id).unwrap().style(),
    ];
    renderer
        .render_scene3d(
            source_device,
            source_queue,
            &identity,
            &source_target,
            &scene,
            camera(),
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    let before = surface::test_read_pixels(source_device, source_queue, &source_target);
    let statistics = scene.statistics();
    let report = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        &recovered_renderer.textures.layout,
        Arc::clone(&recovered_identity),
        &mut scene,
    )
    .unwrap();
    assert_eq!(report.restored_mesh_count(), 1);
    assert_eq!(report.restored_texture_count(), 1);
    assert_eq!(scene.statistics(), statistics);
    assert_eq!(scene.instance(first_id).unwrap().style(), styles[0]);
    assert_eq!(scene.instance(second_id).unwrap().style(), styles[1]);
    for id in [first_id, second_id] {
        let material = scene.instance(id).unwrap().mesh().material().unwrap();
        assert!(material.texture().preserves_alpha());
        assert_eq!(material.texture().pixels(), pixels);
    }
    assert_eq!(
        scene
            .instance(first_id)
            .unwrap()
            .mesh()
            .material()
            .unwrap()
            .tint(),
        first_material.tint()
    );
    assert_eq!(
        scene
            .instance(second_id)
            .unwrap()
            .mesh()
            .material()
            .unwrap()
            .tint(),
        second_material.tint()
    );
    recovered_renderer
        .render_scene3d(
            recovery_device,
            recovery_queue,
            &recovered_identity,
            &recovery_target,
            &scene,
            camera(),
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    assert_eq!(
        surface::test_read_pixels(recovery_device, recovery_queue, &recovery_target),
        before
    );
    assert_eq!(texture.pixels(), pixels);
    assert!(texture.belongs_to(&identity));
}
