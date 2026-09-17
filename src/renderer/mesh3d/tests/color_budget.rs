use super::*;

#[test]
fn optional_vertex_colors_add_exact_upload_bytes_and_obey_device_limit() {
    let vertices = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
    let plain = Mesh3d::new(vertices.clone(), vec![0, 1, 2]).unwrap();
    let colored = Mesh3d::with_attributes(
        vertices,
        vec![0, 1, 2],
        vec![],
        crate::Mesh3dAttributes::new()
            .with_vertex_colors(vec![Color::WHITE; 3])
            .unwrap(),
    )
    .unwrap();
    let plain_layout = preflight_mesh3d_source(&plain, u64::MAX).unwrap();
    let colored_layout = preflight_mesh3d_source(&colored, u64::MAX).unwrap();
    assert_eq!(plain_layout.color_bytes, 0);
    assert_eq!(colored_layout.color_bytes, 48);
    assert_eq!(colored_layout.total_bytes, plain_layout.total_bytes + 48);
    assert!(preflight_mesh3d_source(&plain, 47).is_ok());
    assert_eq!(
        preflight_mesh3d_source(&colored, 47),
        Err(Mesh3dResourceError::CapacityTooLarge)
    );
}

pub(super) fn assert_gpu_color_budget(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = surface::test_target(device, &identity, format, 64, 64);
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(4.0)).unwrap(),
    )
    .unwrap();
    let background = Scene3d::new(Color::rgb(0.25, 0.5, 0.75)).unwrap();
    renderer
        .render_scene3d(
            device,
            queue,
            &identity,
            &target,
            &background,
            camera,
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    let before = surface::test_read_pixels(device, queue, &target);
    let source = Mesh3d::with_attributes(
        vec![
            Vec3::new(-2.0, -0.5, 0.0).unwrap(),
            Vec3::new(0.75, -0.5, 0.0).unwrap(),
            Vec3::new(0.75, 0.5, 0.0).unwrap(),
        ],
        vec![0, 1, 2],
        vec![],
        crate::Mesh3dAttributes::new()
            .with_vertex_colors(vec![Color::WHITE; 3])
            .unwrap(),
    )
    .unwrap();
    let live = preflight_mesh3d_source(&source, device.limits().max_buffer_size).unwrap();
    let staging = (live.vertex_bytes + live.color_bytes) as usize;
    let budget = Mesh3dUploadBudget::new(4096, live.total_bytes as usize, staging).unwrap();
    for small_budget in [
        Mesh3dUploadBudget::new(4096, live.total_bytes as usize - 1, staging).unwrap(),
        Mesh3dUploadBudget::new(4096, live.total_bytes as usize, staging - 1).unwrap(),
    ] {
        assert!(matches!(
            upload::prepare_with_budget(device, source.clone(), small_budget),
            Err(Mesh3dResourceError::BudgetExceeded { .. })
        ));
    }
    let prepared = upload::prepare_with_budget(device, source, budget).unwrap();
    assert_eq!(prepared.staging_capacity_bytes(), staging);
    let mesh = upload_prepared_retained_mesh(device, queue, Arc::clone(&identity), prepared);
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
        )
        .unwrap();
    let exact_bytes =
        6 * (std::mem::size_of::<SurfaceClipVertex>() + std::mem::size_of::<MeshColorGpu>());
    let exact = Mesh3dRenderBudget::new(6, 2, exact_bytes);
    let frame = renderer
        .preflight_scene3d(device, &identity, &target, &scene, camera, exact)
        .unwrap()
        .1;
    assert_eq!(frame.vertices.len(), 6);
    assert_eq!(frame.colors.len(), 6);
    assert_eq!(frame.report.generated_upload_bytes(), exact_bytes);
    assert_eq!(
        renderer.render_scene3d(
            device,
            queue,
            &identity,
            &target,
            &scene,
            camera,
            Mesh3dRenderBudget::new(6, 2, exact_bytes - 1)
        ),
        Err(Mesh3dRenderError::GeneratedGeometryCapacityTooLarge)
    );
    assert!(renderer.clipped_surface_buffer.is_none());
    assert!(renderer.clipped_color_buffer.is_none());
    assert_eq!(surface::test_read_pixels(device, queue, &target), before);
    let report = renderer
        .render_scene3d(device, queue, &identity, &target, &scene, camera, exact)
        .unwrap();
    assert_eq!(report.preflight().generated_upload_bytes(), exact_bytes);
    assert_ne!(surface::test_read_pixels(device, queue, &target), before);
}
