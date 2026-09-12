use super::*;

fn point(x: f32, y: f32) -> Vec3 {
    Vec3::new(x, y, 0.0).unwrap()
}

fn source(right: bool, short: bool, textured: bool) -> Mesh3d {
    let left = if right { 0.25 } else { -0.75 };
    let vertices = vec![
        point(left, -0.5),
        point(left + 0.5, -0.5),
        point(left + 0.5, 0.5),
        point(left, 0.5),
    ];
    let indices = if short {
        vec![0, 1, 2]
    } else {
        vec![0, 1, 2, 0, 2, 3]
    };
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        .into_iter()
        .map(|(start, end)| MeshEdge3d::new(start, end).unwrap())
        .collect();
    if textured {
        Mesh3d::textured(
            vertices,
            vec![crate::TextureCoordinate2d::new(if right { 0.75 } else { 0.25 }, 0.5).unwrap(); 4],
            indices,
            edges,
        )
        .unwrap()
    } else {
        Mesh3d::with_display_edges(vertices, indices, edges).unwrap()
    }
}

fn camera() -> Camera3d {
    Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(4.0)).unwrap(),
    )
    .unwrap()
}

#[test]
fn dynamic_capacity_accepts_non_triangle_reserve_and_rejects_overflow() {
    let source = source(false, true, true);
    let live = preflight_mesh3d_source(&source, u64::MAX).unwrap();
    let allocation = planned_allocation(
        live,
        &source,
        DynamicMesh3dBudget::default().with_minimum_capacity(9, 5, 7),
        u64::MAX,
    )
    .unwrap();
    assert_eq!(allocation.index_bytes, 5 * 4);
    assert_eq!(
        allocation.vertex_bytes,
        9 * std::mem::size_of::<MeshVertexGpu>() as u64
    );
    assert_eq!(
        allocation.texture_coordinate_bytes,
        9 * std::mem::size_of::<MeshUvGpu>() as u64
    );
    for budget in [
        DynamicMesh3dBudget::default().with_minimum_capacity(usize::MAX, 0, 0),
        DynamicMesh3dBudget::default().with_minimum_capacity(0, usize::MAX, 0),
        DynamicMesh3dBudget::default().with_minimum_capacity(0, 0, usize::MAX),
    ] {
        assert_eq!(
            planned_allocation(live, &source, budget, 1024),
            Err(Mesh3dResourceError::CapacityTooLarge)
        );
    }
}

#[test]
fn dynamic_conversion_scratch_reuses_exact_capacity_and_bounds_peak() {
    let mut scratch = DynamicMesh3dScratch::default();
    let first = source(false, false, true);
    let budget = DynamicMesh3dBudget::default();
    assert_eq!(scratch.prepare(&first, budget).unwrap().0, 3);
    let bytes = scratch.capacity_bytes();
    assert_eq!(
        scratch.prepare(&source(true, true, true), budget).unwrap(),
        (0, bytes)
    );
    let pointer = scratch.vertices.as_ptr();
    let error = scratch.prepare(
        &first,
        budget.with_peak_limits(usize::MAX, usize::MAX, bytes - 1),
    );
    assert!(matches!(
        error,
        Err(DynamicMesh3dError::BudgetExceeded {
            resource: DynamicMesh3dBudgetResource::PeakStagingBytes,
            ..
        })
    ));
    assert_eq!(scratch.vertices.as_ptr(), pointer);
    assert_eq!(scratch.capacity_bytes(), bytes);
}

fn pixels(
    renderer: &mut Mesh3dRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    target: &RenderTarget3d,
    scene: &Scene3d,
) -> Vec<u8> {
    let report = renderer
        .render_scene3d(
            device,
            queue,
            identity,
            target,
            scene,
            camera(),
            Mesh3dRenderBudget::default().with_surface_policy(SurfaceRasterization3d::Native),
        )
        .unwrap();
    assert_eq!(
        report.triangle_count(),
        scene
            .instances()
            .iter()
            .filter(|instance| instance.visible)
            .map(|instance| instance.mesh.triangle_count())
            .sum::<usize>()
    );
    surface::test_read_pixels(device, queue, target)
}

pub(in crate::renderer::mesh3d) fn assert_gpu_dynamic_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = surface::test_target(device, &identity, format, 64, 64);
    let texture = texture::create_texture(
        device,
        queue,
        &identity,
        &renderer.textures.layout,
        2,
        1,
        vec![255, 0, 0, 255, 0, 255, 0, 255],
        ImageBudget::default(),
    )
    .unwrap();
    let material = TextureMaterial3d::new(&texture, ImageSampling::Nearest, Color::WHITE).unwrap();
    let original = create_retained_mesh(
        device,
        queue,
        Arc::clone(&identity),
        source(false, false, true),
    )
    .unwrap();
    let original = texture::attach_material(&identity, &original, &material).unwrap();
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap());
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let id = scene
        .try_push(&original, Transform3d::IDENTITY, style)
        .unwrap();
    let other = scene
        .try_push(&original, Transform3d::IDENTITY, style)
        .unwrap();
    scene.set_visible(other, false).unwrap();
    let original_pixels = pixels(&mut renderer, device, queue, &identity, &target, &scene);
    let budget = DynamicMesh3dBudget::default().with_minimum_capacity(16, 25, 12);
    let report = update_scene_mesh(
        device,
        queue,
        &identity,
        &mut renderer.dynamic_scratch,
        &mut scene,
        id,
        source(true, false, true),
        budget,
    )
    .unwrap();
    assert!(report.detached_aliases());
    assert!(report.grew_capacity());
    assert_eq!(report.gpu_allocation_count(), 4);
    assert_eq!(report.upload_calls(), 4);
    assert_eq!(report.scene_statistics().mesh_count(), 2);
    assert_eq!(report.scene_statistics().texture_count(), 1);
    assert_eq!(
        report.scene_statistics().mesh_gpu_bytes(),
        original.gpu_allocation_bytes() + report.capacity_gpu_bytes()
    );
    assert!(Arc::ptr_eq(
        &scene.instance(other).unwrap().mesh.vertex_buffer,
        &original.vertex_buffer
    ));
    assert!(!Arc::ptr_eq(
        &scene.instance(id).unwrap().mesh.vertex_buffer,
        &original.vertex_buffer
    ));
    assert_eq!(
        scene
            .instance(id)
            .unwrap()
            .mesh
            .material()
            .unwrap()
            .texture()
            .identity_key(),
        texture.identity_key()
    );
    let changed_pixels = pixels(&mut renderer, device, queue, &identity, &target, &scene);
    assert_ne!(changed_pixels, original_pixels);
    assert!(changed_pixels.chunks_exact(4).any(|pixel| pixel[1] == 255));
    let pointer = Arc::as_ptr(&scene.instance(id).unwrap().mesh.vertex_buffer);
    let report = update_scene_mesh(
        device,
        queue,
        &identity,
        &mut renderer.dynamic_scratch,
        &mut scene,
        id,
        source(false, false, true),
        budget,
    )
    .unwrap();
    assert!(report.reused_buffers());
    assert_eq!(report.gpu_allocation_count(), 0);
    assert_eq!(report.scratch_reallocations(), 0);
    assert_eq!(
        Arc::as_ptr(&scene.instance(id).unwrap().mesh.vertex_buffer),
        pointer
    );
    assert_eq!(
        pixels(&mut renderer, device, queue, &identity, &target, &scene),
        original_pixels
    );
    let report = update_scene_mesh(
        device,
        queue,
        &identity,
        &mut renderer.dynamic_scratch,
        &mut scene,
        id,
        source(false, true, true),
        budget,
    )
    .unwrap();
    assert!(report.reused_buffers());
    assert_eq!(scene.instance(id).unwrap().mesh.index_count, 3);
    let short_pixels = pixels(&mut renderer, device, queue, &identity, &target, &scene);
    assert_ne!(
        short_pixels, original_pixels,
        "shrinking must not draw the old triangle tail"
    );
    assert_eq!(
        scene.instance(id).unwrap().mesh.source().bounds_max(),
        point(-0.25, 0.5)
    );

    let snapshot = scene.instance(id).unwrap().mesh().clone();
    let snapshot_source = snapshot.source().clone();
    let report = update_scene_mesh(
        device,
        queue,
        &identity,
        &mut renderer.dynamic_scratch,
        &mut scene,
        id,
        source(true, true, true),
        budget,
    )
    .unwrap();
    assert!(report.detached_aliases());
    assert!(!report.grew_capacity());
    assert_eq!(snapshot.source(), &snapshot_source);
    let mut old_scene = Scene3d::new(Color::BLACK).unwrap();
    old_scene
        .try_push(&snapshot, Transform3d::IDENTITY, style)
        .unwrap();
    assert_eq!(
        pixels(&mut renderer, device, queue, &identity, &target, &old_scene),
        short_pixels
    );
    let before = pixels(&mut renderer, device, queue, &identity, &target, &scene);
    let before_stats = scene.statistics();
    let before_buffer = Arc::as_ptr(&scene.instance(id).unwrap().mesh.vertex_buffer);
    for rejected in [
        budget.with_peak_limits(usize::MAX, 0, usize::MAX),
        budget.with_peak_limits(0, usize::MAX, usize::MAX),
        budget.with_peak_limits(usize::MAX, usize::MAX, 0),
        budget.with_minimum_capacity(usize::MAX, 0, 0),
    ] {
        assert!(
            update_scene_mesh(
                device,
                queue,
                &identity,
                &mut renderer.dynamic_scratch,
                &mut scene,
                id,
                source(false, false, true),
                rejected
            )
            .is_err()
        );
        assert_eq!(scene.statistics(), before_stats);
        assert_eq!(
            Arc::as_ptr(&scene.instance(id).unwrap().mesh.vertex_buffer),
            before_buffer
        );
        assert_eq!(
            pixels(&mut renderer, device, queue, &identity, &target, &scene),
            before
        );
    }
    assert!(matches!(
        update_scene_mesh(
            device,
            queue,
            &identity,
            &mut renderer.dynamic_scratch,
            &mut scene,
            id,
            source(false, false, false),
            budget
        ),
        Err(DynamicMesh3dError::Resource(Mesh3dResourceError::Texture(
            Texture3dError::MissingTextureCoordinates
        )))
    ));
    assert_eq!(scene.statistics(), before_stats);
    assert_eq!(
        pixels(&mut renderer, device, queue, &identity, &target, &scene),
        before
    );

    let limited = Scene3dBudget::new(2, 4096, 4096, original.gpu_allocation_bytes()).unwrap();
    let mut limited_scene = Scene3d::with_budget(Color::BLACK, limited).unwrap();
    let limited_id = limited_scene
        .try_push(&original, Transform3d::IDENTITY, style)
        .unwrap();
    let before_stats = limited_scene.statistics();
    let before = pixels(
        &mut renderer,
        device,
        queue,
        &identity,
        &target,
        &limited_scene,
    );
    assert!(matches!(
        update_scene_mesh(
            device,
            queue,
            &identity,
            &mut renderer.dynamic_scratch,
            &mut limited_scene,
            limited_id,
            source(true, false, true),
            budget
        ),
        Err(DynamicMesh3dError::Scene(Scene3dError::BudgetExceeded {
            resource: Scene3dBudgetResource::MeshGpuBytes,
            ..
        }))
    ));
    assert_eq!(limited_scene.statistics(), before_stats);
    assert_eq!(
        pixels(
            &mut renderer,
            device,
            queue,
            &identity,
            &target,
            &limited_scene
        ),
        before
    );
    assert_gpu_dynamic_layout(device, queue, format);
}

fn assert_gpu_dynamic_layout(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = surface::test_target(device, &identity, format, 64, 64);
    let mesh = create_retained_mesh(
        device,
        queue,
        Arc::clone(&identity),
        source(false, false, false),
    )
    .unwrap();
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap())
        .with_wireframe(WireframeStyle3d::visible(Color::WHITE, logical(2.0)).unwrap());
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let id = scene.try_push(&mesh, Transform3d::IDENTITY, style).unwrap();
    drop(mesh);
    let budget = DynamicMesh3dBudget::default().with_minimum_capacity(16, 25, 12);
    let plain_pixels = pixels(&mut renderer, device, queue, &identity, &target, &scene);
    for textured in [true, false] {
        let previous = Arc::as_ptr(&scene.instance(id).unwrap().mesh.vertex_buffer);
        let report = update_scene_mesh(
            device,
            queue,
            &identity,
            &mut renderer.dynamic_scratch,
            &mut scene,
            id,
            source(false, false, textured),
            budget,
        )
        .unwrap();
        assert!(!report.detached_aliases());
        assert_eq!(report.gpu_allocation_count(), if textured { 4 } else { 3 });
        assert_ne!(
            Arc::as_ptr(&scene.instance(id).unwrap().mesh.vertex_buffer),
            previous
        );
        assert_eq!(
            scene
                .instance(id)
                .unwrap()
                .mesh
                .texture_coordinate_buffer
                .is_some(),
            textured
        );
        assert_eq!(
            pixels(&mut renderer, device, queue, &identity, &target, &scene),
            plain_pixels
        );
    }
    let edge_only = Mesh3d::with_display_edges(
        vec![point(-0.75, 0.0), point(-0.25, 0.0)],
        vec![],
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let report = update_scene_mesh(
        device,
        queue,
        &identity,
        &mut renderer.dynamic_scratch,
        &mut scene,
        id,
        edge_only,
        budget,
    )
    .unwrap();
    assert!(report.reused_buffers());
    assert_eq!(scene.instance(id).unwrap().mesh.index_count, 0);
    assert!(scene.instance(id).unwrap().mesh.index_buffer.is_some());
    assert_eq!(scene.instance(id).unwrap().mesh.edge_count, 1);
    let edge_pixels = pixels(&mut renderer, device, queue, &identity, &target, &scene);
    assert!(edge_pixels.chunks_exact(4).any(|pixel| pixel[0] > 200));
    assert_ne!(edge_pixels, plain_pixels);
    let before_stats = scene.statistics();
    let error = update_scene_mesh(
        device,
        queue,
        &Arc::new(()),
        &mut renderer.dynamic_scratch,
        &mut scene,
        id,
        source(true, false, false),
        budget,
    );
    assert!(matches!(
        error,
        Err(DynamicMesh3dError::Resource(
            Mesh3dResourceError::RendererMismatch
        ))
    ));
    assert_eq!(scene.statistics(), before_stats);
    assert_eq!(
        pixels(&mut renderer, device, queue, &identity, &target, &scene),
        edge_pixels
    );
    let removed = scene.remove(id).unwrap();
    let replacement = scene
        .try_push(removed.mesh(), Transform3d::IDENTITY, style)
        .unwrap();
    assert_ne!(replacement, id);
    let before_stats = scene.statistics();
    assert!(
        matches!(update_scene_mesh(device, queue, &identity, &mut renderer.dynamic_scratch, &mut scene, id, source(true, false, false), budget), Err(DynamicMesh3dError::Scene(Scene3dError::ObjectNotFound { object_id })) if object_id == id)
    );
    assert_eq!(scene.statistics(), before_stats);
    assert_eq!(
        pixels(&mut renderer, device, queue, &identity, &target, &scene),
        edge_pixels
    );
}

pub(in crate::renderer::mesh3d) fn assert_gpu_dynamic_recovery(
    source_device: &wgpu::Device,
    source_queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
) {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let source_identity = Arc::new(());
    let recovery_identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(source_device, format);
    let mut recovered_renderer = Mesh3dRenderer::new(recovery_device, format);
    let source_target = surface::test_target(source_device, &source_identity, format, 64, 64);
    let recovery_target = surface::test_target(recovery_device, &recovery_identity, format, 64, 64);
    let mesh = create_retained_mesh(
        source_device,
        source_queue,
        Arc::clone(&source_identity),
        source(false, false, false),
    )
    .unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let id = scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
        )
        .unwrap();
    drop(mesh);
    let budget = DynamicMesh3dBudget::default().with_minimum_capacity(16, 25, 12);
    update_scene_mesh(
        source_device,
        source_queue,
        &source_identity,
        &mut renderer.dynamic_scratch,
        &mut scene,
        id,
        source(true, true, false),
        budget,
    )
    .unwrap();
    let before_pixels = pixels(
        &mut renderer,
        source_device,
        source_queue,
        &source_identity,
        &source_target,
        &scene,
    );
    let allocation = scene.instance(id).unwrap().mesh.allocation;
    let source = scene.instance(id).unwrap().mesh.source().clone();
    let prepared =
        upload::prepare_restoration(recovery_device, scene.instance(id).unwrap().mesh()).unwrap();
    assert_eq!(prepared.allocation, allocation);
    let single = upload_prepared_retained_mesh(
        recovery_device,
        recovery_queue,
        Arc::clone(&recovery_identity),
        prepared,
    );
    assert_eq!(single.allocation, allocation);
    assert_eq!(single.source(), &source);
    restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        &recovered_renderer.textures.layout,
        Arc::clone(&recovery_identity),
        &mut scene,
    )
    .unwrap();
    assert_eq!(scene.instance(id).unwrap().mesh.allocation, allocation);
    assert_eq!(scene.instance(id).unwrap().mesh.source(), &source);
    assert_eq!(
        pixels(
            &mut recovered_renderer,
            recovery_device,
            recovery_queue,
            &recovery_identity,
            &recovery_target,
            &scene
        ),
        before_pixels
    );
    let pointer = Arc::as_ptr(&scene.instance(id).unwrap().mesh.vertex_buffer);
    let report = update_scene_mesh(
        recovery_device,
        recovery_queue,
        &recovery_identity,
        &mut recovered_renderer.dynamic_scratch,
        &mut scene,
        id,
        source.clone(),
        budget,
    )
    .unwrap();
    assert!(report.reused_buffers());
    assert_eq!(
        Arc::as_ptr(&scene.instance(id).unwrap().mesh.vertex_buffer),
        pointer
    );
}
