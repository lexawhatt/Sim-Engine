use super::*;
use crate::{AmbientLight3d, DirectionalLight3d, Lighting3d};

fn with_normals(source: Mesh3d, normal: Vec3) -> Mesh3d {
    let mut attributes = crate::Mesh3dAttributes::new()
        .with_normals(vec![normal; source.vertices().len()])
        .unwrap();
    if !source.vertex_colors().is_empty() {
        attributes = attributes
            .with_vertex_colors(source.vertex_colors().to_vec())
            .unwrap();
    }
    if !source.texture_coordinates().is_empty() {
        attributes = attributes.with_texture_coordinates(source.texture_coordinates().to_vec());
    }
    Mesh3d::with_attributes(
        source.vertices().to_vec(),
        source.triangle_indices().to_vec(),
        source.display_edges().to_vec(),
        attributes,
    )
    .unwrap()
}

pub(super) fn normal_source(normal: Vec3) -> Mesh3d {
    with_normals(
        colored_source_with_uv(false, false, Color::WHITE, true),
        normal,
    )
}

pub(super) fn illuminate(scene: &mut Scene3d) {
    scene.set_lighting(
        Lighting3d::new(AmbientLight3d::new(Color::WHITE, 0.0).unwrap()).with_directional(Some(
            DirectionalLight3d::new(Vec3::Z, Color::WHITE, 1.0).unwrap(),
        )),
    );
}

#[test]
fn normals_are_bounded_in_upload_capacity_and_conversion_scratch() {
    let source = normal_source(Vec3::Z);
    let live = preflight_mesh3d_source(&source, u64::MAX).unwrap();
    assert_eq!(live.normal_bytes, 4 * 12);
    let budget = DynamicMesh3dBudget::default().with_minimum_capacity(16, 25, 12);
    let allocation = planned_allocation(live, &source, budget, u64::MAX).unwrap();
    assert_eq!(allocation.normal_bytes, 16 * 12);
    let mut scratch = DynamicMesh3dScratch::default();
    let (allocations, _) = scratch.prepare(&source, budget).unwrap();
    assert_eq!(allocations, 5);
    assert_eq!(scratch.capacity_bytes(), 4 * (12 + 16 + 8 + 12) + 4 * 24);
    assert_eq!(
        scratch.prepare(&normal_source(Vec3::X), budget).unwrap().0,
        0
    );
    let tiny = DynamicMesh3dBudget::new(
        Mesh3dUploadBudget::new(4096, 4096, scratch.capacity_bytes() - 1).unwrap(),
    );
    assert!(scratch.prepare(&source, tiny).is_err());
    assert!(planned_allocation(live, &source, budget, 16 * 12 - 1).is_err());
}

pub(super) fn assert_gpu_normals(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = surface::test_target(device, &identity, format, 64, 64);
    let mesh =
        create_retained_mesh(device, queue, Arc::clone(&identity), normal_source(Vec3::Z)).unwrap();
    let mesh = texture::attach_material(
        &identity,
        &mesh,
        &test_material(device, queue, &identity, &renderer),
    )
    .unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    illuminate(&mut scene);
    let lit = MeshStyle3d::surface(
        SurfaceStyle3d::opaque(Color::WHITE)
            .unwrap()
            .with_lighting(SurfaceLighting3d::Lambert),
    );
    let id = scene.try_push(&mesh, Transform3d::IDENTITY, lit).unwrap();
    drop(mesh);
    let budget = DynamicMesh3dBudget::default().with_minimum_capacity(16, 25, 12);
    let update = |renderer: &mut Mesh3dRenderer, scene: &mut Scene3d, source, budget| {
        update_scene_mesh(
            device,
            queue,
            &identity,
            &mut renderer.dynamic_scratch,
            scene,
            id,
            source,
            budget,
        )
    };
    let grown = update(&mut renderer, &mut scene, normal_source(Vec3::Z), budget).unwrap();
    assert_eq!(grown.gpu_allocation_count(), 6);
    assert_eq!(grown.upload_calls(), 6);
    let pointer = Arc::as_ptr(
        scene
            .instance(id)
            .unwrap()
            .mesh
            .normal_buffer
            .as_ref()
            .unwrap(),
    );
    let lit_pixels = pixels(&mut renderer, device, queue, &identity, &target, &scene);
    let reused = update(&mut renderer, &mut scene, normal_source(Vec3::X), budget).unwrap();
    assert!(reused.reused_buffers());
    assert_eq!(reused.scratch_reallocations(), 0);
    assert_eq!(
        reused.uploaded_bytes(),
        4 * (12 + 16 + 8 + 12) + 6 * 4 + 4 * 24
    );
    assert_eq!(
        Arc::as_ptr(
            scene
                .instance(id)
                .unwrap()
                .mesh
                .normal_buffer
                .as_ref()
                .unwrap()
        ),
        pointer
    );
    let dark_pixels = pixels(&mut renderer, device, queue, &identity, &target, &scene);
    assert_ne!(lit_pixels, dark_pixels);
    let reference =
        create_retained_mesh(device, queue, Arc::clone(&identity), normal_source(Vec3::X)).unwrap();
    let reference = texture::attach_material(
        &identity,
        &reference,
        scene.instance(id).unwrap().mesh.material().unwrap(),
    )
    .unwrap();
    let mut reference_scene = Scene3d::new(Color::BLACK).unwrap();
    illuminate(&mut reference_scene);
    reference_scene
        .try_push(&reference, Transform3d::IDENTITY, lit)
        .unwrap();
    assert_eq!(
        pixels(
            &mut renderer,
            device,
            queue,
            &identity,
            &target,
            &reference_scene
        ),
        dark_pixels
    );
    update(&mut renderer, &mut scene, normal_source(Vec3::Z), budget).unwrap();
    let snapshot = scene.instance(id).unwrap().mesh.clone();
    let detached = update(&mut renderer, &mut scene, normal_source(Vec3::X), budget).unwrap();
    assert!(detached.detached_aliases());
    assert_eq!(detached.gpu_allocation_count(), 6);
    let mut snapshot_scene = Scene3d::new(Color::BLACK).unwrap();
    illuminate(&mut snapshot_scene);
    snapshot_scene
        .try_push(&snapshot, Transform3d::IDENTITY, lit)
        .unwrap();
    assert_eq!(
        pixels(
            &mut renderer,
            device,
            queue,
            &identity,
            &target,
            &snapshot_scene
        ),
        lit_pixels
    );
    let before = scene.statistics();
    let capacity = scene.instance(id).unwrap().mesh.gpu_allocation_bytes();
    let too_small =
        DynamicMesh3dBudget::new(Mesh3dUploadBudget::new(4096, capacity - 1, 4096).unwrap());
    assert!(update(&mut renderer, &mut scene, normal_source(Vec3::Z), too_small).is_err());
    assert_eq!(scene.statistics(), before);
    assert_eq!(
        pixels(&mut renderer, device, queue, &identity, &target, &scene),
        dark_pixels
    );
    let without = colored_source_with_uv(false, false, Color::WHITE, true);
    assert!(matches!(
        update(&mut renderer, &mut scene, without.clone(), budget),
        Err(DynamicMesh3dError::Scene(Scene3dError::MissingNormals))
    ));
    assert_eq!(scene.statistics(), before);
    assert_eq!(
        pixels(&mut renderer, device, queue, &identity, &target, &scene),
        dark_pixels
    );
    scene
        .set_style(
            id,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
        )
        .unwrap();
    let removed = update(&mut renderer, &mut scene, without, budget).unwrap();
    assert_eq!(removed.gpu_allocation_count(), 5);
    assert!(scene.instance(id).unwrap().mesh.normal_buffer.is_none());
    assert_eq!(scene.set_style(id, lit), Err(Scene3dError::MissingNormals));
    let added = update(&mut renderer, &mut scene, normal_source(Vec3::Z), budget).unwrap();
    assert_eq!(added.gpu_allocation_count(), 6);
    scene.set_style(id, lit).unwrap();
    assert_eq!(
        pixels(&mut renderer, device, queue, &identity, &target, &scene),
        lit_pixels
    );
    assert_eq!(
        pixels(
            &mut renderer,
            device,
            queue,
            &identity,
            &target,
            &snapshot_scene
        ),
        lit_pixels
    );
    // Generated attributes obey the same exact upload budget before target mutation.
    let crossing = Transform3d::new(
        Vec3::new(-0.5, 0.0, 0.0).unwrap(),
        crate::Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    scene.set_transform(id, crossing).unwrap();
    scene.set_fog(Some(
        Fog3d::new(Color::rgb(0.1, 0.2, 0.3), 0.0, 0.2).unwrap(),
    ));
    scene
        .set_style(
            id,
            MeshStyle3d::surface(
                SurfaceStyle3d::opaque(Color::WHITE)
                    .unwrap()
                    .with_lighting(SurfaceLighting3d::Lambert)
                    .with_fog(true),
            ),
        )
        .unwrap();
    let (_, frame) = renderer
        .preflight_scene3d(
            device,
            &identity,
            &target,
            &scene,
            camera(),
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    let generated = frame.report.generated_vertex_count();
    assert!(generated > 0);
    assert_eq!(
        frame.report.generated_upload_bytes(),
        generated * (24 + 16 + 16)
    );
    let exact = Mesh3dRenderBudget::new(generated, generated / 3, generated * 56);
    renderer
        .render_scene3d(device, queue, &identity, &target, &scene, camera(), exact)
        .unwrap();
    let before = surface::test_read_pixels(device, queue, &target);
    assert!(
        renderer
            .render_scene3d(
                device,
                queue,
                &identity,
                &target,
                &scene,
                camera(),
                Mesh3dRenderBudget::new(generated, generated / 3, generated * 56 - 1)
            )
            .is_err()
    );
    assert_eq!(surface::test_read_pixels(device, queue, &target), before);
    // Active lighting rejects erased source directions before touching the target;
    // the same source and extreme transform remain permitted in Native Unlit mode.
    update(&mut renderer, &mut scene, normal_source(Vec3::Y), budget).unwrap();
    scene.set_fog(None);
    let extreme = Transform3d::new(
        Vec3::ZERO,
        crate::Rotation3d::IDENTITY,
        Vec3::new(1e-20, 1e20, 1.0).unwrap(),
    )
    .unwrap();
    scene.set_transform(id, extreme).unwrap();
    let native = Mesh3dRenderBudget::default().with_surface_policy(SurfaceRasterization3d::Native);
    let error = renderer
        .render_scene3d(device, queue, &identity, &target, &scene, camera(), native)
        .err()
        .unwrap();
    assert_eq!(error.source_vertex_index(), Some(0));
    assert_eq!(
        error.surface_reason(),
        Some(Mesh3dSurfaceError::NormalTransform)
    );
    assert_eq!(surface::test_read_pixels(device, queue, &target), before);
    scene
        .set_style(
            id,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
        )
        .unwrap();
    assert!(
        renderer
            .render_scene3d(device, queue, &identity, &target, &scene, camera(), native)
            .is_ok()
    );
}

pub(super) fn recovery_source(source: Mesh3d) -> Mesh3d {
    with_normals(source, Vec3::new(1.0, 0.0, 1.0).unwrap())
}
