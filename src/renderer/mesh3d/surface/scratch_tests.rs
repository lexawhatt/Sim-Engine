//! Active records and retained allocations have independent lifetimes.

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

fn capacities(frame: &SurfaceFrame) -> [usize; 6] {
    [
        frame.vertices.capacity(),
        frame.colors.capacity(),
        frame.lighting.capacity(),
        frame.edges.capacity(),
        frame.objects.capacity(),
        frame.order.capacity(),
    ]
}

fn assert_empty(frame: &SurfaceFrame) {
    assert!(frame.vertices.is_empty());
    assert!(frame.colors.is_empty());
    assert!(frame.lighting.is_empty());
    assert!(frame.edges.is_empty());
    assert!(frame.objects.is_empty());
    assert!(frame.order.is_empty());
}

fn populated_scratch() -> SurfaceFrame {
    let mut frame = SurfaceFrame::default();
    frame.vertices.push(SurfaceClipVertex {
        clip: [1.0; 4],
        uv: [0.5; 2],
    });
    frame.colors.push(MeshColorGpu { color: [1.0; 4] });
    frame.lighting.push(SurfaceLightingVertex::default());
    frame.edges.push(SurfaceClipEdge {
        start: [0.0; 4],
        end: [1.0; 4],
    });
    frame.objects.push(SurfaceObject {
        generated: Some(0..3),
        generated_colors: Some(0..3),
        generated_lighting: Some(0..3),
        transport: SurfaceTransport::default(),
        model_rows: [[0.0; 4]; 3],
        generated_edges: Some(0..1),
    });
    frame.order.reserve_exact(8);
    frame.report.generated_vertices = 3;
    frame.report.generated_upload_bytes = 80;
    frame.report.sorting_capacity_bytes = 123;
    frame
}

#[test]
fn scratch_reset_clears_every_active_record_without_losing_capacity() {
    let mut frame = populated_scratch();
    let before = capacities(&frame);
    let expected_bytes = before
        .into_iter()
        .zip([
            std::mem::size_of::<SurfaceClipVertex>(),
            std::mem::size_of::<MeshColorGpu>(),
            std::mem::size_of::<SurfaceLightingVertex>(),
            std::mem::size_of::<SurfaceClipEdge>(),
            std::mem::size_of::<SurfaceObject>(),
            std::mem::size_of::<material::SurfaceDraw>(),
        ])
        .map(|(capacity, stride)| capacity * stride)
        .sum::<usize>();
    assert_eq!(frame.retained_cpu_bytes(), expected_bytes);
    frame.reset();
    assert_empty(&frame);
    assert_eq!(frame.report, Mesh3dPreflightReport::default());
    assert_eq!(capacities(&frame), before);
    assert_eq!(frame.retained_cpu_bytes(), expected_bytes);
    frame.reset();
    assert_eq!(frame.retained_cpu_bytes(), expected_bytes);
}

#[test]
fn empty_preflight_clears_old_geometry_but_keeps_idle_capacity_out_of_active_budget() {
    let mut frame = populated_scratch();
    let before = capacities(&frame);
    let identity = Arc::new(());
    let scene = Scene3d::new(Color::BLACK).unwrap();
    let uniform = Camera3dUniform::new(camera(), 64, 64, physical_per_logical(1.0)).unwrap();
    for policy in [
        SurfaceRasterization3d::Native,
        SurfaceRasterization3d::StrictPortable,
    ] {
        preflight_into(
            &identity,
            &scene,
            uniform,
            Mesh3dRenderBudget::new(0, 0, 0)
                .with_surface_policy(policy)
                .with_max_sorting_bytes(0),
            0,
            &mut frame,
        )
        .unwrap();
        assert_empty(&frame);
        assert_eq!(frame.report.surface_policy(), policy);
        assert_eq!(frame.report.generated_upload_bytes(), 0);
        assert_eq!(frame.report.sorting_capacity_bytes(), 0);
        assert_eq!(capacities(&frame), before);
    }
}

fn assert_equivalent(left: &SurfaceFrame, right: &SurfaceFrame) {
    assert_eq!(left.report, right.report);
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&left.vertices),
        bytemuck::cast_slice::<_, u8>(&right.vertices)
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&left.colors),
        bytemuck::cast_slice::<_, u8>(&right.colors)
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&left.lighting),
        bytemuck::cast_slice::<_, u8>(&right.lighting)
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&left.edges),
        bytemuck::cast_slice::<_, u8>(&right.edges)
    );
    assert_eq!(left.objects.len(), right.objects.len());
    for (left, right) in left.objects.iter().zip(&right.objects) {
        assert_eq!(left.generated, right.generated);
        assert_eq!(left.generated_colors, right.generated_colors);
        assert_eq!(left.generated_lighting, right.generated_lighting);
        assert_eq!(left.generated_edges, right.generated_edges);
        assert_eq!(
            left.model_rows.map(|row| row.map(f32::to_bits)),
            right.model_rows.map(|row| row.map(f32::to_bits))
        );
    }
    let indices = |frame: &SurfaceFrame| {
        frame
            .order
            .iter()
            .map(|draw| (draw.scene_index, draw.visible_index))
            .collect::<Vec<_>>()
    };
    assert_eq!(indices(left), indices(right));
}

fn prepare(
    identity: &Arc<()>,
    scene: &Scene3d,
    budget: Mesh3dRenderBudget,
    max_buffer_size: u64,
    frame: &mut SurfaceFrame,
) -> Result<(), Mesh3dRenderError> {
    frame.reset();
    material::prepare_order_into(scene, camera(), budget, &mut frame.order)?;
    let mut uniform = Camera3dUniform::new(camera(), 64, 64, physical_per_logical(1.0)).unwrap();
    uniform.environment = SurfaceEnvironmentGpu::new(scene, camera());
    preflight_into(identity, scene, uniform, budget, max_buffer_size, frame)
}

pub(in crate::renderer::mesh3d) fn assert_gpu_preflight_scratch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let identity = Arc::new(());
    let source = Mesh3d::with_attributes(
        vec![
            Vec3::new(-0.5, -0.5, 0.0).unwrap(),
            Vec3::new(1.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.0, 0.5, 0.0).unwrap(),
        ],
        vec![0, 1, 2],
        vec![MeshEdge3d::new(0, 1).unwrap()],
        crate::Mesh3dAttributes::new()
            .with_vertex_colors(vec![
                Color::rgb(1.0, 0.0, 0.0),
                Color::rgb(0.0, 1.0, 0.0),
                Color::rgb(0.0, 0.0, 1.0),
            ])
            .unwrap()
            .with_normals(vec![Vec3::Z; 3])
            .unwrap(),
    )
    .unwrap();
    let mesh = create_retained_mesh(device, queue, Arc::clone(&identity), source).unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene.set_lighting(
        crate::Lighting3d::new(crate::AmbientLight3d::new(Color::WHITE, 0.25).unwrap())
            .with_directional(Some(
                crate::DirectionalLight3d::new(Vec3::Z, Color::WHITE, 0.75).unwrap(),
            )),
    );
    scene.set_fog(Some(crate::Fog3d::new(Color::BLACK, 0.5, 0.2).unwrap()));
    scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(
                SurfaceStyle3d::blend(Color::rgba(1.0, 1.0, 1.0, 0.5))
                    .unwrap()
                    .with_lighting(SurfaceLighting3d::Lambert)
                    .with_fog(true),
            )
            .with_wireframe(WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap()),
        )
        .unwrap();
    let max_buffer_size = device.limits().max_buffer_size;
    let budget = Mesh3dRenderBudget::default()
        .with_max_sorting_bytes(std::mem::size_of::<material::SurfaceDraw>());
    let mut reference = SurfaceFrame::default();
    prepare(&identity, &scene, budget, max_buffer_size, &mut reference).unwrap();
    assert!(reference.report.generated_vertex_count() > 0);
    assert!(!reference.colors.is_empty());
    assert!(!reference.lighting.is_empty());
    assert!(!reference.edges.is_empty());
    let mut reused = SurfaceFrame::default();
    reused.order.reserve_exact(32);
    // A generous active budget may keep larger storage than fresh validation.
    // Geometry proofs/counts agree; only the reported actual allocation differs.
    prepare(
        &identity,
        &scene,
        Mesh3dRenderBudget::default(),
        max_buffer_size,
        &mut reused,
    )
    .unwrap();
    assert!(reused.report.sorting_capacity_bytes() > reference.report.sorting_capacity_bytes());
    let actual_capacity = reused.report.sorting_capacity_bytes;
    reused.report.sorting_capacity_bytes = reference.report.sorting_capacity_bytes;
    assert_equivalent(&reference, &reused);
    reused.report.sorting_capacity_bytes = actual_capacity;
    prepare(&identity, &scene, budget, max_buffer_size, &mut reused).unwrap();
    assert_equivalent(&reference, &reused);
    let capacity = capacities(&reused);
    for _ in 0..3 {
        prepare(&identity, &scene, budget, max_buffer_size, &mut reused).unwrap();
        assert_equivalent(&reference, &reused);
        assert_eq!(capacities(&reused), capacity);
    }

    for policy in [
        SurfaceRasterization3d::Native,
        SurfaceRasterization3d::StrictPortable,
    ] {
        let budget = budget.with_surface_policy(policy);
        prepare(&identity, &scene, budget, max_buffer_size, &mut reused).unwrap();
        let mut fresh = SurfaceFrame::default();
        prepare(&identity, &scene, budget, max_buffer_size, &mut fresh).unwrap();
        assert_equivalent(&fresh, &reused);
        assert_eq!(capacities(&reused), capacity);
    }

    let invalid_budget = Mesh3dRenderBudget::new(0, 0, 0);
    let error = prepare(
        &identity,
        &scene,
        invalid_budget,
        max_buffer_size,
        &mut reused,
    )
    .unwrap_err();
    assert_eq!(error, Mesh3dRenderError::GeneratedGeometryCapacityTooLarge);
    assert_empty(&reused);
    assert_eq!(reused.report, Mesh3dPreflightReport::default());
    assert_eq!(capacities(&reused), capacity);
    let mut fresh = SurfaceFrame::default();
    assert_eq!(
        prepare(
            &identity,
            &scene,
            invalid_budget,
            max_buffer_size,
            &mut fresh
        ),
        Err(error)
    );
    assert_eq!(&capacities(&fresh)[..4], &[0; 4]);

    let invalid = scene
        .try_push(
            &mesh,
            Transform3d::new(
                Vec3::ZERO,
                Rotation3d::IDENTITY,
                Vec3::new(MAX_PORTABLE_SHADER_VALUE, 1.0, 1.0).unwrap(),
            )
            .unwrap(),
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
        )
        .unwrap();
    let mut failed = SurfaceFrame::default();
    let error = prepare(
        &identity,
        &scene,
        Mesh3dRenderBudget::default(),
        max_buffer_size,
        &mut failed,
    )
    .unwrap_err();
    assert_eq!(error.object_id(), Some(invalid));
    assert_eq!(error.source_triangle_index(), Some(0));
    assert_eq!(
        error.surface_reason(),
        Some(Mesh3dSurfaceError::TransformArithmetic)
    );
    assert_empty(&failed);
    assert_eq!(failed.report, Mesh3dPreflightReport::default());
    assert_eq!(&capacities(&failed)[..4], &[0; 4]);
    assert_eq!(
        prepare(
            &identity,
            &scene,
            Mesh3dRenderBudget::default(),
            max_buffer_size,
            &mut reused
        ),
        Err(error)
    );
    assert_empty(&reused);
    assert_eq!(reused.report, Mesh3dPreflightReport::default());
    scene.remove(invalid).unwrap();
    prepare(&identity, &scene, budget, max_buffer_size, &mut reused).unwrap();
    assert_equivalent(&reference, &reused);
}
