use super::*;

fn point(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2]).unwrap()
}

fn camera() -> Camera3d {
    Camera3d::look_at(
        point([0.0, 0.0, 2.0]),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(1.0), world(3.0)).unwrap(),
    )
    .unwrap()
}

fn edge_mesh() -> Mesh3d {
    Mesh3d::with_display_edges(
        vec![point([-0.5, 0.0, 0.0]), point([0.5, 0.0, 0.0])],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap()
}

fn unsafe_dash_style() -> WireframeStyle3d {
    let smallest = logical(f32::MIN_POSITIVE);
    WireframeStyle3d::visible(Color::WHITE, logical(1.0))
        .unwrap()
        .with_hidden(Color::WHITE, logical(1.0), smallest, smallest)
        .unwrap()
}

fn ambiguous_vertex_fixture() -> (Mesh3d, Transform3d, Camera3d) {
    let position = point([-6.593_279e17, 2.460_645_7e18, 9.006_587e17]);
    let mesh = Mesh3d::with_display_edges(
        vec![position, position.checked_scale(0.5).unwrap()],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let axis = point([1.0, 1.0, 1.0]).normalized().unwrap();
    let transform = Transform3d::new(
        Vec3::ZERO,
        Rotation3d::from_axis_angle(axis, std::f32::consts::FRAC_PI_2).unwrap(),
        point([1.0, 1.0, 1.0]),
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::Z,
        Vec3::Y,
        Projection3d::orthographic(world(100_000.0), 1.0, world(1.0), world(3.0e18)).unwrap(),
    )
    .unwrap();
    (mesh, transform, camera)
}

#[test]
fn native_surface_policy_distinguishes_vertex_classification_from_arithmetic() {
    let (mesh, transform, camera) = ambiguous_vertex_fixture();
    let model = transform.model_rows().unwrap();
    let rows = camera.world_to_clip_rows().unwrap();
    assert_eq!(
        validation::validate_point_for_policy(
            mesh.vertices()[0],
            model,
            rows,
            SurfaceRasterization3d::StrictPortable,
        ),
        Err(Mesh3dSurfaceError::ClipPlaneClassification),
    );
    for vertex in mesh.vertices() {
        validation::validate_point_for_policy(*vertex, model, rows, SurfaceRasterization3d::Native)
            .unwrap();
    }
}

#[test]
fn native_surface_policy_does_not_make_unsafe_edge_dash_arithmetic_valid() {
    let mesh = edge_mesh();
    let model = Transform3d::IDENTITY.model_rows().unwrap();
    let rows = camera().world_to_clip_rows().unwrap();
    for vertex in mesh.vertices() {
        validation::validate_point_for_policy(*vertex, model, rows, SurfaceRasterization3d::Native)
            .unwrap();
    }
    assert_eq!(
        validate_edge_projection(
            &mesh,
            model,
            rows,
            unsafe_dash_style(),
            [64.0, 64.0, 1.0, 0.0]
        ),
        Err(Mesh3dRenderError::InvalidEdgeProjection),
    );
}

pub(in crate::renderer::mesh3d) fn assert_gpu_native_edge_validation(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = test_target(device, &identity, format, 64, 64);
    let camera = camera();
    let native = Mesh3dRenderBudget::default().with_surface_policy(SurfaceRasterization3d::Native);
    let background = Scene3d::new(Color::rgb(0.25, 0.0, 0.25)).unwrap();
    renderer
        .render_scene3d(
            device,
            queue,
            &identity,
            &target,
            &background,
            camera,
            native,
        )
        .unwrap();
    let before = test_read_pixels(device, queue, &target);
    let mesh = create_retained_mesh(device, queue, Arc::clone(&identity), edge_mesh()).unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let id = scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::wireframe(unsafe_dash_style()),
        )
        .unwrap();
    for policy in [
        SurfaceRasterization3d::StrictPortable,
        SurfaceRasterization3d::Native,
    ] {
        let error = renderer
            .render_scene3d(
                device,
                queue,
                &identity,
                &target,
                &scene,
                camera,
                native.with_surface_policy(policy),
            )
            .unwrap_err();
        assert_eq!(
            error,
            Mesh3dRenderError::ObjectFailure {
                object_id: id,
                reason: Mesh3dObjectError::InvalidEdgeProjection,
            },
        );
        assert_eq!(error.object_id(), Some(id));
        assert_eq!(error.source_triangle_index(), None);
        assert_eq!(error.source_vertex_index(), None);
        assert_eq!(test_read_pixels(device, queue, &target), before);
    }

    // A bad unreferenced vertex must not lose its source index merely because
    // it has no triangle owner. The first two safe vertices still form an edge.
    let source = Mesh3d::with_display_edges(
        vec![
            point([-0.5, 0.0, 0.0]),
            point([-0.25, 0.25, 0.0]),
            point([3.0, 0.0, 0.0]),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let mesh = create_retained_mesh(device, queue, Arc::clone(&identity), source).unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let maximum = 2.0_f32.powi(119);
    let id = scene
        .try_push(
            &mesh,
            Transform3d::new(Vec3::ZERO, Rotation3d::IDENTITY, point([maximum, 1.0, 1.0])).unwrap(),
            MeshStyle3d::wireframe(WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap()),
        )
        .unwrap();
    for policy in [
        SurfaceRasterization3d::StrictPortable,
        SurfaceRasterization3d::Native,
    ] {
        let error = renderer
            .render_scene3d(
                device,
                queue,
                &identity,
                &target,
                &scene,
                camera,
                native.with_surface_policy(policy),
            )
            .unwrap_err();
        assert_eq!(
            error,
            Mesh3dRenderError::ObjectFailure {
                object_id: id,
                reason: Mesh3dObjectError::Vertex {
                    vertex_index: 2,
                    reason: Mesh3dSurfaceError::TransformArithmetic,
                },
            },
        );
        assert_eq!(error.source_vertex_index(), Some(2));
        assert_eq!(error.source_triangle_index(), None);
        assert_eq!(
            error.surface_reason(),
            Some(Mesh3dSurfaceError::TransformArithmetic)
        );
        assert_eq!(test_read_pixels(device, queue, &target), before);
    }
    let (source, transform, ambiguous_camera) = ambiguous_vertex_fixture();
    let mesh = create_retained_mesh(device, queue, Arc::clone(&identity), source).unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let id = scene
        .try_push(
            &mesh,
            transform,
            MeshStyle3d::wireframe(WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap()),
        )
        .unwrap();
    let error = renderer
        .render_scene3d(
            device,
            queue,
            &identity,
            &target,
            &scene,
            ambiguous_camera,
            Mesh3dRenderBudget::default(),
        )
        .unwrap_err();
    assert_eq!(error.object_id(), Some(id));
    assert_eq!(error.source_triangle_index(), None);
    assert_eq!(error.source_vertex_index(), Some(0));
    assert_eq!(
        error.surface_reason(),
        Some(Mesh3dSurfaceError::ClipPlaneClassification)
    );
    assert_eq!(test_read_pixels(device, queue, &target), before);
    // Native acceptance of the same surface arithmetic must not bypass this
    // edge's independent, ambiguous endpoint classification.
    let error = renderer
        .render_scene3d(
            device,
            queue,
            &identity,
            &target,
            &scene,
            ambiguous_camera,
            native,
        )
        .unwrap_err();
    assert_eq!(error.object_id(), Some(id));
    assert_eq!(test_read_pixels(device, queue, &target), before);
    assert_native_plane_pixels(device, queue, format);
}

fn plane_sources(plane: usize) -> (Mesh3d, Mesh3d) {
    let (source, reference) = if plane < 4 {
        let transform = |[x, y, z]: [f32; 3]| match plane {
            0 => [x, y, z],
            1 => [-x, y, z],
            2 => [y, x, z],
            _ => [y, -x, z],
        };
        (
            [[-2.0, -0.5, 0.0], [0.0, -0.5, 0.0], [0.0, 0.5, 0.0]].map(transform),
            [
                [-1.0, -0.5, 0.0],
                [0.0, -0.5, 0.0],
                [0.0, 0.5, 0.0],
                [-1.0, 0.0, 0.0],
            ]
            .map(transform),
        )
    } else {
        let direction = if plane == 4 { 1.0 } else { -1.0 };
        (
            [
                [-0.5, -0.5, direction * 2.0],
                [0.5, -0.5, 0.0],
                [0.0, 0.5, 0.0],
            ],
            [
                [0.0, -0.5, direction],
                [0.5, -0.5, 0.0],
                [0.0, 0.5, 0.0],
                [-0.25, 0.0, direction],
            ],
        )
    };
    (
        Mesh3d::new(source.into_iter().map(point).collect(), vec![0, 1, 2]).unwrap(),
        Mesh3d::new(
            reference.into_iter().map(point).collect(),
            vec![0, 1, 2, 0, 2, 3],
        )
        .unwrap(),
    )
}

#[test]
fn native_surface_policy_accepts_all_six_plane_source_arithmetic() {
    let model = Transform3d::IDENTITY.model_rows().unwrap();
    let rows = camera().world_to_clip_rows().unwrap();
    for plane in 0..6 {
        let (source, reference) = plane_sources(plane);
        for mesh in [source, reference] {
            for vertex in mesh.vertices() {
                validation::validate_point_for_policy(
                    *vertex,
                    model,
                    rows,
                    SurfaceRasterization3d::Native,
                )
                .unwrap();
            }
        }
    }
}

fn native_pixels(
    renderer: &mut Mesh3dRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &RenderTarget3d,
    identity: &Arc<()>,
    source: Mesh3d,
) -> Vec<u8> {
    let expected_triangles = source.triangle_count();
    let mesh = create_retained_mesh(device, queue, Arc::clone(identity), source).unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
        )
        .unwrap();
    let report = renderer
        .render_scene3d(
            device,
            queue,
            identity,
            target,
            &scene,
            camera(),
            Mesh3dRenderBudget::new(0, 0, 0)
                .with_max_surface_triangles(expected_triangles)
                .with_surface_policy(SurfaceRasterization3d::Native),
        )
        .unwrap();
    assert_eq!(report.triangle_count(), expected_triangles);
    assert_eq!(report.preflight().generated_triangle_count(), 0);
    test_read_pixels(device, queue, target)
}

fn assert_native_plane_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = test_target(device, &identity, format, 64, 64);
    for plane in 0..6 {
        let (source, reference) = plane_sources(plane);
        let source_pixels = native_pixels(&mut renderer, device, queue, &target, &identity, source);
        let reference_pixels =
            native_pixels(&mut renderer, device, queue, &target, &identity, reference);
        assert!(
            source_pixels.chunks_exact(4).any(|pixel| pixel[0] == 255),
            "plane {plane}"
        );
        assert_eq!(
            source_pixels, reference_pixels,
            "native hardware clipping at plane {plane}"
        );
    }
    let edge_on = Mesh3d::new(
        vec![
            point([0.0, -0.5, -0.5]),
            point([0.0, 0.5, -0.5]),
            point([0.0, 0.0, 0.5]),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let pixels = native_pixels(&mut renderer, device, queue, &target, &identity, edge_on);
    assert!(pixels.chunks_exact(4).all(|pixel| pixel[..3] == [0, 0, 0]));

    // Half a physical pixel wide, centered over an actual sample column. A
    // nonzero sliver must not be treated as the exactly edge-on control above.
    let sliver = Mesh3d::new(
        vec![
            point([1.0 / 128.0, -0.5, 0.0]),
            point([3.0 / 128.0, -0.5, 0.0]),
            point([1.0 / 64.0, 0.5, 0.0]),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let pixels = native_pixels(&mut renderer, device, queue, &target, &identity, sliver);
    assert!(
        pixels
            .chunks_exact(4)
            .filter(|pixel| pixel[0] == 255)
            .count()
            >= 16
    );
}
