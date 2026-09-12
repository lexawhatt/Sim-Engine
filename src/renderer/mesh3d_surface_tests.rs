use super::*;

pub(in crate::renderer::mesh3d) fn target(
    device: &wgpu::Device,
    identity: &Arc<()>,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> RenderTarget3d {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface clipping oracle"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = create_depth_texture(device, width, height);
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    RenderTarget3d {
        renderer_identity: Arc::clone(identity),
        color: RenderTarget2d {
            renderer_identity: Arc::clone(identity),
            resource_identity: Arc::new(()),
            _texture: texture,
            view,
            width,
            height,
            allocation_bytes: width as usize * height as usize * 4,
        },
        _depth_texture: depth,
        depth_view,
        logical_viewport: LogicalViewport::new(width as f32, height as f32).unwrap(),
        pixels_per_logical: physical_per_logical(1.0),
    }
}

pub(in crate::renderer::mesh3d) fn read_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &RenderTarget3d,
) -> Vec<u8> {
    let stride = (target.width() * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("surface clipping pixel readback"),
        size: u64::from(stride * target.height()),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target.color._texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(target.height()),
            },
        },
        wgpu::Extent3d {
            width: target.width(),
            height: target.height(),
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let mut result = Vec::new();
    for row in mapped.chunks_exact(stride as usize) {
        result.extend_from_slice(&row[..target.width() as usize * 4]);
    }
    result
}

fn cube(reverse: bool) -> Mesh3d {
    let vertices = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
    .into_iter()
    .map(|[x, y, z]| Vec3::new(x, y, z).unwrap())
    .collect();
    let mut indices = vec![
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    ];
    if reverse {
        for triangle in indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
    }
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    .into_iter()
    .map(|(a, b)| MeshEdge3d::new(a, b).unwrap())
    .collect();
    Mesh3d::with_display_edges(vertices, indices, edges).unwrap()
}

pub(in crate::renderer::mesh3d) fn assert_gpu_surface_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let rotation = Rotation3d::from_euler_xyz(0.17, 0.29, 0.11).unwrap();
    for (width, height) in [(64, 64), (96, 64)] {
        let target = target(device, &identity, format, width, height);
        let aspect = width as f32 / height as f32;
        for perspective in [false, true] {
            let projection = if perspective {
                Projection3d::perspective(
                    std::f32::consts::FRAC_PI_2,
                    aspect,
                    world(0.5),
                    world(4.0),
                )
                .unwrap()
            } else {
                Projection3d::orthographic(world(2.0), aspect, world(0.5), world(4.0)).unwrap()
            };
            let camera = Camera3d::look_at(
                Vec3::ZERO,
                Vec3::new(0.0, 0.0, -1.0).unwrap(),
                Vec3::Y,
                projection,
            )
            .unwrap();
            let side = if perspective { 2.0 } else { 1.0 };
            for reverse in [false, true] {
                let retained =
                    create_retained_mesh(device, queue, Arc::clone(&identity), cube(reverse))
                        .unwrap();
                for (plane, center) in [
                    [-side * aspect, 0.0, -2.0],
                    [side * aspect, 0.0, -2.0],
                    [0.0, -side, -2.0],
                    [0.0, side, -2.0],
                    [0.07, 0.09, -0.55],
                    [0.07, 0.09, -4.0],
                    [10.0, 10.0, -2.0],
                ]
                .into_iter()
                .enumerate()
                {
                    let mut scene = Scene3d::new(Color::BLACK).unwrap();
                    let transform = Transform3d::new(
                        Vec3::new(center[0], center[1], center[2]).unwrap(),
                        rotation,
                        Vec3::new(0.3, 0.3, 0.3).unwrap(),
                    )
                    .unwrap();
                    let style = MeshStyle3d::surface(
                        SurfaceStyle3d::opaque(Color::rgb(0.0, 1.0, 0.0)).unwrap(),
                    )
                    .with_wireframe(
                        WireframeStyle3d::visible(Color::WHITE, logical(1.0))
                            .unwrap()
                            .with_hidden(
                                Color::rgb(1.0, 0.0, 0.0),
                                logical(1.0),
                                logical(3.0),
                                logical(3.0),
                            )
                            .unwrap(),
                    );
                    let object = scene.try_push(&retained, transform, style).unwrap();
                    let report = renderer.render_scene3d(device, queue, &identity, &target, &scene, camera, Mesh3dRenderBudget::default())
                        .unwrap_or_else(|error| panic!("plane={plane} perspective={perspective} reversed={reverse} aspect={aspect}: {error}"));
                    let clipped = read_pixels(device, queue, &target);
                    if plane < 6 {
                        assert!(report.preflight.clipped_source_triangles > 0);
                        assert!(clipped.chunks_exact(4).any(|pixel| pixel[1] > 150));
                        let counts = report.preflight;
                        let exact_budget = Mesh3dRenderBudget::new(
                            counts.generated_vertices,
                            counts.generated_triangles,
                            counts.generated_upload_bytes,
                        );
                        let checked = renderer
                            .preflight_scene3d(
                                device,
                                &identity,
                                &target,
                                &scene,
                                camera,
                                exact_budget,
                            )
                            .unwrap()
                            .1
                            .report;
                        assert_eq!(checked, counts);
                        for budget in [
                            Mesh3dRenderBudget::new(
                                counts.generated_vertices - 1,
                                usize::MAX,
                                usize::MAX,
                            ),
                            Mesh3dRenderBudget::new(
                                usize::MAX,
                                counts.generated_triangles - 1,
                                usize::MAX,
                            ),
                            Mesh3dRenderBudget::new(
                                usize::MAX,
                                usize::MAX,
                                counts.generated_upload_bytes - 1,
                            ),
                        ] {
                            assert_eq!(
                                renderer.render_scene3d(
                                    device, queue, &identity, &target, &scene, camera, budget
                                ),
                                Err(Mesh3dRenderError::GeneratedGeometryCapacityTooLarge)
                            );
                        }
                        assert_eq!(
                            read_pixels(device, queue, &target),
                            clipped,
                            "rejected frame modified target"
                        );
                    } else {
                        assert_eq!(report.preflight.discarded_source_triangles, 12);
                        assert!(clipped.chunks_exact(4).all(|pixel| pixel[..3] == [0, 0, 0]));
                    }
                    // Compare against hardware clipping of the original retained
                    // topology, including visible and dashed hidden edge depth.
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
                    let hardware = read_pixels(device, queue, &target);
                    let mismatched = clipped
                        .chunks_exact(4)
                        .zip(hardware.chunks_exact(4))
                        .filter(|(left, right)| {
                            (0..3).any(|channel| left[channel].abs_diff(right[channel]) > 16)
                        })
                        .count();
                    assert!(
                        mismatched <= 8,
                        "surface/depth/edge disagreement: {mismatched} pixels plane={plane} perspective={perspective} reversed={reverse} aspect={aspect}"
                    );
                    scene.set_visible(object, false).unwrap();
                    assert_eq!(
                        renderer
                            .preflight_scene3d(
                                device,
                                &identity,
                                &target,
                                &scene,
                                camera,
                                Mesh3dRenderBudget::new(0, 0, 0)
                            )
                            .unwrap()
                            .1
                            .report,
                        Mesh3dPreflightReport::default()
                    );
                }
            }
        }
    }
    assert_attribution(device, queue, format, &identity, &mut renderer);
    assert_total_surface_budget(device, queue, format, &identity);
}

fn assert_total_surface_budget(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    identity: &Arc<()>,
) {
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = target(device, identity, format, 64, 64);
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(4.0)).unwrap(),
    )
    .unwrap();
    let surface = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap());
    let triangle = |positions: [[f32; 2]; 3]| {
        create_retained_mesh(
            device,
            queue,
            Arc::clone(identity),
            Mesh3d::new(
                positions
                    .into_iter()
                    .map(|[x, y]| Vec3::new(x, y, 0.0).unwrap())
                    .collect(),
                vec![0, 1, 2],
            )
            .unwrap(),
        )
        .unwrap()
    };
    let inside = triangle([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]]);
    let crossing = triangle([[-2.0, -0.5], [0.5, -0.5], [0.5, 0.5]]);
    let outside = triangle([[3.0, -0.5], [4.0, -0.5], [3.0, 0.5]]);
    let fully_clipped = triangle([[1.5, 0.75], [0.75, 1.5], [1.5, 1.5]]);
    let edge = create_retained_mesh(
        device,
        queue,
        Arc::clone(identity),
        Mesh3d::with_display_edges(
            vec![
                Vec3::new(-0.5, 0.0, 0.0).unwrap(),
                Vec3::new(0.5, 0.0, 0.0).unwrap(),
            ],
            Vec::new(),
            vec![MeshEdge3d::new(0, 1).unwrap()],
        )
        .unwrap(),
    )
    .unwrap();
    let mut empty = Scene3d::new(Color::rgb(0.25, 0.0, 0.25)).unwrap();
    renderer
        .render_scene3d(
            device,
            queue,
            identity,
            &target,
            &empty,
            camera,
            Mesh3dRenderBudget::default().with_max_surface_triangles(0),
        )
        .unwrap();
    let initial_pixels = read_pixels(device, queue, &target);
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    for mesh in [&inside, &crossing, &outside] {
        scene
            .try_push(mesh, Transform3d::IDENTITY, surface)
            .unwrap();
    }
    let hidden = scene
        .try_push(&crossing, Transform3d::IDENTITY, surface)
        .unwrap();
    scene.set_visible(hidden, false).unwrap();
    scene
        .try_push(
            &edge,
            Transform3d::IDENTITY,
            MeshStyle3d::wireframe(WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap()),
        )
        .unwrap();
    let exact = Mesh3dRenderBudget::new(6, 2, 6 * std::mem::size_of::<SurfaceClipVertex>())
        .with_max_surface_triangles(4);
    let checked = renderer
        .preflight_scene3d(device, identity, &target, &scene, camera, exact)
        .unwrap()
        .1
        .report;
    assert_eq!(checked.submitted_triangle_count(), 4);
    assert_eq!(checked.generated_object_count(), 1);
    assert_eq!(checked.generated_triangle_count(), 2);
    assert_eq!(checked.clipped_source_triangle_count(), Some(1));
    assert_eq!(checked.discarded_source_triangle_count(), Some(1));
    assert_eq!(
        renderer.render_scene3d(
            device,
            queue,
            identity,
            &target,
            &scene,
            camera,
            exact.with_max_surface_triangles(3),
        ),
        Err(Mesh3dRenderError::SurfaceTriangleBudgetExceeded {
            limit: 3,
            actual: 4
        }),
    );
    assert!(renderer.clipped_surface_buffer.is_none());
    assert_eq!(renderer.clipped_surface_capacity, 0);
    assert_eq!(read_pixels(device, queue, &target), initial_pixels);
    let rendered = renderer
        .render_scene3d(device, queue, identity, &target, &scene, camera, exact)
        .unwrap();
    assert_eq!(
        rendered.triangle_count(),
        checked.submitted_triangle_count()
    );
    assert_eq!(rendered.preflight(), checked);

    for (mesh, submitted, generated_objects) in [(&outside, 1, 0), (&fully_clipped, 0, 1)] {
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        scene
            .try_push(mesh, Transform3d::IDENTITY, surface)
            .unwrap();
        let checked = renderer
            .preflight_scene3d(
                device,
                identity,
                &target,
                &scene,
                camera,
                Mesh3dRenderBudget::new(0, 0, 0).with_max_surface_triangles(submitted),
            )
            .unwrap()
            .1
            .report;
        assert_eq!(checked.submitted_triangle_count(), submitted);
        assert_eq!(checked.generated_object_count(), generated_objects);
        assert_eq!(checked.generated_triangle_count(), 0);
        assert_eq!(checked.discarded_source_triangle_count(), Some(1));
    }
    empty
        .try_push(
            &edge,
            Transform3d::IDENTITY,
            MeshStyle3d::wireframe(WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap()),
        )
        .unwrap();
    assert_eq!(
        renderer
            .preflight_scene3d(
                device,
                identity,
                &target,
                &empty,
                camera,
                Mesh3dRenderBudget::new(0, 0, 0).with_max_surface_triangles(0),
            )
            .unwrap()
            .1
            .report
            .submitted_triangle_count(),
        0,
    );
}

#[test]
fn total_surface_budget_fixture_has_exact_strict_classification() {
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(4.0)).unwrap(),
    )
    .unwrap();
    let camera = Camera3dUniform::new(camera, 64, 64, physical_per_logical(1.0)).unwrap();
    let model = Transform3d::IDENTITY.model_rows().unwrap();
    for (positions, expected_vertices, expected_crossing) in [
        ([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]], 3, false),
        ([[-2.0, -0.5], [0.5, -0.5], [0.5, 0.5]], 6, true),
        ([[3.0, -0.5], [4.0, -0.5], [3.0, 0.5]], 0, false),
        ([[1.5, 0.75], [0.75, 1.5], [1.5, 1.5]], 0, true),
    ] {
        let mesh = Mesh3d::new(
            positions
                .into_iter()
                .map(|[x, y]| Vec3::new(x, y, 0.0).unwrap())
                .collect(),
            vec![0, 1, 2],
        )
        .unwrap();
        validate_shader_points(&mesh, model, camera.rows()).unwrap();
        let mut visited = 0;
        classify_surface(&mesh, model, camera.rows(), |index, vertices, crossing| {
            assert_eq!(index, 0);
            assert_eq!(vertices.len(), expected_vertices, "{positions:?}");
            assert_eq!(crossing, expected_crossing, "{positions:?}");
            visited += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(visited, 1);
    }
}

#[test]
fn total_surface_budget_preserves_generated_constructor_and_strict_default() {
    let budget = Mesh3dRenderBudget::new(12, 4, 288);
    assert_eq!(budget.max_surface_triangles(), usize::MAX);
    assert_eq!(
        budget.surface_policy(),
        SurfaceRasterization3d::StrictPortable
    );
    let bounded = budget.with_max_surface_triangles(8);
    assert_eq!(bounded.max_surface_triangles(), 8);
    assert_eq!(bounded.max_generated_vertices(), 12);
    assert_eq!(bounded.max_generated_triangles(), 4);
    assert_eq!(bounded.max_generated_upload_bytes(), 288);
}

#[test]
fn total_surface_budget_reports_exact_limits_without_partial_accounting() {
    let mut report = Mesh3dPreflightReport::default();
    let budget = Mesh3dRenderBudget::default().with_max_surface_triangles(4);
    record_surface_submission(&mut report, 1, false, budget).unwrap();
    record_surface_submission(&mut report, 2, true, budget).unwrap();
    record_surface_submission(&mut report, 1, false, budget).unwrap();
    assert_eq!(report.submitted_triangle_count(), 4);
    assert_eq!(report.generated_object_count(), 1);
    let before = report;
    assert_eq!(
        record_surface_submission(&mut report, 1, true, budget),
        Err(Mesh3dRenderError::SurfaceTriangleBudgetExceeded {
            limit: 4,
            actual: 5
        }),
    );
    assert_eq!(report, before);
    record_surface_submission(&mut report, 0, true, budget).unwrap();
    assert_eq!(report.submitted_triangle_count(), 4);
    assert_eq!(report.generated_object_count(), 2);
}

#[test]
fn total_surface_accounting_rejects_integer_overflow() {
    let mut report = Mesh3dPreflightReport {
        submitted_triangles: usize::MAX,
        ..Mesh3dPreflightReport::default()
    };
    let before = report;
    assert_eq!(
        record_surface_submission(&mut report, 1, false, Mesh3dRenderBudget::default()),
        Err(Mesh3dRenderError::GeneratedGeometryCapacityTooLarge),
    );
    assert_eq!(report, before);
    report.generated_objects = usize::MAX;
    let before = report;
    assert_eq!(
        record_surface_submission(&mut report, 0, true, Mesh3dRenderBudget::default()),
        Err(Mesh3dRenderError::GeneratedGeometryCapacityTooLarge),
    );
    assert_eq!(report, before);
}

fn assert_attribution(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    identity: &Arc<()>,
    renderer: &mut Mesh3dRenderer,
) {
    let target = target(device, identity, format, 64, 64);
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 5.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(4.0), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();
    let mesh = Mesh3d::new(
        vec![
            Vec3::new(-0.4, -0.3, 0.0).unwrap(),
            Vec3::new(0.5, -0.2, 0.0).unwrap(),
            Vec3::new(0.1, 0.5, 0.0).unwrap(),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let retained = create_retained_mesh(device, queue, Arc::clone(identity), mesh).unwrap();
    let invalid = Transform3d::new(
        Vec3::new(2.0_f32.powi(120), 0.0, 0.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(2.0_f32.powi(120), 1.0, 1.0).unwrap(),
    )
    .unwrap();
    for invalid_index in 0..3 {
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        let mut invalid_id = None;
        for index in 0..3 {
            let id = scene
                .try_push(
                    &retained,
                    if index == invalid_index {
                        invalid
                    } else {
                        Transform3d::IDENTITY
                    },
                    MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
                )
                .unwrap();
            if index == invalid_index {
                invalid_id = Some(id);
            }
        }
        let id = invalid_id.unwrap();
        let error = renderer
            .preflight_scene3d(
                device,
                identity,
                &target,
                &scene,
                camera,
                Mesh3dRenderBudget::default(),
            )
            .err()
            .unwrap();
        assert_eq!(error.object_id(), Some(id));
        assert!(matches!(
            error,
            Mesh3dRenderError::ObjectFailure {
                reason: Mesh3dObjectError::SurfaceTriangle {
                    triangle_index: 0,
                    reason: Mesh3dSurfaceError::TransformArithmetic,
                },
                ..
            }
        ));
        assert_eq!(
            renderer.render_scene3d(
                device,
                queue,
                identity,
                &target,
                &scene,
                camera,
                Mesh3dRenderBudget::default()
            ),
            Err(error)
        );
        scene.set_visible(id, false).unwrap();
        renderer
            .render_scene3d(
                device,
                queue,
                identity,
                &target,
                &scene,
                camera,
                Mesh3dRenderBudget::default(),
            )
            .unwrap();
    }
}

pub(in crate::renderer::mesh3d) fn assert_gpu_clipped_recovery(
    source_device: &wgpu::Device,
    source_queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
) {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let source_identity = Arc::new(());
    let recovery_identity = Arc::new(());
    let source_target = target(source_device, &source_identity, format, 64, 64);
    let recovery_target = target(recovery_device, &recovery_identity, format, 64, 64);
    let mut source_renderer = Mesh3dRenderer::new(source_device, format);
    let mut recovery_renderer = Mesh3dRenderer::new(recovery_device, format);
    let mesh = create_retained_mesh(
        source_device,
        source_queue,
        Arc::clone(&source_identity),
        cube(false),
    )
    .unwrap();
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, -1.0).unwrap(),
        Vec3::Y,
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(0.5), world(4.0))
            .unwrap(),
    )
    .unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    for (index, (x, color)) in [
        (-1.8, Color::rgb(0.0, 1.0, 0.0)),
        (1.8, Color::rgb(0.0, 0.0, 1.0)),
    ]
    .into_iter()
    .enumerate()
    {
        let transform = Transform3d::new(
            Vec3::new(x, 0.0, -2.0).unwrap(),
            Rotation3d::from_euler_xyz(0.17, 0.29 + index as f32 * 0.1, 0.11).unwrap(),
            Vec3::new(0.4, 0.4, 0.4).unwrap(),
        )
        .unwrap();
        scene
            .try_push(
                &mesh,
                transform,
                MeshStyle3d::surface(SurfaceStyle3d::opaque(color).unwrap())
                    .with_wireframe(WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap()),
            )
            .unwrap();
    }
    let identities: Vec<_> = scene
        .instances()
        .iter()
        .map(|instance| (instance.id(), instance.transform(), instance.style()))
        .collect();
    let before = source_renderer
        .render_scene3d(
            source_device,
            source_queue,
            &source_identity,
            &source_target,
            &scene,
            camera,
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    let pixels = read_pixels(source_device, source_queue, &source_target);
    assert!(before.preflight.generated_triangles > 0);
    assert!(before.preflight.generated_edges > 0);
    assert_eq!(
        recovery_renderer.render_scene3d(
            recovery_device,
            recovery_queue,
            &recovery_identity,
            &recovery_target,
            &scene,
            camera,
            Mesh3dRenderBudget::default()
        ),
        Err(Mesh3dRenderError::RendererMismatch)
    );
    let restore = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        &recovery_renderer.textures.layout,
        Arc::clone(&recovery_identity),
        &mut scene,
    )
    .unwrap();
    assert_eq!(restore.restored_mesh_count(), 1);
    assert_eq!(
        scene
            .instances()
            .iter()
            .map(|instance| (instance.id(), instance.transform(), instance.style()))
            .collect::<Vec<_>>(),
        identities
    );
    let after = recovery_renderer
        .render_scene3d(
            recovery_device,
            recovery_queue,
            &recovery_identity,
            &recovery_target,
            &scene,
            camera,
            Mesh3dRenderBudget::default(),
        )
        .unwrap();
    assert_eq!(after.preflight, before.preflight);
    assert_eq!(
        read_pixels(recovery_device, recovery_queue, &recovery_target),
        pixels
    );
}
