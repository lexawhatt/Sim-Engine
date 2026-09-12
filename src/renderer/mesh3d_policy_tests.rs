use super::*;

fn point(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2]).unwrap()
}

fn logic_camera(eye_z: f32) -> Camera3d {
    logic_camera_at(eye_z, 0.0)
}

fn logic_camera_at(eye_z: f32, yaw_offset: f32) -> Camera3d {
    let eye = Vec3::new(11.75, 7.58, eye_z).unwrap();
    let yaw = 14_f32 * 0.13 + yaw_offset;
    Camera3d::look_at(
        eye,
        Vec3::new(eye.x() + yaw.sin(), 7.18, eye.z() - yaw.cos()).unwrap(),
        Vec3::Y,
        Projection3d::perspective(
            std::f32::consts::FRAC_PI_3,
            1100.0 / 720.0,
            world(0.1),
            world(1000.0),
        )
        .unwrap(),
    )
    .unwrap()
}

fn logic_floor(reverse: bool) -> Mesh3d {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for z in 16..24 {
        for x in 8..16 {
            let start = vertices.len() as u32;
            vertices.extend([
                point([x as f32, 0.0, z as f32]),
                point([(x + 1) as f32, 0.0, z as f32]),
                point([(x + 1) as f32, 0.0, (z + 1) as f32]),
                point([x as f32, 0.0, (z + 1) as f32]),
            ]);
            indices.extend([start, start + 1, start + 2, start, start + 2, start + 3]);
        }
    }
    if reverse {
        for triangle in indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
    }
    Mesh3d::with_display_edges(vertices, indices, Vec::new()).unwrap()
}

#[test]
fn logic_free_camera_reproduces_source_triangle_orientation_failure() {
    for reverse in [false, true] {
        let mesh = logic_floor(reverse);
        assert_eq!(mesh.triangle_count(), 128);
        for eye_z in [11.0, 11.001] {
            let rows = logic_camera(eye_z).world_to_clip_rows().unwrap();
            let model = Transform3d::IDENTITY.model_rows().unwrap();
            validate_shader_points(&mesh, model, rows).unwrap();
            let failure = std::cell::Cell::new(None);
            let result = classify_surface_impl(
                &mesh,
                model,
                rows,
                |index, reason| {
                    failure.set(Some((index, reason)));
                    reason.legacy()
                },
                |_, _, _, _| Ok(()),
            );
            if reverse && eye_z == 11.001 {
                // Reversing a clipped polygon also changes fan arithmetic:
                // this particular nudged control has a provable orientation.
                assert_eq!(result, Ok(()));
                assert_eq!(failure.get(), None);
            } else {
                assert_eq!(result, Err(Mesh3dRenderError::UnportableSurfaceTopology));
                assert_eq!(
                    failure.get(),
                    Some((63, Mesh3dSurfaceError::ProjectedOrientation))
                );
            }
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

#[test]
fn native_surface_policy_checks_arithmetic_without_perspective_divide() {
    let model = Transform3d::IDENTITY.model_rows().unwrap();
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        point([0.0, 0.0, -1.0]),
        Vec3::Y,
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(0.25), world(8.0))
            .unwrap(),
    )
    .unwrap();
    let rows = camera.world_to_clip_rows().unwrap();
    for vertex in [Vec3::ZERO, point([0.0, 0.0, 1.0]), point([1.0, 0.0, -1.0])] {
        validation::validate_point_for_policy(vertex, model, rows, SurfaceRasterization3d::Native)
            .unwrap();
    }
    let mut overflowing = model;
    overflowing[0][0] = f32::MAX;
    assert_eq!(
        validation::validate_point_for_policy(
            point([2.0, 0.0, 0.0]),
            overflowing,
            rows,
            SurfaceRasterization3d::Native
        ),
        Err(Mesh3dSurfaceError::TransformArithmetic),
    );
}

pub(in crate::renderer::mesh3d) fn assert_gpu_native_surface_policy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = test_target(device, &identity, format, 220, 144);
    let budget = Mesh3dRenderBudget::new(0, 0, 0)
        .with_max_surface_triangles(128)
        .with_surface_policy(SurfaceRasterization3d::Native);
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap());
    let mut first_winding = None;
    for reverse in [false, true] {
        let source = logic_floor(reverse);
        let mesh = create_retained_mesh(device, queue, Arc::clone(&identity), source).unwrap();
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        let id = scene.try_push(&mesh, Transform3d::IDENTITY, style).unwrap();
        for eye_z in [11.0, 11.001] {
            let camera = logic_camera(eye_z);
            let report = renderer
                .render_scene3d(device, queue, &identity, &target, &scene, camera, budget)
                .unwrap();
            assert_eq!(report.triangle_count(), 128);
            assert_eq!(report.preflight().submitted_triangle_count(), 128);
            assert_eq!(report.preflight().generated_triangle_count(), 0);
            assert_eq!(report.preflight().clipped_source_triangle_count(), None);
            assert_eq!(report.preflight().discarded_source_triangle_count(), None);
            assert_eq!(
                report.preflight().surface_policy(),
                SurfaceRasterization3d::Native
            );
            let pixels = test_read_pixels(device, queue, &target);
            assert!(
                pixels
                    .chunks_exact(4)
                    .filter(|pixel| pixel[0] > 200)
                    .count()
                    > 16
            );
            if eye_z == 11.0 {
                if let Some(reference) = &first_winding {
                    assert_eq!(
                        &pixels, reference,
                        "two-sided floor must preserve coverage when winding reverses"
                    );
                } else {
                    first_winding = Some(pixels.clone());
                }
            }
            if reverse && eye_z == 11.001 {
                renderer
                    .preflight_scene3d(
                        device,
                        &identity,
                        &target,
                        &scene,
                        camera,
                        Mesh3dRenderBudget::default(),
                    )
                    .unwrap();
            } else {
                let error = renderer
                    .render_scene3d(
                        device,
                        queue,
                        &identity,
                        &target,
                        &scene,
                        camera,
                        Mesh3dRenderBudget::default(),
                    )
                    .unwrap_err();
                assert_eq!(error.object_id(), Some(id));
                assert_eq!(error.source_triangle_index(), Some(63));
                assert_eq!(
                    error.surface_reason(),
                    Some(Mesh3dSurfaceError::ProjectedOrientation)
                );
                assert_eq!(
                    test_read_pixels(device, queue, &target),
                    pixels,
                    "strict rejection must preserve the last native frame"
                );
            }
            let error = renderer
                .render_scene3d(
                    device,
                    queue,
                    &identity,
                    &target,
                    &scene,
                    camera,
                    budget.with_max_surface_triangles(127),
                )
                .unwrap_err();
            assert_eq!(
                error,
                Mesh3dRenderError::SurfaceTriangleBudgetExceeded {
                    limit: 127,
                    actual: 128
                }
            );
            assert_eq!(test_read_pixels(device, queue, &target), pixels);
        }
    }
    assert_isolated_logic_triangles(device, queue, format);
    assert_camera_plane_pixels(device, queue, format);
}

fn assert_isolated_logic_triangles(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = test_target(device, &identity, format, 1100, 720);
    let floor = logic_floor(false);
    for triangle_index in [63, 78] {
        let indices = &floor.triangle_indices()[triangle_index * 3..triangle_index * 3 + 3];
        let points = [
            floor.vertices()[indices[0] as usize],
            floor.vertices()[indices[1] as usize],
            floor.vertices()[indices[2] as usize],
        ];
        for reverse in [false, true] {
            let mesh = create_retained_mesh(
                device,
                queue,
                Arc::clone(&identity),
                Mesh3d::with_display_edges(
                    points.to_vec(),
                    if reverse {
                        vec![0, 2, 1]
                    } else {
                        vec![0, 1, 2]
                    },
                    Vec::new(),
                )
                .unwrap(),
            )
            .unwrap();
            let mut scene = Scene3d::new(Color::BLACK).unwrap();
            scene
                .try_push(
                    &mesh,
                    Transform3d::IDENTITY,
                    MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
                )
                .unwrap();
            for (eye_z, yaw_offset) in [(11.0, 0.0), (11.001, 0.0), (11.0, 0.05), (11.001, 0.1)] {
                let camera = logic_camera_at(eye_z, yaw_offset);
                let report = renderer
                    .render_scene3d(
                        device,
                        queue,
                        &identity,
                        &target,
                        &scene,
                        camera,
                        Mesh3dRenderBudget::new(0, 0, 0)
                            .with_surface_policy(SurfaceRasterization3d::Native)
                            .with_max_surface_triangles(1),
                    )
                    .unwrap();
                assert_eq!(report.triangle_count(), 1);
                let pixels = test_read_pixels(device, queue, &target);
                let projected = points.map(|position| {
                    camera
                        .project_world(position, target.logical_viewport())
                        .unwrap()
                        .logical_position()
                        .to_vec2()
                });
                let cross = |a: Vec2, b: Vec2, x: f64, y: f64| {
                    let dx = f64::from(b.x()) - f64::from(a.x());
                    let dy = f64::from(b.y()) - f64::from(a.y());
                    (dx * (y - f64::from(a.y())) - dy * (x - f64::from(a.x()))) / dx.hypot(dy)
                };
                let sign = cross(
                    projected[0],
                    projected[1],
                    f64::from(projected[2].x()),
                    f64::from(projected[2].y()),
                )
                .signum();
                let mut interior = 0;
                for y in 0..target.height() {
                    for x in 0..target.width() {
                        let distances = std::array::from_fn::<_, 3, _>(|edge| {
                            sign * cross(
                                projected[edge],
                                projected[(edge + 1) % 3],
                                f64::from(x) + 0.5,
                                f64::from(y) + 0.5,
                            )
                        });
                        let value = pixels[((y * target.width() + x) * 4) as usize];
                        // Ignore a one-pixel boundary band; the interior/exterior
                        // oracle does not promise Native cross-adapter edge identity.
                        if distances.iter().all(|distance| *distance > 1.0) {
                            interior += 1;
                            assert!(
                                value > 250,
                                "missing triangle {triangle_index} at ({x},{y}), winding={reverse}, eye={eye_z}"
                            );
                        } else if distances.iter().any(|distance| *distance < -1.0) {
                            assert_eq!(value, 0, "outside triangle {triangle_index} at ({x},{y})");
                        }
                    }
                }
                if yaw_offset == 0.0 {
                    // Original source tips reach only x=1099.976 of a 1100-pixel
                    // target: the subpixel boundary sliver covers no sample center.
                    assert_eq!(interior, 0);
                    assert!(pixels.chunks_exact(4).all(|pixel| pixel[..3] == [0, 0, 0]));
                } else {
                    assert!(
                        interior > 16,
                        "triangle {triangle_index} must become visibly drawable after camera rotation; projected={projected:?}"
                    );
                }
            }
        }
    }
}

fn assert_camera_plane_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = test_target(device, &identity, format, 64, 64);
    let camera = Camera3d::look_at(
        Vec3::ZERO,
        point([0.0, 0.0, -1.0]),
        Vec3::Y,
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(0.25), world(8.0))
            .unwrap(),
    )
    .unwrap();
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap());
    // Both camera-plane and behind-camera apexes reach hardware clipping without
    // a CPU divide. Compare against the explicit cut at the near plane z=-.25.
    for apex_z in [0.0, 1.0] {
        let amount = 0.75 / (apex_z + 1.0);
        let clipped_x = 0.5 * (1.0 - amount);
        let clipped_y = -0.5 + 0.75 * amount;
        let sources = [
            Mesh3d::with_display_edges(
                vec![
                    point([-0.5, -0.5, -1.0]),
                    point([0.5, -0.5, -1.0]),
                    point([0.0, 0.25, apex_z]),
                ],
                vec![0, 1, 2],
                Vec::new(),
            )
            .unwrap(),
            Mesh3d::with_display_edges(
                vec![
                    point([-0.5, -0.5, -1.0]),
                    point([0.5, -0.5, -1.0]),
                    point([clipped_x, clipped_y, -0.25]),
                    point([-clipped_x, clipped_y, -0.25]),
                ],
                vec![0, 1, 2, 0, 2, 3],
                Vec::new(),
            )
            .unwrap(),
        ];
        let mut images = Vec::new();
        for source in sources {
            let mesh = create_retained_mesh(device, queue, Arc::clone(&identity), source).unwrap();
            let mut scene = Scene3d::new(Color::BLACK).unwrap();
            scene.try_push(&mesh, Transform3d::IDENTITY, style).unwrap();
            renderer
                .render_scene3d(
                    device,
                    queue,
                    &identity,
                    &target,
                    &scene,
                    camera,
                    Mesh3dRenderBudget::default()
                        .with_surface_policy(SurfaceRasterization3d::Native),
                )
                .unwrap();
            images.push(test_read_pixels(device, queue, &target));
        }
        assert!(
            images[0]
                .chunks_exact(4)
                .filter(|pixel| pixel[0] > 200)
                .count()
                > 100
        );
        assert_eq!(
            images[0], images[1],
            "camera-plane triangle versus explicitly near-clipped reference, apex_z={apex_z}"
        );
    }
}
