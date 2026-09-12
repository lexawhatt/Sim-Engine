//! Actual retained/generated draw accounting and optional timing-hook regressions.

use super::*;

pub(in crate::renderer) fn assert_gpu_scene_timing_and_upload_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let target = surface::test_target(device, &identity, format, 64, 64);
    let mut renderer = Mesh3dRenderer::new(device, format);
    let mut timing = gpu_timing::GpuTimingCollector::new(device, queue, true);
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 3.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(8.0)).unwrap(),
    )
    .unwrap();
    let triangle = Mesh3d::new(
        vec![
            Vec3::new(-0.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.0, 0.5, 0.0).unwrap(),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let mesh = create_retained_mesh(device, queue, Arc::clone(&identity), triangle).unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    for _ in 0..17 {
        scene
            .try_push(
                &mesh,
                Transform3d::IDENTITY,
                MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
            )
            .unwrap();
    }
    let expected_bytes = std::mem::size_of::<Camera3dUniform>()
        + 17 * (std::mem::size_of::<MeshInstanceGpu>() + renderer.edge_object_stride);
    let native = Mesh3dRenderBudget::default().with_surface_policy(SurfaceRasterization3d::Native);
    let old_gpu = renderer.retained_frame_gpu_bytes();
    let first = renderer
        .render_scene3d_with_timing(
            device,
            queue,
            &identity,
            &target,
            &scene,
            camera,
            native,
            Some(&mut timing),
        )
        .unwrap();
    assert_eq!(first.uploaded_bytes(), expected_bytes);
    assert_eq!(first.upload_calls(), 3);
    assert_eq!(first.buffer_allocation_count(), 2);
    assert_eq!(
        first.upload(),
        first.preflight_duration() + first.staging_upload_duration()
    );
    assert_eq!(
        first.retained_frame_cpu_bytes(),
        first.staging_capacity_bytes()
    );
    assert_eq!(
        first.retained_frame_cpu_bytes(),
        first.peak_frame_cpu_bytes()
    );
    assert_eq!(
        first.retained_frame_gpu_bytes(),
        renderer.retained_frame_gpu_bytes()
    );
    assert_eq!(
        first.peak_frame_gpu_bytes(),
        old_gpu
            + renderer.instance_buffer.size() as usize
            + renderer.edge_object_buffer.size() as usize
    );
    let reused = renderer
        .render_scene3d_with_timing(
            device,
            queue,
            &identity,
            &target,
            &scene,
            camera,
            native,
            Some(&mut timing),
        )
        .unwrap();
    assert_eq!(reused.uploaded_bytes(), expected_bytes);
    assert_eq!(reused.upload_calls(), 3);
    assert_eq!(reused.buffer_allocation_count(), 0);
    assert_eq!(
        reused.retained_frame_gpu_bytes(),
        first.retained_frame_gpu_bytes()
    );
    assert_eq!(
        reused.peak_frame_gpu_bytes(),
        reused.retained_frame_gpu_bytes()
    );
    assert_eq!(
        reused.peak_frame_cpu_bytes(),
        reused.retained_frame_cpu_bytes()
            + renderer.clipped_surface_objects.capacity() * std::mem::size_of::<SurfaceObject>()
    );

    let source = Mesh3d::with_attributes(
        vec![
            Vec3::new(-1.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.5, 0.5, 0.0).unwrap(),
            Vec3::new(-1.5, 0.5, 0.0).unwrap(),
        ],
        vec![0, 1, 2, 0, 2, 3],
        vec![],
        crate::Mesh3dAttributes::new()
            .with_vertex_colors(vec![Color::WHITE; 4])
            .unwrap()
            .with_normals(vec![Vec3::Z; 4])
            .unwrap(),
    )
    .unwrap();
    let mesh = create_retained_mesh(device, queue, Arc::clone(&identity), source).unwrap();
    let mut crossing_scene = Scene3d::new(Color::BLACK).unwrap();
    crossing_scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(
                SurfaceStyle3d::opaque(Color::WHITE)
                    .unwrap()
                    .with_lighting(SurfaceLighting3d::Lambert),
            ),
        )
        .unwrap();
    let generated = renderer
        .render_scene3d_with_timing(
            device,
            queue,
            &identity,
            &target,
            &crossing_scene,
            camera,
            Mesh3dRenderBudget::default(),
            Some(&mut timing),
        )
        .unwrap();
    assert!(generated.preflight().generated_vertex_count() > 0);
    assert_eq!(generated.upload_calls(), 6);
    assert_eq!(generated.buffer_allocation_count(), 3);
    assert_eq!(
        generated.uploaded_bytes(),
        std::mem::size_of::<Camera3dUniform>()
            + std::mem::size_of::<MeshInstanceGpu>()
            + renderer.edge_object_stride
            + generated.preflight().generated_upload_bytes()
    );
    assert!(generated.staging_capacity_bytes() > generated.retained_frame_cpu_bytes());
    assert!(generated.peak_frame_cpu_bytes() >= generated.staging_capacity_bytes());
    let before = timing.statistics();
    assert!(
        renderer
            .render_scene3d_with_timing(
                device,
                queue,
                &identity,
                &target,
                &crossing_scene,
                camera,
                Mesh3dRenderBudget::new(0, 0, 0),
                Some(&mut timing),
            )
            .is_err()
    );
    assert_eq!(timing.statistics().submitted(), before.submitted());
    assert_eq!(timing.statistics().pending(), before.pending());
    let reports = [first, reused, generated];
    let mut seen = [false; 3];
    let deadline = Instant::now() + Duration::from_secs(20);
    while timing.statistics().pending() > 0 {
        for sample in timing.collect(device).samples() {
            let index = reports
                .iter()
                .position(|report| report.gpu_timing_id() == Some(sample.id()))
                .expect("GPU timing must match an actual submitted 3D report");
            assert_eq!(sample.source(), GpuTimingSource::Scene3d);
            assert!(!seen[index]);
            seen[index] = true;
        }
        assert!(Instant::now() < deadline, "timed 3D reports did not finish");
        std::thread::yield_now();
    }
    assert_eq!(seen, [true; 3]);
    assert_eq!(timing.statistics().map_failures(), 0);
    assert_eq!(timing.statistics().invalid_samples(), 0);
    assert_eq!(timing.statistics().poll_failures(), 0);
}
