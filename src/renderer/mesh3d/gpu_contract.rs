//! Semantic pixels, recovery and shader clipping equivalence.

use super::*;

#[cfg(test)]
pub(in crate::renderer) fn assert_gpu_depth_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    accounting_benchmark::run(device, queue);
    batch_tests::assert_gpu_batch_contract(device, queue, format);
    edge_upload_tests::assert_gpu_edge_upload_contract(device, queue, format);
    frame_upload_tests::assert_gpu_dirty_frame_upload_contract(device, queue, format);
    dynamic::assert_gpu_dynamic_contract(device, queue, format);
    vertex_color_tests::assert_gpu_vertex_color_contract(device, queue, format);
    material_tests::assert_gpu_material_contract(device, queue, format);
    lighting_tests::assert_gpu_lighting_contract(device, queue, format);
    material_resource_tests::assert_gpu_sort_budget(device, queue, format);
    color_budget_tests::assert_gpu_color_budget(device, queue, format);
    surface::assert_gpu_surface_contract(device, queue, format);
    surface::assert_gpu_source_cache_contract(device, queue, format);
    surface::assert_gpu_preflight_scratch(device, queue);
    surface::assert_gpu_native_surface_policy(device, queue, format);
    surface::assert_gpu_native_edge_validation(device, queue, format);
    surface::assert_gpu_offscreen_culling(device, queue, &Arc::new(()));
    texture::assert_gpu_texture_contract(device, queue, format);
    texture::assert_gpu_texture_lifecycle(device, queue, format);
    native_acceptance_tests::assert_gpu_native_acceptance(device, queue, format);
    assert_gpu_clip_equivalence(device, queue);
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let vertices = vec![
        Vec3::new(-1.0, -1.0, 0.0).unwrap(),
        Vec3::new(1.0, -1.0, 0.0).unwrap(),
        Vec3::new(0.0, 1.0, 0.0).unwrap(),
    ];
    let near_triangle = Mesh3d::with_display_edges(
        vertices.clone(),
        vec![0, 1, 2],
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let far_triangle = Mesh3d::with_display_edges(
        vertices,
        vec![0, 1, 2],
        vec![MeshEdge3d::new(0, 2).unwrap()],
    )
    .unwrap();
    let near_mesh = create_retained_mesh(device, queue, Arc::clone(&identity), near_triangle)
        .expect("3D test mesh should upload");
    let far_mesh = create_retained_mesh(device, queue, Arc::clone(&identity), far_triangle)
        .expect("3D test mesh should upload");
    let near_crossing_edge = Mesh3d::with_display_edges(
        vec![
            Vec3::new(0.0, 0.0, 4.95).unwrap(),
            Vec3::new(0.5, 0.5, 4.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let near_crossing_mesh =
        create_retained_mesh(device, queue, Arc::clone(&identity), near_crossing_edge)
            .expect("near-crossing 3D test edge should upload");
    let classification_edge = Mesh3d::with_display_edges(
        vec![
            Vec3::new(-0.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.5, -0.5, 0.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let classification_mesh =
        create_retained_mesh(device, queue, Arc::clone(&identity), classification_edge)
            .expect("classification test edge should upload");
    let projection = Projection3d::perspective(0.927_295_2, 1.0, world(0.1), world(20.0)).unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 5.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        projection,
    )
    .unwrap();
    let camera_uniform = Camera3dUniform::new(camera, 64, 64, physical_per_logical(1.0)).unwrap();
    let near_transform = Transform3d::new(
        Vec3::new(0.0, 0.0, 1.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let far_transform = Transform3d::new(
        Vec3::new(0.0, 0.0, -1.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let near_id = scene
        .try_push(
            &near_mesh,
            near_transform,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::rgb(0.0, 1.0, 0.0)).unwrap())
                .with_wireframe(
                    WireframeStyle3d::visible(Color::rgb(0.0, 0.0, 1.0), logical(2.0))
                        .unwrap()
                        .with_hidden(
                            Color::rgb(1.0, 0.0, 1.0),
                            logical(2.0),
                            logical(4.0),
                            logical(4.0),
                        )
                        .unwrap(),
                ),
        )
        .unwrap();
    let mut other_scene = Scene3d::new(Color::BLACK).unwrap();
    let other_scene_id = other_scene
        .try_push(
            &near_mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
        )
        .unwrap();
    assert_eq!(near_id.get(), other_scene_id.get());
    assert_ne!(near_id, other_scene_id);
    assert_eq!(
        other_scene.set_visible(near_id, false),
        Err(Scene3dError::ObjectNotFound { object_id: near_id })
    );
    assert_eq!(other_scene.visible_object_count(), 1);
    assert_eq!(
        scene.set_visible(other_scene_id, false),
        Err(Scene3dError::ObjectNotFound {
            object_id: other_scene_id,
        })
    );
    assert_eq!(scene.visible_object_count(), 1);
    let far_id = scene
        .try_push(
            &far_mesh,
            far_transform,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::rgb(1.0, 0.0, 0.0)).unwrap())
                .with_wireframe(
                    WireframeStyle3d::visible(Color::rgb(1.0, 0.0, 0.0), logical(2.0))
                        .unwrap()
                        .with_hidden(Color::WHITE, logical(2.0), logical(4.0), logical(4.0))
                        .unwrap(),
                ),
        )
        .unwrap();
    scene
        .try_push(
            &near_crossing_mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::wireframe(
                WireframeStyle3d::visible(Color::rgb(1.0, 1.0, 0.0), logical(2.0)).unwrap(),
            ),
        )
        .unwrap();
    let coplanar_transform = Transform3d::new(
        Vec3::new(0.0, 0.0, 1.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    scene
        .try_push(
            &classification_mesh,
            coplanar_transform,
            MeshStyle3d::wireframe(
                WireframeStyle3d::visible(Color::rgb(0.0, 1.0, 1.0), logical(2.0))
                    .unwrap()
                    .with_hidden(
                        Color::rgb(1.0, 0.0, 1.0),
                        logical(2.0),
                        logical(4.0),
                        logical(4.0),
                    )
                    .unwrap(),
            ),
        )
        .unwrap();
    let sub_depth_resolution_transform = Transform3d::new(
        Vec3::new(0.0, 0.2, 0.999_999).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    scene
        .try_push(
            &classification_mesh,
            sub_depth_resolution_transform,
            MeshStyle3d::wireframe(
                WireframeStyle3d::visible(Color::rgb(1.0, 0.5, 0.0), logical(2.0))
                    .unwrap()
                    .with_hidden(
                        Color::rgb(1.0, 0.0, 1.0),
                        logical(2.0),
                        logical(4.0),
                        logical(4.0),
                    )
                    .unwrap(),
            ),
        )
        .unwrap();
    let moved_near_transform = Transform3d::new(
        Vec3::new(0.25, 0.0, 1.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    scene.set_transform(near_id, moved_near_transform).unwrap();
    assert_eq!(scene.instances()[0].transform(), moved_near_transform);
    assert_eq!(scene.instances()[1].transform(), far_transform);
    assert_eq!(scene.instances()[1].id(), far_id);
    scene.set_visible(far_id, false).unwrap();
    assert_eq!(scene.visible_object_count(), 4);
    scene.set_visible(far_id, true).unwrap();
    assert_eq!(scene.visible_object_count(), 5);
    scene.set_transform(near_id, near_transform).unwrap();
    renderer
        .ensure_frame_capacity(device, scene.object_count(), scene.object_count())
        .unwrap();
    renderer.instances = scene
        .instances()
        .iter()
        .map(|instance| {
            let rows = instance.transform.model_rows().unwrap();
            MeshInstanceGpu {
                normal_row_0: [0.0; 4],
                normal_row_1: [0.0; 4],
                normal_row_2: [0.0; 4],
                model_row_0: rows[0],
                model_row_1: rows[1],
                model_row_2: rows[2],
                surface: material::surface_parameters(instance.style.surface_style()),
                uv_transform: texture::material_uv_parameters(instance.mesh.material()),
                color: instance
                    .style
                    .surface_style()
                    .map_or(Color::BLACK, SurfaceStyle3d::color)
                    .to_array(),
            }
        })
        .collect();
    renderer
        .edge_object_bytes
        .resize(scene.object_count() * renderer.edge_object_stride, 0);
    let near_rows = near_transform.model_rows().unwrap();
    let near_edge_uniform = EdgeObjectUniform {
        model_row_0: near_rows[0],
        model_row_1: near_rows[1],
        model_row_2: near_rows[2],
        visible_color: Color::rgb(0.0, 0.0, 1.0).to_array(),
        hidden_color: Color::rgb(1.0, 0.0, 1.0).to_array(),
        edge_style: [2.0, 2.0, 4.0, 4.0],
    };
    renderer.edge_object_bytes[..std::mem::size_of::<EdgeObjectUniform>()]
        .copy_from_slice(bytemuck::bytes_of(&near_edge_uniform));
    let far_rows = far_transform.model_rows().unwrap();
    let far_edge_uniform = EdgeObjectUniform {
        model_row_0: far_rows[0],
        model_row_1: far_rows[1],
        model_row_2: far_rows[2],
        visible_color: Color::rgb(1.0, 0.0, 0.0).to_array(),
        hidden_color: Color::WHITE.to_array(),
        edge_style: [2.0, 2.0, 4.0, 4.0],
    };
    let edge_start = renderer.edge_object_stride;
    let edge_end = edge_start + std::mem::size_of::<EdgeObjectUniform>();
    renderer.edge_object_bytes[edge_start..edge_end]
        .copy_from_slice(bytemuck::bytes_of(&far_edge_uniform));
    let crossing_rows = Transform3d::IDENTITY.model_rows().unwrap();
    let crossing_edge_uniform = EdgeObjectUniform {
        model_row_0: crossing_rows[0],
        model_row_1: crossing_rows[1],
        model_row_2: crossing_rows[2],
        visible_color: Color::rgb(1.0, 1.0, 0.0).to_array(),
        hidden_color: Color::rgb(1.0, 1.0, 0.0).to_array(),
        edge_style: [2.0, 0.0, 1.0, 1.0],
    };
    let crossing_edge_start = 2 * renderer.edge_object_stride;
    let crossing_edge_end = crossing_edge_start + std::mem::size_of::<EdgeObjectUniform>();
    renderer.edge_object_bytes[crossing_edge_start..crossing_edge_end]
        .copy_from_slice(bytemuck::bytes_of(&crossing_edge_uniform));
    let coplanar_rows = coplanar_transform.model_rows().unwrap();
    let coplanar_edge_uniform = EdgeObjectUniform {
        model_row_0: coplanar_rows[0],
        model_row_1: coplanar_rows[1],
        model_row_2: coplanar_rows[2],
        visible_color: Color::rgb(0.0, 1.0, 1.0).to_array(),
        hidden_color: Color::rgb(1.0, 0.0, 1.0).to_array(),
        edge_style: [2.0, 2.0, 4.0, 4.0],
    };
    let coplanar_edge_start = 3 * renderer.edge_object_stride;
    let coplanar_edge_end = coplanar_edge_start + std::mem::size_of::<EdgeObjectUniform>();
    renderer.edge_object_bytes[coplanar_edge_start..coplanar_edge_end]
        .copy_from_slice(bytemuck::bytes_of(&coplanar_edge_uniform));
    let sub_depth_rows = sub_depth_resolution_transform.model_rows().unwrap();
    let sub_depth_edge_uniform = EdgeObjectUniform {
        model_row_0: sub_depth_rows[0],
        model_row_1: sub_depth_rows[1],
        model_row_2: sub_depth_rows[2],
        visible_color: Color::rgb(1.0, 0.5, 0.0).to_array(),
        hidden_color: Color::rgb(1.0, 0.0, 1.0).to_array(),
        edge_style: [2.0, 2.0, 4.0, 4.0],
    };
    let sub_depth_edge_start = 4 * renderer.edge_object_stride;
    let sub_depth_edge_end = sub_depth_edge_start + std::mem::size_of::<EdgeObjectUniform>();
    renderer.edge_object_bytes[sub_depth_edge_start..sub_depth_edge_end]
        .copy_from_slice(bytemuck::bytes_of(&sub_depth_edge_uniform));
    queue.write_buffer(
        &renderer.camera_uniform_buffer,
        0,
        bytemuck::bytes_of(&camera_uniform),
    );
    queue.write_buffer(
        &renderer.instance_buffer,
        0,
        bytemuck::cast_slice(&renderer.instances),
    );
    queue.write_buffer(&renderer.edge_object_buffer, 0, &renderer.edge_object_bytes);
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine 3D depth test color"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_texture = create_depth_texture(device, 64, 64);
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D depth test readback"),
        size: 256 * 64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sim-engine 3D depth test encoder"),
    });
    encode_scene_pass(
        &mut encoder,
        &renderer,
        &color_view,
        &depth_view,
        Color::BLACK,
        scene.instances(),
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &color_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(64),
            },
        },
        wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("3D depth test submission should complete");
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("3D depth test mapping should complete");
    receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("3D depth test callback should run")
        .expect("3D depth readback should map");
    let bytes = slice.get_mapped_range().expect("3D depth bytes");
    let [red, green, blue] = match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2],
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0],
        _ => panic!("unsupported 3D byte-channel oracle format: {format:?}"),
    };
    let center_offset = 32 * 256 + 32 * 4;
    assert!(
        bytes[center_offset + green] > 200,
        "near green surface must win depth testing"
    );
    assert!(
        bytes[center_offset + red] < 20,
        "far red surface must remain occluded"
    );
    for x in 20..44 {
        let offset = 48 * 256 + x * 4;
        assert!(
            bytes[offset + blue] > 180 && bytes[offset + red] < 80,
            "coplanar near edge must remain solid visible blue at x={x}"
        );
    }
    let mut hidden_pattern = Vec::new();
    for step in 1..20 {
        let amount = step as f32 / 20.0;
        let x = (21.33 + (32.0 - 21.33) * amount).round() as usize;
        let y = (42.67 + (21.33 - 42.67) * amount).round() as usize;
        let offset = y * 256 + x * 4;
        hidden_pattern
            .push(bytes[offset] > 220 && bytes[offset + 1] > 220 && bytes[offset + 2] > 220);
    }
    assert!(
        hidden_pattern.windows(2).any(|pair| pair == [true, false])
            && hidden_pattern.windows(2).any(|pair| pair == [false, true]),
        "hidden edge must alternate ordered white dashes and gaps: {hidden_pattern:?}"
    );
    let near_clipped_yellow_pixels = bytes
        .chunks_exact(4)
        .filter(|pixel| pixel[red] > 180 && pixel[green] > 180 && pixel[blue] < 80)
        .count();
    assert!(
        near_clipped_yellow_pixels >= 4,
        "edge crossing the near plane must remain visible after homogeneous clipping: {near_clipped_yellow_pixels} yellow pixels"
    );
    let different_object_coplanar_pixels = bytes
        .chunks_exact(4)
        .filter(|pixel| pixel[red] < 80 && pixel[green] > 180 && pixel[blue] > 180)
        .count();
    assert!(
        different_object_coplanar_pixels >= 4,
        "a coplanar edge from another object must resolve visible: {different_object_coplanar_pixels} cyan pixels"
    );
    let sub_depth_resolution_pixels = bytes
        .chunks_exact(4)
        .filter(|pixel| pixel[red] > 180 && pixel[green] > 100 && pixel[blue] < 80)
        .count();
    assert!(
        sub_depth_resolution_pixels >= 4,
        "sub-depth-resolution separation must conservatively resolve visible: {sub_depth_resolution_pixels} orange pixels"
    );
    drop(bytes);
    readback.unmap();
}

#[cfg(test)]
pub(in crate::renderer) fn assert_gpu_scene_recovery_contract(
    source_device: &wgpu::Device,
    source_queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    objects::accounting_gpu_tests::verify_scene_accounting(
        source_device,
        source_queue,
        recovery_device,
        recovery_queue,
    );
    surface::assert_gpu_offscreen_culling_recovery(
        source_device,
        source_queue,
        recovery_device,
        recovery_queue,
    );
    texture::assert_dev5_contract(
        source_device,
        source_queue,
        recovery_device,
        recovery_queue,
        format,
    );
    texture::assert_gpu_texture_lifecycle_recovery(
        source_device,
        source_queue,
        recovery_device,
        recovery_queue,
        format,
    );
    dynamic::assert_gpu_dynamic_recovery(
        source_device,
        source_queue,
        recovery_device,
        recovery_queue,
    );
    material_resource_tests::assert_gpu_alpha_recovery(
        source_device,
        source_queue,
        recovery_device,
        recovery_queue,
    );
    lifetime_tests::assert_lifetime_contract(
        source_device,
        source_queue,
        recovery_device,
        recovery_queue,
    );
    surface::assert_gpu_clipped_recovery(
        source_device,
        source_queue,
        recovery_device,
        recovery_queue,
    );
    texture::assert_gpu_textured_recovery(
        source_device,
        source_queue,
        recovery_device,
        recovery_queue,
    );
    let source_identity = Arc::new(());
    let recovery_identity = Arc::new(());
    let topology = Mesh3d::with_display_edges(
        vec![
            Vec3::new(-0.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.0, 0.5, 0.0).unwrap(),
        ],
        vec![0, 1, 2],
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let source_mesh = create_retained_mesh(
        source_device,
        source_queue,
        Arc::clone(&source_identity),
        topology,
    )
    .unwrap();
    let transform_a = Transform3d::IDENTITY;
    let transform_b = Transform3d::new(
        Vec3::new(1.0, 2.0, 3.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(2.0, 2.0, 2.0).unwrap(),
    )
    .unwrap();
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap())
        .with_wireframe(WireframeStyle3d::visible(Color::BLACK, logical(1.0)).unwrap());
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let first_id = scene.try_push(&source_mesh, transform_a, style).unwrap();
    let second_id = scene.try_push(&source_mesh, transform_b, style).unwrap();
    scene.set_visible(second_id, false).unwrap();

    assert_eq!(
        validate_mesh_identity(&recovery_identity, scene.instances()[0].mesh()),
        Err(Mesh3dRenderError::RendererMismatch)
    );
    let report = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        &Mesh3dRenderer::new(recovery_device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .textures
            .layout,
        Arc::clone(&recovery_identity),
        &mut scene,
    )
    .unwrap();
    assert_eq!(report.object_count(), 2);
    assert_eq!(report.migrated_object_count(), 2);
    assert_eq!(report.restored_mesh_count(), 1);
    assert_eq!(
        report.restored_gpu_bytes(),
        source_mesh.gpu_allocation_bytes()
    );
    assert_eq!(scene.instances()[0].id(), first_id);
    assert_eq!(scene.instances()[1].id(), second_id);
    assert_eq!(scene.instances()[0].transform(), transform_a);
    assert_eq!(scene.instances()[1].transform(), transform_b);
    assert_eq!(scene.instances()[0].style(), style);
    assert_eq!(scene.instances()[1].style(), style);
    assert!(scene.instances()[0].is_visible());
    assert!(!scene.instances()[1].is_visible());
    assert!(Arc::ptr_eq(
        &scene.instances()[0].mesh.vertex_buffer,
        &scene.instances()[1].mesh.vertex_buffer,
    ));
    for instance in scene.instances() {
        assert_eq!(
            validate_mesh_identity(&recovery_identity, instance.mesh()),
            Ok(())
        );
        assert_eq!(
            validate_mesh_identity(&source_identity, instance.mesh()),
            Err(Mesh3dRenderError::RendererMismatch)
        );
    }

    let no_op_report = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        &Mesh3dRenderer::new(recovery_device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .textures
            .layout,
        recovery_identity,
        &mut scene,
    )
    .unwrap();
    assert_eq!(no_op_report.object_count(), 2);
    assert_eq!(no_op_report.migrated_object_count(), 0);
    assert_eq!(no_op_report.restored_mesh_count(), 0);
    assert_eq!(no_op_report.restored_gpu_bytes(), 0);
}

#[cfg(test)]
fn assert_gpu_clip_equivalence(device: &wgpu::Device, queue: &wgpu::Queue) {
    let inputs = seeded_clip_probe_inputs();
    let input_bytes = bytemuck::cast_slice(&inputs);
    let output_size = (inputs.len() * std::mem::size_of::<ClipProbeOutputGpu>()) as u64;
    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D clip probe input"),
        size: input_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D clip probe output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D clip probe readback"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, input_bytes);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sim-engine 3D clip equivalence shader"),
        source: wgpu::ShaderSource::Wgsl(lighting::shader_source(include_str!("primitive.wgsl"))),
    });
    let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sim-engine empty clip probe layout"),
        entries: &[],
    });
    let probe_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sim-engine 3D clip probe layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sim-engine 3D clip probe pipeline layout"),
        bind_group_layouts: &[
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&probe_layout),
        ],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sim-engine 3D clip probe pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("mesh3d_clip_probe_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sim-engine 3D clip probe bind group"),
        layout: &probe_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sim-engine 3D clip probe encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sim-engine 3D clip probe pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(2, &bind_group, &[]);
        pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_size);
    queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("3D clip probe submission should complete");
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("3D clip probe mapping should complete");
    receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("3D clip probe callback should run")
        .expect("3D clip probe readback should map");
    let bytes = slice.get_mapped_range().expect("3D clip probe bytes");
    let outputs: &[ClipProbeOutputGpu] = bytemuck::cast_slice(&bytes);
    for (case_index, (input, actual)) in inputs.iter().zip(outputs).enumerate() {
        let expected = clip_edge_to_frustum_details(input.start_clip, input.end_clip)
            .expect("finite seeded clip input should not overflow after normalization");
        assert_eq!(
            actual.visible,
            u32::from(expected.is_some()),
            "CPU/WGSL clip visibility differs for seeded case {case_index}: {input:?}"
        );
        let Some(expected) = expected else {
            continue;
        };
        for (actual, expected) in actual
            .start_clip
            .into_iter()
            .chain(actual.end_clip)
            .chain(actual.range)
            .zip(
                expected.clip[0]
                    .into_iter()
                    .chain(expected.clip[1])
                    .chain([expected.enter, expected.exit]),
            )
        {
            let tolerance = 32.0 * f32::EPSILON * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "CPU/WGSL clip value differs for seeded case {case_index}: actual={actual}, expected={expected}, tolerance={tolerance}, input={input:?}"
            );
        }
    }
    drop(bytes);
    readback.unmap();
}

#[cfg(test)]
fn seeded_clip_probe_inputs() -> Vec<ClipProbeInputGpu> {
    let maximum = MAX_PORTABLE_SHADER_VALUE;
    let mut inputs = vec![
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, 0.5, 1.0],
            end_clip: [0.5, 0.5, 0.75, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [-2.0, 0.0, 0.5, 1.0],
            end_clip: [0.5, 0.0, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, 0.5, 1.0],
            end_clip: [2.0, 0.0, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, -2.0, 0.5, 1.0],
            end_clip: [0.0, 0.5, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, 0.5, 1.0],
            end_clip: [0.0, 2.0, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, -1.0, 1.0],
            end_clip: [0.0, 0.0, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, 0.5, 1.0],
            end_clip: [0.0, 0.0, 2.0, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [-2.0, -2.0, 0.5, 1.0],
            end_clip: [0.5, 0.5, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, 0.5, 1.0],
            end_clip: [2.0, 2.0, 2.0, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [-1.0, 1.0, 0.0, 1.0],
            end_clip: [1.0, -1.0, 1.0, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [maximum, 0.0, maximum * 0.5, maximum],
            end_clip: [-maximum, maximum, maximum, maximum],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, -maximum, -maximum],
            end_clip: [0.0, 0.0, -maximum * 0.5, -maximum * 0.5],
        },
    ];
    let mut state = 0x5eed_c1a5_u32;
    for _ in 0..244 {
        let mut next_component = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_215.0) * 4.0 - 2.0
        };
        inputs.push(ClipProbeInputGpu {
            start_clip: [
                next_component(),
                next_component(),
                next_component(),
                next_component(),
            ],
            end_clip: [
                next_component(),
                next_component(),
                next_component(),
                next_component(),
            ],
        });
    }
    debug_assert_eq!(inputs.len(), 256);
    inputs
}
