//! The original (unfiltered) and production encoders execute the same ready
//! list, on the release oracle's actual adapter, format and MSAA selection.
use super::*;

fn binding(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bytes: &[u8],
    image: Option<(&Image2d, &wgpu::Sampler)>,
) -> FrameBinding {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("encoding oracle uniform"),
        contents: bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut entries = Vec::new();
    if let Some((image, sampler)) = image {
        entries.extend([
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&image.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ]);
    }
    entries.push(wgpu::BindGroupEntry {
        binding: if image.is_some() { 2 } else { 0 },
        resource: buffer.as_entire_binding(),
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("encoding oracle binding"),
        layout,
        entries: &entries,
    });
    FrameBinding {
        _buffer: buffer,
        bind_group,
    }
}

fn render<const DEDUPLICATE: bool>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &EncodingResources<'_>,
    ready: &[ReadyItem<'_>],
    bindings: &[FrameBinding],
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> (Vec<u8>, SetterCounts) {
    let size = wgpu::Extent3d {
        width: 64,
        height: 64,
        depth_or_array_layers: 1,
    };
    let texture = |samples| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("encoding oracle target"),
            size,
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | if samples == 1 {
                    wgpu::TextureUsages::COPY_SRC
                } else {
                    wgpu::TextureUsages::empty()
                },
            view_formats: &[],
        })
    };
    let target = texture(1);
    let view = target.create_view(&Default::default());
    let multisampled = (sample_count > 1).then(|| texture(sample_count));
    let msaa_view = multisampled
        .as_ref()
        .map(|texture| texture.create_view(&Default::default()));
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("encoding oracle pixels"),
        size: 64 * 64 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let counts = {
        let mut state = PassState::<DEDUPLICATE>::default();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("encoding state differential oracle"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_view.as_ref().unwrap_or(&view),
                resolve_target: msaa_view.as_ref().map(|_| &view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        for (item, binding) in ready.iter().zip(bindings) {
            encode_ready_item(resources, &mut pass, item, binding, &mut state);
        }
        state.counts
    };
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
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
        size,
    );
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let pixels = readback.slice(..).get_mapped_range().unwrap().to_vec();
    (pixels, counts)
}

pub(in crate::renderer) fn assert_gpu_encoding_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    sample_count: u32,
) {
    let pipelines = create_pipeline(device, format, sample_count);
    let images = ImageRenderer::new(device, format, sample_count);
    let identity = Arc::new(());
    let viewport = ResolvedViewport {
        viewport: LogicalViewport::new(64.0, 64.0).unwrap(),
        origin: Vec2::ZERO,
        scissor: ScissorRect {
            x: 0,
            y: 0,
            width: 64,
            height: 64,
        },
        item_clip: None,
        item_clipped_out: false,
    };
    let clipped = ResolvedViewport {
        item_clip: Some(ScissorRect {
            x: 17,
            y: 0,
            width: 31,
            height: 48,
        }),
        ..viewport
    };
    let omitted = ResolvedViewport {
        item_clipped_out: true,
        ..viewport
    };
    let uniform =
        CameraUniform::new(screen_camera(viewport.viewport).unwrap(), viewport.viewport).unwrap();
    let camera_binding = binding(
        device,
        &pipelines.camera_bind_group_layout,
        bytemuck::bytes_of(&uniform),
        None,
    );
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_rect(
            Rect::from_min_size(Vec2::new(3.0, 3.0), Vec2::new(27.0, 17.0)),
            0.0,
            ShapeStyle::filled(Color::rgba(1.0, 0.0, 0.0, 0.4)),
        )
        .unwrap();
    let mut vertices = Vec::new();
    let mut batches = Vec::new();
    tessellate_scene(&scene, &mut vertices, &mut batches).unwrap();
    let buffer = |bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("encoding oracle vertices"),
            contents: bytes,
            usage: wgpu::BufferUsages::VERTEX,
        })
    };
    let vertex_buffer = Arc::new(buffer(bytemuck::cast_slice(&vertices)));
    let cloned_wrapper = vertex_buffer.as_ref().clone();
    assert_eq!(
        *vertex_buffer, cloned_wrapper,
        "wgpu resource equality includes cloned handles"
    );
    let other_vertices: Vec<_> = vertices
        .iter()
        .copied()
        .map(|mut vertex| {
            vertex.world_position[1] += 24.0;
            vertex.color = [0.0, 0.6, 1.0, 0.7];
            vertex
        })
        .collect();
    let other_buffer = buffer(bytemuck::cast_slice(&other_vertices));
    let dynamic = buffer(bytemuck::cast_slice(&[
        DynamicGpu {
            world_position: [35.0, 32.0],
            depth: 0.0,
            color: [1.0, 0.0, 1.0, 0.5],
        },
        DynamicGpu {
            world_position: [61.0, 32.0],
            depth: 0.0,
            color: [1.0, 0.0, 1.0, 0.5],
        },
        DynamicGpu {
            world_position: [48.0, 60.0],
            depth: 0.0,
            color: [1.0, 0.0, 1.0, 0.5],
        },
    ]));
    let image = image::create_image_resources(
        device,
        queue,
        Arc::clone(&identity),
        1,
        1,
        vec![0, 255, 0, 128],
        ImageBudget::new(1, 1, 4).unwrap(),
    )
    .unwrap();
    let region = LogicalViewportRegion::new(
        LogicalScreenPosition::new(8.0, 9.0),
        LogicalViewport::new(24.0, 22.0).unwrap(),
    )
    .unwrap();
    let batch = image::create_image_batch_resources(
        device,
        queue,
        Arc::clone(&identity),
        &image,
        vec![
            ImageSprite2d::new(
                ImageTexelRect::new(0, 0, 1, 1).unwrap(),
                region,
                Color::WHITE,
            )
            .unwrap(),
        ],
        ImageBatchBudget::new(1, 1024).unwrap(),
    )
    .unwrap();
    let empty_batch = image::create_image_batch_resources(
        device,
        queue,
        Arc::clone(&identity),
        &image,
        vec![],
        ImageBatchBudget::new(1, 1024).unwrap(),
    )
    .unwrap();
    let batch_uniform = image::batch_uniform(
        viewport.viewport,
        Vec2::ZERO,
        ImageBatchPlacement::default(),
    )
    .unwrap();
    let batch_binding = binding(
        device,
        &images.bind_group_layout,
        bytemuck::bytes_of(&batch_uniform),
        Some((&image, images.sampler(ImageSampling::Nearest))),
    );
    let image_uniform = ImageUniform {
        destination: [0.4, 0.4, 0.55, 0.3],
        uv_rect: [0.0, 0.0, 1.0, 1.0],
        tint: [1.0; 4],
        world_clip_x: [0.0; 4],
        world_clip_y: [0.0; 4],
        world_mode: [0.0; 4],
    };
    let image_binding = binding(
        device,
        &images.bind_group_layout,
        bytemuck::bytes_of(&image_uniform),
        Some((&image, images.sampler(ImageSampling::Nearest))),
    );
    let unit_buffer = create_submitted_particle_unit_buffer(device, queue);
    let particle_gpu = [ParticleGpu {
        world_position: [49.0, 17.0],
        radius: 7.0,
        depth: 0.0,
        color: [1.0, 0.6, 0.0, 0.5],
    }];
    let particle_buffer = Arc::new(buffer(bytemuck::cast_slice(&particle_gpu)));
    let particle_field = || ParticleField2d {
        renderer_identity: Arc::clone(&identity),
        instance_buffer: Arc::clone(&particle_buffer),
        statistics: particle_idle_statistics(1),
        instances: particle_gpu.to_vec(),
        visible_instances: vec![],
        instance_capacity: 1,
        budget: ParticleRenderBudget::UNBOUNDED,
    };
    let mut first_particles = particle_field();
    let mut second_particles = particle_field();
    let geometry = |source, viewport, vertex_count| {
        ReadyItem::Geometry(ReadyGeometry {
            source,
            viewport,
            vertex_count,
            camera_uniform: uniform,
            batches: vec![PreparedDrawBatch {
                vertex_range: 0..vertex_count as u32,
                screen_clip: None,
            }],
        })
    };
    let batch_item = |batch, viewport| ReadyItem::ImageBatch {
        image: &image,
        batch,
        sampling: ImageSampling::Nearest,
        uniform: batch_uniform,
        viewport,
    };
    let image_item = |viewport| ReadyItem::Image {
        image: &image,
        sampling: ImageSampling::Nearest,
        uniform: image_uniform,
        viewport,
    };
    let ready = vec![
        geometry(ReadySource::Streaming, viewport, vertices.len()),
        // Same resource, different wrapper: still one binding.
        geometry(
            ReadySource::Prepared(&cloned_wrapper),
            viewport,
            vertices.len(),
        ),
        batch_item(&batch, viewport), // glyphs use this exact batch pipeline.
        batch_item(&batch, viewport),
        image_item(clipped),
        geometry(ReadySource::Streaming, viewport, vertices.len()),
        batch_item(&batch, viewport),
        batch_item(&empty_batch, clipped),
        image_item(omitted),
        geometry(
            ReadySource::Prepared(&other_buffer),
            omitted,
            vertices.len(),
        ),
        geometry(ReadySource::Streaming, clipped, vertices.len()),
        geometry(
            ReadySource::Prepared(&other_buffer),
            viewport,
            vertices.len(),
        ),
        geometry(
            ReadySource::Prepared(&other_buffer),
            viewport,
            vertices.len(),
        ),
        ReadyItem::Particle {
            field: &mut first_particles,
            visible_count: 1,
            pending_statistics: particle_idle_statistics(1),
            initial_cpu_allocation_bytes: 0,
            camera_uniform: uniform,
            viewport,
        },
        image_item(viewport),
        ReadyItem::Particle {
            field: &mut second_particles,
            visible_count: 1,
            pending_statistics: particle_idle_statistics(1),
            initial_cpu_allocation_bytes: 0,
            camera_uniform: uniform,
            viewport,
        },
        geometry(ReadySource::Dynamic(&dynamic), viewport, 3),
        geometry(ReadySource::Dynamic(&dynamic), clipped, 3),
        geometry(ReadySource::Streaming, viewport, 0),
        geometry(ReadySource::Streaming, clipped, vertices.len()),
    ];
    let bindings: Vec<_> = ready
        .iter()
        .map(|item| match item {
            ReadyItem::Geometry(_) | ReadyItem::Particle { .. } => camera_binding.clone(),
            ReadyItem::ImageBatch { .. } => batch_binding.clone(),
            ReadyItem::Image { .. } => image_binding.clone(),
            _ => unreachable!(),
        })
        .collect();
    let resources = EncodingResources {
        pipeline: &pipelines.pipeline,
        dynamic_pipeline: &pipelines.dynamic_pipeline,
        particle_pipeline: &pipelines.particle_pipeline,
        heatmap_pipeline: &pipelines.heatmap_pipeline,
        image_renderer: &images,
        composition_pipelines: &pipelines.composition_pipelines,
        vertex_buffer: &vertex_buffer,
        particle_unit_buffer: &unit_buffer,
        scale_factor: 1.0,
    };
    let (expected, baseline) = render::<false>(
        device,
        queue,
        &resources,
        &ready,
        &bindings,
        format,
        sample_count,
    );
    let (actual, optimized) = render::<true>(
        device,
        queue,
        &resources,
        &ready,
        &bindings,
        format,
        sample_count,
    );
    assert_eq!(
        expected, actual,
        "filtered encoding changed painter-order pixels"
    );
    assert!(
        actual
            .chunks_exact(4)
            .filter(|pixel| pixel[..3] != [0, 0, 0])
            .count()
            > 500,
        "the differential oracle must draw visible pixels"
    );
    assert_eq!(
        (baseline.vertex_buffers, optimized.vertex_buffers),
        (17, 11)
    );
    assert_eq!((baseline.scissors, optimized.scissors), (16, 6));
    // A fresh pass must populate its own state even with identical resources.
    let (repeated, repeated_counts) = render::<true>(
        device,
        queue,
        &resources,
        &ready,
        &bindings,
        format,
        sample_count,
    );
    assert_eq!(actual, repeated);
    assert_eq!(optimized.vertex_buffers, repeated_counts.vertex_buffers);
    assert_eq!(optimized.scissors, repeated_counts.scissors);
    eprintln!(
        "Frame encoding oracle: {format:?} MSAAx{sample_count}, vertex setters {} -> {}, scissor setters {} -> {}, identical pixels",
        baseline.vertex_buffers, optimized.vertex_buffers, baseline.scissors, optimized.scissors
    );
}
