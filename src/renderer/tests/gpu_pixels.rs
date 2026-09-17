use super::*;

pub(super) async fn assert_gpu_stroke_pixel_matrix(
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 248;
    const ROW_BYTES: u32 = WIDTH * 4;
    let sample_count = preferred_sample_count(adapter, format);
    let PipelineResources {
        pipeline,
        camera_uniform_buffer,
        camera_bind_group,
        ..
    } = create_pipeline(device, format, sample_count);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine stroke pixel-matrix resolve target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let multisample = (sample_count > 1).then(|| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine stroke pixel-matrix multisample target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    });
    let multisample_view = multisample
        .as_ref()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
    let viewport = LogicalViewport::new(WIDTH as f32, HEIGHT as f32).unwrap();
    let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    let camera_uniform = CameraUniform::new(camera, viewport).unwrap();
    queue.write_buffer(
        &camera_uniform_buffer,
        0,
        bytemuck::bytes_of(&camera_uniform),
    );

    let caps = [
        crate::StrokeCap2d::Butt,
        crate::StrokeCap2d::Square,
        crate::StrokeCap2d::Round,
    ];
    let joins = [
        crate::StrokeJoin2d::Bevel,
        crate::StrokeJoin2d::Miter,
        crate::StrokeJoin2d::Round,
    ];
    let mut scene = Scene::new(Color::BLACK).unwrap();
    let screen_to_world =
        |x: f32, y: f32| Vec2::new(x - WIDTH as f32 * 0.5, HEIGHT as f32 * 0.5 - y);
    for width_mode in 0..2 {
        for turn_direction in 0..2 {
            for (cap_index, cap) in caps.iter().copied().enumerate() {
                for (join_index, join) in joins.iter().copied().enumerate() {
                    let row = cap_index * joins.len() + join_index;
                    let center_x = 64.0 + (width_mode * 2 + turn_direction) as f32 * 128.0;
                    let center_y = 12.0 + row as f32 * 24.0;
                    let vertical = if turn_direction == 0 { 6.0 } else { -6.0 };
                    let points = vec![
                        screen_to_world(center_x - 14.0, center_y + vertical),
                        screen_to_world(center_x, center_y - vertical),
                        screen_to_world(center_x + 14.0, center_y + vertical),
                    ];
                    let color = Color::rgba(1.0, 1.0, 1.0, 0.5);
                    let style = if width_mode == 0 {
                        crate::StrokeStyle2d::logical(
                            crate::LogicalPixels::new(10.0).unwrap(),
                            color,
                        )
                    } else {
                        crate::StrokeStyle2d::world(crate::WorldLength::new(10.0).unwrap(), color)
                    }
                    .with_cap(cap)
                    .with_join(join)
                    .with_miter_limit(4.0)
                    .unwrap();
                    scene.try_styled_polyline(points, style).unwrap();
                }
            }
        }
    }
    let marker = crate::StrokeMarker2d::arrow(
        crate::LogicalPixels::new(10.0).unwrap(),
        crate::LogicalPixels::new(12.0).unwrap(),
    );
    scene
        .try_styled_line(
            screen_to_world(254.0, 235.0),
            screen_to_world(258.0, 235.0),
            crate::StrokeStyle2d::logical(
                crate::LogicalPixels::new(10.0).unwrap(),
                Color::rgba(1.0, 1.0, 1.0, 0.5),
            )
            .with_cap(crate::StrokeCap2d::Round)
            .with_start_marker(marker)
            .with_end_marker(marker),
        )
        .unwrap();
    let identity = Arc::new(());
    let prepared = prepare_scene_resources(device, queue, identity, &scene)
        .expect("stroke pixel matrix should prepare");
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine stroke pixel-matrix readback"),
        size: u64::from(ROW_BYTES) * u64::from(HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sim-engine stroke pixel-matrix encoder"),
    });
    {
        let attachment_view = multisample_view.as_ref().unwrap_or(&target_view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sim-engine stroke pixel-matrix pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: attachment_view,
                depth_slice: None,
                resolve_target: multisample_view.as_ref().map(|_| &target_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                    store: if multisample_view.is_some() {
                        wgpu::StoreOp::Discard
                    } else {
                        wgpu::StoreOp::Store
                    },
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &camera_bind_group, &[]);
        pass.set_vertex_buffer(0, prepared.vertex_buffer.slice(..));
        for batch in &prepared.draw_batches {
            pass.draw(batch.vertex_range.clone(), 0..1);
        }
    }
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
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("stroke pixel-matrix submission should complete");
    let bytes =
        crate::renderer::test_support::read_buffer(device, &readback, Duration::from_secs(5));
    let pixel = |x: usize, y: usize| &bytes[y * ROW_BYTES as usize + x * 4..][..4];
    let mut covered = [[[0usize; 9]; 2]; 2];
    for (width_mode, turns) in covered.iter_mut().enumerate() {
        for (turn_direction, counts) in turns.iter_mut().enumerate() {
            let center_x = 64usize + (width_mode * 2 + turn_direction) * 128;
            for (row, count) in counts.iter_mut().enumerate() {
                let center_y = 12usize + row * 24;
                let mut maximum = 0u8;
                for y in center_y.saturating_sub(11)..=(center_y + 11).min(HEIGHT as usize - 1) {
                    for x in center_x - 22..=center_x + 22 {
                        let value = pixel(x, y)[0];
                        maximum = maximum.max(value);
                        *count += usize::from(value > 32);
                    }
                }
                assert!(
                    (175..=210).contains(&maximum),
                    "stroke matrix width_mode={width_mode} turn={turn_direction} row={row} has missing or multiply blended pixels: maximum={maximum}, samples={sample_count}"
                );
                assert!(
                    *count > 100,
                    "stroke matrix width_mode={width_mode} turn={turn_direction} row={row} did not rasterize enough pixels: {count}"
                );
            }
            for join in 0..3 {
                assert!(counts[3 + join] > counts[join]);
                assert!(counts[6 + join] > counts[join]);
            }
            for cap in 0..3 {
                assert!(
                    counts[cap * 3 + 1] > counts[cap * 3],
                    "miter/bevel coverage did not differ for width={width_mode} turn={turn_direction} cap={cap}: {counts:?}"
                );
                assert!(
                    counts[cap * 3 + 2] > counts[cap * 3],
                    "round/bevel coverage did not differ for width={width_mode} turn={turn_direction} cap={cap}: {counts:?}"
                );
            }
        }
    }
    for (turn_direction, (logical_rows, world_rows)) in
        covered[0].iter().zip(covered[1].iter()).enumerate()
    {
        for (row, (logical, world)) in logical_rows
            .iter()
            .copied()
            .zip(world_rows.iter().copied())
            .enumerate()
        {
            assert!(
                logical.abs_diff(world) <= 24,
                "logical/world stroke matrix diverged for turn={turn_direction} row={row}: {:?}",
                [logical, world]
            );
        }
    }
    for (width_mode, turns) in covered.iter().enumerate() {
        for (row, (clockwise, counterclockwise)) in turns[0]
            .iter()
            .copied()
            .zip(turns[1].iter().copied())
            .enumerate()
        {
            assert!(
                clockwise.abs_diff(counterclockwise) <= 24,
                "mirrored stroke matrix diverged for width={width_mode} row={row}: {:?}",
                [clockwise, counterclockwise]
            );
        }
    }
    let short_body = pixel(256, 235)[0];
    let short_start_marker = pixel(248, 235)[0];
    let short_end_marker = pixel(264, 235)[0];
    assert!((175..=210).contains(&short_body));
    assert!(short_body.abs_diff(short_start_marker) <= 8);
    assert!(short_body.abs_diff(short_end_marker) <= 8);
    assert!(pixel(241, 235)[0] < 10 && pixel(271, 235)[0] < 10);
    eprintln!(
        "sim-engine stroke pixel matrix: 36 mirrored cap/join/width cells + short dual markers, sample_count={sample_count}"
    );
}

pub(super) async fn assert_gpu_large_center_circle(
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    const EXTENT: u32 = 64;
    const ROW_BYTES: u32 = EXTENT * 4;
    let sample_count = preferred_sample_count(adapter, format);
    let PipelineResources {
        pipeline,
        camera_uniform_buffer,
        camera_bind_group,
        ..
    } = create_pipeline(device, format, sample_count);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine large-center circle resolve target"),
        size: wgpu::Extent3d {
            width: EXTENT,
            height: EXTENT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let multisample = (sample_count > 1).then(|| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine large-center circle multisample target"),
            size: wgpu::Extent3d {
                width: EXTENT,
                height: EXTENT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    });
    let multisample_view = multisample
        .as_ref()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

    let center = Vec2::new(1.0e20, 0.0);
    let viewport = LogicalViewport::new(EXTENT as f32, EXTENT as f32).unwrap();
    let camera = Camera2d::new(center, 8.0).unwrap();
    let camera_uniform = CameraUniform::new(camera, viewport).unwrap();
    queue.write_buffer(
        &camera_uniform_buffer,
        0,
        bytemuck::bytes_of(&camera_uniform),
    );
    let mut scene = Scene::new(Color::BLACK).unwrap();
    scene
        .try_circle(
            center,
            1.0,
            ShapeStyle::fill_stroke(Color::WHITE, 2.0, Color::rgb8(255, 0, 0)),
        )
        .unwrap();
    scene
        .try_styled_line(
            Vec2::new(center.x, 1.5),
            Vec2::new(center.x, 3.5),
            crate::StrokeStyle2d::world(
                crate::WorldLength::new(1.0).unwrap(),
                Color::rgb8(0, 255, 0),
            )
            .with_cap(crate::StrokeCap2d::Butt),
        )
        .unwrap();
    let prepared = prepare_scene_resources(device, queue, Arc::new(()), &scene)
        .expect("large-center circle should prepare without degenerating");
    assert_eq!(prepared.tessellation_stats().rendered_command_count(), 2);
    assert_eq!(prepared.tessellation_stats().dropped_command_count(), 0);

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine large-center circle readback"),
        size: u64::from(ROW_BYTES) * u64::from(EXTENT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sim-engine large-center circle encoder"),
    });
    {
        let attachment_view = multisample_view.as_ref().unwrap_or(&target_view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sim-engine large-center circle pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: attachment_view,
                depth_slice: None,
                resolve_target: multisample_view.as_ref().map(|_| &target_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                    store: if multisample_view.is_some() {
                        wgpu::StoreOp::Discard
                    } else {
                        wgpu::StoreOp::Store
                    },
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &camera_bind_group, &[]);
        pass.set_vertex_buffer(0, prepared.vertex_buffer.slice(..));
        for batch in &prepared.draw_batches {
            pass.draw(batch.vertex_range.clone(), 0..1);
        }
    }
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
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(EXTENT),
            },
        },
        wgpu::Extent3d {
            width: EXTENT,
            height: EXTENT,
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("large-center circle submission should complete");
    let bytes =
        crate::renderer::test_support::read_buffer(device, &readback, Duration::from_secs(5));
    let channels = gpu_oracle_channel_indices(format);
    let pixel = |x: usize, y: usize| &bytes[y * ROW_BYTES as usize + x * 4..][..4];
    let center_pixel = pixel(32, 32);
    assert!(
        center_pixel[channels[0]] > 220
            && center_pixel[channels[1]] > 220
            && center_pixel[channels[2]] > 220,
        "camera-relative large-center circle fill did not rasterize: {center_pixel:?}"
    );
    let outside = pixel(48, 32);
    assert!(
        outside[channels[0]] < 10 && outside[channels[1]] < 10 && outside[channels[2]] < 10,
        "large-center circle exceeded its camera-relative bounds: {outside:?}"
    );
    let world_width_line = pixel(32, 12);
    assert!(
        world_width_line[channels[0]] < 10
            && world_width_line[channels[1]] > 220
            && world_width_line[channels[2]] < 10,
        "world-width line collapsed below its large base ULP: {world_width_line:?}"
    );
    let beside_world_width_line = pixel(38, 12);
    assert!(
        beside_world_width_line[channels[0]] < 10
            && beside_world_width_line[channels[1]] < 10
            && beside_world_width_line[channels[2]] < 10,
        "world-width line exceeded its camera-relative width: {beside_world_width_line:?}"
    );
}

pub(super) fn parse_gpu_oracle_surface_format(name: &str) -> Option<wgpu::TextureFormat> {
    match name {
        "Rgba8Unorm" => Some(wgpu::TextureFormat::Rgba8Unorm),
        "Rgba8UnormSrgb" => Some(wgpu::TextureFormat::Rgba8UnormSrgb),
        "Bgra8Unorm" => Some(wgpu::TextureFormat::Bgra8Unorm),
        "Bgra8UnormSrgb" => Some(wgpu::TextureFormat::Bgra8UnormSrgb),
        _ => None,
    }
}

pub(super) fn gpu_oracle_channel_indices(format: wgpu::TextureFormat) -> [usize; 4] {
    match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2, 3],
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0, 3],
        _ => panic!("unsupported byte-channel oracle format: {format:?}"),
    }
}

#[test]
fn gpu_oracle_surface_format_preserves_production_channel_order() {
    assert_eq!(
        parse_gpu_oracle_surface_format("Rgba8UnormSrgb"),
        Some(wgpu::TextureFormat::Rgba8UnormSrgb)
    );
    assert_eq!(
        parse_gpu_oracle_surface_format("Bgra8UnormSrgb"),
        Some(wgpu::TextureFormat::Bgra8UnormSrgb)
    );
    assert_eq!(
        gpu_oracle_channel_indices(wgpu::TextureFormat::Rgba8UnormSrgb),
        [0, 1, 2, 3]
    );
    assert_eq!(
        gpu_oracle_channel_indices(wgpu::TextureFormat::Bgra8UnormSrgb),
        [2, 1, 0, 3]
    );
    assert_eq!(parse_gpu_oracle_surface_format("Rgba16Float"), None);
}
