use super::*;

mod red_team;
mod scissor;
#[cfg(feature = "text")]
mod text;

fn destination(x: f32, y: f32) -> LogicalViewportRegion {
    LogicalViewportRegion::new(
        LogicalScreenPosition::new(x, y),
        LogicalViewport::new(12.0, 12.0).unwrap(),
    )
    .unwrap()
}

#[test]
fn placement_rejects_nonportable_offsets_and_non_normalized_tint() {
    for value in [f32::NAN, f32::INFINITY, f32::MAX, f32::from_bits(1)] {
        assert_eq!(
            ImageBatchPlacement::new(crate::LogicalScreenVector::new(value, 0.0), Color::WHITE),
            Err(ImageError::InvalidPlacement)
        );
    }
    assert_eq!(
        ImageBatchPlacement::new(
            crate::LogicalScreenVector::default(),
            Color::rgb(2.0, 0.0, 0.0)
        ),
        Err(ImageError::InvalidPlacement)
    );
    let placement = ImageBatchPlacement::new(
        crate::LogicalScreenVector::new(-8.0, 3.0),
        Color::WHITE.with_alpha(0.5),
    )
    .unwrap();
    let uniform = batch_uniform(
        LogicalViewport::new(80.0, 64.0).unwrap(),
        Vec2::new(10.0, 20.0),
        placement,
    )
    .unwrap();
    assert_eq!(uniform.uv_rect, [2.0, 23.0, 0.0, 0.0]);
    assert_eq!(uniform.tint[3], 0.5);
}

pub(in crate::renderer) async fn verify_retained_ui_updates(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    sample_count: u32,
) {
    scissor::verify_scissor_edges(device, queue, format, sample_count).await;
    #[cfg(feature = "text")]
    text::verify_font_coverage(device, queue, format, sample_count).await;
    proof::verify_revision_invalidation(device, queue, recovery_device, recovery_queue);
    red_team::verify_exact_update_budgets(device, queue, format, sample_count).await;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let identity = Arc::new(());
    let image = create_image_resources(
        device,
        queue,
        Arc::clone(&identity),
        1,
        1,
        vec![255; 4],
        ImageBudget::default(),
    )
    .unwrap();
    let sprite =
        ImageSprite2d::new(image.full_rect(), destination(1.0, 1.0), Color::WHITE).unwrap();
    let budget = ImageBatchBudget::new(4, batch_retained_bytes(4)).unwrap();
    let mut batch = create_image_batch_resources(
        device,
        queue,
        Arc::clone(&identity),
        &image,
        vec![sprite; 2],
        budget,
    )
    .unwrap();
    let original_buffer = batch.instance_buffer.clone();
    let cpu_pointer = batch.sprites.as_ptr();
    let staging_pointer = batch.instances.as_ptr();
    let unchanged = update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        [sprite; 2].into_iter(),
        2,
        budget,
    )
    .unwrap();
    assert_eq!(unchanged.uploaded_instance_bytes(), 0);
    let moved = ImageSprite2d::new(
        image.full_rect(),
        destination(-3.0, 7.0),
        Color::WHITE.with_alpha(0.5),
    )
    .unwrap();
    let changed = update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        [moved; 2].into_iter(),
        2,
        budget,
    )
    .unwrap();
    assert_eq!(
        changed.uploaded_instance_bytes(),
        2 * std::mem::size_of::<ImageInstance>()
    );
    assert!(!changed.replaced_instance_buffer());
    assert_eq!(original_buffer, batch.instance_buffer);
    assert_eq!(cpu_pointer, batch.sprites.as_ptr());
    assert_eq!(staging_pointer, batch.instances.as_ptr());
    let invalid = ImageSprite2d::new(
        ImageTexelRect::new(1, 0, 1, 1).unwrap(),
        destination(0.0, 0.0),
        Color::WHITE,
    )
    .unwrap();
    assert_eq!(
        update_image_batch_resources(
            device,
            queue,
            &image,
            &mut batch,
            [invalid].into_iter(),
            1,
            budget
        ),
        Err(ImageError::InvalidSprite)
    );
    assert_eq!(batch.sprites(), &[moved; 2]);
    assert_eq!(original_buffer, batch.instance_buffer);
    assert!(
        update_image_batch_resources(
            device,
            queue,
            &image,
            &mut batch,
            [sprite; 5].into_iter(),
            5,
            budget
        )
        .is_err()
    );
    assert_eq!(batch.sprites(), &[moved; 2]);
    let grown = update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        [sprite; 4].into_iter(),
        4,
        budget,
    )
    .unwrap();
    assert!(grown.replaced_instance_buffer());
    assert_eq!(grown.gpu_capacity(), 4);
    assert_eq!(grown.retained_bytes(), batch_retained_bytes(4));
    assert!(grown.peak_retained_bytes() <= 2 * budget.max_retained_bytes());
    let grown_buffer = batch.instance_buffer.clone();
    let empty = update_image_batch_resources(
        device,
        queue,
        &image,
        &mut batch,
        std::iter::empty(),
        0,
        budget,
    )
    .unwrap();
    assert_eq!(empty.uploaded_instance_bytes(), 0);
    assert_eq!(empty.retained_capacity(), 4);
    assert_eq!(grown_buffer, batch.instance_buffer);
    assert!(batch.sprites().is_empty());
    let foreign = create_image_resources(
        device,
        queue,
        Arc::clone(&identity),
        1,
        1,
        vec![255; 4],
        ImageBudget::default(),
    )
    .unwrap();
    assert_eq!(
        update_image_batch_resources(
            device,
            queue,
            &foreign,
            &mut batch,
            [sprite].into_iter(),
            1,
            budget
        ),
        Err(ImageError::RendererMismatch)
    );

    let entries = vec![GlyphAtlasEntry::new(GlyphId::new(1), image.full_rect())];
    let atlas = glyph::create_glyph_atlas_resources(
        device,
        queue,
        identity,
        1,
        1,
        vec![255; 4],
        entries.clone(),
        GlyphAtlasBudget::default(),
    )
    .unwrap();
    let glyph = PositionedGlyph2d::new(
        GlyphId::new(1),
        destination(-2.0, -2.0),
        Color::WHITE.with_alpha(0.5),
    )
    .unwrap();
    let mut run = glyph::create_glyph_run_resources(
        device,
        queue,
        Arc::clone(&atlas.image.renderer_identity),
        &atlas,
        vec![glyph; 2],
        GlyphRunBudget::default(),
    )
    .unwrap();
    let run_buffer = run.batch.instance_buffer.clone();
    let unchanged =
        glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[glyph; 2]).unwrap();
    assert_eq!(unchanged.instances().uploaded_instance_bytes(), 0);
    let updated =
        glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[glyph]).unwrap();
    assert!(!updated.instances().replaced_instance_buffer());
    assert_eq!(run_buffer, run.batch.instance_buffer);
    let missing =
        PositionedGlyph2d::new(GlyphId::new(99), destination(0.0, 0.0), Color::WHITE).unwrap();
    assert_eq!(
        glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[glyph, missing]),
        Err(GlyphError::MissingGlyph {
            glyph: GlyphId::new(99),
            index: 1
        })
    );
    assert_eq!(run.glyphs(), &[glyph]);

    for physical in [64_u32, 80_u32] {
        let (readback, stride) = render_placements(
            device,
            queue,
            &atlas.image,
            &run.batch,
            format,
            sample_count,
            physical,
        );
        // Submit another revision before waiting for the previous render. Its
        // upload must not change pixels captured by the already-submitted frame.
        let opaque =
            PositionedGlyph2d::new(glyph.glyph(), glyph.destination(), Color::WHITE).unwrap();
        glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[opaque]).unwrap();
        let bytes = read_buffer(device, &readback).await;
        let scale = physical as f32 / 64.0;
        let (red, green, blue) = if matches!(
            format,
            wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm
        ) {
            (2, 1, 0)
        } else {
            (0, 1, 2)
        };
        let pixel = |x: f32, y: f32| {
            let start = (y * scale) as usize * stride + (x * scale) as usize * 4;
            [
                bytes[start + red],
                bytes[start + green],
                bytes[start + blue],
            ]
        };
        let encoded = |linear: f32| -> u8 {
            let value = if format.is_srgb() {
                if linear <= 0.0031308 {
                    linear * 12.92
                } else {
                    1.055 * linear.powf(1.0 / 2.4) - 0.055
                }
            } else {
                linear
            };
            (value * 255.0).round() as u8
        };
        let half = encoded(0.5);
        let quarter = encoded(0.25);
        let near = |x: f32, y: f32, expected: [u8; 3]| {
            let actual = pixel(x, y);
            let physical_x = (x * scale) as usize;
            let physical_y = (y * scale) as usize;
            assert!(
                actual
                    .into_iter()
                    .zip(expected)
                    .all(|(a, b)| a.abs_diff(b) <= 3),
                "placement pixel logical=({x}, {y}) physical=({physical_x}, {physical_y}) \
                 target={physical}x{physical} scale={scale} format={format:?} samples={sample_count}: \
                 {actual:?} != {expected:?}; green physical row={:?}",
                (0..physical as usize)
                    .map(|column| bytes[physical_y * stride + column * 4 + green])
                    .collect::<Vec<_>>()
            )
        };
        near(6.0, 6.0, [half, 0, 0]);
        near(12.0, 12.0, [quarter, 0, half]);
        near(25.0, 6.0, [0, quarter, 0]);
        near(23.0, 6.0, [0, 0, 0]);
        near(31.0, 6.0, [0, 0, 0]);
        near(2.0, 26.0, [0, half, half]);
        near(63.0, 26.0, [0, 0, half]);
        near(44.0, 2.0, [half, 0, half]);
        near(44.0, 63.0, [half, half, 0]);
        near(44.0, 44.0, [0, 0, 0]);
        glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[glyph]).unwrap();
    }
    let recovery_identity = Arc::new(());
    let restored_atlas = glyph::create_glyph_atlas_resources(
        recovery_device,
        recovery_queue,
        Arc::clone(&recovery_identity),
        1,
        1,
        vec![255; 4],
        entries,
        atlas.budget(),
    )
    .unwrap();
    let restored_run = glyph::create_glyph_run_resources(
        recovery_device,
        recovery_queue,
        recovery_identity,
        &restored_atlas,
        run.glyphs().to_vec(),
        run.budget(),
    )
    .unwrap();
    assert_eq!(restored_run.glyphs(), run.glyphs());
    let (restored_pixels, _) = render_placements(
        recovery_device,
        recovery_queue,
        &restored_atlas.image,
        &restored_run.batch,
        format,
        sample_count,
        64,
    );
    let (original_pixels, _) = render_placements(
        device,
        queue,
        &atlas.image,
        &run.batch,
        format,
        sample_count,
        64,
    );
    assert_eq!(
        read_buffer(recovery_device, &restored_pixels).await,
        read_buffer(device, &original_pixels).await
    );
    glyph::update_glyph_run_resources(device, queue, &atlas, &mut run, &[]).unwrap();
    assert!(run.bounds().is_empty());
    assert_eq!(run_buffer, run.batch.instance_buffer);
    assert!(scope.pop().await.is_none());
}

#[allow(clippy::too_many_arguments)]
fn render_placements(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &Image2d,
    batch: &ImageBatch2d,
    format: wgpu::TextureFormat,
    samples: u32,
    physical: u32,
) -> (wgpu::Buffer, usize) {
    let renderer = ImageRenderer::new(device, format, samples);
    let descriptor = wgpu::TextureDescriptor {
        label: Some("retained UI placement oracle"),
        size: wgpu::Extent3d {
            width: physical,
            height: physical,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    let target = device.create_texture(&descriptor);
    let target_view = target.create_view(&Default::default());
    let multisampled = (samples > 1).then(|| {
        device.create_texture(&wgpu::TextureDescriptor {
            sample_count: samples,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            ..descriptor
        })
    });
    let multisampled_view = multisampled
        .as_ref()
        .map(|texture| texture.create_view(&Default::default()));
    let stride = (physical as usize * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("UI placement readback"),
        size: (stride * physical as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let draws = [
        (6.0, 6.0, Color::rgb(1.0, 0.0, 0.0), None),
        (12.0, 12.0, Color::rgb(0.0, 0.0, 1.0), None),
        (
            24.0,
            6.0,
            Color::rgba(0.0, 1.0, 0.0, 0.5),
            Some((24.0, 0.0, 6.0, 64.0)),
        ),
        (-4.0, 24.0, Color::rgb(0.0, 1.0, 1.0), None),
        (64.0, 24.0, Color::rgb(0.0, 0.0, 1.0), None),
        (42.0, -4.0, Color::rgb(1.0, 0.0, 1.0), None),
        (42.0, 64.0, Color::rgb(1.0, 1.0, 0.0), None),
        (42.0, 42.0, Color::WHITE, Some((100.0, 100.0, 1.0, 1.0))),
    ];
    let mut encoder = device.create_command_encoder(&Default::default());
    for (index, (x, y, tint, clip)) in draws.into_iter().enumerate() {
        let placement =
            ImageBatchPlacement::new(crate::LogicalScreenVector::new(x, y), tint).unwrap();
        let uniform = batch_uniform(
            LogicalViewport::new(64.0, 64.0).unwrap(),
            Vec2::ZERO,
            placement,
        )
        .unwrap();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<ImageUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &renderer.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        renderer.sampler(ImageSampling::Nearest),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: multisampled_view.as_ref().unwrap_or(&target_view),
                depth_slice: None,
                resolve_target: multisampled_view.as_ref().map(|_| &target_view),
                ops: wgpu::Operations {
                    load: if index == 0 {
                        wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let Some((x, y, width, height)) = clip {
            let clip = ScreenClipRect::from_min_size(
                LogicalScreenPosition::new(x, y),
                crate::LogicalScreenVector::new(width, height),
            )
            .unwrap();
            let Some(scissor) = screen_clip_to_scissor(
                clip,
                LogicalViewport::new(64.0, 64.0).unwrap(),
                physical as f32 / 64.0,
            ) else {
                continue;
            };
            pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
        }
        pass.set_pipeline(&renderer.batch_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, batch.instance_buffer.slice(..));
        pass.draw(0..6, 0..batch.sprite_count() as u32);
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
                bytes_per_row: Some(stride as u32),
                rows_per_image: Some(physical),
            },
        },
        descriptor.size,
    );
    queue.submit([encoder.finish()]);
    (readback, stride)
}

async fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
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
    let bytes = buffer.slice(..).get_mapped_range().unwrap().to_vec();
    buffer.unmap();
    bytes
}
