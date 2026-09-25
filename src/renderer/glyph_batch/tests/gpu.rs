use super::*;
use crate::renderer::{
    GlyphAtlasBudget, GlyphAtlasEntry, GlyphId, GlyphRunBudget, ImageError, ImageRenderer,
    ImageSampling, ImageTexelRect, PositionedGlyph2d, frame, glyph,
};
use std::{sync::Arc, time::Duration};
use wgpu::util::DeviceExt;

pub(in crate::renderer) fn verify_glyph_batches(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    samples: u32,
) {
    let identity = Arc::new(());
    let atlas = make_atlas(device, queue, Arc::clone(&identity));
    let run = make_run(device, queue, Arc::clone(&identity), &atlas);
    let placements = placements(&run);
    let mut batch = create_resources(
        device,
        queue,
        Arc::clone(&identity),
        &atlas.image,
        &placements,
        GlyphBatchBudget::default(),
    )
    .unwrap();
    assert_eq!(batch.glyph_count(), 4);
    assert_eq!(batch.placement_count(), 4);
    assert_eq!(batch.draw_count(), 3, "only adjacent equal clips may merge");
    assert_eq!(
        batch
            .spans
            .iter()
            .map(|span| (span.first, span.end))
            .collect::<Vec<_>>(),
        [(0, 1), (1, 2), (2, 4)]
    );

    for sample_count in std::iter::once(1).chain((samples != 1).then_some(samples)) {
        for physical in [64, 80] {
            let pixels = render(
                device,
                queue,
                &atlas.image,
                &batch,
                format,
                sample_count,
                physical,
            );
            verify_pixels(&pixels, format, physical);
        }
    }

    let buffer = batch.batch.instance_buffer.clone();
    let bytes = batch.recovery_memory_bytes();
    let (same, allocations) = crate::test_allocations::count(|| {
        update_resources(device, queue, &atlas.image, &mut batch, &placements)
    });
    assert_eq!(
        allocations, 0,
        "an unchanged retained glyph snapshot allocated"
    );
    assert_eq!(same.unwrap().instances().uploaded_instance_bytes(), 0);
    assert_eq!(buffer, batch.batch.instance_buffer);
    assert_eq!(bytes, batch.recovery_memory_bytes());

    let mut moved_clip = placements;
    moved_clip[1] = GlyphRunPlacement2d::new(&run, moved_clip[1].placement());
    let report = update_resources(device, queue, &atlas.image, &mut batch, &moved_clip).unwrap();
    assert_eq!(report.instances().uploaded_instance_bytes(), 0);
    assert_eq!(batch.draw_count(), 1);
    assert_eq!(batch.batch.instance_buffer, buffer);
    update_resources(device, queue, &atlas.image, &mut batch, &placements).unwrap();

    // A different atlas on the same renderer is still not interchangeable.
    let foreign_identity = Arc::clone(&identity);
    let foreign_atlas = make_atlas(device, queue, Arc::clone(&foreign_identity));
    let foreign_run = make_run(device, queue, Arc::clone(&foreign_identity), &foreign_atlas);
    let foreign = [GlyphRunPlacement2d::new(
        &foreign_run,
        ImageBatchPlacement::default(),
    )];
    let before = batch.batch.sprites().to_vec();
    let before_spans = batch.spans.clone();
    let (rejection, allocations) = crate::test_allocations::count(|| {
        update_resources(device, queue, &atlas.image, &mut batch, &foreign)
    });
    assert_eq!(allocations, 0);
    assert!(matches!(
        rejection,
        Err(GlyphError::Image(ImageError::RendererMismatch))
    ));
    assert_eq!(batch.batch.sprites(), before);
    assert_eq!(batch.spans, before_spans);

    // Common scan fusion must still validate every source and the exact lazy
    // iterator length before publishing anything, including an unchanged prefix.
    verify_update_scan_rejections(device, queue, &atlas.image, &mut batch);
    let largest = crate::renderer::MAX_PORTABLE_SHADER_VALUE;
    let large_region = LogicalViewportRegion::new(
        LogicalScreenPosition::new(largest * 0.5, 0.0),
        LogicalViewport::new(largest * 0.0625, 32.0).unwrap(),
    )
    .unwrap();
    let large_run = glyph::create_glyph_run_resources(
        device,
        queue,
        Arc::clone(&identity),
        &atlas,
        vec![PositionedGlyph2d::new(GlyphId::new(7), large_region, Color::WHITE).unwrap()],
        GlyphRunBudget::default(),
    )
    .unwrap();
    let invalid = [GlyphRunPlacement2d::new(
        &large_run,
        ImageBatchPlacement::new(LogicalScreenVector::new(largest, 0.0), Color::WHITE).unwrap(),
    )];
    assert_eq!(
        update_resources(device, queue, &atlas.image, &mut batch, &invalid),
        Err(GlyphError::InvalidGlyph)
    );
    assert_eq!(batch.batch.sprites(), before);
    assert_eq!(batch.spans, before_spans);

    // A fresh logical device must reject old resources, then restore exact
    // pixels/placements/clips with the original image's recovery provenance.
    let restored_identity = Arc::new(());
    let mut restored_atlas = make_atlas(
        recovery_device,
        recovery_queue,
        Arc::clone(&restored_identity),
    );
    assert!(matches!(
        restore_resources(
            recovery_device,
            recovery_queue,
            Arc::clone(&restored_identity),
            &restored_atlas.image,
            &batch
        ),
        Err(GlyphError::Image(ImageError::RecoverySourceMismatch))
    ));
    restored_atlas.image.recovery_identity = Arc::clone(&atlas.image.recovery_identity);
    assert!(image::validate_batch_image(&restored_atlas.image, &batch.batch).is_err());
    let restored = restore_resources(
        recovery_device,
        recovery_queue,
        restored_identity,
        &restored_atlas.image,
        &batch,
    )
    .unwrap();
    assert_eq!(restored.batch.sprites(), batch.batch.sprites());
    assert_eq!(restored.spans, batch.spans);
    assert_eq!(restored.placement_count(), batch.placement_count());
    let pixels = render(
        recovery_device,
        recovery_queue,
        &restored_atlas.image,
        &restored,
        format,
        samples,
        80,
    );
    verify_pixels(&pixels, format, 80);

    verify_budgets(device, queue, identity, &atlas, &run);
    eprintln!(
        "Glyph batch oracle: ordered half-alpha placements, fractional clips, HiDPI, MSAA x{samples}, capacity/identity/atomic update and device restoration passed"
    );
}

fn verify_update_scan_rejections(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &crate::renderer::Image2d,
    batch: &mut GlyphBatch2d,
) {
    let before = batch.batch.sprites().to_vec();
    let buffer = batch.batch.instance_buffer.clone();
    let cpu_bytes = batch.recovery_memory_bytes();
    let budget = batch.batch.budget();
    let mut bad_source = before.clone();
    let last = bad_source.last_mut().unwrap();
    *last = ImageSprite2d::new(
        ImageTexelRect::new(1, 0, 1, 1).unwrap(),
        last.destination(),
        last.tint(),
    )
    .unwrap();
    // Too short, too long, zero-claimed-but-not-empty and a late out-of-atlas
    // source with an exactly unchanged destination must all reject atomically.
    for (input, count) in [
        (before.as_slice(), before.len() + 1),
        (before.as_slice(), before.len() - 1),
        (before.as_slice(), 0),
        (bad_source.as_slice(), before.len()),
    ] {
        let (result, allocations) = crate::test_allocations::count(|| {
            image::update_image_batch_resources(
                device,
                queue,
                image,
                &mut batch.batch,
                input.iter().copied(),
                count,
                budget,
            )
        });
        assert_eq!(result, Err(ImageError::InvalidSprite));
        assert_eq!(allocations, 0, "failed validation allocated staging memory");
        assert_eq!(batch.batch.sprites(), before);
        assert_eq!(batch.batch.instance_buffer, buffer);
        assert_eq!(batch.recovery_memory_bytes(), cpu_bytes);
    }
}

fn make_atlas(device: &wgpu::Device, queue: &wgpu::Queue, identity: Arc<()>) -> GlyphAtlas2d {
    glyph::create_glyph_atlas_resources(
        device,
        queue,
        identity,
        1,
        1,
        vec![255; 4],
        vec![GlyphAtlasEntry::new(
            GlyphId::new(7),
            ImageTexelRect::new(0, 0, 1, 1).unwrap(),
        )],
        GlyphAtlasBudget::default(),
    )
    .unwrap()
}

fn make_run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: Arc<()>,
    atlas: &GlyphAtlas2d,
) -> GlyphRun2d {
    let destination = LogicalViewportRegion::new(
        LogicalScreenPosition::new(0.0, 0.0),
        LogicalViewport::new(32.0, 32.0).unwrap(),
    )
    .unwrap();
    glyph::create_glyph_run_resources(
        device,
        queue,
        identity,
        atlas,
        vec![
            PositionedGlyph2d::new(GlyphId::new(7), destination, Color::WHITE.with_alpha(0.5))
                .unwrap(),
        ],
        GlyphRunBudget::default(),
    )
    .unwrap()
}

fn placements(run: &GlyphRun2d) -> [GlyphRunPlacement2d<'_>; 4] {
    let placement = |x, y, color| {
        GlyphRunPlacement2d::new(
            run,
            ImageBatchPlacement::new(LogicalScreenVector::new(x, y), color).unwrap(),
        )
    };
    [
        placement(0.0, 0.0, Color::rgb(1.0, 0.0, 0.0)),
        placement(8.0, 8.0, Color::rgb(0.0, 1.0, 0.0)).with_clip(
            ScreenClipRect::new(
                LogicalScreenPosition::new(12.3, 0.0),
                LogicalScreenPosition::new(27.6, 40.0),
            )
            .unwrap(),
        ),
        placement(16.0, 16.0, Color::rgb(0.0, 0.0, 1.0)),
        placement(0.0, 36.0, Color::rgb(1.0, 1.0, 0.0)),
    ]
}

fn verify_budgets(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: Arc<()>,
    atlas: &GlyphAtlas2d,
    run: &GlyphRun2d,
) {
    let entry = GlyphRunPlacement2d::new(run, ImageBatchPlacement::default());
    let budget = GlyphBatchBudget::new(
        4,
        3,
        3 * GlyphBatchBudget::RETAINED_BYTES_PER_GLYPH
            + 3 * GlyphBatchBudget::RETAINED_BYTES_PER_PLACEMENT,
    )
    .unwrap();
    let mut batch = create_resources(
        device,
        queue,
        Arc::clone(&identity),
        &atlas.image,
        &[entry],
        budget,
    )
    .unwrap();
    let old_cpu = batch.recovery_memory_bytes();
    let report = update_resources(device, queue, &atlas.image, &mut batch, &[entry; 3]).unwrap();
    assert_eq!(batch.glyph_count(), 3);
    assert_eq!(batch.draw_count(), 1);
    assert!(report.instances().uploaded_instance_bytes() > 0);
    assert!(report.retained_bytes() <= budget.max_retained_bytes());
    assert!(report.peak_retained_bytes() >= old_cpu);
    let buffer = batch.batch.instance_buffer.clone();
    let old_sprites = batch.batch.sprites().to_vec();
    let old_spans = batch.spans.clone();
    let (failure, allocations) = crate::test_allocations::count(|| {
        update_resources(device, queue, &atlas.image, &mut batch, &[entry; 4])
    });
    assert_eq!(allocations, 0, "over-budget glyph snapshot allocated");
    assert!(matches!(
        failure,
        Err(GlyphError::EntryBudgetExceeded {
            limit: 3,
            actual: 4
        })
    ));
    assert_eq!(batch.batch.sprites(), old_sprites);
    assert_eq!(batch.spans, old_spans);
    assert_eq!(batch.batch.instance_buffer, buffer);
    update_resources(device, queue, &atlas.image, &mut batch, &[]).unwrap();
    assert_eq!(batch.glyph_count(), 0);
    assert_eq!(batch.draw_count(), 0);
    assert_eq!(batch.batch.instance_buffer, buffer);
    update_resources(device, queue, &atlas.image, &mut batch, &[entry; 3]).unwrap();
    assert_eq!(batch.batch.instance_buffer, buffer);
    assert_eq!(batch.batch.sprites(), old_sprites);

    let empty_run = glyph::create_glyph_run_resources(
        device,
        queue,
        Arc::clone(&identity),
        atlas,
        vec![],
        GlyphRunBudget::default(),
    )
    .unwrap();
    let empty = GlyphRunPlacement2d::new(&empty_run, ImageBatchPlacement::default()).with_clip(
        ScreenClipRect::new(
            LogicalScreenPosition::new(8.0, 8.0),
            LogicalScreenPosition::new(16.0, 16.0),
        )
        .unwrap(),
    );
    update_resources(
        device,
        queue,
        &atlas.image,
        &mut batch,
        &[entry, empty, entry],
    )
    .unwrap();
    assert_eq!(batch.placement_count(), 3);
    assert_eq!(batch.glyph_count(), 2);
    assert_eq!(
        batch.draw_count(),
        1,
        "an empty clipped run must not split a draw"
    );
    let (failure, allocations) = crate::test_allocations::count(|| {
        update_resources(device, queue, &atlas.image, &mut batch, &[empty; 5])
    });
    assert_eq!(allocations, 0);
    assert!(matches!(
        failure,
        Err(GlyphError::EntryBudgetExceeded {
            limit: 4,
            actual: 5
        })
    ));
    assert_eq!(batch.placement_count(), 3);

    // Snapshots outlive source runs and never observe later source mutation.
    let independent = {
        let mut temporary = make_run(device, queue, Arc::clone(&identity), atlas);
        let snapshot = create_resources(
            device,
            queue,
            Arc::clone(&identity),
            &atlas.image,
            &[GlyphRunPlacement2d::new(
                &temporary,
                ImageBatchPlacement::default(),
            )],
            budget,
        )
        .unwrap();
        glyph::update_glyph_run_resources(device, queue, atlas, &mut temporary, &[]).unwrap();
        assert_eq!(temporary.glyph_count(), 0);
        snapshot
    };
    assert_eq!(independent.glyph_count(), 1);
    assert_eq!(
        independent.batch.sprites()[0]
            .destination()
            .viewport()
            .size(),
        crate::Vec2::splat(32.0)
    );
}

#[allow(clippy::too_many_arguments)]
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &crate::renderer::Image2d,
    batch: &GlyphBatch2d,
    format: wgpu::TextureFormat,
    samples: u32,
    physical: u32,
) -> Vec<u8> {
    let size = wgpu::Extent3d {
        width: physical,
        height: physical,
        depth_or_array_layers: 1,
    };
    let texture = |count| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glyph batch oracle target"),
            size,
            mip_level_count: 1,
            sample_count: count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | if count == 1 {
                    wgpu::TextureUsages::COPY_SRC
                } else {
                    wgpu::TextureUsages::empty()
                },
            view_formats: &[],
        })
    };
    let target = texture(1);
    let view = target.create_view(&Default::default());
    let multisampled = (samples > 1).then(|| texture(samples));
    let multisampled_view = multisampled
        .as_ref()
        .map(|target| target.create_view(&Default::default()));
    let target_logical = LogicalViewport::new(64.0, 64.0).unwrap();
    let region = LogicalViewportRegion::new(
        LogicalScreenPosition::new(8.0, 4.0),
        LogicalViewport::new(48.0, 56.0).unwrap(),
    )
    .unwrap();
    let uniform = image::batch_uniform(
        target_logical,
        region.origin().to_vec2(),
        ImageBatchPlacement::default(),
    )
    .unwrap();
    assert!(
        batch
            .batch
            .sprites_are_safe_for_target(region.origin().to_vec2(), uniform.destination)
    );
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("glyph batch oracle uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let renderer = ImageRenderer::new(device, format, samples);
    let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("glyph batch oracle binding"),
        layout: &renderer.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&image.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(renderer.sampler(ImageSampling::Nearest)),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });
    let row_bytes = (physical * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("glyph batch oracle readback"),
        size: u64::from(row_bytes) * u64::from(physical),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("glyph batch production encoding oracle"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: multisampled_view.as_ref().unwrap_or(&view),
                resolve_target: multisampled_view.as_ref().map(|_| &view),
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
        let clip = ScreenClipRect::new(
            LogicalScreenPosition::new(4.0, 4.0),
            LogicalScreenPosition::new(40.0, 52.0),
        )
        .unwrap();
        frame::encode_glyph_batch_for_test(
            &mut pass,
            &renderer,
            &binding,
            batch,
            region,
            Some(clip),
            physical as f32 / 64.0,
            physical,
            physical,
        );
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
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(physical),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    crate::renderer::test_support::read_buffer(device, &readback, Duration::from_secs(15))
}

fn verify_pixels(pixels: &[u8], format: wgpu::TextureFormat, physical: u32) {
    let scale = physical as f32 / 64.0;
    let stride = ((physical * 4).div_ceil(256) * 256) as usize;
    let channels = if matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        [2, 1, 0]
    } else {
        [0, 1, 2]
    };
    let encode = |value: f32| {
        let value = if format.is_srgb() {
            if value <= 0.0031308 {
                value * 12.92
            } else {
                1.055 * value.powf(1.0 / 2.4) - 0.055
            }
        } else {
            value
        };
        (value * 255.0).round() as u8
    };
    // These rectangles, the conservative fractional clip and painter sequence
    // are an independent reference, not reconstructed from batch span metadata.
    let layers = [
        (0.0, 0.0, [1.0, 0.0, 0.0]),
        (8.0, 8.0, [0.0, 1.0, 0.0]),
        (16.0, 16.0, [0.0, 0.0, 1.0]),
        (0.0, 36.0, [1.0, 1.0, 0.0]),
    ];
    let (green_left, green_right) = if physical == 64 { (20, 36) } else { (25, 45) };
    for y in 0..physical {
        for x in 0..physical {
            let logical_x = (x as f32 + 0.5) / scale - 8.0;
            let logical_y = (y as f32 + 0.5) / scale - 4.0;
            let mut expected = [0.0_f32; 3];
            if (4.0..40.0).contains(&logical_x) && (4.0..52.0).contains(&logical_y) {
                for (index, (left, top, tint)) in layers.iter().enumerate() {
                    let in_glyph = (*left..*left + 32.0).contains(&logical_x)
                        && (*top..*top + 32.0).contains(&logical_y);
                    let in_clip =
                        index != 1 || ((green_left..green_right).contains(&x) && logical_y < 40.0);
                    if in_glyph && in_clip {
                        for channel in 0..3 {
                            expected[channel] = tint[channel] * 0.5 + expected[channel] * 0.5;
                        }
                    }
                }
            }
            let offset = y as usize * stride + x as usize * 4;
            for channel in 0..3 {
                let tolerance = if expected[channel] == 0.0 { 0 } else { 3 };
                assert!(
                    pixels[offset + channels[channel]].abs_diff(encode(expected[channel]))
                        <= tolerance,
                    "glyph batch {physical}px pixel({x},{y}) channel{channel}: got {}, expected {}",
                    pixels[offset + channels[channel]],
                    encode(expected[channel])
                );
            }
            assert_eq!(pixels[offset + 3], 255);
        }
    }
}
