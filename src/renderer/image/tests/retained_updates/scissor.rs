//! Integer physical scissor boundaries must mask every sample outside the
//! half-open rectangle, including with MSAA. Keep a full-image oracle: checking
//! only distant pixels would miss a driver's half-pixel leak at left/top edges.
use super::*;

pub(super) async fn verify_scissor_edges(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    production_samples: u32,
) {
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
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    let sprite = ImageSprite2d::new(
        image.full_rect(),
        LogicalViewportRegion::new(LogicalScreenPosition::new(0.0, 0.0), viewport).unwrap(),
        Color::WHITE.with_alpha(0.5),
    )
    .unwrap();
    let batch = create_image_batch_resources(
        device,
        queue,
        identity,
        &image,
        vec![sprite],
        ImageBatchBudget::new(1, batch_retained_bytes(1)).unwrap(),
    )
    .unwrap();
    let clip = ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(24.0, 18.0),
        crate::LogicalScreenVector::new(6.0, 10.0),
    )
    .unwrap();
    for samples in std::iter::once(1).chain((production_samples != 1).then_some(production_samples))
    {
        for (physical, expected_scissor) in [(64_u32, (24, 18, 6, 10)), (80_u32, (30, 22, 8, 13))] {
            let scale = physical as f32 / 64.0;
            let scissor = screen_clip_to_scissor(clip, viewport, scale).unwrap();
            assert_eq!(
                (scissor.x, scissor.y, scissor.width, scissor.height),
                expected_scissor,
                "physical scissor rounding changed at scale {scale}"
            );
            let pixels = render_scissored_batch(
                device, queue, &image, &batch, format, samples, physical, scissor,
            )
            .await;
            let stride = (physical as usize * 4).div_ceil(256) * 256;
            let (red, green, blue) = if matches!(
                format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            ) {
                (2, 1, 0)
            } else {
                (0, 1, 2)
            };
            // Half-alpha sprite * half-alpha placement writes 0.25 linear
            // green over black. Only this color conversion gets a tolerance;
            // an uncovered pixel must remain exactly the clear value.
            let quarter = if format.is_srgb() {
                ((1.055 * 0.25_f32.powf(1.0 / 2.4) - 0.055) * 255.0).round() as u8
            } else {
                (0.25_f32 * 255.0).round() as u8
            };
            for y in 0..physical {
                for x in 0..physical {
                    let inside = x >= scissor.x
                        && x < scissor.x + scissor.width
                        && y >= scissor.y
                        && y < scissor.y + scissor.height;
                    let start = y as usize * stride + x as usize * 4;
                    let actual = [
                        pixels[start + red],
                        pixels[start + green],
                        pixels[start + blue],
                    ];
                    let expected_green = if inside { quarter } else { 0 };
                    let tolerance = if inside { 3 } else { 0 };
                    assert!(
                        actual[0] == 0
                            && actual[1].abs_diff(expected_green) <= tolerance
                            && actual[2] == 0
                            && pixels[start + 3] == 255,
                        "scissor pixel physical=({x}, {y}) logical_center=({}, {}) \
                         target={physical}x{physical} scale={scale} format={format:?} \
                         samples={samples} scissor={scissor:?}: rgba={:?}, \
                         expected=[0, {expected_green}, 0, 255]; green physical row={:?}",
                        (x as f32 + 0.5) / scale,
                        (y as f32 + 0.5) / scale,
                        [actual[0], actual[1], actual[2], pixels[start + 3]],
                        (0..physical as usize)
                            .map(|column| pixels[y as usize * stride + column * 4 + green])
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    assert!(scope.pop().await.is_none());
}

#[allow(clippy::too_many_arguments)]
async fn render_scissored_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &Image2d,
    batch: &ImageBatch2d,
    format: wgpu::TextureFormat,
    samples: u32,
    physical: u32,
    scissor: ScissorRect,
) -> Vec<u8> {
    let renderer = ImageRenderer::new(device, format, samples);
    let size = wgpu::Extent3d {
        width: physical,
        height: physical,
        depth_or_array_layers: 1,
    };
    let descriptor = wgpu::TextureDescriptor {
        label: Some("image batch scissor oracle"),
        size,
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
    let uniform = batch_uniform(
        LogicalViewport::new(64.0, 64.0).unwrap(),
        Vec2::ZERO,
        ImageBatchPlacement::new(
            crate::LogicalScreenVector::default(),
            Color::rgba(0.0, 1.0, 0.0, 0.5),
        )
        .unwrap(),
    )
    .unwrap();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scissor oracle placement uniform"),
        size: std::mem::size_of::<ImageUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&uniform));
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scissor oracle placement binding"),
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
                resource: buffer.as_entire_binding(),
            },
        ],
    });
    let stride = (physical as usize * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("image batch scissor pixels"),
        size: (stride * physical as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        // Exactly one pass and one resolve, matching the composed image path.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("single-pass image scissor oracle"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: multisampled_view.as_ref().unwrap_or(&target_view),
                depth_slice: None,
                resolve_target: multisampled_view.as_ref().map(|_| &target_view),
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
        pass.set_pipeline(&renderer.batch_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, batch.instance_buffer.slice(..));
        pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
        pass.draw(0..6, 0..1);
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
        size,
    );
    queue.submit([encoder.finish()]);
    read_buffer(device, &readback).await
}
