//! Real font coverage must survive atlas UV mapping, tint, DPI and MSAA without
//! stretching the bitmap or multiplying filtered edge coverage by dark padding.
use super::*;
use crate::text::{FontBudget, FontFace, RasterizedGlyph, TextLayoutBudget, TextStyle};

const COLUMNS: usize = 4;
const GLYPHS: usize = 6;
const ROWS: usize = GLYPHS * 8 / COLUMNS;
const CELL: f32 = 64.0;
const PADDING: u32 = 2;
const FONT_BYTES: [&[u8]; 3] = [
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/assets/fonts/DejaVuSans.ttf"
    )),
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/assets/fonts/DejaVuMathTeXGyre.ttf"
    )),
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/assets/fonts/SimEngineJapaneseSubset.otf"
    )),
];
const FONT_GLYPHS: [(usize, &str); GLYPHS] =
    [(0, "A"), (0, "j"), (0, "Ж"), (1, "𝓣"), (1, "𝔂"), (2, "語")];

struct Case {
    raster: usize,
    baseline: [f32; 2],
    tint: Color,
    sampling: ImageSampling,
}

pub(super) async fn verify_font_coverage(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    production_samples: u32,
) {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let fonts = FONT_BYTES
        .map(|bytes| FontFace::from_bytes(bytes.to_vec(), FontBudget::default()).unwrap());
    for scale in [1.0_f32, 1.25, 2.0] {
        let style = TextStyle::new(
            crate::LogicalPixels::new(22.0).unwrap(),
            crate::PhysicalPerLogical::new(scale).unwrap(),
        )
        .unwrap();
        let rasters: Vec<_> = FONT_GLYPHS
            .into_iter()
            .map(|(font_index, letter)| {
                let font = &fonts[font_index];
                let line = font
                    .shape_line(letter, &style, &TextLayoutBudget::default())
                    .unwrap();
                assert_eq!(line.glyphs().len(), 1);
                let raster = font
                    .rasterize_glyph(
                        line.glyphs()[0].glyph_id(),
                        &style,
                        &TextLayoutBudget::default(),
                    )
                    .unwrap();
                assert!(
                    raster
                        .coverage()
                        .iter()
                        .any(|&alpha| alpha > 0 && alpha < 255)
                );
                raster
            })
            .collect();
        assert!(
            rasters[1].bearing_x() < 0,
            "j must exercise a negative bearing"
        );
        let identity = Arc::new(());
        let (image, regions) = coverage_atlas(device, queue, Arc::clone(&identity), &rasters);
        let cases: Vec<_> = (0..COLUMNS * ROWS)
            .map(|index| coverage_case(index, scale))
            .collect();
        let batches: Vec<_> = [ImageSampling::Nearest, ImageSampling::Linear]
            .into_iter()
            .map(|sampling| {
                let sprites: Vec<_> = cases
                    .iter()
                    .filter(|case| case.sampling == sampling)
                    .map(|case| {
                        ImageSprite2d::new(
                            regions[case.raster],
                            crate::renderer::text::glyph_destination(
                                &rasters[case.raster],
                                case.baseline[0],
                                case.baseline[1],
                                scale,
                            )
                            .unwrap(),
                            case.tint,
                        )
                        .unwrap()
                    })
                    .collect();
                create_image_batch_resources(
                    device,
                    queue,
                    Arc::clone(&identity),
                    &image,
                    sprites,
                    ImageBatchBudget::default(),
                )
                .unwrap()
            })
            .collect();
        for samples in
            std::iter::once(1).chain((production_samples != 1).then_some(production_samples))
        {
            let pixels =
                render_coverage(device, queue, &image, &batches, format, samples, scale).await;
            verify_pixels(&pixels, &cases, &rasters, format, samples, scale);
        }
    }
    assert!(scope.pop().await.is_none());
}

fn coverage_case(index: usize, scale: f32) -> Case {
    let phase = (index / GLYPHS) % 2;
    Case {
        raster: index % GLYPHS,
        baseline: [
            (index % COLUMNS) as f32 * CELL + 16.0 + phase as f32 * 0.25 / scale,
            (index / COLUMNS) as f32 * CELL + 44.0 + phase as f32 * 0.375 / scale,
        ],
        tint: if (index / (GLYPHS * 2)).is_multiple_of(2) {
            Color::WHITE
        } else {
            Color::rgba(0.0, 1.0, 0.0, 0.5)
        },
        sampling: if index < GLYPHS * 4 {
            ImageSampling::Nearest
        } else {
            ImageSampling::Linear
        },
    }
}

#[test]
fn font_coverage_matrix_contains_all_glyph_phase_tint_and_sampling_combinations() {
    assert_eq!(COLUMNS * ROWS, GLYPHS * 8);
    let mut combinations = [[false; 8]; GLYPHS];
    for index in 0..COLUMNS * ROWS {
        let case = coverage_case(index, 1.0);
        let phase = usize::from(case.baseline[0].fract() != 0.0);
        let tint = usize::from(case.tint != Color::WHITE);
        let sampling = usize::from(case.sampling == ImageSampling::Linear);
        let combination = phase + 2 * tint + 4 * sampling;
        assert!(!combinations[case.raster][combination]);
        combinations[case.raster][combination] = true;
    }
    assert!(combinations.into_iter().flatten().all(|covered| covered));
}

fn coverage_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: Arc<()>,
    rasters: &[RasterizedGlyph],
) -> (Image2d, Vec<ImageTexelRect>) {
    let width: u32 = rasters
        .iter()
        .map(|raster| raster.width() + 2 * PADDING)
        .sum();
    let height = rasters.iter().map(|raster| raster.height()).max().unwrap() + 2 * PADDING;
    let mut pixels = vec![255_u8; width as usize * height as usize * 4];
    for pixel in pixels.chunks_exact_mut(4) {
        pixel[3] = 0;
    }
    let mut left = 0;
    let mut regions = Vec::new();
    for raster in rasters {
        let padded_width = raster.width() + 2 * PADDING;
        let padded_height = raster.height() + 2 * PADDING;
        let padded = crate::renderer::text::padded_rgba(raster).unwrap();
        assert_eq!(
            padded.len(),
            padded_width as usize * padded_height as usize * 4
        );
        for y in 0..padded_height {
            let source = y as usize * padded_width as usize * 4;
            let target = (y * width + left) as usize * 4;
            let bytes = padded_width as usize * 4;
            pixels[target..target + bytes].copy_from_slice(&padded[source..source + bytes]);
        }
        regions.push(ImageTexelRect::new(left, 0, padded_width, padded_height).unwrap());
        left += padded_width;
    }
    let image = create_image_resources(
        device,
        queue,
        identity,
        width,
        height,
        pixels,
        ImageBudget::default(),
    )
    .unwrap();
    (image, regions)
}

fn verify_pixels(
    pixels: &[u8],
    cases: &[Case],
    rasters: &[RasterizedGlyph],
    format: wgpu::TextureFormat,
    samples: u32,
    scale: f32,
) {
    let physical_width = (COLUMNS as f32 * CELL * scale) as usize;
    let physical_height = (ROWS as f32 * CELL * scale) as usize;
    let stride = (physical_width * 4).div_ceil(256) * 256;
    let physical_cell = (CELL * scale) as usize;
    let channels = if matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        [2, 1, 0, 3]
    } else {
        [0, 1, 2, 3]
    };
    let mut partial_pixels = 0;
    let mut covered_cases = [false; COLUMNS * ROWS];
    for y in 0..physical_height {
        for x in 0..physical_width {
            let case_index = y / physical_cell * COLUMNS + x / physical_cell;
            let case = &cases[case_index];
            let raster = &rasters[case.raster];
            let local = [
                x as f64 + 0.5
                    - f64::from(case.baseline[0]) * f64::from(scale)
                    - f64::from(raster.bearing_x()),
                y as f64 + 0.5
                    - f64::from(case.baseline[1]) * f64::from(scale)
                    - f64::from(raster.bearing_y()),
            ];
            let coverage = sample_coverage(raster, local, case.sampling);
            let tint = case.tint.to_array();
            let alpha = coverage * f64::from(tint[3]);
            partial_pixels += usize::from(alpha > 0.0 && alpha < 1.0);
            covered_cases[case_index] |= alpha > 0.0;
            let expected = [
                encode_color(alpha * f64::from(tint[0]), format),
                encode_color(alpha * f64::from(tint[1]), format),
                encode_color(alpha * f64::from(tint[2]), format),
                (alpha * 255.0).round() as u8,
            ];
            let offset = y * stride + x * 4;
            let actual = channels.map(|channel| pixels[offset + channel]);
            // Fixed-function UNORM blending may round one linear 8-bit unit
            // before sRGB encoding. Near zero that is several encoded levels,
            // so an encoded-value epsilon would have no consistent meaning.
            // Zero source coverage and zero tint channels remain exact zero.
            assert!(
                (0..3).all(|channel| {
                    color_matches_linear_blend(
                        actual[channel],
                        alpha * f64::from(tint[channel]),
                        format,
                    )
                }) && actual[3].abs_diff(expected[3]) <= u8::from(alpha != 0.0),
                "font coverage case={case_index} glyph={:?} pixel=({x},{y}) local={local:?} \
                 scale={scale} samples={samples} sampling={:?} format={format:?}: \
                 actual={actual:?}, expected={expected:?}",
                FONT_GLYPHS[case.raster].1,
                case.sampling,
            );
        }
    }
    assert!(
        partial_pixels > 100,
        "oracle must inspect nontrivial filtered coverage"
    );
    assert!(covered_cases.into_iter().all(|covered| covered));
}

fn sample_coverage(raster: &RasterizedGlyph, local: [f64; 2], sampling: ImageSampling) -> f64 {
    let at = |x: i64, y: i64| {
        if x < 0 || y < 0 || x >= i64::from(raster.width()) || y >= i64::from(raster.height()) {
            0.0
        } else {
            f64::from(raster.coverage()[y as usize * raster.width() as usize + x as usize]) / 255.0
        }
    };
    if sampling == ImageSampling::Nearest {
        return at(local[0].floor() as i64, local[1].floor() as i64);
    }
    let left = (local[0] - 0.5).floor();
    let top = (local[1] - 0.5).floor();
    let horizontal = local[0] - 0.5 - left;
    let vertical = local[1] - 0.5 - top;
    let x = left as i64;
    let y = top as i64;
    (1.0 - vertical) * ((1.0 - horizontal) * at(x, y) + horizontal * at(x + 1, y))
        + vertical * ((1.0 - horizontal) * at(x, y + 1) + horizontal * at(x + 1, y + 1))
}

fn encode_color(linear: f64, format: wgpu::TextureFormat) -> u8 {
    let encoded = if !format.is_srgb() {
        linear
    } else if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

fn color_matches_linear_blend(actual: u8, expected: f64, format: wgpu::TextureFormat) -> bool {
    if expected == 0.0 {
        return actual == 0;
    }
    let lower = encode_color((expected - 1.0 / 255.0).max(0.0), format).saturating_sub(1);
    let upper = encode_color((expected + 1.0 / 255.0).min(1.0), format).saturating_add(1);
    (lower..=upper).contains(&actual)
}

#[test]
fn coverage_oracle_bounds_linear_quantization_without_accepting_padding_leaks() {
    let format = wgpu::TextureFormat::Bgra8UnormSrgb;
    let expected_linear = 1.5 / 255.0;
    assert_eq!(encode_color(expected_linear, format), 18);
    assert!(color_matches_linear_blend(13, expected_linear, format));
    assert!(!color_matches_linear_blend(40, expected_linear, format));
    assert!(color_matches_linear_blend(0, 0.0, format));
    assert!(!color_matches_linear_blend(1, 0.0, format));
    assert!(!color_matches_linear_blend(0, 0.5, format));
}

#[allow(clippy::too_many_arguments)]
async fn render_coverage(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &Image2d,
    batches: &[ImageBatch2d],
    format: wgpu::TextureFormat,
    samples: u32,
    scale: f32,
) -> Vec<u8> {
    let renderer = ImageRenderer::new(device, format, samples);
    let size = wgpu::Extent3d {
        width: (COLUMNS as f32 * CELL * scale) as u32,
        height: (ROWS as f32 * CELL * scale) as u32,
        depth_or_array_layers: 1,
    };
    let descriptor = wgpu::TextureDescriptor {
        label: Some("font coverage oracle"),
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
        LogicalViewport::new(COLUMNS as f32 * CELL, ROWS as f32 * CELL).unwrap(),
        Vec2::ZERO,
        ImageBatchPlacement::default(),
    )
    .unwrap();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("font coverage uniform"),
        size: std::mem::size_of::<ImageUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&uniform));
    let bindings: Vec<_> = [ImageSampling::Nearest, ImageSampling::Linear]
        .into_iter()
        .map(|sampling| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("font coverage sampling binding"),
                layout: &renderer.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&image.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(renderer.sampler(sampling)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            })
        })
        .collect();
    let stride = (size.width as usize * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("font coverage pixels"),
        size: (stride * size.height as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("font coverage single-pass oracle"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: multisampled_view.as_ref().unwrap_or(&target_view),
                depth_slice: None,
                resolve_target: multisampled_view.as_ref().map(|_| &target_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&renderer.batch_pipeline);
        for (batch, binding) in batches.iter().zip(&bindings) {
            pass.set_bind_group(0, binding, &[]);
            pass.set_vertex_buffer(0, batch.instance_buffer.slice(..));
            pass.draw(0..6, 0..batch.sprites().len() as u32);
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
                bytes_per_row: Some(stride as u32),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    read_buffer(device, &readback).await
}
