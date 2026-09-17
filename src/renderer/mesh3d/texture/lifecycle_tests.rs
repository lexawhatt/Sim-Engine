//! Independent mip, addressing and atomic-update oracles for production 3D draws.

use super::*;

const SIZE: u32 = 64;
const RECT: [f32; 4] = [4.0, 4.0, 60.0, 60.0];
const QUADRANTS: [[u8; 4]; 4] = [
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [0, 0, 255, 255],
    [255, 255, 0, 255],
];

fn linear(byte: u8) -> f64 {
    let value = f64::from(byte) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn encoded(value: f64) -> u8 {
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

fn options(mipmaps: bool) -> Texture3dOptions {
    Texture3dOptions::new()
        .with_alpha_preservation(true)
        .with_mipmaps(if mipmaps {
            TextureMipmaps3d::Generate
        } else {
            TextureMipmaps3d::None
        })
}

fn camera() -> Camera3d {
    Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(4.0)).unwrap(),
    )
    .unwrap()
}

fn quad(rect: [f32; 4], coordinates: [[f32; 2]; 4]) -> Mesh3d {
    let [left, top, right, bottom] = rect;
    Mesh3d::textured(
        [[left, top], [right, top], [right, bottom], [left, bottom]]
            .map(|[x, y]| Vec3::new(x / 32.0 - 1.0, 1.0 - y / 32.0, 0.0).unwrap())
            .to_vec(),
        coordinates
            .map(|[u, v]| TextureCoordinate2d::new(u, v).unwrap())
            .to_vec(),
        vec![0, 1, 2, 0, 2, 3],
        Vec::new(),
    )
    .unwrap()
}

fn full_quad(rect: [f32; 4]) -> Mesh3d {
    quad(rect, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
}

struct Fixture<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    identity: Arc<()>,
    renderer: Mesh3dRenderer,
    target: RenderTarget3d,
    format: wgpu::TextureFormat,
}

impl<'a> Fixture<'a> {
    fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let identity = Arc::new(());
        Self {
            device,
            queue,
            target: surface::test_target(device, &identity, format, SIZE, SIZE),
            renderer: Mesh3dRenderer::new(device, format),
            identity,
            format,
        }
    }

    fn texture(&self, width: u32, height: u32, pixels: Vec<u8>, mipmaps: bool) -> Texture3d {
        create_texture_with_options(
            self.device,
            self.queue,
            &self.identity,
            &self.renderer.textures.layout,
            width,
            height,
            pixels,
            ImageBudget::default(),
            options(mipmaps),
        )
        .unwrap()
    }

    fn upload(&self, source: Mesh3d, material: &TextureMaterial3d) -> RetainedMesh3d {
        let mesh =
            create_retained_mesh(self.device, self.queue, self.identity.clone(), source).unwrap();
        attach_material(&self.identity, &mesh, material).unwrap()
    }

    fn render_scene(&mut self, scene: &Scene3d, policy: SurfaceRasterization3d) -> Vec<u8> {
        self.renderer
            .render_scene3d(
                self.device,
                self.queue,
                &self.identity,
                &self.target,
                scene,
                camera(),
                Mesh3dRenderBudget::default().with_surface_policy(policy),
            )
            .unwrap();
        surface::test_read_pixels(self.device, self.queue, &self.target)
    }

    fn render(
        &mut self,
        sources: &[Mesh3d],
        material: &TextureMaterial3d,
        style: SurfaceStyle3d,
        policy: SurfaceRasterization3d,
    ) -> Vec<u8> {
        let mut scene = Scene3d::with_alpha_background(Color::TRANSPARENT).unwrap();
        for source in sources {
            let mesh = self.upload(source.clone(), material);
            scene
                .try_push(&mesh, Transform3d::IDENTITY, MeshStyle3d::surface(style))
                .unwrap();
        }
        self.render_scene(&scene, policy)
    }

    fn expect(&self, pixels: &[u8], x: u32, y: u32, rgba: [f64; 4]) {
        let channels = match self.format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0, 3],
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2, 3],
            other => panic!("unsupported texture oracle target {other:?}"),
        };
        let offset = ((y * SIZE + x) * 4) as usize;
        for channel in 0..4 {
            let expected = if channel < 3 && self.format.is_srgb() {
                encoded(rgba[channel])
            } else {
                (rgba[channel] * 255.0).round() as u8
            };
            let actual = pixels[offset + channels[channel]];
            assert!(
                actual.abs_diff(expected) <= 2,
                "texture pixel ({x},{y}), channel {channel}: {actual} != {expected}, linear {rgba:?}"
            );
        }
    }

    fn expect_uniform(&self, pixels: &[u8], rect: [f32; 4], rgba: [f64; 4]) {
        let mut samples = 0;
        for y in (rect[1] as u32 + 1)..(rect[3] as u32 - 1) {
            for x in (rect[0] as u32 + 1)..(rect[2] as u32 - 1) {
                self.expect(pixels, x, y, rgba);
                samples += 1;
            }
        }
        assert!(samples >= 4);
    }

    fn levels(&self, texture: &Texture3d) -> Vec<Vec<u8>> {
        (0..texture.mip_level_count())
            .map(|level| read_level(self.device, self.queue, texture, level))
            .collect()
    }
}

fn read_level(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &Texture3d,
    level: u32,
) -> Vec<u8> {
    let (width, height) = texture.mip_level_size(level).unwrap();
    let stride = (width * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("independent full-chain mip readback"),
        size: u64::from(stride * height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture.storage.texture,
            mip_level: level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
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
    mapped
        .chunks_exact(stride as usize)
        .flat_map(|row| row[..width as usize * 4].iter().copied())
        .collect()
}

fn assert_chain(fixture: &Fixture<'_>, texture: &Texture3d, expected: &[Vec<u8>]) {
    assert_eq!(texture.mip_level_count() as usize, expected.len());
    assert_eq!(
        texture.gpu_allocation_bytes(),
        expected.iter().map(Vec::len).sum::<usize>()
    );
    let gpu = fixture.levels(texture);
    for (level, expected) in expected.iter().enumerate() {
        assert_eq!(
            texture.mip_level_pixels(level as u32).unwrap(),
            expected,
            "CPU mip {level}"
        );
        assert_eq!(&gpu[level], expected, "GPU mip {level}");
    }
}

pub(in crate::renderer::mesh3d) fn assert_gpu_texture_lifecycle(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let mut fixture = Fixture::new(device, queue, format);
    assert_isolated_mips(&mut fixture);
    assert_alpha_and_npot_mips(&mut fixture);
    assert_repeated_coordinates(&mut fixture);
    assert_region_updates(&mut fixture);
    assert_scene_updates(&mut fixture);
    assert_scene_policy_and_budget_failures(&mut fixture);
}

fn assert_isolated_mips(fixture: &mut Fixture<'_>) {
    let parent_pixels: Vec<u8> = (0..16)
        .flat_map(|_| {
            (0..32).flat_map(|x| {
                if x < 16 {
                    [255, 0, 0, 255]
                } else {
                    [0, 255, 0, 255]
                }
            })
        })
        .collect();
    let atlas = fixture.texture(32, 16, parent_pixels, false);
    let region = ImageTexelRect::new(16, 0, 16, 16).unwrap();
    let tile = crop_texture_tile(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        &atlas,
        region,
        ImageBudget::default(),
        options(true),
    )
    .unwrap();
    let expected: Vec<Vec<u8>> = [16, 8, 4, 2, 1]
        .map(|side| [0, 255, 0, 255].repeat(side * side))
        .to_vec();
    assert_chain(fixture, &tile, &expected);
    assert_eq!(
        tile.region_coordinates(ImageTexelRect::new(0, 0, 16, 16).unwrap()),
        Err(Texture3dError::MipmappedAtlasRegion)
    );
    for sampling in [ImageSampling::Nearest, ImageSampling::Linear] {
        for side in [48.0, 16.0, 4.0] {
            let rect = [
                32.0 - side / 2.0,
                32.0 - side / 2.0,
                32.0 + side / 2.0,
                32.0 + side / 2.0,
            ];
            for repeats in [1.0, 8.0, 64.0] {
                let material = TextureMaterial3d::with_alpha(&tile, sampling, Color::WHITE)
                    .unwrap()
                    .with_address_mode(TextureAddressMode3d::Repeat)
                    .with_uv_transform(
                        TextureUvTransform3d::new(Vec2::splat(repeats), Vec2::new(-0.25, 0.25))
                            .unwrap(),
                    );
                let pixels = fixture.render(
                    &[full_quad(rect)],
                    &material,
                    SurfaceStyle3d::opaque(Color::WHITE).unwrap(),
                    SurfaceRasterization3d::Native,
                );
                fixture.expect_uniform(&pixels, rect, [0.0, 1.0, 0.0, 1.0]);
            }
        }
    }
    let checker: Vec<u8> = (0..16)
        .flat_map(|y| {
            (0..16).flat_map(move |x| {
                let value = if (x + y) % 2 == 0 { 0 } else { 255 };
                [value, value, value, 255]
            })
        })
        .collect();
    let checker_texture = fixture.texture(16, 16, checker.clone(), true);
    let mut expected = vec![checker];
    expected.extend([8, 4, 2, 1].map(|side| [188, 188, 188, 255].repeat(side * side)));
    assert_chain(fixture, &checker_texture, &expected);
    for sampling in [ImageSampling::Nearest, ImageSampling::Linear] {
        let material = TextureMaterial3d::with_alpha(&checker_texture, sampling, Color::WHITE)
            .unwrap()
            .with_address_mode(TextureAddressMode3d::Repeat)
            .with_uv_transform(TextureUvTransform3d::new(Vec2::splat(64.0), Vec2::ZERO).unwrap());
        let pixels = fixture.render(
            &[full_quad(RECT)],
            &material,
            SurfaceStyle3d::opaque(Color::WHITE).unwrap(),
            SurfaceRasterization3d::Native,
        );
        fixture.expect_uniform(&pixels, RECT, [linear(188), linear(188), linear(188), 1.0]);
    }
    // A fract-before-derivatives implementation aliases detailed lower mips at
    // these repeat rates instead of choosing the last, constant-color level.
    let large_quadrants: Vec<u8> = (0..16)
        .flat_map(|y| {
            (0..16).flat_map(move |x| QUADRANTS[usize::from(x >= 8) + 2 * usize::from(y >= 8)])
        })
        .collect();
    let texture = fixture.texture(16, 16, large_quadrants, true);
    assert_eq!(
        read_level(fixture.device, fixture.queue, &texture, 4),
        [188, 188, 137, 255]
    );
    for repeats in [56.0_f32, 64.0, -56.0, -64.0] {
        let material = TextureMaterial3d::with_alpha(&texture, ImageSampling::Linear, Color::WHITE)
            .unwrap()
            .with_address_mode(TextureAddressMode3d::Repeat)
            .with_uv_transform(
                TextureUvTransform3d::new(
                    Vec2::new(repeats, repeats.abs()),
                    Vec2::new(0.25, -0.25),
                )
                .unwrap(),
            );
        let pixels = fixture.render(
            &[full_quad(RECT)],
            &material,
            SurfaceStyle3d::opaque(Color::WHITE).unwrap(),
            SurfaceRasterization3d::Native,
        );
        fixture.expect_uniform(&pixels, RECT, [linear(188), linear(188), linear(137), 1.0]);
    }
    let bytes = tile.gpu_allocation_bytes();
    assert!(matches!(
        crop_texture_tile(
            fixture.device,
            fixture.queue,
            &fixture.identity,
            &fixture.renderer.textures.layout,
            &atlas,
            region,
            ImageBudget::new(16, 16, bytes - 1).unwrap(),
            options(true)
        ),
        Err(Texture3dError::Image(ImageError::BudgetExceeded { .. }))
    ));
    assert_chain(fixture, &tile, &expected_green_chain());
}

fn expected_green_chain() -> Vec<Vec<u8>> {
    [16, 8, 4, 2, 1]
        .map(|side| [0, 255, 0, 255].repeat(side * side))
        .to_vec()
}

fn assert_alpha_and_npot_mips(fixture: &mut Fixture<'_>) {
    let rgb = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]].concat();
    for (width, height) in [(3, 1), (1, 3)] {
        let texture = fixture.texture(width, height, rgb.clone(), true);
        assert_chain(fixture, &texture, &[rgb.clone(), vec![156, 156, 156, 255]]);
    }
    let texture = fixture.texture(1, 1, vec![0, 255, 0, 255], true);
    assert_chain(fixture, &texture, &[vec![0, 255, 0, 255]]);
    let alpha = [
        [255, 0, 0, 255],
        [0, 0, 255, 0],
        [0, 0, 255, 0],
        [0, 0, 255, 0],
    ]
    .concat();
    let texture = fixture.texture(2, 2, alpha.clone(), true);
    assert_chain(fixture, &texture, &[alpha, vec![255, 0, 0, 64]]);
    for sampling in [ImageSampling::Nearest, ImageSampling::Linear] {
        let material = TextureMaterial3d::with_alpha(&texture, sampling, Color::WHITE)
            .unwrap()
            .with_address_mode(TextureAddressMode3d::Repeat)
            .with_uv_transform(TextureUvTransform3d::new(Vec2::splat(128.0), Vec2::ZERO).unwrap());
        for (style, expected) in [
            (
                SurfaceStyle3d::blend(Color::WHITE).unwrap(),
                [64.0 / 255.0, 0.0, 0.0, 64.0 / 255.0],
            ),
            (
                SurfaceStyle3d::opaque(Color::WHITE).unwrap(),
                [1.0, 0.0, 0.0, 1.0],
            ),
            (SurfaceStyle3d::mask(Color::WHITE, 0.5).unwrap(), [0.0; 4]),
            (
                SurfaceStyle3d::mask(Color::WHITE, 0.1).unwrap(),
                [1.0, 0.0, 0.0, 1.0],
            ),
        ] {
            let pixels = fixture.render(
                &[full_quad(RECT)],
                &material,
                style,
                SurfaceRasterization3d::Native,
            );
            fixture.expect_uniform(&pixels, RECT, expected);
        }
    }
}

// Equal-area four-child reference for power-of-two acceptance images only.
// NPOT filtering is checked against literal independent 3x1/1x3 values above.
fn reference_chain(
    mut width: usize,
    mut height: usize,
    base: Vec<u8>,
    mipmaps: bool,
) -> Vec<Vec<u8>> {
    let mut result = vec![base];
    if !mipmaps {
        return result;
    }
    while width > 1 || height > 1 {
        assert!(width.is_power_of_two() && height.is_power_of_two());
        let next_width = (width / 2).max(1);
        let next_height = (height / 2).max(1);
        let source = result.last().unwrap();
        let mut next = Vec::new();
        for y in 0..next_height {
            for x in 0..next_width {
                let mut alpha = 0.0;
                let mut rgb = [0.0; 3];
                let mut count = 0;
                for row in (y * 2)..((y * 2 + 2).min(height)) {
                    for column in (x * 2)..((x * 2 + 2).min(width)) {
                        let pixel = &source[(row * width + column) * 4..][..4];
                        let coverage = f64::from(pixel[3]) / 255.0;
                        alpha += coverage;
                        for channel in 0..3 {
                            rgb[channel] += linear(pixel[channel]) * coverage;
                        }
                        count += 1;
                    }
                }
                let opacity = (alpha / f64::from(count) * 255.0).round() as u8;
                next.extend(if opacity == 0 {
                    [0, 0, 0, 0]
                } else {
                    [
                        encoded(rgb[0] / alpha),
                        encoded(rgb[1] / alpha),
                        encoded(rgb[2] / alpha),
                        opacity,
                    ]
                });
            }
        }
        result.push(next);
        width = next_width;
        height = next_height;
    }
    result
}

fn apply_patch_pixels(
    base: &mut [u8],
    width: usize,
    region: ImageTexelRect,
    pixels: &[u8],
    stride: usize,
) {
    let row_bytes = region.width() as usize * 4;
    for row in 0..region.height() as usize {
        let start = ((region.y() as usize + row) * width + region.x() as usize) * 4;
        base[start..start + row_bytes]
            .copy_from_slice(&pixels[row * stride..row * stride + row_bytes]);
    }
}

fn patch_rows(color: [u8; 4], height: usize) -> Vec<u8> {
    let mut pixels = Vec::new();
    for row in 0..height {
        pixels.extend(color.repeat(2));
        if row + 1 < height {
            pixels.extend([231, 17, 99, 0]);
        }
    }
    pixels
}

fn texture_display(fixture: &mut Fixture<'_>, texture: &Texture3d) -> Vec<u8> {
    let material =
        TextureMaterial3d::with_alpha(texture, ImageSampling::Nearest, Color::WHITE).unwrap();
    fixture.render(
        &[full_quad(RECT)],
        &material,
        SurfaceStyle3d::blend(Color::WHITE).unwrap(),
        SurfaceRasterization3d::Native,
    )
}

fn assert_region_updates(fixture: &mut Fixture<'_>) {
    for mipmaps in [false, true] {
        let mut model = [0, 0, 255, 255].repeat(64);
        let mut texture = fixture.texture(8, 8, model.clone(), mipmaps);
        let mut old = None;
        let mut old_chain = Vec::new();
        let mut old_display = Vec::new();
        for (index, (region, pixels)) in [
            (
                ImageTexelRect::new(0, 0, 2, 2).unwrap(),
                patch_rows([255, 0, 0, 255], 2),
            ),
            (
                ImageTexelRect::new(6, 6, 2, 2).unwrap(),
                patch_rows([0, 255, 0, 255], 2),
            ),
            (
                ImageTexelRect::new(1, 1, 2, 3).unwrap(),
                patch_rows([255, 255, 0, 128], 3),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            if index == 1 {
                old = Some(texture.clone());
                old_chain = reference_chain(8, 8, model.clone(), mipmaps);
                old_display = texture_display(fixture, &texture);
            }
            let old_identity = texture.identity_key();
            let report = update_texture_region(
                fixture.device,
                fixture.queue,
                &fixture.identity,
                &fixture.renderer.textures.layout,
                &mut texture,
                region,
                &pixels,
                12,
                Texture3dUpdateBudget::default(),
            )
            .unwrap();
            apply_patch_pixels(&mut model, 8, region, &pixels, 12);
            let expected = reference_chain(8, 8, model.clone(), mipmaps);
            assert_chain(fixture, &texture, &expected);
            let base_bytes = region.width() as usize * region.height() as usize * 4;
            let lower_bytes = expected.iter().skip(1).map(Vec::len).sum::<usize>();
            assert_eq!(report.base_upload_bytes(), base_bytes);
            assert_eq!(report.mip_upload_bytes(), lower_bytes);
            assert_eq!(report.uploaded_bytes(), base_bytes + lower_bytes);
            assert_eq!(report.upload_calls(), expected.len());
            assert_eq!(report.regenerated_texel_count(), lower_bytes / 4);
            assert_eq!(report.detached_aliases(), index == 1);
            assert_eq!(report.reused_allocation(), index != 1);
            assert_eq!(report.gpu_allocation_count(), usize::from(index == 1));
            assert_eq!(
                report.gpu_copy_bytes(),
                if index == 1 {
                    texture.gpu_allocation_bytes()
                } else {
                    0
                }
            );
            assert_eq!(
                report.gpu_copy_calls(),
                if index == 1 { expected.len() } else { 0 }
            );
            assert_eq!(report.submission_count(), if index == 1 { 2 } else { 1 });
            assert_eq!(
                report.peak_gpu_bytes(),
                texture.gpu_allocation_bytes() * if index == 1 { 2 } else { 1 }
            );
            assert_eq!(texture.identity_key() == old_identity, index != 1);
            if let Some(old) = &old {
                assert_chain(fixture, old, &old_chain);
                assert_eq!(texture_display(fixture, old), old_display);
            }
        }
        assert_update_failures(fixture, &mut texture, &model, mipmaps);
    }
}

fn assert_update_failures(
    fixture: &mut Fixture<'_>,
    texture: &mut Texture3d,
    model: &[u8],
    mipmaps: bool,
) {
    let expected = reference_chain(8, 8, model.to_vec(), mipmaps);
    let display = texture_display(fixture, texture);
    let identity = texture.identity_key();
    for (region, data, stride) in [
        (ImageTexelRect::new(7, 7, 2, 2).unwrap(), vec![255; 16], 8),
        (ImageTexelRect::new(0, 0, 2, 2).unwrap(), vec![255; 16], 7),
        (ImageTexelRect::new(0, 0, 2, 2).unwrap(), vec![255; 15], 8),
        (ImageTexelRect::new(0, 0, 2, 2).unwrap(), vec![255; 17], 8),
    ] {
        assert!(
            update_texture_region(
                fixture.device,
                fixture.queue,
                &fixture.identity,
                &fixture.renderer.textures.layout,
                texture,
                region,
                &data,
                stride,
                Texture3dUpdateBudget::default()
            )
            .is_err()
        );
        assert_eq!(texture.identity_key(), identity);
        assert_chain(fixture, texture, &expected);
        assert_eq!(texture_display(fixture, texture), display);
    }
    let maximum = usize::MAX;
    for (budget, resource) in [
        (
            Texture3dUpdateBudget::new(0, maximum, maximum, maximum, maximum),
            Texture3dUpdateBudgetResource::UploadBytes,
        ),
        (
            Texture3dUpdateBudget::new(maximum, 0, maximum, maximum, maximum),
            Texture3dUpdateBudgetResource::StagingBytes,
        ),
        (
            Texture3dUpdateBudget::new(maximum, maximum, 0, maximum, maximum),
            Texture3dUpdateBudgetResource::PeakRecoveryBytes,
        ),
        (
            Texture3dUpdateBudget::new(maximum, maximum, maximum, 0, maximum),
            Texture3dUpdateBudgetResource::PeakGpuBytes,
        ),
    ] {
        let error = update_texture_region(
            fixture.device,
            fixture.queue,
            &fixture.identity,
            &fixture.renderer.textures.layout,
            texture,
            ImageTexelRect::new(0, 0, 1, 1).unwrap(),
            &[255; 4],
            4,
            budget,
        )
        .unwrap_err();
        assert!(
            matches!(error, Texture3dUpdateError::BudgetExceeded { resource: actual, limit: 0, .. } if actual == resource)
        );
        assert_eq!(texture.identity_key(), identity);
        assert_chain(fixture, texture, &expected);
        assert_eq!(texture_display(fixture, texture), display);
    }
    let alias = texture.clone();
    let error = update_texture_region(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        texture,
        ImageTexelRect::new(0, 0, 1, 1).unwrap(),
        &[255; 4],
        4,
        Texture3dUpdateBudget::new(maximum, maximum, maximum, maximum, 0),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Texture3dUpdateError::BudgetExceeded {
            resource: Texture3dUpdateBudgetResource::GpuCopyBytes,
            limit: 0,
            ..
        }
    ));
    assert_eq!(texture.identity_key(), identity);
    assert_chain(fixture, texture, &expected);
    assert_chain(fixture, &alias, &expected);
    assert_eq!(texture_display(fixture, texture), display);
}

fn assert_scene_updates(fixture: &mut Fixture<'_>) {
    let (mut scene, object) = {
        let texture = fixture.texture(8, 8, [0, 0, 255, 255].repeat(64), true);
        let material =
            TextureMaterial3d::with_alpha(&texture, ImageSampling::Nearest, Color::WHITE)
                .unwrap()
                .with_address_mode(TextureAddressMode3d::Repeat)
                .with_uv_transform(
                    TextureUvTransform3d::new(Vec2::new(-2.0, 3.0), Vec2::new(0.25, -0.25))
                        .unwrap(),
                );
        let mesh = fixture.upload(full_quad(RECT), &material);
        let mut scene = Scene3d::with_alpha_background(Color::TRANSPARENT).unwrap();
        let object = scene
            .try_push(
                &mesh,
                Transform3d::IDENTITY,
                MeshStyle3d::surface(SurfaceStyle3d::blend(Color::WHITE).unwrap()),
            )
            .unwrap();
        (scene, object)
    };
    let original_key = scene.instances()[0]
        .mesh
        .material()
        .unwrap()
        .texture()
        .identity_key();
    let patch = patch_rows([255, 0, 0, 255], 2);
    let region = ImageTexelRect::new(0, 0, 2, 2).unwrap();
    let report = update_scene_texture_region(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        &mut scene,
        object,
        region,
        &patch,
        12,
        Texture3dUpdateBudget::default(),
    )
    .unwrap();
    assert!(report.reused_allocation());
    assert_eq!(report.base_upload_bytes(), 16);
    assert_eq!(report.mip_upload_bytes(), 84);
    assert_eq!(
        scene.instances()[0]
            .mesh
            .material()
            .unwrap()
            .texture()
            .identity_key(),
        original_key
    );
    let mut model = [0, 0, 255, 255].repeat(64);
    apply_patch_pixels(&mut model, 8, region, &patch, 12);
    assert_chain(
        fixture,
        scene.instances()[0].mesh.material().unwrap().texture(),
        &reference_chain(8, 8, model.clone(), true),
    );
    // A separate scene owns a retained alias; Scene3d itself intentionally has
    // no Clone because its stable object IDs carry scene-local provenance.
    let mut snapshot = Scene3d::with_alpha_background(scene.background()).unwrap();
    let instance = &scene.instances()[0];
    let snapshot_object = snapshot
        .try_push(instance.mesh(), instance.transform(), instance.style())
        .unwrap();
    assert_ne!(snapshot_object, object);
    let before = fixture.render_scene(&snapshot, SurfaceRasterization3d::Native);
    let patch = patch_rows([0, 255, 0, 128], 2);
    let region = ImageTexelRect::new(3, 3, 2, 2).unwrap();
    let report = update_scene_texture_region(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        &mut scene,
        object,
        region,
        &patch,
        12,
        Texture3dUpdateBudget::default(),
    )
    .unwrap();
    assert!(report.detached_aliases());
    assert_ne!(
        scene.instances()[0]
            .mesh
            .material()
            .unwrap()
            .texture()
            .identity_key(),
        original_key
    );
    apply_patch_pixels(&mut model, 8, region, &patch, 12);
    assert_chain(
        fixture,
        scene.instances()[0].mesh.material().unwrap().texture(),
        &reference_chain(8, 8, model, true),
    );
    assert_eq!(
        fixture.render_scene(&snapshot, SurfaceRasterization3d::Native),
        before
    );
    assert_ne!(
        fixture.render_scene(&scene, SurfaceRasterization3d::Native),
        before
    );
    assert_eq!(scene.instances()[0].id(), object);
    assert_eq!(
        scene.instances()[0]
            .mesh
            .material()
            .unwrap()
            .uv_transform()
            .scale(),
        Vec2::new(-2.0, 3.0)
    );
    assert_eq!(
        scene.instances()[0].mesh.material().unwrap().address_mode(),
        TextureAddressMode3d::Repeat
    );
}

fn assert_scene_policy_and_budget_failures(fixture: &mut Fixture<'_>) {
    let mut texture = fixture.texture(8, 8, [0, 0, 255, 255].repeat(64), true);
    let original_chain = reference_chain(8, 8, texture.pixels().to_vec(), true);
    let legacy = TextureMaterial3d::new(&texture, ImageSampling::Nearest, Color::WHITE).unwrap();
    let mesh = fixture.upload(full_quad(RECT), &legacy);
    let mut scene = Scene3d::with_alpha_background(Color::TRANSPARENT).unwrap();
    let id = scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
        )
        .unwrap();
    let before = fixture.render_scene(&scene, SurfaceRasterization3d::Native);
    let error = update_scene_texture_region(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        &mut scene,
        id,
        ImageTexelRect::new(0, 0, 1, 1).unwrap(),
        &[255, 0, 0, 128],
        4,
        Texture3dUpdateBudget::default(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Texture3dUpdateError::Texture(Texture3dError::NonOpaquePixel { texel: 0 })
    );
    assert_chain(fixture, &texture, &original_chain);
    assert_eq!(
        fixture.render_scene(&scene, SurfaceRasterization3d::Native),
        before
    );
    // Standalone alpha-capable edits are legal; rebinding an originally opaque
    // material to the new snapshot is not. Its existing scene remains drawable.
    update_texture_region(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        &mut texture,
        ImageTexelRect::new(0, 0, 1, 1).unwrap(),
        &[255, 0, 0, 128],
        4,
        Texture3dUpdateBudget::default(),
    )
    .unwrap();
    assert!(matches!(
        legacy.with_texture(&texture),
        Err(Texture3dError::NonOpaquePixel { texel: 0 })
    ));
    assert_chain(fixture, legacy.texture(), &original_chain);
    assert_eq!(
        fixture.render_scene(&scene, SurfaceRasterization3d::Native),
        before
    );

    let material =
        TextureMaterial3d::with_alpha(&texture, ImageSampling::Nearest, Color::WHITE).unwrap();
    let budget = Scene3dBudget::default()
        .with_texture_limits(usize::MAX, texture.gpu_allocation_bytes() * 2 - 1);
    let mut bounded = Scene3d::with_budget(Color::BLACK, budget).unwrap();
    let left = fixture.upload(full_quad([4.0, 4.0, 30.0, 60.0]), &material);
    let right = fixture.upload(full_quad([34.0, 4.0, 60.0, 60.0]), &material);
    let style = MeshStyle3d::surface(SurfaceStyle3d::blend(Color::WHITE).unwrap());
    let id = bounded
        .try_push(&left, Transform3d::IDENTITY, style)
        .unwrap();
    bounded
        .try_push(&right, Transform3d::IDENTITY, style)
        .unwrap();
    let expected = fixture.levels(&texture);
    let before = fixture.render_scene(&bounded, SurfaceRasterization3d::Native);
    let error = update_scene_texture_region(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        &mut bounded,
        id,
        ImageTexelRect::new(1, 1, 1, 1).unwrap(),
        &[0, 255, 0, 255],
        4,
        Texture3dUpdateBudget::default(),
    )
    .unwrap_err();
    assert!(
        matches!(error, Texture3dUpdateError::Scene(Scene3dError::BudgetExceeded {
        resource: Scene3dBudgetResource::TextureGpuBytes, limit, actual
    }) if actual == texture.gpu_allocation_bytes() * 2 && limit + 1 == actual)
    );
    assert_eq!(bounded.statistics().texture_count(), 1);
    assert_eq!(
        bounded.statistics().texture_gpu_bytes(),
        texture.gpu_allocation_bytes()
    );
    assert_chain(fixture, &texture, &expected);
    assert_eq!(
        fixture.render_scene(&bounded, SurfaceRasterization3d::Native),
        before
    );
}

pub(in crate::renderer::mesh3d) fn assert_gpu_texture_lifecycle_recovery(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let mut fixture = Fixture::new(device, queue, format);
    let parent: Vec<u8> = (0..8)
        .flat_map(|_| {
            (0..16).flat_map(|x| {
                if x < 8 {
                    [0, 0, 255, 255]
                } else {
                    [0, 255, 0, 255]
                }
            })
        })
        .collect();
    let atlas = fixture.texture(16, 8, parent.clone(), false);
    let mut texture = crop_texture_tile(
        device,
        queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        &atlas,
        ImageTexelRect::new(0, 0, 8, 8).unwrap(),
        ImageBudget::default(),
        options(true),
    )
    .unwrap();
    let patch = patch_rows([255, 0, 0, 128], 2);
    let region = ImageTexelRect::new(0, 0, 2, 2).unwrap();
    update_texture_region(
        device,
        queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        &mut texture,
        region,
        &patch,
        12,
        Texture3dUpdateBudget::default(),
    )
    .unwrap();
    let mut base = [0, 0, 255, 255].repeat(64);
    apply_patch_pixels(&mut base, 8, region, &patch, 12);
    let expected = reference_chain(8, 8, base, true);
    assert_chain(&fixture, &texture, &expected);
    assert_eq!(atlas.pixels(), parent);
    assert_eq!(read_level(device, queue, &atlas, 0), parent);
    let mut scene = Scene3d::with_alpha_background(Color::TRANSPARENT).unwrap();
    for (rect, sampling, scale, offset) in [
        (
            [-12.0, 4.0, 60.0, 30.0],
            ImageSampling::Linear,
            Vec2::new(-64.0, 64.0),
            Vec2::new(0.25, -0.25),
        ),
        (
            [4.0, 34.0, 60.0, 60.0],
            ImageSampling::Nearest,
            Vec2::new(-2.0, 3.0),
            Vec2::new(0.25, -0.25),
        ),
    ] {
        let material = TextureMaterial3d::with_alpha(&texture, sampling, Color::WHITE)
            .unwrap()
            .with_address_mode(TextureAddressMode3d::Repeat)
            .with_uv_transform(TextureUvTransform3d::new(scale, offset).unwrap());
        let mesh = fixture.upload(full_quad(rect), &material);
        scene
            .try_push(
                &mesh,
                Transform3d::IDENTITY,
                MeshStyle3d::surface(SurfaceStyle3d::blend(Color::WHITE).unwrap()),
            )
            .unwrap();
    }
    let ids: Vec<_> = scene.instances().iter().map(Mesh3dInstance::id).collect();
    let before = fixture.render_scene(&scene, SurfaceRasterization3d::StrictPortable);
    let last = &expected.last().unwrap()[..4];
    let alpha = f64::from(last[3]) / 255.0;
    let color = [
        linear(last[0]) * alpha,
        linear(last[1]) * alpha,
        linear(last[2]) * alpha,
        alpha,
    ];
    fixture.expect_uniform(&before, [4.0, 4.0, 60.0, 30.0], color);
    let mut restored = Fixture::new(recovery_device, recovery_queue, format);
    let report = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        &restored.renderer.textures.layout,
        restored.identity.clone(),
        &mut scene,
    )
    .unwrap();
    assert_eq!(report.restored_texture_count(), 1);
    assert_eq!(
        report.restored_texture_bytes(),
        expected.iter().map(Vec::len).sum::<usize>()
    );
    assert_eq!(
        scene
            .instances()
            .iter()
            .map(Mesh3dInstance::id)
            .collect::<Vec<_>>(),
        ids
    );
    for instance in scene.instances() {
        let material = instance.mesh.material().unwrap();
        assert_chain(&restored, material.texture(), &expected);
        assert_eq!(material.address_mode(), TextureAddressMode3d::Repeat);
        assert_eq!(material.uv_transform().offset(), Vec2::new(0.25, -0.25));
        assert_eq!(material.texture().options(), options(true));
    }
    assert_eq!(scene.statistics().texture_count(), 1);
    assert_eq!(
        restored.render_scene(&scene, SurfaceRasterization3d::StrictPortable),
        before
    );
    assert_chain(&fixture, &texture, &expected);
}

fn tile_cuts(scale: f32, offset: f32) -> Vec<f32> {
    let mut result = vec![0.0, 1.0];
    if scale != 0.0 {
        for boundary in -8..=8 {
            let source = (boundary as f32 - offset) / scale;
            if source > 0.0 && source < 1.0 {
                result.push(source);
            }
        }
    }
    result.sort_by(f32::total_cmp);
    result
}

fn tiled_reference(rect: [f32; 4], scale: Vec2, offset: Vec2) -> Vec<Mesh3d> {
    let columns = tile_cuts(scale.x(), offset.x());
    let rows = tile_cuts(scale.y(), offset.y());
    let mut result = Vec::new();
    for row in rows.windows(2) {
        for column in columns.windows(2) {
            let period_u = (((column[0] + column[1]) * 0.5) * scale.x() + offset.x()).floor();
            let period_v = (((row[0] + row[1]) * 0.5) * scale.y() + offset.y()).floor();
            let uv = [
                [column[0], row[0]],
                [column[1], row[0]],
                [column[1], row[1]],
                [column[0], row[1]],
            ]
            .map(|[u, v]| {
                [
                    (u * scale.x() + offset.x() - period_u).clamp(0.0, 1.0),
                    (v * scale.y() + offset.y() - period_v).clamp(0.0, 1.0),
                ]
            });
            let tile = [
                rect[0] + column[0] * (rect[2] - rect[0]),
                rect[1] + row[0] * (rect[3] - rect[1]),
                rect[0] + column[1] * (rect[2] - rect[0]),
                rect[1] + row[1] * (rect[3] - rect[1]),
            ];
            result.push(quad(tile, uv));
        }
    }
    result
}

fn assert_repeated_coordinates(fixture: &mut Fixture<'_>) {
    let texture = fixture.texture(2, 2, QUADRANTS.concat(), false);
    for rect in [RECT, [-12.0, 4.0, 60.0, 60.0]] {
        for (scale, offset) in [
            (Vec2::new(3.0, 2.0), Vec2::new(0.25, -0.25)),
            (Vec2::new(-3.0, 2.0), Vec2::new(1.25, -0.25)),
            (Vec2::new(3.0, -2.0), Vec2::new(-1.25, 0.25)),
            (Vec2::new(0.0, 2.0), Vec2::new(0.25, -0.25)),
        ] {
            for address in [TextureAddressMode3d::Clamp, TextureAddressMode3d::Repeat] {
                let material =
                    TextureMaterial3d::with_alpha(&texture, ImageSampling::Nearest, Color::WHITE)
                        .unwrap()
                        .with_address_mode(address)
                        .with_uv_transform(TextureUvTransform3d::new(scale, offset).unwrap());
                for policy in [
                    SurfaceRasterization3d::Native,
                    SurfaceRasterization3d::StrictPortable,
                ] {
                    let actual = fixture.render(
                        &[full_quad(rect)],
                        &material,
                        SurfaceStyle3d::opaque(Color::WHITE).unwrap(),
                        policy,
                    );
                    let tiled = if address == TextureAddressMode3d::Repeat {
                        let plain = TextureMaterial3d::with_alpha(
                            &texture,
                            ImageSampling::Nearest,
                            Color::WHITE,
                        )
                        .unwrap()
                        .with_address_mode(address);
                        Some(fixture.render(
                            &tiled_reference(rect, scale, offset),
                            &plain,
                            SurfaceStyle3d::opaque(Color::WHITE).unwrap(),
                            SurfaceRasterization3d::Native,
                        ))
                    } else {
                        None
                    };
                    let mut count = 0;
                    for y in 6..58 {
                        for x in (rect[0].max(0.0) as u32 + 2)..58 {
                            let u = ((x as f64 + 0.5 - f64::from(rect[0]))
                                / f64::from(rect[2] - rect[0]))
                                * f64::from(scale.x())
                                + f64::from(offset.x());
                            let v = ((y as f64 + 0.5 - f64::from(rect[1]))
                                / f64::from(rect[3] - rect[1]))
                                * f64::from(scale.y())
                                + f64::from(offset.y());
                            let wrap = |value: f64| {
                                if address == TextureAddressMode3d::Repeat {
                                    value.rem_euclid(1.0)
                                } else {
                                    value.clamp(0.0, 1.0)
                                }
                            };
                            let u = wrap(u);
                            let v = wrap(v);
                            if (u - 0.5).abs() < 0.015
                                || (v - 0.5).abs() < 0.015
                                || (address == TextureAddressMode3d::Repeat
                                    && (u.min(1.0 - u) < 0.015 || v.min(1.0 - v) < 0.015))
                            {
                                continue;
                            }
                            let index = usize::from(u >= 0.5) + 2 * usize::from(v >= 0.5);
                            let expected =
                                QUADRANTS[index].map(|channel| f64::from(channel) / 255.0);
                            fixture.expect(&actual, x, y, expected);
                            if let Some(tiled) = &tiled {
                                fixture.expect(tiled, x, y, expected);
                            }
                            count += 1;
                        }
                    }
                    assert!(
                        count > 1000,
                        "UV oracle must compare substantial non-boundary coverage"
                    );
                }
            }
        }
    }
}
