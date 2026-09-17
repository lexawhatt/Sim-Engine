//! Regional texture updates checked against frozen full CPU and GPU mip bytes.

use super::*;
use crate::{TextureAddressMode3d, TextureUvTransform3d, Vec2};

use full_mip_reference::Level;

#[derive(Clone)]
struct Model {
    width: u32,
    height: u32,
    base: Vec<u8>,
    mipmaps: bool,
}

impl Model {
    fn new(width: u32, height: u32, alpha: bool, mipmaps: bool) -> Self {
        Self {
            width,
            height,
            base: pixels(width, height, alpha, 17),
            mipmaps,
        }
    }

    fn levels(&self) -> Vec<Level> {
        if self.mipmaps {
            full_mip_reference::rebuild(self.width, self.height, self.base.clone())
        } else {
            vec![Level {
                width: self.width,
                height: self.height,
                pixels: self.base.clone(),
            }]
        }
    }

    fn patch(&mut self, region: ImageTexelRect, pixels: &[u8], stride: usize) {
        let row_bytes = region.width() as usize * 4;
        for row in 0..region.height() as usize {
            let destination =
                ((region.y() as usize + row) * self.width as usize + region.x() as usize) * 4;
            self.base[destination..destination + row_bytes]
                .copy_from_slice(&pixels[row * stride..row * stride + row_bytes]);
        }
    }

    fn gpu_bytes(&self) -> usize {
        let (mut width, mut height) = (self.width, self.height);
        let mut bytes = 0;
        loop {
            bytes += width as usize * height as usize * 4;
            if !self.mipmaps || (width == 1 && height == 1) {
                return bytes;
            }
            width = (width / 2).max(1);
            height = (height / 2).max(1);
        }
    }
}

fn pixels(width: u32, height: u32, alpha: bool, seed: usize) -> Vec<u8> {
    (0..width as usize * height as usize)
        .flat_map(|index| {
            let value = index.wrapping_mul(0x045d_9f3b) ^ seed.wrapping_mul(0x119d_e1f3);
            [
                value as u8,
                (value >> 8) as u8,
                (value >> 16) as u8,
                if alpha {
                    [0, 1, 254, 255][(index + seed) % 4]
                } else {
                    255
                },
            ]
        })
        .collect()
}

fn padded_patch(region: ImageTexelRect, alpha: bool, seed: usize) -> (Vec<u8>, usize) {
    let packed = pixels(region.width(), region.height(), alpha, seed);
    let row = region.width() as usize * 4;
    let stride = row + 12;
    let mut padded = vec![0; (region.height() as usize - 1) * stride + row];
    for y in 0..region.height() as usize {
        padded[y * stride..y * stride + row].copy_from_slice(&packed[y * row..(y + 1) * row]);
    }
    (padded, stride)
}

fn affected_axis(source: u32, destination: u32, start: u32, count: u32) -> (u32, u32) {
    let ratio = f64::from(source) / f64::from(destination);
    let affected: Vec<_> = (0..destination)
        .filter(|&texel| {
            (f64::from(texel) + 1.0) * ratio > f64::from(start)
                && f64::from(texel) * ratio < f64::from(start + count)
        })
        .collect();
    (
        *affected.first().unwrap(),
        affected.last().unwrap() - affected.first().unwrap() + 1,
    )
}

fn mip_upload_bytes(model: &Model, region: ImageTexelRect) -> usize {
    let (mut source_width, mut source_height) = (model.width, model.height);
    let (mut x, mut y, mut width, mut height) =
        (region.x(), region.y(), region.width(), region.height());
    let mut bytes = 0;
    while model.mipmaps && (source_width > 1 || source_height > 1) {
        let destination_width = (source_width / 2).max(1);
        let destination_height = (source_height / 2).max(1);
        (x, width) = affected_axis(source_width, destination_width, x, width);
        (y, height) = affected_axis(source_height, destination_height, y, height);
        bytes += width as usize * height as usize * 4;
        source_width = destination_width;
        source_height = destination_height;
    }
    bytes
}

struct Fixture<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    identity: Arc<()>,
    renderer: Mesh3dRenderer,
}

impl<'a> Fixture<'a> {
    fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self {
            device,
            queue,
            identity: Arc::new(()),
            renderer: Mesh3dRenderer::new(device, format),
        }
    }

    fn texture(&self, model: &Model, preserve_alpha: bool) -> Texture3d {
        create_texture_with_options(
            self.device,
            self.queue,
            &self.identity,
            &self.renderer.textures.layout,
            model.width,
            model.height,
            model.base.clone(),
            ImageBudget::default(),
            Texture3dOptions::new()
                .with_alpha_preservation(preserve_alpha)
                .with_mipmaps(if model.mipmaps {
                    TextureMipmaps3d::Generate
                } else {
                    TextureMipmaps3d::None
                }),
        )
        .unwrap()
    }

    fn mesh(&self, material: &TextureMaterial3d) -> RetainedMesh3d {
        let source = Mesh3d::textured(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]
                .map(|[x, y, z]| Vec3::new(x, y, z).unwrap())
                .to_vec(),
            // Source UVs are normalized; signed/repeating mapping is carried
            // by the material transform, whose policy is checked separately.
            [[0.0, 1.0], [1.0, 1.0], [0.5, 0.0]]
                .map(|[u, v]| TextureCoordinate2d::new(u, v).unwrap())
                .to_vec(),
            vec![0, 1, 2],
            Vec::new(),
        )
        .unwrap();
        let mesh =
            create_retained_mesh(self.device, self.queue, Arc::clone(&self.identity), source)
                .unwrap();
        attach_material(&self.identity, &mesh, material).unwrap()
    }

    fn update(
        &self,
        texture: &mut Texture3d,
        region: ImageTexelRect,
        patch: &[u8],
        stride: usize,
        budget: Texture3dUpdateBudget,
    ) -> Result<Texture3dUpdateReport, Texture3dUpdateError> {
        update_texture_region(
            self.device,
            self.queue,
            &self.identity,
            &self.renderer.textures.layout,
            texture,
            region,
            patch,
            stride,
            budget,
        )
    }

    fn update_scene(
        &self,
        scene: &mut Scene3d,
        object: Object3dId,
        region: ImageTexelRect,
        patch: &[u8],
        stride: usize,
    ) -> Result<Texture3dUpdateReport, Texture3dUpdateError> {
        update_scene_texture_region(
            self.device,
            self.queue,
            &self.identity,
            &self.renderer.textures.layout,
            scene,
            object,
            region,
            patch,
            stride,
            Texture3dUpdateBudget::default(),
        )
    }

    fn assert_chain(&self, texture: &Texture3d, model: &Model) {
        let levels = model.levels();
        assert_eq!(texture.mip_level_count() as usize, levels.len());
        assert_eq!(texture.gpu_allocation_bytes(), model.gpu_bytes());
        for (index, level) in levels.iter().enumerate() {
            assert_eq!(
                texture.mip_level_size(index as u32),
                Some((level.width, level.height))
            );
            assert_eq!(
                texture.mip_level_pixels(index as u32).unwrap(),
                level.pixels,
                "CPU mip {index}"
            );
            let stride = (level.width * 4).div_ceil(256) * 256;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("partial mip full-level oracle readback"),
                size: u64::from(stride) * u64::from(level.height),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.storage.texture,
                    mip_level: index as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(stride),
                        rows_per_image: Some(level.height),
                    },
                },
                wgpu::Extent3d {
                    width: level.width,
                    height: level.height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit([encoder.finish()]);
            let slice = buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            receiver
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap();
            let mapped = slice.get_mapped_range().unwrap();
            for (row, expected) in level
                .pixels
                .chunks_exact(level.width as usize * 4)
                .enumerate()
            {
                let start = row * stride as usize;
                assert_eq!(
                    &mapped[start..start + expected.len()],
                    expected,
                    "GPU mip {index}, row {row}"
                );
            }
            drop(mapped);
            buffer.unmap();
        }
    }
}

fn assert_report(
    report: Texture3dUpdateReport,
    model: &Model,
    region: ImageTexelRect,
    old_cpu: usize,
    aliased: bool,
) {
    let base = region.width() as usize * region.height() as usize * 4;
    let lower = mip_upload_bytes(model, region);
    let gpu = model.gpu_bytes();
    let levels = model.levels().len();
    assert_eq!(report.base_upload_bytes(), base);
    assert_eq!(report.mip_upload_bytes(), lower);
    assert_eq!(report.uploaded_bytes(), base + lower);
    assert_eq!(report.regenerated_texel_count(), lower / 4);
    assert_eq!(report.upload_calls(), levels);
    assert_eq!(report.gpu_copy_bytes(), if aliased { gpu } else { 0 });
    assert_eq!(report.gpu_copy_calls(), if aliased { levels } else { 0 });
    assert_eq!(report.gpu_allocation_count(), usize::from(aliased));
    assert_eq!(report.submission_count(), 1 + usize::from(aliased));
    assert_eq!(report.detached_aliases(), aliased);
    assert_eq!(report.gpu_bytes(), gpu);
    assert_eq!(report.peak_gpu_bytes(), gpu * (1 + usize::from(aliased)));
    assert_eq!(
        report.peak_recovery_bytes(),
        old_cpu + report.recovery_bytes()
    );
    assert!(report.staging_bytes() >= base + report.uploaded_bytes());
}

fn unique_and_aliased_updates(fixture: &Fixture<'_>) {
    for (width, height) in [(1, 1), (1, 33), (35, 1), (5, 7), (9, 6), (16, 16)] {
        for mipmaps in [false, true] {
            for alpha in [false, true] {
                let mut model = Model::new(width, height, alpha, mipmaps);
                let mut texture = fixture.texture(&model, alpha);
                for (step, region) in [
                    ImageTexelRect::new(width - 1, height - 1, 1, 1).unwrap(),
                    ImageTexelRect::new(0, 0, width.min(3), height.min(2)).unwrap(),
                    ImageTexelRect::new((width - 1) / 2, (height - 1) / 2, 1, 1).unwrap(),
                    ImageTexelRect::new(0, 0, width, height).unwrap(),
                ]
                .into_iter()
                .enumerate()
                {
                    let aliased = step == 1;
                    let snapshot = aliased.then(|| texture.clone());
                    let previous = model.clone();
                    let key = texture.identity_key();
                    let old_cpu = texture.recovery_memory_bytes();
                    let (patch, stride) = padded_patch(region, alpha, step + 101);
                    let report = fixture
                        .update(
                            &mut texture,
                            region,
                            &patch,
                            stride,
                            Texture3dUpdateBudget::default(),
                        )
                        .unwrap();
                    model.patch(region, &patch, stride);
                    assert_report(report, &model, region, old_cpu, aliased);
                    assert_eq!(texture.identity_key() == key, !aliased);
                    fixture.assert_chain(&texture, &model);
                    if let Some(snapshot) = snapshot {
                        fixture.assert_chain(&snapshot, &previous);
                    }
                    let old_cpu = texture.recovery_memory_bytes();
                    let report = fixture
                        .update(
                            &mut texture,
                            region,
                            &patch,
                            stride,
                            Texture3dUpdateBudget::default(),
                        )
                        .unwrap();
                    assert_report(report, &model, region, old_cpu, false);
                    fixture.assert_chain(&texture, &model);
                }
            }
        }
    }
}

fn operation_budget_boundaries(fixture: &Fixture<'_>) {
    for aliased in [false, true] {
        let mut model = Model::new(9, 7, true, true);
        let mut texture = fixture.texture(&model, true);
        let previous = model.clone();
        let snapshot = aliased.then(|| texture.clone());
        let key = texture.identity_key();
        let old_cpu = texture.recovery_memory_bytes();
        let region = ImageTexelRect::new(3, 2, 2, 3).unwrap();
        let (patch, stride) = padded_patch(region, true, 177);
        let base = region.width() as usize * region.height() as usize * 4;
        let upload = base + mip_upload_bytes(&model, region);
        let gpu = model.gpu_bytes();
        let limits = [
            upload,
            base + upload,
            old_cpu + gpu,
            gpu * (1 + usize::from(aliased)),
            if aliased { gpu } else { 0 },
        ];
        for (index, resource) in [
            Texture3dUpdateBudgetResource::UploadBytes,
            Texture3dUpdateBudgetResource::StagingBytes,
            Texture3dUpdateBudgetResource::PeakRecoveryBytes,
            Texture3dUpdateBudgetResource::PeakGpuBytes,
            Texture3dUpdateBudgetResource::GpuCopyBytes,
        ]
        .into_iter()
        .enumerate()
        {
            if limits[index] == 0 {
                continue;
            }
            let mut limited = limits;
            limited[index] -= 1;
            let error = fixture
                .update(
                    &mut texture,
                    region,
                    &patch,
                    stride,
                    Texture3dUpdateBudget::new(
                        limited[0], limited[1], limited[2], limited[3], limited[4],
                    ),
                )
                .unwrap_err();
            assert!(
                matches!(error, Texture3dUpdateError::BudgetExceeded { resource: actual, limit, actual: needed }
                if actual == resource && limit == limited[index] && needed == limits[index])
            );
            assert_eq!(texture.identity_key(), key);
            fixture.assert_chain(&texture, &model);
            if let Some(snapshot) = snapshot.as_ref() {
                fixture.assert_chain(snapshot, &previous);
            }
        }
        let report = fixture
            .update(
                &mut texture,
                region,
                &patch,
                stride,
                Texture3dUpdateBudget::new(limits[0], limits[1], limits[2], limits[3], limits[4]),
            )
            .unwrap();
        model.patch(region, &patch, stride);
        assert_report(report, &model, region, old_cpu, aliased);
        fixture.assert_chain(&texture, &model);
        if let Some(snapshot) = snapshot {
            fixture.assert_chain(&snapshot, &previous);
        }
    }
}

fn signed_material(texture: &Texture3d) -> TextureMaterial3d {
    TextureMaterial3d::with_alpha(
        texture,
        ImageSampling::Linear,
        Color::rgba(0.75, 0.5, 1.0, 0.5),
    )
    .unwrap()
    .with_address_mode(TextureAddressMode3d::Repeat)
    .with_uv_transform(
        TextureUvTransform3d::new(Vec2::new(-2.0, 3.0), Vec2::new(0.25, -0.5)).unwrap(),
    )
}

fn assert_policy(material: &TextureMaterial3d) {
    assert_eq!(material.sampling(), ImageSampling::Linear);
    assert_eq!(material.tint(), Color::rgba(0.75, 0.5, 1.0, 0.5));
    assert_eq!(material.address_mode(), TextureAddressMode3d::Repeat);
    assert_eq!(material.uv_transform().scale(), Vec2::new(-2.0, 3.0));
    assert_eq!(material.uv_transform().offset(), Vec2::new(0.25, -0.5));
    assert!(!material.require_opaque_texture);
}

fn scene_texture(scene: &Scene3d, id: Object3dId) -> &Texture3d {
    scene
        .instance(id)
        .unwrap()
        .mesh()
        .material()
        .unwrap()
        .texture()
}

fn scene_budget_boundaries(fixture: &Fixture<'_>) {
    for limited in [
        Some(Scene3dBudgetResource::TextureCpuBytes),
        Some(Scene3dBudgetResource::TextureGpuBytes),
        None,
    ] {
        let mut model = Model::new(9, 7, true, true);
        let original = model.clone();
        let texture = fixture.texture(&model, true);
        let gpu = model.gpu_bytes();
        let cpu = texture.recovery_memory_bytes();
        let budget = Scene3dBudget::new(2, usize::MAX, usize::MAX, usize::MAX)
            .unwrap()
            .with_texture_limits(
                cpu + gpu - usize::from(limited == Some(Scene3dBudgetResource::TextureCpuBytes)),
                2 * gpu - usize::from(limited == Some(Scene3dBudgetResource::TextureGpuBytes)),
            );
        let material = signed_material(&texture);
        let mesh = fixture.mesh(&material);
        let mut scene = Scene3d::with_budget(Color::BLACK, budget).unwrap();
        let style = MeshStyle3d::surface(SurfaceStyle3d::blend(Color::WHITE).unwrap());
        let first = scene.try_push(&mesh, Transform3d::IDENTITY, style).unwrap();
        let second = scene.try_push(&mesh, Transform3d::IDENTITY, style).unwrap();
        scene.set_visible(second, false).unwrap();
        let before = scene.statistics();
        let key = scene_texture(&scene, first).identity_key();
        let mesh_key = Arc::as_ptr(&scene.instance(first).unwrap().mesh.vertex_buffer);
        let region = ImageTexelRect::new(4, 1, 2, 3).unwrap();
        let (patch, stride) = padded_patch(region, true, 208);
        let result = fixture.update_scene(&mut scene, first, region, &patch, stride);
        if let Some(resource) = limited {
            assert!(
                matches!(result, Err(Texture3dUpdateError::Scene(Scene3dError::BudgetExceeded { resource: actual, .. })) if actual == resource)
            );
            assert_eq!(scene.statistics(), before);
            assert_eq!(scene_texture(&scene, first).identity_key(), key);
            fixture.assert_chain(scene_texture(&scene, first), &original);
        } else {
            let report = result.unwrap();
            model.patch(region, &patch, stride);
            assert_report(report, &model, region, cpu, true);
            assert_eq!(scene.statistics().texture_count(), 2);
            assert_eq!(scene.statistics().texture_gpu_bytes(), 2 * gpu);
            assert_eq!(
                scene.statistics().texture_cpu_bytes(),
                cpu + report.recovery_bytes()
            );
            fixture.assert_chain(scene_texture(&scene, first), &model);
            let key = scene_texture(&scene, first).identity_key();
            let old_cpu = scene_texture(&scene, first).recovery_memory_bytes();
            let report = fixture
                .update_scene(&mut scene, first, region, &patch, stride)
                .unwrap();
            assert_report(report, &model, region, old_cpu, false);
            assert_eq!(scene_texture(&scene, first).identity_key(), key);
            fixture.assert_chain(scene_texture(&scene, first), &model);
        }
        assert_eq!(
            Arc::as_ptr(&scene.instance(first).unwrap().mesh.vertex_buffer),
            mesh_key
        );
        assert_eq!(scene.instance(first).unwrap().id(), first);
        assert!(!scene.instance(second).unwrap().is_visible());
        assert_policy(scene.instance(first).unwrap().mesh.material().unwrap());
        fixture.assert_chain(scene_texture(&scene, second), &original);
    }
}

fn tile_and_material_rebinding(fixture: &Fixture<'_>) {
    let parent_model = Model::new(13, 11, true, true);
    let parent = fixture.texture(&parent_model, true);
    let crop = ImageTexelRect::new(3, 2, 7, 5).unwrap();
    let mut tile_base = Vec::new();
    for row in crop.y()..crop.y() + crop.height() {
        let start = (row as usize * parent_model.width as usize + crop.x() as usize) * 4;
        tile_base.extend_from_slice(&parent_model.base[start..start + crop.width() as usize * 4]);
    }
    let mut model = Model {
        width: crop.width(),
        height: crop.height(),
        base: tile_base,
        mipmaps: true,
    };
    let original = model.clone();
    let mut tile = crop_texture_tile(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        &parent,
        crop,
        ImageBudget::default(),
        Texture3dOptions::new()
            .with_alpha_preservation(true)
            .with_mipmaps(TextureMipmaps3d::Generate),
    )
    .unwrap();
    fixture.assert_chain(&tile, &model);
    let material = signed_material(&tile);
    let mesh = fixture.mesh(&material);
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let id = scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::blend(Color::WHITE).unwrap()),
        )
        .unwrap();
    let region = ImageTexelRect::new(4, 3, 3, 2).unwrap();
    let (patch, stride) = padded_patch(region, true, 291);
    let old_cpu = tile.recovery_memory_bytes();
    let report = fixture
        .update(
            &mut tile,
            region,
            &patch,
            stride,
            Texture3dUpdateBudget::default(),
        )
        .unwrap();
    model.patch(region, &patch, stride);
    assert_report(report, &model, region, old_cpu, true);
    fixture.assert_chain(&tile, &model);
    fixture.assert_chain(material.texture(), &original);
    fixture.assert_chain(scene_texture(&scene, id), &original);
    fixture.assert_chain(&parent, &parent_model);
    let rebound = material.with_texture(&tile).unwrap();
    assert_policy(&rebound);
    scene.set_texture_material(id, Some(&rebound)).unwrap();
    fixture.assert_chain(scene_texture(&scene, id), &model);
    let before = model.clone();
    let region = ImageTexelRect::new(0, 0, 2, 2).unwrap();
    let (patch, stride) = padded_patch(region, true, 292);
    let old_cpu = scene_texture(&scene, id).recovery_memory_bytes();
    let report = fixture
        .update_scene(&mut scene, id, region, &patch, stride)
        .unwrap();
    model.patch(region, &patch, stride);
    assert_report(report, &model, region, old_cpu, true);
    assert_policy(scene.instance(id).unwrap().mesh.material().unwrap());
    fixture.assert_chain(scene_texture(&scene, id), &model);
    fixture.assert_chain(&tile, &before);
    fixture.assert_chain(&parent, &parent_model);
}

fn invalid_input_and_opacity(fixture: &Fixture<'_>) {
    let mut model = Model::new(9, 7, false, true);
    let mut texture = fixture.texture(&model, false);
    let key = texture.identity_key();
    let region = ImageTexelRect::new(2, 3, 2, 2).unwrap();
    let (mut patch, stride) = padded_patch(region, false, 37);
    for (region, data, stride) in [
        (
            ImageTexelRect::new(8, 6, 2, 2).unwrap(),
            patch.as_slice(),
            stride,
        ),
        (region, &patch[..patch.len() - 1], stride),
        (region, patch.as_slice(), 7),
    ] {
        assert!(
            fixture
                .update(
                    &mut texture,
                    region,
                    data,
                    stride,
                    Texture3dUpdateBudget::default()
                )
                .is_err()
        );
        assert_eq!(texture.identity_key(), key);
        fixture.assert_chain(&texture, &model);
    }
    patch[3] = 0;
    let first_nonopaque = (region.y() * model.width + region.x()) as usize;
    assert_eq!(
        fixture.update(
            &mut texture,
            region,
            &patch,
            stride,
            Texture3dUpdateBudget::default()
        ),
        Err(Texture3dUpdateError::Texture(
            Texture3dError::NonOpaquePixel {
                texel: first_nonopaque
            }
        ))
    );
    fixture.assert_chain(&texture, &model);
    let mut alpha_texture = fixture.texture(&model, true);
    let opaque =
        TextureMaterial3d::new(&alpha_texture, ImageSampling::Nearest, Color::WHITE).unwrap();
    let mesh = fixture.mesh(&opaque);
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let id = scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
        )
        .unwrap();
    let statistics = scene.statistics();
    assert_eq!(
        fixture.update_scene(&mut scene, id, region, &patch, stride),
        Err(Texture3dUpdateError::Texture(
            Texture3dError::NonOpaquePixel {
                texel: first_nonopaque
            }
        ))
    );
    assert_eq!(scene.statistics(), statistics);
    fixture.assert_chain(scene_texture(&scene, id), &model);
    fixture
        .update(
            &mut alpha_texture,
            region,
            &patch,
            stride,
            Texture3dUpdateBudget::default(),
        )
        .unwrap();
    model.patch(region, &patch, stride);
    fixture.assert_chain(&alpha_texture, &model);
    assert!(
        matches!(opaque.with_texture(&alpha_texture), Err(Texture3dError::NonOpaquePixel { texel }) if texel == first_nonopaque)
    );
    fixture.assert_chain(opaque.texture(), &Model::new(9, 7, false, true));
}

pub(in crate::renderer::mesh3d) fn assert_gpu_partial_mip_updates(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let fixture = Fixture::new(device, queue, format);
    unique_and_aliased_updates(&fixture);
    operation_budget_boundaries(&fixture);
    scene_budget_boundaries(&fixture);
    tile_and_material_rebinding(&fixture);
    invalid_input_and_opacity(&fixture);
}

pub(in crate::renderer::mesh3d) fn assert_gpu_partial_mip_recovery(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let source = Fixture::new(device, queue, format);
    let mut model = Model::new(17, 11, true, true);
    let (mut scene, first, second) = {
        let mut texture = source.texture(&model, true);
        let region = ImageTexelRect::new(13, 8, 4, 3).unwrap();
        let (patch, stride) = padded_patch(region, true, 501);
        source
            .update(
                &mut texture,
                region,
                &patch,
                stride,
                Texture3dUpdateBudget::default(),
            )
            .unwrap();
        model.patch(region, &patch, stride);
        let material = signed_material(&texture);
        let mesh = source.mesh(&material);
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        let style = MeshStyle3d::surface(SurfaceStyle3d::blend(Color::WHITE).unwrap());
        let first = scene.try_push(&mesh, Transform3d::IDENTITY, style).unwrap();
        let second = scene.try_push(&mesh, Transform3d::IDENTITY, style).unwrap();
        scene.set_visible(second, false).unwrap();
        (scene, first, second)
    };
    source.assert_chain(scene_texture(&scene, first), &model);
    let snapshot = scene_texture(&scene, first).clone();
    let replacement = Fixture::new(recovery_device, recovery_queue, format);
    let report = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        &replacement.renderer.textures.layout,
        Arc::clone(&replacement.identity),
        &mut scene,
    )
    .unwrap();
    assert_eq!(report.restored_texture_count(), 1);
    for id in [first, second] {
        replacement.assert_chain(scene_texture(&scene, id), &model);
        assert_policy(scene.instance(id).unwrap().mesh.material().unwrap());
    }
    assert!(!scene.instance(second).unwrap().is_visible());
    assert_eq!(
        restore_scene3d_resources(
            recovery_device,
            recovery_queue,
            &replacement.renderer.textures.layout,
            Arc::clone(&replacement.identity),
            &mut scene
        )
        .unwrap()
        .restored_texture_count(),
        0
    );
    let previous = model.clone();
    let region = ImageTexelRect::new(0, 0, 3, 2).unwrap();
    let (patch, stride) = padded_patch(region, true, 502);
    let old_cpu = scene_texture(&scene, first).recovery_memory_bytes();
    let report = replacement
        .update_scene(&mut scene, first, region, &patch, stride)
        .unwrap();
    model.patch(region, &patch, stride);
    assert_report(report, &model, region, old_cpu, true);
    replacement.assert_chain(scene_texture(&scene, first), &model);
    replacement.assert_chain(scene_texture(&scene, second), &previous);
    source.assert_chain(&snapshot, &previous);
    let old_cpu = scene_texture(&scene, first).recovery_memory_bytes();
    let report = replacement
        .update_scene(&mut scene, first, region, &patch, stride)
        .unwrap();
    assert_report(report, &model, region, old_cpu, false);
    replacement.assert_chain(scene_texture(&scene, first), &model);
}
