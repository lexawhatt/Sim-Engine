//! Independent material-recovery and in-place scene-mutation GPU regressions.

use super::*;
use crate::{Mesh3dAttributes, MeshEdge3d, TextureAddressMode3d, TextureUvTransform3d, Vec2};

const SIZE: u32 = 64;

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

    fn texture(&self, pixels: Vec<u8>, mipmaps: bool) -> Texture3d {
        self.texture_with_alpha_policy(pixels, mipmaps, true)
    }

    fn texture_with_alpha_policy(
        &self,
        pixels: Vec<u8>,
        mipmaps: bool,
        preserve_alpha: bool,
    ) -> Texture3d {
        create_texture_with_options(
            self.device,
            self.queue,
            &self.identity,
            &self.renderer.textures.layout,
            8,
            8,
            pixels,
            ImageBudget::default(),
            Texture3dOptions::new()
                .with_alpha_preservation(preserve_alpha)
                .with_mipmaps(if mipmaps {
                    TextureMipmaps3d::Generate
                } else {
                    TextureMipmaps3d::None
                }),
        )
        .unwrap()
    }

    fn mesh(&self, source: Mesh3d, material: Option<&TextureMaterial3d>) -> RetainedMesh3d {
        let mesh =
            create_retained_mesh(self.device, self.queue, self.identity.clone(), source).unwrap();
        match material {
            Some(material) => attach_material(&self.identity, &mesh, material).unwrap(),
            None => mesh,
        }
    }

    fn restore(&self, source: &RetainedMesh3d) -> RetainedMesh3d {
        restore_retained_mesh(
            self.device,
            self.queue,
            self.identity.clone(),
            &self.renderer.textures.layout,
            source,
        )
        .unwrap()
    }

    fn render(&mut self, scene: &Scene3d, policy: SurfaceRasterization3d) -> Vec<u8> {
        let camera = Camera3d::look_at(
            Vec3::new(0.0, 0.0, 2.0).unwrap(),
            Vec3::ZERO,
            Vec3::Y,
            Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(4.0)).unwrap(),
        )
        .unwrap();
        self.renderer
            .render_scene3d(
                self.device,
                self.queue,
                &self.identity,
                &self.target,
                scene,
                camera,
                Mesh3dRenderBudget::default().with_surface_policy(policy),
            )
            .unwrap();
        surface::test_read_pixels(self.device, self.queue, &self.target)
    }

    fn render_mesh(&mut self, mesh: &RetainedMesh3d, policy: SurfaceRasterization3d) -> Vec<u8> {
        let mut scene = Scene3d::with_alpha_background(Color::TRANSPARENT).unwrap();
        scene
            .try_push(mesh, Transform3d::IDENTITY, blend())
            .unwrap();
        self.render(&scene, policy)
    }

    fn expect_region(&self, pixels: &[u8], rect: [u32; 4], linear: [f64; 4]) {
        let channels = match self.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2, 3],
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0, 3],
            other => panic!("unsupported dev5 oracle target {other:?}"),
        };
        let mut samples = 0;
        for y in rect[1]..rect[3] {
            for x in rect[0]..rect[2] {
                let offset = ((y * SIZE + x) * 4) as usize;
                for channel in 0..4 {
                    let value = linear[channel];
                    let value = if channel < 3 && self.format.is_srgb() {
                        if value <= 0.003_130_8 {
                            value * 12.92
                        } else {
                            1.055 * value.powf(1.0 / 2.4) - 0.055
                        }
                    } else {
                        value
                    };
                    let expected = (value * 255.0).round() as u8;
                    let actual = pixels[offset + channels[channel]];
                    assert!(
                        actual.abs_diff(expected) <= 2,
                        "pixel ({x},{y}), channel {channel}: {actual} != {expected}"
                    );
                }
                samples += 1;
            }
        }
        assert!(samples >= 64);
    }
}

fn blend() -> MeshStyle3d {
    MeshStyle3d::surface(SurfaceStyle3d::blend(Color::WHITE).unwrap())
}

fn quad(rect: [f32; 4], color: Color, textured: bool) -> Mesh3d {
    let [left, top, right, bottom] = rect;
    let vertices = [[left, top], [right, top], [right, bottom], [left, bottom]]
        .map(|[x, y]| Vec3::new(x / 32.0 - 1.0, 1.0 - y / 32.0, 0.0).unwrap())
        .to_vec();
    let mut attributes = Mesh3dAttributes::new()
        .with_vertex_colors(vec![color; 4])
        .unwrap()
        .with_normals(vec![Vec3::Z; 4])
        .unwrap();
    if textured {
        attributes = attributes.with_texture_coordinates(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                .map(|[u, v]| TextureCoordinate2d::new(u, v).unwrap())
                .to_vec(),
        );
    }
    Mesh3d::with_attributes(
        vertices,
        vec![0, 1, 2, 0, 2, 3],
        vec![MeshEdge3d::new(0, 1).unwrap()],
        attributes,
    )
    .unwrap()
}

fn quadrants(side: usize, alpha: bool, generated: bool) -> Vec<u8> {
    let colors = if alpha {
        [
            [255, 0, 0, 128],
            [0, 255, 0, 255],
            if generated {
                [0, 0, 0, 0]
            } else {
                [0, 0, 255, 0]
            },
            [255, 255, 0, 192],
        ]
    } else {
        [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
        ]
    };
    (0..side)
        .flat_map(|y| {
            (0..side).flat_map(move |x| {
                colors[usize::from(y >= side / 2) * 2 + usize::from(x >= side / 2)]
            })
        })
        .collect()
}

fn expected_chain(alpha: bool) -> Vec<Vec<u8>> {
    // Each quadrant is constant until the last level. The final alpha-weighted
    // linear-light mean has a closed form, independent of Engine's mip builder.
    vec![
        quadrants(8, alpha, false),
        quadrants(4, alpha, true),
        quadrants(2, alpha, true),
        if alpha {
            vec![197, 228, 0, 144]
        } else {
            vec![188, 188, 137, 255]
        },
    ]
}

fn read_level(fixture: &Fixture<'_>, texture: &Texture3d, level: u32) -> Vec<u8> {
    let (width, height) = texture.mip_level_size(level).unwrap();
    let stride = (width * 4).div_ceil(256) * 256;
    let buffer = fixture.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dev5 independent restored mip readback"),
        size: u64::from(stride * height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = fixture
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
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
    fixture.queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    fixture
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
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

fn assert_chain(fixture: &Fixture<'_>, texture: &Texture3d, alpha: bool) {
    let expected = expected_chain(alpha);
    assert_eq!(texture.mip_level_count(), 4);
    assert_eq!(texture.gpu_allocation_bytes(), 340);
    for (level, expected) in expected.iter().enumerate() {
        let side = 8 >> level;
        assert_eq!(texture.mip_level_size(level as u32), Some((side, side)));
        assert_eq!(
            texture.mip_level_pixels(level as u32),
            Some(expected.as_slice())
        );
        assert_eq!(read_level(fixture, texture, level as u32), *expected);
    }
}

fn assert_streams_equal(first: &RetainedMesh3d, second: &RetainedMesh3d, same_buffers: bool) {
    assert_eq!(first.source().vertices(), second.source().vertices());
    assert_eq!(
        first.source().triangle_indices(),
        second.source().triangle_indices()
    );
    assert_eq!(
        first.source().display_edges(),
        second.source().display_edges()
    );
    assert_eq!(
        first.source().texture_coordinates(),
        second.source().texture_coordinates()
    );
    assert_eq!(
        first.source().vertex_colors(),
        second.source().vertex_colors()
    );
    assert_eq!(first.source().normals(), second.source().normals());
    assert_eq!(first.gpu_allocation_bytes(), second.gpu_allocation_bytes());
    assert_eq!(first.allocation, second.allocation);
    assert_eq!(first.budget(), second.budget());
    assert_eq!(
        Arc::ptr_eq(&first.vertex_buffer, &second.vertex_buffer),
        same_buffers
    );
    for (first, second) in [
        (&first.index_buffer, &second.index_buffer),
        (&first.edge_buffer, &second.edge_buffer),
        (
            &first.texture_coordinate_buffer,
            &second.texture_coordinate_buffer,
        ),
        (&first.color_buffer, &second.color_buffer),
        (&first.normal_buffer, &second.normal_buffer),
    ] {
        match (first, second) {
            (Some(first), Some(second)) => assert_eq!(Arc::ptr_eq(first, second), same_buffers),
            (None, None) => {}
            _ => panic!("optional geometry stream changed during material operation"),
        }
    }
}

fn reserve_mesh(fixture: &mut Fixture<'_>, mesh: &RetainedMesh3d) -> RetainedMesh3d {
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let object = scene
        .try_push(mesh, Transform3d::IDENTITY, blend())
        .unwrap();
    let upload = Mesh3dUploadBudget::new(16_384, 16_384, 16_384).unwrap();
    let report = dynamic::update_scene_mesh(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &mut fixture.renderer.dynamic_scratch,
        &mut scene,
        object,
        mesh.source().clone(),
        DynamicMesh3dBudget::new(upload).with_minimum_capacity(16, 24, 4),
    )
    .unwrap();
    assert!(report.grew_capacity());
    assert!(report.detached_aliases());
    assert_eq!(report.gpu_allocation_count(), 6);
    let retained = scene.instance(object).unwrap().mesh().clone();
    assert_eq!(retained.budget(), upload);
    assert_eq!(
        retained.vertex_buffer.size(),
        16 * std::mem::size_of::<MeshVertexGpu>() as u64
    );
    assert_eq!(retained.index_buffer.as_ref().unwrap().size(), 24 * 4);
    assert_eq!(
        retained.edge_buffer.as_ref().unwrap().size(),
        4 * std::mem::size_of::<MeshEdgeGpu>() as u64
    );
    assert!(retained.gpu_allocation_bytes() > mesh.gpu_allocation_bytes());
    retained
}

fn assert_standalone_recovery(source: &mut Fixture<'_>, recovered: &mut Fixture<'_>) {
    let transparent = source.texture(quadrants(8, true, false), true);
    let opaque = source.texture(quadrants(8, false, false), true);
    let legacy_opaque = source.texture_with_alpha_policy(quadrants(8, false, false), true, false);
    let replacement = recovered.texture(quadrants(8, true, false), true);
    for sampling in [ImageSampling::Nearest, ImageSampling::Linear] {
        for case in 0..4 {
            let texture = match case {
                0 => &transparent,
                // Alpha-capable material policy is independent of this legacy
                // texture's current opaque-only storage policy.
                3 => &legacy_opaque,
                _ => &opaque,
            };
            let tint = if case == 1 {
                Color::rgba(0.75, 0.5, 1.0, 0.5)
            } else {
                Color::WHITE
            };
            let material = TextureMaterial3d::with_alpha(texture, sampling, tint)
                .unwrap()
                .with_uv_transform(
                    TextureUvTransform3d::new(Vec2::new(-2.0, 3.0), Vec2::new(0.25, -0.125))
                        .unwrap(),
                )
                .with_address_mode(TextureAddressMode3d::Repeat);
            let mesh = source.mesh(
                quad([-8.0, 8.0, 56.0, 56.0], Color::WHITE, true),
                Some(&material),
            );
            let mesh = if case == 0 {
                reserve_mesh(source, &mesh)
            } else {
                mesh
            };
            // This helper is the implementation invoked by public restore_mesh3d,
            // rather than the distinct whole-scene restoration path.
            let restored = recovered.restore(&mesh);
            assert_streams_equal(&mesh, &restored, false);
            let restored_material = restored.material().unwrap();
            assert_eq!(restored_material.tint(), tint);
            assert_eq!(restored_material.sampling(), sampling);
            assert_eq!(restored_material.uv_transform(), material.uv_transform());
            assert_eq!(
                restored_material.address_mode(),
                TextureAddressMode3d::Repeat
            );
            assert_eq!(restored_material.texture().options(), texture.options());
            assert_eq!(restored_material.texture().budget(), texture.budget());
            assert!(restored_material.texture().belongs_to(&recovered.identity));
            assert!(!restored_material.texture().belongs_to(&source.identity));
            assert_chain(source, texture, case == 0);
            assert_chain(recovered, restored_material.texture(), case == 0);
            for policy in [
                SurfaceRasterization3d::Native,
                SurfaceRasterization3d::StrictPortable,
            ] {
                let before = source.render_mesh(&mesh, policy);
                assert!(before.chunks_exact(4).filter(|pixel| pixel[3] > 0).count() > 500);
                assert_eq!(
                    recovered.render_mesh(&restored, policy),
                    before,
                    "case={case}, sampling={sampling:?}, policy={policy:?}"
                );
                if case == 2 {
                    let reset = material
                        .clone()
                        .with_uv_transform(TextureUvTransform3d::IDENTITY)
                        .with_address_mode(TextureAddressMode3d::Clamp);
                    let control = attach_material(&source.identity, &mesh, &reset).unwrap();
                    assert_ne!(
                        source.render_mesh(&control, policy),
                        before,
                        "asymmetric UV fixture must detect reset state"
                    );
                }
            }
            if case == 3 {
                let rebound = restored_material.with_texture(&replacement).unwrap();
                let new_mesh = attach_material(&recovered.identity, &restored, &rebound).unwrap();
                let old_material = material.with_texture(&transparent).unwrap();
                let reference = attach_material(&source.identity, &mesh, &old_material).unwrap();
                assert_eq!(
                    recovered.render_mesh(&new_mesh, SurfaceRasterization3d::Native),
                    source.render_mesh(&reference, SurfaceRasterization3d::Native)
                );
            }
        }
        let legacy = TextureMaterial3d::new(&opaque, sampling, Color::WHITE).unwrap();
        let mesh = source.mesh(
            quad([8.0, 8.0, 56.0, 56.0], Color::WHITE, true),
            Some(&legacy),
        );
        let restored = recovered.restore(&mesh);
        assert!(matches!(
            restored.material().unwrap().with_texture(&replacement),
            Err(Texture3dError::NonOpaquePixel { texel: 0 })
        ));
    }

    let material = TextureMaterial3d::with_alpha(
        &transparent,
        ImageSampling::Linear,
        Color::rgba(1.0, 0.75, 0.5, 0.5),
    )
    .unwrap()
    .with_uv_transform(
        TextureUvTransform3d::new(Vec2::new(-3.0, 2.0), Vec2::new(0.25, -0.125)).unwrap(),
    )
    .with_address_mode(TextureAddressMode3d::Repeat);
    let mesh = source.mesh(
        quad([8.0, 8.0, 28.0, 56.0], Color::WHITE, true),
        Some(&material),
    );
    let mut scene = Scene3d::with_alpha_background(Color::TRANSPARENT).unwrap();
    let first = scene
        .try_push(&mesh, Transform3d::IDENTITY, blend())
        .unwrap();
    let transform = Transform3d::new(
        Vec3::new(0.875, 0.0, 0.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let second = scene.try_push(&mesh, transform, blend()).unwrap();
    let pixels = source.render(&scene, SurfaceRasterization3d::Native);
    let report = restore_scene3d_resources(
        recovered.device,
        recovered.queue,
        &recovered.renderer.textures.layout,
        recovered.identity.clone(),
        &mut scene,
    )
    .unwrap();
    assert_eq!(report.restored_mesh_count(), 1);
    assert_eq!(report.restored_texture_count(), 1);
    assert_eq!(report.restored_texture_bytes(), 340);
    assert_eq!(scene.statistics().texture_count(), 1);
    assert_eq!(scene.instances()[0].id(), first);
    assert_eq!(scene.instances()[1].id(), second);
    assert_eq!(
        scene.instances()[0]
            .mesh()
            .material()
            .unwrap()
            .texture()
            .identity_key(),
        scene.instances()[1]
            .mesh()
            .material()
            .unwrap()
            .texture()
            .identity_key()
    );
    assert_eq!(
        recovered.render(&scene, SurfaceRasterization3d::Native),
        pixels
    );
}

fn assert_background(fixture: &mut Fixture<'_>) {
    let mesh = fixture.mesh(
        quad([8.0, 8.0, 32.0, 56.0], Color::rgb(1.0, 0.0, 0.0), false),
        None,
    );
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let first = scene
        .try_push(&mesh, Transform3d::IDENTITY, blend())
        .unwrap();
    let hidden = scene
        .try_push(&mesh, Transform3d::IDENTITY, blend())
        .unwrap();
    scene.set_visible(hidden, false).unwrap();
    let stats = scene.statistics();
    let lighting = scene.lighting();
    let fog = scene.fog();
    for background in [
        Color::rgba(0.25, 0.5, 0.75, 0.5),
        Color::TRANSPARENT,
        Color::WHITE,
    ] {
        let (result, allocations) =
            crate::test_allocations::count(|| scene.set_background(background));
        result.unwrap();
        assert_eq!(allocations, 0);
        assert_eq!(scene.background(), background);
        assert_eq!(scene.statistics(), stats);
        assert_eq!(scene.lighting(), lighting);
        assert_eq!(scene.fog(), fog);
        assert_eq!(scene.instances()[0].id(), first);
        assert_eq!(scene.instances()[1].id(), hidden);
        assert!(!scene.instance(hidden).unwrap().is_visible());
        assert_streams_equal(scene.instance(first).unwrap().mesh(), &mesh, true);
        let pixels = fixture.render(&scene, SurfaceRasterization3d::Native);
        fixture.expect_region(&pixels, [12, 12, 28, 52], [1.0, 0.0, 0.0, 1.0]);
        let rgba = background.to_array().map(f64::from);
        fixture.expect_region(
            &pixels,
            [40, 12, 56, 52],
            [
                rgba[0] * rgba[3],
                rgba[1] * rgba[3],
                rgba[2] * rgba[3],
                rgba[3],
            ],
        );
        for invalid in [
            Color::rgba(f32::NAN, 0.0, 0.0, 1.0),
            Color::rgba(0.0, f32::INFINITY, 0.0, 1.0),
            Color::rgba(-0.1, 0.0, 0.0, 1.0),
            Color::rgba(0.0, 0.0, 0.0, 1.1),
        ] {
            assert_eq!(
                scene.set_background(invalid),
                Err(Scene3dError::InvalidBackground)
            );
            assert_eq!(scene.background(), background);
            assert_eq!(scene.statistics(), stats);
            assert_eq!(
                fixture.render(&scene, SurfaceRasterization3d::Native),
                pixels
            );
        }
    }
}

fn assert_material_mutation(fixture: &mut Fixture<'_>, recovered: &mut Fixture<'_>) {
    let red = fixture.texture([255, 0, 0, 255].repeat(64), false);
    let green = fixture.texture([0, 255, 0, 255].repeat(64), false);
    let red_material = TextureMaterial3d::new(&red, ImageSampling::Nearest, Color::WHITE).unwrap();
    let green_material =
        TextureMaterial3d::new(&green, ImageSampling::Nearest, Color::WHITE).unwrap();
    let mesh = fixture.mesh(
        quad([8.0, 8.0, 28.0, 56.0], Color::rgb(1.0, 0.5, 0.25), true),
        Some(&red_material),
    );
    let mesh = reserve_mesh(fixture, &mesh);
    let (detached, allocations) = crate::test_allocations::count(|| mesh.without_material());
    assert_eq!(allocations, 0);
    assert!(detached.material().is_none());
    assert!(mesh.material().is_some());
    assert_streams_equal(&mesh, &detached, true);
    let detached_pixels = fixture.render_mesh(&detached, SurfaceRasterization3d::Native);
    fixture.expect_region(&detached_pixels, [12, 12, 24, 52], [1.0, 0.5, 0.25, 1.0]);
    let restored = recovered.restore(&detached);
    assert!(restored.material().is_none());
    assert_streams_equal(&detached, &restored, false);
    assert_eq!(
        recovered.render_mesh(&restored, SurfaceRasterization3d::Native),
        detached_pixels
    );

    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let first = scene
        .try_push(&mesh, Transform3d::IDENTITY, blend())
        .unwrap();
    let transform = Transform3d::new(
        Vec3::new(0.875, 0.0, 0.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let second = scene.try_push(&mesh, transform, blend()).unwrap();
    let original = fixture.render(&scene, SurfaceRasterization3d::Native);
    let original_stats = scene.statistics();
    scene
        .set_texture_material(first, Some(&green_material))
        .unwrap();
    scene
        .set_texture_material(first, Some(&red_material))
        .unwrap();
    for material in [Some(&green_material), Some(&red_material), None] {
        let (report, allocations) =
            crate::test_allocations::count(|| scene.set_texture_material(first, material));
        let report = report.unwrap();
        assert_eq!(
            allocations, 0,
            "warmed material rebind allocated CPU storage"
        );
        assert_eq!(report.statistics().mesh_count(), 1);
        assert_eq!(
            report.statistics().mesh_cpu_bytes(),
            original_stats.mesh_cpu_bytes()
        );
        assert_eq!(
            report.statistics().mesh_gpu_bytes(),
            original_stats.mesh_gpu_bytes()
        );
        assert_eq!(
            report.peak_mesh_gpu_bytes(),
            original_stats.mesh_gpu_bytes()
        );
        assert_eq!(
            scene.instance(first).unwrap().transform(),
            Transform3d::IDENTITY
        );
        assert_eq!(scene.instance(first).unwrap().style(), blend());
        assert_eq!(scene.instance(second).unwrap().transform(), transform);
        assert_streams_equal(scene.instance(first).unwrap().mesh(), &mesh, true);
        assert_streams_equal(scene.instance(second).unwrap().mesh(), &mesh, true);
        assert_eq!(
            scene
                .instance(second)
                .unwrap()
                .mesh()
                .material()
                .unwrap()
                .texture()
                .identity_key(),
            red.identity_key()
        );
        let pixels = fixture.render(&scene, SurfaceRasterization3d::Native);
        let expected = match material {
            Some(material) if material.texture().identity_key() == green.identity_key() => {
                [0.0, 0.5, 0.0, 1.0]
            }
            Some(_) => [1.0, 0.0, 0.0, 1.0],
            None => [1.0, 0.5, 0.25, 1.0],
        };
        fixture.expect_region(&pixels, [12, 12, 24, 52], expected);
        fixture.expect_region(&pixels, [40, 12, 52, 52], [1.0, 0.0, 0.0, 1.0]);
        for y in 0..SIZE as usize {
            let start = (y * SIZE as usize + 32) * 4;
            let end = (y + 1) * SIZE as usize * 4;
            assert_eq!(
                &pixels[start..end],
                &original[start..end],
                "second alias/background changed at row {y}"
            );
        }
    }
    scene.set_texture_material(second, None).unwrap();
    assert_eq!(scene.statistics().texture_count(), 0);
    assert_eq!(scene.statistics().texture_cpu_bytes(), 0);
    assert_eq!(scene.statistics().texture_gpu_bytes(), 0);
    assert_eq!(
        mesh.material().unwrap().texture().identity_key(),
        red.identity_key()
    );
    assert_rebinding_failures(fixture, recovered, &mesh, &red_material, &green_material);
}

fn assert_rebinding_failures(
    fixture: &mut Fixture<'_>,
    recovered: &Fixture<'_>,
    mesh: &RetainedMesh3d,
    red: &TextureMaterial3d,
    green: &TextureMaterial3d,
) {
    let red_texture = red.texture();
    let green_texture = green.texture();
    // Limits apply to the accepted scene, not temporary old/new overlap.
    // A sole-owner A->B replacement fits exactly while another A reference
    // below makes the same replacement exceed the final limit.
    let final_budget = Scene3dBudget::default().with_texture_limits(
        green_texture.recovery_memory_bytes(),
        green_texture.gpu_allocation_bytes(),
    );
    let mut sole = Scene3d::with_budget(Color::BLACK, final_budget).unwrap();
    let sole_object = sole.try_push(mesh, Transform3d::IDENTITY, blend()).unwrap();
    let before = fixture.render(&sole, SurfaceRasterization3d::Native);
    let report = sole.set_texture_material(sole_object, Some(green)).unwrap();
    assert_eq!(report.statistics().texture_count(), 1);
    assert_eq!(
        report.statistics().texture_cpu_bytes(),
        green_texture.recovery_memory_bytes()
    );
    assert_eq!(
        report.statistics().texture_gpu_bytes(),
        green_texture.gpu_allocation_bytes()
    );
    assert_eq!(
        report.peak_texture_cpu_bytes(),
        red_texture.recovery_memory_bytes() + green_texture.recovery_memory_bytes()
    );
    assert_eq!(
        report.peak_texture_gpu_bytes(),
        red_texture.gpu_allocation_bytes() + green_texture.gpu_allocation_bytes()
    );
    let after = fixture.render(&sole, SurfaceRasterization3d::Native);
    assert_ne!(after, before);
    fixture.expect_region(&after, [12, 12, 24, 52], [0.0, 0.5, 0.0, 1.0]);
    assert_eq!(
        mesh.material().unwrap().texture().identity_key(),
        red_texture.identity_key()
    );
    for resource in [
        Scene3dBudgetResource::TextureCpuBytes,
        Scene3dBudgetResource::TextureGpuBytes,
    ] {
        let cpu_sum = red_texture.recovery_memory_bytes() + green_texture.recovery_memory_bytes();
        let gpu_sum = red_texture.gpu_allocation_bytes() + green_texture.gpu_allocation_bytes();
        let budget = Scene3dBudget::default().with_texture_limits(
            cpu_sum - usize::from(resource == Scene3dBudgetResource::TextureCpuBytes),
            gpu_sum - usize::from(resource == Scene3dBudgetResource::TextureGpuBytes),
        );
        let mut scene = Scene3d::with_budget(Color::BLACK, budget).unwrap();
        let object = scene
            .try_push(mesh, Transform3d::IDENTITY, blend())
            .unwrap();
        scene
            .try_push(mesh, Transform3d::IDENTITY, blend())
            .unwrap();
        let stats = scene.statistics();
        let before = fixture.render(&scene, SurfaceRasterization3d::Native);
        let error = scene.set_texture_material(object, Some(green)).unwrap_err();
        assert!(
            matches!(error, Scene3dError::BudgetExceeded { resource: actual, .. } if actual == resource)
        );
        assert_eq!(scene.statistics(), stats);
        assert_streams_equal(scene.instance(object).unwrap().mesh(), mesh, true);
        assert_eq!(
            scene
                .instance(object)
                .unwrap()
                .mesh()
                .material()
                .unwrap()
                .texture()
                .identity_key(),
            red_texture.identity_key()
        );
        assert_eq!(
            fixture.render(&scene, SurfaceRasterization3d::Native),
            before
        );
    }

    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let object = scene
        .try_push(mesh, Transform3d::IDENTITY, blend())
        .unwrap();
    let removed = scene
        .try_push(mesh, Transform3d::IDENTITY, blend())
        .unwrap();
    scene.remove(removed).unwrap();
    let mut other = Scene3d::new(Color::BLACK).unwrap();
    let foreign_id = other
        .try_push(mesh, Transform3d::IDENTITY, blend())
        .unwrap();
    let foreign_texture = recovered.texture([255, 255, 255, 255].repeat(64), false);
    let foreign_material =
        TextureMaterial3d::new(&foreign_texture, ImageSampling::Nearest, Color::WHITE).unwrap();
    let stats = scene.statistics();
    let before = fixture.render(&scene, SurfaceRasterization3d::Native);
    for invalid_id in [removed, foreign_id] {
        for material in [Some(green), None] {
            assert!(
                matches!(scene.set_texture_material(invalid_id, material), Err(Scene3dError::ObjectNotFound { object_id }) if object_id == invalid_id)
            );
            assert_eq!(scene.statistics(), stats);
            assert_eq!(
                fixture.render(&scene, SurfaceRasterization3d::Native),
                before
            );
        }
    }
    assert!(matches!(
        scene.set_texture_material(object, Some(&foreign_material)),
        Err(Scene3dError::Texture(Texture3dError::RendererMismatch))
    ));
    assert_eq!(scene.statistics(), stats);
    assert_eq!(
        fixture.render(&scene, SurfaceRasterization3d::Native),
        before
    );

    let no_uv = fixture.mesh(quad([8.0, 8.0, 56.0, 56.0], Color::WHITE, false), None);
    let mut plain_scene = Scene3d::new(Color::BLACK).unwrap();
    let plain_object = plain_scene
        .try_push(&no_uv, Transform3d::IDENTITY, blend())
        .unwrap();
    let stats = plain_scene.statistics();
    let before = fixture.render(&plain_scene, SurfaceRasterization3d::Native);
    assert!(matches!(
        plain_scene.set_texture_material(plain_object, Some(green)),
        Err(Scene3dError::Texture(
            Texture3dError::MissingTextureCoordinates
        ))
    ));
    assert_eq!(plain_scene.statistics(), stats);
    assert_eq!(
        fixture.render(&plain_scene, SurfaceRasterization3d::Native),
        before
    );
}

pub(in crate::renderer::mesh3d) fn assert_dev5_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let mut source = Fixture::new(device, queue, format);
    let mut recovered = Fixture::new(recovery_device, recovery_queue, format);
    assert_standalone_recovery(&mut source, &mut recovered);
    assert_background(&mut source);
    assert_material_mutation(&mut source, &mut recovered);
    eprintln!(
        "sim-engine dev5: standalone alpha/tint/UV/policy recovery, exact mip readback, background and atomic material detach/rebind passed"
    );
}
