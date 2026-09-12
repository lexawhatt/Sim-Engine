//! Independent pixel expectations for alpha, sidedness and pass ordering.

use super::*;
use crate::{Mesh3dAttributes, SurfaceSidedness3d, TextureCoordinate2d};

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

    fn mesh(
        &self,
        perspective: bool,
        crossing: bool,
        reverse: bool,
        color: Option<Color>,
        material: Option<&TextureMaterial3d>,
    ) -> RetainedMesh3d {
        let scale = if perspective { 3.0 } else { 1.0 };
        let left = if crossing { -1.5 } else { -0.8 };
        let vertices = [[left, -0.7], [0.8, -0.7], [0.8, 0.7], [left, 0.7]]
            .map(|[x, y]| Vec3::new(x * scale, y * scale, 0.0).unwrap());
        let mut attributes = Mesh3dAttributes::new();
        if let Some(color) = color {
            attributes = attributes.with_vertex_colors(vec![color; 4]).unwrap();
        }
        if material.is_some() {
            attributes = attributes.with_texture_coordinates(
                [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                    .map(|[u, v]| TextureCoordinate2d::new(u, v).unwrap())
                    .to_vec(),
            );
        }
        let source = Mesh3d::with_attributes(
            vertices.to_vec(),
            if reverse {
                vec![0, 2, 1, 0, 3, 2]
            } else {
                vec![0, 1, 2, 0, 2, 3]
            },
            Vec::new(),
            attributes,
        )
        .unwrap();
        let mesh =
            create_retained_mesh(self.device, self.queue, self.identity.clone(), source).unwrap();
        material.map_or_else(
            || mesh.clone(),
            |material| texture::attach_material(&self.identity, &mesh, material).unwrap(),
        )
    }

    fn material(&self, width: u32, pixels: Vec<u8>, tint: Color) -> TextureMaterial3d {
        let texture = texture::create_texture_with_alpha(
            self.device,
            self.queue,
            &self.identity,
            &self.renderer.textures.layout,
            width,
            1,
            pixels,
            ImageBudget::default(),
        )
        .unwrap();
        TextureMaterial3d::with_alpha(&texture, ImageSampling::Nearest, tint).unwrap()
    }

    fn render(
        &mut self,
        scene: &Scene3d,
        perspective: bool,
        policy: SurfaceRasterization3d,
    ) -> Vec<u8> {
        let ids: Vec<_> = scene.instances().iter().map(Mesh3dInstance::id).collect();
        self.renderer
            .render_scene3d(
                self.device,
                self.queue,
                &self.identity,
                &self.target,
                scene,
                camera(perspective),
                Mesh3dRenderBudget::default().with_surface_policy(policy),
            )
            .unwrap();
        assert_eq!(
            scene
                .instances()
                .iter()
                .map(Mesh3dInstance::id)
                .collect::<Vec<_>>(),
            ids
        );
        surface::test_read_pixels(self.device, self.queue, &self.target)
    }

    fn expect(&self, pixels: &[u8], x: u32, y: u32, premultiplied: [f64; 4]) {
        for py in y..y + 2 {
            for px in x..x + 2 {
                self.expect_pixel(pixels, px, py, premultiplied);
            }
        }
    }

    fn expect_pixel(&self, pixels: &[u8], px: u32, py: u32, premultiplied: [f64; 4]) {
        let channels = match self.format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0, 3],
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2, 3],
            other => panic!("unsupported material oracle format {other:?}"),
        };
        let offset = ((py * SIZE + px) * 4) as usize;
        for channel in 0..4 {
            let linear = premultiplied[channel];
            let encoded = if channel < 3 && self.format.is_srgb() {
                if linear <= 0.003_130_8 {
                    12.92 * linear
                } else {
                    1.055 * linear.powf(1.0 / 2.4) - 0.055
                }
            } else {
                linear
            };
            let expected = (encoded * 255.0).round() as u8;
            let actual = pixels[offset + channels[channel]];
            assert!(
                actual.abs_diff(expected) <= 3,
                "material pixel ({px},{py}), channel {channel}: {actual}, expected {expected}; linear={premultiplied:?}"
            );
        }
    }
}

fn camera(perspective: bool) -> Camera3d {
    let projection = if perspective {
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(0.5), world(8.0))
    } else {
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(8.0))
    }
    .unwrap();
    Camera3d::look_at(
        Vec3::new(0.0, 0.0, 3.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        projection,
    )
    .unwrap()
}

fn transform(z: f32) -> Transform3d {
    Transform3d::new(
        Vec3::new(0.0, 0.0, z).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap()
}

fn add(scene: &mut Scene3d, mesh: &RetainedMesh3d, z: f32, style: SurfaceStyle3d) {
    scene
        .try_push(mesh, transform(z), MeshStyle3d::surface(style))
        .unwrap();
}

pub(super) fn assert_gpu_material_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let mut fixture = Fixture::new(device, queue, format);
    for perspective in [false, true] {
        for crossing in [false, true] {
            for policy in [
                SurfaceRasterization3d::StrictPortable,
                SurfaceRasterization3d::Native,
            ] {
                assert_sorted_blending(&mut fixture, perspective, crossing, policy);
                assert_mask_depth(&mut fixture, perspective, crossing, policy);
                assert_alpha_product_and_sidedness(&mut fixture, perspective, crossing, policy);
            }
        }
    }
    assert_camera_relative_sort(&mut fixture);
    assert_varying_alpha(&mut fixture);
    assert_edge_visibility(&mut fixture);
    assert_background_and_composition(&mut fixture);
}

fn assert_varying_alpha(fixture: &mut Fixture<'_>) {
    let orient = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    for perspective in [false, true] {
        let camera = camera(perspective);
        for crossing in [false, true] {
            let scale = if perspective { 3.0 } else { 1.0 };
            let positions = [
                [if crossing { -2.2 } else { -0.8 }, -0.6, 0.5],
                [0.8, -0.6, -0.5],
                [0.0, 0.8, 0.0],
            ]
            .map(|[x, y, z]| Vec3::new(x * scale, y * scale, z).unwrap());
            let colors = [0.0, 0.5, 1.0].map(|alpha| Color::rgba(1.0, 0.0, 0.0, alpha));
            let source = Mesh3d::with_attributes(
                positions.to_vec(),
                vec![0, 1, 2],
                Vec::new(),
                Mesh3dAttributes::new()
                    .with_vertex_colors(colors.to_vec())
                    .unwrap(),
            )
            .unwrap();
            let mesh = create_retained_mesh(
                fixture.device,
                fixture.queue,
                fixture.identity.clone(),
                source,
            )
            .unwrap();
            let mut scene =
                Scene3d::with_alpha_background(Color::rgba(0.0, 0.0, 0.0, 0.0)).unwrap();
            add(
                &mut scene,
                &mesh,
                0.0,
                SurfaceStyle3d::blend(Color::rgba(1.0, 1.0, 1.0, 0.75)).unwrap(),
            );
            let projected = positions.map(|position| {
                camera
                    .project_world(
                        position,
                        LogicalViewport::new(SIZE as f32, SIZE as f32).unwrap(),
                    )
                    .unwrap()
            });
            let points = projected.map(|point| {
                let position = point.logical_position().to_vec2();
                [f64::from(position.x()), f64::from(position.y())]
            });
            let area = orient(points[0], points[1], points[2]);
            for policy in [
                SurfaceRasterization3d::StrictPortable,
                SurfaceRasterization3d::Native,
            ] {
                let pixels = fixture.render(&scene, perspective, policy);
                let mut samples = 0;
                for y in 2..SIZE - 2 {
                    for x in 2..SIZE - 2 {
                        let center = [x as f64 + 0.5, y as f64 + 0.5];
                        let mut weights = [
                            orient(points[1], points[2], center) / area,
                            orient(points[2], points[0], center) / area,
                            orient(points[0], points[1], center) / area,
                        ];
                        if weights.iter().any(|weight| *weight < 0.07) {
                            continue;
                        }
                        if perspective {
                            for (index, weight) in weights.iter_mut().enumerate() {
                                *weight /= f64::from(projected[index].view_depth());
                            }
                            let sum: f64 = weights.iter().sum();
                            weights.iter_mut().for_each(|weight| *weight /= sum);
                        }
                        let alpha = (weights[1] * 0.5 + weights[2]) * 0.75;
                        fixture.expect_pixel(&pixels, x, y, [alpha, 0.0, 0.0, alpha]);
                        samples += 1;
                    }
                }
                assert!(
                    samples > 128,
                    "insufficient independent alpha samples: {samples}"
                );
            }
        }
    }
}

fn assert_camera_relative_sort(fixture: &mut Fixture<'_>) {
    for perspective in [false, true] {
        let mesh = fixture.mesh(perspective, true, false, None, None);
        for radians in [0.0, 1.3, 3.0] {
            let rotation = Rotation3d::from_axis_angle(Vec3::Y, radians).unwrap();
            let camera = Camera3d::look_at(
                rotation.rotate(Vec3::new(0.0, 0.0, 3.0).unwrap()).unwrap(),
                Vec3::ZERO,
                Vec3::Y,
                camera(perspective).projection(),
            )
            .unwrap();
            let mut scene = Scene3d::new(Color::BLACK).unwrap();
            // Same model transformed independently. At the last angle the
            // world-Z ordering is reversed but camera-forward ordering is not.
            for (z, color) in [
                (0.5, Color::rgba(1.0, 0.0, 0.0, 0.5)),
                (-0.5, Color::rgba(0.0, 0.0, 1.0, 0.5)),
            ] {
                let transform = Transform3d::new(
                    rotation.rotate(Vec3::new(0.0, 0.0, z).unwrap()).unwrap(),
                    rotation,
                    Vec3::new(1.0, 1.0, 1.0).unwrap(),
                )
                .unwrap();
                scene
                    .try_push(
                        &mesh,
                        transform,
                        MeshStyle3d::surface(SurfaceStyle3d::blend(color).unwrap()),
                    )
                    .unwrap();
            }
            for policy in [
                SurfaceRasterization3d::StrictPortable,
                SurfaceRasterization3d::Native,
            ] {
                fixture
                    .renderer
                    .render_scene3d(
                        fixture.device,
                        fixture.queue,
                        &fixture.identity,
                        &fixture.target,
                        &scene,
                        camera,
                        Mesh3dRenderBudget::default().with_surface_policy(policy),
                    )
                    .unwrap();
                let pixels =
                    surface::test_read_pixels(fixture.device, fixture.queue, &fixture.target);
                fixture.expect(&pixels, 31, 28, [0.5, 0.0, 0.25, 1.0]);
            }
        }
    }
}

fn assert_edge_visibility(fixture: &mut Fixture<'_>) {
    let edge = Mesh3d::with_display_edges(
        vec![
            Vec3::new(-0.75, 0.0, 0.0).unwrap(),
            Vec3::new(0.75, 0.0, 0.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let edge = create_retained_mesh(
        fixture.device,
        fixture.queue,
        fixture.identity.clone(),
        edge,
    )
    .unwrap();
    let material = fixture.material(2, vec![0, 255, 0, 0, 0, 255, 0, 255], Color::WHITE);
    let face = fixture.mesh(false, false, false, None, Some(&material));
    for policy in [
        SurfaceRasterization3d::StrictPortable,
        SurfaceRasterization3d::Native,
    ] {
        for masked in [false, true] {
            let mut scene = Scene3d::new(Color::BLACK).unwrap();
            scene
                .try_push(
                    &edge,
                    Transform3d::IDENTITY,
                    MeshStyle3d::wireframe(
                        WireframeStyle3d::visible(Color::WHITE, logical(4.0)).unwrap(),
                    ),
                )
                .unwrap();
            add(
                &mut scene,
                &face,
                0.5,
                if masked {
                    SurfaceStyle3d::mask(Color::WHITE, 0.5).unwrap()
                } else {
                    SurfaceStyle3d::blend(Color::rgba(1.0, 1.0, 1.0, 0.5)).unwrap()
                },
            );
            let pixels = fixture.render(&scene, false, policy);
            fixture.expect(&pixels, 16, 31, [1.0, 1.0, 1.0, 1.0]);
            fixture.expect(
                &pixels,
                46,
                31,
                if masked {
                    [0.0, 1.0, 0.0, 1.0]
                } else {
                    [1.0, 1.0, 1.0, 1.0]
                },
            );
        }
    }
}

fn assert_sorted_blending(
    fixture: &mut Fixture<'_>,
    perspective: bool,
    crossing: bool,
    policy: SurfaceRasterization3d,
) {
    // Near red must be over far blue regardless of submission order. Opaque
    // green is behind both and must execute first even when inserted last.
    for reverse in [false, true] {
        let plain = fixture.mesh(perspective, false, false, None, None);
        let colored = fixture.mesh(
            perspective,
            crossing,
            true,
            Some(Color::rgba(1.0, 1.0, 1.0, 0.5)),
            None,
        );
        let mut scene = Scene3d::with_alpha_background(Color::rgba(0.0, 0.0, 0.0, 0.0)).unwrap();
        for step in 0..2 {
            if (step == 0) != reverse {
                add(
                    &mut scene,
                    &colored,
                    0.5,
                    SurfaceStyle3d::blend(Color::rgb(1.0, 0.0, 0.0)).unwrap(),
                );
            } else {
                add(
                    &mut scene,
                    &plain,
                    -0.5,
                    SurfaceStyle3d::blend(Color::rgba(0.0, 0.0, 1.0, 0.5)).unwrap(),
                );
            }
        }
        // Hidden entries must not shift visible uniform/clipped-range indices.
        let hidden = scene
            .try_push(
                &plain,
                transform(1.0),
                MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
            )
            .unwrap();
        scene.set_visible(hidden, false).unwrap();
        let pixels = fixture.render(&scene, perspective, policy);
        fixture.expect(&pixels, 31, 28, [0.5, 0.0, 0.25, 0.75]);
        add(
            &mut scene,
            &plain,
            -1.0,
            SurfaceStyle3d::opaque(Color::rgb(0.0, 1.0, 0.0)).unwrap(),
        );
        let pixels = fixture.render(&scene, perspective, policy);
        fixture.expect(&pixels, 31, 28, [0.5, 0.25, 0.25, 1.0]);
        // Opaque nearer than both Blend objects must suppress both completely.
        add(
            &mut scene,
            &plain,
            1.0,
            SurfaceStyle3d::opaque(Color::rgb(0.0, 1.0, 0.0)).unwrap(),
        );
        let pixels = fixture.render(&scene, perspective, policy);
        fixture.expect(&pixels, 31, 28, [0.0, 1.0, 0.0, 1.0]);
    }
    // Coplanar Blend ties use insertion order and neither writes depth.
    let mesh = fixture.mesh(perspective, crossing, false, None, None);
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    add(
        &mut scene,
        &mesh,
        0.0,
        SurfaceStyle3d::blend(Color::rgba(1.0, 0.0, 0.0, 0.5)).unwrap(),
    );
    add(
        &mut scene,
        &mesh,
        0.0,
        SurfaceStyle3d::blend(Color::rgba(0.0, 0.0, 1.0, 0.5)).unwrap(),
    );
    let pixels = fixture.render(&scene, perspective, policy);
    fixture.expect(&pixels, 31, 28, [0.25, 0.0, 0.5, 1.0]);
}

fn assert_mask_depth(
    fixture: &mut Fixture<'_>,
    perspective: bool,
    crossing: bool,
    policy: SurfaceRasterization3d,
) {
    let material = fixture.material(2, vec![0, 255, 0, 0, 0, 255, 0, 255], Color::WHITE);
    let cutout = fixture.mesh(perspective, crossing, false, None, Some(&material));
    let plain = fixture.mesh(perspective, false, false, None, None);
    for reverse in [false, true] {
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        for step in 0..2 {
            if (step == 0) != reverse {
                add(
                    &mut scene,
                    &cutout,
                    0.5,
                    SurfaceStyle3d::mask(Color::WHITE, 0.5).unwrap(),
                );
            } else {
                add(
                    &mut scene,
                    &plain,
                    -0.5,
                    SurfaceStyle3d::opaque(Color::rgb(1.0, 0.0, 0.0)).unwrap(),
                );
            }
        }
        add(
            &mut scene,
            &plain,
            0.0,
            SurfaceStyle3d::blend(Color::rgba(0.0, 0.0, 1.0, 0.5)).unwrap(),
        );
        let pixels = fixture.render(&scene, perspective, policy);
        fixture.expect(&pixels, 12, 28, [0.5, 0.0, 0.5, 1.0]);
        fixture.expect(&pixels, 48, 28, [0.0, 1.0, 0.0, 1.0]);
    }
    // Exactly representable cutoff control, plus either side of the boundary.
    let mesh = fixture.mesh(
        perspective,
        crossing,
        false,
        Some(Color::rgba(1.0, 0.0, 0.0, 0.5)),
        None,
    );
    for (cutoff, red) in [(0.25, 1.0), (0.5, 1.0), (0.75, 0.0)] {
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        add(
            &mut scene,
            &mesh,
            0.0,
            SurfaceStyle3d::mask(Color::WHITE, cutoff).unwrap(),
        );
        let pixels = fixture.render(&scene, perspective, policy);
        fixture.expect(&pixels, 31, 28, [red, 0.0, 0.0, 1.0]);
    }
    for (alpha, cutoff, visible) in [(0.0, 0.0, true), (0.0, 1.0, false), (1.0, 1.0, true)] {
        let mesh = fixture.mesh(
            perspective,
            crossing,
            false,
            Some(Color::rgba(1.0, 0.0, 0.0, alpha)),
            None,
        );
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        add(
            &mut scene,
            &mesh,
            0.0,
            SurfaceStyle3d::mask(Color::WHITE, cutoff).unwrap(),
        );
        let pixels = fixture.render(&scene, perspective, policy);
        fixture.expect(
            &pixels,
            31,
            28,
            [if visible { 1.0 } else { 0.0 }, 0.0, 0.0, 1.0],
        );
    }
}

fn assert_alpha_product_and_sidedness(
    fixture: &mut Fixture<'_>,
    perspective: bool,
    crossing: bool,
    policy: SurfaceRasterization3d,
) {
    let material = fixture.material(1, vec![255, 255, 255, 128], Color::rgba(1.0, 1.0, 1.0, 0.5));
    for reverse in [false, true] {
        for textured in [false, true] {
            let mesh = fixture.mesh(
                perspective,
                crossing,
                reverse,
                Some(Color::rgba(1.0, 1.0, 1.0, 0.5)),
                textured.then_some(&material),
            );
            for sidedness in [SurfaceSidedness3d::TwoSided, SurfaceSidedness3d::FrontOnly] {
                for mode in 0..4 {
                    let style = match mode {
                        0 => SurfaceStyle3d::opaque(Color::rgb(1.0, 0.0, 0.0)),
                        1 => SurfaceStyle3d::mask(Color::rgba(1.0, 0.0, 0.0, 0.5), 0.05),
                        2 => SurfaceStyle3d::blend(Color::rgba(1.0, 0.0, 0.0, 0.5)),
                        _ => SurfaceStyle3d::mask(
                            Color::rgba(1.0, 0.0, 0.0, 0.5),
                            if textured { 0.1 } else { 0.3 },
                        ),
                    }
                    .unwrap()
                    .with_sidedness(sidedness);
                    let mut scene =
                        Scene3d::with_alpha_background(Color::rgba(0.0, 0.0, 0.0, 0.0)).unwrap();
                    add(&mut scene, &mesh, 0.0, style);
                    let pixels = fixture.render(&scene, perspective, policy);
                    let culled = reverse && sidedness == SurfaceSidedness3d::FrontOnly;
                    let alpha = if culled || mode == 3 {
                        0.0
                    } else if mode == 2 {
                        0.5 * 0.5 * if textured { (128.0 / 255.0) * 0.5 } else { 1.0 }
                    } else {
                        1.0
                    };
                    fixture.expect(&pixels, 31, 28, [alpha, 0.0, 0.0, alpha]);
                }
            }
        }
    }
}

fn assert_background_and_composition(fixture: &mut Fixture<'_>) {
    let mut scene = Scene3d::with_alpha_background(Color::rgba(0.0, 1.0, 0.0, 0.25)).unwrap();
    let mesh = fixture.mesh(false, false, false, None, None);
    add(
        &mut scene,
        &mesh,
        0.0,
        SurfaceStyle3d::blend(Color::rgba(1.0, 0.0, 0.0, 0.5)).unwrap(),
    );
    let pixels = fixture.render(&scene, false, SurfaceRasterization3d::Native);
    fixture.expect(&pixels, 31, 28, [0.5, 0.125, 0.0, 0.625]);
    fixture.expect(&pixels, 1, 1, [0.0, 0.25, 0.0, 0.25]);

    // Feed the actual rendered texels through production target composition.
    // Copying only adds TEXTURE_BINDING usage; it does not decode/rebuild pixels.
    let sampled = fixture.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("3D alpha composition source"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fixture.format,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = sampled.create_view(&wgpu::TextureViewDescriptor::default());
    let pipelines = create_pipeline(fixture.device, fixture.format, 1).target_composition_pipelines;
    fixture.queue.write_buffer(
        &pipelines.uniform_buffer,
        0,
        bytemuck::bytes_of(&CompositeUniform::full_surface(1.0)),
    );
    let group = fixture
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("3D alpha composition"),
            layout: &pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pipelines.uniform_buffer.as_entire_binding(),
                },
            ],
        });
    let destination = surface::test_target(
        fixture.device,
        &fixture.identity,
        fixture.format,
        SIZE,
        SIZE,
    );
    let mut encoder = fixture
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &fixture.target.color._texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &sampled,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("3D alpha composed over blue"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &destination.color.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(Color::rgb(0.0, 0.0, 1.0).to_wgpu()),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipelines.alpha);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..6, 0..1);
    }
    fixture.queue.submit([encoder.finish()]);
    let pixels = surface::test_read_pixels(fixture.device, fixture.queue, &destination);
    fixture.expect(&pixels, 31, 28, [0.5, 0.125, 0.375, 1.0]);
    fixture.expect(&pixels, 1, 1, [0.0, 0.25, 0.75, 1.0]);
}
