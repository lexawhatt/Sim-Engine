//! Analytic pixel oracles, independent of the runtime lighting/fog helpers.

use super::*;
use crate::{
    AmbientLight3d, DirectionalLight3d, Fog3d, Lighting3d, Mesh3dAttributes, SurfaceLighting3d,
    TextureCoordinate2d,
};

const SIZE: u32 = 64;
const BASE: [f64; 3] = [0.4, 0.3, 0.2];
const FOG: [f64; 3] = [0.1, 0.2, 0.5];

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

    fn upload(&self, source: Mesh3d) -> RetainedMesh3d {
        create_retained_mesh(self.device, self.queue, self.identity.clone(), source).unwrap()
    }

    fn render(
        &mut self,
        scene: &Scene3d,
        camera: Camera3d,
        policy: SurfaceRasterization3d,
    ) -> Vec<u8> {
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

    fn expect(&self, pixels: &[u8], x: u32, y: u32, rgba: [f64; 4]) {
        let channels = match self.format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0, 3],
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2, 3],
            other => panic!("unsupported lighting oracle format {other:?}"),
        };
        let offset = ((y * SIZE + x) * 4) as usize;
        for channel in 0..4 {
            let linear = rgba[channel];
            let encoded = if channel < 3 && self.format.is_srgb() {
                if linear <= 0.003_130_8 {
                    linear * 12.92
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
                "lighting/fog pixel ({x},{y}), channel {channel}: {actual}, expected {expected}; linear={rgba:?}"
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

fn lighting() -> Lighting3d {
    Lighting3d::new(AmbientLight3d::new(Color::rgb(1.0, 0.5, 0.25), 0.2).unwrap()).with_directional(
        Some(DirectionalLight3d::new(Vec3::Z, Color::rgb(0.5, 1.0, 0.75), 0.6).unwrap()),
    )
}

fn fog(start: f32, density: f32) -> Fog3d {
    Fog3d::new(Color::rgb(0.1, 0.2, 0.5), start, density).unwrap()
}

fn nonuniform_transform() -> Transform3d {
    Transform3d::new(
        Vec3::ZERO,
        Rotation3d::IDENTITY,
        Vec3::new(2.0, 1.0, 0.5).unwrap(),
    )
    .unwrap()
}

fn quad(perspective: bool, crossing: bool, reverse: bool, normal: Option<Vec3>) -> Mesh3d {
    let scale = if perspective { 3.0 } else { 1.0 };
    let left = if crossing { -1.5 } else { -0.8 };
    let vertices = [[left, -0.7], [0.8, -0.7], [0.8, 0.7], [left, 0.7]]
        .map(|[x, y]| Vec3::new(x * scale / 2.0, y * scale, 0.0).unwrap())
        .to_vec();
    let mut attributes = Mesh3dAttributes::new()
        .with_vertex_colors(vec![Color::rgba(0.8, 0.6, 0.4, 0.5); 4])
        .unwrap();
    if let Some(normal) = normal {
        attributes = attributes.with_normals(vec![normal; 4]).unwrap();
    }
    Mesh3d::with_attributes(
        vertices,
        if reverse {
            vec![0, 2, 1, 0, 3, 2]
        } else {
            vec![0, 1, 2, 0, 2, 3]
        },
        vec![],
        attributes,
    )
    .unwrap()
}

fn expected_rgb(lambert: Option<f64>, fog_amount: f64) -> [f64; 3] {
    let ambient = [0.2, 0.1, 0.05];
    let sun = [0.3, 0.6, 0.45];
    std::array::from_fn(|i| {
        let lit = BASE[i] * lambert.map_or(1.0, |cosine| ambient[i] + sun[i] * cosine.max(0.0));
        lit * (1.0 - fog_amount) + FOG[i] * fog_amount
    })
}

fn premultiplied(rgb: [f64; 3], alpha: f64) -> [f64; 4] {
    [rgb[0] * alpha, rgb[1] * alpha, rgb[2] * alpha, alpha]
}

pub(super) fn assert_gpu_lighting_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let mut fixture = Fixture::new(device, queue, format);
    let normal = Vec3::new(1.0, 1.0, 1.0).unwrap();
    // Inverse transpose, not the model transform: (1/2, 1, 2) dot +Z.
    let cosine = 2.0 / (0.25_f64 + 1.0 + 4.0).sqrt();
    for perspective in [false, true] {
        for crossing in [false, true] {
            for reverse in [false, true] {
                let mesh = fixture.upload(quad(perspective, crossing, reverse, Some(normal)));
                for policy in [
                    SurfaceRasterization3d::StrictPortable,
                    SurfaceRasterization3d::Native,
                ] {
                    for lit in [false, true] {
                        for fogged in [false, true] {
                            for mode in 0..4 {
                                let tint = Color::rgb(0.5, 0.5, 0.5);
                                let style = match mode {
                                    0 => SurfaceStyle3d::opaque(tint).unwrap(),
                                    1 => SurfaceStyle3d::mask(tint, 0.5).unwrap(),
                                    2 => SurfaceStyle3d::blend(tint).unwrap(),
                                    _ => SurfaceStyle3d::mask(tint, 0.75).unwrap(),
                                }
                                .with_lighting(if lit {
                                    SurfaceLighting3d::Lambert
                                } else {
                                    SurfaceLighting3d::Unlit
                                })
                                .with_fog(fogged);
                                let mut scene =
                                    Scene3d::with_alpha_background(Color::TRANSPARENT).unwrap();
                                scene.set_lighting(lighting());
                                scene.set_fog(Some(fog(1.0, 0.4)));
                                scene
                                    .try_push(
                                        &mesh,
                                        nonuniform_transform(),
                                        MeshStyle3d::surface(style),
                                    )
                                    .unwrap();
                                let pixels = fixture.render(&scene, camera(perspective), policy);
                                let cosine = if reverse { -cosine } else { cosine };
                                let rgb = expected_rgb(
                                    lit.then_some(cosine),
                                    if fogged { 1.0 - (-0.8_f64).exp() } else { 0.0 },
                                );
                                let alpha = if mode == 3 {
                                    0.0
                                } else if mode == 2 {
                                    0.5
                                } else {
                                    1.0
                                };
                                fixture.expect(&pixels, 29, 27, premultiplied(rgb, alpha));
                            }
                        }
                    }
                }
            }
        }
        assert_directions_and_fog_boundaries(&mut fixture, perspective);
        assert_varying_normals(&mut fixture, perspective);
        assert_textured_lighting(&mut fixture, perspective);
        assert_camera_relative_environment(&mut fixture, perspective);
        assert_unaffected_edges(&mut fixture, perspective);
        assert_varying_fog_depth(&mut fixture, perspective);
    }
    assert_cancelled_normals(&mut fixture);
    assert_saturation_before_fog(&mut fixture);
    assert_edge_only_material_ignores_surface_environment(&mut fixture);
}

fn assert_directions_and_fog_boundaries(fixture: &mut Fixture<'_>, perspective: bool) {
    for (normal, cosine) in [
        (Some(Vec3::Z), Some(1.0)),
        (Some(Vec3::X), Some(0.0)),
        (Some(Vec3::new(0.0, 0.0, -1.0).unwrap()), Some(0.0)),
        (None, None),
    ] {
        let mesh = fixture.upload(quad(perspective, true, false, normal));
        for policy in [
            SurfaceRasterization3d::StrictPortable,
            SurfaceRasterization3d::Native,
        ] {
            // All readback points have exactly three world units of view depth,
            // including orthographic projection (whose clip W is only one).
            for (start, density, amount) in [
                (4.0, 1.0, 0.0),
                (3.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 0.5, 1.0 - (-1.0_f64).exp()),
                (0.0, f32::MAX, 1.0),
                (f32::MAX, f32::MAX, 0.0),
            ] {
                let style = SurfaceStyle3d::opaque(Color::rgb(0.5, 0.5, 0.5))
                    .unwrap()
                    .with_lighting(if normal.is_some() {
                        SurfaceLighting3d::Lambert
                    } else {
                        SurfaceLighting3d::Unlit
                    })
                    .with_fog(true);
                let mut scene = Scene3d::new(Color::BLACK).unwrap();
                scene.set_lighting(lighting());
                scene.set_fog(Some(fog(start, density)));
                scene
                    .try_push(&mesh, nonuniform_transform(), MeshStyle3d::surface(style))
                    .unwrap();
                let pixels = fixture.render(&scene, camera(perspective), policy);
                fixture.expect(
                    &pixels,
                    29,
                    27,
                    premultiplied(expected_rgb(cosine, amount), 1.0),
                );
            }
        }
    }
}

fn assert_varying_normals(fixture: &mut Fixture<'_>, perspective: bool) {
    // Crosses two side planes. Different source depths require perspective
    // interpolation. Nonuniform scaling exposes normalization-before-clipping.
    let points = [
        Vec3::new(-2.0, -1.5, 0.0).unwrap(),
        Vec3::new(0.9, -0.8, 0.7).unwrap(),
        Vec3::new(0.1, 1.8, -0.6).unwrap(),
    ];
    let normal_values = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.1, 0.0, 1.0]];
    let normals = normal_values.map(|[x, y, z]| Vec3::new(x, y, z).unwrap());
    let scale = if perspective { 3.0 } else { 1.0 };
    let model_points =
        points.map(|p| Vec3::new(p.x() * scale / 2.0, p.y() * scale, p.z() * 2.0).unwrap());
    let source = Mesh3d::with_attributes(
        model_points.to_vec(),
        vec![0, 1, 2],
        vec![],
        Mesh3dAttributes::new()
            .with_normals(normals.to_vec())
            .unwrap()
            .with_vertex_colors(vec![Color::rgb(0.8, 0.6, 0.4); 3])
            .unwrap(),
    )
    .unwrap();
    let mesh = fixture.upload(source);
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene.set_lighting(lighting());
    scene.set_fog(Some(fog(1.0, 0.4)));
    let object = scene
        .try_push(
            &mesh,
            nonuniform_transform(),
            MeshStyle3d::surface(
                SurfaceStyle3d::opaque(Color::rgb(0.5, 0.5, 0.5))
                    .unwrap()
                    .with_lighting(SurfaceLighting3d::Lambert),
            ),
        )
        .unwrap();
    let camera = camera(perspective);
    let viewport = LogicalViewport::new(SIZE as f32, SIZE as f32).unwrap();
    let projected = points.map(|p| {
        camera
            .project_world(
                Vec3::new(p.x() * scale, p.y() * scale, p.z()).unwrap(),
                viewport,
            )
            .unwrap()
    });
    for fogged in [false, true] {
        scene
            .set_style(
                object,
                MeshStyle3d::surface(
                    SurfaceStyle3d::opaque(Color::rgb(0.5, 0.5, 0.5))
                        .unwrap()
                        .with_lighting(SurfaceLighting3d::Lambert)
                        .with_fog(fogged),
                ),
            )
            .unwrap();
        for policy in [
            SurfaceRasterization3d::StrictPortable,
            SurfaceRasterization3d::Native,
        ] {
            let pixels = fixture.render(&scene, camera, policy);
            for (x, y) in [(25, 25), (35, 25), (25, 35), (35, 35)] {
                let screen = projected.map(|p| {
                    let xy = p.logical_position().to_vec2();
                    [f64::from(xy.x), f64::from(xy.y)]
                });
                let weights = barycentric(screen, [f64::from(x) + 0.5, f64::from(y) + 0.5]);
                assert!(
                    weights.iter().all(|v| *v > 0.05),
                    "oracle point must be well inside source triangle"
                );
                let mut weights = std::array::from_fn::<_, 3, _>(|i| {
                    if perspective {
                        weights[i] / f64::from(projected[i].view_depth())
                    } else {
                        weights[i]
                    }
                });
                let sum: f64 = weights.iter().sum();
                for weight in &mut weights {
                    *weight /= sum;
                }
                let mut interpolated = [0.0; 3];
                let mut incorrectly_normalized = [0.0; 3];
                let mut depth = 0.0;
                for i in 0..3 {
                    let normal: [f64; 3] = normal_values[i].map(f64::from);
                    let length = normal.iter().map(|v| v * v).sum::<f64>().sqrt();
                    for c in 0..3 {
                        interpolated[c] += weights[i] * normal[c] / length / [2.0, 1.0, 0.5][c];
                    }
                    let transformed =
                        std::array::from_fn::<_, 3, _>(|c| normal[c] / length / [2.0, 1.0, 0.5][c]);
                    let transformed_length = transformed.iter().map(|v| v * v).sum::<f64>().sqrt();
                    for c in 0..3 {
                        incorrectly_normalized[c] +=
                            weights[i] * transformed[c] / transformed_length;
                    }
                    depth += weights[i] * f64::from(projected[i].view_depth());
                }
                let length = interpolated.iter().map(|v| v * v).sum::<f64>().sqrt();
                let cosine = interpolated[2] / length;
                let incorrect_cosine = incorrectly_normalized[2]
                    / incorrectly_normalized
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt();
                assert!(
                    (cosine - incorrect_cosine).abs() > 0.1,
                    "oracle must separate fragment normalization from per-vertex normalization"
                );
                let amount = if fogged {
                    1.0 - (-0.4 * (depth - 1.0).max(0.0)).exp()
                } else {
                    0.0
                };
                fixture.expect(
                    &pixels,
                    x,
                    y,
                    premultiplied(expected_rgb(Some(cosine), amount), 1.0),
                );
            }
        }
    }
}

fn barycentric(p: [[f64; 2]; 3], q: [f64; 2]) -> [f64; 3] {
    let denominator =
        (p[1][1] - p[2][1]) * (p[0][0] - p[2][0]) + (p[2][0] - p[1][0]) * (p[0][1] - p[2][1]);
    let a = ((p[1][1] - p[2][1]) * (q[0] - p[2][0]) + (p[2][0] - p[1][0]) * (q[1] - p[2][1]))
        / denominator;
    let b = ((p[2][1] - p[0][1]) * (q[0] - p[2][0]) + (p[0][0] - p[2][0]) * (q[1] - p[2][1]))
        / denominator;
    [a, b, 1.0 - a - b]
}

fn assert_textured_lighting(fixture: &mut Fixture<'_>, perspective: bool) {
    let plain = quad(perspective, true, false, Some(Vec3::Z));
    let source = Mesh3d::with_attributes(
        plain.vertices().to_vec(),
        plain.triangle_indices().to_vec(),
        vec![],
        Mesh3dAttributes::new()
            .with_normals(vec![Vec3::Z; 4])
            .unwrap()
            .with_vertex_colors(vec![Color::rgba(0.8, 0.6, 0.4, 0.5); 4])
            .unwrap()
            .with_texture_coordinates(vec![TextureCoordinate2d::new(0.5, 0.5).unwrap(); 4]),
    )
    .unwrap();
    let texture = texture::create_texture_with_alpha(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &fixture.renderer.textures.layout,
        1,
        1,
        vec![255, 0, 255, 128],
        ImageBudget::default(),
    )
    .unwrap();
    let material = TextureMaterial3d::with_alpha(
        &texture,
        ImageSampling::Nearest,
        Color::rgba(1.0, 0.5, 0.25, 0.5),
    )
    .unwrap();
    let mesh =
        texture::attach_material(&fixture.identity, &fixture.upload(source), &material).unwrap();
    for policy in [
        SurfaceRasterization3d::StrictPortable,
        SurfaceRasterization3d::Native,
    ] {
        let mut scene = Scene3d::with_alpha_background(Color::TRANSPARENT).unwrap();
        scene.set_lighting(lighting());
        scene.set_fog(Some(fog(1.0, 0.4)));
        scene
            .try_push(
                &mesh,
                nonuniform_transform(),
                MeshStyle3d::surface(
                    SurfaceStyle3d::blend(Color::rgb(0.5, 0.5, 0.5))
                        .unwrap()
                        .with_lighting(SurfaceLighting3d::Lambert)
                        .with_fog(true),
                ),
            )
            .unwrap();
        let pixels = fixture.render(&scene, camera(perspective), policy);
        let amount = 1.0 - (-0.8_f64).exp();
        let lit = expected_rgb(Some(1.0), 0.0);
        let rgb = std::array::from_fn(|i| {
            lit[i] * [1.0, 0.0, 0.25][i] * (1.0 - amount) + FOG[i] * amount
        });
        fixture.expect(
            &pixels,
            29,
            27,
            premultiplied(rgb, 0.5 * (128.0 / 255.0) * 0.5),
        );
    }
}

fn assert_cancelled_normals(fixture: &mut Fixture<'_>) {
    // The sample at x=32.5 lies exactly halfway across this binary-exact quad.
    // Opposing source normals cancel there; the defined result is ambient only.
    let vertices = [
        [-0.484375, -0.5],
        [0.515625, -0.5],
        [0.515625, 0.5],
        [-0.484375, 0.5],
    ]
    .map(|[x, y]| Vec3::new(x, y, 0.0).unwrap());
    let negative = Vec3::new(0.0, 0.0, -1.0).unwrap();
    let source = Mesh3d::with_attributes(
        vertices.to_vec(),
        vec![0, 1, 2, 0, 2, 3],
        vec![],
        Mesh3dAttributes::new()
            .with_normals(vec![Vec3::Z, negative, negative, Vec3::Z])
            .unwrap()
            .with_vertex_colors(vec![Color::rgb(0.8, 0.6, 0.4); 4])
            .unwrap(),
    )
    .unwrap();
    let mesh = fixture.upload(source);
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene.set_lighting(lighting());
    scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(
                SurfaceStyle3d::opaque(Color::rgb(0.5, 0.5, 0.5))
                    .unwrap()
                    .with_lighting(SurfaceLighting3d::Lambert),
            ),
        )
        .unwrap();
    for policy in [
        SurfaceRasterization3d::StrictPortable,
        SurfaceRasterization3d::Native,
    ] {
        let pixels = fixture.render(&scene, camera(false), policy);
        fixture.expect(
            &pixels,
            32,
            27,
            premultiplied(expected_rgb(Some(0.0), 0.0), 1.0),
        );
    }
}

fn assert_camera_relative_environment(fixture: &mut Fixture<'_>, perspective: bool) {
    let mesh = fixture.upload(quad(
        perspective,
        true,
        false,
        Some(Vec3::new(1.0, 1.0, 1.0).unwrap()),
    ));
    let translation = Vec3::new(5.0, -3.0, 8.0).unwrap();
    let cosine = 2.0 / (5.25_f64).sqrt();
    for angle in [0.0, 0.9, 2.7] {
        let rotation = Rotation3d::from_axis_angle(Vec3::Y, angle).unwrap();
        let camera = Camera3d::look_at(
            translation
                .checked_add(rotation.rotate(Vec3::new(0.0, 0.0, 3.0).unwrap()).unwrap())
                .unwrap(),
            translation,
            Vec3::Y,
            camera(perspective).projection(),
        )
        .unwrap();
        let transform =
            Transform3d::new(translation, rotation, Vec3::new(2.0, 1.0, 0.5).unwrap()).unwrap();
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        scene.set_lighting(
            Lighting3d::new(AmbientLight3d::new(Color::rgb(1.0, 0.5, 0.25), 0.2).unwrap())
                .with_directional(Some(
                    DirectionalLight3d::new(
                        rotation.rotate(Vec3::Z).unwrap(),
                        Color::rgb(0.5, 1.0, 0.75),
                        0.6,
                    )
                    .unwrap(),
                )),
        );
        scene.set_fog(Some(fog(1.0, 0.4)));
        scene
            .try_push(
                &mesh,
                transform,
                MeshStyle3d::surface(
                    SurfaceStyle3d::opaque(Color::rgb(0.5, 0.5, 0.5))
                        .unwrap()
                        .with_lighting(SurfaceLighting3d::Lambert)
                        .with_fog(true),
                ),
            )
            .unwrap();
        for policy in [
            SurfaceRasterization3d::StrictPortable,
            SurfaceRasterization3d::Native,
        ] {
            let pixels = fixture.render(&scene, camera, policy);
            fixture.expect(
                &pixels,
                29,
                27,
                premultiplied(expected_rgb(Some(cosine), 1.0 - (-0.8_f64).exp()), 1.0),
            );
        }
    }
}

fn assert_unaffected_edges(fixture: &mut Fixture<'_>, perspective: bool) {
    let source = quad(perspective, false, false, Some(Vec3::Z));
    let mut vertices = source.vertices().to_vec();
    let scale = if perspective { 3.0 } else { 1.0 };
    vertices.extend([
        Vec3::new(-0.3 * scale, 0.0, 0.2).unwrap(),
        Vec3::new(0.3 * scale, 0.0, 0.2).unwrap(),
    ]);
    let source = Mesh3d::with_attributes(
        vertices,
        source.triangle_indices().to_vec(),
        vec![MeshEdge3d::new(4, 5).unwrap()],
        Mesh3dAttributes::new()
            .with_normals(vec![Vec3::Z; 6])
            .unwrap(),
    )
    .unwrap();
    let mesh = fixture.upload(source);
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene.set_lighting(Lighting3d::new(
        AmbientLight3d::new(Color::BLACK, 0.0).unwrap(),
    ));
    scene.set_fog(Some(fog(0.0, f32::MAX)));
    let style = MeshStyle3d::surface(
        SurfaceStyle3d::opaque(Color::WHITE)
            .unwrap()
            .with_lighting(SurfaceLighting3d::Lambert)
            .with_fog(true),
    )
    .with_wireframe(WireframeStyle3d::visible(Color::WHITE, logical(4.0)).unwrap());
    scene
        .try_push(&mesh, nonuniform_transform(), style)
        .unwrap();
    for policy in [
        SurfaceRasterization3d::StrictPortable,
        SurfaceRasterization3d::Native,
    ] {
        let pixels = fixture.render(&scene, camera(perspective), policy);
        fixture.expect(&pixels, 29, 27, premultiplied(FOG, 1.0));
        fixture.expect(&pixels, 29, 32, [1.0; 4]);
    }
}

fn assert_varying_fog_depth(fixture: &mut Fixture<'_>, perspective: bool) {
    let depths = [0.75, 6.0, 3.0];
    let projected_xy = [[-1.5, -1.1], [0.9, -0.8], [0.1, 1.4]];
    let vertices = std::array::from_fn::<_, 3, _>(|i| {
        let factor = if perspective { depths[i] } else { 1.0 };
        Vec3::new(
            projected_xy[i][0] * factor,
            projected_xy[i][1] * factor,
            3.0 - depths[i],
        )
        .unwrap()
    });
    let mesh = fixture.upload(Mesh3d::new(vertices.to_vec(), vec![0, 1, 2]).unwrap());
    let mut scene = Scene3d::new(Color::rgb(1.0, 0.0, 0.0)).unwrap();
    scene.set_fog(Some(Fog3d::new(Color::BLACK, 0.0, 0.5).unwrap()));
    scene
        .try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap().with_fog(true)),
        )
        .unwrap();
    for (near, far) in [(0.5, 8.0), (0.1, 100.0)] {
        let projection = if perspective {
            Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(near), world(far))
        } else {
            Projection3d::orthographic(world(2.0), 1.0, world(near), world(far))
        }
        .unwrap();
        let camera = Camera3d::look_at(
            Vec3::new(0.0, 0.0, 3.0).unwrap(),
            Vec3::ZERO,
            Vec3::Y,
            projection,
        )
        .unwrap();
        let screen =
            projected_xy.map(|p| [32.0 + 32.0 * f64::from(p[0]), 32.0 - 32.0 * f64::from(p[1])]);
        for policy in [
            SurfaceRasterization3d::StrictPortable,
            SurfaceRasterization3d::Native,
        ] {
            let pixels = fixture.render(&scene, camera, policy);
            for (x, y) in [(25, 25), (35, 25), (25, 35), (35, 35)] {
                let weights = barycentric(screen, [f64::from(x) + 0.5, f64::from(y) + 0.5]);
                assert!(weights.iter().all(|v| *v > 0.05));
                let affine_depth = (0..3)
                    .map(|i| weights[i] * f64::from(depths[i]))
                    .sum::<f64>();
                let harmonic_depth = 1.0
                    / (0..3)
                        .map(|i| weights[i] / f64::from(depths[i]))
                        .sum::<f64>();
                let depth = if perspective {
                    harmonic_depth
                } else {
                    affine_depth
                };
                let gray = (-0.5 * depth).exp();
                assert!(
                    ((-0.5 * harmonic_depth).exp() - (-0.5 * affine_depth).exp()).abs() > 0.1,
                    "fog oracle must distinguish perspective from affine interpolation"
                );
                fixture.expect(&pixels, x, y, [gray, gray, gray, 1.0]);
            }
        }
    }
}

fn assert_saturation_before_fog(fixture: &mut Fixture<'_>) {
    let source = quad(false, true, false, Some(Vec3::Z));
    let source = Mesh3d::with_attributes(
        source.vertices().to_vec(),
        source.triangle_indices().to_vec(),
        vec![],
        Mesh3dAttributes::new()
            .with_normals(vec![Vec3::Z; 4])
            .unwrap(),
    )
    .unwrap();
    let mesh = fixture.upload(source);
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene.set_lighting(
        Lighting3d::new(AmbientLight3d::new(Color::WHITE, 1.0).unwrap()).with_directional(Some(
            DirectionalLight3d::new(Vec3::Z, Color::WHITE, 1.0).unwrap(),
        )),
    );
    scene.set_fog(Some(
        Fog3d::new(Color::rgb(0.1, 0.1, 0.1), 1.0, std::f32::consts::LN_2 / 2.0).unwrap(),
    ));
    scene
        .try_push(
            &mesh,
            nonuniform_transform(),
            MeshStyle3d::surface(
                SurfaceStyle3d::opaque(Color::rgb(0.75, 0.75, 0.75))
                    .unwrap()
                    .with_lighting(SurfaceLighting3d::Lambert)
                    .with_fog(true),
            ),
        )
        .unwrap();
    for policy in [
        SurfaceRasterization3d::StrictPortable,
        SurfaceRasterization3d::Native,
    ] {
        let pixels = fixture.render(&scene, camera(false), policy);
        // clamp(base .75 * illumination 2) = 1, then 50% fog -> .55.
        fixture.expect(&pixels, 29, 27, [0.55, 0.55, 0.55, 1.0]);
    }
}

fn assert_edge_only_material_ignores_surface_environment(fixture: &mut Fixture<'_>) {
    let mesh = fixture.upload(
        Mesh3d::with_display_edges(
            vec![
                Vec3::new(-0.6, 0.0, 0.0).unwrap(),
                Vec3::new(0.6, 0.0, 0.0).unwrap(),
            ],
            vec![],
            vec![MeshEdge3d::new(0, 1).unwrap()],
        )
        .unwrap(),
    );
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene.set_lighting(lighting());
    scene.set_fog(Some(fog(0.0, f32::MAX)));
    // A reusable material can include surface settings even when this revision
    // contains only construction edges. No surface means no normal requirement.
    let style = MeshStyle3d::surface(
        SurfaceStyle3d::opaque(Color::WHITE)
            .unwrap()
            .with_lighting(SurfaceLighting3d::Lambert)
            .with_fog(true),
    )
    .with_wireframe(WireframeStyle3d::visible(Color::WHITE, logical(4.0)).unwrap());
    scene.try_push(&mesh, Transform3d::IDENTITY, style).unwrap();
    for policy in [
        SurfaceRasterization3d::StrictPortable,
        SurfaceRasterization3d::Native,
    ] {
        let pixels = fixture.render(&scene, camera(false), policy);
        fixture.expect(&pixels, 29, 32, [1.0; 4]);
    }
}
