use super::*;
use crate::{Mesh3dAttributes, TextureCoordinate2d};

const SIZE: u32 = 64;

fn camera(perspective: bool) -> Camera3d {
    let projection = if perspective {
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(0.5), world(4.0))
    } else {
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(4.0))
    }
    .unwrap();
    Camera3d::look_at(
        Vec3::new(0.0, 0.0, 2.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        projection,
    )
    .unwrap()
}

fn positions(perspective: bool, crossing: Option<usize>) -> [Vec3; 3] {
    let mut values = [
        [-0.75, -0.5, if perspective { 0.5 } else { 0.0 }],
        [0.75, -0.5, if perspective { -0.5 } else { 0.0 }],
        [0.0, 0.75, 0.0],
    ];
    match crossing {
        Some(0) => values[0][0] = -3.0,
        Some(1) => values[1][0] = 4.0,
        Some(2) => values[0][1] = -3.0,
        Some(3) => values[2][1] = 3.0,
        Some(4) => values[2] = [0.0, 0.1, 1.75],
        Some(5) => values[2][2] = -3.0,
        Some(6) => {
            values[0][0] = -3.0;
            values[0][1] = -2.25;
            values[2][1] = 3.0;
        }
        None => {}
        _ => panic!("unknown clip boundary"),
    }
    values.map(|[x, y, z]| Vec3::new(x, y, z).unwrap())
}

fn colors() -> [Color; 3] {
    [
        Color::rgba(1.0, 0.1, 0.2, 0.0),
        Color::rgba(0.2, 1.0, 0.1, 0.5),
        Color::rgba(0.1, 0.2, 1.0, 1.0),
    ]
}

fn source(
    positions: [Vec3; 3],
    colors: Option<[Color; 3]>,
    reverse: bool,
    textured: bool,
) -> Mesh3d {
    let mut attributes = Mesh3dAttributes::new();
    if let Some(colors) = colors {
        attributes = attributes.with_vertex_colors(colors.to_vec()).unwrap();
    }
    if textured {
        attributes =
            attributes
                .with_texture_coordinates(vec![TextureCoordinate2d::new(0.5, 0.5).unwrap(); 3]);
    }
    Mesh3d::with_attributes(
        positions.to_vec(),
        if reverse {
            vec![0, 2, 1]
        } else {
            vec![0, 1, 2]
        },
        Vec::new(),
        attributes,
    )
    .unwrap()
}

fn encode_channel(linear: f64, srgb: bool) -> u8 {
    let value = if srgb {
        if linear <= 0.003_130_8 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        }
    } else {
        linear
    };
    (value * 255.0).round() as u8
}

fn channel_order(format: wgpu::TextureFormat) -> [usize; 3] {
    match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2],
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0],
        _ => panic!("unsupported oracle format {format:?}"),
    }
}

fn orient(a: [f64; 2], b: [f64; 2], point: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
}

fn assert_interpolation(
    pixels: &[u8],
    format: wgpu::TextureFormat,
    positions: [Vec3; 3],
    camera: Camera3d,
    colors: [Color; 3],
    tint: Color,
) {
    // The oracle uses the public f64 camera projection and analytic screen
    // barycentrics, not the renderer's clipper or its copied shader rows.
    let viewport = LogicalViewport::new(SIZE as f32, SIZE as f32).unwrap();
    let projected = positions.map(|point| camera.project_world(point, viewport).unwrap());
    let points = projected.map(|point| {
        let logical = point.logical_position().to_vec2();
        [logical.x() as f64, logical.y() as f64]
    });
    let area = orient(points[0], points[1], points[2]);
    let channels = channel_order(format);
    let mut checked_interior = 0;
    let mut checked_exterior = 0;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let point = [x as f64 + 0.5, y as f64 + 0.5];
            let weights = [
                orient(points[1], points[2], point) / area,
                orient(points[2], points[0], point) / area,
                orient(points[0], points[1], point) / area,
            ];
            let depth: f64 = (0..3)
                .map(|index| weights[index] * projected[index].normalized_depth() as f64)
                .sum();
            let offset = (y * SIZE + x) as usize * 4;
            let pixel = &pixels[offset..offset + 4];
            if weights.iter().all(|weight| *weight > 0.04) && (0.02..0.98).contains(&depth) {
                let mut attributes = weights;
                if camera.projection().is_perspective() {
                    for index in 0..3 {
                        attributes[index] /= projected[index].view_depth() as f64;
                    }
                    let sum: f64 = attributes.iter().sum();
                    attributes.iter_mut().for_each(|weight| *weight /= sum);
                }
                for channel in 0..3 {
                    let linear: f64 = (0..3)
                        .map(|index| attributes[index] * colors[index].to_array()[channel] as f64)
                        .sum();
                    let expected =
                        encode_channel(linear * tint.to_array()[channel] as f64, format.is_srgb());
                    assert!(
                        pixel[channels[channel]].abs_diff(expected) <= 3,
                        "vertex-color pixel ({x},{y}) channel {channel}: {pixel:?}, expected {expected}"
                    );
                }
                assert_eq!(pixel[3], 255, "opaque material ignores vertex alpha");
                checked_interior += 1;
            } else if weights.iter().any(|weight| *weight < -0.04)
                || !(-0.02..=1.02).contains(&depth)
            {
                assert_eq!(
                    pixel,
                    &[0, 0, 0, 255],
                    "outside source/frustum at ({x},{y})"
                );
                checked_exterior += 1;
            }
        }
    }
    assert!(
        checked_interior >= 32,
        "only {checked_interior} interior samples"
    );
    assert!(
        checked_exterior >= 128,
        "only {checked_exterior} exterior samples"
    );
}

fn product(left: Color, right: Color) -> Color {
    let a = left.to_array();
    let b = right.to_array();
    Color::rgb(a[0] * b[0], a[1] * b[1], a[2] * b[2])
}

pub(super) fn assert_gpu_vertex_color_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let target = surface::test_target(device, &identity, format, SIZE, SIZE);
    let surface_color = Color::rgb(0.8, 0.5, 0.7);
    let material_tint = Color::rgb(0.6, 0.7, 0.9);
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(surface_color).unwrap());
    let texture = texture::create_texture(
        device,
        queue,
        &identity,
        &renderer.textures.layout,
        1,
        1,
        vec![128, 192, 64, 255],
        ImageBudget::default(),
    )
    .unwrap();
    let material = TextureMaterial3d::new(&texture, ImageSampling::Nearest, material_tint).unwrap();
    let sampled = Color::rgb8(128, 192, 64);
    let mut cases = 0;
    for perspective in [false, true] {
        let camera = camera(perspective);
        for crossing in std::iter::once(None).chain((0..7).map(Some)) {
            let points = positions(perspective, crossing);
            for reverse in [false, true] {
                for textured in [false, true] {
                    let mesh = create_retained_mesh(
                        device,
                        queue,
                        Arc::clone(&identity),
                        source(points, Some(colors()), reverse, textured),
                    )
                    .unwrap();
                    let mesh = if textured {
                        texture::attach_material(&identity, &mesh, &material).unwrap()
                    } else {
                        mesh
                    };
                    let tint = if textured {
                        product(surface_color, product(material_tint, sampled))
                    } else {
                        surface_color
                    };
                    let mut scene = Scene3d::new(Color::BLACK).unwrap();
                    scene.try_push(&mesh, Transform3d::IDENTITY, style).unwrap();
                    for policy in [
                        SurfaceRasterization3d::StrictPortable,
                        SurfaceRasterization3d::Native,
                    ] {
                        let report = renderer
                            .render_scene3d(
                                device,
                                queue,
                                &identity,
                                &target,
                                &scene,
                                camera,
                                Mesh3dRenderBudget::default().with_surface_policy(policy),
                            )
                            .unwrap_or_else(|error| {
                                panic!("color fixture perspective={perspective} boundary={crossing:?} reverse={reverse} textured={textured} policy={policy:?}: {error}")
                            });
                        if crossing.is_some() && policy == SurfaceRasterization3d::StrictPortable {
                            assert!(report.preflight().generated_triangle_count() > 0);
                        } else {
                            assert_eq!(report.preflight().generated_triangle_count(), 0);
                        }
                        let pixels = surface::test_read_pixels(device, queue, &target);
                        assert_interpolation(&pixels, format, points, camera, colors(), tint);
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 128);
    assert_default_parity(
        device,
        queue,
        format,
        &identity,
        &mut renderer,
        &target,
        &material,
    );
    assert_mixed_color_layouts(
        device,
        queue,
        format,
        &identity,
        &mut renderer,
        &target,
        &material,
    );
}

#[allow(clippy::too_many_arguments)]
fn assert_mixed_color_layouts(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    identity: &Arc<()>,
    renderer: &mut Mesh3dRenderer,
    target: &RenderTarget3d,
    material: &TextureMaterial3d,
) {
    let tint = Color::rgb(0.6, 0.8, 0.9);
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(tint).unwrap());
    let color = Color::rgba(0.8, 0.3, 0.6, 0.1);
    let corners = [(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)];
    for crossing in [false, true] {
        for reverse_order in [false, true] {
            let mut scene = Scene3d::new(Color::BLACK).unwrap();
            for step in 0..4 {
                let index = if reverse_order { 3 - step } else { step };
                let (x, y): (f32, f32) = corners[index];
                let points = if crossing {
                    [[1.5, 0.125], [0.125, 0.125], [0.125, 1.5]]
                        .map(|[px, py]| Vec3::new(px * x.signum(), py * y.signum(), 0.0).unwrap())
                } else {
                    positions(false, None)
                };
                let mesh = create_retained_mesh(
                    device,
                    queue,
                    Arc::clone(identity),
                    source(
                        points,
                        (index % 2 == 0).then_some([color; 3]),
                        false,
                        index >= 2,
                    ),
                )
                .unwrap();
                let mesh = if index >= 2 {
                    texture::attach_material(identity, &mesh, material).unwrap()
                } else {
                    mesh
                };
                let transform = if crossing {
                    Transform3d::IDENTITY
                } else {
                    Transform3d::new(
                        Vec3::new(x, y, 0.0).unwrap(),
                        Rotation3d::IDENTITY,
                        Vec3::new(0.4, 0.4, 0.4).unwrap(),
                    )
                    .unwrap()
                };
                scene.try_push(&mesh, transform, style).unwrap();
            }
            let report = renderer
                .render_scene3d(
                    device,
                    queue,
                    identity,
                    target,
                    &scene,
                    camera(false),
                    Mesh3dRenderBudget::default(),
                )
                .unwrap();
            if crossing {
                assert_eq!(report.preflight().generated_object_count(), 4);
                assert!(report.triangle_count() > 4);
            } else {
                assert_eq!(report.triangle_count(), 4);
            }
            let pixels = surface::test_read_pixels(device, queue, target);
            for (index, (x, y)) in corners.iter().copied().enumerate() {
                let x = ((x * 0.5 + 0.5) * SIZE as f32) as usize;
                let y = ((0.5 - y * 0.5) * SIZE as f32) as usize;
                let mut expected = tint;
                if index % 2 == 0 {
                    expected = product(expected, color);
                }
                if index >= 2 {
                    expected = product(
                        expected,
                        product(Color::rgb8(128, 192, 64), material.tint()),
                    );
                }
                let pixel = &pixels[(y * SIZE as usize + x) * 4..][..4];
                for (channel, offset) in channel_order(format).into_iter().enumerate() {
                    let encoded =
                        encode_channel(expected.to_array()[channel] as f64, format.is_srgb());
                    assert!(
                        pixel[offset].abs_diff(encoded) <= 2,
                        "mixed layout {index}: {pixel:?} expected {encoded}"
                    );
                }
                assert_eq!(pixel[3], 255);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_default_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    identity: &Arc<()>,
    renderer: &mut Mesh3dRenderer,
    target: &RenderTarget3d,
    material: &TextureMaterial3d,
) {
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::rgb(0.3, 0.6, 0.9)).unwrap());
    for crossing in [None, Some(0), Some(4)] {
        for textured in [false, true] {
            for policy in [
                SurfaceRasterization3d::StrictPortable,
                SurfaceRasterization3d::Native,
            ] {
                let mut reference = None;
                for colors in [None, Some([Color::WHITE; 3])] {
                    let source = source(positions(true, crossing), colors, false, textured);
                    let mesh =
                        create_retained_mesh(device, queue, Arc::clone(identity), source).unwrap();
                    let mesh = if textured {
                        texture::attach_material(identity, &mesh, material).unwrap()
                    } else {
                        mesh
                    };
                    let mut scene = Scene3d::new(Color::BLACK).unwrap();
                    scene.try_push(&mesh, Transform3d::IDENTITY, style).unwrap();
                    renderer
                        .render_scene3d(
                            device,
                            queue,
                            identity,
                            target,
                            &scene,
                            camera(true),
                            Mesh3dRenderBudget::default().with_surface_policy(policy),
                        )
                        .unwrap();
                    let pixels = surface::test_read_pixels(device, queue, target);
                    if let Some(reference) = &reference {
                        assert_eq!(
                            &pixels, reference,
                            "white/default parity {format:?} {policy:?}"
                        );
                    } else {
                        assert!(pixels.chunks_exact(4).any(|pixel| pixel[1] > 30));
                        reference = Some(pixels);
                    }
                }
            }
        }
    }
}
