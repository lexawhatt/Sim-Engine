//! Independent exact-color and screen-grid raster oracles.
use super::*;
use crate::SurfaceEffects3d;

pub(super) fn verify(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) {
    affine_uv_oracle(device, queue, format);
    let identity = Arc::new(());
    let target = surface::test_target(device, &identity, format, 64, 64);
    let mut renderer = Mesh3dRenderer::new(device, format);
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 3.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.1), world(10.0)).unwrap(),
    )
    .unwrap();
    let make_scene = |lo: f32, hi: f32, color: Color, z: f32| {
        let mesh = Mesh3d::new(
            vec![
                Vec3::new(lo, lo, z).unwrap(),
                Vec3::new(hi, lo, z).unwrap(),
                Vec3::new(hi, hi, z).unwrap(),
                Vec3::new(lo, hi, z).unwrap(),
            ],
            vec![0, 1, 2, 0, 2, 3],
        )
        .unwrap();
        let mesh = create_retained_mesh(device, queue, identity.clone(), mesh).unwrap();
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        scene
            .try_push(
                &mesh,
                Transform3d::IDENTITY,
                MeshStyle3d::surface(SurfaceStyle3d::opaque(color).unwrap()),
            )
            .unwrap();
        scene
    };
    let mut draw = |scene: &Scene3d, budget: Mesh3dRenderBudget| {
        renderer
            .render_scene3d(device, queue, &identity, &target, scene, camera, budget)
            .unwrap();
        surface::test_read_pixels(device, queue, &target)
    };
    let native = Mesh3dRenderBudget::default().with_surface_policy(SurfaceRasterization3d::Native);
    let white = make_scene(-0.9, 0.9, Color::WHITE, 0.0);
    let pixels = draw(
        &white,
        native.with_surface_effects(SurfaceEffects3d::new().with_scanlines(255)),
    );
    for y in 20..40 {
        for x in 20..40 {
            let expected = if y % 2 == 0 { 255 } else { 0 };
            assert_eq!(
                &pixels[(y * 64 + x) * 4..(y * 64 + x) * 4 + 4],
                &[expected, expected, expected, 255]
            );
        }
    }
    let gray = make_scene(-0.9, 0.9, Color::rgb(0.5, 0.5, 0.5), 0.0);
    let pixels = draw(
        &gray,
        native.with_surface_effects(
            SurfaceEffects3d::new()
                .with_quantization(2)
                .unwrap()
                .with_dither(255),
        ),
    );
    // A half-level input with a centered Bayer threshold has a checkerboard of
    // black/white at the finest scale, independent of the shader's table values.
    for y in 20..40 {
        for x in 20..40 {
            let expected = if (x + y) % 2 == 1 { 255 } else { 0 };
            assert_eq!(
                &pixels[(y * 64 + x) * 4..(y * 64 + x) * 4 + 4],
                &[expected, expected, expected, 255]
            );
        }
    }
    let small = make_scene(0.13, 0.71, Color::WHITE, 0.0);
    let pixels = draw(
        &small,
        native.with_surface_effects(SurfaceEffects3d::new().with_vertex_snap([8, 8]).unwrap()),
    );
    for y in 0..64 {
        for x in 0..64 {
            let expected = if (40..56).contains(&x) && (8..24).contains(&y) {
                255
            } else {
                0
            };
            assert_eq!(
                &pixels[(y * 64 + x) * 4..(y * 64 + x) * 4 + 4],
                &[expected, expected, expected, 255],
                "snap pixel {x},{y}"
            );
        }
    }
    let near = make_scene(-0.9, 0.9, Color::WHITE, 0.3);
    let far = make_scene(-0.9, 0.9, Color::BLACK, -0.3);
    draw(&near, native);
    let loaded = draw(&far, native.with_existing_depth(true));
    assert_eq!(
        &loaded[(32 * 64 + 32) * 4..(32 * 64 + 32) * 4 + 4],
        &[255; 4]
    );
    let cleared = draw(&far, native);
    assert_eq!(
        &cleared[(32 * 64 + 32) * 4..(32 * 64 + 32) * 4 + 4],
        &[0, 0, 0, 255]
    );
    let rejected = renderer.render_scene3d(
        device,
        queue,
        &identity,
        &target,
        &near,
        camera,
        Mesh3dRenderBudget::default()
            .with_surface_effects(SurfaceEffects3d::new().with_affine_uv(true)),
    );
    assert!(matches!(
        rejected,
        Err(Mesh3dRenderError::UnsupportedSurfaceEffects)
    ));
    assert_eq!(surface::test_read_pixels(device, queue, &target), cleared);
}

fn affine_uv_oracle(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) {
    use crate::{
        ImageBudget, ImageSampling, Texture3dOptions, TextureCoordinate2d, TextureMaterial3d,
    };
    let identity = Arc::new(());
    let target = surface::test_target(device, &identity, format, 64, 64);
    let mut renderer = Mesh3dRenderer::new(device, format);
    let texels = (0..16)
        .flat_map(|y| (0..16).flat_map(move |x| [x * 16, y * 16, 64, 255]))
        .collect();
    let image = texture::create_texture_with_options(
        device,
        queue,
        &identity,
        &renderer.textures.layout,
        16,
        16,
        texels,
        ImageBudget::default(),
        Texture3dOptions::new(),
    )
    .unwrap();
    let material = TextureMaterial3d::new(&image, ImageSampling::Nearest, Color::WHITE).unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 3.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(0.1), world(10.0))
            .unwrap(),
    )
    .unwrap();
    for left in [-0.8f64, -1.4] {
        let ndc = [[left, -0.8], [0.8, -0.8], [0.0, 0.8]];
        let weights = [1.0, 2.0, 4.0];
        let vertices = std::array::from_fn::<_, 3, _>(|i| {
            Vec3::new(
                (ndc[i][0] * weights[i]) as f32,
                (ndc[i][1] * weights[i]) as f32,
                (3.0 - weights[i]) as f32,
            )
            .unwrap()
        })
        .to_vec();
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let mesh = Mesh3d::textured(
            vertices,
            uv.map(|[u, v]| TextureCoordinate2d::new(u, v).unwrap())
                .to_vec(),
            vec![0, 1, 2],
            vec![],
        )
        .unwrap();
        let mesh = create_retained_mesh(device, queue, identity.clone(), mesh).unwrap();
        let mesh = texture::attach_material(&identity, &mesh, &material).unwrap();
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        scene
            .try_push(
                &mesh,
                Transform3d::IDENTITY,
                MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
            )
            .unwrap();
        let screen = ndc.map(|[x, y]| [(x + 1.0) * 32.0, (1.0 - y) * 32.0]);
        let cross = |a: [f64; 2], b: [f64; 2], p: [f64; 2]| {
            (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        };
        let area = cross(screen[0], screen[1], screen[2]);
        let mut outputs = Vec::new();
        for affine in [false, true] {
            renderer
                .render_scene3d(
                    device,
                    queue,
                    &identity,
                    &target,
                    &scene,
                    camera,
                    Mesh3dRenderBudget::default()
                        .with_surface_policy(SurfaceRasterization3d::Native)
                        .with_surface_effects(SurfaceEffects3d::new().with_affine_uv(affine)),
                )
                .unwrap();
            let pixels = surface::test_read_pixels(device, queue, &target);
            let mut probes = 0;
            for y in 0..64 {
                for x in 0..64 {
                    let p = [x as f64 + 0.5, y as f64 + 0.5];
                    let bary = [
                        cross(screen[1], screen[2], p) / area,
                        cross(screen[2], screen[0], p) / area,
                        cross(screen[0], screen[1], p) / area,
                    ];
                    if bary.iter().any(|value| *value < 0.03) {
                        continue;
                    }
                    let mut coefficients = bary;
                    if !affine {
                        for i in 0..3 {
                            coefficients[i] /= weights[i];
                        }
                        let sum = coefficients.iter().sum::<f64>();
                        for value in &mut coefficients {
                            *value /= sum;
                        }
                    }
                    let sample = std::array::from_fn::<_, 2, _>(|axis| {
                        (0..3)
                            .map(|i| coefficients[i] * f64::from(uv[i][axis]))
                            .sum::<f64>()
                            * 16.0
                    });
                    if sample
                        .iter()
                        .any(|value| (value - value.round()).abs() < 0.02)
                    {
                        continue;
                    }
                    let expected = [
                        sample[0].floor() as u8 * 16,
                        sample[1].floor() as u8 * 16,
                        64,
                        255,
                    ];
                    let offset = (y * 64 + x) * 4;
                    let channels = if matches!(
                        format,
                        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
                    ) {
                        [2, 1, 0, 3]
                    } else {
                        [0, 1, 2, 3]
                    };
                    for channel in 0..4 {
                        assert!(
                            pixels[offset + channels[channel]].abs_diff(expected[channel]) <= 1,
                            "affine={affine} left={left} at {x},{y}: {:?} vs {expected:?}",
                            &pixels[offset..offset + 4]
                        );
                    }
                    probes += 1;
                }
            }
            assert!(probes > 300);
            outputs.push(pixels);
        }
        assert!(
            outputs[0]
                .chunks_exact(4)
                .zip(outputs[1].chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count()
                > 300
        );
    }
}
