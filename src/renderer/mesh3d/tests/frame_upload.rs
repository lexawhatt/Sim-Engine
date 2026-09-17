//! Dirty mirror uploads versus a control that forgets every prior upload.

use super::*;
use crate::{
    AmbientLight3d, DirectionalLight3d, Mesh3dAttributes, TextureAddressMode3d,
    TextureCoordinate2d, TextureUvTransform3d,
};

struct Fixture<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    identity: Arc<()>,
    target: RenderTarget3d,
    optimized: Mesh3dRenderer,
    control: Mesh3dRenderer,
    camera: Camera3d,
}

impl Fixture<'_> {
    fn step(
        &mut self,
        label: &str,
        scene: &Scene3d,
        budget: Mesh3dRenderBudget,
    ) -> Mesh3dRenderReport {
        // Without this poison, an incorrectly skipped pass could leave the
        // previous image intact and falsely satisfy an unchanged-frame oracle.
        let mut encoder = self.device.create_command_encoder(&Default::default());
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dirty frame oracle poison"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target.color.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 0.0,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        }));
        self.queue.submit([encoder.finish()]);
        let report = self
            .optimized
            .render_scene3d(
                self.device,
                self.queue,
                &self.identity,
                &self.target,
                scene,
                self.camera,
                budget,
            )
            .unwrap_or_else(|error| panic!("{label}: optimized: {error}"));
        let actual = surface::test_read_pixels(self.device, self.queue, &self.target);
        self.control.invalidate_upload_history();
        let control = self
            .control
            .render_scene3d(
                self.device,
                self.queue,
                &self.identity,
                &self.target,
                scene,
                self.camera,
                budget,
            )
            .unwrap_or_else(|error| panic!("{label}: control: {error}"));
        let expected = surface::test_read_pixels(self.device, self.queue, &self.target);
        assert_eq!(actual, expected, "{label}: pixels");
        assert_eq!(report.object_count(), control.object_count(), "{label}");
        assert_eq!(report.triangle_count(), control.triangle_count(), "{label}");
        assert_eq!(report.edge_count(), control.edge_count(), "{label}");
        assert_eq!(
            report.draw_call_count(),
            control.draw_call_count(),
            "{label}"
        );
        assert_eq!(report.render_pass_count(), 1, "{label}");
        assert!(
            report.uploaded_bytes() <= control.uploaded_bytes(),
            "{label}"
        );
        assert!(report.upload_calls() <= control.upload_calls(), "{label}");
        report
    }
}

fn expect_upload(report: Mesh3dRenderReport, calls: usize, bytes: usize) {
    assert_eq!(report.upload_calls(), calls);
    assert_eq!(report.uploaded_bytes(), bytes);
}

fn transform(x: f32, y: f32, scale: f32) -> Transform3d {
    Transform3d::new(
        Vec3::new(x, y, 0.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(scale, scale, scale).unwrap(),
    )
    .unwrap()
}

pub(super) fn assert_gpu_dirty_frame_upload_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let identity = Arc::new(());
    let camera_a = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 3.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(8.0)).unwrap(),
    )
    .unwrap();
    let mut fixture = Fixture {
        device,
        queue,
        target: surface::test_target(device, &identity, format, 64, 64),
        optimized: Mesh3dRenderer::new(device, format),
        control: Mesh3dRenderer::new(device, format),
        identity,
        camera: camera_a,
    };
    let mesh = create_retained_mesh(
        device,
        queue,
        Arc::clone(&fixture.identity),
        Mesh3d::with_attributes(
            vec![
                Vec3::new(-0.2, -0.35, 0.0).unwrap(),
                Vec3::new(0.2, -0.35, 0.0).unwrap(),
                Vec3::new(0.2, 0.35, 0.0).unwrap(),
                Vec3::new(-0.2, 0.35, 0.0).unwrap(),
            ],
            vec![0, 1, 2, 0, 2, 3],
            vec![MeshEdge3d::new(0, 1).unwrap()],
            Mesh3dAttributes::new()
                .with_texture_coordinates(
                    [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                        .map(|[u, v]| TextureCoordinate2d::new(u, v).unwrap())
                        .to_vec(),
                )
                .with_vertex_colors(vec![Color::WHITE; 4])
                .unwrap()
                .with_normals(vec![Vec3::Z; 4])
                .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let edge = WireframeStyle3d::visible(Color::WHITE, logical(2.0)).unwrap();
    let style =
        MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()).with_wireframe(edge);
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let ids: Vec<_> = [-0.5, 0.0, 0.5]
        .into_iter()
        .map(|x| {
            scene
                .try_push(&mesh, transform(x, 0.0, 1.0), style)
                .unwrap()
        })
        .collect();
    let native = Mesh3dRenderBudget::default().with_surface_policy(SurfaceRasterization3d::Native);
    let camera_bytes = std::mem::size_of::<Camera3dUniform>();
    let instance_bytes = std::mem::size_of::<MeshInstanceGpu>();
    let edge_bytes = std::mem::size_of::<EdgeObjectUniform>();
    assert!(
        fixture
            .step("first native", &scene, native)
            .uploaded_bytes()
            > 0
    );
    expect_upload(
        fixture.step("identical native still clears", &scene, native),
        0,
        0,
    );

    fixture.camera = Camera3d::look_at(
        Vec3::new(0.3, 0.1, 3.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        camera_a.projection(),
    )
    .unwrap();
    expect_upload(fixture.step("camera B", &scene, native), 1, camera_bytes);
    fixture.camera = camera_a;
    expect_upload(
        fixture.step("camera A again", &scene, native),
        1,
        camera_bytes,
    );
    let mut alternate = Scene3d::new(Color::rgb(0.05, 0.1, 0.15)).unwrap();
    for x in [-0.5, 0.0, 0.5] {
        alternate
            .try_push(&mesh, transform(x, 0.2, 0.7), style)
            .unwrap();
    }
    fixture.step("different scene same object count", &alternate, native);
    fixture.step("original scene restored", &scene, native);
    scene.set_visible(ids[1], false).unwrap();
    fixture.step(
        "middle object hidden compresses visible indices",
        &scene,
        native,
    );
    scene.set_visible(ids[1], true).unwrap();
    fixture.step("middle object visible again", &scene, native);

    fixture.target.logical_viewport = LogicalViewport::new(32.0, 32.0).unwrap();
    fixture.target.pixels_per_logical = physical_per_logical(2.0);
    expect_upload(
        fixture.step("target scale two", &scene, native),
        1,
        camera_bytes,
    );
    fixture.target.logical_viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    fixture.target.pixels_per_logical = physical_per_logical(1.0);
    expect_upload(
        fixture.step("target scale one again", &scene, native),
        1,
        camera_bytes,
    );
    scene
        .set_transform(ids[1], transform(0.0, 0.15, 0.75))
        .unwrap();
    expect_upload(
        fixture.step("one transform record", &scene, native),
        2,
        instance_bytes + edge_bytes,
    );

    let texture = texture::create_texture_with_alpha(
        device,
        queue,
        &fixture.identity,
        &fixture.optimized.textures.layout,
        2,
        2,
        vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ],
        ImageBudget::default(),
    )
    .unwrap();
    let material = TextureMaterial3d::new(&texture, ImageSampling::Nearest, Color::WHITE).unwrap();
    scene.set_texture_material(ids[1], Some(&material)).unwrap();
    expect_upload(
        fixture.step("binding changes without instance bytes", &scene, native),
        0,
        0,
    );
    let material =
        TextureMaterial3d::new(&texture, ImageSampling::Nearest, Color::rgb(0.5, 1.0, 0.75))
            .unwrap()
            .with_uv_transform(
                TextureUvTransform3d::new(Vec2::splat(2.0), Vec2::new(0.25, -0.25)).unwrap(),
            )
            .with_address_mode(TextureAddressMode3d::Repeat);
    scene.set_texture_material(ids[1], Some(&material)).unwrap();
    expect_upload(
        fixture.step("UV tint and addressing", &scene, native),
        1,
        instance_bytes,
    );
    scene
        .set_style(
            ids[1],
            MeshStyle3d::surface(SurfaceStyle3d::blend(Color::rgba(0.8, 1.0, 0.7, 0.5)).unwrap())
                .with_wireframe(edge),
        )
        .unwrap();
    expect_upload(
        fixture.step("surface alpha style", &scene, native),
        1,
        instance_bytes,
    );
    let thicker_edge = WireframeStyle3d::visible(Color::rgb(1.0, 0.5, 0.0), logical(4.0))
        .unwrap()
        .with_hidden(Color::WHITE, logical(1.0), logical(3.0), logical(2.0))
        .unwrap();
    scene.set_wireframe(ids[1], Some(thicker_edge)).unwrap();
    expect_upload(
        fixture.step("edge style only", &scene, native),
        1,
        edge_bytes,
    );
    scene
        .set_style(
            ids[1],
            MeshStyle3d::surface(
                SurfaceStyle3d::opaque(Color::WHITE)
                    .unwrap()
                    .with_lighting(SurfaceLighting3d::Lambert)
                    .with_fog(true),
            )
            .with_wireframe(thicker_edge),
        )
        .unwrap();
    scene.set_lighting(
        Lighting3d::new(AmbientLight3d::new(Color::WHITE, 0.2).unwrap()).with_directional(Some(
            DirectionalLight3d::new(Vec3::Z, Color::WHITE, 0.6).unwrap(),
        )),
    );
    scene.set_fog(Some(
        Fog3d::new(Color::rgb(0.1, 0.2, 0.4), 1.0, 0.3).unwrap(),
    ));
    fixture.step("lighting and fog environment", &scene, native);
    expect_upload(fixture.step("unchanged lit frame", &scene, native), 0, 0);

    let before_failure = surface::test_read_pixels(device, queue, &fixture.target);
    assert!(
        fixture
            .optimized
            .render_scene3d(
                device,
                queue,
                &fixture.identity,
                &fixture.target,
                &scene,
                fixture.camera,
                native.with_max_surface_triangles(0)
            )
            .is_err()
    );
    assert_eq!(
        surface::test_read_pixels(device, queue, &fixture.target),
        before_failure
    );
    expect_upload(
        fixture.step("valid frame after rejected budget", &scene, native),
        0,
        0,
    );

    scene
        .set_transform(ids[1], transform(-0.95, 0.15, 1.0))
        .unwrap();
    for (label, budget) in [
        ("generated strict", Mesh3dRenderBudget::default()),
        ("generated strict repeated", Mesh3dRenderBudget::default()),
        ("native after generated", native),
        ("generated after native", Mesh3dRenderBudget::default()),
    ] {
        let report = fixture.step(label, &scene, budget);
        assert_eq!(
            report.preflight().generated_vertex_count() > 0,
            budget.surface_policy() == SurfaceRasterization3d::StrictPortable,
            "{label}"
        );
    }

    let mut growth = Scene3d::new(Color::BLACK).unwrap();
    let growth_ids: Vec<_> = (0..33)
        .map(|index| {
            growth
                .try_push(
                    &mesh,
                    transform((index % 3) as f32 * 0.5 - 0.5, 0.0, 1.0),
                    style,
                )
                .unwrap()
        })
        .collect();
    let grown = fixture.step("growth replaces GPU and staging", &growth, native);
    assert_eq!(grown.buffer_allocation_count(), 2);
    expect_upload(fixture.step("grown unchanged", &growth, native), 0, 0);
    let retained_bytes = grown.retained_frame_gpu_bytes();
    for id in &growth_ids[3..] {
        growth.set_visible(*id, false).unwrap();
    }
    expect_upload(
        fixture.step("shrink without record changes", &growth, native),
        0,
        0,
    );
    for id in &growth_ids[3..] {
        growth.set_visible(*id, true).unwrap();
    }
    let regrown = fixture.step("regrowth initializes newly active slots", &growth, native);
    assert_eq!(regrown.buffer_allocation_count(), 0);
    assert_eq!(regrown.retained_frame_gpu_bytes(), retained_bytes);
    assert!(regrown.uploaded_bytes() > 0);
    let empty = Scene3d::new(Color::BLACK).unwrap();
    expect_upload(
        fixture.step("empty scene still clears", &empty, native),
        0,
        0,
    );
    fixture.step("return from empty", &growth, native);
    fixture.optimized = Mesh3dRenderer::new(device, format);
    assert!(
        fixture
            .step("fresh renderer has no upload residency", &growth, native)
            .uploaded_bytes()
            > 0
    );
    expect_upload(
        fixture.step("fresh renderer repeated", &growth, native),
        0,
        0,
    );
}
