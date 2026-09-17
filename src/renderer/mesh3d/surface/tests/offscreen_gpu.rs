//! Independent enabled/disabled pixel and transaction oracles for draw omission.

use super::*;
use crate::{
    AmbientLight3d, DirectionalLight3d, Fog3d, Lighting3d, Mesh3dAttributes, TextureCoordinate2d,
};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn point([x, y, z]: [f32; 3]) -> Vec3 {
    Vec3::new(x, y, z).unwrap()
}

fn camera(perspective: bool, x: f32) -> Camera3d {
    let projection = if perspective {
        Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(1.0), world(3.0))
    } else {
        Projection3d::orthographic(world(2.0), 1.0, world(1.0), world(3.0))
    }
    .unwrap();
    Camera3d::look_at(
        point([x, 0.0, 0.0]),
        point([x, 0.0, -1.0]),
        Vec3::Y,
        projection,
    )
    .unwrap()
}

fn transform(position: [f32; 3]) -> Transform3d {
    Transform3d::new(point(position), Rotation3d::IDENTITY, point([1.0; 3])).unwrap()
}

fn quad(edges: bool, attributes: bool) -> Mesh3d {
    let vertices = [
        [-0.25, -0.25, 0.0],
        [0.25, -0.25, 0.0],
        [0.25, 0.25, 0.0],
        [-0.25, 0.25, 0.0],
    ]
    .map(point)
    .to_vec();
    let edges = if edges {
        vec![
            MeshEdge3d::new(0, 1).unwrap(),
            MeshEdge3d::new(1, 2).unwrap(),
        ]
    } else {
        Vec::new()
    };
    let attributes = if attributes {
        Mesh3dAttributes::new()
            .with_texture_coordinates(
                [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                    .map(|[u, v]| TextureCoordinate2d::new(u, v).unwrap())
                    .to_vec(),
            )
            .with_vertex_colors(vec![Color::rgb(0.75, 1.0, 0.5); 4])
            .unwrap()
            .with_normals(vec![Vec3::Z; 4])
            .unwrap()
    } else {
        Mesh3dAttributes::new()
    };
    Mesh3d::with_attributes(vertices, vec![0, 1, 2, 0, 2, 3], edges, attributes).unwrap()
}

fn surface() -> MeshStyle3d {
    MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap())
}

fn native() -> Mesh3dRenderBudget {
    Mesh3dRenderBudget::new(0, 0, 0).with_surface_policy(SurfaceRasterization3d::Native)
}

struct Fixture<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    identity: Arc<()>,
    target: RenderTarget3d,
    enabled: Mesh3dRenderer,
    disabled: Mesh3dRenderer,
}

struct Pair {
    enabled: Mesh3dRenderReport,
    disabled: Mesh3dRenderReport,
    pixels: Vec<u8>,
}

impl<'a> Fixture<'a> {
    fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, identity: &Arc<()>) -> Self {
        Self {
            device,
            queue,
            identity: Arc::clone(identity),
            target: test_target(device, identity, FORMAT, 64, 64),
            enabled: Mesh3dRenderer::new(device, FORMAT),
            disabled: Mesh3dRenderer::new(device, FORMAT),
        }
    }

    fn mesh(&self, source: Mesh3d) -> RetainedMesh3d {
        create_retained_mesh(self.device, self.queue, Arc::clone(&self.identity), source).unwrap()
    }

    fn poison(&self) {
        // A skipped clear must not pass merely because the preceding frame had
        // identical pixels; depth zero also detects stale depth attachments.
        let mut encoder = self.device.create_command_encoder(&Default::default());
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offscreen omission oracle poison"),
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
    }

    fn pair(
        &mut self,
        label: &str,
        scene: &Scene3d,
        camera: Camera3d,
        budget: Mesh3dRenderBudget,
        omitted: [usize; 2],
    ) -> Pair {
        let statistics = scene.statistics();
        let visible = scene.visible_object_count();
        let disabled_budget = budget.with_offscreen_surface_culling(false);
        let enabled_budget = budget.with_offscreen_surface_culling(true);
        let checked = self
            .enabled
            .preflight_scene3d(
                self.device,
                &self.identity,
                &self.target,
                scene,
                camera,
                enabled_budget,
            )
            .unwrap_or_else(|error| panic!("{label}: immutable preflight: {error}"))
            .1
            .report;
        let disabled = self
            .disabled
            .render_scene3d(
                self.device,
                self.queue,
                &self.identity,
                &self.target,
                scene,
                camera,
                disabled_budget,
            )
            .unwrap_or_else(|error| panic!("{label}: disabled: {error}"));
        let expected = test_read_pixels(self.device, self.queue, &self.target);
        self.poison();
        let enabled = self
            .enabled
            .render_scene3d(
                self.device,
                self.queue,
                &self.identity,
                &self.target,
                scene,
                camera,
                enabled_budget,
            )
            .unwrap_or_else(|error| panic!("{label}: enabled: {error}"));
        let actual = test_read_pixels(self.device, self.queue, &self.target);
        assert_eq!(actual, expected, "{label}: complete color attachment");
        let report = enabled.preflight();
        let control = disabled.preflight();
        assert_eq!(report.culled_object_count(), omitted[0], "{label}");
        assert_eq!(report.culled_triangle_count(), omitted[1], "{label}");
        assert_eq!(checked.culled_object_count(), omitted[0], "{label}");
        assert_eq!(checked.culled_triangle_count(), omitted[1], "{label}");
        assert_eq!(control.culled_object_count(), 0, "{label}");
        assert_eq!(control.culled_triangle_count(), 0, "{label}");
        assert_eq!(
            enabled.object_count() + omitted[0],
            disabled.object_count(),
            "{label}"
        );
        assert_eq!(
            enabled.triangle_count() + omitted[1],
            disabled.triangle_count(),
            "{label}"
        );
        assert_eq!(
            enabled.triangle_count(),
            report.submitted_triangle_count(),
            "{label}"
        );
        assert_eq!(
            enabled.triangle_count(),
            checked.submitted_triangle_count(),
            "{label}"
        );
        assert_eq!(enabled.edge_count(), disabled.edge_count(), "{label}");
        assert_eq!(enabled.render_pass_count(), 1, "{label}");
        assert_eq!(
            report.generated_object_count(),
            control.generated_object_count(),
            "{label}"
        );
        assert_eq!(
            report.generated_vertex_count(),
            control.generated_vertex_count(),
            "{label}"
        );
        assert_eq!(
            report.generated_triangle_count(),
            control.generated_triangle_count(),
            "{label}"
        );
        assert_eq!(
            report.generated_upload_bytes(),
            control.generated_upload_bytes(),
            "{label}"
        );
        assert_eq!(
            report.generated_edge_count(),
            control.generated_edge_count(),
            "{label}"
        );
        assert_eq!(
            report.clipped_source_triangle_count(),
            control.clipped_source_triangle_count(),
            "{label}"
        );
        assert_eq!(
            report.discarded_source_triangle_count(),
            control.discarded_source_triangle_count(),
            "{label}"
        );
        assert_eq!(
            enabled.retained_frame_cpu_bytes(),
            disabled.retained_frame_cpu_bytes(),
            "{label}"
        );
        assert_eq!(
            enabled.retained_frame_gpu_bytes(),
            disabled.retained_frame_gpu_bytes(),
            "{label}"
        );
        assert_eq!(scene.statistics(), statistics);
        assert_eq!(scene.visible_object_count(), visible);
        Pair {
            enabled,
            disabled,
            pixels: actual,
        }
    }

    fn error_pair(
        &mut self,
        scene: &Scene3d,
        camera: Camera3d,
        budget: Mesh3dRenderBudget,
    ) -> Mesh3dRenderError {
        self.poison();
        let before = test_read_pixels(self.device, self.queue, &self.target);
        let statistics = scene.statistics();
        let mut expected = None;
        for enabled in [false, true] {
            let budget = budget.with_offscreen_surface_culling(enabled);
            let checked = self
                .enabled
                .preflight_scene3d(
                    self.device,
                    &self.identity,
                    &self.target,
                    scene,
                    camera,
                    budget,
                )
                .err()
                .unwrap();
            let renderer = if enabled {
                &mut self.enabled
            } else {
                &mut self.disabled
            };
            let error = renderer
                .render_scene3d(
                    self.device,
                    self.queue,
                    &self.identity,
                    &self.target,
                    scene,
                    camera,
                    budget,
                )
                .unwrap_err();
            assert_eq!(error, checked);
            assert_eq!(
                test_read_pixels(self.device, self.queue, &self.target),
                before
            );
            assert_eq!(scene.statistics(), statistics);
            if let Some(previous) = expected {
                assert_eq!(error, previous);
            }
            expected = Some(error);
        }
        expected.unwrap()
    }
}

fn planes_and_boundaries(fixture: &mut Fixture<'_>, mesh: &RetainedMesh3d) {
    for perspective in [false, true] {
        let side = if perspective { 4.0 } else { 2.0 };
        for (plane, position) in [
            [-side, 0.0, -2.0],
            [side, 0.0, -2.0],
            [0.0, -side, -2.0],
            [0.0, side, -2.0],
            [0.0, 0.0, -0.25],
            [0.0, 0.0, -4.0],
        ]
        .into_iter()
        .enumerate()
        {
            let mut scene = Scene3d::new(Color::BLACK).unwrap();
            scene
                .try_push(mesh, transform(position), surface())
                .unwrap();
            let pair = fixture.pair(
                &format!("outside plane {plane}, perspective {perspective}"),
                &scene,
                camera(perspective, 0.0),
                native().with_max_surface_triangles(2),
                [1, 2],
            );
            assert_eq!(pair.disabled.draw_call_count(), 1);
            assert_eq!(pair.enabled.draw_call_count(), 0);
            assert!(
                pair.pixels
                    .chunks_exact(4)
                    .all(|pixel| pixel == [0, 0, 0, 255])
            );
            assert_eq!(
                pair.enabled.preflight().discarded_source_triangle_count(),
                None
            );
            assert_eq!(pair.enabled.preflight().generated_object_count(), 0);
            if plane == 0 {
                assert_eq!(
                    fixture.error_pair(
                        &scene,
                        camera(perspective, 0.0),
                        native().with_max_surface_triangles(1)
                    ),
                    Mesh3dRenderError::SurfaceTriangleBudgetExceeded {
                        limit: 1,
                        actual: 2
                    }
                );
                assert_eq!(
                    fixture.error_pair(
                        &scene,
                        camera(perspective, 0.0),
                        native().with_max_surface_triangles(0)
                    ),
                    Mesh3dRenderError::SurfaceTriangleBudgetExceeded {
                        limit: 0,
                        actual: 2
                    }
                );
            }
        }
    }
    for (plane, position) in [
        [-1.25, 0.0, -2.0],
        [1.25, 0.0, -2.0],
        [0.0, -1.25, -2.0],
        [0.0, 1.25, -2.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -3.0],
    ]
    .into_iter()
    .enumerate()
    {
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        scene
            .try_push(mesh, transform(position), surface())
            .unwrap();
        let pair = fixture.pair(
            &format!("touching plane {plane}"),
            &scene,
            camera(false, 0.0),
            native(),
            [0, 0],
        );
        assert_eq!(pair.enabled.draw_call_count(), 1);
    }
    for plane in 0..6 {
        let vertices = if plane < 4 {
            let orient = |[x, y, z]: [f32; 3]| match plane {
                0 => [x, y, z],
                1 => [-x, y, z],
                2 => [y, x, z],
                _ => [y, -x, z],
            };
            [[-1.5, -0.25, -2.0], [0.0, -0.25, -2.0], [0.0, 0.25, -2.0]].map(orient)
        } else {
            let outside = if plane == 4 { -0.5 } else { -3.5 };
            [
                [-0.25, -0.25, outside],
                [0.25, -0.25, -2.0],
                [0.0, 0.25, -2.0],
            ]
        };
        let crossing =
            fixture.mesh(Mesh3d::new(vertices.map(point).to_vec(), vec![0, 1, 2]).unwrap());
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        scene
            .try_push(&crossing, Transform3d::IDENTITY, surface())
            .unwrap();
        let pair = fixture.pair(
            &format!("crossing plane {plane}"),
            &scene,
            camera(false, 0.0),
            native(),
            [0, 0],
        );
        assert!(pair.pixels.chunks_exact(4).any(|pixel| pixel[0] > 0));
    }
    let crossing = fixture.mesh(
        Mesh3d::new(
            [[-0.5, -0.5, -2.0], [0.5, -0.5, -2.0], [0.0, 0.5, 1.0]]
                .map(point)
                .to_vec(),
            vec![0, 1, 2],
        )
        .unwrap(),
    );
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene
        .try_push(&crossing, Transform3d::IDENTITY, surface())
        .unwrap();
    fixture.pair(
        "crossing camera plane",
        &scene,
        camera(true, 0.0),
        native(),
        [0, 0],
    );
}

fn motion_batches_and_edges(fixture: &mut Fixture<'_>, mesh: &RetainedMesh3d) {
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let id = scene
        .try_push(mesh, transform([2.0, 0.0, -2.0]), surface())
        .unwrap();
    for (camera_x, omitted) in [(0.0, [1, 2]), (2.0, [0, 0]), (0.0, [1, 2])] {
        fixture.pair(
            "camera A-B-A",
            &scene,
            camera(false, camera_x),
            native(),
            omitted,
        );
    }
    for position in [[0.0, 0.0, -2.0], [2.0, 2.0, -2.0], [0.0, 0.0, -2.0]] {
        let outside = position[0] != 0.0;
        scene
            .set_transform(
                id,
                Transform3d::new(
                    point(position),
                    Rotation3d::from_euler_xyz(0.0, 0.0, 0.3).unwrap(),
                    point([0.75, 1.25, 0.5]),
                )
                .unwrap(),
            )
            .unwrap();
        fixture.pair(
            "rotated nonuniform motion",
            &scene,
            camera(false, 0.0),
            native(),
            if outside { [1, 2] } else { [0, 0] },
        );
    }
    scene
        .set_transform(
            id,
            Transform3d::new(
                point([2.0, 0.0, -2.0]),
                Rotation3d::from_euler_xyz(0.0, 0.0, 0.3).unwrap(),
                point([0.75, 1.25, 0.5]),
            )
            .unwrap(),
        )
        .unwrap();
    // Rotated Y bounds straddle zero, so the whole-bounds arithmetic proof
    // cannot exclude interior cancellation even though the object is offscreen.
    let uncertain = fixture.pair(
        "inconclusive rotated bounds retain the draw",
        &scene,
        camera(false, 0.0),
        native(),
        [0, 0],
    );
    assert_eq!(uncertain.enabled.draw_call_count(), 1);
    scene.set_visible(id, false).unwrap();
    let pair = fixture.pair(
        "hidden object is not culled",
        &scene,
        camera(false, 0.0),
        native(),
        [0, 0],
    );
    assert_eq!(pair.enabled.object_count(), 0);
    for (positions, omitted, draws) in [
        ([2.0, 3.0, 2.0, 3.0], [4, 8], 0),
        ([0.0, 2.0, 0.0, 2.0], [2, 4], 2),
        ([0.0; 4], [0, 0], 1),
    ] {
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        for x in positions {
            scene
                .try_push(mesh, transform([x, 0.0, -2.0]), surface())
                .unwrap();
        }
        let pair = fixture.pair(
            "shared retained batch",
            &scene,
            camera(false, 0.0),
            native(),
            omitted,
        );
        assert_eq!(pair.disabled.draw_call_count(), 1);
        assert_eq!(pair.enabled.draw_call_count(), draws);
        assert_eq!(scene.visible_object_count(), 4);
    }
    let separate = fixture.mesh(mesh.source().clone());
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    for mesh in [mesh, &separate] {
        scene
            .try_push(mesh, transform([2.0, 0.0, -2.0]), surface())
            .unwrap();
    }
    let pair = fixture.pair(
        "distinct retained uploads",
        &scene,
        camera(false, 0.0),
        native(),
        [2, 4],
    );
    assert_eq!(pair.disabled.draw_call_count(), 2);
    assert_eq!(pair.enabled.draw_call_count(), 0);

    let reflected = fixture.mesh(
        Mesh3d::new(
            mesh.source()
                .vertices()
                .iter()
                .map(|vertex| point([-vertex.x(), vertex.y(), vertex.z()]))
                .collect(),
            mesh.source().triangle_indices().to_vec(),
        )
        .unwrap(),
    );
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene
        .try_push(&reflected, transform([2.0, 0.0, -2.0]), surface())
        .unwrap();
    fixture.pair(
        "reflected source winding",
        &scene,
        camera(false, 0.0),
        native(),
        [1, 2],
    );

    let edged = fixture.mesh(quad(true, false));
    let edge_style = WireframeStyle3d::visible(Color::WHITE, logical(12.0))
        .unwrap()
        .with_hidden(
            Color::rgb(1.0, 0.0, 0.0),
            logical(16.0),
            logical(3.0),
            logical(3.0),
        )
        .unwrap();
    // A thick near-border edge is still strictly inside. An endpoint exactly
    // on the clip plane has its separate strict rejection control below.
    for x in [2.0, 0.65] {
        for style in [
            surface().with_wireframe(edge_style),
            MeshStyle3d::wireframe(edge_style),
        ] {
            let mut scene = Scene3d::new(Color::BLACK).unwrap();
            scene
                .try_push(&edged, transform([x, 0.0, -2.0]), style)
                .unwrap();
            let pair = fixture.pair(
                &format!("thick visible and hidden edges stay submitted at x={x}"),
                &scene,
                camera(false, 0.0),
                native(),
                [0, 0],
            );
            assert_eq!(
                pair.enabled.draw_call_count(),
                pair.disabled.draw_call_count()
            );
            assert_eq!(pair.enabled.edge_count(), 2);
        }
    }
    let mut boundary = Scene3d::new(Color::BLACK).unwrap();
    let id = boundary
        .try_push(
            &edged,
            transform([0.75, 0.0, -2.0]),
            surface().with_wireframe(edge_style),
        )
        .unwrap();
    assert_eq!(
        fixture
            .error_pair(&boundary, camera(false, 0.0), native())
            .object_id(),
        Some(id)
    );
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene
        .try_push(
            mesh,
            transform([2.0, 0.0, -2.0]),
            surface().with_wireframe(edge_style),
        )
        .unwrap();
    fixture.pair(
        "wireframe policy without display edges stays excluded",
        &scene,
        camera(false, 0.0),
        native(),
        [0, 0],
    );
}

fn material_order_and_strict(fixture: &mut Fixture<'_>, mesh: &RetainedMesh3d) {
    let textured = fixture.mesh(quad(false, true));
    let materials = [[255, 32, 64, 128], [32, 255, 64, 192]].map(|pixels| {
        let texture = texture::create_texture_with_alpha(
            fixture.device,
            fixture.queue,
            &fixture.identity,
            &fixture.enabled.textures.layout,
            1,
            1,
            pixels.to_vec(),
            ImageBudget::default(),
        )
        .unwrap();
        TextureMaterial3d::with_alpha(&texture, ImageSampling::Nearest, Color::WHITE).unwrap()
    });
    let meshes = materials
        .each_ref()
        .map(|material| texture::attach_material(&fixture.identity, &textured, material).unwrap());
    for style in [
        SurfaceStyle3d::opaque(Color::WHITE).unwrap(),
        SurfaceStyle3d::mask(Color::WHITE, 0.25).unwrap(),
        SurfaceStyle3d::blend(Color::WHITE).unwrap(),
    ] {
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        scene.set_lighting(
            Lighting3d::new(AmbientLight3d::new(Color::WHITE, 0.25).unwrap()).with_directional(
                Some(DirectionalLight3d::new(Vec3::Z, Color::WHITE, 0.5).unwrap()),
            ),
        );
        scene.set_fog(Some(
            Fog3d::new(Color::rgb(0.1, 0.2, 0.3), 1.0, 0.2).unwrap(),
        ));
        let style = MeshStyle3d::surface(
            style
                .with_lighting(SurfaceLighting3d::Lambert)
                .with_fog(true),
        );
        for (index, x) in [0.0, 2.0, 0.0, 2.0].into_iter().enumerate() {
            scene
                .try_push(&meshes[index / 2], transform([x, 0.0, -2.0]), style)
                .unwrap();
        }
        let exact_sorting = 4 * std::mem::size_of::<material::SurfaceDraw>();
        let budget = native()
            .with_max_surface_triangles(8)
            .with_max_sorting_bytes(exact_sorting);
        let pair = fixture.pair(
            "textured colored Lambert fog material and depth ties",
            &scene,
            camera(false, 0.0),
            budget,
            [2, 4],
        );
        assert!(
            pair.pixels
                .chunks_exact(4)
                .filter(|pixel| pixel[0] > 0)
                .count()
                > 64
        );
        assert_eq!(
            fixture.error_pair(
                &scene,
                camera(false, 0.0),
                budget.with_max_surface_triangles(7)
            ),
            Mesh3dRenderError::SurfaceTriangleBudgetExceeded {
                limit: 7,
                actual: 8
            }
        );
        if style.surface_style().unwrap().alpha_mode() == SurfaceAlphaMode3d::Blend {
            assert!(matches!(
                fixture.error_pair(
                    &scene,
                    camera(false, 0.0),
                    budget.with_max_sorting_bytes(exact_sorting - 1)
                ),
                Mesh3dRenderError::SortingBudgetExceeded { .. }
            ));
        }
    }
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    for x in [0.0, 0.875, 2.0] {
        scene
            .try_push(mesh, transform([x, 0.0, -2.0]), surface())
            .unwrap();
    }
    let strict = Mesh3dRenderBudget::default();
    let counts = fixture
        .disabled
        .preflight_scene3d(
            fixture.device,
            &fixture.identity,
            &fixture.target,
            &scene,
            camera(false, 0.0),
            strict,
        )
        .unwrap()
        .1
        .report;
    assert!(counts.generated_vertex_count() > 0);
    let exact = Mesh3dRenderBudget::new(
        counts.generated_vertex_count(),
        counts.generated_triangle_count(),
        counts.generated_upload_bytes(),
    )
    .with_max_surface_triangles(counts.submitted_triangle_count());
    let pair = fixture.pair(
        "Strict retained/generated/outside unchanged",
        &scene,
        camera(false, 0.0),
        exact,
        [0, 0],
    );
    assert_eq!(
        pair.enabled.draw_call_count(),
        pair.disabled.draw_call_count()
    );
    for budget in [
        Mesh3dRenderBudget::new(
            counts.generated_vertex_count() - 1,
            counts.generated_triangle_count(),
            counts.generated_upload_bytes(),
        ),
        Mesh3dRenderBudget::new(
            counts.generated_vertex_count(),
            counts.generated_triangle_count() - 1,
            counts.generated_upload_bytes(),
        ),
        Mesh3dRenderBudget::new(
            counts.generated_vertex_count(),
            counts.generated_triangle_count(),
            counts.generated_upload_bytes() - 1,
        ),
    ] {
        fixture.error_pair(&scene, camera(false, 0.0), budget);
    }
    fixture.pair(
        "Native after generated frame",
        &scene,
        camera(false, 0.0),
        native(),
        [1, 2],
    );
    fixture.pair(
        "Strict after culled frame",
        &scene,
        camera(false, 0.0),
        exact,
        [0, 0],
    );
}

fn invalid_outside_sources(fixture: &mut Fixture<'_>, mesh: &RetainedMesh3d) {
    let positions = [
        [-0.5, -0.25, -2.0],
        [-0.25, -0.25, -2.0],
        [-0.25, 0.25, -2.0],
        [-3.0, 0.0, -2.0],
    ];
    let invalid_transform = Transform3d::new(
        Vec3::ZERO,
        Rotation3d::IDENTITY,
        point([2.0_f32.powi(119), 1.0, 1.0]),
    )
    .unwrap();
    for indexed in [false, true] {
        let invalid = fixture.mesh(
            Mesh3d::new(
                positions.map(point).to_vec(),
                if indexed {
                    vec![0, 1, 2, 0, 2, 3]
                } else {
                    vec![0, 1, 2]
                },
            )
            .unwrap(),
        );
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        scene
            .try_push(mesh, transform([2.0, 0.0, -2.0]), surface())
            .unwrap();
        let bad = scene
            .try_push(&invalid, invalid_transform, surface())
            .unwrap();
        let error = fixture.error_pair(&scene, camera(false, 0.0), native());
        assert_eq!(error.object_id(), Some(bad));
        assert_eq!(
            error.surface_reason(),
            Some(Mesh3dSurfaceError::TransformArithmetic)
        );
        assert_eq!(
            error.source_vertex_index(),
            if indexed { None } else { Some(3) }
        );
        assert_eq!(
            error.source_triangle_index(),
            if indexed { Some(1) } else { None }
        );
        assert_eq!(
            fixture.error_pair(
                &scene,
                camera(false, 0.0),
                native().with_max_surface_triangles(1)
            ),
            Mesh3dRenderError::SurfaceTriangleBudgetExceeded {
                limit: 1,
                actual: 2
            }
        );
        scene.set_visible(bad, false).unwrap();
        fixture.pair(
            "late failure followed by all-outside success",
            &scene,
            camera(false, 0.0),
            native(),
            [1, 2],
        );
        scene
            .set_transform(scene.instances()[0].id(), transform([0.0, 0.0, -2.0]))
            .unwrap();
        fixture.pair(
            "late failure followed by inside success",
            &scene,
            camera(false, 0.0),
            native(),
            [0, 0],
        );
    }
    let edged = fixture.mesh(quad(true, false));
    let unsafe_dash = WireframeStyle3d::visible(Color::WHITE, logical(1.0))
        .unwrap()
        .with_hidden(
            Color::WHITE,
            logical(1.0),
            logical(f32::MIN_POSITIVE),
            logical(f32::MIN_POSITIVE),
        )
        .unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let bad = scene
        .try_push(
            &edged,
            transform([2.0, 0.0, -2.0]),
            surface().with_wireframe(unsafe_dash),
        )
        .unwrap();
    assert_eq!(
        fixture
            .error_pair(&scene, camera(false, 0.0), native())
            .object_id(),
        Some(bad)
    );

    let foreign = create_retained_mesh(
        fixture.device,
        fixture.queue,
        Arc::new(()),
        quad(false, false),
    )
    .unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene
        .try_push(&foreign, transform([2.0, 0.0, -2.0]), surface())
        .unwrap();
    assert!(matches!(
        fixture.error_pair(&scene, camera(false, 0.0), native()),
        Mesh3dRenderError::RendererMismatch
    ));

    let lit = fixture.mesh(
        Mesh3d::with_attributes(
            mesh.source().vertices().to_vec(),
            mesh.source().triangle_indices().to_vec(),
            Vec::new(),
            Mesh3dAttributes::new()
                .with_normals(vec![Vec3::Y; mesh.source().vertices().len()])
                .unwrap(),
        )
        .unwrap(),
    );
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene.set_lighting(Lighting3d::default().with_directional(Some(
        DirectionalLight3d::new(Vec3::Z, Color::WHITE, 1.0).unwrap(),
    )));
    let bad = scene
        .try_push(
            &lit,
            Transform3d::new(
                point([3.0, 0.0, -2.0]),
                Rotation3d::IDENTITY,
                point([1e-20, 1e20, 1.0]),
            )
            .unwrap(),
            MeshStyle3d::surface(
                SurfaceStyle3d::opaque(Color::WHITE)
                    .unwrap()
                    .with_lighting(SurfaceLighting3d::Lambert),
            ),
        )
        .unwrap();
    let error = fixture.error_pair(&scene, camera(false, 0.0), native());
    assert_eq!(error.object_id(), Some(bad));
    assert_eq!(
        error.surface_reason(),
        Some(Mesh3dSurfaceError::NormalTransform)
    );
    assert_eq!(error.source_vertex_index(), Some(0));

    let mut vertices = mesh.source().vertices().to_vec();
    vertices.push(point([f32::MIN_POSITIVE, 0.0, 0.0]));
    let fogged =
        fixture.mesh(Mesh3d::new(vertices, mesh.source().triangle_indices().to_vec()).unwrap());
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    scene.set_fog(Some(Fog3d::new(Color::WHITE, 1.0, 0.2).unwrap()));
    let bad = scene
        .try_push(
            &fogged,
            Transform3d::new(
                point([2.0, 0.0, -2.0]),
                Rotation3d::IDENTITY,
                point([0.5, 1.0, 1.0]),
            )
            .unwrap(),
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap().with_fog(true)),
        )
        .unwrap();
    let error = fixture.error_pair(&scene, camera(false, 0.0), native());
    assert_eq!(error.object_id(), Some(bad));
    assert_eq!(
        error.surface_reason(),
        Some(Mesh3dSurfaceError::FogArithmetic)
    );
    assert_eq!(error.source_vertex_index(), Some(4));
}

pub(in crate::renderer::mesh3d) fn assert_gpu_offscreen_culling(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
) {
    assert!(!Mesh3dRenderBudget::default().offscreen_surface_culling());
    let mut fixture = Fixture::new(device, queue, identity);
    let mesh = fixture.mesh(quad(false, false));
    planes_and_boundaries(&mut fixture, &mesh);
    motion_batches_and_edges(&mut fixture, &mesh);
    material_order_and_strict(&mut fixture, &mesh);
    invalid_outside_sources(&mut fixture, &mesh);
}

pub(in crate::renderer::mesh3d) fn assert_gpu_offscreen_culling_recovery(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
) {
    let mut source = Fixture::new(device, queue, &Arc::new(()));
    let mesh = source.mesh(quad(false, false));
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let outside = scene
        .try_push(&mesh, transform([2.0, 0.0, -2.0]), surface())
        .unwrap();
    let inside = scene
        .try_push(&mesh, transform([0.0, 0.0, -2.0]), surface())
        .unwrap();
    let hidden = scene
        .try_push(&mesh, transform([2.0, 0.0, -2.0]), surface())
        .unwrap();
    scene.set_visible(hidden, false).unwrap();
    let before = source.pair(
        "before replacement device",
        &scene,
        camera(false, 0.0),
        native(),
        [1, 2],
    );
    let identity = Arc::new(());
    let mut replacement = Fixture::new(recovery_device, recovery_queue, &identity);
    let restored = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        &replacement.enabled.textures.layout,
        Arc::clone(&identity),
        &mut scene,
    )
    .unwrap();
    assert_eq!(restored.restored_mesh_count(), 1);
    let after = replacement.pair(
        "after replacement device",
        &scene,
        camera(false, 0.0),
        native(),
        [1, 2],
    );
    assert_eq!(before.pixels, after.pixels);
    assert_eq!(scene.instance(outside).unwrap().id(), outside);
    assert_eq!(scene.instance(inside).unwrap().id(), inside);
    assert!(!scene.instance(hidden).unwrap().is_visible());
    assert_eq!(
        restore_scene3d_resources(
            recovery_device,
            recovery_queue,
            &replacement.enabled.textures.layout,
            Arc::clone(&identity),
            &mut scene
        )
        .unwrap()
        .restored_mesh_count(),
        0
    );
    replacement.pair(
        "same-device recovery repeats",
        &scene,
        camera(false, 0.0),
        native(),
        [1, 2],
    );
    replacement.pair(
        "camera moves to previously omitted object",
        &scene,
        camera(false, 2.0),
        native(),
        [1, 2],
    );
    scene
        .set_transform(outside, transform([0.0, 0.0, -2.0]))
        .unwrap();
    replacement.pair(
        "restored object moves into view",
        &scene,
        camera(false, 0.0),
        native(),
        [0, 0],
    );
}
