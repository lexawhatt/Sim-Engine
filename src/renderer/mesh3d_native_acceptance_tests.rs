//! Engine-only ready-geometry acceptance, not a reconstruction of Logic terrain.
//!
//! The original Meadow/second-region probe sources are unavailable. The exact
//! floor and triangles 63/78 remain in `mesh3d_policy_tests`; this fixture adds
//! two explicit visual regions, add/remove revisions and 96 camera visits per
//! revision. Its f64 ray oracle does not call the production clipper or shader
//! transform helpers. Boundary coverage remains adapter-specific in Native.

use super::*;

const WIDTH: u32 = 110;
const HEIGHT: u32 = 72;
const COLORS: [Color; 6] = [
    Color::rgb(0.2304, 0.254_399_99, 0.288_000_02),
    Color::rgb(0.12, 0.38, 0.2),
    Color::rgb(0.8, 0.12, 0.06),
    Color::rgb(0.08, 0.22, 0.8),
    Color::rgb(0.75, 0.5, 0.08),
    Color::rgb(0.55, 0.12, 0.6),
];

fn point(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2]).unwrap()
}

fn array(point: Vec3) -> [f64; 3] {
    [point.x(), point.y(), point.z()].map(f64::from)
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| a[axis] - b[axis])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|axis| a[axis] * b[axis]).sum()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalized(value: [f64; 3]) -> [f64; 3] {
    let length = dot(value, value).sqrt();
    value.map(|component| component / length)
}

#[derive(Clone, Copy)]
struct View {
    eye: Vec3,
    target: Vec3,
    fov: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

impl View {
    fn camera(self) -> Camera3d {
        Camera3d::look_at(
            self.eye,
            self.target,
            Vec3::Y,
            Projection3d::perspective(self.fov, self.aspect, world(self.near), world(self.far))
                .unwrap(),
        )
        .unwrap()
    }

    fn rays(self, width: u32, height: u32) -> Rays {
        let forward = normalized(subtract(array(self.target), array(self.eye)));
        let right = normalized(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        Rays {
            origin: array(self.eye),
            forward,
            right,
            up,
            tangent: (f64::from(self.fov) * 0.5).tan(),
            aspect: f64::from(self.aspect),
            width,
            height,
            near: f64::from(self.near),
            far: f64::from(self.far),
        }
    }
}

struct Rays {
    origin: [f64; 3],
    forward: [f64; 3],
    right: [f64; 3],
    up: [f64; 3],
    tangent: f64,
    aspect: f64,
    width: u32,
    height: u32,
    near: f64,
    far: f64,
}

impl Rays {
    fn direction(&self, x: u32, y: u32) -> [f64; 3] {
        let horizontal =
            (2.0 * (f64::from(x) + 0.5) / f64::from(self.width) - 1.0) * self.tangent * self.aspect;
        let vertical = (1.0 - 2.0 * (f64::from(y) + 0.5) / f64::from(self.height)) * self.tangent;
        // This deliberately is not a unit vector: ray parameter is forward
        // view distance, so clipping uses near/far without projection Z math.
        std::array::from_fn(|axis| {
            self.forward[axis] + horizontal * self.right[axis] + vertical * self.up[axis]
        })
    }
}

struct Triangle {
    origin: [f64; 3],
    first: [f64; 3],
    second: [f64; 3],
    color: usize,
}

fn triangles(mesh: &Mesh3d, color: usize) -> Vec<Triangle> {
    mesh.triangle_indices()
        .chunks_exact(3)
        .map(|indices| {
            let points = std::array::from_fn::<_, 3, _>(|index| {
                array(mesh.vertices()[indices[index] as usize])
            });
            Triangle {
                origin: points[0],
                first: subtract(points[1], points[0]),
                second: subtract(points[2], points[0]),
                color,
            }
        })
        .collect()
}

#[derive(Default)]
struct Samples {
    colors: [usize; 6],
    background: usize,
    boundary: usize,
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
    fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let identity = Arc::new(());
        Self {
            device,
            queue,
            target: surface::test_target(device, &identity, format, width, height),
            renderer: Mesh3dRenderer::new(device, format),
            identity,
            format,
        }
    }

    fn upload(&self, mesh: Mesh3d) -> RetainedMesh3d {
        create_retained_mesh(self.device, self.queue, self.identity.clone(), mesh).unwrap()
    }

    fn pixels(&self) -> Vec<u8> {
        surface::test_read_pixels(self.device, self.queue, &self.target)
    }

    fn render(&mut self, scene: &Scene3d, view: View) -> Vec<u8> {
        let count = scene
            .instances()
            .iter()
            .map(|instance| instance.mesh.source().triangle_count())
            .sum();
        let report = self
            .renderer
            .render_scene3d(
                self.device,
                self.queue,
                &self.identity,
                &self.target,
                scene,
                view.camera(),
                native_budget(count),
            )
            .unwrap();
        assert_eq!(report.triangle_count(), count);
        let preflight = report.preflight();
        assert_eq!(preflight.submitted_triangle_count(), count);
        assert_eq!(preflight.surface_policy(), SurfaceRasterization3d::Native);
        assert_eq!(preflight.generated_triangle_count(), 0);
        assert_eq!(preflight.generated_object_count(), 0);
        assert_eq!(preflight.generated_upload_bytes(), 0);
        assert_eq!(preflight.clipped_source_triangle_count(), None);
        assert_eq!(preflight.discarded_source_triangle_count(), None);
        self.pixels()
    }

    fn expect(&self, pixels: &[u8], x: u32, y: u32, color: Option<usize>) {
        let channels = match self.format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0, 3],
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2, 3],
            other => panic!("unsupported Native oracle format {other:?}"),
        };
        let value = color.map_or(Color::BLACK, |index| COLORS[index]).to_array();
        let offset = ((y * self.target.width() + x) * 4) as usize;
        for channel in 0..4 {
            let linear = f64::from(value[channel]);
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
                actual.abs_diff(expected) <= 2,
                "Native ray oracle ({x},{y}) channel {channel}: {actual} != {expected}, visual={color:?}"
            );
        }
    }

    fn compare(&self, pixels: &[u8], view: View, triangles: &[Triangle], step: usize) -> Samples {
        let rays = view.rays(self.target.width(), self.target.height());
        let mut samples = Samples::default();
        for y in (1..rays.height - 1).step_by(step) {
            for x in (1..rays.width - 1).step_by(step) {
                let direction = rays.direction(x, y);
                let mut nearest = f64::INFINITY;
                let mut color = None;
                let mut uncertain = false;
                for triangle in triangles {
                    let perpendicular = cross(direction, triangle.second);
                    let determinant = dot(triangle.first, perpendicular);
                    if determinant.abs() < 1e-12 {
                        continue;
                    }
                    let relative = subtract(rays.origin, triangle.origin);
                    let first_amount = dot(relative, perpendicular) / determinant;
                    let second_cross = cross(relative, triangle.first);
                    let second_amount = dot(direction, second_cross) / determinant;
                    let remaining = 1.0 - first_amount - second_amount;
                    let depth = dot(triangle.second, second_cross) / determinant;
                    let weights = [first_amount, second_amount, remaining];
                    let depth_guard = 0.0001 * depth.abs().max(1.0);
                    if depth < rays.near - depth_guard || depth > rays.far + depth_guard {
                        continue;
                    }
                    // Deliberately exclude a conservative source-edge band and
                    // near/far band. Native does not promise boundary identity
                    // between drivers. Exclusions and real color hits are counted.
                    if weights.iter().all(|weight| *weight >= -0.02)
                        && (weights.iter().any(|weight| *weight <= 0.02)
                            || (depth - rays.near).abs() <= depth_guard
                            || (depth - rays.far).abs() <= depth_guard)
                    {
                        uncertain = true;
                    }
                    if weights.iter().all(|weight| *weight >= 0.0)
                        && depth >= rays.near
                        && depth <= rays.far
                        && depth < nearest
                    {
                        nearest = depth;
                        color = Some(triangle.color);
                    }
                }
                if uncertain {
                    samples.boundary += 1;
                    continue;
                }
                self.expect(pixels, x, y, color);
                if let Some(color) = color {
                    samples.colors[color] += 1;
                } else {
                    samples.background += 1;
                }
            }
        }
        samples
    }
}

fn native_budget(triangles: usize) -> Mesh3dRenderBudget {
    Mesh3dRenderBudget::new(0, 0, 0)
        .with_max_surface_triangles(triangles)
        .with_surface_policy(SurfaceRasterization3d::Native)
}

fn style(color: usize) -> MeshStyle3d {
    MeshStyle3d::surface(SurfaceStyle3d::opaque(COLORS[color]).unwrap())
}

fn winding(vertices: Vec<Vec3>, mut indices: Vec<u32>, reverse: bool) -> Mesh3d {
    if reverse {
        for triangle in indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
    }
    Mesh3d::with_display_edges(vertices, indices, Vec::new()).unwrap()
}

fn floor(offset: f32, reverse: bool) -> Mesh3d {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for z in 16..24 {
        for x in 8..16 {
            let first = vertices.len() as u32;
            let x = x as f32 + offset;
            let z = z as f32;
            vertices.extend([
                point([x, 0.0, z]),
                point([x + 1.0, 0.0, z]),
                point([x + 1.0, 0.0, z + 1.0]),
                point([x, 0.0, z + 1.0]),
            ]);
            indices.extend([first, first + 1, first + 2, first, first + 2, first + 3]);
        }
    }
    winding(vertices, indices, reverse)
}

fn box_mesh(minimum: [f32; 3], reverse: bool) -> Mesh3d {
    let vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.5, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.5, 1.0],
        [0.0, 1.5, 1.0],
    ]
    .map(|value| point(std::array::from_fn(|axis| value[axis] + minimum[axis])));
    winding(
        vertices.to_vec(),
        vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7,
            3, 1, 2, 6, 1, 6, 5,
        ],
        reverse,
    )
}

fn camera_visit(index: usize) -> View {
    let (eye, target) = if index < 2 {
        let eye = point([11.75, 7.58, if index == 0 { 11.0 } else { 11.001 }]);
        let yaw = 14_f32 * 0.13;
        (eye, point([eye.x() + yaw.sin(), 7.18, eye.z() - yaw.cos()]))
    } else {
        let region = (index - 2) / 47;
        let angle = ((index - 2) % 47) as f32 * std::f32::consts::TAU / 47.0;
        let center_x = if region == 0 { 12.0 } else { -4.0 };
        (
            point([
                center_x + 8.0 * angle.cos(),
                5.4 + 0.3 * angle.sin(),
                20.0 + 8.0 * angle.sin(),
            ]),
            point([center_x, 0.8, 20.0]),
        )
    };
    View {
        eye,
        target,
        fov: std::f32::consts::FRAC_PI_3,
        aspect: 1100.0 / 720.0,
        near: 0.1,
        far: 1000.0,
    }
}

pub(super) fn assert_gpu_native_acceptance(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let mut fixture = Fixture::new(device, queue, format, WIDTH, HEIGHT);
    for reverse in [false, true] {
        let sources = [
            floor(0.0, reverse),
            floor(-16.0, reverse),
            box_mesh([10.0, 0.0, 19.0], reverse),
            box_mesh([-6.0, 0.0, 19.0], reverse),
            box_mesh([13.0, 0.0, 21.0], reverse),
            box_mesh([-3.0, 0.0, 21.0], reverse),
        ];
        let meshes = sources.clone().map(|source| fixture.upload(source));
        let mut scene = Scene3d::new(Color::BLACK).unwrap();
        let mut ids = Vec::new();
        for (index, mesh) in meshes.iter().enumerate().take(4) {
            ids.push(
                scene
                    .try_push(mesh, Transform3d::IDENTITY, style(index))
                    .unwrap(),
            );
        }
        for revision in 0..3 {
            if revision == 1 {
                for (index, mesh) in meshes.iter().enumerate().skip(4) {
                    scene
                        .try_push(mesh, Transform3d::IDENTITY, style(index))
                        .unwrap();
                }
            } else if revision == 2 {
                scene.remove(ids[2]).unwrap();
                scene.remove(ids[3]).unwrap();
            }
            let active = |index: usize| match revision {
                0 => index < 4,
                1 => true,
                _ => !(2..4).contains(&index),
            };
            let reference = sources
                .iter()
                .enumerate()
                .filter(|(index, _)| active(*index))
                .flat_map(|(index, mesh)| triangles(mesh, index))
                .collect::<Vec<_>>();
            let mut hits = [0; 6];
            let mut original = Vec::new();
            for visit in 0..96 {
                let view = camera_visit(visit);
                let pixels = fixture.render(&scene, view);
                let samples = fixture.compare(&pixels, view, &reference, 3);
                assert!(samples.background + samples.colors.iter().sum::<usize>() > 100);
                if visit >= 2 {
                    assert!(samples.colors.iter().sum::<usize>() > 5);
                }
                for (sum, count) in hits.iter_mut().zip(samples.colors) {
                    *sum += count;
                }
                if visit == 0 {
                    original = pixels;
                }
            }
            for (index, count) in hits.into_iter().enumerate() {
                if active(index) {
                    assert!(
                        count > 16,
                        "visual {index}, revision {revision}, winding {reverse}"
                    );
                } else {
                    assert_eq!(count, 0);
                }
            }
            assert_eq!(
                fixture.render(&scene, camera_visit(0)),
                original,
                "returning to the original pose must not retain a previous camera's image"
            );
            assert_failure_preservation(&mut fixture, &mut scene, &original, reference.len());
        }
    }
    assert_perspective_boundaries(device, queue, format);
}

fn assert_failure_preservation(
    fixture: &mut Fixture<'_>,
    scene: &mut Scene3d,
    original: &[u8],
    count: usize,
) {
    let camera = camera_visit(0).camera();
    let error = fixture
        .renderer
        .render_scene3d(
            fixture.device,
            fixture.queue,
            &fixture.identity,
            &fixture.target,
            scene,
            camera,
            native_budget(count - 1),
        )
        .unwrap_err();
    assert_eq!(
        error,
        Mesh3dRenderError::SurfaceTriangleBudgetExceeded {
            limit: count - 1,
            actual: count
        }
    );
    assert_eq!(fixture.pixels(), original);
    let source = Mesh3d::new(
        vec![
            point([0.0, 0.0, -2.0]),
            point([3.0, 0.0, -2.0]),
            point([0.0, 1.0, -2.0]),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let mesh = fixture.upload(source);
    let id = scene
        .try_push(
            &mesh,
            Transform3d::new(
                Vec3::ZERO,
                Rotation3d::IDENTITY,
                point([2.0_f32.powi(119), 1.0, 1.0]),
            )
            .unwrap(),
            style(2),
        )
        .unwrap();
    let error = fixture
        .renderer
        .render_scene3d(
            fixture.device,
            fixture.queue,
            &fixture.identity,
            &fixture.target,
            scene,
            camera,
            native_budget(count + 1),
        )
        .unwrap_err();
    assert_eq!(error.object_id(), Some(id));
    assert_eq!(error.source_triangle_index(), Some(0));
    assert_eq!(
        error.surface_reason(),
        Some(Mesh3dSurfaceError::TransformArithmetic)
    );
    assert_eq!(fixture.pixels(), original);
    scene.remove(id).unwrap();
    assert_eq!(fixture.render(scene, camera_visit(0)), original);
}

fn assert_perspective_boundaries(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    let mut fixture = Fixture::new(device, queue, format, 64, 64);
    let view = View {
        eye: Vec3::ZERO,
        target: point([0.0, 0.0, -1.0]),
        fov: std::f32::consts::FRAC_PI_2,
        aspect: 1.0,
        near: 1.0,
        far: 4.0,
    };
    for reverse in [false, true] {
        for plane in 0..6 {
            for crossing in [false, true] {
                let vertices = if plane < 4 {
                    let extent = if crossing { 2.6 } else { 2.0 };
                    [[-extent, -0.8, -2.0], [0.8, -0.8, -2.0], [0.8, 0.8, -2.0]].map(|[x, y, z]| {
                        match plane {
                            0 => [x, y, z],
                            1 => [-x, y, z],
                            2 => [y, x, z],
                            _ => [y, -x, z],
                        }
                    })
                } else {
                    let depth = match (plane, crossing) {
                        (4, false) => 1.0,
                        (4, true) => 0.75,
                        (_, false) => 4.0,
                        (_, true) => 4.5,
                    };
                    [[-0.6, -0.6, -depth], [0.8, -0.6, -2.0], [0.0, 0.8, -2.0]]
                };
                let source = winding(vertices.map(point).to_vec(), vec![0, 1, 2], reverse);
                let reference = triangles(&source, 4);
                let mesh = fixture.upload(source);
                let mut scene = Scene3d::new(Color::BLACK).unwrap();
                scene
                    .try_push(&mesh, Transform3d::IDENTITY, style(4))
                    .unwrap();
                let pixels = fixture.render(&scene, view);
                let samples = fixture.compare(&pixels, view, &reference, 1);
                assert!(samples.colors[4] > 32, "plane {plane}, crossing {crossing}");
                assert!(samples.background > 32);
            }
        }
        for sliver in [false, true] {
            let vertices = if sliver {
                [
                    [1.0 / 64.0, -1.0, -2.0],
                    [3.0 / 64.0, -1.0, -2.0],
                    [1.0 / 32.0, 1.0, -2.0],
                ]
            } else {
                [[0.0, -1.0, -2.0], [0.0, 1.0, -2.0], [0.0, 0.0, -3.0]]
            };
            let source = winding(vertices.map(point).to_vec(), vec![0, 1, 2], reverse);
            let reference = triangles(&source, 4);
            let mesh = fixture.upload(source);
            let mut scene = Scene3d::new(Color::BLACK).unwrap();
            let id = scene
                .try_push(&mesh, Transform3d::IDENTITY, style(4))
                .unwrap();
            let pixels = fixture.render(&scene, view);
            let samples = fixture.compare(&pixels, view, &reference, 1);
            if sliver {
                assert!(
                    samples.colors[4] >= 16,
                    "perspective nonzero sliver must reach samples"
                );
            } else {
                assert_eq!(samples.colors[4], 0);
                assert!(pixels.chunks_exact(4).all(|pixel| pixel[..3] == [0, 0, 0]));
                let error = fixture
                    .renderer
                    .render_scene3d(
                        device,
                        queue,
                        &fixture.identity,
                        &fixture.target,
                        &scene,
                        view.camera(),
                        Mesh3dRenderBudget::default(),
                    )
                    .unwrap_err();
                assert_eq!(error.object_id(), Some(id));
                assert_eq!(error.source_triangle_index(), Some(0));
                assert_eq!(
                    error.surface_reason(),
                    Some(Mesh3dSurfaceError::ProjectedOrientation)
                );
                assert_eq!(fixture.pixels(), pixels);
            }
        }
    }
}

#[test]
fn native_acceptance_keeps_literal_floor_indices_and_camera_inputs() {
    let mesh = floor(0.0, false);
    for (index, expected) in [
        (
            63,
            [[15.0, 0.0, 19.0], [16.0, 0.0, 20.0], [15.0, 0.0, 20.0]],
        ),
        (
            78,
            [[15.0, 0.0, 20.0], [16.0, 0.0, 20.0], [16.0, 0.0, 21.0]],
        ),
    ] {
        let indices = &mesh.triangle_indices()[index * 3..index * 3 + 3];
        for (&index, expected) in indices.iter().zip(expected) {
            assert_eq!(mesh.vertices()[index as usize], point(expected));
        }
    }
    assert_eq!(camera_visit(0).eye, point([11.75, 7.58, 11.0]));
    assert_eq!(camera_visit(1).eye, point([11.75, 7.58, 11.001]));
    let viewport = LogicalViewport::new(WIDTH as f32, HEIGHT as f32).unwrap();
    for index in 0..96 {
        let view = camera_visit(index);
        let rays = view.rays(WIDTH, HEIGHT);
        let ray = rays.direction(41, 29);
        let position = point(std::array::from_fn(|axis| {
            (rays.origin[axis] + ray[axis] * 3.0) as f32
        }));
        let projected = view.camera().project_world(position, viewport).unwrap();
        let screen = projected.logical_position().to_vec2();
        assert!((screen.x() - 41.5).abs() < 0.0002);
        assert!((screen.y() - 29.5).abs() < 0.0002);
        assert!((projected.view_depth() - 3.0).abs() < 0.00001);
    }
}
