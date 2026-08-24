//! Retained depth-tested 3D foundation fixture for Sim;Math stereometry.
//!
//! The cube and octahedron keep immutable topology while their model transforms
//! update independently. This slice exercises opaque surfaces, real depth, and
//! solid/dashed mathematical edges; section materials remain a later pass.

use std::{error::Error, sync::Arc, time::Instant};

use sim_engine::{
    BlendMode, Camera3d, Color, LogicalViewport, Mesh3d, MeshEdge3d, MeshStyle3d, Object3dId,
    Projection3d, RenderTarget3d, RendererPresentMode, Rotation3d, Scene3d, SurfaceStyle3d,
    Transform3d, Vec3, WgpuRenderer, WgpuRendererOptions, WireframeStyle3d,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    EventLoop::new()?.run_app(&mut StereometryApplication::new())?;
    Ok(())
}

struct StereometryApplication {
    window: Option<Arc<Window>>,
    renderer: Option<WgpuRenderer>,
    target: Option<RenderTarget3d>,
    scene: Option<Scene3d>,
    cube_id: Option<Object3dId>,
    octahedron_id: Option<Object3dId>,
    last_update: Instant,
    playing: bool,
    cube_rotates: bool,
    octahedron_rotates: bool,
    cube_phase: f32,
    octahedron_phase: f32,
    camera_orbit: f32,
    camera_distance: f32,
    metrics: ExampleMetrics,
    uncapped: bool,
}

impl StereometryApplication {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            window: None,
            renderer: None,
            target: None,
            scene: None,
            cube_id: None,
            octahedron_id: None,
            last_update: now,
            playing: true,
            cube_rotates: true,
            octahedron_rotates: true,
            cube_phase: 0.0,
            octahedron_phase: 0.0,
            camera_orbit: 0.0,
            camera_distance: 8.0,
            metrics: ExampleMetrics::new(now),
            uncapped: std::env::args().any(|argument| argument == "--uncapped"),
        }
    }

    fn toggle_pause(&mut self, now: Instant) {
        self.playing = !self.playing;
        self.last_update = now;
    }

    fn reset_animation(&mut self, now: Instant) {
        self.last_update = now;
        self.playing = true;
        self.cube_rotates = true;
        self.octahedron_rotates = true;
        self.cube_phase = 0.0;
        self.octahedron_phase = 0.0;
        self.camera_orbit = 0.0;
        self.camera_distance = 8.0;
    }

    fn recreate_target(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        let scale = self
            .window
            .as_ref()
            .map_or(1.0, |window| window.scale_factor()) as f32;
        let viewport = LogicalViewport::new(width as f32 / scale, height as f32 / scale)
            .expect("window logical viewport is valid");
        self.target = Some(
            renderer
                .create_render_target3d(width, height, viewport)
                .expect("window dimensions fit a 3D target"),
        );
    }

    fn camera(&self, width: u32, height: u32) -> Camera3d {
        let aspect_ratio = width as f32 / height.max(1) as f32;
        let projection = Projection3d::perspective(50.0_f32.to_radians(), aspect_ratio, 0.1, 100.0)
            .expect("fixture projection is valid");
        let sine = self.camera_orbit.sin();
        let cosine = self.camera_orbit.cos();
        Camera3d::look_at(
            vector(
                sine * self.camera_distance,
                self.camera_distance * 0.42,
                cosine * self.camera_distance,
            ),
            Vec3::ZERO,
            Vec3::Y,
            projection,
        )
        .expect("fixture camera basis is valid")
    }

    fn update_transforms(&mut self, now: Instant) {
        let delta_seconds = if self.playing {
            now.saturating_duration_since(self.last_update)
                .as_secs_f32()
        } else {
            0.0
        };
        self.last_update = now;
        if self.cube_rotates {
            self.cube_phase += delta_seconds;
        }
        if self.octahedron_rotates {
            self.octahedron_phase += delta_seconds;
        }
        let Some(scene) = self.scene.as_mut() else {
            return;
        };
        let cube_rotation = Rotation3d::from_euler_xyz(
            self.cube_phase * 0.31,
            self.cube_phase * 0.47,
            self.cube_phase * 0.13,
        )
        .expect("cube angles stay finite");
        let octahedron_rotation = Rotation3d::from_euler_xyz(
            -self.octahedron_phase * 0.54,
            self.octahedron_phase * 0.29,
            self.octahedron_phase * 0.41,
        )
        .expect("octahedron angles stay finite");
        let Some(cube_id) = self.cube_id else {
            return;
        };
        let Some(octahedron_id) = self.octahedron_id else {
            return;
        };
        scene
            .set_transform(
                cube_id,
                Transform3d::new(
                    vector(-1.55, 0.0, 0.0),
                    cube_rotation,
                    vector(1.15, 1.15, 1.15),
                )
                .expect("cube transform is valid"),
            )
            .expect("cube object exists");
        scene
            .set_transform(
                octahedron_id,
                Transform3d::new(
                    vector(1.55, 0.0, 0.0),
                    octahedron_rotation,
                    vector(1.35, 1.35, 1.35),
                )
                .expect("octahedron transform is valid"),
            )
            .expect("octahedron object exists");
    }
}

impl ApplicationHandler for StereometryApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Sim;Engine retained 3D stereometry")
                        .with_inner_size(LogicalSize::new(1100.0, 720.0)),
                )
                .expect("create stereometry window"),
        );
        let size = window.inner_size();
        let present_mode = if self.uncapped {
            RendererPresentMode::NoVsync
        } else {
            RendererPresentMode::Vsync
        };
        let options = WgpuRendererOptions::new(present_mode, window.scale_factor())
            .expect("window scale factor is valid");
        let renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            options,
        ))
        .expect("create stereometry renderer");
        let cube = renderer
            .create_mesh3d(cube_mesh())
            .expect("cube topology fits the device");
        let octahedron = renderer
            .create_mesh3d(octahedron_mesh())
            .expect("octahedron topology fits the device");
        let scale = window.scale_factor() as f32;
        let viewport = LogicalViewport::new(
            size.width.max(1) as f32 / scale,
            size.height.max(1) as f32 / scale,
        )
        .expect("window logical viewport is valid");
        let target = renderer
            .create_render_target3d(size.width.max(1), size.height.max(1), viewport)
            .expect("create stereometry color/depth target");
        let mut scene =
            Scene3d::new(Color::rgb(0.008, 0.012, 0.025)).expect("fixture background is opaque");
        let cube_wireframe = WireframeStyle3d::visible(Color::rgb(0.92, 0.97, 1.0), 2.0)
            .and_then(|style| style.with_hidden(Color::rgb(0.46, 0.58, 0.72), 1.25, 7.0, 5.0))
            .expect("cube edge style is valid");
        let cube_style = MeshStyle3d::surface(
            SurfaceStyle3d::opaque(Color::rgb(0.12, 0.46, 0.92)).expect("cube surface is opaque"),
        )
        .with_wireframe(cube_wireframe);
        let cube_id = scene
            .try_push(&cube, Transform3d::IDENTITY, cube_style)
            .expect("insert cube");
        let octahedron_wireframe = WireframeStyle3d::visible(Color::rgb(1.0, 0.94, 0.78), 2.0)
            .and_then(|style| style.with_hidden(Color::rgb(0.68, 0.39, 0.20), 1.25, 6.0, 4.0))
            .expect("octahedron edge style is valid");
        let octahedron_style = MeshStyle3d::surface(
            SurfaceStyle3d::opaque(Color::rgb(0.95, 0.42, 0.10))
                .expect("octahedron surface is opaque"),
        )
        .with_wireframe(octahedron_wireframe);
        let octahedron_id = scene
            .try_push(&octahedron, Transform3d::IDENTITY, octahedron_style)
            .expect("insert octahedron");
        println!(
            "stereometry fixture: {} triangles, {} GPU bytes; Space pause, 1/2 object rotation, A/D orbit, W/S zoom, R reset, Esc exit",
            cube.triangle_count() + octahedron.triangle_count(),
            cube.gpu_allocation_bytes() + octahedron.gpu_allocation_bytes()
        );
        self.window = Some(window.clone());
        self.renderer = Some(renderer);
        self.target = Some(target);
        self.scene = Some(scene);
        self.cube_id = Some(cube_id);
        self.octahedron_id = Some(octahedron_id);
        self.last_update = Instant::now();
        self.metrics = ExampleMetrics::new(self.last_update);
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.clone() else {
            return;
        };
        if window.id() != window_id {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    PhysicalKey::Code(KeyCode::Space) => self.toggle_pause(Instant::now()),
                    PhysicalKey::Code(KeyCode::Digit1) => self.cube_rotates = !self.cube_rotates,
                    PhysicalKey::Code(KeyCode::Digit2) => {
                        self.octahedron_rotates = !self.octahedron_rotates
                    }
                    PhysicalKey::Code(KeyCode::KeyA) => self.camera_orbit -= 0.12,
                    PhysicalKey::Code(KeyCode::KeyD) => self.camera_orbit += 0.12,
                    PhysicalKey::Code(KeyCode::KeyW) => {
                        self.camera_distance = (self.camera_distance - 0.4).max(4.5)
                    }
                    PhysicalKey::Code(KeyCode::KeyS) => {
                        self.camera_distance = (self.camera_distance + 0.4).min(18.0)
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => self.reset_animation(Instant::now()),
                    _ => {}
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
                self.recreate_target(size.width, size.height);
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let size = window.inner_size();
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer
                        .resize_with_scale_factor(size.width, size.height, scale_factor)
                        .expect("window scale factor stays valid");
                }
                self.recreate_target(size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                let frame_started = Instant::now();
                self.update_transforms(frame_started);
                let Some(target_size) = self
                    .target
                    .as_ref()
                    .map(|target| (target.width(), target.height()))
                else {
                    return;
                };
                let camera = self.camera(target_size.0, target_size.1);
                let (Some(renderer), Some(target), Some(scene)) = (
                    self.renderer.as_mut(),
                    self.target.as_ref(),
                    self.scene.as_ref(),
                ) else {
                    return;
                };
                let report = renderer
                    .render_scene3d_to_target(target, scene, camera)
                    .expect("render retained depth-tested scene");
                if !self.uncapped {
                    window.pre_present_notify();
                }
                renderer
                    .compose_render_target(
                        target.color_target(),
                        BlendMode::Replace,
                        1.0,
                        Color::BLACK,
                    )
                    .expect("compose stereometry target");
                if let Some(title) = self.metrics.record(frame_started, report) {
                    window.set_title(&title);
                }
                window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Poll);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
            event_loop.set_control_flow(ControlFlow::Poll);
        }
    }
}

struct ExampleMetrics {
    interval_started: Instant,
    frames: usize,
    renderer_cpu: std::time::Duration,
}

impl ExampleMetrics {
    fn new(now: Instant) -> Self {
        Self {
            interval_started: now,
            frames: 0,
            renderer_cpu: std::time::Duration::ZERO,
        }
    }

    fn record(&mut self, now: Instant, report: sim_engine::Mesh3dRenderReport) -> Option<String> {
        self.frames += 1;
        self.renderer_cpu += report.upload() + report.encode_submit();
        let elapsed = now.saturating_duration_since(self.interval_started);
        if elapsed.as_secs_f32() < 1.0 {
            return None;
        }
        let frames_per_second = self.frames as f64 / elapsed.as_secs_f64();
        let renderer_ms = self.renderer_cpu.as_secs_f64() * 1000.0 / self.frames as f64;
        let title = format!(
            "Sim;Engine 3D | {frames_per_second:.0} FPS | renderer CPU {renderer_ms:.3} ms | {} objects / {} triangles / {} edges",
            report.object_count(),
            report.triangle_count(),
            report.edge_count()
        );
        self.interval_started = now;
        self.frames = 0;
        self.renderer_cpu = std::time::Duration::ZERO;
        Some(title)
    }
}

fn cube_mesh() -> Mesh3d {
    let vertices = vec![
        vector(-1.0, -1.0, -1.0),
        vector(1.0, -1.0, -1.0),
        vector(1.0, 1.0, -1.0),
        vector(-1.0, 1.0, -1.0),
        vector(-1.0, -1.0, 1.0),
        vector(1.0, -1.0, 1.0),
        vector(1.0, 1.0, 1.0),
        vector(-1.0, 1.0, 1.0),
    ];
    let triangles = vec![
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    ];
    let edges = edge_list(&[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]);
    Mesh3d::with_display_edges(vertices, triangles, edges).expect("cube topology is valid")
}

fn octahedron_mesh() -> Mesh3d {
    let vertices = vec![
        vector(0.0, 1.0, 0.0),
        vector(0.0, -1.0, 0.0),
        vector(-1.0, 0.0, 0.0),
        vector(1.0, 0.0, 0.0),
        vector(0.0, 0.0, -1.0),
        vector(0.0, 0.0, 1.0),
    ];
    let triangles = vec![
        0, 4, 3, 0, 2, 4, 0, 5, 2, 0, 3, 5, 1, 3, 4, 1, 4, 2, 1, 2, 5, 1, 5, 3,
    ];
    let edges = edge_list(&[
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (4, 3),
        (3, 5),
        (5, 2),
    ]);
    Mesh3d::with_display_edges(vertices, triangles, edges).expect("octahedron topology is valid")
}

fn edge_list(edges: &[(u32, u32)]) -> Vec<MeshEdge3d> {
    edges
        .iter()
        .map(|(start, end)| MeshEdge3d::new(*start, *end).expect("fixture edge is valid"))
        .collect()
}

fn vector(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z).expect("fixture vector is finite")
}
