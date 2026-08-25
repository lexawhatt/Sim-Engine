//! Smooth retained-3D derivation of cylinder volume and total surface area.
//!
//! The presentation uses a dark mathematical-animation visual language:
//! restrained camera motion, saturated semantic colors, staged decomposition,
//! and formulas in the host window title. Sim;Engine receives ready geometry
//! and transforms; the host owns the educational timeline.

use std::{
    error::Error,
    f32::consts::TAU,
    sync::Arc,
    time::{Duration, Instant},
};

use sim_engine::{
    BlendMode, Camera3d, Color, LogicalPixels, LogicalViewport, Mesh3d, MeshEdge3d, MeshStyle3d,
    Object3dId, Projection3d, RenderTarget3d, RendererPresentMode, Rotation3d, Scene3d,
    SurfaceStyle3d, Transform3d, Vec3, WgpuRenderer, WgpuRendererOptions, WireframeStyle3d,
    WorldLength,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const CYLINDER_SEGMENTS: usize = 36;
const VOLUME_SLICES: usize = 18;
const RADIUS: f32 = 1.35;
const HEIGHT: f32 = 2.8;
const TIMELINE_SECONDS: f32 = 20.0;

fn logical(value: f32) -> LogicalPixels {
    LogicalPixels::new(value).expect("example logical pixel value is valid")
}

fn world(value: f32) -> WorldLength {
    WorldLength::new(value).expect("example world length is valid")
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    EventLoop::new()?.run_app(&mut CylinderApplication::new())?;
    Ok(())
}

struct CylinderApplication {
    window: Option<Arc<Window>>,
    renderer: Option<WgpuRenderer>,
    target: Option<RenderTarget3d>,
    scene: Option<Scene3d>,
    full_cylinder: Option<Object3dId>,
    slices: Vec<Object3dId>,
    panels: Vec<Object3dId>,
    caps: [Option<Object3dId>; 2],
    timeline: f32,
    playing: bool,
    last_update: Instant,
    metrics: PresentationMetrics,
    uncapped: bool,
    benchmark: bool,
}

impl CylinderApplication {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            window: None,
            renderer: None,
            target: None,
            scene: None,
            full_cylinder: None,
            slices: Vec::new(),
            panels: Vec::new(),
            caps: [None, None],
            timeline: 0.0,
            playing: true,
            last_update: now,
            metrics: PresentationMetrics::new(now),
            uncapped: std::env::args().any(|argument| argument == "--uncapped"),
            benchmark: std::env::args().any(|argument| argument == "--benchmark"),
        }
    }

    fn recreate_target(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let (Some(renderer), Some(window)) = (self.renderer.as_ref(), self.window.as_ref()) else {
            return;
        };
        let scale = window.scale_factor() as f32;
        let viewport = LogicalViewport::new(width as f32 / scale, height as f32 / scale)
            .expect("window logical viewport is valid");
        self.target = Some(
            renderer
                .create_render_target3d(width, height, viewport)
                .expect("resize cylinder target"),
        );
    }

    fn camera(&self, viewport: LogicalViewport) -> Camera3d {
        let area_amount = area_camera_amount(self.timeline);
        let start = vector(5.2, 3.4, 7.4);
        let end = vector(0.0, 0.15, 13.0);
        let position = lerp_vec3(start, end, area_amount);
        let projection = Projection3d::perspective(
            46.0_f32.to_radians(),
            viewport.width() / viewport.height(),
            world(0.1),
            world(100.0),
        )
        .expect("presentation projection is valid");
        Camera3d::look_at(position, Vec3::ZERO, Vec3::Y, projection)
            .expect("presentation camera is valid")
    }

    fn advance_timeline(&mut self, now: Instant) {
        let delta = now
            .saturating_duration_since(self.last_update)
            .as_secs_f32()
            .min(0.1);
        self.last_update = now;
        if self.playing {
            self.timeline = (self.timeline + delta) % TIMELINE_SECONDS;
        }
    }

    fn update_visual_state(&mut self) {
        let Some(scene) = self.scene.as_mut() else {
            return;
        };
        let t = self.timeline;
        let show_full = t < 3.0;
        let show_slices = (3.0..8.0).contains(&t);
        let show_area = t >= 8.0;

        if let Some(id) = self.full_cylinder {
            scene
                .set_visible(id, show_full)
                .expect("full cylinder exists");
            if show_full {
                let rotation = Rotation3d::from_euler_xyz(0.12, t * 0.34, -0.04)
                    .expect("intro rotation is finite");
                scene
                    .set_transform(
                        id,
                        Transform3d::new(Vec3::ZERO, rotation, vector(1.0, 1.0, 1.0))
                            .expect("intro transform is valid"),
                    )
                    .expect("full cylinder exists");
            }
        }

        let explode = if t < 5.0 {
            smootherstep((t - 3.0) / 2.0)
        } else {
            1.0 - smootherstep((t - 5.0) / 2.5)
        };
        let slice_height = HEIGHT / VOLUME_SLICES as f32;
        for (index, id) in self.slices.iter().copied().enumerate() {
            scene.set_visible(id, show_slices).expect("slice exists");
            if show_slices {
                let centered = index as f32 - (VOLUME_SLICES as f32 - 1.0) * 0.5;
                let base_y = centered * slice_height;
                let stagger = centered * 0.075 * explode;
                let wave = ((index as f32 * 0.58) + t * 1.7).sin() * 0.025 * explode;
                scene
                    .set_transform(
                        id,
                        Transform3d::new(
                            vector(wave, base_y + stagger, 0.0),
                            Rotation3d::IDENTITY,
                            vector(RADIUS, slice_height, RADIUS),
                        )
                        .expect("slice transform is valid"),
                    )
                    .expect("slice exists");
            }
        }

        let unfold = area_unfold_amount(t);
        let arc_width = TAU * RADIUS / CYLINDER_SEGMENTS as f32;
        let rectangle_width = TAU * RADIUS;
        for (index, id) in self.panels.iter().copied().enumerate() {
            scene.set_visible(id, show_area).expect("panel exists");
            if show_area {
                let angle = TAU * index as f32 / CYLINDER_SEGMENTS as f32;
                let cylinder_position = vector(RADIUS * angle.sin(), 0.0, RADIUS * angle.cos());
                let flat_x = (index as f32 - (CYLINDER_SEGMENTS as f32 - 1.0) * 0.5) * arc_width;
                let flat_position = vector(flat_x, 0.0, 0.0);
                let position = lerp_vec3(cylinder_position, flat_position, unfold);
                let rotation = Rotation3d::from_euler_xyz(0.0, angle * (1.0 - unfold), 0.0)
                    .expect("panel rotation is finite");
                scene
                    .set_transform(
                        id,
                        Transform3d::new(position, rotation, vector(1.0, 1.0, 1.0))
                            .expect("panel transform is valid"),
                    )
                    .expect("panel exists");
            }
        }

        for (cap_index, id) in self.caps.iter().flatten().copied().enumerate() {
            scene.set_visible(id, show_area).expect("cap exists");
            if show_area {
                let sign = if cap_index == 0 { -1.0 } else { 1.0 };
                let cylinder_position = vector(0.0, sign * HEIGHT * 0.5, 0.0);
                let flat_position =
                    vector(sign * (rectangle_width * 0.5 + RADIUS + 0.55), 0.0, 0.0);
                let position = lerp_vec3(cylinder_position, flat_position, unfold);
                let rotation = Rotation3d::from_euler_xyz(
                    std::f32::consts::FRAC_PI_2 * (1.0 - unfold),
                    0.0,
                    0.0,
                )
                .expect("cap rotation is finite");
                scene
                    .set_transform(
                        id,
                        Transform3d::new(position, rotation, vector(RADIUS, RADIUS, RADIUS))
                            .expect("cap transform is valid"),
                    )
                    .expect("cap exists");
            }
        }
    }

    fn scrub(&mut self, seconds: f32) {
        self.timeline = (self.timeline + seconds).rem_euclid(TIMELINE_SECONDS);
        self.last_update = Instant::now();
    }

    fn render_offscreen_frame(&mut self) -> sim_engine::Mesh3dRenderReport {
        self.update_visual_state();
        let viewport = self
            .target
            .as_ref()
            .expect("benchmark target exists")
            .logical_viewport();
        let camera = self.camera(viewport);
        self.renderer
            .as_mut()
            .expect("benchmark renderer exists")
            .render_scene3d_to_target(
                self.target.as_ref().expect("benchmark target exists"),
                self.scene.as_ref().expect("benchmark scene exists"),
                camera,
            )
            .expect("render offscreen benchmark frame")
    }

    fn run_offscreen_benchmark(&mut self) {
        const WARMUP_FRAMES: usize = 30;
        const MEASURED_FRAMES: usize = 360;

        let saved_timeline = self.timeline;
        let saved_playing = self.playing;
        for frame in 0..WARMUP_FRAMES {
            self.timeline = TIMELINE_SECONDS * frame as f32 / WARMUP_FRAMES as f32;
            let _ = self.render_offscreen_frame();
        }
        self.renderer
            .as_ref()
            .expect("benchmark renderer exists")
            .wait_for_gpu_idle()
            .expect("complete benchmark warmup");

        let started = Instant::now();
        let mut renderer_cpu = Duration::ZERO;
        for frame in 0..MEASURED_FRAMES {
            self.timeline = TIMELINE_SECONDS * frame as f32 / MEASURED_FRAMES as f32;
            let report = self.render_offscreen_frame();
            renderer_cpu += report.upload() + report.encode_submit();
        }
        self.renderer
            .as_ref()
            .expect("benchmark renderer exists")
            .wait_for_gpu_idle()
            .expect("complete measured GPU work");
        let elapsed = started.elapsed();
        let throughput = MEASURED_FRAMES as f64 / elapsed.as_secs_f64();
        let average_cpu_ms = renderer_cpu.as_secs_f64() * 1000.0 / MEASURED_FRAMES as f64;
        println!(
            "offscreen completed throughput: {throughput:.0} frames/s | {MEASURED_FRAMES} frames in {:.3} s | 3D submit CPU {average_cpu_ms:.3} ms | excludes surface/compositor",
            elapsed.as_secs_f64(),
        );

        self.timeline = saved_timeline;
        self.playing = saved_playing;
        self.last_update = Instant::now();
        self.update_visual_state();
    }
}

impl ApplicationHandler for CylinderApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Cylinder derivation | loading")
                        .with_inner_size(LogicalSize::new(1200.0, 760.0)),
                )
                .expect("create cylinder window"),
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
        .expect("create cylinder renderer");
        let surface_present_mode = renderer.surface_present_mode();

        let cylinder = renderer
            .create_mesh3d(cylinder_mesh(CYLINDER_SEGMENTS))
            .expect("cylinder topology fits device");
        let unit_slice = renderer
            .create_mesh3d(unit_cylinder_mesh(CYLINDER_SEGMENTS))
            .expect("slice topology fits device");
        let arc_width = TAU * RADIUS / CYLINDER_SEGMENTS as f32;
        let panel = renderer
            .create_mesh3d(panel_mesh(arc_width, HEIGHT))
            .expect("panel topology fits device");
        let disk = renderer
            .create_mesh3d(disk_mesh(CYLINDER_SEGMENTS))
            .expect("disk topology fits device");

        let scale = window.scale_factor() as f32;
        let viewport = LogicalViewport::new(
            size.width.max(1) as f32 / scale,
            size.height.max(1) as f32 / scale,
        )
        .expect("window logical viewport is valid");
        let target = renderer
            .create_render_target3d(size.width.max(1), size.height.max(1), viewport)
            .expect("create cylinder target");
        let mut scene = Scene3d::new(Color::rgb8(9, 12, 20)).expect("background is opaque");

        let outline = WireframeStyle3d::visible(Color::rgb8(224, 238, 255), logical(1.7))
            .expect("outline style is valid")
            .with_hidden(
                Color::rgb8(83, 104, 128),
                logical(1.0),
                logical(6.0),
                logical(5.0),
            )
            .expect("hidden outline style is valid");
        let cylinder_style = MeshStyle3d::surface(
            SurfaceStyle3d::opaque(Color::rgb8(32, 137, 166)).expect("surface is opaque"),
        )
        .with_wireframe(outline);
        let full_cylinder = scene
            .try_push(&cylinder, Transform3d::IDENTITY, cylinder_style)
            .expect("insert full cylinder");

        let mut slices = Vec::with_capacity(VOLUME_SLICES);
        for index in 0..VOLUME_SLICES {
            let color = if index.is_multiple_of(2) {
                Color::rgb8(42, 177, 184)
            } else {
                Color::rgb8(35, 132, 178)
            };
            let style = MeshStyle3d::surface(
                SurfaceStyle3d::opaque(color).expect("slice surface is opaque"),
            );
            let id = scene
                .try_push(&unit_slice, Transform3d::IDENTITY, style)
                .expect("insert volume slice");
            scene.set_visible(id, false).expect("slice exists");
            slices.push(id);
        }

        let panel_outline = WireframeStyle3d::visible(Color::rgb8(105, 230, 218), logical(1.1))
            .expect("panel outline is valid");
        let panel_style = MeshStyle3d::surface(
            SurfaceStyle3d::opaque(Color::rgb8(25, 114, 150)).expect("panel is opaque"),
        )
        .with_wireframe(panel_outline);
        let mut panels = Vec::with_capacity(CYLINDER_SEGMENTS);
        for _ in 0..CYLINDER_SEGMENTS {
            let id = scene
                .try_push(&panel, Transform3d::IDENTITY, panel_style)
                .expect("insert side panel");
            scene.set_visible(id, false).expect("panel exists");
            panels.push(id);
        }

        let cap_outline = WireframeStyle3d::visible(Color::rgb8(255, 226, 112), logical(2.0))
            .expect("cap outline is valid");
        let cap_style = MeshStyle3d::surface(
            SurfaceStyle3d::opaque(Color::rgb8(232, 174, 55)).expect("cap is opaque"),
        )
        .with_wireframe(cap_outline);
        let mut caps = [None, None];
        for cap in &mut caps {
            let id = scene
                .try_push(&disk, Transform3d::IDENTITY, cap_style)
                .expect("insert cap");
            scene.set_visible(id, false).expect("cap exists");
            *cap = Some(id);
        }

        let retained_bytes = cylinder.gpu_allocation_bytes()
            + unit_slice.gpu_allocation_bytes()
            + panel.gpu_allocation_bytes()
            + disk.gpu_allocation_bytes();
        println!(
            "cylinder derivation: {retained_bytes} retained GPU bytes; Space pause, Left/Right scrub, 1 volume, 2 area, R reset, Esc exit"
        );
        println!(
            "presentation: requested {present_mode:?}, configured {surface_present_mode}, refresh-synchronized: {}",
            surface_present_mode.is_refresh_synchronized()
        );
        self.window = Some(window.clone());
        self.renderer = Some(renderer);
        self.target = Some(target);
        self.scene = Some(scene);
        self.full_cylinder = Some(full_cylinder);
        self.slices = slices;
        self.panels = panels;
        self.caps = caps;
        self.last_update = Instant::now();
        self.metrics = PresentationMetrics::new(self.last_update);
        self.update_visual_state();
        if self.benchmark {
            self.run_offscreen_benchmark();
        }
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
                    PhysicalKey::Code(KeyCode::Space) => {
                        self.playing = !self.playing;
                        self.last_update = Instant::now();
                    }
                    PhysicalKey::Code(KeyCode::ArrowLeft) => self.scrub(-1.0),
                    PhysicalKey::Code(KeyCode::ArrowRight) => self.scrub(1.0),
                    PhysicalKey::Code(KeyCode::Digit1) => {
                        self.timeline = 3.2;
                        self.last_update = Instant::now();
                    }
                    PhysicalKey::Code(KeyCode::Digit2) => {
                        self.timeline = 8.2;
                        self.last_update = Instant::now();
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        self.timeline = 0.0;
                        self.playing = true;
                        self.last_update = Instant::now();
                    }
                    _ => {}
                }
                window.request_redraw();
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
                self.advance_timeline(frame_started);
                self.update_visual_state();
                let Some(viewport) = self.target.as_ref().map(RenderTarget3d::logical_viewport)
                else {
                    return;
                };
                let camera = self.camera(viewport);
                let (Some(renderer), Some(target), Some(scene)) = (
                    self.renderer.as_mut(),
                    self.target.as_ref(),
                    self.scene.as_ref(),
                ) else {
                    return;
                };
                let report = renderer
                    .render_scene3d_to_target(target, scene, camera)
                    .expect("render cylinder derivation");
                if !self.uncapped {
                    window.pre_present_notify();
                }
                let composition_report = renderer
                    .compose_render_target(
                        target.color_target(),
                        BlendMode::Replace,
                        1.0,
                        Color::BLACK,
                    )
                    .expect("compose cylinder derivation");
                let frame_completed = Instant::now();
                let surface_present_mode = renderer.surface_present_mode();
                if let Some((title, diagnostics)) = self.metrics.record(
                    frame_started,
                    frame_completed,
                    report,
                    composition_report,
                    surface_present_mode,
                    self.timeline,
                ) {
                    if self.benchmark {
                        println!("{diagnostics}");
                    }
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

struct PresentationMetrics {
    interval_started: Instant,
    frames: usize,
    renderer_cpu: std::time::Duration,
    callback_cpu: std::time::Duration,
    surface_acquire: std::time::Duration,
    composition_cpu: std::time::Duration,
}

impl PresentationMetrics {
    fn new(now: Instant) -> Self {
        Self {
            interval_started: now,
            frames: 0,
            renderer_cpu: std::time::Duration::ZERO,
            callback_cpu: std::time::Duration::ZERO,
            surface_acquire: std::time::Duration::ZERO,
            composition_cpu: std::time::Duration::ZERO,
        }
    }

    fn record(
        &mut self,
        frame_started: Instant,
        frame_completed: Instant,
        report: sim_engine::Mesh3dRenderReport,
        composition_report: sim_engine::RenderReport,
        surface_present_mode: sim_engine::RendererSurfacePresentMode,
        timeline: f32,
    ) -> Option<(String, String)> {
        self.frames += 1;
        self.renderer_cpu += report.upload() + report.encode_submit();
        self.callback_cpu += frame_completed.saturating_duration_since(frame_started);
        self.surface_acquire += composition_report.metrics().surface_acquire();
        self.composition_cpu += composition_report.metrics().total_cpu();
        let elapsed = frame_completed.saturating_duration_since(self.interval_started);
        if elapsed.as_secs_f32() < 1.0 {
            return None;
        }
        let presented_fps = self.frames as f64 / elapsed.as_secs_f64();
        let cpu_ms = self.renderer_cpu.as_secs_f64() * 1000.0 / self.frames as f64;
        let callback_ms = self.callback_cpu.as_secs_f64() * 1000.0 / self.frames as f64;
        let acquire_ms = self.surface_acquire.as_secs_f64() * 1000.0 / self.frames as f64;
        let composition_ms = self.composition_cpu.as_secs_f64() * 1000.0 / self.frames as f64;
        let interval_ms = elapsed.as_secs_f64() * 1000.0 / self.frames as f64;
        let host_gap_ms = (interval_ms - callback_ms).max(0.0);
        let cpu_submit_ceiling = if cpu_ms > 0.0 { 1000.0 / cpu_ms } else { 0.0 };
        let title = format!(
            "{}  |  {:.0} presented FPS  |  3D CPU {:.3} ms  |  {}  |  {} objects",
            stage_caption(timeline),
            presented_fps,
            cpu_ms,
            surface_present_mode,
            report.object_count(),
        );
        let diagnostics = format!(
            "{} | {:.0} presented FPS | callback {callback_ms:.3} ms | host/compositor gap {host_gap_ms:.3} ms | surface acquire {acquire_ms:.3} ms | compose CPU {composition_ms:.3} ms | 3D submit CPU {cpu_ms:.3} ms (~{cpu_submit_ceiling:.0}/s CPU ceiling) | {surface_present_mode} | {} objects",
            stage_caption(timeline),
            presented_fps,
            report.object_count(),
        );
        self.interval_started = frame_completed;
        self.frames = 0;
        self.renderer_cpu = std::time::Duration::ZERO;
        self.callback_cpu = std::time::Duration::ZERO;
        self.surface_acquire = std::time::Duration::ZERO;
        self.composition_cpu = std::time::Duration::ZERO;
        Some((title, diagnostics))
    }
}

fn stage_caption(timeline: f32) -> &'static str {
    match timeline {
        t if t < 3.0 => "Cylinder: radius r, height h",
        t if t < 5.2 => "Base area A = πr² — separate equal disks",
        t if t < 8.0 => "Volume V = A·h = πr²h",
        t if t < 11.5 => "Cut and unwrap the lateral surface",
        t if t < 14.0 => "Rectangle: width 2πr, height h",
        t if t < 17.0 => "Total area S = 2πrh + 2πr²",
        _ => "The pieces fold back into the cylinder",
    }
}

fn area_unfold_amount(timeline: f32) -> f32 {
    if timeline < 8.0 {
        0.0
    } else if timeline < 13.0 {
        smootherstep((timeline - 8.0) / 5.0)
    } else if timeline < 17.0 {
        1.0
    } else {
        1.0 - smootherstep((timeline - 17.0) / 3.0)
    }
}

fn area_camera_amount(timeline: f32) -> f32 {
    if timeline < 8.0 {
        0.0
    } else if timeline < 13.0 {
        smootherstep((timeline - 8.0) / 5.0)
    } else if timeline < 17.0 {
        1.0
    } else {
        1.0 - smootherstep((timeline - 17.0) / 3.0)
    }
}

fn smootherstep(amount: f32) -> f32 {
    let t = amount.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp_vec3(start: Vec3, end: Vec3, amount: f32) -> Vec3 {
    vector(
        start.x() + (end.x() - start.x()) * amount,
        start.y() + (end.y() - start.y()) * amount,
        start.z() + (end.z() - start.z()) * amount,
    )
}

fn cylinder_mesh(segments: usize) -> Mesh3d {
    scaled_cylinder_mesh(segments, RADIUS, HEIGHT)
}

fn unit_cylinder_mesh(segments: usize) -> Mesh3d {
    scaled_cylinder_mesh(segments, 1.0, 1.0)
}

fn scaled_cylinder_mesh(segments: usize, radius: f32, height: f32) -> Mesh3d {
    let half_height = height * 0.5;
    let mut vertices = Vec::with_capacity(segments * 2 + 2);
    for y in [-half_height, half_height] {
        for segment in 0..segments {
            let angle = TAU * segment as f32 / segments as f32;
            vertices.push(vector(radius * angle.sin(), y, radius * angle.cos()));
        }
    }
    let bottom_center = vertices.len() as u32;
    vertices.push(vector(0.0, -half_height, 0.0));
    let top_center = vertices.len() as u32;
    vertices.push(vector(0.0, half_height, 0.0));

    let mut triangles = Vec::with_capacity(segments * 12);
    let mut edges = Vec::with_capacity(segments * 2 + 4);
    for segment in 0..segments {
        let next = (segment + 1) % segments;
        let bottom = segment as u32;
        let bottom_next = next as u32;
        let top = (segment + segments) as u32;
        let top_next = (next + segments) as u32;
        triangles.extend_from_slice(&[bottom, top, top_next, bottom, top_next, bottom_next]);
        triangles.extend_from_slice(&[bottom_center, bottom_next, bottom]);
        triangles.extend_from_slice(&[top_center, top, top_next]);
        edges.push(MeshEdge3d::new(bottom, bottom_next).expect("ring edge is valid"));
        edges.push(MeshEdge3d::new(top, top_next).expect("ring edge is valid"));
        if segment % (segments / 4).max(1) == 0 {
            edges.push(MeshEdge3d::new(bottom, top).expect("generator edge is valid"));
        }
    }
    Mesh3d::with_display_edges(vertices, triangles, edges).expect("cylinder topology is valid")
}

fn panel_mesh(width: f32, height: f32) -> Mesh3d {
    let half_width = width * 0.5;
    let half_height = height * 0.5;
    Mesh3d::with_display_edges(
        vec![
            vector(-half_width, -half_height, 0.0),
            vector(half_width, -half_height, 0.0),
            vector(half_width, half_height, 0.0),
            vector(-half_width, half_height, 0.0),
        ],
        vec![0, 1, 2, 0, 2, 3],
        edge_list(&[(0, 1), (1, 2), (2, 3), (3, 0)]),
    )
    .expect("panel topology is valid")
}

fn disk_mesh(segments: usize) -> Mesh3d {
    let mut vertices = Vec::with_capacity(segments + 1);
    vertices.push(Vec3::ZERO);
    for segment in 0..segments {
        let angle = TAU * segment as f32 / segments as f32;
        vertices.push(vector(angle.cos(), angle.sin(), 0.0));
    }
    let mut triangles = Vec::with_capacity(segments * 3);
    let mut edges = Vec::with_capacity(segments);
    for segment in 0..segments {
        let current = (segment + 1) as u32;
        let next = ((segment + 1) % segments + 1) as u32;
        triangles.extend_from_slice(&[0, current, next]);
        edges.push(MeshEdge3d::new(current, next).expect("disk edge is valid"));
    }
    Mesh3d::with_display_edges(vertices, triangles, edges).expect("disk topology is valid")
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
