//! Bounded renderer fixture for a host-simulated supernova remnant.
//!
//! This is visual workload data, not astrophysics. It exercises a low-resolution
//! scalar gas layer plus budgeted instanced ejecta and a black-hole marker.

use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use sim_engine::{
    Camera2d, Color, ColorMap, ColorStop, LayeredVisualizationOptions, ParticleField2d,
    ParticleInstance2d, ParticleRenderBudget, RenderReport, RenderTarget2d, RendererPresentMode,
    ScalarField, ScalarFieldTexture, Vec2, WgpuRenderer, WgpuRendererOptions,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const FIELD_WIDTH: usize = 384;
const FIELD_HEIGHT: usize = 216;
const TARGET_SCALE: u32 = 2;
const TARGET_FRAME_INTERVAL: Duration = Duration::from_nanos(8_333_333);
const GAS_UPDATE_INTERVAL: u64 = 4;
const MAX_PARTICLE_UPDATES_PER_FRAME: usize = 12_500;
const DEFAULT_PARTICLE_COUNT: usize = 100_000;
const DEFAULT_VISIBLE_BUDGET: usize = 30_000;

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    EventLoop::new()?.run_app(&mut StarRemnantApp::new())?;
    Ok(())
}

struct StarRemnantApp {
    window: Option<Arc<Window>>,
    renderer: Option<WgpuRenderer>,
    gas_texture: Option<ScalarFieldTexture>,
    composite_target: Option<RenderTarget2d>,
    particles: Option<ParticleField2d>,
    color_map: ColorMap,
    visualization_options: LayeredVisualizationOptions,
    camera: Camera2d,
    particle_count: usize,
    visible_budget: usize,
    frame_index: u64,
    started_at: Instant,
    previous_frame_at: Instant,
    next_frame_at: Instant,
    playing: bool,
    uncapped: bool,
    recovery_smoke: bool,
    recovery_count: usize,
    metrics: StressMetrics,
}

impl StarRemnantApp {
    fn new() -> Self {
        let now = Instant::now();
        let particle_count = bounded_env("SIM_ENGINE_STAR_PARTICLES", DEFAULT_PARTICLE_COUNT);
        let visible_budget = bounded_env("SIM_ENGINE_STAR_VISIBLE_BUDGET", DEFAULT_VISIBLE_BUDGET)
            .min(particle_count + 1);
        let color_map = ColorMap::new(vec![
            ColorStop::new(0.0, Color::rgba(0.002, 0.004, 0.015, 1.0)).unwrap(),
            ColorStop::new(0.22, Color::rgba(0.03, 0.02, 0.12, 1.0)).unwrap(),
            ColorStop::new(0.50, Color::rgba(0.35, 0.025, 0.08, 1.0)).unwrap(),
            ColorStop::new(0.78, Color::rgba(0.95, 0.20, 0.025, 1.0)).unwrap(),
            ColorStop::new(1.0, Color::rgba(1.0, 0.92, 0.52, 1.0)).unwrap(),
        ])
        .unwrap();
        Self {
            window: None,
            renderer: None,
            gas_texture: None,
            composite_target: None,
            particles: None,
            color_map,
            visualization_options: LayeredVisualizationOptions::new(
                (0.0, 1.0),
                Color::BLACK,
                Color::BLACK,
            )
            .unwrap(),
            camera: Camera2d::new(Vec2::ZERO, 1.35).unwrap(),
            particle_count,
            visible_budget,
            frame_index: 0,
            started_at: now,
            previous_frame_at: now,
            next_frame_at: now,
            playing: true,
            uncapped: std::env::args().any(|arg| arg == "--benchmark" || arg == "--uncapped"),
            recovery_smoke: std::env::args().any(|arg| arg == "--recovery-smoke"),
            recovery_count: 0,
            metrics: StressMetrics::new(now),
        }
    }

    fn time_seconds(&self, now: Instant) -> f32 {
        if self.playing {
            now.saturating_duration_since(self.started_at).as_secs_f32()
        } else {
            self.previous_frame_at
                .saturating_duration_since(self.started_at)
                .as_secs_f32()
        }
    }

    fn target_size(width: u32, height: u32) -> (u32, u32) {
        (
            (width / TARGET_SCALE).max(1),
            (height / TARGET_SCALE).max(1),
        )
    }

    fn recreate_target(&mut self, width: u32, height: u32) {
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        let (width, height) = Self::target_size(width, height);
        self.composite_target = Some(
            renderer
                .create_render_target(width, height)
                .expect("bounded target fits the active device"),
        );
    }

    fn update_visual_state(&mut self, time: f32) -> (Duration, Duration) {
        let mut gas_update = Duration::ZERO;
        if self.frame_index.is_multiple_of(GAS_UPDATE_INTERVAL) {
            let started = Instant::now();
            let field = build_gas_field(time);
            if let (Some(renderer), Some(texture)) =
                (self.renderer.as_ref(), self.gas_texture.as_mut())
            {
                renderer
                    .update_scalar_field_texture(texture, field)
                    .expect("generated scalar field remains finite");
            }
            gas_update = started.elapsed();
        }

        let started = Instant::now();
        let update_slices = self.particle_count.div_ceil(MAX_PARTICLE_UPDATES_PER_FRAME);
        let slice = self.frame_index as usize % update_slices;
        let slice_size = self.particle_count.div_ceil(update_slices);
        let first = slice * slice_size;
        let end = (first + slice_size).min(self.particle_count);
        if first < end {
            let replacement: Vec<_> = (first..end)
                .map(|index| ejecta_particle(index, time))
                .collect();
            if let (Some(renderer), Some(field)) = (self.renderer.as_ref(), self.particles.as_mut())
            {
                renderer
                    .update_particle_field_range(field, first, &replacement)
                    .expect("particle slice remains inside the retained field");
            }
        }
        (gas_update, started.elapsed())
    }

    fn recreate_renderer_resources(&mut self) -> Duration {
        let started = Instant::now();
        let old_gas = self
            .gas_texture
            .take()
            .expect("recovery needs a gas texture");
        let old_target = self
            .composite_target
            .take()
            .expect("recovery needs a render target");
        let old_particles = self
            .particles
            .take()
            .expect("recovery needs a particle field");
        let renderer = self.renderer.as_mut().expect("recovery needs a renderer");
        pollster::block_on(renderer.recover_device_and_surface())
            .expect("recover renderer device and surface");
        let gas = renderer
            .restore_scalar_field_texture(&old_gas)
            .expect("restore scalar field on replacement device");
        let particles = renderer
            .restore_particle_field(&old_particles)
            .expect("restore particles on replacement device");
        let target = renderer
            .restore_render_target(&old_target)
            .expect("restore empty target on replacement device");
        self.gas_texture = Some(gas);
        self.composite_target = Some(target);
        self.particles = Some(particles);
        self.recovery_count += 1;
        let now = Instant::now();
        self.metrics = StressMetrics::new(now);
        started.elapsed()
    }
}

impl ApplicationHandler for StarRemnantApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Sim;Engine Star Remnant Stress")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("create stress window"),
        );
        let size = window.inner_size();
        let present_mode = if self.uncapped {
            RendererPresentMode::NoVsync
        } else {
            RendererPresentMode::Vsync
        };
        let options = WgpuRendererOptions::new(present_mode, window.scale_factor()).unwrap();
        let renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            options,
        ))
        .expect("create stress renderer");
        let gas_texture = renderer
            .create_scalar_field_texture(build_gas_field(0.0))
            .expect("create gas texture");
        let mut instances: Vec<_> = (0..self.particle_count)
            .map(|index| ejecta_particle(index, 0.0))
            .collect();
        instances.push(black_hole_particle());
        let instance_bytes = std::mem::size_of::<[f32; 8]>();
        let budget = ParticleRenderBudget::new(
            self.visible_budget,
            self.visible_budget * instance_bytes,
            self.visible_budget * instance_bytes,
        )
        .and_then(|budget| budget.with_max_visibility_checks(self.visible_budget))
        .expect("stress budget fits particles");
        let particles = renderer
            .create_particle_field_with_budget(&instances, budget)
            .expect("create budgeted particle field");
        let (target_width, target_height) = Self::target_size(size.width, size.height);
        let target = renderer
            .create_render_target(target_width, target_height)
            .expect("create low-resolution composite target");
        println!(
            "star stress: {} retained particles, {} visible cap, {}x{} field, {}x{} target",
            self.particle_count + 1,
            self.visible_budget,
            FIELD_WIDTH,
            FIELD_HEIGHT,
            target_width,
            target_height
        );
        self.window = Some(window.clone());
        self.renderer = Some(renderer);
        self.gas_texture = Some(gas_texture);
        self.composite_target = Some(target);
        self.particles = Some(particles);
        let now = Instant::now();
        self.started_at = now;
        self.previous_frame_at = now;
        self.next_frame_at = now;
        self.metrics = StressMetrics::new(now);
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
                    PhysicalKey::Code(KeyCode::Space) => self.playing = !self.playing,
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        let elapsed = self.recreate_renderer_resources();
                        println!(
                            "star stress: renderer/surface recovery {} completed in {:.1} ms",
                            self.recovery_count,
                            elapsed.as_secs_f64() * 1000.0
                        );
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
                self.recreate_target(size.width, size.height);
                window.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(renderer) = self.renderer.as_mut() {
                    let size = window.inner_size();
                    renderer
                        .resize_with_scale_factor(size.width, size.height, scale_factor)
                        .expect("window scale is finite");
                    self.recreate_target(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let frame_started = Instant::now();
                let time = self.time_seconds(frame_started);
                let (gas_update, particle_update) = self.update_visual_state(time);
                let (Some(renderer), Some(gas), Some(target), Some(particles)) = (
                    self.renderer.as_mut(),
                    self.gas_texture.as_ref(),
                    self.composite_target.as_ref(),
                    self.particles.as_mut(),
                ) else {
                    return;
                };
                window.pre_present_notify();
                let report = renderer
                    .render_layered_visualization(
                        target,
                        gas,
                        &self.color_map,
                        particles,
                        &self.camera,
                        self.visualization_options,
                    )
                    .expect("render bounded layered visualization");
                if let Some(title) = self.metrics.record(
                    frame_started,
                    gas_update,
                    particle_update,
                    report,
                    (particles, gas, target),
                ) {
                    window.set_title(&title);
                }
                self.frame_index = self.frame_index.wrapping_add(1);
                self.previous_frame_at = frame_started;
                if self.recovery_smoke
                    && self.frame_index.is_multiple_of(240)
                    && self.recovery_count < 2
                {
                    let elapsed = self.recreate_renderer_resources();
                    println!(
                        "star stress: renderer/surface recovery {} completed in {:.1} ms",
                        self.recovery_count,
                        elapsed.as_secs_f64() * 1000.0
                    );
                    if self.recovery_count == 2 {
                        println!("star stress: recovery smoke passed");
                        event_loop.exit();
                        return;
                    }
                }
                if self.uncapped {
                    window.request_redraw();
                    event_loop.set_control_flow(ControlFlow::Poll);
                } else {
                    self.next_frame_at = frame_started + TARGET_FRAME_INTERVAL;
                    event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame_at));
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.uncapped || Instant::now() >= self.next_frame_at {
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame_at));
        }
    }
}

struct StressMetrics {
    started: Instant,
    previous: Instant,
    frames: usize,
    intervals: Duration,
    renderer_cpu: Duration,
    gas_update: Duration,
    particle_update: Duration,
}

impl StressMetrics {
    fn new(now: Instant) -> Self {
        Self {
            started: now,
            previous: now,
            frames: 0,
            intervals: Duration::ZERO,
            renderer_cpu: Duration::ZERO,
            gas_update: Duration::ZERO,
            particle_update: Duration::ZERO,
        }
    }

    fn record(
        &mut self,
        now: Instant,
        gas_update: Duration,
        particle_update: Duration,
        report: RenderReport,
        resources: (&ParticleField2d, &ScalarFieldTexture, &RenderTarget2d),
    ) -> Option<String> {
        let (particles, gas, target) = resources;
        if self.frames > 0 {
            self.intervals += now.saturating_duration_since(self.previous);
        }
        self.previous = now;
        self.frames += 1;
        self.renderer_cpu += report.metrics().total_cpu();
        self.gas_update += gas_update;
        self.particle_update += particle_update;
        let elapsed = now.saturating_duration_since(self.started);
        if elapsed < Duration::from_secs(1) {
            return None;
        }
        let frames = self.frames as f64;
        let fps = frames / elapsed.as_secs_f64();
        let frame_ms =
            self.intervals.as_secs_f64() * 1000.0 / self.frames.saturating_sub(1).max(1) as f64;
        let renderer_ms = self.renderer_cpu.as_secs_f64() * 1000.0 / frames;
        let gas_ms = self.gas_update.as_secs_f64() * 1000.0 / frames;
        let particle_ms = self.particle_update.as_secs_f64() * 1000.0 / frames;
        let stats = particles.statistics();
        let cpu_mb =
            (particles.cpu_allocation_bytes() + gas.recovery_memory_bytes()) as f64 / 1_048_576.0;
        let gpu_mb = (particles.gpu_allocation_bytes()
            + gas.gpu_allocation_bytes()
            + target.allocation_bytes()) as f64
            / 1_048_576.0;
        println!(
            "star stress: {fps:.1} fps, frame {frame_ms:.2} ms, renderer {renderer_ms:.2} ms, gas update {gas_ms:.2} ms, particle prep {particle_ms:.2} ms, checked {}, rendered {}, budget-limited {}, CPU {:.1} MiB, bounded GPU {:.1} MiB",
            stats.visibility_checked(),
            stats.rendered(),
            stats.budget_limited(),
            cpu_mb,
            gpu_mb
        );
        let title = format!("Sim;Engine Star Remnant | {fps:.1} FPS | render {renderer_ms:.2} ms");
        *self = Self::new(now);
        Some(title)
    }
}

fn build_gas_field(time: f32) -> ScalarField {
    let mut values = Vec::with_capacity(FIELD_WIDTH * FIELD_HEIGHT);
    for y in 0..FIELD_HEIGHT {
        let py = (y as f32 / (FIELD_HEIGHT - 1) as f32 - 0.5) * 2.0;
        for x in 0..FIELD_WIDTH {
            let px = (x as f32 / (FIELD_WIDTH - 1) as f32 - 0.5) * 3.2;
            let radius = (px * px + py * py).sqrt();
            let angle = py.atan2(px);
            let shell =
                (-((radius - 0.62 - (angle * 5.0 + time * 0.4).sin() * 0.08) / 0.18).powi(2)).exp();
            let plume = ((angle * 7.0 - radius * 9.0 + time * 0.7).sin() * 0.5 + 0.5)
                * (-radius * 1.3).exp();
            let cavity = (radius / 0.16).clamp(0.0, 1.0);
            values.push((shell * 0.78 + plume * 0.35).clamp(0.0, 1.0) * cavity);
        }
    }
    ScalarField::new(FIELD_WIDTH, FIELD_HEIGHT, values).unwrap()
}

fn ejecta_particle(index: usize, time: f32) -> ParticleInstance2d {
    let phase = index as f32 * 2.399_963_1;
    let lane = (index % 997) as f32 / 996.0;
    let radius = 48.0 + lane.sqrt() * 275.0 + (phase * 0.71 + time * 0.18).sin() * 14.0;
    let angle = phase + time * (0.025 + (index % 13) as f32 * 0.0015);
    let position = Vec2::new(angle.cos() * radius, angle.sin() * radius * 0.58);
    let heat = 1.0 - lane;
    ParticleInstance2d::new(
        position,
        0.7 + heat * 1.8,
        Color::rgba(1.0, 0.12 + heat * 0.55, 0.025, 0.10 + heat * 0.42),
        lane * 4.0,
    )
    .unwrap()
}

fn black_hole_particle() -> ParticleInstance2d {
    ParticleInstance2d::new(Vec2::ZERO, 22.0, Color::BLACK, 10.0).unwrap()
}

fn bounded_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| (1..=1_000_000).contains(value))
        .unwrap_or(default)
}
