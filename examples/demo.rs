use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use sim_engine::{
    Camera2d, Color, Easing, Fill, Layer, LinearGradient, Palette, PreparedScene, Rect,
    RendererFrameMetrics, RendererPresentMode, Scene, ScreenClipRect, Shadow, ShapeStyle, Tween,
    Vec2, WgpuRenderer, WgpuRendererOptions,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let event_loop = EventLoop::new()?;
    let mut application = DemoApplication::new();
    event_loop.run_app(&mut application)?;

    Ok(())
}

struct DemoApplication {
    window: Option<Arc<Window>>,
    renderer: Option<WgpuRenderer>,
    prepared_scene: Option<PreparedScene>,
    present_mode: RendererPresentMode,
    camera: Camera2d,
    zoom: Tween<f32>,
    frame_metrics: FrameMetrics,
    started_at: Instant,
    last_frame: Instant,
}

impl DemoApplication {
    fn new() -> Self {
        let now = Instant::now();
        let camera = Camera2d::new(Vec2::ZERO, 2.2).expect("demo camera zoom is valid");
        Self {
            window: None,
            renderer: None,
            prepared_scene: None,
            present_mode: demo_present_mode(),
            camera,
            zoom: Tween::new(camera.zoom()).to(
                2.8,
                Duration::from_millis(1800),
                Easing::EaseInOutCubic,
            ),
            frame_metrics: FrameMetrics::new(now),
            started_at: now,
            last_frame: now,
        }
    }

    fn update_camera(&mut self, now: Instant) -> f32 {
        let dt = now.saturating_duration_since(self.last_frame);
        self.last_frame = now;

        let time_seconds = now.duration_since(self.started_at).as_secs_f32();
        if !self.zoom.is_active() {
            let target = if self.zoom.target() < 2.5 { 2.8 } else { 2.05 };
            self.zoom
                .set_target(target, Duration::from_millis(1800), Easing::EaseInOutCubic);
        }

        self.camera
            .set_zoom(self.zoom.update(dt))
            .expect("demo zoom tween stays positive");
        self.camera
            .set_center(Vec2::new(
                (time_seconds * 0.35).sin() * 18.0,
                (time_seconds * 0.28).cos() * 10.0,
            ))
            .expect("demo camera center stays finite");
        self.camera
            .set_rotation((time_seconds * 0.18).sin() * 0.045)
            .expect("demo camera rotation stays finite");

        time_seconds
    }
}

impl ApplicationHandler for DemoApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("Sim;Engine demo")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(event_loop.create_window(attributes).expect("create window"));
        let size = window.inner_size();
        let scale_factor = window.scale_factor();
        println!("demo present mode: {:?}", self.present_mode);
        let renderer_options = WgpuRendererOptions::new(self.present_mode, scale_factor)
            .expect("window scale factor is valid");

        let renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            renderer_options,
        ))
        .expect("create renderer");

        if demo_uses_prepared_scene() {
            let logical_size = renderer.logical_size();
            let scene = build_scene(0.0, Vec2::new(logical_size.0, logical_size.1));
            self.prepared_scene = Some(renderer.prepare_scene(&scene));
            println!("demo geometry mode: Prepared");
        } else {
            println!("demo geometry mode: Streaming");
        }

        self.window = Some(window);
        self.renderer = Some(renderer);
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
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer
                        .resize_with_scale_factor(size.width, size.height, window.scale_factor())
                        .expect("window scale factor is valid");
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer
                        .set_scale_factor(scale_factor)
                        .expect("window scale factor is valid");
                }
            }
            WindowEvent::RedrawRequested => {
                let frame_started_at = Instant::now();
                let Some(renderer) = self.renderer.as_ref() else {
                    return;
                };
                let logical_size = renderer.logical_size();
                let time_seconds = self.update_camera(frame_started_at);
                let scene = self
                    .prepared_scene
                    .is_none()
                    .then(|| build_scene(time_seconds, Vec2::new(logical_size.0, logical_size.1)));
                let scene_duration = frame_started_at.elapsed();
                if self.present_mode == RendererPresentMode::Vsync {
                    window.pre_present_notify();
                }

                let (renderer_metrics, command_count) = match self.renderer.as_mut() {
                    Some(renderer) => match self.prepared_scene.as_ref() {
                        Some(prepared_scene) => {
                            match renderer
                                .render_prepared_with_metrics(prepared_scene, &self.camera)
                            {
                                Ok(report) => (report.metrics(), prepared_scene.command_count()),
                                Err(error) => {
                                    eprintln!("demo prepared renderer error: {error:?}");
                                    return;
                                }
                            }
                        }
                        None => {
                            let Some(scene) = scene.as_ref() else {
                                return;
                            };
                            match renderer.render_with_metrics(scene, &self.camera) {
                                Ok(report) => (report.metrics(), scene.command_count()),
                                Err(error) => {
                                    eprintln!("demo renderer error: {error:?}");
                                    return;
                                }
                            }
                        }
                    },
                    None => return,
                };

                self.frame_metrics.record(
                    frame_started_at,
                    scene_duration,
                    renderer_metrics,
                    command_count,
                );

                window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn demo_present_mode() -> RendererPresentMode {
    match std::env::var("SIM_ENGINE_PRESENT_MODE") {
        Ok(value) if value == "no-vsync" || value == "novsync" || value == "immediate" => {
            RendererPresentMode::NoVsync
        }
        _ => RendererPresentMode::Vsync,
    }
}

fn demo_uses_prepared_scene() -> bool {
    std::env::var("SIM_ENGINE_PREPARED_SCENE").is_ok_and(|value| {
        value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
    })
}

struct FrameMetrics {
    report_started_at: Instant,
    previous_frame_started_at: Instant,
    frames: u32,
    total_frame_interval: Duration,
    total_scene_duration: Duration,
    total_renderer_cpu_duration: Duration,
    total_tessellation_duration: Duration,
    total_upload_duration: Duration,
    total_camera_uniform_upload_duration: Duration,
    total_surface_acquire_duration: Duration,
    total_encode_submit_present_duration: Duration,
}

impl FrameMetrics {
    fn new(now: Instant) -> Self {
        Self {
            report_started_at: now,
            previous_frame_started_at: now,
            frames: 0,
            total_frame_interval: Duration::ZERO,
            total_scene_duration: Duration::ZERO,
            total_renderer_cpu_duration: Duration::ZERO,
            total_tessellation_duration: Duration::ZERO,
            total_upload_duration: Duration::ZERO,
            total_camera_uniform_upload_duration: Duration::ZERO,
            total_surface_acquire_duration: Duration::ZERO,
            total_encode_submit_present_duration: Duration::ZERO,
        }
    }

    fn record(
        &mut self,
        frame_started_at: Instant,
        scene_duration: Duration,
        renderer_metrics: RendererFrameMetrics,
        commands: usize,
    ) {
        let geometry_mode = if renderer_metrics.geometry_reused() {
            "prepared"
        } else {
            "streaming"
        };
        if self.frames > 0 {
            self.total_frame_interval +=
                frame_started_at.saturating_duration_since(self.previous_frame_started_at);
        }

        self.previous_frame_started_at = frame_started_at;
        self.frames += 1;
        self.total_scene_duration += scene_duration;
        self.total_renderer_cpu_duration += renderer_metrics.total_cpu();
        self.total_tessellation_duration += renderer_metrics.tessellation();
        self.total_upload_duration += renderer_metrics.upload();
        self.total_camera_uniform_upload_duration += renderer_metrics.camera_uniform_upload();
        self.total_surface_acquire_duration += renderer_metrics.surface_acquire();
        self.total_encode_submit_present_duration += renderer_metrics.encode_submit_present();

        let report_duration = frame_started_at.saturating_duration_since(self.report_started_at);
        if report_duration < Duration::from_secs(1) {
            return;
        }

        let fps = self.frames as f64 / report_duration.as_secs_f64();
        let interval_count = self.frames.saturating_sub(1).max(1);
        let average_frame_ms =
            self.total_frame_interval.as_secs_f64() * 1000.0 / interval_count as f64;
        let average_scene_ms =
            self.total_scene_duration.as_secs_f64() * 1000.0 / self.frames as f64;
        let average_renderer_cpu_ms =
            self.total_renderer_cpu_duration.as_secs_f64() * 1000.0 / self.frames as f64;
        let average_tessellation_ms =
            self.total_tessellation_duration.as_secs_f64() * 1000.0 / self.frames as f64;
        let average_upload_ms =
            self.total_upload_duration.as_secs_f64() * 1000.0 / self.frames as f64;
        let average_camera_uniform_upload_ms =
            self.total_camera_uniform_upload_duration.as_secs_f64() * 1000.0 / self.frames as f64;
        let average_surface_acquire_ms =
            self.total_surface_acquire_duration.as_secs_f64() * 1000.0 / self.frames as f64;
        let average_encode_submit_present_ms =
            self.total_encode_submit_present_duration.as_secs_f64() * 1000.0 / self.frames as f64;
        let average_idle_scheduler_ms =
            (average_frame_ms - average_scene_ms - average_renderer_cpu_ms).max(0.0);

        println!(
            "demo fps: {fps:.1}, avg frame: {average_frame_ms:.2} ms, scene: {average_scene_ms:.2} ms, renderer cpu: {average_renderer_cpu_ms:.2} ms, tessellate: {average_tessellation_ms:.2} ms, geometry upload: {average_upload_ms:.2} ms, camera upload: {average_camera_uniform_upload_ms:.3} ms, acquire/wait: {average_surface_acquire_ms:.2} ms, submit/present cpu: {average_encode_submit_present_ms:.2} ms, idle/scheduler: {average_idle_scheduler_ms:.2} ms, geometry: {geometry_mode}, commands: {commands}"
        );

        *self = Self::new(frame_started_at);
    }
}

fn build_scene(time_seconds: f32, surface_size: Vec2) -> Scene {
    let palette = Palette::sim();
    let mut scene = Scene::new(palette.background);

    let plot_margin = 48.0;
    let plot_clip = ScreenClipRect::from_min_size(
        Vec2::splat(plot_margin),
        Vec2::new(
            (surface_size.x - plot_margin * 2.0).max(0.0),
            (surface_size.y - plot_margin * 2.0).max(0.0),
        ),
    );
    scene.with_screen_clip(plot_clip, |scene| draw_grid(scene, palette));

    let panel = Rect::from_center_size(Vec2::new(0.0, -92.0), Vec2::new(330.0, 58.0));
    scene.rect(
        panel,
        10.0,
        ShapeStyle::fill_stroke(
            palette.surface.with_alpha(0.74),
            1.5,
            Color::rgba8(255, 255, 255, 42),
        )
        .with_shadow(Shadow::new(
            Vec2::new(0.0, 18.0),
            12.0,
            Color::BLACK.with_alpha(0.28),
        )),
    );

    let mut wave = Vec::with_capacity(180);
    for index in 0..180 {
        let x = -210.0 + index as f32 * 420.0 / 179.0;
        let y =
            (x * 0.045 + time_seconds * 2.0).sin() * 28.0 + (x * 0.018 - time_seconds).cos() * 12.0;
        wave.push(Vec2::new(x, y));
    }
    scene.polyline(wave, 3.5, palette.primary.with_alpha(0.92));

    for index in 0..18 {
        let phase = index as f32 * 0.72;
        let orbit = 78.0 + (index % 3) as f32 * 27.0;
        let position = Vec2::new(
            (time_seconds * 0.72 + phase).cos() * orbit,
            (time_seconds * 0.58 + phase * 1.13).sin() * orbit * 0.48 + 24.0,
        );
        let color = if index % 3 == 0 {
            palette.accent
        } else if index % 3 == 1 {
            palette.secondary
        } else {
            palette.primary
        };

        scene.circle(
            position,
            5.5 + (phase + time_seconds).sin().abs() * 4.5,
            ShapeStyle::filled(color.with_alpha(0.88)).with_shadow(Shadow::new(
                Vec2::new(0.0, 8.0),
                6.0,
                color.with_alpha(0.16),
            )),
        );
    }

    scene.circle(
        Vec2::new(0.0, 20.0),
        24.0 + (time_seconds * 1.3).sin() * 3.0,
        ShapeStyle::fill_stroke_with(
            Fill::LinearGradient(LinearGradient::new(
                Vec2::new(-32.0, -8.0),
                Vec2::new(34.0, 48.0),
                palette.warning.with_alpha(0.95),
                palette.accent.with_alpha(0.88),
            )),
            2.0,
            Color::WHITE.with_alpha(0.32),
        )
        .with_shadow(Shadow::new(
            Vec2::new(0.0, 14.0),
            14.0,
            palette.warning.with_alpha(0.20),
        )),
    );

    scene
}

fn draw_grid(scene: &mut Scene, palette: Palette) {
    for line in -10..=10 {
        let coordinate = line as f32 * 32.0;
        let color = if line == 0 {
            palette.axis
        } else {
            palette.grid
        };
        let width = if line == 0 { 2.0 } else { 1.0 };

        scene.line_on_layer(
            Layer::BACKGROUND,
            Vec2::new(coordinate, -220.0),
            Vec2::new(coordinate, 220.0),
            width,
            color,
        );
        scene.line_on_layer(
            Layer::BACKGROUND,
            Vec2::new(-340.0, coordinate),
            Vec2::new(340.0, coordinate),
            width,
            color,
        );
    }
}
