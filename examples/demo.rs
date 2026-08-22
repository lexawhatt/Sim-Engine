use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use sim_engine::{
    BlendMode, Camera2d, Color, ColorMap, DynamicMesh2d, DynamicVertex2d, Easing, Fill,
    Interpolate, Layer, LinearGradient, Palette, ParticleField2d, ParticleInstance2d,
    PreparedScene, Rect, RenderTarget2d, RendererFrameMetrics, RendererPresentMode, ScalarField,
    ScalarFieldTexture, Scene, ScreenClipRect, Shadow, ShapeStyle, TrailBuffer2d, Tween, Vec2,
    VectorField, WgpuRenderer, WgpuRendererOptions,
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
    dynamic_mesh: Option<DynamicMesh2d>,
    particle_field: Option<ParticleField2d>,
    scalar_field_texture: Option<ScalarFieldTexture>,
    heatmap_target: Option<RenderTarget2d>,
    heatmap_trails: Option<TrailBuffer2d>,
    heatmap_color_map: Option<ColorMap>,
    present_mode: RendererPresentMode,
    camera: Camera2d,
    zoom: Tween<f32>,
    frame_metrics: FrameMetrics,
    dynamic_mesh_segments: usize,
    particle_count: usize,
    benchmark_frames_remaining: Option<usize>,
    benchmark_warmup_frames_remaining: usize,
    started_at: Instant,
    last_frame: Instant,
}

impl DemoApplication {
    fn new() -> Self {
        let now = Instant::now();
        let camera = Camera2d::new(Vec2::ZERO, 2.2).expect("demo camera zoom is valid");
        let benchmark = dynamic_mesh_benchmark();
        Self {
            window: None,
            renderer: None,
            prepared_scene: None,
            dynamic_mesh: None,
            particle_field: None,
            scalar_field_texture: None,
            heatmap_target: None,
            heatmap_trails: None,
            heatmap_color_map: None,
            present_mode: demo_present_mode(),
            camera,
            zoom: Tween::new(camera.zoom()).to(
                2.8,
                Duration::from_millis(1800),
                Easing::EaseInOutCubic,
            ),
            frame_metrics: FrameMetrics::new(now),
            dynamic_mesh_segments: benchmark.segments,
            particle_count: particle_demo_count(),
            benchmark_frames_remaining: benchmark.frame_count,
            benchmark_warmup_frames_remaining: benchmark.warmup_frames,
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

        let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            renderer_options,
        ))
        .expect("create renderer");

        if demo_uses_heatmap() {
            self.scalar_field_texture = Some(
                renderer
                    .create_scalar_field_texture(build_heatmap_field(0.0))
                    .expect("demo scalar field is valid"),
            );
            self.heatmap_target = Some(
                renderer
                    .create_render_target(size.width.max(1), size.height.max(1))
                    .expect("demo render target dimensions are valid"),
            );
            if demo_uses_heatmap_trails() {
                self.heatmap_trails = Some(
                    renderer
                        .create_trail_buffer(size.width.max(1), size.height.max(1))
                        .expect("demo trail dimensions are valid"),
                );
            }
            self.heatmap_color_map = Some(
                ColorMap::linear(
                    Color::rgba(0.02, 0.05, 0.14, 1.0),
                    Color::rgba(1.0, 0.66, 0.12, 1.0),
                )
                .expect("demo color map is valid"),
            );
            println!("demo geometry mode: Scalar heatmap");
        } else if self.benchmark_frames_remaining.is_some() || demo_uses_dynamic_mesh() {
            self.dynamic_mesh = Some(
                renderer
                    .create_dynamic_mesh(&build_dynamic_mesh_vertices(
                        0.0,
                        self.dynamic_mesh_segments,
                    ))
                    .expect("demo dynamic mesh vertices are valid"),
            );
            if let Some(frame_count) = self.benchmark_frames_remaining {
                println!(
                    "dynamic mesh benchmark: {frame_count} measured frames after {} warmup frames, {} segments / {} vertices",
                    self.benchmark_warmup_frames_remaining,
                    self.dynamic_mesh_segments,
                    self.dynamic_mesh_segments * 6,
                );
            } else {
                println!("demo geometry mode: Dynamic mesh");
            }
        } else if demo_uses_particle_field() {
            self.particle_field = Some(
                renderer
                    .create_particle_field(&build_particle_instances(0.0, self.particle_count))
                    .expect("demo particle instances are valid"),
            );
            println!(
                "demo geometry mode: Instanced particles ({})",
                self.particle_count
            );
        } else if demo_uses_vector_field() {
            println!("demo geometry mode: Vector field");
        } else if demo_uses_prepared_scene() {
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
                    if self.heatmap_target.is_some() {
                        self.heatmap_target = Some(
                            renderer
                                .create_render_target(size.width.max(1), size.height.max(1))
                                .expect("demo render target dimensions are valid"),
                        );
                    }
                    if self.heatmap_trails.is_some() {
                        self.heatmap_trails = Some(
                            renderer
                                .create_trail_buffer(size.width.max(1), size.height.max(1))
                                .expect("demo trail dimensions are valid"),
                        );
                    }
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
                let scene = (self.prepared_scene.is_none()
                    && self.dynamic_mesh.is_none()
                    && self.particle_field.is_none()
                    && self.scalar_field_texture.is_none())
                .then(|| {
                    let logical_size = Vec2::new(logical_size.0, logical_size.1);
                    if demo_uses_vector_field() {
                        build_vector_field_scene(time_seconds, logical_size)
                    } else {
                        build_scene(time_seconds, logical_size)
                    }
                });
                let scene_duration = frame_started_at.elapsed();
                if self.present_mode == RendererPresentMode::Vsync {
                    window.pre_present_notify();
                }

                let (renderer_metrics, command_count, dynamic_mesh_update) = match self
                    .renderer
                    .as_mut()
                {
                    Some(renderer) => {
                        if let Some(scalar_field_texture) = self.scalar_field_texture.as_mut() {
                            let field = build_heatmap_field(time_seconds);
                            if let Err(error) =
                                renderer.update_scalar_field_texture(scalar_field_texture, field)
                            {
                                eprintln!("demo heatmap update error: {error:?}");
                                return;
                            }
                            let Some(color_map) = self.heatmap_color_map.as_ref() else {
                                return;
                            };
                            let Some(heatmap_target) = self.heatmap_target.as_ref() else {
                                return;
                            };
                            if let Err(error) = renderer.render_scalar_field_texture_to_target(
                                heatmap_target,
                                scalar_field_texture,
                                color_map,
                                (0.0, 1.0),
                                Palette::sim().background(),
                            ) {
                                eprintln!("demo heatmap target renderer error: {error:?}");
                                return;
                            }
                            let composition = if let Some(trails) = self.heatmap_trails.as_mut() {
                                renderer
                                    .accumulate_trail_buffer(
                                        trails,
                                        heatmap_target,
                                        0.90,
                                        0.12,
                                        BlendMode::Alpha,
                                    )
                                    .and_then(|_| {
                                        renderer.compose_trail_buffer(
                                            trails,
                                            BlendMode::Replace,
                                            1.0,
                                            Palette::sim().background(),
                                        )
                                    })
                            } else {
                                renderer.compose_render_target(
                                    heatmap_target,
                                    BlendMode::Replace,
                                    1.0,
                                    Palette::sim().background(),
                                )
                            };
                            match composition {
                                Ok(report) => (
                                    report.metrics(),
                                    scalar_field_texture.field().values().len(),
                                    Duration::ZERO,
                                ),
                                Err(error) => {
                                    eprintln!("demo heatmap composition error: {error:?}");
                                    return;
                                }
                            }
                        } else if let Some(particle_field) = self.particle_field.as_mut() {
                            let instances =
                                build_particle_instances(time_seconds, self.particle_count);
                            if let Err(error) =
                                renderer.update_particle_field(particle_field, &instances)
                            {
                                eprintln!("demo particle field update error: {error:?}");
                                return;
                            }
                            match renderer.render_particle_field_with_metrics(
                                particle_field,
                                Palette::sim().background(),
                                &self.camera,
                            ) {
                                Ok(report) => (
                                    report.metrics(),
                                    particle_field.statistics().rendered(),
                                    Duration::ZERO,
                                ),
                                Err(error) => {
                                    eprintln!("demo particle renderer error: {error:?}");
                                    return;
                                }
                            }
                        } else {
                            match self.dynamic_mesh.as_mut() {
                                Some(dynamic_mesh) => {
                                    let vertices = build_dynamic_mesh_vertices(
                                        time_seconds,
                                        self.dynamic_mesh_segments,
                                    );
                                    let update = match renderer
                                        .update_dynamic_mesh_with_metrics(dynamic_mesh, &vertices)
                                    {
                                        Ok(update) => update,
                                        Err(error) => {
                                            eprintln!("demo dynamic mesh update error: {error:?}");
                                            return;
                                        }
                                    };
                                    match renderer.render_dynamic_mesh_with_metrics(
                                        dynamic_mesh,
                                        Palette::sim().background(),
                                        &self.camera,
                                    ) {
                                        Ok(report) => (
                                            report.metrics(),
                                            dynamic_mesh.vertex_count() / 3,
                                            update.upload(),
                                        ),
                                        Err(error) => {
                                            eprintln!(
                                                "demo dynamic mesh renderer error: {error:?}"
                                            );
                                            return;
                                        }
                                    }
                                }
                                None => match self.prepared_scene.as_ref() {
                                    Some(prepared_scene) => {
                                        match renderer.render_prepared_with_metrics(
                                            prepared_scene,
                                            &self.camera,
                                        ) {
                                            Ok(report) => (
                                                report.metrics(),
                                                prepared_scene.command_count(),
                                                Duration::ZERO,
                                            ),
                                            Err(error) => {
                                                eprintln!(
                                                    "demo prepared renderer error: {error:?}"
                                                );
                                                return;
                                            }
                                        }
                                    }
                                    None => {
                                        let Some(scene) = scene.as_ref() else {
                                            return;
                                        };
                                        match renderer.render_with_metrics(scene, &self.camera) {
                                            Ok(report) => (
                                                report.metrics(),
                                                scene.command_count(),
                                                Duration::ZERO,
                                            ),
                                            Err(error) => {
                                                eprintln!("demo renderer error: {error:?}");
                                                return;
                                            }
                                        }
                                    }
                                },
                            }
                        }
                    }
                    None => return,
                };

                let measuring_benchmark = self.benchmark_frames_remaining.is_some();
                let mut finish_benchmark = false;
                if measuring_benchmark && self.benchmark_warmup_frames_remaining > 0 {
                    self.benchmark_warmup_frames_remaining -= 1;
                    if self.benchmark_warmup_frames_remaining == 0 {
                        self.frame_metrics = FrameMetrics::new(frame_started_at);
                        println!("dynamic mesh benchmark: warmup complete");
                    }
                } else {
                    if let Some(remaining) = self.benchmark_frames_remaining.as_mut() {
                        *remaining -= 1;
                        finish_benchmark = *remaining == 0;
                    }
                    self.frame_metrics.record(
                        frame_started_at,
                        scene_duration,
                        renderer_metrics,
                        command_count,
                        dynamic_mesh_update,
                        finish_benchmark,
                    );
                }

                if finish_benchmark {
                    println!("dynamic mesh benchmark: complete");
                    event_loop.exit();
                    return;
                }

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

fn demo_uses_dynamic_mesh() -> bool {
    std::env::var("SIM_ENGINE_DYNAMIC_MESH_DEMO").is_ok_and(|value| {
        value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
    })
}

fn demo_uses_particle_field() -> bool {
    std::env::var("SIM_ENGINE_PARTICLE_DEMO").is_ok_and(|value| {
        value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
    })
}

fn demo_uses_heatmap() -> bool {
    std::env::var("SIM_ENGINE_HEATMAP_DEMO").is_ok_and(|value| {
        value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
    })
}

fn demo_uses_heatmap_trails() -> bool {
    std::env::var("SIM_ENGINE_HEATMAP_TRAILS").is_ok_and(|value| {
        value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
    })
}

fn demo_uses_vector_field() -> bool {
    std::env::var("SIM_ENGINE_VECTOR_FIELD_DEMO").is_ok_and(|value| {
        value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
    })
}

#[derive(Clone, Copy)]
struct DynamicMeshBenchmark {
    frame_count: Option<usize>,
    warmup_frames: usize,
    segments: usize,
}

fn dynamic_mesh_benchmark() -> DynamicMeshBenchmark {
    const DEFAULT_SEGMENTS: usize = 160;
    let frame_count = std::env::var("SIM_ENGINE_DYNAMIC_MESH_BENCHMARK_FRAMES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| *count > 0);
    let warmup_frames = std::env::var("SIM_ENGINE_DYNAMIC_MESH_BENCHMARK_WARMUP_FRAMES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(120);
    let segments = std::env::var("SIM_ENGINE_DYNAMIC_MESH_SEGMENTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| (1..=1_000_000).contains(count))
        .unwrap_or(DEFAULT_SEGMENTS);
    DynamicMeshBenchmark {
        frame_count,
        warmup_frames,
        segments,
    }
}

fn particle_demo_count() -> usize {
    std::env::var("SIM_ENGINE_PARTICLE_COUNT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| (1..=1_000_000).contains(count))
        .unwrap_or(1_500)
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
    total_dynamic_mesh_update_duration: Duration,
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
            total_dynamic_mesh_update_duration: Duration::ZERO,
        }
    }

    fn record(
        &mut self,
        frame_started_at: Instant,
        scene_duration: Duration,
        renderer_metrics: RendererFrameMetrics,
        commands: usize,
        dynamic_mesh_update: Duration,
        force_report: bool,
    ) {
        let geometry_mode = if renderer_metrics.geometry_reused() {
            "prepared"
        } else if renderer_metrics.geometry_streamed() {
            "dynamic mesh"
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
        self.total_dynamic_mesh_update_duration += dynamic_mesh_update;

        let report_duration = frame_started_at.saturating_duration_since(self.report_started_at);
        if report_duration < Duration::from_secs(1) && !force_report {
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
        let average_dynamic_mesh_update_ms =
            self.total_dynamic_mesh_update_duration.as_secs_f64() * 1000.0 / self.frames as f64;
        let average_idle_scheduler_ms =
            (average_frame_ms - average_scene_ms - average_renderer_cpu_ms).max(0.0);

        println!(
            "demo fps: {fps:.1}, avg frame: {average_frame_ms:.2} ms, scene: {average_scene_ms:.2} ms, renderer cpu: {average_renderer_cpu_ms:.2} ms, tessellate: {average_tessellation_ms:.2} ms, geometry upload: {average_upload_ms:.2} ms, dynamic update: {average_dynamic_mesh_update_ms:.2} ms, camera upload: {average_camera_uniform_upload_ms:.3} ms, acquire/wait: {average_surface_acquire_ms:.2} ms, submit/present cpu: {average_encode_submit_present_ms:.2} ms, idle/scheduler: {average_idle_scheduler_ms:.2} ms, geometry: {geometry_mode}, commands: {commands}"
        );

        *self = Self::new(frame_started_at);
    }
}

fn build_scene(time_seconds: f32, surface_size: Vec2) -> Scene {
    let palette = Palette::sim();
    let mut scene = Scene::new(palette.background()).expect("palette is finite");

    let plot_margin = 48.0;
    let plot_clip = ScreenClipRect::from_min_size(
        Vec2::splat(plot_margin),
        Vec2::new(
            (surface_size.x - plot_margin * 2.0).max(0.0),
            (surface_size.y - plot_margin * 2.0).max(0.0),
        ),
    );
    scene
        .with_screen_clip(plot_clip.expect("plot clip is valid"), |scene| {
            draw_grid(scene, palette)
        })
        .expect("plot clip is valid");

    let panel = Rect::from_center_size(Vec2::new(0.0, -92.0), Vec2::new(330.0, 58.0));
    scene.rect(
        panel,
        10.0,
        ShapeStyle::fill_stroke(
            palette.surface().with_alpha(0.74),
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
    scene.polyline(wave, 3.5, palette.primary().with_alpha(0.92));

    for index in 0..18 {
        let phase = index as f32 * 0.72;
        let orbit = 78.0 + (index % 3) as f32 * 27.0;
        let position = Vec2::new(
            (time_seconds * 0.72 + phase).cos() * orbit,
            (time_seconds * 0.58 + phase * 1.13).sin() * orbit * 0.48 + 24.0,
        );
        let color = if index % 3 == 0 {
            palette.accent()
        } else if index % 3 == 1 {
            palette.secondary()
        } else {
            palette.primary()
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
                palette.warning().with_alpha(0.95),
                palette.accent().with_alpha(0.88),
            )),
            2.0,
            Color::WHITE.with_alpha(0.32),
        )
        .with_shadow(Shadow::new(
            Vec2::new(0.0, 14.0),
            14.0,
            palette.warning().with_alpha(0.20),
        )),
    );

    scene
}

fn build_dynamic_mesh_vertices(time_seconds: f32, segments: usize) -> Vec<DynamicVertex2d> {
    let palette = Palette::sim();
    let mut vertices = Vec::with_capacity(segments * 6);
    for index in 0..segments {
        let amount_start = index as f32 / segments as f32;
        let amount_end = (index + 1) as f32 / segments as f32;
        let x_start = -260.0 + amount_start * 520.0;
        let x_end = -260.0 + amount_end * 520.0;
        let y_start = dynamic_wave_height(x_start, time_seconds);
        let y_end = dynamic_wave_height(x_end, time_seconds);
        let half_width = 3.0
            + (amount_start * std::f32::consts::TAU + time_seconds)
                .sin()
                .abs()
                * 2.0;
        let start_color = palette.primary().with_alpha(0.9);
        let end_color = palette.accent().with_alpha(0.9);
        let top_start = Vec2::new(x_start, y_start - half_width);
        let bottom_start = Vec2::new(x_start, y_start + half_width);
        let top_end = Vec2::new(x_end, y_end - half_width);
        let bottom_end = Vec2::new(x_end, y_end + half_width);
        vertices.extend([
            DynamicVertex2d::new(top_start, 1.5, start_color).expect("finite dynamic mesh vertex"),
            DynamicVertex2d::new(bottom_start, 1.5, start_color)
                .expect("finite dynamic mesh vertex"),
            DynamicVertex2d::new(top_end, 1.5, end_color).expect("finite dynamic mesh vertex"),
            DynamicVertex2d::new(top_end, 1.5, end_color).expect("finite dynamic mesh vertex"),
            DynamicVertex2d::new(bottom_start, 1.5, start_color)
                .expect("finite dynamic mesh vertex"),
            DynamicVertex2d::new(bottom_end, 1.5, end_color).expect("finite dynamic mesh vertex"),
        ]);
    }
    vertices
}

fn build_particle_instances(time_seconds: f32, particle_count: usize) -> Vec<ParticleInstance2d> {
    let palette = Palette::sim();
    let mut particles = Vec::with_capacity(particle_count);
    for index in 0..particle_count {
        let phase = index as f32 * 0.618_034;
        let ring = 24.0 + (index % 80) as f32 * 2.7;
        let angle = phase + time_seconds * (0.35 + (index % 7) as f32 * 0.03);
        let position = Vec2::new(
            angle.cos() * ring + (time_seconds * 0.7 + phase).sin() * 18.0,
            angle.sin() * ring * 0.56 + (time_seconds * 0.45 + phase).cos() * 12.0,
        );
        let color = match index % 3 {
            0 => palette.primary(),
            1 => palette.secondary(),
            _ => palette.accent(),
        }
        .with_alpha(0.48 + (phase + time_seconds).sin().abs() * 0.36);
        particles.push(
            ParticleInstance2d::new(
                position,
                1.5 + (index % 5) as f32 * 0.35,
                color,
                (index % 9) as f32 * 0.15,
            )
            .expect("demo particle is finite"),
        );
    }
    particles
}

fn build_heatmap_field(time_seconds: f32) -> ScalarField {
    const WIDTH: usize = 160;
    const HEIGHT: usize = 96;
    let mut values = Vec::with_capacity(WIDTH * HEIGHT);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let horizontal = x as f32 / (WIDTH - 1) as f32;
            let vertical = y as f32 / (HEIGHT - 1) as f32;
            let wave = ((horizontal * 9.0 + time_seconds * 1.2).sin()
                + (vertical * 7.0 - time_seconds * 0.8).cos())
                * 0.25;
            values.push((horizontal * 0.45 + vertical * 0.35 + wave + 0.25).clamp(0.0, 1.0));
        }
    }
    ScalarField::new(WIDTH, HEIGHT, values).expect("demo heatmap values are finite")
}

fn build_vector_field(time_seconds: f32) -> VectorField {
    const WIDTH: usize = 25;
    const HEIGHT: usize = 15;
    let mut values = Vec::with_capacity(WIDTH * HEIGHT);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let horizontal = x as f32 / (WIDTH - 1) as f32 * 2.0 - 1.0;
            let vertical = y as f32 / (HEIGHT - 1) as f32 * 2.0 - 1.0;
            values.push(Vec2::new(
                -vertical + (time_seconds + horizontal * 2.0).sin() * 0.3,
                horizontal + (time_seconds * 0.7 - vertical * 2.0).cos() * 0.3,
            ));
        }
    }
    VectorField::new(WIDTH, HEIGHT, values).expect("demo vector field is finite")
}

fn build_vector_field_scene(time_seconds: f32, surface_size: Vec2) -> Scene {
    let palette = Palette::sim();
    let mut scene = Scene::new(palette.background()).expect("palette is finite");
    let field = build_vector_field(time_seconds);
    let spacing = Vec2::new(24.0, 24.0);
    let origin = Vec2::new(
        -(field.width().saturating_sub(1) as f32) * spacing.x * 0.5,
        -(field.height().saturating_sub(1) as f32) * spacing.y * 0.5,
    );

    let clip = ScreenClipRect::from_min_size(Vec2::splat(20.0), surface_size - Vec2::splat(40.0))
        .expect("positive demo viewport clip");
    scene
        .with_screen_clip(clip, |scene| {
            for y in 0..field.height() {
                for x in 0..field.width() {
                    let center = origin + Vec2::new(x as f32 * spacing.x, y as f32 * spacing.y);
                    let direction = field
                        .value_at(x, y)
                        .expect("field indices stay in bounds")
                        .normalized();
                    let tip = center + direction * 8.5;
                    let normal = direction.perp() * 3.0;
                    let color = palette
                        .primary()
                        .interpolate(
                            palette.accent(),
                            (y as f32 / (field.height() - 1) as f32).clamp(0.0, 1.0),
                        )
                        .with_alpha(0.88);
                    scene.line(center, tip, 1.6, color);
                    scene.line(tip, tip - direction * 4.8 + normal, 1.4, color);
                    scene.line(tip, tip - direction * 4.8 - normal, 1.4, color);
                }
            }
        })
        .expect("vector field clip is valid");
    scene
}

fn dynamic_wave_height(x: f32, time_seconds: f32) -> f32 {
    (x * 0.045 + time_seconds * 2.0).sin() * 42.0 + (x * 0.018 - time_seconds * 0.8).cos() * 16.0
}

fn draw_grid(scene: &mut Scene, palette: Palette) {
    for line in -10..=10 {
        let coordinate = line as f32 * 32.0;
        let color = if line == 0 {
            palette.axis()
        } else {
            palette.grid()
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
