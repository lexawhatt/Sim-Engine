//! Repeatable surface benchmarks for the named Sim;X rendering workloads.
//!
//! The executable owns a temporary window only to obtain a real presentation
//! surface. It prints engine-side CPU and surface-acquire percentiles plus
//! deterministic work counters. `--gate` applies the project's Linux release
//! floor; it does not claim that raw timings transfer across unrelated GPUs.

use std::{error::Error, sync::Arc, time::Instant};

use sim_engine::{
    Camera2d, Color, FrameBudget, FramePassOptions, FrameReport, GlyphAtlasBudget, GlyphAtlasEntry,
    GlyphId, GlyphRunBudget, ImageBatchBudget, ImageBudget, ImageSampling, ImageSprite2d,
    ImageTexelRect, Layer, LogicalPixels, LogicalScreenPosition, LogicalScreenVector,
    LogicalViewport, LogicalViewportRegion, PositionedGlyph2d, PreparedScreenScene, Rect, Scene,
    SceneBudget, ScreenScene, ShapeStyle, Vec2, WgpuRenderer, WgpuRendererOptions,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const WARMUP_FRAMES: usize = 20;
const MEASURED_FRAMES: usize = 120;
const STATIC_COMMANDS: usize = 9_000;
const STREAMING_COMMANDS: usize = 1_000;

#[derive(Clone, Copy)]
struct GateThresholds {
    minimum_fps: f64,
    maximum_renderer_work_p95_ms: f64,
    maximum_surface_acquire_p95_ms: f64,
}

fn gate_thresholds(fixture: &str) -> Option<GateThresholds> {
    let maximum_renderer_work_p95_ms = match fixture {
        // This is the only fixture that deliberately rebuilds and tessellates
        // one thousand commands every frame. Give that CPU path its own
        // measured ceiling instead of weakening every retained workload.
        "ui_90_10" => 25.0,
        "dpi_reconfigure" => 10.0,
        "ui_static_10k" | "four_viewports" | "image_atlas" | "scientific_text" => 5.0,
        _ => return None,
    };
    Some(GateThresholds {
        minimum_fps: 60.0,
        maximum_renderer_work_p95_ms,
        maximum_surface_acquire_p95_ms: 25.0,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let configuration = parse_configuration()?;
    let mut application = BenchmarkApplication::new(configuration);
    EventLoop::new()?.run_app(&mut application)?;
    if let Some(failure) = application.failure {
        return Err(failure.into());
    }
    Ok(())
}

struct BenchmarkConfiguration {
    fixture: String,
    gate: bool,
}

fn parse_configuration() -> Result<BenchmarkConfiguration, Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let mut fixture = "ui_90_10".to_owned();
    let mut gate = false;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--fixture" => {
                fixture = arguments.next().ok_or("--fixture requires a value")?;
            }
            "--gate" => gate = true,
            "--help" | "-h" => {
                println!(
                    "Usage: rendering_benchmark_suite [--fixture ui_static_10k|ui_90_10|four_viewports|image_atlas|scientific_text|dpi_reconfigure|hidpi_transition] [--gate]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    Ok(BenchmarkConfiguration { fixture, gate })
}

struct BenchmarkApplication {
    fixture: String,
    gate: bool,
    started: bool,
    failure: Option<String>,
    hidpi: Option<HidpiTransitionState>,
}

struct HidpiTransitionState {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    prepared: PreparedScreenScene,
    scale_events: usize,
    resize_events: usize,
    paired_transitions: usize,
    rendered_after_scale: usize,
    awaiting_resize: bool,
}

impl BenchmarkApplication {
    fn new(configuration: BenchmarkConfiguration) -> Self {
        Self {
            fixture: configuration.fixture,
            gate: configuration.gate,
            started: false,
            failure: None,
            hidpi: None,
        }
    }
}

impl ApplicationHandler for BenchmarkApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.started {
            return;
        }
        self.started = true;
        let result = (|| -> Result<(), Box<dyn Error>> {
            let window = Arc::new(
                event_loop.create_window(
                    Window::default_attributes()
                        .with_title(format!("Sim;Engine benchmark | {}", self.fixture))
                        .with_inner_size(LogicalSize::new(1280.0, 720.0)),
                )?,
            );
            let size = window.inner_size();
            let options = WgpuRendererOptions::new(
                sim_engine::RendererPresentMode::NoVsync,
                window.scale_factor(),
            )?;
            let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
                Arc::clone(&window),
                size.width.max(1),
                size.height.max(1),
                options,
            ))?;
            match self.fixture.as_str() {
                "ui_static_10k" => benchmark_static_ui(&mut renderer, self.gate),
                "ui_90_10" => benchmark_ui_90_10(&mut renderer, self.gate),
                "four_viewports" => benchmark_four_viewports(&mut renderer, self.gate),
                "image_atlas" => benchmark_image_atlas(&mut renderer, self.gate),
                "scientific_text" => benchmark_scientific_text(&mut renderer, self.gate),
                "dpi_reconfigure" => {
                    benchmark_dpi_reconfigure(&mut renderer, size.width, size.height, self.gate)
                }
                "hidpi_transition" => {
                    let scene = build_screen_scene(2_500, 0)?;
                    let prepared = renderer.prepare_screen_scene(&scene)?;
                    println!(
                        "hidpi_transition: move this window to a monitor with a different scale factor, then press Esc"
                    );
                    window.request_redraw();
                    self.hidpi = Some(HidpiTransitionState {
                        window,
                        renderer,
                        prepared,
                        scale_events: 0,
                        resize_events: 0,
                        paired_transitions: 0,
                        rendered_after_scale: 0,
                        awaiting_resize: false,
                    });
                    Ok(())
                }
                fixture => Err(format!("unknown fixture: {fixture}").into()),
            }
        })();
        if let Err(error) = result {
            self.failure = Some(error.to_string());
        }
        if self.hidpi.is_none() || self.failure.is_some() {
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.hidpi.as_mut() else {
            return;
        };
        if state.window.id() != window_id {
            return;
        }
        let exit_requested = matches!(&event, WindowEvent::CloseRequested)
            || matches!(
                &event,
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed
                        && matches!(event.logical_key, Key::Named(NamedKey::Escape))
            );
        let transition_event = matches!(
            &event,
            WindowEvent::ScaleFactorChanged { .. } | WindowEvent::Resized(_)
        );
        let result = match event {
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                state.scale_events = state.scale_events.saturating_add(1);
                state.awaiting_resize = true;
                let size = state.window.inner_size();
                println!(
                    "event=ScaleFactorChanged scale={scale_factor:.3} compositor_physical={}x{}",
                    size.width, size.height
                );
                state
                    .renderer
                    .resize_with_scale_factor(size.width.max(1), size.height.max(1), scale_factor)
                    .map_err(|error| error.to_string())
            }
            WindowEvent::Resized(size) => {
                state.resize_events = state.resize_events.saturating_add(1);
                if state.awaiting_resize {
                    state.paired_transitions = state.paired_transitions.saturating_add(1);
                    state.awaiting_resize = false;
                }
                println!(
                    "event=Resized physical={}x{} scale={:.3}",
                    size.width,
                    size.height,
                    state.window.scale_factor()
                );
                state
                    .renderer
                    .resize_with_scale_factor(
                        size.width.max(1),
                        size.height.max(1),
                        state.window.scale_factor(),
                    )
                    .map_err(|error| error.to_string())
            }
            WindowEvent::RedrawRequested => {
                let result = (|| -> Result<(), Box<dyn Error>> {
                    let mut frame = state
                        .renderer
                        .begin_frame(Color::rgb8(9, 12, 18), FrameBudget::default())?;
                    frame.draw_prepared_screen_scene(&state.prepared, FramePassOptions::new(0))?;
                    let _ = frame.present()?;
                    Ok(())
                })();
                if state.scale_events > 0 && result.is_ok() {
                    state.rendered_after_scale = state.rendered_after_scale.saturating_add(1);
                }
                result.map_err(|error| error.to_string())
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape)) =>
            {
                finish_hidpi_transition(state).map_err(|error| error.to_owned())
            }
            WindowEvent::CloseRequested => {
                finish_hidpi_transition(state).map_err(|error| error.to_owned())
            }
            _ => Ok(()),
        };
        if let Err(error) = result {
            self.failure = Some(error);
            event_loop.exit();
            return;
        }
        if exit_requested {
            event_loop.exit();
        } else if transition_event {
            state.window.request_redraw();
        }
    }
}

fn finish_hidpi_transition(state: &HidpiTransitionState) -> Result<(), &'static str> {
    println!(
        "hidpi_transition scale_events={} resize_events={} paired_transitions={} rendered_after_scale={}",
        state.scale_events,
        state.resize_events,
        state.paired_transitions,
        state.rendered_after_scale
    );
    validate_hidpi_evidence(
        state.scale_events,
        state.paired_transitions,
        state.rendered_after_scale,
    )
}

fn validate_hidpi_evidence(
    scale_events: usize,
    paired_transitions: usize,
    rendered_after_scale: usize,
) -> Result<(), &'static str> {
    if scale_events == 0 {
        return Err("no real ScaleFactorChanged event was observed");
    }
    if paired_transitions == 0 {
        return Err("no Resized event followed the compositor scale transition");
    }
    if rendered_after_scale == 0 {
        return Err("the renderer did not present after the compositor scale transition");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidpi_evidence_requires_a_real_event_and_post_transition_present() {
        assert_eq!(
            validate_hidpi_evidence(0, 1, 1),
            Err("no real ScaleFactorChanged event was observed")
        );
        assert_eq!(
            validate_hidpi_evidence(1, 0, 1),
            Err("no Resized event followed the compositor scale transition")
        );
        assert_eq!(
            validate_hidpi_evidence(1, 1, 0),
            Err("the renderer did not present after the compositor scale transition")
        );
        assert_eq!(validate_hidpi_evidence(1, 1, 1), Ok(()));
    }

    #[test]
    fn performance_gate_keeps_streaming_and_retained_ceiling_separate() {
        let retained = gate_thresholds("four_viewports").expect("known retained fixture");
        let streaming = gate_thresholds("ui_90_10").expect("known streaming fixture");
        assert_eq!(retained.minimum_fps, 60.0);
        assert_eq!(retained.maximum_renderer_work_p95_ms, 5.0);
        assert_eq!(streaming.maximum_renderer_work_p95_ms, 25.0);
        assert!(gate_thresholds("unknown").is_none());
    }
}

fn scene_budget() -> SceneBudget {
    SceneBudget::new(
        12_000,
        24_000,
        2_000_000,
        128 * 1024 * 1024,
        128 * 1024 * 1024,
        12_000,
    )
}

fn build_screen_scene(count: usize, phase: usize) -> Result<ScreenScene, Box<dyn Error>> {
    let mut scene = ScreenScene::with_budget(Color::rgb8(9, 12, 18), scene_budget())?;
    let corner = LogicalPixels::new(1.5)?;
    for index in 0..count {
        let cell = index + phase;
        let x = (cell % 128) as f32 * 10.0;
        let y = ((cell / 128) % 72) as f32 * 10.0;
        let color = if index.is_multiple_of(7) {
            Color::rgb8(48, 129, 180)
        } else {
            Color::rgb8(27, 38, 52)
        };
        scene.try_rect_on_layer(
            Layer::new((index % 8) as i32),
            LogicalScreenPosition::new(x + 1.0, y + 1.0),
            LogicalScreenVector::new(8.0, 8.0),
            corner,
            ShapeStyle::filled(color),
        )?;
    }
    Ok(scene)
}

fn build_world_scene(count: usize, phase: usize) -> Result<Scene, Box<dyn Error>> {
    let mut scene = Scene::with_budget(Color::rgb8(9, 12, 18), scene_budget())?;
    for index in 0..count {
        let cell = index + phase;
        let x = (cell % 64) as f32 * 2.2 - 69.0;
        let y = ((cell / 64) % 40) as f32 * 2.2 - 43.0;
        let color = if index.is_multiple_of(7) {
            Color::rgb8(48, 129, 180)
        } else {
            Color::rgb8(27, 38, 52)
        };
        scene.try_rect_on_layer(
            Layer::new((index % 8) as i32),
            Rect::from_center_size(Vec2::new(x, y), Vec2::splat(1.7)),
            0.25,
            ShapeStyle::filled(color),
        )?;
    }
    Ok(scene)
}

fn benchmark_static_ui(renderer: &mut WgpuRenderer, gate: bool) -> Result<(), Box<dyn Error>> {
    let construction_started = Instant::now();
    let scene = build_screen_scene(STATIC_COMMANDS + STREAMING_COMMANDS, 0)?;
    let construction = construction_started.elapsed();
    let preparation_started = Instant::now();
    let prepared = renderer.prepare_screen_scene(&scene)?;
    let preparation = preparation_started.elapsed();
    measure_frames(
        "ui_static_10k",
        renderer,
        construction,
        gate,
        |renderer, _| {
            let mut frame = renderer.begin_frame(scene.background(), FrameBudget::default())?;
            frame.draw_prepared_screen_scene(&prepared, FramePassOptions::new(0))?;
            Ok(frame.present()?)
        },
    )?;
    println!("prepare_cpu_ms={:.3}", preparation.as_secs_f64() * 1_000.0);
    Ok(())
}

fn benchmark_ui_90_10(renderer: &mut WgpuRenderer, gate: bool) -> Result<(), Box<dyn Error>> {
    let construction_started = Instant::now();
    let static_scene = build_screen_scene(STATIC_COMMANDS, 0)?;
    let prepared = renderer.prepare_screen_scene(&static_scene)?;
    let initial_construction = construction_started.elapsed();
    measure_frames(
        "ui_90_10",
        renderer,
        initial_construction,
        gate,
        |renderer, frame_index| {
            let streaming = build_screen_scene(STREAMING_COMMANDS, frame_index)?;
            let mut frame = renderer.begin_frame(Color::rgb8(9, 12, 18), FrameBudget::default())?;
            frame.draw_prepared_screen_scene(&prepared, FramePassOptions::new(0))?;
            frame.draw_screen_scene(&streaming, FramePassOptions::new(1))?;
            Ok(frame.present()?)
        },
    )
}

fn benchmark_four_viewports(renderer: &mut WgpuRenderer, gate: bool) -> Result<(), Box<dyn Error>> {
    let construction_started = Instant::now();
    let scenes = [
        build_world_scene(1_000, 0)?,
        build_world_scene(2_000, 17)?,
        build_world_scene(3_000, 43)?,
        build_world_scene(4_000, 89)?,
    ];
    let prepared = scenes
        .iter()
        .map(|scene| renderer.prepare_scene(scene))
        .collect::<Result<Vec<_>, _>>()?;
    let mut cameras = [
        Camera2d::new(Vec2::new(-8.0, 4.0), 3.4)?,
        Camera2d::new(Vec2::new(8.0, -4.0), 2.8)?,
        Camera2d::new(Vec2::new(-4.0, -7.0), 3.8)?,
        Camera2d::new(Vec2::new(6.0, 6.0), 3.1)?,
    ];
    for (camera, rotation) in cameras.iter_mut().zip([0.0, 0.18, -0.24, 0.35]) {
        camera.set_rotation(rotation)?;
    }
    let construction = construction_started.elapsed();
    let regions = [
        viewport_region(0.0, 0.0, 640.0, 360.0)?,
        viewport_region(640.0, 0.0, 640.0, 360.0)?,
        viewport_region(0.0, 360.0, 640.0, 360.0)?,
        viewport_region(640.0, 360.0, 640.0, 360.0)?,
    ];
    measure_frames(
        "four_viewports",
        renderer,
        construction,
        gate,
        |renderer, _| {
            let mut frame = renderer.begin_frame(Color::rgb8(9, 12, 18), FrameBudget::default())?;
            for (index, ((scene, camera), region)) in prepared
                .iter()
                .zip(cameras.iter().copied())
                .zip(regions.iter().copied())
                .enumerate()
            {
                frame.draw_prepared_scene(
                    scene,
                    camera,
                    FramePassOptions::new(index as i32).with_viewport(region),
                )?;
            }
            Ok(frame.present()?)
        },
    )
}

fn benchmark_image_atlas(renderer: &mut WgpuRenderer, gate: bool) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let pixels = vec![255; 64 * 64 * 4];
    let image =
        renderer.create_image_rgba8(64, 64, pixels, ImageBudget::new(64, 64, 64 * 64 * 4)?)?;
    let source = ImageTexelRect::new(0, 0, 8, 8)?;
    let mut sprites = Vec::with_capacity(512);
    for index in 0..512 {
        sprites.push(ImageSprite2d::new(
            source,
            viewport_region(
                (index % 32) as f32 * 20.0,
                (index / 32) as f32 * 20.0,
                16.0,
                16.0,
            )?,
            Color::WHITE,
        )?);
    }
    let batch =
        renderer.create_image_batch(&image, sprites, ImageBatchBudget::new(512, 512 * 128)?)?;
    measure_frames(
        "image_atlas",
        renderer,
        started.elapsed(),
        gate,
        |renderer, _| {
            let mut frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
            frame.draw_image_batch(
                &image,
                &batch,
                ImageSampling::Nearest,
                FramePassOptions::new(0),
            )?;
            Ok(frame.present()?)
        },
    )
}

fn benchmark_scientific_text(
    renderer: &mut WgpuRenderer,
    gate: bool,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let glyph_id = GlyphId::new('μ' as u32);
    let source = ImageTexelRect::new(0, 0, 8, 8)?;
    let atlas = renderer.create_glyph_atlas(
        8,
        8,
        vec![255; 8 * 8 * 4],
        vec![GlyphAtlasEntry::new(glyph_id, source)],
        GlyphAtlasBudget::new(ImageBudget::new(8, 8, 8 * 8 * 4)?, 1, 128)?,
    )?;
    let mut glyphs = Vec::with_capacity(1_024);
    for index in 0..1_024 {
        glyphs.push(PositionedGlyph2d::new(
            glyph_id,
            viewport_region(
                (index % 64) as f32 * 18.0,
                (index / 64) as f32 * 24.0,
                14.0,
                20.0,
            )?,
            Color::rgb8(215, 230, 245),
        )?);
    }
    let run =
        renderer.create_glyph_run(&atlas, glyphs, GlyphRunBudget::new(1_024, 1_024 * 256)?)?;
    measure_frames(
        "scientific_text",
        renderer,
        started.elapsed(),
        gate,
        |renderer, _| {
            let mut frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
            frame.draw_glyph_run(
                &atlas,
                &run,
                ImageSampling::Nearest,
                FramePassOptions::new(0),
            )?;
            Ok(frame.present()?)
        },
    )
}

fn benchmark_dpi_reconfigure(
    renderer: &mut WgpuRenderer,
    width: u32,
    height: u32,
    gate: bool,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let scene = build_screen_scene(2_500, 0)?;
    let prepared = renderer.prepare_screen_scene(&scene)?;
    let scales = [1.0, 1.25, 1.5, 2.0, 3.0];
    measure_frames(
        "dpi_reconfigure",
        renderer,
        started.elapsed(),
        gate,
        |renderer, frame| {
            renderer.resize_with_scale_factor(width, height, scales[frame % scales.len()])?;
            let mut composed = renderer.begin_frame(scene.background(), FrameBudget::default())?;
            composed.draw_prepared_screen_scene(&prepared, FramePassOptions::new(0))?;
            Ok(composed.present()?)
        },
    )
}

fn viewport_region(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
) -> Result<LogicalViewportRegion, Box<dyn Error>> {
    Ok(LogicalViewportRegion::new(
        LogicalScreenPosition::new(x, y),
        LogicalViewport::new(width, height)?,
    )?)
}

fn measure_frames(
    name: &str,
    renderer: &mut WgpuRenderer,
    construction: std::time::Duration,
    gate: bool,
    mut render: impl FnMut(&mut WgpuRenderer, usize) -> Result<FrameReport, Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    for frame in 0..WARMUP_FRAMES {
        let _ = render(renderer, frame)?;
    }
    renderer.wait_for_gpu_idle()?;

    let wall_started = Instant::now();
    let mut renderer_work_samples = Vec::with_capacity(MEASURED_FRAMES);
    let mut acquire_samples = Vec::with_capacity(MEASURED_FRAMES);
    let mut last = None;
    for frame in 0..MEASURED_FRAMES {
        let report = render(renderer, frame + WARMUP_FRAMES)?;
        let metrics = report.metrics();
        renderer_work_samples.push(
            metrics
                .total_cpu()
                .saturating_sub(metrics.surface_acquire())
                .as_secs_f64()
                * 1_000.0,
        );
        acquire_samples.push(metrics.surface_acquire().as_secs_f64() * 1_000.0);
        last = Some(report);
    }
    renderer.wait_for_gpu_idle()?;
    let wall = wall_started.elapsed();
    renderer_work_samples.sort_by(f64::total_cmp);
    acquire_samples.sort_by(f64::total_cmp);
    let percentile = |samples: &[f64], fraction: f64| {
        let index = ((samples.len() - 1) as f64 * fraction).round() as usize;
        samples[index]
    };
    let report = last.expect("measured frame count is non-zero");
    let statistics = report.statistics();
    validate_fixture_contract(name, statistics)?;
    let sources = statistics.source_counts();
    let wall_fps = MEASURED_FRAMES as f64 / wall.as_secs_f64();
    let work_p95 = percentile(&renderer_work_samples, 0.95);
    let acquire_p95 = percentile(&acquire_samples, 0.95);
    println!(
        "fixture={name} adapter={:?} backend={} driver={:?} driver_info={:?} present_mode={} measured_frames={} wall_fps={:.1} renderer_work_excluding_acquire_ms[p50={:.3},p95={:.3},p99={:.3}] surface_acquire_ms[p50={:.3},p95={:.3},p99={:.3}] construction_ms={:.3}",
        renderer.adapter_name(),
        renderer.adapter_backend(),
        renderer.adapter_driver(),
        renderer.adapter_driver_info(),
        renderer.surface_present_mode(),
        MEASURED_FRAMES,
        wall_fps,
        percentile(&renderer_work_samples, 0.50),
        percentile(&renderer_work_samples, 0.95),
        percentile(&renderer_work_samples, 0.99),
        percentile(&acquire_samples, 0.50),
        percentile(&acquire_samples, 0.95),
        percentile(&acquire_samples, 0.99),
        construction.as_secs_f64() * 1_000.0,
    );
    println!(
        "passes={} commands={} vertices={} streaming_vertices={} reused_vertices={} upload_bytes={} streaming_upload_bytes={} retained_cpu_bytes={} retained_buffer_bytes={} texture_bytes={} draw_calls={} sources[streaming={},prepared={},images={},glyphs={}]",
        statistics.pass_count(),
        statistics.command_count(),
        statistics.vertex_count(),
        statistics.streaming_vertex_count(),
        statistics.reused_vertex_count(),
        statistics.upload_bytes(),
        statistics.streaming_upload_bytes(),
        statistics.retained_cpu_bytes(),
        statistics.retained_buffer_bytes(),
        statistics.texture_bytes(),
        statistics.draw_calls(),
        sources.streaming_scenes(),
        sources.prepared_scenes(),
        sources.images(),
        sources.glyph_runs(),
    );
    if gate {
        let thresholds = gate_thresholds(name)
            .ok_or_else(|| format!("{name} does not define release-gate thresholds"))?;
        if wall_fps < thresholds.minimum_fps {
            return Err(format!(
                "{name} wall throughput {wall_fps:.1} FPS is below the {:.1} FPS gate",
                thresholds.minimum_fps,
            )
            .into());
        }
        if work_p95 > thresholds.maximum_renderer_work_p95_ms {
            return Err(format!(
                "{name} renderer work p95 {work_p95:.3} ms exceeds {:.3} ms",
                thresholds.maximum_renderer_work_p95_ms,
            )
            .into());
        }
        if acquire_p95 > thresholds.maximum_surface_acquire_p95_ms {
            return Err(format!(
                "{name} surface acquire p95 {acquire_p95:.3} ms exceeds {:.3} ms",
                thresholds.maximum_surface_acquire_p95_ms,
            )
            .into());
        }
        println!(
            "gate=passed min_fps={:.1} max_renderer_work_p95_ms={:.3} max_surface_acquire_p95_ms={:.3}",
            thresholds.minimum_fps,
            thresholds.maximum_renderer_work_p95_ms,
            thresholds.maximum_surface_acquire_p95_ms,
        );
    }
    Ok(())
}

fn validate_fixture_contract(
    name: &str,
    statistics: sim_engine::FrameStatistics,
) -> Result<(), Box<dyn Error>> {
    let sources = statistics.source_counts();
    let valid = match name {
        "ui_static_10k" => {
            statistics.command_count() == 10_000
                && sources.prepared_scenes() == 1
                && sources.streaming_scenes() == 0
        }
        "ui_90_10" => {
            statistics.command_count() == 10_000
                && sources.prepared_scenes() == 1
                && sources.streaming_scenes() == 1
        }
        "four_viewports" => {
            statistics.command_count() == 10_000
                && sources.prepared_scenes() == 4
                && sources.streaming_scenes() == 0
                && statistics.retained_cpu_bytes() > 0
                && statistics.retained_buffer_bytes() > 0
        }
        "image_atlas" => sources.images() == 1 && statistics.command_count() == 1,
        "scientific_text" => sources.glyph_runs() == 1 && statistics.command_count() == 1,
        "dpi_reconfigure" => statistics.command_count() == 2_500 && sources.prepared_scenes() == 1,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(format!(
            "{name} no longer matches its deterministic source/count contract: {statistics:?}"
        )
        .into())
    }
}
