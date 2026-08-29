//! Repeatable surface benchmarks for the named Sim;X rendering workloads.
//!
//! The executable owns a temporary window only to obtain a real presentation
//! surface. It prints renderer CPU percentiles plus deterministic work counters;
//! it does not claim universal GPU frame-time thresholds.

use std::{error::Error, sync::Arc, time::Instant};

use sim_engine::{
    Color, FrameBudget, FramePassOptions, FrameReport, GlyphAtlasBudget, GlyphAtlasEntry, GlyphId,
    GlyphRunBudget, ImageBatchBudget, ImageBudget, ImageSampling, ImageSprite2d, ImageTexelRect,
    Layer, LogicalPixels, LogicalScreenPosition, LogicalScreenVector, LogicalViewport,
    LogicalViewportRegion, PositionedGlyph2d, SceneBudget, ScreenScene, ShapeStyle, WgpuRenderer,
    WgpuRendererOptions,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const WARMUP_FRAMES: usize = 20;
const MEASURED_FRAMES: usize = 120;
const STATIC_COMMANDS: usize = 9_000;
const STREAMING_COMMANDS: usize = 1_000;

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let fixture = parse_fixture()?;
    let mut application = BenchmarkApplication::new(fixture);
    EventLoop::new()?.run_app(&mut application)?;
    if let Some(failure) = application.failure {
        return Err(failure.into());
    }
    Ok(())
}

fn parse_fixture() -> Result<String, Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let mut fixture = "ui_90_10".to_owned();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--fixture" => {
                fixture = arguments.next().ok_or("--fixture requires a value")?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: rendering_benchmark_suite [--fixture ui_static_10k|ui_90_10|four_viewports|image_atlas|scientific_text|hidpi_resize]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    Ok(fixture)
}

struct BenchmarkApplication {
    fixture: String,
    started: bool,
    failure: Option<String>,
}

impl BenchmarkApplication {
    fn new(fixture: String) -> Self {
        Self {
            fixture,
            started: false,
            failure: None,
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
                window,
                size.width.max(1),
                size.height.max(1),
                options,
            ))?;
            match self.fixture.as_str() {
                "ui_static_10k" => benchmark_static_ui(&mut renderer),
                "ui_90_10" => benchmark_ui_90_10(&mut renderer),
                "four_viewports" => benchmark_four_viewports(&mut renderer),
                "image_atlas" => benchmark_image_atlas(&mut renderer),
                "scientific_text" => benchmark_scientific_text(&mut renderer),
                "hidpi_resize" => benchmark_hidpi_resize(&mut renderer, size.width, size.height),
                fixture => Err(format!("unknown fixture: {fixture}").into()),
            }
        })();
        if let Err(error) = result {
            self.failure = Some(error.to_string());
        }
        event_loop.exit();
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        _event: winit::event::WindowEvent,
    ) {
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

fn benchmark_static_ui(renderer: &mut WgpuRenderer) -> Result<(), Box<dyn Error>> {
    let construction_started = Instant::now();
    let scene = build_screen_scene(STATIC_COMMANDS + STREAMING_COMMANDS, 0)?;
    let construction = construction_started.elapsed();
    let preparation_started = Instant::now();
    let prepared = renderer.prepare_screen_scene(&scene)?;
    let preparation = preparation_started.elapsed();
    measure_frames("ui_static_10k", renderer, construction, |renderer, _| {
        let mut frame = renderer.begin_frame(scene.background(), FrameBudget::default())?;
        frame.draw_prepared_screen_scene(&prepared, FramePassOptions::new(0))?;
        Ok(frame.present()?)
    })?;
    println!("prepare_cpu_ms={:.3}", preparation.as_secs_f64() * 1_000.0);
    Ok(())
}

fn benchmark_ui_90_10(renderer: &mut WgpuRenderer) -> Result<(), Box<dyn Error>> {
    let construction_started = Instant::now();
    let static_scene = build_screen_scene(STATIC_COMMANDS, 0)?;
    let prepared = renderer.prepare_screen_scene(&static_scene)?;
    let initial_construction = construction_started.elapsed();
    measure_frames(
        "ui_90_10",
        renderer,
        initial_construction,
        |renderer, frame_index| {
            let streaming = build_screen_scene(STREAMING_COMMANDS, frame_index)?;
            let mut frame = renderer.begin_frame(Color::rgb8(9, 12, 18), FrameBudget::default())?;
            frame.draw_prepared_screen_scene(&prepared, FramePassOptions::new(0))?;
            frame.draw_screen_scene(&streaming, FramePassOptions::new(1))?;
            Ok(frame.present()?)
        },
    )
}

fn benchmark_four_viewports(renderer: &mut WgpuRenderer) -> Result<(), Box<dyn Error>> {
    let construction_started = Instant::now();
    let scene = build_screen_scene(2_500, 0)?;
    let prepared = renderer.prepare_screen_scene(&scene)?;
    let construction = construction_started.elapsed();
    let regions = [
        viewport_region(0.0, 0.0, 640.0, 360.0)?,
        viewport_region(640.0, 0.0, 640.0, 360.0)?,
        viewport_region(0.0, 360.0, 640.0, 360.0)?,
        viewport_region(640.0, 360.0, 640.0, 360.0)?,
    ];
    measure_frames("four_viewports", renderer, construction, |renderer, _| {
        let mut frame = renderer.begin_frame(scene.background(), FrameBudget::default())?;
        for (index, region) in regions.iter().copied().enumerate() {
            frame.draw_prepared_screen_scene(
                &prepared,
                FramePassOptions::new(index as i32).with_viewport(region),
            )?;
        }
        Ok(frame.present()?)
    })
}

fn benchmark_image_atlas(renderer: &mut WgpuRenderer) -> Result<(), Box<dyn Error>> {
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
    measure_frames("image_atlas", renderer, started.elapsed(), |renderer, _| {
        let mut frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
        frame.draw_image_batch(
            &image,
            &batch,
            ImageSampling::Nearest,
            FramePassOptions::new(0),
        )?;
        Ok(frame.present()?)
    })
}

fn benchmark_scientific_text(renderer: &mut WgpuRenderer) -> Result<(), Box<dyn Error>> {
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

fn benchmark_hidpi_resize(
    renderer: &mut WgpuRenderer,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let scene = build_screen_scene(2_500, 0)?;
    let prepared = renderer.prepare_screen_scene(&scene)?;
    let scales = [1.0, 1.25, 1.5, 2.0, 3.0];
    measure_frames(
        "hidpi_resize",
        renderer,
        started.elapsed(),
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
    mut render: impl FnMut(&mut WgpuRenderer, usize) -> Result<FrameReport, Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    for frame in 0..WARMUP_FRAMES {
        let _ = render(renderer, frame)?;
    }
    renderer.wait_for_gpu_idle()?;

    let wall_started = Instant::now();
    let mut cpu_samples = Vec::with_capacity(MEASURED_FRAMES);
    let mut last = None;
    for frame in 0..MEASURED_FRAMES {
        let report = render(renderer, frame + WARMUP_FRAMES)?;
        cpu_samples.push(report.metrics().total_cpu().as_secs_f64() * 1_000.0);
        last = Some(report);
    }
    renderer.wait_for_gpu_idle()?;
    let wall = wall_started.elapsed();
    cpu_samples.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let index = ((cpu_samples.len() - 1) as f64 * fraction).round() as usize;
        cpu_samples[index]
    };
    let report = last.expect("measured frame count is non-zero");
    let statistics = report.statistics();
    let sources = statistics.source_counts();
    println!(
        "fixture={name} backend={} measured_frames={} wall_fps={:.1} renderer_cpu_ms[p50={:.3},p95={:.3},p99={:.3}] construction_ms={:.3}",
        renderer.surface_present_mode(),
        MEASURED_FRAMES,
        MEASURED_FRAMES as f64 / wall.as_secs_f64(),
        percentile(0.50),
        percentile(0.95),
        percentile(0.99),
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
    Ok(())
}
