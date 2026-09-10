//! Paired cache-disabled/enabled desktop composition benchmark.
//! `--frames N` sets measured presented frames per case; `--quick` uses one label.
//! CPU allocation counts include wgpu work on this event-loop thread, not GPU
//! execution or worker-thread allocations. Surface acquire is reported separately.
//! Cache off/on controls composition storage/bindings, not source-owned proofs.
//! Use --case/--labels/--cache for focused comparisons and --trials for repeated
//! measurements of identical content; every trial is printed independently.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use sim_engine::{
    Camera2d, Color, FrameBudget, FrameCacheBudget, FramePassOptions, GlyphAtlas2d,
    GlyphAtlasBudget, GlyphAtlasEntry, GlyphId, GlyphRun2d, GlyphRunBudget, Image2d,
    ImageBatchPlacement, ImageBudget, ImageSampling, ImageTexelRect, LogicalScreenPosition,
    LogicalScreenVector, LogicalViewport, LogicalViewportRegion, PositionedGlyph2d, RenderStatus,
    RendererPresentMode, Scene, ScreenScene, ShapeStyle, Vec2, WgpuRenderer, WgpuRendererOptions,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[path = "support/frame_cache_cases.rs"]
mod cases;
use cases::Selection;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn allocated(size: usize) {
    let _ = COUNTING.try_with(|active| {
        if active.get() {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get().saturating_add(1)));
            let _ = ALLOCATED_BYTES.try_with(|count| count.set(count.get().saturating_add(size)));
        }
    });
}

// SAFETY: All allocations and deallocations are forwarded to System with the
// original pointer/layout. The thread-local counters do not allocate.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocated(layout.size());
        // SAFETY: Forward the allocator contract unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocated(layout.size());
        // SAFETY: Forward the allocator contract unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Forward the allocator contract unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocated(size);
        // SAFETY: Forward the allocator contract unchanged.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Content {
    World,
    WorldScreen,
    Mixed64,
    Labels,
    Recolored,
    Moved,
    Mixed,
}

impl Content {
    fn name(self) -> &'static str {
        match self {
            Self::World => "world",
            Self::WorldScreen => "world_screen",
            Self::Mixed64 => "mixed64_images128rects",
            Self::Labels => "unchanged",
            Self::Recolored => "recolored",
            Self::Moved => "moved",
            Self::Mixed => "mixed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Case {
    content: Content,
    labels: usize,
    cached: bool,
}

fn expected_counts(case: Case) -> (usize, usize, usize, usize, usize) {
    match case.content {
        Content::World => (1, 0, 0, 100, 1),
        Content::WorldScreen => (2, 0, 0, 356, 2),
        Content::Mixed64 => (65, 64, 0, 292, 129),
        Content::Labels | Content::Recolored | Content::Moved => {
            (0, 0, case.labels, case.labels, case.labels)
        }
        Content::Mixed => (
            1 + case.labels,
            case.labels,
            case.labels,
            100 + 3 * case.labels,
            1 + 3 * case.labels,
        ),
    }
}

fn p95(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples
        .get((samples.len() * 95).div_ceil(100).saturating_sub(1))
        .copied()
        .unwrap_or_default()
}

#[derive(Default)]
struct Samples {
    draws: usize,
    attempts: usize,
    allocations: usize,
    allocated_bytes: usize,
    buffers: usize,
    groups: usize,
    uploads: usize,
    elapsed: Duration,
    acquire: Duration,
    tessellation: Duration,
    upload: Duration,
    camera_upload: Duration,
    encode: Duration,
    outside_report: Duration,
    measured_started: Option<Instant>,
    work_times: Vec<Duration>,
    acquire_times: Vec<Duration>,
}

impl Samples {
    fn new(frames: usize) -> Self {
        Self {
            work_times: Vec::with_capacity(frames),
            acquire_times: Vec::with_capacity(frames),
            ..Self::default()
        }
    }
}

struct Benchmark {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    atlas: GlyphAtlas2d,
    runs: Vec<GlyphRun2d>,
    image: Image2d,
    world: Scene,
    panel: ScreenScene,
    hud: ScreenScene,
    mixed_panels: Vec<ScreenScene>,
    cases: Vec<Case>,
    case_index: usize,
    samples: Samples,
    measured_frames: usize,
    trials: usize,
}

impl Benchmark {
    fn new(events: &ActiveEventLoop, measured_frames: usize, selection: Selection) -> Result<Self> {
        let cases = selection.cases()?;
        let window = Arc::new(
            events.create_window(
                Window::default_attributes()
                    .with_title("Sim;Engine frame cache benchmark")
                    .with_inner_size(LogicalSize::new(1280.0, 720.0)),
            )?,
        );
        let size = window.inner_size();
        let options =
            WgpuRendererOptions::new(RendererPresentMode::NoVsync, window.scale_factor())?;
        let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width,
            size.height,
            options,
        ))?;
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
        println!(
            "frame_cache_benchmark backend={} adapter={:?} pci_bus_id={} vendor={} device={} driver={:?} driver_info={:?} format={:?} msaa={} present={:?} scale={} physical_width={} physical_height={} allocation_scope=event_loop_thread baseline=cache_disabled gpu_timestamps=false",
            renderer.adapter_backend(),
            renderer.adapter_name(),
            renderer.adapter_pci_bus_id(),
            renderer.adapter_vendor_id(),
            renderer.adapter_device_id(),
            renderer.adapter_driver(),
            renderer.adapter_driver_info(),
            renderer.surface_format(),
            renderer.surface_sample_count(),
            renderer.surface_present_mode(),
            renderer.scale_factor(),
            size.width,
            size.height,
        );
        let atlas = renderer.create_glyph_atlas(
            1,
            1,
            vec![255; 4],
            vec![GlyphAtlasEntry::new(
                GlyphId::new(1),
                ImageTexelRect::new(0, 0, 1, 1)?,
            )],
            GlyphAtlasBudget::default(),
        )?;
        let glyphs = (0..32)
            .map(|index| {
                PositionedGlyph2d::new(
                    GlyphId::new(1),
                    LogicalViewportRegion::new(
                        LogicalScreenPosition::new(index as f32 * 3.0, 0.0),
                        LogicalViewport::new(2.0, 6.0).unwrap(),
                    )
                    .unwrap(),
                    Color::WHITE,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut runs = Vec::new();
        for _ in 0..if selection.quick { 1 } else { 1000 } {
            runs.push(renderer.create_glyph_run(
                &atlas,
                glyphs.clone(),
                GlyphRunBudget::default(),
            )?);
        }
        let image = renderer.create_image_rgba8(1, 1, vec![255; 4], ImageBudget::default())?;
        let mut world = Scene::new(Color::BLACK)?;
        for index in 0..100 {
            world.try_circle(
                Vec2::new((index % 10) as f32 * 12.0, (index / 10) as f32 * 12.0),
                3.0,
                ShapeStyle::filled(Color::rgb(0.2, 0.3, 0.4)),
            )?;
        }
        let mut panel = ScreenScene::new(Color::BLACK)?;
        panel.try_square_rect(
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalScreenVector::new(24.0, 12.0),
            ShapeStyle::filled(Color::rgb(0.0, 0.1, 0.2).with_alpha(0.5)),
        )?;
        let hud = ScreenScene::new(Color::BLACK)?;
        let mut mixed_panels = Vec::new();
        for index in 0..64 {
            let mut scene = ScreenScene::new(Color::BLACK)?;
            let x = (index % 8) as f32 * 140.0;
            let y = (index / 8) as f32 * 80.0;
            for offset in [0.0, 14.0] {
                scene.try_square_rect(
                    LogicalScreenPosition::new(x, y + offset),
                    LogicalScreenVector::new(104.0, 12.0),
                    ShapeStyle::filled(Color::rgb(0.0, 0.1, 0.2).with_alpha(0.5)),
                )?;
            }
            mixed_panels.push(scene);
        }
        renderer.set_frame_cache_budget(cases[0].cache_budget());
        Ok(Self {
            window,
            renderer,
            atlas,
            runs,
            image,
            world,
            panel,
            hud,
            mixed_panels,
            cases,
            case_index: 0,
            samples: Samples::new(measured_frames),
            measured_frames,
            trials: selection.trials,
        })
    }

    fn redraw(&mut self) -> Result<bool> {
        const WARMUP: usize = 20;
        let case = self.cases[self.case_index];
        ALLOCATIONS.set(0);
        ALLOCATED_BYTES.set(0);
        COUNTING.set(true);
        let started = Instant::now();
        let result = self.draw(case);
        let elapsed = started.elapsed();
        COUNTING.set(false);
        let allocations = ALLOCATIONS.get();
        let allocated_bytes = ALLOCATED_BYTES.get();
        let report = result?;
        self.samples.attempts += 1;
        if report.status() != RenderStatus::Drawn || report.surface_present_count() != 1 {
            if self.samples.attempts > (WARMUP + self.measured_frames) * 4 {
                return Err("benchmark cannot collect the requested presented frames".into());
            }
            return Ok(false);
        }
        self.samples.draws += 1;
        let sources = report.statistics().source_counts();
        let observed = (
            sources.streaming_scenes(),
            sources.images(),
            sources.glyph_runs(),
            report.statistics().command_count(),
            report.statistics().pass_count(),
        );
        if observed != expected_counts(case) {
            return Err(format!(
                "case {} source/command counts {observed:?} != {:?}",
                case.content.name(),
                expected_counts(case)
            )
            .into());
        }
        let cache = self.renderer.frame_cache_statistics();
        if self.samples.draws > WARMUP {
            self.samples.measured_started.get_or_insert(started);
            if case.cached && (cache.created_buffers() != 0 || cache.created_bind_groups() != 0) {
                return Err("warmed composition created a new uniform buffer or bind group".into());
            }
            if case.cached
                && matches!(
                    case.content,
                    Content::World
                        | Content::WorldScreen
                        | Content::Mixed64
                        | Content::Labels
                        | Content::Mixed
                )
                && cache.uploaded_uniform_bytes() != 0
            {
                return Err("unchanged warmed composition uploaded an unchanged uniform".into());
            }
            self.samples.allocations += allocations;
            self.samples.allocated_bytes += allocated_bytes;
            self.samples.buffers += cache.created_buffers();
            self.samples.groups += cache.created_bind_groups();
            self.samples.uploads += report.statistics().upload_bytes();
            self.samples.elapsed += elapsed;
            self.samples.acquire += report.metrics().surface_acquire();
            self.samples.tessellation += report.metrics().tessellation();
            self.samples.upload += report.metrics().upload();
            self.samples.camera_upload += report.metrics().camera_uniform_upload();
            self.samples.encode += report.metrics().encode_submit_present();
            self.samples.outside_report += elapsed.saturating_sub(report.metrics().total_cpu());
            self.samples
                .work_times
                .push(elapsed.saturating_sub(report.metrics().surface_acquire()));
            self.samples
                .acquire_times
                .push(report.metrics().surface_acquire());
        }
        if self.samples.draws < WARMUP + self.measured_frames {
            return Ok(false);
        }
        let count = self.measured_frames as f64;
        let measured_seconds = self
            .samples
            .measured_started
            .map(|started| started.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let work_p95 = p95(&mut self.samples.work_times).as_secs_f64() * 1000.0;
        let acquire_p95 = p95(&mut self.samples.acquire_times).as_secs_f64() * 1000.0;
        let trial = self.case_index / (self.cases.len() / self.trials) + 1;
        println!("measurement trial={trial} trials={}", self.trials);
        println!(
            "case={} labels={} glyphs_per_label=32 cache={} drawn={} presented_fps={:.2} main_thread_allocations_per_frame={:.2} allocated_bytes_per_frame={:.0} new_uniform_buffers_per_frame={:.2} new_bind_groups_per_frame={:.2} upload_bytes_per_frame={:.0} renderer_and_build_ms={:.3} acquire_ms={:.3} cache_cpu_bytes={} cache_uniform_bytes={} cache_texture_bytes={} cache_bindings={} cache_peak_cpu_bytes={} cache_peak_uniform_bytes={} source_scenes={} source_images={} source_glyph_runs={} commands={} passes={} renderer_and_build_p95_ms={:.3} acquire_p95_ms={:.3}",
            case.content.name(),
            case.labels,
            case.cached,
            self.measured_frames,
            count / measured_seconds,
            self.samples.allocations as f64 / count,
            self.samples.allocated_bytes as f64 / count,
            self.samples.buffers as f64 / count,
            self.samples.groups as f64 / count,
            self.samples.uploads as f64 / count,
            (self.samples.elapsed.saturating_sub(self.samples.acquire)).as_secs_f64() * 1000.0
                / count,
            self.samples.acquire.as_secs_f64() * 1000.0 / count,
            cache.cpu_bytes(),
            cache.uniform_bytes(),
            cache.texture_bytes(),
            cache.binding_count(),
            cache.peak_cpu_bytes(),
            cache.peak_uniform_bytes(),
            report.statistics().source_counts().streaming_scenes(),
            report.statistics().source_counts().images(),
            report.statistics().source_counts().glyph_runs(),
            report.statistics().command_count(),
            report.statistics().pass_count(),
            work_p95,
            acquire_p95,
        );
        println!(
            "stages case={} labels={} cache={} tessellation_and_preflight_ms={:.3} upload_and_image_bindings_ms={:.3} camera_bindings_ms={:.3} encode_submit_present_ms={:.3} outside_report_ms={:.3}",
            case.content.name(),
            case.labels,
            case.cached,
            self.samples.tessellation.as_secs_f64() * 1000.0 / count,
            self.samples.upload.as_secs_f64() * 1000.0 / count,
            self.samples.camera_upload.as_secs_f64() * 1000.0 / count,
            self.samples.encode.as_secs_f64() * 1000.0 / count,
            self.samples.outside_report.as_secs_f64() * 1000.0 / count,
        );
        self.samples = Samples::new(self.measured_frames);
        self.case_index += 1;
        if self.case_index == self.cases.len() {
            return Ok(true);
        }
        self.renderer
            .set_frame_cache_budget(self.cases[self.case_index].cache_budget());
        Ok(false)
    }

    fn draw(&mut self, case: Case) -> Result<sim_engine::FrameReport> {
        if matches!(case.content, Content::WorldScreen) {
            self.hud.clear();
            let phase = self.samples.draws as f32;
            for index in 0..256 {
                self.hud.try_square_rect(
                    LogicalScreenPosition::new(
                        (index % 16) as f32 * 64.0 + phase.sin() * 2.0,
                        (index / 16) as f32 * 32.0,
                    ),
                    LogicalScreenVector::new(48.0, 24.0),
                    ShapeStyle::filled(Color::rgb(0.1, 0.3 + phase.sin() * 0.1, 0.6)),
                )?;
            }
        }
        let mut frame = self.renderer.begin_frame(
            Color::BLACK,
            FrameBudget::new(
                4096,
                100_000,
                4_000_000,
                256 * 1024 * 1024,
                512 * 1024 * 1024,
                65536,
            ),
        )?;
        if matches!(
            case.content,
            Content::World | Content::WorldScreen | Content::Mixed64 | Content::Mixed
        ) {
            frame.draw_scene(&self.world, Camera2d::default(), FramePassOptions::new(0))?;
        }
        if matches!(case.content, Content::WorldScreen) {
            frame.draw_screen_scene(&self.hud, FramePassOptions::new(1))?;
        }
        if matches!(case.content, Content::Mixed64) {
            for (index, panel) in self.mixed_panels.iter().enumerate() {
                let order = 1 + index as i32 * 2;
                frame.draw_screen_scene(panel, FramePassOptions::new(order))?;
                frame.draw_image(
                    &self.image,
                    None,
                    Color::WHITE,
                    ImageSampling::Nearest,
                    FramePassOptions::new(order + 1).with_viewport(LogicalViewportRegion::new(
                        LogicalScreenPosition::new(
                            (index % 8) as f32 * 140.0,
                            (index / 8) as f32 * 80.0 + 32.0,
                        ),
                        LogicalViewport::new(24.0, 24.0)?,
                    )?),
                )?;
            }
        }
        let tick = self.samples.draws as f32;
        for (index, run) in self.runs[..case.labels].iter().enumerate() {
            let x = (index % 10) as f32 * 112.0
                + if matches!(case.content, Content::Moved) {
                    tick.sin() * 8.0
                } else {
                    0.0
                };
            let y = (index / 10) as f32 * 7.0;
            let tint = if matches!(case.content, Content::Recolored) {
                Color::rgb(0.5 + tick.sin() * 0.5, 0.5, 1.0)
            } else {
                Color::WHITE
            };
            let order = 1 + index as i32 * 3;
            if matches!(case.content, Content::Mixed) {
                frame.draw_screen_scene(&self.panel, FramePassOptions::new(order))?;
                frame.draw_image(
                    &self.image,
                    Some(self.image.full_rect()),
                    Color::WHITE,
                    ImageSampling::Nearest,
                    FramePassOptions::new(order + 1).with_viewport(LogicalViewportRegion::new(
                        LogicalScreenPosition::new(0.0, 0.0),
                        LogicalViewport::new(12.0, 12.0)?,
                    )?),
                )?;
            }
            frame.draw_glyph_run_placed(
                &self.atlas,
                run,
                ImageBatchPlacement::new(LogicalScreenVector::new(x, y), tint)?,
                ImageSampling::Nearest,
                FramePassOptions::new(order + 2),
            )?;
        }
        Ok(frame.present()?)
    }
}

struct App {
    benchmark: Option<Benchmark>,
    frames: usize,
    selection: Selection,
    error: Option<String>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, events: &ActiveEventLoop) {
        if self.benchmark.is_some() {
            return;
        }
        match Benchmark::new(events, self.frames, self.selection) {
            Ok(benchmark) => {
                benchmark.window.request_redraw();
                self.benchmark = Some(benchmark);
            }
            Err(error) => {
                self.error = Some(error.to_string());
                events.exit();
            }
        }
    }
    fn window_event(&mut self, events: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(benchmark) = &mut self.benchmark else {
            return;
        };
        if benchmark.window.id() != id {
            return;
        }
        let result: Result<bool> = match event {
            WindowEvent::CloseRequested => Err("benchmark window closed before completion".into()),
            WindowEvent::Resized(size) => benchmark
                .renderer
                .resize_with_scale_factor(size.width, size.height, benchmark.window.scale_factor())
                .map(|_| false)
                .map_err(Into::into),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let size = benchmark.window.inner_size();
                benchmark
                    .renderer
                    .resize_with_scale_factor(size.width, size.height, scale_factor)
                    .map(|_| false)
                    .map_err(Into::into)
            }
            WindowEvent::RedrawRequested => benchmark.redraw(),
            _ => Ok(false),
        };
        match result {
            Ok(true) => events.exit(),
            Err(error) => {
                self.error = Some(error.to_string());
                events.exit();
            }
            _ => {}
        }
    }
    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(benchmark) = &self.benchmark {
            benchmark.window.request_redraw();
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let mut arguments = std::env::args().skip(1);
    let mut frames = 60;
    let mut selection = Selection::default();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--frames" => {
                frames = arguments
                    .next()
                    .ok_or("--frames requires a count")?
                    .parse()?;
            }
            "--quick" => selection.quick = true,
            "--case" => {
                selection.set_content(&arguments.next().ok_or("--case requires a name")?)?
            }
            "--labels" => {
                selection.labels = Some(
                    arguments
                        .next()
                        .ok_or("--labels requires a count")?
                        .parse()?,
                )
            }
            "--cache" => {
                selection.set_cache(&arguments.next().ok_or("--cache requires on/off/both")?)?
            }
            "--trials" => {
                selection.trials = arguments
                    .next()
                    .ok_or("--trials requires a count")?
                    .parse()?
            }
            "--help" => {
                println!(
                    "frame_cache_benchmark [--frames 1..10000] [--quick] [--case world|world_screen|mixed64_images128rects|unchanged|recolored|moved|mixed] [--labels 0|1|100|1000] [--cache on|off|both] [--trials 1..100]\nEach trial warms up 20 presents; trial order alternates. Defaults retain all 30 cases. This is diagnostic, not a release gate."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {argument}").into()),
        }
    }
    if !(1..=10000).contains(&frames) {
        return Err("--frames must be in 1..=10000".into());
    }
    selection.cases()?;
    let mut app = App {
        benchmark: None,
        frames,
        selection,
        error: None,
    };
    let events = EventLoop::new()?;
    events.set_control_flow(ControlFlow::Poll);
    events.run_app(&mut app)?;
    if let Some(error) = app.error {
        return Err(error.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_controls_keep_requested_scene_and_image_command_counts() {
        assert_eq!(
            expected_counts(Case {
                content: Content::WorldScreen,
                labels: 0,
                cached: true
            }),
            (2, 0, 0, 356, 2)
        );
        assert_eq!(
            expected_counts(Case {
                content: Content::Mixed64,
                labels: 0,
                cached: true
            }),
            (65, 64, 0, 292, 129)
        );
        assert_eq!(
            expected_counts(Case {
                content: Content::Labels,
                labels: 1000,
                cached: true
            }),
            (0, 0, 1000, 1000, 1000)
        );
    }

    #[test]
    fn percentile_uses_nearest_rank_without_dropping_the_last_sample() {
        let mut times = (1..=20)
            .rev()
            .map(Duration::from_millis)
            .collect::<Vec<_>>();
        assert_eq!(p95(&mut times), Duration::from_millis(19));
        assert_eq!(
            p95(&mut [Duration::from_millis(7)]),
            Duration::from_millis(7)
        );
    }
}
