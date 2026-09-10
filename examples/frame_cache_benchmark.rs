//! Paired cache-disabled/enabled desktop composition benchmark.
//! `--frames N` sets measured presented frames per case; `--quick` uses one label.
//! CPU allocation counts include wgpu work on this event-loop thread, not GPU
//! execution or worker-thread allocations. Surface acquire is reported separately.

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

#[derive(Clone, Copy)]
enum Content {
    World,
    WorldScreen,
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
            Self::Labels => "unchanged",
            Self::Recolored => "recolored",
            Self::Moved => "moved",
            Self::Mixed => "mixed",
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    content: Content,
    labels: usize,
    cached: bool,
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
    measured_started: Option<Instant>,
}

struct Benchmark {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    atlas: GlyphAtlas2d,
    runs: Vec<GlyphRun2d>,
    image: Image2d,
    world: Scene,
    panel: ScreenScene,
    cases: Vec<Case>,
    case_index: usize,
    samples: Samples,
    measured_frames: usize,
}

impl Benchmark {
    fn new(events: &ActiveEventLoop, measured_frames: usize, quick: bool) -> Result<Self> {
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
            "frame_cache_benchmark backend={} adapter={:?} present={:?} scale={} allocation_scope=event_loop_thread baseline=cache_disabled",
            renderer.adapter_backend(),
            renderer.adapter_name(),
            renderer.surface_present_mode(),
            renderer.scale_factor()
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
        for _ in 0..if quick { 1 } else { 1000 } {
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
        let mut cases = Vec::new();
        for content in [Content::World, Content::WorldScreen] {
            for cached in [false, true] {
                cases.push(Case {
                    content,
                    labels: 0,
                    cached,
                });
            }
        }
        let counts: &[usize] = if quick { &[1] } else { &[1, 100, 1000] };
        for &labels in counts {
            for content in [
                Content::Labels,
                Content::Recolored,
                Content::Moved,
                Content::Mixed,
            ] {
                for cached in [false, true] {
                    cases.push(Case {
                        content,
                        labels,
                        cached,
                    });
                }
            }
        }
        renderer.set_frame_cache_budget(FrameCacheBudget::new(0, 0, 0, 0));
        Ok(Self {
            window,
            renderer,
            atlas,
            runs,
            image,
            world,
            panel,
            cases,
            case_index: 0,
            samples: Samples::default(),
            measured_frames,
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
        let cache = self.renderer.frame_cache_statistics();
        if self.samples.draws > WARMUP {
            self.samples.measured_started.get_or_insert(started);
            if case.cached && (cache.created_buffers() != 0 || cache.created_bind_groups() != 0) {
                return Err("warmed composition created a new uniform buffer or bind group".into());
            }
            if case.cached
                && matches!(
                    case.content,
                    Content::World | Content::WorldScreen | Content::Labels | Content::Mixed
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
        println!(
            "case={} labels={} glyphs_per_label=32 cache={} drawn={} presented_fps={:.2} main_thread_allocations_per_frame={:.2} allocated_bytes_per_frame={:.0} new_uniform_buffers_per_frame={:.2} new_bind_groups_per_frame={:.2} upload_bytes_per_frame={:.0} renderer_and_build_ms={:.3} acquire_ms={:.3} cache_cpu_bytes={} cache_uniform_bytes={} cache_texture_bytes={} cache_bindings={} cache_peak_cpu_bytes={} cache_peak_uniform_bytes={}",
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
            cache.peak_uniform_bytes()
        );
        self.samples = Samples::default();
        self.case_index += 1;
        if self.case_index == self.cases.len() {
            return Ok(true);
        }
        self.renderer
            .set_frame_cache_budget(if self.cases[self.case_index].cached {
                FrameCacheBudget::default()
            } else {
                FrameCacheBudget::new(0, 0, 0, 0)
            });
        Ok(false)
    }

    fn draw(&mut self, case: Case) -> Result<sim_engine::FrameReport> {
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
        frame.draw_scene(&self.world, Camera2d::default(), FramePassOptions::new(0))?;
        if matches!(case.content, Content::WorldScreen) {
            frame.draw_screen_scene(&self.panel, FramePassOptions::new(1))?;
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
    quick: bool,
    error: Option<String>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, events: &ActiveEventLoop) {
        if self.benchmark.is_some() {
            return;
        }
        match Benchmark::new(events, self.frames, self.quick) {
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
    let mut quick = false;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--frames" => {
                frames = arguments
                    .next()
                    .ok_or("--frames requires a count")?
                    .parse()?;
            }
            "--quick" => quick = true,
            "--help" => {
                println!("frame_cache_benchmark [--frames N] [--quick]");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {argument}").into()),
        }
    }
    if frames == 0 {
        return Err("--frames must be positive".into());
    }
    let mut app = App {
        benchmark: None,
        frames,
        quick,
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
