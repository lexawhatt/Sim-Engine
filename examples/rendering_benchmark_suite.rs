//! Repeatable surface benchmarks for the named Sim;X rendering workloads.
//!
//! The executable owns a temporary window only to obtain a real presentation
//! surface. It prints engine-side CPU and surface-acquire percentiles plus
//! deterministic work counters. `--gate` applies the project's Linux release
//! floor; it does not claim that raw timings transfer across unrelated GPUs.

use std::{
    error::Error,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use sim_engine::{
    Camera2d, Color, FrameBudget, FramePassOptions, FrameReport, GlyphAtlasBudget, GlyphAtlasEntry,
    GlyphId, GlyphRunBudget, ImageBatchBudget, ImageBudget, ImageSampling, ImageSprite2d,
    ImageTexelRect, Layer, LogicalPixels, LogicalScreenPosition, LogicalScreenVector,
    LogicalViewport, LogicalViewportRegion, PositionedGlyph2d, PreparedScreenScene, Rect,
    RenderStatus, RendererSurfacePresentMode, Scene, SceneBudget, ScreenScene, ShapeStyle, Vec2,
    WgpuRenderer, WgpuRendererOptions,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    monitor::MonitorHandle,
    window::{Window, WindowId},
};

const WARMUP_FRAMES: usize = 20;
const MEASURED_FRAMES: usize = 120;
const GATE_TRIALS: usize = 3;
const MAX_OUTPUT_CONFIRMATION_REDRAWS: usize = 120;
const MAX_SURFACE_RESTARTS: usize = 8;
const STATIC_COMMANDS: usize = 9_000;
const STREAMING_COMMANDS: usize = 1_000;

#[derive(Clone, Copy)]
struct GateThresholds {
    minimum_immediate_fps: f64,
    maximum_renderer_work_p95_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PerformanceGateKind {
    UncappedThroughput,
    RefreshSynchronizedWork,
}

#[derive(Debug, Clone, Copy)]
struct BenchmarkGateContext {
    enabled: bool,
    display_refresh_hz: Option<f64>,
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
        minimum_immediate_fps: 60.0,
        maximum_renderer_work_p95_ms,
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
    vsync: bool,
}

fn parse_configuration() -> Result<BenchmarkConfiguration, Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let mut fixture = "ui_90_10".to_owned();
    let mut gate = false;
    let mut vsync = false;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--fixture" => {
                fixture = arguments.next().ok_or("--fixture requires a value")?;
            }
            "--gate" => gate = true,
            "--vsync" => vsync = true,
            "--help" | "-h" => {
                println!(
                    "Usage: rendering_benchmark_suite [--fixture adapter_probe|ui_static_10k|ui_90_10|four_viewports|image_atlas|scientific_text|dpi_reconfigure|hidpi_transition] [--gate] [--vsync]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    Ok(BenchmarkConfiguration {
        fixture,
        gate,
        vsync,
    })
}

struct BenchmarkApplication {
    fixture: String,
    gate: bool,
    vsync: bool,
    started: bool,
    failure: Option<String>,
    benchmark: Option<BenchmarkRunState>,
    hidpi: Option<HidpiTransitionState>,
}

struct BenchmarkRunState {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    physical_width: u32,
    physical_height: u32,
    surface_generation: u64,
    surface_restarts: usize,
    confirmation: OutputConfirmationState,
    gate_context: Option<BenchmarkGateContext>,
    workload: BenchmarkWorkload,
    measurement: BenchmarkMeasurement,
}

type BenchmarkRender =
    Box<dyn FnMut(&mut WgpuRenderer, usize, u32, u32) -> Result<FrameReport, Box<dyn Error>>>;

struct BenchmarkWorkload {
    name: &'static str,
    construction: Duration,
    completion_note: Option<String>,
    render: BenchmarkRender,
}

#[derive(Debug, Clone, Copy)]
enum BenchmarkMeasurementPhase {
    Warmup {
        next_frame: usize,
    },
    Trial {
        trial: usize,
        next_frame: usize,
        started: Option<Instant>,
    },
    Complete,
}

struct BenchmarkMeasurement {
    phase: BenchmarkMeasurementPhase,
    renderer_work_samples: Vec<f64>,
    acquire_samples: Vec<f64>,
    trial_fps: Vec<f64>,
    last_report: Option<FrameReport>,
    discarded_measured_frames: usize,
}

impl BenchmarkMeasurement {
    fn new() -> Self {
        Self {
            phase: BenchmarkMeasurementPhase::Warmup { next_frame: 0 },
            renderer_work_samples: Vec::with_capacity(MEASURED_FRAMES * GATE_TRIALS),
            acquire_samples: Vec::with_capacity(MEASURED_FRAMES * GATE_TRIALS),
            trial_fps: Vec::with_capacity(GATE_TRIALS),
            last_report: None,
            discarded_measured_frames: 0,
        }
    }

    fn reset(&mut self) {
        self.discarded_measured_frames = self
            .discarded_measured_frames
            .saturating_add(self.renderer_work_samples.len());
        self.phase = BenchmarkMeasurementPhase::Warmup { next_frame: 0 };
        self.renderer_work_samples.clear();
        self.acquire_samples.clear();
        self.trial_fps.clear();
        self.last_report = None;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OutputSnapshot {
    monitor: MonitorHandle,
    refresh_millihertz: Option<u32>,
}

#[derive(Debug, Clone)]
enum OutputConfirmationState {
    Required,
    AwaitingMetadata {
        generation: u64,
        completed_presents: usize,
    },
    Confirmed {
        generation: u64,
        output: Option<OutputSnapshot>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkAdvance {
    Continue,
    Complete,
}

impl BenchmarkRunState {
    fn invalidate_surface(&mut self, gated: bool) -> Result<(), String> {
        self.surface_generation = self.surface_generation.saturating_add(1);
        self.surface_restarts = self.surface_restarts.saturating_add(1);
        if self.surface_restarts > MAX_SURFACE_RESTARTS {
            return Err(format!(
                "{} surface configuration changed more than {MAX_SURFACE_RESTARTS} times during measurement",
                self.workload.name,
            ));
        }
        self.measurement.reset();
        self.gate_context = if gated {
            None
        } else {
            Some(BenchmarkGateContext {
                enabled: false,
                display_refresh_hz: None,
            })
        };
        self.confirmation = if gated {
            OutputConfirmationState::Required
        } else {
            OutputConfirmationState::Confirmed {
                generation: self.surface_generation,
                output: None,
            }
        };
        Ok(())
    }
}

struct HidpiTransitionState {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    prepared: PreparedScreenScene,
    evidence: HidpiEvidenceTracker,
    auto_exit: bool,
    ready_announced: bool,
}

#[derive(Debug)]
struct HidpiEvidenceTracker {
    scale_events: usize,
    resize_events: usize,
    paired_transitions: usize,
    completed_transitions: usize,
    next_serial: u64,
    pending: Option<PendingHidpiTransition>,
    committed_scale_factor: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingHidpiTransition {
    serial: u64,
    resized: bool,
    scale_factor: f64,
    physical_width: u32,
    physical_height: u32,
}

impl HidpiEvidenceTracker {
    fn new(initial_scale_factor: f64) -> Self {
        Self {
            scale_events: 0,
            resize_events: 0,
            paired_transitions: 0,
            completed_transitions: 0,
            next_serial: 0,
            pending: None,
            committed_scale_factor: initial_scale_factor,
        }
    }

    fn observe_scale(&mut self, scale_factor: f64, physical_width: u32, physical_height: u32) {
        if scale_factor == self.committed_scale_factor {
            return;
        }
        self.scale_events = self.scale_events.saturating_add(1);
        self.next_serial = self.next_serial.saturating_add(1);
        self.pending = Some(PendingHidpiTransition {
            serial: self.next_serial,
            resized: false,
            scale_factor,
            physical_width,
            physical_height,
        });
    }

    fn observe_resize(&mut self, physical_width: u32, physical_height: u32) {
        self.resize_events = self.resize_events.saturating_add(1);
        let Some(pending) = self.pending.as_mut() else {
            return;
        };
        pending.physical_width = physical_width;
        pending.physical_height = physical_height;
        if !pending.resized {
            pending.resized = true;
            self.paired_transitions = self.paired_transitions.saturating_add(1);
        }
    }

    fn observe_successful_present(&mut self) -> Option<PendingHidpiTransition> {
        if !self.pending.is_some_and(|pending| pending.resized) {
            return None;
        }
        self.completed_transitions = self.completed_transitions.saturating_add(1);
        let completed = self.pending.take()?;
        self.committed_scale_factor = completed.scale_factor;
        Some(completed)
    }
}

impl BenchmarkApplication {
    fn new(configuration: BenchmarkConfiguration) -> Self {
        Self {
            fixture: configuration.fixture,
            gate: configuration.gate,
            vsync: configuration.vsync,
            started: false,
            failure: None,
            benchmark: None,
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
            let requested_present_mode = if self.vsync {
                sim_engine::RendererPresentMode::Vsync
            } else {
                sim_engine::RendererPresentMode::NoVsync
            };
            let options = WgpuRendererOptions::new(requested_present_mode, window.scale_factor())?;
            let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
                Arc::clone(&window),
                size.width.max(1),
                size.height.max(1),
                options,
            ))?;
            let require_adapter_identity =
                std::env::var("SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY").as_deref() == Ok("1");
            validate_renderer_adapter(&renderer, require_adapter_identity)
                .map_err(|error| -> Box<dyn Error> { error.into() })?;
            match self.fixture.as_str() {
                "adapter_probe" => write_surface_probe_evidence(&renderer),
                "ui_static_10k" | "ui_90_10" | "four_viewports" | "image_atlas"
                | "scientific_text" | "dpi_reconfigure" => {
                    let workload = prepare_benchmark_workload(&self.fixture, &mut renderer)?;
                    window.request_redraw();
                    self.benchmark = Some(BenchmarkRunState {
                        window,
                        renderer,
                        physical_width: size.width.max(1),
                        physical_height: size.height.max(1),
                        surface_generation: 0,
                        surface_restarts: 0,
                        confirmation: if self.gate {
                            OutputConfirmationState::Required
                        } else {
                            OutputConfirmationState::Confirmed {
                                generation: 0,
                                output: None,
                            }
                        },
                        gate_context: if self.gate {
                            None
                        } else {
                            Some(BenchmarkGateContext {
                                enabled: false,
                                display_refresh_hz: None,
                            })
                        },
                        workload,
                        measurement: BenchmarkMeasurement::new(),
                    });
                    Ok(())
                }
                "hidpi_transition" => {
                    let scene = build_screen_scene(2_500, 0)?;
                    let prepared = renderer.prepare_screen_scene(&scene)?;
                    println!(
                        "hidpi_transition: move this window to a monitor with a different scale factor, then press Esc"
                    );
                    let initial_scale_factor = window.scale_factor();
                    window.request_redraw();
                    self.hidpi = Some(HidpiTransitionState {
                        window,
                        renderer,
                        prepared,
                        evidence: HidpiEvidenceTracker::new(initial_scale_factor),
                        auto_exit: std::env::var("SIM_ENGINE_HIDPI_AUTO_EXIT")
                            .is_ok_and(|value| value == "1"),
                        ready_announced: false,
                    });
                    Ok(())
                }
                fixture => Err(format!("unknown fixture: {fixture}").into()),
            }
        })();
        if let Err(error) = result {
            self.failure = Some(error.to_string());
        }
        if (self.benchmark.is_none() && self.hidpi.is_none()) || self.failure.is_some() {
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self
            .benchmark
            .as_ref()
            .is_some_and(|state| state.window.id() == window_id)
        {
            match event {
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    let state = self.benchmark.as_mut().expect("benchmark state exists");
                    let size = state.window.inner_size();
                    state.physical_width = size.width.max(1);
                    state.physical_height = size.height.max(1);
                    if let Err(error) = state.renderer.resize_with_scale_factor(
                        state.physical_width,
                        state.physical_height,
                        scale_factor,
                    ) {
                        self.failure = Some(error.to_string());
                        event_loop.exit();
                    } else if let Err(error) = state.invalidate_surface(self.gate) {
                        self.failure = Some(error);
                        event_loop.exit();
                    } else {
                        state.window.request_redraw();
                    }
                }
                WindowEvent::Resized(size) => {
                    let state = self.benchmark.as_mut().expect("benchmark state exists");
                    state.physical_width = size.width.max(1);
                    state.physical_height = size.height.max(1);
                    if let Err(error) = state.renderer.resize_with_scale_factor(
                        state.physical_width,
                        state.physical_height,
                        state.window.scale_factor(),
                    ) {
                        self.failure = Some(error.to_string());
                        event_loop.exit();
                    } else if let Err(error) = state.invalidate_surface(self.gate) {
                        self.failure = Some(error);
                        event_loop.exit();
                    } else {
                        state.window.request_redraw();
                    }
                }
                WindowEvent::RedrawRequested => {
                    let state = self.benchmark.as_mut().expect("benchmark state exists");
                    match advance_benchmark(state, self.gate) {
                        Ok(BenchmarkAdvance::Continue) => state.window.request_redraw(),
                        Ok(BenchmarkAdvance::Complete) => event_loop.exit(),
                        Err(error) => {
                            self.failure = Some(error.to_string());
                            event_loop.exit();
                        }
                    }
                }
                WindowEvent::CloseRequested => {
                    self.failure = Some(
                        "benchmark window closed before the confirmed-output measurement"
                            .to_owned(),
                    );
                    event_loop.exit();
                }
                _ => {}
            }
            return;
        }
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
                let size = state.window.inner_size();
                state
                    .evidence
                    .observe_scale(scale_factor, size.width.max(1), size.height.max(1));
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
                state
                    .evidence
                    .observe_resize(size.width.max(1), size.height.max(1));
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
                let result = (|| -> Result<bool, Box<dyn Error>> {
                    let mut frame = state
                        .renderer
                        .begin_frame(Color::rgb8(9, 12, 18), FrameBudget::default())?;
                    frame.draw_prepared_screen_scene(&state.prepared, FramePassOptions::new(0))?;
                    Ok(matches!(
                        frame.present()?.status(),
                        sim_engine::RenderStatus::Drawn
                    ))
                })();
                (|| -> Result<(), String> {
                    let drawn = result.map_err(|error| error.to_string())?;
                    if drawn && state.evidence.pending.is_none() && !state.ready_announced {
                        if let Ok(path) = std::env::var("SIM_ENGINE_HIDPI_READY_PATH") {
                            std::fs::write(path, "ready\n").map_err(|error| error.to_string())?;
                        }
                        state.ready_announced = true;
                    }
                    if drawn && let Some(completed) = state.evidence.observe_successful_present() {
                        write_hidpi_evidence(state, completed)
                            .map_err(|error| error.to_string())?;
                        if state.auto_exit {
                            event_loop.exit();
                        }
                    }
                    Ok(())
                })()
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
        "hidpi_transition scale_events={} resize_events={} paired_transitions={} completed_transitions={} pending={}",
        state.evidence.scale_events,
        state.evidence.resize_events,
        state.evidence.paired_transitions,
        state.evidence.completed_transitions,
        state.evidence.pending.is_some()
    );
    validate_hidpi_evidence(&state.evidence)
}

fn validate_hidpi_evidence(evidence: &HidpiEvidenceTracker) -> Result<(), &'static str> {
    if evidence.scale_events == 0 {
        return Err("no real ScaleFactorChanged event was observed");
    }
    if evidence.paired_transitions == 0 {
        return Err("no Resized event followed the compositor scale transition");
    }
    if evidence.completed_transitions == 0 {
        return Err("the renderer did not present after the paired resize");
    }
    if evidence.pending.is_some() {
        return Err("a compositor scale transition remains incomplete");
    }
    Ok(())
}

fn validate_gate_context(
    enabled: bool,
    output_confirmed: bool,
    present_mode: RendererSurfacePresentMode,
    display_refresh_hz: Option<f64>,
) -> Result<(), &'static str> {
    if !enabled {
        return Ok(());
    }
    if !output_confirmed {
        return Err("gated surface measurement requires an unmeasured drawn frame first");
    }
    if present_mode.is_refresh_synchronized()
        && display_refresh_hz.is_none_or(|refresh_hz| !refresh_hz.is_finite() || refresh_hz <= 0.0)
    {
        return Err(
            "refresh-synchronized gated measurement requires a positive current-monitor refresh rate",
        );
    }
    Ok(())
}

fn current_output_snapshot(window: &Window) -> Option<OutputSnapshot> {
    window.current_monitor().map(|monitor| OutputSnapshot {
        refresh_millihertz: monitor.refresh_rate_millihertz(),
        monitor,
    })
}

fn output_refresh_hz(output: Option<&OutputSnapshot>) -> Option<f64> {
    confirmed_refresh_hz(output.and_then(|snapshot| snapshot.refresh_millihertz))
}

fn confirmed_refresh_hz(refresh_millihertz: Option<u32>) -> Option<f64> {
    refresh_millihertz
        .filter(|millihertz| *millihertz > 0)
        .map(|millihertz| f64::from(millihertz) / 1_000.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetadataConfirmationAction {
    Confirm,
    PresentAgain,
    Exhausted,
}

fn metadata_confirmation_action(
    present_mode: RendererSurfacePresentMode,
    display_refresh_hz: Option<f64>,
    completed_presents: usize,
) -> MetadataConfirmationAction {
    if !present_mode.is_refresh_synchronized() || display_refresh_hz.is_some() {
        MetadataConfirmationAction::Confirm
    } else if completed_presents < MAX_OUTPUT_CONFIRMATION_REDRAWS {
        MetadataConfirmationAction::PresentAgain
    } else {
        MetadataConfirmationAction::Exhausted
    }
}

fn confirmed_output_is_current<T: PartialEq>(
    confirmed_generation: u64,
    confirmed_output: Option<&T>,
    surface_generation: u64,
    observed_output: Option<&T>,
) -> bool {
    confirmed_generation == surface_generation && confirmed_output == observed_output
}

fn write_surface_probe_evidence(renderer: &WgpuRenderer) -> Result<(), Box<dyn Error>> {
    let path = required_environment("SIM_ENGINE_SURFACE_EVIDENCE_PATH")?;
    let revision = required_environment("SIM_ENGINE_RELEASE_SHA")?;
    let clean = |value: &str| value.replace(['\n', '\r', '='], " ");
    let body = format!(
        "format_version=1\nvcs_sha={}\nbackend={}\nname={}\nvendor={:#06x}\ndevice={:#06x}\npci_bus_id={}\ndriver={}\ndriver_info={}\nsurface_format={:?}\nsample_count={}\n",
        clean(&revision),
        renderer.adapter_backend(),
        clean(renderer.adapter_name()),
        renderer.adapter_vendor_id(),
        renderer.adapter_device_id(),
        clean(renderer.adapter_pci_bus_id()),
        clean(renderer.adapter_driver()),
        clean(renderer.adapter_driver_info()),
        renderer.surface_format(),
        renderer.surface_sample_count(),
    );
    std::fs::write(Path::new(&path), body)?;
    Ok(())
}

fn write_hidpi_evidence(
    state: &HidpiTransitionState,
    completed: PendingHidpiTransition,
) -> Result<(), Box<dyn Error>> {
    let Ok(path) = std::env::var("SIM_ENGINE_HIDPI_EVIDENCE_PATH") else {
        return Ok(());
    };
    let revision = required_environment("SIM_ENGINE_RELEASE_SHA")?;
    let body = format!(
        "format_version=1\nvcs_sha={revision}\nbackend={}\nadapter={}\nvendor={:#06x}\ndevice={:#06x}\npci_bus_id={}\nsurface_format={:?}\nsample_count={}\ntransition_serial={}\nscale_factor={:.3}\nphysical_width={}\nphysical_height={}\nscale_events={}\nresize_events={}\npaired_transitions={}\ncompleted_transitions={}\n",
        state.renderer.adapter_backend(),
        state.renderer.adapter_name(),
        state.renderer.adapter_vendor_id(),
        state.renderer.adapter_device_id(),
        state.renderer.adapter_pci_bus_id(),
        state.renderer.surface_format(),
        state.renderer.surface_sample_count(),
        completed.serial,
        completed.scale_factor,
        completed.physical_width,
        completed.physical_height,
        state.evidence.scale_events,
        state.evidence.resize_events,
        state.evidence.paired_transitions,
        state.evidence.completed_transitions,
    );
    std::fs::write(Path::new(&path), body)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RequiredAdapterIdentity {
    backend: String,
    name: String,
    vendor: u32,
    device: u32,
    pci_bus_id: String,
    surface_format: String,
    sample_count: u32,
}

fn validate_renderer_adapter(renderer: &WgpuRenderer, required: bool) -> Result<(), String> {
    if !required {
        return Ok(());
    }
    let require_surface_contract =
        std::env::var("SIM_ENGINE_REQUIRE_PRODUCTION_SURFACE_FORMAT").as_deref() == Ok("1");
    let expected = RequiredAdapterIdentity {
        backend: required_environment("SIM_ENGINE_REQUIRED_ADAPTER_BACKEND")?,
        name: required_environment("SIM_ENGINE_REQUIRED_ADAPTER_NAME")?,
        vendor: required_hex_environment("SIM_ENGINE_REQUIRED_ADAPTER_VENDOR")?,
        device: required_hex_environment("SIM_ENGINE_REQUIRED_ADAPTER_DEVICE")?,
        pci_bus_id: required_environment("SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID")?,
        surface_format: if require_surface_contract {
            required_environment("SIM_ENGINE_GPU_SURFACE_FORMAT")?
        } else {
            String::new()
        },
        sample_count: if require_surface_contract {
            required_environment("SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT")?
                .parse::<u32>()
                .map_err(|_| "SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT is not a u32".to_owned())?
        } else {
            0
        },
    };
    let actual = RequiredAdapterIdentity {
        backend: renderer.adapter_backend().to_owned(),
        name: renderer.adapter_name().to_owned(),
        vendor: renderer.adapter_vendor_id(),
        device: renderer.adapter_device_id(),
        pci_bus_id: renderer.adapter_pci_bus_id().to_owned(),
        surface_format: format!("{:?}", renderer.surface_format()),
        sample_count: renderer.surface_sample_count(),
    };
    validate_adapter_identity(&actual, &expected, require_surface_contract)
}

fn required_environment(name: &str) -> Result<String, String> {
    std::env::var(name).map_err(|_| format!("release evidence requires {name}"))
}

fn required_hex_environment(name: &str) -> Result<u32, String> {
    let value = required_environment(name)?;
    u32::from_str_radix(value.strip_prefix("0x").unwrap_or(&value), 16)
        .map_err(|_| format!("{name} is not a hexadecimal u32: {value}"))
}

fn validate_adapter_identity(
    actual: &RequiredAdapterIdentity,
    expected: &RequiredAdapterIdentity,
    require_surface_contract: bool,
) -> Result<(), String> {
    let physical_adapter_matches = actual.backend == expected.backend
        && actual.name == expected.name
        && actual.vendor == expected.vendor
        && actual.device == expected.device
        && actual.pci_bus_id == expected.pci_bus_id;
    let surface_contract_matches = actual.surface_format == expected.surface_format
        && actual.sample_count == expected.sample_count;
    if physical_adapter_matches && (!require_surface_contract || surface_contract_matches) {
        return Ok(());
    }
    Err(format!(
        "adapter/surface identity mismatch: expected {expected:?}, selected {actual:?}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidpi_evidence_requires_a_real_event_and_post_transition_present() {
        let mut evidence = HidpiEvidenceTracker::new(1.0);
        assert_eq!(
            validate_hidpi_evidence(&evidence),
            Err("no real ScaleFactorChanged event was observed")
        );
        evidence.observe_scale(1.25, 1_600, 900);
        assert_eq!(
            validate_hidpi_evidence(&evidence),
            Err("no Resized event followed the compositor scale transition")
        );
        assert_eq!(evidence.observe_successful_present(), None);
        evidence.observe_resize(1_600, 900);
        assert_eq!(
            validate_hidpi_evidence(&evidence),
            Err("the renderer did not present after the paired resize")
        );
        let completed = evidence
            .observe_successful_present()
            .expect("the paired resize may now complete");
        assert_eq!(completed.serial, 1);
        assert_eq!(validate_hidpi_evidence(&evidence), Ok(()));
    }

    #[test]
    fn performance_gate_keeps_streaming_and_retained_ceiling_separate() {
        let retained = gate_thresholds("four_viewports").expect("known retained fixture");
        let streaming = gate_thresholds("ui_90_10").expect("known streaming fixture");
        assert_eq!(retained.minimum_immediate_fps, 60.0);
        assert_eq!(retained.maximum_renderer_work_p95_ms, 5.0);
        assert_eq!(streaming.maximum_renderer_work_p95_ms, 25.0);
        assert!(gate_thresholds("unknown").is_none());
    }

    #[test]
    fn skipped_frames_cannot_enter_throughput_samples() {
        assert_eq!(
            require_drawn_frame("fixture", "measurement", 0, RenderStatus::Drawn),
            Ok(())
        );
        let error = require_drawn_frame(
            "fixture",
            "measurement",
            7,
            RenderStatus::Skipped(sim_engine::RendererSurfaceStatus::Timeout),
        )
        .unwrap_err();
        assert!(error.contains("skipped without submission/present"));
        assert!(error.contains("frame 7"));
    }

    #[test]
    fn zero_refresh_is_unconfirmed_output_metadata() {
        assert_eq!(confirmed_refresh_hz(None), None);
        assert_eq!(confirmed_refresh_hz(Some(0)), None);
        assert_eq!(confirmed_refresh_hz(Some(50_000)), Some(50.0));
    }

    #[test]
    fn confirmation_is_bound_to_surface_generation_and_output() {
        let output = 7_u64;
        let moved = 8_u64;
        assert!(confirmed_output_is_current(
            7,
            Some(&output),
            7,
            Some(&output)
        ));
        assert!(!confirmed_output_is_current(
            7,
            Some(&output),
            8,
            Some(&output)
        ));
        assert!(!confirmed_output_is_current(
            7,
            Some(&output),
            7,
            Some(&moved)
        ));
    }

    #[test]
    fn final_metadata_present_receives_a_follow_up_check() {
        assert_eq!(
            metadata_confirmation_action(RendererSurfacePresentMode::Fifo, None, 119),
            MetadataConfirmationAction::PresentAgain
        );
        assert_eq!(
            metadata_confirmation_action(RendererSurfacePresentMode::Fifo, Some(60.0), 120),
            MetadataConfirmationAction::Confirm
        );
        assert_eq!(
            metadata_confirmation_action(RendererSurfacePresentMode::Fifo, None, 120),
            MetadataConfirmationAction::Exhausted
        );
        assert_eq!(
            metadata_confirmation_action(RendererSurfacePresentMode::Immediate, None, 120),
            MetadataConfirmationAction::Confirm
        );
    }

    #[test]
    fn surface_restart_discards_partial_measurement_samples() {
        let mut measurement = BenchmarkMeasurement::new();
        measurement.renderer_work_samples.extend([1.0, 2.0]);
        measurement.acquire_samples.extend([0.1, 0.2]);
        measurement.trial_fps.push(60.0);
        measurement.reset();
        assert_eq!(measurement.discarded_measured_frames, 2);
        assert!(measurement.renderer_work_samples.is_empty());
        assert!(measurement.acquire_samples.is_empty());
        assert!(measurement.trial_fps.is_empty());
        assert!(matches!(
            measurement.phase,
            BenchmarkMeasurementPhase::Warmup { next_frame: 0 }
        ));
    }

    #[test]
    fn gate_context_requires_drawn_confirmation_but_refresh_only_for_synchronized_modes() {
        assert!(
            validate_gate_context(true, false, RendererSurfacePresentMode::Immediate, None,)
                .unwrap_err()
                .contains("drawn frame")
        );
        assert_eq!(
            validate_gate_context(true, true, RendererSurfacePresentMode::Immediate, None,),
            Ok(())
        );
        assert!(
            validate_gate_context(true, true, RendererSurfacePresentMode::Fifo, None,)
                .unwrap_err()
                .contains("positive")
        );
        assert!(
            validate_gate_context(true, true, RendererSurfacePresentMode::Fifo, Some(0.0),)
                .unwrap_err()
                .contains("positive")
        );
        assert_eq!(
            validate_gate_context(true, true, RendererSurfacePresentMode::Fifo, Some(60.0),),
            Ok(())
        );
    }

    #[test]
    fn performance_gate_separates_uncapped_throughput_from_refresh_pacing() {
        let thresholds = gate_thresholds("four_viewports").unwrap();
        assert_eq!(
            validate_performance_gate(
                "four_viewports",
                "vulkan",
                RendererSurfacePresentMode::Immediate,
                None,
                &[120.0, 121.0, 119.0],
                1.0,
                thresholds,
            ),
            Ok(PerformanceGateKind::UncappedThroughput)
        );
        assert_eq!(
            validate_performance_gate(
                "four_viewports",
                "vulkan",
                RendererSurfacePresentMode::Fifo,
                Some(50.0),
                &[50.0, 50.5, 49.5],
                1.0,
                thresholds,
            ),
            Ok(PerformanceGateKind::RefreshSynchronizedWork)
        );
        assert!(
            validate_performance_gate(
                "four_viewports",
                "gl",
                RendererSurfacePresentMode::Immediate,
                None,
                &[120.0],
                1.0,
                thresholds,
            )
            .unwrap_err()
            .contains("Vulkan")
        );
        assert!(
            validate_performance_gate(
                "four_viewports",
                "vulkan",
                RendererSurfacePresentMode::Immediate,
                None,
                &[59.0],
                1.0,
                thresholds,
            )
            .is_err()
        );
        assert!(
            validate_performance_gate(
                "four_viewports",
                "vulkan",
                RendererSurfacePresentMode::Fifo,
                Some(50.0),
                &[5.0],
                1.0,
                thresholds,
            )
            .unwrap_err()
            .contains("refresh-normalized")
        );
        assert!(
            validate_performance_gate(
                "four_viewports",
                "vulkan",
                RendererSurfacePresentMode::Fifo,
                Some(0.0),
                &[60.0],
                1.0,
                thresholds,
            )
            .unwrap_err()
            .contains("positive")
        );
        assert!(
            validate_performance_gate(
                "four_viewports",
                "vulkan",
                RendererSurfacePresentMode::Immediate,
                None,
                &[1.0, 65.0, 66.0],
                1.0,
                thresholds,
            )
            .unwrap_err()
            .contains("worst-trial")
        );
    }

    #[test]
    fn adapter_identity_requires_physical_pci_instance() {
        let expected = RequiredAdapterIdentity {
            backend: "vulkan".to_owned(),
            name: "adapter-a".to_owned(),
            vendor: 0x10de,
            device: 0x1234,
            pci_bus_id: "0000:01:00.0".to_owned(),
            surface_format: "Bgra8UnormSrgb".to_owned(),
            sample_count: 4,
        };
        assert_eq!(
            validate_adapter_identity(&expected, &expected, true),
            Ok(())
        );
        for actual in [
            (
                "gl",
                "adapter-a",
                0x10de,
                0x1234,
                "0000:01:00.0",
                "Bgra8UnormSrgb",
                4,
            ),
            (
                "vulkan",
                "adapter-b",
                0x10de,
                0x1234,
                "0000:01:00.0",
                "Bgra8UnormSrgb",
                4,
            ),
            (
                "vulkan",
                "adapter-a",
                0x8086,
                0x1234,
                "0000:01:00.0",
                "Bgra8UnormSrgb",
                4,
            ),
            (
                "vulkan",
                "adapter-a",
                0x10de,
                0x4321,
                "0000:01:00.0",
                "Bgra8UnormSrgb",
                4,
            ),
            (
                "vulkan",
                "adapter-a",
                0x10de,
                0x1234,
                "0000:02:00.0",
                "Bgra8UnormSrgb",
                4,
            ),
            (
                "vulkan",
                "adapter-a",
                0x10de,
                0x1234,
                "0000:01:00.0",
                "Rgba8UnormSrgb",
                4,
            ),
            (
                "vulkan",
                "adapter-a",
                0x10de,
                0x1234,
                "0000:01:00.0",
                "Bgra8UnormSrgb",
                1,
            ),
        ] {
            let actual = RequiredAdapterIdentity {
                backend: actual.0.to_owned(),
                name: actual.1.to_owned(),
                vendor: actual.2,
                device: actual.3,
                pci_bus_id: actual.4.to_owned(),
                surface_format: actual.5.to_owned(),
                sample_count: actual.6,
            };
            assert!(validate_adapter_identity(&actual, &expected, true).is_err());
            let physical_fields_differ = actual.backend != expected.backend
                || actual.name != expected.name
                || actual.vendor != expected.vendor
                || actual.device != expected.device
                || actual.pci_bus_id != expected.pci_bus_id;
            assert_eq!(
                validate_adapter_identity(&actual, &expected, false).is_err(),
                physical_fields_differ,
            );
        }
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

fn prepare_benchmark_workload(
    name: &str,
    renderer: &mut WgpuRenderer,
) -> Result<BenchmarkWorkload, Box<dyn Error>> {
    match name {
        "ui_static_10k" => prepare_static_ui(renderer),
        "ui_90_10" => prepare_ui_90_10(renderer),
        "four_viewports" => prepare_four_viewports(renderer),
        "image_atlas" => prepare_image_atlas(renderer),
        "scientific_text" => prepare_scientific_text(renderer),
        "dpi_reconfigure" => prepare_dpi_reconfigure(renderer),
        fixture => Err(format!("unknown benchmark fixture: {fixture}").into()),
    }
}

fn prepare_static_ui(renderer: &mut WgpuRenderer) -> Result<BenchmarkWorkload, Box<dyn Error>> {
    let construction_started = Instant::now();
    let scene = build_screen_scene(STATIC_COMMANDS + STREAMING_COMMANDS, 0)?;
    let construction = construction_started.elapsed();
    let preparation_started = Instant::now();
    let prepared = renderer.prepare_screen_scene(&scene)?;
    let preparation = preparation_started.elapsed();
    Ok(BenchmarkWorkload {
        name: "ui_static_10k",
        construction,
        completion_note: Some(format!(
            "prepare_cpu_ms={:.3}",
            preparation.as_secs_f64() * 1_000.0
        )),
        render: Box::new(move |renderer, _, _, _| {
            let mut frame = renderer.begin_frame(scene.background(), FrameBudget::default())?;
            frame.draw_prepared_screen_scene(&prepared, FramePassOptions::new(0))?;
            Ok(frame.present()?)
        }),
    })
}

fn prepare_ui_90_10(renderer: &mut WgpuRenderer) -> Result<BenchmarkWorkload, Box<dyn Error>> {
    let construction_started = Instant::now();
    let static_scene = build_screen_scene(STATIC_COMMANDS, 0)?;
    let prepared = renderer.prepare_screen_scene(&static_scene)?;
    let initial_construction = construction_started.elapsed();
    Ok(BenchmarkWorkload {
        name: "ui_90_10",
        construction: initial_construction,
        completion_note: None,
        render: Box::new(move |renderer, frame_index, _, _| {
            let streaming = build_screen_scene(STREAMING_COMMANDS, frame_index)?;
            let mut frame = renderer.begin_frame(Color::rgb8(9, 12, 18), FrameBudget::default())?;
            frame.draw_prepared_screen_scene(&prepared, FramePassOptions::new(0))?;
            frame.draw_screen_scene(&streaming, FramePassOptions::new(1))?;
            Ok(frame.present()?)
        }),
    })
}

fn prepare_four_viewports(
    renderer: &mut WgpuRenderer,
) -> Result<BenchmarkWorkload, Box<dyn Error>> {
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
    Ok(BenchmarkWorkload {
        name: "four_viewports",
        construction,
        completion_note: None,
        render: Box::new(move |renderer, _, _, _| {
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
        }),
    })
}

fn prepare_image_atlas(renderer: &mut WgpuRenderer) -> Result<BenchmarkWorkload, Box<dyn Error>> {
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
    Ok(BenchmarkWorkload {
        name: "image_atlas",
        construction: started.elapsed(),
        completion_note: None,
        render: Box::new(move |renderer, _, _, _| {
            let mut frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
            frame.draw_image_batch(
                &image,
                &batch,
                ImageSampling::Nearest,
                FramePassOptions::new(0),
            )?;
            Ok(frame.present()?)
        }),
    })
}

fn prepare_scientific_text(
    renderer: &mut WgpuRenderer,
) -> Result<BenchmarkWorkload, Box<dyn Error>> {
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
    Ok(BenchmarkWorkload {
        name: "scientific_text",
        construction: started.elapsed(),
        completion_note: None,
        render: Box::new(move |renderer, _, _, _| {
            let mut frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
            frame.draw_glyph_run(
                &atlas,
                &run,
                ImageSampling::Nearest,
                FramePassOptions::new(0),
            )?;
            Ok(frame.present()?)
        }),
    })
}

fn prepare_dpi_reconfigure(
    renderer: &mut WgpuRenderer,
) -> Result<BenchmarkWorkload, Box<dyn Error>> {
    let started = Instant::now();
    let scene = build_screen_scene(2_500, 0)?;
    let prepared = renderer.prepare_screen_scene(&scene)?;
    let scales = [1.0, 1.25, 1.5, 2.0, 3.0];
    Ok(BenchmarkWorkload {
        name: "dpi_reconfigure",
        construction: started.elapsed(),
        completion_note: None,
        render: Box::new(move |renderer, frame, width, height| {
            renderer.resize_with_scale_factor(width, height, scales[frame % scales.len()])?;
            let mut composed = renderer.begin_frame(scene.background(), FrameBudget::default())?;
            composed.draw_prepared_screen_scene(&prepared, FramePassOptions::new(0))?;
            Ok(composed.present()?)
        }),
    })
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

fn advance_benchmark(
    state: &mut BenchmarkRunState,
    gated: bool,
) -> Result<BenchmarkAdvance, Box<dyn Error>> {
    if gated {
        match state.confirmation.clone() {
            OutputConfirmationState::Required => {
                present_confirmation_frame(
                    &mut state.renderer,
                    state.workload.name,
                    "output-confirmation",
                    0,
                )?;
                state.confirmation = OutputConfirmationState::AwaitingMetadata {
                    generation: state.surface_generation,
                    completed_presents: 0,
                };
                return Ok(BenchmarkAdvance::Continue);
            }
            OutputConfirmationState::AwaitingMetadata {
                generation,
                completed_presents,
            } => {
                if generation != state.surface_generation {
                    state.confirmation = OutputConfirmationState::Required;
                    return Ok(BenchmarkAdvance::Continue);
                }
                let output = current_output_snapshot(&state.window);
                let display_refresh_hz = output_refresh_hz(output.as_ref());
                let present_mode = state.renderer.surface_present_mode();
                match metadata_confirmation_action(
                    present_mode,
                    display_refresh_hz,
                    completed_presents,
                ) {
                    MetadataConfirmationAction::Confirm => {
                        validate_gate_context(true, true, present_mode, display_refresh_hz)
                            .map_err(|error| -> Box<dyn Error> { error.into() })?;
                        state.gate_context = Some(BenchmarkGateContext {
                            enabled: true,
                            display_refresh_hz,
                        });
                        state.confirmation =
                            OutputConfirmationState::Confirmed { generation, output };
                        return Ok(BenchmarkAdvance::Continue);
                    }
                    MetadataConfirmationAction::PresentAgain => {
                        present_confirmation_frame(
                            &mut state.renderer,
                            state.workload.name,
                            "output-metadata-confirmation",
                            completed_presents,
                        )?;
                        state.confirmation = OutputConfirmationState::AwaitingMetadata {
                            generation,
                            completed_presents: completed_presents.saturating_add(1),
                        };
                        return Ok(BenchmarkAdvance::Continue);
                    }
                    MetadataConfirmationAction::Exhausted => {
                        return Err(
                            "gated surface measurement could not confirm the current output and positive refresh rate"
                                .into(),
                        );
                    }
                }
            }
            OutputConfirmationState::Confirmed { generation, output } => {
                let observed_output = current_output_snapshot(&state.window);
                if !confirmed_output_is_current(
                    generation,
                    output.as_ref(),
                    state.surface_generation,
                    observed_output.as_ref(),
                ) {
                    state.invalidate_surface(true)?;
                    return Ok(BenchmarkAdvance::Continue);
                }
            }
        }
    }

    let gate = state
        .gate_context
        .ok_or("benchmark measurement started without confirmed gate context")?;
    let trial_count = if gate.enabled { GATE_TRIALS } else { 1 };
    match state.measurement.phase {
        BenchmarkMeasurementPhase::Warmup { next_frame } => {
            let report = (state.workload.render)(
                &mut state.renderer,
                next_frame,
                state.physical_width,
                state.physical_height,
            )?;
            require_drawn_frame(state.workload.name, "warmup", next_frame, report.status())?;
            if next_frame + 1 == WARMUP_FRAMES {
                state.measurement.phase = BenchmarkMeasurementPhase::Trial {
                    trial: 0,
                    next_frame: 0,
                    started: None,
                };
            } else {
                state.measurement.phase = BenchmarkMeasurementPhase::Warmup {
                    next_frame: next_frame + 1,
                };
            }
            Ok(BenchmarkAdvance::Continue)
        }
        BenchmarkMeasurementPhase::Trial {
            trial,
            next_frame,
            started,
        } => {
            let started = if let Some(started) = started {
                started
            } else {
                state.renderer.wait_for_gpu_idle()?;
                Instant::now()
            };
            let absolute_frame = WARMUP_FRAMES + trial * MEASURED_FRAMES + next_frame;
            let report = (state.workload.render)(
                &mut state.renderer,
                absolute_frame,
                state.physical_width,
                state.physical_height,
            )?;
            require_drawn_frame(
                state.workload.name,
                "measurement",
                absolute_frame,
                report.status(),
            )?;
            let metrics = report.metrics();
            state.measurement.renderer_work_samples.push(
                metrics
                    .total_cpu()
                    .saturating_sub(metrics.surface_acquire())
                    .as_secs_f64()
                    * 1_000.0,
            );
            state
                .measurement
                .acquire_samples
                .push(metrics.surface_acquire().as_secs_f64() * 1_000.0);
            state.measurement.last_report = Some(report);

            if next_frame + 1 == MEASURED_FRAMES {
                state.renderer.wait_for_gpu_idle()?;
                state
                    .measurement
                    .trial_fps
                    .push(MEASURED_FRAMES as f64 / started.elapsed().as_secs_f64());
                if trial + 1 == trial_count {
                    state.measurement.phase = BenchmarkMeasurementPhase::Complete;
                    finish_benchmark(state, gate)?;
                    Ok(BenchmarkAdvance::Complete)
                } else {
                    state.measurement.phase = BenchmarkMeasurementPhase::Trial {
                        trial: trial + 1,
                        next_frame: 0,
                        started: None,
                    };
                    Ok(BenchmarkAdvance::Continue)
                }
            } else {
                state.measurement.phase = BenchmarkMeasurementPhase::Trial {
                    trial,
                    next_frame: next_frame + 1,
                    started: Some(started),
                };
                Ok(BenchmarkAdvance::Continue)
            }
        }
        BenchmarkMeasurementPhase::Complete => Ok(BenchmarkAdvance::Complete),
    }
}

fn finish_benchmark(
    state: &mut BenchmarkRunState,
    gate: BenchmarkGateContext,
) -> Result<(), Box<dyn Error>> {
    let name = state.workload.name;
    let construction = state.workload.construction;
    let trial_count = if gate.enabled { GATE_TRIALS } else { 1 };
    let measured_attempts = MEASURED_FRAMES * trial_count;
    let total_measured_attempts =
        measured_attempts.saturating_add(state.measurement.discarded_measured_frames);
    state
        .measurement
        .renderer_work_samples
        .sort_by(f64::total_cmp);
    state.measurement.acquire_samples.sort_by(f64::total_cmp);
    state.measurement.trial_fps.sort_by(f64::total_cmp);
    let percentile = |samples: &[f64], fraction: f64| {
        let index = ((samples.len() - 1) as f64 * fraction).round() as usize;
        samples[index]
    };
    let report = state
        .measurement
        .last_report
        .as_ref()
        .expect("measured frame count is non-zero");
    let statistics = report.statistics();
    validate_fixture_contract(name, statistics)?;
    let sources = statistics.source_counts();
    let renderer_work_samples = &state.measurement.renderer_work_samples;
    let acquire_samples = &state.measurement.acquire_samples;
    let trial_fps = &state.measurement.trial_fps;
    let minimum_wall_fps = trial_fps[0];
    let median_wall_fps = trial_fps[trial_fps.len() / 2];
    let work_p95 = percentile(renderer_work_samples, 0.95);
    println!(
        "fixture={name} adapter={:?} vendor={:#06x} device={:#06x} pci_bus_id={:?} backend={} driver={:?} driver_info={:?} surface_format={:?} sample_count={} present_mode={} display_refresh_hz={:?} surface_generation={} surface_restarts={} trials={} frames_per_trial={} attempted_frames={} drawn_frames={} accepted_measurement_frames={} discarded_measurement_frames={} minimum_trial_wall_fps={:.1} median_trial_wall_fps={:.1} trial_wall_fps={:?} renderer_work_excluding_acquire_ms[p50={:.3},p95={:.3},p99={:.3}] surface_acquire_ms[p50={:.3},p95={:.3},p99={:.3}] construction_ms={:.3}",
        state.renderer.adapter_name(),
        state.renderer.adapter_vendor_id(),
        state.renderer.adapter_device_id(),
        state.renderer.adapter_pci_bus_id(),
        state.renderer.adapter_backend(),
        state.renderer.adapter_driver(),
        state.renderer.adapter_driver_info(),
        state.renderer.surface_format(),
        state.renderer.surface_sample_count(),
        state.renderer.surface_present_mode(),
        gate.display_refresh_hz,
        state.surface_generation,
        state.surface_restarts,
        trial_count,
        MEASURED_FRAMES,
        total_measured_attempts,
        total_measured_attempts,
        measured_attempts,
        state.measurement.discarded_measured_frames,
        minimum_wall_fps,
        median_wall_fps,
        trial_fps,
        percentile(renderer_work_samples, 0.50),
        percentile(renderer_work_samples, 0.95),
        percentile(renderer_work_samples, 0.99),
        percentile(acquire_samples, 0.50),
        percentile(acquire_samples, 0.95),
        percentile(acquire_samples, 0.99),
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
    if gate.enabled {
        let thresholds = gate_thresholds(name)
            .ok_or_else(|| format!("{name} does not define release-gate thresholds"))?;
        let kind = validate_performance_gate(
            name,
            state.renderer.adapter_backend(),
            state.renderer.surface_present_mode(),
            gate.display_refresh_hz,
            trial_fps,
            work_p95,
            thresholds,
        )
        .map_err(|error| -> Box<dyn Error> { error.into() })?;
        match kind {
            PerformanceGateKind::UncappedThroughput => println!(
                "gate=passed kind=uncapped_throughput worst_trial_wall_fps={minimum_wall_fps:.1} min_wall_fps={:.1} max_renderer_work_p95_ms={:.3} acquire_percentiles=informational",
                thresholds.minimum_immediate_fps, thresholds.maximum_renderer_work_p95_ms,
            ),
            PerformanceGateKind::RefreshSynchronizedWork => println!(
                "gate=passed kind=refresh_synchronized_work worst_trial_wall_fps={minimum_wall_fps:.1} wall_fps=refresh_normalized acquire_percentiles=informational max_renderer_work_p95_ms={:.3}",
                thresholds.maximum_renderer_work_p95_ms,
            ),
        }
    }
    if let Some(note) = &state.workload.completion_note {
        println!("{note}");
    }
    Ok(())
}

fn require_drawn_frame(
    fixture: &str,
    phase: &str,
    index: usize,
    status: RenderStatus,
) -> Result<(), String> {
    match status {
        RenderStatus::Drawn => Ok(()),
        RenderStatus::Skipped(reason) => Err(format!(
            "{fixture} {phase} frame {index} was skipped without submission/present: {reason:?}"
        )),
    }
}

fn present_confirmation_frame(
    renderer: &mut WgpuRenderer,
    fixture: &str,
    phase: &str,
    index: usize,
) -> Result<(), String> {
    let result = (|| -> Result<RenderStatus, Box<dyn Error>> {
        let frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
        Ok(frame.present()?.status())
    })();
    let status = result.map_err(|error| error.to_string())?;
    require_drawn_frame(fixture, phase, index, status)
}

fn validate_performance_gate(
    fixture: &str,
    backend: &str,
    present_mode: RendererSurfacePresentMode,
    display_refresh_hz: Option<f64>,
    trial_wall_fps: &[f64],
    renderer_work_p95_ms: f64,
    thresholds: GateThresholds,
) -> Result<PerformanceGateKind, String> {
    if backend != "vulkan" {
        return Err(format!(
            "{fixture} release performance evidence requires Vulkan, selected backend was {backend}"
        ));
    }
    if renderer_work_p95_ms > thresholds.maximum_renderer_work_p95_ms {
        return Err(format!(
            "{fixture} renderer work p95 {renderer_work_p95_ms:.3} ms exceeds {:.3} ms",
            thresholds.maximum_renderer_work_p95_ms,
        ));
    }
    if trial_wall_fps.is_empty()
        || trial_wall_fps
            .iter()
            .any(|fps| !fps.is_finite() || *fps <= 0.0)
    {
        return Err(format!(
            "{fixture} performance evidence requires positive finite wall-throughput trials"
        ));
    }
    let minimum_trial_fps = trial_wall_fps
        .iter()
        .copied()
        .reduce(f64::min)
        .expect("non-empty trial list was checked");
    if present_mode.is_refresh_synchronized() {
        let refresh_hz = display_refresh_hz.ok_or_else(|| {
            format!(
                "{fixture} refresh-synchronized performance evidence requires monitor refresh metadata"
            )
        })?;
        if !refresh_hz.is_finite() || refresh_hz <= 0.0 {
            return Err(format!(
                "{fixture} refresh-synchronized performance evidence requires a positive finite monitor refresh rate"
            ));
        }
        let minimum_fps = refresh_hz.clamp(30.0, 60.0) * 0.95;
        if minimum_trial_fps < minimum_fps {
            return Err(format!(
                "{fixture} refresh-normalized worst-trial wall throughput {minimum_trial_fps:.1} FPS is below {minimum_fps:.1} FPS for a {refresh_hz:.3} Hz monitor"
            ));
        }
        return Ok(PerformanceGateKind::RefreshSynchronizedWork);
    }
    if minimum_trial_fps < thresholds.minimum_immediate_fps {
        return Err(format!(
            "{fixture} worst-trial wall throughput {minimum_trial_fps:.1} FPS is below the {:.1} FPS gate",
            thresholds.minimum_immediate_fps,
        ));
    }
    Ok(PerformanceGateKind::UncappedThroughput)
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
