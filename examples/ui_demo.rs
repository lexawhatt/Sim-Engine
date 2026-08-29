use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use sim_engine::{
    Camera2d, Color, Fill, FrameBudget, FramePassOptions, GlyphAtlas2d, GlyphAtlasBudget,
    GlyphAtlasEntry, GlyphId, GlyphRun2d, GlyphRunBudget, ImageBudget, ImageSampling,
    ImageTexelRect, LinearGradient, LogicalScreenPosition, LogicalScreenVector, LogicalViewport,
    LogicalViewportRegion, PhysicalScreenPosition, PositionedGlyph2d, Projection2d, Rect,
    RendererFrameMetrics, Scene, ScreenClipRect, ShapeStyle, Vec2, WgpuRenderer,
    WgpuRendererOptions,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorIcon, Window, WindowId},
};

const WAVE_COUNT: usize = 4;
const WAVE_SAMPLE_COUNT: usize = 48;
const TARGET_DASH_COUNT: usize = 8;
const MIN_FREQUENCY: f32 = 0.65;
const MAX_FREQUENCY: f32 = 5.25;
const MIN_SPEED: f32 = 0.0;
const MAX_SPEED: f32 = 2.4;
const SUCCESS_ACCURACY: f32 = 0.93;
const EDGE_CASE_ZOOM: f32 = 10_000.0;
// Schedule at 120 Hz and let FIFO presentation provide the final display pacing.
// A 60 Hz timer accumulates event-loop/compositor latency and otherwise settles
// around 53-59 FPS even when the renderer itself needs only a few milliseconds.
const TARGET_FRAME_INTERVAL: Duration = Duration::from_nanos(8_333_333);

const WAVE_COLORS: [Color; WAVE_COUNT] = [
    Color::rgb(0.073, 0.651, 0.752),
    Color::rgb(0.930, 0.552, 0.063),
    Color::rgb(0.095, 0.604, 0.286),
    Color::rgb(0.965, 0.141, 0.117),
];

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let event_loop = EventLoop::new()?;
    let mut application = UiDemoApplication::new();
    event_loop.run_app(&mut application)?;

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteractionTarget {
    MenuSimulation(DemoScreen),
    PlaybackButton,
    ResetButton,
    FrequencySlider(usize),
    SpeedSelector(usize),
    SpeedSlider,
    AcceptButton,
    DismissButton,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DemoScreen {
    Menu,
    FluidSimulation,
    GasSimulation,
    WaveSimulation,
    EdgeCaseLab,
}

impl DemoScreen {
    fn label(self) -> &'static str {
        match self {
            Self::Menu => "Menu",
            Self::FluidSimulation => "Fluid",
            Self::GasSimulation => "Gas",
            Self::WaveSimulation => "Wave",
            Self::EdgeCaseLab => "Edge",
        }
    }
}

struct ShowcaseFrameMetrics {
    screen: DemoScreen,
    started_at: Instant,
    previous_frame_at: Instant,
    frames: usize,
    frame_intervals: Duration,
    frame_samples: Vec<Duration>,
    scene_time: Duration,
    renderer_time: Duration,
    tessellation_time: Duration,
    upload_time: Duration,
    surface_acquire_time: Duration,
}

impl ShowcaseFrameMetrics {
    fn new(screen: DemoScreen, now: Instant) -> Self {
        Self {
            screen,
            started_at: now,
            previous_frame_at: now,
            frames: 0,
            frame_intervals: Duration::ZERO,
            frame_samples: Vec::with_capacity(240),
            scene_time: Duration::ZERO,
            renderer_time: Duration::ZERO,
            tessellation_time: Duration::ZERO,
            upload_time: Duration::ZERO,
            surface_acquire_time: Duration::ZERO,
        }
    }

    fn record(
        &mut self,
        frame_started_at: Instant,
        scene_time: Duration,
        renderer: RendererFrameMetrics,
        command_count: usize,
    ) -> Option<String> {
        if self.frames > 0 {
            let interval = frame_started_at.saturating_duration_since(self.previous_frame_at);
            self.frame_intervals += interval;
            self.frame_samples.push(interval);
        }
        self.previous_frame_at = frame_started_at;
        self.frames += 1;
        self.scene_time += scene_time;
        self.renderer_time += renderer.total_cpu();
        self.tessellation_time += renderer.tessellation();
        self.upload_time += renderer.upload();
        self.surface_acquire_time += renderer.surface_acquire();

        let elapsed = frame_started_at.saturating_duration_since(self.started_at);
        if elapsed < Duration::from_secs(1) {
            return None;
        }

        let frames = self.frames.max(1) as f64;
        let interval_count = self.frames.saturating_sub(1).max(1) as f64;
        let fps = self.frames as f64 / elapsed.as_secs_f64();
        let frame_ms = self.frame_intervals.as_secs_f64() * 1000.0 / interval_count;
        self.frame_samples.sort_unstable();
        let p99_frame_ms = self
            .frame_samples
            .get(
                (self.frame_samples.len() * 99 / 100)
                    .min(self.frame_samples.len().saturating_sub(1)),
            )
            .copied()
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0;
        let scene_ms = self.scene_time.as_secs_f64() * 1000.0 / frames;
        let renderer_ms = self.renderer_time.as_secs_f64() * 1000.0 / frames;
        let tessellation_ms = self.tessellation_time.as_secs_f64() * 1000.0 / frames;
        let upload_ms = self.upload_time.as_secs_f64() * 1000.0 / frames;
        let acquire_ms = self.surface_acquire_time.as_secs_f64() * 1000.0 / frames;
        let idle_ms = (frame_ms - scene_ms - renderer_ms).max(0.0);
        let label = self.screen.label();
        println!(
            "ui_demo {label}: {fps:.1} fps, frame {frame_ms:.2} ms, p99 frame {p99_frame_ms:.2} ms, scene {scene_ms:.2} ms, renderer {renderer_ms:.2} ms, tessellate {tessellation_ms:.2} ms, upload {upload_ms:.2} ms, acquire/wait {acquire_ms:.2} ms, idle/scheduler {idle_ms:.2} ms, commands {command_count}"
        );
        let title = format!(
            "Sim;Engine {label} | {fps:.1} FPS | scene {scene_ms:.2} ms | tess {tessellation_ms:.2} ms"
        );
        *self = Self::new(self.screen, frame_started_at);
        Some(title)
    }
}

struct UiDemoApplication {
    window: Option<Arc<Window>>,
    renderer: Option<WgpuRenderer>,
    glyph_probe_atlas: Option<GlyphAtlas2d>,
    glyph_probe_run: Option<GlyphRun2d>,
    camera: Camera2d,
    screen: DemoScreen,
    uncapped: bool,
    pointer_logical: Vec2,
    pointer_inside: bool,
    active_drag: Option<InteractionTarget>,
    selected_speed_wave: usize,
    challenge: WaveChallenge,
    playing: bool,
    animation_time: f32,
    success_dismissed: bool,
    previous_frame: Instant,
    next_frame_at: Instant,
    frame_metrics: ShowcaseFrameMetrics,
    random: SimpleRandom,
}

impl UiDemoApplication {
    fn new() -> Self {
        let mut random = SimpleRandom::from_system_time();
        let mut challenge = WaveChallenge::random(&mut random);
        if solved_preview_requested() {
            challenge.snap_to_target();
        }
        let now = Instant::now();
        let screen = requested_initial_screen().unwrap_or(DemoScreen::Menu);
        let uncapped = uncapped_requested();
        Self {
            window: None,
            renderer: None,
            glyph_probe_atlas: None,
            glyph_probe_run: None,
            camera: Camera2d::new(Vec2::ZERO, 1.0).expect("UI demo camera is valid"),
            screen,
            uncapped,
            pointer_logical: Vec2::ZERO,
            pointer_inside: false,
            active_drag: None,
            selected_speed_wave: 0,
            challenge,
            playing: true,
            animation_time: 0.0,
            success_dismissed: false,
            previous_frame: now,
            next_frame_at: now,
            frame_metrics: ShowcaseFrameMetrics::new(screen, now),
            random,
        }
    }

    fn layout(&self) -> Option<UiLayout> {
        self.renderer.as_ref().map(|renderer| {
            let (width, height) = renderer.logical_size();
            UiLayout::new(Vec2::new(width, height))
        })
    }

    fn show_success(&self) -> bool {
        self.challenge.is_solved() && !self.success_dismissed
    }

    fn hovered_target(&self, layout: UiLayout) -> Option<InteractionTarget> {
        if !self.pointer_inside {
            return None;
        }
        match self.screen {
            DemoScreen::Menu => layout.menu_hit_test(self.pointer_logical),
            DemoScreen::WaveSimulation => layout.hit_test(
                self.pointer_logical,
                &self.challenge.frequency_amounts,
                self.challenge.speed_amounts[self.selected_speed_wave],
                self.show_success(),
            ),
            DemoScreen::FluidSimulation | DemoScreen::GasSimulation | DemoScreen::EdgeCaseLab => {
                None
            }
        }
    }

    fn update_drag(&mut self, layout: UiLayout) {
        if self.screen != DemoScreen::WaveSimulation {
            return;
        }
        match self.active_drag {
            Some(InteractionTarget::FrequencySlider(index)) => {
                self.challenge.frequency_amounts[index] =
                    layout.frequency_amount_at(index, self.pointer_logical.x());
                self.success_dismissed = false;
            }
            Some(InteractionTarget::SpeedSlider) => {
                self.challenge.speed_amounts[self.selected_speed_wave] =
                    layout.speed_amount_at(self.pointer_logical.x());
                self.success_dismissed = false;
            }
            _ => {}
        }
    }

    fn start_new_round(&mut self) {
        self.challenge = WaveChallenge::random(&mut self.random);
        self.selected_speed_wave = 0;
        self.animation_time = 0.0;
        self.success_dismissed = false;
    }

    fn hard_reset(&mut self) {
        self.playing = false;
        self.animation_time = 0.0;
    }

    fn show_simulation(&mut self, screen: DemoScreen) {
        if screen == DemoScreen::Menu {
            return;
        }
        self.screen = screen;
        self.active_drag = None;
        let now = Instant::now();
        self.previous_frame = now;
        self.next_frame_at = now;
        self.frame_metrics = ShowcaseFrameMetrics::new(screen, now);
    }

    fn show_menu(&mut self) {
        self.screen = DemoScreen::Menu;
        self.active_drag = None;
        let now = Instant::now();
        self.next_frame_at = now;
        self.frame_metrics = ShowcaseFrameMetrics::new(DemoScreen::Menu, now);
    }

    fn handle_key(&mut self, key_code: KeyCode) {
        match (self.screen, key_code) {
            (DemoScreen::Menu, KeyCode::Digit1 | KeyCode::Enter | KeyCode::Space) => {
                self.show_simulation(DemoScreen::FluidSimulation);
            }
            (DemoScreen::Menu, KeyCode::Digit2) => {
                self.show_simulation(DemoScreen::GasSimulation);
            }
            (DemoScreen::Menu, KeyCode::Digit3) => {
                self.show_simulation(DemoScreen::WaveSimulation);
            }
            (DemoScreen::Menu, KeyCode::Digit4) => {
                self.show_simulation(DemoScreen::EdgeCaseLab);
            }
            (screen, KeyCode::Escape) if screen != DemoScreen::Menu => self.show_menu(),
            (screen, KeyCode::Space) if screen != DemoScreen::Menu => {
                self.playing = !self.playing;
            }
            (screen, KeyCode::KeyR) if screen != DemoScreen::Menu => self.hard_reset(),
            _ => {}
        }
    }

    fn active_camera(&self) -> Camera2d {
        if self.screen == DemoScreen::EdgeCaseLab {
            let mut camera =
                Camera2d::new(Vec2::ZERO, EDGE_CASE_ZOOM).expect("edge-case camera is valid");
            camera.set_projection(
                Projection2d::new(0.42, 0.002).expect("edge-case projection is valid"),
            );
            camera
        } else {
            self.camera
        }
    }

    fn update_cursor(&self, window: &Window, layout: UiLayout) {
        let cursor = match self.active_drag {
            Some(InteractionTarget::FrequencySlider(_) | InteractionTarget::SpeedSlider) => {
                CursorIcon::Grabbing
            }
            Some(_) => CursorIcon::Pointer,
            None => match self.hovered_target(layout) {
                Some(InteractionTarget::FrequencySlider(_) | InteractionTarget::SpeedSlider) => {
                    CursorIcon::Grab
                }
                Some(_) => CursorIcon::Pointer,
                None => CursorIcon::Default,
            },
        };
        window.set_cursor(cursor);
    }

    fn update_animation(&mut self, now: Instant) {
        let elapsed = now
            .saturating_duration_since(self.previous_frame)
            .as_secs_f32()
            .min(0.05);
        self.previous_frame = now;
        if self.screen != DemoScreen::Menu && self.playing {
            self.animation_time += elapsed;
        }
        if !self.challenge.is_solved() {
            self.success_dismissed = false;
        }
    }
}

impl ApplicationHandler for UiDemoApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("Sim;Engine simulation showcase")
            .with_inner_size(LogicalSize::new(1100.0, 760.0))
            .with_min_inner_size(LogicalSize::new(800.0, 620.0));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("create UI window"),
        );
        let size = window.inner_size();
        let present_mode = if self.uncapped {
            sim_engine::RendererPresentMode::NoVsync
        } else {
            sim_engine::RendererPresentMode::Vsync
        };
        let renderer_options = WgpuRendererOptions::new(present_mode, window.scale_factor())
            .expect("window scale factor is valid");
        let renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            renderer_options,
        ))
        .expect("create UI renderer");
        let (glyph_probe_atlas, glyph_probe_run) =
            create_scientific_glyph_probe(&renderer).expect("create retained glyph probe");

        window.request_redraw();
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.glyph_probe_atlas = Some(glyph_probe_atlas);
        self.glyph_probe_run = Some(glyph_probe_run);
        let now = Instant::now();
        self.previous_frame = now;
        self.next_frame_at = now;
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
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                if let PhysicalKey::Code(key_code) = event.physical_key {
                    self.handle_key(key_code);
                    if let Some(layout) = self.layout() {
                        self.update_cursor(&window, layout);
                    }
                    window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(renderer) = self.renderer.as_ref() {
                    self.pointer_logical = renderer
                        .physical_to_logical_screen(PhysicalScreenPosition::new(
                            position.x as f32,
                            position.y as f32,
                        ))
                        .expect("window pointer coordinates must convert")
                        .to_vec2();
                    self.pointer_inside = true;
                }
                if let Some(layout) = self.layout() {
                    self.update_drag(layout);
                    self.update_cursor(&window, layout);
                }
                window.request_redraw();
            }
            WindowEvent::CursorLeft { .. } => {
                self.pointer_inside = false;
                if self.active_drag.is_none() {
                    window.set_cursor(CursorIcon::Default);
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                let Some(layout) = self.layout() else {
                    return;
                };
                match state {
                    ElementState::Pressed => {
                        self.active_drag = self.hovered_target(layout);
                        match self.active_drag {
                            Some(InteractionTarget::MenuSimulation(screen)) => {
                                self.show_simulation(screen);
                            }
                            Some(InteractionTarget::PlaybackButton) => {
                                self.playing = !self.playing;
                            }
                            Some(InteractionTarget::ResetButton) => {
                                self.hard_reset();
                            }
                            Some(InteractionTarget::SpeedSelector(index)) => {
                                self.selected_speed_wave = index;
                            }
                            Some(
                                InteractionTarget::FrequencySlider(_)
                                | InteractionTarget::SpeedSlider,
                            ) => self.update_drag(layout),
                            Some(InteractionTarget::AcceptButton) => self.start_new_round(),
                            Some(InteractionTarget::DismissButton) => {
                                self.success_dismissed = true;
                            }
                            None => {}
                        }
                    }
                    ElementState::Released => self.active_drag = None,
                }
                self.update_cursor(&window, layout);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let frame_started_at = Instant::now();
                self.update_animation(frame_started_at);
                let Some(layout) = self.layout() else {
                    return;
                };
                let hovered = self.hovered_target(layout);
                let scene = match self.screen {
                    DemoScreen::Menu => build_menu_scene(layout, hovered),
                    DemoScreen::FluidSimulation => {
                        build_fluid_scene(layout, self.animation_time, self.playing)
                    }
                    DemoScreen::GasSimulation => {
                        build_gas_scene(layout, self.animation_time, self.playing)
                    }
                    DemoScreen::WaveSimulation => build_ui_scene(
                        layout,
                        &self.challenge,
                        self.animation_time,
                        self.playing,
                        self.selected_speed_wave,
                        hovered,
                        self.active_drag,
                        self.show_success(),
                    ),
                    DemoScreen::EdgeCaseLab => build_edge_case_scene(layout, self.animation_time),
                };
                let scene_time = frame_started_at.elapsed();
                let command_count = scene.command_count();
                let camera = self.active_camera();
                if !self.uncapped {
                    window.pre_present_notify();
                }
                if let (Some(renderer), Some(glyph_atlas), Some(glyph_run)) = (
                    self.renderer.as_mut(),
                    self.glyph_probe_atlas.as_ref(),
                    self.glyph_probe_run.as_ref(),
                ) {
                    let report = renderer
                        .begin_frame(scene.background(), FrameBudget::default())
                        .and_then(|mut frame| {
                            frame.draw_scene(&scene, camera, FramePassOptions::default())?;
                            frame.draw_glyph_run(
                                glyph_atlas,
                                glyph_run,
                                ImageSampling::Nearest,
                                FramePassOptions::new(10_000),
                            )?;
                            frame.present()
                        });
                    match report {
                        Ok(report) => {
                            if let Some(title) = self.frame_metrics.record(
                                frame_started_at,
                                scene_time,
                                report.metrics(),
                                command_count,
                            ) {
                                window.set_title(&title);
                            }
                        }
                        Err(error) => eprintln!("UI demo renderer error: {error:?}"),
                    }
                }
                if self.uncapped {
                    window.request_redraw();
                    event_loop.set_control_flow(ControlFlow::Poll);
                } else {
                    self.next_frame_at = frame_started_at + TARGET_FRAME_INTERVAL;
                    event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame_at));
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.uncapped {
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Poll);
            }
            return;
        }
        let now = Instant::now();
        if now >= self.next_frame_at {
            self.next_frame_at = now + TARGET_FRAME_INTERVAL;
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame_at));
        }
        if self.window.is_none() {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

#[derive(Debug, Clone)]
struct WaveChallenge {
    target_frequencies: [f32; WAVE_COUNT],
    target_speeds: [f32; WAVE_COUNT],
    frequency_amounts: [f32; WAVE_COUNT],
    speed_amounts: [f32; WAVE_COUNT],
    amplitude_amounts: [f32; WAVE_COUNT],
}

impl WaveChallenge {
    fn random(random: &mut SimpleRandom) -> Self {
        let target_frequencies = [
            random.range(0.75, 1.35),
            random.range(4.05, 5.0),
            random.range(1.65, 2.65),
            random.range(2.75, 3.85),
        ];
        let amplitude_amounts = [
            random.range(0.17, 0.24),
            random.range(0.27, 0.34),
            random.range(0.36, 0.44),
            random.range(0.20, 0.29),
        ];
        let target_speeds = [
            random.range(0.35, 0.72),
            random.range(1.78, 2.28),
            random.range(0.82, 1.18),
            random.range(1.30, 1.68),
        ];
        let mut frequency_amounts = target_frequencies.map(frequency_to_amount);
        for amount in &mut frequency_amounts {
            let offset =
                random.range(0.13, 0.31) * if random.next_unit() < 0.5 { -1.0 } else { 1.0 };
            *amount = (*amount + offset).clamp(0.0, 1.0);
        }
        let mut speed_amounts = target_speeds.map(speed_to_amount);
        for amount in &mut speed_amounts {
            let offset =
                random.range(0.14, 0.30) * if random.next_unit() < 0.5 { -1.0 } else { 1.0 };
            *amount = (*amount + offset).clamp(0.0, 1.0);
        }
        Self {
            target_frequencies,
            target_speeds,
            frequency_amounts,
            speed_amounts,
            amplitude_amounts,
        }
    }

    fn frequency(&self, index: usize) -> f32 {
        amount_to_frequency(self.frequency_amounts[index])
    }

    fn speed(&self, index: usize) -> f32 {
        amount_to_speed(self.speed_amounts[index])
    }

    fn raw_accuracy(&self) -> f32 {
        let frequency_error = self
            .target_frequencies
            .iter()
            .enumerate()
            .map(|(index, target)| {
                (self.frequency(index) - target).abs() / (MAX_FREQUENCY - MIN_FREQUENCY)
            })
            .sum::<f32>()
            / WAVE_COUNT as f32;
        let speed_error = self
            .target_speeds
            .iter()
            .enumerate()
            .map(|(index, target)| (self.speed(index) - target).abs() / (MAX_SPEED - MIN_SPEED))
            .sum::<f32>()
            / WAVE_COUNT as f32;
        (1.0 - (frequency_error * 0.62 + speed_error * 0.38) * 3.2).clamp(0.0, 1.0)
    }

    fn displayed_accuracy(&self) -> u32 {
        (self.raw_accuracy() * 99.0).round().clamp(0.0, 99.0) as u32
    }

    fn is_solved(&self) -> bool {
        self.raw_accuracy() >= SUCCESS_ACCURACY
            && self
                .target_frequencies
                .iter()
                .enumerate()
                .all(|(index, target)| (self.frequency(index) - target).abs() <= 0.16)
            && self
                .target_speeds
                .iter()
                .enumerate()
                .all(|(index, target)| (self.speed(index) - target).abs() <= 0.10)
    }

    fn snap_to_target(&mut self) {
        self.frequency_amounts = self.target_frequencies.map(frequency_to_amount);
        self.speed_amounts = self.target_speeds.map(speed_to_amount);
    }

    #[cfg(test)]
    fn exact_test_challenge() -> Self {
        let target_frequencies = [1.0, 4.5, 2.2, 3.2];
        let target_speeds = [0.5, 2.0, 1.0, 1.5];
        Self {
            target_frequencies,
            target_speeds,
            frequency_amounts: target_frequencies.map(frequency_to_amount),
            speed_amounts: target_speeds.map(speed_to_amount),
            amplitude_amounts: [0.2, 0.3, 0.4, 0.25],
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PaneLayout {
    frame: Rect,
    plot: Rect,
    slider_start: Vec2,
    slider_end: Vec2,
}

#[derive(Debug, Clone, Copy)]
struct UiLayout {
    surface_size: Vec2,
    panes: [PaneLayout; WAVE_COUNT],
    playback_button: Rect,
    reset_button: Rect,
    speed_selector_buttons: [Rect; WAVE_COUNT],
    speed_start: Vec2,
    speed_end: Vec2,
    status_origin: Vec2,
    accept_button: Rect,
    dismiss_button: Rect,
}

impl UiLayout {
    fn new(surface_size: Vec2) -> Self {
        let margin = (surface_size.x().min(surface_size.y()) * 0.045).clamp(24.0, 38.0);
        let gap = 16.0;
        let bottom_height = 104.0;
        let grid_bottom = surface_size.y() - bottom_height - margin;
        let pane_width = (surface_size.x() - margin * 2.0 - gap) * 0.5;
        let pane_height = (grid_bottom - margin - gap) * 0.5;
        let panes = std::array::from_fn(|index| {
            let column = index % 2;
            let row = index / 2;
            let frame_min = Vec2::new(
                margin + column as f32 * (pane_width + gap),
                margin + row as f32 * (pane_height + gap),
            );
            let frame = Rect::from_min_size(frame_min, Vec2::new(pane_width, pane_height));
            let plot = Rect::new(
                frame.min() + Vec2::splat(12.0),
                Vec2::new(frame.max().x() - 12.0, frame.max().y() - 48.0),
            );
            let slider_y = frame.max().y() - 23.0;
            PaneLayout {
                frame,
                plot,
                slider_start: Vec2::new(frame.min().x() + 18.0, slider_y),
                slider_end: Vec2::new(frame.max().x() - 18.0, slider_y),
            }
        });
        let controls_y = grid_bottom + (surface_size.y() - margin - grid_bottom) * 0.5;
        let button_size = 54.0;
        let playback_button = Rect::from_center_size(
            Vec2::new(margin + button_size * 0.5, controls_y),
            Vec2::splat(button_size),
        );
        let reset_button = Rect::from_center_size(
            Vec2::new(
                playback_button.max().x() + 12.0 + button_size * 0.5,
                controls_y,
            ),
            Vec2::splat(button_size),
        );
        let selector_size = 32.0;
        let selector_gap = 7.0;
        let selector_start_x = reset_button.max().x() + 18.0;
        let speed_selector_buttons = std::array::from_fn(|index| {
            Rect::from_min_size(
                Vec2::new(
                    selector_start_x + index as f32 * (selector_size + selector_gap),
                    controls_y - selector_size * 0.5,
                ),
                Vec2::splat(selector_size),
            )
        });
        let speed_start = Vec2::new(speed_selector_buttons[3].max().x() + 18.0, controls_y);
        let speed_end = Vec2::new(surface_size.x() * 0.57, controls_y);
        let status_origin = Vec2::new(surface_size.x() * 0.61, controls_y - 14.0);
        let dismiss_button = Rect::from_center_size(
            Vec2::new(surface_size.x() - margin - 25.0, controls_y),
            Vec2::splat(50.0),
        );
        let accept_button = Rect::from_center_size(
            Vec2::new(dismiss_button.min().x() - 39.0, controls_y),
            Vec2::splat(50.0),
        );

        Self {
            surface_size,
            panes,
            playback_button,
            reset_button,
            speed_selector_buttons,
            speed_start,
            speed_end,
            status_origin,
            accept_button,
            dismiss_button,
        }
    }

    fn frequency_amount_at(self, index: usize, logical_x: f32) -> f32 {
        let pane = self.panes[index];
        normalized_track_amount(logical_x, pane.slider_start.x(), pane.slider_end.x())
    }

    fn speed_amount_at(self, logical_x: f32) -> f32 {
        normalized_track_amount(logical_x, self.speed_start.x(), self.speed_end.x())
    }

    fn frequency_knob(self, index: usize, amount: f32) -> Vec2 {
        let pane = self.panes[index];
        pane.slider_start
            .lerp(pane.slider_end, amount.clamp(0.0, 1.0))
    }

    fn speed_knob(self, amount: f32) -> Vec2 {
        self.speed_start
            .lerp(self.speed_end, amount.clamp(0.0, 1.0))
    }

    fn menu_panel(self) -> Rect {
        Rect::from_center_size(
            self.surface_size * 0.5,
            Vec2::new(
                (self.surface_size.x() - 64.0).min(720.0),
                (self.surface_size.y() - 64.0).min(600.0),
            ),
        )
    }

    fn menu_buttons(self) -> [(DemoScreen, Rect); 4] {
        let panel = self.menu_panel();
        let gap = 16.0;
        let button_width = (panel.width() - 96.0 - gap) * 0.5;
        let button_height = 72.0;
        let first_center = Vec2::new(
            panel.center().x() - (button_width + gap) * 0.5,
            panel.max().y() - 188.0,
        );
        let screens = [
            DemoScreen::FluidSimulation,
            DemoScreen::GasSimulation,
            DemoScreen::WaveSimulation,
            DemoScreen::EdgeCaseLab,
        ];
        std::array::from_fn(|index| {
            let column = index % 2;
            let row = index / 2;
            (
                screens[index],
                Rect::from_center_size(
                    first_center
                        + Vec2::new(
                            column as f32 * (button_width + gap),
                            row as f32 * (button_height + gap),
                        ),
                    Vec2::new(button_width, button_height),
                ),
            )
        })
    }

    fn menu_hit_test(self, logical_point: Vec2) -> Option<InteractionTarget> {
        for (screen, button) in self.menu_buttons() {
            if rect_contains(button, logical_point) {
                return Some(InteractionTarget::MenuSimulation(screen));
            }
        }
        None
    }

    fn hit_test(
        self,
        logical_point: Vec2,
        frequency_amounts: &[f32; WAVE_COUNT],
        speed_amount: f32,
        show_success: bool,
    ) -> Option<InteractionTarget> {
        if rect_contains(self.playback_button, logical_point) {
            return Some(InteractionTarget::PlaybackButton);
        }
        if rect_contains(self.reset_button, logical_point) {
            return Some(InteractionTarget::ResetButton);
        }
        for (index, selector) in self.speed_selector_buttons.iter().enumerate() {
            if rect_contains(*selector, logical_point) {
                return Some(InteractionTarget::SpeedSelector(index));
            }
        }
        for (index, pane) in self.panes.iter().enumerate() {
            if (logical_point - self.frequency_knob(index, frequency_amounts[index]))
                .length_squared()
                <= 17.0 * 17.0
                || track_contains(pane.slider_start, pane.slider_end, logical_point)
            {
                return Some(InteractionTarget::FrequencySlider(index));
            }
        }
        if (logical_point - self.speed_knob(speed_amount)).length_squared() <= 18.0 * 18.0
            || track_contains(self.speed_start, self.speed_end, logical_point)
        {
            return Some(InteractionTarget::SpeedSlider);
        }
        if show_success && rect_contains(self.accept_button, logical_point) {
            return Some(InteractionTarget::AcceptButton);
        }
        if show_success && rect_contains(self.dismiss_button, logical_point) {
            return Some(InteractionTarget::DismissButton);
        }
        None
    }

    fn screen_to_world(self, logical: Vec2) -> Vec2 {
        Vec2::new(
            logical.x() - self.surface_size.x() * 0.5,
            self.surface_size.y() * 0.5 - logical.y(),
        )
    }

    fn screen_rect_to_world(self, logical: Rect) -> Rect {
        Rect::new(
            self.screen_to_world(logical.max()),
            self.screen_to_world(logical.min()),
        )
        .normalized()
    }
}

fn build_menu_scene(layout: UiLayout, hovered: Option<InteractionTarget>) -> Scene {
    let background = Color::rgb8(12, 15, 19);
    let panel_color = Color::rgb8(25, 30, 36);
    let panel_border = Color::rgba8(255, 255, 255, 38);
    let muted = Color::rgba8(224, 232, 238, 150);
    let mut scene = Scene::new(background).expect("menu background is finite");
    let panel = layout.menu_panel();

    scene.rect(
        layout.screen_rect_to_world(panel),
        18.0,
        ShapeStyle::fill_stroke(panel_color, 1.5, panel_border),
    );

    draw_centered_pixel_text(
        &mut scene,
        layout,
        "SIM ENGINE",
        Vec2::new(panel.center().x(), panel.min().y() + 54.0),
        4.0,
        muted,
    );
    draw_centered_pixel_text(
        &mut scene,
        layout,
        "USE CASE LABS",
        Vec2::new(panel.center().x(), panel.min().y() + 96.0),
        7.0,
        Color::WHITE,
    );

    let wave_plot = Rect::new(
        Vec2::new(panel.min().x() + 42.0, panel.min().y() + 148.0),
        Vec2::new(panel.max().x() - 42.0, panel.max().y() - 246.0),
    );
    for (index, color) in WAVE_COLORS.into_iter().enumerate() {
        let center_y = wave_plot.min().y() + wave_plot.height() * (index as f32 + 0.5) / 4.0;
        let line_plot = Rect::from_center_size(
            Vec2::new(wave_plot.center().x(), center_y),
            Vec2::new(wave_plot.width(), wave_plot.height() * 0.22),
        );
        let points = wave_screen_points(
            line_plot,
            1.25 + index as f32 * 0.72,
            0.0,
            0.38,
            index as f32 * 0.7,
        )
        .into_iter()
        .map(|point| layout.screen_to_world(point))
        .collect();
        scene.polyline(points, 2.5, color.with_alpha(0.88));
    }

    let labels = [
        ("FLUID SIMULATION", WAVE_COLORS[0]),
        ("GAS SIMULATION", WAVE_COLORS[1]),
        ("WAVE LAB", WAVE_COLORS[2]),
        ("EDGE CASE LAB", WAVE_COLORS[3]),
    ];
    for (index, (screen, button)) in layout.menu_buttons().into_iter().enumerate() {
        let (label, button_color) = labels[index];
        let button_hovered = hovered == Some(InteractionTarget::MenuSimulation(screen));
        scene.rect(
            layout.screen_rect_to_world(button),
            12.0,
            ShapeStyle::fill_stroke(
                if button_hovered {
                    button_color
                } else {
                    button_color.with_alpha(0.15)
                },
                2.0,
                button_color,
            ),
        );
        draw_pixel_text(
            &mut scene,
            layout,
            &(index + 1).to_string(),
            button.min() + Vec2::new(12.0, 10.0),
            2.5,
            if button_hovered {
                background
            } else {
                button_color
            },
        );
        draw_centered_pixel_text(
            &mut scene,
            layout,
            label,
            button.center(),
            2.0,
            if button_hovered {
                background
            } else {
                button_color
            },
        );
    }

    draw_centered_pixel_text(
        &mut scene,
        layout,
        "1 2 3 4 OR CLICK",
        Vec2::new(panel.center().x(), panel.max().y() - 12.0),
        2.0,
        Color::rgba8(224, 232, 238, 105),
    );

    scene
}

fn build_fluid_scene(layout: UiLayout, animation_time: f32, playing: bool) -> Scene {
    let background = Color::rgb8(8, 18, 26);
    let water = Color::rgb8(28, 166, 201);
    let foam = Color::rgb8(154, 238, 244);
    let mut scene = Scene::new(background).expect("fluid background is finite");
    draw_lab_header(&mut scene, layout, "FLUID SIMULATION", water, playing);

    let tank = Rect::new(
        Vec2::new(42.0, 78.0),
        Vec2::new(
            layout.surface_size.x() - 42.0,
            layout.surface_size.y() - 42.0,
        ),
    );
    let tank_world = layout.screen_rect_to_world(tank);
    scene.rect(
        tank_world,
        18.0,
        ShapeStyle::filled_with(Fill::LinearGradient(LinearGradient::new(
            Vec2::new(tank_world.min().x(), tank_world.max().y()),
            Vec2::new(tank_world.max().x(), tank_world.min().y()),
            Color::rgb8(9, 37, 53),
            Color::rgb8(7, 24, 38),
        ))),
    );
    scene.rect(
        tank_world,
        18.0,
        ShapeStyle::stroked(2.0, water.with_alpha(0.55)),
    );

    let obstacle_center = Vec2::new(tank.center().x(), tank.center().y() + 12.0);
    let obstacle_radius = tank.height().min(tank.width()) * 0.115;
    scene.circle(
        layout.screen_to_world(obstacle_center),
        obstacle_radius,
        ShapeStyle::fill_stroke(Color::rgb8(18, 45, 58), 2.0, foam.with_alpha(0.7)),
    );

    for row in 0..5 {
        let row_amount = (row as f32 + 0.5) / 5.0;
        let base_y = tank.min().y() + tank.height() * row_amount;
        let mut points = Vec::with_capacity(36);
        for sample in 0..36 {
            let amount = sample as f32 / 35.0;
            let x = tank.min().x() + tank.width() * amount;
            let obstacle_dx = (x - obstacle_center.x()) / obstacle_radius;
            let obstacle_dy = (base_y - obstacle_center.y()) / obstacle_radius;
            let influence = (1.0 - obstacle_dx.abs()).max(0.0) * (1.0 - obstacle_dy.abs()).max(0.0);
            let deflection = obstacle_dy.signum() * influence * obstacle_radius * 0.72;
            let ripple = (amount * 14.0 + row as f32 * 0.61 - animation_time * 1.8).sin()
                * (3.0 + row_amount * 4.0);
            points.push(layout.screen_to_world(Vec2::new(x, base_y + deflection + ripple)));
        }
        scene.polyline(
            points,
            if row % 3 == 0 { 2.4 } else { 1.4 },
            water.with_alpha(0.32 + row_amount * 0.46),
        );
    }

    for index in 0..36 {
        let seed = index as f32 * 0.618_034;
        let x_amount = (seed + animation_time * (0.035 + (index % 7) as f32 * 0.004)).fract();
        let y_amount = (index as f32 * 0.371).fract();
        let point = Vec2::new(
            tank.min().x() + 18.0 + x_amount * (tank.width() - 36.0),
            tank.min().y()
                + 18.0
                + y_amount * (tank.height() - 36.0)
                + (animation_time + seed * 9.0).sin() * 5.0,
        );
        let size = 3.6 + (index % 4) as f32 * 1.1;
        scene.rect(
            layout.screen_rect_to_world(Rect::from_center_size(point, Vec2::splat(size))),
            0.0,
            ShapeStyle::filled(foam.with_alpha(0.35 + (index % 5) as f32 * 0.1)),
        );
    }

    scene
}

fn build_gas_scene(layout: UiLayout, animation_time: f32, playing: bool) -> Scene {
    let background = Color::rgb8(22, 12, 18);
    let hot = Color::rgb8(255, 101, 66);
    let cold = Color::rgb8(86, 195, 255);
    let mut scene = Scene::new(background).expect("gas background is finite");
    draw_lab_header(&mut scene, layout, "GAS SIMULATION", hot, playing);

    let chamber = Rect::new(
        Vec2::new(42.0, 78.0),
        Vec2::new(
            layout.surface_size.x() - 42.0,
            layout.surface_size.y() - 42.0,
        ),
    );
    scene.rect(
        layout.screen_rect_to_world(chamber),
        16.0,
        ShapeStyle::fill_stroke(
            Color::rgb8(34, 22, 29),
            2.0,
            Color::rgba8(255, 180, 150, 95),
        ),
    );
    let clip = ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(chamber.min().x(), chamber.min().y()),
        LogicalScreenVector::new(chamber.width(), chamber.height()),
    )
    .expect("gas chamber clip is valid");
    scene
        .with_screen_clip(clip, |scene| {
            for index in 0..180 {
                let seed = index as f32 + 1.0;
                let speed = 0.18 + (index % 17) as f32 * 0.013;
                let x_amount = ping_pong(seed * 0.754_877_7 + animation_time * speed);
                let y_amount = ping_pong(seed * 0.569_840_3 + animation_time * speed * 0.73);
                let position = Vec2::new(
                    chamber.min().x() + 8.0 + x_amount * (chamber.width() - 16.0),
                    chamber.min().y() + 8.0 + y_amount * (chamber.height() - 16.0),
                );
                let heat = (index % 29) as f32 / 28.0;
                let color = Color::rgba(
                    cold.red() + (hot.red() - cold.red()) * heat,
                    cold.green() + (hot.green() - cold.green()) * heat,
                    cold.blue() + (hot.blue() - cold.blue()) * heat,
                    0.48 + heat * 0.42,
                );
                let size = 3.6 + (index % 5) as f32 * 0.9;
                scene.rect(
                    layout
                        .screen_rect_to_world(Rect::from_center_size(position, Vec2::splat(size))),
                    0.0,
                    ShapeStyle::filled(color),
                );
            }
        })
        .expect("gas particles use a valid clip");

    scene
}

fn build_edge_case_scene(layout: UiLayout, animation_time: f32) -> Scene {
    let background = Color::rgb8(13, 13, 18);
    let accent = Color::rgb8(255, 105, 97);
    let mut scene = Scene::new(background).expect("edge-case background is finite");
    let panel = Rect::new(
        Vec2::new(34.0, 34.0),
        Vec2::new(
            layout.surface_size.x() - 34.0,
            layout.surface_size.y() - 34.0,
        ),
    );
    scene.rect(
        edge_screen_rect_to_world(layout, panel),
        16.0 / EDGE_CASE_ZOOM,
        ShapeStyle::fill_stroke(
            Color::rgb8(24, 25, 32),
            2.0,
            Color::rgba8(255, 255, 255, 40),
        ),
    );
    draw_edge_centered_text(
        &mut scene,
        layout,
        "EDGE CASE LAB X10000",
        Vec2::new(panel.center().x(), panel.min().y() + 34.0),
        3.0,
        Color::WHITE,
    );
    draw_edge_pixel_text(
        &mut scene,
        layout,
        "ESC MENU   SPACE PAUSE   R RESET",
        panel.min() + Vec2::new(18.0, 62.0),
        1.8,
        Color::rgba8(235, 240, 244, 145),
    );

    let short_line_center = edge_screen_to_world(
        layout,
        Vec2::new(panel.center().x(), panel.min().y() + 126.0),
    );
    scene.line(
        short_line_center - Vec2::new(0.0025, 0.0),
        short_line_center + Vec2::new(0.0025, 0.0),
        5.0,
        Color::rgb8(87, 203, 137),
    );
    draw_edge_centered_text(
        &mut scene,
        layout,
        "0.005 WORLD LINE",
        Vec2::new(panel.center().x(), panel.min().y() + 154.0),
        2.0,
        Color::rgb8(87, 203, 137),
    );

    let gradient_rect = Rect::from_min_size(
        Vec2::new(panel.min().x() + 48.0, panel.min().y() + 200.0),
        Vec2::new(panel.width() * 0.42, 104.0),
    );
    scene.rect(
        edge_screen_rect_to_world(layout, gradient_rect),
        14.0 / EDGE_CASE_ZOOM,
        ShapeStyle::filled_with(Fill::LinearGradient(LinearGradient::new(
            Vec2::new(-f32::MAX, 0.0),
            Vec2::new(f32::MAX, 0.0),
            cold_edge_color(),
            hot_edge_color(),
        ))),
    );
    draw_edge_centered_text(
        &mut scene,
        layout,
        "EXTREME FINITE GRADIENT",
        Vec2::new(gradient_rect.center().x(), gradient_rect.max().y() + 24.0),
        1.7,
        Color::rgba8(235, 240, 244, 170),
    );

    let clip_rect = Rect::from_min_size(
        Vec2::new(panel.center().x() + 40.0, panel.min().y() + 200.0),
        Vec2::new(panel.width() * 0.32, 104.0),
    );
    let clip = ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(clip_rect.min().x(), clip_rect.min().y()),
        LogicalScreenVector::new(clip_rect.width(), clip_rect.height()),
    )
    .expect("edge-case clip is valid");
    scene
        .with_screen_clip(clip, |scene| {
            let pulse = 58.0 + animation_time.sin() * 12.0;
            scene.circle(
                edge_screen_to_world(layout, clip_rect.min() + Vec2::new(14.0, 52.0)),
                pulse / EDGE_CASE_ZOOM,
                ShapeStyle::filled(accent.with_alpha(0.72)),
            );
        })
        .expect("edge-case clipping is valid");
    scene.rect(
        edge_screen_rect_to_world(layout, clip_rect),
        10.0 / EDGE_CASE_ZOOM,
        ShapeStyle::stroked(2.0, accent),
    );
    draw_edge_centered_text(
        &mut scene,
        layout,
        "LOGICAL CLIP",
        Vec2::new(clip_rect.center().x(), clip_rect.max().y() + 24.0),
        1.7,
        accent,
    );

    let depth_center = edge_screen_to_world(
        layout,
        Vec2::new(panel.center().x(), panel.max().y() - 105.0),
    );
    scene.circle(
        depth_center,
        34.0 / EDGE_CASE_ZOOM,
        ShapeStyle::filled(Color::rgb8(86, 195, 255).with_alpha(0.8)),
    );
    scene
        .with_depth(5.0, |scene| {
            scene.circle(
                depth_center,
                24.0 / EDGE_CASE_ZOOM,
                ShapeStyle::filled(Color::rgb8(255, 190, 94).with_alpha(0.9)),
            );
        })
        .expect("edge-case depth is finite");
    draw_edge_centered_text(
        &mut scene,
        layout,
        "DEPTH PROJECTION",
        Vec2::new(panel.center().x(), panel.max().y() - 36.0),
        2.0,
        Color::rgba8(235, 240, 244, 170),
    );

    scene
}

fn draw_lab_header(scene: &mut Scene, layout: UiLayout, title: &str, color: Color, playing: bool) {
    draw_pixel_text(scene, layout, title, Vec2::new(42.0, 28.0), 3.0, color);
    draw_pixel_text(
        scene,
        layout,
        if playing {
            "ESC MENU   SPACE PAUSE   R RESET"
        } else {
            "PAUSED   SPACE PLAY   ESC MENU"
        },
        Vec2::new(42.0, 52.0),
        1.7,
        Color::rgba8(235, 240, 244, 145),
    );
}

fn ping_pong(value: f32) -> f32 {
    let wrapped = value.rem_euclid(2.0);
    if wrapped <= 1.0 {
        wrapped
    } else {
        2.0 - wrapped
    }
}

fn edge_screen_to_world(layout: UiLayout, logical: Vec2) -> Vec2 {
    layout.screen_to_world(logical) / EDGE_CASE_ZOOM
}

fn edge_screen_rect_to_world(layout: UiLayout, logical: Rect) -> Rect {
    Rect::new(
        edge_screen_to_world(layout, logical.max()),
        edge_screen_to_world(layout, logical.min()),
    )
    .normalized()
}

fn cold_edge_color() -> Color {
    Color::rgb8(86, 195, 255)
}

fn hot_edge_color() -> Color {
    Color::rgb8(255, 105, 97)
}

#[allow(clippy::too_many_arguments)]
fn build_ui_scene(
    layout: UiLayout,
    challenge: &WaveChallenge,
    animation_time: f32,
    playing: bool,
    selected_speed_wave: usize,
    hovered: Option<InteractionTarget>,
    active_drag: Option<InteractionTarget>,
    show_success: bool,
) -> Scene {
    let background = Color::rgb8(16, 18, 22);
    let surface = Color::rgb8(29, 33, 38);
    let surface_hover = Color::rgb8(39, 45, 51);
    let border = Color::rgba8(255, 255, 255, 34);
    let target_color = Color::rgba8(218, 224, 229, 126);
    let muted_track = Color::rgba8(255, 255, 255, 42);
    let speed_color = Color::rgb8(255, 105, 97);
    let success_color = Color::rgb8(87, 203, 137);
    let mut scene = Scene::new(background).expect("background is finite");

    draw_pixel_text(
        &mut scene,
        layout,
        "ESC MENU",
        Vec2::new(12.0, 8.0),
        2.0,
        Color::rgba8(235, 240, 244, 150),
    );

    for (index, pane) in layout.panes.iter().copied().enumerate() {
        scene.rect(
            layout.screen_rect_to_world(pane.frame),
            7.0,
            ShapeStyle::fill_stroke(surface, 1.0, border),
        );
        draw_wave_pane(
            &mut scene,
            layout,
            pane,
            challenge,
            index,
            animation_time,
            target_color,
        );
        let slider_active = hovered == Some(InteractionTarget::FrequencySlider(index))
            || active_drag == Some(InteractionTarget::FrequencySlider(index));
        draw_slider(
            &mut scene,
            layout,
            pane.slider_start,
            pane.slider_end,
            challenge.frequency_amounts[index],
            WAVE_COLORS[index],
            slider_active,
            muted_track,
        );
    }

    let button_hovered = hovered == Some(InteractionTarget::PlaybackButton);
    scene.rect(
        layout.screen_rect_to_world(layout.playback_button),
        7.0,
        ShapeStyle::fill_stroke(
            if button_hovered {
                success_color
            } else {
                surface_hover
            },
            1.5,
            success_color.with_alpha(0.88),
        ),
    );
    draw_playback_icon(&mut scene, layout, playing, success_color, button_hovered);

    let reset_hovered = hovered == Some(InteractionTarget::ResetButton);
    scene.rect(
        layout.screen_rect_to_world(layout.reset_button),
        7.0,
        ShapeStyle::fill_stroke(
            if reset_hovered {
                speed_color
            } else {
                surface_hover
            },
            1.5,
            speed_color.with_alpha(0.88),
        ),
    );
    draw_reset_icon(&mut scene, layout, speed_color, reset_hovered);

    for (index, button) in layout.speed_selector_buttons.iter().copied().enumerate() {
        draw_speed_selector_button(
            &mut scene,
            layout,
            button,
            index,
            index == selected_speed_wave,
            hovered == Some(InteractionTarget::SpeedSelector(index)),
        );
    }

    let speed_active = hovered == Some(InteractionTarget::SpeedSlider)
        || active_drag == Some(InteractionTarget::SpeedSlider);
    draw_slider(
        &mut scene,
        layout,
        layout.speed_start,
        layout.speed_end,
        challenge.speed_amounts[selected_speed_wave],
        speed_color,
        speed_active,
        muted_track,
    );

    let status = if show_success {
        format!("YAY {}%", challenge.displayed_accuracy())
    } else {
        format!("ACC {}%", challenge.displayed_accuracy())
    };
    draw_pixel_text(
        &mut scene,
        layout,
        &status,
        layout.status_origin,
        4.0,
        if show_success {
            success_color
        } else {
            Color::rgba8(235, 240, 244, 210)
        },
    );

    if show_success {
        draw_confirmation_button(
            &mut scene,
            layout,
            layout.accept_button,
            true,
            hovered == Some(InteractionTarget::AcceptButton),
        );
        draw_confirmation_button(
            &mut scene,
            layout,
            layout.dismiss_button,
            false,
            hovered == Some(InteractionTarget::DismissButton),
        );
    }

    scene
}

#[allow(clippy::too_many_arguments)]
fn draw_wave_pane(
    scene: &mut Scene,
    layout: UiLayout,
    pane: PaneLayout,
    challenge: &WaveChallenge,
    index: usize,
    animation_time: f32,
    target_color: Color,
) {
    let center_y = pane.plot.center().y();
    scene.line(
        layout.screen_to_world(Vec2::new(pane.plot.min().x(), center_y)),
        layout.screen_to_world(Vec2::new(pane.plot.max().x(), center_y)),
        1.0,
        Color::rgba8(255, 255, 255, 34),
    );
    for division in 1..4 {
        let x = pane.plot.min().x() + pane.plot.width() * division as f32 / 4.0;
        scene.line(
            layout.screen_to_world(Vec2::new(x, pane.plot.min().y())),
            layout.screen_to_world(Vec2::new(x, pane.plot.max().y())),
            1.0,
            Color::rgba8(255, 255, 255, 18),
        );
    }

    for dash_index in 0..TARGET_DASH_COUNT {
        let start_amount = dash_index as f32 / TARGET_DASH_COUNT as f32;
        let end_amount = (dash_index as f32 + 0.58) / TARGET_DASH_COUNT as f32;
        let start = wave_screen_point(
            pane.plot,
            challenge.target_frequencies[index],
            challenge.target_speeds[index],
            challenge.amplitude_amounts[index],
            animation_time,
            start_amount,
        );
        let end = wave_screen_point(
            pane.plot,
            challenge.target_frequencies[index],
            challenge.target_speeds[index],
            challenge.amplitude_amounts[index],
            animation_time,
            end_amount,
        );
        scene.line(
            layout.screen_to_world(start),
            layout.screen_to_world(end),
            1.7,
            target_color,
        );
    }

    let actual = wave_screen_points(
        pane.plot,
        challenge.frequency(index),
        challenge.speed(index),
        challenge.amplitude_amounts[index],
        animation_time,
    )
    .into_iter()
    .map(|point| layout.screen_to_world(point))
    .collect();
    scene.polyline(actual, 3.0, WAVE_COLORS[index].with_alpha(0.94));
}

fn wave_screen_points(
    plot: Rect,
    frequency: f32,
    speed: f32,
    amplitude_amount: f32,
    animation_time: f32,
) -> Vec<Vec2> {
    let mut points = Vec::with_capacity(WAVE_SAMPLE_COUNT);
    for sample in 0..WAVE_SAMPLE_COUNT {
        let amount = sample as f32 / (WAVE_SAMPLE_COUNT - 1) as f32;
        points.push(wave_screen_point(
            plot,
            frequency,
            speed,
            amplitude_amount,
            animation_time,
            amount,
        ));
    }
    points
}

fn wave_screen_point(
    plot: Rect,
    frequency: f32,
    speed: f32,
    amplitude_amount: f32,
    animation_time: f32,
    amount: f32,
) -> Vec2 {
    let horizontal_padding = 10.0;
    let wave_width = plot.width() - horizontal_padding * 2.0;
    let amplitude = plot.height() * amplitude_amount;
    let x = plot.min().x() + horizontal_padding + wave_width * amount;
    let envelope = (amount * std::f32::consts::PI).sin().max(0.0).powf(0.18);
    let y = plot.center().y()
        + (amount * std::f32::consts::TAU * frequency + animation_time * speed).sin()
            * amplitude
            * envelope;
    Vec2::new(x, y)
}

#[allow(clippy::too_many_arguments)]
fn draw_slider(
    scene: &mut Scene,
    layout: UiLayout,
    start: Vec2,
    end: Vec2,
    amount: f32,
    color: Color,
    active: bool,
    muted_track: Color,
) {
    scene.line(
        layout.screen_to_world(start),
        layout.screen_to_world(end),
        5.0,
        muted_track,
    );
    scene.line(
        layout.screen_to_world(start),
        layout.screen_to_world(start.lerp(end, amount.clamp(0.0, 1.0))),
        5.0,
        color.with_alpha(0.90),
    );
    scene.circle(
        layout.screen_to_world(start.lerp(end, amount.clamp(0.0, 1.0))),
        if active { 10.5 } else { 8.5 },
        ShapeStyle::fill_stroke(color, 1.5, Color::WHITE.with_alpha(0.76)),
    );
}

fn draw_playback_icon(
    scene: &mut Scene,
    layout: UiLayout,
    playing: bool,
    color: Color,
    highlighted: bool,
) {
    let center = layout.playback_button.center();
    let icon_color = if highlighted {
        Color::rgb8(17, 24, 26)
    } else {
        color
    };
    if playing {
        for offset in [-5.0, 5.0] {
            scene.line(
                layout.screen_to_world(center + Vec2::new(offset, -8.0)),
                layout.screen_to_world(center + Vec2::new(offset, 8.0)),
                4.0,
                icon_color,
            );
        }
    } else {
        scene.polyline(
            vec![
                layout.screen_to_world(center + Vec2::new(-6.0, -9.0)),
                layout.screen_to_world(center + Vec2::new(9.0, 0.0)),
                layout.screen_to_world(center + Vec2::new(-6.0, 9.0)),
                layout.screen_to_world(center + Vec2::new(-6.0, -9.0)),
            ],
            3.0,
            icon_color,
        );
    }
}

fn draw_reset_icon(scene: &mut Scene, layout: UiLayout, color: Color, highlighted: bool) {
    let center = layout.reset_button.center();
    scene.rect(
        layout.screen_rect_to_world(Rect::from_center_size(center, Vec2::splat(16.0))),
        2.0,
        ShapeStyle::filled(if highlighted {
            Color::rgb8(26, 20, 22)
        } else {
            color
        }),
    );
}

fn draw_speed_selector_button(
    scene: &mut Scene,
    layout: UiLayout,
    rect: Rect,
    index: usize,
    selected: bool,
    hovered: bool,
) {
    let color = WAVE_COLORS[index];
    let fill = if selected {
        color.with_alpha(0.34)
    } else if hovered {
        color.with_alpha(0.18)
    } else {
        Color::rgb8(39, 45, 51)
    };
    scene.rect(
        layout.screen_rect_to_world(rect),
        5.0,
        ShapeStyle::fill_stroke(fill, if selected { 2.0 } else { 1.0 }, color),
    );
    let text_origin = rect.center() + Vec2::new(-6.0, -10.0);
    draw_pixel_text(
        scene,
        layout,
        &(index + 1).to_string(),
        text_origin,
        4.0,
        if selected { Color::WHITE } else { color },
    );
}

fn draw_confirmation_button(
    scene: &mut Scene,
    layout: UiLayout,
    rect: Rect,
    accepted: bool,
    hovered: bool,
) {
    let color = if accepted {
        Color::rgb8(87, 203, 137)
    } else {
        Color::rgb8(255, 105, 97)
    };
    scene.rect(
        layout.screen_rect_to_world(rect),
        7.0,
        ShapeStyle::fill_stroke(
            if hovered {
                color
            } else {
                Color::rgb8(39, 45, 51)
            },
            1.5,
            color,
        ),
    );
    let center = rect.center();
    let icon = if hovered {
        Color::rgb8(17, 24, 26)
    } else {
        color
    };
    if accepted {
        scene.polyline(
            vec![
                layout.screen_to_world(center + Vec2::new(-9.0, 0.0)),
                layout.screen_to_world(center + Vec2::new(-2.0, 7.0)),
                layout.screen_to_world(center + Vec2::new(10.0, -8.0)),
            ],
            4.0,
            icon,
        );
    } else {
        for direction in [-1.0, 1.0] {
            scene.line(
                layout.screen_to_world(center + Vec2::new(-8.0, 8.0 * direction)),
                layout.screen_to_world(center + Vec2::new(8.0, -8.0 * direction)),
                4.0,
                icon,
            );
        }
    }
}

fn draw_pixel_text(
    scene: &mut Scene,
    layout: UiLayout,
    text: &str,
    origin: Vec2,
    pixel_size: f32,
    color: Color,
) {
    let mut cursor_x = origin.x();
    for character in text.chars() {
        let pattern = glyph_pattern(character);
        for (row, bits) in pattern.into_iter().enumerate() {
            for column in 0..3 {
                if bits & (1 << (2 - column)) != 0 {
                    let min = Vec2::new(
                        cursor_x + column as f32 * pixel_size,
                        origin.y() + row as f32 * pixel_size,
                    );
                    scene.rect(
                        layout.screen_rect_to_world(Rect::from_min_size(
                            min,
                            Vec2::splat(pixel_size - 0.7),
                        )),
                        0.0,
                        ShapeStyle::filled(color),
                    );
                }
            }
        }
        cursor_x += pixel_size * 4.0;
    }
}

fn draw_centered_pixel_text(
    scene: &mut Scene,
    layout: UiLayout,
    text: &str,
    center: Vec2,
    pixel_size: f32,
    color: Color,
) {
    let glyph_count = text.chars().count();
    let width_in_pixels = glyph_count
        .checked_sub(1)
        .map_or(0, |spacing_count| spacing_count * 4)
        + 3;
    let text_size = Vec2::new(width_in_pixels as f32 * pixel_size, 5.0 * pixel_size);
    draw_pixel_text(
        scene,
        layout,
        text,
        center - text_size * 0.5,
        pixel_size,
        color,
    );
}

fn draw_edge_pixel_text(
    scene: &mut Scene,
    layout: UiLayout,
    text: &str,
    origin: Vec2,
    pixel_size: f32,
    color: Color,
) {
    let mut cursor_x = origin.x();
    for character in text.chars() {
        let pattern = glyph_pattern(character);
        for (row, bits) in pattern.into_iter().enumerate() {
            for column in 0..3 {
                if bits & (1 << (2 - column)) != 0 {
                    let min = Vec2::new(
                        cursor_x + column as f32 * pixel_size,
                        origin.y() + row as f32 * pixel_size,
                    );
                    scene.rect(
                        edge_screen_rect_to_world(
                            layout,
                            Rect::from_min_size(min, Vec2::splat(pixel_size - 0.35)),
                        ),
                        0.0,
                        ShapeStyle::filled(color),
                    );
                }
            }
        }
        cursor_x += pixel_size * 4.0;
    }
}

fn draw_edge_centered_text(
    scene: &mut Scene,
    layout: UiLayout,
    text: &str,
    center: Vec2,
    pixel_size: f32,
    color: Color,
) {
    let glyph_count = text.chars().count();
    let width_in_pixels = glyph_count
        .checked_sub(1)
        .map_or(0, |spacing_count| spacing_count * 4)
        + 3;
    let text_size = Vec2::new(width_in_pixels as f32 * pixel_size, 5.0 * pixel_size);
    draw_edge_pixel_text(
        scene,
        layout,
        text,
        center - text_size * 0.5,
        pixel_size,
        color,
    );
}

fn create_scientific_glyph_probe(
    renderer: &WgpuRenderer,
) -> Result<(GlyphAtlas2d, GlyphRun2d), Box<dyn Error>> {
    let glyphs = [
        ('μ', [0b000, 0b000, 0b101, 0b101, 0b111]),
        ('Δ', [0b010, 0b101, 0b101, 0b111, 0b000]),
        ('∫', [0b011, 0b010, 0b010, 0b010, 0b110]),
        ('Σ', [0b111, 0b100, 0b010, 0b100, 0b111]),
        ('Ж', [0b101, 0b111, 0b010, 0b111, 0b101]),
    ];
    let width = glyphs.len() as u32 * 3;
    let height = 5;
    let mut pixels = vec![0_u8; width as usize * height as usize * 4];
    let mut entries = Vec::with_capacity(glyphs.len());
    for (glyph_index, (character, pattern)) in glyphs.into_iter().enumerate() {
        let source_x = glyph_index as u32 * 3;
        entries.push(GlyphAtlasEntry::new(
            GlyphId::new(character as u32),
            ImageTexelRect::new(source_x, 0, 3, height)?,
        ));
        for (row, bits) in pattern.into_iter().enumerate() {
            for column in 0..3 {
                if bits & (1 << (2 - column)) == 0 {
                    continue;
                }
                let x = source_x as usize + column;
                let offset = (row * width as usize + x) * 4;
                pixels[offset..offset + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
    }
    let pixel_bytes = pixels.len();
    let image_budget = ImageBudget::new(width, height, pixel_bytes)?;
    let atlas_budget = GlyphAtlasBudget::new(
        image_budget,
        glyphs.len(),
        glyphs.len() * std::mem::size_of::<GlyphAtlasEntry>(),
    )?;
    let atlas = renderer.create_glyph_atlas(width, height, pixels, entries, atlas_budget)?;
    let colors = [
        WAVE_COLORS[0],
        WAVE_COLORS[1],
        WAVE_COLORS[2],
        WAVE_COLORS[3],
        Color::WHITE,
    ];
    let mut positioned = Vec::with_capacity(glyphs.len());
    for (index, ((character, _), color)) in glyphs.into_iter().zip(colors).enumerate() {
        let destination = LogicalViewportRegion::new(
            LogicalScreenPosition::new(18.0 + index as f32 * 16.0, 18.0),
            LogicalViewport::new(12.0, 20.0)?,
        )?;
        positioned.push(PositionedGlyph2d::new(
            GlyphId::new(character as u32),
            destination,
            color,
        )?);
    }
    let run =
        renderer.create_glyph_run(&atlas, positioned, GlyphRunBudget::new(glyphs.len(), 4096)?)?;
    Ok((atlas, run))
}

fn glyph_pattern(character: char) -> [u8; 5] {
    match character {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b111, 0b100, 0b100, 0b100, 0b111],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b111, 0b100, 0b101, 0b101, 0b111],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b111],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b101, 0b111, 0b111, 0b111, 0b101],
        'O' => [0b111, 0b101, 0b101, 0b101, 0b111],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'Q' => [0b111, 0b101, 0b101, 0b111, 0b001],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b111, 0b100, 0b111, 0b001, 0b111],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '%' => [0b101, 0b001, 0b010, 0b100, 0b101],
        _ => [0; 5],
    }
}

fn amount_to_frequency(amount: f32) -> f32 {
    MIN_FREQUENCY + (MAX_FREQUENCY - MIN_FREQUENCY) * amount.clamp(0.0, 1.0)
}

fn frequency_to_amount(frequency: f32) -> f32 {
    ((frequency - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY)).clamp(0.0, 1.0)
}

fn amount_to_speed(amount: f32) -> f32 {
    MIN_SPEED + (MAX_SPEED - MIN_SPEED) * amount.clamp(0.0, 1.0)
}

fn speed_to_amount(speed: f32) -> f32 {
    ((speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)).clamp(0.0, 1.0)
}

fn normalized_track_amount(value: f32, start: f32, end: f32) -> f32 {
    ((value - start) / (end - start)).clamp(0.0, 1.0)
}

fn track_contains(start: Vec2, end: Vec2, point: Vec2) -> bool {
    rect_contains(
        Rect::new(start - Vec2::new(10.0, 14.0), end + Vec2::new(10.0, 14.0)),
        point,
    )
}

fn rect_contains(rect: Rect, point: Vec2) -> bool {
    let rect = rect.normalized();
    point.x() >= rect.min().x()
        && point.x() <= rect.max().x()
        && point.y() >= rect.min().y()
        && point.y() <= rect.max().y()
}

fn solved_preview_requested() -> bool {
    std::env::args().any(|argument| argument == "--solved-preview")
        || std::env::var("SIM_ENGINE_SOLVED_PREVIEW").is_ok_and(|value| {
            value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
        })
}

fn requested_initial_screen() -> Option<DemoScreen> {
    std::env::args().find_map(|argument| match argument.as_str() {
        "--screen=fluid" => Some(DemoScreen::FluidSimulation),
        "--screen=gas" => Some(DemoScreen::GasSimulation),
        "--screen=wave" => Some(DemoScreen::WaveSimulation),
        "--screen=edge" => Some(DemoScreen::EdgeCaseLab),
        _ => None,
    })
}

fn uncapped_requested() -> bool {
    std::env::args().any(|argument| argument == "--uncapped" || argument == "--benchmark")
}

struct SimpleRandom {
    state: u64,
}

impl SimpleRandom {
    fn from_system_time() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos() as u64);
        let seed = nanos ^ (std::process::id() as u64).rotate_left(19);
        Self { state: seed.max(1) }
    }

    fn next_unit(&mut self) -> f32 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        (value as u32) as f32 / u32::MAX as f32
    }

    fn range(&mut self, minimum: f32, maximum: f32) -> f32 {
        minimum + (maximum - minimum) * self.next_unit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_engine::DrawCommand;

    #[test]
    fn track_amount_clamps_to_interactive_range() {
        assert_eq!(normalized_track_amount(-20.0, 10.0, 110.0), 0.0);
        assert_eq!(normalized_track_amount(60.0, 10.0, 110.0), 0.5);
        assert_eq!(normalized_track_amount(160.0, 10.0, 110.0), 1.0);
    }

    #[test]
    fn menu_buttons_have_distinct_navigation_targets() {
        let layout = UiLayout::new(Vec2::new(1100.0, 760.0));
        let expected = [
            DemoScreen::FluidSimulation,
            DemoScreen::GasSimulation,
            DemoScreen::WaveSimulation,
            DemoScreen::EdgeCaseLab,
        ];

        for ((screen, button), expected_screen) in layout.menu_buttons().into_iter().zip(expected) {
            assert_eq!(screen, expected_screen);
            assert_eq!(
                layout.menu_hit_test(button.center()),
                Some(InteractionTarget::MenuSimulation(expected_screen))
            );
        }
        assert!(build_menu_scene(layout, None).command_count() > 20);
    }

    #[test]
    fn keyboard_navigation_enters_simulation_and_escape_returns_to_menu() {
        let mut application = UiDemoApplication::new();
        assert_eq!(application.screen, DemoScreen::Menu);

        application.handle_key(KeyCode::Escape);
        assert_eq!(application.screen, DemoScreen::Menu);

        application.handle_key(KeyCode::Enter);
        assert_eq!(application.screen, DemoScreen::FluidSimulation);

        application.animation_time = 3.5;
        application.handle_key(KeyCode::Space);
        assert!(!application.playing);
        application.handle_key(KeyCode::KeyR);
        assert_eq!(application.animation_time, 0.0);

        application.handle_key(KeyCode::Escape);
        assert_eq!(application.screen, DemoScreen::Menu);

        application.handle_key(KeyCode::Digit4);
        assert_eq!(application.screen, DemoScreen::EdgeCaseLab);
        application.handle_key(KeyCode::Escape);
        application.handle_key(KeyCode::Digit3);
        assert_eq!(application.screen, DemoScreen::WaveSimulation);
    }

    #[test]
    fn showcase_scenes_exercise_dense_clip_and_extreme_geometry_paths() {
        let layout = UiLayout::new(Vec2::new(1100.0, 760.0));
        let fluid = build_fluid_scene(layout, 1.25, true);
        let gas = build_gas_scene(layout, 1.25, true);
        let edge = build_edge_case_scene(layout, 1.25);

        assert!(fluid.command_count() > 80);
        assert!(gas.command_count() > 180);
        assert!(
            gas.commands()
                .iter()
                .filter_map(|command| command.screen_clip())
                .count()
                >= 180
        );
        assert!(edge.commands().iter().any(|command| {
            if let DrawCommand::Line(line) = command.command() {
                ((line.to() - line.from()).length() - 0.005).abs() < 0.000_001
            } else {
                false
            }
        }));
    }

    #[test]
    fn four_frequency_sliders_have_distinct_hit_targets() {
        let layout = UiLayout::new(Vec2::new(1100.0, 760.0));
        let amounts = [0.2, 0.4, 0.6, 0.8];

        for (index, amount) in amounts.into_iter().enumerate() {
            assert_eq!(
                layout.hit_test(layout.frequency_knob(index, amount), &amounts, 0.5, false,),
                Some(InteractionTarget::FrequencySlider(index))
            );
        }
        assert_eq!(
            layout.hit_test(layout.reset_button.center(), &amounts, 0.5, false),
            Some(InteractionTarget::ResetButton)
        );
        for (index, button) in layout.speed_selector_buttons.iter().enumerate() {
            assert_eq!(
                layout.hit_test(button.center(), &amounts, 0.5, false),
                Some(InteractionTarget::SpeedSelector(index))
            );
        }
    }

    #[test]
    fn exact_match_is_successful_but_display_is_capped_below_one_hundred() {
        let challenge = WaveChallenge::exact_test_challenge();

        assert!(challenge.is_solved());
        assert_eq!(challenge.displayed_accuracy(), 99);
    }

    #[test]
    fn randomized_challenge_can_snap_every_speed_and_frequency_to_target() {
        let mut random = SimpleRandom { state: 17 };
        let mut challenge = WaveChallenge::random(&mut random);
        challenge.snap_to_target();

        assert!(challenge.is_solved());
        assert_eq!(challenge.displayed_accuracy(), 99);
    }

    #[test]
    fn changing_frequency_changes_only_the_selected_wave_geometry() {
        let layout = UiLayout::new(Vec2::new(1100.0, 760.0));
        let challenge = WaveChallenge::exact_test_challenge();
        let original = build_ui_scene(layout, &challenge, 0.0, false, 0, None, None, false);
        let mut changed = challenge.clone();
        changed.frequency_amounts[2] = 0.9;
        let changed_scene = build_ui_scene(layout, &changed, 0.0, false, 0, None, None, false);
        let original_waves = actual_wave_points(&original);
        let changed_waves = actual_wave_points(&changed_scene);

        assert_eq!(original_waves.len(), WAVE_COUNT);
        assert_eq!(changed_waves.len(), WAVE_COUNT);
        assert_eq!(original_waves[0], changed_waves[0]);
        assert_eq!(original_waves[1], changed_waves[1]);
        assert_ne!(original_waves[2], changed_waves[2]);
        assert_eq!(original_waves[3], changed_waves[3]);
    }

    #[test]
    fn changing_speed_changes_only_the_selected_wave_geometry() {
        let layout = UiLayout::new(Vec2::new(1100.0, 760.0));
        let challenge = WaveChallenge::exact_test_challenge();
        let original = build_ui_scene(layout, &challenge, 0.8, true, 1, None, None, false);
        let mut changed = challenge.clone();
        changed.speed_amounts[1] = 0.05;
        let changed_scene = build_ui_scene(layout, &changed, 0.8, true, 1, None, None, false);
        let original_waves = actual_wave_points(&original);
        let changed_waves = actual_wave_points(&changed_scene);

        assert_eq!(original_waves[0], changed_waves[0]);
        assert_ne!(original_waves[1], changed_waves[1]);
        assert_eq!(original_waves[2], changed_waves[2]);
        assert_eq!(original_waves[3], changed_waves[3]);
    }

    #[test]
    fn target_and_actual_share_pause_and_hard_reset_timing() {
        let plot = Rect::from_min_size(Vec2::ZERO, Vec2::new(400.0, 120.0));
        let target_at_start = wave_screen_points(plot, 2.0, 1.4, 0.3, 0.0);
        let target_later = wave_screen_points(plot, 2.0, 1.4, 0.3, 0.7);
        assert_ne!(target_at_start, target_later);

        let mut application = UiDemoApplication::new();
        application.animation_time = 3.75;
        application.playing = false;
        application.previous_frame = Instant::now();
        application
            .update_animation(application.previous_frame + std::time::Duration::from_secs(1));
        assert_eq!(application.animation_time, 3.75);

        application.hard_reset();

        assert_eq!(application.animation_time, 0.0);
        assert!(!application.playing);
        assert_eq!(
            wave_screen_points(plot, 2.0, 1.4, 0.3, application.animation_time),
            target_at_start
        );
    }

    fn actual_wave_points(scene: &Scene) -> Vec<&[Vec2]> {
        scene
            .commands()
            .iter()
            .filter_map(|command| {
                if let DrawCommand::Polyline(polyline) = command.command()
                    && polyline.points().len() == WAVE_SAMPLE_COUNT
                {
                    return Some(polyline.points());
                }
                None
            })
            .collect()
    }
}
