//! Interactive visual oracle for the v0.2 stroke and frame-composition paths.
//!
//! Controls: 1-4 pages, Space pause, Left/Right scrub dash phase, +/- zoom,
//! R reset, Esc exit. Pass `--uncapped` to inspect throughput without FIFO pacing.

use std::{error::Error, sync::Arc, time::Instant};

use sim_engine::{
    Camera2d, Color, FrameBudget, FramePassOptions, LogicalPixels, LogicalScreenPosition,
    LogicalScreenVector, RendererPresentMode, Scene, ScreenScene, ShapeStyle, StrokeCap2d,
    StrokeDashPattern2d, StrokeJoin2d, StrokeMarker2d, StrokeStyle2d, Vec2, WgpuRenderer,
    WgpuRendererOptions,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let uncapped = arguments.iter().any(|argument| argument == "--uncapped");
    let initial_page = arguments
        .windows(2)
        .find(|pair| pair[0] == "--page")
        .and_then(|pair| match pair[1].as_str() {
            "1" => Some(GalleryPage::CapsAndJoins),
            "2" => Some(GalleryPage::AlphaContract),
            "3" => Some(GalleryPage::DashesAndMarkers),
            "4" => Some(GalleryPage::CameraAndEdges),
            _ => None,
        })
        .unwrap_or(GalleryPage::CapsAndJoins);
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut application = GalleryApplication::new(uncapped, initial_page);
    event_loop.run_app(&mut application)?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GalleryPage {
    CapsAndJoins,
    AlphaContract,
    DashesAndMarkers,
    CameraAndEdges,
}

impl GalleryPage {
    const fn number(self) -> u8 {
        match self {
            Self::CapsAndJoins => 1,
            Self::AlphaContract => 2,
            Self::DashesAndMarkers => 3,
            Self::CameraAndEdges => 4,
        }
    }

    const fn title(self) -> &'static str {
        match self {
            Self::CapsAndJoins => "caps + bevel/miter/round joins",
            Self::AlphaContract => "translucent joins: body and corner must match",
            Self::DashesAndMarkers => "bounded dashes + trimmed arrow markers",
            Self::CameraAndEdges => "camera stress + short accepted geometry",
        }
    }
}

struct GalleryApplication {
    window: Option<Arc<Window>>,
    renderer: Option<WgpuRenderer>,
    camera: Camera2d,
    page: GalleryPage,
    paused: bool,
    uncapped: bool,
    animation: f32,
    last_frame: Instant,
    metrics_started: Instant,
    metric_frames: usize,
    metric_work_seconds: f64,
    metric_acquire_seconds: f64,
}

impl GalleryApplication {
    fn new(uncapped: bool, page: GalleryPage) -> Self {
        let now = Instant::now();
        Self {
            window: None,
            renderer: None,
            camera: Camera2d::new(Vec2::ZERO, 10_000.0).expect("gallery camera is valid"),
            page,
            paused: false,
            uncapped,
            animation: 0.0,
            last_frame: now,
            metrics_started: now,
            metric_frames: 0,
            metric_work_seconds: 0.0,
            metric_acquire_seconds: 0.0,
        }
    }

    fn reset(&mut self) {
        self.animation = 0.0;
        self.paused = false;
        self.camera.set_zoom(10_000.0).expect("reset zoom is valid");
        self.camera
            .set_rotation(0.0)
            .expect("reset rotation is valid");
    }

    fn update_title(&mut self, now: Instant, work: f64, acquire: f64) {
        self.metric_frames = self.metric_frames.saturating_add(1);
        self.metric_work_seconds += work;
        self.metric_acquire_seconds += acquire;
        let elapsed = now.saturating_duration_since(self.metrics_started);
        if elapsed.as_secs_f64() < 1.0 {
            return;
        }
        let frames = self.metric_frames.max(1) as f64;
        let fps = frames / elapsed.as_secs_f64();
        let work_ms = self.metric_work_seconds * 1_000.0 / frames;
        let acquire_ms = self.metric_acquire_seconds * 1_000.0 / frames;
        if let Some(window) = &self.window {
            window.set_title(&format!(
                "Sim;Engine stroke gallery {} | {} | {fps:.1} FPS | work {work_ms:.3} ms | acquire {acquire_ms:.3} ms",
                self.page.number(),
                self.page.title(),
            ));
        }
        println!(
            "gallery page={} fps={fps:.1} renderer_work_ms={work_ms:.3} acquire_ms={acquire_ms:.3}",
            self.page.number()
        );
        self.metrics_started = now;
        self.metric_frames = 0;
        self.metric_work_seconds = 0.0;
        self.metric_acquire_seconds = 0.0;
    }
}

impl ApplicationHandler for GalleryApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Sim;Engine stroke gallery")
                        .with_inner_size(LogicalSize::new(1280.0, 760.0)),
                )
                .expect("create gallery window"),
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
            Arc::clone(&window),
            size.width.max(1),
            size.height.max(1),
            options,
        ))
        .expect("create gallery renderer");
        println!(
            "stroke gallery: 1 caps/joins, 2 alpha, 3 dash/markers, 4 camera edge cases; Space pause, Left/Right scrub, +/- zoom, R reset, Esc exit"
        );
        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref().map(Arc::clone) else {
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
                        .resize_with_scale_factor(
                            size.width.max(1),
                            size.height.max(1),
                            window.scale_factor(),
                        )
                        .expect("gallery resize is valid");
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(renderer) = self.renderer.as_mut() {
                    let size = window.inner_size();
                    renderer
                        .resize_with_scale_factor(
                            size.width.max(1),
                            size.height.max(1),
                            scale_factor,
                        )
                        .expect("gallery scale transition is valid");
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                match code {
                    KeyCode::Escape => event_loop.exit(),
                    KeyCode::Digit1 => self.page = GalleryPage::CapsAndJoins,
                    KeyCode::Digit2 => self.page = GalleryPage::AlphaContract,
                    KeyCode::Digit3 => self.page = GalleryPage::DashesAndMarkers,
                    KeyCode::Digit4 => self.page = GalleryPage::CameraAndEdges,
                    KeyCode::Space => self.paused = !self.paused,
                    KeyCode::ArrowLeft => self.animation -= 0.15,
                    KeyCode::ArrowRight => self.animation += 0.15,
                    KeyCode::Equal | KeyCode::NumpadAdd => {
                        let zoom = (self.camera.zoom() * 1.25).min(100_000.0);
                        self.camera.set_zoom(zoom).expect("bounded zoom is valid");
                    }
                    KeyCode::Minus | KeyCode::NumpadSubtract => {
                        let zoom = (self.camera.zoom() / 1.25).max(1.0);
                        self.camera.set_zoom(zoom).expect("bounded zoom is valid");
                    }
                    KeyCode::KeyR => self.reset(),
                    _ => {}
                }
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.draw_frame() {
                    eprintln!("stroke gallery frame failed: {error}");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl GalleryApplication {
    fn draw_frame(&mut self) -> Result<(), Box<dyn Error>> {
        let now = Instant::now();
        let delta = now.saturating_duration_since(self.last_frame);
        self.last_frame = now;
        if !self.paused {
            self.animation += delta.as_secs_f32();
        }
        if self.page == GalleryPage::CameraAndEdges {
            self.camera
                .set_rotation((self.animation * 0.35).sin() * 0.22)?;
        }
        let renderer = self
            .renderer
            .as_mut()
            .ok_or("renderer is not initialized")?;
        let (width, height) = renderer.logical_size();
        let screen = build_screen_page(self.page, width, height, self.animation)?;
        let world = (self.page == GalleryPage::CameraAndEdges)
            .then(|| build_world_edge_scene())
            .transpose()?;
        let mut frame = renderer.begin_frame(screen.background(), FrameBudget::default())?;
        frame.draw_screen_scene(&screen, FramePassOptions::new(0))?;
        if let Some(world) = &world {
            frame.draw_scene(world, self.camera, FramePassOptions::new(1))?;
        }
        let report = frame.present()?;
        let metrics = report.metrics();
        self.update_title(
            now,
            metrics
                .total_cpu()
                .saturating_sub(metrics.surface_acquire())
                .as_secs_f64(),
            metrics.surface_acquire().as_secs_f64(),
        );
        Ok(())
    }
}

fn build_screen_page(
    page: GalleryPage,
    width: f32,
    height: f32,
    animation: f32,
) -> Result<ScreenScene, Box<dyn Error>> {
    let mut scene = ScreenScene::new(Color::rgb8(8, 11, 17))?;
    scene.try_rect(
        p(28.0, 28.0),
        LogicalScreenVector::new((width - 56.0).max(1.0), (height - 56.0).max(1.0)),
        px(18.0),
        ShapeStyle::fill_stroke(Color::rgb8(15, 21, 31), 1.0, Color::rgb8(52, 67, 84)),
    )?;
    match page {
        GalleryPage::CapsAndJoins => draw_caps_and_joins(&mut scene, width, height)?,
        GalleryPage::AlphaContract => draw_alpha_contract(&mut scene, width, height)?,
        GalleryPage::DashesAndMarkers => {
            draw_dashes_and_markers(&mut scene, width, height, animation)?
        }
        GalleryPage::CameraAndEdges => draw_camera_frame(&mut scene, width, height)?,
    }
    Ok(scene)
}

fn draw_caps_and_joins(
    scene: &mut ScreenScene,
    width: f32,
    _height: f32,
) -> Result<(), Box<dyn Error>> {
    let left = 110.0;
    let middle = width * 0.46;
    let right = width - 110.0;
    for (index, (cap, color)) in [
        (StrokeCap2d::Butt, Color::rgb8(65, 196, 255)),
        (StrokeCap2d::Square, Color::rgb8(255, 176, 70)),
        (StrokeCap2d::Round, Color::rgb8(106, 224, 142)),
    ]
    .into_iter()
    .enumerate()
    {
        let y = 150.0 + index as f32 * 150.0;
        scene.try_styled_line(
            p(left, y),
            p(middle - 70.0, y),
            StrokeStyle2d::logical(px(22.0), color).with_cap(cap),
        )?;
    }
    for (index, (join, color)) in [
        (StrokeJoin2d::Bevel, Color::rgb8(65, 196, 255)),
        (StrokeJoin2d::Miter, Color::rgb8(255, 176, 70)),
        (StrokeJoin2d::Round, Color::rgb8(106, 224, 142)),
    ]
    .into_iter()
    .enumerate()
    {
        let y = 150.0 + index as f32 * 150.0;
        scene.try_styled_polyline(
            &[
                p(middle + 40.0, y + 42.0),
                p((middle + right) * 0.5, y - 42.0),
                p(right, y + 42.0),
            ],
            StrokeStyle2d::logical(px(22.0), color)
                .with_cap(StrokeCap2d::Butt)
                .with_join(join),
        )?;
    }
    Ok(())
}

fn draw_alpha_contract(
    scene: &mut ScreenScene,
    width: f32,
    height: f32,
) -> Result<(), Box<dyn Error>> {
    let colors = [
        Color::rgba(1.0, 0.12, 0.08, 0.5),
        Color::rgba(0.08, 1.0, 0.24, 0.5),
        Color::rgba(0.08, 0.45, 1.0, 0.5),
    ];
    let joins = [
        StrokeJoin2d::Round,
        StrokeJoin2d::Bevel,
        StrokeJoin2d::Miter,
    ];
    for index in 0..3 {
        let center_x = width * (0.2 + index as f32 * 0.3);
        let center_y = height * 0.46;
        let mut style = StrokeStyle2d::logical(px(34.0), colors[index])
            .with_cap(StrokeCap2d::Round)
            .with_join(joins[index]);
        if joins[index] == StrokeJoin2d::Miter {
            style = style.with_miter_limit(1.0)?;
        }
        scene.try_styled_polyline(
            &[
                p(center_x - 105.0, center_y + 105.0),
                p(center_x, center_y - 95.0),
                p(center_x + 105.0, center_y + 105.0),
            ],
            style,
        )?;
        scene.try_rect(
            p(center_x - 105.0, center_y + 150.0),
            LogicalScreenVector::new(210.0, 34.0),
            px(17.0),
            ShapeStyle::filled(colors[index]),
        )?;
    }
    Ok(())
}

fn draw_dashes_and_markers(
    scene: &mut ScreenScene,
    width: f32,
    height: f32,
    animation: f32,
) -> Result<(), Box<dyn Error>> {
    let phase = (animation * 38.0).floor().rem_euclid(82.0);
    let dash = StrokeDashPattern2d::new(&[28.0, 22.0, 10.0, 22.0], phase, 2_048)?;
    let arrow = StrokeMarker2d::arrow(px(30.0), px(28.0));
    let rows = [
        (Color::rgb8(65, 196, 255), false, true),
        (Color::rgb8(255, 176, 70), true, false),
        (Color::rgba(0.2, 1.0, 0.45, 0.5), true, true),
    ];
    for (index, (color, start, end)) in rows.into_iter().enumerate() {
        let y = height * 0.28 + index as f32 * 155.0;
        let mut style = StrokeStyle2d::logical(px(14.0), color)
            .with_cap(StrokeCap2d::Round)
            .with_join(StrokeJoin2d::Round)
            .with_dash_pattern(dash);
        if start {
            style = style.with_start_marker(arrow);
        }
        if end {
            style = style.with_end_marker(arrow);
        }
        scene.try_styled_polyline(
            &[
                p(100.0, y),
                p(width * 0.42, y - 55.0),
                p(width * 0.67, y + 45.0),
                p(width - 100.0, y),
            ],
            style,
        )?;
    }
    Ok(())
}

fn draw_camera_frame(
    scene: &mut ScreenScene,
    width: f32,
    height: f32,
) -> Result<(), Box<dyn Error>> {
    scene.try_rect(
        p(width * 0.24, height * 0.31),
        LogicalScreenVector::new(width * 0.52, height * 0.38),
        px(8.0),
        ShapeStyle::fill_stroke(
            Color::rgba(0.0, 0.0, 0.0, 0.25),
            2.0,
            Color::rgb8(82, 101, 124),
        ),
    )?;
    Ok(())
}

fn build_world_edge_scene() -> Result<Scene, Box<dyn Error>> {
    let mut scene = Scene::new(Color::BLACK)?;
    scene.try_styled_line(
        Vec2::new(-0.0025, 0.0),
        Vec2::new(0.0025, 0.0),
        StrokeStyle2d::logical(px(8.0), Color::rgb8(65, 196, 255)).with_cap(StrokeCap2d::Round),
    )?;
    let arrow = StrokeMarker2d::arrow(px(18.0), px(16.0));
    scene.try_styled_polyline(
        vec![
            Vec2::new(-0.015, -0.012),
            Vec2::new(0.0, 0.014),
            Vec2::new(0.015, -0.012),
        ],
        StrokeStyle2d::logical(px(7.0), Color::rgba(1.0, 0.58, 0.08, 0.65))
            .with_cap(StrokeCap2d::Round)
            .with_join(StrokeJoin2d::Miter)
            .with_miter_limit(1.0)?
            .with_end_marker(arrow),
    )?;
    Ok(scene)
}

const fn p(x: f32, y: f32) -> LogicalScreenPosition {
    LogicalScreenPosition::new(x, y)
}

fn px(value: f32) -> LogicalPixels {
    LogicalPixels::new(value).expect("gallery uses positive finite logical pixels")
}
