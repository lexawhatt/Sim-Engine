use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use sim_engine::{
    Camera2d, Color, Easing, Palette, Rect, Scene, Shadow, ShapeStyle, Tween, Vec2, WgpuRenderer,
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
    camera: Camera2d,
    zoom: Tween<f32>,
    started_at: Instant,
    last_frame: Instant,
}

impl DemoApplication {
    fn new() -> Self {
        let camera = Camera2d::new(Vec2::ZERO, 2.2);
        Self {
            window: None,
            renderer: None,
            camera,
            zoom: Tween::new(camera.zoom).to(
                2.8,
                Duration::from_millis(1800),
                Easing::EaseInOutCubic,
            ),
            started_at: Instant::now(),
            last_frame: Instant::now(),
        }
    }

    fn update(&mut self) -> Scene {
        let now = Instant::now();
        let dt = now.saturating_duration_since(self.last_frame);
        self.last_frame = now;

        let time_seconds = now.duration_since(self.started_at).as_secs_f32();
        if !self.zoom.is_active() {
            let target = if self.zoom.target() < 2.5 { 2.8 } else { 2.05 };
            self.zoom
                .set_target(target, Duration::from_millis(1800), Easing::EaseInOutCubic);
        }

        self.camera.zoom = self.zoom.update(dt);
        self.camera.center = Vec2::new(
            (time_seconds * 0.35).sin() * 18.0,
            (time_seconds * 0.28).cos() * 10.0,
        );
        self.camera.rotation = (time_seconds * 0.18).sin() * 0.045;

        build_scene(time_seconds)
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
        let renderer = pollster::block_on(WgpuRenderer::new(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
        ))
        .expect("create renderer");

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
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let scene = self.update();
                window.pre_present_notify();

                if let Some(renderer) = self.renderer.as_mut() {
                    let _ = renderer.render(&scene, &self.camera);
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

fn build_scene(time_seconds: f32) -> Scene {
    let palette = Palette::sim();
    let mut scene = Scene::new(palette.background);

    draw_grid(&mut scene, palette);

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
        ShapeStyle::fill_stroke(
            palette.warning.with_alpha(0.9),
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

        scene.line(
            Vec2::new(coordinate, -220.0),
            Vec2::new(coordinate, 220.0),
            width,
            color,
        );
        scene.line(
            Vec2::new(-340.0, coordinate),
            Vec2::new(340.0, coordinate),
            width,
            color,
        );
    }
}
