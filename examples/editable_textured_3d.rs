//! Interactive E04–E08 fixture. Host code owns the fixed 16-object pool, atlas
//! generation, camera controls, mesh revisions and event-loop recovery.
//!
//! N: filter; E: edit; R: remove/reinsert; A/D: orbit; W/S: close/far camera;
//! H: mathematical-edge overlay; Space: pause; F5: device recovery; Esc: exit.
//! For a close free-camera solid view, disable H first. `--acceptance` exercises
//! these transitions automatically, including a native/half-resolution target
//! resize, and counts confirmed presents only. Window scale events are host-owned.

#[path = "support/editable_textured_pool.rs"]
mod pool;

use pool::{ExampleResult, Pool};
use sim_engine::{
    BlendMode, Color, LogicalViewport, RenderStatus, RenderTarget3d, RendererPresentMode,
    WgpuRenderer, WgpuRendererOptions,
};
use std::{sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

fn main() -> ExampleResult<()> {
    env_logger::init();
    let acceptance = std::env::args().any(|value| value == "--acceptance");
    let mut app = Application {
        state: None,
        failure: None,
        acceptance,
    };
    EventLoop::new()?.run_app(&mut app)?;
    if let Some(error) = app.failure {
        return Err(error.into());
    }
    if acceptance && app.state.as_ref().is_none_or(|state| state.presents < 140) {
        return Err("acceptance exited before the final confirmed present".into());
    }
    Ok(())
}

struct Application {
    state: Option<State>,
    failure: Option<String>,
    acceptance: bool,
}

struct State {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    target: RenderTarget3d,
    pool: Pool,
    last_frame: Instant,
    started: Instant,
    presents: usize,
    attempts: usize,
    clipped_triangles: usize,
    recoveries: usize,
    resize_origin: Option<(u32, u32)>,
    confirmed_resize: bool,
    last_error: Option<String>,
}

impl State {
    fn new(event_loop: &ActiveEventLoop) -> ExampleResult<Self> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title(
                    "Editable textured 3D — N filter, E edit, R remove, H edges, W/S zoom, F5 recovery",
                    )
                    .with_inner_size(LogicalSize::new(1000.0, 720.0)),
            )?,
        );
        let size = window.inner_size();
        let options = WgpuRendererOptions::new(RendererPresentMode::Vsync, window.scale_factor())?;
        let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            options,
        ))?;
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
        let target = make_target(&renderer, &window)?;
        let pool = Pool::new(&renderer)?;
        println!(
            "editable textured 3D: 16 objects, six atlas faces, {} texture bytes; N filter, E edit, R remove/reinsert, H edges, A/D orbit, W/S zoom, Space pause, F5 recover, Esc exit",
            pool.scene.statistics().texture_gpu_bytes()
        );
        Ok(Self {
            window,
            renderer,
            target,
            pool,
            last_frame: Instant::now(),
            started: Instant::now(),
            presents: 0,
            attempts: 0,
            clipped_triangles: 0,
            recoveries: 0,
            resize_origin: None,
            confirmed_resize: false,
            last_error: None,
        })
    }

    fn recover(&mut self) -> ExampleResult<()> {
        let ids: Vec<_> = self
            .pool
            .scene
            .instances()
            .iter()
            .map(|object| object.id())
            .collect();
        pollster::block_on(self.renderer.recover_device_and_surface())?;
        let report = self.renderer.restore_scene3d(&mut self.pool.scene)?;
        self.target = self.renderer.restore_render_target3d(&self.target)?;
        assert_eq!(
            self.pool
                .scene
                .instances()
                .iter()
                .map(|object| object.id())
                .collect::<Vec<_>>(),
            ids
        );
        self.recoveries += 1;
        println!(
            "recovery: {} mesh allocations, {} texture allocations, {} texture bytes; IDs preserved",
            report.restored_mesh_count(),
            report.restored_texture_count(),
            report.restored_texture_bytes()
        );
        Ok(())
    }

    fn render(&mut self, acceptance: bool) -> ExampleResult<bool> {
        let now = Instant::now();
        self.pool.update(if acceptance {
            1.0 / 60.0
        } else {
            now.duration_since(self.last_frame).as_secs_f32()
        })?;
        self.last_frame = now;
        let camera = self
            .pool
            .camera(self.target.width(), self.target.height())?;
        let report =
            match self
                .renderer
                .render_scene3d_to_target(&self.target, &self.pool.scene, camera)
            {
                Ok(report) => {
                    self.last_error = None;
                    report
                }
                Err(error) if !acceptance => {
                    let message = format!(
                        "{error}; previous image retained. H disables optional edges; S moves away."
                    );
                    if self.last_error.as_deref() != Some(&message) {
                        eprintln!("{message}");
                        self.window.set_title(&message);
                        self.last_error = Some(message);
                    }
                    return Ok(false);
                }
                Err(error) => return Err(error.into()),
            };
        self.clipped_triangles += report.preflight().generated_triangle_count();
        self.attempts += 1;
        let frame = self.renderer.compose_render_target(
            self.target.color_target(),
            BlendMode::Replace,
            1.0,
            Color::BLACK,
        )?;
        if frame.status() != RenderStatus::Drawn {
            if acceptance && self.attempts > 500 {
                return Err("acceptance exceeded bounded present retries".into());
            }
            return Ok(false);
        }
        self.presents += 1;
        if self
            .resize_origin
            .is_some_and(|original| original != (self.target.width(), self.target.height()))
        {
            self.confirmed_resize = true;
        }
        if acceptance {
            match self.presents {
                20 => self.pool.toggle_filter(&self.renderer)?,
                35 | 95 => self.pool.edit(&self.renderer)?,
                50 | 60 => self.pool.toggle_first()?,
                70 => {
                    self.pool.set_wireframe(false)?;
                    self.pool.distance = 2.4;
                }
                85 => {
                    self.pool.distance = 7.0;
                    self.pool.set_wireframe(true)?;
                }
                100 => {
                    self.resize_origin = Some((self.target.width(), self.target.height()));
                    // Do not claim a window-manager resize request was accepted:
                    // tiled compositors may ignore it. Resize the owned target
                    // explicitly, retaining its logical viewport/edge widths.
                    let (width, height) = if self.target.width().is_multiple_of(2)
                        && self.target.height().is_multiple_of(2)
                    {
                        (self.target.width() / 2, self.target.height() / 2)
                    } else {
                        (
                            self.target
                                .width()
                                .checked_mul(2)
                                .ok_or("target width overflow")?,
                            self.target
                                .height()
                                .checked_mul(2)
                                .ok_or("target height overflow")?,
                        )
                    };
                    self.target = self.renderer.create_render_target3d(
                        width,
                        height,
                        self.target.logical_viewport(),
                    )?;
                    println!(
                        "offscreen target resized to {width}x{height}; logical viewport unchanged"
                    );
                }
                110 => self.recover()?,
                140 => {
                    assert_eq!(self.pool.scene.object_count(), 16);
                    assert_eq!(self.pool.scene.statistics().texture_count(), 1);
                    assert!(
                        self.clipped_triangles > 0,
                        "close camera did not exercise clipping"
                    );
                    assert_eq!(self.recoveries, 1);
                    assert!(
                        self.confirmed_resize,
                        "composition did not present the resized offscreen target"
                    );
                    println!(
                        "editable_textured_3d_acceptance=pass confirmed_presents={} generated_triangles={} live_objects=16 textures=1 recoveries=1 target_resize=confirmed",
                        self.presents, self.clipped_triangles
                    );
                    return Ok(true);
                }
                _ => {}
            }
        }
        if self.presents.is_multiple_of(60) {
            self.window.set_title(&format!("Textured 3D | {:.0} presented FPS | {} objects | {} shared texture bytes | N filter E edit R remove F5 recover",
                self.presents as f64 / self.started.elapsed().as_secs_f64(), self.pool.scene.object_count(), self.pool.scene.statistics().texture_gpu_bytes()));
        }
        Ok(false)
    }

    fn key(&mut self, code: KeyCode) -> ExampleResult<()> {
        match code {
            KeyCode::Space => self.pool.playing = !self.pool.playing,
            KeyCode::KeyN => self.pool.toggle_filter(&self.renderer)?,
            KeyCode::KeyE => self.pool.edit(&self.renderer)?,
            KeyCode::KeyR => self.pool.toggle_first()?,
            KeyCode::KeyH => self.pool.toggle_wireframe()?,
            KeyCode::KeyA => self.pool.orbit -= 0.07,
            KeyCode::KeyD => self.pool.orbit += 0.07,
            KeyCode::KeyW => self.pool.distance = (self.pool.distance - 0.3).max(0.4),
            KeyCode::KeyS => self.pool.distance = (self.pool.distance + 0.3).min(25.0),
            KeyCode::F5 => self.recover()?,
            _ => {}
        }
        Ok(())
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match State::new(event_loop) {
            Ok(mut state) => {
                // Deterministic editing/recovery acceptance is not an orbit
                // precision sweep. The interactive demo animates by default;
                // exact edge-on scientific surfaces may be rejected explicitly.
                if self.acceptance {
                    state.pool.playing = false;
                }
                state.window.request_redraw();
                self.state = Some(state);
            }
            Err(error) => {
                self.failure = Some(error.to_string());
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if state.window.id() != id {
            return;
        }
        let result: ExampleResult<bool> = (|| {
            match event {
                WindowEvent::CloseRequested => return Ok(true),
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed =>
                {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        if code == KeyCode::Escape {
                            return Ok(true);
                        }
                        state.key(code)?;
                    }
                }
                WindowEvent::Resized(size) => {
                    state.renderer.resize(size.width, size.height)?;
                    if size.width > 0 && size.height > 0 {
                        state.target = make_target(&state.renderer, &state.window)?;
                    }
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    let size = state.window.inner_size();
                    state.renderer.resize_with_scale_factor(
                        size.width,
                        size.height,
                        scale_factor,
                    )?;
                    if size.width > 0 && size.height > 0 {
                        state.target = make_target(&state.renderer, &state.window)?;
                    }
                }
                WindowEvent::RedrawRequested => return state.render(self.acceptance),
                _ => {}
            }
            Ok(false)
        })();
        match result {
            Ok(true) => event_loop.exit(),
            Ok(false) => state.window.request_redraw(),
            Err(error) => {
                self.failure = Some(error.to_string());
                event_loop.exit();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::Poll);
    }
}

fn make_target(renderer: &WgpuRenderer, window: &Window) -> ExampleResult<RenderTarget3d> {
    let size = window.inner_size();
    let width = size.width.max(1);
    let height = size.height.max(1);
    let scale = window.scale_factor() as f32;
    Ok(renderer.create_render_target3d(
        width,
        height,
        LogicalViewport::new(width as f32 / scale, height as f32 / scale)?,
    )?)
}
