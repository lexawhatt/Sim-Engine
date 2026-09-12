//! Unlit/Lambert comparison and opt-in fog rows; host-owned presentation only.

#[path = "support/lighting_3d_scene.rs"]
mod gallery;

use gallery::Gallery;
use sim_engine::{
    BlendMode, Camera3d, Color, LogicalViewport, Mesh3dRenderBudget, Projection3d, RenderStatus,
    RenderTarget3d, RendererPresentMode, SurfaceRasterization3d, Vec3, WgpuRenderer,
    WgpuRendererOptions, WorldLength,
};
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

type ExampleResult<T> = Result<T, Box<dyn Error>>;
const PRESENTS: usize = 160;
const MAX_ATTEMPTS: usize = 800;
const TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Default)]
struct Options {
    acceptance: bool,
    uncapped: bool,
    frames: Option<usize>,
}

impl Options {
    fn parse(arguments: impl IntoIterator<Item = String>) -> ExampleResult<Option<Self>> {
        let mut result = Self::default();
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--acceptance" => result.acceptance = true,
                "--uncapped" => result.uncapped = true,
                "--frames" => {
                    let value = arguments.next().ok_or("--frames needs a count")?.parse()?;
                    if !(1..=100_000).contains(&value) {
                        return Err("--frames must be in 1..=100000".into());
                    }
                    result.frames = Some(value);
                }
                "--help" | "-h" => {
                    println!(
                        "lighting_3d [--uncapped] [--frames N | --acceptance]\n\
                        Top: left Unlit, right Lambert; moving marker indicates directional light.\n\
                        Bottom rows: upper fog opt-in, lower fog opt-out; six increasing distances.\n\
                        L sunlight, F fog, N nonuniform scale, M alpha mode, C sidedness, P projection;\n\
                        arrows orbit, Space pause, R reset, F5 recover, Esc exit.\n\
                        No shadows/point lights/PBR; object-level Blend sorting is not generally correct inside closed meshes."
                    );
                    return Ok(None);
                }
                _ => return Err(format!("unknown argument: {argument}").into()),
            }
        }
        if result.acceptance && result.frames.is_some() {
            return Err("--frames and --acceptance are mutually exclusive".into());
        }
        Ok(Some(result))
    }
}

fn main() -> ExampleResult<()> {
    env_logger::init();
    let Some(options) = Options::parse(std::env::args().skip(1))? else {
        return Ok(());
    };
    let mut application = Application {
        options,
        state: None,
        failure: None,
    };
    EventLoop::new()?.run_app(&mut application)?;
    if let Some(error) = application.failure {
        return Err(error.into());
    }
    if options.acceptance {
        let state = application
            .state
            .as_ref()
            .ok_or("acceptance did not initialize")?;
        state.evidence.validate()?;
        println!(
            "lighting_3d_acceptance=pass drawn={} attempts={} skipped_reports={} recoveries={} modes={:?} sun={:?} fog={:?} scale={:?} projection={:?} submitted_triangles={:?}; presentation checks, not a pixel oracle",
            state.evidence.drawn,
            state.attempts,
            state.skipped,
            state.evidence.recoveries,
            state.evidence.modes,
            state.evidence.sun,
            state.evidence.fog,
            state.evidence.scale,
            state.evidence.projection,
            state.evidence.triangles
        );
    } else if let Some(required) = options.frames
        && application
            .state
            .as_ref()
            .is_none_or(|state| state.evidence.drawn < required)
    {
        return Err("window closed before requested confirmed presents".into());
    }
    Ok(())
}

struct Application {
    options: Options,
    state: Option<State>,
    failure: Option<String>,
}

#[derive(Default)]
struct Evidence {
    drawn: usize,
    modes: [usize; 3],
    sun: [usize; 2],
    fog: [usize; 2],
    scale: [usize; 2],
    projection: [usize; 2],
    sidedness: [usize; 2],
    recoveries: usize,
    after_recovery: usize,
    triangles: Option<usize>,
    first_light_x: Option<f32>,
    light_moved: bool,
}

impl Evidence {
    fn validate(&self) -> ExampleResult<()> {
        if self.drawn < PRESENTS
            || self.modes.iter().any(|count| *count < 20)
            || [
                self.sun,
                self.fog,
                self.scale,
                self.projection,
                self.sidedness,
            ]
            .iter()
            .any(|counts| counts.contains(&0))
            || self.recoveries != 1
            || self.after_recovery < 20
            || !self.light_moved
            || self.triangles.is_none()
        {
            return Err(
                "incomplete lighting/fog/material/scale/projection/recovery acceptance".into(),
            );
        }
        Ok(())
    }

    fn record(
        &mut self,
        gallery: &Gallery,
        perspective: bool,
        triangles: usize,
    ) -> ExampleResult<()> {
        if self.triangles.is_some_and(|previous| previous != triangles) {
            return Err(
                "fog/material changes unexpectedly changed submitted source topology".into(),
            );
        }
        self.triangles = Some(triangles);
        let light_x = gallery.light_horizontal();
        if let Some(first) = self.first_light_x {
            self.light_moved |= (light_x - first).abs() > 0.1;
        } else {
            self.first_light_x = Some(light_x);
        }
        self.drawn += 1;
        self.modes[gallery.alpha_mode] += 1;
        self.sun[usize::from(gallery.sun_enabled)] += 1;
        self.fog[usize::from(gallery.fog_enabled)] += 1;
        self.scale[usize::from(gallery.nonuniform)] += 1;
        self.projection[usize::from(perspective)] += 1;
        self.sidedness[usize::from(gallery.front_only)] += 1;
        self.after_recovery += usize::from(self.recoveries > 0);
        Ok(())
    }
}

struct State {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    target: RenderTarget3d,
    gallery: Gallery,
    evidence: Evidence,
    attempts: usize,
    skipped: usize,
    started: Instant,
    last_frame: Instant,
    playing: bool,
    perspective: bool,
    yaw: f32,
    pitch: f32,
}

impl State {
    fn new(event_loop: &ActiveEventLoop, options: Options) -> ExampleResult<Self> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Lighting | left Unlit / right Lambert | upper fog / lower no fog")
                    .with_inner_size(LogicalSize::new(1200.0, 820.0)),
            )?,
        );
        let size = window.inner_size();
        let present = if options.uncapped {
            RendererPresentMode::NoVsync
        } else {
            RendererPresentMode::Vsync
        };
        let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            WgpuRendererOptions::new(present, window.scale_factor())?,
        ))?;
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
        let target = make_target(&renderer, size.width.max(1), size.height.max(1))?;
        let gallery = Gallery::new(&renderer)?;
        println!(
            "lighting_3d adapter={:?} backend={} pci={:?} driver={:?} format={:?} surface_msaa={} target_msaa=1",
            renderer.adapter_name(),
            renderer.adapter_backend(),
            renderer.adapter_pci_bus_id(),
            renderer.adapter_driver_info(),
            renderer.surface_format(),
            renderer.surface_sample_count()
        );
        println!(
            "Top shapes use identical host normals/colors: left Unlit, right Lambert. L toggles the moving sun, N toggles nonuniform scaling on both. Ambient remains 0.16; sunlight intensity is 0.82. M cycles Opaque/Mask/Blend; C toggles front-only."
        );
        println!(
            "Lower swatches are Unlit: upper row opts into fog, lower row opts out. Left-to-right camera-forward distance increases; Opaque/Mask/Blend repeat. Fog: start 9.5 world units, density 0.40/world unit. F toggles fog without removing any geometry."
        );
        println!(
            "P projection; arrows orbit; Space pause; R reset; F5 recover; Esc exit. No shadows, point lights or PBR. Blend sorting is per object: closed-mesh internal triangles/intersecting surfaces are not generally ordered."
        );
        let state = Self {
            window,
            renderer,
            target,
            gallery,
            evidence: Evidence::default(),
            attempts: 0,
            skipped: 0,
            started: Instant::now(),
            last_frame: Instant::now(),
            playing: true,
            perspective: false,
            yaw: 0.0,
            pitch: 0.0,
        };
        state.title();
        Ok(state)
    }

    fn title(&self) {
        self.window.set_title(&format!("Unlit | Lambert ; fog row | no-fog row ; {} ; sun:{} fog:{} scale:{} | L F N M C P arrows Space R F5 Esc",
            self.gallery.alpha_name(), if self.gallery.sun_enabled { "on" } else { "off" },
            if self.gallery.fog_enabled { "on" } else { "off" }, if self.gallery.nonuniform { "nonuniform" } else { "uniform" }));
    }

    fn camera(&self) -> ExampleResult<Camera3d> {
        let aspect = self.target.width() as f32 / self.target.height() as f32;
        let span = (9.9 / aspect).max(7.1);
        let distance = 11.0;
        let near = WorldLength::new(0.1)?;
        let far = WorldLength::new(60.0)?;
        let projection = if self.perspective {
            Projection3d::perspective(2.0 * (span / (distance * 2.0)).atan(), aspect, near, far)?
        } else {
            Projection3d::orthographic(WorldLength::new(span)?, aspect, near, far)?
        };
        Ok(Camera3d::look_at(
            Vec3::new(
                self.yaw.sin() * distance * self.pitch.cos(),
                self.pitch.sin() * distance,
                self.yaw.cos() * distance * self.pitch.cos(),
            )?,
            Vec3::ZERO,
            Vec3::Y,
            projection,
        )?)
    }

    fn recover(&mut self) -> ExampleResult<()> {
        let previous: Vec<_> = self
            .gallery
            .scene
            .instances()
            .iter()
            .map(|object| {
                (
                    object.id(),
                    object.transform(),
                    object.style(),
                    object.is_visible(),
                )
            })
            .collect();
        let lighting = self.gallery.scene.lighting();
        let fog = self.gallery.scene.fog();
        pollster::block_on(self.renderer.recover_device_and_surface())?;
        let report = self.renderer.restore_scene3d(&mut self.gallery.scene)?;
        self.target = self.renderer.restore_render_target3d(&self.target)?;
        let restored: Vec<_> = self
            .gallery
            .scene
            .instances()
            .iter()
            .map(|object| {
                (
                    object.id(),
                    object.transform(),
                    object.style(),
                    object.is_visible(),
                )
            })
            .collect();
        if previous != restored
            || lighting != self.gallery.scene.lighting()
            || fog != self.gallery.scene.fog()
        {
            return Err(
                "recovery changed scene IDs, transforms, styles or lighting/fog state".into(),
            );
        }
        self.evidence.recoveries += 1;
        println!(
            "Recovery preserved scene and environment: {} meshes, {} textures restored.",
            report.restored_mesh_count(),
            report.restored_texture_count()
        );
        Ok(())
    }

    fn render(&mut self, options: Options) -> ExampleResult<bool> {
        self.attempts += 1;
        if options.acceptance && (self.attempts > MAX_ATTEMPTS || self.started.elapsed() > TIMEOUT)
        {
            return Err("acceptance exceeded time/attempt budget".into());
        }
        let now = Instant::now();
        let delta = if options.acceptance || !self.playing {
            0.0
        } else {
            now.duration_since(self.last_frame).as_secs_f32().min(0.1)
        };
        self.last_frame = now;
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(false);
        }
        self.gallery.update(delta)?;
        let camera = self.camera()?;
        let draw = self.renderer.render_scene3d_to_target_with_budget(
            &self.target,
            &self.gallery.scene,
            camera,
            Mesh3dRenderBudget::default()
                .with_surface_policy(SurfaceRasterization3d::Native)
                .with_max_surface_triangles(4096),
        )?;
        let frame = self.renderer.compose_render_target(
            self.target.color_target(),
            BlendMode::Alpha,
            1.0,
            Color::rgb8(13, 20, 31),
        )?;
        if frame.status() != RenderStatus::Drawn {
            self.skipped += 1;
            return Ok(false);
        }
        self.evidence
            .record(&self.gallery, self.perspective, draw.triangle_count())?;
        if options.acceptance {
            self.gallery.update(1.0 / 30.0)?;
            match self.evidence.drawn {
                20 => self.gallery.sun_enabled = false,
                40 => {
                    self.gallery.sun_enabled = true;
                    self.gallery.alpha_mode = 1;
                }
                60 => {
                    self.gallery.alpha_mode = 2;
                    self.gallery.nonuniform = false;
                }
                80 => {
                    self.perspective = true;
                    self.gallery.nonuniform = true;
                    self.gallery.front_only = true;
                }
                100 => self.gallery.fog_enabled = false,
                110 => self.gallery.fog_enabled = true,
                120 => self.recover()?,
                140 => self.reset()?,
                PRESENTS => {
                    self.evidence.validate()?;
                    return Ok(true);
                }
                _ => {}
            }
            self.title();
        }
        Ok(options
            .frames
            .is_some_and(|limit| self.evidence.drawn >= limit))
    }

    fn reset(&mut self) -> ExampleResult<()> {
        self.gallery.reset()?;
        self.playing = true;
        self.perspective = false;
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.last_frame = Instant::now();
        Ok(())
    }

    fn key(&mut self, key: KeyCode) -> ExampleResult<()> {
        match key {
            KeyCode::Space => self.playing = !self.playing,
            KeyCode::KeyL => self.gallery.sun_enabled = !self.gallery.sun_enabled,
            KeyCode::KeyF => self.gallery.fog_enabled = !self.gallery.fog_enabled,
            KeyCode::KeyN => self.gallery.nonuniform = !self.gallery.nonuniform,
            KeyCode::KeyM => self.gallery.alpha_mode = (self.gallery.alpha_mode + 1) % 3,
            KeyCode::KeyC => self.gallery.front_only = !self.gallery.front_only,
            KeyCode::KeyP => self.perspective = !self.perspective,
            KeyCode::ArrowLeft => self.yaw = (self.yaw - 0.08).max(-0.45),
            KeyCode::ArrowRight => self.yaw = (self.yaw + 0.08).min(0.45),
            KeyCode::ArrowUp => self.pitch = (self.pitch + 0.06).min(0.25),
            KeyCode::ArrowDown => self.pitch = (self.pitch - 0.06).max(-0.25),
            KeyCode::KeyR => self.reset()?,
            KeyCode::F5 => self.recover()?,
            _ => {}
        }
        self.gallery.update(0.0)?;
        self.title();
        Ok(())
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match State::new(event_loop, self.options) {
            Ok(state) => {
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
        let Some(state) = &mut self.state else {
            return;
        };
        if state.window.id() != id {
            return;
        }
        let result: ExampleResult<bool> = (|| {
            match event {
                WindowEvent::CloseRequested => return Ok(true),
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed && !event.repeat =>
                {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        if key == KeyCode::Escape {
                            return Ok(true);
                        }
                        if !self.options.acceptance {
                            state.key(key)?;
                        }
                    }
                }
                WindowEvent::Resized(size) => {
                    state.renderer.resize(size.width, size.height)?;
                    if size.width > 0 && size.height > 0 {
                        state.target = make_target(&state.renderer, size.width, size.height)?;
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
                        state.target = make_target(&state.renderer, size.width, size.height)?;
                    }
                }
                WindowEvent::RedrawRequested => return state.render(self.options),
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
            if self.options.acceptance && state.started.elapsed() > TIMEOUT {
                self.failure =
                    Some("acceptance timed out waiting for compositor redraws".to_owned());
                event_loop.exit();
                return;
            }
            state.window.request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::Poll);
    }
}

fn make_target(renderer: &WgpuRenderer, width: u32, height: u32) -> ExampleResult<RenderTarget3d> {
    let scale = renderer.scale_factor() as f32;
    Ok(renderer.create_render_target3d(
        width,
        height,
        LogicalViewport::new(width as f32 / scale, height as f32 / scale)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_requires_all_feature_states_and_post_recovery_presents() {
        let mut evidence = Evidence {
            drawn: PRESENTS,
            modes: [60, 20, 80],
            sun: [20, 140],
            fog: [40, 120],
            scale: [20, 140],
            projection: [100, 60],
            sidedness: [100, 60],
            recoveries: 1,
            after_recovery: 40,
            triangles: Some(1952),
            first_light_x: Some(0.85),
            light_moved: true,
        };
        assert!(evidence.validate().is_ok());
        evidence.after_recovery = 0;
        assert!(evidence.validate().is_err());
        evidence.after_recovery = 40;
        evidence.light_moved = false;
        assert!(evidence.validate().is_err());
        evidence.light_moved = true;
        evidence.fog[0] = 0;
        assert!(evidence.validate().is_err());
    }

    #[test]
    fn bounded_arguments_reject_unknown_and_conflicting_modes() {
        let parse = |arguments: &[&str]| {
            Options::parse(arguments.iter().map(|argument| argument.to_string()))
        };
        assert!(parse(&["--acceptance", "--frames", "160"]).is_err());
        assert!(parse(&["--frames", "0"]).is_err());
        assert!(parse(&["--unknown"]).is_err());
        assert_eq!(
            parse(&["--frames", "25"]).unwrap().unwrap().frames,
            Some(25)
        );
    }
}
