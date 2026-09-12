//! Three side-by-side alpha/material panels using host-owned visual state.
//! No normals, lighting, fog, PBR or order-independent transparency is implied.

#[path = "support/materials_3d_scene.rs"]
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
const ACCEPTANCE_PRESENTS: usize = 120;
const MAX_ACCEPTANCE_ATTEMPTS: usize = 600;
const ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(45);

#[derive(Default, Clone, Copy)]
struct Options {
    acceptance: bool,
    uncapped: bool,
    frames: Option<usize>,
}

impl Options {
    fn parse(arguments: impl IntoIterator<Item = String>) -> ExampleResult<Option<Self>> {
        let mut options = Self::default();
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--acceptance" => options.acceptance = true,
                "--uncapped" => options.uncapped = true,
                "--frames" => {
                    let frames = arguments
                        .next()
                        .ok_or("--frames requires a count")?
                        .parse()?;
                    if !(1..=100_000).contains(&frames) {
                        return Err("--frames must be in 1..=100000".into());
                    }
                    options.frames = Some(frames);
                }
                "--help" | "-h" => {
                    println!(
                        "materials_3d [--uncapped] [--frames N | --acceptance]\n\
                        A or 1/2/3: rotate left/center/right material assignments; C: two-sided/front-only;\n\
                        arrows: orbit; P: perspective/orthographic; Space: pause; R: reset; F5: recover; Esc: exit.\n\
                        Lower swatches stay fixed: vertex RGB, vertex-alpha cutout, overlapping half-alpha faces.\n\
                        Blend sorting is per object; intersecting surfaces and triangles inside one mesh are not generally ordered."
                    );
                    return Ok(None);
                }
                _ => return Err(format!("unknown argument: {argument}").into()),
            }
        }
        if options.acceptance && options.frames.is_some() {
            return Err("--acceptance and --frames are mutually exclusive".into());
        }
        Ok(Some(options))
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
            .ok_or("acceptance did not create a renderer")?;
        state.evidence.validate()?;
        println!(
            "materials_3d_acceptance=pass confirmed_presents={} attempts={} skipped_reports={} recoveries={} preset_presents={:?} sidedness_presents={:?} projection_presents={:?}; no pixel-oracle claim",
            state.evidence.drawn,
            state.attempts,
            state.skipped_reports,
            state.evidence.recoveries,
            state.evidence.presets,
            state.evidence.sidedness,
            state.evidence.projections
        );
    } else if let Some(required) = options.frames
        && application
            .state
            .as_ref()
            .is_none_or(|state| state.evidence.drawn < required)
    {
        return Err("window closed before the requested confirmed presents".into());
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
    presets: [usize; 3],
    sidedness: [usize; 2],
    projections: [usize; 2],
    recoveries: usize,
    after_recovery: usize,
    front_only_front_cards: usize,
    front_only_back_cards: usize,
}

impl Evidence {
    fn validate(&self) -> ExampleResult<()> {
        if self.drawn < ACCEPTANCE_PRESENTS
            || self.presets.iter().any(|count| *count < 20)
            || self.sidedness.contains(&0)
            || self.projections.contains(&0)
            || self.recoveries != 1
            || self.after_recovery < 20
            || self.front_only_front_cards == 0
            || self.front_only_back_cards == 0
        {
            return Err("acceptance exited before every material/sidedness/projection/recovery stage had confirmed presents".into());
        }
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
    skipped_reports: usize,
    started: Instant,
    last_frame: Instant,
    playing: bool,
    yaw: f32,
    pitch: f32,
    perspective: bool,
}

impl State {
    fn new(event_loop: &ActiveEventLoop, options: Options) -> ExampleResult<Self> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Sim;Engine materials | left Opaque / center Mask / right Blend")
                    .with_inner_size(LogicalSize::new(1200.0, 720.0)),
            )?,
        );
        let size = window.inner_size();
        let mode = if options.uncapped {
            RendererPresentMode::NoVsync
        } else {
            RendererPresentMode::Vsync
        };
        let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            WgpuRendererOptions::new(mode, window.scale_factor())?,
        ))?;
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
        let target = make_target(&renderer, size.width.max(1), size.height.max(1))?;
        let gallery = Gallery::new(&renderer)?;
        println!(
            "materials_3d adapter={:?} backend={} pci={:?} driver={:?} format={:?} surface_msaa={} target_msaa=1",
            renderer.adapter_name(),
            renderer.adapter_backend(),
            renderer.adapter_pci_bus_id(),
            renderer.adapter_driver_info(),
            renderer.surface_format(),
            renderer.surface_sample_count(),
        );
        println!(
            "Material cards, left/center/right: Opaque / Mask / Blend. A or 1/2/3 rotates assignments; C toggles two-sided/front-only; arrows orbit; P projection; Space pause; R reset; F5 recovery; Esc exit."
        );
        println!(
            "Checker alpha is 0/96/255: Opaque ignores it, Mask cuts below 0.5, Blend multiplies it by 0.75. Lower swatches: vertex RGB / vertex-alpha cutoff / red+blue 0.5 overlap (near face inserted first). No lighting/fog/mipmap feature is claimed here."
        );
        println!(
            "Transparency is object-sorted, not order-independent: intersecting faces/cyclic overlaps and triangles within one mesh remain limited. Scene RGBA is composed over a dark background exactly once."
        );
        let state = Self {
            window,
            renderer,
            target,
            gallery,
            evidence: Evidence::default(),
            attempts: 0,
            skipped_reports: 0,
            started: Instant::now(),
            last_frame: Instant::now(),
            playing: true,
            yaw: 0.0,
            pitch: 0.0,
            perspective: false,
        };
        state.update_title();
        Ok(state)
    }

    fn update_title(&self) {
        let [left, center, right] = self.gallery.names();
        self.window.set_title(&format!("Materials | L:{left} / C:{center} / R:{right} | {} | {} | A/1/2/3 mode C sides P projection Space pause arrows orbit R reset F5 recover Esc",
            if self.gallery.front_only() { "front-only" } else { "two-sided" },
            if self.perspective { "perspective" } else { "orthographic" }));
    }

    fn camera(&self) -> ExampleResult<Camera3d> {
        let aspect = self.target.width() as f32 / self.target.height() as f32;
        let span = (10.65 / aspect).max(5.4);
        let distance = 11.0;
        let near = WorldLength::new(0.1)?;
        let far = WorldLength::new(60.0)?;
        let projection = if self.perspective {
            Projection3d::perspective(2.0 * (span / (2.0 * distance)).atan(), aspect, near, far)?
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
            .map(|instance| {
                (
                    instance.id(),
                    instance.transform(),
                    instance.style(),
                    instance.is_visible(),
                )
            })
            .collect();
        pollster::block_on(self.renderer.recover_device_and_surface())?;
        let report = self.renderer.restore_scene3d(&mut self.gallery.scene)?;
        self.target = self.renderer.restore_render_target3d(&self.target)?;
        let restored: Vec<_> = self
            .gallery
            .scene
            .instances()
            .iter()
            .map(|instance| {
                (
                    instance.id(),
                    instance.transform(),
                    instance.style(),
                    instance.is_visible(),
                )
            })
            .collect();
        if previous != restored {
            return Err("scene recovery changed stable IDs or presentation state".into());
        }
        self.evidence.recoveries += 1;
        println!(
            "Recovery preserved IDs/styles/transforms: {} meshes, {} textures, {} texture bytes restored.",
            report.restored_mesh_count(),
            report.restored_texture_count(),
            report.restored_texture_bytes()
        );
        Ok(())
    }

    fn render(&mut self, options: Options) -> ExampleResult<bool> {
        self.attempts += 1;
        if options.acceptance
            && (self.attempts > MAX_ACCEPTANCE_ATTEMPTS
                || self.started.elapsed() > ACCEPTANCE_TIMEOUT)
        {
            return Err("acceptance exceeded its bounded attempt/time limit".into());
        }
        let now = Instant::now();
        let delta = if !self.playing || options.acceptance {
            0.0
        } else {
            now.duration_since(self.last_frame).as_secs_f32().min(0.1)
        };
        self.last_frame = now;
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(false);
        }
        self.gallery.animate(delta)?;
        let camera = self.camera()?;
        let facing = self
            .gallery
            .facing_counts(camera.position(), self.perspective)?;
        self.renderer.render_scene3d_to_target_with_budget(
            &self.target,
            &self.gallery.scene,
            camera,
            Mesh3dRenderBudget::default()
                .with_surface_policy(SurfaceRasterization3d::Native)
                .with_max_surface_triangles(2000),
        )?;
        let report = self.renderer.compose_render_target(
            self.target.color_target(),
            BlendMode::Alpha,
            1.0,
            Color::rgb8(12, 16, 24),
        )?;
        if report.status() != RenderStatus::Drawn {
            self.skipped_reports += 1;
            return Ok(false);
        }
        self.evidence.drawn += 1;
        self.evidence.presets[self.gallery.preset()] += 1;
        self.evidence.sidedness[usize::from(self.gallery.front_only())] += 1;
        self.evidence.projections[usize::from(self.perspective)] += 1;
        if self.gallery.front_only() {
            self.evidence.front_only_front_cards += facing.0;
            self.evidence.front_only_back_cards += facing.1;
        }
        if self.evidence.recoveries > 0 {
            self.evidence.after_recovery += 1;
        }
        if options.acceptance {
            // Failed presentation attempts must not advance the deterministic
            // camera/card checkpoints used by this bounded acceptance fixture.
            self.gallery.animate(1.0 / 20.0)?;
            match self.evidence.drawn {
                20 => self.gallery.set_preset(1)?,
                40 => {
                    self.gallery.set_preset(2)?;
                    self.perspective = true;
                    self.gallery.toggle_sidedness()?;
                }
                60 => self.yaw = 0.12,
                80 => self.recover()?,
                100 => self.reset()?,
                ACCEPTANCE_PRESENTS => {
                    self.evidence.validate()?;
                    return Ok(true);
                }
                _ => {}
            }
            self.update_title();
        }
        Ok(options
            .frames
            .is_some_and(|frames| self.evidence.drawn >= frames))
    }

    fn reset(&mut self) -> ExampleResult<()> {
        self.gallery.reset()?;
        self.playing = true;
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.perspective = false;
        self.last_frame = Instant::now();
        Ok(())
    }

    fn key(&mut self, key: KeyCode) -> ExampleResult<()> {
        match key {
            KeyCode::Space => self.playing = !self.playing,
            KeyCode::KeyA => self.gallery.set_preset(self.gallery.preset() + 1)?,
            KeyCode::Digit1 => self.gallery.set_preset(0)?,
            KeyCode::Digit2 => self.gallery.set_preset(1)?,
            KeyCode::Digit3 => self.gallery.set_preset(2)?,
            KeyCode::KeyC => self.gallery.toggle_sidedness()?,
            KeyCode::KeyP => self.perspective = !self.perspective,
            KeyCode::ArrowLeft => self.yaw = (self.yaw - 0.08).max(-0.7),
            KeyCode::ArrowRight => self.yaw = (self.yaw + 0.08).min(0.7),
            KeyCode::ArrowUp => self.pitch = (self.pitch + 0.08).min(0.4),
            KeyCode::ArrowDown => self.pitch = (self.pitch - 0.08).max(-0.4),
            KeyCode::KeyR => self.reset()?,
            KeyCode::F5 => self.recover()?,
            _ => {}
        }
        self.update_title();
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
            if self.options.acceptance && state.started.elapsed() > ACCEPTANCE_TIMEOUT {
                self.failure =
                    Some("acceptance timed out while waiting for compositor redraws".to_owned());
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
    fn acceptance_requires_all_stages_not_only_attempts_or_early_exit() {
        let mut evidence = Evidence {
            drawn: 120,
            presets: [20, 20, 80],
            sidedness: [80, 40],
            projections: [60, 60],
            recoveries: 1,
            after_recovery: 40,
            front_only_front_cards: 8,
            front_only_back_cards: 90,
        };
        assert!(evidence.validate().is_ok());
        evidence.drawn = 119;
        assert!(evidence.validate().is_err());
        evidence.drawn = 120;
        evidence.presets[1] = 0;
        assert!(evidence.validate().is_err());
        evidence.presets[1] = 20;
        evidence.sidedness[1] = 0;
        assert!(evidence.validate().is_err());
        evidence.sidedness[1] = 40;
        evidence.after_recovery = 0;
        assert!(evidence.validate().is_err());
    }

    #[test]
    fn bounded_cli_rejects_unknown_and_conflicting_modes() {
        let parse = |arguments: &[&str]| {
            Options::parse(arguments.iter().map(|argument| argument.to_string()))
        };
        assert!(parse(&["--frames", "0"]).is_err());
        assert!(parse(&["--frames", "100001"]).is_err());
        assert!(parse(&["--acceptance", "--frames", "120"]).is_err());
        assert!(parse(&["--unknown"]).is_err());
        assert_eq!(
            parse(&["--frames", "25"]).unwrap().unwrap().frames,
            Some(25)
        );
    }
}
