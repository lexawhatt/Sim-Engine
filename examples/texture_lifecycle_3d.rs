//! Interactive mip/UV/texture-edit comparison; application-owned gallery only.

#[path = "support/texture_lifecycle_3d_scene.rs"]
mod gallery;
#[path = "support/mesh3d_dev5_acceptance.rs"]
mod standalone_acceptance;

use gallery::{BACKGROUND, Gallery};
use sim_engine::{
    BlendMode, Camera3d, GpuTimingId, GpuTimingSource, GpuTimingStatus, LogicalViewport,
    Mesh3dRenderBudget, Projection3d, RenderStatus, RenderTarget3d, RendererPresentMode,
    SurfaceRasterization3d, Vec3, WgpuRenderer, WgpuRendererOptions, WorldLength,
};
use standalone_acceptance::StandaloneRecovery;
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

type ExampleResult<T> = Result<T, Box<dyn Error>>;
const PRESENTS: usize = 180;
const MAX_ATTEMPTS: usize = 1200;
const TIMEOUT: Duration = Duration::from_secs(90);

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
                    let value = arguments
                        .next()
                        .ok_or("--frames requires a count")?
                        .parse()?;
                    if !(1..=100_000).contains(&value) {
                        return Err("--frames must be 1..=100000".into());
                    }
                    result.frames = Some(value);
                }
                "--help" | "-h" => {
                    println!(
                        "texture_lifecycle_3d [--uncapped] [--frames N | --acceptance]\n\
                        TOP: identical repeated checker, left mip-zero / right full mip chain.\n\
                        MIDDLE: isolated cropped tile, left positive UV / right mirrored UV.\n\
                        BOTTOM: left editable object / right immutable original alias.\n\
                        U edit region; T repeat/clamp; X mirror; Space pause UV scroll;\n\
                        M Opaque/Mask/Blend; L Lambert/Unlit; F fog; C sidedness; P projection;\n\
                        arrows orbit; wheel or +/- zoom; R reset view; F5 recover; Esc exit.\n\
                        Mips are alpha-aware but do not preserve Mask coverage or remove all straight-alpha bilinear fringes."
                    );
                    return Ok(None);
                }
                _ => return Err(format!("unknown argument: {argument}").into()),
            }
        }
        if result.acceptance && result.frames.is_some() {
            return Err("--acceptance and --frames are mutually exclusive".into());
        }
        Ok(Some(result))
    }
}

#[derive(Default)]
struct Evidence {
    drawn: usize,
    skipped: usize,
    recoveries: usize,
    after_recovery: usize,
    alpha: [usize; 3],
    repeat: [usize; 2],
    mirrored: [usize; 2],
    lit: [usize; 2],
    fog: [usize; 2],
    front: [usize; 2],
    projection: [usize; 2],
}

impl Evidence {
    fn record(&mut self, gallery: &Gallery, perspective: bool) {
        self.drawn += 1;
        self.after_recovery += usize::from(self.recoveries > 0);
        self.alpha[gallery.alpha_mode] += 1;
        self.repeat[usize::from(gallery.repeat)] += 1;
        self.mirrored[usize::from(gallery.mirrored)] += 1;
        self.lit[usize::from(gallery.lit)] += 1;
        self.fog[usize::from(gallery.fog)] += 1;
        self.front[usize::from(gallery.front_only)] += 1;
        self.projection[usize::from(perspective)] += 1;
    }
    fn validate(&self, edits: usize, detached: usize, reused: usize) -> ExampleResult<()> {
        if self.drawn < PRESENTS
            || self.recoveries != 1
            || self.after_recovery < 20
            || self.alpha.contains(&0)
            || [
                self.repeat,
                self.mirrored,
                self.lit,
                self.fog,
                self.front,
                self.projection,
            ]
            .iter()
            .any(|counts| counts.contains(&0))
            || edits < 3
            || detached == 0
            || reused == 0
        {
            return Err("incomplete texture mip/UV/material/edit/alias/recovery acceptance".into());
        }
        Ok(())
    }
}

/// Public-path timing evidence is optional, bounded and independent of FPS.
/// The gallery measures only Scene3d; its legacy target compositor is untimed.
struct TimingEvidence {
    generation: usize,
    status: [GpuTimingStatus; 2],
    samples: [usize; 2],
    issued: Vec<(GpuTimingId, bool)>,
    latest: Option<GpuTimingId>,
    old_fence: Option<GpuTimingId>,
}

impl TimingEvidence {
    fn new(status: GpuTimingStatus) -> ExampleResult<Self> {
        if status == GpuTimingStatus::Disabled {
            return Err("acceptance requested GPU timing but initialization disabled it".into());
        }
        Ok(Self {
            generation: 0,
            status: [status; 2],
            samples: [0; 2],
            issued: Vec::with_capacity(MAX_ATTEMPTS),
            latest: None,
            old_fence: None,
        })
    }

    fn submitted(&mut self, id: Option<GpuTimingId>) -> ExampleResult<()> {
        let Some(id) = id else {
            return Ok(());
        };
        if self.status[self.generation] != GpuTimingStatus::Enabled
            || self.latest.is_some_and(|last| id <= last)
            || self.old_fence.is_some_and(|old| id <= old)
            || self.issued.len() >= MAX_ATTEMPTS
        {
            return Err(
                "GPU timing report ID violates availability, generation or monotonicity".into(),
            );
        }
        self.latest = Some(id);
        self.issued.push((id, false));
        Ok(())
    }

    fn collect(&mut self, renderer: &mut WgpuRenderer) -> ExampleResult<()> {
        let batch = renderer.collect_gpu_timings();
        if renderer.gpu_timing_statistics().status() != self.status[self.generation] {
            return Err(
                "GPU timing availability changed without an explicit device recovery".into(),
            );
        }
        for sample in batch.samples() {
            if sample.source() != GpuTimingSource::Scene3d
                || self.old_fence.is_some_and(|old| sample.id() <= old)
            {
                return Err("unexpected or stale old-device GPU timing sample".into());
            }
            let index = self
                .issued
                .binary_search_by_key(&sample.id(), |entry| entry.0)
                .map_err(|_| "GPU timing sample has no matching submitted Scene3d report")?;
            if self.issued[index].1 {
                return Err("duplicate GPU timing sample".into());
            }
            self.issued[index].1 = true;
            self.samples[self.generation] += 1;
        }
        Ok(())
    }

    fn recovered(&mut self, renderer: &mut WgpuRenderer, same_adapter: bool) -> ExampleResult<()> {
        if self.generation != 0 {
            return Err("acceptance expected exactly one timing recovery".into());
        }
        if self.status[0] == GpuTimingStatus::Enabled && self.samples[0] == 0 {
            return Err("no correlated GPU timing sample before recovery".into());
        }
        let stats = renderer.gpu_timing_statistics();
        if stats.status() == GpuTimingStatus::Disabled
            || (same_adapter && stats.status() != self.status[0])
            || stats.pending() != 0
            || stats.submitted() != 0
            || stats.completed() != 0
            || stats.dropped() != 0
            || stats.map_failures() != 0
            || stats.invalid_samples() != 0
            || stats.poll_failures() != 0
        {
            return Err(
                "recovery lost timing opt-in or failed to reset its current-device counters".into(),
            );
        }
        self.old_fence = self.latest;
        self.issued.clear();
        self.generation = 1;
        self.status[1] = stats.status();
        // No new pass has been submitted. Polling must never expose the pending
        // old ring, even when its callbacks completed during device recovery.
        self.collect(renderer)?;
        if self.samples[1] != 0 {
            return Err("recovery delivered an unsubmitted new-device sample".into());
        }
        println!(
            "gpu_timing_recovery same_adapter={} status_before={:?} status_after={:?} old_id_fence={:?} pending_after_reset=0 submitted_after_reset=0",
            same_adapter,
            self.status[0],
            self.status[1],
            self.old_fence.map(GpuTimingId::get)
        );
        Ok(())
    }

    fn validate(&self) -> ExampleResult<()> {
        if self.generation != 1
            || (0..2).any(|index| {
                self.status[index] == GpuTimingStatus::Enabled && self.samples[index] == 0
            })
        {
            return Err("missing correlated Scene3d GPU samples before/after recovery".into());
        }
        Ok(())
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
    let state = application
        .state
        .as_ref()
        .ok_or("gallery did not initialize")?;
    if options.acceptance {
        state
            .standalone
            .as_ref()
            .ok_or("acceptance lost public standalone recovery controls")?
            .validate()?;
        state.evidence.validate(
            state.gallery.edits,
            state.gallery.detached_edits,
            state.gallery.reused_edits,
        )?;
        let timing = state
            .timing
            .as_ref()
            .ok_or("acceptance lost timing evidence")?;
        timing.validate()?;
        println!(
            "gpu_timing_acceptance status_before={:?} status_after={:?} correlated_scene_samples_before={} correlated_scene_samples_after={}; unavailable is explicit, never a CPU or zero-duration substitute",
            timing.status[0], timing.status[1], timing.samples[0], timing.samples[1]
        );
        println!(
            "texture_lifecycle_3d_acceptance=pass drawn={} attempts={} skipped={} edits={} detached={} reused={} uploaded_bytes={} copied_bytes={} recoveries={} after_recovery={}; confirmed presentation and ownership checks, not a pixel oracle",
            state.evidence.drawn,
            state.attempts,
            state.evidence.skipped,
            state.gallery.edits,
            state.gallery.detached_edits,
            state.gallery.reused_edits,
            state.gallery.uploaded_bytes,
            state.gallery.copied_bytes,
            state.evidence.recoveries,
            state.evidence.after_recovery
        );
    } else if options
        .frames
        .is_some_and(|count| state.evidence.drawn < count)
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

struct State {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    target: RenderTarget3d,
    gallery: Gallery,
    standalone: Option<StandaloneRecovery>,
    evidence: Evidence,
    timing: Option<TimingEvidence>,
    attempts: usize,
    started: Instant,
    last_frame: Instant,
    playing: bool,
    perspective: bool,
    yaw: f32,
    pitch: f32,
    zoom: f32,
}

impl State {
    fn new(event_loop: &ActiveEventLoop, options: Options) -> ExampleResult<Self> {
        let window = Arc::new(event_loop.create_window(Window::default_attributes()
            .with_title("Texture lifecycle: mip comparison / isolated tiles / selected edit vs original")
            .with_inner_size(LogicalSize::new(1200.0, 840.0)))?);
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
            WgpuRendererOptions::new(mode, window.scale_factor())?
                .with_gpu_timing(options.acceptance),
        ))?;
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
        let target = make_target(&renderer, size.width.max(1), size.height.max(1))?;
        let gallery = Gallery::new(&renderer)?;
        let standalone = options
            .acceptance
            .then(|| StandaloneRecovery::new(&mut renderer))
            .transpose()?;
        println!(
            "texture_lifecycle_3d adapter={:?} backend={} pci={:?} format={:?} MSAA_surface={} target=1",
            renderer.adapter_name(),
            renderer.adapter_backend(),
            renderer.adapter_pci_bus_id(),
            renderer.surface_format(),
            renderer.surface_sample_count()
        );
        println!(
            "TOP: left mip-zero checker / right identical full-mip checker. Orbit or zoom to compare distant aliasing. MIDDLE: isolated green/yellow tile; its discarded atlas neighbor is magenta. Right panel mirrors the left tile. BOTTOM: left editable / right original immutable alias."
        );
        println!(
            "U paints a padded-row region: first edit detaches, later edits reuse. T Repeat/Clamp, X mirror, Space scroll/pause, M alpha mode, L light, F fog, C front-only, P perspective, arrows orbit, wheel +/- zoom, R reset view, F5 recover, Esc exit. Alpha-aware mip generation does not preserve mask coverage; straight-alpha filtering can fringe. No shadows/PBR/OIT."
        );
        let timing = options
            .acceptance
            .then(|| TimingEvidence::new(renderer.gpu_timing_statistics().status()))
            .transpose()?;
        let state = Self {
            window,
            renderer,
            target,
            gallery,
            standalone,
            evidence: Evidence::default(),
            timing,
            attempts: 0,
            started: Instant::now(),
            last_frame: Instant::now(),
            playing: true,
            perspective: false,
            yaw: 0.0,
            pitch: 0.0,
            zoom: 1.0,
        };
        state.title();
        Ok(state)
    }

    fn title(&self) {
        self.window.set_title(&format!("TOP mip0 | mips ; MIDDLE tile | mirror ; BOTTOM edit | original ; {} {} light:{} fog:{} edits:{} | U T X M L F C P F5",
            self.gallery.alpha_name(), if self.gallery.repeat { "Repeat" } else { "Clamp" },
            self.gallery.lit, self.gallery.fog, self.gallery.edits));
    }

    fn camera(&self) -> ExampleResult<Camera3d> {
        let aspect = self.target.width() as f32 / self.target.height() as f32;
        let span = (10.0 / aspect).max(7.4) / self.zoom;
        let distance = 11.0;
        let projection = if self.perspective {
            Projection3d::perspective(
                2.0 * (span / (distance * 2.0)).atan(),
                aspect,
                WorldLength::new(0.1)?,
                WorldLength::new(60.0)?,
            )?
        } else {
            Projection3d::orthographic(
                WorldLength::new(span)?,
                aspect,
                WorldLength::new(0.1)?,
                WorldLength::new(60.0)?,
            )?
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
        let before = fingerprint(&self.gallery.scene)?;
        let statistics = self.gallery.scene.statistics();
        let lighting = self.gallery.scene.lighting();
        let fog = self.gallery.scene.fog();
        let adapter_before = adapter_identity(&self.renderer);
        if self.timing.is_some()
            && self.renderer.gpu_timing_statistics().status() == GpuTimingStatus::Enabled
            && self.renderer.gpu_timing_statistics().pending() == 0
        {
            return Err(
                "timing recovery acceptance requires an outstanding old-device sample".into(),
            );
        }
        pollster::block_on(self.renderer.recover_device_and_surface())?;
        let same_adapter = adapter_before == adapter_identity(&self.renderer);
        if let Some(timing) = &mut self.timing {
            timing.recovered(&mut self.renderer, same_adapter)?;
        }
        let report = self.renderer.restore_scene3d(&mut self.gallery.scene)?;
        self.target = self.renderer.restore_render_target3d(&self.target)?;
        let restored_statistics = self.gallery.scene.statistics();
        // Recovery can repack scene bookkeeping. Mesh/texture allocations and
        // their deduplication, unlike instance Vec capacity, must be preserved.
        if before != fingerprint(&self.gallery.scene)?
            || lighting != self.gallery.scene.lighting()
            || fog != self.gallery.scene.fog()
            || statistics.object_count() != restored_statistics.object_count()
            || statistics.mesh_count() != restored_statistics.mesh_count()
            || statistics.mesh_cpu_bytes() != restored_statistics.mesh_cpu_bytes()
            || statistics.mesh_gpu_bytes() != restored_statistics.mesh_gpu_bytes()
            || statistics.texture_count() != restored_statistics.texture_count()
            || statistics.texture_cpu_bytes() != restored_statistics.texture_cpu_bytes()
            || statistics.texture_gpu_bytes() != restored_statistics.texture_gpu_bytes()
            || report.restored_mesh_count() != statistics.mesh_count()
            || report.restored_texture_count() != statistics.texture_count()
        {
            return Err(
                "recovery changed IDs, style, UV settings, mip bytes or deduplicated allocations"
                    .into(),
            );
        }
        self.gallery.validate_snapshot()?;
        let camera = self.camera()?;
        if let Some(standalone) = &mut self.standalone {
            let ids = standalone.restore_and_submit(&mut self.renderer, &self.target, camera)?;
            if let Some(timing) = &mut self.timing {
                for id in ids {
                    timing.submitted(Some(id))?;
                }
            }
        }
        self.evidence.recoveries += 1;
        println!(
            "recovered textures={} meshes={}; exact CPU mip chains, material settings and object IDs preserved",
            report.restored_texture_count(),
            report.restored_mesh_count()
        );
        Ok(())
    }

    fn render(&mut self, options: Options) -> ExampleResult<bool> {
        self.attempts += 1;
        if options.acceptance && (self.attempts > MAX_ATTEMPTS || self.started.elapsed() > TIMEOUT)
        {
            return Err("texture acceptance exceeded bounded attempts/time".into());
        }
        if let Some(timing) = &mut self.timing {
            timing.collect(&mut self.renderer)?;
        }
        let now = Instant::now();
        if !options.acceptance && self.playing {
            self.gallery.phase = (self.gallery.phase
                + now.duration_since(self.last_frame).as_secs_f32().min(0.1) * 0.12)
                .rem_euclid(2.0);
        }
        self.last_frame = now;
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(false);
        }
        self.gallery.apply(&self.renderer)?;
        let draw = self.renderer.render_scene3d_to_target_with_budget(
            &self.target,
            &self.gallery.scene,
            self.camera()?,
            Mesh3dRenderBudget::default()
                .with_surface_policy(SurfaceRasterization3d::Native)
                .with_max_surface_triangles(12),
        )?;
        if draw.triangle_count() != 12 {
            return Err("texture toggles changed submitted topology".into());
        }
        if let Some(timing) = &mut self.timing {
            timing.submitted(draw.gpu_timing_id())?;
        }
        let report = self.renderer.compose_render_target(
            self.target.color_target(),
            BlendMode::Alpha,
            1.0,
            BACKGROUND,
        )?;
        if report.status() != RenderStatus::Drawn {
            self.evidence.skipped += 1;
            return Ok(false);
        }
        self.evidence.record(&self.gallery, self.perspective);
        self.gallery.validate_snapshot()?;
        if options.acceptance {
            self.gallery.phase = (self.gallery.phase + 0.01).rem_euclid(2.0);
            match self.evidence.drawn {
                20 => {
                    self.gallery.edit(&self.renderer)?;
                }
                40 => {
                    self.gallery.edit(&self.renderer)?;
                    self.gallery.alpha_mode = 1;
                }
                60 => {
                    self.gallery.lit = true;
                    self.gallery.fog = true;
                }
                80 => self.gallery.repeat = false,
                100 => {
                    self.gallery.mirrored = false;
                    self.gallery.alpha_mode = 2;
                    self.perspective = true;
                    self.zoom = 1.2;
                }
                120 => self.recover()?,
                140 => {
                    self.gallery.repeat = true;
                    self.gallery.mirrored = true;
                    self.gallery.front_only = true;
                    self.yaw = 2.8;
                }
                160 => {
                    self.gallery.edit(&self.renderer)?;
                    self.reset_view();
                }
                PRESENTS => {
                    self.standalone
                        .as_ref()
                        .ok_or("missing public standalone recovery acceptance")?
                        .validate()?;
                    self.evidence.validate(
                        self.gallery.edits,
                        self.gallery.detached_edits,
                        self.gallery.reused_edits,
                    )?;
                    self.timing
                        .as_ref()
                        .ok_or("missing GPU timing acceptance state")?
                        .validate()?;
                    return Ok(true);
                }
                _ => {}
            }
            self.title();
        }
        Ok(options
            .frames
            .is_some_and(|count| self.evidence.drawn >= count))
    }

    fn reset_view(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.zoom = 1.0;
        self.perspective = false;
    }
    fn zoom(&mut self, steps: f32) {
        self.zoom = (self.zoom * 1.1_f32.powf(steps.clamp(-10.0, 10.0))).clamp(0.4, 3.0);
    }
    fn key(&mut self, key: KeyCode) -> ExampleResult<()> {
        match key {
            KeyCode::KeyU => {
                self.gallery.edit(&self.renderer)?;
            }
            KeyCode::KeyT => self.gallery.repeat = !self.gallery.repeat,
            KeyCode::KeyX => self.gallery.mirrored = !self.gallery.mirrored,
            KeyCode::KeyL => self.gallery.lit = !self.gallery.lit,
            KeyCode::KeyF => self.gallery.fog = !self.gallery.fog,
            KeyCode::KeyC => self.gallery.front_only = !self.gallery.front_only,
            KeyCode::KeyM => self.gallery.alpha_mode = (self.gallery.alpha_mode + 1) % 3,
            KeyCode::KeyP => self.perspective = !self.perspective,
            KeyCode::Space => self.playing = !self.playing,
            KeyCode::ArrowLeft => self.yaw -= 0.1,
            KeyCode::ArrowRight => self.yaw += 0.1,
            KeyCode::ArrowUp => self.pitch = (self.pitch + 0.08).min(0.7),
            KeyCode::ArrowDown => self.pitch = (self.pitch - 0.08).max(-0.7),
            KeyCode::Equal | KeyCode::NumpadAdd => self.zoom(1.0),
            KeyCode::Minus | KeyCode::NumpadSubtract => self.zoom(-1.0),
            KeyCode::KeyR => self.reset_view(),
            KeyCode::F5 => self.recover()?,
            _ => {}
        }
        self.title();
        Ok(())
    }
}

fn adapter_identity(renderer: &WgpuRenderer) -> (String, String, String) {
    (
        renderer.adapter_backend().to_owned(),
        renderer.adapter_name().to_owned(),
        renderer.adapter_pci_bus_id().to_owned(),
    )
}

// Recovery compares exact public state, not GPU resource identities. Formatting
// only covers small metadata; complete mip bytes remain byte-exact in the key.
fn fingerprint(scene: &sim_engine::Scene3d) -> ExampleResult<Vec<(String, Vec<u8>)>> {
    scene
        .instances()
        .iter()
        .map(|object| {
            let material = object.mesh().material().ok_or("object lost texture")?;
            let texture = material.texture();
            let mut pixels = Vec::new();
            for level in 0..texture.mip_level_count() {
                pixels.extend_from_slice(
                    texture
                        .mip_level_pixels(level)
                        .ok_or("missing committed mip")?,
                );
            }
            Ok((
                format!(
                    "{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
                    object.id(),
                    object.transform(),
                    object.style(),
                    object.is_visible(),
                    material.uv_transform(),
                    material.address_mode(),
                    material.sampling(),
                    material.tint(),
                    texture.options()
                ),
                pixels,
            ))
        })
        .collect()
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
        if id != state.window.id() {
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
                WindowEvent::MouseWheel { delta, .. } if !self.options.acceptance => {
                    state.zoom(match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(position) => position.y as f32 / 100.0,
                    });
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
                    Some("texture acceptance timed out waiting for compositor".to_owned());
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
    fn timing_acceptance_distinguishes_unavailable_from_missing_supported_samples() {
        assert!(TimingEvidence::new(GpuTimingStatus::Disabled).is_err());
        let mut enabled = TimingEvidence::new(GpuTimingStatus::Enabled).unwrap();
        assert!(enabled.validate().is_err());
        enabled.generation = 1;
        enabled.samples = [1, 0];
        assert!(enabled.validate().is_err());
        enabled.samples = [1, 1];
        assert!(enabled.validate().is_ok());
        let mut unavailable = TimingEvidence::new(GpuTimingStatus::Unavailable).unwrap();
        unavailable.generation = 1;
        assert!(unavailable.validate().is_ok());
        unavailable.status[1] = GpuTimingStatus::Enabled;
        assert!(unavailable.validate().is_err());
        unavailable.samples[1] = 1;
        assert!(unavailable.validate().is_ok());
    }

    #[test]
    fn arguments_and_acceptance_reject_incomplete_execution() {
        let parse = |args: &[&str]| Options::parse(args.iter().map(|arg| arg.to_string()));
        assert!(parse(&["--acceptance", "--frames", "180"]).is_err());
        assert!(parse(&["--frames", "0"]).is_err());
        assert!(parse(&["--bad"]).is_err());
        let mut evidence = Evidence {
            drawn: PRESENTS,
            skipped: 0,
            recoveries: 1,
            after_recovery: 60,
            alpha: [40, 60, 80],
            repeat: [40, 140],
            mirrored: [40, 140],
            lit: [60, 120],
            fog: [60, 120],
            front: [140, 40],
            projection: [120, 60],
        };
        assert!(evidence.validate(3, 1, 2).is_ok());
        assert!(evidence.validate(3, 3, 0).is_err());
        evidence.after_recovery = 0;
        assert!(evidence.validate(3, 1, 2).is_err());
    }
}
