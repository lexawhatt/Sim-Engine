//! Two resolutions, shared native visibility and final GPU composition capture.
use sim_engine::{
    BlendMode, Camera3d, CaptureCorrelation, Color, Fog3d, FrameBudget, FramePassOptions,
    LayeredTarget3d, LogicalViewport, Mesh3d, Mesh3dRenderBudget, MeshStyle3d, Object3dId,
    Projection3d, RenderStatus, RenderTarget2d, RenderTargetBudget, Rotation3d, Scene3d,
    SurfaceEffects3d, SurfaceRasterization3d, SurfaceStyle3d, Transform3d, Vec3, WgpuRenderer,
    WorldLength,
};
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const LIMITS: RenderTargetBudget = RenderTargetBudget::new(64 * 1024 * 1024, 128 * 1024 * 1024);
struct State {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    target: LayeredTarget3d,
    final_target: RenderTarget2d,
    base: Scene3d,
    detail: Scene3d,
    board: Object3dId,
    #[cfg(feature = "text")]
    text_object: Object3dId,
    drawn: u64,
    attempts: u64,
    acceptance: bool,
    resized: bool,
    recovered: bool,
    started: Instant,
}
impl State {
    fn new(event_loop: &ActiveEventLoop, acceptance: bool) -> Result<Self> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title(
                        "Sim;Engine - low-resolution color / native occlusion / GPU capture",
                    )
                    .with_inner_size(PhysicalSize::new(1152, 720))
                    .with_resizable(false),
            )?,
        );
        let size = window.inner_size();
        if acceptance && size != PhysicalSize::new(1152, 720) {
            return Err("layered acceptance requires a 1152x720 physical output".into());
        }
        let mut renderer =
            pollster::block_on(WgpuRenderer::new(window.clone(), size.width, size.height))?;
        if let Ok(required) = std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID")
            && renderer.diagnostics().pci_bus_id() != required
        {
            return Err("layered fixture selected a different physical adapter".into());
        }
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
        let viewport = LogicalViewport::new(size.width as f32, size.height as f32)?;
        let low = if size.width % 3 == 0 && size.height % 3 == 0 {
            (size.width / 3, size.height / 3)
        } else {
            (size.width, size.height)
        };
        let target =
            renderer.create_layered_target3d(low, (size.width, size.height), viewport, LIMITS)?;
        let final_target =
            renderer.create_render_target_with_budget(size.width, size.height, LIMITS)?;
        let mut base = Scene3d::new(Color::rgb(0.025, 0.035, 0.05))?;
        let mut detail = Scene3d::new(Color::BLACK)?;
        let mesh = renderer.create_mesh3d(Mesh3d::new(
            vec![
                Vec3::new(-2.0, -1.0, 0.0)?,
                Vec3::new(2.0, -1.0, 0.0)?,
                Vec3::new(2.0, 1.0, 0.0)?,
                Vec3::new(-2.0, 1.0, 0.0)?,
            ],
            vec![0, 1, 2, 0, 2, 3],
        )?)?;
        let board = base.try_push(
            &mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(
                SurfaceStyle3d::opaque(Color::rgb(0.35, 0.65, 0.45))?.with_fog(true),
            ),
        )?;
        base.set_fog(Some(Fog3d::new(Color::rgb(0.1, 0.15, 0.25), 2.0, 0.15)?));
        // A consumer-owned wedge passes through the moving base surface.
        let wedge = renderer.create_mesh3d(Mesh3d::new(
            vec![
                Vec3::new(-0.9, -0.7, 0.8)?,
                Vec3::new(1.1, -0.7, -0.8)?,
                Vec3::new(0.1, 1.2, 0.0)?,
            ],
            vec![0, 1, 2],
        )?)?;
        detail.try_push(
            &wedge,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::rgb(0.9, 0.3, 0.08))?),
        )?;
        add_mask_probe(&renderer, &mut base)?;
        #[cfg(feature = "text")]
        let text_object = add_curved_text(&renderer, &mut detail)?;
        println!(
            "layered_3d adapter={:?} format={:?} low={low:?} native={:?} target_bytes={}",
            renderer.diagnostics(),
            renderer.surface_format(),
            size,
            target.allocation_bytes()
        );
        Ok(Self {
            window,
            renderer,
            target,
            final_target,
            base,
            detail,
            board,
            #[cfg(feature = "text")]
            text_object,
            drawn: 0,
            attempts: 0,
            acceptance,
            resized: false,
            recovered: false,
            started: Instant::now(),
        })
    }
    fn draw(&mut self) -> Result<()> {
        self.attempts += 1;
        if self.acceptance && self.drawn == 40 && !self.resized {
            let size = self.target.size();
            let old = self.target.allocation_bytes();
            assert!(
                self.renderer
                    .replace_layered_target3d(
                        &mut self.target,
                        (size.0 / 2, size.1 / 2),
                        size,
                        LogicalViewport::new(size.0 as f32, size.1 as f32)?,
                        RenderTargetBudget::new(0, 0)
                    )
                    .is_err()
            );
            assert_eq!(self.target.allocation_bytes(), old);
            self.renderer.replace_layered_target3d(
                &mut self.target,
                (size.0 / 2, size.1 / 2),
                size,
                LogicalViewport::new(size.0 as f32, size.1 as f32)?,
                LIMITS,
            )?;
            self.resized = true;
        }
        if self.acceptance && self.drawn == 80 && !self.recovered {
            let expected_pci = self.renderer.diagnostics().pci_bus_id().to_owned();
            pollster::block_on(self.renderer.recover_device_and_surface())?;
            if self.renderer.diagnostics().pci_bus_id() != expected_pci {
                return Err("layered recovery selected a different physical adapter".into());
            }
            println!(
                "layered_3d recovered_adapter={:?}",
                self.renderer.diagnostics()
            );
            self.renderer.restore_scene3d(&mut self.base)?;
            self.renderer.restore_scene3d(&mut self.detail)?;
            let size = self.target.size();
            let low = self.target.low_size();
            self.renderer.replace_layered_target3d(
                &mut self.target,
                low,
                size,
                LogicalViewport::new(size.0 as f32, size.1 as f32)?,
                LIMITS,
            )?;
            self.renderer.replace_render_target_with_budget(
                &mut self.final_target,
                size.0,
                size.1,
                LIMITS,
            )?;
            self.recovered = true;
        }
        let phase = self.drawn as f32 / 60.0;
        self.base.set_transform(
            self.board,
            Transform3d::new(
                Vec3::ZERO,
                Rotation3d::from_euler_xyz(0.2 * phase.sin(), 0.5 * phase.sin(), 0.0)?,
                Vec3::new(1.0, 1.0, 1.0)?,
            )?,
        )?;
        #[cfg(feature = "text")]
        self.detail.set_transform(
            self.text_object,
            Transform3d::new(
                Vec3::ZERO,
                Rotation3d::from_euler_xyz(0.05 * phase.sin(), 0.15 * phase.sin(), 0.0)?,
                Vec3::new(1.0, 1.0, 1.0)?,
            )?,
        )?;
        let size = self.target.size();
        let camera = Camera3d::look_at(
            Vec3::new(0.0, 0.0, 5.0)?,
            Vec3::ZERO,
            Vec3::Y,
            Projection3d::perspective(
                1.0,
                size.0 as f32 / size.1 as f32,
                WorldLength::new(0.1)?,
                WorldLength::new(30.0)?,
            )?,
        )?;
        let ordinary =
            Mesh3dRenderBudget::default().with_surface_policy(SurfaceRasterization3d::Native);
        let retro = ordinary.with_surface_effects(
            SurfaceEffects3d::new()
                .with_quantization(8)?
                .with_dither(255),
        );
        self.renderer.render_layered_scene3d(
            &self.target,
            &self.base,
            &self.detail,
            camera,
            [retro, ordinary],
        )?;
        let mut frame = self
            .renderer
            .begin_frame(Color::BLACK, FrameBudget::default())?;
        frame.draw_render_target(
            self.target.output().color_target(),
            BlendMode::Replace,
            1.0,
            FramePassOptions::default(),
        )?;
        let report = frame.render_to_target(&self.final_target)?;
        assert_eq!(report.status(), RenderStatus::Drawn);
        assert_eq!(report.surface_present_count(), 0);
        if self.drawn == 0 || (self.acceptance && [39, 79, 119].contains(&self.drawn)) {
            let captured = self.renderer.capture_render_target(
                &self.final_target,
                CaptureCorrelation {
                    frame_id: self.drawn,
                    texture_revision: 0,
                },
                32 * 1024 * 1024,
                Duration::from_secs(15),
            )?;
            if self.acceptance {
                verify_mask_coverage(&captured)?;
                verify_occlusion(&captured, self.base.instance(self.board)?.transform())?;
                let before = captured.pixels().to_vec();
                let committed = captured.content_id().clone();
                let mut alias_frame = self
                    .renderer
                    .begin_frame(Color::WHITE, FrameBudget::default())?;
                alias_frame.draw_render_target(
                    &self.final_target,
                    BlendMode::Replace,
                    1.0,
                    FramePassOptions::default(),
                )?;
                assert!(alias_frame.render_to_target(&self.final_target).is_err());
                assert_eq!(
                    self.renderer
                        .capture_render_target(
                            &self.final_target,
                            captured.correlation(),
                            32 * 1024 * 1024,
                            Duration::from_secs(15)
                        )?
                        .pixels(),
                    before
                );
                assert_eq!(self.final_target.content_id().as_ref(), Some(&committed));
                if self.drawn == 119
                    && let Ok(path) = std::env::var("SIM_ENGINE_LAYERED_CAPTURE")
                {
                    write_ppm(&path, &captured)?;
                }
            }
            println!(
                "capture size={:?} bytes={} staging={} correlation={:?}",
                captured.size(),
                captured.pixels().len(),
                captured.staging_bytes(),
                captured.correlation()
            );
        }
        let mut presentation = self
            .renderer
            .begin_frame(Color::BLACK, FrameBudget::default())?;
        presentation.draw_render_target(
            &self.final_target,
            BlendMode::Replace,
            1.0,
            FramePassOptions::default(),
        )?;
        let report = presentation.present()?;
        assert_eq!(
            report.surface_present_count(),
            usize::from(report.status() == RenderStatus::Drawn)
        );
        if report.status() == RenderStatus::Drawn {
            self.drawn += 1;
        }
        Ok(())
    }
}
#[derive(Default)]
struct App {
    state: Option<State>,
    failure: Option<String>,
    frames: Option<u64>,
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            match State::new(event_loop, self.frames.is_some()) {
                Ok(state) => {
                    state.window.request_redraw();
                    self.state = Some(state)
                }
                Err(error) => {
                    self.failure = Some(error.to_string());
                    event_loop.exit();
                }
            }
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.frames.is_some() {
            if self
                .state
                .as_ref()
                .is_some_and(|state| state.started.elapsed() > Duration::from_secs(60))
            {
                self.failure = Some("layered acceptance timed out".into());
                event_loop.exit();
                return;
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_secs(1),
            ));
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(state) = &mut self.state {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::RedrawRequested => {
                    if let Err(error) = state.draw() {
                        self.failure = Some(error.to_string());
                        event_loop.exit();
                        return;
                    }
                    if self.frames.is_some_and(|frames| state.drawn >= frames) {
                        event_loop.exit();
                    } else if state.acceptance && state.attempts > 1000 {
                        self.failure = Some("too many skipped frames".into());
                        event_loop.exit();
                    } else {
                        state.window.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }
}
fn main() -> Result<()> {
    let mut app = App::default();
    if std::env::args().any(|arg| arg == "--acceptance") {
        app.frames = Some(120);
    }
    EventLoop::new()?.run_app(&mut app)?;
    if let Some(error) = app.failure {
        return Err(error.into());
    }
    if let Some(frames) = app.frames
        && app.state.as_ref().is_none_or(|state| state.drawn < frames)
    {
        return Err("closed before acceptance".into());
    }
    if let Some(state) = app.state.as_ref().filter(|_| app.frames.is_some()) {
        if !state.resized || !state.recovered {
            return Err("acceptance missed resize/recovery".into());
        }
        println!(
            "layered_3d_acceptance=passed drawn={} attempts={} skipped={} resized={} recovered={}",
            state.drawn,
            state.attempts,
            state.attempts - state.drawn,
            state.resized,
            state.recovered
        );
    }
    Ok(())
}

#[cfg(feature = "text")]
fn add_curved_text(renderer: &WgpuRenderer, scene: &mut Scene3d) -> Result<Object3dId> {
    use sim_engine::{
        FontBudget, FontFace, ImageBudget, ImageSampling, LogicalPixels, PhysicalPerLogical,
        TextAtlas2d, TextAtlasBudget, TextLayoutBudget, TextStyle, TextureMaterial3d,
    };
    let font = FontFace::from_bytes(
        include_bytes!("assets/fonts/DejaVuSans.ttf").to_vec(),
        FontBudget::default(),
    )?;
    let mut atlas = TextAtlas2d::new(
        renderer,
        font,
        TextStyle::new(LogicalPixels::new(48.0)?, PhysicalPerLogical::new(1.0)?)?,
        TextAtlasBudget::default(),
    )?;
    let run = atlas.prepare(renderer, "Curved GPU text", TextLayoutBudget::default())?;
    let texture =
        renderer.create_texture3d_from_glyph_atlas(atlas.atlas(), ImageBudget::default())?;
    let material = TextureMaterial3d::with_alpha(&texture, ImageSampling::Linear, Color::WHITE)?;
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    for glyph in run.glyph_run().glyphs() {
        let Some(region) = atlas.atlas().source_for(glyph.glyph()) else {
            continue;
        };
        let bounds = glyph.destination();
        let origin = bounds.origin().to_vec2();
        let size = bounds.viewport();
        let start = positions.len() as u32;
        for (x, y) in [
            (origin.x(), origin.y()),
            (origin.x() + size.width(), origin.y()),
            (origin.x() + size.width(), origin.y() + size.height()),
            (origin.x(), origin.y() + size.height()),
        ] {
            let world_x = (x - run.advance() / 2.0) * 0.007;
            positions.push(Vec3::new(
                world_x,
                -0.8 - y * 0.007,
                1.0 + 0.15 * (world_x * 2.0).sin(),
            )?);
        }
        uvs.extend_from_slice(&texture.region_coordinates(region)?);
        indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    }
    let mesh = renderer.create_mesh3d(Mesh3d::textured(positions, uvs, indices, Vec::new())?)?;
    let mesh = renderer.with_mesh3d_material(&mesh, &material)?;
    let object = scene.try_push(
        &mesh,
        Transform3d::IDENTITY,
        MeshStyle3d::surface(SurfaceStyle3d::mask(Color::WHITE, 0.5)?),
    )?;
    Ok(object)
}

fn write_ppm(path: &str, capture: &sim_engine::RenderTargetCapture) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create_new(path)?;
    let (width, height) = capture.size();
    write!(file, "P6\n{width} {height}\n255\n")?;
    let bgra = matches!(
        capture.format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    for pixel in capture.pixels().chunks_exact(4) {
        let rgb = if bgra {
            [pixel[2], pixel[1], pixel[0]]
        } else {
            [pixel[0], pixel[1], pixel[2]]
        };
        file.write_all(&rgb)?;
    }
    Ok(())
}
fn verify_occlusion(
    capture: &sim_engine::RenderTargetCapture,
    transform: Transform3d,
) -> Result<()> {
    fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|i| a[i] - b[i])
    }
    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a.iter().zip(b).map(|(a, b)| a * b).sum()
    }
    fn hit(direction: [f64; 3], points: [[f64; 3]; 3]) -> Option<f64> {
        let first = sub(points[1], points[0]);
        let second = sub(points[2], points[0]);
        let p = cross(direction, second);
        let determinant = dot(first, p);
        if determinant.abs() < 1e-10 {
            return None;
        }
        let origin = sub([0.0, 0.0, 5.0], points[0]);
        let u = dot(origin, p) / determinant;
        let q = cross(origin, first);
        let v = dot(direction, q) / determinant;
        if u < 0.04 || v < 0.04 || u + v > 0.96 {
            return None;
        }
        Some(dot(second, q) / determinant)
    }
    let board = [
        [-2.0, -1.0, 0.0],
        [2.0, -1.0, 0.0],
        [2.0, 1.0, 0.0],
        [-2.0, 1.0, 0.0],
    ]
    .map(|[x, y, z]| {
        transform
            .transform_point(Vec3::new(x, y, z).unwrap())
            .unwrap()
    })
    .map(|p| [f64::from(p.x()), f64::from(p.y()), f64::from(p.z())]);
    let wedge = [[-0.9, -0.7, 0.8], [1.1, -0.7, -0.8], [0.1, 1.2, 0.0]];
    let (width, height) = capture.size();
    let mut front = 0;
    let mut back = 0;
    let bgra = matches!(
        capture.format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    for y in (height / 3..height * 2 / 3).step_by(5) {
        for x in (width / 4..width * 3 / 4).step_by(5) {
            let direction = [
                (2.0 * (f64::from(x) + 0.5) / f64::from(width) - 1.0) * f64::from(width)
                    / f64::from(height)
                    * 0.5f64.tan(),
                (1.0 - 2.0 * (f64::from(y) + 0.5) / f64::from(height)) * 0.5f64.tan(),
                -1.0,
            ];
            let a = hit(direction, [board[0], board[1], board[2]])
                .or_else(|| hit(direction, [board[0], board[2], board[3]]));
            let b = hit(direction, wedge);
            if let (Some(base), Some(detail)) = (a, b) {
                if (base - detail).abs() < 0.03 {
                    continue;
                }
                let pixel = &capture.pixels()[((y * width + x) * 4) as usize..][..4];
                let red = pixel[if bgra { 2 } else { 0 }];
                let green = pixel[1];
                let orange = red > green.saturating_add(50);
                if detail < base {
                    assert!(orange, "native detail lost at {x},{y}");
                    front += 1;
                } else {
                    assert!(!orange, "occluded detail leaked at {x},{y}");
                    back += 1;
                }
            }
        }
    }
    if front < 30 || back < 30 {
        return Err(format!("insufficient independent occlusion probes: {front}/{back}").into());
    }
    println!(
        "native_occlusion_oracle frame={} visible={front} occluded={back}",
        capture.correlation().frame_id
    );
    Ok(())
}

fn add_mask_probe(renderer: &WgpuRenderer, scene: &mut Scene3d) -> Result<()> {
    use sim_engine::{ImageBudget, ImageSampling, TextureCoordinate2d, TextureMaterial3d};
    let texture = renderer.create_texture3d_rgba8_with_alpha(
        4,
        1,
        vec![
            255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 0,
        ],
        ImageBudget::default(),
    )?;
    let material = TextureMaterial3d::with_alpha(&texture, ImageSampling::Nearest, Color::WHITE)?;
    let vertices = [
        [-0.5, 1.25, 0.1],
        [0.5, 1.25, 0.1],
        [0.5, 1.7, 0.1],
        [-0.5, 1.7, 0.1],
    ]
    .map(|[x, y, z]| Vec3::new(x, y, z))
    .into_iter()
    .collect::<std::result::Result<Vec<_>, _>>()?;
    let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        .map(|[u, v]| TextureCoordinate2d::new(u, v))
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mesh = renderer.create_mesh3d(Mesh3d::textured(
        vertices,
        uvs,
        vec![0, 1, 2, 0, 2, 3],
        Vec::new(),
    )?)?;
    let mesh = renderer.with_mesh3d_material(&mesh, &material)?;
    scene.try_push(
        &mesh,
        Transform3d::IDENTITY,
        MeshStyle3d::surface(SurfaceStyle3d::mask(Color::WHITE, 0.5)?),
    )?;
    Ok(())
}
fn verify_mask_coverage(capture: &sim_engine::RenderTargetCapture) -> Result<()> {
    let (width, height) = capture.size();
    let mut probes = 0;
    let bgra = matches!(
        capture.format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    let background = [0.025f64, 0.035, 0.05].map(|linear| {
        let value = if capture.format().is_srgb() {
            if linear <= 0.0031308 {
                12.92 * linear
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            }
        } else {
            linear
        };
        (value * 255.0).round() as u8
    });
    for y in 0..height / 3 {
        for x in width / 3..width * 2 / 3 {
            let wx = (2.0 * (f64::from(x) + 0.5) / f64::from(width) - 1.0) * f64::from(width)
                / f64::from(height)
                * 0.5f64.tan()
                * 4.9;
            let wy = (1.0 - 2.0 * (f64::from(y) + 0.5) / f64::from(height)) * 0.5f64.tan() * 4.9;
            if !(-0.53..0.53).contains(&wx) || !(1.22..1.73).contains(&wy) {
                continue;
            }
            let texel = (wx + 0.5) * 4.0;
            // Avoid exact hardware coverage and nearest-sampling threshold ties.
            if (texel - texel.round()).abs() < 0.001
                || (wy - 1.25).abs() < 0.001
                || (wy - 1.7).abs() < 0.001
            {
                continue;
            }
            let white = (-0.5..0.5).contains(&wx)
                && (1.25..1.7).contains(&wy)
                && (texel.floor() as i32) % 2 == 0;
            let expected = if white { [255; 3] } else { background };
            let pixel = &capture.pixels()[((y * width + x) * 4) as usize..][..4];
            for channel in 0..3 {
                let actual = pixel[if bgra { 2 - channel } else { channel }];
                assert!(
                    actual.abs_diff(expected[channel]) <= 1,
                    "mask/native silhouette {x},{y}: {pixel:?}, expected {expected:?}"
                );
            }
            probes += 1;
        }
    }
    if probes < 100 {
        return Err("insufficient mask probes".into());
    }
    println!("mask_coverage_oracle probes={probes}");
    Ok(())
}
