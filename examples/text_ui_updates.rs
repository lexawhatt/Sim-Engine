//! Real TTF text with retained updates, independent placement/tint, and clipping.
//! Space pauses, R resets, Esc exits. Use --help for font and smoke-test options.

#[path = "support/retained_ui_acceptance.rs"]
mod retained_ui_acceptance;
#[path = "support/text_font_gallery.rs"]
mod text_font_gallery;
#[path = "support/text_prepared_acceptance.rs"]
mod text_prepared_acceptance;
#[path = "support/text_ui_acceptance.rs"]
mod text_ui_acceptance;
#[path = "support/text_ui_content.rs"]
mod text_ui_content;

use std::{error::Error, path::PathBuf, sync::Arc, time::Instant};

use sim_engine::{
    Color, FontBudget, FontFace, FrameBudget, FramePassOptions, LogicalScreenPosition,
    LogicalScreenVector, RenderStatus, RendererPresentMode, ScreenScene, ShapeStyle, WgpuRenderer,
    WgpuRendererOptions,
};
use text_font_gallery::{FontGallery, GalleryFonts, PAGE_COUNT, PRESENTS_PER_PAGE, TextPage};
use text_ui_content::TextContent;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const EMBEDDED_FONT: &[u8] = include_bytes!("assets/fonts/DejaVuSans.ttf");
const ACCEPTANCE_TEXT_PRESENTS: usize = PAGE_COUNT * PRESENTS_PER_PAGE;
const MAX_SKIPPED_ATTEMPTS: usize = 120;

#[derive(Default)]
struct Options {
    uncapped: bool,
    frame_limit: Option<usize>,
    acceptance: bool,
    font_path: Option<PathBuf>,
    help: bool,
    page: TextPage,
}

impl Options {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut options = Self::default();
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--uncapped" => options.uncapped = true,
                "--acceptance" => options.acceptance = true,
                "--help" | "-h" => options.help = true,
                "--font" => {
                    let path = arguments.next().ok_or("--font requires a file path")?;
                    options.font_path = Some(path.into());
                }
                "--page" => {
                    options.page =
                        TextPage::parse(&arguments.next().ok_or("--page requires a name")?)?;
                }
                "--frames" => {
                    let count = arguments
                        .next()
                        .ok_or("--frames requires a count")?
                        .parse()?;
                    if count == 0 {
                        return Err("--frames must be positive".into());
                    }
                    options.frame_limit = Some(count);
                }
                _ => return Err(format!("unknown argument: {argument}; use --help").into()),
            }
        }
        if options.acceptance && options.frame_limit.is_some() {
            return Err("--acceptance has its own bounded frame count; omit --frames".into());
        }
        Ok(options)
    }

    fn font_bytes(&self) -> Result<Vec<u8>> {
        match &self.font_path {
            Some(path) => std::fs::read(path)
                .map_err(|error| format!("cannot read font {}: {error}", path.display()).into()),
            None => Ok(EMBEDDED_FONT.to_vec()),
        }
    }

    fn verify_completion(&self, drawn: usize, acceptance_finished: bool) -> Result<()> {
        if self.acceptance && (!acceptance_finished || drawn < ACCEPTANCE_TEXT_PRESENTS) {
            return Err(format!(
                "text acceptance interrupted: {drawn}/{ACCEPTANCE_TEXT_PRESENTS} TTF presents; recovery completed={acceptance_finished}"
            )
            .into());
        }
        if let Some(limit) = self.frame_limit
            && drawn < limit
        {
            return Err(
                format!("text smoke interrupted: {drawn}/{limit} confirmed presents").into(),
            );
        }
        Ok(())
    }
}

struct Demo {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    font: FontFace,
    gallery_fonts: GalleryFonts,
    gallery: Option<FontGallery>,
    page: TextPage,
    panel_page: TextPage,
    page_presents: [usize; PAGE_COUNT],
    scroll: f32,
    panel_scroll: f32,
    text: Option<TextContent>,
    panels: ScreenScene,
    panel_width: f32,
    seconds: f64,
    last_frame: Instant,
    paused: bool,
    drawn: usize,
    skipped: usize,
    frame_limit: Option<usize>,
    acceptance_requested: bool,
    acceptance: Option<retained_ui_acceptance::RetainedUiAcceptance>,
    text_acceptance: Option<text_ui_acceptance::TextAcceptance>,
}

impl Demo {
    fn new(events: &ActiveEventLoop, options: &Options, font: FontFace) -> Result<Self> {
        let window = Arc::new(
            events.create_window(
                Window::default_attributes()
                    .with_title("Sim;Engine 0.3 | Type Lab: 1 Fonts / 2 Unicode / 3 Motion | Esc")
                    .with_inner_size(LogicalSize::new(1120.0, 820.0)),
            )?,
        );
        let size = window.inner_size();
        let renderer_options = WgpuRendererOptions::new(
            if options.uncapped {
                RendererPresentMode::NoVsync
            } else {
                RendererPresentMode::Vsync
            },
            window.scale_factor(),
        )?;
        let mut renderer = pollster::block_on(WgpuRenderer::new_with_options(
            window.clone(),
            size.width,
            size.height,
            renderer_options,
        ))?;
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
        if options.acceptance {
            text_prepared_acceptance::verify(&renderer)?;
        }
        let text_acceptance = options
            .acceptance
            .then(|| text_ui_acceptance::TextAcceptance::new(&renderer, font.clone()))
            .transpose()?;
        let acceptance = options
            .acceptance
            .then(|| retained_ui_acceptance::RetainedUiAcceptance::new(&mut renderer))
            .transpose()?;
        println!(
            "TTF text: font={}, physical_size={}x{}, dpi={:.3}, present={:?}; Space pause, R reset, Esc exit",
            options.font_path.as_ref().map_or_else(
                || "embedded DejaVu Sans".to_owned(),
                |path| path.display().to_string()
            ),
            size.width,
            size.height,
            renderer.scale_factor(),
            renderer.surface_present_mode(),
        );
        Ok(Self {
            window,
            renderer,
            gallery_fonts: GalleryFonts::load(font.clone(), options.font_path.is_some())?,
            gallery: None,
            page: options.page,
            panel_page: options.page,
            page_presents: [0; PAGE_COUNT],
            scroll: 0.0,
            panel_scroll: 0.0,
            font,
            text: None,
            panels: ScreenScene::new(Color::BLACK)?,
            panel_width: 0.0,
            seconds: 0.0,
            last_frame: Instant::now(),
            paused: false,
            drawn: 0,
            skipped: 0,
            frame_limit: options.frame_limit,
            acceptance_requested: options.acceptance,
            acceptance,
            text_acceptance,
        })
    }

    fn redraw(&mut self) -> Result<bool> {
        if let Some(acceptance) = &mut self.acceptance {
            if acceptance.step(&mut self.renderer)? {
                self.acceptance = None;
                if let Some(text_acceptance) = &mut self.text_acceptance {
                    text_acceptance.restore_after_recovery(&self.renderer)?;
                }
                self.last_frame = Instant::now();
                println!("Low-level retained UI acceptance passed; starting real TTF presents.");
            }
            return Ok(false);
        }
        let now = Instant::now();
        if !self.paused {
            self.seconds += now.duration_since(self.last_frame).as_secs_f64().min(0.1);
        }
        self.last_frame = now;
        let counter = if self.acceptance_requested {
            9 + self.drawn as u64
        } else {
            9 + (self.seconds * 10.0) as u64
        };
        let dpi = self.renderer.scale_factor() as f32;
        let page = if self.acceptance_requested {
            TextPage::acceptance_page(self.drawn)
        } else {
            self.page
        };
        if page != TextPage::Motion
            && self
                .gallery
                .as_ref()
                .is_none_or(|gallery| gallery.dpi() != dpi)
        {
            let gallery = FontGallery::new(&self.renderer, &self.gallery_fonts, dpi)?;
            let (cpu, gpu) = gallery.retained_bytes();
            println!(
                "Font gallery ready: dpi={dpi:.3}, cache_cpu_bytes={cpu}, cache_gpu_bytes={gpu}; font source bytes are separate"
            );
            self.gallery = Some(gallery);
        }
        if page == TextPage::Motion && self.text.as_ref().is_none_or(|text| text.dpi() != dpi) {
            // Publish all three new raster scales together; old resources stay
            // drawable if any font preparation or GPU allocation is rejected.
            let rebuilt = TextContent::new(&self.renderer, self.font.clone(), dpi, counter)?;
            println!("TTF atlases rebuilt: logical sizes 16/24/48 px, physical scale {dpi:.3}");
            self.text = Some(rebuilt);
        }
        let changed = if page == TextPage::Motion {
            self.text
                .as_mut()
                .ok_or("text resources were not initialized")?
                .update_counter(&self.renderer, counter)?
        } else {
            false
        };
        let panel_width = (self.renderer.logical_size().0 - 64.0).max(1.0);
        let scroll = if page == TextPage::Motion {
            0.0
        } else {
            self.scroll
                .clamp(0.0, (820.0 - self.renderer.logical_size().1).max(0.0))
        };
        if self.panel_width != panel_width || self.panel_page != page || self.panel_scroll != scroll
        {
            let mut panels = ScreenScene::new(Color::BLACK)?;
            let bands: Vec<_> = if page == TextPage::Motion {
                vec![
                    (82.0, 64.0, Color::rgb8(27, 39, 60)),
                    (158.0, 66.0, Color::rgb8(27, 39, 60)),
                    (236.0, 86.0, Color::rgb8(27, 39, 60)),
                    (344.0, 72.0, Color::rgb8(49, 37, 28)),
                    (452.0, 94.0, Color::rgb8(21, 49, 43)),
                ]
            } else {
                let baselines = if page == TextPage::Fonts {
                    vec![142.0, 247.0, 352.0, 457.0, 562.0, 667.0]
                } else {
                    vec![142.0, 242.0, 335.0, 428.0, 521.0, 614.0, 707.0]
                };
                baselines
                    .into_iter()
                    .map(|baseline| (baseline - 70.0, 91.0, Color::rgb8(23, 32, 48)))
                    .collect()
            };
            for (y, height, color) in bands {
                panels.try_square_rect(
                    LogicalScreenPosition::new(32.0, y - scroll),
                    LogicalScreenVector::new(panel_width, height),
                    ShapeStyle::filled(color),
                )?;
            }
            self.panels = panels;
            self.panel_width = panel_width;
            self.panel_page = page;
            self.panel_scroll = scroll;
        }
        let mut frame = self
            .renderer
            .begin_frame(Color::rgb8(12, 16, 25), FrameBudget::default())?;
        frame.draw_screen_scene(&self.panels, FramePassOptions::new(0))?;
        if page == TextPage::Motion {
            self.text
                .as_ref()
                .ok_or("text resources were not initialized")?
                .draw(&mut frame, self.seconds as f32, panel_width)?;
            if let Some(text_acceptance) = &self.text_acceptance {
                text_acceptance.draw(&mut frame)?;
            }
        } else {
            self.gallery
                .as_ref()
                .ok_or("font gallery was not initialized")?
                .draw(&mut frame, page, panel_width, scroll)?;
        }
        let report = frame.present()?;
        if report.status() == RenderStatus::Drawn {
            self.drawn += 1;
            self.page_presents[page.index()] += 1;
        } else {
            self.skipped += 1;
            if self.skipped >= MAX_SKIPPED_ATTEMPTS
                && (self.acceptance_requested || self.frame_limit.is_some())
            {
                return Err(format!(
                    "TTF example did not complete: {} drawn, {} skipped attempts ({:?})",
                    self.drawn,
                    self.skipped,
                    report.status()
                )
                .into());
            }
        }
        if changed && self.drawn.is_multiple_of(120) {
            println!(
                "counter={counter}, drawn={}, skipped={}, dpi={dpi:.3}, text updated without rerasterizing cached glyphs",
                self.drawn, self.skipped,
            );
        }
        if self.acceptance_requested && self.drawn >= ACCEPTANCE_TEXT_PRESENTS {
            return Ok(true);
        }
        Ok(self.frame_limit.is_some_and(|limit| self.drawn >= limit))
    }
}

struct Application {
    demo: Option<Demo>,
    options: Options,
    font: FontFace,
    error: Option<String>,
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, events: &ActiveEventLoop) {
        if self.demo.is_some() {
            return;
        }
        match Demo::new(events, &self.options, self.font.clone()) {
            Ok(demo) => {
                demo.window.request_redraw();
                self.demo = Some(demo);
            }
            Err(error) => {
                self.error = Some(error.to_string());
                events.exit();
            }
        }
    }

    fn window_event(&mut self, events: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(demo) = &mut self.demo else {
            return;
        };
        if demo.window.id() != id {
            return;
        }
        let result: Result<bool> = match event {
            WindowEvent::CloseRequested => Ok(true),
            WindowEvent::Resized(size) => demo
                .renderer
                .resize_with_scale_factor(size.width, size.height, demo.window.scale_factor())
                .map(|_| false)
                .map_err(Into::into),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                let size = demo.window.inner_size();
                demo.renderer
                    .resize_with_scale_factor(size.width, size.height, scale_factor)
                    .map(|_| false)
                    .map_err(Into::into)
            }
            WindowEvent::MouseWheel { delta, .. } if !demo.acceptance_requested => {
                let movement = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => lines * 42.0,
                    MouseScrollDelta::PixelDelta(position) => {
                        (position.y / demo.window.scale_factor()) as f32
                    }
                };
                if movement.is_finite() {
                    demo.scroll = (demo.scroll - movement)
                        .clamp(0.0, (820.0 - demo.renderer.logical_size().1).max(0.0));
                }
                Ok(false)
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => events.exit(),
                    PhysicalKey::Code(KeyCode::Space) => demo.paused = !demo.paused,
                    PhysicalKey::Code(KeyCode::KeyR) => demo.seconds = 0.0,
                    PhysicalKey::Code(KeyCode::Digit1) => {
                        demo.page = TextPage::Fonts;
                        demo.scroll = 0.0;
                    }
                    PhysicalKey::Code(KeyCode::Digit2) => {
                        demo.page = TextPage::Unicode;
                        demo.scroll = 0.0;
                    }
                    PhysicalKey::Code(KeyCode::Digit3) => {
                        demo.page = TextPage::Motion;
                        demo.scroll = 0.0;
                    }
                    _ => {}
                }
                Ok(false)
            }
            WindowEvent::RedrawRequested => demo.redraw(),
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
        if let Some(demo) = &self.demo {
            demo.window.request_redraw();
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let options = Options::parse(std::env::args().skip(1))?;
    if options.help {
        println!(
            "cargo run --release --features text --example text_ui_updates -- [OPTIONS]\n\
             --font PATH   Use a caller-provided TTF/OTF instead of embedded DejaVu Sans\n\
             --page NAME   fonts / unicode (default) / motion; switch live with 1 / 2 / 3\n\
             --uncapped    Request Immediate presentation, with backend fallback\n\
             --frames N    Exit after N confirmed text presents\n\
             --acceptance  Run recovery checks, then 40 Drawn on each of the three pages\n\
             1/2/3 switch pages, wheel scrolls, Space pauses motion, R resets, Esc exits.\n\
             Moving the window between different-DPI outputs rebuilds glyph rasters."
        );
        return Ok(());
    }
    let font = FontFace::from_bytes(options.font_bytes()?, FontBudget::default())?;
    let mut application = Application {
        demo: None,
        options,
        font,
        error: None,
    };
    let events = EventLoop::new()?;
    events.set_control_flow(ControlFlow::Poll);
    events.run_app(&mut application)?;
    if let Some(error) = application.error {
        return Err(error.into());
    }
    let (drawn, skipped, acceptance_finished) =
        application.demo.as_ref().map_or((0, 0, false), |demo| {
            (
                demo.drawn,
                demo.skipped,
                demo.acceptance.is_none()
                    && demo
                        .text_acceptance
                        .as_ref()
                        .is_some_and(|text| text.restored())
                    && demo
                        .page_presents
                        .into_iter()
                        .all(|count| count >= PRESENTS_PER_PAGE),
            )
        });
    application
        .options
        .verify_completion(drawn, acceptance_finished)?;
    if application.options.acceptance {
        println!(
            "Real font acceptance passed: {drawn} Drawn frames, {skipped} skipped attempts; 40 per page (Fonts/Unicode/Motion), Japanese/math/combining/RTL plus restored runs. This is surface/API smoke, not a pixel oracle."
        );
    } else if application.options.frame_limit.is_some() {
        println!("TTF smoke passed: {drawn} Drawn frames, {skipped} skipped attempts");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn text_example_options_require_complete_bounded_inputs() {
        for invalid in [
            vec!["--font"],
            vec!["--frames"],
            vec!["--frames", "0"],
            vec!["--frames", "oops"],
            vec!["--unknown"],
            vec!["--acceptance", "--frames", "1"],
        ] {
            assert!(Options::parse(arguments(&invalid)).is_err());
        }
        let options = Options::parse(arguments(&[
            "--font",
            "a font.ttf",
            "--frames",
            "120",
            "--uncapped",
        ]))
        .unwrap();
        assert_eq!(options.font_path, Some(PathBuf::from("a font.ttf")));
        assert_eq!(options.frame_limit, Some(120));
        assert!(options.uncapped);
    }

    #[test]
    fn embedded_font_loads_without_system_font_discovery() {
        FontFace::from_bytes(EMBEDDED_FONT.to_vec(), FontBudget::default()).unwrap();
    }

    #[test]
    fn early_window_close_cannot_pass_acceptance_or_bounded_smoke() {
        let acceptance = Options {
            acceptance: true,
            ..Options::default()
        };
        assert!(acceptance.verify_completion(0, false).is_err());
        assert!(acceptance.verify_completion(119, true).is_err());
        assert!(acceptance.verify_completion(120, false).is_err());
        assert!(acceptance.verify_completion(120, true).is_ok());
        let smoke = Options {
            frame_limit: Some(7),
            ..Options::default()
        };
        assert!(smoke.verify_completion(0, false).is_err());
        assert!(smoke.verify_completion(6, false).is_err());
        assert!(smoke.verify_completion(7, false).is_ok());
        assert!(Options::default().verify_completion(0, false).is_ok());
    }
}
