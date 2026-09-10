//! Host-owned tiny numeric font demonstrating retained updates and shared layout.
//! Space pauses, R resets, Esc exits. Optional --uncapped and --frames N.

#[path = "support/retained_ui_acceptance.rs"]
mod retained_ui_acceptance;

use std::{error::Error, sync::Arc, time::Instant};

use sim_engine::{
    Color, FrameBudget, FramePassOptions, GlyphAtlas2d, GlyphAtlasBudget, GlyphAtlasEntry, GlyphId,
    GlyphRun2d, GlyphRunBudget, Image2d, ImageBatch2d, ImageBatchBudget, ImageBatchPlacement,
    ImageBudget, ImageSampling, ImageSprite2d, ImageTexelRect, LogicalScreenPosition,
    LogicalScreenVector, LogicalViewport, LogicalViewportRegion, PositionedGlyph2d, RenderStatus,
    RendererPresentMode, ScreenClipRect, ScreenScene, ShapeStyle, WgpuRenderer,
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

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const DIGITS: [[u8; 5]; 10] = [
    [7, 5, 5, 5, 7],
    [2, 6, 2, 2, 7],
    [7, 1, 7, 4, 7],
    [7, 1, 7, 1, 7],
    [5, 5, 7, 1, 1],
    [7, 4, 7, 1, 7],
    [7, 4, 7, 5, 7],
    [7, 1, 1, 1, 1],
    [7, 5, 7, 5, 7],
    [7, 5, 7, 1, 7],
];

fn font() -> (Vec<u8>, Vec<GlyphAtlasEntry>) {
    let mut pixels = vec![0; 50 * 7 * 4];
    let mut entries = Vec::new();
    for (digit, rows) in DIGITS.iter().enumerate() {
        for (y, row) in rows.iter().enumerate() {
            for x in 0..3 {
                if row & (4 >> x) != 0 {
                    let start = ((y + 1) * 50 + digit * 5 + x + 1) * 4;
                    pixels[start..start + 4].fill(255);
                }
            }
        }
        entries.push(GlyphAtlasEntry::new(
            GlyphId::new(digit as u32),
            ImageTexelRect::new(digit as u32 * 5 + 1, 1, 3, 5).unwrap(),
        ));
    }
    (pixels, entries)
}

fn region(x: f32, y: f32, width: f32, height: f32) -> Result<LogicalViewportRegion> {
    Ok(LogicalViewportRegion::new(
        LogicalScreenPosition::new(x, y),
        LogicalViewport::new(width, height)?,
    )?)
}

fn layout(number: u64, output: &mut Vec<PositionedGlyph2d>) -> Result<()> {
    output.clear();
    let mut digits = [0_u32; 20];
    let mut count = 0;
    let mut value = number;
    loop {
        digits[count] = (value % 10) as u32;
        count += 1;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    for (index, digit) in digits[..count].iter().rev().enumerate() {
        output.push(PositionedGlyph2d::new(
            GlyphId::new(*digit),
            region(index as f32 * 32.0 - 2.0, -4.0, 24.0, 40.0)?,
            Color::WHITE,
        )?);
    }
    Ok(())
}

struct Demo {
    window: Arc<Window>,
    renderer: WgpuRenderer,
    atlas: GlyphAtlas2d,
    run: GlyphRun2d,
    image: Image2d,
    sprites: ImageBatch2d,
    glyphs: Vec<PositionedGlyph2d>,
    panels: ScreenScene,
    seconds: f64,
    last_frame: Instant,
    paused: bool,
    attempts: usize,
    frame_limit: Option<usize>,
    acceptance: Option<retained_ui_acceptance::RetainedUiAcceptance>,
}

impl Demo {
    fn new(
        events: &ActiveEventLoop,
        uncapped: bool,
        frame_limit: Option<usize>,
        acceptance: bool,
    ) -> Result<Self> {
        let window = Arc::new(events.create_window(Window::default_attributes()
            .with_title("0.3 text/UI: shared counter, tinted scrolling copy, moving atlas sprite | Space/R/Esc")
            .with_inner_size(LogicalSize::new(900.0, 500.0)))?);
        let size = window.inner_size();
        let options = WgpuRendererOptions::new(
            if uncapped {
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
            options,
        ))?;
        let notify = window.clone();
        renderer.set_pre_present_notify(move || notify.pre_present_notify());
        let acceptance = acceptance
            .then(|| retained_ui_acceptance::RetainedUiAcceptance::new(&mut renderer))
            .transpose()?;
        let (pixels, entries) = font();
        let atlas = renderer.create_glyph_atlas(
            50,
            7,
            pixels.clone(),
            entries,
            GlyphAtlasBudget::default(),
        )?;
        let mut glyphs = Vec::with_capacity(20);
        layout(9, &mut glyphs)?;
        let run =
            renderer.create_glyph_run(&atlas, glyphs.clone(), GlyphRunBudget::new(20, 8192)?)?;
        let image = renderer.create_image_rgba8(50, 7, pixels, ImageBudget::default())?;
        let sprites = renderer.create_image_batch(
            &image,
            vec![ImageSprite2d::new(
                ImageTexelRect::new(1, 1, 3, 5)?,
                region(0.0, 0.0, 36.0, 60.0)?,
                Color::WHITE,
            )?],
            ImageBatchBudget::new(16, 8192)?,
        )?;
        Ok(Self {
            window,
            renderer,
            atlas,
            run,
            image,
            sprites,
            glyphs,
            panels: ScreenScene::new(Color::BLACK)?,
            seconds: 0.0,
            last_frame: Instant::now(),
            paused: false,
            attempts: 0,
            frame_limit,
            acceptance,
        })
    }

    fn redraw(&mut self) -> Result<bool> {
        if let Some(acceptance) = &mut self.acceptance {
            return acceptance.step(&mut self.renderer);
        }
        let now = Instant::now();
        if !self.paused {
            self.seconds += now.duration_since(self.last_frame).as_secs_f64().min(0.1);
        }
        self.last_frame = now;
        let counter = 9 + (self.seconds * 10.0) as u64;
        layout(counter, &mut self.glyphs)?;
        let run_update =
            self.renderer
                .update_glyph_run(&self.atlas, &mut self.run, &self.glyphs)?;
        let angle = self.seconds as f32;
        let digit = (counter % 10) as u32;
        let sprite = ImageSprite2d::new(
            ImageTexelRect::new(digit * 5 + 1, 1, 3, 5)?,
            region(360.0 + angle.sin() * 300.0, 320.0, 36.0, 60.0)?,
            Color::rgb8(255, 184, 75),
        )?;
        let sprite_update =
            self.renderer
                .update_image_batch(&self.image, &mut self.sprites, &[sprite])?;
        self.panels.clear();
        for (y, color) in [
            (36.0, Color::rgb8(27, 39, 60)),
            (148.0, Color::rgb8(21, 49, 43)),
            (292.0, Color::rgb8(49, 37, 28)),
        ] {
            self.panels.try_square_rect(
                LogicalScreenPosition::new(24.0, y),
                LogicalScreenVector::new(810.0, 104.0),
                ShapeStyle::filled(color),
            )?;
        }
        let mut frame = self
            .renderer
            .begin_frame(Color::rgb8(12, 16, 25), FrameBudget::default())?;
        frame.draw_screen_scene(&self.panels, FramePassOptions::new(0))?;
        frame.draw_glyph_run_placed(
            &self.atlas,
            &self.run,
            ImageBatchPlacement::new(
                LogicalScreenVector::new(56.0, 72.0),
                Color::rgb8(108, 203, 255),
            )?,
            ImageSampling::Nearest,
            FramePassOptions::new(1),
        )?;
        frame.draw_glyph_run_placed(
            &self.atlas,
            &self.run,
            ImageBatchPlacement::new(
                LogicalScreenVector::new(360.0 + angle.sin() * 460.0, 184.0),
                Color::rgb8(90, 240, 155).with_alpha(0.5),
            )?,
            ImageSampling::Nearest,
            FramePassOptions::new(2).with_clip(ScreenClipRect::from_min_size(
                LogicalScreenPosition::new(24.0, 148.0),
                LogicalScreenVector::new(810.0, 104.0),
            )?),
        )?;
        frame.draw_image_batch(
            &self.image,
            &self.sprites,
            ImageSampling::Nearest,
            FramePassOptions::new(3),
        )?;
        let report = frame.present()?;
        self.attempts += 1;
        if self.attempts.is_multiple_of(120) || run_update.instances().replaced_instance_buffer() {
            println!(
                "count={counter} glyph_capacity={} glyph_upload={} glyph_buffer_grew={} sprite_upload={} sprite_buffer_grew={} present={:?}",
                run_update.capacity(),
                run_update.instances().uploaded_instance_bytes(),
                run_update.instances().replaced_instance_buffer(),
                sprite_update.uploaded_instance_bytes(),
                sprite_update.replaced_instance_buffer(),
                report.status()
            );
        }
        if report.status() == RenderStatus::Drawn {
            self.window.request_redraw();
        }
        Ok(self.frame_limit.is_some_and(|limit| self.attempts >= limit))
    }
}

struct Application {
    demo: Option<Demo>,
    uncapped: bool,
    frame_limit: Option<usize>,
    acceptance: bool,
    error: Option<String>,
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, events: &ActiveEventLoop) {
        if self.demo.is_some() {
            return;
        }
        match Demo::new(events, self.uncapped, self.frame_limit, self.acceptance) {
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
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => events.exit(),
                    PhysicalKey::Code(KeyCode::Space) => demo.paused = !demo.paused,
                    PhysicalKey::Code(KeyCode::KeyR) => demo.seconds = 0.0,
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
    let arguments = std::env::args().collect::<Vec<_>>();
    let frame_limit = arguments
        .windows(2)
        .find(|pair| pair[0] == "--frames")
        .map(|pair| pair[1].parse())
        .transpose()?;
    let mut application = Application {
        demo: None,
        uncapped: arguments.iter().any(|arg| arg == "--uncapped"),
        frame_limit,
        acceptance: arguments.iter().any(|arg| arg == "--acceptance"),
        error: None,
    };
    let events = EventLoop::new()?;
    events.set_control_flow(ControlFlow::Poll);
    events.run_app(&mut application)?;
    if let Some(error) = application.error {
        return Err(error.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counter_layout_grows_without_reversing_digits_and_preserves_bearings() {
        let mut glyphs = Vec::with_capacity(20);
        layout(100, &mut glyphs).unwrap();
        assert_eq!(
            glyphs
                .iter()
                .map(|glyph| glyph.glyph().value())
                .collect::<Vec<_>>(),
            vec![1, 0, 0]
        );
        assert_eq!(
            glyphs[0].destination().origin(),
            LogicalScreenPosition::new(-2.0, -4.0)
        );
        layout(0, &mut glyphs).unwrap();
        assert_eq!(glyphs.len(), 1);
        assert_eq!(glyphs.capacity(), 20);
    }
}
