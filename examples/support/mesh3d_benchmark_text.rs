//! Pre-shaped, changing labels. Host shaping is outside measured frame work.

use super::cases::Result;
use sim_engine::{
    Color, FontBudget, FontFace, FrameComposer, FramePassOptions, GlyphRunBudget,
    ImageBatchPlacement, ImageSampling, LogicalPixels, LogicalScreenVector, PhysicalPerLogical,
    ShapedLine, TextAtlas2d, TextAtlasBudget, TextLayoutBudget, TextRun2d, TextShapingSession,
    TextStyle, WgpuRenderer,
};

pub struct Labels {
    atlas: TextAtlas2d,
    run: TextRun2d,
    lines: [ShapedLine; 2],
}

impl Labels {
    pub fn new(renderer: &WgpuRenderer) -> Result<Self> {
        let font = FontFace::from_bytes(
            include_bytes!("../assets/fonts/DejaVuSans.ttf").to_vec(),
            FontBudget::default(),
        )?;
        let style = TextStyle::new(
            LogicalPixels::new(24.0)?,
            PhysicalPerLogical::new(renderer.scale_factor() as f32)?,
        )?;
        let budget = TextLayoutBudget::default();
        let mut session = TextShapingSession::new(&font, style, budget)?;
        let lines = [
            session.shape_line("Engine: temperature 1200 K / Привет")?,
            session.shape_line("Engine: temperature 1400 K / Привет")?,
        ];
        let mut atlas = TextAtlas2d::new(
            renderer,
            font.clone(),
            style,
            TextAtlasBudget::new(512, 256, 128, GlyphRunBudget::default())?,
        )?;
        let mut run = atlas.prepare_from_shaped(renderer, &lines[0], budget)?;
        // Warm all glyphs, so measured work is changing prepared instances, not font loading.
        atlas.update_from_shaped(renderer, &mut run, &lines[1], budget)?;
        Ok(Self { atlas, run, lines })
    }

    pub fn update(&mut self, renderer: &WgpuRenderer, frame: usize) -> Result<usize> {
        Ok(self
            .atlas
            .update_from_shaped(
                renderer,
                &mut self.run,
                &self.lines[frame % 2],
                TextLayoutBudget::default(),
            )?
            .instance_uploaded_bytes())
    }

    pub fn draw<'a>(&'a self, frame: &mut FrameComposer<'a>) -> Result<()> {
        frame.draw_glyph_run_placed(
            self.atlas.atlas(),
            self.run.glyph_run(),
            ImageBatchPlacement::new(LogicalScreenVector::new(24.0, 48.0), Color::WHITE)?,
            ImageSampling::Linear,
            FramePassOptions::new(1),
        )?;
        Ok(())
    }

    pub fn cpu_bytes(&self) -> usize {
        self.atlas.recovery_memory_bytes() + self.run.recovery_memory_bytes()
    }
    pub fn gpu_bytes(&self) -> usize {
        self.atlas.gpu_allocation_bytes() + self.run.glyph_run().gpu_allocation_bytes()
    }
}
