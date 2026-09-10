//! Public text API checks followed by real draws after logical-device recovery.

use sim_engine::{
    Color, FontError, FontFace, FrameComposer, FramePassOptions, GlyphRunBudget,
    ImageBatchPlacement, ImageSampling, LogicalPixels, LogicalScreenVector, PhysicalPerLogical,
    PositionedGlyph2d, TextAtlas2d, TextAtlasBudget, TextError, TextLayoutBudget, TextRun2d,
    TextStyle, WgpuRenderer,
};

use super::Result;

pub struct TextAcceptance {
    atlas: TextAtlas2d,
    run: TextRun2d,
    glyphs: Vec<PositionedGlyph2d>,
    metrics: [f32; 4],
    tiny_atlas: TextAtlas2d,
    tiny_run: TextRun2d,
    restored: bool,
}

impl TextAcceptance {
    pub fn new(renderer: &WgpuRenderer, font: FontFace) -> Result<Self> {
        let style = TextStyle::new(
            LogicalPixels::new(24.0)?,
            PhysicalPerLogical::new(renderer.scale_factor() as f32)?,
        )?;
        let budget = TextLayoutBudget::default();
        let atlas_budget = TextAtlasBudget::new(512, 256, 128, GlyphRunBudget::default())?;
        let mut atlas = TextAtlas2d::new(renderer, font.clone(), style, atlas_budget)?;
        let mut run = atlas.prepare(renderer, "AV Привет", budget)?;
        let glyphs = run.glyph_run().glyphs().to_vec();
        let baseline_metrics = metrics(&run);
        let cache_count = atlas.cached_glyph_count();
        let unchanged = atlas.update(renderer, &mut run, "AV Привет", budget)?;
        ensure(
            !unchanged.changed()
                && unchanged.instance_uploaded_bytes() == 0
                && unchanged.cached_glyphs_added() == 0
                && atlas.cached_glyph_count() == cache_count,
            "unchanged TTF update must perform no instance upload or glyph insertion",
        )?;
        for (text, missing) in [("line\nbreak", false), ("\u{10ffff}", true)] {
            let error = atlas.update(renderer, &mut run, text, budget);
            ensure(
                matches!(error, Err(TextError::Font(FontError::MissingGlyph { .. }))) && missing
                    || matches!(
                        error,
                        Err(TextError::Font(FontError::UnsupportedText { .. }))
                    ) && !missing,
                "invalid TTF text must return its specific structured rejection",
            )?;
            ensure(
                run.text() == "AV Привет"
                    && run.glyph_run().glyphs() == glyphs
                    && metrics(&run) == baseline_metrics,
                "failed text update changed retained text, glyphs or baseline metrics",
            )?;
        }
        let empty = atlas.prepare(renderer, "", budget)?;
        let spaces = atlas.prepare(renderer, "   ", budget)?;
        ensure(
            empty.advance() == 0.0
                && empty.glyph_run().glyph_count() == 0
                && spaces.advance() > 0.0
                && spaces.glyph_run().glyph_count() == 0,
            "empty/spacing text must preserve advance without drawable quads",
        )?;
        let mut foreign = TextAtlas2d::new(renderer, font.clone(), style, atlas_budget)?;
        ensure(
            matches!(
                foreign.update(renderer, &mut run, "A", budget),
                Err(TextError::AtlasMismatch)
            ),
            "same-font foreign atlas must reject another atlas's retained run",
        )?;
        let mut limited = TextAtlas2d::new(
            renderer,
            font.clone(),
            style,
            TextAtlasBudget::new(
                64,
                64,
                16,
                GlyphRunBudget::new(100, GlyphRunBudget::RETAINED_BYTES_PER_GLYPH)?,
            )?,
        )?;
        ensure(
            matches!(
                limited.prepare(renderer, "AB", budget),
                Err(TextError::Glyph(
                    sim_engine::GlyphError::MetadataBudgetExceeded { .. }
                ))
            ) && limited.cached_glyph_count() == 0
                && limited.atlas().entries().is_empty(),
            "known run byte overflow must reject before rasterizing or caching glyphs",
        )?;

        let dot = font.shape_line(".", &style, &budget)?;
        let raster = font.rasterize_glyph(dot.glyphs()[0].glyph_id(), &style, &budget)?;
        // One complete padded glyph fills the entire atlas. Any distinct
        // drawable glyph must fail without replacing the existing dot.
        let tiny_budget = TextAtlasBudget::new(
            raster.width() + 4,
            raster.height() + 4,
            8,
            GlyphRunBudget::default(),
        )?;
        let mut tiny_atlas = TextAtlas2d::new(renderer, font, style, tiny_budget)?;
        let tiny_run = tiny_atlas.prepare(renderer, ".", budget)?;
        let tiny_glyphs = tiny_run.glyph_run().glyphs().to_vec();
        let tiny_entries = tiny_atlas.atlas().entries().to_vec();
        ensure(
            matches!(
                tiny_atlas.prepare(renderer, "A", budget),
                Err(TextError::AtlasFull)
            ),
            "a completely filled tiny atlas must reject another drawable glyph",
        )?;
        ensure(
            tiny_run.text() == "."
                && tiny_run.glyph_run().glyphs() == tiny_glyphs
                && tiny_atlas.atlas().entries() == tiny_entries,
            "atlas-full rejection changed the existing drawable run or atlas entry",
        )?;
        validate_cache(&atlas)?;
        validate_cache(&tiny_atlas)?;
        validate_late_update_failure(renderer, atlas.font().clone(), style)?;
        Ok(Self {
            atlas,
            run,
            glyphs,
            metrics: baseline_metrics,
            tiny_atlas,
            tiny_run,
            restored: false,
        })
    }

    pub fn restore_after_recovery(&mut self, renderer: &WgpuRenderer) -> Result<()> {
        ensure(
            self.atlas
                .update(
                    renderer,
                    &mut self.run,
                    "AV Привет",
                    TextLayoutBudget::default(),
                )
                .is_err(),
            "even unchanged TTF updates must reject stale GPU resources",
        )?;
        self.atlas.restore(renderer)?;
        self.atlas.restore_run(renderer, &mut self.run)?;
        self.tiny_atlas.restore(renderer)?;
        self.tiny_atlas.restore_run(renderer, &mut self.tiny_run)?;
        ensure(
            self.run.text() == "AV Привет"
                && self.run.glyph_run().glyphs() == self.glyphs
                && metrics(&self.run) == self.metrics,
            "TTF resource recovery changed text, glyph positions or baseline metrics",
        )?;
        let unchanged = self.atlas.update(
            renderer,
            &mut self.run,
            "AV Привет",
            TextLayoutBudget::default(),
        )?;
        ensure(
            !unchanged.changed() && unchanged.instance_uploaded_bytes() == 0,
            "unchanged restored text must not upload glyph instances",
        )?;
        validate_cache(&self.atlas)?;
        self.restored = true;
        println!(
            "Public TTF contracts passed: no-op/update rejection/empty/space/foreign atlas/full atlas/recovery. Drawing restored runs next."
        );
        Ok(())
    }

    pub fn restored(&self) -> bool {
        self.restored
    }

    pub fn draw<'a>(&'a self, frame: &mut FrameComposer<'a>) -> Result<()> {
        ensure(
            self.restored,
            "TTF recovery must precede the post-recovery draw",
        )?;
        for (atlas, run, x) in [
            (&self.atlas, &self.run, 32.0),
            (&self.tiny_atlas, &self.tiny_run, 950.0),
        ] {
            frame.draw_glyph_run_placed(
                atlas.atlas(),
                run.glyph_run(),
                ImageBatchPlacement::new(LogicalScreenVector::new(x, 623.0), Color::WHITE)?,
                ImageSampling::Linear,
                FramePassOptions::new(3),
            )?;
        }
        Ok(())
    }
}

fn validate_late_update_failure(
    renderer: &WgpuRenderer,
    font: FontFace,
    style: TextStyle,
) -> Result<()> {
    let budget = TextLayoutBudget::default();
    let mut atlas = TextAtlas2d::new(
        renderer,
        font,
        style,
        TextAtlasBudget::new(256, 256, 2, GlyphRunBudget::default())?,
    )?;
    let mut run = atlas.prepare(renderer, "A", budget)?;
    let glyphs = run.glyph_run().glyphs().to_vec();
    let baseline_metrics = metrics(&run);
    ensure(
        matches!(
            atlas.update(renderer, &mut run, "BC", budget),
            Err(TextError::GlyphCacheFull)
        ),
        "late text update must fail when its final glyph exceeds the cache budget",
    )?;
    ensure(
        atlas.cached_glyph_count() == 2
            && run.text() == "A"
            && run.glyph_run().glyphs() == glyphs
            && metrics(&run) == baseline_metrics,
        "late cache-warming failure must preserve the old drawable text and metrics",
    )?;
    validate_cache(&atlas)
}

fn metrics(run: &TextRun2d) -> [f32; 4] {
    [
        run.advance(),
        run.ascent(),
        run.descent(),
        run.line_height(),
    ]
}

fn validate_cache(atlas: &TextAtlas2d) -> Result<()> {
    ensure(
        atlas.cached_glyph_count() <= atlas.budget().max_cached_glyphs()
            && atlas.atlas().entries().len() <= atlas.cached_glyph_count(),
        "text cache metadata exceeded its explicit count budget",
    )?;
    let (width, height) = atlas.atlas().size();
    for entry in atlas.atlas().entries() {
        let source = entry.source();
        ensure(
            source
                .x()
                .checked_add(source.width())
                .is_some_and(|right| right <= width)
                && source
                    .y()
                    .checked_add(source.height())
                    .is_some_and(|bottom| bottom <= height),
            "cached glyph source extends outside its fixed physical atlas",
        )?;
    }
    Ok(())
}

fn ensure(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
