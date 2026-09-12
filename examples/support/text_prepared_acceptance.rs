//! Public-API prepared text parity and transactional boundary checks.

use sim_engine::{
    FontBudget, FontBudgetResource, FontError, FontFace, GlyphRunBudget, LogicalPixels,
    PhysicalPerLogical, PreparedTextError, ShapedLineError, TextAtlas2d, TextAtlasBudget,
    TextDirection, TextError, TextLayoutBudget, TextRun2d, TextShapingSession, TextStyle,
    WgpuRenderer,
};

use super::Result;

pub fn verify(renderer: &WgpuRenderer) -> Result<()> {
    // Use the licensed fixture for this multi-script oracle. A caller-selected
    // display font need not cover Arabic; its own coverage is checked separately.
    let fixture_font = FontFace::from_bytes(
        include_bytes!("../assets/fonts/DejaVuSans.ttf").to_vec(),
        FontBudget::default(),
    )?;
    let font = &fixture_font;
    let budget = TextLayoutBudget::default();
    let atlas_budget = TextAtlasBudget::new(512, 512, 256, GlyphRunBudget::default())?;
    for direction in [TextDirection::Auto, TextDirection::Ltr, TextDirection::Rtl] {
        let style = TextStyle::new(
            LogicalPixels::new(24.0)?,
            PhysicalPerLogical::new(renderer.scale_factor() as f32)?,
        )?
        .with_direction(direction);
        let mut direct = TextAtlas2d::new(renderer, font.clone(), style, atlas_budget)?;
        let mut prepared = TextAtlas2d::new(renderer, font.clone(), style, atlas_budget)?;
        let mut session = TextShapingSession::new(font, style, budget)?;
        let mut shaped = session.shape_line("AV ffi Привет")?;
        let mut direct_run = direct.prepare(renderer, shaped.text(), budget)?;
        let mut prepared_run = prepared.prepare_from_shaped(renderer, &shaped, budget)?;
        parity(&direct, &direct_run, &prepared, &prepared_run)?;
        for text in [
            "Cafe\u{301} q\u{301}",
            "office AV",
            "مرحبا بالعالم",
            "   ",
            "",
            "AV",
        ] {
            session.update_line(&mut shaped, text)?;
            let direct_report = direct.update(renderer, &mut direct_run, text, budget)?;
            let prepared_report =
                prepared.update_from_shaped(renderer, &mut prepared_run, &shaped, budget)?;
            ensure(
                direct_report == prepared_report,
                "prepared and UTF-8 update work reports differ",
            )?;
            parity(&direct, &direct_run, &prepared, &prepared_run)?;
            ensure(
                metrics(&prepared_run)
                    == [
                        shaped.advance(),
                        shaped.ascent(),
                        shaped.descent(),
                        shaped.line_height(),
                    ],
                "CPU/GPU baseline metrics differ",
            )?;
        }

        let no_work = TextLayoutBudget::new(0, 0, 0, 0);
        let bytes = prepared_run.recovery_memory_bytes();
        let pointer = prepared_run.glyph_run().glyphs().as_ptr();
        let report = prepared.update_from_shaped(renderer, &mut prepared_run, &shaped, no_work)?;
        ensure(
            !report.changed()
                && report.instance_uploaded_bytes() == 0
                && report.cached_glyphs_added() == 0
                && prepared_run.recovery_memory_bytes() == bytes
                && prepared_run.glyph_run().glyphs().as_ptr() == pointer,
            "unchanged prepared text must preserve storage and report zero work even with zero work budgets",
        )?;

        let foreign_font = FontFace::from_bytes(font.font_data().to_vec(), FontBudget::default())?;
        let foreign = foreign_font.shape_line(shaped.text(), &style, &budget)?;
        ensure(
            matches!(
                prepared.update_from_shaped(renderer, &mut prepared_run, &foreign, no_work),
                Err(PreparedTextError::Shaped(ShapedLineError::FontMismatch))
            ),
            "unchanged UTF-8 must not bypass prepared font provenance",
        )?;
        let other_style =
            TextStyle::new(LogicalPixels::new(25.0)?, style.scale())?.with_direction(direction);
        let foreign = font.shape_line(shaped.text(), &other_style, &budget)?;
        ensure(
            matches!(
                prepared.update_from_shaped(renderer, &mut prepared_run, &foreign, no_work),
                Err(PreparedTextError::Shaped(ShapedLineError::StyleMismatch))
            ),
            "unchanged UTF-8 must not bypass prepared style provenance",
        )?;
        let other_scale = TextStyle::new(
            style.logical_em_size(),
            PhysicalPerLogical::new(style.scale().get() * 1.25)?,
        )?
        .with_direction(direction);
        let foreign = font.shape_line(shaped.text(), &other_scale, &budget)?;
        ensure(
            matches!(
                prepared.prepare_from_shaped(renderer, &foreign, budget),
                Err(PreparedTextError::Shaped(ShapedLineError::StyleMismatch))
            ),
            "prepared text must reject foreign raster DPI",
        )?;
        let other_direction = style.with_direction(if direction == TextDirection::Rtl {
            TextDirection::Ltr
        } else {
            TextDirection::Rtl
        });
        let foreign = font.shape_line(shaped.text(), &other_direction, &budget)?;
        ensure(
            matches!(
                prepared.prepare_from_shaped(renderer, &foreign, budget),
                Err(PreparedTextError::Shaped(ShapedLineError::StyleMismatch))
            ),
            "prepared text must reject foreign requested direction",
        )?;

        let glyphs = prepared_run.glyph_run().glyphs().to_vec();
        let entries = prepared.atlas().entries().to_vec();
        let original_metrics = metrics(&prepared_run);
        let changed = font.shape_line("ABC", &style, &budget)?;
        for (small, resource) in [
            (
                TextLayoutBudget::new(2, 10, 0, 0),
                FontBudgetResource::TextBytes,
            ),
            (
                TextLayoutBudget::new(10, 2, 0, 0),
                FontBudgetResource::ShapedGlyphs,
            ),
        ] {
            ensure(
                matches!(prepared.update_from_shaped(renderer, &mut prepared_run, &changed, small), Err(PreparedTextError::Shaped(ShapedLineError::Font(FontError::BudgetExceeded { resource: rejected, .. }))) if rejected == resource),
                "changed prepared text must honor exact input/output budgets",
            )?;
            ensure(
                prepared_run.text() == "AV"
                    && prepared_run.glyph_run().glyphs() == glyphs
                    && prepared.atlas().entries() == entries
                    && metrics(&prepared_run) == original_metrics
                    && prepared_run.recovery_memory_bytes() == bytes,
                "prepared budget failure changed drawable state or cache",
            )?;
        }
        ensure(
            matches!(
                direct.update_from_shaped(renderer, &mut prepared_run, &shaped, budget),
                Err(PreparedTextError::Text(TextError::AtlasMismatch))
            ),
            "prepared updates must reject foreign retained run identity",
        )?;
    }
    println!(
        "Prepared text acceptance: three directions, script/mark/ligature/whitespace parity, exact metrics/instances/cache bytes, provenance, no-op and failure preservation passed. Existing glyph pixel oracle covers the shared raster/draw path."
    );
    Ok(())
}

fn parity(
    left_atlas: &TextAtlas2d,
    left: &TextRun2d,
    right_atlas: &TextAtlas2d,
    right: &TextRun2d,
) -> Result<()> {
    ensure(
        left.text() == right.text()
            && left.glyph_run().glyphs() == right.glyph_run().glyphs()
            && metrics(left) == metrics(right)
            && left.recovery_memory_bytes() == right.recovery_memory_bytes()
            && left.glyph_run().gpu_allocation_bytes() == right.glyph_run().gpu_allocation_bytes()
            && left_atlas.cached_glyph_count() == right_atlas.cached_glyph_count()
            && left_atlas.atlas().entries() == right_atlas.atlas().entries()
            && left_atlas.recovery_memory_bytes() == right_atlas.recovery_memory_bytes()
            && left_atlas.gpu_allocation_bytes() == right_atlas.gpu_allocation_bytes(),
        "prepared route differs from UTF-8 route in exact retained state/resource accounting",
    )
}

fn metrics(run: &TextRun2d) -> [f32; 4] {
    [
        run.advance(),
        run.ascent(),
        run.descent(),
        run.line_height(),
    ]
}

fn ensure(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
