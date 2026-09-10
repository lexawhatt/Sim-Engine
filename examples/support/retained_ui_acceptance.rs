//! Public-API retained UI acceptance fixture for a real window renderer.
//!
//! Drive `step` once per `RedrawRequested`, allowing the host event loop to
//! process resize/scale events between attempts. This verifies API contracts and
//! actual surface submission, not screenshot pixels or GPU performance.
//! One successful run deliberately replaces the renderer's logical device.

use std::error::Error;

use sim_engine::{
    Color, FrameBudget, FrameComposerError, FramePassOptions, FrameSourceKind, GlyphAtlas2d,
    GlyphAtlasBudget, GlyphAtlasEntry, GlyphError, GlyphId, GlyphRun2d, GlyphRunBudget, Image2d,
    ImageBatch2d, ImageBatchBudget, ImageBatchPlacement, ImageBudget, ImageError, ImageSampling,
    ImageSprite2d, ImageTexelRect, LogicalScreenPosition, LogicalScreenVector, LogicalViewport,
    LogicalViewportRegion, PositionedGlyph2d, RenderStatus, ScreenClipRect, ScreenScene,
    ShapeStyle, WgpuRenderer,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const MAX_SKIPPED_ATTEMPTS: usize = 120;

pub struct RetainedUiAcceptance {
    atlas: GlyphAtlas2d,
    run: GlyphRun2d,
    image: Image2d,
    batch: ImageBatch2d,
    panel: ScreenScene,
    recovered: bool,
    warmed: bool,
    confirmed_presents: usize,
    complete: bool,
    skipped_attempts: usize,
}

impl RetainedUiAcceptance {
    /// Exercises allocation-aware updates before any surface draw. Construct
    /// ordinary demo resources after this fixture finishes: recovery makes all
    /// resources created before it stale.
    pub fn new(renderer: &mut WgpuRenderer) -> Result<Self> {
        ensure(
            renderer.remaining_device_recoveries() > 0,
            "retained UI acceptance requires one available device recovery",
        )?;
        let atlas = create_atlas(renderer)?;
        let run = exercise_glyph_updates(renderer, &atlas)?;
        let image = renderer.create_image_rgba8(2, 1, vec![255; 8], ImageBudget::default())?;
        let batch = exercise_sprite_updates(renderer, &image)?;
        let mut panel = ScreenScene::new(Color::BLACK)?;
        panel.try_square_rect(
            LogicalScreenPosition::new(18.0, 18.0),
            LogicalScreenVector::new(32.0, 32.0),
            ShapeStyle::filled(Color::rgb(0.0, 0.0, 1.0).with_alpha(0.5)),
        )?;
        ensure(
            ImageBatchPlacement::new(LogicalScreenVector::new(f32::MAX, 0.0), Color::WHITE)
                == Err(ImageError::InvalidPlacement),
            "nonportable placement must be rejected",
        )?;
        ensure(
            ScreenClipRect::from_min_size(
                LogicalScreenPosition::new(0.0, 0.0),
                LogicalScreenVector::new(0.0, 0.0),
            ) == Err(sim_engine::SceneError::InvalidScreenClip),
            "explicitly empty clip must reject at construction",
        )?;
        let mut warm_cpu_bytes = None;
        for _ in 0..2 {
            {
                let mut frame = renderer
                    .begin_frame(Color::BLACK, FrameBudget::new(1, 100, 1000, 4096, 4096, 10))?;
                frame.draw_glyph_run(
                    &atlas,
                    &run,
                    ImageSampling::Nearest,
                    FramePassOptions::default(),
                )?;
                let before = frame.planned_statistics();
                ensure(
                    matches!(
                        frame.draw_glyph_run(
                            &atlas,
                            &run,
                            ImageSampling::Nearest,
                            FramePassOptions::default()
                        ),
                        Err(FrameComposerError::BudgetExceeded { .. })
                    ),
                    "over-budget composer append must be rejected",
                )?;
                ensure(
                    frame.planned_statistics() == before,
                    "rejected composer append changed planned counters",
                )?;
                // Abort without present: retained references must leave the cache.
            }
            let cache = renderer.frame_cache_statistics();
            ensure(
                cache.binding_count() == 0,
                "aborted composer created a GPU binding",
            )?;
            if let Some(previous) = warm_cpu_bytes {
                ensure(
                    cache.cpu_bytes() == previous,
                    "repeated aborted composer grew retained storage",
                )?;
            }
            warm_cpu_bytes = Some(cache.cpu_bytes());
        }
        Ok(Self {
            atlas,
            run,
            image,
            batch,
            panel,
            recovered: false,
            warmed: false,
            confirmed_presents: 0,
            complete: false,
            skipped_attempts: 0,
        })
    }

    /// Returns true only after two confirmed surface presentations before and
    /// after public device recovery. A skipped frame is retried on a later
    /// redraw, never treated as a successful presentation.
    pub fn step(&mut self, renderer: &mut WgpuRenderer) -> Result<bool> {
        if self.complete {
            return Ok(true);
        }
        let (width, height) = renderer.logical_size();
        if renderer.size().0 == 0 || renderer.size().1 == 0 {
            return self.skipped("zero-size surface");
        }
        let retained_cpu = self.run.recovery_memory_bytes();
        let retained_gpu = self.run.gpu_allocation_bytes();
        let glyph_pointer = self.run.glyphs().as_ptr();
        let mut frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
        // Intentionally submit in a different order from the painter keys.
        frame.draw_screen_scene(&self.panel, FramePassOptions::new(1))?;
        frame.draw_glyph_run_placed(
            &self.atlas,
            &self.run,
            placement(8.0, 8.0, Color::rgb(1.0, 0.0, 0.0))?,
            ImageSampling::Nearest,
            FramePassOptions::new(0),
        )?;
        frame.draw_glyph_run_placed(
            &self.atlas,
            &self.run,
            placement(24.0, 24.0, Color::rgb(0.0, 1.0, 0.0).with_alpha(0.5))?,
            ImageSampling::Nearest,
            FramePassOptions::new(2).with_clip(clip(24.0, 24.0, 16.0, 24.0)?),
        )?;
        for (index, (x, y)) in [
            (-4.0, 60.0),
            (width - 4.0, 60.0),
            (60.0, -4.0),
            (60.0, height - 4.0),
        ]
        .into_iter()
        .enumerate()
        {
            frame.draw_glyph_run_placed(
                &self.atlas,
                &self.run,
                placement(x, y, Color::WHITE)?,
                ImageSampling::Nearest,
                FramePassOptions::new(3 + index as i32),
            )?;
        }
        frame.draw_image_batch_placed(
            &self.image,
            &self.batch,
            placement(48.0, 48.0, Color::rgb(1.0, 1.0, 0.0).with_alpha(0.5))?,
            ImageSampling::Nearest,
            FramePassOptions::new(7).with_clip(clip(48.0, 48.0, 24.0, 24.0)?),
        )?;
        frame.draw_glyph_run_placed(
            &self.atlas,
            &self.run,
            ImageBatchPlacement::default(),
            ImageSampling::Nearest,
            // A valid clip outside the target has an empty effective scissor.
            FramePassOptions::new(8).with_clip(clip(width + 16.0, height + 16.0, 1.0, 1.0)?),
        )?;
        let report = frame.present()?;
        if let RenderStatus::Skipped(reason) = report.status() {
            ensure(
                report.surface_present_count() == 0,
                "skipped frame unexpectedly reports a surface presentation",
            )?;
            return self.skipped(&format!("{reason:?}"));
        }
        ensure(
            report.surface_present_count() == 1,
            "Drawn acceptance frame must request exactly one surface presentation",
        )?;
        ensure(
            report.statistics().source_counts().glyph_runs() == 7
                && report.statistics().source_counts().images() == 1,
            "composer did not account for all shared glyph and sprite placements",
        )?;
        ensure(
            self.run.recovery_memory_bytes() == retained_cpu
                && self.run.gpu_allocation_bytes() == retained_gpu
                && self.run.glyphs().as_ptr() == glyph_pointer,
            "presentation-only placement changed retained glyph storage",
        )?;
        self.skipped_attempts = 0;
        self.confirmed_presents += 1;
        let cache = renderer.frame_cache_statistics();
        let limits = renderer.frame_cache_budget();
        ensure(
            cache.cpu_bytes() <= limits.max_cpu_bytes()
                && cache.uniform_bytes() <= limits.max_uniform_bytes()
                && cache.texture_bytes() <= limits.max_texture_bytes()
                && cache.binding_count() <= limits.max_bindings(),
            "composition cache exceeded configured idle retention limits",
        )?;
        if !self.warmed {
            self.warmed = true;
            return Ok(false);
        }
        ensure(
            cache.created_buffers() == 0
                && cache.created_bind_groups() == 0
                && cache.uploaded_uniform_bytes() == 0,
            "unchanged second placement frame created GPU objects or re-uploaded uniforms",
        )?;
        if self.recovered {
            self.complete = true;
            println!(
                "retained_ui_acceptance=pass confirmed_presents={} public_recoveries=1 \
                 glyph_cpu_bytes={} glyph_gpu_bytes={} sprite_cpu_bytes={} \
                 sprite_gpu_bytes={} pixel_oracle=false",
                self.confirmed_presents,
                self.run.recovery_memory_bytes(),
                self.run.gpu_allocation_bytes(),
                self.batch.recovery_memory_bytes(),
                self.batch.gpu_allocation_bytes(),
            );
            return Ok(true);
        }
        pollster::block_on(renderer.recover_device_and_surface())?;
        let empty_cache = renderer.frame_cache_statistics();
        ensure(
            empty_cache.cpu_bytes() == 0
                && empty_cache.binding_count() == 0
                && empty_cache.uniform_bytes() == 0
                && empty_cache.texture_bytes() == 0,
            "logical-device recovery retained a previous-generation composition cache",
        )?;
        self.verify_stale_and_restore(renderer)?;
        self.recovered = true;
        self.warmed = false;
        Ok(false)
    }

    fn skipped(&mut self, reason: &str) -> Result<bool> {
        self.skipped_attempts += 1;
        if self.skipped_attempts >= MAX_SKIPPED_ATTEMPTS {
            return Err(format!(
                "retained UI acceptance could not present after {} attempts: {reason}",
                self.skipped_attempts,
            )
            .into());
        }
        Ok(false)
    }

    fn verify_stale_and_restore(&mut self, renderer: &mut WgpuRenderer) -> Result<()> {
        let glyphs = self.run.glyphs().to_vec();
        let sprites = self.batch.sprites().to_vec();
        ensure(
            renderer.update_glyph_run(&self.atlas, &mut self.run, &glyphs)
                == Err(GlyphError::Image(ImageError::RendererMismatch)),
            "stale glyph update must reject the previous renderer generation",
        )?;
        ensure(
            renderer.update_image_batch(&self.image, &mut self.batch, &sprites)
                == Err(ImageError::RendererMismatch),
            "stale sprite update must reject the previous renderer generation",
        )?;
        {
            let mut frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
            ensure(
                frame.draw_glyph_run_placed(
                    &self.atlas,
                    &self.run,
                    ImageBatchPlacement::default(),
                    ImageSampling::Nearest,
                    FramePassOptions::default(),
                ) == Err(FrameComposerError::RendererMismatch {
                    source: FrameSourceKind::Glyph,
                }),
                "composer must reject a stale glyph run",
            )?;
            ensure(
                frame.draw_image_batch_placed(
                    &self.image,
                    &self.batch,
                    ImageBatchPlacement::default(),
                    ImageSampling::Nearest,
                    FramePassOptions::default(),
                ) == Err(FrameComposerError::RendererMismatch {
                    source: FrameSourceKind::Image,
                }),
                "composer must reject a stale sprite batch",
            )?;
            // Dropping this failed composer must release references and counters.
        }
        let atlas = renderer.restore_glyph_atlas(&self.atlas)?;
        let run = renderer.restore_glyph_run(&atlas, &self.run)?;
        let image = renderer.restore_image(&self.image)?;
        let batch = renderer.restore_image_batch(&image, &self.batch)?;
        ensure(
            run.glyphs() == glyphs && run.bounds() == self.run.bounds(),
            "public glyph restoration changed layout or bounds",
        )?;
        ensure(
            batch.sprites() == sprites,
            "public sprite restoration changed source or placement",
        )?;
        self.atlas = atlas;
        self.run = run;
        self.image = image;
        self.batch = batch;
        ensure(
            renderer
                .update_glyph_run(&self.atlas, &mut self.run, &glyphs)?
                .instances()
                .uploaded_instance_bytes()
                == 0,
            "unchanged restored glyph run uploaded instances",
        )?;
        ensure(
            renderer
                .update_image_batch(&self.image, &mut self.batch, &sprites)?
                .uploaded_instance_bytes()
                == 0,
            "unchanged restored sprite batch uploaded instances",
        )?;
        Ok(())
    }
}

fn create_atlas(renderer: &WgpuRenderer) -> Result<GlyphAtlas2d> {
    Ok(renderer.create_glyph_atlas(
        2,
        1,
        vec![255; 8],
        vec![
            GlyphAtlasEntry::new(GlyphId::new(1), ImageTexelRect::new(0, 0, 1, 1)?),
            GlyphAtlasEntry::new(GlyphId::new(2), ImageTexelRect::new(1, 0, 1, 1)?),
        ],
        GlyphAtlasBudget::default(),
    )?)
}

fn exercise_glyph_updates(renderer: &WgpuRenderer, atlas: &GlyphAtlas2d) -> Result<GlyphRun2d> {
    let first = glyph(1, -2.0, Color::WHITE)?;
    let probe = renderer.create_glyph_run(atlas, vec![first], GlyphRunBudget::default())?;
    let one_glyph_bytes = probe.recovery_memory_bytes();
    drop(probe);
    // Count limit deliberately admits five; the byte limit admits exactly four.
    let budget = GlyphRunBudget::new(5, one_glyph_bytes * 4)?;
    let mut run = renderer.create_glyph_run(atlas, vec![first], budget)?;
    let pointer = run.glyphs().as_ptr();
    let unchanged = renderer.update_glyph_run(atlas, &mut run, &[first])?;
    ensure(
        unchanged.instances().uploaded_instance_bytes() == 0
            && !unchanged.instances().replaced_instance_buffer(),
        "unchanged glyph layout must not upload or replace its instance buffer",
    )?;
    let changed = glyph(2, 2.0, Color::WHITE.with_alpha(0.5))?;
    let stable = renderer.update_glyph_run(atlas, &mut run, &[changed])?;
    ensure(
        stable.instances().uploaded_instance_bytes() > 0
            && !stable.instances().replaced_instance_buffer()
            && run.glyphs().as_ptr() == pointer,
        "same-capacity glyph update must reuse CPU and GPU storage",
    )?;
    let glyphs = [
        first,
        changed,
        glyph(1, 14.0, Color::WHITE)?,
        glyph(2, 28.0, Color::WHITE.with_alpha(0.5))?,
    ];
    let growth = renderer.update_glyph_run(atlas, &mut run, &glyphs)?;
    ensure(
        growth.glyph_count() == 4
            && growth.instances().replaced_instance_buffer()
            && growth.retained_bytes() == budget.max_retained_bytes()
            && growth.peak_retained_bytes() <= budget.max_retained_bytes() + one_glyph_bytes,
        "glyph growth must publish the exact bounded old/new allocation report",
    )?;
    let pointer = run.glyphs().as_ptr();
    let bytes = run.gpu_allocation_bytes();
    let empty = renderer.update_glyph_run(atlas, &mut run, &[])?;
    ensure(
        empty.glyph_count() == 0
            && empty.capacity() >= 4
            && empty.instances().uploaded_instance_bytes() == 0
            && run.bounds().is_empty(),
        "empty glyph update must retain reusable capacity and clear bounds",
    )?;
    let refill = renderer.update_glyph_run(atlas, &mut run, &glyphs)?;
    ensure(
        !refill.instances().replaced_instance_buffer() && run.glyphs().as_ptr() == pointer,
        "glyph refill must reuse storage retained by an empty update",
    )?;
    let bounds = run.bounds();
    ensure(
        matches!(
            renderer.update_glyph_run(atlas, &mut run, &[first; 5]),
            Err(GlyphError::MetadataBudgetExceeded { .. })
        ),
        "fifth glyph must exceed the byte budget, independently of count limit",
    )?;
    let missing = glyph(99, 0.0, Color::WHITE)?;
    ensure(
        renderer.update_glyph_run(atlas, &mut run, &[first, changed, missing])
            == Err(GlyphError::MissingGlyph {
                glyph: GlyphId::new(99),
                index: 2,
            }),
        "missing glyph error must identify its glyph and input index",
    )?;
    let foreign = create_atlas(renderer)?;
    ensure(
        renderer.update_glyph_run(&foreign, &mut run, &glyphs)
            == Err(GlyphError::Image(ImageError::RendererMismatch)),
        "byte-identical foreign atlas must not own this glyph run",
    )?;
    ensure(
        run.glyphs() == glyphs
            && run.glyphs().as_ptr() == pointer
            && run.bounds() == bounds
            && run.gpu_allocation_bytes() == bytes,
        "rejected glyph updates changed retained state",
    )?;
    Ok(run)
}

fn exercise_sprite_updates(renderer: &WgpuRenderer, image: &Image2d) -> Result<ImageBatch2d> {
    let first = ImageSprite2d::new(image.full_rect(), region(-2.0, -2.0)?, Color::WHITE)?;
    let probe = renderer.create_image_batch(image, vec![first], ImageBatchBudget::default())?;
    let one_sprite_bytes = probe.recovery_memory_bytes();
    drop(probe);
    let budget = ImageBatchBudget::new(5, one_sprite_bytes * 4)?;
    let mut batch = renderer.create_image_batch(image, vec![first], budget)?;
    let pointer = batch.sprites().as_ptr();
    ensure(
        renderer
            .update_image_batch(image, &mut batch, &[first])?
            .uploaded_instance_bytes()
            == 0,
        "unchanged sprite must not upload instances",
    )?;
    let changed = ImageSprite2d::new(
        ImageTexelRect::new(1, 0, 1, 1)?,
        region(2.0, 2.0)?,
        Color::WHITE.with_alpha(0.5),
    )?;
    let stable = renderer.update_image_batch(image, &mut batch, &[changed])?;
    ensure(
        stable.uploaded_instance_bytes() > 0
            && !stable.replaced_instance_buffer()
            && batch.sprites().as_ptr() == pointer,
        "same-capacity sprite update must reuse CPU and GPU storage",
    )?;
    let sprites = [first, changed, first, changed];
    let growth = renderer.update_image_batch(image, &mut batch, &sprites)?;
    ensure(
        growth.replaced_instance_buffer()
            && growth.sprite_count() == 4
            && growth.retained_bytes() == budget.max_retained_bytes(),
        "sprite growth must fit the exact retained-byte budget",
    )?;
    let pointer = batch.sprites().as_ptr();
    let bytes = batch.gpu_allocation_bytes();
    let empty = renderer.update_image_batch(image, &mut batch, &[])?;
    ensure(
        empty.sprite_count() == 0
            && empty.retained_capacity() >= 4
            && empty.uploaded_instance_bytes() == 0,
        "empty sprite update must retain capacity without an upload",
    )?;
    let refill = renderer.update_image_batch(image, &mut batch, &sprites)?;
    ensure(
        !refill.replaced_instance_buffer() && batch.sprites().as_ptr() == pointer,
        "sprite refill must reuse retained storage",
    )?;
    let invalid = ImageSprite2d::new(
        ImageTexelRect::new(2, 0, 1, 1)?,
        region(0.0, 0.0)?,
        Color::WHITE,
    )?;
    ensure(
        renderer.update_image_batch(image, &mut batch, &[invalid])
            == Err(ImageError::InvalidSprite),
        "source outside the actual image must be rejected",
    )?;
    ensure(
        matches!(
            renderer.update_image_batch(image, &mut batch, &[first; 5]),
            Err(ImageError::BatchBudgetExceeded { .. })
        ),
        "fifth sprite must exceed the exact retained-byte budget",
    )?;
    let foreign = renderer.create_image_rgba8(2, 1, vec![255; 8], ImageBudget::default())?;
    ensure(
        renderer.update_image_batch(&foreign, &mut batch, &sprites)
            == Err(ImageError::RendererMismatch),
        "byte-identical foreign image must not own this sprite batch",
    )?;
    ensure(
        batch.sprites() == sprites
            && batch.sprites().as_ptr() == pointer
            && batch.gpu_allocation_bytes() == bytes,
        "rejected sprite updates changed retained state",
    )?;
    Ok(batch)
}

fn ensure(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(message.to_owned().into());
    }
    Ok(())
}

fn region(x: f32, y: f32) -> Result<LogicalViewportRegion> {
    Ok(LogicalViewportRegion::new(
        LogicalScreenPosition::new(x, y),
        LogicalViewport::new(12.0, 12.0)?,
    )?)
}

fn glyph(id: u32, x: f32, color: Color) -> Result<PositionedGlyph2d> {
    Ok(PositionedGlyph2d::new(
        GlyphId::new(id),
        region(x, -2.0)?,
        color,
    )?)
}

fn placement(x: f32, y: f32, tint: Color) -> Result<ImageBatchPlacement> {
    Ok(ImageBatchPlacement::new(
        LogicalScreenVector::new(x, y),
        tint,
    )?)
}

fn clip(x: f32, y: f32, width: f32, height: f32) -> Result<ScreenClipRect> {
    Ok(ScreenClipRect::from_min_size(
        LogicalScreenPosition::new(x, y),
        LogicalScreenVector::new(width, height),
    )?)
}
