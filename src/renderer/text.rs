use super::*;
use crate::{FontError, FontFace, RasterizedGlyph, ShapedLine, TextLayoutBudget, TextStyle};

mod prepared;
mod types;
pub use prepared::PreparedTextError;
pub use types::{TextAtlasBudget, TextError, TextUpdateReport};

#[derive(Clone, Copy)]
struct CachedGlyph {
    id: u16,
    width: u32,
    height: u32,
    bearing_x: i32,
    bearing_y: i32,
}

#[derive(Clone, Copy, Default)]
struct Shelf {
    x: u32,
    y: u32,
    height: u32,
}

impl Shelf {
    fn insert(
        &mut self,
        width: u32,
        height: u32,
        size: (u32, u32),
    ) -> Result<ImageTexelRect, TextError> {
        if width > size.0 || height > size.1 {
            return Err(TextError::AtlasFull);
        }
        let mut next = *self;
        if next.x.checked_add(width).is_none_or(|right| right > size.0) {
            next.x = 0;
            next.y = next
                .y
                .checked_add(next.height)
                .ok_or(TextError::AtlasFull)?;
            next.height = 0;
        }
        if next
            .y
            .checked_add(height)
            .is_none_or(|bottom| bottom > size.1)
        {
            return Err(TextError::AtlasFull);
        }
        let rectangle = ImageTexelRect::new(next.x, next.y, width, height)?;
        next.x += width;
        next.height = next.height.max(height);
        *self = next;
        Ok(rectangle)
    }
}

/// Antialiased font cache for one font, logical EM size, direction and DPI.
///
/// Enable the optional `text` feature. Glyphs never move or disappear: the fixed
/// atlas returns [`TextError::AtlasFull`] instead of evicting live runs. Create a
/// new atlas and prepare replacement runs before swapping when size or DPI changes.
/// The input font must be a trusted application asset, not an untrusted-font sandbox.
///
/// Draw [`Self::atlas`] and [`TextRun2d::glyph_run`] with [`ImageSampling::Linear`].
/// Placements are baseline-relative, in logical pixels. Bitmap coverage is cached
/// at an integer physical baseline; fractional placement uses bilinear filtering,
/// not per-subpixel hinting. No fonts are discovered or downloaded automatically.
///
/// ```no_run
/// use sim_engine::{
///     Color, FontFace, FrameBudget, FramePassOptions, ImageBatchPlacement, ImageSampling,
///     LogicalPixels, LogicalScreenVector, PhysicalPerLogical, TextAtlas2d,
///     TextAtlasBudget, TextLayoutBudget, TextStyle, WgpuRenderer,
/// };
///
/// fn draw_label(renderer: &mut WgpuRenderer, font: FontFace)
///     -> Result<(), Box<dyn std::error::Error>>
/// {
///     let style = TextStyle::new(
///         LogicalPixels::new(24.0)?,
///         PhysicalPerLogical::new(renderer.scale_factor() as f32)?,
///     )?;
///     // Keep these two resources between frames in an application.
///     let mut atlas = TextAtlas2d::new(renderer, font, style, TextAtlasBudget::default())?;
///     let label = atlas.prepare(renderer, "Hello, world!", TextLayoutBudget::default())?;
///     let mut frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
///     frame.draw_glyph_run_placed(
///         atlas.atlas(), label.glyph_run(),
///         ImageBatchPlacement::new(LogicalScreenVector::new(24.0, 80.0), Color::WHITE)?,
///         ImageSampling::Linear, FramePassOptions::new(0),
///     )?;
///     let _report = frame.present()?;
///     Ok(())
/// }
/// ```
pub struct TextAtlas2d {
    font: FontFace,
    style: TextStyle,
    budget: TextAtlasBudget,
    atlas: GlyphAtlas2d,
    identity: Arc<()>,
    cached: Vec<CachedGlyph>,
    shelf: Shelf,
}

/// Retained, shaped single-line text and its baseline metrics.
///
/// Drawing an unchanged run needs no shaping, rasterization or geometry upload.
/// Whitespace contributes to [`Self::advance`] but emits no glyph quad. Drawable
/// bounds include filtering support and are not typographic advance bounds.
pub struct TextRun2d {
    run: GlyphRun2d,
    atlas_identity: Arc<()>,
    text: String,
    advance: f32,
    ascent: f32,
    descent: f32,
    line_height: f32,
}

impl TextRun2d {
    /// Returns the retained glyph instances for a composed glyph draw.
    pub const fn glyph_run(&self) -> &GlyphRun2d {
        &self.run
    }
    /// Returns the original UTF-8 text.
    pub fn text(&self) -> &str {
        &self.text
    }
    /// Returns horizontal advance, including spaces, in logical pixels.
    pub const fn advance(&self) -> f32 {
        self.advance
    }
    /// Returns the font ascender above the baseline in logical pixels.
    pub const fn ascent(&self) -> f32 {
        self.ascent
    }
    /// Returns the signed font descender below the baseline in logical pixels.
    pub const fn descent(&self) -> f32 {
        self.descent
    }
    /// Returns the font-recommended baseline spacing in logical pixels.
    pub const fn line_height(&self) -> f32 {
        self.line_height
    }
    /// Returns retained UTF-8 and glyph-layout allocation bytes, excluding allocator overhead.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.text
            .capacity()
            .saturating_add(self.run.recovery_memory_bytes())
    }
}

impl TextAtlas2d {
    /// Creates a fixed-capacity atlas, checking device dimensions before pixel allocation.
    pub fn new(
        renderer: &WgpuRenderer,
        font: FontFace,
        style: TextStyle,
        budget: TextAtlasBudget,
    ) -> Result<Self, TextError> {
        let image_budget = budget.image_budget()?;
        let bytes = image::validate_image_shape(
            budget.width(),
            budget.height(),
            image_budget.max_bytes(),
            image_budget,
            renderer.device.limits().max_texture_dimension_2d,
        )?;
        let mut pixels = reserve_vec(bytes)?;
        pixels.resize(bytes, 255);
        for pixel in pixels.chunks_exact_mut(4) {
            pixel[3] = 0;
        }
        // Reserve the bounded metadata once: inserting a successfully uploaded
        // glyph must not fail later and leave the packing cursor inconsistent.
        let cached = reserve_vec(budget.max_cached_glyphs())?;
        let metadata_bytes = budget
            .max_cached_glyphs()
            .checked_mul(std::mem::size_of::<GlyphAtlasEntry>())
            .ok_or(TextError::InvalidBudget)?;
        let atlas = renderer.create_glyph_atlas(
            budget.width(),
            budget.height(),
            pixels,
            Vec::new(),
            GlyphAtlasBudget::new(image_budget, budget.max_cached_glyphs(), metadata_bytes)?,
        )?;
        Ok(Self {
            font,
            style,
            budget,
            atlas,
            identity: Arc::new(()),
            cached,
            shelf: Shelf::default(),
        })
    }

    /// Returns the atlas to use with a retained glyph-run draw. Use linear sampling.
    pub const fn atlas(&self) -> &GlyphAtlas2d {
        &self.atlas
    }
    /// Returns the immutable font/style/DPI configuration.
    pub const fn style(&self) -> &TextStyle {
        &self.style
    }
    /// Returns the original font, shared cheaply through [`FontFace::clone`].
    pub const fn font(&self) -> &FontFace {
        &self.font
    }
    /// Returns immutable atlas and run limits.
    pub const fn budget(&self) -> TextAtlasBudget {
        self.budget
    }
    /// Returns cached glyph count, including non-drawing spacing glyphs.
    pub fn cached_glyph_count(&self) -> usize {
        self.cached.len()
    }
    /// Returns retained atlas pixels and metadata bytes, excluding shared font data and allocator overhead.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.atlas.recovery_memory_bytes().saturating_add(
            self.cached
                .capacity()
                .saturating_mul(std::mem::size_of::<CachedGlyph>()),
        )
    }
    /// Returns retained GPU image allocation bytes; runs own separate instance buffers.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.atlas.gpu_allocation_bytes()
    }

    /// Shapes a line and rasterizes only glyphs missing from this immutable-style cache.
    ///
    /// Font/input/output limits apply to shaping; raster limits apply to cache misses.
    /// Run capacity is conservatively preflighted for all shaped glyphs, including
    /// whitespace, before allocating placements or populating the glyph cache.
    /// A failure preserves all existing drawable runs, but may warm the bounded
    /// cache with some new glyphs before a later glyph or run allocation fails.
    /// There is no implicit atlas growth, eviction, font fallback or wrapping.
    pub fn prepare(
        &mut self,
        renderer: &WgpuRenderer,
        text: &str,
        layout_budget: TextLayoutBudget,
    ) -> Result<TextRun2d, TextError> {
        renderer.validate_image(&self.atlas.image)?;
        let shaped = self.font.shape_line(text, &self.style, &layout_budget)?;
        self.prepare_validated(renderer, &shaped, layout_budget)
    }

    fn prepare_validated(
        &mut self,
        renderer: &WgpuRenderer,
        shaped: &ShapedLine,
        layout_budget: TextLayoutBudget,
    ) -> Result<TextRun2d, TextError> {
        let owned = owned_text(shaped.text())?;
        let glyphs = self.position_glyphs(renderer, shaped, &layout_budget)?;
        let run = renderer.create_glyph_run(&self.atlas, glyphs, self.budget.run_budget())?;
        Ok(TextRun2d {
            run,
            atlas_identity: Arc::clone(&self.identity),
            text: owned,
            advance: shaped.advance(),
            ascent: shaped.ascent(),
            descent: shaped.descent(),
            line_height: shaped.line_height(),
        })
    }

    /// Replaces a line atomically while reusing its instance-buffer capacity.
    ///
    /// Exact unchanged UTF-8 returns immediately after resource-identity validation;
    /// no shaping/raster work occurs, so `layout_budget` is not consumed or rechecked.
    /// For changed text, a synchronous error preserves the old run/text/metrics,
    /// but may warm atlas entries as documented by [`Self::prepare`].
    pub fn update(
        &mut self,
        renderer: &WgpuRenderer,
        run: &mut TextRun2d,
        text: &str,
        layout_budget: TextLayoutBudget,
    ) -> Result<TextUpdateReport, TextError> {
        self.validate_run(run)?;
        renderer.validate_glyph_run(&self.atlas, &run.run)?;
        if run.text == text {
            return Ok(TextUpdateReport::default());
        }
        let shaped = self.font.shape_line(text, &self.style, &layout_budget)?;
        self.update_validated(renderer, run, &shaped, layout_budget)
    }

    fn update_validated(
        &mut self,
        renderer: &WgpuRenderer,
        run: &mut TextRun2d,
        shaped: &ShapedLine,
        layout_budget: TextLayoutBudget,
    ) -> Result<TextUpdateReport, TextError> {
        let owned = owned_text(shaped.text())?;
        let old_cache_count = self.cached.len();
        let glyphs = self.position_glyphs(renderer, shaped, &layout_budget)?;
        let upload = renderer.update_glyph_run(&self.atlas, &mut run.run, &glyphs)?;
        run.text = owned;
        run.advance = shaped.advance();
        run.ascent = shaped.ascent();
        run.descent = shaped.descent();
        run.line_height = shaped.line_height();
        Ok(TextUpdateReport::new(
            upload.instances().uploaded_instance_bytes(),
            self.cached.len() - old_cache_count,
        ))
    }

    /// Restores exact atlas pixels on a replacement renderer without rerasterizing.
    ///
    /// This replaces the atlas only after successful allocation. Restore each live
    /// run using [`Self::restore_run`] before drawing on the replacement renderer.
    pub fn restore(&mut self, renderer: &WgpuRenderer) -> Result<(), TextError> {
        self.atlas = renderer.restore_glyph_atlas(&self.atlas)?;
        Ok(())
    }

    /// Restores one live run against this atlas, preserving text and baseline metrics.
    pub fn restore_run(
        &self,
        renderer: &WgpuRenderer,
        run: &mut TextRun2d,
    ) -> Result<(), TextError> {
        self.validate_run(run)?;
        run.run = renderer.restore_glyph_run(&self.atlas, &run.run)?;
        Ok(())
    }

    fn validate_run(&self, run: &TextRun2d) -> Result<(), TextError> {
        if !Arc::ptr_eq(&self.identity, &run.atlas_identity) {
            return Err(TextError::AtlasMismatch);
        }
        Ok(())
    }

    fn position_glyphs(
        &mut self,
        renderer: &WgpuRenderer,
        shaped: &ShapedLine,
        layout_budget: &TextLayoutBudget,
    ) -> Result<Vec<PositionedGlyph2d>, TextError> {
        let run_budget = self.budget.run_budget();
        glyph::validate_glyph_run_budget(shaped.glyphs().len(), run_budget)?;
        image::preflight_image_batch_capacity(
            &renderer.device,
            shaped.glyphs().len(),
            ImageBatchBudget::new(run_budget.max_glyphs(), run_budget.max_retained_bytes())?,
        )?;
        let mut glyphs = reserve_vec(shaped.glyphs().len())?;
        for glyph in shaped.glyphs() {
            let cached = self.cache_glyph(renderer, glyph.glyph_id(), layout_budget)?;
            if cached.width == 0 || cached.height == 0 {
                continue;
            }
            let destination = destination_from_metrics(
                cached,
                glyph.logical_x(),
                glyph.logical_y(),
                self.style.scale().get(),
            )?;
            glyphs.push(PositionedGlyph2d::new(
                GlyphId::new(u32::from(cached.id)),
                destination,
                Color::WHITE,
            )?);
        }
        Ok(glyphs)
    }

    fn cache_glyph(
        &mut self,
        renderer: &WgpuRenderer,
        id: u16,
        layout_budget: &TextLayoutBudget,
    ) -> Result<CachedGlyph, TextError> {
        let insertion = match self.cached.binary_search_by_key(&id, |glyph| glyph.id) {
            Ok(index) => return Ok(self.cached[index]),
            Err(index) => index,
        };
        if self.cached.len() == self.budget.max_cached_glyphs() {
            return Err(TextError::GlyphCacheFull);
        }
        let raster = self.font.rasterize_glyph(id, &self.style, layout_budget)?;
        let cached = CachedGlyph {
            id,
            width: raster.width(),
            height: raster.height(),
            bearing_x: raster.bearing_x(),
            bearing_y: raster.bearing_y(),
        };
        if raster.width() > 0 && raster.height() > 0 {
            let width = raster.width().checked_add(4).ok_or(TextError::AtlasFull)?;
            let height = raster.height().checked_add(4).ok_or(TextError::AtlasFull)?;
            let mut shelf = self.shelf;
            let source = shelf.insert(width, height, self.atlas.size())?;
            let pixels = padded_rgba(&raster)?;
            renderer.upload_glyph(
                &mut self.atlas,
                GlyphAtlasEntry::new(GlyphId::new(u32::from(id)), source),
                &pixels,
            )?;
            self.shelf = shelf;
        }
        self.cached.insert(insertion, cached);
        Ok(cached)
    }
}

fn destination_from_metrics(
    glyph: CachedGlyph,
    x: f32,
    y: f32,
    scale: f32,
) -> Result<LogicalViewportRegion, TextError> {
    // Image quads map texel centers to their boundaries. Two transparent border
    // texels preserve bitmap scale and keep the bilinear halo inside full MSAA
    // coverage, even when the baseline falls between physical pixels.
    LogicalViewportRegion::new(
        LogicalScreenPosition::new(
            x + (glyph.bearing_x as f32 - 1.5) / scale,
            y + (glyph.bearing_y as f32 - 1.5) / scale,
        ),
        LogicalViewport::new(
            (glyph.width as f32 + 3.0) / scale,
            (glyph.height as f32 + 3.0) / scale,
        )
        .map_err(|_| TextError::InvalidPlacement)?,
    )
    .map_err(|_| TextError::InvalidPlacement)
}

#[cfg(test)]
pub(super) fn glyph_destination(
    raster: &RasterizedGlyph,
    x: f32,
    y: f32,
    scale: f32,
) -> Result<LogicalViewportRegion, TextError> {
    destination_from_metrics(
        CachedGlyph {
            id: 0,
            width: raster.width(),
            height: raster.height(),
            bearing_x: raster.bearing_x(),
            bearing_y: raster.bearing_y(),
        },
        x,
        y,
        scale,
    )
}

pub(super) fn padded_rgba(raster: &RasterizedGlyph) -> Result<Vec<u8>, TextError> {
    let width = raster.width() as usize + 4;
    let height = raster.height() as usize + 4;
    let bytes = width
        .checked_mul(height)
        .and_then(|area| area.checked_mul(4))
        .ok_or(TextError::AtlasFull)?;
    let mut pixels = reserve_vec(bytes)?;
    pixels.resize(bytes, 255);
    // White transparent padding avoids dark fringes under straight-alpha sampling.
    for pixel in pixels.chunks_exact_mut(4) {
        pixel[3] = 0;
    }
    for y in 0..raster.height() as usize {
        for x in 0..raster.width() as usize {
            pixels[((y + 2) * width + x + 2) * 4 + 3] =
                raster.coverage()[y * raster.width() as usize + x];
        }
    }
    Ok(pixels)
}

fn reserve_vec<T>(count: usize) -> Result<Vec<T>, TextError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| TextError::AllocationFailed)?;
    Ok(values)
}

fn owned_text(text: &str) -> Result<String, TextError> {
    let mut owned = String::new();
    owned
        .try_reserve_exact(text.len())
        .map_err(|_| TextError::AllocationFailed)?;
    owned.push_str(text);
    Ok(owned)
}

#[cfg(test)]
mod tests;
