//! Ordered snapshots of already prepared glyph runs sharing one atlas.

use super::{
    Color, GlyphAtlas2d, GlyphError, GlyphRun2d, ImageBatch2d, ImageBatchBudget,
    ImageBatchPlacement, ImageBatchUploadReport, ImageSprite2d, LogicalScreenPosition,
    LogicalViewportRegion, ScreenClipRect, WgpuRenderer, image,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(super) use tests::verify_glyph_batches;

/// One prepared run's placement in an ordered atlas batch.
///
/// Translation uses local logical pixels and tint multiplies the per-glyph tint.
/// The optional clip is relative to the drawing viewport, is not translated with
/// the run, and intersects the frame item's clip. No shaping or rasterization is
/// performed. A batch snapshots this input; later changes to the source run do
/// not alter the batch until [`WgpuRenderer::update_glyph_batch`] is called.
#[derive(Clone, Copy)]
pub struct GlyphRunPlacement2d<'run> {
    run: &'run GlyphRun2d,
    placement: ImageBatchPlacement,
    clip: Option<ScreenClipRect>,
}

impl<'run> GlyphRunPlacement2d<'run> {
    /// Places an already prepared run without changing its source layout.
    pub const fn new(run: &'run GlyphRun2d, placement: ImageBatchPlacement) -> Self {
        Self {
            run,
            placement,
            clip: None,
        }
    }

    /// Sets a validated viewport-local clip, independent of the translation.
    pub const fn with_clip(mut self, clip: ScreenClipRect) -> Self {
        self.clip = Some(clip);
        self
    }

    /// Returns the prepared source run.
    pub const fn run(self) -> &'run GlyphRun2d {
        self.run
    }

    /// Returns the run's local logical translation and normalized tint.
    pub const fn placement(self) -> ImageBatchPlacement {
        self.placement
    }

    /// Returns the optional viewport-local logical clip.
    pub const fn clip(self) -> Option<ScreenClipRect> {
        self.clip
    }
}

/// Explicit limits for a retained many-placement glyph batch.
///
/// Retained bytes cover sprite descriptions, reusable instance conversion
/// storage, and clip spans. Atlas storage, input runs and backend allocations are
/// separate. GPU capacity remains bounded by the glyph count and device limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlyphBatchBudget {
    max_placements: usize,
    max_glyphs: usize,
    max_retained_bytes: usize,
}

impl GlyphBatchBudget {
    /// CPU bytes required by one glyph-capacity slot, excluding clip spans.
    pub const RETAINED_BYTES_PER_GLYPH: usize = ImageBatchBudget::RETAINED_BYTES_PER_SPRITE;
    /// Maximum additional CPU bytes for one non-empty placement's clip span.
    pub const RETAINED_BYTES_PER_PLACEMENT: usize = std::mem::size_of::<GlyphBatchSpan>();

    /// Creates non-zero limits sufficient for one glyph and one clip span.
    pub fn new(
        max_placements: usize,
        max_glyphs: usize,
        max_retained_bytes: usize,
    ) -> Result<Self, GlyphError> {
        if max_placements == 0
            || max_glyphs == 0
            || max_retained_bytes
                < Self::RETAINED_BYTES_PER_GLYPH + Self::RETAINED_BYTES_PER_PLACEMENT
        {
            return Err(GlyphError::InvalidBudget);
        }
        Ok(Self {
            max_placements,
            max_glyphs,
            max_retained_bytes,
        })
    }

    /// Returns the maximum submitted placements, including empty runs.
    pub const fn max_placements(self) -> usize {
        self.max_placements
    }
    /// Returns the maximum sum of glyphs across all placements.
    pub const fn max_glyphs(self) -> usize {
        self.max_glyphs
    }
    /// Returns the retained CPU allocation ceiling, including reusable capacity.
    pub const fn max_retained_bytes(self) -> usize {
        self.max_retained_bytes
    }
}

impl Default for GlyphBatchBudget {
    fn default() -> Self {
        Self {
            max_placements: 16_384,
            max_glyphs: 100_000,
            max_retained_bytes: 32 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::renderer) struct GlyphBatchSpan {
    pub(in crate::renderer) first: u32,
    pub(in crate::renderer) end: u32,
    pub(in crate::renderer) clip: Option<ScreenClipRect>,
}

/// A retained, ordered snapshot of prepared glyph placements sharing one atlas.
///
/// One instance buffer holds all glyphs. Only adjacent placements with identical
/// clips are merged; glyphs are never reordered. A shared clip therefore needs
/// at most one draw, while distinct adjacent clips require separate scissor
/// draws. The entire batch is one frame item and cannot interleave other frame
/// sources inside it. Use separate batches at painter-order boundaries.
///
/// Updates reuse CPU/GPU capacity and equal sprite input causes no upload. A
/// frame reuses exact sprite geometry proofs at an unchanged viewport. Device
/// restoration clears proofs and requires the restored original atlas.
pub struct GlyphBatch2d {
    pub(in crate::renderer) batch: ImageBatch2d,
    pub(in crate::renderer) spans: Vec<GlyphBatchSpan>,
    placement_count: usize,
    budget: GlyphBatchBudget,
}

impl GlyphBatch2d {
    /// Returns the number of submitted placements, including empty source runs.
    pub const fn placement_count(&self) -> usize {
        self.placement_count
    }
    /// Returns the total retained glyph count across all placements.
    pub fn glyph_count(&self) -> usize {
        self.batch.sprite_count()
    }
    /// Returns the conservative draw count before viewport clipping.
    pub fn draw_count(&self) -> usize {
        self.spans.len()
    }
    /// Returns reusable glyph storage capacity, excluding clip metadata.
    pub fn capacity(&self) -> usize {
        self.batch.capacity()
    }
    /// Returns the limits retained for updates and device restoration.
    pub const fn budget(&self) -> GlyphBatchBudget {
        self.budget
    }
    /// Returns CPU recovery/conversion/clip capacity bytes, excluding source runs.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.batch
            .recovery_memory_bytes()
            .saturating_add(span_bytes(self.spans.capacity()))
    }
    /// Returns actual GPU instance-buffer allocation bytes, including spare capacity.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.batch.gpu_allocation_bytes()
    }
}

/// Allocation and upload outcome of a transactional glyph-batch update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlyphBatchUploadReport {
    instances: ImageBatchUploadReport,
    placements: usize,
    draws: usize,
    retained_bytes: usize,
    peak_retained_bytes: usize,
}

impl GlyphBatchUploadReport {
    /// Returns exact instance-upload bytes and GPU allocation/growth counters.
    pub const fn instances(self) -> ImageBatchUploadReport {
        self.instances
    }
    /// Returns accepted source placement count, including empty runs.
    pub const fn placement_count(self) -> usize {
        self.placements
    }
    /// Returns the conservative ordered scissor draw count.
    pub const fn draw_count(self) -> usize {
        self.draws
    }
    /// Returns final retained CPU capacity bytes.
    pub const fn retained_bytes(self) -> usize {
        self.retained_bytes
    }
    /// Returns conservative old/new overlapping CPU storage during the update.
    pub const fn peak_retained_bytes(self) -> usize {
        self.peak_retained_bytes
    }
}

impl WgpuRenderer {
    /// Snapshots same-atlas prepared placements without shaping or font work.
    ///
    /// Foreign atlases/runs, overflowing destinations and count/byte/device-limit
    /// excess are rejected before GPU allocation. Input order is painter order.
    pub fn create_glyph_batch(
        &self,
        atlas: &GlyphAtlas2d,
        placements: &[GlyphRunPlacement2d<'_>],
        budget: GlyphBatchBudget,
    ) -> Result<GlyphBatch2d, GlyphError> {
        self.validate_image(&atlas.image)?;
        create_resources(
            &self.device,
            &self.queue,
            std::sync::Arc::clone(&self.renderer_identity),
            &atlas.image,
            placements,
            budget,
        )
    }

    /// Atomically replaces the snapshot, reusing CPU arrays and GPU capacity.
    ///
    /// All runs must belong to the same current atlas. Synchronous validation or
    /// allocation failure preserves the old drawable batch. Source run mutations
    /// are not observed until this call. Clip-only changes do not upload glyphs;
    /// translation/tint changes update instance data without redoing text layout.
    /// Shrinking retains reusable capacity. GPU/device failure is not a rollback
    /// of queue work already submitted.
    pub fn update_glyph_batch(
        &self,
        atlas: &GlyphAtlas2d,
        batch: &mut GlyphBatch2d,
        placements: &[GlyphRunPlacement2d<'_>],
    ) -> Result<GlyphBatchUploadReport, GlyphError> {
        self.validate_image_batch(&atlas.image, &batch.batch)?;
        update_resources(&self.device, &self.queue, &atlas.image, batch, placements)
    }

    /// Restores a snapshot against the restored copy of its original atlas.
    ///
    /// The original runs need not remain alive. Translation, tint, clips and
    /// painter order are preserved. Another atlas, even with identical pixels,
    /// is rejected. The restored batch has no viewport-dependent cached proof.
    pub fn restore_glyph_batch(
        &self,
        atlas: &GlyphAtlas2d,
        source: &GlyphBatch2d,
    ) -> Result<GlyphBatch2d, GlyphError> {
        self.validate_image(&atlas.image)?;
        restore_resources(
            &self.device,
            &self.queue,
            std::sync::Arc::clone(&self.renderer_identity),
            &atlas.image,
            source,
        )
    }
}

fn restore_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: std::sync::Arc<()>,
    image: &super::Image2d,
    source: &GlyphBatch2d,
) -> Result<GlyphBatch2d, GlyphError> {
    if !std::sync::Arc::ptr_eq(
        &image.recovery_identity,
        &source.batch.image_recovery_identity,
    ) {
        return Err(super::ImageError::RecoverySourceMismatch.into());
    }
    image::preflight_image_batch_capacity(device, source.glyph_count(), source.batch.budget())?;
    let mut spans = reserve_spans(source.spans.len())?;
    spans.extend_from_slice(&source.spans);
    let restored_budget = image_budget(source.budget, spans.capacity())?;
    let batch = image::restore_image_batch_resources(
        device,
        queue,
        renderer_identity,
        image,
        &source.batch,
        restored_budget,
    )?;
    Ok(GlyphBatch2d {
        batch,
        spans,
        placement_count: source.placement_count,
        budget: source.budget,
    })
}

fn span_bytes(count: usize) -> usize {
    count.saturating_mul(std::mem::size_of::<GlyphBatchSpan>())
}

fn reserve_spans(count: usize) -> Result<Vec<GlyphBatchSpan>, GlyphError> {
    let mut spans = Vec::new();
    spans
        .try_reserve_exact(count)
        .map_err(|_| GlyphError::AllocationFailed {
            requested_bytes: span_bytes(count),
        })?;
    Ok(spans)
}

fn placed_sprite(
    sprite: ImageSprite2d,
    placement: ImageBatchPlacement,
) -> Result<ImageSprite2d, GlyphError> {
    let sprite = translated_sprite(sprite, placement)?;
    if !image::logical_image_region_is_portable(sprite.destination()) {
        return Err(GlyphError::InvalidGlyph);
    }
    Ok(sprite)
}

// The shared image updater performs the exact portable-coordinate proof once,
// before publication. Its second iterator traversal only needs this inexpensive
// transformation; do not repeat the interval proof during instance emission.
fn translated_sprite(
    sprite: ImageSprite2d,
    placement: ImageBatchPlacement,
) -> Result<ImageSprite2d, GlyphError> {
    let origin = sprite.destination().origin().to_vec2() + placement.translation().to_vec2();
    let destination = LogicalViewportRegion::new(
        LogicalScreenPosition::from_vec2(origin),
        sprite.destination().viewport(),
    )
    .map_err(|_| GlyphError::InvalidGlyph)?;
    let first = sprite.tint().to_array();
    let second = placement.tint().to_array();
    Ok(ImageSprite2d::new(
        sprite.source(),
        destination,
        Color::rgba(
            first[0] * second[0],
            first[1] * second[1],
            first[2] * second[2],
            first[3] * second[3],
        ),
    )?)
}

fn validate_placements(
    image: &super::Image2d,
    placements: &[GlyphRunPlacement2d<'_>],
    budget: GlyphBatchBudget,
) -> Result<(usize, usize), GlyphError> {
    if placements.len() > budget.max_placements {
        return Err(GlyphError::EntryBudgetExceeded {
            limit: budget.max_placements,
            actual: placements.len(),
        });
    }
    let mut glyphs = 0_usize;
    let mut spans = 0;
    let mut previous_clip = None;
    for entry in placements {
        image::validate_batch_image(image, &entry.run.batch)?;
        glyphs =
            glyphs
                .checked_add(entry.run.glyph_count())
                .ok_or(GlyphError::EntryBudgetExceeded {
                    limit: budget.max_glyphs,
                    actual: usize::MAX,
                })?;
        if glyphs > budget.max_glyphs {
            return Err(GlyphError::EntryBudgetExceeded {
                limit: budget.max_glyphs,
                actual: glyphs,
            });
        }
        if entry.run.glyph_count() != 0 {
            if previous_clip != Some(entry.clip) {
                spans += 1;
            }
            previous_clip = Some(entry.clip);
        }
    }
    Ok((glyphs, spans))
}

fn image_budget(
    budget: GlyphBatchBudget,
    span_capacity: usize,
) -> Result<ImageBatchBudget, GlyphError> {
    let remaining = budget
        .max_retained_bytes
        .checked_sub(span_bytes(span_capacity))
        .ok_or(GlyphError::MetadataBudgetExceeded {
            limit: budget.max_retained_bytes,
            actual: span_bytes(span_capacity),
        })?;
    if remaining < ImageBatchBudget::RETAINED_BYTES_PER_SPRITE {
        return Err(GlyphError::MetadataBudgetExceeded {
            limit: budget.max_retained_bytes,
            actual: span_bytes(span_capacity)
                .saturating_add(ImageBatchBudget::RETAINED_BYTES_PER_SPRITE),
        });
    }
    Ok(ImageBatchBudget::new(budget.max_glyphs, remaining)?)
}

fn fill_spans(spans: &mut Vec<GlyphBatchSpan>, placements: &[GlyphRunPlacement2d<'_>]) {
    spans.clear();
    let mut first = 0_u32;
    for entry in placements {
        let count = entry.run.glyph_count() as u32;
        if count == 0 {
            continue;
        }
        let end = first + count;
        if let Some(last) = spans.last_mut()
            && last.clip == entry.clip
        {
            last.end = end;
        } else {
            spans.push(GlyphBatchSpan {
                first,
                end,
                clip: entry.clip,
            });
        }
        first = end;
    }
}

fn create_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: std::sync::Arc<()>,
    image: &super::Image2d,
    placements: &[GlyphRunPlacement2d<'_>],
    budget: GlyphBatchBudget,
) -> Result<GlyphBatch2d, GlyphError> {
    let (count, span_count) = validate_placements(image, placements, budget)?;
    image::preflight_image_batch_capacity(device, count, image_budget(budget, span_count)?)?;
    let mut spans = reserve_spans(span_count)?;
    let image_budget = image_budget(budget, spans.capacity())?;
    image::preflight_image_batch_capacity(device, count, image_budget)?;
    let mut sprites = Vec::new();
    sprites
        .try_reserve_exact(count)
        .map_err(|_| GlyphError::AllocationFailed {
            requested_bytes: count.saturating_mul(std::mem::size_of::<ImageSprite2d>()),
        })?;
    for entry in placements {
        for sprite in entry.run.batch.sprites() {
            sprites.push(placed_sprite(*sprite, entry.placement)?);
        }
    }
    let batch = image::create_image_batch_resources(
        device,
        queue,
        renderer_identity,
        image,
        sprites,
        image_budget,
    )?;
    fill_spans(&mut spans, placements);
    Ok(GlyphBatch2d {
        batch,
        spans,
        placement_count: placements.len(),
        budget,
    })
}

fn update_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &super::Image2d,
    batch: &mut GlyphBatch2d,
    placements: &[GlyphRunPlacement2d<'_>],
) -> Result<GlyphBatchUploadReport, GlyphError> {
    image::validate_batch_image(image, &batch.batch)?;
    let (count, span_count) = validate_placements(image, placements, batch.budget)?;
    image::preflight_image_batch_capacity(device, count, image_budget(batch.budget, span_count)?)?;
    let old_span_bytes = span_bytes(batch.spans.capacity());
    let replacement = (span_count > batch.spans.capacity())
        .then(|| reserve_spans(span_count))
        .transpose()?;
    let span_capacity = replacement
        .as_ref()
        .map_or(batch.spans.capacity(), Vec::capacity);
    let budget = image_budget(batch.budget, span_capacity)?;
    // The common updater validates each transformed sprite exactly once, and
    // verifies the full count before any publication. A failed transformation
    // removes an entry here, so count validation rejects the entire update.
    let sprites = placements.iter().flat_map(|entry| {
        entry
            .run
            .batch
            .sprites()
            .iter()
            .filter_map(move |sprite| translated_sprite(*sprite, entry.placement).ok())
    });
    let instances = image::update_image_batch_resources(
        device,
        queue,
        image,
        &mut batch.batch,
        sprites,
        count,
        budget,
    )
    .map_err(|error| match error {
        super::ImageError::InvalidSprite => GlyphError::InvalidGlyph,
        error => GlyphError::Image(error),
    })?;
    let peak_retained_bytes = instances
        .peak_retained_bytes()
        .saturating_add(old_span_bytes)
        .saturating_add(if replacement.is_some() {
            span_bytes(span_capacity)
        } else {
            0
        });
    if let Some(spans) = replacement {
        batch.spans = spans;
    }
    fill_spans(&mut batch.spans, placements);
    batch.placement_count = placements.len();
    Ok(GlyphBatchUploadReport {
        instances,
        placements: placements.len(),
        draws: batch.spans.len(),
        retained_bytes: batch.recovery_memory_bytes(),
        peak_retained_bytes,
    })
}
