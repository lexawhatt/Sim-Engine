use super::*;

/// Opaque host-provided glyph identity within one [`GlyphAtlas2d`].
///
/// The value may be a font glyph index, a Unicode scalar value, or a host
/// registry key. Sim;Engine only compares it for exact equality and does not
/// perform shaping or assign text semantics to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlyphId(u32);

impl GlyphId {
    /// Wraps one host-selected glyph identity.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the original host-selected identity.
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// Maps one glyph identity to a non-empty rectangle in an atlas image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlyphAtlasEntry {
    glyph: GlyphId,
    source: ImageTexelRect,
}

impl GlyphAtlasEntry {
    /// Associates a glyph with one physical-texel atlas rectangle.
    pub const fn new(glyph: GlyphId, source: ImageTexelRect) -> Self {
        Self { glyph, source }
    }

    /// Returns the opaque glyph identity.
    pub const fn glyph(self) -> GlyphId {
        self.glyph
    }

    /// Returns the physical atlas source rectangle.
    pub const fn source(self) -> ImageTexelRect {
        self.source
    }
}

/// Image and metadata limits retained with a [`GlyphAtlas2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlyphAtlasBudget {
    image: ImageBudget,
    max_entries: usize,
    max_metadata_bytes: usize,
}

impl GlyphAtlasBudget {
    /// Creates non-zero image, entry-count, and metadata-byte limits.
    pub fn new(
        image: ImageBudget,
        max_entries: usize,
        max_metadata_bytes: usize,
    ) -> Result<Self, GlyphError> {
        if max_entries == 0 || max_metadata_bytes < std::mem::size_of::<GlyphAtlasEntry>() {
            return Err(GlyphError::InvalidBudget);
        }
        Ok(Self {
            image,
            max_entries,
            max_metadata_bytes,
        })
    }

    /// Returns the atlas image limits.
    pub const fn image(self) -> ImageBudget {
        self.image
    }

    /// Returns the maximum registered glyph count.
    pub const fn max_entries(self) -> usize {
        self.max_entries
    }

    /// Returns the maximum retained glyph-metadata bytes.
    pub const fn max_metadata_bytes(self) -> usize {
        self.max_metadata_bytes
    }
}

impl Default for GlyphAtlasBudget {
    fn default() -> Self {
        Self {
            image: ImageBudget::default(),
            max_entries: 65_536,
            max_metadata_bytes: 4 * 1024 * 1024,
        }
    }
}

/// Retained glyph-run count and byte limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlyphRunBudget {
    max_glyphs: usize,
    max_retained_bytes: usize,
}

impl GlyphRunBudget {
    /// Creates non-zero limits for positioned glyphs and recovery metadata.
    pub fn new(max_glyphs: usize, max_retained_bytes: usize) -> Result<Self, GlyphError> {
        if max_glyphs == 0 || max_retained_bytes < glyph_retained_bytes(1).unwrap_or(usize::MAX) {
            return Err(GlyphError::InvalidBudget);
        }
        Ok(Self {
            max_glyphs,
            max_retained_bytes,
        })
    }

    /// Returns the maximum number of submitted positioned glyphs.
    pub const fn max_glyphs(self) -> usize {
        self.max_glyphs
    }

    /// Returns the maximum retained run and sprite-description bytes.
    pub const fn max_retained_bytes(self) -> usize {
        self.max_retained_bytes
    }
}

impl Default for GlyphRunBudget {
    fn default() -> Self {
        Self {
            max_glyphs: 100_000,
            max_retained_bytes: 32 * 1024 * 1024,
        }
    }
}

/// Failure while constructing, updating, or restoring glyph resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlyphError {
    /// Image creation, update, ownership, or capacity validation failed.
    Image(ImageError),
    /// A glyph budget must fit at least one entry or positioned glyph.
    InvalidBudget,
    /// Atlas or run metadata exceeded its configured entry limit.
    EntryBudgetExceeded {
        /// Configured count ceiling.
        limit: usize,
        /// Requested entry count.
        actual: usize,
    },
    /// Atlas or run recovery metadata exceeded its configured byte limit.
    MetadataBudgetExceeded {
        /// Configured byte ceiling.
        limit: usize,
        /// Required retained bytes.
        actual: usize,
    },
    /// Two atlas entries use the same glyph identity.
    DuplicateGlyph {
        /// Repeated identity.
        glyph: GlyphId,
    },
    /// Two distinct glyph identities claim overlapping atlas texels.
    OverlappingGlyph {
        /// Identity whose rectangle was rejected.
        glyph: GlyphId,
        /// Existing identity whose rectangle overlaps it.
        conflicting_glyph: GlyphId,
    },
    /// A positioned run references a glyph absent from its atlas.
    MissingGlyph {
        /// Missing identity.
        glyph: GlyphId,
        /// Zero-based positioned-glyph index in the submitted run.
        index: usize,
    },
    /// A glyph destination, tint, or atlas rectangle is invalid.
    InvalidGlyph,
    /// CPU storage for retained metadata or temporary validation/conversion
    /// scratch could not be reserved.
    AllocationFailed {
        /// Minimum bytes requested by the failed reservation.
        requested_bytes: usize,
    },
}

impl fmt::Display for GlyphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(error) => write!(formatter, "glyph image error: {error}"),
            Self::InvalidBudget => write!(formatter, "glyph budget must fit one entry"),
            Self::EntryBudgetExceeded { limit, actual } => {
                write!(formatter, "glyph count {actual} exceeds limit {limit}")
            }
            Self::MetadataBudgetExceeded { limit, actual } => write!(
                formatter,
                "glyph recovery metadata requires {actual} bytes, over limit {limit}"
            ),
            Self::DuplicateGlyph { glyph } => {
                write!(formatter, "glyph identity {} is duplicated", glyph.value())
            }
            Self::OverlappingGlyph {
                glyph,
                conflicting_glyph,
            } => write!(
                formatter,
                "glyph identity {} overlaps atlas texels owned by glyph {}",
                glyph.value(),
                conflicting_glyph.value()
            ),
            Self::MissingGlyph { glyph, index } => write!(
                formatter,
                "glyph identity {} at run index {index} is absent from the atlas",
                glyph.value()
            ),
            Self::InvalidGlyph => write!(formatter, "glyph placement or atlas region is invalid"),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for glyph metadata or scratch storage"
            ),
        }
    }
}

impl Error for GlyphError {}

impl From<ImageError> for GlyphError {
    fn from(error: ImageError) -> Self {
        Self::Image(error)
    }
}

/// Result of uploading one glyph image into an existing atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlyphUploadReport {
    uploaded_bytes: usize,
    inserted_entry: bool,
}

impl GlyphUploadReport {
    /// Returns the exact number of RGBA bytes submitted to the GPU queue.
    pub const fn uploaded_bytes(self) -> usize {
        self.uploaded_bytes
    }

    /// Returns whether a new glyph identity was registered.
    pub const fn inserted_entry(self) -> bool {
        self.inserted_entry
    }
}

/// Renderer-owned sRGB RGBA glyph atlas with exact pixels and sorted metadata.
///
/// Sim;Engine does not evict entries implicitly. Hosts can therefore treat a
/// successful upload as stable until they explicitly replace or restore the
/// atlas. Straight-alpha pixels follow the same color contract as [`Image2d`].
pub struct GlyphAtlas2d {
    pub(super) image: Image2d,
    entries: Vec<GlyphAtlasEntry>,
    budget: GlyphAtlasBudget,
}

impl GlyphAtlas2d {
    /// Returns atlas dimensions in physical source texels.
    pub const fn size(&self) -> (u32, u32) {
        self.image.size()
    }

    /// Returns registered entries sorted by glyph identity.
    pub fn entries(&self) -> &[GlyphAtlasEntry] {
        &self.entries
    }

    /// Returns one atlas rectangle, or `None` for an unregistered identity.
    pub fn source_for(&self, glyph: GlyphId) -> Option<ImageTexelRect> {
        find_entry(&self.entries, glyph).map(GlyphAtlasEntry::source)
    }

    /// Returns the retained atlas limits.
    pub const fn budget(&self) -> GlyphAtlasBudget {
        self.budget
    }

    /// Returns exact CPU bytes retained for image and glyph recovery.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.image.recovery_memory_bytes().saturating_add(
            self.entries
                .capacity()
                .saturating_mul(std::mem::size_of::<GlyphAtlasEntry>()),
        )
    }

    /// Returns nominal single-level RGBA8 texel-storage bytes, excluding
    /// opaque backend alignment, tiling, allocation pages, and metadata.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.image.gpu_allocation_bytes()
    }
}

/// One host-shaped glyph quad in local logical-screen coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionedGlyph2d {
    glyph: GlyphId,
    destination: LogicalViewportRegion,
    tint: Color,
}

impl PositionedGlyph2d {
    /// Creates a finite logical-pixel quad with normalized straight-linear tint.
    pub fn new(
        glyph: GlyphId,
        destination: LogicalViewportRegion,
        tint: Color,
    ) -> Result<Self, GlyphError> {
        if !tint.is_normalized() || !logical_region_arithmetic_is_finite(destination) {
            return Err(GlyphError::InvalidGlyph);
        }
        Ok(Self {
            glyph,
            destination,
            tint,
        })
    }

    /// Returns the atlas-local glyph identity.
    pub const fn glyph(self) -> GlyphId {
        self.glyph
    }

    /// Returns the exact logical-pixel destination supplied by the host shaper.
    pub const fn destination(self) -> LogicalViewportRegion {
        self.destination
    }

    /// Returns normalized straight-linear tint and opacity.
    pub const fn tint(self) -> Color {
        self.tint
    }
}

/// Deterministic logical-pixel bounds of a submitted positioned glyph run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphRunBounds {
    region: Option<LogicalViewportRegion>,
}

impl GlyphRunBounds {
    /// Returns `None` for an empty run or the exact union of its glyph quads.
    pub const fn region(self) -> Option<LogicalViewportRegion> {
        self.region
    }

    /// Returns whether the submitted run contains no glyph quads.
    pub const fn is_empty(self) -> bool {
        self.region.is_none()
    }
}

/// Bounded retained glyph-run statistics independent of frame timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlyphRunStatistics {
    submitted_glyphs: usize,
    rendered_quads: usize,
    atlas_misses: usize,
    retained_bytes: usize,
}

impl GlyphRunStatistics {
    /// Returns positioned glyphs accepted from the host.
    pub const fn submitted_glyphs(self) -> usize {
        self.submitted_glyphs
    }

    /// Returns retained drawable quads. Successful runs have one per glyph.
    pub const fn rendered_quads(self) -> usize {
        self.rendered_quads
    }

    /// Returns atlas misses. Successful retained runs always report zero.
    pub const fn atlas_misses(self) -> usize {
        self.atlas_misses
    }

    /// Returns exact retained glyph and sprite-description bytes.
    pub const fn retained_bytes(self) -> usize {
        self.retained_bytes
    }
}

/// Renderer-owned positioned glyph run drawn as at most one instanced atlas batch.
///
/// A run references exactly one atlas. Mixed fallback fonts remain explicit:
/// build one run per atlas and submit them at the same frame order. The host is
/// responsible for shaping, baseline selection, fallback, and line breaking.
/// An empty run is valid and emits no draw call.
pub struct GlyphRun2d {
    pub(super) batch: ImageBatch2d,
    glyphs: Vec<PositionedGlyph2d>,
    bounds: GlyphRunBounds,
    budget: GlyphRunBudget,
}

impl GlyphRun2d {
    /// Returns the exact host-positioned glyph descriptions.
    pub fn glyphs(&self) -> &[PositionedGlyph2d] {
        &self.glyphs
    }

    /// Returns the retained glyph count.
    pub fn glyph_count(&self) -> usize {
        self.glyphs.len()
    }

    /// Returns deterministic logical-pixel bounds without font interpretation.
    pub const fn bounds(&self) -> GlyphRunBounds {
        self.bounds
    }

    /// Returns creation and restoration limits.
    pub const fn budget(&self) -> GlyphRunBudget {
        self.budget
    }

    /// Returns retained work counters for this run.
    pub fn statistics(&self) -> GlyphRunStatistics {
        GlyphRunStatistics {
            submitted_glyphs: self.glyphs.len(),
            rendered_quads: self.glyphs.len(),
            atlas_misses: 0,
            retained_bytes: self.recovery_memory_bytes(),
        }
    }

    /// Returns exact CPU glyph and sprite-description recovery bytes.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.glyphs
            .capacity()
            .saturating_mul(std::mem::size_of::<PositionedGlyph2d>())
            .saturating_add(self.batch.recovery_memory_bytes())
    }

    /// Returns GPU instance-buffer bytes actively addressable by the run.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.batch.gpu_allocation_bytes()
    }
}

impl WgpuRenderer {
    /// Creates a bounded glyph atlas from exact RGBA pixels and initial entries.
    pub fn create_glyph_atlas(
        &self,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        entries: Vec<GlyphAtlasEntry>,
        budget: GlyphAtlasBudget,
    ) -> Result<GlyphAtlas2d, GlyphError> {
        create_glyph_atlas_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            width,
            height,
            pixels,
            entries,
            budget,
        )
    }

    /// Uploads or replaces one glyph rectangle after complete metadata preflight.
    ///
    /// Reusing an identity requires the same rectangle. Changing atlas packing
    /// is an explicit new-atlas operation so existing retained runs cannot
    /// silently sample unrelated texels.
    pub fn upload_glyph(
        &self,
        atlas: &mut GlyphAtlas2d,
        entry: GlyphAtlasEntry,
        pixels: &[u8],
    ) -> Result<GlyphUploadReport, GlyphError> {
        self.validate_image(&atlas.image)?;
        if !entry.source.fits(atlas.image.width(), atlas.image.height()) {
            return Err(GlyphError::InvalidGlyph);
        }
        image::validate_image_region_upload(&atlas.image, entry.source, pixels.len())?;
        let insertion = validate_glyph_insertion(&atlas.entries, entry)?;
        if insertion.is_some() {
            validate_metadata_capacity(atlas.entries.len().saturating_add(1), atlas.budget)?;
        }
        let replacement_entries = if let Some(index) = insertion {
            let new_len = atlas.entries.len() + 1;
            let requested_bytes = new_len
                .checked_mul(std::mem::size_of::<GlyphAtlasEntry>())
                .ok_or(GlyphError::InvalidGlyph)?;
            let mut replacement = Vec::new();
            replacement
                .try_reserve_exact(new_len)
                .map_err(|_| GlyphError::AllocationFailed { requested_bytes })?;
            replacement.extend_from_slice(&atlas.entries[..index]);
            replacement.push(entry);
            replacement.extend_from_slice(&atlas.entries[index..]);
            let actual = replacement
                .capacity()
                .saturating_mul(std::mem::size_of::<GlyphAtlasEntry>());
            if actual > atlas.budget.max_metadata_bytes {
                return Err(GlyphError::MetadataBudgetExceeded {
                    limit: atlas.budget.max_metadata_bytes,
                    actual,
                });
            }
            Some(replacement)
        } else {
            None
        };
        let report = self.update_image_region(&mut atlas.image, entry.source, pixels)?;
        if let Some(replacement) = replacement_entries {
            atlas.entries = replacement;
        }
        Ok(GlyphUploadReport {
            uploaded_bytes: report.uploaded_bytes(),
            inserted_entry: insertion.is_some(),
        })
    }

    /// Creates one retained instanced run from host-positioned glyph quads.
    pub fn create_glyph_run(
        &self,
        atlas: &GlyphAtlas2d,
        glyphs: Vec<PositionedGlyph2d>,
        budget: GlyphRunBudget,
    ) -> Result<GlyphRun2d, GlyphError> {
        self.validate_image(&atlas.image)?;
        create_glyph_run_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            atlas,
            glyphs,
            budget,
        )
    }

    /// Restores exact atlas pixels and metadata for this renderer generation.
    pub fn restore_glyph_atlas(&self, source: &GlyphAtlas2d) -> Result<GlyphAtlas2d, GlyphError> {
        image::validate_image_shape(
            source.image.width(),
            source.image.height(),
            source.image.pixels().len(),
            source.budget.image,
            self.device.limits().max_texture_dimension_2d,
        )?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(source.entries.len())
            .map_err(|_| GlyphError::AllocationFailed {
                requested_bytes: std::mem::size_of_val(source.entries.as_slice()),
            })?;
        entries.extend_from_slice(&source.entries);
        validate_atlas_entries(
            &mut entries,
            source.image.width(),
            source.image.height(),
            source.budget,
        )?;
        entries = compact_glyph_vec(entries, source.budget.max_metadata_bytes)?;
        let actual = entries
            .capacity()
            .saturating_mul(std::mem::size_of::<GlyphAtlasEntry>());
        if actual > source.budget.max_metadata_bytes {
            return Err(GlyphError::MetadataBudgetExceeded {
                limit: source.budget.max_metadata_bytes,
                actual,
            });
        }
        let image = self.restore_image(&source.image)?;
        Ok(GlyphAtlas2d {
            image,
            entries,
            budget: source.budget,
        })
    }

    /// Restores a run against the restored copy of its original atlas.
    pub fn restore_glyph_run(
        &self,
        atlas: &GlyphAtlas2d,
        source: &GlyphRun2d,
    ) -> Result<GlyphRun2d, GlyphError> {
        self.validate_image(&atlas.image)?;
        if !Arc::ptr_eq(
            &atlas.image.recovery_identity,
            &source.batch.image_recovery_identity,
        ) {
            return Err(ImageError::RecoverySourceMismatch.into());
        }
        validate_glyph_run_sources(&atlas.entries, &source.glyphs, source.batch.sprites())?;
        validate_glyph_run_budget(source.glyphs.len(), source.budget)?;
        validate_glyph_membership(&atlas.entries, &source.glyphs)?;
        measure_glyphs(&source.glyphs)?;
        let batch_budget =
            ImageBatchBudget::new(source.budget.max_glyphs, source.budget.max_retained_bytes)?;
        image::preflight_image_batch_capacity(&self.device, source.glyphs.len(), batch_budget)?;
        let retained_bytes = source
            .glyphs
            .len()
            .checked_mul(std::mem::size_of::<PositionedGlyph2d>())
            .ok_or(GlyphError::InvalidGlyph)?;
        let mut glyphs = Vec::new();
        glyphs.try_reserve_exact(source.glyphs.len()).map_err(|_| {
            GlyphError::AllocationFailed {
                requested_bytes: retained_bytes,
            }
        })?;
        glyphs.extend_from_slice(&source.glyphs);
        self.create_glyph_run(atlas, glyphs, source.budget)
    }

    pub(super) fn validate_glyph_run(
        &self,
        atlas: &GlyphAtlas2d,
        run: &GlyphRun2d,
    ) -> Result<(), GlyphError> {
        self.validate_image_batch(&atlas.image, &run.batch)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_glyph_atlas_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    mut entries: Vec<GlyphAtlasEntry>,
    budget: GlyphAtlasBudget,
) -> Result<GlyphAtlas2d, GlyphError> {
    image::validate_image_shape(
        width,
        height,
        pixels.len(),
        budget.image,
        device.limits().max_texture_dimension_2d,
    )?;
    validate_atlas_entries(&mut entries, width, height, budget)?;
    entries = compact_glyph_vec(entries, budget.max_metadata_bytes)?;
    let actual_metadata_bytes = entries
        .capacity()
        .saturating_mul(std::mem::size_of::<GlyphAtlasEntry>());
    if actual_metadata_bytes > budget.max_metadata_bytes {
        return Err(GlyphError::MetadataBudgetExceeded {
            limit: budget.max_metadata_bytes,
            actual: actual_metadata_bytes,
        });
    }
    let image = image::create_image_resources(
        device,
        queue,
        renderer_identity,
        width,
        height,
        pixels,
        budget.image,
    )?;
    Ok(GlyphAtlas2d {
        image,
        entries,
        budget,
    })
}

pub(super) fn create_glyph_run_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    atlas: &GlyphAtlas2d,
    glyphs: Vec<PositionedGlyph2d>,
    budget: GlyphRunBudget,
) -> Result<GlyphRun2d, GlyphError> {
    validate_glyph_run_budget(glyphs.len(), budget)?;
    validate_glyph_membership(&atlas.entries, &glyphs)?;
    let bounds = measure_glyphs(&glyphs)?;
    let batch_budget = ImageBatchBudget::new(budget.max_glyphs, budget.max_retained_bytes)?;
    image::preflight_image_batch_capacity(device, glyphs.len(), batch_budget)?;
    let glyphs = compact_glyph_vec(glyphs, budget.max_retained_bytes)?;
    let sprites = resolve_glyph_sprites(&atlas.entries, &glyphs)?;
    let actual_retained_bytes = glyphs
        .capacity()
        .saturating_mul(std::mem::size_of::<PositionedGlyph2d>())
        .saturating_add(
            sprites
                .capacity()
                .saturating_mul(std::mem::size_of::<ImageSprite2d>()),
        );
    if actual_retained_bytes > budget.max_retained_bytes {
        return Err(GlyphError::MetadataBudgetExceeded {
            limit: budget.max_retained_bytes,
            actual: actual_retained_bytes,
        });
    }
    let batch = image::create_image_batch_resources(
        device,
        queue,
        renderer_identity,
        &atlas.image,
        sprites,
        batch_budget,
    )?;
    Ok(GlyphRun2d {
        batch,
        glyphs,
        bounds,
        budget,
    })
}

fn compact_glyph_vec<T: Copy>(values: Vec<T>, maximum_bytes: usize) -> Result<Vec<T>, GlyphError> {
    if values.capacity().saturating_mul(std::mem::size_of::<T>()) <= maximum_bytes {
        return Ok(values);
    }
    let requested_bytes = values
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(GlyphError::InvalidGlyph)?;
    let mut compact = Vec::new();
    compact
        .try_reserve_exact(values.len())
        .map_err(|_| GlyphError::AllocationFailed { requested_bytes })?;
    compact.extend_from_slice(&values);
    Ok(compact)
}

fn validate_atlas_entries(
    entries: &mut [GlyphAtlasEntry],
    width: u32,
    height: u32,
    budget: GlyphAtlasBudget,
) -> Result<(), GlyphError> {
    validate_metadata_capacity(entries.len(), budget)?;
    entries.sort_unstable_by_key(|entry| entry.glyph);
    for (index, entry) in entries.iter().enumerate() {
        if !entry.source.fits(width, height) {
            return Err(GlyphError::InvalidGlyph);
        }
        if index > 0 && entries[index - 1].glyph == entry.glyph {
            return Err(GlyphError::DuplicateGlyph { glyph: entry.glyph });
        }
    }
    validate_non_overlapping_glyph_entries(entries)
}

#[derive(Clone, Copy)]
struct GlyphSweepEvent {
    x: u32,
    starts: bool,
    entry_index: usize,
    minimum_y: u32,
    maximum_y: u32,
}

fn validate_non_overlapping_glyph_entries(entries: &[GlyphAtlasEntry]) -> Result<(), GlyphError> {
    if entries.len() < 2 {
        return Ok(());
    }
    let endpoint_count = entries
        .len()
        .checked_mul(2)
        .ok_or(GlyphError::InvalidGlyph)?;
    let mut events = Vec::new();
    events
        .try_reserve_exact(endpoint_count)
        .map_err(|_| GlyphError::AllocationFailed {
            requested_bytes: endpoint_count.saturating_mul(std::mem::size_of::<GlyphSweepEvent>()),
        })?;
    let mut vertical_coordinates = Vec::new();
    vertical_coordinates
        .try_reserve_exact(endpoint_count)
        .map_err(|_| GlyphError::AllocationFailed {
            requested_bytes: endpoint_count.saturating_mul(std::mem::size_of::<u32>()),
        })?;
    for (entry_index, entry) in entries.iter().copied().enumerate() {
        let source = entry.source;
        let maximum_x = source
            .x()
            .checked_add(source.width())
            .ok_or(GlyphError::InvalidGlyph)?;
        let maximum_y = source
            .y()
            .checked_add(source.height())
            .ok_or(GlyphError::InvalidGlyph)?;
        events.push(GlyphSweepEvent {
            x: source.x(),
            starts: true,
            entry_index,
            minimum_y: source.y(),
            maximum_y,
        });
        events.push(GlyphSweepEvent {
            x: maximum_x,
            starts: false,
            entry_index,
            minimum_y: source.y(),
            maximum_y,
        });
        vertical_coordinates.push(source.y());
        vertical_coordinates.push(maximum_y);
    }
    // End events precede start events at the same x so edge-touching atlas
    // rectangles remain valid.
    events.sort_unstable_by_key(|event| (event.x, event.starts));
    vertical_coordinates.sort_unstable();
    vertical_coordinates.dedup();
    let interval_count = vertical_coordinates.len().saturating_sub(1);
    let tree_len = interval_count
        .checked_mul(4)
        .and_then(|value| value.checked_add(4))
        .ok_or(GlyphError::InvalidGlyph)?;
    let mut maximum_coverage = Vec::new();
    maximum_coverage
        .try_reserve_exact(tree_len)
        .map_err(|_| GlyphError::AllocationFailed {
            requested_bytes: tree_len.saturating_mul(std::mem::size_of::<i32>()),
        })?;
    maximum_coverage.resize(tree_len, 0_i32);
    let mut lazy_coverage = Vec::new();
    lazy_coverage
        .try_reserve_exact(tree_len)
        .map_err(|_| GlyphError::AllocationFailed {
            requested_bytes: tree_len.saturating_mul(std::mem::size_of::<i32>()),
        })?;
    lazy_coverage.resize(tree_len, 0_i32);
    let mut active = Vec::new();
    active
        .try_reserve_exact(entries.len())
        .map_err(|_| GlyphError::AllocationFailed {
            requested_bytes: entries.len().saturating_mul(std::mem::size_of::<bool>()),
        })?;
    active.resize(entries.len(), false);

    for event in events {
        let minimum = vertical_coordinates
            .binary_search(&event.minimum_y)
            .expect("glyph sweep endpoints came from the coordinate set");
        let maximum = vertical_coordinates
            .binary_search(&event.maximum_y)
            .expect("glyph sweep endpoints came from the coordinate set");
        if event.starts {
            if glyph_coverage_max(
                &maximum_coverage,
                &lazy_coverage,
                1,
                0,
                interval_count,
                minimum,
                maximum,
            ) > 0
            {
                let entry = entries[event.entry_index];
                let conflict = active
                    .iter()
                    .enumerate()
                    .find(|(index, active)| {
                        **active && glyph_rectangles_overlap(entries[*index].source, entry.source)
                    })
                    .map(|(index, _)| entries[index].glyph)
                    .expect("positive sweep coverage has an active owning glyph");
                return Err(GlyphError::OverlappingGlyph {
                    glyph: entry.glyph,
                    conflicting_glyph: conflict,
                });
            }
            glyph_coverage_add(
                &mut maximum_coverage,
                &mut lazy_coverage,
                1,
                0,
                interval_count,
                minimum,
                maximum,
                1,
            );
            active[event.entry_index] = true;
        } else {
            glyph_coverage_add(
                &mut maximum_coverage,
                &mut lazy_coverage,
                1,
                0,
                interval_count,
                minimum,
                maximum,
                -1,
            );
            active[event.entry_index] = false;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn glyph_coverage_add(
    maximum_coverage: &mut [i32],
    lazy_coverage: &mut [i32],
    node: usize,
    left: usize,
    right: usize,
    query_left: usize,
    query_right: usize,
    amount: i32,
) {
    if query_left <= left && right <= query_right {
        maximum_coverage[node] += amount;
        lazy_coverage[node] += amount;
        return;
    }
    let middle = left + (right - left) / 2;
    if query_left < middle {
        glyph_coverage_add(
            maximum_coverage,
            lazy_coverage,
            node * 2,
            left,
            middle,
            query_left,
            query_right,
            amount,
        );
    }
    if query_right > middle {
        glyph_coverage_add(
            maximum_coverage,
            lazy_coverage,
            node * 2 + 1,
            middle,
            right,
            query_left,
            query_right,
            amount,
        );
    }
    maximum_coverage[node] =
        lazy_coverage[node] + maximum_coverage[node * 2].max(maximum_coverage[node * 2 + 1]);
}

#[allow(clippy::too_many_arguments)]
fn glyph_coverage_max(
    maximum_coverage: &[i32],
    lazy_coverage: &[i32],
    node: usize,
    left: usize,
    right: usize,
    query_left: usize,
    query_right: usize,
) -> i32 {
    if query_left <= left && right <= query_right {
        return maximum_coverage[node];
    }
    let middle = left + (right - left) / 2;
    let mut maximum = 0;
    if query_left < middle {
        maximum = maximum.max(glyph_coverage_max(
            maximum_coverage,
            lazy_coverage,
            node * 2,
            left,
            middle,
            query_left,
            query_right,
        ));
    }
    if query_right > middle {
        maximum = maximum.max(glyph_coverage_max(
            maximum_coverage,
            lazy_coverage,
            node * 2 + 1,
            middle,
            right,
            query_left,
            query_right,
        ));
    }
    lazy_coverage[node] + maximum
}

fn validate_glyph_insertion(
    entries: &[GlyphAtlasEntry],
    entry: GlyphAtlasEntry,
) -> Result<Option<usize>, GlyphError> {
    match entries.binary_search_by_key(&entry.glyph, |candidate| candidate.glyph) {
        Ok(index) => {
            if entries[index].source != entry.source {
                return Err(GlyphError::DuplicateGlyph { glyph: entry.glyph });
            }
            Ok(None)
        }
        Err(index) => {
            if let Some(conflict) = entries
                .iter()
                .find(|candidate| glyph_rectangles_overlap(candidate.source, entry.source))
            {
                return Err(GlyphError::OverlappingGlyph {
                    glyph: entry.glyph,
                    conflicting_glyph: conflict.glyph,
                });
            }
            Ok(Some(index))
        }
    }
}

fn glyph_rectangles_overlap(left: ImageTexelRect, right: ImageTexelRect) -> bool {
    let left_max_x = u64::from(left.x()) + u64::from(left.width());
    let left_max_y = u64::from(left.y()) + u64::from(left.height());
    let right_max_x = u64::from(right.x()) + u64::from(right.width());
    let right_max_y = u64::from(right.y()) + u64::from(right.height());
    u64::from(left.x()) < right_max_x
        && u64::from(right.x()) < left_max_x
        && u64::from(left.y()) < right_max_y
        && u64::from(right.y()) < left_max_y
}

fn validate_metadata_capacity(
    entry_count: usize,
    budget: GlyphAtlasBudget,
) -> Result<(), GlyphError> {
    if entry_count > budget.max_entries {
        return Err(GlyphError::EntryBudgetExceeded {
            limit: budget.max_entries,
            actual: entry_count,
        });
    }
    let bytes = entry_count
        .checked_mul(std::mem::size_of::<GlyphAtlasEntry>())
        .ok_or(GlyphError::InvalidGlyph)?;
    if bytes > budget.max_metadata_bytes {
        return Err(GlyphError::MetadataBudgetExceeded {
            limit: budget.max_metadata_bytes,
            actual: bytes,
        });
    }
    Ok(())
}

fn validate_glyph_run_budget(glyph_count: usize, budget: GlyphRunBudget) -> Result<(), GlyphError> {
    if glyph_count > budget.max_glyphs {
        return Err(GlyphError::EntryBudgetExceeded {
            limit: budget.max_glyphs,
            actual: glyph_count,
        });
    }
    let retained_bytes = glyph_retained_bytes(glyph_count).ok_or(GlyphError::InvalidGlyph)?;
    if retained_bytes > budget.max_retained_bytes {
        return Err(GlyphError::MetadataBudgetExceeded {
            limit: budget.max_retained_bytes,
            actual: retained_bytes,
        });
    }
    Ok(())
}

fn glyph_retained_bytes(glyph_count: usize) -> Option<usize> {
    let glyph_bytes = glyph_count.checked_mul(std::mem::size_of::<PositionedGlyph2d>())?;
    let sprite_bytes = glyph_count.checked_mul(std::mem::size_of::<ImageSprite2d>())?;
    glyph_bytes.checked_add(sprite_bytes)
}

fn find_entry(entries: &[GlyphAtlasEntry], glyph: GlyphId) -> Option<GlyphAtlasEntry> {
    entries
        .binary_search_by_key(&glyph, |entry| entry.glyph)
        .ok()
        .map(|index| entries[index])
}

fn resolve_glyph_sprites(
    entries: &[GlyphAtlasEntry],
    glyphs: &[PositionedGlyph2d],
) -> Result<Vec<ImageSprite2d>, GlyphError> {
    validate_glyph_membership(entries, glyphs)?;
    let mut sprites = Vec::new();
    sprites
        .try_reserve_exact(glyphs.len())
        .map_err(|_| GlyphError::AllocationFailed {
            requested_bytes: glyphs
                .len()
                .saturating_mul(std::mem::size_of::<ImageSprite2d>()),
        })?;
    for (index, glyph) in glyphs.iter().copied().enumerate() {
        let source = find_entry(entries, glyph.glyph)
            .map(GlyphAtlasEntry::source)
            // The complete lookup scan above makes this branch defensive.
            .ok_or(GlyphError::MissingGlyph {
                glyph: glyph.glyph,
                index,
            })?;
        sprites.push(ImageSprite2d::new(source, glyph.destination, glyph.tint)?);
    }
    Ok(sprites)
}

fn validate_glyph_membership(
    entries: &[GlyphAtlasEntry],
    glyphs: &[PositionedGlyph2d],
) -> Result<(), GlyphError> {
    if let Some((index, glyph)) = glyphs
        .iter()
        .enumerate()
        .find(|(_, glyph)| find_entry(entries, glyph.glyph).is_none())
    {
        return Err(GlyphError::MissingGlyph {
            glyph: glyph.glyph,
            index,
        });
    }
    Ok(())
}

fn validate_glyph_run_sources(
    entries: &[GlyphAtlasEntry],
    glyphs: &[PositionedGlyph2d],
    sprites: &[ImageSprite2d],
) -> Result<(), GlyphError> {
    if glyphs.len() != sprites.len()
        || glyphs.iter().zip(sprites).any(|(glyph, sprite)| {
            find_entry(entries, glyph.glyph).map(GlyphAtlasEntry::source) != Some(sprite.source())
        })
    {
        return Err(ImageError::RecoverySourceMismatch.into());
    }
    Ok(())
}

fn measure_glyphs(glyphs: &[PositionedGlyph2d]) -> Result<GlyphRunBounds, GlyphError> {
    if glyphs
        .iter()
        .any(|glyph| !image::logical_image_region_is_portable(glyph.destination))
    {
        return Err(GlyphError::InvalidGlyph);
    }
    let Some(first) = glyphs.first().copied() else {
        return Ok(GlyphRunBounds { region: None });
    };
    let first_origin = first.destination.origin().to_vec2();
    let first_size = first.destination.viewport().size();
    let mut minimum = first_origin;
    let mut maximum = first_origin + first_size;
    for glyph in &glyphs[1..] {
        if !logical_region_arithmetic_is_finite(glyph.destination) {
            return Err(GlyphError::InvalidGlyph);
        }
        let origin = glyph.destination.origin().to_vec2();
        let end = origin + glyph.destination.viewport().size();
        minimum.x = minimum.x.min(origin.x);
        minimum.y = minimum.y.min(origin.y);
        maximum.x = maximum.x.max(end.x);
        maximum.y = maximum.y.max(end.y);
    }
    if !minimum.is_finite() || !maximum.is_finite() {
        return Err(GlyphError::InvalidGlyph);
    }
    let viewport = LogicalViewport::new(maximum.x - minimum.x, maximum.y - minimum.y)
        .map_err(|_| GlyphError::InvalidGlyph)?;
    let region = LogicalViewportRegion::new(LogicalScreenPosition::from_vec2(minimum), viewport)
        .map_err(|_| GlyphError::InvalidGlyph)?;
    Ok(GlyphRunBounds {
        region: Some(region),
    })
}

fn logical_region_arithmetic_is_finite(region: LogicalViewportRegion) -> bool {
    let origin = region.origin().to_vec2();
    let end = origin + region.viewport().size();
    origin.is_finite() && end.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rectangle(x: u32) -> ImageTexelRect {
        ImageTexelRect::new(x, 0, 1, 1).unwrap()
    }

    fn positioned(glyph: u32, x: f32) -> PositionedGlyph2d {
        PositionedGlyph2d::new(
            GlyphId::new(glyph),
            LogicalViewportRegion::new(
                LogicalScreenPosition::new(x, 2.0),
                LogicalViewport::new(4.0, 6.0).unwrap(),
            )
            .unwrap(),
            Color::WHITE,
        )
        .unwrap()
    }

    #[test]
    fn atlas_entries_sort_and_reject_duplicate_or_out_of_bounds_glyphs() {
        let budget = GlyphAtlasBudget::default();
        let mut entries = vec![
            GlyphAtlasEntry::new(GlyphId::new(2), rectangle(1)),
            GlyphAtlasEntry::new(GlyphId::new(1), rectangle(0)),
        ];
        assert_eq!(validate_atlas_entries(&mut entries, 2, 1, budget), Ok(()));
        assert_eq!(entries[0].glyph(), GlyphId::new(1));

        let mut duplicate = vec![
            GlyphAtlasEntry::new(GlyphId::new(7), rectangle(0)),
            GlyphAtlasEntry::new(GlyphId::new(7), rectangle(1)),
        ];
        assert_eq!(
            validate_atlas_entries(&mut duplicate, 2, 1, budget),
            Err(GlyphError::DuplicateGlyph {
                glyph: GlyphId::new(7)
            })
        );

        let mut outside = vec![GlyphAtlasEntry::new(GlyphId::new(9), rectangle(2))];
        assert_eq!(
            validate_atlas_entries(&mut outside, 2, 1, budget),
            Err(GlyphError::InvalidGlyph)
        );

        let mut overlapping = vec![
            GlyphAtlasEntry::new(GlyphId::new(10), ImageTexelRect::new(0, 0, 2, 2).unwrap()),
            GlyphAtlasEntry::new(GlyphId::new(11), ImageTexelRect::new(1, 1, 2, 2).unwrap()),
        ];
        assert_eq!(
            validate_atlas_entries(&mut overlapping, 4, 4, budget),
            Err(GlyphError::OverlappingGlyph {
                glyph: GlyphId::new(11),
                conflicting_glyph: GlyphId::new(10),
            })
        );
    }

    #[test]
    fn incremental_glyph_overlap_is_rejected_before_metadata_or_pixel_update() {
        let entries = [GlyphAtlasEntry::new(
            GlyphId::new(1),
            ImageTexelRect::new(0, 0, 4, 4).unwrap(),
        )];
        let overlapping =
            GlyphAtlasEntry::new(GlyphId::new(2), ImageTexelRect::new(3, 0, 4, 4).unwrap());
        assert_eq!(
            validate_glyph_insertion(&entries, overlapping),
            Err(GlyphError::OverlappingGlyph {
                glyph: GlyphId::new(2),
                conflicting_glyph: GlyphId::new(1),
            })
        );
        assert_eq!(entries.len(), 1);

        let adjacent =
            GlyphAtlasEntry::new(GlyphId::new(2), ImageTexelRect::new(4, 0, 4, 4).unwrap());
        assert_eq!(validate_glyph_insertion(&entries, adjacent), Ok(Some(1)));

        let below = GlyphAtlasEntry::new(GlyphId::new(2), ImageTexelRect::new(0, 8, 4, 4).unwrap());
        assert_eq!(validate_glyph_insertion(&entries, below), Ok(Some(1)));

        let entries_below = [GlyphAtlasEntry::new(
            GlyphId::new(1),
            ImageTexelRect::new(0, 8, 4, 4).unwrap(),
        )];
        let above = GlyphAtlasEntry::new(GlyphId::new(2), ImageTexelRect::new(0, 0, 4, 4).unwrap());
        assert_eq!(validate_glyph_insertion(&entries_below, above), Ok(Some(1)));
    }

    #[test]
    fn default_maximum_non_overlapping_atlas_validation_is_subquadratic() {
        let mut entries = Vec::with_capacity(65_536);
        for glyph in 0..65_536_u32 {
            entries.push(GlyphAtlasEntry::new(
                GlyphId::new(glyph),
                ImageTexelRect::new(glyph % 256, glyph / 256, 1, 1).unwrap(),
            ));
        }
        assert_eq!(
            validate_atlas_entries(&mut entries, 256, 256, GlyphAtlasBudget::default()),
            Ok(())
        );
    }

    #[test]
    fn positioned_run_resolves_opaque_scientific_glyphs_and_measures_bounds() {
        let entries = [
            GlyphAtlasEntry::new(GlyphId::new('Δ' as u32), rectangle(1)),
            GlyphAtlasEntry::new(GlyphId::new('μ' as u32), rectangle(0)),
            GlyphAtlasEntry::new(GlyphId::new('∫' as u32), rectangle(2)),
        ];
        let glyphs = [
            positioned('μ' as u32, 1.0),
            positioned('Δ' as u32, 5.0),
            positioned('∫' as u32, 9.0),
        ];
        assert_eq!(resolve_glyph_sprites(&entries, &glyphs).unwrap().len(), 3);
        let bounds = measure_glyphs(&glyphs).unwrap().region().unwrap();
        assert_eq!(bounds.origin(), LogicalScreenPosition::new(1.0, 2.0));
        assert_eq!(bounds.viewport().size(), Vec2::new(12.0, 6.0));

        let missing = [positioned('Σ' as u32, 1.0)];
        assert_eq!(
            resolve_glyph_sprites(&entries, &missing),
            Err(GlyphError::MissingGlyph {
                glyph: GlyphId::new('Σ' as u32),
                index: 0,
            })
        );
    }

    #[test]
    fn glyph_run_recovery_rejects_forked_atlas_mapping() {
        let glyph = positioned(42, 1.0);
        let left = rectangle(0);
        let right = rectangle(1);
        let source_entries = [GlyphAtlasEntry::new(GlyphId::new(42), left)];
        let forked_entries = [GlyphAtlasEntry::new(GlyphId::new(42), right)];
        let source_sprites = [ImageSprite2d::new(left, glyph.destination(), glyph.tint()).unwrap()];

        assert_eq!(
            validate_glyph_run_sources(&source_entries, &[glyph], &source_sprites),
            Ok(())
        );
        assert_eq!(
            validate_glyph_run_sources(&forked_entries, &[glyph], &source_sprites),
            Err(GlyphError::Image(ImageError::RecoverySourceMismatch))
        );
    }
}
