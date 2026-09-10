use super::*;

/// Fixed atlas texel/cache capacity and per-run limits for optional font rendering.
///
/// The default is a 1024x1024 RGBA atlas (4 MiB CPU plus 4 MiB GPU), at most
/// 4096 cached glyphs and the default [`GlyphRunBudget`]. Glyph cache metadata is
/// reserved once and grows neither beyond its count limit nor through eviction.
/// Shared font bytes, shaping/rasterizer scratch and host-owned runs are separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextAtlasBudget {
    width: u32,
    height: u32,
    max_cached_glyphs: usize,
    run: GlyphRunBudget,
}

impl TextAtlasBudget {
    /// Creates fixed physical-texel dimensions and a glyph count in `1..=65535`.
    /// Dimensions are additionally checked against the actual GPU at atlas creation.
    pub fn new(
        width: u32,
        height: u32,
        max_cached_glyphs: usize,
        run: GlyphRunBudget,
    ) -> Result<Self, TextError> {
        if width < 5 || height < 5 || !(1..=65535).contains(&max_cached_glyphs) {
            return Err(TextError::InvalidBudget);
        }
        let budget = Self {
            width,
            height,
            max_cached_glyphs,
            run,
        };
        budget.image_budget()?;
        Ok(budget)
    }
    /// Returns fixed atlas width in physical texels.
    pub const fn width(self) -> u32 {
        self.width
    }
    /// Returns fixed atlas height in physical texels.
    pub const fn height(self) -> u32 {
        self.height
    }
    /// Returns maximum cached glyphs, including spacing glyphs.
    pub const fn max_cached_glyphs(self) -> usize {
        self.max_cached_glyphs
    }
    /// Returns limits applied to each created retained glyph run.
    pub const fn run_budget(self) -> GlyphRunBudget {
        self.run
    }

    pub(super) fn image_budget(self) -> Result<ImageBudget, TextError> {
        let bytes = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|area| area.checked_mul(4))
            .ok_or(TextError::InvalidBudget)?;
        Ok(ImageBudget::new(self.width, self.height, bytes)?)
    }
}

impl Default for TextAtlasBudget {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            max_cached_glyphs: 4096,
            run: GlyphRunBudget::default(),
        }
    }
}

/// A rejected text asset, layout, cache operation or GPU resource operation.
#[derive(Debug)]
pub enum TextError {
    /// Font parsing, shaping or rasterization failed its validation or work limits.
    Font(FontError),
    /// The underlying glyph/image allocation or identity contract was violated.
    Glyph(GlyphError),
    /// Atlas dimensions/counts or their checked byte calculation are invalid.
    InvalidBudget,
    /// The fixed shelf-packed atlas cannot fit a new padded glyph.
    AtlasFull,
    /// The fixed number of cached glyph identities has been reached.
    GlyphCacheFull,
    /// A run belongs to another font atlas, even if both use the same font bytes.
    AtlasMismatch,
    /// Bitmap bearings or positioned glyph coordinates cannot form a finite logical quad.
    InvalidPlacement,
    /// An engine-owned CPU allocation could not be reserved.
    AllocationFailed,
}

impl fmt::Display for TextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Font(error) => write!(formatter, "text font: {error}"),
            Self::Glyph(error) => write!(formatter, "text glyph resource: {error}"),
            Self::InvalidBudget => formatter.write_str("invalid text atlas budget"),
            Self::AtlasFull => formatter.write_str("fixed text atlas is full"),
            Self::GlyphCacheFull => formatter.write_str("text glyph cache count limit reached"),
            Self::AtlasMismatch => formatter.write_str("text run belongs to another atlas"),
            Self::InvalidPlacement => {
                formatter.write_str("text placement is not finite or representable")
            }
            Self::AllocationFailed => formatter.write_str("text storage allocation failed"),
        }
    }
}
impl Error for TextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Font(error) => Some(error),
            Self::Glyph(error) => Some(error),
            _ => None,
        }
    }
}
impl From<FontError> for TextError {
    fn from(error: FontError) -> Self {
        Self::Font(error)
    }
}
impl From<GlyphError> for TextError {
    fn from(error: GlyphError) -> Self {
        Self::Glyph(error)
    }
}
impl From<ImageError> for TextError {
    fn from(error: ImageError) -> Self {
        Self::Glyph(error.into())
    }
}

/// Successful update diagnostics. Unchanged text reports zero work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TextUpdateReport {
    changed: bool,
    instance_uploaded_bytes: usize,
    cached_glyphs_added: usize,
}
impl TextUpdateReport {
    pub(super) const fn new(instance_uploaded_bytes: usize, cached_glyphs_added: usize) -> Self {
        Self {
            changed: true,
            instance_uploaded_bytes,
            cached_glyphs_added,
        }
    }
    /// Returns whether the retained UTF-8 text changed.
    pub const fn changed(self) -> bool {
        self.changed
    }
    /// Returns glyph instance upload bytes, excluding cache-miss atlas pixel uploads.
    pub const fn instance_uploaded_bytes(self) -> usize {
        self.instance_uploaded_bytes
    }
    /// Returns newly cached glyph count, including spacing glyphs.
    pub const fn cached_glyphs_added(self) -> usize {
        self.cached_glyphs_added
    }
}
