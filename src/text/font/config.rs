use crate::{LogicalPixels, PhysicalPerLogical};

/// Resource whose explicit font or layout limit was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontBudgetResource {
    /// Capacity of the owned source-font byte vector.
    FontBytes,
    /// Glyph entries declared by the font.
    FontGlyphs,
    /// UTF-8 bytes submitted for one line, checked before shaping.
    TextBytes,
    /// Shaped output glyphs, checked after dependency shaping.
    ShapedGlyphs,
    /// Pixel cells in one glyph's physical coverage bitmap.
    GlyphPixels,
    /// Outline commands in one glyph, counted before outline/raster allocation.
    OutlineSegments,
}

/// Explicit loading, shaping or rasterization failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FontError {
    /// Invalid, unsupported or inconsistent TrueType/OpenType source.
    InvalidFont,
    /// A source or layout resource exceeded its configured limit.
    BudgetExceeded {
        /// Rejected resource category.
        resource: FontBudgetResource,
        /// Required elements or bytes.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Logical em size and DPI produce non-normal or overflowing physical size.
    InvalidScale,
    /// Single-line shaping does not accept control characters or line separators.
    UnsupportedText {
        /// UTF-8 byte offset of the unsupported character.
        byte_index: usize,
    },
    /// The selected font has no glyph; this API never silently selects a fallback.
    MissingGlyph {
        /// UTF-8 byte offset of the missing cluster.
        byte_index: usize,
    },
    /// Raster input is outside this font's glyph table.
    InvalidGlyph {
        /// Rejected font-local glyph identifier.
        glyph_id: u16,
    },
    /// A glyph requires color/SVG/bitmap presentation rather than this outline path.
    UnsupportedGlyph {
        /// Font-local identifier requiring an unsupported glyph representation.
        glyph_id: u16,
    },
    /// Shaped positions, outline bounds or raster arithmetic are not representable.
    InvalidGeometry,
    /// A library-owned vector could not be reserved.
    AllocationFailed {
        /// Requested payload or metadata allocation size.
        requested_bytes: usize,
    },
}

impl std::fmt::Display for FontError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFont => formatter.write_str("invalid or unsupported outline font"),
            Self::BudgetExceeded {
                resource,
                required,
                limit,
            } => write!(
                formatter,
                "font {resource:?} requires {required}, limit is {limit}"
            ),
            Self::InvalidScale => formatter.write_str("invalid physical font scale"),
            Self::UnsupportedText { byte_index } => write!(
                formatter,
                "unsupported single-line character at UTF-8 byte {byte_index}"
            ),
            Self::MissingGlyph { byte_index } => write!(
                formatter,
                "font has no glyph for UTF-8 cluster at byte {byte_index}"
            ),
            Self::InvalidGlyph { glyph_id } => {
                write!(formatter, "invalid font glyph id {glyph_id}")
            }
            Self::UnsupportedGlyph { glyph_id } => write!(
                formatter,
                "font glyph {glyph_id} requires unsupported color or bitmap presentation"
            ),
            Self::InvalidGeometry => {
                formatter.write_str("font geometry is not finite or representable")
            }
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "font allocation failed for {requested_bytes} bytes"
            ),
        }
    }
}

impl std::error::Error for FontError {}

/// Limits checked before taking ownership of a parsed font.
/// Parser-owned metadata and allocator overhead are not included in source bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FontBudget {
    max_font_bytes: usize,
    max_font_glyphs: usize,
}

impl FontBudget {
    /// Creates exact limits. Zero is valid and rejects nonempty resources.
    pub const fn new(max_font_bytes: usize, max_font_glyphs: usize) -> Self {
        Self {
            max_font_bytes,
            max_font_glyphs,
        }
    }
    /// Maximum source Vec capacity in bytes, not merely its initialized length.
    pub const fn max_font_bytes(self) -> usize {
        self.max_font_bytes
    }
    /// Maximum number of glyph entries in the selected font face.
    pub const fn max_font_glyphs(self) -> usize {
        self.max_font_glyphs
    }
}

impl Default for FontBudget {
    fn default() -> Self {
        Self::new(8 * 1024 * 1024, 65_535)
    }
}

/// Per-line and per-glyph work limits.
///
/// UTF-8 input is bounded before rustybuzz runs; its output is checked before
/// library-owned glyph allocation. Dependency parser/shaper allocations are
/// not allocation-fallible through this API. Raster pixel and outline limits
/// are checked before ab_glyph allocates its pixel scratch (roughly four bytes
/// per pixel). These are not an OS OOM or hostile-font execution-time sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextLayoutBudget {
    max_text_bytes: usize,
    max_shaped_glyphs: usize,
    max_glyph_pixels: usize,
    max_outline_segments: usize,
}

impl TextLayoutBudget {
    /// Creates exact limits; raster limits apply independently to each glyph.
    pub const fn new(
        text_bytes: usize,
        shaped_glyphs: usize,
        glyph_pixels: usize,
        outline_segments: usize,
    ) -> Self {
        Self {
            max_text_bytes: text_bytes,
            max_shaped_glyphs: shaped_glyphs,
            max_glyph_pixels: glyph_pixels,
            max_outline_segments: outline_segments,
        }
    }
    /// Maximum UTF-8 source-line length in bytes.
    pub const fn max_text_bytes(self) -> usize {
        self.max_text_bytes
    }
    /// Maximum shaped output glyph count.
    pub const fn max_shaped_glyphs(self) -> usize {
        self.max_shaped_glyphs
    }
    /// Maximum coverage pixels in one rasterized glyph.
    pub const fn max_glyph_pixels(self) -> usize {
        self.max_glyph_pixels
    }
    /// Maximum outline commands per glyph, including moves and closes.
    pub const fn max_outline_segments(self) -> usize {
        self.max_outline_segments
    }
}

impl Default for TextLayoutBudget {
    fn default() -> Self {
        Self::new(16 * 1024, 16 * 1024, 1024 * 1024, 16 * 1024)
    }
}

/// Horizontal shaping direction for one single-script directional run.
/// This does not perform Unicode bidi paragraph segmentation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextDirection {
    /// Infer direction and script from this line's first strong character.
    #[default]
    Auto,
    /// Explicit left-to-right shaping.
    Ltr,
    /// Explicit right-to-left shaping; glyph ordering follows rustybuzz.
    Rtl,
}

/// Immutable logical em size and raster DPI, independent of text color/placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextStyle {
    logical_em_size: LogicalPixels,
    scale: PhysicalPerLogical,
    direction: TextDirection,
}

impl TextStyle {
    /// Creates a logical-pixel em size and physical-pixels-per-logical-pixel scale.
    /// Rejects an overflowing or subnormal physical em size.
    pub fn new(
        logical_em_size: LogicalPixels,
        scale: PhysicalPerLogical,
    ) -> Result<Self, FontError> {
        let physical = logical_em_size.get() * scale.get();
        if !physical.is_normal() || physical <= 0.0 {
            return Err(FontError::InvalidScale);
        }
        Ok(Self {
            logical_em_size,
            scale,
            direction: TextDirection::Auto,
        })
    }
    /// Selects shaping direction without changing em size or raster DPI.
    pub const fn with_direction(mut self, direction: TextDirection) -> Self {
        self.direction = direction;
        self
    }
    /// Logical pixels per em; not font ascent-minus-descent height.
    pub const fn logical_em_size(self) -> LogicalPixels {
        self.logical_em_size
    }
    /// Physical pixels per logical pixel used for glyph rasterization.
    pub const fn scale(self) -> PhysicalPerLogical {
        self.scale
    }
    /// Requested horizontal shaping direction.
    pub const fn direction(self) -> TextDirection {
        self.direction
    }
    /// Physical pixels per em, validated at construction.
    pub fn physical_em_size(self) -> f32 {
        self.logical_em_size.get() * self.scale.get()
    }
}

pub(super) fn check(
    resource: FontBudgetResource,
    required: usize,
    limit: usize,
) -> Result<(), FontError> {
    if required > limit {
        Err(FontError::BudgetExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

pub(super) fn reserve<T>(values: &mut Vec<T>, count: usize) -> Result<(), FontError> {
    let requested_bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(FontError::InvalidGeometry)?;
    values
        .try_reserve_exact(count)
        .map_err(|_| FontError::AllocationFailed { requested_bytes })
}
