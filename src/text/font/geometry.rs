//! Borrowed editor geometry derived from the shaped pen, never glyph ink origins.

use super::{FontError, ShapedLine, TextDirection, represent};
use std::{error::Error, fmt, iter::FusedIterator, ops::Range};

/// Finite baseline-relative pen position in logical pixels, positive y downward.
/// Font glyph placement offsets and ink bearings are excluded. Raster DPI does
/// not change this position; the host adds its own baseline/scroll transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextPenPosition {
    logical_x: f32,
    logical_y: f32,
}
impl TextPenPosition {
    const ORIGIN: Self = Self {
        logical_x: 0.0,
        logical_y: 0.0,
    };
    pub(super) fn new(x: f64, y: f64) -> Result<Self, FontError> {
        Ok(Self {
            logical_x: represent(x)?,
            logical_y: represent(y)?,
        })
    }
    /// Horizontal baseline-relative coordinate in logical pixels.
    pub const fn logical_x(self) -> f32 {
        self.logical_x
    }
    /// Vertical baseline-relative coordinate in logical pixels, positive downward.
    pub const fn logical_y(self) -> f32 {
        self.logical_y
    }
}

/// Insertion policy used by shaped-line editor geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LigatureCaretPolicy {
    /// Only complete shaping-cluster boundaries are insertion positions.
    /// Ligatures and merged combining sequences are indivisible. No GDEF caret
    /// table, equal-width subdivision or grapheme-editing policy is inferred.
    ClusterBoundaries,
}

/// A requested editor range cannot be represented by exact cluster boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShapedGeometryError {
    /// Offset exceeds the input length or falls inside a UTF-8 code point.
    InvalidByteOffset {
        /// Requested offset into the original UTF-8 input.
        byte_offset: usize,
    },
    /// Offset is a character boundary inside an indivisible shaping cluster.
    InsideCluster {
        /// Requested offset into the original UTF-8 input.
        byte_offset: usize,
        /// Complete cluster range; the caller must explicitly choose an endpoint.
        cluster_range: Range<usize>,
    },
    /// A logical selection must have start <= end, regardless of visual direction.
    ReversedRange,
}
impl fmt::Display for ShapedGeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidByteOffset { byte_offset } => {
                write!(formatter, "invalid text UTF-8 byte offset {byte_offset}")
            }
            Self::InsideCluster {
                byte_offset,
                cluster_range,
            } => write!(
                formatter,
                "UTF-8 byte {byte_offset} is inside shaped cluster {}..{}",
                cluster_range.start, cluster_range.end
            ),
            Self::ReversedRange => formatter.write_str("reversed logical text selection range"),
        }
    }
}
impl Error for ShapedGeometryError {}

/// One exact cluster-boundary insertion position, borrowing its originating line.
/// This is pen geometry, not an ink bound or a complete cursor rectangle.
#[derive(Debug, Clone, Copy)]
pub struct ShapedCaret<'line> {
    line: &'line ShapedLine,
    byte_offset: usize,
    position: TextPenPosition,
}
impl<'line> ShapedCaret<'line> {
    /// Originating immutable line, including source text and font/style provenance.
    pub const fn line(self) -> &'line ShapedLine {
        self.line
    }
    /// UTF-8 source offset; always a complete shaping-cluster boundary.
    pub const fn byte_offset(self) -> usize {
        self.byte_offset
    }
    /// Baseline-relative position in logical pixels, excluding glyph offsets.
    pub const fn position(self) -> TextPenPosition {
        self.position
    }
}

/// One indivisible shaped cluster in visual output order, borrowing its line.
/// Logical source order may differ from visual order. Pen bounds include spaces
/// and kerning; they are neither glyph ink bounds nor paragraph bidi geometry.
#[derive(Debug, Clone)]
pub struct ShapedCluster<'line> {
    line: &'line ShapedLine,
    byte_range: Range<usize>,
    glyph_range: Range<usize>,
    visual_index: usize,
}
impl<'line> ShapedCluster<'line> {
    /// Originating line, preserving exact font identity, style and source text.
    pub const fn line(&self) -> &'line ShapedLine {
        self.line
    }
    /// Half-open logical UTF-8 range in the complete original source string.
    pub fn byte_range(&self) -> Range<usize> {
        self.byte_range.clone()
    }
    /// Half-open range into `line().glyphs()` in shaper output order.
    pub fn glyph_range(&self) -> Range<usize> {
        self.glyph_range.clone()
    }
    /// Zero-based cluster index in visual output order, not logical byte order.
    pub const fn visual_index(&self) -> usize {
        self.visual_index
    }
    /// Pen at the first glyph in visual output order, excluding glyph offsets.
    /// For RTL this corresponds to the logical source range's end.
    pub fn pen_start(&self) -> TextPenPosition {
        if self.glyph_range.is_empty() {
            TextPenPosition::ORIGIN
        } else {
            self.line.glyphs[self.glyph_range.start].pen_start
        }
    }
    /// Pen after the final glyph in visual output order, including shaped advance.
    /// For RTL this corresponds to the logical source range's start.
    pub fn pen_end(&self) -> TextPenPosition {
        if self.glyph_range.is_empty() {
            TextPenPosition::ORIGIN
        } else {
            self.line.glyphs[self.glyph_range.end - 1].pen_end
        }
    }
    /// Signed `[x, y]` displacement in logical pixels from pen start to pen end.
    /// Subtracts the stored f32 endpoints in f64 to avoid overflow or additional
    /// f32 subtraction rounding. This is not the sum of glyph ink widths.
    pub fn advance(&self) -> [f64; 2] {
        let start = self.pen_start();
        let end = self.pen_end();
        [
            f64::from(end.logical_x) - f64::from(start.logical_x),
            f64::from(end.logical_y) - f64::from(start.logical_y),
        ]
    }
    /// Insertion position at the logical UTF-8 range's start, including in RTL.
    pub fn start_caret(&self) -> ShapedCaret<'line> {
        ShapedCaret {
            line: self.line,
            byte_offset: self.byte_range.start,
            position: if self.line.resolved_direction == TextDirection::Rtl {
                self.pen_end()
            } else {
                self.pen_start()
            },
        }
    }
    /// Insertion position at the logical UTF-8 range's end, including in RTL.
    pub fn end_caret(&self) -> ShapedCaret<'line> {
        ShapedCaret {
            line: self.line,
            byte_offset: self.byte_range.end,
            position: if self.line.resolved_direction == TextDirection::Rtl {
                self.pen_start()
            } else {
                self.pen_end()
            },
        }
    }
}

/// Allocation-free cluster traversal in visual output order, linear in glyphs.
/// Borrows the immutable line; updating that line requires releasing the view.
#[derive(Debug, Clone)]
pub struct ShapedClusters<'line> {
    line: &'line ShapedLine,
    glyph_index: usize,
    visual_index: usize,
}
impl<'line> Iterator for ShapedClusters<'line> {
    type Item = ShapedCluster<'line>;
    fn next(&mut self) -> Option<Self::Item> {
        let glyphs = &self.line.glyphs;
        // Some sparse fonts remove all default ignorables. Preserve accepted
        // source text as one zero-width cluster rather than inventing glyphs or
        // rejecting a drawable that the ordinary shaping path accepts.
        if glyphs.is_empty() && !self.line.text.is_empty() && self.visual_index == 0 {
            self.visual_index = 1;
            return Some(ShapedCluster {
                line: self.line,
                byte_range: 0..self.line.text.len(),
                glyph_range: 0..0,
                visual_index: 0,
            });
        }
        let start = self.glyph_index;
        let first = glyphs.get(start)?;
        self.glyph_index += 1;
        while glyphs
            .get(self.glyph_index)
            .is_some_and(|glyph| glyph.cluster == first.cluster)
        {
            self.glyph_index += 1;
        }
        // Monotone clusters partition the original UTF-8 string. For RTL the
        // previous visual group supplies the next higher logical byte offset.
        let byte_end = if self.line.resolved_direction == TextDirection::Rtl {
            if start == 0 {
                self.line.text.len()
            } else {
                glyphs[start - 1].cluster
            }
        } else {
            glyphs
                .get(self.glyph_index)
                .map_or(self.line.text.len(), |glyph| glyph.cluster)
        };
        let cluster = ShapedCluster {
            line: self.line,
            byte_range: first.cluster..byte_end,
            glyph_range: start..self.glyph_index,
            visual_index: self.visual_index,
        };
        self.visual_index += 1;
        Some(cluster)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let zero_output =
            self.line.glyphs.is_empty() && !self.line.text.is_empty() && self.visual_index == 0;
        (
            0,
            Some(self.line.glyphs.len() - self.glyph_index + usize::from(zero_output)),
        )
    }
}
impl FusedIterator for ShapedClusters<'_> {}

/// Exact whole-cluster selection spans in visual order, without allocation.
/// Each item retains its logical byte range and original visual index. The host
/// chooses vertical highlight metrics and performs paragraph bidi segmentation.
#[derive(Debug, Clone)]
pub struct ShapedSelection<'line> {
    clusters: ShapedClusters<'line>,
    byte_range: Range<usize>,
}
impl ShapedSelection<'_> {
    /// Validated half-open logical source range, unchanged during traversal.
    pub fn byte_range(&self) -> Range<usize> {
        self.byte_range.clone()
    }
}
impl<'line> Iterator for ShapedSelection<'line> {
    type Item = ShapedCluster<'line>;
    fn next(&mut self) -> Option<Self::Item> {
        self.clusters.find(|cluster| {
            cluster.byte_range.start >= self.byte_range.start
                && cluster.byte_range.end <= self.byte_range.end
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.clusters.size_hint()
    }
}
impl FusedIterator for ShapedSelection<'_> {}

impl ShapedLine {
    /// Actual horizontal direction chosen for this shaped run, never Auto.
    /// This does not resolve mixed-direction paragraphs or editing graphemes.
    pub const fn resolved_direction(&self) -> TextDirection {
        self.resolved_direction
    }
    /// Exact cluster geometry's fixed ligature/combining-sequence insertion policy.
    pub const fn caret_policy(&self) -> LigatureCaretPolicy {
        LigatureCaretPolicy::ClusterBoundaries
    }
    /// Borrows exact pen-based cluster geometry without reshaping or allocation.
    /// Uses monotone-graphemes shaping clusters, not Unicode editing boundaries.
    /// Empty input yields no clusters and still has a caret at byte zero.
    /// Nonempty input entirely removed by shaping yields one zero-width cluster
    /// spanning the source, with glyph range `0..0`; both endpoint carets are at
    /// the origin. Interior offsets still follow the cluster-boundary policy.
    ///
    /// ```no_run
    /// use sim_engine::{FontBudget, FontFace, LogicalPixels, PhysicalPerLogical,
    ///     TextLayoutBudget, TextStyle};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let font = FontFace::from_bytes(std::fs::read("assets/Label.ttf")?, FontBudget::default())?;
    /// let style = TextStyle::new(LogicalPixels::new(24.0)?, PhysicalPerLogical::new(1.0)?)?;
    /// let line = font.shape_line("office", &style, &TextLayoutBudget::default())?;
    /// let caret = line.caret_at(0)?;
    /// assert_eq!(caret.position().logical_x(), 0.0);
    /// for span in line.selection(0..line.text().len())? {
    ///     let source = &line.text()[span.byte_range()];
    ///     let from = span.start_caret().position();
    ///     let to = span.end_caret().position();
    ///     // Use min/max x for a horizontal highlight; RTL logical start is on the right.
    ///     println!("{source}: {}..{}", from.logical_x(), to.logical_x());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn clusters(&self) -> ShapedClusters<'_> {
        ShapedClusters {
            line: self,
            glyph_index: 0,
            visual_index: 0,
        }
    }
    /// Looks up a source insertion position in O(glyph count) without allocation.
    /// Accepts cluster boundaries, including 0 and text length. Invalid UTF-8
    /// offsets and cluster interiors return explicit errors; no snapping occurs.
    /// Empty input's only caret is `(0, 0)` at byte zero in either direction.
    pub fn caret_at(&self, byte_offset: usize) -> Result<ShapedCaret<'_>, ShapedGeometryError> {
        if !self.text.is_char_boundary(byte_offset) {
            return Err(ShapedGeometryError::InvalidByteOffset { byte_offset });
        }
        if self.text.is_empty() {
            return Ok(ShapedCaret {
                line: self,
                byte_offset: 0,
                position: TextPenPosition {
                    logical_x: 0.0,
                    logical_y: 0.0,
                },
            });
        }
        for cluster in self.clusters() {
            if byte_offset == cluster.byte_range.start {
                return Ok(cluster.start_caret());
            }
            if byte_offset == cluster.byte_range.end {
                return Ok(cluster.end_caret());
            }
            if cluster.byte_range.contains(&byte_offset) {
                return Err(ShapedGeometryError::InsideCluster {
                    byte_offset,
                    cluster_range: cluster.byte_range,
                });
            }
        }
        // Shaping validates complete source coverage before committing a line.
        Err(ShapedGeometryError::InvalidByteOffset { byte_offset })
    }
    /// Validates a logical UTF-8 range and borrows its whole clusters in visual
    /// order. Both endpoints must satisfy `caret_at`; reversed ranges are errors.
    /// Empty ranges yield no spans. Total validation/traversal is O(glyph count)
    /// with no extra allocations; offsets remain relative to the complete line.
    pub fn selection(
        &self,
        byte_range: Range<usize>,
    ) -> Result<ShapedSelection<'_>, ShapedGeometryError> {
        if byte_range.start > byte_range.end {
            return Err(ShapedGeometryError::ReversedRange);
        }
        self.caret_at(byte_range.start)?;
        self.caret_at(byte_range.end)?;
        Ok(ShapedSelection {
            clusters: self.clusters(),
            byte_range,
        })
    }
}
