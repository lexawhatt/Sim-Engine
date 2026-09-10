//! Single-font horizontal shaping and bounded grayscale outline rasterization.

use ab_glyph::Font;

mod config;
mod raster;
#[cfg(test)]
mod tests;
pub use config::{
    FontBudget, FontBudgetResource, FontError, TextDirection, TextLayoutBudget, TextStyle,
};
use config::{check, reserve};
pub use raster::RasterizedGlyph;

/// One owned TTF/OpenType outline face (collection index zero).
///
/// Source bytes are retained once inside ab_glyph; rustybuzz borrows those bytes
/// while shaping. The source Vec capacity is checked before parsing. Font
/// discovery, fallback selection, variable-font axes, color/bitmap glyphs,
/// wrapping and bidi paragraph layout are outside this API.
/// Cloning shares the original bytes and parsed font without copying them.
#[derive(Debug, Clone)]
pub struct FontFace {
    font: std::sync::Arc<ab_glyph::FontVec>,
    source_capacity: usize,
    glyph_count: usize,
}

impl FontFace {
    /// Loads already-read TTF/OTF bytes without duplicating their backing storage.
    /// The caller owns filesystem/network I/O and font licensing.
    pub fn from_bytes(bytes: Vec<u8>, budget: FontBudget) -> Result<Self, FontError> {
        check(
            FontBudgetResource::FontBytes,
            bytes.capacity(),
            budget.max_font_bytes(),
        )?;
        let face =
            rustybuzz::ttf_parser::Face::parse(&bytes, 0).map_err(|_| FontError::InvalidFont)?;
        let tables = face.tables();
        if tables.glyf.is_none() && tables.cff.is_none() && tables.cff2.is_none() {
            return Err(FontError::InvalidFont);
        }
        let glyph_count = usize::from(face.number_of_glyphs());
        check(
            FontBudgetResource::FontGlyphs,
            glyph_count,
            budget.max_font_glyphs(),
        )?;
        if face.units_per_em() == 0 || face.ascender() <= face.descender() {
            return Err(FontError::InvalidFont);
        }
        let source_capacity = bytes.capacity();
        let font = ab_glyph::FontVec::try_from_vec(bytes).map_err(|_| FontError::InvalidFont)?;
        Ok(Self {
            font: std::sync::Arc::new(font),
            source_capacity,
            glyph_count,
        })
    }

    /// Original font bytes, borrowed without an additional owned copy.
    pub fn font_data(&self) -> &[u8] {
        self.font.as_slice()
    }
    /// Source Vec capacity referenced by this handle; clones share it.
    /// Dependency-parsed metadata is excluded.
    pub const fn allocation_bytes(&self) -> usize {
        self.source_capacity
    }
    /// Number of font-local glyph identifiers, including the missing-glyph entry.
    pub const fn glyph_count(&self) -> usize {
        self.glyph_count
    }

    /// Shapes a horizontal single-script directional run at baseline `(0, 0)`.
    ///
    /// Positions/advances use logical pixels, with positive y downward. Kerning,
    /// ligatures and combining-mark offsets come from rustybuzz/OpenType tables.
    /// Auto infers one direction/script, not a mixed-direction bidi paragraph.
    /// Newlines, controls and missing glyphs return errors; spaces retain advance.
    /// DPI changes raster resolution, not logical layout or kerning decisions.
    pub fn shape_line(
        &self,
        text: &str,
        style: &TextStyle,
        budget: &TextLayoutBudget,
    ) -> Result<ShapedLine, FontError> {
        check(
            FontBudgetResource::TextBytes,
            text.len(),
            budget.max_text_bytes().min(u32::MAX as usize),
        )?;
        for (byte_index, character) in text.char_indices() {
            if character.is_control() || matches!(character, '\u{2028}' | '\u{2029}') {
                return Err(FontError::UnsupportedText { byte_index });
            }
        }
        let face =
            rustybuzz::Face::from_slice(self.font.as_slice(), 0).ok_or(FontError::InvalidFont)?;
        let factor = f64::from(style.logical_em_size().get()) / f64::from(face.units_per_em());
        let ascent = represent(f64::from(face.ascender()) * factor)?;
        let descent = represent(f64::from(face.descender()) * factor)?;
        let line_height = represent(
            (f64::from(face.ascender()) - f64::from(face.descender()) + f64::from(face.line_gap()))
                * factor,
        )?;
        if line_height <= 0.0 {
            return Err(FontError::InvalidFont);
        }
        let mut input = rustybuzz::UnicodeBuffer::new();
        input.push_str(text);
        match style.direction() {
            TextDirection::Auto => {}
            TextDirection::Ltr => input.set_direction(rustybuzz::Direction::LeftToRight),
            TextDirection::Rtl => input.set_direction(rustybuzz::Direction::RightToLeft),
        }
        input.guess_segment_properties();
        let output = rustybuzz::shape(&face, &[], input);
        check(
            FontBudgetResource::ShapedGlyphs,
            output.len(),
            budget.max_shaped_glyphs(),
        )?;
        let mut glyphs = Vec::new();
        reserve(&mut glyphs, output.len())?;
        let mut pen_x = 0.0f64;
        let mut pen_y = 0.0f64;
        for (information, position) in output.glyph_infos().iter().zip(output.glyph_positions()) {
            if information.glyph_id == 0 {
                return Err(FontError::MissingGlyph {
                    byte_index: information.cluster as usize,
                });
            }
            let glyph_id =
                u16::try_from(information.glyph_id).map_err(|_| FontError::InvalidGeometry)?;
            glyphs.push(ShapedGlyph {
                glyph_id,
                logical_x: represent((pen_x + f64::from(position.x_offset)) * factor)?,
                logical_y: represent(-(pen_y + f64::from(position.y_offset)) * factor)?,
                cluster: information.cluster as usize,
            });
            pen_x += f64::from(position.x_advance);
            pen_y += f64::from(position.y_advance);
        }
        Ok(ShapedLine {
            glyphs,
            advance: represent(pen_x * factor)?,
            ascent,
            descent,
            line_height,
        })
    }

    /// Rasterizes one glyph at physical em size and baseline `(0, 0)`.
    /// Outlines are preflighted without allocating curves; pixel bounds and
    /// command counts are checked before bounded curve and coverage allocations.
    /// No hinting or subpixel-position variants are applied. Empty outlines
    /// (for example spaces) return an empty bitmap, not a missing-glyph error.
    pub fn rasterize_glyph(
        &self,
        glyph_id: u16,
        style: &TextStyle,
        budget: &TextLayoutBudget,
    ) -> Result<RasterizedGlyph, FontError> {
        raster::rasterize(self, glyph_id, style, budget)
    }
}

/// A shaped glyph's font-local identity and baseline-relative logical placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapedGlyph {
    glyph_id: u16,
    logical_x: f32,
    logical_y: f32,
    cluster: usize,
}

impl ShapedGlyph {
    /// Identifier in the FontFace that produced this line.
    pub const fn glyph_id(self) -> u16 {
        self.glyph_id
    }
    /// Baseline-relative horizontal origin, in logical pixels.
    pub const fn logical_x(self) -> f32 {
        self.logical_x
    }
    /// Baseline-relative vertical origin, in logical pixels, positive downward.
    pub const fn logical_y(self) -> f32 {
        self.logical_y
    }
    /// Original UTF-8 byte offset associated with this shaped cluster.
    pub const fn cluster(self) -> usize {
        self.cluster
    }
}

/// Immutable shaped placement; rasterization and GPU ownership are separate.
#[derive(Debug)]
pub struct ShapedLine {
    glyphs: Vec<ShapedGlyph>,
    advance: f32,
    ascent: f32,
    descent: f32,
    line_height: f32,
}

impl ShapedLine {
    /// Shaper output order. Drawing this sequence at its offsets preserves marks.
    pub fn glyphs(&self) -> &[ShapedGlyph] {
        &self.glyphs
    }
    /// Horizontal pen advance in logical pixels, including whitespace.
    pub const fn advance(&self) -> f32 {
        self.advance
    }
    /// Signed font ascender in logical pixels; normally positive above baseline.
    pub const fn ascent(&self) -> f32 {
        self.ascent
    }
    /// Signed font descender in logical pixels; normally negative below baseline.
    pub const fn descent(&self) -> f32 {
        self.descent
    }
    /// Font-recommended line spacing in logical pixels, including line gap.
    pub const fn line_height(&self) -> f32 {
        self.line_height
    }
    /// Library-owned glyph metadata Vec capacity in bytes.
    pub fn allocation_bytes(&self) -> usize {
        self.glyphs
            .capacity()
            .saturating_mul(std::mem::size_of::<ShapedGlyph>())
    }
}

fn represent(value: f64) -> Result<f32, FontError> {
    let result = value as f32;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(FontError::InvalidGeometry)
    }
}
