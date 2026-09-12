use super::*;

/// A prepared line cannot be reused with the requested font, style or limits.
/// Kept separate from [`FontError`] so existing exhaustive matches remain valid.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShapedLineError {
    /// The line came from a separately loaded font, even if its bytes match.
    FontMismatch,
    /// Logical em size, raster DPI or requested direction differs.
    StyleMismatch,
    /// Input, output, allocation or geometry validation failed.
    Font(FontError),
}

impl std::fmt::Display for ShapedLineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FontMismatch => formatter.write_str("prepared text belongs to another font face"),
            Self::StyleMismatch => formatter.write_str("prepared text uses another text style"),
            Self::Font(error) => write!(formatter, "prepared text: {error}"),
        }
    }
}

impl std::error::Error for ShapedLineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Font(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FontError> for ShapedLineError {
    fn from(error: FontError) -> Self {
        Self::Font(error)
    }
}

#[derive(PartialEq)]
struct PlanKey {
    script: rustybuzz::Script,
    direction: rustybuzz::Direction,
    language: Option<rustybuzz::Language>,
}

/// Caller-owned reusable CPU shaping state for one borrowed font and fixed style.
/// Available with `fonts`, without a renderer or GPU dependency.
///
/// The parsed rustybuzz face, one exact-property shaping plan and one buffer are
/// reused. An inferred script/direction change replaces that single plan rather
/// than growing a cache. Language is unspecified and OpenType features use the
/// same defaults as [`FontFace::shape_line`]; no cross-font/global cache exists.
///
/// [`Self::update_line`] reuses two bounded UTF-8/glyph allocations transactionally.
/// Input and output limits remain mandatory. Dependency-owned face/plan/buffer
/// allocations are excluded from [`Self::allocation_bytes`]: rustybuzz does not
/// expose their capacities or fallible allocation. These are trusted-font work
/// limits, not an exact process-memory or hostile-font sandbox. Dependency buffer
/// storage is dropped after a shaping failure; [`Self::clear_scratch`] releases
/// successful-run scratch and the one cached plan on demand.
///
/// ```no_run
/// use sim_engine::{FontBudget, FontFace, LogicalPixels, PhysicalPerLogical,
///     TextLayoutBudget, TextShapingSession, TextStyle};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let font = FontFace::from_bytes(std::fs::read("assets/Label.ttf")?, FontBudget::default())?;
/// let style = TextStyle::new(LogicalPixels::new(24.0)?, PhysicalPerLogical::new(1.0)?)?;
/// let mut session = TextShapingSession::new(&font, style, TextLayoutBudget::default())?;
/// let mut line = session.shape_line("Count: 100")?;
/// session.update_line(&mut line, "Count: 101")?;
/// // Hand `line` to TextAtlas2d::prepare_from_shaped when using the `text` feature.
/// assert_eq!(line.text(), "Count: 101");
/// # Ok(())
/// # }
/// ```
pub struct TextShapingSession<'font> {
    font: &'font FontFace,
    face: rustybuzz::Face<'font>,
    style: TextStyle,
    budget: TextLayoutBudget,
    input: rustybuzz::UnicodeBuffer,
    plan: Option<(PlanKey, rustybuzz::ShapePlan)>,
    glyphs: Vec<ShapedGlyph>,
    text: String,
    factor: f64,
    ascent: f32,
    descent: f32,
    line_height: f32,
}

impl<'font> TextShapingSession<'font> {
    /// Parses the borrowed font once and fixes style and work limits for this session.
    /// Does not preallocate the maximum permitted line length.
    pub fn new(
        font: &'font FontFace,
        style: TextStyle,
        budget: TextLayoutBudget,
    ) -> Result<Self, FontError> {
        let face =
            rustybuzz::Face::from_slice(font.font_data(), 0).ok_or(FontError::InvalidFont)?;
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
        Ok(Self {
            font,
            face,
            style,
            budget,
            input: rustybuzz::UnicodeBuffer::new(),
            plan: None,
            glyphs: Vec::new(),
            text: String::new(),
            factor,
            ascent,
            descent,
            line_height,
        })
    }

    /// Returns the immutable style, including requested direction and raster DPI.
    pub const fn style(&self) -> &TextStyle {
        &self.style
    }

    /// Returns fixed per-line and per-glyph work limits.
    pub const fn budget(&self) -> TextLayoutBudget {
        self.budget
    }

    /// Shapes a new owned line; later updates can reuse its allocation.
    /// Separate returned lines each own their own bounded UTF-8 and glyph vectors.
    pub fn shape_line(&mut self, text: &str) -> Result<ShapedLine, FontError> {
        let advance = self.shape_to_scratch(text)?;
        Ok(ShapedLine {
            font: self.font.clone(),
            style: self.style,
            text: std::mem::take(&mut self.text),
            glyphs: std::mem::take(&mut self.glyphs),
            advance,
            ascent: self.ascent,
            descent: self.descent,
            line_height: self.line_height,
        })
    }

    /// Replaces a compatible line atomically, returning whether its UTF-8 changed.
    ///
    /// Font/style and current line limits are checked even for unchanged text.
    /// A successful change swaps the session's scratch with the previous line's
    /// storage, allowing equal or shorter warmed lengths to avoid engine-owned
    /// allocations. The old line's capacities must also fit this session's input
    /// and output limits so swapping cannot import unbounded retained storage.
    /// Any returned error preserves the old text, glyphs, metrics and allocations.
    pub fn update_line(
        &mut self,
        line: &mut ShapedLine,
        text: &str,
    ) -> Result<bool, ShapedLineError> {
        line.validate_for(self.font, &self.style, &self.budget)?;
        check(
            FontBudgetResource::TextBytes,
            line.text.capacity(),
            self.budget.max_text_bytes(),
        )?;
        check(
            FontBudgetResource::ShapedGlyphs,
            line.glyphs.capacity(),
            self.budget.max_shaped_glyphs(),
        )?;
        if line.text == text {
            return Ok(false);
        }
        let advance = self.shape_to_scratch(text)?;
        std::mem::swap(&mut line.text, &mut self.text);
        std::mem::swap(&mut line.glyphs, &mut self.glyphs);
        line.advance = advance;
        Ok(true)
    }

    /// Engine-owned scratch capacity in bytes, excluding returned lines, shared
    /// source font and dependency-owned face/plan/buffer allocations.
    pub fn allocation_bytes(&self) -> usize {
        self.glyphs
            .capacity()
            .saturating_mul(std::mem::size_of::<ShapedGlyph>())
            .saturating_add(self.text.capacity())
    }

    /// Number of retained plans, always zero or one regardless of strings seen.
    pub fn cached_plan_count(&self) -> usize {
        usize::from(self.plan.is_some())
    }

    /// Releases scratch and cached plan while preserving the borrowed parsed face.
    /// Previously returned lines remain valid and independent of this session.
    pub fn clear_scratch(&mut self) {
        self.input = rustybuzz::UnicodeBuffer::new();
        self.plan = None;
        self.glyphs = Vec::new();
        self.text = String::new();
    }

    fn shape_to_scratch(&mut self, text: &str) -> Result<f32, FontError> {
        validate_text(text, self.budget)?;
        self.input.clear();
        self.input.push_str(text);
        match self.style.direction() {
            TextDirection::Auto => {}
            TextDirection::Ltr => self.input.set_direction(rustybuzz::Direction::LeftToRight),
            TextDirection::Rtl => self.input.set_direction(rustybuzz::Direction::RightToLeft),
        }
        self.input.guess_segment_properties();
        let key = PlanKey {
            script: self.input.script(),
            direction: self.input.direction(),
            language: self.input.language(),
        };
        if self
            .plan
            .as_ref()
            .is_none_or(|(previous, _)| *previous != key)
        {
            let plan = rustybuzz::ShapePlan::new(
                &self.face,
                key.direction,
                Some(key.script),
                key.language.as_ref(),
                &[],
            );
            self.plan = Some((key, plan));
        }
        let Some((_, plan)) = &self.plan else {
            return Err(FontError::InvalidFont);
        };
        let output = rustybuzz::shape_with_plan(&self.face, plan, std::mem::take(&mut self.input));
        let result = self.copy_output(text, &output);
        // A rejected expanded output must not remain as hidden retained scratch.
        if result.is_ok() {
            self.input = output.clear();
        }
        result
    }

    fn copy_output(
        &mut self,
        text: &str,
        output: &rustybuzz::GlyphBuffer,
    ) -> Result<f32, FontError> {
        check(
            FontBudgetResource::ShapedGlyphs,
            output.len(),
            self.budget.max_shaped_glyphs(),
        )?;
        self.glyphs.clear();
        reserve(&mut self.glyphs, output.len())?;
        let mut pen_x = 0.0f64;
        let mut pen_y = 0.0f64;
        for (information, position) in output.glyph_infos().iter().zip(output.glyph_positions()) {
            if information.glyph_id == 0 {
                return Err(FontError::MissingGlyph {
                    byte_index: information.cluster as usize,
                });
            }
            self.glyphs.push(ShapedGlyph {
                glyph_id: u16::try_from(information.glyph_id)
                    .map_err(|_| FontError::InvalidGeometry)?,
                logical_x: represent((pen_x + f64::from(position.x_offset)) * self.factor)?,
                logical_y: represent(-(pen_y + f64::from(position.y_offset)) * self.factor)?,
                cluster: information.cluster as usize,
            });
            pen_x += f64::from(position.x_advance);
            pen_y += f64::from(position.y_advance);
        }
        let advance = represent(pen_x * self.factor)?;
        self.text.clear();
        self.text
            .try_reserve_exact(text.len())
            .map_err(|_| FontError::AllocationFailed {
                requested_bytes: text.len(),
            })?;
        self.text.push_str(text);
        Ok(advance)
    }
}

fn validate_text(text: &str, budget: TextLayoutBudget) -> Result<(), FontError> {
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
    Ok(())
}
