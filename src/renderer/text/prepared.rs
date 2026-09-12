use super::*;
use crate::ShapedLineError;

/// Rejected prepared-text provenance/limits or an underlying atlas/run operation.
/// Existing UTF-8 APIs retain their original [`TextError`] return type.
#[derive(Debug)]
#[non_exhaustive]
pub enum PreparedTextError {
    /// The immutable CPU line does not match this font/style or requested limits.
    Shaped(ShapedLineError),
    /// Rasterization, atlas capacity, run storage or GPU resource validation failed.
    Text(TextError),
}

impl fmt::Display for PreparedTextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shaped(error) => write!(formatter, "prepared text: {error}"),
            Self::Text(error) => write!(formatter, "prepared text: {error}"),
        }
    }
}

impl Error for PreparedTextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Shaped(error) => Some(error),
            Self::Text(error) => Some(error),
        }
    }
}

impl From<TextError> for PreparedTextError {
    fn from(error: TextError) -> Self {
        Self::Text(error)
    }
}

impl From<ShapedLineError> for PreparedTextError {
    fn from(error: ShapedLineError) -> Self {
        Self::Shaped(error)
    }
}

impl TextAtlas2d {
    /// Prepares a validated CPU line without invoking the text shaper again.
    ///
    /// The line must originate from this exact [`FontFace`] (clones share identity)
    /// and [`TextStyle`], including DPI and requested direction. Separately loaded
    /// identical font bytes are intentionally not treated as the same identity.
    /// Text/glyph limits are checked before placement allocation or cache mutation;
    /// raster limits apply to uncached glyphs. Run capacity includes whitespace as
    /// in [`Self::prepare`]. The source line is borrowed, never consumed or modified.
    /// Failure may warm bounded atlas entries but preserves all existing runs.
    pub fn prepare_from_shaped(
        &mut self,
        renderer: &WgpuRenderer,
        shaped: &ShapedLine,
        layout_budget: TextLayoutBudget,
    ) -> Result<TextRun2d, PreparedTextError> {
        renderer
            .validate_image(&self.atlas.image)
            .map_err(TextError::from)?;
        shaped.validate_for(&self.font, &self.style, &layout_budget)?;
        Ok(self.prepare_validated(renderer, shaped, layout_budget)?)
    }

    /// Atomically updates a retained run from already shaped CPU data.
    ///
    /// Atlas/run/device identity and CPU font/style provenance are checked first,
    /// even if UTF-8 is unchanged. A compatible unchanged line reports zero work
    /// without consuming or rechecking `layout_budget`, matching [`Self::update`].
    /// For a changed line, input/output limits are checked without reshaping;
    /// raster limits apply only to new cached glyphs. An error preserves old UTF-8,
    /// metrics, instances and drawable GPU contents. A later error may leave newly
    /// cached glyphs as documented by [`Self::prepare`].
    pub fn update_from_shaped(
        &mut self,
        renderer: &WgpuRenderer,
        run: &mut TextRun2d,
        shaped: &ShapedLine,
        layout_budget: TextLayoutBudget,
    ) -> Result<TextUpdateReport, PreparedTextError> {
        self.validate_run(run)?;
        renderer
            .validate_glyph_run(&self.atlas, &run.run)
            .map_err(TextError::from)?;
        shaped.validate_provenance(&self.font, &self.style)?;
        if run.text == shaped.text() {
            return Ok(TextUpdateReport::default());
        }
        shaped.validate_for(&self.font, &self.style, &layout_budget)?;
        Ok(self.update_validated(renderer, run, shaped, layout_budget)?)
    }
}
