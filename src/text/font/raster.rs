use super::*;
use ab_glyph::{Outline, OutlineCurve, Point, PxScaleFactor, point};
use rustybuzz::ttf_parser::OutlineBuilder;

/// Grayscale coverage at physical-pixel resolution, relative to baseline zero.
/// Coverage is linear alpha (0..255), not an sRGB intensity or premultiplied RGB.
#[derive(Debug)]
pub struct RasterizedGlyph {
    width: u32,
    height: u32,
    bearing_x: i32,
    bearing_y: i32,
    coverage: Vec<u8>,
}

impl RasterizedGlyph {
    /// Physical bitmap width; zero for an empty outline such as a space.
    pub const fn width(&self) -> u32 {
        self.width
    }
    /// Physical bitmap height; zero for an empty outline such as a space.
    pub const fn height(&self) -> u32 {
        self.height
    }
    /// Left bitmap edge relative to the physical baseline origin; may be negative.
    pub const fn bearing_x(&self) -> i32 {
        self.bearing_x
    }
    /// Top bitmap edge relative to the physical baseline origin, positive downward.
    pub const fn bearing_y(&self) -> i32 {
        self.bearing_y
    }
    /// Row-major linear alpha, exactly width times height bytes.
    pub fn coverage(&self) -> &[u8] {
        &self.coverage
    }
    /// Owned coverage Vec capacity in bytes.
    pub fn allocation_bytes(&self) -> usize {
        self.coverage.capacity()
    }
}

pub(super) fn rasterize(
    font: &FontFace,
    glyph_id: u16,
    style: &TextStyle,
    budget: &TextLayoutBudget,
) -> Result<RasterizedGlyph, FontError> {
    if usize::from(glyph_id) >= font.glyph_count {
        return Err(FontError::InvalidGlyph { glyph_id });
    }
    let face = rustybuzz::ttf_parser::Face::parse(font.font_data(), 0)
        .map_err(|_| FontError::InvalidFont)?;
    let factor =
        style.physical_em_size() / font.font.units_per_em().ok_or(FontError::InvalidFont)?;
    if !factor.is_normal() || factor <= 0.0 {
        return Err(FontError::InvalidScale);
    }
    let id = rustybuzz::ttf_parser::GlyphId(glyph_id);
    if face.is_color_glyph(id) || face.glyph_svg_image(id).is_some() {
        return Err(FontError::UnsupportedGlyph { glyph_id });
    }
    let mut counter = OutlineSink::new(factor, budget.max_outline_segments());
    let bounds = face.outline_glyph(id, &mut counter);
    counter.finish();
    check(
        FontBudgetResource::OutlineSegments,
        counter.commands,
        budget.max_outline_segments(),
    )?;
    if !counter.valid {
        return Err(FontError::InvalidGeometry);
    }
    let Some(bounds) = bounds else {
        if face.glyph_raster_image(id, u16::MAX).is_some() {
            return Err(FontError::UnsupportedGlyph { glyph_id });
        }
        if counter.commands != 0 {
            return Err(FontError::InvalidFont);
        }
        return Ok(RasterizedGlyph {
            width: 0,
            height: 0,
            bearing_x: 0,
            bearing_y: 0,
            coverage: Vec::new(),
        });
    };
    let scale_factor = PxScaleFactor {
        horizontal: factor,
        vertical: factor,
    };
    let mut outline = Outline {
        bounds: ab_glyph::Rect {
            min: point(f32::from(bounds.x_min), f32::from(bounds.y_max)),
            max: point(f32::from(bounds.x_max), f32::from(bounds.y_min)),
        },
        curves: Vec::new(),
    };
    let pixels = outline.px_bounds(scale_factor, point(0.0, 0.0));
    let left = pixel_coordinate(pixels.min.x)?;
    let top = pixel_coordinate(pixels.min.y)?;
    let right = pixel_coordinate(pixels.max.x)?;
    let bottom = pixel_coordinate(pixels.max.y)?;
    let width = u32::try_from(i64::from(right) - i64::from(left))
        .map_err(|_| FontError::InvalidGeometry)?;
    let height = u32::try_from(i64::from(bottom) - i64::from(top))
        .map_err(|_| FontError::InvalidGeometry)?;
    let count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(FontError::InvalidGeometry)?;
    check(
        FontBudgetResource::GlyphPixels,
        count,
        budget.max_glyph_pixels(),
    )?;
    // ab_glyph_rasterizer uses count+4 f32 cells. Check its arithmetic as well
    // as the public u8 payload before handing it the dimensions.
    count
        .checked_add(4)
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<f32>()))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or(FontError::InvalidGeometry)?;
    if width == 0 || height == 0 {
        return Ok(RasterizedGlyph {
            width: 0,
            height: 0,
            bearing_x: left,
            bearing_y: top,
            coverage: Vec::new(),
        });
    }
    let mut sink = OutlineSink::new(factor, counter.commands);
    reserve(&mut sink.curves, counter.commands)?;
    sink.collect = true;
    let second_bounds = face.outline_glyph(id, &mut sink);
    sink.finish();
    if !sink.valid || sink.commands != counter.commands || second_bounds != Some(bounds) {
        return Err(FontError::InvalidFont);
    }
    outline.curves = sink.curves;
    let mut coverage = Vec::new();
    reserve(&mut coverage, count)?;
    coverage.resize(count, 0);
    let glyph = ab_glyph::Glyph {
        id: ab_glyph::GlyphId(glyph_id),
        scale: ab_glyph::PxScale::from(style.physical_em_size()),
        position: point(0.0, 0.0),
    };
    // OutlinedGlyph's explicit scale factor is font-units -> pixels. PxScale's
    // font-height convention must not be mistaken for the API's em convention.
    let outlined = ab_glyph::OutlinedGlyph::new(glyph, outline, scale_factor);
    let mut valid = true;
    outlined.draw(|x, y, alpha| {
        if !alpha.is_finite() || x >= width || y >= height {
            valid = false;
            return;
        }
        coverage[y as usize * width as usize + x as usize] =
            (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
    });
    if !valid {
        return Err(FontError::InvalidGeometry);
    }
    Ok(RasterizedGlyph {
        width,
        height,
        bearing_x: left,
        bearing_y: top,
        coverage,
    })
}

fn pixel_coordinate(value: f32) -> Result<i32, FontError> {
    if !value.is_finite()
        || f64::from(value) < f64::from(i32::MIN)
        || f64::from(value) > f64::from(i32::MAX)
    {
        Err(FontError::InvalidGeometry)
    } else {
        Ok(value as i32)
    }
}

/// The first traversal counts commands and checks coordinates without allocating
/// curves. The second writes only into reserved storage. Dependency traversal
/// itself cannot be interrupted through OutlineBuilder and is not a CPU sandbox.
struct OutlineSink {
    factor: f32,
    limit: usize,
    commands: usize,
    curves: Vec<OutlineCurve>,
    collect: bool,
    valid: bool,
    last: Point,
    start: Option<Point>,
}

impl OutlineSink {
    fn new(factor: f32, limit: usize) -> Self {
        Self {
            factor,
            limit,
            commands: 0,
            curves: Vec::new(),
            collect: false,
            valid: true,
            last: point(0.0, 0.0),
            start: None,
        }
    }
    fn command(&mut self) {
        self.commands = self.commands.saturating_add(1);
    }
    fn position(&mut self, x: f32, y: f32) -> Point {
        self.valid &= [x, y].into_iter().all(|value| {
            value.is_finite()
                && (f64::from(value) * f64::from(self.factor)).abs() < f64::from(i32::MAX)
        });
        point(x, y)
    }
    fn curve(&mut self, curve: OutlineCurve) {
        if self.collect {
            if self.commands <= self.limit && self.curves.len() < self.curves.capacity() {
                self.curves.push(curve);
            } else {
                self.valid = false;
            }
        }
    }
    fn finish(&mut self) {
        if self.start.is_some() {
            self.close();
        }
    }
}

impl OutlineBuilder for OutlineSink {
    fn move_to(&mut self, x: f32, y: f32) {
        self.command();
        self.last = self.position(x, y);
        self.start = Some(self.last);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.command();
        let end = self.position(x, y);
        self.curve(OutlineCurve::Line(self.last, end));
        self.last = end;
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.command();
        let control = self.position(x1, y1);
        let end = self.position(x, y);
        self.curve(OutlineCurve::Quad(self.last, control, end));
        self.last = end;
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.command();
        let control1 = self.position(x1, y1);
        let control2 = self.position(x2, y2);
        let end = self.position(x, y);
        self.curve(OutlineCurve::Cubic(self.last, control1, control2, end));
        self.last = end;
    }
    fn close(&mut self) {
        if let Some(start) = self.start.take() {
            self.command();
            self.curve(OutlineCurve::Line(self.last, start));
        }
    }
}
