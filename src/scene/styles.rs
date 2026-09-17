//! Bounded stroke, paint, gradient and shadow values for ordinary scenes.

use std::{error::Error, fmt};

use crate::{Color, Interpolate, LogicalPixels, LogicalScreenVector, Vec2, WorldLength};

use super::{MAX_STROKE_DASH_SUBSEGMENTS, SceneError, ScenePrimitive};

const MAX_DASH_ELEMENTS: usize = 8;
const MAX_MITER_LIMIT: f32 = 1_000.0;

/// Coordinate space used by a line or polyline stroke width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeWidthMode2d {
    /// Width remains constant in logical screen pixels while the camera zooms.
    LogicalPixels,
    /// Width is measured in the same world units as the path coordinates.
    WorldUnits,
}

/// Presentation used at an open line or dash endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeCap2d {
    /// Stop exactly at the endpoint.
    Butt,
    /// Extend by half the stroke width beyond the endpoint.
    Square,
    /// Add a semicircular endpoint.
    Round,
}

/// Presentation used where two visible polyline segments meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeJoin2d {
    /// Connect the two segment corners directly.
    Bevel,
    /// Fill the corner with a circular join.
    Round,
    /// Extend corner edges to their intersection, bounded by the style's
    /// miter limit and falling back to a bevel beyond it.
    Miter,
}

/// Fixed-size, allocation-free dash pattern in source path-coordinate units.
///
/// Values alternate visible and hidden lengths, beginning with a visible
/// length. The element count must be even and at most eight. `phase` advances
/// into the repeating pattern in the same path units. `max_subsegments` is a
/// hard limit on visible pieces generated for one command.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeDashPattern2d {
    lengths: [f32; MAX_DASH_ELEMENTS],
    length_count: u8,
    pub(super) phase: f32,
    pub(super) max_subsegments: usize,
}

impl StrokeDashPattern2d {
    /// Builds a bounded alternating visible/hidden dash pattern.
    pub fn new(
        lengths: &[f32],
        phase: f32,
        max_subsegments: usize,
    ) -> Result<Self, StrokeStyleError> {
        if lengths.len() < 2
            || lengths.len() > MAX_DASH_ELEMENTS
            || !lengths.len().is_multiple_of(2)
        {
            return Err(StrokeStyleError::InvalidDashElementCount);
        }
        if lengths
            .iter()
            .any(|length| !length.is_finite() || *length <= 0.0)
        {
            return Err(StrokeStyleError::InvalidDashLength);
        }
        if !phase.is_finite() {
            return Err(StrokeStyleError::InvalidDashPhase);
        }
        if max_subsegments == 0 || max_subsegments > MAX_STROKE_DASH_SUBSEGMENTS {
            return Err(StrokeStyleError::InvalidDashExpansionLimit);
        }
        let mut stored = [0.0; MAX_DASH_ELEMENTS];
        stored[..lengths.len()].copy_from_slice(lengths);
        Ok(Self {
            lengths: stored,
            length_count: lengths.len() as u8,
            phase,
            max_subsegments,
        })
    }

    /// Returns the alternating visible/hidden lengths in path units.
    pub fn lengths(&self) -> &[f32] {
        &self.lengths[..usize::from(self.length_count)]
    }

    /// Returns the repeating-pattern phase in path units.
    pub const fn phase(self) -> f32 {
        self.phase
    }

    /// Returns the hard visible-subsegment expansion limit.
    pub const fn max_subsegments(self) -> usize {
        self.max_subsegments
    }
}

/// Placement of a logical-pixel arrow relative to its mathematical path endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeMarkerAnchor2d {
    /// Put the marker base at the endpoint and extend its tip outward.
    BaseAtEndpoint,
    /// Put the marker tip exactly at the endpoint and shorten the shaft inward.
    ///
    /// Supported on undashed two-point paths only. Both shaft ends use butt
    /// boundaries. Each inward length is bounded to one quarter of the projected
    /// segment's larger absolute X/Y span, preserving a non-inverted shaft even
    /// with two markers. Marker width stays in logical pixels. World-width shaft
    /// boundaries are perpendicular to the projected tangent, keeping their
    /// projected perpendicular thickness without shear into the arrowhead.
    TipAtEndpoint,
}

/// Reusable arrow marker definition for a line endpoint.
///
/// Marker dimensions are logical pixels even when the line body uses a world
/// width, keeping scientific annotations readable under camera zoom. By default,
/// a marked endpoint ignores the ordinary cap: the body ends with a butt boundary at
/// the path endpoint, which is also the marker base. The filled triangle grows
/// outward from the path, so markers remain interior-disjoint from arbitrarily
/// short, dashed, or camera-scaled terminal segments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeMarker2d {
    length: LogicalPixels,
    width: LogicalPixels,
    anchor: StrokeMarkerAnchor2d,
}

impl StrokeMarker2d {
    /// Builds a filled triangular arrow marker.
    pub const fn arrow(length: LogicalPixels, width: LogicalPixels) -> Self {
        Self {
            length,
            width,
            anchor: StrokeMarkerAnchor2d::BaseAtEndpoint,
        }
    }

    /// Selects the marker anchor. Tip anchoring uses the bounded exact-vector
    /// contract documented by [`StrokeMarkerAnchor2d::TipAtEndpoint`].
    pub const fn with_anchor(mut self, anchor: StrokeMarkerAnchor2d) -> Self {
        self.anchor = anchor;
        self
    }

    /// Returns how the marker is anchored to the mathematical endpoint.
    pub const fn anchor(self) -> StrokeMarkerAnchor2d {
        self.anchor
    }

    /// Returns marker length along the endpoint tangent.
    pub const fn length(self) -> LogicalPixels {
        self.length
    }

    /// Returns full marker width perpendicular to the endpoint tangent.
    pub const fn width(self) -> LogicalPixels {
        self.width
    }
}

/// Complete bounded styling for open 2D lines and polylines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeStyle2d {
    pub(super) stroke: Stroke,
    pub(super) width_mode: StrokeWidthMode2d,
    pub(super) cap: StrokeCap2d,
    pub(super) join: StrokeJoin2d,
    pub(super) miter_limit: f32,
    pub(super) dash: Option<StrokeDashPattern2d>,
    pub(super) start_marker: Option<StrokeMarker2d>,
    pub(super) end_marker: Option<StrokeMarker2d>,
}

impl StrokeStyle2d {
    /// Builds the legacy logical-pixel round-cap joined-strip style.
    pub const fn new(width: f32, color: Color) -> Self {
        Self {
            stroke: Stroke::new(width, color),
            width_mode: StrokeWidthMode2d::LogicalPixels,
            cap: StrokeCap2d::Round,
            join: StrokeJoin2d::Miter,
            miter_limit: MAX_MITER_LIMIT,
            dash: None,
            start_marker: None,
            end_marker: None,
        }
    }

    /// Builds a validated logical-pixel-width style.
    pub const fn logical(width: LogicalPixels, color: Color) -> Self {
        Self::new(width.get(), color)
    }

    /// Builds a validated world-unit-width style.
    pub const fn world(width: WorldLength, color: Color) -> Self {
        let mut style = Self::new(width.get(), color);
        style.width_mode = StrokeWidthMode2d::WorldUnits;
        style
    }

    /// Selects the open endpoint presentation.
    pub const fn with_cap(mut self, cap: StrokeCap2d) -> Self {
        self.cap = cap;
        self
    }

    /// Selects the connected-segment presentation.
    pub const fn with_join(mut self, join: StrokeJoin2d) -> Self {
        self.join = join;
        self
    }

    /// Sets the miter-length multiple in `1.0..=1000.0`.
    pub fn with_miter_limit(mut self, limit: f32) -> Result<Self, StrokeStyleError> {
        if !limit.is_finite() || !(1.0..=MAX_MITER_LIMIT).contains(&limit) {
            return Err(StrokeStyleError::InvalidMiterLimit);
        }
        self.miter_limit = limit;
        Ok(self)
    }

    /// Applies an allocation-free bounded dash pattern.
    pub const fn with_dash_pattern(mut self, dash: StrokeDashPattern2d) -> Self {
        self.dash = Some(dash);
        self
    }

    /// Adds a triangular marker at the first path point using its selected anchor.
    pub const fn with_start_marker(mut self, marker: StrokeMarker2d) -> Self {
        self.start_marker = Some(marker);
        self
    }

    /// Adds a triangular marker at the last path point using its selected anchor.
    pub const fn with_end_marker(mut self, marker: StrokeMarker2d) -> Self {
        self.end_marker = Some(marker);
        self
    }

    /// Returns color and scalar width.
    pub const fn stroke(self) -> Stroke {
        self.stroke
    }

    /// Returns whether width is logical-screen or world-space.
    pub const fn width_mode(self) -> StrokeWidthMode2d {
        self.width_mode
    }

    /// Returns endpoint presentation.
    pub const fn cap(self) -> StrokeCap2d {
        self.cap
    }

    /// Returns connected-segment presentation.
    pub const fn join(self) -> StrokeJoin2d {
        self.join
    }

    /// Returns the bounded miter multiple.
    pub const fn miter_limit(self) -> f32 {
        self.miter_limit
    }

    /// Returns the optional bounded dash pattern.
    pub const fn dash_pattern(self) -> Option<StrokeDashPattern2d> {
        self.dash
    }

    /// Returns the optional first-endpoint marker.
    pub const fn start_marker(self) -> Option<StrokeMarker2d> {
        self.start_marker
    }

    /// Returns the optional last-endpoint marker.
    pub const fn end_marker(self) -> Option<StrokeMarker2d> {
        self.end_marker
    }

    pub(crate) fn has_tip_marker(self) -> bool {
        [self.start_marker, self.end_marker]
            .into_iter()
            .flatten()
            .any(|marker| marker.anchor == StrokeMarkerAnchor2d::TipAtEndpoint)
    }

    pub(super) fn is_valid(self) -> bool {
        self.stroke.is_valid()
            && self.miter_limit.is_finite()
            && (1.0..=MAX_MITER_LIMIT).contains(&self.miter_limit)
    }
}

/// Invalid bounded line-style configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeStyleError {
    /// Dash patterns require an even count from two through eight.
    InvalidDashElementCount,
    /// Every dash and gap length must be positive and finite.
    InvalidDashLength,
    /// Dash phase must be finite.
    InvalidDashPhase,
    /// Dash expansion limit must be in `1..=1_000_000`.
    InvalidDashExpansionLimit,
    /// Miter limit must be finite and in `1.0..=1000.0`.
    InvalidMiterLimit,
}

impl fmt::Display for StrokeStyleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid bounded 2D stroke style: {self:?}")
    }
}

impl Error for StrokeStyleError {}

/// Fill, stroke, and shadow styling for solid primitives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeStyle {
    pub(super) fill: Option<Fill>,
    pub(super) stroke: Option<Stroke>,
    pub(super) shadow: Option<Shadow>,
}

impl ShapeStyle {
    /// Builds explicit fill, stroke, and shadow choices.
    pub const fn new(fill: Option<Fill>, stroke: Option<Stroke>, shadow: Option<Shadow>) -> Self {
        Self {
            fill,
            stroke,
            shadow,
        }
    }

    /// Returns fill paint, if any.
    pub const fn fill(self) -> Option<Fill> {
        self.fill
    }
    /// Returns outline stroke, if any.
    pub const fn stroke(self) -> Option<Stroke> {
        self.stroke
    }
    /// Returns shadow styling, if any.
    pub const fn shadow(self) -> Option<Shadow> {
        self.shadow
    }
    /// Creates a style with fill color only.
    pub fn filled(color: Color) -> Self {
        Self::filled_with(Fill::Solid(color))
    }

    /// Creates a style with fill paint only.
    pub fn filled_with(fill: Fill) -> Self {
        Self {
            fill: Some(fill),
            stroke: None,
            shadow: None,
        }
    }

    /// Creates a style with stroke only.
    ///
    /// `width` is in logical screen pixels.
    pub fn stroked(width: f32, color: Color) -> Self {
        Self {
            fill: None,
            stroke: Some(Stroke { width, color }),
            shadow: None,
        }
    }

    /// Creates a style with both fill and stroke.
    ///
    /// `stroke_width` is in logical screen pixels.
    pub fn fill_stroke(fill: Color, stroke_width: f32, stroke: Color) -> Self {
        Self::fill_stroke_with(Fill::Solid(fill), stroke_width, stroke)
    }

    /// Creates a style with both fill paint and stroke.
    ///
    /// `stroke_width` is in logical screen pixels.
    pub fn fill_stroke_with(fill: Fill, stroke_width: f32, stroke: Color) -> Self {
        Self {
            fill: Some(fill),
            stroke: Some(Stroke {
                width: stroke_width,
                color: stroke,
            }),
            shadow: None,
        }
    }

    /// Adds a shadow to the style.
    pub fn with_shadow(mut self, shadow: Shadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    pub(super) fn validate(self, primitive: ScenePrimitive) -> Result<(), SceneError> {
        if self.fill.is_none() && self.stroke.is_none() && self.shadow.is_none() {
            return Err(SceneError::MissingStyle(primitive));
        }
        if self.fill.is_some_and(|fill| !fill.is_valid()) {
            return Err(SceneError::InvalidFill(primitive));
        }
        if self.stroke.is_some_and(|stroke| !stroke.is_valid()) {
            return Err(SceneError::InvalidStroke(primitive));
        }
        if self.shadow.is_some_and(|shadow| !shadow.is_valid()) {
            return Err(SceneError::InvalidShadow(primitive));
        }
        Ok(())
    }
}

/// Paint used to fill a shape.
///
/// Gradient coordinates are in world units. Renderers may approximate gradients
/// per vertex until a more advanced shader path exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fill {
    /// Single color over the whole shape.
    Solid(Color),
    /// Color interpolation along a world-space axis.
    LinearGradient(LinearGradient),
    /// Color interpolation outward from a world-space center.
    RadialGradient(RadialGradient),
}

impl Fill {
    /// Returns the color at a world-space point.
    ///
    /// Degenerate gradients fall back to their end color rather than producing
    /// NaN.
    pub fn color_at(self, world: Vec2) -> Color {
        self.color_at_with_offset(world, Vec2::ZERO)
    }

    pub(crate) fn color_at_with_offset(self, world_base: Vec2, world_offset: Vec2) -> Color {
        match self {
            Self::Solid(color) => color,
            Self::LinearGradient(gradient) => {
                gradient.color_at_with_offset(world_base, world_offset)
            }
            Self::RadialGradient(gradient) => {
                gradient.color_at_with_offset(world_base, world_offset)
            }
        }
    }

    fn is_valid(self) -> bool {
        match self {
            Self::Solid(color) => color.is_normalized(),
            Self::LinearGradient(gradient) => gradient.is_valid(),
            Self::RadialGradient(gradient) => gradient.is_valid(),
        }
    }
}

impl From<Color> for Fill {
    fn from(color: Color) -> Self {
        Self::Solid(color)
    }
}

/// Linear color gradient in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearGradient {
    start: Vec2,
    end: Vec2,
    start_color: Color,
    end_color: Color,
}

impl LinearGradient {
    /// Builds a linear gradient between two world-space points.
    pub fn new(start: Vec2, end: Vec2, start_color: Color, end_color: Color) -> Self {
        Self {
            start,
            end,
            start_color,
            end_color,
        }
    }

    /// Returns the world-space gradient start.
    pub const fn start(self) -> Vec2 {
        self.start
    }
    /// Returns the world-space gradient end.
    pub const fn end(self) -> Vec2 {
        self.end
    }
    /// Returns the color at the start.
    pub const fn start_color(self) -> Color {
        self.start_color
    }
    /// Returns the color at the end.
    pub const fn end_color(self) -> Color {
        self.end_color
    }

    /// Samples the gradient at a world-space point.
    pub fn color_at(self, world: Vec2) -> Color {
        self.color_at_with_offset(world, Vec2::ZERO)
    }

    fn color_at_with_offset(self, world_base: Vec2, world_offset: Vec2) -> Color {
        if !self.is_valid() || !world_base.is_finite() || !world_offset.is_finite() {
            return self.end_color.clamp();
        }

        let axis_x = f64::from(self.end.x) - f64::from(self.start.x);
        let axis_y = f64::from(self.end.y) - f64::from(self.start.y);
        let relative_x =
            f64::from(world_base.x) - f64::from(self.start.x) + f64::from(world_offset.x);
        let relative_y =
            f64::from(world_base.y) - f64::from(self.start.y) + f64::from(world_offset.y);
        let axis_length_squared = axis_x * axis_x + axis_y * axis_y;
        let amount = ((relative_x * axis_x + relative_y * axis_y) / axis_length_squared)
            .clamp(0.0, 1.0) as f32;
        self.start_color.interpolate(self.end_color, amount)
    }

    fn is_valid(self) -> bool {
        self.start.is_finite()
            && self.end.is_finite()
            && self.start_color.is_normalized()
            && self.end_color.is_normalized()
            && self.start != self.end
    }
}

/// Radial color gradient in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialGradient {
    /// Center point in world coordinates.
    center: Vec2,
    /// Radius where `inner_color` is reached, in world units.
    inner_radius: f32,
    /// Radius where `outer_color` is reached, in world units.
    outer_radius: f32,
    /// Color at or inside `inner_radius`.
    inner_color: Color,
    /// Color at or outside `outer_radius`.
    outer_color: Color,
}

impl RadialGradient {
    /// Builds a radial gradient in world coordinates.
    ///
    /// Reversed finite radii are normalized together with their colors so the
    /// color attached to each radius remains stable.
    pub fn new(
        center: Vec2,
        inner_radius: f32,
        outer_radius: f32,
        inner_color: Color,
        outer_color: Color,
    ) -> Self {
        if inner_radius.is_finite() && outer_radius.is_finite() && inner_radius > outer_radius {
            Self {
                center,
                inner_radius: outer_radius,
                outer_radius: inner_radius,
                inner_color: outer_color,
                outer_color: inner_color,
            }
        } else {
            Self {
                center,
                inner_radius,
                outer_radius,
                inner_color,
                outer_color,
            }
        }
    }

    /// Returns the world-space gradient center.
    pub const fn center(self) -> Vec2 {
        self.center
    }

    /// Returns the inner radius in world units.
    pub const fn inner_radius(self) -> f32 {
        self.inner_radius
    }

    /// Returns the outer radius in world units.
    pub const fn outer_radius(self) -> f32 {
        self.outer_radius
    }

    /// Returns the inner linear color.
    pub const fn inner_color(self) -> Color {
        self.inner_color
    }

    /// Returns the outer linear color.
    pub const fn outer_color(self) -> Color {
        self.outer_color
    }

    /// Samples the gradient at a world-space point.
    pub fn color_at(self, world: Vec2) -> Color {
        self.color_at_with_offset(world, Vec2::ZERO)
    }

    fn color_at_with_offset(self, world_base: Vec2, world_offset: Vec2) -> Color {
        if !self.is_valid() || !world_base.is_finite() || !world_offset.is_finite() {
            return self.outer_color.clamp();
        }

        let radius_range = self.outer_radius - self.inner_radius;
        if radius_range == 0.0 {
            return self.outer_color;
        }

        let horizontal =
            f64::from(world_base.x) - f64::from(self.center.x) + f64::from(world_offset.x);
        let vertical =
            f64::from(world_base.y) - f64::from(self.center.y) + f64::from(world_offset.y);
        let distance = horizontal.hypot(vertical);
        let amount = ((distance - f64::from(self.inner_radius)) / f64::from(radius_range))
            .clamp(0.0, 1.0) as f32;
        self.inner_color.interpolate(self.outer_color, amount)
    }

    fn is_valid(self) -> bool {
        self.center.is_finite()
            && self.inner_radius.is_finite()
            && self.outer_radius.is_finite()
            && self.inner_radius >= 0.0
            && self.outer_radius >= self.inner_radius
            && self.inner_color.is_normalized()
            && self.outer_color.is_normalized()
    }
}

/// Stroke color and width for lines and shape outlines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stroke {
    pub(super) width: f32,
    color: Color,
}

impl Stroke {
    /// Builds a stroke used by shape outlines and legacy logical-width lines.
    ///
    /// [`StrokeStyle2d::world`] reuses the same scalar/color value while making
    /// the width's coordinate space explicit on the containing style.
    pub const fn new(width: f32, color: Color) -> Self {
        Self { width, color }
    }
    /// Returns scalar width in the coordinate space selected by its owner.
    pub const fn width(self) -> f32 {
        self.width
    }
    /// Returns the stroke color.
    pub const fn color(self) -> Color {
        self.color
    }
    fn is_valid(self) -> bool {
        self.width.is_finite() && self.width > 0.0 && self.color.is_normalized()
    }
}

/// Simple screen-space shadow used by the first renderer slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shadow {
    offset: LogicalScreenVector,
    pub(super) spread: f32,
    color: Color,
}

impl Shadow {
    /// Builds a shadow from logical screen-space offset, spread, and color.
    pub fn new(offset: LogicalScreenVector, spread: f32, color: Color) -> Self {
        Self {
            offset,
            spread,
            color,
        }
    }

    /// Returns the logical-pixel shadow offset.
    pub const fn offset(self) -> LogicalScreenVector {
        self.offset
    }
    /// Returns extra logical-pixel spread.
    pub const fn spread(self) -> f32 {
        self.spread
    }
    /// Returns the shadow color.
    pub const fn color(self) -> Color {
        self.color
    }

    fn is_valid(self) -> bool {
        self.offset.is_finite()
            && self.spread.is_finite()
            && self.spread >= 0.0
            && (self.spread * 2.0).is_finite()
            && self.color.is_normalized()
    }
}
