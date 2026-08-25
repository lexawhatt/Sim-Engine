use std::{
    error::Error,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
};

use crate::{Color, Interpolate, LogicalScreenPosition, LogicalScreenVector, Rect, Vec2};

/// Primitive category attached to structured scene validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePrimitive {
    /// Circle geometry or style.
    Circle,
    /// Rectangle geometry or style.
    Rect,
    /// Single line geometry or stroke.
    Line,
    /// Connected polyline geometry or stroke.
    Polyline,
}

/// Reason a command or temporary scene state was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneError {
    /// Clear color is non-finite or outside normalized RGBA.
    InvalidBackground,
    /// Primitive coordinates contain NaN or infinity.
    NonFiniteGeometry(ScenePrimitive),
    /// Radius, size, corner radius, or stroke width is outside its valid range.
    InvalidDimension(ScenePrimitive),
    /// Line or polyline has no drawable segment.
    DegenerateGeometry(ScenePrimitive),
    /// Shape has no fill, stroke, or shadow.
    MissingStyle(ScenePrimitive),
    /// Fill color or gradient configuration is invalid.
    InvalidFill(ScenePrimitive),
    /// Stroke width or color is invalid.
    InvalidStroke(ScenePrimitive),
    /// Shadow offset, spread, or color is invalid.
    InvalidShadow(ScenePrimitive),
    /// Active logical screen clip is non-finite or empty.
    InvalidScreenClip,
    /// Pseudo-depth must be finite.
    NonFiniteDepth,
}

impl fmt::Display for SceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "scene command rejected: {self:?}")
    }
}

impl Error for SceneError {}

/// Ordered list of visual draw commands for a single frame.
///
/// A scene contains renderable primitives only. It does not own simulation
/// entities, domain rules, or time stepping.
#[derive(Debug, Clone)]
pub struct Scene {
    background: Color,
    /// Commands stored in draw order.
    commands: Vec<SceneCommand>,
    next_order: u64,
    current_screen_clip: Option<ScreenClipRect>,
    current_depth: f32,
}

impl Scene {
    /// Creates an empty scene with a normalized linear-RGBA background color.
    pub fn new(background: Color) -> Result<Self, SceneError> {
        if !background.is_normalized() {
            return Err(SceneError::InvalidBackground);
        }
        Ok(Self {
            background,
            commands: Vec::new(),
            next_order: 0,
            current_screen_clip: None,
            current_depth: 0.0,
        })
    }

    /// Returns the normalized clear color used before drawing commands.
    pub fn background(&self) -> Color {
        self.background
    }

    /// Replaces the clear color used before drawing commands.
    pub fn set_background(&mut self, background: Color) -> Result<(), SceneError> {
        if !background.is_normalized() {
            return Err(SceneError::InvalidBackground);
        }
        self.background = background;
        Ok(())
    }

    /// Removes all draw commands and active clipping without changing the background color.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.next_order = 0;
        self.current_screen_clip = None;
        self.current_depth = 0.0;
    }

    /// Returns accepted commands in stable layer and insertion order.
    pub fn commands(&self) -> &[SceneCommand] {
        &self.commands
    }

    /// Returns the number of accepted render commands.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Replaces the screen-space clipping rectangle captured by new commands.
    ///
    /// The rectangle uses logical screen pixels with the origin at the top-left
    /// of the render surface. Existing commands keep the clip they captured when
    /// they were appended. `None` disables clipping for subsequent commands.
    /// The previous clip is returned so callers can restore explicit drawing
    /// state without tracking it separately. Invalid clips are rejected here,
    /// before they can affect a later command insertion.
    pub fn set_screen_clip(
        &mut self,
        screen_clip: Option<ScreenClipRect>,
    ) -> Result<Option<ScreenClipRect>, SceneError> {
        if screen_clip.is_some_and(|clip| !clip.is_valid()) {
            return Err(SceneError::InvalidScreenClip);
        }
        Ok(std::mem::replace(
            &mut self.current_screen_clip,
            screen_clip,
        ))
    }

    /// Appends commands from `draw` using a temporary screen-space clip.
    ///
    /// The rectangle uses logical screen pixels. Nested clips are intersected,
    /// so inner drawing cannot escape an outer clip. The previous clip is
    /// restored after `draw` returns.
    pub fn with_screen_clip<R>(
        &mut self,
        screen_clip: ScreenClipRect,
        draw: impl FnOnce(&mut Self) -> R,
    ) -> Result<R, SceneError> {
        if !screen_clip.is_valid() {
            return Err(SceneError::InvalidScreenClip);
        }
        let previous = self.current_screen_clip;
        let combined = match previous {
            Some(outer) => outer.intersection(screen_clip)?,
            None => screen_clip,
        };
        if !combined.is_valid() {
            return Err(SceneError::InvalidScreenClip);
        }
        self.current_screen_clip = Some(combined);
        let result = catch_unwind(AssertUnwindSafe(|| draw(self)));
        self.current_screen_clip = previous;
        match result {
            Ok(result) => Ok(result),
            Err(payload) => resume_unwind(payload),
        }
    }

    /// Replaces the pseudo-depth captured by subsequently appended commands.
    ///
    /// Depth is measured in caller-defined units and converted through
    /// [`crate::Projection2d::depth_scale`]. It affects projection only, not draw
    /// order. Non-finite values return `false` and leave the current depth unchanged.
    pub fn set_depth(&mut self, depth: f32) -> bool {
        self.try_set_depth(depth).is_ok()
    }

    /// Replaces pseudo-depth and reports why invalid input was rejected.
    pub fn try_set_depth(&mut self, depth: f32) -> Result<(), SceneError> {
        if !depth.is_finite() {
            return Err(SceneError::NonFiniteDepth);
        }
        self.current_depth = depth;
        Ok(())
    }

    /// Appends commands with a temporary pseudo-depth value.
    ///
    /// The previous depth is restored after `draw` returns or unwinds. Non-finite
    /// depth returns an error without calling `draw`.
    pub fn with_depth<R>(
        &mut self,
        depth: f32,
        draw: impl FnOnce(&mut Self) -> R,
    ) -> Result<R, SceneError> {
        if !depth.is_finite() {
            return Err(SceneError::NonFiniteDepth);
        }

        let previous = self.current_depth;
        self.current_depth = depth;
        let result = catch_unwind(AssertUnwindSafe(|| draw(self)));
        self.current_depth = previous;
        match result {
            Ok(result) => Ok(result),
            Err(payload) => resume_unwind(payload),
        }
    }

    /// Appends a valid command to the default layer.
    ///
    /// Returns `false` and leaves the scene unchanged when the primitive contains
    /// non-finite coordinates, invalid dimensions, or no drawable geometry.
    pub fn push(&mut self, command: DrawCommand) -> bool {
        self.try_push(command).is_ok()
    }

    /// Appends a command to the default layer with structured rejection diagnostics.
    pub fn try_push(&mut self, command: DrawCommand) -> Result<(), SceneError> {
        self.try_push_to_layer(Layer::DEFAULT, command)
    }

    /// Appends a command to a layer.
    ///
    /// Lower layer values are drawn first. Commands on the same layer preserve
    /// insertion order.
    /// Returns `false` and leaves ordering unchanged when `command` is invalid.
    pub fn push_to_layer(&mut self, layer: Layer, command: DrawCommand) -> bool {
        self.try_push_to_layer(layer, command).is_ok()
    }

    /// Appends a command to a layer with structured rejection diagnostics.
    pub fn try_push_to_layer(
        &mut self,
        layer: Layer,
        command: DrawCommand,
    ) -> Result<(), SceneError> {
        command.validate()?;
        if self
            .current_screen_clip
            .is_some_and(|screen_clip| !screen_clip.is_valid())
        {
            return Err(SceneError::InvalidScreenClip);
        }

        let order = self.next_order;
        self.next_order = self.next_order.saturating_add(1);
        let scene_command = SceneCommand {
            layer,
            order,
            depth: self.current_depth,
            screen_clip: self.current_screen_clip,
            command,
        };
        let position = self.commands.partition_point(|existing| {
            existing.layer < scene_command.layer
                || (existing.layer == scene_command.layer && existing.order <= scene_command.order)
        });
        self.commands.insert(position, scene_command);
        Ok(())
    }

    /// Appends a circle in world coordinates.
    ///
    /// `radius` is in world units. Returns `false` for non-positive or non-finite
    /// radius, non-finite center, or invalid style values.
    pub fn circle(&mut self, center: Vec2, radius: f32, style: ShapeStyle) -> bool {
        self.try_circle(center, radius, style).is_ok()
    }

    /// Appends a circle and returns a structured validation error on rejection.
    pub fn try_circle(
        &mut self,
        center: Vec2,
        radius: f32,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_circle_on_layer(Layer::DEFAULT, center, radius, style)
    }

    /// Appends a circle in world coordinates to a layer.
    ///
    /// `radius` is in world units. Returns `false` for non-positive or non-finite
    /// radius, non-finite center, or invalid style values.
    pub fn circle_on_layer(
        &mut self,
        layer: Layer,
        center: Vec2,
        radius: f32,
        style: ShapeStyle,
    ) -> bool {
        self.try_circle_on_layer(layer, center, radius, style)
            .is_ok()
    }

    /// Appends a circle to a layer with structured rejection diagnostics.
    pub fn try_circle_on_layer(
        &mut self,
        layer: Layer,
        center: Vec2,
        radius: f32,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_push_to_layer(
            layer,
            DrawCommand::Circle(Circle {
                center,
                radius,
                style,
            }),
        )
    }

    /// Appends an axis-aligned rectangle in world coordinates.
    ///
    /// `corner_radius` is in world units and renderers clamp it to half the
    /// rectangle size. Invalid bounds, negative radius, and invalid styles return
    /// `false` without adding a command.
    pub fn rect(&mut self, rect: Rect, corner_radius: f32, style: ShapeStyle) -> bool {
        self.try_rect(rect, corner_radius, style).is_ok()
    }

    /// Appends a rectangle and returns a structured validation error on rejection.
    pub fn try_rect(
        &mut self,
        rect: Rect,
        corner_radius: f32,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_rect_on_layer(Layer::DEFAULT, rect, corner_radius, style)
    }

    /// Appends an axis-aligned rectangle in world coordinates to a layer.
    ///
    /// `corner_radius` is in world units and renderers clamp it to half the
    /// rectangle size. Invalid bounds, negative radius, and invalid styles return
    /// `false` without adding a command.
    pub fn rect_on_layer(
        &mut self,
        layer: Layer,
        rect: Rect,
        corner_radius: f32,
        style: ShapeStyle,
    ) -> bool {
        self.try_rect_on_layer(layer, rect, corner_radius, style)
            .is_ok()
    }

    /// Appends a rectangle to a layer with structured rejection diagnostics.
    pub fn try_rect_on_layer(
        &mut self,
        layer: Layer,
        rect: Rect,
        corner_radius: f32,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_push_to_layer(
            layer,
            DrawCommand::Rect(RectShape {
                rect,
                corner_radius,
                style,
            }),
        )
    }

    /// Appends a stroked line between world-space points.
    ///
    /// `width` is in logical screen pixels so line thickness stays readable while the
    /// camera zooms. Degenerate geometry and invalid width or color return
    /// `false`.
    pub fn line(&mut self, from: Vec2, to: Vec2, width: f32, color: Color) -> bool {
        self.try_line(from, to, width, color).is_ok()
    }

    /// Appends a line and returns a structured validation error on rejection.
    pub fn try_line(
        &mut self,
        from: Vec2,
        to: Vec2,
        width: f32,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_line_on_layer(Layer::DEFAULT, from, to, width, color)
    }

    /// Appends a stroked line between world-space points to a layer.
    ///
    /// `width` is in logical screen pixels so line thickness stays readable while the
    /// camera zooms. Degenerate geometry and invalid width or color return
    /// `false`.
    pub fn line_on_layer(
        &mut self,
        layer: Layer,
        from: Vec2,
        to: Vec2,
        width: f32,
        color: Color,
    ) -> bool {
        self.try_line_on_layer(layer, from, to, width, color)
            .is_ok()
    }

    /// Appends a line to a layer with structured rejection diagnostics.
    pub fn try_line_on_layer(
        &mut self,
        layer: Layer,
        from: Vec2,
        to: Vec2,
        width: f32,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_push_to_layer(
            layer,
            DrawCommand::Line(Line {
                from,
                to,
                stroke: Stroke { width, color },
            }),
        )
    }

    /// Appends connected stroked segments through world-space points.
    ///
    /// `width` is in logical screen pixels. Empty, single-point, fully degenerate, and
    /// non-finite input returns `false` without adding a command.
    pub fn polyline(&mut self, points: Vec<Vec2>, width: f32, color: Color) -> bool {
        self.try_polyline(points, width, color).is_ok()
    }

    /// Appends a polyline and returns a structured validation error on rejection.
    pub fn try_polyline(
        &mut self,
        points: Vec<Vec2>,
        width: f32,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_polyline_on_layer(Layer::DEFAULT, points, width, color)
    }

    /// Appends connected stroked segments through world-space points to a layer.
    ///
    /// `width` is in logical screen pixels. Empty, single-point, fully degenerate, and
    /// non-finite input returns `false` without adding a command.
    pub fn polyline_on_layer(
        &mut self,
        layer: Layer,
        points: Vec<Vec2>,
        width: f32,
        color: Color,
    ) -> bool {
        self.try_polyline_on_layer(layer, points, width, color)
            .is_ok()
    }

    /// Appends a polyline to a layer with structured rejection diagnostics.
    pub fn try_polyline_on_layer(
        &mut self,
        layer: Layer,
        points: Vec<Vec2>,
        width: f32,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_push_to_layer(
            layer,
            DrawCommand::Polyline(Polyline {
                points,
                stroke: Stroke { width, color },
            }),
        )
    }
}

/// Draw layer used to order scene commands.
///
/// Lower values are drawn first. Higher values appear above lower layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Layer {
    order: i32,
}

impl Layer {
    /// Default layer used by convenience drawing methods.
    pub const DEFAULT: Self = Self::new(0);
    /// Conventional layer for background guides or grids.
    pub const BACKGROUND: Self = Self::new(-100);
    /// Conventional layer for foreground annotations or overlays.
    pub const FOREGROUND: Self = Self::new(100);

    /// Builds a layer from an order value.
    pub const fn new(order: i32) -> Self {
        Self { order }
    }

    /// Returns the order value used for sorting.
    pub const fn order(self) -> i32 {
        self.order
    }
}

/// Axis-aligned clipping rectangle in logical screen pixels.
///
/// The coordinate origin is the top-left of the render surface and positive y
/// points downward. Constructors reject empty and non-finite bounds.
/// Renderers clamp valid clips to the target viewport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenClipRect {
    rect: Rect,
}

impl ScreenClipRect {
    /// Builds a non-empty clip from two corners in logical screen pixels.
    pub fn new(min: LogicalScreenPosition, max: LogicalScreenPosition) -> Result<Self, SceneError> {
        Self::from_rect(Rect::new(min.to_vec2(), max.to_vec2()))
    }

    /// Builds a clip from its top-left corner and size in logical screen pixels.
    ///
    /// Negative size components are accepted and normalized before validation.
    pub fn from_min_size(
        min: LogicalScreenPosition,
        size: LogicalScreenVector,
    ) -> Result<Self, SceneError> {
        Self::from_rect(Rect::from_min_size(min.to_vec2(), size.to_vec2()))
    }

    fn from_rect(rect: Rect) -> Result<Self, SceneError> {
        let clip = Self { rect };
        clip.is_valid()
            .then_some(clip)
            .ok_or(SceneError::InvalidScreenClip)
    }

    #[cfg(feature = "wgpu")]
    pub(crate) fn rect(self) -> Rect {
        self.rect
    }

    /// Returns the normalized top-left corner in logical screen pixels.
    pub fn min(self) -> LogicalScreenPosition {
        LogicalScreenPosition::from_vec2(self.rect.normalized().min)
    }

    /// Returns the normalized bottom-right corner in logical screen pixels.
    pub fn max(self) -> LogicalScreenPosition {
        LogicalScreenPosition::from_vec2(self.rect.normalized().max)
    }

    /// Returns the overlap between two screen-space clips.
    ///
    /// Disjoint inputs return [`SceneError::InvalidScreenClip`].
    pub fn intersection(self, other: Self) -> Result<Self, SceneError> {
        let first = self.rect.normalized();
        let second = other.rect.normalized();
        let min = Vec2::new(first.min.x.max(second.min.x), first.min.y.max(second.min.y));
        let max = Vec2::new(first.max.x.min(second.max.x), first.max.y.min(second.max.y));

        Self::new(
            LogicalScreenPosition::from_vec2(min),
            LogicalScreenPosition::from_vec2(Vec2::new(max.x.max(min.x), max.y.max(min.y))),
        )
    }

    fn is_valid(self) -> bool {
        if !self.rect.min.is_finite() || !self.rect.max.is_finite() {
            return false;
        }

        let rect = self.rect.normalized();
        rect.width() > 0.0 && rect.height() > 0.0
    }
}

/// Draw command with layer and stable insertion order metadata.
#[derive(Debug, Clone)]
pub struct SceneCommand {
    /// Layer used to sort commands before rendering.
    layer: Layer,
    /// Monotonic insertion order within the scene.
    order: u64,
    /// Caller-defined pseudo-depth used by camera projection.
    depth: f32,
    /// Optional clipping rectangle in logical screen pixels.
    screen_clip: Option<ScreenClipRect>,
    /// Visual primitive to render.
    command: DrawCommand,
}

impl SceneCommand {
    /// Returns the layer used for stable scene ordering.
    pub fn layer(&self) -> Layer {
        self.layer
    }

    /// Returns the insertion order assigned by the scene.
    pub fn order(&self) -> u64 {
        self.order
    }

    /// Returns caller-defined pseudo-depth captured by this command.
    ///
    /// Depth affects camera projection only. Commands on one layer retain their
    /// insertion order regardless of depth.
    pub fn depth(&self) -> f32 {
        self.depth
    }

    /// Returns the optional clip captured when the command was appended.
    pub fn screen_clip(&self) -> Option<ScreenClipRect> {
        self.screen_clip
    }

    /// Returns the visual primitive stored by this command.
    pub fn command(&self) -> &DrawCommand {
        &self.command
    }
}

/// Renderable primitive command.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DrawCommand {
    /// Filled and/or stroked circle.
    Circle(Circle),
    /// Filled and/or stroked rectangle.
    Rect(RectShape),
    /// Single stroked segment.
    Line(Line),
    /// Connected stroked segments.
    Polyline(Polyline),
}

impl DrawCommand {
    fn validate(&self) -> Result<(), SceneError> {
        match self {
            Self::Circle(circle) => {
                if !circle.center.is_finite() {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Circle));
                }
                if !circle.radius.is_finite() || circle.radius <= 0.0 {
                    return Err(SceneError::InvalidDimension(ScenePrimitive::Circle));
                }
                if !(circle.center + Vec2::splat(circle.radius)).is_finite()
                    || !(circle.center - Vec2::splat(circle.radius)).is_finite()
                {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Circle));
                }
                circle.style.validate(ScenePrimitive::Circle)
            }
            Self::Rect(rectangle) => {
                let rect = rectangle.rect.normalized();
                if !rect.min.is_finite() || !rect.max.is_finite() {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Rect));
                }
                let width = rect.width();
                let height = rect.height();
                if !width.is_finite() || !height.is_finite() {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Rect));
                }
                if width <= 0.0
                    || height <= 0.0
                    || !rectangle.corner_radius.is_finite()
                    || rectangle.corner_radius < 0.0
                {
                    return Err(SceneError::InvalidDimension(ScenePrimitive::Rect));
                }
                rectangle.style.validate(ScenePrimitive::Rect)
            }
            Self::Line(line) => {
                if !line.from.is_finite() || !line.to.is_finite() {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Line));
                }
                if !(line.to - line.from).is_finite() {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Line));
                }
                if !drawable_segment(line.from, line.to) {
                    return Err(SceneError::DegenerateGeometry(ScenePrimitive::Line));
                }
                if !line.stroke.is_valid() {
                    return Err(SceneError::InvalidStroke(ScenePrimitive::Line));
                }
                Ok(())
            }
            Self::Polyline(polyline) => {
                if !polyline.points.iter().all(|point| point.is_finite()) {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Polyline));
                }
                if polyline
                    .points
                    .windows(2)
                    .any(|pair| !(pair[1] - pair[0]).is_finite())
                {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Polyline));
                }
                if polyline.points.len() < 2
                    || !polyline
                        .points
                        .windows(2)
                        .any(|pair| drawable_segment(pair[0], pair[1]))
                {
                    return Err(SceneError::DegenerateGeometry(ScenePrimitive::Polyline));
                }
                if !polyline.stroke.is_valid() {
                    return Err(SceneError::InvalidStroke(ScenePrimitive::Polyline));
                }
                Ok(())
            }
        }
    }
}

fn drawable_segment(from: Vec2, to: Vec2) -> bool {
    let delta = to - from;
    delta.is_finite() && (delta.x != 0.0 || delta.y != 0.0)
}

/// Circle primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct Circle {
    center: Vec2,
    radius: f32,
    style: ShapeStyle,
}

impl Circle {
    /// Returns the center in world coordinates.
    pub const fn center(&self) -> Vec2 {
        self.center
    }
    /// Returns the radius in world units.
    pub const fn radius(&self) -> f32 {
        self.radius
    }
    /// Returns fill, stroke, and shadow styling.
    pub const fn style(&self) -> ShapeStyle {
        self.style
    }
}

/// Axis-aligned rectangle primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct RectShape {
    rect: Rect,
    corner_radius: f32,
    style: ShapeStyle,
}

impl RectShape {
    /// Returns rectangle bounds in world coordinates.
    pub const fn rect(&self) -> Rect {
        self.rect
    }
    /// Returns corner radius in world units.
    pub const fn corner_radius(&self) -> f32 {
        self.corner_radius
    }
    /// Returns fill, stroke, and shadow styling.
    pub const fn style(&self) -> ShapeStyle {
        self.style
    }
}

/// Stroked line primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct Line {
    from: Vec2,
    to: Vec2,
    stroke: Stroke,
}

impl Line {
    /// Returns the start point in world coordinates.
    pub const fn from(&self) -> Vec2 {
        self.from
    }
    /// Returns the end point in world coordinates.
    pub const fn to(&self) -> Vec2 {
        self.to
    }
    /// Returns the logical-pixel stroke.
    pub const fn stroke(&self) -> Stroke {
        self.stroke
    }
}

/// Connected line-strip primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct Polyline {
    points: Vec<Vec2>,
    stroke: Stroke,
}

impl Polyline {
    /// Returns connected points in world coordinates.
    pub fn points(&self) -> &[Vec2] {
        &self.points
    }
    /// Returns the logical-pixel stroke.
    pub const fn stroke(&self) -> Stroke {
        self.stroke
    }
}

/// Fill, stroke, and shadow styling for solid primitives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeStyle {
    fill: Option<Fill>,
    stroke: Option<Stroke>,
    shadow: Option<Shadow>,
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

    fn validate(self, primitive: ScenePrimitive) -> Result<(), SceneError> {
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
        match self {
            Self::Solid(color) => color,
            Self::LinearGradient(gradient) => gradient.color_at(world),
            Self::RadialGradient(gradient) => gradient.color_at(world),
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
        if !self.is_valid() || !world.is_finite() {
            return self.end_color.clamp();
        }

        let axis_x = f64::from(self.end.x) - f64::from(self.start.x);
        let axis_y = f64::from(self.end.y) - f64::from(self.start.y);
        let relative_x = f64::from(world.x) - f64::from(self.start.x);
        let relative_y = f64::from(world.y) - f64::from(self.start.y);
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
        if !self.is_valid() || !world.is_finite() {
            return self.outer_color.clamp();
        }

        let radius_range = self.outer_radius - self.inner_radius;
        if radius_range.abs() <= f32::EPSILON {
            return self.outer_color;
        }

        let horizontal = f64::from(world.x) - f64::from(self.center.x);
        let vertical = f64::from(world.y) - f64::from(self.center.y);
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
    width: f32,
    color: Color,
}

impl Stroke {
    /// Builds a logical-pixel stroke.
    pub const fn new(width: f32, color: Color) -> Self {
        Self { width, color }
    }
    /// Returns width in logical screen pixels.
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
    spread: f32,
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

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use crate::{
        Color, DrawCommand, Fill, Layer, LinearGradient, LogicalScreenPosition,
        LogicalScreenVector, RadialGradient, Rect, Scene, SceneError, ScenePrimitive,
        ScreenClipRect, ShapeStyle, Vec2,
    };

    #[test]
    fn scene_collects_visual_commands_only() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));

        assert_eq!(scene.commands.len(), 1);
    }

    #[test]
    fn scene_keeps_commands_sorted_by_layer() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.circle_on_layer(
            Layer::FOREGROUND,
            Vec2::ZERO,
            10.0,
            ShapeStyle::filled(Color::WHITE),
        );
        scene.line_on_layer(Layer::BACKGROUND, Vec2::ZERO, Vec2::X, 1.0, Color::WHITE);

        assert!(matches!(scene.commands[0].command, DrawCommand::Line(_)));
        assert!(matches!(scene.commands[1].command, DrawCommand::Circle(_)));
    }

    #[test]
    fn linear_gradient_samples_along_axis() {
        let gradient = LinearGradient::new(Vec2::ZERO, Vec2::X, Color::BLACK, Color::WHITE);

        let sampled = Fill::LinearGradient(gradient).color_at(Vec2::new(0.5, 4.0));

        assert!((sampled.red() - 0.5).abs() < 0.001);
        assert!((sampled.green() - 0.5).abs() < 0.001);
        assert!((sampled.blue() - 0.5).abs() < 0.001);
    }

    #[test]
    fn extreme_gradients_sample_without_f32_overflow() {
        let gradient = LinearGradient::new(
            Vec2::new(-f32::MAX, 0.0),
            Vec2::new(f32::MAX, 0.0),
            Color::BLACK,
            Color::WHITE,
        );
        let mut scene = Scene::new(Color::BLACK).unwrap();

        assert!(
            scene
                .try_circle(
                    Vec2::ZERO,
                    1.0,
                    ShapeStyle::filled_with(Fill::LinearGradient(gradient)),
                )
                .is_ok()
        );
        assert_eq!(
            gradient.color_at(Vec2::ZERO),
            Color::rgba(0.5, 0.5, 0.5, 1.0)
        );

        let radial = RadialGradient::new(
            Vec2::splat(f32::MAX),
            0.0,
            f32::MAX,
            Color::BLACK,
            Color::WHITE,
        );
        assert_eq!(radial.color_at(Vec2::splat(-f32::MAX)), Color::WHITE);
    }

    #[test]
    fn overflowing_circle_and_shadow_are_rejected_at_scene_boundary() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        assert_eq!(
            scene.try_circle(
                Vec2::splat(f32::MAX),
                f32::MAX,
                ShapeStyle::filled(Color::WHITE),
            ),
            Err(SceneError::NonFiniteGeometry(ScenePrimitive::Circle))
        );
        assert_eq!(
            scene.try_circle(
                Vec2::ZERO,
                1.0,
                ShapeStyle::filled(Color::WHITE).with_shadow(super::Shadow::new(
                    LogicalScreenVector::new(0.0, 0.0),
                    f32::MAX,
                    Color::WHITE,
                )),
            ),
            Err(SceneError::InvalidShadow(ScenePrimitive::Circle))
        );
        assert_eq!(
            scene.try_line(
                Vec2::new(-f32::MAX, 0.0),
                Vec2::new(f32::MAX, 0.0),
                1.0,
                Color::WHITE,
            ),
            Err(SceneError::NonFiniteGeometry(ScenePrimitive::Line))
        );
        assert_eq!(
            scene.try_rect(
                Rect::new(Vec2::new(-f32::MAX, 0.0), Vec2::new(f32::MAX, 1.0),),
                0.0,
                ShapeStyle::filled(Color::WHITE),
            ),
            Err(SceneError::NonFiniteGeometry(ScenePrimitive::Rect))
        );
    }

    #[test]
    fn commands_capture_temporary_screen_clip() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        let outer = ScreenClipRect::from_min_size(
            LogicalScreenPosition::new(10.0, 20.0),
            LogicalScreenVector::new(100.0, 80.0),
        )
        .unwrap();
        let inner = ScreenClipRect::from_min_size(
            LogicalScreenPosition::new(50.0, 0.0),
            LogicalScreenVector::new(100.0, 50.0),
        )
        .unwrap();

        scene
            .with_screen_clip(outer, |scene| {
                scene
                    .with_screen_clip(inner, |scene| {
                        scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));
                    })
                    .unwrap();
            })
            .unwrap();
        scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));

        assert_eq!(
            scene.commands[0].screen_clip,
            Some(
                ScreenClipRect::new(
                    LogicalScreenPosition::new(50.0, 20.0),
                    LogicalScreenPosition::new(110.0, 50.0),
                )
                .unwrap(),
            )
        );
        assert_eq!(scene.commands[1].screen_clip, None);
    }

    #[test]
    fn scene_rejects_invalid_primitives_before_ordering() {
        let mut scene = Scene::new(Color::BLACK).unwrap();

        assert!(!scene.circle(Vec2::ZERO, -1.0, ShapeStyle::filled(Color::WHITE)));
        assert!(!scene.line(Vec2::ZERO, Vec2::X, f32::NAN, Color::WHITE));
        assert!(!scene.polyline(vec![Vec2::ZERO], 1.0, Color::WHITE));
        assert_eq!(scene.command_count(), 0);

        assert!(scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE)));
        assert_eq!(scene.commands()[0].order(), 0);
    }

    #[test]
    fn background_and_clip_are_validated_when_created() {
        assert!(matches!(
            Scene::new(Color::rgba(f32::NAN, 0.0, 0.0, 1.0)),
            Err(SceneError::InvalidBackground)
        ));
        assert!(matches!(
            Scene::new(Color::rgb(1.01, 0.0, 0.0)),
            Err(SceneError::InvalidBackground)
        ));
        assert_eq!(
            ScreenClipRect::new(
                LogicalScreenPosition::new(0.0, 0.0),
                LogicalScreenPosition::new(0.0, 10.0),
            ),
            Err(SceneError::InvalidScreenClip)
        );
        assert_eq!(
            ScreenClipRect::new(
                LogicalScreenPosition::new(f32::NAN, 0.0),
                LogicalScreenPosition::new(10.0, 10.0),
            ),
            Err(SceneError::InvalidScreenClip)
        );
    }

    #[test]
    fn render_bound_scene_colors_must_be_normalized() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        assert_eq!(
            scene.try_circle(
                Vec2::ZERO,
                1.0,
                ShapeStyle::filled(Color::rgb(2.0, 0.0, 0.0)),
            ),
            Err(SceneError::InvalidFill(ScenePrimitive::Circle))
        );
        assert_eq!(
            scene.try_line(Vec2::ZERO, Vec2::X, 1.0, Color::rgba(0.0, 0.0, 0.0, -0.01),),
            Err(SceneError::InvalidStroke(ScenePrimitive::Line))
        );
        assert_eq!(scene.command_count(), 0);
    }

    #[test]
    fn depth_does_not_reorder_commands_within_a_layer() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene
            .with_depth(4.0, |scene| {
                scene.circle(Vec2::X, 1.0, ShapeStyle::filled(Color::WHITE));
            })
            .unwrap();
        scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE));

        assert_eq!(scene.commands()[0].depth(), 4.0);
        assert_eq!(scene.commands()[1].depth(), 0.0);
    }

    #[test]
    fn structured_scene_errors_preserve_rejection_reason() {
        let mut scene = Scene::new(Color::BLACK).unwrap();

        assert_eq!(
            scene.try_circle(Vec2::ZERO, -1.0, ShapeStyle::filled(Color::WHITE)),
            Err(SceneError::InvalidDimension(ScenePrimitive::Circle))
        );
        assert_eq!(
            scene.try_line(Vec2::ZERO, Vec2::ZERO, 1.0, Color::WHITE),
            Err(SceneError::DegenerateGeometry(ScenePrimitive::Line))
        );
        assert_eq!(
            scene.try_circle(
                Vec2::ZERO,
                1.0,
                ShapeStyle {
                    fill: None,
                    stroke: None,
                    shadow: None,
                },
            ),
            Err(SceneError::MissingStyle(ScenePrimitive::Circle))
        );
        assert_eq!(scene.command_count(), 0);
    }

    #[test]
    fn commands_capture_depth_and_restore_it_after_unwind() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            let _ = scene.with_depth(4.5, |scene| {
                scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE));
                panic!("intentional depth unwind");
            });
        }));
        assert!(panic_result.is_err());

        scene.circle(Vec2::X, 1.0, ShapeStyle::filled(Color::WHITE));

        assert_eq!(scene.commands()[0].depth(), 4.5);
        assert_eq!(scene.commands()[1].depth(), 0.0);
        assert_eq!(
            scene.with_depth(f32::NAN, |_| {}),
            Err(SceneError::NonFiniteDepth)
        );
    }

    #[test]
    fn gradients_sanitize_invalid_and_reversed_configuration() {
        let invalid_linear = LinearGradient::new(
            Vec2::new(f32::NAN, 0.0),
            Vec2::X,
            Color::BLACK,
            Color::WHITE,
        );
        assert_eq!(invalid_linear.color_at(Vec2::ZERO), Color::WHITE);

        let reversed = RadialGradient::new(Vec2::ZERO, 10.0, 5.0, Color::WHITE, Color::BLACK);
        assert_eq!(reversed.inner_radius(), 5.0);
        assert_eq!(reversed.outer_radius(), 10.0);
        assert_eq!(reversed.color_at(Vec2::ZERO), Color::BLACK);
    }

    #[test]
    fn temporary_screen_clip_restores_state_after_unwind() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        let clip = ScreenClipRect::from_min_size(
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalScreenVector::new(100.0, 100.0),
        )
        .unwrap();

        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            let _ = scene.with_screen_clip(clip, |_| panic!("intentional test panic"));
        }));
        assert!(panic_result.is_err());

        assert!(scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE)));
        assert_eq!(scene.commands()[0].screen_clip(), None);
    }
}
