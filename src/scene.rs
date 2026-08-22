use crate::{Color, Rect, Vec2};

/// Ordered list of visual draw commands for a single frame.
///
/// A scene contains renderable primitives only. It does not own simulation
/// entities, domain rules, or time stepping.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Clear color used before drawing commands.
    pub background: Color,
    /// Commands drawn in insertion order.
    pub commands: Vec<DrawCommand>,
}

impl Scene {
    /// Creates an empty scene with a background color.
    pub fn new(background: Color) -> Self {
        Self {
            background,
            commands: Vec::new(),
        }
    }

    /// Removes all draw commands without changing the background color.
    pub fn clear(&mut self) {
        self.commands.clear();
    }

    /// Appends a command to the draw order.
    pub fn push(&mut self, command: DrawCommand) {
        self.commands.push(command);
    }

    /// Appends a circle in world coordinates.
    ///
    /// `radius` is in world units. Non-positive radius values are ignored by
    /// renderers.
    pub fn circle(&mut self, center: Vec2, radius: f32, style: ShapeStyle) {
        self.push(DrawCommand::Circle(Circle {
            center,
            radius,
            style,
        }));
    }

    /// Appends an axis-aligned rectangle in world coordinates.
    ///
    /// `corner_radius` is in world units and renderers clamp it to the rectangle
    /// size.
    pub fn rect(&mut self, rect: Rect, corner_radius: f32, style: ShapeStyle) {
        self.push(DrawCommand::Rect(RectShape {
            rect,
            corner_radius,
            style,
        }));
    }

    /// Appends a stroked line between world-space points.
    ///
    /// `width` is in screen pixels so line thickness stays readable while the
    /// camera zooms.
    pub fn line(&mut self, from: Vec2, to: Vec2, width: f32, color: Color) {
        self.push(DrawCommand::Line(Line {
            from,
            to,
            stroke: Stroke { width, color },
        }));
    }

    /// Appends connected stroked segments through world-space points.
    ///
    /// `width` is in screen pixels. Empty and single-point polylines are ignored
    /// by renderers.
    pub fn polyline(&mut self, points: Vec<Vec2>, width: f32, color: Color) {
        self.push(DrawCommand::Polyline(Polyline {
            points,
            stroke: Stroke { width, color },
        }));
    }
}

/// Renderable primitive command.
#[derive(Debug, Clone)]
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

/// Circle primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct Circle {
    /// Center in world coordinates.
    pub center: Vec2,
    /// Radius in world units.
    pub radius: f32,
    /// Fill, stroke, and shadow styling.
    pub style: ShapeStyle,
}

/// Axis-aligned rectangle primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct RectShape {
    /// Rectangle bounds in world coordinates.
    pub rect: Rect,
    /// Corner radius in world units.
    pub corner_radius: f32,
    /// Fill, stroke, and shadow styling.
    pub style: ShapeStyle,
}

/// Stroked line primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct Line {
    /// Start point in world coordinates.
    pub from: Vec2,
    /// End point in world coordinates.
    pub to: Vec2,
    /// Stroke style. Width is in screen pixels.
    pub stroke: Stroke,
}

/// Connected line-strip primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct Polyline {
    /// Points in world coordinates.
    pub points: Vec<Vec2>,
    /// Stroke style. Width is in screen pixels.
    pub stroke: Stroke,
}

/// Fill, stroke, and shadow styling for solid primitives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeStyle {
    /// Fill color, or `None` for no fill.
    pub fill: Option<Color>,
    /// Stroke style, or `None` for no stroke.
    pub stroke: Option<Stroke>,
    /// Shadow style, or `None` for no shadow.
    pub shadow: Option<Shadow>,
}

impl ShapeStyle {
    /// Creates a style with fill color only.
    pub fn filled(color: Color) -> Self {
        Self {
            fill: Some(color),
            stroke: None,
            shadow: None,
        }
    }

    /// Creates a style with stroke only.
    ///
    /// `width` is in screen pixels.
    pub fn stroked(width: f32, color: Color) -> Self {
        Self {
            fill: None,
            stroke: Some(Stroke { width, color }),
            shadow: None,
        }
    }

    /// Creates a style with both fill and stroke.
    ///
    /// `stroke_width` is in screen pixels.
    pub fn fill_stroke(fill: Color, stroke_width: f32, stroke: Color) -> Self {
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
}

/// Stroke color and width for lines and shape outlines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stroke {
    /// Width in screen pixels.
    pub width: f32,
    /// Stroke color.
    pub color: Color,
}

/// Simple screen-space shadow used by the first renderer slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shadow {
    /// Offset in screen pixels.
    pub offset: Vec2,
    /// Extra spread in screen pixels.
    pub spread: f32,
    /// Shadow color.
    pub color: Color,
}

impl Shadow {
    /// Builds a shadow from screen-space offset, spread, and color.
    pub fn new(offset: Vec2, spread: f32, color: Color) -> Self {
        Self {
            offset,
            spread,
            color,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Color, Scene, ShapeStyle, Vec2};

    #[test]
    fn scene_collects_visual_commands_only() {
        let mut scene = Scene::new(Color::BLACK);
        scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));

        assert_eq!(scene.commands.len(), 1);
    }
}
