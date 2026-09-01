use std::mem::size_of;

use crate::scene::validate_styled_polyline;
#[cfg(any(feature = "wgpu", test))]
use crate::{Camera2d, LogicalViewport};
use crate::{
    Color, Fill, Layer, LinearGradient, LogicalPixels, LogicalScreenPosition, LogicalScreenVector,
    RadialGradient, Rect, Scene, SceneBudget, SceneBudgetResource, SceneError, ScenePrimitive,
    SceneStatistics, ScreenClipRect, ShapeStyle, StrokeStyle2d, StrokeWidthMode2d, Vec2,
};

/// A bounded 2D scene whose geometry is expressed in logical screen pixels.
///
/// The origin is the top-left of the active logical viewport and positive y
/// points downward. Geometry stays fixed while world cameras pan, zoom, rotate,
/// or change pseudo-projection. Physical DPI conversion remains a renderer
/// boundary operation. Gradient coordinates inside a [`ShapeStyle`] are also
/// interpreted in this top-left/downward space and converted exactly once at
/// insertion; the same style passed to an ordinary [`Scene`] instead uses
/// world coordinates.
#[derive(Debug, Clone)]
pub struct ScreenScene {
    scene: Scene,
}

impl ScreenScene {
    /// Creates an explicitly unbounded logical-screen scene.
    pub fn new(background: Color) -> Result<Self, SceneError> {
        Scene::new(background).map(|scene| Self { scene })
    }

    /// Creates a logical-screen scene with explicit work and upload limits.
    pub fn with_budget(background: Color, budget: SceneBudget) -> Result<Self, SceneError> {
        Scene::with_budget(background, budget).map(|scene| Self { scene })
    }

    /// Returns the normalized linear-RGBA clear color.
    pub fn background(&self) -> Color {
        self.scene.background()
    }

    /// Replaces the clear color after normalized-color validation.
    pub fn set_background(&mut self, background: Color) -> Result<(), SceneError> {
        self.scene.set_background(background)
    }

    /// Removes all screen commands and resets their work statistics.
    pub fn clear(&mut self) {
        self.scene.clear();
    }

    /// Returns the explicit scene budget, if configured.
    pub const fn budget(&self) -> Option<SceneBudget> {
        self.scene.budget()
    }

    /// Returns accepted, rejected, and conservatively estimated work.
    pub const fn statistics(&self) -> SceneStatistics {
        self.scene.statistics()
    }

    /// Returns the number of accepted screen commands.
    pub fn command_count(&self) -> usize {
        self.scene.command_count()
    }

    /// Returns actual command/polyline allocation bytes retained for reuse.
    pub fn allocation_bytes(&self) -> usize {
        self.scene.allocation_bytes()
    }

    /// Replaces the logical-screen clip captured by subsequent commands.
    pub fn set_screen_clip(
        &mut self,
        clip: Option<ScreenClipRect>,
    ) -> Result<Option<ScreenClipRect>, SceneError> {
        self.scene.set_screen_clip(clip)
    }

    /// Draws with a temporary nested logical-screen clip.
    pub fn with_screen_clip<R>(
        &mut self,
        clip: ScreenClipRect,
        draw: impl FnOnce(&mut Self) -> R,
    ) -> Result<R, SceneError> {
        let previous = self.scene.current_screen_clip();
        let clip = match previous {
            Some(outer) => outer.intersection(clip)?,
            None => clip,
        };
        self.scene.set_screen_clip(Some(clip))?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| draw(self)));
        let restore = self.scene.set_screen_clip(previous);
        match result {
            Ok(value) => {
                restore?;
                Ok(value)
            }
            Err(payload) => {
                let _ = restore;
                std::panic::resume_unwind(payload)
            }
        }
    }

    /// Appends a filled/stroked circle in logical screen pixels.
    pub fn try_circle(
        &mut self,
        center: LogicalScreenPosition,
        radius: LogicalPixels,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_circle_on_layer(Layer::DEFAULT, center, radius, style)
    }

    /// Appends a logical-screen circle to an explicit shared draw layer.
    pub fn try_circle_on_layer(
        &mut self,
        layer: Layer,
        center: LogicalScreenPosition,
        radius: LogicalPixels,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.scene.try_circle_on_layer(
            layer,
            screen_to_internal(center),
            radius.get(),
            screen_shape_style_to_internal(style),
        )
    }

    /// Appends a rectangle from a logical top-left position and positive size.
    pub fn try_rect(
        &mut self,
        min: LogicalScreenPosition,
        size: LogicalScreenVector,
        corner_radius: LogicalPixels,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_rect_on_layer(Layer::DEFAULT, min, size, corner_radius, style)
    }

    /// Appends a square-cornered rectangle from a logical top-left position
    /// and positive size.
    ///
    /// This entry point represents an exact zero corner radius without
    /// weakening [`LogicalPixels`]' strictly-positive length invariant.
    pub fn try_square_rect(
        &mut self,
        min: LogicalScreenPosition,
        size: LogicalScreenVector,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_square_rect_on_layer(Layer::DEFAULT, min, size, style)
    }

    /// Appends a logical-screen rectangle with positive size to a shared layer.
    pub fn try_rect_on_layer(
        &mut self,
        layer: Layer,
        min: LogicalScreenPosition,
        size: LogicalScreenVector,
        corner_radius: LogicalPixels,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_rect_with_radius_on_layer(layer, min, size, corner_radius.get(), style)
    }

    fn try_rect_with_radius_on_layer(
        &mut self,
        layer: Layer,
        min: LogicalScreenPosition,
        size: LogicalScreenVector,
        corner_radius: f32,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        let size_value = size.to_vec2();
        let max = min.to_vec2() + size_value;
        if !min.is_finite() || !size.is_finite() || !max.is_finite() {
            self.scene.record_external_rejection(ScenePrimitive::Rect);
            return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Rect));
        }
        if size_value.x <= 0.0 || size_value.y <= 0.0 {
            self.scene.record_external_rejection(ScenePrimitive::Rect);
            return Err(SceneError::InvalidDimension(ScenePrimitive::Rect));
        }
        self.scene.try_rect_on_layer(
            layer,
            Rect::new(
                screen_vec_to_internal(min.to_vec2()),
                screen_vec_to_internal(max),
            ),
            corner_radius,
            screen_shape_style_to_internal(style),
        )
    }

    /// Appends a square-cornered logical-screen rectangle to a shared layer.
    pub fn try_square_rect_on_layer(
        &mut self,
        layer: Layer,
        min: LogicalScreenPosition,
        size: LogicalScreenVector,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_rect_with_radius_on_layer(layer, min, size, 0.0, style)
    }

    /// Appends a line whose endpoints and width use logical pixels.
    pub fn try_line(
        &mut self,
        from: LogicalScreenPosition,
        to: LogicalScreenPosition,
        width: LogicalPixels,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_line_on_layer(Layer::DEFAULT, from, to, width, color)
    }

    /// Appends a logical-screen line to an explicit shared draw layer.
    pub fn try_line_on_layer(
        &mut self,
        layer: Layer,
        from: LogicalScreenPosition,
        to: LogicalScreenPosition,
        width: LogicalPixels,
        color: Color,
    ) -> Result<(), SceneError> {
        self.scene.try_line_on_layer(
            layer,
            screen_to_internal(from),
            screen_to_internal(to),
            width.get(),
            color,
        )
    }

    /// Appends a logical-screen line with explicit bounded stroke styling.
    pub fn try_styled_line(
        &mut self,
        from: LogicalScreenPosition,
        to: LogicalScreenPosition,
        style: StrokeStyle2d,
    ) -> Result<(), SceneError> {
        self.try_styled_line_on_layer(Layer::DEFAULT, from, to, style)
    }

    /// Appends an explicitly styled logical-screen line to a shared layer.
    pub fn try_styled_line_on_layer(
        &mut self,
        layer: Layer,
        from: LogicalScreenPosition,
        to: LogicalScreenPosition,
        style: StrokeStyle2d,
    ) -> Result<(), SceneError> {
        if style.width_mode() != StrokeWidthMode2d::LogicalPixels {
            self.scene.record_external_rejection(ScenePrimitive::Line);
            return Err(SceneError::InvalidStroke(ScenePrimitive::Line));
        }
        self.scene.try_styled_line_on_layer(
            layer,
            screen_to_internal(from),
            screen_to_internal(to),
            style,
        )
    }

    /// Appends a connected logical-screen line strip with fallible point copying.
    pub fn try_polyline(
        &mut self,
        points: &[LogicalScreenPosition],
        width: LogicalPixels,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_polyline_on_layer(Layer::DEFAULT, points, width, color)
    }

    /// Appends a logical-screen line strip to an explicit shared draw layer.
    pub fn try_polyline_on_layer(
        &mut self,
        layer: Layer,
        points: &[LogicalScreenPosition],
        width: LogicalPixels,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_styled_polyline_on_layer(layer, points, StrokeStyle2d::logical(width, color))
    }

    /// Appends an explicitly styled logical-screen line strip.
    pub fn try_styled_polyline(
        &mut self,
        points: &[LogicalScreenPosition],
        style: StrokeStyle2d,
    ) -> Result<(), SceneError> {
        self.try_styled_polyline_on_layer(Layer::DEFAULT, points, style)
    }

    /// Appends an explicitly styled logical-screen line strip to a shared layer.
    pub fn try_styled_polyline_on_layer(
        &mut self,
        layer: Layer,
        points: &[LogicalScreenPosition],
        style: StrokeStyle2d,
    ) -> Result<(), SceneError> {
        if style.width_mode() != StrokeWidthMode2d::LogicalPixels {
            self.scene
                .record_external_rejection(ScenePrimitive::Polyline);
            return Err(SceneError::InvalidStroke(ScenePrimitive::Polyline));
        }
        if let Err(error) =
            validate_styled_polyline(points.iter().copied().map(screen_to_internal), style)
        {
            self.scene
                .record_external_rejection(ScenePrimitive::Polyline);
            return Err(error);
        }
        if let Some(budget) = self.scene.budget() {
            let requested = self
                .scene
                .statistics()
                .retained_points()
                .saturating_add(points.len());
            if requested > budget.max_points() {
                self.scene
                    .record_external_rejection(ScenePrimitive::Polyline);
                return Err(SceneError::BudgetExceeded {
                    resource: SceneBudgetResource::Points,
                    limit: budget.max_points(),
                    requested,
                });
            }
        }
        let minimum_point_bytes = points.len().saturating_mul(size_of::<Vec2>());
        if let Err(error) = self.scene.preflight_command_storage(1, minimum_point_bytes) {
            self.scene
                .record_external_rejection(ScenePrimitive::Polyline);
            return Err(error);
        }
        let mut internal = Vec::new();
        if internal.try_reserve_exact(points.len()).is_err() {
            self.scene
                .record_external_rejection(ScenePrimitive::Polyline);
            return Err(SceneError::AllocationFailed {
                requested_bytes: minimum_point_bytes,
            });
        }
        let actual_point_bytes = internal.capacity().saturating_mul(size_of::<Vec2>());
        if let Err(error) = self.scene.preflight_command_storage(1, actual_point_bytes) {
            self.scene
                .record_external_rejection(ScenePrimitive::Polyline);
            return Err(error);
        }
        internal.extend(points.iter().copied().map(screen_to_internal));
        self.scene
            .try_styled_polyline_on_layer(layer, internal, style)
    }

    /// Creates a linear-gradient fill using typed logical-screen endpoints.
    pub fn linear_gradient(
        start: LogicalScreenPosition,
        end: LogicalScreenPosition,
        start_color: Color,
        end_color: Color,
    ) -> Fill {
        Fill::LinearGradient(LinearGradient::new(
            start.to_vec2(),
            end.to_vec2(),
            start_color,
            end_color,
        ))
    }

    /// Creates a radial-gradient fill centered in logical screen pixels.
    pub fn radial_gradient(
        center: LogicalScreenPosition,
        inner_radius: Option<LogicalPixels>,
        outer_radius: LogicalPixels,
        inner_color: Color,
        outer_color: Color,
    ) -> Fill {
        Fill::RadialGradient(RadialGradient::new(
            center.to_vec2(),
            inner_radius.map_or(0.0, LogicalPixels::get),
            outer_radius.get(),
            inner_color,
            outer_color,
        ))
    }

    #[cfg(feature = "wgpu")]
    pub(crate) const fn as_scene(&self) -> &Scene {
        &self.scene
    }
}

#[cfg(any(feature = "wgpu", test))]
pub(crate) fn screen_camera(viewport: LogicalViewport) -> Result<Camera2d, SceneError> {
    Camera2d::new(
        Vec2::new(viewport.width() * 0.5, viewport.height() * -0.5),
        1.0,
    )
    .map_err(|_| SceneError::NonFiniteGeometry(ScenePrimitive::Rect))
}

fn screen_to_internal(position: LogicalScreenPosition) -> Vec2 {
    screen_vec_to_internal(position.to_vec2())
}

fn screen_vec_to_internal(value: Vec2) -> Vec2 {
    Vec2::new(value.x, -value.y)
}

fn screen_shape_style_to_internal(style: ShapeStyle) -> ShapeStyle {
    let fill = style.fill().map(|fill| match fill {
        Fill::Solid(color) => Fill::Solid(color),
        Fill::LinearGradient(gradient) => Fill::LinearGradient(LinearGradient::new(
            screen_vec_to_internal(gradient.start()),
            screen_vec_to_internal(gradient.end()),
            gradient.start_color(),
            gradient.end_color(),
        )),
        Fill::RadialGradient(gradient) => Fill::RadialGradient(RadialGradient::new(
            screen_vec_to_internal(gradient.center()),
            gradient.inner_radius(),
            gradient.outer_radius(),
            gradient.inner_color(),
            gradient.outer_color(),
        )),
    });
    ShapeStyle::new(fill, style.stroke(), style.shadow())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_screen_geometry_maps_to_top_left_coordinates() {
        let viewport = LogicalViewport::new(800.0, 600.0).unwrap();
        let camera = screen_camera(viewport).unwrap();
        for point in [
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalScreenPosition::new(123.5, 456.25),
            LogicalScreenPosition::new(800.0, 600.0),
        ] {
            let projected = camera
                .world_to_screen(screen_to_internal(point), viewport)
                .unwrap();
            assert_eq!(projected, point);
        }
    }

    #[test]
    fn logical_screen_polyline_checks_budget_before_copying_points() {
        let budget = SceneBudget::new(1, 2, usize::MAX, usize::MAX, usize::MAX, usize::MAX, 1);
        let mut scene = ScreenScene::with_budget(Color::BLACK, budget).unwrap();
        let points = [
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalScreenPosition::new(1.0, 0.0),
            LogicalScreenPosition::new(2.0, 0.0),
        ];

        assert_eq!(
            scene.try_polyline(&points, LogicalPixels::new(1.0).unwrap(), Color::WHITE),
            Err(SceneError::BudgetExceeded {
                resource: SceneBudgetResource::Points,
                limit: 2,
                requested: 3,
            })
        );
        assert_eq!(scene.command_count(), 0);
        assert_eq!(scene.statistics().rejected_commands(), 1);
    }

    #[test]
    fn logical_screen_polyline_checks_allocation_before_copying_points() {
        let budget = SceneBudget::new(1, usize::MAX, usize::MAX, usize::MAX, 1, usize::MAX, 1);
        let mut scene = ScreenScene::with_budget(Color::BLACK, budget).unwrap();
        let points = [
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalScreenPosition::new(1.0, 0.0),
            LogicalScreenPosition::new(2.0, 0.0),
        ];

        assert!(matches!(
            scene.try_polyline(&points, LogicalPixels::new(1.0).unwrap(), Color::WHITE),
            Err(SceneError::BudgetExceeded {
                resource: SceneBudgetResource::AllocationBytes,
                limit: 1,
                ..
            })
        ));
        assert_eq!(scene.command_count(), 0);
        assert_eq!(scene.allocation_bytes(), 0);
        assert_eq!(scene.statistics().rejected_commands(), 1);
    }

    #[test]
    fn logical_screen_polyline_validates_before_caller_sized_allocation() {
        let budget = SceneBudget::new(1, 1, usize::MAX, usize::MAX, 1, usize::MAX, 1);
        let mut scene = ScreenScene::with_budget(Color::BLACK, budget).unwrap();
        let non_finite = [
            LogicalScreenPosition::new(f32::NAN, 0.0),
            LogicalScreenPosition::new(1.0, 0.0),
        ];

        assert_eq!(
            scene.try_polyline(&non_finite, LogicalPixels::new(1.0).unwrap(), Color::WHITE,),
            Err(SceneError::NonFiniteGeometry(ScenePrimitive::Polyline))
        );
        assert_eq!(scene.allocation_bytes(), 0);

        let degenerate = [
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalScreenPosition::new(1.0, 0.0),
        ];
        assert_eq!(
            scene.try_polyline(&degenerate, LogicalPixels::new(1.0).unwrap(), Color::WHITE,),
            Err(SceneError::DegenerateGeometry(ScenePrimitive::Polyline))
        );
        assert_eq!(scene.allocation_bytes(), 0);
        assert_eq!(scene.statistics().rejected_commands(), 2);
    }

    #[test]
    fn logical_screen_styled_paths_reject_world_width_before_allocation() {
        let mut scene = ScreenScene::new(Color::BLACK).unwrap();
        let style = StrokeStyle2d::world(crate::WorldLength::new(2.0).unwrap(), Color::WHITE);
        let start = LogicalScreenPosition::new(0.0, 0.0);
        let end = LogicalScreenPosition::new(8.0, 0.0);

        assert_eq!(
            scene.try_styled_line(start, end, style),
            Err(SceneError::InvalidStroke(ScenePrimitive::Line))
        );
        assert_eq!(
            scene.try_styled_polyline(&[start, end], style),
            Err(SceneError::InvalidStroke(ScenePrimitive::Polyline))
        );
        assert_eq!(scene.command_count(), 0);
        assert_eq!(scene.allocation_bytes(), 0);
        assert_eq!(scene.statistics().rejected_commands(), 2);
    }

    #[test]
    fn logical_screen_rect_requires_positive_size_at_declared_top_left() {
        let mut scene = ScreenScene::new(Color::BLACK).unwrap();
        let min = LogicalScreenPosition::new(100.0, 100.0);
        for size in [
            LogicalScreenVector::new(-50.0, 20.0),
            LogicalScreenVector::new(50.0, -20.0),
            LogicalScreenVector::new(0.0, 20.0),
            LogicalScreenVector::new(50.0, 0.0),
        ] {
            assert_eq!(
                scene.try_rect(
                    min,
                    size,
                    LogicalPixels::new(1.0).unwrap(),
                    ShapeStyle::filled(Color::WHITE),
                ),
                Err(SceneError::InvalidDimension(ScenePrimitive::Rect))
            );
        }
        assert_eq!(scene.command_count(), 0);
        assert_eq!(scene.statistics().rejected_commands(), 4);
    }

    #[test]
    fn logical_screen_square_rect_expresses_an_exact_zero_corner_radius() {
        let mut scene = ScreenScene::new(Color::BLACK).unwrap();
        scene
            .try_square_rect(
                LogicalScreenPosition::new(10.0, 20.0),
                LogicalScreenVector::new(100.0, 50.0),
                ShapeStyle::filled(Color::WHITE),
            )
            .unwrap();

        let crate::DrawCommand::Rect(rect) = scene.scene.commands()[0].command() else {
            panic!("square screen rectangle should retain a rectangle command");
        };
        assert_eq!(rect.corner_radius(), 0.0);
        assert_eq!(rect.rect().min(), Vec2::new(10.0, -20.0));
        assert_eq!(rect.rect().max(), Vec2::new(110.0, -70.0));
    }

    #[test]
    fn screen_shape_gradients_are_converted_exactly_once() {
        let red = Color::rgb8(255, 0, 0);
        let blue = Color::rgb8(0, 0, 255);
        let mut scene = ScreenScene::new(Color::BLACK).unwrap();
        scene
            .try_rect(
                LogicalScreenPosition::new(0.0, 0.0),
                LogicalScreenVector::new(100.0, 100.0),
                LogicalPixels::new(1.0).unwrap(),
                ShapeStyle::filled_with(Fill::LinearGradient(LinearGradient::new(
                    Vec2::ZERO,
                    Vec2::new(0.0, 100.0),
                    red,
                    blue,
                ))),
            )
            .unwrap();
        scene
            .try_circle(
                LogicalScreenPosition::new(20.0, 30.0),
                LogicalPixels::new(10.0).unwrap(),
                ShapeStyle::filled_with(ScreenScene::radial_gradient(
                    LogicalScreenPosition::new(20.0, 30.0),
                    None,
                    LogicalPixels::new(10.0).unwrap(),
                    red,
                    blue,
                )),
            )
            .unwrap();

        let crate::DrawCommand::Rect(rect) = scene.scene.commands()[0].command() else {
            panic!("screen rectangle should retain a rectangle command");
        };
        let Some(Fill::LinearGradient(linear)) = rect.style().fill() else {
            panic!("screen rectangle should retain its linear gradient");
        };
        assert_eq!(linear.start(), Vec2::ZERO);
        assert_eq!(linear.end(), Vec2::new(0.0, -100.0));
        assert_eq!(linear.color_at(Vec2::ZERO), red);
        assert_eq!(linear.color_at(Vec2::new(0.0, -100.0)), blue);

        let crate::DrawCommand::Circle(circle) = scene.scene.commands()[1].command() else {
            panic!("screen circle should retain a circle command");
        };
        let Some(Fill::RadialGradient(radial)) = circle.style().fill() else {
            panic!("screen circle should retain its radial gradient");
        };
        assert_eq!(radial.center(), Vec2::new(20.0, -30.0));
        assert_eq!(radial.color_at(Vec2::new(20.0, -30.0)), red);
        assert_eq!(radial.color_at(Vec2::new(30.0, -30.0)), blue);
    }
}
