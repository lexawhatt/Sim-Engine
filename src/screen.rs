use std::mem::size_of;

#[cfg(any(feature = "wgpu", test))]
use crate::{Camera2d, LogicalViewport};
use crate::{
    Color, Fill, Layer, LinearGradient, LogicalPixels, LogicalScreenPosition, LogicalScreenVector,
    RadialGradient, Rect, Scene, SceneBudget, SceneBudgetResource, SceneCommand, SceneError,
    ScenePrimitive, SceneStatistics, ScreenClipRect, ShapeStyle, Vec2,
};

/// A bounded 2D scene whose geometry is expressed in logical screen pixels.
///
/// The origin is the top-left of the active logical viewport and positive y
/// points downward. Geometry stays fixed while world cameras pan, zoom, rotate,
/// or change pseudo-projection. Physical DPI conversion remains a renderer
/// boundary operation.
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

    /// Returns accepted commands in shared layer and insertion order.
    pub fn commands(&self) -> &[SceneCommand] {
        self.scene.commands()
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
        self.scene
            .try_circle_on_layer(layer, screen_to_internal(center), radius.get(), style)
    }

    /// Appends a rectangle from a logical top-left position and size.
    pub fn try_rect(
        &mut self,
        min: LogicalScreenPosition,
        size: LogicalScreenVector,
        corner_radius: LogicalPixels,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_rect_on_layer(Layer::DEFAULT, min, size, corner_radius, style)
    }

    /// Appends a logical-screen rectangle to an explicit shared draw layer.
    pub fn try_rect_on_layer(
        &mut self,
        layer: Layer,
        min: LogicalScreenPosition,
        size: LogicalScreenVector,
        corner_radius: LogicalPixels,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        let max = min.to_vec2() + size.to_vec2();
        if !min.is_finite() || !size.is_finite() || !max.is_finite() {
            self.scene.record_external_rejection();
            return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Rect));
        }
        self.scene.try_rect_on_layer(
            layer,
            Rect::new(
                screen_vec_to_internal(min.to_vec2()),
                screen_vec_to_internal(max),
            ),
            corner_radius.get(),
            style,
        )
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
        if let Some(budget) = self.scene.budget() {
            let requested = self
                .scene
                .statistics()
                .retained_points()
                .saturating_add(points.len());
            if requested > budget.max_points() {
                self.scene.record_external_rejection();
                return Err(SceneError::BudgetExceeded {
                    resource: SceneBudgetResource::Points,
                    limit: budget.max_points(),
                    requested,
                });
            }
        }
        let mut internal = Vec::new();
        if internal.try_reserve(points.len()).is_err() {
            self.scene.record_external_rejection();
            return Err(SceneError::AllocationFailed {
                requested_bytes: points.len().saturating_mul(size_of::<Vec2>()),
            });
        }
        internal.extend(points.iter().copied().map(screen_to_internal));
        self.scene
            .try_polyline_on_layer(layer, internal, width.get(), color)
    }

    /// Creates a linear-gradient fill using typed logical-screen endpoints.
    pub fn linear_gradient(
        start: LogicalScreenPosition,
        end: LogicalScreenPosition,
        start_color: Color,
        end_color: Color,
    ) -> Fill {
        Fill::LinearGradient(LinearGradient::new(
            screen_to_internal(start),
            screen_to_internal(end),
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
            screen_to_internal(center),
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
        let budget = SceneBudget::new(1, 2, usize::MAX, usize::MAX, usize::MAX, 1);
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
}
