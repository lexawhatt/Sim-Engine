use std::{error::Error, fmt};

use crate::{
    math::Vec2,
    tween::{Interpolate, Tween},
};

const MIN_ZOOM: f32 = 0.0001;
const MIN_PROJECTION_COSINE: f32 = 0.001;

/// Point measured in logical screen pixels with a top-left origin.
///
/// Camera picking accepts this type so physical pointer coordinates require an
/// explicit DPI conversion before they can enter camera math.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LogicalScreenPosition {
    value: Vec2,
}

impl LogicalScreenPosition {
    /// Builds a position from logical horizontal and vertical pixel coordinates.
    pub const fn new(x: f32, y: f32) -> Self {
        Self {
            value: Vec2::new(x, y),
        }
    }

    /// Builds a position from a vector whose units are explicitly logical pixels.
    pub const fn from_vec2(value: Vec2) -> Self {
        Self { value }
    }

    /// Returns the position as a logical-pixel vector.
    pub const fn to_vec2(self) -> Vec2 {
        self.value
    }

    /// Returns true when both coordinates are finite.
    pub fn is_finite(self) -> bool {
        self.value.is_finite()
    }
}

/// Point measured in physical surface pixels with a top-left origin.
///
/// Window systems commonly report pointer positions in this space. A renderer
/// converts them to [`LogicalScreenPosition`] using its validated DPI scale.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PhysicalScreenPosition {
    value: Vec2,
}

impl PhysicalScreenPosition {
    /// Builds a position from physical horizontal and vertical pixel coordinates.
    pub const fn new(x: f32, y: f32) -> Self {
        Self {
            value: Vec2::new(x, y),
        }
    }

    /// Builds a position from a vector whose units are explicitly physical pixels.
    pub const fn from_vec2(value: Vec2) -> Self {
        Self { value }
    }

    /// Returns the position as a physical-pixel vector.
    pub const fn to_vec2(self) -> Vec2 {
        self.value
    }

    /// Returns true when both coordinates are finite.
    pub fn is_finite(self) -> bool {
        self.value.is_finite()
    }
}

/// Logical pixel dimensions of the viewport currently being drawn into.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogicalViewport {
    width: f32,
    height: f32,
}

impl LogicalViewport {
    /// Builds a viewport from logical pixel dimensions.
    ///
    /// Both dimensions must be finite and strictly positive.
    pub fn new(width: f32, height: f32) -> Result<Self, LogicalViewportError> {
        if width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0 {
            Ok(Self { width, height })
        } else {
            Err(LogicalViewportError::InvalidDimensions { width, height })
        }
    }

    /// Returns viewport width in logical screen pixels.
    pub fn width(self) -> f32 {
        self.width
    }

    /// Returns viewport height in logical screen pixels.
    pub fn height(self) -> f32 {
        self.height
    }

    /// Returns the center point in logical screen pixel coordinates.
    pub fn center(self) -> LogicalScreenPosition {
        LogicalScreenPosition::new(self.width * 0.5, self.height * 0.5)
    }

    /// Returns viewport size in logical screen pixels.
    pub fn size(self) -> Vec2 {
        Vec2::new(self.width, self.height)
    }
}

/// Invalid logical dimensions supplied to [`LogicalViewport`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogicalViewportError {
    /// Width and height must both be finite and strictly positive.
    InvalidDimensions {
        /// Rejected width in logical screen pixels.
        width: f32,
        /// Rejected height in logical screen pixels.
        height: f32,
    },
}

impl fmt::Display for LogicalViewportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => write!(
                formatter,
                "viewport dimensions must be finite and positive, got {width}x{height}"
            ),
        }
    }
}

impl Error for LogicalViewportError {}

/// Lightweight pseudo-depth projection for 2D scenes.
///
/// This is not a mesh renderer or z-buffer. It offsets 2D points using a scalar
/// depth value so host applications can create camera-like depth transitions
/// without leaving the 2D rendering model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection2d {
    tilt: f32,
    depth_scale: f32,
}

impl Projection2d {
    /// Projection that leaves points in their original 2D plane.
    pub const FLAT: Self = Self {
        tilt: 0.0,
        depth_scale: 1.0,
    };

    /// Builds a projection from tilt in radians and depth conversion scale.
    ///
    /// `depth_scale` is measured in world units per caller-defined depth unit.
    /// Both values must be finite.
    pub fn new(tilt: f32, depth_scale: f32) -> Result<Self, Projection2dError> {
        if tilt.is_finite() && depth_scale.is_finite() {
            Ok(Self { tilt, depth_scale })
        } else {
            Err(Projection2dError::NonFinite { tilt, depth_scale })
        }
    }

    /// Returns projection tilt in radians.
    pub fn tilt(self) -> f32 {
        self.tilt
    }

    /// Returns depth conversion in world units per caller-defined depth unit.
    pub fn depth_scale(self) -> f32 {
        self.depth_scale
    }

    /// Projects a world-space point with caller-provided pseudo-depth.
    pub fn project(self, point: Vec2, depth: f32) -> Vec2 {
        let lifted = depth * self.depth_scale;
        Vec2::new(
            point.x + lifted * self.tilt.sin() * 0.5,
            point.y * self.tilt.cos() + lifted * self.tilt.sin(),
        )
    }
}

/// Invalid pseudo-depth projection parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection2dError {
    /// Tilt and depth scale must both be finite.
    NonFinite {
        /// Rejected projection tilt in radians.
        tilt: f32,
        /// Rejected world units per depth unit.
        depth_scale: f32,
    },
}

impl fmt::Display for Projection2dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { tilt, depth_scale } => write!(
                formatter,
                "projection tilt and depth scale must be finite, got tilt {tilt} and scale {depth_scale}"
            ),
        }
    }
}

impl Error for Projection2dError {}

/// 2D camera mapping world coordinates into logical screen pixels.
///
/// `center` and all input positions are in world units. `zoom` is logical screen
/// pixels per world unit. `rotation` is in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2d {
    center: Vec2,
    zoom: f32,
    rotation: f32,
    projection: Projection2d,
}

/// Invalid camera configuration or inverse projection request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Camera2dError {
    /// Camera center must contain finite world coordinates.
    InvalidCenter {
        /// Rejected world-space center.
        center: Vec2,
    },
    /// Zoom must be finite and strictly positive because it is logical pixels per world unit.
    InvalidZoom {
        /// Rejected zoom value.
        zoom: f32,
    },
    /// Camera rotation must be finite.
    InvalidRotation {
        /// Rejected clockwise rotation in radians.
        rotation: f32,
    },
    /// Screen coordinates and pseudo-depth used for picking must be finite.
    InvalidPickingInput {
        /// Rejected point in logical screen pixels.
        screen: LogicalScreenPosition,
        /// Rejected caller-defined pseudo-depth.
        depth: f32,
    },
    /// The current projection collapses the requested picking plane.
    SingularProjection {
        /// Projection tilt in radians.
        tilt: f32,
    },
}

impl fmt::Display for Camera2dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCenter { center } => {
                write!(formatter, "camera center must be finite, got {center:?}")
            }
            Self::InvalidZoom { zoom } => {
                write!(
                    formatter,
                    "camera zoom must be finite and positive, got {zoom}"
                )
            }
            Self::InvalidRotation { rotation } => {
                write!(formatter, "camera rotation must be finite, got {rotation}")
            }
            Self::InvalidPickingInput { screen, depth } => write!(
                formatter,
                "screen coordinates and picking depth must be finite, got {screen:?} at depth {depth}"
            ),
            Self::SingularProjection { tilt } => {
                write!(
                    formatter,
                    "projection tilt is singular for screen picking: {tilt}"
                )
            }
        }
    }
}

impl Error for Camera2dError {}

impl Camera2d {
    /// Smallest zoom accepted by camera constructors and setters.
    pub const MIN_ZOOM: f32 = MIN_ZOOM;
}

impl Camera2d {
    /// Builds a camera centered on a world-space point.
    ///
    /// `zoom` is logical screen pixels per world unit and must be finite and
    /// positive. Invalid values return [`Camera2dError::InvalidZoom`] instead of
    /// being silently clamped.
    pub fn new(center: Vec2, zoom: f32) -> Result<Self, Camera2dError> {
        validate_center(center)?;
        validate_zoom(zoom)?;

        Ok(Self {
            center,
            zoom,
            rotation: 0.0,
            projection: Projection2d::FLAT,
        })
    }

    /// Returns the world-space point displayed at the viewport center.
    pub fn center(self) -> Vec2 {
        self.center
    }

    /// Replaces the world-space point displayed at the viewport center.
    /// Invalid coordinates leave the current center unchanged.
    pub fn set_center(&mut self, center: Vec2) -> Result<(), Camera2dError> {
        validate_center(center)?;
        self.center = center;
        Ok(())
    }

    /// Returns camera scale in logical screen pixels per world unit.
    pub fn zoom(self) -> f32 {
        self.zoom
    }

    /// Replaces camera scale in logical screen pixels per world unit.
    ///
    /// Invalid values return [`Camera2dError::InvalidZoom`] and leave the
    /// current zoom unchanged.
    pub fn set_zoom(&mut self, zoom: f32) -> Result<(), Camera2dError> {
        validate_zoom(zoom)?;
        self.zoom = zoom;
        Ok(())
    }

    /// Returns clockwise screen rotation in radians.
    pub fn rotation(self) -> f32 {
        self.rotation
    }

    /// Replaces clockwise screen rotation in radians.
    /// Non-finite values leave the current rotation unchanged.
    pub fn set_rotation(&mut self, rotation: f32) -> Result<(), Camera2dError> {
        validate_rotation(rotation)?;
        self.rotation = rotation;
        Ok(())
    }

    /// Returns the pseudo-depth projection applied before camera transform.
    pub fn projection(self) -> Projection2d {
        self.projection
    }

    /// Replaces the pseudo-depth projection applied before camera transform.
    pub fn set_projection(&mut self, projection: Projection2d) {
        self.projection = projection;
    }

    /// Converts a world-space point into logical screen pixel coordinates.
    pub fn world_to_screen(self, world: Vec2, viewport: LogicalViewport) -> LogicalScreenPosition {
        self.projected_world_to_screen(world, 0.0, viewport)
    }

    /// Converts a world-space point and pseudo-depth into logical screen pixels.
    pub fn projected_world_to_screen(
        self,
        world: Vec2,
        depth: f32,
        viewport: LogicalViewport,
    ) -> LogicalScreenPosition {
        let relative = world - self.center;
        let lifted = depth * self.projection.depth_scale;
        let translated = Vec2::new(
            relative.x + lifted * self.projection.tilt.sin() * 0.5,
            relative.y * self.projection.tilt.cos() + lifted * self.projection.tilt.sin(),
        );
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        let rotated = Vec2::new(
            translated.x * cos - translated.y * sin,
            translated.x * sin + translated.y * cos,
        );

        LogicalScreenPosition::new(
            viewport.width() * 0.5 + rotated.x * self.zoom,
            viewport.height() * 0.5 - rotated.y * self.zoom,
        )
    }

    /// Converts logical screen pixel coordinates back into world coordinates.
    ///
    /// This is equivalent to [`Camera2d::screen_to_world_at_depth`] with
    /// `depth = 0.0`.
    pub fn screen_to_world(
        self,
        screen: LogicalScreenPosition,
        viewport: LogicalViewport,
    ) -> Result<Vec2, Camera2dError> {
        self.screen_to_world_at_depth(screen, 0.0, viewport)
    }

    /// Converts logical screen pixel coordinates into world coordinates on a pseudo-depth plane.
    ///
    /// `depth` must match the same caller-defined scalar depth passed to
    /// [`Camera2d::projected_world_to_screen`]. If projection tilt is close to
    /// 90 degrees, the inverse mapping is singular and returns
    /// [`Camera2dError::SingularProjection`].
    pub fn screen_to_world_at_depth(
        self,
        screen: LogicalScreenPosition,
        depth: f32,
        viewport: LogicalViewport,
    ) -> Result<Vec2, Camera2dError> {
        if !screen.is_finite() || !depth.is_finite() {
            return Err(Camera2dError::InvalidPickingInput { screen, depth });
        }

        let screen = screen.to_vec2();

        let translated = Vec2::new(
            (screen.x - viewport.width() * 0.5) / self.zoom,
            -(screen.y - viewport.height() * 0.5) / self.zoom,
        );
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        let projected_delta = Vec2::new(
            translated.x * cos + translated.y * sin,
            -translated.x * sin + translated.y * cos,
        );
        let projection_cos = self.projection.tilt.cos();

        if projection_cos.abs() < MIN_PROJECTION_COSINE {
            return Err(Camera2dError::SingularProjection {
                tilt: self.projection.tilt,
            });
        }

        let lifted = depth * self.projection.depth_scale;
        let relative = Vec2::new(
            projected_delta.x - lifted * self.projection.tilt.sin() * 0.5,
            (projected_delta.y - lifted * self.projection.tilt.sin()) / projection_cos,
        );
        let world = self.center + relative;
        if world.is_finite() {
            Ok(world)
        } else {
            Err(Camera2dError::InvalidPickingInput {
                screen: LogicalScreenPosition::from_vec2(screen),
                depth,
            })
        }
    }

    /// Creates a tween initialized with this camera state.
    pub fn tween(self) -> Tween<Self> {
        Tween::new(self)
    }
}

impl Default for Camera2d {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            zoom: 1.0,
            rotation: 0.0,
            projection: Projection2d::FLAT,
        }
    }
}

impl Interpolate for Projection2d {
    fn interpolate(self, end: Self, amount: f32) -> Self {
        let amount = finite_interpolation_amount(amount);
        let tilt = self.tilt.interpolate(end.tilt, amount);
        let depth_scale = self.depth_scale.interpolate(end.depth_scale, amount);
        Self {
            tilt: finite_or(tilt, self.tilt),
            depth_scale: finite_or(depth_scale, self.depth_scale),
        }
    }
}

impl Interpolate for Camera2d {
    fn interpolate(self, end: Self, amount: f32) -> Self {
        let amount = finite_interpolation_amount(amount);
        let center = self.center.interpolate(end.center, amount);
        let zoom = self.zoom.interpolate(end.zoom, amount);
        let rotation = self.rotation.interpolate(end.rotation, amount);
        Self {
            center: if center.is_finite() {
                center
            } else {
                self.center
            },
            zoom: if zoom.is_finite() {
                zoom.max(MIN_ZOOM)
            } else {
                self.zoom
            },
            rotation: finite_or(rotation, self.rotation),
            projection: self.projection.interpolate(end.projection, amount),
        }
    }
}

fn validate_zoom(zoom: f32) -> Result<(), Camera2dError> {
    if zoom.is_finite() && zoom >= MIN_ZOOM {
        Ok(())
    } else {
        Err(Camera2dError::InvalidZoom { zoom })
    }
}

fn validate_center(center: Vec2) -> Result<(), Camera2dError> {
    if center.is_finite() {
        Ok(())
    } else {
        Err(Camera2dError::InvalidCenter { center })
    }
}

fn validate_rotation(rotation: f32) -> Result<(), Camera2dError> {
    if rotation.is_finite() {
        Ok(())
    } else {
        Err(Camera2dError::InvalidRotation { rotation })
    }
}

fn finite_interpolation_amount(amount: f32) -> f32 {
    if amount.is_finite() { amount } else { 0.0 }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

#[cfg(test)]
mod tests {
    use crate::{
        Camera2d, Camera2dError, Interpolate, LogicalScreenPosition, LogicalViewport,
        LogicalViewportError, PhysicalScreenPosition, Projection2d, Projection2dError, Vec2,
    };

    fn test_viewport() -> LogicalViewport {
        let Ok(viewport) = LogicalViewport::new(1280.0, 720.0) else {
            panic!("test viewport should be valid");
        };
        viewport
    }

    fn test_projection(tilt: f32, depth_scale: f32) -> Projection2d {
        let Ok(projection) = Projection2d::new(tilt, depth_scale) else {
            panic!("test projection should be valid");
        };
        projection
    }

    #[test]
    fn camera_roundtrip_without_projection() {
        let Ok(mut camera) = Camera2d::new(Vec2::new(20.0, -10.0), 2.5) else {
            panic!("test camera should be valid");
        };
        assert!(camera.set_rotation(0.35).is_ok());

        let viewport = test_viewport();
        let world = Vec2::new(44.0, 12.0);

        let screen = camera.world_to_screen(world, viewport);
        let Ok(roundtrip) = camera.screen_to_world(screen, viewport) else {
            panic!("flat projection should be invertible");
        };

        assert!((roundtrip.x - world.x).abs() < 0.001);
        assert!((roundtrip.y - world.y).abs() < 0.001);
    }

    #[test]
    fn camera_roundtrip_at_depth() {
        let Ok(mut camera) = Camera2d::new(Vec2::new(20.0, -10.0), 2.5) else {
            panic!("test camera should be valid");
        };
        assert!(camera.set_rotation(0.35).is_ok());
        camera.set_projection(test_projection(0.5, 1.25));

        let viewport = test_viewport();
        let world = Vec2::new(44.0, 12.0);
        let depth = 9.0;

        let screen = camera.projected_world_to_screen(world, depth, viewport);
        let Ok(roundtrip) = camera.screen_to_world_at_depth(screen, depth, viewport) else {
            panic!("projection should be invertible at this tilt");
        };

        assert!((roundtrip.x - world.x).abs() < 0.001);
        assert!((roundtrip.y - world.y).abs() < 0.001);
    }

    #[test]
    fn camera_rejects_invalid_zoom() {
        let result = Camera2d::new(Vec2::ZERO, 0.0);

        assert!(matches!(
            result,
            Err(Camera2dError::InvalidZoom { zoom }) if zoom == 0.0
        ));
    }

    #[test]
    fn projected_camera_center_stays_at_viewport_center() {
        let center = Vec2::new(0.0, 100.0);
        let Ok(mut camera) = Camera2d::new(center, 2.0) else {
            panic!("test camera should be valid");
        };
        camera.set_projection(test_projection(std::f32::consts::FRAC_PI_3, 1.0));
        assert!(camera.set_rotation(0.4).is_ok());
        let viewport = test_viewport();

        let screen = camera.world_to_screen(center, viewport);

        assert_eq!(screen, viewport.center());
    }

    #[test]
    fn picking_requires_a_logical_screen_position_at_hidpi() {
        let viewport = test_viewport();
        let camera = Camera2d::default();
        let physical_pointer = PhysicalScreenPosition::new(1_280.0, 720.0);
        let logical_pointer = LogicalScreenPosition::from_vec2(physical_pointer.to_vec2() / 2.0);

        let Ok(world) = camera.screen_to_world(logical_pointer, viewport) else {
            panic!("logical center should be pickable");
        };

        assert_eq!(world, Vec2::ZERO);
    }

    #[test]
    fn large_camera_center_uses_relative_projection_math() {
        let center = Vec2::new(2.0e38, 0.0);
        let world = Vec2::new(center.x + 1.0e33, 0.0);
        let Ok(camera) = Camera2d::new(center, 2.0) else {
            panic!("large finite camera should be valid");
        };
        let viewport = test_viewport();

        let screen = camera.world_to_screen(world, viewport);

        assert!(screen.is_finite());
        assert!(screen.to_vec2().x > viewport.center().to_vec2().x);
    }

    #[test]
    fn picking_rejects_non_finite_and_near_singular_inputs() {
        let Ok(mut camera) = Camera2d::new(Vec2::ZERO, 1.0) else {
            panic!("test camera should be valid");
        };
        let viewport = test_viewport();

        assert!(matches!(
            camera.screen_to_world(LogicalScreenPosition::new(f32::NAN, 0.0), viewport),
            Err(Camera2dError::InvalidPickingInput { .. })
        ));

        camera.set_projection(test_projection(std::f32::consts::FRAC_PI_2 - 0.00001, 1.0));
        assert!(matches!(
            camera.screen_to_world(viewport.center(), viewport),
            Err(Camera2dError::SingularProjection { .. })
        ));
    }

    #[test]
    fn camera_projection_and_viewport_reject_non_finite_configuration() {
        assert!(matches!(
            Camera2d::new(Vec2::new(f32::INFINITY, 0.0), 1.0),
            Err(Camera2dError::InvalidCenter { .. })
        ));
        assert!(matches!(
            Projection2d::new(f32::NAN, 1.0),
            Err(Projection2dError::NonFinite { .. })
        ));
        assert!(matches!(
            LogicalViewport::new(0.0, f32::INFINITY),
            Err(LogicalViewportError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn camera_interpolation_preserves_finite_invariants() {
        let Ok(start) = Camera2d::new(Vec2::splat(f32::MAX), f32::MAX) else {
            panic!("finite extreme camera should be valid");
        };
        let Ok(end) = Camera2d::new(Vec2::splat(-f32::MAX), 1.0) else {
            panic!("finite extreme camera should be valid");
        };

        let interpolated = start.interpolate(end, 0.5);

        assert!(interpolated.center().is_finite());
        assert!(interpolated.zoom().is_finite());
    }
}
