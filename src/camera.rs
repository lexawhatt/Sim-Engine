use std::{error::Error, fmt};

use crate::{
    math::{Rect, Vec2},
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

    /// Projects a camera-relative world-space point with caller-provided pseudo-depth.
    ///
    /// This is the same relative-coordinate operation used by [`Camera2d`].
    /// It is not an absolute-world transform and must not be applied before
    /// [`Camera2d::world_to_screen`], which performs it itself.
    pub fn project_relative(self, point: Vec2, depth: f32) -> Result<Vec2, Projection2dError> {
        if !point.is_finite() || !depth.is_finite() {
            return Err(Projection2dError::InvalidInput { point, depth });
        }
        let lifted = depth * self.depth_scale;
        let projected = Vec2::new(
            point.x + lifted * self.tilt.sin() * 0.5,
            point.y * self.tilt.cos() + lifted * self.tilt.sin(),
        );
        if projected.is_finite() {
            Ok(projected)
        } else {
            Err(Projection2dError::Overflow { point, depth })
        }
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
    /// Relative world coordinates or depth are non-finite.
    InvalidInput {
        /// Rejected camera-relative world point.
        point: Vec2,
        /// Rejected caller-defined depth.
        depth: f32,
    },
    /// Finite inputs overflowed while applying projection.
    Overflow {
        /// Camera-relative world point.
        point: Vec2,
        /// Caller-defined depth.
        depth: f32,
    },
}

impl fmt::Display for Projection2dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { tilt, depth_scale } => write!(
                formatter,
                "projection tilt and depth scale must be finite, got tilt {tilt} and scale {depth_scale}"
            ),
            Self::InvalidInput { point, depth } => write!(
                formatter,
                "projection point and depth must be finite, got {point:?} at depth {depth}"
            ),
            Self::Overflow { point, depth } => write!(
                formatter,
                "projection overflowed for {point:?} at depth {depth}"
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
    /// World coordinates or depth supplied for forward projection are non-finite.
    InvalidProjectionInput {
        /// Rejected world-space point.
        world: Vec2,
        /// Rejected caller-defined pseudo-depth.
        depth: f32,
    },
    /// Finite forward-projection inputs overflowed logical screen coordinates.
    ProjectionOverflow {
        /// World-space point that could not be represented on screen.
        world: Vec2,
        /// Caller-defined pseudo-depth.
        depth: f32,
    },
    /// World-space panning delta was non-finite or overflowed the center.
    InvalidPan {
        /// Rejected world-space delta.
        delta: Vec2,
    },
    /// Cursor-centered zoom needs a finite, strictly positive multiplier.
    InvalidZoomFactor {
        /// Rejected multiplier.
        factor: f32,
    },
    /// Bounds fitting requires finite, non-empty bounds and usable padding.
    InvalidFitBounds {
        /// Rejected world-space bounds.
        bounds: Rect,
        /// Rejected logical-pixel padding.
        padding: f32,
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
            Self::InvalidProjectionInput { world, depth } => write!(
                formatter,
                "world coordinates and projection depth must be finite, got {world:?} at depth {depth}"
            ),
            Self::ProjectionOverflow { world, depth } => write!(
                formatter,
                "world-to-screen projection overflowed for {world:?} at depth {depth}"
            ),
            Self::InvalidPan { delta } => {
                write!(
                    formatter,
                    "camera pan delta must be finite and keep the center finite, got {delta:?}"
                )
            }
            Self::InvalidZoomFactor { factor } => {
                write!(
                    formatter,
                    "camera zoom factor must be finite and positive, got {factor}"
                )
            }
            Self::InvalidFitBounds { bounds, padding } => write!(
                formatter,
                "camera fit bounds must be finite and non-empty with usable padding, got {bounds:?} and padding {padding}"
            ),
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

    /// Returns the pseudo-depth projection applied to camera-relative coordinates.
    pub fn projection(self) -> Projection2d {
        self.projection
    }

    /// Replaces the pseudo-depth projection applied to camera-relative coordinates.
    pub fn set_projection(&mut self, projection: Projection2d) {
        self.projection = projection;
    }

    /// Pans the displayed world center by a finite world-space delta.
    pub fn pan_by(&mut self, delta: Vec2) -> Result<(), Camera2dError> {
        let center = self.center + delta;
        if !delta.is_finite() || !center.is_finite() {
            return Err(Camera2dError::InvalidPan { delta });
        }
        self.center = center;
        Ok(())
    }

    /// Changes zoom while keeping the depth-zero world point under `anchor` fixed.
    ///
    /// `anchor` is measured in logical pixels. The operation is atomic: invalid
    /// factors, singular projections, or overflows leave this camera unchanged.
    pub fn zoom_about_screen(
        &mut self,
        factor: f32,
        anchor: LogicalScreenPosition,
        viewport: LogicalViewport,
    ) -> Result<(), Camera2dError> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(Camera2dError::InvalidZoomFactor { factor });
        }
        let world_before = self.screen_to_world(anchor, viewport)?;
        let zoom = self.zoom * factor;
        validate_zoom(zoom)?;
        let mut candidate = *self;
        candidate.zoom = zoom;
        let world_after = candidate.screen_to_world(anchor, viewport)?;
        let center = candidate.center + (world_before - world_after);
        validate_center(center)?;
        candidate.center = center;
        *self = candidate;
        Ok(())
    }

    /// Centers and zooms the camera so finite non-empty world bounds fit a viewport.
    ///
    /// `padding` is measured in logical pixels on every edge. The extent
    /// calculation accounts for this camera's rotation and depth-zero tilt.
    pub fn fit_to_bounds(
        &mut self,
        bounds: Rect,
        padding: f32,
        viewport: LogicalViewport,
    ) -> Result<(), Camera2dError> {
        let bounds = bounds.normalized();
        if !bounds.min.is_finite()
            || !bounds.max.is_finite()
            || !padding.is_finite()
            || padding < 0.0
        {
            return Err(Camera2dError::InvalidFitBounds { bounds, padding });
        }
        let available = viewport.size() - Vec2::splat(padding * 2.0);
        let size = bounds.size();
        let projection_y = self.projection.tilt.cos().abs();
        let cos = self.rotation.cos().abs();
        let sin = self.rotation.sin().abs();
        let screen_extent = Vec2::new(
            cos * size.x + sin * size.y * projection_y,
            sin * size.x + cos * size.y * projection_y,
        );
        if size.x <= 0.0
            || size.y <= 0.0
            || available.x <= 0.0
            || available.y <= 0.0
            || screen_extent.x <= 0.0
            || screen_extent.y <= 0.0
            || !screen_extent.is_finite()
        {
            return Err(Camera2dError::InvalidFitBounds { bounds, padding });
        }
        let zoom = (available.x / screen_extent.x).min(available.y / screen_extent.y);
        validate_zoom(zoom)?;
        let center = bounds.center();
        validate_center(center)?;
        self.center = center;
        self.zoom = zoom;
        Ok(())
    }

    /// Converts a world-space point into logical screen pixel coordinates.
    ///
    /// Returns an error if finite values overflow during the transform.
    pub fn world_to_screen(
        self,
        world: Vec2,
        viewport: LogicalViewport,
    ) -> Result<LogicalScreenPosition, Camera2dError> {
        self.projected_world_to_screen(world, 0.0, viewport)
    }

    /// Converts a world-space point and pseudo-depth into logical screen pixels.
    pub fn projected_world_to_screen(
        self,
        world: Vec2,
        depth: f32,
        viewport: LogicalViewport,
    ) -> Result<LogicalScreenPosition, Camera2dError> {
        if !world.is_finite() || !depth.is_finite() {
            return Err(Camera2dError::InvalidProjectionInput { world, depth });
        }
        let relative = world - self.center;
        let translated = self
            .projection
            .project_relative(relative, depth)
            .map_err(|_| Camera2dError::ProjectionOverflow { world, depth })?;
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        let rotated = Vec2::new(
            translated.x * cos - translated.y * sin,
            translated.x * sin + translated.y * cos,
        );

        let screen = LogicalScreenPosition::new(
            viewport.width() * 0.5 + rotated.x * self.zoom,
            viewport.height() * 0.5 - rotated.y * self.zoom,
        );
        if screen.is_finite() {
            Ok(screen)
        } else {
            Err(Camera2dError::ProjectionOverflow { world, depth })
        }
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

        let screen = camera.world_to_screen(world, viewport).unwrap();
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

        let screen = camera
            .projected_world_to_screen(world, depth, viewport)
            .unwrap();
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

        let screen = camera.world_to_screen(center, viewport).unwrap();

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

        let screen = camera.world_to_screen(world, viewport).unwrap();

        assert!(screen.is_finite());
        assert!(screen.to_vec2().x > viewport.center().to_vec2().x);
    }

    #[test]
    fn forward_projection_reports_finite_overflow() {
        let camera = Camera2d::new(Vec2::ZERO, f32::MAX).unwrap();
        let viewport = test_viewport();

        assert!(matches!(
            camera.world_to_screen(Vec2::new(f32::MAX, 0.0), viewport),
            Err(Camera2dError::ProjectionOverflow { .. })
        ));
    }

    #[test]
    fn pan_cursor_zoom_and_fit_preserve_coordinate_contracts() {
        let viewport = test_viewport();
        let mut camera = Camera2d::new(Vec2::new(5.0, -3.0), 2.0).unwrap();
        camera.pan_by(Vec2::new(4.0, 7.0)).unwrap();
        assert_eq!(camera.center(), Vec2::new(9.0, 4.0));

        let anchor = LogicalScreenPosition::new(192.0, 531.0);
        let world_before = camera.screen_to_world(anchor, viewport).unwrap();
        camera.zoom_about_screen(1.75, anchor, viewport).unwrap();
        let world_after = camera.screen_to_world(anchor, viewport).unwrap();
        assert!((world_before.x - world_after.x).abs() < 0.0001);
        assert!((world_before.y - world_after.y).abs() < 0.0001);

        camera.set_rotation(0.35).unwrap();
        camera.set_projection(test_projection(0.4, 1.0));
        let bounds = crate::Rect::from_center_size(Vec2::new(30.0, -20.0), Vec2::new(80.0, 40.0));
        camera.fit_to_bounds(bounds, 24.0, viewport).unwrap();
        assert_eq!(camera.center(), bounds.center());
        for corner in [bounds.min, bounds.max] {
            let screen = camera.world_to_screen(corner, viewport).unwrap().to_vec2();
            assert!(screen.x >= 24.0 && screen.x <= viewport.width() - 24.0);
            assert!(screen.y >= 24.0 && screen.y <= viewport.height() - 24.0);
        }
    }

    #[test]
    fn camera_helpers_reject_invalid_inputs_without_mutation() {
        let viewport = test_viewport();
        let mut camera = Camera2d::default();
        let before = camera;
        assert!(matches!(
            camera.pan_by(Vec2::new(f32::NAN, 0.0)),
            Err(Camera2dError::InvalidPan { .. })
        ));
        assert_eq!(camera, before);
        assert!(matches!(
            camera.zoom_about_screen(0.0, viewport.center(), viewport),
            Err(Camera2dError::InvalidZoomFactor { .. })
        ));
        assert_eq!(camera, before);
        assert!(matches!(
            camera.fit_to_bounds(
                crate::Rect::from_center_size(Vec2::ZERO, Vec2::new(0.0, 1.0)),
                0.0,
                viewport
            ),
            Err(Camera2dError::InvalidFitBounds { .. })
        ));
        assert_eq!(camera, before);
    }

    #[test]
    fn relative_projection_matches_camera_centered_pipeline() {
        let center = Vec2::new(10.0, 100.0);
        let mut camera = Camera2d::new(center, 2.0).unwrap();
        let projection = test_projection(0.6, 1.25);
        camera.set_projection(projection);
        let viewport = test_viewport();

        assert_eq!(
            projection.project_relative(Vec2::ZERO, 0.0).unwrap(),
            Vec2::ZERO
        );
        assert_eq!(
            camera
                .projected_world_to_screen(center, 0.0, viewport)
                .unwrap(),
            viewport.center()
        );
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
