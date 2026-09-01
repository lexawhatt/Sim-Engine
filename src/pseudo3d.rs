use std::{error::Error, fmt};

use crate::{LogicalScreenPosition, LogicalViewport, WorldLength};

/// Finite vector or point in a right-handed 3D coordinate space.
///
/// Construction and arithmetic are fallible so finite inputs cannot silently
/// produce NaN or infinity at the visual-state boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    /// Origin or zero-length vector.
    pub const ZERO: Self = Self::new_unchecked(0.0, 0.0, 0.0);
    /// Positive X unit axis.
    pub const X: Self = Self::new_unchecked(1.0, 0.0, 0.0);
    /// Positive Y unit axis.
    pub const Y: Self = Self::new_unchecked(0.0, 1.0, 0.0);
    /// Positive Z unit axis.
    pub const Z: Self = Self::new_unchecked(0.0, 0.0, 1.0);

    /// Builds a vector from finite caller-defined world components.
    pub fn new(x: f32, y: f32, z: f32) -> Result<Self, Pseudo3dError> {
        if x.is_finite() && y.is_finite() && z.is_finite() {
            Ok(Self::new_unchecked(x, y, z))
        } else {
            Err(Pseudo3dError::NonFiniteVector { x, y, z })
        }
    }

    pub(crate) const fn new_unchecked(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn from_f64(x: f64, y: f64, z: f64) -> Result<Self, Pseudo3dError> {
        if x.is_finite()
            && y.is_finite()
            && z.is_finite()
            && x.abs() <= f32::MAX as f64
            && y.abs() <= f32::MAX as f64
            && z.abs() <= f32::MAX as f64
        {
            Ok(Self::new_unchecked(x as f32, y as f32, z as f32))
        } else {
            Err(Pseudo3dError::ArithmeticOverflow)
        }
    }

    /// Returns the X component in caller-defined world units.
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Returns the Y component in caller-defined world units.
    pub const fn y(self) -> f32 {
        self.y
    }

    /// Returns the Z component in caller-defined world units.
    pub const fn z(self) -> f32 {
        self.z
    }

    /// Adds two vectors while rejecting a finite-input overflow.
    pub fn checked_add(self, other: Self) -> Result<Self, Pseudo3dError> {
        Self::from_f64(
            self.x as f64 + other.x as f64,
            self.y as f64 + other.y as f64,
            self.z as f64 + other.z as f64,
        )
    }

    /// Subtracts two vectors while rejecting a finite-input overflow.
    pub fn checked_sub(self, other: Self) -> Result<Self, Pseudo3dError> {
        Self::from_f64(
            self.x as f64 - other.x as f64,
            self.y as f64 - other.y as f64,
            self.z as f64 - other.z as f64,
        )
    }

    /// Multiplies every component by a finite scalar.
    pub fn checked_scale(self, scale: f32) -> Result<Self, Pseudo3dError> {
        if !scale.is_finite() {
            return Err(Pseudo3dError::NonFiniteScalar { value: scale });
        }
        Self::from_f64(
            self.x as f64 * scale as f64,
            self.y as f64 * scale as f64,
            self.z as f64 * scale as f64,
        )
    }

    /// Computes a dot product using wider intermediates and rejects an f32 overflow.
    pub fn checked_dot(self, other: Self) -> Result<f32, Pseudo3dError> {
        f64_to_f32(
            self.x as f64 * other.x as f64
                + self.y as f64 * other.y as f64
                + self.z as f64 * other.z as f64,
        )
    }

    /// Computes a cross product using wider intermediates.
    pub fn checked_cross(self, other: Self) -> Result<Self, Pseudo3dError> {
        Self::from_f64(
            self.y as f64 * other.z as f64 - self.z as f64 * other.y as f64,
            self.z as f64 * other.x as f64 - self.x as f64 * other.z as f64,
            self.x as f64 * other.y as f64 - self.y as f64 * other.x as f64,
        )
    }

    /// Returns a unit vector without overflowing on large finite components.
    pub fn normalized(self) -> Result<Self, Pseudo3dError> {
        let length = length_f64(self);
        if length == 0.0 {
            return Err(Pseudo3dError::ZeroLengthVector);
        }
        Self::from_f64(
            self.x as f64 / length,
            self.y as f64 / length,
            self.z as f64 / length,
        )
    }
}

/// Normalized 3D rotation applied in a right-handed coordinate system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation3d {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Rotation3d {
    /// Rotation that leaves vectors unchanged.
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// Builds a rotation around a non-zero finite axis by radians.
    pub fn from_axis_angle(axis: Vec3, radians: f32) -> Result<Self, Pseudo3dError> {
        if !radians.is_finite() {
            return Err(Pseudo3dError::NonFiniteAngle { radians });
        }
        let axis = axis.normalized()?;
        let (sine, cosine) = (radians as f64 * 0.5).sin_cos();
        Self::normalized_from_f64(
            axis.x as f64 * sine,
            axis.y as f64 * sine,
            axis.z as f64 * sine,
            cosine,
        )
    }

    /// Builds a rotation that applies X, then Y, then Z rotations in radians.
    pub fn from_euler_xyz(x: f32, y: f32, z: f32) -> Result<Self, Pseudo3dError> {
        let x_rotation = Self::from_axis_angle(Vec3::X, x)?;
        let y_rotation = Self::from_axis_angle(Vec3::Y, y)?;
        let z_rotation = Self::from_axis_angle(Vec3::Z, z)?;
        x_rotation.then(y_rotation)?.then(z_rotation)
    }

    /// Composes rotations so `self` is applied first and `next` second.
    pub fn then(self, next: Self) -> Result<Self, Pseudo3dError> {
        let left = next;
        let right = self;
        Self::normalized_from_f64(
            left.w as f64 * right.x as f64
                + left.x as f64 * right.w as f64
                + left.y as f64 * right.z as f64
                - left.z as f64 * right.y as f64,
            left.w as f64 * right.y as f64 - left.x as f64 * right.z as f64
                + left.y as f64 * right.w as f64
                + left.z as f64 * right.x as f64,
            left.w as f64 * right.z as f64 + left.x as f64 * right.y as f64
                - left.y as f64 * right.x as f64
                + left.z as f64 * right.w as f64,
            left.w as f64 * right.w as f64
                - left.x as f64 * right.x as f64
                - left.y as f64 * right.y as f64
                - left.z as f64 * right.z as f64,
        )
    }

    /// Rotates a finite vector and rejects arithmetic overflow.
    pub fn rotate(self, vector: Vec3) -> Result<Vec3, Pseudo3dError> {
        let qx = self.x as f64;
        let qy = self.y as f64;
        let qz = self.z as f64;
        let qw = self.w as f64;
        let vx = vector.x as f64;
        let vy = vector.y as f64;
        let vz = vector.z as f64;

        let tx = 2.0 * (qy * vz - qz * vy);
        let ty = 2.0 * (qz * vx - qx * vz);
        let tz = 2.0 * (qx * vy - qy * vx);
        Vec3::from_f64(
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        )
    }

    fn normalized_from_f64(x: f64, y: f64, z: f64, w: f64) -> Result<Self, Pseudo3dError> {
        let length = (x * x + y * y + z * z + w * w).sqrt();
        if !length.is_finite() || length == 0.0 {
            return Err(Pseudo3dError::ArithmeticOverflow);
        }
        let x = f64_to_f32(x / length)?;
        let y = f64_to_f32(y / length)?;
        let z = f64_to_f32(z / length)?;
        let w = f64_to_f32(w / length)?;
        Ok(Self { x, y, z, w })
    }
}

/// Model transform with translation, rotation, and positive per-axis scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform3d {
    translation: Vec3,
    rotation: Rotation3d,
    scale: Vec3,
}

impl Transform3d {
    /// Transform that leaves model-space points unchanged.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Rotation3d::IDENTITY,
        scale: Vec3::new_unchecked(1.0, 1.0, 1.0),
    };

    /// Builds a model transform. Every scale component must be positive.
    pub fn new(
        translation: Vec3,
        rotation: Rotation3d,
        scale: Vec3,
    ) -> Result<Self, Pseudo3dError> {
        if scale.x <= 0.0 || scale.y <= 0.0 || scale.z <= 0.0 {
            return Err(Pseudo3dError::InvalidScale { scale });
        }
        Ok(Self {
            translation,
            rotation,
            scale,
        })
    }

    /// Builds a transform with the same positive finite scale on every axis.
    pub fn from_rotation_scale(rotation: Rotation3d, scale: f32) -> Result<Self, Pseudo3dError> {
        let scale = Vec3::new(scale, scale, scale)?;
        Self::new(Vec3::ZERO, rotation, scale)
    }

    /// Returns model translation in world units.
    pub const fn translation(self) -> Vec3 {
        self.translation
    }

    /// Returns model rotation.
    pub const fn rotation(self) -> Rotation3d {
        self.rotation
    }

    /// Returns positive per-axis model scale.
    pub const fn scale(self) -> Vec3 {
        self.scale
    }

    /// Applies scale, then rotation, then translation to a model-space point.
    pub fn transform_point(self, point: Vec3) -> Result<Vec3, Pseudo3dError> {
        let scaled = Vec3::from_f64(
            point.x as f64 * self.scale.x as f64,
            point.y as f64 * self.scale.y as f64,
            point.z as f64 * self.scale.z as f64,
        )?;
        self.rotation.rotate(scaled)?.checked_add(self.translation)
    }

    #[cfg(feature = "wgpu")]
    pub(crate) fn model_rows(self) -> Result<[[f32; 4]; 3], Pseudo3dError> {
        let x = self.rotation.x as f64;
        let y = self.rotation.y as f64;
        let z = self.rotation.z as f64;
        let w = self.rotation.w as f64;
        let scale_x = self.scale.x as f64;
        let scale_y = self.scale.y as f64;
        let scale_z = self.scale.z as f64;
        let rows = [
            [
                (1.0 - 2.0 * (y * y + z * z)) * scale_x,
                (2.0 * (x * y - z * w)) * scale_y,
                (2.0 * (x * z + y * w)) * scale_z,
                self.translation.x as f64,
            ],
            [
                (2.0 * (x * y + z * w)) * scale_x,
                (1.0 - 2.0 * (x * x + z * z)) * scale_y,
                (2.0 * (y * z - x * w)) * scale_z,
                self.translation.y as f64,
            ],
            [
                (2.0 * (x * z - y * w)) * scale_x,
                (2.0 * (y * z + x * w)) * scale_y,
                (1.0 - 2.0 * (x * x + y * y)) * scale_z,
                self.translation.z as f64,
            ],
        ];
        Ok([
            f64_row_to_f32(rows[0])?,
            f64_row_to_f32(rows[1])?,
            f64_row_to_f32(rows[2])?,
        ])
    }
}

/// Perspective or orthographic camera projection with a finite depth range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection3d {
    kind: Projection3dKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Projection3dKind {
    Perspective {
        vertical_fov_radians: f32,
        aspect_ratio: f32,
        near: WorldLength,
        far: WorldLength,
    },
    Orthographic {
        vertical_span: WorldLength,
        aspect_ratio: f32,
        near: WorldLength,
        far: WorldLength,
    },
}

impl Projection3d {
    /// Builds a perspective projection.
    ///
    /// The vertical field of view is in radians and must be in `0..PI`.
    /// Aspect ratio and near distance must be positive; far must exceed near.
    pub fn perspective(
        vertical_fov_radians: f32,
        aspect_ratio: f32,
        near: WorldLength,
        far: WorldLength,
    ) -> Result<Self, Pseudo3dError> {
        if !valid_projection_range(aspect_ratio, near.get(), far.get())
            || !vertical_fov_radians.is_finite()
            || vertical_fov_radians <= 0.0
            || vertical_fov_radians >= std::f32::consts::PI
            || !(vertical_fov_radians as f64 * 0.5).tan().is_finite()
        {
            return Err(Pseudo3dError::InvalidProjection);
        }
        Ok(Self {
            kind: Projection3dKind::Perspective {
                vertical_fov_radians,
                aspect_ratio,
                near,
                far,
            },
        })
    }

    /// Builds an orthographic projection from its vertical world-space span.
    pub fn orthographic(
        vertical_span: WorldLength,
        aspect_ratio: f32,
        near: WorldLength,
        far: WorldLength,
    ) -> Result<Self, Pseudo3dError> {
        if !valid_projection_range(aspect_ratio, near.get(), far.get()) {
            return Err(Pseudo3dError::InvalidProjection);
        }
        Ok(Self {
            kind: Projection3dKind::Orthographic {
                vertical_span,
                aspect_ratio,
                near,
                far,
            },
        })
    }

    /// Returns the viewport width-to-height ratio.
    pub const fn aspect_ratio(self) -> f32 {
        match self.kind {
            Projection3dKind::Perspective { aspect_ratio, .. }
            | Projection3dKind::Orthographic { aspect_ratio, .. } => aspect_ratio,
        }
    }

    /// Returns the nearest visible positive view-space depth.
    pub const fn near(self) -> WorldLength {
        match self.kind {
            Projection3dKind::Perspective { near, .. }
            | Projection3dKind::Orthographic { near, .. } => near,
        }
    }

    /// Returns the farthest visible positive view-space depth.
    pub const fn far(self) -> WorldLength {
        match self.kind {
            Projection3dKind::Perspective { far, .. }
            | Projection3dKind::Orthographic { far, .. } => far,
        }
    }

    /// Returns true for perspective and false for orthographic projection.
    pub const fn is_perspective(self) -> bool {
        matches!(self.kind, Projection3dKind::Perspective { .. })
    }
}

/// Validated 3D view and projection used for stereometry rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera3d {
    position: Vec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
    projection: Projection3d,
}

impl Camera3d {
    /// Builds a camera looking from `position` toward `target`.
    ///
    /// `up_hint` must not be zero or parallel to the view direction.
    pub fn look_at(
        position: Vec3,
        target: Vec3,
        up_hint: Vec3,
        projection: Projection3d,
    ) -> Result<Self, Pseudo3dError> {
        let forward = target.checked_sub(position)?.normalized()?;
        let right = cross_f64(forward, up_hint)
            .and_then(normalize_f64)
            .ok_or(Pseudo3dError::InvalidCameraBasis)?;
        let up = cross_f64(right, forward)
            .and_then(normalize_f64)
            .ok_or(Pseudo3dError::InvalidCameraBasis)?;
        Ok(Self {
            position,
            right,
            up,
            forward,
            projection,
        })
    }

    /// Returns camera position in world units.
    pub const fn position(self) -> Vec3 {
        self.position
    }

    /// Returns the current perspective or orthographic projection.
    pub const fn projection(self) -> Projection3d {
        self.projection
    }

    /// Projects a world point into logical screen coordinates.
    ///
    /// Points behind the camera return an error. Outside-frustum points whose
    /// projection remains representable return finite coordinates with
    /// `inside_view` set to false, allowing host-side classification without
    /// losing the projected anchor; extreme finite inputs may instead return
    /// [`Pseudo3dError::ArithmeticOverflow`]. In the v0.2 retained renderer,
    /// explicit display edges are clipped while partially clipped surface
    /// triangles are rejected fail-closed as unportable topology.
    pub fn project_world(
        self,
        point: Vec3,
        viewport: LogicalViewport,
    ) -> Result<ProjectedPoint3d, Pseudo3dError> {
        let relative = subtract_f64(point, self.position);
        let view_x = dot_f64_tuple(relative, self.right);
        let view_y = dot_f64_tuple(relative, self.up);
        let view_depth = dot_f64_tuple(relative, self.forward);
        if !view_depth.is_finite() || view_depth <= 0.0 {
            return Err(Pseudo3dError::PointBehindCamera);
        }

        let (ndc_x, ndc_y, normalized_depth, near, far) = match self.projection.kind {
            Projection3dKind::Perspective {
                vertical_fov_radians,
                aspect_ratio,
                near,
                far,
            } => {
                let near_value = near.get();
                let far_value = far.get();
                let half_height = view_depth * (vertical_fov_radians as f64 * 0.5).tan();
                (
                    view_x / (half_height * aspect_ratio as f64),
                    view_y / half_height,
                    far_value as f64 / (far_value as f64 - near_value as f64)
                        - (far_value as f64 * near_value as f64)
                            / ((far_value as f64 - near_value as f64) * view_depth),
                    near,
                    far,
                )
            }
            Projection3dKind::Orthographic {
                vertical_span,
                aspect_ratio,
                near,
                far,
            } => {
                let near_value = near.get();
                let far_value = far.get();
                let normalized_depth =
                    (view_depth - near_value as f64) / (far_value as f64 - near_value as f64);
                (
                    view_x / (vertical_span.get() as f64 * aspect_ratio as f64 * 0.5),
                    view_y / (vertical_span.get() as f64 * 0.5),
                    normalized_depth,
                    near,
                    far,
                )
            }
        };

        let screen_x = (ndc_x * 0.5 + 0.5) * viewport.width() as f64;
        let screen_y = (0.5 - ndc_y * 0.5) * viewport.height() as f64;
        let screen_x = f64_to_f32(screen_x)?;
        let screen_y = f64_to_f32(screen_y)?;
        let view_depth_f32 = f64_to_f32(view_depth)?;
        let normalized_depth_f32 = f64_to_f32(normalized_depth)?;
        let inside_view = view_depth >= near.get() as f64
            && view_depth <= far.get() as f64
            && (-1.0..=1.0).contains(&ndc_x)
            && (-1.0..=1.0).contains(&ndc_y);
        Ok(ProjectedPoint3d {
            logical_position: LogicalScreenPosition::new(screen_x, screen_y),
            view_depth: view_depth_f32,
            normalized_depth: normalized_depth_f32,
            inside_view,
        })
    }

    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn world_to_clip_rows(self) -> Result<[[f32; 4]; 4], Pseudo3dError> {
        let right_offset = -dot_f64(self.position, self.right);
        let up_offset = -dot_f64(self.position, self.up);
        let depth_offset = -dot_f64(self.position, self.forward);
        let rows = match self.projection.kind {
            Projection3dKind::Perspective {
                vertical_fov_radians,
                aspect_ratio,
                near,
                far,
            } => {
                let near = near.get() as f64;
                let far = far.get() as f64;
                let vertical_scale = (vertical_fov_radians as f64 * 0.5).tan();
                let horizontal_scale = vertical_scale * aspect_ratio as f64;
                let depth_scale = far / (far - near);
                let depth_translation = -(far * near) / (far - near);
                [
                    camera_row(self.right, right_offset, 1.0 / horizontal_scale),
                    camera_row(self.up, up_offset, 1.0 / vertical_scale),
                    camera_row(
                        self.forward,
                        depth_offset + depth_translation / depth_scale,
                        depth_scale,
                    ),
                    [
                        self.forward.x as f64,
                        self.forward.y as f64,
                        self.forward.z as f64,
                        depth_offset,
                    ],
                ]
            }
            Projection3dKind::Orthographic {
                vertical_span,
                aspect_ratio,
                near,
                far,
            } => {
                let near = near.get() as f64;
                let far = far.get() as f64;
                let vertical_span = vertical_span.get() as f64;
                let depth_extent = far - near;
                [
                    camera_row(
                        self.right,
                        right_offset,
                        2.0 / (vertical_span * aspect_ratio as f64),
                    ),
                    camera_row(self.up, up_offset, 2.0 / vertical_span),
                    camera_row(self.forward, depth_offset - near, 1.0 / depth_extent),
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
        };
        Ok([
            f64_row_to_f32(rows[0])?,
            f64_row_to_f32(rows[1])?,
            f64_row_to_f32(rows[2])?,
            f64_row_to_f32(rows[3])?,
        ])
    }
}

/// Result of projecting a 3D point into a logical viewport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedPoint3d {
    logical_position: LogicalScreenPosition,
    view_depth: f32,
    normalized_depth: f32,
    inside_view: bool,
}

impl ProjectedPoint3d {
    /// Returns the point in logical screen pixels with a top-left origin.
    pub const fn logical_position(self) -> LogicalScreenPosition {
        self.logical_position
    }

    /// Returns positive distance along the camera forward axis in world units.
    pub const fn view_depth(self) -> f32 {
        self.view_depth
    }

    /// Returns the hardware depth value, normally in `0.0..=1.0`.
    ///
    /// Orthographic depth is linear. Perspective depth follows the nonlinear
    /// projection used by the GPU depth attachment so CPU anchors and rendered
    /// surfaces share one visibility coordinate.
    pub const fn normalized_depth(self) -> f32 {
        self.normalized_depth
    }

    /// Returns whether the point lies inside every configured clip boundary.
    pub const fn inside_view(self) -> bool {
        self.inside_view
    }
}

/// Invalid pseudo-3D visual state or arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pseudo3dError {
    /// A 3D vector component was NaN or infinite.
    NonFiniteVector {
        /// Rejected X component.
        x: f32,
        /// Rejected Y component.
        y: f32,
        /// Rejected Z component.
        z: f32,
    },
    /// A scalar input was NaN or infinite.
    NonFiniteScalar {
        /// Rejected scalar.
        value: f32,
    },
    /// A rotation angle was NaN or infinite.
    NonFiniteAngle {
        /// Rejected angle in radians.
        radians: f32,
    },
    /// A direction or rotation axis had zero length.
    ZeroLengthVector,
    /// Model scale contained a zero or negative component.
    InvalidScale {
        /// Rejected per-axis scale.
        scale: Vec3,
    },
    /// Projection parameters did not define a finite usable frustum.
    InvalidProjection,
    /// View direction and up hint could not form an orthonormal camera basis.
    InvalidCameraBasis,
    /// A point lies on or behind the camera plane.
    PointBehindCamera,
    /// Finite inputs produced a value outside the representable f32 range.
    ArithmeticOverflow,
}

impl fmt::Display for Pseudo3dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteVector { x, y, z } => {
                write!(formatter, "3D vector must be finite, got ({x}, {y}, {z})")
            }
            Self::NonFiniteScalar { value } => {
                write!(formatter, "3D scalar must be finite, got {value}")
            }
            Self::NonFiniteAngle { radians } => {
                write!(formatter, "3D rotation angle must be finite, got {radians}")
            }
            Self::ZeroLengthVector => write!(formatter, "3D direction must have non-zero length"),
            Self::InvalidScale { scale } => {
                write!(formatter, "3D model scale must be positive, got {scale:?}")
            }
            Self::InvalidProjection => write!(formatter, "3D projection parameters are invalid"),
            Self::InvalidCameraBasis => write!(formatter, "3D camera basis is degenerate"),
            Self::PointBehindCamera => write!(formatter, "3D point is behind the camera"),
            Self::ArithmeticOverflow => write!(formatter, "3D arithmetic overflowed f32 output"),
        }
    }
}

impl Error for Pseudo3dError {}

fn valid_projection_range(aspect_ratio: f32, near: f32, far: f32) -> bool {
    aspect_ratio.is_finite()
        && aspect_ratio > 0.0
        && near.is_finite()
        && near > 0.0
        && far.is_finite()
        && far > near
        && (far as f64 - near as f64) <= f32::MAX as f64
}

fn f64_to_f32(value: f64) -> Result<f32, Pseudo3dError> {
    if value.is_finite() && value.abs() <= f32::MAX as f64 {
        Ok(value as f32)
    } else {
        Err(Pseudo3dError::ArithmeticOverflow)
    }
}

fn length_f64(vector: Vec3) -> f64 {
    let x = vector.x as f64;
    let y = vector.y as f64;
    let z = vector.z as f64;
    (x * x + y * y + z * z).sqrt()
}

fn subtract_f64(left: Vec3, right: Vec3) -> (f64, f64, f64) {
    (
        left.x as f64 - right.x as f64,
        left.y as f64 - right.y as f64,
        left.z as f64 - right.z as f64,
    )
}

fn dot_f64_tuple(left: (f64, f64, f64), right: Vec3) -> f64 {
    left.0 * right.x as f64 + left.1 * right.y as f64 + left.2 * right.z as f64
}

#[cfg(any(feature = "wgpu", test))]
fn dot_f64(left: Vec3, right: Vec3) -> f64 {
    left.x as f64 * right.x as f64 + left.y as f64 * right.y as f64 + left.z as f64 * right.z as f64
}

#[cfg(any(feature = "wgpu", test))]
fn camera_row(axis: Vec3, offset: f64, scale: f64) -> [f64; 4] {
    [
        axis.x as f64 * scale,
        axis.y as f64 * scale,
        axis.z as f64 * scale,
        offset * scale,
    ]
}

#[cfg(any(feature = "wgpu", test))]
fn f64_row_to_f32(row: [f64; 4]) -> Result<[f32; 4], Pseudo3dError> {
    Ok([
        f64_to_f32(row[0])?,
        f64_to_f32(row[1])?,
        f64_to_f32(row[2])?,
        f64_to_f32(row[3])?,
    ])
}

fn cross_f64(left: Vec3, right: Vec3) -> Option<(f64, f64, f64)> {
    let result = (
        left.y as f64 * right.z as f64 - left.z as f64 * right.y as f64,
        left.z as f64 * right.x as f64 - left.x as f64 * right.z as f64,
        left.x as f64 * right.y as f64 - left.y as f64 * right.x as f64,
    );
    result.0.is_finite().then_some(result)
}

fn normalize_f64(vector: (f64, f64, f64)) -> Option<Vec3> {
    let length = (vector.0 * vector.0 + vector.1 * vector.1 + vector.2 * vector.2).sqrt();
    if !length.is_finite() || length == 0.0 {
        return None;
    }
    Vec3::from_f64(vector.0 / length, vector.1 / length, vector.2 / length).ok()
}

#[cfg(test)]
mod tests {
    use super::{Camera3d, Projection3d, Pseudo3dError, Rotation3d, Transform3d, Vec3};
    use crate::{LogicalViewport, WorldLength};

    fn vector(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).unwrap()
    }

    fn world(value: f32) -> WorldLength {
        WorldLength::new(value).unwrap()
    }

    fn close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 0.0001, "{actual} != {expected}");
    }

    #[test]
    fn vector_normalization_preserves_extreme_finite_direction() {
        let normalized = vector(f32::MAX, -f32::MAX, f32::MAX).normalized().unwrap();
        let expected = 1.0 / 3.0_f32.sqrt();
        close(normalized.x(), expected);
        close(normalized.y(), -expected);
        close(normalized.z(), expected);
        assert_eq!(
            Vec3::ZERO.normalized(),
            Err(Pseudo3dError::ZeroLengthVector)
        );
    }

    #[test]
    fn rotation_composition_uses_documented_application_order() {
        let quarter_turn = std::f32::consts::FRAC_PI_2;
        let rotation = Rotation3d::from_euler_xyz(quarter_turn, quarter_turn, 0.0).unwrap();
        let rotated = rotation.rotate(Vec3::Y).unwrap();
        close(rotated.x(), 1.0);
        close(rotated.y(), 0.0);
        close(rotated.z(), 0.0);
    }

    #[test]
    fn transform_rejects_invalid_scale_and_keeps_point_finite() {
        let rotation = Rotation3d::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2).unwrap();
        assert!(matches!(
            Transform3d::new(Vec3::ZERO, rotation, vector(1.0, 0.0, 1.0)),
            Err(Pseudo3dError::InvalidScale { .. })
        ));
        let transform =
            Transform3d::new(vector(3.0, 4.0, 5.0), rotation, vector(2.0, 2.0, 2.0)).unwrap();
        let point = transform.transform_point(Vec3::X).unwrap();
        close(point.x(), 3.0);
        close(point.y(), 6.0);
        close(point.z(), 5.0);
    }

    #[test]
    fn perspective_camera_projects_center_and_reports_clip_membership() {
        let viewport = LogicalViewport::new(800.0, 600.0).unwrap();
        let projection = Projection3d::perspective(
            std::f32::consts::FRAC_PI_2,
            800.0 / 600.0,
            world(0.1),
            world(100.0),
        )
        .unwrap();
        let camera =
            Camera3d::look_at(vector(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y, projection).unwrap();
        let projected = camera.project_world(Vec3::ZERO, viewport).unwrap();
        close(projected.logical_position().to_vec2().x(), 400.0);
        close(projected.logical_position().to_vec2().y(), 300.0);
        close(projected.view_depth(), 5.0);
        assert!(projected.inside_view());

        let outside = camera
            .project_world(vector(50.0, 0.0, 0.0), viewport)
            .unwrap();
        assert!(!outside.inside_view());
        assert_eq!(
            camera.project_world(vector(0.0, 0.0, 6.0), viewport),
            Err(Pseudo3dError::PointBehindCamera)
        );
    }

    #[test]
    fn perspective_camera_reports_unrepresentable_outside_projection() {
        let camera = Camera3d::look_at(
            Vec3::ZERO,
            vector(0.0, 0.0, -1.0),
            Vec3::Y,
            Projection3d::perspective(std::f32::consts::FRAC_PI_2, 1.0, world(0.1), world(10.0))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            camera.project_world(
                vector(f32::MAX, 0.0, -f32::MIN_POSITIVE),
                LogicalViewport::new(100.0, 100.0).unwrap(),
            ),
            Err(Pseudo3dError::ArithmeticOverflow)
        );
    }

    #[test]
    fn orthographic_camera_preserves_size_across_depth() {
        let viewport = LogicalViewport::new(800.0, 600.0).unwrap();
        let projection =
            Projection3d::orthographic(world(6.0), 800.0 / 600.0, world(0.1), world(100.0))
                .unwrap();
        let camera =
            Camera3d::look_at(vector(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y, projection).unwrap();
        let near = camera
            .project_world(vector(1.0, 0.0, 0.0), viewport)
            .unwrap();
        let far = camera
            .project_world(vector(1.0, 0.0, -20.0), viewport)
            .unwrap();
        close(
            near.logical_position().to_vec2().x(),
            far.logical_position().to_vec2().x(),
        );
        assert!(near.normalized_depth() < far.normalized_depth());
    }

    #[test]
    fn camera_and_projection_reject_degenerate_configuration() {
        assert_eq!(
            Projection3d::perspective(0.0, 1.0, world(0.1), world(10.0)),
            Err(Pseudo3dError::InvalidProjection)
        );
        let projection = Projection3d::perspective(1.0, 1.0, world(0.1), world(10.0)).unwrap();
        assert_eq!(
            Camera3d::look_at(Vec3::ZERO, Vec3::ZERO, Vec3::Y, projection),
            Err(Pseudo3dError::ZeroLengthVector)
        );
        assert_eq!(
            Camera3d::look_at(vector(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Z, projection),
            Err(Pseudo3dError::InvalidCameraBasis)
        );
    }

    #[test]
    fn gpu_clip_rows_match_public_camera_projection() {
        let viewport = LogicalViewport::new(800.0, 600.0).unwrap();
        for projection in [
            Projection3d::perspective(1.1, 800.0 / 600.0, world(0.1), world(50.0)).unwrap(),
            Projection3d::orthographic(world(7.0), 800.0 / 600.0, world(0.1), world(50.0)).unwrap(),
        ] {
            let camera = Camera3d::look_at(
                vector(3.0, 2.0, 6.0),
                vector(0.0, 0.5, 0.0),
                Vec3::Y,
                projection,
            )
            .unwrap();
            let point = vector(0.5, 0.25, -1.0);
            let projected = camera.project_world(point, viewport).unwrap();
            let rows = camera.world_to_clip_rows().unwrap();
            let homogeneous = [point.x(), point.y(), point.z(), 1.0];
            let clip = rows.map(|row| {
                row[0] * homogeneous[0] + row[1] * homogeneous[1] + row[2] * homogeneous[2] + row[3]
            });
            let ndc_x = clip[0] / clip[3];
            let ndc_y = clip[1] / clip[3];
            let screen_x = (ndc_x * 0.5 + 0.5) * viewport.width();
            let screen_y = (0.5 - ndc_y * 0.5) * viewport.height();
            close(screen_x, projected.logical_position().to_vec2().x());
            close(screen_y, projected.logical_position().to_vec2().y());
            close(clip[2] / clip[3], projected.normalized_depth());
        }
    }
}
