//! Logical-screen wireframe presentation and extensible mesh style.

use std::{error::Error, fmt};

use super::SurfaceStyle3d;
use crate::{Color, LogicalPixels};

const MAX_LOGICAL_EDGE_METRIC: f32 = 1_048_576.0;

/// Extensible visual material bundle for a retained 3D object.
///
/// Surface and wireframe presentation are independent. Edge-only construction
/// geometry uses `surface = None`; future section materials can extend
/// [`SurfaceStyle3d`] without changing scene insertion or object addressing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshStyle3d {
    surface: Option<SurfaceStyle3d>,
    wireframe: Option<WireframeStyle3d>,
}

impl MeshStyle3d {
    /// Creates an object with the selected surface material and no display edges.
    pub const fn surface(surface: SurfaceStyle3d) -> Self {
        Self {
            surface: Some(surface),
            wireframe: None,
        }
    }

    /// Creates display edges without a surface/depth-writing face pass.
    pub const fn wireframe(wireframe: WireframeStyle3d) -> Self {
        Self {
            surface: None,
            wireframe: Some(wireframe),
        }
    }

    /// Adds or replaces display-edge presentation.
    pub const fn with_wireframe(mut self, wireframe: WireframeStyle3d) -> Self {
        self.wireframe = Some(wireframe);
        self
    }

    /// Returns the optional surface material.
    pub const fn surface_style(self) -> Option<SurfaceStyle3d> {
        self.surface
    }

    /// Returns the optional edge presentation.
    pub const fn wireframe_style(self) -> Option<WireframeStyle3d> {
        self.wireframe
    }
}

/// Logical-screen presentation for explicit mathematical mesh edges.
///
/// Visible fragments are solid. Fragments occluded beyond the depth buffer's
/// conservative coplanar tolerance may be drawn with a logical-pixel dash/gap
/// pattern. Classification uses one implementation depth unit away from the
/// camera for hidden fragments and one unit toward it for visible fragments,
/// with visible fragments rendered last. Therefore coplanar and
/// sub-depth-resolution separations intentionally resolve as visible; this is
/// raster visibility, not exact analytic solid geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WireframeStyle3d {
    visible_color: Color,
    visible_width: LogicalPixels,
    hidden: Option<HiddenEdgeStyle3d>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HiddenEdgeStyle3d {
    color: Color,
    width: LogicalPixels,
    dash_length: LogicalPixels,
    gap_length: LogicalPixels,
}

impl WireframeStyle3d {
    /// Largest accepted width, dash, or gap in logical pixels.
    ///
    /// The bound keeps every style-derived shader operation inside the
    /// cross-backend arithmetic envelope.
    pub const MAX_LOGICAL_PIXELS: f32 = MAX_LOGICAL_EDGE_METRIC;

    /// Creates solid visible edges with hidden fragments disabled.
    pub fn visible(color: Color, width: LogicalPixels) -> Result<Self, Mesh3dStyleError> {
        validate_edge_color_width(color, width)?;
        Ok(Self {
            visible_color: color,
            visible_width: width,
            hidden: None,
        })
    }

    /// Enables dashed depth-occluded fragments in logical screen pixels.
    ///
    /// Occlusion must exceed the conservative two-unit coplanar tolerance
    /// documented on [`WireframeStyle3d`] before a fragment resolves hidden.
    pub fn with_hidden(
        mut self,
        color: Color,
        width: LogicalPixels,
        dash_length: LogicalPixels,
        gap_length: LogicalPixels,
    ) -> Result<Self, Mesh3dStyleError> {
        validate_edge_color_width(color, width)?;
        if !is_stable_edge_metric(dash_length.get())
            || !is_stable_edge_metric(gap_length.get())
            || !(dash_length.get() + gap_length.get()).is_finite()
        {
            return Err(Mesh3dStyleError::InvalidDashPattern);
        }
        self.hidden = Some(HiddenEdgeStyle3d {
            color,
            width,
            dash_length,
            gap_length,
        });
        Ok(self)
    }

    /// Returns the solid visible-edge color in linear RGBA.
    pub const fn visible_color(self) -> Color {
        self.visible_color
    }

    /// Returns visible-edge width in logical screen pixels.
    pub const fn visible_width(self) -> LogicalPixels {
        self.visible_width
    }

    /// Returns whether depth-occluded edge fragments should be drawn.
    pub const fn hidden_enabled(self) -> bool {
        self.hidden.is_some()
    }

    /// Returns hidden-edge color when enabled.
    pub const fn hidden_color(self) -> Option<Color> {
        match self.hidden {
            Some(hidden) => Some(hidden.color),
            None => None,
        }
    }

    /// Returns hidden-edge width in logical screen pixels when enabled.
    pub const fn hidden_width(self) -> Option<LogicalPixels> {
        match self.hidden {
            Some(hidden) => Some(hidden.width),
            None => None,
        }
    }

    /// Returns hidden dash and gap lengths in logical screen pixels.
    pub const fn hidden_pattern(self) -> Option<(LogicalPixels, LogicalPixels)> {
        match self.hidden {
            Some(hidden) => Some((hidden.dash_length, hidden.gap_length)),
            None => None,
        }
    }
}

/// Rejection reason for logical-screen 3D edge presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mesh3dStyleError {
    /// A mask/blend tint must have finite straight-linear RGBA in 0..=1.
    InvalidMaterialColor,
    /// The alpha-mask cutoff must be zero or normal and finite in 0..=1.
    InvalidMaskCutoff,
    /// The legacy opaque constructor requires normalized color with alpha one.
    InvalidSurfaceColor,
    /// Edge color must be normalized and opaque, and width must be a normal
    /// positive value no greater than [`WireframeStyle3d::MAX_LOGICAL_PIXELS`].
    InvalidColorOrWidth,
    /// Hidden dash and gap must be normal positive values, each no greater than
    /// [`WireframeStyle3d::MAX_LOGICAL_PIXELS`], with a finite sum.
    InvalidDashPattern,
}

impl fmt::Display for Mesh3dStyleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMaterialColor => write!(
                formatter,
                "3D material color must be normalized straight-linear RGBA"
            ),
            Self::InvalidMaskCutoff => write!(
                formatter,
                "3D mask cutoff must be zero or normal and finite in 0..=1"
            ),
            Self::InvalidSurfaceColor => write!(formatter, "3D surface color must be opaque"),
            Self::InvalidColorOrWidth => {
                write!(
                    formatter,
                    "3D edge color must be opaque and width must be normal and in 0..={MAX_LOGICAL_EDGE_METRIC} logical pixels"
                )
            }
            Self::InvalidDashPattern => {
                write!(
                    formatter,
                    "3D hidden-edge dash and gap must each be normal and in 0..={MAX_LOGICAL_EDGE_METRIC} logical pixels with a finite sum"
                )
            }
        }
    }
}

impl Error for Mesh3dStyleError {}

fn validate_edge_color_width(color: Color, width: LogicalPixels) -> Result<(), Mesh3dStyleError> {
    if color.is_normalized() && color.alpha() == 1.0 && is_stable_edge_metric(width.get()) {
        Ok(())
    } else {
        Err(Mesh3dStyleError::InvalidColorOrWidth)
    }
}

fn is_stable_edge_metric(value: f32) -> bool {
    value.is_normal() && value <= MAX_LOGICAL_EDGE_METRIC
}
