use super::super::{PreparedDrawBatch, is_portable_shader_source};
use crate::{
    Color, DrawCommand, Fill, LogicalPixels, LogicalScreenPosition, LogicalScreenVector,
    ScreenClipRect, ScreenScene, StrokeCap2d, StrokeWidthMode2d, Vec2,
};
use std::{error::Error, fmt};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::renderer) struct ScreenPrimitiveGpu {
    pub(super) anchor: [f32; 2],
    pub(super) corners: [[f32; 2]; 4],
    pub(super) shape: [f32; 2],
    pub(super) color: [f32; 4],
}

/// One validated solid primitive in viewport-local logical pixels.
/// Coordinates have a top-left origin and positive y downward. Clipping is
/// viewport-local, not translated with the primitive. Shapes are independent;
/// a sequence of capsules is not a polyline join implementation.
/// Constructors reject non-portable geometry, non-normalized linear RGBA and
/// unrepresentable source precision. Drawing additionally checks a combined
/// boundary-error bound of 0.25 physical pixels at the actual target density;
/// accepted construction does not guarantee acceptance at every later DPI.
#[derive(Clone, Copy, Debug)]
pub struct ScreenPrimitive2d {
    pub(super) gpu: ScreenPrimitiveGpu,
    pub(super) clip: Option<ScreenClipRect>,
    pub(super) source_error: f64,
}

impl ScreenPrimitive2d {
    /// Creates a filled analytic circle. Colors must be finite normalized linear RGBA.
    pub fn circle(
        center: LogicalScreenPosition,
        radius: LogicalPixels,
        color: Color,
    ) -> Result<Self, ScreenPrimitiveError> {
        Self::build::<0>(
            [f64::from(center.to_vec2().x), f64::from(center.to_vec2().y)],
            [f64::from(radius.get()), 0.0],
            [0.0, f64::from(radius.get())],
            0.0,
            color,
        )
    }

    /// Creates an independent round-capped line of positive logical-pixel width.
    /// Coincident endpoints are rejected; use a circle for a point. Extremely
    /// disproportionate length/width is rejected when cap coordinates lose
    /// portable f32 resolution, rather than silently rendering square ends.
    pub fn round_line(
        from: LogicalScreenPosition,
        to: LogicalScreenPosition,
        width: LogicalPixels,
        color: Color,
    ) -> Result<Self, ScreenPrimitiveError> {
        let from = from.to_vec2();
        let to = to.to_vec2();
        let direction = [
            f64::from(to.x) - f64::from(from.x),
            f64::from(to.y) - f64::from(from.y),
        ];
        // Coordinates originate as f32: their f64 differences and squares
        // cannot overflow or underflow f64, so scaled hypot is unnecessary.
        let length = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        if !length.is_finite() || length == 0.0 {
            return Err(ScreenPrimitiveError::InvalidGeometry);
        }
        let radius = f64::from(width.get()) * 0.5;
        let ratio = length * 0.5 / radius;
        if !is_portable_shader_source(ratio as f32) {
            return Err(ScreenPrimitiveError::InvalidGeometry);
        }
        if ratio as f32 + 1.0 == ratio as f32 {
            return Err(ScreenPrimitiveError::PrecisionLimit);
        }
        let unit = [direction[0] / length, direction[1] / length];
        let center = [
            (f64::from(from.x) + f64::from(to.x)) * 0.5,
            (f64::from(from.y) + f64::from(to.y)) * 0.5,
        ];
        let half_extent = length * 0.5 + radius;
        Self::build::<1>(
            center,
            [unit[0] * half_extent, unit[1] * half_extent],
            [-unit[1] * radius, unit[0] * radius],
            ratio,
            color,
        )
    }

    /// Creates a square-cornered filled rectangle with strictly positive logical size.
    pub fn rect(
        minimum: LogicalScreenPosition,
        size: LogicalScreenVector,
        color: Color,
    ) -> Result<Self, ScreenPrimitiveError> {
        let minimum = minimum.to_vec2();
        let size = size.to_vec2();
        if size.x <= 0.0 || size.y <= 0.0 {
            return Err(ScreenPrimitiveError::InvalidGeometry);
        }
        let center = [
            f64::from(minimum.x) + f64::from(size.x) * 0.5,
            f64::from(minimum.y) + f64::from(size.y) * 0.5,
        ];
        Self::build::<2>(
            center,
            [f64::from(size.x) * 0.5, 0.0],
            [0.0, f64::from(size.y) * 0.5],
            0.0,
            color,
        )
    }

    /// Captures a previously validated viewport-local logical clip.
    pub const fn with_clip(mut self, clip: ScreenClipRect) -> Self {
        self.clip = Some(clip);
        self
    }
    /// Optional viewport-local logical clip captured by this primitive.
    pub const fn clip(self) -> Option<ScreenClipRect> {
        self.clip
    }

    #[inline]
    fn build<const KIND: u32>(
        center: [f64; 2],
        along: [f64; 2],
        across: [f64; 2],
        ratio: f64,
        color: Color,
    ) -> Result<Self, ScreenPrimitiveError> {
        if !color.is_normalized() {
            return Err(ScreenPrimitiveError::InvalidColor);
        }
        let signs = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let precise_corners =
            signs.map(|sign| [0, 1].map(|axis| along[axis] * sign[0] + across[axis] * sign[1]));
        let corners = precise_corners.map(|corner| corner.map(|value| value as f32));
        let gpu = ScreenPrimitiveGpu {
            anchor: [center[0] as f32, -center[1] as f32],
            corners: corners.map(|corner| [corner[0], -corner[1]]),
            shape: [KIND as f32, ratio as f32],
            color: color.to_array(),
        };
        if !gpu
            .anchor
            .into_iter()
            .chain(gpu.corners.into_iter().flatten())
            .chain(gpu.shape)
            .all(is_portable_shader_source)
            || corners[0] == corners[1]
            || corners[1] == corners[2]
        {
            return Err(ScreenPrimitiveError::InvalidGeometry);
        }
        // Bound conversion and fragment classification separately from topology.
        // A finite, nondegenerate quad can still move a visible edge by hundreds
        // of pixels when a huge midpoint/extent cancels near the viewport.
        let mut conversion_error = 0.0_f64;
        for (precise, stored) in precise_corners.into_iter().zip(corners) {
            for axis in 0..2 {
                conversion_error = conversion_error.max(
                    (f64::from(center[axis] as f32) - center[axis]).abs()
                        + (f64::from(stored[axis]) - precise[axis]).abs(),
                );
            }
        }
        let extent = (along[0] * along[0] + along[1] * along[1]).sqrt()
            + (across[0] * across[0] + across[1] * across[1]).sqrt();
        // The local circle/capsule calculation contains bounded products and
        // subtraction after interpolation. In logical units its absolute error
        // scales with extent, including ratio rounding at long capsule ends.
        let fragment_error = if KIND == 2 {
            0.0
        } else {
            64.0 * f64::from(f32::EPSILON) * (extent + 1.0)
        };
        let source_error = conversion_error * 2.0
            + fragment_error
            + 64.0 * f64::EPSILON * (center[0].abs() + center[1].abs() + extent + 1.0);
        if !source_error.is_finite() || source_error > 0.125 {
            return Err(ScreenPrimitiveError::PrecisionLimit);
        }
        Ok(Self {
            gpu,
            clip: None,
            source_error,
        })
    }

    /// Converts supported commands in stable scene order without partial output.
    /// Only solid filled unstroked circles/square rectangles and independent
    /// round logical-width lines are accepted. Gradients, shadows, polylines,
    /// dashes, markers and other styles return their source command index.
    /// The scene background is not part of the batch; pass it to begin_frame.
    pub fn from_screen_scene(scene: &ScreenScene) -> Result<Vec<Self>, ScreenPrimitiveError> {
        let commands = scene.as_scene().commands();
        let mut primitives = Vec::new();
        primitives.try_reserve_exact(commands.len()).map_err(|_| {
            ScreenPrimitiveError::AllocationFailed {
                requested_bytes: commands.len().saturating_mul(std::mem::size_of::<Self>()),
            }
        })?;
        let position = |point: Vec2| LogicalScreenPosition::new(point.x, -point.y);
        for (index, command) in commands.iter().enumerate() {
            let unsupported = || ScreenPrimitiveError::UnsupportedCommand { index };
            let mut primitive = match command.command() {
                DrawCommand::Circle(circle) => {
                    let style = circle.style();
                    if style.stroke().is_some() || style.shadow().is_some() {
                        return Err(unsupported());
                    }
                    let Some(Fill::Solid(color)) = style.fill() else {
                        return Err(unsupported());
                    };
                    Self::circle(
                        position(circle.center()),
                        LogicalPixels::new(circle.radius()).map_err(|_| unsupported())?,
                        color,
                    )?
                }
                DrawCommand::Rect(rectangle) => {
                    let style = rectangle.style();
                    if rectangle.corner_radius() != 0.0
                        || style.stroke().is_some()
                        || style.shadow().is_some()
                    {
                        return Err(unsupported());
                    }
                    let Some(Fill::Solid(color)) = style.fill() else {
                        return Err(unsupported());
                    };
                    let rect = rectangle.rect().normalized();
                    Self::rect(
                        position(Vec2::new(rect.min().x, rect.max().y)),
                        LogicalScreenVector::new(rect.width(), rect.height()),
                        color,
                    )?
                }
                DrawCommand::Line(line) => {
                    let style = line.stroke_style();
                    if style.cap() != StrokeCap2d::Round
                        || style.width_mode() != StrokeWidthMode2d::LogicalPixels
                        || style.dash_pattern().is_some()
                        || style.start_marker().is_some()
                        || style.end_marker().is_some()
                    {
                        return Err(unsupported());
                    }
                    Self::round_line(
                        position(line.from()),
                        position(line.to()),
                        LogicalPixels::new(style.stroke().width()).map_err(|_| unsupported())?,
                        style.stroke().color(),
                    )?
                }
                DrawCommand::Polyline(_) => return Err(unsupported()),
            };
            primitive.clip = command.screen_clip();
            primitives.push(primitive);
        }
        Ok(primitives)
    }
}

/// Explicit primitive, resident CPU/GPU and per-update upload limits.
/// During capacity growth old/new resources may overlap, bounded by twice the
/// resident limits. Input slices and driver/in-flight allocations are excluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenPrimitiveBudget {
    pub(super) max_primitives: usize,
    pub(super) max_retained_bytes: usize,
    pub(super) max_buffer_bytes: usize,
    pub(super) max_upload_bytes: usize,
}

impl ScreenPrimitiveBudget {
    /// Constructs explicit limits; zero count is a valid empty-only budget.
    /// Even an empty batch needs one 64-byte GPU buffer slot.
    pub const fn new(
        max_primitives: usize,
        max_retained_bytes: usize,
        max_buffer_bytes: usize,
        max_upload_bytes: usize,
    ) -> Self {
        Self {
            max_primitives,
            max_retained_bytes,
            max_buffer_bytes,
            max_upload_bytes,
        }
    }
    /// Maximum submitted primitive count, never silently truncated.
    pub const fn max_primitives(self) -> usize {
        self.max_primitives
    }
    /// Retained instance and clip-span CPU capacity ceiling.
    pub const fn max_retained_bytes(self) -> usize {
        self.max_retained_bytes
    }
    /// GPU instance-buffer capacity ceiling.
    pub const fn max_buffer_bytes(self) -> usize {
        self.max_buffer_bytes
    }
    /// Maximum host instance payload for creation or one complete update.
    pub const fn max_upload_bytes(self) -> usize {
        self.max_upload_bytes
    }
}

impl Default for ScreenPrimitiveBudget {
    fn default() -> Self {
        Self::new(100_000, 32 * 1024 * 1024, 8 * 1024 * 1024, 8 * 1024 * 1024)
    }
}

/// Actual work of one accepted full snapshot update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenPrimitiveUploadReport {
    pub(super) uploaded_bytes: usize,
    pub(super) buffer_reused: bool,
}
impl ScreenPrimitiveUploadReport {
    /// Host bytes enqueued, excluding previously submitted GPU work.
    pub const fn uploaded_bytes(self) -> usize {
        self.uploaded_bytes
    }
    /// Whether the existing GPU buffer allocation was reused.
    pub const fn buffer_reused(self) -> bool {
        self.buffer_reused
    }
}

/// Explicit rejection from analytic primitive construction or resource updates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScreenPrimitiveError {
    /// Geometry is degenerate or outside the portable finite shader envelope.
    InvalidGeometry,
    /// Compact f32 representation cannot preserve the subpixel source boundary.
    /// Use ordinary scene geometry or reduce the shape's coordinate/extent range.
    PrecisionLimit,
    /// Color channels are not finite and normalized to 0..=1.
    InvalidColor,
    /// Source scene command cannot be represented by this narrow solid path.
    UnsupportedCommand {
        /// Index in stable scene draw order.
        index: usize,
    },
    /// A caller work/residency bound would be exceeded; nothing was truncated.
    BudgetExceeded {
        /// Which explicitly configured limit failed.
        resource: &'static str,
        /// Configured ceiling.
        limit: usize,
        /// Required value.
        required: usize,
    },
    /// Host allocation failed before the visible snapshot was changed.
    AllocationFailed {
        /// Requested CPU allocation payload.
        requested_bytes: usize,
    },
    /// Instance capacity exceeds u32 draw addressing or the logical device limit.
    DeviceCapacityExceeded,
    /// Resource belongs to another logical renderer; restore it first.
    RendererMismatch,
}
impl fmt::Display for ScreenPrimitiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "analytic screen primitive error: {self:?}")
    }
}
impl Error for ScreenPrimitiveError {}

pub(super) fn cpu_bytes(instances: usize, batches: usize) -> usize {
    instances
        .saturating_mul(std::mem::size_of::<ScreenPrimitiveGpu>())
        .saturating_add(batches.saturating_mul(std::mem::size_of::<PreparedDrawBatch>()))
}
