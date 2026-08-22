use std::{
    borrow::Cow,
    error::Error,
    fmt,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    Camera2d, Circle, Color, ColorMap, DrawCommand, Fill, Line, LogicalScreenPosition,
    LogicalViewport, PhysicalScreenPosition, Polyline, Rect, ScalarField, Scene, ScreenClipRect,
    Shadow, ShapeStyle, Stroke, Vec2,
};

const INITIAL_VERTEX_CAPACITY: usize = 4096;
const CIRCLE_SEGMENTS: usize = 64;
const CORNER_SEGMENTS: usize = 12;
const PREFERRED_SAMPLE_COUNT: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    world_position: [f32; 2],
    depth: f32,
    screen_offset: [f32; 2],
    previous_direction: [f32; 2],
    next_direction: [f32; 2],
    normal_distance: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    camera_center: [f32; 4],
    world_to_screen_x: [f32; 4],
    world_to_screen_y: [f32; 4],
    screen_to_clip: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct HeatmapUniform {
    value_range: [f32; 4],
    dimensions: [u32; 4],
}

/// Scalar-field sampling mode used by heatmap rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalarFieldSampling {
    /// Select one exact source texel for each output fragment.
    #[default]
    Nearest,
    /// Bilinearly interpolate neighboring source texels in shader math.
    Linear,
}

impl ScalarFieldSampling {
    fn shader_value(self) -> u32 {
        match self {
            Self::Nearest => 0,
            Self::Linear => 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GeometryExtents {
    world_min: Vec2,
    world_max: Vec2,
    depth_min: f32,
    depth_max: f32,
    direction_min: Vec2,
    direction_max: Vec2,
    screen_offset_max_abs: Vec2,
    normal_distance_max_abs: f32,
    empty: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScissorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct PreparedDrawBatch {
    vertex_range: Range<u32>,
    screen_clip: Option<ScreenClipRect>,
}

/// Outcome of converting accepted scene commands into GPU vertices.
///
/// A dropped command had finite source fields but produced no finite triangle
/// vertices, usually because arithmetic overflowed while expanding geometry.
/// Hosts can surface this separately from presentation success.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TessellationStats {
    command_count: usize,
    rendered_command_count: usize,
    dropped_command_count: usize,
}

impl TessellationStats {
    /// Returns accepted source commands examined by the tessellator.
    pub fn command_count(self) -> usize {
        self.command_count
    }

    /// Returns source commands that emitted finite triangle vertices.
    pub fn rendered_command_count(self) -> usize {
        self.rendered_command_count
    }

    /// Returns source commands discarded because tessellation emitted no valid geometry.
    pub fn dropped_command_count(self) -> usize {
        self.dropped_command_count
    }
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32,
        2 => Float32x2,
        3 => Float32x2,
        4 => Float32x2,
        5 => Float32,
        6 => Float32x4
    ];

    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };

    fn is_finite(self) -> bool {
        self.world_position.iter().all(|value| value.is_finite())
            && self.depth.is_finite()
            && self.screen_offset.iter().all(|value| value.is_finite())
            && self
                .previous_direction
                .iter()
                .all(|value| value.is_finite())
            && self.next_direction.iter().all(|value| value.is_finite())
            && self.normal_distance.is_finite()
            && self.color.iter().all(|value| value.is_finite())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleUnitVertex {
    direction: [f32; 2],
}

impl ParticleUnitVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleGpu {
    world_position: [f32; 2],
    depth: f32,
    radius: f32,
    color: [f32; 4],
}

impl ParticleGpu {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        1 => Float32x2,
        2 => Float32,
        3 => Float32,
        4 => Float32x4
    ];

    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &Self::ATTRIBUTES,
    };

    fn screen_position(self, camera: CameraUniform) -> Vec2 {
        let relative_x = self.world_position[0] - camera.camera_center[0];
        let relative_y = self.world_position[1] - camera.camera_center[1];
        Vec2::new(
            camera.world_to_screen_x[0] * relative_x
                + camera.world_to_screen_x[1] * relative_y
                + camera.world_to_screen_x[2] * self.depth
                + camera.world_to_screen_x[3],
            camera.world_to_screen_y[0] * relative_x
                + camera.world_to_screen_y[1] * relative_y
                + camera.world_to_screen_y[2] * self.depth
                + camera.world_to_screen_y[3],
        )
    }

    fn is_safe_for(self, camera: CameraUniform) -> bool {
        let relative_x = self.world_position[0] - camera.camera_center[0];
        let relative_y = self.world_position[1] - camera.camera_center[1];
        let screen = self.screen_position(camera);
        let screen_x = screen.x;
        let screen_y = screen.y;
        let clip_x = screen_x * camera.screen_to_clip[0] + camera.screen_to_clip[2];
        let clip_y = screen_y * camera.screen_to_clip[1] + camera.screen_to_clip[3];
        relative_x.is_finite()
            && relative_y.is_finite()
            && screen_x.is_finite()
            && screen_y.is_finite()
            && (screen_x + self.radius).is_finite()
            && (screen_x - self.radius).is_finite()
            && (screen_y + self.radius).is_finite()
            && (screen_y - self.radius).is_finite()
            && clip_x.is_finite()
            && clip_y.is_finite()
    }

    fn intersects_viewport(self, camera: CameraUniform, viewport: LogicalViewport) -> bool {
        let screen = self.screen_position(camera);
        screen.x + self.radius >= 0.0
            && screen.x - self.radius <= viewport.width()
            && screen.y + self.radius >= 0.0
            && screen.y - self.radius <= viewport.height()
    }
}

impl CameraUniform {
    fn new(camera: Camera2d, viewport: LogicalViewport) -> Option<Self> {
        let projection_cosine = camera.projection().tilt().cos();
        let rotation_cosine = camera.rotation().cos();
        let rotation_sine = camera.rotation().sin();
        let zoom = camera.zoom();
        let center = camera.center();
        let projection_sine = camera.projection().tilt().sin();
        let depth_scale = camera.projection().depth_scale();

        let horizontal_x = zoom * rotation_cosine;
        let horizontal_y = -zoom * rotation_sine * projection_cosine;
        let vertical_x = -zoom * rotation_sine;
        let vertical_y = -zoom * rotation_cosine * projection_cosine;
        let horizontal_depth =
            zoom * depth_scale * projection_sine * (rotation_cosine * 0.5 - rotation_sine);
        let vertical_depth =
            -zoom * depth_scale * projection_sine * (rotation_sine * 0.5 + rotation_cosine);

        let uniform = Self {
            camera_center: [center.x, center.y, 0.0, 0.0],
            world_to_screen_x: [
                horizontal_x,
                horizontal_y,
                horizontal_depth,
                viewport.width() * 0.5,
            ],
            world_to_screen_y: [
                vertical_x,
                vertical_y,
                vertical_depth,
                viewport.height() * 0.5,
            ],
            screen_to_clip: [2.0 / viewport.width(), -2.0 / viewport.height(), -1.0, 1.0],
        };
        uniform.is_finite().then_some(uniform)
    }

    fn is_finite(self) -> bool {
        self.camera_center
            .iter()
            .chain(self.world_to_screen_x.iter())
            .chain(self.world_to_screen_y.iter())
            .chain(self.screen_to_clip.iter())
            .all(|value| value.is_finite())
    }

    #[cfg(test)]
    fn world_to_screen(self, world: Vec2, depth: f32) -> Vec2 {
        let relative = world - Vec2::new(self.camera_center[0], self.camera_center[1]);
        Vec2::new(
            self.world_to_screen_x[0] * relative.x
                + self.world_to_screen_x[1] * relative.y
                + self.world_to_screen_x[2] * depth
                + self.world_to_screen_x[3],
            self.world_to_screen_y[0] * relative.x
                + self.world_to_screen_y[1] * relative.y
                + self.world_to_screen_y[2] * depth
                + self.world_to_screen_y[3],
        )
    }

    #[cfg(test)]
    fn direction_to_screen(self, direction: Vec2) -> Vec2 {
        Vec2::new(
            self.world_to_screen_x[0] * direction.x + self.world_to_screen_x[1] * direction.y,
            self.world_to_screen_y[0] * direction.x + self.world_to_screen_y[1] * direction.y,
        )
    }
}

impl GeometryExtents {
    fn from_vertices(vertices: &[Vertex]) -> Self {
        let mut extents = Self {
            world_min: Vec2::splat(f32::INFINITY),
            world_max: Vec2::splat(f32::NEG_INFINITY),
            depth_min: f32::INFINITY,
            depth_max: f32::NEG_INFINITY,
            direction_min: Vec2::splat(f32::INFINITY),
            direction_max: Vec2::splat(f32::NEG_INFINITY),
            screen_offset_max_abs: Vec2::ZERO,
            normal_distance_max_abs: 0.0,
            empty: vertices.is_empty(),
        };

        for vertex in vertices {
            let world = Vec2::new(vertex.world_position[0], vertex.world_position[1]);
            extents.world_min.x = extents.world_min.x.min(world.x);
            extents.world_min.y = extents.world_min.y.min(world.y);
            extents.world_max.x = extents.world_max.x.max(world.x);
            extents.world_max.y = extents.world_max.y.max(world.y);
            extents.depth_min = extents.depth_min.min(vertex.depth);
            extents.depth_max = extents.depth_max.max(vertex.depth);

            for direction in [vertex.previous_direction, vertex.next_direction] {
                extents.direction_min.x = extents.direction_min.x.min(direction[0]);
                extents.direction_min.y = extents.direction_min.y.min(direction[1]);
                extents.direction_max.x = extents.direction_max.x.max(direction[0]);
                extents.direction_max.y = extents.direction_max.y.max(direction[1]);
            }

            extents.screen_offset_max_abs.x = extents
                .screen_offset_max_abs
                .x
                .max(vertex.screen_offset[0].abs());
            extents.screen_offset_max_abs.y = extents
                .screen_offset_max_abs
                .y
                .max(vertex.screen_offset[1].abs());
            extents.normal_distance_max_abs = extents
                .normal_distance_max_abs
                .max(vertex.normal_distance.abs());
        }

        extents
    }

    fn is_safe_for(self, uniform: CameraUniform) -> bool {
        if self.empty {
            return true;
        }

        let center = Vec2::new(uniform.camera_center[0], uniform.camera_center[1]);
        let world_horizontal = transformed_world_interval(
            uniform.world_to_screen_x,
            self.world_min,
            self.world_max,
            self.depth_min,
            self.depth_max,
            center,
        );
        let world_vertical = transformed_world_interval(
            uniform.world_to_screen_y,
            self.world_min,
            self.world_max,
            self.depth_min,
            self.depth_max,
            center,
        );
        let direction_horizontal = transformed_direction_interval(
            uniform.world_to_screen_x,
            self.direction_min,
            self.direction_max,
        );
        let direction_vertical = transformed_direction_interval(
            uniform.world_to_screen_y,
            self.direction_min,
            self.direction_max,
        );
        let maximum_miter = self.normal_distance_max_abs as f64 * 1_000.0;
        let horizontal_limit = interval_max_abs(world_horizontal)
            + self.screen_offset_max_abs.x as f64
            + maximum_miter;
        let vertical_limit =
            interval_max_abs(world_vertical) + self.screen_offset_max_abs.y as f64 + maximum_miter;

        [
            world_horizontal.0,
            world_horizontal.1,
            world_vertical.0,
            world_vertical.1,
            direction_horizontal.0,
            direction_horizontal.1,
            direction_vertical.0,
            direction_vertical.1,
            horizontal_limit,
            vertical_limit,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value.abs() <= f32::MAX as f64)
    }
}

fn transformed_world_interval(
    row: [f32; 4],
    minimum: Vec2,
    maximum: Vec2,
    depth_minimum: f32,
    depth_maximum: f32,
    center: Vec2,
) -> (f64, f64) {
    let relative_minimum = minimum - center;
    let relative_maximum = maximum - center;
    let horizontal = interval_products(row[0], relative_minimum.x, relative_maximum.x);
    let vertical = interval_products(row[1], relative_minimum.y, relative_maximum.y);
    let depth = interval_products(row[2], depth_minimum, depth_maximum);
    (
        horizontal.0 + vertical.0 + depth.0 + row[3] as f64,
        horizontal.1 + vertical.1 + depth.1 + row[3] as f64,
    )
}

fn transformed_direction_interval(row: [f32; 4], minimum: Vec2, maximum: Vec2) -> (f64, f64) {
    let horizontal = interval_products(row[0], minimum.x, maximum.x);
    let vertical = interval_products(row[1], minimum.y, maximum.y);
    (horizontal.0 + vertical.0, horizontal.1 + vertical.1)
}

fn interval_products(coefficient: f32, minimum: f32, maximum: f32) -> (f64, f64) {
    let first = coefficient as f64 * minimum as f64;
    let second = coefficient as f64 * maximum as f64;
    (first.min(second), first.max(second))
}

fn interval_max_abs(interval: (f64, f64)) -> f64 {
    interval.0.abs().max(interval.1.abs())
}

/// Result of attempting to draw a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStatus {
    /// Commands were submitted and the frame was presented.
    Drawn,
    /// The frame was skipped because the window surface was temporarily unavailable.
    Skipped(RendererSurfaceStatus),
}

/// CPU-side durations measured while preparing and submitting one renderer frame.
///
/// These values do not measure GPU execution or the monitor scanout timestamp.
/// FIFO presentation back-pressure normally appears in `surface_acquire` because
/// `wgpu` blocks frame acquisition when the presentation queue is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RendererFrameMetrics {
    tessellation: Duration,
    upload: Duration,
    camera_uniform_upload: Duration,
    surface_acquire: Duration,
    encode_submit_present: Duration,
    total_cpu: Duration,
    geometry_reused: bool,
    geometry_streamed: bool,
    tessellation_stats: TessellationStats,
}

impl RendererFrameMetrics {
    /// Returns CPU time spent validating and tessellating scene commands.
    pub fn tessellation(self) -> Duration {
        self.tessellation
    }

    /// Returns CPU time spent writing transient vertex data into the GPU buffer.
    pub fn upload(self) -> Duration {
        self.upload
    }

    /// Returns CPU time spent updating the small per-frame camera uniform.
    pub fn camera_uniform_upload(self) -> Duration {
        self.camera_uniform_upload
    }

    /// Returns CPU time spent acquiring the next presentation surface texture.
    ///
    /// In FIFO modes this includes queue back-pressure and is the closest CPU-side
    /// approximation to VSync wait exposed by `wgpu`.
    pub fn surface_acquire(self) -> Duration {
        self.surface_acquire
    }

    /// Returns CPU time spent encoding, submitting, and dispatching present.
    ///
    /// Present dispatch does not mean the monitor has already scanned out the
    /// frame; `wgpu` does not expose that timestamp here.
    pub fn encode_submit_present(self) -> Duration {
        self.encode_submit_present
    }

    /// Returns total CPU time inside the profiled render call.
    pub fn total_cpu(self) -> Duration {
        self.total_cpu
    }

    /// Returns whether this frame reused geometry from a [`PreparedScene`].
    pub fn geometry_reused(self) -> bool {
        self.geometry_reused
    }

    /// Returns whether this frame rendered geometry updated through [`DynamicMesh2d`].
    pub fn geometry_streamed(self) -> bool {
        self.geometry_streamed
    }

    /// Returns command-level tessellation results for this frame.
    pub fn tessellation_stats(self) -> TessellationStats {
        self.tessellation_stats
    }
}

/// Status and CPU timing breakdown for one renderer frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderReport {
    status: RenderStatus,
    metrics: RendererFrameMetrics,
}

impl RenderReport {
    /// Returns whether the frame was drawn or skipped by the surface.
    pub fn status(self) -> RenderStatus {
        self.status
    }

    /// Returns CPU-side stage durations for the render call.
    pub fn metrics(self) -> RendererFrameMetrics {
        self.metrics
    }
}

/// Recoverable or fatal state reported by the presentation surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererSurfaceStatus {
    /// Acquiring the next frame timed out; try again on a later frame.
    Timeout,
    /// The surface is occluded or minimized; rendering can be skipped.
    Occluded,
    /// The surface configuration is stale; resizing or reconfiguring is needed.
    Outdated,
    /// The surface was lost and should be recreated by the host application.
    Lost,
    /// `wgpu` reported a validation error while acquiring the frame.
    Validation,
}

/// Fatal failure while preparing or submitting one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererFrameError {
    /// The presentation surface was lost or rejected frame acquisition.
    Surface(RendererSurfaceStatus),
    /// Camera and geometry are finite separately but overflow when transformed together.
    InvalidGeometryTransform,
}

impl fmt::Display for RendererFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Surface(status) => write!(formatter, "renderer surface failed: {status:?}"),
            Self::InvalidGeometryTransform => {
                write!(formatter, "camera and geometry overflow the GPU transform")
            }
        }
    }
}

impl Error for RendererFrameError {}

/// Errors that can happen while creating the `wgpu` renderer.
#[derive(Debug)]
pub enum RendererInitError {
    /// Surface creation failed for the provided window or canvas target.
    CreateSurface(wgpu::CreateSurfaceError),
    /// No compatible GPU adapter could be selected.
    RequestAdapter(wgpu::RequestAdapterError),
    /// A logical GPU device and queue could not be created.
    RequestDevice(wgpu::RequestDeviceError),
    /// The surface did not expose a usable default configuration.
    NoSurfaceConfig,
}

impl fmt::Display for RendererInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateSurface(error) => write!(formatter, "failed to create surface: {error}"),
            Self::RequestAdapter(error) => write!(formatter, "failed to request adapter: {error}"),
            Self::RequestDevice(error) => write!(formatter, "failed to request device: {error}"),
            Self::NoSurfaceConfig => write!(formatter, "surface has no supported default config"),
        }
    }
}

impl Error for RendererInitError {}

/// Invalid runtime or initialization configuration for [`WgpuRenderer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RendererConfigurationError {
    /// Logical-to-physical scale must produce finite logical dimensions for every supported surface.
    InvalidScaleFactor {
        /// Rejected physical pixels per logical screen pixel.
        scale_factor: f64,
    },
}

impl fmt::Display for RendererConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScaleFactor { scale_factor } => write!(
                formatter,
                "renderer scale factor must be finite, positive, representable as f32, and keep a u32 surface finite in logical pixels, got {scale_factor}"
            ),
        }
    }
}

impl Error for RendererConfigurationError {}

/// Failure while converting between logical and physical screen coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RendererCoordinateError {
    /// The source position is non-finite or the scaled result cannot be represented as `f32`.
    NonFiniteConversion,
}

impl fmt::Display for RendererCoordinateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "screen coordinate conversion produced a non-finite result"
        )
    }
}

impl Error for RendererCoordinateError {}

/// Failure to draw geometry prepared by [`WgpuRenderer::prepare_scene`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedSceneRenderError {
    /// The prepared geometry belongs to a different renderer and GPU device.
    RendererMismatch,
    /// Frame rendering failed after prepared-scene ownership validation.
    Frame(RendererFrameError),
}

impl fmt::Display for PreparedSceneRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererMismatch => {
                write!(formatter, "prepared scene belongs to a different renderer")
            }
            Self::Frame(error) => write!(formatter, "prepared frame failed: {error}"),
        }
    }
}

impl Error for PreparedSceneRenderError {}

/// Immutable scene geometry tessellated and uploaded once for repeated drawing.
///
/// Camera movement, zoom, rotation, projection tilt, viewport resizing, and DPI
/// changes do not invalidate prepared geometry. Shape, style, gradient, layer,
/// or clip changes require preparing a replacement. A prepared scene can only be
/// rendered by the [`WgpuRenderer`] that created it. Use
/// [`WgpuRenderer::restore_prepared_scene`] to recreate its GPU buffer for a
/// replacement renderer after device loss.
pub struct PreparedScene {
    renderer_identity: Arc<()>,
    background: Color,
    vertex_buffer: Arc<wgpu::Buffer>,
    vertices: Arc<[Vertex]>,
    command_count: usize,
    vertex_count: usize,
    geometry_extents: GeometryExtents,
    draw_batches: Vec<PreparedDrawBatch>,
    tessellation: TessellationStats,
}

/// One world-space vertex in a dynamic triangle-list mesh.
///
/// Dynamic meshes use plain filled triangles. Positions are measured in world
/// units, `depth` uses the active camera projection, and colors are linear RGBA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicVertex2d {
    world_position: Vec2,
    depth: f32,
    color: Color,
}

impl DynamicVertex2d {
    /// Builds a finite dynamic vertex.
    pub fn new(world_position: Vec2, depth: f32, color: Color) -> Result<Self, DynamicMeshError> {
        if !world_position.is_finite() || !depth.is_finite() || !color.is_finite() {
            return Err(DynamicMeshError::InvalidVertex);
        }
        Ok(Self {
            world_position,
            depth,
            color,
        })
    }

    /// Returns the world-space triangle position.
    pub fn world_position(self) -> Vec2 {
        self.world_position
    }

    /// Returns pseudo-depth in caller-defined units.
    pub fn depth(self) -> f32 {
        self.depth
    }

    /// Returns the linear RGBA vertex color.
    pub fn color(self) -> Color {
        self.color
    }
}

/// Validation or ownership failure while changing a dynamic mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicMeshError {
    /// Triangle-list vertex counts must be divisible by three.
    InvalidVertexCount,
    /// A position, depth, or color contains NaN or infinity.
    InvalidVertex,
    /// A partial update lies outside the mesh's current vertex range.
    UpdateRangeOutOfBounds,
    /// The mesh belongs to a different renderer and GPU device.
    RendererMismatch,
}

impl fmt::Display for DynamicMeshError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVertexCount => write!(
                formatter,
                "dynamic mesh vertex count must be divisible by three"
            ),
            Self::InvalidVertex => write!(formatter, "dynamic mesh vertices must be finite"),
            Self::UpdateRangeOutOfBounds => write!(
                formatter,
                "dynamic mesh update range is outside the current mesh"
            ),
            Self::RendererMismatch => {
                write!(formatter, "dynamic mesh belongs to a different renderer")
            }
        }
    }
}

impl Error for DynamicMeshError {}

/// Mutable GPU triangle-list geometry for high-frequency simulation visuals.
///
/// A mesh retains its CPU vertices for validation and recreates its GPU buffer
/// only when a replacement exceeds its current capacity. Use
/// [`WgpuRenderer::update_dynamic_mesh_range`] for in-place partial updates.
pub struct DynamicMesh2d {
    renderer_identity: Arc<()>,
    vertex_buffer: Arc<wgpu::Buffer>,
    vertices: Vec<Vertex>,
    vertex_capacity: usize,
    geometry_extents: GeometryExtents,
}

impl DynamicMesh2d {
    /// Returns the number of triangle-list vertices currently stored.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the number of vertices that fit without reallocating its GPU buffer.
    pub fn vertex_capacity(&self) -> usize {
        self.vertex_capacity
    }

    /// Returns retained CPU memory used for validation and future updates.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<Vertex>()
    }
}

/// CPU-side outcome of one dynamic mesh update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicMeshUpdateReport {
    vertex_count: usize,
    upload: Duration,
    reallocated: bool,
}

impl DynamicMeshUpdateReport {
    /// Returns the current triangle-list vertex count after the update.
    pub fn vertex_count(self) -> usize {
        self.vertex_count
    }

    /// Returns CPU time spent validating, allocating when needed, and writing GPU data.
    pub fn upload(self) -> Duration {
        self.upload
    }

    /// Returns whether this update grew and replaced the mesh GPU buffer.
    pub fn reallocated(self) -> bool {
        self.reallocated
    }
}

/// One particle instance for the instanced particle renderer.
///
/// `world_position` is measured in world units, `radius` is measured in logical
/// screen pixels, `color` is linear RGBA, and `depth` uses the active camera's
/// pseudo-depth projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleInstance2d {
    world_position: Vec2,
    radius: f32,
    color: Color,
    depth: f32,
}

impl ParticleInstance2d {
    /// Builds a finite particle instance with a strictly positive logical-pixel radius.
    pub fn new(
        world_position: Vec2,
        radius: f32,
        color: Color,
        depth: f32,
    ) -> Result<Self, ParticleInstanceError> {
        if !world_position.is_finite()
            || !radius.is_finite()
            || !color.is_finite()
            || !depth.is_finite()
        {
            return Err(ParticleInstanceError::NonFinite);
        }
        if radius <= 0.0 {
            return Err(ParticleInstanceError::InvalidRadius);
        }
        Ok(Self {
            world_position,
            radius,
            color,
            depth,
        })
    }

    /// Returns the world-space particle center.
    pub fn world_position(self) -> Vec2 {
        self.world_position
    }

    /// Returns the radius in logical screen pixels.
    pub fn radius(self) -> f32 {
        self.radius
    }

    /// Returns the linear RGBA particle color.
    pub fn color(self) -> Color {
        self.color
    }

    /// Returns caller-defined pseudo-depth.
    pub fn depth(self) -> f32 {
        self.depth
    }
}

/// Rejection reason for a particle instance before it reaches GPU memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleInstanceError {
    /// Position, radius, color, or pseudo-depth contains NaN or infinity.
    NonFinite,
    /// Radius must be strictly positive in logical screen pixels.
    InvalidRadius,
}

impl fmt::Display for ParticleInstanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(formatter, "particle instance values must be finite"),
            Self::InvalidRadius => write!(formatter, "particle radius must be finite and positive"),
        }
    }
}

impl Error for ParticleInstanceError {}

/// Counts associated with one particle-field update or draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParticleStatistics {
    submitted: usize,
    visible: usize,
    culled: usize,
    dropped: usize,
    rendered: usize,
}

impl ParticleStatistics {
    /// Returns instances supplied by the host.
    pub fn submitted(self) -> usize {
        self.submitted
    }

    /// Returns instances remaining after future culling passes.
    pub fn visible(self) -> usize {
        self.visible
    }

    /// Returns instances outside the current camera viewport.
    pub fn culled(self) -> usize {
        self.culled
    }

    /// Returns instances rejected before GPU submission.
    pub fn dropped(self) -> usize {
        self.dropped
    }

    /// Returns instances submitted to the particle draw call.
    pub fn rendered(self) -> usize {
        self.rendered
    }
}

/// Renderer-owned instanced particles for high-volume simulation visuals.
///
/// The field retains its validated CPU instances so it can be recreated after
/// device loss. Its GPU instance buffer grows only when a replacement exceeds
/// the current capacity.
pub struct ParticleField2d {
    renderer_identity: Arc<()>,
    instance_buffer: Arc<wgpu::Buffer>,
    instances: Vec<ParticleGpu>,
    instance_capacity: usize,
    statistics: ParticleStatistics,
}

impl ParticleField2d {
    /// Returns the number of particle instances currently stored.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Returns the number of instances that fit without reallocating GPU memory.
    pub fn instance_capacity(&self) -> usize {
        self.instance_capacity
    }

    /// Returns the most recent update or draw counts.
    pub fn statistics(&self) -> ParticleStatistics {
        self.statistics
    }

    /// Returns retained CPU memory used for recovery and validation.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.instances.len() * std::mem::size_of::<ParticleGpu>()
    }
}

/// Failure while creating or updating a [`ParticleField2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleFieldError {
    /// An input instance violates the finite particle contract.
    InvalidInstance,
    /// A partial update lies outside the field's current instance range.
    UpdateRangeOutOfBounds,
    /// The field belongs to another renderer and GPU device.
    RendererMismatch,
}

impl fmt::Display for ParticleFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInstance => write!(formatter, "particle field instances must be valid"),
            Self::UpdateRangeOutOfBounds => {
                write!(
                    formatter,
                    "particle update range is outside the current field"
                )
            }
            Self::RendererMismatch => {
                write!(formatter, "particle field belongs to a different renderer")
            }
        }
    }
}

impl Error for ParticleFieldError {}

/// CPU-side outcome of one particle-field update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParticleFieldUpdateReport {
    statistics: ParticleStatistics,
    upload: Duration,
    reallocated: bool,
}

impl ParticleFieldUpdateReport {
    /// Returns submitted, visible, dropped, and rendered instance counts.
    pub fn statistics(self) -> ParticleStatistics {
        self.statistics
    }

    /// Returns CPU time spent validating, allocating when needed, and uploading.
    pub fn upload(self) -> Duration {
        self.upload
    }

    /// Returns whether this update grew and replaced the GPU instance buffer.
    pub fn reallocated(self) -> bool {
        self.reallocated
    }
}

/// Failure while rendering a [`ParticleField2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleFieldRenderError {
    /// The field belongs to another renderer and GPU device.
    RendererMismatch,
    /// The clear color is non-finite.
    InvalidBackground,
    /// Camera transformation would produce non-finite particle geometry.
    InvalidGeometryTransform,
    /// Surface acquisition or presentation failed.
    Frame(RendererFrameError),
}

impl fmt::Display for ParticleFieldRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererMismatch => {
                write!(formatter, "particle field belongs to a different renderer")
            }
            Self::InvalidBackground => {
                write!(formatter, "particle field background must be finite")
            }
            Self::InvalidGeometryTransform => {
                write!(formatter, "particle field geometry transform is invalid")
            }
            Self::Frame(error) => write!(formatter, "particle field frame failed: {error}"),
        }
    }
}

impl Error for ParticleFieldRenderError {}

/// Renderer-owned scalar texture retaining its validated source field.
pub struct ScalarFieldTexture {
    renderer_identity: Arc<()>,
    texture: wgpu::Texture,
    field: ScalarField,
}

impl ScalarFieldTexture {
    /// Returns the current grid width in texels.
    pub fn width(&self) -> usize {
        self.field.width()
    }

    /// Returns the current grid height in texels.
    pub fn height(&self) -> usize {
        self.field.height()
    }

    /// Returns retained CPU scalar data used for recovery and uploads.
    pub fn field(&self) -> &ScalarField {
        &self.field
    }

    /// Returns retained CPU scalar bytes used for device-loss recovery.
    pub fn recovery_memory_bytes(&self) -> usize {
        std::mem::size_of_val(self.field.values())
    }
}

/// Failure while creating or updating a [`ScalarFieldTexture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarFieldTextureError {
    /// The texture belongs to another renderer and GPU device.
    RendererMismatch,
    /// Grid dimensions exceed the addressable `wgpu` texture range.
    DimensionsTooLarge,
    /// A region does not fit inside the current scalar texture.
    UpdateRegionOutOfBounds,
    /// Region values did not match the region dimensions.
    InvalidUpdateValueCount,
    /// A region value was NaN or infinite.
    NonFiniteUpdateValue,
}

impl fmt::Display for ScalarFieldTextureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererMismatch => write!(
                formatter,
                "scalar field texture belongs to a different renderer"
            ),
            Self::DimensionsTooLarge => write!(
                formatter,
                "scalar field dimensions exceed GPU texture limits"
            ),
            Self::UpdateRegionOutOfBounds => {
                write!(formatter, "scalar texture update region is out of bounds")
            }
            Self::InvalidUpdateValueCount => {
                write!(
                    formatter,
                    "scalar texture update values do not match the region"
                )
            }
            Self::NonFiniteUpdateValue => {
                write!(formatter, "scalar texture update values must be finite")
            }
        }
    }
}

impl Error for ScalarFieldTextureError {}

/// Failure while rendering a scalar texture as a heatmap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarFieldRenderError {
    /// The texture belongs to another renderer and GPU device.
    RendererMismatch,
    /// The scalar range was non-finite or did not have positive extent.
    InvalidValueRange {
        /// Lower bound supplied by the caller.
        minimum: f32,
        /// Upper bound supplied by the caller.
        maximum: f32,
    },
    /// The clear color is non-finite.
    InvalidBackground,
    /// Surface acquisition or presentation failed.
    Frame(RendererFrameError),
}

impl fmt::Display for ScalarFieldRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererMismatch => write!(
                formatter,
                "scalar field texture belongs to a different renderer"
            ),
            Self::InvalidValueRange { minimum, maximum } => {
                write!(formatter, "invalid scalar value range {minimum}..{maximum}")
            }
            Self::InvalidBackground => write!(formatter, "scalar field background must be finite"),
            Self::Frame(error) => write!(formatter, "scalar field frame failed: {error}"),
        }
    }
}

impl Error for ScalarFieldRenderError {}

/// CPU-side outcome of one scalar-field texture upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarFieldUploadReport {
    upload: Duration,
    reallocated: bool,
}

impl ScalarFieldUploadReport {
    /// Returns CPU time spent allocating when needed and enqueuing the upload.
    pub fn upload(self) -> Duration {
        self.upload
    }

    /// Returns whether differing dimensions recreated the GPU texture.
    pub fn reallocated(self) -> bool {
        self.reallocated
    }
}

/// Failure while rendering a [`DynamicMesh2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicMeshRenderError {
    /// The mesh belongs to another renderer and GPU device.
    RendererMismatch,
    /// The clear color is non-finite.
    InvalidBackground,
    /// Frame rendering failed after dynamic-mesh ownership validation.
    Frame(RendererFrameError),
}

impl fmt::Display for DynamicMeshRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererMismatch => {
                write!(formatter, "dynamic mesh belongs to a different renderer")
            }
            Self::InvalidBackground => write!(formatter, "dynamic mesh background must be finite"),
            Self::Frame(error) => write!(formatter, "dynamic mesh frame failed: {error}"),
        }
    }
}

impl Error for DynamicMeshRenderError {}

impl PreparedScene {
    /// Returns the number of accepted source scene commands.
    pub fn command_count(&self) -> usize {
        self.command_count
    }

    /// Returns the number of triangle-list vertices stored in the GPU buffer.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Returns the number of clip-compatible draw batches.
    pub fn draw_batch_count(&self) -> usize {
        self.draw_batches.len()
    }

    /// Returns retained CPU vertex bytes available for device-loss recovery.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<Vertex>()
    }

    /// Returns command-level tessellation results captured during preparation.
    pub fn tessellation_stats(&self) -> TessellationStats {
        self.tessellation
    }
}

/// Presentation behavior for the window surface.
///
/// `Vsync` requests strict FIFO presentation for stable frame pacing and no
/// tearing. `NoVsync` requests the fastest available non-VSync mode, but may
/// fall back to FIFO when the platform exposes no Immediate or Mailbox mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererPresentMode {
    /// Use strict FIFO presentation synchronized to display refresh.
    Vsync,
    /// Prefer Immediate, then Mailbox, and fall back to FIFO when necessary.
    NoVsync,
}

impl RendererPresentMode {
    fn to_wgpu(self) -> wgpu::PresentMode {
        match self {
            Self::Vsync => wgpu::PresentMode::Fifo,
            Self::NoVsync => wgpu::PresentMode::AutoNoVsync,
        }
    }
}

/// Options used when creating a [`WgpuRenderer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WgpuRendererOptions {
    present_mode: RendererPresentMode,
    scale_factor: f64,
}

impl WgpuRendererOptions {
    /// Builds options from presentation behavior and display scale.
    ///
    /// `scale_factor` is physical surface pixels per logical screen pixel. It
    /// must be finite, positive, representable as `f32`, and large enough that
    /// every supported physical surface has finite logical dimensions.
    pub fn new(
        present_mode: RendererPresentMode,
        scale_factor: f64,
    ) -> Result<Self, RendererConfigurationError> {
        validate_scale_factor(scale_factor)?;
        Ok(Self {
            present_mode,
            scale_factor,
        })
    }

    /// Returns presentation behavior for the surface.
    pub fn present_mode(self) -> RendererPresentMode {
        self.present_mode
    }

    /// Returns physical surface pixels per logical screen pixel.
    pub fn scale_factor(self) -> f64 {
        self.scale_factor
    }
}

impl Default for WgpuRendererOptions {
    fn default() -> Self {
        Self {
            present_mode: RendererPresentMode::Vsync,
            scale_factor: 1.0,
        }
    }
}

struct MultisampleTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// `wgpu` backend that renders [`Scene`] commands into a presentation surface.
///
/// The renderer owns the GPU device, queue, surface, pipeline, and transient
/// vertex buffer. Window creation stays outside the library so each application
/// can choose its own windowing framework.
pub struct WgpuRenderer {
    renderer_identity: Arc<()>,
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    scale_factor: f64,
    pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,
    heatmap_pipeline: wgpu::RenderPipeline,
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    heatmap_uniform_buffer: wgpu::Buffer,
    heatmap_bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: Arc<wgpu::Buffer>,
    particle_unit_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    multisample_target: Option<MultisampleTarget>,
    sample_count: u32,
    vertices: Vec<Vertex>,
    draw_batches: Vec<PreparedDrawBatch>,
}

impl WgpuRenderer {
    /// Creates a renderer for a window or canvas surface target.
    ///
    /// `width` and `height` are physical surface pixels. Zero sizes are clamped
    /// to one pixel because `wgpu` surfaces cannot be configured at zero size.
    /// This convenience constructor assumes a display scale factor of `1.0`;
    /// HiDPI hosts should use [`WgpuRenderer::new_with_options`].
    pub async fn new(
        surface_target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Self, RendererInitError> {
        Self::new_with_options(
            surface_target,
            width,
            height,
            WgpuRendererOptions::default(),
        )
        .await
    }

    /// Creates a renderer with explicit options.
    ///
    /// Use this when measuring renderer throughput or when the host application
    /// needs control over presentation behavior or display scale.
    pub async fn new_with_options(
        surface_target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
        options: WgpuRendererOptions,
    ) -> Result<Self, RendererInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(surface_target)
            .map_err(RendererInitError::CreateSurface)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .map_err(RendererInitError::RequestAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-engine device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(RendererInitError::RequestDevice)?;

        let width = width.max(1);
        let height = height.max(1);
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(RendererInitError::NoSurfaceConfig)?;
        config.present_mode = options.present_mode().to_wgpu();
        surface.configure(&device, &config);

        let sample_count = preferred_sample_count(&adapter, config.format);
        let (
            pipeline,
            particle_pipeline,
            heatmap_pipeline,
            camera_uniform_buffer,
            camera_bind_group,
            heatmap_uniform_buffer,
            heatmap_bind_group_layout,
        ) = create_pipeline(&device, config.format, sample_count);
        let vertex_buffer = Arc::new(create_vertex_buffer(&device, INITIAL_VERTEX_CAPACITY));
        let particle_unit_buffer = create_particle_unit_buffer(&device, &queue);
        let multisample_target = create_multisample_target(&device, &config, sample_count);

        Ok(Self {
            renderer_identity: Arc::new(()),
            _instance: instance,
            surface,
            _adapter: adapter,
            device,
            queue,
            config,
            scale_factor: options.scale_factor(),
            pipeline,
            particle_pipeline,
            heatmap_pipeline,
            camera_uniform_buffer,
            camera_bind_group,
            heatmap_uniform_buffer,
            heatmap_bind_group_layout,
            vertex_buffer,
            particle_unit_buffer,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
            multisample_target,
            sample_count,
            vertices: Vec::with_capacity(INITIAL_VERTEX_CAPACITY),
            draw_batches: Vec::new(),
        })
    }

    /// Returns the configured surface size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Returns the render viewport size in logical screen pixels.
    ///
    /// Scene clipping, stroke widths, shadows, and camera zoom use this logical
    /// coordinate space so their visual size remains stable across display DPI.
    pub fn logical_size(&self) -> (f32, f32) {
        let scale_factor = self.scale_factor as f32;
        (
            self.config.width as f32 / scale_factor,
            self.config.height as f32 / scale_factor,
        )
    }

    /// Returns the current camera viewport in logical screen pixels.
    pub fn logical_viewport(&self) -> Result<LogicalViewport, crate::LogicalViewportError> {
        let (width, height) = self.logical_size();
        LogicalViewport::new(width, height)
    }

    /// Converts a physical surface position into logical screen pixels.
    pub fn physical_to_logical_screen(
        &self,
        position: PhysicalScreenPosition,
    ) -> Result<LogicalScreenPosition, RendererCoordinateError> {
        physical_to_logical_screen(position, self.scale_factor as f32)
    }

    /// Converts a logical screen position into physical surface pixels.
    pub fn logical_to_physical_screen(
        &self,
        position: LogicalScreenPosition,
    ) -> Result<PhysicalScreenPosition, RendererCoordinateError> {
        logical_to_physical_screen(position, self.scale_factor as f32)
    }

    /// Returns physical surface pixels per logical screen pixel.
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Replaces display scale without changing physical surface dimensions.
    ///
    /// Invalid values return an error and leave the current scale unchanged.
    pub fn set_scale_factor(
        &mut self,
        scale_factor: f64,
    ) -> Result<(), RendererConfigurationError> {
        validate_scale_factor(scale_factor)?;
        self.scale_factor = scale_factor;
        Ok(())
    }

    /// Reconfigures the surface after a host window resize.
    ///
    /// Zero width or height is ignored because minimized windows often report
    /// zero size and `wgpu` cannot configure a zero-sized surface.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.multisample_target =
            create_multisample_target(&self.device, &self.config, self.sample_count);
    }

    /// Reconfigures physical surface size and logical-to-physical display scale.
    ///
    /// Invalid scale returns an error without changing either value. Zero
    /// physical dimensions update the scale but leave the minimized surface
    /// configuration unchanged.
    pub fn resize_with_scale_factor(
        &mut self,
        width: u32,
        height: u32,
        scale_factor: f64,
    ) -> Result<(), RendererConfigurationError> {
        validate_scale_factor(scale_factor)?;
        self.scale_factor = scale_factor;
        self.resize(width, height);
        Ok(())
    }

    /// Draws a scene using the supplied camera.
    ///
    /// Scene positions and sizes are in world units unless a style explicitly
    /// says logical screen pixels. The renderer converts world coordinates to
    /// logical screen coordinates through [`Camera2d`], then to normalized device
    /// coordinates for `wgpu`. Scissor rectangles are converted to physical
    /// surface pixels at the final backend boundary.
    pub fn render(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
    ) -> Result<RenderStatus, RendererFrameError> {
        self.render_with_metrics(scene, camera)
            .map(RenderReport::status)
    }

    /// Draws a scene and returns a CPU-side timing breakdown.
    ///
    /// Metrics separate tessellation, buffer upload, surface acquisition, and
    /// encode/submit/present dispatch. They do not measure GPU completion or the
    /// monitor scanout timestamp. Use this for diagnostics; [`WgpuRenderer::render`]
    /// is the simpler equivalent when stage timings are not needed.
    pub fn render_with_metrics(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
    ) -> Result<RenderReport, RendererFrameError> {
        let frame_started_at = Instant::now();
        let tessellation_started_at = Instant::now();
        self.vertices.clear();
        self.draw_batches.clear();
        let tessellation_stats =
            tessellate_scene(scene, &mut self.vertices, &mut self.draw_batches);
        self.ensure_vertex_capacity(self.vertices.len());
        let tessellation = tessellation_started_at.elapsed();

        let upload_started_at = Instant::now();
        if !self.vertices.is_empty() {
            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        }
        let upload = upload_started_at.elapsed();

        let vertex_buffer = Arc::clone(&self.vertex_buffer);
        let draw_batches = self.draw_batches.clone();
        let geometry_extents = GeometryExtents::from_vertices(&self.vertices);
        self.draw_geometry(
            scene.background(),
            &vertex_buffer,
            self.vertices.len(),
            geometry_extents,
            &draw_batches,
            *camera,
            tessellation,
            upload,
            false,
            false,
            tessellation_stats,
            frame_started_at,
        )
    }

    /// Tessellates a scene once and uploads immutable geometry to a dedicated GPU buffer.
    ///
    /// Preparing is appropriate for geometry that will be drawn repeatedly while
    /// only the camera or target dimensions change. Any shape, style, gradient,
    /// ordering, or clipping change requires preparing a replacement scene.
    pub fn prepare_scene(&self, scene: &Scene) -> PreparedScene {
        prepare_scene_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            scene,
        )
    }

    /// Recreates prepared GPU resources on this renderer from a retained CPU snapshot.
    ///
    /// This supports renderer recreation after device loss without requiring the
    /// original high-level [`Scene`]. The returned snapshot belongs to this renderer.
    pub fn restore_prepared_scene(&self, source: &PreparedScene) -> PreparedScene {
        restore_prepared_scene_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            source,
        )
    }

    /// Recreates a dynamic mesh on this renderer from its retained CPU vertices.
    ///
    /// Use this after recreating a renderer following device loss. The returned
    /// mesh belongs to this renderer and preserves the source mesh's capacity so
    /// subsequent updates retain the same allocation behavior.
    pub fn restore_dynamic_mesh(&self, source: &DynamicMesh2d) -> DynamicMesh2d {
        restore_dynamic_mesh_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            source,
        )
    }

    /// Creates mutable triangle-list geometry for a frequently changing visual.
    ///
    /// The vertex count must be divisible by three. The mesh is owned by this
    /// renderer and may only be updated or drawn through it.
    pub fn create_dynamic_mesh(
        &self,
        vertices: &[DynamicVertex2d],
    ) -> Result<DynamicMesh2d, DynamicMeshError> {
        let vertices = dynamic_vertices_to_gpu(vertices)?;
        let vertex_capacity = dynamic_vertex_capacity(vertices.len());
        let vertex_buffer = Arc::new(create_vertex_buffer(&self.device, vertex_capacity));
        if !vertices.is_empty() {
            self.queue
                .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }
        Ok(DynamicMesh2d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            geometry_extents: GeometryExtents::from_vertices(&vertices),
            vertex_buffer,
            vertices,
            vertex_capacity,
        })
    }

    /// Replaces all dynamic mesh vertices, growing GPU capacity only when needed.
    pub fn update_dynamic_mesh(
        &self,
        mesh: &mut DynamicMesh2d,
        vertices: &[DynamicVertex2d],
    ) -> Result<(), DynamicMeshError> {
        self.update_dynamic_mesh_with_metrics(mesh, vertices)
            .map(|_| ())
    }

    /// Replaces all dynamic mesh vertices and returns CPU-side update metrics.
    pub fn update_dynamic_mesh_with_metrics(
        &self,
        mesh: &mut DynamicMesh2d,
        vertices: &[DynamicVertex2d],
    ) -> Result<DynamicMeshUpdateReport, DynamicMeshError> {
        let update_started_at = Instant::now();
        self.validate_dynamic_mesh(mesh)?;
        let vertices = dynamic_vertices_to_gpu(vertices)?;
        let reallocated = vertices.len() > mesh.vertex_capacity;
        if vertices.len() > mesh.vertex_capacity {
            mesh.vertex_capacity = dynamic_vertex_capacity(vertices.len());
            mesh.vertex_buffer = Arc::new(create_vertex_buffer(&self.device, mesh.vertex_capacity));
        }
        if !vertices.is_empty() {
            self.queue
                .write_buffer(&mesh.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }
        mesh.geometry_extents = GeometryExtents::from_vertices(&vertices);
        mesh.vertices = vertices;
        Ok(DynamicMeshUpdateReport {
            vertex_count: mesh.vertices.len(),
            upload: update_started_at.elapsed(),
            reallocated,
        })
    }

    /// Replaces a contiguous vertex range without reallocating the mesh buffer.
    ///
    /// The range must stay inside the current triangle list and its length must
    /// preserve triangle alignment, so callers cannot create partial triangles.
    pub fn update_dynamic_mesh_range(
        &self,
        mesh: &mut DynamicMesh2d,
        first_vertex: usize,
        vertices: &[DynamicVertex2d],
    ) -> Result<(), DynamicMeshError> {
        self.update_dynamic_mesh_range_with_metrics(mesh, first_vertex, vertices)
            .map(|_| ())
    }

    /// Replaces a triangle-aligned range and returns CPU-side update metrics.
    pub fn update_dynamic_mesh_range_with_metrics(
        &self,
        mesh: &mut DynamicMesh2d,
        first_vertex: usize,
        vertices: &[DynamicVertex2d],
    ) -> Result<DynamicMeshUpdateReport, DynamicMeshError> {
        let update_started_at = Instant::now();
        self.validate_dynamic_mesh(mesh)?;
        let end = first_vertex
            .checked_add(vertices.len())
            .ok_or(DynamicMeshError::UpdateRangeOutOfBounds)?;
        if !first_vertex.is_multiple_of(3)
            || !vertices.len().is_multiple_of(3)
            || end > mesh.vertices.len()
        {
            return Err(DynamicMeshError::UpdateRangeOutOfBounds);
        }
        let vertices = dynamic_vertices_to_gpu(vertices)?;
        if !vertices.is_empty() {
            let offset = (first_vertex * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress;
            self.queue
                .write_buffer(&mesh.vertex_buffer, offset, bytemuck::cast_slice(&vertices));
            mesh.vertices[first_vertex..end].copy_from_slice(&vertices);
            mesh.geometry_extents = GeometryExtents::from_vertices(&mesh.vertices);
        }
        Ok(DynamicMeshUpdateReport {
            vertex_count: mesh.vertices.len(),
            upload: update_started_at.elapsed(),
            reallocated: false,
        })
    }

    /// Draws dynamic triangle-list geometry with a finite clear color.
    pub fn render_dynamic_mesh(
        &mut self,
        mesh: &DynamicMesh2d,
        background: Color,
        camera: &Camera2d,
    ) -> Result<RenderStatus, DynamicMeshRenderError> {
        self.render_dynamic_mesh_with_metrics(mesh, background, camera)
            .map(RenderReport::status)
    }

    /// Draws dynamic triangle-list geometry and returns per-frame CPU metrics.
    pub fn render_dynamic_mesh_with_metrics(
        &mut self,
        mesh: &DynamicMesh2d,
        background: Color,
        camera: &Camera2d,
    ) -> Result<RenderReport, DynamicMeshRenderError> {
        self.validate_dynamic_mesh(mesh)
            .map_err(|_| DynamicMeshRenderError::RendererMismatch)?;
        if !background.is_finite() {
            return Err(DynamicMeshRenderError::InvalidBackground);
        }
        let draw_batches = (!mesh.vertices.is_empty()).then_some(PreparedDrawBatch {
            vertex_range: 0..mesh.vertices.len() as u32,
            screen_clip: None,
        });
        self.draw_geometry(
            background,
            &mesh.vertex_buffer,
            mesh.vertices.len(),
            mesh.geometry_extents,
            draw_batches.as_slice(),
            *camera,
            Duration::ZERO,
            Duration::ZERO,
            false,
            true,
            TessellationStats::default(),
            Instant::now(),
        )
        .map_err(DynamicMeshRenderError::Frame)
    }

    /// Creates an instanced particle field owned by this renderer.
    ///
    /// Each instance draws as one screen-space circle quad through a single GPU
    /// draw call. The retained CPU snapshot enables later restoration on a
    /// recreated renderer.
    pub fn create_particle_field(
        &self,
        instances: &[ParticleInstance2d],
    ) -> Result<ParticleField2d, ParticleFieldError> {
        let instances = particle_instances_to_gpu(instances)?;
        let instance_capacity = particle_instance_capacity(instances.len());
        let instance_buffer = Arc::new(create_particle_instance_buffer(
            &self.device,
            instance_capacity,
        ));
        if !instances.is_empty() {
            self.queue
                .write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&instances));
        }
        Ok(ParticleField2d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            instance_buffer,
            statistics: particle_statistics(instances.len(), instances.len(), 0),
            instances,
            instance_capacity,
        })
    }

    /// Replaces all particle instances, reusing GPU capacity whenever possible.
    pub fn update_particle_field(
        &self,
        field: &mut ParticleField2d,
        instances: &[ParticleInstance2d],
    ) -> Result<ParticleStatistics, ParticleFieldError> {
        self.update_particle_field_with_metrics(field, instances)
            .map(|report| report.statistics())
    }

    /// Replaces all particle instances and returns CPU-side upload metrics.
    pub fn update_particle_field_with_metrics(
        &self,
        field: &mut ParticleField2d,
        instances: &[ParticleInstance2d],
    ) -> Result<ParticleFieldUpdateReport, ParticleFieldError> {
        let update_started_at = Instant::now();
        self.validate_particle_field(field)?;
        let instances = particle_instances_to_gpu(instances)?;
        let reallocated = instances.len() > field.instance_capacity;
        if reallocated {
            field.instance_capacity = particle_instance_capacity(instances.len());
            field.instance_buffer = Arc::new(create_particle_instance_buffer(
                &self.device,
                field.instance_capacity,
            ));
        }
        if !instances.is_empty() {
            self.queue
                .write_buffer(&field.instance_buffer, 0, bytemuck::cast_slice(&instances));
        }
        field.statistics = particle_statistics(instances.len(), instances.len(), 0);
        field.instances = instances;
        Ok(ParticleFieldUpdateReport {
            statistics: field.statistics,
            upload: update_started_at.elapsed(),
            reallocated,
        })
    }

    /// Replaces a contiguous particle range without reallocating its GPU buffer.
    pub fn update_particle_field_range(
        &self,
        field: &mut ParticleField2d,
        first_instance: usize,
        instances: &[ParticleInstance2d],
    ) -> Result<ParticleStatistics, ParticleFieldError> {
        self.update_particle_field_range_with_metrics(field, first_instance, instances)
            .map(|report| report.statistics())
    }

    /// Replaces a particle range and returns CPU-side upload metrics.
    pub fn update_particle_field_range_with_metrics(
        &self,
        field: &mut ParticleField2d,
        first_instance: usize,
        instances: &[ParticleInstance2d],
    ) -> Result<ParticleFieldUpdateReport, ParticleFieldError> {
        let update_started_at = Instant::now();
        self.validate_particle_field(field)?;
        let range = particle_update_range(first_instance, instances.len(), field.instances.len())?;
        let instances = particle_instances_to_gpu(instances)?;
        if !instances.is_empty() {
            let offset = (range.start * std::mem::size_of::<ParticleGpu>()) as wgpu::BufferAddress;
            self.queue.write_buffer(
                &field.instance_buffer,
                offset,
                bytemuck::cast_slice(&instances),
            );
            field.instances[range].copy_from_slice(&instances);
        }
        field.statistics = particle_statistics(field.instances.len(), field.instances.len(), 0);
        Ok(ParticleFieldUpdateReport {
            statistics: field.statistics,
            upload: update_started_at.elapsed(),
            reallocated: false,
        })
    }

    /// Recreates a particle field on this renderer from retained CPU instances.
    pub fn restore_particle_field(&self, source: &ParticleField2d) -> ParticleField2d {
        let instance_buffer = Arc::new(create_particle_instance_buffer(
            &self.device,
            source.instance_capacity,
        ));
        if !source.instances.is_empty() {
            self.queue
                .write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&source.instances));
        }
        ParticleField2d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            instance_buffer,
            instances: source.instances.clone(),
            instance_capacity: source.instance_capacity,
            statistics: source.statistics,
        }
    }

    /// Creates a single-channel floating-point texture from a validated scalar grid.
    pub fn create_scalar_field_texture(
        &self,
        field: ScalarField,
    ) -> Result<ScalarFieldTexture, ScalarFieldTextureError> {
        create_scalar_field_texture_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            field,
        )
    }

    /// Replaces scalar data, recreating the texture only when dimensions change.
    pub fn update_scalar_field_texture(
        &self,
        texture: &mut ScalarFieldTexture,
        field: ScalarField,
    ) -> Result<ScalarFieldUploadReport, ScalarFieldTextureError> {
        let upload_started_at = Instant::now();
        self.validate_scalar_field_texture(texture)?;
        let reallocated = texture.width() != field.width() || texture.height() != field.height();
        if reallocated {
            texture.texture = create_scalar_field_texture(&self.device, &field)?;
        }
        upload_scalar_field_texture(&self.queue, &texture.texture, &field)?;
        texture.field = field;
        Ok(ScalarFieldUploadReport {
            upload: upload_started_at.elapsed(),
            reallocated,
        })
    }

    /// Updates a rectangular scalar texture region without recreating it.
    pub fn update_scalar_field_texture_region(
        &self,
        texture: &mut ScalarFieldTexture,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        values: &[f32],
    ) -> Result<ScalarFieldUploadReport, ScalarFieldTextureError> {
        let upload_started_at = Instant::now();
        self.validate_scalar_field_texture(texture)?;
        texture
            .field
            .replace_region(x, y, width, height, values)
            .map_err(|error| match error {
                crate::ScalarFieldError::InvalidValueCount { .. } => {
                    ScalarFieldTextureError::InvalidUpdateValueCount
                }
                crate::ScalarFieldError::NonFiniteValue => {
                    ScalarFieldTextureError::NonFiniteUpdateValue
                }
                _ => ScalarFieldTextureError::UpdateRegionOutOfBounds,
            })?;
        upload_scalar_field_texture_region(
            &self.queue,
            &texture.texture,
            x,
            y,
            width,
            height,
            values,
        )?;
        Ok(ScalarFieldUploadReport {
            upload: upload_started_at.elapsed(),
            reallocated: false,
        })
    }

    /// Recreates a scalar texture on this renderer from its retained CPU grid.
    pub fn restore_scalar_field_texture(
        &self,
        source: &ScalarFieldTexture,
    ) -> Result<ScalarFieldTexture, ScalarFieldTextureError> {
        self.create_scalar_field_texture(source.field.clone())
    }

    /// Draws a scalar texture across the logical viewport through a color map.
    ///
    /// `value_range` maps `minimum` to the first color-map entry and `maximum`
    /// to the last. Values outside the range clamp to those endpoints.
    pub fn render_scalar_field_texture(
        &mut self,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        value_range: (f32, f32),
        background: Color,
    ) -> Result<RenderStatus, ScalarFieldRenderError> {
        self.render_scalar_field_texture_with_metrics(texture, color_map, value_range, background)
            .map(RenderReport::status)
    }

    /// Draws a scalar texture and returns normal CPU-side renderer metrics.
    pub fn render_scalar_field_texture_with_metrics(
        &mut self,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        (minimum, maximum): (f32, f32),
        background: Color,
    ) -> Result<RenderReport, ScalarFieldRenderError> {
        self.validate_scalar_field_texture(texture)
            .map_err(|_| ScalarFieldRenderError::RendererMismatch)?;
        if !minimum.is_finite() || !maximum.is_finite() || maximum <= minimum {
            return Err(ScalarFieldRenderError::InvalidValueRange { minimum, maximum });
        }
        if !background.is_finite() {
            return Err(ScalarFieldRenderError::InvalidBackground);
        }
        self.draw_scalar_field_texture(
            texture,
            color_map,
            minimum,
            maximum,
            ScalarFieldSampling::Nearest,
            background,
            Instant::now(),
        )
        .map_err(ScalarFieldRenderError::Frame)
    }

    /// Draws a scalar texture with an explicit texel sampling mode.
    pub fn render_scalar_field_texture_with_sampling(
        &mut self,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        (minimum, maximum): (f32, f32),
        sampling: ScalarFieldSampling,
        background: Color,
    ) -> Result<RenderStatus, ScalarFieldRenderError> {
        self.validate_scalar_field_texture(texture)
            .map_err(|_| ScalarFieldRenderError::RendererMismatch)?;
        if !minimum.is_finite() || !maximum.is_finite() || maximum <= minimum {
            return Err(ScalarFieldRenderError::InvalidValueRange { minimum, maximum });
        }
        if !background.is_finite() {
            return Err(ScalarFieldRenderError::InvalidBackground);
        }
        self.draw_scalar_field_texture(
            texture,
            color_map,
            minimum,
            maximum,
            sampling,
            background,
            Instant::now(),
        )
        .map(RenderReport::status)
        .map_err(ScalarFieldRenderError::Frame)
    }

    /// Draws a particle field with a finite clear color.
    pub fn render_particle_field(
        &mut self,
        field: &mut ParticleField2d,
        background: Color,
        camera: &Camera2d,
    ) -> Result<RenderStatus, ParticleFieldRenderError> {
        self.render_particle_field_with_metrics(field, background, camera)
            .map(RenderReport::status)
    }

    /// Draws a particle field and returns regular per-frame renderer metrics.
    pub fn render_particle_field_with_metrics(
        &mut self,
        field: &mut ParticleField2d,
        background: Color,
        camera: &Camera2d,
    ) -> Result<RenderReport, ParticleFieldRenderError> {
        self.validate_particle_field(field)
            .map_err(|_| ParticleFieldRenderError::RendererMismatch)?;
        if !background.is_finite() {
            return Err(ParticleFieldRenderError::InvalidBackground);
        }
        let report = self
            .draw_particle_field(background, field, *camera, Instant::now())
            .map_err(|error| match error {
                RendererFrameError::InvalidGeometryTransform => {
                    ParticleFieldRenderError::InvalidGeometryTransform
                }
                other => ParticleFieldRenderError::Frame(other),
            })?;
        field.statistics.rendered = match report.status() {
            RenderStatus::Drawn => field.statistics.visible,
            RenderStatus::Skipped(_) => 0,
        };
        Ok(report)
    }

    /// Draws geometry previously uploaded by [`WgpuRenderer::prepare_scene`].
    ///
    /// Camera and viewport changes are applied by the vertex shader and do not
    /// rebuild or re-upload the prepared geometry.
    pub fn render_prepared(
        &mut self,
        scene: &PreparedScene,
        camera: &Camera2d,
    ) -> Result<RenderStatus, PreparedSceneRenderError> {
        self.render_prepared_with_metrics(scene, camera)
            .map(RenderReport::status)
    }

    /// Draws prepared geometry and reports per-frame CPU timing.
    ///
    /// Tessellation and geometry upload durations are zero because both happened
    /// in [`WgpuRenderer::prepare_scene`]. The camera uniform is still updated
    /// once per frame.
    pub fn render_prepared_with_metrics(
        &mut self,
        scene: &PreparedScene,
        camera: &Camera2d,
    ) -> Result<RenderReport, PreparedSceneRenderError> {
        if !prepared_scene_belongs_to(&self.renderer_identity, &scene.renderer_identity) {
            return Err(PreparedSceneRenderError::RendererMismatch);
        }

        self.draw_geometry(
            scene.background,
            &scene.vertex_buffer,
            scene.vertex_count,
            scene.geometry_extents,
            &scene.draw_batches,
            *camera,
            Duration::ZERO,
            Duration::ZERO,
            true,
            false,
            scene.tessellation,
            Instant::now(),
        )
        .map_err(PreparedSceneRenderError::Frame)
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_geometry(
        &mut self,
        background: Color,
        vertex_buffer: &wgpu::Buffer,
        vertex_count: usize,
        geometry_extents: GeometryExtents,
        draw_batches: &[PreparedDrawBatch],
        camera: Camera2d,
        tessellation: Duration,
        upload: Duration,
        geometry_reused: bool,
        geometry_streamed: bool,
        tessellation_stats: TessellationStats,
        frame_started_at: Instant,
    ) -> Result<RenderReport, RendererFrameError> {
        let (logical_width, logical_height) = self.logical_size();
        let viewport = LogicalViewport::new(logical_width, logical_height)
            .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
        let camera_uniform_upload_started_at = Instant::now();
        let Some(camera_uniform) = CameraUniform::new(camera, viewport) else {
            return Err(RendererFrameError::InvalidGeometryTransform);
        };
        if !geometry_extents.is_safe_for(camera_uniform) {
            return Err(RendererFrameError::InvalidGeometryTransform);
        }
        self.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        let camera_uniform_upload = camera_uniform_upload_started_at.elapsed();

        let surface_acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    tessellation,
                    upload,
                    camera_uniform_upload,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    geometry_reused,
                    geometry_streamed,
                    tessellation_stats,
                ));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
                    tessellation,
                    upload,
                    camera_uniform_upload,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    geometry_reused,
                    geometry_streamed,
                    tessellation_stats,
                ));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(self.config.width, self.config.height);
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                    tessellation,
                    upload,
                    camera_uniform_upload,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    geometry_reused,
                    geometry_streamed,
                    tessellation_stats,
                ));
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err(RendererFrameError::Surface(RendererSurfaceStatus::Lost));
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RendererFrameError::Surface(
                    RendererSurfaceStatus::Validation,
                ));
            }
        };
        let surface_acquire = surface_acquire_started_at.elapsed();

        let encode_submit_present_started_at = Instant::now();
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (view, resolve_target) = match &self.multisample_target {
            Some(target) => (&target.view, Some(&surface_view)),
            None => (&surface_view, None),
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine render encoder"),
            });

        {
            let color_attachment = wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background.to_wgpu()),
                    store: wgpu::StoreOp::Store,
                },
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine render pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if vertex_count > 0 {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                for batch in draw_batches {
                    let scissor = match batch.screen_clip {
                        Some(screen_clip) => {
                            let Some(scissor) = screen_clip_to_scissor(
                                screen_clip,
                                viewport,
                                self.scale_factor as f32,
                            ) else {
                                continue;
                            };
                            scissor
                        }
                        None => ScissorRect {
                            x: 0,
                            y: 0,
                            width: self.config.width,
                            height: self.config.height,
                        },
                    };
                    pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                    pass.draw(batch.vertex_range.clone(), 0..1);
                }
            }
        }

        self.queue.submit([encoder.finish()]);
        self.queue.present(surface_texture);
        let encode_submit_present = encode_submit_present_started_at.elapsed();

        Ok(render_report(
            RenderStatus::Drawn,
            tessellation,
            upload,
            camera_uniform_upload,
            surface_acquire,
            encode_submit_present,
            frame_started_at.elapsed(),
            geometry_reused,
            geometry_streamed,
            tessellation_stats,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_scalar_field_texture(
        &mut self,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        minimum: f32,
        maximum: f32,
        sampling: ScalarFieldSampling,
        background: Color,
        frame_started_at: Instant,
    ) -> Result<RenderReport, RendererFrameError> {
        let upload_started_at = Instant::now();
        let color_map_texture = create_color_map_texture(&self.device, &self.queue, color_map);
        let scalar_view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let color_map_view = color_map_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniform = HeatmapUniform {
            value_range: [minimum, maximum - minimum, 0.0, 0.0],
            dimensions: [
                texture.width() as u32,
                texture.height() as u32,
                sampling.shader_value(),
                0,
            ],
        };
        self.queue.write_buffer(
            &self.heatmap_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniform),
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine heatmap bind group"),
            layout: &self.heatmap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scalar_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.heatmap_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let upload = upload_started_at.elapsed();
        let acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    Duration::ZERO,
                    upload,
                    Duration::ZERO,
                    acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
                    Duration::ZERO,
                    upload,
                    Duration::ZERO,
                    acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(self.config.width, self.config.height);
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                    Duration::ZERO,
                    upload,
                    Duration::ZERO,
                    acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err(RendererFrameError::Surface(RendererSurfaceStatus::Lost));
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RendererFrameError::Surface(
                    RendererSurfaceStatus::Validation,
                ));
            }
        };
        let surface_acquire = acquire_started_at.elapsed();
        let encode_started_at = Instant::now();
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (view, resolve_target) = match &self.multisample_target {
            Some(target) => (&target.view, Some(&surface_view)),
            None => (&surface_view, None),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine heatmap encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine heatmap pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(background.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.heatmap_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(surface_texture);
        Ok(render_report(
            RenderStatus::Drawn,
            Duration::ZERO,
            upload,
            Duration::ZERO,
            surface_acquire,
            encode_started_at.elapsed(),
            frame_started_at.elapsed(),
            false,
            true,
            TessellationStats::default(),
        ))
    }

    fn draw_particle_field(
        &mut self,
        background: Color,
        field: &mut ParticleField2d,
        camera: Camera2d,
        frame_started_at: Instant,
    ) -> Result<RenderReport, RendererFrameError> {
        let (logical_width, logical_height) = self.logical_size();
        let viewport = LogicalViewport::new(logical_width, logical_height)
            .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
        let camera_uniform_upload_started_at = Instant::now();
        let Some(camera_uniform) = CameraUniform::new(camera, viewport) else {
            return Err(RendererFrameError::InvalidGeometryTransform);
        };
        let visible_instances =
            visible_particle_instances(&field.instances, camera_uniform, viewport)?;
        field.statistics = particle_statistics(field.instances.len(), visible_instances.len(), 0);
        let upload_started_at = Instant::now();
        if !visible_instances.is_empty() {
            self.queue.write_buffer(
                &field.instance_buffer,
                0,
                bytemuck::cast_slice(&visible_instances),
            );
        }
        let upload = upload_started_at.elapsed();
        self.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        let camera_uniform_upload = camera_uniform_upload_started_at.elapsed();

        let surface_acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    upload,
                    upload,
                    camera_uniform_upload,
                    surface_acquire_started_at.elapsed(),
                    upload,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
                    Duration::ZERO,
                    Duration::ZERO,
                    camera_uniform_upload,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(self.config.width, self.config.height);
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                    Duration::ZERO,
                    Duration::ZERO,
                    camera_uniform_upload,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err(RendererFrameError::Surface(RendererSurfaceStatus::Lost));
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RendererFrameError::Surface(
                    RendererSurfaceStatus::Validation,
                ));
            }
        };
        let surface_acquire = surface_acquire_started_at.elapsed();
        let encode_submit_present_started_at = Instant::now();
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (view, resolve_target) = match &self.multisample_target {
            Some(target) => (&target.view, Some(&surface_view)),
            None => (&surface_view, None),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine particle render encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine particle render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(background.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !visible_instances.is_empty() {
                pass.set_pipeline(&self.particle_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.particle_unit_buffer.slice(..));
                pass.set_vertex_buffer(1, field.instance_buffer.slice(..));
                pass.draw(0..6, 0..visible_instances.len() as u32);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(surface_texture);
        Ok(render_report(
            RenderStatus::Drawn,
            upload,
            Duration::ZERO,
            camera_uniform_upload,
            surface_acquire,
            encode_submit_present_started_at.elapsed(),
            frame_started_at.elapsed(),
            false,
            true,
            TessellationStats::default(),
        ))
    }

    fn ensure_vertex_capacity(&mut self, vertex_count: usize) {
        if vertex_count <= self.vertex_capacity {
            return;
        }

        self.vertex_capacity = vertex_count.next_power_of_two();
        self.vertex_buffer = Arc::new(create_vertex_buffer(&self.device, self.vertex_capacity));
    }

    fn validate_dynamic_mesh(&self, mesh: &DynamicMesh2d) -> Result<(), DynamicMeshError> {
        prepared_scene_belongs_to(&self.renderer_identity, &mesh.renderer_identity)
            .then_some(())
            .ok_or(DynamicMeshError::RendererMismatch)
    }

    fn validate_particle_field(&self, field: &ParticleField2d) -> Result<(), ParticleFieldError> {
        prepared_scene_belongs_to(&self.renderer_identity, &field.renderer_identity)
            .then_some(())
            .ok_or(ParticleFieldError::RendererMismatch)
    }

    fn validate_scalar_field_texture(
        &self,
        texture: &ScalarFieldTexture,
    ) -> Result<(), ScalarFieldTextureError> {
        prepared_scene_belongs_to(&self.renderer_identity, &texture.renderer_identity)
            .then_some(())
            .ok_or(ScalarFieldTextureError::RendererMismatch)
    }
}

fn prepared_scene_belongs_to(renderer_identity: &Arc<()>, scene_identity: &Arc<()>) -> bool {
    Arc::ptr_eq(renderer_identity, scene_identity)
}

fn dynamic_vertices_to_gpu(vertices: &[DynamicVertex2d]) -> Result<Vec<Vertex>, DynamicMeshError> {
    if !vertices.len().is_multiple_of(3) {
        return Err(DynamicMeshError::InvalidVertexCount);
    }
    vertices
        .iter()
        .copied()
        .map(|vertex| {
            if !vertex.world_position.is_finite()
                || !vertex.depth.is_finite()
                || !vertex.color.is_finite()
            {
                return Err(DynamicMeshError::InvalidVertex);
            }
            Ok(Vertex {
                world_position: [vertex.world_position.x, vertex.world_position.y],
                depth: vertex.depth,
                screen_offset: [0.0; 2],
                previous_direction: [0.0; 2],
                next_direction: [0.0; 2],
                normal_distance: 0.0,
                color: vertex.color.to_array(),
            })
        })
        .collect()
}

fn dynamic_vertex_capacity(vertex_count: usize) -> usize {
    vertex_count.max(1).next_power_of_two()
}

fn particle_instances_to_gpu(
    instances: &[ParticleInstance2d],
) -> Result<Vec<ParticleGpu>, ParticleFieldError> {
    instances
        .iter()
        .copied()
        .map(|instance| {
            if !instance.world_position.is_finite()
                || !instance.radius.is_finite()
                || instance.radius <= 0.0
                || !instance.color.is_finite()
                || !instance.depth.is_finite()
            {
                return Err(ParticleFieldError::InvalidInstance);
            }
            Ok(ParticleGpu {
                world_position: [instance.world_position.x, instance.world_position.y],
                depth: instance.depth,
                radius: instance.radius,
                color: instance.color.to_array(),
            })
        })
        .collect()
}

fn particle_instance_capacity(instance_count: usize) -> usize {
    instance_count.max(1).next_power_of_two()
}

fn particle_update_range(
    first_instance: usize,
    replacement_count: usize,
    current_count: usize,
) -> Result<Range<usize>, ParticleFieldError> {
    let end = first_instance
        .checked_add(replacement_count)
        .ok_or(ParticleFieldError::UpdateRangeOutOfBounds)?;
    (end <= current_count)
        .then_some(first_instance..end)
        .ok_or(ParticleFieldError::UpdateRangeOutOfBounds)
}

fn visible_particle_instances(
    instances: &[ParticleGpu],
    camera: CameraUniform,
    viewport: LogicalViewport,
) -> Result<Vec<ParticleGpu>, RendererFrameError> {
    if !instances
        .iter()
        .copied()
        .all(|instance| instance.is_safe_for(camera))
    {
        return Err(RendererFrameError::InvalidGeometryTransform);
    }
    Ok(instances
        .iter()
        .copied()
        .filter(|instance| instance.intersects_viewport(camera, viewport))
        .collect())
}

fn particle_statistics(
    instance_count: usize,
    visible: usize,
    rendered: usize,
) -> ParticleStatistics {
    ParticleStatistics {
        submitted: instance_count,
        visible,
        culled: instance_count.saturating_sub(visible),
        dropped: 0,
        rendered,
    }
}

fn prepare_scene_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    scene: &Scene,
) -> PreparedScene {
    let mut vertices = Vec::new();
    let mut draw_batches = Vec::new();
    let tessellation = tessellate_scene(scene, &mut vertices, &mut draw_batches);
    let geometry_extents = GeometryExtents::from_vertices(&vertices);
    let vertex_buffer = Arc::new(create_vertex_buffer(device, vertices.len()));
    if !vertices.is_empty() {
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
    }
    let vertex_count = vertices.len();
    let vertices: Arc<[Vertex]> = vertices.into();

    PreparedScene {
        renderer_identity,
        background: scene.background(),
        vertex_buffer,
        vertices,
        command_count: scene.command_count(),
        vertex_count,
        geometry_extents,
        draw_batches,
        tessellation,
    }
}

fn restore_prepared_scene_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    source: &PreparedScene,
) -> PreparedScene {
    let vertex_buffer = Arc::new(create_vertex_buffer(device, source.vertices.len()));
    if !source.vertices.is_empty() {
        queue.write_buffer(
            &vertex_buffer,
            0,
            bytemuck::cast_slice(source.vertices.as_ref()),
        );
    }

    PreparedScene {
        renderer_identity,
        background: source.background,
        vertex_buffer,
        vertices: Arc::clone(&source.vertices),
        command_count: source.command_count,
        vertex_count: source.vertex_count,
        geometry_extents: source.geometry_extents,
        draw_batches: source.draw_batches.clone(),
        tessellation: source.tessellation,
    }
}

fn restore_dynamic_mesh_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    source: &DynamicMesh2d,
) -> DynamicMesh2d {
    let vertex_buffer = Arc::new(create_vertex_buffer(device, source.vertex_capacity));
    if !source.vertices.is_empty() {
        queue.write_buffer(
            &vertex_buffer,
            0,
            bytemuck::cast_slice(source.vertices.as_slice()),
        );
    }
    DynamicMesh2d {
        renderer_identity,
        vertex_buffer,
        vertices: source.vertices.clone(),
        vertex_capacity: source.vertex_capacity,
        geometry_extents: source.geometry_extents,
    }
}

#[allow(clippy::too_many_arguments)]
fn render_report(
    status: RenderStatus,
    tessellation: Duration,
    upload: Duration,
    camera_uniform_upload: Duration,
    surface_acquire: Duration,
    encode_submit_present: Duration,
    total_cpu: Duration,
    geometry_reused: bool,
    geometry_streamed: bool,
    tessellation_stats: TessellationStats,
) -> RenderReport {
    RenderReport {
        status,
        metrics: RendererFrameMetrics {
            tessellation,
            upload,
            camera_uniform_upload,
            surface_acquire,
            encode_submit_present,
            total_cpu,
            geometry_reused,
            geometry_streamed,
            tessellation_stats,
        },
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::Buffer,
    wgpu::BindGroup,
    wgpu::Buffer,
    wgpu::BindGroupLayout,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sim-engine flat color shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("primitive.wgsl"))),
    });
    let camera_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine camera uniform buffer"),
        size: std::mem::size_of::<CameraUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let camera_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine camera bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sim-engine camera bind group"),
        layout: &camera_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_uniform_buffer.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sim-engine shape pipeline layout"),
        bind_group_layouts: &[Some(&camera_bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine shape pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(Vertex::LAYOUT)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine particle pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("particle_vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(ParticleUnitVertex::LAYOUT), Some(ParticleGpu::LAYOUT)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let heatmap_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine heatmap uniform buffer"),
        size: std::mem::size_of::<HeatmapUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let heatmap_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine heatmap bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let heatmap_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sim-engine heatmap pipeline layout"),
        bind_group_layouts: &[Some(&heatmap_bind_group_layout)],
        immediate_size: 0,
    });
    let heatmap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine heatmap pipeline"),
        layout: Some(&heatmap_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("heatmap_vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("heatmap_fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    (
        pipeline,
        particle_pipeline,
        heatmap_pipeline,
        camera_uniform_buffer,
        camera_bind_group,
        heatmap_uniform_buffer,
        heatmap_bind_group_layout,
    )
}

fn preferred_sample_count(adapter: &wgpu::Adapter, format: wgpu::TextureFormat) -> u32 {
    let flags = adapter.get_texture_format_features(format).flags;
    if flags.contains(
        wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4
            | wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE,
    ) {
        PREFERRED_SAMPLE_COUNT
    } else {
        1
    }
}

fn create_vertex_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine vertex buffer"),
        size: (capacity.max(1) * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn scalar_field_extent(field: &ScalarField) -> Result<wgpu::Extent3d, ScalarFieldTextureError> {
    let width =
        u32::try_from(field.width()).map_err(|_| ScalarFieldTextureError::DimensionsTooLarge)?;
    let height =
        u32::try_from(field.height()).map_err(|_| ScalarFieldTextureError::DimensionsTooLarge)?;
    let _bytes_per_row = width
        .checked_mul(std::mem::size_of::<f32>() as u32)
        .ok_or(ScalarFieldTextureError::DimensionsTooLarge)?;
    Ok(wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    })
}

fn create_scalar_field_texture(
    device: &wgpu::Device,
    field: &ScalarField,
) -> Result<wgpu::Texture, ScalarFieldTextureError> {
    let extent = scalar_field_extent(field)?;
    Ok(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine scalar field texture"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }))
}

fn upload_scalar_field_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    field: &ScalarField,
) -> Result<(), ScalarFieldTextureError> {
    let extent = scalar_field_extent(field)?;
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(field.values()),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(extent.width * std::mem::size_of::<f32>() as u32),
            rows_per_image: Some(extent.height),
        },
        extent,
    );
    Ok(())
}

fn upload_scalar_field_texture_region(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    values: &[f32],
) -> Result<(), ScalarFieldTextureError> {
    let width = u32::try_from(width).map_err(|_| ScalarFieldTextureError::DimensionsTooLarge)?;
    let height = u32::try_from(height).map_err(|_| ScalarFieldTextureError::DimensionsTooLarge)?;
    let x = u32::try_from(x).map_err(|_| ScalarFieldTextureError::DimensionsTooLarge)?;
    let y = u32::try_from(y).map_err(|_| ScalarFieldTextureError::DimensionsTooLarge)?;
    let bytes_per_row = width
        .checked_mul(std::mem::size_of::<f32>() as u32)
        .ok_or(ScalarFieldTextureError::DimensionsTooLarge)?;
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(values),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

fn create_scalar_field_texture_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    field: ScalarField,
) -> Result<ScalarFieldTexture, ScalarFieldTextureError> {
    let texture = create_scalar_field_texture(device, &field)?;
    upload_scalar_field_texture(queue, &texture, &field)?;
    Ok(ScalarFieldTexture {
        renderer_identity,
        texture,
        field,
    })
}

fn create_color_map_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_map: &ColorMap,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine color map texture"),
        size: wgpu::Extent3d {
            width: 256,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut bytes = Vec::with_capacity(256 * 4);
    for index in 0..256 {
        let color = color_map
            .sample(index as f32 / 255.0)
            .expect("finite color map sample");
        bytes.extend(
            color
                .to_array()
                .map(|channel| (channel * 255.0).round() as u8),
        );
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(1024),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 256,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    texture
}

fn create_particle_unit_buffer(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine particle unit quad buffer"),
        size: (6 * std::mem::size_of::<ParticleUnitVertex>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let unit_quad = [
        ParticleUnitVertex {
            direction: [-1.0, -1.0],
        },
        ParticleUnitVertex {
            direction: [1.0, -1.0],
        },
        ParticleUnitVertex {
            direction: [1.0, 1.0],
        },
        ParticleUnitVertex {
            direction: [-1.0, -1.0],
        },
        ParticleUnitVertex {
            direction: [1.0, 1.0],
        },
        ParticleUnitVertex {
            direction: [-1.0, 1.0],
        },
    ];
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&unit_quad));
    buffer
}

fn create_particle_instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine particle instance buffer"),
        size: (capacity.max(1) * std::mem::size_of::<ParticleGpu>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_multisample_target(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
) -> Option<MultisampleTarget> {
    if sample_count <= 1 {
        return None;
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine multisample target"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    Some(MultisampleTarget {
        _texture: texture,
        view,
    })
}

fn validate_scale_factor(scale_factor: f64) -> Result<(), RendererConfigurationError> {
    let scale_factor_f32 = scale_factor as f32;
    let minimum_scale_factor = u32::MAX as f64 / f32::MAX as f64;
    if scale_factor.is_finite()
        && scale_factor_f32.is_finite()
        && scale_factor >= minimum_scale_factor
    {
        Ok(())
    } else {
        Err(RendererConfigurationError::InvalidScaleFactor { scale_factor })
    }
}

fn physical_to_logical_screen(
    position: PhysicalScreenPosition,
    scale_factor: f32,
) -> Result<LogicalScreenPosition, RendererCoordinateError> {
    let logical = position.to_vec2() / scale_factor;
    (position.is_finite() && logical.is_finite())
        .then_some(LogicalScreenPosition::from_vec2(logical))
        .ok_or(RendererCoordinateError::NonFiniteConversion)
}

fn logical_to_physical_screen(
    position: LogicalScreenPosition,
    scale_factor: f32,
) -> Result<PhysicalScreenPosition, RendererCoordinateError> {
    let physical = position.to_vec2() * scale_factor;
    (position.is_finite() && physical.is_finite())
        .then_some(PhysicalScreenPosition::from_vec2(physical))
        .ok_or(RendererCoordinateError::NonFiniteConversion)
}

fn tessellate_scene(
    scene: &Scene,
    vertices: &mut Vec<Vertex>,
    draw_batches: &mut Vec<PreparedDrawBatch>,
) -> TessellationStats {
    let mut stats = TessellationStats {
        command_count: scene.command_count(),
        ..TessellationStats::default()
    };
    for scene_command in scene.commands() {
        let screen_clip = scene_command.screen_clip();
        let vertex_start = vertices.len();

        match scene_command.command() {
            DrawCommand::Circle(circle) => tessellate_circle(circle, vertices),
            DrawCommand::Rect(rectangle) => tessellate_rect(
                rectangle.rect(),
                rectangle.corner_radius(),
                rectangle.style(),
                vertices,
            ),
            DrawCommand::Line(line) => tessellate_line(line, vertices),
            DrawCommand::Polyline(polyline) => {
                tessellate_polyline(polyline, vertices);
            }
        }

        for vertex in &mut vertices[vertex_start..] {
            vertex.depth = scene_command.depth();
        }

        let vertex_end = vertices.len();
        if vertices[vertex_start..]
            .iter()
            .any(|vertex| !vertex.is_finite())
        {
            vertices.truncate(vertex_start);
            stats.dropped_command_count += 1;
            continue;
        }
        if vertex_end == vertex_start {
            stats.dropped_command_count += 1;
            continue;
        }

        stats.rendered_command_count += 1;

        let vertex_start = vertex_start as u32;
        let vertex_end = vertex_end as u32;
        match draw_batches.last_mut() {
            Some(batch)
                if batch.screen_clip == screen_clip && batch.vertex_range.end == vertex_start =>
            {
                batch.vertex_range.end = vertex_end;
            }
            _ => draw_batches.push(PreparedDrawBatch {
                vertex_range: vertex_start..vertex_end,
                screen_clip,
            }),
        }
    }
    stats
}

fn screen_clip_to_scissor(
    screen_clip: ScreenClipRect,
    viewport: LogicalViewport,
    scale_factor: f32,
) -> Option<ScissorRect> {
    let rect = screen_clip.rect();
    if !rect.min.is_finite()
        || !rect.max.is_finite()
        || !scale_factor.is_finite()
        || scale_factor <= 0.0
    {
        return None;
    }
    let rect = rect.normalized();
    let physical_width = (viewport.width() * scale_factor).round();
    let physical_height = (viewport.height() * scale_factor).round();

    let min_x = (rect.min.x * scale_factor)
        .floor()
        .clamp(0.0, physical_width);
    let min_y = (rect.min.y * scale_factor)
        .floor()
        .clamp(0.0, physical_height);
    let max_x = (rect.max.x * scale_factor)
        .ceil()
        .clamp(0.0, physical_width);
    let max_y = (rect.max.y * scale_factor)
        .ceil()
        .clamp(0.0, physical_height);
    if max_x <= min_x || max_y <= min_y {
        return None;
    }

    Some(ScissorRect {
        x: min_x as u32,
        y: min_y as u32,
        width: (max_x - min_x) as u32,
        height: (max_y - min_y) as u32,
    })
}

fn tessellate_circle(circle: &Circle, vertices: &mut Vec<Vertex>) {
    if circle.radius() <= 0.0 {
        return;
    }

    if let Some(shadow) = circle.style().shadow() {
        push_circle_shadow_world(circle.center(), circle.radius(), shadow, vertices);
    }

    if let Some(fill) = circle.style().fill() {
        push_circle_fill_world(circle.center(), circle.radius(), fill, Vec2::ZERO, vertices);
    }

    if let Some(stroke) = circle.style().stroke() {
        push_circle_stroke_world(circle.center(), circle.radius(), stroke, vertices);
    }
}

fn push_circle_shadow_world(
    center_world: Vec2,
    radius_world: f32,
    shadow: Shadow,
    vertices: &mut Vec<Vertex>,
) {
    push_circle_fill_world(
        center_world,
        radius_world,
        Fill::Solid(shadow.color()),
        shadow.offset(),
        vertices,
    );

    if shadow.spread() > 0.0 {
        let points = circle_world_points(center_world, radius_world);
        push_closed_polyline_world(
            &points,
            shadow.spread() * 2.0,
            shadow.color(),
            shadow.offset(),
            vertices,
        );
    }
}

fn push_circle_stroke_world(
    center_world: Vec2,
    radius_world: f32,
    stroke: Stroke,
    vertices: &mut Vec<Vertex>,
) {
    let points = circle_world_points(center_world, radius_world);
    push_closed_polyline_world(
        &points,
        stroke.width(),
        stroke.color(),
        Vec2::ZERO,
        vertices,
    );
}

fn tessellate_rect(rect: Rect, corner_radius: f32, style: ShapeStyle, vertices: &mut Vec<Vertex>) {
    let rect = rect.normalized();
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }

    if let Some(shadow) = style.shadow() {
        push_rect_world(
            rect,
            corner_radius,
            Fill::Solid(shadow.color()),
            shadow.offset(),
            vertices,
        );
        if shadow.spread() > 0.0 {
            let points = rounded_rect_points(rect, corner_radius);
            push_closed_polyline_world(
                &points,
                shadow.spread() * 2.0,
                shadow.color(),
                shadow.offset(),
                vertices,
            );
        }
    }

    if let Some(fill) = style.fill() {
        push_rect_world(rect, corner_radius, fill, Vec2::ZERO, vertices);
    }

    if let Some(stroke) = style.stroke() {
        let points = rounded_rect_points(rect, corner_radius);
        push_closed_polyline_world(
            &points,
            stroke.width(),
            stroke.color(),
            Vec2::ZERO,
            vertices,
        );
    }
}

fn tessellate_line(line: &Line, vertices: &mut Vec<Vertex>) {
    push_round_line_world(
        line.from(),
        line.to(),
        line.stroke().width(),
        line.stroke().color(),
        Vec2::ZERO,
        vertices,
    );
}

fn tessellate_polyline(polyline: &Polyline, vertices: &mut Vec<Vertex>) {
    let mut emitted_segment = false;
    for pair in polyline.points().windows(2) {
        emitted_segment |= push_line_body_world(
            pair[0],
            pair[1],
            polyline.stroke().width(),
            polyline.stroke().color(),
            Vec2::ZERO,
            vertices,
        );
    }

    if emitted_segment {
        let radius = polyline.stroke().width() * 0.5;
        for point in polyline.points() {
            push_circle_screen_at_world(
                *point,
                radius,
                polyline.stroke().color(),
                Vec2::ZERO,
                vertices,
            );
        }
    }
}

fn push_circle_screen_at_world(
    center_world: Vec2,
    radius: f32,
    color: Color,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) {
    if radius <= 0.0 {
        return;
    }

    let center_vertex = world_vertex(center_world, screen_offset, color);
    for index in 0..CIRCLE_SEGMENTS {
        let angle_start = index as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        let angle_end = (index + 1) as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        vertices.push(center_vertex);
        vertices.push(world_vertex(
            center_world,
            screen_offset + Vec2::new(angle_start.cos(), -angle_start.sin()) * radius,
            color,
        ));
        vertices.push(world_vertex(
            center_world,
            screen_offset + Vec2::new(angle_end.cos(), -angle_end.sin()) * radius,
            color,
        ));
    }
}

fn push_circle_fill_world(
    center_world: Vec2,
    radius_world: f32,
    fill: Fill,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) {
    if radius_world <= 0.0 {
        return;
    }

    let center_vertex = world_vertex(center_world, screen_offset, fill.color_at(center_world));

    for index in 0..CIRCLE_SEGMENTS {
        let angle_start = index as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        let angle_end = (index + 1) as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        let world_start =
            center_world + Vec2::new(angle_start.cos(), angle_start.sin()) * radius_world;
        let world_end = center_world + Vec2::new(angle_end.cos(), angle_end.sin()) * radius_world;

        vertices.push(center_vertex);
        vertices.push(world_vertex(
            world_start,
            screen_offset,
            fill.color_at(world_start),
        ));
        vertices.push(world_vertex(
            world_end,
            screen_offset,
            fill.color_at(world_end),
        ));
    }
}

fn circle_world_points(center_world: Vec2, radius_world: f32) -> Vec<Vec2> {
    let mut points = Vec::with_capacity(CIRCLE_SEGMENTS + 1);
    for index in 0..=CIRCLE_SEGMENTS {
        let angle = index as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        points.push(center_world + Vec2::new(angle.cos(), angle.sin()) * radius_world);
    }
    points
}

fn push_rect_world(
    rect: Rect,
    corner_radius: f32,
    fill: Fill,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) {
    let points = rounded_rect_points(rect, corner_radius);
    if points.len() < 3 {
        return;
    }

    let center = rect.center();
    for index in 0..points.len() - 1 {
        vertices.push(world_vertex(
            center,
            screen_offset,
            fill.color_at(rect.center()),
        ));
        vertices.push(world_vertex(
            points[index],
            screen_offset,
            fill.color_at(points[index]),
        ));
        vertices.push(world_vertex(
            points[index + 1],
            screen_offset,
            fill.color_at(points[index + 1]),
        ));
    }
}

fn rounded_rect_points(rect: Rect, corner_radius: f32) -> Vec<Vec2> {
    let radius = corner_radius
        .max(0.0)
        .min(rect.width().abs() * 0.5)
        .min(rect.height().abs() * 0.5);

    if radius <= f32::EPSILON {
        return vec![
            Vec2::new(rect.max.x, rect.min.y),
            Vec2::new(rect.max.x, rect.max.y),
            Vec2::new(rect.min.x, rect.max.y),
            Vec2::new(rect.min.x, rect.min.y),
            Vec2::new(rect.max.x, rect.min.y),
        ];
    }

    let corners = [
        (
            Vec2::new(rect.max.x - radius, rect.max.y - radius),
            0.0,
            std::f32::consts::FRAC_PI_2,
        ),
        (
            Vec2::new(rect.min.x + radius, rect.max.y - radius),
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
        ),
        (
            Vec2::new(rect.min.x + radius, rect.min.y + radius),
            std::f32::consts::PI,
            std::f32::consts::PI * 1.5,
        ),
        (
            Vec2::new(rect.max.x - radius, rect.min.y + radius),
            std::f32::consts::PI * 1.5,
            std::f32::consts::TAU,
        ),
    ];

    let mut points = Vec::with_capacity(CORNER_SEGMENTS * 4 + 1);
    for (center, start_angle, end_angle) in corners {
        for step in 0..=CORNER_SEGMENTS {
            let amount = step as f32 / CORNER_SEGMENTS as f32;
            let angle = start_angle + (end_angle - start_angle) * amount;
            points.push(center + Vec2::new(angle.cos(), angle.sin()) * radius);
        }
    }
    points.push(points[0]);

    points
}

fn push_round_line_world(
    from: Vec2,
    to: Vec2,
    width: f32,
    color: Color,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) {
    if push_line_body_world(from, to, width, color, screen_offset, vertices) {
        let radius = width * 0.5;
        push_circle_screen_at_world(from, radius, color, screen_offset, vertices);
        push_circle_screen_at_world(to, radius, color, screen_offset, vertices);
    }
}

fn push_closed_polyline_world(
    points: &[Vec2],
    width: f32,
    color: Color,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) {
    if points.len() < 4 || width <= 0.0 || !width.is_finite() {
        return;
    }

    let mut unique_points = Vec::with_capacity(points.len() - 1);
    for point in &points[..points.len() - 1] {
        if unique_points
            .last()
            .is_none_or(|previous| (*point - *previous).length_squared() > f32::EPSILON)
        {
            unique_points.push(*point);
        }
    }
    if unique_points.len() > 1 {
        let first = unique_points[0];
        let last = unique_points[unique_points.len() - 1];
        if (first - last).length_squared() <= f32::EPSILON {
            unique_points.pop();
        }
    }
    if unique_points.len() < 3 {
        return;
    }

    let point_count = unique_points.len();
    let half_width = width * 0.5;
    for index in 0..point_count {
        let next = (index + 1) % point_count;
        let previous = (index + point_count - 1) % point_count;
        let after_next = (next + 1) % point_count;
        let current_previous_direction = unique_points[index] - unique_points[previous];
        let current_next_direction = unique_points[next] - unique_points[index];
        let next_next_direction = unique_points[after_next] - unique_points[next];

        vertices.push(stroke_vertex(
            unique_points[index],
            screen_offset,
            current_previous_direction,
            current_next_direction,
            half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            unique_points[next],
            screen_offset,
            current_next_direction,
            next_next_direction,
            half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            unique_points[next],
            screen_offset,
            current_next_direction,
            next_next_direction,
            -half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            unique_points[index],
            screen_offset,
            current_previous_direction,
            current_next_direction,
            half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            unique_points[next],
            screen_offset,
            current_next_direction,
            next_next_direction,
            -half_width,
            color,
        ));
        vertices.push(stroke_vertex(
            unique_points[index],
            screen_offset,
            current_previous_direction,
            current_next_direction,
            -half_width,
            color,
        ));
    }
}

fn push_line_body_world(
    from: Vec2,
    to: Vec2,
    width: f32,
    color: Color,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) -> bool {
    if width <= 0.0 {
        return false;
    }

    let delta = to - from;
    if !delta.is_finite() || (delta.x == 0.0 && delta.y == 0.0) {
        return false;
    }

    let half_width = width * 0.5;
    vertices.push(stroke_vertex(
        from,
        screen_offset,
        delta,
        delta,
        half_width,
        color,
    ));
    vertices.push(stroke_vertex(
        to,
        screen_offset,
        delta,
        delta,
        half_width,
        color,
    ));
    vertices.push(stroke_vertex(
        to,
        screen_offset,
        delta,
        delta,
        -half_width,
        color,
    ));
    vertices.push(stroke_vertex(
        from,
        screen_offset,
        delta,
        delta,
        half_width,
        color,
    ));
    vertices.push(stroke_vertex(
        to,
        screen_offset,
        delta,
        delta,
        -half_width,
        color,
    ));
    vertices.push(stroke_vertex(
        from,
        screen_offset,
        delta,
        delta,
        -half_width,
        color,
    ));
    true
}

fn world_vertex(world: Vec2, screen_offset: Vec2, color: Color) -> Vertex {
    Vertex {
        world_position: [world.x, world.y],
        depth: 0.0,
        screen_offset: [screen_offset.x, screen_offset.y],
        previous_direction: [0.0; 2],
        next_direction: [0.0; 2],
        normal_distance: 0.0,
        color: color.to_array(),
    }
}

fn stroke_vertex(
    world: Vec2,
    screen_offset: Vec2,
    previous_direction: Vec2,
    next_direction: Vec2,
    normal_distance: f32,
    color: Color,
) -> Vertex {
    Vertex {
        world_position: [world.x, world.y],
        depth: 0.0,
        screen_offset: [screen_offset.x, screen_offset.y],
        previous_direction: [previous_direction.x, previous_direction.y],
        next_direction: [next_direction.x, next_direction.y],
        normal_distance,
        color: color.to_array(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;

    fn tessellate_for_test(scene: &Scene) -> (Vec<Vertex>, Vec<PreparedDrawBatch>) {
        let mut vertices = Vec::new();
        let mut draw_batches = Vec::new();
        let _ = tessellate_scene(scene, &mut vertices, &mut draw_batches);
        (vertices, draw_batches)
    }

    #[test]
    fn dynamic_mesh_vertices_require_complete_finite_triangles() {
        let vertex = DynamicVertex2d::new(Vec2::ZERO, 0.0, Color::WHITE).unwrap();
        assert!(matches!(
            dynamic_vertices_to_gpu(&[vertex, vertex]),
            Err(DynamicMeshError::InvalidVertexCount)
        ));
        assert_eq!(
            DynamicVertex2d::new(Vec2::new(f32::NAN, 0.0), 0.0, Color::WHITE),
            Err(DynamicMeshError::InvalidVertex)
        );

        let vertices = dynamic_vertices_to_gpu(&[vertex; 3]).unwrap();
        assert_eq!(vertices.len(), 3);
        assert_eq!(dynamic_vertex_capacity(3), 4);
    }

    #[test]
    fn particle_instances_validate_visual_contract() {
        let particle =
            ParticleInstance2d::new(Vec2::new(3.0, -2.0), 4.5, Color::WHITE, 1.0).unwrap();
        assert_eq!(particle.world_position(), Vec2::new(3.0, -2.0));
        assert_eq!(particle.radius(), 4.5);
        assert_eq!(particle.depth(), 1.0);
        assert_eq!(particle.color(), Color::WHITE);
        assert_eq!(
            ParticleInstance2d::new(Vec2::ZERO, 0.0, Color::WHITE, 0.0),
            Err(ParticleInstanceError::InvalidRadius)
        );
        assert_eq!(
            ParticleInstance2d::new(Vec2::ZERO, 1.0, Color::WHITE, f32::NAN),
            Err(ParticleInstanceError::NonFinite)
        );
    }

    #[test]
    fn particle_gpu_instances_preserve_counts_and_capacity_contract() {
        let particle = ParticleInstance2d::new(Vec2::new(2.0, -3.0), 2.0, Color::WHITE, 4.0)
            .expect("finite particle should be valid");
        let gpu_instances = particle_instances_to_gpu(&[particle]).unwrap();
        assert_eq!(gpu_instances.len(), 1);
        assert_eq!(particle_instance_capacity(0), 1);
        assert_eq!(particle_instance_capacity(3), 4);
        let statistics = particle_statistics(gpu_instances.len(), gpu_instances.len(), 1);
        assert_eq!(statistics.submitted(), 1);
        assert_eq!(statistics.visible(), 1);
        assert_eq!(statistics.culled(), 0);
        assert_eq!(statistics.dropped(), 0);
        assert_eq!(statistics.rendered(), 1);
    }

    #[test]
    fn particle_culling_keeps_circles_that_intersect_the_logical_viewport() {
        let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
        let viewport = LogicalViewport::new(100.0, 100.0).unwrap();
        let uniform = CameraUniform::new(camera, viewport).unwrap();
        let visible = ParticleGpu {
            world_position: [55.0, 0.0],
            depth: 0.0,
            radius: 6.0,
            color: Color::WHITE.to_array(),
        };
        let culled = ParticleGpu {
            world_position: [57.0, 0.0],
            depth: 0.0,
            radius: 6.0,
            color: Color::WHITE.to_array(),
        };
        assert!(visible.is_safe_for(uniform));
        assert!(visible.intersects_viewport(uniform, viewport));
        assert!(!culled.intersects_viewport(uniform, viewport));
    }

    #[test]
    fn particle_partial_updates_require_an_existing_contiguous_range() {
        assert_eq!(particle_update_range(2, 3, 5), Ok(2..5));
        assert_eq!(particle_update_range(5, 0, 5), Ok(5..5));
        assert_eq!(
            particle_update_range(4, 2, 5),
            Err(ParticleFieldError::UpdateRangeOutOfBounds)
        );
        assert_eq!(
            particle_update_range(usize::MAX, 1, 5),
            Err(ParticleFieldError::UpdateRangeOutOfBounds)
        );
    }

    fn vertex_screen_position(vertex: Vertex, camera: Camera2d, viewport: LogicalViewport) -> Vec2 {
        let Some(uniform) = CameraUniform::new(camera, viewport) else {
            panic!("test camera uniform should be finite");
        };
        let world = Vec2::new(vertex.world_position[0], vertex.world_position[1]);
        let mut screen = uniform.world_to_screen(world, vertex.depth)
            + Vec2::new(vertex.screen_offset[0], vertex.screen_offset[1]);
        if vertex.normal_distance.abs() > 0.0 {
            let previous = uniform.direction_to_screen(Vec2::new(
                vertex.previous_direction[0],
                vertex.previous_direction[1],
            ));
            let next = uniform.direction_to_screen(Vec2::new(
                vertex.next_direction[0],
                vertex.next_direction[1],
            ));
            let previous_normal = previous.normalized().perp();
            let next_normal = next.normalized().perp();
            let combined_normal = previous_normal + next_normal;
            let mut extrusion = next_normal * vertex.normal_distance;
            if combined_normal.length_squared() > 0.000001 {
                let miter = combined_normal.normalized();
                let denominator = miter.dot(next_normal);
                if denominator.abs() > 0.001 {
                    extrusion = miter * (vertex.normal_distance / denominator);
                }
            }
            screen += extrusion;
        }
        screen
    }

    #[test]
    fn filled_circle_tessellates_to_triangle_fan() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.circle(Vec2::ZERO, 12.0, ShapeStyle::filled(Color::WHITE));

        let (vertices, draw_batches) = tessellate_for_test(&scene);

        assert_eq!(vertices.len(), CIRCLE_SEGMENTS * 3);
        assert_eq!(draw_batches.len(), 1);
    }

    #[test]
    fn line_tessellates_with_round_caps() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.line(
            Vec2::new(-10.0, 0.0),
            Vec2::new(10.0, 0.0),
            2.0,
            Color::WHITE,
        );

        let (vertices, _) = tessellate_for_test(&scene);

        assert_eq!(vertices.len(), 6 + CIRCLE_SEGMENTS * 6);
    }

    #[test]
    fn short_accepted_line_emits_vertices() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        assert!(scene.line(Vec2::ZERO, Vec2::new(0.005, 0.0), 2.0, Color::WHITE));

        let (vertices, _) = tessellate_for_test(&scene);

        assert!(!vertices.is_empty());
    }

    #[test]
    fn overflowing_finite_geometry_is_reported_as_dropped() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        assert!(scene.circle(
            Vec2::splat(f32::MAX),
            f32::MAX,
            ShapeStyle::filled(Color::WHITE),
        ));
        let mut vertices = Vec::new();
        let mut draw_batches = Vec::new();
        let stats = tessellate_scene(&scene, &mut vertices, &mut draw_batches);

        assert_eq!(stats.command_count(), 1);
        assert_eq!(stats.rendered_command_count(), 0);
        assert_eq!(stats.dropped_command_count(), 1);
        assert!(vertices.is_empty());
    }

    #[test]
    fn invalid_primitives_do_not_emit_vertices() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.circle(Vec2::ZERO, 0.0, ShapeStyle::filled(Color::WHITE));
        scene.line(Vec2::ZERO, Vec2::ZERO, 2.0, Color::WHITE);
        scene.line(Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), 0.0, Color::WHITE);

        let (vertices, draw_batches) = tessellate_for_test(&scene);

        assert!(vertices.is_empty());
        assert!(draw_batches.is_empty());
    }

    #[test]
    fn gradient_fill_reaches_vertex_colors() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.rect(
            Rect::from_center_size(Vec2::ZERO, Vec2::new(2.0, 2.0)),
            0.0,
            ShapeStyle::filled_with(Fill::LinearGradient(crate::LinearGradient::new(
                Vec2::new(-1.0, 0.0),
                Vec2::new(1.0, 0.0),
                Color::BLACK,
                Color::WHITE,
            ))),
        );

        let (vertices, _) = tessellate_for_test(&scene);

        assert!(vertices.iter().any(|vertex| vertex.color[0] < 0.01));
        assert!(vertices.iter().any(|vertex| vertex.color[0] > 0.99));
    }

    #[test]
    fn flat_rectangle_emits_every_fan_sector() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.rect(
            Rect::from_center_size(Vec2::ZERO, Vec2::new(20.0, 10.0)),
            0.0,
            ShapeStyle::filled(Color::WHITE),
        );

        let (vertices, _) = tessellate_for_test(&scene);

        assert_eq!(vertices.len(), 12);
    }

    #[test]
    fn projected_circle_follows_camera_tilt() {
        let Ok(mut camera) = Camera2d::new(Vec2::ZERO, 2.0) else {
            panic!("test camera should be valid");
        };
        let Ok(projection) = crate::Projection2d::new(0.8, 1.0) else {
            panic!("test projection should be valid");
        };
        camera.set_projection(projection);
        let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
            panic!("test viewport should be valid");
        };
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));
        let mut vertices = Vec::new();
        let mut draw_batches = Vec::new();

        tessellate_scene(&scene, &mut vertices, &mut draw_batches);

        let positions: Vec<_> = vertices
            .iter()
            .copied()
            .map(|vertex| vertex_screen_position(vertex, camera, viewport))
            .collect();
        let width = positions
            .iter()
            .map(|point| point.x)
            .fold(f32::NEG_INFINITY, f32::max)
            - positions
                .iter()
                .map(|point| point.x)
                .fold(f32::INFINITY, f32::min);
        let height = positions
            .iter()
            .map(|point| point.y)
            .fold(f32::NEG_INFINITY, f32::max)
            - positions
                .iter()
                .map(|point| point.y)
                .fold(f32::INFINITY, f32::min);

        assert!((width - 40.0).abs() < 0.01);
        assert!((height - 40.0 * 0.8_f32.cos()).abs() < 0.05);
    }

    #[test]
    fn camera_uniform_matches_public_camera_projection() {
        let Ok(mut camera) = Camera2d::new(Vec2::new(17.0, -23.0), 2.75) else {
            panic!("test camera should be valid");
        };
        let Ok(projection) = crate::Projection2d::new(0.63, 4.0) else {
            panic!("test projection should be valid");
        };
        camera.set_projection(projection);
        if camera.set_rotation(0.31).is_err() {
            panic!("test rotation should be valid");
        }
        let Ok(viewport) = LogicalViewport::new(1_137.0, 683.0) else {
            panic!("test viewport should be valid");
        };
        let Some(uniform) = CameraUniform::new(camera, viewport) else {
            panic!("test camera uniform should be finite");
        };

        for (world, depth) in [
            (Vec2::ZERO, 0.0),
            (camera.center(), 0.0),
            (Vec2::new(-81.5, 44.25), 7.5),
            (Vec2::new(319.0, -127.0), -13.25),
        ] {
            let expected = camera
                .projected_world_to_screen(world, depth, viewport)
                .unwrap()
                .to_vec2();
            let actual = uniform.world_to_screen(world, depth);
            assert!((actual.x - expected.x).abs() < 0.001);
            assert!((actual.y - expected.y).abs() < 0.001);
        }
    }

    #[test]
    fn scene_depth_reaches_tessellated_vertices() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        assert!(
            scene
                .with_depth(12.5, |scene| {
                    scene.circle(Vec2::ZERO, 8.0, ShapeStyle::filled(Color::WHITE));
                })
                .is_ok()
        );

        let (vertices, _) = tessellate_for_test(&scene);

        assert!(!vertices.is_empty());
        assert!(vertices.iter().all(|vertex| vertex.depth == 12.5));
    }

    #[test]
    fn maximum_radius_rounded_rect_stroke_has_no_collapsed_directions() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.rect(
            Rect::from_center_size(Vec2::ZERO, Vec2::splat(20.0)),
            100.0,
            ShapeStyle::stroked(3.0, Color::WHITE),
        );

        let (vertices, _) = tessellate_for_test(&scene);

        assert!(!vertices.is_empty());
        assert!(vertices.iter().all(|vertex| {
            Vec2::new(vertex.previous_direction[0], vertex.previous_direction[1]).length_squared()
                > f32::EPSILON
                && Vec2::new(vertex.next_direction[0], vertex.next_direction[1]).length_squared()
                    > f32::EPSILON
        }));
    }

    #[test]
    fn gpu_extrusion_contract_keeps_line_width_in_screen_pixels() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.line(
            Vec2::new(-20.0, -13.0),
            Vec2::new(40.0, 27.0),
            7.0,
            Color::WHITE,
        );
        let (vertices, _) = tessellate_for_test(&scene);
        let Ok(mut camera) = Camera2d::new(Vec2::new(5.0, 9.0), 6.0) else {
            panic!("test camera should be valid");
        };
        let Ok(projection) = crate::Projection2d::new(0.72, 1.0) else {
            panic!("test projection should be valid");
        };
        camera.set_projection(projection);
        if camera.set_rotation(-0.41).is_err() {
            panic!("test rotation should be valid");
        }
        let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
            panic!("test viewport should be valid");
        };

        let positive = vertex_screen_position(vertices[0], camera, viewport);
        let negative = vertex_screen_position(vertices[5], camera, viewport);

        assert!(((positive - negative).length() - 7.0).abs() < 0.001);
    }

    #[test]
    fn geometry_extents_reject_shader_arithmetic_overflow() {
        let vertices = [world_vertex(
            Vec2::new(f32::MAX * 0.75, 0.0),
            Vec2::ZERO,
            Color::WHITE,
        )];
        let extents = GeometryExtents::from_vertices(&vertices);
        let Ok(camera) = Camera2d::new(Vec2::ZERO, 2.0) else {
            panic!("test camera should be valid");
        };
        let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
            panic!("test viewport should be valid");
        };
        let Some(uniform) = CameraUniform::new(camera, viewport) else {
            panic!("test camera uniform should be finite");
        };

        assert!(!extents.is_safe_for(uniform));
    }

    #[test]
    fn geometry_extents_accept_valid_geometry_relative_to_large_camera_center() {
        let center = Vec2::new(2.0e38, 0.0);
        let world = Vec2::new(center.x + 1.0e33, 0.0);
        let vertices = [world_vertex(world, Vec2::ZERO, Color::WHITE)];
        let extents = GeometryExtents::from_vertices(&vertices);
        let Ok(camera) = Camera2d::new(center, 2.0) else {
            panic!("large finite camera should be valid");
        };
        let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
            panic!("test viewport should be valid");
        };
        let Some(uniform) = CameraUniform::new(camera, viewport) else {
            panic!("relative camera uniform should remain finite");
        };

        assert!(uniform.world_to_screen(world, 0.0).is_finite());
        assert!(extents.is_safe_for(uniform));
    }

    #[test]
    fn non_finite_and_overflowing_geometry_never_reaches_batches() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        assert!(!scene.circle(
            Vec2::new(f32::NAN, 0.0),
            10.0,
            ShapeStyle::filled(Color::WHITE)
        ));
        assert!(scene.circle(
            Vec2::new(f32::MAX, f32::MAX),
            f32::MAX,
            ShapeStyle::filled(Color::WHITE)
        ));

        let (vertices, draw_batches) = tessellate_for_test(&scene);

        assert!(vertices.is_empty());
        assert!(draw_batches.is_empty());
    }

    #[test]
    fn clipped_commands_create_scissor_batches() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        let clip =
            ScreenClipRect::from_min_size(Vec2::new(10.25, 20.75), Vec2::new(100.0, 80.0)).unwrap();
        scene
            .with_screen_clip(clip, |scene| {
                scene.circle(Vec2::ZERO, 8.0, ShapeStyle::filled(Color::WHITE));
                scene.circle(Vec2::X, 8.0, ShapeStyle::filled(Color::WHITE));
            })
            .unwrap();
        scene.circle(Vec2::Y, 8.0, ShapeStyle::filled(Color::WHITE));

        let (_, draw_batches) = tessellate_for_test(&scene);

        assert_eq!(draw_batches.len(), 2);
        assert_eq!(draw_batches[0].screen_clip, Some(clip));
        assert_eq!(draw_batches[1].screen_clip, None);
    }

    #[test]
    fn offscreen_clip_keeps_prepared_geometry_but_resolves_to_no_scissor() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        let screen_clip =
            ScreenClipRect::from_min_size(Vec2::new(900.0, 700.0), Vec2::new(20.0, 20.0)).unwrap();
        scene
            .with_screen_clip(screen_clip, |scene| {
                scene.circle(Vec2::ZERO, 8.0, ShapeStyle::filled(Color::WHITE))
            })
            .unwrap();

        let (vertices, draw_batches) = tessellate_for_test(&scene);
        let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
            panic!("test viewport should be valid");
        };

        assert!(!vertices.is_empty());
        assert_eq!(draw_batches[0].screen_clip, Some(screen_clip));
        assert_eq!(screen_clip_to_scissor(screen_clip, viewport, 1.0), None);
    }

    #[test]
    fn non_finite_screen_clip_is_rejected_immediately() {
        let scene = Scene::new(Color::BLACK).unwrap();
        assert_eq!(
            ScreenClipRect::new(Vec2::new(f32::NAN, 0.0), Vec2::new(100.0, 100.0)),
            Err(crate::SceneError::InvalidScreenClip)
        );
        assert_eq!(scene.command_count(), 0);
    }

    #[test]
    fn logical_clip_converts_to_hidpi_physical_scissor() {
        let Ok(viewport) = LogicalViewport::new(800.0, 600.0) else {
            panic!("test viewport should be valid");
        };
        let clip =
            ScreenClipRect::from_min_size(Vec2::new(10.25, 20.75), Vec2::new(100.0, 80.0)).unwrap();

        let scissor = screen_clip_to_scissor(clip, viewport, 2.0);

        assert_eq!(
            scissor,
            Some(ScissorRect {
                x: 20,
                y: 41,
                width: 201,
                height: 161,
            })
        );
    }

    #[test]
    fn renderer_screen_position_conversion_is_explicit_at_hidpi() {
        let physical = PhysicalScreenPosition::new(800.0, 600.0);

        let logical = physical_to_logical_screen(physical, 2.0).unwrap();
        let roundtrip = logical_to_physical_screen(logical, 2.0).unwrap();

        assert_eq!(logical, LogicalScreenPosition::new(400.0, 300.0));
        assert_eq!(roundtrip, physical);
        assert_eq!(
            physical_to_logical_screen(PhysicalScreenPosition::new(f32::MAX, 0.0), 0.5),
            Err(RendererCoordinateError::NonFiniteConversion)
        );
    }

    #[test]
    fn prepared_scene_identity_guard_rejects_another_renderer() {
        let first_renderer = Arc::new(());
        let same_renderer = Arc::clone(&first_renderer);
        let second_renderer = Arc::new(());

        assert!(prepared_scene_belongs_to(&first_renderer, &same_renderer));
        assert!(!prepared_scene_belongs_to(
            &first_renderer,
            &second_renderer
        ));
    }

    #[test]
    fn offscreen_gpu_readback_verifies_camera_depth_and_clip_contract() {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                })
                .await
            else {
                return;
            };
            let Ok((device, queue)) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("sim-engine offscreen test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                })
                .await
            else {
                panic!("adapter should create a test device");
            };

            let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let format = wgpu::TextureFormat::Rgba8UnormSrgb;
            let (
                pipeline,
                particle_pipeline,
                heatmap_pipeline,
                camera_uniform_buffer,
                camera_bind_group,
                heatmap_uniform_buffer,
                heatmap_bind_group_layout,
            ) = create_pipeline(&device, format, 1);
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("sim-engine offscreen test target"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let mut scene = Scene::new(Color::BLACK).unwrap();
            let clip =
                ScreenClipRect::from_min_size(Vec2::new(30.0, 20.0), Vec2::new(8.0, 12.0)).unwrap();
            assert!(
                scene
                    .with_depth(3.0, |scene| {
                        scene
                            .with_screen_clip(clip, |scene| {
                                scene.rect(
                                    Rect::from_center_size(Vec2::ZERO, Vec2::splat(4.0)),
                                    0.0,
                                    ShapeStyle::filled(Color::WHITE),
                                );
                            })
                            .unwrap();
                    })
                    .is_ok()
            );
            let source_identity = Arc::new(());
            let prepared =
                prepare_scene_resources(&device, &queue, Arc::clone(&source_identity), &scene);
            let replacement_identity = Arc::new(());
            let restored = restore_prepared_scene_resources(
                &device,
                &queue,
                Arc::clone(&replacement_identity),
                &prepared,
            );
            assert!(prepared_scene_belongs_to(
                &source_identity,
                &prepared.renderer_identity
            ));
            assert!(!prepared_scene_belongs_to(
                &source_identity,
                &restored.renderer_identity
            ));
            assert!(prepared_scene_belongs_to(
                &replacement_identity,
                &restored.renderer_identity
            ));
            assert_eq!(restored.vertex_count(), prepared.vertex_count());
            assert_eq!(
                restored.recovery_memory_bytes(),
                restored.vertex_count() * std::mem::size_of::<Vertex>()
            );
            assert!(Arc::ptr_eq(&restored.vertices, &prepared.vertices));

            let dynamic_vertex = DynamicVertex2d::new(Vec2::ZERO, 0.0, Color::WHITE).unwrap();
            let dynamic_vertices = dynamic_vertices_to_gpu(&[dynamic_vertex; 3]).unwrap();
            let dynamic_source = DynamicMesh2d {
                renderer_identity: Arc::clone(&source_identity),
                vertex_buffer: Arc::new(create_vertex_buffer(&device, 8)),
                geometry_extents: GeometryExtents::from_vertices(&dynamic_vertices),
                vertices: dynamic_vertices,
                vertex_capacity: 8,
            };
            queue.write_buffer(
                &dynamic_source.vertex_buffer,
                0,
                bytemuck::cast_slice(dynamic_source.vertices.as_slice()),
            );
            let restored_dynamic = restore_dynamic_mesh_resources(
                &device,
                &queue,
                Arc::clone(&replacement_identity),
                &dynamic_source,
            );
            assert_eq!(restored_dynamic.vertex_count(), 3);
            assert_eq!(restored_dynamic.vertex_capacity(), 8);
            assert_eq!(
                restored_dynamic.recovery_memory_bytes(),
                dynamic_source.recovery_memory_bytes()
            );
            assert!(prepared_scene_belongs_to(
                &replacement_identity,
                &restored_dynamic.renderer_identity
            ));

            let scalar_source = ScalarField::new(2, 2, vec![0.0, 0.25, 0.5, 1.0]).unwrap();
            let scalar_texture = create_scalar_field_texture_resources(
                &device,
                &queue,
                Arc::clone(&source_identity),
                scalar_source,
            )
            .expect("finite scalar texture should upload");
            assert_eq!((scalar_texture.width(), scalar_texture.height()), (2, 2));
            assert_eq!(scalar_texture.recovery_memory_bytes(), 16);
            let restored_scalar_texture = create_scalar_field_texture_resources(
                &device,
                &queue,
                Arc::clone(&replacement_identity),
                scalar_texture.field.clone(),
            )
            .expect("scalar texture should restore");
            assert_eq!(
                restored_scalar_texture.field().values(),
                scalar_texture.field().values()
            );
            let heatmap_color_map = ColorMap::linear(Color::BLACK, Color::WHITE).unwrap();
            let heatmap_lut = create_color_map_texture(&device, &queue, &heatmap_color_map);
            let heatmap_scalar_view = scalar_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let heatmap_lut_view = heatmap_lut.create_view(&wgpu::TextureViewDescriptor::default());
            let heatmap_uniform = HeatmapUniform {
                value_range: [0.0, 1.0, 0.0, 0.0],
                dimensions: [2, 2, 0, 0],
            };
            queue.write_buffer(
                &heatmap_uniform_buffer,
                0,
                bytemuck::bytes_of(&heatmap_uniform),
            );
            let heatmap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sim-engine offscreen heatmap bind group"),
                layout: &heatmap_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&heatmap_scalar_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&heatmap_lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: heatmap_uniform_buffer.as_entire_binding(),
                    },
                ],
            });
            let heatmap_target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("sim-engine offscreen heatmap target"),
                size: wgpu::Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let heatmap_view = heatmap_target.create_view(&wgpu::TextureViewDescriptor::default());
            let heatmap_readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sim-engine offscreen heatmap readback"),
                size: 512,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let Ok(mut camera) = Camera2d::new(Vec2::ZERO, 1.0) else {
                panic!("test camera should be valid");
            };
            let Ok(projection) = crate::Projection2d::new(0.5, 2.0) else {
                panic!("test projection should be valid");
            };
            camera.set_projection(projection);
            let Ok(viewport) = LogicalViewport::new(64.0, 64.0) else {
                panic!("test viewport should be valid");
            };
            let Some(camera_uniform) = CameraUniform::new(camera, viewport) else {
                panic!("test camera uniform should be finite");
            };
            queue.write_buffer(
                &camera_uniform_buffer,
                0,
                bytemuck::bytes_of(&camera_uniform),
            );
            let particle_unit_buffer = create_particle_unit_buffer(&device, &queue);
            let particle_instances = [
                ParticleGpu {
                    world_position: [18.0, -18.0],
                    depth: 0.0,
                    radius: 6.0,
                    color: Color::rgba(1.0, 0.0, 0.0, 1.0).to_array(),
                },
                ParticleGpu {
                    world_position: [1_000.0, 0.0],
                    depth: 0.0,
                    radius: 6.0,
                    color: Color::WHITE.to_array(),
                },
            ];
            let visible_particles =
                visible_particle_instances(&particle_instances, camera_uniform, viewport).unwrap();
            assert_eq!(
                visible_particles.len(),
                1,
                "offscreen particle should be culled"
            );
            let particle_instance_buffer = create_particle_instance_buffer(&device, 1);
            queue.write_buffer(
                &particle_instance_buffer,
                0,
                bytemuck::cast_slice(&visible_particles),
            );
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sim-engine offscreen readback"),
                size: (256 * 64) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine offscreen test encoder"),
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sim-engine offscreen heatmap pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &heatmap_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&heatmap_pipeline);
                pass.set_bind_group(0, &heatmap_bind_group, &[]);
                pass.draw(0..6, 0..1);
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sim-engine offscreen test pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &camera_bind_group, &[]);
                pass.set_vertex_buffer(0, restored.vertex_buffer.slice(..));
                for batch in &restored.draw_batches {
                    if let Some(clip) = batch.screen_clip {
                        let scissor = screen_clip_to_scissor(clip, viewport, 1.0).unwrap();
                        pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                    }
                    pass.draw(batch.vertex_range.clone(), 0..1);
                }
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sim-engine offscreen particle pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&particle_pipeline);
                pass.set_bind_group(0, &camera_bind_group, &[]);
                pass.set_vertex_buffer(0, particle_unit_buffer.slice(..));
                pass.set_vertex_buffer(1, particle_instance_buffer.slice(..));
                pass.draw(0..6, 0..1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(64),
                    },
                },
                wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
            );
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &heatmap_target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &heatmap_readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(2),
                    },
                },
                wgpu::Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            );

            let _submission = queue.submit([encoder.finish()]);
            if let Err(error) = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            }) {
                panic!("offscreen GPU submission did not complete: {error:?}");
            }
            if let Some(error) = validation_scope.pop().await {
                panic!("offscreen GPU validation failed: {error}");
            }
            let slice = readback.slice(..);
            let (sender, receiver) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap()
            });
            if let Err(error) = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            }) {
                panic!("offscreen GPU readback did not complete: {error:?}");
            }
            receiver
                .recv_timeout(Duration::from_secs(5))
                .expect("offscreen readback callback")
                .expect("offscreen readback should map");
            let bytes = slice.get_mapped_range().expect("offscreen mapped bytes");
            let pixel = |x: usize, y: usize| &bytes[y * 256 + x * 4..y * 256 + x * 4 + 4];
            assert!(pixel(33, 29)[0] > 200, "camera/depth pixel was not drawn");
            assert!(pixel(33, 34)[0] < 10, "clip failed to remove outside pixel");
            assert!(
                pixel(50, 48)[0] > 200,
                "instanced particle center was not drawn"
            );
            assert!(
                pixel(50, 48)[1] < 10,
                "instanced particle color was not applied"
            );
            assert!(
                pixel(56, 54)[0] < 10,
                "particle circle mask did not discard its corner"
            );
            drop(bytes);
            readback.unmap();
            let heatmap_slice = heatmap_readback.slice(..);
            let (heatmap_sender, heatmap_receiver) = mpsc::channel();
            heatmap_slice.map_async(wgpu::MapMode::Read, move |result| {
                heatmap_sender.send(result).unwrap()
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(5)),
                })
                .expect("offscreen heatmap readback should complete");
            heatmap_receiver
                .recv_timeout(Duration::from_secs(5))
                .expect("offscreen heatmap callback")
                .expect("offscreen heatmap should map");
            let heatmap_bytes = heatmap_slice
                .get_mapped_range()
                .expect("offscreen heatmap bytes");
            let heatmap_pixel =
                |x: usize, y: usize| &heatmap_bytes[y * 256 + x * 4..y * 256 + x * 4 + 4];
            assert!(
                heatmap_pixel(0, 0)[0] < 8,
                "minimum scalar should map to black"
            );
            assert!(
                heatmap_pixel(1, 1)[0] > 247,
                "maximum scalar should map to white"
            );
            assert!(
                heatmap_pixel(1, 0)[0] > 130 && heatmap_pixel(1, 0)[0] < 145,
                "quarter scalar should map through LUT then sRGB encode"
            );
            assert!(
                heatmap_pixel(0, 1)[0] > 180 && heatmap_pixel(0, 1)[0] < 195,
                "half scalar should map through LUT then sRGB encode"
            );
            drop(heatmap_bytes);
            heatmap_readback.unmap();
        });
    }

    #[test]
    fn renderer_options_reject_invalid_scale_factor() {
        assert!(matches!(
            WgpuRendererOptions::new(RendererPresentMode::Vsync, f64::NAN),
            Err(RendererConfigurationError::InvalidScaleFactor { .. })
        ));
        assert!(matches!(
            WgpuRendererOptions::new(RendererPresentMode::Vsync, 0.0),
            Err(RendererConfigurationError::InvalidScaleFactor { .. })
        ));
        assert!(matches!(
            WgpuRendererOptions::new(RendererPresentMode::Vsync, f64::MIN_POSITIVE),
            Err(RendererConfigurationError::InvalidScaleFactor { .. })
        ));
        assert!(matches!(
            WgpuRendererOptions::new(RendererPresentMode::Vsync, f32::MIN_POSITIVE as f64),
            Err(RendererConfigurationError::InvalidScaleFactor { .. })
        ));
    }

    #[test]
    fn renderer_present_modes_have_explicit_fallback_contracts() {
        assert_eq!(
            RendererPresentMode::Vsync.to_wgpu(),
            wgpu::PresentMode::Fifo
        );
        assert_eq!(
            RendererPresentMode::NoVsync.to_wgpu(),
            wgpu::PresentMode::AutoNoVsync
        );
    }
}
