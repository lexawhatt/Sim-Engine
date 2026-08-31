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
    LogicalViewport, LogicalViewportRegion, ParticleInstance2d, PhysicalPerLogical,
    PhysicalScreenPosition, Polyline, PrimitiveCommandCounts, Rect, ScalarField, Scene,
    SceneBudgetResource, ScreenClipRect, ScreenScene, Shadow, ShapeStyle, Stroke, Vec2,
    scene::{CIRCLE_SEGMENTS, CORNER_SEGMENTS, ROUND_CAP_SEGMENTS, TESSELLATED_VERTEX_BYTES},
    screen::screen_camera,
};
use config::select_surface_present_mode;

const INITIAL_VERTEX_CAPACITY: usize = 4096;
const PREFERRED_SAMPLE_COUNT: u32 = 4;
const COLOR_MAP_LUT_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    world_position: [f32; 2],
    depth: f32,
    world_offset: [f32; 2],
    screen_offset: [f32; 2],
    previous_direction: [f32; 2],
    next_direction: [f32; 2],
    normal_distance: f32,
    tangent_distance: f32,
    miter_limit: f32,
    stroke_role: f32,
    stroke_parameter: f32,
    color: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<Vertex>() == TESSELLATED_VERTEX_BYTES);

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct DynamicGpu {
    world_position: [f32; 2],
    depth: f32,
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
    destination: [f32; 4],
}

impl HeatmapUniform {
    fn new(
        minimum: f32,
        value_extent: f32,
        width: usize,
        height: usize,
        sampling: ScalarFieldSampling,
    ) -> Self {
        Self {
            value_range: [minimum, value_extent, 0.0, 0.0],
            dimensions: [width as u32, height as u32, sampling.shader_value(), 0],
            destination: [1.0, 1.0, 0.0, 0.0],
        }
    }

    fn in_region(mut self, region: LogicalViewportRegion, target: LogicalViewport) -> Option<Self> {
        let destination = CompositeUniform::in_region(1.0, region, target)?.destination;
        self.destination = destination;
        Some(self)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniform {
    opacity: [f32; 4],
    destination: [f32; 4],
}

impl CompositeUniform {
    const fn full_surface(opacity: f32) -> Self {
        Self {
            opacity: [opacity, 0.0, 0.0, 0.0],
            destination: [1.0, 1.0, 0.0, 0.0],
        }
    }

    fn in_region(
        opacity: f32,
        region: LogicalViewportRegion,
        target: LogicalViewport,
    ) -> Option<Self> {
        let origin = region.origin().to_vec2();
        let viewport = region.viewport();
        let scale_x = viewport.width() / target.width();
        let scale_y = viewport.height() / target.height();
        let offset_x = (origin.x.mul_add(2.0, viewport.width())) / target.width() - 1.0;
        let offset_y = 1.0 - (origin.y.mul_add(2.0, viewport.height())) / target.height();
        [scale_x, scale_y, offset_x, offset_y]
            .into_iter()
            .all(f32::is_finite)
            .then_some(Self {
                opacity: [opacity, 0.0, 0.0, 0.0],
                destination: [scale_x, scale_y, offset_x, offset_y],
            })
    }
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

/// How a render target is combined with the presentation surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Standard source-alpha compositing.
    #[default]
    Alpha,
    /// Adds source light to the existing destination, useful for glow and trails.
    Additive,
    /// Replaces destination pixels with the source target.
    Replace,
}

/// How a target-rendering pass initializes its destination pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderTargetLoad {
    /// Preserve existing target pixels and blend new geometry over them.
    Load,
    /// Clear the target to a finite straight-alpha color before drawing.
    Clear(Color),
}

#[derive(Debug, Clone, Copy)]
struct GeometryExtents {
    world_min: Vec2,
    world_max: Vec2,
    world_offset_min: Vec2,
    world_offset_max: Vec2,
    depth_min: f32,
    depth_max: f32,
    direction_min: Vec2,
    direction_max: Vec2,
    screen_offset_max_abs: Vec2,
    normal_distance_max_abs: f32,
    tangent_distance_max_abs: f32,
    miter_limit_max: f32,
    empty: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScissorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, Copy)]
struct ParticleDrawPreparation {
    visible_count: usize,
    upload: Duration,
    camera_uniform_upload: Duration,
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
    command_counts: PrimitiveCommandCounts,
    rendered_counts: PrimitiveCommandCounts,
    dropped_counts: PrimitiveCommandCounts,
    vertex_count: usize,
    draw_batch_count: usize,
    upload_bytes: usize,
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

    /// Returns accepted commands examined, grouped by primitive category.
    pub const fn command_counts(self) -> PrimitiveCommandCounts {
        self.command_counts
    }

    /// Returns commands that emitted geometry, grouped by primitive category.
    pub const fn rendered_counts(self) -> PrimitiveCommandCounts {
        self.rendered_counts
    }

    /// Returns commands that emitted no geometry, grouped by primitive category.
    pub const fn dropped_counts(self) -> PrimitiveCommandCounts {
        self.dropped_counts
    }

    /// Returns finite triangle-list vertices generated for this scene.
    pub fn vertex_count(self) -> usize {
        self.vertex_count
    }

    /// Returns draw batches generated after compatible adjacent commands merge.
    pub fn draw_batch_count(self) -> usize {
        self.draw_batch_count
    }

    /// Returns bytes required to upload the generated vertices.
    pub fn upload_bytes(self) -> usize {
        self.upload_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TessellationError {
    AllocationFailed {
        requested_bytes: usize,
    },
    CapacityTooLarge,
    BudgetExceeded {
        resource: SceneBudgetResource,
        limit: usize,
        actual: usize,
    },
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 12] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32,
        2 => Float32x2,
        3 => Float32x2,
        4 => Float32x2,
        5 => Float32x2,
        6 => Float32,
        7 => Float32,
        8 => Float32,
        9 => Float32,
        10 => Float32,
        11 => Float32x4
    ];

    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };

    fn is_finite(self) -> bool {
        self.world_position.iter().all(|value| value.is_finite())
            && self.depth.is_finite()
            && self.world_offset.iter().all(|value| value.is_finite())
            && self.screen_offset.iter().all(|value| value.is_finite())
            && self
                .previous_direction
                .iter()
                .all(|value| value.is_finite())
            && self.next_direction.iter().all(|value| value.is_finite())
            && self.normal_distance.is_finite()
            && self.tangent_distance.is_finite()
            && self.miter_limit.is_finite()
            && self.stroke_role.is_finite()
            && self.stroke_parameter.is_finite()
            && self.color.iter().all(|value| value.is_finite())
    }
}

impl DynamicGpu {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32, 2 => Float32x4];

    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };
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
        Self::new_in_region(camera, viewport, Vec2::ZERO, viewport)
    }

    fn new_in_region(
        camera: Camera2d,
        viewport: LogicalViewport,
        target_origin: Vec2,
        target_viewport: LogicalViewport,
    ) -> Option<Self> {
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
                target_origin.x + viewport.width() * 0.5,
            ],
            world_to_screen_y: [
                vertical_x,
                vertical_y,
                vertical_depth,
                target_origin.y + viewport.height() * 0.5,
            ],
            screen_to_clip: [
                2.0 / target_viewport.width(),
                -2.0 / target_viewport.height(),
                -1.0,
                1.0,
            ],
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
    fn empty(is_empty: bool) -> Self {
        Self {
            world_min: Vec2::splat(f32::INFINITY),
            world_max: Vec2::splat(f32::NEG_INFINITY),
            world_offset_min: Vec2::splat(f32::INFINITY),
            world_offset_max: Vec2::splat(f32::NEG_INFINITY),
            depth_min: f32::INFINITY,
            depth_max: f32::NEG_INFINITY,
            direction_min: Vec2::splat(f32::INFINITY),
            direction_max: Vec2::splat(f32::NEG_INFINITY),
            screen_offset_max_abs: Vec2::ZERO,
            normal_distance_max_abs: 0.0,
            tangent_distance_max_abs: 0.0,
            miter_limit_max: 1.0,
            empty: is_empty,
        }
    }

    fn include(&mut self, vertex: Vertex) {
        let world = Vec2::new(vertex.world_position[0], vertex.world_position[1]);
        self.world_min.x = self.world_min.x.min(world.x);
        self.world_min.y = self.world_min.y.min(world.y);
        self.world_max.x = self.world_max.x.max(world.x);
        self.world_max.y = self.world_max.y.max(world.y);
        let world_offset = Vec2::new(vertex.world_offset[0], vertex.world_offset[1]);
        self.world_offset_min.x = self.world_offset_min.x.min(world_offset.x);
        self.world_offset_min.y = self.world_offset_min.y.min(world_offset.y);
        self.world_offset_max.x = self.world_offset_max.x.max(world_offset.x);
        self.world_offset_max.y = self.world_offset_max.y.max(world_offset.y);
        self.depth_min = self.depth_min.min(vertex.depth);
        self.depth_max = self.depth_max.max(vertex.depth);

        for direction in [vertex.previous_direction, vertex.next_direction] {
            self.direction_min.x = self.direction_min.x.min(direction[0]);
            self.direction_min.y = self.direction_min.y.min(direction[1]);
            self.direction_max.x = self.direction_max.x.max(direction[0]);
            self.direction_max.y = self.direction_max.y.max(direction[1]);
        }

        self.screen_offset_max_abs.x = self
            .screen_offset_max_abs
            .x
            .max(vertex.screen_offset[0].abs());
        self.screen_offset_max_abs.y = self
            .screen_offset_max_abs
            .y
            .max(vertex.screen_offset[1].abs());
        self.normal_distance_max_abs = self
            .normal_distance_max_abs
            .max(vertex.normal_distance.abs());
        self.tangent_distance_max_abs = self
            .tangent_distance_max_abs
            .max(vertex.tangent_distance.abs());
        self.miter_limit_max = self.miter_limit_max.max(vertex.miter_limit);
    }

    fn include_dynamic(&mut self, vertex: DynamicGpu) {
        let world = Vec2::new(vertex.world_position[0], vertex.world_position[1]);
        self.world_min.x = self.world_min.x.min(world.x);
        self.world_min.y = self.world_min.y.min(world.y);
        self.world_max.x = self.world_max.x.max(world.x);
        self.world_max.y = self.world_max.y.max(world.y);
        self.world_offset_min = Vec2::ZERO;
        self.world_offset_max = Vec2::ZERO;
        self.depth_min = self.depth_min.min(vertex.depth);
        self.depth_max = self.depth_max.max(vertex.depth);
        self.direction_min = Vec2::ZERO;
        self.direction_max = Vec2::ZERO;
    }

    fn from_dynamic_vertices(vertices: &[DynamicGpu]) -> Self {
        let mut extents = Self::empty(vertices.is_empty());
        for vertex in vertices {
            extents.include_dynamic(*vertex);
        }
        extents
    }

    fn from_vertices(vertices: &[Vertex]) -> Self {
        let mut extents = Self::empty(vertices.is_empty());
        for vertex in vertices {
            extents.include(*vertex);
        }

        extents
    }

    fn is_safe_for(self, uniform: CameraUniform) -> bool {
        if self.empty {
            return true;
        }

        let center = Vec2::new(uniform.camera_center[0], uniform.camera_center[1]);
        let mut world_horizontal = transformed_world_interval(
            uniform.world_to_screen_x,
            self.world_min,
            self.world_max,
            self.depth_min,
            self.depth_max,
            center,
        );
        let mut world_vertical = transformed_world_interval(
            uniform.world_to_screen_y,
            self.world_min,
            self.world_max,
            self.depth_min,
            self.depth_max,
            center,
        );
        world_horizontal = interval_add(
            world_horizontal,
            transformed_direction_interval(
                uniform.world_to_screen_x,
                self.world_offset_min,
                self.world_offset_max,
            ),
        );
        world_vertical = interval_add(
            world_vertical,
            transformed_direction_interval(
                uniform.world_to_screen_y,
                self.world_offset_min,
                self.world_offset_max,
            ),
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
        let maximum_miter = self.normal_distance_max_abs as f64 * self.miter_limit_max as f64;
        let horizontal_limit = interval_max_abs(world_horizontal)
            + self.screen_offset_max_abs.x as f64
            + maximum_miter
            + self.tangent_distance_max_abs as f64;
        let vertical_limit = interval_max_abs(world_vertical)
            + self.screen_offset_max_abs.y as f64
            + maximum_miter
            + self.tangent_distance_max_abs as f64;

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

fn interval_add(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    (left.0 + right.0, left.1 + right.1)
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
    /// A logical viewport is outside the active surface or render target.
    InvalidViewport,
    /// Tessellated geometry exceeds the active device's vertex-buffer limit.
    GeometryCapacityTooLarge,
    /// CPU storage for scene tessellation could not be reserved.
    SceneAllocationFailed {
        /// Minimum additional bytes requested by the failed reservation.
        requested_bytes: usize,
    },
    /// Actual renderer work exceeded a scene's explicit limit.
    SceneBudgetExceeded {
        /// Work category whose post-tessellation limit was exceeded.
        resource: SceneBudgetResource,
        /// Configured maximum for the category.
        limit: usize,
        /// Actual renderer work observed.
        actual: usize,
    },
}

impl fmt::Display for RendererFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Surface(status) => write!(formatter, "renderer surface failed: {status:?}"),
            Self::InvalidGeometryTransform => {
                write!(formatter, "camera and geometry overflow the GPU transform")
            }
            Self::InvalidViewport => {
                write!(
                    formatter,
                    "logical viewport lies outside the active render target"
                )
            }
            Self::GeometryCapacityTooLarge => {
                write!(formatter, "geometry exceeds the GPU vertex-buffer limit")
            }
            Self::SceneAllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} additional bytes for scene tessellation"
            ),
            Self::SceneBudgetExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "scene {resource:?} work exceeded its limit {limit} after tessellation: {actual}"
            ),
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
    /// The bounded previous-device quarantine is full.
    RecoveryLimitReached {
        /// Configured maximum number of quarantined logical devices.
        limit: usize,
    },
}

impl fmt::Display for RendererInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateSurface(error) => write!(formatter, "failed to create surface: {error}"),
            Self::RequestAdapter(error) => write!(formatter, "failed to request adapter: {error}"),
            Self::RequestDevice(error) => write!(formatter, "failed to request device: {error}"),
            Self::NoSurfaceConfig => write!(formatter, "surface has no supported default config"),
            Self::RecoveryLimitReached { limit } => write!(
                formatter,
                "device recovery limit reached with {limit} quarantined devices"
            ),
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
    /// Previous-device quarantine must remain inside the supported bounded range.
    InvalidRecoveryLimit {
        /// Rejected maximum retained-device count.
        limit: usize,
    },
}

impl fmt::Display for RendererConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScaleFactor { scale_factor } => write!(
                formatter,
                "renderer scale factor must be finite, positive, representable as f32, and keep a u32 surface finite in logical pixels, got {scale_factor}"
            ),
            Self::InvalidRecoveryLimit { limit } => write!(
                formatter,
                "renderer recovery quarantine must retain between 1 and 8 devices, got {limit}"
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

/// Failure while uploading or restoring immutable prepared-scene geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedSceneError {
    /// Tessellated geometry exceeds the active device's vertex-buffer limit.
    CapacityTooLarge,
    /// CPU storage for scene tessellation could not be reserved.
    AllocationFailed {
        /// Minimum additional bytes requested by the failed reservation.
        requested_bytes: usize,
    },
    /// Actual prepared-scene work exceeded an explicit scene limit.
    BudgetExceeded {
        /// Work category whose post-tessellation limit was exceeded.
        resource: SceneBudgetResource,
        /// Configured maximum for the category.
        limit: usize,
        /// Actual renderer work observed.
        actual: usize,
    },
}

impl fmt::Display for PreparedSceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityTooLarge => {
                write!(
                    formatter,
                    "prepared scene exceeds the GPU vertex-buffer limit"
                )
            }
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} additional bytes for prepared-scene tessellation"
            ),
            Self::BudgetExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "prepared scene {resource:?} work exceeded its limit {limit}: {actual}"
            ),
        }
    }
}

impl From<TessellationError> for RendererFrameError {
    fn from(error: TessellationError) -> Self {
        match error {
            TessellationError::AllocationFailed { requested_bytes } => {
                Self::SceneAllocationFailed { requested_bytes }
            }
            TessellationError::CapacityTooLarge => Self::GeometryCapacityTooLarge,
            TessellationError::BudgetExceeded {
                resource,
                limit,
                actual,
            } => Self::SceneBudgetExceeded {
                resource,
                limit,
                actual,
            },
        }
    }
}

impl From<TessellationError> for PreparedSceneError {
    fn from(error: TessellationError) -> Self {
        match error {
            TessellationError::AllocationFailed { requested_bytes } => {
                Self::AllocationFailed { requested_bytes }
            }
            TessellationError::CapacityTooLarge => Self::CapacityTooLarge,
            TessellationError::BudgetExceeded {
                resource,
                limit,
                actual,
            } => Self::BudgetExceeded {
                resource,
                limit,
                actual,
            },
        }
    }
}

impl Error for PreparedSceneError {}

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

/// Immutable prepared geometry whose positions are logical screen pixels.
///
/// This distinct wrapper prevents a world camera from being supplied to fixed
/// UI geometry. It is created by [`WgpuRenderer::prepare_screen_scene`].
pub struct PreparedScreenScene {
    scene: PreparedScene,
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
        if !world_position.is_finite() || !depth.is_finite() || !color.is_normalized() {
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
    /// The mesh capacity exceeds the current device's vertex-buffer limit.
    CapacityTooLarge,
    /// At least one complete triangle must fit in every configured limit.
    InvalidBudget,
    /// A bounded mesh operation exceeded a host-selected limit.
    BudgetExceeded {
        /// Work category that exceeded its ceiling.
        resource: DynamicMeshBudgetResource,
        /// Configured upper bound.
        limit: usize,
        /// Required work for the requested mesh state.
        actual: usize,
    },
    /// CPU recovery storage could not be reserved without changing the mesh.
    AllocationFailed {
        /// Bytes requested by the failed reservation.
        requested_bytes: usize,
    },
}

impl fmt::Display for DynamicMeshError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVertexCount => write!(
                formatter,
                "dynamic mesh vertex count must be divisible by three"
            ),
            Self::InvalidVertex => write!(
                formatter,
                "dynamic mesh positions/depth must be finite and colors normalized"
            ),
            Self::UpdateRangeOutOfBounds => write!(
                formatter,
                "dynamic mesh update range is outside the current mesh"
            ),
            Self::RendererMismatch => {
                write!(formatter, "dynamic mesh belongs to a different renderer")
            }
            Self::CapacityTooLarge => {
                write!(
                    formatter,
                    "dynamic mesh exceeds the GPU vertex-buffer limit"
                )
            }
            Self::InvalidBudget => write!(formatter, "dynamic mesh budget must fit one triangle"),
            Self::BudgetExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "dynamic mesh {resource:?} work {actual} exceeds limit {limit}"
            ),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for dynamic mesh recovery data"
            ),
        }
    }
}

impl Error for DynamicMeshError {}

/// Host-selected work category constrained by [`DynamicMeshBudget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicMeshBudgetResource {
    /// Triangle-list vertex count.
    Vertices,
    /// CPU bytes retained for exact recovery.
    RetainedBytes,
    /// GPU bytes uploaded by a full replacement.
    UploadBytes,
}

/// Explicit retained and upload limits for caller-provided colored triangles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicMeshBudget {
    max_vertices: usize,
    max_retained_bytes: usize,
    max_upload_bytes: usize,
}

impl DynamicMeshBudget {
    /// Creates limits large enough for at least one complete triangle.
    pub fn new(
        max_vertices: usize,
        max_retained_bytes: usize,
        max_upload_bytes: usize,
    ) -> Result<Self, DynamicMeshError> {
        let triangle_bytes = 3 * std::mem::size_of::<DynamicGpu>();
        if max_vertices < 3
            || max_retained_bytes < triangle_bytes
            || max_upload_bytes < triangle_bytes
        {
            return Err(DynamicMeshError::InvalidBudget);
        }
        Ok(Self {
            max_vertices,
            max_retained_bytes,
            max_upload_bytes,
        })
    }

    /// Returns the maximum retained triangle-list vertex count.
    pub const fn max_vertices(self) -> usize {
        self.max_vertices
    }

    /// Returns the maximum exact CPU recovery bytes.
    pub const fn max_retained_bytes(self) -> usize {
        self.max_retained_bytes
    }

    /// Returns the maximum bytes uploaded by one complete replacement.
    pub const fn max_upload_bytes(self) -> usize {
        self.max_upload_bytes
    }
}

impl Default for DynamicMeshBudget {
    fn default() -> Self {
        Self {
            max_vertices: 1_000_000,
            max_retained_bytes: 64 * 1024 * 1024,
            max_upload_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Mutable GPU triangle-list geometry for high-frequency simulation visuals.
///
/// A mesh retains its CPU vertices for validation and recreates its GPU buffer
/// only when a replacement exceeds its current capacity. Use
/// [`WgpuRenderer::update_dynamic_mesh_range`] for in-place partial updates.
pub struct DynamicMesh2d {
    renderer_identity: Arc<()>,
    vertex_buffer: Arc<wgpu::Buffer>,
    vertices: Vec<DynamicGpu>,
    vertex_capacity: usize,
    geometry_extents: GeometryExtents,
    budget: Option<DynamicMeshBudget>,
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
        self.vertices.len() * std::mem::size_of::<DynamicGpu>()
    }

    /// Returns explicit limits, or `None` for the compatibility constructor.
    pub const fn budget(&self) -> Option<DynamicMeshBudget> {
        self.budget
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

/// Counts associated with one particle-field update or draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParticleStatistics {
    submitted: usize,
    visibility_checked: usize,
    visible: usize,
    culled: usize,
    budget_limited: usize,
    dropped: usize,
    rendered: usize,
}

impl ParticleStatistics {
    /// Returns instances supplied by the host.
    pub fn submitted(self) -> usize {
        self.submitted
    }

    /// Returns instances tested against the camera viewport this frame.
    pub fn visibility_checked(self) -> usize {
        self.visibility_checked
    }

    /// Returns checked instances that intersect the camera viewport.
    pub fn visible(self) -> usize {
        self.visible
    }

    /// Returns instances outside the current camera viewport.
    pub fn culled(self) -> usize {
        self.culled
    }

    /// Returns unchecked or camera-visible instances omitted by the render budget.
    pub fn budget_limited(self) -> usize {
        self.budget_limited
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

/// Hard per-field limits that keep particle visualization from starving a host simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParticleRenderBudget {
    max_visible_instances: usize,
    max_gpu_bytes: usize,
    max_upload_bytes_per_frame: usize,
    max_visibility_checks_per_frame: usize,
}

impl ParticleRenderBudget {
    /// No application-level cap beyond active-device allocation limits.
    pub const UNBOUNDED: Self = Self {
        max_visible_instances: usize::MAX,
        max_gpu_bytes: usize::MAX,
        max_upload_bytes_per_frame: usize::MAX,
        max_visibility_checks_per_frame: usize::MAX,
    };

    /// Creates explicit visible-instance, GPU-memory, and per-frame upload caps.
    pub fn new(
        max_visible_instances: usize,
        max_gpu_bytes: usize,
        max_upload_bytes_per_frame: usize,
    ) -> Result<Self, ParticleBudgetError> {
        let minimum_bytes = std::mem::size_of::<ParticleGpu>();
        if max_visible_instances == 0
            || max_gpu_bytes < minimum_bytes
            || max_upload_bytes_per_frame < minimum_bytes
        {
            return Err(ParticleBudgetError::InvalidLimit);
        }
        Ok(Self {
            max_visible_instances,
            max_gpu_bytes,
            max_upload_bytes_per_frame,
            max_visibility_checks_per_frame: usize::MAX,
        })
    }

    /// Caps camera visibility checks while sampling candidates uniformly across
    /// the retained field. The cap must fit the effective visible-instance cap.
    pub fn with_max_visibility_checks(
        mut self,
        max_visibility_checks_per_frame: usize,
    ) -> Result<Self, ParticleBudgetError> {
        if max_visibility_checks_per_frame < self.instance_limit() {
            return Err(ParticleBudgetError::InvalidLimit);
        }
        self.max_visibility_checks_per_frame = max_visibility_checks_per_frame;
        Ok(self)
    }

    /// Returns the maximum camera-visible instances considered for drawing.
    pub const fn max_visible_instances(self) -> usize {
        self.max_visible_instances
    }

    /// Returns the maximum particle instance-buffer allocation in bytes.
    pub const fn max_gpu_bytes(self) -> usize {
        self.max_gpu_bytes
    }

    /// Returns the maximum particle bytes uploaded by one render call.
    pub const fn max_upload_bytes_per_frame(self) -> usize {
        self.max_upload_bytes_per_frame
    }

    /// Returns the maximum retained instances checked against the viewport.
    pub const fn max_visibility_checks_per_frame(self) -> usize {
        self.max_visibility_checks_per_frame
    }

    fn instance_limit(self) -> usize {
        self.max_visible_instances
            .min(self.max_gpu_bytes / std::mem::size_of::<ParticleGpu>())
            .min(self.max_upload_bytes_per_frame / std::mem::size_of::<ParticleGpu>())
    }
}

impl Default for ParticleRenderBudget {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

/// Invalid particle visualization resource budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleBudgetError {
    /// Every enabled budget must fit at least one particle instance.
    InvalidLimit,
}

impl fmt::Display for ParticleBudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "particle budget must fit at least one visible instance"
        )
    }
}

impl Error for ParticleBudgetError {}

/// Renderer-owned instanced particles for high-volume simulation visuals.
///
/// The field retains its validated CPU instances so it can be recreated after
/// device loss. Its GPU instance buffer grows only when a replacement exceeds
/// the current capacity.
pub struct ParticleField2d {
    renderer_identity: Arc<()>,
    instance_buffer: Arc<wgpu::Buffer>,
    instances: Vec<ParticleGpu>,
    visible_instances: Vec<ParticleGpu>,
    instance_capacity: usize,
    budget: ParticleRenderBudget,
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

    /// Returns the hard visualization budget applied by rendering.
    pub fn budget(&self) -> ParticleRenderBudget {
        self.budget
    }

    /// Returns retained CPU memory used for recovery and validation.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.instances.len() * std::mem::size_of::<ParticleGpu>()
    }

    /// Returns currently reserved CPU bytes for retained and culled instance lists.
    pub fn cpu_allocation_bytes(&self) -> usize {
        self.instances
            .capacity()
            .saturating_add(self.visible_instances.capacity())
            .saturating_mul(std::mem::size_of::<ParticleGpu>())
    }

    /// Returns allocated GPU instance-buffer bytes.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.instance_capacity
            .saturating_mul(std::mem::size_of::<ParticleGpu>())
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
    /// The instance capacity exceeds the current device's buffer limit.
    CapacityTooLarge,
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
            Self::CapacityTooLarge => {
                write!(
                    formatter,
                    "particle field exceeds the GPU instance-buffer limit"
                )
            }
        }
    }
}

impl Error for ParticleFieldError {}

/// CPU-side outcome of one particle-field update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParticleFieldUpdateReport {
    statistics: ParticleStatistics,
    preparation: Duration,
    reallocated: bool,
}

impl ParticleFieldUpdateReport {
    /// Returns submitted, visible, dropped, and rendered instance counts.
    pub fn statistics(self) -> ParticleStatistics {
        self.statistics
    }

    /// Returns CPU time spent validating and retaining the new visual state.
    ///
    /// GPU upload is deferred until rendering, after camera culling has selected
    /// the compact visible set.
    pub fn preparation(self) -> Duration {
        self.preparation
    }

    /// Backward-compatible alias for [`ParticleFieldUpdateReport::preparation`].
    #[deprecated(note = "particle GPU upload is deferred to render; use preparation()")]
    pub fn upload(self) -> Duration {
        self.preparation
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
    /// The clear color is not normalized linear RGBA.
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
                write!(formatter, "particle field background must be normalized")
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

    /// Returns allocated `R32Float` texel bytes on the GPU.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.recovery_memory_bytes()
    }
}

/// A renderer-owned offscreen color target in physical texture pixels.
///
/// Targets preserve GPU pixels only; a host must redraw them after device
/// recreation. Use the renderer's physical [`WgpuRenderer::size`] when a
/// target should match the presentation surface exactly.
pub struct RenderTarget2d {
    renderer_identity: Arc<()>,
    resource_identity: Arc<()>,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    allocation_bytes: usize,
}

/// A pair of offscreen targets used for bounded temporal accumulation.
///
/// Each accumulation retains a caller-selected amount of the previous frame
/// and then composites a distinct source target over it. The buffer owns no
/// domain state and can therefore serve particles, scalar fields, or any later
/// target-rendering producer.
pub struct TrailBuffer2d {
    renderer_identity: Arc<()>,
    front: RenderTarget2d,
    back: RenderTarget2d,
}

impl TrailBuffer2d {
    /// Returns trail-buffer width in physical texture pixels.
    pub fn width(&self) -> u32 {
        self.front.width()
    }

    /// Returns trail-buffer height in physical texture pixels.
    pub fn height(&self) -> u32 {
        self.front.height()
    }

    /// Returns trail-buffer physical dimensions.
    pub fn size(&self) -> (u32, u32) {
        self.front.size()
    }

    /// Returns total allocated bytes for both ping-pong textures.
    pub fn allocation_bytes(&self) -> usize {
        self.front
            .allocation_bytes()
            .saturating_add(self.back.allocation_bytes())
    }
}

impl RenderTarget2d {
    /// Returns target width in physical texture pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns target height in physical texture pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the physical texture dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the exact single-level GPU texture allocation implied by its format.
    pub fn allocation_bytes(&self) -> usize {
        self.allocation_bytes
    }
}

/// Failure while creating or composing a [`RenderTarget2d`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderTargetError {
    /// Target dimensions must both be non-zero.
    ZeroDimension,
    /// Dimensions exceed the current device's supported target size or allocation range.
    DimensionsTooLarge,
    /// The target belongs to another renderer and GPU device.
    RendererMismatch,
    /// Opacity must be finite and within `0.0..=1.0`.
    InvalidOpacity,
    /// A temporal source must not be one of its destination ping-pong targets.
    SourceAliasesDestination,
    /// The destination clear color was not normalized linear RGBA.
    InvalidBackground,
    /// Surface acquisition or presentation failed.
    Frame(RendererFrameError),
}

impl fmt::Display for RenderTargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(formatter, "render target dimensions must be non-zero"),
            Self::DimensionsTooLarge => {
                write!(formatter, "render target dimensions exceed device limits")
            }
            Self::RendererMismatch => {
                write!(formatter, "render target belongs to a different renderer")
            }
            Self::InvalidOpacity => write!(
                formatter,
                "render target opacity must be finite and in 0.0..=1.0"
            ),
            Self::SourceAliasesDestination => {
                write!(
                    formatter,
                    "trail source must be distinct from its destination targets"
                )
            }
            Self::InvalidBackground => {
                write!(formatter, "render target background must be normalized")
            }
            Self::Frame(error) => write!(formatter, "render target frame failed: {error}"),
        }
    }
}

impl Error for RenderTargetError {}

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
    /// The clear color is not normalized linear RGBA.
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
            Self::InvalidBackground => {
                write!(formatter, "scalar field background must be normalized")
            }
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
    /// The clear color is not normalized linear RGBA.
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
            Self::InvalidBackground => {
                write!(formatter, "dynamic mesh background must be normalized")
            }
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

impl PreparedScreenScene {
    /// Returns the number of accepted source screen commands.
    pub fn command_count(&self) -> usize {
        self.scene.command_count()
    }

    /// Returns the retained triangle-list vertex count.
    pub fn vertex_count(&self) -> usize {
        self.scene.vertex_count()
    }

    /// Returns the number of clip-compatible draw batches.
    pub fn draw_batch_count(&self) -> usize {
        self.scene.draw_batch_count()
    }

    /// Returns retained CPU bytes available for device-loss recovery.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.scene.recovery_memory_bytes()
    }

    /// Returns preparation-time tessellation statistics.
    pub fn tessellation_stats(&self) -> TessellationStats {
        self.scene.tessellation_stats()
    }
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
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    requested_present_mode: RendererPresentMode,
    surface_present_mode: RendererSurfacePresentMode,
    pre_present_notify: Option<Arc<dyn Fn() + Send + Sync>>,
    scale_factor: f64,
    pipeline: wgpu::RenderPipeline,
    target_pipeline: wgpu::RenderPipeline,
    dynamic_pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,
    target_particle_pipeline: wgpu::RenderPipeline,
    heatmap_pipeline: wgpu::RenderPipeline,
    target_heatmap_pipeline: wgpu::RenderPipeline,
    composition_pipelines: CompositionPipelines,
    target_composition_pipelines: CompositionPipelines,
    image_renderer: ImageRenderer,
    mesh3d_renderer: Mesh3dRenderer,
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    heatmap_uniform_buffer: wgpu::Buffer,
    heatmap_bind_group_layout: wgpu::BindGroupLayout,
    color_map_cache: Option<CachedColorMap>,
    vertex_buffer: Arc<wgpu::Buffer>,
    particle_unit_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    multisample_target: Option<MultisampleTarget>,
    sample_count: u32,
    vertices: Vec<Vertex>,
    draw_batches: Vec<PreparedDrawBatch>,
    retired_devices: Vec<RetiredDevice>,
    max_quarantined_devices: usize,
}

struct RetiredDevice {
    _adapter: wgpu::Adapter,
    _device: wgpu::Device,
    _queue: wgpu::Queue,
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
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
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
        let adapter_info = adapter.get_info();

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
        let surface_capabilities = surface.get_capabilities(&adapter);
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(RendererInitError::NoSurfaceConfig)?;
        let surface_present_mode = select_surface_present_mode(
            options.present_mode(),
            &surface_capabilities.present_modes,
        );
        config.present_mode = surface_present_mode.to_wgpu();
        surface.configure(&device, &config);

        let sample_count = preferred_sample_count(&adapter, config.format);
        let PipelineResources {
            pipeline,
            target_pipeline,
            dynamic_pipeline,
            particle_pipeline,
            target_particle_pipeline,
            heatmap_pipeline,
            target_heatmap_pipeline,
            composition_pipelines,
            target_composition_pipelines,
            camera_uniform_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            heatmap_uniform_buffer,
            heatmap_bind_group_layout,
        } = create_pipeline(&device, config.format, sample_count);
        let vertex_buffer = Arc::new(create_vertex_buffer(&device, INITIAL_VERTEX_CAPACITY));
        let particle_unit_buffer = create_particle_unit_buffer(&device, &queue);
        let multisample_target = create_multisample_target(&device, &config, sample_count);
        let image_renderer = ImageRenderer::new(&device, config.format, sample_count);
        let mesh3d_renderer = Mesh3dRenderer::new(&device, config.format);

        Ok(Self {
            renderer_identity: Arc::new(()),
            _instance: instance,
            surface,
            _adapter: adapter,
            adapter_info,
            device,
            queue,
            config,
            requested_present_mode: options.present_mode(),
            surface_present_mode,
            pre_present_notify: None,
            scale_factor: options.scale_factor(),
            pipeline,
            target_pipeline,
            dynamic_pipeline,
            particle_pipeline,
            target_particle_pipeline,
            heatmap_pipeline,
            target_heatmap_pipeline,
            composition_pipelines,
            target_composition_pipelines,
            image_renderer,
            mesh3d_renderer,
            camera_uniform_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            heatmap_uniform_buffer,
            heatmap_bind_group_layout,
            color_map_cache: None,
            vertex_buffer,
            particle_unit_buffer,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
            multisample_target,
            sample_count,
            vertices: Vec::with_capacity(INITIAL_VERTEX_CAPACITY),
            draw_batches: Vec::new(),
            retired_devices: Vec::new(),
            max_quarantined_devices: options.max_quarantined_devices(),
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

    /// Returns the concrete presentation mode selected for the active surface.
    ///
    /// `RendererPresentMode::NoVsync` prefers Immediate, then Mailbox, then
    /// FIFO. This method reveals which supported fallback was configured. The
    /// desktop compositor may still pace host redraw callbacks independently.
    pub fn surface_present_mode(&self) -> RendererSurfacePresentMode {
        self.surface_present_mode
    }

    /// Installs a host callback invoked immediately before synchronized
    /// surface presentation.
    ///
    /// A winit host should capture its `Arc<Window>` and call
    /// `Window::pre_present_notify` from this callback. The renderer invokes
    /// it after command submission and immediately before `Queue::present`,
    /// but only when the concrete surface mode is Mailbox, FIFO, or
    /// FIFO-relaxed. Immediate presentation deliberately remains uncapped.
    pub fn set_pre_present_notify(&mut self, notify: impl Fn() + Send + Sync + 'static) {
        self.pre_present_notify = Some(Arc::new(notify));
    }

    /// Removes the synchronized-presentation callback, if one is installed.
    pub fn clear_pre_present_notify(&mut self) {
        self.pre_present_notify = None;
    }

    fn notify_before_present(&self) {
        invoke_pre_present_notify(
            self.surface_present_mode,
            self.pre_present_notify.as_deref(),
        );
    }

    /// Returns the active graphics adapter's human-readable name.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Returns the active graphics API backend as a stable lowercase name.
    pub fn adapter_backend(&self) -> &'static str {
        self.adapter_info.backend.to_str()
    }

    /// Returns the PCI vendor identifier reported for the active adapter.
    pub fn adapter_vendor_id(&self) -> u32 {
        self.adapter_info.vendor
    }

    /// Returns the device identifier reported for the active adapter.
    pub fn adapter_device_id(&self) -> u32 {
        self.adapter_info.device
    }

    /// Returns the backend-reported PCI bus address for the active adapter.
    ///
    /// Vulkan adapters expose this as `domain:bus:device.function` when the
    /// driver supports `VK_EXT_pci_bus_info`. An empty string means that the
    /// backend cannot identify a physical adapter instance; release evidence
    /// must not treat model identifiers alone as instance identity.
    pub fn adapter_pci_bus_id(&self) -> &str {
        &self.adapter_info.device_pci_bus_id
    }

    /// Returns the texture format selected for the active presentation surface.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Returns the raster sample count selected for the active surface format.
    pub fn surface_sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Returns the driver name reported by the active graphics adapter.
    pub fn adapter_driver(&self) -> &str {
        &self.adapter_info.driver
    }

    /// Returns the driver version or implementation detail reported by the adapter.
    pub fn adapter_driver_info(&self) -> &str {
        &self.adapter_info.driver_info
    }

    /// Returns previous logical devices retained for safe native-driver teardown.
    pub fn quarantined_device_count(&self) -> usize {
        self.retired_devices.len()
    }

    /// Returns the configured maximum previous-device quarantine size.
    pub fn max_quarantined_device_count(&self) -> usize {
        self.max_quarantined_devices
    }

    /// Returns successful recoveries available before the bounded quarantine is full.
    pub fn remaining_device_recoveries(&self) -> usize {
        self.max_quarantined_devices
            .saturating_sub(self.retired_devices.len())
    }

    /// Blocks until all previously submitted GPU work has completed.
    ///
    /// This is intended for diagnostic throughput measurements, readback, and
    /// controlled resource teardown. Calling it in an interactive frame loop
    /// defeats normal CPU/GPU pipelining.
    pub fn wait_for_gpu_idle(&self) -> Result<(), wgpu::PollError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
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

    /// Draws fixed logical-screen geometry independently of any world camera.
    pub fn render_screen_scene(
        &mut self,
        scene: &ScreenScene,
    ) -> Result<RenderStatus, RendererFrameError> {
        self.render_screen_scene_with_metrics(scene)
            .map(RenderReport::status)
    }

    /// Draws fixed logical-screen geometry and returns CPU-side stage timings.
    pub fn render_screen_scene_with_metrics(
        &mut self,
        scene: &ScreenScene,
    ) -> Result<RenderReport, RendererFrameError> {
        let viewport = LogicalViewport::new(self.logical_size().0, self.logical_size().1)
            .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
        let camera =
            screen_camera(viewport).map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
        self.render_with_metrics(scene.as_scene(), &camera)
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
        self.render_with_metrics_in_region(scene, camera, None)
    }

    /// Draws a world scene through a camera into one bounded logical viewport.
    ///
    /// Scene-local clips are interpreted relative to the viewport and are
    /// intersected with its physical bounds. The surface outside the viewport
    /// is cleared but receives no scene geometry.
    pub fn render_scene_in_viewport(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
        viewport: LogicalViewportRegion,
    ) -> Result<RenderStatus, RendererFrameError> {
        self.render_scene_in_viewport_with_metrics(scene, camera, viewport)
            .map(RenderReport::status)
    }

    /// Draws one bounded scene viewport and returns CPU-side stage timings.
    pub fn render_scene_in_viewport_with_metrics(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
        viewport: LogicalViewportRegion,
    ) -> Result<RenderReport, RendererFrameError> {
        self.render_with_metrics_in_region(scene, camera, Some(viewport))
    }

    /// Renders an ordinary world scene into an offscreen target.
    ///
    /// Target dimensions are physical texels. `pixel_scale` defines their
    /// logical viewport density independently of window DPI. Scene-local clips
    /// are relative to `viewport`, or to the complete logical target when no
    /// region is supplied. `load` explicitly controls preservation or clearing.
    #[allow(clippy::too_many_arguments)]
    pub fn render_scene_to_target(
        &mut self,
        target: &RenderTarget2d,
        scene: &Scene,
        camera: &Camera2d,
        pixel_scale: PhysicalPerLogical,
        viewport: Option<LogicalViewportRegion>,
        load: RenderTargetLoad,
    ) -> Result<RenderReport, RenderTargetError> {
        self.validate_render_target(target)?;
        if let RenderTargetLoad::Clear(color) = load
            && !color.is_normalized()
        {
            return Err(RenderTargetError::InvalidBackground);
        }

        let frame_started_at = Instant::now();
        let tessellation_started_at = Instant::now();
        self.vertices.clear();
        self.draw_batches.clear();
        let tessellation_stats =
            tessellate_scene(scene, &mut self.vertices, &mut self.draw_batches)
                .map_err(RendererFrameError::from)
                .map_err(RenderTargetError::Frame)?;
        self.ensure_vertex_capacity(self.vertices.len())
            .map_err(RenderTargetError::Frame)?;
        let tessellation = tessellation_started_at.elapsed();

        let upload_started_at = Instant::now();
        if !self.vertices.is_empty() {
            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        }
        let upload = upload_started_at.elapsed();

        let scale = pixel_scale.get();
        let target_viewport = LogicalViewport::new(
            target.width() as f32 / scale,
            target.height() as f32 / scale,
        )
        .map_err(|_| RenderTargetError::Frame(RendererFrameError::InvalidViewport))?;
        let (local_viewport, origin) = match viewport {
            Some(region) => (region.viewport(), region.origin().to_vec2()),
            None => (target_viewport, Vec2::ZERO),
        };
        let max = origin + local_viewport.size();
        if !max.is_finite()
            || origin.x < 0.0
            || origin.y < 0.0
            || max.x > target_viewport.width()
            || max.y > target_viewport.height()
        {
            return Err(RenderTargetError::Frame(
                RendererFrameError::InvalidViewport,
            ));
        }
        let target_scissor = logical_viewport_scissor(
            origin,
            local_viewport,
            scale,
            target.width(),
            target.height(),
        )
        .ok_or(RenderTargetError::Frame(
            RendererFrameError::InvalidViewport,
        ))?;
        let camera_uniform =
            CameraUniform::new_in_region(*camera, local_viewport, origin, target_viewport).ok_or(
                RenderTargetError::Frame(RendererFrameError::InvalidGeometryTransform),
            )?;
        let extents = GeometryExtents::from_vertices(&self.vertices);
        if !extents.is_safe_for(camera_uniform) {
            return Err(RenderTargetError::Frame(
                RendererFrameError::InvalidGeometryTransform,
            ));
        }
        let uniform_started_at = Instant::now();
        self.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        let camera_uniform_upload = uniform_started_at.elapsed();

        let encode_started_at = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine scene target encoder"),
            });
        {
            let load = match load {
                RenderTargetLoad::Load => wgpu::LoadOp::Load,
                RenderTargetLoad::Clear(color) => wgpu::LoadOp::Clear(color.to_wgpu()),
            };
            let color_attachment = wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine scene target pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !self.vertices.is_empty() {
                pass.set_pipeline(&self.target_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                for batch in &self.draw_batches {
                    let scissor = match batch.screen_clip {
                        Some(clip) => {
                            let Some(local) = screen_clip_to_scissor(clip, local_viewport, scale)
                            else {
                                continue;
                            };
                            let Some(scissor) = offset_scissor(local, target_scissor) else {
                                continue;
                            };
                            scissor
                        }
                        None => target_scissor,
                    };
                    pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                    pass.draw(batch.vertex_range.clone(), 0..1);
                }
            }
        }
        self.queue.submit([encoder.finish()]);
        let encode = encode_started_at.elapsed();

        Ok(render_report(
            RenderStatus::Drawn,
            tessellation,
            upload,
            camera_uniform_upload,
            Duration::ZERO,
            encode,
            frame_started_at.elapsed(),
            false,
            true,
            tessellation_stats,
        ))
    }

    fn render_with_metrics_in_region(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
        viewport: Option<LogicalViewportRegion>,
    ) -> Result<RenderReport, RendererFrameError> {
        let frame_started_at = Instant::now();
        let tessellation_started_at = Instant::now();
        self.vertices.clear();
        self.draw_batches.clear();
        let tessellation_stats =
            tessellate_scene(scene, &mut self.vertices, &mut self.draw_batches)?;
        self.ensure_vertex_capacity(self.vertices.len())?;
        let tessellation = tessellation_started_at.elapsed();

        let upload_started_at = Instant::now();
        if !self.vertices.is_empty() {
            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        }
        let upload = upload_started_at.elapsed();

        let vertex_buffer = Arc::clone(&self.vertex_buffer);
        let draw_batches = std::mem::take(&mut self.draw_batches);
        let geometry_extents = GeometryExtents::from_vertices(&self.vertices);
        let result = self.draw_geometry(
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
            viewport,
        );
        self.draw_batches = draw_batches;
        result
    }

    /// Tessellates a scene once and uploads immutable geometry to a dedicated GPU buffer.
    ///
    /// Preparing is appropriate for geometry that will be drawn repeatedly while
    /// only the camera or target dimensions change. Any shape, style, gradient,
    /// ordering, or clipping change requires preparing a replacement scene.
    pub fn prepare_scene(&self, scene: &Scene) -> Result<PreparedScene, PreparedSceneError> {
        prepare_scene_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            scene,
        )
    }

    /// Tessellates fixed logical-screen geometry once for repeated composition.
    pub fn prepare_screen_scene(
        &self,
        scene: &ScreenScene,
    ) -> Result<PreparedScreenScene, PreparedSceneError> {
        self.prepare_scene(scene.as_scene())
            .map(|scene| PreparedScreenScene { scene })
    }

    /// Restores prepared logical-screen geometry on a recreated renderer.
    pub fn restore_prepared_screen_scene(
        &self,
        source: &PreparedScreenScene,
    ) -> Result<PreparedScreenScene, PreparedSceneError> {
        self.restore_prepared_scene(&source.scene)
            .map(|scene| PreparedScreenScene { scene })
    }

    /// Recreates prepared GPU resources on this renderer from a retained CPU snapshot.
    ///
    /// This supports renderer recreation after device loss without requiring the
    /// original high-level [`Scene`]. The returned snapshot belongs to this renderer.
    pub fn restore_prepared_scene(
        &self,
        source: &PreparedScene,
    ) -> Result<PreparedScene, PreparedSceneError> {
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
    pub fn restore_dynamic_mesh(
        &self,
        source: &DynamicMesh2d,
    ) -> Result<DynamicMesh2d, DynamicMeshError> {
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
        self.create_dynamic_mesh_inner(vertices, None)
    }

    /// Creates bounded mutable colored triangles for composition in a frame.
    ///
    /// Validation and CPU reservation complete before any GPU allocation. The
    /// retained budget is also enforced by every later full or partial update.
    pub fn create_dynamic_mesh_with_budget(
        &self,
        vertices: &[DynamicVertex2d],
        budget: DynamicMeshBudget,
    ) -> Result<DynamicMesh2d, DynamicMeshError> {
        self.create_dynamic_mesh_inner(vertices, Some(budget))
    }

    fn create_dynamic_mesh_inner(
        &self,
        vertices: &[DynamicVertex2d],
        budget: Option<DynamicMeshBudget>,
    ) -> Result<DynamicMesh2d, DynamicMeshError> {
        if let Some(budget) = budget {
            validate_dynamic_mesh_budget(budget, vertices.len())?;
        }
        let vertices = dynamic_vertices_to_gpu(vertices)?;
        let vertex_capacity = dynamic_vertex_capacity(vertices.len())
            .filter(|capacity| buffer_capacity_fits::<DynamicGpu>(&self.device, *capacity))
            .ok_or(DynamicMeshError::CapacityTooLarge)?;
        let vertex_buffer = Arc::new(create_dynamic_vertex_buffer(&self.device, vertex_capacity));
        if !vertices.is_empty() {
            self.queue
                .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }
        Ok(DynamicMesh2d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            geometry_extents: GeometryExtents::from_dynamic_vertices(&vertices),
            vertex_buffer,
            vertices,
            vertex_capacity,
            budget,
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
        self.validate_dynamic_mesh(mesh)?;
        replace_dynamic_mesh_resources(&self.device, &self.queue, mesh, vertices)
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
        if let Some(budget) = mesh.budget {
            validate_dynamic_mesh_budget(budget, mesh.vertices.len())?;
            let upload_bytes = dynamic_mesh_bytes(vertices.len())?;
            if upload_bytes > budget.max_upload_bytes {
                return Err(DynamicMeshError::BudgetExceeded {
                    resource: DynamicMeshBudgetResource::UploadBytes,
                    limit: budget.max_upload_bytes,
                    actual: upload_bytes,
                });
            }
        }
        let vertices = dynamic_vertices_to_gpu(vertices)?;
        if !vertices.is_empty() {
            let offset = (first_vertex * std::mem::size_of::<DynamicGpu>()) as wgpu::BufferAddress;
            self.queue
                .write_buffer(&mesh.vertex_buffer, offset, bytemuck::cast_slice(&vertices));
            mesh.vertices[first_vertex..end].copy_from_slice(&vertices);
            mesh.geometry_extents = GeometryExtents::from_dynamic_vertices(&mesh.vertices);
        }
        Ok(DynamicMeshUpdateReport {
            vertex_count: mesh.vertices.len(),
            upload: update_started_at.elapsed(),
            reallocated: false,
        })
    }

    /// Draws dynamic triangle-list geometry with a normalized clear color.
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
        if !background.is_normalized() {
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
            None,
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
        self.create_particle_field_with_budget(instances, ParticleRenderBudget::UNBOUNDED)
    }

    /// Creates an instanced particle field with hard GPU-memory and upload limits.
    pub fn create_particle_field_with_budget(
        &self,
        instances: &[ParticleInstance2d],
        budget: ParticleRenderBudget,
    ) -> Result<ParticleField2d, ParticleFieldError> {
        let instances = particle_instances_to_gpu(instances)?;
        let instance_capacity = particle_budgeted_capacity(instances.len(), budget)
            .filter(|capacity| buffer_capacity_fits::<ParticleGpu>(&self.device, *capacity))
            .ok_or(ParticleFieldError::CapacityTooLarge)?;
        let instance_buffer = Arc::new(create_particle_instance_buffer(
            &self.device,
            instance_capacity,
        ));
        Ok(ParticleField2d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            instance_buffer,
            statistics: particle_statistics(instances.len(), instances.len(), 0),
            instances,
            visible_instances: Vec::new(),
            instance_capacity,
            budget,
        })
    }

    /// Replaces a particle field's hard resource budget, shrinking memory when required.
    pub fn set_particle_field_budget(
        &self,
        field: &mut ParticleField2d,
        budget: ParticleRenderBudget,
    ) -> Result<(), ParticleFieldError> {
        self.validate_particle_field(field)?;
        let desired_capacity = particle_budgeted_capacity(field.instances.len(), budget)
            .filter(|capacity| buffer_capacity_fits::<ParticleGpu>(&self.device, *capacity))
            .ok_or(ParticleFieldError::CapacityTooLarge)?;
        if desired_capacity != field.instance_capacity {
            field.instance_buffer = Arc::new(create_particle_instance_buffer(
                &self.device,
                desired_capacity,
            ));
            field.instance_capacity = desired_capacity;
        }
        field.budget = budget;
        field.visible_instances.clear();
        field.statistics = particle_statistics(field.instances.len(), field.instances.len(), 0);
        Ok(())
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

    /// Replaces all particle instances and returns CPU-side preparation metrics.
    pub fn update_particle_field_with_metrics(
        &self,
        field: &mut ParticleField2d,
        instances: &[ParticleInstance2d],
    ) -> Result<ParticleFieldUpdateReport, ParticleFieldError> {
        let update_started_at = Instant::now();
        self.validate_particle_field(field)?;
        let instances = particle_instances_to_gpu(instances)?;
        let desired_capacity = particle_budgeted_capacity(instances.len(), field.budget)
            .filter(|capacity| buffer_capacity_fits::<ParticleGpu>(&self.device, *capacity))
            .ok_or(ParticleFieldError::CapacityTooLarge)?;
        let reallocated = desired_capacity > field.instance_capacity;
        if reallocated {
            field.instance_capacity = desired_capacity;
            field.instance_buffer = Arc::new(create_particle_instance_buffer(
                &self.device,
                field.instance_capacity,
            ));
        }
        field.statistics = particle_statistics(instances.len(), instances.len(), 0);
        field.instances = instances;
        Ok(ParticleFieldUpdateReport {
            statistics: field.statistics,
            preparation: update_started_at.elapsed(),
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

    /// Replaces a particle range and returns CPU-side preparation metrics.
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
            field.instances[range].copy_from_slice(&instances);
        }
        field.statistics = particle_statistics(field.instances.len(), field.instances.len(), 0);
        Ok(ParticleFieldUpdateReport {
            statistics: field.statistics,
            preparation: update_started_at.elapsed(),
            reallocated: false,
        })
    }

    /// Recreates a particle field on this renderer from retained CPU instances.
    pub fn restore_particle_field(
        &self,
        source: &ParticleField2d,
    ) -> Result<ParticleField2d, ParticleFieldError> {
        restore_particle_field_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            source,
        )
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

    /// Creates an offscreen target with physical texture dimensions.
    pub fn create_render_target(
        &self,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget2d, RenderTargetError> {
        if width == 0 || height == 0 {
            return Err(RenderTargetError::ZeroDimension);
        }
        if width > self.device.limits().max_texture_dimension_2d
            || height > self.device.limits().max_texture_dimension_2d
        {
            return Err(RenderTargetError::DimensionsTooLarge);
        }
        let allocation_bytes = render_target_allocation_bytes(self.config.format, width, height)
            .ok_or(RenderTargetError::DimensionsTooLarge)?;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(RenderTarget2d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            resource_identity: Arc::new(()),
            _texture: texture,
            view,
            width,
            height,
            allocation_bytes,
        })
    }

    /// Creates an empty ping-pong target pair for temporal accumulation.
    pub fn create_trail_buffer(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<TrailBuffer2d, RenderTargetError> {
        let front = self.create_render_target(width, height)?;
        let back = self.create_render_target(width, height)?;
        let trails = TrailBuffer2d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            front,
            back,
        };
        self.draw_clear_trail_buffer(&trails, Color::rgba(0.0, 0.0, 0.0, 0.0), Instant::now());
        Ok(trails)
    }

    /// Recreates an empty target with the source dimensions after device recovery.
    ///
    /// GPU pixels are deliberately not retained; callers redraw their source
    /// state into the restored target.
    pub fn restore_render_target(
        &self,
        source: &RenderTarget2d,
    ) -> Result<RenderTarget2d, RenderTargetError> {
        self.create_render_target(source.width(), source.height())
    }

    /// Recreates an empty, deterministically cleared trail buffer after recovery.
    pub fn restore_trail_buffer(
        &mut self,
        source: &TrailBuffer2d,
    ) -> Result<TrailBuffer2d, RenderTargetError> {
        self.create_trail_buffer(source.width(), source.height())
    }

    /// Retains one bounded frame of history and composites a fresh source target.
    ///
    /// `retention` and `source_opacity` must be finite values in `0.0..=1.0`.
    /// The source must be a distinct target; use [`WgpuRenderer::clear_trail_buffer`]
    /// to discard accumulated history deterministically.
    pub fn accumulate_trail_buffer(
        &mut self,
        trails: &mut TrailBuffer2d,
        source: &RenderTarget2d,
        retention: f32,
        source_opacity: f32,
        source_blend: BlendMode,
    ) -> Result<RenderReport, RenderTargetError> {
        self.validate_trail_buffer(trails)?;
        self.validate_render_target(source)?;
        if Arc::ptr_eq(&source.resource_identity, &trails.front.resource_identity)
            || Arc::ptr_eq(&source.resource_identity, &trails.back.resource_identity)
        {
            return Err(RenderTargetError::SourceAliasesDestination);
        }
        if !retention.is_finite()
            || !(0.0..=1.0).contains(&retention)
            || !source_opacity.is_finite()
            || !(0.0..=1.0).contains(&source_opacity)
        {
            return Err(RenderTargetError::InvalidOpacity);
        }
        Ok(self.draw_trail_accumulation(
            trails,
            source,
            retention,
            source_opacity,
            source_blend,
            Instant::now(),
        ))
    }

    /// Clears both ping-pong targets, resetting all retained temporal history.
    pub fn clear_trail_buffer(
        &mut self,
        trails: &mut TrailBuffer2d,
        color: Color,
    ) -> Result<RenderReport, RenderTargetError> {
        self.validate_trail_buffer(trails)?;
        if !color.is_normalized() {
            return Err(RenderTargetError::InvalidBackground);
        }
        Ok(self.draw_clear_trail_buffer(trails, color, Instant::now()))
    }

    /// Presents the most recently accumulated temporal target.
    pub fn compose_trail_buffer(
        &mut self,
        trails: &TrailBuffer2d,
        blend_mode: BlendMode,
        opacity: f32,
        background: Color,
    ) -> Result<RenderReport, RenderTargetError> {
        self.validate_trail_buffer(trails)?;
        self.compose_render_target(&trails.front, blend_mode, opacity, background)
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
        let value_extent = scalar_value_range_extent(minimum, maximum)
            .ok_or(ScalarFieldRenderError::InvalidValueRange { minimum, maximum })?;
        if !background.is_normalized() {
            return Err(ScalarFieldRenderError::InvalidBackground);
        }
        self.draw_scalar_field_texture(
            texture,
            color_map,
            minimum,
            value_extent,
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
        let value_extent = scalar_value_range_extent(minimum, maximum)
            .ok_or(ScalarFieldRenderError::InvalidValueRange { minimum, maximum })?;
        if !background.is_normalized() {
            return Err(ScalarFieldRenderError::InvalidBackground);
        }
        self.draw_scalar_field_texture(
            texture,
            color_map,
            minimum,
            value_extent,
            sampling,
            background,
            Instant::now(),
        )
        .map(RenderReport::status)
        .map_err(ScalarFieldRenderError::Frame)
    }

    /// Renders a scalar heatmap into an offscreen target instead of presenting it.
    pub fn render_scalar_field_texture_to_target(
        &mut self,
        target: &RenderTarget2d,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        value_range: (f32, f32),
        background: Color,
    ) -> Result<RenderReport, ScalarFieldRenderError> {
        self.render_scalar_field_texture_to_target_with_sampling(
            target,
            texture,
            color_map,
            value_range,
            ScalarFieldSampling::Nearest,
            background,
        )
    }

    /// Renders a scalar heatmap into a target with explicit source sampling.
    pub fn render_scalar_field_texture_to_target_with_sampling(
        &mut self,
        target: &RenderTarget2d,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        (minimum, maximum): (f32, f32),
        sampling: ScalarFieldSampling,
        background: Color,
    ) -> Result<RenderReport, ScalarFieldRenderError> {
        self.validate_render_target(target)
            .map_err(|_| ScalarFieldRenderError::RendererMismatch)?;
        self.validate_scalar_field_texture(texture)
            .map_err(|_| ScalarFieldRenderError::RendererMismatch)?;
        let value_extent = scalar_value_range_extent(minimum, maximum)
            .ok_or(ScalarFieldRenderError::InvalidValueRange { minimum, maximum })?;
        if !background.is_normalized() {
            return Err(ScalarFieldRenderError::InvalidBackground);
        }
        Ok(self.draw_scalar_field_texture_to_target(
            target,
            texture,
            color_map,
            minimum,
            value_extent,
            sampling,
            background,
            Instant::now(),
        ))
    }

    /// Composes a previously rendered target over the presentation surface.
    pub fn compose_render_target(
        &mut self,
        target: &RenderTarget2d,
        blend_mode: BlendMode,
        opacity: f32,
        background: Color,
    ) -> Result<RenderReport, RenderTargetError> {
        self.validate_render_target(target)?;
        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return Err(RenderTargetError::InvalidOpacity);
        }
        if !background.is_normalized() {
            return Err(RenderTargetError::InvalidBackground);
        }
        self.draw_composed_render_target(target, blend_mode, opacity, background, Instant::now())
            .map_err(RenderTargetError::Frame)
    }

    /// Draws a particle field with a normalized clear color.
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
        if !background.is_normalized() {
            return Err(ParticleFieldRenderError::InvalidBackground);
        }
        self.draw_particle_field(background, field, *camera, Instant::now())
            .map_err(|error| match error {
                RendererFrameError::InvalidGeometryTransform => {
                    ParticleFieldRenderError::InvalidGeometryTransform
                }
                other => ParticleFieldRenderError::Frame(other),
            })
    }

    /// Draws a particle field into an offscreen target.
    ///
    /// Target dimensions may be lower than the presentation surface to bound
    /// raster cost. Camera coordinates and particle radii remain expressed in
    /// logical presentation pixels, so composing the target back to the surface
    /// preserves layout while trading only raster resolution.
    pub fn render_particle_field_to_target(
        &mut self,
        target: &RenderTarget2d,
        field: &mut ParticleField2d,
        camera: &Camera2d,
        load: RenderTargetLoad,
    ) -> Result<RenderReport, ParticleFieldRenderError> {
        self.validate_render_target(target)
            .map_err(|_| ParticleFieldRenderError::RendererMismatch)?;
        self.validate_particle_field(field)
            .map_err(|_| ParticleFieldRenderError::RendererMismatch)?;
        if matches!(load, RenderTargetLoad::Clear(color) if !color.is_normalized()) {
            return Err(ParticleFieldRenderError::InvalidBackground);
        }
        self.draw_particle_field_to_target(target, field, *camera, load, Instant::now())
            .map_err(|error| match error {
                RendererFrameError::InvalidGeometryTransform => {
                    ParticleFieldRenderError::InvalidGeometryTransform
                }
                other => ParticleFieldRenderError::Frame(other),
            })
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

    /// Draws prepared logical-screen geometry independently of a world camera.
    ///
    /// The prepared resource must have been built from a [`ScreenScene`]. The
    /// renderer cannot infer the original coordinate space from raw prepared
    /// vertices, so passing world-space geometry is a caller contract error.
    pub fn render_prepared_screen_scene(
        &mut self,
        scene: &PreparedScreenScene,
    ) -> Result<RenderStatus, PreparedSceneRenderError> {
        let (width, height) = self.logical_size();
        let viewport = LogicalViewport::new(width, height).map_err(|_| {
            PreparedSceneRenderError::Frame(RendererFrameError::InvalidGeometryTransform)
        })?;
        let camera = screen_camera(viewport).map_err(|_| {
            PreparedSceneRenderError::Frame(RendererFrameError::InvalidGeometryTransform)
        })?;
        self.render_prepared(&scene.scene, &camera)
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
            None,
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
        viewport_region: Option<LogicalViewportRegion>,
    ) -> Result<RenderReport, RendererFrameError> {
        let (logical_width, logical_height) = self.logical_size();
        let target_viewport = LogicalViewport::new(logical_width, logical_height)
            .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
        let (viewport, viewport_origin) = match viewport_region {
            Some(region) => (region.viewport(), region.origin().to_vec2()),
            None => (target_viewport, Vec2::ZERO),
        };
        let viewport_max = viewport_origin + viewport.size();
        if !viewport_origin.is_finite()
            || !viewport_max.is_finite()
            || viewport_origin.x < 0.0
            || viewport_origin.y < 0.0
            || viewport_max.x > target_viewport.width()
            || viewport_max.y > target_viewport.height()
        {
            return Err(RendererFrameError::InvalidViewport);
        }
        let viewport_scissor = logical_viewport_scissor(
            viewport_origin,
            viewport,
            self.scale_factor as f32,
            self.config.width,
            self.config.height,
        )
        .ok_or(RendererFrameError::InvalidViewport)?;
        let camera_uniform_upload_started_at = Instant::now();
        let Some(camera_uniform) =
            CameraUniform::new_in_region(camera, viewport, viewport_origin, target_viewport)
        else {
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
                let pipeline = if geometry_streamed {
                    &self.dynamic_pipeline
                } else {
                    &self.pipeline
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                for batch in draw_batches {
                    let scissor = match batch.screen_clip {
                        Some(screen_clip) => {
                            let Some(local_scissor) = screen_clip_to_scissor(
                                screen_clip,
                                viewport,
                                self.scale_factor as f32,
                            ) else {
                                continue;
                            };
                            let Some(scissor) = offset_scissor(local_scissor, viewport_scissor)
                            else {
                                continue;
                            };
                            scissor
                        }
                        None => viewport_scissor,
                    };
                    pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                    pass.draw(batch.vertex_range.clone(), 0..1);
                }
            }
        }

        self.queue.submit([encoder.finish()]);
        self.notify_before_present();
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
        value_extent: f32,
        sampling: ScalarFieldSampling,
        background: Color,
        frame_started_at: Instant,
    ) -> Result<RenderReport, RendererFrameError> {
        let upload_started_at = Instant::now();
        let color_map_view = self.color_map_view(color_map);
        let scalar_view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let uniform = HeatmapUniform::new(
            minimum,
            value_extent,
            texture.width(),
            texture.height(),
            sampling,
        );
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
        self.notify_before_present();
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

    #[allow(clippy::too_many_arguments)]
    fn draw_scalar_field_texture_to_target(
        &mut self,
        target: &RenderTarget2d,
        texture: &ScalarFieldTexture,
        color_map: &ColorMap,
        minimum: f32,
        value_extent: f32,
        sampling: ScalarFieldSampling,
        background: Color,
        frame_started_at: Instant,
    ) -> RenderReport {
        let upload_started_at = Instant::now();
        let color_map_view = self.color_map_view(color_map);
        let scalar_view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.queue.write_buffer(
            &self.heatmap_uniform_buffer,
            0,
            bytemuck::bytes_of(&HeatmapUniform::new(
                minimum,
                value_extent,
                texture.width(),
                texture.height(),
                sampling,
            )),
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine target heatmap bind group"),
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
        let encode_started_at = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine render-target heatmap encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine render-target heatmap pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(premultiplied_wgpu_color(background)),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.target_heatmap_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        render_report(
            RenderStatus::Drawn,
            Duration::ZERO,
            upload,
            Duration::ZERO,
            Duration::ZERO,
            encode_started_at.elapsed(),
            frame_started_at.elapsed(),
            false,
            true,
            TessellationStats::default(),
        )
    }

    fn draw_composed_render_target(
        &mut self,
        target: &RenderTarget2d,
        blend_mode: BlendMode,
        opacity: f32,
        background: Color,
        frame_started_at: Instant,
    ) -> Result<RenderReport, RendererFrameError> {
        let upload_started_at = Instant::now();
        self.queue.write_buffer(
            &self.composition_pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(opacity)),
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine composition bind group"),
            layout: &self.composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .composition_pipelines
                        .uniform_buffer
                        .as_entire_binding(),
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
            Some(multisample_target) => (&multisample_target.view, Some(&surface_view)),
            None => (&surface_view, None),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine composition encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine composition pass"),
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
            pass.set_pipeline(self.composition_pipelines.pipeline(blend_mode));
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.notify_before_present();
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

    #[allow(clippy::too_many_arguments)]
    fn draw_trail_accumulation(
        &mut self,
        trails: &mut TrailBuffer2d,
        source: &RenderTarget2d,
        retention: f32,
        source_opacity: f32,
        source_blend: BlendMode,
        frame_started_at: Instant,
    ) -> RenderReport {
        let upload_started_at = Instant::now();
        self.queue.write_buffer(
            &self.target_composition_pipelines.secondary_uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(retention)),
        );
        self.queue.write_buffer(
            &self.target_composition_pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(source_opacity)),
        );
        let history_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine trail history bind group"),
            layout: &self.target_composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&trails.front.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .target_composition_pipelines
                        .secondary_uniform_buffer
                        .as_entire_binding(),
                },
            ],
        });
        let source_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine trail source bind group"),
            layout: &self.target_composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .target_composition_pipelines
                        .uniform_buffer
                        .as_entire_binding(),
                },
            ],
        });
        let upload = upload_started_at.elapsed();
        let encode_started_at = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine trail accumulation encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine trail history pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &trails.back.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 0.0).to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.target_composition_pipelines.alpha);
            pass.set_bind_group(0, &history_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine trail source pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &trails.back.view,
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
            pass.set_pipeline(self.target_composition_pipelines.pipeline(source_blend));
            pass.set_bind_group(0, &source_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        std::mem::swap(&mut trails.front, &mut trails.back);
        render_report(
            RenderStatus::Drawn,
            Duration::ZERO,
            upload,
            Duration::ZERO,
            Duration::ZERO,
            encode_started_at.elapsed(),
            frame_started_at.elapsed(),
            false,
            true,
            TessellationStats::default(),
        )
    }

    fn draw_clear_trail_buffer(
        &self,
        trails: &TrailBuffer2d,
        color: Color,
        frame_started_at: Instant,
    ) -> RenderReport {
        let encode_started_at = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine trail clear encoder"),
            });
        for view in [&trails.front.view, &trails.back.view] {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine trail clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(premultiplied_wgpu_color(color)),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit([encoder.finish()]);
        render_report(
            RenderStatus::Drawn,
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            encode_started_at.elapsed(),
            frame_started_at.elapsed(),
            false,
            true,
            TessellationStats::default(),
        )
    }

    fn prepare_particle_draw(
        &self,
        field: &mut ParticleField2d,
        camera: Camera2d,
    ) -> Result<ParticleDrawPreparation, RendererFrameError> {
        let (logical_width, logical_height) = self.logical_size();
        let viewport = LogicalViewport::new(logical_width, logical_height)
            .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
        let Some(camera_uniform) = CameraUniform::new(camera, viewport) else {
            return Err(RendererFrameError::InvalidGeometryTransform);
        };
        let instance_count = field.instances.len();
        let visibility_checked = instance_count.min(field.budget.max_visibility_checks_per_frame);
        if visibility_checked < instance_count {
            field.visible_instances.clear();
            for candidate_index in 0..visibility_checked {
                let source_index =
                    uniformly_sampled_index(candidate_index, instance_count, visibility_checked);
                let instance = field.instances[source_index];
                if !instance.is_safe_for(camera_uniform) {
                    return Err(RendererFrameError::InvalidGeometryTransform);
                }
                if instance.intersects_viewport(camera_uniform, viewport) {
                    field.visible_instances.push(instance);
                }
            }
            let visible_count = field.visible_instances.len();
            let selected_count = visible_count.min(field.budget.instance_limit());
            if selected_count < visible_count {
                let mut visible_index = 0;
                field.visible_instances.retain(|_| {
                    let selected = particle_visible_index_is_selected(
                        visible_index,
                        visible_count,
                        selected_count,
                    );
                    visible_index += 1;
                    selected
                });
            }
            field.statistics = particle_statistics_with_budget(
                instance_count,
                visibility_checked,
                visible_count,
                selected_count,
                0,
            );
            return self.upload_particle_draw(field, camera_uniform, selected_count);
        }

        let visible_count = visible_particle_count(&field.instances, camera_uniform, viewport)?;
        let selected_count = visible_count.min(field.budget.instance_limit());
        field.statistics = particle_statistics_with_budget(
            instance_count,
            instance_count,
            visible_count,
            selected_count,
            0,
        );
        if selected_count == field.instances.len() {
            field.visible_instances.clear();
        } else {
            field.visible_instances.clear();
            let mut visible_index = 0;
            for instance in field.instances.iter().copied() {
                if !instance.intersects_viewport(camera_uniform, viewport) {
                    continue;
                }
                let selected_index = field.visible_instances.len();
                if selected_index < selected_count
                    && particle_visible_index_is_selected(
                        visible_index,
                        visible_count,
                        selected_count,
                    )
                {
                    field.visible_instances.push(instance);
                }
                visible_index += 1;
            }
        }
        self.upload_particle_draw(field, camera_uniform, selected_count)
    }

    fn upload_particle_draw(
        &self,
        field: &ParticleField2d,
        camera_uniform: CameraUniform,
        visible_count: usize,
    ) -> Result<ParticleDrawPreparation, RendererFrameError> {
        let visible_instances =
            if field.visible_instances.is_empty() && visible_count == field.instances.len() {
                field.instances.as_slice()
            } else {
                field.visible_instances.as_slice()
            };
        let upload_started_at = Instant::now();
        if !visible_instances.is_empty() {
            self.queue.write_buffer(
                &field.instance_buffer,
                0,
                bytemuck::cast_slice(visible_instances),
            );
        }
        let upload = upload_started_at.elapsed();
        let camera_uniform_upload_started_at = Instant::now();
        self.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        let camera_uniform_upload = camera_uniform_upload_started_at.elapsed();

        Ok(ParticleDrawPreparation {
            visible_count,
            upload,
            camera_uniform_upload,
        })
    }

    fn draw_particle_field(
        &mut self,
        background: Color,
        field: &mut ParticleField2d,
        camera: Camera2d,
        frame_started_at: Instant,
    ) -> Result<RenderReport, RendererFrameError> {
        let preparation = self.prepare_particle_draw(field, camera)?;

        let surface_acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    Duration::ZERO,
                    preparation.upload,
                    preparation.camera_uniform_upload,
                    surface_acquire_started_at.elapsed(),
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
                    preparation.upload,
                    preparation.camera_uniform_upload,
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
                    preparation.upload,
                    preparation.camera_uniform_upload,
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
            if preparation.visible_count > 0 {
                pass.set_pipeline(&self.particle_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.particle_unit_buffer.slice(..));
                pass.set_vertex_buffer(1, field.instance_buffer.slice(..));
                pass.draw(0..6, 0..preparation.visible_count as u32);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.notify_before_present();
        self.queue.present(surface_texture);
        field.statistics.rendered = preparation.visible_count;
        Ok(render_report(
            RenderStatus::Drawn,
            Duration::ZERO,
            preparation.upload,
            preparation.camera_uniform_upload,
            surface_acquire,
            encode_submit_present_started_at.elapsed(),
            frame_started_at.elapsed(),
            false,
            true,
            TessellationStats::default(),
        ))
    }

    fn draw_particle_field_to_target(
        &self,
        target: &RenderTarget2d,
        field: &mut ParticleField2d,
        camera: Camera2d,
        load: RenderTargetLoad,
        frame_started_at: Instant,
    ) -> Result<RenderReport, RendererFrameError> {
        let preparation = self.prepare_particle_draw(field, camera)?;
        let encode_started_at = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine target particle encoder"),
            });
        {
            let load = match load {
                RenderTargetLoad::Load => wgpu::LoadOp::Load,
                RenderTargetLoad::Clear(color) => {
                    wgpu::LoadOp::Clear(premultiplied_wgpu_color(color))
                }
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine target particle pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if preparation.visible_count > 0 {
                pass.set_pipeline(&self.target_particle_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.particle_unit_buffer.slice(..));
                pass.set_vertex_buffer(1, field.instance_buffer.slice(..));
                pass.draw(0..6, 0..preparation.visible_count as u32);
            }
        }
        self.queue.submit([encoder.finish()]);
        field.statistics.rendered = preparation.visible_count;
        Ok(render_report(
            RenderStatus::Drawn,
            Duration::ZERO,
            preparation.upload,
            preparation.camera_uniform_upload,
            Duration::ZERO,
            encode_started_at.elapsed(),
            frame_started_at.elapsed(),
            false,
            true,
            TessellationStats::default(),
        ))
    }

    fn ensure_vertex_capacity(&mut self, vertex_count: usize) -> Result<(), RendererFrameError> {
        if vertex_count <= self.vertex_capacity {
            return Ok(());
        }

        self.vertex_capacity = dynamic_vertex_capacity(vertex_count)
            .filter(|capacity| buffer_capacity_fits::<Vertex>(&self.device, *capacity))
            .ok_or(RendererFrameError::GeometryCapacityTooLarge)?;
        self.vertex_buffer = Arc::new(create_vertex_buffer(&self.device, self.vertex_capacity));
        Ok(())
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

    fn color_map_view(&mut self, color_map: &ColorMap) -> wgpu::TextureView {
        if let Some(cached) = self
            .color_map_cache
            .as_ref()
            .filter(|cached| cached.source == *color_map)
        {
            return cached.view.clone();
        }
        let cached = create_cached_color_map(&self.device, &self.queue, color_map);
        let view = cached.view.clone();
        self.color_map_cache = Some(cached);
        view
    }

    fn validate_render_target(&self, target: &RenderTarget2d) -> Result<(), RenderTargetError> {
        prepared_scene_belongs_to(&self.renderer_identity, &target.renderer_identity)
            .then_some(())
            .ok_or(RenderTargetError::RendererMismatch)
    }

    fn validate_trail_buffer(&self, trails: &TrailBuffer2d) -> Result<(), RenderTargetError> {
        prepared_scene_belongs_to(&self.renderer_identity, &trails.renderer_identity)
            .then_some(())
            .ok_or(RenderTargetError::RendererMismatch)
    }
}

fn invoke_pre_present_notify(
    present_mode: RendererSurfacePresentMode,
    notify: Option<&(dyn Fn() + Send + Sync)>,
) {
    if present_mode.is_refresh_synchronized()
        && let Some(notify) = notify
    {
        notify();
    }
}

fn prepared_scene_belongs_to(renderer_identity: &Arc<()>, scene_identity: &Arc<()>) -> bool {
    Arc::ptr_eq(renderer_identity, scene_identity)
}

fn render_target_allocation_bytes(
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Option<usize> {
    let bytes_per_texel = u64::from(format.block_copy_size(None)?);
    let byte_count = u64::from(width)
        .checked_mul(u64::from(height))?
        .checked_mul(bytes_per_texel)?;
    usize::try_from(byte_count).ok()
}

fn dynamic_vertices_to_gpu(
    vertices: &[DynamicVertex2d],
) -> Result<Vec<DynamicGpu>, DynamicMeshError> {
    validate_dynamic_vertices(vertices)?;
    let requested_bytes = dynamic_mesh_bytes(vertices.len())?;
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(vertices.len())
        .map_err(|_| DynamicMeshError::AllocationFailed { requested_bytes })?;
    converted.extend(vertices.iter().copied().map(dynamic_vertex_to_gpu));
    Ok(converted)
}

fn replace_dynamic_mesh_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &mut DynamicMesh2d,
    vertices: &[DynamicVertex2d],
) -> Result<DynamicMeshUpdateReport, DynamicMeshError> {
    let update_started_at = Instant::now();
    validate_dynamic_vertices(vertices)?;
    if let Some(budget) = mesh.budget {
        validate_dynamic_mesh_budget(budget, vertices.len())?;
    }
    let replacement_vertices = dynamic_vertices_to_gpu(vertices)?;
    let reallocated = vertices.len() > mesh.vertex_capacity;
    let replacement = if reallocated {
        let capacity = dynamic_vertex_capacity(vertices.len())
            .filter(|capacity| buffer_capacity_fits::<DynamicGpu>(device, *capacity))
            .ok_or(DynamicMeshError::CapacityTooLarge)?;
        let buffer = Arc::new(create_dynamic_vertex_buffer(device, capacity));
        Some((buffer, capacity))
    } else {
        None
    };
    let upload_buffer = replacement
        .as_ref()
        .map_or(mesh.vertex_buffer.as_ref(), |(buffer, _)| buffer.as_ref());
    if !replacement_vertices.is_empty() {
        queue.write_buffer(
            upload_buffer,
            0,
            bytemuck::cast_slice(&replacement_vertices),
        );
    }
    if let Some((buffer, capacity)) = replacement {
        mesh.vertex_buffer = buffer;
        mesh.vertex_capacity = capacity;
    }
    mesh.geometry_extents = GeometryExtents::from_dynamic_vertices(&replacement_vertices);
    mesh.vertices = replacement_vertices;
    Ok(DynamicMeshUpdateReport {
        vertex_count: mesh.vertices.len(),
        upload: update_started_at.elapsed(),
        reallocated,
    })
}

fn dynamic_mesh_bytes(vertex_count: usize) -> Result<usize, DynamicMeshError> {
    vertex_count
        .checked_mul(std::mem::size_of::<DynamicGpu>())
        .ok_or(DynamicMeshError::CapacityTooLarge)
}

fn validate_dynamic_mesh_budget(
    budget: DynamicMeshBudget,
    vertex_count: usize,
) -> Result<(), DynamicMeshError> {
    if vertex_count > budget.max_vertices {
        return Err(DynamicMeshError::BudgetExceeded {
            resource: DynamicMeshBudgetResource::Vertices,
            limit: budget.max_vertices,
            actual: vertex_count,
        });
    }
    let bytes = dynamic_mesh_bytes(vertex_count)?;
    if bytes > budget.max_retained_bytes {
        return Err(DynamicMeshError::BudgetExceeded {
            resource: DynamicMeshBudgetResource::RetainedBytes,
            limit: budget.max_retained_bytes,
            actual: bytes,
        });
    }
    if bytes > budget.max_upload_bytes {
        return Err(DynamicMeshError::BudgetExceeded {
            resource: DynamicMeshBudgetResource::UploadBytes,
            limit: budget.max_upload_bytes,
            actual: bytes,
        });
    }
    Ok(())
}

fn validate_dynamic_vertices(vertices: &[DynamicVertex2d]) -> Result<(), DynamicMeshError> {
    if !vertices.len().is_multiple_of(3) {
        return Err(DynamicMeshError::InvalidVertexCount);
    }
    // Private fields and the only public constructor make finiteness an
    // invariant of DynamicVertex2d. Keep the assertion in development without
    // rescanning an already-validated high-volume stream in release builds.
    debug_assert!(vertices.iter().all(|vertex| {
        vertex.world_position.is_finite()
            && vertex.depth.is_finite()
            && vertex.color.is_normalized()
    }));
    Ok(())
}

fn dynamic_vertex_to_gpu(vertex: DynamicVertex2d) -> DynamicGpu {
    DynamicGpu {
        world_position: [vertex.world_position.x, vertex.world_position.y],
        depth: vertex.depth,
        color: vertex.color.to_array(),
    }
}

fn dynamic_vertex_capacity(vertex_count: usize) -> Option<usize> {
    vertex_count.max(1).checked_next_power_of_two()
}

fn particle_instances_to_gpu(
    instances: &[ParticleInstance2d],
) -> Result<Vec<ParticleGpu>, ParticleFieldError> {
    instances
        .iter()
        .copied()
        .map(|instance| {
            let world_position = instance.world_position();
            let radius = instance.radius();
            let color = instance.color();
            let depth = instance.depth();
            if !world_position.is_finite()
                || !radius.is_finite()
                || radius <= 0.0
                || !color.is_normalized()
                || !depth.is_finite()
            {
                return Err(ParticleFieldError::InvalidInstance);
            }
            Ok(ParticleGpu {
                world_position: [world_position.x, world_position.y],
                depth,
                radius,
                color: color.to_array(),
            })
        })
        .collect()
}

fn particle_instance_capacity(instance_count: usize) -> Option<usize> {
    instance_count.max(1).checked_next_power_of_two()
}

fn particle_budgeted_capacity(
    instance_count: usize,
    budget: ParticleRenderBudget,
) -> Option<usize> {
    let limit = budget.instance_limit();
    let required = instance_count.min(limit).max(1);
    particle_instance_capacity(required).map(|capacity| capacity.min(limit))
}

fn buffer_allocation_bytes<T>(capacity: usize) -> Option<u64> {
    u64::try_from(capacity)
        .ok()?
        .checked_mul(std::mem::size_of::<T>() as u64)
}

fn buffer_capacity_fits<T>(device: &wgpu::Device, capacity: usize) -> bool {
    buffer_allocation_bytes::<T>(capacity)
        .is_some_and(|bytes| bytes <= device.limits().max_buffer_size)
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

#[cfg(test)]
fn visible_particle_instances(
    instances: &[ParticleGpu],
    camera: CameraUniform,
    viewport: LogicalViewport,
) -> Result<Vec<ParticleGpu>, RendererFrameError> {
    visible_particle_count(instances, camera, viewport)?;
    Ok(instances
        .iter()
        .copied()
        .filter(|instance| instance.intersects_viewport(camera, viewport))
        .collect())
}

fn visible_particle_count(
    instances: &[ParticleGpu],
    camera: CameraUniform,
    viewport: LogicalViewport,
) -> Result<usize, RendererFrameError> {
    let mut visible = 0;
    for instance in instances.iter().copied() {
        if !instance.is_safe_for(camera) {
            return Err(RendererFrameError::InvalidGeometryTransform);
        }
        visible += usize::from(instance.intersects_viewport(camera, viewport));
    }
    Ok(visible)
}

fn particle_visible_index_is_selected(
    visible_index: usize,
    visible_count: usize,
    selected_count: usize,
) -> bool {
    if visible_count == 0 || selected_count == 0 {
        return false;
    }
    let before = visible_index as u128 * selected_count as u128 / visible_count as u128;
    let after = (visible_index as u128 + 1) * selected_count as u128 / visible_count as u128;
    before < after
}

fn uniformly_sampled_index(sample_index: usize, source_count: usize, sample_count: usize) -> usize {
    debug_assert!(sample_index < sample_count);
    debug_assert!(sample_count <= source_count);
    (((sample_index as u128 + 1) * source_count as u128 - 1) / sample_count as u128) as usize
}

fn particle_statistics(
    instance_count: usize,
    visible: usize,
    rendered: usize,
) -> ParticleStatistics {
    particle_statistics_with_budget(instance_count, instance_count, visible, visible, rendered)
}

fn particle_statistics_with_budget(
    instance_count: usize,
    visibility_checked: usize,
    visible: usize,
    selected: usize,
    rendered: usize,
) -> ParticleStatistics {
    ParticleStatistics {
        submitted: instance_count,
        visibility_checked,
        visible,
        culled: visibility_checked.saturating_sub(visible),
        budget_limited: instance_count
            .saturating_sub(visibility_checked)
            .saturating_add(visible.saturating_sub(selected)),
        dropped: 0,
        rendered,
    }
}

fn prepare_scene_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    scene: &Scene,
) -> Result<PreparedScene, PreparedSceneError> {
    let mut vertices = Vec::new();
    let mut draw_batches = Vec::new();
    let tessellation = tessellate_scene(scene, &mut vertices, &mut draw_batches)?;
    let geometry_extents = GeometryExtents::from_vertices(&vertices);
    if !buffer_capacity_fits::<Vertex>(device, vertices.len().max(1)) {
        return Err(PreparedSceneError::CapacityTooLarge);
    }
    let vertex_buffer = Arc::new(create_vertex_buffer(device, vertices.len()));
    if !vertices.is_empty() {
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
    }
    let vertex_count = vertices.len();
    let vertices: Arc<[Vertex]> = vertices.into();

    Ok(PreparedScene {
        renderer_identity,
        background: scene.background(),
        vertex_buffer,
        vertices,
        command_count: scene.command_count(),
        vertex_count,
        geometry_extents,
        draw_batches,
        tessellation,
    })
}

fn restore_prepared_scene_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    source: &PreparedScene,
) -> Result<PreparedScene, PreparedSceneError> {
    if !buffer_capacity_fits::<Vertex>(device, source.vertices.len().max(1)) {
        return Err(PreparedSceneError::CapacityTooLarge);
    }
    let vertex_buffer = Arc::new(create_vertex_buffer(device, source.vertices.len()));
    if !source.vertices.is_empty() {
        queue.write_buffer(
            &vertex_buffer,
            0,
            bytemuck::cast_slice(source.vertices.as_ref()),
        );
    }

    Ok(PreparedScene {
        renderer_identity,
        background: source.background,
        vertex_buffer,
        vertices: Arc::clone(&source.vertices),
        command_count: source.command_count,
        vertex_count: source.vertex_count,
        geometry_extents: source.geometry_extents,
        draw_batches: source.draw_batches.clone(),
        tessellation: source.tessellation,
    })
}

fn restore_dynamic_mesh_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    source: &DynamicMesh2d,
) -> Result<DynamicMesh2d, DynamicMeshError> {
    if !buffer_capacity_fits::<DynamicGpu>(device, source.vertex_capacity) {
        return Err(DynamicMeshError::CapacityTooLarge);
    }
    if let Some(budget) = source.budget {
        validate_dynamic_mesh_budget(budget, source.vertices.len())?;
    }
    let requested_bytes = dynamic_mesh_bytes(source.vertices.len())?;
    let mut vertices = Vec::new();
    vertices
        .try_reserve_exact(source.vertices.len())
        .map_err(|_| DynamicMeshError::AllocationFailed { requested_bytes })?;
    vertices.extend_from_slice(&source.vertices);
    let vertex_buffer = Arc::new(create_dynamic_vertex_buffer(device, source.vertex_capacity));
    if !vertices.is_empty() {
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(vertices.as_slice()));
    }
    Ok(DynamicMesh2d {
        renderer_identity,
        vertex_buffer,
        vertices,
        vertex_capacity: source.vertex_capacity,
        geometry_extents: source.geometry_extents,
        budget: source.budget,
    })
}

fn restore_particle_field_resources(
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    source: &ParticleField2d,
) -> Result<ParticleField2d, ParticleFieldError> {
    if !buffer_capacity_fits::<ParticleGpu>(device, source.instance_capacity) {
        return Err(ParticleFieldError::CapacityTooLarge);
    }
    let instance_buffer = Arc::new(create_particle_instance_buffer(
        device,
        source.instance_capacity,
    ));
    Ok(ParticleField2d {
        renderer_identity,
        instance_buffer,
        instances: source.instances.clone(),
        visible_instances: Vec::new(),
        instance_capacity: source.instance_capacity,
        budget: source.budget,
        statistics: particle_statistics(source.instances.len(), source.instances.len(), 0),
    })
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

struct PipelineResources {
    pipeline: wgpu::RenderPipeline,
    target_pipeline: wgpu::RenderPipeline,
    dynamic_pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,
    target_particle_pipeline: wgpu::RenderPipeline,
    heatmap_pipeline: wgpu::RenderPipeline,
    target_heatmap_pipeline: wgpu::RenderPipeline,
    composition_pipelines: CompositionPipelines,
    target_composition_pipelines: CompositionPipelines,
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    heatmap_uniform_buffer: wgpu::Buffer,
    heatmap_bind_group_layout: wgpu::BindGroupLayout,
}

fn create_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> PipelineResources {
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

    let target_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine target shape pipeline"),
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
        multisample: wgpu::MultisampleState::default(),
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

    let dynamic_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine dynamic mesh pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("dynamic_vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(DynamicGpu::LAYOUT)],
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

    let target_particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine target particle pipeline"),
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
            count: 1,
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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

    let target_heatmap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine render-target heatmap pipeline"),
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
        multisample: wgpu::MultisampleState::default(),
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

    let composition_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine composition uniform buffer"),
        size: std::mem::size_of::<CompositeUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let composition_secondary_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine composition secondary uniform buffer"),
        size: std::mem::size_of::<CompositeUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let composition_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine composition bind group layout"),
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let composition_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-engine composition pipeline layout"),
            bind_group_layouts: &[Some(&composition_bind_group_layout)],
            immediate_size: 0,
        });
    let composition_pipelines = CompositionPipelines {
        alpha: create_composition_pipeline(
            device,
            &shader,
            &composition_pipeline_layout,
            format,
            sample_count,
            BlendMode::Alpha,
        ),
        additive: create_composition_pipeline(
            device,
            &shader,
            &composition_pipeline_layout,
            format,
            sample_count,
            BlendMode::Additive,
        ),
        replace: create_composition_pipeline(
            device,
            &shader,
            &composition_pipeline_layout,
            format,
            sample_count,
            BlendMode::Replace,
        ),
        uniform_buffer: composition_uniform_buffer,
        secondary_uniform_buffer: composition_secondary_uniform_buffer,
        bind_group_layout: composition_bind_group_layout,
    };
    let target_composition_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine target composition uniform buffer"),
        size: std::mem::size_of::<CompositeUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let target_composition_secondary_uniform_buffer =
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine target composition secondary uniform buffer"),
            size: std::mem::size_of::<CompositeUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    let target_composition_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine target composition bind group layout"),
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let target_composition_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-engine target composition pipeline layout"),
            bind_group_layouts: &[Some(&target_composition_bind_group_layout)],
            immediate_size: 0,
        });
    let target_composition_pipelines = CompositionPipelines {
        alpha: create_composition_pipeline(
            device,
            &shader,
            &target_composition_pipeline_layout,
            format,
            1,
            BlendMode::Alpha,
        ),
        additive: create_composition_pipeline(
            device,
            &shader,
            &target_composition_pipeline_layout,
            format,
            1,
            BlendMode::Additive,
        ),
        replace: create_composition_pipeline(
            device,
            &shader,
            &target_composition_pipeline_layout,
            format,
            1,
            BlendMode::Replace,
        ),
        uniform_buffer: target_composition_uniform_buffer,
        secondary_uniform_buffer: target_composition_secondary_uniform_buffer,
        bind_group_layout: target_composition_bind_group_layout,
    };

    PipelineResources {
        pipeline,
        target_pipeline,
        dynamic_pipeline,
        particle_pipeline,
        target_particle_pipeline,
        heatmap_pipeline,
        target_heatmap_pipeline,
        composition_pipelines,
        target_composition_pipelines,
        camera_uniform_buffer,
        camera_bind_group,
        camera_bind_group_layout,
        heatmap_uniform_buffer,
        heatmap_bind_group_layout,
    }
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
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn create_dynamic_vertex_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine dynamic vertex buffer"),
        size: (capacity.max(1) * std::mem::size_of::<DynamicGpu>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
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

fn scalar_value_range_extent(minimum: f32, maximum: f32) -> Option<f32> {
    if !minimum.is_finite() || !maximum.is_finite() || maximum <= minimum {
        return None;
    }
    let extent = maximum - minimum;
    (extent.is_finite() && extent > 0.0).then_some(extent)
}

fn premultiplied_wgpu_color(color: Color) -> wgpu::Color {
    let [red, green, blue, alpha] = color.to_array();
    wgpu::Color {
        r: f64::from(red * alpha),
        g: f64::from(green * alpha),
        b: f64::from(blue * alpha),
        a: f64::from(alpha),
    }
}

fn create_scalar_field_texture(
    device: &wgpu::Device,
    field: &ScalarField,
) -> Result<wgpu::Texture, ScalarFieldTextureError> {
    let extent = scalar_field_extent(field)?;
    let limits = device.limits();
    if extent.width > limits.max_texture_dimension_2d
        || extent.height > limits.max_texture_dimension_2d
    {
        return Err(ScalarFieldTextureError::DimensionsTooLarge);
    }
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

fn create_cached_color_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_map: &ColorMap,
) -> CachedColorMap {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine color map texture"),
        size: wgpu::Extent3d {
            width: COLOR_MAP_LUT_SIZE,
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
    let mut bytes = Vec::with_capacity(COLOR_MAP_LUT_SIZE as usize * 4);
    for index in 0..COLOR_MAP_LUT_SIZE {
        let color = color_map.sample_normalized(index as f32 / (COLOR_MAP_LUT_SIZE - 1) as f32);
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
            width: COLOR_MAP_LUT_SIZE,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    CachedColorMap {
        source: color_map.clone(),
        _texture: texture,
        view,
    }
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

mod config;
mod frame;
mod glyph;
mod image;
mod mesh3d;
mod tessellation;
mod visualization;
use config::{
    MultisampleTarget, create_multisample_target, logical_to_physical_screen,
    physical_to_logical_screen, validate_scale_factor,
};
pub use config::{RendererPresentMode, RendererSurfacePresentMode, WgpuRendererOptions};
pub use frame::{
    FrameBudget, FrameBudgetResource, FrameComposer, FrameComposerError, FramePassOptions,
    FrameReport, FrameSourceKind, FrameSourceStatistics, FrameStatistics,
};
pub use glyph::{
    GlyphAtlas2d, GlyphAtlasBudget, GlyphAtlasEntry, GlyphError, GlyphId, GlyphRun2d,
    GlyphRunBounds, GlyphRunBudget, GlyphRunStatistics, GlyphUploadReport, PositionedGlyph2d,
};
pub use image::{
    Image2d, ImageBatch2d, ImageBatchBudget, ImageBudget, ImageError, ImageSampling, ImageSprite2d,
    ImageTexelRect, ImageUploadReport,
};
use image::{ImageRenderer, ImageUniform};
use mesh3d::Mesh3dRenderer;
pub use mesh3d::{
    Mesh3dInstance, Mesh3dRenderError, Mesh3dRenderReport, Mesh3dResourceError, Object3dId,
    RenderTarget3d, RetainedMesh3d, Scene3d, Scene3dError, Scene3dRestoreReport,
};
use tessellation::{
    logical_viewport_scissor, offset_scissor, screen_clip_to_scissor, tessellate_scene,
};
use visualization::{CachedColorMap, CompositionPipelines, create_composition_pipeline};
pub use visualization::{LayeredVisualizationError, LayeredVisualizationOptions};

#[cfg(test)]
use tessellation::world_vertex;

#[cfg(test)]
mod tests;
