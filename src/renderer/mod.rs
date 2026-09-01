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
// Stay inside the WGSL division accuracy domain and reject denormal geometric
// operands. This deliberately narrow transform envelope is shared by every GPU path.
const MAX_PORTABLE_SHADER_VALUE: f32 = f32::from_bits((247_u32) << 23); // 2^120

fn is_portable_shader_source(value: f32) -> bool {
    value.is_finite()
        && value.abs() <= MAX_PORTABLE_SHADER_VALUE
        && (value == 0.0 || value.abs() >= f32::MIN_POSITIVE)
}

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
    /// Clear the target to normalized finite straight-linear RGBA before drawing.
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
    vertex_count: usize,
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
    camera_uniform: CameraUniform,
    visible_count: usize,
    statistics: ParticleStatistics,
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

    #[cfg(test)]
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

    fn is_safe_for(self, camera: CameraUniform, viewport: LogicalViewport) -> bool {
        self.validated_viewport_intersection(camera, viewport)
            .is_some()
    }

    fn transform_sources_are_portable(self) -> bool {
        self.world_position
            .into_iter()
            .chain([self.depth, self.radius])
            .all(is_portable_shader_source)
    }

    fn projected_screen_bounds(self, camera: CameraUniform) -> Option<((f64, f64), (f64, f64))> {
        let center = self.projected_screen_center_bounds(camera)?;
        let radius = f64::from(self.radius);
        Some((
            shader_interval_sum_range([center.0, (-radius, radius)])?,
            shader_interval_sum_range([center.1, (-radius, radius)])?,
        ))
    }

    fn projected_screen_center_bounds(
        self,
        camera: CameraUniform,
    ) -> Option<((f64, f64), (f64, f64))> {
        let relative_x = shader_relative_component_bounds(
            self.world_position[0],
            self.world_position[0],
            camera.camera_center[0],
            0.0,
            0.0,
        )?;
        let relative_y = shader_relative_component_bounds(
            self.world_position[1],
            self.world_position[1],
            camera.camera_center[1],
            0.0,
            0.0,
        )?;
        let relative_minimum = [relative_x.0, relative_y.0];
        let relative_maximum = [relative_x.1, relative_y.1];
        Some((
            shader_world_dot_range(
                camera.world_to_screen_x,
                relative_minimum,
                relative_maximum,
                self.depth,
                self.depth,
            )?,
            shader_world_dot_range(
                camera.world_to_screen_y,
                relative_minimum,
                relative_maximum,
                self.depth,
                self.depth,
            )?,
        ))
    }

    fn screen_bounds_intersect_viewport(
        bounds: ((f64, f64), (f64, f64)),
        viewport: LogicalViewport,
    ) -> bool {
        bounds.0.1 >= 0.0
            && bounds.0.0 <= f64::from(viewport.width())
            && bounds.1.1 >= 0.0
            && bounds.1.0 <= f64::from(viewport.height())
    }

    fn validated_viewport_intersection(
        self,
        camera: CameraUniform,
        viewport: LogicalViewport,
    ) -> Option<bool> {
        if !self.transform_sources_are_portable() || !camera.sources_are_portable() {
            return None;
        }
        let center = self.projected_screen_center_bounds(camera)?;
        if !self.clip_expansion_is_stable(center, camera) {
            return None;
        }
        let radius = f64::from(self.radius);
        let width = f64::from(viewport.width());
        let height = f64::from(viewport.height());
        let wholly_outside = center.0.1 + radius < 0.0
            || center.0.0 - radius > width
            || center.1.1 + radius < 0.0
            || center.1.0 - radius > height;
        if wholly_outside {
            return Some(false);
        }
        // Every permitted dot/FMA association must leave a positive-area
        // portion of the quad inside the viewport. If visibility itself can
        // change between legal shader lowerings, fail closed before upload.
        let always_intersects = center.0.0 + radius > 0.0
            && center.0.1 - radius < width
            && center.1.0 + radius > 0.0
            && center.1.1 - radius < height;
        always_intersects.then_some(true)
    }

    fn clip_expansion_is_stable(
        self,
        center: ((f64, f64), (f64, f64)),
        camera: CameraUniform,
    ) -> bool {
        (0..2).all(|axis| {
            let center_clip = rounded_f32_product_range(
                center_axis(center, axis),
                (
                    f64::from(camera.screen_to_clip[axis]),
                    f64::from(camera.screen_to_clip[axis]),
                ),
            )
            .and_then(|scaled| {
                rounded_f32_add_range(
                    scaled,
                    (
                        f64::from(camera.screen_to_clip[axis + 2]),
                        f64::from(camera.screen_to_clip[axis + 2]),
                    ),
                    false,
                )
            });
            let extent = rounded_f32_product_range(
                (f64::from(self.radius), f64::from(self.radius)),
                (
                    f64::from(camera.screen_to_clip[axis].abs()),
                    f64::from(camera.screen_to_clip[axis].abs()),
                ),
            );
            let (Some(center_clip), Some(extent)) = (center_clip, extent) else {
                return false;
            };
            if extent.0 < f64::from(f32::MIN_POSITIVE) {
                return false;
            }
            let Some(negative) = rounded_f32_add_range(center_clip, extent, true) else {
                return false;
            };
            let Some(positive) = rounded_f32_add_range(center_clip, extent, false) else {
                return false;
            };
            positive.0 - negative.1 >= f64::from(f32::MIN_POSITIVE)
        })
    }

    fn intersects_viewport(self, camera: CameraUniform, viewport: LogicalViewport) -> bool {
        // Validation precedes production culling. If this helper is called on
        // an invalid instance, keep it conservatively instead of silently
        // dropping geometry based on one CPU association of the shader dot.
        self.projected_screen_bounds(camera)
            .is_none_or(|bounds| Self::screen_bounds_intersect_viewport(bounds, viewport))
    }
}

fn center_axis(center: ((f64, f64), (f64, f64)), axis: usize) -> (f64, f64) {
    if axis == 0 { center.0 } else { center.1 }
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
        (uniform.is_finite() && uniform.sources_are_portable()).then_some(uniform)
    }

    fn sources_are_portable(self) -> bool {
        self.camera_center
            .into_iter()
            .chain(self.world_to_screen_x)
            .chain(self.world_to_screen_y)
            .chain(self.screen_to_clip)
            .all(is_portable_shader_source)
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
            vertex_count: 0,
            empty: is_empty,
        }
    }

    fn include(&mut self, vertex: Vertex) {
        self.vertex_count += 1;
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
        self.vertex_count += 1;
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
        // Match the vertex shader's split-anchor operation order. Generated
        // local offsets are projected separately so they cannot disappear
        // when their world anchor is much larger than the local geometry.
        let Some(relative_horizontal) = shader_relative_component_bounds(
            self.world_min.x,
            self.world_max.x,
            center.x,
            0.0,
            0.0,
        ) else {
            return false;
        };
        let Some(relative_vertical) = shader_relative_component_bounds(
            self.world_min.y,
            self.world_max.y,
            center.y,
            0.0,
            0.0,
        ) else {
            return false;
        };
        let relative_minimum = [relative_horizontal.0, relative_vertical.0];
        let relative_maximum = [relative_horizontal.1, relative_vertical.1];
        let Some(projected_world_horizontal) = shader_world_dot_range(
            uniform.world_to_screen_x,
            relative_minimum,
            relative_maximum,
            self.depth_min,
            self.depth_max,
        ) else {
            return false;
        };
        let Some(projected_world_vertical) = shader_world_dot_range(
            uniform.world_to_screen_y,
            relative_minimum,
            relative_maximum,
            self.depth_min,
            self.depth_max,
        ) else {
            return false;
        };
        let Some(projected_offset_horizontal) = shader_direction_dot_range(
            uniform.world_to_screen_x,
            self.world_offset_min,
            self.world_offset_max,
        ) else {
            return false;
        };
        let Some(projected_offset_vertical) = shader_direction_dot_range(
            uniform.world_to_screen_y,
            self.world_offset_min,
            self.world_offset_max,
        ) else {
            return false;
        };
        let Some(world_horizontal) =
            shader_interval_sum_range([projected_world_horizontal, projected_offset_horizontal])
        else {
            return false;
        };
        let Some(world_vertical) =
            shader_interval_sum_range([projected_world_vertical, projected_offset_vertical])
        else {
            return false;
        };
        let Some(direction_horizontal) = shader_direction_dot_range(
            uniform.world_to_screen_x,
            self.direction_min,
            self.direction_max,
        ) else {
            return false;
        };
        let Some(direction_vertical) = shader_direction_dot_range(
            uniform.world_to_screen_y,
            self.direction_min,
            self.direction_max,
        ) else {
            return false;
        };
        if [
            self.screen_offset_max_abs.x,
            self.screen_offset_max_abs.y,
            self.normal_distance_max_abs,
            self.tangent_distance_max_abs,
        ]
        .into_iter()
        .any(is_nonzero_subnormal)
        {
            return false;
        }
        let Some(horizontal_bounds) = shader_stroke_screen_bounds(
            world_horizontal,
            self.screen_offset_max_abs.x,
            self.normal_distance_max_abs,
            self.tangent_distance_max_abs,
            self.miter_limit_max,
        ) else {
            return false;
        };
        let Some(vertical_bounds) = shader_stroke_screen_bounds(
            world_vertical,
            self.screen_offset_max_abs.y,
            self.normal_distance_max_abs,
            self.tangent_distance_max_abs,
            self.miter_limit_max,
        ) else {
            return false;
        };

        if !shader_clip_interval_is_safe(
            horizontal_bounds.0,
            horizontal_bounds.1,
            uniform.screen_to_clip[0],
            uniform.screen_to_clip[2],
        ) || !shader_clip_interval_is_safe(
            vertical_bounds.0,
            vertical_bounds.1,
            uniform.screen_to_clip[1],
            uniform.screen_to_clip[3],
        ) {
            return false;
        }

        [
            world_horizontal.0,
            world_horizontal.1,
            world_vertical.0,
            world_vertical.1,
            direction_horizontal.0,
            direction_horizontal.1,
            direction_vertical.0,
            direction_vertical.1,
            horizontal_bounds.0,
            horizontal_bounds.1,
            vertical_bounds.0,
            vertical_bounds.1,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value.abs() <= f32::MAX as f64)
    }
}

fn shader_stroke_screen_bounds(
    world_screen: (f64, f64),
    screen_offset_max_abs: f32,
    normal_distance_max_abs: f32,
    tangent_distance_max_abs: f32,
    miter_limit_max: f32,
) -> Option<(f64, f64)> {
    // `safe_unit` and the miter path are backend-selected inverse-square-root
    // arithmetic. A component of a mathematically unit vector is at most one;
    // two is a deliberately loose portable bound that also contains its
    // permitted approximation error. Propagate the shader's two additions
    // separately so a tiny later clip scale cannot hide screen-space overflow.
    const UNIT_COMPONENT_BOUND: f64 = 2.0;
    let normal = f64::from(normal_distance_max_abs);
    let tangent = f64::from(tangent_distance_max_abs);
    let miter = normal * f64::from(miter_limit_max.max(1.0));
    let normal_component = UNIT_COMPONENT_BOUND * normal.max(miter);
    let tangent_component = UNIT_COMPONENT_BOUND * tangent;
    let extrusion = shader_interval_sum_range([
        (-normal_component, normal_component),
        (-tangent_component, tangent_component),
    ])?;
    let offset = f64::from(screen_offset_max_abs);
    shader_interval_sum_range([world_screen, extrusion, (-offset, offset)])
}

#[derive(Clone, Copy)]
enum GeometryValidationSource<'a> {
    Tessellated(&'a [Vertex]),
    Dynamic(&'a [DynamicGpu]),
}

fn geometry_is_safe_for(
    extents: GeometryExtents,
    source: GeometryValidationSource<'_>,
    uniform: CameraUniform,
) -> bool {
    geometry_sources_are_portable(source)
        && match source {
            GeometryValidationSource::Tessellated(vertices) => {
                geometry_vertex_centers_are_portable(source, uniform)
                    && tessellated_triangle_topology_is_portable(vertices, uniform)
                    && vertices
                        .iter()
                        .all(|vertex| logical_stroke_branches_are_stable(*vertex, uniform))
            }
            GeometryValidationSource::Dynamic(vertices) => {
                // The triangle proof projects every vertex itself, so do not
                // duplicate the same per-vertex dot envelopes first.
                dynamic_triangle_topology_is_portable(vertices, uniform)
            }
        }
        && uniform.sources_are_portable()
        && extents.is_safe_for(uniform)
}

fn dynamic_triangle_topology_is_portable(vertices: &[DynamicGpu], uniform: CameraUniform) -> bool {
    if !vertices.len().is_multiple_of(3) {
        return false;
    }
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    for triangle in vertices.chunks_exact(3) {
        let Some(first) = dynamic_vertex_clip_ranges(triangle[0], uniform) else {
            return false;
        };
        let Some(second) = dynamic_vertex_clip_ranges(triangle[1], uniform) else {
            return false;
        };
        let Some(third) = dynamic_vertex_clip_ranges(triangle[2], uniform) else {
            return false;
        };
        let clip = [first, second, third];
        if clip_triangle_is_wholly_outside(clip) {
            continue;
        }
        if !clip_triangle_is_wholly_inside(clip) {
            // v0.2 does not yet carry conservative interval polygons through
            // hardware clipping. Reject partially clipped dynamic triangles
            // rather than permit backend-selected topology.
            return false;
        }
        if !clip_triangle_has_stable_signed_area(clip, minimum_normal) {
            return false;
        }
    }
    true
}

fn tessellated_triangle_topology_is_portable(vertices: &[Vertex], uniform: CameraUniform) -> bool {
    if !vertices.len().is_multiple_of(3) {
        return false;
    }
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    for triangle in vertices.chunks_exact(3) {
        let has_shader_extrusion = !triangle
            .iter()
            .all(|vertex| vertex.normal_distance == 0.0 && vertex.tangent_distance == 0.0);
        if has_shader_extrusion {
            let Some(first) = logical_stroke_vertex_screen_ranges(triangle[0], uniform) else {
                return false;
            };
            let Some(second) = logical_stroke_vertex_screen_ranges(triangle[1], uniform) else {
                return false;
            };
            let Some(third) = logical_stroke_vertex_screen_ranges(triangle[2], uniform) else {
                return false;
            };
            if [first, second, third].iter().all(|output| output.1)
                && triangle_position_sources_equal(triangle)
            {
                continue;
            }
            let Some(first) = screen_ranges_to_clip(first.0, uniform) else {
                return false;
            };
            let Some(second) = screen_ranges_to_clip(second.0, uniform) else {
                return false;
            };
            let Some(third) = screen_ranges_to_clip(third.0, uniform) else {
                return false;
            };
            let clip = [first, second, third];
            if !clip_triangle_is_wholly_outside(clip)
                && !clip_triangle_has_stable_signed_area(clip, minimum_normal)
            {
                return false;
            }
            continue;
        }
        let Some(first) = tessellated_vertex_clip_ranges(triangle[0], uniform) else {
            return false;
        };
        let Some(second) = tessellated_vertex_clip_ranges(triangle[1], uniform) else {
            return false;
        };
        let Some(third) = tessellated_vertex_clip_ranges(triangle[2], uniform) else {
            return false;
        };
        let clip = [first, second, third];
        if clip_triangle_is_wholly_outside(clip) {
            continue;
        }
        if !clip_triangle_has_stable_signed_area(clip, minimum_normal) {
            return false;
        }
    }
    true
}

fn clip_triangle_is_wholly_outside(clip: [[(f64, f64); 2]; 3]) -> bool {
    (0..2).any(|axis| {
        clip.iter().all(|point| point[axis].1 < -1.0)
            || clip.iter().all(|point| point[axis].0 > 1.0)
    })
}

fn clip_triangle_is_wholly_inside(clip: [[(f64, f64); 2]; 3]) -> bool {
    clip.iter()
        .all(|point| point.iter().all(|axis| axis.0 >= -1.0 && axis.1 <= 1.0))
}

fn clip_triangle_has_stable_signed_area(
    clip: [[(f64, f64); 2]; 3],
    minimum_magnitude: f64,
) -> bool {
    let Some(first_x) = shader_interval_difference(clip[1][0], clip[0][0]) else {
        return false;
    };
    let Some(first_y) = shader_interval_difference(clip[2][1], clip[0][1]) else {
        return false;
    };
    let Some(second_y) = shader_interval_difference(clip[1][1], clip[0][1]) else {
        return false;
    };
    let Some(second_x) = shader_interval_difference(clip[2][0], clip[0][0]) else {
        return false;
    };
    let Some(positive) = shader_interval_product(first_x, first_y) else {
        return false;
    };
    let Some(negative) = shader_interval_product(second_y, second_x) else {
        return false;
    };
    let Some(area) = shader_interval_difference(positive, negative) else {
        return false;
    };
    area.0 >= minimum_magnitude || area.1 <= -minimum_magnitude
}

fn dynamic_vertex_clip_ranges(
    vertex: DynamicGpu,
    uniform: CameraUniform,
) -> Option<[(f64, f64); 2]> {
    let relative_x = shader_relative_component_bounds(
        vertex.world_position[0],
        vertex.world_position[0],
        uniform.camera_center[0],
        0.0,
        0.0,
    )?;
    let relative_y = shader_relative_component_bounds(
        vertex.world_position[1],
        vertex.world_position[1],
        uniform.camera_center[1],
        0.0,
        0.0,
    )?;
    let minimum = [relative_x.0, relative_y.0];
    let maximum = [relative_x.1, relative_y.1];
    let screen = [
        shader_world_dot_range(
            uniform.world_to_screen_x,
            minimum,
            maximum,
            vertex.depth,
            vertex.depth,
        )?,
        shader_world_dot_range(
            uniform.world_to_screen_y,
            minimum,
            maximum,
            vertex.depth,
            vertex.depth,
        )?,
    ];
    screen_ranges_to_clip(screen, uniform)
}

fn tessellated_vertex_clip_ranges(
    vertex: Vertex,
    uniform: CameraUniform,
) -> Option<[(f64, f64); 2]> {
    screen_ranges_to_clip(tessellated_vertex_screen_ranges(vertex, uniform)?, uniform)
}

fn tessellated_vertex_screen_ranges(
    vertex: Vertex,
    uniform: CameraUniform,
) -> Option<[(f64, f64); 2]> {
    let base = tessellated_vertex_base_screen_ranges(vertex, uniform)?;
    Some([
        rounded_f32_add_range(
            base[0],
            (
                f64::from(vertex.screen_offset[0]),
                f64::from(vertex.screen_offset[0]),
            ),
            false,
        )?,
        rounded_f32_add_range(
            base[1],
            (
                f64::from(vertex.screen_offset[1]),
                f64::from(vertex.screen_offset[1]),
            ),
            false,
        )?,
    ])
}

fn tessellated_vertex_base_screen_ranges(
    vertex: Vertex,
    uniform: CameraUniform,
) -> Option<[(f64, f64); 2]> {
    let relative_x = shader_relative_component_bounds(
        vertex.world_position[0],
        vertex.world_position[0],
        uniform.camera_center[0],
        0.0,
        0.0,
    )?;
    let relative_y = shader_relative_component_bounds(
        vertex.world_position[1],
        vertex.world_position[1],
        uniform.camera_center[1],
        0.0,
        0.0,
    )?;
    let minimum = [relative_x.0, relative_y.0];
    let maximum = [relative_x.1, relative_y.1];
    let world_offset = Vec2::new(vertex.world_offset[0], vertex.world_offset[1]);
    Some([
        shader_interval_sum_range([
            shader_world_dot_range(
                uniform.world_to_screen_x,
                minimum,
                maximum,
                vertex.depth,
                vertex.depth,
            )?,
            shader_direction_dot_range(uniform.world_to_screen_x, world_offset, world_offset)?,
        ])?,
        shader_interval_sum_range([
            shader_world_dot_range(
                uniform.world_to_screen_y,
                minimum,
                maximum,
                vertex.depth,
                vertex.depth,
            )?,
            shader_direction_dot_range(uniform.world_to_screen_y, world_offset, world_offset)?,
        ])?,
    ])
}

fn triangle_position_sources_equal(triangle: &[Vertex]) -> bool {
    triangle.windows(2).all(|pair| {
        pair[0].world_position == pair[1].world_position
            && pair[0].depth == pair[1].depth
            && pair[0].world_offset == pair[1].world_offset
            && pair[0].screen_offset == pair[1].screen_offset
    })
}

fn logical_stroke_vertex_screen_ranges(
    vertex: Vertex,
    uniform: CameraUniform,
) -> Option<([(f64, f64); 2], bool)> {
    let base = tessellated_vertex_base_screen_ranges(vertex, uniform)?;
    let previous_source = Vec2::new(vertex.previous_direction[0], vertex.previous_direction[1]);
    let next_source = Vec2::new(vertex.next_direction[0], vertex.next_direction[1]);
    let previous_projected = [
        shader_direction_dot_range(uniform.world_to_screen_x, previous_source, previous_source)?,
        shader_direction_dot_range(uniform.world_to_screen_y, previous_source, previous_source)?,
    ];
    let next_projected = if vertex.previous_direction == vertex.next_direction {
        previous_projected
    } else {
        [
            shader_direction_dot_range(uniform.world_to_screen_x, next_source, next_source)?,
            shader_direction_dot_range(uniform.world_to_screen_y, next_source, next_source)?,
        ]
    };
    let previous_tangent = stroke_safe_unit_range(previous_projected)?;
    let next_tangent = if vertex.previous_direction == vertex.next_direction {
        previous_tangent
    } else {
        stroke_safe_unit_range(next_projected)?
    };
    let previous_normal = [range_negate(previous_tangent[1]), previous_tangent[0]];
    let next_normal = [range_negate(next_tangent[1]), next_tangent[0]];
    let turn = if vertex.previous_direction == vertex.next_direction {
        (0.0, 0.0)
    } else {
        rounded_f32_add_range(
            rounded_f32_product_range(previous_tangent[0], next_tangent[1])?,
            rounded_f32_product_range(previous_tangent[1], next_tangent[0])?,
            true,
        )?
    };
    let turn_sign = if turn.0 > 0.0 {
        1.0
    } else if turn.1 < 0.0 {
        -1.0
    } else if vertex.previous_direction == vertex.next_direction {
        0.0
    } else {
        return None;
    };
    let combined_normal = [
        rounded_f32_add_range(previous_normal[0], next_normal[0], false)?,
        rounded_f32_add_range(previous_normal[1], next_normal[1], false)?,
    ];
    let miter = stroke_safe_unit_range(combined_normal);
    let miter_state = miter.and_then(|miter| {
        let denominator = range_dot(miter, next_normal)?;
        if denominator.0 <= 0.001 && denominator.1 >= -0.001 {
            return None;
        }
        let reciprocal = rounded_f32_division_range((1.0, 1.0), denominator)?;
        let multiple = range_abs(reciprocal);
        let limit = f64::from(vertex.miter_limit);
        let within_limit = if multiple.1 <= limit {
            true
        } else if multiple.0 > limit {
            false
        } else {
            return None;
        };
        let scalar = rounded_f32_product_range(
            reciprocal,
            (
                f64::from(vertex.normal_distance),
                f64::from(vertex.normal_distance),
            ),
        )?;
        Some((range_vector_scale(miter, scalar)?, within_limit))
    });
    let normal_scalar = (
        f64::from(vertex.normal_distance),
        f64::from(vertex.normal_distance),
    );
    let tangent_scalar = (
        f64::from(vertex.tangent_distance),
        f64::from(vertex.tangent_distance),
    );
    let next_normal_offset = range_vector_scale(next_normal, normal_scalar)?;
    let previous_normal_offset = range_vector_scale(previous_normal, normal_scalar)?;
    let tangent_offset = range_vector_scale(next_tangent, tangent_scalar)?;
    let mut inactive_candidate = false;
    let mut extrusion = range_vector_add(next_normal_offset, tangent_offset)?;
    if (1.0..=3.0).contains(&vertex.stroke_role) {
        if turn_sign == 0.0 {
            extrusion = next_normal_offset;
        } else {
            let side = vertex.normal_distance.signum();
            let outer_side = -turn_sign;
            if side * outer_side <= 0.0 {
                extrusion = miter_state
                    .filter(|state| state.1)
                    .map_or([(0.0, 0.0); 2], |state| state.0);
            } else if vertex.stroke_role == 2.0 && miter_state.is_some_and(|state| state.1) {
                extrusion = miter_state?.0;
            } else if vertex.stroke_parameter < 0.0 {
                extrusion = previous_normal_offset;
            } else {
                extrusion = next_normal_offset;
            }
        }
        extrusion = range_vector_add(extrusion, tangent_offset)?;
    } else if vertex.stroke_role >= 4.0 {
        let inner = matches!(vertex.stroke_role as i32, 5 | 7 | 9);
        let side = vertex.normal_distance.signum();
        let candidate_side = if inner { -side } else { side };
        let mut active = turn_sign != 0.0 && candidate_side * -turn_sign > 0.0;
        if matches!(vertex.stroke_role as i32, 6 | 7) {
            active &= miter_state.is_none_or(|state| !state.1);
        }
        if !active {
            inactive_candidate = true;
            extrusion = [(0.0, 0.0); 2];
        } else if inner {
            extrusion = miter_state
                .filter(|state| state.1)
                .map_or([(0.0, 0.0); 2], |state| state.0);
        } else if vertex.stroke_role == 8.0 {
            let side_range = (f64::from(candidate_side), f64::from(candidate_side));
            let start = range_vector_scale(previous_normal, side_range)?;
            let finish = range_vector_scale(next_normal, side_range)?;
            let amount = f64::from(vertex.stroke_parameter);
            let mixed = range_vector_add(
                range_vector_scale(start, (1.0 - amount, 1.0 - amount))?,
                range_vector_scale(finish, (amount, amount))?,
            )?;
            extrusion = range_vector_scale(
                stroke_safe_unit_range(mixed)?,
                (
                    f64::from(vertex.normal_distance.abs()),
                    f64::from(vertex.normal_distance.abs()),
                ),
            )?;
        } else if vertex.stroke_parameter < 0.0 {
            extrusion = previous_normal_offset;
        } else {
            extrusion = next_normal_offset;
        }
    } else if miter_state.is_some_and(|state| state.1) {
        extrusion = range_vector_add(miter_state?.0, tangent_offset)?;
    }
    let screen = range_vector_add(base, extrusion)?;
    let screen_offset = [
        (
            f64::from(vertex.screen_offset[0]),
            f64::from(vertex.screen_offset[0]),
        ),
        (
            f64::from(vertex.screen_offset[1]),
            f64::from(vertex.screen_offset[1]),
        ),
    ];
    Some((range_vector_add(screen, screen_offset)?, inactive_candidate))
}

fn range_vector_add(left: [(f64, f64); 2], right: [(f64, f64); 2]) -> Option<[(f64, f64); 2]> {
    Some([
        rounded_f32_add_range(left[0], right[0], false)?,
        rounded_f32_add_range(left[1], right[1], false)?,
    ])
}

fn range_vector_scale(vector: [(f64, f64); 2], scalar: (f64, f64)) -> Option<[(f64, f64); 2]> {
    Some([
        rounded_f32_product_range(vector[0], scalar)?,
        rounded_f32_product_range(vector[1], scalar)?,
    ])
}

fn range_dot(left: [(f64, f64); 2], right: [(f64, f64); 2]) -> Option<(f64, f64)> {
    rounded_f32_add_range(
        rounded_f32_product_range(left[0], right[0])?,
        rounded_f32_product_range(left[1], right[1])?,
        false,
    )
}

fn range_negate(value: (f64, f64)) -> (f64, f64) {
    (-value.1, -value.0)
}

fn range_abs(value: (f64, f64)) -> (f64, f64) {
    if value.0 <= 0.0 && value.1 >= 0.0 {
        (0.0, value.0.abs().max(value.1.abs()))
    } else {
        (
            value.0.abs().min(value.1.abs()),
            value.0.abs().max(value.1.abs()),
        )
    }
}

fn stroke_safe_unit_range(direction: [(f64, f64); 2]) -> Option<[(f64, f64); 2]> {
    let horizontal_abs = range_abs(direction[0]);
    let vertical_abs = range_abs(direction[1]);
    let scale = (
        horizontal_abs.0.max(vertical_abs.0),
        horizontal_abs.1.max(vertical_abs.1),
    );
    if scale.0 < f64::from(f32::MIN_POSITIVE) {
        return None;
    }
    let scaled = [
        rounded_f32_division_range(direction[0], scale)?,
        rounded_f32_division_range(direction[1], scale)?,
    ];
    let length_squared = range_dot(scaled, scaled)?;
    if length_squared.0 < f64::from(f32::MIN_POSITIVE) {
        return None;
    }
    let mut inverse_length =
        rounded_f32_range(1.0 / length_squared.1.sqrt(), 1.0 / length_squared.0.sqrt())?;
    // WGSL permits two ULP error for inverseSqrt. Add two outward neighbours;
    // an inexact endpoint may already carry one additional rounding neighbour.
    for _ in 0..2 {
        inverse_length = (
            f64::from(next_f32_down(inverse_length.0 as f32)?),
            f64::from(next_f32_up(inverse_length.1 as f32)?),
        );
    }
    range_vector_scale(scaled, inverse_length)
}

fn rounded_f32_division_range(
    numerator: (f64, f64),
    denominator: (f64, f64),
) -> Option<(f64, f64)> {
    if denominator.0 <= 0.0 && denominator.1 >= 0.0 {
        return None;
    }
    let quotients = [
        numerator.0 / denominator.0,
        numerator.0 / denominator.1,
        numerator.1 / denominator.0,
        numerator.1 / denominator.1,
    ];
    let mut range = rounded_f32_range(
        quotients.into_iter().fold(f64::INFINITY, f64::min),
        quotients.into_iter().fold(f64::NEG_INFINITY, f64::max),
    )?;
    // WGSL permits 2.5 ULP division error. Three outward neighbours cover it;
    // an inexact endpoint may already carry one additional rounding neighbour.
    for _ in 0..3 {
        range = (
            f64::from(next_f32_down(range.0 as f32)?),
            f64::from(next_f32_up(range.1 as f32)?),
        );
    }
    Some(range)
}

fn rounded_f32_product_range(left: (f64, f64), right: (f64, f64)) -> Option<(f64, f64)> {
    let products = [
        left.0 * right.0,
        left.0 * right.1,
        left.1 * right.0,
        left.1 * right.1,
    ];
    rounded_f32_range(
        products.into_iter().fold(f64::INFINITY, f64::min),
        products.into_iter().fold(f64::NEG_INFINITY, f64::max),
    )
}

fn rounded_f32_add_range(
    left: (f64, f64),
    right: (f64, f64),
    subtract: bool,
) -> Option<(f64, f64)> {
    let (minimum, maximum) = if subtract {
        (left.0 - right.1, left.1 - right.0)
    } else {
        (left.0 + right.0, left.1 + right.1)
    };
    rounded_f32_range(minimum, maximum)
}

fn rounded_f32_range(minimum: f64, maximum: f64) -> Option<(f64, f64)> {
    if !minimum.is_finite()
        || !maximum.is_finite()
        || minimum < -f64::from(MAX_PORTABLE_SHADER_VALUE)
        || maximum > f64::from(MAX_PORTABLE_SHADER_VALUE)
    {
        return None;
    }
    if minimum == maximum && f64::from(minimum as f32) == minimum {
        return Some((minimum, maximum));
    }
    let mut minimum = f64::from(next_f32_down(minimum as f32)?);
    let mut maximum = f64::from(next_f32_up(maximum as f32)?);
    let minimum_normal = f64::from(f32::MIN_POSITIVE);
    if maximum > 0.0 && minimum < minimum_normal {
        minimum = minimum.min(0.0);
    }
    if minimum < 0.0 && maximum > -minimum_normal {
        maximum = maximum.max(0.0);
    }
    Some((minimum, maximum))
}

fn next_f32_down(value: f32) -> Option<f32> {
    if !value.is_finite() {
        return None;
    }
    let next = if value == 0.0 {
        -f32::from_bits(1)
    } else if value > 0.0 {
        f32::from_bits(value.to_bits().checked_sub(1)?)
    } else {
        f32::from_bits(value.to_bits().checked_add(1)?)
    };
    next.is_finite().then_some(next)
}

fn next_f32_up(value: f32) -> Option<f32> {
    if !value.is_finite() {
        return None;
    }
    let next = if value == 0.0 {
        f32::from_bits(1)
    } else if value > 0.0 {
        f32::from_bits(value.to_bits().checked_add(1)?)
    } else {
        f32::from_bits(value.to_bits().checked_sub(1)?)
    };
    next.is_finite().then_some(next)
}

fn screen_ranges_to_clip(
    screen: [(f64, f64); 2],
    uniform: CameraUniform,
) -> Option<[(f64, f64); 2]> {
    Some([
        shader_interval_sum_range([
            interval_products_f64(uniform.screen_to_clip[0], screen[0].0, screen[0].1),
            (
                f64::from(uniform.screen_to_clip[2]),
                f64::from(uniform.screen_to_clip[2]),
            ),
        ])?,
        shader_interval_sum_range([
            interval_products_f64(uniform.screen_to_clip[1], screen[1].0, screen[1].1),
            (
                f64::from(uniform.screen_to_clip[3]),
                f64::from(uniform.screen_to_clip[3]),
            ),
        ])?,
    ])
}

fn shader_interval_difference(left: (f64, f64), right: (f64, f64)) -> Option<(f64, f64)> {
    shader_interval_sum_range([left, (-right.1, -right.0)])
}

fn shader_interval_product(left: (f64, f64), right: (f64, f64)) -> Option<(f64, f64)> {
    let products = [
        left.0 * right.0,
        left.0 * right.1,
        left.1 * right.0,
        left.1 * right.1,
    ];
    shader_interval_sum_range([(
        products.into_iter().fold(f64::INFINITY, f64::min),
        products.into_iter().fold(f64::NEG_INFINITY, f64::max),
    )])
}

fn geometry_vertex_centers_are_portable(
    source: GeometryValidationSource<'_>,
    uniform: CameraUniform,
) -> bool {
    let dynamic_center_is_portable = |world: [f32; 2], depth: f32| {
        let Some(relative_x) = shader_relative_component_bounds(
            world[0],
            world[0],
            uniform.camera_center[0],
            0.0,
            0.0,
        ) else {
            return false;
        };
        let Some(relative_y) = shader_relative_component_bounds(
            world[1],
            world[1],
            uniform.camera_center[1],
            0.0,
            0.0,
        ) else {
            return false;
        };
        let minimum = [relative_x.0, relative_y.0];
        let maximum = [relative_x.1, relative_y.1];
        shader_world_dot_range(uniform.world_to_screen_x, minimum, maximum, depth, depth).is_some()
            && shader_world_dot_range(uniform.world_to_screen_y, minimum, maximum, depth, depth)
                .is_some()
    };
    match source {
        GeometryValidationSource::Tessellated(vertices) => vertices
            .iter()
            .all(|vertex| tessellated_vertex_clip_ranges(*vertex, uniform).is_some()),
        GeometryValidationSource::Dynamic(vertices) => vertices
            .iter()
            .all(|vertex| dynamic_center_is_portable(vertex.world_position, vertex.depth)),
    }
}

fn geometry_sources_are_portable(source: GeometryValidationSource<'_>) -> bool {
    match source {
        GeometryValidationSource::Tessellated(vertices) => vertices.iter().all(|vertex| {
            vertex
                .world_position
                .into_iter()
                .chain(vertex.world_offset)
                .chain(vertex.screen_offset)
                .chain(vertex.previous_direction)
                .chain(vertex.next_direction)
                .chain([
                    vertex.depth,
                    vertex.normal_distance,
                    vertex.tangent_distance,
                    vertex.miter_limit,
                ])
                .all(is_portable_shader_source)
        }),
        GeometryValidationSource::Dynamic(vertices) => vertices.iter().all(|vertex| {
            vertex
                .world_position
                .into_iter()
                .chain([vertex.depth])
                .all(is_portable_shader_source)
        }),
    }
}

fn logical_stroke_branches_are_stable(vertex: Vertex, uniform: CameraUniform) -> bool {
    if vertex.normal_distance == 0.0 && vertex.tangent_distance == 0.0 {
        return true;
    }
    let project = |direction: [f32; 2]| {
        [
            f64::from(uniform.world_to_screen_x[0]) * f64::from(direction[0])
                + f64::from(uniform.world_to_screen_x[1]) * f64::from(direction[1]),
            f64::from(uniform.world_to_screen_y[0]) * f64::from(direction[0])
                + f64::from(uniform.world_to_screen_y[1]) * f64::from(direction[1]),
        ]
    };
    let stable_projected_direction = |direction: [f32; 2]| {
        let direction = Vec2::new(direction[0], direction[1]);
        let horizontal =
            shader_direction_dot_range(uniform.world_to_screen_x, direction, direction)?;
        let vertical = shader_direction_dot_range(uniform.world_to_screen_y, direction, direction)?;
        let fixed = project([direction.x, direction.y]);
        let component_minimum_magnitude = |range: (f64, f64)| {
            if range.0 > 0.0 {
                range.0
            } else if range.1 < 0.0 {
                -range.1
            } else {
                0.0
            }
        };
        let lower_length =
            component_minimum_magnitude(horizontal).hypot(component_minimum_magnitude(vertical));
        let fixed_length = fixed[0].hypot(fixed[1]);
        let uncertainty = (horizontal.0 - fixed[0])
            .abs()
            .max((horizontal.1 - fixed[0]).abs())
            .hypot(
                (vertical.0 - fixed[1])
                    .abs()
                    .max((vertical.1 - fixed[1]).abs()),
            );
        let minimum_normal = f64::from(f32::MIN_POSITIVE);
        (lower_length >= minimum_normal
            && fixed_length.is_finite()
            && uncertainty.is_finite()
            && uncertainty <= fixed_length * 1.0e-6)
            .then_some(fixed)
    };
    let Some(previous_projected) = stable_projected_direction(vertex.previous_direction) else {
        return false;
    };
    let Some(next_projected) = stable_projected_direction(vertex.next_direction) else {
        return false;
    };
    if vertex.previous_direction == vertex.next_direction {
        // The shader detects source equality and reuses one projection and
        // normalization result, so identical directions cannot acquire a
        // backend-dependent artificial turn.
        return true;
    }

    let normalize = |value: [f64; 2]| {
        let scale = value[0].abs().max(value[1].abs());
        if !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        let scaled = [value[0] / scale, value[1] / scale];
        let length = scaled[0].hypot(scaled[1]);
        (length.is_finite() && length > 0.0).then_some([scaled[0] / length, scaled[1] / length])
    };
    let Some(previous) = normalize(previous_projected) else {
        return false;
    };
    let Some(next) = normalize(next_projected) else {
        return false;
    };
    let turn = previous[0] * next[1] - previous[1] * next[0];
    let tangent_dot = previous[0] * next[0] + previous[1] * next[1];
    if !turn.is_finite() || !tangent_dot.is_finite() {
        return false;
    }
    // Keep every topology-changing comparison far outside WGSL's permitted
    // normal-operation error. Exact repeated directions took the fast path.
    if turn.abs() <= 1.0e-4 || tangent_dot <= -0.999_9 {
        return false;
    }

    let previous_normal = [-previous[1], previous[0]];
    let next_normal = [-next[1], next[0]];
    let combined = [
        previous_normal[0] + next_normal[0],
        previous_normal[1] + next_normal[1],
    ];
    let Some(miter) = normalize(combined) else {
        return false;
    };
    let denominator = miter[0] * next_normal[0] + miter[1] * next_normal[1];
    if !denominator.is_finite() || denominator.abs() <= 0.002 {
        return false;
    }
    let miter_multiple = 1.0 / denominator.abs();
    let limit = f64::from(vertex.miter_limit);
    let comparison_margin = 1.0e-4 * miter_multiple.abs().max(limit.abs()).max(1.0);
    (miter_multiple - limit).abs() > comparison_margin
}

fn shader_clip_interval_is_safe(
    screen_minimum: f64,
    screen_maximum: f64,
    scale: f32,
    offset: f32,
) -> bool {
    if ((is_nonzero_subnormal_f64(screen_minimum) || is_nonzero_subnormal_f64(screen_maximum))
        && scale != 0.0)
        || (is_nonzero_subnormal(scale) && (screen_minimum != 0.0 || screen_maximum != 0.0))
    {
        return false;
    }
    shader_interval_sum_range([
        interval_products_f64(scale, screen_minimum, screen_maximum),
        (f64::from(offset), f64::from(offset)),
    ])
    .is_some()
}

fn shader_relative_component_bounds(
    world_minimum: f32,
    world_maximum: f32,
    camera_center: f32,
    offset_minimum: f32,
    offset_maximum: f32,
) -> Option<(f64, f64)> {
    // Exact equal f32 operands subtract to exact zero on every backend. Keep
    // that important camera-relative case tight so separately projected local
    // tessellation offsets remain usable at large world anchors.
    let subtraction = if world_minimum == camera_center && world_maximum == camera_center {
        (0.0, 0.0)
    } else {
        shader_interval_sum_range([
            (f64::from(world_minimum), f64::from(world_maximum)),
            (-f64::from(camera_center), -f64::from(camera_center)),
        ])?
    };
    if offset_minimum == 0.0 && offset_maximum == 0.0 {
        Some(subtraction)
    } else {
        shader_interval_sum_range([
            subtraction,
            (f64::from(offset_minimum), f64::from(offset_maximum)),
        ])
    }
}

fn shader_world_dot_range(
    row: [f32; 4],
    minimum: [f64; 2],
    maximum: [f64; 2],
    depth_minimum: f32,
    depth_maximum: f32,
) -> Option<(f64, f64)> {
    if (is_nonzero_subnormal(row[0]) && (minimum[0] != 0.0 || maximum[0] != 0.0))
        || (is_nonzero_subnormal(row[1]) && (minimum[1] != 0.0 || maximum[1] != 0.0))
        || (is_nonzero_subnormal(row[2]) && (depth_minimum != 0.0 || depth_maximum != 0.0))
        || ((is_nonzero_subnormal_f64(minimum[0]) || is_nonzero_subnormal_f64(maximum[0]))
            && row[0] != 0.0)
        || ((is_nonzero_subnormal_f64(minimum[1]) || is_nonzero_subnormal_f64(maximum[1]))
            && row[1] != 0.0)
        || ((is_nonzero_subnormal(depth_minimum) || is_nonzero_subnormal(depth_maximum))
            && row[2] != 0.0)
    {
        return None;
    }
    shader_interval_sum_range([
        interval_products_f64(row[0], minimum[0], maximum[0]),
        interval_products_f64(row[1], minimum[1], maximum[1]),
        interval_products(row[2], depth_minimum, depth_maximum),
        (f64::from(row[3]), f64::from(row[3])),
    ])
}

fn shader_direction_dot_range(row: [f32; 4], minimum: Vec2, maximum: Vec2) -> Option<(f64, f64)> {
    if (is_nonzero_subnormal(row[0]) && (minimum.x != 0.0 || maximum.x != 0.0))
        || (is_nonzero_subnormal(row[1]) && (minimum.y != 0.0 || maximum.y != 0.0))
        || ((is_nonzero_subnormal(minimum.x) || is_nonzero_subnormal(maximum.x)) && row[0] != 0.0)
        || ((is_nonzero_subnormal(minimum.y) || is_nonzero_subnormal(maximum.y)) && row[1] != 0.0)
    {
        return None;
    }
    shader_interval_sum_range([
        interval_products(row[0], minimum.x, maximum.x),
        interval_products(row[1], minimum.y, maximum.y),
    ])
}

#[cfg(test)]
fn shader_interval_sum_is_safe<const N: usize>(terms: [(f64, f64); N]) -> bool {
    shader_interval_sum_range(terms).is_some()
}

fn shader_interval_sum_range<const N: usize>(terms: [(f64, f64); N]) -> Option<(f64, f64)> {
    // WGSL's `dot` may be lowered with a backend-selected association or FMA
    // pattern. Bound all multiplication/addition rounding as well as every
    // same-sign partial sum, then return the complete finite output interval.
    // Keeping that interval is essential when a later shader stage multiplies
    // a cancellation residual by another large coefficient.
    let maximum = f64::from(MAX_PORTABLE_SHADER_VALUE);
    let mut positive_sum = 0.0;
    let mut negative_sum = 0.0;
    let mut minimum_sum = 0.0;
    let mut maximum_sum = 0.0;
    let mut magnitude_sum = 0.0;
    for term in terms {
        // WGSL permits implementations to flush subnormal arithmetic to zero.
        // A flushed product can become visually significant after a later
        // large camera or clip multiplier, so reject it structurally.
        if is_nonzero_subnormal_f64(term.0)
            || is_nonzero_subnormal_f64(term.1)
            || !term.0.is_finite()
            || !term.1.is_finite()
            || term.0 < -maximum
            || term.1 > maximum
        {
            return None;
        }
        positive_sum += term.1.max(0.0);
        negative_sum += term.0.min(0.0);
        minimum_sum += term.0;
        maximum_sum += term.1;
        magnitude_sum += term.0.abs().max(term.1.abs());
    }

    // Exact zero terms do not participate in any association, and an isolated
    // exactly representable product needs no rounding allowance. This keeps
    // component-selection rows tight inside the portability envelope while
    // retaining a conservative margin for operations which can round.
    let active_terms = terms
        .iter()
        .filter(|term| term.0 != 0.0 || term.1 != 0.0)
        .count();
    if active_terms == 1 {
        let term = terms
            .into_iter()
            .find(|term| term.0 != 0.0 || term.1 != 0.0)?;
        // A single interval term has no association ambiguity. When both
        // endpoints are already representable f32 values, they also prove
        // that the producing multiply rounded exactly at the extrema; adding
        // the remaining exact zero terms cannot enlarge the range. This is
        // important for a later dot row which merely selects or negates one
        // component from an earlier association envelope.
        if [term.0, term.1]
            .into_iter()
            .all(|value| f64::from(value as f32) == value)
        {
            return Some(term);
        }
    }
    let inexact_products = terms
        .iter()
        .filter(|term| {
            term.0 != term.1 || !term.0.is_finite() || f64::from(term.0 as f32) != term.0
        })
        .count();
    let operation_count = inexact_products + active_terms.saturating_sub(1);
    let operation_count = operation_count as f64;
    // WGSL correctly-rounded operations may select either adjacent f32 value;
    // use the directed-rounding bound rather than Rust's round-to-nearest.
    let unit_roundoff = 2.0_f64.powi(-23);
    let gamma = operation_count * unit_roundoff / (1.0 - operation_count * unit_roundoff);
    let subnormal_margin = if magnitude_sum > 0.0 {
        operation_count * f64::from(f32::MIN_POSITIVE)
    } else {
        0.0
    };
    let rounding_margin = gamma * magnitude_sum + subnormal_margin;
    let minimum_output = minimum_sum - rounding_margin;
    let maximum_output = maximum_sum + rounding_margin;
    if !positive_sum.is_finite()
        || !negative_sum.is_finite()
        || !minimum_output.is_finite()
        || !maximum_output.is_finite()
        || positive_sum + rounding_margin > maximum
        || negative_sum - rounding_margin < -maximum
        || minimum_output < -maximum
        || maximum_output > maximum
    {
        return None;
    }
    Some((minimum_output, maximum_output))
}

fn is_nonzero_subnormal(value: f32) -> bool {
    value != 0.0 && value.abs() < f32::MIN_POSITIVE
}

fn is_nonzero_subnormal_f64(value: f64) -> bool {
    value != 0.0 && value.abs() < f64::from(f32::MIN_POSITIVE)
}

fn interval_products(coefficient: f32, minimum: f32, maximum: f32) -> (f64, f64) {
    interval_products_f64(coefficient, f64::from(minimum), f64::from(maximum))
}

fn interval_products_f64(coefficient: f32, minimum: f64, maximum: f64) -> (f64, f64) {
    let first = f64::from(coefficient) * minimum;
    let second = f64::from(coefficient) * maximum;
    (first.min(second), first.max(second))
}

/// Result of attempting to draw a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStatus {
    /// Commands were submitted successfully. Surface paths also requested
    /// presentation; offscreen paths completed their target submission only.
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
    /// Returns CPU time spent preparing visual work before GPU uploads.
    ///
    /// For an ordinary streaming scene this is validation and tessellation.
    /// A heterogeneous [`FrameComposer`](crate::FrameComposer) frame also
    /// includes viewport and source validation, particle visibility selection,
    /// scalar LUT generation, and preparation of retained image/glyph draws.
    pub fn tessellation(self) -> Duration {
        self.tessellation
    }

    /// Returns CPU time spent preparing and enqueueing non-camera per-frame GPU data.
    ///
    /// Depending on the rendering path this includes transient vertex or
    /// instance writes, scalar LUT and pass-uniform uploads, and creation of
    /// the per-frame bindings needed by image, glyph, scalar, particle, or
    /// composition passes. The camera-only write remains separately available
    /// through [`Self::camera_uniform_upload`]. This is CPU preparation/enqueue
    /// time, not GPU execution time. An ordinary streaming frame can include
    /// pre-acquire transient buffer
    /// capacity creation here, so a skipped report may have a non-zero duration
    /// even though it performed no queue write and reports zero uploaded bytes.
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

    /// Returns whether this frame referenced retained visual geometry.
    ///
    /// This includes prepared scenes and retained sprite/glyph batches. It is
    /// a broad workload-classification hint rather than a statement that the
    /// frame performed no uploads, and it can be true together with
    /// [`Self::geometry_streamed`].
    pub fn geometry_reused(self) -> bool {
        self.geometry_reused
    }

    /// Returns whether this frame contained dynamically supplied visual work.
    ///
    /// This includes dynamic meshes, particle visibility instances, and scalar
    /// field passes. It is a broad workload-classification hint rather than an
    /// allocation or upload guarantee, and it can be true together with
    /// [`Self::geometry_reused`].
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
    /// Camera, geometry, stroke arithmetic, or dynamic-triangle topology
    /// cannot be proven portable on the GPU.
    ///
    /// This includes nonzero subnormal or excessively large shader sources,
    /// transform ranges that may overflow under permitted GPU arithmetic, and
    /// stroke branch decisions that are ambiguous after projection, every
    /// dynamic triangle requiring partial frustum clipping, and projected
    /// triangle orientation that may change with legal shader association.
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
    /// CPU storage for the camera-visible particle subset could not be reserved.
    ParticleAllocationFailed {
        /// Bytes requested for the rejected visible-instance vector.
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
                write!(
                    formatter,
                    "camera, geometry, or dynamic topology is outside the portable GPU envelope"
                )
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
            Self::ParticleAllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for visible particle instances"
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
    /// Requested physical dimensions exceed the selected device limit.
    SurfaceDimensionsTooLarge {
        /// Rejected physical width.
        width: u32,
        /// Rejected physical height.
        height: u32,
        /// Device maximum for either dimension.
        limit: u32,
    },
    /// The bounded previous-device quarantine is full.
    RecoveryLimitReached {
        /// Configured maximum number of quarantined logical devices.
        limit: usize,
    },
    /// CPU storage for quarantining the previous device could not be reserved.
    RecoveryAllocationFailed {
        /// Additional bytes requested for one quarantine entry.
        requested_bytes: usize,
    },
}

impl fmt::Display for RendererInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateSurface(error) => write!(formatter, "failed to create surface: {error}"),
            Self::RequestAdapter(error) => write!(formatter, "failed to request adapter: {error}"),
            Self::RequestDevice(error) => write!(formatter, "failed to request device: {error}"),
            Self::NoSurfaceConfig => write!(formatter, "surface has no supported default config"),
            Self::SurfaceDimensionsTooLarge {
                width,
                height,
                limit,
            } => write!(
                formatter,
                "surface dimensions {width}x{height} exceed device limit {limit}"
            ),
            Self::RecoveryLimitReached { limit } => write!(
                formatter,
                "device recovery limit reached with {limit} quarantined devices"
            ),
            Self::RecoveryAllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for device recovery quarantine"
            ),
        }
    }
}

impl Error for RendererInitError {}

/// Invalid runtime or initialization configuration for [`WgpuRenderer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RendererConfigurationError {
    /// Logical-to-physical scale must keep all surface transforms normal and finite.
    InvalidScaleFactor {
        /// Rejected physical pixels per logical screen pixel.
        scale_factor: f64,
    },
    /// Previous-device quarantine must remain inside the supported bounded range.
    InvalidRecoveryLimit {
        /// Rejected maximum retained-device count.
        limit: usize,
    },
    /// Requested physical dimensions exceed the active device limit.
    SurfaceDimensionsTooLarge {
        /// Rejected physical width.
        width: u32,
        /// Rejected physical height.
        height: u32,
        /// Device maximum for either dimension.
        limit: u32,
    },
}

impl fmt::Display for RendererConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScaleFactor { scale_factor } => write!(
                formatter,
                "renderer scale factor must be finite, representable as f32, and keep every u32 surface transform in the normal f32 range, got {scale_factor}"
            ),
            Self::InvalidRecoveryLimit { limit } => write!(
                formatter,
                "renderer recovery quarantine must retain between 1 and 8 devices, got {limit}"
            ),
            Self::SurfaceDimensionsTooLarge {
                width,
                height,
                limit,
            } => write!(
                formatter,
                "surface dimensions {width}x{height} exceed device limit {limit}"
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
    /// Command sources or derived tessellated vertices are outside the
    /// portable GPU input envelope.
    InvalidGeometrySources,
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
            Self::InvalidGeometrySources => write!(
                formatter,
                "prepared scene contains non-portable GPU geometry sources"
            ),
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
    vertices: Arc<Vec<Vertex>>,
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
    /// A position/depth is outside the portable shader envelope or a color is invalid.
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
                "dynamic mesh positions/depth must be portable and colors normalized"
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
        self.vertices
            .capacity()
            .saturating_mul(std::mem::size_of::<DynamicGpu>())
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

/// Hard steady-state per-field limits for bounding particle visualization.
///
/// Atomic replacement temporarily retains the old and fully prepared new
/// state together. Its CPU/GPU peak is therefore bounded by the sum of the old
/// and new budgets, while every committed field stays within its own budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParticleRenderBudget {
    max_visible_instances: usize,
    max_retained_bytes: usize,
    max_gpu_bytes: usize,
    max_upload_bytes_per_frame: usize,
    max_visibility_checks_per_frame: usize,
}

impl ParticleRenderBudget {
    /// Bytes used by one retained or visibility-staging particle slot.
    pub const INSTANCE_BYTES: usize = std::mem::size_of::<ParticleGpu>();

    /// No application-level cap beyond active-device allocation limits.
    pub const UNBOUNDED: Self = Self {
        max_visible_instances: usize::MAX,
        max_retained_bytes: usize::MAX,
        max_gpu_bytes: usize::MAX,
        max_upload_bytes_per_frame: usize::MAX,
        max_visibility_checks_per_frame: usize::MAX,
    };

    /// Creates explicit visible-instance, retained-CPU, GPU-memory, and
    /// per-frame upload caps.
    pub fn new(
        max_visible_instances: usize,
        max_retained_bytes: usize,
        max_gpu_bytes: usize,
        max_upload_bytes_per_frame: usize,
    ) -> Result<Self, ParticleBudgetError> {
        let minimum_bytes = Self::INSTANCE_BYTES;
        let minimum_retained_bytes = minimum_bytes.saturating_mul(2);
        if max_visible_instances == 0
            || max_visible_instances > u32::MAX as usize
            || max_retained_bytes < minimum_retained_bytes
            || max_gpu_bytes < minimum_bytes
            || max_upload_bytes_per_frame < minimum_bytes
        {
            return Err(ParticleBudgetError::InvalidLimit);
        }
        Ok(Self {
            max_visible_instances,
            max_retained_bytes,
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

    /// Returns the maximum steady-state engine-owned particle allocation in bytes.
    pub const fn max_retained_bytes(self) -> usize {
        self.max_retained_bytes
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
    /// CPU retention must fit one source plus one staging slot; GPU/upload
    /// limits must each fit at least one visible particle instance.
    InvalidLimit,
}

impl fmt::Display for ParticleBudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "particle budget must fit one retained instance, its visibility staging slot, and one visible GPU upload"
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
        self.instances
            .capacity()
            .saturating_mul(std::mem::size_of::<ParticleGpu>())
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
    /// An input instance violates the portable particle contract.
    InvalidInstance,
    /// A partial update lies outside the field's current instance range.
    UpdateRangeOutOfBounds,
    /// The field belongs to another renderer and GPU device.
    RendererMismatch,
    /// The instance capacity exceeds the current device's buffer limit.
    CapacityTooLarge,
    /// The retained CPU allocation exceeds the host-selected hard ceiling.
    RetainedBudgetExceeded {
        /// Configured retained-allocation ceiling.
        limit: usize,
        /// Required or actually reserved retained bytes.
        actual: usize,
    },
    /// Particle retention, replacement, or visibility-staging storage could
    /// not be reserved without panicking.
    AllocationFailed {
        /// Bytes requested by the failed particle-storage reservation.
        requested_bytes: usize,
    },
}

impl fmt::Display for ParticleFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInstance => write!(
                formatter,
                "particle field positions, depth, and radius must be portable and colors valid"
            ),
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
            Self::RetainedBudgetExceeded { limit, actual } => write!(
                formatter,
                "particle retained CPU allocation {actual} bytes exceeds limit {limit}"
            ),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for particle storage"
            ),
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
    /// Particle projection cannot be proven inside the portable GPU envelope.
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
    source_minimum: f32,
    source_maximum: f32,
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
        self.field.value_allocation_bytes()
    }

    /// Returns nominal `R32Float` texel-storage bytes requested from the GPU,
    /// excluding backend row/tile/page alignment and metadata.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.field
            .values()
            .len()
            .saturating_mul(std::mem::size_of::<f32>())
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

    /// Returns nominal texel-storage bytes for both ping-pong textures.
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

    /// Returns nominal single-level texel-storage bytes implied by its format.
    /// Backend row/tile/page alignment and resource metadata are not observable
    /// through wgpu and are therefore excluded.
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
    /// A finite scalar source lies outside the portable GPU arithmetic envelope.
    NonPortableValue,
    /// CPU recovery storage could not be reserved.
    AllocationFailed {
        /// Bytes requested for the rejected retained scalar copy.
        requested_bytes: usize,
    },
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
            Self::NonPortableValue => write!(
                formatter,
                "scalar values must be normal-or-zero and remain inside the portable GPU arithmetic envelope"
            ),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for scalar field recovery data"
            ),
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

    /// Returns currently allocated CPU recovery bytes for vertices and draw batches.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.vertices
            .capacity()
            .saturating_mul(std::mem::size_of::<Vertex>())
            .saturating_add(
                self.draw_batches
                    .capacity()
                    .saturating_mul(std::mem::size_of::<PreparedDrawBatch>()),
            )
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
        let dimension_limit = device.limits().max_texture_dimension_2d;
        if width > dimension_limit || height > dimension_limit {
            return Err(RendererInitError::SurfaceDimensionsTooLarge {
                width,
                height,
                limit: dimension_limit,
            });
        }
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
        let particle_unit_buffer = create_submitted_particle_unit_buffer(&device, &queue);
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
            vertices: Vec::new(),
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
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RendererConfigurationError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        validate_surface_dimensions(width, height, self.device.limits().max_texture_dimension_2d)?;

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.multisample_target =
            create_multisample_target(&self.device, &self.config, self.sample_count);
        Ok(())
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
        if width != 0 && height != 0 {
            validate_surface_dimensions(
                width,
                height,
                self.device.limits().max_texture_dimension_2d,
            )?;
        }
        self.resize(width, height)?;
        self.scale_factor = scale_factor;
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

        let tessellation_started_at = Instant::now();
        if !scene_estimate_fits_streaming_device(scene, &self.device, 0, self.vertex_capacity) {
            return Err(RenderTargetError::Frame(
                RendererFrameError::GeometryCapacityTooLarge,
            ));
        }
        self.vertices.clear();
        self.draw_batches.clear();
        let tessellation_stats =
            tessellate_scene(scene, &mut self.vertices, &mut self.draw_batches)
                .map_err(RendererFrameError::from)
                .map_err(RenderTargetError::Frame)?;
        let tessellation = tessellation_started_at.elapsed();

        let extents = GeometryExtents::from_vertices(&self.vertices);
        if !geometry_is_safe_for(
            extents,
            GeometryValidationSource::Tessellated(&self.vertices),
            camera_uniform,
        ) {
            return Err(RenderTargetError::Frame(
                RendererFrameError::InvalidGeometryTransform,
            ));
        }
        let upload_started_at = Instant::now();
        self.ensure_vertex_capacity(self.vertices.len())
            .map_err(RenderTargetError::Frame)?;
        if !self.vertices.is_empty() {
            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        }
        let upload = upload_started_at.elapsed();
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
                RenderTargetLoad::Clear(color) => {
                    wgpu::LoadOp::Clear(premultiplied_wgpu_color(color))
                }
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
        let (_, _, camera_uniform) = self.surface_geometry_context(*camera, viewport)?;
        let tessellation_started_at = Instant::now();
        if !scene_estimate_fits_streaming_device(scene, &self.device, 0, self.vertex_capacity) {
            return Err(RendererFrameError::GeometryCapacityTooLarge);
        }
        self.vertices.clear();
        self.draw_batches.clear();
        let tessellation_stats =
            tessellate_scene(scene, &mut self.vertices, &mut self.draw_batches)?;
        let tessellation = tessellation_started_at.elapsed();

        let geometry_extents = GeometryExtents::from_vertices(&self.vertices);
        if !geometry_is_safe_for(
            geometry_extents,
            GeometryValidationSource::Tessellated(&self.vertices),
            camera_uniform,
        ) {
            return Err(RendererFrameError::InvalidGeometryTransform);
        }

        let upload_started_at = Instant::now();
        self.ensure_vertex_capacity(self.vertices.len())?;
        let upload = upload_started_at.elapsed();

        let vertex_buffer = Arc::clone(&self.vertex_buffer);
        let draw_batches = std::mem::take(&mut self.draw_batches);
        let vertices = std::mem::take(&mut self.vertices);
        let result = self.draw_geometry(
            scene.background(),
            &vertex_buffer,
            vertices.len(),
            geometry_extents,
            GeometryValidationSource::Tessellated(&vertices),
            &draw_batches,
            *camera,
            tessellation,
            upload,
            false,
            true,
            false,
            Some(&vertices),
            tessellation_stats,
            frame_started_at,
            viewport,
        );
        self.vertices = vertices;
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
        let vertex_capacity = dynamic_vertex_capacity(vertices.len())
            .filter(|capacity| buffer_capacity_fits::<DynamicGpu>(&self.device, *capacity))
            .ok_or(DynamicMeshError::CapacityTooLarge)?;
        let vertices = dynamic_vertices_to_gpu(vertices)?;
        if let Some(budget) = budget {
            validate_dynamic_retained_capacity(budget, &vertices)?;
        }
        let vertex_buffer = Arc::new(create_dynamic_vertex_buffer(&self.device, vertex_capacity));
        if !vertices.is_empty() {
            self.queue
                .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            submit_pending_uploads(&self.queue);
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
            submit_pending_uploads(&self.queue);
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
    ///
    /// Every potentially visible triangle must remain fully inside the full
    /// surface clip volume and have a portable projected orientation. A
    /// hardware-clipped or ambiguous triangle returns
    /// `DynamicMeshRenderError::Frame(RendererFrameError::InvalidGeometryTransform)`.
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
    ///
    /// Full-surface clip-volume crossing and association-dependent projected
    /// topology are rejected as
    /// `DynamicMeshRenderError::Frame(RendererFrameError::InvalidGeometryTransform)`.
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
            GeometryValidationSource::Dynamic(&mesh.vertices),
            draw_batches.as_slice(),
            *camera,
            Duration::ZERO,
            Duration::ZERO,
            false,
            true,
            true,
            None,
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

    /// Creates an instanced particle field with hard retained-CPU, GPU-memory,
    /// upload, visible-instance, and visibility-work limits.
    pub fn create_particle_field_with_budget(
        &self,
        instances: &[ParticleInstance2d],
        budget: ParticleRenderBudget,
    ) -> Result<ParticleField2d, ParticleFieldError> {
        validate_particle_retained_count(instances.len(), budget)?;
        let instance_capacity = particle_budgeted_capacity(instances.len(), budget)
            .filter(|capacity| buffer_capacity_fits::<ParticleGpu>(&self.device, *capacity))
            .ok_or(ParticleFieldError::CapacityTooLarge)?;
        let instances = particle_instances_to_gpu(instances)?;
        let visible_instances = allocate_particle_staging(instances.len(), budget)?;
        validate_particle_retained_capacities(
            instances.capacity(),
            visible_instances.capacity(),
            budget,
        )?;
        let instance_buffer = Arc::new(create_particle_instance_buffer(
            &self.device,
            instance_capacity,
        ));
        Ok(ParticleField2d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            instance_buffer,
            statistics: particle_idle_statistics(instances.len()),
            instances,
            visible_instances,
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
        validate_particle_retained_count(field.instances.len(), budget)?;
        let desired_capacity = particle_budgeted_capacity(field.instances.len(), budget)
            .filter(|capacity| buffer_capacity_fits::<ParticleGpu>(&self.device, *capacity))
            .ok_or(ParticleFieldError::CapacityTooLarge)?;
        let visible_instances = allocate_particle_staging(field.instances.len(), budget)?;
        let replacement_instances = compact_particle_instances(
            &field.instances,
            field.instances.capacity(),
            visible_instances.capacity(),
            budget,
        )?;
        let replacement_buffer = (desired_capacity != field.instance_capacity).then(|| {
            Arc::new(create_particle_instance_buffer(
                &self.device,
                desired_capacity,
            ))
        });
        if let Some(instance_buffer) = replacement_buffer {
            field.instance_buffer = instance_buffer;
            field.instance_capacity = desired_capacity;
        }
        if let Some(instances) = replacement_instances {
            field.instances = instances;
        }
        field.budget = budget;
        field.visible_instances = visible_instances;
        field.statistics = particle_idle_statistics(field.instances.len());
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
        validate_particle_retained_count(instances.len(), field.budget)?;
        let desired_capacity = particle_budgeted_capacity(instances.len(), field.budget)
            .filter(|capacity| buffer_capacity_fits::<ParticleGpu>(&self.device, *capacity))
            .ok_or(ParticleFieldError::CapacityTooLarge)?;
        let instances = particle_instances_to_gpu(instances)?;
        let visible_instances = allocate_particle_staging(instances.len(), field.budget)?;
        validate_particle_retained_capacities(
            instances.capacity(),
            visible_instances.capacity(),
            field.budget,
        )?;
        let reallocated = desired_capacity > field.instance_capacity;
        let replacement_buffer = reallocated.then(|| {
            Arc::new(create_particle_instance_buffer(
                &self.device,
                desired_capacity,
            ))
        });
        if let Some(instance_buffer) = replacement_buffer {
            field.instance_buffer = instance_buffer;
            field.instance_capacity = desired_capacity;
        }
        field.statistics = particle_idle_statistics(instances.len());
        field.instances = instances;
        field.visible_instances = visible_instances;
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
        field.statistics = particle_idle_statistics(field.instances.len());
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
        validate_scalar_field_device_extent(&self.device, &field)?;
        if !scalar_field_sources_are_portable(&field) {
            return Err(ScalarFieldTextureError::NonPortableValue);
        }
        let field = compact_scalar_field_for_retention(field)?;
        let (source_minimum, source_maximum) = field.value_range();
        let reallocated = texture.width() != field.width() || texture.height() != field.height();
        if reallocated {
            texture.texture = create_scalar_field_texture(&self.device, &field)?;
        }
        upload_scalar_field_texture(&self.queue, &texture.texture, &field)?;
        texture.field = field;
        texture.source_minimum = source_minimum;
        texture.source_maximum = source_maximum;
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
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ScalarFieldTextureError::NonFiniteUpdateValue);
        }
        if values
            .iter()
            .copied()
            .any(|value| !is_portable_shader_source(value))
        {
            return Err(ScalarFieldTextureError::NonPortableValue);
        }
        validate_scalar_field_texture_region(
            texture.width(),
            texture.height(),
            x,
            y,
            width,
            height,
            values.len(),
        )?;
        let update_minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
        let update_maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let overwrites_extremum = (y..y + height).any(|row| {
            let start = row * texture.width() + x;
            texture.field.values()[start..start + width]
                .iter()
                .any(|value| *value == texture.source_minimum || *value == texture.source_maximum)
        });
        let (source_minimum, source_maximum) = if overwrites_extremum {
            scalar_region_result_range(&texture.field, x, y, width, height, values)
        } else {
            (
                texture.source_minimum.min(update_minimum),
                texture.source_maximum.max(update_maximum),
            )
        };
        let source_extent = f64::from(source_maximum) - f64::from(source_minimum);
        if !source_extent.is_finite() || source_extent > f64::from(MAX_PORTABLE_SHADER_VALUE) {
            return Err(ScalarFieldTextureError::NonPortableValue);
        }
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
        texture.source_minimum = source_minimum;
        texture.source_maximum = source_maximum;
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
        validate_scalar_field_device_extent(&self.device, &source.field)?;
        let field = clone_scalar_field_for_restore(&source.field)?;
        self.create_scalar_field_texture(field)
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
        if !scalar_normalization_is_portable(texture, minimum, value_extent) {
            return Err(ScalarFieldRenderError::InvalidValueRange { minimum, maximum });
        }
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
        if !scalar_normalization_is_portable(texture, minimum, value_extent) {
            return Err(ScalarFieldRenderError::InvalidValueRange { minimum, maximum });
        }
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
        if !scalar_normalization_is_portable(texture, minimum, value_extent) {
            return Err(ScalarFieldRenderError::InvalidValueRange { minimum, maximum });
        }
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
            GeometryValidationSource::Tessellated(&scene.vertices),
            &scene.draw_batches,
            *camera,
            Duration::ZERO,
            Duration::ZERO,
            true,
            false,
            false,
            None,
            scene.tessellation,
            Instant::now(),
            None,
        )
        .map_err(PreparedSceneRenderError::Frame)
    }

    fn surface_geometry_context(
        &self,
        camera: Camera2d,
        viewport_region: Option<LogicalViewportRegion>,
    ) -> Result<(LogicalViewport, ScissorRect, CameraUniform), RendererFrameError> {
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
        let camera_uniform =
            CameraUniform::new_in_region(camera, viewport, viewport_origin, target_viewport)
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
        Ok((viewport, viewport_scissor, camera_uniform))
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_geometry(
        &mut self,
        background: Color,
        vertex_buffer: &wgpu::Buffer,
        vertex_count: usize,
        geometry_extents: GeometryExtents,
        geometry_validation: GeometryValidationSource<'_>,
        draw_batches: &[PreparedDrawBatch],
        camera: Camera2d,
        tessellation: Duration,
        mut upload: Duration,
        geometry_reused: bool,
        geometry_streamed: bool,
        uses_dynamic_pipeline: bool,
        pending_vertex_upload: Option<&[Vertex]>,
        tessellation_stats: TessellationStats,
        frame_started_at: Instant,
        viewport_region: Option<LogicalViewportRegion>,
    ) -> Result<RenderReport, RendererFrameError> {
        let (viewport, viewport_scissor, camera_uniform) =
            self.surface_geometry_context(camera, viewport_region)?;
        if !geometry_is_safe_for(geometry_extents, geometry_validation, camera_uniform) {
            return Err(RendererFrameError::InvalidGeometryTransform);
        }

        let surface_acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    tessellation,
                    upload,
                    Duration::ZERO,
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
                    Duration::ZERO,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    geometry_reused,
                    geometry_streamed,
                    tessellation_stats,
                ));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                let _ = self.resize(self.config.width, self.config.height);
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                    tessellation,
                    upload,
                    Duration::ZERO,
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
        let vertex_upload_started_at = Instant::now();
        if let Some(vertices) = pending_vertex_upload
            && !vertices.is_empty()
        {
            self.queue
                .write_buffer(vertex_buffer, 0, bytemuck::cast_slice(vertices));
        }
        upload = upload.saturating_add(vertex_upload_started_at.elapsed());
        let camera_uniform_upload_started_at = Instant::now();
        self.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        let camera_uniform_upload = camera_uniform_upload_started_at.elapsed();

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
                let pipeline = if uses_dynamic_pipeline {
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
        let uniform = HeatmapUniform::new(
            minimum,
            value_extent,
            texture.width(),
            texture.height(),
            sampling,
        );
        let acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    Duration::ZERO,
                    Duration::ZERO,
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
                    Duration::ZERO,
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
                let _ = self.resize(self.config.width, self.config.height);
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                    Duration::ZERO,
                    Duration::ZERO,
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
        let upload_started_at = Instant::now();
        let color_map_view = self.color_map_view(color_map);
        let scalar_view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
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
        let acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    Duration::ZERO,
                    Duration::ZERO,
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
                    Duration::ZERO,
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
                let _ = self.resize(self.config.width, self.config.height);
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                    Duration::ZERO,
                    Duration::ZERO,
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
            validate_particle_staging_capacity(&field.visible_instances, visibility_checked)?;
            for candidate_index in 0..visibility_checked {
                let source_index =
                    uniformly_sampled_index(candidate_index, instance_count, visibility_checked);
                let instance = field.instances[source_index];
                let Some(intersects) =
                    instance.validated_viewport_intersection(camera_uniform, viewport)
                else {
                    return Err(RendererFrameError::InvalidGeometryTransform);
                };
                if intersects {
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
            let statistics = particle_statistics_with_budget(
                instance_count,
                visibility_checked,
                visible_count,
                selected_count,
                0,
            );
            return Ok(ParticleDrawPreparation {
                camera_uniform,
                visible_count: selected_count,
                statistics,
            });
        }

        let visible_count = visible_particle_count(&field.instances, camera_uniform, viewport)?;
        let selected_count = visible_count.min(field.budget.instance_limit());
        let statistics = particle_statistics_with_budget(
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
            validate_particle_staging_capacity(&field.visible_instances, selected_count)?;
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
        Ok(ParticleDrawPreparation {
            camera_uniform,
            visible_count: selected_count,
            statistics,
        })
    }

    fn upload_particle_draw(
        &self,
        field: &ParticleField2d,
        preparation: ParticleDrawPreparation,
    ) -> (Duration, Duration) {
        let visible_instances = if field.visible_instances.is_empty()
            && preparation.visible_count == field.instances.len()
        {
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
            bytemuck::bytes_of(&preparation.camera_uniform),
        );
        let camera_uniform_upload = camera_uniform_upload_started_at.elapsed();
        (upload, camera_uniform_upload)
    }

    fn draw_particle_field(
        &mut self,
        background: Color,
        field: &mut ParticleField2d,
        camera: Camera2d,
        frame_started_at: Instant,
    ) -> Result<RenderReport, RendererFrameError> {
        let preparation_started_at = Instant::now();
        let preparation = self.prepare_particle_draw(field, camera)?;
        let preparation_duration = preparation_started_at.elapsed();

        let surface_acquire_started_at = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                field.statistics = preparation.statistics;
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                    preparation_duration,
                    Duration::ZERO,
                    Duration::ZERO,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                field.statistics = preparation.statistics;
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
                    preparation_duration,
                    Duration::ZERO,
                    Duration::ZERO,
                    surface_acquire_started_at.elapsed(),
                    Duration::ZERO,
                    frame_started_at.elapsed(),
                    false,
                    true,
                    TessellationStats::default(),
                ));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                let _ = self.resize(self.config.width, self.config.height);
                field.statistics = preparation.statistics;
                return Ok(render_report(
                    RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                    preparation_duration,
                    Duration::ZERO,
                    Duration::ZERO,
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
        let (upload, camera_uniform_upload) = self.upload_particle_draw(field, preparation);
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
        field.statistics = ParticleStatistics {
            rendered: preparation.visible_count,
            ..preparation.statistics
        };
        Ok(render_report(
            RenderStatus::Drawn,
            preparation_duration,
            upload,
            camera_uniform_upload,
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
        let preparation_started_at = Instant::now();
        let preparation = self.prepare_particle_draw(field, camera)?;
        let preparation_duration = preparation_started_at.elapsed();
        let (upload, camera_uniform_upload) = self.upload_particle_draw(field, preparation);
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
        field.statistics = ParticleStatistics {
            rendered: preparation.visible_count,
            ..preparation.statistics
        };
        Ok(render_report(
            RenderStatus::Drawn,
            preparation_duration,
            upload,
            camera_uniform_upload,
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
        let lut = color_map_lut(color_map);
        if let Some(cached) = self
            .color_map_cache
            .as_ref()
            .filter(|cached| cached.lut == lut)
        {
            return cached.view.clone();
        }
        let cached = create_cached_color_map(&self.device, &self.queue, lut);
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
    let reallocated = vertices.len() > mesh.vertex_capacity;
    let replacement_capacity = if reallocated {
        let capacity = dynamic_vertex_capacity(vertices.len())
            .filter(|capacity| buffer_capacity_fits::<DynamicGpu>(device, *capacity))
            .ok_or(DynamicMeshError::CapacityTooLarge)?;
        Some(capacity)
    } else {
        None
    };
    let replacement_vertices = dynamic_vertices_to_gpu(vertices)?;
    if let Some(budget) = mesh.budget {
        validate_dynamic_retained_capacity(budget, &replacement_vertices)?;
    }
    let replacement = if let Some(capacity) = replacement_capacity {
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
        submit_pending_uploads(queue);
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
    if u32::try_from(vertex_count).is_err() {
        return Err(DynamicMeshError::CapacityTooLarge);
    }
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

fn validate_dynamic_retained_capacity(
    budget: DynamicMeshBudget,
    vertices: &Vec<DynamicGpu>,
) -> Result<(), DynamicMeshError> {
    let actual = vertices
        .capacity()
        .checked_mul(std::mem::size_of::<DynamicGpu>())
        .ok_or(DynamicMeshError::CapacityTooLarge)?;
    if actual > budget.max_retained_bytes {
        return Err(DynamicMeshError::BudgetExceeded {
            resource: DynamicMeshBudgetResource::RetainedBytes,
            limit: budget.max_retained_bytes,
            actual,
        });
    }
    Ok(())
}

fn validate_dynamic_vertices(vertices: &[DynamicVertex2d]) -> Result<(), DynamicMeshError> {
    if u32::try_from(vertices.len()).is_err() {
        return Err(DynamicMeshError::CapacityTooLarge);
    }
    if !vertices.len().is_multiple_of(3) {
        return Err(DynamicMeshError::InvalidVertexCount);
    }
    if vertices.iter().any(|vertex| {
        ![
            vertex.world_position.x,
            vertex.world_position.y,
            vertex.depth,
        ]
        .into_iter()
        .all(is_portable_shader_source)
    }) {
        return Err(DynamicMeshError::InvalidVertex);
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
    if u32::try_from(instances.len()).is_err() {
        return Err(ParticleFieldError::CapacityTooLarge);
    }
    let requested_bytes = instances
        .len()
        .checked_mul(std::mem::size_of::<ParticleGpu>())
        .ok_or(ParticleFieldError::CapacityTooLarge)?;
    if instances.iter().copied().any(|instance| {
        let world_position = instance.world_position();
        let radius = instance.radius();
        let color = instance.color();
        let depth = instance.depth();
        ![world_position.x, world_position.y, radius, depth]
            .into_iter()
            .all(is_portable_shader_source)
            || radius <= 0.0
            || !color.is_normalized()
    }) {
        return Err(ParticleFieldError::InvalidInstance);
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(instances.len())
        .map_err(|_| ParticleFieldError::AllocationFailed { requested_bytes })?;
    output.extend(instances.iter().copied().map(|instance| {
        let world_position = instance.world_position();
        ParticleGpu {
            world_position: [world_position.x, world_position.y],
            depth: instance.depth(),
            radius: instance.radius(),
            color: instance.color().to_array(),
        }
    }));
    Ok(output)
}

fn particle_retained_bytes(instance_count: usize) -> Option<usize> {
    instance_count.checked_mul(std::mem::size_of::<ParticleGpu>())
}

fn validate_particle_retained_count(
    instance_count: usize,
    budget: ParticleRenderBudget,
) -> Result<(), ParticleFieldError> {
    let staging_count = particle_staging_target(instance_count, budget);
    let retained_count = instance_count
        .checked_add(staging_count)
        .ok_or(ParticleFieldError::CapacityTooLarge)?;
    let actual =
        particle_retained_bytes(retained_count).ok_or(ParticleFieldError::CapacityTooLarge)?;
    if actual > budget.max_retained_bytes {
        return Err(ParticleFieldError::RetainedBudgetExceeded {
            limit: budget.max_retained_bytes,
            actual,
        });
    }
    Ok(())
}

fn validate_particle_retained_capacities(
    instance_capacity: usize,
    staging_capacity: usize,
    budget: ParticleRenderBudget,
) -> Result<(), ParticleFieldError> {
    let retained_capacity = instance_capacity
        .checked_add(staging_capacity)
        .ok_or(ParticleFieldError::CapacityTooLarge)?;
    let actual =
        particle_retained_bytes(retained_capacity).ok_or(ParticleFieldError::CapacityTooLarge)?;
    if actual > budget.max_retained_bytes {
        return Err(ParticleFieldError::RetainedBudgetExceeded {
            limit: budget.max_retained_bytes,
            actual,
        });
    }
    Ok(())
}

fn compact_particle_instances(
    source: &[ParticleGpu],
    current_capacity: usize,
    staging_capacity: usize,
    budget: ParticleRenderBudget,
) -> Result<Option<Vec<ParticleGpu>>, ParticleFieldError> {
    validate_particle_retained_count(source.len(), budget)?;
    if validate_particle_retained_capacities(current_capacity, staging_capacity, budget).is_ok() {
        return Ok(None);
    }
    let requested_bytes =
        particle_retained_bytes(source.len()).ok_or(ParticleFieldError::CapacityTooLarge)?;
    let mut compact = Vec::new();
    compact
        .try_reserve_exact(source.len())
        .map_err(|_| ParticleFieldError::AllocationFailed { requested_bytes })?;
    compact.extend_from_slice(source);
    validate_particle_retained_capacities(compact.capacity(), staging_capacity, budget)?;
    Ok(Some(compact))
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

fn allocate_particle_staging(
    instance_count: usize,
    budget: ParticleRenderBudget,
) -> Result<Vec<ParticleGpu>, ParticleFieldError> {
    let target = particle_staging_target(instance_count, budget);
    let requested_bytes = target
        .checked_mul(std::mem::size_of::<ParticleGpu>())
        .ok_or(ParticleFieldError::CapacityTooLarge)?;
    let mut staging = Vec::new();
    staging
        .try_reserve_exact(target)
        .map_err(|_| ParticleFieldError::AllocationFailed { requested_bytes })?;
    Ok(staging)
}

fn particle_staging_target(instance_count: usize, budget: ParticleRenderBudget) -> usize {
    let visibility_checked = instance_count.min(budget.max_visibility_checks_per_frame);
    if visibility_checked < instance_count {
        visibility_checked
    } else {
        instance_count.min(budget.instance_limit())
    }
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

fn scene_estimate_fits_prepared_device(scene: &Scene, device: &wgpu::Device) -> bool {
    buffer_capacity_fits::<Vertex>(
        device,
        scene.statistics().estimated_tessellated_vertices().max(1),
    )
}

fn scene_estimate_fits_streaming_device(
    scene: &Scene,
    device: &wgpu::Device,
    prefix: usize,
    current_capacity: usize,
) -> bool {
    let Some(required) = prefix.checked_add(scene.statistics().estimated_tessellated_vertices())
    else {
        return false;
    };
    required <= current_capacity
        || dynamic_vertex_capacity(required)
            .is_some_and(|capacity| buffer_capacity_fits::<Vertex>(device, capacity))
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

fn validate_particle_staging_capacity(
    instances: &Vec<ParticleGpu>,
    target_len: usize,
) -> Result<(), RendererFrameError> {
    let requested_bytes = target_len
        .checked_mul(std::mem::size_of::<ParticleGpu>())
        .ok_or(RendererFrameError::GeometryCapacityTooLarge)?;
    (instances.capacity() >= target_len)
        .then_some(())
        .ok_or(RendererFrameError::ParticleAllocationFailed { requested_bytes })
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
        let Some(intersects) = instance.validated_viewport_intersection(camera, viewport) else {
            return Err(RendererFrameError::InvalidGeometryTransform);
        };
        visible += usize::from(intersects);
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

fn particle_idle_statistics(instance_count: usize) -> ParticleStatistics {
    ParticleStatistics {
        submitted: instance_count,
        visibility_checked: 0,
        visible: 0,
        culled: 0,
        budget_limited: 0,
        dropped: 0,
        rendered: 0,
    }
}

#[cfg(test)]
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
    if !scene_command_sources_are_portable(scene) {
        return Err(PreparedSceneError::InvalidGeometrySources);
    }
    if !scene_estimate_fits_prepared_device(scene, device) {
        return Err(PreparedSceneError::CapacityTooLarge);
    }
    let mut vertices = Vec::new();
    let mut draw_batches = Vec::new();
    let tessellation = tessellate_scene(scene, &mut vertices, &mut draw_batches)?;
    let geometry_extents = GeometryExtents::from_vertices(&vertices);
    if !geometry_sources_are_portable(GeometryValidationSource::Tessellated(&vertices)) {
        return Err(PreparedSceneError::InvalidGeometrySources);
    }
    if !buffer_capacity_fits::<Vertex>(device, vertices.len().max(1)) {
        return Err(PreparedSceneError::CapacityTooLarge);
    }
    // Retain the tessellator allocation directly. Wrapping a Vec is O(1) and
    // avoids an infallible caller-scale Vec -> Arc<[T]> copy after GPU writes.
    let vertices = Arc::new(vertices);
    let vertex_buffer = Arc::new(create_vertex_buffer(device, vertices.len()));
    if !vertices.is_empty() {
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        submit_pending_uploads(queue);
    }
    let vertex_count = vertices.len();

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

fn scene_command_sources_are_portable(scene: &Scene) -> bool {
    scene.commands().iter().all(|command| {
        is_portable_shader_source(command.depth())
            && command.screen_clip().is_none_or(|clip| {
                portable_vec2(clip.min().to_vec2()) && portable_vec2(clip.max().to_vec2())
            })
            && match command.command() {
                DrawCommand::Circle(circle) => {
                    portable_vec2(circle.center())
                        && is_portable_shader_source(circle.radius())
                        && portable_shape_style(circle.style())
                }
                DrawCommand::Rect(rectangle) => {
                    portable_vec2(rectangle.rect().min())
                        && portable_vec2(rectangle.rect().max())
                        && is_portable_shader_source(rectangle.corner_radius())
                        && portable_shape_style(rectangle.style())
                }
                DrawCommand::Line(line) => {
                    portable_vec2(line.from())
                        && portable_vec2(line.to())
                        && portable_stroke_style(line.stroke_style())
                }
                DrawCommand::Polyline(polyline) => {
                    polyline.points().iter().copied().all(portable_vec2)
                        && portable_stroke_style(polyline.stroke_style())
                }
            }
    })
}

fn portable_vec2(value: Vec2) -> bool {
    is_portable_shader_source(value.x) && is_portable_shader_source(value.y)
}

fn portable_shape_style(style: ShapeStyle) -> bool {
    style
        .stroke()
        .is_none_or(|stroke| is_portable_shader_source(stroke.width()))
        && style.shadow().is_none_or(|shadow| {
            portable_vec2(shadow.offset().to_vec2()) && is_portable_shader_source(shadow.spread())
        })
}

fn portable_stroke_style(style: crate::StrokeStyle2d) -> bool {
    let stroke = style.stroke();
    is_portable_shader_source(stroke.width())
        && is_portable_shader_source(style.miter_limit())
        && style.dash_pattern().is_none_or(|dash| {
            dash.lengths()
                .iter()
                .copied()
                .chain([dash.phase()])
                .all(is_portable_shader_source)
        })
        && style.start_marker().is_none_or(|marker| {
            is_portable_shader_source(marker.length().get())
                && is_portable_shader_source(marker.width().get())
        })
        && style.end_marker().is_none_or(|marker| {
            is_portable_shader_source(marker.length().get())
                && is_portable_shader_source(marker.width().get())
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
    if !geometry_sources_are_portable(GeometryValidationSource::Tessellated(&source.vertices)) {
        return Err(PreparedSceneError::InvalidGeometrySources);
    }
    let requested_bytes = source
        .draw_batches
        .len()
        .checked_mul(std::mem::size_of::<PreparedDrawBatch>())
        .ok_or(PreparedSceneError::CapacityTooLarge)?;
    let mut draw_batches = Vec::new();
    draw_batches
        .try_reserve_exact(source.draw_batches.len())
        .map_err(|_| PreparedSceneError::AllocationFailed { requested_bytes })?;
    draw_batches.extend_from_slice(&source.draw_batches);
    let vertex_buffer = Arc::new(create_vertex_buffer(device, source.vertices.len()));
    if !source.vertices.is_empty() {
        queue.write_buffer(
            &vertex_buffer,
            0,
            bytemuck::cast_slice(source.vertices.as_ref()),
        );
        submit_pending_uploads(queue);
    }

    Ok(PreparedScene {
        renderer_identity,
        background: source.background,
        vertex_buffer,
        vertices: Arc::clone(&source.vertices),
        command_count: source.command_count,
        vertex_count: source.vertex_count,
        geometry_extents: source.geometry_extents,
        draw_batches,
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
    if let Some(budget) = source.budget {
        validate_dynamic_retained_capacity(budget, &vertices)?;
    }
    let vertex_buffer = Arc::new(create_dynamic_vertex_buffer(device, source.vertex_capacity));
    if !vertices.is_empty() {
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(vertices.as_slice()));
        submit_pending_uploads(queue);
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
    validate_particle_retained_count(source.instances.len(), source.budget)?;
    if !buffer_capacity_fits::<ParticleGpu>(device, source.instance_capacity) {
        return Err(ParticleFieldError::CapacityTooLarge);
    }
    let requested_bytes = source
        .instances
        .len()
        .checked_mul(std::mem::size_of::<ParticleGpu>())
        .ok_or(ParticleFieldError::CapacityTooLarge)?;
    let mut instances = Vec::new();
    instances
        .try_reserve_exact(source.instances.len())
        .map_err(|_| ParticleFieldError::AllocationFailed { requested_bytes })?;
    instances.extend_from_slice(&source.instances);
    let visible_instances = allocate_particle_staging(instances.len(), source.budget)?;
    validate_particle_retained_capacities(
        instances.capacity(),
        visible_instances.capacity(),
        source.budget,
    )?;
    let instance_buffer = Arc::new(create_particle_instance_buffer(
        device,
        source.instance_capacity,
    ));
    Ok(ParticleField2d {
        renderer_identity,
        instance_buffer,
        instances,
        visible_instances,
        instance_capacity: source.instance_capacity,
        budget: source.budget,
        statistics: particle_idle_statistics(source.instances.len()),
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

fn clone_scalar_field_for_restore(
    source: &ScalarField,
) -> Result<ScalarField, ScalarFieldTextureError> {
    let requested_bytes = source
        .values()
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ScalarFieldTextureError::DimensionsTooLarge)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(source.values().len())
        .map_err(|_| ScalarFieldTextureError::AllocationFailed { requested_bytes })?;
    values.extend_from_slice(source.values());
    // The retained source already satisfies ScalarField's shape and finite
    // invariants. Keep the impossible defensive branch structured.
    ScalarField::new(source.width(), source.height(), values)
        .map_err(|_| ScalarFieldTextureError::DimensionsTooLarge)
}

fn scalar_value_range_extent(minimum: f32, maximum: f32) -> Option<f32> {
    if !is_portable_shader_source(minimum)
        || !is_portable_shader_source(maximum)
        || maximum <= minimum
    {
        return None;
    }
    let extent = maximum - minimum;
    (is_portable_shader_source(extent) && extent > 0.0).then_some(extent)
}

fn scalar_field_sources_are_portable(field: &ScalarField) -> bool {
    if field
        .values()
        .iter()
        .copied()
        .any(|value| !is_portable_shader_source(value))
    {
        return false;
    }
    let (minimum, maximum) = field.value_range();
    let extent = f64::from(maximum) - f64::from(minimum);
    extent.is_finite() && extent <= f64::from(MAX_PORTABLE_SHADER_VALUE)
}

fn scalar_normalization_is_portable(
    texture: &ScalarFieldTexture,
    minimum: f32,
    value_extent: f32,
) -> bool {
    if !is_portable_shader_source(minimum)
        || !is_portable_shader_source(value_extent)
        || value_extent <= 0.0
    {
        return false;
    }
    let Some(numerator) = shader_interval_sum_range([
        (
            f64::from(texture.source_minimum),
            f64::from(texture.source_maximum),
        ),
        (-f64::from(minimum), -f64::from(minimum)),
    ]) else {
        return false;
    };
    [numerator.0, numerator.1].into_iter().all(|value| {
        let normalized = value / f64::from(value_extent);
        normalized.is_finite() && normalized.abs() <= f64::from(MAX_PORTABLE_SHADER_VALUE)
    })
}

fn compact_scalar_field_for_retention(
    field: ScalarField,
) -> Result<ScalarField, ScalarFieldTextureError> {
    let requested_bytes = field
        .values()
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ScalarFieldTextureError::DimensionsTooLarge)?;
    if field.value_allocation_bytes() <= requested_bytes {
        return Ok(field);
    }
    clone_scalar_field_for_restore(&field)
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

fn validate_scalar_field_device_extent(
    device: &wgpu::Device,
    field: &ScalarField,
) -> Result<wgpu::Extent3d, ScalarFieldTextureError> {
    let extent = scalar_field_extent(field)?;
    let limit = device.limits().max_texture_dimension_2d;
    if extent.width > limit || extent.height > limit {
        return Err(ScalarFieldTextureError::DimensionsTooLarge);
    }
    Ok(extent)
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
    submit_pending_uploads(queue);
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
    submit_pending_uploads(queue);
    Ok(())
}

/// Submits deferred `Queue::write_*` transfers without waiting for GPU completion.
///
/// Native `wgpu` retains staging allocations until a queue submission starts
/// their transfer. Standalone retained-resource mutations call this immediately;
/// render paths instead combine their writes with the draw submission.
pub(super) fn submit_pending_uploads(queue: &wgpu::Queue) {
    let _ = queue.submit([]);
}

#[allow(clippy::too_many_arguments)]
fn validate_scalar_field_texture_region(
    texture_width: usize,
    texture_height: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    value_count: usize,
) -> Result<(), ScalarFieldTextureError> {
    let expected = width
        .checked_mul(height)
        .ok_or(ScalarFieldTextureError::DimensionsTooLarge)?;
    if value_count != expected {
        return Err(ScalarFieldTextureError::InvalidUpdateValueCount);
    }
    if width == 0
        || height == 0
        || x.checked_add(width).is_none_or(|end| end > texture_width)
        || y.checked_add(height).is_none_or(|end| end > texture_height)
    {
        return Err(ScalarFieldTextureError::UpdateRegionOutOfBounds);
    }
    for value in [x, y, width, height] {
        u32::try_from(value).map_err(|_| ScalarFieldTextureError::DimensionsTooLarge)?;
    }
    u32::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(std::mem::size_of::<f32>() as u32))
        .ok_or(ScalarFieldTextureError::DimensionsTooLarge)?;
    Ok(())
}

fn scalar_region_result_range(
    field: &ScalarField,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    values: &[f32],
) -> (f32, f32) {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for row in 0..field.height() {
        for column in 0..field.width() {
            let value = if (x..x + width).contains(&column) && (y..y + height).contains(&row) {
                values[(row - y) * width + (column - x)]
            } else {
                field.values()[row * field.width() + column]
            };
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
    }
    (minimum, maximum)
}

fn create_scalar_field_texture_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    field: ScalarField,
) -> Result<ScalarFieldTexture, ScalarFieldTextureError> {
    validate_scalar_field_device_extent(device, &field)?;
    if !scalar_field_sources_are_portable(&field) {
        return Err(ScalarFieldTextureError::NonPortableValue);
    }
    let field = compact_scalar_field_for_retention(field)?;
    let (source_minimum, source_maximum) = field.value_range();
    let texture = create_scalar_field_texture(device, &field)?;
    upload_scalar_field_texture(queue, &texture, &field)?;
    Ok(ScalarFieldTexture {
        renderer_identity,
        texture,
        field,
        source_minimum,
        source_maximum,
    })
}

fn color_map_lut(color_map: &ColorMap) -> [u8; COLOR_MAP_LUT_SIZE as usize * 4] {
    let mut bytes = [0_u8; COLOR_MAP_LUT_SIZE as usize * 4];
    for index in 0..COLOR_MAP_LUT_SIZE {
        let color = color_map.sample_normalized(index as f32 / (COLOR_MAP_LUT_SIZE - 1) as f32);
        let offset = index as usize * 4;
        bytes[offset..offset + 4].copy_from_slice(
            &color
                .to_array()
                .map(|channel| (channel * 255.0).round() as u8),
        );
    }
    bytes
}

fn create_cached_color_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lut: [u8; COLOR_MAP_LUT_SIZE as usize * 4],
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
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &lut,
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
        lut,
        _texture: texture,
        view,
    }
}

fn create_submitted_particle_unit_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> wgpu::Buffer {
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
    submit_pending_uploads(queue);
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
    physical_to_logical_screen, validate_scale_factor, validate_surface_dimensions,
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
    Image2d, ImageBatch2d, ImageBatchBudget, ImageBatchUploadReport, ImageBudget, ImageError,
    ImageSampling, ImageSprite2d, ImageTexelRect, ImageUploadReport,
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
pub use visualization::{
    LayeredVisualizationError, LayeredVisualizationOptions, LayeredVisualizationReport,
};

#[cfg(test)]
use tessellation::world_vertex;

#[cfg(test)]
mod tests;
