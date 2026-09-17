use std::{
    borrow::Cow,
    error::Error,
    fmt,
    ops::Range,
    sync::{Arc, Mutex},
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
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
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

mod validation2d;
#[cfg(test)]
use validation2d::{
    GEOMETRY_VALIDATION_CACHE_CAPACITY, dynamic_triangle_topology_is_portable,
    geometry_vertex_centers_are_portable, logical_stroke_branches_are_stable,
    shader_interval_sum_is_safe, shader_stroke_screen_bounds,
    tessellated_triangle_topology_is_portable, tessellated_vertex_clip_ranges,
};
use validation2d::{
    GeometryValidationCache, GeometryValidationSource, center_axis,
    clip_triangle_has_stable_signed_area, clip_triangle_is_wholly_outside, geometry_is_safe_for,
    geometry_is_safe_for_cached, geometry_sources_are_portable, interval_products_f64,
    is_nonzero_subnormal, logical_stroke_vertex_screen_ranges, range_abs, range_dot, range_negate,
    range_vector_add, range_vector_scale, rounded_f32_add_range, rounded_f32_product_range,
    screen_ranges_to_clip, shader_clip_interval_is_safe, shader_direction_dot_range,
    shader_interval_sum_range, shader_relative_component_bounds, shader_world_dot_range,
    stroke_safe_unit_range, tessellated_vertex_base_screen_ranges, triangle_position_sources_equal,
};

mod outcomes;
pub use outcomes::{
    PreparedSceneError, PreparedSceneRenderError, RenderReport, RenderStatus,
    RendererConfigurationError, RendererCoordinateError, RendererFrameError, RendererFrameMetrics,
    RendererInitError, RendererSurfaceStatus,
};

/// Immutable scene geometry tessellated and uploaded once for repeated drawing.
///
/// Camera movement, zoom, rotation, projection tilt, viewport resizing, and DPI
/// changes do not invalidate prepared geometry. Shape, style, gradient, layer,
/// or clip changes require preparing a replacement. A prepared scene can only be
/// rendered by the [`WgpuRenderer`] that created it. Use
/// [`WgpuRenderer::restore_prepared_scene`] to recreate its GPU buffer for a
/// replacement renderer after device loss. Portable shader arithmetic is proved
/// once per exact camera/viewport uniform and retained in a bounded cache.
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
    geometry_validation_cache: GeometryValidationCache,
}

/// Immutable prepared geometry whose positions are logical screen pixels.
///
/// This distinct wrapper prevents a world camera from being supplied to fixed
/// UI geometry. It is created by [`WgpuRenderer::prepare_screen_scene`].
pub struct PreparedScreenScene {
    scene: PreparedScene,
}

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
    gpu_timing: gpu_timing::GpuTimingCollector,
    gpu_timing_requested: bool,
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    heatmap_uniform_buffer: wgpu::Buffer,
    heatmap_bind_group_layout: wgpu::BindGroupLayout,
    color_map_cache: Option<CachedColorMap>,
    frame_cache: frame::FrameCache,
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
    mesh.geometry_validation_cache.clear();
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
        geometry_validation_cache: GeometryValidationCache::default(),
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
        geometry_validation_cache: GeometryValidationCache::default(),
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
        geometry_validation_cache: GeometryValidationCache::default(),
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
mod dynamic_mesh;
mod exact_markers;
mod frame;
mod geometry;
mod glyph;
mod gpu_timing;
mod image;
mod initialization;
mod mesh3d;
mod particles;
mod pipelines;
mod scalar_fields;
mod scenes;
mod targets;
mod tessellation;
#[cfg(feature = "text")]
mod text;
#[cfg(feature = "text")]
pub use text::{
    PreparedTextError, TextAtlas2d, TextAtlasBudget, TextError, TextRun2d, TextUpdateReport,
};
mod visualization;
use config::{
    MultisampleTarget, create_multisample_target, logical_to_physical_screen,
    physical_to_logical_screen, validate_scale_factor, validate_surface_dimensions,
};
pub use config::{RendererPresentMode, RendererSurfacePresentMode, WgpuRendererOptions};
pub use dynamic_mesh::{
    DynamicMesh2d, DynamicMeshBudget, DynamicMeshBudgetResource, DynamicMeshError,
    DynamicMeshRenderError, DynamicMeshUpdateReport, DynamicVertex2d,
};
pub use frame::{
    FrameBudget, FrameBudgetResource, FrameCacheBudget, FrameCacheStatistics, FrameComposer,
    FrameComposerError, FramePassOptions, FrameReport, FrameSourceKind, FrameSourceStatistics,
    FrameStatistics,
};
pub use glyph::{
    GlyphAtlas2d, GlyphAtlasBudget, GlyphAtlasEntry, GlyphError, GlyphId, GlyphRun2d,
    GlyphRunBounds, GlyphRunBudget, GlyphRunStatistics, GlyphRunUploadReport, GlyphUploadReport,
    PositionedGlyph2d,
};
pub use gpu_timing::{
    GpuTimingBatch, GpuTimingId, GpuTimingSample, GpuTimingSource, GpuTimingStatistics,
    GpuTimingStatus,
};
pub use image::{
    Image2d, ImageBatch2d, ImageBatchBudget, ImageBatchPlacement, ImageBatchUploadReport,
    ImageBudget, ImageError, ImageSampling, ImageSprite2d, ImageTexelRect, ImageUploadReport,
};
use image::{ImageRenderer, ImageUniform};
use mesh3d::Mesh3dRenderer;
pub use mesh3d::{
    DynamicMesh3dBudget, DynamicMesh3dBudgetResource, DynamicMesh3dError,
    DynamicMesh3dUpdateReport, Mesh3dInstance, Mesh3dObjectError, Mesh3dPreflightReport,
    Mesh3dRenderBudget, Mesh3dRenderError, Mesh3dRenderReport, Mesh3dResourceError,
    Mesh3dSurfaceError, Mesh3dUploadBudget, Mesh3dUploadBudgetResource, Mesh3dUploadReport,
    Object3dId, RenderTarget3d, RetainedMesh3d, Scene3d, Scene3dBudget, Scene3dBudgetResource,
    Scene3dError, Scene3dMeshUpdateReport, Scene3dRestoreReport, Scene3dStatistics,
    SurfaceRasterization3d, Texture3d, Texture3dError, Texture3dOptions, Texture3dUpdateBudget,
    Texture3dUpdateBudgetResource, Texture3dUpdateError, Texture3dUpdateReport, TextureMaterial3d,
    TextureMipmaps3d,
};
pub use particles::{
    ParticleBudgetError, ParticleField2d, ParticleFieldError, ParticleFieldRenderError,
    ParticleFieldUpdateReport, ParticleRenderBudget, ParticleStatistics,
};
use pipelines::{PipelineResources, create_pipeline};
pub use scalar_fields::{
    ScalarFieldRenderError, ScalarFieldTexture, ScalarFieldTextureError, ScalarFieldUploadReport,
};
pub use targets::{RenderTarget2d, RenderTargetError, TrailBuffer2d};
use tessellation::{
    logical_viewport_scissor, offset_scissor, screen_clip_to_scissor, tessellate_scene,
};
use visualization::{CachedColorMap, CompositionPipelines};
pub use visualization::{
    LayeredVisualizationError, LayeredVisualizationOptions, LayeredVisualizationReport,
};

#[cfg(test)]
use tessellation::world_vertex;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_support;
