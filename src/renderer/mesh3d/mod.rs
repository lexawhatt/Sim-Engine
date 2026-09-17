#[cfg(test)]
mod gpu_contract;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(super) use gpu_contract::{assert_gpu_depth_contract, assert_gpu_scene_recovery_contract};
mod encoding;
use encoding::*;
mod api;
mod frame;
#[cfg(test)]
#[path = "tests/frame_upload.rs"]
mod frame_upload_tests;
mod pipelines;
mod report;
mod upload_changes;
pub use report::{
    Mesh3dObjectError, Mesh3dRenderError, Mesh3dRenderReport, Mesh3dResourceError,
    Scene3dRestoreReport,
};
mod allocation;
use allocation::*;
mod scene_restore;
use scene_restore::*;
mod upload_layout;
use upload_layout::*;
mod resources;
use resources::*;
mod vertex_proof;
use vertex_proof::*;
mod edge_proof;
use edge_proof::*;
mod shader_math;
use shader_math::*;
use shader_math::{rounded_f32_add_range, rounded_f32_product_range};

use super::*;
use crate::{
    Camera3d, LogicalPixels, Mesh3d, MeshStyle3d, PhysicalPerLogical, Transform3d, Vec3,
    WireframeStyle3d,
};
use std::sync::atomic::{AtomicU64, Ordering};

mod validation;
use validation::validate_points_for_policy;
pub use validation::{Mesh3dSurfaceError, SurfaceRasterization3d};

mod batch;
#[cfg(test)]
#[path = "tests/batch.rs"]
mod batch_tests;
#[cfg(test)]
#[path = "tests/edge_upload.rs"]
mod edge_upload_tests;

mod surface;
pub use surface::{Mesh3dPreflightReport, Mesh3dRenderBudget};
use surface::{SurfaceClipEdge, SurfaceClipVertex, SurfaceFrame};

mod texture;
use texture::{MeshTextureRenderer, MeshUvGpu};
mod restoration;
use restoration::restore_retained_mesh;
mod colors;
use crate::mesh3d::{SurfaceAlphaMode3d, SurfaceSidedness3d};
use colors::MeshColorGpu;
mod normals;
use normals::MeshNormalGpu;
mod lighting;
use crate::mesh3d::{Fog3d, Lighting3d, SurfaceLighting3d};
use lighting::{SurfaceEnvironmentGpu, SurfaceLightingVertex, SurfaceTransport};
mod surface_pipeline;
use surface_pipeline::{SurfaceLayout, SurfacePipelines, create_surface_pipelines};
#[cfg(test)]
#[path = "tests/lighting.rs"]
mod lighting_tests;
mod material;
#[cfg(test)]
#[path = "tests/material.rs"]
mod material_tests;
#[cfg(test)]
#[path = "tests/native_acceptance.rs"]
mod native_acceptance_tests;

#[cfg(test)]
#[path = "tests/material_resource.rs"]
mod material_resource_tests;
pub use texture::{
    Texture3d, Texture3dError, Texture3dOptions, Texture3dUpdateBudget,
    Texture3dUpdateBudgetResource, Texture3dUpdateError, Texture3dUpdateReport, TextureMaterial3d,
    TextureMipmaps3d,
};

#[cfg(test)]
use crate::{MeshEdge3d, Projection3d, Rotation3d, SurfaceStyle3d, WorldLength};

#[cfg(test)]
#[path = "tests/lifetime.rs"]
mod lifetime_tests;

#[cfg(test)]
#[path = "tests/vertex_color.rs"]
mod vertex_color_tests;

#[cfg(test)]
#[path = "tests/color_budget.rs"]
mod color_budget_tests;

#[cfg(test)]
#[path = "tests/timing.rs"]
mod timing_tests;
#[cfg(test)]
pub(super) use timing_tests::assert_gpu_scene_timing_and_upload_contract;

#[cfg(test)]
fn logical(value: f32) -> LogicalPixels {
    LogicalPixels::new(value).unwrap()
}

#[cfg(test)]
fn physical_per_logical(value: f32) -> PhysicalPerLogical {
    PhysicalPerLogical::new(value).unwrap()
}

#[cfg(test)]
fn world(value: f32) -> WorldLength {
    WorldLength::new(value).unwrap()
}

const INITIAL_INSTANCE_CAPACITY: usize = 16;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
// Hidden fragments are biased one implementation depth unit away from the
// camera; visible fragments are biased one unit toward it and are rendered
// last. This defines a two-unit coplanar tolerance: exact/shared surface edges
// resolve solid, while occlusion must exceed that tolerance to resolve dashed.
const HIDDEN_EDGE_DEPTH_BIAS: i32 = 1;
const VISIBLE_EDGE_DEPTH_BIAS: i32 = -1;
static NEXT_SCENE3D_ID: AtomicU64 = AtomicU64::new(1);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshVertexGpu {
    position: [f32; 3],
}

impl MeshVertexGpu {
    const ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshInstanceGpu {
    model_row_0: [f32; 4],
    model_row_1: [f32; 4],
    model_row_2: [f32; 4],
    color: [f32; 4],
    surface: [f32; 4],
    normal_row_0: [f32; 4],
    normal_row_1: [f32; 4],
    normal_row_2: [f32; 4],
    uv_transform: [f32; 4],
}

impl MeshInstanceGpu {
    const ATTRIBUTES: [wgpu::VertexAttribute; 9] = wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 7 => Float32x4, 9 => Float32x4, 10 => Float32x4, 11 => Float32x4, 12 => Float32x4];
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &Self::ATTRIBUTES,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Camera3dUniform {
    clip_row_0: [f32; 4],
    clip_row_1: [f32; 4],
    clip_row_2: [f32; 4],
    clip_row_3: [f32; 4],
    viewport: [f32; 4],
    environment: SurfaceEnvironmentGpu,
}

impl Camera3dUniform {
    fn new(
        camera: Camera3d,
        width: u32,
        height: u32,
        scale_factor: PhysicalPerLogical,
    ) -> Result<Self, Mesh3dRenderError> {
        let rows = camera
            .world_to_clip_rows()
            .map_err(|_| Mesh3dRenderError::InvalidGeometryTransform)?;
        let uniform = Self {
            clip_row_0: rows[0],
            clip_row_1: rows[1],
            clip_row_2: rows[2],
            clip_row_3: rows[3],
            viewport: [width as f32, height as f32, scale_factor.get(), 0.0],
            environment: SurfaceEnvironmentGpu::default(),
        };
        uniform
            .rows()
            .into_iter()
            .flatten()
            .chain(uniform.viewport)
            .all(is_portable_shader_source)
            .then_some(uniform)
            .ok_or(Mesh3dRenderError::InvalidGeometryTransform)
    }

    fn rows(self) -> [[f32; 4]; 4] {
        [
            self.clip_row_0,
            self.clip_row_1,
            self.clip_row_2,
            self.clip_row_3,
        ]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshEdgeGpu {
    start: [f32; 3],
    end: [f32; 3],
}

impl MeshEdgeGpu {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &Self::ATTRIBUTES,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EdgeObjectUniform {
    model_row_0: [f32; 4],
    model_row_1: [f32; 4],
    model_row_2: [f32; 4],
    visible_color: [f32; 4],
    hidden_color: [f32; 4],
    edge_style: [f32; 4],
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ClipProbeInputGpu {
    start_clip: [f32; 4],
    end_clip: [f32; 4],
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ClipProbeOutputGpu {
    start_clip: [f32; 4],
    end_clip: [f32; 4],
    range: [f32; 2],
    visible: u32,
    padding: u32,
}

pub(super) struct Mesh3dRenderer {
    dynamic_scratch: dynamic::DynamicMesh3dScratch,
    textures: MeshTextureRenderer,
    pipeline: SurfacePipelines,
    colored_pipeline: SurfacePipelines,
    colored_clipped_pipeline: SurfacePipelines,
    clipped_color_buffer: Option<wgpu::Buffer>,
    clipped_color_capacity: usize,
    clipped_lighting_buffer: Option<wgpu::Buffer>,
    clipped_lighting_capacity: usize,
    clipped_surface_pipeline: SurfacePipelines,
    clipped_surface_buffer: Option<wgpu::Buffer>,
    clipped_surface_capacity: usize,
    surface_frame: SurfaceFrame,
    clipped_visible_edge_pipeline: wgpu::RenderPipeline,
    clipped_hidden_edge_pipeline: wgpu::RenderPipeline,
    clipped_edge_buffer: Option<wgpu::Buffer>,
    clipped_edge_capacity: usize,
    visible_edge_pipeline: wgpu::RenderPipeline,
    hidden_edge_pipeline: wgpu::RenderPipeline,
    camera_uniform_buffer: wgpu::Buffer,
    previous_camera: Option<Camera3dUniform>,
    camera_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    instances: Vec<MeshInstanceGpu>,
    edge_object_layout: wgpu::BindGroupLayout,
    edge_object_bind_group: wgpu::BindGroup,
    edge_object_buffer: wgpu::Buffer,
    edge_object_stride: usize,
    edge_object_capacity: usize,
    edge_object_bytes: Vec<u8>,
}

/// Immutable retained GPU buffers for one validated [`Mesh3d`].
///
/// The resource belongs to the renderer that created it and retains the core
/// topology for explicit device-loss restoration.
#[derive(Clone)]
pub struct RetainedMesh3d {
    renderer_identity: Arc<()>,
    vertex_buffer: Arc<wgpu::Buffer>,
    index_buffer: Option<Arc<wgpu::Buffer>>,
    edge_buffer: Option<Arc<wgpu::Buffer>>,
    texture_coordinate_buffer: Option<Arc<wgpu::Buffer>>,
    color_buffer: Option<Arc<wgpu::Buffer>>,
    normal_buffer: Option<Arc<wgpu::Buffer>>,
    material: Option<TextureMaterial3d>,
    source: Mesh3d,
    index_count: u32,
    edge_count: u32,
    gpu_allocation_bytes: usize,
    budget: Mesh3dUploadBudget,
    allocation: Mesh3dUploadLayout,
}

impl RetainedMesh3d {
    /// Returns a material-free handle sharing the same immutable topology and
    /// attribute buffers. No CPU geometry is copied and no GPU resource is
    /// allocated or uploaded. Existing handles and scene objects are unchanged.
    /// UVs, normals, vertex colors, display edges and reserved capacity remain
    /// available for later rebinding. Device provenance is unchanged; a stale
    /// handle still needs restoration before rendering on a replacement device.
    pub fn without_material(&self) -> Self {
        let mut mesh = self.clone();
        mesh.material = None;
        mesh
    }

    /// Returns the optional shared texture/filter/tint material for this handle.
    /// Mesh clones share buffers while retaining independent material selection.
    pub const fn material(&self) -> Option<&TextureMaterial3d> {
        self.material.as_ref()
    }
    /// Returns this revision's upload and restoration capacity limits.
    pub const fn budget(&self) -> Mesh3dUploadBudget {
        self.budget
    }
    /// Returns the immutable core topology retained for recovery.
    pub fn source(&self) -> &Mesh3d {
        &self.source
    }

    /// Returns the retained triangle count.
    pub fn triangle_count(&self) -> usize {
        self.source.triangle_count()
    }

    /// Returns allocated vertex, optional UV, triangle-index and display-edge
    /// buffer capacity bytes, including dynamic reserve beyond live geometry.
    /// Shared material texels are reported separately by `Texture3d`.
    pub const fn gpu_allocation_bytes(&self) -> usize {
        self.gpu_allocation_bytes
    }

    /// Returns CPU topology/UV bytes retained for device-loss restoration.
    /// Shared material pixels are reported separately by `Texture3d`.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.source.recovery_memory_bytes()
    }
}

mod dynamic;
mod objects;
mod upload;
pub use dynamic::{
    DynamicMesh3dBudget, DynamicMesh3dBudgetResource, DynamicMesh3dError, DynamicMesh3dUpdateReport,
};
pub use objects::{
    Mesh3dInstance, Object3dId, Scene3d, Scene3dBudget, Scene3dBudgetResource, Scene3dError,
    Scene3dMeshUpdateReport, Scene3dStatistics,
};
pub use upload::{Mesh3dUploadBudget, Mesh3dUploadBudgetResource, Mesh3dUploadReport};

/// Offscreen color and depth attachments for stereometry rendering.
pub struct RenderTarget3d {
    renderer_identity: Arc<()>,
    color: RenderTarget2d,
    _depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    logical_viewport: LogicalViewport,
    pixels_per_logical: PhysicalPerLogical,
}

impl RenderTarget3d {
    /// Returns target width in physical texture pixels.
    pub fn width(&self) -> u32 {
        self.color.width()
    }

    /// Returns target height in physical texture pixels.
    pub fn height(&self) -> u32 {
        self.color.height()
    }

    /// Returns the logical viewport represented by the offscreen texture.
    pub const fn logical_viewport(&self) -> LogicalViewport {
        self.logical_viewport
    }

    /// Returns target texels per logical screen pixel.
    pub const fn pixels_per_logical(&self) -> PhysicalPerLogical {
        self.pixels_per_logical
    }

    /// Returns the composable 2D color attachment.
    pub const fn color_target(&self) -> &RenderTarget2d {
        &self.color
    }

    /// Returns nominal color plus depth texel-storage bytes, excluding opaque
    /// backend alignment and resource metadata.
    pub fn allocation_bytes(&self) -> usize {
        self.color.allocation_bytes().saturating_add(
            (self.width() as usize)
                .saturating_mul(self.height() as usize)
                .saturating_mul(4),
        )
    }
}

fn validate_mesh_style(mesh: &RetainedMesh3d, style: MeshStyle3d) -> Result<(), Scene3dError> {
    validate_mesh_source_style(mesh.source(), style)
}

fn validate_mesh_source_style(source: &Mesh3d, style: MeshStyle3d) -> Result<(), Scene3dError> {
    let has_surface = style.surface_style().is_some() && source.triangle_count() > 0;
    if has_surface
        && style
            .surface_style()
            .is_some_and(|surface| surface.lighting() == SurfaceLighting3d::Lambert)
        && source.normals().is_empty()
    {
        return Err(Scene3dError::MissingNormals);
    }
    let has_wireframe = style.wireframe_style().is_some() && !source.display_edges().is_empty();
    (has_surface || has_wireframe)
        .then_some(())
        .ok_or(Scene3dError::StyleHasNoMatchingGeometry)
}

fn validate_scene3d_transform(transform: Transform3d) -> Result<(), Scene3dError> {
    transform
        .model_rows()
        .ok()
        .is_some_and(|rows| rows.into_iter().flatten().all(is_portable_shader_source))
        .then_some(())
        .ok_or(Scene3dError::InvalidTransform)
}

fn target_pixels_per_logical(
    width: u32,
    height: u32,
    viewport: LogicalViewport,
) -> Option<PhysicalPerLogical> {
    let horizontal = width as f64 / viewport.width() as f64;
    let vertical = height as f64 / viewport.height() as f64;
    if !horizontal.is_finite()
        || !vertical.is_finite()
        || horizontal <= 0.0
        || vertical <= 0.0
        || !aspect_matches(horizontal, vertical)
        || horizontal > f32::MAX as f64
    {
        return None;
    }
    PhysicalPerLogical::new(horizontal as f32).ok()
}

fn aspect_matches(left: f64, right: f64) -> bool {
    if !left.is_finite() || !right.is_finite() || left <= 0.0 || right <= 0.0 {
        return false;
    }
    let scale = left.abs().max(right.abs());
    (left - right).abs() <= scale * 1.0e-5
}

fn validate_camera_target_aspect(
    camera: Camera3d,
    viewport: LogicalViewport,
) -> Result<(), Mesh3dRenderError> {
    aspect_matches(
        f64::from(camera.projection().aspect_ratio()),
        f64::from(viewport.width()) / f64::from(viewport.height()),
    )
    .then_some(())
    .ok_or(Mesh3dRenderError::CameraTargetAspectMismatch)
}

fn validate_target_identity(
    renderer_identity: &Arc<()>,
    target: &RenderTarget3d,
) -> Result<(), Mesh3dRenderError> {
    (Arc::ptr_eq(renderer_identity, &target.renderer_identity)
        && Arc::ptr_eq(renderer_identity, &target.color.renderer_identity))
    .then_some(())
    .ok_or(Mesh3dRenderError::RendererMismatch)
}

fn validate_mesh_identity(
    renderer_identity: &Arc<()>,
    mesh: &RetainedMesh3d,
) -> Result<(), Mesh3dRenderError> {
    (Arc::ptr_eq(renderer_identity, &mesh.renderer_identity)
        && mesh
            .material()
            .is_none_or(|material| material.texture().belongs_to(renderer_identity)))
    .then_some(())
    .ok_or(Mesh3dRenderError::RendererMismatch)
}
