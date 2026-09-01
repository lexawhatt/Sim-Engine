use super::*;
use crate::{
    Camera3d, LogicalPixels, Mesh3d, MeshStyle3d, PhysicalPerLogical, Transform3d, Vec3,
    WireframeStyle3d,
};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use crate::{MeshEdge3d, Projection3d, Rotation3d, SurfaceStyle3d, WorldLength};

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
}

impl MeshInstanceGpu {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4];
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
        Ok(Self {
            clip_row_0: rows[0],
            clip_row_1: rows[1],
            clip_row_2: rows[2],
            clip_row_3: rows[3],
            viewport: [width as f32, height as f32, scale_factor.get(), 0.0],
        })
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
    pipeline: wgpu::RenderPipeline,
    visible_edge_pipeline: wgpu::RenderPipeline,
    hidden_edge_pipeline: wgpu::RenderPipeline,
    camera_uniform_buffer: wgpu::Buffer,
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

impl Mesh3dRenderer {
    pub(super) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sim-engine retained 3D mesh shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("mesh3d.wgsl"))),
        });
        let camera_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine 3D camera uniform buffer"),
            size: std::mem::size_of::<Camera3dUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine 3D camera bind group layout"),
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
            label: Some("sim-engine 3D camera bind group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buffer.as_entire_binding(),
            }],
        });
        let edge_object_stride = align_to(
            std::mem::size_of::<EdgeObjectUniform>(),
            device.limits().min_uniform_buffer_offset_alignment as usize,
        );
        let edge_object_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sim-engine 3D edge object bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<EdgeObjectUniform>() as u64,
                        ),
                    },
                    count: None,
                }],
            });
        let edge_object_buffer =
            create_edge_object_buffer(device, edge_object_stride, INITIAL_INSTANCE_CAPACITY);
        let edge_object_bind_group =
            create_edge_object_bind_group(device, &edge_object_layout, &edge_object_buffer);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-engine retained 3D mesh pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sim-engine retained 3D mesh pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("mesh3d_vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(MeshVertexGpu::LAYOUT), Some(MeshInstanceGpu::LAYOUT)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("mesh3d_fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let edge_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-engine 3D edge pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&edge_object_layout)],
            immediate_size: 0,
        });
        let visible_edge_pipeline = create_edge_pipeline(
            device,
            &shader,
            &edge_pipeline_layout,
            format,
            wgpu::CompareFunction::LessEqual,
            "mesh3d_visible_edge_fs_main",
            "sim-engine visible 3D edge pipeline",
        );
        let hidden_edge_pipeline = create_edge_pipeline(
            device,
            &shader,
            &edge_pipeline_layout,
            format,
            wgpu::CompareFunction::Greater,
            "mesh3d_hidden_edge_fs_main",
            "sim-engine hidden 3D edge pipeline",
        );
        Self {
            pipeline,
            visible_edge_pipeline,
            hidden_edge_pipeline,
            camera_uniform_buffer,
            camera_bind_group,
            instance_buffer: create_instance_buffer(device, INITIAL_INSTANCE_CAPACITY),
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            instances: Vec::with_capacity(INITIAL_INSTANCE_CAPACITY),
            edge_object_layout,
            edge_object_bind_group,
            edge_object_buffer,
            edge_object_stride,
            edge_object_capacity: INITIAL_INSTANCE_CAPACITY,
            edge_object_bytes: Vec::with_capacity(edge_object_stride * INITIAL_INSTANCE_CAPACITY),
        }
    }

    fn ensure_instance_capacity(
        &mut self,
        device: &wgpu::Device,
        instance_count: usize,
    ) -> Result<(), Mesh3dRenderError> {
        if instance_count <= self.instance_capacity {
            return Ok(());
        }
        let capacity = instance_count
            .checked_next_power_of_two()
            .filter(|capacity| buffer_capacity_fits::<MeshInstanceGpu>(device, *capacity))
            .ok_or(Mesh3dRenderError::InstanceCapacityTooLarge)?;
        self.instance_buffer = create_instance_buffer(device, capacity);
        self.instance_capacity = capacity;
        self.instances
            .reserve(capacity.saturating_sub(self.instances.capacity()));
        Ok(())
    }

    fn ensure_edge_object_capacity(
        &mut self,
        device: &wgpu::Device,
        object_count: usize,
    ) -> Result<(), Mesh3dRenderError> {
        if object_count <= self.edge_object_capacity {
            return Ok(());
        }
        let capacity = object_count
            .checked_next_power_of_two()
            .filter(|capacity| {
                capacity
                    .checked_mul(self.edge_object_stride)
                    .is_some_and(|bytes| {
                        bytes as u64 <= device.limits().max_buffer_size
                            && bytes <= u32::MAX as usize
                    })
            })
            .ok_or(Mesh3dRenderError::InstanceCapacityTooLarge)?;
        self.edge_object_buffer =
            create_edge_object_buffer(device, self.edge_object_stride, capacity);
        self.edge_object_bind_group = create_edge_object_bind_group(
            device,
            &self.edge_object_layout,
            &self.edge_object_buffer,
        );
        self.edge_object_capacity = capacity;
        self.edge_object_bytes.reserve(
            capacity
                .saturating_mul(self.edge_object_stride)
                .saturating_sub(self.edge_object_bytes.capacity()),
        );
        Ok(())
    }
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
    source: Mesh3d,
    index_count: u32,
    edge_count: u32,
    gpu_allocation_bytes: usize,
}

impl RetainedMesh3d {
    /// Returns the immutable core topology retained for recovery.
    pub fn source(&self) -> &Mesh3d {
        &self.source
    }

    /// Returns the retained triangle count.
    pub fn triangle_count(&self) -> usize {
        self.source.triangle_count()
    }

    /// Returns exact vertex, triangle-index, and display-edge buffer bytes.
    pub const fn gpu_allocation_bytes(&self) -> usize {
        self.gpu_allocation_bytes
    }

    /// Returns CPU topology bytes retained for device-loss restoration.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.source.recovery_memory_bytes()
    }
}

/// Stable scene handle for a retained 3D object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Object3dId {
    scene_id: u64,
    object_id: u64,
}

impl Object3dId {
    /// Returns the stable scene-local numeric value for diagnostics.
    ///
    /// Equality and hashing also include private scene provenance. Store the
    /// complete handle, rather than only this number, in host object maps.
    pub const fn get(self) -> u64 {
        self.object_id
    }
}

/// One retained mesh reference plus its independent model transform and style.
#[derive(Clone)]
pub struct Mesh3dInstance {
    id: Object3dId,
    mesh: RetainedMesh3d,
    transform: Transform3d,
    style: MeshStyle3d,
    visible: bool,
}

impl Mesh3dInstance {
    fn new(
        id: Object3dId,
        mesh: &RetainedMesh3d,
        transform: Transform3d,
        style: MeshStyle3d,
    ) -> Self {
        Self {
            id,
            mesh: mesh.clone(),
            transform,
            style,
            visible: true,
        }
    }

    /// Returns the stable scene object identifier.
    pub const fn id(&self) -> Object3dId {
        self.id
    }

    /// Returns the retained GPU mesh reference.
    pub const fn mesh(&self) -> &RetainedMesh3d {
        &self.mesh
    }

    /// Returns this object's model-to-world transform.
    pub const fn transform(&self) -> Transform3d {
        self.transform
    }

    /// Returns its extensible surface/edge material bundle.
    pub const fn style(&self) -> MeshStyle3d {
        self.style
    }

    /// Returns whether this object participates in drawing and render counts.
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    /// Returns explicit edge presentation when enabled for this object.
    pub const fn wireframe(&self) -> Option<WireframeStyle3d> {
        self.style.wireframe_style()
    }
}

/// Ready 3D visual state rendered with one camera and depth attachment.
pub struct Scene3d {
    scene_id: u64,
    background: Color,
    instances: Vec<Mesh3dInstance>,
    next_object_id: u64,
}

impl Scene3d {
    /// Creates an empty scene with a normalized opaque clear color.
    pub fn new(background: Color) -> Result<Self, Scene3dError> {
        if !background.is_normalized() || background.alpha() != 1.0 {
            return Err(Scene3dError::InvalidBackground);
        }
        let scene_id = NEXT_SCENE3D_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| Scene3dError::SceneIdExhausted)?;
        Ok(Self {
            scene_id,
            background,
            instances: Vec::new(),
            next_object_id: 0,
        })
    }

    /// Adds a retained object and returns a stable handle for later updates.
    pub fn try_push(
        &mut self,
        mesh: &RetainedMesh3d,
        transform: Transform3d,
        style: MeshStyle3d,
    ) -> Result<Object3dId, Scene3dError> {
        validate_mesh_style(mesh, style)?;
        let id = Object3dId {
            scene_id: self.scene_id,
            object_id: self.next_object_id,
        };
        self.next_object_id = self
            .next_object_id
            .checked_add(1)
            .ok_or(Scene3dError::ObjectIdExhausted)?;
        self.instances
            .push(Mesh3dInstance::new(id, mesh, transform, style));
        Ok(id)
    }

    /// Returns the finite opaque target clear color.
    pub const fn background(&self) -> Color {
        self.background
    }

    /// Returns objects in host insertion order; depth determines visibility.
    pub fn instances(&self) -> &[Mesh3dInstance] {
        &self.instances
    }

    /// Returns the number of independently transformed objects.
    pub fn object_count(&self) -> usize {
        self.instances.len()
    }

    /// Returns objects currently participating in rendering.
    pub fn visible_object_count(&self) -> usize {
        self.instances
            .iter()
            .filter(|instance| instance.visible)
            .count()
    }

    /// Replaces one object's model transform without touching retained topology.
    pub fn set_transform(
        &mut self,
        object_id: Object3dId,
        transform: Transform3d,
    ) -> Result<(), Scene3dError> {
        let instance = self.instance_mut(object_id)?;
        instance.transform = transform;
        Ok(())
    }

    /// Replaces one object's complete extensible visual material bundle.
    pub fn set_style(
        &mut self,
        object_id: Object3dId,
        style: MeshStyle3d,
    ) -> Result<(), Scene3dError> {
        let instance = self.instance_mut(object_id)?;
        validate_mesh_style(&instance.mesh, style)?;
        instance.style = style;
        Ok(())
    }

    /// Shows or hides one object without releasing retained topology.
    pub fn set_visible(
        &mut self,
        object_id: Object3dId,
        visible: bool,
    ) -> Result<(), Scene3dError> {
        let instance = self.instance_mut(object_id)?;
        instance.visible = visible;
        Ok(())
    }

    /// Enables or disables explicit visible/hidden edges for one object.
    pub fn set_wireframe(
        &mut self,
        object_id: Object3dId,
        wireframe: Option<WireframeStyle3d>,
    ) -> Result<(), Scene3dError> {
        let instance = self.instance_mut(object_id)?;
        let surface = instance.style.surface_style();
        let style = match (surface, wireframe) {
            (Some(surface), Some(wireframe)) => {
                MeshStyle3d::surface(surface).with_wireframe(wireframe)
            }
            (Some(surface), None) => MeshStyle3d::surface(surface),
            (None, Some(wireframe)) => MeshStyle3d::wireframe(wireframe),
            (None, None) => return Err(Scene3dError::EmptyStyle),
        };
        validate_mesh_style(&instance.mesh, style)?;
        instance.style = style;
        Ok(())
    }

    fn instance_mut(&mut self, object_id: Object3dId) -> Result<&mut Mesh3dInstance, Scene3dError> {
        if object_id.scene_id != self.scene_id {
            return Err(Scene3dError::ObjectNotFound { object_id });
        }
        self.instances
            .iter_mut()
            .find(|instance| instance.id == object_id)
            .ok_or(Scene3dError::ObjectNotFound { object_id })
    }
}

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

    /// Returns color plus depth allocation bytes.
    pub fn allocation_bytes(&self) -> usize {
        self.color.allocation_bytes().saturating_add(
            (self.width() as usize)
                .saturating_mul(self.height() as usize)
                .saturating_mul(4),
        )
    }
}

/// Rejection reason for 3D scene visual state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scene3dError {
    /// The target clear color must be normalized and opaque.
    InvalidBackground,
    /// Surface and wireframe were both disabled.
    EmptyStyle,
    /// The selected surface/edge modes have no corresponding mesh topology.
    StyleHasNoMatchingGeometry,
    /// No more process-unique scene identifiers can be allocated.
    SceneIdExhausted,
    /// No more stable object identifiers can be allocated.
    ObjectIdExhausted,
    /// An object update referenced a missing stable handle.
    ObjectNotFound {
        /// Stable handle that is absent from this scene.
        object_id: Object3dId,
    },
}

impl fmt::Display for Scene3dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBackground => write!(formatter, "3D scene background must be opaque"),
            Self::EmptyStyle => write!(formatter, "3D object requires a surface or wireframe"),
            Self::StyleHasNoMatchingGeometry => write!(
                formatter,
                "3D object style has no matching triangles or display edges"
            ),
            Self::SceneIdExhausted => write!(formatter, "3D scene identifiers exhausted"),
            Self::ObjectIdExhausted => write!(formatter, "3D scene object identifiers exhausted"),
            Self::ObjectNotFound { object_id } => write!(
                formatter,
                "3D scene object {} does not exist in this scene",
                object_id.get()
            ),
        }
    }
}

impl Error for Scene3dError {}

fn validate_mesh_style(mesh: &RetainedMesh3d, style: MeshStyle3d) -> Result<(), Scene3dError> {
    let has_surface = style.surface_style().is_some() && mesh.triangle_count() > 0;
    let has_wireframe =
        style.wireframe_style().is_some() && !mesh.source().display_edges().is_empty();
    (has_surface || has_wireframe)
        .then_some(())
        .ok_or(Scene3dError::StyleHasNoMatchingGeometry)
}

/// Resource creation or ownership failure for retained 3D rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mesh3dResourceError {
    /// Vertex, index, or edge data exceeds the active device's buffer-size limit.
    CapacityTooLarge,
    /// Host staging memory could not be reserved after capacity preflight.
    HostAllocationFailed {
        /// Bytes requested for the rejected staging buffer.
        requested_bytes: u64,
    },
    /// Color/depth target creation failed.
    Target(RenderTargetError),
    /// Physical texture and logical viewport aspect ratios differ.
    InvalidViewportAspect,
}

impl fmt::Display for Mesh3dResourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityTooLarge => write!(formatter, "3D mesh exceeds GPU buffer limits"),
            Self::HostAllocationFailed { requested_bytes } => write!(
                formatter,
                "3D mesh could not reserve a {requested_bytes}-byte host staging buffer"
            ),
            Self::Target(error) => write!(formatter, "3D target creation failed: {error}"),
            Self::InvalidViewportAspect => write!(
                formatter,
                "3D target texture and logical viewport must have one aspect ratio"
            ),
        }
    }
}

impl Error for Mesh3dResourceError {}

/// Failure while encoding a depth-tested 3D scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mesh3dRenderError {
    /// A mesh or target belongs to another renderer/device identity.
    RendererMismatch,
    /// A finite model/camera configuration would overflow shader arithmetic.
    InvalidGeometryTransform,
    /// The per-frame object buffer exceeds the active device limit.
    InstanceCapacityTooLarge,
    /// Camera projection aspect does not match the target logical viewport.
    CameraTargetAspectMismatch,
    /// Edge projection or style arithmetic is not finite for this target.
    InvalidEdgeProjection,
}

impl fmt::Display for Mesh3dRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererMismatch => {
                write!(formatter, "3D scene resource belongs to another renderer")
            }
            Self::InvalidGeometryTransform => {
                write!(formatter, "3D transform would produce invalid GPU geometry")
            }
            Self::InstanceCapacityTooLarge => write!(
                formatter,
                "3D object count exceeds the GPU instance-buffer limit"
            ),
            Self::CameraTargetAspectMismatch => write!(
                formatter,
                "3D camera aspect does not match the target logical viewport"
            ),
            Self::InvalidEdgeProjection => write!(
                formatter,
                "3D edge clipping or screen-space arithmetic overflowed"
            ),
        }
    }
}

impl Error for Mesh3dRenderError {}

/// CPU-side result of one depth-tested offscreen 3D draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mesh3dRenderReport {
    object_count: usize,
    triangle_count: usize,
    edge_count: usize,
    upload: Duration,
    encode_submit: Duration,
}

impl Mesh3dRenderReport {
    /// Returns independently transformed retained objects drawn.
    pub const fn object_count(self) -> usize {
        self.object_count
    }

    /// Returns total indexed triangles submitted across every object.
    pub const fn triangle_count(self) -> usize {
        self.triangle_count
    }

    /// Returns explicit mathematical edges submitted for depth classification.
    pub const fn edge_count(self) -> usize {
        self.edge_count
    }

    /// Returns CPU time spent validating and enqueuing camera/instance uploads.
    pub const fn upload(self) -> Duration {
        self.upload
    }

    /// Returns CPU time spent encoding and submitting the pass.
    pub const fn encode_submit(self) -> Duration {
        self.encode_submit
    }
}

/// Result of atomically migrating a retained 3D scene to the active renderer device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scene3dRestoreReport {
    object_count: usize,
    migrated_object_count: usize,
    restored_mesh_count: usize,
    restored_gpu_bytes: usize,
}

impl Scene3dRestoreReport {
    /// Returns all scene objects whose stable IDs and visual state were preserved.
    pub const fn object_count(self) -> usize {
        self.object_count
    }

    /// Returns objects whose stale mesh reference was replaced in this call.
    pub const fn migrated_object_count(self) -> usize {
        self.migrated_object_count
    }

    /// Returns distinct retained mesh resources recreated on the active device.
    pub const fn restored_mesh_count(self) -> usize {
        self.restored_mesh_count
    }

    /// Returns total GPU bytes represented by the recreated distinct meshes.
    pub const fn restored_gpu_bytes(self) -> usize {
        self.restored_gpu_bytes
    }
}

impl WgpuRenderer {
    /// Uploads validated immutable topology into retained GPU buffers.
    ///
    /// Counts, byte arithmetic, draw-count representation, and active-device
    /// buffer limits are checked before conversion staging allocation. Host
    /// staging reservation failure is returned explicitly; GPU allocation
    /// itself follows wgpu's device-error model.
    pub fn create_mesh3d(&self, source: Mesh3d) -> Result<RetainedMesh3d, Mesh3dResourceError> {
        create_retained_mesh(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            source,
        )
    }

    /// Restores retained topology onto this renderer after device replacement.
    pub fn restore_mesh3d(
        &self,
        source: &RetainedMesh3d,
    ) -> Result<RetainedMesh3d, Mesh3dResourceError> {
        self.create_mesh3d(source.source.clone())
    }

    /// Atomically restores every stale retained mesh referenced by a 3D scene.
    ///
    /// Distinct shared mesh resources are uploaded once. Object IDs, insertion
    /// order, transforms, styles, visibility, scene provenance, and the next ID
    /// remain unchanged. If any capacity or host-staging allocation fails, the
    /// original scene is not modified. Targets remain separate resources and
    /// must be restored with [`WgpuRenderer::restore_render_target3d`].
    pub fn restore_scene3d(
        &self,
        scene: &mut Scene3d,
    ) -> Result<Scene3dRestoreReport, Mesh3dResourceError> {
        restore_scene3d_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            scene,
        )
    }

    /// Creates color/depth attachments for an explicit logical viewport.
    ///
    /// `width` and `height` are physical target texels. Their aspect must match
    /// `logical_viewport`; the resulting texels-per-logical-pixel ratio controls
    /// wireframe width independently of the window DPI scale.
    pub fn create_render_target3d(
        &self,
        width: u32,
        height: u32,
        logical_viewport: LogicalViewport,
    ) -> Result<RenderTarget3d, Mesh3dResourceError> {
        let pixels_per_logical = target_pixels_per_logical(width, height, logical_viewport)
            .ok_or(Mesh3dResourceError::InvalidViewportAspect)?;
        let color = self
            .create_render_target(width, height)
            .map_err(Mesh3dResourceError::Target)?;
        let depth_texture = create_depth_texture(&self.device, width, height);
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(RenderTarget3d {
            renderer_identity: Arc::clone(&self.renderer_identity),
            color,
            _depth_texture: depth_texture,
            depth_view,
            logical_viewport,
            pixels_per_logical,
        })
    }

    /// Restores empty color/depth attachments with the source dimensions.
    pub fn restore_render_target3d(
        &self,
        source: &RenderTarget3d,
    ) -> Result<RenderTarget3d, Mesh3dResourceError> {
        self.create_render_target3d(source.width(), source.height(), source.logical_viewport())
    }

    /// Draws opaque retained objects into a reusable color/depth target.
    ///
    /// Object insertion order does not determine visibility. Every surface
    /// writes and tests hardware depth. Model transforms are uploaded through a
    /// reusable instance buffer; retained topology is never retessellated.
    pub fn render_scene3d_to_target(
        &mut self,
        target: &RenderTarget3d,
        scene: &Scene3d,
        camera: Camera3d,
    ) -> Result<Mesh3dRenderReport, Mesh3dRenderError> {
        validate_target_identity(&self.renderer_identity, target)?;
        for instance in scene.instances() {
            validate_mesh_identity(&self.renderer_identity, &instance.mesh)?;
        }
        validate_camera_target_aspect(camera, target.logical_viewport())?;
        let camera_uniform = Camera3dUniform::new(
            camera,
            target.width(),
            target.height(),
            target.pixels_per_logical(),
        )?;
        self.mesh3d_renderer
            .ensure_instance_capacity(&self.device, scene.object_count())?;
        self.mesh3d_renderer
            .ensure_edge_object_capacity(&self.device, scene.object_count())?;
        self.mesh3d_renderer.instances.clear();
        self.mesh3d_renderer.edge_object_bytes.clear();
        self.mesh3d_renderer.edge_object_bytes.resize(
            scene
                .object_count()
                .saturating_mul(self.mesh3d_renderer.edge_object_stride),
            0,
        );
        for (object_index, instance) in scene.instances().iter().enumerate() {
            if !instance.visible {
                self.mesh3d_renderer.instances.push(MeshInstanceGpu {
                    model_row_0: [1.0, 0.0, 0.0, 0.0],
                    model_row_1: [0.0, 1.0, 0.0, 0.0],
                    model_row_2: [0.0, 0.0, 1.0, 0.0],
                    color: Color::BLACK.to_array(),
                });
                continue;
            }
            let model_rows = instance
                .transform
                .model_rows()
                .map_err(|_| Mesh3dRenderError::InvalidGeometryTransform)?;
            validate_shader_transform(instance.mesh.source(), model_rows, camera_uniform.rows())?;
            self.mesh3d_renderer.instances.push(MeshInstanceGpu {
                model_row_0: model_rows[0],
                model_row_1: model_rows[1],
                model_row_2: model_rows[2],
                color: instance
                    .style
                    .surface_style()
                    .map_or(Color::BLACK, |surface| surface.color())
                    .to_array(),
            });
            if let Some(style) = instance.wireframe() {
                validate_edge_projection(
                    instance.mesh.source(),
                    model_rows,
                    camera_uniform.rows(),
                    style,
                    camera_uniform.viewport,
                )?;
                let hidden_color = style.hidden_color().unwrap_or(style.visible_color());
                let hidden_width = style.hidden_width().map_or(0.0, LogicalPixels::get);
                let (dash_length, gap_length) = style
                    .hidden_pattern()
                    .map_or((1.0, 1.0), |(dash, gap)| (dash.get(), gap.get()));
                let edge_uniform = EdgeObjectUniform {
                    model_row_0: model_rows[0],
                    model_row_1: model_rows[1],
                    model_row_2: model_rows[2],
                    visible_color: style.visible_color().to_array(),
                    hidden_color: hidden_color.to_array(),
                    edge_style: [
                        style.visible_width().get(),
                        hidden_width,
                        dash_length,
                        gap_length,
                    ],
                };
                let start = object_index * self.mesh3d_renderer.edge_object_stride;
                let end = start + std::mem::size_of::<EdgeObjectUniform>();
                self.mesh3d_renderer.edge_object_bytes[start..end]
                    .copy_from_slice(bytemuck::bytes_of(&edge_uniform));
            }
        }

        let upload_started_at = Instant::now();
        self.queue.write_buffer(
            &self.mesh3d_renderer.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        if !self.mesh3d_renderer.instances.is_empty() {
            self.queue.write_buffer(
                &self.mesh3d_renderer.instance_buffer,
                0,
                bytemuck::cast_slice(&self.mesh3d_renderer.instances),
            );
        }
        if !self.mesh3d_renderer.edge_object_bytes.is_empty() {
            self.queue.write_buffer(
                &self.mesh3d_renderer.edge_object_buffer,
                0,
                &self.mesh3d_renderer.edge_object_bytes,
            );
        }
        let upload = upload_started_at.elapsed();
        let encode_started_at = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine retained 3D scene encoder"),
            });
        encode_scene_pass(
            &mut encoder,
            &self.mesh3d_renderer,
            &target.color.view,
            &target.depth_view,
            scene.background(),
            scene.instances(),
        );
        self.queue.submit([encoder.finish()]);
        let triangle_count = scene.instances().iter().fold(0usize, |count, instance| {
            if instance.visible && instance.style.surface_style().is_some() {
                count.saturating_add(instance.mesh.triangle_count())
            } else {
                count
            }
        });
        let edge_count = scene.instances().iter().fold(0usize, |count, instance| {
            if instance.visible && instance.wireframe().is_some() {
                count.saturating_add(instance.mesh.source().display_edges().len())
            } else {
                count
            }
        });
        Ok(Mesh3dRenderReport {
            object_count: scene.visible_object_count(),
            triangle_count,
            edge_count,
            upload,
            encode_submit: encode_started_at.elapsed(),
        })
    }
}

fn create_retained_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    source: Mesh3d,
) -> Result<RetainedMesh3d, Mesh3dResourceError> {
    let layout = preflight_mesh3d_upload(
        source.vertices().len(),
        source.triangle_indices().len(),
        source.display_edges().len(),
        device.limits().max_buffer_size,
    )?;
    let mut vertices = Vec::new();
    vertices
        .try_reserve_exact(source.vertices().len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: layout.vertex_bytes,
        })?;
    vertices.extend(source.vertices().iter().map(|vertex| MeshVertexGpu {
        position: [vertex.x(), vertex.y(), vertex.z()],
    }));
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(source.display_edges().len())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: layout.edge_bytes,
        })?;
    edges.extend(source.display_edges().iter().map(|edge| {
        let start = source.vertices()[edge.start() as usize];
        let end = source.vertices()[edge.end() as usize];
        MeshEdgeGpu {
            start: [start.x(), start.y(), start.z()],
            end: [end.x(), end.y(), end.z()],
        }
    }));
    let vertex_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine retained 3D vertex buffer"),
        size: layout.vertex_bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let index_buffer = if layout.index_bytes == 0 {
        None
    } else {
        Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine retained 3D index buffer"),
            size: layout.index_bytes,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })))
    };
    queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
    if let Some(index_buffer) = index_buffer.as_ref() {
        queue.write_buffer(
            index_buffer,
            0,
            bytemuck::cast_slice(source.triangle_indices()),
        );
    }
    let edge_buffer = if layout.edge_bytes == 0 {
        None
    } else {
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine retained 3D edge buffer"),
            size: layout.edge_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&edges));
        Some(buffer)
    };
    Ok(RetainedMesh3d {
        renderer_identity,
        vertex_buffer,
        index_buffer,
        edge_buffer,
        source,
        index_count: layout.index_count,
        edge_count: layout.edge_count,
        gpu_allocation_bytes: usize::try_from(layout.total_bytes).unwrap_or(usize::MAX),
    })
}

fn restore_scene3d_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    scene: &mut Scene3d,
) -> Result<Scene3dRestoreReport, Mesh3dResourceError> {
    let mut restored = Vec::<(Arc<wgpu::Buffer>, RetainedMesh3d)>::new();
    for instance in &scene.instances {
        if Arc::ptr_eq(&renderer_identity, &instance.mesh.renderer_identity)
            || restored.iter().any(|(old_vertex_buffer, _)| {
                Arc::ptr_eq(old_vertex_buffer, &instance.mesh.vertex_buffer)
            })
        {
            continue;
        }
        let old_vertex_buffer = Arc::clone(&instance.mesh.vertex_buffer);
        let replacement = create_retained_mesh(
            device,
            queue,
            Arc::clone(&renderer_identity),
            instance.mesh.source.clone(),
        )?;
        restored.push((old_vertex_buffer, replacement));
    }

    let mut replacement_instances = scene.instances.clone();
    let mut migrated_object_count = 0;
    for instance in &mut replacement_instances {
        let Some((_, replacement)) = restored.iter().find(|(old_vertex_buffer, _)| {
            Arc::ptr_eq(old_vertex_buffer, &instance.mesh.vertex_buffer)
        }) else {
            continue;
        };
        instance.mesh = replacement.clone();
        migrated_object_count += 1;
    }
    let restored_gpu_bytes = restored.iter().fold(0_usize, |total, (_, mesh)| {
        total.saturating_add(mesh.gpu_allocation_bytes())
    });
    scene.instances = replacement_instances;
    Ok(Scene3dRestoreReport {
        object_count: scene.object_count(),
        migrated_object_count,
        restored_mesh_count: restored.len(),
        restored_gpu_bytes,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mesh3dUploadLayout {
    vertex_bytes: u64,
    index_bytes: u64,
    edge_bytes: u64,
    total_bytes: u64,
    index_count: u32,
    edge_count: u32,
}

fn preflight_mesh3d_upload(
    vertex_count: usize,
    index_count: usize,
    edge_count: usize,
    max_buffer_size: u64,
) -> Result<Mesh3dUploadLayout, Mesh3dResourceError> {
    let draw_index_count =
        u32::try_from(index_count).map_err(|_| Mesh3dResourceError::CapacityTooLarge)?;
    let draw_edge_count =
        u32::try_from(edge_count).map_err(|_| Mesh3dResourceError::CapacityTooLarge)?;
    let checked_buffer_bytes = |element_size: usize, element_count: usize| {
        u64::try_from(element_size)
            .ok()
            .and_then(|size| u64::try_from(element_count).ok()?.checked_mul(size))
            .filter(|bytes| *bytes <= max_buffer_size)
            .ok_or(Mesh3dResourceError::CapacityTooLarge)
    };
    let vertex_bytes = checked_buffer_bytes(std::mem::size_of::<MeshVertexGpu>(), vertex_count)?;
    let index_bytes = checked_buffer_bytes(std::mem::size_of::<u32>(), index_count)?;
    let edge_bytes = checked_buffer_bytes(std::mem::size_of::<MeshEdgeGpu>(), edge_count)?;
    let total_bytes = vertex_bytes
        .checked_add(index_bytes)
        .and_then(|bytes| bytes.checked_add(edge_bytes))
        .ok_or(Mesh3dResourceError::CapacityTooLarge)?;
    Ok(Mesh3dUploadLayout {
        vertex_bytes,
        index_bytes,
        edge_bytes,
        total_bytes,
        index_count: draw_index_count,
        edge_count: draw_edge_count,
    })
}

fn create_instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D instance buffer"),
        size: (capacity.max(1) * std::mem::size_of::<MeshInstanceGpu>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn create_edge_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    depth_compare: wgpu::CompareFunction,
    fragment_entry: &'static str,
    label: &'static str,
) -> wgpu::RenderPipeline {
    let (vertex_entry, depth_bias) = if depth_compare == wgpu::CompareFunction::Greater {
        (
            "mesh3d_hidden_edge_vs_main",
            wgpu::DepthBiasState {
                constant: HIDDEN_EDGE_DEPTH_BIAS,
                slope_scale: 0.0,
                clamp: 0.0,
            },
        )
    } else {
        (
            "mesh3d_visible_edge_vs_main",
            wgpu::DepthBiasState {
                constant: VISIBLE_EDGE_DEPTH_BIAS,
                slope_scale: 0.0,
                clamp: 0.0,
            },
        )
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(MeshEdgeGpu::LAYOUT)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: depth_bias,
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn align_to(value: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment).saturating_mul(alignment)
    }
}

fn create_edge_object_buffer(
    device: &wgpu::Device,
    stride: usize,
    capacity: usize,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D edge object uniform buffer"),
        size: capacity.max(1).saturating_mul(stride) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_edge_object_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sim-engine 3D edge object bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<EdgeObjectUniform>() as u64),
            }),
        }],
    })
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine 3D depth target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
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
        || !aspect_matches(horizontal as f32, vertical as f32)
        || horizontal > f32::MAX as f64
    {
        return None;
    }
    PhysicalPerLogical::new(horizontal as f32).ok()
}

fn aspect_matches(left: f32, right: f32) -> bool {
    if !left.is_finite() || !right.is_finite() || left <= 0.0 || right <= 0.0 {
        return false;
    }
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= scale * 1.0e-5
}

fn validate_camera_target_aspect(
    camera: Camera3d,
    viewport: LogicalViewport,
) -> Result<(), Mesh3dRenderError> {
    aspect_matches(
        camera.projection().aspect_ratio(),
        viewport.width() / viewport.height(),
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
    Arc::ptr_eq(renderer_identity, &mesh.renderer_identity)
        .then_some(())
        .ok_or(Mesh3dRenderError::RendererMismatch)
}

fn validate_shader_transform(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
) -> Result<(), Mesh3dRenderError> {
    // Eight AABB corners are the constant-cost common path. If that
    // conservative envelope fails, inspect only coordinates consumed by the
    // GPU: synthetic corners lose axis correlation and may reject a sparse
    // mesh even though every actual vertex is safe.
    if bounds_corners(mesh.bounds_min(), mesh.bounds_max())
        .into_iter()
        .all(|corner| validate_shader_point(corner, model_rows, camera_rows).is_ok())
    {
        return Ok(());
    }
    for vertex in mesh.vertices() {
        validate_shader_point(*vertex, model_rows, camera_rows)?;
    }
    Ok(())
}

fn validate_shader_point(
    point: Vec3,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
) -> Result<(), Mesh3dRenderError> {
    let model_point = [point.x(), point.y(), point.z(), 1.0];
    let world = [
        shader_dot(model_rows[0], model_point)?,
        shader_dot(model_rows[1], model_point)?,
        shader_dot(model_rows[2], model_point)?,
        1.0,
    ];
    for row in camera_rows {
        let _ = shader_dot(row, world)?;
    }
    Ok(())
}

fn validate_edge_projection(
    mesh: &Mesh3d,
    model_rows: [[f32; 4]; 3],
    camera_rows: [[f32; 4]; 4],
    style: WireframeStyle3d,
    viewport: [f32; 4],
) -> Result<(), Mesh3dRenderError> {
    let pixels_per_logical = PhysicalPerLogical::new(viewport[2])
        .map_err(|_| Mesh3dRenderError::InvalidEdgeProjection)?;
    let visible_raster = edge_raster_envelope(style.visible_width(), pixels_per_logical)
        .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?;
    let hidden_raster = match style.hidden_width() {
        Some(width) => Some(
            edge_raster_envelope(width, pixels_per_logical)
                .ok_or(Mesh3dRenderError::InvalidEdgeProjection)?,
        ),
        None => None,
    };
    let hidden_period = style
        .hidden_pattern()
        .map(|(dash, gap)| dash.get() + gap.get());
    if hidden_period.is_some_and(|period| !period.is_finite()) {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }

    for edge in mesh.display_edges() {
        let mut clip = [[0.0_f32; 4]; 2];
        for (endpoint_index, vertex_index) in [edge.start(), edge.end()].into_iter().enumerate() {
            let vertex = mesh.vertices()[vertex_index as usize];
            let model = [vertex.x(), vertex.y(), vertex.z(), 1.0];
            let world = [
                shader_dot(model_rows[0], model)?,
                shader_dot(model_rows[1], model)?,
                shader_dot(model_rows[2], model)?,
                1.0,
            ];
            clip[endpoint_index] = [
                shader_dot(camera_rows[0], world)?,
                shader_dot(camera_rows[1], world)?,
                shader_dot(camera_rows[2], world)?,
                shader_dot(camera_rows[3], world)?,
            ];
        }
        let Some(clipped) = clip_edge_to_frustum(clip[0], clip[1])? else {
            continue;
        };
        let mut screen = [[0.0_f32; 2]; 2];
        for (endpoint_index, clip) in clipped.into_iter().enumerate() {
            let ndc_x = clip[0] / clip[3];
            let ndc_y = clip[1] / clip[3];
            screen[endpoint_index] = [
                (ndc_x * 0.5 + 0.5) * viewport[0],
                (0.5 - ndc_y * 0.5) * viewport[1],
            ];
            if !screen[endpoint_index][0].is_finite() || !screen[endpoint_index][1].is_finite() {
                return Err(Mesh3dRenderError::InvalidEdgeProjection);
            }
        }
        let delta_x = screen[1][0] - screen[0][0];
        let delta_y = screen[1][1] - screen[0][1];
        if !delta_x.is_finite()
            || !delta_y.is_finite()
            || !(delta_x * delta_x + delta_y * delta_y).is_finite()
        {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
        let screen_length = (delta_x * delta_x + delta_y * delta_y).sqrt();
        let screen_normal = if screen_length > 0.0001 {
            [-delta_y / screen_length, delta_x / screen_length]
        } else {
            [0.0, 0.0]
        };
        if !screen_normal.into_iter().all(f32::is_finite) {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
        validate_edge_shader_arithmetic(
            clipped,
            screen_length,
            screen_normal,
            viewport,
            visible_raster,
            None,
        )?;
        if let Some(hidden_raster) = hidden_raster {
            validate_edge_shader_arithmetic(
                clipped,
                screen_length,
                screen_normal,
                viewport,
                hidden_raster,
                hidden_period,
            )?;
        }
    }
    Ok(())
}

fn clip_edge_to_frustum(
    start: [f32; 4],
    end: [f32; 4],
) -> Result<Option<[[f32; 4]; 2]>, Mesh3dRenderError> {
    Ok(clip_edge_to_frustum_details(start, end)?.map(|clipped| clipped.clip))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ClippedEdgeCpu {
    clip: [[f32; 4]; 2],
    enter: f32,
    exit: f32,
}

fn clip_edge_to_frustum_details(
    start: [f32; 4],
    end: [f32; 4],
) -> Result<Option<ClippedEdgeCpu>, Mesh3dRenderError> {
    let pair_max = start
        .into_iter()
        .chain(end)
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    if !pair_max.is_finite() {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    let homogeneous_scale = (1.0 / pair_max.max(1.0)).max(f32::MIN_POSITIVE);
    let start = start.map(|component| component * homogeneous_scale);
    let end = end.map(|component| component * homogeneous_scale);
    let start_distances = clip_plane_distances(start)?;
    let end_distances = clip_plane_distances(end)?;
    let mut enter = 0.0_f32;
    let mut exit = 1.0_f32;
    for (start_distance, end_distance) in start_distances.into_iter().zip(end_distances) {
        if start_distance < 0.0 && end_distance < 0.0 {
            return Ok(None);
        }
        if start_distance < 0.0 || end_distance < 0.0 {
            let denominator = start_distance - end_distance;
            let amount = start_distance / denominator;
            if !denominator.is_finite() || !amount.is_finite() {
                return Err(Mesh3dRenderError::InvalidEdgeProjection);
            }
            if start_distance < 0.0 {
                enter = enter.max(amount);
            } else {
                exit = exit.min(amount);
            }
        }
    }
    if enter > exit {
        return Ok(None);
    }
    let clipped_start = shader_lerp_clip(start, end, enter)?;
    let clipped_end = shader_lerp_clip(start, end, exit)?;
    if clipped_start[3] <= 0.0 || clipped_end[3] <= 0.0 {
        return Ok(None);
    }
    Ok(Some(ClippedEdgeCpu {
        clip: [clipped_start, clipped_end],
        enter,
        exit,
    }))
}

fn clip_plane_distances(clip: [f32; 4]) -> Result<[f32; 6], Mesh3dRenderError> {
    let add = |left: f32, right: f32| {
        let result = left + right;
        result
            .is_finite()
            .then_some(result)
            .ok_or(Mesh3dRenderError::InvalidEdgeProjection)
    };
    let subtract = |left: f32, right: f32| {
        let result = left - right;
        result
            .is_finite()
            .then_some(result)
            .ok_or(Mesh3dRenderError::InvalidEdgeProjection)
    };
    Ok([
        add(clip[3], clip[0])?,
        subtract(clip[3], clip[0])?,
        add(clip[3], clip[1])?,
        subtract(clip[3], clip[1])?,
        clip[2],
        subtract(clip[3], clip[2])?,
    ])
}

fn shader_lerp_clip(
    start: [f32; 4],
    end: [f32; 4],
    amount: f32,
) -> Result<[f32; 4], Mesh3dRenderError> {
    let mut output = [0.0_f32; 4];
    for axis in 0..4 {
        let delta = end[axis] - start[axis];
        let value = start[axis] + delta * amount;
        if !delta.is_finite() || !value.is_finite() {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
        output[axis] = value;
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy)]
struct EdgeRasterEnvelope {
    physical_half_width: f32,
    raster_half_width: f32,
}

fn edge_raster_envelope(
    logical_width: LogicalPixels,
    pixels_per_logical: PhysicalPerLogical,
) -> Option<EdgeRasterEnvelope> {
    let physical_half_width = logical_width.get() * pixels_per_logical.get() * 0.5;
    let raster_half_width = (physical_half_width + 0.5).max(1.0);
    let doubled_raster_width = raster_half_width * 2.0;
    (physical_half_width.is_finite()
        && physical_half_width > 0.0
        && raster_half_width.is_finite()
        && doubled_raster_width.is_finite())
    .then_some(EdgeRasterEnvelope {
        physical_half_width,
        raster_half_width,
    })
}

fn validate_edge_shader_arithmetic(
    clipped: [[f32; 4]; 2],
    screen_length: f32,
    screen_normal: [f32; 2],
    viewport: [f32; 4],
    raster: EdgeRasterEnvelope,
    dash_period: Option<f32>,
) -> Result<(), Mesh3dRenderError> {
    let logical_distance = screen_length / viewport[2];
    if !logical_distance.is_finite()
        || !(raster.physical_half_width + 0.5).is_finite()
        || !(raster.raster_half_width * 2.0).is_finite()
    {
        return Err(Mesh3dRenderError::InvalidEdgeProjection);
    }
    if let Some(period) = dash_period {
        let repetition = logical_distance / period;
        let repeated_distance = repetition.floor() * period;
        let within_period = logical_distance - repeated_distance;
        if !period.is_finite()
            || period <= 0.0
            || !repetition.is_finite()
            || !repeated_distance.is_finite()
            || !within_period.is_finite()
        {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
    }

    let dimensions = [viewport[0].max(1.0), viewport[1].max(1.0)];
    for axis in 0..2 {
        // Match `screen_normal * side_pattern * raster_half_width` before the
        // shader's multiply-by-two. The two signs are checked at clip addition.
        let screen_offset_max_abs = screen_normal[axis].abs() * raster.raster_half_width;
        let doubled_screen_offset = screen_offset_max_abs * 2.0;
        let ndc_offset_max_abs = doubled_screen_offset / dimensions[axis];
        if !ndc_offset_max_abs.is_finite() {
            return Err(Mesh3dRenderError::InvalidEdgeProjection);
        }
        for endpoint in clipped {
            let clip_offset_max_abs = ndc_offset_max_abs * endpoint[3].abs();
            if !clip_offset_max_abs.is_finite()
                || !(endpoint[axis] + clip_offset_max_abs).is_finite()
                || !(endpoint[axis] - clip_offset_max_abs).is_finite()
            {
                return Err(Mesh3dRenderError::InvalidEdgeProjection);
            }
        }
    }
    Ok(())
}

fn shader_dot(row: [f32; 4], point: [f32; 4]) -> Result<f32, Mesh3dRenderError> {
    let terms: [(f64, f64); 4] = std::array::from_fn(|axis| {
        let product = f64::from(row[axis]) * f64::from(point[axis]);
        (product, product)
    });
    if !shader_interval_sum_is_safe(terms) {
        return Err(Mesh3dRenderError::InvalidGeometryTransform);
    }
    let mut result = 0.0_f32;
    for axis in 0..4 {
        let product = row[axis] * point[axis];
        result += product;
        if !product.is_finite() || !result.is_finite() {
            return Err(Mesh3dRenderError::InvalidGeometryTransform);
        }
    }
    Ok(result)
}

fn bounds_corners(minimum: Vec3, maximum: Vec3) -> [Vec3; 8] {
    [
        Vec3::new_unchecked(minimum.x(), minimum.y(), minimum.z()),
        Vec3::new_unchecked(maximum.x(), minimum.y(), minimum.z()),
        Vec3::new_unchecked(minimum.x(), maximum.y(), minimum.z()),
        Vec3::new_unchecked(maximum.x(), maximum.y(), minimum.z()),
        Vec3::new_unchecked(minimum.x(), minimum.y(), maximum.z()),
        Vec3::new_unchecked(maximum.x(), minimum.y(), maximum.z()),
        Vec3::new_unchecked(minimum.x(), maximum.y(), maximum.z()),
        Vec3::new_unchecked(maximum.x(), maximum.y(), maximum.z()),
    ]
}

fn encode_scene_pass(
    encoder: &mut wgpu::CommandEncoder,
    renderer: &Mesh3dRenderer,
    color_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    background: Color,
    instances: &[Mesh3dInstance],
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("sim-engine retained 3D mesh pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(background.to_wgpu()),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&renderer.pipeline);
    pass.set_bind_group(0, &renderer.camera_bind_group, &[]);
    for (instance_index, instance) in instances.iter().enumerate() {
        if !instance.visible {
            continue;
        }
        if instance.style.surface_style().is_none() {
            continue;
        }
        let Some(index_buffer) = instance.mesh.index_buffer.as_ref() else {
            continue;
        };
        let instance_start =
            (instance_index * std::mem::size_of::<MeshInstanceGpu>()) as wgpu::BufferAddress;
        let instance_end =
            instance_start + std::mem::size_of::<MeshInstanceGpu>() as wgpu::BufferAddress;
        pass.set_vertex_buffer(0, instance.mesh.vertex_buffer.slice(..));
        pass.set_vertex_buffer(
            1,
            renderer.instance_buffer.slice(instance_start..instance_end),
        );
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..instance.mesh.index_count, 0, 0..1);
    }
    pass.set_pipeline(&renderer.hidden_edge_pipeline);
    for (object_index, instance) in instances.iter().enumerate() {
        if !instance.visible {
            continue;
        }
        let Some(style) = instance.wireframe() else {
            continue;
        };
        let Some(edge_buffer) = instance.mesh.edge_buffer.as_ref() else {
            continue;
        };
        if !style.hidden_enabled() {
            continue;
        }
        let dynamic_offset = (object_index * renderer.edge_object_stride) as u32;
        pass.set_bind_group(1, &renderer.edge_object_bind_group, &[dynamic_offset]);
        pass.set_vertex_buffer(0, edge_buffer.slice(..));
        pass.draw(0..6, 0..instance.mesh.edge_count);
    }
    pass.set_pipeline(&renderer.visible_edge_pipeline);
    for (object_index, instance) in instances.iter().enumerate() {
        if !instance.visible {
            continue;
        }
        if instance.wireframe().is_none() {
            continue;
        }
        let Some(edge_buffer) = instance.mesh.edge_buffer.as_ref() else {
            continue;
        };
        let dynamic_offset = (object_index * renderer.edge_object_stride) as u32;
        pass.set_bind_group(1, &renderer.edge_object_bind_group, &[dynamic_offset]);
        pass.set_vertex_buffer(0, edge_buffer.slice(..));
        pass.draw(0..6, 0..instance.mesh.edge_count);
    }
}

#[cfg(test)]
pub(super) fn assert_gpu_depth_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) {
    assert_gpu_clip_equivalence(device, queue);
    let identity = Arc::new(());
    let mut renderer = Mesh3dRenderer::new(device, format);
    let vertices = vec![
        Vec3::new(-1.0, -1.0, 0.0).unwrap(),
        Vec3::new(1.0, -1.0, 0.0).unwrap(),
        Vec3::new(0.0, 1.0, 0.0).unwrap(),
    ];
    let near_triangle = Mesh3d::with_display_edges(
        vertices.clone(),
        vec![0, 1, 2],
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let far_triangle = Mesh3d::with_display_edges(
        vertices,
        vec![0, 1, 2],
        vec![MeshEdge3d::new(0, 2).unwrap()],
    )
    .unwrap();
    let near_mesh = create_retained_mesh(device, queue, Arc::clone(&identity), near_triangle)
        .expect("3D test mesh should upload");
    let far_mesh = create_retained_mesh(device, queue, Arc::clone(&identity), far_triangle)
        .expect("3D test mesh should upload");
    let near_crossing_edge = Mesh3d::with_display_edges(
        vec![
            Vec3::new(0.0, 0.0, 4.95).unwrap(),
            Vec3::new(0.5, 0.5, 4.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let near_crossing_mesh =
        create_retained_mesh(device, queue, Arc::clone(&identity), near_crossing_edge)
            .expect("near-crossing 3D test edge should upload");
    let classification_edge = Mesh3d::with_display_edges(
        vec![
            Vec3::new(-0.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.5, -0.5, 0.0).unwrap(),
        ],
        Vec::new(),
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let classification_mesh =
        create_retained_mesh(device, queue, Arc::clone(&identity), classification_edge)
            .expect("classification test edge should upload");
    let projection = Projection3d::perspective(0.927_295_2, 1.0, world(0.1), world(20.0)).unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 5.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        projection,
    )
    .unwrap();
    let camera_uniform = Camera3dUniform::new(camera, 64, 64, physical_per_logical(1.0)).unwrap();
    let near_transform = Transform3d::new(
        Vec3::new(0.0, 0.0, 1.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let far_transform = Transform3d::new(
        Vec3::new(0.0, 0.0, -1.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let near_id = scene
        .try_push(
            &near_mesh,
            near_transform,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::rgb(0.0, 1.0, 0.0)).unwrap())
                .with_wireframe(
                    WireframeStyle3d::visible(Color::rgb(0.0, 0.0, 1.0), logical(2.0))
                        .unwrap()
                        .with_hidden(
                            Color::rgb(1.0, 0.0, 1.0),
                            logical(2.0),
                            logical(4.0),
                            logical(4.0),
                        )
                        .unwrap(),
                ),
        )
        .unwrap();
    let mut other_scene = Scene3d::new(Color::BLACK).unwrap();
    let other_scene_id = other_scene
        .try_push(
            &near_mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap()),
        )
        .unwrap();
    assert_eq!(near_id.get(), other_scene_id.get());
    assert_ne!(near_id, other_scene_id);
    assert_eq!(
        other_scene.set_visible(near_id, false),
        Err(Scene3dError::ObjectNotFound { object_id: near_id })
    );
    assert_eq!(other_scene.visible_object_count(), 1);
    assert_eq!(
        scene.set_visible(other_scene_id, false),
        Err(Scene3dError::ObjectNotFound {
            object_id: other_scene_id,
        })
    );
    assert_eq!(scene.visible_object_count(), 1);
    let far_id = scene
        .try_push(
            &far_mesh,
            far_transform,
            MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::rgb(1.0, 0.0, 0.0)).unwrap())
                .with_wireframe(
                    WireframeStyle3d::visible(Color::rgb(1.0, 0.0, 0.0), logical(2.0))
                        .unwrap()
                        .with_hidden(Color::WHITE, logical(2.0), logical(4.0), logical(4.0))
                        .unwrap(),
                ),
        )
        .unwrap();
    scene
        .try_push(
            &near_crossing_mesh,
            Transform3d::IDENTITY,
            MeshStyle3d::wireframe(
                WireframeStyle3d::visible(Color::rgb(1.0, 1.0, 0.0), logical(2.0)).unwrap(),
            ),
        )
        .unwrap();
    let coplanar_transform = Transform3d::new(
        Vec3::new(0.0, 0.0, 1.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    scene
        .try_push(
            &classification_mesh,
            coplanar_transform,
            MeshStyle3d::wireframe(
                WireframeStyle3d::visible(Color::rgb(0.0, 1.0, 1.0), logical(2.0))
                    .unwrap()
                    .with_hidden(
                        Color::rgb(1.0, 0.0, 1.0),
                        logical(2.0),
                        logical(4.0),
                        logical(4.0),
                    )
                    .unwrap(),
            ),
        )
        .unwrap();
    let sub_depth_resolution_transform = Transform3d::new(
        Vec3::new(0.0, 0.2, 0.999_999).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    scene
        .try_push(
            &classification_mesh,
            sub_depth_resolution_transform,
            MeshStyle3d::wireframe(
                WireframeStyle3d::visible(Color::rgb(1.0, 0.5, 0.0), logical(2.0))
                    .unwrap()
                    .with_hidden(
                        Color::rgb(1.0, 0.0, 1.0),
                        logical(2.0),
                        logical(4.0),
                        logical(4.0),
                    )
                    .unwrap(),
            ),
        )
        .unwrap();
    let moved_near_transform = Transform3d::new(
        Vec3::new(0.25, 0.0, 1.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    scene.set_transform(near_id, moved_near_transform).unwrap();
    assert_eq!(scene.instances()[0].transform(), moved_near_transform);
    assert_eq!(scene.instances()[1].transform(), far_transform);
    assert_eq!(scene.instances()[1].id(), far_id);
    scene.set_visible(far_id, false).unwrap();
    assert_eq!(scene.visible_object_count(), 4);
    scene.set_visible(far_id, true).unwrap();
    assert_eq!(scene.visible_object_count(), 5);
    scene.set_transform(near_id, near_transform).unwrap();
    renderer
        .ensure_instance_capacity(device, scene.object_count())
        .unwrap();
    renderer.instances = scene
        .instances()
        .iter()
        .map(|instance| {
            let rows = instance.transform.model_rows().unwrap();
            MeshInstanceGpu {
                model_row_0: rows[0],
                model_row_1: rows[1],
                model_row_2: rows[2],
                color: instance
                    .style
                    .surface_style()
                    .map_or(Color::BLACK, SurfaceStyle3d::color)
                    .to_array(),
            }
        })
        .collect();
    renderer
        .edge_object_bytes
        .resize(scene.object_count() * renderer.edge_object_stride, 0);
    let near_rows = near_transform.model_rows().unwrap();
    let near_edge_uniform = EdgeObjectUniform {
        model_row_0: near_rows[0],
        model_row_1: near_rows[1],
        model_row_2: near_rows[2],
        visible_color: Color::rgb(0.0, 0.0, 1.0).to_array(),
        hidden_color: Color::rgb(1.0, 0.0, 1.0).to_array(),
        edge_style: [2.0, 2.0, 4.0, 4.0],
    };
    renderer.edge_object_bytes[..std::mem::size_of::<EdgeObjectUniform>()]
        .copy_from_slice(bytemuck::bytes_of(&near_edge_uniform));
    let far_rows = far_transform.model_rows().unwrap();
    let far_edge_uniform = EdgeObjectUniform {
        model_row_0: far_rows[0],
        model_row_1: far_rows[1],
        model_row_2: far_rows[2],
        visible_color: Color::rgb(1.0, 0.0, 0.0).to_array(),
        hidden_color: Color::WHITE.to_array(),
        edge_style: [2.0, 2.0, 4.0, 4.0],
    };
    let edge_start = renderer.edge_object_stride;
    let edge_end = edge_start + std::mem::size_of::<EdgeObjectUniform>();
    renderer.edge_object_bytes[edge_start..edge_end]
        .copy_from_slice(bytemuck::bytes_of(&far_edge_uniform));
    let crossing_rows = Transform3d::IDENTITY.model_rows().unwrap();
    let crossing_edge_uniform = EdgeObjectUniform {
        model_row_0: crossing_rows[0],
        model_row_1: crossing_rows[1],
        model_row_2: crossing_rows[2],
        visible_color: Color::rgb(1.0, 1.0, 0.0).to_array(),
        hidden_color: Color::rgb(1.0, 1.0, 0.0).to_array(),
        edge_style: [2.0, 0.0, 1.0, 1.0],
    };
    let crossing_edge_start = 2 * renderer.edge_object_stride;
    let crossing_edge_end = crossing_edge_start + std::mem::size_of::<EdgeObjectUniform>();
    renderer.edge_object_bytes[crossing_edge_start..crossing_edge_end]
        .copy_from_slice(bytemuck::bytes_of(&crossing_edge_uniform));
    let coplanar_rows = coplanar_transform.model_rows().unwrap();
    let coplanar_edge_uniform = EdgeObjectUniform {
        model_row_0: coplanar_rows[0],
        model_row_1: coplanar_rows[1],
        model_row_2: coplanar_rows[2],
        visible_color: Color::rgb(0.0, 1.0, 1.0).to_array(),
        hidden_color: Color::rgb(1.0, 0.0, 1.0).to_array(),
        edge_style: [2.0, 2.0, 4.0, 4.0],
    };
    let coplanar_edge_start = 3 * renderer.edge_object_stride;
    let coplanar_edge_end = coplanar_edge_start + std::mem::size_of::<EdgeObjectUniform>();
    renderer.edge_object_bytes[coplanar_edge_start..coplanar_edge_end]
        .copy_from_slice(bytemuck::bytes_of(&coplanar_edge_uniform));
    let sub_depth_rows = sub_depth_resolution_transform.model_rows().unwrap();
    let sub_depth_edge_uniform = EdgeObjectUniform {
        model_row_0: sub_depth_rows[0],
        model_row_1: sub_depth_rows[1],
        model_row_2: sub_depth_rows[2],
        visible_color: Color::rgb(1.0, 0.5, 0.0).to_array(),
        hidden_color: Color::rgb(1.0, 0.0, 1.0).to_array(),
        edge_style: [2.0, 2.0, 4.0, 4.0],
    };
    let sub_depth_edge_start = 4 * renderer.edge_object_stride;
    let sub_depth_edge_end = sub_depth_edge_start + std::mem::size_of::<EdgeObjectUniform>();
    renderer.edge_object_bytes[sub_depth_edge_start..sub_depth_edge_end]
        .copy_from_slice(bytemuck::bytes_of(&sub_depth_edge_uniform));
    queue.write_buffer(
        &renderer.camera_uniform_buffer,
        0,
        bytemuck::bytes_of(&camera_uniform),
    );
    queue.write_buffer(
        &renderer.instance_buffer,
        0,
        bytemuck::cast_slice(&renderer.instances),
    );
    queue.write_buffer(&renderer.edge_object_buffer, 0, &renderer.edge_object_bytes);
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine 3D depth test color"),
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
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_texture = create_depth_texture(device, 64, 64);
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D depth test readback"),
        size: 256 * 64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sim-engine 3D depth test encoder"),
    });
    encode_scene_pass(
        &mut encoder,
        &renderer,
        &color_view,
        &depth_view,
        Color::BLACK,
        scene.instances(),
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &color_texture,
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
    queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("3D depth test submission should complete");
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("3D depth test mapping should complete");
    receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("3D depth test callback should run")
        .expect("3D depth readback should map");
    let bytes = slice.get_mapped_range().expect("3D depth bytes");
    let [red, green, blue] = match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => [0, 1, 2],
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [2, 1, 0],
        _ => panic!("unsupported 3D byte-channel oracle format: {format:?}"),
    };
    let center_offset = 32 * 256 + 32 * 4;
    assert!(
        bytes[center_offset + green] > 200,
        "near green surface must win depth testing"
    );
    assert!(
        bytes[center_offset + red] < 20,
        "far red surface must remain occluded"
    );
    for x in 20..44 {
        let offset = 48 * 256 + x * 4;
        assert!(
            bytes[offset + blue] > 180 && bytes[offset + red] < 80,
            "coplanar near edge must remain solid visible blue at x={x}"
        );
    }
    let mut hidden_pattern = Vec::new();
    for step in 1..20 {
        let amount = step as f32 / 20.0;
        let x = (21.33 + (32.0 - 21.33) * amount).round() as usize;
        let y = (42.67 + (21.33 - 42.67) * amount).round() as usize;
        let offset = y * 256 + x * 4;
        hidden_pattern
            .push(bytes[offset] > 220 && bytes[offset + 1] > 220 && bytes[offset + 2] > 220);
    }
    assert!(
        hidden_pattern.windows(2).any(|pair| pair == [true, false])
            && hidden_pattern.windows(2).any(|pair| pair == [false, true]),
        "hidden edge must alternate ordered white dashes and gaps: {hidden_pattern:?}"
    );
    let near_clipped_yellow_pixels = bytes
        .chunks_exact(4)
        .filter(|pixel| pixel[red] > 180 && pixel[green] > 180 && pixel[blue] < 80)
        .count();
    assert!(
        near_clipped_yellow_pixels >= 4,
        "edge crossing the near plane must remain visible after homogeneous clipping: {near_clipped_yellow_pixels} yellow pixels"
    );
    let different_object_coplanar_pixels = bytes
        .chunks_exact(4)
        .filter(|pixel| pixel[red] < 80 && pixel[green] > 180 && pixel[blue] > 180)
        .count();
    assert!(
        different_object_coplanar_pixels >= 4,
        "a coplanar edge from another object must resolve visible: {different_object_coplanar_pixels} cyan pixels"
    );
    let sub_depth_resolution_pixels = bytes
        .chunks_exact(4)
        .filter(|pixel| pixel[red] > 180 && pixel[green] > 100 && pixel[blue] < 80)
        .count();
    assert!(
        sub_depth_resolution_pixels >= 4,
        "sub-depth-resolution separation must conservatively resolve visible: {sub_depth_resolution_pixels} orange pixels"
    );
    drop(bytes);
    readback.unmap();
}

#[cfg(test)]
pub(super) fn assert_gpu_scene_recovery_contract(
    source_device: &wgpu::Device,
    source_queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
) {
    let source_identity = Arc::new(());
    let recovery_identity = Arc::new(());
    let topology = Mesh3d::with_display_edges(
        vec![
            Vec3::new(-0.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.5, -0.5, 0.0).unwrap(),
            Vec3::new(0.0, 0.5, 0.0).unwrap(),
        ],
        vec![0, 1, 2],
        vec![MeshEdge3d::new(0, 1).unwrap()],
    )
    .unwrap();
    let source_mesh = create_retained_mesh(
        source_device,
        source_queue,
        Arc::clone(&source_identity),
        topology,
    )
    .unwrap();
    let transform_a = Transform3d::IDENTITY;
    let transform_b = Transform3d::new(
        Vec3::new(1.0, 2.0, 3.0).unwrap(),
        Rotation3d::IDENTITY,
        Vec3::new(2.0, 2.0, 2.0).unwrap(),
    )
    .unwrap();
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap())
        .with_wireframe(WireframeStyle3d::visible(Color::BLACK, logical(1.0)).unwrap());
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let first_id = scene.try_push(&source_mesh, transform_a, style).unwrap();
    let second_id = scene.try_push(&source_mesh, transform_b, style).unwrap();
    scene.set_visible(second_id, false).unwrap();

    assert_eq!(
        validate_mesh_identity(&recovery_identity, scene.instances()[0].mesh()),
        Err(Mesh3dRenderError::RendererMismatch)
    );
    let report = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        Arc::clone(&recovery_identity),
        &mut scene,
    )
    .unwrap();
    assert_eq!(report.object_count(), 2);
    assert_eq!(report.migrated_object_count(), 2);
    assert_eq!(report.restored_mesh_count(), 1);
    assert_eq!(
        report.restored_gpu_bytes(),
        source_mesh.gpu_allocation_bytes()
    );
    assert_eq!(scene.instances()[0].id(), first_id);
    assert_eq!(scene.instances()[1].id(), second_id);
    assert_eq!(scene.instances()[0].transform(), transform_a);
    assert_eq!(scene.instances()[1].transform(), transform_b);
    assert_eq!(scene.instances()[0].style(), style);
    assert_eq!(scene.instances()[1].style(), style);
    assert!(scene.instances()[0].is_visible());
    assert!(!scene.instances()[1].is_visible());
    assert!(Arc::ptr_eq(
        &scene.instances()[0].mesh.vertex_buffer,
        &scene.instances()[1].mesh.vertex_buffer,
    ));
    for instance in scene.instances() {
        assert_eq!(
            validate_mesh_identity(&recovery_identity, instance.mesh()),
            Ok(())
        );
        assert_eq!(
            validate_mesh_identity(&source_identity, instance.mesh()),
            Err(Mesh3dRenderError::RendererMismatch)
        );
    }

    let no_op_report = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        recovery_identity,
        &mut scene,
    )
    .unwrap();
    assert_eq!(no_op_report.object_count(), 2);
    assert_eq!(no_op_report.migrated_object_count(), 0);
    assert_eq!(no_op_report.restored_mesh_count(), 0);
    assert_eq!(no_op_report.restored_gpu_bytes(), 0);
}

#[cfg(test)]
fn assert_gpu_clip_equivalence(device: &wgpu::Device, queue: &wgpu::Queue) {
    let inputs = seeded_clip_probe_inputs();
    let input_bytes = bytemuck::cast_slice(&inputs);
    let output_size = (inputs.len() * std::mem::size_of::<ClipProbeOutputGpu>()) as u64;
    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D clip probe input"),
        size: input_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D clip probe output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine 3D clip probe readback"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, input_bytes);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sim-engine 3D clip equivalence shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("mesh3d.wgsl"))),
    });
    let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sim-engine empty clip probe layout"),
        entries: &[],
    });
    let probe_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sim-engine 3D clip probe layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sim-engine 3D clip probe pipeline layout"),
        bind_group_layouts: &[
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&probe_layout),
        ],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sim-engine 3D clip probe pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("mesh3d_clip_probe_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sim-engine 3D clip probe bind group"),
        layout: &probe_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sim-engine 3D clip probe encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sim-engine 3D clip probe pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(2, &bind_group, &[]);
        pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_size);
    queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("3D clip probe submission should complete");
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("3D clip probe mapping should complete");
    receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("3D clip probe callback should run")
        .expect("3D clip probe readback should map");
    let bytes = slice.get_mapped_range().expect("3D clip probe bytes");
    let outputs: &[ClipProbeOutputGpu] = bytemuck::cast_slice(&bytes);
    for (case_index, (input, actual)) in inputs.iter().zip(outputs).enumerate() {
        let expected = clip_edge_to_frustum_details(input.start_clip, input.end_clip)
            .expect("finite seeded clip input should not overflow after normalization");
        assert_eq!(
            actual.visible,
            u32::from(expected.is_some()),
            "CPU/WGSL clip visibility differs for seeded case {case_index}: {input:?}"
        );
        let Some(expected) = expected else {
            continue;
        };
        for (actual, expected) in actual
            .start_clip
            .into_iter()
            .chain(actual.end_clip)
            .chain(actual.range)
            .zip(
                expected.clip[0]
                    .into_iter()
                    .chain(expected.clip[1])
                    .chain([expected.enter, expected.exit]),
            )
        {
            let tolerance = 32.0 * f32::EPSILON * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "CPU/WGSL clip value differs for seeded case {case_index}: actual={actual}, expected={expected}, tolerance={tolerance}, input={input:?}"
            );
        }
    }
    drop(bytes);
    readback.unmap();
}

#[cfg(test)]
fn seeded_clip_probe_inputs() -> Vec<ClipProbeInputGpu> {
    let maximum = f32::from_bits(f32::MAX.to_bits() - 1);
    let mut inputs = vec![
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, 0.5, 1.0],
            end_clip: [0.5, 0.5, 0.75, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [-2.0, 0.0, 0.5, 1.0],
            end_clip: [0.5, 0.0, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, 0.5, 1.0],
            end_clip: [2.0, 0.0, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, -2.0, 0.5, 1.0],
            end_clip: [0.0, 0.5, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, 0.5, 1.0],
            end_clip: [0.0, 2.0, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, -1.0, 1.0],
            end_clip: [0.0, 0.0, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, 0.5, 1.0],
            end_clip: [0.0, 0.0, 2.0, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [-2.0, -2.0, 0.5, 1.0],
            end_clip: [0.5, 0.5, 0.5, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, 0.5, 1.0],
            end_clip: [2.0, 2.0, 2.0, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [-1.0, 1.0, 0.0, 1.0],
            end_clip: [1.0, -1.0, 1.0, 1.0],
        },
        ClipProbeInputGpu {
            start_clip: [maximum, 0.0, maximum * 0.5, maximum],
            end_clip: [-maximum, maximum, maximum, maximum],
        },
        ClipProbeInputGpu {
            start_clip: [0.0, 0.0, -maximum, -maximum],
            end_clip: [0.0, 0.0, -maximum * 0.5, -maximum * 0.5],
        },
    ];
    let mut state = 0x5eed_c1a5_u32;
    for _ in 0..244 {
        let mut next_component = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_215.0) * 4.0 - 2.0
        };
        inputs.push(ClipProbeInputGpu {
            start_clip: [
                next_component(),
                next_component(),
                next_component(),
                next_component(),
            ],
            end_clip: [
                next_component(),
                next_component(),
                next_component(),
                next_component(),
            ],
        });
    }
    debug_assert_eq!(inputs.len(), 256);
    inputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_transform_validation_rejects_finite_overflow() {
        let maximum = f32::MAX;
        let mesh = Mesh3d::new(
            vec![
                Vec3::new(maximum, 0.0, 0.0).unwrap(),
                Vec3::new(maximum, 1.0, 0.0).unwrap(),
                Vec3::new(maximum, 0.0, 1.0).unwrap(),
            ],
            vec![0, 1, 2],
        )
        .unwrap();
        let transform = Transform3d::new(
            Vec3::ZERO,
            Rotation3d::IDENTITY,
            Vec3::new(maximum, 1.0, 1.0).unwrap(),
        )
        .unwrap();
        let camera = Camera3d::look_at(
            Vec3::new(0.0, 0.0, 5.0).unwrap(),
            Vec3::ZERO,
            Vec3::Y,
            Projection3d::orthographic(world(4.0), 1.0, world(0.1), world(20.0)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            validate_shader_transform(
                &mesh,
                transform.model_rows().unwrap(),
                camera.world_to_clip_rows().unwrap(),
            ),
            Err(Mesh3dRenderError::InvalidGeometryTransform)
        );
    }

    #[test]
    fn shader_transform_rejects_pairwise_dot_overflow_after_finite_cpu_transform() {
        let maximum = f32::MAX;
        let angle = 0.5_f32.acos();
        let model_x = -0.2 * maximum;
        let model_z = (0.51 * maximum) / angle.sin();
        let mesh = Mesh3d::with_display_edges(
            vec![
                Vec3::new(model_x, -1.0e30, model_z).unwrap(),
                Vec3::new(model_x, 1.0e30, model_z).unwrap(),
            ],
            Vec::new(),
            vec![MeshEdge3d::new(0, 1).unwrap()],
        )
        .unwrap();
        let transform = Transform3d::new(
            Vec3::new(0.51 * maximum, 0.0, 0.0).unwrap(),
            Rotation3d::from_axis_angle(Vec3::Y, angle).unwrap(),
            Vec3::new(1.0, 1.0, 1.0).unwrap(),
        )
        .unwrap();
        let model_center = Vec3::new(model_x, 0.0, model_z).unwrap();
        let world_center = transform.transform_point(model_center).unwrap();
        assert!(
            world_center.x().is_finite()
                && world_center.y().is_finite()
                && world_center.z().is_finite()
        );

        let camera = Camera3d::look_at(
            Vec3::new(
                world_center.x(),
                world_center.y(),
                world_center.z() - 1.0e32,
            )
            .unwrap(),
            world_center,
            Vec3::Y,
            Projection3d::perspective(2.8, 1.0, world(1.0e30), world(1.0e34)).unwrap(),
        )
        .unwrap();
        assert!(
            camera
                .project_world(world_center, LogicalViewport::new(64.0, 64.0).unwrap())
                .unwrap()
                .inside_view()
        );
        assert_eq!(
            validate_shader_transform(
                &mesh,
                transform.model_rows().unwrap(),
                camera.world_to_clip_rows().unwrap(),
            ),
            Err(Mesh3dRenderError::InvalidGeometryTransform)
        );
    }

    #[test]
    fn shader_transform_accepts_safe_correlated_sparse_vertices() {
        let extent = 0.9 * f32::MAX;
        let mesh = Mesh3d::with_display_edges(
            vec![
                Vec3::new(extent, 0.0, 0.0).unwrap(),
                Vec3::new(0.0, extent, 0.0).unwrap(),
            ],
            Vec::new(),
            vec![MeshEdge3d::new(0, 1).unwrap()],
        )
        .unwrap();
        let transform = Transform3d::new(
            Vec3::ZERO,
            Rotation3d::from_axis_angle(Vec3::Z, -std::f32::consts::FRAC_PI_4).unwrap(),
            Vec3::new(1.0, 1.0, 1.0).unwrap(),
        )
        .unwrap();
        let first = transform.transform_point(mesh.vertices()[0]).unwrap();
        let second = transform.transform_point(mesh.vertices()[1]).unwrap();
        for vertex in [first, second] {
            assert!(vertex.x().is_finite() && vertex.y().is_finite() && vertex.z().is_finite());
        }
        let target = Vec3::new(first.x(), 0.0, 0.0).unwrap();
        let camera = Camera3d::look_at(
            Vec3::new(target.x(), 0.0, 2.0e38).unwrap(),
            target,
            Vec3::Y,
            Projection3d::perspective(2.8, 1.0, world(1.0), world(3.0e38)).unwrap(),
        )
        .unwrap();
        let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
        assert!(camera.project_world(first, viewport).unwrap().inside_view());
        assert!(
            camera
                .project_world(second, viewport)
                .unwrap()
                .inside_view()
        );
        assert_eq!(
            validate_shader_transform(
                &mesh,
                transform.model_rows().unwrap(),
                camera.world_to_clip_rows().unwrap(),
            ),
            Ok(())
        );
    }

    #[test]
    fn edge_projection_rejects_post_width_shader_overflow() {
        let mesh = Mesh3d::with_display_edges(
            vec![
                Vec3::new(-0.5, 0.0, 0.0).unwrap(),
                Vec3::new(0.5, 0.0, 0.0).unwrap(),
            ],
            Vec::new(),
            vec![MeshEdge3d::new(0, 1).unwrap()],
        )
        .unwrap();
        let style = WireframeStyle3d::visible(Color::WHITE, logical(1_048_576.0)).unwrap();
        let camera = Camera3d::look_at(
            Vec3::new(0.0, 0.0, 2.0).unwrap(),
            Vec3::ZERO,
            Vec3::Y,
            Projection3d::orthographic(world(2.0), 1.0, world(0.1), world(10.0)).unwrap(),
        )
        .unwrap();

        assert_eq!(
            validate_edge_projection(
                &mesh,
                Transform3d::IDENTITY.model_rows().unwrap(),
                camera.world_to_clip_rows().unwrap(),
                style,
                [1.0, 1.0, 5.0e32, 0.0],
            ),
            Err(Mesh3dRenderError::InvalidEdgeProjection)
        );
    }

    #[test]
    fn edge_projection_rejects_logical_distance_overflow() {
        let mesh = Mesh3d::with_display_edges(
            vec![
                Vec3::new(-0.5, 0.0, 0.0).unwrap(),
                Vec3::new(0.5, 0.0, 0.0).unwrap(),
            ],
            Vec::new(),
            vec![MeshEdge3d::new(0, 1).unwrap()],
        )
        .unwrap();
        let style = WireframeStyle3d::visible(Color::WHITE, logical(1_048_576.0)).unwrap();
        let camera = Camera3d::look_at(
            Vec3::new(0.0, 0.0, 2.0).unwrap(),
            Vec3::ZERO,
            Vec3::Y,
            Projection3d::orthographic(world(2.0), 1.0, world(0.1), world(10.0)).unwrap(),
        )
        .unwrap();

        assert_eq!(
            validate_edge_projection(
                &mesh,
                Transform3d::IDENTITY.model_rows().unwrap(),
                camera.world_to_clip_rows().unwrap(),
                style,
                [1_024.0, 1_024.0, f32::MIN_POSITIVE, 0.0],
            ),
            Err(Mesh3dRenderError::InvalidEdgeProjection)
        );
    }

    #[test]
    fn edge_projection_rejects_hidden_dash_phase_overflow() {
        let mesh = Mesh3d::with_display_edges(
            vec![
                Vec3::new(-0.5, 0.0, 0.0).unwrap(),
                Vec3::new(0.5, 0.0, 0.0).unwrap(),
            ],
            Vec::new(),
            vec![MeshEdge3d::new(0, 1).unwrap()],
        )
        .unwrap();
        let smallest = logical(f32::MIN_POSITIVE);
        let style = WireframeStyle3d::visible(Color::WHITE, logical(1.0))
            .unwrap()
            .with_hidden(Color::WHITE, logical(1.0), smallest, smallest)
            .unwrap();
        let camera = Camera3d::look_at(
            Vec3::new(0.0, 0.0, 2.0).unwrap(),
            Vec3::ZERO,
            Vec3::Y,
            Projection3d::orthographic(world(2.0), 1.0, world(0.1), world(10.0)).unwrap(),
        )
        .unwrap();

        assert_eq!(
            validate_edge_projection(
                &mesh,
                Transform3d::IDENTITY.model_rows().unwrap(),
                camera.world_to_clip_rows().unwrap(),
                style,
                [1_024.0, 1_024.0, 1.0, 0.0],
            ),
            Err(Mesh3dRenderError::InvalidEdgeProjection)
        );
    }

    #[test]
    fn edge_projection_uses_actual_screen_normal_for_extreme_width() {
        let maximum_width = logical(1_048_576.0);
        let requested_scale = f32::MAX / maximum_width.get();
        let logical_viewport =
            LogicalViewport::new(1.0 / requested_scale, 8_192.0 / requested_scale).unwrap();
        let pixels_per_logical = target_pixels_per_logical(1, 8_192, logical_viewport).unwrap();
        let scaled_width = maximum_width.get() * pixels_per_logical.get();
        assert!(scaled_width.is_finite());

        let view_depth = f32::from_bits(0x7f4d_1ce4);
        let aspect = 1.0 / 8_192.0;
        let vertical_fov = 1.0_f32;
        let horizontal_scale = (vertical_fov * 0.5).tan() * aspect;
        let half_x = view_depth * 0.5 * horizontal_scale;
        let mesh = Mesh3d::with_display_edges(
            vec![
                Vec3::new(-half_x, 0.0, -view_depth).unwrap(),
                Vec3::new(half_x, 0.0, -view_depth).unwrap(),
            ],
            Vec::new(),
            vec![MeshEdge3d::new(0, 1).unwrap()],
        )
        .unwrap();
        let camera = Camera3d::look_at(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0).unwrap(),
            Vec3::Y,
            Projection3d::perspective(vertical_fov, aspect, world(1.0), world(f32::MAX)).unwrap(),
        )
        .unwrap();
        let style = WireframeStyle3d::visible(Color::WHITE, maximum_width).unwrap();

        assert_eq!(
            validate_edge_projection(
                &mesh,
                Transform3d::IDENTITY.model_rows().unwrap(),
                camera.world_to_clip_rows().unwrap(),
                style,
                [1.0, 8_192.0, pixels_per_logical.get(), 0.0],
            ),
            Ok(())
        );
    }

    #[test]
    fn scene3d_rejects_out_of_range_clear_colors() {
        assert!(matches!(
            Scene3d::new(Color::rgb(1.01, 0.0, 0.0)),
            Err(Scene3dError::InvalidBackground)
        ));
    }

    #[test]
    fn edge_crossing_near_plane_is_clipped_before_wgsl_division() {
        let mesh = Mesh3d::with_display_edges(
            vec![
                Vec3::new(0.0, 0.0, -0.05).unwrap(),
                Vec3::new(0.5, 0.0, -1.0).unwrap(),
            ],
            Vec::new(),
            vec![MeshEdge3d::new(0, 1).unwrap()],
        )
        .unwrap();
        let camera = Camera3d::look_at(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0).unwrap(),
            Vec3::Y,
            Projection3d::perspective(1.0, 1.0, world(0.1), world(10.0)).unwrap(),
        )
        .unwrap();
        let style = WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap();
        assert_eq!(
            validate_edge_projection(
                &mesh,
                Transform3d::IDENTITY.model_rows().unwrap(),
                camera.world_to_clip_rows().unwrap(),
                style,
                [100.0, 100.0, 1.0, 0.0],
            ),
            Ok(())
        );
    }

    #[test]
    fn fully_clipped_edges_do_not_reject_the_frame() {
        let mesh = Mesh3d::with_display_edges(
            vec![
                Vec3::new(0.0, 0.0, -0.01).unwrap(),
                Vec3::new(0.01, 0.0, -0.05).unwrap(),
            ],
            Vec::new(),
            vec![MeshEdge3d::new(0, 1).unwrap()],
        )
        .unwrap();
        let camera = Camera3d::look_at(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0).unwrap(),
            Vec3::Y,
            Projection3d::perspective(1.0, 1.0, world(0.1), world(10.0)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            validate_edge_projection(
                &mesh,
                Transform3d::IDENTITY.model_rows().unwrap(),
                camera.world_to_clip_rows().unwrap(),
                WireframeStyle3d::visible(Color::WHITE, logical(1.0)).unwrap(),
                [100.0, 100.0, 1.0, 0.0],
            ),
            Ok(())
        );
    }

    #[test]
    fn homogeneous_edge_clipping_covers_near_far_and_camera_half_spaces() {
        let near = clip_edge_to_frustum([0.0, 0.0, -0.5, 1.0], [0.0, 0.0, 0.5, 1.0])
            .unwrap()
            .unwrap();
        assert!(near[0][2].abs() <= f32::EPSILON);
        assert_eq!(near[1], [0.0, 0.0, 0.5, 1.0]);

        let far = clip_edge_to_frustum([0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 2.0, 1.0])
            .unwrap()
            .unwrap();
        assert!((far[1][2] - far[1][3]).abs() <= f32::EPSILON);

        assert_eq!(
            clip_edge_to_frustum([0.0, 0.0, -2.0, -1.0], [0.0, 0.0, -1.0, -0.5]),
            Ok(None)
        );
        assert_eq!(
            clip_edge_to_frustum([-3.0, 0.0, 0.5, 1.0], [-2.0, 0.0, 0.5, 1.0]),
            Ok(None)
        );
    }

    #[test]
    fn target_pixel_scale_is_independent_of_window_dpi() {
        let viewport = LogicalViewport::new(1000.0, 500.0).unwrap();
        assert_eq!(
            target_pixels_per_logical(2000, 1000, viewport),
            Some(physical_per_logical(2.0))
        );
        assert_eq!(
            target_pixels_per_logical(500, 250, viewport),
            Some(physical_per_logical(0.5))
        );
        assert_eq!(target_pixels_per_logical(500, 300, viewport), None);
        assert!(aspect_matches(16.0 / 9.0, 1920.0 / 1080.0));
        assert!(!aspect_matches(16.0 / 9.0, 4.0 / 3.0));
    }

    #[test]
    fn camera_target_aspect_mismatch_is_rejected_by_render_contract() {
        let camera = Camera3d::look_at(
            Vec3::new(0.0, 0.0, 5.0).unwrap(),
            Vec3::ZERO,
            Vec3::Y,
            Projection3d::perspective(1.0, 16.0 / 9.0, world(0.1), world(10.0)).unwrap(),
        )
        .unwrap();
        let target = LogicalViewport::new(800.0, 600.0).unwrap();
        assert_eq!(
            validate_camera_target_aspect(camera, target),
            Err(Mesh3dRenderError::CameraTargetAspectMismatch)
        );
    }

    #[test]
    fn logical_edge_width_is_equal_for_native_and_half_resolution_target() {
        let logical_width = logical(1.0);
        for pixels_per_logical in [physical_per_logical(2.0), physical_per_logical(0.5)] {
            let physical_width = edge_raster_envelope(logical_width, pixels_per_logical)
                .unwrap()
                .physical_half_width
                * 2.0;
            assert_eq!(
                physical_width / pixels_per_logical.get(),
                logical_width.get()
            );
        }
    }

    #[test]
    fn mesh_upload_preflight_rejects_each_buffer_before_staging_allocation() {
        let max_buffer_size = 48;
        let layout = preflight_mesh3d_upload(4, 12, 2, max_buffer_size).unwrap();
        assert_eq!(layout.vertex_bytes, 48);
        assert_eq!(layout.index_bytes, 48);
        assert_eq!(layout.edge_bytes, 48);
        assert_eq!(layout.total_bytes, 144);

        assert_eq!(
            preflight_mesh3d_upload(5, 0, 0, max_buffer_size),
            Err(Mesh3dResourceError::CapacityTooLarge)
        );
        assert_eq!(
            preflight_mesh3d_upload(1, 13, 0, max_buffer_size),
            Err(Mesh3dResourceError::CapacityTooLarge)
        );
        assert_eq!(
            preflight_mesh3d_upload(1, 0, 3, max_buffer_size),
            Err(Mesh3dResourceError::CapacityTooLarge)
        );

        if usize::BITS > u32::BITS {
            let first_draw_count_overflow = u32::MAX as usize + 1;
            assert_eq!(
                preflight_mesh3d_upload(1, first_draw_count_overflow, 0, u64::MAX),
                Err(Mesh3dResourceError::CapacityTooLarge)
            );
            assert_eq!(
                preflight_mesh3d_upload(1, 0, first_draw_count_overflow, u64::MAX),
                Err(Mesh3dResourceError::CapacityTooLarge)
            );
        }
    }
}
