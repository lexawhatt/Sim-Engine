//! Immutable opaque texture materials for the retained 3D surface path.

use super::*;
use crate::mesh3d::TextureCoordinate2d;

#[cfg(test)]
#[path = "mesh3d_texture_red_tests.rs"]
mod red_tests;
#[cfg(test)]
#[path = "mesh3d_texture_tests.rs"]
mod tests;
#[cfg(test)]
pub(super) use tests::{assert_gpu_texture_contract, assert_gpu_textured_recovery};

/// Validation or allocation failure for an opaque retained 3D texture/material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Texture3dError {
    /// Image dimensions, retained bytes, allocation or device limits failed.
    Image(ImageError),
    /// The first texel with alpha other than 255 in row-major source order.
    NonOpaquePixel {
        /// Zero-based source texel index.
        texel: usize,
    },
    /// A material tint must be normalized linear RGBA with alpha exactly one.
    InvalidTint,
    /// Attaching a texture requires a surface and one UV per mesh vertex.
    MissingTextureCoordinates,
    /// Texture and mesh must belong to the renderer's current device identity.
    RendererMismatch,
    /// An atlas region does not fit the texture.
    InvalidRegion,
}

impl fmt::Display for Texture3dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(error) => write!(formatter, "3D texture: {error}"),
            Self::NonOpaquePixel { texel } => {
                write!(formatter, "3D texture texel {texel} is not opaque")
            }
            Self::InvalidTint => write!(formatter, "3D texture tint must be normalized and opaque"),
            Self::MissingTextureCoordinates => {
                write!(formatter, "textured 3D surfaces require per-vertex UVs")
            }
            Self::RendererMismatch => {
                write!(formatter, "3D texture/mesh belongs to another renderer")
            }
            Self::InvalidRegion => write!(formatter, "3D texture atlas region is out of bounds"),
        }
    }
}
impl Error for Texture3dError {}

struct TextureStorage {
    renderer_identity: Arc<()>,
    image: Image2d,
    nearest: wgpu::BindGroup,
    linear: wgpu::BindGroup,
}

/// Cheap shared handle for immutable opaque, single-level sRGB RGBA8 texels.
///
/// Clones share CPU recovery pixels and the GPU allocation. U/V has a top-left
/// origin; addressing clamps to the image boundary, with no mipmap generation.
/// Only alpha 255 is accepted. Restoring returns a new device-owned handle;
/// existing clones remain stale until explicitly rebound or scene-restored.
#[derive(Clone)]
pub struct Texture3d {
    storage: Arc<TextureStorage>,
}

impl Texture3d {
    /// Returns physical source texel dimensions.
    pub fn size(&self) -> (u32, u32) {
        self.storage.image.size()
    }
    /// Returns immutable retained row-major, top-to-bottom sRGB RGBA8 texels.
    pub fn pixels(&self) -> &[u8] {
        self.storage.image.pixels()
    }
    /// Returns creation limits retained for restoration.
    pub fn budget(&self) -> ImageBudget {
        self.storage.image.budget()
    }
    /// Returns shared CPU recovery-pixel capacity, excluding fixed handle metadata.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.storage.image.recovery_memory_bytes()
    }
    /// Returns nominal single-level GPU texel bytes, excluding driver metadata.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.storage.image.gpu_allocation_bytes()
    }
    /// Maps top-left, top-right, bottom-right and bottom-left corners of an
    /// atlas region to its outer texel centers. Nearest sampling avoids adjacent
    /// cells under this mip-zero policy. For strict linear-filter isolation,
    /// provide host-owned extruded gutters around atlas cells: interpolation
    /// rounding can extend the filtering footprint slightly beyond a center.
    /// A one-texel extent maps both edges to its center.
    pub fn region_coordinates(
        &self,
        region: ImageTexelRect,
    ) -> Result<[TextureCoordinate2d; 4], Texture3dError> {
        let (width, height) = self.size();
        if !region.fits(width, height) {
            return Err(Texture3dError::InvalidRegion);
        }
        let left = (region.x() as f64 + 0.5) / f64::from(width);
        let right = (f64::from(region.x()) + f64::from(region.width()) - 0.5) / f64::from(width);
        let top = (region.y() as f64 + 0.5) / f64::from(height);
        let bottom = (f64::from(region.y()) + f64::from(region.height()) - 0.5) / f64::from(height);
        let make = |u: f64, v: f64| {
            TextureCoordinate2d::new(u as f32, v as f32).map_err(|_| Texture3dError::InvalidRegion)
        };
        Ok([
            make(left, top)?,
            make(right, top)?,
            make(right, bottom)?,
            make(left, bottom)?,
        ])
    }
    pub(super) fn identity_key(&self) -> usize {
        Arc::as_ptr(&self.storage) as usize
    }
    pub(super) fn belongs_to(&self, identity: &Arc<()>) -> bool {
        Arc::ptr_eq(identity, &self.storage.renderer_identity)
    }
    fn bind_group(&self, sampling: ImageSampling) -> &wgpu::BindGroup {
        match sampling {
            ImageSampling::Nearest => &self.storage.nearest,
            ImageSampling::Linear => &self.storage.linear,
        }
    }
}

/// Cheap immutable unlit material sharing one opaque texture and filter state.
///
/// Sampled linear RGB is multiplied by this normalized opaque tint and the
/// object's `SurfaceStyle3d` color. Depth writes/tests and no-culling winding
/// semantics are identical to ordinary opaque surfaces. Transparency, lighting
/// and alpha-cutout are intentionally separate capabilities.
#[derive(Clone)]
pub struct TextureMaterial3d {
    texture: Texture3d,
    sampling: ImageSampling,
    tint: Color,
}

impl TextureMaterial3d {
    /// Creates an immutable material; tint is normalized and fully opaque.
    pub fn new(
        texture: &Texture3d,
        sampling: ImageSampling,
        tint: Color,
    ) -> Result<Self, Texture3dError> {
        if !tint.is_normalized() || tint.alpha() != 1.0 {
            return Err(Texture3dError::InvalidTint);
        }
        Ok(Self {
            texture: texture.clone(),
            sampling,
            tint,
        })
    }
    /// Returns the shared immutable texture handle.
    pub const fn texture(&self) -> &Texture3d {
        &self.texture
    }
    /// Returns nearest or linear mip-zero sampling.
    pub const fn sampling(&self) -> ImageSampling {
        self.sampling
    }
    /// Returns multiplicative opaque linear-RGBA tint.
    pub const fn tint(&self) -> Color {
        self.tint
    }
    pub(super) fn bind_group(&self) -> &wgpu::BindGroup {
        self.texture.bind_group(self.sampling)
    }
    pub(super) fn with_restored_texture(&self, texture: &Texture3d) -> Self {
        Self {
            texture: texture.clone(),
            sampling: self.sampling,
            tint: self.tint,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MeshUvGpu {
    pub(super) coordinate: [f32; 2],
}

impl MeshUvGpu {
    const ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![5 => Float32x2];
    pub(super) const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };
}

pub(super) struct MeshTextureRenderer {
    pub(super) layout: wgpu::BindGroupLayout,
    pub(super) retained_pipeline: wgpu::RenderPipeline,
    pub(super) clipped_pipeline: wgpu::RenderPipeline,
    pub(super) colored_retained_pipeline: wgpu::RenderPipeline,
    pub(super) colored_clipped_pipeline: wgpu::RenderPipeline,
}

impl MeshTextureRenderer {
    pub(super) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine 3D texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sim-engine textured 3D surfaces"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("mesh3d_texture.wgsl"))),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-engine textured 3D pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |clipped: bool, colored: bool| {
            let mut buffers = if clipped {
                vec![
                    Some(SurfaceClipVertex::TEXTURED_LAYOUT),
                    Some(MeshInstanceGpu::LAYOUT),
                ]
            } else {
                vec![
                    Some(MeshVertexGpu::LAYOUT),
                    Some(MeshUvGpu::LAYOUT),
                    Some(MeshInstanceGpu::LAYOUT),
                ]
            };
            if colored {
                buffers.push(Some(MeshColorGpu::LAYOUT));
            }
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sim-engine opaque textured 3D pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(match (clipped, colored) {
                        (false, false) => "retained_vs_main",
                        (true, false) => "clipped_vs_main",
                        (false, true) => "colored_retained_vs_main",
                        (true, true) => "colored_clipped_vs_main",
                    }),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &buffers,
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
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            retained_pipeline: pipeline(false, false),
            clipped_pipeline: pipeline(true, false),
            colored_retained_pipeline: pipeline(false, true),
            colored_clipped_pipeline: pipeline(true, true),
            layout,
        }
    }
}

impl WgpuRenderer {
    /// Creates immutable opaque sRGB RGBA8 texels with shared nearest/linear bindings.
    /// Image/device limits and every alpha byte are validated before GPU creation.
    pub fn create_texture3d_rgba8(
        &self,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        budget: ImageBudget,
    ) -> Result<Texture3d, Texture3dError> {
        create_texture(
            &self.device,
            &self.queue,
            &self.renderer_identity,
            &self.mesh3d_renderer.textures.layout,
            width,
            height,
            pixels,
            budget,
        )
    }

    /// Restores exact retained pixels on the current device. Existing handles
    /// remain unchanged; atomic `restore_scene3d` restores shared textures once.
    pub fn restore_texture3d(&self, source: &Texture3d) -> Result<Texture3d, Texture3dError> {
        image::validate_image_shape(
            source.size().0,
            source.size().1,
            source.pixels().len(),
            source.budget(),
            self.device.limits().max_texture_dimension_2d,
        )
        .map_err(Texture3dError::Image)?;
        let pixels = copy_recovery_pixels(source)?;
        self.create_texture3d_rgba8(source.size().0, source.size().1, pixels, source.budget())
    }

    /// Associates a shared material with a new retained mesh handle without
    /// allocating/uploading topology. Existing clones and scene users keep
    /// their previous material; rebind selected IDs using `Scene3d::set_mesh`.
    pub fn with_mesh3d_material(
        &self,
        mesh: &RetainedMesh3d,
        material: &TextureMaterial3d,
    ) -> Result<RetainedMesh3d, Texture3dError> {
        attach_material(&self.renderer_identity, mesh, material)
    }
}

pub(super) fn attach_material(
    identity: &Arc<()>,
    mesh: &RetainedMesh3d,
    material: &TextureMaterial3d,
) -> Result<RetainedMesh3d, Texture3dError> {
    if !Arc::ptr_eq(identity, &mesh.renderer_identity) || !material.texture.belongs_to(identity) {
        return Err(Texture3dError::RendererMismatch);
    }
    if mesh.source.texture_coordinates().len() != mesh.source.vertices().len()
        || mesh.source.triangle_count() == 0
        || mesh.texture_coordinate_buffer.is_none()
    {
        return Err(Texture3dError::MissingTextureCoordinates);
    }
    let mut result = mesh.clone();
    result.material = Some(material.clone());
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    layout: &wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    budget: ImageBudget,
) -> Result<Texture3d, Texture3dError> {
    image::validate_image_shape(
        width,
        height,
        pixels.len(),
        budget,
        device.limits().max_texture_dimension_2d,
    )
    .map_err(Texture3dError::Image)?;
    if let Some(texel) = pixels.chunks_exact(4).position(|pixel| pixel[3] != 255) {
        return Err(Texture3dError::NonOpaquePixel { texel });
    }
    let image = image::create_image_resources(
        device,
        queue,
        Arc::clone(identity),
        width,
        height,
        pixels,
        budget,
    )
    .map_err(Texture3dError::Image)?;
    let binding = |filter| {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sim-engine 3D texture sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine 3D texture binding"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        })
    };
    let nearest = binding(wgpu::FilterMode::Nearest);
    let linear = binding(wgpu::FilterMode::Linear);
    Ok(Texture3d {
        storage: Arc::new(TextureStorage {
            renderer_identity: Arc::clone(identity),
            image,
            nearest,
            linear,
        }),
    })
}

pub(super) fn copy_recovery_pixels(source: &Texture3d) -> Result<Vec<u8>, Texture3dError> {
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(source.pixels().len())
        .map_err(|_| {
            Texture3dError::Image(ImageError::AllocationFailed {
                requested_bytes: source.pixels().len(),
            })
        })?;
    pixels.extend_from_slice(source.pixels());
    if pixels.capacity() > source.budget().max_bytes() {
        return Err(Texture3dError::Image(ImageError::BudgetExceeded {
            limit: source.budget().max_bytes(),
            actual: pixels.capacity(),
        }));
    }
    Ok(pixels)
}

pub(super) struct PreparedTextureRestore {
    pub(super) key: usize,
    pub(super) source: Texture3d,
    pub(super) pixels: Vec<u8>,
}

pub(super) fn prepare_scene_restoration(
    device: &wgpu::Device,
    identity: &Arc<()>,
    scene: &Scene3d,
) -> Result<Vec<PreparedTextureRestore>, Mesh3dResourceError> {
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(scene.object_count())
        .map_err(|_| Mesh3dResourceError::HostAllocationFailed {
            requested_bytes: scene
                .object_count()
                .saturating_mul(std::mem::size_of::<PreparedTextureRestore>())
                as u64,
        })?;
    for instance in scene.instances() {
        if let Some(material) = instance.mesh.material()
            && !material.texture().belongs_to(identity)
        {
            prepared.push(PreparedTextureRestore {
                key: material.texture().identity_key(),
                source: material.texture().clone(),
                pixels: Vec::new(),
            });
        }
    }
    prepared.sort_unstable_by_key(|texture| texture.key);
    prepared.dedup_by_key(|texture| texture.key);
    for texture in &mut prepared {
        let source = &texture.source;
        image::validate_image_shape(
            source.size().0,
            source.size().1,
            source.pixels().len(),
            source.budget(),
            device.limits().max_texture_dimension_2d,
        )
        .map_err(|error| Mesh3dResourceError::Texture(Texture3dError::Image(error)))?;
        texture.pixels = copy_recovery_pixels(source).map_err(Mesh3dResourceError::Texture)?;
    }
    Ok(prepared)
}
