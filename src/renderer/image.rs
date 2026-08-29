use super::*;

const IMAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Immutable resource limits retained with an [`Image2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageBudget {
    max_width: u32,
    max_height: u32,
    max_bytes: usize,
}

impl ImageBudget {
    /// Creates non-zero image dimension and byte limits.
    pub fn new(max_width: u32, max_height: u32, max_bytes: usize) -> Result<Self, ImageError> {
        if max_width == 0 || max_height == 0 || max_bytes < 4 {
            return Err(ImageError::InvalidBudget);
        }
        Ok(Self {
            max_width,
            max_height,
            max_bytes,
        })
    }

    /// Returns maximum image width in physical texels.
    pub const fn max_width(self) -> u32 {
        self.max_width
    }

    /// Returns maximum image height in physical texels.
    pub const fn max_height(self) -> u32 {
        self.max_height
    }

    /// Returns maximum retained and GPU RGBA bytes.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
}

impl Default for ImageBudget {
    fn default() -> Self {
        Self {
            max_width: 4096,
            max_height: 4096,
            max_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Validated non-empty atlas rectangle in physical source texels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageTexelRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl ImageTexelRect {
    /// Creates a non-empty source rectangle. Image bounds are checked at use.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Result<Self, ImageError> {
        if width == 0 || height == 0 {
            return Err(ImageError::ZeroDimension);
        }
        x.checked_add(width)
            .zip(y.checked_add(height))
            .ok_or(ImageError::DimensionsTooLarge)?;
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    /// Returns the left source texel.
    pub const fn x(self) -> u32 {
        self.x
    }

    /// Returns the top source texel.
    pub const fn y(self) -> u32 {
        self.y
    }

    /// Returns source width in texels.
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns source height in texels.
    pub const fn height(self) -> u32 {
        self.height
    }

    pub(super) fn fits(self, width: u32, height: u32) -> bool {
        self.x
            .checked_add(self.width)
            .is_some_and(|maximum| maximum <= width)
            && self
                .y
                .checked_add(self.height)
                .is_some_and(|maximum| maximum <= height)
    }
}

/// Texture filtering used by a composed image or atlas region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageSampling {
    /// Preserve exact texel boundaries for pixel art and masks.
    #[default]
    Nearest,
    /// Bilinearly interpolate neighboring source texels.
    Linear,
}

/// Failure while creating, restoring, or updating an RGBA image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageError {
    /// A width or height was zero.
    ZeroDimension,
    /// A budget must fit at least one RGBA texel.
    InvalidBudget,
    /// Dimensions or their RGBA byte arithmetic exceed a configured/device limit.
    DimensionsTooLarge,
    /// Pixel data does not contain exactly width × height × four bytes.
    InvalidPixelCount,
    /// The image exceeds its host-selected byte limit.
    BudgetExceeded {
        /// Configured byte ceiling.
        limit: usize,
        /// Required RGBA bytes.
        actual: usize,
    },
    /// CPU storage for an owned recovery snapshot could not be reserved.
    AllocationFailed {
        /// Bytes requested by the failed reservation.
        requested_bytes: usize,
    },
    /// A retained image belongs to another renderer generation.
    RendererMismatch,
    /// A partial update does not fit in the image.
    UpdateRegionOutOfBounds,
    /// A sprite source rectangle or tint is invalid for its image.
    InvalidSprite,
    /// A sprite batch exceeds its count or retained/GPU byte limit.
    BatchBudgetExceeded {
        /// Configured upper bound.
        limit: usize,
        /// Requested count or bytes.
        actual: usize,
    },
}

impl fmt::Display for ImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(formatter, "image dimensions must be non-zero"),
            Self::InvalidBudget => write!(formatter, "image budget must fit one RGBA texel"),
            Self::DimensionsTooLarge => write!(formatter, "image dimensions exceed limits"),
            Self::InvalidPixelCount => write!(formatter, "image RGBA byte count is invalid"),
            Self::BudgetExceeded { limit, actual } => {
                write!(
                    formatter,
                    "image requires {actual} bytes, over limit {limit}"
                )
            }
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for image recovery data"
            ),
            Self::RendererMismatch => write!(formatter, "image belongs to another renderer"),
            Self::UpdateRegionOutOfBounds => write!(formatter, "image update is out of bounds"),
            Self::InvalidSprite => write!(formatter, "image sprite source or tint is invalid"),
            Self::BatchBudgetExceeded { limit, actual } => {
                write!(formatter, "image batch work {actual} exceeds limit {limit}")
            }
        }
    }
}

impl Error for ImageError {}

/// CPU outcome of a complete or partial image upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageUploadReport {
    uploaded_bytes: usize,
    replaced_texture: bool,
}

impl ImageUploadReport {
    /// Returns RGBA bytes submitted to the queue.
    pub const fn uploaded_bytes(self) -> usize {
        self.uploaded_bytes
    }

    /// Returns whether the operation replaced the underlying texture.
    pub const fn replaced_texture(self) -> bool {
        self.replaced_texture
    }
}

/// Renderer-owned straight-alpha sRGB RGBA image with exact recovery pixels.
///
/// Source bytes use row-major top-to-bottom `R8 G8 B8 A8` texels. The sRGB GPU
/// texture decodes RGB to linear light before tinting and alpha composition;
/// alpha remains linear. Retained bytes make device-loss restoration exact.
pub struct Image2d {
    renderer_identity: Arc<()>,
    pub(super) resource_identity: Arc<()>,
    texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    budget: ImageBudget,
}

impl Image2d {
    /// Returns image width in source texels.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns image height in source texels.
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns complete image dimensions.
    pub const fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the retained creation/replacement budget.
    pub const fn budget(&self) -> ImageBudget {
        self.budget
    }

    /// Returns exact CPU recovery pixels in row-major sRGB RGBA8 form.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Returns retained CPU recovery bytes.
    pub fn recovery_memory_bytes(&self) -> usize {
        self.pixels.len()
    }

    /// Returns exact single-level GPU texture bytes.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.pixels.len()
    }

    /// Returns a full-image atlas rectangle.
    pub fn full_rect(&self) -> ImageTexelRect {
        ImageTexelRect {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        }
    }
}

/// One validated atlas region placed in local logical-screen coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageSprite2d {
    source: ImageTexelRect,
    destination: LogicalViewportRegion,
    tint: Color,
}

impl ImageSprite2d {
    /// Creates a sprite. Atlas bounds are checked against an image at upload.
    pub fn new(
        source: ImageTexelRect,
        destination: LogicalViewportRegion,
        tint: Color,
    ) -> Result<Self, ImageError> {
        let origin = destination.origin().to_vec2();
        let end = origin + destination.viewport().size();
        if !tint.is_normalized() || !origin.is_finite() || !end.is_finite() {
            return Err(ImageError::InvalidSprite);
        }
        Ok(Self {
            source,
            destination,
            tint,
        })
    }

    /// Returns its physical atlas source rectangle.
    pub const fn source(self) -> ImageTexelRect {
        self.source
    }

    /// Returns its local logical destination rectangle.
    pub const fn destination(self) -> LogicalViewportRegion {
        self.destination
    }

    /// Returns its normalized straight-linear tint.
    pub const fn tint(self) -> Color {
        self.tint
    }
}

/// Hard retained-count and byte limits for an [`ImageBatch2d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageBatchBudget {
    max_sprites: usize,
    max_retained_bytes: usize,
}

impl ImageBatchBudget {
    /// Creates non-zero sprite-count and retained-byte limits.
    pub fn new(max_sprites: usize, max_retained_bytes: usize) -> Result<Self, ImageError> {
        if max_sprites == 0 || max_retained_bytes < std::mem::size_of::<ImageSprite2d>() {
            return Err(ImageError::InvalidBudget);
        }
        Ok(Self {
            max_sprites,
            max_retained_bytes,
        })
    }

    /// Returns maximum sprite instances.
    pub const fn max_sprites(self) -> usize {
        self.max_sprites
    }

    /// Returns maximum retained sprite-description bytes.
    pub const fn max_retained_bytes(self) -> usize {
        self.max_retained_bytes
    }
}

impl Default for ImageBatchBudget {
    fn default() -> Self {
        Self {
            max_sprites: 100_000,
            max_retained_bytes: 16 * 1024 * 1024,
        }
    }
}

/// Renderer-owned atlas sprite batch drawn with one instanced draw call.
///
/// Sprite positions are local logical pixels and can be placed in any frame
/// viewport without rebuilding or uploading the batch. The exact CPU sprite
/// list is retained for renderer recovery.
pub struct ImageBatch2d {
    renderer_identity: Arc<()>,
    image_identity: Arc<()>,
    pub(super) instance_buffer: wgpu::Buffer,
    sprites: Vec<ImageSprite2d>,
    budget: ImageBatchBudget,
}

impl ImageBatch2d {
    /// Returns retained sprite count.
    pub fn sprite_count(&self) -> usize {
        self.sprites.len()
    }

    /// Returns the exact retained sprite descriptions.
    pub fn sprites(&self) -> &[ImageSprite2d] {
        &self.sprites
    }

    /// Returns its creation/replacement budget.
    pub const fn budget(&self) -> ImageBatchBudget {
        self.budget
    }

    /// Returns retained CPU recovery bytes.
    pub fn recovery_memory_bytes(&self) -> usize {
        std::mem::size_of_val(self.sprites.as_slice())
    }

    /// Returns GPU instance-buffer bytes actively addressed by the batch.
    pub fn gpu_allocation_bytes(&self) -> usize {
        self.sprites
            .len()
            .max(1)
            .saturating_mul(std::mem::size_of::<ImageInstance>())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ImageInstance {
    pub(super) destination: [f32; 4],
    pub(super) uv_rect: [f32; 4],
    pub(super) tint: [f32; 4],
}

impl ImageInstance {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4];
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &Self::ATTRIBUTES,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ImageUniform {
    pub(super) destination: [f32; 4],
    pub(super) uv_rect: [f32; 4],
    pub(super) tint: [f32; 4],
    pub(super) world_clip_x: [f32; 4],
    pub(super) world_clip_y: [f32; 4],
    pub(super) world_mode: [f32; 4],
}

pub(super) struct ImageRenderer {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) batch_pipeline: wgpu::RenderPipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    nearest_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
}

impl ImageRenderer {
    pub(super) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sim-engine image shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("image.wgsl"))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine image bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-engine image pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sim-engine image pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("image_vs_main"),
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
                entry_point: Some("image_fs_main"),
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
        let batch_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sim-engine image batch pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("image_batch_vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(ImageInstance::LAYOUT)],
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
                entry_point: Some("image_fs_main"),
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
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sim-engine nearest image sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sim-engine linear image sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            batch_pipeline,
            bind_group_layout,
            nearest_sampler,
            linear_sampler,
        }
    }

    pub(super) fn sampler(&self, sampling: ImageSampling) -> &wgpu::Sampler {
        match sampling {
            ImageSampling::Nearest => &self.nearest_sampler,
            ImageSampling::Linear => &self.linear_sampler,
        }
    }
}

impl WgpuRenderer {
    /// Creates an image while taking ownership of its exact recovery pixels.
    pub fn create_image_rgba8(
        &self,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        budget: ImageBudget,
    ) -> Result<Image2d, ImageError> {
        create_image_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            width,
            height,
            pixels,
            budget,
        )
    }

    /// Copies a bounded byte slice into a retained image recovery snapshot.
    pub fn create_image_rgba8_from_slice(
        &self,
        width: u32,
        height: u32,
        pixels: &[u8],
        budget: ImageBudget,
    ) -> Result<Image2d, ImageError> {
        let byte_count = validate_image_shape(
            width,
            height,
            pixels.len(),
            budget,
            self.device.limits().max_texture_dimension_2d,
            usize::try_from(self.device.limits().max_buffer_size).unwrap_or(usize::MAX),
        )?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(byte_count)
            .map_err(|_| ImageError::AllocationFailed {
                requested_bytes: byte_count,
            })?;
        owned.extend_from_slice(pixels);
        self.create_image_rgba8(width, height, owned, budget)
    }

    /// Replaces dimensions and pixels atomically after full preflight.
    pub fn replace_image_rgba8(
        &self,
        image: &mut Image2d,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Result<ImageUploadReport, ImageError> {
        self.validate_image(image)?;
        let replacement = self.create_image_rgba8(width, height, pixels, image.budget)?;
        let uploaded_bytes = replacement.pixels.len();
        *image = replacement;
        Ok(ImageUploadReport {
            uploaded_bytes,
            replaced_texture: true,
        })
    }

    /// Uploads one exact row-major RGBA region and updates recovery pixels.
    pub fn update_image_region(
        &self,
        image: &mut Image2d,
        region: ImageTexelRect,
        pixels: &[u8],
    ) -> Result<ImageUploadReport, ImageError> {
        self.validate_image(image)?;
        if !region.fits(image.width, image.height) {
            return Err(ImageError::UpdateRegionOutOfBounds);
        }
        let required =
            image_byte_count(region.width, region.height).ok_or(ImageError::DimensionsTooLarge)?;
        if pixels.len() != required {
            return Err(ImageError::InvalidPixelCount);
        }
        write_image_texture(
            &self.queue,
            &image.texture,
            region.x,
            region.y,
            region.width,
            region.height,
            pixels,
        );
        let source_stride = region.width as usize * 4;
        let destination_stride = image.width as usize * 4;
        for row in 0..region.height as usize {
            let source_start = row * source_stride;
            let destination_start =
                (region.y as usize + row) * destination_stride + region.x as usize * 4;
            image.pixels[destination_start..destination_start + source_stride]
                .copy_from_slice(&pixels[source_start..source_start + source_stride]);
        }
        Ok(ImageUploadReport {
            uploaded_bytes: required,
            replaced_texture: false,
        })
    }

    /// Recreates an exact image for this renderer from retained source bytes.
    pub fn restore_image(&self, source: &Image2d) -> Result<Image2d, ImageError> {
        self.create_image_rgba8_from_slice(
            source.width,
            source.height,
            &source.pixels,
            source.budget,
        )
    }

    /// Creates one retained instanced batch tied to an image or atlas.
    pub fn create_image_batch(
        &self,
        image: &Image2d,
        sprites: Vec<ImageSprite2d>,
        budget: ImageBatchBudget,
    ) -> Result<ImageBatch2d, ImageError> {
        self.validate_image(image)?;
        create_image_batch_resources(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            image,
            sprites,
            budget,
        )
    }

    /// Atomically replaces a batch after validating every source rectangle.
    pub fn replace_image_batch(
        &self,
        image: &Image2d,
        batch: &mut ImageBatch2d,
        sprites: Vec<ImageSprite2d>,
    ) -> Result<ImageUploadReport, ImageError> {
        self.validate_image_batch(image, batch)?;
        let replacement = self.create_image_batch(image, sprites, batch.budget)?;
        let uploaded_bytes = replacement
            .sprites
            .len()
            .saturating_mul(std::mem::size_of::<ImageInstance>());
        *batch = replacement;
        Ok(ImageUploadReport {
            uploaded_bytes,
            replaced_texture: true,
        })
    }

    /// Restores a batch against the restored copy of its original atlas.
    pub fn restore_image_batch(
        &self,
        image: &Image2d,
        source: &ImageBatch2d,
    ) -> Result<ImageBatch2d, ImageError> {
        let mut sprites = Vec::new();
        sprites
            .try_reserve_exact(source.sprites.len())
            .map_err(|_| ImageError::AllocationFailed {
                requested_bytes: source.recovery_memory_bytes(),
            })?;
        sprites.extend_from_slice(&source.sprites);
        self.create_image_batch(image, sprites, source.budget)
    }

    pub(super) fn validate_image(&self, image: &Image2d) -> Result<(), ImageError> {
        prepared_scene_belongs_to(&self.renderer_identity, &image.renderer_identity)
            .then_some(())
            .ok_or(ImageError::RendererMismatch)
    }

    pub(super) fn validate_image_batch(
        &self,
        image: &Image2d,
        batch: &ImageBatch2d,
    ) -> Result<(), ImageError> {
        self.validate_image(image)?;
        if !prepared_scene_belongs_to(&self.renderer_identity, &batch.renderer_identity)
            || !prepared_scene_belongs_to(&image.resource_identity, &batch.image_identity)
        {
            return Err(ImageError::RendererMismatch);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_image_batch_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    image: &Image2d,
    sprites: Vec<ImageSprite2d>,
    budget: ImageBatchBudget,
) -> Result<ImageBatch2d, ImageError> {
    let retained_bytes = sprites
        .len()
        .checked_mul(std::mem::size_of::<ImageSprite2d>())
        .ok_or(ImageError::DimensionsTooLarge)?;
    if sprites.len() > budget.max_sprites {
        return Err(ImageError::BatchBudgetExceeded {
            limit: budget.max_sprites,
            actual: sprites.len(),
        });
    }
    if retained_bytes > budget.max_retained_bytes {
        return Err(ImageError::BatchBudgetExceeded {
            limit: budget.max_retained_bytes,
            actual: retained_bytes,
        });
    }
    let gpu_bytes = sprites
        .len()
        .max(1)
        .checked_mul(std::mem::size_of::<ImageInstance>())
        .ok_or(ImageError::DimensionsTooLarge)?;
    if gpu_bytes as u64 > device.limits().max_buffer_size {
        return Err(ImageError::DimensionsTooLarge);
    }
    let mut instances = Vec::new();
    instances
        .try_reserve_exact(sprites.len())
        .map_err(|_| ImageError::AllocationFailed {
            requested_bytes: sprites
                .len()
                .saturating_mul(std::mem::size_of::<ImageInstance>()),
        })?;
    for sprite in &sprites {
        if !sprite.source.fits(image.width, image.height) || !sprite.tint.is_normalized() {
            return Err(ImageError::InvalidSprite);
        }
        let origin = sprite.destination.origin().to_vec2();
        let viewport = sprite.destination.viewport();
        let image_width = image.width as f32;
        let image_height = image.height as f32;
        let source = sprite.source;
        let instance = ImageInstance {
            destination: [origin.x, origin.y, viewport.width(), viewport.height()],
            uv_rect: [
                (source.x as f32 + 0.5) / image_width,
                (source.y as f32 + 0.5) / image_height,
                (source.x as f32 + source.width as f32 - 0.5) / image_width,
                (source.y as f32 + source.height as f32 - 0.5) / image_height,
            ],
            tint: sprite.tint.to_array(),
        };
        if !instance
            .destination
            .iter()
            .chain(instance.uv_rect.iter())
            .chain(instance.tint.iter())
            .all(|value| value.is_finite())
        {
            return Err(ImageError::InvalidSprite);
        }
        instances.push(instance);
    }
    let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine retained image sprite instances"),
        size: gpu_bytes as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !instances.is_empty() {
        queue.write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&instances));
    }
    Ok(ImageBatch2d {
        renderer_identity,
        image_identity: Arc::clone(&image.resource_identity),
        instance_buffer,
        sprites,
        budget,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_image_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    budget: ImageBudget,
) -> Result<Image2d, ImageError> {
    validate_image_shape(
        width,
        height,
        pixels.len(),
        budget,
        device.limits().max_texture_dimension_2d,
        usize::try_from(device.limits().max_buffer_size).unwrap_or(usize::MAX),
    )?;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine retained RGBA image"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: IMAGE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_image_texture(queue, &texture, 0, 0, width, height, &pixels);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok(Image2d {
        renderer_identity,
        resource_identity: Arc::new(()),
        texture,
        view,
        width,
        height,
        pixels,
        budget,
    })
}

fn validate_image_shape(
    width: u32,
    height: u32,
    pixel_count: usize,
    budget: ImageBudget,
    max_texture_dimension: u32,
    max_allocation_bytes: usize,
) -> Result<usize, ImageError> {
    if width == 0 || height == 0 {
        return Err(ImageError::ZeroDimension);
    }
    if width > budget.max_width
        || height > budget.max_height
        || width > max_texture_dimension
        || height > max_texture_dimension
        || width.checked_mul(4).is_none()
    {
        return Err(ImageError::DimensionsTooLarge);
    }
    let bytes = image_byte_count(width, height).ok_or(ImageError::DimensionsTooLarge)?;
    if bytes > budget.max_bytes {
        return Err(ImageError::BudgetExceeded {
            limit: budget.max_bytes,
            actual: bytes,
        });
    }
    if bytes > max_allocation_bytes {
        return Err(ImageError::DimensionsTooLarge);
    }
    if pixel_count != bytes {
        return Err(ImageError::InvalidPixelCount);
    }
    Ok(bytes)
}

fn image_byte_count(width: u32, height: u32) -> Option<usize> {
    (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)
}

#[allow(clippy::too_many_arguments)]
fn write_image_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    pixels: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_shape_checks_budget_device_and_exact_pixels_before_allocation() {
        let budget = ImageBudget::new(8, 8, 8 * 8 * 4).unwrap();
        assert_eq!(validate_image_shape(8, 8, 256, budget, 8, 256), Ok(256));
        assert_eq!(
            validate_image_shape(8, 8, 255, budget, 8, 256),
            Err(ImageError::InvalidPixelCount)
        );
        assert_eq!(
            validate_image_shape(9, 1, 36, budget, 16, 256),
            Err(ImageError::DimensionsTooLarge)
        );
        let strict = ImageBudget::new(8, 8, 64).unwrap();
        assert_eq!(
            validate_image_shape(8, 8, 256, strict, 8, 256),
            Err(ImageError::BudgetExceeded {
                limit: 64,
                actual: 256,
            })
        );
    }

    #[test]
    fn atlas_rect_rejects_empty_and_overflowing_ranges() {
        assert_eq!(
            ImageTexelRect::new(0, 0, 0, 1),
            Err(ImageError::ZeroDimension)
        );
        assert_eq!(
            ImageTexelRect::new(u32::MAX, 0, 2, 1),
            Err(ImageError::DimensionsTooLarge)
        );
        assert!(ImageTexelRect::new(1, 2, 3, 4).unwrap().fits(4, 6));
    }

    #[test]
    fn sprite_rejects_finite_inputs_whose_logical_extent_overflows() {
        let source = ImageTexelRect::new(0, 0, 1, 1).unwrap();
        let destination = LogicalViewportRegion::new(
            LogicalScreenPosition::new(f32::MAX, 0.0),
            LogicalViewport::new(f32::MAX, 1.0).unwrap(),
        )
        .unwrap();
        assert_eq!(
            ImageSprite2d::new(source, destination, Color::WHITE),
            Err(ImageError::InvalidSprite)
        );
    }
}
