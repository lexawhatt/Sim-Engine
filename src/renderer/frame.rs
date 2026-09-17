use super::*;

mod cache;
mod compose;
#[cfg(test)]
#[path = "frame/dedup_diagnostics.rs"]
mod dedup_diagnostics;
mod encoding;
#[cfg(test)]
pub(super) use encoding::assert_gpu_encoding_contract;
mod present;
mod streaming;
#[cfg(test)]
mod tests;
mod uniform_uploads;
pub(super) use cache::FrameCache;
#[cfg(test)]
pub(super) use cache::assert_gpu_binding_sharing_contract;
pub use cache::{FrameCacheBudget, FrameCacheStatistics};
#[cfg(test)]
pub(super) use uniform_uploads::assert_gpu_uniform_upload_contract;

/// Work category constrained by a [`FrameBudget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameBudgetResource {
    /// Number of ordered sources in one frame.
    Passes,
    /// Scene commands referenced by the frame.
    Commands,
    /// Generated or retained vertices referenced by the frame.
    Vertices,
    /// Bytes uploaded while preparing the frame.
    UploadBytes,
    /// Nominal retained texel-storage bytes referenced by images, fields, or targets.
    TextureBytes,
    /// Conservative number of GPU draw calls encoded by the frame.
    DrawCalls,
}

/// Explicit upper bounds for one heterogeneous presentation frame.
///
/// The default is deliberately finite. Production hosts should select limits
/// from their frame-time and memory budgets instead of treating it as a device
/// capability query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameBudget {
    max_passes: usize,
    max_commands: usize,
    max_vertices: usize,
    max_upload_bytes: usize,
    max_texture_bytes: usize,
    max_draw_calls: usize,
}

impl FrameBudget {
    /// Creates an exact set of frame-work limits.
    pub const fn new(
        max_passes: usize,
        max_commands: usize,
        max_vertices: usize,
        max_upload_bytes: usize,
        max_texture_bytes: usize,
        max_draw_calls: usize,
    ) -> Self {
        Self {
            max_passes,
            max_commands,
            max_vertices,
            max_upload_bytes,
            max_texture_bytes,
            max_draw_calls,
        }
    }

    /// Returns the maximum number of ordered sources.
    pub const fn max_passes(self) -> usize {
        self.max_passes
    }

    /// Returns the maximum referenced scene-command count.
    pub const fn max_commands(self) -> usize {
        self.max_commands
    }

    /// Returns the maximum referenced or generated vertex count.
    pub const fn max_vertices(self) -> usize {
        self.max_vertices
    }

    /// Returns the maximum bytes uploaded while presenting the frame.
    pub const fn max_upload_bytes(self) -> usize {
        self.max_upload_bytes
    }

    /// Returns the maximum referenced nominal texel-storage bytes.
    pub const fn max_texture_bytes(self) -> usize {
        self.max_texture_bytes
    }

    /// Returns the maximum conservative draw-call count.
    pub const fn max_draw_calls(self) -> usize {
        self.max_draw_calls
    }
}

impl Default for FrameBudget {
    fn default() -> Self {
        Self::new(
            64,
            100_000,
            4_000_000,
            256 * 1024 * 1024,
            512 * 1024 * 1024,
            65_536,
        )
    }
}

/// Shared placement and ordering state for one frame item.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FramePassOptions {
    order: i32,
    viewport: Option<LogicalViewportRegion>,
    clip: Option<ScreenClipRect>,
}

impl FramePassOptions {
    /// Creates a full-surface, unclipped item at `order`.
    pub const fn new(order: i32) -> Self {
        Self {
            order,
            viewport: None,
            clip: None,
        }
    }

    /// Positions the item in a logical sub-viewport of the surface.
    pub const fn with_viewport(mut self, viewport: LogicalViewportRegion) -> Self {
        self.viewport = Some(viewport);
        self
    }

    /// Intersects all source-local clips with an additional item-local clip.
    pub const fn with_clip(mut self, clip: ScreenClipRect) -> Self {
        self.clip = Some(clip);
        self
    }

    /// Returns the stable ordering key. Lower values draw first.
    pub const fn order(self) -> i32 {
        self.order
    }

    /// Returns the optional positioned logical viewport.
    pub const fn viewport(self) -> Option<LogicalViewportRegion> {
        self.viewport
    }

    /// Returns the optional item-local logical clip.
    pub const fn clip(self) -> Option<ScreenClipRect> {
        self.clip
    }
}

/// Renderer-owned source category used in structured composer errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSourceKind {
    /// Streaming world-space or fixed-screen scene tessellated for this frame.
    StreamingScene,
    /// Immutable prepared world or screen geometry.
    PreparedScene,
    /// Mutable retained triangle geometry.
    DynamicMesh,
    /// Instanced retained particle resource.
    ParticleField,
    /// Renderer-owned scalar texture.
    ScalarField,
    /// Retained sRGB RGBA image or atlas.
    Image,
    /// Retained host-shaped glyph atlas and positioned run.
    Glyph,
    /// Offscreen color texture.
    RenderTarget,
}

/// Ordered frame inputs grouped by their renderer source path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameSourceStatistics {
    streaming_scenes: usize,
    prepared_scenes: usize,
    dynamic_meshes: usize,
    particle_fields: usize,
    scalar_fields: usize,
    images: usize,
    glyph_runs: usize,
    render_targets: usize,
}

impl FrameSourceStatistics {
    /// Returns streaming world and fixed-screen scenes.
    pub const fn streaming_scenes(self) -> usize {
        self.streaming_scenes
    }

    /// Returns immutable prepared world and fixed-screen scenes.
    pub const fn prepared_scenes(self) -> usize {
        self.prepared_scenes
    }

    /// Returns retained dynamic triangle meshes.
    pub const fn dynamic_meshes(self) -> usize {
        self.dynamic_meshes
    }

    /// Returns retained particle fields.
    pub const fn particle_fields(self) -> usize {
        self.particle_fields
    }

    /// Returns retained scalar fields.
    pub const fn scalar_fields(self) -> usize {
        self.scalar_fields
    }

    /// Returns image, world-image, and atlas-batch sources.
    pub const fn images(self) -> usize {
        self.images
    }

    /// Returns host-shaped glyph runs.
    pub const fn glyph_runs(self) -> usize {
        self.glyph_runs
    }

    /// Returns composed 2D or retained-3D color targets.
    pub const fn render_targets(self) -> usize {
        self.render_targets
    }

    /// Returns all accepted frame sources.
    pub const fn total(self) -> usize {
        self.streaming_scenes
            .saturating_add(self.prepared_scenes)
            .saturating_add(self.dynamic_meshes)
            .saturating_add(self.particle_fields)
            .saturating_add(self.scalar_fields)
            .saturating_add(self.images)
            .saturating_add(self.glyph_runs)
            .saturating_add(self.render_targets)
    }

    const fn single(source: FrameSourceKind) -> Self {
        let mut counts = Self {
            streaming_scenes: 0,
            prepared_scenes: 0,
            dynamic_meshes: 0,
            particle_fields: 0,
            scalar_fields: 0,
            images: 0,
            glyph_runs: 0,
            render_targets: 0,
        };
        match source {
            FrameSourceKind::StreamingScene => counts.streaming_scenes = 1,
            FrameSourceKind::PreparedScene => counts.prepared_scenes = 1,
            FrameSourceKind::DynamicMesh => counts.dynamic_meshes = 1,
            FrameSourceKind::ParticleField => counts.particle_fields = 1,
            FrameSourceKind::ScalarField => counts.scalar_fields = 1,
            FrameSourceKind::Image => counts.images = 1,
            FrameSourceKind::Glyph => counts.glyph_runs = 1,
            FrameSourceKind::RenderTarget => counts.render_targets = 1,
        }
        counts
    }

    const fn adding(self, other: Self) -> Self {
        Self {
            streaming_scenes: self.streaming_scenes.saturating_add(other.streaming_scenes),
            prepared_scenes: self.prepared_scenes.saturating_add(other.prepared_scenes),
            dynamic_meshes: self.dynamic_meshes.saturating_add(other.dynamic_meshes),
            particle_fields: self.particle_fields.saturating_add(other.particle_fields),
            scalar_fields: self.scalar_fields.saturating_add(other.scalar_fields),
            images: self.images.saturating_add(other.images),
            glyph_runs: self.glyph_runs.saturating_add(other.glyph_runs),
            render_targets: self.render_targets.saturating_add(other.render_targets),
        }
    }
}

/// Failure while constructing or presenting a heterogeneous frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameComposerError {
    /// The one frame clear color was not normalized linear RGBA.
    InvalidBackground,
    /// A target-composition opacity was non-finite or outside `0.0..=1.0`.
    InvalidOpacity,
    /// An image tint was not normalized linear RGBA.
    InvalidTint,
    /// An atlas source rectangle did not fit its image.
    InvalidImageRegion,
    /// A world image rectangle or pseudo-depth was invalid.
    InvalidWorldImage,
    /// Scalar endpoints were invalid or their finite subtraction overflowed.
    InvalidValueRange {
        /// Lower scalar endpoint supplied by the host.
        minimum: f32,
        /// Upper scalar endpoint supplied by the host.
        maximum: f32,
    },
    /// A retained source belongs to another renderer or recovery generation.
    RendererMismatch {
        /// Category of the rejected retained source.
        source: FrameSourceKind,
    },
    /// A frame-work limit would be exceeded.
    BudgetExceeded {
        /// Work category whose limit was exceeded.
        resource: FrameBudgetResource,
        /// Configured upper bound.
        limit: usize,
        /// Conservative work after accepting the requested item.
        actual: usize,
    },
    /// CPU storage for frame construction could not be reserved.
    AllocationFailed {
        /// Minimum additional bytes requested by the failed reservation.
        requested_bytes: usize,
    },
    /// Validation, surface acquisition, or GPU-capacity work failed.
    Frame(RendererFrameError),
}

impl fmt::Display for FrameComposerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBackground => write!(formatter, "frame background must be normalized"),
            Self::InvalidOpacity => write!(formatter, "frame opacity must be finite and in 0..=1"),
            Self::InvalidTint => write!(formatter, "frame image tint must be normalized"),
            Self::InvalidImageRegion => write!(formatter, "frame image region is out of bounds"),
            Self::InvalidWorldImage => {
                write!(formatter, "frame world image rectangle or depth is invalid")
            }
            Self::InvalidValueRange { minimum, maximum } => {
                write!(formatter, "invalid frame scalar range {minimum}..{maximum}")
            }
            Self::RendererMismatch { source } => {
                write!(formatter, "frame {source:?} belongs to another renderer")
            }
            Self::BudgetExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "frame {resource:?} work exceeds its limit {limit}: {actual}"
            ),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} additional bytes for frame construction"
            ),
            Self::Frame(error) => write!(formatter, "frame presentation failed: {error}"),
        }
    }
}

impl Error for FrameComposerError {}

impl From<RendererFrameError> for FrameComposerError {
    fn from(error: RendererFrameError) -> Self {
        Self::Frame(error)
    }
}

/// Bounded work and retained allocations referenced by one composed frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameStatistics {
    pass_count: usize,
    command_count: usize,
    vertex_count: usize,
    streaming_vertex_count: usize,
    reused_vertex_count: usize,
    upload_bytes: usize,
    streaming_upload_bytes: usize,
    texture_bytes: usize,
    retained_cpu_bytes: usize,
    retained_buffer_bytes: usize,
    draw_calls: usize,
    source_counts: FrameSourceStatistics,
}

impl FrameStatistics {
    /// Returns ordered source count.
    pub const fn pass_count(self) -> usize {
        self.pass_count
    }

    /// Returns referenced scene-command count.
    pub const fn command_count(self) -> usize {
        self.command_count
    }

    /// Returns referenced or generated vertex count.
    pub const fn vertex_count(self) -> usize {
        self.vertex_count
    }

    /// Returns streaming vertices selected by draws during this frame.
    /// Repeated references may share one within-frame upload; use
    /// [`Self::streaming_upload_bytes`] for the actual transferred payload.
    pub const fn streaming_vertex_count(self) -> usize {
        self.streaming_vertex_count
    }

    /// Returns retained prepared or dynamic vertices reused without frame upload.
    pub const fn reused_vertex_count(self) -> usize {
        self.reused_vertex_count
    }

    /// Returns bytes uploaded for streaming vertices and per-item uniforms.
    pub const fn upload_bytes(self) -> usize {
        self.upload_bytes
    }

    /// Returns the subset uploaded for streaming geometry or instances.
    pub const fn streaming_upload_bytes(self) -> usize {
        self.streaming_upload_bytes
    }

    /// Returns nominal retained texel-storage bytes referenced by frame items.
    ///
    /// Repeated image, scalar-field, and render-target identities are
    /// deduplicated. Opaque backend alignment, tiling, page allocation, and
    /// metadata are excluded because wgpu does not expose them. Color-map LUT
    /// bytes follow the renderer's one-entry cache:
    /// adjacent equal LUTs share one allocation, while an `A, B, A` sequence
    /// keeps three views/allocations alive until submission.
    pub const fn texture_bytes(self) -> usize {
        self.texture_bytes
    }

    /// Returns CPU recovery or source bytes held by unique referenced resources.
    ///
    /// Drawing the same scene, buffer, image, atlas, or run more than once does
    /// not multiply this value. Distinct retained resources are summed even when
    /// their contents happen to be equal.
    pub const fn retained_cpu_bytes(self) -> usize {
        self.retained_cpu_bytes
    }

    /// Returns GPU vertex or instance-buffer bytes held by unique referenced resources.
    pub const fn retained_buffer_bytes(self) -> usize {
        self.retained_buffer_bytes
    }

    /// Returns the conservative number of scheduled draw calls.
    pub const fn draw_calls(self) -> usize {
        self.draw_calls
    }

    /// Returns accepted ordered inputs grouped by renderer source path.
    pub const fn source_counts(self) -> FrameSourceStatistics {
        self.source_counts
    }

    fn adding(self, other: Self) -> Self {
        Self {
            pass_count: self.pass_count.saturating_add(other.pass_count),
            command_count: self.command_count.saturating_add(other.command_count),
            vertex_count: self.vertex_count.saturating_add(other.vertex_count),
            streaming_vertex_count: self
                .streaming_vertex_count
                .saturating_add(other.streaming_vertex_count),
            reused_vertex_count: self
                .reused_vertex_count
                .saturating_add(other.reused_vertex_count),
            upload_bytes: self.upload_bytes.saturating_add(other.upload_bytes),
            streaming_upload_bytes: self
                .streaming_upload_bytes
                .saturating_add(other.streaming_upload_bytes),
            texture_bytes: self.texture_bytes.saturating_add(other.texture_bytes),
            retained_cpu_bytes: self
                .retained_cpu_bytes
                .saturating_add(other.retained_cpu_bytes),
            retained_buffer_bytes: self
                .retained_buffer_bytes
                .saturating_add(other.retained_buffer_bytes),
            draw_calls: self.draw_calls.saturating_add(other.draw_calls),
            source_counts: self.source_counts.adding(other.source_counts),
        }
    }

    fn without_uploads(mut self) -> Self {
        self.upload_bytes = 0;
        self.streaming_upload_bytes = 0;
        self
    }
}

/// Outcome and diagnostics for one composed frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameReport {
    status: RenderStatus,
    statistics: FrameStatistics,
    metrics: RendererFrameMetrics,
    gpu_timing_id: Option<GpuTimingId>,
}

impl FrameReport {
    /// Returns whether the frame was drawn or temporarily skipped.
    pub const fn status(self) -> RenderStatus {
        self.status
    }

    /// Returns bounded work referenced by the frame.
    pub const fn statistics(self) -> FrameStatistics {
        self.statistics
    }

    /// Returns CPU-side timing and tessellation diagnostics.
    pub const fn metrics(self) -> RendererFrameMetrics {
        self.metrics
    }

    /// Correlation key for this submitted pass's optional asynchronous GPU time.
    /// Skipped frames never reserve a key. `None` also covers disabled/unavailable
    /// timing or a full diagnostics ring; inspect renderer timing statistics.
    pub const fn gpu_timing_id(self) -> Option<GpuTimingId> {
        self.gpu_timing_id
    }

    /// Returns command encoders submitted by this composed frame.
    pub const fn command_encoder_count(self) -> usize {
        match self.status {
            RenderStatus::Drawn => 1,
            RenderStatus::Skipped(_) => 0,
        }
    }

    /// Returns GPU render passes encoded by this composed frame.
    pub const fn render_pass_count(self) -> usize {
        match self.status {
            RenderStatus::Drawn => 1,
            RenderStatus::Skipped(_) => 0,
        }
    }

    /// Returns queue submissions performed by this composed frame.
    pub const fn queue_submission_count(self) -> usize {
        match self.status {
            RenderStatus::Drawn => 1,
            RenderStatus::Skipped(_) => 0,
        }
    }

    /// Returns surface presents performed by this composed frame.
    pub const fn surface_present_count(self) -> usize {
        match self.status {
            RenderStatus::Drawn => 1,
            RenderStatus::Skipped(_) => 0,
        }
    }
}

/// One bounded, stably ordered surface frame under construction.
///
/// Calling [`FrameComposer::present`] performs all fallible geometry and
/// ownership validation before acquiring the surface. A successful frame uses
/// one clear, one command encoder, one queue submission, and one present.
#[must_use = "a frame composer does no rendering until present is called"]
pub struct FrameComposer<'frame> {
    renderer: &'frame mut WgpuRenderer,
    background: Color,
    budget: FrameBudget,
    items: Vec<FrameItem<'frame>>,
    retained_resources: Vec<RetainedResourceKey>,
    scalar_luts: Vec<ScalarLutPlan>,
    scalar_lut_upload_count: usize,
    scalar_lut_allocation_count: usize,
    planned: FrameStatistics,
    next_insertion: usize,
    cache: FrameCache,
}

impl Drop for FrameComposer<'_> {
    fn drop(&mut self) {
        self.cache.items = cache::recycle_items(std::mem::take(&mut self.items));
        self.retained_resources.clear();
        self.scalar_luts.clear();
        self.cache.retained_resources = std::mem::take(&mut self.retained_resources);
        self.cache.scalar_luts = std::mem::take(&mut self.scalar_luts);
        self.cache.finish();
        self.renderer.frame_cache = std::mem::take(&mut self.cache);
    }
}

struct ScalarLutPlan {
    sort_key: (i32, usize),
    lut: [u8; COLOR_MAP_LUT_SIZE as usize * 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RetainedResourceKey {
    address: usize,
    class: u8,
}

#[derive(Debug, Clone, Copy)]
struct RetainedResourceAccounting {
    key: RetainedResourceKey,
    cpu_bytes: usize,
    buffer_bytes: usize,
    texture_bytes: usize,
}

impl RetainedResourceAccounting {
    const fn new(
        key: RetainedResourceKey,
        cpu_bytes: usize,
        buffer_bytes: usize,
        texture_bytes: usize,
    ) -> Self {
        Self {
            key,
            cpu_bytes,
            buffer_bytes,
            texture_bytes,
        }
    }
}

fn retained_key<T>(resource: &T, class: u8) -> RetainedResourceKey {
    RetainedResourceKey {
        address: std::ptr::from_ref(resource).cast::<()>() as usize,
        class,
    }
}

fn retained_arc_key<T>(resource: &Arc<T>, class: u8) -> RetainedResourceKey {
    RetainedResourceKey {
        address: Arc::as_ptr(resource).cast::<()>() as usize,
        class,
    }
}

fn account_new_retained_resources(
    existing: &[RetainedResourceKey],
    resources: &[RetainedResourceAccounting],
    mut work: FrameStatistics,
) -> (FrameStatistics, usize) {
    let mut missing = 0usize;
    for (index, resource) in resources.iter().enumerate() {
        let already_in_request = resources[..index]
            .iter()
            .any(|previous| previous.key == resource.key);
        if !already_in_request && !existing.contains(&resource.key) {
            missing = missing.saturating_add(1);
            work.retained_cpu_bytes = work.retained_cpu_bytes.saturating_add(resource.cpu_bytes);
            work.retained_buffer_bytes = work
                .retained_buffer_bytes
                .saturating_add(resource.buffer_bytes);
            work.texture_bytes = work.texture_bytes.saturating_add(resource.texture_bytes);
        }
    }
    (work, missing)
}

fn scalar_lut_counts_with_inserted(
    planned: &[ScalarLutPlan],
    insertion: usize,
    inserted: &[u8; COLOR_MAP_LUT_SIZE as usize * 4],
    initial_cache: Option<&[u8; COLOR_MAP_LUT_SIZE as usize * 4]>,
) -> (usize, usize) {
    debug_assert!(insertion <= planned.len());
    let mut previous: Option<&[u8; COLOR_MAP_LUT_SIZE as usize * 4]> = None;
    let mut allocation_count = 0usize;
    let mut first_matches_initial = false;
    for merged_index in 0..=planned.len() {
        let current = if merged_index == insertion {
            inserted
        } else {
            let source_index = merged_index - usize::from(merged_index > insertion);
            &planned[source_index].lut
        };
        if previous.is_none() {
            first_matches_initial = initial_cache == Some(current);
        }
        if previous != Some(current) {
            allocation_count = allocation_count.saturating_add(1);
        }
        previous = Some(current);
    }
    let upload_count = allocation_count.saturating_sub(usize::from(first_matches_initial));
    (upload_count, allocation_count)
}

enum FrameItem<'frame> {
    Scene {
        scene: &'frame Scene,
        camera: Camera2d,
        options: FramePassOptions,
        insertion: usize,
    },
    ScreenScene {
        scene: &'frame ScreenScene,
        options: FramePassOptions,
        insertion: usize,
    },
    Prepared {
        scene: &'frame PreparedScene,
        camera: Camera2d,
        options: FramePassOptions,
        insertion: usize,
    },
    PreparedScreen {
        scene: &'frame PreparedScreenScene,
        options: FramePassOptions,
        insertion: usize,
    },
    Dynamic {
        mesh: &'frame DynamicMesh2d,
        camera: Camera2d,
        options: FramePassOptions,
        insertion: usize,
    },
    Particle {
        field: &'frame mut ParticleField2d,
        camera: Camera2d,
        options: FramePassOptions,
        insertion: usize,
    },
    Scalar {
        texture: &'frame ScalarFieldTexture,
        color_map: &'frame ColorMap,
        minimum: f32,
        maximum: f32,
        value_extent: f32,
        sampling: ScalarFieldSampling,
        options: FramePassOptions,
        insertion: usize,
    },
    Image {
        image: &'frame Image2d,
        source: ImageTexelRect,
        tint: Color,
        sampling: ImageSampling,
        options: FramePassOptions,
        insertion: usize,
    },
    WorldImage {
        image: &'frame Image2d,
        source: ImageTexelRect,
        rectangle: Rect,
        depth: f32,
        camera: Camera2d,
        tint: Color,
        sampling: ImageSampling,
        options: FramePassOptions,
        insertion: usize,
    },
    ImageBatch {
        image: &'frame Image2d,
        batch: &'frame ImageBatch2d,
        placement: ImageBatchPlacement,
        sampling: ImageSampling,
        options: FramePassOptions,
        insertion: usize,
    },
    Target {
        target: &'frame RenderTarget2d,
        blend_mode: BlendMode,
        opacity: f32,
        options: FramePassOptions,
        insertion: usize,
    },
}

impl FrameItem<'_> {
    fn sort_key(&self) -> (i32, usize) {
        match self {
            Self::Scene {
                options, insertion, ..
            }
            | Self::ScreenScene {
                options, insertion, ..
            }
            | Self::Prepared {
                options, insertion, ..
            }
            | Self::PreparedScreen {
                options, insertion, ..
            }
            | Self::Dynamic {
                options, insertion, ..
            }
            | Self::Particle {
                options, insertion, ..
            }
            | Self::Scalar {
                options, insertion, ..
            }
            | Self::Image {
                options, insertion, ..
            }
            | Self::WorldImage {
                options, insertion, ..
            }
            | Self::ImageBatch {
                options, insertion, ..
            }
            | Self::Target {
                options, insertion, ..
            } => (options.order(), *insertion),
        }
    }
}

impl WgpuRenderer {
    /// Begins one bounded heterogeneous frame with a single clear color.
    pub fn begin_frame(
        &mut self,
        background: Color,
        budget: FrameBudget,
    ) -> Result<FrameComposer<'_>, FrameComposerError> {
        if !background.is_normalized() {
            return Err(FrameComposerError::InvalidBackground);
        }
        let mut cache = std::mem::take(&mut self.frame_cache);
        cache.begin();
        let items = std::mem::take(&mut cache.items);
        let retained_resources = std::mem::take(&mut cache.retained_resources);
        let scalar_luts = std::mem::take(&mut cache.scalar_luts);
        Ok(FrameComposer {
            renderer: self,
            background,
            budget,
            items,
            retained_resources,
            scalar_luts,
            scalar_lut_upload_count: 0,
            scalar_lut_allocation_count: 0,
            planned: FrameStatistics::default(),
            next_insertion: 0,
            cache,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedViewport {
    viewport: LogicalViewport,
    origin: Vec2,
    scissor: ScissorRect,
    item_clip: Option<ScissorRect>,
    item_clipped_out: bool,
}

enum ReadySource<'frame> {
    Streaming,
    Prepared(&'frame wgpu::Buffer),
    Dynamic(&'frame wgpu::Buffer),
}

struct ReadyGeometry<'frame> {
    source: ReadySource<'frame>,
    vertex_count: usize,
    batches: Vec<PreparedDrawBatch>,
    camera_uniform: CameraUniform,
    viewport: ResolvedViewport,
}

enum ReadyItem<'frame> {
    Geometry(ReadyGeometry<'frame>),
    Particle {
        field: &'frame mut ParticleField2d,
        visible_count: usize,
        pending_statistics: ParticleStatistics,
        initial_cpu_allocation_bytes: usize,
        camera_uniform: CameraUniform,
        viewport: ResolvedViewport,
    },
    Scalar {
        texture: &'frame ScalarFieldTexture,
        color_map: &'frame ColorMap,
        uniform: HeatmapUniform,
        viewport: ResolvedViewport,
    },
    Image {
        image: &'frame Image2d,
        sampling: ImageSampling,
        uniform: ImageUniform,
        viewport: ResolvedViewport,
    },
    ImageBatch {
        image: &'frame Image2d,
        batch: &'frame ImageBatch2d,
        sampling: ImageSampling,
        uniform: ImageUniform,
        viewport: ResolvedViewport,
    },
    Target {
        target: &'frame RenderTarget2d,
        blend_mode: BlendMode,
        uniform: CompositeUniform,
        viewport: ResolvedViewport,
    },
}

#[derive(Clone)]
struct FrameBinding {
    _buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

fn preflight_frame_items(
    renderer: &WgpuRenderer,
    target_viewport: LogicalViewport,
    items: &[FrameItem<'_>],
) -> Result<(), FrameComposerError> {
    for item in items {
        match item {
            FrameItem::Scene {
                camera, options, ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                CameraUniform::new_in_region(
                    *camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
            }
            FrameItem::ScreenScene { options, .. } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                let camera = screen_camera(viewport.viewport)
                    .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
                CameraUniform::new_in_region(
                    camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
            }
            FrameItem::Prepared {
                scene,
                camera,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                let uniform = CameraUniform::new_in_region(
                    *camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                if !geometry_is_safe_for_cached(
                    Some(&scene.geometry_validation_cache),
                    scene.geometry_extents,
                    GeometryValidationSource::Tessellated(&scene.vertices),
                    uniform,
                ) {
                    return Err(RendererFrameError::InvalidGeometryTransform.into());
                }
            }
            FrameItem::PreparedScreen { scene, options, .. } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                let camera = screen_camera(viewport.viewport)
                    .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
                let uniform = CameraUniform::new_in_region(
                    camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                if !geometry_is_safe_for_cached(
                    Some(&scene.scene.geometry_validation_cache),
                    scene.scene.geometry_extents,
                    GeometryValidationSource::Tessellated(&scene.scene.vertices),
                    uniform,
                ) {
                    return Err(RendererFrameError::InvalidGeometryTransform.into());
                }
            }
            FrameItem::Dynamic {
                mesh,
                camera,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                let uniform = CameraUniform::new_in_region(
                    *camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                if !geometry_is_safe_for_cached(
                    Some(&mesh.geometry_validation_cache),
                    mesh.geometry_extents,
                    GeometryValidationSource::Dynamic(&mesh.vertices),
                    uniform,
                ) {
                    return Err(RendererFrameError::InvalidGeometryTransform.into());
                }
            }
            FrameItem::Particle {
                camera, options, ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                CameraUniform::new(*camera, viewport.viewport)
                    .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                CameraUniform::new_in_region(
                    *camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
            }
            FrameItem::Scalar {
                texture,
                minimum,
                maximum,
                value_extent,
                sampling,
                options,
                ..
            } => {
                if !scalar_normalization_is_portable(texture, *minimum, *value_extent) {
                    return Err(FrameComposerError::InvalidValueRange {
                        minimum: *minimum,
                        maximum: *maximum,
                    });
                }
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                let region = LogicalViewportRegion::new(
                    LogicalScreenPosition::from_vec2(viewport.origin),
                    viewport.viewport,
                )
                .map_err(|_| RendererFrameError::InvalidViewport)?;
                HeatmapUniform::new(
                    *minimum,
                    *value_extent,
                    texture.width(),
                    texture.height(),
                    *sampling,
                )
                .in_region(region, target_viewport)
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
            }
            FrameItem::Image { options, .. } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                let region = LogicalViewportRegion::new(
                    LogicalScreenPosition::from_vec2(viewport.origin),
                    viewport.viewport,
                )
                .map_err(|_| RendererFrameError::InvalidViewport)?;
                CompositeUniform::in_region(1.0, region, target_viewport)
                    .ok_or(RendererFrameError::InvalidGeometryTransform)?;
            }
            FrameItem::Target {
                opacity, options, ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                let region = LogicalViewportRegion::new(
                    LogicalScreenPosition::from_vec2(viewport.origin),
                    viewport.viewport,
                )
                .map_err(|_| RendererFrameError::InvalidViewport)?;
                CompositeUniform::in_region(*opacity, region, target_viewport)
                    .ok_or(RendererFrameError::InvalidGeometryTransform)?;
            }
            FrameItem::WorldImage {
                rectangle,
                depth,
                camera,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                let camera_uniform = CameraUniform::new_in_region(
                    *camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                let minimum = rectangle.min();
                let maximum = rectangle.max();
                for world in [
                    Vec2::new(minimum.x, maximum.y),
                    maximum,
                    minimum,
                    Vec2::new(maximum.x, minimum.y),
                ] {
                    let screen = camera_uniform.world_to_screen(world, *depth);
                    let clip = Vec2::new(
                        screen.x.mul_add(
                            camera_uniform.screen_to_clip[0],
                            camera_uniform.screen_to_clip[2],
                        ),
                        screen.y.mul_add(
                            camera_uniform.screen_to_clip[1],
                            camera_uniform.screen_to_clip[3],
                        ),
                    );
                    if ![clip.x, clip.y].into_iter().all(is_portable_shader_source) {
                        return Err(RendererFrameError::InvalidGeometryTransform.into());
                    }
                }
            }
            FrameItem::ImageBatch {
                batch,
                options,
                placement,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                let uniform = image::batch_uniform(target_viewport, viewport.origin, *placement)?;
                if !batch.sprites_are_safe_for_target(
                    Vec2::new(uniform.uv_rect[0], uniform.uv_rect[1]),
                    uniform.destination,
                ) {
                    return Err(RendererFrameError::InvalidGeometryTransform.into());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn image_sprites_are_safe_for_target(
    sprites: &[ImageSprite2d],
    viewport_origin: Vec2,
    clip_transform: [f32; 4],
) -> bool {
    image::sprites_are_safe_for_target(sprites, viewport_origin, clip_transform)
}

fn set_particle_rendered(items: &mut [ReadyItem<'_>], presented: bool) {
    for item in items {
        if let ReadyItem::Particle {
            field,
            visible_count,
            pending_statistics,
            viewport,
            ..
        } = item
        {
            field.statistics = ParticleStatistics {
                rendered: if presented && !viewport.item_clipped_out {
                    *visible_count
                } else {
                    0
                },
                ..*pending_statistics
            };
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_particle_item<'frame>(
    field: &'frame mut ParticleField2d,
    camera: Camera2d,
    viewport: ResolvedViewport,
    target_viewport: LogicalViewport,
    ready: &mut Vec<ReadyItem<'frame>>,
    statistics: &mut FrameStatistics,
) -> Result<(), FrameComposerError> {
    let initial_cpu_allocation_bytes = field.cpu_allocation_bytes();
    let local_uniform = CameraUniform::new(camera, viewport.viewport)
        .ok_or(RendererFrameError::InvalidGeometryTransform)?;
    let target_uniform =
        CameraUniform::new_in_region(camera, viewport.viewport, viewport.origin, target_viewport)
            .ok_or(RendererFrameError::InvalidGeometryTransform)?;
    let instance_count = field.instances.len();
    let visibility_checked = instance_count.min(field.budget.max_visibility_checks_per_frame);

    let (visible_count, selected_count) = if visibility_checked < instance_count {
        field.visible_instances.clear();
        validate_particle_staging_capacity(&field.visible_instances, visibility_checked)?;
        for candidate_index in 0..visibility_checked {
            let source_index =
                uniformly_sampled_index(candidate_index, instance_count, visibility_checked);
            let instance = field.instances[source_index];
            if !instance.is_safe_for(target_uniform, target_viewport) {
                return Err(RendererFrameError::InvalidGeometryTransform.into());
            }
            let Some(intersects) =
                instance.validated_viewport_intersection(local_uniform, viewport.viewport)
            else {
                return Err(RendererFrameError::InvalidGeometryTransform.into());
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
        (visible_count, selected_count)
    } else {
        let visible_count = visible_particle_count_for_frame(
            &field.instances,
            local_uniform,
            target_uniform,
            viewport.viewport,
            target_viewport,
        )?;
        let selected_count = visible_count.min(field.budget.instance_limit());
        if selected_count == field.instances.len() {
            field.visible_instances.clear();
        } else {
            field.visible_instances.clear();
            validate_particle_staging_capacity(&field.visible_instances, selected_count)?;
            let mut visible_index = 0;
            for instance in field.instances.iter().copied() {
                if !instance.intersects_viewport(local_uniform, viewport.viewport) {
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
        (visible_count, selected_count)
    };
    let pending_statistics = particle_statistics_with_budget(
        instance_count,
        visibility_checked,
        visible_count,
        selected_count,
        0,
    );
    *statistics = statistics.adding(FrameStatistics {
        pass_count: 1,
        command_count: usize::from(instance_count > 0),
        vertex_count: selected_count.saturating_mul(6),
        streaming_vertex_count: selected_count.saturating_mul(6),
        reused_vertex_count: 0,
        upload_bytes: selected_count
            .saturating_mul(std::mem::size_of::<ParticleGpu>())
            .saturating_add(std::mem::size_of::<CameraUniform>()),
        streaming_upload_bytes: selected_count.saturating_mul(std::mem::size_of::<ParticleGpu>()),
        texture_bytes: 0,
        retained_cpu_bytes: 0,
        retained_buffer_bytes: 0,
        draw_calls: usize::from(selected_count > 0),
        source_counts: FrameSourceStatistics::default(),
    });
    ready.push(ReadyItem::Particle {
        field,
        visible_count: selected_count,
        pending_statistics,
        initial_cpu_allocation_bytes,
        camera_uniform: target_uniform,
        viewport,
    });
    Ok(())
}

fn visible_particle_count_for_frame(
    instances: &[ParticleGpu],
    local_camera: CameraUniform,
    target_camera: CameraUniform,
    viewport: LogicalViewport,
    target_viewport: LogicalViewport,
) -> Result<usize, RendererFrameError> {
    let mut visible = 0;
    for instance in instances {
        let Some(intersects) = instance.validated_viewport_intersection(local_camera, viewport)
        else {
            return Err(RendererFrameError::InvalidGeometryTransform);
        };
        if !instance.is_safe_for(target_camera, target_viewport) {
            return Err(RendererFrameError::InvalidGeometryTransform);
        }
        visible += usize::from(intersects);
    }
    Ok(visible)
}

// Keep grown scratch reachable on every structured failure, both for reuse
// and for the cache's conservative owned-capacity peak accounting.
fn with_streaming_batches(
    pool: &mut Vec<Vec<PreparedDrawBatch>>,
    prepare: impl FnOnce(&mut Vec<PreparedDrawBatch>) -> Result<(), FrameComposerError>,
) -> Result<(), FrameComposerError> {
    let mut batches = pool.pop().unwrap_or_default();
    let result = prepare(&mut batches);
    if batches.capacity() > 0 {
        batches.clear();
        pool.push(batches);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn prepare_streaming_scene<'frame>(
    scene: &'frame Scene,
    camera: Camera2d,
    options: FramePassOptions,
    renderer: &WgpuRenderer,
    target_viewport: LogicalViewport,
    streaming_vertices: &mut Vec<Vertex>,
    ready: &mut Vec<ReadyItem<'frame>>,
    statistics: &mut FrameStatistics,
    aggregate: &mut TessellationStats,
    batches: &mut Vec<PreparedDrawBatch>,
    proofs: &mut streaming::StreamingSceneProofs<'frame>,
) -> Result<(), FrameComposerError> {
    let viewport = resolve_viewport(renderer, target_viewport, options)?;
    let camera_uniform =
        CameraUniform::new_in_region(camera, viewport.viewport, viewport.origin, target_viewport)
            .ok_or(RendererFrameError::InvalidGeometryTransform)?;
    if !scene_estimate_fits_streaming_device(
        scene,
        &renderer.device,
        streaming_vertices.len(),
        renderer.vertex_capacity,
    ) {
        return Err(RendererFrameError::GeometryCapacityTooLarge.into());
    }
    proofs.prepare(
        scene,
        camera_uniform,
        viewport,
        streaming_vertices,
        ready,
        statistics,
        aggregate,
        batches,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_streaming_scene_resolved<'frame>(
    scene: &Scene,
    camera_uniform: CameraUniform,
    viewport: ResolvedViewport,
    streaming_vertices: &mut Vec<Vertex>,
    ready: &mut Vec<ReadyItem<'frame>>,
    statistics: &mut FrameStatistics,
    aggregate: &mut TessellationStats,
    batches: &mut Vec<PreparedDrawBatch>,
) -> Result<(), FrameComposerError> {
    let vertex_start = streaming_vertices.len();
    let stats =
        tessellate_scene(scene, streaming_vertices, batches).map_err(RendererFrameError::from)?;
    let vertices = &streaming_vertices[vertex_start..];
    let extents = GeometryExtents::from_vertices(vertices);
    if !geometry_is_safe_for(
        extents,
        GeometryValidationSource::Tessellated(vertices),
        camera_uniform,
    ) {
        streaming_vertices.truncate(vertex_start);
        return Err(RendererFrameError::InvalidGeometryTransform.into());
    }
    *statistics = statistics.adding(FrameStatistics {
        pass_count: 1,
        command_count: scene.command_count(),
        vertex_count: stats.vertex_count,
        streaming_vertex_count: stats.vertex_count,
        reused_vertex_count: 0,
        upload_bytes: stats
            .upload_bytes
            .saturating_add(std::mem::size_of::<CameraUniform>()),
        streaming_upload_bytes: stats.upload_bytes,
        texture_bytes: 0,
        retained_cpu_bytes: 0,
        retained_buffer_bytes: 0,
        draw_calls: batches.len(),
        source_counts: FrameSourceStatistics::default(),
    });
    add_tessellation_stats(aggregate, stats);
    ready.push(ReadyItem::Geometry(ReadyGeometry {
        source: ReadySource::Streaming,
        vertex_count: vertices.len(),
        batches: std::mem::take(batches),
        camera_uniform,
        viewport,
    }));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_retained_geometry<'frame>(
    source: ReadySource<'frame>,
    vertex_count: usize,
    extents: GeometryExtents,
    geometry_validation: GeometryValidationSource<'_>,
    geometry_validation_cache: Option<&GeometryValidationCache>,
    batches: &[PreparedDrawBatch],
    command_count: usize,
    source_stats: TessellationStats,
    camera: Camera2d,
    viewport: ResolvedViewport,
    target_viewport: LogicalViewport,
    ready: &mut Vec<ReadyItem<'frame>>,
    statistics: &mut FrameStatistics,
    aggregate: &mut TessellationStats,
    mut owned_batches: Vec<PreparedDrawBatch>,
) -> Result<(), FrameComposerError> {
    let camera_uniform =
        CameraUniform::new_in_region(camera, viewport.viewport, viewport.origin, target_viewport)
            .ok_or(RendererFrameError::InvalidGeometryTransform)?;
    if !geometry_is_safe_for_cached(
        geometry_validation_cache,
        extents,
        geometry_validation,
        camera_uniform,
    ) {
        return Err(RendererFrameError::InvalidGeometryTransform.into());
    }
    owned_batches
        .try_reserve(batches.len())
        .map_err(|_| FrameComposerError::AllocationFailed {
            requested_bytes: batches
                .len()
                .saturating_mul(std::mem::size_of::<PreparedDrawBatch>()),
        })?;
    owned_batches.extend_from_slice(batches);
    *statistics = statistics.adding(FrameStatistics {
        pass_count: 1,
        command_count,
        vertex_count,
        streaming_vertex_count: 0,
        reused_vertex_count: vertex_count,
        upload_bytes: std::mem::size_of::<CameraUniform>(),
        streaming_upload_bytes: 0,
        texture_bytes: 0,
        retained_cpu_bytes: 0,
        retained_buffer_bytes: 0,
        draw_calls: owned_batches.len(),
        source_counts: FrameSourceStatistics::default(),
    });
    add_tessellation_stats(aggregate, source_stats);
    ready.push(ReadyItem::Geometry(ReadyGeometry {
        source,
        vertex_count,
        batches: owned_batches,
        camera_uniform,
        viewport,
    }));
    Ok(())
}

fn resolve_viewport(
    renderer: &WgpuRenderer,
    target: LogicalViewport,
    options: FramePassOptions,
) -> Result<ResolvedViewport, FrameComposerError> {
    let (viewport, origin) = match options.viewport() {
        Some(region) => (region.viewport(), region.origin().to_vec2()),
        None => (target, Vec2::ZERO),
    };
    let max = origin + viewport.size();
    if !origin.is_finite()
        || !max.is_finite()
        || origin.x < 0.0
        || origin.y < 0.0
        || max.x > target.width()
        || max.y > target.height()
    {
        return Err(RendererFrameError::InvalidViewport.into());
    }
    let scissor = logical_viewport_scissor(
        origin,
        viewport,
        renderer.scale_factor as f32,
        renderer.config.width,
        renderer.config.height,
    )
    .ok_or(RendererFrameError::InvalidViewport)?;
    let (item_clip, item_clipped_out) = match options.clip() {
        Some(clip) => {
            let resolved = screen_clip_to_scissor(clip, viewport, renderer.scale_factor as f32)
                .and_then(|clip| offset_scissor(clip, scissor));
            (resolved, resolved.is_none())
        }
        None => (None, false),
    };
    Ok(ResolvedViewport {
        viewport,
        origin,
        scissor,
        item_clip,
        item_clipped_out,
    })
}

fn create_frame_binding(renderer: &mut WgpuRenderer, item: &ReadyItem<'_>) -> FrameBinding {
    match item {
        ReadyItem::Geometry(geometry) => create_camera_binding(renderer, &geometry.camera_uniform),
        ReadyItem::Particle { camera_uniform, .. } => {
            create_camera_binding(renderer, camera_uniform)
        }
        ReadyItem::Scalar {
            texture,
            color_map,
            uniform,
            ..
        } => {
            let color_map_view = renderer.color_map_view(color_map);
            let scalar_view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let bytes = bytemuck::bytes_of(uniform);
            let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sim-engine frame heatmap uniform"),
                size: bytes.len() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            renderer.queue.write_buffer(&buffer, 0, bytes);
            let bind_group = renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sim-engine frame heatmap bind group"),
                    layout: &renderer.heatmap_bind_group_layout,
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
                            resource: buffer.as_entire_binding(),
                        },
                    ],
                });
            FrameBinding {
                _buffer: buffer,
                bind_group,
            }
        }
        ReadyItem::Image {
            image,
            sampling,
            uniform,
            ..
        }
        | ReadyItem::ImageBatch {
            image,
            sampling,
            uniform,
            ..
        } => {
            let bytes = bytemuck::bytes_of(uniform);
            let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sim-engine frame image uniform"),
                size: bytes.len() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            renderer.queue.write_buffer(&buffer, 0, bytes);
            let bind_group = renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sim-engine frame image bind group"),
                    layout: &renderer.image_renderer.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&image.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                renderer.image_renderer.sampler(*sampling),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buffer.as_entire_binding(),
                        },
                    ],
                });
            FrameBinding {
                _buffer: buffer,
                bind_group,
            }
        }
        ReadyItem::Target {
            target, uniform, ..
        } => {
            let bytes = bytemuck::bytes_of(uniform);
            let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sim-engine frame composition uniform"),
                size: bytes.len() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            renderer.queue.write_buffer(&buffer, 0, bytes);
            let bind_group = renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sim-engine frame composition bind group"),
                    layout: &renderer.composition_pipelines.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&target.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffer.as_entire_binding(),
                        },
                    ],
                });
            FrameBinding {
                _buffer: buffer,
                bind_group,
            }
        }
    }
}

fn ready_uniform_bytes(item: &ReadyItem<'_>) -> usize {
    match item {
        ReadyItem::Geometry(_) | ReadyItem::Particle { .. } => std::mem::size_of::<CameraUniform>(),
        ReadyItem::Scalar { .. } => std::mem::size_of::<HeatmapUniform>(),
        ReadyItem::Image { .. } | ReadyItem::ImageBatch { .. } => {
            std::mem::size_of::<ImageUniform>()
        }
        ReadyItem::Target { .. } => std::mem::size_of::<CompositeUniform>(),
    }
}

fn create_camera_binding(renderer: &WgpuRenderer, uniform: &CameraUniform) -> FrameBinding {
    let label = "sim-engine frame camera uniform";
    let bytes = bytemuck::bytes_of(uniform);
    let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    renderer.queue.write_buffer(&buffer, 0, bytes);
    let bind_group = renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine frame camera bind group"),
            layout: &renderer.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
    FrameBinding {
        _buffer: buffer,
        bind_group,
    }
}

fn effective_scissor(
    viewport: ResolvedViewport,
    source_clip: Option<ScreenClipRect>,
    scale_factor: f32,
) -> Option<ScissorRect> {
    if viewport.item_clipped_out {
        return None;
    }
    let source = match source_clip {
        Some(clip) => screen_clip_to_scissor(clip, viewport.viewport, scale_factor)
            .and_then(|clip| offset_scissor(clip, viewport.scissor))?,
        None => viewport.scissor,
    };
    match viewport.item_clip {
        Some(item) => intersect_scissors(source, item),
        None => Some(source),
    }
}

fn intersect_scissors(first: ScissorRect, second: ScissorRect) -> Option<ScissorRect> {
    let x = first.x.max(second.x);
    let y = first.y.max(second.y);
    let max_x = first
        .x
        .saturating_add(first.width)
        .min(second.x.saturating_add(second.width));
    let max_y = first
        .y
        .saturating_add(first.height)
        .min(second.y.saturating_add(second.height));
    (max_x > x && max_y > y).then_some(ScissorRect {
        x,
        y,
        width: max_x - x,
        height: max_y - y,
    })
}

fn validate_frame_budget(
    budget: FrameBudget,
    statistics: FrameStatistics,
) -> Result<(), FrameComposerError> {
    let work = [
        (
            FrameBudgetResource::Passes,
            budget.max_passes,
            statistics.pass_count,
        ),
        (
            FrameBudgetResource::Commands,
            budget.max_commands,
            statistics.command_count,
        ),
        (
            FrameBudgetResource::Vertices,
            budget.max_vertices,
            statistics.vertex_count,
        ),
        (
            FrameBudgetResource::UploadBytes,
            budget.max_upload_bytes,
            statistics.upload_bytes,
        ),
        (
            FrameBudgetResource::TextureBytes,
            budget.max_texture_bytes,
            statistics.texture_bytes,
        ),
        (
            FrameBudgetResource::DrawCalls,
            budget.max_draw_calls,
            statistics.draw_calls,
        ),
    ];
    for (resource, limit, actual) in work {
        if actual > limit {
            return Err(FrameComposerError::BudgetExceeded {
                resource,
                limit,
                actual,
            });
        }
    }
    Ok(())
}

fn add_tessellation_stats(aggregate: &mut TessellationStats, source: TessellationStats) {
    aggregate.command_count = aggregate.command_count.saturating_add(source.command_count);
    aggregate.rendered_command_count = aggregate
        .rendered_command_count
        .saturating_add(source.rendered_command_count);
    aggregate.dropped_command_count = aggregate
        .dropped_command_count
        .saturating_add(source.dropped_command_count);
    aggregate.command_counts = aggregate.command_counts.adding(source.command_counts);
    aggregate.rendered_counts = aggregate.rendered_counts.adding(source.rendered_counts);
    aggregate.dropped_counts = aggregate.dropped_counts.adding(source.dropped_counts);
    aggregate.vertex_count = aggregate.vertex_count.saturating_add(source.vertex_count);
    aggregate.draw_batch_count = aggregate
        .draw_batch_count
        .saturating_add(source.draw_batch_count);
    aggregate.upload_bytes = aggregate.upload_bytes.saturating_add(source.upload_bytes);
}

#[allow(clippy::too_many_arguments)]
fn frame_report(
    status: RenderStatus,
    statistics: FrameStatistics,
    tessellation: Duration,
    upload: Duration,
    camera_uniform_upload: Duration,
    surface_acquire: Duration,
    encode_submit_present: Duration,
    total_cpu: Duration,
    geometry_reused: bool,
    geometry_streamed: bool,
    tessellation_stats: TessellationStats,
) -> FrameReport {
    let report = render_report(
        status,
        tessellation,
        upload,
        camera_uniform_upload,
        surface_acquire,
        encode_submit_present,
        total_cpu,
        geometry_reused,
        geometry_streamed,
        tessellation_stats,
    );
    FrameReport {
        status: report.status,
        statistics,
        metrics: report.metrics,
        gpu_timing_id: None,
    }
}
