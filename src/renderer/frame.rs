use super::*;

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

    /// Returns vertices generated or selected during this frame.
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
        Ok(FrameComposer {
            renderer: self,
            background,
            budget,
            items: Vec::new(),
            retained_resources: Vec::new(),
            scalar_luts: Vec::new(),
            scalar_lut_upload_count: 0,
            scalar_lut_allocation_count: 0,
            planned: FrameStatistics::default(),
            next_insertion: 0,
        })
    }
}

impl<'frame> FrameComposer<'frame> {
    /// Adds a streaming world-space scene through its own camera.
    pub fn draw_scene(
        &mut self,
        scene: &'frame Scene,
        camera: Camera2d,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        let statistics = scene.statistics();
        let work = FrameStatistics {
            pass_count: 1,
            command_count: scene.command_count(),
            vertex_count: statistics.estimated_tessellated_vertices(),
            streaming_vertex_count: statistics.estimated_tessellated_vertices(),
            reused_vertex_count: 0,
            upload_bytes: statistics
                .estimated_upload_bytes()
                .saturating_add(std::mem::size_of::<CameraUniform>()),
            streaming_upload_bytes: statistics.estimated_upload_bytes(),
            texture_bytes: 0,
            retained_cpu_bytes: 0,
            retained_buffer_bytes: 0,
            draw_calls: statistics.estimated_draw_batches(),
            source_counts: FrameSourceStatistics::single(FrameSourceKind::StreamingScene),
        };
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_key(scene, 1),
                scene.allocation_bytes(),
                0,
                0,
            )],
            work,
            FrameItem::Scene {
                scene,
                camera,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds a streaming logical-screen scene unaffected by world cameras.
    pub fn draw_screen_scene(
        &mut self,
        scene: &'frame ScreenScene,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        let statistics = scene.statistics();
        let work = FrameStatistics {
            pass_count: 1,
            command_count: scene.command_count(),
            vertex_count: statistics.estimated_tessellated_vertices(),
            streaming_vertex_count: statistics.estimated_tessellated_vertices(),
            reused_vertex_count: 0,
            upload_bytes: statistics
                .estimated_upload_bytes()
                .saturating_add(std::mem::size_of::<CameraUniform>()),
            streaming_upload_bytes: statistics.estimated_upload_bytes(),
            texture_bytes: 0,
            retained_cpu_bytes: 0,
            retained_buffer_bytes: 0,
            draw_calls: statistics.estimated_draw_batches(),
            source_counts: FrameSourceStatistics::single(FrameSourceKind::StreamingScene),
        };
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_key(scene, 2),
                scene.allocation_bytes(),
                0,
                0,
            )],
            work,
            FrameItem::ScreenScene {
                scene,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds immutable prepared world-space geometry through its own camera.
    pub fn draw_prepared_scene(
        &mut self,
        scene: &'frame PreparedScene,
        camera: Camera2d,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if !prepared_scene_belongs_to(&self.renderer.renderer_identity, &scene.renderer_identity) {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::PreparedScene,
            });
        }
        let work = FrameStatistics {
            pass_count: 1,
            command_count: scene.command_count,
            vertex_count: scene.vertex_count,
            streaming_vertex_count: 0,
            reused_vertex_count: scene.vertex_count,
            upload_bytes: std::mem::size_of::<CameraUniform>(),
            streaming_upload_bytes: 0,
            texture_bytes: 0,
            retained_cpu_bytes: 0,
            retained_buffer_bytes: 0,
            draw_calls: scene.draw_batches.len(),
            source_counts: FrameSourceStatistics::single(FrameSourceKind::PreparedScene),
        };
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_arc_key(&scene.vertices, 13),
                    scene
                        .vertices
                        .capacity()
                        .saturating_mul(std::mem::size_of::<Vertex>()),
                    0,
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_arc_key(&scene.vertex_buffer, 3),
                    0,
                    scene
                        .vertex_count
                        .max(1)
                        .saturating_mul(std::mem::size_of::<Vertex>()),
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_key(&scene.draw_batches, 14),
                    scene
                        .draw_batches
                        .capacity()
                        .saturating_mul(std::mem::size_of::<PreparedDrawBatch>()),
                    0,
                    0,
                ),
            ],
            work,
            FrameItem::Prepared {
                scene,
                camera,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds immutable prepared logical-screen geometry.
    pub fn draw_prepared_screen_scene(
        &mut self,
        scene: &'frame PreparedScreenScene,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if !prepared_scene_belongs_to(
            &self.renderer.renderer_identity,
            &scene.scene.renderer_identity,
        ) {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::PreparedScene,
            });
        }
        let work = FrameStatistics {
            pass_count: 1,
            command_count: scene.scene.command_count,
            vertex_count: scene.scene.vertex_count,
            streaming_vertex_count: 0,
            reused_vertex_count: scene.scene.vertex_count,
            upload_bytes: std::mem::size_of::<CameraUniform>(),
            streaming_upload_bytes: 0,
            texture_bytes: 0,
            retained_cpu_bytes: 0,
            retained_buffer_bytes: 0,
            draw_calls: scene.scene.draw_batches.len(),
            source_counts: FrameSourceStatistics::single(FrameSourceKind::PreparedScene),
        };
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_arc_key(&scene.scene.vertices, 13),
                    scene
                        .scene
                        .vertices
                        .capacity()
                        .saturating_mul(std::mem::size_of::<Vertex>()),
                    0,
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_arc_key(&scene.scene.vertex_buffer, 3),
                    0,
                    scene
                        .scene
                        .vertex_count
                        .max(1)
                        .saturating_mul(std::mem::size_of::<Vertex>()),
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_key(&scene.scene.draw_batches, 14),
                    scene
                        .scene
                        .draw_batches
                        .capacity()
                        .saturating_mul(std::mem::size_of::<PreparedDrawBatch>()),
                    0,
                    0,
                ),
            ],
            work,
            FrameItem::PreparedScreen {
                scene,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds retained dynamic triangles through their own world camera.
    ///
    /// Camera-dependent topology is validated by [`FrameComposer::present`],
    /// which rejects triangles crossing the full target clip volume or having
    /// an ambiguous projected orientation. Item viewport/scissor clipping is
    /// axis-aligned fragment clipping and remains supported.
    pub fn draw_dynamic_mesh(
        &mut self,
        mesh: &'frame DynamicMesh2d,
        camera: Camera2d,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_dynamic_mesh(mesh).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::DynamicMesh,
            });
        }
        let work = FrameStatistics {
            pass_count: 1,
            command_count: usize::from(!mesh.vertices.is_empty()),
            vertex_count: mesh.vertices.len(),
            streaming_vertex_count: 0,
            reused_vertex_count: mesh.vertices.len(),
            upload_bytes: std::mem::size_of::<CameraUniform>(),
            streaming_upload_bytes: 0,
            texture_bytes: 0,
            retained_cpu_bytes: 0,
            retained_buffer_bytes: 0,
            draw_calls: usize::from(!mesh.vertices.is_empty()),
            source_counts: FrameSourceStatistics::single(FrameSourceKind::DynamicMesh),
        };
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_arc_key(&mesh.vertex_buffer, 4),
                mesh.recovery_memory_bytes(),
                mesh.vertex_capacity
                    .saturating_mul(std::mem::size_of::<DynamicGpu>()),
                0,
            )],
            work,
            FrameItem::Dynamic {
                mesh,
                camera,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds a budgeted instanced particle field through its own world camera.
    pub fn draw_particle_field(
        &mut self,
        field: &'frame mut ParticleField2d,
        camera: Camera2d,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_particle_field(field).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::ParticleField,
            });
        }
        let candidate_count = field
            .instances
            .len()
            .min(field.budget.max_visibility_checks_per_frame)
            .min(field.budget.instance_limit());
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_arc_key(&field.instance_buffer, 5),
                field.cpu_allocation_bytes(),
                field.gpu_allocation_bytes(),
                0,
            )],
            FrameStatistics {
                pass_count: 1,
                command_count: usize::from(!field.instances.is_empty()),
                vertex_count: candidate_count.saturating_mul(6),
                streaming_vertex_count: candidate_count.saturating_mul(6),
                reused_vertex_count: 0,
                upload_bytes: candidate_count
                    .saturating_mul(std::mem::size_of::<ParticleGpu>())
                    .saturating_add(std::mem::size_of::<CameraUniform>()),
                streaming_upload_bytes: candidate_count
                    .saturating_mul(std::mem::size_of::<ParticleGpu>()),
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: usize::from(candidate_count > 0),
                source_counts: FrameSourceStatistics::single(FrameSourceKind::ParticleField),
            },
            FrameItem::Particle {
                field,
                camera,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds a retained scalar heatmap scaled into the optional viewport.
    pub fn draw_scalar_field(
        &mut self,
        texture: &'frame ScalarFieldTexture,
        color_map: &'frame ColorMap,
        (minimum, maximum): (f32, f32),
        sampling: ScalarFieldSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self
            .renderer
            .validate_scalar_field_texture(texture)
            .is_err()
        {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::ScalarField,
            });
        }
        let value_extent = scalar_value_range_extent(minimum, maximum)
            .ok_or(FrameComposerError::InvalidValueRange { minimum, maximum })?;
        if !scalar_normalization_is_portable(texture, minimum, value_extent) {
            return Err(FrameComposerError::InvalidValueRange { minimum, maximum });
        }
        let color_map_bytes = COLOR_MAP_LUT_SIZE as usize * 4;
        let scalar_lut = ScalarLutPlan {
            sort_key: (options.order(), self.next_insertion),
            lut: color_map_lut(color_map),
        };
        self.scalar_luts
            .try_reserve(1)
            .map_err(|_| FrameComposerError::AllocationFailed {
                requested_bytes: std::mem::size_of::<ScalarLutPlan>(),
            })?;
        let insertion = self
            .scalar_luts
            .partition_point(|planned| planned.sort_key < scalar_lut.sort_key);
        let (lut_upload_count, lut_allocation_count) = scalar_lut_counts_with_inserted(
            &self.scalar_luts,
            insertion,
            &scalar_lut.lut,
            self.renderer
                .color_map_cache
                .as_ref()
                .map(|cached| &cached.lut),
        );
        let additional_lut_uploads = lut_upload_count.saturating_sub(self.scalar_lut_upload_count);
        let additional_lut_allocations =
            lut_allocation_count.saturating_sub(self.scalar_lut_allocation_count);
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_key(&texture.texture, 6),
                    texture.recovery_memory_bytes(),
                    0,
                    texture.gpu_allocation_bytes(),
                ),
                RetainedResourceAccounting::new(
                    retained_key(color_map, 15),
                    color_map.allocation_bytes(),
                    0,
                    0,
                ),
            ],
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<HeatmapUniform>()
                    .saturating_add(additional_lut_uploads.saturating_mul(color_map_bytes)),
                streaming_upload_bytes: 0,
                texture_bytes: additional_lut_allocations.saturating_mul(color_map_bytes),
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: 1,
                source_counts: FrameSourceStatistics::single(FrameSourceKind::ScalarField),
            },
            FrameItem::Scalar {
                texture,
                color_map,
                minimum,
                maximum,
                value_extent,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )?;
        self.scalar_luts.insert(insertion, scalar_lut);
        self.scalar_lut_upload_count = lut_upload_count;
        self.scalar_lut_allocation_count = lut_allocation_count;
        Ok(())
    }

    /// Adds one image or atlas region scaled into the optional viewport.
    ///
    /// With no viewport the image covers the complete logical surface. `tint`
    /// is normalized straight linear RGBA and multiplies decoded image color.
    pub fn draw_image(
        &mut self,
        image: &'frame Image2d,
        source: Option<ImageTexelRect>,
        tint: Color,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_image(image).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::Image,
            });
        }
        if !tint.is_normalized() {
            return Err(FrameComposerError::InvalidTint);
        }
        let source = source.unwrap_or_else(|| image.full_rect());
        if !source.fits(image.width(), image.height()) {
            return Err(FrameComposerError::InvalidImageRegion);
        }
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_arc_key(&image.resource_identity, 7),
                image.recovery_memory_bytes(),
                0,
                image.gpu_allocation_bytes(),
            )],
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: 1,
                source_counts: FrameSourceStatistics::single(FrameSourceKind::Image),
            },
            FrameItem::Image {
                image,
                source,
                tint,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds one image or atlas region on an axis-aligned world-space rectangle.
    ///
    /// The rectangle follows the supplied camera's zoom, rotation, and
    /// pseudo-depth projection. Its UV top edge maps to the rectangle's maximum
    /// world Y edge. The optional frame viewport selects the camera viewport and
    /// the item remains constrained by that viewport and clip.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_world_image(
        &mut self,
        image: &'frame Image2d,
        source: Option<ImageTexelRect>,
        rectangle: Rect,
        depth: f32,
        camera: Camera2d,
        tint: Color,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_image(image).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::Image,
            });
        }
        if !tint.is_normalized() {
            return Err(FrameComposerError::InvalidTint);
        }
        let source = source.unwrap_or_else(|| image.full_rect());
        if !source.fits(image.width(), image.height()) {
            return Err(FrameComposerError::InvalidImageRegion);
        }
        let rectangle = rectangle.normalized();
        if !rectangle.min().is_finite()
            || !rectangle.max().is_finite()
            || !rectangle.width().is_finite()
            || !rectangle.height().is_finite()
            || rectangle.width() <= 0.0
            || rectangle.height() <= 0.0
            || !depth.is_finite()
        {
            return Err(FrameComposerError::InvalidWorldImage);
        }
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_arc_key(&image.resource_identity, 7),
                image.recovery_memory_bytes(),
                0,
                image.gpu_allocation_bytes(),
            )],
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: 1,
                source_counts: FrameSourceStatistics::single(FrameSourceKind::Image),
            },
            FrameItem::WorldImage {
                image,
                source,
                rectangle,
                depth,
                camera,
                tint,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds a retained atlas batch as one instanced draw when non-empty.
    /// Empty batches are valid frame items and emit no draw call.
    pub fn draw_image_batch(
        &mut self,
        image: &'frame Image2d,
        batch: &'frame ImageBatch2d,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_image_batch(image, batch).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::Image,
            });
        }
        self.push_retained_image_batch(image, batch, sampling, options, FrameSourceKind::Image)
    }

    /// Adds one host-shaped retained glyph run as one instanced draw when non-empty.
    ///
    /// Positions are local logical pixels. Mixed font fallback is represented
    /// by multiple runs submitted at the same order; stable insertion order is
    /// preserved between them. An empty run is valid and emits no draw call.
    pub fn draw_glyph_run(
        &mut self,
        atlas: &'frame GlyphAtlas2d,
        run: &'frame GlyphRun2d,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_glyph_run(atlas, run).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::Glyph,
            });
        }
        let image = &atlas.image;
        let batch = &run.batch;
        let vertex_count = batch.sprite_count().saturating_mul(6);
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_arc_key(&image.resource_identity, 7),
                    image.recovery_memory_bytes(),
                    0,
                    image.gpu_allocation_bytes(),
                ),
                RetainedResourceAccounting::new(
                    retained_key(atlas, 9),
                    atlas
                        .recovery_memory_bytes()
                        .saturating_sub(image.recovery_memory_bytes()),
                    0,
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_key(run, 10),
                    run.recovery_memory_bytes(),
                    run.gpu_allocation_bytes(),
                    0,
                ),
            ],
            FrameStatistics {
                pass_count: 1,
                command_count: usize::from(batch.sprite_count() > 0),
                vertex_count,
                streaming_vertex_count: 0,
                reused_vertex_count: vertex_count,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: usize::from(batch.sprite_count() > 0),
                source_counts: FrameSourceStatistics::single(FrameSourceKind::Glyph),
            },
            FrameItem::ImageBatch {
                image,
                batch,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    fn push_retained_image_batch(
        &mut self,
        image: &'frame Image2d,
        batch: &'frame ImageBatch2d,
        sampling: ImageSampling,
        options: FramePassOptions,
        source: FrameSourceKind,
    ) -> Result<(), FrameComposerError> {
        let vertex_count = batch.sprite_count().saturating_mul(6);
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_arc_key(&image.resource_identity, 7),
                    image.recovery_memory_bytes(),
                    0,
                    image.gpu_allocation_bytes(),
                ),
                RetainedResourceAccounting::new(
                    retained_key(batch, 8),
                    batch.recovery_memory_bytes(),
                    batch.gpu_allocation_bytes(),
                    0,
                ),
            ],
            FrameStatistics {
                pass_count: 1,
                command_count: usize::from(batch.sprite_count() > 0),
                vertex_count,
                streaming_vertex_count: 0,
                reused_vertex_count: vertex_count,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: usize::from(batch.sprite_count() > 0),
                source_counts: FrameSourceStatistics::single(source),
            },
            FrameItem::ImageBatch {
                image,
                batch,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds an offscreen target, scaled into the optional logical viewport.
    pub fn draw_render_target(
        &mut self,
        target: &'frame RenderTarget2d,
        blend_mode: BlendMode,
        opacity: f32,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_render_target(target).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::RenderTarget,
            });
        }
        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return Err(FrameComposerError::InvalidOpacity);
        }
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_arc_key(&target.resource_identity, 11),
                0,
                0,
                target.allocation_bytes,
            )],
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<CompositeUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: 1,
                source_counts: FrameSourceStatistics::single(FrameSourceKind::RenderTarget),
            },
            FrameItem::Target {
                target,
                blend_mode,
                opacity,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Returns conservative work accepted so far without presenting it.
    pub const fn planned_statistics(&self) -> FrameStatistics {
        self.planned
    }

    /// Validates, orders, encodes, submits, and presents the complete frame.
    pub fn present(self) -> Result<FrameReport, FrameComposerError> {
        present_frame(self)
    }

    fn push_item(
        &mut self,
        work: FrameStatistics,
        item: FrameItem<'frame>,
    ) -> Result<(), FrameComposerError> {
        let planned = self.planned.adding(work);
        validate_frame_budget(self.budget, planned)?;
        self.items
            .try_reserve(1)
            .map_err(|_| FrameComposerError::AllocationFailed {
                requested_bytes: std::mem::size_of::<FrameItem<'frame>>(),
            })?;
        self.items.push(item);
        self.planned = planned;
        self.next_insertion = self.next_insertion.saturating_add(1);
        Ok(())
    }

    fn push_accounted_item(
        &mut self,
        resources: &[RetainedResourceAccounting],
        mut work: FrameStatistics,
        item: FrameItem<'frame>,
    ) -> Result<(), FrameComposerError> {
        let (accounted_work, missing) =
            account_new_retained_resources(&self.retained_resources, resources, work);
        work = accounted_work;
        self.retained_resources.try_reserve(missing).map_err(|_| {
            FrameComposerError::AllocationFailed {
                requested_bytes: missing.saturating_mul(std::mem::size_of::<RetainedResourceKey>()),
            }
        })?;
        self.push_item(work, item)?;
        for resource in resources {
            if !self.retained_resources.contains(&resource.key) {
                self.retained_resources.push(resource.key);
            }
        }
        Ok(())
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

struct FrameBinding {
    _buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

fn present_frame(composer: FrameComposer<'_>) -> Result<FrameReport, FrameComposerError> {
    let FrameComposer {
        renderer,
        background,
        budget,
        items,
        retained_resources: _,
        scalar_luts: _,
        scalar_lut_upload_count: _,
        scalar_lut_allocation_count: _,
        planned,
        next_insertion: _,
    } = composer;
    // Always return the transient allocation, including structured error
    // paths before surface acquisition.
    let mut streaming_vertices = std::mem::take(&mut renderer.vertices);
    streaming_vertices.clear();
    let result = present_frame_with_vertices(
        renderer,
        background,
        budget,
        items,
        planned,
        &mut streaming_vertices,
    );
    renderer.vertices = streaming_vertices;
    result
}

fn present_frame_with_vertices<'frame>(
    renderer: &mut WgpuRenderer,
    background: Color,
    budget: FrameBudget,
    mut items: Vec<FrameItem<'frame>>,
    planned: FrameStatistics,
    streaming_vertices: &mut Vec<Vertex>,
) -> Result<FrameReport, FrameComposerError> {
    let frame_started_at = Instant::now();
    items.sort_unstable_by_key(FrameItem::sort_key);

    let target_viewport = renderer
        .logical_viewport()
        .map_err(|_| RendererFrameError::InvalidViewport)?;
    let tessellation_started_at = Instant::now();
    preflight_frame_items(renderer, target_viewport, &items)?;
    let mut ready = Vec::new();
    ready
        .try_reserve(items.len())
        .map_err(|_| FrameComposerError::AllocationFailed {
            requested_bytes: items
                .len()
                .saturating_mul(std::mem::size_of::<ReadyItem<'_>>()),
        })?;
    let mut statistics = FrameStatistics::default();
    let mut tessellation_stats = TessellationStats::default();
    let mut geometry_reused = false;
    let mut geometry_streamed = false;
    let mut simulated_color_map_lut = renderer.color_map_cache.as_ref().map(|cache| cache.lut);

    for item in items {
        match item {
            FrameItem::Scene {
                scene,
                camera,
                options,
                ..
            } => {
                prepare_streaming_scene(
                    scene,
                    camera,
                    options,
                    renderer,
                    target_viewport,
                    &mut *streaming_vertices,
                    &mut ready,
                    &mut statistics,
                    &mut tessellation_stats,
                )?;
                geometry_streamed = true;
            }
            FrameItem::ScreenScene { scene, options, .. } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let camera = screen_camera(viewport.viewport)
                    .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
                let camera_uniform = CameraUniform::new_in_region(
                    camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                if !scene_estimate_fits_streaming_device(
                    scene.as_scene(),
                    &renderer.device,
                    streaming_vertices.len(),
                    renderer.vertex_capacity,
                ) {
                    return Err(RendererFrameError::GeometryCapacityTooLarge.into());
                }
                prepare_streaming_scene_resolved(
                    scene.as_scene(),
                    camera_uniform,
                    viewport,
                    &mut *streaming_vertices,
                    &mut ready,
                    &mut statistics,
                    &mut tessellation_stats,
                )?;
                geometry_streamed = true;
            }
            FrameItem::Prepared {
                scene,
                camera,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                prepare_retained_geometry(
                    ReadySource::Prepared(&scene.vertex_buffer),
                    scene.vertex_count,
                    scene.geometry_extents,
                    GeometryValidationSource::Tessellated(&scene.vertices),
                    Some(&scene.geometry_validation_cache),
                    &scene.draw_batches,
                    scene.command_count,
                    scene.tessellation,
                    camera,
                    viewport,
                    target_viewport,
                    &mut ready,
                    &mut statistics,
                    &mut tessellation_stats,
                )?;
                geometry_reused = true;
            }
            FrameItem::PreparedScreen { scene, options, .. } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let camera = screen_camera(viewport.viewport)
                    .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
                prepare_retained_geometry(
                    ReadySource::Prepared(&scene.scene.vertex_buffer),
                    scene.scene.vertex_count,
                    scene.scene.geometry_extents,
                    GeometryValidationSource::Tessellated(&scene.scene.vertices),
                    Some(&scene.scene.geometry_validation_cache),
                    &scene.scene.draw_batches,
                    scene.scene.command_count,
                    scene.scene.tessellation,
                    camera,
                    viewport,
                    target_viewport,
                    &mut ready,
                    &mut statistics,
                    &mut tessellation_stats,
                )?;
                geometry_reused = true;
            }
            FrameItem::Dynamic {
                mesh,
                camera,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let batch = (!mesh.vertices.is_empty()).then_some(PreparedDrawBatch {
                    vertex_range: 0..mesh.vertices.len() as u32,
                    screen_clip: None,
                });
                prepare_retained_geometry(
                    ReadySource::Dynamic(&mesh.vertex_buffer),
                    mesh.vertices.len(),
                    mesh.geometry_extents,
                    GeometryValidationSource::Dynamic(&mesh.vertices),
                    Some(&mesh.geometry_validation_cache),
                    batch.as_slice(),
                    usize::from(!mesh.vertices.is_empty()),
                    TessellationStats::default(),
                    camera,
                    viewport,
                    target_viewport,
                    &mut ready,
                    &mut statistics,
                    &mut tessellation_stats,
                )?;
                geometry_streamed = true;
            }
            FrameItem::Particle {
                field,
                camera,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                prepare_particle_item(
                    field,
                    camera,
                    viewport,
                    target_viewport,
                    &mut ready,
                    &mut statistics,
                )?;
                geometry_streamed = true;
            }
            FrameItem::Scalar {
                texture,
                color_map,
                minimum,
                maximum,
                value_extent,
                sampling,
                options,
                ..
            } => {
                if !scalar_normalization_is_portable(texture, minimum, value_extent) {
                    return Err(FrameComposerError::InvalidValueRange { minimum, maximum });
                }
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let region = LogicalViewportRegion::new(
                    LogicalScreenPosition::from_vec2(viewport.origin),
                    viewport.viewport,
                )
                .map_err(|_| RendererFrameError::InvalidViewport)?;
                let uniform = HeatmapUniform::new(
                    minimum,
                    value_extent,
                    texture.width(),
                    texture.height(),
                    sampling,
                )
                .in_region(region, target_viewport)
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                let color_map_bytes = COLOR_MAP_LUT_SIZE as usize * 4;
                let lut = color_map_lut(color_map);
                let color_map_upload_bytes = if simulated_color_map_lut == Some(lut) {
                    0
                } else {
                    simulated_color_map_lut = Some(lut);
                    color_map_bytes
                };
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: 1,
                    vertex_count: 6,
                    streaming_vertex_count: 0,
                    reused_vertex_count: 0,
                    upload_bytes: std::mem::size_of::<HeatmapUniform>()
                        .saturating_add(color_map_upload_bytes),
                    streaming_upload_bytes: 0,
                    texture_bytes: 0,
                    retained_cpu_bytes: 0,
                    retained_buffer_bytes: 0,
                    draw_calls: 1,
                    source_counts: FrameSourceStatistics::default(),
                });
                ready.push(ReadyItem::Scalar {
                    texture,
                    color_map,
                    uniform,
                    viewport,
                });
                geometry_streamed = true;
            }
            FrameItem::Image {
                image,
                source,
                tint,
                sampling,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let region = LogicalViewportRegion::new(
                    LogicalScreenPosition::from_vec2(viewport.origin),
                    viewport.viewport,
                )
                .map_err(|_| RendererFrameError::InvalidViewport)?;
                let destination = CompositeUniform::in_region(1.0, region, target_viewport)
                    .ok_or(RendererFrameError::InvalidGeometryTransform)?
                    .destination;
                let image_width = image.width() as f32;
                let image_height = image.height() as f32;
                let uv_min_x = (source.x() as f32 + 0.5) / image_width;
                let uv_min_y = (source.y() as f32 + 0.5) / image_height;
                let uv_max_x = (source.x() as f32 + source.width() as f32 - 0.5) / image_width;
                let uv_max_y = (source.y() as f32 + source.height() as f32 - 0.5) / image_height;
                let uniform = ImageUniform {
                    destination,
                    uv_rect: [uv_min_x, uv_min_y, uv_max_x, uv_max_y],
                    tint: tint.to_array(),
                    world_clip_x: [0.0; 4],
                    world_clip_y: [0.0; 4],
                    world_mode: [0.0; 4],
                };
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: 1,
                    vertex_count: 6,
                    streaming_vertex_count: 0,
                    reused_vertex_count: 0,
                    upload_bytes: std::mem::size_of::<ImageUniform>(),
                    streaming_upload_bytes: 0,
                    texture_bytes: image.gpu_allocation_bytes(),
                    retained_cpu_bytes: 0,
                    retained_buffer_bytes: 0,
                    draw_calls: 1,
                    source_counts: FrameSourceStatistics::default(),
                });
                ready.push(ReadyItem::Image {
                    image,
                    sampling,
                    uniform,
                    viewport,
                });
            }
            FrameItem::WorldImage {
                image,
                source,
                rectangle,
                depth,
                camera,
                tint,
                sampling,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let camera_uniform = CameraUniform::new_in_region(
                    camera,
                    viewport.viewport,
                    viewport.origin,
                    target_viewport,
                )
                .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                let minimum = rectangle.min();
                let maximum = rectangle.max();
                let world_corners = [
                    Vec2::new(minimum.x, maximum.y),
                    maximum,
                    minimum,
                    Vec2::new(maximum.x, minimum.y),
                ];
                let mut clip_corners = [Vec2::ZERO; 4];
                for (clip, world) in clip_corners.iter_mut().zip(world_corners) {
                    let screen = camera_uniform.world_to_screen(world, depth);
                    *clip = Vec2::new(
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
                let image_width = image.width() as f32;
                let image_height = image.height() as f32;
                let uniform = ImageUniform {
                    destination: [0.0; 4],
                    uv_rect: [
                        (source.x() as f32 + 0.5) / image_width,
                        (source.y() as f32 + 0.5) / image_height,
                        (source.x() as f32 + source.width() as f32 - 0.5) / image_width,
                        (source.y() as f32 + source.height() as f32 - 0.5) / image_height,
                    ],
                    tint: tint.to_array(),
                    world_clip_x: [
                        clip_corners[0].x,
                        clip_corners[1].x,
                        clip_corners[2].x,
                        clip_corners[3].x,
                    ],
                    world_clip_y: [
                        clip_corners[0].y,
                        clip_corners[1].y,
                        clip_corners[2].y,
                        clip_corners[3].y,
                    ],
                    world_mode: [1.0, 0.0, 0.0, 0.0],
                };
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: 1,
                    vertex_count: 6,
                    streaming_vertex_count: 0,
                    reused_vertex_count: 0,
                    upload_bytes: std::mem::size_of::<ImageUniform>(),
                    streaming_upload_bytes: 0,
                    texture_bytes: image.gpu_allocation_bytes(),
                    retained_cpu_bytes: 0,
                    retained_buffer_bytes: 0,
                    draw_calls: 1,
                    source_counts: FrameSourceStatistics::default(),
                });
                ready.push(ReadyItem::Image {
                    image,
                    sampling,
                    uniform,
                    viewport,
                });
            }
            FrameItem::ImageBatch {
                image,
                batch,
                sampling,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let uniform = ImageUniform {
                    destination: [
                        2.0 / target_viewport.width(),
                        -2.0 / target_viewport.height(),
                        -1.0,
                        1.0,
                    ],
                    uv_rect: [viewport.origin.x, viewport.origin.y, 0.0, 0.0],
                    tint: Color::WHITE.to_array(),
                    world_clip_x: [0.0; 4],
                    world_clip_y: [0.0; 4],
                    world_mode: [0.0; 4],
                };
                if !image_sprites_are_safe_for_target(
                    batch.sprites(),
                    viewport.origin,
                    uniform.destination,
                ) {
                    return Err(RendererFrameError::InvalidGeometryTransform.into());
                }
                let vertex_count = batch.sprite_count().saturating_mul(6);
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: usize::from(batch.sprite_count() > 0),
                    vertex_count,
                    streaming_vertex_count: 0,
                    reused_vertex_count: vertex_count,
                    upload_bytes: std::mem::size_of::<ImageUniform>(),
                    streaming_upload_bytes: 0,
                    texture_bytes: image.gpu_allocation_bytes(),
                    retained_cpu_bytes: 0,
                    retained_buffer_bytes: 0,
                    draw_calls: usize::from(batch.sprite_count() > 0),
                    source_counts: FrameSourceStatistics::default(),
                });
                ready.push(ReadyItem::ImageBatch {
                    image,
                    batch,
                    sampling,
                    uniform,
                    viewport,
                });
                geometry_reused = true;
            }
            FrameItem::Target {
                target,
                blend_mode,
                opacity,
                options,
                ..
            } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let region = LogicalViewportRegion::new(
                    LogicalScreenPosition::from_vec2(viewport.origin),
                    viewport.viewport,
                )
                .map_err(|_| RendererFrameError::InvalidViewport)?;
                let uniform = CompositeUniform::in_region(opacity, region, target_viewport)
                    .ok_or(RendererFrameError::InvalidGeometryTransform)?;
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: 1,
                    vertex_count: 6,
                    streaming_vertex_count: 0,
                    reused_vertex_count: 0,
                    upload_bytes: std::mem::size_of::<CompositeUniform>(),
                    streaming_upload_bytes: 0,
                    texture_bytes: target.allocation_bytes,
                    retained_cpu_bytes: 0,
                    retained_buffer_bytes: 0,
                    draw_calls: 1,
                    source_counts: FrameSourceStatistics::default(),
                });
                ready.push(ReadyItem::Target {
                    target,
                    blend_mode,
                    uniform,
                    viewport,
                });
            }
        }
    }
    statistics.texture_bytes = planned.texture_bytes;
    let (particle_cpu_before, particle_cpu_after) =
        ready
            .iter()
            .fold((0usize, 0usize), |(before, after), item| match item {
                ReadyItem::Particle {
                    field,
                    initial_cpu_allocation_bytes,
                    ..
                } => (
                    before.saturating_add(*initial_cpu_allocation_bytes),
                    after.saturating_add(field.cpu_allocation_bytes()),
                ),
                _ => (before, after),
            });
    statistics.retained_cpu_bytes = planned
        .retained_cpu_bytes
        .saturating_sub(particle_cpu_before)
        .saturating_add(particle_cpu_after);
    statistics.retained_buffer_bytes = planned.retained_buffer_bytes;
    statistics.source_counts = planned.source_counts;
    validate_frame_budget(budget, statistics)?;
    renderer.ensure_vertex_capacity(streaming_vertices.len())?;

    let mut bindings = Vec::new();
    bindings
        .try_reserve(ready.len())
        .map_err(|_| FrameComposerError::AllocationFailed {
            requested_bytes: ready
                .len()
                .saturating_mul(std::mem::size_of::<FrameBinding>()),
        })?;
    let tessellation = tessellation_started_at.elapsed();

    // Surface availability is resolved after all fallible CPU preparation but
    // before bind-group creation or queue writes. Repeated skipped frames must
    // not accumulate wgpu's deferred upload staging allocations.
    let acquire_started_at = Instant::now();
    let surface_texture = match renderer.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture)
        | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
        wgpu::CurrentSurfaceTexture::Timeout => {
            set_particle_rendered(&mut ready, false);
            return Ok(frame_report(
                RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                statistics.without_uploads(),
                tessellation,
                Duration::ZERO,
                Duration::ZERO,
                acquire_started_at.elapsed(),
                Duration::ZERO,
                frame_started_at.elapsed(),
                geometry_reused,
                geometry_streamed,
                tessellation_stats,
            ));
        }
        wgpu::CurrentSurfaceTexture::Occluded => {
            set_particle_rendered(&mut ready, false);
            return Ok(frame_report(
                RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
                statistics.without_uploads(),
                tessellation,
                Duration::ZERO,
                Duration::ZERO,
                acquire_started_at.elapsed(),
                Duration::ZERO,
                frame_started_at.elapsed(),
                geometry_reused,
                geometry_streamed,
                tessellation_stats,
            ));
        }
        wgpu::CurrentSurfaceTexture::Outdated => {
            let _ = renderer.resize(renderer.config.width, renderer.config.height);
            set_particle_rendered(&mut ready, false);
            return Ok(frame_report(
                RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                statistics.without_uploads(),
                tessellation,
                Duration::ZERO,
                Duration::ZERO,
                acquire_started_at.elapsed(),
                Duration::ZERO,
                frame_started_at.elapsed(),
                geometry_reused,
                geometry_streamed,
                tessellation_stats,
            ));
        }
        wgpu::CurrentSurfaceTexture::Lost => {
            return Err(RendererFrameError::Surface(RendererSurfaceStatus::Lost).into());
        }
        wgpu::CurrentSurfaceTexture::Validation => {
            return Err(RendererFrameError::Surface(RendererSurfaceStatus::Validation).into());
        }
    };
    let surface_acquire = acquire_started_at.elapsed();

    let mut binding_upload = Duration::ZERO;
    let mut camera_uniform_upload = Duration::ZERO;
    for item in &ready {
        let binding_started_at = Instant::now();
        bindings.push(create_frame_binding(renderer, item));
        let elapsed = binding_started_at.elapsed();
        match item {
            ReadyItem::Geometry(_) | ReadyItem::Particle { .. } => {
                camera_uniform_upload += elapsed;
            }
            ReadyItem::Scalar { .. }
            | ReadyItem::Image { .. }
            | ReadyItem::ImageBatch { .. }
            | ReadyItem::Target { .. } => {
                binding_upload += elapsed;
            }
        }
    }

    let upload_started_at = Instant::now();
    if !streaming_vertices.is_empty() {
        renderer.queue.write_buffer(
            &renderer.vertex_buffer,
            0,
            bytemuck::cast_slice(streaming_vertices),
        );
    }
    for item in &ready {
        if let ReadyItem::Particle {
            field,
            visible_count,
            ..
        } = item
        {
            let visible_instances =
                if field.visible_instances.is_empty() && *visible_count == field.instances.len() {
                    field.instances.as_slice()
                } else {
                    field.visible_instances.as_slice()
                };
            if !visible_instances.is_empty() {
                renderer.queue.write_buffer(
                    &field.instance_buffer,
                    0,
                    bytemuck::cast_slice(visible_instances),
                );
            }
        }
    }
    let upload = upload_started_at.elapsed() + binding_upload;
    let encode_started_at = Instant::now();
    let surface_view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let (view, resolve_target) = match &renderer.multisample_target {
        Some(target) => (&target.view, Some(&surface_view)),
        None => (&surface_view, None),
    };
    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine composed frame encoder"),
        });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sim-engine composed frame pass"),
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
        for (item, binding) in ready.iter().zip(&bindings) {
            encode_ready_item(renderer, &mut pass, item, binding);
        }
    }
    renderer.queue.submit([encoder.finish()]);
    renderer.notify_before_present();
    renderer.queue.present(surface_texture);
    set_particle_rendered(&mut ready, true);
    let encode_submit_present = encode_started_at.elapsed();
    Ok(frame_report(
        RenderStatus::Drawn,
        statistics,
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
            FrameItem::ImageBatch { batch, options, .. } => {
                let viewport = resolve_viewport(renderer, target_viewport, *options)?;
                let clip_transform = [
                    2.0 / target_viewport.width(),
                    -2.0 / target_viewport.height(),
                    -1.0,
                    1.0,
                ];
                if !image_sprites_are_safe_for_target(
                    batch.sprites(),
                    viewport.origin,
                    clip_transform,
                ) {
                    return Err(RendererFrameError::InvalidGeometryTransform.into());
                }
            }
        }
    }
    Ok(())
}

fn image_sprites_are_safe_for_target(
    sprites: &[ImageSprite2d],
    viewport_origin: Vec2,
    clip_transform: [f32; 4],
) -> bool {
    if ![viewport_origin.x, viewport_origin.y]
        .into_iter()
        .chain(clip_transform)
        .all(is_portable_shader_source)
    {
        return false;
    }
    sprites.iter().all(|sprite| {
        let destination = sprite.destination();
        let origin = destination.origin().to_vec2();
        let size = destination.viewport().size();
        if ![origin.x, origin.y, size.x, size.y]
            .into_iter()
            .all(is_portable_shader_source)
        {
            return false;
        }
        let horizontal = shader_interval_sum_range([
            (f64::from(viewport_origin.x), f64::from(viewport_origin.x)),
            (f64::from(origin.x), f64::from(origin.x)),
            (0.0, f64::from(size.x)),
        ]);
        let vertical = shader_interval_sum_range([
            (f64::from(viewport_origin.y), f64::from(viewport_origin.y)),
            (f64::from(origin.y), f64::from(origin.y)),
            (0.0, f64::from(size.y)),
        ]);
        horizontal.is_some_and(|horizontal| {
            shader_clip_interval_is_safe(
                horizontal.0,
                horizontal.1,
                clip_transform[0],
                clip_transform[2],
            )
        }) && vertical.is_some_and(|vertical| {
            shader_clip_interval_is_safe(
                vertical.0,
                vertical.1,
                clip_transform[1],
                clip_transform[3],
            )
        })
    })
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

#[allow(clippy::too_many_arguments)]
fn prepare_streaming_scene<'frame>(
    scene: &Scene,
    camera: Camera2d,
    options: FramePassOptions,
    renderer: &WgpuRenderer,
    target_viewport: LogicalViewport,
    streaming_vertices: &mut Vec<Vertex>,
    ready: &mut Vec<ReadyItem<'frame>>,
    statistics: &mut FrameStatistics,
    aggregate: &mut TessellationStats,
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
    prepare_streaming_scene_resolved(
        scene,
        camera_uniform,
        viewport,
        streaming_vertices,
        ready,
        statistics,
        aggregate,
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
) -> Result<(), FrameComposerError> {
    let vertex_start = streaming_vertices.len();
    let mut batches = Vec::new();
    let stats = tessellate_scene(scene, streaming_vertices, &mut batches)
        .map_err(RendererFrameError::from)?;
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
        batches,
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
    let mut owned_batches = Vec::new();
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

fn encode_ready_item<'pass>(
    renderer: &'pass WgpuRenderer,
    pass: &mut wgpu::RenderPass<'pass>,
    item: &'pass ReadyItem<'_>,
    binding: &'pass FrameBinding,
) {
    match item {
        ReadyItem::Geometry(geometry) => {
            if geometry.vertex_count == 0 {
                return;
            }
            let (pipeline, vertex_buffer) = match geometry.source {
                ReadySource::Streaming => (&renderer.pipeline, renderer.vertex_buffer.as_ref()),
                ReadySource::Prepared(buffer) => (&renderer.pipeline, buffer),
                ReadySource::Dynamic(buffer) => (&renderer.dynamic_pipeline, buffer),
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &binding.bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            for batch in &geometry.batches {
                let Some(scissor) = effective_scissor(
                    geometry.viewport,
                    batch.screen_clip,
                    renderer.scale_factor as f32,
                ) else {
                    continue;
                };
                pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                pass.draw(batch.vertex_range.clone(), 0..1);
            }
        }
        ReadyItem::Particle {
            field,
            visible_count,
            viewport,
            ..
        } => {
            if *visible_count == 0 || viewport.item_clipped_out {
                return;
            }
            let scissor = viewport.item_clip.unwrap_or(viewport.scissor);
            pass.set_pipeline(&renderer.particle_pipeline);
            pass.set_bind_group(0, &binding.bind_group, &[]);
            pass.set_vertex_buffer(0, renderer.particle_unit_buffer.slice(..));
            pass.set_vertex_buffer(1, field.instance_buffer.slice(..));
            pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
            pass.draw(0..6, 0..*visible_count as u32);
        }
        ReadyItem::Scalar { viewport, .. } => {
            if viewport.item_clipped_out {
                return;
            }
            let scissor = viewport.item_clip.unwrap_or(viewport.scissor);
            pass.set_pipeline(&renderer.heatmap_pipeline);
            pass.set_bind_group(0, &binding.bind_group, &[]);
            pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
            pass.draw(0..6, 0..1);
        }
        ReadyItem::Image { viewport, .. } => {
            if viewport.item_clipped_out {
                return;
            }
            let scissor = viewport.item_clip.unwrap_or(viewport.scissor);
            pass.set_pipeline(&renderer.image_renderer.pipeline);
            pass.set_bind_group(0, &binding.bind_group, &[]);
            pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
            pass.draw(0..6, 0..1);
        }
        ReadyItem::ImageBatch {
            batch, viewport, ..
        } => {
            if batch.sprite_count() == 0 || viewport.item_clipped_out {
                return;
            }
            let scissor = viewport.item_clip.unwrap_or(viewport.scissor);
            pass.set_pipeline(&renderer.image_renderer.batch_pipeline);
            pass.set_bind_group(0, &binding.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.instance_buffer.slice(..));
            pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
            pass.draw(0..6, 0..batch.sprite_count() as u32);
        }
        ReadyItem::Target {
            blend_mode,
            viewport,
            ..
        } => {
            if viewport.item_clipped_out {
                return;
            }
            let scissor = viewport.item_clip.unwrap_or(viewport.scissor);
            pass.set_pipeline(renderer.composition_pipelines.pipeline(*blend_mode));
            pass.set_bind_group(0, &binding.bind_group, &[]);
            pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
            pass.draw(0..6, 0..1);
        }
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_lut_accounting_matches_one_entry_cache_runs() {
        const BYTES: usize = COLOR_MAP_LUT_SIZE as usize * 4;
        let a = [1; BYTES];
        let b = [2; BYTES];
        let adjacent = vec![
            ScalarLutPlan {
                sort_key: (0, 0),
                lut: a,
            },
            ScalarLutPlan {
                sort_key: (0, 2),
                lut: a,
            },
        ];
        assert_eq!(
            scalar_lut_counts_with_inserted(&adjacent, 1, &a, Some(&a)),
            (0, 1)
        );
        assert_eq!(
            scalar_lut_counts_with_inserted(&adjacent, 1, &b, Some(&a)),
            (2, 3),
            "A, B, A performs two uploads and keeps three LUT allocations alive"
        );
        assert_eq!(
            scalar_lut_counts_with_inserted(&adjacent[..1], 1, &b, None),
            (2, 2)
        );
    }

    #[test]
    fn image_and_glyph_sprite_bounds_include_final_clip_arithmetic() {
        let source = ImageTexelRect::new(0, 0, 1, 1).unwrap();
        let safe = ImageSprite2d::new(
            source,
            LogicalViewportRegion::new(
                LogicalScreenPosition::new(0.0, 0.0),
                LogicalViewport::new(1.0, 1.0).unwrap(),
            )
            .unwrap(),
            Color::WHITE,
        )
        .unwrap();
        let overflowing = ImageSprite2d::new(
            source,
            LogicalViewportRegion::new(
                LogicalScreenPosition::new(-2.0e38, 0.0),
                LogicalViewport::new(3.0e38, 1.0).unwrap(),
            )
            .unwrap(),
            Color::WHITE,
        )
        .unwrap();
        let one_pixel_clip = [2.0, -2.0, -1.0, 1.0];

        assert!(image_sprites_are_safe_for_target(
            &[safe],
            Vec2::ZERO,
            one_pixel_clip,
        ));
        assert!(!image_sprites_are_safe_for_target(
            &[overflowing],
            Vec2::ZERO,
            one_pixel_clip,
        ));
    }

    #[test]
    fn frame_budget_accepts_exact_limit_and_rejects_one_over() {
        let budget = FrameBudget::new(1, 2, 3, 4, 5, 6);
        let exact = FrameStatistics {
            pass_count: 1,
            command_count: 2,
            vertex_count: 3,
            streaming_vertex_count: 0,
            reused_vertex_count: 0,
            upload_bytes: 4,
            streaming_upload_bytes: 0,
            texture_bytes: 5,
            retained_cpu_bytes: 7,
            retained_buffer_bytes: 8,
            draw_calls: 6,
            source_counts: FrameSourceStatistics::single(FrameSourceKind::StreamingScene),
        };
        assert_eq!(validate_frame_budget(budget, exact), Ok(()));
        assert_eq!(
            validate_frame_budget(
                budget,
                FrameStatistics {
                    draw_calls: 7,
                    ..exact
                }
            ),
            Err(FrameComposerError::BudgetExceeded {
                resource: FrameBudgetResource::DrawCalls,
                limit: 6,
                actual: 7,
            })
        );
    }

    #[test]
    fn frame_statistics_group_sources_and_retained_memory_without_changing_budgets() {
        let streaming = FrameStatistics {
            pass_count: 1,
            retained_cpu_bytes: 128,
            source_counts: FrameSourceStatistics::single(FrameSourceKind::StreamingScene),
            ..FrameStatistics::default()
        };
        let glyph = FrameStatistics {
            pass_count: 1,
            retained_cpu_bytes: 256,
            retained_buffer_bytes: 512,
            source_counts: FrameSourceStatistics::single(FrameSourceKind::Glyph),
            ..FrameStatistics::default()
        };

        let combined = streaming.adding(glyph);

        assert_eq!(combined.pass_count(), 2);
        assert_eq!(combined.retained_cpu_bytes(), 384);
        assert_eq!(combined.retained_buffer_bytes(), 512);
        assert_eq!(combined.source_counts().streaming_scenes(), 1);
        assert_eq!(combined.source_counts().glyph_runs(), 1);
        assert_eq!(combined.source_counts().total(), 2);
        assert_eq!(
            validate_frame_budget(FrameBudget::new(2, 0, 0, 0, 0, 0), combined),
            Ok(())
        );
    }

    #[test]
    fn repeated_retained_references_count_each_allocation_once() {
        let first = RetainedResourceKey {
            address: 0x1000,
            class: 3,
        };
        let second = RetainedResourceKey {
            address: 0x2000,
            class: 7,
        };
        let resources = [
            RetainedResourceAccounting::new(first, 128, 256, 0),
            RetainedResourceAccounting::new(first, 128, 256, 0),
            RetainedResourceAccounting::new(second, 64, 0, 512),
        ];

        let (first_frame, missing) =
            account_new_retained_resources(&[], &resources, FrameStatistics::default());
        assert_eq!(missing, 2);
        assert_eq!(first_frame.retained_cpu_bytes(), 192);
        assert_eq!(first_frame.retained_buffer_bytes(), 256);
        assert_eq!(first_frame.texture_bytes(), 512);

        let (repeated, missing) = account_new_retained_resources(
            &[first, second],
            &resources,
            FrameStatistics::default(),
        );
        assert_eq!(missing, 0);
        assert_eq!(repeated.retained_cpu_bytes(), 0);
        assert_eq!(repeated.retained_buffer_bytes(), 0);
        assert_eq!(repeated.texture_bytes(), 0);
    }

    #[test]
    fn frame_report_exposes_single_present_contract_and_skipped_zeroes() {
        let drawn = FrameReport {
            status: RenderStatus::Drawn,
            statistics: FrameStatistics::default(),
            metrics: RendererFrameMetrics::default(),
        };
        assert_eq!(drawn.command_encoder_count(), 1);
        assert_eq!(drawn.render_pass_count(), 1);
        assert_eq!(drawn.queue_submission_count(), 1);
        assert_eq!(drawn.surface_present_count(), 1);

        let skipped = FrameReport {
            status: RenderStatus::Skipped(RendererSurfaceStatus::Occluded),
            ..drawn
        };
        assert_eq!(skipped.command_encoder_count(), 0);
        assert_eq!(skipped.render_pass_count(), 0);
        assert_eq!(skipped.queue_submission_count(), 0);
        assert_eq!(skipped.surface_present_count(), 0);

        let planned = FrameStatistics {
            upload_bytes: 128,
            streaming_upload_bytes: 96,
            vertex_count: 12,
            ..FrameStatistics::default()
        };
        let skipped_statistics = planned.without_uploads();
        assert_eq!(skipped_statistics.upload_bytes(), 0);
        assert_eq!(skipped_statistics.streaming_upload_bytes(), 0);
        assert_eq!(skipped_statistics.vertex_count(), 12);
    }

    #[test]
    fn frame_items_sort_stably_by_order_then_insertion() {
        let scene = Scene::new(Color::BLACK).unwrap();
        let camera = Camera2d::default();
        let mut items = [
            FrameItem::Scene {
                scene: &scene,
                camera,
                options: FramePassOptions::new(4),
                insertion: 0,
            },
            FrameItem::Scene {
                scene: &scene,
                camera,
                options: FramePassOptions::new(-1),
                insertion: 1,
            },
            FrameItem::Scene {
                scene: &scene,
                camera,
                options: FramePassOptions::new(4),
                insertion: 2,
            },
        ];
        items.sort_unstable_by_key(FrameItem::sort_key);
        assert_eq!(
            items.iter().map(FrameItem::sort_key).collect::<Vec<_>>(),
            vec![(-1, 1), (4, 0), (4, 2)]
        );
    }

    #[test]
    fn late_streaming_transform_rejection_rolls_back_appended_vertices() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene
            .try_line(
                Vec2::new(1.0e30, 0.0),
                Vec2::new(1.1e30, 0.0),
                1.0,
                Color::WHITE,
            )
            .unwrap();
        let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
        let camera = Camera2d::new(Vec2::ZERO, 1.0e10).unwrap();
        let uniform = CameraUniform::new_in_region(camera, viewport, Vec2::ZERO, viewport).unwrap();
        let resolved = ResolvedViewport {
            viewport,
            origin: Vec2::ZERO,
            scissor: ScissorRect {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            },
            item_clip: None,
            item_clipped_out: false,
        };
        let mut vertices = Vec::new();
        let mut ready = Vec::new();
        let mut statistics = FrameStatistics::default();
        let mut aggregate = TessellationStats::default();

        assert_eq!(
            prepare_streaming_scene_resolved(
                &scene,
                uniform,
                resolved,
                &mut vertices,
                &mut ready,
                &mut statistics,
                &mut aggregate,
            ),
            Err(FrameComposerError::Frame(
                RendererFrameError::InvalidGeometryTransform
            ))
        );
        assert!(vertices.is_empty());
        assert!(ready.is_empty());
        assert_eq!(statistics, FrameStatistics::default());
        assert_eq!(aggregate, TessellationStats::default());
    }

    #[test]
    fn streaming_screen_scene_rejects_generated_subnormal_geometry() {
        let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
        let mut scene = ScreenScene::new(Color::BLACK).unwrap();
        scene
            .try_circle(
                LogicalScreenPosition::new(32.0, 32.0),
                crate::LogicalPixels::new(f32::MIN_POSITIVE).unwrap(),
                ShapeStyle::filled(Color::WHITE),
            )
            .unwrap();
        let camera = screen_camera(viewport).unwrap();
        let uniform = CameraUniform::new_in_region(camera, viewport, Vec2::ZERO, viewport).unwrap();
        let resolved = ResolvedViewport {
            viewport,
            origin: Vec2::ZERO,
            scissor: ScissorRect {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            },
            item_clip: None,
            item_clipped_out: false,
        };
        let mut vertices = Vec::new();
        let mut ready = Vec::new();
        let mut statistics = FrameStatistics::default();
        let mut aggregate = TessellationStats::default();

        assert_eq!(
            prepare_streaming_scene_resolved(
                scene.as_scene(),
                uniform,
                resolved,
                &mut vertices,
                &mut ready,
                &mut statistics,
                &mut aggregate,
            ),
            Err(FrameComposerError::Frame(
                RendererFrameError::InvalidGeometryTransform
            ))
        );
        assert!(vertices.is_empty());
        assert!(ready.is_empty());
    }

    #[test]
    fn scissor_intersection_rejects_disjoint_rectangles() {
        let first = ScissorRect {
            x: 2,
            y: 3,
            width: 8,
            height: 9,
        };
        let overlapping = ScissorRect {
            x: 8,
            y: 10,
            width: 8,
            height: 8,
        };
        assert_eq!(
            intersect_scissors(first, overlapping),
            Some(ScissorRect {
                x: 8,
                y: 10,
                width: 2,
                height: 2,
            })
        );
        assert_eq!(
            intersect_scissors(
                first,
                ScissorRect {
                    x: 10,
                    y: 3,
                    width: 2,
                    height: 2,
                }
            ),
            None
        );
    }

    #[test]
    fn target_destination_maps_logical_region_to_clip_space() {
        let target = LogicalViewport::new(800.0, 600.0).unwrap();
        let region = LogicalViewportRegion::new(
            LogicalScreenPosition::new(200.0, 150.0),
            LogicalViewport::new(400.0, 300.0).unwrap(),
        )
        .unwrap();
        let uniform = CompositeUniform::in_region(0.5, region, target).unwrap();
        assert_eq!(uniform.opacity[0], 0.5);
        assert_eq!(uniform.destination, [0.5, 0.5, 0.0, 0.0]);
    }
}
