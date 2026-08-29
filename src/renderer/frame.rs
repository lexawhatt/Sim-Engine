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
    /// Retained texture bytes referenced by images, fields, or targets.
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

    /// Returns the maximum referenced retained texture allocation bytes.
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

/// Conservative work referenced by one composed frame.
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
    draw_calls: usize,
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

    /// Returns retained texture bytes referenced by frame items.
    ///
    /// Repeated image and render-target identities are deduplicated. Scalar
    /// field and color-map work is conservatively counted per item.
    pub const fn texture_bytes(self) -> usize {
        self.texture_bytes
    }

    /// Returns the conservative number of scheduled draw calls.
    pub const fn draw_calls(self) -> usize {
        self.draw_calls
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
            draw_calls: self.draw_calls.saturating_add(other.draw_calls),
        }
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
    texture_resources: Vec<usize>,
    planned: FrameStatistics,
    next_insertion: usize,
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
            texture_resources: Vec::new(),
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
            draw_calls: statistics.estimated_draw_batches(),
        };
        self.push_item(
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
            draw_calls: statistics.estimated_draw_batches(),
        };
        self.push_item(
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
            draw_calls: scene.draw_batches.len(),
        };
        self.push_item(
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
            draw_calls: scene.scene.draw_batches.len(),
        };
        self.push_item(
            work,
            FrameItem::PreparedScreen {
                scene,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds retained dynamic triangles through their own world camera.
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
            draw_calls: usize::from(!mesh.vertices.is_empty()),
        };
        self.push_item(
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
        self.push_item(
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
                draw_calls: usize::from(candidate_count > 0),
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
        let color_map_bytes = COLOR_MAP_LUT_SIZE as usize * 4;
        self.push_item(
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<HeatmapUniform>().saturating_add(color_map_bytes),
                streaming_upload_bytes: 0,
                texture_bytes: texture
                    .gpu_allocation_bytes()
                    .saturating_add(color_map_bytes),
                draw_calls: 1,
            },
            FrameItem::Scalar {
                texture,
                color_map,
                minimum,
                value_extent,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )
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
        self.push_texture_item(
            Arc::as_ptr(&image.resource_identity) as usize,
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: image.gpu_allocation_bytes(),
                draw_calls: 1,
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
        self.push_texture_item(
            Arc::as_ptr(&image.resource_identity) as usize,
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: image.gpu_allocation_bytes(),
                draw_calls: 1,
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

    /// Adds a retained atlas batch as one instanced draw call.
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
        self.push_retained_image_batch(image, batch, sampling, options)
    }

    /// Adds one host-shaped retained glyph run as one instanced draw call.
    ///
    /// Positions are local logical pixels. Mixed font fallback is represented
    /// by multiple runs submitted at the same order; stable insertion order is
    /// preserved between them.
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
        self.push_retained_image_batch(&atlas.image, &run.batch, sampling, options)
    }

    fn push_retained_image_batch(
        &mut self,
        image: &'frame Image2d,
        batch: &'frame ImageBatch2d,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        let vertex_count = batch.sprite_count().saturating_mul(6);
        self.push_texture_item(
            Arc::as_ptr(&image.resource_identity) as usize,
            FrameStatistics {
                pass_count: 1,
                command_count: usize::from(batch.sprite_count() > 0),
                vertex_count,
                streaming_vertex_count: 0,
                reused_vertex_count: vertex_count,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: image.gpu_allocation_bytes(),
                draw_calls: usize::from(batch.sprite_count() > 0),
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
        self.push_texture_item(
            Arc::as_ptr(&target.resource_identity) as usize,
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<CompositeUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: target.allocation_bytes,
                draw_calls: 1,
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

    fn push_texture_item(
        &mut self,
        resource_identity: usize,
        mut work: FrameStatistics,
        item: FrameItem<'frame>,
    ) -> Result<(), FrameComposerError> {
        let is_new = !self.texture_resources.contains(&resource_identity);
        if !is_new {
            work.texture_bytes = 0;
        } else {
            self.texture_resources.try_reserve(1).map_err(|_| {
                FrameComposerError::AllocationFailed {
                    requested_bytes: std::mem::size_of::<usize>(),
                }
            })?;
        }
        self.push_item(work, item)?;
        if is_new {
            self.texture_resources.push(resource_identity);
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
        mut items,
        texture_resources: _,
        planned,
        next_insertion: _,
    } = composer;
    let frame_started_at = Instant::now();
    items.sort_unstable_by_key(FrameItem::sort_key);

    let target_viewport = renderer
        .logical_viewport()
        .map_err(|_| RendererFrameError::InvalidViewport)?;
    let tessellation_started_at = Instant::now();
    let mut streaming_vertices = Vec::new();
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

    for item in items {
        match item {
            FrameItem::Scene {
                scene,
                camera,
                options,
                ..
            } => prepare_streaming_scene(
                scene,
                camera,
                options,
                renderer,
                target_viewport,
                &mut streaming_vertices,
                &mut ready,
                &mut statistics,
                &mut tessellation_stats,
            )?,
            FrameItem::ScreenScene { scene, options, .. } => {
                let viewport = resolve_viewport(renderer, target_viewport, options)?;
                let camera = screen_camera(viewport.viewport)
                    .map_err(|_| RendererFrameError::InvalidGeometryTransform)?;
                prepare_streaming_scene_resolved(
                    scene.as_scene(),
                    camera,
                    viewport,
                    target_viewport,
                    &mut streaming_vertices,
                    &mut ready,
                    &mut statistics,
                    &mut tessellation_stats,
                )?;
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
                let batches = (!mesh.vertices.is_empty())
                    .then_some(PreparedDrawBatch {
                        vertex_range: 0..mesh.vertices.len() as u32,
                        screen_clip: None,
                    })
                    .into_iter()
                    .collect::<Vec<_>>();
                prepare_retained_geometry(
                    ReadySource::Dynamic(&mesh.vertex_buffer),
                    mesh.vertices.len(),
                    mesh.geometry_extents,
                    &batches,
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
                value_extent,
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
                statistics = statistics.adding(FrameStatistics {
                    pass_count: 1,
                    command_count: 1,
                    vertex_count: 6,
                    streaming_vertex_count: 0,
                    reused_vertex_count: 0,
                    upload_bytes: std::mem::size_of::<HeatmapUniform>()
                        .saturating_add(color_map_bytes),
                    streaming_upload_bytes: 0,
                    texture_bytes: texture
                        .gpu_allocation_bytes()
                        .saturating_add(color_map_bytes),
                    draw_calls: 1,
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
                    draw_calls: 1,
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
                    if !clip.is_finite() {
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
                    draw_calls: 1,
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
                    draw_calls: usize::from(batch.sprite_count() > 0),
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
                    draw_calls: 1,
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
    let tessellation = tessellation_started_at.elapsed();
    statistics.texture_bytes = planned.texture_bytes;
    validate_frame_budget(budget, statistics)?;
    renderer.ensure_vertex_capacity(streaming_vertices.len())?;

    let upload_started_at = Instant::now();
    if !streaming_vertices.is_empty() {
        renderer.queue.write_buffer(
            &renderer.vertex_buffer,
            0,
            bytemuck::cast_slice(&streaming_vertices),
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
    let mut upload = upload_started_at.elapsed();
    let mut camera_uniform_upload = Duration::ZERO;
    let mut bindings = Vec::new();
    bindings
        .try_reserve(ready.len())
        .map_err(|_| FrameComposerError::AllocationFailed {
            requested_bytes: ready
                .len()
                .saturating_mul(std::mem::size_of::<FrameBinding>()),
        })?;
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
                upload += elapsed;
            }
        }
    }
    renderer.vertices = streaming_vertices;

    let acquire_started_at = Instant::now();
    let surface_texture = match renderer.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture)
        | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
        wgpu::CurrentSurfaceTexture::Timeout => {
            set_particle_rendered(&mut ready, false);
            return Ok(frame_report(
                RenderStatus::Skipped(RendererSurfaceStatus::Timeout),
                statistics,
                tessellation,
                upload,
                camera_uniform_upload,
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
                statistics,
                tessellation,
                upload,
                camera_uniform_upload,
                acquire_started_at.elapsed(),
                Duration::ZERO,
                frame_started_at.elapsed(),
                geometry_reused,
                geometry_streamed,
                tessellation_stats,
            ));
        }
        wgpu::CurrentSurfaceTexture::Outdated => {
            renderer.resize(renderer.config.width, renderer.config.height);
            set_particle_rendered(&mut ready, false);
            return Ok(frame_report(
                RenderStatus::Skipped(RendererSurfaceStatus::Outdated),
                statistics,
                tessellation,
                upload,
                camera_uniform_upload,
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

fn set_particle_rendered(items: &mut [ReadyItem<'_>], presented: bool) {
    for item in items {
        if let ReadyItem::Particle {
            field,
            visible_count,
            viewport,
            ..
        } = item
        {
            field.statistics.rendered = if presented && !viewport.item_clipped_out {
                *visible_count
            } else {
                0
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
    let local_uniform = CameraUniform::new(camera, viewport.viewport)
        .ok_or(RendererFrameError::InvalidGeometryTransform)?;
    let target_uniform =
        CameraUniform::new_in_region(camera, viewport.viewport, viewport.origin, target_viewport)
            .ok_or(RendererFrameError::InvalidGeometryTransform)?;
    let instance_count = field.instances.len();
    let visibility_checked = instance_count.min(field.budget.max_visibility_checks_per_frame);

    let (visible_count, selected_count) = if visibility_checked < instance_count {
        field.visible_instances.clear();
        for candidate_index in 0..visibility_checked {
            let source_index =
                uniformly_sampled_index(candidate_index, instance_count, visibility_checked);
            let instance = field.instances[source_index];
            if !instance.is_safe_for(target_uniform) {
                return Err(RendererFrameError::InvalidGeometryTransform.into());
            }
            if instance.intersects_viewport(local_uniform, viewport.viewport) {
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
        )?;
        let selected_count = visible_count.min(field.budget.instance_limit());
        if selected_count == field.instances.len() {
            field.visible_instances.clear();
        } else {
            field.visible_instances.clear();
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
    field.statistics = particle_statistics_with_budget(
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
        draw_calls: usize::from(selected_count > 0),
    });
    ready.push(ReadyItem::Particle {
        field,
        visible_count: selected_count,
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
) -> Result<usize, RendererFrameError> {
    let mut visible = 0;
    for instance in instances {
        if !instance.is_safe_for(local_camera) || !instance.is_safe_for(target_camera) {
            return Err(RendererFrameError::InvalidGeometryTransform);
        }
        visible += usize::from(instance.intersects_viewport(local_camera, viewport));
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
    prepare_streaming_scene_resolved(
        scene,
        camera,
        viewport,
        target_viewport,
        streaming_vertices,
        ready,
        statistics,
        aggregate,
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_streaming_scene_resolved<'frame>(
    scene: &Scene,
    camera: Camera2d,
    viewport: ResolvedViewport,
    target_viewport: LogicalViewport,
    streaming_vertices: &mut Vec<Vertex>,
    ready: &mut Vec<ReadyItem<'frame>>,
    statistics: &mut FrameStatistics,
    aggregate: &mut TessellationStats,
) -> Result<(), FrameComposerError> {
    let mut vertices = Vec::new();
    let mut batches = Vec::new();
    let stats =
        tessellate_scene(scene, &mut vertices, &mut batches).map_err(RendererFrameError::from)?;
    let base = u32::try_from(streaming_vertices.len())
        .map_err(|_| RendererFrameError::GeometryCapacityTooLarge)?;
    streaming_vertices
        .try_reserve(vertices.len())
        .map_err(|_| FrameComposerError::AllocationFailed {
            requested_bytes: vertices.len().saturating_mul(std::mem::size_of::<Vertex>()),
        })?;
    for batch in &mut batches {
        batch.vertex_range.start = batch.vertex_range.start.saturating_add(base);
        batch.vertex_range.end = batch.vertex_range.end.saturating_add(base);
    }
    let extents = GeometryExtents::from_vertices(&vertices);
    streaming_vertices.extend_from_slice(&vertices);
    let camera_uniform =
        CameraUniform::new_in_region(camera, viewport.viewport, viewport.origin, target_viewport)
            .ok_or(RendererFrameError::InvalidGeometryTransform)?;
    if !extents.is_safe_for(camera_uniform) {
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
        draw_calls: batches.len(),
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
    if !extents.is_safe_for(camera_uniform) {
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
        draw_calls: owned_batches.len(),
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
            draw_calls: 6,
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
