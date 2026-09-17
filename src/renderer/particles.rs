//! Retained particle-field updates, bounded visibility and draw submission.

use super::{
    Arc, Camera2d, CameraUniform, Color, Duration, Error, Instant, LogicalViewport,
    ParticleDrawPreparation, ParticleGpu, ParticleInstance2d, RenderReport, RenderStatus,
    RenderTarget2d, RenderTargetLoad, RendererFrameError, RendererSurfaceStatus, TessellationStats,
    WgpuRenderer, allocate_particle_staging, buffer_capacity_fits, compact_particle_instances,
    create_particle_instance_buffer, fmt, particle_budgeted_capacity, particle_idle_statistics,
    particle_instances_to_gpu, particle_statistics_with_budget, particle_update_range,
    particle_visible_index_is_selected, premultiplied_wgpu_color, prepared_scene_belongs_to,
    render_report, restore_particle_field_resources, uniformly_sampled_index,
    validate_particle_retained_capacities, validate_particle_retained_count,
    validate_particle_staging_capacity, visible_particle_count,
};

/// Counts associated with one particle-field update or draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParticleStatistics {
    pub(super) submitted: usize,
    pub(super) visibility_checked: usize,
    pub(super) visible: usize,
    pub(super) culled: usize,
    pub(super) budget_limited: usize,
    pub(super) dropped: usize,
    pub(super) rendered: usize,
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
    pub(super) max_visible_instances: usize,
    pub(super) max_retained_bytes: usize,
    pub(super) max_gpu_bytes: usize,
    pub(super) max_upload_bytes_per_frame: usize,
    pub(super) max_visibility_checks_per_frame: usize,
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

    pub(super) fn instance_limit(self) -> usize {
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
    pub(super) renderer_identity: Arc<()>,
    pub(super) instance_buffer: Arc<wgpu::Buffer>,
    pub(super) instances: Vec<ParticleGpu>,
    pub(super) visible_instances: Vec<ParticleGpu>,
    pub(super) instance_capacity: usize,
    pub(super) budget: ParticleRenderBudget,
    pub(super) statistics: ParticleStatistics,
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

impl WgpuRenderer {
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

    pub(super) fn prepare_particle_draw(
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

    pub(super) fn upload_particle_draw(
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

    pub(super) fn validate_particle_field(
        &self,
        field: &ParticleField2d,
    ) -> Result<(), ParticleFieldError> {
        prepared_scene_belongs_to(&self.renderer_identity, &field.renderer_identity)
            .then_some(())
            .ok_or(ParticleFieldError::RendererMismatch)
    }
}
