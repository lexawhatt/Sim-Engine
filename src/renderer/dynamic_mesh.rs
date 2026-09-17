//! Mutable triangle-mesh creation, updates, recovery and drawing.

use super::{
    Arc, Camera2d, Color, Duration, DynamicGpu, Error, GeometryExtents, GeometryValidationCache,
    GeometryValidationSource, Instant, PreparedDrawBatch, RenderReport, RenderStatus,
    RendererFrameError, TessellationStats, Vec2, WgpuRenderer, buffer_capacity_fits,
    create_dynamic_vertex_buffer, dynamic_mesh_bytes, dynamic_vertex_capacity,
    dynamic_vertices_to_gpu, fmt, prepared_scene_belongs_to, replace_dynamic_mesh_resources,
    restore_dynamic_mesh_resources, submit_pending_uploads, validate_dynamic_mesh_budget,
    validate_dynamic_retained_capacity,
};

/// One world-space vertex in a dynamic triangle-list mesh.
///
/// Dynamic meshes use plain filled triangles. Positions are measured in world
/// units, `depth` uses the active camera projection, and colors are linear RGBA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicVertex2d {
    pub(super) world_position: Vec2,
    pub(super) depth: f32,
    pub(super) color: Color,
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
    pub(super) max_vertices: usize,
    pub(super) max_retained_bytes: usize,
    pub(super) max_upload_bytes: usize,
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
/// Its bounded camera-validation cache is cleared by every successful mutation.
pub struct DynamicMesh2d {
    pub(super) renderer_identity: Arc<()>,
    pub(super) vertex_buffer: Arc<wgpu::Buffer>,
    pub(super) vertices: Vec<DynamicGpu>,
    pub(super) vertex_capacity: usize,
    pub(super) geometry_extents: GeometryExtents,
    pub(super) geometry_validation_cache: GeometryValidationCache,
    pub(super) budget: Option<DynamicMeshBudget>,
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
    pub(super) vertex_count: usize,
    pub(super) upload: Duration,
    pub(super) reallocated: bool,
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

impl WgpuRenderer {
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
            geometry_validation_cache: GeometryValidationCache::default(),
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
            mesh.geometry_validation_cache.clear();
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
            Some(&mesh.geometry_validation_cache),
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

    pub(super) fn validate_dynamic_mesh(
        &self,
        mesh: &DynamicMesh2d,
    ) -> Result<(), DynamicMeshError> {
        prepared_scene_belongs_to(&self.renderer_identity, &mesh.renderer_identity)
            .then_some(())
            .ok_or(DynamicMeshError::RendererMismatch)
    }
}
