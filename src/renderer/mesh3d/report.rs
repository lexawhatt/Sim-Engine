//! Public retained 3D errors and submission/recovery diagnostics.

use super::*;

/// Resource creation or ownership failure for retained 3D rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mesh3dResourceError {
    /// Texture creation, restoration or material validation failed.
    Texture(Texture3dError),
    /// A retained UV is subnormal and outside the portable shader envelope.
    NonPortableTextureCoordinate,
    /// Explicit mesh upload limits must be nonzero.
    InvalidBudget,
    /// The replaced resource belongs to another logical renderer generation.
    RendererMismatch,
    /// Upload limits rejected bytes before caller-scale allocation.
    BudgetExceeded {
        /// Rejected byte category.
        resource: Mesh3dUploadBudgetResource,
        /// Configured maximum.
        limit: usize,
        /// Required bytes.
        actual: usize,
    },
    /// Vertex, index, or edge data exceeds the active device's buffer-size limit.
    CapacityTooLarge,
    /// At least one retained model-space vertex is outside the portable shader envelope.
    NonPortableVertex,
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
            Self::Texture(error) => write!(formatter, "{error}"),
            Self::NonPortableTextureCoordinate => write!(
                formatter,
                "3D texture coordinate is outside the portable shader envelope"
            ),
            Self::InvalidBudget => write!(formatter, "3D mesh upload limits must be nonzero"),
            Self::RendererMismatch => {
                write!(formatter, "3D mesh belongs to another renderer generation")
            }
            Self::BudgetExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "3D mesh {resource:?} budget exceeded: {actual} > {limit}"
            ),
            Self::CapacityTooLarge => write!(formatter, "3D mesh exceeds GPU buffer limits"),
            Self::NonPortableVertex => write!(
                formatter,
                "3D mesh contains a vertex outside the portable shader envelope"
            ),
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
    /// Transient Blend ordering capacity exceeded its explicit byte ceiling.
    SortingBudgetExceeded {
        /// Configured sorting-array byte limit.
        limit: usize,
        /// Required or actually reserved bytes.
        actual: usize,
    },
    /// A mesh or target belongs to another renderer/device identity.
    RendererMismatch,
    /// Model/camera inputs or their arithmetic leave the portable GPU envelope.
    InvalidGeometryTransform,
    /// A surface triangle has clipping topology or a projected orientation
    /// that is not stable across portable shader arithmetic.
    UnportableSurfaceTopology,
    /// The visible per-frame object buffers exceed a GPU or host capacity.
    InstanceCapacityTooLarge,
    /// Camera projection aspect does not match the target logical viewport.
    CameraTargetAspectMismatch,
    /// Edge clipping, projection, or style arithmetic is not portable for this target.
    InvalidEdgeProjection,
    /// A visible object's transform, surface or edge proof failed. The stable
    /// handle identifies that exact object even when several share one mesh.
    ObjectFailure {
        /// Complete scene-provenance-bearing object handle.
        object_id: Object3dId,
        /// Source-context or object-level numerical validation failure.
        reason: Mesh3dObjectError,
    },
    /// Generated clipping topology exceeds a caller budget or device capacity.
    GeneratedGeometryCapacityTooLarge,
    /// Exact retained plus generated surface submissions exceed the caller limit.
    SurfaceTriangleBudgetExceeded {
        /// Configured maximum surface submissions for this frame.
        limit: usize,
        /// Submitted count reached by authoritative preflight before rejection.
        actual: usize,
    },
}

/// Instance-local numerical rejection category from authoritative 3D preflight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mesh3dObjectError {
    /// Model/camera arithmetic cannot be represented portably.
    InvalidGeometryTransform,
    /// Clipped surface topology or projected orientation remains ambiguous.
    UnportableSurfaceTopology,
    /// Display-edge clipping or extrusion cannot be represented portably.
    InvalidEdgeProjection,
    /// A source surface triangle failed authoritative validation. The index is
    /// zero-based in the source mesh, not in a generated clipping fan.
    SurfaceTriangle {
        /// Original index into `Mesh3d::triangle_indices().chunks_exact(3)`.
        triangle_index: usize,
        /// Specific arithmetic or topology proof that failed.
        reason: Mesh3dSurfaceError,
    },
    /// A source vertex not referenced by any surface triangle failed validation.
    Vertex {
        /// Zero-based position in the source mesh vertex array.
        vertex_index: usize,
        /// Specific arithmetic or clip-classification failure.
        reason: Mesh3dSurfaceError,
    },
}

impl Mesh3dRenderError {
    pub(super) fn for_object(self, object_id: Object3dId) -> Self {
        let reason = match self {
            Self::InvalidGeometryTransform => Mesh3dObjectError::InvalidGeometryTransform,
            Self::UnportableSurfaceTopology => Mesh3dObjectError::UnportableSurfaceTopology,
            Self::InvalidEdgeProjection => Mesh3dObjectError::InvalidEdgeProjection,
            _ => return self,
        };
        Self::ObjectFailure { object_id, reason }
    }

    /// Returns an offending object when the failure is uniquely instance-local.
    /// Ownership, camera/target mismatch and aggregate capacity errors return
    /// `None`; they must not be attributed to an arbitrary visible object.
    pub const fn object_id(self) -> Option<Object3dId> {
        match self {
            Self::ObjectFailure { object_id, .. } => Some(object_id),
            _ => None,
        }
    }

    /// Returns the original source triangle index for a surface-local failure.
    /// Unused-vertex, edge, scene capacity and resource ownership failures have
    /// no source triangle and return `None`.
    pub const fn source_triangle_index(self) -> Option<usize> {
        match self {
            Self::ObjectFailure {
                reason: Mesh3dObjectError::SurfaceTriangle { triangle_index, .. },
                ..
            } => Some(triangle_index),
            _ => None,
        }
    }

    /// Returns the source vertex index for failures without a surface triangle.
    /// A referenced vertex is attributed to its first source triangle instead.
    pub const fn source_vertex_index(self) -> Option<usize> {
        match self {
            Self::ObjectFailure {
                reason: Mesh3dObjectError::Vertex { vertex_index, .. },
                ..
            } => Some(vertex_index),
            _ => None,
        }
    }

    /// Returns the specific source-surface or vertex rejection reason, when available.
    pub const fn surface_reason(self) -> Option<Mesh3dSurfaceError> {
        match self {
            Self::ObjectFailure {
                reason:
                    Mesh3dObjectError::SurfaceTriangle { reason, .. }
                    | Mesh3dObjectError::Vertex { reason, .. },
                ..
            } => Some(reason),
            _ => None,
        }
    }
}

impl fmt::Display for Mesh3dRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SortingBudgetExceeded { limit, actual } => write!(
                formatter,
                "3D sorting storage {actual} exceeds {limit} bytes"
            ),
            Self::RendererMismatch => {
                write!(formatter, "3D scene resource belongs to another renderer")
            }
            Self::InvalidGeometryTransform => {
                write!(
                    formatter,
                    "3D transform is outside the portable GPU geometry envelope"
                )
            }
            Self::UnportableSurfaceTopology => write!(
                formatter,
                "3D surface clipping or projected triangle topology is outside the portable GPU envelope"
            ),
            Self::InstanceCapacityTooLarge => write!(
                formatter,
                "visible 3D objects exceed per-frame GPU or host buffer capacity"
            ),
            Self::CameraTargetAspectMismatch => write!(
                formatter,
                "3D camera aspect does not match the target logical viewport"
            ),
            Self::InvalidEdgeProjection => write!(
                formatter,
                "3D edge clipping or screen-space arithmetic is outside the portable GPU envelope"
            ),
            Self::ObjectFailure { object_id, reason } => write!(
                formatter,
                "3D object {} failed preflight: {reason:?}",
                object_id.get()
            ),
            Self::GeneratedGeometryCapacityTooLarge => write!(
                formatter,
                "generated 3D surface topology exceeds frame or device capacity"
            ),
            Self::SurfaceTriangleBudgetExceeded { limit, actual } => write!(
                formatter,
                "3D surface submissions {actual} exceed frame triangle limit {limit}"
            ),
        }
    }
}

impl Error for Mesh3dRenderError {}

/// CPU-side result of one depth-tested offscreen 3D draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mesh3dRenderReport {
    pub(super) object_count: usize,
    pub(super) triangle_count: usize,
    pub(super) edge_count: usize,
    pub(super) render_pass_count: usize,
    pub(super) draw_call_count: usize,
    pub(super) upload: Duration,
    pub(super) preflight_duration: Duration,
    pub(super) staging_upload_duration: Duration,
    pub(super) encode_submit: Duration,
    pub(super) preflight: Mesh3dPreflightReport,
    pub(super) gpu_timing_id: Option<GpuTimingId>,
    pub(super) uploaded_bytes: usize,
    pub(super) upload_calls: usize,
    pub(super) buffer_allocation_count: usize,
    pub(super) retained_frame_cpu_bytes: usize,
    pub(super) retained_frame_gpu_bytes: usize,
    pub(super) staging_capacity_bytes: usize,
    pub(super) peak_frame_cpu_bytes: usize,
    pub(super) peak_frame_gpu_bytes: usize,
}

impl Mesh3dRenderReport {
    /// Returns exact clipping/generated-work counts from authoritative preflight.
    pub const fn preflight(self) -> Mesh3dPreflightReport {
        self.preflight
    }
    /// Returns independently transformed visible objects submitted for drawing.
    ///
    /// Outside objects count unless optional Native surface culling proves and
    /// omits them. See `preflight()` for separate culled-object and CPU clipping
    /// counts. This is not a visible-fragment or host-visibility count.
    pub const fn object_count(self) -> usize {
        self.object_count
    }

    /// Returns indexed and generated triangles submitted across every object.
    pub const fn triangle_count(self) -> usize {
        self.triangle_count
    }

    /// Returns explicit mathematical edges submitted for depth classification.
    pub const fn edge_count(self) -> usize {
        self.edge_count
    }

    /// Returns GPU render passes encoded for the retained scene.
    ///
    /// A successful draw always encodes one pass because clearing the color
    /// and depth attachments is observable even when the scene is empty.
    pub const fn render_pass_count(self) -> usize {
        self.render_pass_count
    }

    /// Returns surface, hidden-edge, and visible-edge draw calls actually
    /// encoded for the retained scene.
    pub const fn draw_call_count(self) -> usize {
        self.draw_call_count
    }

    /// Returns CPU time spent validating, staging, growing reusable buffers,
    /// and enqueuing camera/instance uploads.
    pub const fn upload(self) -> Duration {
        self.upload
    }

    /// CPU time in authoritative source validation, clipping and draw ordering.
    pub const fn preflight_duration(self) -> Duration {
        self.preflight_duration
    }

    /// CPU time after preflight spent preparing/growing staging and GPU buffers
    /// and enqueueing writes. Together with preflight this forms `upload()`;
    /// none of these CPU measurements represents GPU execution time.
    pub const fn staging_upload_duration(self) -> Duration {
        self.staging_upload_duration
    }

    /// Correlation key for optional asynchronous GPU pass timing. `None` means
    /// disabled/unavailable timing or a full ring, not a zero-duration pass.
    pub const fn gpu_timing_id(self) -> Option<GpuTimingId> {
        self.gpu_timing_id
    }

    /// Actual camera, instance, edge-uniform and generated-stream bytes written
    /// for this draw. Retained topology/texture updates and diagnostic timestamp
    /// resolution/readback are separate operations and are excluded.
    /// Exact unchanged camera/instance/edge bytes are not rewritten. A zero
    /// count still performs full preflight, target clearing and submission.
    pub const fn uploaded_bytes(self) -> usize {
        self.uploaded_bytes
    }

    /// Number of nonempty queue buffer writes corresponding to `uploaded_bytes`.
    pub const fn upload_calls(self) -> usize {
        self.upload_calls
    }

    /// Reusable frame GPU buffers newly allocated by this draw. Excludes renderer
    /// initialization, immutable scene resources, bind groups and driver internals.
    pub const fn buffer_allocation_count(self) -> usize {
        self.buffer_allocation_count
    }

    /// Retained CPU instance, edge-uniform, object, generated-stream and sorting
    /// array capacity after the draw, including idle high-water storage. Excludes
    /// scene sources, dynamic-update scratch and backend memory.
    pub const fn retained_frame_cpu_bytes(self) -> usize {
        self.retained_frame_cpu_bytes
    }

    /// Nominal reusable camera/instance/edge/generated GPU buffer bytes after this
    /// draw. Render targets, scene resources and timestamp diagnostics are excluded.
    pub const fn retained_frame_gpu_bytes(self) -> usize {
        self.retained_frame_gpu_bytes
    }

    /// Live CPU frame-array capacities during encoding. All arrays are reusable,
    /// so this equals `retained_frame_cpu_bytes` after a successful draw.
    /// Fixed-size numerical proof stack storage is not included.
    pub const fn staging_capacity_bytes(self) -> usize {
        self.staging_capacity_bytes
    }

    /// Conservative upper bound on old/new frame-array capacity during preparation.
    /// Reused arrays count once; capacity changes count both old and new storage,
    /// even if the allocator could grow in place. This is not measured process RSS.
    /// Excludes caller data, fixed-size numerical proof stack storage, backend
    /// allocations and allocator metadata.
    pub const fn peak_frame_cpu_bytes(self) -> usize {
        self.peak_frame_cpu_bytes
    }

    /// Old reusable GPU buffers plus newly allocated replacements for this draw;
    /// driver retirement/in-flight allocations are not inferred from this count.
    pub const fn peak_frame_gpu_bytes(self) -> usize {
        self.peak_frame_gpu_bytes
    }

    /// Returns CPU time spent encoding and submitting the pass.
    pub const fn encode_submit(self) -> Duration {
        self.encode_submit
    }
}

/// Result of atomically migrating a retained 3D scene to the active renderer device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scene3dRestoreReport {
    pub(super) object_count: usize,
    pub(super) migrated_object_count: usize,
    pub(super) restored_mesh_count: usize,
    pub(super) restored_gpu_bytes: usize,
    pub(super) restored_texture_count: usize,
    pub(super) restored_texture_bytes: usize,
}

impl Scene3dRestoreReport {
    /// Returns distinct shared textures recreated on the active device.
    pub const fn restored_texture_count(self) -> usize {
        self.restored_texture_count
    }
    /// Returns nominal texel bytes recreated across distinct shared textures.
    pub const fn restored_texture_bytes(self) -> usize {
        self.restored_texture_bytes
    }
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
