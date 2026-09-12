use super::*;

/// Limits one scene-owned mesh update, including reserved buffers and overlap.
///
/// GPU reserve grows exactly to the larger of the current capacity, incoming
/// live length and requested minimum. No geometric over-allocation is hidden.
/// UV capacity follows vertex capacity while the incoming source has UVs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicMesh3dBudget {
    pub(super) upload: Mesh3dUploadBudget,
    pub(super) minimum_vertices: usize,
    pub(super) minimum_indices: usize,
    pub(super) minimum_edges: usize,
    pub(super) peak_recovery: usize,
    pub(super) peak_gpu: usize,
    pub(super) peak_staging: usize,
}

impl DynamicMesh3dBudget {
    /// Sets final source, GPU reserve and reusable scratch limits. Default peak
    /// limits are twice each corresponding final limit (saturating at usize::MAX),
    /// including old/new overlap.
    pub const fn new(upload: Mesh3dUploadBudget) -> Self {
        Self {
            upload,
            minimum_vertices: 0,
            minimum_indices: 0,
            minimum_edges: 0,
            peak_recovery: upload.max_recovery_bytes().saturating_mul(2),
            peak_gpu: upload.max_gpu_bytes().saturating_mul(2),
            peak_staging: upload.max_staging_bytes().saturating_mul(2),
        }
    }

    /// Requests minimum vertex, index and display-edge capacities, not live
    /// counts. Device limits and checked byte arithmetic are validated on update.
    pub const fn with_minimum_capacity(
        mut self,
        vertices: usize,
        indices: usize,
        edges: usize,
    ) -> Self {
        self.minimum_vertices = vertices;
        self.minimum_indices = indices;
        self.minimum_edges = edges;
        self
    }

    /// Sets exact peak source, GPU and conversion-scratch byte limits. Source/GPU
    /// peaks cover the outgoing and incoming revision once, not unrelated host
    /// snapshots or opaque in-flight driver memory. Zero prohibits that peak.
    pub const fn with_peak_limits(mut self, recovery: usize, gpu: usize, staging: usize) -> Self {
        self.peak_recovery = recovery;
        self.peak_gpu = gpu;
        self.peak_staging = staging;
        self
    }

    /// Final source, allocated GPU capacity and retained conversion-scratch ceilings.
    pub const fn upload_budget(self) -> Mesh3dUploadBudget {
        self.upload
    }
    /// Maximum operation-local old-plus-new source capacity.
    pub const fn max_peak_recovery_bytes(self) -> usize {
        self.peak_recovery
    }
    /// Maximum operation-local old-plus-new GPU buffer capacity.
    pub const fn max_peak_gpu_bytes(self) -> usize {
        self.peak_gpu
    }
    /// Maximum old scratch plus temporarily reserved replacement scratch capacity.
    pub const fn max_peak_staging_bytes(self) -> usize {
        self.peak_staging
    }
}

impl Default for DynamicMesh3dBudget {
    fn default() -> Self {
        Self::new(Mesh3dUploadBudget::default())
    }
}

/// Operation-local overlap category that exceeded the dynamic update budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DynamicMesh3dBudgetResource {
    /// Outgoing and incoming CPU source topology capacity.
    PeakRecoveryBytes,
    /// Outgoing and incoming GPU bundles, deduplicating the reused bundle.
    PeakGpuBytes,
    /// Retained conversion scratch and temporary replacement arrays.
    PeakStagingBytes,
}

/// Synchronous update failure; the object's drawable revision remains unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DynamicMesh3dError {
    /// Geometry ownership, numerical, allocation or device-limit failure.
    Resource(Mesh3dResourceError),
    /// Object lookup, style or deduplicated scene capacity failure.
    Scene(Scene3dError),
    /// A bounded transient overlap exceeds its configured limit.
    BudgetExceeded {
        /// Resource whose operation-local peak exceeded the ceiling.
        resource: DynamicMesh3dBudgetResource,
        /// Configured byte ceiling.
        limit: usize,
        /// Exact capacity bytes needed by this update.
        actual: usize,
    },
}

impl From<Mesh3dResourceError> for DynamicMesh3dError {
    fn from(error: Mesh3dResourceError) -> Self {
        Self::Resource(error)
    }
}
impl From<Scene3dError> for DynamicMesh3dError {
    fn from(error: Scene3dError) -> Self {
        Self::Scene(error)
    }
}
impl fmt::Display for DynamicMesh3dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resource(error) => write!(formatter, "dynamic 3D mesh: {error}"),
            Self::Scene(error) => write!(formatter, "dynamic 3D scene: {error}"),
            Self::BudgetExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "dynamic 3D mesh {resource:?} peak {actual} exceeds {limit}"
            ),
        }
    }
}
impl Error for DynamicMesh3dError {}

/// Exact engine-owned work/capacity counters for a committed full source update.
///
/// An identical source is still uploaded; no equality/no-op optimization is
/// promised. Driver staging allocations and caller-side mesh construction are
/// outside these counters. Every GPU allocation is one newly created buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicMesh3dUpdateReport {
    pub(super) uploaded_bytes: usize,
    pub(super) upload_calls: usize,
    pub(super) gpu_allocations: usize,
    pub(super) detached_aliases: bool,
    pub(super) grew_capacity: bool,
    pub(super) capacity_gpu_bytes: usize,
    pub(super) scratch_capacity_bytes: usize,
    pub(super) scratch_reallocations: usize,
    pub(super) peak_staging_bytes: usize,
    pub(super) peak_gpu_bytes: usize,
    pub(super) peak_recovery_bytes: usize,
    pub(super) scene_statistics: Scene3dStatistics,
}

impl DynamicMesh3dUpdateReport {
    /// Bytes enqueued for live vertices, indices, UVs and display edges.
    pub const fn uploaded_bytes(self) -> usize {
        self.uploaded_bytes
    }
    /// Actual nonempty queue buffer-write calls for this update.
    pub const fn upload_calls(self) -> usize {
        self.upload_calls
    }
    /// Newly created GPU buffers; zero proves engine buffer reuse.
    pub const fn gpu_allocation_count(self) -> usize {
        self.gpu_allocations
    }
    /// Whether every previous GPU buffer was reused in this update.
    pub const fn reused_buffers(self) -> bool {
        self.gpu_allocations == 0
    }
    /// Whether immutable aliases required detaching a new complete GPU bundle.
    pub const fn detached_aliases(self) -> bool {
        self.detached_aliases
    }
    /// Whether any reserved buffer capacity increased.
    pub const fn grew_capacity(self) -> bool {
        self.grew_capacity
    }
    /// Bytes representing the live source, excluding unused GPU reserve.
    pub const fn live_gpu_bytes(self) -> usize {
        self.uploaded_bytes
    }
    /// Final GPU bundle capacity, including unused reserved bytes.
    pub const fn capacity_gpu_bytes(self) -> usize {
        self.capacity_gpu_bytes
    }
    /// Retained CPU conversion-array capacity shared by subsequent renderer updates.
    pub const fn scratch_capacity_bytes(self) -> usize {
        self.scratch_capacity_bytes
    }
    /// Number of conversion arrays newly reserved by the engine in this update.
    pub const fn scratch_reallocations(self) -> usize {
        self.scratch_reallocations
    }
    /// Maximum old scratch plus temporary replacement scratch capacity.
    pub const fn peak_staging_bytes(self) -> usize {
        self.peak_staging_bytes
    }
    /// Outgoing plus incoming GPU bytes; a reused allocation is counted once.
    pub const fn peak_gpu_bytes(self) -> usize {
        self.peak_gpu_bytes
    }
    /// Outgoing plus incoming source capacities; identical source storage counts once.
    pub const fn peak_recovery_bytes(self) -> usize {
        self.peak_recovery_bytes
    }
    /// Exact deduplicated scene-owned statistics after commit.
    pub const fn scene_statistics(self) -> Scene3dStatistics {
        self.scene_statistics
    }
}
