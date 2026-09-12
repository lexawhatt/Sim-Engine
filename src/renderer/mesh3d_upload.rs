//! Explicit immutable mesh replacement budgets and peak accounting.

use super::*;

/// Byte category bounded before immutable mesh upload or replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mesh3dUploadBudgetResource {
    /// Retained source-topology capacities.
    RecoveryBytes,
    /// Vertex/UV/index/display-edge GPU buffer bytes; excludes material texels.
    GpuBytes,
    /// Temporary CPU conversion-array bytes.
    StagingBytes,
}

/// Per-upload limits checked before caller-scale staging or GPU allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mesh3dUploadBudget {
    max_recovery_bytes: usize,
    max_gpu_bytes: usize,
    max_staging_bytes: usize,
}

impl Mesh3dUploadBudget {
    /// Creates nonzero source-capacity, GPU-buffer and conversion-staging limits.
    pub fn new(
        max_recovery_bytes: usize,
        max_gpu_bytes: usize,
        max_staging_bytes: usize,
    ) -> Result<Self, Mesh3dResourceError> {
        if [max_recovery_bytes, max_gpu_bytes, max_staging_bytes].contains(&0) {
            return Err(Mesh3dResourceError::InvalidBudget);
        }
        Ok(Self {
            max_recovery_bytes,
            max_gpu_bytes,
            max_staging_bytes,
        })
    }
    /// Maximum retained core topology capacity, excluding Arc control blocks.
    pub const fn max_recovery_bytes(self) -> usize {
        self.max_recovery_bytes
    }
    /// Maximum vertex/UV/index/display-edge bytes for the accepted revision.
    pub const fn max_gpu_bytes(self) -> usize {
        self.max_gpu_bytes
    }
    /// Maximum temporary CPU vertex/UV/edge conversion bytes.
    pub const fn max_staging_bytes(self) -> usize {
        self.max_staging_bytes
    }
}

impl Default for Mesh3dUploadBudget {
    fn default() -> Self {
        Self {
            max_recovery_bytes: 256 * 1024 * 1024,
            max_gpu_bytes: 256 * 1024 * 1024,
            max_staging_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Nominal buffer/source accounting for an accepted immutable replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mesh3dUploadReport {
    recovery_bytes: usize,
    uploaded_bytes: usize,
    staging_bytes: usize,
    peak_recovery_bytes: usize,
    peak_gpu_bytes: usize,
}

impl Mesh3dUploadReport {
    /// CPU source topology capacity of the accepted revision.
    pub const fn recovery_bytes(self) -> usize {
        self.recovery_bytes
    }
    /// Bytes uploaded to new vertex/UV/index/display-edge buffers.
    pub const fn uploaded_bytes(self) -> usize {
        self.uploaded_bytes
    }
    /// Temporary CPU conversion-array bytes, excluding backend upload staging.
    pub const fn staging_bytes(self) -> usize {
        self.staging_bytes
    }
    /// Old plus new unique source capacities during replacement.
    pub const fn peak_recovery_bytes(self) -> usize {
        self.peak_recovery_bytes
    }
    /// Old plus new GPU buffer bytes, not a promise of immediate driver release.
    pub const fn peak_gpu_bytes(self) -> usize {
        self.peak_gpu_bytes
    }
    /// Immutable replacement always allocates new buffers, even at equal size.
    pub const fn replaced_buffers(self) -> bool {
        true
    }
}

impl WgpuRenderer {
    /// Creates an immutable revision with explicit pre-allocation byte limits.
    pub fn create_mesh3d_with_budget(
        &self,
        source: Mesh3d,
        budget: Mesh3dUploadBudget,
    ) -> Result<RetainedMesh3d, Mesh3dResourceError> {
        let prepared = prepare_with_budget(&self.device, source, budget)?;
        Ok(upload_prepared_retained_mesh(
            &self.device,
            &self.queue,
            Arc::clone(&self.renderer_identity),
            prepared,
        ))
    }

    /// Atomically replaces this retained handle with a newly uploaded revision.
    ///
    /// Clones and scene objects referencing the previous revision are unchanged;
    /// use [`Scene3d::set_mesh`] to rebind selected live objects. This is immutable
    /// replacement, not a stable-capacity in-place upload. Validation/allocation
    /// errors preserve the old handle; asynchronous device failure is separate.
    /// In-flight work keeps its old references. Nominal transient GPU/source
    /// overlap is at most the old revision plus the incoming configured limits.
    pub fn replace_mesh3d(
        &self,
        mesh: &mut RetainedMesh3d,
        source: Mesh3d,
        budget: Mesh3dUploadBudget,
    ) -> Result<Mesh3dUploadReport, Mesh3dResourceError> {
        replace_resources(
            &self.device,
            &self.queue,
            &self.renderer_identity,
            mesh,
            source,
            budget,
        )
    }
}

pub(super) fn prepare_with_budget(
    device: &wgpu::Device,
    source: Mesh3d,
    budget: Mesh3dUploadBudget,
) -> Result<PreparedRetainedMeshUpload, Mesh3dResourceError> {
    let layout = preflight_mesh3d_source(&source, device.limits().max_buffer_size)?;
    for (resource, limit, actual) in [
        (
            Mesh3dUploadBudgetResource::RecoveryBytes,
            budget.max_recovery_bytes,
            source.recovery_memory_bytes(),
        ),
        (
            Mesh3dUploadBudgetResource::GpuBytes,
            budget.max_gpu_bytes,
            layout.total_bytes as usize,
        ),
        (
            Mesh3dUploadBudgetResource::StagingBytes,
            budget.max_staging_bytes,
            layout
                .vertex_bytes
                .saturating_add(layout.edge_bytes)
                .saturating_add(layout.texture_coordinate_bytes) as usize,
        ),
    ] {
        if actual > limit {
            return Err(Mesh3dResourceError::BudgetExceeded {
                resource,
                limit,
                actual,
            });
        }
    }
    let mut prepared = prepare_retained_mesh_upload(device, source)?;
    prepared.budget = budget;
    Ok(prepared)
}

pub(super) fn prepare_restoration(
    device: &wgpu::Device,
    source: &RetainedMesh3d,
) -> Result<PreparedRetainedMeshUpload, Mesh3dResourceError> {
    validate_allocation(device, source.allocation, source.budget)?;
    let mut prepared = prepare_with_budget(device, source.source.clone(), source.budget)?;
    prepared.allocation = source.allocation;
    Ok(prepared)
}

pub(super) fn validate_allocation(
    device: &wgpu::Device,
    allocation: Mesh3dUploadLayout,
    budget: Mesh3dUploadBudget,
) -> Result<(), Mesh3dResourceError> {
    if [
        allocation.vertex_bytes,
        allocation.index_bytes,
        allocation.edge_bytes,
        allocation.texture_coordinate_bytes,
    ]
    .into_iter()
    .any(|bytes| bytes > device.limits().max_buffer_size)
        || usize::try_from(allocation.total_bytes).is_err()
    {
        return Err(Mesh3dResourceError::CapacityTooLarge);
    }
    if allocation.total_bytes as usize > budget.max_gpu_bytes() {
        return Err(Mesh3dResourceError::BudgetExceeded {
            resource: Mesh3dUploadBudgetResource::GpuBytes,
            limit: budget.max_gpu_bytes(),
            actual: allocation.total_bytes as usize,
        });
    }
    Ok(())
}

pub(super) fn replace_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    identity: &Arc<()>,
    mesh: &mut RetainedMesh3d,
    source: Mesh3d,
    budget: Mesh3dUploadBudget,
) -> Result<Mesh3dUploadReport, Mesh3dResourceError> {
    if !Arc::ptr_eq(identity, &mesh.renderer_identity) {
        return Err(Mesh3dResourceError::RendererMismatch);
    }
    if mesh.material().is_some()
        && (source.texture_coordinates().len() != source.vertices().len()
            || source.triangle_count() == 0)
    {
        return Err(Mesh3dResourceError::Texture(
            Texture3dError::MissingTextureCoordinates,
        ));
    }
    let prepared = prepare_with_budget(device, source, budget)?;
    let report = Mesh3dUploadReport {
        recovery_bytes: prepared.source.recovery_memory_bytes(),
        uploaded_bytes: prepared.layout.total_bytes as usize,
        staging_bytes: prepared
            .layout
            .vertex_bytes
            .saturating_add(prepared.layout.edge_bytes)
            .saturating_add(prepared.layout.texture_coordinate_bytes)
            as usize,
        peak_recovery_bytes: mesh.recovery_memory_bytes().saturating_add(
            if mesh.source.vertices().as_ptr() == prepared.source.vertices().as_ptr() {
                0
            } else {
                prepared.source.recovery_memory_bytes()
            },
        ),
        peak_gpu_bytes: mesh
            .gpu_allocation_bytes()
            .saturating_add(prepared.layout.total_bytes as usize),
    };
    let mut replacement =
        upload_prepared_retained_mesh(device, queue, Arc::clone(identity), prepared);
    replacement.material = mesh.material.clone();
    *mesh = replacement;
    Ok(report)
}
