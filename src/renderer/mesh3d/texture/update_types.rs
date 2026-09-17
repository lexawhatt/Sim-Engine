use super::*;

/// Explicit operation-local limits for atomic texture edits and alias detachment.
///
/// Peak recovery includes old plus candidate CPU chains. Staging limits cover
/// owned packed-patch capacity plus nominal queued upload bytes, not unobservable
/// driver allocations. GPU copy is separate from host upload. Caller input,
/// unrelated snapshots and previously submitted driver work are excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Texture3dUpdateBudget {
    pub(super) upload: usize,
    pub(super) staging: usize,
    pub(super) recovery: usize,
    pub(super) gpu: usize,
    pub(super) copy: usize,
}
impl Texture3dUpdateBudget {
    /// Sets host-upload, staging, old-plus-candidate CPU, old-plus-new GPU and
    /// GPU-copy byte limits in that order. Zero prohibits the corresponding work.
    pub const fn new(
        upload_bytes: usize,
        staging_bytes: usize,
        peak_recovery_bytes: usize,
        peak_gpu_bytes: usize,
        gpu_copy_bytes: usize,
    ) -> Self {
        Self {
            upload: upload_bytes,
            staging: staging_bytes,
            recovery: peak_recovery_bytes,
            gpu: peak_gpu_bytes,
            copy: gpu_copy_bytes,
        }
    }
    /// Maximum bytes uploaded from the host, excluding GPU-to-GPU copies.
    pub const fn max_upload_bytes(self) -> usize {
        self.upload
    }
    /// Maximum owned packed-patch capacity plus nominal queued-upload bytes.
    pub const fn max_staging_bytes(self) -> usize {
        self.staging
    }
    /// Maximum old plus candidate CPU chain capacity.
    pub const fn max_peak_recovery_bytes(self) -> usize {
        self.recovery
    }
    /// Maximum old plus new nominal GPU texture storage.
    pub const fn max_peak_gpu_bytes(self) -> usize {
        self.gpu
    }
    /// Maximum GPU-to-GPU copy bytes when aliases require detachment.
    pub const fn max_gpu_copy_bytes(self) -> usize {
        self.copy
    }
}
impl Default for Texture3dUpdateBudget {
    fn default() -> Self {
        Self::new(
            64 * 1024 * 1024,
            128 * 1024 * 1024,
            128 * 1024 * 1024,
            128 * 1024 * 1024,
            64 * 1024 * 1024,
        )
    }
}

/// The operation-local limit exceeded before a texture edit was published.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Texture3dUpdateBudgetResource {
    /// Host upload bytes: base patch plus regenerated lower mip levels.
    UploadBytes,
    /// Packed patch capacity plus nominal queued-upload bytes.
    StagingBytes,
    /// Old and candidate retained CPU chain capacity.
    PeakRecoveryBytes,
    /// Old and replacement GPU chain storage, or only old storage when reused.
    PeakGpuBytes,
    /// Old GPU chain copied to preserve immutable aliases.
    GpuCopyBytes,
}

/// Synchronous texture-update failure; existing drawable state is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Texture3dUpdateError {
    /// Texture provenance, source alpha, dimensions or host allocation failed.
    Texture(Texture3dError),
    /// Scene object lookup, accounting or resource reservation failed.
    Scene(Scene3dError),
    /// Row stride is smaller than one RGBA row or required source length differs.
    InvalidSourceLayout,
    /// The selected object has no texture material.
    MissingTexture,
    /// Checked size arithmetic cannot represent the requested edit.
    CapacityTooLarge,
    /// A caller limit rejected planned or actual operation-local work.
    BudgetExceeded {
        /// Rejected accounting category.
        resource: Texture3dUpdateBudgetResource,
        /// Configured maximum byte count.
        limit: usize,
        /// Required byte count.
        actual: usize,
    },
}
impl From<Texture3dError> for Texture3dUpdateError {
    fn from(error: Texture3dError) -> Self {
        Self::Texture(error)
    }
}
impl From<Scene3dError> for Texture3dUpdateError {
    fn from(error: Scene3dError) -> Self {
        Self::Scene(error)
    }
}
impl fmt::Display for Texture3dUpdateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Texture(error) => write!(formatter, "3D texture update: {error}"),
            Self::Scene(error) => write!(formatter, "3D texture scene update: {error}"),
            Self::InvalidSourceLayout => formatter
                .write_str("3D texture update requires an exact strided RGBA source layout"),
            Self::MissingTexture => formatter.write_str("3D object has no texture material"),
            Self::CapacityTooLarge => {
                formatter.write_str("3D texture update capacity arithmetic overflowed")
            }
            Self::BudgetExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "3D texture update {resource:?} budget exceeded: {actual} > {limit}"
            ),
        }
    }
}
impl Error for Texture3dUpdateError {}

/// Exact edit work and nominal resource storage; no GPU timing is inferred.
///
/// Every successful edit uploads its patch, including identical input. Generated
/// chains regenerate/upload all lower levels; their work is not region-local.
/// Aliased edits copy the old GPU chain before patch uploads. Queue submissions
/// and CPU durations describe this edit, not physical GPU completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Texture3dUpdateReport {
    pub(super) base: usize,
    pub(super) mips: usize,
    pub(super) uploads: usize,
    pub(super) copy: usize,
    pub(super) copies: usize,
    pub(super) allocations: usize,
    pub(super) submissions: usize,
    pub(super) recovery: usize,
    pub(super) gpu: usize,
    pub(super) staging: usize,
    pub(super) peak_recovery: usize,
    pub(super) peak_gpu: usize,
    pub(super) preparation: Duration,
    pub(super) encode_submit: Duration,
}
impl Texture3dUpdateReport {
    /// Bytes uploaded for the requested base-level rectangle, excluding source padding.
    pub const fn base_upload_bytes(self) -> usize {
        self.base
    }
    /// Bytes uploaded for complete regenerated lower mip levels.
    pub const fn mip_upload_bytes(self) -> usize {
        self.mips
    }
    /// Sum of base patch and derived mip host uploads, excluding GPU copies.
    pub const fn uploaded_bytes(self) -> usize {
        self.base + self.mips
    }
    /// Number of queue texture upload calls.
    pub const fn upload_calls(self) -> usize {
        self.uploads
    }
    /// Bytes copied on the GPU when detaching an aliased chain.
    pub const fn gpu_copy_bytes(self) -> usize {
        self.copy
    }
    /// Number of GPU texture-copy commands.
    pub const fn gpu_copy_calls(self) -> usize {
        self.copies
    }
    /// Number of new GPU texture allocations; bindings/driver metadata are excluded.
    pub const fn gpu_allocation_count(self) -> usize {
        self.allocations
    }
    /// Queue submissions: one upload flush, plus one copy submission when detached.
    pub const fn submission_count(self) -> usize {
        self.submissions
    }
    /// Whether the existing GPU texture allocation was reused.
    pub const fn reused_allocation(self) -> bool {
        self.allocations == 0
    }
    /// Whether old immutable aliases required GPU copy-on-write.
    pub const fn detached_aliases(self) -> bool {
        self.allocations != 0
    }
    /// Committed CPU chain capacity, excluding fixed storage metadata.
    pub const fn recovery_bytes(self) -> usize {
        self.recovery
    }
    /// Committed nominal all-level GPU texel bytes.
    pub const fn gpu_bytes(self) -> usize {
        self.gpu
    }
    /// Actual packed-patch capacity plus nominal queued upload bytes.
    pub const fn staging_bytes(self) -> usize {
        self.staging
    }
    /// Old plus candidate CPU chain capacity at preparation peak.
    pub const fn peak_recovery_bytes(self) -> usize {
        self.peak_recovery
    }
    /// Old plus replacement nominal GPU bytes, deduplicating a reused allocation.
    pub const fn peak_gpu_bytes(self) -> usize {
        self.peak_gpu
    }
    /// Number of output texels regenerated in complete lower mip levels.
    pub const fn regenerated_texel_count(self) -> usize {
        self.mips / 4
    }
    /// CPU validation, copying and mip generation duration; not GPU execution time.
    pub const fn preparation_cpu(self) -> Duration {
        self.preparation
    }
    /// CPU resource creation, copy encoding and queue submission duration.
    pub const fn encode_submit_cpu(self) -> Duration {
        self.encode_submit
    }
}
