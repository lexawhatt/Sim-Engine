//! Frame outcomes, timing reports and public rendering errors.

use super::{Duration, Error, SceneBudgetResource, TessellationError, TessellationStats, fmt};

/// Result of attempting to draw a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStatus {
    /// Commands were submitted successfully. Surface paths also requested
    /// presentation; offscreen paths completed their target submission only.
    Drawn,
    /// The frame was skipped because the window surface was temporarily unavailable.
    Skipped(RendererSurfaceStatus),
}

/// CPU-side durations measured while preparing and submitting one renderer frame.
///
/// These values do not measure GPU execution or the monitor scanout timestamp.
/// FIFO presentation back-pressure normally appears in `surface_acquire` because
/// `wgpu` blocks frame acquisition when the presentation queue is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RendererFrameMetrics {
    pub(super) tessellation: Duration,
    pub(super) upload: Duration,
    pub(super) camera_uniform_upload: Duration,
    pub(super) surface_acquire: Duration,
    pub(super) encode_submit_present: Duration,
    pub(super) total_cpu: Duration,
    pub(super) geometry_reused: bool,
    pub(super) geometry_streamed: bool,
    pub(super) tessellation_stats: TessellationStats,
}

impl RendererFrameMetrics {
    /// Returns CPU time spent preparing visual work before GPU uploads.
    ///
    /// For an ordinary streaming scene this is validation and tessellation.
    /// A heterogeneous [`FrameComposer`](crate::FrameComposer) frame also
    /// includes viewport and source validation, particle visibility selection,
    /// scalar LUT generation, and preparation of retained image/glyph draws.
    pub fn tessellation(self) -> Duration {
        self.tessellation
    }

    /// Returns CPU time spent preparing and enqueueing non-camera per-frame GPU data.
    ///
    /// Depending on the rendering path this includes transient vertex or
    /// instance writes, scalar LUT and pass-uniform uploads, and creation of
    /// the per-frame bindings needed by image, glyph, scalar, particle, or
    /// composition passes. The camera-only write remains separately available
    /// through [`Self::camera_uniform_upload`]. This is CPU preparation/enqueue
    /// time, not GPU execution time. An ordinary streaming frame can include
    /// pre-acquire transient buffer
    /// capacity creation here, so a skipped report may have a non-zero duration
    /// even though it performed no queue write and reports zero uploaded bytes.
    pub fn upload(self) -> Duration {
        self.upload
    }

    /// Returns CPU time spent updating the small per-frame camera uniform.
    pub fn camera_uniform_upload(self) -> Duration {
        self.camera_uniform_upload
    }

    /// Returns CPU time spent acquiring the next presentation surface texture.
    ///
    /// In FIFO modes this includes queue back-pressure and is the closest CPU-side
    /// approximation to VSync wait exposed by `wgpu`.
    pub fn surface_acquire(self) -> Duration {
        self.surface_acquire
    }

    /// Returns CPU time spent encoding, submitting, and dispatching present.
    ///
    /// Present dispatch does not mean the monitor has already scanned out the
    /// frame; `wgpu` does not expose that timestamp here.
    pub fn encode_submit_present(self) -> Duration {
        self.encode_submit_present
    }

    /// Returns total CPU time inside the profiled render call.
    pub fn total_cpu(self) -> Duration {
        self.total_cpu
    }

    /// Returns whether this frame referenced retained visual geometry.
    ///
    /// This includes prepared scenes and retained sprite/glyph batches. It is
    /// a broad workload-classification hint rather than a statement that the
    /// frame performed no uploads, and it can be true together with
    /// [`Self::geometry_streamed`].
    pub fn geometry_reused(self) -> bool {
        self.geometry_reused
    }

    /// Returns whether this frame contained dynamically supplied visual work.
    ///
    /// This includes dynamic meshes, particle visibility instances, and scalar
    /// field passes. It is a broad workload-classification hint rather than an
    /// allocation or upload guarantee, and it can be true together with
    /// [`Self::geometry_reused`].
    pub fn geometry_streamed(self) -> bool {
        self.geometry_streamed
    }

    /// Returns command-level tessellation results for this frame.
    pub fn tessellation_stats(self) -> TessellationStats {
        self.tessellation_stats
    }
}

/// Status and CPU timing breakdown for one renderer frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderReport {
    pub(super) status: RenderStatus,
    pub(super) metrics: RendererFrameMetrics,
}

impl RenderReport {
    /// Returns whether the frame was drawn or skipped by the surface.
    pub fn status(self) -> RenderStatus {
        self.status
    }

    /// Returns CPU-side stage durations for the render call.
    pub fn metrics(self) -> RendererFrameMetrics {
        self.metrics
    }
}

/// Recoverable or fatal state reported by the presentation surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererSurfaceStatus {
    /// Acquiring the next frame timed out; try again on a later frame.
    Timeout,
    /// The surface is occluded or minimized; rendering can be skipped.
    Occluded,
    /// The surface configuration is stale; resizing or reconfiguring is needed.
    Outdated,
    /// The surface was lost and should be recreated by the host application.
    Lost,
    /// `wgpu` reported a validation error while acquiring the frame.
    Validation,
}

/// Fatal failure while preparing or submitting one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererFrameError {
    /// The presentation surface was lost or rejected frame acquisition.
    Surface(RendererSurfaceStatus),
    /// Camera, geometry, stroke arithmetic, or dynamic-triangle topology
    /// cannot be proven portable on the GPU.
    ///
    /// This includes nonzero subnormal or excessively large shader sources,
    /// transform ranges that may overflow under permitted GPU arithmetic, and
    /// stroke branch decisions that are ambiguous after projection, every
    /// dynamic triangle requiring partial frustum clipping, and projected
    /// triangle orientation that may change with legal shader association.
    InvalidGeometryTransform,
    /// A logical viewport is outside the active surface or render target.
    InvalidViewport,
    /// Tessellated geometry exceeds the active device's vertex-buffer limit.
    GeometryCapacityTooLarge,
    /// CPU storage for scene tessellation could not be reserved.
    SceneAllocationFailed {
        /// Minimum additional bytes requested by the failed reservation.
        requested_bytes: usize,
    },
    /// CPU storage for the camera-visible particle subset could not be reserved.
    ParticleAllocationFailed {
        /// Bytes requested for the rejected visible-instance vector.
        requested_bytes: usize,
    },
    /// Actual renderer work exceeded a scene's explicit limit.
    SceneBudgetExceeded {
        /// Work category whose post-tessellation limit was exceeded.
        resource: SceneBudgetResource,
        /// Configured maximum for the category.
        limit: usize,
        /// Actual renderer work observed.
        actual: usize,
    },
}

impl fmt::Display for RendererFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Surface(status) => write!(formatter, "renderer surface failed: {status:?}"),
            Self::InvalidGeometryTransform => {
                write!(
                    formatter,
                    "camera, geometry, or dynamic topology is outside the portable GPU envelope"
                )
            }
            Self::InvalidViewport => {
                write!(
                    formatter,
                    "logical viewport lies outside the active render target"
                )
            }
            Self::GeometryCapacityTooLarge => {
                write!(formatter, "geometry exceeds the GPU vertex-buffer limit")
            }
            Self::SceneAllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} additional bytes for scene tessellation"
            ),
            Self::ParticleAllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for visible particle instances"
            ),
            Self::SceneBudgetExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "scene {resource:?} work exceeded its limit {limit} after tessellation: {actual}"
            ),
        }
    }
}

impl Error for RendererFrameError {}

/// Errors that can happen while creating the `wgpu` renderer.
#[derive(Debug)]
pub enum RendererInitError {
    /// Surface creation failed for the provided window or canvas target.
    CreateSurface(wgpu::CreateSurfaceError),
    /// No compatible GPU adapter could be selected.
    RequestAdapter(wgpu::RequestAdapterError),
    /// A logical GPU device and queue could not be created.
    RequestDevice(wgpu::RequestDeviceError),
    /// The surface did not expose a usable default configuration.
    NoSurfaceConfig,
    /// Requested physical dimensions exceed the selected device limit.
    SurfaceDimensionsTooLarge {
        /// Rejected physical width.
        width: u32,
        /// Rejected physical height.
        height: u32,
        /// Device maximum for either dimension.
        limit: u32,
    },
    /// The bounded previous-device quarantine is full.
    RecoveryLimitReached {
        /// Configured maximum number of quarantined logical devices.
        limit: usize,
    },
    /// CPU storage for quarantining the previous device could not be reserved.
    RecoveryAllocationFailed {
        /// Additional bytes requested for one quarantine entry.
        requested_bytes: usize,
    },
}

impl fmt::Display for RendererInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateSurface(error) => write!(formatter, "failed to create surface: {error}"),
            Self::RequestAdapter(error) => write!(formatter, "failed to request adapter: {error}"),
            Self::RequestDevice(error) => write!(formatter, "failed to request device: {error}"),
            Self::NoSurfaceConfig => write!(formatter, "surface has no supported default config"),
            Self::SurfaceDimensionsTooLarge {
                width,
                height,
                limit,
            } => write!(
                formatter,
                "surface dimensions {width}x{height} exceed device limit {limit}"
            ),
            Self::RecoveryLimitReached { limit } => write!(
                formatter,
                "device recovery limit reached with {limit} quarantined devices"
            ),
            Self::RecoveryAllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for device recovery quarantine"
            ),
        }
    }
}

impl Error for RendererInitError {}

/// Invalid runtime or initialization configuration for [`WgpuRenderer`](crate::WgpuRenderer).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RendererConfigurationError {
    /// Logical-to-physical scale must keep all surface transforms normal and finite.
    InvalidScaleFactor {
        /// Rejected physical pixels per logical screen pixel.
        scale_factor: f64,
    },
    /// Previous-device quarantine must remain inside the supported bounded range.
    InvalidRecoveryLimit {
        /// Rejected maximum retained-device count.
        limit: usize,
    },
    /// Requested physical dimensions exceed the active device limit.
    SurfaceDimensionsTooLarge {
        /// Rejected physical width.
        width: u32,
        /// Rejected physical height.
        height: u32,
        /// Device maximum for either dimension.
        limit: u32,
    },
}

impl fmt::Display for RendererConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScaleFactor { scale_factor } => write!(
                formatter,
                "renderer scale factor must be finite, representable as f32, and keep every u32 surface transform in the normal f32 range, got {scale_factor}"
            ),
            Self::InvalidRecoveryLimit { limit } => write!(
                formatter,
                "renderer recovery quarantine must retain between 1 and 8 devices, got {limit}"
            ),
            Self::SurfaceDimensionsTooLarge {
                width,
                height,
                limit,
            } => write!(
                formatter,
                "surface dimensions {width}x{height} exceed device limit {limit}"
            ),
        }
    }
}

impl Error for RendererConfigurationError {}

/// Failure while converting between logical and physical screen coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RendererCoordinateError {
    /// The source position is non-finite or the scaled result cannot be represented as `f32`.
    NonFiniteConversion,
}

impl fmt::Display for RendererCoordinateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "screen coordinate conversion produced a non-finite result"
        )
    }
}

impl Error for RendererCoordinateError {}

/// Failure while uploading or restoring immutable prepared-scene geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedSceneError {
    /// Tessellated geometry exceeds the active device's vertex-buffer limit.
    CapacityTooLarge,
    /// Command sources or derived tessellated vertices are outside the
    /// portable GPU input envelope.
    InvalidGeometrySources,
    /// CPU storage for scene tessellation could not be reserved.
    AllocationFailed {
        /// Minimum additional bytes requested by the failed reservation.
        requested_bytes: usize,
    },
    /// Actual prepared-scene work exceeded an explicit scene limit.
    BudgetExceeded {
        /// Work category whose post-tessellation limit was exceeded.
        resource: SceneBudgetResource,
        /// Configured maximum for the category.
        limit: usize,
        /// Actual renderer work observed.
        actual: usize,
    },
}

impl fmt::Display for PreparedSceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityTooLarge => {
                write!(
                    formatter,
                    "prepared scene exceeds the GPU vertex-buffer limit"
                )
            }
            Self::InvalidGeometrySources => write!(
                formatter,
                "prepared scene contains non-portable GPU geometry sources"
            ),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} additional bytes for prepared-scene tessellation"
            ),
            Self::BudgetExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "prepared scene {resource:?} work exceeded its limit {limit}: {actual}"
            ),
        }
    }
}

impl From<TessellationError> for RendererFrameError {
    fn from(error: TessellationError) -> Self {
        match error {
            TessellationError::AllocationFailed { requested_bytes } => {
                Self::SceneAllocationFailed { requested_bytes }
            }
            TessellationError::CapacityTooLarge => Self::GeometryCapacityTooLarge,
            TessellationError::BudgetExceeded {
                resource,
                limit,
                actual,
            } => Self::SceneBudgetExceeded {
                resource,
                limit,
                actual,
            },
        }
    }
}

impl From<TessellationError> for PreparedSceneError {
    fn from(error: TessellationError) -> Self {
        match error {
            TessellationError::AllocationFailed { requested_bytes } => {
                Self::AllocationFailed { requested_bytes }
            }
            TessellationError::CapacityTooLarge => Self::CapacityTooLarge,
            TessellationError::BudgetExceeded {
                resource,
                limit,
                actual,
            } => Self::BudgetExceeded {
                resource,
                limit,
                actual,
            },
        }
    }
}

impl Error for PreparedSceneError {}

/// Failure to draw geometry prepared by [`WgpuRenderer::prepare_scene`](crate::WgpuRenderer::prepare_scene).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedSceneRenderError {
    /// The prepared geometry belongs to a different renderer and GPU device.
    RendererMismatch,
    /// Frame rendering failed after prepared-scene ownership validation.
    Frame(RendererFrameError),
}

impl fmt::Display for PreparedSceneRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RendererMismatch => {
                write!(formatter, "prepared scene belongs to a different renderer")
            }
            Self::Frame(error) => write!(formatter, "prepared frame failed: {error}"),
        }
    }
}

impl Error for PreparedSceneRenderError {}
