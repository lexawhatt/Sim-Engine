//! Explicit admission and transactional replacement of offscreen attachments.

use super::{
    LogicalViewport, Mesh3dResourceError, RenderTarget2d, RenderTarget3d, RenderTargetError,
    WgpuRenderer, render_target_allocation_bytes,
};
use std::{error::Error, fmt};

/// Nominal attachment limits for one target allocation or replacement.
///
/// Includes color and, for 3D, Depth32Float storage. Peak includes the old and
/// replacement attachments simultaneously, including across device recovery.
/// These are per-operation limits, not a renderer-wide pool. Other live targets,
/// driver metadata/alignment and backend-retained in-flight allocations are not
/// measurable here and are excluded. Target creation performs no CPU upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTargetBudget {
    max_target_bytes: usize,
    max_peak_bytes: usize,
}

impl RenderTargetBudget {
    /// Sets final-target and old-plus-new peak limits in nominal bytes.
    pub const fn new(max_target_bytes: usize, max_peak_bytes: usize) -> Self {
        Self {
            max_target_bytes,
            max_peak_bytes,
        }
    }

    /// Returns the maximum size of the resulting target.
    pub const fn max_target_bytes(self) -> usize {
        self.max_target_bytes
    }

    /// Returns the maximum simultaneous old-plus-new target storage.
    pub const fn max_peak_bytes(self) -> usize {
        self.max_peak_bytes
    }

    pub(super) fn admit(
        self,
        old: usize,
        new: usize,
    ) -> Result<RenderTargetReplacementReport, RenderTargetLifecycleError> {
        let peak = old
            .checked_add(new)
            .ok_or(RenderTargetLifecycleError::SizeOverflow)?;
        if new > self.max_target_bytes || peak > self.max_peak_bytes {
            return Err(RenderTargetLifecycleError::BudgetExceeded {
                target_bytes: new,
                peak_bytes: peak,
                budget: self,
            });
        }
        Ok(RenderTargetReplacementReport {
            previous_bytes: old,
            target_bytes: new,
            peak_bytes: peak,
        })
    }
}

/// Nominal attachment accounting after a successful replacement.
///
/// The replacement contains no previous pixels or depth: redraw before use.
/// Dropping the old handle does not force immediate driver deallocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTargetReplacementReport {
    /// Nominal attachment bytes held by the previous handle.
    pub previous_bytes: usize,
    /// Nominal attachment bytes held by the replacement.
    pub target_bytes: usize,
    /// Sum of old and new storage during the operation.
    pub peak_bytes: usize,
}

/// Failure before committing an offscreen target replacement.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum RenderTargetLifecycleError {
    /// Byte accounting exceeded the platform's addressable range.
    SizeOverflow,
    /// Nominal storage would exceed the supplied per-operation limits.
    BudgetExceeded {
        /// Requested final attachment storage.
        target_bytes: usize,
        /// Requested old-plus-new storage.
        peak_bytes: usize,
        /// Limits supplied by the caller.
        budget: RenderTargetBudget,
    },
    /// Color target dimensions or renderer provenance were invalid.
    Target(RenderTargetError),
    /// The 3D viewport or resource configuration was invalid.
    Target3d(Mesh3dResourceError),
}

impl fmt::Display for RenderTargetLifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => write!(f, "render target storage accounting overflow"),
            Self::BudgetExceeded {
                target_bytes,
                peak_bytes,
                ..
            } => write!(
                f,
                "render target budget exceeded: {target_bytes} final bytes, {peak_bytes} peak bytes"
            ),
            Self::Target(error) => write!(f, "{error}"),
            Self::Target3d(error) => write!(f, "{error}"),
        }
    }
}
impl Error for RenderTargetLifecycleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Target(error) => Some(error),
            Self::Target3d(error) => Some(error),
            _ => None,
        }
    }
}

impl WgpuRenderer {
    pub(super) fn target_storage_bytes(
        &self,
        width: u32,
        height: u32,
        depth: bool,
    ) -> Result<usize, RenderTargetLifecycleError> {
        target_storage_bytes(
            self.config.format,
            self.device.limits().max_texture_dimension_2d,
            width,
            height,
            depth,
        )
    }

    /// Creates a physical-pixel color target after checking explicit storage limits.
    /// GPU allocation failures follow wgpu's device-error model, as with the
    /// original unbudgeted constructor; these limits are admission checks.
    pub fn create_render_target_with_budget(
        &self,
        width: u32,
        height: u32,
        budget: RenderTargetBudget,
    ) -> Result<RenderTarget2d, RenderTargetLifecycleError> {
        budget.admit(0, self.target_storage_bytes(width, height, false)?)?;
        self.create_render_target(width, height)
            .map_err(RenderTargetLifecycleError::Target)
    }

    /// Creates a color/depth target with physical dimensions and a matching logical aspect.
    /// Admission counts both attachments before allocating either one.
    pub fn create_render_target3d_with_budget(
        &self,
        width: u32,
        height: u32,
        viewport: LogicalViewport,
        budget: RenderTargetBudget,
    ) -> Result<RenderTarget3d, RenderTargetLifecycleError> {
        budget.admit(0, self.target_storage_bytes(width, height, true)?)?;
        self.create_render_target3d(width, height, viewport)
            .map_err(RenderTargetLifecycleError::Target3d)
    }

    /// Replaces a color target for resize or device recovery, counting old/new overlap.
    /// Accepts an old-device handle explicitly. On any returned error the old
    /// handle and its pixels are unchanged (a lost device cannot preserve usable
    /// pixels). On success all old contents are invalidated, even at equal size;
    /// redraw before composition. No surface acquisition is performed.
    pub fn replace_render_target_with_budget(
        &self,
        target: &mut RenderTarget2d,
        width: u32,
        height: u32,
        budget: RenderTargetBudget,
    ) -> Result<RenderTargetReplacementReport, RenderTargetLifecycleError> {
        let report = budget.admit(
            target.allocation_bytes(),
            self.target_storage_bytes(width, height, false)?,
        )?;
        let replacement = self.create_render_target_with_budget(width, height, budget)?;
        *target = replacement;
        Ok(report)
    }

    /// Replaces color and depth together for resize or device recovery.
    /// The viewport aspect must match the physical dimensions. Both old attachments
    /// survive every returned error; successful replacement invalidates their
    /// contents and requires a complete redraw. Accepts old-device handles.
    pub fn replace_render_target3d_with_budget(
        &self,
        target: &mut RenderTarget3d,
        width: u32,
        height: u32,
        viewport: LogicalViewport,
        budget: RenderTargetBudget,
    ) -> Result<RenderTargetReplacementReport, RenderTargetLifecycleError> {
        let report = budget.admit(
            target.allocation_bytes(),
            self.target_storage_bytes(width, height, true)?,
        )?;
        let replacement =
            self.create_render_target3d_with_budget(width, height, viewport, budget)?;
        *target = replacement;
        Ok(report)
    }
}

fn target_storage_bytes(
    format: wgpu::TextureFormat,
    dimension_limit: u32,
    width: u32,
    height: u32,
    depth: bool,
) -> Result<usize, RenderTargetLifecycleError> {
    if width == 0 || height == 0 {
        return Err(RenderTargetLifecycleError::Target(
            RenderTargetError::ZeroDimension,
        ));
    }
    if width > dimension_limit || height > dimension_limit {
        return Err(RenderTargetLifecycleError::Target(
            RenderTargetError::DimensionsTooLarge,
        ));
    }
    let color = render_target_allocation_bytes(format, width, height)
        .ok_or(RenderTargetLifecycleError::SizeOverflow)?;
    let depth_bytes = if depth {
        (width as usize)
            .checked_mul(height as usize)
            .and_then(|value| value.checked_mul(4))
            .ok_or(RenderTargetLifecycleError::SizeOverflow)?
    } else {
        0
    };
    color
        .checked_add(depth_bytes)
        .ok_or(RenderTargetLifecycleError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_resolutions_charge_both_attachments_and_format() {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        assert_eq!(
            target_storage_bytes(format, 4096, 384, 240, false).unwrap(),
            368_640
        );
        assert_eq!(
            target_storage_bytes(format, 4096, 1152, 720, true).unwrap(),
            6_635_520
        );
        assert_eq!(
            target_storage_bytes(wgpu::TextureFormat::Rgba16Float, 4096, 384, 240, true).unwrap(),
            1_105_920
        );
        for (width, height) in [(0, 240), (384, 0)] {
            assert_eq!(
                target_storage_bytes(format, 4096, width, height, true),
                Err(RenderTargetLifecycleError::Target(
                    RenderTargetError::ZeroDimension
                ))
            );
        }
        assert_eq!(
            target_storage_bytes(format, 1024, 1152, 720, true),
            Err(RenderTargetLifecycleError::Target(
                RenderTargetError::DimensionsTooLarge
            ))
        );
        let old = target_storage_bytes(format, 4096, 384, 240, true).unwrap();
        let new = target_storage_bytes(format, 4096, 1152, 720, true).unwrap();
        assert!(RenderTargetBudget::new(new, new).admit(old, new).is_err());
        assert_eq!(
            RenderTargetBudget::new(new, 7_372_800)
                .admit(old, new)
                .unwrap()
                .peak_bytes,
            7_372_800
        );
    }

    #[test]
    fn final_and_overlap_limits_are_independent_and_inclusive() {
        let budget = RenderTargetBudget::new(80, 120);
        let report = budget.admit(40, 80).unwrap();
        assert_eq!(
            report,
            RenderTargetReplacementReport {
                previous_bytes: 40,
                target_bytes: 80,
                peak_bytes: 120
            }
        );
        assert!(matches!(
            budget.admit(41, 80),
            Err(RenderTargetLifecycleError::BudgetExceeded {
                peak_bytes: 121,
                ..
            })
        ));
        assert!(matches!(
            budget.admit(0, 81),
            Err(RenderTargetLifecycleError::BudgetExceeded {
                target_bytes: 81,
                ..
            })
        ));
        assert!(matches!(
            RenderTargetBudget::new(usize::MAX, usize::MAX).admit(1, usize::MAX),
            Err(RenderTargetLifecycleError::SizeOverflow)
        ));
        assert!(RenderTargetBudget::new(80, 79).admit(0, 80).is_err());
    }
}
