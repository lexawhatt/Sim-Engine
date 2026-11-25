//! Explicit, bounded synchronous readback for visual verification.

use super::{RenderTarget2d, WgpuRenderer};
use std::{
    error::Error,
    fmt,
    sync::{Arc, mpsc},
    time::Duration,
};

/// Host correlation attached to a capture request, not inferred scene identity.
/// The host supplies the committed frame and texture-revision IDs used to build
/// its composition. Engine preserves these labels unchanged; they do not prove
/// that a surface present occurred. Pair with the corresponding frame report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureCorrelation {
    /// Host frame ID for the rendered visual state.
    pub frame_id: u64,
    /// Host texture/content revision for that frame.
    pub texture_revision: u64,
}

/// A tightly packed top-to-bottom snapshot of actual GPU target texels.
/// Channels remain in `format()` order (RGBA or BGRA); sRGB encoding and
/// premultiplied alpha are preserved, without unpremultiplication or conversion.
/// This verifies offscreen composition, not window-system presentation.
#[derive(Debug)]
pub struct RenderTargetCapture {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    pixels: Vec<u8>,
    staging_bytes: usize,
    correlation: CaptureCorrelation,
    content_id: super::RenderTargetContentId,
}
impl RenderTargetCapture {
    /// Engine-owned identity/revision of the captured committed target contents.
    /// Compare this with the ID recorded after a successful draw to detect stale
    /// content regardless of the caller-supplied correlation labels.
    pub fn content_id(&self) -> &super::RenderTargetContentId {
        &self.content_id
    }

    /// Physical dimensions in texels.
    pub const fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    /// Exact channel order and transfer encoding of the returned bytes.
    pub const fn format(&self) -> wgpu::TextureFormat {
        self.format
    }
    /// Tightly packed bytes, four per texel, with no row padding.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
    /// Host correlation labels supplied to this request.
    pub const fn correlation(&self) -> CaptureCorrelation {
        self.correlation
    }
    /// Nominal GPU readback buffer size including 256-byte row padding.
    pub const fn staging_bytes(&self) -> usize {
        self.staging_bytes
    }
    /// Actual retained CPU vector capacity in bytes.
    pub fn retained_bytes(&self) -> usize {
        self.pixels.capacity()
    }
}

/// Capture failed without changing source pixels or presenting a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CaptureError {
    /// Target belongs to another renderer/device generation.
    RendererMismatch,
    /// Target content revision counter exhausted its non-repeating range.
    RevisionExhausted,
    /// An earlier capture is still awaiting GPU completion, or another capture is active.
    Busy,
    /// Only RGBA8/BGRA8 normalized linear or sRGB formats are supported.
    UnsupportedFormat,
    /// Checked size or device buffer range was exceeded.
    SizeOverflow,
    /// Padded staging plus CPU output capacity exceeds the supplied byte limit.
    BudgetExceeded,
    /// Host output reservation failed.
    AllocationFailed,
    /// GPU completion polling failed or exceeded the supplied timeout.
    PollFailed,
    /// GPU readback mapping failed or did not complete.
    MapFailed,
}
impl fmt::Display for CaptureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "render target capture failed: {self:?}")
    }
}
impl Error for CaptureError {}

#[derive(Default)]
pub(super) struct CaptureState {
    pending: std::sync::Mutex<Option<PendingCapture>>,
}
struct PendingCapture {
    buffer: wgpu::Buffer,
    device: wgpu::Device,
    complete: Arc<std::sync::atomic::AtomicBool>,
}

impl CaptureState {
    fn retained_bytes(&self) -> Result<usize, CaptureError> {
        let mut pending = self.pending.try_lock().map_err(|_| CaptureError::Busy)?;
        if let Some(previous) = pending.as_ref() {
            let _ = previous.device.poll(wgpu::PollType::Poll);
            if previous.complete.load(std::sync::atomic::Ordering::Acquire) {
                previous.buffer.unmap();
                *pending = None;
            }
        }
        Ok(pending
            .as_ref()
            .map_or(0, |entry| entry.buffer.size() as usize))
    }
}
impl WgpuRenderer {
    /// Nominal staging still held by timed-out captures, including quarantined
    /// devices. Completed buffers are retired while querying. New capture limits
    /// include this retained storage as well as the requested staging/output.
    pub fn capture_retained_staging_bytes(&self) -> Result<usize, CaptureError> {
        self.retired_devices
            .iter()
            .try_fold(self.capture_state.retained_bytes()?, |sum, device| {
                sum.checked_add(device._capture.retained_bytes()?)
                    .ok_or(CaptureError::SizeOverflow)
            })
    }

    /// Copies a target's current GPU contents and waits up to `timeout` for GPU
    /// completion. Intended for diagnostics, not the animation hot path.
    ///
    /// `max_bytes` bounds padded readback storage plus retained CPU output
    /// capacity, before submission. Driver allocation metadata is excluded.
    /// A timed-out readback remains owned by the renderer. New requests return
    /// Busy until that GPU operation completes, preventing unbounded retry staging.
    /// Earlier queue work is ordered before the copy. No surface is acquired:
    /// retain the actual composition/presentation report for rejected/skipped
    /// frame diagnostics. After device replacement restore/redraw the target.
    pub fn capture_render_target(
        &self,
        target: &RenderTarget2d,
        correlation: CaptureCorrelation,
        max_bytes: usize,
        timeout: Duration,
    ) -> Result<RenderTargetCapture, CaptureError> {
        if !Arc::ptr_eq(&self.renderer_identity, &target.renderer_identity) {
            return Err(CaptureError::RendererMismatch);
        }
        let retained = self
            .retired_devices
            .iter()
            .try_fold(0usize, |sum, device| {
                sum.checked_add(device._capture.retained_bytes()?)
                    .ok_or(CaptureError::SizeOverflow)
            })?;
        let max_bytes = max_bytes
            .checked_sub(retained)
            .ok_or(CaptureError::BudgetExceeded)?;
        capture(
            &self.capture_state,
            &self.device,
            &self.queue,
            target,
            correlation,
            max_bytes,
            timeout,
        )
    }
}

fn layout(width: u32, height: u32, maximum: usize) -> Result<(u32, usize, usize), CaptureError> {
    let row = width.checked_mul(4).ok_or(CaptureError::SizeOverflow)?;
    let padded = row
        .checked_add(255)
        .map(|value| value / 256 * 256)
        .ok_or(CaptureError::SizeOverflow)?;
    let staging = (padded as usize)
        .checked_mul(height as usize)
        .ok_or(CaptureError::SizeOverflow)?;
    let output = (row as usize)
        .checked_mul(height as usize)
        .ok_or(CaptureError::SizeOverflow)?;
    if staging
        .checked_add(output)
        .ok_or(CaptureError::SizeOverflow)?
        > maximum
    {
        return Err(CaptureError::BudgetExceeded);
    }
    Ok((padded, staging, output))
}

pub(super) fn capture(
    state: &CaptureState,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &RenderTarget2d,
    correlation: CaptureCorrelation,
    max_bytes: usize,
    timeout: Duration,
) -> Result<RenderTargetCapture, CaptureError> {
    let mut pending = state.pending.try_lock().map_err(|_| CaptureError::Busy)?;
    if let Some(previous) = pending.as_ref() {
        let _ = previous.device.poll(wgpu::PollType::Poll);
        if !previous.complete.load(std::sync::atomic::Ordering::Acquire) {
            return Err(CaptureError::Busy);
        }
        previous.buffer.unmap();
        *pending = None;
    }
    let content_id = target.content_id().ok_or(CaptureError::RevisionExhausted)?;
    let format = target._texture.format();
    if !matches!(
        format,
        wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        return Err(CaptureError::UnsupportedFormat);
    }
    let (padded, staging_bytes, output_bytes) = layout(target.width(), target.height(), max_bytes)?;
    if staging_bytes as u64 > device.limits().max_buffer_size {
        return Err(CaptureError::SizeOverflow);
    }
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(output_bytes)
        .map_err(|_| CaptureError::AllocationFailed)?;
    if staging_bytes
        .checked_add(pixels.capacity())
        .ok_or(CaptureError::SizeOverflow)?
        > max_bytes
    {
        return Err(CaptureError::BudgetExceeded);
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine bounded capture"),
        size: staging_bytes as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sim-engine capture copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target._texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(target.height()),
            },
        },
        wgpu::Extent3d {
            width: target.width(),
            height: target.height(),
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::sync_channel(1);
    let complete = Arc::new(std::sync::atomic::AtomicBool::new(false));
    *pending = Some(PendingCapture {
        buffer: buffer.clone(),
        device: device.clone(),
        complete: Arc::clone(&complete),
    });
    let callback_complete = Arc::clone(&complete);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
        callback_complete.store(true, std::sync::atomic::Ordering::Release);
    });
    let result = (|| {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(timeout),
            })
            .map_err(|_| CaptureError::PollFailed)?;
        receiver
            .try_recv()
            .map_err(|_| CaptureError::MapFailed)?
            .map_err(|_| CaptureError::MapFailed)?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|_| CaptureError::MapFailed)?;
        for row in mapped.chunks_exact(padded as usize) {
            pixels.extend_from_slice(&row[..target.width() as usize * 4]);
        }
        Ok(())
    })();
    if result.is_ok() || complete.load(std::sync::atomic::Ordering::Acquire) {
        buffer.unmap();
        *pending = None;
    }
    result?;
    Ok(RenderTargetCapture {
        width: target.width(),
        height: target.height(),
        format,
        pixels,
        staging_bytes,
        correlation,
        content_id,
    })
}

#[cfg(test)]
pub(super) fn verify_backpressure(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &RenderTarget2d,
) {
    let state = CaptureState::default();
    let labels = CaptureCorrelation {
        frame_id: 1,
        texture_revision: 2,
    };
    // Holding the slot proves concurrent requests cannot allocate/submit.
    {
        let _guard = state.pending.lock().unwrap();
        assert_eq!(
            capture(
                &state,
                device,
                queue,
                target,
                labels,
                usize::MAX,
                Duration::ZERO
            )
            .unwrap_err(),
            CaptureError::Busy
        );
    }
    match capture(
        &state,
        device,
        queue,
        target,
        labels,
        usize::MAX,
        Duration::ZERO,
    ) {
        Ok(_) => {}
        Err(CaptureError::PollFailed) => {
            for _ in 0..8 {
                let previous = state
                    .pending
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|pending| Arc::clone(&pending.complete));
                match capture(
                    &state,
                    device,
                    queue,
                    target,
                    labels,
                    usize::MAX,
                    Duration::ZERO,
                ) {
                    Err(CaptureError::Busy) => {
                        let pending = state.pending.lock().unwrap();
                        assert!(Arc::ptr_eq(
                            previous.as_ref().unwrap(),
                            &pending.as_ref().unwrap().complete
                        ));
                    }
                    Ok(_) | Err(CaptureError::PollFailed) => {}
                    other => panic!("unexpected retry result: {other:?}"),
                }
            }
        }
        other => panic!("unexpected zero-timeout result: {other:?}"),
    }
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(15)),
        })
        .unwrap();
    let result = capture(
        &state,
        device,
        queue,
        target,
        labels,
        usize::MAX,
        Duration::from_secs(15),
    )
    .unwrap();
    assert_eq!(Some(result.content_id()), target.content_id().as_ref());
    assert!(state.pending.lock().unwrap().is_none());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padding_and_peak_are_admitted_before_allocation() {
        assert_eq!(layout(3, 2, 536), Ok((256, 512, 24)));
        assert_eq!(layout(3, 2, 535), Err(CaptureError::BudgetExceeded));
        assert_eq!(
            layout(u32::MAX, 1, usize::MAX),
            Err(CaptureError::SizeOverflow)
        );
        assert_eq!(
            layout(1152, 720, 6_635_520),
            Ok((4608, 3_317_760, 3_317_760))
        );
    }
}
