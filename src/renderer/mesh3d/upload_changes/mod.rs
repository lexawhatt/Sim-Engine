//! One allocation-free write span over a validated, retained CPU byte mirror.

use std::ops::Range;

#[cfg(test)]
mod tests;

/// Merges changed records into at most one queue write, including unchanged
/// bytes between them. The caller owns mirror capacity and GPU initialization.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct UploadSpan {
    changed: Option<Range<usize>>,
}

impl UploadSpan {
    /// Marks bytes inside a prevalidated mirror whose size fits in `isize`.
    /// Empty ranges do not turn an unchanged frame into an upload.
    pub(super) fn mark(&mut self, bytes: Range<usize>) {
        debug_assert!(bytes.start <= bytes.end);
        debug_assert!(bytes.end <= isize::MAX as usize);
        if bytes.is_empty() {
            return;
        }
        if let Some(changed) = &mut self.changed {
            changed.start = changed.start.min(bytes.start);
            changed.end = changed.end.max(bytes.end);
        } else {
            self.changed = Some(bytes);
        }
    }

    /// GPU replacement or loss of the previous CPU mirror invalidates all
    /// equality-based skips. An empty active prefix still requires no write.
    pub(super) fn force_full(&mut self, active_bytes: usize) {
        self.changed = None;
        self.mark(0..active_bytes);
    }

    /// Returns outward-rounded COPY_BUFFER_ALIGNMENT bounds. The caller's
    /// active mirror must include the complete aligned final word.
    pub(super) fn range(&self) -> Option<Range<usize>> {
        let alignment = wgpu::COPY_BUFFER_ALIGNMENT as usize;
        self.changed.as_ref().map(|bytes| {
            // mark bounds every offset by isize::MAX, leaving enough usize
            // headroom to round the final word without overflowing.
            bytes.start / alignment * alignment..bytes.end.div_ceil(alignment) * alignment
        })
    }

    pub(super) fn byte_count(&self) -> usize {
        self.range().map_or(0, |bytes| bytes.len())
    }

    /// Replaces one existing POD record and marks it only if its exact bytes
    /// differ. In particular, signed zero and NaN payloads are not float-equal.
    pub(super) fn replace<T: bytemuck::Pod>(
        &mut self,
        byte_offset: usize,
        previous: &mut T,
        next: T,
    ) -> bool {
        self.replace_bytes(
            byte_offset,
            bytemuck::bytes_of_mut(previous),
            bytemuck::bytes_of(&next),
        )
    }

    /// Replaces equal-length existing records within the caller's validated
    /// mirror. Newly active slots must be marked even when all bytes are zero:
    /// equality with newly initialized CPU storage proves no GPU residency.
    pub(super) fn replace_bytes(
        &mut self,
        byte_offset: usize,
        previous: &mut [u8],
        next: &[u8],
    ) -> bool {
        debug_assert_eq!(previous.len(), next.len());
        if previous == next {
            return false;
        }
        previous.copy_from_slice(next);
        self.mark(byte_offset..byte_offset + next.len());
        true
    }
}
