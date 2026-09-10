//! Bounded optional upload packing; draw bindings retain independent uniforms.

use super::*;

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(in crate::renderer) use gpu_tests::assert_gpu_uniform_upload_contract;

const MINIMUM_COPY_COUNT: usize = 32;

#[derive(Clone, Copy)]
struct PendingCopy {
    binding_index: usize,
    source_offset: u64,
    size: u64,
}

#[derive(Default, Clone, Copy)]
struct UploadPlan {
    max_bytes: usize,
    max_copies: usize,
    allocation_limit: u64,
}

/// Only the packed source is shared. No uniform destination, binding or draw
/// offset is shared or changed by this optimization.
#[derive(Default)]
pub(super) struct UniformUploadBatch {
    bytes: Vec<u8>,
    copies: Vec<PendingCopy>,
    source: Option<wgpu::Buffer>,
    plan: UploadPlan,
    active: bool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct UploadBatchReport {
    pub(super) created_buffers: usize,
    pub(super) buffer_bytes: usize,
    pub(super) peak_buffer_bytes: usize,
    pub(super) uploaded_bytes: usize,
    pub(super) copy_count: usize,
}

impl UniformUploadBatch {
    /// Starts fresh pending work without discarding retained scratch capacity.
    /// The caller must already have submitted the previous frame's encoder.
    pub(super) fn begin(&mut self) {
        self.bytes.clear();
        self.copies.clear();
        self.active = false;
    }

    pub(super) fn cpu_bytes(&self) -> usize {
        self.bytes.capacity().saturating_add(
            self.copies
                .capacity()
                .saturating_mul(std::mem::size_of::<PendingCopy>()),
        )
    }

    pub(super) fn allocation_bytes(&self) -> usize {
        self.source.as_ref().map_or(0, |buffer| {
            usize::try_from(buffer.size()).unwrap_or(usize::MAX)
        })
    }

    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }

    /// Reserves an upper bound, never performing GPU work. `count` bounds
    /// staged copies, not destination binding indices: changed slots can be
    /// sparse within a larger immutable binding list.
    ///
    /// False selects the existing direct-write path. Even failed preparation
    /// discards the previous pending plan; actual retained capacities remain
    /// visible through `cpu_bytes` for the caller's idle eviction policy.
    pub(super) fn prepare(
        &mut self,
        bytes: usize,
        count: usize,
        byte_limit: usize,
        max_buffer_size: u64,
    ) -> bool {
        self.begin();
        if self.bytes.capacity() > byte_limit {
            self.bytes = Vec::new();
        }
        let alignment = wgpu::COPY_BUFFER_ALIGNMENT as usize;
        let Some(minimum_bytes) = count.checked_mul(alignment) else {
            return false;
        };
        let Ok(gpu_bytes) = u64::try_from(bytes) else {
            return false;
        };
        if count < MINIMUM_COPY_COUNT
            || bytes < minimum_bytes
            || !bytes.is_multiple_of(alignment)
            || bytes > byte_limit
            || gpu_bytes > max_buffer_size
            || count
                .checked_mul(std::mem::size_of::<PendingCopy>())
                .is_none()
        {
            return false;
        }
        if self.bytes.try_reserve_exact(bytes).is_err() {
            return false;
        }
        // try_reserve_exact may legally return spare capacity; the payload
        // allocation itself must still respect the configured staging cap.
        if self.bytes.capacity() > byte_limit {
            self.bytes = Vec::new();
            return false;
        }
        if self.copies.try_reserve_exact(count).is_err() {
            return false;
        }
        self.plan = UploadPlan {
            max_bytes: bytes,
            max_copies: count,
            allocation_limit: u64::try_from(byte_limit)
                .unwrap_or(u64::MAX)
                .min(max_buffer_size),
        };
        self.active = true;
        true
    }

    /// Appends one complete changed uniform without allocating. Failed stages
    /// preserve previously packed writes; the caller writes this item directly.
    pub(super) fn stage(&mut self, binding_index: usize, bytes: &[u8]) -> bool {
        if !self.active {
            return false;
        }
        let plan = self.plan;
        let Some(end) = self.bytes.len().checked_add(bytes.len()) else {
            return false;
        };
        if bytes.is_empty()
            || !bytes
                .len()
                .is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
            || end > plan.max_bytes
            || self.copies.len() >= plan.max_copies
        {
            return false;
        }
        // prepare reserved both exact upper bounds before any GPU mutation.
        debug_assert!(end <= self.bytes.capacity());
        debug_assert!(self.copies.len() < self.copies.capacity());
        self.copies.push(PendingCopy {
            binding_index,
            source_offset: self.bytes.len() as u64,
            size: bytes.len() as u64,
        });
        self.bytes.extend_from_slice(bytes);
        true
    }

    /// Executes after successful surface acquisition and before the render pass.
    /// The source receives exactly one write; disjoint source offsets preserve
    /// every destination's contents when wgpu executes queue writes before the
    /// submitted copy commands. The caller submits this encoder exactly once.
    pub(super) fn flush(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &[FrameBinding],
    ) -> UploadBatchReport {
        let old_bytes = self.allocation_bytes();
        let mut report = UploadBatchReport {
            buffer_bytes: old_bytes,
            peak_buffer_bytes: old_bytes,
            ..UploadBatchReport::default()
        };
        if self.copies.is_empty() {
            self.begin();
            return report;
        }
        debug_assert!(self.active);
        let plan = self.plan;
        // Destination indices come from the caller's fixed frame-binding list,
        // not the number of changed copies. Catch internal misuse before writes.
        debug_assert!(self.copies.iter().all(|copy| {
            bindings
                .get(copy.binding_index)
                .is_some_and(|binding| copy.size <= binding._buffer.size())
        }));
        let size = self.bytes.len() as u64;
        if self
            .source
            .as_ref()
            .is_some_and(|source| source.size() < size || source.size() > plan.allocation_limit)
        {
            self.source = None;
        }
        let source = self.source.get_or_insert_with(|| {
            // Exact growth avoids hidden padding beyond the declared limit.
            report.created_buffers = 1;
            report.buffer_bytes = size as usize;
            report.peak_buffer_bytes = old_bytes.saturating_add(report.buffer_bytes);
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sim-engine packed frame uniforms"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        queue.write_buffer(source, 0, &self.bytes);
        for copy in &self.copies {
            encoder.copy_buffer_to_buffer(
                source,
                copy.source_offset,
                &bindings[copy.binding_index]._buffer,
                0,
                copy.size,
            );
        }
        report.uploaded_bytes = self.bytes.len();
        report.copy_count = self.copies.len();
        self.begin();
        report
    }
}
