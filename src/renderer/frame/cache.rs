use super::*;

#[cfg(test)]
#[path = "cache_diagnostics.rs"]
mod diagnostics;
#[cfg(test)]
#[path = "cache_sharing_tests.rs"]
mod sharing_tests;
#[cfg(test)]
pub(in crate::renderer) use sharing_tests::assert_gpu_binding_sharing_contract;

/// Idle retention limits for composition scratch and cached draw bindings.
///
/// Active work remains bounded by `FrameBudget`. When a frame exceeds these
/// cache limits it still renders, but excess storage is released afterwards.
/// Zero limits disable retention. Driver allocations and in-flight retirement
/// are outside these library-owned byte counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameCacheBudget {
    max_cpu_bytes: usize,
    max_uniform_bytes: usize,
    max_texture_bytes: usize,
    max_bindings: usize,
    max_upload_staging_bytes: usize,
}

impl FrameCacheBudget {
    /// Creates exact retention limits with packed uploads disabled. Zero is a
    /// valid disabled cache; use `with_upload_staging_bytes` to opt into packing.
    pub const fn new(
        cpu_bytes: usize,
        uniform_bytes: usize,
        texture_bytes: usize,
        bindings: usize,
    ) -> Self {
        Self {
            max_cpu_bytes: cpu_bytes,
            max_uniform_bytes: uniform_bytes,
            max_texture_bytes: texture_bytes,
            max_bindings: bindings,
            max_upload_staging_bytes: 0,
        }
    }
    /// Maximum idle CPU scratch, slot metadata and saved uniform bytes.
    pub const fn max_cpu_bytes(self) -> usize {
        self.max_cpu_bytes
    }
    /// Maximum retained GPU uniform buffer bytes.
    pub const fn max_uniform_bytes(self) -> usize {
        self.max_uniform_bytes
    }
    /// Maximum distinct texture bytes kept alive by cached bindings.
    pub const fn max_texture_bytes(self) -> usize {
        self.max_texture_bytes
    }
    /// Maximum cached per-item binding slots.
    pub const fn max_bindings(self) -> usize {
        self.max_bindings
    }
    /// Enables bounded packing of changed retained uniforms into one upload.
    /// The cap limits the retained GPU transfer buffer and CPU payload separately;
    /// payload and copy-record capacities also count toward `max_cpu_bytes`.
    /// Zero disables batching. Small/oversized batches keep direct writes, so
    /// this optional optimization does not change accepted frame budgets.
    /// `new` disables batching; `default` permits 256 KiB.
    pub const fn with_upload_staging_bytes(mut self, bytes: usize) -> Self {
        self.max_upload_staging_bytes = bytes;
        self
    }
    /// Maximum GPU transfer-buffer bytes for packed uniform uploads, separate
    /// from the uniform-buffer limit. In-flight driver retirement is excluded.
    pub const fn max_upload_staging_bytes(self) -> usize {
        self.max_upload_staging_bytes
    }
}

impl Default for FrameCacheBudget {
    fn default() -> Self {
        Self::new(8 * 1024 * 1024, 1024 * 1024, 256 * 1024 * 1024, 4096)
            .with_upload_staging_bytes(256 * 1024)
    }
}

/// Library-owned retained allocation sizes and the last composition's cache work.
/// CPU byte counts use actual Vec capacities, uniform and transfer-storage bytes
/// use buffer sizes, and texture bytes count each cached resource identity once.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FrameCacheStatistics {
    pub(super) cpu_bytes: usize,
    pub(super) uniform_bytes: usize,
    pub(super) texture_bytes: usize,
    pub(super) binding_count: usize,
    pub(super) created_buffers: usize,
    pub(super) created_bind_groups: usize,
    pub(super) reused_bind_groups: usize,
    pub(super) shared_bind_groups: usize,
    pub(super) uploaded_uniform_bytes: usize,
    pub(super) uniform_upload_calls: usize,
    pub(super) batched_uniform_copies: usize,
    pub(super) batched_uniform_bytes: usize,
    pub(super) upload_staging_bytes: usize,
    pub(super) peak_upload_staging_bytes: usize,
    pub(super) created_upload_staging_buffers: usize,
    pub(super) peak_cpu_bytes: usize,
    pub(super) peak_uniform_bytes: usize,
    pub(super) peak_texture_bytes: usize,
}

impl FrameCacheStatistics {
    /// Idle retained CPU allocation bytes, including spare capacities.
    pub const fn cpu_bytes(self) -> usize {
        self.cpu_bytes
    }
    /// Idle retained uniform buffer allocation bytes.
    pub const fn uniform_bytes(self) -> usize {
        self.uniform_bytes
    }
    /// Distinct nominal texture bytes kept alive by cached bindings.
    pub const fn texture_bytes(self) -> usize {
        self.texture_bytes
    }
    /// Number of populated reusable binding slots.
    pub const fn binding_count(self) -> usize {
        self.binding_count
    }
    /// Uniform buffers created by the previous composed frame.
    pub const fn created_buffers(self) -> usize {
        self.created_buffers
    }
    /// Bind groups created by the previous composed frame.
    pub const fn created_bind_groups(self) -> usize {
        self.created_bind_groups
    }
    /// Bind groups reused from retained slots or earlier items of this frame.
    pub const fn reused_bind_groups(self) -> usize {
        self.reused_bind_groups
    }
    /// Subset of reused bind groups shared with an earlier identical draw
    /// binding in the same frame. Shared aliases are never retained in a second
    /// mutable uniform slot. The bounded memo may miss older duplicates.
    pub const fn shared_bind_groups(self) -> usize {
        self.shared_bind_groups
    }
    /// Uniform bytes actually written by the previous composed frame.
    pub const fn uploaded_uniform_bytes(self) -> usize {
        self.uploaded_uniform_bytes
    }
    /// Actual uniform-payload queue writes, including a packed write as one
    /// call. Copies from packed storage into uniform buffers are not uploads.
    pub const fn uniform_upload_calls(self) -> usize {
        self.uniform_upload_calls
    }
    /// GPU-to-GPU copies from packed storage to independent uniform buffers.
    /// Their bytes are not counted a second time in `uploaded_uniform_bytes`.
    pub const fn batched_uniform_copies(self) -> usize {
        self.batched_uniform_copies
    }
    /// Additional GPU-to-GPU transfer bytes from packed storage. This is a
    /// subset of uploaded uniform payload, not a second host upload. Batching
    /// trades this bounded copy work for fewer queue staging allocations.
    pub const fn batched_uniform_bytes(self) -> usize {
        self.batched_uniform_bytes
    }
    /// Idle retained GPU transfer-buffer bytes, separate from uniform bytes.
    pub const fn upload_staging_bytes(self) -> usize {
        self.upload_staging_bytes
    }
    /// Conservative old/new GPU transfer-buffer overlap during the last frame.
    pub const fn peak_upload_staging_bytes(self) -> usize {
        self.peak_upload_staging_bytes
    }
    /// GPU transfer buffers created by the previous composed frame.
    pub const fn created_upload_staging_buffers(self) -> usize {
        self.created_upload_staging_buffers
    }
    /// Conservative CPU scratch bound: initial retained bytes plus twice the
    /// final owned capacities, covering transient old/new Vec growth. This is
    /// not an allocator high-water measurement and excludes driver staging.
    pub const fn peak_cpu_bytes(self) -> usize {
        self.peak_cpu_bytes
    }
    /// Conservative peak including buffers replaced during this frame.
    pub const fn peak_uniform_bytes(self) -> usize {
        self.peak_uniform_bytes
    }
    /// Conservative peak of distinct cached old/new texture references.
    pub const fn peak_texture_bytes(self) -> usize {
        self.peak_texture_bytes
    }
}

#[derive(Clone)]
enum BindingKey {
    Camera,
    Image {
        identity: Arc<()>,
        sampling: ImageSampling,
        texture_bytes: usize,
    },
    Target {
        identity: Arc<()>,
        texture_bytes: usize,
    },
}

/// Temporary descriptions borrow provenance. Only a new retained slot clones
/// its identity; a warmed draw must not increment/decrement a shared Arc.
#[derive(Clone, Copy)]
enum BindingKeyRef<'a> {
    Camera,
    Image {
        identity: &'a Arc<()>,
        sampling: ImageSampling,
        texture_bytes: usize,
    },
    Target {
        identity: &'a Arc<()>,
        texture_bytes: usize,
    },
}

impl<'a> BindingKeyRef<'a> {
    fn matches(self, other: BindingKeyRef<'_>) -> bool {
        match (self, other) {
            (Self::Camera, BindingKeyRef::Camera) => true,
            (
                Self::Image {
                    identity: a,
                    sampling: sa,
                    ..
                },
                BindingKeyRef::Image {
                    identity: b,
                    sampling: sb,
                    ..
                },
            ) => Arc::ptr_eq(a, b) && sa == sb,
            (Self::Target { identity: a, .. }, BindingKeyRef::Target { identity: b, .. }) => {
                Arc::ptr_eq(a, b)
            }
            _ => false,
        }
    }

    fn texture(self) -> Option<(&'a Arc<()>, usize)> {
        match self {
            Self::Camera => None,
            Self::Image {
                identity,
                texture_bytes,
                ..
            }
            | Self::Target {
                identity,
                texture_bytes,
            } => Some((identity, texture_bytes)),
        }
    }

    fn into_owned(self) -> BindingKey {
        match self {
            Self::Camera => BindingKey::Camera,
            Self::Image {
                identity,
                sampling,
                texture_bytes,
            } => BindingKey::Image {
                identity: Arc::clone(identity),
                sampling,
                texture_bytes,
            },
            Self::Target {
                identity,
                texture_bytes,
            } => BindingKey::Target {
                identity: Arc::clone(identity),
                texture_bytes,
            },
        }
    }
}

impl BindingKey {
    fn as_ref(&self) -> BindingKeyRef<'_> {
        match self {
            Self::Camera => BindingKeyRef::Camera,
            Self::Image {
                identity,
                sampling,
                texture_bytes,
            } => BindingKeyRef::Image {
                identity,
                sampling: *sampling,
                texture_bytes: *texture_bytes,
            },
            Self::Target {
                identity,
                texture_bytes,
            } => BindingKeyRef::Target {
                identity,
                texture_bytes: *texture_bytes,
            },
        }
    }

    fn texture(&self) -> Option<(&Arc<()>, usize)> {
        self.as_ref().texture()
    }
    fn matches(&self, other: BindingKeyRef<'_>) -> bool {
        self.as_ref().matches(other)
    }
}

type BindingDescription<'a> = (BindingKeyRef<'a>, &'a [u8]);

#[derive(Clone, Copy)]
struct SharedBinding<'a> {
    description: BindingDescription<'a>,
    index: usize,
}

/// Fixed stack scratch for one binding-construction loop. It borrows immutable
/// ready descriptions, never GPU slots or the growing current bindings Vec.
/// Recording a persistent hit is O(1); only a slot miss searches eight entries.
#[derive(Default)]
pub(super) struct FrameBindingSharing<'a> {
    entries: [Option<SharedBinding<'a>>; 8],
    next: usize,
}

impl<'a> FrameBindingSharing<'a> {
    fn find(&self, (key, bytes): BindingDescription<'_>) -> Option<usize> {
        self.entries.iter().flatten().find_map(|entry| {
            (entry.description.0.matches(key) && entry.description.1 == bytes)
                .then_some(entry.index)
        })
    }

    fn remember(&mut self, description: Option<BindingDescription<'a>>, index: usize) {
        if let Some(description) = description {
            self.entries[self.next] = Some(SharedBinding { description, index });
            self.next = (self.next + 1) % self.entries.len();
        }
    }
}

struct CachedBinding {
    key: BindingKey,
    binding: FrameBinding,
    bytes: [u8; std::mem::size_of::<ImageUniform>()],
    len: usize,
}

#[derive(Default)]
pub(in crate::renderer) struct FrameCache {
    pub(super) items: Vec<FrameItem<'static>>,
    pub(super) retained_resources: Vec<RetainedResourceKey>,
    pub(super) scalar_luts: Vec<ScalarLutPlan>,
    pub(super) ready: Vec<ReadyItem<'static>>,
    pub(super) bindings: Vec<FrameBinding>,
    pub(super) batches: Vec<Vec<PreparedDrawBatch>>,
    slots: Vec<Option<CachedBinding>>,
    uniform_uploads: uniform_uploads::UniformUploadBatch,
    budget: FrameCacheBudget,
    statistics: FrameCacheStatistics,
    initial_cpu_bytes: usize,
}

impl FrameCache {
    pub(super) fn uploaded_uniform_bytes(&self) -> usize {
        self.statistics.uploaded_uniform_bytes
    }
    pub(super) fn begin(&mut self) {
        self.uniform_uploads.begin();
        self.initial_cpu_bytes = self.cpu_bytes();
        self.statistics.created_buffers = 0;
        self.statistics.created_bind_groups = 0;
        self.statistics.reused_bind_groups = 0;
        self.statistics.shared_bind_groups = 0;
        self.statistics.uploaded_uniform_bytes = 0;
        self.statistics.uniform_upload_calls = 0;
        self.statistics.batched_uniform_copies = 0;
        self.statistics.batched_uniform_bytes = 0;
        self.statistics.created_upload_staging_buffers = 0;
        self.statistics.peak_upload_staging_bytes = self.uniform_uploads.allocation_bytes();
        self.statistics.peak_cpu_bytes = self.initial_cpu_bytes;
        self.statistics.peak_uniform_bytes = self.statistics.uniform_bytes;
        self.statistics.peak_texture_bytes = self.statistics.texture_bytes;
    }

    fn cpu_bytes(&self) -> usize {
        vec_bytes(&self.items)
            .saturating_add(vec_bytes(&self.retained_resources))
            .saturating_add(vec_bytes(&self.scalar_luts))
            .saturating_add(vec_bytes(&self.ready))
            .saturating_add(vec_bytes(&self.bindings))
            .saturating_add(vec_bytes(&self.slots))
            .saturating_add(vec_bytes(&self.batches))
            .saturating_add(self.batches.iter().map(vec_bytes).sum::<usize>())
            .saturating_add(self.uniform_uploads.cpu_bytes())
    }

    pub(super) fn finish(&mut self) {
        // No pending references or payload from this presentation may become
        // next frame's upload, including aborted/skipped frames.
        self.uniform_uploads.begin();
        let mut cpu_bytes = self.cpu_bytes();
        self.statistics.peak_cpu_bytes = self
            .initial_cpu_bytes
            .saturating_add(cpu_bytes.saturating_mul(2));
        if cpu_bytes > self.budget.max_cpu_bytes {
            self.items = Vec::new();
            self.retained_resources = Vec::new();
            self.scalar_luts = Vec::new();
            self.ready = Vec::new();
            self.bindings = Vec::new();
            self.batches = Vec::new();
            self.uniform_uploads.clear();
            cpu_bytes = self.cpu_bytes();
        }
        if cpu_bytes > self.budget.max_cpu_bytes {
            self.slots = Vec::new();
            self.statistics.uniform_bytes = 0;
            self.statistics.texture_bytes = 0;
            cpu_bytes = 0;
        }
        self.statistics.cpu_bytes = cpu_bytes;
        self.statistics.upload_staging_bytes = self.uniform_uploads.allocation_bytes();
        self.statistics.binding_count = self.slots.iter().flatten().count();
    }

    pub(in crate::renderer) fn clear(&mut self) {
        let budget = self.budget;
        *self = Self {
            budget,
            ..Self::default()
        };
    }

    pub(super) fn reserve_slots(&mut self, count: usize) -> Result<(), FrameComposerError> {
        let count = count.min(self.budget.max_bindings);
        if count > self.slots.len() {
            self.slots
                .try_reserve_exact(count - self.slots.len())
                .map_err(|_| FrameComposerError::AllocationFailed {
                    requested_bytes: count
                        .saturating_mul(std::mem::size_of::<Option<CachedBinding>>()),
                })?;
            self.slots.resize_with(count, || None);
        }
        Ok(())
    }

    pub(super) fn prepare_uniform_uploads(
        &mut self,
        ready: &[ReadyItem<'_>],
        max_buffer_size: u64,
    ) {
        let mut size = 0usize;
        let mut count = 0usize;
        if self.budget.max_upload_staging_bytes != 0 {
            for (item, cached) in ready.iter().zip(&self.slots) {
                if let Some((key, bytes)) = binding_description(item)
                    && let Some(cached) = cached
                    && cached.key.matches(key)
                    && cached.len == bytes.len()
                    && cached.bytes[..cached.len] != *bytes
                {
                    let Some(next) = size.checked_add(bytes.len()) else {
                        self.uniform_uploads.begin();
                        return;
                    };
                    size = next;
                    count += 1;
                }
            }
        }
        // Failure to reserve optional scratch falls back to the proven direct
        // path. No slot snapshot or GPU contents change during this preflight.
        self.uniform_uploads.prepare(
            size,
            count,
            self.budget.max_upload_staging_bytes,
            max_buffer_size,
        );
    }

    pub(super) fn flush_uniform_uploads(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &[FrameBinding],
    ) {
        let report = self.uniform_uploads.flush(device, queue, encoder, bindings);
        self.statistics.created_upload_staging_buffers += report.created_buffers;
        self.statistics.upload_staging_bytes = report.buffer_bytes;
        self.statistics.peak_upload_staging_bytes = self
            .statistics
            .peak_upload_staging_bytes
            .max(report.peak_buffer_bytes);
        self.statistics.batched_uniform_copies += report.copy_count;
        self.statistics.batched_uniform_bytes += report.uploaded_bytes;
        self.statistics.uniform_upload_calls += usize::from(report.uploaded_bytes != 0);
    }

    fn other_owns_texture(&self, slot: usize, identity: &Arc<()>) -> bool {
        self.slots.iter().enumerate().any(|(index, cached)| {
            index != slot
                && cached
                    .as_ref()
                    .and_then(|cached| cached.key.texture())
                    .is_some_and(|(other, _)| Arc::ptr_eq(identity, other))
        })
    }

    fn remove_slot(&mut self, index: usize) {
        if let Some(old) = self.slots[index].take() {
            self.statistics.uniform_bytes = self
                .statistics
                .uniform_bytes
                .saturating_sub(old.binding._buffer.size() as usize);
            if let Some((identity, bytes)) = old.key.texture()
                && !self.other_owns_texture(index, identity)
            {
                self.statistics.texture_bytes = self.statistics.texture_bytes.saturating_sub(bytes);
            }
        }
    }

    pub(super) fn binding<'a>(
        &mut self,
        renderer: &mut WgpuRenderer,
        item: &'a ReadyItem<'_>,
        slot: usize,
        sharing: &mut FrameBindingSharing<'a>,
        bindings: &[FrameBinding],
    ) -> FrameBinding {
        let description = binding_description(item);
        if let Some(binding) =
            self.reuse_binding(&renderer.queue, description, slot, sharing, bindings)
        {
            return binding;
        }
        let binding = create_frame_binding(renderer, item);
        self.retain_new_binding(binding, description, slot, sharing)
    }

    fn reuse_binding<'a>(
        &mut self,
        queue: &wgpu::Queue,
        description: Option<BindingDescription<'a>>,
        slot: usize,
        sharing: &mut FrameBindingSharing<'a>,
        bindings: &[FrameBinding],
    ) -> Option<FrameBinding> {
        if let Some((key, bytes)) = &description
            && let Some(Some(cached)) = self.slots.get_mut(slot)
            && cached.key.matches(*key)
            && cached.len == bytes.len()
        {
            if &cached.bytes[..cached.len] != *bytes {
                if !self.uniform_uploads.stage(slot, bytes) {
                    queue.write_buffer(&cached.binding._buffer, 0, bytes);
                    self.statistics.uniform_upload_calls += 1;
                }
                cached.bytes[..cached.len].copy_from_slice(bytes);
                self.statistics.uploaded_uniform_bytes += bytes.len();
            }
            self.statistics.reused_bind_groups += 1;
            sharing.remember(description, slot);
            return Some(cached.binding.clone());
        }
        if let Some(description) = description
            && let Some(index) = sharing.find(description)
            && let Some(binding) = bindings.get(index)
        {
            // Do not publish this clone into `slots[slot]`: a later frame may
            // write a different uniform into either independent item slot.
            self.statistics.reused_bind_groups += 1;
            self.statistics.shared_bind_groups += 1;
            sharing.remember(Some(description), slot);
            return Some(binding.clone());
        }
        None
    }

    fn retain_new_binding<'a>(
        &mut self,
        binding: FrameBinding,
        description: Option<BindingDescription<'a>>,
        slot: usize,
        sharing: &mut FrameBindingSharing<'a>,
    ) -> FrameBinding {
        sharing.remember(description, slot);
        let uniform_size = binding._buffer.size() as usize;
        self.statistics.created_buffers += 1;
        self.statistics.created_bind_groups += 1;
        self.statistics.uploaded_uniform_bytes += uniform_size;
        self.statistics.uniform_upload_calls += 1;
        self.statistics.peak_uniform_bytes = self
            .statistics
            .peak_uniform_bytes
            .saturating_add(uniform_size);
        let Some((key, bytes)) = description else {
            return binding;
        };
        if slot >= self.slots.len() {
            return binding;
        }
        let new_texture = key.texture().map_or(0, |(identity, size)| {
            if self.other_owns_texture(slot, identity)
                || self.slots[slot]
                    .as_ref()
                    .and_then(|cached| cached.key.texture())
                    .is_some_and(|(old, _)| Arc::ptr_eq(identity, old))
            {
                0
            } else {
                size
            }
        });
        let retained_uniform = self
            .statistics
            .uniform_bytes
            .saturating_sub(
                self.slots[slot]
                    .as_ref()
                    .map_or(0, |old| old.binding._buffer.size() as usize),
            )
            .saturating_add(uniform_size);
        let texture_peak = self.statistics.texture_bytes.saturating_add(new_texture);
        self.statistics.peak_texture_bytes = self.statistics.peak_texture_bytes.max(texture_peak);
        if retained_uniform > self.budget.max_uniform_bytes
            || texture_peak > self.budget.max_texture_bytes
        {
            self.remove_slot(slot);
            return binding;
        }
        self.remove_slot(slot);
        if let Some((identity, size)) = key.texture()
            && !self.other_owns_texture(slot, identity)
        {
            self.statistics.texture_bytes = self.statistics.texture_bytes.saturating_add(size);
        }
        self.statistics.uniform_bytes = self.statistics.uniform_bytes.saturating_add(uniform_size);
        let mut stored = [0; std::mem::size_of::<ImageUniform>()];
        stored[..bytes.len()].copy_from_slice(bytes);
        self.slots[slot] = Some(CachedBinding {
            key: key.into_owned(),
            binding: binding.clone(),
            bytes: stored,
            len: bytes.len(),
        });
        binding
    }
}

fn binding_description<'a>(item: &'a ReadyItem<'_>) -> Option<(BindingKeyRef<'a>, &'a [u8])> {
    match item {
        ReadyItem::Geometry(geometry) => Some((
            BindingKeyRef::Camera,
            bytemuck::bytes_of(&geometry.camera_uniform),
        )),
        ReadyItem::Particle { camera_uniform, .. } => {
            Some((BindingKeyRef::Camera, bytemuck::bytes_of(camera_uniform)))
        }
        ReadyItem::Image {
            image,
            uniform,
            sampling,
            ..
        }
        | ReadyItem::ImageBatch {
            image,
            uniform,
            sampling,
            ..
        } => Some((
            BindingKeyRef::Image {
                identity: &image.resource_identity,
                sampling: *sampling,
                texture_bytes: image.gpu_allocation_bytes(),
            },
            bytemuck::bytes_of(uniform),
        )),
        ReadyItem::Target {
            target, uniform, ..
        } => Some((
            BindingKeyRef::Target {
                identity: &target.resource_identity,
                texture_bytes: target.allocation_bytes,
            },
            bytemuck::bytes_of(uniform),
        )),
        ReadyItem::Scalar { .. } => None,
    }
}

fn vec_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity().saturating_mul(std::mem::size_of::<T>())
}

// The standard library's in-place Vec collection recycles equal-layout storage.
// Every element is dropped, so no borrowed scene/particle reference survives.
// Unlike lifetime casts this remains memory-safe if collection ever reallocates.
#[allow(
    clippy::unnecessary_filter_map,
    reason = "filter cannot change the empty Vec's element lifetime"
)]
pub(super) fn recycle_items<'source, 'target>(
    values: Vec<FrameItem<'source>>,
) -> Vec<FrameItem<'target>> {
    values
        .into_iter()
        .filter_map(|_| None::<FrameItem<'target>>)
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unnecessary_filter_map,
    reason = "filter cannot change the empty Vec's element lifetime"
)]
pub(super) fn recycle_ready<'source, 'target>(
    values: Vec<ReadyItem<'source>>,
) -> Vec<ReadyItem<'target>> {
    values
        .into_iter()
        .filter_map(|_| None::<ReadyItem<'target>>)
        .collect()
}

pub(super) fn recycle_ready_with_batches<'source, 'target>(
    values: Vec<ReadyItem<'source>>,
    batches: &mut Vec<Vec<PreparedDrawBatch>>,
) -> Vec<ReadyItem<'target>> {
    values
        .into_iter()
        .filter_map(|value| {
            if let ReadyItem::Geometry(mut geometry) = value
                && batches.len() < batches.capacity()
            {
                geometry.batches.clear();
                batches.push(geometry.batches);
            }
            None::<ReadyItem<'target>>
        })
        .collect()
}

impl WgpuRenderer {
    /// Returns renderer-owned idle composition cache limits.
    pub fn frame_cache_budget(&self) -> FrameCacheBudget {
        self.frame_cache.budget
    }
    /// Changes limits and releases the old cache. In-flight resource retirement
    /// is handled by wgpu; the next frame warms storage under the new limits.
    pub fn set_frame_cache_budget(&mut self, budget: FrameCacheBudget) {
        self.frame_cache.clear();
        self.frame_cache.budget = budget;
    }
    /// Releases cached composition allocations and GPU resource references.
    pub fn clear_frame_cache(&mut self) {
        self.frame_cache.clear();
    }
    /// Returns idle retained bytes and the previous composed frame's cache work.
    /// This does not include queue/backend allocations, source geometry, or LUTs.
    pub fn frame_cache_statistics(&self) -> FrameCacheStatistics {
        self.frame_cache.statistics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_lifetime_recycling_retains_allocations_and_releases_mutable_borrows() {
        let scene = Scene::new(Color::BLACK).unwrap();
        let mut items = Vec::with_capacity(8);
        items.push(FrameItem::Scene {
            scene: &scene,
            camera: Camera2d::default(),
            options: FramePassOptions::default(),
            insertion: 0,
        });
        let pointer = items.as_ptr().cast::<()>();
        let recycled: Vec<FrameItem<'static>> = recycle_items(items);
        assert!(recycled.is_empty());
        assert_eq!(recycled.capacity(), 8);
        assert_eq!(recycled.as_ptr().cast::<()>(), pointer);
        let ready: Vec<ReadyItem<'_>> = Vec::with_capacity(8);
        let pointer = ready.as_ptr().cast::<()>();
        let recycled: Vec<ReadyItem<'static>> = recycle_ready(ready);
        assert_eq!(recycled.capacity(), 8);
        assert_eq!(recycled.as_ptr().cast::<()>(), pointer);
    }

    #[test]
    fn zero_cache_budget_drops_idle_storage() {
        let mut cache = FrameCache {
            budget: FrameCacheBudget::new(0, 0, 0, 0),
            ..FrameCache::default()
        };
        cache.items.reserve(4);
        cache.retained_resources.reserve(4);
        cache.finish();
        assert_eq!(cache.statistics.cpu_bytes(), 0);
        assert!(cache.statistics.peak_cpu_bytes() > 0);
    }

    #[test]
    fn late_transform_failure_accounts_grown_batches_before_idle_eviction() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        for index in 0..256 {
            scene
                .set_screen_clip(Some(
                    ScreenClipRect::from_min_size(
                        LogicalScreenPosition::new((index % 2) as f32, 0.0),
                        crate::LogicalScreenVector::new(32.0, 32.0),
                    )
                    .unwrap(),
                ))
                .unwrap();
            scene
                .try_line(
                    Vec2::new(1.0e30, 0.0),
                    Vec2::new(1.1e30, 0.0),
                    1.0,
                    Color::WHITE,
                )
                .unwrap();
        }
        let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
        let camera = Camera2d::new(Vec2::ZERO, 1.0e10).unwrap();
        let uniform = CameraUniform::new(camera, viewport).unwrap();
        let resolved = ResolvedViewport {
            viewport,
            origin: Vec2::ZERO,
            scissor: ScissorRect {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            },
            item_clip: None,
            item_clipped_out: false,
        };
        for budget in [
            FrameCacheBudget::default(),
            FrameCacheBudget::new(0, 0, 0, 0),
        ] {
            let mut cache = FrameCache {
                budget,
                ..FrameCache::default()
            };
            cache.begin();
            // Production reserves the outer pool for the complete frame first.
            cache.batches.try_reserve(1).unwrap();
            let mut vertices = Vec::new();
            let mut ready = Vec::new();
            let mut statistics = FrameStatistics::default();
            let mut aggregate = TessellationStats::default();
            let result = with_streaming_batches(&mut cache.batches, |batches| {
                prepare_streaming_scene_resolved(
                    &scene,
                    uniform,
                    resolved,
                    &mut vertices,
                    &mut ready,
                    &mut statistics,
                    &mut aggregate,
                    batches,
                )
            });
            assert_eq!(
                result,
                Err(RendererFrameError::InvalidGeometryTransform.into())
            );
            assert!(vertices.is_empty() && ready.is_empty());
            assert_eq!(statistics, FrameStatistics::default());
            assert_eq!(aggregate, TessellationStats::default());
            assert_eq!(cache.batches.len(), 1);
            assert!(cache.batches[0].is_empty());
            assert!(cache.batches[0].capacity() >= scene.command_count());
            let owned = cache.cpu_bytes();
            assert!(owned >= scene.command_count() * std::mem::size_of::<PreparedDrawBatch>());
            cache.finish();
            assert!(cache.statistics.peak_cpu_bytes() >= owned * 2);
            if budget.max_cpu_bytes() == 0 {
                assert_eq!(cache.statistics.cpu_bytes(), 0);
                assert!(cache.batches.is_empty());
            } else {
                assert_eq!(cache.statistics.cpu_bytes(), owned);
            }
        }
    }

    #[test]
    fn cache_retention_honors_exact_aggregate_bytes_and_slot_limit() {
        let mut cache = FrameCache::default();
        cache.items.reserve(4);
        cache.retained_resources.reserve(4);
        let exact_bytes = cache.cpu_bytes();
        cache.budget = FrameCacheBudget::new(exact_bytes, 0, 0, 0);
        cache.finish();
        assert_eq!(cache.statistics.cpu_bytes(), exact_bytes);
        cache.budget = FrameCacheBudget::new(exact_bytes - 1, 0, 0, 0);
        cache.finish();
        assert_eq!(cache.statistics.cpu_bytes(), 0);

        cache.budget = FrameCacheBudget::new(4096, 256, 4096, 2);
        cache.reserve_slots(3).unwrap();
        assert_eq!(cache.slots.len(), 2);
        assert!(cache.slots.iter().all(Option::is_none));
    }

    #[test]
    fn bindings_compare_resource_provenance_and_sampling_not_content() {
        let identity = Arc::new(());
        let nearest = BindingKey::Image {
            identity: Arc::clone(&identity),
            sampling: ImageSampling::Nearest,
            texture_bytes: 4,
        };
        assert!(nearest.matches(nearest.as_ref()));
        assert!(!nearest.matches(BindingKeyRef::Image {
            identity: &Arc::new(()),
            sampling: ImageSampling::Nearest,
            texture_bytes: 4,
        }));
        assert!(!nearest.matches(BindingKeyRef::Image {
            identity: &identity,
            sampling: ImageSampling::Linear,
            texture_bytes: 4,
        }));
        assert!(!nearest.matches(BindingKeyRef::Target {
            identity: &identity,
            texture_bytes: 4
        }));
    }

    #[test]
    fn geometry_recycling_preserves_batch_and_ready_storage() {
        let viewport = LogicalViewport::new(32.0, 32.0).unwrap();
        let mut batches = Vec::with_capacity(4);
        batches.push(PreparedDrawBatch {
            vertex_range: 0..3,
            screen_clip: None,
        });
        let batch_pointer = batches.as_ptr();
        let mut ready = Vec::with_capacity(4);
        ready.push(ReadyItem::Geometry(ReadyGeometry {
            source: ReadySource::Streaming,
            vertex_count: 3,
            batches,
            camera_uniform: CameraUniform::new(Camera2d::default(), viewport).unwrap(),
            viewport: ResolvedViewport {
                viewport,
                origin: Vec2::ZERO,
                scissor: ScissorRect {
                    x: 0,
                    y: 0,
                    width: 32,
                    height: 32,
                },
                item_clip: None,
                item_clipped_out: false,
            },
        }));
        let ready_pointer = ready.as_ptr().cast::<()>();
        let mut pool = Vec::with_capacity(1);
        let reused: Vec<ReadyItem<'static>> = recycle_ready_with_batches(ready, &mut pool);
        assert!(reused.is_empty());
        assert_eq!(reused.as_ptr().cast::<()>(), ready_pointer);
        assert_eq!(pool.len(), 1);
        assert!(pool[0].is_empty());
        assert_eq!(pool[0].as_ptr(), batch_pointer);
    }
}
