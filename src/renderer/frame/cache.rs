use super::*;

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
}

impl FrameCacheBudget {
    /// Creates exact retention limits; zero is a valid disabled cache.
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
}

impl Default for FrameCacheBudget {
    fn default() -> Self {
        Self::new(8 * 1024 * 1024, 1024 * 1024, 256 * 1024 * 1024, 4096)
    }
}

/// Library-owned retained allocation sizes and the last composition's cache work.
/// CPU byte counts use actual Vec capacities, uniform bytes use buffer sizes,
/// and referenced texture bytes count each cached resource identity once.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FrameCacheStatistics {
    pub(super) cpu_bytes: usize,
    pub(super) uniform_bytes: usize,
    pub(super) texture_bytes: usize,
    pub(super) binding_count: usize,
    pub(super) created_buffers: usize,
    pub(super) created_bind_groups: usize,
    pub(super) reused_bind_groups: usize,
    pub(super) uploaded_uniform_bytes: usize,
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
    /// Existing bind groups reused by the previous composed frame.
    pub const fn reused_bind_groups(self) -> usize {
        self.reused_bind_groups
    }
    /// Uniform bytes actually written by the previous composed frame.
    pub const fn uploaded_uniform_bytes(self) -> usize {
        self.uploaded_uniform_bytes
    }
    /// Conservative old-plus-new owned allocation peak; excludes driver staging.
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

impl BindingKey {
    fn texture(&self) -> Option<(&Arc<()>, usize)> {
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
            } => Some((identity, *texture_bytes)),
        }
    }
    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Camera, Self::Camera) => true,
            (
                Self::Image {
                    identity: a,
                    sampling: sa,
                    ..
                },
                Self::Image {
                    identity: b,
                    sampling: sb,
                    ..
                },
            ) => Arc::ptr_eq(a, b) && sa == sb,
            (Self::Target { identity: a, .. }, Self::Target { identity: b, .. }) => {
                Arc::ptr_eq(a, b)
            }
            _ => false,
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
    budget: FrameCacheBudget,
    statistics: FrameCacheStatistics,
    initial_cpu_bytes: usize,
}

impl FrameCache {
    pub(super) fn uploaded_uniform_bytes(&self) -> usize {
        self.statistics.uploaded_uniform_bytes
    }
    pub(super) fn begin(&mut self) {
        self.initial_cpu_bytes = self.cpu_bytes();
        self.statistics.created_buffers = 0;
        self.statistics.created_bind_groups = 0;
        self.statistics.reused_bind_groups = 0;
        self.statistics.uploaded_uniform_bytes = 0;
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
    }

    pub(super) fn finish(&mut self) {
        self.statistics.peak_cpu_bytes = self.initial_cpu_bytes.saturating_add(self.cpu_bytes());
        if self.cpu_bytes() > self.budget.max_cpu_bytes {
            self.items = Vec::new();
            self.retained_resources = Vec::new();
            self.scalar_luts = Vec::new();
            self.ready = Vec::new();
            self.bindings = Vec::new();
            self.batches = Vec::new();
        }
        if self.cpu_bytes() > self.budget.max_cpu_bytes {
            self.slots = Vec::new();
            self.statistics.uniform_bytes = 0;
            self.statistics.texture_bytes = 0;
        }
        self.statistics.cpu_bytes = self.cpu_bytes();
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

    pub(super) fn binding(
        &mut self,
        renderer: &mut WgpuRenderer,
        item: &ReadyItem<'_>,
        slot: usize,
    ) -> FrameBinding {
        let description = binding_description(item);
        if let Some((key, bytes)) = &description
            && let Some(Some(cached)) = self.slots.get_mut(slot)
            && cached.key.matches(key)
            && cached.len == bytes.len()
        {
            if &cached.bytes[..cached.len] != *bytes {
                renderer
                    .queue
                    .write_buffer(&cached.binding._buffer, 0, bytes);
                cached.bytes[..cached.len].copy_from_slice(bytes);
                self.statistics.uploaded_uniform_bytes += bytes.len();
            }
            self.statistics.reused_bind_groups += 1;
            return cached.binding.clone();
        }
        let binding = create_frame_binding(renderer, item);
        let uniform_size = binding._buffer.size() as usize;
        self.statistics.created_buffers += 1;
        self.statistics.created_bind_groups += 1;
        self.statistics.uploaded_uniform_bytes += uniform_size;
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
            key,
            binding: binding.clone(),
            bytes: stored,
            len: bytes.len(),
        });
        binding
    }
}

fn binding_description<'a>(item: &'a ReadyItem<'_>) -> Option<(BindingKey, &'a [u8])> {
    match item {
        ReadyItem::Geometry(geometry) => Some((
            BindingKey::Camera,
            bytemuck::bytes_of(&geometry.camera_uniform),
        )),
        ReadyItem::Particle { camera_uniform, .. } => {
            Some((BindingKey::Camera, bytemuck::bytes_of(camera_uniform)))
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
            BindingKey::Image {
                identity: Arc::clone(&image.resource_identity),
                sampling: *sampling,
                texture_bytes: image.gpu_allocation_bytes(),
            },
            bytemuck::bytes_of(uniform),
        )),
        ReadyItem::Target {
            target, uniform, ..
        } => Some((
            BindingKey::Target {
                identity: Arc::clone(&target.resource_identity),
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
        assert!(nearest.matches(&nearest.clone()));
        assert!(!nearest.matches(&BindingKey::Image {
            identity: Arc::new(()),
            sampling: ImageSampling::Nearest,
            texture_bytes: 4,
        }));
        assert!(!nearest.matches(&BindingKey::Image {
            identity: Arc::clone(&identity),
            sampling: ImageSampling::Linear,
            texture_bytes: 4,
        }));
        assert!(!nearest.matches(&BindingKey::Target {
            identity,
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
