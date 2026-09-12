//! Opt-in, bounded render-pass timestamp collection without GPU waits.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

const CAPACITY: usize = 8;
const QUERY_BYTES: u64 = 16;
const FREE: u8 = 0;
const RESERVED: u8 = 1;
const MAPPING: u8 = 2;
const READY: u8 = 3;
const FAILED: u8 = 4;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn requested_features(available: wgpu::Features, requested: bool) -> wgpu::Features {
    if requested {
        available & wgpu::Features::TIMESTAMP_QUERY
    } else {
        wgpu::Features::empty()
    }
}

#[cfg(test)]
pub(super) use tests::assert_gpu_timing_contract;

/// Availability of the explicitly requested render-pass GPU diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTimingStatus {
    /// Timing was not requested; no queries or readback resources are created.
    Disabled,
    /// Timing was requested but is unavailable on the current logical device.
    /// Unsupported features, an unusable timestamp period or a polling failure
    /// produce this state. CPU submission/wait time is never substituted.
    Unavailable,
    /// Timestamp queries are active; individual samples can still be dropped
    /// when the fixed ring is full or rejected when readback fails.
    Enabled,
}

/// Render pass measured by a sample, not CPU uploads or presentation latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTimingSource {
    /// A retained 3D scene's offscreen surface and mathematical-edge pass.
    Scene3d,
    /// A heterogeneous composed frame's surface render pass, including MSAA
    /// resolution but excluding acquisition, scanout and separate 3D passes.
    FrameComposer,
}

/// Opaque correlation key returned by a submitted render report.
///
/// Keys are monotonic and never reused across renderer instances or recovery.
/// A key identifies a submitted pass, not confirmed monitor scanout. Match the
/// complete key to its report when excluding warmup or unrelated work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuTimingId(u64);

impl GpuTimingId {
    /// Returns the process-local sequence value for diagnostics and logs.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One completed GPU render-pass interval obtained from hardware timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuTimingSample {
    id: GpuTimingId,
    source: GpuTimingSource,
    elapsed: Duration,
}

impl GpuTimingSample {
    /// Returns the originating submitted report's correlation key.
    pub const fn id(self) -> GpuTimingId {
        self.id
    }

    /// Returns the measured pass category.
    pub const fn source(self) -> GpuTimingSource {
        self.source
    }

    /// Returns `(end_tick - begin_tick) * timestamp_period`, rounded to the
    /// nearest representable nanosecond. Timestamp resolution is backend/device
    /// dependent; a zero interval is possible below that resolution. Uploads,
    /// query resolution/readback, queue waits and presentation are excluded.
    pub const fn elapsed(self) -> Duration {
        self.elapsed
    }
}

/// Fixed-capacity result of one nonblocking GPU-diagnostics collection.
///
/// At most eight completed samples are returned, ordered by their correlation
/// keys. Later submissions may complete first; match IDs rather than treating
/// collection order as frame order. This value allocates no sample array.
#[derive(Debug, Clone, Copy)]
pub struct GpuTimingBatch {
    samples: [GpuTimingSample; CAPACITY],
    count: usize,
    collection_cpu: Duration,
}

impl GpuTimingBatch {
    fn empty() -> Self {
        Self {
            samples: [GpuTimingSample {
                id: GpuTimingId(0),
                source: GpuTimingSource::Scene3d,
                elapsed: Duration::ZERO,
            }; CAPACITY],
            count: 0,
            collection_cpu: Duration::ZERO,
        }
    }

    /// Returns completed valid samples from this collection only.
    pub fn samples(&self) -> &[GpuTimingSample] {
        &self.samples[..self.count]
    }

    /// Returns CPU time spent polling once, copying up to 128 timestamp bytes,
    /// validating intervals and unmapping. The poll does not wait for the GPU,
    /// but may perform backend maintenance or invoke other ready callbacks.
    pub const fn collection_cpu(self) -> Duration {
        self.collection_cpu
    }
}

/// Current-device bounded diagnostics resource and loss accounting.
///
/// Counters reset during renderer recovery; previously pending samples are
/// discarded. Query/resolve/map operations add measurement overhead. Buffer
/// bytes below are nominal, excluding driver alignment, query storage and
/// callback/backend allocation overhead. No render-thread GPU waits occur.
#[derive(Debug, Clone, Copy)]
pub struct GpuTimingStatistics {
    status: GpuTimingStatus,
    pending: usize,
    submitted: u64,
    completed: u64,
    dropped: u64,
    map_failures: u64,
    invalid_samples: u64,
    poll_failures: u64,
    collection_cpu: Duration,
    timestamp_period_nanoseconds: Option<f64>,
}

impl GpuTimingStatistics {
    /// Returns whether timing was requested and is usable on this device.
    pub const fn status(self) -> GpuTimingStatus {
        self.status
    }

    /// Maximum simultaneously reserved/submitted samples; zero when unavailable.
    pub const fn capacity(self) -> usize {
        if matches!(self.status, GpuTimingStatus::Enabled) {
            CAPACITY
        } else {
            0
        }
    }

    /// Reserved, mapping or completed-but-not-yet-collected ring entries.
    pub const fn pending(self) -> usize {
        self.pending
    }

    /// Submitted timestamp intervals, including eventual failures or invalid data.
    pub const fn submitted(self) -> u64 {
        self.submitted
    }

    /// Valid samples delivered through collection batches.
    pub const fn completed(self) -> u64 {
        self.completed
    }

    /// Passes not sampled because all slots were occupied or IDs were exhausted.
    /// Disabled/unavailable timing does not increment this counter.
    pub const fn dropped(self) -> u64 {
        self.dropped
    }

    /// Failed asynchronous maps or mapped-range access attempts.
    pub const fn map_failures(self) -> u64 {
        self.map_failures
    }

    /// Rejected intervals, including end-before-start/wrapped timestamps and
    /// durations outside the representable range. No zero substitute is emitted.
    pub const fn invalid_samples(self) -> u64 {
        self.invalid_samples
    }

    /// Device-poll failures; these disable the ring until renderer recovery.
    pub const fn poll_failures(self) -> u64 {
        self.poll_failures
    }

    /// Nominal bytes in eight 16-byte resolve and eight 16-byte readback buffers.
    pub const fn gpu_buffer_bytes(self) -> usize {
        self.capacity() * QUERY_BYTES as usize * 2
    }

    /// Number of timestamp query slots; query-set driver storage is not exposed.
    pub const fn query_count(self) -> usize {
        self.capacity() * 2
    }

    /// Bytes queued for query resolution and readback copies over submitted
    /// samples. Each sample resolves and copies exactly two 64-bit timestamps;
    /// these diagnostic transfers are separate from scene upload statistics.
    pub const fn readback_bytes(self) -> u64 {
        self.submitted.saturating_mul(QUERY_BYTES)
    }

    /// Sum of collection CPU costs, separate from measured GPU pass intervals.
    pub const fn collection_cpu(self) -> Duration {
        self.collection_cpu
    }

    /// Nanoseconds per raw query tick, or `None` when timing is unavailable.
    pub const fn timestamp_period_nanoseconds(self) -> Option<f64> {
        self.timestamp_period_nanoseconds
    }
}

struct Slot {
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    state: Arc<AtomicU8>,
    request: Option<(GpuTimingId, GpuTimingSource)>,
}

struct Ring {
    queries: wgpu::QuerySet,
    slots: [Slot; CAPACITY],
    period: f64,
}

pub(super) struct GpuTimingCollector {
    ring: Option<Ring>,
    statistics: GpuTimingStatistics,
}

impl GpuTimingCollector {
    /// A fixed ring is only constructed after optional TIMESTAMP_QUERY has been
    /// requested on the actual device. Unrequested timing has no GPU resource cost.
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue, requested: bool) -> Self {
        let feature_enabled =
            requested && device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let period = if feature_enabled {
            f64::from(queue.get_timestamp_period())
        } else {
            0.0
        };
        let enabled = feature_enabled && period.is_finite() && period > 0.0;
        let status = if enabled {
            GpuTimingStatus::Enabled
        } else if requested {
            GpuTimingStatus::Unavailable
        } else {
            GpuTimingStatus::Disabled
        };
        let ring = enabled.then(|| Ring {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("sim-engine bounded GPU timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: (CAPACITY * 2) as u32,
            }),
            slots: std::array::from_fn(|_| Slot {
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sim-engine GPU timestamp resolve"),
                    size: QUERY_BYTES,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sim-engine GPU timestamp readback"),
                    size: QUERY_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(FREE)),
                request: None,
            }),
            period,
        });
        Self {
            ring,
            statistics: GpuTimingStatistics {
                status,
                pending: 0,
                submitted: 0,
                completed: 0,
                dropped: 0,
                map_failures: 0,
                invalid_samples: 0,
                poll_failures: 0,
                collection_cpu: Duration::ZERO,
                timestamp_period_nanoseconds: enabled.then_some(period),
            },
        }
    }

    pub(super) fn reserve(&mut self, source: GpuTimingSource) -> Option<GpuTimingReservation> {
        let ring = self.ring.as_mut()?;
        let Some(index) = ring
            .slots
            .iter()
            .position(|slot| slot.state.load(Ordering::Acquire) == FREE)
        else {
            self.statistics.dropped = self.statistics.dropped.saturating_add(1);
            return None;
        };
        let Ok(sequence) = NEXT_ID.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        }) else {
            self.statistics.dropped = self.statistics.dropped.saturating_add(1);
            return None;
        };
        let id = GpuTimingId(sequence);
        let slot = &mut ring.slots[index];
        slot.request = Some((id, source));
        slot.state.store(RESERVED, Ordering::Release);
        Some(GpuTimingReservation {
            queries: ring.queries.clone(),
            resolve: slot.resolve.clone(),
            readback: slot.readback.clone(),
            state: Arc::clone(&slot.state),
            query_start: (index * 2) as u32,
            id,
            submitted: false,
        })
    }

    /// Called immediately after queue submission, before exposing the report ID.
    pub(super) fn submitted(&mut self, reservation: GpuTimingReservation) -> GpuTimingId {
        self.statistics.submitted = self.statistics.submitted.saturating_add(1);
        reservation.submitted()
    }

    pub(super) fn statistics(&self) -> GpuTimingStatistics {
        let mut statistics = self.statistics;
        statistics.pending = self.ring.as_ref().map_or(0, |ring| {
            ring.slots
                .iter()
                .filter(|slot| slot.state.load(Ordering::Acquire) != FREE)
                .count()
        });
        statistics
    }

    pub(super) fn collect(&mut self, device: &wgpu::Device) -> GpuTimingBatch {
        let mut batch = GpuTimingBatch::empty();
        if self.ring.is_none() {
            return batch;
        }
        let started = Instant::now();
        if device.poll(wgpu::PollType::Poll).is_err() {
            self.statistics.poll_failures = self.statistics.poll_failures.saturating_add(1);
            self.statistics.status = GpuTimingStatus::Unavailable;
            self.statistics.timestamp_period_nanoseconds = None;
            // A late old-device callback owns only its old atomic cell. Dropping
            // the complete ring prevents it from completing a reused request.
            self.ring = None;
        } else if let Some(ring) = &mut self.ring {
            for slot in &mut ring.slots {
                match slot.state.load(Ordering::Acquire) {
                    READY => {
                        match slot.readback.get_mapped_range(0..QUERY_BYTES) {
                            Ok(bytes) => {
                                if let Some(elapsed) = elapsed_from_bytes(&bytes, ring.period) {
                                    if let Some((id, source)) = slot.request {
                                        batch.samples[batch.count] = GpuTimingSample {
                                            id,
                                            source,
                                            elapsed,
                                        };
                                        batch.count += 1;
                                        self.statistics.completed =
                                            self.statistics.completed.saturating_add(1);
                                    }
                                } else {
                                    self.statistics.invalid_samples =
                                        self.statistics.invalid_samples.saturating_add(1);
                                }
                                // Release the borrowed view before unmapping.
                                // Failed/cancelled mappings are already unmapped;
                                // calling unmap there produces a validation error.
                                drop(bytes);
                                slot.readback.unmap();
                            }
                            Err(_) => {
                                self.statistics.map_failures =
                                    self.statistics.map_failures.saturating_add(1);
                            }
                        }
                        slot.request = None;
                        slot.state.store(FREE, Ordering::Release);
                    }
                    FAILED => {
                        self.statistics.map_failures =
                            self.statistics.map_failures.saturating_add(1);
                        slot.request = None;
                        slot.state.store(FREE, Ordering::Release);
                    }
                    _ => {}
                }
            }
        }
        batch.samples[..batch.count].sort_unstable_by_key(|sample| sample.id);
        batch.collection_cpu = started.elapsed();
        self.statistics.collection_cpu = self
            .statistics
            .collection_cpu
            .saturating_add(batch.collection_cpu);
        batch
    }
}

/// Owns its slot until cancellation or submission. Only fixed-size GPU handles
/// are cloned; the callback captures one bounded completion cell, never a queue.
pub(super) struct GpuTimingReservation {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    state: Arc<AtomicU8>,
    query_start: u32,
    id: GpuTimingId,
    submitted: bool,
}

impl GpuTimingReservation {
    pub(super) fn timestamp_writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(self.query_start),
            end_of_pass_write_index: Some(self.query_start + 1),
        }
    }

    /// Called after the timed pass ends, in the same encoder. Individual resolve
    /// buffers use offset zero, satisfying the stricter query-resolve alignment.
    pub(super) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.queries,
            self.query_start..self.query_start + 2,
            &self.resolve,
            0,
        );
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, QUERY_BYTES);
    }

    fn submitted(mut self) -> GpuTimingId {
        self.submitted = true;
        self.state.store(MAPPING, Ordering::Release);
        let state = Arc::clone(&self.state);
        self.readback
            .map_async(wgpu::MapMode::Read, 0..QUERY_BYTES, move |result| {
                state.store(
                    if result.is_ok() { READY } else { FAILED },
                    Ordering::Release,
                );
            });
        self.id
    }
}

impl Drop for GpuTimingReservation {
    fn drop(&mut self) {
        if !self.submitted {
            self.state.store(FREE, Ordering::Release);
        }
    }
}

fn elapsed_from_bytes(bytes: &[u8], period: f64) -> Option<Duration> {
    let begin = u64::from_ne_bytes(bytes.get(..8)?.try_into().ok()?);
    let end = u64::from_ne_bytes(bytes.get(8..16)?.try_into().ok()?);
    let ticks = end.checked_sub(begin)?;
    if !period.is_finite() || period <= 0.0 {
        return None;
    }
    // Round in nanoseconds first: converting an exact half-nanosecond through
    // fractional seconds can shift the tie downward through binary rounding.
    let nanoseconds = (ticks as f64 * period).round();
    Duration::try_from_secs_f64(nanoseconds / 1_000_000_000.0).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(begin: u64, end: u64) -> [u8; 16] {
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&begin.to_ne_bytes());
        bytes[8..].copy_from_slice(&end.to_ne_bytes());
        bytes
    }

    #[test]
    fn timestamp_conversion_subtracts_integer_ticks_before_float_rounding() {
        let bytes = interval(u64::MAX - 10, u64::MAX);
        assert_eq!(
            elapsed_from_bytes(&bytes, 2.5),
            Some(Duration::from_nanos(25))
        );
        assert_eq!(
            elapsed_from_bytes(&interval(100, 103), 0.5),
            Some(Duration::from_nanos(2))
        );
        assert_eq!(
            elapsed_from_bytes(&interval(5, 5), 1.0),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn timestamp_conversion_rejects_invalid_data_instead_of_fabricating_zero() {
        assert_eq!(elapsed_from_bytes(&[0; 15], 1.0), None);
        assert_eq!(elapsed_from_bytes(&interval(10, 2), 1.0), None);
        for period in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(elapsed_from_bytes(&interval(10, 20), period), None);
        }
        assert_eq!(elapsed_from_bytes(&interval(0, u64::MAX), f64::MAX), None);
    }

    #[test]
    fn empty_collection_has_bounded_inline_storage_and_no_reported_time() {
        let batch = GpuTimingBatch::empty();
        assert!(batch.samples().is_empty());
        assert_eq!(batch.collection_cpu(), Duration::ZERO);
        assert_eq!(batch.samples.len(), CAPACITY);
    }

    /// Root calls this with a separately requested TIMESTAMP_QUERY device when
    /// supported, and may also call it with its ordinary feature-disabled device.
    /// The fixture never requests an adapter/window or changes a runtime option.
    pub(in crate::renderer) fn assert_gpu_timing_contract(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let ((mut disabled, unavailable_request), allocations) =
            crate::test_allocations::count(|| {
                let mut disabled = GpuTimingCollector::new(device, queue, false);
                let unavailable_request = disabled.reserve(GpuTimingSource::Scene3d);
                assert!(disabled.collect(device).samples().is_empty());
                (disabled, unavailable_request)
            });
        assert_eq!(
            allocations, 0,
            "disabled GPU diagnostics must allocate nothing"
        );
        assert!(unavailable_request.is_none());
        assert_eq!(disabled.statistics().status(), GpuTimingStatus::Disabled);
        assert_eq!(disabled.statistics().capacity(), 0);
        assert_eq!(disabled.statistics().gpu_buffer_bytes(), 0);
        assert_eq!(disabled.statistics().query_count(), 0);
        assert_eq!(disabled.statistics().readback_bytes(), 0);
        assert_eq!(disabled.statistics().collection_cpu(), Duration::ZERO);
        assert_eq!(disabled.collect(device).collection_cpu(), Duration::ZERO);

        let mut collector = GpuTimingCollector::new(device, queue, true);
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            assert_eq!(
                collector.statistics().status(),
                GpuTimingStatus::Unavailable
            );
            assert!(collector.reserve(GpuTimingSource::FrameComposer).is_none());
            assert!(collector.collect(device).samples().is_empty());
            assert_eq!(collector.statistics().gpu_buffer_bytes(), 0);
            assert_eq!(collector.statistics().query_count(), 0);
            assert_eq!(collector.statistics().timestamp_period_nanoseconds(), None);
            assert_eq!(collector.statistics().dropped(), 0);
            eprintln!("sim-engine GPU timestamps: unavailable on feature-disabled device");
            return;
        }
        assert_eq!(collector.statistics().status(), GpuTimingStatus::Enabled);
        assert_eq!(collector.statistics().capacity(), CAPACITY);
        assert_eq!(collector.statistics().gpu_buffer_bytes(), 256);
        assert_eq!(collector.statistics().query_count(), 16);
        assert!(collector.collect(device).samples().is_empty());

        let canceled = collector.reserve(GpuTimingSource::Scene3d).unwrap();
        let canceled_id = canceled.id;
        assert_eq!(collector.statistics().pending(), 1);
        drop(canceled);
        assert_eq!(collector.statistics().pending(), 0);
        assert_eq!(collector.statistics().submitted(), 0);
        assert!(collector.collect(device).samples().is_empty());

        let fixture = Workload::new(device);
        let requests: Vec<_> = (0..CAPACITY)
            .map(|index| {
                let source = if index % 2 == 0 {
                    GpuTimingSource::Scene3d
                } else {
                    GpuTimingSource::FrameComposer
                };
                let id = fixture.submit(device, queue, &mut collector, source);
                assert!(id > canceled_id);
                (id, source)
            })
            .collect();
        // Completed-but-uncollected entries still occupy slots. No hidden poll
        // or unbounded result queue allows a ninth sample to overwrite them.
        assert_eq!(collector.statistics().pending(), CAPACITY);
        assert!(collector.reserve(GpuTimingSource::Scene3d).is_none());
        assert_eq!(collector.statistics().dropped(), 1);
        let samples = drain(&mut collector, device, CAPACITY);
        for (id, source) in requests {
            let sample = samples.iter().find(|sample| sample.id() == id).unwrap();
            assert_eq!(sample.source(), source);
            assert!(
                sample.elapsed() > Duration::ZERO,
                "real draw work must be measured"
            );
        }
        assert_eq!(collector.statistics().pending(), 0);
        assert_eq!(collector.statistics().submitted(), CAPACITY as u64);
        assert_eq!(collector.statistics().completed(), CAPACITY as u64);
        assert_eq!(collector.statistics().readback_bytes(), 128);
        assert_eq!(collector.statistics().map_failures(), 0);
        assert_eq!(collector.statistics().invalid_samples(), 0);
        assert_eq!(collector.statistics().poll_failures(), 0);
        assert!(collector.collect(device).samples().is_empty());

        // Cancel a real async map before collecting it. Depending on scheduling,
        // cancellation reaches either the map callback or mapped-range access;
        // both must count one failure and release the slot without a fake sample.
        let _failed_id = fixture.submit(device, queue, &mut collector, GpuTimingSource::Scene3d);
        for slot in &collector.ring.as_ref().unwrap().slots {
            if slot.state.load(Ordering::Acquire) != FREE {
                slot.readback.unmap();
            }
        }
        let failed = drain(&mut collector, device, 0);
        assert!(failed.is_empty());
        assert_eq!(collector.statistics().map_failures(), 1);
        let reused = fixture.submit(
            device,
            queue,
            &mut collector,
            GpuTimingSource::FrameComposer,
        );
        let samples = drain(&mut collector, device, 1);
        assert_eq!(samples[0].id(), reused);
        assert_eq!(samples[0].source(), GpuTimingSource::FrameComposer);
        assert_eq!(collector.statistics().map_failures(), 1);

        // Replacing a ring while an old map is pending models logical-device
        // recovery without requiring a second physical adapter. Old callbacks
        // retain only old cells; their completion cannot enter the new batch.
        let stale = fixture.submit(device, queue, &mut collector, GpuTimingSource::Scene3d);
        drop(collector);
        let mut replacement = GpuTimingCollector::new(device, queue, true);
        assert!(replacement.collect(device).samples().is_empty());
        assert_eq!(replacement.statistics().submitted(), 0);
        assert_eq!(replacement.statistics().completed(), 0);
        assert_eq!(replacement.statistics().pending(), 0);
        let current = fixture.submit(
            device,
            queue,
            &mut replacement,
            GpuTimingSource::FrameComposer,
        );
        assert!(current > stale);
        let samples = drain(&mut replacement, device, 1);
        assert_eq!(samples[0].id(), current);
        assert_eq!(samples[0].source(), GpuTimingSource::FrameComposer);
        assert!(replacement.collect(device).samples().is_empty());
        eprintln!(
            "sim-engine GPU timestamp contract: bounded 8-slot correlation, saturation, cancellation, failed map and replacement passed; period_ns={:?}",
            replacement.statistics().timestamp_period_nanoseconds()
        );
    }

    fn drain(
        collector: &mut GpuTimingCollector,
        device: &wgpu::Device,
        expected: usize,
    ) -> Vec<GpuTimingSample> {
        let deadline = Instant::now() + Duration::from_secs(20);
        let mut samples = Vec::new();
        loop {
            samples.extend_from_slice(collector.collect(device).samples());
            if collector.statistics().pending() == 0 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "timestamp callbacks did not finish"
            );
            std::thread::yield_now();
        }
        assert_eq!(samples.len(), expected);
        samples
    }

    struct Workload {
        _target: wgpu::Texture,
        view: wgpu::TextureView,
        pipeline: wgpu::RenderPipeline,
    }

    impl Workload {
        fn new(device: &wgpu::Device) -> Self {
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("sim-engine timestamp fixture target"),
                size: wgpu::Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sim-engine timestamp fixture work"),
                source: wgpu::ShaderSource::Wgsl(
                    "
                    @vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
                        let positions = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                        return vec4(positions[index], 0.0, 1.0);
                    }
                    @fragment fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
                        var value = position.xy / 256.0;
                        for (var i = 0u; i < 16u; i += 1u) {
                            value = fract(value * vec2(1.73, 1.31) + vec2(0.11, 0.07));
                        }
                        return vec4(value, 0.5, 0.25);
                    }
                    "
                    .into(),
                ),
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sim-engine timestamp fixture pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            Self {
                _target: target,
                view,
                pipeline,
            }
        }

        fn submit(
            &self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            collector: &mut GpuTimingCollector,
            source: GpuTimingSource,
        ) -> GpuTimingId {
            let reservation = collector.reserve(source).unwrap();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine timestamp fixture encoder"),
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sim-engine timestamp fixture pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: Some(reservation.timestamp_writes()),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline);
                for _ in 0..16 {
                    pass.draw(0..3, 0..1);
                }
            }
            reservation.resolve(&mut encoder);
            queue.submit([encoder.finish()]);
            collector.submitted(reservation)
        }
    }
}
