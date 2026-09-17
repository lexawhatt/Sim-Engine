use super::*;
use crate::renderer::frame::*;
use crate::renderer::*;

mod accounting;
mod sharing;
pub(in crate::renderer) use sharing::assert_gpu_binding_sharing_contract;

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
