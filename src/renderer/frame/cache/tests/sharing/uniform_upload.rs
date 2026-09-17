//! Production cache collector/reuse/flush coverage. Surface acquisition is not
//! mocked: the pre-acquire skip boundary is exercised directly as prepare+finish.
use super::*;

const COUNT: usize = 80;
const UNIFORM_BYTES: usize = std::mem::size_of::<CameraUniform>();

pub(super) fn assert_cache_uniform_upload_contract(device: &wgpu::Device, queue: &wgpu::Queue) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cache uniform collector oracle layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let mut ready = (0..COUNT)
        .map(|index| geometry(index, 0))
        .collect::<Vec<_>>();
    let mut cache = FrameCache::default();
    collect_and_check(device, queue, &layout, &mut cache, &ready);
    assert_eq!(cache.statistics.uniform_upload_calls(), COUNT);
    assert_eq!(cache.statistics.batched_uniform_copies(), 0);
    assert_eq!(cache.statistics.upload_staging_bytes(), 0);

    for (index, item) in ready.iter_mut().enumerate() {
        *item = geometry(index, 1);
    }
    let snapshots = cache
        .slots
        .iter()
        .map(|cached| cached.as_ref().unwrap().bytes)
        .collect::<Vec<_>>();
    // A surface skip or pre-submit CPU rejection stops here in production:
    // preparation may reserve scratch but cannot publish new slot snapshots.
    cache.begin();
    cache.prepare_uniform_uploads(&ready, device.limits().max_buffer_size);
    assert!(cache.uniform_uploads.cpu_bytes() > 0);
    cache.finish();
    assert_eq!(cache.statistics.uniform_upload_calls(), 0);
    assert_eq!(cache.statistics.uploaded_uniform_bytes(), 0);
    assert_eq!(cache.statistics.created_upload_staging_buffers(), 0);
    for (cached, before) in cache.slots.iter().zip(&snapshots) {
        assert_eq!(cached.as_ref().unwrap().bytes, *before);
    }
    let old_bindings = cache
        .slots
        .iter()
        .map(|cached| cached.as_ref().unwrap().binding.clone())
        .collect::<Vec<_>>();
    check_uniforms(
        device,
        queue,
        &old_bindings,
        &snapshots
            .iter()
            .map(|bytes| bytes[..UNIFORM_BYTES].to_vec())
            .collect::<Vec<_>>(),
    );
    drop(old_bindings);

    collect_and_check(device, queue, &layout, &mut cache, &ready);
    assert_eq!(cache.statistics.uniform_upload_calls(), 1);
    assert_eq!(cache.statistics.batched_uniform_copies(), COUNT);
    assert_eq!(
        cache.statistics.batched_uniform_bytes(),
        COUNT * UNIFORM_BYTES
    );
    assert_eq!(
        cache.statistics.uploaded_uniform_bytes(),
        COUNT * UNIFORM_BYTES
    );
    assert_eq!(cache.statistics.created_buffers(), 0);
    assert_eq!(cache.statistics.created_upload_staging_buffers(), 1);
    assert_eq!(
        cache.statistics.upload_staging_bytes(),
        COUNT * UNIFORM_BYTES
    );
    assert_eq!(cache.statistics.uniform_bytes(), COUNT * UNIFORM_BYTES);
    assert_eq!(cache.statistics.cpu_bytes(), cache.cpu_bytes());
    assert!(cache.statistics.peak_cpu_bytes() >= cache.statistics.cpu_bytes());

    collect_and_check(device, queue, &layout, &mut cache, &ready);
    assert_eq!(cache.statistics.uniform_upload_calls(), 0);
    assert_eq!(cache.statistics.batched_uniform_copies(), 0);
    assert_eq!(cache.statistics.uploaded_uniform_bytes(), 0);
    assert_eq!(cache.statistics.created_upload_staging_buffers(), 0);

    // Below the optional packing threshold, dirty uniforms remain direct writes.
    for (index, item) in ready.iter_mut().enumerate().skip(COUNT - 31) {
        *item = geometry(index, 2);
    }
    collect_and_check(device, queue, &layout, &mut cache, &ready);
    assert_eq!(cache.statistics.uniform_upload_calls(), 31);
    assert_eq!(cache.statistics.batched_uniform_copies(), 0);
    assert_eq!(
        cache.statistics.uploaded_uniform_bytes(),
        31 * UNIFORM_BYTES
    );

    for index in (0..31).chain([COUNT - 1]) {
        ready[index] = geometry(index, 3);
    }
    collect_and_check(device, queue, &layout, &mut cache, &ready);
    assert_eq!(cache.statistics.uniform_upload_calls(), 1);
    assert_eq!(cache.statistics.batched_uniform_copies(), 32);
    assert_eq!(
        cache.statistics.uploaded_uniform_bytes(),
        32 * UNIFORM_BYTES
    );
    assert_eq!(cache.statistics.created_upload_staging_buffers(), 0);

    // One frame combines packed changed slots, unchanged slots, a new direct
    // uniform and a temporary alias in a hole. Only changed retained hits pack.
    for (index, item) in ready.iter_mut().enumerate().take(32) {
        *item = geometry(index, 4);
    }
    cache.remove_slot(60);
    ready[60] = geometry(60, 5);
    cache.remove_slot(70);
    ready[70] = geometry(69, 2);
    collect_and_check(device, queue, &layout, &mut cache, &ready);
    assert_eq!(cache.statistics.uniform_upload_calls(), 2);
    assert_eq!(cache.statistics.batched_uniform_copies(), 32);
    assert_eq!(
        cache.statistics.uploaded_uniform_bytes(),
        33 * UNIFORM_BYTES
    );
    assert_eq!(cache.statistics.created_buffers(), 1);
    assert_eq!(cache.statistics.shared_bind_groups(), 1);
    assert!(cache.slots[70].is_none());
    collect_and_check(device, queue, &layout, &mut cache, &ready);
    assert_eq!(cache.statistics.uniform_upload_calls(), 0);
    assert_eq!(cache.statistics.batched_uniform_copies(), 0);
    assert_eq!(cache.statistics.uploaded_uniform_bytes(), 0);
    assert_eq!(cache.statistics.shared_bind_groups(), 1);
    cache.clear();
    assert_eq!(cache.statistics, FrameCacheStatistics::default());
    assert_eq!(cache.cpu_bytes(), 0);
    assert_eq!(cache.uniform_uploads.allocation_bytes(), 0);

    // The unchanged public constructor disables packing. Uniform retention and
    // ordinary updates still work without any transfer-buffer allocation.
    let mut direct = FrameCache {
        budget: FrameCacheBudget::default().with_upload_staging_bytes(0),
        ..FrameCache::default()
    };
    ready = (0..COUNT).map(|index| geometry(index, 0)).collect();
    collect_and_check(device, queue, &layout, &mut direct, &ready);
    ready = (0..COUNT).map(|index| geometry(index, 1)).collect();
    collect_and_check(device, queue, &layout, &mut direct, &ready);
    assert_eq!(direct.statistics.uniform_upload_calls(), COUNT);
    assert_eq!(direct.statistics.batched_uniform_copies(), 0);
    assert_eq!(direct.statistics.created_upload_staging_buffers(), 0);
    assert_eq!(direct.statistics.upload_staging_bytes(), 0);

    // Exactly enough idle CPU space for slots, not packing scratch: a changed
    // frame can pack, but finish must evict its scratch and retain honest peaks.
    let slot_bytes = COUNT * std::mem::size_of::<Option<CachedBinding>>();
    let mut bounded = FrameCache {
        budget: FrameCacheBudget::new(slot_bytes, COUNT * UNIFORM_BYTES, 0, COUNT)
            .with_upload_staging_bytes(COUNT * UNIFORM_BYTES),
        ..FrameCache::default()
    };
    ready = (0..COUNT).map(|index| geometry(index, 0)).collect();
    collect_and_check(device, queue, &layout, &mut bounded, &ready);
    ready = (0..COUNT).map(|index| geometry(index, 1)).collect();
    collect_and_check(device, queue, &layout, &mut bounded, &ready);
    assert_eq!(bounded.statistics.uniform_upload_calls(), 1);
    assert_eq!(bounded.statistics.batched_uniform_copies(), COUNT);
    assert_eq!(bounded.statistics.created_upload_staging_buffers(), 1);
    assert_eq!(bounded.statistics.upload_staging_bytes(), 0);
    assert_eq!(
        bounded.statistics.peak_upload_staging_bytes(),
        COUNT * UNIFORM_BYTES
    );
    assert_eq!(bounded.statistics.uniform_bytes(), COUNT * UNIFORM_BYTES);
    assert!(bounded.statistics.cpu_bytes() <= slot_bytes);
    assert!(bounded.statistics.peak_cpu_bytes() > slot_bytes);
}

fn geometry(index: usize, generation: usize) -> ReadyItem<'static> {
    let viewport = LogicalViewport::new(64.0, 64.0).unwrap();
    let camera = Camera2d::new(Vec2::new(index as f32, generation as f32), 1.0).unwrap();
    ReadyItem::Geometry(ReadyGeometry {
        source: ReadySource::Streaming,
        vertex_count: 0,
        batches: Vec::new(),
        camera_uniform: CameraUniform::new(camera, viewport).unwrap(),
        viewport: ResolvedViewport {
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
        },
    })
}

fn collect_and_check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    cache: &mut FrameCache,
    ready: &[ReadyItem<'_>],
) {
    cache.begin();
    cache.reserve_slots(ready.len()).unwrap();
    cache.prepare_uniform_uploads(ready, device.limits().max_buffer_size);
    let mut sharing = FrameBindingSharing::default();
    let mut bindings = Vec::with_capacity(ready.len());
    let mut expected = Vec::with_capacity(ready.len());
    for (slot, item) in ready.iter().enumerate() {
        let description = binding_description(item);
        let bytes = description.unwrap().1;
        expected.push(bytes.to_vec());
        let binding = cache
            .reuse_binding(queue, description, slot, &mut sharing, &bindings)
            .unwrap_or_else(|| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cache uniform collector oracle destination"),
                    size: bytes.len() as u64,
                    usage: wgpu::BufferUsages::UNIFORM
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                // Mirror the production cold-binding upload so call/byte stats are real.
                queue.write_buffer(&buffer, 0, bytes);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cache uniform collector oracle binding"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });
                cache.retain_new_binding(
                    FrameBinding {
                        _buffer: buffer,
                        bind_group,
                    },
                    description,
                    slot,
                    &mut sharing,
                )
            });
        bindings.push(binding);
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    cache.flush_uniform_uploads(device, queue, &mut encoder, &bindings);
    let readback = capture_uniforms(device, &mut encoder, &bindings);
    queue.submit([encoder.finish()]);
    cache.finish();
    verify_readback(device, &readback, &expected);
}

fn capture_uniforms(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    bindings: &[FrameBinding],
) -> wgpu::Buffer {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cache uniform collector oracle readback"),
        size: (bindings.len() * UNIFORM_BYTES) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    for (index, binding) in bindings.iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            &binding._buffer,
            0,
            &readback,
            (index * UNIFORM_BYTES) as u64,
            UNIFORM_BYTES as u64,
        );
    }
    readback
}

fn check_uniforms(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bindings: &[FrameBinding],
    expected: &[Vec<u8>],
) {
    let mut encoder = device.create_command_encoder(&Default::default());
    let readback = capture_uniforms(device, &mut encoder, bindings);
    queue.submit([encoder.finish()]);
    verify_readback(device, &readback, expected);
}

fn verify_readback(device: &wgpu::Device, readback: &wgpu::Buffer, expected: &[Vec<u8>]) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let actual = slice.get_mapped_range().unwrap();
    let expected = expected.iter().flatten().copied().collect::<Vec<_>>();
    assert_eq!(actual.as_ref(), expected.as_slice());
}
