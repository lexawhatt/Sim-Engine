use super::*;

#[path = "cache_uniform_upload_tests.rs"]
mod uniform_upload_tests;

#[test]
fn shared_binding_key_requires_kind_identity_sampling_and_exact_uniform_bytes() {
    let image = Arc::new(());
    let other = Arc::new(());
    let bytes = [0_u8; 16];
    let mut changed = bytes;
    changed[15] = 1;
    let key = BindingKeyRef::Image {
        identity: &image,
        sampling: ImageSampling::Nearest,
        texture_bytes: 4,
    };
    let mut memo = FrameBindingSharing::default();
    memo.remember(Some((key, &bytes)), 7);
    assert_eq!(memo.find((key, &bytes)), Some(7));
    for mismatch in [
        BindingKeyRef::Camera,
        BindingKeyRef::Target {
            identity: &image,
            texture_bytes: 4,
        },
        BindingKeyRef::Image {
            identity: &other,
            sampling: ImageSampling::Nearest,
            texture_bytes: 4,
        },
        BindingKeyRef::Image {
            identity: &image,
            sampling: ImageSampling::Linear,
            texture_bytes: 4,
        },
    ] {
        assert_eq!(memo.find((mismatch, &bytes)), None);
    }
    assert_eq!(memo.find((key, &changed)), None);
    assert_eq!(memo.find((key, &bytes[..12])), None);
    // Scalar/LUT bindings have no complete description and never enter the memo.
    memo.remember(None, 8);
    assert_eq!(memo.find((key, &bytes)), Some(7));
}

#[test]
fn sharing_ring_is_bounded_allocation_free_and_records_recent_aliases() {
    let uniforms: [[u8; 16]; 9] = std::array::from_fn(|index| [index as u8; 16]);
    let (_, allocations) = crate::test_allocations::count(|| {
        let mut memo = FrameBindingSharing::default();
        for (index, bytes) in uniforms.iter().enumerate() {
            memo.remember(Some((BindingKeyRef::Camera, bytes)), index);
        }
        assert_eq!(memo.find((BindingKeyRef::Camera, &uniforms[0])), None);
        assert_eq!(memo.find((BindingKeyRef::Camera, &uniforms[8])), Some(8));
        for index in 9..33 {
            memo.remember(Some((BindingKeyRef::Camera, &uniforms[8])), index);
        }
        assert!(memo.find((BindingKeyRef::Camera, &uniforms[8])).is_some());
        assert_eq!(memo.find((BindingKeyRef::Camera, &uniforms[1])), None);
    });
    assert_eq!(allocations, 0);
}

/// Uses real GPU uniform storage and queue writes. Buffer readback, rather than
/// a separate shader copy of the renderer, proves that retained slots never
/// alias after an earlier frame temporarily shared their binding handles.
pub(in crate::renderer) fn assert_gpu_binding_sharing_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sim-engine frame sharing oracle layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let red = [1.0_f32, 0.0, 0.0, 1.0];
    let green = [0.0_f32, 1.0, 0.0, 1.0];
    for (budget, expected_retained) in [
        (FrameCacheBudget::default(), 2),
        (FrameCacheBudget::new(0, 0, 0, 0), 0),
        (FrameCacheBudget::new(8192, 16, 0, 3), 1),
    ] {
        let mut cache = FrameCache {
            budget,
            ..FrameCache::default()
        };
        // Cold A/A/B creates two actual GPU allocations. The second A remains
        // an empty persistent slot, not a mutable alias to the first A.
        let initial = [red, red, green];
        let bindings = build_bindings(device, queue, &layout, &mut cache, &initial);
        assert_uniform_readback(device, queue, &bindings, &initial);
        assert_eq!(cache.statistics.created_buffers(), 2);
        assert_eq!(cache.statistics.created_bind_groups(), 2);
        assert_eq!(cache.statistics.shared_bind_groups(), 1);
        assert_eq!(cache.statistics.uploaded_uniform_bytes(), 32);
        assert_eq!(cache.statistics.binding_count(), expected_retained);
        assert_eq!(cache.statistics.uniform_bytes(), expected_retained * 16);
        assert!(cache.slots.get(1).is_none_or(Option::is_none));
        drop(bindings);

        let bindings = build_bindings(device, queue, &layout, &mut cache, &initial);
        assert_uniform_readback(device, queue, &bindings, &initial);
        assert_eq!(cache.statistics.shared_bind_groups(), 1);
        assert_eq!(cache.statistics.created_buffers(), 2 - expected_retained);
        assert_eq!(
            cache.statistics.uploaded_uniform_bytes(),
            (2 - expected_retained) * 16
        );
        drop(bindings);

        // B/A/B changes slot 0 and requires an independent A at slot 1. If the
        // first-frame alias were retained in slot 1, one queued write would
        // silently overwrite the other item's uniform before GPU execution.
        let changed = [green, red, green];
        let bindings = build_bindings(device, queue, &layout, &mut cache, &changed);
        assert_uniform_readback(device, queue, &bindings, &changed);
        assert_eq!(cache.statistics.uploaded_uniform_bytes(), 32);
        assert_eq!(
            cache.statistics.created_buffers(),
            if expected_retained == 0 { 2 } else { 1 }
        );
        assert_eq!(
            cache.statistics.shared_bind_groups(),
            usize::from(expected_retained < 2)
        );
        assert_eq!(
            cache.statistics.binding_count(),
            if expected_retained == 2 {
                3
            } else {
                expected_retained
            }
        );
        assert_eq!(cache.statistics.texture_bytes(), 0);
        assert_eq!(
            cache.statistics.reused_bind_groups() + cache.statistics.created_bind_groups(),
            3
        );
    }
    assert_gpu_texture_sharing_accounting(device, queue);
    uniform_upload_tests::assert_cache_uniform_upload_contract(device, queue);
}

fn build_bindings(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    cache: &mut FrameCache,
    colors: &[[f32; 4]; 3],
) -> Vec<FrameBinding> {
    let descriptions: [_; 3] = std::array::from_fn(|index| {
        (
            (BindingKeyRef::Camera, bytemuck::bytes_of(&colors[index])),
            None,
        )
    });
    build_described_bindings(device, queue, layout, cache, &descriptions)
}

fn build_described_bindings(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    cache: &mut FrameCache,
    descriptions: &[(BindingDescription<'_>, Option<&wgpu::TextureView>); 3],
) -> Vec<FrameBinding> {
    cache.begin();
    cache.reserve_slots(descriptions.len()).unwrap();
    let mut memo = FrameBindingSharing::default();
    let mut bindings = Vec::with_capacity(descriptions.len());
    for (slot, (description, view)) in descriptions.iter().enumerate() {
        let bytes = description.1;
        let description = Some(*description);
        let binding = cache
            .reuse_binding(queue, description, slot, &mut memo, &bindings)
            .unwrap_or_else(|| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sim-engine frame sharing oracle uniform"),
                    size: bytes.len() as u64,
                    usage: wgpu::BufferUsages::UNIFORM
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buffer, 0, bytes);
                let mut entries = vec![wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }];
                if let Some(view) = view {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                }
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sim-engine frame sharing oracle binding"),
                    layout,
                    entries: &entries,
                });
                cache.retain_new_binding(
                    FrameBinding {
                        _buffer: buffer,
                        bind_group,
                    },
                    description,
                    slot,
                    &mut memo,
                )
            });
        bindings.push(binding);
    }
    cache.finish();
    bindings
}

fn assert_gpu_texture_sharing_accounting(device: &wgpu::Device, queue: &wgpu::Queue) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sim-engine frame texture sharing oracle layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let identities = [Arc::new(()), Arc::new(())];
    let views = identities.each_ref().map(|_| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("sim-engine frame sharing oracle texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    });
    let image_keys = identities.each_ref().map(|identity| BindingKeyRef::Image {
        identity,
        sampling: ImageSampling::Nearest,
        texture_bytes: 4,
    });
    let target_key = BindingKeyRef::Target {
        identity: &identities[0],
        texture_bytes: 4,
    };
    let color = [1.0_f32, 0.0, 0.0, 1.0];
    let bytes = bytemuck::bytes_of(&color);
    for texture_budget in [4, 8] {
        let mut cache = FrameCache {
            budget: FrameCacheBudget::new(8192, 64, texture_budget, 3),
            ..FrameCache::default()
        };
        let initial = [
            ((image_keys[0], bytes), Some(&views[0])),
            ((image_keys[1], bytes), Some(&views[1])),
            ((image_keys[0], bytes), Some(&views[0])),
        ];
        let bindings = build_described_bindings(device, queue, &layout, &mut cache, &initial);
        assert_uniform_readback(device, queue, &bindings, &[color; 3]);
        assert_eq!(cache.statistics.created_buffers(), 2);
        assert_eq!(cache.statistics.shared_bind_groups(), 1);
        assert_eq!(cache.statistics.texture_bytes(), texture_budget);
        assert_eq!(cache.statistics.peak_texture_bytes(), 8);
        drop(bindings);

        // The new second A shares this frame only. If B's old slot exists it
        // still owns B's texture and must remain in distinct retained bytes.
        // A target key is distinct from image A even with identical bytes/view.
        let changed = [
            ((image_keys[0], bytes), Some(&views[0])),
            ((image_keys[0], bytes), Some(&views[0])),
            ((target_key, bytes), Some(&views[0])),
        ];
        let bindings = build_described_bindings(device, queue, &layout, &mut cache, &changed);
        assert_uniform_readback(device, queue, &bindings, &[color; 3]);
        assert_eq!(cache.statistics.created_buffers(), 1);
        assert_eq!(cache.statistics.shared_bind_groups(), 1);
        assert_eq!(cache.statistics.uploaded_uniform_bytes(), 16);
        assert_eq!(cache.statistics.texture_bytes(), texture_budget);
        assert_eq!(cache.statistics.peak_texture_bytes(), texture_budget);
        assert_eq!(
            cache.statistics.uniform_bytes(),
            16 * (texture_budget / 4 + 1)
        );
        if texture_budget == 8 {
            assert!(cache.slots[1].as_ref().unwrap().key.matches(image_keys[1]));
        } else {
            assert!(cache.slots[1].is_none());
        }
        assert_eq!(Arc::strong_count(&identities[0]), 3);
        assert_eq!(
            Arc::strong_count(&identities[1]),
            usize::from(texture_budget == 8) + 1
        );
        drop(bindings);

        let other_color = [0.0_f32, 1.0, 0.0, 1.0];
        let back_to_b = [
            (
                (image_keys[0], bytemuck::bytes_of(&other_color)),
                Some(&views[0]),
            ),
            ((image_keys[1], bytes), Some(&views[1])),
            ((target_key, bytes), Some(&views[0])),
        ];
        let bindings = build_described_bindings(device, queue, &layout, &mut cache, &back_to_b);
        assert_uniform_readback(device, queue, &bindings, &[other_color, color, color]);
        assert_eq!(cache.statistics.shared_bind_groups(), 0);
        assert_eq!(
            cache.statistics.created_buffers(),
            usize::from(texture_budget == 4)
        );
        assert_eq!(
            cache.statistics.uploaded_uniform_bytes(),
            if texture_budget == 4 { 32 } else { 16 }
        );
        assert_eq!(cache.statistics.texture_bytes(), texture_budget);
        assert_eq!(cache.statistics.peak_texture_bytes(), 8);
        drop(bindings);
        cache.clear();
        assert_eq!(cache.statistics, FrameCacheStatistics::default());
        assert_eq!(Arc::strong_count(&identities[0]), 1);
        assert_eq!(Arc::strong_count(&identities[1]), 1);
    }
}

fn assert_uniform_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bindings: &[FrameBinding],
    expected: &[[f32; 4]; 3],
) {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine frame sharing oracle readback"),
        size: 48,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for (index, binding) in bindings.iter().enumerate() {
        encoder.copy_buffer_to_buffer(&binding._buffer, 0, &readback, (index * 16) as u64, 16);
    }
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let actual = slice.get_mapped_range().unwrap();
    assert_eq!(actual.as_ref(), bytemuck::cast_slice(expected));
}
