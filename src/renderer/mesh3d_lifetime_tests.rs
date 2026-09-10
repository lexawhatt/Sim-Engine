use super::*;

fn topology(triangles: usize) -> Mesh3d {
    let mut vertices = Vec::with_capacity(triangles * 3);
    let mut indices = Vec::with_capacity(triangles * 3);
    for triangle in 0..triangles {
        let x = triangle as f32;
        vertices.extend([
            Vec3::new(x, 0.0, 0.0).unwrap(),
            Vec3::new(x + 0.5, 0.0, 0.0).unwrap(),
            Vec3::new(x, 0.5, 0.0).unwrap(),
        ]);
        indices.extend((triangle * 3..triangle * 3 + 3).map(|index| index as u32));
    }
    Mesh3d::new(vertices, indices).unwrap()
}

pub(super) fn assert_lifetime_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
) {
    let identity = Arc::new(());
    let meshes = [1, 2, 3].map(|triangles| {
        create_retained_mesh(device, queue, Arc::clone(&identity), topology(triangles)).unwrap()
    });
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap());
    if std::env::var("SIM_ENGINE_BENCH_OBJECT_LOOKUP").as_deref() == Ok("1") {
        paired_object_lookup_benchmark(&meshes[0]);
    }
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let mut ids: Vec<_> = (0..16)
        .map(|_| {
            scene
                .try_push(&meshes[0], Transform3d::IDENTITY, style)
                .unwrap()
        })
        .collect();
    let shared = scene.statistics();
    assert_eq!(shared.object_count(), 16);
    assert_eq!(shared.mesh_count(), 1);
    assert_eq!(shared.mesh_cpu_bytes(), meshes[0].recovery_memory_bytes());
    assert_eq!(shared.mesh_gpu_bytes(), meshes[0].gpu_allocation_bytes());
    scene.set_mesh(ids[0], &meshes[1]).unwrap();
    scene.set_mesh(ids[1], &meshes[2]).unwrap();
    let warmed_storage = scene.statistics().storage_bytes();
    let warmed_slots = scene.statistics().slot_capacity();
    let (_, allocations) = crate::test_allocations::count(|| {
        for iteration in 0..96 {
            let index = iteration % 16;
            let old_id = ids[index];
            scene.set_transform(old_id, Transform3d::IDENTITY).unwrap();
            scene.set_visible(old_id, iteration % 2 == 0).unwrap();
            scene.set_mesh(old_id, &meshes[iteration % 3]).unwrap();
            let removed = scene.remove(old_id).unwrap();
            assert!(matches!(
                scene.instance(old_id),
                Err(Scene3dError::ObjectNotFound { .. })
            ));
            assert_eq!(removed.id(), old_id);
            ids[index] = scene
                .try_push(&meshes[(iteration + 1) % 3], Transform3d::IDENTITY, style)
                .unwrap();
            assert_ne!(ids[index], old_id);
            assert!(ids[index].get() > old_id.get());
            assert!(scene.set_visible(old_id, true).is_err());
            assert!(
                scene
                    .instances()
                    .windows(2)
                    .all(|pair| pair[0].id().get() < pair[1].id().get())
            );
        }
    });
    assert_eq!(
        allocations, 0,
        "warmed object update/remove/reinsert allocated"
    );
    assert_eq!(scene.statistics().storage_bytes(), warmed_storage);
    assert_eq!(scene.statistics().slot_capacity(), warmed_slots);
    assert_eq!(scene.object_count(), 16);
    let foreign_scene = Scene3d::new(Color::BLACK).unwrap();
    assert!(matches!(
        foreign_scene.instance(ids[0]),
        Err(Scene3dError::ObjectNotFound { .. })
    ));

    // One shared source uploaded twice must count once on CPU and twice on GPU.
    let duplicate = create_retained_mesh(
        device,
        queue,
        Arc::clone(&identity),
        meshes[0].source.clone(),
    )
    .unwrap();
    let mut deduplicated = Scene3d::new(Color::BLACK).unwrap();
    let first = deduplicated
        .try_push(&meshes[0], Transform3d::IDENTITY, style)
        .unwrap();
    let second = deduplicated
        .try_push(&duplicate, Transform3d::IDENTITY, style)
        .unwrap();
    assert_eq!(deduplicated.statistics().mesh_count(), 2);
    assert_eq!(
        deduplicated.statistics().mesh_cpu_bytes(),
        meshes[0].recovery_memory_bytes()
    );
    assert_eq!(
        deduplicated.statistics().mesh_gpu_bytes(),
        meshes[0].gpu_allocation_bytes() * 2
    );
    let rebind = deduplicated.set_mesh(first, &meshes[1]).unwrap();
    assert!(Arc::ptr_eq(
        &deduplicated.instance(second).unwrap().mesh.vertex_buffer,
        &duplicate.vertex_buffer
    ));
    assert_eq!(
        rebind.peak_mesh_gpu_bytes(),
        meshes[0].gpu_allocation_bytes() * 2 + meshes[1].gpu_allocation_bytes()
    );
    assert_eq!(deduplicated.instance(first).unwrap().id(), first);
    deduplicated.remove(second).unwrap();
    assert_eq!(
        deduplicated.statistics().mesh_cpu_bytes(),
        meshes[1].recovery_memory_bytes()
    );
    assert_eq!(
        deduplicated.statistics().mesh_gpu_bytes(),
        meshes[1].gpu_allocation_bytes()
    );

    let limit = Scene3dBudget::new(
        1,
        usize::MAX,
        meshes[0].recovery_memory_bytes(),
        meshes[0].gpu_allocation_bytes(),
    )
    .unwrap();
    let mut bounded = Scene3d::with_budget(Color::BLACK, limit).unwrap();
    let bounded_id = bounded
        .try_push(&meshes[0], Transform3d::IDENTITY, style)
        .unwrap();
    let before = bounded.statistics();
    let (result, allocations) =
        crate::test_allocations::count(|| bounded.set_mesh(bounded_id, &meshes[2]));
    assert!(matches!(result, Err(Scene3dError::BudgetExceeded { .. })));
    assert_eq!(allocations, 0);
    assert_eq!(bounded.statistics(), before);
    assert!(Arc::ptr_eq(
        &bounded.instance(bounded_id).unwrap().mesh.vertex_buffer,
        &meshes[0].vertex_buffer
    ));
    assert!(
        bounded
            .try_push(&meshes[0], Transform3d::IDENTITY, style)
            .is_err()
    );
    assert_eq!(bounded.statistics(), before);

    let mut revision = meshes[1].clone();
    let old_clone = revision.clone();
    let tiny_budget = Mesh3dUploadBudget::new(1, 1, 1).unwrap();
    let (result, allocations) = crate::test_allocations::count(|| {
        upload::replace_resources(
            device,
            queue,
            &identity,
            &mut revision,
            meshes[0].source.clone(),
            tiny_budget,
        )
    });
    assert!(matches!(
        result,
        Err(Mesh3dResourceError::BudgetExceeded { .. })
    ));
    assert_eq!(allocations, 0);
    assert!(Arc::ptr_eq(
        &revision.vertex_buffer,
        &old_clone.vertex_buffer
    ));
    for source in [
        meshes[0].source.clone(),
        meshes[0].source.clone(),
        meshes[2].source.clone(),
    ] {
        let before_gpu = revision.gpu_allocation_bytes();
        let report = upload::replace_resources(
            device,
            queue,
            &identity,
            &mut revision,
            source,
            Mesh3dUploadBudget::default(),
        )
        .unwrap();
        assert!(report.replaced_buffers());
        assert_eq!(
            report.peak_gpu_bytes(),
            before_gpu + revision.gpu_allocation_bytes()
        );
        assert_eq!(report.uploaded_bytes(), revision.gpu_allocation_bytes());
        assert!(!Arc::ptr_eq(
            &revision.vertex_buffer,
            &old_clone.vertex_buffer
        ));
        assert_eq!(old_clone.triangle_count(), 2);
    }

    let removed_id = ids[3];
    scene.remove(removed_id).unwrap();
    let expected: Vec<_> = scene
        .instances()
        .iter()
        .map(|instance| {
            (
                instance.id(),
                instance.transform(),
                instance.style(),
                instance.is_visible(),
            )
        })
        .collect();
    let original_stats = scene.statistics();
    let recovery_identity = Arc::new(());
    let recovery = restore_scene3d_resources(
        recovery_device,
        recovery_queue,
        Arc::clone(&recovery_identity),
        &mut scene,
    )
    .unwrap();
    assert_eq!(recovery.object_count(), 15);
    assert_eq!(recovery.restored_mesh_count(), original_stats.mesh_count());
    assert_eq!(
        scene.statistics().mesh_cpu_bytes(),
        original_stats.mesh_cpu_bytes()
    );
    assert_eq!(
        scene.statistics().mesh_gpu_bytes(),
        original_stats.mesh_gpu_bytes()
    );
    assert_eq!(
        scene.statistics().slot_capacity(),
        original_stats.slot_capacity()
    );
    for (id, transform, style, visible) in expected {
        let instance = scene.instance(id).unwrap();
        assert_eq!(instance.transform(), transform);
        assert_eq!(instance.style(), style);
        assert_eq!(instance.is_visible(), visible);
        assert!(Arc::ptr_eq(
            &instance.mesh.renderer_identity,
            &recovery_identity
        ));
    }
    assert!(scene.instance(removed_id).is_err());
    assert_eq!(
        scene.try_push(&meshes[0], Transform3d::IDENTITY, style),
        Err(Scene3dError::RendererMismatch)
    );
    let restored_mesh = scene.instances()[0].mesh.clone();
    let new_id = scene
        .try_push(&restored_mesh, Transform3d::IDENTITY, style)
        .unwrap();
    assert_ne!(new_id, removed_id);
    assert!(scene.instance(removed_id).is_err());

    // Increasing only the object-count ceiling cannot invalidate an identical
    // three-object population that fits the exact retained-storage byte budget.
    let mut storage_probe = Scene3d::with_budget(
        Color::BLACK,
        Scene3dBudget::new(3, usize::MAX, usize::MAX, usize::MAX).unwrap(),
    )
    .unwrap();
    for _ in 0..3 {
        storage_probe
            .try_push(&meshes[0], Transform3d::IDENTITY, style)
            .unwrap();
    }
    let storage_limit = storage_probe.statistics().storage_bytes();
    let mut exact_storage = Scene3d::with_budget(
        Color::BLACK,
        Scene3dBudget::new(4, storage_limit, usize::MAX, usize::MAX).unwrap(),
    )
    .unwrap();
    for _ in 0..3 {
        exact_storage
            .try_push(&meshes[0], Transform3d::IDENTITY, style)
            .unwrap();
    }
    assert!(exact_storage.statistics().storage_bytes() <= storage_limit);
    let statistics = exact_storage.statistics();
    assert!(matches!(
        exact_storage.try_push(&meshes[0], Transform3d::IDENTITY, style),
        Err(Scene3dError::BudgetExceeded {
            resource: Scene3dBudgetResource::StorageBytes,
            ..
        })
    ));
    assert_eq!(exact_storage.statistics(), statistics);
}

fn linear_set_transform(
    scene: &mut Scene3d,
    id: Object3dId,
    transform: Transform3d,
) -> Result<(), Scene3dError> {
    validate_scene3d_transform(transform)?;
    let instance = scene
        .instances
        .iter_mut()
        .find(|instance| instance.id == id)
        .ok_or(Scene3dError::ObjectNotFound { object_id: id })?;
    instance.transform = transform;
    Ok(())
}

fn linear_set_style(
    scene: &mut Scene3d,
    id: Object3dId,
    style: MeshStyle3d,
) -> Result<(), Scene3dError> {
    let instance = scene
        .instances
        .iter_mut()
        .find(|instance| instance.id == id)
        .ok_or(Scene3dError::ObjectNotFound { object_id: id })?;
    validate_mesh_style(&instance.mesh, style)?;
    instance.style = style;
    Ok(())
}

fn linear_set_visible(
    scene: &mut Scene3d,
    id: Object3dId,
    visible: bool,
) -> Result<(), Scene3dError> {
    let instance = scene
        .instances
        .iter_mut()
        .find(|instance| instance.id == id)
        .ok_or(Scene3dError::ObjectNotFound { object_id: id })?;
    instance.visible = visible;
    Ok(())
}

fn apply_setters<const INDEXED: bool>(scene: &mut Scene3d, ids: &[Object3dId], updates: &[usize]) {
    let transform = Transform3d::IDENTITY;
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::rgb(0.25, 0.5, 0.75)).unwrap());
    for &index in std::hint::black_box(updates) {
        let id = ids[index];
        if INDEXED {
            scene.set_transform(id, transform).unwrap();
            scene.set_style(id, style).unwrap();
            scene.set_visible(id, false).unwrap();
        } else {
            linear_set_transform(scene, id, transform).unwrap();
            linear_set_style(scene, id, style).unwrap();
            linear_set_visible(scene, id, false).unwrap();
        }
    }
    std::hint::black_box(scene.instances());
}

fn paired_object_lookup_benchmark(mesh: &RetainedMesh3d) {
    if cfg!(debug_assertions) {
        panic!("object lookup benchmark requires --release");
    }
    for count in [1, 64, 256, 4096] {
        let mut indexed = Scene3d::new(Color::BLACK).unwrap();
        let mut linear = Scene3d::new(Color::BLACK).unwrap();
        let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap());
        let indexed_ids: Vec<_> = (0..count)
            .map(|_| {
                indexed
                    .try_push(mesh, Transform3d::IDENTITY, style)
                    .unwrap()
            })
            .collect();
        let linear_ids: Vec<_> = (0..count)
            .map(|_| linear.try_push(mesh, Transform3d::IDENTITY, style).unwrap())
            .collect();
        for workload in ["none", "one", "sparse", "all"] {
            let updates: Vec<_> = match workload {
                "none" => Vec::new(),
                "one" => vec![count - 1],
                "sparse" => (0..count).step_by(16).collect(),
                _ => (0..count).collect(),
            };
            let (_, indexed_allocations) = crate::test_allocations::count(|| {
                apply_setters::<true>(&mut indexed, &indexed_ids, &updates)
            });
            let (_, linear_allocations) = crate::test_allocations::count(|| {
                apply_setters::<false>(&mut linear, &linear_ids, &updates)
            });
            assert_eq!((indexed_allocations, linear_allocations), (0, 0));
            let repetitions = (8192 / updates.len().max(1)).clamp(1, 128);
            let mut indexed_samples = Vec::with_capacity(15);
            let mut linear_samples = Vec::with_capacity(15);
            for round in 0..19 {
                let mut measure = |use_indexed| {
                    let started = Instant::now();
                    for _ in 0..repetitions {
                        if use_indexed {
                            apply_setters::<true>(&mut indexed, &indexed_ids, &updates);
                        } else {
                            apply_setters::<false>(&mut linear, &linear_ids, &updates);
                        }
                    }
                    let elapsed = started.elapsed().as_secs_f64() / repetitions as f64;
                    if round >= 4 {
                        if use_indexed {
                            indexed_samples.push(elapsed);
                        } else {
                            linear_samples.push(elapsed);
                        }
                    }
                };
                measure(round % 2 == 0);
                measure(round % 2 != 0);
                assert_eq!(indexed.statistics(), linear.statistics());
                for (actual, expected) in indexed.instances().iter().zip(linear.instances()) {
                    assert_eq!(actual.id().get(), expected.id().get());
                    assert_eq!(actual.transform(), expected.transform());
                    assert_eq!(actual.style(), expected.style());
                    assert_eq!(actual.is_visible(), expected.is_visible());
                }
            }
            indexed_samples.sort_by(f64::total_cmp);
            linear_samples.sort_by(f64::total_cmp);
            println!(
                "object_lookup objects={count} workload={workload} updated_objects={} setters={} indexed_us={:.3} linear_us={:.3} ratio={:.3} indexed_p10_p90_us={:.3}/{:.3} linear_p10_p90_us={:.3}/{:.3} allocations={indexed_allocations}/{linear_allocations}",
                updates.len(),
                updates.len() * 3,
                indexed_samples[7] * 1e6,
                linear_samples[7] * 1e6,
                indexed_samples[7] / linear_samples[7],
                indexed_samples[1] * 1e6,
                indexed_samples[13] * 1e6,
                linear_samples[1] * 1e6,
                linear_samples[13] * 1e6,
            );
        }
    }
}
