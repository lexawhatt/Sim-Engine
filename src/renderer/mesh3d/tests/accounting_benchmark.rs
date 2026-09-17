//! Opt-in CPU scene construction and mutation diagnostics on preuploaded meshes.

use super::*;
use std::hint::black_box;

fn summary(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    (
        samples[samples.len() / 2],
        samples[samples.len() * 95 / 100],
    )
}

fn populate(count: usize, meshes: &[RetainedMesh3d], ids: &mut Vec<Object3dId>) -> Scene3d {
    ids.clear();
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap());
    for index in 0..count {
        ids.push(
            scene
                .try_push(&meshes[index % meshes.len()], Transform3d::IDENTITY, style)
                .unwrap(),
        );
    }
    scene
}

pub(super) fn run(device: &wgpu::Device, queue: &wgpu::Queue) {
    if std::env::var("SIM_ENGINE_BENCH_SCENE_ACCOUNTING").as_deref() != Ok("1") {
        return;
    }
    if cfg!(debug_assertions) {
        panic!("accounting benchmark requires --release");
    }
    let identity = Arc::new(());
    let source = Mesh3d::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![0, 1, 2]).unwrap();
    // GPU allocation and source preparation are deliberately outside CPU timing.
    let meshes: Vec<_> = (0..4096)
        .map(|_| {
            create_retained_mesh(device, queue, Arc::clone(&identity), source.clone()).unwrap()
        })
        .collect();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
    println!(
        "scene_accounting_fixture=v1 source_sha={} fixture_sha={} scope=cpu_only_preuploaded_meshes rounds=15 warmup=3",
        std::env::var("SIM_ENGINE_RELEASE_SHA").unwrap_or_default(),
        std::env::var("SIM_ENGINE_ACCOUNTING_FIXTURE_SHA").unwrap_or_default()
    );
    for count in [64, 1024, 4096] {
        for unique in [1, count] {
            let pool = &meshes[..unique];
            let mut ids = Vec::with_capacity(count);
            let mut samples = Vec::with_capacity(15);
            let mut allocations = Vec::with_capacity(15);
            for round in 0..18 {
                let ((scene, elapsed), count_allocations) = crate::test_allocations::count(|| {
                    let started = Instant::now();
                    let scene = populate(count, pool, &mut ids);
                    (scene, started.elapsed().as_secs_f64() * 1000.0)
                });
                let stats = scene.statistics();
                assert_eq!(stats.object_count(), count);
                assert_eq!(stats.mesh_count(), unique);
                assert_eq!(stats.mesh_cpu_bytes(), source.recovery_memory_bytes());
                assert_eq!(scene.visible_object_count(), count);
                black_box(&scene);
                if round >= 3 {
                    samples.push(elapsed);
                    allocations.push(count_allocations);
                }
            }
            println!(
                "scene_accounting_samples operation=construct objects={count} unique={unique} ms={samples:?} allocations={allocations:?}"
            );
            let (p50, p95) = summary(&mut samples);
            println!(
                "scene_accounting operation=construct objects={count} unique={unique} p50_ms={p50:.6} p95_ms={p95:.6} allocations_min={} allocations_max={}",
                allocations.iter().min().unwrap(),
                allocations.iter().max().unwrap()
            );

            // Query an unchanged scene: mutations must not shrink the distinct
            // resource table underneath the statistics control.
            let query_scene = populate(count, pool, &mut ids);
            let mut scene = populate(count, pool, &mut ids);
            let mut pool_keys: Vec<_> = pool
                .iter()
                .enumerate()
                .map(|(index, mesh)| (Arc::as_ptr(&mesh.vertex_buffer) as usize, index))
                .collect();
            pool_keys.sort_unstable();
            let mut query_samples = Vec::with_capacity(15);
            let mut mutation_samples = Vec::with_capacity(15);
            let mut mutation_allocations = Vec::with_capacity(15);
            for round in 0..18 {
                let (_, query_allocations) = crate::test_allocations::count(|| {
                    let started = Instant::now();
                    for _ in 0..4096 {
                        black_box(black_box(&query_scene).statistics());
                        black_box(black_box(&query_scene).visible_object_count());
                    }
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    if round >= 3 {
                        query_samples.push(elapsed);
                    }
                });
                assert_eq!(query_allocations, 0);
                let (elapsed, allocation_count) = crate::test_allocations::count(|| {
                    let started = Instant::now();
                    for change in 0..256 {
                        let index = (change * 104729 + round * 31) % count;
                        scene
                            .set_mesh(ids[index], &pool[(index + round + change) % unique])
                            .unwrap();
                        scene
                            .set_visible(ids[index], (change + round) % 2 == 0)
                            .unwrap();
                        black_box(scene.statistics());
                    }
                    started.elapsed().as_secs_f64() * 1000.0
                });
                if round >= 3 {
                    mutation_samples.push(elapsed);
                    mutation_allocations.push(allocation_count);
                    let digest =
                        scene
                            .instances()
                            .iter()
                            .fold(0xcbf29ce484222325u64, |state, instance| {
                                let key = Arc::as_ptr(&instance.mesh.vertex_buffer) as usize;
                                let index = pool_keys
                                    .binary_search_by_key(&key, |entry| entry.0)
                                    .unwrap();
                                [
                                    instance.id().get(),
                                    pool_keys[index].1 as u64,
                                    u64::from(instance.is_visible()),
                                ]
                                .into_iter()
                                .fold(state, |state, value| {
                                    (state ^ value).wrapping_mul(0x100000001b3)
                                })
                            });
                    println!(
                        "scene_accounting_mutation round={} objects={count} pool_unique={unique} live_unique={} visible={} digest={digest:016x}",
                        round - 2,
                        scene.statistics().mesh_count(),
                        scene.visible_object_count()
                    );
                }
                assert_eq!(scene.object_count(), count);
                assert_eq!(
                    scene.visible_object_count(),
                    scene
                        .instances()
                        .iter()
                        .filter(|item| item.is_visible())
                        .count()
                );
            }
            println!(
                "scene_accounting_samples operation=statistics4096 objects={count} unique={unique} ms={query_samples:?}"
            );
            println!(
                "scene_accounting_samples operation=rebind_visibility256 objects={count} pool_unique={unique} ms={mutation_samples:?} allocations={mutation_allocations:?}"
            );
            let (p50, p95) = summary(&mut query_samples);
            println!(
                "scene_accounting operation=statistics4096 objects={count} unique={unique} p50_ms={p50:.6} p95_ms={p95:.6} allocations=0"
            );
            let (p50, p95) = summary(&mut mutation_samples);
            println!(
                "scene_accounting operation=rebind_visibility256 objects={count} pool_unique={unique} p50_ms={p50:.6} p95_ms={p95:.6} allocations_min={} allocations_max={}",
                mutation_allocations.iter().min().unwrap(),
                mutation_allocations.iter().max().unwrap()
            );
        }
    }
}
