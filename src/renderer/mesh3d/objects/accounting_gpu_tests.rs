use super::*;
use crate::{Mesh3dAttributes, SurfaceLighting3d, TextureAddressMode3d, TextureCoordinate2d};
use std::collections::BTreeMap;

type ResourceMap = BTreeMap<(u8, usize), (usize, usize)>;

fn record(resources: &mut ResourceMap, category: u8, key: usize, bytes: usize) {
    if bytes == 0 {
        return;
    }
    assert_ne!(
        key, 0,
        "temporary preparation identity escaped into a live instance"
    );
    resources
        .entry((category, key))
        .and_modify(|entry| {
            assert_eq!(entry.1, bytes, "aliases disagree about allocation capacity");
            entry.0 += 1;
        })
        .or_insert((1, bytes));
}

fn live_resources(scene: &Scene3d) -> ResourceMap {
    let mut resources = ResourceMap::new();
    for instance in &scene.instances {
        let mesh = &instance.mesh;
        record(
            &mut resources,
            0,
            mesh.source.vertices().as_ptr() as usize,
            mesh.source.recovery_memory_bytes(),
        );
        let gpu_bytes = mesh.vertex_buffer.size() as usize
            + [
                &mesh.index_buffer,
                &mesh.edge_buffer,
                &mesh.texture_coordinate_buffer,
                &mesh.color_buffer,
                &mesh.normal_buffer,
            ]
            .into_iter()
            .flatten()
            .map(|buffer| buffer.size() as usize)
            .sum::<usize>();
        record(
            &mut resources,
            1,
            Arc::as_ptr(&mesh.vertex_buffer) as usize,
            gpu_bytes,
        );
        if let Some(material) = mesh.material() {
            let texture = material.texture();
            record(
                &mut resources,
                2,
                texture.identity_key(),
                texture.recovery_memory_bytes(),
            );
            let texel_bytes = (0..texture.mip_level_count())
                .map(|level| {
                    let (width, height) = texture.mip_level_size(level).unwrap();
                    let bytes = width as usize * height as usize * 4;
                    assert_eq!(texture.mip_level_pixels(level).unwrap().len(), bytes);
                    bytes
                })
                .sum();
            record(&mut resources, 3, texture.identity_key(), texel_bytes);
        }
    }
    resources
}

fn assert_scene(scene: &Scene3d) {
    // The expected map starts at live allocation identities, independently of
    // both mesh_resources and the committed resource table being verified.
    let expected = live_resources(scene);
    assert_eq!(scene.resources.len(), expected.len());
    for (actual, (key, (references, bytes))) in scene.resources.iter().zip(&expected) {
        assert_eq!(
            (actual.key, actual.references, actual.bytes),
            (*key, *references, *bytes)
        );
    }
    let mut bytes = [0; 4];
    let mut counts = [0; 4];
    for ((category, _), (_, allocation_bytes)) in expected {
        bytes[usize::from(category)] += allocation_bytes;
        counts[usize::from(category)] += 1;
    }
    let expected_statistics = Scene3dStatistics {
        object_count: scene.instances.len(),
        slot_capacity: scene.slots.capacity(),
        storage_bytes: scene.instances.capacity() * std::mem::size_of::<Mesh3dInstance>()
            + scene.slots.capacity() * std::mem::size_of::<ObjectSlot>()
            + scene.resources.capacity() * std::mem::size_of::<ResourceUsage>(),
        mesh_count: counts[1],
        mesh_cpu_bytes: bytes[0],
        mesh_gpu_bytes: bytes[1],
        texture_count: counts[3],
        texture_cpu_bytes: bytes[2],
        texture_gpu_bytes: bytes[3],
    };
    assert_eq!(scene.statistics(), expected_statistics);
    assert_eq!(scene.resource_totals.bytes, bytes);
    assert_eq!(scene.resource_totals.counts, counts);
    let visible = scene
        .instances
        .iter()
        .filter(|instance| instance.visible)
        .count();
    assert_eq!(scene.visible_count, visible);
    assert_eq!(scene.visible_object_count(), visible);
    for instance in &scene.instances {
        assert_eq!(scene.instance(instance.id).unwrap().id(), instance.id);
    }
}

#[derive(Debug, PartialEq)]
struct ObjectState {
    id: Object3dId,
    source: usize,
    gpu: usize,
    texture: Option<usize>,
    pixels: Vec<Vec<u8>>,
    transform: Transform3d,
    style: MeshStyle3d,
    visible: bool,
}

#[derive(Debug, PartialEq)]
struct SceneState {
    resources: ResourceMap,
    statistics: Scene3dStatistics,
    objects: Vec<ObjectState>,
    next_object_id: u64,
}

fn state(scene: &Scene3d) -> SceneState {
    assert_scene(scene);
    SceneState {
        resources: live_resources(scene),
        statistics: scene.statistics(),
        objects: scene
            .instances
            .iter()
            .map(|instance| {
                let texture = instance.mesh.material().map(TextureMaterial3d::texture);
                ObjectState {
                    id: instance.id,
                    source: instance.mesh.source.vertices().as_ptr() as usize,
                    gpu: Arc::as_ptr(&instance.mesh.vertex_buffer) as usize,
                    texture: texture.map(Texture3d::identity_key),
                    pixels: texture.map_or_else(Vec::new, |texture| {
                        (0..texture.mip_level_count())
                            .map(|level| texture.mip_level_pixels(level).unwrap().to_vec())
                            .collect()
                    }),
                    transform: instance.transform,
                    style: instance.style,
                    visible: instance.visible,
                }
            })
            .collect(),
        next_object_id: scene.next_object_id,
    }
}

fn source(colored: bool, triangles: usize) -> Mesh3d {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for triangle in 0..triangles {
        let offset = triangle as f32 * 0.05;
        vertices.extend([
            Vec3::new(-0.5 + offset, -0.5, 0.0).unwrap(),
            Vec3::new(0.5 + offset, -0.5, 0.0).unwrap(),
            Vec3::new(offset, 0.5, 0.0).unwrap(),
        ]);
        indices.extend((triangle * 3..triangle * 3 + 3).map(|index| index as u32));
    }
    let mut attributes = Mesh3dAttributes::new().with_texture_coordinates(vec![
            TextureCoordinate2d::new(0.5, 0.5)
                .unwrap();
            vertices.len()
        ]);
    if colored {
        attributes = attributes
            .with_vertex_colors(vec![Color::WHITE; vertices.len()])
            .unwrap();
        attributes = attributes
            .with_normals(vec![Vec3::Z; vertices.len()])
            .unwrap();
    }
    Mesh3d::with_attributes(vertices, indices, Vec::new(), attributes).unwrap()
}

fn style() -> MeshStyle3d {
    MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE).unwrap())
}

struct Fixture<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    identity: Arc<()>,
    renderer: Mesh3dRenderer,
}

impl<'a> Fixture<'a> {
    fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            identity: Arc::new(()),
            renderer: Mesh3dRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb),
        }
    }

    fn mesh(&self, source: Mesh3d) -> RetainedMesh3d {
        create_retained_mesh(self.device, self.queue, Arc::clone(&self.identity), source).unwrap()
    }

    fn material(&self, capacity: usize) -> TextureMaterial3d {
        let mut pixels = Vec::with_capacity(capacity.max(64));
        pixels.resize(64, 255);
        let texture = texture::create_texture_with_options(
            self.device,
            self.queue,
            &self.identity,
            &self.renderer.textures.layout,
            4,
            4,
            pixels,
            ImageBudget::default(),
            Texture3dOptions::new().with_mipmaps(TextureMipmaps3d::Generate),
        )
        .unwrap();
        TextureMaterial3d::new(&texture, ImageSampling::Nearest, Color::WHITE).unwrap()
    }

    fn update_mesh(
        &mut self,
        scene: &mut Scene3d,
        id: Object3dId,
        source: Mesh3d,
    ) -> DynamicMesh3dUpdateReport {
        let report = dynamic::update_scene_mesh(
            self.device,
            self.queue,
            &self.identity,
            &mut self.renderer.dynamic_scratch,
            scene,
            id,
            source,
            DynamicMesh3dBudget::default().with_minimum_capacity(18, 18, 0),
        )
        .unwrap();
        assert_scene(scene);
        report
    }

    fn update_texture(
        &self,
        scene: &mut Scene3d,
        id: Object3dId,
        channel: u8,
    ) -> Texture3dUpdateReport {
        let report = texture::update_scene_texture_region(
            self.device,
            self.queue,
            &self.identity,
            &self.renderer.textures.layout,
            scene,
            id,
            ImageTexelRect::new(1, 1, 1, 1).unwrap(),
            &[channel, 91, 143, 255],
            4,
            Texture3dUpdateBudget::default(),
        )
        .unwrap();
        assert_scene(scene);
        report
    }

    fn restore(&self, scene: &mut Scene3d) -> Scene3dRestoreReport {
        let report = restore_scene3d_resources(
            self.device,
            self.queue,
            &self.renderer.textures.layout,
            Arc::clone(&self.identity),
            scene,
        )
        .unwrap();
        assert_scene(scene);
        report
    }
}

fn push(scene: &mut Scene3d, mesh: &RetainedMesh3d) -> Object3dId {
    let id = scene
        .try_push(mesh, Transform3d::IDENTITY, style())
        .unwrap();
    assert_scene(scene);
    id
}

fn verify_dynamic_commits(fixture: &mut Fixture<'_>) {
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let mesh = fixture.mesh(source(false, 1));
    let first = push(&mut scene, &mesh);
    let alias = push(&mut scene, &mesh);
    scene.set_visible(alias, false).unwrap();
    assert_scene(&scene);
    let report = fixture.update_mesh(&mut scene, first, source(false, 1));
    assert!(report.detached_aliases());
    let reused_key = Arc::as_ptr(&scene.instance(first).unwrap().mesh.vertex_buffer);
    let report = fixture.update_mesh(&mut scene, first, source(false, 2));
    assert!(report.reused_buffers());
    assert_eq!(
        Arc::as_ptr(&scene.instance(first).unwrap().mesh.vertex_buffer),
        reused_key
    );
    let report = fixture.update_mesh(&mut scene, first, source(true, 3));
    assert!(report.grew_capacity());
    assert!(!report.reused_buffers());
    fixture.update_mesh(&mut scene, first, source(false, 1));
    scene.remove(alias).unwrap();
    assert_scene(&scene);
    scene.remove(first).unwrap();
    assert_scene(&scene);

    let unique = {
        let material = fixture.material(1024);
        let mesh = texture::attach_material(&fixture.identity, &mesh, &material).unwrap();
        push(&mut scene, &mesh)
    };
    let old = state(&scene);
    let report = fixture.update_texture(&mut scene, unique, 40);
    assert!(report.reused_allocation());
    assert_eq!(state(&scene).objects[0].texture, old.objects[0].texture);
    assert!(scene.statistics().texture_cpu_bytes() < old.statistics.texture_cpu_bytes());
    let shared = scene.instance(unique).unwrap().mesh.clone();
    let alias = push(&mut scene, &shared);
    scene.set_visible(unique, false).unwrap();
    assert_scene(&scene);
    assert!(
        fixture
            .update_texture(&mut scene, unique, 50)
            .detached_aliases()
    );
    assert!(
        fixture
            .update_texture(&mut scene, unique, 60)
            .reused_allocation()
    );
    scene.remove(alias).unwrap();
    assert_scene(&scene);
    scene.remove(unique).unwrap();
    assert_scene(&scene);
}

fn verify_budget_boundaries(mesh: &RetainedMesh3d) {
    let mut probe = Scene3d::new(Color::BLACK).unwrap();
    push(&mut probe, mesh);
    let resources = live_resources(&probe);
    let mut bytes = [0; 4];
    for ((category, _), (_, size)) in &resources {
        bytes[usize::from(*category)] += size;
    }
    let storage = std::mem::size_of::<Mesh3dInstance>()
        + std::mem::size_of::<ObjectSlot>()
        + resources.len() * std::mem::size_of::<ResourceUsage>();
    let exact = Scene3dBudget::new(1, storage, bytes[0], bytes[1])
        .unwrap()
        .with_texture_limits(bytes[2], bytes[3]);
    let mut scene = Scene3d::with_budget(Color::BLACK, exact).unwrap();
    push(&mut scene, mesh);
    let before = state(&scene);
    assert!(matches!(
        scene.try_push(mesh, Transform3d::IDENTITY, style()),
        Err(Scene3dError::BudgetExceeded {
            resource: Scene3dBudgetResource::Objects,
            ..
        })
    ));
    assert_eq!(state(&scene), before);
    for resource in [
        Scene3dBudgetResource::StorageBytes,
        Scene3dBudgetResource::MeshCpuBytes,
        Scene3dBudgetResource::MeshGpuBytes,
        Scene3dBudgetResource::TextureCpuBytes,
        Scene3dBudgetResource::TextureGpuBytes,
    ] {
        let mut budget = exact;
        match resource {
            Scene3dBudgetResource::StorageBytes => budget.max_storage_bytes -= 1,
            Scene3dBudgetResource::MeshCpuBytes => budget.max_mesh_cpu_bytes -= 1,
            Scene3dBudgetResource::MeshGpuBytes => budget.max_mesh_gpu_bytes -= 1,
            Scene3dBudgetResource::TextureCpuBytes => budget.max_texture_cpu_bytes -= 1,
            Scene3dBudgetResource::TextureGpuBytes => budget.max_texture_gpu_bytes -= 1,
            Scene3dBudgetResource::Objects => unreachable!(),
        }
        let mut scene = Scene3d::with_budget(Color::BLACK, budget).unwrap();
        let before = state(&scene);
        assert!(matches!(
            scene.try_push(mesh, Transform3d::IDENTITY, style()),
            Err(Scene3dError::BudgetExceeded { resource: actual, .. }) if actual == resource
        ));
        assert_eq!(state(&scene), before);
    }
}

fn verify_rejections(fixture: &mut Fixture<'_>, mesh: &RetainedMesh3d) {
    let mut prepared_scene = Scene3d::new(Color::BLACK).unwrap();
    let prepared_id = push(&mut prepared_scene, mesh);
    push(&mut prepared_scene, mesh);
    let before_preparation = state(&prepared_scene);
    let incoming = source(true, 2);
    let prepared = prepared_scene
        .prepare_dynamic_mesh_change(prepared_id, &incoming, mesh.gpu_allocation_bytes(), true)
        .unwrap();
    assert!(prepared.resources.is_some());
    assert_eq!(state(&prepared_scene), before_preparation);
    drop(prepared);
    let failure = dynamic::update_scene_mesh(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &mut fixture.renderer.dynamic_scratch,
        &mut prepared_scene,
        prepared_id,
        incoming,
        DynamicMesh3dBudget::default().with_peak_limits(usize::MAX, usize::MAX, 0),
    );
    assert!(matches!(
        failure,
        Err(DynamicMesh3dError::BudgetExceeded {
            resource: DynamicMesh3dBudgetResource::PeakStagingBytes,
            ..
        })
    ));
    assert_eq!(state(&prepared_scene), before_preparation);

    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let id = push(&mut scene, mesh);
    let stale = push(&mut scene, mesh);
    scene.remove(stale).unwrap();
    assert_scene(&scene);
    let mut foreign_scene = Scene3d::new(Color::BLACK).unwrap();
    let foreign = push(&mut foreign_scene, mesh);
    for invalid in [stale, foreign] {
        let before = state(&scene);
        assert!(scene.set_visible(invalid, false).is_err());
        assert_eq!(state(&scene), before);
        assert!(scene.remove(invalid).is_err());
        assert_eq!(state(&scene), before);
        assert!(scene.set_mesh(invalid, mesh).is_err());
        assert_eq!(state(&scene), before);
        assert!(scene.set_texture_material(invalid, None).is_err());
        assert_eq!(state(&scene), before);
    }
    let plain = fixture.mesh(source(false, 1));
    let id = scene.set_mesh(id, &plain).map(|_| id).unwrap();
    assert_scene(&scene);
    let before = state(&scene);
    let invalid_style = MeshStyle3d::surface(
        SurfaceStyle3d::opaque(Color::WHITE)
            .unwrap()
            .with_lighting(SurfaceLighting3d::Lambert),
    );
    assert_eq!(
        scene.set_style(id, invalid_style),
        Err(Scene3dError::MissingNormals)
    );
    assert_eq!(state(&scene), before);
    assert_eq!(
        scene.try_push(&plain, Transform3d::IDENTITY, invalid_style),
        Err(Scene3dError::MissingNormals)
    );
    assert_eq!(state(&scene), before);
    let foreign_mesh = create_retained_mesh(
        fixture.device,
        fixture.queue,
        Arc::new(()),
        source(false, 1),
    )
    .unwrap();
    assert!(matches!(
        scene.set_mesh(id, &foreign_mesh),
        Err(Scene3dError::RendererMismatch)
    ));
    assert_eq!(state(&scene), before);
    let failure = dynamic::update_scene_mesh(
        fixture.device,
        fixture.queue,
        &fixture.identity,
        &mut fixture.renderer.dynamic_scratch,
        &mut scene,
        id,
        source(true, 2),
        DynamicMesh3dBudget::default().with_peak_limits(0, 0, 0),
    );
    assert!(failure.is_err());
    assert_eq!(state(&scene), before);
    let incoming = source(true, 2);
    let previous_budget = scene.budget;
    scene.budget.max_mesh_cpu_bytes = before.statistics.mesh_cpu_bytes();
    assert!(
        scene
            .prepare_dynamic_mesh_change(id, &incoming, plain.gpu_allocation_bytes(), true)
            .is_err()
    );
    assert_eq!(state(&scene), before);
    scene.budget = previous_budget;
    scene.set_mesh(id, mesh).unwrap();
    assert_scene(&scene);
    let before = state(&scene);
    assert!(
        texture::update_scene_texture_region(
            fixture.device,
            fixture.queue,
            &fixture.identity,
            &fixture.renderer.textures.layout,
            &mut scene,
            id,
            ImageTexelRect::new(0, 0, 1, 1).unwrap(),
            &[4, 5, 6, 255],
            4,
            Texture3dUpdateBudget::new(0, 0, 0, 0, 0),
        )
        .is_err()
    );
    assert_eq!(state(&scene), before);
    scene.budget.max_texture_cpu_bytes = before.statistics.texture_cpu_bytes();
    assert!(
        scene
            .prepare_dynamic_texture_change(
                id,
                before.statistics.texture_cpu_bytes() + 1,
                before.statistics.texture_gpu_bytes(),
                false
            )
            .is_err()
    );
    assert_eq!(state(&scene), before);
    scene.budget = previous_budget;
    push(&mut scene, mesh);
    let before = state(&scene);
    scene.budget.max_texture_gpu_bytes = before.statistics.texture_gpu_bytes();
    assert!(matches!(
        texture::update_scene_texture_region(
            fixture.device,
            fixture.queue,
            &fixture.identity,
            &fixture.renderer.textures.layout,
            &mut scene,
            id,
            ImageTexelRect::new(0, 0, 1, 1).unwrap(),
            &[4, 5, 6, 255],
            4,
            Texture3dUpdateBudget::default(),
        ),
        Err(Texture3dUpdateError::Scene(Scene3dError::BudgetExceeded {
            resource: Scene3dBudgetResource::TextureGpuBytes,
            ..
        }))
    ));
    assert_eq!(state(&scene), before);
}

fn verify_randomized_mutations(
    fixture: &mut Fixture<'_>,
    meshes: &[RetainedMesh3d],
    materials: &[TextureMaterial3d],
) {
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let mut random = 0xe7b4_912a_d63c_8f05_u64;
    let mut operations = [0_usize; 12];
    for step in 0..240 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let index = random as usize;
        if scene.object_count() == 0 || (index.is_multiple_of(12) && scene.object_count() < 12) {
            push(&mut scene, &meshes[(index >> 8) % meshes.len()]);
            operations[0] += 1;
            continue;
        }
        let id = scene.instances[(index >> 16) % scene.object_count()].id;
        operations[index % 12] += 1;
        match index % 12 {
            0 | 1 => {
                scene
                    .set_mesh(id, &meshes[(index >> 8) % meshes.len()])
                    .unwrap();
            }
            2 => {
                scene.remove(id).unwrap();
            }
            3 => {
                let visible = random & (1 << 32) != 0;
                scene.set_visible(id, visible).unwrap();
                assert_scene(&scene);
                scene.set_visible(id, visible).unwrap();
            }
            4 => {
                scene
                    .set_texture_material(id, Some(&materials[(index >> 8) % materials.len()]))
                    .unwrap();
            }
            5 => {
                scene.set_texture_material(id, None).unwrap();
            }
            6 => {
                fixture.update_mesh(&mut scene, id, source(step % 2 == 0, 1 + step % 3));
            }
            7 => {
                if scene.instance(id).unwrap().mesh.material().is_none() {
                    scene.set_texture_material(id, Some(&materials[0])).unwrap();
                    assert_scene(&scene);
                }
                fixture.update_texture(&mut scene, id, step as u8);
            }
            8 => {
                let before = state(&scene);
                assert!(
                    dynamic::update_scene_mesh(
                        fixture.device,
                        fixture.queue,
                        &fixture.identity,
                        &mut fixture.renderer.dynamic_scratch,
                        &mut scene,
                        id,
                        source(true, 2),
                        DynamicMesh3dBudget::default().with_peak_limits(0, 0, 0),
                    )
                    .is_err()
                );
                assert_eq!(state(&scene), before);
            }
            9 => {
                scene.set_transform(id, Transform3d::IDENTITY).unwrap();
            }
            10 => {
                scene.set_style(id, style()).unwrap();
            }
            11 => {
                scene.set_background(Color::rgb(0.25, 0.5, 0.75)).unwrap();
            }
            _ => unreachable!(),
        }
        assert_scene(&scene);
    }
    assert!(operations.into_iter().all(|count| count > 0));
    while let Some(id) = scene.instances.first().map(|instance| instance.id) {
        scene.remove(id).unwrap();
        assert_scene(&scene);
    }
}

fn verify_recovery(
    fixture: &mut Fixture<'_>,
    replacement: &mut Fixture<'_>,
    meshes: &[RetainedMesh3d],
    materials: &[TextureMaterial3d],
) {
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let first = push(&mut scene, &meshes[0]);
    let second = push(&mut scene, &meshes[1]);
    let third = push(&mut scene, &meshes[0]);
    for (id, material) in [
        (first, &materials[0]),
        (second, &materials[1]),
        (third, &materials[0]),
    ] {
        scene.set_texture_material(id, Some(material)).unwrap();
        assert_scene(&scene);
    }
    scene.set_visible(second, false).unwrap();
    assert_scene(&scene);
    let before = state(&scene);
    assert_eq!(scene.resource_totals.counts, [1, 2, 1, 1]);
    assert_eq!(fixture.restore(&mut scene).restored_mesh_count(), 0);
    assert_eq!(state(&scene).resources, before.resources);
    let report = replacement.restore(&mut scene);
    assert_eq!(report.restored_mesh_count(), 2);
    assert_eq!(report.restored_texture_count(), 1);
    assert_eq!(report.migrated_object_count(), 3);
    assert_eq!(scene.resource_totals.counts, [1, 2, 1, 1]);
    for previous in &before.objects {
        let instance = scene.instance(previous.id).unwrap();
        assert_eq!(instance.visible, previous.visible);
        assert_eq!(instance.transform, previous.transform);
        assert_eq!(instance.style, previous.style);
        assert!(Arc::ptr_eq(
            &instance.mesh.renderer_identity,
            &replacement.identity
        ));
        assert_ne!(
            Arc::as_ptr(&instance.mesh.vertex_buffer) as usize,
            previous.gpu
        );
        let texture = instance.mesh.material().unwrap().texture();
        assert_ne!(Some(texture.identity_key()), previous.texture);
        for (level, pixels) in previous.pixels.iter().enumerate() {
            assert_eq!(texture.mip_level_pixels(level as u32).unwrap(), pixels);
        }
    }
    for (id, material) in [(first, &materials[0]), (second, &materials[1])] {
        let restored = scene.instance(id).unwrap().mesh.material().unwrap();
        assert_eq!(restored.sampling(), material.sampling());
        assert_eq!(restored.tint(), material.tint());
        assert_eq!(restored.address_mode(), material.address_mode());
        assert_eq!(restored.uv_transform(), material.uv_transform());
    }
    let restored = state(&scene);
    assert_eq!(replacement.restore(&mut scene).restored_mesh_count(), 0);
    assert_eq!(state(&scene), restored);
    replacement.update_mesh(&mut scene, second, source(true, 2));
    assert!(
        replacement
            .update_texture(&mut scene, first, 70)
            .detached_aliases()
    );
    assert!(
        replacement
            .update_texture(&mut scene, first, 80)
            .reused_allocation()
    );
    scene.set_texture_material(third, None).unwrap();
    assert_scene(&scene);
    scene.remove(second).unwrap();
    assert_scene(&scene);
    let surviving = scene.instance(first).unwrap().mesh.clone();
    push(&mut scene, &surviving);
    assert_eq!(scene.visible_object_count(), 3);
}

pub(in crate::renderer::mesh3d) fn verify_scene_accounting(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    recovery_device: &wgpu::Device,
    recovery_queue: &wgpu::Queue,
) {
    let mut fixture = Fixture::new(device, queue);
    let shared_source = source(false, 1);
    let meshes = [
        fixture.mesh(shared_source.clone()),
        fixture.mesh(shared_source),
        fixture.mesh(source(true, 2)),
        fixture.mesh(source(false, 3)),
    ];
    let material = fixture.material(84);
    let alternate_policy = TextureMaterial3d::with_alpha(
        material.texture(),
        ImageSampling::Linear,
        Color::rgba(0.5, 0.75, 1.0, 0.5),
    )
    .unwrap()
    .with_address_mode(TextureAddressMode3d::Repeat);
    let materials = [material, alternate_policy, fixture.material(84)];
    let textured = texture::attach_material(&fixture.identity, &meshes[0], &materials[0]).unwrap();
    verify_budget_boundaries(&textured);
    verify_rejections(&mut fixture, &textured);
    verify_dynamic_commits(&mut fixture);
    verify_randomized_mutations(&mut fixture, &meshes, &materials);
    let mut replacement = Fixture::new(recovery_device, recovery_queue);
    verify_recovery(&mut fixture, &mut replacement, &meshes, &materials);
}
