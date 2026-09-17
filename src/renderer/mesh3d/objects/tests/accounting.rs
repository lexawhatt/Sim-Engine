use super::*;
use std::collections::BTreeMap;

impl Scene3d {
    pub(in crate::renderer::mesh3d) fn set_visible_linear_reference(
        &mut self,
        object_id: Object3dId,
        visible: bool,
    ) -> Result<(), Scene3dError> {
        let instance = self
            .instances
            .iter_mut()
            .find(|instance| instance.id == object_id)
            .ok_or(Scene3dError::ObjectNotFound { object_id })?;
        if instance.visible != visible {
            if visible {
                self.visible_count += 1;
            } else {
                self.visible_count -= 1;
            }
            instance.visible = visible;
        }
        Ok(())
    }
}

fn assert_scanned_totals(scene: &Scene3d) {
    let mut bytes = [0; 4];
    let mut counts = [0; 4];
    for resource in &scene.resources {
        assert!(resource.bytes > 0);
        assert!(resource.references > 0);
        let category = usize::from(resource.key.0);
        bytes[category] += resource.bytes;
        counts[category] += 1;
    }
    assert!(
        scene
            .resources
            .windows(2)
            .all(|pair| pair[0].key < pair[1].key)
    );
    assert_eq!(scene.resource_totals.bytes, bytes);
    assert_eq!(scene.resource_totals.counts, counts);
    let actual = scene.statistics();
    assert_eq!(actual.mesh_cpu_bytes(), bytes[0]);
    assert_eq!(actual.mesh_gpu_bytes(), bytes[1]);
    assert_eq!(actual.texture_cpu_bytes(), bytes[2]);
    assert_eq!(actual.texture_gpu_bytes(), bytes[3]);
    assert_eq!(actual.mesh_count(), counts[1]);
    assert_eq!(actual.texture_count(), counts[3]);
}

#[test]
fn resource_totals_follow_references_and_stored_bytes() {
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    assert_scanned_totals(&scene);
    for category in 0..4 {
        let original = ResourceUsage {
            key: (category, 41),
            references: 1,
            bytes: 17 + usize::from(category),
        };
        scene.add_resource(original);
        assert_scanned_totals(&scene);
        scene.add_resource(ResourceUsage {
            bytes: 0,
            ..original
        });
        scene.add_resource(ResourceUsage {
            key: (category, 0),
            bytes: 0,
            ..original
        });
        scene.remove_resource((category, usize::MAX));
        assert_eq!(scene.resources.last().unwrap().references, 1);
        assert_scanned_totals(&scene);

        // An identity keeps its recorded size until its final reference retires.
        let changed = ResourceUsage {
            bytes: 4096,
            ..original
        };
        scene.add_resource(changed);
        assert_scanned_totals(&scene);
        scene.remove_resource(changed.key);
        assert_eq!(
            scene.resource_totals.bytes[usize::from(category)],
            original.bytes
        );
        assert_scanned_totals(&scene);
        scene.remove_resource(changed.key);
        assert_eq!(scene.resource_totals.bytes[usize::from(category)], 0);
        assert_scanned_totals(&scene);
        scene.add_resource(changed);
        assert_eq!(scene.resource_totals.bytes[usize::from(category)], 4096);
        assert_scanned_totals(&scene);
        scene.remove_resource(changed.key);
        scene.remove_resource(changed.key);
        assert_scanned_totals(&scene);
    }
    assert!(scene.resources.is_empty());
}

#[test]
fn randomized_resource_totals_match_an_independent_reference_table() {
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let mut reference = BTreeMap::<(u8, usize), (usize, usize)>::new();
    let mut random = 0x86a3_67db_103b_5f91_u64;
    for step in 0..4096 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let key = (((random >> 8) % 4) as u8, (random % 31) as usize);
        if random & 4 == 0 {
            scene.remove_resource(key);
            if let Some((references, _)) = reference.get_mut(&key) {
                *references -= 1;
                if *references == 0 {
                    reference.remove(&key);
                }
            }
        } else {
            let bytes = if step % 19 == 0 { 0 } else { 1 + step % 127 };
            scene.add_resource(ResourceUsage {
                key,
                references: 1,
                bytes,
            });
            if bytes > 0 {
                reference
                    .entry(key)
                    .and_modify(|entry| entry.0 += 1)
                    .or_insert((1, bytes));
            }
        }
        assert_scanned_totals(&scene);
        assert_eq!(scene.resources.len(), reference.len());
        for (actual, (key, (references, bytes))) in scene.resources.iter().zip(&reference) {
            assert_eq!(
                (actual.key, actual.references, actual.bytes),
                (*key, *references, *bytes)
            );
        }
    }
    for (key, (references, _)) in reference {
        for _ in 0..references {
            scene.remove_resource(key);
            assert_scanned_totals(&scene);
        }
    }
    assert_eq!(scene.resource_totals.bytes, [0; 4]);
    assert_eq!(scene.resource_totals.counts, [0; 4]);
}

#[test]
fn resource_preparation_and_capacity_replacement_do_not_publish_totals() {
    let mut scene = Scene3d::new(Color::BLACK).unwrap();
    let outgoing = ResourceUsage {
        key: (0, 42),
        references: 1,
        bytes: 60,
    };
    let incoming = ResourceUsage {
        key: (0, 84),
        references: 1,
        bytes: 40,
    };
    scene.add_resource(outgoing);
    scene.budget.max_mesh_cpu_bytes = 60;
    assert_eq!(
        scene.validate_resource_change(&[incoming], Some(&[outgoing])),
        Ok([100, 0, 0, 0])
    );
    assert_scanned_totals(&scene);
    scene.add_resource(outgoing);
    assert!(matches!(
        scene.validate_resource_change(&[incoming], Some(&[outgoing])),
        Err(Scene3dError::BudgetExceeded {
            resource: Scene3dBudgetResource::MeshCpuBytes,
            actual: 100,
            ..
        })
    ));
    assert_scanned_totals(&scene);
    let before = scene.resource_totals.bytes;
    scene.reserve_storage(0, 0, 12).unwrap();
    assert_scanned_totals(&scene);
    assert_eq!(scene.resource_totals.bytes, before);
    scene.rebuild_resource_keys();
    assert_scanned_totals(&scene);
    assert_eq!(scene.resource_totals.bytes, [0; 4]);
    assert_eq!(scene.visible_object_count(), 0);
}
