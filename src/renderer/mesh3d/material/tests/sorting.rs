use super::*;

fn draw(index: usize) -> SurfaceDraw {
    SurfaceDraw {
        scene_index: index,
        visible_index: index,
        depth: index as f64,
        blended: true,
    }
}

#[test]
fn sorting_scratch_reuses_capacity_and_replaces_an_oversized_idle_allocation() {
    let mut order = Vec::with_capacity(32);
    let original = order.as_ptr();
    let budget = Mesh3dRenderBudget::default();
    prepare_order_storage(&mut order, 8, budget, |order| {
        order.extend((0..8).map(draw));
        Ok(())
    })
    .unwrap();
    assert_eq!(order.as_ptr(), original);
    assert_eq!(order.len(), 8);
    let single = std::mem::size_of::<SurfaceDraw>();
    prepare_order_storage(
        &mut order,
        1,
        budget.with_max_sorting_bytes(single),
        |order| {
            order.push(draw(7));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(order.capacity(), 1);
    assert_eq!(order[0].scene_index, 7);
}

#[test]
fn sorting_scratch_failure_clears_records_and_preserves_replaced_storage() {
    let mut order = Vec::with_capacity(32);
    order.push(draw(0));
    let original = order.as_ptr();
    let capacity = order.capacity();
    for budget in [
        Mesh3dRenderBudget::default(),
        Mesh3dRenderBudget::default().with_max_sorting_bytes(std::mem::size_of::<SurfaceDraw>()),
    ] {
        assert_eq!(
            prepare_order_storage(&mut order, 1, budget, |order| {
                order.push(draw(9));
                Err(Mesh3dRenderError::InvalidGeometryTransform)
            }),
            Err(Mesh3dRenderError::InvalidGeometryTransform)
        );
        assert!(order.is_empty());
        assert_eq!(order.as_ptr(), original);
        assert_eq!(order.capacity(), capacity);
    }
    assert_eq!(
        prepare_order_storage(
            &mut order,
            1,
            Mesh3dRenderBudget::default().with_max_sorting_bytes(0),
            |_| panic!("count budget must fail before population"),
        ),
        Err(Mesh3dRenderError::SortingBudgetExceeded {
            limit: 0,
            actual: std::mem::size_of::<SurfaceDraw>(),
        })
    );
    assert!(order.is_empty());
    assert_eq!(order.as_ptr(), original);
    assert_eq!(order.capacity(), capacity);
}

#[test]
fn sorting_scratch_no_blend_ignores_idle_capacity_under_zero_budget() {
    let mut order = vec![draw(0); 32];
    let original = order.as_ptr();
    let capacity = order.capacity();
    let scene = Scene3d::new(Color::BLACK).unwrap();
    let camera = Camera3d::look_at(
        Vec3::new(0.0, 0.0, 3.0).unwrap(),
        Vec3::ZERO,
        Vec3::Y,
        Projection3d::orthographic(world(2.0), 1.0, world(0.5), world(8.0)).unwrap(),
    )
    .unwrap();
    prepare_order_into(
        &scene,
        camera,
        Mesh3dRenderBudget::default().with_max_sorting_bytes(0),
        &mut order,
    )
    .unwrap();
    assert!(order.is_empty());
    assert_eq!(order.as_ptr(), original);
    assert_eq!(order.capacity(), capacity);
}

#[test]
fn surface_depth_order_keeps_opaque_first_and_insertion_ties() {
    let make = |index, depth, blended| SurfaceDraw {
        scene_index: index,
        visible_index: index,
        depth,
        blended,
    };
    let mut values = [
        make(0, 1.0, true),
        make(1, 0.0, false),
        make(2, 5.0, true),
        make(3, 5.0, true),
    ];
    values.sort_unstable_by(compare_draws);
    assert_eq!(values.map(|value| value.scene_index), [1, 2, 3, 0]);
}

#[test]
fn signed_zero_depth_uses_insertion_order_for_both_sign_orders() {
    for sign in [-1.0, 1.0] {
        let mut values = std::array::from_fn::<_, 4, _>(|index| SurfaceDraw {
            scene_index: index,
            visible_index: index,
            depth: if index % 2 == 0 {
                sign * 0.0
            } else {
                -sign * 0.0
            },
            blended: true,
        });
        values.reverse();
        values.sort_unstable_by(compare_draws);
        assert_eq!(values.map(|value| value.scene_index), [0, 1, 2, 3]);
    }
}
#[test]
fn alpha_background_requires_explicit_constructor_and_normalized_channels() {
    let color = Color::rgba(0.25, 0.5, 0.75, 0.5);
    assert!(Scene3d::new(color).is_err());
    assert_eq!(
        Scene3d::with_alpha_background(color).unwrap().background(),
        color
    );
    assert!(Scene3d::with_alpha_background(Color::rgba(0.0, f32::NAN, 0.0, 0.0)).is_err());
    let clear = premultiplied_wgpu_color(color);
    assert_eq!(
        [clear.r, clear.g, clear.b, clear.a],
        [0.125, 0.25, 0.375, 0.5]
    );
}
