//! Bounded transient draw ordering and material uniform encoding.

use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct SurfaceDraw {
    pub(super) scene_index: usize,
    pub(super) visible_index: usize,
    depth: f64,
    blended: bool,
}

pub(super) fn surface_parameters(style: Option<crate::SurfaceStyle3d>) -> [f32; 4] {
    style.map_or([0.0; 4], |style| {
        [
            match style.alpha_mode() {
                SurfaceAlphaMode3d::Opaque => 0.0,
                SurfaceAlphaMode3d::Mask => 1.0,
                SurfaceAlphaMode3d::Blend => 2.0,
            },
            style.mask_cutoff().unwrap_or(0.0),
            if style.lighting() == SurfaceLighting3d::Lambert {
                1.0
            } else {
                0.0
            },
            if style.fog_enabled() { 1.0 } else { 0.0 },
        ]
    })
}

pub(super) fn prepare_order(
    scene: &Scene3d,
    camera: Camera3d,
    budget: Mesh3dRenderBudget,
) -> Result<Vec<SurfaceDraw>, Mesh3dRenderError> {
    if !scene.instances().iter().any(|instance| {
        instance.visible
            && instance
                .style
                .surface_style()
                .is_some_and(|surface| surface.alpha_mode() == SurfaceAlphaMode3d::Blend)
    }) {
        return Ok(Vec::new());
    }
    let count = scene.visible_object_count();
    let check = |capacity: usize| {
        let actual = capacity
            .checked_mul(std::mem::size_of::<SurfaceDraw>())
            .ok_or(Mesh3dRenderError::InstanceCapacityTooLarge)?;
        if actual > budget.max_sorting_bytes() {
            return Err(Mesh3dRenderError::SortingBudgetExceeded {
                limit: budget.max_sorting_bytes(),
                actual,
            });
        }
        Ok(())
    };
    check(count)?;
    let mut order = Vec::new();
    order
        .try_reserve_exact(count)
        .map_err(|_| Mesh3dRenderError::InstanceCapacityTooLarge)?;
    check(order.capacity())?;
    for (visible_index, (scene_index, instance)) in scene
        .instances()
        .iter()
        .enumerate()
        .filter(|(_, instance)| instance.visible)
        .enumerate()
    {
        let blended = instance
            .style
            .surface_style()
            .is_some_and(|surface| surface.alpha_mode() == SurfaceAlphaMode3d::Blend);
        let depth = if blended {
            let rows = instance
                .transform
                .model_rows()
                .map_err(|_| Mesh3dRenderError::InvalidGeometryTransform.for_object(instance.id))?;
            bounds_depth(instance.mesh.source(), rows, camera)
        } else {
            0.0
        };
        order.push(SurfaceDraw {
            scene_index,
            visible_index,
            depth,
            blended,
        });
    }
    // Explicit insertion tie-break makes an allocation-free unstable sort stable
    // for the public contract, while retaining visible indices for GPU uniforms.
    order.sort_unstable_by(compare_draws);
    Ok(order)
}

fn compare_draws(left: &SurfaceDraw, right: &SurfaceDraw) -> std::cmp::Ordering {
    left.blended
        .cmp(&right.blended)
        .then_with(|| {
            // Signed zero is one geometric depth, not two ordering buckets.
            if left.depth == right.depth {
                std::cmp::Ordering::Equal
            } else {
                right.depth.total_cmp(&left.depth)
            }
        })
        .then_with(|| left.visible_index.cmp(&right.visible_index))
}

fn bounds_depth(source: &Mesh3d, model: [[f32; 4]; 3], camera: Camera3d) -> f64 {
    let minimum = source.bounds_min();
    let maximum = source.bounds_max();
    let center = [
        (f64::from(minimum.x()) + f64::from(maximum.x())) * 0.5,
        (f64::from(minimum.y()) + f64::from(maximum.y())) * 0.5,
        (f64::from(minimum.z()) + f64::from(maximum.z())) * 0.5,
        1.0,
    ];
    let world = model.map(|row| {
        row.into_iter()
            .zip(center)
            .map(|(a, b)| f64::from(a) * b)
            .sum::<f64>()
    });
    let position = camera.position();
    let forward = camera.forward();
    // Products of finite f32 source/model coefficients remain bounded in f64;
    // never reduce a potentially distant bounds center back to f32 for sorting.
    (world[0] - f64::from(position.x())) * f64::from(forward.x())
        + (world[1] - f64::from(position.y())) * f64::from(forward.y())
        + (world[2] - f64::from(position.z())) * f64::from(forward.z())
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
