//! Reusable bounded draw ordering and material uniform encoding.

use super::{
    Camera3d, Mesh3d, Mesh3dRenderBudget, Mesh3dRenderError, Scene3d, SurfaceAlphaMode3d,
    SurfaceLighting3d,
};

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

#[cfg(test)]
pub(super) fn prepare_order(
    scene: &Scene3d,
    camera: Camera3d,
    budget: Mesh3dRenderBudget,
) -> Result<Vec<SurfaceDraw>, Mesh3dRenderError> {
    let mut order = Vec::new();
    prepare_order_into(scene, camera, budget, &mut order)?;
    Ok(order)
}

pub(super) fn prepare_order_into(
    scene: &Scene3d,
    camera: Camera3d,
    budget: Mesh3dRenderBudget,
    order: &mut Vec<SurfaceDraw>,
) -> Result<(), Mesh3dRenderError> {
    order.clear();
    if !scene.instances().iter().any(|instance| {
        instance.visible
            && instance
                .style
                .surface_style()
                .is_some_and(|surface| surface.alpha_mode() == SurfaceAlphaMode3d::Blend)
    }) {
        return Ok(());
    }
    let count = scene.visible_object_count();
    prepare_order_storage(order, count, budget, |order| {
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
                let rows = instance.transform.model_rows().map_err(|_| {
                    Mesh3dRenderError::InvalidGeometryTransform.for_object(instance.id)
                })?;
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
        Ok(())
    })
}

fn prepare_order_storage(
    order: &mut Vec<SurfaceDraw>,
    count: usize,
    budget: Mesh3dRenderBudget,
    populate: impl FnOnce(&mut Vec<SurfaceDraw>) -> Result<(), Mesh3dRenderError>,
) -> Result<(), Mesh3dRenderError> {
    order.clear();
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

    // A smaller current budget constrains active sorting, not an old idle
    // high-water allocation. Commit its bounded replacement only on success.
    if check(order.capacity()).is_err() {
        let mut replacement = Vec::new();
        replacement
            .try_reserve_exact(count)
            .map_err(|_| Mesh3dRenderError::InstanceCapacityTooLarge)?;
        check(replacement.capacity())?;
        populate(&mut replacement)?;
        *order = replacement;
        return Ok(());
    }
    order
        .try_reserve_exact(count)
        .map_err(|_| Mesh3dRenderError::InstanceCapacityTooLarge)?;
    check(order.capacity())?;
    let result = populate(order);
    if result.is_err() {
        order.clear();
    }
    result
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
mod tests;
