use super::*;

#[test]
fn particle_gpu_instances_preserve_counts_and_capacity_contract() {
    let particle = ParticleInstance2d::new(Vec2::new(2.0, -3.0), 2.0, Color::WHITE, 4.0)
        .expect("finite particle should be valid");
    let gpu_instances = particle_instances_to_gpu(&[particle]).unwrap();
    assert_eq!(gpu_instances.len(), 1);
    assert_eq!(particle_instance_capacity(0), Some(1));
    assert_eq!(particle_instance_capacity(3), Some(4));
    let statistics = particle_statistics(gpu_instances.len(), gpu_instances.len(), 1);
    assert_eq!(statistics.submitted(), 1);
    assert_eq!(statistics.visible(), 1);
    assert_eq!(statistics.culled(), 0);
    assert_eq!(statistics.dropped(), 0);
    assert_eq!(statistics.rendered(), 1);
    let subnormal =
        ParticleInstance2d::new(Vec2::new(f32::from_bits(1), 0.0), 1.0, Color::WHITE, 0.0).unwrap();
    assert!(matches!(
        particle_instances_to_gpu(&[subnormal]),
        Err(ParticleFieldError::InvalidInstance)
    ));
}

#[test]
fn particle_budget_caps_memory_upload_and_samples_evenly() {
    let instance_bytes = std::mem::size_of::<ParticleGpu>();
    assert_eq!(
        ParticleRenderBudget::new(0, instance_bytes * 2, instance_bytes, instance_bytes),
        Err(ParticleBudgetError::InvalidLimit)
    );
    assert_eq!(
        ParticleRenderBudget::new(1, instance_bytes * 2 - 1, instance_bytes, instance_bytes,),
        Err(ParticleBudgetError::InvalidLimit)
    );
    let minimum_budget =
        ParticleRenderBudget::new(1, instance_bytes * 2, instance_bytes, instance_bytes).unwrap();
    assert_eq!(validate_particle_retained_count(1, minimum_budget), Ok(()));
    assert_eq!(
        validate_particle_retained_capacities(1, 1, minimum_budget),
        Ok(())
    );
    let budget = ParticleRenderBudget::new(
        30,
        instance_bytes * 112,
        instance_bytes * 20,
        instance_bytes * 12,
    )
    .expect("budget fits at least one instance");
    assert_eq!(budget.max_retained_bytes(), instance_bytes * 112);
    assert_eq!(budget.instance_limit(), 12);
    assert_eq!(particle_budgeted_capacity(100, budget), Some(12));
    assert_eq!(
        budget.with_max_visibility_checks(11),
        Err(ParticleBudgetError::InvalidLimit)
    );
    let bounded_checks = budget.with_max_visibility_checks(12).unwrap();
    assert_eq!(bounded_checks.max_visibility_checks_per_frame(), 12);
    assert_eq!(
        validate_particle_retained_count(100, bounded_checks),
        Ok(())
    );
    assert_eq!(
        validate_particle_retained_count(101, bounded_checks),
        Err(ParticleFieldError::RetainedBudgetExceeded {
            limit: instance_bytes * 112,
            actual: instance_bytes * 113,
        })
    );

    let selected: Vec<_> = (0..100)
        .filter(|index| particle_visible_index_is_selected(*index, 100, 12))
        .collect();
    assert_eq!(selected.len(), 12);
    assert!(selected.windows(2).all(|pair| pair[1] > pair[0]));

    let candidates: Vec<_> = (0..12)
        .map(|index| uniformly_sampled_index(index, 100, 12))
        .collect();
    assert_eq!(candidates.last(), Some(&99));
    assert!(candidates.windows(2).all(|pair| pair[1] > pair[0]));

    let statistics = particle_statistics_with_budget(140, 140, 100, 12, 12);
    assert_eq!(statistics.culled(), 40);
    assert_eq!(statistics.budget_limited(), 88);
    assert_eq!(statistics.rendered(), 12);

    let bounded_statistics = particle_statistics_with_budget(140, 12, 12, 12, 12);
    assert_eq!(bounded_statistics.visibility_checked(), 12);
    assert_eq!(bounded_statistics.budget_limited(), 128);
}

#[test]
fn particle_culling_keeps_circles_that_intersect_the_logical_viewport() {
    let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    let viewport = LogicalViewport::new(100.0, 100.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport).unwrap();
    let visible = ParticleGpu {
        world_position: [55.0, 0.0],
        depth: 0.0,
        radius: 6.0,
        color: Color::WHITE.to_array(),
    };
    let culled = ParticleGpu {
        world_position: [57.0, 0.0],
        depth: 0.0,
        radius: 6.0,
        color: Color::WHITE.to_array(),
    };
    assert!(visible.is_safe_for(uniform, viewport));
    assert!(visible.intersects_viewport(uniform, viewport));
    assert!(!culled.intersects_viewport(uniform, viewport));
}

#[test]
fn particle_safety_includes_radius_in_final_clip_arithmetic() {
    let camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    let viewport = LogicalViewport::new(1.0, 1.0).unwrap();
    let uniform = CameraUniform::new(camera, viewport).unwrap();
    let particle = ParticleGpu {
        world_position: [0.0, 0.0],
        depth: 0.0,
        radius: 2.0e38,
        color: Color::WHITE.to_array(),
    };

    assert!(!particle.is_safe_for(uniform, viewport));
}

#[test]
fn visible_particle_rejects_backend_dependent_dot_overflow() {
    let viewport = LogicalViewport::new(1.0e30, 1.0e30).unwrap();
    let mut camera = Camera2d::new(Vec2::ZERO, 3.830_149e16).unwrap();
    camera.set_rotation(1.378_629).unwrap();
    camera.set_projection(crate::Projection2d::new(-0.124_491_87, -1.0).unwrap());
    let uniform = CameraUniform::new(camera, viewport).unwrap();
    let particle = ParticleGpu {
        world_position: [4.914_133_7e21, 5.089_416e21],
        depth: -7.576_658e22,
        radius: 1.0,
        color: Color::WHITE.to_array(),
    };

    assert!(particle.screen_position(uniform).is_finite());
    assert!(!particle.is_safe_for(uniform, viewport));
}

#[test]
fn particle_validation_rejects_extreme_camera_clip_configuration() {
    let depth = 0.9 * f32::MAX;
    let tilt = std::f32::consts::FRAC_PI_4;
    let particle = ParticleGpu {
        world_position: [-depth * tilt.sin() * 0.5, -depth * tilt.sin() / tilt.cos()],
        depth,
        radius: 1.0,
        color: Color::WHITE.to_array(),
    };
    let mut camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    camera.set_projection(crate::Projection2d::new(tilt, 1.0).unwrap());
    let tiny_extent = 8.0 / f32::MAX;
    let viewport = LogicalViewport::new(tiny_extent, tiny_extent).unwrap();
    assert!(CameraUniform::new(camera, viewport).is_none());
    assert!(particle.world_position.into_iter().all(f32::is_finite));
}

#[test]
fn particle_partial_updates_require_an_existing_contiguous_range() {
    assert_eq!(particle_update_range(2, 3, 5), Ok(2..5));
    assert_eq!(particle_update_range(5, 0, 5), Ok(5..5));
    assert_eq!(
        particle_update_range(4, 2, 5),
        Err(ParticleFieldError::UpdateRangeOutOfBounds)
    );
    assert_eq!(
        particle_update_range(usize::MAX, 1, 5),
        Err(ParticleFieldError::UpdateRangeOutOfBounds)
    );
}

#[test]
fn particle_rejects_backend_dependent_viewport_classification() {
    let viewport_extent = 2.0_f32.powi(45);
    let viewport = LogicalViewport::new(viewport_extent, viewport_extent).unwrap();
    let mut camera = Camera2d::new(Vec2::ZERO, 1.0).unwrap();
    camera.set_projection(crate::Projection2d::new(std::f32::consts::FRAC_PI_4, 1.0).unwrap());
    let uniform = CameraUniform::new(camera, viewport).unwrap();
    let particle = ParticleGpu {
        world_position: [f32::from_bits(0xe1b5_00f9), f32::from_bits(0xe27f_fa60)],
        depth: f32::from_bits(0x627f_fa61),
        radius: 2.0_f32.powi(43),
        color: Color::WHITE.to_array(),
    };

    assert_eq!(
        particle.screen_position(uniform),
        Vec2::splat(viewport_extent * 0.5)
    );
    let fused = Vec2::new(
        uniform.world_to_screen_x[2].mul_add(
            particle.depth,
            uniform.world_to_screen_x[0] * particle.world_position[0]
                + uniform.world_to_screen_x[1] * particle.world_position[1],
        ) + uniform.world_to_screen_x[3],
        uniform.world_to_screen_y[2].mul_add(
            particle.depth,
            uniform.world_to_screen_y[0] * particle.world_position[0]
                + uniform.world_to_screen_y[1] * particle.world_position[1],
        ) + uniform.world_to_screen_y[3],
    );
    assert!(fused.y + particle.radius < 0.0);
    assert!(!particle.is_safe_for(uniform, viewport));
    assert_eq!(
        particle.validated_viewport_intersection(uniform, viewport),
        None
    );
}

#[test]
fn particle_quad_rejects_unrepresentable_large_viewport_origin() {
    let local_viewport = LogicalViewport::new(100.0, 100.0).unwrap();
    let target_extent = 2.0_f32.powi(45);
    let target_viewport = LogicalViewport::new(target_extent, target_extent).unwrap();
    let origin = Vec2::new(target_extent * 0.5, 100.0);
    let uniform =
        CameraUniform::new_in_region(Camera2d::default(), local_viewport, origin, target_viewport)
            .unwrap();
    let particle = ParticleGpu {
        world_position: [0.0, 0.0],
        depth: 0.0,
        radius: 20.0,
        color: Color::WHITE.to_array(),
    };
    let center = particle.screen_position(uniform);
    assert_eq!(center.x + particle.radius, center.x);
    let clip_center = center
        .x
        .mul_add(uniform.screen_to_clip[0], uniform.screen_to_clip[2]);
    let clip_radius = particle.radius * uniform.screen_to_clip[0];
    assert!(clip_radius.is_normal() && clip_center + clip_radius != clip_center);
    // The vertical center is near clip +1, where the same tiny radius cannot
    // produce two distinct f32 positions. Reject the whole quad rather than
    // letting one axis collapse backend-dependently.
    assert!(!particle.is_safe_for(uniform, target_viewport));
}
