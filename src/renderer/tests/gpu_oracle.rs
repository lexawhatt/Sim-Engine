use super::gpu_pixels::{
    assert_gpu_large_center_circle, assert_gpu_stroke_pixel_matrix, gpu_oracle_channel_indices,
    parse_gpu_oracle_surface_format,
};
use super::*;

pub(super) fn run() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = if let Ok(required_pci_bus_id) =
            std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID")
        {
            instance
                .enumerate_adapters(wgpu::Backends::all())
                .await
                .into_iter()
                .find(|candidate| candidate.get_info().device_pci_bus_id == required_pci_bus_id)
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                })
                .await
                .ok()
        };
        let Some(adapter) = adapter else {
            assert_ne!(
                std::env::var("SIM_ENGINE_REQUIRE_GPU_TESTS").as_deref(),
                Ok("1"),
                "the required physical GPU adapter is unavailable"
            );
            return;
        };
        let adapter_info = adapter.get_info();
        let required_surface_format = std::env::var("SIM_ENGINE_GPU_SURFACE_FORMAT").ok();
        assert!(
            required_surface_format.is_some()
                || std::env::var("SIM_ENGINE_REQUIRE_PRODUCTION_SURFACE_FORMAT").as_deref()
                    != Ok("1"),
            "SIM_ENGINE_REQUIRE_PRODUCTION_SURFACE_FORMAT=1 requires SIM_ENGINE_GPU_SURFACE_FORMAT"
        );
        let format = required_surface_format
            .as_deref()
            .map(|name| {
                parse_gpu_oracle_surface_format(name)
                    .unwrap_or_else(|| panic!("unsupported production surface format: {name}"))
            })
            .unwrap_or(wgpu::TextureFormat::Rgba8UnormSrgb);
        let sample_count = preferred_sample_count(&adapter, format);
        if let Ok(expected) = std::env::var("SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT") {
            assert_eq!(
                sample_count,
                expected
                    .parse::<u32>()
                    .expect("SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT must be a u32"),
                "offscreen oracle MSAA differs from the production surface selection"
            );
        }
        if std::env::var("SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY").as_deref() == Ok("1") {
            assert_eq!(
                adapter_info.backend.to_str(),
                std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_BACKEND").unwrap(),
            );
            assert_eq!(
                adapter_info.name,
                std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_NAME").unwrap(),
            );
            assert_eq!(
                format!("{:#06x}", adapter_info.vendor),
                std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_VENDOR").unwrap(),
            );
            assert_eq!(
                format!("{:#06x}", adapter_info.device),
                std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_DEVICE").unwrap(),
            );
            assert_eq!(
                adapter_info.device_pci_bus_id,
                std::env::var("SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID").unwrap(),
                "semantic oracle selected a different physical adapter instance"
            );
        }
        if let Ok(path) = std::env::var("SIM_ENGINE_GPU_EVIDENCE_PATH") {
            let clean = |value: &str| value.replace(['\n', '\r', '='], " ");
            let revision = std::env::var("SIM_ENGINE_RELEASE_SHA")
                .expect("GPU evidence requires SIM_ENGINE_RELEASE_SHA");
            let evidence = format!(
                "format_version=1\nvcs_sha={}\ncrate_version={}\nbackend={:?}\nname={}\ndevice_type={:?}\ndriver={}\ndriver_info={}\nvendor={:#06x}\ndevice={:#06x}\npci_bus_id={}\noracle_format={:?}\noracle_sample_count={}\n",
                clean(&revision),
                env!("CARGO_PKG_VERSION"),
                adapter_info.backend,
                clean(&adapter_info.name),
                adapter_info.device_type,
                clean(&adapter_info.driver),
                clean(&adapter_info.driver_info),
                adapter_info.vendor,
                adapter_info.device,
                clean(&adapter_info.device_pci_bus_id),
                format,
                sample_count,
            );
            std::fs::write(&path, evidence)
                .unwrap_or_else(|error| panic!("write GPU evidence to {path}: {error}"));
        }
        if std::env::var("SIM_ENGINE_REQUIRE_GPU_TESTS").as_deref() == Ok("1") {
            eprintln!(
                "sim-engine GPU evidence: name={:?}, type={:?}, backend={:?}, driver={:?}, driver_info={:?}, vendor={:#06x}, device={:#06x}, pci_bus_id={:?}, surface_format={:?}, sample_count={}",
                adapter_info.name,
                adapter_info.device_type,
                adapter_info.backend,
                adapter_info.driver,
                adapter_info.driver_info,
                adapter_info.vendor,
                adapter_info.device,
                adapter_info.device_pci_bus_id,
                format,
                sample_count,
            );
            if std::env::var("SIM_ENGINE_REQUIRE_VULKAN").as_deref() == Ok("1") {
                assert_eq!(
                    adapter_info.backend,
                    wgpu::Backend::Vulkan,
                    "SIM_ENGINE_REQUIRE_VULKAN=1 requires a Vulkan adapter"
                );
            }
        }
        let Ok((device, queue)) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-engine offscreen test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
        else {
            panic!("adapter should create a test device");
        };
        let Ok((recovery_device, recovery_queue)) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-engine offscreen recovery test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
        else {
            panic!("adapter should create a second recovery device");
        };

        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let recovery_validation_scope =
            recovery_device.push_error_scope(wgpu::ErrorFilter::Validation);
        gpu_timing::assert_gpu_timing_contract(&device, &queue);
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            let (timing_device, timing_queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("sim-engine optional GPU timestamp test device"),
                    required_features: wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: wgpu::Limits::default(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("timestamp-capable adapter should create a timing device");
            let scope = timing_device.push_error_scope(wgpu::ErrorFilter::Validation);
            gpu_timing::assert_gpu_timing_contract(&timing_device, &timing_queue);
            mesh3d::assert_gpu_scene_timing_and_upload_contract(
                &timing_device,
                &timing_queue,
                format,
            );
            let timing_validation_error = scope.pop().await;
            assert!(
                timing_validation_error.is_none(),
                "GPU timestamp validation failed: {timing_validation_error:?}"
            );
        } else {
            eprintln!(
                "sim-engine GPU timestamps: adapter feature unavailable, no substitute measurement"
            );
        }
        assert_gpu_large_center_circle(&adapter, &device, &queue, format).await;
        assert_gpu_stroke_pixel_matrix(&adapter, &device, &queue, format).await;
        exact_markers::assert_gpu_pixels(&device, &queue, format, sample_count);
        image::verify_retained_ui_updates(
            &device,
            &queue,
            &recovery_device,
            &recovery_queue,
            format,
            sample_count,
        )
        .await;
        frame::assert_gpu_binding_sharing_contract(&device, &queue);
        frame::assert_gpu_uniform_upload_contract(&device, &queue);
        frame::assert_gpu_encoding_contract(&device, &queue, format, sample_count);
        mesh3d::assert_gpu_depth_contract(&device, &queue, format);
        mesh3d::assert_gpu_scene_recovery_contract(
            &device,
            &queue,
            &recovery_device,
            &recovery_queue,
            format,
        );
        let vertex_limit = device.limits().max_buffer_size / std::mem::size_of::<Vertex>() as u64;
        if let Ok(first_invalid_capacity) = usize::try_from(vertex_limit.saturating_add(1)) {
            assert!(!buffer_capacity_fits::<Vertex>(
                &device,
                first_invalid_capacity
            ));
        }
        let PipelineResources {
            pipeline,
            target_pipeline: _,
            dynamic_pipeline: _,
            particle_pipeline: _,
            target_particle_pipeline,
            heatmap_pipeline,
            target_heatmap_pipeline,
            composition_pipelines: _,
            target_composition_pipelines,
            camera_uniform_buffer,
            camera_bind_group,
            camera_bind_group_layout: _,
            heatmap_uniform_buffer,
            heatmap_bind_group_layout,
        } = create_pipeline(&device, format, 1);
        let image_renderer = ImageRenderer::new(&device, format, 1);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen test target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut scene = Scene::new(Color::BLACK).unwrap();
        let clip = ScreenClipRect::from_min_size(
            LogicalScreenPosition::new(30.0, 20.0),
            LogicalScreenVector::new(8.0, 12.0),
        )
        .unwrap();
        assert!(
            scene
                .with_depth(3.0, |scene| {
                    scene
                        .with_screen_clip(clip, |scene| {
                            scene.rect(
                                Rect::from_center_size(Vec2::ZERO, Vec2::splat(4.0)),
                                0.0,
                                ShapeStyle::filled(Color::WHITE),
                            );
                        })
                        .unwrap();
                })
                .is_ok()
        );
        let stroke_dash = crate::StrokeDashPattern2d::new(&[4.0, 4.0], 0.0, 8).unwrap();
        scene
            .try_styled_line(
                Vec2::new(-20.0, 20.0),
                Vec2::new(4.0, 20.0),
                crate::StrokeStyle2d::logical(
                    crate::LogicalPixels::new(4.0).unwrap(),
                    Color::WHITE,
                )
                .with_cap(crate::StrokeCap2d::Butt)
                .with_dash_pattern(stroke_dash),
            )
            .unwrap();
        let translucent_red = Color::rgba(1.0, 0.0, 0.0, 0.5);
        scene
            .try_styled_polyline(
                vec![
                    Vec2::new(-26.0, -8.0),
                    Vec2::new(-18.0, -2.0),
                    Vec2::new(-10.0, -8.0),
                ],
                crate::StrokeStyle2d::logical(
                    crate::LogicalPixels::new(6.0).unwrap(),
                    translucent_red,
                )
                .with_cap(crate::StrokeCap2d::Butt)
                .with_join(crate::StrokeJoin2d::Round),
            )
            .unwrap();
        let translucent_green = Color::rgba(0.0, 1.0, 0.0, 0.5);
        scene
            .try_styled_polyline(
                vec![
                    Vec2::new(2.0, -8.0),
                    Vec2::new(10.0, -2.0),
                    Vec2::new(18.0, -8.0),
                ],
                crate::StrokeStyle2d::logical(
                    crate::LogicalPixels::new(6.0).unwrap(),
                    translucent_green,
                )
                .with_cap(crate::StrokeCap2d::Butt)
                .with_join(crate::StrokeJoin2d::Bevel),
            )
            .unwrap();
        let translucent_yellow = Color::rgba(1.0, 1.0, 0.0, 0.5);
        scene
            .try_styled_polyline(
                vec![
                    Vec2::new(10.0, 4.5),
                    Vec2::new(18.0, 11.4),
                    Vec2::new(26.0, 4.5),
                ],
                crate::StrokeStyle2d::logical(
                    crate::LogicalPixels::new(6.0).unwrap(),
                    translucent_yellow,
                )
                .with_cap(crate::StrokeCap2d::Butt)
                .with_join(crate::StrokeJoin2d::Miter)
                .with_miter_limit(1.0)
                .unwrap(),
            )
            .unwrap();
        let translucent_blue = Color::rgba(0.0, 0.0, 1.0, 0.5);
        let arrow = crate::StrokeMarker2d::arrow(
            crate::LogicalPixels::new(8.0).unwrap(),
            crate::LogicalPixels::new(12.0).unwrap(),
        );
        scene
            .try_styled_line(
                Vec2::new(-24.0, -24.0),
                Vec2::new(6.0, -24.0),
                crate::StrokeStyle2d::logical(
                    crate::LogicalPixels::new(6.0).unwrap(),
                    translucent_blue,
                )
                .with_cap(crate::StrokeCap2d::Round)
                .with_end_marker(arrow),
            )
            .unwrap();
        let source_identity = Arc::new(());
        let prepared =
            prepare_scene_resources(&device, &queue, Arc::clone(&source_identity), &scene)
                .expect("small prepared scene should fit the test device");
        let replacement_identity = Arc::new(());
        let restored = restore_prepared_scene_resources(
            &device,
            &queue,
            Arc::clone(&replacement_identity),
            &prepared,
        )
        .expect("small prepared scene should restore on the test device");
        assert!(prepared_scene_belongs_to(
            &source_identity,
            &prepared.renderer_identity
        ));
        assert!(!prepared_scene_belongs_to(
            &source_identity,
            &restored.renderer_identity
        ));
        assert!(prepared_scene_belongs_to(
            &replacement_identity,
            &restored.renderer_identity
        ));
        assert_eq!(restored.vertex_count(), prepared.vertex_count());
        assert_eq!(
            restored.recovery_memory_bytes(),
            restored.vertices.capacity() * std::mem::size_of::<Vertex>()
                + restored.draw_batches.capacity() * std::mem::size_of::<PreparedDrawBatch>()
        );
        assert!(Arc::ptr_eq(&restored.vertices, &prepared.vertices));
        let recovery_identity = Arc::new(());
        let recovered_on_another_device = restore_prepared_scene_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            &prepared,
        )
        .expect("prepared scene should migrate to a second device");
        assert!(prepared_scene_belongs_to(
            &recovery_identity,
            &recovered_on_another_device.renderer_identity
        ));
        assert_eq!(
            recovered_on_another_device.vertex_count(),
            prepared.vertex_count()
        );
        assert!(Arc::ptr_eq(
            &recovered_on_another_device.vertices,
            &prepared.vertices
        ));

        let dynamic_vertex = DynamicVertex2d::new(Vec2::ZERO, 0.0, Color::WHITE).unwrap();
        let dynamic_vertices = dynamic_vertices_to_gpu(&[dynamic_vertex; 3]).unwrap();
        let triangle_bytes = 3 * std::mem::size_of::<DynamicGpu>();
        let mut dynamic_source = DynamicMesh2d {
            renderer_identity: Arc::clone(&source_identity),
            vertex_buffer: Arc::new(create_dynamic_vertex_buffer(&device, 8)),
            geometry_extents: GeometryExtents::from_dynamic_vertices(&dynamic_vertices),
            geometry_validation_cache: GeometryValidationCache::default(),
            vertices: dynamic_vertices,
            vertex_capacity: 8,
            budget: Some(DynamicMeshBudget::new(3, triangle_bytes, triangle_bytes).unwrap()),
        };
        queue.write_buffer(
            &dynamic_source.vertex_buffer,
            0,
            bytemuck::cast_slice(dynamic_source.vertices.as_slice()),
        );
        let original_dynamic_buffer = Arc::clone(&dynamic_source.vertex_buffer);
        let original_dynamic_vertices = dynamic_source.vertices.clone();
        assert_eq!(
            replace_dynamic_mesh_resources(
                &device,
                &queue,
                &mut dynamic_source,
                &[dynamic_vertex; 6],
            ),
            Err(DynamicMeshError::BudgetExceeded {
                resource: DynamicMeshBudgetResource::Vertices,
                limit: 3,
                actual: 6,
            })
        );
        assert!(Arc::ptr_eq(
            &dynamic_source.vertex_buffer,
            &original_dynamic_buffer
        ));
        assert_eq!(dynamic_source.vertices, original_dynamic_vertices);
        let restored_dynamic = restore_dynamic_mesh_resources(
            &device,
            &queue,
            Arc::clone(&replacement_identity),
            &dynamic_source,
        )
        .expect("small dynamic mesh should restore on the test device");
        assert_eq!(restored_dynamic.vertex_count(), 3);
        assert_eq!(restored_dynamic.vertex_capacity(), 8);
        assert_eq!(
            restored_dynamic.recovery_memory_bytes(),
            dynamic_source.recovery_memory_bytes()
        );
        assert!(prepared_scene_belongs_to(
            &replacement_identity,
            &restored_dynamic.renderer_identity
        ));
        let recovered_dynamic_on_another_device = restore_dynamic_mesh_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            &dynamic_source,
        )
        .expect("dynamic mesh should migrate to a second device");
        assert_eq!(recovered_dynamic_on_another_device.vertex_count(), 3);
        assert_eq!(recovered_dynamic_on_another_device.vertex_capacity(), 8);
        assert!(prepared_scene_belongs_to(
            &recovery_identity,
            &recovered_dynamic_on_another_device.renderer_identity
        ));

        let scalar_source = ScalarField::new(2, 2, vec![0.0, 0.25, 0.5, 1.0]).unwrap();
        let maximum_texture_dimension = device.limits().max_texture_dimension_2d;
        if maximum_texture_dimension < 1_000_000 {
            let oversized = ScalarField::filled(maximum_texture_dimension as usize + 1, 1, 0.0)
                .expect("GPU-limit probe should be a valid CPU scalar field");
            assert!(matches!(
                create_scalar_field_texture(&device, &oversized),
                Err(ScalarFieldTextureError::DimensionsTooLarge)
            ));
        }
        let scalar_texture = create_scalar_field_texture_resources(
            &device,
            &queue,
            Arc::clone(&source_identity),
            scalar_source,
        )
        .expect("finite scalar texture should upload");
        assert_eq!((scalar_texture.width(), scalar_texture.height()), (2, 2));
        assert_eq!(scalar_texture.recovery_memory_bytes(), 16);
        let restored_scalar_texture = create_scalar_field_texture_resources(
            &device,
            &queue,
            Arc::clone(&replacement_identity),
            scalar_texture.field.clone(),
        )
        .expect("scalar texture should restore");
        assert_eq!(
            restored_scalar_texture.field().values(),
            scalar_texture.field().values()
        );
        let recovered_scalar_texture_on_another_device = create_scalar_field_texture_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            scalar_texture.field.clone(),
        )
        .expect("scalar texture should migrate to a second device");
        assert_eq!(
            recovered_scalar_texture_on_another_device.field().values(),
            scalar_texture.field().values()
        );
        let heatmap_color_map = ColorMap::linear(Color::BLACK, Color::WHITE).unwrap();
        let heatmap_lut =
            create_cached_color_map(&device, &queue, color_map_lut(&heatmap_color_map));
        let heatmap_scalar_view = scalar_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let heatmap_lut_view = &heatmap_lut.view;
        let heatmap_uniform = HeatmapUniform::new(0.0, 1.0, 2, 2, ScalarFieldSampling::Nearest);
        queue.write_buffer(
            &heatmap_uniform_buffer,
            0,
            bytemuck::bytes_of(&heatmap_uniform),
        );
        let heatmap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen heatmap bind group"),
            layout: &heatmap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&heatmap_scalar_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(heatmap_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: heatmap_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let heatmap_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen heatmap target"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let heatmap_view = heatmap_target.create_view(&wgpu::TextureViewDescriptor::default());
        let heatmap_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen heatmap readback"),
            size: 512,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let Ok(mut camera) = Camera2d::new(Vec2::ZERO, 1.0) else {
            panic!("test camera should be valid");
        };
        let Ok(projection) = crate::Projection2d::new(0.5, 2.0) else {
            panic!("test projection should be valid");
        };
        camera.set_projection(projection);
        let Ok(viewport) = LogicalViewport::new(64.0, 64.0) else {
            panic!("test viewport should be valid");
        };
        let Some(camera_uniform) = CameraUniform::new(camera, viewport) else {
            panic!("test camera uniform should be finite");
        };
        queue.write_buffer(
            &camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        let particle_unit_buffer = create_submitted_particle_unit_buffer(&device, &queue);
        let particle_instances = [
            ParticleGpu {
                world_position: [18.0, -18.0],
                depth: 0.0,
                radius: 6.0,
                color: Color::rgba(1.0, 0.0, 0.0, 1.0).to_array(),
            },
            ParticleGpu {
                world_position: [1_000.0, 0.0],
                depth: 0.0,
                radius: 6.0,
                color: Color::WHITE.to_array(),
            },
        ];
        let particle_source = ParticleField2d {
            renderer_identity: Arc::clone(&source_identity),
            instance_buffer: Arc::new(create_particle_instance_buffer(&device, 2)),
            instances: particle_instances.to_vec(),
            visible_instances: Vec::new(),
            instance_capacity: 2,
            budget: ParticleRenderBudget::UNBOUNDED,
            statistics: particle_statistics(2, 1, 1),
        };
        queue.write_buffer(
            &particle_source.instance_buffer,
            0,
            bytemuck::cast_slice(&particle_source.instances),
        );
        let recovered_particles_on_another_device = restore_particle_field_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            &particle_source,
        )
        .expect("particle field should migrate to a second device");
        assert_eq!(recovered_particles_on_another_device.instance_count(), 2);
        assert_eq!(recovered_particles_on_another_device.instance_capacity(), 2);
        assert!(prepared_scene_belongs_to(
            &recovery_identity,
            &recovered_particles_on_another_device.renderer_identity
        ));
        assert_eq!(
            recovered_particles_on_another_device.statistics(),
            particle_idle_statistics(2),
            "recovery must not preserve stale culling or draw statistics"
        );

        let image_pixels = vec![255, 0, 0, 255, 0, 255, 0, 128];
        let image_source = image::create_image_resources(
            &device,
            &queue,
            Arc::clone(&source_identity),
            2,
            1,
            image_pixels,
            ImageBudget::new(2, 1, 8).unwrap(),
        )
        .expect("bounded image should upload");
        let sprite_region = LogicalViewportRegion::new(
            LogicalScreenPosition::new(3.0, 4.0),
            LogicalViewport::new(8.0, 6.0).unwrap(),
        )
        .unwrap();
        let image_sprites = vec![
            ImageSprite2d::new(
                ImageTexelRect::new(0, 0, 1, 1).unwrap(),
                sprite_region,
                Color::WHITE,
            )
            .unwrap(),
        ];
        let image_batch = image::create_image_batch_resources(
            &device,
            &queue,
            Arc::clone(&source_identity),
            &image_source,
            image_sprites.clone(),
            ImageBatchBudget::new(2, 1024).unwrap(),
        )
        .expect("bounded image batch should upload");
        let recovered_image = image::create_image_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            image_source.width(),
            image_source.height(),
            image_source.pixels().to_vec(),
            image_source.budget(),
        )
        .expect("image should migrate to a second device");
        let recovered_image_batch = image::create_image_batch_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            &recovered_image,
            image_sprites,
            image_batch.budget(),
        )
        .expect("image batch should migrate to a second device");
        assert_eq!(recovered_image.pixels(), image_source.pixels());
        assert_eq!(recovered_image_batch.sprites(), image_batch.sprites());

        let atlas_entries = vec![
            GlyphAtlasEntry::new(
                GlyphId::new('μ' as u32),
                ImageTexelRect::new(0, 0, 1, 1).unwrap(),
            ),
            GlyphAtlasEntry::new(
                GlyphId::new('Δ' as u32),
                ImageTexelRect::new(1, 0, 1, 1).unwrap(),
            ),
        ];
        let glyph_atlas = glyph::create_glyph_atlas_resources(
            &device,
            &queue,
            Arc::clone(&source_identity),
            2,
            1,
            image_source.pixels().to_vec(),
            atlas_entries.clone(),
            GlyphAtlasBudget::new(ImageBudget::new(2, 1, 8).unwrap(), 4, 1024).unwrap(),
        )
        .expect("glyph atlas should upload");
        let glyphs = vec![
            PositionedGlyph2d::new(GlyphId::new('μ' as u32), sprite_region, Color::WHITE).unwrap(),
            PositionedGlyph2d::new(
                GlyphId::new('Δ' as u32),
                LogicalViewportRegion::new(
                    LogicalScreenPosition::new(11.0, 4.0),
                    LogicalViewport::new(8.0, 6.0).unwrap(),
                )
                .unwrap(),
                Color::WHITE,
            )
            .unwrap(),
        ];
        let glyph_run = glyph::create_glyph_run_resources(
            &device,
            &queue,
            Arc::clone(&source_identity),
            &glyph_atlas,
            glyphs.clone(),
            GlyphRunBudget::new(4, 4096).unwrap(),
        )
        .expect("glyph run should upload");
        let recovered_atlas = glyph::create_glyph_atlas_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            2,
            1,
            glyph_atlas.image.pixels().to_vec(),
            atlas_entries,
            glyph_atlas.budget(),
        )
        .expect("glyph atlas should migrate to a second device");
        let recovered_glyph_run = glyph::create_glyph_run_resources(
            &recovery_device,
            &recovery_queue,
            Arc::clone(&recovery_identity),
            &recovered_atlas,
            glyphs,
            glyph_run.budget(),
        )
        .expect("glyph run should migrate to a second device");
        assert_eq!(recovered_atlas.entries(), glyph_atlas.entries());
        assert_eq!(recovered_glyph_run.glyphs(), glyph_run.glyphs());
        assert_eq!(recovered_glyph_run.statistics().rendered_quads(), 2);

        let recovery_vertex_bytes = bytemuck::cast_slice(prepared.vertices.as_ref());
        let recovery_readback = recovery_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine recovery vertex readback"),
            size: recovery_vertex_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut recovery_encoder =
            recovery_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine recovery readback encoder"),
            });
        recovery_encoder.copy_buffer_to_buffer(
            &recovered_on_another_device.vertex_buffer,
            0,
            &recovery_readback,
            0,
            recovery_vertex_bytes.len() as wgpu::BufferAddress,
        );
        let _recovery_submission = recovery_queue.submit([recovery_encoder.finish()]);
        recovery_device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("recovery upload should complete on the second device");
        let recovery_slice = recovery_readback.slice(..);
        let (recovery_sender, recovery_receiver) = mpsc::channel();
        recovery_slice.map_async(wgpu::MapMode::Read, move |result| {
            recovery_sender.send(result).unwrap()
        });
        recovery_device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("recovery readback should complete on the second device");
        recovery_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("recovery readback callback")
            .expect("recovery readback should map");
        let recovered_bytes = recovery_slice
            .get_mapped_range()
            .expect("recovery vertex bytes");
        assert_eq!(recovered_bytes.as_ref(), recovery_vertex_bytes);
        drop(recovered_bytes);
        recovery_readback.unmap();
        if let Some(error) = recovery_validation_scope.pop().await {
            panic!("second-device recovery validation failed: {error}");
        }
        let visible_particles =
            visible_particle_instances(&particle_instances, camera_uniform, viewport).unwrap();
        assert_eq!(
            visible_particles.len(),
            1,
            "offscreen particle should be culled"
        );
        let particle_instance_buffer = create_particle_instance_buffer(&device, 1);
        queue.write_buffer(
            &particle_instance_buffer,
            0,
            bytemuck::cast_slice(&visible_particles),
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen readback"),
            size: (256 * 64) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine offscreen test encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen heatmap pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &heatmap_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_heatmap_pipeline);
            pass.set_bind_group(0, &heatmap_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &camera_bind_group, &[]);
            pass.set_vertex_buffer(0, restored.vertex_buffer.slice(..));
            for batch in &restored.draw_batches {
                let scissor = batch.screen_clip.map_or(
                    ScissorRect {
                        x: 0,
                        y: 0,
                        width: 64,
                        height: 64,
                    },
                    |clip| screen_clip_to_scissor(clip, viewport, 1.0).unwrap(),
                );
                pass.set_scissor_rect(scissor.x, scissor.y, scissor.width, scissor.height);
                pass.draw(batch.vertex_range.clone(), 0..1);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen particle pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_particle_pipeline);
            pass.set_bind_group(0, &camera_bind_group, &[]);
            pass.set_vertex_buffer(0, particle_unit_buffer.slice(..));
            pass.set_vertex_buffer(1, particle_instance_buffer.slice(..));
            pass.draw(0..6, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
            },
            wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &heatmap_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &heatmap_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(2),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );

        let _submission = queue.submit([encoder.finish()]);
        if let Err(error) = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        }) {
            panic!("offscreen GPU submission did not complete: {error:?}");
        }
        if let Some(error) = validation_scope.pop().await {
            panic!("offscreen GPU validation failed: {error}");
        }
        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
        if let Err(error) = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        }) {
            panic!("offscreen GPU readback did not complete: {error:?}");
        }
        receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen readback callback")
            .expect("offscreen readback should map");
        let bytes = slice.get_mapped_range().expect("offscreen mapped bytes");
        let pixel = |x: usize, y: usize| &bytes[y * 256 + x * 4..y * 256 + x * 4 + 4];
        let [red, green, blue, _alpha] = gpu_oracle_channel_indices(format);
        assert!(pixel(33, 29)[0] > 200, "camera/depth pixel was not drawn");
        assert!(pixel(33, 34)[0] < 10, "clip failed to remove outside pixel");
        assert!(pixel(14, 14)[0] > 200, "styled dash body was not drawn");
        assert!(pixel(18, 14)[0] < 10, "styled dash gap was not preserved");
        assert!(pixel(22, 14)[0] > 200, "styled dash phase did not repeat");
        let assert_uniform_translucency = |body: &[u8], detail: &[u8], channel: usize, name| {
            assert!(
                (160..=210).contains(&body[channel]),
                "{name} body did not preserve half-alpha linear color: {body:?}"
            );
            assert!(
                body[channel].abs_diff(detail[channel]) <= 8,
                "{name} detail was alpha-blended more than once: body={body:?}, detail={detail:?}"
            );
        };
        assert_uniform_translucency(pixel(10, 37), pixel(14, 32), red, "round join");
        assert_uniform_translucency(pixel(38, 37), pixel(42, 32), green, "bevel join");
        assert_uniform_translucency(pixel(46, 25), pixel(50, 20), red, "miter fallback");
        assert!(
            pixel(50, 17)[red] < 10 && pixel(50, 17)[green] < 10,
            "over-limit miter spike was not replaced by bevel geometry: {:?}",
            pixel(50, 17)
        );
        assert_uniform_translucency(pixel(20, 53), pixel(42, 53), blue, "arrow marker");
        assert_uniform_translucency(pixel(20, 53), pixel(6, 53), blue, "round cap");
        assert!(
            pixel(48, 53)[blue] < 10,
            "the endpoint cap protruded beyond the arrow tip: {:?}",
            pixel(48, 53)
        );
        assert!(
            pixel(50, 48)[red] > 200,
            "instanced particle center was not drawn"
        );
        assert!(
            pixel(50, 48)[green] < 10,
            "instanced particle color was not applied"
        );
        assert!(
            pixel(56, 54)[red] < 10,
            "particle circle mask did not discard its corner"
        );
        drop(bytes);
        readback.unmap();
        let heatmap_slice = heatmap_readback.slice(..);
        let (heatmap_sender, heatmap_receiver) = mpsc::channel();
        heatmap_slice.map_async(wgpu::MapMode::Read, move |result| {
            heatmap_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen heatmap readback should complete");
        heatmap_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen heatmap callback")
            .expect("offscreen heatmap should map");
        let heatmap_bytes = heatmap_slice
            .get_mapped_range()
            .expect("offscreen heatmap bytes");
        let heatmap_pixel =
            |x: usize, y: usize| &heatmap_bytes[y * 256 + x * 4..y * 256 + x * 4 + 4];
        assert!(
            heatmap_pixel(0, 0)[0] < 8,
            "minimum scalar should map to black"
        );
        assert!(
            heatmap_pixel(1, 1)[0] > 247,
            "maximum scalar should map to white"
        );
        assert!(
            heatmap_pixel(1, 0)[0] > 130 && heatmap_pixel(1, 0)[0] < 145,
            "quarter scalar should map through LUT then sRGB encode"
        );
        assert!(
            heatmap_pixel(0, 1)[0] > 180 && heatmap_pixel(0, 1)[0] < 195,
            "half scalar should map through LUT then sRGB encode"
        );
        drop(heatmap_bytes);
        heatmap_readback.unmap();

        let composition_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen composition target"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let composition_target_view =
            composition_target.create_view(&wgpu::TextureViewDescriptor::default());
        let premultiplied_source = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen premultiplied composition source"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &premultiplied_source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[188, 0, 0, 128],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let premultiplied_source_view =
            premultiplied_source.create_view(&wgpu::TextureViewDescriptor::default());
        let composition_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen composition uniform"),
            size: std::mem::size_of::<CompositeUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &composition_uniform,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(1.0)),
        );
        let composition_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen composition bind group"),
            layout: &target_composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&premultiplied_source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: composition_uniform.as_entire_binding(),
                },
            ],
        });
        let composition_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen composition readback"),
            size: 512,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut composition_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine offscreen composition encoder"),
            });
        {
            let mut pass = composition_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen composition pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composition_target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_composition_pipelines.alpha);
            pass.set_bind_group(0, &composition_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        composition_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &composition_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &composition_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(2),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([composition_encoder.finish()]);
        let composition_slice = composition_readback.slice(..);
        let (composition_sender, composition_receiver) = mpsc::channel();
        composition_slice.map_async(wgpu::MapMode::Read, move |result| {
            composition_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen composition readback should complete");
        composition_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen composition callback")
            .expect("offscreen composition should map");
        let composition_bytes = composition_slice
            .get_mapped_range()
            .expect("offscreen composition bytes");
        assert!(
            composition_bytes[0] > 180 && composition_bytes[0] < 195,
            "premultiplied render-target RGB must be unpremultiplied before alpha composition"
        );
        drop(composition_bytes);
        composition_readback.unmap();

        let composition_region = LogicalViewportRegion::new(
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalViewport::new(1.0, 1.0).unwrap(),
        )
        .unwrap();
        let region_uniform = CompositeUniform::in_region(
            1.0,
            composition_region,
            LogicalViewport::new(2.0, 2.0).unwrap(),
        )
        .unwrap();
        queue.write_buffer(&composition_uniform, 0, bytemuck::bytes_of(&region_uniform));
        let mut region_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine offscreen region composition encoder"),
        });
        {
            let mut pass = region_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen region composition pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composition_target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_composition_pipelines.alpha);
            pass.set_bind_group(0, &composition_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        region_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &composition_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &composition_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(2),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([region_encoder.finish()]);
        let region_slice = composition_readback.slice(..);
        let (region_sender, region_receiver) = mpsc::channel();
        region_slice.map_async(wgpu::MapMode::Read, move |result| {
            region_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen region composition should complete");
        region_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen region composition callback")
            .expect("offscreen region composition should map");
        let region_bytes = region_slice
            .get_mapped_range()
            .expect("offscreen region composition bytes");
        assert!(region_bytes[0] > 180, "region top-left pixel was not drawn");
        assert!(
            region_bytes[4] < 8 && region_bytes[256] < 8,
            "region composition escaped its logical destination"
        );
        drop(region_bytes);
        composition_readback.unmap();

        queue.write_buffer(
            &target_composition_pipelines.secondary_uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(0.5)),
        );
        queue.write_buffer(
            &target_composition_pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&CompositeUniform::full_surface(0.5)),
        );
        let trail_history_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen trail history bind group"),
            layout: &target_composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&heatmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target_composition_pipelines
                        .secondary_uniform_buffer
                        .as_entire_binding(),
                },
            ],
        });
        let trail_source_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen trail source bind group"),
            layout: &target_composition_pipelines.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&heatmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target_composition_pipelines
                        .uniform_buffer
                        .as_entire_binding(),
                },
            ],
        });
        let mut trail_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine offscreen trail encoder"),
        });
        {
            let mut pass = trail_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen trail history pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composition_target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 0.0).to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_composition_pipelines.alpha);
            pass.set_bind_group(0, &trail_history_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        {
            let mut pass = trail_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen trail source pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composition_target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&target_composition_pipelines.alpha);
            pass.set_bind_group(0, &trail_source_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        trail_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &composition_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &composition_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(2),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([trail_encoder.finish()]);
        let trail_slice = composition_readback.slice(..);
        let (trail_sender, trail_receiver) = mpsc::channel();
        trail_slice.map_async(wgpu::MapMode::Read, move |result| {
            trail_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen trail readback should complete");
        trail_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen trail callback")
            .expect("offscreen trail should map");
        let trail_bytes = trail_slice
            .get_mapped_range()
            .expect("offscreen trail bytes");
        assert!(
            trail_bytes[4] > 112 && trail_bytes[4] < 128,
            "half-retained history plus half-opacity source should accumulate predictably"
        );
        drop(trail_bytes);
        composition_readback.unmap();

        let linear_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen linear heatmap uniform"),
            size: std::mem::size_of::<HeatmapUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &linear_uniform_buffer,
            0,
            bytemuck::bytes_of(&HeatmapUniform::new(
                0.0,
                1.0,
                2,
                2,
                ScalarFieldSampling::Linear,
            )),
        );
        let linear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen linear heatmap bind group"),
            layout: &heatmap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&heatmap_scalar_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(heatmap_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: linear_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let linear_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen linear heatmap target"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let linear_view = linear_target.create_view(&wgpu::TextureViewDescriptor::default());
        let linear_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen linear heatmap readback"),
            size: 2048,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut linear_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine offscreen linear heatmap encoder"),
        });
        {
            let mut pass = linear_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen linear heatmap pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &linear_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&heatmap_pipeline);
            pass.set_bind_group(0, &linear_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        linear_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &linear_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &linear_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(8),
                },
            },
            wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([linear_encoder.finish()]);
        let linear_slice = linear_readback.slice(..);
        let (linear_sender, linear_receiver) = mpsc::channel();
        linear_slice.map_async(wgpu::MapMode::Read, move |result| {
            linear_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen linear heatmap readback should complete");
        linear_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen linear heatmap callback")
            .expect("offscreen linear heatmap should map");
        let linear_bytes = linear_slice
            .get_mapped_range()
            .expect("offscreen linear heatmap bytes");
        let linear_pixel = |x: usize, y: usize| &linear_bytes[y * 256 + x * 4..y * 256 + x * 4 + 4];
        assert!(
            linear_pixel(0, 0)[0] < 8,
            "linear clamp-to-edge mixed the top-left corner with a neighbour"
        );
        assert!(
            linear_pixel(7, 0)[0] > 130 && linear_pixel(7, 0)[0] < 145,
            "linear clamp-to-edge mixed the top-right corner vertically"
        );
        assert!(
            linear_pixel(0, 7)[0] > 180 && linear_pixel(0, 7)[0] < 195,
            "linear clamp-to-edge mixed the bottom-left corner horizontally"
        );
        assert!(
            linear_pixel(7, 7)[0] > 247,
            "linear clamp-to-edge changed the bottom-right corner"
        );
        drop(linear_bytes);
        linear_readback.unmap();

        let image_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let image_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen atlas image"),
            size: wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &image_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 0, 0, 255, 0, 255, 0, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let image_view = image_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let image_uniform = ImageUniform {
            destination: [0.5, 1.0, -0.5, 0.0],
            uv_rect: [0.25, 0.5, 0.25, 0.5],
            tint: [1.0, 1.0, 1.0, 0.5],
            world_clip_x: [0.0; 4],
            world_clip_y: [0.0; 4],
            world_mode: [0.0; 4],
        };
        let image_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen image uniform"),
            size: std::mem::size_of::<ImageUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&image_uniform_buffer, 0, bytemuck::bytes_of(&image_uniform));
        let image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen image bind group"),
            layout: &image_renderer.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        image_renderer.sampler(ImageSampling::Linear),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: image_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let world_image_uniform = ImageUniform {
            destination: [0.0; 4],
            uv_rect: [0.75, 0.5, 0.75, 0.5],
            tint: [1.0, 1.0, 1.0, 0.5],
            world_clip_x: [0.0, 1.0, 0.0, 1.0],
            world_clip_y: [1.0, 1.0, -1.0, -1.0],
            world_mode: [1.0, 0.0, 0.0, 0.0],
        };
        let world_image_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen world image uniform"),
            size: std::mem::size_of::<ImageUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &world_image_uniform_buffer,
            0,
            bytemuck::bytes_of(&world_image_uniform),
        );
        let world_image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen world image bind group"),
            layout: &image_renderer.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        image_renderer.sampler(ImageSampling::Linear),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: world_image_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let image_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sim-engine offscreen image target"),
            size: wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let image_target_view = image_target.create_view(&wgpu::TextureViewDescriptor::default());
        let image_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen image readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut image_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine offscreen image encoder"),
        });
        {
            let mut pass = image_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen image pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &image_target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&image_renderer.pipeline);
            pass.set_bind_group(0, &image_bind_group, &[]);
            pass.draw(0..6, 0..1);
            pass.set_bind_group(0, &world_image_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        image_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &image_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &image_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([image_encoder.finish()]);
        let image_slice = image_readback.slice(..);
        let (image_sender, image_receiver) = mpsc::channel();
        image_slice.map_async(wgpu::MapMode::Read, move |result| {
            image_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen image readback should complete");
        image_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen image callback")
            .expect("offscreen image should map");
        if let Some(error) = image_scope.pop().await {
            panic!("offscreen image validation failed: {error}");
        }
        let image_bytes = image_slice
            .get_mapped_range()
            .expect("offscreen image bytes");
        assert!(image_bytes[red] > 180 && image_bytes[red] < 195);
        assert!(
            image_bytes[green] < 8,
            "screen image atlas sampling bled into the next texel"
        );
        assert!(
            image_bytes[4 + red] < 8,
            "world image atlas sampling bled into the previous texel"
        );
        assert!(image_bytes[4 + green] > 180 && image_bytes[4 + green] < 195);
        drop(image_bytes);
        image_readback.unmap();

        let image_batch_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let image_batch_uniform = ImageUniform {
            destination: [1.0, -2.0, -1.0, 1.0],
            uv_rect: [0.0, 0.0, 0.0, 0.0],
            tint: Color::WHITE.to_array(),
            world_clip_x: [0.0; 4],
            world_clip_y: [0.0; 4],
            world_mode: [0.0; 4],
        };
        let image_batch_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen image batch uniform"),
            size: std::mem::size_of::<ImageUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &image_batch_uniform_buffer,
            0,
            bytemuck::bytes_of(&image_batch_uniform),
        );
        let image_batch_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine offscreen image batch bind group"),
            layout: &image_renderer.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        image_renderer.sampler(ImageSampling::Linear),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: image_batch_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let image_instances = [
            image::ImageInstance {
                destination: [0.0, 0.0, 1.0, 1.0],
                uv_rect: [0.25, 0.5, 0.25, 0.5],
                tint: Color::WHITE.to_array(),
            },
            image::ImageInstance {
                destination: [1.0, 0.0, 1.0, 1.0],
                uv_rect: [0.75, 0.5, 0.75, 0.5],
                tint: Color::WHITE.with_alpha(0.5).to_array(),
            },
        ];
        let image_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine offscreen image batch instances"),
            size: std::mem::size_of_val(&image_instances) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &image_instance_buffer,
            0,
            bytemuck::cast_slice(&image_instances),
        );
        let mut image_batch_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine offscreen image batch encoder"),
            });
        {
            let mut pass = image_batch_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine offscreen image batch pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &image_target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK.to_wgpu()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&image_renderer.batch_pipeline);
            pass.set_bind_group(0, &image_batch_bind_group, &[]);
            pass.set_vertex_buffer(0, image_instance_buffer.slice(..));
            pass.draw(0..6, 0..2);
        }
        image_batch_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &image_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &image_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([image_batch_encoder.finish()]);
        let image_batch_slice = image_readback.slice(..);
        let (image_batch_sender, image_batch_receiver) = mpsc::channel();
        image_batch_slice.map_async(wgpu::MapMode::Read, move |result| {
            image_batch_sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .expect("offscreen image batch readback should complete");
        image_batch_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("offscreen image batch callback")
            .expect("offscreen image batch should map");
        if let Some(error) = image_batch_scope.pop().await {
            panic!("offscreen image batch validation failed: {error}");
        }
        let image_batch_bytes = image_batch_slice
            .get_mapped_range()
            .expect("offscreen image batch bytes");
        assert!(image_batch_bytes[red] > 247 && image_batch_bytes[green] < 8);
        assert!(image_batch_bytes[4 + red] < 8);
        assert!(image_batch_bytes[4 + green] > 180 && image_batch_bytes[4 + green] < 195);
        drop(image_batch_bytes);
        image_readback.unmap();
    });
}
