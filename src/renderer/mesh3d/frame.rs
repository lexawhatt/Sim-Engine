//! Retained 3D frame validation, staging, upload and submission.

use super::*;

// Keep visible-index dynamic offsets unchanged when edges exist. Surface-only
// frames need neither padded edge staging nor a proportional edge GPU buffer.
pub(super) fn edge_uniform_object_count(scene: &Scene3d) -> usize {
    if scene.instances().iter().any(|instance| {
        instance.visible && instance.wireframe().is_some() && instance.mesh.edge_count > 0
    }) {
        scene.visible_object_count()
    } else {
        0
    }
}

impl Mesh3dRenderer {
    pub(super) fn retained_frame_cpu_bytes(&self) -> usize {
        self.instances
            .capacity()
            .saturating_mul(std::mem::size_of::<MeshInstanceGpu>())
            .saturating_add(self.edge_object_bytes.capacity())
            .saturating_add(
                self.clipped_surface_objects
                    .capacity()
                    .saturating_mul(std::mem::size_of::<SurfaceObject>()),
            )
    }

    pub(super) fn retained_frame_gpu_bytes(&self) -> usize {
        [
            Some(&self.camera_uniform_buffer),
            Some(&self.instance_buffer),
            Some(&self.edge_object_buffer),
            self.clipped_surface_buffer.as_ref(),
            self.clipped_color_buffer.as_ref(),
            self.clipped_lighting_buffer.as_ref(),
            self.clipped_edge_buffer.as_ref(),
        ]
        .into_iter()
        .flatten()
        .fold(0usize, |total, buffer| {
            total.saturating_add(buffer.size() as usize)
        })
    }

    pub(super) fn preflight_scene3d(
        &self,
        device: &wgpu::Device,
        renderer_identity: &Arc<()>,
        target: &RenderTarget3d,
        scene: &Scene3d,
        camera: Camera3d,
        budget: Mesh3dRenderBudget,
    ) -> Result<(Camera3dUniform, SurfaceFrame), Mesh3dRenderError> {
        validate_target_identity(renderer_identity, target)?;
        validate_camera_target_aspect(camera, target.logical_viewport())?;
        let mut camera_uniform = Camera3dUniform::new(
            camera,
            target.width(),
            target.height(),
            target.pixels_per_logical(),
        )?;
        if scene.visible_object_count() > u32::MAX as usize {
            return Err(Mesh3dRenderError::InstanceCapacityTooLarge);
        }
        let capacity = scene
            .visible_object_count()
            .max(1)
            .checked_next_power_of_two()
            .ok_or(Mesh3dRenderError::InstanceCapacityTooLarge)?;
        let edge_capacity = edge_uniform_object_count(scene)
            .checked_next_power_of_two()
            .ok_or(Mesh3dRenderError::InstanceCapacityTooLarge)?;
        if !buffer_capacity_fits::<MeshInstanceGpu>(device, capacity)
            || edge_capacity
                .checked_mul(self.edge_object_stride)
                .is_none_or(|bytes| {
                    bytes as u64 > device.limits().max_buffer_size || bytes > u32::MAX as usize
                })
        {
            return Err(Mesh3dRenderError::InstanceCapacityTooLarge);
        }
        camera_uniform.environment = SurfaceEnvironmentGpu::new(scene, camera);
        let order = material::prepare_order(scene, camera, budget)?;
        let mut frame = surface::preflight(
            renderer_identity,
            scene,
            camera_uniform,
            budget,
            device.limits().max_buffer_size,
        )?;
        frame.report.sorting_capacity_bytes =
            order.capacity() * std::mem::size_of::<material::SurfaceDraw>();
        frame.order = order;
        Ok((camera_uniform, frame))
    }
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_scene3d(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer_identity: &Arc<()>,
        target: &RenderTarget3d,
        scene: &Scene3d,
        camera: Camera3d,
        budget: Mesh3dRenderBudget,
    ) -> Result<Mesh3dRenderReport, Mesh3dRenderError> {
        self.render_scene3d_with_timing(
            device,
            queue,
            renderer_identity,
            target,
            scene,
            camera,
            budget,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_scene3d_with_timing(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer_identity: &Arc<()>,
        target: &RenderTarget3d,
        scene: &Scene3d,
        camera: Camera3d,
        budget: Mesh3dRenderBudget,
        mut gpu_timing: Option<&mut gpu_timing::GpuTimingCollector>,
    ) -> Result<Mesh3dRenderReport, Mesh3dRenderError> {
        let upload_started_at = Instant::now();
        let previous_frame_cpu_bytes = self.retained_frame_cpu_bytes();
        let previous_frame_gpu_bytes = self.retained_frame_gpu_bytes();
        let (camera_uniform, surface_frame) =
            self.preflight_scene3d(device, renderer_identity, target, scene, camera, budget)?;
        let preflight_duration = upload_started_at.elapsed();
        let transient_frame_cpu_bytes = surface_frame
            .vertices
            .capacity()
            .saturating_mul(std::mem::size_of::<SurfaceClipVertex>())
            .saturating_add(
                surface_frame
                    .colors
                    .capacity()
                    .saturating_mul(std::mem::size_of::<MeshColorGpu>()),
            )
            .saturating_add(
                surface_frame
                    .lighting
                    .capacity()
                    .saturating_mul(std::mem::size_of::<SurfaceLightingVertex>()),
            )
            .saturating_add(
                surface_frame
                    .edges
                    .capacity()
                    .saturating_mul(std::mem::size_of::<SurfaceClipEdge>()),
            )
            .saturating_add(
                surface_frame
                    .order
                    .capacity()
                    .saturating_mul(std::mem::size_of::<material::SurfaceDraw>()),
            );
        let incoming_object_bytes = surface_frame
            .objects
            .capacity()
            .saturating_mul(std::mem::size_of::<SurfaceObject>());
        let visible_count = scene.visible_object_count();
        let edge_object_count = edge_uniform_object_count(scene);
        let uploaded_bytes = std::mem::size_of::<Camera3dUniform>()
            .saturating_add(visible_count.saturating_mul(std::mem::size_of::<MeshInstanceGpu>()))
            .saturating_add(edge_object_count.saturating_mul(self.edge_object_stride))
            .saturating_add(surface_frame.report.generated_upload_bytes());
        let upload_calls = 1
            + usize::from(visible_count > 0)
            + usize::from(edge_object_count > 0)
            + usize::from(!surface_frame.vertices.is_empty())
            + usize::from(!surface_frame.colors.is_empty())
            + usize::from(!surface_frame.lighting.is_empty())
            + usize::from(!surface_frame.edges.is_empty());
        let replacement_surface_buffer =
            if surface_frame.vertices.len() > self.clipped_surface_capacity {
                Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sim-engine clipped 3D surface buffer"),
                    size: (surface_frame.vertices.len() * std::mem::size_of::<SurfaceClipVertex>())
                        as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }))
            } else {
                None
            };
        let replacement_lighting_buffer = if surface_frame.lighting.len()
            > self.clipped_lighting_capacity
        {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sim-engine canonical 3D lighting attributes"),
                size: (surface_frame.lighting.len() * std::mem::size_of::<SurfaceLightingVertex>())
                    as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };
        let replacement_color_buffer = if surface_frame.colors.len() > self.clipped_color_capacity {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sim-engine clipped 3D vertex color buffer"),
                size: (surface_frame.colors.len() * std::mem::size_of::<MeshColorGpu>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };
        let replacement_edge_buffer = if surface_frame.edges.len() > self.clipped_edge_capacity {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sim-engine canonical clip-space 3D edge buffer"),
                size: (surface_frame.edges.len() * std::mem::size_of::<SurfaceClipEdge>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };
        let replace_instance_buffer = visible_count > self.instance_capacity;
        let replace_edge_buffer = edge_object_count > self.edge_object_capacity;
        let replace_instance_staging = visible_count > self.instances.capacity();
        let replace_edge_staging =
            edge_object_count * self.edge_object_stride > self.edge_object_bytes.capacity();
        let generated_replacements = [
            replacement_surface_buffer.as_ref(),
            replacement_lighting_buffer.as_ref(),
            replacement_color_buffer.as_ref(),
            replacement_edge_buffer.as_ref(),
        ];
        let buffer_allocation_count = generated_replacements.iter().flatten().count()
            + usize::from(replace_instance_buffer)
            + usize::from(replace_edge_buffer);
        let replacement_generated_bytes = generated_replacements
            .into_iter()
            .flatten()
            .fold(0usize, |total, buffer| {
                total.saturating_add(buffer.size() as usize)
            });
        self.ensure_frame_capacity(device, visible_count, edge_object_count)?;
        let peak_frame_cpu_bytes = previous_frame_cpu_bytes
            .saturating_add(transient_frame_cpu_bytes)
            .saturating_add(incoming_object_bytes)
            .saturating_add(if replace_instance_staging {
                self.instances
                    .capacity()
                    .saturating_mul(std::mem::size_of::<MeshInstanceGpu>())
            } else {
                0
            })
            .saturating_add(if replace_edge_staging {
                self.edge_object_bytes.capacity()
            } else {
                0
            });
        let peak_frame_gpu_bytes = previous_frame_gpu_bytes
            .saturating_add(replacement_generated_bytes)
            .saturating_add(if replace_instance_buffer {
                self.instance_buffer.size() as usize
            } else {
                0
            })
            .saturating_add(if replace_edge_buffer {
                self.edge_object_buffer.size() as usize
            } else {
                0
            });
        if let Some(buffer) = replacement_surface_buffer {
            self.clipped_surface_buffer = Some(buffer);
            self.clipped_surface_capacity = surface_frame.vertices.len();
        }
        if let Some(buffer) = replacement_edge_buffer {
            self.clipped_edge_buffer = Some(buffer);
            self.clipped_edge_capacity = surface_frame.edges.len();
        }
        if let Some(buffer) = replacement_lighting_buffer {
            self.clipped_lighting_buffer = Some(buffer);
            self.clipped_lighting_capacity = surface_frame.lighting.len();
        }
        if let Some(buffer) = &self.clipped_lighting_buffer
            && !surface_frame.lighting.is_empty()
        {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&surface_frame.lighting));
        }
        if let Some(buffer) = replacement_color_buffer {
            self.clipped_color_buffer = Some(buffer);
            self.clipped_color_capacity = surface_frame.colors.len();
        }
        self.clipped_surface_objects = surface_frame.objects;
        if let Some(buffer) = &self.clipped_color_buffer
            && !surface_frame.colors.is_empty()
        {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&surface_frame.colors));
        }
        if let Some(buffer) = &self.clipped_surface_buffer
            && !surface_frame.vertices.is_empty()
        {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&surface_frame.vertices));
        }
        if let Some(buffer) = &self.clipped_edge_buffer
            && !surface_frame.edges.is_empty()
        {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&surface_frame.edges));
        }
        self.instances.clear();
        self.edge_object_bytes.clear();
        self.edge_object_bytes
            .resize(edge_object_count.saturating_mul(self.edge_object_stride), 0);
        let triangle_count = surface_frame.report.submitted_triangle_count();
        let mut edge_count = 0usize;
        for (object_index, instance) in scene
            .instances()
            .iter()
            .filter(|instance| instance.visible)
            .enumerate()
        {
            let model_rows = self.clipped_surface_objects[object_index].model_rows;
            let normal_rows = self.clipped_surface_objects[object_index]
                .transport
                .normal_rows;
            self.instances.push(MeshInstanceGpu {
                normal_row_0: normal_rows[0],
                normal_row_1: normal_rows[1],
                normal_row_2: normal_rows[2],
                model_row_0: model_rows[0],
                model_row_1: model_rows[1],
                model_row_2: model_rows[2],
                surface: material::surface_parameters(instance.style.surface_style()),
                uv_transform: texture::material_uv_parameters(instance.mesh.material()),
                color: {
                    let color = instance
                        .style
                        .surface_style()
                        .map_or(Color::BLACK, |surface| surface.color())
                        .to_array();
                    let tint = instance
                        .mesh
                        .material()
                        .map_or(Color::WHITE, TextureMaterial3d::tint)
                        .to_array();
                    std::array::from_fn(|index| color[index] * tint[index])
                },
            });
            if let Some(style) = instance
                .wireframe()
                .filter(|_| instance.mesh.edge_count > 0)
            {
                let instance_edge_count = instance.mesh.source().display_edges().len();
                edge_count = edge_count.saturating_add(instance_edge_count);
                let hidden_color = style.hidden_color().unwrap_or(style.visible_color());
                let hidden_width = style.hidden_width().map_or(0.0, LogicalPixels::get);
                let (dash_length, gap_length) = style
                    .hidden_pattern()
                    .map_or((1.0, 1.0), |(dash, gap)| (dash.get(), gap.get()));
                let edge_uniform = EdgeObjectUniform {
                    model_row_0: model_rows[0],
                    model_row_1: model_rows[1],
                    model_row_2: model_rows[2],
                    visible_color: style.visible_color().to_array(),
                    hidden_color: hidden_color.to_array(),
                    edge_style: [
                        style.visible_width().get(),
                        hidden_width,
                        dash_length,
                        gap_length,
                    ],
                };
                let start = object_index * self.edge_object_stride;
                let end = start + std::mem::size_of::<EdgeObjectUniform>();
                self.edge_object_bytes[start..end]
                    .copy_from_slice(bytemuck::bytes_of(&edge_uniform));
            }
        }

        queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
        if !self.instances.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances),
            );
        }
        if !self.edge_object_bytes.is_empty() {
            queue.write_buffer(&self.edge_object_buffer, 0, &self.edge_object_bytes);
        }
        let upload = upload_started_at.elapsed();
        let staging_upload_duration = upload.saturating_sub(preflight_duration);
        let encode_started_at = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-engine retained 3D scene encoder"),
        });
        let timing = gpu_timing
            .as_deref_mut()
            .and_then(|collector| collector.reserve(GpuTimingSource::Scene3d));
        let draw_call_count = encode_ordered_scene_pass(
            &mut encoder,
            self,
            &target.color.view,
            &target.depth_view,
            scene.background(),
            scene.instances(),
            &surface_frame.order,
            timing.as_ref().map(|timing| timing.timestamp_writes()),
        );
        if let Some(timing) = &timing {
            timing.resolve(&mut encoder);
        }
        queue.submit([encoder.finish()]);
        let gpu_timing_id = match (gpu_timing, timing) {
            (Some(collector), Some(timing)) => Some(collector.submitted(timing)),
            _ => None,
        };
        let encode_submit = encode_started_at.elapsed();
        Ok(Mesh3dRenderReport {
            object_count: visible_count,
            triangle_count,
            edge_count,
            render_pass_count: 1,
            draw_call_count,
            upload,
            preflight_duration,
            staging_upload_duration,
            encode_submit,
            preflight: surface_frame.report,
            gpu_timing_id,
            uploaded_bytes,
            upload_calls,
            buffer_allocation_count,
            retained_frame_cpu_bytes: self.retained_frame_cpu_bytes(),
            retained_frame_gpu_bytes: self.retained_frame_gpu_bytes(),
            staging_capacity_bytes: self
                .retained_frame_cpu_bytes()
                .saturating_add(transient_frame_cpu_bytes),
            peak_frame_cpu_bytes,
            peak_frame_gpu_bytes,
        })
    }
}
