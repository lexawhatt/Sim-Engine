//! 3D pipeline initialization and bounded frame resource growth.

use super::*;

impl Mesh3dRenderer {
    pub(in crate::renderer) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sim-engine retained 3D mesh shader"),
            source: wgpu::ShaderSource::Wgsl(lighting::shader_source(include_str!(
                "primitive.wgsl"
            ))),
        });
        let camera_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine 3D camera uniform buffer"),
            size: std::mem::size_of::<Camera3dUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim-engine 3D camera bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-engine 3D camera bind group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buffer.as_entire_binding(),
            }],
        });
        let edge_object_stride = align_to(
            std::mem::size_of::<EdgeObjectUniform>(),
            device.limits().min_uniform_buffer_offset_alignment as usize,
        );
        let edge_object_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sim-engine 3D edge object bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<EdgeObjectUniform>() as u64,
                        ),
                    },
                    count: None,
                }],
            });
        let edge_object_buffer =
            create_edge_object_buffer(device, edge_object_stride, INITIAL_INSTANCE_CAPACITY);
        let edge_object_bind_group =
            create_edge_object_bind_group(device, &edge_object_layout, &edge_object_buffer);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-engine retained 3D mesh pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let pipeline = create_surface_pipelines(
            device,
            format,
            &pipeline_layout,
            &shader,
            SurfaceLayout {
                clipped: false,
                textured: false,
                colored: false,
            },
        );
        let clipped_surface_pipeline = create_surface_pipelines(
            device,
            format,
            &pipeline_layout,
            &shader,
            SurfaceLayout {
                clipped: true,
                textured: false,
                colored: false,
            },
        );
        let edge_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim-engine 3D edge pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&edge_object_layout)],
            immediate_size: 0,
        });
        let visible_edge_pipeline = create_edge_pipeline(
            device,
            &shader,
            &edge_pipeline_layout,
            format,
            wgpu::CompareFunction::LessEqual,
            "mesh3d_visible_edge_fs_main",
            "sim-engine visible 3D edge pipeline",
            false,
        );
        let hidden_edge_pipeline = create_edge_pipeline(
            device,
            &shader,
            &edge_pipeline_layout,
            format,
            wgpu::CompareFunction::Greater,
            "mesh3d_hidden_edge_fs_main",
            "sim-engine hidden 3D edge pipeline",
            false,
        );
        let clipped_visible_edge_pipeline = create_edge_pipeline(
            device,
            &shader,
            &edge_pipeline_layout,
            format,
            wgpu::CompareFunction::LessEqual,
            "mesh3d_visible_edge_fs_main",
            "sim-engine canonical visible 3D edge pipeline",
            true,
        );
        let clipped_hidden_edge_pipeline = create_edge_pipeline(
            device,
            &shader,
            &edge_pipeline_layout,
            format,
            wgpu::CompareFunction::Greater,
            "mesh3d_hidden_edge_fs_main",
            "sim-engine canonical hidden 3D edge pipeline",
            true,
        );
        Self {
            dynamic_scratch: dynamic::DynamicMesh3dScratch::default(),
            textures: MeshTextureRenderer::new(device, format, &camera_layout),
            pipeline,
            colored_pipeline: create_surface_pipelines(
                device,
                format,
                &pipeline_layout,
                &shader,
                SurfaceLayout {
                    clipped: false,
                    textured: false,
                    colored: true,
                },
            ),
            colored_clipped_pipeline: create_surface_pipelines(
                device,
                format,
                &pipeline_layout,
                &shader,
                SurfaceLayout {
                    clipped: true,
                    textured: false,
                    colored: true,
                },
            ),
            clipped_color_buffer: None,
            clipped_color_capacity: 0,
            clipped_lighting_buffer: None,
            clipped_lighting_capacity: 0,
            clipped_surface_pipeline,
            clipped_surface_buffer: None,
            clipped_surface_capacity: 0,
            clipped_surface_objects: Vec::new(),
            clipped_visible_edge_pipeline,
            clipped_hidden_edge_pipeline,
            clipped_edge_buffer: None,
            clipped_edge_capacity: 0,
            visible_edge_pipeline,
            hidden_edge_pipeline,
            camera_uniform_buffer,
            camera_bind_group,
            instance_buffer: create_instance_buffer(device, INITIAL_INSTANCE_CAPACITY),
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            instances: Vec::new(),
            edge_object_layout,
            edge_object_bind_group,
            edge_object_buffer,
            edge_object_stride,
            edge_object_capacity: INITIAL_INSTANCE_CAPACITY,
            edge_object_bytes: Vec::new(),
        }
    }

    pub(super) fn ensure_frame_capacity(
        &mut self,
        device: &wgpu::Device,
        object_count: usize,
        edge_object_count: usize,
    ) -> Result<(), Mesh3dRenderError> {
        let instance_capacity = if object_count > self.instance_capacity {
            Some(
                object_count
                    .checked_next_power_of_two()
                    .filter(|capacity| buffer_capacity_fits::<MeshInstanceGpu>(device, *capacity))
                    .ok_or(Mesh3dRenderError::InstanceCapacityTooLarge)?,
            )
        } else {
            None
        };
        let edge_capacity = if edge_object_count > self.edge_object_capacity {
            Some(
                edge_object_count
                    .checked_next_power_of_two()
                    .filter(|capacity| {
                        capacity
                            .checked_mul(self.edge_object_stride)
                            .is_some_and(|bytes| {
                                bytes as u64 <= device.limits().max_buffer_size
                                    && bytes <= u32::MAX as usize
                            })
                    })
                    .ok_or(Mesh3dRenderError::InstanceCapacityTooLarge)?,
            )
        } else {
            None
        };
        let required_edge_bytes = edge_object_count
            .checked_mul(self.edge_object_stride)
            .ok_or(Mesh3dRenderError::InstanceCapacityTooLarge)?;
        let replace_instance_staging = self.instances.capacity() < object_count;
        let replace_edge_staging = self.edge_object_bytes.capacity() < required_edge_bytes;
        if instance_capacity.is_none()
            && edge_capacity.is_none()
            && !replace_instance_staging
            && !replace_edge_staging
        {
            return Ok(());
        }
        let replacement_instances = if replace_instance_staging {
            let mut instances = Vec::new();
            instances
                .try_reserve_exact(object_count)
                .map_err(|_| Mesh3dRenderError::InstanceCapacityTooLarge)?;
            Some(instances)
        } else {
            None
        };
        let replacement_edge_bytes = if replace_edge_staging {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(required_edge_bytes)
                .map_err(|_| Mesh3dRenderError::InstanceCapacityTooLarge)?;
            Some(bytes)
        } else {
            None
        };
        let replacement_instance_buffer =
            instance_capacity.map(|capacity| create_instance_buffer(device, capacity));
        let replacement_edge_resources = edge_capacity.map(|capacity| {
            let buffer = create_edge_object_buffer(device, self.edge_object_stride, capacity);
            let bind_group =
                create_edge_object_bind_group(device, &self.edge_object_layout, &buffer);
            (buffer, bind_group)
        });
        // No fallible operation remains after the first renderer field changes.
        if let Some(instances) = replacement_instances {
            self.instances = instances;
        }
        if let (Some(capacity), Some(buffer)) = (instance_capacity, replacement_instance_buffer) {
            self.instance_buffer = buffer;
            self.instance_capacity = capacity;
        }
        if let Some(bytes) = replacement_edge_bytes {
            self.edge_object_bytes = bytes;
        }
        if let (Some(capacity), Some((buffer, bind_group))) =
            (edge_capacity, replacement_edge_resources)
        {
            self.edge_object_buffer = buffer;
            self.edge_object_bind_group = bind_group;
            self.edge_object_capacity = capacity;
        }
        Ok(())
    }
}
