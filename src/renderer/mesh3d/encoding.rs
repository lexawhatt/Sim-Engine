//! Ordered surface and mathematical-edge draw encoding.

use super::{
    Color, Mesh3dInstance, Mesh3dRenderer, MeshColorGpu, MeshInstanceGpu, SurfaceClipVertex,
    SurfaceLighting3d, SurfaceLightingVertex, batch, material, premultiplied_wgpu_color,
};

#[cfg(test)]
pub(super) fn encode_scene_pass(
    encoder: &mut wgpu::CommandEncoder,
    renderer: &Mesh3dRenderer,
    color_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    background: Color,
    instances: &[Mesh3dInstance],
) {
    encode_ordered_scene_pass(
        encoder,
        renderer,
        color_view,
        depth_view,
        background,
        instances,
        &[],
        None,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_ordered_scene_pass(
    encoder: &mut wgpu::CommandEncoder,
    renderer: &Mesh3dRenderer,
    color_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    background: Color,
    instances: &[Mesh3dInstance],
    order: &[material::SurfaceDraw],
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> usize {
    let mut draw_call_count = 0;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("sim-engine retained 3D mesh pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(premultiplied_wgpu_color(background)),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&renderer.pipeline);
    pass.set_bind_group(0, &renderer.camera_bind_group, &[]);
    let mut insertion = instances
        .iter()
        .filter(|instance| instance.visible)
        .enumerate();
    let mut sorted = order.iter();
    let mut draws = std::iter::from_fn(|| {
        if order.is_empty() {
            insertion.next()
        } else {
            sorted
                .next()
                .map(|entry| (entry.visible_index, &instances[entry.scene_index]))
        }
    })
    .peekable();
    while let Some((instance_index, instance)) = draws.next() {
        if instance.style.surface_style().is_none() || instance.mesh.index_count == 0 {
            continue;
        }
        let instance_start =
            (instance_index * std::mem::size_of::<MeshInstanceGpu>()) as wgpu::BufferAddress;
        let instance_end =
            instance_start + std::mem::size_of::<MeshInstanceGpu>() as wgpu::BufferAddress;
        let Some(surface_style) = instance.style.surface_style() else {
            continue;
        };
        let material = instance.mesh.material();
        let colored = instance.mesh.color_buffer.is_some();
        if let Some(material) = material {
            pass.set_bind_group(1, material.bind_group(), &[]);
        }
        let object = renderer.surface_frame.objects.get(instance_index);
        if let Some(range) = object.and_then(|object| object.generated.as_ref()) {
            if range.is_empty() {
                continue;
            }
            let Some(buffer) = &renderer.clipped_surface_buffer else {
                continue;
            };
            let pipelines = match (material.is_some(), colored) {
                (false, false) => &renderer.clipped_surface_pipeline,
                (false, true) => &renderer.colored_clipped_pipeline,
                (true, false) => &renderer.textures.clipped_pipeline,
                (true, true) => &renderer.textures.colored_clipped_pipeline,
            };
            pass.set_pipeline(pipelines.for_style(surface_style));
            let stride = std::mem::size_of::<SurfaceClipVertex>() as u64;
            pass.set_vertex_buffer(
                0,
                buffer.slice(u64::from(range.start) * stride..u64::from(range.end) * stride),
            );
            pass.set_vertex_buffer(
                1,
                renderer.instance_buffer.slice(instance_start..instance_end),
            );
            if let (Some(buffer), Some(colors)) = (
                &renderer.clipped_color_buffer,
                object.and_then(|object| object.generated_colors.as_ref()),
            ) {
                let stride = std::mem::size_of::<MeshColorGpu>() as u64;
                pass.set_vertex_buffer(
                    2,
                    buffer.slice(u64::from(colors.start) * stride..u64::from(colors.end) * stride),
                );
            }
            if let (Some(buffer), Some(lighting)) = (
                &renderer.clipped_lighting_buffer,
                object.and_then(|object| object.generated_lighting.as_ref()),
            ) {
                let stride = std::mem::size_of::<SurfaceLightingVertex>() as u64;
                pass.set_vertex_buffer(
                    2 + u32::from(colored),
                    buffer.slice(
                        u64::from(lighting.start) * stride..u64::from(lighting.end) * stride,
                    ),
                );
            }
            pass.draw(0..range.end - range.start, 0..1);
            draw_call_count += 1;
            continue;
        }
        let Some(index_buffer) = &instance.mesh.index_buffer else {
            continue;
        };
        let mut instance_count = 1;
        while let Some(&(next_index, next)) = draws.peek() {
            if next_index != instance_index + instance_count
                || renderer
                    .surface_frame
                    .objects
                    .get(next_index)
                    .is_some_and(|object| object.generated.is_some())
                || !batch::compatible(instance, next)
            {
                break;
            }
            instance_count += 1;
            draws.next();
        }
        let instance_end = instance_start
            + (instance_count * std::mem::size_of::<MeshInstanceGpu>()) as wgpu::BufferAddress;
        let pipelines = match (material.is_some(), colored) {
            (false, false) => &renderer.pipeline,
            (false, true) => &renderer.colored_pipeline,
            (true, false) => &renderer.textures.retained_pipeline,
            (true, true) => &renderer.textures.colored_retained_pipeline,
        };
        pass.set_pipeline(pipelines.for_style(surface_style));
        pass.set_vertex_buffer(0, instance.mesh.vertex_buffer.slice(..));
        let instance_slot = if material.is_some() {
            if let Some(coordinates) = &instance.mesh.texture_coordinate_buffer {
                pass.set_vertex_buffer(1, coordinates.slice(..));
            }
            2
        } else {
            1
        };
        pass.set_vertex_buffer(
            instance_slot,
            renderer.instance_buffer.slice(instance_start..instance_end),
        );
        if let Some(buffer) = &instance.mesh.color_buffer {
            pass.set_vertex_buffer(instance_slot + 1, buffer.slice(..));
        }
        if surface_style.lighting() == SurfaceLighting3d::Lambert
            && let Some(buffer) = &instance.mesh.normal_buffer
        {
            pass.set_vertex_buffer(instance_slot + 1 + u32::from(colored), buffer.slice(..));
        }
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..instance.mesh.index_count, 0, 0..instance_count as u32);
        draw_call_count += 1;
    }
    pass.set_pipeline(&renderer.hidden_edge_pipeline);
    for (object_index, instance) in instances
        .iter()
        .filter(|instance| instance.visible)
        .enumerate()
    {
        let Some(style) = instance.wireframe() else {
            continue;
        };
        let Some(edge_buffer) = instance.mesh.edge_buffer.as_ref() else {
            continue;
        };
        if !style.hidden_enabled() || instance.mesh.edge_count == 0 {
            continue;
        }
        let dynamic_offset = (object_index * renderer.edge_object_stride) as u32;
        pass.set_bind_group(1, &renderer.edge_object_bind_group, &[dynamic_offset]);
        if let Some(range) = renderer
            .surface_frame
            .objects
            .get(object_index)
            .and_then(|object| object.generated_edges.as_ref())
        {
            if let Some(buffer) = &renderer.clipped_edge_buffer
                && !range.is_empty()
            {
                pass.set_pipeline(&renderer.clipped_hidden_edge_pipeline);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..6, range.clone());
                draw_call_count += 1;
            }
            continue;
        }
        pass.set_pipeline(&renderer.hidden_edge_pipeline);
        pass.set_vertex_buffer(0, edge_buffer.slice(..));
        pass.draw(0..6, 0..instance.mesh.edge_count);
        draw_call_count += 1;
    }
    pass.set_pipeline(&renderer.visible_edge_pipeline);
    for (object_index, instance) in instances
        .iter()
        .filter(|instance| instance.visible)
        .enumerate()
    {
        if instance.wireframe().is_none() || instance.mesh.edge_count == 0 {
            continue;
        }
        let Some(edge_buffer) = instance.mesh.edge_buffer.as_ref() else {
            continue;
        };
        let dynamic_offset = (object_index * renderer.edge_object_stride) as u32;
        pass.set_bind_group(1, &renderer.edge_object_bind_group, &[dynamic_offset]);
        if let Some(range) = renderer
            .surface_frame
            .objects
            .get(object_index)
            .and_then(|object| object.generated_edges.as_ref())
        {
            if let Some(buffer) = &renderer.clipped_edge_buffer
                && !range.is_empty()
            {
                pass.set_pipeline(&renderer.clipped_visible_edge_pipeline);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..6, range.clone());
                draw_call_count += 1;
            }
            continue;
        }
        pass.set_pipeline(&renderer.visible_edge_pipeline);
        pass.set_vertex_buffer(0, edge_buffer.slice(..));
        pass.draw(0..6, 0..instance.mesh.edge_count);
        draw_call_count += 1;
    }
    draw_call_count
}
