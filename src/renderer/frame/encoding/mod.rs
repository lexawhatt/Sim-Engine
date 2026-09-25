//! Per-pass encoding state. Draw order and every draw are preserved.

use std::sync::Arc;

use super::{FrameBinding, ReadyItem, ReadySource, effective_scissor};
use crate::renderer::{CompositionPipelines, ImageRenderer, ScissorRect, WgpuRenderer};

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(in crate::renderer) use tests::assert_gpu_encoding_contract;

/// Buffer equality is wgpu resource identity, including cloned wrappers.
/// Keys are borrowed only for the active pass: no handle can be replaced or
/// expire while its state is cached. All current bindings use entire buffers.
struct LastValue<T> {
    value: Option<T>,
}
impl<T> Default for LastValue<T> {
    fn default() -> Self {
        Self { value: None }
    }
}
impl<T: PartialEq + Copy> LastValue<T> {
    fn changed(&mut self, value: T) -> bool {
        if self.value == Some(value) {
            return false;
        }
        self.value = Some(value);
        true
    }
}

#[cfg(test)]
#[derive(Debug, Default, Clone, Copy)]
struct SetterCounts {
    vertex_buffers: usize,
    scissors: usize,
}

struct PassState<'pass, const DEDUPLICATE: bool> {
    vertex_buffers: [LastValue<&'pass wgpu::Buffer>; 2],
    scissor: LastValue<ScissorRect>,
    #[cfg(test)]
    counts: SetterCounts,
}

impl<const DEDUPLICATE: bool> Default for PassState<'_, DEDUPLICATE> {
    fn default() -> Self {
        Self {
            vertex_buffers: std::array::from_fn(|_| LastValue::default()),
            scissor: LastValue::default(),
            #[cfg(test)]
            counts: SetterCounts::default(),
        }
    }
}

impl<'pass, const DEDUPLICATE: bool> PassState<'pass, DEDUPLICATE> {
    fn vertex_buffer(
        &mut self,
        pass: &mut wgpu::RenderPass<'pass>,
        slot: usize,
        buffer: &'pass wgpu::Buffer,
    ) {
        if !DEDUPLICATE || self.vertex_buffers[slot].changed(buffer) {
            pass.set_vertex_buffer(slot as u32, buffer.slice(..));
            #[cfg(test)]
            {
                self.counts.vertex_buffers += 1;
            }
        }
    }
    fn scissor(&mut self, pass: &mut wgpu::RenderPass<'pass>, rect: ScissorRect) {
        if !DEDUPLICATE || self.scissor.changed(rect) {
            pass.set_scissor_rect(rect.x, rect.y, rect.width, rect.height);
            #[cfg(test)]
            {
                self.counts.scissors += 1;
            }
        }
    }
}

/// wgpu30 already eliminates duplicate pipelines during command recording.
/// Vertex/scissor setters still append commands, validate and touch HAL state,
/// so only those receive an additional cache here. Pipeline changes preserve
/// vertex slots in wgpu; all bind groups are still explicitly selected per item.
struct EncodingResources<'pass> {
    pipeline: &'pass wgpu::RenderPipeline,
    screen_primitive_pipeline: &'pass wgpu::RenderPipeline,
    dynamic_pipeline: &'pass wgpu::RenderPipeline,
    particle_pipeline: &'pass wgpu::RenderPipeline,
    heatmap_pipeline: &'pass wgpu::RenderPipeline,
    image_renderer: &'pass ImageRenderer,
    composition_pipelines: &'pass CompositionPipelines,
    vertex_buffer: &'pass Arc<wgpu::Buffer>,
    particle_unit_buffer: &'pass wgpu::Buffer,
    scale_factor: f64,
}

pub(super) fn encode_items<'pass>(
    renderer: &'pass WgpuRenderer,
    pass: &mut wgpu::RenderPass<'pass>,
    ready: &'pass [ReadyItem<'_>],
    bindings: &'pass [FrameBinding],
) {
    let resources = EncodingResources {
        pipeline: &renderer.pipeline,
        screen_primitive_pipeline: &renderer.screen_primitive_pipeline,
        dynamic_pipeline: &renderer.dynamic_pipeline,
        particle_pipeline: &renderer.particle_pipeline,
        heatmap_pipeline: &renderer.heatmap_pipeline,
        image_renderer: &renderer.image_renderer,
        composition_pipelines: &renderer.composition_pipelines,
        vertex_buffer: &renderer.vertex_buffer,
        particle_unit_buffer: &renderer.particle_unit_buffer,
        scale_factor: renderer.scale_factor,
    };
    // This must be initialized after every begin_render_pass, never retained
    // between passes or frames, and never shared with another command encoder.
    let mut state = PassState::<true>::default();
    for (item, binding) in ready.iter().zip(bindings) {
        encode_ready_item(&resources, pass, item, binding, &mut state);
    }
}

fn encode_ready_item<'pass, const DEDUPLICATE: bool>(
    renderer: &EncodingResources<'pass>,
    pass: &mut wgpu::RenderPass<'pass>,
    item: &'pass ReadyItem<'_>,
    binding: &'pass FrameBinding,
    state: &mut PassState<'pass, DEDUPLICATE>,
) {
    match item {
        ReadyItem::Geometry(geometry) => {
            let (pipeline, vertex_buffer) = match geometry.source {
                ReadySource::Streaming => (renderer.pipeline, renderer.vertex_buffer.as_ref()),
                ReadySource::Prepared(buffer) => (renderer.pipeline, buffer),
                ReadySource::Dynamic(buffer) => (renderer.dynamic_pipeline, buffer),
                ReadySource::ScreenPrimitives(buffer) => {
                    (renderer.screen_primitive_pipeline, buffer)
                }
            };
            encode_geometry(
                pass,
                pipeline,
                &binding.bind_group,
                vertex_buffer,
                geometry.vertex_count,
                &geometry.batches,
                matches!(geometry.source, ReadySource::ScreenPrimitives(_)),
                geometry.viewport,
                renderer.scale_factor as f32,
                state,
            );
        }
        ReadyItem::Particle {
            field,
            visible_count,
            viewport,
            ..
        } => {
            if *visible_count == 0 || viewport.item_clipped_out {
                return;
            }
            let scissor = viewport.item_clip.unwrap_or(viewport.scissor);
            pass.set_pipeline(renderer.particle_pipeline);
            pass.set_bind_group(0, &binding.bind_group, &[]);
            state.vertex_buffer(pass, 0, renderer.particle_unit_buffer);
            state.vertex_buffer(pass, 1, &field.instance_buffer);
            state.scissor(pass, scissor);
            pass.draw(0..6, 0..*visible_count as u32);
        }
        ReadyItem::Scalar { viewport, .. } => {
            if viewport.item_clipped_out {
                return;
            }
            let scissor = viewport.item_clip.unwrap_or(viewport.scissor);
            pass.set_pipeline(renderer.heatmap_pipeline);
            pass.set_bind_group(0, &binding.bind_group, &[]);
            state.scissor(pass, scissor);
            pass.draw(0..6, 0..1);
        }
        ReadyItem::Image { viewport, .. } => {
            if viewport.item_clipped_out {
                return;
            }
            let scissor = viewport.item_clip.unwrap_or(viewport.scissor);
            pass.set_pipeline(&renderer.image_renderer.pipeline);
            pass.set_bind_group(0, &binding.bind_group, &[]);
            state.scissor(pass, scissor);
            pass.draw(0..6, 0..1);
        }
        ReadyItem::ImageBatch {
            batch,
            spans,
            viewport,
            ..
        } => encode_image_batch(
            pass,
            renderer.image_renderer,
            &binding.bind_group,
            batch,
            *spans,
            *viewport,
            renderer.scale_factor as f32,
            state,
        ),
        ReadyItem::Target {
            blend_mode,
            viewport,
            ..
        } => {
            if viewport.item_clipped_out {
                return;
            }
            let scissor = viewport.item_clip.unwrap_or(viewport.scissor);
            pass.set_pipeline(renderer.composition_pipelines.pipeline(*blend_mode));
            pass.set_bind_group(0, &binding.bind_group, &[]);
            state.scissor(pass, scissor);
            pass.draw(0..6, 0..1);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_geometry<'pass, const DEDUPLICATE: bool>(
    pass: &mut wgpu::RenderPass<'pass>,
    pipeline: &'pass wgpu::RenderPipeline,
    binding: &'pass wgpu::BindGroup,
    buffer: &'pass wgpu::Buffer,
    vertex_count: usize,
    batches: &[crate::renderer::PreparedDrawBatch],
    instanced: bool,
    viewport: super::ResolvedViewport,
    scale: f32,
    state: &mut PassState<'pass, DEDUPLICATE>,
) {
    if vertex_count == 0 {
        return;
    }
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, binding, &[]);
    state.vertex_buffer(pass, 0, buffer);
    for batch in batches {
        let Some(scissor) = effective_scissor(viewport, batch.screen_clip, scale) else {
            continue;
        };
        state.scissor(pass, scissor);
        if instanced {
            pass.draw(0..6, batch.vertex_range.clone());
        } else {
            pass.draw(batch.vertex_range.clone(), 0..1);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::renderer) fn encode_screen_primitives<'pass>(
    pass: &mut wgpu::RenderPass<'pass>,
    pipeline: &'pass wgpu::RenderPipeline,
    binding: &'pass wgpu::BindGroup,
    batch: &'pass crate::renderer::ScreenPrimitiveBatch2d,
    region: crate::LogicalViewportRegion,
    item_clip: Option<crate::ScreenClipRect>,
    scale: f32,
    width: u32,
    height: u32,
) {
    let Some(scissor) = crate::renderer::logical_viewport_scissor(
        region.origin().to_vec2(),
        region.viewport(),
        scale,
        width,
        height,
    ) else {
        return;
    };
    let clip = item_clip
        .and_then(|clip| crate::renderer::screen_clip_to_scissor(clip, region.viewport(), scale))
        .and_then(|clip| crate::renderer::offset_scissor(clip, scissor));
    let viewport = super::ResolvedViewport {
        viewport: region.viewport(),
        origin: region.origin().to_vec2(),
        scissor,
        item_clip: clip,
        item_clipped_out: item_clip.is_some() && clip.is_none(),
    };
    encode_geometry(
        pass,
        pipeline,
        binding,
        &batch.instance_buffer,
        batch.vertex_count(),
        &batch.draw_batches,
        true,
        viewport,
        scale,
        &mut PassState::<true>::default(),
    );
}

#[cfg(test)]
pub(in crate::renderer) use encode_screen_primitives as encode_screen_primitives_for_test;

#[allow(clippy::too_many_arguments)]
fn encode_image_batch<'pass, const DEDUPLICATE: bool>(
    pass: &mut wgpu::RenderPass<'pass>,
    renderer: &'pass ImageRenderer,
    binding: &'pass wgpu::BindGroup,
    batch: &'pass crate::renderer::ImageBatch2d,
    spans: Option<&[super::GlyphBatchSpan]>,
    viewport: super::ResolvedViewport,
    scale_factor: f32,
    state: &mut PassState<'pass, DEDUPLICATE>,
) {
    if batch.sprite_count() == 0 || viewport.item_clipped_out {
        return;
    }
    pass.set_pipeline(&renderer.batch_pipeline);
    pass.set_bind_group(0, binding, &[]);
    state.vertex_buffer(pass, 0, &batch.instance_buffer);
    if let Some(spans) = spans {
        for span in spans {
            if let Some(scissor) = effective_scissor(viewport, span.clip, scale_factor) {
                state.scissor(pass, scissor);
                pass.draw(0..6, span.first..span.end);
            }
        }
    } else {
        state.scissor(pass, viewport.item_clip.unwrap_or(viewport.scissor));
        pass.draw(0..6, 0..batch.sprite_count() as u32);
    }
}

// Transport into the production encoder without a window surface. The fixture
// owns pixel expectations, color conversion, input data and sample selection.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(in crate::renderer) fn encode_glyph_batch_for_test<'pass>(
    pass: &mut wgpu::RenderPass<'pass>,
    renderer: &'pass ImageRenderer,
    binding: &'pass wgpu::BindGroup,
    batch: &'pass crate::renderer::GlyphBatch2d,
    region: crate::LogicalViewportRegion,
    item_clip: Option<crate::ScreenClipRect>,
    scale_factor: f32,
    width: u32,
    height: u32,
) {
    let origin = region.origin().to_vec2();
    let viewport = region.viewport();
    let scissor =
        super::logical_viewport_scissor(origin, viewport, scale_factor, width, height).unwrap();
    let (item_clip, item_clipped_out) = match item_clip {
        Some(clip) => {
            let resolved = super::screen_clip_to_scissor(clip, viewport, scale_factor)
                .and_then(|clip| super::offset_scissor(clip, scissor));
            (resolved, resolved.is_none())
        }
        None => (None, false),
    };
    let viewport = super::ResolvedViewport {
        viewport,
        origin,
        scissor,
        item_clip,
        item_clipped_out,
    };
    encode_image_batch(
        pass,
        renderer,
        binding,
        &batch.batch,
        Some(&batch.spans),
        viewport,
        scale_factor,
        &mut PassState::<true>::default(),
    );
}
