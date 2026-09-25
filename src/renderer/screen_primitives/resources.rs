use super::super::{PreparedDrawBatch, buffer_capacity_fits, submit_pending_uploads};
use super::types::cpu_bytes;
use super::{
    ScreenPrimitive2d, ScreenPrimitiveBatch2d, ScreenPrimitiveBudget, ScreenPrimitiveError,
    ScreenPrimitiveGpu, ScreenPrimitiveUploadReport,
};
use std::sync::{Arc, Mutex};

fn limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), ScreenPrimitiveError> {
    if required > limit {
        Err(ScreenPrimitiveError::BudgetExceeded {
            resource,
            limit,
            required,
        })
    } else {
        Ok(())
    }
}

fn reserve<T>(count: usize) -> Result<Vec<T>, ScreenPrimitiveError> {
    let mut items = Vec::new();
    items
        .try_reserve_exact(count)
        .map_err(|_| ScreenPrimitiveError::AllocationFailed {
            requested_bytes: count.saturating_mul(std::mem::size_of::<T>()),
        })?;
    Ok(items)
}

fn span_count(primitives: &[ScreenPrimitive2d]) -> usize {
    primitives
        .iter()
        .enumerate()
        .filter(|(index, primitive)| *index == 0 || primitives[*index - 1].clip != primitive.clip)
        .count()
}

pub(super) struct SnapshotMetadata {
    pub(super) counts: crate::PrimitiveCommandCounts,
    pub(super) certificate: super::certificate::Certificate,
    pub(super) quality: super::quality::Quality,
}

pub(super) fn fill_instances(
    instances: &mut Vec<ScreenPrimitiveGpu>,
    primitives: &[ScreenPrimitive2d],
) -> SnapshotMetadata {
    let mut counts = crate::PrimitiveCommandCounts::default();
    let mut certificate = super::certificate::Certificate::empty();
    let mut source_error = 0.0_f64;
    instances.clear();
    // One trusted-length input traversal emits instances and collects all
    // immutable proof/count metadata. Coordinate maxima are shared with the
    // quality bound instead of scanning the same corners again.
    instances.extend(primitives.iter().map(|primitive| {
        counts.increment(match primitive.gpu.shape[0] as u32 {
            0 => crate::ScenePrimitive::Circle,
            1 => crate::ScenePrimitive::Line,
            _ => crate::ScenePrimitive::Rect,
        });
        certificate.include(&primitive.gpu);
        source_error = source_error.max(primitive.source_error);
        primitive.gpu
    }));
    SnapshotMetadata {
        counts,
        certificate,
        quality: super::quality::Quality::from_certificate(source_error, certificate),
    }
}

fn fill_spans(batches: &mut Vec<PreparedDrawBatch>, primitives: &[ScreenPrimitive2d]) {
    batches.clear();
    for (index, primitive) in primitives.iter().enumerate() {
        if let Some(last) = batches.last_mut()
            && last.screen_clip == primitive.clip
        {
            last.vertex_range.end = index as u32 + 1;
        } else {
            batches.push(PreparedDrawBatch {
                vertex_range: index as u32..index as u32 + 1,
                screen_clip: primitive.clip,
            });
        }
    }
}

fn preflight(
    device: &wgpu::Device,
    count: usize,
    capacity: usize,
    spans: usize,
    budget: ScreenPrimitiveBudget,
) -> Result<(), ScreenPrimitiveError> {
    limit("primitives", count, budget.max_primitives)?;
    let capacity = capacity.max(count).max(1);
    if u32::try_from(count).is_err()
        || !buffer_capacity_fits::<ScreenPrimitiveGpu>(device, capacity)
    {
        return Err(ScreenPrimitiveError::DeviceCapacityExceeded);
    }
    limit(
        "retained CPU bytes",
        cpu_bytes(count, spans),
        budget.max_retained_bytes,
    )?;
    limit(
        "GPU buffer bytes",
        capacity.saturating_mul(std::mem::size_of::<ScreenPrimitiveGpu>()),
        budget.max_buffer_bytes,
    )?;
    limit(
        "upload bytes",
        count.saturating_mul(std::mem::size_of::<ScreenPrimitiveGpu>()),
        budget.max_upload_bytes,
    )
}

fn preflight_count(
    device: &wgpu::Device,
    count: usize,
    budget: ScreenPrimitiveBudget,
) -> Result<(), ScreenPrimitiveError> {
    limit("primitives", count, budget.max_primitives)?;
    if u32::try_from(count).is_err()
        || !buffer_capacity_fits::<ScreenPrimitiveGpu>(device, count.max(1))
    {
        return Err(ScreenPrimitiveError::DeviceCapacityExceeded);
    }
    Ok(())
}

fn buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine analytic screen primitive instances"),
        size: (capacity.max(1) * std::mem::size_of::<ScreenPrimitiveGpu>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub(in crate::renderer) fn create(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    primitives: &[ScreenPrimitive2d],
    budget: ScreenPrimitiveBudget,
) -> Result<ScreenPrimitiveBatch2d, ScreenPrimitiveError> {
    let count = primitives.len();
    preflight_count(device, count, budget)?;
    let spans = span_count(primitives);
    preflight(device, count, count, spans, budget)?;
    let mut instances = reserve(count)?;
    let mut draw_batches = reserve(spans)?;
    limit(
        "retained CPU bytes",
        cpu_bytes(instances.capacity(), draw_batches.capacity()),
        budget.max_retained_bytes,
    )?;
    let metadata = fill_instances(&mut instances, primitives);
    fill_spans(&mut draw_batches, primitives);
    let instance_buffer = buffer(device, count);
    if !instances.is_empty() {
        queue.write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&instances));
        submit_pending_uploads(queue);
    }
    Ok(ScreenPrimitiveBatch2d {
        renderer_identity,
        instance_buffer,
        instances,
        draw_batches,
        capacity: count.max(1),
        budget,
        primitive_counts: metadata.counts,
        certificate: metadata.certificate,
        quality: metadata.quality,
        proof_cache: Mutex::new([(None, false); 8]),
    })
}

pub(in crate::renderer) fn update(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    batch: &mut ScreenPrimitiveBatch2d,
    primitives: &[ScreenPrimitive2d],
) -> Result<ScreenPrimitiveUploadReport, ScreenPrimitiveError> {
    let count = primitives.len();
    preflight_count(device, count, batch.budget)?;
    let spans = span_count(primitives);
    preflight(device, count, batch.capacity, spans, batch.budget)?;
    // Growth is staged separately, so even a second allocation failure cannot
    // change the previous snapshot or its retained allocation accounting.
    let replacement_instances = if count > batch.instances.capacity() {
        Some(reserve(count)?)
    } else {
        None
    };
    let replacement_batches = if spans > batch.draw_batches.capacity() {
        Some(reserve(spans)?)
    } else {
        None
    };
    let cpu_capacity = replacement_instances
        .as_ref()
        .map_or(batch.instances.capacity(), Vec::capacity);
    let batch_capacity = replacement_batches
        .as_ref()
        .map_or(batch.draw_batches.capacity(), Vec::capacity);
    limit(
        "retained CPU bytes",
        cpu_bytes(cpu_capacity, batch_capacity),
        batch.budget.max_retained_bytes,
    )?;
    let replacement_buffer = (count > batch.capacity).then(|| buffer(device, count));
    let reused = replacement_buffer.is_none();
    let changed = count != batch.instances.len()
        || batch
            .instances
            .iter()
            .zip(primitives)
            .any(|(old, new)| bytemuck::bytes_of(old) != bytemuck::bytes_of(&new.gpu));
    if let Some(instances) = replacement_instances {
        batch.instances = instances;
    }
    if let Some(batches) = replacement_batches {
        batch.draw_batches = batches;
    }
    if let Some(instance_buffer) = replacement_buffer {
        batch.instance_buffer = instance_buffer;
        batch.capacity = count;
    }
    fill_spans(&mut batch.draw_batches, primitives);
    let quality = if changed {
        let metadata = fill_instances(&mut batch.instances, primitives);
        batch.primitive_counts = metadata.counts;
        batch.certificate = metadata.certificate;
        metadata.quality
    } else {
        let source_error = primitives
            .iter()
            .map(|primitive| primitive.source_error)
            .fold(0.0_f64, f64::max);
        super::quality::Quality::from_certificate(source_error, batch.certificate)
    };
    // Two source snapshots can round to identical GPU instances but have
    // different conversion errors, so refresh this source bound independently.
    let quality_changed = batch.quality != quality;
    batch.quality = quality;
    let uploaded_bytes = if changed && count > 0 {
        queue.write_buffer(
            &batch.instance_buffer,
            0,
            bytemuck::cast_slice(&batch.instances),
        );
        submit_pending_uploads(queue);
        count * std::mem::size_of::<ScreenPrimitiveGpu>()
    } else {
        0
    };
    // Source error can change even when GPU bytes do not. Clip-only and fully
    // identical updates retain their numeric proofs.
    if changed || quality_changed {
        *batch
            .proof_cache
            .get_mut()
            .unwrap_or_else(|error| error.into_inner()) = [(None, false); 8];
    }
    Ok(ScreenPrimitiveUploadReport {
        uploaded_bytes,
        buffer_reused: reused,
    })
}

pub(in crate::renderer) fn restore(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer_identity: Arc<()>,
    source: &ScreenPrimitiveBatch2d,
) -> Result<ScreenPrimitiveBatch2d, ScreenPrimitiveError> {
    preflight(
        device,
        source.instances.len(),
        source.instances.len(),
        source.draw_batches.len(),
        source.budget,
    )?;
    let mut instances = reserve(source.instances.len())?;
    let mut draw_batches = reserve(source.draw_batches.len())?;
    limit(
        "retained CPU bytes",
        cpu_bytes(instances.capacity(), draw_batches.capacity()),
        source.budget.max_retained_bytes,
    )?;
    instances.extend_from_slice(&source.instances);
    draw_batches.extend_from_slice(&source.draw_batches);
    let instance_buffer = buffer(device, instances.len());
    if !instances.is_empty() {
        queue.write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&instances));
        submit_pending_uploads(queue);
    }
    let capacity = instances.len().max(1);
    Ok(ScreenPrimitiveBatch2d {
        renderer_identity,
        instance_buffer,
        instances,
        draw_batches,
        capacity,
        budget: source.budget,
        primitive_counts: source.primitive_counts,
        certificate: source.certificate,
        quality: source.quality,
        proof_cache: Mutex::new([(None, false); 8]),
    })
}
