use super::*;

pub(in crate::renderer) fn batch_retained_bytes(count: usize) -> usize {
    count.saturating_mul(ImageBatchBudget::RETAINED_BYTES_PER_SPRITE)
}

impl WgpuRenderer {
    /// Atomically updates retained sprite descriptions from borrowed input.
    ///
    /// Reuses both CPU arrays and the GPU buffer while capacity permits. Equal
    /// input submits no upload. Shrinking, including to zero sprites, retains
    /// capacity. Growth reserves exact bounded replacement arrays before changing
    /// drawable state. Synchronous validation/allocation errors preserve the old
    /// batch. Device failure is not a rollback of already submitted GPU work.
    ///
    /// The original image and current renderer generation must match. The queue
    /// orders uploads after previously submitted draws; the frame borrow prevents
    /// updating a batch still referenced by an unsubmitted composer.
    pub fn update_image_batch(
        &self,
        image: &Image2d,
        batch: &mut ImageBatch2d,
        sprites: &[ImageSprite2d],
    ) -> Result<ImageBatchUploadReport, ImageError> {
        self.validate_image_batch(image, batch)?;
        update_image_batch_resources(
            &self.device,
            &self.queue,
            image,
            batch,
            sprites.iter().copied(),
            sprites.len(),
            batch.budget,
        )
    }
}

fn replacement_array<T: Copy>(
    current_capacity: usize,
    count: usize,
    compact: bool,
) -> Result<Option<Vec<T>>, ImageError> {
    if count <= current_capacity && !compact {
        return Ok(None);
    }
    let mut replacement = Vec::new();
    replacement
        .try_reserve_exact(count)
        .map_err(|_| ImageError::AllocationFailed {
            requested_bytes: count.saturating_mul(std::mem::size_of::<T>()),
        })?;
    Ok(Some(replacement))
}

fn instance_for(image: &Image2d, sprite: ImageSprite2d) -> ImageInstance {
    let origin = sprite.destination.origin().to_vec2();
    let viewport = sprite.destination.viewport();
    let source = sprite.source;
    ImageInstance {
        destination: [origin.x, origin.y, viewport.width(), viewport.height()],
        uv_rect: [
            (source.x as f32 + 0.5) / image.width as f32,
            (source.y as f32 + 0.5) / image.height as f32,
            (source.x as f32 + source.width as f32 - 0.5) / image.width as f32,
            (source.y as f32 + source.height as f32 - 0.5) / image.height as f32,
        ],
        tint: sprite.tint.to_array(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::renderer) fn update_image_batch_resources<I>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &Image2d,
    batch: &mut ImageBatch2d,
    sprites: I,
    count: usize,
    budget: ImageBatchBudget,
) -> Result<ImageBatchUploadReport, ImageError>
where
    I: Iterator<Item = ImageSprite2d> + Clone,
{
    if !Arc::ptr_eq(&image.renderer_identity, &batch.renderer_identity)
        || !Arc::ptr_eq(&image.resource_identity, &batch.image_identity)
    {
        return Err(ImageError::RendererMismatch);
    }
    preflight_image_batch_capacity(device, count, budget)?;
    if sprites.clone().count() != count {
        return Err(ImageError::InvalidSprite);
    }
    let mut destinations_unchanged = count == batch.sprites.len();
    for (index, sprite) in sprites.clone().enumerate() {
        if !sprite.source.fits(image.width, image.height)
            || !sprite.tint.is_normalized()
            || !logical_image_region_is_portable(sprite.destination)
        {
            return Err(ImageError::InvalidSprite);
        }
        destinations_unchanged = destinations_unchanged
            && batch.sprites.get(index).is_some_and(|previous| {
                proof::destination_bits(*previous) == proof::destination_bits(sprite)
            });
    }

    let old_cpu = batch.recovery_memory_bytes();
    let old_gpu = batch.gpu_allocation_bytes();
    if batch.sprites.iter().copied().eq(sprites.clone()) && old_cpu <= budget.max_retained_bytes {
        batch.budget = budget;
        return Ok(report(batch, 0, false, old_cpu, old_gpu));
    }
    let compact = old_cpu > budget.max_retained_bytes;
    let new_sprites: Option<Vec<ImageSprite2d>> =
        replacement_array(batch.sprites.capacity(), count, compact)?;
    let new_instances: Option<Vec<ImageInstance>> =
        replacement_array(batch.instances.capacity(), count, compact)?;
    let sprite_capacity = new_sprites
        .as_ref()
        .map_or(batch.sprites.capacity(), Vec::capacity);
    let instance_capacity = new_instances
        .as_ref()
        .map_or(batch.instances.capacity(), Vec::capacity);
    let cpu_bytes = sprite_capacity
        .saturating_mul(std::mem::size_of::<ImageSprite2d>())
        .saturating_add(instance_capacity.saturating_mul(std::mem::size_of::<ImageInstance>()));
    if cpu_bytes > budget.max_retained_bytes {
        return Err(ImageError::BatchBudgetExceeded {
            limit: budget.max_retained_bytes,
            actual: cpu_bytes,
        });
    }
    let peak_cpu = old_cpu
        .saturating_add(new_sprites.as_ref().map_or(0, |v| {
            v.capacity()
                .saturating_mul(std::mem::size_of::<ImageSprite2d>())
        }))
        .saturating_add(new_instances.as_ref().map_or(0, |v| {
            v.capacity()
                .saturating_mul(std::mem::size_of::<ImageInstance>())
        }));
    let gpu_bytes = count.max(1) * std::mem::size_of::<ImageInstance>();
    let replacement = (gpu_bytes > old_gpu).then(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim-engine grown image sprite instances"),
            size: gpu_bytes as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    let replaced = replacement.is_some();
    let peak_gpu = old_gpu.saturating_add(if replaced { gpu_bytes } else { 0 });
    // No remaining synchronous rejection precedes publishing the new arrays.
    // UV/tint never participate in the geometry proof. Keep that exact result
    // when every ordered destination component is bit-identical; any changed
    // position, size or count invalidates before the new arrays are published.
    if !destinations_unchanged {
        batch.geometry_proof.invalidate();
    }
    if let Some(values) = new_sprites {
        batch.sprites = values;
    }
    if let Some(values) = new_instances {
        batch.instances = values;
    }
    if let Some(buffer) = replacement {
        batch.instance_buffer = buffer;
    }
    batch.sprites.clear();
    batch.instances.clear();
    for sprite in sprites {
        batch.instances.push(instance_for(image, sprite));
        batch.sprites.push(sprite);
    }
    batch.budget = budget;
    let uploaded = count * std::mem::size_of::<ImageInstance>();
    if uploaded > 0 {
        queue.write_buffer(
            &batch.instance_buffer,
            0,
            bytemuck::cast_slice(&batch.instances),
        );
        submit_pending_uploads(queue);
    }
    Ok(report(batch, uploaded, replaced, peak_cpu, peak_gpu))
}

fn report(
    batch: &ImageBatch2d,
    uploaded: usize,
    replaced: bool,
    peak_cpu: usize,
    peak_gpu: usize,
) -> ImageBatchUploadReport {
    ImageBatchUploadReport {
        uploaded_instance_bytes: uploaded,
        replaced_instance_buffer: replaced,
        sprite_count: batch.sprite_count(),
        retained_capacity: batch.capacity(),
        gpu_capacity: batch.gpu_allocation_bytes() / std::mem::size_of::<ImageInstance>(),
        retained_bytes: batch.recovery_memory_bytes(),
        peak_retained_bytes: peak_cpu,
        peak_gpu_bytes: peak_gpu,
    }
}
