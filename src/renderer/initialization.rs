//! Renderer initialization, surface sizing and host-facing diagnostics.

use super::*;

impl WgpuRenderer {
    /// Creates a renderer for a window or canvas surface target.
    ///
    /// `width` and `height` are physical surface pixels. Zero sizes are clamped
    /// to one pixel because `wgpu` surfaces cannot be configured at zero size.
    /// This convenience constructor assumes a display scale factor of `1.0`;
    /// HiDPI hosts should use [`WgpuRenderer::new_with_options`].
    pub async fn new(
        surface_target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Self, RendererInitError> {
        Self::new_with_options(
            surface_target,
            width,
            height,
            WgpuRendererOptions::default(),
        )
        .await
    }

    /// Creates a renderer with explicit options.
    ///
    /// Use this when measuring renderer throughput or when the host application
    /// needs control over presentation behavior or display scale.
    pub async fn new_with_options(
        surface_target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
        options: WgpuRendererOptions,
    ) -> Result<Self, RendererInitError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(surface_target)
            .map_err(RendererInitError::CreateSurface)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .map_err(RendererInitError::RequestAdapter)?;
        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-engine device"),
                required_features: gpu_timing::requested_features(
                    adapter.features(),
                    options.gpu_timing(),
                ),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(RendererInitError::RequestDevice)?;

        let width = width.max(1);
        let height = height.max(1);
        let dimension_limit = device.limits().max_texture_dimension_2d;
        if width > dimension_limit || height > dimension_limit {
            return Err(RendererInitError::SurfaceDimensionsTooLarge {
                width,
                height,
                limit: dimension_limit,
            });
        }
        let surface_capabilities = surface.get_capabilities(&adapter);
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(RendererInitError::NoSurfaceConfig)?;
        let surface_present_mode = select_surface_present_mode(
            options.present_mode(),
            &surface_capabilities.present_modes,
        );
        config.present_mode = surface_present_mode.to_wgpu();
        surface.configure(&device, &config);

        let sample_count = preferred_sample_count(&adapter, config.format);
        let PipelineResources {
            pipeline,
            target_pipeline,
            dynamic_pipeline,
            particle_pipeline,
            target_particle_pipeline,
            heatmap_pipeline,
            target_heatmap_pipeline,
            composition_pipelines,
            target_composition_pipelines,
            camera_uniform_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            heatmap_uniform_buffer,
            heatmap_bind_group_layout,
        } = create_pipeline(&device, config.format, sample_count);
        let vertex_buffer = Arc::new(create_vertex_buffer(&device, INITIAL_VERTEX_CAPACITY));
        let particle_unit_buffer = create_submitted_particle_unit_buffer(&device, &queue);
        let multisample_target = create_multisample_target(&device, &config, sample_count);
        let image_renderer = ImageRenderer::new(&device, config.format, sample_count);
        let mesh3d_renderer = Mesh3dRenderer::new(&device, config.format);
        let gpu_timing = gpu_timing::GpuTimingCollector::new(&device, &queue, options.gpu_timing());

        Ok(Self {
            renderer_identity: Arc::new(()),
            _instance: instance,
            surface,
            _adapter: adapter,
            adapter_info,
            device,
            queue,
            config,
            requested_present_mode: options.present_mode(),
            surface_present_mode,
            pre_present_notify: None,
            scale_factor: options.scale_factor(),
            pipeline,
            target_pipeline,
            dynamic_pipeline,
            particle_pipeline,
            target_particle_pipeline,
            heatmap_pipeline,
            target_heatmap_pipeline,
            composition_pipelines,
            target_composition_pipelines,
            image_renderer,
            mesh3d_renderer,
            gpu_timing,
            gpu_timing_requested: options.gpu_timing(),
            camera_uniform_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            heatmap_uniform_buffer,
            heatmap_bind_group_layout,
            color_map_cache: None,
            frame_cache: frame::FrameCache::default(),
            vertex_buffer,
            particle_unit_buffer,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
            multisample_target,
            sample_count,
            vertices: Vec::new(),
            draw_batches: Vec::new(),
            retired_devices: Vec::new(),
            max_quarantined_devices: options.max_quarantined_devices(),
        })
    }

    /// Returns the configured surface size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Returns optional GPU-query availability, bounded resources and losses.
    /// These diagnostics are independent of frame CPU timings and scene uploads.
    pub fn gpu_timing_statistics(&self) -> GpuTimingStatistics {
        self.gpu_timing.statistics()
    }

    /// Collects at most eight completed GPU pass intervals without waiting.
    ///
    /// Call regularly when timing is enabled; uncollected samples occupy the
    /// fixed ring and eventually cause new samples to be dropped. Match sample
    /// IDs to submitted reports when excluding warmup or unrelated work. A call
    /// performs one nonblocking device poll and reports its CPU overhead. Device
    /// recovery discards pending old-device samples and resets the counters.
    pub fn collect_gpu_timings(&mut self) -> GpuTimingBatch {
        self.gpu_timing.collect(&self.device)
    }

    /// Returns the render viewport size in logical screen pixels.
    ///
    /// Scene clipping, stroke widths, shadows, and camera zoom use this logical
    /// coordinate space so their visual size remains stable across display DPI.
    pub fn logical_size(&self) -> (f32, f32) {
        let scale_factor = self.scale_factor as f32;
        (
            self.config.width as f32 / scale_factor,
            self.config.height as f32 / scale_factor,
        )
    }

    /// Returns the current camera viewport in logical screen pixels.
    pub fn logical_viewport(&self) -> Result<LogicalViewport, crate::LogicalViewportError> {
        let (width, height) = self.logical_size();
        LogicalViewport::new(width, height)
    }

    /// Converts a physical surface position into logical screen pixels.
    pub fn physical_to_logical_screen(
        &self,
        position: PhysicalScreenPosition,
    ) -> Result<LogicalScreenPosition, RendererCoordinateError> {
        physical_to_logical_screen(position, self.scale_factor as f32)
    }

    /// Converts a logical screen position into physical surface pixels.
    pub fn logical_to_physical_screen(
        &self,
        position: LogicalScreenPosition,
    ) -> Result<PhysicalScreenPosition, RendererCoordinateError> {
        logical_to_physical_screen(position, self.scale_factor as f32)
    }

    /// Returns physical surface pixels per logical screen pixel.
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Returns the concrete presentation mode selected for the active surface.
    ///
    /// `RendererPresentMode::NoVsync` prefers Immediate, then Mailbox, then
    /// FIFO. This method reveals which supported fallback was configured. The
    /// desktop compositor may still pace host redraw callbacks independently.
    pub fn surface_present_mode(&self) -> RendererSurfacePresentMode {
        self.surface_present_mode
    }

    /// Installs a host callback invoked immediately before synchronized
    /// surface presentation.
    ///
    /// A winit host should capture its `Arc<Window>` and call
    /// `Window::pre_present_notify` from this callback. The renderer invokes
    /// it after command submission and immediately before `Queue::present`,
    /// but only when the concrete surface mode is Mailbox, FIFO, or
    /// FIFO-relaxed. Immediate presentation deliberately remains uncapped.
    pub fn set_pre_present_notify(&mut self, notify: impl Fn() + Send + Sync + 'static) {
        self.pre_present_notify = Some(Arc::new(notify));
    }

    /// Removes the synchronized-presentation callback, if one is installed.
    pub fn clear_pre_present_notify(&mut self) {
        self.pre_present_notify = None;
    }

    pub(super) fn notify_before_present(&self) {
        invoke_pre_present_notify(
            self.surface_present_mode,
            self.pre_present_notify.as_deref(),
        );
    }

    /// Returns the active graphics adapter's human-readable name.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Returns the active graphics API backend as a stable lowercase name.
    pub fn adapter_backend(&self) -> &'static str {
        self.adapter_info.backend.to_str()
    }

    /// Returns the PCI vendor identifier reported for the active adapter.
    pub fn adapter_vendor_id(&self) -> u32 {
        self.adapter_info.vendor
    }

    /// Returns the device identifier reported for the active adapter.
    pub fn adapter_device_id(&self) -> u32 {
        self.adapter_info.device
    }

    /// Returns the backend-reported PCI bus address for the active adapter.
    ///
    /// Vulkan adapters expose this as `domain:bus:device.function` when the
    /// driver supports `VK_EXT_pci_bus_info`. An empty string means that the
    /// backend cannot identify a physical adapter instance; release evidence
    /// must not treat model identifiers alone as instance identity.
    pub fn adapter_pci_bus_id(&self) -> &str {
        &self.adapter_info.device_pci_bus_id
    }

    /// Returns the texture format selected for the active presentation surface.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Returns the raster sample count selected for the active surface format.
    pub fn surface_sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Returns the driver name reported by the active graphics adapter.
    pub fn adapter_driver(&self) -> &str {
        &self.adapter_info.driver
    }

    /// Returns the driver version or implementation detail reported by the adapter.
    pub fn adapter_driver_info(&self) -> &str {
        &self.adapter_info.driver_info
    }

    /// Returns previous logical devices retained for safe native-driver teardown.
    pub fn quarantined_device_count(&self) -> usize {
        self.retired_devices.len()
    }

    /// Returns the configured maximum previous-device quarantine size.
    pub fn max_quarantined_device_count(&self) -> usize {
        self.max_quarantined_devices
    }

    /// Returns successful recoveries available before the bounded quarantine is full.
    pub fn remaining_device_recoveries(&self) -> usize {
        self.max_quarantined_devices
            .saturating_sub(self.retired_devices.len())
    }

    /// Blocks until all previously submitted GPU work has completed.
    ///
    /// This is intended for diagnostic throughput measurements, readback, and
    /// controlled resource teardown. Calling it in an interactive frame loop
    /// defeats normal CPU/GPU pipelining.
    pub fn wait_for_gpu_idle(&self) -> Result<(), wgpu::PollError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
    }

    /// Replaces display scale without changing physical surface dimensions.
    ///
    /// Invalid values return an error and leave the current scale unchanged.
    pub fn set_scale_factor(
        &mut self,
        scale_factor: f64,
    ) -> Result<(), RendererConfigurationError> {
        validate_scale_factor(scale_factor)?;
        self.scale_factor = scale_factor;
        Ok(())
    }

    /// Reconfigures the surface after a host window resize.
    ///
    /// Zero width or height is ignored because minimized windows often report
    /// zero size and `wgpu` cannot configure a zero-sized surface.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RendererConfigurationError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        validate_surface_dimensions(width, height, self.device.limits().max_texture_dimension_2d)?;

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.multisample_target =
            create_multisample_target(&self.device, &self.config, self.sample_count);
        Ok(())
    }

    /// Reconfigures physical surface size and logical-to-physical display scale.
    ///
    /// Invalid scale returns an error without changing either value. Zero
    /// physical dimensions update the scale but leave the minimized surface
    /// configuration unchanged.
    pub fn resize_with_scale_factor(
        &mut self,
        width: u32,
        height: u32,
        scale_factor: f64,
    ) -> Result<(), RendererConfigurationError> {
        validate_scale_factor(scale_factor)?;
        if width != 0 && height != 0 {
            validate_surface_dimensions(
                width,
                height,
                self.device.limits().max_texture_dimension_2d,
            )?;
        }
        self.resize(width, height)?;
        self.scale_factor = scale_factor;
        Ok(())
    }
}
