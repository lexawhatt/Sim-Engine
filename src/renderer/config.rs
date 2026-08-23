use super::*;

pub(super) struct MultisampleTarget {
    pub(super) _texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
}

/// Presentation behavior for the window surface.
///
/// `Vsync` requests strict FIFO presentation for stable frame pacing and no
/// tearing. `NoVsync` requests the fastest available non-VSync mode, but may
/// fall back to FIFO when the platform exposes no Immediate or Mailbox mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererPresentMode {
    /// Use strict FIFO presentation synchronized to display refresh.
    Vsync,
    /// Prefer Immediate, then Mailbox, and fall back to FIFO when necessary.
    NoVsync,
}

impl RendererPresentMode {
    pub(super) fn to_wgpu(self) -> wgpu::PresentMode {
        match self {
            Self::Vsync => wgpu::PresentMode::Fifo,
            Self::NoVsync => wgpu::PresentMode::AutoNoVsync,
        }
    }
}

/// Options used when creating a [`WgpuRenderer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WgpuRendererOptions {
    present_mode: RendererPresentMode,
    scale_factor: f64,
}

impl WgpuRendererOptions {
    /// Builds options from presentation behavior and display scale.
    ///
    /// `scale_factor` is physical surface pixels per logical screen pixel. It
    /// must be finite, positive, representable as `f32`, and large enough that
    /// every supported physical surface has finite logical dimensions.
    pub fn new(
        present_mode: RendererPresentMode,
        scale_factor: f64,
    ) -> Result<Self, RendererConfigurationError> {
        validate_scale_factor(scale_factor)?;
        Ok(Self {
            present_mode,
            scale_factor,
        })
    }

    /// Returns presentation behavior for the surface.
    pub fn present_mode(self) -> RendererPresentMode {
        self.present_mode
    }

    /// Returns physical surface pixels per logical screen pixel.
    pub fn scale_factor(self) -> f64 {
        self.scale_factor
    }
}

impl Default for WgpuRendererOptions {
    fn default() -> Self {
        Self {
            present_mode: RendererPresentMode::Vsync,
            scale_factor: 1.0,
        }
    }
}

pub(super) fn create_multisample_target(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
) -> Option<MultisampleTarget> {
    if sample_count <= 1 {
        return None;
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-engine multisample target"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    Some(MultisampleTarget {
        _texture: texture,
        view,
    })
}

pub(super) fn validate_scale_factor(scale_factor: f64) -> Result<(), RendererConfigurationError> {
    let scale_factor_f32 = scale_factor as f32;
    let minimum_scale_factor = u32::MAX as f64 / f32::MAX as f64;
    if scale_factor.is_finite()
        && scale_factor_f32.is_finite()
        && scale_factor >= minimum_scale_factor
    {
        Ok(())
    } else {
        Err(RendererConfigurationError::InvalidScaleFactor { scale_factor })
    }
}

pub(super) fn physical_to_logical_screen(
    position: PhysicalScreenPosition,
    scale_factor: f32,
) -> Result<LogicalScreenPosition, RendererCoordinateError> {
    let logical = position.to_vec2() / scale_factor;
    (position.is_finite() && logical.is_finite())
        .then_some(LogicalScreenPosition::from_vec2(logical))
        .ok_or(RendererCoordinateError::NonFiniteConversion)
}

pub(super) fn logical_to_physical_screen(
    position: LogicalScreenPosition,
    scale_factor: f32,
) -> Result<PhysicalScreenPosition, RendererCoordinateError> {
    let physical = position.to_vec2() * scale_factor;
    (position.is_finite() && physical.is_finite())
        .then_some(PhysicalScreenPosition::from_vec2(physical))
        .ok_or(RendererCoordinateError::NonFiniteConversion)
}

impl WgpuRenderer {
    /// Requests a replacement logical device for the existing surface and
    /// recreates all renderer-owned transient pipelines and buffers.
    ///
    /// External prepared, dynamic, particle, scalar, target, and trail
    /// resources deliberately retain the previous renderer identity. Recreate
    /// them with the matching `restore_*` methods after this call succeeds.
    /// Targets and trails restore empty and must be redrawn.
    ///
    /// Replaced logical devices remain retained until this renderer is dropped.
    /// Some native swapchain drivers crash if a healthy previous device is
    /// destroyed immediately after the surface migrates to its replacement.
    /// Device recovery should therefore remain exceptional rather than a normal
    /// quality-switching mechanism.
    pub async fn recover_device_and_surface(&mut self) -> Result<(), RendererInitError> {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let adapter = self
            ._instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&self.surface),
                apply_limit_buckets: false,
            })
            .await
            .map_err(RendererInitError::RequestAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-engine recovered device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(RendererInitError::RequestDevice)?;
        let mut config = self
            .surface
            .get_default_config(&adapter, self.config.width, self.config.height)
            .ok_or(RendererInitError::NoSurfaceConfig)?;
        config.present_mode = self.config.present_mode;
        let sample_count = preferred_sample_count(&adapter, config.format);
        let (
            pipeline,
            dynamic_pipeline,
            particle_pipeline,
            target_particle_pipeline,
            heatmap_pipeline,
            target_heatmap_pipeline,
            composition_pipelines,
            target_composition_pipelines,
            camera_uniform_buffer,
            camera_bind_group,
            heatmap_uniform_buffer,
            heatmap_bind_group_layout,
        ) = create_pipeline(&device, config.format, sample_count);
        let vertex_buffer = Arc::new(create_vertex_buffer(&device, INITIAL_VERTEX_CAPACITY));
        let particle_unit_buffer = create_particle_unit_buffer(&device, &queue);
        let multisample_target = create_multisample_target(&device, &config, sample_count);
        self.surface.configure(&device, &config);

        self.renderer_identity = Arc::new(());
        let retired_device = RetiredDevice {
            _adapter: std::mem::replace(&mut self._adapter, adapter),
            _device: std::mem::replace(&mut self.device, device),
            _queue: std::mem::replace(&mut self.queue, queue),
        };
        self.retired_devices.push(retired_device);
        self.config = config;
        self.pipeline = pipeline;
        self.dynamic_pipeline = dynamic_pipeline;
        self.particle_pipeline = particle_pipeline;
        self.target_particle_pipeline = target_particle_pipeline;
        self.heatmap_pipeline = heatmap_pipeline;
        self.target_heatmap_pipeline = target_heatmap_pipeline;
        self.composition_pipelines = composition_pipelines;
        self.target_composition_pipelines = target_composition_pipelines;
        self.camera_uniform_buffer = camera_uniform_buffer;
        self.camera_bind_group = camera_bind_group;
        self.heatmap_uniform_buffer = heatmap_uniform_buffer;
        self.heatmap_bind_group_layout = heatmap_bind_group_layout;
        self.color_map_cache = None;
        self.vertex_buffer = vertex_buffer;
        self.particle_unit_buffer = particle_unit_buffer;
        self.vertex_capacity = INITIAL_VERTEX_CAPACITY;
        self.multisample_target = multisample_target;
        self.sample_count = sample_count;
        self.vertices.clear();
        self.draw_batches.clear();
        Ok(())
    }
}
