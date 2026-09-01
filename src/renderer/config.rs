use super::*;
use std::fmt;

const DEFAULT_MAX_QUARANTINED_DEVICES: usize = 4;
const MAX_QUARANTINED_DEVICES: usize = 8;

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

/// Concrete presentation mode selected from the active Linux surface.
///
/// This is deliberately separate from [`RendererPresentMode`]: `NoVsync` is a
/// preference with fallbacks, while this type reports the mode actually passed
/// to `wgpu` after inspecting the surface capabilities. A compositor may still
/// pace redraw callbacks or scanout independently of this selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererSurfacePresentMode {
    /// Immediate replacement without a presentation queue or refresh pacing.
    Immediate,
    /// Single-slot replacement queue synchronized to display refresh.
    Mailbox,
    /// Strict first-in-first-out queue synchronized to display refresh.
    Fifo,
    /// FIFO that may tear when a frame misses the refresh interval.
    FifoRelaxed,
}

impl RendererSurfacePresentMode {
    pub(super) const fn to_wgpu(self) -> wgpu::PresentMode {
        match self {
            Self::Immediate => wgpu::PresentMode::Immediate,
            Self::Mailbox => wgpu::PresentMode::Mailbox,
            Self::Fifo => wgpu::PresentMode::Fifo,
            Self::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
        }
    }

    /// Returns whether this mode presents on display refresh boundaries.
    pub const fn is_refresh_synchronized(self) -> bool {
        !matches!(self, Self::Immediate)
    }
}

impl fmt::Display for RendererSurfacePresentMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Immediate => formatter.write_str("Immediate"),
            Self::Mailbox => formatter.write_str("Mailbox"),
            Self::Fifo => formatter.write_str("FIFO"),
            Self::FifoRelaxed => formatter.write_str("FIFO relaxed"),
        }
    }
}

pub(super) fn select_surface_present_mode(
    requested: RendererPresentMode,
    supported: &[wgpu::PresentMode],
) -> RendererSurfacePresentMode {
    match requested {
        RendererPresentMode::Vsync => RendererSurfacePresentMode::Fifo,
        RendererPresentMode::NoVsync => [
            RendererSurfacePresentMode::Immediate,
            RendererSurfacePresentMode::Mailbox,
            RendererSurfacePresentMode::Fifo,
        ]
        .into_iter()
        .find(|candidate| supported.contains(&candidate.to_wgpu()))
        .unwrap_or(RendererSurfacePresentMode::Fifo),
    }
}

/// Options used when creating a [`WgpuRenderer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WgpuRendererOptions {
    present_mode: RendererPresentMode,
    scale_factor: f64,
    max_quarantined_devices: usize,
}

impl WgpuRendererOptions {
    /// Builds options from presentation behavior and display scale.
    ///
    /// `scale_factor` is physical surface pixels per logical screen pixel. It
    /// must be finite, representable as `f32`, and bounded so every non-empty
    /// `u32` surface keeps its logical dimensions and reciprocal clip scale in
    /// the normal finite `f32` range.
    pub fn new(
        present_mode: RendererPresentMode,
        scale_factor: f64,
    ) -> Result<Self, RendererConfigurationError> {
        validate_scale_factor(scale_factor)?;
        Ok(Self {
            present_mode,
            scale_factor,
            max_quarantined_devices: DEFAULT_MAX_QUARANTINED_DEVICES,
        })
    }

    /// Sets the bounded number of previous logical devices retained after recovery.
    ///
    /// The default is 4 and the supported range is `1..=8`. Retention avoids a
    /// native Linux driver crash observed when a healthy device is destroyed
    /// immediately after surface migration. Once the limit is reached, another
    /// recovery returns [`RendererInitError::RecoveryLimitReached`] before
    /// requesting or installing a replacement device.
    pub fn with_max_quarantined_devices(
        mut self,
        limit: usize,
    ) -> Result<Self, RendererConfigurationError> {
        if !(1..=MAX_QUARANTINED_DEVICES).contains(&limit) {
            return Err(RendererConfigurationError::InvalidRecoveryLimit { limit });
        }
        self.max_quarantined_devices = limit;
        Ok(self)
    }

    /// Returns presentation behavior for the surface.
    pub fn present_mode(self) -> RendererPresentMode {
        self.present_mode
    }

    /// Returns physical surface pixels per logical screen pixel.
    pub fn scale_factor(self) -> f64 {
        self.scale_factor
    }

    /// Returns the maximum previous logical devices retained after recovery.
    pub fn max_quarantined_devices(self) -> usize {
        self.max_quarantined_devices
    }
}

impl Default for WgpuRendererOptions {
    fn default() -> Self {
        Self {
            present_mode: RendererPresentMode::Vsync,
            scale_factor: 1.0,
            max_quarantined_devices: DEFAULT_MAX_QUARANTINED_DEVICES,
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
    if scale_factor_f32.is_finite()
        && crate::units::stable_physical_per_logical(scale_factor)
        && crate::units::stable_physical_per_logical(f64::from(scale_factor_f32))
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
    /// Replaced logical devices enter a bounded quarantine until this renderer
    /// is dropped. Some native swapchain drivers crash if a healthy previous
    /// device is destroyed immediately after the surface migrates to its
    /// replacement. Device recovery should therefore remain exceptional rather
    /// than a normal quality switch. Query
    /// [`WgpuRenderer::remaining_device_recoveries`] before attempting it.
    pub async fn recover_device_and_surface(&mut self) -> Result<(), RendererInitError> {
        if !recovery_quarantine_has_capacity(
            self.retired_devices.len(),
            self.max_quarantined_devices,
        ) {
            return Err(RendererInitError::RecoveryLimitReached {
                limit: self.max_quarantined_devices,
            });
        }
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
        let adapter_info = adapter.get_info();
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
        let surface_capabilities = self.surface.get_capabilities(&adapter);
        let surface_present_mode = select_surface_present_mode(
            self.requested_present_mode,
            &surface_capabilities.present_modes,
        );
        config.present_mode = surface_present_mode.to_wgpu();
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
        let particle_unit_buffer = create_particle_unit_buffer(&device, &queue);
        let multisample_target = create_multisample_target(&device, &config, sample_count);
        let image_renderer = ImageRenderer::new(&device, config.format, sample_count);
        let mesh3d_renderer = Mesh3dRenderer::new(&device, config.format);
        self.surface.configure(&device, &config);

        self.renderer_identity = Arc::new(());
        let retired_device = RetiredDevice {
            _adapter: std::mem::replace(&mut self._adapter, adapter),
            _device: std::mem::replace(&mut self.device, device),
            _queue: std::mem::replace(&mut self.queue, queue),
        };
        self.retired_devices.push(retired_device);
        self.adapter_info = adapter_info;
        self.config = config;
        self.surface_present_mode = surface_present_mode;
        self.pipeline = pipeline;
        self.target_pipeline = target_pipeline;
        self.dynamic_pipeline = dynamic_pipeline;
        self.particle_pipeline = particle_pipeline;
        self.target_particle_pipeline = target_particle_pipeline;
        self.heatmap_pipeline = heatmap_pipeline;
        self.target_heatmap_pipeline = target_heatmap_pipeline;
        self.composition_pipelines = composition_pipelines;
        self.target_composition_pipelines = target_composition_pipelines;
        self.image_renderer = image_renderer;
        self.mesh3d_renderer = mesh3d_renderer;
        self.camera_uniform_buffer = camera_uniform_buffer;
        self.camera_bind_group = camera_bind_group;
        self.camera_bind_group_layout = camera_bind_group_layout;
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

pub(super) fn recovery_quarantine_has_capacity(retired: usize, limit: usize) -> bool {
    retired < limit
}
