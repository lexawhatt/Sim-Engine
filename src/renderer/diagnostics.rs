//! Borrowed, allocation-free context for correlating host and renderer measurements.

use super::{RendererSurfacePresentMode, WgpuRenderer};

/// Adapter and presentation context captured from the active renderer.
///
/// This view does not allocate, query the GPU, or wait for completion. A host
/// should record it with performance reports instead of guessing the selected
/// adapter from the machine's hardware. Borrowing prevents resize or device
/// replacement while the view is in use; obtain a new view after either event.
/// The strings describe driver-reported identity, not an independent hardware
/// attestation. An empty PCI address means instance identity is unavailable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RendererDiagnostics<'renderer> {
    adapter_name: &'renderer str,
    backend: &'static str,
    vendor_id: u32,
    device_id: u32,
    pci_bus_id: &'renderer str,
    driver: &'renderer str,
    driver_info: &'renderer str,
    scale_factor: f64,
    physical_size: [u32; 2],
    surface_format: wgpu::TextureFormat,
    sample_count: u32,
    present_mode: RendererSurfacePresentMode,
}

impl RendererDiagnostics<'_> {
    /// Human-readable name of the adapter actually selected by the renderer.
    pub fn adapter_name(&self) -> &str {
        self.adapter_name
    }

    /// Stable lowercase graphics API name, such as `vulkan`.
    pub const fn backend(&self) -> &'static str {
        self.backend
    }

    /// Driver-reported PCI vendor ID; together with device ID identifies a model.
    pub const fn vendor_id(&self) -> u32 {
        self.vendor_id
    }

    /// Driver-reported device model ID, not a physical-instance identifier.
    pub const fn device_id(&self) -> u32 {
        self.device_id
    }

    /// PCI address of the selected physical GPU, or empty when unavailable.
    pub fn pci_bus_id(&self) -> &str {
        self.pci_bus_id
    }

    /// Driver name reported by the active adapter.
    pub fn driver(&self) -> &str {
        self.driver
    }

    /// Driver version or implementation details reported by the adapter.
    pub fn driver_info(&self) -> &str {
        self.driver_info
    }

    /// Physical surface pixels per logical screen pixel at capture time.
    pub const fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Configured surface width and height in physical pixels.
    pub const fn physical_size(&self) -> [u32; 2] {
        self.physical_size
    }

    /// Actual presentation texture format, not an assumed offscreen format.
    pub const fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    /// Raster sample count selected for the actual presentation format.
    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Actual presentation mode after capability-dependent fallback.
    pub const fn present_mode(&self) -> RendererSurfacePresentMode {
        self.present_mode
    }
}

impl WgpuRenderer {
    /// Returns allocation-free adapter, driver, DPI, and presentation metadata.
    ///
    /// Record this alongside CPU preparation/upload timings and separately
    /// correlated asynchronous GPU timestamp samples. Metadata does not turn
    /// surface acquisition or a CPU interval into a GPU execution measurement.
    pub fn diagnostics(&self) -> RendererDiagnostics<'_> {
        RendererDiagnostics {
            adapter_name: self.adapter_name(),
            backend: self.adapter_backend(),
            vendor_id: self.adapter_vendor_id(),
            device_id: self.adapter_device_id(),
            pci_bus_id: self.adapter_pci_bus_id(),
            driver: self.adapter_driver(),
            driver_info: self.adapter_driver_info(),
            scale_factor: self.scale_factor,
            physical_size: [self.config.width, self.config.height],
            surface_format: self.config.format,
            sample_count: self.sample_count,
            present_mode: self.surface_present_mode,
        }
    }
}
