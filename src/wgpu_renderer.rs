use std::{borrow::Cow, error::Error, fmt};

use crate::{
    Camera2d, Circle, Color, DrawCommand, Line, Palette, Polyline, Rect, Scene, Shadow, ShapeStyle,
    Stroke, Vec2, Viewport,
};

const INITIAL_VERTEX_CAPACITY: usize = 4096;
const CIRCLE_SEGMENTS: usize = 64;
const CORNER_SEGMENTS: usize = 12;
const PREFERRED_SAMPLE_COUNT: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];

    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };
}

/// Result of attempting to draw a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStatus {
    /// Commands were submitted and the frame was presented.
    Drawn,
    /// The frame was skipped because the window surface was temporarily unavailable.
    Skipped(RendererSurfaceStatus),
}

/// Recoverable or fatal state reported by the presentation surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererSurfaceStatus {
    /// Acquiring the next frame timed out; try again on a later frame.
    Timeout,
    /// The surface is occluded or minimized; rendering can be skipped.
    Occluded,
    /// The surface configuration is stale; resizing or reconfiguring is needed.
    Outdated,
    /// The surface was lost and should be recreated by the host application.
    Lost,
    /// `wgpu` reported a validation error while acquiring the frame.
    Validation,
}

/// Errors that can happen while creating the `wgpu` renderer.
#[derive(Debug)]
pub enum RendererInitError {
    /// Surface creation failed for the provided window or canvas target.
    CreateSurface(wgpu::CreateSurfaceError),
    /// No compatible GPU adapter could be selected.
    RequestAdapter(wgpu::RequestAdapterError),
    /// A logical GPU device and queue could not be created.
    RequestDevice(wgpu::RequestDeviceError),
    /// The surface did not expose a usable default configuration.
    NoSurfaceConfig,
}

impl fmt::Display for RendererInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateSurface(error) => write!(formatter, "failed to create surface: {error}"),
            Self::RequestAdapter(error) => write!(formatter, "failed to request adapter: {error}"),
            Self::RequestDevice(error) => write!(formatter, "failed to request device: {error}"),
            Self::NoSurfaceConfig => write!(formatter, "surface has no supported default config"),
        }
    }
}

impl Error for RendererInitError {}

struct MultisampleTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// `wgpu` backend that renders [`Scene`] commands into a presentation surface.
///
/// The renderer owns the GPU device, queue, surface, pipeline, and transient
/// vertex buffer. Window creation stays outside the library so Sim;X can choose
/// its own host application framework.
pub struct WgpuRenderer {
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    multisample_target: Option<MultisampleTarget>,
    sample_count: u32,
    vertices: Vec<Vertex>,
}

impl WgpuRenderer {
    /// Creates a renderer for a window or canvas surface target.
    ///
    /// `width` and `height` are physical surface pixels. Zero sizes are clamped
    /// to one pixel because `wgpu` surfaces cannot be configured at zero size.
    pub async fn new(
        surface_target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Self, RendererInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-engine device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(RendererInitError::RequestDevice)?;

        let width = width.max(1);
        let height = height.max(1);
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(RendererInitError::NoSurfaceConfig)?;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let sample_count = preferred_sample_count(&adapter, config.format);
        let pipeline = create_pipeline(&device, config.format, sample_count);
        let vertex_buffer = create_vertex_buffer(&device, INITIAL_VERTEX_CAPACITY);
        let multisample_target = create_multisample_target(&device, &config, sample_count);

        Ok(Self {
            _instance: instance,
            surface,
            _adapter: adapter,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
            multisample_target,
            sample_count,
            vertices: Vec::with_capacity(INITIAL_VERTEX_CAPACITY),
        })
    }

    /// Returns the configured surface size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigures the surface after a host window resize.
    ///
    /// Zero width or height is ignored because minimized windows often report
    /// zero size and `wgpu` cannot configure a zero-sized surface.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.multisample_target =
            create_multisample_target(&self.device, &self.config, self.sample_count);
    }

    /// Draws a scene using the supplied camera.
    ///
    /// Scene positions and sizes are in world units unless a style explicitly
    /// says screen pixels. The renderer converts world coordinates to screen
    /// pixels through [`Camera2d`], then to normalized device coordinates for
    /// `wgpu`.
    pub fn render(
        &mut self,
        scene: &Scene,
        camera: &Camera2d,
    ) -> Result<RenderStatus, RendererSurfaceStatus> {
        self.vertices.clear();
        let viewport = Viewport::new(self.config.width as f32, self.config.height as f32);
        tessellate_scene(scene, *camera, viewport, &mut self.vertices);
        self.ensure_vertex_capacity(self.vertices.len());

        if !self.vertices.is_empty() {
            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        }

        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(RenderStatus::Skipped(RendererSurfaceStatus::Timeout));
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(RenderStatus::Skipped(RendererSurfaceStatus::Occluded));
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(self.config.width, self.config.height);
                return Ok(RenderStatus::Skipped(RendererSurfaceStatus::Outdated));
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err(RendererSurfaceStatus::Lost);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RendererSurfaceStatus::Validation);
            }
        };

        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (view, resolve_target) = match &self.multisample_target {
            Some(target) => (&target.view, Some(&surface_view)),
            None => (&surface_view, None),
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sim-engine render encoder"),
            });

        {
            let color_attachment = wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(scene.background.to_wgpu()),
                    store: wgpu::StoreOp::Store,
                },
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sim-engine render pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if !self.vertices.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.draw(0..self.vertices.len() as u32, 0..1);
            }
        }

        self.queue.submit([encoder.finish()]);
        self.queue.present(surface_texture);

        Ok(RenderStatus::Drawn)
    }

    fn ensure_vertex_capacity(&mut self, vertex_count: usize) {
        if vertex_count <= self.vertex_capacity {
            return;
        }

        self.vertex_capacity = vertex_count.next_power_of_two();
        self.vertex_buffer = create_vertex_buffer(&self.device, self.vertex_capacity);
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sim-engine flat color shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sim-engine shape pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(Vertex::LAYOUT)],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn preferred_sample_count(adapter: &wgpu::Adapter, format: wgpu::TextureFormat) -> u32 {
    let flags = adapter.get_texture_format_features(format).flags;
    if flags.contains(
        wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4
            | wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE,
    ) {
        PREFERRED_SAMPLE_COUNT
    } else {
        1
    }
}

fn create_vertex_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-engine vertex buffer"),
        size: (capacity.max(1) * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_multisample_target(
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

fn tessellate_scene(
    scene: &Scene,
    camera: Camera2d,
    viewport: Viewport,
    vertices: &mut Vec<Vertex>,
) {
    for command in &scene.commands {
        match command {
            DrawCommand::Circle(circle) => tessellate_circle(circle, camera, viewport, vertices),
            DrawCommand::Rect(rectangle) => tessellate_rect(
                rectangle.rect,
                rectangle.corner_radius,
                rectangle.style,
                camera,
                viewport,
                vertices,
            ),
            DrawCommand::Line(line) => tessellate_line(line, camera, viewport, vertices),
            DrawCommand::Polyline(polyline) => {
                tessellate_polyline(polyline, camera, viewport, vertices);
            }
        }
    }
}

fn tessellate_circle(
    circle: &Circle,
    camera: Camera2d,
    viewport: Viewport,
    vertices: &mut Vec<Vertex>,
) {
    if circle.radius <= 0.0 {
        return;
    }

    if let Some(shadow) = circle.style.shadow {
        let radius_pixels = circle.radius * camera.zoom + shadow.spread;
        push_circle_screen(
            camera.world_to_screen(circle.center, viewport) + shadow.offset,
            radius_pixels.max(0.0),
            shadow.color,
            viewport,
            vertices,
        );
    }

    if let Some(fill) = circle.style.fill {
        push_circle_screen(
            camera.world_to_screen(circle.center, viewport),
            circle.radius * camera.zoom,
            fill,
            viewport,
            vertices,
        );
    }

    if let Some(stroke) = circle.style.stroke {
        push_circle_ring_screen(
            camera.world_to_screen(circle.center, viewport),
            circle.radius * camera.zoom,
            stroke.width,
            stroke.color,
            viewport,
            vertices,
        );
    }
}

fn tessellate_rect(
    rect: Rect,
    corner_radius: f32,
    style: ShapeStyle,
    camera: Camera2d,
    viewport: Viewport,
    vertices: &mut Vec<Vertex>,
) {
    let rect = rect.normalized();
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }

    if let Some(shadow) = style.shadow {
        let safe_zoom = camera.zoom.max(0.0001);
        push_rect_world(
            rect.expand(shadow.spread / safe_zoom),
            corner_radius,
            shadow.color,
            camera,
            viewport,
            shadow.offset,
            vertices,
        );
    }

    if let Some(fill) = style.fill {
        push_rect_world(
            rect,
            corner_radius,
            fill,
            camera,
            viewport,
            Vec2::ZERO,
            vertices,
        );
    }

    if let Some(stroke) = style.stroke {
        let min = rect.min;
        let max = rect.max;
        let points = [
            Vec2::new(min.x, min.y),
            Vec2::new(max.x, min.y),
            Vec2::new(max.x, max.y),
            Vec2::new(min.x, max.y),
            Vec2::new(min.x, min.y),
        ];

        for pair in points.windows(2) {
            push_line_screen(
                camera.world_to_screen(pair[0], viewport),
                camera.world_to_screen(pair[1], viewport),
                stroke.width,
                stroke.color,
                viewport,
                vertices,
            );
        }
    }
}

fn tessellate_line(line: &Line, camera: Camera2d, viewport: Viewport, vertices: &mut Vec<Vertex>) {
    push_line_screen(
        camera.world_to_screen(line.from, viewport),
        camera.world_to_screen(line.to, viewport),
        line.stroke.width,
        line.stroke.color,
        viewport,
        vertices,
    );
}

fn tessellate_polyline(
    polyline: &Polyline,
    camera: Camera2d,
    viewport: Viewport,
    vertices: &mut Vec<Vertex>,
) {
    for pair in polyline.points.windows(2) {
        push_line_screen(
            camera.world_to_screen(pair[0], viewport),
            camera.world_to_screen(pair[1], viewport),
            polyline.stroke.width,
            polyline.stroke.color,
            viewport,
            vertices,
        );
    }
}

fn push_circle_screen(
    center: Vec2,
    radius: f32,
    color: Color,
    viewport: Viewport,
    vertices: &mut Vec<Vertex>,
) {
    if radius <= 0.0 {
        return;
    }

    let center_vertex = vertex(center, color, viewport);
    for index in 0..CIRCLE_SEGMENTS {
        let angle_start = index as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        let angle_end = (index + 1) as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        vertices.push(center_vertex);
        vertices.push(vertex(
            center + Vec2::new(angle_start.cos(), -angle_start.sin()) * radius,
            color,
            viewport,
        ));
        vertices.push(vertex(
            center + Vec2::new(angle_end.cos(), -angle_end.sin()) * radius,
            color,
            viewport,
        ));
    }
}

fn push_circle_ring_screen(
    center: Vec2,
    radius: f32,
    width: f32,
    color: Color,
    viewport: Viewport,
    vertices: &mut Vec<Vertex>,
) {
    if radius <= 0.0 || width <= 0.0 {
        return;
    }

    let inner_radius = (radius - width * 0.5).max(0.0);
    let outer_radius = radius + width * 0.5;

    for index in 0..CIRCLE_SEGMENTS {
        let angle_start = index as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        let angle_end = (index + 1) as f32 / CIRCLE_SEGMENTS as f32 * std::f32::consts::TAU;
        let direction_start = Vec2::new(angle_start.cos(), -angle_start.sin());
        let direction_end = Vec2::new(angle_end.cos(), -angle_end.sin());

        push_quad(
            center + direction_start * inner_radius,
            center + direction_start * outer_radius,
            center + direction_end * outer_radius,
            center + direction_end * inner_radius,
            color,
            viewport,
            vertices,
        );
    }
}

fn push_rect_world(
    rect: Rect,
    corner_radius: f32,
    color: Color,
    camera: Camera2d,
    viewport: Viewport,
    screen_offset: Vec2,
    vertices: &mut Vec<Vertex>,
) {
    let points = rounded_rect_points(rect, corner_radius);
    if points.len() < 3 {
        return;
    }

    let center = camera.world_to_screen(rect.center(), viewport) + screen_offset;
    for index in 1..points.len() - 1 {
        vertices.push(vertex(center, color, viewport));
        vertices.push(vertex(
            camera.world_to_screen(points[index], viewport) + screen_offset,
            color,
            viewport,
        ));
        vertices.push(vertex(
            camera.world_to_screen(points[index + 1], viewport) + screen_offset,
            color,
            viewport,
        ));
    }
}

fn rounded_rect_points(rect: Rect, corner_radius: f32) -> Vec<Vec2> {
    let radius = corner_radius
        .max(0.0)
        .min(rect.width().abs() * 0.5)
        .min(rect.height().abs() * 0.5);

    if radius <= f32::EPSILON {
        return vec![
            Vec2::new(rect.max.x, rect.min.y),
            Vec2::new(rect.max.x, rect.max.y),
            Vec2::new(rect.min.x, rect.max.y),
            Vec2::new(rect.min.x, rect.min.y),
            Vec2::new(rect.max.x, rect.min.y),
        ];
    }

    let corners = [
        (
            Vec2::new(rect.max.x - radius, rect.max.y - radius),
            0.0,
            std::f32::consts::FRAC_PI_2,
        ),
        (
            Vec2::new(rect.min.x + radius, rect.max.y - radius),
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
        ),
        (
            Vec2::new(rect.min.x + radius, rect.min.y + radius),
            std::f32::consts::PI,
            std::f32::consts::PI * 1.5,
        ),
        (
            Vec2::new(rect.max.x - radius, rect.min.y + radius),
            std::f32::consts::PI * 1.5,
            std::f32::consts::TAU,
        ),
    ];

    let mut points = Vec::with_capacity(CORNER_SEGMENTS * 4 + 1);
    for (center, start_angle, end_angle) in corners {
        for step in 0..=CORNER_SEGMENTS {
            let amount = step as f32 / CORNER_SEGMENTS as f32;
            let angle = start_angle + (end_angle - start_angle) * amount;
            points.push(center + Vec2::new(angle.cos(), angle.sin()) * radius);
        }
    }
    points.push(points[0]);

    points
}

fn push_line_screen(
    from: Vec2,
    to: Vec2,
    width: f32,
    color: Color,
    viewport: Viewport,
    vertices: &mut Vec<Vertex>,
) {
    if width <= 0.0 {
        return;
    }

    let delta = to - from;
    if delta.length_squared() <= 0.0001 {
        return;
    }

    let normal = delta.perp().normalized() * (width * 0.5);
    push_quad(
        from + normal,
        to + normal,
        to - normal,
        from - normal,
        color,
        viewport,
        vertices,
    );
}

fn push_quad(
    a: Vec2,
    b: Vec2,
    c: Vec2,
    d: Vec2,
    color: Color,
    viewport: Viewport,
    vertices: &mut Vec<Vertex>,
) {
    vertices.push(vertex(a, color, viewport));
    vertices.push(vertex(b, color, viewport));
    vertices.push(vertex(c, color, viewport));
    vertices.push(vertex(a, color, viewport));
    vertices.push(vertex(c, color, viewport));
    vertices.push(vertex(d, color, viewport));
}

fn vertex(screen: Vec2, color: Color, viewport: Viewport) -> Vertex {
    Vertex {
        position: [
            screen.x / viewport.width * 2.0 - 1.0,
            1.0 - screen.y / viewport.height * 2.0,
        ],
        color: color.to_array(),
    }
}

#[allow(dead_code)]
fn debug_scene() -> Scene {
    let palette = Palette::sim();
    let mut scene = Scene::new(palette.background);
    scene.circle(
        Vec2::ZERO,
        32.0,
        ShapeStyle::filled(palette.primary).with_shadow(Shadow::new(
            Vec2::new(0.0, 12.0),
            10.0,
            Color::BLACK.with_alpha(0.25),
        )),
    );
    scene.line(
        Vec2::new(-100.0, 0.0),
        Vec2::new(100.0, 0.0),
        2.0,
        Stroke {
            width: 2.0,
            color: palette.axis,
        }
        .color,
    );
    scene
}
