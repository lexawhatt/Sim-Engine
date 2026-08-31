# Sim;Engine v0.2.0 Documentation

This document is the public guide and engineering reference for the official
Sim;Engine v0.2.0 release. It is divided into two parts:

- [Integration Handbook](#part-i-integration-handbook) - how to add the crate,
  construct visual state, choose a rendering path, recover resources, and
  measure performance.
- [Engineering Reference](#part-ii-engineering-reference) - how the crate is
  structured, which invariants it enforces, and where its responsibilities
  end.

For exact signatures and error variants, generate the Rust API reference:

```bash
cargo doc --all-features --no-deps --open
```

Sim;Engine v0.2.0 remains pre-1.0. Linux with
Vulkan is its supported release target, and Rust 1.90 is the minimum supported
Rust version.

## Part I: Integration Handbook

### 1. Scope

Sim;Engine renders ready visual state. The host application owns simulation
stepping, physics, mathematical meaning, entities, navigation, plugins, and UI
policy. The engine owns:

- validated 2D commands and visual styles;
- camera transforms, typed coordinate conversion, clipping, and pseudo-depth;
- visual interpolation;
- prepared, dynamic, image, glyph, particle, and scalar-field GPU resources;
- offscreen targets, composition, and bounded trails;
- retained stereometry meshes, depth, and visible or hidden display edges;
- renderer diagnostics, ownership checks, and resource recovery.

Sim;X is the primary consumer and design driver, but no Sim;X domain type is
required by the public API.

The renderer is not globally asynchronous. Adapter/device creation and device
recovery are async. Normal updates and render calls synchronously encode and
submit GPU work, without normally waiting for its completion.

### 2. Installation and features

The default feature set includes the `wgpu` backend:

```toml
[dependencies]
sim-engine = "0.2"
```

Use the CPU-side visual-state APIs without GPU dependencies:

```toml
[dependencies]
sim-engine = { version = "0.2", default-features = false }
```

| Configuration | Provides |
| --- | --- |
| default / `wgpu` | `WgpuRenderer`, GPU resources, targets, composition, particles, heatmaps, and retained 3D drawing |
| `--no-default-features` | scenes, cameras, colors, fields, particles, tweening, 3D math, mesh topology, and styles |

Window creation is deliberately outside the crate. The host may use `winit`,
as the examples do, or another framework that can provide a compatible
`wgpu::SurfaceTarget<'static>`.

`ParticleInstance2d` is part of the renderer-independent core. Hosts can
generate and benchmark particle visual state without compiling `wgpu`.
`ParticleField2d`, GPU upload, culling, and drawing require the renderer.

### 3. Frame model

A normal frame has four host-owned phases:

1. Advance the simulation or application state.
2. Convert that state into bounded Sim;Engine visual state.
3. Update reusable GPU resources that changed.
4. Render and inspect the returned report.

Do not move domain simulation into a scene builder or renderer callback. A
simulation can run on worker threads, then hand a ready visual snapshot to the
thread that owns the renderer.

### 4. Build a 2D scene

```rust
use sim_engine::{
    Camera2d, Color, Fill, LinearGradient, Rect, Scene, ShapeStyle, Vec2,
};

fn build_scene() -> Result<(Scene, Camera2d), Box<dyn std::error::Error>> {
    let mut scene = Scene::new(Color::rgb8(12, 14, 18))?;

    scene.try_circle(
        Vec2::new(24.0, 12.0),
        8.0,
        ShapeStyle::filled(Color::rgb8(86, 195, 255)),
    )?;

    scene.try_rect(
        Rect::from_center_size(Vec2::ZERO, Vec2::new(120.0, 64.0)),
        8.0,
        ShapeStyle::filled_with(Fill::LinearGradient(LinearGradient::new(
            Vec2::new(-60.0, 0.0),
            Vec2::new(60.0, 0.0),
            Color::rgb8(86, 195, 255),
            Color::rgb8(255, 190, 94),
        ))),
    )?;

    let camera = Camera2d::new(Vec2::ZERO, 2.0)?;
    Ok((scene, camera))
}
```

Use `try_*` methods when rejection must be observable. Convenience methods such
as `circle`, `rect`, `line`, and `polyline` return `bool`; use them only when
discarding invalid optional decoration is intentional.

| Primitive | Scene methods | Presentation |
| --- | --- | --- |
| Circle | `circle`, `try_circle`, and layer variants | fill, stroke, shadow |
| Rectangle | `rect`, `try_rect`, and layer variants | rounded corners, fill, stroke, shadow |
| Line | `line`, `try_line`, styled, and layer variants | bounded caps, dashes, markers, logical/world width |
| Polyline | `polyline`, `try_polyline`, styled, and layer variants | bounded joins, continuous dashes, markers |

Styles include solid, linear-gradient, and radial-gradient fills; strokes; and
logical-screen shadows. Colors are straight linear RGBA internally.
`Color::rgb8` and `Color::rgba8` convert familiar sRGB bytes to linear light.
`Color::rgb` and `Color::rgba` accept values already in linear space. Render
boundaries require every channel in `0.0..=1.0`; animation may overshoot, but
the host must call `Color::clamp` explicitly before inserting that value.
Circle centers and generated local offsets remain separate through the GPU
camera transform. A radius that is meaningful relative to a camera centered at
`1e20` therefore does not disappear merely because `center + radius` rounds
back to the same source `f32`. Fill, stroke, shadow/spread, and radial-gradient
sampling all use that same relative representation.

#### Rich bounded strokes

Legacy `line` and `polyline` methods retain their existing logical-pixel
presentation. Use `StrokeStyle2d` when a diagram needs explicit styling:

```rust
use sim_engine::{
    Color, LogicalPixels, Scene, StrokeCap2d, StrokeDashPattern2d, StrokeJoin2d,
    StrokeMarker2d, StrokeStyle2d, Vec2,
};

let dash = StrokeDashPattern2d::new(&[8.0, 4.0], 2.0, 256)?;
let arrow = StrokeMarker2d::arrow(
    LogicalPixels::new(10.0)?,
    LogicalPixels::new(8.0)?,
);
let style = StrokeStyle2d::logical(LogicalPixels::new(2.0)?, Color::WHITE)
    .with_cap(StrokeCap2d::Butt)
    .with_join(StrokeJoin2d::Miter)
    .with_miter_limit(4.0)?
    .with_dash_pattern(dash)
    .with_end_marker(arrow);

let mut scene = Scene::new(Color::BLACK)?;
scene.try_styled_polyline(
    vec![Vec2::ZERO, Vec2::new(20.0, 0.0), Vec2::new(30.0, 10.0)],
    style,
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Dash/gap lengths and phase use source path-coordinate units: world-coordinate
units for `Scene`, logical coordinate units for `ScreenScene`. A visible dash
continues across polyline vertices and receives the configured join. Width is
independent: `StrokeStyle2d::logical` stays constant under camera zoom, while
`StrokeStyle2d::world` uses a validated `WorldLength` and scales with the
camera. Endpoint markers always use logical pixels so annotations remain
readable. A marker's base is the path endpoint and its tip extends outward;
the marked body endpoint is forced to a butt boundary. Marker length therefore
cannot invert a short body or collide with a terminal join. Exact 180-degree
retraces and repeated adjacent points are rejected as structured scene errors
because no interior-disjoint translucent stroke topology exists for them.
Every consecutive polyline segment must be drawable; `DegenerateGeometry`
identifies a line segment or at least one consecutive polyline segment that is
not.

Every dash pattern carries `max_subsegments` in `1..=1_000_000`. Scene
insertion counts visible pieces before tessellation and returns
`SceneError::StrokeExpansionLimitExceeded` atomically when the ceiling would
be crossed. Dash boundaries that collapse at the path's `f32` coordinate scale
return `UnrepresentableStrokePattern` instead of entering a non-progressing
expansion loop. Miter length is bounded by `1.0..=1000.0`; geometry beyond the
configured limit falls back to bevel presentation. Extreme finite widths whose
derived extrusion would overflow are rejected as `InvalidStroke`.

#### Bounded scenes

Production adapters that translate externally sized visual state should use
`Scene::with_budget`. All six limits are hard upper bounds; zero is valid for a
background-only scene. Rejection is structured and leaves retained commands
unchanged.

```rust
use sim_engine::{Color, DrawCommand, Layer, Scene, SceneBudget, ShapeStyle, Vec2};

let budget = SceneBudget::new(
    20_000,
    100_000,
    2_000_000,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    20_000,
);
let mut scene = Scene::with_budget(Color::BLACK, budget)?;
let circle = DrawCommand::circle(
    Vec2::ZERO,
    4.0,
    ShapeStyle::filled(Color::WHITE),
)?;
scene.try_extend_to_layers([(Layer::DEFAULT, circle)])?;
assert_eq!(scene.statistics().accepted_commands(), 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `try_extend_to_layers` for large alternating/mixed-layer batches. It
validates the transaction before replacing the ordered command store, then
performs one `O(N log N)` sort using unique insertion order as the stable
tie-breaker. The core-only `scene_construction_benchmark` measures this path.
The compatibility constructor `Scene::new` remains explicitly unbounded.

### 5. Ordering, clipping, and pseudo-depth

2D commands are ordered by `Layer`, then insertion order. Pseudo-depth affects
projection only; it never reorders commands within a layer.

```rust
use sim_engine::{
    Color, LogicalScreenPosition, LogicalScreenVector, ScreenClipRect, Vec2,
};

# fn add_clipped_line(scene: &mut sim_engine::Scene) -> Result<(), Box<dyn std::error::Error>> {
let clip = ScreenClipRect::from_min_size(
    LogicalScreenPosition::new(40.0, 40.0),
    LogicalScreenVector::new(720.0, 420.0),
)?;

scene.with_screen_clip(clip, |scene| {
    scene.line(
        Vec2::new(-1_000.0, 0.0),
        Vec2::new(1_000.0, 0.0),
        2.0,
        Color::WHITE,
    );
})?;
# Ok(())
# }
```

Nested clips are intersected. Each command captures the active clip when it is
inserted, so later state changes cannot alter accepted commands. Clips use
logical pixels and a top-left origin.

`Projection2d` provides lightweight camera-relative tilt/depth presentation.
It is useful for layered 2.5D visuals, but it is not a z-buffer and does not
replace the retained 3D path.

### 6. Coordinates and HiDPI

| Type | Coordinate space |
| --- | --- |
| `Vec2` | caller-defined 2D world/vector value |
| `LogicalScreenPosition` | logical pixels, top-left origin |
| `LogicalScreenVector` | logical-pixel offset or size |
| `LogicalPixels` | positive logical-pixel scalar length |
| `PhysicalScreenPosition` | surface pixels, top-left origin |
| `PhysicalPerLogical` | physical target texels per logical pixel |
| `LogicalViewport` | finite logical viewport dimensions |
| `LogicalViewportRegion` | positioned local viewport inside a logical target |
| `WorldLength` | positive scalar distance in caller-defined 2D or 3D world units |
| `RenderTarget2d` dimensions | physical texture pixels |

Camera zoom is logical pixels per world unit. Stroke widths, screen shadows,
and clips are logical pixels, so monitor scale does not change their intended
visual size. Retained 3D wireframe widths and dash/gap lengths require
`LogicalPixels`; projection near/far distances and orthographic span require
`WorldLength`; `RenderTarget3d::pixels_per_logical` returns
`PhysicalPerLogical`. Their private representations prevent accidental direct
substitution between world, logical, and physical scales.

Convert physical pointer input before picking:

```rust,ignore
let logical_pointer = renderer.physical_to_logical_screen(
    PhysicalScreenPosition::new(pointer_x as f32, pointer_y as f32),
)?;
let world = camera.screen_to_world(
    logical_pointer,
    renderer.logical_viewport()?,
)?;
```

On window changes, update both physical size and scale:

```rust,ignore
renderer.resize_with_scale_factor(width, height, window.scale_factor())?;
```

Zero physical dimensions are ignored because a minimized surface cannot be
configured at zero size. Scale factors that cannot produce a finite logical
viewport are rejected.

#### Fixed UI and viewports

`ScreenScene` accepts typed `LogicalScreenPosition`, `LogicalScreenVector`, and
`LogicalPixels` geometry. Its top-left origin and downward y axis match UI hit
coordinates. `WgpuRenderer::render_screen_scene` derives the fixed camera
internally, while `PreparedScreenScene` prevents a world camera from being
supplied to prepared UI geometry.

```rust,ignore
let mut ui = ScreenScene::with_budget(Color::BLACK, ui_budget)?;
ui.try_rect(
    LogicalScreenPosition::new(16.0, 16.0),
    LogicalScreenVector::new(240.0, 80.0),
    LogicalPixels::new(8.0)?,
    ShapeStyle::filled(Color::rgb8(24, 31, 44)),
)?;
renderer.render_screen_scene(&ui)?;
```

`LogicalViewportRegion` positions a camera-local logical viewport on the
surface. `render_scene_in_viewport` always clips the scene to that region;
scene-local clips are relative to the region and intersect at physical scissor
conversion.

For offscreen rendering, `render_scene_to_target` requires an explicit
`PhysicalPerLogical`, optional local viewport, and `RenderTargetLoad`. Target
dimensions remain physical texels and never inherit window DPI. The scene
background is not an implicit target clear; pass
`RenderTargetLoad::Clear(scene.background())` when that is intended.

#### One composed frame

`FrameComposer` is the preferred surface path when UI, scientific viewports,
prepared geometry, particles, scalar fields, or offscreen targets must share a
defined order. It validates the complete frame before surface acquisition and
uses one clear, encoder, submission, and present.

```rust,ignore
let mut frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
frame.draw_prepared_screen_scene(
    &static_ui,
    FramePassOptions::new(0),
)?;
frame.draw_scene(
    &physics,
    physics_camera,
    FramePassOptions::new(10).with_viewport(canvas_region),
)?;
frame.draw_particle_field(
    &mut particles,
    physics_camera,
    FramePassOptions::new(20).with_viewport(canvas_region),
)?;
frame.draw_glyph_run(
    &font_atlas,
    &inspector_labels,
    ImageSampling::Linear,
    FramePassOptions::new(30),
)?;
let report = frame.present()?;
```

Items with equal `order` retain insertion order. Scene-local clips remain local
to that item's viewport and intersect the optional item clip. Prepared and
dynamic resources are rejected immediately when they belong to another
renderer generation. A retained 3D result participates through
`RenderTarget3d::color_target()`.

`FrameBudget` limits passes, referenced commands, vertices, uniform/streaming
uploads, referenced texture bytes, and conservative draw calls. Additions are
atomic, and actual post-tessellation work is checked again before GPU upload or
surface acquisition. `FrameStatistics` separates streaming vertices/uploads
from retained vertices reused without a per-frame geometry upload.
`FrameReport` also exposes actual command-encoder, render-pass, queue-submission,
and surface-present counts: each is one for `Drawn` and zero for a skipped
surface frame.

Images and glyph runs use the same ordering rule and clip/viewport
intersection as geometry. Referencing the same atlas several times counts its
texture allocation once toward the frame texture budget. Atlas pixels and
retained instances are not uploaded again merely because the frame draws them.

### 7. Cameras and motion

`Camera2d` supports finite pan, zoom, rotation, fit-to-bounds, picking, and
zooming about a logical-screen anchor. Forward and inverse conversion methods
are fallible because valid finite inputs can still overflow during arithmetic
or produce a singular projection. Preserve the last valid camera when an
interactive update fails.

`Tween<T>` is fallible for the same reason:

```rust
use std::time::Duration;
use sim_engine::{Easing, Tween};

let mut zoom = Tween::new(1.0)?
    .to(4.0, Duration::from_millis(300), Easing::EaseInOutCubic)?;
let current = zoom.update(Duration::from_millis(16))?;

# let _ = current;
# Ok::<(), sim_engine::TweenError>(())
```

Built-in `f32` and `Vec2` interpolation protects extreme finite endpoints from
overflow. A custom `Interpolate` implementation must report valid values
truthfully; invalid initial, target, or intermediate state returns
`TweenError` without partially mutating the tween.

### 8. Initialize and configure the renderer

```rust,ignore
use sim_engine::{RendererPresentMode, WgpuRenderer, WgpuRendererOptions};

let options = WgpuRendererOptions::new(
    RendererPresentMode::Vsync,
    window.scale_factor(),
)?;
let mut renderer = WgpuRenderer::new_with_options(
    window.clone(),
    window.inner_size().width.max(1),
    window.inner_size().height.max(1),
    options,
).await?;
let notify_window = window.clone();
renderer.set_pre_present_notify(move || notify_window.pre_present_notify());
```

`RendererPresentMode::Vsync` requests FIFO. `NoVsync` requests the fastest
advertised non-VSync mode and may fall back to FIFO. Query
`renderer.surface_present_mode()` for the concrete Immediate, Mailbox, FIFO,
or FIFO-relaxed configuration chosen by the surface. This does not guarantee
that a desktop compositor will scan out above the monitor refresh rate.
`adapter_name()`, `adapter_backend()`, `adapter_vendor_id()`,
`adapter_device_id()`, `adapter_pci_bus_id()`, `adapter_driver()`, and
`adapter_driver_info()` expose the adapter/API evidence that must travel with
benchmark results. The PCI bus address distinguishes physical instances of the
same GPU model when Vulkan exposes `VK_EXT_pci_bus_info`; an empty address is
not sufficient for release identity. `surface_format()` and
`surface_sample_count()` expose the matching production raster contract.
Recovery updates every value to the replacement logical device.

On Wayland, `Window::pre_present_notify()` installs compositor pacing. Register
it once through `WgpuRenderer::set_pre_present_notify`, as above. The renderer
invokes the callback after command submission and immediately before surface
present only when the concrete mode is FIFO, FIFO-relaxed, or Mailbox.
Immediate presentation deliberately omits the callback and remains suitable
for explicit uncapped diagnostics. The callback survives logical-device
recovery; call `clear_pre_present_notify` if the host window is replaced.

`renderer.wait_for_gpu_idle()` is for bounded benchmarks and controlled
readback. Waiting every interactive frame destroys CPU/GPU overlap.

### 9. Render reports and errors

```rust,ignore
match renderer.render_with_metrics(&scene, &camera) {
    Ok(report) => {
        let status = report.status();
        let metrics = report.metrics();
        let renderer_total = metrics.total_cpu();
        let surface_wait = metrics.surface_acquire();
        let renderer_work = renderer_total.saturating_sub(surface_wait);
        let dropped = metrics.tessellation_stats().dropped_command_count();
        record(status, renderer_work, surface_wait, dropped);
    }
    Err(error) => handle_render_error(error),
}
```

`RenderStatus::Skipped` is not a drawn frame. A timeout, occlusion, outdated
surface, or zero-sized surface can be transient, but a throughput fixture must
reject that attempt or exclude it from both its numerator and timing samples.
The release matrix fails on any skipped warmup or measured report.
Gated workloads run three independent 120-frame measurement trials. Wall
throughput must meet its floor in every trial; the median remains a reported
diagnostic. Renderer-work/acquire percentiles use all 360 drawn samples.
`RendererFrameMetrics`
contains CPU wall-clock durations for tessellation, upload, camera-uniform
upload, surface acquisition, encode/submit/present dispatch, and total call
time. `total_cpu()` includes `surface_acquire()`; subtract it when comparing
engine-side CPU work and report acquisition separately because FIFO/compositor
back-pressure normally appears there. These metrics do not measure display
scanout; the bounded matrix additionally uses completed wall throughput as its
GPU/back-pressure regression signal.

`TessellationStats` distinguishes accepted commands, rendered commands, and
dropped commands, and groups each by circle/rectangle/line/polyline category.
`SceneStatistics` exposes the same primitive grouping for requested, accepted,
and rejected construction work. Required visuals should use fallible scene
APIs and hosts should surface a non-zero dropped count in diagnostics.

For heterogeneous frames, `FrameStatistics::source_counts` separates
streaming scenes, prepared scenes, dynamic meshes, particles, scalar fields,
images, glyph runs, and targets. `retained_cpu_bytes`,
`retained_buffer_bytes`, and `texture_bytes` count unique referenced resource
allocations: drawing one prepared scene in several viewports does not multiply
its memory. Work counts still count every draw. This makes the prepared/static
invariant directly observable: after warm-up,
`streaming_vertex_count` and `streaming_upload_bytes` must include only sources
that actually changed.

Error-handling rules:

- `RendererMismatch` indicates a host lifecycle bug.
- Capacity errors require a smaller workload or a different device budget.
- Invalid transforms require corrected visual state; retrying unchanged data
  is not useful.
- Transient skipped surface states can be retried on a later event-loop turn.
- Device/surface loss requires explicit recovery and resource restoration.

### 10. Select a rendering path

| Workload | API | Expected update cost |
| --- | --- | --- |
| Small changing scene | `Scene` + `render_with_metrics` | validate, tessellate, upload every frame |
| Static shapes, moving camera | `PreparedScene` | prepare once, update camera per frame |
| Frequently changing triangles | `DynamicMesh2d` | upload ready triangle data |
| Images or atlas sprites | `Image2d` + `ImageBatch2d` | retain pixels and instances; update dirty regions |
| Host-shaped text | `GlyphAtlas2d` + `GlyphRun2d` | retain atlas/run; one quad per glyph |
| Many circles or points | `ParticleField2d` | cull, budget, compact, instanced draw |
| Dense scalar grid | `ScalarFieldTexture` | texture update and heatmap shader |
| Field plus particle overlay | `render_layered_visualization` | one encoder and one queue submission |
| Stereometry solids | `Mesh3d` + `Scene3d` | retained topology and per-object transforms |

#### Prepared scenes

```rust,ignore
let prepared = renderer.prepare_scene(&scene)?;
let report = renderer.render_prepared_with_metrics(&prepared, &camera)?;
```

Preparation captures geometry, background, clips, draw batches, and a CPU
recovery snapshot. Any geometry, style, order, or clip change requires a new
prepared scene. A prepared resource belongs to the renderer that created it.

#### Dynamic triangle geometry

`DynamicVertex2d` stores world position, pseudo-depth, and linear color.
`create_dynamic_mesh` accepts a triangle list whose vertex count is divisible
by three. Full updates replace the list; triangle-aligned range updates reuse
the existing allocation. Update reports expose upload time and reallocation.

Use this path for ready visual triangles, not as a new simulation data model.

Externally sized triangle input should use an explicit `DynamicMeshBudget`:

```rust,ignore
let budget = DynamicMeshBudget::new(
    30_000,          // triangle-list vertices
    2 * 1024 * 1024, // exact CPU recovery bytes
    2 * 1024 * 1024, // bytes in one full upload
)?;
let mesh = renderer.create_dynamic_mesh_with_budget(&vertices, budget)?;
frame.draw_dynamic_mesh(
    &mesh,
    camera,
    FramePassOptions::new(15).with_viewport(canvas_region),
)?;
```

This is the bounded raw-filled-triangle path for host-tessellated vector art
and scientific diagrams. Vertex counts must be divisible by three. Sim;Engine
does not infer a polygon fill rule because the host supplies the final triangle
list. Full updates allocate and validate a replacement CPU snapshot and any
required larger GPU buffer before committing retained state. Triangle-aligned
range updates remain in-place and enforce the retained upload budget.

#### Retained images and atlas sprites

`Image2d` owns one straight-alpha, top-to-bottom, row-major sRGB RGBA8 image.
RGB is decoded to linear light by the GPU texture; alpha remains linear.
`ImageBudget` constrains source dimensions and bytes before texture creation.
The owned constructor consumes a `Vec<u8>` without first making another
full-image copy; the slice constructor performs one fallible recovery copy.

```rust,ignore
let image_budget = ImageBudget::new(1024, 1024, 4 * 1024 * 1024)?;
let atlas = renderer.create_image_rgba8(
    atlas_width,
    atlas_height,
    rgba_pixels,
    image_budget,
)?;

let sprite = ImageSprite2d::new(
    ImageTexelRect::new(0, 0, 32, 32)?,
    icon_destination,
    Color::WHITE,
)?;
let batch = renderer.create_image_batch(
    &atlas,
    vec![sprite],
    ImageBatchBudget::new(256, 64 * 1024)?,
)?;

frame.draw_image_batch(
    &atlas,
    &batch,
    ImageSampling::Linear,
    FramePassOptions::new(20),
)?;
```

Sprite destinations are local logical pixels with a top-left origin. A frame
viewport positions the entire retained batch without rebuilding its instance
buffer. `draw_image` scales one full image or atlas rectangle into the item
viewport. `draw_world_image` instead maps an atlas rectangle onto an
axis-aligned world `Rect` at one pseudo-depth; the resulting quad follows that
item's camera zoom, rotation, projection, viewport, and clip.

Nearest filtering preserves masks and pixel art. Linear filtering samples only
between the selected sub-rectangle's first and last texel centers, so adjacent
atlas entries do not bleed at their own texel centers. Use padding when a
scaled asset intentionally filters near an outer edge.

`update_image_region` validates the exact row pitch, byte count, checked extent,
and atlas bounds before updating both GPU pixels and the CPU recovery snapshot.
`replace_image_rgba8` is atomic and creates a new resource identity, so batches
against the previous packing must be rebuilt. No implicit atlas eviction
occurs.

#### Host-shaped glyph runs

The text layer deliberately starts below font selection and shaping. The host
chooses fonts, fallback, localization, bidi behavior, baselines, advances, and
line breaks. It gives Sim;Engine opaque `GlyphId` values, atlas rectangles, and
already-positioned logical quads:

```rust,ignore
let entries = vec![
    GlyphAtlasEntry::new(mu_id, ImageTexelRect::new(0, 0, 18, 24)?),
    GlyphAtlasEntry::new(delta_id, ImageTexelRect::new(18, 0, 20, 24)?),
];
let atlas = renderer.create_glyph_atlas(
    width,
    height,
    atlas_pixels,
    entries,
    GlyphAtlasBudget::default(),
)?;
let glyphs = vec![
    PositionedGlyph2d::new(mu_id, mu_destination, Color::WHITE)?,
    PositionedGlyph2d::new(delta_id, delta_destination, Color::WHITE)?,
];
let run = renderer.create_glyph_run(
    &atlas,
    glyphs,
    GlyphRunBudget::default(),
)?;
let logical_bounds = run.bounds().region();
```

One successful run becomes one instanced draw and one retained quad per glyph.
Missing glyphs return `GlyphError::MissingGlyph` with identity and run index;
they are never silently replaced. `upload_glyph` can fill a bounded atlas
region and register new sorted metadata. Reusing an identity with a different
rectangle is rejected so old retained runs cannot begin sampling unrelated
pixels.

A run references one atlas. Mixed fallback fonts are explicit multiple runs at
the same frame order; stable insertion order preserves the host's chosen
overlap. `GlyphRunStatistics` exposes submitted glyphs, rendered quads, misses,
and retained bytes. Successful retained runs have zero misses and cause no
texture upload after warm-up.

### 11. Particles and hard budgets

`ParticleInstance2d` contains world position, logical-pixel radius, color, and
pseudo-depth. `ParticleField2d` retains validated instances and draws selected
particles through one instanced path.

```rust,ignore
use sim_engine::ParticleRenderBudget;

let budget = ParticleRenderBudget::new(
    30_000,          // maximum visible instances
    2 * 1024 * 1024, // maximum GPU instance bytes
    2 * 1024 * 1024, // maximum upload bytes per frame
)?
.with_max_visibility_checks(60_000)?;

let mut particles =
    renderer.create_particle_field_with_budget(&instances, budget)?;
```

When the visibility-check cap is below the retained population, the renderer
samples candidates uniformly rather than pretending that every instance was
classified. Inspect `ParticleStatistics`: submitted, visibility-checked,
visible, culled, budget-limited, dropped, and rendered counts are distinct.

Memory observability is explicit through `cpu_allocation_bytes`,
`gpu_allocation_bytes`, and `recovery_memory_bytes`. Use those limits so
rendering leaves capacity for the simulation itself.

### 12. Scalar fields and color maps

`ScalarField` stores a finite row-major grid. Dimensions must be non-zero and
match the value count. Full replacement and rectangular region updates are
validated atomically.

`ScalarField::filled` performs a fallible reservation under a 256 MiB default
value-buffer budget. Use `filled_with_byte_limit` for a different explicit host
budget, or `ScalarField::new` when the host already owns the allocated values.

```rust,ignore
let field = ScalarField::filled(160, 96, 0.0)?;
let mut texture = renderer.create_scalar_field_texture(field)?;

renderer.update_scalar_field_texture_region(
    &mut texture,
    x,
    y,
    width,
    height,
    &changed_values,
)?;
```

`ColorMap::sample` is piecewise-linear on the CPU. The renderer deliberately
quantizes a color map to a 256-entry RGBA8 lookup texture. Closely spaced stops
or HDR distinctions below that resolution cannot be preserved by the GPU
heatmap path.

Use `ScalarFieldSampling::Nearest` for exact texels and `Linear` for manual,
deterministic bilinear scalar sampling. Value-range endpoints and their
subtraction must be finite.

### 13. Targets, composition, and trails

`RenderTarget2d` dimensions are physical texture pixels. A typical multipass
frame renders an expensive layer into a target, then composes it onto the
surface using `BlendMode::Alpha`, `Additive`, or `Replace`.

Public colors are straight linear RGBA. Offscreen target storage is
premultiplied, and the composition shaders and blend states preserve that
contract. Do not manually premultiply a public `Color`.

`TrailBuffer2d` owns two ping-pong targets. Accumulation reads history from one
target, writes retained history and a fresh source into the other, then swaps
only after successful submission. Source/destination aliasing is rejected.
`clear_trail_buffer` clears both textures.

For a scalar field with a particle overlay, prefer
`render_layered_visualization`. It encodes heatmap, budgeted particles, and
surface composition with one command encoder and one queue submission.

### 14. Retained 3D stereometry

The retained 3D path is intentionally separate from 2D pseudo-depth:

```rust
use sim_engine::{
    Camera3d, LogicalViewport, Projection3d, Transform3d, Vec3, WorldLength,
};

let projection = Projection3d::perspective(
    std::f32::consts::FRAC_PI_3,
    16.0 / 9.0,
    WorldLength::new(0.1)?,
    WorldLength::new(100.0)?,
)?;
let camera = Camera3d::look_at(
    Vec3::new(0.0, 0.0, 5.0)?,
    Vec3::ZERO,
    Vec3::Y,
    projection,
)?;
let world = Transform3d::IDENTITY.transform_point(Vec3::X)?;
let projected = camera.project_world(
    world,
    LogicalViewport::new(1280.0, 720.0)?,
)?;

# let _ = projected;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Build immutable topology once, retain it on the GPU, and update only object
transforms during animation:

```rust,ignore
let topology = Mesh3d::with_display_edges(vertices, triangles, edges)?;
let retained = renderer.create_mesh3d(topology)?;
let logical_viewport = LogicalViewport::new(logical_width, logical_height)?;
let target = renderer.create_render_target3d(width, height, logical_viewport)?;

let wireframe = WireframeStyle3d::visible(
    Color::WHITE,
    LogicalPixels::new(2.0)?,
)?
.with_hidden(
    Color::rgb(0.45, 0.55, 0.70),
    LogicalPixels::new(1.25)?,
    LogicalPixels::new(7.0)?,
    LogicalPixels::new(5.0)?,
)?;
let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(surface_color)?)
    .with_wireframe(wireframe);
let mut scene = Scene3d::new(Color::BLACK)?;
let object = scene.try_push(&retained, Transform3d::IDENTITY, style)?;

scene.set_transform(object, next_transform)?;
let report = renderer.render_scene3d_to_target(&target, &scene, camera)?;
renderer.compose_render_target(
    target.color_target(),
    BlendMode::Replace,
    1.0,
    Color::BLACK,
)?;
```

Triangle indices define optional surfaces. Explicit `MeshEdge3d` values define
only mathematical display edges, so triangulation diagonals need not appear.
Edge-only meshes are valid. `Object3dId` is a stable opaque handle carrying
private scene provenance. A handle from another `Scene3d` returns
`ObjectNotFound` even when both objects have the same local numeric value.
`Scene3d::set_visible` hides an object without releasing retained topology.

Opaque surfaces write `Depth32Float`. Edge classification is conservative:
fragments occluded beyond a two-implementation-depth-unit tolerance receive a
logical-pixel dash pattern, while coplanar and sub-depth-resolution separations
resolve visible and solid. Host insertion order does not decide 3D visibility.

Every `RenderTarget3d` declares both physical texture dimensions and the
logical viewport represented by those texels. Their aspect ratios must match,
and the camera projection aspect must match the target logical aspect. This
keeps edge width stable for native, downsampled, and supersampled targets.

Display-edge segments are homogeneously clipped against all six frustum planes
before shader perspective division and screen-space expansion. A partially
visible edge is shortened; a fully clipped edge emits no fragments without
rejecting the rest of the frame.

Current 3D scope is deliberately focused: opaque surfaces, retained transforms,
hardware depth, solid visible edges, and dashed hidden edges. Translucent or
hatched sections, projected label anchors, text, and 3D picking are not in this
release.

### 15. Recovery

`recover_device_and_surface().await` replaces the device, queue, pipelines,
transient buffers, and renderer identity while reusing the surface. External
resources from the previous identity are rejected until restored.

| Resource | Restore method | Result |
| --- | --- | --- |
| `PreparedScene` | `restore_prepared_scene` | exact retained geometry |
| `DynamicMesh2d` | `restore_dynamic_mesh` | exact retained vertices/capacity |
| `Image2d` | `restore_image` | exact sRGB RGBA pixels and limits |
| `ImageBatch2d` | `restore_image_batch` | exact sprite instances against the restored image |
| `GlyphAtlas2d` | `restore_glyph_atlas` | exact pixels, IDs, rectangles, and limits |
| `GlyphRun2d` | `restore_glyph_run` | exact positioned glyphs against the restored atlas |
| `ParticleField2d` | `restore_particle_field` | exact instances and budget |
| `ScalarFieldTexture` | `restore_scalar_field_texture` | exact scalar grid |
| `RenderTarget2d` | `restore_render_target` | empty target; redraw required |
| `TrailBuffer2d` | `restore_trail_buffer` | empty history; redraw required |
| `RetainedMesh3d` | `restore_mesh3d` | exact topology and display edges |
| `Scene3d` | `restore_scene3d` | stable IDs plus exact transform/style/visibility; shared meshes uploaded once |
| `RenderTarget3d` | `restore_render_target3d` | empty color/depth; redraw required |

For a retained 3D scene, restore the scene and target after device recovery:

```rust,ignore
renderer.recover_device_and_surface().await?;
let scene_report = renderer.restore_scene3d(&mut scene)?;
target = renderer.restore_render_target3d(&target)?;

// Previously stored Object3dId values remain valid in `scene`.
scene.set_visible(selected_object, true)?;
```

`restore_scene3d` is atomic at the scene boundary. It recreates every distinct
stale mesh once and commits replacements only after all uploads succeed, while
preserving object IDs, order, transforms, styles, visibility, and next-ID state.

CPU-retained resources can recreate their exact content. GPU-only targets and
history cannot reconstruct prior pixels. Recovery is exceptional; it is not a
normal quality or adapter switch. Previous logical devices enter a bounded
quarantine because immediate teardown crashes some native Linux drivers. The
default limit is four and can be configured from one through eight. Once full,
recovery returns `RecoveryLimitReached` before creating another device; inspect
`quarantined_device_count` and `remaining_device_recoveries` in diagnostics.

### 16. Performance guidance

- Prepare static scenes once.
- Use dynamic meshes for ready changing triangles.
- Batch unchanged icons and glyphs in retained atlas instance buffers.
- Use particle instancing for large point/circle populations.
- Put dense scalar fields in textures and update dirty regions only.
- Render expensive fields below surface resolution when quality permits.
- Stagger visualization cadence independently of simulation and presentation.
- Fuse common field/particle passes.
- Set hard particle memory, upload, visibility-check, and draw budgets.
- Benchmark release builds without VSync and record the concrete present mode.
- Report surface acquisition separately from renderer CPU time.
- Compare primitive/source counts and retained/upload bytes before comparing
  milliseconds, so workload drift is not mistaken for a renderer regression.

A display-limited frame rate is not renderer throughput. Surface acquisition
can wait for FIFO or compositor pacing while renderer CPU work remains small.
The current public metrics do not include GPU timestamp queries.

The named release-mode matrix is:

```bash
./scripts/rendering_benchmark_matrix.sh
```

The release wrapper, standalone matrix, and standalone HiDPI script fail before
collecting evidence when the worktree is dirty. A common launcher creates a
read-only detached worktree at the captured revision and gives it a separate
Cargo target directory; compilation and execution never consume files from the
mutable calling checkout. The caller's `HEAD` and clean state are checked again
before success is accepted. Replacement refs and legacy grafts are rejected,
and `GIT_NO_REPLACE_OBJECTS=1` applies to checkout, build, and verification.
Child scripts receive only a hidden staging directory. The launcher writes a
completion manifest and atomically renames that directory to
`target/linux-release-evidence/` only after the entire requested gate returns
successfully. A failed or forcibly killed child can leave at most unpublished
staging, never a bundle claiming completion. This binds evidence to Git object
data for the recorded revision instead of relying on periodic checks of a
mutable source tree. A nonblocking `flock` serializes invalidation and
publication, so concurrent wrapper, matrix, or HiDPI invocations fail before
they can disturb an active gate's evidence.

The real compositor fixture requires the Linux executables
`dbus-run-session`, `kwin_wayland`, and `kscreen-doctor`. Their absence is a
hard gate failure, not a skipped test.

It runs `ui_static_10k`, `ui_90_10`, `four_viewports`, `image_atlas`,
`scientific_text`, `mixed_layers`, `budget_rejection`, `dpi_reconfigure`, and
`recovery_frame`. The viewport fixture owns four distinct prepared world
scenes and four distinct cameras. Surface fixtures print p50/p95/p99 renderer
work excluding acquire, separate acquire percentiles, observed wall
throughput, construction time, physical surface extent, scale factor, source
counts, vertices, uploads, unique retained memory, textures, and draw calls.
Gated performance fixtures require exactly `1280x720` physical pixels at scale
`1.0`; a compositor resize, minimization, or scale change cannot turn the
release workload into a cheaper raster test. A production-surface probe first
selects the high-performance Vulkan adapter and records its PCI bus address,
format, and MSAA count. The semantic oracle must use that exact physical GPU,
surface format, and sample count; backend, name, vendor/model IDs, and PCI bus
address must then match in every surface and HiDPI process. The semantic oracle
uses the probed production format/MSAA contract. A nested compositor's own
surface format may legitimately differ, so HiDPI pins the physical adapter but
records rather than equates that separate surface contract. Renderer-work p95
is capped at 5 ms for retained UI, four-camera,
image, and glyph workloads, 10 ms for repeated DPI reconfiguration, and 25 ms
for `ui_90_10`, which deliberately rebuilds and tessellates one thousand
commands per frame. These fixture-specific ceilings keep a streaming baseline
from weakening the retained-path oracle. If the selected surface mode is
Immediate, the gate additionally requires at least 60 observed FPS. Mailbox
and FIFO must sustain 95% of the confirmed current monitor's reported refresh
rate after clamping that release reference to 30-60 Hz. Before measuring, the
fixture always presents an unmeasured `Drawn` frame so Wayland can associate
the surface with an output. Immediate can proceed without monitor refresh
metadata. Mailbox/FIFO require a positive refresh rate from that current
monitor; zero or missing refresh is unconfirmed, and the fixture never
substitutes the primary or first enumerated monitor.
Measurement itself advances by one frame per event-loop redraw. The fixture
registers a presentation callback with the renderer; FIFO, FIFO-relaxed, and
Mailbox invoke `Window::pre_present_notify` after queue submission and
immediately before surface present, while Immediate deliberately skips it.
This lets synchronized modes schedule the next redraw against the compositor
frame callback instead of an application-only wakeup without accidentally
pacing the uncapped throughput oracle. It also lets
`ScaleFactorChanged`, `Resized`, and current-output changes run between every
warmup or measured sample. Confirmation carries the renderer surface-generation
number and output identity; any mismatch discards all partial samples and
requires a new unmeasured `Drawn` before restarting warmup. Metadata retry is
bounded to 120 confirmation presents, with a final follow-up redraw that can
accept metadata produced by the 120th present. After the final measured
present, a separate finalization redraw repeats the surface-generation and
output snapshot checks before computing or publishing a verdict.
Every warmup and measured frame must report `Drawn`. Surface-acquire
percentiles remain visible diagnostics in every mode, but a scheduler-sensitive
p95 from one trial is not an independent release threshold. Each gated fixture
uses three 120-frame trials, requires every trial to clear the wall-throughput
floor, reports the median for diagnosis, and combines all 360 frames for work
percentiles. The matrix includes an explicit FIFO surface run so the
refresh-normalized branch is exercised. `mixed_layers`
remains core-only;
`recovery_frame` is the mandatory Vulkan semantic fixture because it restores
every retained source on a second logical device and verifies bytes/pixels.
Record the adapter, driver, backend, and concrete present mode with results.
The checked thresholds are the project's Linux release floor, not a claim that
raw timings transfer between unrelated GPUs. A matrix is invalid if semantic,
performance, and compositor evidence do not name one identical adapter.

The automated `dpi_reconfigure` workload deliberately tests the renderer API.
Its renderer-work sample includes the timed `resize_with_scale_factor` call and
`surface.configure` work before `begin_frame`, so the 10 ms p95 applies to the
reconfiguration as well as the following frame.
The matrix additionally starts a nested KWin compositor and changes its real
output scale from 1.00 to 1.25 through `kscreen-doctor`. It accepts evidence
only when `ScaleFactorChanged`, its following `Resized`, and a drawn frame form
one completed transaction for the exact release revision. The same fixture can
still be inspected manually by moving its window between differently scaled
monitors and pressing Esc after the transition:

```bash
cargo run --release --example rendering_benchmark_suite -- \
  --fixture hidpi_transition
```

It logs `ScaleFactorChanged` and `Resized` order, applies the compositor's
physical size and scale at their event boundaries, and does not count an
unpaired redraw or the initial window-creation events as transition evidence.

### 17. Examples

```bash
# Basic renderer integration
cargo run --release --example demo

# Menu with Fluid, Gas, Wave, and edge-case workloads
# plus a retained host-shaped scientific glyph atlas probe
cargo run --release --example ui_demo -- --uncapped

# Bounded star-remnant gas/particle workload
cargo run --release --example star_remnant_stress -- --benchmark

# CPU-only particle-state baseline
cargo run --release --no-default-features --example particle_cpu_benchmark

# Named surface benchmark (default fixture is ui_90_10)
cargo run --release --example rendering_benchmark_suite -- --fixture ui_90_10

# Interactive pixel-level gallery for caps, joins, alpha, dashes, and markers
cargo run --release --example stroke_gallery -- --uncapped

# Complete named performance/contract matrix
./scripts/rendering_benchmark_matrix.sh

# Independently rotating cube and octahedron with hidden edges
cargo run --release --example stereometry_3d -- --uncapped

# Animated cylinder volume and surface-area derivation
cargo run --release --example cylinder_derivation_3d -- --uncapped --benchmark
```

Interactive controls and benchmark flags are printed by each example at
startup.

## Part II: Engineering Reference

### 18. Architectural role

The public boundary is validated visual state:

```text
host domain/UI state
    -> bounded visual snapshot
    -> Scene / particles / scalar fields / meshes
    -> WgpuRenderer and GPU passes
    -> Linux presentation surface
    -> status, metrics, and diagnostics back to the host
```

The host decides what a particle, field, cube, or section means. Sim;Engine
decides how ready visual data is validated, transformed, clipped, uploaded,
composed, drawn, measured, and recovered.

### 19. Module map

| Module | Responsibility |
| --- | --- |
| `math.rs` | checked 2D vectors and rectangles |
| `color.rs` | linear color, sRGB byte conversion, palette |
| `easing.rs`, `tween.rs` | fallible visual interpolation |
| `camera.rs` | 2D camera, pseudo-depth, typed screen spaces |
| `scene.rs` | validated ordered 2D command stream and styles |
| `field.rs` | finite scalar grid and CPU color-map contracts |
| `particle.rs` | renderer-independent particle visual state |
| `pseudo3d.rs` | checked 3D math, transforms, and CPU projection |
| `mesh3d.rs` | retained topology and edge/style contracts |
| `renderer/config.rs` | surface mode, DPI, renderer options, recovery setup |
| `renderer/tessellation.rs` | 2D scene command to triangle conversion |
| `renderer/visualization.rs` | fused scientific visualization path |
| `renderer/mesh3d.rs` | retained mesh resources and depth/edge passes |
| `renderer/primitive.wgsl` | 2D, particle, heatmap, and composition shaders |
| `renderer/mesh3d.wgsl` | 3D projection and screen-space edge expansion |

The entire renderer module is behind the `wgpu` feature. CPU-side contracts
remain testable without it.

### 20. Coordinate and projection model

The 2D transform is conceptually:

```text
world position + pseudo-depth
    -> camera-relative projection
    -> logical screen pixels
    -> physical scissor/texture pixels where required
    -> normalized device coordinates
```

The vertex shader receives compact camera rows mapping relative world X/Y and
scalar depth into logical screen X/Y. Generated circle vertices carry a world
anchor and local offset separately, so the subtraction from the camera happens
before a small radius is added. CPU validation checks both components against
the same arithmetic envelope before submission. Lines retain logical-pixel
width by carrying a screen-extrusion direction separately from world position.

The retained 3D transform is explicit:

```text
model point
    -> Transform3d
    -> Camera3d view basis
    -> Projection3d clip coordinates
    -> normalized depth and logical-screen position
```

The 3D convention is right-handed with positive Y up. Cameras look along local
negative Z; exposed view depth is positive distance forward. Large finite
vector operations use wider intermediates before checked `f32` output.

### 21. Color and alpha model

- Public `Color` is straight linear RGBA.
- Byte constructors decode sRGB to linear light.
- Render-bound colors require every channel in `0.0..=1.0`.
- Scene vertices remain linear through tessellation.
- Surface formats apply their configured output conversion.
- Offscreen target storage is premultiplied alpha.
- Composition shaders and blend states preserve that premultiplied storage.
- `Color::clamp` is explicit sanitization, not implicit scene validation.
- GPU color maps are deliberately quantized to 256 RGBA8 samples.

### 22. Scene and tessellation invariants

`Scene` owns a validated background, an ordered command list, temporary clip
and pseudo-depth state, and monotonically increasing insertion order. Accepted
commands capture immutable primitive/style data plus layer, insertion order,
depth, and optional clip.

Insertion checks source finiteness, drawability, styles, derived bounds,
gradient arithmetic, and operations that can overflow despite finite inputs.
Fallible methods preserve rejection reasons. Sorting is stable by layer and
never depends on pseudo-depth.

The general scene path tessellates on the CPU:

- circles become triangle fans;
- rectangles become fans or rounded sectors;
- lines become screen-extruded strips with round caps;
- polylines share joins and emit only two end caps;
- gradients are sampled at generated vertices;
- shadows generate separate offset geometry.

If late tessellation still cannot emit an accepted optional command, that loss
is visible in `TessellationStats` rather than hidden behind a successful frame.

### 23. Renderer lifecycle and ownership

`WgpuRenderer` owns one surface, adapter, logical device, queue, surface
configuration, pipelines, uniforms, transient scene buffers, optional MSAA
target, and cached lookup resources.

Renderer-owned external resources store the identity of the logical device
that created them. Every operation checks that identity before touching GPU
handles. Cross-renderer use therefore becomes a structured error before wgpu
validation. `Object3dId` similarly carries private scene provenance.

Capacity-bearing APIs validate integer arithmetic, host byte budgets, and active
device buffer or texture limits before allocation. Mutating fallible methods
validate first and replace retained state only after all checks pass.

### 24. GPU path internals

#### Streaming scenes

The renderer clears and reuses its previous transient CPU allocation,
tessellates commands directly into that upload payload, validates
camera/geometry arithmetic, grows the GPU vertex buffer when required, uploads
vertices, acquires the surface, encodes clipped batches, submits, and presents.
Circle and rounded-corner unit samples are immutable process-wide lookup data;
per-command positions still receive the exact documented segment counts but do
not recalculate the same trigonometric samples every frame. Circle samples are
uploaded as local offsets from their retained center, preserving fill, stroke,
shadow/spread, and radial-gradient behavior below the center's `f32` ULP. The
80-byte tessellated vertex size, scene estimates, upload budgets, and reported
upload bytes all include this relative-world component.

#### Prepared scenes

Preparation creates an immutable dedicated GPU buffer and retains a CPU vertex
snapshot for recovery. Rendering reuses geometry and only updates camera state.

#### Dynamic meshes

The resource retains a CPU copy and capacity-managed triangle buffer. Full
updates grow amortized capacity; aligned range updates reuse it. Bounded meshes
preflight vertex/retained/upload work, use fallible staging reservation, and
commit a full update only after replacement resources exist. Geometry extents
are recomputed after mutation.

#### Images and glyph runs

Images use an `Rgba8UnormSrgb` sampled texture and exact retained source bytes.
Single logical or world images draw one shader-generated quad. Atlas and glyph
batches store logical destination, UV-center bounds, and straight-linear tint
in one retained instance buffer and issue one instanced draw. Glyph metadata is
kept sorted for deterministic allocation-free lookup while runs are built.

#### Particle fields

All validated instances stay on the CPU. Rendering tests circle/viewport
intersection, uniformly samples candidates when visibility checks are capped,
compacts the selected list, performs one upload, and issues one instanced draw.
GPU allocation and upload are proportional to the budget rather than the full
host population.

#### Scalar fields

The renderer stores an `R32Float` texture plus a retained CPU grid. Full and
rectangular updates are validated before queue writes. Heatmap shaders map a
finite value range through a cached lookup texture.

#### Retained 3D

Immutable vertex/index/edge topology is mirrored into GPU buffers and retained
on the CPU for recovery. Scene objects store scene-provenance IDs and
independent model transforms. Opaque surfaces populate color and depth. Display
edges are homogeneously clipped, conservatively classified against depth,
rendered dashed when hidden beyond the coplanar tolerance, and rendered solid
when visible. Edge expansion uses the target's logical-to-physical ratio rather
than the window DPI.

Mesh upload preflights vertex, index, and edge counts, checked byte sizes,
draw-count representation, and the active device's `max_buffer_size` before
allocating conversion staging memory. Staging vectors use fallible reservation
and report `HostAllocationFailed`. GPU buffer creation still follows wgpu's
device-error model and is not presented as a catchable system-OOM boundary.

### 25. Validation philosophy

The nearest public boundary rejects:

- NaN and infinity;
- prohibited zero or negative dimensions, radii, zoom, and scale;
- derived overflow from otherwise finite inputs;
- invalid gradient or value-range arithmetic;
- non-normalized render-bound colors;
- filled scalar allocations beyond the active host byte budget;
- allocation beyond active device limits;
- resource/renderer identity mismatch;
- update regions outside retained state;
- invalid opacity and composition aliasing;
- unsafe camera, near-plane, perspective-divide, or edge-style arithmetic.

This is a behavioral contract: an error should identify invalid required
visual state before it silently becomes different pixels.

### 26. Async and threading

The renderer is designed to be owned by the host render/event-loop thread.
Initialization and recovery are async because adapter and device requests are
async. Ordinary resource mutations and drawing are synchronous submissions.

The engine does not create a simulation scheduler or background render thread.
Async simulation is useful only when snapshot and synchronization cost is
smaller than the work it overlaps. Making individual scene mutations async
would add coordination overhead without improving GPU submission.

### 27. API stability policy

Sim;Engine is pre-1.0. Minor versions may contain intentional source-breaking
changes while real consumers exercise the contracts. Such changes must be
listed in `CHANGELOG.md` with migration guidance.

Patch releases preserve these behavioral contracts:

- invalid or overflowing required input is rejected structurally;
- coordinate spaces are not silently interchanged;
- pseudo-depth does not reorder a 2D layer;
- public colors are straight linear RGBA and target alpha is premultiplied;
- GPU resources reject foreign renderer identities;
- recovery behavior and retained snapshots match their documentation;
- failed updates do not partially mutate retained state;
- the crate accepts visual state and does not own application domain rules.

Public fields are avoided so validation, units, transforms, and future
representation changes remain evolvable. New public paths need a real consumer,
boundary validation, core and GPU tests where applicable, memory accounting,
recovery behavior, and a valid no-default-features build.

The project will not claim 1.0 until its supported platform/backend matrix,
recovery behavior, performance budgets, public documentation, and release
automation are repeatable.

### 28. Known boundaries

- Linux with Vulkan is the only release-gated platform/backend contract.
- A Vulkan adapter is supported for 0.2 only when the mandatory semantic
  fixture passes on that concrete adapter/driver. CI records Mesa software
  evidence and the release evidence records the exact tested adapter; untested
  AMD/NVIDIA drivers are not silently certified by those results.
- The crate is pre-1.0.
- Font loading, text shaping, fallback selection, line breaking, and automatic
  atlas eviction are not implemented. The low-level API renders
  host-shaped glyph runs and retains their atlas/cache resources.
- Retained 3D supports opaque surfaces and depth-classified edges, not section
  materials, projected anchors, labels, or picking.
- Renderer timing is CPU-side; public GPU timestamps are not available.
- Independent multi-window recovery is not yet proven.
- Render-target and trail pixels cannot be reconstructed after device loss.

### 29. Verification and contribution gate

The complete local Linux release gate is:

```bash
./scripts/linux_release_gate.sh
```

It checks formatting, Rust 1.90 compatibility, all targets with and without
default features, strict clippy, doctests, warning-free rustdoc, a mandatory
Vulkan semantic GPU readback fixture with backend assertion, the Vulkan-pinned
performance matrix, a real nested-KWin HiDPI transition, `git diff --check`,
and the offline package boundary. A successful local wrapper atomically
publishes `target/linux-release-evidence/`, containing `completion.txt`,
`linux-vulkan-surface.txt`, `linux-vulkan-adapter.txt`, and
`linux-vulkan-performance.txt`, and `linux-hidpi-transition.txt`. The
performance manifest records the exact SHA plus every fixture's adapter,
surface, present mode, trial FPS, CPU/acquire percentiles, deterministic work
counters, threshold, and passed verdict. CI's narrower semantic job writes and
publishes `target/linux-vulkan-adapter.txt` directly as the
`linux-vulkan-adapter` artifact. The manifest names the exact VCS SHA, backend,
adapter type, vendor/model IDs, PCI bus address when available, driver, oracle
format, and sample count for the semantic run. CI supplies `github.sha` and
asserts that the artifact does not contain `vcs_sha=unknown`.
The bundled HiDPI manifest records the exact VCS revision, Vulkan backend,
scale, physical size, and transactional event counts.

The wrapper and each standalone evidence-producing surface/HiDPI script must
run from a clean worktree and finish on the exact revision captured at start.
Their shared nonblocking `flock` prevents concurrent gates from deleting,
nesting, or relabeling one another's evidence, and Linux `mv -T` is the sole
publication boundary.
Hardware performance or recovery
claims must additionally name the Linux adapter, PCI vendor/device and bus
address, backend, driver, surface format/sample count, workload, present mode,
confirmed current-monitor refresh where applicable, drawn/attempted counts,
and measurement method.
