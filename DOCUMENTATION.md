# Sim;Engine v0.1.0 Documentation

This document is the public guide and engineering reference for the official
Sim;Engine v0.1.0 release. It is divided into two parts:

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

Sim;Engine v0.1.0 is the first official release and remains pre-1.0. Linux with
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
- prepared, dynamic, particle, and scalar-field GPU resources;
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
sim-engine = "0.1"
```

Use the CPU-side visual-state APIs without GPU dependencies:

```toml
[dependencies]
sim-engine = { version = "0.1", default-features = false }
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
| Line | `line`, `try_line`, and layer variants | round caps, logical-pixel width |
| Polyline | `polyline`, `try_polyline`, and layer variants | joined segments and round end caps |

Styles include solid, linear-gradient, and radial-gradient fills; strokes; and
logical-screen shadows. Colors are straight linear RGBA internally.
`Color::rgb8` and `Color::rgba8` convert familiar sRGB bytes to linear light.
`Color::rgb` and `Color::rgba` accept values already in linear space. Render
boundaries require every channel in `0.0..=1.0`; animation may overshoot, but
the host must call `Color::clamp` explicitly before inserting that value.

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
| `WorldLength` | positive scalar distance in caller-defined 3D world units |
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
```

`RendererPresentMode::Vsync` requests FIFO. `NoVsync` requests the fastest
advertised non-VSync mode and may fall back to FIFO. Query
`renderer.surface_present_mode()` for the concrete Immediate, Mailbox, FIFO,
or FIFO-relaxed configuration chosen by the surface. This does not guarantee
that a desktop compositor will scan out above the monitor refresh rate.

On Wayland, `Window::pre_present_notify()` installs compositor pacing. Call it
for ordinary VSync presentation. An explicit uncapped diagnostic loop may omit
it when the selected surface mode is Immediate. Mailbox and FIFO still impose
their own back-pressure.

`renderer.wait_for_gpu_idle()` is for bounded benchmarks and controlled
readback. Waiting every interactive frame destroys CPU/GPU overlap.

### 9. Render reports and errors

```rust,ignore
match renderer.render_with_metrics(&scene, &camera) {
    Ok(report) => {
        let status = report.status();
        let metrics = report.metrics();
        let renderer_cpu = metrics.total_cpu();
        let dropped = metrics.tessellation_stats().dropped_command_count();
        record(status, renderer_cpu, dropped);
    }
    Err(error) => handle_render_error(error),
}
```

`RenderStatus::Skipped` is not a drawn frame. A timeout, occlusion, outdated
surface, or zero-sized surface can be transient. `RendererFrameMetrics`
contains CPU wall-clock durations for tessellation, upload, camera-uniform
upload, surface acquisition, encode/submit/present dispatch, and total renderer
work. It does not measure GPU completion or display scanout.

`TessellationStats` distinguishes accepted commands, rendered commands, and
dropped commands. Required visuals should use fallible scene APIs and hosts
should surface a non-zero dropped count in diagnostics.

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
- Use particle instancing for large point/circle populations.
- Put dense scalar fields in textures and update dirty regions only.
- Render expensive fields below surface resolution when quality permits.
- Stagger visualization cadence independently of simulation and presentation.
- Fuse common field/particle passes.
- Set hard particle memory, upload, visibility-check, and draw budgets.
- Benchmark release builds without VSync and record the concrete present mode.
- Report surface acquisition separately from renderer CPU time.

A display-limited frame rate is not renderer throughput. Surface acquisition
can wait for FIFO or compositor pacing while renderer CPU work remains small.
The current public metrics do not include GPU timestamp queries.

### 17. Examples

```bash
# Basic renderer integration
cargo run --release --example demo

# Menu with Fluid, Gas, Wave, and edge-case workloads
cargo run --release --example ui_demo -- --uncapped

# Bounded star-remnant gas/particle workload
cargo run --release --example star_remnant_stress -- --benchmark

# CPU-only particle-state baseline
cargo run --release --no-default-features --example particle_cpu_benchmark

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
scalar depth into logical screen X/Y. CPU validation checks geometry extents
against the same arithmetic envelope before submission. Lines retain
logical-pixel width by carrying a screen-extrusion direction separately from
world position.

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

The renderer clears reusable CPU storage, tessellates commands, validates
camera/geometry arithmetic, grows a transient vertex buffer when required,
uploads vertices, acquires the surface, encodes clipped batches, submits, and
presents.

#### Prepared scenes

Preparation creates an immutable dedicated GPU buffer and retains a CPU vertex
snapshot for recovery. Rendering reuses geometry and only updates camera state.

#### Dynamic meshes

The resource retains a CPU copy and capacity-managed triangle buffer. Full
updates grow amortized capacity; aligned range updates reuse it. Geometry
extents are recomputed after mutation.

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
- A Vulkan adapter is supported for 0.1 only when the mandatory semantic
  fixture passes on that concrete adapter/driver. CI records Mesa software
  evidence and the v0.1.0 release evidence records Intel Mesa; untested
  AMD/NVIDIA drivers are not silently certified by those results.
- The crate is pre-1.0.
- Text shaping and glyph caching are not implemented.
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
Vulkan semantic GPU readback fixture with backend assertion, `git diff --check`,
and the offline package boundary. The GPU step writes
`target/linux-vulkan-adapter.txt`; CI publishes the same manifest as the
`linux-vulkan-adapter` artifact. The manifest names backend, adapter type,
vendor/device IDs, driver, and driver version for the exact semantic run.

The gate must run from a clean worktree. Hardware performance or recovery
claims must additionally name the Linux adapter, backend, driver, workload,
present mode, and measurement method.
