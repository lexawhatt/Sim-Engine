# Sim;Engine Integration Guide

This guide explains how a host application supplies visual state to Sim;Engine,
selects the appropriate rendering path, integrates a `wgpu` surface, handles
errors, and keeps rendering cost bounded. For implementation internals, see the
[Architecture Reference](ARCHITECTURE_REFERENCE.md).

## 1. Scope And Status

Sim;Engine renders state; it does not calculate that state. Sim;X or another
host owns simulation stepping, domain entities, physical constants,
mathematical construction rules, UI navigation, and plugin execution. The
engine owns:

- validated 2D scene commands and visual styles;
- camera transforms, coordinate conversion, clipping, and pseudo-depth;
- visual tweening;
- dynamic, prepared, particle, and scalar-field GPU resources;
- offscreen targets, composition, and bounded trails;
- renderer metrics, ownership checks, and resource recovery;
- the validated math/camera foundation for the planned Sim;Math stereometry
  path.

The 2D renderer and scientific-visualization paths are functional. `Vec3`,
`Rotation3d`, `Transform3d`, `Projection3d`, and `Camera3d` are available, but
retained 3D meshes, depth-tested surfaces, and hidden-line rendering are still
under development. Do not build a production stereometry renderer by treating
`Scene::depth` as a z-buffer.

The first supported release platform is Linux.

The minimum supported Rust version is 1.87, matching the `wgpu 30.0.1`
dependency used by the default feature set.

## 2. Dependency And Features

The default feature set includes the `wgpu` renderer:

```toml
[dependencies]
sim-engine = "0.1"
```

For scene construction, cameras, fields, colors, tweening, and pseudo-3D math
without a GPU dependency:

```toml
[dependencies]
sim-engine = { version = "0.1", default-features = false }
```

Feature summary:

| Feature | Default | Provides |
| --- | --- | --- |
| `wgpu` | yes | `WgpuRenderer`, GPU resources, render targets, particles, heatmaps, and composition |
| no default features | no | CPU visual-state types, cameras, fields, colors, scenes, tweening, and pseudo-3D foundation |

Window creation is intentionally not part of the crate. A host may use `winit`
as the bundled examples do, or any framework capable of producing a
`wgpu::SurfaceTarget<'static>`.

## 3. The Basic Frame Model

A normal frame has four host-owned steps:

1. Advance domain simulation.
2. Convert domain state into Sim;Engine visual state.
3. Update reusable GPU resources where needed.
4. Render and inspect the returned status/metrics.

The engine is not globally asynchronous. GPU adapter/device creation and
explicit recovery are `async`; steady-state updates and render calls are
synchronous Rust methods that encode and submit GPU work. They do not normally
wait for GPU completion. Surface acquisition may block under FIFO/VSync
back-pressure.

## 4. Build A 2D Scene

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

Use `try_*` methods when rejection must be observable. The shorter `circle`,
`rect`, `line`, and `polyline` methods return `bool`; they are appropriate only
when the caller deliberately treats rejected optional decoration as harmless.
Invalid required content should never be ignored.

### Supported primitives

| Primitive | Scene methods | Notes |
| --- | --- | --- |
| Circle | `circle`, `try_circle`, layer variants | fill, stroke, and shadow |
| Rectangle | `rect`, `try_rect`, layer variants | optional rounded corners |
| Line | `line`, `try_line`, layer variants | round caps, logical-pixel width |
| Polyline | `polyline`, `try_polyline`, layer variants | joined segments and round end caps |

### Styles

- `ShapeStyle::filled` creates a solid fill.
- `ShapeStyle::filled_with` accepts `Fill::Solid`, `Fill::LinearGradient`, or
  `Fill::RadialGradient`.
- `ShapeStyle::stroked` creates an outline.
- `ShapeStyle::fill_stroke` and `fill_stroke_with` combine both.
- `with_shadow` adds a logical-screen offset/spread shadow.

Colors are linear RGBA internally. Use `Color::rgb8` or `rgba8` for familiar
sRGB byte colors. Direct `Color::rgb`/`rgba` values are already linear and are
not sRGB-decoded.

## 5. Ordering, Clipping, And Pseudo-Depth

Commands are ordered by `Layer`, then insertion order. `Layer::BACKGROUND`,
`DEFAULT`, and `FOREGROUND` cover common cases; `Layer::new` creates an explicit
order. Pseudo-depth changes projection only and never reorders commands.

```rust
use sim_engine::{
    LogicalScreenPosition, LogicalScreenVector, ScreenClipRect,
};

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
```

Nested clips are intersected. Commands capture the active clip at insertion;
later clip changes do not mutate existing commands. Clips use logical pixels
with a top-left origin and are converted to physical scissor pixels by the
renderer.

`Projection2d` supplies a lightweight tilt/depth presentation effect. It takes
a 2D world point and caller-defined scalar depth. It is useful for layered 2.5D
visuals, not real 3D occlusion.

## 6. Coordinates And HiDPI

Keep these spaces distinct:

| Type | Space |
| --- | --- |
| `Vec2` | caller-defined 2D world/vector value |
| `LogicalScreenPosition` | logical pixels, top-left origin |
| `LogicalScreenVector` | logical-pixel offset or size |
| `PhysicalScreenPosition` | surface pixels, top-left origin |
| `LogicalViewport` | logical viewport dimensions |
| `RenderTarget2d` dimensions | physical texture pixels |

Camera zoom is logical pixels per world unit. Stroke widths, shadows, and clips
are logical pixels, so they remain visually stable across display scale.

Convert window pointer events before camera picking:

```rust
use sim_engine::PhysicalScreenPosition;

let logical_pointer = renderer.physical_to_logical_screen(
    PhysicalScreenPosition::new(pointer_x as f32, pointer_y as f32),
)?;
let world = camera.screen_to_world(logical_pointer, renderer.logical_viewport()?)?;
```

On resize, pass both the new physical dimensions and the current display scale:

```rust
renderer.resize_with_scale_factor(width, height, window_scale_factor)?;
```

Zero physical dimensions are ignored because minimized windows cannot configure
a zero-sized `wgpu` surface.

## 7. Camera And Interaction

`Camera2d` provides:

- finite center, zoom, and rotation setters;
- `pan_by` in world units;
- `zoom_about_screen`, preserving the world point under a logical-pixel anchor;
- `fit_to_bounds` with logical-pixel padding;
- `world_to_screen` and `screen_to_world` at depth zero;
- projected forward and inverse conversions at an explicit pseudo-depth;
- `tween()` for validated camera interpolation.

Every forward/inverse operation is fallible when finite inputs overflow or the
projection becomes singular. Preserve the old camera if an interactive update
returns an error.

## 8. Tweening

```rust
use std::time::Duration;
use sim_engine::{Easing, Tween};

let mut zoom = Tween::new(1.0)?
    .to(4.0, Duration::from_millis(300), Easing::EaseInOutCubic)?;

let current = zoom.update(Duration::from_millis(16))?;
```

`Tween<T>` works with `Interpolate`. Built-in implementations validate extreme
arithmetic. Custom implementations must provide a truthful validity predicate;
invalid initial, target, or intermediate values return `TweenError` without
partially mutating retained state.

## 9. Create The GPU Renderer

Initialization is asynchronous because adapter and device acquisition may wait
on the graphics backend:

```rust
use sim_engine::{
    RendererPresentMode, WgpuRenderer, WgpuRendererOptions,
};

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

Use `pollster::block_on` from a synchronous desktop event-loop callback if that
fits the host. Do not make the simulation loop async solely because renderer
initialization is async.

`RendererPresentMode::Vsync` requests FIFO. `NoVsync` requests the fastest
available non-VSync mode and may fall back to FIFO. Performance measurements
must record the actual environment and separate surface wait from renderer CPU.

## 10. Render And Interpret Results

```rust
match renderer.render_with_metrics(&scene, &camera) {
    Ok(report) => {
        let status = report.status();
        let metrics = report.metrics();
        let renderer_cpu = metrics.total_cpu();
        let dropped = metrics.tessellation_stats().dropped_command_count();
        // Record or surface these values in host diagnostics.
    }
    Err(error) => {
        // Route surface loss, transform overflow, or capacity errors explicitly.
    }
}
```

`RenderStatus::Skipped` is not equivalent to a drawn frame. Timeout, occlusion,
and outdated surface states can be transient. `RendererFrameMetrics` are CPU
durations; they do not measure completed GPU execution or display scanout.

Metrics expose:

- CPU tessellation;
- geometry upload;
- camera-uniform upload;
- surface acquisition/back-pressure;
- encode/submit/present dispatch;
- total renderer CPU;
- prepared/dynamic path flags;
- accepted, rendered, and dropped scene command counts.

## 11. Choose The Correct Geometry Path

| Workload | API | Update cost |
| --- | --- | --- |
| Small changing scene | `Scene` + `render_with_metrics` | validate/tessellate/upload each frame |
| Static shapes, moving camera | `PreparedScene` | prepare once, camera upload per frame |
| Frequently changing triangles | `DynamicMesh2d` | caller builds triangle list, upload updates |
| Many circles/points | `ParticleField2d` | instanced draw, culling and explicit budgets |
| Dense scalar grid | `ScalarFieldTexture` | texture upload plus color-map shader |
| Mixed gas/particles | `render_layered_visualization` | fused target render and one queue submit |

### Prepared scenes

```rust
let prepared = renderer.prepare_scene(&scene)?;
renderer.render_prepared_with_metrics(&prepared, &camera)?;
```

Preparing captures geometry, background, batches, clips, and a CPU recovery
snapshot. Any geometry/style/order/clip change requires a replacement. Prepared
resources belong to the renderer that created them.

### Dynamic meshes

`DynamicVertex2d` carries world position, pseudo-depth, and linear color.
`create_dynamic_mesh` requires a triangle-list count divisible by three.
`update_dynamic_mesh` replaces the list; `update_dynamic_mesh_range` modifies a
triangle-aligned range without reallocating. Update reports expose upload time
and buffer growth.

Use dynamic meshes for ready visual triangles, not as a place to move domain
simulation logic into the renderer.

## 12. Particle Fields And Hard Budgets

`ParticleInstance2d` contains world position, logical-pixel radius, color, and
pseudo-depth. `ParticleField2d` retains validated instances and uses one
instanced circle path.

Always set a budget for expensive host simulations:

```rust
use sim_engine::ParticleRenderBudget;

let budget = ParticleRenderBudget::new(
    30_000,       // maximum visible instances
    2 * 1024 * 1024, // maximum GPU instance bytes
    2 * 1024 * 1024, // maximum upload bytes per frame
)?
.with_max_visibility_checks(60_000)?;

let mut particles = renderer.create_particle_field_with_budget(&instances, budget)?;
```

Rendering samples candidates uniformly when the visibility-check cap is below
the retained count. Inspect `ParticleStatistics::{visibility_checked,
budget_limited, visible, culled, rendered}`; do not describe an approximated
frame as if every retained particle was classified.

`cpu_allocation_bytes`, `gpu_allocation_bytes`, and `recovery_memory_bytes`
support workload accounting.

## 13. Scalar Fields And Color Maps

`ScalarField` stores a finite row-major grid. Dimensions must be non-zero and
match the value count. Full replacement and rectangular region updates are
validated atomically.

`ColorMap` is a CPU piecewise-linear map with sorted normalized stops. GPU
heatmaps use an explicitly quantized 256-entry RGBA8 lookup texture.

```rust
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

Choose `ScalarFieldSampling::Nearest` for exact texels or `Linear` for manual
deterministic bilinear interpolation. Value ranges must be finite, ordered, and
have a finite subtraction.

## 14. Render Targets, Composition, And Trails

`RenderTarget2d` dimensions are physical texture pixels. A typical multipass
flow is:

1. Create a target, often below surface resolution.
2. Render a scalar or particle layer into it.
3. Compose it to the surface using `BlendMode::Alpha`, `Additive`, or `Replace`.

Target alpha storage is premultiplied. Public colors remain straight linear
RGBA; shaders and blend pipelines perform the conversion at the target
boundary.

`TrailBuffer2d` owns two ping-pong targets. `accumulate_trail_buffer` combines
retained history and a fresh source, validates both opacities, rejects source
aliasing, and swaps buffers only after submission. `clear_trail_buffer` clears
both textures.

For the common scalar-field plus particle overlay, prefer
`render_layered_visualization`. It encodes heatmap, budgeted particles, and
surface composition with one command encoder and one queue submission.

## 15. Pseudo-3D Foundation

Current CPU-side types are intentionally separate from `Projection2d`:

```rust
use sim_engine::{Camera3d, LogicalViewport, Projection3d, Rotation3d, Transform3d, Vec3};

let projection = Projection3d::perspective(
    std::f32::consts::FRAC_PI_3,
    16.0 / 9.0,
    0.1,
    100.0,
)?;
let camera = Camera3d::look_at(
    Vec3::new(0.0, 0.0, 5.0)?,
    Vec3::ZERO,
    Vec3::Y,
    projection,
)?;
let rotation = Rotation3d::from_euler_xyz(0.2, 0.4, 0.0)?;
let transform = Transform3d::from_rotation_scale(rotation, 1.0)?;
let world = transform.transform_point(Vec3::X)?;
let projected = camera.project_world(world, LogicalViewport::new(1280.0, 720.0)?)?;
```

`ProjectedPoint3d` exposes logical position, positive view depth, normalized
depth, and clip membership. Points behind the camera return an error.

This foundation does not yet provide `Mesh3d` or GPU depth/hidden-line output.
See [Pseudo-3D Rendering Specification](PSEUDO_3D.md) before integrating it.

## 16. Device And Surface Recovery

`recover_device_and_surface().await` replaces the logical device, queue,
pipelines, transient buffers, and renderer identity while reusing the existing
surface. Every external GPU resource from the previous identity becomes invalid
until restored:

| Resource | Restore method | Restored content |
| --- | --- | --- |
| `PreparedScene` | `restore_prepared_scene` | exact retained geometry |
| `DynamicMesh2d` | `restore_dynamic_mesh` | exact retained vertices/capacity |
| `ParticleField2d` | `restore_particle_field` | exact retained instances/budget |
| `ScalarFieldTexture` | `restore_scalar_field_texture` | exact retained scalar grid |
| `RenderTarget2d` | `restore_render_target` | empty target; redraw required |
| `TrailBuffer2d` | `restore_trail_buffer` | empty history; redraw required |

Do not call recovery as an ordinary quality switch. On affected Linux drivers,
old healthy devices are retained until renderer drop to avoid native swapchain
teardown crashes.

## 17. Error Strategy

- Construct required state with fallible constructors and `try_*` methods.
- Treat `RendererMismatch` as a host lifecycle bug.
- Treat capacity errors as a workload/device-limit negotiation failure.
- Treat invalid transforms as bad visual state; do not retry unchanged data.
- Retry transient skipped surface states on later event-loop turns.
- Recover a lost surface/device explicitly, then restore every retained
  resource and redraw GPU-only targets.
- Record tessellation dropped-command counts even when the frame presents.

No public success status should be interpreted as proof that optional ignored
commands were drawn. Use structured APIs for required visuals.

## 18. Performance Rules

- Build static geometry once with `PreparedScene`.
- Use `DynamicMesh2d` for ready changing triangles, not thousands of Scene
  primitives.
- Use `ParticleField2d` for circle/point populations.
- Put dense fields in textures; update only dirty regions.
- Render expensive gas/field layers below surface resolution when acceptable.
- Stagger visualization update cadence independently of presentation cadence.
- Fuse common multipass workloads with `render_layered_visualization`.
- Set particle memory/upload/visibility caps so rendering leaves budget for the
  Sim;X simulation.
- Measure no-VSync release builds and report CPU renderer time separately from
  surface acquisition and frame interval.

The renderer currently exposes CPU timings, not GPU timestamps.

## 19. Public API Catalogue

### Core math, color, and motion

| Types | Purpose |
| --- | --- |
| `Vec2`, `Rect` | 2D world math and bounds |
| `Color`, `Palette` | linear RGBA and default visual palette |
| `Easing`, `Interpolate`, `Tween<T>` | validated visual interpolation |

### Cameras and coordinate markers

| Types | Purpose |
| --- | --- |
| `Camera2d`, `Projection2d` | 2D view and lightweight scalar pseudo-depth |
| `LogicalScreenPosition`, `LogicalScreenVector` | typed logical-pixel values |
| `PhysicalScreenPosition`, `LogicalViewport` | surface input and viewport boundary |
| `Vec3`, `Rotation3d`, `Transform3d` | validated pseudo-3D model math |
| `Projection3d`, `Camera3d`, `ProjectedPoint3d` | perspective/orthographic CPU projection |

### Scene description

| Types | Purpose |
| --- | --- |
| `Scene`, `SceneCommand`, `DrawCommand`, `ScenePrimitive` | validated ordered visual command stream |
| `Circle`, `RectShape`, `Line`, `Polyline` | immutable accepted primitive data |
| `ShapeStyle`, `Fill`, `Stroke`, `Shadow` | primitive appearance |
| `LinearGradient`, `RadialGradient` | world-space fill sampling |
| `Layer`, `ScreenClipRect` | deterministic ordering and logical clipping |

### Scientific visual state

| Types | Purpose |
| --- | --- |
| `ScalarField`, `ColorStop`, `ColorMap` | finite dense grid and color mapping |
| `DynamicVertex2d`, `DynamicMesh2d` | mutable triangle-list GPU geometry |
| `ParticleInstance2d`, `ParticleField2d` | retained instanced circles/points |
| `ParticleRenderBudget`, `ParticleStatistics` | hard resource/culling limits and observability |
| `ScalarFieldTexture`, `ScalarFieldSampling` | GPU scalar texture and sampling contract |
| `LayeredVisualizationOptions` | fused scalar/particle composition settings |

### Renderer and composition

| Types | Purpose |
| --- | --- |
| `WgpuRenderer`, `WgpuRendererOptions`, `RendererPresentMode` | surface renderer lifecycle |
| `PreparedScene` | immutable retained scene geometry |
| `RenderTarget2d`, `RenderTargetLoad`, `BlendMode` | offscreen output and composition |
| `TrailBuffer2d` | bounded temporal ping-pong history |
| `RenderStatus`, `RenderReport`, `RendererFrameMetrics` | frame result and CPU telemetry |
| `TessellationStats` | accepted/rendered/dropped command accounting |

Every subsystem exposes a corresponding structured error enum. Consult rustdoc
for complete variants and exact method signatures:

```bash
cargo doc --all-features --no-deps --open
```

## 20. Examples And Verification

```bash
# General renderer and performance modes
cargo run --release --example demo

# Interactive Fluid, Gas, Wave, and Edge Case screens
cargo run --release --example ui_demo

# Bounded star-remnant visualization workload
cargo run --release --example star_remnant_stress -- --benchmark

# Deterministic host-side particle generation baseline
cargo run --release --example particle_cpu_benchmark

# Complete Linux release gate
./scripts/linux_release_gate.sh
```

Example-specific environment controls and benchmark caveats remain documented
in each example source and the repository changelog.
