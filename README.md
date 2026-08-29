# Sim;Engine

Sim;Engine is a validated visualization library built for simulation products.
It provides a high-performance 2D renderer, scientific visualization paths,
and a focused retained 3D stereometry pipeline.

Sim;X is the primary consumer and design driver, but the crate is reusable.
Applications provide ready visual state. Sim;Engine owns cameras, drawing,
clipping, interpolation, GPU resources, composition, recovery, and rendering
diagnostics. Physics, simulation stepping, domain entities, UI navigation, and
plugins remain in the host application.

Version **0.2.0** is the current official Sim;Engine release. The crate remains
pre-1.0, and its supported release target is Linux with Vulkan. A concrete
adapter/driver is supported when the mandatory semantic fixture passes on it;
untested drivers are not inferred from Mesa evidence. The minimum supported
Rust version is 1.90.

Version 0.2.0 adds the bounded Sim;X integration foundation: fixed-screen
scenes, positioned 2D viewports, offscreen scene rendering, a heterogeneous
single-present frame composer, retained RGBA images and atlas batches,
host-shaped glyph runs, and explicitly budgeted dynamic triangles. See the
[0.2.0 changelog](CHANGELOG.md#020---2026-08-29) for the complete delta from
0.1.0.

## Documentation

- [Library documentation](DOCUMENTATION.md) - installation, concepts, every
  rendering path, recovery, performance, architecture, and examples.
- [Changelog](CHANGELOG.md) - release history and user-visible changes.
- Generated API reference:

  ```bash
  cargo doc --all-features --no-deps --open
  ```

## Capabilities

- validated circles, rectangles, rounded rectangles, lines, and polylines;
- solid, linear-gradient, and radial-gradient fills;
- logical-pixel strokes, shadows, clipping, and stable layer ordering;
- 2D camera pan, rotation, zoom, fit, and picking;
- fallible tweening and easing;
- streaming, prepared, and dynamic triangle geometry;
- retained sRGB RGBA images, atlas sprite batches, and world-space image quads;
- host-shaped, bounded glyph atlas runs with deterministic logical bounds;
- budgeted instanced particle rendering;
- scalar-field textures, color maps, partial updates, and heatmaps;
- offscreen targets, composition, and bounded trails;
- device recovery with explicit retained-resource restoration;
- typed logical-pixel, 3D world-length, and target-scale boundaries;
- retained 3D meshes, independent transforms, hardware depth, and visible or
  dashed hidden mathematical edges;
- atomic retained-3D scene recovery that preserves stable object IDs and visual
  state across logical-device replacement.

Ordinary scenes support explicit construction, tessellation, and upload
budgets, fixed logical-screen geometry, and independent positioned viewport or
offscreen-target rendering. `FrameComposer` orders streaming and prepared
scenes, dynamic meshes, particles, scalar fields, images, glyph runs, and 2D/3D
color targets under one frame budget and one surface presentation.
`DynamicMeshBudget` limits caller-provided filled triangles and makes full
updates allocation-fallible and atomic.

Translucent section materials, hatching, projected 3D anchors, and 3D picking
are not part of v0.2.0.

## Installation

The default feature set includes the `wgpu` renderer:

```toml
[dependencies]
sim-engine = "0.2"
```

Use core visual-state APIs without GPU dependencies:

```toml
[dependencies]
sim-engine = { version = "0.2", default-features = false }
```

## Quick Start

```rust
use sim_engine::{Camera2d, Color, Rect, Scene, ShapeStyle, Vec2};

let camera = Camera2d::new(Vec2::ZERO, 2.0)?;
let mut scene = Scene::new(Color::rgb8(12, 14, 18))?;

scene.try_circle(
    Vec2::new(10.0, 20.0),
    8.0,
    ShapeStyle::filled(Color::rgb8(86, 195, 255)),
)?;
scene.try_rect(
    Rect::from_center_size(Vec2::ZERO, Vec2::new(120.0, 64.0)),
    8.0,
    ShapeStyle::fill_stroke(
        Color::rgb8(24, 31, 44),
        2.0,
        Color::rgb8(86, 195, 255),
    ),
)?;

# let _ = camera;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Window creation stays in the host. Renderer initialization is asynchronous;
steady-state updates and rendering are synchronous submissions:

```rust,ignore
let options = WgpuRendererOptions::new(
    RendererPresentMode::Vsync,
    window.scale_factor(),
)?;
let mut renderer = WgpuRenderer::new_with_options(
    window.clone(),
    width,
    height,
    options,
).await?;

let mut frame = renderer.begin_frame(scene.background(), FrameBudget::default())?;
frame.draw_scene(&scene, camera, FramePassOptions::default())?;
let report = frame.present()?;
```

## Choose A Rendering Path

| Workload | Preferred API |
| --- | --- |
| Small changing scene | `Scene` |
| Static geometry with a moving camera | `PreparedScene` |
| Frequently changing triangles | `DynamicMesh2d` |
| Images and atlas sprites | `Image2d` plus `ImageBatch2d` |
| Host-shaped scientific text | `GlyphAtlas2d` plus `GlyphRun2d` |
| Large point or circle population | `ParticleField2d` |
| Dense scalar grid | `ScalarFieldTexture` |
| Mixed UI, viewports, fields, and targets | `FrameComposer` |
| Bounded gas plus particle overlay | `render_layered_visualization` |
| Stereometry solids and hidden edges | `Mesh3d` plus `Scene3d` |

## Examples

```bash
cargo run --release --example demo
cargo run --release --example ui_demo -- --uncapped
cargo run --release --example star_remnant_stress -- --benchmark
cargo run --release --no-default-features --example particle_cpu_benchmark
cargo run --release --no-default-features --example scene_construction_benchmark
cargo run --release --example stereometry_3d -- --uncapped
cargo run --release --example cylinder_derivation_3d -- --uncapped --benchmark
```

`ui_demo` includes a retained host-rasterized scientific glyph probe above all
four independently changing scene workloads. It demonstrates one-time atlas
and run creation without making the engine responsible for font selection or
text shaping.

## Verification

The v0.2.0 Linux release gate checks the declared Rust 1.90 MSRV, all targets with
and without the renderer, strict clippy, rustdoc, Vulkan semantic GPU readback
with a backend assertion, and the publishable package boundary:

```bash
./scripts/linux_release_gate.sh
```

The gate records the exact adapter/backend/driver run in
`target/linux-vulkan-adapter.txt`; CI publishes the same file as an artifact.

## License

Licensed under either
[MIT](https://github.com/lexawhatt/Sim-Engine/blob/main/LICENSE) or
[Apache-2.0](https://github.com/lexawhatt/Sim-Engine/blob/main/LICENSE-APACHE),
at your option.
