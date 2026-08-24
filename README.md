# Sim;Engine

Sim;Engine is the visual rendering layer designed around Sim;X. It provides a
validated, high-performance 2D scene and scientific-visualization renderer,
with a focused pseudo-3D stereometry path under development for Sim;Math.

Sim;X is the primary product consumer and design driver. The engine remains
reusable: hosts provide already-computed visual state, while Sim;Engine owns
projection, camera control, drawing, clipping, GPU resources, composition,
recovery, and rendering diagnostics. Physics, chemistry, biology, mathematical
meaning, simulation stepping, UI navigation, and plugins remain in the host.

The crate is pre-1.0. The first supported release target is Linux and the
minimum supported Rust version is 1.87.

## Documentation

- [Integration Guide](docs/INTEGRATION_GUIDE.md) - installation, core concepts,
  every rendering path, recovery, performance guidance, and API catalogue.
- [Architecture Reference](docs/ARCHITECTURE_REFERENCE.md) - coordinate/color
  contracts, CPU/GPU data flow, ownership, validation, and extension rules.
- [Documentation Index](docs/README.md) - stability, release, changelog, and
  pseudo-3D specification links.
- Generated Rust API reference:

  ```bash
  cargo doc --all-features --no-deps --open
  ```

## Capabilities

- validated circles, rectangles, rounded rectangles, lines, and polylines;
- solid, linear-gradient, and radial-gradient fills;
- logical-pixel strokes, shadows, clipping, and stable layer ordering;
- 2D camera pan, rotation, zoom, cursor anchoring, fit, and picking;
- fallible tweening and easing;
- streaming and prepared Scene geometry;
- mutable triangle-list `DynamicMesh2d` resources;
- instanced `ParticleField2d` with draw, memory, upload, and culling budgets;
- finite `ScalarField` data, color maps, partial texture updates, and heatmaps;
- offscreen targets, explicit blend modes, ping-pong trails, and fused
  field/particle/surface composition;
- device/surface recovery with explicit retained-resource restoration;
- CPU frame-stage metrics and semantic GPU readback tests;
- validated `Vec3`, `Rotation3d`, `Transform3d`, `Projection3d`, and `Camera3d`
  foundation.

Retained 3D meshes, depth-tested surfaces, and hidden-line rendering are not yet
implemented. `Projection2d` is a lightweight 2.5D presentation effect, not a
replacement for the planned stereometry pipeline.

## Quick Start

The default feature set includes `wgpu`:

```toml
[dependencies]
sim-engine = "0.1"
```

Use CPU visual-state APIs without the renderer:

```toml
[dependencies]
sim-engine = { version = "0.1", default-features = false }
```

Build a scene:

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

# Ok::<(), Box<dyn std::error::Error>>(())
```

Window creation stays in the host. Construct `WgpuRenderer` asynchronously from
a compatible surface target, then render synchronously in the event loop:

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

let report = renderer.render_with_metrics(&scene, &camera)?;
```

See the [Integration Guide](docs/INTEGRATION_GUIDE.md) for HiDPI conversion,
resource updates, errors, recovery, and complete examples.

## Rendering Paths

| Workload | Preferred path |
| --- | --- |
| Small changing visual | `Scene` |
| Static shapes with moving camera | `PreparedScene` |
| Frequently changing triangles | `DynamicMesh2d` |
| Large circle/point population | `ParticleField2d` |
| Dense scalar grid | `ScalarFieldTexture` |
| Bounded gas plus particle overlay | `render_layered_visualization` |

The renderer is asynchronous only for adapter/device creation and explicit
recovery. Frame updates and rendering are synchronous submission methods; the
engine does not own the host simulation scheduler.

## Examples

```bash
# General primitives and renderer modes
cargo run --release --example demo

# Fluid, Gas, Wave, and Edge Case interactive screens
cargo run --release --example ui_demo

# Bounded star-remnant visualization workload
cargo run --release --example star_remnant_stress -- --benchmark

# CPU-only particle-state baseline
cargo run --release --example particle_cpu_benchmark
```

The star-remnant fixture demonstrates the primary resource rule: rendering a
large Sim;X state must have explicit visible-instance, upload, visibility-check,
texture-resolution, and memory budgets so simulation work retains headroom.

## Release Verification

Run the complete Linux gate:

```bash
./scripts/linux_release_gate.sh
```

It checks formatting, all targets/features, the core-only build, strict clippy,
semantic GPU readback, whitespace, and an offline package rebuild. See the
[release checklist](docs/RELEASING.md) for required hardware evidence.

## License

Licensed under either of the following, at your option:

- [MIT](https://github.com/lexawhatt/Sim-Engine/blob/main/LICENSE)
- [Apache-2.0](https://github.com/lexawhatt/Sim-Engine/blob/main/LICENSE-APACHE)
