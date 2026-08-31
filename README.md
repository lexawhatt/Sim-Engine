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
host-shaped glyph runs, explicitly budgeted dynamic triangles, richer bounded
2D strokes, and a named rendering benchmark matrix. See the
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

- validated circles, rectangles, rounded rectangles, lines, and polylines,
  including sub-ULP radii, rounded corners, and world-width strokes preserved
  as camera-relative local offsets;
- solid, linear-gradient, and radial-gradient fills;
- logical/world-width strokes with bounded caps, joins, dashes, markers,
  clipping, and stable layer ordering;
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
  state across logical-device replacement;
- primitive/source-grouped diagnostics and repeatable release-mode workloads
  for static UI, prepared/streaming UI, viewports, atlases, glyphs, mixed
  layers, budget rejection, DPI reconfiguration, real compositor scale
  transitions, and recovery.

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
let notify_window = window.clone();
renderer.set_pre_present_notify(move || notify_window.pre_present_notify());

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
cargo run --release --example rendering_benchmark_suite -- --fixture ui_90_10
cargo run --release --example stroke_gallery -- --uncapped
cargo run --release --example stereometry_3d -- --uncapped
cargo run --release --example cylinder_derivation_3d -- --uncapped --benchmark
```

Run the complete named performance/contract matrix on a Vulkan-capable Linux
machine with `./scripts/rendering_benchmark_matrix.sh`. Gated surface workloads
require a `1280x720` physical surface at scale `1.0`; both values are printed
in every result. Absolute timings are comparable only when surface extent,
scale, adapter, driver, backend, present mode, and workload are recorded
together; deterministic command/vertex/upload counters are portable.
The matrix first probes the real production surface. It pins the selected
high-performance Vulkan adapter by PCI bus address as well as backend, name,
vendor, and model, then requires the offscreen semantic oracle and every
surface/HiDPI process to use that same physical GPU. The oracle uses the
production surface format and its selected MSAA count. A nested compositor may
advertise a different surface format; its HiDPI evidence remains pinned by the
physical PCI identity rather than pretending two surfaces have identical
capabilities. Every warmup and
measured report must be `Drawn`; a timeout, occlusion, or outdated surface
fails the fixture instead of inflating throughput with skipped attempts.
Gated workloads use three independent 120-frame trials. Every trial must meet
the wall-throughput floor; the reported median is diagnostic only.
Renderer-work percentiles cover all 360 drawn frames, so one failed trial
cannot be hidden by two faster trials.

The matrix always enforces fixture-specific p95 ceilings for renderer work
excluding surface acquisition. Immediate additionally requires 60 presented
FPS. Mailbox/FIFO wall throughput must reach 95% of the confirmed current
monitor refresh, bounded to a 30-60 Hz release floor. The window is mapped by
an unmeasured drawn frame before any output metadata is used. Immediate does
not require refresh metadata; Mailbox/FIFO require a positive refresh rate
from the window's confirmed current monitor. Zero/unknown refresh and
primary/first-monitor fallbacks are never accepted for synchronized evidence.
Warmup and measured frames advance one `RedrawRequested` at a time, allowing
compositor events between every sample. A resize, scale transition, or changed
current-output identity invalidates the confirmation and discards partial
timing samples; a surface outside the fixed release extent/scale fails rather
than being measured as a cheaper workload. The benchmark registers
`Window::pre_present_notify` with the renderer. The callback runs after queue
submission and immediately before present only for the concrete FIFO, FIFO-
relaxed, or Mailbox modes; Immediate remains unpaced. The final measured
present then yields through a compositor-aware redraw boundary and repeats the
generation/output check before evidence is published.
Acquire percentiles remain reported but do not independently flip the verdict
from one scheduler-sensitive sample.
The matrix also drives a nested KWin compositor through a real scale
1.00-to-1.25 transition and records the paired event and successful redraw for
the exact revision on that same physical adapter.
The automatic transition requires `dbus-run-session`, `kwin_wayland`, and
`kscreen-doctor`; a missing executable fails the matrix instead of silently
skipping HiDPI evidence.
The wrapper, standalone matrix, and standalone HiDPI gate reject a dirty
worktree, then build and execute from a read-only detached worktree of the
captured commit with a separate Cargo target directory. They recheck the
calling checkout before accepting results. Replacement refs and legacy grafts
are rejected, and all Git operations run with replacement objects disabled.
Evidence therefore remains bound to Git object data for the exact recorded SHA
even if the mutable calling worktree changes while a long gate is running.
Child gates write only to hidden staging. The launcher atomically publishes a
completed bundle after the entire requested gate succeeds; failed, killed, or
crashed runs cannot leave a bundle that claims success. A nonblocking process
lock rejects concurrent release gates before either invocation can invalidate
or publish evidence.

`stroke_gallery` is the visual oracle for the v0.2 stroke contract. Pages 1-4
show every cap/join, half-alpha overlap probes, bounded animated dashes,
outward endpoint arrow markers, miter fallback, camera rotation, and the
accepted 0.005-world-unit line at zoom 10,000. Press `Space`, arrows, `+`/`-`, or `R`
to pause, scrub, zoom, and reset.

`ui_demo` includes a retained host-rasterized scientific glyph probe above all
four independently changing scene workloads. It demonstrates one-time atlas
and run creation without making the engine responsible for font selection or
text shaping.

## Verification

The v0.2.0 Linux release gate checks the declared Rust 1.90 MSRV, all targets
with and without the renderer, strict clippy, rustdoc, Vulkan semantic GPU
readback with a backend assertion, the Vulkan performance matrix on a real
surface, a transactional nested-compositor HiDPI transition, and the
publishable package boundary:

```bash
./scripts/linux_release_gate.sh
```

After all 11 steps pass, the launcher atomically publishes
`target/linux-release-evidence/`. Its `completion.txt` binds the successful
gate and exact SHA; the same directory contains `linux-vulkan-surface.txt`,
`linux-vulkan-adapter.txt`, `linux-vulkan-performance.txt`, and
`linux-hidpi-transition.txt`. The performance manifest preserves the complete
fixture counters, timings, thresholds, and passed verdicts for all seven
surface runs instead of leaving them only in terminal output.

## License

Licensed under either
[MIT](https://github.com/lexawhatt/Sim-Engine/blob/main/LICENSE) or
[Apache-2.0](https://github.com/lexawhatt/Sim-Engine/blob/main/LICENSE-APACHE),
at your option.
