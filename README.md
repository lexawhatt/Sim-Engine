# Sim;Engine

Sim;Engine is a reusable standalone Rust library for polished 2D rendering.

It is a visual engine, not an application-specific domain engine. Sim;X is the first major consumer, but Sim;Engine should be usable by other products that need rich 2D visuals. Application code owns physics, chemistry, biology, math domain rules, constants, entities, plugins, and simulation stepping. Sim;Engine receives already-computed visual state and renders it with camera control, tweening, scene commands, styles, and a `wgpu` backend.

## Why This Exists

The goal is to avoid the default "debug panel" look of immediate-mode GUI tools. Simulation visuals should feel smooth and product-ready by default: animated transitions, stable colors, readable composition, anti-aliased primitives, and explicit coordinate handling.

## Current Status

Implemented now:

- `Vec2` and `Rect`
- `Color` and default `Palette`
- sRGB byte-color conversion into linear rendering values
- `Fill`, `LinearGradient`, and `RadialGradient`
- `Easing`
- `Tween<T>`
- `Camera2d`
- world-space pan, cursor-centered logical-pixel zoom, and fit-to-bounds helpers
- `Projection2d`
- explicit logical and physical screen-position types
- `Scene` draw commands
- per-command pseudo-depth projection
- structured scene insertion errors
- `Layer` and stable layered draw order
- nested screen-space clipping
- finite-input validation before scene ordering and GPU upload
- `WgpuRenderer`
- GPU camera uniforms for camera-independent world geometry
- `PreparedScene` for reusable GPU-resident geometry
- `ParticleField2d` for instanced circle particles with retained recovery data
- `ScalarField` and validated piecewise-linear `ColorMap` data contracts
- `VectorField` with finite rectangular velocity/flow samples
- `ScalarFieldTexture` for renderer-owned `R32Float` scalar uploads and recovery
- full-viewport color-mapped scalar heatmap rendering
- `RenderTarget2d` for offscreen heatmaps and explicit multi-pass composition
- `BlendMode::{Alpha, Additive, Replace}` for target-to-surface composition
- validated logical-to-physical display scale for HiDPI targets
- live `winit` demo in `examples/demo.rs`
- randomized four-wave matching game in `examples/ui_demo.rs`

Supported first-slice primitives:

- circles
- circle strokes
- rectangles
- rounded rectangles
- lines with round caps
- polylines with round joins
- simple screen-space shadows
- solid, linear gradient, and radial gradient fills
- background/default/foreground layer ordering
- logical-pixel clipping converted to physical GPU scissor batches

## Example

```rust
use sim_engine::{
    Camera2d, Color, Fill, LinearGradient, Rect, Scene, ScreenClipRect, ShapeStyle, Vec2,
};

let camera = Camera2d::new(Vec2::ZERO, 2.0)?;
let mut scene = Scene::new(Color::rgb8(12, 14, 18))?;

scene.circle(
    Vec2::new(10.0, 20.0),
    8.0,
    ShapeStyle::filled(Color::rgb8(86, 195, 255)),
);

scene.rect(
    Rect::from_center_size(Vec2::ZERO, Vec2::new(120.0, 64.0)),
    8.0,
    ShapeStyle::filled_with(Fill::LinearGradient(LinearGradient::new(
        Vec2::new(-60.0, 0.0),
        Vec2::new(60.0, 0.0),
        Color::rgb8(86, 195, 255),
        Color::rgb8(255, 190, 94),
    ))),
);

scene.with_screen_clip(
    ScreenClipRect::from_min_size(Vec2::new(40.0, 40.0), Vec2::new(720.0, 420.0))?,
    |scene| {
        scene.line(
            Vec2::new(-1_000.0, 0.0),
            Vec2::new(1_000.0, 0.0),
            2.0,
            Color::WHITE,
        );
    },
)?;

scene.with_depth(4.0, |scene| {
    scene.circle(
        Vec2::new(28.0, -12.0),
        6.0,
        ShapeStyle::filled(Color::WHITE),
    );
})?;
```

Screen clips use logical pixels with a top-left origin. Commands capture the
active clip when they are appended, nested clips are intersected, and the
renderer converts them to physical scissor pixels using its display scale factor.

Camera zoom, stroke widths, shadows, and clips use logical pixels. Surface width
and height remain physical pixels. Camera picking requires a
`LogicalScreenPosition`; use the renderer's `physical_to_logical_screen` method
when a host event supplies `PhysicalScreenPosition`. `WgpuRenderer::new` assumes
scale `1.0`; HiDPI hosts should construct `WgpuRendererOptions` with the window
scale factor.

Scene primitive methods return `true` when a command is accepted. They return
`false` without changing command order when geometry, dimensions, colors, or
styles are non-finite or otherwise non-drawable. The corresponding `try_*`
methods return `SceneError` when the host needs a precise rejection reason.

Window setup is intentionally outside the core scene API. The demo uses `winit`, but Sim;Engine should not force a host app to use a specific app framework.

Set `SIM_ENGINE_DYNAMIC_MESH_DEMO=1` when running `demo` to exercise the
`DynamicMesh2d` path: the animated wave ribbon is updated as one mutable
triangle-list mesh rather than rebuilt as scene primitives each frame. Its
once-per-second diagnostics include dynamic update CPU time and buffer-growth
events are exposed by `DynamicMeshUpdateReport`.

For a repeatable dynamic-mesh measurement, run a fixed warm-up and measured
frame count. The fixture exits by itself and prints a final aggregate interval:

```bash
SIM_ENGINE_DYNAMIC_MESH_BENCHMARK_FRAMES=600 \
SIM_ENGINE_DYNAMIC_MESH_BENCHMARK_WARMUP_FRAMES=120 \
SIM_ENGINE_DYNAMIC_MESH_SEGMENTS=10000 \
SIM_ENGINE_PRESENT_MODE=no-vsync \
cargo run --release --example demo
```

The benchmark mode always selects the dynamic-mesh path. Segment count is
bounded to 1 through 1,000,000 (six triangle-list vertices per segment); use a
count that fits the target machine's CPU and GPU memory. It measures CPU-side
dynamic update and renderer stages, not completed GPU time or display scanout.

Set `SIM_ENGINE_PARTICLE_DEMO=1` to render 1,500 animated particles through a
single instanced circle draw call. `ParticleField2d` validates every instance
before upload, reuses its instance-buffer capacity across updates, and exposes
submitted/visible/culled/dropped/rendered counters. It culls circles wholly
outside the logical viewport before issuing the instance draw. This is a separate visualization
mode, so it takes precedence over the prepared-scene and dynamic-mesh demo
switches. `update_particle_field_range` replaces a contiguous subset in place;
`ParticleFieldUpdateReport` exposes its CPU upload time and whether a full
replacement grew the GPU buffer.

Particle measurement scenarios use the same deterministic particle distribution
at 10k, 100k, or 1M instances. For actual renderer/surface measurements, choose
the count explicitly and retain the once-per-second renderer diagnostics:

```bash
SIM_ENGINE_PARTICLE_DEMO=1 SIM_ENGINE_PARTICLE_COUNT=10000 \
SIM_ENGINE_PRESENT_MODE=no-vsync cargo run --release --example demo
```

Replace `10000` with `100000` or `1000000` only when the target machine has
sufficient memory. `SIM_ENGINE_PARTICLE_COUNT` is bounded to 1 through 1M.
For deterministic CPU-only fallback numbers on CI or without a window/GPU:

```bash
SIM_ENGINE_PARTICLE_CPU_BENCHMARK_COUNT=100000 \
SIM_ENGINE_PARTICLE_CPU_BENCHMARK_FRAMES=300 \
cargo run --release --example particle_cpu_benchmark
```

The CPU example measures only host-side deterministic generation and
`ParticleInstance2d` validation; it intentionally does not imply GPU upload,
culling, rasterization, or presentation throughput.

Set `SIM_ENGINE_HEATMAP_DEMO=1` to render an animated 160×96 scalar field as a
full-viewport color-mapped heatmap. This mode takes precedence over particle,
dynamic-mesh, prepared, and streaming demo modes.

For subregions that change independently, use
`update_scalar_field_texture_region`; it validates the rectangular range and
values, updates the retained grid, and uploads only that texture region.
`render_scalar_field_texture_with_sampling` makes heatmap sampling explicit:
`Nearest` is exact texel sampling and `Linear` performs deterministic bilinear
interpolation in WGSL for `R32Float` fields.

`RenderTarget2d` dimensions are physical texture pixels. Create one with
`create_render_target`, render a heatmap with
`render_scalar_field_texture_to_target`, then present it with
`compose_render_target`. Composition validates renderer ownership, opacity,
and background, and explicitly selects `Alpha`, `Additive`, or `Replace`.
Targets retain GPU pixels only, so redraw them after recreating a renderer.
`allocation_bytes` exposes format-aware target memory use; `TrailBuffer2d`
reports the sum for its two textures. `restore_render_target` and
`restore_trail_buffer` intentionally restore empty, cleared resources, making
the missing visual redraw step explicit after device recovery.

For bounded temporal history, `TrailBuffer2d` owns two ping-pong targets.
`accumulate_trail_buffer` takes retained-history and fresh-source opacities in
`0.0..=1.0`, rejects feedback aliases, and swaps only after GPU submission.
`clear_trail_buffer` clears both targets deterministically. Set both
`SIM_ENGINE_HEATMAP_DEMO=1` and `SIM_ENGINE_HEATMAP_TRAILS=1` to view the
animated heatmap through this temporal path.

Set `SIM_ENGINE_VECTOR_FIELD_DEMO=1` to render an animated finite vector grid
as clipped arrow glyphs. It is a scene-based consumer intended for velocity and
flow data; heatmap/vector composition will use the dedicated render-target API
introduced in the next rendering stage.

## Run The Demo

```bash
cargo run --example demo
```

The demo opens a window, creates a `WgpuRenderer`, animates the camera, and renders a small visual scene.

The UI interaction example uses only Sim;Engine primitives for its visual layer:

```bash
cargo run --release --example ui_demo
```

Use `cargo run --release --example ui_demo -- --solved-preview` to inspect the
success state without manually matching all eight parameters.

Each pane has a colored frequency slider and a moving gray dashed target. Every
wave has an independent speed; four selector buttons choose which speed the
single red bottom slider edits. Pause freezes both target and colored waves at
their current phase, while the adjacent stop/reset button also returns their
shared phase to zero. Accuracy is capped at `99%`, and a sufficiently close
match reveals `YAY`, accept, and dismiss controls. Targets and starting values
are randomized for every process restart. Hit testing and game state stay in
the host example; Sim;Engine receives the resulting visual scene.

Static geometry can be prepared once and drawn under changing cameras and
viewport dimensions without per-frame tessellation or geometry upload:

```rust
let prepared = renderer.prepare_scene(&scene);
renderer.render_prepared(&prepared, &camera)?;
```

Prepared geometry is an immutable snapshot tied to the renderer that created
it. Shape, style, order, background, or logical clip changes require preparing a
replacement. Viewport-relative clips also need rebuilding when the host wants
their authored bounds to follow a resize. Each snapshot retains a CPU vertex
copy so `restore_prepared_scene` can upload it to a replacement renderer after
device loss without retaining the original high-level `Scene`.

The default `Vsync` mode requests strict FIFO presentation. The demo prints a
CPU timing breakdown for scene construction, tessellation, upload, surface
acquisition, and submit/present dispatch. `idle/scheduler` is the remaining
frame interval outside scene construction and the renderer call. On Wayland it
can include compositor frame-callback pacing requested by `winit`.

These metrics do not report GPU completion or the monitor scanout timestamp.
Generic `wgpu` surface presentation does not expose that timestamp through this
renderer API.

For renderer throughput measurements without the monitor refresh cap:

```bash
SIM_ENGINE_PRESENT_MODE=no-vsync cargo run --release --example demo
```

The demo skips `winit` pre-present pacing in this mode so Wayland frame
callbacks do not cap the throughput measurement. `NoVsync` requests Immediate,
then Mailbox, and can fall back to FIFO when the platform supports neither.

To exercise the retained GPU geometry path:

```bash
SIM_ENGINE_PREPARED_SCENE=1 SIM_ENGINE_PRESENT_MODE=no-vsync cargo run --release --example demo
```

Current implementation status, limitations, and development priorities are
recorded in `READ_FIRST_SIM_ENGINE_ROADMAP.md`.

## License

Sim;Engine is available under either of the following licenses, at your option:

- [MIT License](LICENSE)
- [Apache License, Version 2.0](LICENSE-APACHE)

## Verified Commands

```bash
cargo fmt
cargo test --all-features
cargo test --no-default-features
cargo check --example demo
cargo clippy --all-targets --all-features -- -D warnings
```
