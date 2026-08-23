# Sim;Engine implementation notes

## 2026-08-22 - Stage 1 dynamic geometry foundation

Implemented the first Stage 1 slice: `DynamicMesh2d`, a renderer-owned mutable
triangle-list mesh for frequently changing simulation visuals.

- `DynamicVertex2d` validates finite world position, pseudo-depth, and linear
  color at the API boundary.
- `WgpuRenderer::create_dynamic_mesh` uploads initial triangle-list data.
- `update_dynamic_mesh` replaces mesh data while retaining GPU capacity until
  growth is required; `update_dynamic_mesh_range` updates triangle-aligned
  ranges in place.
- Dynamic mesh rendering has its own API and metrics flag so streamed geometry
  is distinguishable from scene tessellation and prepared-scene reuse.
- CPU snapshots are retained for validation, extent checks, and later device-loss
  restoration work.

Verified with the crate's feature and no-default-feature test suites, formatting,
clippy, and whitespace checks.

Next: exercise the API in a live simulation-style example and add explicit
dynamic-mesh recovery after renderer recreation.

## 2026-08-22 - Stage 1A completed

- Added `WgpuRenderer::restore_dynamic_mesh`, preserving retained CPU vertices
  and the source capacity after renderer recreation.
- Added an offscreen GPU restoration regression test covering identity,
  capacity, and retained-memory behavior.
- Added `SIM_ENGINE_DYNAMIC_MESH_DEMO=1` to the demo. It renders an animated
  wave ribbon through repeated dynamic-mesh updates, demonstrating a concrete
  per-frame simulation visualization consumer.

Next: Stage 1B adds repeatable dynamic-update measurements and clearer runtime
diagnostics before work starts on instanced particles.

## 2026-08-22 - Stage 1B measurement foundation

- Added `DynamicMeshUpdateReport` with CPU update duration, vertex count, and
  buffer-growth information for replacement and partial-range writes.
- Dynamic demo diagnostics now report average mesh update CPU time separately
  from scene tessellation and frame submission.

Next: make the dynamic demo's measurement scenario reproducible as a benchmark
fixture, then begin the validated instanced-particle data model.

## 2026-08-22 - Stage 2A particle input boundary

- Added `ParticleInstance2d`, whose constructor validates finite world position,
  logical-pixel radius, linear color, and pseudo-depth before data can reach an
  instance buffer.
- Added `ParticleStatistics` as the common submitted/visible/dropped/rendered
  diagnostics shape that the upcoming particle field will populate.

Next: add the instanced circle GPU pipeline and a renderer-owned particle field
that reports these statistics from actual updates and draw calls.

## 2026-08-22 - Stage 2A instanced particle field

- Added `ParticleField2d`: a renderer-owned, CPU-recoverable collection of
  validated particle instances backed by one capacity-reusing GPU instance
  buffer.
- Added creation, full replacement update, restoration, and render APIs. The
  renderer rejects cross-renderer fields and non-finite camera output instead
  of silently disappearing geometry.
- Added a dedicated instanced circle pipeline. WGSL projects each instance with
  the existing camera/depth contract, applies radius in logical pixels, and
  discards square-quad corners in the fragment stage.
- Particle statistics now accurately describe the current no-culling slice:
  submitted equals visible, dropped is zero for valid input, and rendered is
  set only after a successful draw submission.
- Added `SIM_ENGINE_PARTICLE_DEMO=1`, which animates 1,500 particles in one
  instanced draw call, and expanded the offscreen readback test to verify the
  particle GPU path by pixels rather than shader validation alone.

Next: finish the deferred Stage 1B repeatable dynamic-mesh benchmark fixture,
then extend particle fields with partial updates and camera-aware culling.

## 2026-08-22 - Stage 2B partial particle updates

- Added metrics-bearing full particle updates and contiguous partial updates.
  Full replacement preserves instance-buffer capacity until growth; partial
  replacement never reallocates.
- Added checked range validation and a structured `UpdateRangeOutOfBounds`
  failure, so invalid ranges cannot become incomplete GPU state.
- Added `ParticleFieldUpdateReport` for update CPU time, current counters, and
  buffer-growth visibility, plus regression coverage for valid and invalid
  range boundaries.

Next: finish the Stage 1B repeatable benchmark fixture, then add camera-aware
particle culling and field-specific offscreen coverage.

## 2026-08-22 - Stage 2B camera-aware particle culling

- Particle rendering now filters the retained field against the active logical
  viewport before the instance upload and draw. A circle touching the boundary
  remains visible; only wholly offscreen circles are culled.
- `ParticleStatistics` gained a separate `culled` counter, while non-finite
  transform arithmetic remains a structured rendering error rather than a
  culling outcome.
- Culling uploads are included in the normal geometry-upload timing, and a
  camera/viewport regression test protects the boundary behavior.

Next: finish Stage 1B's repeatable benchmark fixture, then add field-owned
offscreen readback coverage to close Stage 2B.

## 2026-08-22 - Stage 2B offscreen culling coverage

- Factored the production visibility calculation and used it in the offscreen
  GPU test before the instance-buffer upload.
- The test now submits one visible and one fully offscreen particle, verifies
  that only the visible instance survives culling, and still reads its circle
  pixels and discarded corner back from the GPU.
- Stage 2B is complete: instance capacity reuse, full and partial updates,
  camera/depth projection, viewport culling, diagnostics, and GPU pixel
  coverage are all present.

Next: complete Stage 1B's repeatable dynamic-mesh benchmark fixture, then start
Stage 2C particle measurement scenarios.

## 2026-08-22 - Stage 1B repeatable dynamic-mesh fixture

- Added a demo benchmark mode with fixed warm-up and measured frame counts. It
  forces the dynamic-mesh path, prints its final aggregate metrics, and exits
  automatically.
- Added a bounded `SIM_ENGINE_DYNAMIC_MESH_SEGMENTS` workload parameter: 1 to
  1,000,000 segments, each producing six triangle-list vertices.
- Documented a no-VSync release command and clarified that the fixture measures
  CPU update/renderer stages, not asynchronous GPU completion or scanout.

Next: start Stage 2C particle measurement scenarios with a deterministic CPU
fallback, while keeping the remaining Stage 1 renderer-level limits queued.

## 2026-08-22 - Stage 2C particle measurement foundation

- Added `particle_cpu_benchmark`, a deterministic, windowless fallback for
  measuring host-side particle generation and `ParticleInstance2d` validation.
  It deliberately makes no GPU throughput claim.
- The live particle demo now accepts `SIM_ENGINE_PARTICLE_COUNT` from 1 through
  1,000,000, enabling the documented 10k/100k/1M no-VSync renderer scenarios.
- Added documented commands for both scope-separated measurements and smoke-ran
  the 10k CPU scenario.

Next: collect hardware-specific renderer baselines when available, then begin
Stage 3A scalar-field data and color-map APIs.

## 2026-08-22 - Stage 3A scalar-field data contract

- Added `ScalarField`: a finite, non-zero, checked rectangular scalar grid with
  bounds-checked cell mutation and replacement operations.
- Added validated normalized `ColorStop` and piecewise-linear `ColorMap` APIs
  operating in linear RGBA space.
- Added regression coverage for invalid dimensions/values and color-map
  ordering, clamping, interpolation, and non-finite samples.

Next: upload scalar fields as renderer-owned textures and add a heatmap demo.

## 2026-08-22 - Stage 3A scalar texture upload

- Added `ScalarFieldTexture`, a renderer-owned `R32Float` resource retaining
  the validated CPU grid for device-loss restoration.
- Full replacement uploads reuse texture allocation until dimensions change;
  `ScalarFieldUploadReport` exposes upload CPU time and recreation.
- Extended the real-GPU validation test to create, upload, restore, and compare
  scalar grid resources.

Next: add the color-mapped heatmap render path and a demo consumer.

## 2026-08-22 - Stage 3A heatmap render pipeline

- Added `render_scalar_field_texture` and metrics variant: a full-logical-
  viewport heatmap with explicit value range and finite background contracts.
- The WGSL path uses exact `R32Float` texel loads and a transient 256-entry
  linear-RGBA color-map LUT, avoiding ambiguous scalar filtering.
- The shared real-GPU pipeline test now validates the heatmap shader and its
  texture binding layout.

Next: add a demo consumer and offscreen heatmap pixel semantics.

## 2026-08-22 - Stage 3A heatmap demo consumer

- Added `SIM_ENGINE_HEATMAP_DEMO=1`, which updates a 160×96 finite scalar grid
  and its renderer-owned texture every frame before color-mapped rendering.
- Heatmap mode is intentionally exclusive and takes precedence over the other
  demo geometry modes, preserving the one-clear/one-present render contract.

Next: add real offscreen pixel assertions for scalar-range and color-map
semantics.

## 2026-08-22 - Stage 3A heatmap pixel semantics

- Added a dedicated 2×2 offscreen heatmap render target and readback to the
  real-GPU integration test.
- The test proves exact scalar texel addressing and color-map mapping at 0,
  0.25, 0.5, and 1.0, including expected sRGB target encoding for intermediate
  linear values.
- Stage 3A is complete: validated grid data, color map, texture upload,
  full-viewport heatmap renderer, demo consumer, and GPU pixel semantics.

Next: Stage 3B incremental updates, filtering/color-space contracts, and a
vector-field overlay consumer.

## 2026-08-22 - Stage 3B incremental scalar updates

- Added checked rectangular replacement to `ScalarField` and matching partial
  GPU texture upload through `update_scalar_field_texture_region`.
- The operation retains texture allocation, synchronizes the recovery snapshot,
  and rejects bad bounds/counts/non-finite values structurally.

Next: explicit heatmap sampling and a vector-field overlay consumer.

## 2026-08-22 - Stage 3B scalar sampling contract

- Added public nearest and manual-linear sampling modes for heatmaps.
- Linear interpolation executes explicitly in WGSL, which keeps behavior stable
  even where `R32Float` cannot use a filtering sampler.

Next: color-space probe and vector-field overlay before composition work.

## 2026-08-22 - Stage 3B field visualization completion

- Added `SIM_ENGINE_VECTOR_FIELD_DEMO=1`: animated clipped arrow glyphs are a
  concrete standalone consumer of host-owned vector samples. Contract review
  later removed the temporary public `VectorField` model because it was not a
  renderer input and imposed simulation-grid ownership on the engine.
- Scalar regions have regression coverage for successful, out-of-bounds,
  count, and non-finite mutations.
- The real-GPU test exercises manual-linear heatmap sampling and sRGB target
  conversion rather than only validating pipeline creation.

Verified: `cargo fmt --check`, 61 all-feature tests, 32 no-default-feature
tests, strict clippy, and `git diff --check`.

Stage 3B is complete. Next: Stage 4A composition targets and render-to-texture.

## 2026-08-22 - Stage 4A composition targets

- Added renderer-owned `RenderTarget2d` resources in explicit physical texture
  pixels. `render_scalar_field_texture_to_target` supplies the first
  render-to-texture API; `compose_render_target` presents it with validated
  `Alpha`, `Additive`, or `Replace` behavior.
- Migrated the heatmap demo to the real two-pass path (field → target →
  surface) and recreate its target after a resize.
- Expanded the real-GPU integration test to render into a heatmap target and
  sample that target in a second composition pass before pixel readback.

Verified: formatting, example check, 61 all-feature tests, 32 no-default-
feature tests, strict clippy, and diff check.

Stage 4A is complete. Next: bounded temporal accumulation and reset semantics.

## 2026-08-22 - Stage 4B temporal composition

- Added initialized ping-pong `TrailBuffer2d` resources. Accumulation applies
  a bounded history opacity and fresh-source opacity, rejects source aliases,
  swaps only after submission, and exposes a deterministic two-target clear.
- Added `SIM_ENGINE_HEATMAP_TRAILS=1` as a concrete animated temporal consumer
  on top of the 4A heatmap target path.
- Extended the offscreen GPU readback with 0.5 retained history plus 0.5 fresh
  source; the observed sRGB pixel locks the actual two-pass temporal math.

Verified: formatting, example check, 61 all-feature tests, 32 no-default-
feature tests, strict clippy, and diff check.

Stage 4B is complete. Next: Stage 5A camera helpers and coordinate tests.

## 2026-08-22 - Stage 5A camera interaction helpers

- Added atomic `Camera2d::pan_by`, `zoom_about_screen`, and `fit_to_bounds`.
  Cursor anchors stay in logical-pixel space, while the fit calculation accounts
  for the current depth-zero tilt and screen rotation.
- Added regression tests for anchor preservation, transformed bounds, and all
  invalid paths retaining the original camera state.

Verified: 63 all-feature tests.

Stage 5A is complete. Next: deterministic visual QA and recovery coverage.

## 2026-08-22 - Stage 6A target recovery/accounting slice

- `RenderTarget2d` and `TrailBuffer2d` now expose exact format-aware texture
  allocation bytes. Target creation validates device dimensions and allocation
  representability before it calls into wgpu.
- Added explicit empty restore APIs for targets and trails: they document that
  GPU pixels do not survive recovery and force the host to redraw rather than
  accidentally relying on stale presentation state.
- Added an allocation regression test for 8-bit and 16-bit color formats.

Verified: 64 all-feature tests, 34 no-default-feature tests, strict clippy,
formatting, and diff check.

6A remains in progress for platform/independent-renderer matrix coverage.

## 2026-08-22 - Adversarial renderer contract repair

Resolved the release-blocking review before continuing feature work.

- Corrected manual bilinear Y addressing. The GPU fixture now reads all four
  2x2 texel centers, so a vertically mirrored shader can no longer pass through
  a symmetric average.
- Made offscreen target/trail storage consistently premultiplied, including
  clear colors and `Replace`, then unpremultiply once at composition. A
  half-alpha red readback distinguishes the correct 0.5 contribution from the
  former double-alpha 0.25 result.
- Reject scalar ranges whose finite endpoints overflow on subtraction and
  scalar textures beyond the active device's real 2D texture limit.
- Restored layer/insertion command ordering. Pseudo-depth now affects camera
  projection only and cannot silently couple overlap to `depth_scale` sign.
- Scene validation now rejects derived overflow in circles, rectangles, line
  deltas, polyline segments, linear gradients, and doubled shadow spread.
- Tween scalar/vector interpolation uses a bounded `f64` intermediate so
  opposite `f32` extrema remain finite.
- Removed the engine-owned `VectorField`; the demo retains its own row-major
  samples and submits ready arrow geometry.
- Added normalized `ColorMap` stop colors and documented its 256-entry RGBA8
  renderer LUT, preventing silent HDR clamp while making quantization explicit.
- Added `LogicalScreenVector`; clipping and shadow APIs now distinguish logical
  screen values from world/physical vectors. `Vec2`, `Rect`, and
  `RadialGradient` fields are closed behind constructors/accessors.
- Corrected particle timings for skipped `Timeout`/`Occluded` frames and fixed
  `ui_demo` test-harness access to private polyline data.

Verified with 67 all-feature library tests, all 7 example tests under
`--all-targets`, 36 no-default-feature tests, strict all-target clippy,
formatting, real-GPU pixel readback, and `git diff --check`.

The contract repair moved the estimate from 70% to 76% universal standalone
readiness before the release-discipline slice below.

## 2026-08-22 - Stage 6B release discipline slice

- Added an API stability policy that names the pre-1.0 compatibility rules and
  the non-negotiable finite, coordinate, color, ownership, and domain boundaries.
- Added a changelog with migrations for the typed screen API, depth ordering,
  color-map contract, and removal of the unused domain `VectorField`.
- Added a repeatable release checklist covering all targets, no-default mode,
  strict clippy, package inspection, platform evidence, and scoped performance
  claims.
- The GPU semantic fixture now supports `SIM_ENGINE_REQUIRE_GPU_TESTS=1`; release
  verification fails when no adapter exists instead of silently skipping.
- Added a three-OS GitHub Actions matrix plus a dedicated format/clippy/core-only
  job. The matrix compiles and runs every example test harness.

The roadmap estimate is now 80% universal standalone readiness. Remaining 6A
work is strict backend evidence and independent surface/device recovery; those
items still block a production-ready claim.

## 2026-08-22 - Finite motion and second-device recovery hardening

- Made `Tween` construction, retargeting, snapping, and updates fallible. Core
  scalar/vector interpolation uses `f64` intermediates; custom interpolators
  must validate stored and produced values, and a failed update is atomic.
- Stabilized extreme finite `Vec2` interpolation, rectangle centers, and linear
  and radial gradient sampling so valid opposite-`f32` extrema do not become
  NaN or infinity during derived arithmetic.
- Added checked vertex/instance capacity growth against
  `wgpu::Limits::max_buffer_size`. Prepared scenes and dynamic/particle restore
  APIs now report structured capacity errors rather than panicking or reaching
  GPU validation.
- Reset restored particle statistics to a pre-draw state instead of carrying
  stale rendered/culling values from the lost device.
- Extended the real-GPU fixture with a second logical `wgpu::Device`; prepared,
  dynamic, particle, and scalar resources are recreated there, and restored
  prepared vertices are compared byte-for-byte through GPU readback.
- Removed the remaining production `expect` paths from scalar-range uniform and
  color-map LUT construction.

Verified with 70 library tests plus all 7 example test harnesses under
`cargo test --all-targets --all-features`, 39 core-only tests under
`--no-default-features`, strict all-target clippy, formatting, diff validation,
and the second-device GPU readback.

The universal standalone estimate is now 82%. Strict backend evidence on the
supported OS matrix and independent surface-loss recreation remain the next 6A
work; examples stay frozen until correctness hardening is complete.

## 2026-08-23 - Stage 1C interactive performance hardening

The four-screen `ui_demo` is now a measured use-case showcase rather than an
unpaced visual smoke test. Fluid, Gas, Wave, and Edge Case labs can be opened
from the menu or selected with `--screen=...`; keyboard control is `1`-`4`,
`Escape`, `Space`, and `R`. `--benchmark`/`--uncapped` requests no-VSync and
reports average FPS, average and p99 frame time, scene construction,
tessellation, upload, acquisition, renderer CPU, and scheduler/compositor time.
The normal FIFO mode schedules at 120 Hz so timer overhead does not turn a
nominal 60 Hz target into 53-59 FPS.

Renderer and geometry changes:

- Open polylines now use one joined GPU-miter strip and round caps only at the
  two endpoints. The previous path emitted a 64-sector circle at every point.
- Screen-space round caps use 16 sectors; large world-space circles retain the
  existing 64-sector quality.
- Streaming scene rendering temporarily moves and restores its retained draw-
  batch vector instead of cloning every batch each frame.
- Dynamic meshes now reuse their CPU allocation and fuse conversion with extent
  collection.
- Dynamic meshes have a dedicated 28-byte GPU vertex and shader entry instead
  of uploading the universal 56-byte stroke vertex with five unused fields.
  Recovery memory and GPU upload bandwidth are therefore halved without a
  public API change.
- The demo dynamic-ribbon generator reuses shared endpoints and constructs each
  unique validated vertex once.
- The development profile uses `opt-level = 1`, avoiding misleadingly slow
  unoptimized tessellation for the normal `cargo run` workflow.

Measured on the development Vulkan system in release/no-VSync mode after
warm-up:

- Fluid showcase: 119-126 FPS, renderer 0.68-0.75 ms.
- Gas showcase: 118-126 FPS, renderer 0.65-0.72 ms.
- Wave showcase: 121-124 FPS, renderer 1.03-1.18 ms; tessellation fell from
  roughly 1.2-1.6 ms to 0.30-0.33 ms.
- Edge Case Lab: 120-122 FPS, renderer 0.86-0.99 ms.
- Showcase p99 frame time: about 14-16 ms across all four screens.
- 100k instanced particles: approximately 69-86 FPS.
- 10k dynamic segments / 60k vertices: approximately 209 FPS.
- 100k dynamic segments / 600k vertices: approximately 48-57 FPS after
  improving from 15-26 FPS; this intentionally records the remaining
  non-indexed triangle-list bandwidth limit instead of claiming 60 FPS for an
  unbounded workload.

The showcase therefore clears the requested 60 FPS floor and reaches the
100+ FPS target on every interactive screen. The roadmap moves from 82% to 84%
universal standalone readiness. Remaining performance work is indexed dynamic
geometry, backend-specific baselines, and GPU-completion timing rather than more
generic Scene primitives.

Verified with 71 library tests and all 10 `ui_demo` tests under
`cargo test --all-targets --all-features`, 39 core-only tests, strict all-target
clippy, formatting, diff validation, the real-GPU readback fixture, and a visual
inspection of the compact dynamic-mesh shader path.

## 2026-08-23 - Stage 2D extreme-workload budget foundation

The motivating consumer is a Sim;X supernova-remnant simulation containing a
dense gas field, hot ejecta, and a black hole. The renderer must remain a
bounded consumer of ready visual state so CPU time, GPU time, memory bandwidth,
and VRAM remain available to the host simulation. The same contract applies to
large fluids, galaxy fields, combustion, weather grids, and dense agent models.

The first audit found that `ParticleField2d` uploaded every full/partial update
immediately, then uploaded the camera-culled visible list again during render.
The eager upload never reached a draw call and doubled particle transfer
bandwidth. Particle updates now retain validated CPU visual state only; render
culls it and performs exactly one compact visible upload. Fully visible fields
upload the retained slice directly, while partially visible fields reuse a
retained culling scratch vector instead of allocating one every frame.

Particle timing was corrected at the same time: visible-buffer transfer is now
reported as geometry upload rather than tessellation, camera-uniform timing no
longer includes culling, and skipped/outdated frames retain the upload already
performed. `ParticleFieldUpdateReport::preparation` names the CPU update stage;
the old `upload` accessor remains as a deprecated compatibility alias.
`ParticleField2d` now exposes reserved CPU and GPU allocation bytes in addition
to recovery snapshot bytes.

On the same 100k-particle Vulkan fixture, steady throughput improved from about
69-86 FPS to 92-103 FPS. Renderer CPU moved from roughly 3.5-4.4 ms to
2.3-3.3 ms, and the corrected single visible upload measures about 0.72-0.79
ms. This is useful headroom for Sim;X, but it is not yet a complete star-remnant
path.

The intended bounded composition is:

- density/temperature gas as a low-resolution scalar texture updated in dirty
  regions and independently from presentation cadence;
- only visually important ejecta as capped instanced particles;
- the black-hole/accretion geometry as a small prepared or dynamic layer;
- optional trails at half or quarter physical resolution;
- final composition and UI at surface resolution.

Next Stage 2D slices are particle/scene render-to-target producers, an explicit
visible-instance/upload/memory budget, resolution-scaled targets, and a
repeatable star-remnant stress fixture. GPU timestamp evidence is required
before automatic quality adaptation can claim to protect simulation GPU time.

## 2026-08-23 - Stage 2D bounded extreme visualization completed

Finished the concrete supernova-remnant display path while keeping physics and
domain stepping in Sim;X. `examples/star_remnant_stress.rs` supplies only ready
visual state: a 384x216 gas field, ejecta particles, and a black-hole marker.
It renders through a half-resolution target and staggers gas/particle visual
updates so presentation does not demand a full host-state rebuild every frame.

The resource contract is now explicit:

- `ParticleRenderBudget` caps visible instances, GPU instance-buffer bytes,
  upload bytes per frame, and camera visibility checks per frame.
- A visibility-check cap samples candidates uniformly across retained state;
  `visibility_checked` and `budget_limited` expose the approximation instead of
  pretending every retained particle was classified.
- Updates retain validated CPU data but no longer upload it eagerly. The one
  selected visible list is uploaded immediately before its draw.
- Particle layers can render into `RenderTarget2d`, while
  `render_layered_visualization` encodes field, particles, and final composition
  in one command encoder/queue submission.
- Color-map LUT textures are retained and reused. Particle, scalar, and target
  resources expose workload memory accounting.

The default fixture retains 100,001 particles but checks and draws at most
30,000. On the development Vulkan system, a repeated 12-second release/no-VSync
run sustained approximately 89-116 FPS after initialization. Renderer CPU was
about 1.5-1.9 ms; tracked workload allocation was about 5.3 MiB CPU and 2.1 MiB
GPU. A 50k cap remains available for visual-density experiments, but 30k is the
default because it preserves materially more headroom for the host simulation.

The same fixture was then run with 1,000,001 retained particles. A first version
fell to roughly 44 FPS because the example rebuilt a fixed one eighth of the
whole array—125,000 visual instances—every frame. Cadence is now capped at
12,500 visual updates per frame, independent of retained count. The corrected
million-particle run sustained about 63-85 FPS after initialization; particle
preparation was about 0.7-1.1 ms, renderer CPU about 4.2-5.5 ms, tracked CPU
state 33.3 MiB, and tracked GPU workload memory remained 2.1 MiB.

The renderer layout was also decomposed along real boundaries:
`config.rs`, `tessellation.rs`, `visualization.rs`, and `tests.rs` now sit beside
the backend entry module. `renderer/mod.rs` is below 5,000 lines; pipeline and
resource lifetime code remain the next useful splits, rather than fragmenting
individual types arbitrarily.

Universal standalone readiness moves from 84% to 86%. Stage 2D is complete for
CPU/upload/memory/raster bounds. GPU timestamps and cross-backend evidence still
block any claim that the engine can automatically reserve a fixed amount of GPU
time for Sim;X.

## 2026-08-23 - Live device and surface recovery fixture

Added `WgpuRenderer::recover_device_and_surface`. It waits for outstanding work,
requests a replacement adapter/device compatible with the existing surface,
rebuilds every renderer-owned pipeline, uniform, transient vertex buffer, and
MSAA attachment, reconfigures the surface, and changes renderer identity. Old
prepared/dynamic/particle/scalar/target resources are therefore rejected until
the host calls their explicit restore API.

The star-remnant fixture exercises the complete path with `R` or
`--recovery-smoke`: scalar and particle CPU snapshots migrate, the target is
restored empty and redrawn, and rendering continues. Two automatic cycles passed
on the development NVIDIA/Linux system in roughly 108-122 ms per recovery and
returned to 110+ FPS.

The first implementation destroyed and recreated a native surface for the same
window and crashed in the vendor driver. Reconfiguring the existing surface is
stable, but immediately dropping the previous healthy logical device after
swapchain migration also reproduced a driver `present` crash. The renderer now
retains replaced devices until renderer destruction. This bounded exceptional
cost is documented; recovery is not intended as a frequent adaptive-quality
operation. Independent multi-window surface recovery and injected true device
loss remain open Stage 6A evidence.

Final verification for this session: 73 library tests and all 10 UI example
tests passed under `cargo test --all-targets --all-features`; 39 core-only tests
passed with `--no-default-features`; strict all-target clippy, formatting, and
`git diff --check` are green. The live recovery fixture also completed two
replacement-device/surface cycles.
