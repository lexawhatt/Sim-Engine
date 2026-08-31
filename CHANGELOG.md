# Changelog

All notable user-visible changes to Sim;Engine are recorded here. The project
uses Keep a Changelog-style structure and follows Semantic Versioning once the
public API reaches 1.0.

## Unreleased

No unreleased changes.

## 0.2.0 - 2026-08-29

Second official Linux release. This release turns the Sim;X integration
foundation developed after 0.1.0 into supported, bounded APIs for complete
composed frames, fixed-screen UI, retained images, and host-shaped text.

### Added

- Explicit `SceneBudget` limits and `SceneStatistics` for commands, retained
  points/bytes, conservative and actual tessellated vertices, upload bytes,
  and draw batches.
- Atomic `Scene::try_extend_to_layers` for high-volume mixed-layer construction
  with one `O(N log N)` ordering pass, plus validated `DrawCommand` builders.
- `ScreenScene` and `PreparedScreenScene` for fixed top-left logical-pixel
  geometry that cannot accidentally receive a world camera.
- Positioned `LogicalViewportRegion` rendering for ordinary scenes and an
  explicit ordinary-scene-to-`RenderTarget2d` path with independent
  `PhysicalPerLogical` scale and `RenderTargetLoad` behavior.
- `FrameComposer`, `FrameBudget`, per-item ordering/viewports/clips, and
  `FrameReport` for one bounded surface frame containing streaming or prepared
  world/screen scenes, dynamic meshes, particles, scalar fields, and composable
  2D or retained-3D color targets.
- Frame diagnostics for total, streaming, and reused vertices, streaming and
  total upload bytes, referenced texture bytes, passes, commands, and draw
  calls.
- Actual encoder, render-pass, queue-submission, and surface-present counts on
  `FrameReport`, including zeroes for skipped surface frames.
- Bounded retained `Image2d` resources, partial region upload, atlas source
  rectangles, nearest/linear sampling, logical sprite batches, world-space
  image quads, exact pixels for recovery, and image composition in
  `FrameComposer`.
- Low-level host-shaped text through `GlyphAtlas2d`, opaque `GlyphId` metadata,
  bounded `GlyphRun2d` instancing, deterministic logical bounds, structured
  missing-glyph outcomes, incremental uploads, and exact atlas/run recovery.
- `DynamicMeshBudget` and `create_dynamic_mesh_with_budget` for bounded raw
  filled triangles in the common frame ordering path.
- Reusable `StrokeStyle2d` with explicit logical-pixel or world-unit widths,
  butt/square/round caps, bevel/round/bounded-miter joins, allocation-free dash
  patterns with phase and expansion limits, and logical-pixel arrow markers.
- Styled line/polyline entry points on `Scene`, `ScreenScene`, and
  `DrawCommand`, while legacy methods preserve their existing presentation.
- Primitive-grouped requested/accepted/rejected `SceneStatistics` and
  rendered/dropped `TessellationStats`, plus source-grouped frame counts and
  referenced CPU/GPU retained-memory diagnostics.
- A core-only adversarial scene-construction benchmark.
- A release-mode `rendering_benchmark_suite` and matrix runner covering the
  named static-10k, 90/10 prepared/streaming, four-viewport, image-atlas,
  scientific-text, mixed-layer, budget-rejection, HiDPI-resize, and retained-
  resource recovery fixtures.
- Nested-KWin HiDPI gate scripts, packaged with the crate, which fail closed
  when the required compositor or output-control tooling is unavailable.
- A retained scientific glyph-atlas probe in `ui_demo`, reused above all four
  world-camera workloads without steady-state atlas or instance upload.

### Changed

- Ordinary tessellation now uses fallible CPU reservations and rechecks actual
  vertex, upload-byte, and batch work against the originating scene budget
  before GPU submission.
- `TessellationStats` now reports actual vertex count, upload bytes, and draw
  batches in addition to command outcomes and primitive categories.
- Dash phase remains continuous through polyline vertices; a visible dash that
  crosses a bend receives one configured join and no artificial internal caps.
- Open strokes now use one non-overlapping joined topology. Inner segment
  corners meet at a bounded intersection, only the projected outer bevel/round
  fan survives, and over-limit logical miters produce a real bevel wedge.
- Arrow markers use the path endpoint as a shared butt boundary and extend
  outward from it. The body never reverses or overlaps a marker, even when two
  markers are longer than a short path or the terminal dash run is absent.
- Exact and numerically indistinguishable 180-degree polyline retraces, plus
  repeated adjacent points, are rejected during scene validation instead of
  producing pinched or multiply blended stroke quads.
- Frame retained-memory diagnostics count each referenced CPU, GPU buffer, and
  texture allocation once even when the same resource is drawn multiple times.
- Target and heatmap composition uniforms now carry an explicit logical
  destination, allowing offscreen resources to scale into bounded frame
  viewports without abusing scissor clipping.
- Dynamic-mesh creation, replacement, and restoration now reserve CPU recovery
  storage fallibly; full replacement commits retained state only after all
  validation, reservation, and replacement-buffer creation succeeds.
- Streaming composition reuses its transient vertex allocation, tessellates
  directly into the upload buffer without an intermediate full-payload copy,
  and caches the immutable circle/corner unit samples used by primitive
  tessellation. The unchanged 90/10 release fixture therefore measures host
  scene work and GPU upload rather than repeated trigonometry and allocator
  churn.
- Cached circle and quarter-circle samples use exact cardinal endpoints, so
  large finite circles close without a residual retrace and maximum-radius
  rounded rectangles meet straight edges without a scale-amplified tangent
  kink.
- Circle vertices retain the center and local radius offset separately through
  camera-relative WGSL arithmetic. Fill, stroke, shadow/spread, and radial
  gradients therefore remain drawable when a small radius lies below a large
  finite center's source `f32` ULP. Tessellated vertices are now 80 bytes, and
  scene estimates, upload budgets, recovery memory, and diagnostics reflect
  that exact payload.
- Rounded-corner arcs and world-unit stroke bodies, joins, and caps use the
  same anchor-plus-local-offset representation. Any positive representable
  radius/range remains nonzero; closed-stroke deduplication no longer erases
  geometry merely because its squared length is below `f32::EPSILON`.
- Camera/geometry validation preserves base/offset correlation between
  commands and rejects both pre-transform relative-coordinate overflow and
  final screen-to-clip overflow, including particle-radius extrusion and
  retained image/glyph sprite bounds.
- Streaming composition returns its reusable transient vertex allocation to
  the renderer on every structured error path, preserving steady-state memory
  behavior after a rejected frame.
- `WgpuRenderer::set_pre_present_notify` owns the host pacing boundary:
  registered callbacks run after queue submission and immediately before
  FIFO/FIFO-relaxed/Mailbox presentation, while Immediate stays uncapped.
- Adapter diagnostics now expose PCI vendor/device IDs, physical PCI bus
  address, surface format, and sample count. Release evidence rejects any
  semantic, performance, or compositor process that selects a different
  physical adapter instance.
- `SceneError::DegenerateGeometry` documentation now matches validation: one
  non-drawable consecutive segment is sufficient to reject a polyline.

### Testing

- Release performance fixtures now require and report a fixed `1280x720`
  physical surface at scale `1.0`, reject zero-sized or drifted surfaces, and
  perform a post-present event-loop finalization check before publishing the
  last trial.
- The DPI reconfiguration p95 now includes the timed renderer resize/surface
  configure operation instead of measuring only the following frame.
- Release scripts now compile and run from a read-only detached worktree of the
  exact evidence SHA with replacement objects disabled and a separate Cargo
  target directory. Evidence is staged and atomically published as one bundle
  only after the complete gate succeeds, so crashes cannot expose partial
  manifests as a passed run.
- A nonblocking process lock serializes release-evidence invalidation and
  publication, and Linux `mv -T` makes the completed directory replacement
  explicit. The bundle now includes a structured performance manifest with
  all seven surface fixture reports and their exact-SHA passed verdicts.
- Surface benchmark and HiDPI renderers register winit's
  `Window::pre_present_notify`; synchronized modes invoke it after submission
  and immediately before present, while Immediate benchmarks remain unpaced.
  Finalization still crosses a compositor-aware redraw boundary before
  accepting the final output/generation snapshot.

- Added exact-limit/one-over budget tests, atomic batch rejection tests,
  conservative-estimate checks against actual tessellation, positioned
  viewport/DPI math coverage, logical-screen coordinate regressions, frame
  budget/order/clip tests, and semantic GPU readback of positioned target
  composition.
- Added image dimension/row-pitch/atlas-boundary tests, glyph metadata/missing
  glyph/measurement tests, second-device image/glyph recovery, and semantic GPU
  readback for screen/world image placement, sRGB alpha tint, atlas isolation,
  and instanced sprite drawing.
- Added complete cap/join combinations, miter fallback, dash phase/continuity,
  clipping, logical/world zoom behavior, finite-width overflow, bounded dash
  expansion, deterministic topology, and real GPU dash/gap readback coverage.
- Extended the mandatory GPU oracle with half-alpha pixel comparisons for all
  36 cap/join/width-mode/turn-direction combinations at the production-selected
  MSAA count, plus a four-pixel body with short start-and-end arrow markers and
  miter-to-bevel probes. It rejects repeated alpha blending, asymmetric CW/CCW
  shader behavior, and any cap protruding through a marker boundary.
- Replaced the synthetic `hidpi_resize` claim with a deterministic
  `dpi_reconfigure` workload and an event-driven `hidpi_transition` fixture.
  The release matrix drives a nested KWin compositor from scale 1.00 to 1.25
  and accepts evidence only for one paired `ScaleFactorChanged -> Resized ->
  successful present` transaction from the exact release revision.
- The named matrix now uses four independent prepared world scenes and four
  independent cameras, reports renderer work separately from surface acquire,
  prints the actual adapter/backend/driver and present mode, validates fixture
  source/count contracts, and pins semantic/performance/HiDPI evidence to one
  physical Vulkan adapter through its PCI bus address. A real surface probe
  also makes the semantic oracle use the production surface format and selected
  MSAA count. The matrix enforces per-fixture renderer-work p95
  thresholds. Immediate requires 60 FPS; Mailbox/FIFO require 95% of the
  positive confirmed current-monitor refresh within a 30-60 Hz release
  reference. Every gated run first submits an unmeasured `Drawn` frame;
  Immediate does not require refresh metadata, while synchronized modes reject
  missing or zero refresh and never substitute another enumerated monitor.
  Every warmup/measured frame must be `Drawn`. Each of the three independent
  120-frame trials must clear the wall-throughput floor; the median is reported
  only for diagnosis, and work percentiles combine all 360 frames. Acquire
  percentiles are diagnostics rather than a scheduler-sensitive one-run
  verdict. Standalone matrix and HiDPI scripts reject dirty worktrees before
  assigning evidence to `HEAD`, retain the starting SHA, and recheck unchanged
  `HEAD` plus clean state before promotion and on completion. Gated workloads
  yield to the event loop between every frame; resize, scale, or output changes
  invalidate generation-bound confirmation and restart timing after a new
  unmeasured `Drawn`. The 120th refresh-metadata confirmation present receives
  its own final follow-up check. HiDPI evidence pins the same physical PCI
  adapter without incorrectly requiring a nested compositor to advertise the
  desktop surface's format; production-format/MSAA equality remains mandatory
  for the semantic oracle and production performance fixtures.
- CI semantic evidence records and asserts the exact GitHub revision instead
  of permitting `vcs_sha=unknown`.
- Added the interactive `stroke_gallery` visual oracle with four pages for all
  v0.2 stroke styles, alpha contracts, markers, dashes, camera motion, and short
  accepted geometry.

## 0.1.0 - 2026-08-25

First official Linux release of the validated 2D scene, scientific
visualization renderer, and focused retained 3D stereometry path.

### Added

- Validated 2D primitives, styles, gradients, clipping, cameras, pseudo-depth,
  and fallible tweening.
- Streaming scenes, prepared geometry, and capacity-managed dynamic triangles.
- Instanced particle fields with explicit draw, memory, upload, culling, and
  visibility-check budgets.
- Finite scalar fields, incremental `R32Float` texture updates, deterministic
  nearest/manual-linear sampling, and quantized GPU color maps.
- Offscreen render targets, explicit alpha/additive/replace composition,
  bounded trail accumulation, and fused scalar/particle/surface rendering.
- Renderer metrics covering CPU tessellation, uploads, surface acquisition,
  submission, command counts, and dropped-command diagnostics.
- Device and surface recovery with explicit restoration for retained geometry,
  dynamic meshes, particles, scalar textures, targets, trails, and 3D meshes.
- Validated `Vec3`, `Rotation3d`, `Transform3d`, `Projection3d`, and `Camera3d`.
- Retained `Mesh3d` topology, stable `Object3dId` scene handles, independent
  transforms, hardware depth, and logical-pixel visible/dashed hidden edges.
- Interactive 2D, particle, star-remnant, cube/octahedron, and cylinder
  derivation examples with bounded performance diagnostics.
- Concrete Immediate/Mailbox/FIFO presentation diagnostics and a completed-work
  GPU synchronization point for controlled benchmarks.
- Scene provenance in `Object3dId`, preventing a handle from one `Scene3d`
  from mutating an object with the same local number in another scene.
- Bounded device-recovery quarantine configuration and telemetry through
  `WgpuRendererOptions::with_max_quarantined_devices`,
  `quarantined_device_count`, and `remaining_device_recoveries`.
- Fallible, byte-budgeted `ScalarField::filled` allocation with an explicit
  `filled_with_byte_limit` override.
- Explicit `LogicalPixels`, `WorldLength`, and `PhysicalPerLogical` scalar unit
  types for 3D edge, projection-range, and target-scale boundaries.
- Atomic `restore_scene3d` migration that preserves object IDs and instance
  state while recreating each distinct stale mesh once.
- Seeded CPU/WGSL clipping equivalence readback and persistent Vulkan adapter
  evidence artifacts in the local and CI release gates.

### Changed

- The supported v0.1.0 target is explicitly Linux with Vulkan. Windows, macOS,
  and web are not release-gated platforms.
- The minimum supported Rust version is 1.90, matching the complete currently
  resolved renderer dependency graph.
- Public data fields are private behind validated constructors and accessors.
- Pseudo-depth affects projection but never reorders commands within a layer.
- Screen clips and shadow offsets use explicit logical-screen types.
- GPU color maps use a documented 256-entry RGBA8 lookup table.
- `ParticleInstance2d` remains available without the `wgpu` feature so visual
  state generation can be validated and benchmarked CPU-only.
- No-VSync selection resolves to an advertised concrete surface mode and can
  be inspected through `RendererSurfacePresentMode`.
- Render-bound colors must be normalized linear RGBA. Finite overshoot remains
  available for interpolation but must be explicitly clamped before insertion.
- Hidden-line classification is explicitly conservative within two
  implementation depth units; coplanar and sub-resolution separation resolves
  visible.
- CI actions and Rust toolchains are pinned, and CI now verifies doctests,
  warning-free rustdoc, the package boundary, and the package archive.
- 3D wireframe metrics, projection distances, and target scale now use opaque
  unit types instead of interchangeable raw scalars.

### Fixed

- Unified accepted short-line invariants across Scene and tessellation.
- Rejected finite source geometry whose derived arithmetic would overflow.
- Made forward camera and DPI conversions fallible on non-finite output.
- Validated backgrounds and clips at their public mutation boundaries.
- Corrected manual bilinear heatmap Y orientation.
- Corrected double alpha attenuation in target and trail composition.
- Rejected scalar textures, buffers, and retained resources beyond real device
  limits before creating invalid GPU objects.
- Corrected particle metrics for skipped surface frames and recovery.
- Removed Wayland frame-callback throttling from uncapped example paths.
- Removed renderer-internal product/debug scene construction.
- Homogeneously clip 3D display edges against the complete frustum before
  perspective division, instead of rejecting the whole frame when an edge
  crosses the camera or near plane.
- Reject display edges whose distinct indices reference coincident model-space
  vertices instead of accepting a zero-length mathematical edge.
- Preflight retained 3D vertex, index, and edge buffer sizes and draw-count
  representation before fallible host staging allocation.
- Require and assert Vulkan in the strict Linux semantic GPU gate.
- Prevent unbounded accumulation of previous logical GPU devices after repeated
  recovery.
- Keep extreme homogeneous clipping equivalent on Vulkan implementations that
  flush subnormal reciprocals by using a shared normal-range scale.

### Testing

- Added finite and overflow boundary coverage across math, cameras, scenes,
  tweening, scalar fields, particles, resources, and 3D projection.
- Added semantic GPU readback for camera/depth/clip, scalar sampling direction,
  sRGB conversion, half-alpha composition, retained-resource restoration,
  3D depth ordering, coplanar visible edges, and dashed hidden edges.
- Added a 256-case seeded CPU/WGSL homogeneous-clipping equivalence fixture and
  second-logical-device `Scene3d` recovery validation.
- The Linux gate builds and lints all targets with and without default features,
  checks Rust 1.90, generates warning-free rustdoc, requires a GPU semantic
  fixture, records exact adapter/driver evidence, and verifies the offline
  package archive.
