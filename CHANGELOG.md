# Changelog

All notable user-visible changes to Sim;Engine are recorded here. The project
uses Keep a Changelog-style sections and Semantic Versioning once the public API
reaches 1.0.

## Unreleased

### Added

- The first Sim;Math pseudo-3D foundation: overflow-aware `Vec3`, normalized
  `Rotation3d`, validated `Transform3d`, perspective/orthographic `Projection3d`,
  and logical-viewport `Camera3d` projection.
- A specification for retained stereometry meshes, depth-tested hidden lines,
  hatched sections, projected anchors, and the cube/octahedron consumer fixture.
- Comprehensive Integration Guide and Architecture Reference documentation,
  with the README reused as the generated rustdoc landing page.
- Hard particle visualization budgets for draw count, GPU allocation, per-frame
  upload, and camera visibility checks, with observable budget-limited counts.
- Fused one-submit scalar-field, particle-overlay, and target composition.
- A bounded supernova-remnant stress fixture with 100k and 1M retained-particle
  workloads, memory diagnostics, and live device/surface recovery smoke mode.
- `WgpuRenderer::recover_device_and_surface` for rebuilding transient backend
  state before retained resources are restored onto a replacement device.

### Changed

- Defined the 0.1 release as Linux-first; Windows, macOS, and web remain
  non-blocking future portability targets.
- Added a single Linux release-gate script and Linux CI jobs for all targets and
  strict Mesa-backed semantic GPU readback.
- Separated internal roadmaps, engineering logs, and Sim;X product drafts into
  an ignored local workbench; release packages now use an explicit include list.
- Screen clips and shadows now require explicit logical-screen position/vector
  types instead of unitless vectors.
- Pseudo-depth affects projection only; commands on one layer retain insertion
  order.
- `Vec2`, `Rect`, scene primitives, styles, colors, palettes, and gradients use
  constructors/accessors instead of externally mutable public fields.
- Heatmap `ColorMap` stops require normalized linear RGBA and the renderer's
  256-entry RGBA8 lookup-table quantization is an explicit contract.
- `Tween::new`, retargeting, snapping, and updates now return `Result`; custom
  `Interpolate` implementations must define their value-validity predicate.
- Prepared-scene creation and prepared/dynamic/particle restoration now return
  capacity errors when retained data cannot fit the active GPU device.
- Removed the unused public `VectorField`; vector samples remain host-owned and
  can be submitted as ready scene geometry.
- Particle updates defer GPU transfer until visibility/budget selection, and
  color-map lookup textures are cached across unchanged heatmap frames.

### Fixed

- Corrected vertical orientation in manual bilinear scalar-field sampling.
- Corrected double-alpha attenuation in render-target and trail composition.
- Rejected finite inputs whose derived geometry, scalar range, texture extent,
  gradient math, or tween interpolation would overflow.
- Corrected particle timing reports for skipped surface frames.
- Reset stale particle draw/culling statistics after device restoration.
- Rejected vertex and instance allocations beyond the active device's real
  buffer-size limit instead of reaching wgpu validation or integer overflow.
- Included example test harnesses in the full release gate.

### Testing

- GPU readback now verifies camera/depth/clip behavior, all four bilinear texel
  centers, sRGB conversion, half-alpha target composition, and byte-exact
  prepared-geometry restoration onto a second logical GPU device.
- `SIM_ENGINE_REQUIRE_GPU_TESTS=1` makes absence of a GPU adapter fail instead of
  silently skipping the GPU fixture.

## 0.1.0

Initial development release of the standalone 2D scene, camera, motion, and
optional `wgpu` renderer foundation.
