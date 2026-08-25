# Changelog

All notable user-visible changes to Sim;Engine are recorded here. The project
uses Keep a Changelog-style structure and follows Semantic Versioning once the
public API reaches 1.0.

## Unreleased

No unreleased changes.

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
