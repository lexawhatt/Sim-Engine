# Sim;Engine Wishlist for Sim;X

Status: integration wishlist for Sim;Engine after the Sim;X four-slice Physics
milestone.

Audience: Sim;Engine maintainers and Sim;X renderer-adapter maintainers.

Implementation status on the 0.2.0 release branch:

- W-002: implemented, including atomic mixed-layer batch construction,
  fallible tessellation reservations, actual-work enforcement, metrics, and a
  core-only adversarial benchmark.
- W-003: implemented through typed `ScreenScene` and `PreparedScreenScene`,
  including common-frame viewport/clip behavior.
- W-004: direct positioned surface viewport and scene-to-target paths
  implemented and consumable through the common frame contract.
- W-001: implemented for streaming/prepared world and screen scenes, dynamic
  meshes, particles, scalar fields, and 2D or retained-3D color targets. It
  provides stable ordering, local viewport/clip intersection, renderer-
  generation checks, six work limits, one clear/encoder/submission/present,
  and aggregate diagnostics.
- W-005: composition foundation implemented: prepared and streaming subscenes
  share one frame, and prepared vertices contribute no streaming geometry
  upload. Per-instance prepared transforms/opacity remain a later extension.
- W-006: engine resource slice implemented for bounded RGBA images, region
  updates, logical atlas batches, world quads, frame composition, and exact
  second-device recovery. Sim;X icon migration remains the consumer gate.
- W-007: low-level host-shaped atlas/run slice implemented with opaque glyph
  IDs, structured misses, deterministic measurement, one instanced draw per
  run, incremental atlas upload, and exact recovery. Sim;X font-adapter
  migration remains the consumer gate.
- W-008: the allowed bounded raw-triangle option is implemented through
  `DynamicMeshBudget` plus the common frame path. Polygon/path tessellation is
  intentionally still host-owned.
- W-009: implemented through primitive/source-grouped diagnostics, retained
  CPU/GPU memory accounting, the release-mode surface benchmark suite, and the
  complete named matrix runner. Optional hardware GPU timestamps remain a
  capability-dependent later extension, not part of the CPU metrics contract.
- W-010: implemented for logical/world widths, all bounded cap/join choices,
  dash phase and continuity, expansion limits, and logical arrow markers.

This document turns the confirmed rendering gaps in
`SIM_ENGINE_RENDERING_GAPS.md` into a proposed delivery order. It is not a bug
report, a demand for Sim;Engine to become a UI framework, or a request for the
renderer to own simulation, application state, layout, input, localization, or
window policy.

The API names below are illustrative. The important part is the capability,
its invariants, budgets, recovery contract, and measurable acceptance gate.

## Desired Outcome

Sim;Engine should let a host build one bounded frame containing:

- fixed logical-screen UI geometry;
- one or more independently controlled world-space viewports;
- streaming `Scene` content;
- prepared static content;
- dynamic meshes, images, glyph runs, particles, scalar fields, and retained
  3D targets where applicable;
- explicit ordering, clipping, blending, load operations, and recovery;
- structured work budgets and diagnostics before expensive allocation or
  tessellation.

For Sim;X, success means that a Physics editor can keep panels and text fixed
while the scientific viewport pans and zooms, without manually projecting all
world values into UI pixels or rebuilding the complete workspace every frame.

## Priority Overview

| Priority | Package | Why it comes now |
|---|---|---|
| P0 | W-001 bounded frame composer | foundation for all mixed rendering |
| P0 | W-002 ordinary Scene budgets | protects realtime work and memory |
| P0 | W-003 logical-screen scene space | removes the fixed-camera UI workaround |
| P0 | W-004 scene viewport and scene-to-target path | enables a real scientific camera |
| P1 | W-005 prepared plus streaming composition | removes full-frame retessellation |
| P1 | W-006 generic images and atlases | replaces raster rectangles for assets |
| P1 | W-007 glyph atlas and glyph-run rendering | unlocks scientific text and logs |
| P2 | W-008 bounded filled paths and triangles | supports vector diagrams and artwork |
| P1 | W-009 performance metrics and benchmark gate | prevents regressions and guides work |
| P2 | W-010 richer 2D stroke presentation | improves diagrams without command inflation |

P0 is the smallest architectural package that prevents later resources from
becoming isolated surface render paths. Images or text delivered before a
general composition contract would help locally but would not solve ordering,
viewport, clipping, or static/dynamic reuse.

## W-001 - Bounded General Frame Composer

Priority: **P0**.

### Capability

Expose one frame-building operation that can order heterogeneous renderer
inputs before one surface presentation. A frame should be able to reference:

- a streaming `Scene`;
- a `PreparedScene`;
- a `DynamicMesh2d`;
- a `ParticleField2d`;
- a `ScalarFieldTexture`;
- a `RenderTarget2d` composition;
- later image batches and glyph batches;
- retained 3D output through an existing render target.

Each pass or item needs explicit:

- order or layer;
- camera where a world-space source needs one;
- logical viewport;
- logical-screen clip;
- load/clear behavior;
- blend mode and opacity;
- source ownership/generation validation;
- resource and work budget contribution.

An illustrative direction is:

```rust,ignore
let mut frame = renderer.begin_frame(frame_budget)?;

frame.draw_scene(ScenePass {
    source: SceneSource::Prepared(&static_ui),
    space: SceneSpace::LogicalScreen,
    order: 0,
    ..ScenePass::default()
})?;

frame.draw_scene(ScenePass {
    source: SceneSource::Streaming(&physics_scene),
    camera: Some(&physics_camera),
    viewport: Some(canvas_viewport),
    clip: Some(canvas_clip),
    order: 10,
    ..ScenePass::default()
})?;

frame.draw_scene(ScenePass {
    source: SceneSource::Streaming(&dynamic_ui),
    space: SceneSpace::LogicalScreen,
    order: 20,
    ..ScenePass::default()
})?;

let report = frame.present()?;
```

This is an example of required semantics, not a required Rust design.

### Required invariants

- The surface is acquired, cleared, submitted, and presented at most once for
  a successfully completed frame.
- A later pass cannot accidentally erase an earlier pass unless an explicit
  load/replace operation authorizes it.
- Ordering is stable and documented across every supported source type.
- A viewport and clip cannot escape the physical target after DPI conversion.
- A resource from another renderer or recovery generation fails structurally
  before encoding.
- Failure before submission does not partially present a frame.
- Frame construction has an explicit pass, command, vertex, upload-byte,
  texture-byte, and draw-call budget.
- Device loss has one documented outcome for the complete frame, not a
  different implicit policy per resource path.

### Acceptance gate

A single frame must render prepared UI, a streaming scene through a separate
camera, and a dynamic overlay in a declared order with one surface present.
Automated tests must cover invalid ownership, overlapping viewports, nested
clips, clear/load combinations, device recovery, and budget rejection.

## W-002 - Budgets for Ordinary Scenes and Tessellation

Priority: **P0**.

### Capability

Give ordinary `Scene` construction the same bounded-work philosophy already
visible in particle and selected texture APIs.

A scene budget should be able to limit at least:

- accepted command count;
- retained command bytes;
- copied polyline/path points;
- estimated and actual tessellated vertex count;
- tessellation CPU scratch growth;
- generated draw batches;
- per-frame upload bytes.

An illustrative direction is:

```rust,ignore
let budget = SceneBudget::new(
    max_commands,
    max_points,
    max_vertices,
    max_retained_bytes,
    max_upload_bytes,
)?;
let mut scene = Scene::with_budget(background, budget)?;
```

The engine cannot retroactively bound memory the host allocated before passing
an owned `Vec`. It should still bound every engine-owned copy, command store,
tessellation result, and GPU upload. Slice-based or builder-based APIs may
avoid forcing an additional unbounded host allocation.

### Required behavior

- Budget rejection returns a structured error and leaves the scene unchanged.
- A command accepted by a strict budget cannot later exceed the declared
  vertex budget silently; either estimation is conservative or finalization is
  fallible before GPU submission.
- Metrics distinguish requested, accepted, rejected, tessellated, dropped, and
  rendered work.
- `Scene::new` can keep a documented default budget or remain explicitly
  unbounded for compatibility, but production hosts need an unmistakable
  bounded constructor.
- The API must report allocation-reservation failure rather than rely on
  process abort for ordinary recoverable capacity pressure.

### Performance opportunity

`Scene::try_push_to_layer` currently keeps the command vector ordered with
`partition_point` plus insertion. Adversarial or alternating layer order can
cause repeated movement of existing commands. Possible implementations include
per-layer storage, append-then-stable-finalize, or another representation that
keeps public stable ordering without quadratic insertion behavior.

The implementation choice is internal. The acceptance target is that building
an N-command scene with adversarial layer order should scale close to
`O(N log N)` or better, not `O(N^2)` command movement.

### Acceptance gate

Add tests at exactly-at-limit and one-over-limit boundaries for every budget,
including a large polyline and rounded shapes with high tessellation cost. Add
an adversarial mixed-layer benchmark and allocation-failure simulation where
the platform permits it.

## W-003 - Explicit Logical-Screen Scene Space

Priority: **P0**.

### Capability

Provide a type-safe way to draw fixed geometry in logical screen pixels. This
can be a separate `ScreenScene`, a coordinate-space value captured by commands,
or another design that does not overload world coordinates implicitly.

Required fixed-space primitives initially match ordinary 2D coverage:

- rectangle;
- circle;
- line and polyline;
- clipping;
- fill, stroke, gradient, shadow, opacity, and layer;
- future image and glyph commands.

### Required invariants

- Logical coordinates use a top-left origin and DPI-independent pixels.
- Physical conversion happens once at the backend boundary.
- UI geometry is unaffected by a world camera.
- Fixed-space and world-space items have one explicit ordering model.
- Screen clips remain compatible with nested clipping already provided by
  `ScreenClipRect`.
- Invalid or overflowing logical-to-physical conversion returns a structured
  outcome without partially encoding the item.

### Acceptance gate

Render the same logical-screen fixture at scale factors `1.0`, `1.25`, `1.5`,
`2.0`, and `3.0`. Its logical bounds, line widths, clipping, and hit-reference
coordinates must remain stable within the documented raster tolerance while a
world camera underneath it pans, zooms, and rotates.

## W-004 - Ordinary Scene Viewports and Scene-to-Target Rendering

Priority: **P0**.

### Capability

Allow an ordinary 2D `Scene` to render:

1. into a bounded viewport on the surface with its own `Camera2d`; and/or
2. into `RenderTarget2d` with explicit load, camera, viewport, and clip.

The frame composer should then place that result behind or between other frame
items.

### Why both forms are useful

A direct viewport pass avoids an unnecessary intermediate texture for an
ordinary editor canvas. A scene-to-target path enables caching, downsampling,
post-processing, trails, transitions, thumbnails, and mixing 2D scenes with
retained 3D or specialized field output.

### Required invariants

- Camera aspect and viewport dimensions are explicit and validated.
- Viewport clipping is independent from scene-local nested clips; the effective
  clip is their intersection.
- Render-target dimensions remain physical pixels while viewport and UI
  placement remain typed logical pixels.
- Load, clear, alpha, color-space, and premultiplication semantics match
  existing target composition.
- Rendering a target into itself or otherwise aliasing source/destination is
  rejected before encoding.
- Target contents after recovery remain explicitly empty until redrawn.

### Acceptance gate

Render four ordinary 2D scenes with four independent cameras into four logical
viewports while fixed UI remains stationary. Repeat with two scenes rendered
to targets and composed. Resize and change DPI without moving UI-relative
viewport bounds or corrupting camera aspect.

## W-005 - Prepared Subscenes Mixed with Streaming Content

Priority: **P1**, immediately after the P0 frame foundation.

### Capability

Permit `PreparedScene` to participate in the same ordered frame as streaming
scene content and other resources. Prefer prepared subscenes rather than only
one all-or-nothing prepared surface scene.

Useful per-instance state:

- order/layer;
- translation or validated 2D transform;
- opacity;
- logical viewport and clip;
- visibility;
- camera or coordinate space.

A transform should not require re-tessellating immutable geometry when it can
be applied safely in a uniform or instance record.

### Performance objective

For a workspace where 90% of vertices are static and 10% change each frame,
the steady-state frame should tessellate and upload approximately the dynamic
10%, not the complete scene. Prepared geometry should contribute zero streaming
vertex upload after warm-up unless its resource is restored or replaced.

### Acceptance gate

Build a fixture with static panels, grid, labels, and icons plus a moving body
and changing measurements. Compare full streaming against prepared-static plus
streaming-dynamic. Report CPU tessellation, upload bytes, draw batches, and
total frame CPU. The mixed path must preserve identical ordering and clipping
within the renderer's raster tolerance.

## W-006 - Generic Images, Sprites, and Texture Atlases

Priority: **P1**.

Implementation status: **engine slice complete; Sim;X acceptance migration
pending**. Logical batches and single world quads share `FrameComposer`
ordering/clipping. Rotation is supplied by the world camera rather than a
separate image-local transform.

### Capability

Add a retained RGBA image resource and an image scene/frame item supporting:

- immutable creation and bounded replacement;
- full image or atlas sub-rectangle selection;
- destination rectangle;
- tint and opacity;
- nearest and linear filtering;
- layer/order and clip;
- logical-screen and world placement;
- optional rotation only if the general transform contract already supports
  it;
- restoration after device loss.

Direct SVG parsing does not need to be part of Sim;Engine. Sim;X can rasterize
SVG externally if the renderer can consume the bounded RGBA result efficiently.

### Budget and recovery requirements

- Validate dimensions, row pitch, integer arithmetic, format, and byte budget
  before resource allocation.
- Avoid an unconditional extra full-size copy when ownership can be transferred
  safely.
- Retain CPU pixels, accept a host restoration callback, or explicitly mark a
  resource non-restorable. The policy must be visible in the type/API.
- Region updates must be bounded and validated before queue writes.
- Atlas allocation and eviction, if engine-managed, are presentation-only and
  must expose misses, uploads, and evictions in metrics.

### Acceptance gate

Replace the current Sim;X three-icon raster-run fixture with three atlas draws.
The command count should be constant per icon rather than proportional to
visible pixel runs. Verify nearest/linear sampling, alpha edges, tint, clipping,
DPI, atlas bounds, byte limits, and exact restoration behavior.

## W-007 - Glyph Atlas and Bounded Glyph-Run Rendering

Priority: **P1**.

Implementation status: **low-level engine slice complete; Sim;X acceptance
migration pending**. Shaping, fallback selection, and line breaking remain
host responsibilities as proposed below.

### Recommended scope split

The most reusable first layer is not a complete UI text system. It is a
renderer-owned glyph atlas plus a host-provided positioned glyph-run API:

```rust,ignore
renderer.update_glyph_atlas(...)?;
screen_scene.glyph_run(font_atlas, positioned_glyphs, color, clip, order)?;
```

The host may perform shaping with its chosen library and localization policy.
An optional higher-level Sim;Engine feature can later manage fonts, shaping,
fallback, line breaking, and measurement. This separation lets Sim;Engine
solve batching, atlas upload, clipping, DPI, ordering, budgets, and recovery
without owning product copy or layout.

### Minimum capability

- atlas creation and bounded region upload;
- positioned glyph quads with UV bounds;
- color, opacity, clip, order, and coordinate space;
- deterministic measurement of the submitted positioned run;
- structured missing-glyph and atlas-capacity outcomes;
- support for Unicode glyph identities supplied by the host;
- restoration contract for atlas pixels and glyph metadata.

### Scientific-text acceptance fixture

The fixture should include:

- superscripts and subscripts;
- `mu`, degree, delta, integral, summation, and Greek variable glyphs;
- signed exponents and scientific notation;
- Latin and Cyrillic user-visible names;
- mixed fallback fonts;
- clipping inside a narrow inspector row;
- scale-factor changes without baseline drift.

The engine need not interpret these characters scientifically. It must render
the shaped glyph run faithfully and with bounded work.

### Performance objective

After atlas warm-up, unchanged text should generate no glyph texture upload.
One visible glyph should normally become one quad/instance rather than many
rectangle commands. Metrics should distinguish shaped glyph count, visible
glyph count, atlas misses, uploaded pixels, evictions, and rendered quads.

## W-008 - Bounded Filled Paths, Polygons, and Raw 2D Triangles

Priority: **P2**.

Implementation status: **bounded raw-triangle option complete**. The engine
does not currently tessellate polygon contours or assign a fill rule; hosts can
submit their deterministic final triangles through a budgeted retained
`DynamicMesh2d` at any common frame order/clip.

### Capability

Provide at least one of:

- a validated filled polygon/path primitive;
- a bounded caller-provided colored triangle item composable with a `Scene`;
- a retained vector mesh that can enter the general frame composer at an
  ordinary order/clip.

A full SVG document model is not required. Host applications may parse SVG or
construct scientific paths themselves.

### Required safety contract

- explicit point, contour, generated-vertex, and byte budgets;
- finite-coordinate and overflow validation;
- documented fill rule;
- deterministic tessellation for identical inputs;
- structured rejection of self-intersection or unsupported geometry if the
  chosen tessellator cannot define it safely;
- no partial mutation of retained resources on failed updates;
- exact recovery of retained geometry.

### Acceptance gate

Cover convex and concave polygons, holes, both supported fill rules, extreme
finite coordinates, degenerate edges, self-intersection policy, budget limits,
clipping, gradients where supported, and composition between panels and text.

## W-009 - Performance Observability and Regression Benchmarks

Priority: **P1**, developed alongside every preceding package.

Sim;Engine already exposes useful CPU-side frame metrics. The wishlist is to
extend them so optimization work is driven by evidence rather than aggregate
frame time.

### Additional metrics

- scene construction/finalization time where engine builders are used;
- requested, accepted, rejected, and rendered commands by primitive/source;
- estimated and actual tessellated vertices;
- draw batch count and batch breaks by reason, including clip or pipeline;
- streaming upload bytes and retained bytes;
- vertex/index/uniform/texture buffer growth events;
- prepared vertices reused;
- image and glyph atlas uploads, misses, and evictions;
- pass count, encoder count, queue submissions, and surface presents;
- per-viewport culled and rendered work where culling exists;
- optional GPU timestamp spans on adapters that support timestamp queries.

GPU timestamps should be capability-detected and optional. CPU-only metrics
must remain available on the current supported path.

### Required benchmark fixtures

1. `ui_static_10k`: 10,000 static rectangles/glyph quads.
2. `ui_90_10`: 90% prepared static geometry plus 10% streaming changes.
3. `four_viewports`: four cameras, clips, and mixed scene sizes.
4. `image_atlas`: hundreds of sprites from one and several atlases.
5. `scientific_text`: long inspector/log text with warm and cold atlases.
6. `mixed_layers`: adversarial command insertion order across many layers.
7. `budget_rejection`: oversized commands rejected before expensive work.
8. `hidpi_resize`: repeated resize and scale-factor transitions.
9. `recovery_frame`: restore every retained source and redraw one mixed frame.

### Regression policy

Record fixture hardware/backend and compare both absolute values and relative
changes. Avoid one universal millisecond promise across GPUs. A release gate
should still flag statistically meaningful regressions in:

- total renderer CPU;
- tessellation CPU;
- upload bytes;
- steady-state allocations;
- draw batches/submissions;
- retained memory;
- frame work at budget boundaries.

For prepared/static fixtures, a stronger deterministic gate is possible: zero
retessellation and zero geometry upload after warm-up unless invalidation or
recovery occurs.

## W-010 - Richer Bounded 2D Stroke Presentation

Priority: **P2**.

### Capability

Extend line and polyline styling with a bounded subset useful for scientific
diagrams:

- butt, square, and round caps;
- bevel, miter-with-limit, and round joins;
- bounded dash/gap patterns with explicit phase;
- optional arrowhead/marker geometry through reusable retained definitions;
- screen-pixel and world-unit width modes where semantics are explicit.

### Why this matters

Grid styles, hidden relationships, constraints, selected paths, vectors, and
measurement guides otherwise require hosts to split one conceptual stroke into
many scene commands. That increases command count and makes dash phase,
clipping, joins, and zoom behavior inconsistent.

### Acceptance gate

Test every cap/join combination, miter limits, dash continuity across polyline
segments, clipping, camera zoom, extreme but finite widths, budgeted dash
expansion, and deterministic generated-vertex counts.

## Cross-Cutting Requirements

Every accepted wishlist package should include all of the following.

### Validation and atomicity

- Validate finite values and checked integer arithmetic at the public boundary.
- Reject invalid or over-budget work before partial retained-state mutation.
- Return structured errors that identify resource, pass, and budget category.
- Keep boolean convenience APIs only where the fallible equivalent remains
  available and documented.

### Ownership and recovery

- Retained resources carry renderer ownership/generation.
- Cross-renderer use fails before encoding.
- Device recovery has an exact restoration or explicit redraw contract.
- CPU recovery snapshots have explicit memory accounting.
- A mixed frame cannot silently combine restored and stale resources.

### Coordinate and color semantics

- Keep world, logical-screen, and physical-pixel types distinct.
- State whether dimensions, widths, offsets, viewports, and clips are world or
  logical values.
- Preserve the documented straight-linear public color and premultiplied target
  storage model.
- State filtering and color-space behavior for images, atlases, and target
  composition.

### Budgets

- Every externally sized collection has a host-configurable ceiling.
- Budget checks cover CPU retained bytes, scratch work, GPU bytes, uploads,
  generated vertices, and draw work where applicable.
- Default budgets are documented and can be made stricter by a host.
- Metrics reveal both requested and accepted work.

### Verification

- CPU tests work without default GPU features where the resource model permits.
- GPU semantic fixtures cover the supported Vulkan release path.
- Warning-free rustdoc explains units, ownership, recovery, and failure.
- Examples demonstrate bounded integration rather than only happy-path output.
- New optional dependencies remain feature-gated and do not break the offline
  or no-default-features package boundary.

## Suggested Delivery Sequence

### Phase A - Composition foundation

1. W-002 ordinary Scene budgets and adversarial insertion benchmark.
2. W-001 general frame composer with streaming scenes and render targets.
3. W-003 logical-screen scene space.
4. W-004 independent scene viewports and scene-to-target rendering.

Exit condition: Sim;X can render fixed UI plus one independently panning and
zooming Physics scene in one bounded frame.

### Phase B - Static/dynamic efficiency

1. W-005 prepared plus streaming composition.
2. W-009 expanded metrics and `ui_90_10` release benchmark.
3. Per-instance prepared transforms only after composition semantics are fixed.

Exit condition: static UI is not tessellated or uploaded every frame while a
small scientific overlay changes.

### Phase C - Raster resources and text

1. W-006 retained images and atlases.
2. W-007 low-level glyph atlas and positioned glyph runs.
3. Optional higher-level text shaping only after the lower-level contract is
   stable and a concrete host requests it.

Exit condition: Sim;X removes raster-run icons and the rectangle-per-pixel font
without giving Sim;Engine product layout or localization ownership.

### Phase D - Vector presentation

1. W-008 filled paths or composable bounded triangles.
2. W-010 richer stroke styles.
3. Add vector/path benchmarks to W-009.

Exit condition: scientific diagrams and supplied vector-derived artwork no
longer need object-specific rectangle or segment expansion.

## Compatibility Strategy

Existing focused APIs should remain useful:

- `render(scene, camera)` can internally create a one-pass frame;
- `render_prepared` can internally create a one-source frame;
- specialized field/particle helpers can remain convenience wrappers around
  the general composer;
- existing `Scene` methods can continue to mean world-space commands;
- new screen-space behavior should use new types or explicit spaces rather
  than silently changing existing coordinate semantics.

This provides a migration path without freezing the renderer into isolated
surface entry points.

## Explicit Non-Goals

This wishlist does not ask Sim;Engine to own:

- Sim;X navigation, panels, docking, Inspector behavior, or Workspace policy;
- mouse/keyboard input dispatch or accessibility focus;
- application window creation or event-loop scheduling;
- scientific entities, units, stepping, or domain state;
- SVG parsing as a mandatory engine dependency;
- localization, copy, font choice, or user-content policy;
- an unbounded immediate-mode UI framework;
- automatic recovery of canonical simulation state.

Sim;Engine should provide bounded rendering mechanisms. Sim;X remains
responsible for deciding what the interface and scientific scene mean.

## Sim;X Integration Definition of Done

The combined wishlist is successful for Sim;X when the adapter can:

1. remove manual logical-screen-to-world conversion for fixed UI;
2. use an independent `Camera2d` inside the scientific canvas;
3. keep panels fixed while that camera pans and zooms;
4. submit prepared static UI and streaming scientific state in one frame;
5. render each social icon with constant image/atlas work;
6. replace the 3-by-5 rectangle font with bounded glyph runs;
7. enforce a complete scene/frame work budget before unbounded renderer work;
8. surface useful command, vertex, upload, batch, pass, and recovery metrics;
9. recover retained UI resources without touching canonical simulation state;
10. preserve all current renderer/domain ownership boundaries.

Until then, the existing Sim;X workarounds remain deliberately local and
bounded. They must not evolve into a second general renderer inside Sim;X.
