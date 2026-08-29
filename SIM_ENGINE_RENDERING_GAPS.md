# Sim;Engine Rendering Gaps Used by Sim;X

Status: frozen pre-migration renderer-integration record for Sim;Engine
`0.1.0`. A new engine version is expected. Re-check every item against the
actual updated API and Sim;X rendering path before deleting a workaround or
marking a gap resolved.

## Sim;Engine 0.2.0 Re-check - 2026-08-29

The post-0.1 engine branch now has a concrete API answer for every recorded
integration gap. This does not authorize deleting a Sim;X workaround until its
adapter migration and visual/performance comparison pass.

| Gap | Engine-side status |
|---|---|
| R-001 | `Image2d`, sub-rect draws, retained `ImageBatch2d`, region upload, logical/world placement, budgets, and recovery implemented |
| R-002 | Bounded raw colored triangles implemented through `DynamicMeshBudget`; SVG/path parsing and triangulation remain host-owned |
| R-003 | `FrameComposer` orders dynamic meshes with scenes and all other supported sources in one presentation |
| R-004 | Host-shaped `GlyphAtlas2d`/`GlyphRun2d` implemented with structured misses, measurement, batching, upload, and recovery |
| R-005 | Six-axis `SceneBudget`, fallible tessellation, actual-work recheck, and scene/frame statistics implemented |
| R-006 | Typed `ScreenScene`/`PreparedScreenScene` and logical image/glyph placement implemented |
| R-007 | Independent positioned viewports, cameras, scene-to-target, and one-present multi-source composition implemented |
| R-008 | Prepared and streaming scenes mix in one frame with separate reused/streaming vertex and upload accounting |

Remaining work belongs to the acceptance boundary: migrate the three social
icons and pixel font in Sim;X, compare pixels and command/upload counts, then
remove only the superseded local adapters.

## Strict Scope

This document records only confirmed limitations or friction in the rendering
surface exposed by Sim;Engine and the concrete effect on the Sim;X rendering
adapter.

It must not contain complaints about:

- scientific simulation or numerical models;
- Sim;X domain architecture;
- product behavior or UI policy;
- input handling, window creation, audio, persistence, or plugins;
- features Sim;X has not attempted to render.

An item is added only after checking the local `0.1.0` source/API and the
relevant rendering path. Missing convenience is not automatically a defect.
Each item must distinguish an engine limitation from a deliberate Sim;X
workaround.

## Current Priority Summary

| ID | Sim;X severity | Immediate pressure |
|---|---|---|
| R-001 | high | supplied images and SVG-derived assets |
| R-002 | medium | filled vector artwork and arbitrary 2D geometry |
| R-003 | medium | ordering specialized meshes with ordinary scenes |
| R-004 | medium now, high later | scientific text, units, logs, localization |
| R-005 | high | bounded frame construction and tessellation work |
| R-006 | high | fixed UI chrome rendered through a world-only scene |
| R-007 | high | an independently controlled scientific viewport |
| R-008 | medium to high | avoiding full static UI tessellation every frame |

The priority reflects integration pressure on Sim;X, not a general quality
score for Sim;Engine. R-006 through R-008 became concrete only after Sim;X
implemented a full-screen Editor and a separately running View.

## R-001 — No Generic 2D Image or Sprite Primitive

Severity for Sim;X: **high integration friction**.

Observed API:

- `Scene` accepts circles, rectangles, lines, and polylines;
- retained renderer resources cover prepared scenes, dynamic meshes,
  particles, scalar fields, render targets, trails, and retained 3D;
- no public generic RGBA image, sampled texture, sprite, texture-atlas region,
  or image command can be inserted into a 2D `Scene`;
- there is no direct SVG rendering entry point.

Concrete impact:

Sim;X cannot draw supplied GitHub, YouTube, and Telegram assets as ordinary
images inside the existing UI scene. Rasterizing SVG is straightforward, but
there is no scene image primitive to consume the resulting pixels.

Current Sim;X workaround:

Rasterize each small SVG once in the presentation adapter, run-length encode
the visible pixels, and submit the runs as bounded colored rectangles. This
preserves the supplied artwork but increases scene-command and tessellation
work. It is acceptable for three small static icons, not as a general image
system.

Desired renderer capability:

- create or restore a bounded immutable RGBA texture;
- draw a texture or atlas sub-rectangle in `Scene` with position, size, tint,
  opacity, filtering mode, layer, and clip;
- report capacity/upload failures structurally;
- retain enough CPU-side data or a host restoration contract for device loss.

## R-002 — No Arbitrary Filled 2D Path or Polygon in `Scene`

Severity for Sim;X: **medium integration friction**.

Observed API:

`Scene` exposes rounded rectangles and circles as filled shapes, but line and
polyline primitives are stroke-only. It has no public arbitrary filled path,
filled polygon, or caller-provided triangle command.

Concrete impact:

Even after parsing an SVG path, the Sim;X adapter cannot submit the GitHub
silhouette, Telegram paper plane, or YouTube play triangle as filled scene
geometry. Approximating these with thick lines changes the artwork.

Current Sim;X workaround:

The same bounded raster-run adapter used for R-001. Sim;X must not copy
object-specific icon approximations into unrelated UI scene code again.

Desired renderer capability:

- a validated filled polygon/path primitive with explicit vertex/segment
  budgets; or
- a way to insert bounded colored 2D triangles into the same ordered/clipped
  scene command stream.

## R-003 — Dynamic 2D Mesh Is Not a `Scene` Composition Primitive

Severity for Sim;X: **medium integration friction**.

Observed API:

`DynamicMesh2d` accepts ready triangles, but its public render call renders the
dynamic mesh as its own surface path. A dynamic mesh cannot be inserted at a
specific layer among ordinary `Scene` commands. The specialized layered path
combines scalar fields and particles, not a generic scene plus dynamic mesh.

Concrete impact:

Triangulating SVG paths into `DynamicMesh2d` would not by itself solve icon
rendering: the icons must participate in the same UI ordering, clipping, and
surface presentation as panels and text.

Current Sim;X workaround:

Keep icon geometry in the ordinary scene through bounded raster rectangles.
Do not add a second independent surface render that could erase or reorder the
UI.

Desired renderer capability:

- compose a retained/dynamic 2D mesh with a `Scene` in one ordered frame; or
- expose a general render graph/pass API whose ordering, load operation,
  clipping, blending, and recovery semantics are explicit.

## R-004 — No Text or Glyph Rendering Path

Severity for Sim;X: **medium current cost, potentially high later**.

Observed API:

Sim;Engine `0.1.0` has no font, glyph atlas, shaped text, or text scene
primitive.

Concrete impact:

The current shell implements a tiny fixed 3-by-5 pixel font using one rectangle
per lit pixel. This is sufficient for a prototype title and short labels, but
not for scientific notation, localization, long logs, searchable object names,
accessibility scaling, or readable dense inspectors.

Unsupported glyphs currently become empty 3-by-5 patterns. Sim;X therefore
spells derived units as forms such as `M/S/S` instead of using superscripts and
cannot faithfully display symbols such as micro, degree, delta, integral,
Greek variable names, or user-provided non-ASCII names. Every visible glyph
also multiplies the ordinary `Scene` command count because each lit pixel is a
separate rectangle.

Current Sim;X workaround:

The pixel font remains presentation-only and bounded. It must not grow into a
home-made general typography engine.

Desired renderer capability:

- host-provided or engine-managed glyph atlas support in the 2D scene;
- Unicode text with explicit font selection and fallback policy;
- measurable layout, DPI-aware sizing, clipping, tint/opacity, and structured
  atlas-capacity failures;
- no requirement that Sim;Engine own product localization or UI layout.

## R-005 — `Scene` Has No Command or Tessellation Work Budget

Severity for Sim;X: **high safety responsibility in the adapter**.

Observed API:

`Scene::try_push_to_layer` validates the command and inserts it into an ordinary
`Vec<SceneCommand>`. The API has no maximum command count, vertex/tessellation
budget, byte budget, fallible reservation outcome, or `SceneCapacity` error.
The particle path has explicit resource budgets, but the ordinary 2D scene path
does not expose an equivalent limit.

Concrete impact:

A renderer adapter can construct an arbitrarily large scene and reach
unbounded CPU work/allocation before presentation. R-001 makes this especially
relevant: a naive image-to-rectangle conversion could emit one command per
pixel.

Current Sim;X workaround:

Every adapter-generated collection must have a Sim;X-owned hard ceiling before
scene submission. Social SVGs are rasterized to a fixed 32-by-32 grid and
run-length encoded; each icon is rejected if it exceeds 1,024 runs. This is a
local guard, not an engine-wide guarantee.

Desired renderer capability:

- a configurable per-scene command, generated-vertex, and allocation budget;
- validation before unbounded allocation/tessellation;
- a structured budget-exceeded result and metrics for submitted, accepted,
  dropped, and rendered work.

## R-006 — No Logical-Screen Geometry Layer for Fixed UI

Severity for Sim;X: **high integration friction**.

Observed API:

- circle, rectangle, line, and polyline positions are expressed in world
  coordinates and transformed by one `Camera2d`;
- line widths, shadows, and `ScreenClipRect` use logical screen pixels, but
  there are no equivalent logical-screen rectangle, circle, line, image, or
  text commands;
- a command cannot select `World` or `LogicalScreen` as its coordinate space.

Concrete impact:

The complete Sim;X shell is fixed UI chrome, but it must be encoded as world
geometry. `scenes.rs`, `pixel_font.rs`, and `social_icons.rs` each convert
top-left logical UI coordinates into centered world coordinates. The presenter
then holds a synthetic camera at origin with zoom `1.0` so those world values
behave like pixels.

This works only by flattening renderer coordinates and UI coordinates into the
same temporary convention. A real world camera cannot pan, zoom, rotate, or
use pseudo-depth without also moving the menu, panels, text, and icons. The
adapter duplicates conversion helpers and owns invariants that should be
expressed by a renderer coordinate-space type.

Current Sim;X workaround:

Keep the surface camera fixed at origin and one world unit per logical pixel.
Project every UI rectangle and point manually before scene insertion. Project
scientific snapshots into canvas pixel positions in Sim;X instead of giving
their world values to an independently controlled viewport camera.

Desired renderer capability:

- logical-screen variants of ordinary 2D primitives, images, and future text;
- or an explicit coordinate-space value captured by each scene command;
- DPI-aware logical pixels with the same validation, clipping, ordering,
  opacity, and recovery behavior as world commands;
- composition with world-space commands without requiring Sim;Engine to own
  Sim;X layout or input policy.

## R-007 — No General 2D Scene Viewport or Multi-Camera Composition

Severity for Sim;X: **high product and integration friction**.

Observed API:

- the ordinary surface path renders one `Scene` through one `Camera2d`;
- `ScreenClipRect` limits fragments but does not provide a different camera or
  projection inside the clipped region;
- `RenderTarget2d` can receive specialized scalar-field and particle paths,
  while retained 3D has its own target path;
- there is no public generic `Scene`-to-`RenderTarget2d` operation and no
  general API that renders several 2D scenes with separate cameras and
  viewports into one ordered surface frame.

Concrete impact:

The Sim;Phys Editor and View both contain a scientific canvas surrounded by
fixed panels. The scientific canvas needs its own pan, zoom, scale, and clip,
while UI chrome must remain fixed. The current renderer surface cannot express
that relationship directly.

Sim;X therefore calculates body positions, field probes, thermal nodes, and
wave samples directly into the canvas rectangle and draws them through the
same fixed camera as the UI. This is a display mapping, not a reusable viewport
camera. It blocks a correct implementation of middle-mouse world panning,
camera zoom, multiple scientific viewports, or a minimizable viewport without
expanding the adapter workaround.

Current Sim;X workaround:

Use one full-screen `Scene` and one fixed camera. Treat the scientific canvas
as a rectangle in UI coordinates, manually map every scientific snapshot into
that rectangle, and keep all camera-like state outside the renderer.

Desired renderer capability:

- render an ordinary `Scene` into a bounded logical viewport with its own
  `Camera2d` and screen clip; or
- render an ordinary `Scene` to `RenderTarget2d`, then compose that target with
  UI and other targets in a declared order;
- explicit load, blend, clear, clip, viewport, and failure semantics;
- one frame submission that can combine multiple cameras without one pass
  accidentally clearing or replacing another.

## R-008 — Prepared Scenes Cannot Be Layered with Dynamic Scene Content

Severity for Sim;X: **medium current cost, high for denser workspaces**.

Observed API:

- `PreparedScene` retains and restores one complete immutable scene;
- `render_prepared` presents that prepared scene as its own surface path;
- an ordinary `Scene` cannot contain a `PreparedScene`, and there is no public
  layered path combining prepared geometry with a changing `Scene`;
- changing geometry, style, order, or clipping requires preparation again.

Concrete impact:

Most of a Sim;X workspace is static or changes rarely: panel backgrounds,
canvas borders, grid lines, long labels, and many pixel-font rectangles. A
small part changes every frame: hover animation, playback measurements, a
moving body, or a wave line. Because the static and dynamic portions cannot be
composed through the same ordered frame, Sim;X rebuilds and re-tessellates the
entire UI scene on every redraw.

R-004 amplifies this cost because even unchanged text is many rectangles.
R-001 amplifies it again because unchanged icons are many raster-run
rectangles. Preparing the entire scene is not useful while any measurement or
hover state changes, and issuing separate surface render calls would not
preserve one explicit ordered composition.

Current Sim;X workaround:

Build one fresh `Scene` per rendered frame. Keep the UI visually simple and
bound adapter-owned collections. Do not introduce a home-made retained scene
cache whose ordering, clipping, invalidation, and device-loss behavior would
duplicate renderer responsibilities.

Desired renderer capability:

- insert a prepared subscene into a frame with an explicit layer, transform,
  opacity, and clip; or
- submit prepared and streaming scene batches together through one ordered
  frame/pass builder;
- validate ownership and recovery generation for every retained input;
- expose separate metrics for reused prepared work and newly tessellated work.

## Confirmed Capabilities That Are Not Gaps

The following are already present in Sim;Engine `0.1.0` and must not be added
later as complaints:

- nested logical-screen clipping through `ScreenClipRect` and
  `Scene::with_screen_clip`;
- stable layer plus insertion ordering for ordinary scene commands;
- solid, linear-gradient, and radial-gradient fills;
- structured fallible scene insertion APIs alongside boolean conveniences;
- CPU-side frame and tessellation metrics, including accepted, rendered, and
  dropped command counts;
- explicit budgets for particle and selected texture/resource paths;
- retained-resource restoration APIs after device recovery.

These capabilities may still be unused or incompletely integrated by Sim;X.
That would be a Sim;X adapter issue, not a Sim;Engine rendering gap.

## Review Template for New Items

```text
## R-NNN — Renderer-only title

Severity for Sim;X:
Observed API:
Concrete impact:
Current Sim;X workaround:
Desired renderer capability:
```

Before adding an item, confirm that the problem is actually in rendering. For
example, window/event-loop ownership belongs to host integration and must not
be recorded here merely because Sim;Engine does not create a window.
