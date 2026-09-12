# Sim;Engine Documentation

This document contains the published 0.3.0 integration guide plus the explicit
[0.4 development preview](#04-development-preview) below. The checkout version
is `0.4.0-dev.5`; it is not a final 0.4.0 release. For older integrations, use the
[archived 0.2 guide](https://github.com/lexawhatt/Sim-Engine/blob/v0.2.0/DOCUMENTATION.md).
The [0.3.0 changelog](CHANGELOG.md#030---2026-09-11) includes migration notes.
This guide is divided into two parts:

- [Integration Handbook](#part-i-integration-handbook) - how to add the crate,
  construct visual state, choose a rendering path, recover resources, and
  measure performance.
- [Engineering Reference](#part-ii-engineering-reference) - how the crate is
  structured, which invariants it enforces, and where its responsibilities
  end.

For exact signatures and error variants, generate the Rust API reference:

```bash
cargo doc --all-features --no-deps --open
```

Sim;Engine remains pre-1.0. Linux with
Vulkan is its supported release target, and Rust 1.90 is the minimum supported
Rust version.

## 0.4 development preview

This combined development candidate includes the additions below. Its git
handoff is separate from a final registry release and host integration approval.
Development handoff uses a tested git commit, pinned with Cargo's `rev` field
and the host lockfile, not an assumed stable `master` branch. A moving `_DEV`
label is a convenience, not reproducible release evidence. The maintainer will
provide the qualified revision after integration checks; do not publish this
preview to crates.io as the completed release.

The 0.4.0 milestone prioritizes the requested integration capabilities, their
correctness/resource contracts, and reproducible performance measurements.
Intensive profiling-driven optimization is planned for 0.4.1. Existing reuse
requirements and regression gates remain part of 0.4.0; the later optimization
phase is not a prerequisite for the Sim;Logic DEV test drive.

### Consumer corrections and in-place presentation changes (dev.5)

Use the dev.5 pin or later for standalone `renderer.restore_mesh3d(&mesh)`.
Dev.4 incorrectly recreated a legacy opaque material in that route: transparent
pixels/tints could reject, and UV/addressing/alpha-rebinding policy could reset.
Restoration now preserves complete mip pixels/options, sampling, tint, signed
UV transformation, addressing, opaque/alpha-capable rebinding policy and every
reserved geometry/attribute capacity. Existing source aliases do not change.
Separate standalone restore calls recreate separate GPU resources; use
`renderer.restore_scene3d(&mut scene)` when shared-resource deduplication and
stable object IDs across device replacement are required. Surface styles still
belong to scene objects, not standalone mesh handles.

For an animated clear color, mutate the existing scene:

```rust
use sim_engine::{Color, Scene3d};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut scene = Scene3d::new(Color::BLACK)?;
scene.set_background(Color::rgba(0.1, 0.2, 0.3, 0.5))?;
# Ok(())
# }
```

The setter explicitly accepts normalized straight-linear RGBA, even on a scene
created by the legacy opaque constructor. Non-finite or out-of-range channels
return `Scene3dError::InvalidBackground` without modifying the previous color.
This performs no allocation or GPU upload and changes no IDs, object/material
state, resource capacity, lighting or fog. Offscreen rendering premultiplies
the clear color; do not premultiply the setter input. It changes the target
clear, not an environment sky mesh or simulation state.

Use `scene.set_texture_material(id, Some(&material))` to attach/replace a
texture material, and `scene.set_texture_material(id, None)` to remove it.
The object's topology, UVs, normals, vertex colors, edge buffers and reserved
capacities are shared unchanged. No GPU allocation, copy or upload occurs;
fallible CPU scene-accounting reservation can still be needed for new texture
references. Existing mesh handles and other scene objects retain their previous
materials. Transform, surface/edge style, visibility and stable ID do not change.
`mesh.without_material()` returns an independent material-free handle with the
same buffers if a standalone snapshot is preferred. It does not restore a stale
resource onto a new device.

Material rebinding validates the object ID first, then texture/mesh generation,
UV availability, styles and final scene resource/storage limits. Errors leave
the old visual state intact; texture-specific failures are nested in
`Scene3dError::Texture`. The `Scene3dMeshUpdateReport` includes unique scene-wide
old-plus-incoming CPU/GPU peaks, just like `set_mesh`; these are reports, not new
peak limits or a promise that previously submitted resources retire immediately.
Already uploaded incoming textures and external aliases remain host-owned.
Detaching the last scene reference removes its texture bytes from final scene
statistics, not necessarily from the process or GPU driver.

### CPU shaping without GPU dependencies

Enable `fonts` with default features disabled for CPU font/metrics work. The
existing `text` feature still enables both fonts and wgpu text integration.
`FontFace::shape_line` remains available. A caller-owned session reduces repeated
face parsing, plan construction and storage allocation for changing labels:

```rust
use sim_engine::{FontBudget, FontFace, LogicalPixels, PhysicalPerLogical,
    TextLayoutBudget, TextShapingSession, TextStyle};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let font = FontFace::from_bytes(std::fs::read("assets/Label.ttf")?, FontBudget::default())?;
let style = TextStyle::new(LogicalPixels::new(24.0)?, PhysicalPerLogical::new(1.25)?)?;
let budget = TextLayoutBudget::default();
let mut shaping = TextShapingSession::new(&font, style, budget)?;
let mut line = shaping.shape_line("Count: 100")?;
shaping.update_line(&mut line, "Count: 101")?;
# Ok(())
# }
```

Each session fixes font/style/work limits, retains at most one shaping plan
matched to resolved script/direction/language, and has no global string cache.
Script changes may rebuild that plan and allocate. Engine scratch counters
exclude dependency-owned face/plan/buffer allocations whose capacities are not
exposed; this is not a hostile-font memory sandbox. `clear_scratch` releases the
session's scratch/plan without invalidating previously returned lines.

With `text`, pass the immutable CPU line to
`TextAtlas2d::prepare_from_shaped(renderer, &line, budget)`, then update with
`update_from_shaped(renderer, &mut run, &line, budget)`. These methods do not
invoke shaping again. They require the exact originating font identity and
style, including DPI and requested direction; font clones share identity,
independently loaded identical font bytes do not. Changed-run failures preserve
the old drawable, with the existing bounded atlas-cache-warming exception.
Compatible unchanged GPU updates report zero work and do not consume the work
budget; provenance checks still run first.

Measure the named CPU workloads independently of a window/GPU:

```bash
cargo run --release --no-default-features --features fonts \
  --example text_shaping_benchmark -- --iterations 1000
```

### Surface policy, preflight budgets and source diagnostics

The default `StrictPortable` surface policy is unchanged. An application whose
free camera routinely produces grazing/edge-on surfaces can explicitly select
native rasterization:

```rust
use sim_engine::{Mesh3dRenderBudget, SurfaceRasterization3d};

let budget = Mesh3dRenderBudget::new(0, 0, 0)
    .with_surface_policy(SurfaceRasterization3d::Native)
    .with_max_surface_triangles(200_000);
```

Pass this budget to `validate_scene3d_for_target` or
`render_scene3d_to_target_with_budget`. Native keeps original indexed triangles
and lets hardware clip them. It does not divide surface endpoints on the CPU,
drop uncertain source triangles, or weaken overflow validation. Raster coverage
at numerical boundaries is adapter-dependent. Display-edge clipping/extrusion
remains independently validated and may still reject an object.

`submitted_triangle_count` is the exact retained-plus-generated submission
count, **not** the number of visible triangles. Outside retained indices count
if submitted. `generated_object_count` describes objects replaced by canonical
CPU clipping geometry. Native generates none and returns `None` from the CPU
clipped/discarded-source count accessors; strict mode returns `Some(count)`.
Budget failure occurs before generated staging reservation, uploads or target
mutation, although bounded per-object preflight metadata may be allocated.

For an error, `object_id`, `source_triangle_index`, `source_vertex_index` and
`surface_reason` expose applicable source context without another clipping pass.
Triangle indices refer to original index triples, not generated fans. A bad
shared vertex is attributed to its first referencing source triangle; vertices
with no surface reference retain vertex context. Camera/resource/aggregate
capacity errors remain scene-level, and edge errors remain object-attributed.
See the changelog for exhaustive-match and `Option` migrations from 0.3.0.

### Application-supplied vertex colors

`Mesh3dAttributes` accepts optional per-vertex linear RGBA colors and UVs without
GPU dependencies. Attribute values are owned by the immutable source mesh;
building it does not change any earlier CPU or retained GPU snapshot.

```rust
use sim_engine::{Color, Mesh3d, Mesh3dAttributes, Vec3};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let attributes = Mesh3dAttributes::new().with_vertex_colors(vec![
    Color::rgb(1.0, 0.0, 0.0),
    Color::rgb(0.0, 1.0, 0.0),
    Color::rgb(0.0, 0.0, 1.0),
])?;
let mesh = Mesh3d::with_attributes(
    vec![Vec3::ZERO, Vec3::X, Vec3::Y],
    vec![0, 1, 2],
    Vec::new(),
    attributes,
)?;
# let _ = mesh;
# Ok(())
# }
```

All channels must be finite and normalized to `0..=1`; unsupported HDR or NaN
values return an error with the source vertex index instead of being clamped.
When supplied, the color array must have exactly one entry per vertex, including
vertices unused by the index list. Missing colors multiply by white and preserve
the colorless rendering path. Explicitly supplied empty colors are not a valid
attribute array for a nonempty mesh. Use `with_texture_coordinates` on the same
descriptor to supply UVs as well.

Colors interpolate perspective-correctly. RGB multiplies the surface color,
optional sampled texture and texture-material tint in linear space. Strict CPU
clipping carries colors with the same intersection parameter as positions/UVs;
Native leaves this interpolation to the GPU. Mathematical-edge colors remain
independent. Opaque deliberately ignores combined alpha and writes alpha one;
Mask and Blend use the combined alpha as described below. Legacy texture
creation still requires opaque texels; alpha-bearing textures use the explicit
new creation path.

Meshes without colors allocate no color buffer. Colored sources, uploads,
generated clipping streams and dynamic-update scratch count the additional
storage in their existing budgets. Restoration preserves the exact colors and
reserved capacity; color attribute layout changes follow the same whole-bundle
replacement contract as UV changes.

### Alpha materials and surface sidedness

`SurfaceStyle3d` now separates coverage from tint. Input colors are normalized
straight-linear RGBA; texture bytes are sRGB RGBA8 with linear alpha. The shader
multiplies surface color, vertex color, sampled texture and texture-material tint.
Missing vertex colors or textures multiply by white.

| Mode | Coverage | Depth |
| --- | --- | --- |
| `opaque(color)` | Ignores combined alpha; output alpha is one | Test and write |
| `mask(color, cutoff)` | Discards combined alpha below cutoff; equality survives | Test and write for surviving fragments |
| `blend(color)` | Uses combined alpha | Test, no write |

```rust
use sim_engine::{Color, SurfaceSidedness3d, SurfaceStyle3d};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let solid = SurfaceStyle3d::opaque(Color::rgb(0.4, 0.6, 0.8))?;
let cutout = SurfaceStyle3d::mask(Color::WHITE, 0.5)?;
let ghost = SurfaceStyle3d::blend(Color::rgba(0.2, 0.8, 1.0, 0.35))?
    .with_sidedness(SurfaceSidedness3d::FrontOnly);
# let _ = (solid, cutout, ghost);
# Ok(())
# }
```

Two-sided rendering remains the default. `FrontOnly` retains projected
counter-clockwise faces and culls back faces; it does not infer globally outward
normals or fix reversed source topology. Transform scales remain positive.
Sidedness does not remove explicit display edges.

Mask cutoff must be zero or a normal finite value in `0..=1`; positive subnormal
cutoffs are rejected rather than relying on backend flush-to-zero behavior.
The comparison uses the filtered/interpolated alpha, not source texel coverage;
Mask does not imply alpha-to-coverage or coverage-preserving mip filtering.

For alpha-bearing image data use
`renderer.create_texture3d_rgba8_with_alpha(width, height, pixels, budget)` and
`TextureMaterial3d::with_alpha(&texture, sampling, tint)`. The legacy
`create_texture3d_rgba8` and `TextureMaterial3d::new` retain their opaque-input
validation. Transparent texels only create holes under Mask or transparency
under Blend; an Opaque surface ignores their alpha. Legacy creation keeps one
clamp-addressed mip level. Dev.4 adds explicit mip generation, tile isolation
and addressing options described below.

Opaque and masked surfaces render first. Blend objects follow back-to-front,
using the camera-forward distance of each transformed model-bounds center and
insertion order for equal keys. This works for either camera projection and
does not mutate `Scene3d::instances()` or object IDs. It is object-level sorting:
intersecting/cyclic transparent surfaces and triangle order within one mesh can
still produce incorrect overlap. It is not order-independent transparency,
physical glass, refraction or a general translucent-section solution.

Mathematical edges render last. Blend surfaces do not write the depth that
classifies those edges; Mask holes likewise do not occlude them. Surviving
masked fragments and opaque surfaces retain normal depth occlusion.

`Scene3d::with_alpha_background(color)` opts into a normalized straight-alpha
clear color; `with_alpha_background_and_budget` also accepts explicit scene
limits. The old `new`/`with_budget` constructors still require opaque clears.
Renderer targets store premultiplied color, including the clear, for use with
existing target/frame composition. Do not premultiply the input colors yourself.
Device and scene restoration preserve alpha texture data and material styles.

Blend sorting uses bounded fallible staging before uploads or target clearing.
Set `Mesh3dRenderBudget::with_max_sorting_bytes` to bound it; the preflight
report's `sorting_capacity_bytes()` describes this transient allocation, not
additional retained mesh memory. Scenes without Blend need no sorting array.

Manual inspection (development checkout, not registry 0.3.0):

```bash
cargo run --release --example materials_3d
cargo run --release --example materials_3d -- --acceptance
```

Use A or 1/2/3 to rotate material assignments across the three panels, C for
sidedness, arrows to orbit, P for camera projection, Space to pause, R to reset,
F5 for device recovery and Esc to exit. The window title labels the current
panels. The acceptance mode requires 120 confirmed presents and exercises
material, sidedness, projection and recovery transitions; it is not an FPS gate
or a substitute for the automated pixel tests.

### Normals, lighting and distance fog (dev.3)

Lighting is opt-in and does not change the default scientific/stylized color
path. Supply one nonzero model-space normal per source vertex through
`Mesh3dAttributes::with_normals`; Engine robustly normalizes these directions
but does not generate face normals or smoothing groups. Duplicate vertices
along hard normal seams. Supplied arrays include unused vertices and must
match the vertex count exactly; omission is different from an empty array.

```rust
use sim_engine::{AmbientLight3d, Color, DirectionalLight3d, Fog3d, Lighting3d,
    Mesh3d, Mesh3dAttributes, SurfaceLighting3d, SurfaceStyle3d, Vec3};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let attributes = Mesh3dAttributes::new().with_normals(vec![Vec3::Z; 3])?;
let source = Mesh3d::with_attributes(
    vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![0, 1, 2], vec![], attributes,
)?;
let style = SurfaceStyle3d::opaque(Color::rgb(0.5, 0.7, 0.9))?
    .with_lighting(SurfaceLighting3d::Lambert)
    .with_fog(true);
let ambient = AmbientLight3d::new(Color::WHITE, 0.2)?;
// World-space direction from the surface toward the light, not a position.
let sunlight = DirectionalLight3d::new(Vec3::new(1.0, 2.0, 3.0)?, Color::WHITE, 0.7)?;
let lighting = Lighting3d::new(ambient).with_directional(Some(sunlight));
// Start is world distance; density has inverse-world-distance units.
let fog = Fog3d::new(Color::rgb(0.15, 0.2, 0.3), 20.0, 0.025)?;
# let _ = (source, style, lighting, fog);
# Ok(())
# }
```

Apply these values with `scene.set_lighting(lighting)` and
`scene.set_fog(Some(fog))`. The default environment is white ambient at intensity
one, no directional light and no fog; materials default to `Unlit` and
`with_fog(false)`. `Lighting3d::with_directional(None)` removes the directional
term; `scene.set_fog(None)` disables scene fog. These are ready visual settings,
not Engine-owned light entities or simulation state.

Light colors are normalized opaque linear RGB and intensities are finite in
`0..=1`. The Lambert term uses the nonnegative dot product of the world normal
and direction toward the light, plus ambient illumination. RGB is intentionally
saturated to the normalized LDR range. Lighting multiplies the base
surface/vertex/texture/material RGB; it never changes their combined alpha.
Opaque, Mask and Blend keep the coverage/depth behavior described above.
There is no PBR, specular reflection, refraction, shadowing or point-light array.

Normals use inverse-transpose direction transport under nonuniform scale.
The same unnormalized transformed attributes interpolate through Native and
StrictPortable surfaces, then normalize in the fragment; an exactly cancelled
interpolated normal receives ambient illumination only. Two-sided back faces
flip the shading normal. `FrontOnly` still uses source winding for culling,
not the supplied normals. Unusable transformed source normals fail before
render submission when Lambert is active; `Unlit` does not evaluate them.

Fog is applied after lighting to RGB using:

```text
amount = 1 - exp(-density * max(camera_forward_world_distance - start, 0))
rgb = illuminated_rgb * (1 - amount) + fog_rgb * amount
```

This is distance along the camera's forward direction, not radial distance,
clip W or normalized depth; orthographic cameras follow the same rule. Start
and density must be zero or positive normal finite values. At/before the start,
or at zero density, the original color is preserved. Very dense fog saturates
without overflowing the shader product. `Unlit` materials can still opt into
fog, or keep exact colors by opting out. Mathematical edges and the scene clear
color are not lit or fogged. Fog does not cull geometry, reduce uploads or hide
it from resource/submission budgets.

Normal arrays and active environments survive scene restoration. Normal buffers
participate in source/upload/generated-geometry budgets and dynamic whole-bundle
copy-on-write. Updating a Lambert object must retain usable normals; change its
style to Unlit before removing that attribute. No normal buffer is allocated
for a legacy normal-less mesh.

```bash
cargo run --release --example lighting_3d
cargo run --release --example lighting_3d -- --acceptance
```

The gallery compares Unlit/Lambert objects and fog-participating/control rows.
L toggles sunlight, F fog, N nonuniform scale, M alpha mode, C sidedness and P
projection. Use arrows to orbit, Space to pause, R to reset, F5 to recover and
Esc to exit. The bounded acceptance run counts confirmed presents across these
states and recovery; analytic GPU pixel tests separately verify the formulas.

### Dynamic mesh revisions and capacity reuse

Use `renderer.update_scene3d_mesh(&mut scene, object_id, new_source, budget)`
to change one object's geometry. `new_source` is an already validated, nonempty
`Mesh3d`; its construction stays in the host. Hide or remove an empty chunk.
Object ID, transform, visibility, style and texture material are preserved.
A textured material still requires source UVs; Lambert requires source normals.

`DynamicMesh3dBudget::new(Mesh3dUploadBudget::default())` accepts explicit
`with_minimum_capacity(vertices, indices, display_edges)` reserves. Indices
are capacity counts, not triangles; unused capacity need not form whole triples.
Final source/GPU/scratch limits and `with_peak_limits(recovery, gpu, staging)`
bound nominal old/new overlap before writes. Peak limits cover this operation's
two revisions, not unrelated host snapshots or opaque driver allocations.

If the scene object uniquely owns every GPU buffer and the capacity/layout fits,
the buffers are reused. If another scene object or an exported mesh/instance
clone shares them, the update creates a separate complete bundle. After that
detachment, later unique updates reuse capacity. Creating a snapshot with
`scene.instance(id)?.mesh().clone()` deliberately restores copy-on-write behavior
for the next update. CPU source snapshots do not by themselves prevent GPU reuse.

Growth or adding/removing UV/color/normal layout also replaces the complete bundle, keeping
unique allocation accounting correct. Synchronous validation/allocation errors
preserve the prior drawable; GPU device loss follows the existing asynchronous
recovery contract. Both `restore_mesh3d` and `restore_scene3d` preserve reserved
buffer capacities, not just the current live topology.

Every accepted update uploads all live positions, indices, UVs, colors and display
edges, even when the supplied source is unchanged. Partial mesh edits and an
unchanged-source no-op are not promised by this path. The returned report
separates uploaded/live bytes, reserved bytes, buffer allocations, alias
detachment and scratch reuse. `mesh3d_update_scratch_bytes` and
`clear_mesh3d_update_scratch` expose renderer-owned conversion storage, separate
from `Scene3dStatistics` and `FrameCacheBudget`. Clear it before applying a
smaller retained-scratch ceiling than an earlier update required.

Compare identical prevalidated inputs through both update routes:

```bash
cargo run --release --example mesh3d_scene_benchmark -- \
  --case immutable --objects 64 --side 32 --frames 120 --trials 3
cargo run --release --example mesh3d_scene_benchmark -- \
  --case dynamic --objects 64 --side 32 --frames 120 --trials 3
```

This changes one object per frame; the remaining objects share a static mesh.
Initial detachment and scratch growth are reported during warmup, separately
from steady-state allocation counts. Other cases include `repeated`, `outside`,
and `host_hidden`, for example `--objects 1024 --side 1`. `outside` still submits
the geometry; `host_hidden` explicitly hides the same distant objects. No
automatic distance/frustum culling is inferred.

All measured frames must be `Drawn`, source revision order is deterministic,
and output changes abort measurement. The target stays 1280x720; actual surface
size/DPI and adapter/presentation metadata are logged. CPU work excludes
surface acquire; completed-batch wall FPS includes the final GPU completion
drain. Neither is GPU execution time; dev.4 separately enables bounded timestamp
queries and matches them to measured report IDs where supported. Mesh
upload/allocation counters exclude camera/composition resources; thread-local
allocation counts include backend calls on that thread, not worker threads.
Host source snapshot bytes overlap scene CPU bytes and must not be added to
them. This baseline does not replace the release matrix. Intensive large-scene optimization
using these measurements is planned for 0.4.1.

### Texture lifecycle and repeating tiles (dev.4)

The default texture constructors preserve the old single-level contract.
For reduced shimmering at distance, request a complete chain explicitly:

```rust,ignore
let options = Texture3dOptions::new()
    .with_mipmaps(TextureMipmaps3d::Generate)
    .with_alpha_preservation(true);
let texture = renderer.create_texture3d_rgba8_with_options(
    width, height, rgba8_pixels, ImageBudget::default(), options,
)?;
let material = TextureMaterial3d::with_alpha(
    &texture, ImageSampling::Linear, Color::WHITE,
)?
.with_uv_transform(TextureUvTransform3d::new(
    Vec2::new(8.0, 4.0), Vec2::new(-0.25, 0.125),
)?)
.with_address_mode(TextureAddressMode3d::Repeat);
let textured_mesh = renderer.with_mesh3d_material(&mesh, &material)?;
```

UV transform order is `uv * scale + offset`. Zero scale selects a constant
coordinate; negative scale mirrors the image, and negative coordinates repeat
normally. The validated endpoint domain is bounded to +/-65536. Wrapping happens
in the sampler, not through a discontinuous shader `fract`, so derivatives used
for implicit LOD remain continuous. Linear sampling interpolates texels and mip
levels; nearest sampling selects nearest texels/levels. No anisotropy is implied.

Mip generation filters linear-light RGB with premultiplied-alpha weights, then
stores straight sRGB RGBA8 again. NPOT levels include the entire preceding image;
one-pixel axes remain one pixel. Fully transparent output has zero RGB. This
avoids hidden RGB contaminating generated levels; it does not promise
alpha-aware *base-level bilinear filtering*. Mask coverage is not preserved:
thin cutouts can disappear under minification. These are explicit quality
limitations, not a physically based material model.

For an atlas, crop first, then generate the isolated tile's chain:

```rust,ignore
let tile = renderer.crop_texture3d_tile(
    &atlas_texture, ImageTexelRect::new(tile_x, tile_y, tile_width, tile_height)?,
    ImageBudget::default(), options,
)?;
// Use full 0..1 tile UVs and the same Repeat material settings above.
```

The cropped texture owns its pixels and does not follow later atlas edits.
Each tile has independent mip storage/bindings, a deliberate memory/draw tradeoff.
Generating a chain for a whole packed atlas is allowed as an ordinary image,
but is **not** tile isolation. `region_coordinates` rejects a mipmapped image;
an inset cannot repair neighbouring colours already mixed into lower levels.

`mip_level_count`, `mip_level_size` and `mip_level_pixels` expose committed CPU
levels. CPU recovery and nominal GPU byte counters include the entire chain,
not just mip zero. Image limits and GPU dimensions are checked before chain
creation. Restoring a texture/scene uploads the retained levels exactly.

### Partial 3D texture updates (dev.4)

`ImageTexelRect` uses physical source texels, not logical pixels or world units.
The supplied byte slice starts at the patch origin; its exact length is
`(height - 1) * bytes_per_row + width * 4`. Padding is allowed between rows.

```rust,ignore
let report = renderer.update_scene3d_texture_region(
    &mut scene, object_id, ImageTexelRect::new(12, 8, 4, 4)?,
    &patch_rgba8, 4 * 4, Texture3dUpdateBudget::default(),
)?;
println!("upload {} bytes, GPU copy {} bytes",
    report.uploaded_bytes(), report.gpu_copy_bytes());
```

Use `update_texture3d_region(&mut texture, ...)` for a standalone handle; existing
materials retain their previous immutable snapshot until explicitly rebound.
The scene-owned variant updates one stable object ID and preserves all other
objects/aliases. An unshared texture reuses its GPU allocation. A shared texture
gets a new allocation, GPU copies of the old mip levels, then the patch uploads;
it does not upload the unchanged base image from CPU memory. Material alpha
policy, UV transform, addressing, sampling and tint survive rebinding/recovery.

Every fallible validation and CPU preparation step precedes GPU writes. Failed
rectangle/stride/alpha/budget checks leave CPU pixels, GPU pixels and scene
accounting unchanged. Device loss remains an external GPU failure handled by
the existing recovery contract, not a promise to roll back physical hardware.

The no-mipmap path uploads only the patch. The initial mipmapped implementation
regenerates and uploads **all lower levels**, not only affected lower rectangles.
`Texture3dUpdateBudget` separately limits upload, staging, peak recovery, peak GPU
and GPU-copy bytes. Reports separate base/mip uploads, calls, GPU copies,
submissions, allocations, regenerated texels and CPU preparation/encoding costs.
This is a bounded, documented cost; finer dirty-mip optimization is 0.4.1 work.

Inspect the six-panel lifecycle gallery independently of benchmarks:

```bash
cargo run --release --example texture_lifecycle_3d
cargo run --release --example texture_lifecycle_3d -- --acceptance
```

Top: identical mip-zero pixels with mipmaps disabled/enabled. Middle: an
isolated atlas tile with positive/negative UV repetition. Bottom: a region-edited
texture next to its unchanged immutable alias. `U` edits; `T` toggles repeat;
`X` mirrors; `M` switches alpha mode; `L` lighting; `F` fog; `C` sidedness;
`P` projection. Arrows orbit, wheel/`+`/`-` zoom, Space pauses, `R` resets the view,
`F5` replaces/restores the logical device and Esc exits. The bounded acceptance
mode exercises these states and requires confirmed Drawn frames; manual
inspection remains separate from pixel-oracle qualification.
Since dev.5, the same recovery also restores four standalone meshes through
the public API: transparent texels, alpha tint, signed/repeating UVs and an
alpha-capable material whose current texture is opaque. It checks retained
attributes/reserves, every CPU mip, alpha-image rebinding, background mutation
and material removal without topology replacement. The mandatory GPU oracle
separately reads the restored mip textures and rendered pixels on a new device.

### GPU diagnostics and the full changing-scene matrix (dev.4)

GPU timestamps are opt-in and require only the adapter's `TIMESTAMP_QUERY`
feature. Enabling them is not a device requirement on unsupported adapters:

```rust,ignore
let options = WgpuRendererOptions::new(present_mode, scale_factor)?
    .with_gpu_timing(true);
// Initialize with these options, then render normally.
let submitted_id = report.gpu_timing_id();
for sample in renderer.collect_gpu_timings().samples() {
    // Match the ID to the exact submitted Scene3d/FrameComposer report.
    println!("{:?} {:?}: {:?}", sample.id(), sample.source(), sample.elapsed());
}
```

`GpuTimingStatus` distinguishes Disabled, Unavailable and Enabled. Samples cover
the named GPU render pass, not CPU preflight, texture transfers, query resolution,
queue waits, monitor scanout or total GPU utilisation. The fixed eight-slot ring
uses 16 query indices and 256 nominal buffer bytes; backend/query bookkeeping
is additional. Each submitted reading copies 16 bytes. Collection polls once
without waiting, returns at most eight samples and allocates no result vector.
Never match by arrival order. A full ring drops diagnostics, not drawing;
`gpu_timing_statistics` exposes losses, mapping/interval failures, pending slots,
readback bytes and CPU collection cost. Recovery discards old pending results.
Zero duration can mean work below timer resolution; Unavailable is never zero.

`Mesh3dRenderReport::preflight_duration` and `staging_upload_duration` split the
legacy combined `upload()` CPU interval. `encode_submit` stays CPU time.
The diagnostic example reports these separately from acquisition and GPU samples:

```bash
cargo run --release --all-features --example mesh3d_scene_benchmark -- \
  --case texture_update --policy native --objects 256 --frames 60 --trials 3
SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID=0000:01:00.0 \
  ./scripts/mesh3d_diagnostics_matrix.sh
```

Replace the PCI address with the adapter being qualified. The matrix takes a
read-only exact-commit source snapshot and records its executable hash. It
rejects source changes before publishing a uniquely named result directory;
failed `.pending` logs remain available. Direct example runs without this
wrapper are labelled unverified working-tree runs, not exact-SHA evidence.

Fixtures include shared/distinct resources; mostly outside/host-hidden/crossing
geometry under Native and StrictPortable; immutable/dynamic chunk updates;
capacity growth; repeated mipmapped tiles; Mask/Blend; region updates; changing
pre-shaped labels. Growth deliberately creates a fresh small mesh before a
larger update each frame and counts **both** allocations/uploads. Native
hardware-clipping totals are reported as unobserved rather than fabricated CPU
culling. The prepared-label fixture performs no measured reshaping; the separate
CPU shaping benchmark measures that cost.

Every measured attempt must return Drawn. Slow trials are printed independently;
skips/output transitions fail the attempt instead of inflating throughput.
Warmup, diagnostics losses, end-of-trial drain and retained/shared accounting
are explicit. On timestamp-capable devices, every measured frame must have both
matched pass samples before a trial qualifies; incomplete readings remain in
failed-attempt logs, not a successful GPU percentile baseline. Unsupported
devices explicitly report Unavailable. Query instrumentation adds overhead and is off by default in
applications. This matrix supplies platform-specific baselines for 0.4.1,
not a universal 60/100 FPS promise or a replacement for existing release gates.

## Part I: Integration Handbook

### 1. Scope

Sim;Engine renders ready visual state. The host application owns simulation
stepping, physics, mathematical meaning, entities, navigation, plugins, and UI
policy. The engine owns:

- validated 2D commands and visual styles;
- camera transforms, typed coordinate conversion, clipping, and pseudo-depth;
- visual interpolation;
- prepared, dynamic, image, glyph, particle, and scalar-field GPU resources;
- offscreen targets, composition, and bounded trails;
- retained stereometry meshes, depth, and visible or hidden display edges;
- renderer diagnostics, ownership checks, and resource recovery.

Sim;X is the primary consumer and design driver, but no Sim;X domain type is
required by the public API.

The renderer is not globally asynchronous. Adapter/device creation and device
recovery are async. Normal updates and render calls synchronously encode and
submit GPU work, without normally waiting for its completion.

### 2. Installation and features

The default feature set includes the `wgpu` backend:

```toml
[dependencies]
sim-engine = "0.3"
```

Use the CPU-side visual-state APIs without GPU dependencies:

```toml
[dependencies]
sim-engine = { version = "0.3", default-features = false }
```

| Configuration | Provides |
| --- | --- |
| default / `wgpu` | `WgpuRenderer`, GPU resources, targets, composition, particles, heatmaps, and retained 3D drawing |
| `text` (opt-in; includes `wgpu`) | TTF/OTF loading from bytes, single-line shaping, antialiased glyph rasterization, fixed-capacity font atlases and retained text runs |
| `--no-default-features` | scenes, cameras, colors, fields, particles, tweening, 3D math, mesh topology, and styles |

Window creation is deliberately outside the crate. The host may use `winit`,
as the examples do, or another framework that can provide a compatible
`wgpu::SurfaceTarget<'static>`.

`ParticleInstance2d` is part of the renderer-independent core. Hosts can
generate and benchmark particle visual state without compiling `wgpu`.
`ParticleField2d`, GPU upload, culling, and drawing require the renderer.

### 3. Frame model

A normal frame has four host-owned phases:

1. Advance the simulation or application state.
2. Convert that state into bounded Sim;Engine visual state.
3. Update reusable GPU resources that changed.
4. Render and inspect the returned report.

Do not move domain simulation into a scene builder or renderer callback. A
simulation can run on worker threads, then hand a ready visual snapshot to the
thread that owns the renderer.

### 4. Build a 2D scene

```rust
use sim_engine::{
    Camera2d, Color, Fill, LinearGradient, Rect, Scene, ShapeStyle, Vec2,
};

fn build_scene() -> Result<(Scene, Camera2d), Box<dyn std::error::Error>> {
    let mut scene = Scene::new(Color::rgb8(12, 14, 18))?;

    scene.try_circle(
        Vec2::new(24.0, 12.0),
        8.0,
        ShapeStyle::filled(Color::rgb8(86, 195, 255)),
    )?;

    scene.try_rect(
        Rect::from_center_size(Vec2::ZERO, Vec2::new(120.0, 64.0)),
        8.0,
        ShapeStyle::filled_with(Fill::LinearGradient(LinearGradient::new(
            Vec2::new(-60.0, 0.0),
            Vec2::new(60.0, 0.0),
            Color::rgb8(86, 195, 255),
            Color::rgb8(255, 190, 94),
        ))),
    )?;

    let camera = Camera2d::new(Vec2::ZERO, 2.0)?;
    Ok((scene, camera))
}
```

Use `try_*` methods when rejection must be observable. Convenience methods such
as `circle`, `rect`, `line`, and `polyline` return `bool`; use them only when
discarding invalid optional decoration is intentional.

| Primitive | Scene methods | Presentation |
| --- | --- | --- |
| Circle | `circle`, `try_circle`, and layer variants | fill, stroke, shadow |
| Rectangle | `rect`, `try_rect`, and layer variants | rounded corners, fill, stroke, shadow |
| Line | `line`, `try_line`, styled, and layer variants | bounded caps, dashes, markers, logical/world width |
| Polyline | `polyline`, `try_polyline`, styled, and layer variants | bounded joins, continuous dashes, markers |

Styles include solid, linear-gradient, and radial-gradient fills; strokes; and
logical-screen shadows. Colors are straight linear RGBA internally.
`Color::rgb8` and `Color::rgba8` convert familiar sRGB bytes to linear light.
`Color::rgb` and `Color::rgba` accept values already in linear space. Render
boundaries require every channel in `0.0..=1.0`; animation may overshoot, but
the host must call `Color::clamp` explicitly before inserting that value.
Generated local offsets remain separate from their world anchors through the
GPU camera transform: the anchor and local offset are projected independently
before their logical-screen results are added. A circle radius, rounded corner, or world-unit stroke
width that is meaningful under the active zoom therefore does not disappear
merely because adding it to a large source coordinate would round back to the
same `f32`. Fill, stroke, joins/caps, shadow/spread, and gradient sampling share
that representation. Any positive normal `f32` radial range remains a real
range rather than being treated as zero by an epsilon threshold. Nonzero
subnormal coordinate, length, direction, camera, and transform operands are not
portable across WGSL backends, so every GPU geometry path rejects them even
when one translation would hide the difference. Normalized linear color
channels are not part of that geometric envelope and may include subnormal
values; their contribution is below current normalized-target precision.
Display and offscreen pixel scales are bounded so even a one-texel target has
normal finite logical dimensions, half-viewport camera translations, and clip
coefficients. Transform sources are restricted to normal finite values with
magnitude at most `2^120`, keeping reciprocals inside WGSL's specified
division-accuracy domain. Directed-rounding intervals include legal
association/FMA error. Retained 3D validation rejects any conservative range
that cannot prove stable frustum-plane and edge classification. Likewise,
logical joins near reversal, straight/miter, or extrusion thresholds are
rejected instead of allowing backend-dependent topology.

#### Rich bounded strokes

Legacy `line` and `polyline` methods retain their existing logical-pixel
presentation. Use `StrokeStyle2d` when a diagram needs explicit styling:

```rust
use sim_engine::{
    Color, LogicalPixels, Scene, StrokeCap2d, StrokeDashPattern2d, StrokeJoin2d,
    StrokeMarker2d, StrokeStyle2d, Vec2,
};

let dash = StrokeDashPattern2d::new(&[8.0, 4.0], 2.0, 256)?;
let arrow = StrokeMarker2d::arrow(
    LogicalPixels::new(10.0)?,
    LogicalPixels::new(8.0)?,
);
let style = StrokeStyle2d::logical(LogicalPixels::new(2.0)?, Color::WHITE)
    .with_cap(StrokeCap2d::Butt)
    .with_join(StrokeJoin2d::Miter)
    .with_miter_limit(4.0)?
    .with_dash_pattern(dash)
    .with_end_marker(arrow);

let mut scene = Scene::new(Color::BLACK)?;
scene.try_styled_polyline(
    vec![Vec2::ZERO, Vec2::new(20.0, 0.0), Vec2::new(30.0, 10.0)],
    style,
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Dash/gap lengths and phase use source path-coordinate units: world-coordinate
units for `Scene`, logical coordinate units for `ScreenScene`. A visible dash
continues across polyline vertices and receives the configured join. Width is
independent: `StrokeStyle2d::logical` stays constant under camera zoom, while
`StrokeStyle2d::world` uses a validated `WorldLength` and scales with the
camera. `ScreenScene` rejects `StrokeStyle2d::world`; its styled path widths
must also use `LogicalPixels`. Endpoint markers always use logical pixels so annotations remain
readable. By default (`BaseAtEndpoint`), a marker's base is the path endpoint and its tip extends outward;
the marked body endpoint is forced to a butt boundary. Marker length therefore
cannot invert a short body or collide with a terminal join. Exact or
numerically indistinguishable 180-degree retraces, almost-collinear turns whose
shader branch cannot be selected portably, and repeated adjacent points are
rejected as structured scene errors because no stable interior-disjoint
translucent stroke topology exists for them.
Every consecutive polyline segment must be drawable; `DegenerateGeometry`
identifies a line segment or at least one consecutive polyline segment that is
not.

Every dash pattern carries `max_subsegments` in `1..=1_000_000`. Scene
insertion counts visible pieces before tessellation and returns
`SceneError::StrokeExpansionLimitExceeded` atomically when the ceiling would
be crossed. Dash boundaries that collapse at the path's `f32` coordinate scale
return `UnrepresentableStrokePattern` instead of entering a non-progressing
expansion loop. Miter length is bounded by `1.0..=1000.0`; geometry beyond the
configured limit falls back to bevel presentation. Extreme finite widths whose
derived extrusion would overflow are rejected as `InvalidStroke`.

`StrokeMarker2d::arrow(...).with_anchor(StrokeMarkerAnchor2d::TipAtEndpoint)`
keeps the visible tip at the supplied mathematical endpoint. The default
`BaseAtEndpoint` still extends outward. Exact-tip mode currently accepts only
undashed two-point lines/polylines; longer or dashed paths return
`UnsupportedTipMarkerPath`. Both shaft ends are butt-ended. Marker length is
clamped to one quarter of the projected dominant-axis span, so even two inward
markers on a very short vector leave a non-inverted, disjoint shaft. Widths may
be logical or world units; marker dimensions remain logical pixels. There is
no camera-dependent world-endpoint adjustment in host visual state.
`cargo run --release --example stroke_gallery -- --page 5` shows exact tips
against mathematical endpoint crosshairs, including reversed and very short
dual arrows. Cyan is logical width, orange world width; Space pauses and +/-
changes zoom. The page keeps its illustrative vectors within the viewport.

#### Bounded scenes

Production adapters that translate externally sized visual state should use
`Scene::with_budget`. All seven limits are hard bounds on committed scene state;
zero is valid for a background-only scene. Rejection is structured and leaves
retained commands unchanged.

```rust
use sim_engine::{Color, DrawCommand, Layer, Scene, SceneBudget, ShapeStyle, Vec2};

let budget = SceneBudget::new(
    20_000,
    100_000,
    2_000_000,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    20_000,
);
let mut scene = Scene::with_budget(Color::BLACK, budget)?;
let circle = DrawCommand::circle(
    Vec2::ZERO,
    4.0,
    ShapeStyle::filled(Color::WHITE),
)?;
scene.try_extend_to_layers([(Layer::DEFAULT, circle)])?;
assert_eq!(scene.statistics().accepted_commands(), 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The fourth limit is logical retained command payload; the fifth is the committed
command-vector plus owned-polyline allocation ceiling, including spare `Vec`
capacity kept for reuse. Budgeted insertion grows staging and replacement
vectors transactionally, validates allocator-reported capacity before taking
ownership, and rejects above the fifth limit without changing the scene.
`Scene::allocation_bytes()` exposes the same committed byte definition.

Atomic replacement temporarily coexists with the old store. One-command growth
can therefore require up to twice the committed allocation allowance; batch
growth may momentarily hold the current, old staging, and replacement staging
command buffers, up to three times that allowance (owned polyline buffers are
moved, not cloned). A host with a strict process-wide ceiling should reserve
that headroom, scale the fifth limit accordingly, or use incremental insertion
during a controlled construction phase.

Use `try_extend_to_layers` for large alternating/mixed-layer batches. It
validates the transaction before replacing the ordered command store, then
performs one `O(N log N)` sort using unique insertion order as the stable
tie-breaker. The core-only `scene_construction_benchmark` measures this path.
The compatibility constructor `Scene::new` remains explicitly unbounded.

For already sorted streams, ordinary insertion now checks the final ordering
key and appends directly. Validation, depth/layer/insertion ordering and budget
accounting are unchanged; out-of-order streams retain the existing ordered
insertion path. The paired `paired_scene_append_benchmark` ignored test
compares both algorithms with equal warmed capacities and allocation counts;
this isolates insertion work, not complete application extraction or rendering.

### 5. Ordering, clipping, and pseudo-depth

2D commands are ordered by `Layer`, then insertion order. Pseudo-depth affects
projection only; it never reorders commands within a layer.

```rust
use sim_engine::{
    Color, LogicalScreenPosition, LogicalScreenVector, ScreenClipRect, Vec2,
};

# fn add_clipped_line(scene: &mut sim_engine::Scene) -> Result<(), Box<dyn std::error::Error>> {
let clip = ScreenClipRect::from_min_size(
    LogicalScreenPosition::new(40.0, 40.0),
    LogicalScreenVector::new(720.0, 420.0),
)?;

scene.with_screen_clip(clip, |scene| {
    scene.line(
        Vec2::new(-1_000.0, 0.0),
        Vec2::new(1_000.0, 0.0),
        2.0,
        Color::WHITE,
    );
})?;
# Ok(())
# }
```

Nested clips are intersected. Each command captures the active clip when it is
inserted, so later state changes cannot alter accepted commands. Clips use
logical pixels and a top-left origin.

`Projection2d` provides lightweight camera-relative tilt/depth presentation.
It is useful for layered 2.5D visuals, but it is not a z-buffer and does not
replace the retained 3D path.

### 6. Coordinates and HiDPI

| Type | Coordinate space |
| --- | --- |
| `Vec2` | caller-defined 2D world/vector value |
| `LogicalScreenPosition` | logical pixels, top-left origin |
| `LogicalScreenVector` | logical-pixel offset or size |
| `LogicalPixels` | positive logical-pixel scalar length |
| `PhysicalScreenPosition` | surface pixels, top-left origin |
| `PhysicalPerLogical` | physical target texels per logical pixel |
| `LogicalViewport` | finite logical viewport dimensions |
| `LogicalViewportRegion` | positioned local viewport inside a logical target |
| `WorldLength` | positive scalar distance in caller-defined 2D or 3D world units |
| `RenderTarget2d` dimensions | physical texture pixels |

Camera zoom is logical pixels per world unit. Stroke widths, screen shadows,
and clips are logical pixels, so monitor scale does not change their intended
visual size. Retained 3D wireframe widths and dash/gap lengths require
`LogicalPixels`; projection near/far distances and orthographic span require
`WorldLength`; `RenderTarget3d::pixels_per_logical` returns
`PhysicalPerLogical`. Their private representations prevent accidental direct
substitution between world, logical, and physical scales.

Convert physical pointer input before picking:

```rust,ignore
let logical_pointer = renderer.physical_to_logical_screen(
    PhysicalScreenPosition::new(pointer_x as f32, pointer_y as f32),
)?;
let world = camera.screen_to_world(
    logical_pointer,
    renderer.logical_viewport()?,
)?;
```

On window changes, update both physical size and scale:

```rust,ignore
renderer.resize_with_scale_factor(width, height, window.scale_factor())?;
```

Zero physical dimensions are ignored because a minimized surface cannot be
configured at zero size. Display and target scales are bounded so every
non-empty `u32` target keeps both logical dimensions and reciprocal
screen-to-clip coefficients in the normal finite `f32` range; more extreme
ratios are rejected at construction or resize.

#### Fixed UI and viewports

`ScreenScene` accepts typed `LogicalScreenPosition`, `LogicalScreenVector`, and
`LogicalPixels` geometry. Its top-left origin and downward y axis match UI hit
coordinates. `WgpuRenderer::render_screen_scene` derives the fixed camera
internally, while `PreparedScreenScene` prevents a world camera from being
supplied to prepared UI geometry. The internal Y-flipped ordinary `Scene`
commands are deliberately not exposed; use `command_count`, `statistics`, and
`allocation_bytes` for introspection until a typed screen-command view is
introduced. Rectangle sizes must have positive X and Y components, so `min`
always remains the declared top-left. Linear/radial gradient coordinates in a
`ShapeStyle` are interpreted in the same top-left/downward-Y space and converted
exactly once at insertion; the `ScreenScene::linear_gradient` and
`ScreenScene::radial_gradient` helpers make that context explicit. Use
`try_square_rect` for an exact zero-radius UI rectangle; `try_rect` takes a
strictly positive typed corner radius.

```rust,ignore
let mut ui = ScreenScene::with_budget(Color::BLACK, ui_budget)?;
ui.try_rect(
    LogicalScreenPosition::new(16.0, 16.0),
    LogicalScreenVector::new(240.0, 80.0),
    LogicalPixels::new(8.0)?,
    ShapeStyle::filled(Color::rgb8(24, 31, 44)),
)?;
renderer.render_screen_scene(&ui)?;
```

`LogicalViewportRegion` positions a camera-local logical viewport on the
surface. `render_scene_in_viewport` always clips the scene to that region;
scene-local clips are relative to the region and intersect at physical scissor
conversion.

For offscreen rendering, `render_scene_to_target` requires an explicit
`PhysicalPerLogical`, optional local viewport, and `RenderTargetLoad`. Target
dimensions remain physical texels and never inherit window DPI. The scene
background is not an implicit target clear; pass
`RenderTargetLoad::Clear(scene.background())` when that is intended.

#### One composed frame

`FrameComposer` is the preferred surface path when UI, scientific viewports,
prepared geometry, particles, scalar fields, or offscreen targets must share a
defined order. It validates the complete frame before surface acquisition and
uses one clear, encoder, submission, and present.

```rust,ignore
let mut frame = renderer.begin_frame(Color::BLACK, FrameBudget::default())?;
frame.draw_prepared_screen_scene(
    &static_ui,
    FramePassOptions::new(0),
)?;
frame.draw_scene(
    &physics,
    physics_camera,
    FramePassOptions::new(10).with_viewport(canvas_region),
)?;
frame.draw_particle_field(
    &mut particles,
    physics_camera,
    FramePassOptions::new(20).with_viewport(canvas_region),
)?;
frame.draw_glyph_run(
    &font_atlas,
    &inspector_labels,
    ImageSampling::Linear,
    FramePassOptions::new(30),
)?;
let report = frame.present()?;
```

Items with equal `order` retain insertion order. Scene-local clips remain local
to that item's viewport and intersect the optional item clip. Prepared and
dynamic resources are rejected immediately when they belong to another
renderer generation. A retained 3D result participates through
`RenderTarget3d::color_target()`.

`FrameBudget` limits passes, referenced commands, vertices, uniform/streaming
uploads, referenced nominal texel-storage bytes, and conservative draw calls.
Texture byte counters use tightly packed format payload and deliberately exclude
opaque backend alignment, tiling, allocation pages, and metadata; they are a
stable engine budget metric, not a driver-memory oracle. Additions are
atomic, and actual post-tessellation work is checked again before GPU upload or
surface acquisition. `FrameStatistics` separates streaming vertices/uploads
from retained vertices reused without a per-frame geometry upload.
`FrameReport` also exposes actual command-encoder, render-pass, queue-submission,
and surface-present counts: each is one for `Drawn` and zero for a skipped
surface frame.

Images and glyph runs use the same ordering rule and clip/viewport
intersection as geometry. Referencing the same atlas several times counts its
nominal texel payload once toward the frame texture budget. Atlas pixels and
retained instances are not uploaded again merely because the frame draws them.

`FrameCacheBudget` independently bounds idle CPU scratch, uniform buffers,
referenced textures and binding slots. `set_frame_cache_budget` configures it;
`clear_frame_cache` releases cached references. Active frames still obey
`FrameBudget`; work larger than the idle cache can render but is not retained
unboundedly afterwards. `frame_cache_statistics` reports current capacities,
last-frame buffer/bind-group creation, reused bindings, uniform writes and
nominal transient peaks. Zero limits disable this composition storage/binding
retention for paired comparisons; they do not disable source-owned validation
proofs or allocation-free work reuse within a single frame.
The CPU peak is a conservative capacity bound, not a sampled allocator peak:
initial retained bytes plus twice final owned capacities covers transient Vec
growth. Even late rejected streaming geometry returns its batch scratch before
this accounting and idle-budget eviction.
Frame construction still reserves conservative uniform-upload budget before
cache lookup; an eventual cache hit reduces reported actual upload, not the
minimum construction budget required for that item.
Dropping an unpresented composer clears all borrowed sources. Device recovery
invalidates cached bindings; a new generation warms up independently.

Stable ordered camera/image/target slots reuse bindings and buffers. Changed
uniform bytes are queued after prior submissions; unchanged bytes are not
uploaded. On a slot miss, a fixed eight-entry, per-present memo can share an
earlier binding with identical resource identity, sampling and exact uniform
bytes. Shared aliases are not retained in independently mutable slots, so a
later frame can change either draw without overwriting the other's uniform.
`shared_bind_groups()` counts this subset of `reused_bind_groups()`; creation,
upload and retained-uniform counters count actual resources, not aliases.
Persistent slot hits keep their constant-time lookup. This optimization also
applies when idle retention is disabled; scalar-field bindings remain excluded.
Within each render pass the encoder also skips unchanged full-buffer vertex
bindings and scissor rectangles. It keeps every draw and bind-group selection
in order; no state is reused across passes or frames.

This is not a promise of zero backend allocations: queue staging,
encoding, surface acquisition, and currently uncached scalar bindings remain
separate costs. Keep mixed item order intact; regrouping by texture would change
alpha composition.

Changed retained uniforms can use one packed host upload followed by copies
into their existing independent GPU buffers. The default cache permits a
256 KiB transfer buffer. `FrameCacheBudget::new(...)` leaves this optimization
disabled; opt in with `.with_upload_staging_bytes(bytes)`, or pass zero to keep
direct writes while retaining other caches. Batches smaller than 32 changed
slots, larger than the cap/device limit, or unable to reserve optional CPU
scratch fall back to direct writes. No shader layout or draw ordering changes.

Packing trades fewer queue staging allocations for bounded CPU scratch, a GPU
transfer buffer and extra GPU-to-GPU copy work. `upload_staging_bytes()` and
`peak_upload_staging_bytes()` report that buffer separately from uniforms;
payload/copy-record capacities count in cache CPU bytes and idle eviction.
`created_upload_staging_buffers()`, `uniform_upload_calls()`,
`batched_uniform_copies()` and `batched_uniform_bytes()` expose the work.
Actual host-upload bytes are counted once, not again for the internal copies;
construction upload budgets remain unchanged. Scratch is reserved before
surface acquisition, but queued writes and copies occur only after success.
Skipped/rejected frames do not publish pending uniform changes. Clear/recovery
releases transfer storage along with the other cached resources.

For a paired measurement, run `frame_cache_benchmark --frames 120` through
`cargo run --release --example frame_cache_benchmark -- --frames 120`. It
compares disabled/enabled caching on 1/100/1000 independent 32-glyph labels,
unchanged/recolored/moved and mixed-order variants, a 100-circle world pass,
world plus 256 changing HUD rectangles, and 64 images interleaved with 128
rectangles. It asserts source/command counts and warmed uniform-creation
invariants and reports actual presented FPS, build/renderer CPU time, acquire
time, main-thread allocations and cache bytes separately. Additional stage
lines separate preflight/tessellation, uploads/image bindings, camera bindings,
encode/submit/present CPU time, and time outside the renderer report (including
host frame construction and cleanup). To repeat one unchanged workload:

```bash
cargo run --release --example frame_cache_benchmark -- \
  --case mixed --labels 1000 --cache both --trials 3 --frames 120
```

Filtering does not reduce source counts or quality. Every trial starts with
20 unmeasured presents; even-numbered trials reverse case order. Each trial is
reported separately, including slow trials. With no filters, all 30 original
cases remain enabled. `--cache on` focuses on warmed behavior. This is a diagnostic
benchmark, not a GPU-timestamp measurement or a universal 60 FPS promise.

For a same-binary comparison of changing text, keep cache retention enabled and
compare the following command with and without `--upload-staging 0`:

```bash
cargo run --release --example frame_cache_benchmark -- \
  --case recolored --labels 1000 --cache on --trials 3 --frames 120
```

The `uniform_uploads` line distinguishes queue writes, internal copies and
retained transfer-buffer bytes; the existing upload count remains host bytes.

The 0.3 CPU validation path avoids duplicate position proofs and memoizes eight
exact position keys within each call, including the shared world-anchor
projection of circle/rounded-corner vertices. The interval helper uses a single
term-counting pass and exact precomputed rounding coefficients, with the
original formula as its general fallback. It reuses the same interval arithmetic;
inconclusive or overflowing geometry is still rejected. Image/glyph batches
keep one fixed inline, thread-safe proof keyed by the exact six transform
components used by their shader. Unchanged/recolored draws reuse it; moving a
run or changing its target transform revalidates it. Successful geometry
updates invalidate the entry when destination coordinates, sizes or count
change. UV/tint-only instance updates, as well as rejected/identical updates,
preserve the proof; UV/tint validation and changed-instance uploads still run.
Restored resources start empty. This metadata has no heap allocation and is
not included in the capacity-based source/cache allocation counters.

Within one `present`, up to eight streaming scene/uniform pairs can reuse
their already validated tessellation. The keys retain immutable scene borrows
and compare exact uniform bytes; no scene address survives into a later frame.
Each draw still has its own scissor/clip and appears in its original painter
order, but identical streamed geometry shares one uploaded range. Vertex/source
counts count draw references; actual upload-byte counters exclude duplicate
transfers. The per-source `TessellationStats` aggregate remains nominal geometry
work, just as for prepared scenes, not a second actual-upload counter. Frame
construction still requires its conservative per-source budget before this
reuse is known. Distinct scenes,
new transforms and evicted keys take the full validation path. Differential
tests retain the previous validators as behavioral references. These CPU
optimizations do not reduce GPU draw calls or relax any precision guard.

### 7. Cameras and motion

`Camera2d` supports finite pan, zoom, rotation, fit-to-bounds, picking, and
zooming about a logical-screen anchor. Forward and inverse conversion methods
are fallible because valid finite inputs can still overflow during arithmetic
or produce a singular projection. Preserve the last valid camera when an
interactive update fails.

`Tween<T>` is fallible for the same reason:

```rust
use std::time::Duration;
use sim_engine::{Easing, Tween};

let mut zoom = Tween::new(1.0)?
    .to(4.0, Duration::from_millis(300), Easing::EaseInOutCubic)?;
let current = zoom.update(Duration::from_millis(16))?;

# let _ = current;
# Ok::<(), sim_engine::TweenError>(())
```

Built-in `f32` and `Vec2` interpolation protects extreme finite endpoints from
overflow. A custom `Interpolate` implementation must report valid values
truthfully; invalid initial, target, or intermediate state returns
`TweenError` without partially mutating the tween.

### 8. Initialize and configure the renderer

```rust,ignore
use sim_engine::{RendererPresentMode, WgpuRenderer, WgpuRendererOptions};

let options = WgpuRendererOptions::new(
    RendererPresentMode::Vsync,
    window.scale_factor(),
)?;
let mut renderer = WgpuRenderer::new_with_options(
    window.clone(),
    window.inner_size().width.max(1),
    window.inner_size().height.max(1),
    options,
).await?;
let notify_window = window.clone();
renderer.set_pre_present_notify(move || notify_window.pre_present_notify());
```

`RendererPresentMode::Vsync` requests FIFO. `NoVsync` requests the fastest
advertised non-VSync mode and may fall back to FIFO. Query
`renderer.surface_present_mode()` for the concrete Immediate, Mailbox, FIFO,
or FIFO-relaxed configuration chosen by the surface. This does not guarantee
that a desktop compositor will scan out above the monitor refresh rate.
`adapter_name()`, `adapter_backend()`, `adapter_vendor_id()`,
`adapter_device_id()`, `adapter_pci_bus_id()`, `adapter_driver()`, and
`adapter_driver_info()` expose the adapter/API evidence that must travel with
benchmark results. The PCI bus address distinguishes physical instances of the
same GPU model when Vulkan exposes `VK_EXT_pci_bus_info`; an empty address is
not sufficient for release identity. `surface_format()` and
`surface_sample_count()` expose the matching production raster contract.
Recovery updates every value to the replacement logical device.

On Wayland, `Window::pre_present_notify()` installs compositor pacing. Register
it once through `WgpuRenderer::set_pre_present_notify`, as above. The renderer
invokes the callback after command submission and immediately before surface
present only when the concrete mode is FIFO, FIFO-relaxed, or Mailbox.
Immediate presentation deliberately omits the callback and remains suitable
for explicit uncapped diagnostics. The callback survives logical-device
recovery; call `clear_pre_present_notify` if the host window is replaced.

`renderer.wait_for_gpu_idle()` is for bounded benchmarks and controlled
readback. Waiting every interactive frame destroys CPU/GPU overlap.

### 9. Render reports and errors

```rust,ignore
match renderer.render_with_metrics(&scene, &camera) {
    Ok(report) => {
        let status = report.status();
        let metrics = report.metrics();
        let renderer_total = metrics.total_cpu();
        let surface_wait = metrics.surface_acquire();
        let renderer_work = renderer_total.saturating_sub(surface_wait);
        let dropped = metrics.tessellation_stats().dropped_command_count();
        record(status, renderer_work, surface_wait, dropped);
    }
    Err(error) => handle_render_error(error),
}
```

`RenderStatus::Skipped` is not a drawn frame. A timeout, occlusion, outdated
surface, or zero-sized surface can be transient, but a throughput fixture must
reject that attempt or exclude it from both its numerator and timing samples.
The release matrix fails on any skipped warmup or measured report.
Gated workloads run three independent 120-frame measurement trials. Wall
throughput must meet its floor in every trial; the median remains a reported
diagnostic. Renderer-work/acquire percentiles use all 360 drawn samples.
`RendererFrameMetrics`
contains CPU wall-clock durations for tessellation, upload, camera-uniform
upload, surface acquisition, encode/submit/present dispatch, and total call
time. The historical `tessellation()` name is the common pre-upload visual
preparation stage: in addition to streamed triangle generation it includes
camera/source validation and particle visibility selection in particle,
layered-visualization, and heterogeneous composer paths. A skipped surface
report retains CPU preparation that already completed. It performs no queue
writes, camera-uniform upload, encoding, submission, or presentation, and its
actual upload-byte counters are zero. The ordinary streaming path can still
show a non-zero `upload()` duration when pre-acquire transient GPU-buffer
capacity creation already occurred; that duration does not claim a queue
upload.
`total_cpu()` includes `surface_acquire()`; subtract it when comparing
engine-side CPU work and report acquisition separately because FIFO/compositor
back-pressure normally appears there. These metrics do not measure display
scanout; the bounded matrix additionally uses completed wall throughput as its
GPU/back-pressure regression signal.

`TessellationStats` distinguishes accepted commands, rendered commands, and
dropped commands, and groups each by circle/rectangle/line/polyline category.
`SceneStatistics` exposes the same primitive grouping for requested, accepted,
and rejected construction work. Required visuals should use fallible scene
APIs and hosts should surface a non-zero dropped count in diagnostics.

For heterogeneous frames, `FrameStatistics::source_counts` separates
streaming scenes, prepared scenes, dynamic meshes, particles, scalar fields,
images, glyph runs, and targets. `retained_cpu_bytes` and
`retained_buffer_bytes` count unique referenced allocations, while
`texture_bytes` counts unique nominal tightly packed texel payloads: drawing
one prepared scene in several viewports does not multiply either metric. Work
counts still count every draw. This makes the prepared/static
invariant directly observable: after warm-up, unchanged prepared scenes,
images, atlases, and glyph runs do not re-enter streaming geometry counters.
Optional text shaping/rasterization happens before composition. Font bytes,
shaped-line scratch, UTF-8 copies and `TextAtlas2d` cache metadata are not frame
source allocations; inspect `FontFace::allocation_bytes`,
`TextAtlas2d::recovery_memory_bytes` and `TextRun2d::recovery_memory_bytes`
separately. The latter two already include their low-level glyph resources,
so adding them to frame retained totals would double-count those allocations.
Dynamic meshes remain caller-uploaded resources, while particle fields are
culled, compacted, and uploaded on every drawn frame even when their retained
source instances did not change; their bytes correctly remain streaming.

Error-handling rules:

- `RendererMismatch` indicates a host lifecycle bug.
- Capacity errors require a smaller workload or a different device budget.
- Invalid transforms require corrected visual state; retrying unchanged data
  is not useful.
- Transient skipped surface states can be retried on a later event-loop turn.
- Device/surface loss requires explicit recovery and resource restoration.

### 10. Select a rendering path

| Workload | API | Expected update cost |
| --- | --- | --- |
| Small changing scene | `Scene` + `render_with_metrics` | validate, tessellate, upload every frame |
| Static shapes, moving camera | `PreparedScene` | prepare once, update camera per frame |
| Frequently changing triangles | `DynamicMesh2d` | upload ready triangle data |
| Images or atlas sprites | `Image2d` + `ImageBatch2d` | retain pixels and instances; update dirty regions |
| Host-shaped text | `GlyphAtlas2d` + `GlyphRun2d` | retain atlas/run; one quad per glyph |
| Many circles or points | `ParticleField2d` | cull, budget, compact, instanced draw |
| Dense scalar grid | `ScalarFieldTexture` | texture update and heatmap shader |
| Field plus particle overlay | `render_layered_visualization` | one encoder and one queue submission |
| Stereometry solids | `Mesh3d` + `Scene3d` | retained topology and per-object transforms |

#### Prepared scenes

```rust,ignore
let prepared = renderer.prepare_scene(&scene)?;
let report = renderer.render_prepared_with_metrics(&prepared, &camera)?;
```

Preparation captures geometry, background, clips, draw batches, and a CPU
recovery snapshot. Any geometry, style, order, or clip change requires a new
prepared scene. A zero-allocation command-source preflight rejects inputs
outside the portable GPU envelope before tessellation can reserve
caller-sized staging; derived vertices receive a second portability check
before retained GPU allocation or upload. A prepared resource belongs to the
renderer that created it.

#### Dynamic triangle geometry

`DynamicVertex2d` stores world position, pseudo-depth, and linear color.
`create_dynamic_mesh` accepts a triangle list whose vertex count is divisible
by three. Full updates replace the list; triangle-aligned range updates reuse
the existing allocation. Update reports expose upload time and reallocation.

Use this path for ready visual triangles, not as a new simulation data model.
At draw time, every potentially visible dynamic triangle must be fully inside
the full render-target clip volume and retain one normal projected signed-area
direction across legal shader association/FMA choices. A triangle wholly
outside one common clip plane remains a deterministic raster no-op. Crossing a
full-target hardware clip plane or having an ambiguous projected orientation returns
`RendererFrameError::InvalidGeometryTransform`; split such geometry at a
host-controlled boundary before submission. An item's positioned viewport and
logical clip use a scissor rectangle after projection, so they may safely trim
an otherwise portable triangle.

Externally sized triangle input should use an explicit `DynamicMeshBudget`:

```rust,ignore
let budget = DynamicMeshBudget::new(
    30_000,          // triangle-list vertices
    2 * 1024 * 1024, // exact CPU recovery bytes
    2 * 1024 * 1024, // bytes in one full upload
)?;
let mesh = renderer.create_dynamic_mesh_with_budget(&vertices, budget)?;
frame.draw_dynamic_mesh(
    &mesh,
    camera,
    FramePassOptions::new(15).with_viewport(canvas_region),
)?;
```

This is the bounded raw-filled-triangle path for host-tessellated vector art
and scientific diagrams. Vertex counts must be divisible by three. Sim;Engine
does not infer a polygon fill rule because the host supplies the final triangle
list. Full updates allocate and validate a replacement CPU snapshot and any
required larger GPU buffer before committing retained state. Triangle-aligned
range updates remain in-place and enforce the retained upload budget.

#### Retained images and atlas sprites

`Image2d` owns one straight-alpha, top-to-bottom, row-major sRGB RGBA8 image.
RGB is decoded to linear light by the GPU texture; alpha remains linear.
`ImageBudget` constrains source dimensions and bytes before texture creation.
The owned constructor consumes a `Vec<u8>` without first making another
full-image copy; the slice constructor performs one fallible recovery copy.

```rust,ignore
let image_budget = ImageBudget::new(1024, 1024, 4 * 1024 * 1024)?;
let atlas = renderer.create_image_rgba8(
    atlas_width,
    atlas_height,
    rgba_pixels,
    image_budget,
)?;

let sprite = ImageSprite2d::new(
    ImageTexelRect::new(0, 0, 32, 32)?,
    icon_destination,
    Color::WHITE,
)?;
let batch = renderer.create_image_batch(
    &atlas,
    vec![sprite],
    ImageBatchBudget::new(256, 64 * 1024)?,
)?;

frame.draw_image_batch(
    &atlas,
    &batch,
    ImageSampling::Linear,
    FramePassOptions::new(20),
)?;
```

Sprite destinations are local logical pixels with a top-left origin. A frame
viewport positions the entire retained batch without rebuilding its instance
buffer. `draw_image` scales one full image or atlas rectangle into the item
viewport. `draw_world_image` instead maps an atlas rectangle onto an
axis-aligned world `Rect` at one pseudo-depth; the resulting quad follows that
item's camera zoom, rotation, projection, viewport, and clip.

Nearest filtering preserves masks and pixel art. Linear filtering samples only
between the selected sub-rectangle's first and last texel centers, so adjacent
atlas entries do not bleed at their own texel centers. Use padding when a
scaled asset intentionally filters near an outer edge.

`update_image_region` validates the exact row pitch, byte count, checked extent,
and atlas bounds before updating both GPU pixels and the CPU recovery snapshot.
`replace_image_rgba8` is atomic and creates a new resource identity, so batches
against the previous packing must be rebuilt. No implicit atlas eviction
occurs. Texture mutation returns `ImageUploadReport`, whose byte count is RGBA
texture data and whose replacement flag refers only to the texture.
`replace_image_batch` instead returns `ImageBatchUploadReport`; its byte count
and replacement flag refer only to the instance buffer, never to the atlas
texture.

For moving sprites, `update_image_batch(&image, &mut batch, &sprites)` accepts
borrowed descriptions. Unchanged input uploads zero bytes. Stable-capacity
changes reuse CPU description/conversion arrays and the GPU buffer; shrinking
to empty retains capacity. Growth reserves bounded candidates before changing
the drawable batch. `replace_image_batch` uses the same route. Inspect count,
CPU/GPU capacity, current/peak bytes and `replaced_instance_buffer` in its report.
The CPU budget includes both descriptions and retained GPU-instance conversion
storage; hand-sized 0.2 byte limits need adjustment.
Use `ImageBatchBudget::RETAINED_BYTES_PER_SPRITE` and
`GPU_BYTES_PER_SPRITE` when sizing exact capacities instead of hard-coding the
private GPU layout; empty batches still own one minimum GPU buffer slot.

#### Loading TTF/OTF fonts (optional `text` feature)

For application labels and counters, enable the optional font path. It uses
rustybuzz for OpenType shaping and ab_glyph for grayscale outline coverage.
No system font is implicitly selected, and no font is included in the library
binary. The demonstration font and its license live under `examples/assets/fonts`.

```toml
sim-engine = { version = "0.3", features = ["text"] }
```

The default `wgpu` feature alone still accepts host-shaped glyph atlases and
does not enable the font loader or shaping dependencies.

Load a trusted, licensed TrueType or OpenType outline font from bytes:

```rust,no_run
# #[cfg(feature = "text")]
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use sim_engine::{FontBudget, FontFace};

let font = FontFace::from_bytes(
    std::fs::read("assets/MyFont.ttf")?,
    FontBudget::default(),
)?;
// Alternatively, embed your asset: include_bytes!("../assets/MyFont.ttf").to_vec().
// FontFace::clone shares this allocation instead of copying the font.
# let _ = font;
# Ok(())
# }
# #[cfg(not(feature = "text"))]
# fn main() {}
```

File I/O belongs to the host: `std::fs::read` allocates before Engine receives
the bytes. If the path is user-controlled, bound the file read separately.
`FontBudget` bounds the supplied Vec capacity and the font's glyph count;
it does not make arbitrary font files safe to process. Dependency parser and
shaper allocations are not all allocation-fallible, and their internal
scratch/traversal is not an operating-system OOM or hostile-font sandbox.

Create one atlas per font/style/DPI, then prepare the label once:

```rust,ignore
use sim_engine::{
    Color, FrameBudget, FramePassOptions, ImageBatchPlacement, ImageSampling,
    LogicalPixels, LogicalScreenVector, PhysicalPerLogical,
    TextAtlas2d, TextAtlasBudget, TextLayoutBudget, TextStyle,
};

let style = TextStyle::new(
    LogicalPixels::new(24.0)?,             // logical pixels per EM
    PhysicalPerLogical::new(window.scale_factor() as f32)?,
)?;
let mut text_atlas = TextAtlas2d::new(
    &renderer, font.clone(), style, TextAtlasBudget::default(),
)?;
let mut label = text_atlas.prepare(
    &renderer, "Simulation: 60 FPS", TextLayoutBudget::default(),
)?;

// Draw at a baseline, not at the top-left of the glyph bitmap.
let baseline = LogicalScreenVector::new(24.0, 80.0);
let mut frame = renderer.begin_frame(Color::rgb8(12, 14, 18), FrameBudget::default())?;
frame.draw_glyph_run_placed(
    text_atlas.atlas(), label.glyph_run(),
    ImageBatchPlacement::new(baseline, Color::WHITE)?,
    ImageSampling::Linear, FramePassOptions::new(10),
)?;
let report = frame.present()?;

// Update only when content changes. Exact unchanged text skips all text work.
let update = text_atlas.update(
    &renderer, &mut label, "Simulation: 61 FPS", TextLayoutBudget::default(),
)?;
```

`advance()` includes spaces; `ascent()`, signed `descent()` and `line_height()`
are logical-pixel font metrics. `glyph_run().bounds()` instead measures drawable
quads including transparent filter support. Negative bearings are valid. An
empty string or spaces can have no drawable quads, without being an error.
Latin and Cyrillic work when the font covers them. Kerning, ligatures and mark
offsets come from the font's shaping tables. `TextDirection` chooses one
horizontal directional run; automatic direction does **not** implement
mixed-direction paragraph bidi or per-script segmentation.

Use `ImageSampling::Linear`, not nearest-neighbor magnification of a small
bitmap. The cache rasterizes at the configured physical EM size. A two-texel
transparent white border preserves filtered coverage and avoids both dark
straight-alpha fringes and MSAA clipping of the fractional-pixel halo. Logical
placement preserves raster texel scale under the image API's center-to-center
UV convention. Fractional baseline movement filters existing coverage rather
than generating a new hinted/subpixel raster variant on every frame.

Resource and error rules:

- The default atlas is fixed at 1024x1024 RGBA: 4 MiB retained CPU pixels and
  4 MiB GPU texels, plus bounded cache metadata. `TextAtlasBudget::new` selects
  smaller/larger dimensions, cache count and `GlyphRunBudget`. Device limits
  are checked before atlas pixel allocation. There is no implicit growth,
  eviction, repacking, or invalidation of older runs.
- `TextLayoutBudget` bounds input bytes before shaping, output glyphs after
  shaping, and each cache-miss glyph's pixel/outline work before raster
  allocation. Warm glyphs are reused; exact unchanged UTF-8 performs no work
  and does not recheck a newly supplied layout budget.
  Per-run count/byte/device capacity is conservatively preflighted for every
  shaped glyph, including whitespace, before placements or cache misses.
- `AtlasFull`, `GlyphCacheFull`, missing/unsupported glyphs and invalid text
  return structured errors. A failed prepare/update preserves every old run,
  but may leave successfully cached new glyphs from earlier in the operation.
  Cache warming is not an all-or-nothing transaction. Device failure cannot
  roll back queue work already submitted.
- Color, opacity, scrolling and clipping use the existing per-draw placement
  API; they do not change the cached glyphs or require layout updates.
- After font size or DPI changes, construct a replacement atlas and runs at
  the new `TextStyle`, then swap only after success. Font bytes can be shared
  through `FontFace::clone`; both old/new atlases temporarily consume memory.
- After device recovery, call `text_atlas.restore(&renderer)` and then
  `text_atlas.restore_run(&renderer, &mut label)` for every live run. Exact
  retained pixels/layout are restored without shaping or rasterization.

The current slice supports single-face outline fonts (collection index zero).
It does not provide font discovery/fallback, line breaking, paragraph layout,
variable-axis selection, color emoji, text editing, selection or UI navigation.
Applications needing those policies can keep using the low-level API below.

##### Multi-font and Unicode gallery

```bash
cargo run --release --features text --example text_ui_updates -- --page unicode
cargo run --release --features text --example text_ui_updates -- --page fonts
cargo run --release --features text --example text_ui_updates -- --page motion
```

Switch pages with `1`/`2`/`3`; wheel-scroll a shorter gallery window. Fonts are
explicitly chosen per row, not discovered through automatic fallback:

- Sans/serif/monospace comparison: DejaVu Sans, Inter (CFF), DejaVu Serif,
  DejaVu Sans Mono, and DejaVu Math TeX Gyre.
- `𝓣𝔂𝓹𝓮 𝓼𝓸𝓶𝓮𝓽𝓱𝓲𝓷𝓰 𝓽𝓸 𝓼𝓽𝓪𝓻𝓽` consists of supplementary-plane
  mathematical Unicode characters. DejaVu Math covers them; they are not ASCII
  letters with a style flag. The original characters and UTF-8 clusters are
  preserved. This is symbol rendering, not a TeX/math-expression layout engine.
- Japanese hiragana, katakana and selected kanji use a renamed 89 KiB subset
  derived from Noto Sans CJK JP. It includes the demonstrated characters, not
  arbitrary Japanese text. Full application fonts should be loaded with
  `FontFace::from_bytes` and suitable source-byte limits.
- NFC/decomposed accents test combining-mark placement; an explicit Arabic
  right-to-left row tests single-run shaping, not paragraph bidi resolution.

Font sources, exact coverage, licenses and the reproducible Japanese-subset
script are in `examples/assets/fonts`. Python/fontTools are needed only to
regenerate that asset, never to build or run the library. A missing character
still returns `FontError::MissingGlyph`; the gallery does not replace it with
a different glyph or silently pick another font. `--font PATH` affects the
selected-font comparison and motion sample; built-in Unicode rows retain the
font selected for their character repertoire.

The `--acceptance` mode restores old text resources after device replacement,
then requires 40 confirmed `Drawn` frames on **each** of the three pages.
Early window closure is failure in bounded modes. GPU readback additionally
checks Latin, Cyrillic, mathematical script/fraktur and Japanese glyph pixels
at fractional DPI/baselines, with linear/nearest sampling, alpha tint and MSAA.

#### Host-shaped glyph runs

This lower-level path starts below font selection and shaping. The host
chooses fonts, fallback, localization, bidi behavior, baselines, advances, and
line breaks. It gives Sim;Engine opaque `GlyphId` values, atlas rectangles, and
already-positioned logical quads:

```rust,ignore
let entries = vec![
    GlyphAtlasEntry::new(mu_id, ImageTexelRect::new(0, 0, 18, 24)?),
    GlyphAtlasEntry::new(delta_id, ImageTexelRect::new(18, 0, 20, 24)?),
];
let atlas = renderer.create_glyph_atlas(
    width,
    height,
    atlas_pixels,
    entries,
    GlyphAtlasBudget::default(),
)?;
let glyphs = vec![
    PositionedGlyph2d::new(mu_id, mu_destination, Color::WHITE)?,
    PositionedGlyph2d::new(delta_id, delta_destination, Color::WHITE)?,
];
let run = renderer.create_glyph_run(
    &atlas,
    glyphs,
    GlyphRunBudget::default(),
)?;
let logical_bounds = run.bounds().region();
```

One non-empty successful run becomes one instanced draw and one retained quad
per glyph. An empty run is valid and emits no draw call.
Missing glyphs return `GlyphError::MissingGlyph` with identity and run index;
they are never silently replaced. `upload_glyph` can fill a bounded atlas
region and register new sorted metadata. Reusing an identity with a different
rectangle is rejected, as is any rectangle that overlaps texels owned by a
different identity, so old retained runs cannot begin sampling unrelated
pixels. Initial and restored atlases validate overlap in `O(N log N)` bounded
scratch work; one incremental insertion checks the retained entry set before
any texture write.

A run references one atlas. Mixed fallback fonts are explicit multiple runs at
the same frame order; stable insertion order preserves the host's chosen
overlap. `GlyphRunStatistics` exposes submitted glyphs, rendered quads, misses,
and retained bytes. Successful retained runs have zero misses and cause no
texture upload after warm-up.

Use `update_glyph_run(&atlas, &mut run, &positioned_glyphs)` for actual text or
layout changes. The report exposes logical glyph count, capacity, instance
upload/replacement, and aggregate retained/peak CPU bytes. The budget includes
glyph descriptions, sprite descriptions and persistent conversion staging.
Unknown glyphs, foreign/stale atlases, count/byte overflow and synchronous
allocation failure leave the old run drawable with its old bounds. GPU device
loss is not a rollback guarantee for work already submitted.
`GlyphRunBudget::RETAINED_BYTES_PER_GLYPH` exposes the per-capacity CPU cost;
atlas bytes are budgeted separately. Spare capacities and growth overlap are
reported, not inferred from the current glyph count.

For panel motion or hover color, do not rewrite the run:

```rust,ignore
let placement = ImageBatchPlacement::new(
    LogicalScreenVector::new(scroll_x, scroll_y),
    Color::WHITE.with_alpha(0.5),
)?;
frame.draw_glyph_run_placed(
    &atlas, &run, placement, ImageSampling::Linear,
    FramePassOptions::new(20).with_clip(panel_clip),
)?;
```

Content translation is relative to the selected viewport origin. Per-draw tint
multiplies per-glyph tint in straight linear RGBA. Viewport and clip do not move.
Negative bearings and partially or wholly offscreen content are valid; a valid
clip lying outside the target produces an empty effective scissor. Zero-size
`ScreenClipRect` itself remains invalid. The same run may appear multiple times
with independent translation/tint. `draw_image_batch_placed` follows the same
contract. Existing unplaced calls mean zero translation and white tint.

Try `cargo run --release --features text --example text_ui_updates -- --page motion`; Space pauses, R resets,
Esc exits. `--acceptance` runs bounded public update/recovery checks and requires
real `Drawn` presentations; it is not a substitute for the GPU pixel oracle.

### 11. Particles and hard budgets

`ParticleInstance2d` contains world position, logical-pixel radius, color, and
pseudo-depth. `ParticleField2d` retains validated instances and draws selected
particles through one instanced path.

```rust,ignore
use sim_engine::ParticleRenderBudget;

let budget = ParticleRenderBudget::new(
    30_000,          // maximum visible instances
    6 * 1024 * 1024, // maximum retained + visibility-staging CPU bytes
    2 * 1024 * 1024, // maximum GPU instance bytes
    2 * 1024 * 1024, // maximum upload bytes per frame
)?
.with_max_visibility_checks(60_000)?;

let mut particles =
    renderer.create_particle_field_with_budget(&instances, budget)?;
```

When the visibility-check cap is below the retained population, the renderer
samples candidates uniformly rather than pretending that every instance was
classified. Inspect `ParticleStatistics`: submitted, visibility-checked,
visible, culled, budget-limited, dropped, and rendered counts are distinct.

The retained-byte ceiling includes both the exact recovery snapshot and the
worst visibility-staging allocation. `ParticleRenderBudget::INSTANCE_BYTES`
lets a host derive a byte limit from its chosen source and staging counts.
Memory observability is explicit through `cpu_allocation_bytes`,
`gpu_allocation_bytes`, and `recovery_memory_bytes`; successful bounded fields
always report `cpu_allocation_bytes() <= budget.max_retained_bytes()`. Use those
limits so rendering leaves capacity for the simulation itself.

The ceiling is a committed steady-state bound. Full updates and budget changes
are atomic: the old field and a completely validated replacement coexist until
commit, so transient CPU and replacement-buffer peaks are at most the sum of
the old and new field budgets. Hosts with a strict process-wide memory ceiling
must reserve that update headroom (normally twice the steady-state field
allowance), update smaller ranges in place, or recreate during a controlled
loading phase.

### 12. Scalar fields and color maps

`ScalarField` stores a finite row-major grid. Dimensions must be non-zero and
match the value count. Full replacement and rectangular region updates are
validated atomically.

`ScalarField::filled` performs a fallible reservation under a 256 MiB default
value-buffer budget. Use `filled_with_byte_limit` for a different explicit host
budget, or `ScalarField::new` when the host already owns the allocated values.

```rust,ignore
let field = ScalarField::filled(160, 96, 0.0)?;
let mut texture = renderer.create_scalar_field_texture(field)?;

renderer.update_scalar_field_texture_region(
    &mut texture,
    x,
    y,
    width,
    height,
    &changed_values,
)?;
```

`ColorMap::sample` is piecewise-linear on the CPU. The renderer deliberately
quantizes a color map to a 256-entry RGBA8 lookup texture. Closely spaced stops
or HDR distinctions below that resolution cannot be preserved by the GPU
heatmap path. Composed-frame upload and texture statistics mirror the
renderer's one-entry LUT cache after stable pass ordering: adjacent equal maps
reuse one LUT, while `A, B, A` creates and reports three allocations whose
views remain alive through submission.

Use `ScalarFieldSampling::Nearest` for exact texels and `Linear` for manual,
deterministic bilinear scalar sampling. Value-range endpoints and their
subtraction must be finite.

### 13. Targets, composition, and trails

`RenderTarget2d` dimensions are physical texture pixels. A typical multipass
frame renders an expensive layer into a target, then composes it onto the
surface using `BlendMode::Alpha`, `Additive`, or `Replace`.

Public colors are straight linear RGBA. Offscreen target storage is
premultiplied, and the composition shaders and blend states preserve that
contract. Do not manually premultiply a public `Color`.

`TrailBuffer2d` owns two ping-pong targets. Accumulation reads history from one
target, writes retained history and a fresh source into the other, then swaps
only after successful submission. Source/destination aliasing is rejected.
`clear_trail_buffer` clears both textures.

For a scalar field with a particle overlay, prefer
`render_layered_visualization`. It encodes heatmap, budgeted particles, and
surface composition with one command encoder and one queue submission.
`LayeredVisualizationReport` exposes status/timings plus the render passes and
draw calls actually encoded; skipped surface frames report zero for both work
counters.

### 14. Retained 3D stereometry

The retained 3D path is intentionally separate from 2D pseudo-depth:

```rust
use sim_engine::{
    Camera3d, LogicalViewport, Projection3d, Transform3d, Vec3, WorldLength,
};

let projection = Projection3d::perspective(
    std::f32::consts::FRAC_PI_3,
    16.0 / 9.0,
    WorldLength::new(0.1)?,
    WorldLength::new(100.0)?,
)?;
let camera = Camera3d::look_at(
    Vec3::new(0.0, 0.0, 5.0)?,
    Vec3::ZERO,
    Vec3::Y,
    projection,
)?;
let world = Transform3d::IDENTITY.transform_point(Vec3::X)?;
let projected = camera.project_world(
    world,
    LogicalViewport::new(1280.0, 720.0)?,
)?;

# let _ = projected;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Build immutable topology once, retain it on the GPU, and update only object
transforms during animation:

```rust,ignore
let topology = Mesh3d::with_display_edges(vertices, triangles, edges)?;
let retained = renderer.create_mesh3d(topology)?;
let logical_viewport = LogicalViewport::new(logical_width, logical_height)?;
let target = renderer.create_render_target3d(width, height, logical_viewport)?;

let wireframe = WireframeStyle3d::visible(
    Color::WHITE,
    LogicalPixels::new(2.0)?,
)?
.with_hidden(
    Color::rgb(0.45, 0.55, 0.70),
    LogicalPixels::new(1.25)?,
    LogicalPixels::new(7.0)?,
    LogicalPixels::new(5.0)?,
)?;
let style = MeshStyle3d::surface(SurfaceStyle3d::opaque(surface_color)?)
    .with_wireframe(wireframe);
let mut scene = Scene3d::new(Color::BLACK)?;
let object = scene.try_push(&retained, Transform3d::IDENTITY, style)?;

scene.set_transform(object, next_transform)?;
let report = renderer.render_scene3d_to_target(&target, &scene, camera)?;
renderer.compose_render_target(
    target.color_target(),
    BlendMode::Replace,
    1.0,
    Color::BLACK,
)?;
```

Triangle indices define optional surfaces. Explicit `MeshEdge3d` values define
only mathematical display edges, so triangulation diagonals need not appear.
Edge-only meshes are valid. `Object3dId` is a stable opaque handle carrying
private scene provenance. A handle from another `Scene3d` returns
`ObjectNotFound` even when both objects have the same local numeric value.
`Scene3d::set_visible` hides an object without releasing retained topology.

Handles resolve through reusable indexed slots: `instance`, `set_transform`,
`set_style` and `set_visible` perform O(1) object lookup. `remove(id)` returns
the retired instance, preserves all survivor IDs and insertion order, and takes
O(N) to compact dense storage. A reused slot receives a never-repeated ID;
removed and foreign handles remain rejected. Drop the returned instance when
its mesh is no longer needed; backend in-flight references may outlive it.

`Scene3d::with_budget` bounds live objects, CPU bookkeeping and distinct mesh
CPU/GPU allocations. `statistics()` separates reusable slot capacity from
live count and deduplicates shared topology/buffers. External host references
and opaque driver allocations are excluded. `set_mesh(id, &replacement)`
validates ownership, style compatibility and final resource limits before
rebinding only that object. Its report includes old/new mesh overlap. Other
objects sharing the previous mesh retain their previous revision.

`create_mesh3d_with_budget` checks source capacity, conversion staging and GPU
bytes before caller-scale allocation. `replace_mesh3d` atomically assigns a new
immutable revision to one retained handle and reports its upload/peak bytes;
it does **not** modify clones or reuse same-capacity buffers. Scene rebinding is
explicit. CPU source and nominal GPU overlap are bounded by old plus incoming
revision limits, not by an immediate driver-deallocation promise.

Opaque surfaces write `Depth32Float`. Edge classification is conservative:
fragments occluded beyond a two-implementation-depth-unit tolerance receive a
logical-pixel dash pattern, while coplanar and sub-depth-resolution separations
resolve visible and solid. Host insertion order does not decide 3D visibility.

Every `RenderTarget3d` declares both physical texture dimensions and the
logical viewport represented by those texels. Their aspect ratios must match,
and the camera projection aspect must match the target logical aspect. This
keeps edge width stable for native, downsampled, and supersampled targets.

Display-edge segments are homogeneously clipped against all six frustum planes
before shader perspective division and screen-space expansion. A partially
visible edge is shortened; a fully clipped edge emits no fragments without
rejecting the rest of the frame. Before submission, model and camera dot
products are bounded independently of backend association/FMA choices, and the
possible model-dot result interval becomes the input to camera validation. The
surface validator proves a stable, normal projected signed-area direction.
In 0.3, an object containing crossing triangles uses a bounded homogeneous
polygon clipper across all six planes. It carries conservative transform and
intersection intervals and emits canonical clip-space triangle lists only
when topology is provable. Its display edges use the same canonical transform,
so independently associated GPU dot products cannot separate an edge from its
own surface. Fully inside objects keep the original retained indexed path.
Entirely outside triangles remain deterministic no-ops; genuinely ambiguous,
grazing, edge-on or unrepresentable geometry still fails closed.

`validate_scene3d_for_target` and `render_scene3d_to_target_with_budget` use the
same authoritative preflight. `Mesh3dRenderBudget` caps generated vertices,
triangles and combined surface/edge upload bytes. `Mesh3dPreflightReport`
distinguishes generated topology, clipped source triangles and discarded source
triangles. Validation covers the complete visible set and additional work
before target mutation or submission. Object-local errors carry `object_id()`
and their underlying category; camera, target, renderer ownership and aggregate
budget errors are scene-level. Hidden invalid objects are excluded. A successful
preflight is valid for that exact scene/camera/target/budget, not a reusable
permission to submit a subsequently changed scene.
The edge validator then mirrors the remaining shader order through physical-width
expansion, logical-distance and dash-phase calculation, NDC extrusion,
homogeneous scaling, and final clip-coordinate addition. Hidden dash division
is additionally bounded by the complete clipped viewport diagonal so it does
not depend on one CPU association of the transform. Any overflow is
reported as `InvalidGeometryTransform` or `InvalidEdgeProjection`; it is never
submitted as non-finite clip geometry. Transform checks scan every actual
vertex of every visible retained instance; an AABB-only proof is not used
because it can miss interior cancellation and flush-to-zero cases. Frustum
intersection and perspective division propagate conservative intervals.
Edge expansion then uses a deliberately conservative full-axis component bound
for its normalized screen direction, so backend approximation cannot escape
the proven envelope.

Current 3D scope is deliberately focused: opaque surfaces, retained transforms,
hardware depth, solid visible edges, and dashed hidden edges. Translucent or
hatched sections, projected label anchors, text, and 3D picking are not in this
release.

#### Opaque textured surfaces

`Mesh3d::textured(vertices, uvs, triangles, display_edges)` accepts one
`TextureCoordinate2d` per vertex. UVs are finite normalized coordinates in
`[0, 1]`, with `(0, 0)` at the image's top-left and positive V downward. The
core topology API works without `wgpu`; the renderer additionally checks the
portable shader envelope. Duplicate vertices at texture seams so each face
can have independent UVs. Existing constructors still create untextured
meshes without a UV GPU buffer; display edges do not sample a texture.

```rust,ignore
let texture = renderer.create_texture3d_rgba8(
    atlas_width, atlas_height, opaque_srgb_rgba8_pixels, ImageBudget::default(),
)?;
let material = TextureMaterial3d::new(
    &texture, ImageSampling::Nearest, Color::WHITE,
)?;
let topology = Mesh3d::textured(vertices, uvs, triangles, display_edges)?;
let geometry = renderer.create_mesh3d(topology)?;
let textured = renderer.with_mesh3d_material(&geometry, &material)?;
let object = scene.try_push(
    &textured,
    Transform3d::IDENTITY,
    MeshStyle3d::surface(SurfaceStyle3d::opaque(Color::WHITE)?),
)?;
```

The source is row-major, top-to-bottom, straight sRGB RGBA8, with alpha **255
in every texel**. Nonopaque pixels fail with their source texel index before
texture creation. Sampling decodes RGB to linear light, then multiplies it by
both the material tint and the instance's surface color. Both tints must be
normalized and opaque. Surfaces keep the existing depth-writing, no-culling
semantics: winding is not silently reinterpreted for game meshes.

Textures have one mip level, clamp-to-edge addressing and nearest or linear
filtering. `texture.region_coordinates(region)` returns corners in TL, TR,
BR, BL order, inset to the atlas cell's outer texel centers. Use those UVs
instead of the cell's outer boundaries; nearest sampling then stays inside
the selected cell. Mipmap generation, repeat addressing and automatic atlas
padding are not provided. High-quality minification or a different sampling
policy requires host-owned asset preparation, not undocumented engine padding.

`Texture3d` clones share immutable CPU pixels, GPU texels and sampling
bindings. `with_mesh3d_material` creates a new mesh handle sharing the original
topology buffers; it does not mutate existing clones. To change a single
object's material, create that handle and call `scene.set_mesh(id, &handle)`.
Both the mesh and texture must belong to the current renderer generation.
Texture dimensions, pixel bytes and CPU retention are bounded by `ImageBudget`;
scene accounting counts shared textures separately from shared mesh buffers.
`Scene3dBudget::with_texture_limits(cpu_bytes, gpu_bytes)` sets these separate
scene ceilings (each defaults to 64 MiB; zero disallows textured resources).
`Scene3dStatistics` reports `texture_count`, `texture_cpu_bytes` and
`texture_gpu_bytes`, including references from hidden objects. The rebind
report exposes `peak_texture_cpu_bytes` and `peak_texture_gpu_bytes` for
old/new overlap. These are library-retained capacity and nominal texel bytes,
not driver allocation-page measurements.

UVs travel through the same bounded homogeneous clipping path as positions,
then use perspective-correct GPU interpolation. Generated clip-space vertices
contain four position and two UV floats (24 bytes); generated-work limits use
that actual stride, including for untextured crossing surfaces. This does not
change the 12-byte retained untextured position buffer.

### 15. Recovery

`recover_device_and_surface().await` replaces the device, queue, pipelines,
transient buffers, and renderer identity while reusing the surface. External
resources from the previous identity are rejected until restored.

| Resource | Restore method | Result |
| --- | --- | --- |
| `PreparedScene` | `restore_prepared_scene` | exact retained geometry |
| `PreparedScreenScene` | `restore_prepared_screen_scene` | exact retained fixed-screen geometry |
| `DynamicMesh2d` | `restore_dynamic_mesh` | exact retained vertices/capacity |
| `Image2d` | `restore_image` | exact sRGB RGBA pixels and limits |
| `ImageBatch2d` | `restore_image_batch` | exact sprite instances against the restored image |
| `GlyphAtlas2d` | `restore_glyph_atlas` | exact pixels, IDs, rectangles, and limits |
| `GlyphRun2d` | `restore_glyph_run` | exact positioned glyphs against the restored atlas |
| `ParticleField2d` | `restore_particle_field` | exact instances and budget |
| `ScalarFieldTexture` | `restore_scalar_field_texture` | exact scalar grid |
| `RenderTarget2d` | `restore_render_target` | empty target; redraw required |
| `TrailBuffer2d` | `restore_trail_buffer` | empty history; redraw required |
| `Texture3d` | `restore_texture3d` | exact opaque sRGB pixels and original limits; new device identity |
| `RetainedMesh3d` | `restore_mesh3d` | exact topology, UVs, display edges and optional material |
| `Scene3d` | `restore_scene3d` | stable IDs plus exact transform/style/visibility/material; shared meshes and textures restored once |
| `RenderTarget3d` | `restore_render_target3d` | empty color/depth; redraw required |

For a retained 3D scene, restore the scene and target after device recovery:

```rust,ignore
renderer.recover_device_and_surface().await?;
let scene_report = renderer.restore_scene3d(&mut scene)?;
target = renderer.restore_render_target3d(&target)?;

// Previously stored Object3dId values remain valid in `scene`.
scene.set_visible(selected_object, true)?;
```

`restore_scene3d` is atomic at the scene boundary. It recreates every distinct
stale mesh once and commits replacements only after all uploads succeed, while
preserving object IDs, order, transforms, styles, visibility, and next-ID state.
Shared texture restoration is deduplicated independently of mesh geometry, so
different mesh/material variants using one atlas keep one restored atlas.
External old handles remain stale: obtain current mesh/material handles from
the restored scene or explicitly restore them before reuse.

CPU-retained resources can recreate their exact content. GPU-only targets and
history cannot reconstruct prior pixels. Recovery is exceptional; it is not a
normal quality or adapter switch. Previous logical devices enter a bounded
quarantine because immediate teardown crashes some native Linux drivers. The
default limit is four and can be configured from one through eight. Once full,
recovery returns `RecoveryLimitReached` before creating another device; inspect
`quarantined_device_count` and `remaining_device_recoveries` in diagnostics.

### 16. Performance guidance

- Prepare static scenes once.
- Use dynamic meshes for ready changing triangles.
- Batch unchanged icons and glyphs in retained atlas instance buffers.
- Use particle instancing for large point/circle populations.
- Put dense scalar fields in textures and update dirty regions only.
- Render expensive fields below surface resolution when quality permits.
- Stagger visualization cadence independently of simulation and presentation.
- Fuse common field/particle passes.
- Set hard particle memory, upload, visibility-check, and draw budgets.
- Benchmark release builds without VSync and record the concrete present mode.
- Report surface acquisition separately from renderer CPU time.
- Compare primitive/source counts and retained/upload bytes before comparing
  milliseconds, so workload drift is not mistaken for a renderer regression.

A display-limited frame rate is not renderer throughput. Surface acquisition
can wait for FIFO or compositor pacing while renderer CPU work remains small.
The published 0.3.0 metrics do not include GPU timestamp queries. The dev.4
opt-in pass diagnostics are described in the development section above.

From a clean Sim;Engine repository checkout, the named release-mode matrix is:

```bash
./scripts/rendering_benchmark_matrix.sh
```

These release scripts are repository tooling and are intentionally excluded
from the crates.io source archive, which has no Git metadata from which to
produce revision-bound evidence. The release wrapper, standalone matrix, and
standalone HiDPI script fail before
collecting evidence when the worktree is dirty. A common launcher creates a
read-only detached worktree at the captured revision and gives it a separate
Cargo target directory; compilation and execution never consume files from the
mutable calling checkout. The caller's `HEAD` and clean state are checked again
before success is accepted. Replacement refs and legacy grafts are rejected,
and `GIT_NO_REPLACE_OBJECTS=1` applies to checkout, build, and verification.
Child scripts receive only a hidden staging directory. The launcher writes a
completion manifest and atomically renames that directory to
`target/linux-release-evidence/` only after the entire requested gate returns
successfully. A failed or forcibly killed child can leave at most unpublished
staging, never a bundle claiming completion. This binds evidence to Git object
data for the recorded revision instead of relying on periodic checks of a
mutable source tree. A nonblocking `flock` serializes invalidation and
publication, so concurrent wrapper, matrix, or HiDPI invocations fail before
they can disturb an active gate's evidence.

The real compositor fixture requires the Linux executables
`dbus-run-session`, `kwin_wayland`, and `kscreen-doctor`. Their absence is a
hard gate failure, not a skipped test.

The named performance matrix runs undecorated benchmark surfaces inside a
private, single-output virtual KWin session fixed at 1280×720 and scale 1.0.
This prevents window-manager decoration policy, user clicks, occlusion, desktop
monitor migration, or output churn from resizing or closing a release fixture
while retaining the selected physical Vulkan adapter. The separate HiDPI oracle
starts its own controlled KWin session and performs the real compositor
scale/resize transaction there.

Its named surface fixtures are `ui_static_10k`, `ui_90_10`,
`four_viewports`, `image_atlas`, `scientific_text`, `particle_scalar`,
`retained_3d`, and `dpi_reconfigure`. `particle_scalar` exercises the fused
bounded scalar/particle presentation path with 16,384 retained instances, an
8,192-instance visibility/upload cap, and a 256×144 scalar field.
`retained_3d` updates 48 independent transforms, validates and renders opaque
cube surfaces plus visible/hidden mathematical edges into a retained depth
target, composes that target to the surface, and validates 4,096 budgeted
dynamic triangles in the same frame. Its machine-checked compound report merges
the offscreen retained-3D pass with the composed surface frame: two actual
render passes, 146 actual draw calls, unique retained mesh/dynamic CPU and GPU
buffer bytes, and both color-target and depth-target texel bytes.
The same matrix covers mixed-layer construction through
`scene_construction_benchmark`, atomic budget rejection through its exact core
regression, retained-resource recovery through the semantic GPU oracle, and a
transactional real-compositor HiDPI transition through the nested-KWin gate.
The surface fixtures use monotonic layers so their setup remains linear; the
separate construction benchmark owns the adversarial interleaved-layer cost and
verifies the atomic batch path explicitly.
Ordinary incremental insertion also maintains the scene's owned-payload byte
total incrementally; allocation checks do not rescan every retained command.
Cloning a scene recomputes capacity-derived payload and statistics accounting
for the clone's actual polyline allocations.
Prepared scenes and dynamic meshes cache the complete portable-shader geometry
proof for up to eight exact camera/viewport uniforms. This keeps fixed and
multi-viewport retained rendering O(1) with respect to validation after the
first use of a configuration. Every successful dynamic-mesh mutation clears
that cache, so changed vertices are proved again before they can be encoded.
Streaming and prepared fixed-screen scenes retain generated-source and
per-triangle shader proofs. Almost-collinear turns whose join branch is not
numerically stable are rejected when the command enters the scene. The
`ui_90_10` release workload streams square-cornered rectangles while its
retained 90% preserves rounded geometry, keeping its changing-UI cost explicit.
Mixed-layer construction, budget rejection, and recovery are not accepted
`--fixture` values of `rendering_benchmark_suite`; `adapter_probe` and
`hidpi_transition` are accepted non-performance utility fixtures used by the
release tooling. The viewport fixture owns four distinct prepared world scenes
and four distinct cameras. Every surface fixture prints p50/p95/p99 renderer
work excluding acquire, separate acquire percentiles, observed wall
throughput, construction time, physical surface extent, and scale factor.
Composed `FrameReport` fixtures additionally print source counts, vertices,
uploads, unique retained memory, textures, and draw calls. The fused
`particle_scalar` path instead emits and machine-checks its three passes, three
draw calls, scalar/target dimensions, retained CPU/buffer/texture bytes, and
submitted/checked/visible/culled/budget-limited/dropped/rendered particle
counters from `LayeredVisualizationReport`; that specialized result exposes
the passes and draw calls actually encoded by the fused renderer path.
Gated performance fixtures require exactly `1280x720` physical pixels at scale
`1.0`; a compositor resize, minimization, or scale change cannot turn the
release workload into a cheaper raster test. A production-surface probe first
selects the high-performance Vulkan adapter and records its PCI bus address,
format, and MSAA count. The semantic oracle must use that exact physical GPU,
surface format, and sample count; backend, name, vendor/model IDs, and PCI bus
address must then match in every surface and HiDPI process. The semantic oracle
uses the probed production format/MSAA contract. A nested compositor's own
surface format may legitimately differ, so HiDPI pins the physical adapter but
records rather than equates that separate surface contract. Renderer-work p95
is capped at 5 ms for retained UI, four-camera,
image, and glyph workloads, 10 ms for repeated DPI reconfiguration, 12 ms for
the bounded particle/scalar and retained-3D paths, and 25 ms for `ui_90_10`,
which deliberately rebuilds and tessellates one thousand commands per frame.
These fixture-specific ceilings keep a streaming baseline from weakening the
retained-path oracle. If the selected surface mode is
Immediate, the gate additionally requires at least 60 observed FPS. Mailbox
and FIFO must sustain 95% of the confirmed current monitor's reported refresh
rate after clamping that release reference to 30-60 Hz. Before measuring, the
fixture always presents an unmeasured `Drawn` frame so Wayland can associate
the surface with an output. Immediate can proceed without monitor refresh
metadata. Mailbox/FIFO require a positive refresh rate from that current
monitor; zero or missing refresh is unconfirmed, and the fixture never
substitutes the primary or first enumerated monitor.
Measurement itself advances by one frame per event-loop redraw. The fixture
registers a presentation callback with the renderer; FIFO, FIFO-relaxed, and
Mailbox invoke `Window::pre_present_notify` after queue submission and
immediately before surface present, while Immediate deliberately skips it.
This lets synchronized modes schedule the next redraw against the compositor
frame callback instead of an application-only wakeup without accidentally
pacing the uncapped throughput oracle. It also lets
`ScaleFactorChanged`, `Resized`, and current-output changes run between every
warmup or measured sample. Confirmation carries the renderer surface-generation
number and output identity; any mismatch discards all partial samples and
requires a new unmeasured `Drawn` before restarting warmup. Metadata retry is
bounded to 120 confirmation presents, with a final follow-up redraw that can
accept metadata produced by the 120th present. After the final measured
present, a separate finalization redraw repeats the surface-generation and
output snapshot checks before computing or publishing a verdict.
Every warmup and measured frame must report `Drawn`. Surface-acquire
percentiles remain visible diagnostics in every mode, but a scheduler-sensitive
p95 from one trial is not an independent release threshold. Each gated fixture
uses three 120-frame trials, requires every trial to clear the wall-throughput
floor, reports the median for diagnosis, and combines all 360 frames for work
percentiles. The matrix includes an explicit FIFO surface run so the
refresh-normalized branch is exercised. `mixed_layers`
remains core-only;
`recovery_frame` is the mandatory Vulkan semantic fixture because it restores
every retained source on a second logical device and verifies bytes/pixels.
Record the adapter, driver, backend, and concrete present mode with results.
The checked thresholds are the project's Linux release floor, not a claim that
raw timings transfer between unrelated GPUs. A matrix is invalid if semantic,
performance, and compositor evidence do not name one identical adapter.

The automated `dpi_reconfigure` workload deliberately tests the renderer API.
Its renderer-work sample includes the timed `resize_with_scale_factor` call and
`surface.configure` work before `begin_frame`, so the 10 ms p95 applies to the
reconfiguration as well as the following frame.
The matrix additionally starts a nested KWin compositor and changes its real
output scale from 1.00 to 1.25 through `kscreen-doctor`. It accepts evidence
only when `ScaleFactorChanged`, its following `Resized`, and a drawn frame form
one completed transaction for the exact release revision. The same fixture can
still be inspected manually by moving its window between differently scaled
monitors and pressing Esc after the transition:

```bash
cargo run --release --example rendering_benchmark_suite -- \
  --fixture hidpi_transition
```

It logs `ScaleFactorChanged` and `Resized` order, applies the compositor's
physical size and scale at their event boundaries, and does not count an
unpaired redraw or the initial window-creation events as transition evidence.

### 17. Examples

```bash
# Basic renderer integration
cargo run --release --example demo

# Menu with Fluid, Gas, Wave, and edge-case workloads
# plus a retained host-shaped scientific glyph atlas probe
cargo run --release --example ui_demo -- --uncapped

# Bounded star-remnant gas/particle workload
cargo run --release --example star_remnant_stress -- --benchmark

# CPU-only particle-state baseline
cargo run --release --no-default-features --example particle_cpu_benchmark

# Named surface benchmark (default fixture is ui_90_10)
cargo run --release --example rendering_benchmark_suite -- --fixture ui_90_10

# Interactive pixel-level gallery for caps, joins, alpha, dashes, and markers
cargo run --release --example stroke_gallery -- --uncapped

# Exact-tip scientific vectors, both width modes and short reversed paths
cargo run --release --example stroke_gallery -- --page 5

# Changing labels and independently placed/tinted retained copies
cargo run --release --features text --example text_ui_updates
cargo run --release --features text --example text_ui_updates -- --font /path/to/MyFont.ttf
cargo run --release --features text --example text_ui_updates -- --acceptance

# Paired cache off/on surface workloads (not a universal FPS guarantee)
cargo run --release --example frame_cache_benchmark

# Six textured faces, a bounded 16-object pool, edits and recovery
cargo run --release --example editable_textured_3d
cargo run --release --example editable_textured_3d -- --acceptance

# Complete named performance/contract matrix (repository checkout only)
./scripts/rendering_benchmark_matrix.sh

# Independently rotating cube and octahedron with hidden edges
cargo run --release --example stereometry_3d -- --uncapped

# Animated cylinder volume and surface-area derivation
cargo run --release --example cylinder_derivation_3d -- --uncapped --benchmark
```

The interactive 3D, gallery, and stress examples print their controls at
startup. `rendering_benchmark_suite --help` lists its fixture and gate flags;
the minimal `demo` and menu-driven `ui_demo` do not print a universal help
banner.

In `editable_textured_3d`, use N for nearest/linear filtering, E for a mesh
revision, R for removal/reinsertion, A/D to orbit, W/S to move the camera,
H for mathematical edges, Space to pause and F5 for device recovery. The atlas
and fixed 16-object pool belong to the example, not to the library. Its
`--acceptance` path requires actual presents, clipping, changed target size and
one recovery with preserved live IDs. It is not a replacement for pixel readback.

This is not unrestricted free-camera rendering: numerically ambiguous clipped
slivers/corners or mathematical-edge projections can still return an attributed
error, even with finite inputs. The interactive example logs the error and
keeps its last valid image so the camera can be corrected. Disable H when
inspecting close solid-only views; the bounded automatic fixture tests these
two presentation modes separately. No precision guard is disabled for a demo.

## Part II: Engineering Reference

### 18. Architectural role

The public boundary is validated visual state:

```text
host domain/UI state
    -> bounded visual snapshot
    -> Scene / particles / scalar fields / meshes
    -> WgpuRenderer and GPU passes
    -> Linux presentation surface
    -> status, metrics, and diagnostics back to the host
```

The host decides what a particle, field, cube, or section means. Sim;Engine
decides how ready visual data is validated, transformed, clipped, uploaded,
composed, drawn, measured, and recovered.

### 19. Module map

| Module | Responsibility |
| --- | --- |
| `math.rs` | checked 2D vectors and rectangles |
| `color.rs` | linear color, sRGB byte conversion, palette |
| `easing.rs`, `tween.rs` | fallible visual interpolation |
| `camera.rs` | 2D camera, pseudo-depth, typed screen spaces |
| `scene.rs` | validated ordered 2D command stream and styles |
| `field.rs` | finite scalar grid and CPU color-map contracts |
| `particle.rs` | renderer-independent particle visual state |
| `pseudo3d.rs` | checked 3D math, transforms, and CPU projection |
| `mesh3d.rs` | retained topology and edge/style contracts |
| `renderer/config.rs` | surface mode, DPI, renderer options, recovery setup |
| `renderer/tessellation.rs` | 2D scene command to triangle conversion |
| `renderer/visualization.rs` | fused scientific visualization path |
| `renderer/mesh3d.rs` | retained mesh resources and depth/edge passes |
| `renderer/mesh3d_objects.rs` | indexed object IDs, lifetime and shared-resource accounting |
| `renderer/mesh3d_upload.rs` | bounded immutable mesh upload/replacement |
| `renderer/mesh3d_surface.rs` | interval-proven bounded surface clipping and preflight |
| `renderer/mesh3d_texture.rs` | opaque texture/material ownership and recovery |
| `renderer/frame/cache.rs` | bounded frame scratch, uniform and binding reuse |
| `renderer/frame/encoding.rs` | ordered mixed-source draw encoding and pass-local state reuse |
| `renderer/frame/uniform_uploads.rs` | bounded packed transfers into independent retained uniforms |
| `renderer/primitive.wgsl` | 2D, particle, heatmap, and composition shaders |
| `renderer/mesh3d.wgsl` | 3D projection and screen-space edge expansion |

The entire renderer module is behind the `wgpu` feature. CPU-side contracts
remain testable without it.

### 20. Coordinate and projection model

The 2D transform is conceptually:

```text
world position + pseudo-depth
    -> camera-relative projection
    -> logical screen pixels
    -> physical scissor/texture pixels where required
    -> normalized device coordinates
```

The vertex shader receives compact camera rows mapping relative world X/Y and
scalar depth into logical screen X/Y. Generated circle vertices carry a world
anchor and local offset separately; it projects the small offset independently
instead of adding it to a much larger relative coordinate first. Particle
radius is likewise converted directly to a clip-space offset after projecting
the center. CPU validation uses a conservative envelope
for directed f32 rounding, reassociation, and FMA effects through the final
screen-to-clip operation. The accepted GPU numeric domain excludes subnormal
sources and magnitudes above `2^120`; values outside it return a structured
transform error even when one CPU evaluation is finite. Particle projection,
quad expansion, viewport classification, and culling use the same envelope.
Base and local-offset bounds remain independent. Aggregate bounds provide the
first conservative proof, and exact source scans then cover per-vertex FTZ,
triangle orientation, visibility, direction, and stroke-branch hazards that
extrema cannot represent. Failure at either layer returns a structured
transform error before queue mutation.
Lines retain logical-pixel width by
carrying a screen-extrusion direction separately from world position.

The retained 3D transform is explicit:

```text
model point
    -> Transform3d
    -> Camera3d view basis
    -> Projection3d clip coordinates
    -> normalized depth and logical-screen position
```

The 3D convention is right-handed with positive Y up. Cameras look along local
negative Z; exposed view depth is positive distance forward. Large finite
vector operations use wider intermediates before checked `f32` output. GPU
model/view/projection dot products are validated for every visible retained
mesh vertex with a directed-rounding error envelope. This deliberate per-frame
scan catches interior cancellation and flush-to-zero behavior that an AABB
alone cannot prove. Frustum clipping, homogeneous division, and edge extrusion
propagate conservative intervals and enforce the normal denominator domain
required for portable WGSL division. Values which cannot be proven inside
those bounds are rejected before staging growth, upload, or submission.

### 21. Color and alpha model

- Public `Color` is straight linear RGBA.
- Byte constructors decode sRGB to linear light.
- Render-bound colors require every channel in `0.0..=1.0`.
- Scene vertices remain linear through tessellation.
- Surface formats apply their configured output conversion.
- Offscreen target storage is premultiplied alpha.
- Composition shaders and blend states preserve that premultiplied storage.
- `Color::clamp` is explicit sanitization, not implicit scene validation.
- GPU color maps are deliberately quantized to 256 RGBA8 samples.

### 22. Scene and tessellation invariants

`Scene` owns a validated background, an ordered command list, temporary clip
and pseudo-depth state, and monotonically increasing insertion order. Accepted
commands capture immutable primitive/style data plus layer, insertion order,
depth, and optional clip.

Insertion checks source finiteness, drawability, styles, derived bounds,
gradient arithmetic, and operations that can overflow despite finite inputs.
Fallible methods preserve rejection reasons. Sorting is stable by layer and
never depends on pseudo-depth.

The general scene path tessellates on the CPU:

- circles become triangle fans;
- rectangles become fans or rounded sectors;
- lines become screen-extruded strips with the configured butt, square, or
  round endpoint caps;
- connected polyline segments share configured joins; each visible dashed run
  receives its configured caps, while only actual path endpoints are eligible
  for arrow markers;
- gradients are sampled at generated vertices;
- shadows generate separate offset geometry.

If late tessellation still cannot emit an accepted optional command, that loss
is visible in `TessellationStats` rather than hidden behind a successful frame.

### 23. Renderer lifecycle and ownership

`WgpuRenderer` owns one surface, adapter, logical device, queue, surface
configuration, pipelines, uniforms, transient scene buffers, optional MSAA
target, and cached lookup resources.

Renderer-owned external resources store the identity of the logical device
that created them. Every operation checks that identity before touching GPU
handles. Cross-renderer use therefore becomes a structured error before wgpu
validation. `Object3dId` similarly carries private scene provenance.

Capacity-bearing APIs validate integer arithmetic, host byte budgets, and active
device buffer or texture limits before allocation. Mutating fallible methods
validate first and replace retained state only after all checks pass.

### 24. GPU path internals

#### Streaming scenes

The renderer clears and reuses its previous transient CPU allocation,
tessellates commands directly into that upload payload, validates
camera/geometry arithmetic, and grows the GPU vertex buffer when required. It
then acquires the surface; only a successful acquisition uploads vertices and
camera data before clipped batches are encoded, submitted, and presented.
Circle and rounded-corner unit samples are immutable process-wide lookup data;
per-command positions still receive the exact documented segment counts but do
not recalculate the same trigonometric samples every frame. Circle and rounded
corner samples, plus world-width stroke extrusion, are uploaded as local
offsets from retained anchors, preserving their behavior below an anchor's
`f32` ULP. Correlated bounds validate the actual anchor/offset pairs; validation
also covers relative-coordinate intermediates and the final screen-to-clip
multiply/add for ordinary, dynamic, particle, retained image-batch, and glyph
geometry. The 80-byte tessellated vertex size, scene estimates, upload budgets,
and reported upload bytes all include this relative-world component.

#### Prepared scenes

Preparation creates an immutable dedicated GPU buffer and retains a CPU vertex
snapshot for recovery. Rendering reuses geometry and only updates camera state.

#### Dynamic meshes

The resource retains a CPU copy and capacity-managed triangle buffer. Full
updates grow amortized capacity; aligned range updates reuse it. Bounded meshes
preflight vertex/retained/upload work, use fallible staging reservation, and
commit a full update only after replacement resources exist. Geometry extents
are recomputed after mutation.

#### Images and glyph runs

Images use an `Rgba8UnormSrgb` sampled texture and exact retained source bytes.
Single logical or world images draw one shader-generated quad. Atlas and glyph
batches store logical destination, UV-center bounds, and straight-linear tint
in one retained instance buffer and issue one instanced draw when non-empty.
Empty batches and runs are valid and issue no draw. Glyph metadata is kept
sorted for deterministic allocation-free lookup while runs are built.

#### Particle fields

All validated instances stay on the CPU. Rendering tests circle/viewport
intersection, uniformly samples candidates when visibility checks are capped,
and compacts the selected list. A non-empty selection performs one instance
upload and one instanced draw; an empty or fully culled field performs neither.
GPU allocation and upload are proportional to the budget rather than the full
host population.

#### Scalar fields

The renderer stores an `R32Float` texture plus a retained CPU grid. Full and
rectangular updates are validated before queue writes. Heatmap shaders map a
finite value range through a cached lookup texture.

#### Retained 3D

Immutable vertex/index/edge topology is mirrored into GPU buffers and retained
on the CPU for recovery. Scene objects store scene-provenance IDs and
independent model transforms. Opaque surfaces populate color and depth. Display
edges are homogeneously clipped, conservatively classified against depth,
rendered dashed when hidden beyond the coplanar tolerance, and rendered solid
when visible. Edge expansion uses the target's logical-to-physical ratio rather
than the window DPI.

Mesh upload preflights vertex, index, and edge counts, checked byte sizes,
draw-count representation, and the active device's `max_buffer_size` before
allocating conversion staging memory. Staging vectors use fallible reservation
and report `HostAllocationFailed`. GPU buffer creation still follows wgpu's
device-error model and is not presented as a catchable system-OOM boundary.

### 25. Validation philosophy

The nearest public boundary rejects:

- NaN and infinity;
- prohibited zero or negative dimensions, radii, zoom, and scale;
- derived overflow from otherwise finite inputs;
- invalid gradient or value-range arithmetic;
- non-normalized render-bound colors;
- filled scalar allocations beyond the active host byte budget;
- allocation beyond active device limits;
- resource/renderer identity mismatch;
- update regions outside retained state;
- invalid opacity and composition aliasing;
- unsafe camera, near-plane, perspective-divide, or edge-style arithmetic.

This is a behavioral contract: an error should identify invalid required
visual state before it silently becomes different pixels.

### 26. Async and threading

The renderer is designed to be owned by the host render/event-loop thread.
Initialization and recovery are async because adapter and device requests are
async. Ordinary resource mutations and drawing are synchronous submissions.
Each standalone retained-resource mutation starts its queued transfer with an
empty submission but does not wait for GPU completion; this bounds native
`wgpu` staging lifetime even while a window is occluded. Surface rendering
finishes all fallible CPU preparation first, acquires the surface, and only then
enqueues per-frame writes. `Timeout`, `Occluded`, and `Outdated` therefore
return `Skipped` with no deferred queue uploads.

The engine does not create a simulation scheduler or background render thread.
Async simulation is useful only when snapshot and synchronization cost is
smaller than the work it overlaps. Making individual scene mutations async
would add coordination overhead without improving GPU submission.

### 27. API stability policy

Sim;Engine is pre-1.0. Minor versions may contain intentional source-breaking
changes while real consumers exercise the contracts. Such changes must be
listed in `CHANGELOG.md` with migration guidance.

Patch releases preserve these behavioral contracts:

- invalid or overflowing required input is rejected structurally;
- coordinate spaces are not silently interchanged;
- pseudo-depth does not reorder a 2D layer;
- public colors are straight linear RGBA and target alpha is premultiplied;
- GPU resources reject foreign renderer identities;
- recovery behavior and retained snapshots match their documentation;
- failed updates preserve previous drawable resources; bounded text cache
  warming may retain new glyphs without changing an old run;
- the crate accepts visual state and does not own application domain rules.

Public fields are avoided so validation, units, transforms, and future
representation changes remain evolvable. New public paths need a real consumer,
boundary validation, core and GPU tests where applicable, memory accounting,
recovery behavior, and a valid no-default-features build.

The project will not claim 1.0 until its supported platform/backend matrix,
recovery behavior, performance budgets, public documentation, and release
automation are repeatable.

### 28. Known boundaries

- Linux with Vulkan is the only release-gated platform/backend contract.
- A Vulkan adapter is supported only when the mandatory semantic
  fixture passes on that concrete adapter/driver. CI records Mesa software
  evidence and the release evidence records the exact tested adapter; untested
  AMD/NVIDIA drivers are not silently certified by those results.
- Software-Vulkan CI uses packaged lavapipe from Ubuntu 26.04 and requires Mesa
  25.3 or newer. Mesa 25.2.8 llvmpipe was reproduced leaking half-covered MSAA
  samples outside integer scissor bounds. The upstream scissor-plane fix is
  included in [Mesa 25.3](https://docs.mesa3d.org/relnotes/25.3.0.html). The
  oracle retains strict outside-clip expectations and tests all four edges at
  single-sample/production MSAA and scale 1/1.25. This software-driver CI floor
  neither changes host OS requirements nor certifies untested driver versions.
- The crate is pre-1.0.
- Optional `text` loads trusted TTF/OTF outlines and shapes horizontal single-run
  text. Font fallback, paragraph bidi, line breaking, color emoji and automatic
  atlas eviction are not implemented. The low-level API remains available for
  host-shaped glyph runs and externally managed font policies.
- Published 0.3.0 retained 3D supports opaque surfaces and depth-classified edges.
  The development candidate adds the documented Mask/Blend, lighting and texture
  lifecycle paths. General section materials, projected anchors and 3D picking
  remain outside this release scope.
- Published 0.3.0 renderer timing is CPU-side. Dev.4 offers opt-in, bounded GPU
  pass timestamps; it does not measure scanout or total GPU utilisation.
- Independent multi-window recovery is not yet proven.
- Render-target and trail pixels cannot be reconstructed after device loss.

### 29. Verification and contribution gate

From a clean Sim;Engine repository checkout, the complete local Linux release
gate is:

```bash
./scripts/linux_release_gate.sh
```

It checks formatting, Rust 1.90 compatibility, all targets with and without
default features, strict clippy, doctests, warning-free rustdoc, a mandatory
Vulkan semantic GPU readback fixture with backend assertion, the Vulkan-pinned
performance matrix, a real nested-KWin HiDPI transition, `git diff --check`,
and the offline package boundary. A successful local wrapper atomically
publishes `target/linux-release-evidence/`, containing `completion.txt`,
`linux-vulkan-surface.txt`, `linux-vulkan-adapter.txt`, and
`linux-vulkan-performance.txt`, and `linux-hidpi-transition.txt`. The
performance manifest records the exact SHA plus every fixture's adapter,
surface, present mode, trial FPS, CPU/acquire percentiles, deterministic work
counters, threshold, and passed verdict. CI's narrower semantic job writes and
publishes `target/linux-vulkan-adapter.txt` and the installed driver/loader
package versions in `target/linux-vulkan-packages.txt` as the
`linux-vulkan-adapter` artifact. It explicitly selects the packaged lavapipe
ICD and asserts llvmpipe plus MSAA x4. The manifest names the exact VCS SHA, backend,
adapter type, vendor/model IDs, PCI bus address when available, driver, oracle
format, and sample count for the semantic run. CI supplies `github.sha` and
asserts that the artifact does not contain `vcs_sha=unknown`.
The bundled HiDPI manifest records the exact VCS revision, Vulkan backend,
scale, physical size, and transactional event counts.

The wrapper and each standalone evidence-producing surface/HiDPI script must
run from a clean worktree and finish on the exact revision captured at start.
Their shared nonblocking `flock` prevents concurrent gates from deleting,
nesting, or relabeling one another's evidence, and Linux `mv -T` is the sole
publication boundary.
Hardware performance or recovery
claims must additionally name the Linux adapter, PCI vendor/device and bus
address, backend, driver, surface format/sample count, workload, present mode,
confirmed current-monitor refresh where applicable, drawn/attempted counts,
and measurement method.

### 30. Official release procedure

The commands below target the 0.3.0 release. For a later release, substitute its
version and finish its dated changelog and migration notes first. Do not run
publication or create a tag while review or the exact-commit release gate is
pending.

Publishing a crates.io version is permanent: the same version cannot be
overwritten or deleted. A broken version can be yanked, but its archive remains
available to existing lockfiles. For that reason, publish only the exact clean
commit that passed the complete Linux release gate, and never use
`--allow-dirty` or `--no-verify` for an official release.

#### Prerequisites

- The version in `Cargo.toml`, the release heading in `CHANGELOG.md`, and the
  versioned links in `README.md` must agree.
- `HEAD` must be the intended public commit and equal `origin/master`.
- The maintainer must own `sim-engine` on crates.io, have a crates.io API token,
  and be able to push tags and create releases in the GitHub repository.
- The worktree must be clean. The release gate enforces this again.

Authenticate with the crates.io token through Cargo's credential provider. Do
not use a GitHub token here and do not place the token in a command-line
argument or repository file:

```bash
cargo login
```

#### 1. Prove the exact release commit

```bash
git fetch origin
test -z "$(git status --porcelain --untracked-files=all)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)"
test -z "$(git tag -l v0.3.0)"
test -z "$(git ls-remote --tags origin refs/tags/v0.3.0)"
./scripts/linux_release_gate.sh
```

After the wrapper succeeds, bind the evidence to the current revision and
create and inspect the package boundary one last time:

```bash
release_sha=$(git rev-parse HEAD)
grep -Fxq "vcs_sha=$release_sha" \
  target/linux-release-evidence/completion.txt
grep -Fxq 'status=passed' target/linux-release-evidence/completion.txt
cargo package --list
cargo package --locked
crate=target/package/sim-engine-0.3.0.crate
tar -xOf "$crate" sim-engine-0.3.0/.cargo_vcs_info.json \
  | grep -Fq "\"sha1\": \"$release_sha\""
! tar -xOf "$crate" sim-engine-0.3.0/.cargo_vcs_info.json \
  | grep -Fq '"dirty": true'
cargo publish --dry-run --locked
```

`cargo package` explicitly refreshes the local archive; do not assume a publish
dry run replaced a pre-existing `.crate` file. The provenance checks then prove
that `.cargo_vcs_info.json` names the release commit and is not marked dirty.
The publish dry run verifies the upload path without uploading. Review the
archive and file list if anything changed since the gate. Do not continue if
`HEAD`, the worktree, or evidence changes.

#### 2. Publish the immutable crate

```bash
cargo publish --locked
```

Cargo may time out while waiting for the new version to appear in the registry
index even after a successful upload. Before retrying, check the crates.io
package page or run `cargo info sim-engine@0.3.0`; retrying an accepted version
cannot overwrite it.

#### 3. Tag the published commit

Create the tag only after crates.io confirms the package. This avoids a public
release tag for an upload that never succeeded, while the packaged
`.cargo_vcs_info.json` still binds the uploaded source to the clean commit:

```bash
test "$(git rev-parse HEAD)" = "$release_sha"
git tag -a v0.3.0 -m "Sim;Engine v0.3.0"
test "$(git rev-list -n 1 v0.3.0)" = "$release_sha"
git push origin v0.3.0
```

Tags for published versions are immutable release history. Never move or
force-push one. If `v0.3.0` already exists, stop and verify its target instead
of replacing it.

#### 4. Create the GitHub Release and verify public artifacts

Prepare release notes from the `0.3.0` changelog section, then either use the
GitHub web interface or the GitHub CLI:

```bash
gh release create v0.3.0 \
  --verify-tag \
  --title "Sim;Engine v0.3.0" \
  --notes-file /tmp/sim-engine-v0.3.0-notes.md
```

Finally verify all four public identities:

- `https://crates.io/crates/sim-engine/0.3.0` shows version 0.3.0;
- `https://docs.rs/sim-engine/0.3.0` completes successfully;
- Git tag `v0.3.0` points to `$release_sha`;
- the GitHub Release names the same tag and is not marked as a prerelease.

If a serious defect is discovered after publishing, do not attempt to delete
or overwrite 0.3.0. Yank it and prepare a corrected patch release:

```bash
cargo yank --version 0.3.0 sim-engine
```

The authoritative external references are Cargo's
[publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html),
[`cargo publish` reference](https://doc.rust-lang.org/cargo/commands/cargo-publish.html),
and GitHub's
[release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
