# Pseudo-3D Rendering Specification

## Purpose

Sim;X is the primary product consumer and design driver for Sim;Engine. Most
Sim;X domains are 2D, but Sim;Math stereometry is the documented exception:
solid geometry loses essential information when reduced to a flat scene.

This subsystem renders ready 3D visual geometry. It does not own mathematical
proofs, named solids, construction rules, physical simulation, or educational
state. Sim;Math decides that an object is a cube, octahedron, section, or
labelled point. Sim;Engine owns transforms, projection, visibility, line style,
depth, composition, and GPU resource lifetime.

`Projection2d` remains a lightweight 2D presentation effect. It is not the
foundation of this subsystem and its scalar `depth` must not be reinterpreted
as a general 3D coordinate.

## Public Types

The minimum reusable surface is:

- `Vec3`: finite three-component world vector;
- `Rotation3d`: normalized rotation with axis-angle and Euler constructors;
- `Transform3d`: translation, rotation, and positive finite scale;
- `Camera3d`: validated view transform and orthographic or perspective
  projection;
- `Mesh3d`: retained vertices and triangle topology with optional explicit
  display edges;
- `Scene3d`: object instances referencing retained meshes and per-object model
  transforms;
- `WireframeStyle3d`: visible and hidden edge colors, logical-pixel widths, and
  dash pattern;
- `SurfaceStyle3d`: opaque, translucent, or hatched face presentation;
- projected anchors for composing Sim;Math labels through the 2D overlay.

Types may be introduced in smaller slices, but temporary APIs must not collapse
model, view, and projection transforms into one ambiguous matrix.

## Coordinate Spaces And Units

- 3D world and model coordinates are right-handed, with positive Y up.
- Cameras look along their local negative Z axis.
- Positions and mesh sizes are caller-defined world units.
- Edge widths and dash/gap lengths are logical screen pixels so textbook
  diagrams stay readable while zooming.
- Render targets use physical pixels only at the renderer boundary.
- Labels and interaction use existing logical-screen position types.

Every conversion between model, world, view, clip, logical-screen, and physical
texture space must be explicit in naming and documentation.

## Invariants

- Public constructors reject NaN, infinity, zero-length rotation axes,
  degenerate projection ranges, invalid indices, and arithmetic overflow.
- Rotating one object changes only its model transform. It must not rebuild or
  retessellate retained topology.
- Draw order is not a substitute for 3D visibility. Opaque surfaces and edges
  use a depth attachment.
- Hidden edges are classified by depth against rendered surfaces. Back-face
  adjacency alone is insufficient for an inner solid, intersecting section, or
  one object occluding another.
- Visible edges use a solid logical-pixel pattern. Hidden fragments use the
  configured logical-pixel dash/gap pattern and may be disabled.
- A translucent or hatched section participates in depth testing while keeping
  the solid and its important construction edges readable.
- 2D scene layering remains deterministic and separate from 3D depth. The host
  composes UI and labels over the resolved 3D viewport explicitly.

## Rendering Pipeline

The stereometry path uses retained GPU mesh buffers and per-object transform
updates:

1. Write opaque faces to color and depth, including optionally invisible depth
   faces used only for hidden-line classification.
2. Draw hidden edge fragments with depth comparison behind the surface and a
   screen-space dash pattern.
3. Draw visible edge fragments with normal depth comparison and a solid pattern.
4. Draw translucent or hatched section surfaces with their explicit depth and
   blending contract.
5. Composite projected point markers and labels through the 2D overlay.

The exact pass split may be fused where semantics remain pixel-equivalent.
Independent object rotation should normally update only a small uniform or
instance record.

## Error Handling

Fallible construction and updates return structured errors. A submitted object
must not disappear silently because a finite intermediate overflowed. Renderer
reports distinguish rejected geometry, clipped geometry, and surface/device
failure.

When hardware limits cannot hold a retained mesh, creation returns a capacity
error before calling an invalid GPU operation.

## Interaction

The first consumer fixture must support:

- independent cube and octahedron rotation;
- orbit camera and zoom;
- pause/reset controls;
- stable visible/hidden cube edges during motion;
- a section attached to the octahedron with dynamic hatching;
- projected anchor output suitable for later point labels and hit testing.

## Performance Budget

For textbook solids, rotation must not allocate per frame after warm-up and
must not perform CPU tessellation. The target is comfortably above 100 FPS in
no-VSync release mode on the named Linux development GPU, with renderer CPU
time reported separately from presentation wait.

The 3D path must coexist with the existing high-volume 2D visualization paths;
it is not permission to move Sim;X domain simulation onto the renderer.

## Tests Required

- finite and overflow boundaries for every public math type;
- rotation composition and transform identity tests;
- camera projection, unprojection ray, and viewport conversion tests;
- invalid mesh topology and degenerate triangle tests;
- CPU reference visibility cases for cube and nested octahedron;
- GPU readback for visible solid edges, occluded dashed edges, depth ordering,
  section transparency, and hatching orientation;
- independent-transform regression proving one object's update does not mutate
  another object's retained state;
- Linux live fixture and frame-allocation/performance measurements.

## Delivery Slices

1. Core `Vec3`, rotation, transform, camera, and projection contracts.
2. Retained triangle mesh and real depth attachment.
3. Screen-space solid/hidden edge pipeline.
4. Translucent and hatched section material.
5. Sim;Math cube/octahedron fixture with independent interaction and GPU
   readback.
6. Projected anchors, picking, recovery, and performance evidence.
