# API stability policy

Sim;Engine is currently pre-1.0. The renderer contracts are being exercised by
real consumers, so minor releases may still contain intentional breaking API
changes. Such changes must be listed in `CHANGELOG.md` with a migration note.

## Stable contracts during 0.x

Even before 1.0, patches must preserve these behavioral contracts:

- invalid or overflowing input is rejected structurally at the nearest public
  boundary, or is reported through explicit tessellation diagnostics;
- world, logical-screen, physical-screen, and physical-texture coordinates are
  not silently interchanged;
- depth changes projection and does not reorder commands inside a layer;
- colors are linear RGBA internally, byte colors enter through sRGB conversion,
  and offscreen alpha storage is premultiplied;
- renderer-owned resources reject use with a different renderer and retain the
  documented CPU recovery snapshot where applicable;
- fallible motion and upload APIs reject invalid intermediate arithmetic and
  device-limit overflow without partially mutating their retained state;
- the core crate owns visual state, never physics or application domain rules.

## Public API changes

New public types require a real example or consumer, boundary validation, docs,
and regression coverage. Public fields are avoided so validation, coordinate
markers, transforms, and future representation changes do not require parallel
APIs. Feature-gated APIs must also compile with `--no-default-features`.

Deprecation is preferred when a compatible migration is possible. Before 1.0,
an unsound or misleading contract may be removed directly when keeping it would
freeze invalid behavior; the changelog must say why and show the replacement.

## 1.0 threshold

The project will not claim 1.0 until the supported platform/backend matrix,
device and surface recovery behavior, performance budgets, public documentation,
and release automation are repeatable. After 1.0, Semantic Versioning governs
the public API and behavioral contracts above.
