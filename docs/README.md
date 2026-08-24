# Sim;Engine Documentation

Sim;Engine is the visual rendering layer designed around Sim;X requirements.
It accepts ready visual state from a host and provides validated scene, camera,
animation, GPU resource, and scientific-visualization facilities. It does not
run domain simulation.

## Start Here

- [Integration Guide](INTEGRATION_GUIDE.md) - dependency setup, API concepts,
  examples, renderer lifecycle, performance paths, recovery, and a public-type
  catalogue.
- [Architecture Reference](ARCHITECTURE_REFERENCE.md) - module boundaries,
  coordinate and color contracts, CPU/GPU data flow, resource ownership,
  composition, performance model, and extension rules.

## Contracts And Operations

- [API Stability Policy](API_STABILITY.md) - guarantees during the pre-1.0
  period and the 1.0 threshold.
- [Linux Release Checklist](RELEASING.md) - required tests, strict GPU evidence,
  package inspection, and release artifacts.
- [Pseudo-3D Specification](PSEUDO_3D.md) - the staged Sim;Math stereometry
  renderer. Core math/camera types exist; retained 3D meshes and the depth-based
  hidden-line renderer are not implemented yet.
- [Changelog](../CHANGELOG.md) - user-visible additions, fixes, and migrations.

## Source-Level API Reference

Every public Rust item has rustdoc. Generate the exact reference for the current
checkout with:

```bash
cargo doc --all-features --no-deps --open
```

Use the guides above for lifecycle and architectural context; use rustdoc for
complete signatures, error variants, units, and per-method invariants.

## Current Release Scope

The first supported release target is Linux. Windows, macOS, and web are future
portability targets and do not currently carry release guarantees.
