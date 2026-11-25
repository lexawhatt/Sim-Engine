# Sim;Engine - agent instructions

These are the canonical repository-wide engineering instructions. Keep enduring
rules here; version-specific plans, progress and evidence belong in `.workbench/`.
The old READ_FIRST code-style and WORKFLOW files link here, not to separate rules.

## Start here

- Check the current request, `git status`, and [Cargo.toml](Cargo.toml) before
  editing. Preserve unrelated user changes; do not revive completed historical
  tasks or infer publication from a version string.
- Read [README.md](README.md) for scope and the relevant sections of
  [DOCUMENTATION.md](DOCUMENTATION.md) for API contracts. Record public behavior
  changes and migration notes in [CHANGELOG.md](CHANGELOG.md).
- When the private workbench exists, read its [index](.workbench/README.md) and
  [current roadmap](.workbench/READ_FIRST_SIM_ENGINE_ROADMAP.md), then the active
  plan and original consumer report relevant to the task. Load historical notes
  only as needed, not the entire archive or evidence tree.
- `.workbench/` is Git-ignored and absent from ordinary clones and crate packages.
  Its absence does not block normal development: use these instructions, public
  contracts and the user's request; do not invent missing private requirements.

## Engine and consumer boundaries

Sim;Engine renders already-computed visual state. It owns validated 2D and
retained-3D descriptions, cameras/projections, interpolation, composition, GPU
resources, clipping, recovery and diagnostics. Sim;X is the primary design
driver; Sim;Logic is another consumer. The library remains reusable.

Physics, chemistry, biology, mathematical constructions, domain constants and
entities, simulation stepping, plugins, navigation and application layout belong
to the host. Consumer requests do not authorize moving those systems into Engine
or editing sibling repositories. Keep fixture-specific mathematics, grids and
product palettes in examples/tests, not renderer fallbacks.

Keep CPU visual state separate from GPU implementations. `src/mesh3d/` and
`src/renderer/mesh3d/` serve different layers, as do the text modules. Preserve
the feature boundaries: core without default features, CPU-only `fonts`, optional
`wgpu` (enabled by default), and `text` combining fonts with GPU text rendering.
Read the manifest for the current MSRV and dependencies; do not raise them silently.

## API and implementation rules

- Write identifiers, comments, errors and repository documentation in English.
  Use ASCII hyphens, rustfmt defaults and full descriptive names. Established
  graphics/math terms such as `uv`, `rgba`, `ndc` and `dt` are fine.
- Comments explain intent, invariants and tradeoffs, not obvious mechanics.
  Document public types, functions and relevant fields with `///`, including
  behavior, coordinate spaces, units, ranges, failure and ownership semantics.
- Runtime library code returns structured errors for fallible operations; do
  not add `unwrap()` or `expect()` there. Test assertions and fail-fast demo
  setup are separate from the library's error contract.
- Keep validated representations private and use constructors/builders/setters.
  Preserve typed logical/physical/world boundaries; name transforms explicitly.
  Finite inputs alone do not prove finite intermediate or GPU arithmetic.
- Preserve numeric rejection, clipping, painter/depth order, color/alpha,
  alias isolation, resource provenance and device-replacement contracts. Do not
  silently drop accepted geometry or relax validation to improve a benchmark.
- Budgets must cover actual allocations, staging/overlap and uploads. A rejected
  transactional update must preserve prior valid state. Caches require owned
  lifetime or stable identity, complete invalidation, bounded memory and recovery;
  a reused pointer alone is not a cross-frame identity.
- Keep explicit 2D/3D contracts. Reuse existing paths before adding abstraction;
  do not introduce a generic dimensional framework without a demonstrated need.

## Source and test layout

- Split by responsibility, not arbitrary line counts. Avoid god modules and
  catch-all `utils.rs` or `structs.rs` files.
- A module with child files lives in `name/mod.rs`; a leaf stays `name.rs`.
  Small leaf tests may stay inline under `#[cfg(test)]`; external tests live in
  the owner's `tests/mod.rs`, with contract-named children as needed.
- Do not add `tests.rs`, `*_tests.rs`, `#[path]` workarounds or historical test
  names such as `dev5`, `red` and `red_team`. Do not broaden production visibility
  just to relocate tests. Production imports are explicit; tests may glob-import
  their parent to inspect private contracts.
- Shared GPU test support shares transport/setup mechanics, not expected pixels,
  mathematical oracles, channel interpretation, tolerances or adapter policy.
  Keep frozen benchmark/reference implementations independent.
- Run [the source layout checker](scripts/check_source_layout.sh) after moves.
  Update module declarations and documentation links in the same change.

## Delivery and agent coordination

1. Preserve the incoming report. Separate measured facts, hypotheses and desired
   acceptance; identify the baseline revision, workload and limitations.
2. Implement one coherent slice with explicit API, units, errors, budgets,
   invalidation and recovery. Add independent regressions and a human-visible
   example when raster output changes.
3. When delegation is authorized, the coordinating agent owns integration and
   the checklist. Assign one writer per connected API/layout/shader/validation/
   resource contract, not merely per file. Reserve disjoint write scopes;
   reviewers inspect read-only. A clean Git merge is not semantic validation.
4. Rerun integrated checks after related changes are assembled or ownership
   changes. For release work obtain independent adversarial review, resolve
   findings and repeat the affected gates; do not claim a self-check is external
   review or weaken assertions to get approval.
5. Commit/push coherent verified changes only when authorized. Publication,
   release tags and GitHub releases are separate maintainer actions, not implied
   by a request to prepare a release candidate.

## Verification

Choose checks proportional to the change. Documentation-only work needs link,
command and whitespace checks, not an invented claim that GPU tests were rerun.
For implementation work, start with focused regressions, then cover affected
feature combinations. Before release, run the complete maintained gate.

Common checks from the repository root:

```sh
cargo fmt --all -- --check
bash scripts/check_source_layout.sh
git diff --check
cargo test --all-targets --no-default-features
cargo test --all-targets --no-default-features --features fonts
cargo test --all-targets
cargo test --all-targets --all-features
cargo clippy --all-targets --no-default-features -- -D warnings
cargo clippy --all-targets --no-default-features --features fonts -- -D warnings
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
```

`--all-features` alone does not test example harnesses: retain `--all-targets`.
Use [CI](.github/workflows/ci.yml) and
[the Linux release wrapper](scripts/linux_release_gate.sh) as executable sources
for MSRV, doctest/rustdoc, shell regressions, strict Vulkan, real surface/HiDPI,
performance and package verification. Inspect their prerequisites before running
`./scripts/linux_release_gate.sh`; it qualifies a clean exact-source snapshot,
not an arbitrary dirty worktree. Report unavailable gates as unverified.

Linux/Vulkan is the currently documented release target, not proof of Windows or
macOS support. A skipped GPU fixture is not a semantic pass. VM/software-adapter
results do not qualify a physical GPU or establish hardware frame rate. Never
fix a driver-specific oracle failure by silently changing tolerance or disabling
the required backend.

## Performance and evidence

- Measure paired baseline/candidate workloads without concurrent builds, GPU
  tests, OS/driver updates or compositor changes. Keep workload, geometry,
  quality, clips, colors and rejection guarantees comparable.
- Identify exact source/dirty state, binary, physical adapter (PCI ID where
  available), backend/driver, surface/target format, MSAA, scale and present mode.
  Pin the same physical adapter across semantic and performance qualification;
  inventory or model name alone does not prove which GPU rendered a frame.
- Separate host work, preparation, upload, acquire/present waits and asynchronous
  GPU timestamp intervals. CPU timing is not GPU time. Count only confirmed
  Drawn frames as successful throughput; retain skipped/dropped/error counts.
- Report cold/warm behavior, all trials, sample counts, p50/p95/p99, allocations,
  retained memory, uploads, vertices, draw calls and recovery costs. Preserve
  failed runs and slow trials instead of hiding them in an aggregate.
- A missed target stays open. Do not promise universal FPS, invent completion
  percentages or equate consumer acceptance with exact-source release evidence.
- Keep original logs, manifests, binaries and package archives append-only in
  uniquely named evidence directories. Never edit a recorded result to repair
  a link or change its verdict. `target/` and `/tmp/` are not durable evidence.
  A changed revision or driver needs its own qualification, not a relabeled run.

## Documentation and handoff

Public contracts live in README, CHANGELOG, DOCUMENTATION and Rust API docs.
This `AGENTS.md` belongs in Git but not the published crate; retain the explicit
package allowlist. Never include private workbench files in the crate.

When present, use the [workbench directory index](.workbench/README.md): plans
and session logs in `product/`, consumer reports in `integrations/`, design in
`architecture/`, measurements in `reference/`, audits in `reviews/`, final
handoffs in `releases/`, original artifacts in `evidence/`. Maintain reusable
gates in `scripts/`; archive superseded drafts and private helpers without
destroying provenance. Preserve a hash/path mapping when reorganizing artifacts.
Do not run archived scripts against the current checkout or rewrite historical
reports as current facts.

End each meaningful session by recording what changed, checks actually run,
remaining work and the next concrete step in the active plan/log linked from
the roadmap. Without a private workbench, include that handoff in the response
and update applicable public docs. Keep implemented, verified, consumer-accepted
and published states distinct. Do not modify old release evidence for a new task.
