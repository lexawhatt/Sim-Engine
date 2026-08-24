# Release checklist

Run release checks from a clean worktree on the release commit.

## Linux release scope

The first supported release target is Linux. Windows, macOS, web, and other
targets may compile or work, but they are portability work rather than release
gates until they receive their own documented hardware and recovery evidence.
Do not advertise them as supported by the 0.1 release.

## Required Linux gate

The complete local gate is available as `scripts/linux_release_gate.sh`. Its
individual commands are:

```bash
cargo fmt --all -- --check
cargo test --all-targets --all-features
cargo test --no-default-features
cargo clippy --all-targets --all-features -- -D warnings
cargo test --doc --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
SIM_ENGINE_REQUIRE_GPU_TESTS=1 cargo test --all-features \
  renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract \
  -- --nocapture
git diff --check
cargo package --allow-dirty --offline --list
cargo package --allow-dirty --offline
```

At least one release machine must have a working adapter and run the semantic
GPU fixture in strict mode:

```bash
SIM_ENGINE_REQUIRE_GPU_TESTS=1 cargo test --all-features \
  renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract \
  -- --nocapture
```

The strict fixture prints the adapter, backend, driver, vendor, and device. Save
that output in the release evidence. A normal developer run may skip when no
adapter exists; a release run may not. Mesa software rendering is sufficient
for deterministic shader/readback semantics in CI. Performance and live
surface-recovery claims require a named Linux hardware adapter and driver. The
fixture creates a second logical device, restores retained geometry onto it,
and compares GPU-readback bytes with the CPU recovery snapshot.

On a machine with a window system, run the live surface/device recovery fixture:

```bash
cargo run --release --example star_remnant_stress -- \
  --benchmark --recovery-smoke
```

It must complete two recovery cycles and print `recovery smoke passed`. Record
the adapter/backend/driver because native swapchain teardown behavior is
driver-specific.

## Platform evidence

Before publishing, run the required gate on Linux and exercise at least one
native Linux hardware backend for the live surface/device recovery fixture.
The CI Mesa adapter provides an additional deterministic software-backend
contract check. Windows, macOS, and web results are welcome but non-blocking and
must not be presented as supported solely because they compile.

## Version and artifacts

1. Move relevant `Unreleased` entries in `CHANGELOG.md` into the new version.
2. Confirm `Cargo.toml` version and MSRV, license files, README doctest, public
   guides, and strict rustdoc output.
3. Run `cargo package --allow-dirty --offline` only for inspection; the
   published package must be produced from a clean tagged commit.
4. Inspect the package file list. It must contain public source, examples,
   licenses, scripts, and documentation, but no `.workbench`, IDE state, CI
   metadata, or build output.
5. Tag only after the Linux results and GPU adapter evidence are recorded.

Performance claims require a named workload, build profile, hardware, backend,
sample count, and separate CPU/GPU scope. The windowless particle fixture is a
CPU validation benchmark and must not be presented as GPU throughput.
