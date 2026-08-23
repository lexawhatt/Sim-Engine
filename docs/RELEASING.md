# Release checklist

Run release checks from a clean worktree on the release commit.

## Required local gate

```bash
cargo fmt --all -- --check
cargo test --all-targets --all-features
cargo test --no-default-features
cargo clippy --all-targets --all-features -- -D warnings
git diff --check
```

At least one release machine must have a working adapter and run the semantic
GPU fixture in strict mode:

```bash
SIM_ENGINE_REQUIRE_GPU_TESTS=1 cargo test --all-features \
  renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract
```

Record the OS, adapter, backend, driver, and result in the release notes. A
normal developer run may skip when no adapter exists; a release run may not.
The fixture creates a second logical device, restores retained geometry onto it,
and compares GPU-readback bytes with the CPU recovery snapshot.

On a machine with a window system, run the live surface/device recovery fixture:

```bash
cargo run --release --example star_remnant_stress -- \
  --benchmark --recovery-smoke
```

It must complete two recovery cycles and print `recovery smoke passed`. Record
the adapter/backend/driver because native swapchain teardown behavior is
driver-specific.

## Platform matrix

Before publishing, run the required gate on Windows, Linux, and macOS. Exercise
at least the native primary backend on each platform. Web targets remain
experimental until a browser fixture is added and must not be advertised as
release-supported solely because they compile.

## Version and artifacts

1. Move relevant `Unreleased` entries in `CHANGELOG.md` into the new version.
2. Confirm `Cargo.toml` version, license files, README example, and public docs.
3. Run `cargo package --allow-dirty` only for inspection; the published package
   must be produced from a clean tagged commit.
4. Inspect the package file list and build the packaged crate with all features
   and without default features.
5. Tag only after the platform results and GPU adapter evidence are recorded.

Performance claims require a named workload, build profile, hardware, backend,
sample count, and separate CPU/GPU scope. The windowless particle fixture is a
CPU validation benchmark and must not be presented as GPU throughput.
