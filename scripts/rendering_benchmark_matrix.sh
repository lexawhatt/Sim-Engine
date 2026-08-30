#!/usr/bin/env sh
set -eu

# Run from the repository root. Surface fixtures record absolute timing and
# deterministic work counters; compare results only on the same documented
# adapter/backend/present-mode setup.

export WGPU_BACKEND=vulkan
export SIM_ENGINE_REQUIRE_VULKAN=1

release_sha=$(git rev-parse HEAD)
mkdir -p target
SIM_ENGINE_SURFACE_EVIDENCE_PATH=target/linux-vulkan-surface.txt \
SIM_ENGINE_RELEASE_SHA="$release_sha" \
SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY=0 \
cargo run --release --example rendering_benchmark_suite -- --fixture adapter_probe
grep -Fxq "vcs_sha=$release_sha" target/linux-vulkan-surface.txt
grep -Fxq 'backend=vulkan' target/linux-vulkan-surface.txt

SIM_ENGINE_REQUIRED_ADAPTER_BACKEND=$(sed -n 's/^backend=//p' target/linux-vulkan-surface.txt)
SIM_ENGINE_REQUIRED_ADAPTER_NAME=$(sed -n 's/^name=//p' target/linux-vulkan-surface.txt)
SIM_ENGINE_REQUIRED_ADAPTER_VENDOR=$(sed -n 's/^vendor=//p' target/linux-vulkan-surface.txt)
SIM_ENGINE_REQUIRED_ADAPTER_DEVICE=$(sed -n 's/^device=//p' target/linux-vulkan-surface.txt)
SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID=$(sed -n 's/^pci_bus_id=//p' target/linux-vulkan-surface.txt)
SIM_ENGINE_GPU_SURFACE_FORMAT=$(sed -n 's/^surface_format=//p' target/linux-vulkan-surface.txt)
SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT=$(sed -n 's/^sample_count=//p' target/linux-vulkan-surface.txt)
export SIM_ENGINE_REQUIRED_ADAPTER_BACKEND
export SIM_ENGINE_REQUIRED_ADAPTER_NAME
export SIM_ENGINE_REQUIRED_ADAPTER_VENDOR
export SIM_ENGINE_REQUIRED_ADAPTER_DEVICE
export SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID
export SIM_ENGINE_GPU_SURFACE_FORMAT
export SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT
export SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY=1
export SIM_ENGINE_REQUIRE_PRODUCTION_SURFACE_FORMAT=1

test -n "$SIM_ENGINE_REQUIRED_ADAPTER_BACKEND"
test -n "$SIM_ENGINE_REQUIRED_ADAPTER_NAME"
test -n "$SIM_ENGINE_REQUIRED_ADAPTER_VENDOR"
test -n "$SIM_ENGINE_REQUIRED_ADAPTER_DEVICE"
test -n "$SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID"
test -n "$SIM_ENGINE_GPU_SURFACE_FORMAT"
test -n "$SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT"

SIM_ENGINE_REQUIRE_GPU_TESTS=1 \
SIM_ENGINE_GPU_EVIDENCE_PATH=target/linux-vulkan-adapter.txt \
SIM_ENGINE_RELEASE_SHA="$release_sha" \
cargo test --release --all-features \
  renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract \
  -- --exact
grep -Fxq "vcs_sha=$release_sha" target/linux-vulkan-adapter.txt
grep -Fxq 'backend=Vulkan' target/linux-vulkan-adapter.txt
grep -Fxq "name=$SIM_ENGINE_REQUIRED_ADAPTER_NAME" target/linux-vulkan-adapter.txt
grep -Fxq "vendor=$SIM_ENGINE_REQUIRED_ADAPTER_VENDOR" target/linux-vulkan-adapter.txt
grep -Fxq "device=$SIM_ENGINE_REQUIRED_ADAPTER_DEVICE" target/linux-vulkan-adapter.txt
grep -Fxq "pci_bus_id=$SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID" target/linux-vulkan-adapter.txt
grep -Fxq "oracle_format=$SIM_ENGINE_GPU_SURFACE_FORMAT" target/linux-vulkan-adapter.txt
grep -Fxq "oracle_sample_count=$SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT" target/linux-vulkan-adapter.txt
echo "matrix adapter: $SIM_ENGINE_REQUIRED_ADAPTER_NAME ($SIM_ENGINE_REQUIRED_ADAPTER_VENDOR:$SIM_ENGINE_REQUIRED_ADAPTER_DEVICE at $SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID), format=$SIM_ENGINE_GPU_SURFACE_FORMAT, samples=$SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT"

cargo run --release --example rendering_benchmark_suite -- --fixture ui_static_10k --gate
cargo run --release --example rendering_benchmark_suite -- --fixture ui_static_10k --gate --vsync
cargo run --release --example rendering_benchmark_suite -- --fixture ui_90_10 --gate
cargo run --release --example rendering_benchmark_suite -- --fixture four_viewports --gate
cargo run --release --example rendering_benchmark_suite -- --fixture image_atlas --gate
cargo run --release --example rendering_benchmark_suite -- --fixture scientific_text --gate
cargo run --release --example scene_construction_benchmark -- --commands 10000 --iterations 5
cargo test --release --no-default-features scene::tests::scene_budget_rejection_is_atomic_and_counted -- --exact
cargo run --release --example rendering_benchmark_suite -- --fixture dpi_reconfigure --gate
./scripts/hidpi_transition_gate.sh
