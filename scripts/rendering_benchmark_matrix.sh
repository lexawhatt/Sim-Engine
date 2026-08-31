#!/usr/bin/env sh
set -eu

# Run from the repository root. Surface fixtures record absolute timing and
# deterministic work counters; compare results only on the same documented
# adapter/backend/present-mode setup.

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"
if [ "${SIM_ENGINE_RELEASE_SNAPSHOT:-0}" != 1 ]; then
    exec "$project_root/scripts/release_snapshot_gate.sh" \
        scripts/rendering_benchmark_matrix.sh "$@"
fi

start_sha=${SIM_ENGINE_RELEASE_SHA:?release snapshot did not provide an exact SHA}
output_dir=${SIM_ENGINE_RELEASE_OUTPUT_DIR:?release snapshot did not provide an output directory}
assert_provenance() {
    if [ "$(git rev-parse HEAD)" != "$start_sha" ]; then
        echo "rendering benchmark matrix HEAD changed during the gate" >&2
        exit 1
    fi
    if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
        echo "rendering benchmark matrix requires a clean worktree" >&2
        exit 1
    fi
}
assert_provenance

export WGPU_BACKEND=vulkan
export SIM_ENGINE_REQUIRE_VULKAN=1

release_sha=$start_sha
mkdir -p "$output_dir"
surface_evidence=$(mktemp "$output_dir/.linux-vulkan-surface.XXXXXX")
adapter_evidence=$(mktemp "$output_dir/.linux-vulkan-adapter.XXXXXX")
gate_complete=0
cleanup() {
    rm -f -- "$surface_evidence" "$adapter_evidence"
    if [ "$gate_complete" -ne 1 ]; then
        rm -f -- "$output_dir/linux-vulkan-surface.txt" \
            "$output_dir/linux-vulkan-adapter.txt" \
            "$output_dir/linux-hidpi-transition.txt"
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

assert_provenance
SIM_ENGINE_SURFACE_EVIDENCE_PATH="$surface_evidence" \
SIM_ENGINE_RELEASE_SHA="$release_sha" \
SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY=0 \
cargo run --release --example rendering_benchmark_suite -- --fixture adapter_probe
assert_provenance
grep -Fxq "vcs_sha=$release_sha" "$surface_evidence"
grep -Fxq 'backend=vulkan' "$surface_evidence"

SIM_ENGINE_REQUIRED_ADAPTER_BACKEND=$(sed -n 's/^backend=//p' "$surface_evidence")
SIM_ENGINE_REQUIRED_ADAPTER_NAME=$(sed -n 's/^name=//p' "$surface_evidence")
SIM_ENGINE_REQUIRED_ADAPTER_VENDOR=$(sed -n 's/^vendor=//p' "$surface_evidence")
SIM_ENGINE_REQUIRED_ADAPTER_DEVICE=$(sed -n 's/^device=//p' "$surface_evidence")
SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID=$(sed -n 's/^pci_bus_id=//p' "$surface_evidence")
SIM_ENGINE_GPU_SURFACE_FORMAT=$(sed -n 's/^surface_format=//p' "$surface_evidence")
SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT=$(sed -n 's/^sample_count=//p' "$surface_evidence")
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

assert_provenance
SIM_ENGINE_REQUIRE_GPU_TESTS=1 \
SIM_ENGINE_GPU_EVIDENCE_PATH="$adapter_evidence" \
SIM_ENGINE_RELEASE_SHA="$release_sha" \
cargo test --release --all-features \
  renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract \
  -- --exact
assert_provenance
grep -Fxq "vcs_sha=$release_sha" "$adapter_evidence"
grep -Fxq 'backend=Vulkan' "$adapter_evidence"
grep -Fxq "name=$SIM_ENGINE_REQUIRED_ADAPTER_NAME" "$adapter_evidence"
grep -Fxq "vendor=$SIM_ENGINE_REQUIRED_ADAPTER_VENDOR" "$adapter_evidence"
grep -Fxq "device=$SIM_ENGINE_REQUIRED_ADAPTER_DEVICE" "$adapter_evidence"
grep -Fxq "pci_bus_id=$SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID" "$adapter_evidence"
grep -Fxq "oracle_format=$SIM_ENGINE_GPU_SURFACE_FORMAT" "$adapter_evidence"
grep -Fxq "oracle_sample_count=$SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT" "$adapter_evidence"
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

assert_provenance
mv "$surface_evidence" "$output_dir/linux-vulkan-surface.txt"
mv "$adapter_evidence" "$output_dir/linux-vulkan-adapter.txt"
assert_provenance
gate_complete=1
