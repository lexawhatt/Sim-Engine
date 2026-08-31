#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
if [[ "${SIM_ENGINE_RELEASE_SNAPSHOT:-0}" != 1 ]]; then
    exec "$project_root/scripts/release_snapshot_gate.sh" \
        scripts/linux_release_gate.sh "$@"
fi

echo "[1/11] clean release revision"
start_sha=${SIM_ENGINE_RELEASE_SHA:?release snapshot did not provide an exact SHA}
output_dir=${SIM_ENGINE_RELEASE_OUTPUT_DIR:?release snapshot did not provide an output directory}
mkdir -p "$output_dir"
assert_provenance() {
    if [[ "$(git rev-parse HEAD)" != "$start_sha" ]]; then
        echo "release gate HEAD changed during the gate" >&2
        exit 1
    fi
    if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
        echo "release gate requires a clean worktree" >&2
        exit 1
    fi
}
assert_provenance

echo "[2/11] formatting"
cargo fmt --all -- --check

echo "[3/11] Rust 1.90 minimum version"
cargo +1.90.0 check --all-targets --no-default-features
cargo +1.90.0 check --all-targets --all-features

echo "[4/11] all Linux targets"
cargo test --all-targets --all-features

echo "[5/11] core-only"
cargo test --all-targets --no-default-features

echo "[6/11] strict clippy"
cargo clippy --all-targets --no-default-features -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings

echo "[7/11] public documentation"
cargo test --doc --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

echo "[8/11] strict Linux Vulkan semantics"
assert_provenance
release_sha=$start_sha
WGPU_BACKEND=vulkan \
SIM_ENGINE_SURFACE_EVIDENCE_PATH="$output_dir/linux-vulkan-surface.txt" \
SIM_ENGINE_RELEASE_SHA="$release_sha" \
SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY=0 \
cargo run --release --example rendering_benchmark_suite -- --fixture adapter_probe
required_adapter_name=$(sed -n 's/^name=//p' "$output_dir/linux-vulkan-surface.txt")
required_adapter_vendor=$(sed -n 's/^vendor=//p' "$output_dir/linux-vulkan-surface.txt")
required_adapter_device=$(sed -n 's/^device=//p' "$output_dir/linux-vulkan-surface.txt")
required_adapter_pci_bus_id=$(sed -n 's/^pci_bus_id=//p' "$output_dir/linux-vulkan-surface.txt")
required_surface_format=$(sed -n 's/^surface_format=//p' "$output_dir/linux-vulkan-surface.txt")
required_surface_sample_count=$(sed -n 's/^sample_count=//p' "$output_dir/linux-vulkan-surface.txt")
test -n "$required_adapter_pci_bus_id"
WGPU_BACKEND=vulkan \
SIM_ENGINE_REQUIRE_GPU_TESTS=1 \
SIM_ENGINE_REQUIRE_VULKAN=1 \
SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY=1 \
SIM_ENGINE_REQUIRED_ADAPTER_BACKEND=vulkan \
SIM_ENGINE_REQUIRED_ADAPTER_NAME="$required_adapter_name" \
SIM_ENGINE_REQUIRED_ADAPTER_VENDOR="$required_adapter_vendor" \
SIM_ENGINE_REQUIRED_ADAPTER_DEVICE="$required_adapter_device" \
SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID="$required_adapter_pci_bus_id" \
SIM_ENGINE_REQUIRE_PRODUCTION_SURFACE_FORMAT=1 \
SIM_ENGINE_GPU_SURFACE_FORMAT="$required_surface_format" \
SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT="$required_surface_sample_count" \
SIM_ENGINE_GPU_EVIDENCE_PATH="$output_dir/linux-vulkan-adapter.txt" \
SIM_ENGINE_RELEASE_SHA="$release_sha" \
cargo test --all-features \
  renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract \
  -- --nocapture
grep -Fxq 'backend=Vulkan' "$output_dir/linux-vulkan-adapter.txt"
grep -Fxq "vcs_sha=$release_sha" "$output_dir/linux-vulkan-adapter.txt"
grep -Fxq "pci_bus_id=$required_adapter_pci_bus_id" "$output_dir/linux-vulkan-adapter.txt"
grep -Fxq "oracle_format=$required_surface_format" "$output_dir/linux-vulkan-adapter.txt"
grep -Fxq "oracle_sample_count=$required_surface_sample_count" "$output_dir/linux-vulkan-adapter.txt"
echo "GPU evidence: $output_dir/linux-vulkan-adapter.txt"

echo "[9/11] named rendering performance matrix"
assert_provenance
WGPU_BACKEND=vulkan \
SIM_ENGINE_REQUIRE_VULKAN=1 \
./scripts/rendering_benchmark_matrix.sh

echo "[10/11] package boundary"
assert_provenance
git diff --check
if cargo package --offline --list \
    | grep -Eq '^(\.workbench/|\.idea/|\.github/|target/)'; then
    echo "private or generated files leaked into the package" >&2
    exit 1
fi

echo "[11/11] package verification"
cargo package --offline
assert_provenance

echo "Linux release gate passed"
