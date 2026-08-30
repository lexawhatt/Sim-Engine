#!/usr/bin/env bash
set -euo pipefail

echo "[1/11] clean release revision"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "release gate requires a clean worktree" >&2
    exit 1
fi

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
release_sha=$(git rev-parse HEAD)
WGPU_BACKEND=vulkan \
SIM_ENGINE_REQUIRE_GPU_TESTS=1 \
SIM_ENGINE_REQUIRE_VULKAN=1 \
SIM_ENGINE_GPU_EVIDENCE_PATH=target/linux-vulkan-adapter.txt \
SIM_ENGINE_RELEASE_SHA="$release_sha" \
cargo test --all-features \
  renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract \
  -- --nocapture
grep -Fxq 'backend=Vulkan' target/linux-vulkan-adapter.txt
grep -Fxq "vcs_sha=$release_sha" target/linux-vulkan-adapter.txt
echo "GPU evidence: target/linux-vulkan-adapter.txt"

echo "[9/11] named rendering performance matrix"
WGPU_BACKEND=vulkan \
SIM_ENGINE_REQUIRE_VULKAN=1 \
./scripts/rendering_benchmark_matrix.sh

echo "[10/11] package boundary"
git diff --check
if cargo package --offline --list \
    | grep -Eq '^(\.workbench/|\.idea/|\.github/|target/)'; then
    echo "private or generated files leaked into the package" >&2
    exit 1
fi

echo "[11/11] package verification"
cargo package --offline

echo "Linux release gate passed"
