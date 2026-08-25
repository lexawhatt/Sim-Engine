#!/usr/bin/env bash
set -euo pipefail

echo "[1/10] clean release revision"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "release gate requires a clean worktree" >&2
    exit 1
fi

echo "[2/10] formatting"
cargo fmt --all -- --check

echo "[3/10] Rust 1.90 minimum version"
cargo +1.90.0 check --all-targets --no-default-features
cargo +1.90.0 check --all-targets --all-features

echo "[4/10] all Linux targets"
cargo test --all-targets --all-features

echo "[5/10] core-only"
cargo test --all-targets --no-default-features

echo "[6/10] strict clippy"
cargo clippy --all-targets --no-default-features -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings

echo "[7/10] public documentation"
cargo test --doc --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

echo "[8/10] strict Linux Vulkan semantics"
WGPU_BACKEND=vulkan \
SIM_ENGINE_REQUIRE_GPU_TESTS=1 \
SIM_ENGINE_REQUIRE_VULKAN=1 \
cargo test --all-features \
  renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract \
  -- --nocapture

echo "[9/10] package boundary"
git diff --check
if cargo package --offline --list \
    | grep -Eq '^(\.workbench/|\.idea/|\.github/|target/)'; then
    echo "private or generated files leaked into the package" >&2
    exit 1
fi

echo "[10/10] package verification"
cargo package --offline

echo "Linux release gate passed"
