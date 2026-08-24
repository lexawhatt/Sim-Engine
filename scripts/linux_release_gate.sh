#!/usr/bin/env bash
set -euo pipefail

echo "[1/9] formatting"
cargo fmt --all -- --check

echo "[2/9] all Linux targets"
cargo test --all-targets --all-features

echo "[3/9] core-only"
cargo test --no-default-features

echo "[4/9] strict clippy"
cargo clippy --all-targets --all-features -- -D warnings

echo "[5/9] public documentation"
cargo test --doc --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

echo "[6/9] strict Linux GPU semantics"
SIM_ENGINE_REQUIRE_GPU_TESTS=1 cargo test --all-features \
  renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract \
  -- --nocapture

echo "[7/9] whitespace"
git diff --check

echo "[8/9] package boundary"
if cargo package --allow-dirty --offline --list \
    | grep -Eq '^(\.workbench/|\.idea/|\.github/|target/)'; then
    echo "private or generated files leaked into the package" >&2
    exit 1
fi

echo "[9/9] package verification"
cargo package --allow-dirty --offline

echo "Linux release gate passed"
