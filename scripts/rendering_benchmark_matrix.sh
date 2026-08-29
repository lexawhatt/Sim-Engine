#!/usr/bin/env sh
set -eu

# Run from the repository root. Surface fixtures record absolute timing and
# deterministic work counters; compare results only on the same documented
# adapter/backend/present-mode setup.

cargo run --release --example rendering_benchmark_suite -- --fixture ui_static_10k --gate
cargo run --release --example rendering_benchmark_suite -- --fixture ui_90_10 --gate
cargo run --release --example rendering_benchmark_suite -- --fixture four_viewports --gate
cargo run --release --example rendering_benchmark_suite -- --fixture image_atlas --gate
cargo run --release --example rendering_benchmark_suite -- --fixture scientific_text --gate
cargo run --release --example scene_construction_benchmark -- --commands 10000 --iterations 5
cargo test --release --no-default-features scene::tests::scene_budget_rejection_is_atomic_and_counted -- --exact
cargo run --release --example rendering_benchmark_suite -- --fixture dpi_reconfigure --gate
SIM_ENGINE_REQUIRE_GPU_TESTS=1 SIM_ENGINE_REQUIRE_VULKAN=1 cargo test --release --all-features renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract -- --exact
