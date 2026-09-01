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

# Keep the performance oracle isolated from user interaction, window occlusion,
# monitor migration, and desktop output churn. The inner invocation receives a
# single fixed 1280x720@1 output from KWin and still renders through the same
# Vulkan adapter selected by wgpu.
if [ "${SIM_ENGINE_MATRIX_COMPOSITOR:-0}" != 1 ]; then
    for command in dbus-run-session kwin_wayland; do
        if ! command -v "$command" >/dev/null 2>&1; then
            echo "rendering benchmark matrix requires $command" >&2
            exit 1
        fi
    done
    compositor_root=$(mktemp -d "$output_dir/.matrix-compositor.XXXXXX")
    cleanup_compositor() {
        rm -rf -- "$compositor_root"
    }
    trap cleanup_compositor EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    mkdir "$compositor_root/runtime" "$compositor_root/config" \
        "$compositor_root/data" "$compositor_root/cache"
    chmod 700 "$compositor_root/runtime"
    export SIM_ENGINE_MATRIX_SCRIPT="$project_root/scripts/rendering_benchmark_matrix.sh"
    export SIM_ENGINE_MATRIX_COMPOSITOR=1
    dbus-run-session -- env \
        XDG_RUNTIME_DIR="$compositor_root/runtime" \
        XDG_CONFIG_HOME="$compositor_root/config" \
        XDG_DATA_HOME="$compositor_root/data" \
        XDG_CACHE_HOME="$compositor_root/cache" \
        kwin_wayland \
        --virtual \
        --socket sim-engine-matrix \
        --width 1280 \
        --height 720 \
        --scale 1 \
        --output-count 1 \
        --no-lockscreen \
        --no-global-shortcuts \
        --exit-with-session "$project_root/scripts/rendering_benchmark_matrix_controller.sh"
    exit 0
fi

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
performance_evidence=$(mktemp "$output_dir/.linux-vulkan-performance.XXXXXX")
active_fixture_output=""
gate_complete=0
cleanup() {
    rm -f -- "$surface_evidence" "$adapter_evidence" "$performance_evidence"
    if [ -n "$active_fixture_output" ]; then
        rm -f -- "$active_fixture_output"
    fi
    if [ "$gate_complete" -ne 1 ]; then
        rm -f -- "$output_dir/linux-vulkan-surface.txt" \
            "$output_dir/linux-vulkan-adapter.txt" \
            "$output_dir/linux-vulkan-performance.txt" \
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

printf 'format_version=1\nvcs_sha=%s\n' "$release_sha" >"$performance_evidence"
run_gated_fixture() {
    fixture=$1
    shift
    active_fixture_output=$(mktemp "$output_dir/.linux-vulkan-fixture.XXXXXX")
    if cargo run --release --example rendering_benchmark_suite -- \
        --fixture "$fixture" --gate "$@" >"$active_fixture_output" 2>&1; then
        fixture_status=0
    else
        fixture_status=$?
    fi
    cat "$active_fixture_output"
    if [ "$fixture_status" -ne 0 ]; then
        rm -f -- "$active_fixture_output"
        active_fixture_output=""
        return "$fixture_status"
    fi
    grep -q '^fixture=' "$active_fixture_output"
    grep -q '^gate=passed ' "$active_fixture_output"
    grep -E '^(fixture=|passes=|layered\[|retained_3d\[|gate=passed |prepare_cpu_ms=|particle_scalar_contract=|retained_3d_contract=)' \
        "$active_fixture_output" >>"$performance_evidence"
    rm -f -- "$active_fixture_output"
    active_fixture_output=""
}

run_gated_fixture ui_static_10k
run_gated_fixture ui_static_10k --vsync
run_gated_fixture ui_90_10
run_gated_fixture four_viewports
run_gated_fixture image_atlas
run_gated_fixture scientific_text
run_gated_fixture particle_scalar
run_gated_fixture retained_3d
cargo run --release --example scene_construction_benchmark -- --commands 10000 --iterations 5
cargo test --release --no-default-features scene::tests::scene_budget_rejection_is_atomic_and_counted -- --exact
run_gated_fixture dpi_reconfigure
./scripts/hidpi_transition_gate.sh

assert_provenance
test "$(grep -c '^fixture=' "$performance_evidence")" -eq 9
test "$(grep -c '^gate=passed ' "$performance_evidence")" -eq 9
grep -q '^fixture=particle_scalar ' "$performance_evidence"
test "$(grep -c '^layered\[passes=3,draw_calls=3,scalar=256x144,target=640x360,.*particles_submitted=16384,particles_checked=8192,particles_visible=6848,particles_culled=1344,particles_budget_limited=8192,particles_dropped=0,particles_rendered=6848\]$' "$performance_evidence")" -eq 1
test "$(grep -c '^particle_scalar_contract=retained:16384,visibility_cap:8192,field:256x144,target:640x360$' "$performance_evidence")" -eq 1
grep -q '^fixture=retained_3d ' "$performance_evidence"
test "$(grep -c '^passes=2 commands=2 vertices=12294 .*draw_calls=2 sources\[streaming=0,prepared=0,dynamic=1,particles=0,scalars=0,images=0,glyphs=0,targets=1\]$' "$performance_evidence")" -eq 1
test "$(grep -Ec '^retained_3d\[objects=48,triangles=576,edges=576,render_passes=2,draw_calls=146,retained_cpu_bytes=[1-9][0-9]*,retained_buffer_bytes=[1-9][0-9]*,texture_bytes=[1-9][0-9]*\]$' "$performance_evidence")" -eq 1
test "$(grep -c '^retained_3d_contract=objects:48,triangles:576,edges:576,dynamic_triangles:4096$' "$performance_evidence")" -eq 1
mv "$surface_evidence" "$output_dir/linux-vulkan-surface.txt"
mv "$adapter_evidence" "$output_dir/linux-vulkan-adapter.txt"
mv "$performance_evidence" "$output_dir/linux-vulkan-performance.txt"
assert_provenance
gate_complete=1
