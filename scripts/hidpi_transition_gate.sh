#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
if [[ "${SIM_ENGINE_RELEASE_SNAPSHOT:-0}" != 1 ]]; then
    exec "$project_root/scripts/release_snapshot_gate.sh" \
        scripts/hidpi_transition_gate.sh "$@"
fi

start_sha=${SIM_ENGINE_RELEASE_SHA:?release snapshot did not provide an exact SHA}
output_dir=${SIM_ENGINE_RELEASE_OUTPUT_DIR:?release snapshot did not provide an output directory}
mkdir -p "$output_dir"
published_evidence="$output_dir/linux-hidpi-transition.txt"
fixture_root=""
publish_complete=0
cleanup() {
    if [[ -n "$fixture_root" ]]; then
        rm -rf -- "$fixture_root"
    fi
    if [[ "$publish_complete" -ne 1 ]]; then
        rm -f -- "$published_evidence"
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

assert_provenance() {
    if [[ "$(git rev-parse HEAD)" != "$start_sha" ]]; then
        echo "HiDPI transition gate HEAD changed during the gate" >&2
        exit 1
    fi
    if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
        echo "HiDPI transition gate requires a clean worktree" >&2
        exit 1
    fi
}
assert_provenance

for command in dbus-run-session kwin_wayland kscreen-doctor; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "mandatory HiDPI gate requires $command" >&2
        exit 1
    fi
done

cargo build --release --example rendering_benchmark_suite
assert_provenance

fixture_root=$(mktemp -d)
runtime_dir="$fixture_root/runtime"
mkdir "$runtime_dir"
mkdir "$fixture_root/config" "$fixture_root/data" "$fixture_root/cache"
chmod 700 "$runtime_dir"

release_sha=$start_sha
export SIM_ENGINE_HIDPI_BINARY="${CARGO_TARGET_DIR:?}/release/examples/rendering_benchmark_suite"
export SIM_ENGINE_HIDPI_READY_PATH="$fixture_root/ready"
export SIM_ENGINE_HIDPI_EVIDENCE_PATH="$fixture_root/evidence.txt"
export SIM_ENGINE_HIDPI_AUTO_EXIT=1
export SIM_ENGINE_RELEASE_SHA="$release_sha"
export WGPU_BACKEND=vulkan

dbus-run-session -- env \
    XDG_RUNTIME_DIR="$runtime_dir" \
    XDG_CONFIG_HOME="$fixture_root/config" \
    XDG_DATA_HOME="$fixture_root/data" \
    XDG_CACHE_HOME="$fixture_root/cache" \
    SIM_ENGINE_REQUIRE_PRODUCTION_SURFACE_FORMAT=0 \
    kwin_wayland \
    --virtual \
    --socket sim-engine-hidpi \
    --width 1280 \
    --height 720 \
    --scale 1 \
    --output-count 1 \
    --no-lockscreen \
    --no-global-shortcuts \
    --exit-with-session "$project_root/scripts/hidpi_transition_controller.sh"

assert_provenance
grep -Fxq "vcs_sha=$release_sha" "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
grep -Fxq 'backend=vulkan' "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
if [[ "${SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY:-0}" == 1 ]]; then
    grep -Fxq "adapter=$SIM_ENGINE_REQUIRED_ADAPTER_NAME" "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
    grep -Fxq "vendor=$SIM_ENGINE_REQUIRED_ADAPTER_VENDOR" "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
    grep -Fxq "device=$SIM_ENGINE_REQUIRED_ADAPTER_DEVICE" "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
    grep -Fxq "pci_bus_id=$SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID" "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
fi
grep -Fxq 'scale_factor=1.250' "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
grep -Fxq 'paired_transitions=1' "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
grep -Fxq 'completed_transitions=1' "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"

assert_provenance
cp "$SIM_ENGINE_HIDPI_EVIDENCE_PATH" "$published_evidence"
assert_provenance
publish_complete=1
echo "HiDPI evidence: $published_evidence"
