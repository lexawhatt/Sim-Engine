#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"

for command in dbus-run-session kwin_wayland kscreen-doctor; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "mandatory HiDPI gate requires $command" >&2
        exit 1
    fi
done

cargo build --release --example rendering_benchmark_suite

fixture_root=$(mktemp -d)
runtime_dir="$fixture_root/runtime"
mkdir "$runtime_dir"
mkdir "$fixture_root/config" "$fixture_root/data" "$fixture_root/cache"
chmod 700 "$runtime_dir"
trap 'rm -rf -- "$fixture_root"' EXIT

release_sha=$(git rev-parse HEAD)
export SIM_ENGINE_HIDPI_BINARY="$project_root/target/release/examples/rendering_benchmark_suite"
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

grep -Fxq "vcs_sha=$release_sha" "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
grep -Fxq 'backend=vulkan' "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
grep -Fxq 'scale_factor=1.250' "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
grep -Fxq 'paired_transitions=1' "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"
grep -Fxq 'completed_transitions=1' "$SIM_ENGINE_HIDPI_EVIDENCE_PATH"

mkdir -p target
cp "$SIM_ENGINE_HIDPI_EVIDENCE_PATH" target/linux-hidpi-transition.txt
echo "HiDPI evidence: target/linux-hidpi-transition.txt"
