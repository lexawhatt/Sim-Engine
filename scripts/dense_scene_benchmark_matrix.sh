#!/usr/bin/env bash
set -euo pipefail

# Diagnostics retain all baseline failures. --acceptance is a separate, fixed
# analytic+batch 16.67ms workload on one isolated 1280x800 KWin output; it does
# not replace semantic/HiDPI gates or label unsupported legacy baselines passed.
case "$#:${1:-}" in
    0:) ;;
    1:--acceptance) export SIM_ENGINE_DENSE_ACCEPTANCE=1 ;;
    *) echo "usage: dense_scene_benchmark_matrix.sh [--acceptance]" >&2; exit 2 ;;
esac
acceptance=${SIM_ENGINE_DENSE_ACCEPTANCE:-0}
[[ "$acceptance" == 0 || "$acceptance" == 1 ]] || { echo "invalid acceptance mode" >&2; exit 2; }
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
export GIT_NO_REPLACE_OBJECTS=1
start_sha=$(git rev-parse HEAD)
assert_provenance() {
    if [[ "$(git rev-parse HEAD)" != "$start_sha" || -n "$(git status --porcelain --untracked-files=all)" ]]; then
        echo "dense-scene diagnostics require unchanged, clean source provenance" >&2
        exit 1
    fi
}
assert_provenance
if [[ -z "${SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID:-}" ]]; then
    echo "set SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID to the reference physical GPU" >&2
    exit 1
fi
if [[ "$acceptance" == 1 && "${SIM_ENGINE_DENSE_COMPOSITOR:-0}" != 1 ]]; then
    for command in dbus-run-session kwin_wayland; do
        command -v "$command" >/dev/null 2>&1 || { echo "dense acceptance requires $command" >&2; exit 1; }
    done
    dense_compositor_root=$(mktemp -d -t sim-engine-dense.XXXXXX)
    cleanup_compositor() { rm -rf -- "$dense_compositor_root"; }
    trap cleanup_compositor EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    mkdir "$dense_compositor_root/runtime" "$dense_compositor_root/config" \
        "$dense_compositor_root/data" "$dense_compositor_root/cache"
    chmod 700 "$dense_compositor_root/runtime"
    export SIM_ENGINE_DENSE_COMPOSITOR=1
    export SIM_ENGINE_DENSE_ACCEPTANCE=1
    export SIM_ENGINE_MATRIX_SCRIPT="$project_root/scripts/dense_scene_benchmark_matrix.sh"
    # KWin can exit successfully even when its session child fails. Require a
    # fresh per-invocation completion token written by the actual child gate.
    export SIM_ENGINE_DENSE_COMPLETION_PATH="$dense_compositor_root/child-complete"
    export SIM_ENGINE_DENSE_COMPLETION_TOKEN="$start_sha:${dense_compositor_root##*/}"
    dbus-run-session -- env \
        XDG_RUNTIME_DIR="$dense_compositor_root/runtime" \
        XDG_CONFIG_HOME="$dense_compositor_root/config" \
        XDG_DATA_HOME="$dense_compositor_root/data" \
        XDG_CACHE_HOME="$dense_compositor_root/cache" \
        kwin_wayland --virtual --socket sim-engine-dense --width 1280 --height 800 \
        --scale 1 --output-count 1 --no-lockscreen --no-global-shortcuts \
        --exit-with-session "$project_root/scripts/rendering_benchmark_matrix_controller.sh"
    assert_provenance
    if [[ ! -f "$SIM_ENGINE_DENSE_COMPLETION_PATH" ||
        "$(<"$SIM_ENGINE_DENSE_COMPLETION_PATH")" != "$SIM_ENGINE_DENSE_COMPLETION_TOKEN" ]]; then
        echo "dense acceptance child did not confirm successful completion" >&2
        exit 1
    fi
    exit 0
fi
if [[ "$acceptance" == 1 ]]; then
    : "${SIM_ENGINE_DENSE_COMPLETION_PATH:?missing dense completion path}"
    : "${SIM_ENGINE_DENSE_COMPLETION_TOKEN:?missing dense completion token}"
fi
frames=${SIM_ENGINE_DENSE_FRAMES:-120}
trials=${SIM_ENGINE_DENSE_TRIALS:-3}
warmup=${SIM_ENGINE_DENSE_WARMUP:-20}
read -r -a modes <<<"${SIM_ENGINE_DENSE_MODES:-streaming prepared analytic}"
extra_arguments=()
release_gate=false
if [[ "$acceptance" == 1 ]]; then
    frames=120
    trials=3
    warmup=20
    modes=(analytic)
    extra_arguments=(--maximum-ms 16.67)
    release_gate=true
fi
[[ "${#modes[@]}" -gt 0 ]] || { echo "at least one mode is required" >&2; exit 2; }
for mode in "${modes[@]}"; do
    case "$mode" in streaming | prepared | analytic) ;; *) echo "unknown dense mode: $mode" >&2; exit 2 ;; esac
done
for count in "$frames" "$trials" "$warmup"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "frames/trials/warmup must be positive integers" >&2; exit 2; }
done
if [[ "${SIM_ENGINE_RELEASE_SNAPSHOT:-0}" == 1 ]]; then
    [[ "$start_sha" == "${SIM_ENGINE_RELEASE_SHA:?missing release SHA}" ]] || { echo "snapshot SHA mismatch" >&2; exit 1; }
    output_parent="${SIM_ENGINE_RELEASE_OUTPUT_DIR:?missing release output}/dense-scene"
    : "${CARGO_TARGET_DIR:?missing snapshot build directory}"
else
    output_parent="$project_root/target/dense-scene-diagnostics"
    export CARGO_TARGET_DIR="$project_root/target/dense-scene-build"
fi
mkdir -p "$output_parent" "$CARGO_TARGET_DIR"
exec 8>"$CARGO_TARGET_DIR/.dense-matrix.lock"
flock -n 8 || { echo "another dense-scene matrix is running" >&2; exit 1; }
staged=$(mktemp -d "$output_parent/.pending.XXXXXX")
mkdir "$staged/source"
git archive "$start_sha" | tar -x -C "$staged/source"
chmod -R a-w "$staged/source"
# Preserve the snapshot and failed logs in the unique pending directory. Never
# overwrite an existing successful bundle or discard a failed workload.
export WGPU_BACKEND=vulkan
export SIM_ENGINE_RELEASE_SHA="$start_sha"
printf 'vcs_sha=%s\nrequested_backend=Vulkan\nrequested_pci=%s\nframes=%s\ntrials=%s\nwarmup=%s\nrelease_gate=%s\nrun_started=true\n' \
    "$start_sha" "$SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID" "$frames" "$trials" "$warmup" "$release_gate" >"$staged/manifest.txt"
printf 'requested_modes=%s\n' "${modes[*]}" >>"$staged/manifest.txt"
printf 'requested_present_mode=NoVsync\n' >>"$staged/manifest.txt"
if [[ "$acceptance" == 1 ]]; then
    printf 'maximum_end_to_end_p95_ms=16.67\nreference_output=1280x800@1 isolated KWin\nrequired_gpu_timestamps=true\nrequired_recovery=true\n' >>"$staged/manifest.txt"
fi
rustc -vV >"$staged/rustc.txt"
uname -a >"$staged/kernel.txt"
if command -v lscpu >/dev/null 2>&1; then lscpu >"$staged/cpu.txt"; fi
cargo build --manifest-path "$staged/source/Cargo.toml" --offline --locked \
    --release --all-features --example dense_scene_benchmark 2>&1 | tee "$staged/build.log"
binary="$staged/dense_scene_benchmark"
install -m 555 "$CARGO_TARGET_DIR/release/examples/dense_scene_benchmark" "$binary"
(cd "$staged" && sha256sum dense_scene_benchmark) >"$staged/binary-sha256.txt"
(cd "$staged" && sha256sum source/examples/dense_scene_benchmark.rs \
    source/examples/support/dense_scene_*.rs) >"$staged/fixture-sha256.txt"
assert_provenance
failures=0
completed=0
for constructions in 100 500 2000; do
    for mode in "${modes[@]}"; do
        if [[ "$mode" == streaming ]]; then label_mode=runs; else label_mode=batch; fi
        label_modes=(off "$label_mode")
        if [[ "$acceptance" == 1 ]]; then label_modes=(batch); fi
        for labels in "${label_modes[@]}"; do
            assert_provenance
            case_name="$mode-$labels-$constructions"
            echo "Dense scene: $case_name"
            set +e
            timeout 1800 "$binary" --constructions "$constructions" --mode "$mode" \
                --labels "$labels" --frames "$frames" --trials "$trials" --warmup "$warmup" \
                --require-gpu --recovery "${extra_arguments[@]}" 2>&1 | tee "$staged/$case_name.log"
            statuses=("${PIPESTATUS[@]}")
            status=${statuses[0]}
            set -e
            if [[ "${statuses[1]}" -ne 0 ]]; then
                echo "failed to preserve workload output" >&2
                exit 1
            fi
            printf 'case=%s exit_status=%s\n' "$case_name" "$status" >>"$staged/cases.txt"
            if [[ "$status" -ne 0 ]]; then
                failures=$((failures + 1))
            elif ! grep -Eq '^DENSE_COMPLETE .*skipped=0 .*target_misses=0$' "$staged/$case_name.log"; then
                failures=$((failures + 1))
                echo "case exited without a complete successful report" >>"$staged/cases.txt"
            else
                completed=$((completed + 1))
            fi
            assert_provenance
        done
    done
done
assert_provenance
if [[ "$failures" -ne 0 ]]; then
    printf 'status=failed\ncompleted_cases=%s\nfailed_cases=%s\n' "$completed" "$failures" >>"$staged/manifest.txt"
    echo "Dense diagnostics contain failed workloads; preserved at $staged" >&2
    exit 1
fi
printf 'status=complete\ncompleted_cases=%s\nfailed_cases=0\n' "$completed" >>"$staged/manifest.txt"
(cd "$staged" && sha256sum ./*.log) >"$staged/logs-sha256.txt"
final="$output_parent/$start_sha-$(date -u +%Y%m%dT%H%M%SZ)-${staged##*.}"
mv -T -- "$staged" "$final"
if [[ "$acceptance" == 1 && "${SIM_ENGINE_RELEASE_SNAPSHOT:-0}" == 1 ]]; then
    cp "$final/manifest.txt" "$SIM_ENGINE_RELEASE_OUTPUT_DIR/linux-dense-scene-performance.txt"
fi
assert_provenance
echo "Dense diagnostics complete: $final"
if [[ "$acceptance" == 1 ]]; then
    printf '%s\n' "$SIM_ENGINE_DENSE_COMPLETION_TOKEN" >"$SIM_ENGINE_DENSE_COMPLETION_PATH"
fi
