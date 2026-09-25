#!/usr/bin/env bash
# Parser and oracle-result regressions; no graphics, Cargo builds or network.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
test_root=$(mktemp -d -t sim-engine-dense-semantics.XXXXXX)
cleanup() { rm -rf -- "$test_root"; }
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$test_root/source/scripts" "$test_root/bin"
cp "$project_root/scripts/dense_scene_contexts.awk" "$test_root/source/scripts/"
touch "$test_root/source/Cargo.toml"
cp "$project_root/scripts/tests/fixtures/dense_semantics_cargo.sh" "$test_root/bin/cargo"
chmod u+x "$test_root/bin/cargo"
export PATH="$test_root/bin:$PATH"
export DENSE_SEMANTICS_TEST_SOURCE="$test_root/source"
export DENSE_SEMANTICS_TEST_CALLS="$test_root/calls"
# Inherited requirements belong to an earlier matrix and must be cleared.
export SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY=1
export SIM_ENGINE_REQUIRED_ADAPTER_NAME=wrong SIM_ENGINE_REQUIRED_ADAPTER_VENDOR=wrong
export SIM_ENGINE_REQUIRED_ADAPTER_DEVICE=wrong SIM_ENGINE_REQUIRED_ADAPTER_BACKEND=wrong
revision=1111111111111111111111111111111111111111
pci=0000:01:00.0
context() {
    printf 'DENSE_CONTEXT mode=Analytic labels=Batch constructions=100 phase=drag trial=0 generation=3 fixture_version=6 glyph_batch_chunks=24 vcs_sha=Some("%s") adapter="quoted \\"name\\" format=DoNotParseThis" backend=vulkan vendor=4318 device=9637 pci="%s" driver="driver" driver_info="info with spaces" physical=[1280, 800] scale=1 format=%s samples=%s present=Immediate refresh_millihertz=Some(60000) gpu_timing=Enabled\n' \
        "$revision" "$pci" "$1" "$2"
}
{
    context Rgba8UnormSrgb 4
    printf 'DENSE_RECOVERY restored_cpu_ms=120.0\n'
    context Bgra8UnormSrgb 1
    printf 'DENSE_RECOVERY status=drawn source_commands=1900\n'
    printf 'DENSE_COMPLETE attempts=20 drawn=20 skipped=0 phases=4 trials=1 target_misses=0\n'
} >"$test_root/good.log"
cp "$test_root/good.log" "$test_root/duplicate-pairs.log"
run() {
    bash "$project_root/scripts/dense_scene_semantics.sh" "$test_root/source" \
        "$test_root/output-$1" "$revision" "$pci" "${@:2}" >"$test_root/$1.out" 2>&1
}
run complete "$test_root/good.log" "$test_root/duplicate-pairs.log" || { cat "$test_root/complete.out" >&2; exit 1; }
[[ "$(wc -l <"$DENSE_SEMANTICS_TEST_CALLS")" == 2 ]]
grep -Fxq 'Bgra8UnormSrgb 1' "$DENSE_SEMANTICS_TEST_CALLS"
grep -Fxq 'Rgba8UnormSrgb 4' "$DENSE_SEMANTICS_TEST_CALLS"
grep -Fxq 'verified_format_pairs=2' "$test_root/output-complete/manifest.txt"
(cd "$test_root/output-complete" && sha256sum -c sha256.txt >/dev/null)
echo 'dense semantics regression passed: all actual/recovery pairs, deduplication, clean identity'

printf 'not a workload context\n' >"$test_root/no-context.log"
touch "$test_root/empty.log"
sed 's/ format=Rgba8UnormSrgb//' "$test_root/good.log" >"$test_root/missing-format.log"
sed 's/ samples=4/ samples=4 samples=1/' "$test_root/good.log" >"$test_root/duplicate-field.log"
sed 's/ samples=4/ samples=0/' "$test_root/good.log" >"$test_root/zero-msaa.log"
sed 's/format=Rgba8UnormSrgb/format=Unknown/' "$test_root/good.log" >"$test_root/unknown-format.log"
sed 's/backend=vulkan/backend=gl/' "$test_root/good.log" >"$test_root/foreign-backend.log"
sed 's/0000:01:00.0/0000:02:00.0/' "$test_root/good.log" >"$test_root/foreign-pci.log"
sed "s/$revision/0000000000000000000000000000000000000000/" "$test_root/good.log" >"$test_root/foreign-sha.log"
sed 's/gpu_timing=Enabled/gpu_timing=Disabled/' "$test_root/good.log" >"$test_root/no-timestamps.log"
sed '/format=Bgra8UnormSrgb/d' "$test_root/good.log" >"$test_root/no-recovery-context.log"
sed '/DENSE_RECOVERY/d' "$test_root/good.log" >"$test_root/no-recovery.log"
sed 's/driver="driver"/driver="unterminated/' "$test_root/good.log" >"$test_root/malformed-quote.log"
sed '/DENSE_RECOVERY status=drawn/i DENSE_RESTART reason=transition' "$test_root/good.log" >"$test_root/recovery-transition.log"
for scenario in no-context empty missing-format duplicate-field zero-msaa unknown-format \
    foreign-backend foreign-pci foreign-sha no-timestamps no-recovery-context no-recovery \
    malformed-quote recovery-transition; do
    if run "$scenario" "$test_root/good.log" "$test_root/$scenario.log"; then
        echo "incorrectly accepted malformed/missing context: $scenario" >&2; exit 1
    fi
    [[ ! -f "$test_root/output-$scenario/manifest.txt" ]]
    echo "dense semantics regression passed: reject $scenario"
done
for scenario in wrong_sha wrong_pci wrong_format wrong_samples wrong_backend duplicate_key \
    failure zero_tests ignored no_footer; do
    export DENSE_SEMANTICS_TEST_MODE="$scenario"
    if run "$scenario" "$test_root/good.log"; then
        echo "incorrectly accepted unverified oracle: $scenario" >&2; exit 1
    fi
    [[ ! -f "$test_root/output-$scenario/manifest.txt" ]]
    echo "dense semantics regression passed: reject oracle $scenario"
done
