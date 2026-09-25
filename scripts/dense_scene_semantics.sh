#!/usr/bin/env bash
# Qualify every format/MSAA pair actually used by dense workloads and recovery.
set -euo pipefail
if [[ $# -lt 5 ]]; then
    echo "usage: dense_scene_semantics.sh SOURCE_DIR OUTPUT_DIR SHA PCI WORKLOAD_LOG..." >&2
    exit 2
fi
source_dir=$1
output_dir=$2
revision=$3
pci=$4
shift 4
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source revision" >&2; exit 1; }
[[ "$pci" =~ ^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$ ]] || {
    echo "semantic qualification requires a physical PCI address" >&2; exit 1;
}
[[ -f "$source_dir/Cargo.toml" ]] || { echo "missing archived source" >&2; exit 1; }
for workload_log in "$@"; do
    [[ -s "$workload_log" && -r "$workload_log" ]] || {
        echo "missing or empty workload log: $workload_log" >&2; exit 1;
    }
done
mkdir "$output_dir"
export LC_ALL=C
awk -v expected_sha="$revision" -v expected_pci="$pci" \
    -f "$source_dir/scripts/dense_scene_contexts.awk" "$@" >"$output_dir/observed-contexts.tsv"
cut -f3,4 "$output_dir/observed-contexts.tsv" | sort -u >"$output_dir/observed-formats.tsv"
[[ -s "$output_dir/observed-formats.tsv" ]] || { echo "no observed formats" >&2; exit 1; }
test_name=renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract
verified=0
while IFS=$'\t' read -r format samples; do
    stem="$format-msaa$samples"
    evidence="$output_dir/$stem.txt"
    log="$output_dir/$stem.log"
    # Identity is selected by the PCI address, then revalidated in the evidence.
    # Stale name/vendor/format requirements from another matrix cannot override it.
    env -u SIM_ENGINE_REQUIRED_ADAPTER_NAME -u SIM_ENGINE_REQUIRED_ADAPTER_VENDOR \
        -u SIM_ENGINE_REQUIRED_ADAPTER_DEVICE -u SIM_ENGINE_REQUIRED_ADAPTER_BACKEND \
        WGPU_BACKEND=vulkan SIM_ENGINE_REQUIRE_GPU_TESTS=1 SIM_ENGINE_REQUIRE_VULKAN=1 \
        SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY=0 SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID="$pci" \
        SIM_ENGINE_REQUIRE_PRODUCTION_SURFACE_FORMAT=1 SIM_ENGINE_GPU_SURFACE_FORMAT="$format" \
        SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT="$samples" SIM_ENGINE_GPU_EVIDENCE_PATH="$evidence" \
        SIM_ENGINE_RELEASE_SHA="$revision" \
        cargo test --manifest-path "$source_dir/Cargo.toml" --offline --locked --all-features \
        --lib "$test_name" -- --exact --nocapture --test-threads=1 2>&1 | tee "$log"
    grep -Fxq 'running 1 test' "$log"
    grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out;' "$log"
    for expected in "vcs_sha=$revision" 'backend=Vulkan' "pci_bus_id=$pci" \
        "oracle_format=$format" "oracle_sample_count=$samples"; do
        key=${expected%%=*}
        # Reject duplicate contradictory keys, not merely any matching line.
        [[ "$(grep -c "^$key=" "$evidence")" == 1 ]]
        grep -Fxq "$expected" "$evidence"
    done
    verified=$((verified + 1))
done <"$output_dir/observed-formats.tsv"
printf 'vcs_sha=%s\npci_bus_id=%s\nbackend=Vulkan\nverified_format_pairs=%s\nstatus=complete\n' \
    "$revision" "$pci" "$verified" >"$output_dir/manifest.txt"
(cd "$output_dir" && sha256sum ./*-msaa*.log ./*-msaa*.txt ./manifest.txt ./observed-*.tsv) >"$output_dir/sha256.txt"
echo "Dense semantics verified $verified observed format/MSAA pairs"
