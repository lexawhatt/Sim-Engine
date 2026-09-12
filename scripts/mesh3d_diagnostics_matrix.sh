#!/usr/bin/env bash
set -euo pipefail

# Diagnostic baselines only. Existing release/HiDPI/performance gates remain
# authoritative and are neither replaced nor weakened by this workload matrix.
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
export GIT_NO_REPLACE_OBJECTS=1
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "3D diagnostics evidence requires a clean source tree" >&2
    exit 1
fi
start_sha=$(git rev-parse HEAD)
snapshot_parent=$(mktemp -d)
cleanup() {
    chmod -R u+w "$snapshot_parent" 2>/dev/null || true
    # Exact mktemp directory, never the project or a caller-supplied target.
    rm -rf -- "$snapshot_parent"
}
trap cleanup EXIT
mkdir "$snapshot_parent/source"
git archive "$start_sha" | tar -x -C "$snapshot_parent/source"
chmod -R a-w "$snapshot_parent/source"
export CARGO_TARGET_DIR="$project_root/target/mesh3d-diagnostics-build"
mkdir -p "$CARGO_TARGET_DIR"
exec 9>"$CARGO_TARGET_DIR/.matrix.lock"
flock -n 9 || { echo "another 3D diagnostics matrix is running" >&2; exit 1; }
export SIM_ENGINE_RELEASE_SHA="$start_sha"
export WGPU_BACKEND=vulkan
if [[ -z "${SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID:-}" ]]; then
    echo "set SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID to pin the physical GPU" >&2
    exit 1
fi
cargo build --manifest-path "$snapshot_parent/source/Cargo.toml" --locked --offline \
    --release --all-features --example mesh3d_scene_benchmark
binary="$snapshot_parent/mesh3d_scene_benchmark"
install -m 555 "$CARGO_TARGET_DIR/release/examples/mesh3d_scene_benchmark" "$binary"
export SIM_ENGINE_BENCHMARK_BINARY_SHA256
SIM_ENGINE_BENCHMARK_BINARY_SHA256=$(sha256sum "$binary" | cut -d ' ' -f1)
output_parent="$project_root/target/mesh3d-diagnostics"
mkdir -p "$output_parent"
staged=$(mktemp -d "$output_parent/.pending.XXXXXX")
printf 'vcs_sha=%s\nbinary_sha256=%s\nrequested_backend=Vulkan\nrequested_pci=%s\nrelease_gate=false\n' \
    "$start_sha" "$SIM_ENGINE_BENCHMARK_BINARY_SHA256" "$SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID" >"$staged/manifest.txt"
rustc -vV >"$staged/rustc.txt"
uname -a >"$staged/kernel.txt"
if command -v lscpu >/dev/null 2>&1; then lscpu >"$staged/cpu.txt"; fi
run_case() {
    local name=$1 policy=$2 objects=$3 side=$4
    echo "3D diagnostics: $name / $policy / $objects objects / side $side"
    timeout 600 "$binary" --case "$name" --policy "$policy" --objects "$objects" \
        --side "$side" --frames 60 --trials 3 2>&1 | tee "$staged/$name-$policy.txt"
    if [[ "$(git rev-parse HEAD)" != "$start_sha" || -n "$(git status --porcelain --untracked-files=all)" ]]; then
        echo "source changed during diagnostics; incomplete bundle is not evidence" >&2
        exit 1
    fi
}
for policy in native strict; do
    run_case repeated "$policy" 512 1
    run_case distinct "$policy" 256 1
    run_case outside "$policy" 512 1
    run_case host_hidden "$policy" 512 1
    run_case crossing "$policy" 256 1
done
for name in immutable dynamic growth; do run_case "$name" native 64 16; done
for name in textured mask blend texture_update prepared_text; do
    run_case "$name" native 256 1
done
if [[ "$(git rev-parse HEAD)" != "$start_sha" || -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "source changed before diagnostics publication" >&2
    exit 1
fi
printf 'status=complete\nconfirmed_trials=54\n' >>"$staged/manifest.txt"
# Unique result directory preserves all older attempts and failed .pending logs.
final="$output_parent/$start_sha-$(date -u +%Y%m%dT%H%M%SZ)"
mv -T "$staged" "$final"
echo "3D diagnostics complete: $final"
