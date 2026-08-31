#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: release_snapshot_gate.sh <repository-relative gate script> [arguments...]" >&2
    exit 2
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gate_script=$1
shift
case "$gate_script" in
    scripts/linux_release_gate.sh | scripts/rendering_benchmark_matrix.sh | scripts/hidpi_transition_gate.sh) ;;
    *)
        echo "release snapshot gate only runs repository scripts" >&2
        exit 2
        ;;
esac

cd "$project_root"
output_dir="$project_root/target"
mkdir -p "$output_dir"
final_bundle="$output_dir/linux-release-evidence"
staging_dir=$(mktemp -d "$output_dir/.linux-release-evidence.XXXXXX")
snapshot_parent=""
snapshot_root=""
snapshot_added=0
gate_complete=0
cleanup() {
    if [[ "$snapshot_added" -eq 1 ]]; then
        chmod -R u+w "$snapshot_root" 2>/dev/null || true
        git -C "$project_root" worktree remove --force "$snapshot_root" >/dev/null 2>&1 || true
    fi
    if [[ -n "$snapshot_parent" ]]; then
        rm -rf -- "$snapshot_parent"
    fi
    if [[ "$gate_complete" -ne 1 ]]; then
        rm -rf -- "$staging_dir"
    fi
    if [[ "$gate_complete" -ne 1 ]]; then
        rm -rf -- "$final_bundle"
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Never let a failed new invocation leave an older successful bundle looking
# current, including failures before tool discovery or compilation. Individual
# legacy manifests are also invalidated during the transition to the bundle.
rm -rf -- "$final_bundle"
rm -f -- "$output_dir/linux-vulkan-surface.txt" \
    "$output_dir/linux-vulkan-adapter.txt" \
    "$output_dir/linux-hidpi-transition.txt"

if [[ -n "$(git replace -l)" ]]; then
    echo "release snapshot gate rejects Git replacement refs" >&2
    exit 1
fi
grafts_path=$(git rev-parse --git-path info/grafts)
if [[ "$grafts_path" != /* ]]; then
    grafts_path="$project_root/$grafts_path"
fi
if [[ -s "$grafts_path" ]]; then
    echo "release snapshot gate rejects legacy Git grafts" >&2
    exit 1
fi
export GIT_NO_REPLACE_OBJECTS=1

start_sha=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "release snapshot gate requires a clean worktree" >&2
    exit 1
fi

snapshot_parent=$(mktemp -d)
snapshot_root="$snapshot_parent/source"

git worktree add --detach --quiet "$snapshot_root" "$start_sha"
snapshot_added=1
# Cargo writes only to the separate target directory. Making the detached
# source tree read-only turns accidental build-script/source mutation into a
# hard failure while the object database pins every input to start_sha.
chmod -R a-w "$snapshot_root"

set +e
env \
    SIM_ENGINE_RELEASE_SNAPSHOT=1 \
    SIM_ENGINE_RELEASE_SHA="$start_sha" \
    SIM_ENGINE_RELEASE_OUTPUT_DIR="$staging_dir" \
    CARGO_TARGET_DIR="$snapshot_parent/cargo-target" \
    "$snapshot_root/$gate_script" "$@"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
    exit "$status"
fi
if [[ "$(git -C "$project_root" rev-parse HEAD)" != "$start_sha" ]]; then
    echo "release source HEAD changed while the exact-SHA snapshot was running" >&2
    exit 1
fi
if [[ -n "$(git -C "$project_root" status --porcelain --untracked-files=all)" ]]; then
    echo "release source worktree changed while the exact-SHA snapshot was running" >&2
    exit 1
fi

case "$gate_script" in
    scripts/linux_release_gate.sh | scripts/rendering_benchmark_matrix.sh)
        required_evidence=(
            linux-vulkan-surface.txt
            linux-vulkan-adapter.txt
            linux-hidpi-transition.txt
        )
        ;;
    scripts/hidpi_transition_gate.sh)
        required_evidence=(linux-hidpi-transition.txt)
        ;;
esac
for evidence in "${required_evidence[@]}"; do
    if [[ ! -s "$staging_dir/$evidence" ]]; then
        echo "release gate completed without required staged evidence: $evidence" >&2
        exit 1
    fi
done
printf 'format_version=1\nvcs_sha=%s\ngate_script=%s\nstatus=passed\n' \
    "$start_sha" "$gate_script" >"$staging_dir/completion.txt"
# This directory rename is the only publication boundary. SIGKILL before it
# can leave a hidden staging directory, never a bundle that claims success.
mv -- "$staging_dir" "$final_bundle"
echo "Release evidence bundle: $final_bundle"
gate_complete=1
