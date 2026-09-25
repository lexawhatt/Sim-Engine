#!/usr/bin/env bash
# Exercise the standalone outer branch, without a display, Cargo or GPU work.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
test_root=$(mktemp -d -t sim-engine-dense-handshake.XXXXXX)
cleanup() { rm -rf -- "$test_root"; }
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
repository="$test_root/repository"
mkdir -p "$repository/scripts" "$test_root/bin"
cp "$project_root/scripts/dense_scene_benchmark_matrix.sh" "$repository/scripts/"
cp "$project_root/scripts/tests/fixtures/dense_dbus.sh" "$test_root/bin/dbus-run-session"
cp "$project_root/scripts/tests/fixtures/dense_kwin.sh" "$test_root/bin/kwin_wayland"
chmod u+x "$test_root/bin/"*
git -C "$repository" init --quiet
git -C "$repository" config user.name 'Handshake fixture'
git -C "$repository" config user.email 'handshake@example.invalid'
git -C "$repository" config commit.gpgsign false
git -C "$repository" add .
git -C "$repository" commit --quiet -m 'Fixture'
for scenario in missing wrong complete; do
    set +e
    env -u SIM_ENGINE_DENSE_COMPOSITOR -u SIM_ENGINE_RELEASE_SNAPSHOT \
        PATH="$test_root/bin:$PATH" DENSE_HANDSHAKE_TEST_MODE="$scenario" \
        SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID=fixture-only \
        SIM_ENGINE_DENSE_COMPLETION_TOKEN=stale \
        bash "$repository/scripts/dense_scene_benchmark_matrix.sh" --acceptance \
        >"$test_root/$scenario.log" 2>&1
    status=$?
    set -e
    if [[ "$scenario" == complete ]]; then
        if [[ "$status" -ne 0 ]]; then
            sed -n '1,120p' "$test_root/$scenario.log" >&2
            exit 1
        fi
    elif [[ "$status" -eq 0 ]]; then
        echo "accepted compositor success without correct child completion: $scenario" >&2
        exit 1
    fi
    echo "dense handshake regression passed: $scenario"
done
