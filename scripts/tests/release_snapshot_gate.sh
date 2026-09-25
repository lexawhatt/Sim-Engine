#!/usr/bin/env bash
# No Cargo, display or GPU. Exercise the real wrapper in disposable repositories.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
test_root=$(mktemp -d -t sim-engine-snapshot-test.XXXXXX)
cleanup() {
    chmod -R u+w "$test_root"
    rm -rf -- "$test_root"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

for scenario in repeat fail dirty head; do
    repository="$test_root/$scenario"
    mkdir -p "$repository/scripts"
    cp "$project_root/scripts/release_snapshot_gate.sh" "$repository/scripts/"
    cp "$project_root/scripts/tests/fixtures/snapshot_gate.sh" "$repository/scripts/linux_release_gate.sh"
    chmod u+x "$repository/scripts/"*.sh
    git -C "$repository" init --quiet
    git -C "$repository" config user.name 'Snapshot fixture'
    git -C "$repository" config user.email 'snapshot@example.invalid'
    git -C "$repository" config commit.gpgsign false
    # Keep runtime output outside provenance checks, exactly like the real repo.
    printf '/target/\n' >"$repository/.gitignore"
    git -C "$repository" add .
    git -C "$repository" commit --quiet -m 'Fixture'
    start_sha=$(git -C "$repository" rev-parse HEAD)
    export SNAPSHOT_TEST_REPOSITORY="$repository"
    export SNAPSHOT_TEST_MODE=pass
    env -u SIM_ENGINE_RELEASE_SNAPSHOT -u SIM_ENGINE_RELEASE_SHA \
        bash "$repository/scripts/release_snapshot_gate.sh" scripts/linux_release_gate.sh \
        >"$test_root/$scenario-first.log" 2>&1
    grep -Fxq 'status=passed' "$repository/target/linux-release-evidence/completion.txt"
    grep -Fxq "vcs_sha=$start_sha" "$repository/target/linux-release-evidence/completion.txt"
    [[ "$scenario" == repeat ]] || export SNAPSHOT_TEST_MODE="$scenario"
    set +e
    env -u SIM_ENGINE_RELEASE_SNAPSHOT -u SIM_ENGINE_RELEASE_SHA \
        bash "$repository/scripts/release_snapshot_gate.sh" scripts/linux_release_gate.sh \
        >"$test_root/$scenario-second.log" 2>&1
    status=$?
    set -e
    if [[ "$scenario" == repeat ]]; then
        [[ "$status" -eq 0 ]]
        grep -Fxq 'status=passed' "$repository/target/linux-release-evidence/completion.txt"
    else
        [[ "$status" -ne 0 ]]
        if [[ "$scenario" == fail ]]; then [[ "$status" -eq 17 ]]; fi
        [[ ! -e "$repository/target/linux-release-evidence" ]]
        failures=("$repository"/target/linux-release-failed-*)
        [[ "${#failures[@]}" -eq 1 && -d "${failures[0]}" ]]
        grep -Fxq 'status=failed' "${failures[0]}/completion.txt"
        grep -Fxq "vcs_sha=$start_sha" "${failures[0]}/completion.txt"
        grep -Fxq 'retained diagnostic' "${failures[0]}/dense-scene/raw.log"
        grep -Fxq 'immutable input' "${failures[0]}/dense-scene/source/nested/input.txt"
    fi
    echo "snapshot regression passed: $scenario"
done
