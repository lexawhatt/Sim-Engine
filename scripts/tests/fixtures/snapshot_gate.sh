#!/usr/bin/env bash
# Hermetic child for release_snapshot_gate.sh tests, never a real GPU gate.
set -euo pipefail
output=${SIM_ENGINE_RELEASE_OUTPUT_DIR:?}
mkdir -p "$output/dense-scene/source/nested"
printf 'retained diagnostic\n' >"$output/dense-scene/raw.log"
printf 'immutable input\n' >"$output/dense-scene/source/nested/input.txt"
chmod -R a-w "$output/dense-scene/source"
for evidence in linux-vulkan-surface linux-vulkan-adapter linux-vulkan-performance \
    linux-hidpi-transition linux-dense-scene-performance; do
    printf 'stub only\n' >"$output/$evidence.txt"
done
case "${SNAPSHOT_TEST_MODE:?}" in
    pass) ;;
    fail) exit 17 ;;
    dirty) touch "$SNAPSHOT_TEST_REPOSITORY/changed-during-gate" ;;
    head) git -C "$SNAPSHOT_TEST_REPOSITORY" commit --quiet --allow-empty -m 'Concurrent commit' ;;
    *) exit 2 ;;
esac
