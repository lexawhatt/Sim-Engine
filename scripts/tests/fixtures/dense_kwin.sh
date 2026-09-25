#!/usr/bin/env bash
# Simulate a compositor returning zero independently of child completion.
set -euo pipefail
: "${SIM_ENGINE_DENSE_COMPLETION_PATH:?}"
: "${SIM_ENGINE_DENSE_COMPLETION_TOKEN:?}"
[[ "$SIM_ENGINE_DENSE_COMPLETION_TOKEN" != stale ]]
case "${DENSE_HANDSHAKE_TEST_MODE:?}" in
    missing) ;;
    wrong) printf 'wrong invocation\n' >"$SIM_ENGINE_DENSE_COMPLETION_PATH" ;;
    complete) printf '%s\n' "$SIM_ENGINE_DENSE_COMPLETION_TOKEN" >"$SIM_ENGINE_DENSE_COMPLETION_PATH" ;;
    *) exit 2 ;;
esac
exit 0
