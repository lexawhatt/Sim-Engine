#!/usr/bin/env bash
set -euo pipefail

: "${SIM_ENGINE_HIDPI_BINARY:?missing benchmark binary}"
: "${SIM_ENGINE_HIDPI_READY_PATH:?missing ready path}"

"$SIM_ENGINE_HIDPI_BINARY" --fixture hidpi_transition &
fixture_pid=$!

ready=0
for _ in {1..150}; do
    if [[ -f "$SIM_ENGINE_HIDPI_READY_PATH" ]]; then
        ready=1
        break
    fi
    if ! kill -0 "$fixture_pid" 2>/dev/null; then
        wait "$fixture_pid"
    fi
    sleep 0.1
done
if [[ "$ready" != 1 ]]; then
    kill "$fixture_pid" 2>/dev/null || true
    wait "$fixture_pid" 2>/dev/null || true
    echo "HiDPI fixture did not become ready" >&2
    exit 1
fi

output_id=$(kscreen-doctor -j | sed -n 's/^[[:space:]]*"id": \([0-9][0-9]*\),/\1/p' | head -n 1)
if [[ -z "$output_id" ]]; then
    kill "$fixture_pid" 2>/dev/null || true
    wait "$fixture_pid" 2>/dev/null || true
    echo "nested compositor exposed no configurable output" >&2
    exit 1
fi

kscreen-doctor "output.${output_id}.scale.1.25"
completed=0
for _ in {1..150}; do
    if ! kill -0 "$fixture_pid" 2>/dev/null; then
        wait "$fixture_pid"
        completed=1
        break
    fi
    sleep 0.1
done
if [[ "$completed" != 1 ]]; then
    kill "$fixture_pid" 2>/dev/null || true
    wait "$fixture_pid" 2>/dev/null || true
    echo "HiDPI fixture did not complete the compositor transaction" >&2
    exit 1
fi
