#!/usr/bin/env bash
# Transport-only stub: preserve the real matrix's env/child invocation shape.
set -euo pipefail
[[ "$1" == -- ]]
shift
exec "$@"
