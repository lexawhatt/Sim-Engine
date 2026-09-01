#!/usr/bin/env bash
set -euo pipefail

: "${SIM_ENGINE_MATRIX_SCRIPT:?missing benchmark-matrix script}"

exec "$SIM_ENGINE_MATRIX_SCRIPT"
