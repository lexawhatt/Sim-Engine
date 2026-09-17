#!/usr/bin/env bash
# Keep source layout rules independent of installed Rust/Clippy versions.
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_root=${1:-"$project_root/src"}
if [[ ! -d "$source_root" ]]; then
    echo "source directory does not exist: $source_root" >&2
    exit 1
fi

status=0
files=0
while IFS= read -r -d '' file; do
    files=$((files + 1))
    name=${file##*/}
    if [[ "$name" != mod.rs && -d "${file%.rs}" ]]; then
        echo "module with children must use ${file%.rs}/mod.rs: $file" >&2
        status=1
    fi
    case "$file" in
        */tests.rs | *_tests.rs | */*_tests/*)
            echo "external tests belong under the owning module's tests/: $file" >&2
            status=1
            ;;
        */tests/dev[0-9]*.rs | */tests/red.rs | */tests/red_team.rs)
            echo "name tests by their contract, not development history: $file" >&2
            status=1
            ;;
    esac
    if grep -nE '^[[:space:]]*#\[path[[:space:]]*=' "$file"; then
        echo "use ordinary module declarations, not path overrides: $file" >&2
        status=1
    fi
    case "$file" in
        */tests/*) ;;
        *)
            if grep -nE '^use[[:space:]].*::\*;' "$file"; then
                echo "production module boundaries require explicit imports: $file" >&2
                status=1
            fi
            ;;
    esac
done < <(find "$source_root" -type f -name '*.rs' -print0)
if [[ "$files" -eq 0 ]]; then
    echo "no Rust source files found: $source_root" >&2
    exit 1
fi
exit "$status"
