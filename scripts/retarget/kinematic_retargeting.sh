#!/bin/bash
# --profile selects the environment; remaining arguments are Hydra configuration.
set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
profile=isaac
overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) profile="${2:?--profile requires a value}"; shift 2 ;;
        *) overrides+=("$1"); shift ;;
    esac
done
case "$profile" in
    isaac) env_file="$REPO_DIR/scripts/env.sh" ;;
    mjwarp|newton) env_file="$REPO_DIR/env/native.sh" ;;
    *) echo "[retarget] unknown profile: $profile" >&2; exit 1 ;;
esac
if [[ ! -f "$env_file" ]]; then
    echo "[retarget] missing activation: $env_file; run scripts/setup.sh --profile $profile" >&2
    exit 1
fi
source "$env_file"
exec python "$REPO_DIR/src/retarget/kinematic_retargeting.py" "${overrides[@]}"
