#!/bin/bash
# MOTION=<motion.pt> bash scripts/train_flashdexretarget.sh [--profile isaac|mjwarp|newton] [--config_name NAME] [Hydra overrides]
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

profile=isaac
config_name=""
overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) profile="${2:?--profile requires a value}"; shift 2 ;;
        --config_name) config_name="${2:?--config_name requires a value}"; shift 2 ;;
        -h|--help)
            echo "Usage: MOTION=<motion.pt> bash scripts/train_flashdexretarget.sh [--profile isaac|mjwarp|newton] [--config_name NAME] [Hydra overrides]"
            exit 0 ;;
        *) overrides+=("$1"); shift ;;
    esac
done
if [[ -n "${ROBOT:-}" ]]; then
    echo '[train] ROBOT is no longer a config selector; unset ROBOT and pass envs/robot@robot=<config>, e.g. envs/robot@robot=sharpa_bimanual' >&2
    exit 1
fi
case "$profile" in
    isaac) env_file=scripts/env.sh; default_config=flashdexretarget ;;
    mjwarp|newton) env_file=env/native.sh; default_config=native ;;
    *) echo "[train] unknown profile: $profile" >&2; exit 1 ;;
esac
source "$env_file"
unset HEADLESS   # Isaac Lab's own env var; use the headless=false override instead
motion="$(realpath "${MOTION:?set MOTION=<dataset>/motion.pt}")"
dataset="$(basename "$(dirname "$motion")")"
args=(--config_name "${config_name:-$default_config}")
if [[ "$profile" != isaac ]]; then
    args+=(--overrides "envs/simulator@simulator=$profile")
fi
args+=(
    --overrides "commands.motion_file=$motion"
    --overrides "logger.wandb.run_name=flashdexretarget-${dataset}-${profile}"
    --overrides "logger.wandb.tags=[${dataset},${profile}]"
)
for ov in "${overrides[@]}"; do args+=(--overrides "$ov"); done
python train.py "${args[@]}"
