#!/bin/bash
# Provision one explicitly selected runtime; engine/task selection stays in Hydra.
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE=isaac
CONDA_ENV=""
DEXFORGE_ROOT=""
NATIVE_MOTION_FILE=""
ROBOT_CONFIGS=()
SKIP_ASSETS=0
REQUIREMENTS_LOCK=""
PACKAGE_WHEELS=()

usage() {
    cat <<EOF
Usage: bash scripts/setup.sh [--profile isaac|mjwarp|newton] [--env NAME]
       [--robot-config YAML ... | --native-motion-file PATH] [--skip_assets]
       [--dexforge-root PATH] [--requirements-lock FILE] [--package-wheel FILE ...]

  --profile             isaac (Python 3.11, default) or native mjwarp/newton (3.12)
  --env NAME            conda env (default: FlashDexRetarget or FlashDexRetargetNative)
  --robot-config YAML   stock robot/resolved run config to convert; repeat for several
  --native-motion-file PATH  motion.pt beside the native scene manifest
  --skip_assets         skip Isaac asset conversion
  --dexforge-root PATH  required for Newton's public DexForge core/tracking packages
  --requirements-lock FILE  native-only complete third-party closure; install --no-deps
  --package-wheel FILE  native-only whole dependency wheel; repeat for several artifacts
EOF
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="${2:?--profile requires a value}"; shift 2 ;;
        --env) CONDA_ENV="${2:?--env requires a value}"; shift 2 ;;
        --env=*) CONDA_ENV="${1#*=}"; shift ;;
        --robot-config) ROBOT_CONFIGS+=("${2:?--robot-config requires a path}"); shift 2 ;;
        --native-motion-file) NATIVE_MOTION_FILE="${2:?--native-motion-file requires a path}"; shift 2 ;;
        --dexforge-root) DEXFORGE_ROOT="${2:?--dexforge-root requires a path}"; shift 2 ;;
        --skip_assets) SKIP_ASSETS=1; shift ;;
        --requirements-lock) REQUIREMENTS_LOCK="${2:?--requirements-lock requires a path}"; shift 2 ;;
        --package-wheel) PACKAGE_WHEELS+=("${2:?--package-wheel requires a path}"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[setup] unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
case "$PROFILE" in
    isaac)
        PYTHON_VERSION=3.11
        CONDA_ENV="${CONDA_ENV:-FlashDexRetarget}"
        ENV_FILE="$PROJECT_DIR/scripts/env.sh"
        ;;
    mjwarp|newton)
        PYTHON_VERSION=3.12
        CONDA_ENV="${CONDA_ENV:-FlashDexRetargetNative}"
        ENV_FILE="$PROJECT_DIR/env/native.sh"
        if [[ ${#ROBOT_CONFIGS[@]} -gt 0 || -n "$NATIVE_MOTION_FILE" ]]; then
            echo "[setup] native profiles use authored MJCF assets; Isaac conversion options require --profile isaac" >&2
            exit 1
        fi
        ;;
    *) echo "[setup] unknown profile: $PROFILE" >&2; exit 1 ;;
esac
if [[ "$PROFILE" == isaac && ( -n "$REQUIREMENTS_LOCK" || ${#PACKAGE_WHEELS[@]} -gt 0 ) ]]; then
    echo '[setup] --requirements-lock and --package-wheel require a native profile' >&2
    exit 1
fi
if [[ -n "$REQUIREMENTS_LOCK" ]]; then
    [[ -f "$REQUIREMENTS_LOCK" ]] || { echo "[setup] missing requirements lock: $REQUIREMENTS_LOCK" >&2; exit 1; }
    REQUIREMENTS_LOCK="$(cd "$(dirname "$REQUIREMENTS_LOCK")" && pwd)/$(basename "$REQUIREMENTS_LOCK")"
fi
for index in "${!PACKAGE_WHEELS[@]}"; do
    wheel="${PACKAGE_WHEELS[$index]}"
    [[ -f "$wheel" ]] || { echo "[setup] missing package wheel: $wheel" >&2; exit 1; }
    PACKAGE_WHEELS[$index]="$(cd "$(dirname "$wheel")" && pwd)/$(basename "$wheel")"
done
if [[ -n "$NATIVE_MOTION_FILE" && ${#ROBOT_CONFIGS[@]} -gt 0 ]]; then
    echo "[setup] select stock robot configs or one resolved native scene" >&2
    exit 1
fi
if [[ "$PROFILE" == newton ]]; then
    : "${DEXFORGE_ROOT:?--profile newton requires --dexforge-root PATH}"
    DEXFORGE_ROOT="$(cd "$DEXFORGE_ROOT" && pwd)"
    for package in dexforge_core dexforge_tracking; do
        [[ -f "$DEXFORGE_ROOT/packages/$package/pyproject.toml" ]] || {
            echo "[setup] missing public package: $DEXFORGE_ROOT/packages/$package" >&2
            exit 1
        }
    done
elif [[ -n "$DEXFORGE_ROOT" ]]; then
    echo "[setup] --dexforge-root is used by --profile newton" >&2
    exit 1
fi
cd "$PROJECT_DIR"

# Initialize the recorded submodule commit, not a separately selected trainer.
git submodule update --init third_party/rsl_rl_flashsac
if ! command -v conda &>/dev/null; then
    for p in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$p/etc/profile.d/conda.sh" ] && { source "$p/etc/profile.d/conda.sh"; break; }
    done
fi
command -v conda &>/dev/null || { echo "[setup] conda not found" >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "[setup] creating $CONDA_ENV (Python $PYTHON_VERSION, $PROFILE)"
    conda create -n "$CONDA_ENV" "python=$PYTHON_VERSION" -y
fi
conda activate "$CONDA_ENV"
unset PYTHONPATH
# Reject an existing environment for the other ABI before replacing its packages.
python - "$PYTHON_VERSION" <<'PY'
import sys

expected = tuple(map(int, sys.argv[1].split(".")))
if sys.version_info[:2] != expected:
    raise SystemExit(f"Selected profile requires Python {sys.argv[1]}, got {sys.version.split()[0]}")
PY
pip install "setuptools<81" wheel
if [[ "$PROFILE" == isaac ]]; then
    pip install --no-build-isolation flatdict==4.0.1
    FREEZE=$(mktemp)
    trap 'rm -f "$FREEZE"' EXIT
    grep -v -E '^(rsl-rl-lib|tensordict)==' requirements/isaac_cu128.txt > "$FREEZE"
    pip install --no-deps --prefer-binary -r "$FREEZE" \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.nvidia.com
    pip install --no-deps rsl-rl-lib==5.5.1 tensordict==0.13.0 pyvers==0.2.2
    pip install --no-deps coacd==1.0.10 cuacd==0.1.0 smplx==0.1.28 fast-simplification==0.1.13
elif [[ -n "$REQUIREMENTS_LOCK" ]]; then
    # The selected lock already contains the complete third-party closure.
    pip install --no-deps -r "$REQUIREMENTS_LOCK"
else
    # The separate pinned set may resolve its dependencies only in this native env.
    pip install --prefer-binary -r requirements/native_cu128.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128
fi
for wheel in "${PACKAGE_WHEELS[@]}"; do
    pip install --no-deps --force-reinstall "$wheel"
done
pip install --no-deps -e third_party/rsl_rl_flashsac
pip install --no-deps -e .
if [[ "$PROFILE" == newton ]]; then
    pip install --no-deps -e "$DEXFORGE_ROOT/packages/dexforge_core" \
        -e "$DEXFORGE_ROOT/packages/dexforge_tracking"
fi

# Generated activation files reflect this selected environment, not a previous run.
mkdir -p "$(dirname "$ENV_FILE")"
{
    echo '# Generated by scripts/setup.sh; keep operator overrides in the calling shell.'
    printf 'source %q\n' "$CONDA_BASE/etc/profile.d/conda.sh"
    printf 'conda activate %q || return $?\n' "$CONDA_ENV"
    echo 'unset PYTHONPATH'
    printf "python - %q <<'PY' || return \$?\n" "$PYTHON_VERSION"
    cat <<'PY'
import sys

expected = tuple(map(int, sys.argv[1].split(".")))
if sys.version_info[:2] != expected:
    raise SystemExit(f"Selected profile requires Python {sys.argv[1]}, got {sys.version.split()[0]}")
PY
    echo 'PY'
    echo 'export MUJOCO_GL=${MUJOCO_GL:-disabled}'
    echo 'export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/flashdexretarget-${UID}/${CONDA_DEFAULT_ENV}}"'
    echo 'export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/cuda}"'
    echo 'export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch-extensions}"'
    echo 'export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/inductor}"'
    echo 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"'
    echo 'export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$XDG_CACHE_HOME/warp}"'
    printf 'PROJECT_DIR=%q\n' "$PROJECT_DIR"
} > "$ENV_FILE"

if [[ "$PROFILE" == isaac && "$SKIP_ASSETS" == 0 ]]; then
    if [[ -n "$NATIVE_MOTION_FILE" ]]; then
        MUJOCO_GL=disabled python src/simulator/isaacsim/convert_assets.py \
            --native-motion-file "$NATIVE_MOTION_FILE"
    else
        if [[ ${#ROBOT_CONFIGS[@]} == 0 ]]; then
            ROBOT_CONFIGS=(config/envs/robot/xhand_bimanual.yaml)
        fi
        for robot_config in "${ROBOT_CONFIGS[@]}"; do
            usd=$(python - "$robot_config" <<'PY'
import sys

from simulator.isaacsim.assets import load_robot_asset

print(load_robot_asset(sys.argv[1]).usd_path)
PY
)
            if [[ ! -f "$usd" ]]; then
                MUJOCO_GL=disabled python src/simulator/isaacsim/convert_assets.py \
                    --robot-config "$robot_config"
            fi
        done
    fi
fi

# Report installed metadata without importing any engine or allocating a GPU context.
python - "$PROFILE" <<'PY'
import importlib.metadata as md
import sys

packages = ["torch", "numpy", "rsl-rl-lib", "rsl-rl-flashsac", "tensordict", "FlashDexRetarget"]
packages += ["isaacsim", "isaaclab"] if sys.argv[1] == "isaac" else ["mujoco", "mujoco-warp", "warp-lang", "newton"]
if sys.argv[1] == "newton":
    packages += ["dexforge-core", "dexforge-tracking"]
for package in packages:
    print(f"[setup] {package} {md.version(package)}")
PY
echo "[setup] done. Activation: $ENV_FILE"
echo "[setup] train: MOTION=<dataset>/motion.pt bash scripts/train_flashdexretarget.sh --profile $PROFILE [Hydra overrides]"
