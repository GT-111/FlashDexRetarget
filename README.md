# FlashDexRetarget

Official implementation of

**FlashDexRetarget: Accelerating Dexterous Manipulation Data Generation through Multi-Motion Retargeting**

[![PDF](https://img.shields.io/badge/PDF-arXiv-b31b1b.svg)](https://arxiv.org/abs/2610.01849)
[![Project Page](https://img.shields.io/badge/Project_Page-Visit-blue.svg)](https://holiday-robot.github.io/FlashDexRetarget/)

> [Kyungmin Lee](https://kyungminn.github.io/)\*<sup>1</sup>, [Sibeen Kim](https://sibisibi.github.io/)\*<sup>1</sup>, [Dongyoon Hwang](https://godnpeter.github.io/)\*<sup>1</sup>, [Yoonsang Oh](https://yoonsangoh.github.io/)\*<sup>1</sup>, [Donghu Kim](https://i-am-proto.github.io/)<sup>2</sup>, [Youngdo Lee](https://leeyngdo.github.io/)<sup>2</sup>, [I Made Aswin Nahrendra](https://anahrendra.github.io/)<sup>2</sup>, [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)<sup>1&dagger;</sup>, [Hojoon Lee](https://joonleesky.github.io/)<sup>2&dagger;</sup>
>
> **<sup>1</sup>KAIST AI, <sup>2</sup>Holiday Robotics**
>
> arXiv'2026. (\* indicates equal contribution, <sup>&dagger;</sup> indicates corresponding author)

**FlashDexRetarget** retargets large collections of human hand-object demonstrations to a dexterous robot hand
by training **one RL policy jointly across all reference motions**, instead of optimizing every demonstration separately.
<p align="center">
  <img src="docs/images/results_grid.gif" alt="Rollouts of a single FlashDexRetarget policy across many reference motions" width="100%">
</p>

## Quickstart

```sh
# 1. environment (conda env, Isaac Sim 5.1 / Isaac Lab 2.3, robot USDs)
bash scripts/setup.sh
# 2. data: MANO models in assets/mano/ (see Data), then a released dataset
#    -> <output_dataset_path>/<robot>/motion.pt (taco.sh | oakink2.sh | hot3d_1obj.sh)
bash scripts/retarget/taco.sh <raw_dataset_path> <output_dataset_path>
# 3. train (~110 GB of GPU memory; with runner.agent.buffer_device_type=cpu ~16 GB GPU + ~100 GB RAM)
MOTION=<output_dataset_path>/xhand/motion.pt bash scripts/train_flashdexretarget.sh
```

## Documentation

- [`docs/guide/flashdexretarget.md`](docs/guide/flashdexretarget.md): training options (launcher variables,
  Hydra overrides), GPU memory, stop / resume, evaluation, outputs.
- [`docs/guide/kinematic_retargeting.md`](docs/guide/kinematic_retargeting.md): data, from a released dataset to
  `motion.pt`: dataset layout, clip format, every pipeline step and its settings, `motion.pt` contents.

## DexForge integration in this fork

The `codex/dexforge-f1-box` branch starts from upstream
`6f95e133aa4b7ca6a784eac93725df3de4d41509` with FlashSAC
`9822b1b7e0b5cce45e7f79d6b3e2c169fa042faa`. The source import carries original
OakInk frame IDs, the actual ordinal resampling and later IK crop through
`motion.pt`. These additions preserve the upstream training and admission rules.

For factual native export, select
`evaluation.callbacks.rollout_record.NativeRolloutRecord` in `eval.callbacks`
and supply its `binding` JSON. The binding names the actual bilateral hand
bodies, native joint correspondence, ordered object parts, source-to-scene
transform, prepared source clocks and assistance receipt. The recorder captures
reset before subsequent achieved states, preserving measured Q/Qvel and raw
action validity. Global robot roots and raw actions do not acquire wrist or
installed-command authority. The original `RolloutRecord` remains available.
Success archives retain their first-success selection meaning; they are not a
frozen final-policy evaluation or a qualification of a different physics backend.

The native Isaac interface uses `NativeSceneCfg` to select actual floating hand
articulations and one object articulation from an authored MJCF. Each hand keeps
its FREE root and declared independent actuators; an articulated object keeps
its passive joints. `convert_assets.py --native-scene-config <resolved.yaml>`
splits those named assets for conversion. Select
`envs.actions.native.NativeResidualActionCfg` for world-frame wrist residuals
and absolute or reference-relative fingers. Action dimensions and side ownership
come from the named bindings. The root PD law recomputes mass, inertia and the
root-link wrench at every physical substep, without applying object assistance.

The example F1 action recipe has translation/rotation component scales of
0.1 m/1 rad, raw clipping at +/-1 and configured wrist filtering. It is a separate
producer condition from DexForge's main-tracker wrist recipe. Preserve the
resolved action configuration alongside exported installed targets and root-law
metadata. Four CPU contract cases cover mechanics, scene splitting, actual
action/reward callers and root wrench computation; imported USD, PhysX dynamics
and physical task execution still need qualification.

A trusted prepared clip can carry an explicit `native_reference` mapping in its
NPZ. The existing crop and `build_motion_pt` path retain its per-row native
joint Q/Qvel, named hand roots, object parts and passive joints, independent
support masks, exact prepared `time_ns` and provenance. Root/part states use
position, WXYZ rotation, world link velocity and world angular velocity.
`MotionLibrary` maps physical joint columns by the runtime joint names and
exposes these facts separately from MANO and stock scalar-wrist references.
Unknown rates have zero values and false support; cropping preserves the clock
origin and complete source frame inventory. Three CPU cases qualify this
pack/reader contract. Actual native reference generation and physical admission
remain required.

Select the existing trainer's native task with `--config_name native`. Supply
`native_recording_binding` and all six `native_tracking_tolerances` explicitly;
the latter select reference-tracking archives, independently of contact or task
success. The profile uses named hand roots and object-part SDFs, independent
finger-command BC, supported future references and native state terminations.
It disables object pinning, perturbations and adaptive assistance. BOX keeps
one free base and its passive hinge. For a real zero-joint rigid object, pass
`--overrides 'envs/native_object@native_object=rigid'` and the declared scene
bindings; no artificial hinge is added. Eight CPU cases cover normal Hydra
composition, command clocks, part/joint ordering, action and support routing,
reset, wrench cleanup, complete endpoint selection and finite-state admission.
Offline reference generation, USD/PhysX import and physical response remain
unqualified.

## Setup

```sh
git clone --recursive https://github.com/Holiday-Robot/FlashDexRetarget.git
cd FlashDexRetarget
bash scripts/setup.sh
```

One script builds the conda env `FlashDexRetarget` (pinned Isaac freeze, rsl_rl_flashsac), `scripts/env.sh`
and the XHAND and Sharpa Wave USDs.

> [!CAUTION]
> `setup.sh` installs every package with `--no-deps`. Add packages to this env the same way
> (`pip install --no-deps <pkg>`): a plain `pip install` pulls newer torch / warp and breaks Isaac Lab.

<details>
<summary>Troubleshooting</summary>

- Export `OMNI_KIT_ACCEPT_EULA=1` in every shell that imports Isaac Sim (the launchers do); otherwise
  even `import isaacsim` waits for a EULA prompt and dies with `EOF when reading a line`.
- No `libEGL` (headless servers): `export MUJOCO_GL=disabled`.
- A corrupted shader cache after an interrupted first boot: `rm -rf ~/.cache/ov/Kit`.
- `simulation_app.close()` segfaults on isaacsim 5.1, so the runner exits with `os._exit`; a non-zero
  exit code at the very end of a finished run is expected.
- `Unresolved reference prim path ... configuration/*_base.usd` warnings while the scene builds concern
  visual meshes only; physics loads.
- Out of GPU memory: `runner.agent.buffer_device_type=cpu`, or a smaller `runner.agent.buffer_max_length`
  ([GPU memory](docs/guide/flashdexretarget.md#gpu-memory)).

</details>

## Data

Motion data is not part of this repository. A dataset is one directory holding `motion.pt` and the
`objects/` it uses; one script builds it from a released TACO, OakInk2 or HOT3D (segments in which both
hands handle one object).

The import runs the MANO hand model, which is not redistributed here: register at
[mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/), download **Models & Code** (`mano_v1_2.zip`) and put
the two hand models in `assets/mano/` (or symlink an existing copy: `ln -s <dir> assets/mano`):

```sh
mkdir -p assets/mano
unzip mano_v1_2.zip
cp mano_v1_2/models/MANO_{RIGHT,LEFT}.pkl assets/mano/
```

Then:

```sh
ROBOTS="xhand sharpa" bash scripts/retarget/taco.sh <raw_dataset_path> <output_dataset_path>   # oakink2.sh | hot3d_1obj.sh
```

It writes `<output_dataset_path>/<robot>/motion.pt`; every step and its settings are in
[`docs/guide/kinematic_retargeting.md`](docs/guide/kinematic_retargeting.md).

### 50-motion benchmark

A fixed set of 50 clips (TACO 10, OakInk2 15, HOT3D 25), built from the three releases in one go:

```sh
bash scripts/retarget/50_motion_benchmark.sh <taco_raw> <oakink2_raw> <hot3d_raw> <output_dataset_path>
```

## Training

```sh
MOTION=<output_dataset_path>/xhand/motion.pt bash scripts/train_flashdexretarget.sh
# Hydra overrides go after the script, e.g. a run name and the replay buffer in host RAM
MOTION=<output_dataset_path>/xhand/motion.pt bash scripts/train_flashdexretarget.sh \
    logger.wandb.run_name=my_run runner.agent.buffer_device_type=cpu
```

`ROBOT=sharpa` (with its `sharpa/motion.pt`) trains the Sharpa Wave hand. The defaults live in `config/`
(learner: `config/runner/flashsac.yaml`). The policy is evaluated on every clip each 20M env steps, and
checkpoints and the passing rollouts go to `logs/flashsac/<run>/`. Every option, GPU memory and resuming:
[`docs/guide/flashdexretarget.md`](docs/guide/flashdexretarget.md).

## Visualization

```sh
# a retargeted dataset (<output_dataset_path>/<robot>/motion.pt, or one robot dir) in a viser web viewer
bash scripts/visualization.sh <output_dataset_path>
# plus the archived rollouts of training runs (a run dir, its success/ dir, or logs/flashsac for every run)
bash scripts/visualization.sh <output_dataset_path> logs/flashsac/<run>
```

Open the printed URL (`PORT=8080` by default) in a browser. With a saved run the simulated hands and objects are
drawn solid over the reference (translucent blue); a run trained on another `motion.pt` is tagged `[other reference]`.

## Project Structure

```
src/
  agents/       # FlashSAC agent (per-hand actor-critic, replay buffer, future trajectory encoder)
  runner/       # Off-policy training loop, checkpoints, resume
  envs/         # Observations, actions, rewards, terminations, curriculum, motion command
  evaluation/   # Per-clip evaluation and success archive
  simulator/    # Isaac Sim scene, sensors and asset conversion
  retarget/     # Human demos -> motion.pt (import, IK retargeting, packing)
config/         # Hydra configs (envs, runner, eval, logger, retarget)
scripts/        # Setup, training launcher, viewer, retarget/ (one script per dataset)
assets/         # Robot models (XHAND, Sharpa Wave); MANO models go in assets/mano/
docs/           # Project page and guides (docs/guide/)
train.py        # Training entry point
```

## Citation

```bibtex
@article{lee2026flashdexretarget,
  title   = {FlashDexRetarget: Accelerating Dexterous Manipulation Data Generation through Multi-Motion Retargeting},
  author  = {Lee, Kyungmin and Kim, Sibeen and Hwang, Dongyoon and Oh, Yoonsang and Kim, Donghu and
             Lee, Youngdo and Nahrendra, I Made Aswin and Choo, Jaegul and Lee, Hojoon},
  journal = {arXiv preprint arXiv:2610.01849},
  year    = {2026}
}
```
