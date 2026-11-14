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
action validity. Source rows come from the actual pre-advance command reference
phase; the next-target cursor is retained separately, including its clamped
final value. When the action term exposes accepted finger and FREE-root targets,
those targets, reference IDs, source support and resolved action recipe are
recorded independently of raw actions. Root wrench samples are last-substep
control values, not contact forces or interval averages. Actual object joint
Q/Qvel keeps its own support, including empty axes for a rigid object. Global
robot roots and raw actions do not acquire wrist or installed-command authority.
Five CPU cases exercise reset, two actual action/command applications, the final
endpoint and ordinary DexForge reader reopening. The original `RolloutRecord`
remains available.
Success archives retain their first-success selection meaning; they are not a
frozen final-policy evaluation or a qualification of a different physics backend.

The native Isaac interface uses `NativeSceneCfg` to select actual floating hand
articulations and one object articulation from an authored MJCF. Each hand keeps
its FREE root and declared independent actuators; an articulated object keeps
its passive joints. `convert_assets.py --native-scene-config <resolved-run.yaml>`
reads the resolved run configuration: `robot.native_scene` owns mechanics and
`sim.native_assets` owns named USD destinations. It splits those named assets
for conversion. Select
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

Use the existing retarget entry with `input=prepared_native` to pack an explicit
prepared population. Its default `<dataset>/native-reference.json` names the
dataset-relative clip directory, NPZ stems, fps and control period. This route
bypasses human-demo desk/filter processing and rigid IK, then uses the same
`build_motion_pt` and `MotionLibrary`; it requires an actual `native_reference`
and leaves native scene support with the authored mechanics. The default
`input=human_demo` route and motion.pt-only object preparation remain available.
See [prepared native input](docs/guide/kinematic_retargeting.md#prepared-native-input)
for the command, manifest and qualification boundary.

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

The native task also has config-selected direct MuJoCo Warp and Newton ports.
Select `envs/simulator@simulator=mjwarp` or `envs/simulator@simulator=newton` with the native profile;
neither launches Kit nor requires USD conversion. These ports reuse the same
manager loop, native actions, reference phase, observations, reward, evaluator,
FlashSAC and recorder. Newton delegates to the installed DexForge native engine;
direct MuJoCo Warp loads the authored MJCF itself. Newton's selected
`SolverMuJoCo` also uses MuJoCo Warp: these are different import/contact routes,
not two independent dynamics solvers. The default Newton contact route is
MuJoCo; original-triangle BVH requires an explicit `collision_generation` recipe.

All three native engine profiles declare 5 ms physics and four substeps per
20 ms control endpoint. The prepared carrier must match that actual cadence;
the command rejects a different reference grid. The stock author Isaac profile
keeps its 1/240 s physics recipe.

Native replay keeps `runner.agent.buffer_compact_obs=true`. Its future window
is reconstructed by the same observation builder used during collection, from
stored wrist pose/twist, complete named hand Q/Qvel, object-joint Q/Qvel and
the row's independent int32 motion/phase address. Sampling does not read current
live worlds. Original int64 reference clocks, support and missing tails retain
their authority. Resolved entity and joint names route Q/Qvel columns into the
native command order; their stored column order may differ. Required named
columns must be present and observe the command's actual robot entity.
Reconstruction sources must remain unclipped with nonzero scales. Network
widths still follow the actual observation space.
Choose replay capacity and placement from the resolved storage layout: storing
the raw native F1/BOX K10 future window at 50 million float16 rows alone would
require 149 GB, before the other replay fields. Do not disable compaction to
bypass a missing reconstruction route.

Backend dimensions and actuator/body/site ordering come from the compiled
scene. FREE-root wrench drives remain separate from independent finger
commands; object roots and hinges stay passive. Both ports retain achieved
state, installed targets and source reference clocks separately. Native contact
outcomes use solved, body-pair-filtered normal forces and collider separation,
with each sample's actual solver time at the beginning of the last solved
substep. These are derived target-execution facts, separate from the original
source contact availability. Achieved named-hand WORLD twists and actual endpoint
solver times retain their own fields and reset origin, independently of the
original source clock and nominal control elapsed time. Unmeasured reset contact
samples stay unsupported. The
author's Isaac net-force/SDF observation views retain their own meaning.
Configuration, software checks and an import alone do not qualify either port's
physical tracking or task success. Each actual scene/backend needs the declared
two-world native step, reset, contact, snapshot and SAC/replay continuation gates.
The author Isaac freeze remains Python 3.11. This native campaign selects an
existing Python 3.12 environment; the fork's package metadata allows both.
Install this repository with `--no-deps` in the selected environment so its
author-stack dependency list does not resolve or replace the native runtime.
`requirements/native_cu128.txt` is a separate candidate constraint set. Actual
installed engine versions and physical qualification come from the selected
runtime receipt; that file does not establish them or authorize environment
replacement. `scripts/setup.sh` continues to create the author Isaac environment.

## Setup

Before the first Isaac import or USD conversion, read and accept the
[Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/licenses-isaac-sim.html)
and [Omniverse terms](https://docs.omniverse.nvidia.com/enterprise/latest/common/NVIDIA_Omniverse_License_Agreement.html).
After accepting, export `OMNI_KIT_ACCEPT_EULA=YES` in the operator's shell;
setup and launchers inherit that choice. On Ubuntu headless hosts, install
the required GLU system library with `sudo apt-get install libglu1-mesa`.

```sh
git clone --recursive https://github.com/Holiday-Robot/FlashDexRetarget.git
cd FlashDexRetarget
bash scripts/setup.sh
```

One script builds the conda env `FlashDexRetarget` (pinned Isaac freeze, rsl_rl_flashsac), `scripts/env.sh`
and the XHAND and Sharpa Wave USDs.

Each stock robot config declares `sim.robot_asset`: the source URDF, destination
USD and any MJCF-to-USD body-name mapping. Setup and runtime read that same
binding; another robot is never inferred to be XHAND from its MJCF filename.
XHAND selects the `_spec` asset by default. Its config alone admits the optional
`FDR_ROBOT_USD` override, so an existing XHAND override cannot select an asset
for Sharpa or a new robot. Convert a registered stock asset with:

```sh
python src/simulator/isaacsim/convert_assets.py \
    --robot-config config/envs/robot/xhand_bimanual.yaml
```

The same option accepts a resolved run YAML. The separate native route keeps
`robot.native_scene` and its named `sim.native_assets` bindings; it does not use
the stock robot binding. See [asset configuration](docs/guide/flashdexretarget.md#robot-assets)
for registering another asset and the collision-visual debug variant.

> [!CAUTION]
> `setup.sh` installs every package with `--no-deps`. Add packages to this env the same way
> (`pip install --no-deps <pkg>`): a plain `pip install` pulls newer torch / warp and breaks Isaac Lab.

<details>
<summary>Troubleshooting</summary>

- Keep the accepted `OMNI_KIT_ACCEPT_EULA` in every shell that imports Isaac Sim; otherwise
  even `import isaacsim` waits for a EULA prompt and dies with `EOF when reading a line`.
- No `libEGL` (headless servers): `export MUJOCO_GL=disabled`.
- Keep Kit caches in task-owned node-local storage. Remove a cache only after
  its task has stopped and no reader uses it; shared HOME caches may serve other jobs.
- `simulation_app.close()` segfaults on isaacsim 5.1, so the runner exits with `os._exit`; a non-zero
  exit code at the very end of a finished run is expected.
- Empty URDF links can produce unresolved visual references. The converter
  blocks those empty references while preserving real visuals and the physical
  graph; generated files alone do not establish successful physics execution.
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
