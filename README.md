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
prepared-native source identity and placement from the motion library's
`native_reference_provenance`, with requested-reference authority. Human imports
retain their own capture identity and source-to-scene placement. Neither route
can substitute achieved state for reference provenance. The recorder captures
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
Explicit frozen evaluation can additionally retain every actual attempt through
`evaluate(recording_dir=..., frozen_authority=...)`, with separate replica
identity and unavailable slots. The [evaluation guide](docs/guide/flashdexretarget.md#evaluation)
describes the selected recorder and checkpoint/source authority.

The shared manager binds scene-owned tracking views after command/reference
admission and before observations or rewards are constructed. Contact rows follow
the configured sites and named bodies. Isaac's existing observations associate
WORLD net link force with the active object's SDF; they are not measured
hand–part pair forces. `tracking_contact_metadata()` records that authority,
distance sign and force availability. NOREPORT zeros remain a feature ablation,
not observed zero force. Unsigned geometry does not provide signed penetration.
Native `hand_part_contact_facts()` supplies solved pair facts; Isaac reports them
unavailable. The ordinary recorder preserves this distinction, including supported
measured-zero force, in exports. These software interfaces do not establish successful physical interaction.

The native Isaac interface uses `NativeSceneCfg` to select actual floating hand
articulations and the ordered `robot.native_scene.objects` collection from an authored MJCF. Each hand keeps
its FREE root and declared independent actuators. An articulated object keeps
its passive joints. `convert_assets.py --native-motion-file <dataset>/motion.pt`
reads the sibling `native-reference.json` and splits its named MJCF subtrees
into `<dataset>/native-usd`. Select
`envs.actions.native.NativeResidualActionCfg` for world-frame wrist residuals
and absolute or reference-relative fingers. Action dimensions and side ownership
come from the named bindings. The root PD law recomputes mass, inertia and the
root-link wrench at every physical substep, without applying object assistance.

The example F1 action recipe has translation/rotation component scales of
0.1 m/1 rad, raw clipping at +/-1 and configured wrist filtering. It is a separate
producer condition from DexForge's main-tracker wrist recipe. Preserve the
resolved action configuration alongside exported installed targets and root-law
metadata. Qualify mechanics, scene splitting, actual action/reward callers and root wrench
computation before admitting a new layout. Imported USD, PhysX dynamics and
physical task execution require their own qualification.

A trusted prepared clip can carry an explicit `native_reference` mapping in its
NPZ. The existing crop and `build_motion_pt` path retain its per-row native
joint Q/Qvel, named hand roots, object parts and passive joints, independent
support masks, exact prepared `time_ns` and provenance. Root/part states use
position, WXYZ rotation, world link velocity and world angular velocity.
`object_root_body_names` maps each actual object entity to its FREE base; the
flat named part/joint arrays retain configuration order and independent support.
Native packing does not create hand-side object aliases, asset slots or estimated
stock contact targets; native part geometry and solved contacts remain separate.
`MotionLibrary` maps physical joint columns by the runtime joint names and
exposes these facts separately from MANO and stock scalar-wrist references.
Unknown rates have zero values and false support; cropping preserves the clock
origin and complete source frame inventory. Actual native reference generation and physical admission remain required.

Use the existing retarget entry with `input=prepared_native` to pack an explicit
prepared population. Its default `<dataset>/native-reference.json` names the
dataset-relative clip directory, NPZ stems, fps and control period. This route
bypasses human-demo desk/filter processing and rigid IK, then uses the same
`build_motion_pt` and `MotionLibrary`; it requires an actual `native_reference`
and leaves native scene support with the authored mechanics. The default
`input=human_demo` route and motion.pt-only object preparation remain available.
See [prepared native input](docs/guide/kinematic_retargeting.md#prepared-native-input)
for the command, manifest and qualification boundary.

The native trainer and evaluator read `native_scene` and `recording_binding`
from the `native-reference.json` beside `commands.motion_file`. Scene and
surface mesh paths resolve relative to that manifest. A scene can contain
rigid objects and articulated objects together, with genuinely absent joints
omitted. Each hand can declare `tip_source_point_names` from the prepared
MANO reference alongside the corresponding physical `tip_site_names`. The
prepared profile binds both hands by those roles and uses the command
actuators declared by the scene. Named logical targets in the prepared
reference can fan out to several physical drives. Each drive uses its
compiled limits. Rollouts retain the logical request, installed drive
targets and measured scalar joint state as separate arrays, with solved
generalized actuator effort when the backend supplies it. Older one-to-one
references use the exact driven coordinate of each named actuator.
New native checkpoints bind observation order and the actual named action
and state columns at both resume and frozen replay.

Select `--config_name native_hoi_surface envs/robot@robot=prepared_native`
for the HOI producer with declared fingertip roles.
Its tracking tolerances control reference-tracking evaluation and the
prepared manifest retains source, scene and command correspondence.

The native task also has config-selected direct MuJoCo Warp and Newton ports.
Select `envs/simulator@simulator=mjwarp` or `envs/simulator@simulator=newton` with the native profile;
neither launches Kit nor requires USD conversion. These ports reuse the same
manager loop, native actions, reference phase, observations, reward, evaluator,
FlashSAC and recorder. Newton delegates to the installed DexForge native engine;
direct MuJoCo Warp loads the authored MJCF itself. Native scalar position drives
can retain the authored constant effort bias alongside stiffness and damping.
Isaac's implicit drives require zero constant effort bias. Newton's selected
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
widths still follow the actual observation space. Retained FP32 facts and
clock/support authority stay exact; derived geometry has a separately declared
[numerical reconstruction criterion](docs/guide/flashdexretarget.md#stop--resume).
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

Public partial reset refreshes the selected environments' source phase,
observation history, delay and episode bias. Unselected environments keep their
source and physical clocks, controller state and observation history; reset
returns a new observation packet. Full reset initializes phase zero, and each
ordinary global step advances every environment once, including autoreset rows.

Named installed mass, COM and body-frame inertia feed the shared root and
object-control owners. Isaac gravity uses its real scene-global API; native
MjWarp/Newton currently admit constant constructor gravity only. See
[installed dynamics and gravity](docs/guide/flashdexretarget.md#installed-dynamics-and-gravity)
for property frames, assistance selection and qualification limits.
Configuration, software checks and an import alone do not qualify either port's
physical tracking or task success. Each actual scene/backend needs the declared
two-world native step, reset, contact, snapshot and SAC/replay continuation gates.
The author Isaac freeze remains Python 3.11. Native profiles use a separate
Python 3.12 environment; the fork's package metadata allows both.
Install this repository with `--no-deps` in the selected environment so its
author-stack dependency list does not resolve or replace the native runtime.
`requirements/native_cu128.txt` is a separate candidate constraint set. Actual
installed engine versions and physical qualification come from the selected
runtime receipt; that file does not establish them or authorize environment
replacement. Setup selects the isolated author or native environment explicitly.
Newton's recorded backend identity includes the installed Newton, MuJoCo,
MuJoCo Warp and Warp package versions.

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

The default `--profile isaac` builds the Python 3.11 conda environment
`FlashDexRetarget`, the pinned author stack and `scripts/env.sh`. It converts the
default XHand asset. Select stock robot configurations explicitly; the script
does not maintain a second robot roster:

```sh
bash scripts/setup.sh --profile isaac \
    --robot-config config/envs/robot/sharpa_bimanual.yaml
# Repeat --robot-config for multiple stock assets, or use a prepared native dataset.
bash scripts/setup.sh --profile isaac --native-motion-file /path/to/dataset/motion.pt
```

Direct MuJoCo Warp and Newton select the separate Python 3.12 environment
`FlashDexRetargetNative` and its generated `env/native.sh`. They install the
existing `requirements/native_cu128.txt` without installing, importing or
converting Isaac assets. Newton also requires the operator's selected DexForge
checkout and installs its public core/tracking packages with `--no-deps`:

```sh
bash scripts/setup.sh --profile mjwarp
bash scripts/setup.sh --profile newton --dexforge-root /path/to/DexForge
```

The native requirements pair Newton 1.7 with MuJoCo and MuJoCo Warp 3.15
and Warp 1.19. This includes the upstream implicit free-body integration
correction for high-spin asymmetric bodies. Newton recordings retain the
installed engine versions in the existing backend identity packet.

To reproduce a qualified native dependency closure or select a built dependency
artifact, use a separately named environment. The lock contains all noneditable
third-party requirements; it is installed with `--no-deps`. Public project and
FlashSAC sources still come from this selected checkout and the explicit
`--dexforge-root`, rather than an editable workspace captured in the lock.
Each `--package-wheel` is installed as a whole package with `--no-deps`; record
its normal source/build publication alongside the profile. Default setup keeps
the existing native requirements and resolver behavior.

```sh
bash scripts/setup.sh --profile newton --env selected-native-profile \
    --dexforge-root /path/to/selected-public-source \
    --requirements-lock /path/to/qualified-third-party-lock.txt \
    --package-wheel /path/to/published-dependency.whl
```

Use the lock's owning package indexes through pip's normal index configuration.
Keep the existing environment and its cache paths unchanged while qualifying
the new profile. Runtime and checkpoint qualification remain separate from
installation. These artifact inputs apply to the native profiles.

The native profile pins WandB `0.28.0` for Python 3.12. The author Python 3.11
freeze retains WandB `0.15.12`; installing that older pin in native pulls
`pathtools`, whose build imports the removed Python `imp` module. Resolve the
native requirements normally, including WandB's dependencies; do not reuse the
author freeze or skip WandB dependency resolution.

Both the Python 3.11 Isaac profile and the separate Python 3.12 native profiles
pin NumPy `1.26.4` for the ordinary trusted packed-motion and NPZ readers.
NumPy `1.26.1` added the required
[`numpy._core` pickle stubs](https://numpy.org/doc/2.0/release/1.26.1-notes.html#numpy-core-submodule-stubs)
for arrays written with NumPy 2; `1.26.4` supports both Python versions and
satisfies [IsaacLab 2.3's `numpy<2` dependency](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/setup.py#L18).
This updates the Isaac recipe's former NumPy `1.26.0` pin. Historical environment
receipts and qualifications retain their recorded versions; they do not qualify
a changed profile. Record the selected environment's actual versions and open
the unchanged carrier with the normal `MotionLibrary` before a new SDK gate.
Preserve the existing clip bytes, clocks and provenance. No reader alias,
unpickler override or re-export is required.

For a manually managed native environment, create a fresh Python 3.12 venv.
Keep the recorded FlashSAC source in `third_party/rsl_rl_flashsac`; a normal
checkout initializes that submodule, while an admitted source publication may
already include it:

```sh
python3.12 -m venv /path/to/new-native-env
source /path/to/new-native-env/bin/activate
unset PYTHONPATH
python -m pip install 'setuptools<81' wheel
python -m pip install --prefer-binary -r requirements/native_cu128.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-deps -e third_party/rsl_rl_flashsac
python -m pip install --no-deps -e .
```

Call the ordinary Python entry with the selected native Hydra configuration;
shell wrappers require setup-generated activation files. Newton additionally
needs the selected DexForge public core/tracking packages, installed as described
by `--dexforge-root` above. Keep temporary and pip/compiler caches node-local,
record actual resolved versions, and qualify logging, engines and continuation
separately. This fresh environment does not replace an active runtime.

`--env NAME` selects a different conda environment; setup checks its Python minor
version before installation and regenerates that profile's activation file.
Newly generated activation files also check that minor version when sourced and
clear inherited `PYTHONPATH`. Keep machine overrides in the calling shell.
Cache defaults use `${TMPDIR:-/tmp}/flashdexretarget-${UID}/${CONDA_DEFAULT_ENV}`
for XDG, CUDA, Torch extensions/Inductor, Triton and Warp; explicit cache paths
and the operator's existing license acceptance are preserved.
Before the selected runtime boots, normal training/evaluation and the standalone
asset converter create nonempty requested `TMPDIR`, `XDG_CACHE_HOME`,
`CUDA_CACHE_PATH`, `TORCH_EXTENSIONS_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR` and `WARP_CACHE_PATH` directories, including missing parents.
Set these paths in the launch environment, including for manually managed
profiles; they must be writable and on the operator's selected local storage.
Unset paths retain their library defaults. Kit's configured portable root and
the run/checkpoint/replay directories keep their existing launcher and writer
ownership. Directory creation does not remove cache contents or change an
active environment.
Use a short task-owned `TMPDIR` on local storage for native GPU compilation;
NVRTC can reject a long operation-derived path even when the directory exists.

The preparation wrapper uses the same explicit `--profile isaac|mjwarp|newton`
selection. `isaac` is the default; native profiles require their generated
`env/native.sh`. Preparation selects only the environment: choose its input,
steps, robot and task through ordinary Hydra arguments. Native setup omits the
human-processing stack, so full human/MANO import and physics IK use the author
environment. See [prepared native input](docs/guide/kinematic_retargeting.md#prepared-native-input)
for packing an existing carrier. A missing selected activation is an error;
manually managed environments can call the public Python entry directly.

The native engine pins are candidate
constraints, not a claim that a new environment or scene passed physical or
learning qualification. Select a new or stopped environment; setup does not
manage running jobs or protect an active environment from package replacement.

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

The same option accepts a resolved stock run YAML. Prepared native datasets
use the `--native-motion-file` conversion described above. See
[asset configuration](docs/guide/flashdexretarget.md#robot-assets) for
registering another asset and the collision-visual debug variant.

> [!CAUTION]
> The author Isaac freeze and local Flash/SAC packages use `--no-deps`; a plain
> `pip install` in that environment can replace its Torch/Warp stack. Native setup
> resolves the separate pinned native requirements inside its own environment;
> Flash/SAC and the selected DexForge packages still use `--no-deps` so the
> author-only project metadata cannot pull Isaac into native.

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

Select another stock robot through Hydra, for example
`envs/robot@robot=sharpa_bimanual` with its `sharpa/motion.pt`.
For a prepared native dataset, choose the matching robot anatomy and backend.
Training and evaluation bind the same scene and recording facts from the
dataset manifest.

```sh
MOTION=<prepared_dataset>/motion.pt bash scripts/train_flashdexretarget.sh \
    --profile newton --config_name native_hoi_surface \
    envs/robot@robot=inspire_rh56f1_bimanual
```

Use `--profile mjwarp` for direct MuJoCo Warp. For Isaac, convert the named USD
assets first, then use `--profile isaac --config_name native_hoi_surface`.
The defaults live in `config/`
(learner: `config/runner/flashsac.yaml`). The policy is evaluated on every clip each 20M env steps, and
all outputs share the configured `run_dir` (default `logs/flashsac/<run>/`).
Set `run_dir=/absolute/node-local/path/new-run` for campaign placement; the entry
accepts only a new or empty directory. Creating `<run_dir>/stop.request` requests
a full saved stop after the current iteration, including replay warmup. A
successful FULL consumes the request and returns through normal cleanup;
explicit resume loads the parent checkpoint into a new output directory.
Every option, GPU memory and resuming:
[`docs/guide/flashdexretarget.md`](docs/guide/flashdexretarget.md).

## Visualization

```sh
# a retargeted dataset (<output_dataset_path>/<robot>/motion.pt, or one robot dir) in a viser web viewer
bash scripts/visualization.sh <output_dataset_path>
# plus the archived rollouts of training runs (a run dir, its success/ dir, or logs/flashsac for every run)
bash scripts/visualization.sh <output_dataset_path> logs/flashsac/<run>
# prepared native FREE-wrist dataset and recorded rollouts (headless validation via --check)
python src/utils/visualization.py --dataset <prepared_dataset> --saved <run_or_eval_dir> --check
```

Open the printed URL (`PORT=8080` by default) in a browser. With a saved run the simulated hands and objects are
drawn solid over the reference (translucent blue); a run trained on another `motion.pt` is tagged `[other reference]`.
For native FREE-wrist datasets, Viser uses `native-reference.json` to identify the hand and named object parts.
It renders object tracking error, per-hand contact forces and supported interval peaks of applied force.
Pass a directory containing one prepared dataset per hand to compare hands on the same clips and object layout.
For a hand with multiple prepared selections, point `--dataset` at the selection containing the desired `motion.pt`.

### Prepared native HOI recipe (`native_hoi_surface`)

`config/native_hoi_surface.yaml` combines worst-named-part surface tracking, `native_source_tip_reference_error`,
`native_source_hand_part_intent`, same-part thumb and fingertip approach and contact credit,
contact-gated rotational `xfrc` scaling (`contact_scale: 0.5`), post-decay wrench
perturbations (`require_assist_decayed: true`), and adaptive `xfrc -> 0` decay
with the hand actuators and contact sites declared in the prepared scene.
Pairing `envs/simulator@simulator=newton` with CoACD convex hulls capped at `<= 16` hulls and `<= 32` vertices/hull runs
native `SolverMuJoCo` contacts. `NativeRolloutRecord` retains post-solve WORLD
force and torque vectors and support for each substep from `mjw_data.xfrc_applied`.
Interval peaks are derived from the same vectors.
Native object disturbances use known source hand-part engagement. An object's
burst can start after its configured engagement lag and the same delay after
reset. The wrench acts on its FREE root and clears when source engagement
ends. This also covers objects manipulated on a support surface.
Each burst applies 1.5 N of force and up to 0.03 N·m of directed torque. At
each substep, the struck root's installed COM inertia and current orientation
bound the torque's angular acceleration contribution to 1000 rad/s².
The native wrist action approaches a source-engaged part within the ordinary
wrist translation budget. The approach uses the achieved fingertip surface
normal while that part remains within 20 mm and 20° of its reference pose.
It clears as soon as the source engagement, pose tolerance, or nearby
surface support fails at a control decision. The same action runs in training
and evaluation. Wrist feedforward differentiates the effective translation
after this guide and the wrist cap, on the actual control interval.
On native Newton and MuJoCo Warp, set
`++actions.source_contact_approach.pad_compression_m=<meters>` to evaluate a
bounded pad preload. The guide follows the declared carrier axes and projects
normal travel at each loaded pad through the solved point Jacobian into the
named finger command coordinates. Use the same setting for training and both
frozen evaluations. Grip adjustment acts through hand position targets and
retains the original source roles, installed actuator bounds, and perturbation.
With pad preload enabled, the pose and tip guard admits new approach guidance.
An existing guide can retain its bounded wrist offset and pad correction while
the same source-assigned part is still force-bearing in the actual solve.
Losing that measured contact clears its finger correction and releases the
wrist offset through the same filter.
Set `++actions.source_contact_approach.stationary_hold_force_n=<newtons>`
to regulate the assigned hand's normal force during a complete stationary
source task on a passive plane. The guide requires measured object-plane
support and uses the last solved hand-part force. It retreats when support
is lost or solved hand penetration exceeds 2.5 mm. The wrist approach
stays within the translation budget and advances through the normal wrist
pose and twist targets. Apply the same setting during training and both
frozen evaluations.
The HOI recipe treats a take that exceeds 5 cm of part translation or 20° of
part rotation as a training failure. The part-position reward uses the same
5 cm scale. Frozen evaluation still records the entire source take so the
failure window can be inspected.
Set `eval.interval=0` for an HOI training run whose exact-zero FULL checkpoint
will be evaluated separately in nominal and perturbed modes. Runs that begin
with zero assistance must still execute the full 15-epoch zero hold before an
eligible checkpoint is written. The saved actor optimizer must have applied
an update during that zero hold. A run that exhausts its budget during replay
warm-up saves a resumable FULL without a qualification receipt.

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
