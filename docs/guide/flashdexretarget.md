# FlashDexRetarget: training options and evaluation

Setup is in the [README](../../README.md#setup); the data pipeline (release -> `motion.pt`) is in
[`kinematic_retargeting.md`](kinematic_retargeting.md).

## Training

`scripts/train_flashdexretarget.sh` runs the existing `train.py` with the selected
runtime's activation file. `MOTION` supplies the data; robot, scene and learner
selection use ordinary Hydra overrides. Nonempty `ROBOT` is rejected with the
replacement syntax, so an old Sharpa invocation cannot silently launch XHand.

```sh
# XHAND, on a dataset prepared with scripts/retarget/<release>.sh
MOTION=<output_dataset_path>/xhand/motion.pt bash scripts/train_flashdexretarget.sh

# Sharpa Wave (clips retargeted to assets/robot/sharpa/bimanual.xml)
MOTION=<output_dataset_path>/sharpa/motion.pt bash scripts/train_flashdexretarget.sh \
    envs/robot@robot=sharpa_bimanual

# any config key, e.g. a run name, fewer envs, a smaller buffer, no wandb upload
MOTION=<output_dataset_path>/xhand/motion.pt bash scripts/train_flashdexretarget.sh \
    logger.wandb.run_name=small num_envs=1024 runner.agent.buffer_max_length=20000000 logger.wandb.mode=offline
```

## Options

### Launcher

| input | default | meaning |
|---|---|---|
| `MOTION` | – (required) | motion `.pt` to train on ([`kinematic_retargeting.md`](kinematic_retargeting.md)) |
| `--profile` | `isaac` | `isaac` activates `scripts/env.sh`; `mjwarp`/`newton` activate `env/native.sh` and select the native task and plant |
| `--config_name` | profile default | existing run config; use `--profile isaac --config_name native` for the native Isaac task |
| `envs/robot@robot` | run-config default | actual Hydra robot configuration and its asset bindings |

The run is named `flashdexretarget-<directory>-<profile>` and tagged
`<directory>`, `<profile>` in wandb, where `<directory>` is the immediate parent
of `MOTION`;
`logger.wandb.run_name=<name>` replaces it.

### Robot assets

The selected stock robot YAML owns both its authoritative `robot.xml_path` and
the Isaac import binding:

```yaml
sim:
  robot_asset:
    source_urdf_path: assets/robot/my_hand/urdf/my_hand.urdf
    usd_path: assets/robot/my_hand/urdf/converted/my_hand.usd
    body_name_map: {}  # MJCF body name -> imported USD body name; omitted names retain identity
```

Paths are relative to the repository root unless absolute. Add a mapping only
for names the source URDF actually changes. The converter, articulation view
and contact-sensor prims share this mapping; joint, actuator, mimic, inertial
and collision definitions retain their existing source owners. Required link
names come from the actual MJCF metadata and complete source URDF link roster,
including authored fixed fingertip frames, rather than a fixed finger roster.
Runtime requires this explicit binding for
the stock route. It does not guess a hand or a USD from the MJCF filename.

```sh
python src/simulator/isaacsim/convert_assets.py \
    --robot-config config/envs/robot/sharpa_bimanual.yaml
# A resolved run YAML selects the same binding.
python src/simulator/isaacsim/convert_assets.py --robot-config <resolved-run.yaml>
```

XHAND declares its trained `_spec` URDF/USD and palm renames in
`config/envs/robot/xhand_bimanual.yaml`; Sharpa declares its own asset and the
palm renames present in its source URDF. Only the XHAND config reads
`FDR_ROBOT_USD`; Sharpa and custom registrations are independent of it.
`scripts/env.sh` supplies the environment rather than a global robot asset.

A robot may also declare `collision_visual_usd_path`. Convert that explicit
destination with `--collision-visual` and select it with `FDR_COLVIS=1`.
XHAND registers this debug variant; requesting it for a registration without
one fails. Debug visuals use per-collision convex hulls, retaining the original
collision mesh paths, transforms, scales and inertials. This changes visual
representation, not the declared collider recipe. Objects keep their existing
`.isaac_usd_colvis` route. Neither converted files nor named-link validation
qualifies physical contact or task success.

Native FREE-root scenes instead use `robot.native_scene` and named
`sim.native_assets.static` / `sim.native_assets.articulations` USDs with
`--native-scene-config <resolved-run.yaml>`. Keep that per-entity route for
floating hands and passive articulated objects; the stock fixed-root URDF
binding does not replace it.

The native scene declares an ordered `objects` collection of actual FREE-base
articulations. Each owns its original passive joints and named part surfaces.
Prepared native clips bind the same entities through `object_root_body_names`;
part/joint rows remain flat, named, and independently supported. Objects are not
assigned to hand-side slots. All declared objects participate in the same coupled
plant, and contact/export metadata retains each `(entity_name, body_name)` owner.
A changed object/part layout is a new configured model interface; common code
does not establish that a trained policy generalizes across layouts or tasks.

### Native Isaac articulated assets

The native Isaac profile converts the configured MJCF subtrees through the existing `convert_assets.py --native-scene-config <resolved-native-config>` entry. It retains separate FREE hand articulations and one FREE object articulation with its original passive hinge. The scene's named body, actuator and part bindings remain the source of the dimensions and addresses.

After the normal Isaac launch, the native converter enables
`isaacsim.asset.importer.mjcf` through the public extension manager before using
the Lab converter. Extension discovery alone does not register its Kit commands;
failed activation stops conversion. This dependency belongs to native MJCF
conversion and leaves the stock URDF and direct MjWarp/Newton routes unchanged.

The currently supported scalar import chart is hinge/slide joints with zero authored joint references. Supported mimic constraints are active affine joint equalities between limited revolute joints in the same FREE articulation, with one passive follower and a nonzero driver coefficient. Nonlinear, constant-only, cross-articulation, inactive and other equality constraints are rejected before conversion. Nonzero scalar references are also rejected; the current F1/BOX recipe uses zero references. This boundary does not claim that the importer preserves a nonzero-reference joint frame.

For a supported equality `q_f - q_f0 = p0 + p1 * (q_d - q_d0)`, the USD mimic equation is `q_f + gearing * q_d + offset = 0`: `gearing = -p1`, and `offset` is the negative affine offset in degrees. Source limits are retained instead of expanded to accommodate the constraint. Mimic followers have zero own position-drive stiffness/damping. Their original passive damping is authored through Physics107.3's codeless `PhysxJointAxisAPI:angular`, applied with `Usd.Prim.ApplyAPI(JOINT_AXIS_API, "angular")`, as the angular viscous coefficient in torque-seconds per degree (`D * pi / 180`). Read-only admission requires that API instance as well as the coefficient; a bare attribute is not sufficient. Runtime joint-property installation preserves this separation from drive damping and Coulomb friction. This is an explicit PhysX realization; it is not a claim of numerical equivalence to MuJoCo's soft equality solver.

Inline native MJCF meshes are materialized with the actual compiled triangle faces and their compiled geom chart. This retains the authored convex envelope, local placement and scale without applying mesh recentering twice. It does not convert the original object task surface into a collision mesh or replace the original part/SDF surface with the compiled hull. The object remains one articulation, with its source passive hinge, gravity and authored support; no object motion assistance is added.

Each split MJCF has an empty meshdir and mesh references relative to that
generated XML. This preserves original file-backed assets and compiled inline
hulls while matching the pinned Isaac5.1 importer path rules. Source mesh files
are not renamed or modified. The importer itself may create a temporary sibling
directory beside a resolved mesh; source file preservation does not imply
readonly-directory support. Successful conversion and installed constraint
response require their separate runtime qualification.

Native conversion completes importer-authored articulation APIs before strict
admission. Isaac5.1 can mark a disconnected, non-rigid world-geometry scaffold
as an articulation root. Completion removes only its articulation APIs when it
has no rigid-body/joint descendants or joint-endpoint involvement; it preserves
the prim, its other attributes and referenced source layers. Actual mechanical
roots remain subject to the one-declared-FREE-root requirement. Read-only
admission of an external USD does not perform this normalization.

Converted and externally supplied native USDs pass the same read-only named-body/joint, FREE-root, limit, mimic instance/reference-axis and follower-drive admission before the first physics parse. An absent own DriveAPI adds no position-drive authority; present passive drives must have zero stiffness, and present mimic drives must have zero stiffness and damping. The pinned normal BOX import itself authors its source zero-stiffness drive. USD admission and software forwarding tests are separate from actual installed property/constraint response, which needs a pinned-Isaac F1/BOX qualification; stock XHand SAC results do not qualify these native assets. Isaac net-link force and SDF proximity remain distinct from native solved hand-part pair forces.

Native joint defaults enter IsaacLab configuration as Python scalar values.
Runtime damping, armature and friction tensors use the dtype of their actual
SDK destination buffers. Shared compiled metadata retains its original numeric
precision; this boundary does not change Torch's global default dtype.

Native contact sensors resolve their configured body names against the actual
spawned asset's rigid-body prims before reset. Their paths retain the asset's
environment regex and the imported body's relative hierarchy. An importer
wrapper with the same name as a body is not the physical contact address.
Contact reporter activation remains enabled; resolving an address does not
qualify installed reporter behavior or hand–part force measurement.

Stock USD collision overrides are installed after the scene owner makes nested
references editable, before material binding and the first physics reset.
Robot and parked-object edits precede environment cloning; per-world objects,
including ghost spawns, receive the same selected configuration on their actual
world path. Procedural support disks keep their direct installation. A `None`
configuration or field preserves its existing no-override meaning.

The SDK's spawn-time traversal skips instanced collision subtrees. Material
binding alone cannot install properties that were skipped earlier. Applying the
configured collision enable/contact/rest offsets after deinstance changes the
realized physical recipe where those settings were previously ignored. Earlier
learner/replay results require separate qualification on the corrected source.
The native USD path supplies no such override and keeps its authored authority;
this stock correction does not alter XML geometry or native pair-contact facts.

Body correspondences use the existing `[robot, MANO role]` pairs, with the robot member now an explicit mapping from each available hand side to its complete body name. For example, `[{left: armA/segment, right: receiverB/finger_stage}, index_proximal]` preserves full namespaces and supports asymmetric names. Declare only actual physical correspondences; missing anatomy must stay absent. The former suffix-string schema is no longer accepted. The command resolves those names against the actual entity once, then uses the admitted full names for reference FK.

All shipped XHand, Sharpa and F1 registrations retain their prior body names, MANO roles and ordering. Stock and native commands use the same nonempty declared level groups for geometry access. Stock metrics follow those groups; native metrics retain their separate registered-root and object-part meanings. This does not add native anatomical metric outputs. F1 retains its seven sparse all-body correspondences and no level groups. Reward selection, source/support authority, controllers, clocks and physics are unchanged.

Native references may carry source geometry availability as the paired fields
`source_point_names` and `source_point_position_support`, each keyed by the
actual hand sides. Names address the existing clip's MANO world positions;
boolean support has shape `[frames, points]` in that named order, independently
of robot joint and root support. Cropping retains the corresponding mask rows,
and the motion library aligns masks with its actual MANO point order. An absent
pair means unavailable source support; a library cannot mix clips with and
without this modality. These facts do not declare contact or a successful
interaction, and do not change the default reward or observation selection.

When the native command provides source-tip geometry facts, current-attempt
recordings retain the consumed reference IDs, exact nanosecond clock, independent
source/achieved position masks, and named-part distance-query masks. Recorded
positions use scene-local coordinates; distance values and support keep their
producer meaning. Flat named-point metadata preserves unilateral hands and
unequal point counts. Missing facts produce no geometry group. These arrays
describe geometry queries, not measured contact or task success.

The optional native source-geometry observation `native_source_tip_reference_error` exposes current supported observed source-tip errors in the actual physical wrist axes, followed by their masks. Select it for both actor and critic through the existing observation config. Point counts and source/site names come from the admitted library and physical registrations; registered target-root support or target FK cannot supply source-point support. Old carriers without the named independent mask cannot enable these consumers.

The optional native observation `native_wrist_control_state` exposes the prior
filtered wrist action coordinates: translation XYZ then rotation-vector XYZ
for each hand in the action's configured `hand_sides` order, followed by one
`filter_pending` bit per world. These are processed action-chart values, before
the physical residual scales, not achieved poses or source facts. Pending reset
overwrites the old filter on the next action, so its effective carry is zero.
Otherwise the carry remains available even when the current source root is
unsupported. Select this term in both `obs.obs_dict.actor_obs` and
`obs.obs_dict.critic_obs` through the ordinary observation configuration; it is
absent from the default recipe. The action's public `wrist_control_state`
accessor supplies the same detached coordinates, pending state and hand order.
This changes the selected actor/critic input graph and needs an explicit new
condition initializer; a strict old FULL cannot silently acquire the inputs.
It does not change filtering, installed targets, physics, reward or assistance.

The optional native rewards `native_source_tip_position` (`reward_scales.<name>: scale`, inverse meters) and `native_source_tip_object_bound` (`reward_params.<name>.sigma_m`, meters) adapt the official source-tip position and object-proximity mechanisms. They use the action's consumed reference IDs/steps at the achieved endpoint, while the observation uses the current decision cursor. They are absent from the default recipe. Split-side values share the full supported denominator and conserve the existing two half-weight terms, including unilateral/no-support cases. This is environmental scalar reward, not replay-action actor BC or a new gradient into the reference producer.

The object-bound adaptation queries desired and live named-part SDFs and takes both minima over the same compatible supported part subset. A point with no common admitted part has unavailable reward mass. Domain masks certify only finite samples inside the configured baked grid. They do not certify full source-mesh coverage, signed penetration, force/contact, source engagement or task success; mixed declared distance-sign unions reject. This articulated-SDF adaptation is not a claim of numerical equivalence to the stock visual-vertex reference distance. The ordinary recorder may retain the same detached endpoint operands through `source_tip_geometry_facts`, independently of success.

Adding the observation changes the configured actor/critic graph. Existing strict FULL state cannot silently resume across that change; any declared initializer/new-condition lifecycle belongs to the learning owner. Default observations, reward weights, ABS action semantics, physics, assistance and task criteria are unchanged by merely publishing these methods.

### Native articulated training assistance

The `native` recipe uses the authored scene's gravity and passive support with no object assistance. Select `--config_name native_assisted` to train with explicit per-part object PD forces and the existing adaptive xfrc curriculum. The robot, physical part names, command actuators and runtime remain selected through the same native configuration. The object retains one FREE base and its passive articulation; training assistance is not an object motor, weld or playback.

The configured PD acts at actual body COMs in WORLD axes. Its linear error uses body-origin pose/velocity, and its rotational inertia is the installed COM inertia expressed in the body frame. Unsupported reference poses receive no force; unsupported velocities use zero feedforward with false independent rate support. Native assistance currently admits the explicit PD/adaptive recipe, not dynamic gravity, contact-dependent scaling, perturbation or implicit damping.

Object PD is recomputed before every configured physics substep. This is a changed force-update recipe for assisted runs; checkpoints and replay qualified with an earlier update schedule retain their own producer identity and should not be described as numerically equivalent. Changing assistance creates a new physical learning condition. The existing `agent_load_path` can warm compatible actor/critic/optimizer state with `buffer_load_path=null`; strict FULL continuation requires the same curriculum and numerical owners.

Ordinary evaluation disables pinning, removes the adaptive controller and clears external object requests. The native recorder reports last-substep **submitted** body-COM force/torque separately from actual pair contacts; it does not measure total delivered force or certify physical task success. Passive tables and gravity remain active. Use the frozen task/family evaluator for articulated opening/closing, contact coverage and complete-horizon outcomes.

### Hydra overrides

The defaults below come from `config/` (`envs/`, `runner/`, `eval/`, `logger/`); arguments to the launcher
are applied last.

**Run**

| key | default | meaning |
|---|---|---|
| `num_envs` | `2048` | parallel environments |
| `seed` | `42` | random seed |
| `device` | `cuda:0` | simulation and learner device |
| `sim.kit_args` | `""` | extra arguments forwarded to the existing Isaac AppLauncher, including the graphics-device selection |
| `episode_length_s` | `20.0` | episode time limit; an episode also ends with its clip |

With a CUDA visibility mask, `device` addresses the visible CUDA devices while
Kit's renderer ordinal can refer to the full physical GPU table. The resource
owner supplies that mapping through `sim.kit_args`, for example
`'sim.kit_args="--/renderer/activeGpu=1"'` when the selected graphics device is
ordinal 1 and its masked physics device is `cuda:0`. An empty string preserves
AppLauncher's default mapping. A CUDA mask alone does not isolate graphics
contexts; include any auxiliary context in the owning allocation's budget.

Robot conversion removes importer-created visual references for URDF links
without visual geometry from their introducing layers inside the generated
USD directory. An empty reference list on an outer assembly cannot remove
those nested arcs. Keep the generated configuration layers with the robot USD;
real visuals, source meshes, colliders, inertials and joints are preserved.

**Learner** (`config/runner/flashsac.yaml`)

| key | default | meaning |
|---|---|---|
| `gamma` / `n_step` | `0.98` / `5` | discount and n-step return |
| `updates_per_interaction_step` | `2` | gradient updates per env step |
| `per_hand_actor_critic` | `true` | one actor and one critic per hand, each on its own hand's reward; `false` is a single actor-critic on the summed reward |
| `runner.agent.buffer_max_length` | `50000000` | replay buffer rows (1.76 GB per million rows) |
| `runner.agent.buffer_device_type` | `cuda` | `cpu` keeps the replay buffer in pinned host RAM ([GPU memory](#gpu-memory)) |
| `runner.agent.sample_batch_size` | `4096` | batch size per update |
| `runner.agent.critic_hidden_dim` / `critic_num_blocks` | `1024` / `2` | critic width / depth |
| `runner.agent.actor_hidden_dim` / `actor_num_blocks` | `128` / `2` | actor width / depth |
| `runner.agent.learning_rate_init` / `_peak` / `_end` | `3e-4` / `3e-4` / `1.5e-4` | learning-rate schedule |
| `runner.agent.traj_encoder.enable` | `true` | Conv1D encoder over the next 10 reference frames |
| `runner.agent.use_compile` / `use_amp` | `true` / `true` | `torch.compile` / mixed precision |

`commands.name` selects the registered command. The runner passes that actual
command to the flat observation/reward-head wrapper, so hand ownership follows
its declared sides. Task-reward early reset uses the same injected command name.
The default `motion` and a configured rename share the same rewards and thresholds.

Reward heads use each `RewardTermCfg.params.side`, not a letter inferred from
its name; `runner.agent.reward_heads.overrides` remains an explicit override.
Terms without a side are shared equally, so head rewards sum to the manager's
global reward. `rewards.split_sides` expands supported side terms through the
existing builder; `rewards.shared_terms` keeps declared scalar terms whole.
The native recipe assigns root tracking, supported independent-finger BC and
action cost to their actual hand. It retains the original global support/action
normalization, including missing support and unequal action widths. Passive
object part/articulation tracking and global termination remain single shared
terms; they are divided among heads once. Native currently has no contact reward:
contact observations/measurement do not create contact credit or task success.
This changes reward-head targets and requires its own declared learning condition;
an unchanged FULL continuation must retain its original reward configuration.

Native scalar coordinates retain the compiled MJCF joint type in actual named entity order: hinge positions/rates are radians and radians per second; slide positions/rates are metres and metres per second. Articulation reward `scale_rad`, angular velocity limit `joint_rad_per_s`, and tracking metrics/thresholds ending in `_rad` keep their hinge meaning and values. An object with passive slides must also supply finite `rewards.reward_params.native_object_articulation.scale_m` (positive) and `native_tracking_tolerances.joint_position_error_m` (nonnegative). Selected slide commands require finite, nonnegative `native_tracking_tolerances.command_position_error_m`. Any hand or object scalar slide also requires finite, nonnegative `terminations.native_velocity_diverged.params.joint_m_per_s`. Unused metre settings stay unset; passive and command error reports separate angular and linear columns with independent support.

Recordings retain raw coordinate values and order. `object_joint_metadata_json` declares passive types and units; `installed_control_metadata_json.finger_joint_coordinates` declares the selected command types and units. Its position units also apply to the existing `installed_finger_target_rad` array and `action_recipe.finger_effective_limits_rad` values; those field names do not convert slide coordinates to radians. This is reference tracking, not an articulated-task or contact success criterion.

**Run length and checkpoints** (`config/runner/base.yaml`)

| key | default | meaning |
|---|---|---|
| `num_env_steps` | `500000000` | env-step budget |
| `save_checkpoint_per_env_step` | `20000000` | checkpoint stride in env steps |
| `save_replay_buffer` | `false` | also store the replay buffer in each `step<N>/` checkpoint; set `true` to resume from one |
| `runner.keep_last_checkpoint_only` | `true` | keep only the newest `step<N>/` directory |

**Evaluation** (`config/eval/motion_tracking.yaml`)

| key | default | meaning |
|---|---|---|
| `eval.interval` | `20000000` | env steps between evaluations |
| `eval.success_criterion` | `obj` | `obj` (object errors only) or `full` (adds the finger errors) for the success archive |
| `eval.stop_when_all_solved` | `true` | end the run once every clip has passed |
| `eval.save_success_data` | `true` | archive the first passing rollout of each clip |

**Environment**

| key | default | meaning |
|---|---|---|
| `rewards.reward_weights.<term>` | see `config/envs/rewards/flashdexretarget.yaml` | per-term reward weight; `0.0` turns a term off |
| `commands.sampling.mode` | `uniform` | episode start: `uniform` over clip frames, or `start` (first frame) |
| `commands.object.park_inactive_on_ground` | stock `true`, native `false` | leave inactive stock objects resting on the ground, or repeat their parking; native has one declared object |

Parking is an explicit task recipe; it replaces `FDR_SLEEP_PARK` and makes no
claim about a backend exposing MuJoCo's sleep flag. Scene-owned contact views
bind after command admission and use the robot's configured site/body order.
Isaac net-link force plus SDF association remains distinct from native solved
hand–part force. `tracking_contact_metadata()` declares this authority and
availability; NOREPORT feature zeros are not measurements. Unsigned surfaces
reject signed-penetration requests. `NativeRolloutRecord` exports optional
`native_contact_association_metadata_json` without turning unavailable pair
forces into measured zero or original source contact labels.

Native execution exposes two contact readouts. `hand_part_contact_facts()`
retains the last solved physics sample and its substep-begin clock.
`hand_part_contact_interval_facts()` retains every configured substep from the
current manager control interval, with solver begin/end clocks, independent
recording/measurement/clock support and named hand-subtree × object-part
addresses. These discrete solved samples do not certify continuous force.

Direct MjWarp and Newton expose `physics_execution_clock()`: int64 counts of
accepted backend advances. MjWarp checks the installed fixed timestep against
its actual device cast; Newton checks the public compiled native timestep used
by its owning backend and declares its own `accepted_newton_fixed_step_clock_v1`
schema without claiming MjWarp's device-cast witness. Only an
actual advance increments the count; forward and authored state installation do
not. Selected reset clears only selected counts, and same-plant snapshot/restore
retains them. Isaac leaves this clock unavailable.

Direct MjWarp synchronously checks the SDK's per-world `OverflowType` bits after
each solver step, before accepting the advance or publishing solved contacts.
A nonzero bit raises an ordinary `RuntimeError` with a JSON witness containing
raw bits, SDK flag names, affected worlds, attempted substeps, configured and
resolved capacities, and actual contact/constraint/solver counts. EPA and solver
iteration flags are not mislabeled as exhausted contact capacity. The failed
attempt may have changed raw solver state/time; it is not rolled back or added
to the accepted clock, and its control interval remains incomplete. Preserve
the failure before selecting a physics change; no bit is ignored or retried.

The recorder preserves `native_execution_steps` and matching per-substep count
boundaries beside unchanged `native_physics_time_s` and raw contact clocks.
Consumers may form a configured schedule by rounding count differences times
the configured timestep once to nanoseconds, including steps such as 1/240s.
This schedule is distinct from the accumulated floating solver counter and from
an exact sum of binary device dt. Missing advances remain unsupported; recorder
row numbers alone cannot supply execution evidence.

Isaac's scene clock reads `SimulationContext.current_time` as FP64 values
repeated across clone worlds. This is the SDK's global physics-step callback
time; partial entity resets leave it unchanged. Each factual tape retains its
actual reset epoch. It is separate from source/reference time, nominal control
elapsed time and an accepted-step ledger. Exact hand–part pair forces and their
intervals remain unavailable from Isaac net-link/SDF views.

Native factual recording follows the hand sides admitted by the motion library
and recording binding. `native_hand_sides` labels the hand axis of
`native_hand_pose` and `native_hand_twist_w`. Bilateral recordings retain
left-then-right order; unilateral recordings contain only the available side.
Named contact rows use that same side order and the configured object-part
order. Installed-control arrays retain their independent `hand_sides` metadata.
Downstream task readers keep their declared hand requirements; the bimanual BOX
adapter still requires both hands.

The readouts borrow stopped views of fixed storage until the next control,
state change or reset. Capacity comes from `decimation`. Reset or authored
state installation discards affected rows; physical restore preserves its
last-solve facts but does not invent an executed control interval. Starting a
control discards its old interval before action or physics work, so an
interrupted control cannot reuse earlier completed evidence. Unknown force
remains distinct from measured zero contact.

`NativeRolloutRecord` copies these samples into `hand_part_interval_*` arrays
with explicit metadata. Existing last-solve fields retain their meaning.
The native measurement callback reduces supported sample counts, duration
quadrature and force-duration quadrature into the normal current-policy
per-motion JSON. Its thresholds are configured in `eval/native.yaml`; it
does not certify source-required engagement, a BOX phase or task success.
The device tape is bounded by worlds, configured substeps and actual pairs;
the optional full-trajectory CPU recorder still needs its own memory budget.

Isaac net-link forces and SDF association leave exact pair-force intervals
unavailable. Derived target-force evidence does not replace original
ARCTIC/OakInk source-contact support. Source roles, missing spans, phase
durations and full-horizon outcomes retain their separate source/task authority.

### Native wrist target twists

`actions.wrist_target_twist` selects the velocity convention for the existing
WORLD-left wrist correction. The default `source_only` retains supported source
linear velocity and transports supported source angular velocity by the current
correction. Pose targets, action filtering, finger commands and root gains are
unchanged.

`filtered_correction_feedforward` also adds the change of the filtered translation
and the WORLD rotation increment between consecutive filtered corrections, divided
by the actual control period. The first packet after a reset contributes zero
correction velocity. The estimate is computed once per control and held through
its physics substeps; a held raw action can still move the EMA. Missing source
pose admits no correction effort, while filter carry continues independently.
Source linear/angular support remains the support of the source component, not
an assertion that the correction velocity was measured. Recorded `root_state_w`
contains the actual total target twist and `action_recipe` identifies its mode.
An exactly static correction contributes exactly zero rate while retaining its
local derivative. The vendored manager quaternion exp/log use their small-angle formulas
at zero without evaluating an undefined division in an unused branch.

This is a sampled feedforward convention, not the instantaneous derivative of a
zero-order-held target. Finite rotations do not generally commute. It does not
change the control law or establish physical tracking improvement. Keep the mode
in the resolved condition; changing it is not an unchanged FULL continuation.
Typed action contexts include the held correction twist and reset validity.
Ordinary learner FULL restore retains its existing fresh-physical-episode reset.

### Initial native control contexts

`ManagerBasedRlEnv.capture_initial_control_context()` captures a committed
all-world reset boundary before the first control interval.
`restore_initial_control_context(context)` restores that context in the same
environment instance and configuration without resetting, resampling the source
or advancing physics. The context includes the NativePlant integration and
contact state, scene targets, pending native state installation, action LPF and
three-action history, installed goals and support, current command cursor and
consumed reference phase, manager buffers/counters and global RNG state.
Restoration invalidates cached physical/reference readouts; it does not reinstall
a target over the restored plant control.

This interface requires native action and command owners with paused assistance
curriculum and inactive manager metrics/recorders. Nonempty observation history
or delay, stateful observation/reward/reset/termination terms, running step or
interval events, and non-NativePlant backends reject explicitly. It provides no
arbitrary mid-interval, cross-instance or Isaac state restoration. Static assets,
model weights and configuration stay with their existing owners.

The manager's action space contains shape metadata, not a sampler. Its context
captures global Python, NumPy and Torch generators through
`capture_rng_state(None)`. A caller that owns a real Gym action-space sampler
must capture it separately with `capture_rng_state(action_space)` and restore it
last through `restore_rng_state`; the supplied-sampler helper behavior is
unchanged. Evaluation callbacks and recording remain caller-owned. An exact
repeated action sequence is the required physical repeatability check before
interpreting a declared control intervention; the software context contract
alone establishes no task success.

### Optional source hand/part intent

An explicit `native_reference.source_hand_part_intent` carries desired source
engagement independently of achieved target contacts. It names `hand_sides` and
`part_body_names` drawn from the actual native reference; neither order nor count
is fixed. A source may declare only a subset. The remaining hand/part pairs stay
unknown. The carrier contains:

- `query_time_ns`: one original-source-clock query for each prepared native row.
  Its offsets must match `native_reference.time_ns`; cropping retains its phase.
- `capture_time_ns`, `available_time_ns`, `valid_until_time_ns`,
  `source_frame_index`, `canonical_frame_index`: chronological int64 source-row
  vectors, independent of the prepared query count.
- `active`, `known`: boolean `[source row, declared hand, declared part]` masks.
  An active value requires independent knowledge support.
- `authority` and `provenance`: the source method and original resource identity.
  Geometric intent must not be described as measured force or task success.

The reader selects the last source row available at each exact int64 query and
admits it only within its declared clock cell. Delayed, missing and expired rows
cannot become active through interpolation or endpoint hold. Native NPZ, normal
crop, packed PT and `MotionLibrary` preserve this carrier. Libraries may contain
clips without it; their pairs remain unknown, and a completely absent modality
remains `None`.

Select `native_source_hand_part_intent` through the existing actor/critic
`obs_dict` to append active values and knowledge masks in actual hand/part order.
Its width is derived from those names. Selecting it without any declared source
carrier fails explicitly. This changes the observation graph and requires the
ordinary explicit actor initialization and replay/continuation contract; an old
FULL must not silently acquire the new inputs. Default observations and rewards
are unchanged.

`source_hand_part_intent_facts()` and `NativeRolloutRecord` use the installed
packet's motion/step, independently of command-cursor advance. Reset rows retain
their source intent with false execution support. The recorder exports exact
integer clocks, source addresses, active/known masks and original provenance
under `source_hand_part_intent_*`. These arrays are desired source facts, not
target force measurements. Source-specific adapters still own qualification and
construction; this port does not add BOX phase extrema, infer new OakInk labels,
install a contact reward or establish task/generalization success.

### Optional source-intent interaction credit

The existing reward builder can select `native_source_hand_part_proximity` and
`native_source_hand_part_contact` through ordinary `reward_weights` and
`reward_params`. Native defaults remain unchanged. These terms use the
**installed** source-intent packet; they do not require adding the optional
intent observation or changing the actor/critic graph.

For each source-known required hand, proximity takes the closest supported
achieved site to any of that hand's active candidate parts and returns
`exp(-abs(surface_distance_m) / distance_scale_m)`. It retains the baked grid,
pose and point-query support; it is geometric approach credit, not measured
contact, penetration clearance or a guarantee of mesh coverage. Existing
source-tip position and object-bound terms retain their separate meanings.

Contact reads the completed native control interval, matched to the current
solver endpoint. For each actual named hand/body and active candidate part,
positive-force samples contribute `min(normal_force_n / force_scale_n, 1)` only
when an actual force-bearing contact has supported separation no greater than
`separation_bound_m`. The best candidate part is used for each hand/substep;
substeps are weighted by their supported solver durations. Extra force above
the configured physical scale earns no extra credit. Named addresses, not part
ordinals or fixed asset names, bind the measurements.

Both terms divide by the same source-known required-hand count, preserving the
existing split-side reward mass. Missing/expired source cells, reset rows,
missing site or pair measurements, incomplete/stale control intervals and
unavailable pair-force backends contribute no credit. Factual source knowledge,
query support and measured absence remain separate in the existing intent,
geometry and contact recorder arrays; zero reward does not turn an unavailable
sample into measured zero force. This endpoint-conditioned training credit is
not the task layer's source-duration contact score or a task-success gate.

A selected recipe supplies the positive `distance_scale_m` and `force_scale_n`,
nonnegative `separation_bound_m`, and explicit weights. For example, a declared
physical force scale is a reward saturation scale, not an author-paper result
or proof that larger force is productive. Source adapters must provide original
int64 capture/availability/cell-end clocks, masks and source provenance through
the ordinary native carrier. Proximity-derived source intent must retain that
label. The fork does not infer source force from achieved contact.

Changed credit also changes replay reward targets. Ordinary compatible agent
state can be warm initialized with an explicitly fresh replay/collector; old
stored rewards must not be relabeled as if they contained the new credit.
Training assistance/curriculum and evaluation's zero-assistance selection stay
explicit. These methods establish no source feasibility or successful teacher.

### Installed dynamics and gravity

The shared MJCF metadata loader requires a built-in scalar actuator roster:
fixed gain, no or affine bias, and no plugin force implementation. Scalar
position actuators must implement `force = kp * (target - q) - kv * qvel`, with
finite positive `kp`, nonnegative `kv`, unit scalar gear, no activation filter
and no control delay. Equivalent `<general>` servos and `dampratio`-authored
damping are supported. FREE-root WORLD-wrench motors keep their separate
control contract. An unsupported actuator chart fails at this common input
boundary before it can be mistaken for a native position request.

`entity.body_inertial_properties()` exposes named, stopped properties with an
explicit world axis: mass `[world,body]`, body-local COM `[world,body,3]`, and
COM inertia expressed in the body frame `[world,body,3,3]`. Isaac reads the
installed PhysX views after reset and any selected property installation;
it does not reconstruct execution inertia from a source mesh or USD estimate.
PhysX's COM quaternion does not rotate this body-frame tensor a second time.
The fixed native recipes use the actual compiled named body rows, shared
across worlds. These snapshots do not add mid-run inertia randomization.

Root control, assistance, implicit damping and perturbations consume these
properties. Object selection follows the same world, active slot and shared
alias as achieved state. The optional frequency-based assistance curriculum
uses the mean selected object mass across configured sides in each world;
scalar logging means do not replace those execution operands. Native BOX
remains unassisted. In-place model swapping requires an executing model-field
write route; an unsupported engine rejects it before harvesting swap data.

Isaac's gravity curriculum calls the actual PhysX global setter, reads its
vector back and refreshes robot gravity compensation. Resetting a subset does
not create per-world gravity. Scene reset clears SDK state before adapters
restore persistent gravity-compensation requests, so selected rows retain
those requests through the ordinary next write. Native MjWarp/Newton keep constructor gravity;
a ramp or different requested vector is unsupported. Editing CPU metadata
does not update executing dynamics. Actual SDK forwarding and finite gravity
response are separate qualification gates from CPU adapter tests.

All prepared-native profiles, including native Isaac, use the same static,
unassisted task admission. A nonempty curriculum is rejected before engine/SDK
scene construction; the stock Isaac task retains its real global-gravity
curriculum. Each hand with nonzero root gravity compensation must declare the
same WORLD vector as the compiled MJCF constructor. A mismatched vector is
rejected; a zero-compensation hand does not consume its unused vector. This is
static constructor admission, without adding dynamic native gravity support.

The existing `FDR_PIN_AT_COM=1` option applies a global wrench at the installed
COM transformed by the achieved world pose. Local wrench calls use the
body-local point. The default application request is unchanged.

**Viewer, logging and resume**

| key | default | meaning |
|---|---|---|
| `headless` | `true` | `false` opens the Isaac Sim viewport |
| `logger.wandb.project_name` / `group_name` / `run_name` | `FlashDexRetarget` / `flashdexretarget` / see above | wandb naming; the run name is also the log directory name |
| `logger.wandb.mode` | `online` | `offline` or `disabled` to run without uploading |
| `logger.wandb.entity` | `null` | wandb entity (`null`: `$WANDB_ENTITY` or your default) |
| `agent_load_path` / `buffer_load_path` | `null` | checkpoint directory to load networks / replay buffer from |
| `learning_state_load_path` | `null` | explicit RNG/curriculum/progress source, independent of replay; omitted FULL resume uses `buffer_load_path` |
| `step_offset` | `0` | env step the run starts from (set on resume) |
| `logger.wandb.resume` / `resumeid` | `false` / `null` | continue an existing wandb run |

`log_metrics` sends the original complete scalar mapping to W&B and immediately
prints that same mapping to stdout without selecting a prefix or rounding values.
This preserves already returned actor/critic/temperature losses alongside episode
metrics in an archived stdout log. Logging cadence and the existing `AverageMeter`
means stay unchanged. The logger computes no additional metrics; gradient norms
are not returned by the current FlashSAC update and are therefore not a logged
series.

The off-policy runner also reports `time/collect_host_wall_s`,
`time/sac_update_host_wall_s` and `time/sac_update_calls_per_collection` through
that same meter and logging cadence. They are means per collected iteration
since the last log, including zero update time/calls during replay warmup.
Collection includes acting, environment stepping and replay ingestion; updates
include sampling, optimization and returned metrics. These are host-call wall
durations with any natural transfer or metric-read waits, not solver-only or
CUDA kernel timings. Update calls do not establish applied optimizer visits.
The metrics add no CUDA synchronization. The optional `FDR_ITER_TIMING_FILE`
CSV retains its existing synchronized combined-iteration timing behavior.

`python train.py --overrides commands.motion_file=<pt> --overrides <key>=<value> ...` runs the same
config without the launcher (XHAND; `--gui` opens the viewport).

## GPU memory

The replay buffer dominates: 50M rows take 88 GB.

| | GPU memory | host RAM |
|---|---|---|
| default (`runner.agent.buffer_device_type=cuda`) | ~110 GB | ~16 GB |
| `runner.agent.buffer_device_type=cpu` | ~16 GB | ~100 GB (88 GB of it the pinned buffer) |


On `cpu` the buffer stays in pinned host RAM and the GPU reads and writes it directly (no CPU-side
gather), so a GPU with less memory can run the same config. The buffer scales with
`runner.agent.buffer_max_length`, e.g. `runner.agent.buffer_max_length=20000000` for 35 GB.

## Stop / resume

Set `run_dir` to one new output directory. The entry resolves it once before
starting the selected runtime and writes `config.yaml` there. An existing empty
directory is accepted; nonempty directories, including stale stop requests, are
preserved and rejected. Keep launch/configuration evidence outside this directory
until the entry has admitted it.

Create `<run_dir>/stop.request` to request a saved stop. The loop checks after
each complete collection/update, including replay warmup, and around optional
evaluation/checkpoint work. It writes a full `stop<N>.tmp`, renames it to
`stop<N>/`, then moves the request into that committed directory and returns
through the entry's environment/logger/runtime cleanup. A failed FULL leaves the
request in place and propagates failure. Confirm the `graceful stop` message and
complete `stop<N>/` before resuming. SIGUSR1 is no longer a stop transport: native
compiler code can replace its handler after Python registration.

```sh
FDR_RUN_DIR=/absolute/node-local/path/new-run
MOTION=<output_dataset_path>/xhand/motion.pt bash scripts/train_flashdexretarget.sh \
    run_dir="$FDR_RUN_DIR"
# After startup, in a shell with FDR_RUN_DIR set to that same path:
touch "$FDR_RUN_DIR/stop.request"
```

A full learning checkpoint contains networks and target critics, Adam and
LR-scheduler states, the AMP scaler, reward-normalizer statistics,
`replay_buffer.pt`, `learning_state.pt` and `resume_state.json`.
A periodic `step<N>/` includes replay and learning state only with `save_replay_buffer=true`.
Weights-only checkpoints remain usable for evaluation and policy initialization.

Resume full learning into a **new** `run_dir` with `agent_load_path=<ckpt> buffer_load_path=<ckpt> step_offset=<N>
logger.wandb.resume=true logger.wandb.resumeid=<id> ...`, using the actual env step and wandb id in
`resume_state.json`. The current full format requires matching world count, replay capacity, n-step
length and gamma. `learning_state.pt` restores Python, NumPy, Torch CPU and visible CUDA generators,
the independent Gym action sampler, and the adaptive curriculum's gains, histories and partial epoch.
The visible CUDA layout must match. Eval and checkpoint points stay on the absolute env-step grid;
the original LR horizon and optimizer age continue.

For a warm start with fresh replay and episodes, set `agent_load_path=<ckpt>` and
`buffer_load_path=null`; select `runner.agent.load_optimizer=true` to retain optimizer
age and schedules. Learned networks and running reward statistics stay loaded, while
the existing collection reset clears per-world episode returns, pending transitions
and correlated exploration state. Replay, RNG and curriculum start fresh. Report the
parent optimizer/update age separately from the new run's `step_offset`. Full learning
restore keeps its existing replay/learning-state owner and resets collection once
after those owners load.

For a same-layout changed-reward child that retains learned agent state and
saved learning generators/curriculum but starts fresh replay, select
`agent_load_path=<parent> buffer_load_path=null learning_state_load_path=<parent>
step_offset=<saved global env_step>`. The explicit learning-state source is read
after environment reset and agent loading, invokes the same strict progress/world
count/curriculum checks, and restores RNG only after collection reset. It does
not open the old replay or recompute its rewards. Keep new external condition
steps separate from the retained parent/global counter and Adam histories;
normal fresh-buffer warmup applies. With `learning_state_load_path=null`, FULL
resume still reads learning state from `buffer_load_path`, and a legacy
agent-only warm start retains its existing fresh-generator/curriculum behavior.
An actor observation extension cannot select this learning-state restore.

For native actor-only warm initialization, set `actor_input_extension_init_path=<parent checkpoint>` and `actor_input_extension_init_config_path=<published parent resolved config.yaml>` in a new run. The named input layout may stay the same or add flat features; existing named features cannot be removed or resized. Keep `agent_load_path=null`, `buffer_load_path=null`, `step_offset=0` and `runner.ever_state_path=null`. This explicit initializer requires the same named native action/scene and future encoder/query contract, maps saved named actor inputs and shifted future columns, and zeros only the added input columns. It retains the actor's running BN state without an initialization-time renormalization; critic/targets, reward statistics, replay, optimizers and learning progress start fresh. Strict FULL does not admit an observation-layout change. After the new condition writes its own FULL, clear both initialization paths and use the ordinary same-condition FULL resume commands. C0 preservation and new-column learning are owning software-test claims only after their actual qualification; later normalization/optimizer behavior is a new learning condition.

New named flat inputs may define their own `obs_params`, `obs_scales` and
`noise_scales`. Every other observation setting, retained input transform and
encoded-window query must remain unchanged; unrelated map entries are not
admitted as new inputs. A surface-context child must explicitly retain its
parent's selected wrist-control carry term. The new input normalization starts
with weight1, bias0, running mean0 and variance1, while mapped parent columns
retain their actual values. This initializes the actor function; it does not
restore a physical episode, curriculum, exploration carry or replay history.

The explicit actor initializer resolves both robot and action scene bindings through `NativeSceneCfg` before comparing their physical meaning. It can read an archived parent's single `object` as one real entry in `objects`; the live child and ordinary runtime/FULL configuration remain collection-only. The scene path, ordered hands, actuators, object roots and parts, root-control law, every other action setting and existing tensor/layout checks must still match. This does not permit adding a hand/object, changing mechanics, remapping unknown action coordinates, or resuming old replay under a changed assistance condition. The initialization receipt records the resolved named scene and whether that archived parent representation was used.

Physical worlds and episodes start fresh. Per-world return carry, correlated exploration noise and
pending n-step transitions are cleared, while learned and running statistics are retained. The replay
stores explicit factual successors for the old fragment's edge, so new episodes cannot become its
next states. The same sealing occurs when evaluation shares and resets the training worlds; the
curriculum also drops its previous physical episode lengths while retaining global history. These
semantics do not claim an identical simulator or CUDA-graph trajectory after restart.

A replay save copies only occupied rows into independent CPU storage; it retains the configured live
capacity and sampler. At high occupancy, allow host RAM and disk space for that complete occupied
snapshot. The new run writes `learning-restored.pt` and `learning-restore.json` before collection so
its actual restored RNG and curriculum state can be compared with the full checkpoint. Old replay
files lacking the current fragment and learning state are not full learning resumes.

The native task uses the same compact replay implementation. Its future
reference span is dropped; its factual base observations stay stored. The shared
native future builder reconstructs that span from scene-local wrist p/WXYZ,
WORLD linear/angular velocity, complete named hand Q/Qvel, object-joint Q/Qvel,
and the stored `(motion_id, motion_step)`. The immutable library supplies actual
native column correspondence, independent support and exact int64 query times.
The physical world count and replay sample count may differ. Reconstruction
does not consult live achieved state or replace a saved phase with the current
command cursor. Existing final-observation and fragment-address handling applies.
Resolved observation entity/joint columns select the native commands by their
actual names. A different Q/Qvel column order is supported; missing required
command columns, clipped reconstruction inputs or zero source scales cannot
recover the factual input and are rejected. Q/Qvel must observe the command's
actual robot entity; a different entity with matching scalar names cannot supply
its physical facts.

Select `--config_name native_surface` for optional named-part surface context.
This recipe extends both actor and critic with `native_part_point_cloud_wrist`
and `native_next_part_point_cloud_delta_wrist`. Point and candidate-pool counts
come from `native_surface.n_points` and `native_surface.n_obj_verts` (defaults
128 and 2048). Each configured real part supplies its existing `mesh_path` and
body-from-mesh transform. Sampling uses the shared deterministic area-uniform
surface pool and farthest-point selection once per configured count/pool/device;
every consumer uses the same cached body-local samples. No hand-associated
object slots, mesh recentering, or observation size normalization are introduced.

Each surface observation orders XYZ by actual hand side, configured entity/part,
then sample, and appends independent hand/part support masks. For `H` hands,
`P` parts and `N` samples its width is `H*P*(3*N+1)`. The live cloud uses the
physical wrist frame. The displacement observation subtracts the live cloud from
the next supported reference cloud in that same frame; next queries retain the
native int64 clock and unavailable tails have zero values with false support.
Actual rigid roots and passive articulated parts keep their normal body poses.
The shared surface reward instead reads the **consumed installed reference**.
It computes `dt + (R_reference - R_live)*p`, averages the largest configured
point displacements per part, applies the deadband, then returns
`exp(-scale*error)`. Only this reward optionally rescales each part's local
rotational lever arms to `norm_radius`; translation and both observations stay
in physical metres. `part_reduce: mean` uses the independently supported part
population; `worst` selects its lowest supported part reward. No support yields
zero reward mass. The example explicitly sets `native_object_part_pose` to zero
and selects one shared surface term at weight 1, scale 10, worst 3 points,
zero deadband and radius 0.058 m. Assistance remains a separate configuration.
Per-part pooling and normalization are a declared extension of the original
rigid-object mechanism, not established cross-object generalization.

The existing replay compactor drops both surface spans and rebuilds them from
stored `native_object_parts_wrist`, scene-local wrist p/WXYZ, the exact stored
motion ID/step, immutable source masks and the same cached geometry. It decodes
the stored per-hand feature-major chart `[all positions, all rotation6D, all
linear rates, all angular rates]`. It does not consult live worlds or store an
extra dense next cloud. A bilateral singleton with 128 samples therefore avoids
1536 XYZ values per replay row, plus the two surface terms' four support values;
its factual part/wrist state remains stored. The same lossless-source rules and
separate FP16 qualification below apply.

`obs_layout.json` records the selected static surface geometry even when replay
compaction is disabled. `NativeRolloutRecord` exports
`native_part_surface_metadata_json` once per tape: actual entity/part/body
addresses, body-from-mesh transforms, ordered sampled body-local points and
count/pool/frame conventions. It does not duplicate clouds per step. The changed
actor/critic input graph needs a genuinely fresh condition or the explicit named
actor input extension initializer above; strict FULL requires its unchanged
layout. Contact wrench/location observations are separate from these surface
features. Software or replay parity alone does not establish successful physical
interaction or full parity with the original training method.

With FP32 storage, retained factual columns must preserve their original bits.
Integer addresses, source support, unavailable queries and int64-clock semantics
remain exact. Recomputed quaternion, rotation-vector and body-relative geometry
must be finite and satisfy the pinned runtime's public
`torch.testing.assert_close` FP32 dtype defaults, including under row permutation.
Record the runtime version and resolved defaults with that qualification. This
criterion applies to derived observation geometry; physical snapshot/replay and
task criteria retain their separately declared requirements. Preserve the result
of an earlier bit-exact gate when reporting a new numerical criterion.

The native profiles use 5 ms physics and four substeps per 20 ms control endpoint;
the prepared carrier must match the actual clock. The stock Isaac author profile
uses the full-precision reciprocal of 240 Hz. Isaac Sim 5.1 stores
`int(1 / dt)` steps per second, so the old rounded `0.0041666667` literal
requests 239 Hz. This source prediction does not establish an unrecorded
historical runtime clock. Record the installed scene frequency and actual dt
when qualifying physics. Configure `n_future`, replay capacity, storage dtype and
device in `runner.agent` and admit memory for the resulting layout. Float16 replay
quantizes stored factual observations; it does not quantize the integer phase,
source clocks or library support. The ordinary sampler casts non-aux storage
groups back to FP32 before geometry reconstruction and keeps integer aux
unchanged. Qualify this storage condition separately: require finite values and
exact support, phase, query and clock semantics, and report factual-term and
derived-field quantization errors. Reconstructed geometry inherits the stored
facts' precision. These errors alone do not establish physical tracking quality
or task success.

## Other algorithms

The runner config picks the algorithm: `runner=flashsac` (the default). A new algorithm is
`src/agents/<name>.py` implementing `agents.Agent` (`src/agents/base.py`) plus `config/runner/<name>.yaml`
with `defaults: [base, _self_]`, `algorithm: <name>` (checkpoints go to `logs/<name>/`) and
`runner.agent.class_name`.
Its `Agent.load_policy` must load that same registered acting architecture for
frozen evaluation without constructing the algorithm's training owners.

## Evaluation

Every `eval.interval` env steps (20M) the policy is rolled out deterministically from the first frame of
every clip, on the training envs. A clip is scored by its **time-mean** tracking error (worst hand):

| error | threshold |
|---|---|
| object position | 3 cm |
| object orientation | 30 deg |
| fingertips vs MANO tips | 6 cm |
| finger links vs the retarget | 8 cm |

`success_rate_obj_<k>` requires the two object errors within `k` x threshold, `success_rate_<k>` all four.
Each clip that passes SR@1.0 (`eval.success_criterion`, default `obj`) is archived once under
`<run>/success/` (`manifest.json` + one rollout `.npz` per motion), and `eval.stop_when_all_solved` ends
the run once every clip is archived.

Each evaluation also writes `evaluation/current-policy-<env-step>-<UTC>.json`
under `runner.run_dir`. It includes every selected library motion and every
observed attempt, including failures, unscored windows and motions absent from
the plan. Raw metrics, configured thresholds and observed denominators remain
available; per-motion counts retain all replicas. Existing aggregate metrics
and SR thresholds stay unchanged. Nonfinite and unsupported operands remain
explicit, and an empty observation window is not a measured pass.

These reports describe the deterministic in-memory policy at the recorded env
step, separately from first-passing archives and cumulative ever-success. Bind
the evaluated policy to its normal post-evaluation FULL through the publication
receipt; the report infers no checkpoint path. Include `evaluation/` in durable
run publication. Periodic and final evaluations at the same env step retain
separate timestamped reports. Native reference tracking and net-link-force/SDF
association do not certify physical interaction or articulated BOX success.

### Evaluate a frozen checkpoint without replay

Activate the selected runtime's existing environment, then use the public entry
with the checkpoint's resolved scene and learner configuration. Use a new or
empty output directory and record the committed executable source separately
from the checkpoint producer:

```sh
FLASH_SOURCE_REF=<committed-evaluator-source-sha> python evaluate.py \
    --checkpoint /path/to/checkpoint \
    --config_path /path/to/resolved-config-directory --config_name evaluation \
    --overrides run_dir=/path/to/new-evaluation \
    --overrides runner.run_dir=/path/to/new-evaluation
```

`--config_name` is the YAML stem, and `--overrides` may be repeated for ordinary
Hydra overrides. Preserve the checkpoint's robot, motion roster, observation and
action layouts, physics, evaluation reset, thresholds and assistance settings.
The config still selects the existing runtime and environment factory; this
entry adds no environment profile or alternative actor architecture.

The registered `runner.agent.class_name` supplies a policy-only loader.
FlashSAC uses the same actor builder as normal learning and strictly restores
its complete `actor_state_dict`, including running BatchNorm and trajectory
encoder buffers. It applies the same observation-layout checks and action
affine coordinates. Configured compilation uses FlashSAC's ordinary mode
resolver, including the supported Torch mode selected by `compile_mode=auto`.
Policy loading constructs no critic, optimizer, reward
normalizer or replay and restores no saved curriculum or RNG state. The ordinary
environment factory still creates its configured managers and uses the existing
evaluation transitions. The common `flashsac.pt` tensor container is CPU-mapped;
only the actor state is installed in the acting module. Neither
`replay_buffer.pt` nor `learning_state.pt` is opened.

One ordinary flat-wrapper metadata reset supplies the real environment layout
and configured command identity before scoring. The evaluator keeps its normal
scoring and exit resets. The config seed starts a fresh process; this does not
replay the saved simulator or training RNG state. `resume_state.json` supplies
the checkpoint's actual training env step; do not substitute a nominal save or
decision grid. Bind the checkpoint producer and the committed executable
revision separately in the operation receipt. Keep experiment-specific motion
rosters, criteria and checkpoint outcomes in that operation's evidence; the
public criteria above remain the default configuration's authority.

The entry writes `evaluation-entry.json` and the ordinary all-motion
`evaluation/current-policy-<actual-env-step>-<UTC>.json`. Failed, unscored and
absent attempts remain in their existing denominators. It does not load
first-success or ever-success state or select a success-only archive. This CLI
writes the ordinary scalar report by default. Add `--recording_dir /path/to/new-recordings`
to retain every actual attempt through the existing recording API. This selection
requires `FLASH_SOURCE_REF`; the entry binds the checkpoint it loaded, its actual
training env step, the executable revision and the policy loader receipt. Select
the desired rollout recorder in `eval.callbacks`. `NativeRolloutRecord` additionally
requires its explicit source, named-body and joint binding; the whole robot root
is not substituted for either wrist. The existing recorder's scene-local position
and WORLD link-twist conventions remain unchanged. A new capture is a new
measurement sweep of the frozen policy, not a reconstruction of an earlier scalar
report.

Keep training and evaluation assistance distinct. Active reference-centered
object PD assistance differs from passive support. With
`eval.env.disable_object_pin=true`, ordinary shared evaluation disables the
command's pin/controller and submits its existing wrench clear. The entry
records the configured flag and actual command pin state before/during/after
evaluation. These are request/flag facts: accepted or applied
object wrench remains UNKNOWN when the scene exposes no public readback.
Net-link force/SDF association and a tracking pass do not establish exact
hand-part forces, successful interaction, F1/BOX or cross-hand transfer.

For a frozen checkpoint comparison, the public `evaluate` call accepts
`recording_dir` and `frozen_authority={"checkpoint": ..., "source_revision": ...}`.
The caller must bind the checkpoint actually loaded and the source actually
executed; the exporter does not verify loading history. Select the normal rollout
recorder in `eval.callbacks`. This explicit call retains that recorder even when
`save_success_data=false`; it does not add a recorder automatically.

The recording directory receives a timestamped `manifest.json` and a separate
NPZ for every recorded attempt, including failed and unscored attempts. Actual
batch/world addresses and per-motion replica indices join the scalar report to
the NPZ's embedded attempt identity. Planned but unexecuted slots and unavailable
recorders retain reasons without invented arrays. This selected-library roster
does not replace an external task family's original planned population.

### Evaluate an analytic native reference follower

Select `eval=native_reference` and omit `--checkpoint` to use the configured
`evaluation.policies.NativeReferenceFollower` through this same entry and
callbacks. Supply the normal native scene, source carrier, recording binding and
task configuration. The follower reads `command_reference()` at the actual next
control endpoint, maps named independent fingers through their real command
bounds and inverse action affine, and requests zero FREE-root correction. Normal
projection, unsupported-command hold and reference pose/rate installation remain
with the existing native action owner. Its finger filter must be disabled and
the absolute finger path selected. No actor, critic, replay or optimizer is built.

The explicit acting authority has `kind="analytic"`, a configured method ID,
`has_learned_parameters=false` and `checkpoint=null`. Direct callers pass that
policy authority and actual `source_revision` in `frozen_authority`; no fake
checkpoint path is required. Selecting both an analytic policy and checkpoint
is an error. Analytic reference following measures execution feasibility and
tracking; it is not a learned-policy or successful-interaction claim.

When an acting or scoring failure interrupts a recorded evaluation, already
completed attempts and the current recorded prefix are exported before the
original exception is raised. Their report records the failure stage and UNKNOWN
scoring, while unexecuted planned slots retain their normal reasons and denominator.
No partial metric callback is promoted to successful task evidence.

Native source phases, exact source clocks, physical clocks, raw interval contacts
and their support retain the recorder's arrays and dtypes. When the underlying
`NewtonPlant` exposes post-solve `mjw_data.xfrc_applied` readback, `NativeRolloutRecord`
also exports `applied_object_wrench_{force_w,torque_w,measurement_support}` and
`evaluate.py` reports `MEASURED_ZERO_POST_SOLVE` (or `MEASURED_NONZERO_POST_SOLVE`)
in `evaluation-entry.json`. First-success archives
keep their existing selection meaning and share the NPZ serializer. Ordinary
training omits `recording_dir` and continues to write scalar attempt reports
without automatically storing every rollout.

## Outputs

```
<run_dir>/
├── config.yaml      resolved configuration, written before runtime launch
├── wandb/           logger output
├── step<N>/         networks (+ replay_buffer.pt + learning_state.pt) + resume_state.json, every `save_checkpoint_per_env_step` env steps
├── stop<N>/         committed FULL plus consumed stop.request
├── final/           networks at the end of the run
├── success/         first passing rollout per motion
├── evaluation/      current-policy per-motion/attempt JSONs, including failures
├── learning-restored.pt    actual restored RNG/curriculum before resumed collection
└── learning-restore.json   restoration scope (last two files on FULL resume)
```

The default is `logs/<algorithm>/<YYYY-MM-DD_HH-MM-SS>_<run_name>/` relative to
the launch directory. Logger, checkpoints, success archive and restore receipts
share the resolved root. Explicit checkpoint load paths remain separate inputs;
they do not implicitly reuse a parent's output directory. SDK shutdown preserves
the actual successful or failing exit status.

## Pipeline

```text
motion .pt (mocap retargeted to the robot) ──► MotionTrackingCommand (reference, sampling, object assist)
                                                       │
Isaac Sim (PhysX, N envs) ◄── scene: bimanual robot USD + object USDs (+ support disks)
        │
ManagerBasedRlEnv (vendored mjlab managers)
   obs · residual PD action (+ LPF) · rewards · terminations · curricula
        │  FlatObsVecEnv: flat obs, one reward column per hand
        ▼
OffPolicyRunner ──► agents.Agent (runner.agent.class_name) — FlashSAC: side actors, per-hand
        │           critic heads, compact fp16 replay buffer, Conv1D future-trajectory encoder
        │ every eval.interval (20M) env steps
        ▼
evaluation/evaluate.py: deterministic rollout over every clip ──► SR@k, ever-success,
                        current-policy attempts, success archive (first passing clip per motion)
```
