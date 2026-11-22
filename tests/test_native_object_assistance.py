"""Real shared force/manager/recorder contracts; CPU software only, no SDK solve."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import simulator.isaacsim  # noqa: F401 -- expose the maintained manager subset

# isort: split
from hydra import compose, initialize_config_dir
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from omegaconf import OmegaConf, open_dict
from test_installed_dynamics import _ObjectCommand, _rigid
from test_native_backend_ports import _mechanics, _ProtocolPlant
from test_native_contract import _scene, write_native_scene_manifest
from test_native_task_contract import _add_loose_object, _command

from common.native_reference import bind_prepared_native_reference
from envs import build_manager_cfgs
from envs.commands.motion_tracking.motion_tracking import MotionTrackingCommand
from envs.commands.motion_tracking.motion_tracking_cfg import ObjectCfg
from envs.commands.motion_tracking.native import NativeMotionTrackingCommand
from envs.commands.motion_tracking.objects.assist import ObjectAssistMixin
from envs.scene.native import bind_native_scene_config
from evaluation.shared import SharedEvalEnv
from simulator.env import ManagerBasedRlEnv, ManagerEnvCfg
from simulator.isaacsim.commands_isaac import IsaacMotionTrackingCommand
from simulator.isaacsim.entity import FreeArticulationAdapter
from simulator.mjcf_meta import load_robot_mjcf_meta
from simulator.native.entity import _FreshnessClock
from simulator.native.scene import NativeScene, NativeSim


@pytest.mark.parametrize("backend", ("native_isaacsim", "mjwarp", "newton"))
def test_native_assisted_config_uses_real_curriculum_and_keeps_static_gravity(tmp_path, backend):
    _, mapping = _scene(tmp_path)
    motion_file = write_native_scene_manifest(tmp_path, mapping)
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1] / "config"), version_base=None):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
        cfg = compose(config_name="native_assisted", overrides=[f"envs/simulator@simulator={backend}",
            f"commands.motion_file={motion_file}",
            "native_tracking_tolerances={root_position_error_m: 1, root_rotation_error_rad: 1, part_position_error_m: 1, part_rotation_error_rad: 1, joint_position_error_rad: 1, command_position_error_rad: 1}"])
    bind_prepared_native_reference(cfg)
    layout = bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=(0., 0., -9.81))
    assert layout.objects[0].root_body_name == "base" and cfg.commands.object.pin_mode == "xfrc"
    built = build_manager_cfgs(cfg)
    adaptive = built["curriculum"]["xfrc_curriculum_adaptive"]
    assert adaptive.params["command_name"] == cfg.commands.name
    assert adaptive.params["completion_term"] in built["terminations"]
    assert set(adaptive.params["reward_terms"]) == {"task", "imi", "bc"}
    for members in adaptive.params["reward_terms"].values():
        assert any(name in built["rewards"] or name + "_r" in built["rewards"] for name in members)
    with open_dict(cfg.curriculum):
        cfg.curriculum.gravity_curriculum = {"params": {"schedule_steps": 1}}
    with pytest.raises(ValueError, match="gravity remains the compiled constructor"):
        bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=(0., 0., -9.81))
    del cfg.curriculum.gravity_curriculum
    cfg.commands.object.pin_objects = False
    with pytest.raises(ValueError, match="curriculum must be empty"):
        bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=(0., 0., -9.81))


@dataclass(kw_only=True)
class _SubstepCfg(CommandTermCfg):
    def build(self, env):
        return _SubstepCommand(self, env)


class _SubstepCommand(CommandTerm):
    @property
    def command(self):
        return torch.zeros(self.num_envs, 0)

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        pass

    def apply_substep(self):
        self._env.events.append("assist")


@pytest.mark.parametrize("substeps", (2, 5))
def test_shared_manager_dispatches_assistance_after_action_before_every_physics_substep(substeps):
    events = []
    scene = SimpleNamespace(num_envs=1, bind_tracking_command=lambda command: None,
        write_data_to_sim=lambda: events.append("write"), update=lambda dt: events.append("update"))
    sim = SimpleNamespace(expand_model_fields=lambda fields: None,
        begin_control_step=lambda: events.append("begin"), end_control_step=lambda: events.append("end"),
        step=lambda: events.append("solve"), forward=lambda: None, sense=lambda: None)
    env = ManagerBasedRlEnv(ManagerEnvCfg(decimation=substeps, physics_dt=.005,
        commands={"renamed": _SubstepCfg(resampling_time_range=(1e9, 1e9))}), scene, sim, "cpu")
    env.events = events
    env.action_manager.apply_action = lambda: events.append("hand")
    env.step(torch.zeros(1, 0))
    assert events == ["begin"] + [item for _ in range(substeps)
        for item in ("hand", "assist", "write", "solve", "update")] + ["end"]


def _native_assist(tmp_path):
    layout, _, model = _mechanics(tmp_path)
    plant = _ProtocolPlant(model)
    sim = NativeSim(plant)
    scene = NativeScene(layout, load_robot_mjcf_meta(layout.scene_xml_path), sim, "robot")
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02, scene=scene)
    command = object.__new__(NativeMotionTrackingCommand)
    command._env, command.native_layout = env, layout
    command.cfg = SimpleNamespace(object=ObjectCfg(pin_objects=True, pin_mode="xfrc",
        entity_names={"left": "object", "right": "object"}))
    command.cfg.object.xfrc.rot_inertia_scale = False
    command._part_body_ids = [scene["object"].body_names.index(part.body_name) for part in layout.objects[0].parts]
    command._part_entities = tuple(scene[obj.entity_name] for obj, _ in layout.part_bindings)
    command._object_part_columns = {"object": list(range(len(layout.part_bindings)))}
    command.motion_ids, command.motion_steps = torch.zeros(2, dtype=torch.long), torch.arange(2)
    state = command.sim_object_part_state_w.clone()
    reference = state.clone()
    reference[..., 0] += .02
    reference[..., 3:7] = torch.tensor([np.cos(.05), 0., 0., np.sin(.05)])
    reference[..., 7:] = 999.0  # Unsupported rates must not become feedforward.
    pose_support = torch.tensor([[True, True], [True, False]])
    reference[1, 1, :7] = float("nan")
    command.motion_lib = SimpleNamespace(length_starts=torch.tensor([0]), num_trajectories=1,
        native_object_part_state_w=reference, native_object_part_pose_support=pose_support,
        native_object_part_linear_velocity_support=torch.zeros(2, 2, dtype=torch.bool),
        native_object_part_angular_velocity_support=torch.zeros(2, 2, dtype=torch.bool))
    command._init_object_assist()
    command._init_native_object_assist()
    command._xfrc_kp_pos_env = torch.tensor([80., 20.])
    command._xfrc_kv_pos_env = torch.tensor([5., 1.])
    command._xfrc_kp_rot_env = torch.tensor([10., 4.])
    command._xfrc_kv_rot_env = torch.tensor([1., .5])
    command._xfrc_force_range = 50.0
    return command, env, plant, model


def test_native_assist_uses_named_parts_preserves_passive_hinge_and_clears_eval_reset_snapshot(tmp_path):
    command, env, plant, model = _native_assist(tmp_path)
    ids = [model.body(part.body_name).id for part in command.native_layout.objects[0].parts]
    before_control = plant.control.clone()
    command.apply_substep()
    submitted = plant.external_body_wrench_w[:, ids].clone()
    torch.testing.assert_close(submitted[..., 0], torch.tensor([[1.6, 1.6], [.4, 0.]]))
    torch.testing.assert_close(submitted[..., 5], torch.tensor([[1., 1.], [.4, 0.]]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(submitted[..., 1:3], torch.zeros(2, 2, 2))
    torch.testing.assert_close(plant.control, before_control, atol=0, rtol=0)
    meta = load_robot_mjcf_meta(command.native_layout.scene_xml_path).subtree("base")
    assert meta.actuator_names == [] and meta.joint_names == ["hinge"]
    assert len(meta.free_roots) == 1 and not meta.joint_limited[0]
    addresses = [value.data_ptr() for value in (command._assist_force_w, command._assist_last_force_w)]
    for _ in range(3):
        command.apply_substep()
    assert addresses == [value.data_ptr() for value in (command._assist_force_w, command._assist_last_force_w)]
    assert not command._assist_last_force_w.requires_grad
    assert not command.object_assist_facts()["linear_rate_support"].any()
    snapshot = plant.snapshot()
    command.clear_object_wrenches(torch.tensor([0]))
    assert not plant.external_body_wrench_w[0].any()
    torch.testing.assert_close(plant.external_body_wrench_w[1, ids], submitted[1])
    assert command._assist_submission_support.tolist() == [False, True]
    plant.restore(snapshot)
    torch.testing.assert_close(plant.external_body_wrench_w[:, ids], submitted)
    command._xfrc_curr_ctrl = object()
    command.set_eval_mode = lambda **kwargs: None
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    env.termination_manager, env.curriculum_manager = object(), object()
    evaluation = SharedEvalEnv(env, SimpleNamespace(command_name="motion", num_per_motion=1,
        command=SimpleNamespace(sampling_mode="start", noise_to_initial_level=0., start_frame=0)))
    evaluation.enter_eval()
    assert not command.cfg.object.pin_objects and command._xfrc_curr_ctrl is None
    assert not plant.external_body_wrench_w.any() and not command._assist_submission_support.any()
    command.apply_substep()
    assert not plant.external_body_wrench_w.any() and command._assist_submission_support.all()
    assert not command.object_assist_facts()["force_w"].any()
    command.clear_object_wrenches(torch.tensor([0]))
    assert command._assist_submission_support.tolist() == [False, True]


def test_native_training_perturbs_only_free_root_and_resets_disturbance_carry(tmp_path):
    command, env, plant, model = _native_assist(tmp_path)
    command._object_root_columns = torch.tensor([0])
    command._eval_mode = False
    command._xfrc_curr_ctrl = SimpleNamespace(has_zero_gains=lambda: True)
    perturb = command.cfg.object.perturb
    perturb.enabled, perturb.start_prob, perturb.gate_lag_s = True, 1.0, 0.0
    perturb.force_scale, perturb.torque_scale = 1.5, 0.03
    perturb.max_angular_acceleration_rad_s2 = 1e6
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    ml = command.motion_lib
    ml.native_time_ns = torch.arange(2) * 20_000_000
    active = torch.ones(2, 2, 2, dtype=torch.bool)
    ml.native_source_hand_part_intent = dict(active=active, known=active.clone())
    command._init_native_object_assist()
    command.apply_substep()

    root = model.body("base").id
    torch.testing.assert_close(plant.external_body_wrench_w[:, root, :3].norm(dim=-1), torch.full((2,), 1.5))
    torch.testing.assert_close(plant.external_body_wrench_w[:, root, 3:].norm(dim=-1), torch.full((2,), 0.03))
    assert not plant.external_body_wrench_w[:, model.body("lid").id].any()
    command.clear_object_wrenches(torch.tensor([0]))
    assert not plant.external_body_wrench_w[0].any()
    assert plant.external_body_wrench_w[1, root].any()


def test_native_training_disturbs_only_the_source_engaged_object_after_lag():
    command, env, calls = _command()
    _add_loose_object(command, env, calls)
    command.cfg.object = ObjectCfg(pin_objects=True)
    command._init_object_assist()
    command._eval_mode = False
    command._xfrc_curr_ctrl = SimpleNamespace(has_zero_gains=lambda: True)
    perturb = command.cfg.object.perturb
    perturb.enabled, perturb.start_prob, perturb.continue_prob = True, 1.0, 1.0
    perturb.gate_lag_s, perturb.force_scale, perturb.torque_scale = .02, 1.5, .03
    perturb.max_angular_acceleration_rad_s2 = 1000.
    ml = command.motion_lib
    active = torch.zeros(3, 2, 3, dtype=torch.bool)
    active[:2, 1, 1] = True   # Recorded contact is with the articulated part.
    active[1:, 0, 2] = True   # Another hand and object engage a frame later.
    known = active.clone()
    known[0, 0, 2] = False
    active[0, 0, 2] = True    # Unavailable intent cannot start the lag.
    ml.native_source_hand_part_intent = dict(active=active, known=known)
    submitted = {}
    for name, obj in command.native_objects.items():
        obj.body_inertial_properties = lambda obj=obj: SimpleNamespace(
            inertia_com_body_kg_m2=torch.eye(3).expand(1, len(obj.body_names), 3, 3))
        obj.write_external_wrench_to_sim = lambda f, t, *, body_ids, name=name: submitted.update(
            {name: torch.cat((f, t), -1).clone()})
    command._init_native_object_assist()

    for frame, expected in enumerate(((False, False), (True, False), (False, True))):
        command.motion_steps[:] = frame
        env.episode_length_buf[:] = frame
        command.apply_substep()
        assert tuple(bool(wrench.any()) for wrench in submitted.values()) == expected
        assert not submitted["object"][:, 1].any()
        for wrench, carries_disturbance in zip(submitted.values(), expected, strict=True):
            norms = torch.stack((wrench[0, 0, :3].norm(), wrench[0, 0, 3:].norm()))
            torch.testing.assert_close(norms, torch.tensor([1.5, .03] if carries_disturbance else [0., 0.]))

    # Starting/resetting in the middle of an active source phase still waits.
    command.clear_object_wrenches(torch.tensor([0]))
    env.episode_length_buf.zero_()
    command.apply_substep()
    assert not any(wrench.any() for wrench in submitted.values())


def test_native_perturbation_preserves_linear_force_and_bounds_root_angular_acceleration(monkeypatch):
    command, env, calls = _command()
    heavy = _add_loose_object(command, env, calls)
    light = env.scene["object"]
    command.cfg.object = ObjectCfg(pin_objects=True)
    command._init_object_assist()
    command._eval_mode = False
    command._xfrc_curr_ctrl = SimpleNamespace(has_zero_gains=lambda: True)
    perturb = command.cfg.object.perturb
    perturb.enabled, perturb.start_prob, perturb.continue_prob, perturb.gate_lag_s = True, 1., 1., 0.
    perturb.force_scale, perturb.torque_scale = 1.5, .03
    perturb.max_angular_acceleration_rad_s2 = 60.
    active = torch.zeros(3, 2, 3, dtype=torch.bool)
    active[:, 0, 1] = True  # This contact with a passive part disturbs its FREE root.
    active[:, 1, 2] = True
    command.motion_lib.native_source_hand_part_intent = dict(active=active, known=active.clone())

    light_inertia = torch.tensor([1e-5, .003, .006])
    heavy_inertia = torch.tensor([.02, .03, .04])
    light.body_inertial_properties = lambda: SimpleNamespace(
        inertia_com_body_kg_m2=torch.diag_embed(torch.stack((light_inertia, heavy_inertia)))[None])
    heavy.body_inertial_properties = lambda: SimpleNamespace(
        inertia_com_body_kg_m2=torch.diag(heavy_inertia)[None, None])
    light.data.body_link_quat_w[0, 0] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    submitted = {}
    for name, obj in command.native_objects.items():
        obj.write_external_wrench_to_sim = lambda f, t, *, body_ids, name=name: submitted.update(
            {name: torch.cat((f, t), -1).clone()})
    directions = torch.tensor([[1., -2., 3.], [0., 1., .5]])
    monkeypatch.setattr(torch, "randn", lambda size, **kwargs: directions.expand(size).clone())
    command._init_native_object_assist()
    command.apply_substep()

    expected_force = directions[0] * (1.5 / directions[0].norm())
    full_torque = directions[1] * (.03 / directions[1].norm())
    local_full_torque = full_torque[[1, 0, 2]]  # The y-world impulse acts on the thin local x-axis.
    expected_torque = full_torque * (60. / (local_full_torque / light_inertia).norm())
    light_wrench, heavy_wrench = submitted["object"][0, 0], submitted["loose_prop"][0, 0]
    torch.testing.assert_close(light_wrench[:3], expected_force)
    torch.testing.assert_close(heavy_wrench[:3], expected_force)
    torch.testing.assert_close(light_wrench[3:], expected_torque)
    torch.testing.assert_close((light_wrench[3:][[1, 0, 2]] / light_inertia).norm(), torch.tensor(60.))
    torch.testing.assert_close(heavy_wrench[3:], full_torque)
    torch.testing.assert_close(command.object_assist_facts()["torque_w"][0, 0], expected_torque)
    assert not submitted["object"][0, 1].any()

    # An unchanged burst recovers full torque when the struck body rotates.
    light.data.body_link_quat_w[0, 0] = torch.tensor([1., 0., 0., 0.])
    perturb.start_prob = 0.
    command.apply_substep()
    torch.testing.assert_close(submitted["object"][0, 0, 3:], full_torque)


@pytest.mark.parametrize("compiled", (False, True))
def test_native_pair_plans_are_stable_for_source_roles_across_batches(tmp_path, compiled):
    command, env, plant, model = _native_assist(tmp_path)
    command._object_root_columns = torch.tensor([0])
    command._eval_mode = True
    command.cfg.object.pin_objects = False
    env.physics_dt = .005
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    perturb = command.cfg.object.perturb
    perturb.enabled, perturb.start_prob, perturb.continue_prob, perturb.gate_lag_s = True, 0., .5, .02
    perturb.force_scale = 1.5
    ml = command.motion_lib
    rows = 8
    ml.length_starts = torch.tensor([0, rows])
    ml._motion_num_frames = torch.tensor([rows, rows])
    ml.num_trajectories = 2
    ml.motion_files = ["anywhere/motion.pt#one", "elsewhere/motion.pt#two"]
    ml.hand_sides = ("left", "right")
    ml.native_time_ns = (torch.arange(rows) * 20_000_000).repeat(2)
    ml.native_object_part_pose_support = torch.ones(2 * rows, 2, dtype=torch.bool)
    ml.native_object_part_linear_velocity_support = torch.zeros_like(ml.native_object_part_pose_support)
    ml.native_object_part_angular_velocity_support = torch.zeros_like(ml.native_object_part_pose_support)
    engaged = torch.zeros(2 * rows, 2, 2, dtype=torch.bool)
    engaged[:, 0, 1] = True     # Movable body has a hand role, FREE root takes the disturbance.
    source_frames = torch.arange(100, 100 + rows)
    ml.native_source_hand_part_intent = dict(
        active=engaged, known=engaged.clone(), source_frame_index=source_frames.repeat(2))
    ml.source_lineage = [
        dict(source_id="source", source_revision="release", source_unit_id=f"capture_{index}",
             selected_source_frame_index=source_frames.tolist(), processing=dict(source_task_phases=[
                 dict(kind=phase, source_frame_range_half_open=[103, 106], authority="annotated", hand_side="left")
             ])) for index, phase in enumerate(("interaction", "retreat"))]
    if compiled:
        for source in ml.source_lineage:
            source["processing"]["task_phases"] = [
                dict(kind="interaction", source_frame_range_half_open=[101, 104],
                     hand_side="left", authority="compiled"),
                *[dict(kind="articulation", source_frame_range_half_open=bounds,
                       authority="compiled", entity_name=command.native_layout.objects[0].entity_name,
                       part_body_name="lid", joint_name="hinge", coordinate_kind="hinge",
                       native_direction=direction) for bounds, direction in (([103, 105], 1), ([105, 108], -1))],
            ]
        engaged[rows:, :, 1] = False
        engaged[rows:, 0, 0] = True  # Support contact cannot back a moving-part phase.
        ml.native_source_hand_part_intent["known"] = engaged.clone()
    command._init_native_object_assist()
    command.enable_hoi_evaluation_perturbation("perturbed")
    command.motion_ids[:] = torch.tensor([0, 1])
    command.motion_steps[:] = 1
    first, unusable = command.plan_hoi_evaluation_attempts(torch.tensor([0, 1]), [0, 0])
    assert unusable["status"] == "no_usable_source_phase" and not unusable["bursts"]
    burst = first["bursts"][0]
    assert burst["root_body_name"] == "base"
    if compiled:
        assert [b["phase"]["native_direction"] for b in first["bursts"]] == [1, -1]
        assert unusable["missing_phases"]
    else:
        assert burst["phase"] == ml.source_lineage[0]["processing"]["source_task_phases"][0]
    assert first["first_reference_motion_step"] == 1
    assert 8 <= burst["start_substep"] < burst["stop_substep_exclusive"] <= 20

    applied = []
    for control in range(rows - 1):
        command.motion_steps[:] = control + 1
        env.episode_length_buf[:] = control
        for _ in range(4):
            command.apply_substep()
            applied.append(plant.external_body_wrench_w[:, model.body("base").id].clone())
            assert not plant.external_body_wrench_w[:, model.body("lid").id].any()
    acquired = torch.stack(applied)
    assert not acquired[:, 1].any()
    corrected = torch.nonzero(acquired[:, 0].any(-1), as_tuple=False).flatten().tolist()
    intervals = [list(range(b["start_substep"], b["stop_substep_exclusive"])) for b in first["bursts"]]
    assert corrected == sorted(step for interval in intervals for step in interval)
    for b, interval in zip(first["bursts"], intervals, strict=True):
        torch.testing.assert_close(acquired[interval, 0, :3], torch.tensor(b["force_world_n"]).expand(len(interval), 3))

    command.clear_object_wrenches()
    command.motion_ids[:] = torch.tensor([1, 0])
    command.motion_steps[:] = 1
    second, = command.plan_hoi_evaluation_attempts(torch.tensor([1]), [0])
    replica, = command.plan_hoi_evaluation_attempts(torch.tensor([1]), [1])
    assert second == first
    assert replica["replica_index"] == 1 and replica["bursts"] != first["bursts"]

    command.enable_hoi_evaluation_perturbation("nominal")
    nominal, = command.plan_hoi_evaluation_attempts(torch.tensor([1]), [0])
    assert nominal == first
    command._assist_eval_tick[1] = burst["start_substep"]
    command.motion_steps[1] = 1 + burst["start_substep"] // 4
    command.apply_substep()
    assert not plant.external_body_wrench_w.any()


def test_unsupported_object_rates_do_not_add_velocity_drag(tmp_path, monkeypatch):
    command, _, plant, model = _native_assist(tmp_path)
    ids = [model.body(part.body_name).id for part in command.native_layout.objects[0].parts]
    command.apply_substep()
    still_wrench = plant.external_body_wrench_w[:, ids].clone()
    moving = command.sim_object_part_state_w.clone()
    moving[..., 7:] = 3.0
    monkeypatch.setattr(NativeMotionTrackingCommand, "sim_object_part_state_w", property(lambda _: moving))

    command.apply_substep()
    torch.testing.assert_close(plant.external_body_wrench_w[:, ids], still_wrench)


def test_shared_pd_keeps_stock_omega_and_configured_world_mass_behavior():
    assert MotionTrackingCommand.apply_substep is ObjectAssistMixin.apply_substep
    obj, writes, transforms, _ = _rigid([2., 3.], name="payload")
    transforms[:, 3:] = torch.tensor([0., 0., 0., 1.])
    fixture = _ObjectCommand({"payload": obj}, {"right": "payload", "left": "payload"},
        {side: torch.tensor([0, 0]) for side in ("right", "left")})
    command = object.__new__(IsaacMotionTrackingCommand)
    command.__dict__.update(fixture.__dict__)
    command.motion_lib.length_starts = torch.arange(2)
    command.motion_steps = torch.zeros(2, dtype=torch.long)
    command.cfg.object.xfrc.kp.pos = 7.0
    command.cfg.object.xfrc.kd.pos = 0.0
    command.cfg.object.xfrc.omega_rot = 2.0
    command.cfg.object.xfrc.zeta_rot = 1.0
    command.apply_substep()
    assert not writes
    command._apply_object_assist()
    assert len(writes) == 1  # Both hand roles address one real object entity.
    force, torque = writes[-1][:2]
    torch.testing.assert_close(force[:, 0, 0], torch.tensor([7., 7.]))
    torch.testing.assert_close(torque[:, 0, 2], torch.tensor([2.4, 3.6]), rtol=1e-6, atol=1e-6)
    command.cfg.object.pin_objects = False
    command.apply_substep()
    assert len(writes) == 2  # Disabled also clears the shared entity only once.
    assert not writes[-1][0].any() and not writes[-1][1].any()


def test_isaac_native_body_wrench_uses_named_world_com_and_preserves_other_world_on_clear(tmp_path):
    layout, _ = _scene(tmp_path)
    meta = load_robot_mjcf_meta(layout.scene_xml_path).subtree("base")
    count, bodies = 2, len(meta.body_names)
    names = list(reversed(meta.body_names))
    transforms = torch.zeros(count, bodies, 7)
    transforms[..., :3] = torch.tensor([2., -3., 5.])
    transforms[..., 3:] = torch.tensor([0., 0., 2**-.5, 2**-.5])
    com = transforms.clone()
    com[..., :3] = torch.tensor([.03, -.02, .01])
    view = SimpleNamespace(get_masses=lambda: torch.ones(count, bodies), get_coms=lambda: com,
        get_inertias=lambda: torch.eye(3).reshape(1, 1, 9).expand(count, bodies, -1),
        get_link_transforms=lambda: transforms, get_link_velocities=lambda: torch.zeros(count, bodies, 6))
    calls = []
    sdk = SimpleNamespace(is_fixed_base=False, num_instances=count, root_physx_view=view,
        find_bodies=lambda requested, preserve_order: ([names.index(name) for name in requested], requested),
        find_joints=lambda requested, preserve_order: (list(range(len(requested))), requested),
        set_external_force_and_torque=lambda f, t, **kw: calls.append((f.clone(), t.clone(), kw)),
        reset=lambda env_ids=None: calls.append(("sdk_reset", env_ids)))
    entity = FreeArticulationAdapter(sdk, meta, _FreshnessClock(), "cpu", root_body_name="base")
    force = torch.tensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]])
    entity.write_external_wrench_to_sim(force, -force, body_ids=[1, 0])
    f, t, kw = calls[-1]
    assert kw == {"body_ids": [1, 0], "is_global": True}
    torch.testing.assert_close(f, force[:, [1, 0]])
    torch.testing.assert_close(t, -force[:, [1, 0]])
    # Nonidentity placement/COM never rotates or translates this WORLD COM request.
    entity.reset(torch.tensor([0]))
    f, t, kw = calls[-1]
    assert not f[0].any() and not t[0].any()
    torch.testing.assert_close(f[1], force[1, [1, 0]])
    entity.clear_external_wrenches()
    assert not calls[-1][0].any() and not calls[-1][1].any()


def test_ordinary_native_recorder_preserves_submitted_assist_request_and_unknown_force_authority(tmp_path):
    from test_native_rollout_record import record_reference_phase_fixture

    source = dict(source_unit_id="synthetic", prepared_time_ns=[0, 20_000_000, 40_000_000],
        selected_source_frame_index=[0, 1, 2], prepared_source_ordinal=[0., 1., 2.],
        prepared_elapsed_time_ns=[0, 20_000_000, 40_000_000])
    binding = dict(scene_from_source_world=np.eye(4).tolist(),
        hands={side: dict(body_name=side + "_root", physical_joint_names={"driver": side + "_driver"})
               for side in ("left", "right")},
        object_part_names=["base", "lid"], object_parts={name: dict(entity_name="object", body_name=body)
            for name, body in (("base", "object_root"), ("lid", "object_lid"))}, clips={"fixture": source})

    def setup(env, command):
        def submitted():
            row = int(command.reference_phase["motion_steps"].item())
            force = torch.zeros(1, 2, 3)
            force[0, :, 0] = torch.tensor([2., 5.]) * row  # deliberately lid/base
            return dict(entity_names=("object", "object"), body_names=("object_lid", "object_root"), force_w=force, torque_w=-force,
                pose_support=torch.ones(1, 2, dtype=torch.bool),
                linear_rate_support=torch.zeros(1, 2, dtype=torch.bool),
                angular_rate_support=torch.zeros(1, 2, dtype=torch.bool),
                submission_support=torch.tensor([row > 0]), clock_support=torch.tensor([row > 0]),
                sample_time_s=torch.tensor([1.25 + row * .02 - .005 if row else float("nan")], dtype=torch.float64),
                authority="last_physics_substep_submitted_WORLD_body_COM_wrench_not_measured_force",
                unsupported_velocity="zero target does not establish source rate support",
                force_unit="N", torque_unit="N m", clock="synthetic last substep begin")
        command.object_assist_facts = submitted

    manifest_path = record_reference_phase_fixture(tmp_path, binding, native=True, setup_scene=setup)
    entry = json.loads(manifest_path.read_text())["saved"][0]
    with np.load(manifest_path.parent / entry["rollout"], allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["object_assist_metadata_json"]))
        assert metadata["body_names"] == ["object_root", "object_lid"]
        assert "not_measured_force" in metadata["authority"]
        np.testing.assert_array_equal(arrays["object_assist_submission_support"], [False, True, True])
        np.testing.assert_allclose(arrays["object_assist_force_w"][:, :, 0], [[0., 0.], [5., 2.], [10., 4.]])
        np.testing.assert_array_equal(arrays["object_assist_torque_w"], -arrays["object_assist_force_w"])
        assert not arrays["object_assist_linear_rate_support"].any()
        assert np.isnan(arrays["object_assist_sample_time_s"][0])
