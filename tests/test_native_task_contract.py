"""Synthetic native task contracts and CPU FK; no learner, Kit or integration."""

import math
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pytest
import torch

from envs.commands.motion_tracking.native import NativeMotionTrackingCommand
from envs.observations.native import NativeMotionTrackingObs
from envs.rewards.native import NativeMotionTrackingRewards
from envs.scene.native import NativeSceneCfg
from envs.terminations.native import NativeMotionTrackingTerminations
from evaluation.callbacks.tracking_performance import NativeReferenceTrackingPerformance
from simulator.isaacsim.commands_isaac import (
    IsaacMotionTrackingCommand,
)
from simulator.isaacsim.entity import FreeArticulationAdapter
from simulator.native.entity import NamedArticulationView


def _declare_joint_coordinates(entity, kinds):
    """Declare the compiled scalar chart bypassed by a synthetic entity."""
    entity._meta = SimpleNamespace(joint_names=list(entity.joint_names), joint_types=list(kinds))
    entity.joint_coordinate_metadata = NamedArticulationView.joint_coordinate_metadata.__get__(entity)


def _command(*, rigid=False, passive_types=None, hand_types=None):
    """Explicit synthetic source and physical-state protocol, never native-data qualification."""

    def states(count):
        value = torch.zeros(3, count, 13)
        value[..., 3] = 1
        return value

    parts = [dict(name="body", body_name="object_root", mesh_path="synthetic.obj")]
    if not rigid:
        parts.append(
            dict(name="lid", body_name="object_lid", mesh_path="synthetic-lid.obj")
        )
    control = dict(
        position_stiffness=400,
        position_damping=40,
        rotation_stiffness=400,
        rotation_damping=40,
        force_limit_n=10,
        torque_limit_nm=1,
    )
    layout = NativeSceneCfg.from_mapping(
        dict(
            scene_xml_path="synthetic-not-a-model.xml",
            hands=[
                dict(
                    side=s,
                    entity_name=s,
                    root_body_name=s + "_root",
                    root_joint_name=s + "_free",
                    command_actuator_names=["command_" + s],
                    root_control=control,
                )
                for s in ("left", "right")
            ],
            objects=(dict(
                entity_name="object",
                root_body_name="object_root",
                root_joint_name="object_free",
                parts=parts,
            ),),
        )
    )
    part_state = states(len(parts))
    if not rigid:
        part_state[:, 1, 0] = 1
    object_joints = () if rigid else ("hinge_a", "hinge_b")
    declared_joints = tuple(reversed(object_joints))
    object_q = torch.zeros(3, len(object_joints))
    if not rigid:
        object_q[:] = torch.tensor([0.2, 0.1])
    ml = SimpleNamespace(
        native_source_point_position_support=None,
        native_source_hand_part_intent=None,
        native_time_ns=torch.tensor(
            [1_000_000_000, 1_020_000_000, 1_040_000_000], dtype=torch.int64
        ),
        length_starts=torch.tensor([0]),
        _motion_num_frames=torch.tensor([3]),
        num_trajectories=1,
        native_joint_position_support=torch.ones(3, 3, dtype=torch.bool),
        native_joint_velocity_support=torch.zeros(3, 3, dtype=torch.bool),
        robot_joint_pos=torch.zeros(3, 3),
        robot_joint_vel=torch.zeros(3, 3),
        native_root_body_names={s: s + "_root" for s in ("left", "right")},
        native_root_state_w={s: states(1)[:, 0] for s in ("left", "right")},
        native_root_pose_support={
            s: torch.ones(3, dtype=torch.bool) for s in ("left", "right")
        },
        native_root_linear_velocity_support={
            s: torch.zeros(3, dtype=torch.bool) for s in ("left", "right")
        },
        native_root_angular_velocity_support={
            s: torch.zeros(3, dtype=torch.bool) for s in ("left", "right")
        },
        native_object_root_body_names={"object": layout.objects[0].root_body_name},
        native_object_part_body_names=tuple(p["body_name"] for p in parts),
        native_object_part_state_w=part_state,
        native_object_part_pose_support=torch.ones(3, len(parts), dtype=torch.bool),
        native_object_part_linear_velocity_support=torch.zeros(
            3, len(parts), dtype=torch.bool
        ),
        native_object_part_angular_velocity_support=torch.zeros(
            3, len(parts), dtype=torch.bool
        ),
        native_object_joint_names=declared_joints,
        native_object_joint_pos=object_q,
        native_object_joint_vel=torch.zeros_like(object_q),
        native_object_joint_position_support=torch.ones_like(
            object_q, dtype=torch.bool
        ),
        native_object_joint_velocity_support=torch.zeros_like(
            object_q, dtype=torch.bool
        ),
    )
    calls = []
    obj = SimpleNamespace(
        body_names=list(ml.native_object_part_body_names),
        joint_names=object_joints,
        data=SimpleNamespace(
            body_link_pos_w=part_state[:1, :, :3].clone(),
            body_link_quat_w=part_state[:1, :, 3:7].clone(),
            body_link_lin_vel_w=part_state[:1, :, 7:10].clone(),
            body_link_ang_vel_w=part_state[:1, :, 10:13].clone(),
        ),
        write_root_state_to_sim=lambda value, ids: calls.append(
            ("object_root", value.clone(), ids.clone())
        ),
        clear_external_wrenches=lambda ids=None: calls.append(("clear_object", ids)),
    )
    if not rigid:
        obj.data.joint_pos = torch.tensor([[0.1, 0.2]])
        obj.data.joint_vel = torch.zeros(1, 2)
        obj.write_joint_state_to_sim = lambda q, v, env_ids: calls.append(
            ("object_joints", q.clone(), v.clone())
        )
    robot = SimpleNamespace(
        joint_names=("left_driver", "passive_mimic", "right_driver"),
        data=SimpleNamespace(
            joint_pos=torch.zeros(1, 3),
            joint_vel=torch.zeros(1, 3),
            body_link_pos_w=torch.zeros(1, 2, 3),
            body_link_quat_w=torch.tensor([[[1.0, 0, 0, 0]] * 2]),
            body_link_lin_vel_w=torch.zeros(1, 2, 3),
            body_link_ang_vel_w=torch.zeros(1, 2, 3),
        ),
        hands={
            s: SimpleNamespace(
                write_root_state_to_sim=lambda state, ids, s=s: calls.append(
                    (s, state.clone(), ids.clone())
                )
            )
            for s in ("left", "right")
        },
        write_joint_state_to_sim=lambda q, v, env_ids: calls.append(
            ("hand_joints", q.clone(), v.clone())
        ),
    )
    _declare_joint_coordinates(obj, ("hinge",) * len(object_joints) if passive_types is None else passive_types)
    _declare_joint_coordinates(robot, ("hinge",) * len(robot.joint_names) if hand_types is None else hand_types)

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        scene=Scene(robot=robot, object=obj),
        episode_length_buf=torch.tensor([20]),
    )
    command = object.__new__(NativeMotionTrackingCommand)
    command.native_layout, command.motion_lib, command.robot = layout, ml, robot
    command._env = env
    command._side_list = ["left", "right"]
    command._root_body_ids = [0, 1]
    command._finger_joint_ids = torch.tensor([0, 2])
    command.motion_ids, command.motion_steps = torch.tensor([0]), torch.tensor([0])
    command._eval_mode = False
    command.cfg = SimpleNamespace(
        sampling=SimpleNamespace(mode="start", start_frame=0),
        hand=SimpleNamespace(zero_init_vel=False),
        object=SimpleNamespace(zero_init_vel=False, pin_objects=False),
    )
    command.metrics = {}
    command._bind_native_reference()
    command._init_native_object_assist()
    command._init_metrics()
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    return command, env, calls


def _add_loose_object(command, env, calls):
    """Add a distinct physical protocol entity; no joint or hand-side alias."""
    from dataclasses import replace

    first = command.native_layout.objects[0]
    binding = replace(first, entity_name="loose_prop", root_body_name="prop_root",
        root_joint_name="prop_free", parts=(replace(first.parts[0], body_name="prop_root"),))
    command.native_layout = replace(command.native_layout, objects=(*command.native_layout.objects, binding))
    ml = command.motion_lib
    state = ml.native_object_part_state_w.new_zeros(len(ml.native_time_ns), 1, 13)
    state[..., 0], state[..., 3] = 2., 1.
    ml.native_object_root_body_names = command.native_layout.object_root_body_names
    ml.native_object_part_body_names += ("prop_root",)
    ml.native_object_part_state_w = torch.cat((ml.native_object_part_state_w, state), 1)
    for field in ("object_part_pose_support", "object_part_linear_velocity_support",
                  "object_part_angular_velocity_support"):
        extra = torch.full(state.shape[:2], field == "object_part_pose_support", dtype=torch.bool)
        setattr(ml, "native_" + field, torch.cat((getattr(ml, "native_" + field), extra), 1))
    loose = SimpleNamespace(body_names=("prop_root",), joint_names=(), data=SimpleNamespace(
        body_link_pos_w=state[:1, :, :3].clone(), body_link_quat_w=state[:1, :, 3:7].clone(),
        body_link_lin_vel_w=state[:1, :, 7:10].clone(), body_link_ang_vel_w=state[:1, :, 10:13].clone()),
        write_root_state_to_sim=lambda value, ids: calls.append(("prop_root", value.clone(), ids.clone())),
        clear_external_wrenches=lambda ids=None: calls.append(("clear_prop", ids)))
    _declare_joint_coordinates(loose, ())
    env.scene["loose_prop"] = loose
    command._bind_native_reference()
    command._init_native_object_assist()
    return loose


def test_multiple_object_reset_and_assistance_follow_actual_tree_ownership():
    from envs.commands.motion_tracking.motion_tracking_cfg import ObjectCfg

    command, env, calls = _command()
    loose = _add_loose_object(command, env, calls)
    command._resample_command(torch.tensor([0]))
    assert [entry[0] for entry in calls].count("object_root") == 1
    assert [entry[0] for entry in calls].count("prop_root") == 1
    assert [entry[0] for entry in calls].count("object_joints") == 1
    assert command.native_object_joint_pos.shape == (1, 2)
    assert command.object_joint_coordinate_metadata["entity_names"] == ("object", "object")
    assert command.ref_obj_trans_w.shape == (1, 2, 3)
    torch.testing.assert_close(command.ref_obj_trans_w[0, :, 0], torch.tensor([0., 2.]))

    submitted = {}
    for name, obj in command.native_objects.items():
        obj.body_inertial_properties = lambda obj=obj: SimpleNamespace(
            inertia_com_body_kg_m2=torch.eye(3).expand(1, len(obj.body_names), 3, 3))
        obj.write_external_wrench_to_sim = lambda f, t, *, body_ids, name=name: submitted.update(
            {name: (f.clone(), t.clone(), list(body_ids))})
    command.cfg.object = ObjectCfg(pin_objects=True, pin_mode="xfrc")
    command._init_object_assist()
    command.cfg.object.xfrc.kp.pos = 10.
    command.cfg.object.xfrc.kd.pos = 0.
    loose.data.body_link_pos_w[..., 0] += 1.
    command._apply_object_assist()
    assert set(submitted) == {"object", "loose_prop"}
    assert submitted["object"][0].shape == (1, 2, 3) and submitted["loose_prop"][0].shape == (1, 1, 3)
    assert submitted["loose_prop"][2] == [0]
    torch.testing.assert_close(submitted["object"][0], torch.zeros(1, 2, 3))
    torch.testing.assert_close(submitted["loose_prop"][0][..., 0], torch.tensor([[-10.]]))
    command.clear_object_wrenches()
    assert ("clear_prop", None) in calls
    command.motion_lib.native_object_part_pose_support[0, -1] = False
    with pytest.raises(ValueError, match="loose_prop.*FREE-base pose"):
        command._resample_command(torch.tensor([0]))


def test_native_profile_builds_articulated_and_rigid_managers(tmp_path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from test_native_contract import _scene, write_native_scene_manifest

    from common.native_reference import bind_prepared_native_reference
    from envs import build_manager_cfgs
    from envs.scene.native import bind_native_scene_config

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None,
    ):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
        for count in (2, 1):
            _, mapping = _scene(tmp_path)
            del mapping["objects"][0]["parts"][count:]
            motion_file = write_native_scene_manifest(tmp_path, mapping)
            cfg = compose(config_name="native", overrides=[f"commands.motion_file={motion_file}"])
            bind_prepared_native_reference(cfg)
            cfg.native_tracking_tolerances = {
                "root_position_error_m": 0.03,
                "root_rotation_error_rad": 0.1,
                "part_position_error_m": 0.03,
                "part_rotation_error_rad": 0.1,
                "joint_position_error_rad": 0.1,
                "command_position_error_rad": 0.1,
            }
            OmegaConf.resolve(
                cfg
            )  # Same public composition step as train.run, without AppLauncher.
            layout = bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=(0.0, 0.0, -9.81))
            assert len(layout.objects[0].parts) == count
            managers = build_manager_cfgs(cfg)
            assert not managers["curriculum"]
            assert {"native_command_bc_r", "native_command_bc_l"} <= set(managers["rewards"])
            for name in cfg.rewards.shared_terms:
                assert name in managers["rewards"]
                assert "side" not in managers["rewards"][name].params
            assert "tracking_joint_pos_bc_exp" not in managers["rewards"]
            assert set(managers["observations"]) == {"actor", "critic", "future_traj"}
            assert "robot_links_obj_sdf" not in managers["observations"]["actor"].terms
            assert (
                "native_tip_part_distances" in managers["observations"]["actor"].terms
            )
            assert "native_state_nonfinite" in managers["terminations"]
            assert managers["actions"]["native"].translation_residual_m == 0.1
            assert managers["actions"]["native"].rotation_residual_rad == 1
            assert not managers["commands"]["motion"].object.pin_objects
        stock = compose(
            config_name="flashdexretarget",
            overrides=["commands.motion_file=synthetic.pt"],
        )
        assert stock.actions.name == "residual"
        assert stock.commands.object.pin_objects
        assert "tracking_joint_pos_bc_exp" in stock.rewards.reward_weights


def test_native_binding_admits_clock_phase_and_maps_actual_object_joint_order(tmp_path):
    del tmp_path
    command, env, _ = _command()
    torch.testing.assert_close(
        command.object_joint_reference()["position"], torch.tensor([[0.1, 0.2]])
    )
    # A cropped rounded 60Hz grid differs from a new zero-origin grid by 1ns.
    env.step_dt = 1 / 60
    command.motion_lib.native_time_ns = torch.tensor(
        [9_016_666_667, 9_033_333_333, 9_050_000_000]
    )
    command._bind_native_reference()
    command.motion_lib.native_time_ns[1] += 100
    with TestCase().assertRaisesRegex(ValueError, "cadence"):
        command._bind_native_reference()
    rigid, _, _ = _command(rigid=True)
    assert rigid.object_joint_reference()["position"].shape == (1, 0)
    assert rigid.native_object_joint_pos.shape == (1, 0)


def test_native_bc_uses_installed_independent_reference_and_support(tmp_path):
    del tmp_path
    command, env, _ = _command()
    installed = dict(
        finger_target_rad=torch.tensor([[0.2, 99.0]]),
        finger_source_support=torch.tensor([[True, False]]),
        installed_valid=torch.tensor([True]),
    )
    term = SimpleNamespace(
        installed_control=installed,
        ref_command_pos_used=torch.tensor([[0.2, 0.3]]),
        finger_actuator_limits=torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
        actuator_command_indices=torch.tensor([0, 1]),
        actuator_command_weights=torch.ones(2),
    )
    env.action_manager = SimpleNamespace(_terms={"native": term})
    command.robot.data.joint_pos[:] = (
        999  # BC is accepted action versus its installed reference.
    )
    torch.testing.assert_close(
        NativeMotionTrackingRewards.native_command_bc(env, "motion", 500.0),
        torch.ones(1),
    )
    installed["finger_target_rad"][0, 0] = 0.3
    torch.testing.assert_close(
        NativeMotionTrackingRewards.native_command_bc(env, "motion", 500.0),
        torch.exp(torch.tensor([-2.5])),
    )
    installed["installed_valid"][:] = False
    assert (
        NativeMotionTrackingRewards.native_command_bc(env, "motion", 500.0).item() == 0
    )


def test_native_part_sdf_observations_and_future_masks_use_named_geometry(tmp_path):
    del tmp_path
    command, env, _ = _command()
    command._tip_site_ids = {"left": [0], "right": [1]}
    command.robot.data.site_pos_w = torch.tensor([[[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    command._obj_sdf_extent = 2.0
    axis = torch.linspace(-2, 2, 2)
    grid = torch.zeros(4, 2, 2, 2)
    grid[0] = axis[:, None, None]
    grid[1] = 1
    command._part_sdf = [grid, grid.clone()]
    distance = NativeMotionTrackingObs.native_tip_part_distances(env, "motion")
    torch.testing.assert_close(distance, torch.tensor([[0.2, 0.8, 0.4, 0.6]]))
    normal = NativeMotionTrackingObs.native_tip_part_normals_wrist(
        env, "motion"
    ).reshape(1, 2, 1, 2, 3)
    torch.testing.assert_close(
        normal[0, 0, 0], torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]])
    )
    command.motion_steps[:] = 1
    future = NativeMotionTrackingObs.native_future_reference(
        env, "motion", n_future=2, stride=1
    ).reshape(1, 2, -1)
    assert bool(future[:, 0].abs().any())
    assert not bool(future[:, 1].any())
    torch.testing.assert_close(future[:, 0, -1], torch.tensor([0.02]))


def test_native_reset_preserves_named_states_and_real_zero_width_rigid(tmp_path):
    del tmp_path
    for rigid in (False, True):
        command, env, calls = _command(rigid=rigid)
        env.scene.env_origins[:] = torch.tensor([1.0, 2.0, 3.0])
        command.motion_lib.native_root_state_w["left"][:, 7:] = 99
        command._resample_command(torch.tensor([0]))
        left = next(row for row in calls if row[0] == "left")[1]
        torch.testing.assert_close(left[:, :3], env.scene.env_origins)
        assert not bool(left[:, 7:].any())
        assert sum(row[0] == "object_root" for row in calls) == 1
        assert sum(row[0] == "clear_object" for row in calls) == 1
        joint_rows = [row for row in calls if row[0] == "object_joints"]
        assert len(joint_rows) == (0 if rigid else 1)
        if joint_rows:
            torch.testing.assert_close(joint_rows[0][1], torch.tensor([[0.1, 0.2]]))
        command.motion_lib.native_root_pose_support["left"][0] = False
        with TestCase().assertRaisesRegex(ValueError, "known registered"):
            command._resample_command(torch.tensor([0]))


def test_native_selection_requires_all_part_support_and_finite_actual_state(tmp_path):
    from mjlab.managers import CommandTerm

    from evaluation.shared import _NullTerminationManager

    del tmp_path
    command, env, _ = _command()
    thresholds = {
        "root_position_error_m": 0.03,
        "root_rotation_error_rad": 0.1,
        "part_position_error_m": 0.03,
        "part_rotation_error_rad": 0.1,
        "joint_position_error_rad": 0.1,
        "command_position_error_rad": 0.1,
    }
    command._eval_mode = True
    command.time_left = torch.ones(1)
    command._xfrc_curr_ctrl = None
    # The real Isaac reset calls this public method at dt=0: metrics describe
    # reset row0 while the cursor becomes the first requested endpoint row1.
    CommandTerm.compute(command, dt=0.0)
    assert command.motion_steps.item() == 1
    assert command.native_step_metrics["reference_time_ns"].item() == 1_000_000_000
    callback = NativeReferenceTrackingPerformance("motion", thresholds)
    callback.on_start(env)
    assert callback.rollout_steps == 2  # Three actual states contain two intervals.
    CommandTerm.compute(command, dt=env.step_dt)
    assert command.native_step_metrics["reference_time_ns"].item() == 1_020_000_000
    callback.on_step(env)
    assert not callback.per_env_fail([callback.collect_state()]).any()
    command.motion_lib.native_object_part_pose_support[2, 1] = False
    CommandTerm.compute(command, dt=env.step_dt)
    assert command.native_step_metrics["reference_time_ns"].item() == 1_040_000_000
    callback.on_step(env)
    assert callback.observed_steps.item() == 2
    assert not callback.per_env_scored([callback.collect_state()]).any()
    command.motion_lib.native_object_part_pose_support[2, 1] = True
    # Repeat the real counter phase with a bad final endpoint instead of missing
    # support: it must fail after the first good interval, not be dropped.
    command.motion_steps.zero_()
    CommandTerm.compute(command, dt=0.0)
    callback.on_start(env)
    CommandTerm.compute(command, dt=env.step_dt)
    callback.on_step(env)
    assert not callback.per_env_fail([callback.collect_state()]).any()
    next(iter(command.native_objects.values())).data.body_link_pos_w[:, 1, 0] += 1
    CommandTerm.compute(command, dt=env.step_dt)
    callback.on_step(env)
    assert callback.observed_steps.item() == 2
    assert callback.per_env_fail([callback.collect_state()]).all()
    next(iter(command.native_objects.values())).data.body_link_pos_w[:, 1, 0] -= 1
    # SharedEvalEnv disables training terminations. A passive mimic rate can be
    # invalid without changing any of the supported pose/command errors.
    env.termination_manager = _NullTerminationManager(env.num_envs, env.device)
    command.robot.data.joint_vel[0, 1] = float("nan")
    callback.on_start(env)
    command._update_metrics()
    assert all(
        torch.isfinite(command.native_step_metrics[key]).all() for key in thresholds
    )
    assert not env.termination_manager.compute().any()
    callback.on_step(env)
    assert callback.per_env_fail([callback.collect_state()]).all()
    assert callback.per_env_fail_obj([callback.collect_state()]).all()
    command.robot.data.joint_vel[0, 1] = 0
    command.motion_steps.zero_()
    next(iter(command.native_objects.values())).data.body_link_pos_w[:, 1, 0] += 1
    assert NativeMotionTrackingTerminations.native_reference_diverged(
        env, "motion", 0.2, 1.0, 0.2, 1.0, 0
    ).item()
    next(iter(command.native_objects.values())).data.body_link_pos_w[:, 0, 2] -= 0.2
    assert NativeMotionTrackingTerminations.native_object_dropped(
        env, "motion", 0.1
    ).item()
    command.motion_lib.native_object_part_pose_support[0, 0] = False
    assert not NativeMotionTrackingTerminations.native_object_dropped(
        env, "motion", 0.1
    ).item()
    assert not NativeMotionTrackingTerminations.native_state_nonfinite(
        env, "motion"
    ).item()
    command.robot.data.body_link_lin_vel_w[0, 0, 0] = float("inf")
    assert NativeMotionTrackingTerminations.native_state_nonfinite(env, "motion").item()
    assert "no contact/task qualification" in callback.success_criterion_label


def test_native_object_wrench_clear_uses_actual_bodies_not_rigid_api(tmp_path):
    del tmp_path
    calls = []
    adapter = object.__new__(FreeArticulationAdapter)
    adapter.device = "cpu"
    adapter.body_names = ["object_root", "object_lid"]
    adapter._isaac_body_ids = torch.tensor([3, 1])
    adapter._external_force_w = torch.ones(2, 2, 3)
    adapter._external_torque_w = torch.full((2, 2, 3), 2.0)
    adapter._art = SimpleNamespace(
        num_instances=2,
        set_external_force_and_torque=lambda f, t, **kw: calls.append(
            (f.clone(), t.clone(), kw)
        ),
    )
    adapter.clear_external_wrenches(torch.tensor([1]))
    force, torque, kwargs = calls.pop()
    assert force.shape == (2, 2, 3)
    assert not force[1].any() and not torque[1].any()
    assert force[0].eq(1.0).all() and torque[0].eq(2.0).all()
    assert kwargs["body_ids"] == [3, 1] and kwargs["is_global"] is True
    command, _, _ = _command()
    command._env.scene["object"] = adapter
    command.clear_object_wrenches()
    assert calls[-1][0].shape == (2, 2, 3)
    stock = object.__new__(IsaacMotionTrackingCommand)
    stock._env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={
            "shared": SimpleNamespace(
                write_external_wrench_to_sim=lambda f, t, **kw: calls.append((f, t, kw))
            )
        },
    )
    stock._side_list = ["left", "right"]
    stock.obj_entity_names = lambda side: ["shared"]
    stock.clear_object_wrenches()
    assert len(calls) == 2  # Whole native object + one shared stock rigid object.


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    torch.set_num_threads(1)
    cases = (
        test_native_profile_builds_articulated_and_rigid_managers,
        test_native_binding_admits_clock_phase_and_maps_actual_object_joint_order,
        test_native_bc_uses_installed_independent_reference_and_support,
        test_native_part_sdf_observations_and_future_masks_use_named_geometry,
        test_native_reset_preserves_named_states_and_real_zero_width_rigid,
        test_native_selection_requires_all_part_support_and_finite_actual_state,
        test_native_object_wrench_clear_uses_actual_bodies_not_rigid_api,
    )
    for case in cases:
        with TemporaryDirectory() as directory:
            case(Path(directory))
        print(f"PASS {case.__name__}", flush=True)
    print(
        f"{len(cases)} passed; synthetic software contracts only, no model/FK/Kit/physics",
        flush=True,
    )



def _native_static_gravity_config(tmp_path, backend="native_isaacsim"):
    """Ordinary selected configs and compiled MJCF, without constructing a plant."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    from test_native_contract import _scene

    _, mapping = _scene(tmp_path)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
        cfg = compose(config_name="native", overrides=[f"envs/simulator@simulator={backend}"])
    with open_dict(cfg):
        cfg.robot.native_scene = mapping
    return cfg


@pytest.mark.parametrize("backend", ("native_isaacsim", "mjwarp", "newton"))
def test_native_static_gravity_rejects_curriculum_and_vector_mismatch_before_engine(tmp_path, backend):
    import sys

    from omegaconf import open_dict

    from envs import create_envs

    cfg = _native_static_gravity_config(tmp_path, backend)
    before = set(sys.modules)
    # Both direct constructors are real selected code owners, never invoked.
    # Isaac SDK imports occur after common native admission, so a wrong order
    # also fails this test in the normal native environment without the SDK.
    with patch("simulator.native.mjwarp.MjWarpPlant") as warp, patch("simulator.native.newton.NewtonPlant") as newton:
        with open_dict(cfg):
            cfg.curriculum = {"gravity_curriculum": {"params": {"schedule_steps": 10}}}
        with pytest.raises(ValueError, match="static unassisted recipe; curriculum must be empty"):
            create_envs(cfg, device="cpu")
        with open_dict(cfg):
            cfg.curriculum = {}
            cfg.robot.native_scene.hands[0].root_control.gravity_world_m_per_s2 = [0., 0., -3.]
        with pytest.raises(ValueError, match="hand 'left'.*compiled constructor gravity"):
            create_envs(cfg, device="cpu")
        warp.assert_not_called()
        newton.assert_not_called()
    assert not any(name.startswith(("isaaclab", "isaacsim", "omni.")) for name in set(sys.modules) - before)


def test_native_static_gravity_matches_actual_compiled_xml_and_allows_zero_compensation(tmp_path):
    import xml.etree.ElementTree as ET

    from omegaconf import open_dict

    from envs.scene.native import bind_native_scene_config
    from simulator.mjcf_meta import load_robot_mjcf_meta

    cfg = _native_static_gravity_config(tmp_path)
    path = Path(cfg.robot.native_scene.scene_xml_path)
    tree = ET.parse(path)
    ET.SubElement(tree.getroot(), "option", gravity="-2 1 -3")
    tree.write(path)
    constructor = load_robot_mjcf_meta(path).gravity
    assert tuple(constructor) == (-2., 1., -3.)
    # A configured hand vector cannot silently remain at the old -9.81 recipe.
    with pytest.raises(ValueError, match="compiled constructor gravity"):
        bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=constructor)
    for hand in cfg.robot.native_scene.hands:
        with open_dict(hand.root_control):
            hand.root_control.gravity_world_m_per_s2 = constructor.tolist()
    layout = bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=constructor)
    assert all(tuple(hand.root_control.gravity_world_m_per_s2) == tuple(constructor) for hand in layout.hands)
    with open_dict(cfg.robot.native_scene.hands[0].root_control):
        cfg.robot.native_scene.hands[0].root_control.gravity_compensation = 0.
        cfg.robot.native_scene.hands[0].root_control.gravity_world_m_per_s2 = [0., 0., -9.81]
    layout = bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=constructor)
    assert layout.hands[0].root_control.gravity_compensation == 0.
    assert layout.hands[1].root_control.gravity_compensation == 1.


def test_stock_isaac_curriculum_does_not_enter_native_static_admission():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from envs import create_envs

    class ReachedStockDiscovery(Exception):
        pass

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
        cfg = compose(config_name="flashdexretarget")
    assert cfg.curriculum and cfg.robot.get("native_scene") is None
    with patch("simulator.isaacsim.create._discover_object_slots", side_effect=ReachedStockDiscovery) as discover, \
            pytest.raises(ReachedStockDiscovery):
        create_envs(cfg, device="cpu")
    discover.assert_called_once()


def _configured_native_runner(tmp_path, monkeypatch, command_name):
    import gymnasium as gym
    import numpy as np
    from hydra import compose, initialize_config_dir
    from mjlab.managers import ActionManager, CommandManager, RewardManager
    from omegaconf import OmegaConf
    from test_native_reference import _pipeline_config, _prepared_native_dataset

    import runner.off_policy_runner as runner_module
    from envs import build_actions, build_commands, build_rewards
    from envs.scene.native import bind_native_scene_config
    from retarget.kinematic_retargeting import PreparedNativeInput

    dataset, clip = _prepared_native_dataset(tmp_path)
    PreparedNativeInput(dataset, _pipeline_config(dataset), 1).pack()
    native = clip["native_reference"]
    mesh = str(dataset / "objects/synthetic-mesh/visual.obj")
    from test_native_contract import write_native_scene_manifest

    from common.native_reference import bind_prepared_native_reference

    mapping = dict(
        scene_xml_path="synthetic-not-read.xml",
        hands=[dict(side=side, entity_name=side, root_body_name=body,
            root_joint_name=side + "_free", command_actuator_names=["command_" + side],
            tip_site_names=[side + "_tip"], root_control=dict(position_stiffness=0,
                position_damping=0, rotation_stiffness=0, rotation_damping=0,
                force_limit_n=None, torque_limit_nm=None))
            for side, body in native["root_body_names"].items()],
        objects=(dict(entity_name=next(iter(native["object_root_body_names"])), root_body_name="base",
            root_joint_name="object_free", parts=[dict(name=body, body_name=body, mesh_path=mesh)
                for body in native["object_part_body_names"]]),),
    )
    motion_file = write_native_scene_manifest(dataset, mapping)
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None,
    ):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
        cfg = compose(config_name="native", overrides=[f"commands.name={command_name}",
            f"commands.motion_file={motion_file}", "device=cpu", "num_envs=1",
            f"run_dir={tmp_path / 'run'}"])
        cfg.native_tracking_tolerances = {name: .1 for name in cfg.native_tracking_tolerances}
        cfg.robot.body_mapping = dict(all=[], level1={}, level2={})
        bind_prepared_native_reference(cfg)
        bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=(0., 0., -9.81))
        OmegaConf.resolve(cfg)
    assert cfg.per_hand_actor_critic and cfg.runner.agent.reward_heads.enable

    names = ("left_a", "left_mimic", "right_b")
    command_joints = {"command_left": "left_a", "command_right": "right_b"}
    robot = SimpleNamespace(num_joints=len(names), joint_names=names,
        body_names=("left_root", "right_root"),
        find_sites=lambda requested, preserve_order: ([0], requested),
        find_joints_by_actuator_names=lambda requested: (
            [names.index(command_joints[n]) for n in requested], [command_joints[n] for n in requested]),
        native_command_limits=lambda requested: np.tile([-1., 1.], (len(requested), 1)),
        data=SimpleNamespace(joint_pos=torch.zeros(1, 3), joint_vel=torch.zeros(1, 3),
            joint_pos_target=torch.zeros(1, 3), body_link_pos_w=torch.zeros(1, 2, 3),
            body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]] * 2]),
            body_link_lin_vel_w=torch.zeros(1, 2, 3), body_link_ang_vel_w=torch.zeros(1, 2, 3)))
    object_body_names = ("base", "lid")
    # The native getter reads WORLD body-link origins in entity order; the
    # prepared part order is deliberately lid/base rather than base/lid.
    part_columns = [native["object_part_body_names"].index(name) for name in object_body_names]
    object_state = torch.as_tensor(
        native["object_part_state_w"][:1, part_columns], dtype=torch.float32,
    ).clone()
    obj = SimpleNamespace(joint_names=("hinge_a", "hinge_b"), body_names=object_body_names,
        data=SimpleNamespace(joint_pos=torch.zeros(1, 2), joint_vel=torch.zeros(1, 2),
            body_link_pos_w=object_state[..., :3], body_link_quat_w=object_state[..., 3:7],
            body_link_lin_vel_w=object_state[..., 7:10], body_link_ang_vel_w=object_state[..., 10:13]))
    _declare_joint_coordinates(obj, ("hinge",) * len(obj.joint_names))
    _declare_joint_coordinates(robot, ("hinge",) * len(robot.joint_names))

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=.02,
        scene=Scene(robot=robot, articulated_object=obj), cfg=SimpleNamespace(),
        episode_length_buf=torch.zeros(1, dtype=torch.long))
    # The name route does not depend on mesh queries, a solver, or a learned actor.
    monkeypatch.setattr("utils.sdf.bake_object_sdf_grid", lambda *_a, **_kw: torch.zeros(4, 2, 2, 2))
    env.command_manager = CommandManager(build_commands(cfg.commands), env)
    command = env.command_manager.get_term(command_name)
    assert type(command) is NativeMotionTrackingCommand
    assert env.command_manager.active_terms == [command_name]
    env.action_manager = ActionManager(build_actions(cfg.actions), env)
    env.reward_recipe = cfg.rewards
    env.reward_manager = RewardManager(build_rewards(cfg.rewards, command_name=command_name,
        entity_name=cfg.robot.entity_name), env)
    observations = {name: torch.zeros(1, 2) for name in ("actor", "critic")}
    env.single_observation_space = gym.spaces.Dict({name: gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
                                                  for name in observations})
    env.single_action_space = gym.spaces.Box(-np.inf, np.inf, (env.action_manager.total_action_dim,), np.float32)
    env.observation_manager = SimpleNamespace(active_terms={name: ["synthetic"] for name in observations},
        group_obs_term_dim={name: [(2,)] for name in observations})
    env.reset = lambda: (observations, {})
    received = []

    def build_agent(_cfg, flat, info, name):
        received.append((flat, info, name))
        return SimpleNamespace()

    monkeypatch.setattr(runner_module, "build_agent", build_agent)
    monkeypatch.setattr(runner_module.wandb, "run", None)
    runner = runner_module.OffPolicyRunner(SimpleNamespace(unwrapped=env), cfg.runner, cfg.eval,
                                         eval_env=SimpleNamespace())
    return runner, command, env, received


@pytest.mark.parametrize("command_name", ["native_transfer", "motion"])
def test_configured_native_command_routes_real_runner_heads_and_early_reset(tmp_path, monkeypatch, command_name):
    from omegaconf import OmegaConf

    from envs.terminations import build_terminations

    runner, command, env, received = _configured_native_runner(tmp_path, monkeypatch, command_name)
    flat, info, used_name = received[0]
    assert runner._train_cmd is command and flat is runner._train_env
    assert used_name == command_name and info["reward_heads"] == command._side_list
    action = next(iter(env.action_manager._terms.values()))
    for side, indices in info["action_sides"].items():
        assert indices == [i for i, owner in enumerate(action.action_sides) if owner == side]
    env.reward_manager._step_reward[:] = torch.arange(1, len(env.reward_manager.active_terms) + 1)
    total = env.reward_manager._step_reward.sum(-1) * env.step_dt
    heads = flat._head_rewards(total)
    torch.testing.assert_close(heads.sum(-1), total)

    # This stock term also accepts the configured command through its existing builder.
    terms = build_terminations(OmegaConf.create(dict(
        _target_="envs.terminations.motion_tracking.MotionTrackingTerminations",
        task_reward_early_reset=dict(params=dict(task_term="native_root_pose", threshold=.6, interval=5)),
    )), command_name=command_name, entity_name="robot")
    term = terms["task_reward_early_reset"]
    assert term.params["command_name"] == command_name
    from envs.rewards import side_twin_name
    indices = [env.reward_manager.active_terms.index(side_twin_name("native_root_pose", side[0]))
               for side in command._side_list]
    weights = torch.tensor([env.reward_manager.get_term_cfg(env.reward_manager.active_terms[index]).weight
                            for index in indices])
    for rate, expected in ((.7, False), (.1, True)):
        env.reward_manager._step_reward[:, indices] = rate * weights
        for step in range(1, 7):
            env.episode_length_buf[:] = step
            fired = term.func(env, **term.params)
            assert fired.tolist() == [expected and step == 6]


@pytest.mark.parametrize("sides", [("right", "left"), ("left", "right")])
def test_reward_head_ownership_uses_metadata_not_term_spelling(sides):
    from mjlab.managers import RewardManager
    from mjlab.managers.reward_manager import RewardTermCfg

    from envs.vec_env import FlatObsVecEnv

    def local(env, side):
        del side
        return torch.ones(env.num_envs)

    def shared(env):
        return torch.ones(env.num_envs)

    # Deliberately misleading names must not override the builder's real side.
    terms = {
        "misleading_l_name": RewardTermCfg(func=local, weight=1., params={"side": "right"}),
        "no_side_token": RewardTermCfg(func=local, weight=1., params={"side": "left"}),
        "shared_r_name": RewardTermCfg(func=shared, weight=1., params={}),
    }
    action = SimpleNamespace(action_names=("l_command", "r_command"),
                             action_sides=("left", "right"), action_dim=2)
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=.02,
                          action_manager=SimpleNamespace(_terms={"native": action}))
    env.reward_manager = RewardManager(terms, env)
    wrapper = object.__new__(FlatObsVecEnv)
    wrapper._env, wrapper._device = env, "cpu"
    wrapper._setup_reward_heads({}, SimpleNamespace(_side_list=list(sides)))
    env.reward_manager._step_reward[:] = torch.tensor([2., 6., 10.])
    total = env.reward_manager._step_reward.sum(-1) * env.step_dt
    heads = wrapper._head_rewards(total)
    expected = {"right": 7., "left": 11.}
    torch.testing.assert_close(heads, torch.tensor([[expected[side] * env.step_dt for side in sides]]))
    torch.testing.assert_close(heads.sum(-1), total)
    # The existing explicit shared override is still authoritative.
    wrapper._setup_reward_heads({"misleading_l_name": "both"}, SimpleNamespace(_side_list=list(sides)))
    torch.testing.assert_close(wrapper._head_rewards(total).sum(-1), total)


def test_native_reward_routing_preserves_scalar_under_support_and_single_hand_changes(tmp_path, monkeypatch):
    from mjlab.managers import RewardManager
    from omegaconf import OmegaConf

    from envs import build_rewards

    runner, command, env, _ = _configured_native_runner(tmp_path, monkeypatch, "native_credit")
    flat = runner._train_env
    action = next(iter(env.action_manager._terms.values()))
    env.reset_terminated = torch.zeros(env.num_envs, dtype=torch.bool)
    scalar_cfg = OmegaConf.create(OmegaConf.to_container(env.reward_recipe, resolve=True))
    scalar_cfg.split_sides = False
    scalar_manager = RewardManager(build_rewards(scalar_cfg, command_name="native_credit", entity_name="robot"), env)
    assert set(scalar_manager.active_terms) == set(env.reward_recipe.reward_weights)
    installed = action.installed_control
    action.ref_command_pos_used = command.command_reference()["position"].clone()
    installed["finger_target_rad"][:] = action.ref_command_pos_used[:, action.actuator_command_indices]
    installed["finger_source_support"][:] = True
    installed["installed_valid"][:] = True
    for side in command._side_list:
        body = command.robot.body_names.index(side + "_root")
        slot = command._side_list.index(side)
        command.robot.data.body_link_pos_w[:, body] = command.ref_native_root_state_w[:, slot, :3]
        command.robot.data.body_link_quat_w[:, body] = command.ref_native_root_state_w[:, slot, 3:7]

    def read():
        total = env.reward_manager.compute(env.step_dt).clone()
        expected = scalar_manager.compute(env.step_dt).clone()
        heads = flat._head_rewards(total).clone()
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(heads.sum(-1), expected)
        return heads

    before = read()
    moved, other = command._side_list
    body = command.robot.body_names.index(moved + "_root")
    command.robot.data.body_link_pos_w[:, body, 0] += .1
    after = read()
    torch.testing.assert_close(after[:, flat.reward_head_names.index(other)], before[:, flat.reward_head_names.index(other)])
    assert bool((after[:, flat.reward_head_names.index(moved)] < before[:, flat.reward_head_names.index(moved)]).all())

    # A missing root and unequal finger support must not silently halve global reward.
    command.motion_lib.native_root_pose_support[moved][command._motion_flat_ids] = False
    owners = dict(zip(action.action_names, action.action_sides, strict=True))
    finger_columns = [i for i, name in enumerate(action.finger_actuator_names) if owners[name] == moved]
    installed["finger_source_support"][:, finger_columns] = False
    before = read()
    supported_columns = [i for i, name in enumerate(action.finger_actuator_names) if owners[name] == other]
    installed["finger_target_rad"][:, supported_columns] += .1
    after = read()
    torch.testing.assert_close(after[:, flat.reward_head_names.index(moved)], before[:, flat.reward_head_names.index(moved)])
    assert bool((after[:, flat.reward_head_names.index(other)] < before[:, flat.reward_head_names.index(other)]).all())

    # A shared global failure remains one penalty, divided across heads once.
    before = read()
    env.reset_terminated[:] = True
    after = read()
    penalty = env.reward_recipe.reward_weights.termination_penalty * env.step_dt / len(command._side_list)
    torch.testing.assert_close(after - before, torch.full_like(after, penalty))


@pytest.mark.parametrize("sides", [("right", "left"), ("left", "right")])
def test_native_action_credit_conserves_total_with_unequal_hand_widths(sides):
    # A supported native chart with different declared finger counts per hand.
    root_names = [f"{side}_{kind}_{axis}" for kind in ("translation", "rotation")
                  for side in sides for axis in "xyz"]
    root_sides = [side for _kind in ("translation", "rotation") for side in sides for _axis in "xyz"]
    fingers = ("left_a", "right_a", "right_b")
    names = (*root_names, *fingers)
    owners = (*root_sides, "left", "right", "right")
    action = SimpleNamespace(action_names=names, action_sides=owners)
    values = torch.arange(1, len(names) + 1, dtype=torch.float32)[None]
    env = SimpleNamespace(device="cpu", action_manager=SimpleNamespace(action=values, _terms={"native": action}),
                          command_manager=SimpleNamespace(get_term=lambda _name: SimpleNamespace(_side_list=list(sides))))
    global_value = NativeMotionTrackingRewards.action_penalty(env, "native_credit")
    local = [NativeMotionTrackingRewards.action_penalty(env, "native_credit", side) / len(sides) for side in sides]
    torch.testing.assert_close(sum(local), global_value)
    for side, value in zip(sides, local, strict=True):
        expected = sum(values[:, i].square() for i, owner in enumerate(owners) if owner == side) / len(names)
        torch.testing.assert_close(value, expected)


def test_stock_reward_twins_keep_metadata_and_full_termination_weights():
    from hydra import compose, initialize_config_dir

    from envs import build_rewards
    from envs.rewards import side_twin_name

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        cfg = compose(config_name="flashdexretarget", overrides=["commands.motion_file=synthetic-not-read.pt"])
        terms = build_rewards(cfg.rewards)
        ordinary = "tracking_ref_mano_wrist_trans_error_exp"
        for side in ("left", "right"):
            twin = terms[side_twin_name(ordinary, side[0])]
            assert twin.params["side"] == side
            assert twin.weight == cfg.rewards.reward_weights[ordinary] / 2
            failure = terms[side_twin_name("termination_penalty", side[0])]
            assert failure.params["side"] == side
            assert failure.weight == cfg.rewards.reward_weights.termination_penalty

    command, env, _ = _command()
    env.reset_terminated = torch.ones(env.num_envs, dtype=torch.bool)
    env._side_fail = {"actual_failure_owner": torch.tensor([[True, False]])}
    for side, expected in zip(command._side_list, (1., 0.), strict=True):
        term = terms[side_twin_name("termination_penalty", side[0])]
        torch.testing.assert_close(term.func(env, **term.params), torch.tensor([expected]))
    # Original author semantics charge both full weights for an unattributed failure.
    env._side_fail = {}
    weighted = sum(terms[side_twin_name("termination_penalty", side[0])].func(
        env, **terms[side_twin_name("termination_penalty", side[0])].params
    ) * terms[side_twin_name("termination_penalty", side[0])].weight for side in command._side_list)
    torch.testing.assert_close(weighted, torch.tensor([2 * cfg.rewards.reward_weights.termination_penalty]))


def _full_name_body_fixture(tmp_path, root_mode):
    """Compile tiny named geometry and exercise the selected real setup/FK owners.

    No plant, SDF, complete task or policy is constructed. Live state facts are
    explicit analytic points; no integration is performed.
    """
    import mujoco
    import numpy as np
    from mjlab.managers import CommandTerm

    from envs.commands.motion_tracking.motion_tracking import MotionTrackingCommand
    from retarget.human_demo import clips as C

    sides = ("right", "left")
    prefix = {"left": "Alpha", "right": "receiver-B"}
    origin = {"left": np.array((.4, .2, .8)), "right": np.array((-.3, -.1, 1.1))}
    angles = {"left": np.array((.25, -.4)), "right": np.array((-.3, .6))}
    rates = {"left": np.array((.7, -.5)), "right": np.array((-.6, .8))}
    native = root_mode == "native-free"

    def hand_xml(side):
        name = prefix[side]
        position = " ".join(str(x) for x in origin[side])
        free = f'<freejoint name="{name}/root"/>' if native else ""
        return f'''<body name="{name}/base" pos="{position}">
          {free}<geom type="sphere" size=".03" mass=".1"/>
          <site name="{name}/palm"/>
          <site name="{name}/thumb-tip" pos="0 .1 0"/>
          <body name="{name}/prox">
            <joint name="{name}/hinge" axis="0 0 1"/>
            <geom type="sphere" size=".02" mass=".02"/>
            <body name="{name}/dist" pos=".2 0 0">
              <geom type="sphere" size=".02" mass=".02"/>
              <site name="{name}/tip" pos=".1 0 0"/>
            </body>
          </body>
        </body>'''

    xml = '<mujoco><worldbody>' + ''.join(hand_xml(side) for side in sides)
    xml += '''<body name="object" pos="0 0 2"><freejoint name="object-free"/>
      <geom type="sphere" size=".02" mass=".1"/></body></worldbody><actuator>'''
    xml += ''.join(f'<position name="{prefix[side]}/drive" joint="{prefix[side]}/hinge" kp="3" kv=".1"/>'
                   for side in sides) + '</actuator></mujoco>'
    path = tmp_path / "namespaced.xml"
    path.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(path))
    body_names = tuple(reversed([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                                for i in range(1, model.nbody)]))
    joint_names = tuple(reversed([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                                 for i in range(model.njnt)
                                 if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]))
    site_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
                       for i in range(model.nsite))
    root_states, expected = {}, {}
    for si, side in enumerate(sides):
        state = np.zeros((2, 13))
        state[:, :3] = origin[side]
        theta = np.zeros(2)
        if native:
            state[:, :3] += np.array(((.01, 0., 0.), (.03, -.01, .02))) * (si + 1)
            theta = np.array((.2, -.1)) * (1. if side == "left" else -1.)
            state[:, 7:10] = (.05 * (si + 1), -.02, .01)
            state[:, 12] = .4 if side == "left" else -.2
        state[:, 3], state[:, 6] = np.cos(theta / 2), np.sin(theta / 2)
        root_states[side] = torch.from_numpy(state)
        total = theta + angles[side]
        distal_offset = .2 * np.stack((np.cos(total), np.sin(total), np.zeros(2)), -1)
        index_offset = distal_offset * 1.5
        thumb_offset = .1 * np.stack((-np.sin(theta), np.cos(theta), np.zeros(2)), -1)
        total_omega = np.zeros((2, 3))
        total_omega[:, 2] = state[:, 12] + rates[side]
        root_omega = np.zeros((2, 3))
        root_omega[:, 2] = state[:, 12]
        expected[side] = dict(
            root=state[:, :3], quaternion=state[:, 3:7],
            all_position=np.stack((state[:, :3] + distal_offset, state[:, :3]), 1),
            all_velocity=np.stack((state[:, 7:10] + np.cross(total_omega, distal_offset), state[:, 7:10]), 1),
            tip_position=np.stack((state[:, :3] + index_offset, state[:, :3] + thumb_offset), 1),
            tip_velocity=np.stack((state[:, 7:10] + np.cross(total_omega, index_offset),
                                   state[:, 7:10] + np.cross(root_omega, thumb_offset)), 1),
        )
    column_side = {prefix[side] + "/hinge": side for side in sides}
    q = torch.tensor(np.stack([angles[column_side[name]] for name in joint_names], -1), dtype=torch.float32)
    v = torch.tensor(np.stack([rates[column_side[name]] for name in joint_names], -1), dtype=torch.float32)
    live_pos, live_vel = torch.zeros(1, len(body_names), 3), torch.zeros(1, len(body_names), 3)
    for side in sides:
        for suffix, slot in (("dist", 0), ("prox", 1)):
            column = body_names.index(prefix[side] + "/" + suffix)
            live_pos[0, column] = torch.tensor(expected[side]["all_position"][0, slot], dtype=torch.float32)
            live_vel[0, column] = torch.tensor(expected[side]["all_velocity"][0, slot], dtype=torch.float32)
    robot = SimpleNamespace(body_names=body_names, joint_names=joint_names, site_names=site_names,
        cfg=SimpleNamespace(spec_fn=lambda: mujoco.MjSpec.from_file(str(path))),
        find_sites=lambda names, preserve_order: ([site_names.index(name) for name in names], list(names)),
        data=SimpleNamespace(body_link_pos_w=live_pos, body_link_lin_vel_w=live_vel))
    distal = {side: prefix[side] + "/dist" for side in sides}
    proximal = {side: prefix[side] + "/prox" for side in sides}
    mapping = dict(all=[[distal, "index_distal"], [proximal, "index_proximal"]],
                   level1={"index": [proximal, "index_proximal"]}, level2={},
                   level3={"index": [distal, "index_distal"]})
    cfg = SimpleNamespace(body_mapping=mapping, object=SimpleNamespace(entity_names=None),
        site_names={"palm": {side: prefix[side] + "/palm" for side in sides},
            "tip": {side: [prefix[side] + "/tip", prefix[side] + "/thumb-tip"] for side in sides}})
    cfg.site_names["contact"] = cfg.site_names["tip"]
    mano_names = list(reversed(C.MANO_JOINT_NAMES))
    mano = torch.arange(2 * len(mano_names) * 3, dtype=torch.float32).reshape(2, len(mano_names), 3) / 100
    library = SimpleNamespace(hand_sides=sides, robot_joint_pos=q, robot_joint_vel=v,
        length_starts=torch.tensor([0]), mano_joint_names={side: mano_names for side in sides},
        mano_joint_pos={side: mano.clone() for side in sides}, native_root_state_w=root_states)
    cls = NativeMotionTrackingCommand if native else MotionTrackingCommand
    command = object.__new__(cls)
    env = SimpleNamespace(device="cpu", num_envs=1, scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
    CommandTerm.__init__(command, cfg, env)
    command.robot, command.motion_lib = robot, library
    command._side_list, command.finger_names = list(sides), ("index", "thumb")
    command.motion_ids, command.motion_steps = torch.tensor([0]), torch.tensor([0])
    if native:
        command.native_layout = NativeSceneCfg.from_mapping(dict(scene_xml_path=str(path),
            hands=[dict(side=side, entity_name=side, root_body_name=prefix[side] + "/base",
                root_joint_name=prefix[side] + "/root", command_actuator_names=[prefix[side] + "/drive"],
                root_control=dict(position_stiffness=0, position_damping=0, rotation_stiffness=0,
                    rotation_damping=0, force_limit_n=None, torque_limit_nm=None)) for side in sides],
            objects=(dict(entity_name="object", root_body_name="object", root_joint_name="object-free",
                parts=[dict(name="body", body_name="object", mesh_path="not-queried.obj")]),)))
    command._init_site_ids(cfg)
    command._init_robot_body_ids(cfg)
    command._init_mano_body_ids(cfg)
    command._init_metrics()
    command._compute_robot_ref_state()
    return command, expected


@pytest.mark.parametrize("root_mode", ["stock-sites", "native-free"])
def test_explicit_body_correspondence_joins_live_and_reference_fk(tmp_path, root_mode):
    import numpy as np

    command, expected = _full_name_body_fixture(tmp_path, root_mode)
    assert command._levels() == (1, 3)  # Missing thumb level3 is not invented.
    if root_mode == "native-free":
        assert set(command.metrics) == {"native_root_position_error_m", "native_part_position_error_m"}
    else:
        assert "error_level3_l" in command.metrics and "error_level3_r" in command.metrics
        assert "error_level2_l" not in command.metrics
    for si, side in enumerate(command._side_list):
        np.testing.assert_allclose(command._robot_ref_all_pos[:, si], expected[side]["all_position"], atol=1e-6)
        np.testing.assert_allclose(command._robot_ref_all_lin_vel[:, si], expected[side]["all_velocity"], atol=1e-6)
        np.testing.assert_allclose(command._robot_ref_tip_pos[:, si], expected[side]["tip_position"], atol=1e-6)
        np.testing.assert_allclose(command._robot_ref_tip_lin_vel[:, si], expected[side]["tip_velocity"], atol=1e-6)
        np.testing.assert_allclose(command._robot_ref_wrist_pos[:, si], expected[side]["root"], atol=1e-6)
        np.testing.assert_allclose(command._robot_ref_wrist_quat[:, si], expected[side]["quaternion"], atol=1e-6)
        np.testing.assert_allclose(command._robot_ref_level1_pos[:, si, 0], expected[side]["root"], atol=1e-6)
        np.testing.assert_allclose(command.robot_level_trans_w(side, 3)[0, 0],
                                   expected[side]["all_position"][0, 0], atol=1e-6)
        mano_index = command.motion_lib.mano_joint_names[side].index("index_distal")
        torch.testing.assert_close(command.mano_level_trans_w(side, 3),
                                   command.motion_lib.mano_joint_pos[side][:1, [mano_index]])
        assert len(command._levelmid_body_ids[side]) == len(command._levelmid_mano_ids[side]) == 1


@pytest.mark.parametrize("profile", ["xhand_bimanual", "sharpa_bimanual", "inspire_rh56f1_bimanual"])
def test_registered_body_correspondence_preserves_literal_asset_names(profile):
    """Check shared name admission and stock geometry metrics, not native task scoring."""
    import xml.etree.ElementTree as ET

    from mjlab.managers import CommandTerm
    from omegaconf import OmegaConf

    from envs.commands.motion_tracking.motion_tracking import MotionTrackingCommand
    from retarget.human_demo import clips as C

    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "config/envs/robot" / (profile + ".yaml")).robot
    suffixes = {
        "xhand_bimanual": ("hand_thumb_bend_link", "hand_thumb_rota_link1", "hand_thumb_rota_link2",
            "hand_index_bend_link", "hand_index_rota_link1", "hand_index_rota_link2", "hand_mid_link1",
            "hand_mid_link2", "hand_ring_link1", "hand_ring_link2", "hand_pinky_link1", "hand_pinky_link2"),
        "sharpa_bimanual": ("thumb_MC", "thumb_PP", "thumb_DP", "index_MCP_VL", "index_MP", "index_DP",
            "middle_MCP_VL", "middle_MP", "middle_DP", "ring_MCP_VL", "ring_MP", "ring_DP",
            "pinky_MCP_VL", "pinky_MP", "pinky_DP"),
        "inspire_rh56f1_bimanual": ("thumb_4", "index_1", "index_2", "middle_2", "ring_2", "little_1", "little_2"),
    }
    roles = {
        "xhand_bimanual": ("thumb_proximal", "thumb_intermediate", "thumb_distal", "index_proximal",
            "index_proximal", "index_distal", "middle_proximal", "middle_distal", "ring_proximal",
            "ring_distal", "pinky_proximal", "pinky_distal"),
        "sharpa_bimanual": tuple(f"{finger}_{part}" for finger in C.FINGERS
                                 for part in ("proximal", "intermediate", "distal")),
        "inspire_rh56f1_bimanual": ("thumb_distal", "index_proximal", "index_intermediate",
            "middle_intermediate", "ring_intermediate", "pinky_proximal", "pinky_intermediate"),
    }
    expected_levels = {"xhand_bimanual": (1, 2), "sharpa_bimanual": (1, 2, 3),
                       "inspire_rh56f1_bimanual": ()}[profile]
    if profile == "inspire_rh56f1_bimanual":
        # The external composed model is not reopened for this registration regression.
        body_names = list(dict.fromkeys(name for names in cfg.contact_links.values() for name in names))
    else:
        asset = "xhand" if profile == "xhand_bimanual" else "sharpa"
        body_names = [node.attrib["name"] for node in ET.parse(root / "assets/robot" / asset / "bimanual.xml").iter("body")]
    robot = SimpleNamespace(body_names=list(reversed(body_names)))
    source_names = list(reversed(C.MANO_JOINT_NAMES))
    library = SimpleNamespace(hand_sides=("right", "left"),
                              mano_joint_names={side: source_names for side in ("left", "right")})
    command = object.__new__(MotionTrackingCommand)
    CommandTerm.__init__(command, SimpleNamespace(body_mapping=cfg.body_mapping,
        object=SimpleNamespace(entity_names=None)), SimpleNamespace(device="cpu", num_envs=1))
    command.robot, command.motion_lib = robot, library
    command._side_list, command.finger_names = list(library.hand_sides), tuple(cfg.body_names.fingers)
    command._init_robot_body_ids(command.cfg)
    command._init_mano_body_ids(command.cfg)
    command._init_metrics()
    assert command._levels() == expected_levels
    for side in library.hand_sides:
        assert [robot.body_names[i] for i in command._all_body_ids[side]] == [
            f"{side}_{suffix}" for suffix in suffixes[profile]]
        assert [source_names[i] for i in command._all_mano_ids[side]] == list(roles[profile])
        for level in (1, 2, 3):
            field = {1: "_level1_body_ids", 2: "_level2_body_ids", 3: "_levelmid_body_ids"}[level]
            assert len(getattr(command, field).get(side, ())) == (5 if level in expected_levels else 0)
            assert (f"error_level{level}_{side[0]}" in command.metrics) == (level in expected_levels)


def _source_geometry_command(widths=(("right", 3), ("left", 5))):
    """Synthetic observed points/physical sites and real query owners; no solver."""
    from dataclasses import replace

    from retarget.human_demo import clips as C

    command, env, calls = _command()
    command._side_list = [side for side, _ in widths]
    command.native_layout = replace(command.native_layout, hands=tuple(
        hand for side, _ in widths for hand in command.native_layout.hands if hand.side == side))
    command._finger_joint_ids = torch.tensor([0 if side == "left" else 2 for side, _ in widths])
    command._root_body_ids = [0 if side == "left" else 1 for side, _ in widths]
    env.scene.env_origins = torch.tensor([[2., -1., .5]])
    command.robot.data.body_link_pos_w += env.scene.env_origins[:, None]
    next(iter(command.native_objects.values())).data.body_link_pos_w += env.scene.env_origins[:, None]
    command.robot.data.body_link_quat_w[:] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    names = list(reversed(C.MANO_JOINT_NAMES))
    ml = command.motion_lib
    ml.native_time_ns += 2**60
    ml.mano_joint_names = {side: names for side, _ in widths}
    ml.tip_ids, ml.mano_joint_pos, ml.native_source_point_position_support = {}, {}, {}
    requested = []
    for side, width in widths:
        columns = [names.index(f"{finger}_tip") for finger in reversed(C.FINGERS[:width])]
        ml.tip_ids[side] = torch.tensor(columns)
        points = torch.zeros(3, len(names), 3)
        points[..., 0] = torch.arange(3)[:, None] * .1 + torch.arange(len(names))[None] * .003
        points[..., 1] = torch.arange(len(names))[None] * .002
        ml.mano_joint_pos[side] = points
        ml.native_source_point_position_support[side] = torch.ones(3, len(names), dtype=torch.bool)
        requested.extend((side, i, f"physical/{side}/site-{i}") for i in range(width))
    command.robot.site_names = tuple(name for _, _, name in reversed(requested))
    command.robot.data.site_pos_w = torch.zeros(1, len(requested), 3)
    command._tip_site_ids = {side: [] for side, _ in widths}
    for side, point, name in requested:
        column = command.robot.site_names.index(name)
        command._tip_site_ids[side].append(column)
        source = ml.mano_joint_pos[side][1, ml.tip_ids[side][point]]
        shift = .025 if side == "right" else .06
        command.robot.data.site_pos_w[0, column] = source + env.scene.env_origins[0] + torch.tensor([shift, .01, 0.])
    command.motion_steps[:] = 1
    command.reference_phase = dict(motion_ids=torch.tensor([0]), motion_steps=torch.tensor([0]))
    installed = dict(installed_valid=torch.tensor([True]), reference_motion_ids=torch.tensor([0]),
                     reference_motion_steps=torch.tensor([1]))
    term = SimpleNamespace(installed_control=installed)
    env.action_manager = SimpleNamespace(_terms={"configured_action": term})
    env._sim_step_counter, command._obj_state_version = 0, 0
    command._obj_sdf_extent = 2.
    axis = torch.linspace(-2, 2, 2)
    grid = torch.zeros(4, 2, 2, 2)
    grid[0], grid[1] = axis[:, None, None], 1.
    command._part_sdf = [grid.clone() for _ in command.native_layout.objects[0].parts]
    return command, env, installed, calls


def _source_geometry_observations(env):
    from mjlab.managers import ObservationManager
    from omegaconf import OmegaConf

    from envs import build_observations

    recipe = OmegaConf.create(dict(_target_="envs.observations.native.NativeMotionTrackingObs",
        obs_dict=dict(actor_obs=["native_source_tip_reference_error"],
                      critic_obs=["native_source_tip_reference_error"])))
    return ObservationManager(build_observations(recipe, command_name="motion"), env)


def _wrist_control_case(widths=(("left", 5), ("right", 5)), **action_options):
    """Real action/observation owners over fixed synthetic source/physical facts."""
    from dataclasses import asdict

    from mjlab.utils.lab_api.string import resolve_matching_names

    from envs.actions.native import NativeResidualAction, NativeResidualActionCfg

    command, env, _, _ = _source_geometry_command(widths)
    robot = command.robot
    robot.data.joint_pos_target = torch.zeros_like(robot.data.joint_pos)
    robot.data.site_lin_vel_w = torch.zeros_like(robot.data.site_pos_w)
    robot.num_joints = len(robot.joint_names)
    robot.find_joints = lambda names, preserve_order=False: resolve_matching_names(
        names, robot.joint_names, preserve_order=preserve_order)
    joint_ids = {"left": 0, "right": 2}
    actuators = {name: joint_ids[hand.side] for hand in command.native_layout.hands
                 for name in hand.command_actuator_names}
    robot.native_command_limits = lambda names: torch.tensor([[0., 1.]] * len(names))
    robot.find_joints_by_actuator_names = lambda names: (
        [actuators[name] for name in names], [robot.joint_names[actuators[name]] for name in names])
    robot.set_joint_position_target = lambda value, ids: robot.data.joint_pos_target.__setitem__(
        (slice(None), ids), value)
    for hand in command.native_layout.hands:
        robot.hands[hand.side].apply_root_pose_target = lambda target, support, control: (
            torch.zeros_like(target[:, :3]), torch.zeros_like(target[:, :3]))
    options = dict(
        entity_name="robot", command_name="motion", native_scene=asdict(command.native_layout),
        translation_residual_m=.1, rotation_residual_rad=1.,
        cutoff_hz={"wrist_trans": 5., "wrist_rot": 5.}, clip={".*": (-1., 1.)})
    action = NativeResidualAction(NativeResidualActionCfg(**{**options, **action_options}), env)
    command._finger_joint_ids = action._finger_ids.clone()
    env.action_manager = SimpleNamespace(_terms={"configured_action": action})
    return command, env, action


def test_native_wrist_approach_releases_from_unsupported_or_unloaded_source_parts():
    command, env, action = _wrist_control_case((("right", 3), ("left", 5)), translation_residual_m=.04,
        source_contact_approach=dict(translation_m=.024, max_part_position_error_m=.02,
            max_part_rotation_error_rad=math.radians(20.), max_tip_distance_m=.042))
    library = command.motion_lib
    library.hand_sides = tuple(command._side_list)
    active = torch.zeros(3, 2, 2, dtype=torch.bool)
    active[1, 0, 1] = True
    library.native_source_hand_part_intent = dict(active=active, known=torch.ones_like(active))
    library.native_source_hand_part_intent_provenance = []
    target = command.object_part_reference(command._motion_flat_ids)
    object_data = next(iter(command.native_objects.values())).data
    object_data.body_link_pos_w.copy_(target[..., :3])
    object_data.body_link_quat_w.copy_(target[..., 3:7])

    distance = torch.full((1, 8, 2), .01)
    distance[:, :, 0] = .001
    direction = torch.zeros(1, 8, 2, 3)
    direction[..., 0, 1] = 1.
    direction[..., 1, 0] = 1.
    support = torch.ones_like(distance, dtype=torch.bool)
    command.sdf_query_parts = lambda *args, **kwargs: (distance, direction, support)
    pairs = {f"{hand.side}/{obj.entity_name}/{part.name}": {"normal_force_n": torch.zeros(1)}
             for hand in command.native_layout.hands for obj, part in command.native_layout.part_bindings}
    env.scene.hand_part_contact_facts = lambda: {"pairs": pairs, "measurement_support": torch.ones(1, dtype=torch.bool)}
    raw = torch.zeros(1, action.action_dim)

    def residual():
        action.process_actions(raw)
        action.apply_actions()
        installed = action.installed_control
        assert torch.equal(installed["reference_motion_steps"], command.motion_steps)
        return installed["root_state_w"][..., :3] - command.ref_native_root_state_w[..., :3]

    def released():
        for _ in range(32):
            remaining = residual()
        torch.testing.assert_close(remaining, torch.zeros_like(remaining), atol=1e-6, rtol=0.)

    torch.testing.assert_close(residual()[0], torch.tensor([[-.0024, 0., 0.], [0., 0., 0.]]))
    object_data.body_link_pos_w[0, 1, 2] -= .025
    released()
    object_data.body_link_pos_w.copy_(target[..., :3])
    torch.testing.assert_close(residual()[0, 0], torch.tensor([-.0024, 0., 0.]))
    object_data.body_link_quat_w[0, 1] = torch.tensor([math.cos(.2), 0., math.sin(.2), 0.])
    released()
    object_data.body_link_quat_w.copy_(target[..., 3:7])
    assert residual().any()
    distance[:, :, 1] = .05
    released()
    distance[:, :, 1] = .01
    assert residual().any()
    support[:, :, 1] = False
    released()
    support[:, :, 1] = True
    assert residual().any()
    active[1, 0, 1] = False
    released()
    active[1, 0, 1] = True
    assert residual().any()
    active[1, 0] = torch.tensor([True, False])
    for _ in range(33):
        moved = residual()
    torch.testing.assert_close(moved[0, 0], torch.tensor([0., -.024, 0.]), atol=1e-6, rtol=0.)
    active[1, 0] = torch.tensor([False, True])
    action.reset(torch.tensor([0]))
    torch.testing.assert_close(residual()[0, 0], torch.tensor([-.0024, 0., 0.]))
    action.reset(torch.tensor([0]))
    raw[0, 0] = -1.0
    torch.testing.assert_close(residual()[0, 0], torch.tensor([-.04, 0., 0.]))


@pytest.mark.parametrize("widths", [(("right", 3), ("left", 5)), (("left", 3),)],
                         ids=["right-left", "left-only"])
def test_native_wrist_control_observation_retains_named_carry_and_ignores_obsolete_reset(widths):
    from mjlab.managers import ObservationManager
    from omegaconf import OmegaConf

    from envs import build_observations

    command, env, action = _wrist_control_case(widths)
    recipe = OmegaConf.create(dict(_target_="envs.observations.native.NativeMotionTrackingObs",
        obs_dict=dict(actor_obs=["native_wrist_control_state"],
                      critic_obs=["native_wrist_control_state"])))
    manager = ObservationManager(build_observations(recipe), env)
    raw = torch.linspace(-.7, .7, action.action_dim)[None]
    action.process_actions(raw)
    state = action.wrist_control_state
    roots = len(widths)
    expected = torch.cat((raw[:, :3 * roots].reshape(1, roots, 3),
                          raw[:, 3 * roots:6 * roots].reshape(1, roots, 3)), -1)
    assert state["hand_sides"] == tuple(side for side, _ in widths)
    torch.testing.assert_close(state["filtered_coordinates"], expected, rtol=0, atol=0)
    observed = manager.compute(update_history=True)
    assert observed["actor"].shape == (1, 6 * roots + 1)
    torch.testing.assert_close(observed["actor"], observed["critic"], rtol=0, atol=0)
    torch.testing.assert_close(observed["actor"][:, :-1], expected.flatten(1), rtol=0, atol=0)
    assert observed["actor"][0, -1] == 0 and not observed["actor"].requires_grad
    for side, _ in widths:
        command.motion_lib.native_root_pose_support[side][:] = False
    torch.testing.assert_close(manager.compute(update_history=True)["actor"], observed["actor"], rtol=0, atol=0)
    old_filter = action.capture_control_context().filter_state
    action.reset(torch.tensor([0]))
    reset = manager.compute(update_history=True)
    torch.testing.assert_close(reset["actor"][:, :-1], torch.zeros_like(observed["actor"][:, :-1]), rtol=0, atol=0)
    assert reset["actor"][0, -1] == 1
    torch.testing.assert_close(action.capture_control_context().filter_state, old_filter, rtol=0, atol=0)
    action.process_actions(-raw)
    torch.testing.assert_close(action.wrist_control_state["filtered_coordinates"], -expected, rtol=0, atol=0)
    assert not action.wrist_control_state["filter_pending"].any()
    default = OmegaConf.load(Path(__file__).resolve().parents[1] / "config/envs/native.yaml")
    assert all("native_wrist_control_state" not in values for values in default.obs.obs_dict.values())


@pytest.mark.parametrize("widths", [(("right", 3), ("left", 5)), (("left", 3),)],
                         ids=["bilateral-3-5", "left-only-3"])
def test_native_source_tip_decision_observation_has_named_independent_support(widths):
    command, env, installed, _ = _source_geometry_command(widths)
    side = command._side_list[0]
    bad = command.motion_lib.tip_ids[side][0]
    command.motion_lib.native_source_point_position_support[side][1, bad] = False
    command.motion_lib.mano_joint_pos[side][1, bad] = float("nan")
    # Registered target-root support is not source-point authority.
    for name in command._side_list:
        command.motion_lib.native_root_pose_support[name][1] = False
    installed["reference_motion_steps"][:] = 0
    current = command.source_point_reference(command._motion_flat_ids)
    assert current["source_point_names"][side][0] == command.motion_lib.mano_joint_names[side][bad]
    assert current["target_site_names"][side] == tuple(command.robot.site_names[i] for i in command._tip_site_ids[side])
    assert current["reference_time_ns"].dtype == torch.int64
    assert int(current["reference_time_ns"][0]) == 2**60 + 1_020_000_000
    manager = _source_geometry_observations(env)
    observed = manager.compute()
    total = sum(width for _, width in widths)
    assert observed["actor"].shape == (1, total * 4)
    torch.testing.assert_close(observed["actor"], observed["critic"])
    error, masks = observed["actor"][:, :3 * total].reshape(1, total, 3), observed["actor"][:, 3 * total:]
    torch.testing.assert_close(error[0, 0], torch.zeros(3))
    assert masks[0, 0] == 0 and int(masks.sum()) == total - 1
    # Distinct per-side WORLD errors, expressed in a +90deg actual wrist frame.
    shift = .025 if side == "right" else .06
    torch.testing.assert_close(error[0, 1], torch.tensor([-.01, shift, 0.]), atol=1e-6, rtol=1e-6)
    assert bool(torch.isfinite(observed["actor"]).all())
    command.motion_lib.native_source_point_position_support = None
    assert command.source_tip_geometry_facts() is None
    with pytest.raises(ValueError, match="source-point position support"):
        NativeMotionTrackingObs.native_source_tip_reference_error(env, "motion")
    for method, params in ((NativeMotionTrackingRewards.native_source_tip_position, {"scale": 20.}),
                           (NativeMotionTrackingRewards.native_source_tip_object_bound, {"sigma_m": .01})):
        with pytest.raises(ValueError, match="source-point position support"):
            method(env, "motion", **params)


def test_native_source_geometry_uses_consumed_packet_at_real_step_and_reset():
    from mjlab.managers import RewardManager
    from omegaconf import OmegaConf

    from envs import build_rewards
    from simulator.env import ManagerBasedRlEnv

    command, protocol, installed, _ = _source_geometry_command()
    # Normal scene binding and assist initialization, bypassed by this fixture.
    command._xfrc_curr_ctrl = None
    env = object.__new__(ManagerBasedRlEnv)
    env.scene, env._device = protocol.scene, "cpu"
    env.scene.num_envs = 1
    env.cfg = SimpleNamespace(decimation=2, physics_dt=.01, auto_reset=False)
    env._sim_step_counter, env.common_step_counter = 0, 0
    env._manual_reset_pending = torch.zeros(1, dtype=torch.bool)
    env.episode_length_buf, env.extras = torch.zeros(1, dtype=torch.long), {}
    command._env = env
    command._eval_mode = True
    command.cfg.resampling_time_range = (100., 100.)
    for points in command.motion_lib.mano_joint_pos.values():
        points.requires_grad_(True)
    command.time_left, command.command_counter = torch.full((1,), 100.), torch.zeros(1, dtype=torch.long)
    env.command_manager = SimpleNamespace(get_term=lambda _name: command,
        compute=lambda dt: command.compute(dt), compute_substep=lambda: None)

    def install():
        installed["installed_valid"][:] = True
        installed["reference_motion_ids"][:] = command.motion_ids
        installed["reference_motion_steps"][:] = command.motion_steps

    env.action_manager = protocol.action_manager
    env.action_manager.process_action = lambda _action: None
    env.action_manager.apply_action = install
    env.sim = SimpleNamespace(begin_control_step=lambda: None, end_control_step=lambda: None,
        step=lambda: command.robot.data.site_pos_w.add_(torch.tensor([.001, 0., 0.])),
        forward=lambda: None, sense=lambda: None)
    env.scene.write_data_to_sim, env.scene.update = lambda: None, lambda dt: None
    env.termination_manager = SimpleNamespace(compute=lambda: torch.zeros(1, dtype=torch.bool),
        terminated=torch.zeros(1, dtype=torch.bool), time_outs=torch.zeros(1, dtype=torch.bool))
    env.event_manager = SimpleNamespace(available_modes=())
    recipe = OmegaConf.create(dict(_target_="envs.rewards.native.NativeMotionTrackingRewards",
        reward_weights=dict(native_source_tip_position=.3, native_source_tip_object_bound=.3),
        reward_scales=dict(native_source_tip_position=20.),
        reward_params=dict(native_source_tip_object_bound=dict(sigma_m=.01))))
    env.reward_manager = RewardManager(build_rewards(recipe), env)
    env.observation_manager = _source_geometry_observations(env)
    installed["installed_valid"][:] = False
    with patch.object(command, "sdf_query_parts", wraps=command.sdf_query_parts) as query:
        reset = command.source_tip_geometry_facts()
        assert reset["reference_motion_steps"].tolist() == [0] and not bool(reset["execution_valid"].any())
        assert query.call_count == 2
        assert NativeMotionTrackingRewards.native_source_tip_position(env, "motion", 20.).item() == 0
        observations, _, _, _, _ = env.step(torch.zeros(1, 1))
        endpoint = command.source_tip_geometry_facts()
        assert query.call_count == 4  # Two total queries shared by both rewards and recorder.
        assert endpoint is command.source_tip_geometry_facts()
        assert endpoint["reference_motion_steps"].tolist() == [1]
        assert command.reference_phase["motion_steps"].tolist() == [1]
        assert command.motion_steps.tolist() == [2] and env._sim_step_counter == 2
        assert int(endpoint["reference_time_ns"][0]) == 2**60 + 1_020_000_000
        assert bool(endpoint["execution_valid"].all())
        assert not any(value.requires_grad for value in endpoint.values() if isinstance(value, torch.Tensor))
        current = command.source_point_reference(command._motion_flat_ids)
        assert int(current["reference_time_ns"][0]) == 2**60 + 1_040_000_000
        torch.testing.assert_close(observations["actor"], NativeMotionTrackingObs.native_source_tip_reference_error(env, "motion"))
        old_position = endpoint["achieved_position_w"].clone()
        env.step(torch.zeros(1, 1))
        tail = command.source_tip_geometry_facts()
        assert query.call_count == 6 and tail["reference_motion_steps"].tolist() == [2]
        assert command.motion_steps.tolist() == [2]  # Real tail clamp, no invented reference row.
        torch.testing.assert_close(endpoint["achieved_position_w"], old_position)
        version = command._obj_state_version
        installed["installed_valid"][:] = False
        command.reset(torch.tensor([0]))
        command.compute(0.)
        assert command._obj_state_version == version + 1
        reset_again = command.source_tip_geometry_facts()
        assert query.call_count == 8
        assert reset_again["reference_motion_steps"].tolist() == [0]
        assert not bool(reset_again["execution_valid"].any())
        assert command.motion_steps.tolist() == [1]


def test_native_source_part_query_domain_preserves_actual_mesh_chart_and_old_api():
    from dataclasses import replace

    command, env, _, _ = _source_geometry_command()
    del env
    part = replace(command.native_layout.objects[0].parts[0],
        body_from_mesh_position_m=(.2, .1, 0.),
        body_from_mesh_quat_wxyz=(2**-.5, 0., 0., 2**-.5))
    command.native_layout = replace(command.native_layout, objects=(replace(
        command.native_layout.objects[0], parts=(part, command.native_layout.objects[0].parts[1])),))
    state = command.sim_object_part_state_w
    state[:, 0, :3] = torch.tensor([.3, -.2, 0.])
    state[:, 0, 3:7] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    state[:, 1, 0] = 5.
    command._obj_sdf_extent = .5
    axis = torch.linspace(-.5, .5, 2)
    grid = torch.zeros(4, 2, 2, 2)
    grid[0], grid[1] = axis[:, None, None], 1.
    command._part_sdf = [grid, grid.clone()]
    # Mesh origin(.2,0,0), WORLD rotation180deg: local x=.1,-.1,.6.
    points = torch.tensor([[[.1, 0., 0.], [.3, 0., 0.], [-.4, 0., 0.], [float("nan"), 0., 0.]]])
    distance, gradient, support = command.sdf_query_parts(points, part_state_w=state, return_support=True)
    torch.testing.assert_close(distance[0, :3, 0], torch.tensor([.1, .1, .5]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gradient[0, 0, 0], torch.tensor([-1., 0., 0.]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gradient[0, 1, 0], torch.tensor([1., 0., 0.]), atol=1e-6, rtol=1e-6)
    assert support[0, :, 0].tolist() == [True, True, False, False]
    assert not bool(support[..., 1].any())
    old_distance, old_gradient = command.sdf_query_parts(points[:, :3], part_state_w=state)
    torch.testing.assert_close(old_distance, distance[:, :3])
    torch.testing.assert_close(old_gradient, gradient[:, :3])


@pytest.mark.parametrize("widths", [(("right", 3), ("left", 5)), (("left", 3),)],
                         ids=["bilateral-3-5", "left-only-3"])
def test_native_source_geometry_reward_split_conserves_supported_mass(widths):
    from mjlab.managers import RewardManager
    from omegaconf import OmegaConf

    from envs import build_rewards

    command, env, _, _ = _source_geometry_command(widths)
    facts = command.source_tip_geometry_facts()
    n = len(facts["metadata"]["source_point_names"])
    # Unequal available masses; every side remains on the same global denominator.
    facts["source_position_support"][:] = False
    facts["source_position_support"][0, :2] = True
    if n > 3:
        facts["source_position_support"][0, 3:7] = True
    facts["desired_part_query_support"][:] = False
    facts["live_part_query_support"][:] = False
    # Disjoint named parts are unavailable, even if independent minima look perfect.
    facts["desired_part_query_support"][0, 0, 0] = True
    facts["live_part_query_support"][0, 0, 1] = True
    facts["desired_part_distance_m"][0, 0] = torch.tensor([.1, .9])
    facts["live_part_distance_m"][0, 0] = torch.tensor([.9, .1])
    # Only part1 is common. Minima over different subsets would use the wrong d_ref.
    facts["desired_part_query_support"][0, 1] = True
    facts["live_part_query_support"][0, 1, 1] = True
    facts["desired_part_distance_m"][0, 1] = torch.tensor([0., .2])
    facts["live_part_distance_m"][0, 1] = torch.tensor([.1, .5])
    sigma = .2
    expected_bound = torch.exp(torch.tensor(-.3 / sigma))
    torch.testing.assert_close(NativeMotionTrackingRewards.native_source_tip_object_bound(env, "motion", sigma), expected_bound[None])
    with patch.object(command, "source_tip_geometry_facts", return_value=facts):
        for name, params in (("native_source_tip_position", dict(scale=20.)),
                             ("native_source_tip_object_bound", dict(sigma_m=sigma))):
            recipe = dict(_target_="envs.rewards.native.NativeMotionTrackingRewards",
                reward_weights={name: .3}, reward_params={name: params})
            whole = RewardManager(build_rewards(OmegaConf.create(recipe)), env).compute(env.step_dt).clone()
            recipe["split_sides"] = True
            split = RewardManager(build_rewards(OmegaConf.create(recipe)), env).compute(env.step_dt).clone()
            torch.testing.assert_close(split, whole)
            facts["execution_valid"][:] = False
            assert getattr(NativeMotionTrackingRewards, name)(env, "motion", **params).item() == 0
            facts["execution_valid"][:] = True
        facts["source_position_support"][:] = False
        facts["desired_part_query_support"][:] = False
        facts["live_part_query_support"][:] = False
        assert NativeMotionTrackingRewards.native_source_tip_position(env, "motion", 20.).item() == 0
        assert NativeMotionTrackingRewards.native_source_tip_object_bound(env, "motion", sigma).item() == 0
    from dataclasses import replace

    parts = command.native_layout.objects[0].parts
    command.native_layout = replace(command.native_layout, objects=(replace(
        command.native_layout.objects[0], parts=(parts[0], replace(parts[1], distance_sign="winding"))),))
    with pytest.raises(ValueError, match="compatible part distance signs"):
        NativeMotionTrackingRewards.native_source_tip_object_bound(env, "motion", sigma)


def test_native_mixed_passive_reward_and_tracking_keep_dimensional_columns():
    from mjlab.managers import RewardManager
    from omegaconf import OmegaConf

    from envs import build_rewards

    command, env, _ = _command(passive_types=("slide", "hinge"),
                                hand_types=("slide", "hinge", "hinge"))
    command.motion_steps.fill_(1)
    reference = command.object_joint_reference()
    assert reference["joint_names"] == ("hinge_a", "hinge_b")
    assert reference["joint_types"] == ("slide", "hinge")
    assert reference["position_units"] == ("m", "rad")
    # The prepared columns are reversed relative to the physical entity.
    torch.testing.assert_close(reference["position"], torch.tensor([[.1, .2]]))
    next(iter(command.native_objects.values())).data.joint_pos += torch.tensor([[.02, .05]])
    command.motion_lib.robot_joint_pos[1] = torch.tensor([.03, .8, .4])
    command.robot.data.joint_pos[0] = torch.tensor([.05, -.2, .45])
    command_ref = command.command_reference()
    assert command_ref["joint_names"] == ("left_driver", "right_driver")
    assert command_ref["joint_types"] == ("slide", "hinge")
    assert command_ref["position_units"] == ("m", "rad")
    torch.testing.assert_close(command_ref["position"], torch.tensor([[.03, .4]]))
    recipe = OmegaConf.create(dict(
        _target_="envs.rewards.native.NativeMotionTrackingRewards", split_sides=True,
        shared_terms=["native_object_articulation"],
        reward_weights=dict(native_object_articulation=.3),
        reward_params=dict(native_object_articulation=dict(scale_rad=.1, scale_m=.04)),
    ))
    manager = RewardManager(build_rewards(recipe), env)
    error = reference["position"] - command.native_object_joint_pos
    expected = (torch.exp(-error[:, 0].square() / .04**2)
                + torch.exp(-error[:, 1].square() / .1**2)) / 2
    torch.testing.assert_close(manager.compute(dt=1.), expected * .3)
    with pytest.raises(ValueError, match="scale_m"):
        NativeMotionTrackingRewards.native_object_articulation(env, "motion", .1)
    for invalid in (float("nan"), float("inf"), 0.):
        with pytest.raises(ValueError, match="scale_m"):
            NativeMotionTrackingRewards.native_object_articulation(env, "motion", .1, invalid)

    thresholds = dict(root_position_error_m=.1, root_rotation_error_rad=.1,
        part_position_error_m=.1, part_rotation_error_rad=.1,
        joint_position_error_rad=.1, joint_position_error_m=.01,
        command_position_error_rad=.1, command_position_error_m=.03)
    command._update_metrics()
    torch.testing.assert_close(command.native_step_metrics["command_position_error_rad"], torch.tensor([[.05]]))
    torch.testing.assert_close(command.native_step_metrics["command_position_error_m"], torch.tensor([[.02]]))
    callback = NativeReferenceTrackingPerformance("motion", thresholds)
    callback.on_start(env)
    callback.on_step(env)
    raw = callback.current_policy_outcomes([callback.collect_state()])
    assert raw["metadata"]["feature_names"]["joint_position_error_rad"] == ("hinge_b",)
    assert raw["metadata"]["feature_names"]["joint_position_error_m"] == ("hinge_a",)
    assert raw["metadata"]["passive_joint_coordinates"]["velocity_units"] == ("m/s", "rad/s")
    assert raw["metadata"]["feature_names"]["command_position_error_rad"] == ("right_driver",)
    assert raw["metadata"]["feature_names"]["command_position_error_m"] == ("left_driver",)
    assert raw["metadata"]["command_joint_coordinates"]["velocity_units"] == ("m/s", "rad/s")
    assert raw["columns"]["masses"]["joint_position_error_rad"].tolist() == [[1]]
    assert raw["columns"]["masses"]["joint_position_error_m"].tolist() == [[1]]
    assert raw["columns"]["masses"]["command_position_error_rad"].tolist() == [[1]]
    assert raw["columns"]["masses"]["command_position_error_m"].tolist() == [[1]]
    assert bool(callback.per_env_fail_obj([callback.collect_state()]).all())  # Passive metre bound only.
    with pytest.raises(ValueError, match="joint_position_error_m"):
        NativeReferenceTrackingPerformance("motion", {**thresholds, "joint_position_error_m": None}).on_start(env)
    for invalid in (None, float("nan"), float("inf"), -1.):
        with pytest.raises(ValueError, match="command_position_error_m"):
            NativeReferenceTrackingPerformance("motion", {**thresholds, "command_position_error_m": invalid}).on_start(env)
    command_only = NativeReferenceTrackingPerformance("motion", {
        **thresholds, "joint_position_error_m": .03, "command_position_error_m": .01})
    command_only.on_start(env)
    command_only.on_step(env)
    assert bool(command_only.per_env_fail([command_only.collect_state()]).all())
    assert not bool(command_only.per_env_fail_obj([command_only.collect_state()]).any())

    # Missing source support never adds a metre observation or a passing attempt.
    column = command.motion_lib.native_object_joint_names.index("hinge_a")
    command.motion_lib.native_object_joint_position_support[1, column] = False
    command._update_metrics()
    callback.on_start(env)
    callback.on_step(env)
    raw = callback.current_policy_outcomes([callback.collect_state()])
    assert raw["columns"]["masses"]["joint_position_error_m"].tolist() == [[0]]
    assert not bool(callback.per_env_scored([callback.collect_state()]).any())
    torch.testing.assert_close(NativeMotionTrackingRewards.native_object_articulation(env, "motion", .1, .04),
                               torch.exp(-error[:, 1].square() / .1**2))
    command.motion_lib.native_object_joint_position_support[1, column] = True
    command.motion_lib.native_joint_position_support[1, 0] = False
    command._update_metrics()
    callback.on_start(env)
    callback.on_step(env)
    raw = callback.current_policy_outcomes([callback.collect_state()])
    assert raw["columns"]["masses"]["command_position_error_m"].tolist() == [[0]]
    assert raw["columns"]["masses"]["command_position_error_rad"].tolist() == [[1]]
    assert not bool(callback.per_env_scored([callback.collect_state()]).any())

    # The existing hinge-only formula and empty rigid reduction stay exact.
    hinge, hinge_env, _ = _command()
    next(iter(hinge.native_objects.values())).data.joint_pos += torch.tensor([[.031, .087]])
    ref = hinge.object_joint_reference()
    old = torch.exp(-(ref["position"] - hinge.native_object_joint_pos).square() / .1**2).mean(-1)
    assert torch.equal(NativeMotionTrackingRewards.native_object_articulation(hinge_env, "motion", .1), old)
    hinge._update_metrics()
    command_error = (hinge.command_reference()["position"]
                     - hinge.robot.data.joint_pos[:, hinge.native_command_joint_ids]).abs()
    assert torch.equal(hinge.native_step_metrics["command_position_error_rad"], command_error)
    assert hinge.native_step_metrics["command_position_error_m"].shape == (1, 0)
    rigid, rigid_env, _ = _command(rigid=True)
    assert NativeMotionTrackingRewards.native_object_articulation(rigid_env, "motion", .1).tolist() == [0.]
    rigid._update_metrics()
    callback.on_start(rigid_env)
    assert callback.groups["joint_position_error_rad"][0] == ()
    assert "joint_position_error_m" not in callback.groups
    assert "command_position_error_m" not in callback.groups


def test_native_scalar_velocity_limits_use_object_and_hand_coordinate_units():
    command, env, _ = _command(passive_types=("slide", "hinge"),
                                hand_types=("hinge", "slide", "hinge"))
    params = dict(linear_m_per_s=100., angular_rad_per_s=200., joint_rad_per_s=.05, joint_m_per_s=.5)
    check = NativeMotionTrackingTerminations.native_velocity_diverged
    next(iter(command.native_objects.values())).data.joint_vel[0, 0] = .4
    command.robot.data.joint_vel[0, 1] = .4
    assert not bool(check(env, "motion", **params).any())  # Metres are not checked as radians.
    for values, column, speed in ((next(iter(command.native_objects.values())).data.joint_vel, 1, .06),
                                  (next(iter(command.native_objects.values())).data.joint_vel, 0, .6),
                                  (command.robot.data.joint_vel, 0, .06),
                                  (command.robot.data.joint_vel, 1, .6)):
        before = values[0, column].clone()
        values[0, column] = speed
        assert bool(check(env, "motion", **params).all())
        values[0, column] = before
    with pytest.raises(ValueError, match="joint_m_per_s"):
        check(env, "motion", **{**params, "joint_m_per_s": None})
    for invalid in (float("nan"), float("inf"), -1.):
        with pytest.raises(ValueError, match="joint_m_per_s"):
            check(env, "motion", **{**params, "joint_m_per_s": invalid})
    _, hinge_env, _ = _command()
    assert not bool(check(hinge_env, "motion", **{**params, "joint_m_per_s": None}).any())


def _source_intent_reward_case(widths=(("right", 3), ("left", 5))):
    """Real command/fact owners with renamed synthetic parts and uneven sites."""
    from dataclasses import replace

    command, env, installed, _ = _source_geometry_command(widths)
    names = ("housing_panel", "rotating_leaf")
    original = command.native_layout.objects[0]
    command.native_layout = replace(command.native_layout, objects=(replace(
        original, entity_name="renamed_item", root_body_name=names[0], parts=tuple(
            replace(part, name=f"surface_{i}", body_name=name)
            for i, (part, name) in enumerate(zip(original.parts, names, strict=True)))),))
    env.scene["renamed_item"] = env.scene.pop("object")
    env.scene["renamed_item"].body_names = list(names)
    ml = command.motion_lib
    ml.native_object_root_body_names, ml.native_object_part_body_names = {"renamed_item": names[0]}, names
    command._bind_native_reference()
    ml.hand_sides = tuple(command._side_list)
    shape = (3, len(widths), len(names))
    active = torch.zeros(shape, dtype=torch.bool)
    for hand in range(len(widths)):
        active[0, hand, 1 - hand % 2] = True
        active[1, hand, hand % 2] = True
    ml.native_source_hand_part_intent = dict(active=active, known=torch.ones_like(active))
    for key in ("query_time_ns", "capture_time_ns", "available_time_ns"):
        ml.native_source_hand_part_intent[key] = ml.native_time_ns.clone()
    ml.native_source_hand_part_intent["valid_until_time_ns"] = ml.native_time_ns + 20_000_000
    for key in ("source_frame_index", "canonical_frame_index"):
        ml.native_source_hand_part_intent[key] = torch.arange(3, dtype=torch.int64)
    ml.native_source_hand_part_intent_provenance = [{"authority": "synthetic geometric intent, not source force"}]
    geometry = command.source_tip_geometry_facts()
    geometry["live_part_distance_m"][:] = 0.0  # Irrelevant candidate minima must be ignored.
    geometry["live_part_query_support"][:] = False
    for hand, side in enumerate(command._side_list):
        begin = geometry["metadata"]["hand_offsets"][hand]
        geometry["live_part_query_support"][0, begin, hand % 2] = True
        geometry["live_part_distance_m"][0, begin, hand % 2] = .01 * (hand + 1)
    addresses, pairs = {}, {}
    for hand, binding in enumerate(command.native_layout.hands):
        for part, name in enumerate(names):
            key = f"arbitrary-{part}-{hand}"
            addresses[key] = dict(hand_side=binding.side, hand_body_name=binding.root_body_name,
                part_entity_name="renamed_item", part_body_name=name)
            force = (.25, 10.) if hand == 0 else (1., 1.)
            distance = (.01, .01) if hand == 0 else (.01, .03)
            pairs[key] = dict(normal_force_n=torch.tensor([force]),
                force_bearing_contact_present=torch.ones(1, 2, dtype=torch.bool),
                force_bearing_distance_max_m=torch.tensor([distance]))
    addresses["unrelated"] = dict(hand_side=command._side_list[0], hand_body_name="other_root",
        part_entity_name="other_item", part_body_name=names[0])
    pairs["unrelated"] = dict(normal_force_n=torch.full((1, 2), 1e6),
        force_bearing_contact_present=torch.ones(1, 2, dtype=torch.bool),
        force_bearing_distance_max_m=torch.zeros(1, 2))
    interval = dict(addresses=addresses, pairs=dict(reversed(tuple(pairs.items()))), num_substeps=2,
        sample_recorded=torch.ones(1, 2, dtype=torch.bool),
        measurement_support=torch.ones(1, 2, dtype=torch.bool),
        clock_support=torch.ones(1, 2, dtype=torch.bool), interval_complete=torch.ones(1, dtype=torch.bool),
        sample_begin_time_s=torch.tensor([[0., .005]], dtype=torch.float64),
        sample_end_time_s=torch.tensor([[.005, .02]], dtype=torch.float64))
    env.scene.physics_time_s = lambda: torch.tensor([.02], dtype=torch.float64)
    env.scene.hand_part_contact_interval_facts = lambda: interval
    return command, env, installed, geometry, interval


@pytest.mark.parametrize("widths", [(("right", 3), ("left", 5)), (("right", 2),)],
                         ids=["unequal-bilateral", "right-only"])
def test_source_intent_rewards_use_named_candidates_and_preserve_split_mass(widths):
    from mjlab.managers import RewardManager
    from omegaconf import OmegaConf

    from envs import build_rewards

    defaults = Path(__file__).resolve().parents[1] / "config"
    native = OmegaConf.load(defaults / "envs/native.yaml")
    actions = OmegaConf.load(defaults / "envs/actions/native.yaml")["actions"]
    assert "native_source_hand_part_contact" not in native.rewards.reward_weights
    assert "native_source_hand_part_proximity" not in native.rewards.reward_weights
    assert actions.finger_residual_rad is None and actions.cutoff_hz.finger is None
    assert actions.translation_residual_m == .1 and actions.rotation_residual_rad == 1.
    command, env, _, geometry, interval = _source_intent_reward_case(widths)
    expected_proximity = torch.exp(-torch.arange(1, len(widths) + 1, dtype=torch.float32)).mean()[None]
    expected_contact = torch.tensor([.5625 if len(widths) == 2 else .875])
    for name, params, expected in (
        ("native_source_hand_part_proximity", dict(distance_scale_m=.01), expected_proximity),
        ("native_source_hand_part_contact", dict(force_scale_n=.5, separation_bound_m=.02), expected_contact),
    ):
        torch.testing.assert_close(getattr(NativeMotionTrackingRewards, name)(env, "motion", **params), expected)
        recipe = dict(_target_="envs.rewards.native.NativeMotionTrackingRewards",
            reward_weights={name: .3}, reward_params={name: params})
        whole = RewardManager(build_rewards(OmegaConf.create(recipe)), env).compute(env.step_dt).clone()
        recipe["split_sides"] = True
        split = RewardManager(build_rewards(OmegaConf.create(recipe)), env).compute(env.step_dt).clone()
        torch.testing.assert_close(split, whole)
    # Force above the physical scale cannot buy additional credit.
    key = next(key for key, address in interval["addresses"].items()
               if address["hand_side"] == command._side_list[0] and address["part_body_name"] == "housing_panel")
    interval["pairs"][key]["normal_force_n"][0, 1] = 1e6
    torch.testing.assert_close(NativeMotionTrackingRewards.native_source_hand_part_contact(env, "motion", .5, .02), expected_contact)
    # Proximity to a non-candidate part and unrelated named-pair force are not credit.
    assert geometry["live_part_distance_m"].min().item() == 0.
    assert interval["pairs"]["unrelated"]["normal_force_n"].max().item() == 1e6


def test_source_intent_reward_unavailable_stale_and_measured_absence_stay_distinct():
    command, env, installed, geometry, interval = _source_intent_reward_case()
    contact = NativeMotionTrackingRewards.native_source_hand_part_contact
    proximity = NativeMotionTrackingRewards.native_source_hand_part_proximity
    interval["measurement_support"][:] = False
    assert contact(env, "motion", .5, .02).item() == 0.
    assert not interval["measurement_support"].any()  # Reward did not fabricate a known zero sample.
    interval["measurement_support"][:] = True
    for pair in interval["pairs"].values():
        pair["normal_force_n"].zero_()
        pair["force_bearing_contact_present"].zero_()
        pair["force_bearing_distance_max_m"][:] = float("nan")
    assert contact(env, "motion", .5, .02).item() == 0.
    assert interval["measurement_support"].all()  # Same credit, different factual authority.
    for pair in interval["pairs"].values():
        pair["normal_force_n"].fill_(1.)
        pair["force_bearing_contact_present"].fill_(True)
        pair["force_bearing_distance_max_m"].fill_(.01)
    assert contact(env, "motion", .5, .02).item() == 1.
    interval["sample_end_time_s"][0, -1] = .04
    assert contact(env, "motion", .5, .02).item() == 0.
    interval["sample_end_time_s"][0, -1] = .02
    interval["interval_complete"][:] = False
    assert contact(env, "motion", .5, .02).item() == 0.
    interval["interval_complete"][:] = True
    assert proximity(env, "motion", .01).item() > 0.
    geometry["reference_motion_steps"] = geometry["reference_motion_steps"] + 1
    assert proximity(env, "motion", .01).item() == 0.
    geometry["reference_motion_steps"] -= 1
    installed["installed_valid"][:] = False
    assert proximity(env, "motion", .01).item() == 0.
    assert contact(env, "motion", .5, .02).item() == 0.
    installed["installed_valid"][:] = True
    command.motion_lib.native_source_hand_part_intent["known"][:] = False
    assert proximity(env, "motion", .01).item() == 0.
    assert contact(env, "motion", .5, .02).item() == 0.
    command.motion_lib.native_source_hand_part_intent = None
    assert proximity(env, "motion", .01).item() == 0.
    assert contact(env, "motion", .5, .02).item() == 0.
    command.motion_lib.native_source_point_position_support = None
    env.scene.hand_part_contact_interval_facts = lambda: None
    assert proximity(env, "motion", .01).item() == 0.
    assert contact(env, "motion", .5, .02).item() == 0.


def test_source_contact_credit_requires_the_same_hand_to_load_both_objects_together():
    from dataclasses import replace

    command, env, _, _, interval = _source_intent_reward_case((("right", 3),))
    original, = command.native_layout.objects
    first, second = original.parts
    command.native_layout = replace(command.native_layout, objects=(
        replace(original, parts=(first,)),
        replace(original, entity_name="second_object", root_body_name=second.body_name,
                root_joint_name="second_free", parts=(second,)),
    ))
    command.motion_lib.native_source_hand_part_intent["active"][1, 0] = True
    interval["addresses"]["arbitrary-1-0"]["part_entity_name"] = "second_object"
    interval["pairs"]["arbitrary-0-0"]["normal_force_n"][:] = torch.tensor([1., 0.])
    interval["pairs"]["arbitrary-1-0"]["normal_force_n"][:] = torch.tensor([0., 1.])
    contact = NativeMotionTrackingRewards.native_source_hand_part_contact
    assert contact(env, "motion", .5, .02).item() == 0.
    interval["pairs"]["arbitrary-0-0"]["normal_force_n"][0, 1] = .5
    assert contact(env, "motion", .5, .02).item() == pytest.approx(.75)


def test_source_lift_load_uses_signed_force_across_assigned_parts_at_the_installed_clock():
    command, env, _, _, interval = _source_intent_reward_case()
    env.scene.gravity_up_w = torch.tensor([0., 1., 0.])
    env.scene.native_object_weights_n = torch.tensor([2.])
    parts = command.motion_lib.native_object_part_state_w
    parts[..., :3].zero_()
    parts[2, 0, 1] = .01
    for pair in interval["pairs"].values():
        pair["force_on_second_w_n"] = torch.tensor([[[100., 2., 0.], [100., 2., 0.]]])
    interval["pairs"]["arbitrary-0-0"]["force_on_second_w_n"][0, 0, 1] = -3.
    interval["pairs"]["arbitrary-1-1"]["force_bearing_distance_max_m"].fill_(.01)
    value = NativeMotionTrackingRewards.native_source_lift_load
    params = dict(lookahead_s=.02, minimum_speed_m_s=.05, separation_bound_m=.02)
    assert value(env, "motion", **params).item() == pytest.approx(.625)
    root = env.scene["renamed_item"].data.body_link_quat_w[:, 0]
    original = root.clone()
    root[:] = torch.tensor([0., 1., 0., 0.])
    assert value(env, "motion", **params).item() < 0.
    root.copy_(original)
    interval["pairs"]["arbitrary-0-0"]["force_on_second_w_n"].fill_(float("nan"))
    assert value(env, "motion", **params).item() == 0.
    interval["pairs"]["arbitrary-0-0"]["force_on_second_w_n"].fill_(1.)
    interval["measurement_support"][0, -1] = False
    assert value(env, "motion", **params).item() == 0.
    interval["measurement_support"][0, -1] = True
    parts[2, 0, 1] = 0.
    parts[2, 1, 1] = .01
    assert value(env, "motion", **params).item() == 0.
    parts[2, 0, 1] = .01
    interval["sample_end_time_s"][0, -1] = .04
    assert value(env, "motion", **params).item() == 0.


def test_source_joint_drive_requires_signed_solved_effort_and_achieved_motion():
    command, env, _, _, interval = _source_intent_reward_case()
    obj = env.scene["renamed_item"]
    _declare_joint_coordinates(obj, ("hinge", "slide"))
    command._bind_native_reference()
    ml = command.motion_lib
    ml.native_object_joint_vel[1] = torch.tensor([
        -.04 if name == "hinge_b" else .2 for name in ml.native_object_joint_names])
    ml.native_object_joint_velocity_support[1] = True
    obj.data.joint_vel[:] = torch.tensor([.1, -.04])
    env.scene.native_object_weights_n = torch.tensor([2.])
    for pair in interval["pairs"].values():
        pair["joint_effort_on_second"] = torch.zeros(1, 2, 2)
    drive = interval["pairs"]["arbitrary-0-0"]["joint_effort_on_second"]
    drive[:] = torch.tensor([.05, -2.])
    reward = NativeMotionTrackingRewards.native_source_joint_drive
    assert reward(env, "motion").item() == pytest.approx(.5)

    drive[0, 1] *= -1.
    assert reward(env, "motion").item() == pytest.approx(.125)
    obj.data.joint_vel[:, 1] *= -1.
    assert reward(env, "motion").item() == 0.
    obj.data.joint_vel[:, 1] *= -1.
    drive[0, 1, 1] = torch.nan
    assert reward(env, "motion").item() == 0.
    drive[:] = torch.tensor([.05, -2.])

    # The named source scope may select closing even when a local rate is positive.
    ml.native_task_phases = dict(joint_declared=torch.ones(3, 2, dtype=torch.bool),
        joint_direction=torch.tensor([[0, 0], [-1, -1], [0, 0]], dtype=torch.int8))
    assert reward(env, "motion").item() == 0.
    obj.data.joint_vel[:, 0] *= -1.
    drive[..., 0] *= -1.
    assert reward(env, "motion").item() == pytest.approx(.5)


def test_solved_contact_observation_retains_support_and_wrist_force_direction():
    command, env, _, _, interval = _source_intent_reward_case()
    env.scene.gravity_up_w = torch.tensor([0., 0., 1.])
    env.scene.native_object_weights_n = torch.tensor([2.])
    del interval["addresses"]["unrelated"]
    for pair in interval["pairs"].values():
        pair["force_on_second_w_n"] = torch.tensor([[[2., 0., 1.], [2., 0., 1.]]])
        pair["force_bearing_tangent_speed_max_m_s"] = torch.full((1, 2), .1)
        pair["joint_effort_on_second"] = torch.zeros(1, 2, 2)
    key = next(iter(command.native_layout.support_pair_bindings))
    env.scene.support_contact_interval_pairs = lambda: {
        key: {"force_on_second_w_n": torch.tensor([[[0., 0., 2.], [0., 0., 2.]]])},
    }
    observe = NativeMotionTrackingObs.native_solved_contact_feedback
    observation = observe(env, "motion")
    assert observation.shape == (1, 44)
    torch.testing.assert_close(observation[0, :3], torch.tanh(torch.tensor([0., -1., .5])))
    torch.testing.assert_close(observation[0, -4:], torch.tensor([math.tanh(1.), 1., 0., 0.]))
    interval["measurement_support"][0, -1] = False
    assert not observe(env, "motion").any()


def test_source_approach_rewards_two_digit_groups_on_the_same_supported_part():
    command, env, _, geometry, _ = _source_intent_reward_case((("right", 3),))
    command.motion_lib.native_source_hand_part_intent["active"][1] = True
    names = geometry["metadata"]["source_point_names"]
    thumb, finger = names.index("thumb_tip"), names.index("index_tip")
    distances = geometry["live_part_distance_m"]
    distances.fill_(.09)
    distances[0, thumb, 0] = -.004  # Absolute signed distance measures approach from either side.
    distances[0, finger, 1] = .005
    support = geometry["live_part_query_support"]
    support.fill_(True)
    desired = geometry["desired_part_distance_m"]
    desired.fill_(.09)
    desired[0, [thumb, finger]] = .004
    geometry["desired_part_query_support"].fill_(True)

    proximity = NativeMotionTrackingRewards.native_source_hand_part_proximity
    contact = NativeMotionTrackingRewards.native_source_hand_part_contact
    approach = dict(distance_scale_m=.04, source_digit_distance_m=.042)
    solved = dict(force_scale_n=.5, separation_bound_m=.02, source_digit_distance_m=.042,
                  digit_contact_distance_m=.02, unmatched_digit_scale=.15)
    assert proximity(env, "motion", **approach).item() == pytest.approx(math.exp(-.09 / .04))
    assert contact(env, "motion", **solved).item() == pytest.approx(.875 * .15)

    distances[0, finger, 0] = .021
    assert proximity(env, "motion", **approach).item() == pytest.approx(math.exp(-.021 / .04))
    assert contact(env, "motion", **solved).item() == pytest.approx(.875 * .15)
    distances[0, finger, 0] = .015
    assert proximity(env, "motion", **approach).item() == pytest.approx(math.exp(-.015 / .04))
    assert contact(env, "motion", **solved).item() == pytest.approx(.875)

    support[0, finger, 0] = False
    assert proximity(env, "motion", **approach).item() == pytest.approx(math.exp(-.09 / .04))
    assert contact(env, "motion", **solved).item() == pytest.approx(.875 * .15)
    # A source push only asks for its represented digit group on that part.
    command.motion_lib.native_source_hand_part_intent["active"][1, 0, 1] = False
    support[0, finger, 0] = True
    desired[0, thumb, 0] = distances[0, thumb, 0] = .09
    assert proximity(env, "motion", **approach).item() == pytest.approx(math.exp(-.015 / .04))
    assert contact(env, "motion", **solved).item() == pytest.approx(.875)
    geometry["desired_part_query_support"][0, thumb, 0] = False
    assert proximity(env, "motion", **approach).item() == 0.0
    assert contact(env, "motion", **solved).item() == 0.0
    geometry["desired_part_query_support"][0, thumb, 0] = True
    support.zero_()
    assert proximity(env, "motion", **approach).item() == 0.0


def test_source_intent_rewards_follow_real_manager_consumed_packet_and_reset():
    from mjlab.managers import RewardManager
    from omegaconf import OmegaConf

    from envs import build_rewards
    from simulator.env import ManagerBasedRlEnv
    from simulator.native.contacts import ControlIntervalContacts

    command, protocol, installed, _, prior = _source_intent_reward_case()
    command._source_tip_geometry_cache = None
    command._xfrc_curr_ctrl = None
    command._eval_mode = True
    command.cfg.resampling_time_range = (100., 100.)
    command.time_left, command.command_counter = torch.full((1,), 100.), torch.zeros(1, dtype=torch.long)
    env = object.__new__(ManagerBasedRlEnv)
    env.scene, env._device = protocol.scene, "cpu"
    env.scene.num_envs = 1
    env.cfg = SimpleNamespace(decimation=2, physics_dt=.01, auto_reset=False)
    env._sim_step_counter, env.common_step_counter = 0, 0
    env._manual_reset_pending = torch.zeros(1, dtype=torch.bool)
    env.episode_length_buf, env.extras = torch.zeros(1, dtype=torch.long), {}
    command._env = env
    env.command_manager = SimpleNamespace(get_term=lambda _name: command,
        compute=lambda dt: command.compute(dt), compute_substep=lambda: None)
    plant = SimpleNamespace(num_envs=1, device="cpu", contact_facts=prior["pairs"],
        contact_support=torch.ones(1, dtype=torch.bool),
        contact_sample_time_s=torch.zeros(1, dtype=torch.float64),
        time=torch.zeros(1, dtype=torch.float64), execution_clock=lambda: None)
    # The tape stores one solved sample per advance, not the fixture's two-sample arrays.
    plant.contact_facts = {name: {key: value[:, 0].clone() for key, value in pair.items()}
                           for name, pair in prior["pairs"].items()}
    plant.physics_time = lambda: plant.time
    tape = ControlIntervalContacts(plant, 2)
    env.scene.hand_part_contact_interval_facts = lambda: {**tape.facts(), "addresses": prior["addresses"]}
    env.scene.physics_time_s = plant.physics_time

    def install():
        installed["installed_valid"][:] = True
        installed["reference_motion_ids"][:] = command.motion_ids
        installed["reference_motion_steps"][:] = command.motion_steps

    def advance():
        plant.contact_sample_time_s = plant.time.clone()
        plant.time += .01
        command.robot.data.site_pos_w.add_(torch.tensor([.001, 0., 0.]))
        tape.append(plant)

    env.action_manager = protocol.action_manager
    env.action_manager.process_action = lambda _action: None
    env.action_manager.apply_action = install
    env.sim = SimpleNamespace(begin_control_step=tape.begin, end_control_step=tape.finish,
        step=advance, forward=lambda: None, sense=lambda: None)
    env.scene.write_data_to_sim, env.scene.update = lambda: None, lambda dt: None
    env.termination_manager = SimpleNamespace(compute=lambda: torch.zeros(1, dtype=torch.bool),
        terminated=torch.zeros(1, dtype=torch.bool), time_outs=torch.zeros(1, dtype=torch.bool))
    env.event_manager = SimpleNamespace(available_modes=())
    recipe = OmegaConf.create(dict(_target_="envs.rewards.native.NativeMotionTrackingRewards",
        reward_weights=dict(native_source_hand_part_proximity=.3, native_source_hand_part_contact=.3),
        reward_params=dict(native_source_hand_part_proximity=dict(distance_scale_m=.01),
                           native_source_hand_part_contact=dict(force_scale_n=.5, separation_bound_m=.02))))
    env.reward_manager = RewardManager(build_rewards(recipe), env)
    env.observation_manager = SimpleNamespace(compute=lambda update_history: {})
    installed["installed_valid"][:] = False
    assert NativeMotionTrackingRewards.native_source_hand_part_contact(env, "motion", .5, .02).item() == 0.
    with patch.object(command, "sdf_query_parts", wraps=command.sdf_query_parts) as query:
        _, reward, _, _, _ = env.step(torch.zeros(1, 1))
        assert reward.item() > 0 and query.call_count == 2
        facts = command.source_hand_part_intent_facts()
        assert facts["reference_motion_steps"].tolist() == [1]
        assert command.motion_steps.tolist() == [2]
        assert not command.source_hand_part_intent(command._motion_flat_ids)["active"].any()
        assert NativeMotionTrackingRewards.native_source_hand_part_contact(env, "motion", .5, .02).item() > 0.
        assert query.call_count == 2
    installed["installed_valid"][:] = False
    tape.discard()
    command.reset(torch.tensor([0]))
    command.compute(0.)
    assert NativeMotionTrackingRewards.native_source_hand_part_proximity(env, "motion", .01).item() == 0.
    assert NativeMotionTrackingRewards.native_source_hand_part_contact(env, "motion", .5, .02).item() == 0.
