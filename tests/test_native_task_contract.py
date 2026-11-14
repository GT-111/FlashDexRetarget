"""Synthetic native task DTO/caller contracts; no learned model, FK, Kit or physics."""

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


def _command(*, rigid=False):
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
            object=dict(
                entity_name="object",
                root_body_name="object_root",
                root_joint_name="object_free",
                parts=parts,
            ),
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
        native_object_entity_name="object",
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
    command._init_metrics()
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    return command, env, calls


def test_native_profile_builds_real_box_and_rigid_managers(tmp_path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from envs import build_manager_cfgs
    from envs.scene.native import bind_native_scene_config

    environment = {
        "FDR_NATIVE_SCENE_XML": str(tmp_path / "synthetic.xml"),
        "FDR_NATIVE_USD_DIR": str(tmp_path),
        "FDR_NATIVE_BOX_BASE_MESH": "base.obj",
        "FDR_NATIVE_BOX_LID_MESH": "lid.obj",
        "FDR_NATIVE_OBJECT_ROOT_BODY": "rigid_root",
        "FDR_NATIVE_OBJECT_ROOT_JOINT": "rigid_free",
        "FDR_NATIVE_OBJECT_MESH": "rigid.obj",
    }
    with (
        patch.dict("os.environ", environment),
        initialize_config_dir(
            config_dir=str(Path(__file__).resolve().parents[1] / "config"),
            version_base=None,
        ),
    ):
        OmegaConf.register_new_resolver(
            "eval", lambda expression: eval(expression), replace=True
        )
        for family, count in (("box", 2), ("rigid", 1)):
            cfg = compose(
                config_name="native",
                overrides=[
                    f"envs/native_object@native_object={family}",
                    "commands.motion_file=synthetic.pt",
                ],
            )
            assert OmegaConf.is_missing(cfg, "native_recording_binding")
            cfg.native_recording_binding = "synthetic-recording-binding.json"
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
            assert len(layout.object.parts) == count
            managers = build_manager_cfgs(cfg)
            assert not managers["curriculum"]
            assert "native_command_bc" in managers["rewards"]
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
        ref_joint_pos_used=torch.tensor([[0.2, -100.0, 0.3]]),
        finger_command_limits=torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
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
    command.cfg.object.entity_names = {"left": "object", "right": "object"}
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
    command.native_object.data.body_link_pos_w[:, 1, 0] += 1
    CommandTerm.compute(command, dt=env.step_dt)
    callback.on_step(env)
    assert callback.observed_steps.item() == 2
    assert callback.per_env_fail([callback.collect_state()]).all()
    command.native_object.data.body_link_pos_w[:, 1, 0] -= 1
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
    command.native_object.data.body_link_pos_w[:, 1, 0] += 1
    assert NativeMotionTrackingTerminations.native_reference_diverged(
        env, "motion", 0.2, 1.0, 0.2, 1.0, 0
    ).item()
    command.native_object.data.body_link_pos_w[:, 0, 2] -= 0.2
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
    adapter._art = SimpleNamespace(
        num_instances=2,
        set_external_force_and_torque=lambda f, t, **kw: calls.append(
            (f.clone(), t.clone(), kw)
        ),
    )
    adapter.clear_external_wrenches(torch.tensor([1]))
    force, torque, kwargs = calls.pop()
    assert force.shape == (1, 2, 3) and not force.any() and not torque.any()
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
        test_native_profile_builds_real_box_and_rigid_managers,
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
    from omegaconf import OmegaConf, open_dict
    from test_native_reference import _pipeline_config, _prepared_native_dataset

    import runner.off_policy_runner as runner_module
    from envs import build_actions, build_commands, build_rewards
    from envs.scene.native import bind_native_scene_config
    from retarget.kinematic_retargeting import PreparedNativeInput

    dataset, clip = _prepared_native_dataset(tmp_path)
    PreparedNativeInput(dataset, _pipeline_config(dataset), 1, "native-reference.json").pack()
    native = clip["native_reference"]
    mesh = str(dataset / "objects/synthetic-mesh/visual.obj")
    environment = {"FDR_NATIVE_SCENE_XML": "synthetic-not-read.xml",
                   "FDR_NATIVE_USD_DIR": str(tmp_path),
                   "FDR_NATIVE_BOX_BASE_MESH": mesh, "FDR_NATIVE_BOX_LID_MESH": mesh}
    with patch.dict("os.environ", environment), initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None,
    ):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
        cfg = compose(config_name="native", overrides=[f"commands.name={command_name}",
            f"commands.motion_file={dataset / 'motion.pt'}", "device=cpu", "num_envs=1",
            f"run_dir={tmp_path / 'run'}"])
        cfg.native_recording_binding = "synthetic-not-read-binding.json"
        cfg.native_tracking_tolerances = {name: .1 for name in cfg.native_tracking_tolerances}
        with open_dict(cfg.robot):
            cfg.robot.native_scene = dict(
                scene_xml_path="synthetic-not-read.xml",
                hands=[dict(side=side, entity_name=side, root_body_name=body,
                    root_joint_name=side + "_free", command_actuator_names=["command_" + side],
                    root_control=dict(position_stiffness=0, position_damping=0,
                        rotation_stiffness=0, rotation_damping=0,
                        force_limit_n=None, torque_limit_nm=None))
                    for side, body in native["root_body_names"].items()],
                object=dict(entity_name=native["object_entity_name"], root_body_name="base",
                    root_joint_name="object_free", parts=[dict(name=body, body_name=body, mesh_path=mesh)
                        for body in native["object_part_body_names"]]))
            cfg.robot.site_names = {kind: {side: [side + "_tip"] for side in ("left", "right")}
                                    for kind in ("tip", "contact")}
            cfg.robot.body_mapping = dict(all=[], level1={}, level2={})
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
    obj = SimpleNamespace(joint_names=("hinge_a", "hinge_b"), body_names=("base", "lid"),
                         data=SimpleNamespace(joint_pos=torch.zeros(1, 2), joint_vel=torch.zeros(1, 2)))

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
    index = env.reward_manager.active_terms.index("native_root_pose")
    weight = env.reward_manager._term_cfgs[index].weight
    for rate, expected in ((.7, False), (.1, True)):
        env.reward_manager._step_reward[:, index] = rate * weight
        for step in range(1, 7):
            env.episode_length_buf[:] = step
            fired = term.func(env, **term.params)
            assert fired.tolist() == [expected and step == 6]
