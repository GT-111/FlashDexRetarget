"""Native replay/clock software contracts; no scene construction or physics."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_native_task_contract import _command

from agents.storage.obs_compact import ObsCompactor, flat_obs_layout
from envs.observations.native import NativeMotionTrackingObs


def _replay_case(*, rigid, joint_order=(2, 0, 1), source_entity="robot"):
    command, env, _ = _command(rigid=rigid)
    ml = command.motion_lib
    # Two genuine reference address ranges with large, independent int64 origins.
    for name, value in vars(ml).copy().items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 3:
            setattr(ml, name, torch.cat((value, value.clone())))
        elif isinstance(value, dict) and all(
            isinstance(rows, torch.Tensor) for rows in value.values()
        ):
            setattr(
                ml,
                name,
                {side: torch.cat((rows, rows.clone())) for side, rows in value.items()},
            )
    ml.length_starts = torch.tensor([0, 3])
    ml._motion_num_frames = torch.tensor([3, 3])
    ml.num_trajectories = 2
    ml.native_time_ns = torch.tensor(
        [4_000_000_000_000_000_000 + k * 20_000_000 for k in range(3)]
        + [5_000_000_000_000_000_000 + k * 20_000_000 for k in range(3)]
    )
    ml.obj_trans = {}
    ml.robot_joint_pos[:] = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.3, 0.8, 0.4],
            [0.5, 0.7, 0.6],
            [1.1, 1.9, 1.2],
            [1.3, 1.8, 1.4],
            [1.5, 1.7, 1.6],
        ]
    )
    ml.robot_joint_vel[:] = torch.tensor([0.5, 0.0, 0.0])
    ml.native_joint_velocity_support[:, 0] = True
    ml.native_joint_position_support[2, 2] = False
    for side in command._side_list:
        roots = ml.native_root_state_w[side]
        roots[:, 0] = torch.arange(6) * 0.125
        roots[:, 7:10] = torch.tensor([0.25, -0.5, 0.125])
        ml.native_root_linear_velocity_support[side][:] = True
    ml.native_root_pose_support["right"][2] = False
    ml.native_object_part_state_w[..., :3] = torch.tensor([1.0, 0.2, 0.3])
    ml.native_object_part_linear_velocity_support[:] = True
    if not rigid:
        ml.native_object_joint_pos[:] = torch.tensor([0.75, -0.25])
        ml.native_object_joint_vel[:] = torch.tensor([0.0, 0.25])
        ml.native_object_joint_velocity_support[:, 1] = True
        ml.native_object_part_pose_support[2, 1] = False

    env.num_envs = 2
    env.scene.env_origins = torch.tensor([[10.0, -20.0, 0.75], [-5.0, 8.0, 1.0]])
    local_pos = torch.tensor(
        [
            [[0.25, -0.1, 0.1], [-0.25, 0.5, -0.125]],
            [[0.125, 0.25, 0.5], [0.25, -0.25, 0.125]],
        ]
    )
    robot = command.robot.data
    robot.body_link_pos_w = local_pos + env.scene.env_origins[:, None]
    robot.body_link_quat_w = torch.zeros(2, 2, 4)
    robot.body_link_quat_w[..., 0] = 1
    robot.body_link_quat_w[0, 0] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    robot.body_link_lin_vel_w = torch.tensor(
        [
            [[0.125, -0.25, 0.5], [0.25, 0.125, -0.5]],
            [[-0.125, 0.25, 0.125], [0.5, -0.125, 0.25]],
        ]
    )
    robot.body_link_ang_vel_w = torch.zeros(2, 2, 3)
    robot.body_link_ang_vel_w[:, 0, 2] = 0.5
    robot.joint_pos = torch.tensor([[0.25, 0.91, -0.5], [0.125, 0.71, 0.25]])
    robot.joint_vel = torch.tensor([[0.5, 0.9, -0.25], [0.25, 0.7, 0.125]])
    if not rigid:
        obj = command.native_object.data
        obj.joint_pos = torch.tensor([[0.5, -0.125], [-0.25, 0.25]])
        obj.joint_vel = torch.tensor([[0.125, -0.5], [0.25, 0.5]])
    # Both native independent-command order and object-reference order differ
    # from the full physical scalar order; the passive hand column stays factual.
    command._finger_joint_ids = torch.tensor([2, 0])
    command.motion_ids = torch.tensor([1, 0])
    command.motion_steps = torch.tensor([0, 1])
    if source_entity != "robot":
        env.scene[source_entity] = SimpleNamespace(
            joint_names=command.robot.joint_names,
            data=SimpleNamespace(
                joint_pos=robot.joint_pos + 0.25,
                joint_vel=robot.joint_vel - 0.125,
            ),
        )
    source = env.scene[source_entity].data
    fields = [
        (
            "robot_wrist_pose_w",
            torch.cat(
                (
                    robot.body_link_pos_w - env.scene.env_origins[:, None],
                    robot.body_link_quat_w,
                ),
                -1,
            ).flatten(1),
        ),
        ("robot_wrist_lin_vel_w", robot.body_link_lin_vel_w.flatten(1)),
        ("robot_wrist_ang_vel_w", robot.body_link_ang_vel_w.flatten(1)),
        ("robot_joint_pos", source.joint_pos[:, joint_order]),
        ("robot_joint_vel", source.joint_vel[:, joint_order]),
        (
            "native_object_joint_state",
            NativeMotionTrackingObs.native_object_joint_state(env, "motion"),
        ),
        (
            "native_future_reference",
            NativeMotionTrackingObs.native_future_reference(env, "motion", 2, 1),
        ),
    ]
    groups = {"actor": fields[:-1], "future_traj": fields[-1:]}
    terms = {}
    for group, values in groups.items():
        for name, _value in values:
            params = {"n_future": 2, "stride": 1} if group == "future_traj" else {}
            if name in ("robot_joint_pos", "robot_joint_vel"):
                params["asset_cfg"] = SimpleNamespace(
                    name=source_entity, joint_ids=joint_order
                )
            terms[group, name] = SimpleNamespace(
                params=params, scale=None, clip=None, noise=None
            )
    env.observation_manager = SimpleNamespace(
        active_terms={
            group: [name for name, _ in values] for group, values in groups.items()
        },
        group_obs_term_dim={
            group: [(value.shape[-1],) for _, value in values]
            for group, values in groups.items()
        },
        get_term_cfg=lambda group, name: terms[group, name],
    )
    spans = flat_obs_layout(env, ["actor", "future_traj"])
    full = torch.cat([value for _, value in fields], -1)
    compactor = ObsCompactor(command, spans, full.shape[-1], "cpu")
    return command, env, full, compactor, spans[-1]


@pytest.mark.parametrize("rigid", [False, True], ids=["passive-articulation", "rigid"])
def test_native_replay_reconstructs_named_future_without_live_world_reads(rigid):
    command, _env, full, compactor, future = _replay_case(rigid=rigid)
    compact, aux = compactor.compact(full), compactor.aux_from_command().clone()
    assert compactor.full_dim - compactor.compact_dim == future.end - future.start
    # Sampled replay has another row count/order and retains each row's captured
    # phase, even after live worlds reset or move to a different command.
    select = torch.tensor([1, 0, 1, 1, 0])
    command.motion_ids.zero_()
    command.motion_steps.fill_(2)
    command.robot.data.joint_pos.fill_(0.75)
    command.robot.data.joint_vel.fill_(-0.5)
    command.robot.data.body_link_pos_w.fill_(1.0)
    if not rigid:
        command.native_object.data.joint_pos.fill_(0.25)
    actual = compactor.expand(compact[select], aux[select])
    torch.testing.assert_close(actual, full[select], rtol=0, atol=0)


def test_native_future_retains_independent_support_clocks_and_physical_columns():
    command, _env, full, _compactor, span = _replay_case(rigid=False)
    rows = full[:, span.start : span.end].reshape(2, 2, -1)
    torch.testing.assert_close(
        rows[:, :, -1], torch.tensor([[0.02, 0.04], [0.02, 0.0]]), rtol=0, atol=0
    )
    # The last lookahead of the second sample lies beyond its own capture.
    assert not bool(rows[1, 1].any())
    # Right root pose is unknown at that sample's last supported endpoint;
    # known linear-rate flags do not fabricate a supported pose or velocity.
    assert not bool(rows[1, 0, 15:30].any())
    q_start = 15 * len(command._side_list)
    torch.testing.assert_close(
        rows[0, 0, q_start : q_start + 2], torch.tensor([1.9, 1.05])
    )
    # Velocity for the first command (physical scalar column2) is unknown.
    torch.testing.assert_close(
        rows[0, 0, q_start + 2 : q_start + 4], torch.tensor([0.0, 0.0])
    )
    torch.testing.assert_close(
        rows[0, 0, q_start + 6 : q_start + 8], torch.tensor([0.0, 1.0])
    )
    object_start = q_start + 4 * command.native_command_joint_ids.numel()
    torch.testing.assert_close(
        rows[0, 0, object_start : object_start + 2], torch.tensor([-0.75, 0.875])
    )
    # 90-degree WORLD-z wrist rotation: (x,y,z) -> (y,-x,z).
    part_start = object_start + 4 * len(command.native_object.joint_names)
    wrist_position = full[0, :3]
    displacement = torch.tensor([1.0, 0.2, 0.3]) - wrist_position
    expected = torch.stack((displacement[1], -displacement[0], displacement[2]))
    torch.testing.assert_close(rows[0, 0, part_start : part_start + 3], expected)


def test_native_replay_fp16_keeps_exact_phase_and_support_authority():
    command, _env, full, compactor, future = _replay_case(rigid=False)
    compact = compactor.compact(full).to(torch.float16).to(torch.float32)
    aux = compactor.aux_from_command()
    actual = compactor.expand(compact, aux)
    before = full[:, future.start : future.end].reshape(2, 2, -1)
    after = actual[:, future.start : future.end].reshape(2, 2, -1)
    assert bool(torch.isfinite(actual).all())
    assert aux.dtype == torch.int32
    torch.testing.assert_close(after[..., -1], before[..., -1], rtol=0, atol=0)
    assert not bool(after[1, 1].any())
    assert not bool(after[1, 0, 15:30].any())
    command_width = command.native_command_joint_ids.numel()
    start = 15 * len(command._side_list) + 2 * command_width
    torch.testing.assert_close(
        after[..., start : start + 2 * command_width],
        before[..., start : start + 2 * command_width],
        rtol=0,
        atol=0,
    )


def test_native_replay_rejects_missing_named_command_columns():
    with pytest.raises(ValueError, match="missing required named command columns"):
        _replay_case(rigid=False, joint_order=(0, 1))


def test_native_replay_rejects_another_entity_with_the_same_scalar_names():
    with pytest.raises(ValueError, match="command's actual robot entity"):
        _replay_case(rigid=False, source_entity="other_robot")


def test_native_profiles_declare_50hz_without_changing_author_clock():
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        for backend in ("native_isaacsim", "mjwarp", "newton"):
            cfg = compose(
                config_name="native", overrides=[f"envs/simulator@simulator={backend}"]
            )
            assert float(cfg.sim.dt) == 0.005
            assert float(cfg.sim.dt) * int(cfg.decimation) == 0.02
        author = compose(config_name="flashdexretarget")
        assert float(author.sim.dt) == 0.0041666667
