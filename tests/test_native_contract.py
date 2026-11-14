"""Native topology/action contracts; no Isaac app, MjData, FK or simulation."""

import xml.etree.ElementTree as ET
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch

from envs.actions.native import NativeResidualAction, NativeResidualActionCfg
from envs.scene.native import NativeSceneCfg
from simulator.isaacsim.convert_assets import write_native_mjcf_assets
from simulator.isaacsim.entity import FreeArticulationAdapter, _FreshnessClock
from simulator.mjcf_meta import load_robot_mjcf_meta


def _scene(tmp_path):
    source = tmp_path / "source.xml"
    source.write_text("""<mujoco><compiler angle="radian"/>
    <worldbody>
      <geom name="support" type="box" size=".1 .1 .02"/>
      <body name="left_root"><freejoint name="left_free"/>
        <geom type="box" size=".02 .02 .02" mass=".2"/>
        <body name="left_driver"><joint name="left_joint" range="0 1" actuatorfrcrange="-.8 .8"/>
          <geom type="box" size=".01 .01 .01" mass=".05"/>
          <body name="left_follower"><joint name="left_mimic" range="0 .6"/>
            <geom type="box" size=".01 .01 .01" mass=".05"/>
            <site name="left_tip" pos="0 0 .02"/>
          </body>
        </body>
      </body>
      <body name="right_root"><freejoint name="right_free"/>
        <geom type="box" size=".02 .02 .02" mass=".2"/>
        <body name="right_driver"><joint name="right_joint" range="0 1" actuatorfrcrange="-1 1"/>
          <geom type="box" size=".01 .01 .01" mass=".1"/>
          <site name="right_tip"/>
        </body>
        <body name="right_second"><joint name="right_other" range="0 2" actuatorfrcrange="-1 1"/>
          <geom type="box" size=".01 .01 .01" mass=".1"/>
        </body>
      </body>
      <body name="base"><freejoint name="base_free"/>
        <geom type="box" size=".1 .1 .01" mass=".075"/>
        <body name="lid"><joint name="hinge" limited="false" damping=".01" armature=".001"/>
          <geom type="box" size=".1 .1 .01" mass=".025"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <position name="command_A" joint="left_joint" kp="20" ctrllimited="false"/>
      <position name="command_B" joint="right_joint" kp="20" ctrllimited="false"/>
      <position name="command_C" joint="right_other" kp="20" ctrllimited="false"/>
    </actuator>
    <equality><joint name="mimic" joint1="left_mimic" joint2="left_joint" polycoef="0 1.5 0 0 0"/></equality>
    </mujoco>""")
    control = dict(
        position_stiffness=400.0,
        position_damping=40.0,
        rotation_stiffness=400.0,
        rotation_damping=40.0,
        force_limit_n=10.0,
        torque_limit_nm=1.0,
        gravity_compensation=1.0,
    )
    mapping = dict(
        scene_xml_path=str(source),
        hands=[
            dict(
                side=side,
                entity_name=f"hand_{side}",
                root_body_name=f"{side}_root",
                root_joint_name=f"{side}_free",
                command_actuator_names=commands,
                root_control=control,
            )
            for side, commands in (
                ("left", ["command_A"]),
                ("right", ["command_B", "command_C"]),
            )
        ],
        object=dict(
            entity_name="object",
            root_body_name="base",
            root_joint_name="base_free",
            parts=[
                dict(name=n, body_name=n, mesh_path=f"{n}.obj") for n in ("base", "lid")
            ],
        ),
    )
    return NativeSceneCfg.from_mapping(mapping), mapping


def test_native_metadata_keeps_free_roots_passive_mimics_and_real_effort_bounds(
    tmp_path,
):
    layout, _ = _scene(tmp_path)
    meta = load_robot_mjcf_meta(layout.scene_xml_path)
    assert [r.name for r in meta.free_roots] == ["left_free", "right_free", "base_free"]
    assert meta.joint_names == [
        "left_joint",
        "left_mimic",
        "right_joint",
        "right_other",
        "hinge",
    ]
    left = meta.subtree("left_root")
    assert left.driven_joint_names == ["left_joint"]
    assert left.mimics[0].reference_joint_name == "left_joint"
    np.testing.assert_allclose(left.command_limits(("command_A",)), [[0.0, 0.4]])
    # The source bounds the joint's actuator force, not actuator_forcerange.
    np.testing.assert_allclose(left.actuator_effort_limit, [0.8])
    obj = meta.subtree("base")
    assert obj.actuator_names == []
    assert obj.body_names == ["base", "lid"]
    assert obj.joint_names == ["hinge"] and not obj.joint_limited[0]
    np.testing.assert_allclose(obj.body_mass, [0.075, 0.025])


def test_native_conversion_inputs_preserve_subtree_inertials_and_one_articulated_object(
    tmp_path,
):
    layout, _ = _scene(tmp_path)
    original = load_robot_mjcf_meta(layout.scene_xml_path)
    paths = write_native_mjcf_assets(layout, {
        "static": str(tmp_path / "static.usd"),
        "articulations": {b.entity_name: str(tmp_path / (b.entity_name + ".usd"))
                          for b in layout.articulations},
    })
    for binding in layout.articulations:
        saved = load_robot_mjcf_meta(paths[binding.entity_name])
        expected = original.subtree(binding.root_body_name)
        assert saved.body_names == expected.body_names
        assert saved.joint_names == expected.joint_names
        assert len(saved.free_roots) == 1
        np.testing.assert_array_equal(saved.body_mass, expected.body_mass)
        np.testing.assert_array_equal(saved.body_ipos, expected.body_ipos)
        np.testing.assert_allclose(
            saved.body_inertia, expected.body_inertia, rtol=1e-15
        )
    left = load_robot_mjcf_meta(paths["hand_left"])
    assert len(left.mimics) == 1
    obj = ET.parse(paths["object"]).getroot()
    assert len(obj.findall(".//freejoint") + obj.findall(".//joint[@type='free']")) == 1
    assert obj.find(".//joint[@name='hinge']") is not None
    static = ET.parse(paths["static"]).getroot()
    assert static.find(".//geom[@name='support']") is not None
    assert not static.findall(".//freejoint") + static.findall(".//joint[@type='free']")


def test_actual_native_action_derives_dimensions_and_records_accepted_supported_targets(
    tmp_path,
):
    layout, mapping = _scene(tmp_path)
    meta = load_robot_mjcf_meta(layout.scene_xml_path)
    names = ["left_joint", "left_mimic", "right_joint", "right_other"]
    accepted = []

    def install(target, support, control):
        accepted.append((target.clone(), support.clone(), asdict(control)))
        return torch.zeros(2, 3), torch.zeros(2, 3)

    class Scene(dict):
        env_origins = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    target = torch.tensor([[0.2, 0.3, 0.4, 0.5]]).repeat(2, 1)
    entity = SimpleNamespace(
        data=SimpleNamespace(joint_pos_target=target),
        hands={
            s: SimpleNamespace(apply_root_pose_target=install)
            for s in ("left", "right")
        },
        native_command_limits=meta.command_limits,
        find_joints_by_actuator_names=lambda aa: (
            [names.index(meta.actuator_joint[a]) for a in aa],
            [meta.actuator_joint[a] for a in aa],
        ),
        set_joint_position_target=lambda value, ids: target.__setitem__(
            (slice(None), ids), value
        ),
    )
    roots = {
        s: torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 1.0, 0.0, 0.0]]
        ).repeat(2, 1)
        for s in ("left", "right")
    }
    ml = SimpleNamespace(
        native_joint_position_support=torch.tensor(
            [[True, True, False, True], [True, True, False, True]]
        ),
        native_root_body_names={s: f"{s}_root" for s in roots},
        native_root_state_w=roots,
        native_root_pose_support={s: torch.ones(2, dtype=torch.bool) for s in roots},
        native_root_linear_velocity_support={
            s: torch.zeros(2, dtype=torch.bool) for s in roots
        },
        native_root_angular_velocity_support={
            s: torch.ones(2, dtype=torch.bool) for s in roots
        },
    )
    command = SimpleNamespace(
        motion_lib=ml,
        _motion_flat_ids=torch.tensor([0, 1]),
        motion_ids=torch.tensor([2, 3]),
        motion_steps=torch.tensor([4, 5]),
        ref_joint_pos=torch.zeros(2, 4),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        scene=Scene(robot=entity),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        close=lambda: None,
    )
    action = NativeResidualAction(
        NativeResidualActionCfg(
            entity_name="robot",
            command_name="motion",
            native_scene=mapping,
            translation_residual_m=0.1,
            rotation_residual_rad=np.pi / 2,
            clip={".*": (-1.0, 1.0)},
        ),
        env,
    )
    assert action.action_dim == 2 * 6 + 3
    from envs.vec_env import FlatObsVecEnv

    env.action_manager = SimpleNamespace(_terms={"native": action})
    wrapper = object.__new__(FlatObsVecEnv)
    wrapper._env = env
    side_ids = wrapper._action_side_indices(["left", "right"])
    assert side_ids == {
        "left": [0, 1, 2, 6, 7, 8, 12],
        "right": [3, 4, 5, 9, 10, 11, 13, 14],
    }
    from envs.actions.residual import ResidualAction
    from envs.rewards.base import BaseRewards

    env.action_manager.action = torch.arange(action.action_dim).float().repeat(2, 1)
    for side, indices in side_ids.items():
        mask = BaseRewards._side_action_mask(env, side)
        assert torch.nonzero(mask).flatten().tolist() == indices
        torch.testing.assert_close(
            BaseRewards.action_penalty(env, side),
            env.action_manager.action[:, indices].square().mean(-1),
        )
    stock_action = object.__new__(ResidualAction)
    stock_action._entity = SimpleNamespace(joint_names=["R_finger", "L_wrist"])
    stock_action._all_joint_ids = torch.tensor([1, 0])
    assert stock_action.action_names == ("L_wrist", "R_finger")
    assert stock_action.action_sides == ("left", "right")
    raw = torch.ones(2, action.action_dim) * 2
    # Rotate only about WORLD z; the supported source omega must rotate too.
    raw[:, 6:12] = 0.0
    raw[:, 8] = raw[:, 11] = 1.0
    action.process_actions(raw)
    assert not action.installed_control["installed_valid"].any()
    action.apply_actions()
    record = action.installed_control
    np.testing.assert_allclose(
        record["finger_target_rad"], [[0.4, 0.4, 2.0], [0.4, 0.4, 2.0]]
    )
    assert record["finger_joint_names"] == ("left_joint", "right_joint", "right_other")
    np.testing.assert_allclose(record["root_state_w"][..., 7:10], 0.0)
    np.testing.assert_allclose(
        record["root_state_w"][..., 10:13], [[[0.0, 1.0, 0.0]] * 2] * 2, atol=1e-6
    )
    np.testing.assert_allclose(
        record["root_state_w"][:, 0, :3], env.scene.env_origins + 0.1
    )
    assert not record["root_linear_velocity_support"].any()
    np.testing.assert_array_equal(record["reference_motion_steps"], [4, 5])
    assert len(accepted) == 2 and record["installed_valid"].all()
    assert record["root_control"]["main_tracker_numeric_parity"] is False
    recipe = record["action_recipe"]
    assert recipe["translation_residual_per_axis_m"] == 0.1
    assert recipe["rotation_residual_per_axis_rad"] == np.pi / 2
    assert recipe["action_names"] == action.action_names
    assert recipe["action_sides"] == action.action_sides
    import json

    json.dumps(recipe)  # A recorder can retain the actual accepted action recipe.
    action.reset(torch.tensor([0]))
    np.testing.assert_array_equal(record["installed_valid"], [False, True])


def test_native_free_root_wrench_is_root_only_and_recomputed_from_current_body_state(
    tmp_path,
):
    layout, _ = _scene(tmp_path)
    meta = load_robot_mjcf_meta(layout.scene_xml_path).subtree("left_root")
    transforms = torch.zeros(1, len(meta.body_names), 7)
    transforms[..., 6] = 1.0  # PhysX XYZW
    transforms[:, 1:, 1] = 0.02
    calls = []
    view = SimpleNamespace(
        get_coms=lambda: torch.zeros_like(transforms),
        get_link_transforms=lambda: transforms,
        get_link_velocities=lambda: torch.zeros(1, len(meta.body_names), 6),
    )
    art = SimpleNamespace(
        is_fixed_base=False,
        num_instances=1,
        body_names=meta.body_names,
        find_bodies=lambda names, preserve_order: (list(range(len(names))), names),
        find_joints=lambda names, preserve_order: (list(range(len(names))), names),
        root_physx_view=view,
        set_external_force_and_torque=lambda f, t, **kw: calls.append(
            (f.clone(), t.clone(), kw)
        ),
    )
    clock = _FreshnessClock()
    hand = FreeArticulationAdapter(art, meta, clock, "cpu", root_body_name="left_root")
    state = torch.tensor(
        [[0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    force1, _ = hand.apply_root_pose_target(
        state, torch.tensor([True]), layout.hands[0].root_control
    )
    transforms[..., 0] = 0.04
    clock.bump()
    force2, _ = hand.apply_root_pose_target(
        state, torch.tensor([True]), layout.hands[0].root_control
    )
    assert force1[0, 0] > 0 and force2[0, 0] < 0
    for forces, torques, kwargs in calls:
        assert torch.count_nonzero(forces[:, 1:]) == 0
        assert torch.count_nonzero(torques[:, 1:]) == 0
        assert kwargs["is_global"] is True
        assert float(forces[:, 0].norm()) <= 10.0
        assert float(torques[:, 0].norm()) <= 1.0


if __name__ == "__main__":
    # The pinned author environment need not install a separate test runner.
    from pathlib import Path
    from tempfile import TemporaryDirectory

    torch.set_num_threads(1)
    cases = (
        test_native_metadata_keeps_free_roots_passive_mimics_and_real_effort_bounds,
        test_native_conversion_inputs_preserve_subtree_inertials_and_one_articulated_object,
        test_actual_native_action_derives_dimensions_and_records_accepted_supported_targets,
        test_native_free_root_wrench_is_root_only_and_recomputed_from_current_body_state,
    )
    for case in cases:
        with TemporaryDirectory() as directory:
            case(Path(directory))
        print(f"PASS {case.__name__}", flush=True)
    print(
        f"{len(cases)} passed; no FK, dynamics steps or Isaac application", flush=True
    )
