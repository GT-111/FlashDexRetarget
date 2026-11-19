"""Factual export additions exercise the actual author recorder without a sim."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.callbacks.rollout_record import NativeRolloutRecord
from evaluation.callbacks.success_export import export_success


def record_reference_phase_fixture(tmp_path, binding, *, native, rigid=False, setup_scene=None):
    """Run actual command/action owners with explicit synthetic backend facts.

    The physical state is supplied by this protocol fixture; no dynamics model,
    FK, simulator application or policy is executed.
    """
    from simulator.isaacsim import bootstrap

    bootstrap()
    from mjlab.managers import CommandTerm
    from mjlab.managers.action_manager import ActionManager
    from test_native_task_contract import _command

    from envs.actions.native import NativeResidualActionCfg
    from envs.actions.residual import ResidualActionCfg
    from envs.commands.motion_tracking.monitor.metrics import MetricsMixin
    from envs.commands.motion_tracking.motion_tracking import MotionTrackingCommand

    native_command, env, _calls = _command(rigid=rigid)
    robot, library = native_command.robot, native_command.motion_lib
    library.hand_sides = tuple(native_command._side_list)
    robot.body_names = ["left_root", "right_root"]
    robot.data.root_link_pos_w = torch.full((1, 3), 999.)
    robot.data.root_link_quat_w = torch.tensor([[1., 0., 0., 0.]])
    robot.data.joint_pos_target = torch.zeros((1, 3))
    robot.data.soft_joint_pos_limits = torch.tensor([[[-1., 1.]] * 3])
    correspondence = {"command_left": "left_driver", "command_right": "right_driver"}
    robot.find_joints_by_actuator_names = lambda names: (
        [robot.joint_names.index(correspondence[name]) for name in names],
        [correspondence[name] for name in names])
    robot.native_command_limits = lambda names: np.tile((-1., 1.), (len(names), 1))
    robot.set_joint_position_target = lambda target, joint_ids: robot.data.joint_pos_target.__setitem__(
        (slice(None), joint_ids), target)
    for hand in robot.hands.values():
        hand.apply_root_pose_target = lambda target, support, control: (torch.ones((1, 3)), torch.full((1, 3), 2.))
    name, source = next(iter(binding["clips"].items()))
    library.motion_files = ["synthetic#" + name]
    if native:
        binding = dict(binding)
        placement = binding.pop("scene_from_source_world")
        source = {**source, "processing": {**source.get("processing", {}), "scene_from_source_world": placement}}
        binding["clips"] = {name: source}
        library.source_lineage = [dict(source)]
        library.native_reference_provenance = [dict(
            authority="requested_reference_not_achieved_state", source=dict(source),
            scene_from_source_world=placement)]
    else:
        library.source_lineage = [{"capture_id": source["source_unit_id"],
            "selected_source_frame_index": source["selected_source_frame_index"],
            "scene_from_source_world": binding["scene_from_source_world"]}]
    library.source_query_ordinal = [np.asarray(source["prepared_source_ordinal"])]
    library.source_import_elapsed_ns = [np.asarray(source["prepared_elapsed_time_ns"], np.int64) + 1_000_000_000]
    env.scene.env_origins = torch.tensor([[1., 2., 3.]])
    env.close = lambda: None
    if not native:
        native_command.cfg.object.entity_names = {"left": "object", "right": "object"}
    native_command._eval_mode = True
    native_command._xfrc_curr_ctrl = None
    native_command.time_left = torch.ones(1)
    if native:
        command = native_command
        action_cfg = NativeResidualActionCfg(entity_name="robot", command_name="motion",
            native_scene=asdict(command.native_layout), translation_residual_m=.1, rotation_residual_rad=1.,
            cutoff_hz={"wrist_trans": 5., "wrist_rot": 5., "finger": None}, clip={".*": (-1., 1.)})
    else:
        zeros = torch.zeros((1, 2, 3))
        quaternions = torch.tensor([[[1., 0., 0., 0.]] * 2])
        tips = torch.zeros((1, 2, 1, 3))
        command = SimpleNamespace(motion_ids=native_command.motion_ids, motion_steps=native_command.motion_steps,
            motion_lib=library, robot=robot, cfg=native_command.cfg, _eval_mode=True, _xfrc_curr_ctrl=None,
            time_left=torch.ones(1), _side_list=["left", "right"], finger_names=("index",), metrics={},
            has_objects=False, _levels=lambda: (1, 2), _apply_object_assist=lambda: None,
            mano_wrist_trans_w=zeros, robot_wrist_trans_w=zeros, mano_wrist_quat_w=quaternions,
            robot_wrist_quat_w=quaternions, mano_tip_trans_w=tips, robot_tip_trans_w=tips,
            mano_wrist_lin_vel_w=zeros, robot_wrist_lin_vel_w=zeros,
            mano_wrist_ang_vel_w=zeros, robot_wrist_ang_vel_w=zeros,
            mano_tip_lin_vel_w=tips, robot_tip_lin_vel_w=tips,
            mano_level_trans_w=lambda side, level: torch.zeros((1, 1, 3)),
            robot_level_trans_w=lambda side, level: torch.zeros((1, 1, 3)),
            mano_all_joints_lin_vel_w=lambda side: torch.zeros((1, 1, 3)),
            robot_all_joints_lin_vel_w=lambda side: torch.zeros((1, 1, 3)),
            ref_joint_pos=torch.zeros((1, 3)), sim_obj_trans_w=torch.zeros((1, 2, 3)),
            sim_obj_quat_w=quaternions)
        command._update_metrics = MethodType(MetricsMixin._update_metrics, command)
        command._update_command = MethodType(MotionTrackingCommand._update_command, command)
        action_cfg = ResidualActionCfg(entity_name="robot", command_name="motion",
            actuator_names={"wrist_trans": (), "wrist_rot": (), "finger": tuple(correspondence)},
            residual_scale={"wrist_trans": .1, "wrist_rot": 1., "finger": None}, clip={".*": (-1., 1.)})
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    if setup_scene is not None:
        setup_scene(env, command)
    if native:
        action_cfg.native_scene = asdict(command.native_layout)
    env.action_manager = ActionManager({"actual": action_cfg}, env)
    env.action_manager.reset()
    CommandTerm.compute(command, dt=0.)
    recorder = NativeRolloutRecord(binding=binding)
    recorder.on_start(env)
    assert command.reference_phase["motion_steps"].item() == 0
    assert command.motion_steps.item() == 1
    for row in (1, 2):
        raw = torch.full((1, env.action_manager.total_action_dim), float(row + 1))
        env.action_manager.process_action(raw)
        env.action_manager.apply_action()
        # Separate observed-state fixture values, not the accepted target or an invented plant.
        robot.data.joint_pos += .01
        robot.data.joint_vel[:] = row + .25
        robot.data.body_link_pos_w += .001
        if native and not rigid:
            next(iter(command.native_objects.values())).data.joint_vel[:] = row + .5
        CommandTerm.compute(command, dt=env.step_dt)
        recorder.on_step(env)
        assert command.reference_phase["motion_steps"].item() == row
    output = tmp_path / "phase-success"
    assert export_success(output, recorder, [recorder.collect_state()], torch.tensor([True]), torch.tensor([0]),
        123, "synthetic phase fixture; not physics/task success", motion_names=[name]) == 1
    return output / "manifest.json"


def test_real_author_recorder_keeps_reset_named_bodies_raw_actions_and_source_lineage(tmp_path):
    selected = [1, 2, 8]
    ordinal = np.array((0., 2.))
    imported_time = np.array((0, 16_666_667), np.int64)
    source = {"source_unit_id": "capture-001", "prepared_time_ns": [0, 58_333_333],
              "selected_source_frame_index": selected, "prepared_source_ordinal": ordinal.tolist(),
              "prepared_elapsed_time_ns": imported_time.tolist()}
    binding = {"hands": {
        "left": {"body_name": "left_wrist", "physical_joint_names": {"hinge": "l_joint"}},
        "right": {"body_name": "right_wrist", "physical_joint_names": {"hinge": "r_joint"}}},
        "object_part_names": ["base"], "scene_from_source_world": np.eye(4).tolist(),
        "object_parts": {"base": {"entity_name": "box", "body_name": "base"}},
        "clips": {"capture-001__stage-01": source}}
    robot = SimpleNamespace(joint_names=["r_joint", "l_joint"],
        body_names=["global_root", "right_wrist", "left_wrist"],
        data=SimpleNamespace(joint_pos=torch.tensor([[.1, .2]]), joint_vel=torch.tensor([[4., 8.]]),
            root_link_pos_w=torch.full((1, 3), 999.), root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
            body_link_pos_w=torch.tensor([[[999., 999., 999.], [3., 4., 5.], [6., 7., 8.]]]),
            body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]] * 3]),
            body_link_lin_vel_w=torch.tensor([[[999., 999., 999.], [1., 2., 3.], [4., 5., 6.]]]),
            body_link_ang_vel_w=torch.tensor([[[999., 999., 999.], [.1, .2, .3], [.4, .5, .6]]])))
    command = SimpleNamespace(robot=robot, motion_steps=torch.ones(1, dtype=torch.int64),
        motion_ids=torch.zeros(1, dtype=torch.int64), _xfrc_curr_ctrl=None,
        reference_phase={"motion_ids": torch.zeros(1, dtype=torch.int64),
                         "motion_steps": torch.zeros(1, dtype=torch.int64)},
        cfg=SimpleNamespace(object=SimpleNamespace(pin_objects=False)),
        sim_obj_trans_w=torch.zeros((1, 1, 3)), sim_obj_quat_w=torch.tensor([[[1., 0., 0., 0.]]]),
        motion_lib=SimpleNamespace(hand_sides=("right", "left"),
            motion_files=["file#capture-001__stage-01"], _motion_num_frames=torch.tensor([2]),
            source_lineage=[{"capture_id": "capture-001", "selected_source_frame_index": selected,
                             "scene_from_source_world": np.eye(4).tolist()}],
            source_query_ordinal=[ordinal], source_import_elapsed_ns=[imported_time]))
    class Scene(dict):
        env_origins = torch.tensor([[1., 2., 3.]])

    scene = Scene(box=SimpleNamespace(body_names=["base"],
        data=SimpleNamespace(body_link_pos_w=torch.tensor([[[8., 9., 10.]]]),
                             body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]]]))))
    env = SimpleNamespace(step_dt=.02, num_envs=1, scene=scene,
        action_manager=SimpleNamespace(action=torch.tensor([[77., 88.]]), active_terms=[]),
        command_manager=SimpleNamespace(get_term=lambda name: command))
    callback = NativeRolloutRecord(binding=binding)
    callback.on_start(env)
    command.reference_phase = {"motion_ids": command.motion_ids.clone(), "motion_steps": torch.ones(1, dtype=torch.int64)}
    robot.data.body_link_pos_w += .25
    robot.data.joint_pos += .1
    callback.on_step(env)
    state = callback.collect_state()
    output = callback.rollout_arrays([state], 0)
    np.testing.assert_array_equal(output["recorded_motion_steps"], (0, 1))
    np.testing.assert_array_equal(output["native_hand_sides"], ("left", "right"))
    np.testing.assert_array_equal(output["raw_action_valid"], (False, True))
    np.testing.assert_array_equal(output["elapsed_time_ns"], (0, 20_000_000))
    np.testing.assert_array_equal(output["native_hand_pose"][0, :, :3], ((5., 5., 5.), (2., 2., 2.)))
    np.testing.assert_array_equal(output["native_hand_pose"][1, :, :3], ((5.25, 5.25, 5.25), (2.25, 2.25, 2.25)))
    np.testing.assert_array_equal(output["joint_vel"], ((4., 8.), (4., 8.)))
    np.testing.assert_allclose(output["native_hand_twist_w"][0], ((4., 5., 6., .4, .5, .6), (1., 2., 3., .1, .2, .3)))
    np.testing.assert_array_equal(output["native_object_part_pose"][:, 0, :3], ((7., 7., 7.), (7., 7., 7.)))
    np.testing.assert_array_equal(output["actions"], ((77., 88.), (77., 88.)))
    assert np.max(output["native_hand_pose"][..., :3]) < 999.
    assert json.loads(str(output["native_binding_json"]))["clips"]["capture-001__stage-01"] == source
    assert export_success(tmp_path, callback, [state], torch.tensor([True]), torch.tensor([0]),
        20_000_000, "obj SR@1.0", motion_names=["capture-001__stage-01"]) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["selection"].startswith("first_success_ever")
    assert manifest["saved"][0]["frames"] == 2
    with np.load(tmp_path / manifest["saved"][0]["rollout"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["elapsed_time_ns"], (0, 20_000_000))
        np.testing.assert_array_equal(saved["native_hand_pose"], output["native_hand_pose"])


def test_ik_crop_preserves_full_id_inventory_and_query_origin(tmp_path):
    from retarget.robot_hand.hand_retarget import _crop

    # Load the actual numeric reader without importing the Isaac env builders.
    path = Path(__file__).parents[1] / "src/envs/commands/common/motion_library.py"
    spec = importlib.util.spec_from_file_location("flashdex_motion_library", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    lineage = {"capture_id": "capture-001", "selected_source_frame_index": [1, 2, 8]}
    clip = {"mano_left_wrist_pos": np.zeros((3, 3)), "qpos": np.arange(6).reshape(3, 2),
            "source_lineage_json": np.asarray(json.dumps(lineage)),
            "source_query_ordinal": np.array((0., 1., 2.)),
            "source_import_elapsed_ns": np.array((0, 8_333_333, 16_666_667), np.int64)}
    cropped = _crop(clip, 1)
    assert json.loads(str(cropped["source_lineage_json"])) == lineage
    np.testing.assert_array_equal(cropped["source_query_ordinal"], (1., 2.))
    np.testing.assert_array_equal(cropped["source_import_elapsed_ns"], (8_333_333, 16_666_667))
    packed = {"joint_pos": torch.from_numpy(cropped["qpos"]),
        "source_query_ordinal": torch.from_numpy(cropped["source_query_ordinal"]),
        "source_import_elapsed_ns": torch.from_numpy(cropped["source_import_elapsed_ns"]),
        "motion_num_frames": torch.tensor([2]), "length_starts": torch.tensor([0]),
        "motion_filename": ["capture-001__stage-01"], "motion_source_lineage": [lineage]}
    path = tmp_path / "motion.pt"
    torch.save(packed, path)
    loaded, names = module._load_packed_pt(str(path))
    assert names == [f"{path}#capture-001__stage-01"]
    assert loaded[0]["source_lineage"] == lineage
    np.testing.assert_array_equal(loaded[0]["source_query_ordinal"], (1., 2.))


def test_frozen_attempt_npz_keeps_actual_native_schema_clocks_interval_and_first_success(tmp_path):
    from dexforge_adaptation.flashdex_retarget import flashdex_source_mapping

    from evaluation.callbacks.success_export import export_current_attempts

    origin = 2**54 + 31
    source = flashdex_source_mapping(source_id="synthetic", source_revision="fixture",
        source_unit_id="fixture", source_episode_index=174, fold="train",
        source_frame_index=(0, 1, 2), source_time_ns=(origin, origin+20_000_000, origin+40_000_000),
        selected_source_frame_index=(0, 1, 2), prepared_source_ordinal=(0., 1., 2.),
        prepared_elapsed_time_ns=(0, 20_000_000, 40_000_000),
        clock_authority="synthetic exact source clock, independent of physical time", processing={"fixture": True})
    binding = dict(producer_identity={"method": "synthetic serialization, not physics"},
        scene_identity={"fixture": True}, scene_from_source_world=np.eye(4).tolist(),
        assistance_receipt={"evaluation_external_wrenches_cleared": True, "evaluation_object_playback": False},
        hands={side: {"body_name": side+"_root", "command_joint_names": {"driver": side+"_driver"},
                      "physical_joint_names": {"driver": side+"_driver"}} for side in ("left", "right")},
        object_part_names=["base", "lid"], object_parts={name: {"entity_name": "object", "body_name": body}
            for name, body in (("base", "object_root"), ("lid", "object_lid"))}, clips={"fixture": source})

    def setup(env, command):
        def contacts():
            row = int(command.reference_phase["motion_steps"].item())
            addresses = {side+"/"+name: dict(hand_side=side, hand_body_name=side+"_root",
                part_entity_name="object", part_body_name=part["body_name"])
                for side in ("left", "right") for name, part in binding["object_parts"].items()}
            return dict(addresses=addresses, pairs={name: {
                "normal_force_n": torch.tensor([.75 if row else float("nan")]),
                "force_bearing_contact_present": torch.tensor([row > 0]),
                "force_bearing_distance_max_m": torch.tensor([.01 if row else float("nan")])} for name in addresses},
                measurement_support=torch.tensor([row > 0]), sample_time_s=torch.tensor([1.25+row*.02], dtype=torch.float64),
                clock="synthetic solver", normal_force_authority="synthetic protocol", distance_authority="synthetic protocol")

        def interval():
            facts = contacts()
            row = int(command.reference_phase["motion_steps"].item())
            begin = 1.25+(row-1)*.02+torch.arange(3, dtype=torch.float64)[None]*(.02/3)
            if not row:
                begin[:] = float("nan")
            return {**facts, "num_substeps": 3, "sampling": "synthetic discrete protocol",
                "pairs": {name: {"normal_force_n": torch.tensor([[.75, 0., .4] if row else [float("nan")]*3]),
                    "force_bearing_contact_present": torch.tensor([[True, False, True] if row else [False]*3]),
                    "force_bearing_distance_max_m": torch.tensor([[.01, float("nan"), .03] if row else [float("nan")]*3])}
                    for name in facts["addresses"]},
                "measurement_support": torch.full((1, 3), row > 0), "sample_recorded": torch.full((1, 3), row > 0),
                "clock_support": torch.full((1, 3), row > 0), "interval_complete": torch.tensor([row > 0]),
                "sample_begin_time_s": begin, "sample_end_time_s": begin+.02/3}

        env.scene.hand_part_contact_facts = contacts
        env.scene.hand_part_contact_interval_facts = interval
        env.scene.physics_time_s = lambda: torch.tensor(
            [1.25+int(command.reference_phase["motion_steps"].item())*.02], dtype=torch.float64)

    success_manifest = record_reference_phase_fixture(tmp_path, binding, native=True, setup_scene=setup)
    original_manifest = success_manifest.read_bytes()
    first = json.loads(original_manifest)["saved"][0]
    with np.load(success_manifest.parent / first["rollout"], allow_pickle=False) as old:
        arrays = {key: old[key].copy() for key in old.files}
    recorder = SimpleNamespace(rollout_arrays=lambda states, row: arrays if row == 0 else None)
    authority = {"checkpoint": "/fixture/frozen-full", "source_revision": "fixture-source"}
    manifest_path, motions = export_current_attempts(tmp_path / "frozen", recorder, [], [
        {"evaluation_row": 0, "batch_index": 0, "env_index": 1, "motion_id": 0},
        {"evaluation_row": 1, "batch_index": 1, "env_index": 0, "motion_id": 0}],
        evaluation_id="frozen-fixture", env_step=123, frozen_authority=authority,
        motion_names=["fixture", "missing"], num_per_motion=2, motion_file="fixture-motion.pt",
        evaluation_report=tmp_path / "current-policy.json")
    entry = motions[0]["attempts"][0]
    with np.load(manifest_path.parent / entry["recording"]["rollout"], allow_pickle=False) as saved:
        for key, value in arrays.items():
            assert saved[key].dtype == value.dtype
            np.testing.assert_array_equal(saved[key], value)
        identity = json.loads(str(saved["evaluation_attempt_json"]))
        assert identity == {"evaluation_id": "frozen-fixture", "attempt_id": "m00000-a00000", "motion_id": 0,
            "replica_index": 0, "evaluation_row": 0, "env_step": 123, "frozen_authority": authority}
        assert json.loads(str(saved["native_binding_json"]))["clips"]["fixture"]["prepared_time_ns"][0] == origin
        np.testing.assert_array_equal(saved["recorded_motion_steps"], (0, 1, 2))
        assert saved["command_motion_steps"][-1] == 2  # Keep actual clamped cursor distinct.
        np.testing.assert_array_equal(saved["native_hand_sides"], ("left", "right"))
        assert saved["native_physics_time_s"].dtype == np.float64
        assert saved["hand_part_interval_sample_begin_time_s"].dtype == np.float64
        assert saved["hand_part_interval_sample_recorded"].dtype == bool
        assert not saved["hand_part_interval_sample_recorded"][0].any()
        assert np.isnan(saved["hand_part_interval_normal_force_n"][0]).all()
    assert motions[0]["attempts"][1]["recording"]["reason"] == "recorder_returned_no_factual_tape"
    assert all(row["recording"]["reason"] == "attempt_not_evaluated" for row in motions[1]["attempts"])
    assert export_success(success_manifest.parent, recorder, [], torch.tensor([True]), torch.tensor([0]),
        99999, "fixture", motion_names=["fixture"]) == 0
    assert success_manifest.read_bytes() == original_manifest


def _prepared_native_recorder_fixture(tmp_path, *, hand_sides=("left", "right"), source_intent=False):
    """Real prepared-input/pack/library admission; only achieved state is synthetic."""
    from dexforge_adaptation.flashdex_retarget import flashdex_source_mapping
    from test_native_reference import _pipeline_config, _prepared_native_dataset

    from envs.commands.common.motion_library import MotionLibrary
    from retarget.human_demo import clips as C
    from retarget.kinematic_retargeting import PreparedNativeInput

    dataset, clip = _prepared_native_dataset(tmp_path)
    elapsed = np.arange(3, dtype=np.int64) * 20_000_000
    origin = 2**60 + 37
    source = flashdex_source_mapping(source_id="synthetic", source_revision="synthetic-fixture",
        source_unit_id="episode-000174", source_episode_index=174, fold="train",
        source_frame_index=(0, 1, 2), source_time_ns=origin + elapsed,
        selected_source_frame_index=(0, 1, 2), prepared_source_ordinal=(0., 1., 2.),
        prepared_elapsed_time_ns=elapsed, clock_authority="synthetic exact source clock",
        processing={"fixture": "recorder software admission, not physical execution"})
    placement = np.eye(4)
    placement[:3, 3] = (1., -2., .3)
    source["processing"]["scene_from_source_world"] = placement.tolist()
    native = clip["native_reference"]
    physical_joints = {"left": ("left_a", "left_mimic"), "right": ("right_b",)}
    joint_names = tuple(name for side in ("left", "right") if side in hand_sides
                        for name in physical_joints[side])
    columns = [i for i, name in enumerate(native["joint_names"]) if name in joint_names]
    native["joint_names"] = tuple(native["joint_names"][i] for i in columns)
    for field in ("joint_pos", "joint_vel", "joint_position_support", "joint_velocity_support"):
        native[field] = native[field][:, columns]
    for field in ("root_body_names", "root_state_w", "root_pose_support",
                  "root_linear_velocity_support", "root_angular_velocity_support"):
        native[field] = {side: native[field][side] for side in hand_sides}
    for absent_side in set(("left", "right")) - set(hand_sides):
        for key in list(clip):
            if key.startswith((f"mano_{absent_side}_", f"obj_{absent_side}_")):
                del clip[key]
    native["time_ns"] = elapsed.copy()
    native["provenance"] = dict(authority="requested_reference_not_achieved_state",
        source=dict(source), scene_from_source_world=placement.tolist())
    clip["source_lineage_json"] = np.asarray(json.dumps(source))
    clip["source_query_ordinal"] = np.asarray(source["prepared_source_ordinal"])
    clip["source_import_elapsed_ns"] = elapsed.copy()
    if source_intent:
        from test_native_reference import _with_source_hand_part_intent

        _with_source_hand_part_intent(clip)
    C.save_clip(dataset / "native_reference/synthetic-stage.npz", clip)
    cfg = _pipeline_config(dataset)
    prepared = PreparedNativeInput(dataset, cfg, workers=1)
    prepared.pack()

    robot = SimpleNamespace(num_joints=len(joint_names), joint_names=joint_names,
        body_names=tuple(native["root_body_names"][side] for side in hand_sides), data=SimpleNamespace(
            joint_pos=torch.zeros(1, len(joint_names)), joint_vel=torch.zeros(1, len(joint_names)),
            root_link_pos_w=torch.zeros(1, 3), root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
            body_link_pos_w=torch.zeros(1, len(hand_sides), 3),
            body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]] * len(hand_sides)]),
            body_link_lin_vel_w=torch.zeros(1, len(hand_sides), 3),
            body_link_ang_vel_w=torch.zeros(1, len(hand_sides), 3)))
    library = MotionLibrary(str(dataset / "motion.pt"), robot, finger_names=C.FINGERS, device="cpu")
    command = SimpleNamespace(robot=robot, motion_lib=library,
        motion_ids=torch.zeros(1, dtype=torch.int64), motion_steps=torch.ones(1, dtype=torch.int64),
        reference_phase={"motion_ids": torch.zeros(1, dtype=torch.int64),
                         "motion_steps": torch.zeros(1, dtype=torch.int64)},
        cfg=SimpleNamespace(object=SimpleNamespace(pin_objects=False)), _xfrc_curr_ctrl=None,
        sim_obj_trans_w=torch.zeros(1, len(hand_sides), 3),
        sim_obj_quat_w=torch.tensor([[[1., 0., 0., 0.]] * len(hand_sides)]))

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene(articulated_object=SimpleNamespace(body_names=("lid", "base"),
        data=SimpleNamespace(body_link_pos_w=torch.zeros(1, 2, 3),
            body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]] * 2]))))
    env = SimpleNamespace(step_dt=.02, num_envs=1, scene=scene,
        action_manager=SimpleNamespace(action=torch.zeros(1, len(joint_names)), active_terms=[]),
        command_manager=SimpleNamespace(get_term=lambda name: command))
    binding = dict(hands={"left": dict(body_name="left_root", physical_joint_names={"driver": "left_a", "mimic": "left_mimic"}),
               "right": dict(body_name="right_root", physical_joint_names={"driver": "right_b"})},
        object_part_names=["lid", "base"], object_parts={name: dict(entity_name="articulated_object", body_name=name)
            for name in ("lid", "base")}, clips={"synthetic-stage": source})
    binding["hands"] = {side: binding["hands"][side] for side in hand_sides}
    return env, binding


def test_source_hand_part_intent_export_keeps_consumed_phase_clocks_and_owned_masks(tmp_path):
    from envs.commands.motion_tracking.native import NativeMotionTrackingCommand

    env, binding = _prepared_native_recorder_fixture(tmp_path, source_intent=True)
    command = env.command_manager.get_term("motion")
    command._side_list, command._env = ["left", "right"], env
    for name in ("source_hand_part_intent", "_consumed_reference_phase", "source_hand_part_intent_facts"):
        setattr(command, name, MethodType(getattr(NativeMotionTrackingCommand, name), command))
    installed = dict(installed_valid=torch.tensor([False]), reference_motion_ids=torch.tensor([0]),
                     reference_motion_steps=torch.tensor([0]))
    env.action_manager._terms = {"native": SimpleNamespace(installed_control=installed)}
    recorder = NativeRolloutRecord(binding=binding)
    recorder.on_start(env)
    # Installation owns the consumed packet even after the command advances.
    installed["installed_valid"][:] = True
    installed["reference_motion_steps"][:] = 2
    command.reference_phase["motion_steps"][:] = 1
    command.motion_steps[:] = 2
    recorder.on_step(env)
    state = recorder.collect_state(torch.tensor([True]))
    command.motion_lib.native_source_hand_part_intent["active"][:] = False
    recorder.source_hand_part_intent_metadata_json = "not this collected batch"
    arrays = recorder.rollout_arrays([state], 0)
    np.testing.assert_array_equal(arrays["source_hand_part_intent_reference_motion_steps"], [0, 2])
    np.testing.assert_array_equal(arrays["source_hand_part_intent_execution_valid"], [False, True])
    np.testing.assert_array_equal(arrays["source_hand_part_intent_query_time_ns"],
                                  2**60 + 37 + np.array((0, 40_000_000), np.int64))
    assert arrays["source_hand_part_intent_capture_time_ns"].dtype == np.int64
    assert arrays["source_hand_part_intent_known"].dtype == bool
    assert arrays["source_hand_part_intent_active"][0, 0, 1]
    assert arrays["source_hand_part_intent_active"][1, 0, 0]
    assert not arrays["source_hand_part_intent_known"][1, 0, 1]
    metadata = json.loads(str(arrays["source_hand_part_intent_metadata_json"]))
    assert metadata["hand_sides"] == ["left", "right"]
    assert metadata["part_body_names"] == ["lid", "base"]
    assert metadata["source_provenance"][0]["provenance"]["resource"].startswith("synthetic source")


def test_prepared_native_input_pack_library_recorder_keeps_canonical_source_and_placement(tmp_path):
    env, binding = _prepared_native_recorder_fixture(tmp_path)
    library = env.command_manager.get_term("motion").motion_lib
    assert "capture_id" not in library.source_lineage[0]
    assert "scene_from_source_world" not in library.source_lineage[0]
    assert library.source_lineage[0] == binding["clips"]["synthetic-stage"]
    assert library.native_reference_provenance[0]["source"] == library.source_lineage[0]
    recorder = NativeRolloutRecord(binding=binding)
    recorder.on_start(env)
    assert recorder.rollout_steps == 2
    arrays = recorder.rollout_arrays([recorder.collect_state()], 0)
    np.testing.assert_array_equal(arrays["recorded_motion_steps"], [0])
    assert json.loads(str(arrays["native_binding_json"])) == binding


def test_native_recorder_selects_clip_placement_within_a_shared_scene(tmp_path):
    env, binding = _prepared_native_recorder_fixture(tmp_path)
    command = env.command_manager.get_term("motion")
    library = command.motion_lib
    first = binding["clips"]["synthetic-stage"]
    placement = np.eye(4)
    placement[:3, 3] = (-0.2, 0.4, 0.8)
    second = {**first, "source_unit_id": "second-fixture",
              "processing": {**first["processing"], "scene_from_source_world": placement.tolist()}}
    binding["clips"]["second-stage"] = second
    library.motion_files.append("motion.pt#second-stage")
    library._motion_num_frames = library._motion_num_frames.repeat(2)
    library.source_lineage.append(second)
    library.source_query_ordinal.append(library.source_query_ordinal[0])
    library.source_import_elapsed_ns.append(library.source_import_elapsed_ns[0])
    library.native_reference_provenance.append({
        **library.native_reference_provenance[0], "source": second,
        "scene_from_source_world": placement.tolist(),
    })
    command.motion_ids[:] = 1
    command.reference_phase["motion_ids"][:] = 1

    recorder = NativeRolloutRecord(binding=binding)
    recorder.on_start(env)
    arrays = recorder.rollout_arrays([recorder.collect_state()], 0)
    recorded = json.loads(str(arrays["native_binding_json"]))
    assert "scene_from_source_world" not in recorded
    assert int(arrays["motion_id"]) == 1
    assert recorded["clips"] == {"second-stage": second}


@pytest.mark.parametrize("mismatch", ["source", "placement", "authority"])
def test_prepared_native_recorder_rejects_different_reference_source_or_placement(tmp_path, mismatch):
    env, binding = _prepared_native_recorder_fixture(tmp_path)
    provenance = env.command_manager.get_term("motion").motion_lib.native_reference_provenance[0]
    if mismatch == "source":
        provenance["source"] = {**provenance["source"], "source_revision": "another-release"}
    elif mismatch == "placement":
        provenance["scene_from_source_world"][0][3] += .01
    else:
        provenance["authority"] = "achieved_state"
    with pytest.raises(ValueError, match="imported frame/scene lineage"):
        NativeRolloutRecord(binding=binding).on_start(env)


def test_isaac_callback_clock_survives_ordinary_native_recorder_and_reader(monkeypatch, tmp_path):
    from dexforge_adaptation.flashdex_retarget import load_flashdex_current_attempt
    from test_scene_contact_authority import _isaac_scene_type

    from evaluation.callbacks.success_export import export_current_attempts

    env, binding = _prepared_native_recorder_fixture(tmp_path)
    binding.update(
        producer_identity={"method": "synthetic clock protocol, not physical execution"},
        scene_identity={"fixture": True},
        assistance_receipt={"evaluation_external_wrenches_cleared": True, "evaluation_object_playback": False},
    )
    for hand in binding["hands"].values():
        hand["command_joint_names"] = {"driver": hand["physical_joint_names"]["driver"]}
    command = env.command_manager.get_term("motion")
    sdk_scene = SimpleNamespace(
        cfg=SimpleNamespace(num_envs=env.num_envs), env_origins=env.scene.env_origins,
    )
    context = SimpleNamespace(current_time=7.125000000000007)
    scene = _isaac_scene_type(monkeypatch)(sdk_scene, "cpu", context)
    scene.entities.update(env.scene)
    env.scene = scene
    callback = NativeRolloutRecord(binding=binding)
    callback.on_start(env)
    times = [context.current_time, 7.144999999552972, 7.165000123456789]
    for row, time_s in enumerate(times[1:], 1):
        context.current_time = time_s
        command.reference_phase["motion_steps"][:] = row
        command.motion_steps[:] = min(row + 1, 2)
        callback.on_step(env)
    state = callback.collect_state()
    manifest, _motions = export_current_attempts(
        tmp_path / "clock-attempts", callback, [state],
        [dict(motion_id=0, evaluation_row=0, batch_index=0, env_index=0)],
        env_step=17, motion_names=["synthetic-stage"], motion_file=command.motion_lib.motion_file,
        num_per_motion=1, evaluation_id="clock-protocol",
        frozen_authority={"checkpoint": "synthetic-not-a-policy.pt", "source_revision": "protocol-only"},
        evaluation_report=tmp_path / "current-policy.json",
    )
    recorded = load_flashdex_current_attempt(manifest, attempt_id="m00000-a00000")
    arrays = recorded.arrays
    assert arrays["native_physics_time_s"].dtype == np.float64
    np.testing.assert_array_equal(arrays["native_physics_time_s"], times)
    assert arrays["native_physics_reset_time_s"] == times[0]
    np.testing.assert_array_equal(arrays["elapsed_time_ns"], [0, 20_000_000, 40_000_000])
    np.testing.assert_array_equal(arrays["recorded_motion_steps"], [0, 1, 2])
    metadata = json.loads(str(arrays["native_backend_metadata_json"]))
    assert metadata["physics_clock"] == scene.backend_identity["physics_clock"]
    clock = json.loads(str(arrays["native_physics_clock_metadata_json"]))
    assert "owning scene" in clock["authority"]
    assert clock["origin"] == "first recorded reset sample; raw backend time retained"
    assert "native_execution_steps" not in arrays
    assert "hand_part_interval_metadata_json" not in arrays
    assert "hand_part_contact_metadata_json" not in arrays


@pytest.mark.parametrize("hand_sides", [("left",), ("right",), ("right", "left")],
                         ids=["left", "right", "bilateral"])
def test_available_native_hands_survive_prepared_pack_library_and_factual_export(tmp_path, hand_sides):
    from evaluation.callbacks.success_export import export_current_attempts

    env, binding = _prepared_native_recorder_fixture(tmp_path, hand_sides=hand_sides)
    command = env.command_manager.get_term("motion")
    assert set(command.motion_lib.hand_sides) == set(hand_sides)
    assert set(command.motion_lib.native_root_body_names) == set(hand_sides)
    expected_sides = ("left", "right") if len(hand_sides) == 2 else hand_sides
    # Physical body columns and binding insertion order must not set the archive hand axis.
    robot = command.robot
    robot.body_names = tuple(reversed(robot.body_names))
    hand_position = {"left": (10., 11., 12.), "right": (20., 21., 22.)}
    for side in hand_sides:
        column = robot.body_names.index(binding["hands"][side]["body_name"])
        robot.data.body_link_pos_w[0, column] = torch.tensor(hand_position[side])
        robot.data.body_link_lin_vel_w[0, column] = torch.tensor(hand_position[side]) + 1.
        robot.data.body_link_ang_vel_w[0, column] = torch.tensor(hand_position[side]) + 2.
    named_pairs = [
        (f"pair-{index}", dict(hand_side=side, hand_body_name=binding["hands"][side]["body_name"],
            part_entity_name=part["entity_name"], part_body_name=part["body_name"]), index + .5)
        for index, (side, part) in enumerate((side, part) for side in expected_sides
                                           for part in binding["object_parts"].values())
    ]

    def contacts():
        executed = bool(command.reference_phase["motion_steps"].item())
        return dict(addresses={name: address for name, address, _force in reversed(named_pairs)},
            pairs={name: dict(normal_force_n=torch.tensor([force if executed else float("nan")]),
                force_bearing_contact_present=torch.tensor([executed]),
                force_bearing_distance_max_m=torch.tensor([.01 if executed else float("nan")]))
                for name, _address, force in reversed(named_pairs)},
            measurement_support=torch.tensor([executed]),
            sample_time_s=torch.tensor([1.25 if executed else float("nan")], dtype=torch.float64),
            clock="synthetic stopped protocol", normal_force_authority="synthetic protocol facts",
            distance_authority="synthetic protocol facts")

    env.scene.hand_part_contact_facts = contacts
    callback = NativeRolloutRecord(binding=binding)
    callback.on_start(env)
    command.reference_phase["motion_steps"][:] = 1
    command.motion_steps[:] = 2
    callback.on_step(env)
    manifest, motions = export_current_attempts(
        tmp_path / "available-hand-attempts", callback, [callback.collect_state()],
        [dict(motion_id=0, evaluation_row=0, batch_index=0, env_index=0)],
        env_step=0, motion_names=["synthetic-stage"], motion_file=command.motion_lib.motion_file,
        num_per_motion=1, evaluation_id="available-hand-protocol",
        frozen_authority={"kind": "serialization_protocol", "checkpoint": None, "learned_policy": False},
        evaluation_report=tmp_path / "current-policy.json",
    )
    entry = motions[0]["attempts"][0]["recording"]
    assert entry["status"] == "recorded" and entry["frames"] == 2
    assert "native_hand_sides" in entry["fields"]
    with np.load(manifest.parent / entry["rollout"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["native_hand_sides"], expected_sides)
        assert arrays["native_hand_pose"].shape == (2, len(expected_sides), 7)
        assert arrays["native_hand_twist_w"].shape == (2, len(expected_sides), 6)
        np.testing.assert_array_equal(arrays["native_hand_pose"][0, :, :3],
            [hand_position[side] for side in expected_sides])
        np.testing.assert_array_equal(arrays["native_hand_twist_w"][0, :, :3],
            np.asarray([hand_position[side] for side in expected_sides]) + 1.)
        np.testing.assert_array_equal(arrays["native_hand_twist_w"][0, :, 3:],
            np.asarray([hand_position[side] for side in expected_sides]) + 2.)
        assert np.isnan(arrays["hand_part_normal_force_n"][0]).all()
        assert not arrays["hand_part_measurement_support"][0].any()
        np.testing.assert_array_equal(arrays["hand_part_normal_force_n"][1],
            [force for _name, _address, force in named_pairs])
        contact_metadata = json.loads(str(arrays["hand_part_contact_metadata_json"]))
        assert contact_metadata["pairs"] == [address for _name, address, _force in named_pairs]
        np.testing.assert_array_equal(arrays["recorded_motion_steps"], (0, 1))
        assert json.loads(str(arrays["native_binding_json"]))["clips"] == binding["clips"]
    bad_binding = {**binding, "hands": {}}
    with pytest.raises(ValueError, match="actual motion library sides"):
        NativeRolloutRecord(binding=bad_binding).on_start(env)


def _source_geometry_protocol(env, command, hand_sides, part_names, part_body_names):
    """Stopped public endpoint facts; no geometry/SDF/physics is calculated here."""
    widths = tuple(index + 2 for index, _side in enumerate(hand_sides))
    offsets = [0]
    for width in widths:
        offsets.append(offsets[-1] + width)
    point_sides = tuple(side for side, width in zip(hand_sides, widths, strict=True) for _ in range(width))
    metadata = dict(schema="native_source_tip_geometry_v1", hand_sides=tuple(hand_sides), hand_offsets=offsets,
        point_hand_sides=point_sides,
        source_point_names=tuple(f"source_{side}_{index}" for side, width in zip(hand_sides, widths, strict=True)
                                 for index in range(width)),
        target_site_names=tuple(f"actual_{side}_site_{index}" for side, width in zip(hand_sides, widths, strict=True)
                                for index in range(width)),
        part_names=tuple(part_names), part_body_names=tuple(part_body_names),
        part_distance_sign=tuple("unsigned" for _ in part_names), position_frame="WORLD includes env origin once",
        position_unit="m", distance_unit="m", query_authority="synthetic in-domain facts, not contact truth",
        reference_authority="consumed installed packet or actual reset reference_phase")
    trace = []

    def facts():
        phase = command.reference_phase
        batch, points, parts = env.num_envs, len(point_sides), len(part_names)
        ids = command.motion_lib.length_starts[phase["motion_ids"]] + phase["motion_steps"]
        local = (torch.arange(points * 3, dtype=torch.float64).reshape(1, points, 3) / 8
                 + torch.arange(batch, dtype=torch.float64)[:, None, None] * 64
                 + phase["motion_steps"][:, None, None] / 4)
        source = local + env.scene.env_origins[:, None]
        source_support = torch.ones(batch, points, dtype=torch.bool)
        source_support[:, 1] = False
        achieved_support = torch.ones_like(source_support)
        achieved_support[:, 0] = False
        desired_pose = torch.ones(batch, parts, dtype=torch.bool)
        desired_pose[:, 0] = False
        live_pose = torch.ones_like(desired_pose)
        desired_query = source_support[:, :, None] & desired_pose[:, None, :]
        live_query = achieved_support[:, :, None] & live_pose[:, None, :]
        desired_query[:, 0, -1] = False
        live_query[:, -1, 0] = False
        distance = torch.arange(points * parts, dtype=torch.float64).reshape(1, points, parts).expand(batch, -1, -1) / 32
        controls = [env.action_manager.get_term(name).installed_control for name in env.action_manager.active_terms
                    if hasattr(env.action_manager.get_term(name), "installed_control")]
        execution = controls[0]["installed_valid"].clone() if controls else phase["motion_steps"] > 0
        result = dict(metadata=metadata, reference_motion_ids=phase["motion_ids"].clone(),
            reference_motion_steps=phase["motion_steps"].clone(), reference_time_ns=command.motion_lib.native_time_ns[ids],
            execution_valid=execution, source_position_w=source, source_position_support=source_support,
            achieved_position_w=source + .5, achieved_position_support=achieved_support,
            desired_part_pose_support=desired_pose, live_part_pose_support=live_pose,
            desired_part_distance_m=distance.clone(), live_part_distance_m=distance + .125,
            desired_part_query_support=desired_query, live_part_query_support=live_query)
        trace.append(dict(facts=result, local=local.clone()))
        return result

    command.source_tip_geometry_facts = facts
    return metadata, trace


@pytest.mark.parametrize("hand_sides", [("left",), ("right",), ("right", "left")],
                         ids=["left", "right", "ragged-bilateral"])
def test_source_tip_geometry_active_mask_and_unscored_attempt_reopen(tmp_path, hand_sides):
    from dexforge_adaptation.flashdex_retarget import load_flashdex_current_attempt

    from evaluation.callbacks.success_export import export_current_attempts

    env, binding = _prepared_native_recorder_fixture(tmp_path, hand_sides=hand_sides)
    command = env.command_manager.get_term("motion")
    binding.update(producer_identity={"method": "synthetic geometry serialization, not physics"},
        scene_identity={"fixture": True},
        assistance_receipt={"evaluation_external_wrenches_cleared": True, "evaluation_object_playback": False})
    for hand in binding["hands"].values():
        hand["command_joint_names"] = {"driver": hand["physical_joint_names"]["driver"]}
    env.num_envs = 3
    env.scene.env_origins = torch.tensor([[8., 16., 24.], [32., 40., 48.], [56., 64., 72.]], dtype=torch.float64)
    for entity in (command.robot, *env.scene.values()):
        for field, value in vars(entity.data).items():
            if isinstance(value, torch.Tensor):
                setattr(entity.data, field, value.repeat((env.num_envs,) + (1,) * (value.ndim - 1)))
    env.action_manager.action = env.action_manager.action.repeat(env.num_envs, 1)
    command.sim_obj_trans_w = command.sim_obj_trans_w.repeat(env.num_envs, 1, 1)
    command.sim_obj_quat_w = command.sim_obj_quat_w.repeat(env.num_envs, 1, 1)
    command.motion_ids = command.motion_ids.repeat(env.num_envs)
    command.motion_steps = command.motion_steps.repeat(env.num_envs)
    command.reference_phase = {key: value.repeat(env.num_envs) for key, value in command.reference_phase.items()}
    clock_origin = 2**60 + 37
    command.motion_lib.native_time_ns += clock_origin
    metadata, trace = _source_geometry_protocol(env, command, hand_sides,
        tuple(binding["object_part_names"]), tuple(part["body_name"] for part in binding["object_parts"].values()))
    callback = NativeRolloutRecord(binding=binding)
    callback.on_start(env)
    assert len(trace) == 1  # Reset is one endpoint, not an extra setup query.
    # A producer can reuse its current scratch without changing a saved row.
    trace[0]["facts"]["source_position_w"][:] = -999.
    trace[0]["facts"]["source_position_support"][:] = False
    command.reference_phase["motion_steps"][:] = 1
    command.motion_steps[:] = 2
    callback.on_step(env)
    assert len(trace) == 2
    state = callback.collect_state(torch.tensor([False, True, True]))
    # Collected batch metadata cannot be replaced by a later callback batch.
    callback.source_tip_geometry_metadata_json = "not this collected batch"
    manifest, motions = export_current_attempts(tmp_path / "unscored-attempts", callback, [state], [
        dict(motion_id=0, evaluation_row=0, batch_index=0, env_index=1),
        dict(motion_id=0, evaluation_row=1, batch_index=0, env_index=2)],
        env_step=0, motion_names=["synthetic-stage"], motion_file=command.motion_lib.motion_file,
        num_per_motion=3, evaluation_id="geometry-serialization",
        frozen_authority={"kind": "unscored geometry protocol, no success predicate"},
        evaluation_report=tmp_path / "unscored-current-policy.json")
    assert motions[0]["attempts"][2]["recording"]["reason"] == "attempt_not_evaluated"
    for index, original_env in enumerate((1, 2)):
        if len(hand_sides) == 2:
            arrays = load_flashdex_current_attempt(manifest, attempt_id=f"m00000-a{index:05d}").arrays
        else:
            # The general fork export permits one side; the Core bridge is bilateral.
            recording = motions[0]["attempts"][index]["recording"]
            with np.load(manifest.parent / recording["rollout"], allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
        saved = json.loads(str(arrays["source_tip_geometry_metadata_json"]))
        assert saved["hand_sides"] == list(hand_sides)
        assert saved["hand_offsets"] == metadata["hand_offsets"]
        assert saved["source_point_names"] == list(metadata["source_point_names"])
        assert saved["target_site_names"] == list(metadata["target_site_names"])
        assert saved["part_names"] == list(binding["object_part_names"])
        assert saved["position_frame"] == "env-local (scene env_origins subtracted)"
        assert saved["producer_position_frame"] == metadata["position_frame"]
        for field in ("reference_motion_ids", "reference_motion_steps", "reference_time_ns"):
            assert arrays["source_tip_geometry_" + field].dtype == np.int64
        np.testing.assert_array_equal(arrays["source_tip_geometry_reference_motion_steps"], [0, 1])
        np.testing.assert_array_equal(arrays["command_motion_steps"], [1, 2])
        np.testing.assert_array_equal(arrays["source_tip_geometry_reference_time_ns"],
                                      [clock_origin, clock_origin + 20_000_000])
        np.testing.assert_array_equal(arrays["source_tip_geometry_execution_valid"], [False, True])
        expected = np.stack([row["local"][original_env].numpy() for row in trace])
        np.testing.assert_array_equal(arrays["source_tip_geometry_source_position"], expected)
        np.testing.assert_array_equal(arrays["source_tip_geometry_achieved_position"], expected + .5)
        assert arrays["source_tip_geometry_source_position"].dtype == np.float64
        assert arrays["source_tip_geometry_source_position_support"][0, 0]
        assert not arrays["source_tip_geometry_source_position_support"][:, 1].any()
        assert not arrays["source_tip_geometry_achieved_position_support"][:, 0].any()
        for field in ("source_position_support", "achieved_position_support", "desired_part_pose_support",
                      "live_part_pose_support", "desired_part_query_support", "live_part_query_support"):
            assert arrays["source_tip_geometry_" + field].dtype == bool
        for field in ("desired_part_pose_support", "live_part_pose_support", "desired_part_distance_m",
                      "live_part_distance_m", "desired_part_query_support", "live_part_query_support"):
            np.testing.assert_array_equal(arrays["source_tip_geometry_" + field][1],
                                          trace[1]["facts"][field][original_env].numpy())
        assert arrays["source_tip_geometry_desired_part_distance_m"].shape == (
            2, len(metadata["source_point_names"]), len(metadata["part_names"]))
        assert not any(key.endswith("position_w") for key in arrays if key.startswith("source_tip_geometry_"))


def test_source_tip_geometry_uses_real_consumed_phase_through_final_endpoint(tmp_path):
    from dexforge_adaptation.flashdex_retarget import flashdex_source_mapping

    clock_origin = 2**60 + 37
    source = flashdex_source_mapping(source_id="synthetic", source_revision="fixture", source_unit_id="fixture",
        source_episode_index=174, fold="train", source_frame_index=(0, 1, 2),
        source_time_ns=(clock_origin, clock_origin + 20_000_000, clock_origin + 40_000_000),
        selected_source_frame_index=(0, 1, 2), prepared_source_ordinal=(0., 1., 2.),
        prepared_elapsed_time_ns=(0, 20_000_000, 40_000_000),
        clock_authority="synthetic exact reference clock", processing={"fixture": True})
    binding = dict(producer_identity={"method": "synthetic recorder phase protocol"},
        scene_identity={"fixture": True}, scene_from_source_world=np.eye(4).tolist(),
        assistance_receipt={"evaluation_external_wrenches_cleared": True, "evaluation_object_playback": False},
        hands={side: dict(body_name=side + "_root", command_joint_names={"driver": side + "_driver"},
                         physical_joint_names={"driver": side + "_driver"}) for side in ("left", "right")},
        object_part_names=["base", "lid"], object_parts={name: dict(entity_name="object", body_name=body)
            for name, body in (("base", "object_root"), ("lid", "object_lid"))}, clips={"fixture": source})
    trace = []

    def setup(env, command):
        command.motion_lib.native_time_ns = (
            command.motion_lib.native_time_ns - command.motion_lib.native_time_ns[0] + clock_origin
        )
        parts = command.native_layout.objects[0].parts
        _metadata, rows = _source_geometry_protocol(env, command, command._side_list,
            tuple(part.name for part in parts), tuple(part.body_name for part in parts))
        trace.append(rows)

    manifest = record_reference_phase_fixture(tmp_path, binding, native=True, setup_scene=setup)
    archive = json.loads(manifest.read_text())
    with np.load(manifest.parent / archive["saved"][0]["rollout"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["recorded_motion_steps"], [0, 1, 2])
        np.testing.assert_array_equal(arrays["command_motion_steps"], [1, 2, 2])
        np.testing.assert_array_equal(arrays["source_tip_geometry_reference_motion_ids"], [0, 0, 0])
        np.testing.assert_array_equal(arrays["source_tip_geometry_reference_motion_steps"], [0, 1, 2])
        np.testing.assert_array_equal(arrays["source_tip_geometry_reference_time_ns"],
            [clock_origin, clock_origin + 20_000_000, clock_origin + 40_000_000])
        np.testing.assert_array_equal(arrays["source_tip_geometry_execution_valid"], [False, True, True])
        assert arrays["source_tip_geometry_source_position_support"][0].any()
        assert len(trace[0]) == 3


def test_source_tip_geometry_absent_modality_exports_no_invented_group(tmp_path):
    env, binding = _prepared_native_recorder_fixture(tmp_path, hand_sides=("right",))
    command = env.command_manager.get_term("motion")
    command.source_tip_geometry_facts = lambda: None
    callback = NativeRolloutRecord(binding=binding)
    callback.on_start(env)
    arrays = callback.rollout_arrays([callback.collect_state()], 0)
    assert not any(key.startswith("source_tip_geometry_") for key in arrays)


def test_native_multiple_objects_roundtrip_named_parts_passive_units_and_assistance(tmp_path):
    from dexforge_adaptation.flashdex_retarget import (
        flashdex_source_mapping,
        load_flashdex_rollout,
    )
    from test_native_task_contract import _add_loose_object, _declare_joint_coordinates

    source = flashdex_source_mapping(source_id="synthetic", source_revision="synthetic", source_unit_id="fixture",
        source_episode_index=0, fold="train", source_frame_index=(0, 1, 2),
        source_time_ns=(0, 20_000_000, 40_000_000), selected_source_frame_index=(0, 1, 2),
        prepared_source_ordinal=(0., 1., 2.), prepared_elapsed_time_ns=(0, 20_000_000, 40_000_000),
        clock_authority="synthetic reference clock", processing={"fixture": True})
    binding = dict(producer_identity={"method": "synthetic typed state serialization"},
        scene_identity={"fixture": True}, scene_from_source_world=np.eye(4).tolist(),
        assistance_receipt={"evaluation_external_wrenches_cleared": True, "evaluation_object_playback": False},
        hands={side: dict(body_name=side + "_root", command_joint_names={"driver": side + "_driver"},
                         physical_joint_names={"driver": side + "_driver"}) for side in ("left", "right")},
        object_part_names=["proximal", "distal", "prop"], object_parts=dict(
            proximal=dict(entity_name="object", body_name="object_root"),
            distal=dict(entity_name="object", body_name="object_lid"),
            prop=dict(entity_name="loose_prop", body_name="prop_root")), clips={"fixture": source})

    def setup(env, command):
        _declare_joint_coordinates(next(iter(command.native_objects.values())), ("slide", "hinge"))
        _declare_joint_coordinates(command.robot, ("slide", "hinge", "hinge"))
        command._bind_native_reference()
        _add_loose_object(command, env, [])
        # A type does not rename, convert, or reorder observed coordinates.
        next(iter(command.native_objects.values())).data.joint_pos[:] = torch.tensor([[.023, -.4]])
        def applied_interval():
            step = int(command.reference_phase["motion_steps"][0])
            return dict(entity_names=("loose_prop", "object", "object"),
                        part_body_names=("prop_root", "object_lid", "object_root"), num_substeps=4,
                        authority="post_solve_mjw_xfrc_applied_WORLD_body_COM_wrench",
                        peak_force_n=step * torch.tensor([[3., 2., 1.]]),
                        peak_torque_nm=step * torch.tensor([[.3, .2, .1]]),
                        measurement_support=torch.tensor([step > 0]))
        env.scene.applied_object_wrench_interval_facts = applied_interval

    manifest = record_reference_phase_fixture(tmp_path, binding, native=True, setup_scene=setup)
    recorded = load_flashdex_rollout(manifest, motion_id=0)
    metadata = recorded.object_joint_metadata
    assert metadata["object_root_body_names"] == {"object": "object_root", "loose_prop": "prop_root"}
    assert metadata["entity_names"] == ["object", "object"]
    assert metadata["joint_names"] == ["hinge_a", "hinge_b"]
    assert metadata["joint_types"] == ["slide", "hinge"]
    assert metadata["position_units"] == ["m", "rad"]
    assert metadata["velocity_units"] == ["m/s", "rad/s"]
    assert "observed simulator scalar state" in metadata["authority"]
    commands = recorded.installed_control_metadata["finger_joint_coordinates"]
    assert commands["joint_names"] == ["left_driver", "right_driver"]
    assert commands["joint_types"] == ["slide", "hinge"]
    assert commands["position_units"] == ["m", "rad"]
    assert commands["velocity_units"] == ["m/s", "rad/s"]
    np.testing.assert_array_equal(recorded.arrays["installed_finger_target_rad"],
                                  np.array([[0., 0.], [1., 1.], [1., 1.]], np.float32))
    assert recorded.installed_control_metadata["action_recipe"]["finger_effective_limits_rad"] == [[-1., 1.], [-1., 1.]]
    expected = torch.tensor([[.023, -.4]]).repeat(3, 1).numpy()
    np.testing.assert_array_equal(recorded.arrays["object_joint_pos"], expected)
    np.testing.assert_array_equal(recorded.arrays["object_joint_vel"],
                                  np.array([[0., 0.], [1.5, 1.5], [2.5, 2.5]], np.float32))
    assert recorded.arrays["object_joint_pos"].dtype == np.float32
    assert recorded.arrays["object_joint_position_support"].dtype == bool
    np.testing.assert_array_equal(recorded.arrays["recorded_motion_steps"], [0, 1, 2])
    assert recorded.arrays["native_object_part_pose"].shape == (3, 3, 7)
    assistance = json.loads(str(recorded.arrays["object_assist_metadata_json"]))
    assert assistance["entity_names"] == ["object", "object", "loose_prop"]
    assert assistance["body_names"] == ["object_root", "object_lid", "prop_root"]
    np.testing.assert_array_equal(recorded.arrays["object_assist_force_w"], np.zeros((3, 3, 3), np.float32))
    prefix = "applied_object_wrench_interval_"
    interval = json.loads(str(recorded.arrays[prefix + "metadata_json"]))
    assert interval["part_body_names"] == ["object_root", "object_lid", "prop_root"]
    assert interval["num_substeps"] == 4
    np.testing.assert_array_equal(recorded.arrays[prefix + "measurement_support"], [False, True, True])
    np.testing.assert_array_equal(recorded.arrays[prefix + "peak_force_n"], [[0, 0, 0], [1, 2, 3], [2, 4, 6]])
    np.testing.assert_allclose(recorded.arrays[prefix + "peak_torque_nm"], [[0, 0, 0], [.1, .2, .3], [.2, .4, .6]])
