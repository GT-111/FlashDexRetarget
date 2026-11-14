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
        library.source_lineage = [dict(source)]
        library.native_reference_provenance = [dict(
            authority="requested_reference_not_achieved_state", source=dict(source),
            scene_from_source_world=binding["scene_from_source_world"])]
    else:
        library.source_lineage = [{"capture_id": source["source_unit_id"],
            "selected_source_frame_index": source["selected_source_frame_index"],
            "scene_from_source_world": binding["scene_from_source_world"]}]
    library.source_query_ordinal = [np.asarray(source["prepared_source_ordinal"])]
    library.source_import_elapsed_ns = [np.asarray(source["prepared_elapsed_time_ns"], np.int64) + 1_000_000_000]
    env.scene.env_origins = torch.tensor([[1., 2., 3.]])
    env.close = lambda: None
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
            command.native_object.data.joint_vel[:] = row + .5
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
    source = {"source_unit_id": "S10001", "prepared_time_ns": [0, 58_333_333],
              "selected_source_frame_index": selected, "prepared_source_ordinal": ordinal.tolist(),
              "prepared_elapsed_time_ns": imported_time.tolist()}
    binding = {"hands": {
        "left": {"body_name": "left_wrist", "physical_joint_names": {"hinge": "l_joint"}},
        "right": {"body_name": "right_wrist", "physical_joint_names": {"hinge": "r_joint"}}},
        "object_part_names": ["base"], "scene_from_source_world": np.eye(4).tolist(),
        "object_parts": {"base": {"entity_name": "box", "body_name": "base"}},
        "clips": {"S10001__st00": source}}
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
        motion_lib=SimpleNamespace(motion_files=["file#S10001__st00"], _motion_num_frames=torch.tensor([2]),
            source_lineage=[{"capture_id": "S10001", "selected_source_frame_index": selected,
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
    np.testing.assert_array_equal(output["raw_action_valid"], (False, True))
    np.testing.assert_array_equal(output["elapsed_time_ns"], (0, 20_000_000))
    np.testing.assert_array_equal(output["native_hand_pose"][0, :, :3], ((5., 5., 5.), (2., 2., 2.)))
    np.testing.assert_array_equal(output["native_hand_pose"][1, :, :3], ((5.25, 5.25, 5.25), (2.25, 2.25, 2.25)))
    np.testing.assert_array_equal(output["joint_vel"], ((4., 8.), (4., 8.)))
    np.testing.assert_allclose(output["native_hand_twist_w"][0], ((4., 5., 6., .4, .5, .6), (1., 2., 3., .1, .2, .3)))
    np.testing.assert_array_equal(output["native_object_part_pose"][:, 0, :3], ((7., 7., 7.), (7., 7., 7.)))
    np.testing.assert_array_equal(output["actions"], ((77., 88.), (77., 88.)))
    assert np.max(output["native_hand_pose"][..., :3]) < 999.
    assert json.loads(str(output["native_binding_json"]))["clips"]["S10001__st00"] == source
    assert export_success(tmp_path, callback, [state], torch.tensor([True]), torch.tensor([0]),
        20_000_000, "obj SR@1.0", motion_names=["S10001__st00"]) == 1
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

    lineage = {"capture_id": "S10001", "selected_source_frame_index": [1, 2, 8]}
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
        "motion_filename": ["S10001__st00"], "motion_source_lineage": [lineage]}
    path = tmp_path / "motion.pt"
    torch.save(packed, path)
    loaded, names = module._load_packed_pt(str(path))
    assert names == [f"{path}#S10001__st00"]
    assert loaded[0]["source_lineage"] == lineage
    np.testing.assert_array_equal(loaded[0]["source_query_ordinal"], (1., 2.))


def test_frozen_attempt_npz_keeps_actual_native_schema_clocks_interval_and_first_success(tmp_path):
    from dexforge_adaptation.flashdex_retarget import flashdex_source_mapping

    from evaluation.callbacks.success_export import export_current_attempts

    origin = 2**54 + 31
    source = flashdex_source_mapping(source_id="arctic", source_revision="fixture",
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


def _prepared_native_recorder_fixture(tmp_path):
    """Real prepared-input/pack/library admission; only achieved state is synthetic."""
    from dexforge_adaptation.flashdex_retarget import flashdex_source_mapping
    from test_native_reference import _pipeline_config, _prepared_native_dataset

    from envs.commands.common.motion_library import MotionLibrary
    from retarget.human_demo import clips as C
    from retarget.kinematic_retargeting import PreparedNativeInput

    dataset, clip = _prepared_native_dataset(tmp_path)
    elapsed = np.arange(3, dtype=np.int64) * 20_000_000
    origin = 2**60 + 37
    source = flashdex_source_mapping(source_id="arctic", source_revision="synthetic-fixture",
        source_unit_id="episode-000174", source_episode_index=174, fold="train",
        source_frame_index=(0, 1, 2), source_time_ns=origin + elapsed,
        selected_source_frame_index=(0, 1, 2), prepared_source_ordinal=(0., 1., 2.),
        prepared_elapsed_time_ns=elapsed, clock_authority="synthetic exact source clock",
        processing={"fixture": "recorder software admission, not physical execution"})
    placement = np.eye(4)
    placement[:3, 3] = (1., -2., .3)
    native = clip["native_reference"]
    native["time_ns"] = elapsed.copy()
    native["provenance"] = dict(authority="requested_reference_not_achieved_state",
        source=dict(source), scene_from_source_world=placement.tolist())
    clip["source_lineage_json"] = np.asarray(json.dumps(source))
    clip["source_query_ordinal"] = np.asarray(source["prepared_source_ordinal"])
    clip["source_import_elapsed_ns"] = elapsed.copy()
    C.save_clip(dataset / "native_reference/synthetic-stage.npz", clip)
    cfg = _pipeline_config(dataset)
    prepared = PreparedNativeInput(dataset, cfg, workers=1, manifest="native-reference.json")
    prepared.pack()

    joint_names = ("left_a", "left_mimic", "right_b")
    robot = SimpleNamespace(num_joints=len(joint_names), joint_names=joint_names,
        body_names=("left_root", "right_root"), data=SimpleNamespace(
            joint_pos=torch.zeros(1, 3), joint_vel=torch.zeros(1, 3),
            root_link_pos_w=torch.zeros(1, 3), root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
            body_link_pos_w=torch.zeros(1, 2, 3),
            body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]] * 2]),
            body_link_lin_vel_w=torch.zeros(1, 2, 3), body_link_ang_vel_w=torch.zeros(1, 2, 3)))
    library = MotionLibrary(str(dataset / "motion.pt"), robot, finger_names=C.FINGERS, device="cpu")
    command = SimpleNamespace(robot=robot, motion_lib=library,
        motion_ids=torch.zeros(1, dtype=torch.int64), motion_steps=torch.ones(1, dtype=torch.int64),
        reference_phase={"motion_ids": torch.zeros(1, dtype=torch.int64),
                         "motion_steps": torch.zeros(1, dtype=torch.int64)},
        cfg=SimpleNamespace(object=SimpleNamespace(pin_objects=False)), _xfrc_curr_ctrl=None,
        sim_obj_trans_w=torch.zeros(1, 2, 3),
        sim_obj_quat_w=torch.tensor([[[1., 0., 0., 0.]] * 2]))

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene(articulated_object=SimpleNamespace(body_names=("lid", "base"),
        data=SimpleNamespace(body_link_pos_w=torch.zeros(1, 2, 3),
            body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]] * 2]))))
    env = SimpleNamespace(step_dt=.02, num_envs=1, scene=scene,
        action_manager=SimpleNamespace(action=torch.zeros(1, 3), active_terms=[]),
        command_manager=SimpleNamespace(get_term=lambda name: command))
    binding = dict(scene_from_source_world=placement.tolist(),
        hands={"left": dict(body_name="left_root", physical_joint_names={"driver": "left_a", "mimic": "left_mimic"}),
               "right": dict(body_name="right_root", physical_joint_names={"driver": "right_b"})},
        object_part_names=["lid", "base"], object_parts={name: dict(entity_name="articulated_object", body_name=name)
            for name in ("lid", "base")}, clips={"synthetic-stage": source})
    return env, binding


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
