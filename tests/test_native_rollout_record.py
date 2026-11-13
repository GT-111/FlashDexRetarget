"""Factual export additions exercise the actual author recorder without a sim."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation.callbacks.rollout_record import NativeRolloutRecord
from evaluation.callbacks.success_export import export_success


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
            body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]] * 3])))
    command = SimpleNamespace(robot=robot, motion_steps=torch.zeros(1, dtype=torch.int64),
        motion_ids=torch.zeros(1, dtype=torch.int64), _xfrc_curr_ctrl=None,
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
        action_manager=SimpleNamespace(action=torch.tensor([[77., 88.]])),
        command_manager=SimpleNamespace(get_term=lambda name: command))
    callback = NativeRolloutRecord(binding=binding)
    callback.on_start(env)
    command.motion_steps += 1
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
