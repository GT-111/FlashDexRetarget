"""Actual prepared-clip/pack/reader seams with explicit synthetic native facts."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import torch

from retarget.build_motion_pt import build
from retarget.human_demo import clips as C
from retarget.robot_hand.hand_retarget import _crop


def _clip():
    frames = 3
    time_ns = 2**60 + np.array((0, 16_666_667, 33_333_333), np.int64)
    qpos = np.arange(9, dtype=np.float32).reshape(3, 3) / 10
    velocity_support = np.ones((frames, 3), bool)
    velocity_support[1, 1] = False
    qvel = np.full((frames, 3), 7., np.float32)
    qvel[~velocity_support] = 0.
    state = np.zeros((frames, 13), np.float32)
    state[:, 3] = 1.
    state[:, 10] = 2.
    part_state = np.zeros((frames, 2, 13), np.float32)
    part_state[:, :, 0] = (20., 10.)  # The declared lid/base order is nonlexical.
    part_state[:, :, 3] = 1.
    native = dict(
        time_ns=time_ns,
        clock_authority="synthetic prepared integer query clock; physical capture/availability unknown",
        provenance={"test_fixture": "synthetic software facts; no physical producer qualification"},
        joint_names=("right_b", "left_mimic", "left_a"),
        joint_pos=qpos, joint_vel=qvel,
        joint_position_support=np.ones((frames, 3), bool),
        joint_velocity_support=velocity_support,
        root_body_names={"left": "left_root", "right": "right_root"},
        root_state_w={side: state.copy() for side in ("left", "right")},
        root_pose_support={side: np.array((True, False, True)) for side in ("left", "right")},
        root_linear_velocity_support={side: np.zeros(frames, bool) for side in ("left", "right")},
        root_angular_velocity_support={side: np.ones(frames, bool) for side in ("left", "right")},
        object_entity_name="articulated_object", object_root_body_name="base",
        object_part_body_names=("lid", "base"), object_part_state_w=part_state,
        object_part_pose_support=np.array(((True, True), (False, True), (True, True))),
        object_part_linear_velocity_support=np.zeros((frames, 2), bool),
        object_part_angular_velocity_support=np.zeros((frames, 2), bool),
        object_joint_names=("hinge_b", "hinge_a"),
        object_joint_pos=np.arange(6, dtype=np.float32).reshape(frames, 2) / 20,
        object_joint_vel=np.zeros((frames, 2), np.float32),
        object_joint_position_support=np.ones((frames, 2), bool),
        object_joint_velocity_support=np.zeros((frames, 2), bool),
    )
    native["joint_position_support"][1, 0] = False  # Carried position remains finite.
    clip = dict(fps=np.float32(60), qpos=np.full((frames, 6), 999., np.float32),
        native_reference=native, source_query_ordinal=np.array((1., 3., 5.)),
        source_import_elapsed_ns=time_ns.copy(), source_lineage_json=np.asarray(json.dumps({
            "capture_id": "synthetic-capture", "selected_source_frame_index": [1, 2, 8, 10, 11, 15]})))
    for side in ("left", "right"):
        clip[f"mano_{side}_wrist_pos"] = np.zeros((frames, 3), np.float32)
        clip[f"mano_{side}_wrist_rot"] = np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1))
        clip[f"mano_{side}_joints"] = np.zeros((frames, 20, 3), np.float32)
        clip[f"obj_{side}_pose"] = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
        clip[f"obj_{side}_id"] = np.asarray("synthetic-mesh")
    return clip


def _reader(path, *, joint_names=("left_a", "left_mimic", "right_b")):
    # Import the actual numeric reader without importing any Isaac app/backend.
    source = Path(__file__).parents[1] / "src/envs/commands/common/motion_library.py"
    spec = importlib.util.spec_from_file_location("native_reference_motion_library", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot = SimpleNamespace(num_joints=len(joint_names), joint_names=joint_names)
    return module.MotionLibrary(str(path), robot, finger_names=C.FINGERS, device="cpu")


def _pack(tmp_path, clip):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    dataset = tmp_path / "dataset"
    mesh = dataset / "objects/synthetic-mesh/visual.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("v 0 0 0\nv .01 0 0\nv 0 .01 0\nv 0 0 .01\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    C.save_clip(prepared / "synthetic-stage.npz", clip)
    build(prepared, dataset, 60., SimpleNamespace(vel_sigma=2., contact_threshold=.015))
    return dataset / "motion.pt"


def test_actual_native_pack_and_reader_keep_names_independent_support_and_exact_clock(tmp_path):
    clip = _clip()
    path = _pack(tmp_path, clip)
    library = _reader(path)
    native = clip["native_reference"]
    columns = [2, 1, 0]
    np.testing.assert_array_equal(library.robot_joint_pos, native["joint_pos"][:, columns])
    np.testing.assert_array_equal(library.robot_joint_vel, native["joint_vel"][:, columns])
    np.testing.assert_array_equal(library.native_joint_position_support, native["joint_position_support"][:, columns])
    np.testing.assert_array_equal(library.native_joint_velocity_support, native["joint_velocity_support"][:, columns])
    np.testing.assert_array_equal(library.native_time_ns, native["time_ns"])
    assert library.native_time_ns.dtype == torch.int64
    assert library.native_joint_names == ("left_a", "left_mimic", "right_b")
    assert library.native_root_body_names == native["root_body_names"]
    for side in ("left", "right"):
        np.testing.assert_array_equal(library.native_root_state_w[side], native["root_state_w"][side])
        np.testing.assert_array_equal(library.native_root_pose_support[side], native["root_pose_support"][side])
        np.testing.assert_array_equal(library.native_root_angular_velocity_support[side], (True, True, True))
        np.testing.assert_array_equal(library.native_root_linear_velocity_support[side], (False, False, False))
    assert library.native_object_part_body_names == ("lid", "base")
    assert library.native_object_joint_names == ("hinge_b", "hinge_a")
    np.testing.assert_array_equal(library.native_object_part_state_w, native["object_part_state_w"])
    np.testing.assert_array_equal(library.native_object_part_pose_support, native["object_part_pose_support"])
    np.testing.assert_array_equal(library.native_object_joint_velocity_support, np.zeros((3, 2), bool))
    packed = torch.load(path, weights_only=False)
    assert len(packed["motion_native_reference"]) == 1
    assert packed["motion_native_reference"][0]["provenance"] == native["provenance"]


def test_actual_crop_pack_and_reopen_keep_native_rows_static_names_and_source_inventory(tmp_path):
    original = _clip()
    cropped = _crop(original, 1)
    path = _pack(tmp_path, cropped)
    library = _reader(path)
    native = original["native_reference"]
    np.testing.assert_array_equal(library.native_time_ns, native["time_ns"][1:])
    assert int(library.native_time_ns[0]) > 2**53
    np.testing.assert_array_equal(library.robot_joint_vel, native["joint_vel"][1:, [2, 1, 0]])
    assert library.native_joint_names == ("left_a", "left_mimic", "right_b")
    assert library.native_reference_provenance == [native["provenance"]]
    assert library.source_lineage[0]["selected_source_frame_index"] == [1, 2, 8, 10, 11, 15]
    np.testing.assert_array_equal(library.source_query_ordinal[0], (3., 5.))
    np.testing.assert_array_equal(library.native_root_pose_support["left"], (False, True))
    np.testing.assert_array_equal(library.native_object_part_pose_support, ((False, True), (True, True)))
    assert library.native_object_part_body_names == ("lid", "base")


def test_native_boundary_rejects_clock_row_support_and_actual_joint_identity_mismatch(tmp_path):
    clip = _clip()
    clock_mismatch = deepcopy(clip)
    clock_mismatch["native_reference"]["time_ns"][1] += 1
    with TestCase().assertRaisesRegex(ValueError, "clock differs"):
        C.native_reference(clock_mismatch)
    rows_mismatch = deepcopy(clip)
    rows_mismatch["native_reference"]["joint_position_support"] = np.ones((2, 3), bool)
    with TestCase().assertRaisesRegex(ValueError, "joint_position_support"):
        C.native_reference(rows_mismatch)
    unknown_rate = deepcopy(clip)
    unknown_rate["native_reference"]["joint_velocity_support"][0, 0] = False
    with TestCase().assertRaisesRegex(ValueError, "unknown joint velocities"):
        C.native_reference(unknown_rate)
    path = _pack(tmp_path, clip)
    with TestCase().assertRaisesRegex(ValueError, "actual named robot scalar coordinates"):
        _reader(path, joint_names=("left_a", "left_mimic", "unrelated_joint"))


if __name__ == "__main__":
    # Match the existing native-contract harness; the author env needs no pytest install.
    import sys
    from tempfile import TemporaryDirectory

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = (
        test_actual_native_pack_and_reader_keep_names_independent_support_and_exact_clock,
        test_actual_crop_pack_and_reopen_keep_native_rows_static_names_and_source_inventory,
        test_native_boundary_rejects_clock_row_support_and_actual_joint_identity_mismatch,
    )
    selected = set(sys.argv[1:]) if len(sys.argv) > 1 else {case.__name__ for case in cases}
    for case in cases:
        if case.__name__ not in selected:
            continue
        with TemporaryDirectory() as directory:
            case(Path(directory))
        print(f"PASS {case.__name__}", flush=True)
    print(f"{len(selected)} passed; no model, FK, physics or Isaac application", flush=True)
