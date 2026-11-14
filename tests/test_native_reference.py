"""Actual prepared-clip/pack/reader seams with explicit synthetic native facts."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException

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


def _mesh(dataset):
    mesh = dataset / "objects/synthetic-mesh/visual.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("v 0 0 0\nv .01 0 0\nv 0 .01 0\nv 0 0 .01\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")


def _pack(tmp_path, clip):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    dataset = tmp_path / "dataset"
    _mesh(dataset)
    C.save_clip(prepared / "synthetic-stage.npz", clip)
    build(prepared, dataset, 60., SimpleNamespace(vel_sigma=2., contact_threshold=.015))
    return dataset / "motion.pt"


def _prepared_native_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    prepared = dataset / "native_reference"
    prepared.mkdir(parents=True)
    _mesh(dataset)
    clip = _clip()
    clip["fps"] = np.float32(50)
    time_ns = 2**60 + np.arange(3, dtype=np.int64) * 20_000_000
    clip["native_reference"]["time_ns"] = time_ns
    clip["source_import_elapsed_ns"] = time_ns.copy()
    C.save_clip(prepared / "synthetic-stage.npz", clip)
    (dataset / "native-reference.json").write_text(json.dumps(dict(
        clip_directory="native_reference", clips=["synthetic-stage"], fps=50., control_period_ns=20_000_000)))
    return dataset, clip


def _pipeline_config(dataset, *, prepared=True):
    overrides = ["dataset=" + str(dataset), "workers=1", "steps.convex=false", "steps.usd=false", "steps.sdf=false"]
    if prepared:
        overrides.append("input=prepared_native")
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config/retarget"), version_base=None,
    ):
        return compose(config_name="kinematic_retargeting", overrides=overrides)


def test_prepared_native_entry_packs_selected_stems_and_admits_normal_library_command(tmp_path):
    from envs.commands.common.motion_library import MotionLibrary
    from envs.commands.motion_tracking.native import NativeMotionTrackingCommand
    from envs.scene.native import NativeSceneCfg
    from retarget import kinematic_retargeting as pipeline

    dataset, clip = _prepared_native_dataset(tmp_path)
    # A stale human-demo directory and an unselected carrier must not choose the route or population.
    (dataset / "human_demo").mkdir()
    (dataset / "human_demo/rejected-by-stock-route.npz").write_text("unselected human demo")
    (dataset / "native_reference/unselected.npz").write_text("unselected native carrier")
    cfg = _pipeline_config(dataset)
    assert cfg.steps.retarget  # Prepared input itself owns the bypass; no redundant flag is required.
    pipeline.main.__wrapped__(cfg)

    joint_names = ("left_a", "left_mimic", "right_b")
    library = MotionLibrary(str(dataset / "motion.pt"),
        SimpleNamespace(num_joints=len(joint_names), joint_names=joint_names), finger_names=C.FINGERS, device="cpu")
    native = clip["native_reference"]
    assert library.motion_files == [str(dataset / "motion.pt") + "#synthetic-stage"]
    assert library.support_disks.shape == (1, 0, 5)
    assert not (dataset / "report.csv").exists()
    assert not (dataset / "retargeted").exists()
    np.testing.assert_array_equal(library.native_time_ns, native["time_ns"])
    np.testing.assert_array_equal(library.native_joint_velocity_support, native["joint_velocity_support"][:, [2, 1, 0]])
    assert library.native_reference_provenance == [native["provenance"]]

    # Exercise the actual native command's reader admission, without constructing a physical model.
    layout = NativeSceneCfg.from_mapping(dict(
        scene_xml_path="synthetic-not-a-model.xml",
        hands=[dict(side=side, entity_name=side, root_body_name=body, root_joint_name=side + "_free",
            command_actuator_names=["command_" + side], root_control=dict(position_stiffness=0,
                position_damping=0, rotation_stiffness=0, rotation_damping=0, force_limit_n=None,
                torque_limit_nm=None)) for side, body in native["root_body_names"].items()],
        object=dict(entity_name=native["object_entity_name"], root_body_name=native["object_root_body_name"],
            root_joint_name="synthetic_object_free", parts=[dict(name=body, body_name=body, mesh_path="synthetic.obj")
                for body in native["object_part_body_names"]])))
    obj = SimpleNamespace(joint_names=("hinge_a", "hinge_b"), body_names=("base", "lid"))
    command = object.__new__(NativeMotionTrackingCommand)
    command.native_layout, command.motion_lib = layout, library
    command._env = SimpleNamespace(device="cpu", num_envs=1, step_dt=0.02,
        scene={native["object_entity_name"]: obj})
    command._finger_joint_ids = torch.tensor([0, 2])
    command.motion_ids, command.motion_steps = torch.tensor([0]), torch.tensor([1])
    command._bind_native_reference()
    torch.testing.assert_close(command.object_joint_reference()["position"], torch.tensor([[.15, .1]]))
    torch.testing.assert_close(command.command_reference()["position"], torch.tensor([[.5, .3]]))
    assert command.command_reference()["position_support"].tolist() == [[True, False]]
    assert not command.motion_lib.native_root_pose_support["left"][1]
    assert command.motion_lib.native_root_angular_velocity_support["left"][1]


def test_prepared_native_entry_rejects_stock_carrier_or_support_before_packing(tmp_path):
    from retarget import kinematic_retargeting as pipeline

    for invalid in ("missing_native", "stock_support"):
        dataset, clip = _prepared_native_dataset(tmp_path / invalid)
        if invalid == "missing_native":
            del clip["native_reference"]
            message = "requires native_reference"
        else:
            clip["support_disks"] = np.zeros((0, 5), np.float32)
            message = "support belongs to its authored scene"
        C.save_clip(dataset / "native_reference/synthetic-stage.npz", clip)
        with TestCase().assertRaisesRegex(InstantiationException, message):
            pipeline.main.__wrapped__(_pipeline_config(dataset))
        assert not (dataset / "motion.pt").exists()


def test_default_human_entry_retains_preprocess_retarget_and_report(tmp_path):
    from retarget import kinematic_retargeting as pipeline

    dataset = tmp_path / "dataset"
    demo = dataset / "human_demo"
    demo.mkdir(parents=True)
    path = demo / "human-stage.npz"
    C.save_clip(path, _clip())
    cfg = _pipeline_config(dataset, prepared=False)
    cfg.steps.pack = False
    rows, kept = [{"clip": path.stem, "reject": ""}], {path: None}
    with patch("retarget.human_demo.preprocess.preprocess_clips", return_value=(rows, kept)) as preprocess:
        with patch("retarget.robot_hand.hand_retarget.retarget_clips", return_value=[{"clip": path.stem, "source": "ik"}]) as ik:
            pipeline.main.__wrapped__(cfg)
    preprocess.assert_called_once_with(demo, dataset, cfg, 60., 1)
    ik.assert_called_once_with([(path, None)], dataset, dataset / "retargeted", cfg.robot, cfg.retarget, 60., 1)
    assert "source" in (dataset / "report.csv").read_text()


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
        test_prepared_native_entry_packs_selected_stems_and_admits_normal_library_command,
        test_prepared_native_entry_rejects_stock_carrier_or_support_before_packing,
        test_default_human_entry_retains_preprocess_retarget_and_report,
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
