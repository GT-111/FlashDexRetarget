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


def _with_source_point_support(clip):
    native = clip["native_reference"]
    native["source_point_names"], native["source_point_position_support"] = {}, {}
    base_names = C.MANO_JOINT_NAMES
    frames, points = C.num_frames(clip), len(base_names)
    positions = np.arange(frames * points * 3, dtype=np.float32).reshape(frames, points, 3) / 1000
    for side_index, side in enumerate(C.sides(clip)):
        clip_names = base_names[::-1]
        source_names = base_names[side_index + 1:] + base_names[:side_index + 1]
        clip[f"mano_{side}_joint_names"] = np.asarray(clip_names)
        clip[f"mano_{side}_joints"] = positions[:, [base_names.index(name) for name in clip_names]]
        support = np.ones((frames, points), bool)
        support[0, base_names.index("thumb_tip")] = False
        support[2, base_names.index("index_tip")] = False
        native["source_point_names"][side] = source_names
        native["source_point_position_support"][side] = support[:, [base_names.index(name) for name in source_names]]
    return clip


def _with_source_hand_part_intent(clip):
    native = clip["native_reference"]
    offsets = native["time_ns"] - native["time_ns"][0]
    origin = 2**60 + 37
    known = np.ones((2, 2, 2), bool)
    known[1, 0, 0] = False
    active = np.zeros_like(known)
    active[0, 0, 0] = active[0, 1, 1] = active[1, 0, 1] = True
    native["source_hand_part_intent"] = dict(
        hand_sides=("left", "right"), part_body_names=("base", "lid"),
        query_time_ns=origin + offsets,
        capture_time_ns=origin + np.array((0, int(offsets[-1]) - 1), np.int64),
        available_time_ns=origin + np.array((0, int(offsets[-1])), np.int64),
        valid_until_time_ns=origin + np.array((int(offsets[1]) // 2, int(offsets[-1]) + 1), np.int64),
        source_frame_index=np.array((100, 120), np.int64), canonical_frame_index=np.array((10, 12), np.int64),
        active=active, known=known, authority="synthetic source geometric intent, not measured contact",
        provenance={"resource": "synthetic source tape with a missing clock cell"},
    )
    return clip


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
    from simulator.native.entity import NamedArticulationView

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
    robot = SimpleNamespace(joint_names=joint_names)
    for entity in (obj, robot):
        entity._meta = SimpleNamespace(joint_names=entity.joint_names,
                                       joint_types=("hinge",) * len(entity.joint_names))
        entity.joint_coordinate_metadata = NamedArticulationView.joint_coordinate_metadata.__get__(entity)
    command = object.__new__(NativeMotionTrackingCommand)
    command.native_layout, command.motion_lib, command.robot = layout, library, robot
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


def test_source_point_masks_follow_crop_pack_and_actual_mano_name_order(tmp_path):
    original = _with_source_point_support(_clip())
    cropped = _crop(original, 1)
    path = _pack(tmp_path, cropped)
    library = _reader(path)
    native = original["native_reference"]
    assert library.native_source_point_position_support is not None
    np.testing.assert_array_equal(library.native_time_ns, native["time_ns"][1:])
    packed = torch.load(path, weights_only=False)
    for side in C.sides(original):
        source_names = native["source_point_names"][side]
        canonical = [source_names.index(name) for name in C.MANO_JOINT_NAMES]
        actual = library.native_source_point_position_support[side]
        np.testing.assert_array_equal(actual, native["source_point_position_support"][side][1:, canonical])
        assert actual.dtype == torch.bool
        assert library.mano_joint_names[side] == list(C.MANO_JOINT_NAMES)
        np.testing.assert_array_equal(library.mano_joint_pos[side], C.joints(original, side)[1:])
        assert actual[0].all() and not library.native_root_pose_support[side][0]
        assert not actual[1, C.MANO_JOINT_NAMES.index("index_tip")]
        assert library.native_root_pose_support[side][1]
        carried = packed["motion_native_reference"][0]
        assert carried["source_point_names"][side] == source_names
        np.testing.assert_array_equal(carried["source_point_position_support"][side],
                                      native["source_point_position_support"][side][1:])


def test_absent_source_point_modality_remains_unavailable_for_native_and_stock(tmp_path):
    for native in (True, False):
        clip = _clip()
        if not native:
            del clip["native_reference"]
        directory = tmp_path / ("native" if native else "stock")
        directory.mkdir()
        library = _reader(_pack(directory, clip))
        assert library.native_source_point_position_support is None


def test_source_point_boundary_rejects_unpaired_sides_names_and_nonboolean_rows(tmp_path):
    for invalid in ("unpaired", "side", "duplicate", "unknown", "shape", "numeric", "point_rows"):
        clip = _with_source_point_support(_clip())
        native = clip["native_reference"]
        names = native["source_point_names"]["left"]
        if invalid == "unpaired":
            del native["source_point_names"]
            message = "supplied together"
        elif invalid == "side":
            del native["source_point_position_support"]["left"]
            message = "different source hand sides"
        elif invalid == "duplicate":
            native["source_point_names"]["left"] = names[:-1] + (names[0],)
            message = "distinct actual MANO point names"
        elif invalid == "unknown":
            native["source_point_names"]["left"] = names[:-1] + ("not_a_source_point",)
            message = "distinct actual MANO point names"
        elif invalid == "shape":
            native["source_point_position_support"]["left"] = np.ones((2, len(names)), bool)
            message = "explicit boolean"
        elif invalid == "numeric":
            native["source_point_position_support"]["left"] = np.ones((3, len(names)), np.float32)
            message = "explicit boolean"
        else:
            clip["mano_left_joints"] = clip["mano_left_joints"][:, :-1]
            message = "actual MANO point rows"
        with TestCase().assertRaisesRegex(ValueError, message):
            C.native_reference(clip)


def test_native_reader_rejects_mixed_source_point_modality_presence(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    dataset = tmp_path / "dataset"
    _mesh(dataset)
    C.save_clip(prepared / "supported.npz", _with_source_point_support(_clip()))
    C.save_clip(prepared / "unavailable.npz", _clip())
    build(prepared, dataset, 60., SimpleNamespace(vel_sigma=2., contact_threshold=.015))
    with TestCase().assertRaisesRegex(ValueError, "mix available and unavailable source-point support"):
        _reader(dataset / "motion.pt")


def test_source_hand_part_intent_crop_pack_preserves_causal_cells_and_exact_clocks(tmp_path):
    original = _with_source_hand_part_intent(_clip())
    before = deepcopy(original["native_reference"]["source_hand_part_intent"])
    C.save_clip(tmp_path / "intent.npz", original)
    loaded = C.load_clip(tmp_path / "intent.npz")
    cropped = _crop(loaded, 1)
    intent = cropped["native_reference"]["source_hand_part_intent"]
    np.testing.assert_array_equal(intent["capture_time_ns"], before["capture_time_ns"])
    np.testing.assert_array_equal(intent["query_time_ns"], before["query_time_ns"][1:])
    packed = _pack(tmp_path, cropped)
    library = _reader(packed)
    actual = library.native_source_hand_part_intent
    assert library.hand_sides == ("right", "left")
    assert library.native_object_part_body_names == ("lid", "base")
    assert actual["known"].dtype == torch.bool
    assert not actual["known"][0].any()  # An expired cell never fills the gap.
    assert actual["known"][1].tolist() == [[True, True], [True, False]]
    assert actual["active"][1].tolist() == [[False, False], [True, False]]
    for field in ("query_time_ns", "capture_time_ns", "available_time_ns", "valid_until_time_ns",
                  "source_frame_index", "canonical_frame_index"):
        assert actual[field].dtype == torch.int64
    np.testing.assert_array_equal(actual["query_time_ns"], before["query_time_ns"][1:])
    np.testing.assert_array_equal(actual["source_frame_index"], [100, 120])
    np.testing.assert_array_equal(actual["available_time_ns"][1], before["available_time_ns"][1])
    assert library.native_source_hand_part_intent_provenance == [
        {key: before[key] for key in ("authority", "provenance")}]
    stored = torch.load(packed, weights_only=False)["motion_native_reference"][0]["source_hand_part_intent"]
    np.testing.assert_array_equal(stored["active"], before["active"])
    np.testing.assert_array_equal(original["native_reference"]["source_hand_part_intent"]["query_time_ns"],
                                  before["query_time_ns"])


def test_source_hand_part_intent_missing_clips_and_pairs_remain_unknown(tmp_path):
    supported = _with_source_hand_part_intent(_clip())
    native = supported["native_reference"]
    # A third real part and a partially declared source defeat fixed two-part assumptions.
    native["object_part_body_names"] += ("handle",)
    for field in ("object_part_state_w", "object_part_pose_support", "object_part_linear_velocity_support",
                  "object_part_angular_velocity_support"):
        native[field] = np.concatenate((native[field], native[field][:, :1]), axis=1)
    intent = native["source_hand_part_intent"]
    intent["hand_sides"], intent["part_body_names"] = ("left",), ("base",)
    for field in ("active", "known"):
        intent[field] = intent[field][:, :1, :1]
    absent = deepcopy(supported)
    del absent["native_reference"]["source_hand_part_intent"]
    prepared, dataset = tmp_path / "prepared", tmp_path / "dataset"
    prepared.mkdir()
    _mesh(dataset)
    C.save_clip(prepared / "a-supported.npz", supported)
    C.save_clip(prepared / "b-absent.npz", absent)
    build(prepared, dataset, 60., SimpleNamespace(vel_sigma=2., contact_threshold=.015))
    library = _reader(dataset / "motion.pt")
    actual = library.native_source_hand_part_intent
    assert actual["known"].shape == (6, 2, 3)
    assert actual["known"][0, 1, 1] and actual["active"][0, 1, 1]
    assert actual["known"].sum().item() == 1
    assert not actual["active"][3:].any() and not actual["known"][3:].any()
    assert (actual["query_time_ns"][3:] == -1).all()
    assert library.native_source_hand_part_intent_provenance[1] is None


def test_source_hand_part_intent_rejects_false_support_clocks_and_names(tmp_path):
    del tmp_path
    for invalid in ("unknown_active", "float_clock", "query_phase", "available_before_capture", "bad_part",
                    "numeric_mask", "empty_source", "wrong_rows", "duplicate_side"):
        clip = _with_source_hand_part_intent(_clip())
        intent = clip["native_reference"]["source_hand_part_intent"]
        if invalid == "unknown_active":
            intent["active"][1, 0, 0] = True
        elif invalid == "float_clock":
            intent["capture_time_ns"] = intent["capture_time_ns"].astype(np.float64)
        elif invalid == "query_phase":
            intent["query_time_ns"][1] += 1
        elif invalid == "available_before_capture":
            intent["available_time_ns"][1] = intent["capture_time_ns"][1] - 1
        elif invalid == "bad_part":
            intent["part_body_names"] = ("base", "not_in_scene")
        elif invalid == "numeric_mask":
            intent["known"] = intent["known"].astype(np.float32)
        elif invalid == "empty_source":
            intent["capture_time_ns"] = np.empty(0, np.int64)
        elif invalid == "wrong_rows":
            intent["source_frame_index"] = intent["source_frame_index"][:1]
        else:
            intent["hand_sides"] = ("left", "left")
        with TestCase().assertRaisesRegex(ValueError, "source intent|unknown source intent"):
            C.native_reference(clip)


def test_source_hand_part_intent_decision_and_consumed_packet_are_independent(tmp_path):
    from envs.commands.motion_tracking.native import NativeMotionTrackingCommand
    from envs.observations.native import NativeMotionTrackingObs

    library = _reader(_pack(tmp_path, _with_source_hand_part_intent(_clip())))
    command = object.__new__(NativeMotionTrackingCommand)
    command.motion_lib, command._side_list = library, ["left", "right"]
    command.motion_ids, command.motion_steps = torch.tensor([0, 0]), torch.tensor([2, 2])
    command.reference_phase = dict(motion_ids=torch.tensor([0, 0]), motion_steps=torch.tensor([1, 1]))
    installed = dict(installed_valid=torch.tensor([True, False]), reference_motion_ids=torch.tensor([0, 0]),
                     reference_motion_steps=torch.tensor([0, 0]))
    env = command._env = SimpleNamespace(
        action_manager=SimpleNamespace(_terms={"native": SimpleNamespace(installed_control=installed)}),
        command_manager=SimpleNamespace(get_term=lambda name: command))
    decision = command.source_hand_part_intent(command._motion_flat_ids)
    consumed = command.source_hand_part_intent_facts()
    assert decision["active"][:, 0, 0].all()  # Declared left/lid at the current decision.
    assert consumed["reference_motion_steps"].tolist() == [0, 1]
    assert consumed["execution_valid"].tolist() == [True, False]
    assert consumed["active"][0, 0, 1] and not consumed["known"][1].any()
    obs = NativeMotionTrackingObs.native_source_hand_part_intent(env, "motion")
    assert obs.shape == (2, 8)
    torch.testing.assert_close(obs[:, :4], decision["active"].flatten(1).float())
    torch.testing.assert_close(obs[:, 4:], decision["known"].flatten(1).float())
    library.native_source_hand_part_intent = None
    assert command.source_hand_part_intent_facts() is None
    with TestCase().assertRaisesRegex(ValueError, "explicit source carrier"):
        NativeMotionTrackingObs.native_source_hand_part_intent(env, "motion")


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
        test_source_point_masks_follow_crop_pack_and_actual_mano_name_order,
        test_absent_source_point_modality_remains_unavailable_for_native_and_stock,
        test_source_point_boundary_rejects_unpaired_sides_names_and_nonboolean_rows,
        test_native_reader_rejects_mixed_source_point_modality_presence,
        test_source_hand_part_intent_crop_pack_preserves_causal_cells_and_exact_clocks,
        test_source_hand_part_intent_missing_clips_and_pairs_remain_unknown,
        test_source_hand_part_intent_rejects_false_support_clocks_and_names,
        test_source_hand_part_intent_decision_and_consumed_packet_are_independent,
    )
    selected = set(sys.argv[1:]) if len(sys.argv) > 1 else {case.__name__ for case in cases}
    for case in cases:
        if case.__name__ not in selected:
            continue
        with TemporaryDirectory() as directory:
            case(Path(directory))
        print(f"PASS {case.__name__}", flush=True)
    print(f"{len(selected)} passed; no model, FK, physics or Isaac application", flush=True)
