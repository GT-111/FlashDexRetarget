from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MANO_JOINT_NAMES = tuple(
    sorted(f"{f}_{p}" for f in FINGERS for p in ("proximal", "intermediate", "distal", "tip"))
)
SIDES = ("right", "left")

_NATIVE_FRAME_ARRAYS = (
    "time_ns", "joint_pos", "joint_vel", "joint_position_support", "joint_velocity_support",
    "object_part_state_w", "object_part_pose_support", "object_part_linear_velocity_support",
    "object_part_angular_velocity_support", "object_joint_pos", "object_joint_vel",
    "object_joint_position_support", "object_joint_velocity_support",
)
_NATIVE_ROOT_ARRAYS = (
    "root_state_w", "root_pose_support", "root_linear_velocity_support", "root_angular_velocity_support",
)
_NATIVE_COMMAND_ARRAYS = ("command_pos", "command_position_support")


def mano_reference(joints: np.ndarray, verts: np.ndarray, global_rot: np.ndarray, side: str,
                   tip_vertex: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MANO layer output (joints (T, 16, 3), verts (T, 778, 3), global_rot (T, 3, 3)) -> clip hand arrays;
    tip_vertex = fingertip vertex per finger (cfg.mano_tip_vertex[side])."""
    first_joint = {"index": 1, "middle": 4, "pinky": 7, "ring": 10, "thumb": 13}
    wrist = joints[:, 0]
    named = {}
    for finger, j in first_joint.items():
        for k, part in enumerate(("proximal", "intermediate", "distal")):
            named[f"{finger}_{part}"] = joints[:, j + k]
        named[f"{finger}_tip"] = verts[:, tip_vertex[finger]]
    joints20 = np.stack([named[n] for n in MANO_JOINT_NAMES], axis=1)
    rebase = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
    if side == "left":
        rebase = rebase @ np.diag([-1.0, -1.0, 1.0])
    rot = np.asarray(global_rot, dtype=np.float64) @ rebase
    return wrist.astype(np.float32), rot.astype(np.float32), joints20.astype(np.float32)


def load_clip(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        clip = {k: z[k] for k in z.files}
    if "native_reference" in clip:
        # These are trusted author clips, as for the existing object-valued NPZ fields.
        clip["native_reference"] = clip["native_reference"].item()
    return clip


def native_reference(clip: dict) -> dict | None:
    """Read explicit prepared native facts; never derive state or support from MANO.

    The nested carrier is preserved in the author's trusted NPZ/PT formats.
    Q/Qvel describe named physical scalar joints. Optional named logical
    command targets are authored independently of Q. FREE roots are separate
    p/WXYZ/world-link-velocity/world-omega states. Unknown rates are zero with
    independent false support. ``time_ns`` is the actual prepared query clock,
    retaining its original phase after crop, rather than a capture-time claim.
    """
    value = clip.get("native_reference")
    if value is None:
        return None
    reference = dict(value.item() if isinstance(value, np.ndarray) else value)
    frames = num_frames(clip)

    def names(field):
        result = tuple(str(name) for name in reference[field])
        if len(set(result)) != len(result):
            raise ValueError(f"native reference {field} must name distinct physical coordinates")
        reference[field] = result
        return result

    def array(field, shape, *, support=False, parent=None):
        owner = reference if parent is None else parent
        result = np.asarray(owner[field])
        if result.shape != shape or (support and result.dtype.kind != "b"):
            raise ValueError(f"native reference {field} must have shape {shape} and explicit boolean support")
        if not support and not np.isfinite(result).all():
            raise ValueError(f"native reference {field} must retain finite carried values")
        owner[field] = result
        return result

    clock = array("time_ns", (frames,))
    if clock.dtype.kind not in "iu" or clock[0] < 0 or np.any(np.diff(clock) <= 0):
        raise ValueError("native reference time_ns must retain exact chronological integer query times")
    reference["time_ns"] = clock.astype(np.int64, copy=False)
    if not reference["clock_authority"] or not reference["provenance"]:
        raise ValueError("native reference requires explicit prepared-clock and state provenance")
    if "source_import_elapsed_ns" in clip and not np.array_equal(clock, clip["source_import_elapsed_ns"]):
        raise ValueError("native reference clock differs from the actual prepared source rows")
    joint_names = names("joint_names")
    if not joint_names:
        raise ValueError("native hand reference must name its physical scalar joints")
    array("joint_pos", (frames, len(joint_names)))
    velocity = array("joint_vel", (frames, len(joint_names)))
    array("joint_position_support", velocity.shape, support=True)
    velocity_support = array("joint_velocity_support", velocity.shape, support=True)
    if np.any(velocity[~velocity_support] != 0):
        raise ValueError("native unknown joint velocities require zero values and false support")
    command_fields = ("command_names", *_NATIVE_COMMAND_ARRAYS)
    if any(field in reference for field in command_fields):
        if not all(field in reference for field in command_fields) or not names("command_names"):
            raise ValueError("native logical commands require named positions and boolean support")
        shape = (frames, len(reference["command_names"]))
        array("command_pos", shape)
        array("command_position_support", shape, support=True)
    roots = dict(reference["root_body_names"])
    if set(roots) != set(sides(clip)) or len(set(roots.values())) != len(roots):
        raise ValueError("native roots must name the distinct actual bodies for the available hand sides")
    reference["root_body_names"] = roots
    for field in _NATIVE_ROOT_ARRAYS:
        reference[field] = dict(reference[field])
        if set(reference[field]) != set(roots):
            raise ValueError(f"native reference {field} has different hand sides")
    for side in roots:
        state = {field: reference[field][side] for field in _NATIVE_ROOT_ARRAYS}
        root = array("root_state_w", (frames, 13), parent=state)
        array("root_pose_support", (frames,), support=True, parent=state)
        linear = array("root_linear_velocity_support", (frames,), support=True, parent=state)
        angular = array("root_angular_velocity_support", (frames,), support=True, parent=state)
        if np.any(root[~linear, 7:10] != 0) or np.any(root[~angular, 10:13] != 0):
            raise ValueError("native unknown root rates require zero values and independent false support")
        for field in _NATIVE_ROOT_ARRAYS:
            reference[field][side] = state[field]
    source_fields = ("source_point_names", "source_point_position_support")
    if any(field in reference for field in source_fields):
        if not all(field in reference for field in source_fields):
            raise ValueError("native source-point names and position support must be supplied together")
        for field in source_fields:
            reference[field] = dict(reference[field])
            if set(reference[field]) != set(roots):
                raise ValueError(f"native reference {field} has different source hand sides")
        for side in roots:
            point_names = tuple(str(name) for name in reference["source_point_names"][side])
            clip_names = tuple(str(name) for name in clip.get(f"mano_{side}_joint_names", MANO_JOINT_NAMES))
            if (len(set(point_names)) != len(point_names) or len(set(clip_names)) != len(clip_names)
                    or set(point_names) != set(clip_names)):
                raise ValueError("native source_point_names must match the distinct actual MANO point names")
            if np.asarray(clip[f"mano_{side}_joints"]).shape != (frames, len(clip_names), 3):
                raise ValueError("native source point names differ from the actual MANO point rows")
            reference["source_point_names"][side] = point_names
            support = np.asarray(reference["source_point_position_support"][side])
            if support.shape != (frames, len(point_names)) or support.dtype.kind != "b":
                raise ValueError("native source_point_position_support requires explicit boolean [frames, points]")
            reference["source_point_position_support"][side] = support
    parts = names("object_part_body_names")
    object_roots = dict(reference["object_root_body_names"])
    if (not object_roots or any(not entity for entity in object_roots)
            or len(set(object_roots.values())) != len(object_roots)
            or not set(object_roots.values()) <= set(parts)):
        raise ValueError("native object reference must name its distinct actual entities, FREE bases and parts")
    reference["object_root_body_names"] = object_roots
    part_state = array("object_part_state_w", (frames, len(parts), 13))
    array("object_part_pose_support", (frames, len(parts)), support=True)
    linear = array("object_part_linear_velocity_support", (frames, len(parts)), support=True)
    angular = array("object_part_angular_velocity_support", (frames, len(parts)), support=True)
    if np.any(part_state[..., 7:10][~linear] != 0) or np.any(part_state[..., 10:13][~angular] != 0):
        raise ValueError("native unknown part rates require zero values and independent false support")
    object_joints = names("object_joint_names")
    array("object_joint_pos", (frames, len(object_joints)))
    velocity = array("object_joint_vel", (frames, len(object_joints)))
    array("object_joint_position_support", velocity.shape, support=True)
    velocity_support = array("object_joint_velocity_support", velocity.shape, support=True)
    if np.any(velocity[~velocity_support] != 0):
        raise ValueError("native unknown passive-joint rates require zero values and false support")
    if "source_hand_part_intent" in reference:
        reference["source_hand_part_intent"] = source_hand_part_intent(reference)
    return reference


def source_hand_part_intent(reference: dict) -> dict:
    """Validate source intent on its original clock, independently of native queries.

    This is a source's declared geometric/functional intent, never achieved
    contact or force. Cropping changes query rows, not the original source tape.
    """
    value = dict(reference["source_hand_part_intent"])
    for field, actual in (("hand_sides", reference["root_body_names"]),
                          ("part_body_names", reference["object_part_body_names"])):
        names = tuple(value[field])
        if not names or len(set(names)) != len(names) or not set(names) <= set(actual):
            raise ValueError(f"source intent {field} must name distinct admitted hands/parts")
        value[field] = names
    if not value["authority"] or not value["provenance"]:
        raise ValueError("source intent requires its own authority and provenance")
    for field in ("query_time_ns", "capture_time_ns", "available_time_ns", "valid_until_time_ns",
                  "source_frame_index", "canonical_frame_index"):
        rows = np.asarray(value[field])
        if rows.ndim != 1 or not rows.size or rows.dtype != np.int64 or np.any(rows < 0):
            raise ValueError("source intent clocks/addresses require nonnegative int64 vectors")
        value[field] = rows
    query, capture = value["query_time_ns"], value["capture_time_ns"]
    if (query.shape != reference["time_ns"].shape
            or not np.array_equal(query - query[0], reference["time_ns"] - reference["time_ns"][0])):
        raise ValueError("source intent queries must retain the native query grid and original source phase")
    if (any(value[field].shape != capture.shape for field in (
            "available_time_ns", "valid_until_time_ns", "source_frame_index", "canonical_frame_index"))
            or np.any(np.diff(capture) <= 0) or np.any(np.diff(value["available_time_ns"]) < 0)
            or np.any(value["available_time_ns"] < capture)
            or np.any(value["valid_until_time_ns"] <= capture)
            or np.any(np.diff(value["source_frame_index"]) <= 0)
            or np.any(np.diff(value["canonical_frame_index"]) <= 0)):
        raise ValueError("source intent clocks/addresses must retain chronological source cells")
    shape = (len(capture), len(value["hand_sides"]), len(value["part_body_names"]))
    for field in ("active", "known"):
        rows = np.asarray(value[field])
        if rows.shape != shape or rows.dtype.kind != "b":
            raise ValueError("source intent requires boolean [source row, actual hand, actual part] masks")
        value[field] = rows
    if np.any(value["active"] & ~value["known"]):
        raise ValueError("unknown source intent cannot be active")
    return value


def crop_native_reference(reference: dict, start: int) -> dict:
    """Slice only declared row fields; names, provenance and clock origin remain."""
    result = dict(reference)
    for field in _NATIVE_FRAME_ARRAYS:
        result[field] = reference[field][start:]
    if "command_names" in reference:
        for field in _NATIVE_COMMAND_ARRAYS:
            result[field] = reference[field][start:]
    for field in _NATIVE_ROOT_ARRAYS:
        result[field] = {side: value[start:] for side, value in reference[field].items()}
    if "source_point_position_support" in reference:
        result["source_point_position_support"] = {
            side: value[start:] for side, value in reference["source_point_position_support"].items()}
    if "source_hand_part_intent" in reference:
        intent = dict(reference["source_hand_part_intent"])
        intent["query_time_ns"] = intent["query_time_ns"][start:]
        result["source_hand_part_intent"] = intent
    return result


def save_clip(path: str | Path, clip: dict[str, np.ndarray]) -> None:
    """Write via a temp file so an interrupted run never leaves a truncated clip."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **clip)
    os.replace(tmp, path)


def list_clips(clip_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(clip_dir).glob("*.npz") if not p.name.endswith(".tmp.npz"))


def sides(clip: dict) -> tuple[str, ...]:
    return tuple(s for s in SIDES if f"mano_{s}_wrist_pos" in clip)


def num_frames(clip: dict) -> int:
    return len(clip[f"mano_{sides(clip)[0]}_wrist_pos"])


def joints(clip: dict, side: str) -> np.ndarray:
    """(T, 20, 3) MANO joints in MANO_JOINT_NAMES order."""
    names = MANO_JOINT_NAMES
    if f"mano_{side}_joint_names" in clip:
        names = [str(n) for n in clip[f"mano_{side}_joint_names"]]
    j = clip[f"mano_{side}_joints"]
    return j[:, [names.index(n) for n in MANO_JOINT_NAMES]].astype(np.float64)


def tips(clip: dict, side: str) -> np.ndarray:
    """(T, 5, 3) MANO fingertips, thumb .. pinky."""
    cols = [MANO_JOINT_NAMES.index(f"{f}_tip") for f in FINGERS]
    return joints(clip, side)[:, cols]


def obj_id(clip: dict, side: str) -> str:
    return str(clip[f"obj_{side}_id"])


def obj_scale(clip: dict, side: str) -> float:
    return float(clip[f"obj_{side}_scale"]) if f"obj_{side}_scale" in clip else 1.0


def obj_pose(clip: dict, side: str) -> np.ndarray:
    return clip[f"obj_{side}_pose"].astype(np.float64)


def shared_object(clip: dict) -> bool:
    """Both hands on one object: same id and the same pose track."""
    if sides(clip) != SIDES or obj_id(clip, "right") != obj_id(clip, "left"):
        return False
    return bool(np.allclose(obj_pose(clip, "right"), obj_pose(clip, "left"), atol=1e-6))


def objects(clip: dict) -> list[str]:
    """The clip's distinct object bodies, as hand sides (a shared object counts once)."""
    return ["right"] if shared_object(clip) else list(sides(clip))


class ObjectMeshes:
    """visual.obj (scaled) per (object, scale): vertices, KD-tree, convex hull and centre of mass."""

    def __init__(self, pool: str | Path) -> None:
        self.pool = Path(pool)
        self._cache: dict[tuple[str, float], dict] = {}

    def get(self, oid: str, scale: float = 1.0) -> dict:
        key = (oid, float(scale))
        if key not in self._cache:
            path = self.pool / "objects" / oid / "visual.obj"
            mesh = trimesh.load(path, force="mesh", process=False, skip_materials=True)
            mesh.apply_scale(scale)
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            com = mesh.center_mass if mesh.is_watertight else verts.mean(0)
            self._cache[key] = {
                "verts": verts,
                "tree": cKDTree(verts),
                "hull": np.asarray(mesh.convex_hull.vertices, dtype=np.float64),
                "com": np.asarray(com, dtype=np.float64),
            }
        return self._cache[key]


def clip_fps(paths) -> float:
    """The frame rate the clips share; every clip stores its own fps."""
    rates = set()
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            if "fps" not in z.files:
                raise ValueError(f"{p.name} has no fps")
            rates.add(float(z["fps"]))
    if len(rates) != 1:
        raise ValueError(f"the clips need one fps, got {sorted(rates)}")
    return rates.pop()


def object_ids(paths) -> list[str]:
    ids = set()
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            ids.update(str(z[f"obj_{s}_id"]) for s in SIDES if f"obj_{s}_id" in z.files)
    return sorted(ids)
