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
    Q/Qvel describe named physical scalar joints; FREE roots are separate
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
    parts = names("object_part_body_names")
    if (not parts or not reference["object_entity_name"]
            or reference["object_root_body_name"] not in parts):
        raise ValueError("native object reference must name its actual entity, FREE base and parts")
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
    return reference


def crop_native_reference(reference: dict, start: int) -> dict:
    """Slice only declared row fields; names, provenance and clock origin remain."""
    result = dict(reference)
    for field in _NATIVE_FRAME_ARRAYS:
        result[field] = reference[field][start:]
    for field in _NATIVE_ROOT_ARRAYS:
        result[field] = {side: value[start:] for side, value in reference[field].items()}
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
