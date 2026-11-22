import json
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from retarget.human_demo.sources import hot3d
from retarget.human_demo.sources.base import Source, overlapping_frames


def test_near_matches_bounded_kdtree_for_both_object_scales(tmp_path, monkeypatch):
    source = Source(tmp_path, None)
    mesh = trimesh.creation.box()
    monkeypatch.setattr(source, "load_mesh", lambda obj_id: mesh)
    vertices = np.asarray(mesh.vertices)
    offsets = np.array(
        [
            [0, 0, 0],
            [-0.009, 0, 0],
            [-0.02, 0, 0],
            [-0.019, -0.019, 0],
            [0, -0.021, 0],
            [0, 0, 0.015],
        ]
    )
    for scale in (1.0, 2.0):
        points = vertices[0] * scale + offsets
        tree = cKDTree(vertices * scale)
        expected = tree.query(points, distance_upper_bound=0.02)[0]
        np.testing.assert_array_equal(
            source.near("shared", points, 0.02, scale), expected
        )


def test_world_bounds_enclose_the_actual_local_distance_query(tmp_path, monkeypatch):
    source = Source(tmp_path, None)
    mesh = trimesh.creation.box()
    monkeypatch.setattr(source, "load_mesh", lambda obj_id: mesh)
    poses = np.repeat(np.eye(4)[None], repeats=2, axis=0)
    poses[:, :3, :3] = [[0.1, 0.1, 0], [0, 1, 0], [0, 0, 1]]
    poses[:, 0, 3] = [0.6, 1.0]
    local = np.asarray(mesh.vertices)[0] + [[-0.01, 0, 0], [-2, 0, 0]]
    world = (local[:, None] @ np.linalg.inv(poses[:, :3, :3])).squeeze(1) + poses[
        :, :3, 3
    ]
    selected = overlapping_frames(
        (world, world), source.mesh_world_bounds("shared", poses, 0.02)
    )
    np.testing.assert_array_equal(selected, [0])
    np.testing.assert_allclose(source.near("shared", local[selected], 0.02), [0.01])


def test_hot3d_segments_keep_contact_near_rotating_mesh_bounds(tmp_path, monkeypatch):
    segment = SimpleNamespace(
        tip_contact=0.02, link_contact=0.01, gap_frames=1, min_frames=2
    )
    tips = {side: {"thumb": 0} for side in ("left", "right")}
    source = hot3d.Source(tmp_path, SimpleNamespace(segment=segment), tip_vertex=tips)
    mesh = trimesh.creation.box(extents=[0.06, 0.22, 0.11])
    monkeypatch.setattr(source, "load_mesh", lambda obj_id: mesh)
    poses = np.tile(np.eye(4), (8, 1, 1))
    poses[:, :3, :3] = Rotation.from_euler(
        "zx", [[0.31 * i, 0.22] for i in range(8)]
    ).as_matrix()
    poses[:, 1, 3] = np.arange(8) * 0.1
    local = np.broadcast_to(np.asarray(mesh.vertices)[0] - [0.009, 0, 0], (8, 3)).copy()
    local[[0, 4, 5], 0] += 6.0
    world = (poses[:, :3, :3] @ local[..., None]).squeeze(-1) + poses[:, :3, 3]
    verts = np.broadcast_to(world[:, None, :], (8, 16, 3))
    weights = SimpleNamespace(numpy=lambda: np.eye(16))
    monkeypatch.setattr(hot3d, "_mano", lambda *_: SimpleNamespace(lbs_weights=weights))
    monkeypatch.setattr(hot3d, "_forward", lambda *_: (None, verts))
    hands = dict.fromkeys(tips)
    assert source._segments(hands, {"shared": poses}, ["shared"]) == [
        (1, 4, "B", "shared"),
        (6, 8, "B", "shared"),
    ]


def test_hot3d_withheld_recording_with_hand_file_is_not_imported(tmp_path):
    for name, has_gt in (("train", True), ("test", False)):
        recording = tmp_path / "dataset" / name
        recording.mkdir(parents=True)
        (recording / "mano_hand_pose_trajectory.jsonl").touch()
        (recording / "metadata.json").write_text(
            json.dumps({"have_hand_object_pose_gt": has_gt})
        )
    assert hot3d.Source(tmp_path, None).units() == ["train"]
