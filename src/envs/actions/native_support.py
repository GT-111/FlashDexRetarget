"""Clearance over fixed support faces in the compiled collision geometry."""

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul
from scipy.spatial.transform import Rotation


def _collider_corners(model, geom):
    kind = mujoco.mjtGeom(int(model.geom_type[geom]))
    size = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        low, high = vertices.min(0), vertices.max(0)
    elif kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        low, high = -size, size
    elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
                  mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius = size[0]
        height = {mujoco.mjtGeom.mjGEOM_SPHERE: radius,
                  mujoco.mjtGeom.mjGEOM_CAPSULE: size[1] + radius,
                  mujoco.mjtGeom.mjGEOM_CYLINDER: size[1]}[kind]
        high = np.array([radius, radius, height])
        low = -high
    else:
        return None
    corners = np.where(np.array(list(np.ndindex((2, 2, 2)))), high, low)
    rotation = Rotation.from_quat(model.geom_quat[geom], scalar_first=True)
    return rotation.apply(corners) + model.geom_pos[geom]


def bind_idle_support_faces(model, hand_body_names, body_names, device):
    """Bind each collidable fixed plane or horizontal box face and its footprint."""
    gravity = model.opt.gravity
    if not np.linalg.norm(gravity):
        return ()
    up = -gravity / np.linalg.norm(gravity)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    faces, frames, extents, planes = [], [], [], []
    for geom in np.flatnonzero(model.body_weldid[model.geom_bodyid] == 0):
        kind = mujoco.mjtGeom(int(model.geom_type[geom]))
        frame = data.geom_xmat[geom].reshape(3, 3)
        alignment = frame.T @ up
        if kind == mujoco.mjtGeom.mjGEOM_PLANE and alignment[2] > 1. - 1e-6:
            extent = [np.inf, np.inf, 0.]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX and np.max(abs(alignment)) > 1. - 1e-6:
            vertical = int(np.argmax(abs(alignment)))
            order = [i for i in range(3) if i != vertical] + [vertical]
            frame = frame[:, order]
            frame[:, 2] *= np.sign(alignment[vertical])
            extent = model.geom_size[geom, order]
        else:
            continue
        faces.append(geom)
        frames.append(frame)
        extents.append(extent)
        planes.append(kind == mujoco.mjtGeom.mjGEOM_PLANE)
    if not faces:
        return ()

    def tensor(values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)

    centers = tensor(data.geom_xpos[faces])
    frames, extents = tensor(frames), tensor(extents)
    planes = torch.as_tensor(planes, dtype=torch.bool, device=device)
    explicit_pairs = {tuple(sorted(pair)) for pair in zip(model.pair_geom1, model.pair_geom2, strict=True)}
    bindings = []
    for hand, name in enumerate(hand_body_names):
        root = model.body(name).id
        geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == root)
        possible = ((model.geom_contype[geoms, None] & model.geom_conaffinity[faces])
                    | (model.geom_conaffinity[geoms, None] & model.geom_contype[faces])) != 0
        for col, face in enumerate(faces):
            if explicit_pairs:
                possible[:, col] |= [tuple(sorted((geom, face))) in explicit_pairs for geom in geoms]
        hit = possible.any(-1)
        if not hit.any():
            continue
        geoms = geoms[hit]
        corners = [_collider_corners(model, geom) for geom in geoms]
        if any(points is None for points in corners):
            continue
        link_ids = [body_names.index(model.body(model.geom_bodyid[geom]).name) for geom in geoms]
        bindings.append((hand, body_names.index(name), torch.as_tensor(link_ids, device=device),
                         tensor(corners), torch.as_tensor(possible[hit], device=device),
                         centers, frames, extents, planes))
    return tuple(bindings)


def idle_support_depth(position, rotation, target_position, target_rotation, binding):
    """Project the achieved hand shape into the requested wrist frame and support bounds."""
    hand, root, links, corners, allowed, centers, frames, extents, planes = binding
    mapping = quat_mul(target_rotation[:, hand], quat_inv(rotation[:, root]))
    mapping = mapping[:, None].expand_as(rotation[:, links])
    desired_link = target_position[:, hand, None] + quat_apply(mapping, position[:, links] - position[:, root, None])
    orientation = quat_mul(mapping, rotation[:, links])
    vertex = quat_apply(orientation[:, :, None].expand(-1, -1, corners.shape[1], -1),
                        corners[None].expand(len(position), -1, -1, -1)) + desired_link[:, :, None]
    local = torch.einsum("wgvsc,scd->wgvsd", vertex[..., None, :] - centers, frames)
    low, high = local.amin(dim=2), local.amax(dim=2)
    footprint = ((low[..., :2] <= extents[:, :2]) & (high[..., :2] >= -extents[:, :2])).all(-1)
    intersects = allowed & footprint & (planes | (high[..., 2] >= -extents[:, 2]))
    depth = torch.where(intersects, extents[:, 2] - low[..., 2], -torch.inf)
    return depth.amax(dim=(1, 2))
