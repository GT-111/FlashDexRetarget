"""Fixed-plane clearance in the compiled native hand collision geometry."""

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul
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


def bind_idle_support_planes(model, hand_body_names, body_names, device):
    """Bind collider bounds and horizontal fixed planes that can collide with each hand."""
    gravity = model.opt.gravity
    if not np.linalg.norm(gravity):
        return ()
    up = -gravity / np.linalg.norm(gravity)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    planes = np.flatnonzero((model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
                            & (model.body_weldid[model.geom_bodyid] == 0)
                            & (data.geom_xmat.reshape(-1, 3, 3)[..., 2] @ up > 1. - 1e-6))
    if not len(planes):
        return ()
    heights = data.geom_xpos @ up
    explicit_pairs = {tuple(sorted(pair)) for pair in zip(model.pair_geom1, model.pair_geom2, strict=True)}
    bindings = []
    for hand, name in enumerate(hand_body_names):
        root = model.body(name).id
        geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == root)
        possible = ((model.geom_contype[geoms, None] & model.geom_conaffinity[planes])
                    | (model.geom_conaffinity[geoms, None] & model.geom_contype[planes])) != 0
        for col, plane in enumerate(planes):
            if explicit_pairs:
                possible[:, col] |= [tuple(sorted((geom, plane))) in explicit_pairs for geom in geoms]
        hit = possible.any(-1)
        if not hit.any():
            continue
        geoms = geoms[hit]
        corners = [_collider_corners(model, geom) for geom in geoms]
        if any(points is None for points in corners):
            continue
        link_ids = [body_names.index(model.body(model.geom_bodyid[geom]).name) for geom in geoms]
        plane_heights = np.where(possible[hit], heights[planes], -np.inf).max(-1)
        bindings.append((hand, body_names.index(name), torch.as_tensor(link_ids, device=device),
                         torch.as_tensor(np.stack(corners), dtype=torch.float32, device=device),
                         torch.as_tensor(plane_heights, dtype=torch.float32, device=device)))
    return tuple(bindings)


def idle_plane_depth(position, rotation, target_position, target_rotation, up, binding):
    """Project actual collider bounds in the requested rigid wrist transform."""
    hand, root, links, corners, heights = binding
    mapping = quat_mul(rotation[:, root], quat_inv(target_rotation[:, hand]))
    current_up = quat_apply(mapping, up.expand_as(position[:, root]))
    local_up = quat_apply_inverse(rotation[:, links], current_up[:, None].expand_as(position[:, links]))
    nearest_corner = (local_up[..., None, :] * corners).sum(-1).amin(-1)
    link_height = ((position[:, links] - position[:, root, None]) * current_up[:, None]).sum(-1)
    target_height = (target_position[:, hand] @ up)[:, None]
    return (heights - (target_height + link_height + nearest_corner)).amax(-1)
