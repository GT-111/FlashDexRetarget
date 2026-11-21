"""Build the native task once; the configured plant owns its actual physics."""

from __future__ import annotations

import numpy as np
from hydra.utils import get_class
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import OmegaConf

from envs import build_manager_cfgs
from envs.scene.native import bind_native_scene_config
from evaluation.shared import SharedEvalEnv
from simulator.env import ManagerBasedRlEnv, ManagerEnvCfg
from simulator.mjcf_meta import load_robot_mjcf_meta

from .scene import NativeScene, NativeSim


def _digit_body_groups(model, hand_root, tip_sites, finger_roles):
    """Assign only bodies below anatomically unique, physically installed tip paths."""
    if len(tip_sites) != len(finger_roles) or "thumb" not in finger_roles or len(tip_sites) < 2:
        return {}
    root = model.body(hand_root).id
    paths = {"thumb": set(), "fingers": set()}
    for tip, role in zip(tip_sites, finger_roles, strict=True):
        current = int(model.site_bodyid[model.site(tip).id])
        path = paths["thumb" if role == "thumb" else "fingers"]
        while current != root:
            if current == 0:
                raise ValueError("native digit tip is outside its declared physical hand")
            path.add(current)
            current = int(model.body_parentid[current])
    shared = paths["thumb"] & paths["fingers"]
    groups = {name: branch - shared for name, branch in paths.items()}
    if not all(groups.values()):
        return {}
    # Include pad/collider descendants that live on a child body of a digit link.
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        for group in groups.values():
            if parent in group:
                group.add(body)
    return {group: sorted(members) for group, members in groups.items()}


def create_native_envs(cfg, device="cuda"):
    meta = load_robot_mjcf_meta(cfg.robot.native_scene.scene_xml_path)
    layout = bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=meta.gravity)
    if {r.body_name for r in meta.free_roots} != {b.root_body_name for b in layout.articulations}:
        raise ValueError("native scene binding must declare every actual FREE articulation")
    config = OmegaConf.to_container(cfg.sim.plant, resolve=True)
    plant_class = get_class(config.pop("_target_"))
    # Resolve actual body addresses from the same authored scene used by the plant.
    import mujoco
    model = mujoco.MjModel.from_xml_path(layout.scene_xml_path)
    subtrees = {entry.root_body_name: meta.subtree(entry.root_body_name) for entry in layout.articulations}
    bodies = {root: [model.body(name).id for name in subtree.body_names] for root, subtree in subtrees.items()}
    pairs = {
        f"{hand.side}/{obj.entity_name}/{part.name}": (
            bodies[hand.root_body_name],
            [model.body(part.body_name).id],
        )
        for hand in layout.hands for obj, part in layout.part_bindings
    }
    tip_sites = cfg.robot.get("site_names", {}).get("tip", {})
    finger_roles = cfg.robot.get("body_names", {}).get("fingers", ())
    digit_groups = {hand.side: _digit_body_groups(model, hand.root_body_name, tip_sites.get(hand.side, ()), finger_roles)
                    for hand in layout.hands}
    digit_contacts = {}
    if all(digit_groups.values()):
        for hand in layout.hands:
            for obj, part in layout.part_bindings:
                key = f"{hand.side}/{obj.entity_name}/{part.name}"
                digit_contacts[key] = {}
                for name, bodies_in_group in digit_groups[hand.side].items():
                    child_key = f"{key}/{name}"
                    pairs[child_key] = (bodies_in_group, [model.body(part.body_name).id])
                    digit_contacts[key][name] = (child_key, [model.body(i).name for i in bodies_in_group])
    pairs.update({key: (bodies[first.root_body_name], bodies[second.root_body_name])
                  for key, (first, second) in layout.object_pair_bindings.items()})
    fixed_geom = (model.body_weldid[model.geom_bodyid] == 0) & ((model.geom_contype | model.geom_conaffinity) != 0)
    fixed_support = np.unique(model.geom_bodyid[fixed_geom]).tolist()
    if fixed_support:
        pairs.update({key: (fixed_support, [model.body(part.body_name).id])
                      for key, (_, part) in layout.support_pair_bindings.items()})
    plant = plant_class(
        scene_xml_path=layout.scene_xml_path, num_envs=int(cfg.scene.num_envs),
        sim_dt=float(cfg.sim.dt), device=device, contact_pairs=pairs, **config,
    )
    plant.configure_hand_part_feedback({
        f"{hand.side}/{obj.entity_name}/{part.name}": tuple(
            subtrees[obj.root_body_name].joint_dof_address.tolist())
        for hand in layout.hands for obj, part in layout.part_bindings
    })
    plant.configure_control_interval(int(cfg.decimation))
    sim = NativeSim(plant)
    scene = NativeScene(layout, meta, sim, cfg.robot.entity_name, digit_contacts=digit_contacts)
    env = ManagerBasedRlEnv(
        ManagerEnvCfg(
            decimation=int(cfg.decimation), physics_dt=float(cfg.sim.dt),
            episode_length_s=float(cfg.episode_length_s), seed=cfg.seed,
            num_envs=int(cfg.scene.num_envs), **build_manager_cfgs(cfg),
        ), scene, sim, device=device,
    )
    return RslRlVecEnvWrapper(env), SharedEvalEnv(env, cfg.eval)
