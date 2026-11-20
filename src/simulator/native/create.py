"""Build the native task once; the configured plant owns its actual physics."""

from __future__ import annotations

from hydra.utils import get_class
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import OmegaConf

from envs import build_manager_cfgs
from envs.scene.native import bind_native_scene_config
from evaluation.shared import SharedEvalEnv
from simulator.env import ManagerBasedRlEnv, ManagerEnvCfg
from simulator.mjcf_meta import load_robot_mjcf_meta

from .scene import NativeScene, NativeSim


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
    bodies = {entry.root_body_name: [model.body(name).id for name in meta.subtree(entry.root_body_name).body_names]
              for entry in layout.articulations}
    pairs = {
        f"{hand.side}/{obj.entity_name}/{part.name}": (
            bodies[hand.root_body_name],
            [model.body(part.body_name).id],
        )
        for hand in layout.hands for obj, part in layout.part_bindings
    }
    pairs.update({key: (bodies[first.root_body_name], bodies[second.root_body_name])
                  for key, (first, second) in layout.object_pair_bindings.items()})
    approach = cfg.get("actions", {}).get("source_contact_approach") or {}
    if approach.get("stationary_hold_force_n", 0.):
        world_collision = (model.geom_bodyid == 0) & ((model.geom_contype | model.geom_conaffinity) != 0)
        if world_collision.any() and (model.geom_type[world_collision] == mujoco.mjtGeom.mjGEOM_PLANE).all():
            pairs.update({key: ([0], [model.body(part.body_name).id])
                          for key, (_, part) in layout.support_pair_bindings.items()})
    plant = plant_class(
        scene_xml_path=layout.scene_xml_path, num_envs=int(cfg.scene.num_envs),
        sim_dt=float(cfg.sim.dt), device=device, contact_pairs=pairs, **config,
    )
    plant.configure_control_interval(int(cfg.decimation))
    sim = NativeSim(plant)
    scene = NativeScene(layout, meta, sim, cfg.robot.entity_name)
    env = ManagerBasedRlEnv(
        ManagerEnvCfg(
            decimation=int(cfg.decimation), physics_dt=float(cfg.sim.dt),
            episode_length_s=float(cfg.episode_length_s), seed=cfg.seed,
            num_envs=int(cfg.scene.num_envs), **build_manager_cfgs(cfg),
        ), scene, sim, device=device,
    )
    return RslRlVecEnvWrapper(env), SharedEvalEnv(env, cfg.eval)
