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
    layout = bind_native_scene_config(cfg)
    meta = load_robot_mjcf_meta(layout.scene_xml_path)
    if {r.body_name for r in meta.free_roots} != {b.root_body_name for b in layout.articulations}:
        raise ValueError("native scene binding must declare every actual FREE articulation")
    if cfg.curriculum:
        raise ValueError("native producer uses the declared unassisted task recipe")
    config = OmegaConf.to_container(cfg.sim.plant, resolve=True)
    plant_class = get_class(config.pop("_target_"))
    # Resolve actual body addresses from the same authored scene used by the plant.
    import mujoco
    model = mujoco.MjModel.from_xml_path(layout.scene_xml_path)
    pairs = {
        f"{hand.side}/{part.name}": (
            [model.body(n).id for n in meta.subtree(hand.root_body_name).body_names],
            [model.body(part.body_name).id],
        )
        for hand in layout.hands for part in layout.object.parts
    }
    plant = plant_class(
        scene_xml_path=layout.scene_xml_path, num_envs=int(cfg.scene.num_envs),
        sim_dt=float(cfg.sim.dt), device=device, contact_pairs=pairs, **config,
    )
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
