"""Resolve the prepared native dataset's scene before importing a simulator."""

from __future__ import annotations

import json
from pathlib import Path

_NATIVE_COMMAND = "envs.commands.motion_tracking.native.NativeMotionTrackingCommandCfg"


def load_native_reference(motion_file: str | Path) -> dict:
    """Read the sibling manifest and resolve its scene and part surfaces."""
    manifest_path = Path(motion_file).expanduser().absolute().with_name("native-reference.json")
    declaration = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = declaration.get("native_scene")
    if not isinstance(scene, dict):
        raise ValueError(f"{manifest_path} requires a native_scene mapping")

    scene["scene_xml_path"] = str((manifest_path.parent / scene["scene_xml_path"]).resolve())
    for obj in scene["objects"]:
        for part in obj["parts"]:
            part["mesh_path"] = str((manifest_path.parent / part["mesh_path"]).resolve())
    return declaration


def native_usd_paths(motion_file: str | Path, scene: dict) -> dict:
    """Isaac converts the native scene into one USD per named articulation."""
    usd = Path(motion_file).expanduser().absolute().parent / "native-usd"
    return {
        "static": str(usd / "static.usd"),
        "articulations": {
            entry["entity_name"]: str(usd / f"{entry['entity_name']}.usd")
            for entry in [*scene["hands"], *scene["objects"]]
        },
    }


def _bind_tip_sites(robot, hands: list[dict]) -> None:
    if not any("tip_source_point_names" in hand for hand in hands):
        if not robot.body_names.fingers:
            raise ValueError("prepared native profile requires tip_source_point_names for each hand")
        for hand in hands:
            if "tip_site_names" in hand:
                robot.site_names.tip[hand["side"]] = hand["tip_site_names"]
        return

    sites_by_side = {}
    for hand in hands:
        side = hand["side"]
        points = hand.get("tip_source_point_names", ())
        sites = hand.get("tip_site_names", ())
        if (not points or len(points) != len(sites) or len(set(points)) != len(points)
                or len(set(sites)) != len(sites) or any(not point.endswith("_tip") for point in points)):
            raise ValueError(f"native hand {side!r} requires one distinct MANO fingertip role per physical tip site")
        sites_by_side[side] = dict(zip(points, sites, strict=True))
    point_order = next(iter(sites_by_side.values()))
    if any(sites.keys() != point_order.keys() for sites in sites_by_side.values()):
        raise ValueError("native hands must declare the same fingertip source roles")
    robot.body_names.fingers = [point.removesuffix("_tip") for point in point_order]
    robot.joint_names.finger = [name for hand in hands for name in hand["command_actuator_names"]]
    for side, sites in sites_by_side.items():
        robot.site_names.tip[side] = [sites[point] for point in point_order]


def bind_prepared_native_reference(cfg) -> None:
    """Bind a native Hydra configuration before resolving its interpolations."""
    if cfg.commands.get("_target_") != _NATIVE_COMMAND:
        return

    motion_file = Path(cfg.commands.motion_file).expanduser().absolute()
    declaration = load_native_reference(motion_file)
    scene = declaration["native_scene"]
    binding = declaration.get("recording_binding")
    if not isinstance(binding, dict):
        raise ValueError(f"{motion_file.with_name('native-reference.json')} requires a recording_binding mapping")

    if "scene_identity" in binding:
        binding["scene_identity"]["native_scene"] = scene

    from omegaconf import open_dict

    cfg.commands.motion_file = str(motion_file)
    if len(scene["hands"]) == 1 and "per_hand_actor_critic" in cfg:
        cfg.per_hand_actor_critic = False
    with open_dict(cfg), open_dict(cfg.robot):
        cfg.native_recording_binding = binding
        cfg.robot.native_scene = scene
        cfg.robot.xml_path = scene["scene_xml_path"]
        _bind_tip_sites(cfg.robot, scene["hands"])
    if "native_assets" in cfg.sim:
        cfg.sim.native_assets = native_usd_paths(motion_file, scene)
