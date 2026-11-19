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
    with open_dict(cfg), open_dict(cfg.robot):
        cfg.native_recording_binding = binding
        cfg.robot.native_scene = scene
        cfg.robot.xml_path = scene["scene_xml_path"]
        for hand in scene["hands"]:
            if "tip_site_names" in hand:
                cfg.robot.site_names.tip[hand["side"]] = hand["tip_site_names"]
    if "native_assets" in cfg.sim:
        cfg.sim.native_assets = native_usd_paths(motion_file, scene)
