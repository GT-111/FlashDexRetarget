"""Named native articulations shared by scene, action and reference owners.

The MJCF is the mechanics authority. This configuration selects its subtrees;
engine-specific converted assets remain in the simulator configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, kw_only=True)
class NativeRootControlCfg:
    # Acceleration gains: translation is multiplied by subtree mass, rotation
    # by the current subtree inertia about its COM. All vectors are WORLD.
    position_stiffness: float
    position_damping: float
    rotation_stiffness: float
    rotation_damping: float
    force_limit_n: float | None
    torque_limit_nm: float | None
    gravity_compensation: float = 0.0
    gravity_world_m_per_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass(frozen=True, kw_only=True)
class NativeArticulationCfg:
    entity_name: str
    root_body_name: str
    root_joint_name: str


@dataclass(frozen=True, kw_only=True)
class NativeHandCfg(NativeArticulationCfg):
    side: str
    command_actuator_names: tuple[str, ...]
    root_control: NativeRootControlCfg


@dataclass(frozen=True, kw_only=True)
class NativeObjectPartCfg:
    name: str
    body_name: str
    # Original part-local surface used for reference/SDF queries. Collision
    # geometry and inertials remain those of scene_xml_path, not this mesh.
    mesh_path: str
    body_from_mesh_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    body_from_mesh_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    distance_sign: str = "unsigned"


@dataclass(frozen=True, kw_only=True)
class NativeObjectCfg(NativeArticulationCfg):
    parts: tuple[NativeObjectPartCfg, ...]


@dataclass(frozen=True, kw_only=True)
class NativeSceneCfg:
    scene_xml_path: str
    hands: tuple[NativeHandCfg, ...]
    object: NativeObjectCfg

    @classmethod
    def from_mapping(cls, config: Mapping) -> NativeSceneCfg:
        hands = tuple(
            NativeHandCfg(
                **{
                    k: str(h[k])
                    for k in (
                        "side",
                        "entity_name",
                        "root_body_name",
                        "root_joint_name",
                    )
                },
                command_actuator_names=tuple(h["command_actuator_names"]),
                root_control=NativeRootControlCfg(**dict(h["root_control"])),
            )
            for h in config["hands"]
        )
        obj = config["object"]
        return cls(
            scene_xml_path=str(config["scene_xml_path"]),
            hands=hands,
            object=NativeObjectCfg(
                **{
                    k: str(obj[k])
                    for k in (
                        "entity_name",
                        "root_body_name",
                        "root_joint_name",
                    )
                },
                parts=tuple(NativeObjectPartCfg(**dict(p)) for p in obj["parts"]),
            ),
        )

    @property
    def articulations(self) -> tuple[NativeArticulationCfg, ...]:
        return (*self.hands, self.object)


def bind_native_scene_config(cfg) -> NativeSceneCfg:
    """Bind mechanical names to reference/task selection, independently of engine."""
    from omegaconf import open_dict

    layout = NativeSceneCfg.from_mapping(cfg.robot.native_scene)
    with open_dict(cfg.commands):
        cfg.commands.native_scene = cfg.robot.native_scene
        cfg.commands.object.entity_names = {
            h.side: layout.object.entity_name for h in layout.hands
        }
    if cfg.commands.object.pin_objects:
        raise ValueError("native object task requires pin_objects=false")
    if cfg.commands.object.support_disks.enable:
        raise ValueError("native scene uses its authored static support")
    return layout
