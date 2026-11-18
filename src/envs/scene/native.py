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
    objects: tuple[NativeObjectCfg, ...]

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
        objects = tuple(
            NativeObjectCfg(
                **{k: str(obj[k]) for k in (
                    "entity_name", "root_body_name", "root_joint_name",
                )},
                parts=tuple(NativeObjectPartCfg(**dict(p)) for p in obj["parts"]),
            )
            for obj in config["objects"]
        )
        if not objects:
            raise ValueError("native scene requires its declared object articulations")
        layout = cls(
            scene_xml_path=str(config["scene_xml_path"]),
            hands=hands,
            objects=objects,
        )
        for field in ("entity_name", "root_body_name", "root_joint_name"):
            values = [getattr(binding, field) for binding in layout.articulations]
            if len(set(values)) != len(values):
                raise ValueError(f"native articulations require distinct {field}")
        return layout

    @property
    def articulations(self) -> tuple[NativeArticulationCfg, ...]:
        return (*self.hands, *self.objects)

    @property
    def part_bindings(self) -> tuple[tuple[NativeObjectCfg, NativeObjectPartCfg], ...]:
        """One actual owner and part order shared by references, contacts and export."""
        return tuple((obj, part) for obj in self.objects for part in obj.parts)

    @property
    def object_root_body_names(self) -> dict[str, str]:
        return {obj.entity_name: obj.root_body_name for obj in self.objects}


def bind_native_scene_config(cfg, *, constructor_gravity_world_m_per_s2) -> NativeSceneCfg:
    """Admit a native recipe with fixed constructor gravity and explicit force assistance."""
    from omegaconf import open_dict

    layout = NativeSceneCfg.from_mapping(cfg.robot.native_scene)
    assisted = bool(cfg.commands.object.pin_objects)
    if cfg.curriculum and not assisted:
        raise ValueError("native producer uses the declared static unassisted recipe; curriculum must be empty")
    if set(cfg.curriculum) - {"xfrc_curriculum_adaptive"}:
        raise ValueError("native assistance admits only xfrc_curriculum_adaptive; gravity remains the compiled constructor value")
    if assisted:
        obj = cfg.commands.object
        if obj.pin_mode != "xfrc":
            raise ValueError("native object assistance requires the explicit xfrc force mode")
        parts = layout.part_bindings
        if len({part.body_name for _, part in parts}) != len(parts):
            raise ValueError("native assistance requires one declared reference part per actual body")
        cscale = float(obj.xfrc.contact_scale)
        if obj.xfrc.implicit_damping or not (0.0 <= cscale <= 1.0):
            raise ValueError("native xfrc requires contact_scale in [0, 1] without implicit damping")
        if obj.perturb.enabled and not bool(obj.perturb.get("require_assist_decayed", True)):
            raise ValueError("native post-decay perturbation requires require_assist_decayed=True")
    gravity = tuple(float(value) for value in constructor_gravity_world_m_per_s2)
    for hand in layout.hands:
        control = hand.root_control
        if (control.gravity_compensation != 0.0
                and tuple(control.gravity_world_m_per_s2) != gravity):
            raise ValueError(
                f"native hand {hand.side!r} compensation gravity differs from compiled constructor gravity {gravity}"
            )
    with open_dict(cfg.commands):
        cfg.commands.native_scene = cfg.robot.native_scene
    if cfg.commands.object.support_disks.enable:
        raise ValueError("native scene uses its authored static support")
    return layout
