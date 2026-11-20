"""Named native articulations shared by scene, action and reference owners.

The MJCF is the mechanics authority. This configuration selects its subtrees;
engine-specific converted assets remain in the simulator configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
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
    tip_site_names: tuple[str, ...] = ()
    command_names: tuple[str, ...] = ()
    actuator_command_indices: tuple[int, ...] = ()
    command_limits: tuple[tuple[float, float], ...] = ()
    command_state_joint_names: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.command_actuator_names or len(set(self.command_actuator_names)) != len(self.command_actuator_names):
            raise ValueError(f"native hand {self.side!r} requires distinct physical actuators")
        if not any((self.command_names, self.actuator_command_indices, self.command_limits, self.command_state_joint_names)):
            return
        n = len(self.command_names)
        if (not n or len(set(self.command_names)) != n
                or len(self.actuator_command_indices) != len(self.command_actuator_names)
                or any(type(index) is not int for index in self.actuator_command_indices)
                or set(self.actuator_command_indices) != set(range(n))
                or len(self.command_limits) != n or len(self.command_state_joint_names) != n
                or any(len(bounds) != 2 or not all(math.isfinite(v) for v in bounds)
                       or bounds[0] >= bounds[1] for bounds in self.command_limits)
                or any(not names or len(set(names)) != len(names) for names in self.command_state_joint_names)):
            raise ValueError(f"native hand {self.side!r} has an invalid named logical command binding")

    @property
    def command_count(self) -> int:
        return len(self.command_names or self.command_actuator_names)


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
                tip_site_names=tuple(h.get("tip_site_names", ())),
                command_names=tuple(h.get("command_names", ())),
                actuator_command_indices=tuple(h.get("actuator_command_indices", ())),
                command_limits=tuple(tuple(bounds) for bounds in h.get("command_limits", ())),
                command_state_joint_names=tuple(tuple(names) for names in h.get("command_state_joint_names", ())),
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
            for obj in config.get("objects", ())
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

    def command_coordinates(self, actuator_joint_names):
        """Ordered logical names, associated measured Q groups and physical A-to-C indices."""
        names, groups, indices = [], [], []
        start = 0
        for hand in self.hands:
            end = start + len(hand.command_actuator_names)
            actual = tuple(actuator_joint_names[start:end])
            indices.extend(len(names) + i for i in (
                hand.actuator_command_indices or range(len(actual))))
            names.extend(hand.command_names or actual)
            groups.extend(hand.command_state_joint_names or ((name,) for name in actual))
            start = end
        if (len(actuator_joint_names) != start or len(set(names)) != len(names)
                or len(set(actuator_joint_names)) != len(actuator_joint_names)):
            raise ValueError("native hand control requires distinct named logical commands and driven joints")
        return tuple(names), tuple(groups), tuple(indices)

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

    @property
    def object_pair_bindings(self) -> dict[str, tuple[NativeObjectCfg, NativeObjectCfg]]:
        return {f"objects/{first.entity_name}/{second.entity_name}": (first, second)
                for first, second in combinations(self.objects, 2)}


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
