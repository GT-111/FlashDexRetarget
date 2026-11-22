"""Isaac Lab scene construction + mjlab-Scene-compatible adapter (DirectRLEnv
build order). Import only AFTER the Isaac Sim app is running."""

from __future__ import annotations

import os
from importlib import metadata as package_metadata
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from envs.scene._object_setup import object_park_position
from envs.scene.native import NativeSceneCfg
from simulator.mjcf_meta import RobotMjcfMeta
from simulator.native.contacts import digit_body_groups
from simulator.native.entity import NativeRobotAdapter, _FreshnessClock

from .assets import IsaacRobotAsset, resolve_robot_asset
from .convert_assets import colvis_enabled, validate_native_usd
from .entity import (
    ArticulationAdapter,
    FreeArticulationAdapter,
    RigidObjectAdapter,
)
from .native_contacts import (
    IsaacHandPartContact,
    IsaacNativeContactReadback,
    create_named_contact_view,
)
from .sensors import build_tracking_contact_views
from .sim import IsaacSimAdapter


def _abl(name: str) -> bool:
    """Physics-ablation probe switch (FDR_ABL_*), all default off.
    Used by scripts/run_physics_ablation_b200.sh to attribute PhysX cost."""
    return os.environ.get(f"FDR_ABL_{name}", "0") == "1"


def _alllink_bodies_by_side(cfg) -> dict[str, list[str]]:
    """Configured sensor order, or the selected robot's explicit contact links."""
    out: dict[str, list[str]] = {}
    for side in ("right", "left"):
        pat = None
        for sn in (cfg.scene.get("sensors", None) or []):
            if str(sn.get("name", "")) == f"{side[0]}_alllink_contact":
                pat = [str(b) for b in sn["primary"]["pattern"]]
                break
        out[side] = pat if pat is not None else list(cfg.robot.contact_links.get(side, []))
    return out


class IsaacSceneAdapter:
    """Duck-typed mjlab ``Scene``: name lookup, origins, step-loop hooks."""

    def __init__(
        self, scene: InteractiveScene, device: str, sim_ctx: sim_utils.SimulationContext
    ) -> None:
        self.backend_identity = {
            "backend": "isaacsim", "solver": "PhysX",
            "physics_clock": {
                "authority": "Isaac Sim SimulationContext.current_time physics-step callback accumulation",
                "scope": "scene-global time repeated over cloned environments",
                "partial_reset": "entity reset does not reset the global SDK clock",
                "continuation": "raw callback clock; no accepted-step ledger or opaque solver snapshot",
            },
        }
        self._sim_ctx = sim_ctx
        self.root_wrench_application = {
            "frame": "WORLD", "point": "root_link_COM",
            "recorded_wrench_point": "root_link_COM",
        }
        self._scene = scene
        self.device = device
        self.entities: dict[str, object] = {}
        self.sensors: dict[str, object] = {}
        self.sensor_context = None
        self._reset_entities = None
        self._tracking_banks = {}
        self.native_contacts = None

    def bind_tracking_command(self, command) -> None:
        """Bind this scene's views after the command has admitted its references."""
        for bank in self._tracking_banks.values():
            bank.bind_command(command)

    def tracking_contact_metadata(self):
        return {side: bank.measurement_metadata for side, bank in self._tracking_banks.items()}

    def hand_part_contact_facts(self):
        return None if self.native_contacts is None else self.native_contacts.facts()

    def hand_part_contact_interval_facts(self):
        return None if self.native_contacts is None else self.native_contacts.interval_facts()

    def applied_object_wrench_interval_facts(self):
        return None if self.native_contacts is None else self.native_contacts.applied_wrench_interval_facts()

    def support_contact_interval_pairs(self):
        if self.native_contacts is None:
            return {}
        interval = self.native_contacts.interval
        return {name: interval.pairs[name] for name in self.native_layout.support_pair_bindings
                if name in interval.pairs}

    def physics_time_s(self):
        """Raw scene-global SDK physics time, repeated over the clone-world axis."""
        return torch.full(
            (self.num_envs,), self._sim_ctx.current_time, dtype=torch.float64, device=self.device
        )

    def physics_execution_clock(self):
        return None if self.native_contacts is None else self.native_contacts.execution_clock()

    @property
    def num_envs(self) -> int:
        return self._scene.cfg.num_envs

    @property
    def env_origins(self) -> torch.Tensor:
        return self._scene.env_origins

    def __getitem__(self, name: str):
        if name in self.entities:
            return self.entities[name]
        if name in self.sensors:
            return self.sensors[name]
        raise KeyError(name)

    def __contains__(self, name: str) -> bool:
        return name in self.entities or name in self.sensors

    def keys(self):
        return list(self.entities.keys()) + list(self.sensors.keys())

    def write_data_to_sim(self) -> None:
        self._scene.write_data_to_sim()

    def update(self, dt: float) -> None:
        self._scene.update(dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # SDK resets clear persistent wrenches; adapter restoration must be last.
        self._scene.reset(env_ids)
        entities = self.entities.values() if self._reset_entities is None else self._reset_entities
        for ent in entities:
            ent.reset(env_ids)


def _robot_articulation_cfg(
    meta: RobotMjcfMeta, asset: IsaacRobotAsset,
    self_collision: bool, pos_iters: int, vel_iters: int,
    contact_offset: float = 0.002, rest_offset: float = 0.0,
    max_depen: float = 5.0, max_contact_impulse: float | None = None,
) -> ArticulationCfg:
    """Explicit converted USD with MJCF-derived implicit PD gains."""
    robot_usd = asset.usd_path
    if not robot_usd.exists():
        raise FileNotFoundError(
            f"{robot_usd} missing — run convert_assets.py --robot-config <robot-or-run.yaml>"
            + (" --collision-visual" if colvis_enabled() else "")
        )

    def _gains(joints: list[str]) -> tuple[dict, dict]:
        kp = {}
        kv = {}
        for j in joints:
            p, v = meta.kp_kv_for_joint(j)
            kp[j] = p
            # MJCF total joint damping = servo kv + passive joint damping.
            kv[j] = v + float(meta.joint_damping[meta.joint_names.index(j)])
        return kp, kv

    kp, kv = _gains(meta.joint_names)
    actuators = {
        "all": ImplicitActuatorCfg(
            joint_names_expr=list(meta.joint_names),
            stiffness=kp,
            damping=kv,
            armature={j: float(meta.joint_armature[i]) for i, j in enumerate(meta.joint_names)},
            friction={
                j: float(meta.joint_frictionloss[i])
                for i, j in enumerate(meta.joint_names)
            },
            velocity_limit_sim=100.0,
            effort_limit_sim={
                j: float(meta.actuator_forcerange[meta.actuator_names.index(a)][1])
                for a, j in meta.actuator_joint.items()
            },
        )
    }
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(robot_usd),
            activate_contact_sensors=not _abl("NOREPORT"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=max_depen,
                **({} if max_contact_impulse is None
                   else {"max_contact_impulse": max_contact_impulse}),
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=contact_offset, rest_offset=rest_offset
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collision,
                solver_position_iteration_count=pos_iters,
                solver_velocity_iteration_count=vel_iters,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={j: 0.0 for j in meta.joint_names},
        ),
        actuators=actuators,
    )


def _usd_scale(scale: float) -> tuple[float, float, float] | None:
    """Uniform prim scale for a slot, or None at 1.0 so no xformOp:scale is authored."""
    return None if float(scale) == 1.0 else (float(scale),) * 3


def _object_rigid_cfg(
    entity_name: str, usd_path: str, park: tuple, pos_iters: int, vel_iters: int,
    scale: float = 1.0, contact_offset: float = 0.002, rest_offset: float = 0.0,
    max_depen: float = 5.0, max_contact_impulse: float | None = None,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{entity_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=_usd_scale(scale),
            activate_contact_sensors=not _abl("NOREPORT"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=max_depen,
                **({} if max_contact_impulse is None
                   else {"max_contact_impulse": max_contact_impulse}),
                linear_damping=0.0,
                angular_damping=0.0,
                max_angular_velocity=200.0,
                solver_position_iteration_count=pos_iters,
                solver_velocity_iteration_count=vel_iters,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=not _abl("NOOBJCOL"),
                contact_offset=contact_offset, rest_offset=rest_offset
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=park),
    )



def _spawn_support_disks(cfg, stage, num_envs: int, coff: float, roff: float, mat_path: str) -> dict:
    """CHORD support disks: count x radii kinematic cylinders per env (parked), one RigidObject
    view per (k, radius bucket). Same grid as envs/scene/_support_disk_setup.py (mujoco)."""
    sd = cfg.commands.object.support_disks
    if not bool(sd.enable):
        return {}
    from omegaconf import open_dict
    from pxr import Sdf, UsdPhysics

    from envs.scene._support_disk_setup import support_disk_name, support_park_position

    filter_robot = not bool(sd.get("robot_collision", True))
    views: dict[str, RigidObject] = {}
    names: list[list[str]] = []
    for k in range(int(sd.count)):
        row = []
        for j, r in enumerate(sd.radii):
            name = support_disk_name(k, j)
            c = sim_utils.CylinderCfg(
                radius=float(r),
                height=float(sd.height),
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=coff, rest_offset=roff
                ),
                # isaaclab's spawn_preview_surface re-attaches the stage per prim (O(stage));
                # per-env mode spawns count x radii x num_envs disks, so headless skips the material.
                visual_material=(
                    sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(float(v) for v in sd.rgba[:3])
                    )
                    if os.environ.get("FDR_DISK_VISMAT") == "1"
                    else None
                ),
            )
            park = support_park_position(sd, k, j)
            for i in range(num_envs):
                path = f"/World/envs/env_{i}/{name}"
                # Reference clones (copy_from_source=False) already show env_0's disk in every env.
                if stage.GetPrimAtPath(path).IsValid():
                    continue
                c.func(path, c, translation=park)
                _apply_contact_material(stage, path, mat_path)
                if filter_robot:
                    # CHORD's hand task masks robot<->support-surface pairs; nested prims are
                    # included and the clone arc remaps the target onto each env's own Robot.
                    UsdPhysics.FilteredPairsAPI.Apply(
                        stage.GetPrimAtPath(path)
                    ).CreateFilteredPairsRel().AddTarget(
                        Sdf.Path(f"/World/envs/env_{i}/Robot")
                    )
            views[name] = RigidObject(
                RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/{name}",
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=park),
                )
            )
            row.append(name)
        names.append(row)
    with open_dict(sd):
        sd.entity_names = names
    if filter_robot:
        # Reference clones carry env_0's mask through the arc: check both ends of the env range,
        # every disk, or some envs would silently keep hitting the desk.
        for i in sorted({0, num_envs - 1}):
            for n in (n for row in names for n in row):
                path = f"/World/envs/env_{i}/{n}"
                rel = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(path)).GetFilteredPairsRel()
                targets = [str(t) for t in (rel.GetTargets() if rel else [])]
                if targets != [f"/World/envs/env_{i}/Robot"]:
                    raise RuntimeError(f"support-disk robot mask missing on {path}: {targets}")
        print(
            f"[isaac] support disks: robot contact pairs masked on {len(views)} disks x {num_envs} "
            f"envs (support_disks.robot_collision=false)"
        )
    return views


class SupportDiskAdapter:
    """Kinematic disk view with the mjlab mocap write API the command uses."""

    def __init__(self, obj: RigidObject, device: str) -> None:
        self._obj = obj
        self.device = device

    def write_mocap_pose_to_sim(self, pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self._obj.write_root_link_pose_to_sim(pose, env_ids=env_ids)

    def reset(self, env_ids=None) -> None:
        self._obj.reset(env_ids)


def _apply_contact_material(stage, prim_path: str, mat_path: str, collision_props=None) -> None:
    """Make referenced colliders editable, install their selected properties/material."""
    from pxr import Usd

    prim = stage.GetPrimAtPath(prim_path)
    prim.SetInstanceable(False)
    paths = [
        pr.GetPath()
        for pr in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
        if pr.IsInstanceable()
    ]
    for path in paths:
        sub = stage.GetPrimAtPath(path)
        if sub:
            sub.SetInstanceable(False)
    if collision_props is not None:
        sim_utils.modify_collision_properties(prim_path, collision_props)
    from isaaclab.sim.utils import bind_physics_material

    bind_physics_material(prim_path, mat_path)


def build_isaac_scene(
    cfg,
    sim_ctx,
    device: str,
    meta: RobotMjcfMeta,
    object_slots: list[dict],
    assigned_obj_slot: torch.Tensor | None = None,
):
    """Author + clone the scene, boot physics, wrap in adapters. assigned_obj_slot
    (B,) selects PER-ENV-OBJECT mode; None = legacy parked-slot mode."""
    if cfg.robot.get("native_scene") is not None:
        return _build_native_isaac_scene(cfg, sim_ctx, device, meta)
    robot_asset = resolve_robot_asset(cfg, collision_visual=colvis_enabled())
    num_envs = int(cfg.scene.num_envs)
    per_env_mode = assigned_obj_slot is not None
    scene = InteractiveScene(
        InteractiveSceneCfg(
            num_envs=num_envs,
            env_spacing=float(cfg.scene.env_spacing),
            replicate_physics=not (per_env_mode or _abl("NOREPLICATE")),
        )
    )

    # ── static scene ──────────────────────────────────────────────────────
    # A second Kit process on the node loses the shared kvdb asset cache, so the nucleus
    # ground plane stops resolving; ISAAC_ASSET_ROOT points at an on-disk mirror instead
    # (the cfg default is baked at import, hence the explicit override here).
    if not _abl("NOGROUND"):
        ground = sim_utils.GroundPlaneCfg()
        if os.environ.get("ISAAC_ASSET_ROOT"):
            ground.usd_path = (
                f"{os.environ['ISAAC_ASSET_ROOT']}/Isaac/Environments/Grid/default_environment.usd"
            )
        ground.func("/World/ground", ground)
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
    )

    # Contact material (friction parity + solimp-like compliance), bound per
    # asset because instanced references shadow scene-default materials.
    import isaacsim.core.utils.stage as stage_utils

    stage = stage_utils.get_current_stage()
    _MAT = "/World/Materials/handContact"
    _MAT_OBJ = "/World/Materials/objContact"
    _px = cfg.sim.physx
    # object_friction=null keeps objects on the hand's material, as before.
    _hand_fric = float(_px.get("friction", 1.0))
    _obj_fric_raw = _px.get("object_friction", None)
    _obj_fric = _hand_fric if _obj_fric_raw is None else float(_obj_fric_raw)

    # "max" on the non-object materials pins hand/disk-object friction at the hand value
    # (PhysX: MAX beats AVERAGE) so object_friction then only changes object-object pairs.
    _combine = str(_px.get("nonobject_friction_combine_mode", "average"))

    def _mat(path: str, fric: float, combine: str = "average"):
        c = sim_utils.RigidBodyMaterialCfg(
            static_friction=fric,
            dynamic_friction=fric,
            restitution=0.0,
            friction_combine_mode=combine,
            compliant_contact_stiffness=float(
                _px.get("compliant_contact_stiffness", 0.0)
            ),
            compliant_contact_damping=float(_px.get("compliant_contact_damping", 0.0)),
        )
        c.func(path, c)

    _mat(_MAT, _hand_fric, _combine)
    _mat(_MAT_OBJ, _obj_fric)
    # disk_friction=null keeps the support disks on the object material, as before.
    _MAT_DISK = _MAT_OBJ
    if _px.get("disk_friction", None) is not None:
        _MAT_DISK = "/World/Materials/diskContact"
        _mat(_MAT_DISK, float(_px.get("disk_friction")), _combine)

    # ── env_0 assets ──────────────────────────────────────────────────────
    px = cfg.sim.physx
    pos_iters = int(px.solver_position_iteration_count)
    vel_iters = int(px.solver_velocity_iteration_count)
    coff = float(px.get("contact_offset", 0.002))
    roff = float(px.get("rest_offset", 0.0))
    mdep = float(px.get("max_depenetration_velocity", 5.0))
    mimp = px.get("max_contact_impulse", None)
    mimp = None if mimp is None else float(mimp)
    robot_cfg = _robot_articulation_cfg(
        meta, robot_asset, bool(cfg.robot.get("self_collision", False)), pos_iters, vel_iters,
        coff, roff, mdep, mimp,
    )
    robot = Articulation(robot_cfg.replace(
        spawn=robot_cfg.spawn.replace(collision_props=None)
    ))
    _apply_contact_material(
        stage, "/World/envs/env_0/Robot", _MAT, robot_cfg.spawn.collision_props
    )
    objects: dict[str, RigidObject] = {}
    if not per_env_mode:
        for slot, spec in enumerate(object_slots):
            park = object_park_position(slot)
            object_cfg = _object_rigid_cfg(
                spec["entity_name"], spec["usd_path"], park, pos_iters, vel_iters,
                float(spec.get("scale", 1.0)), coff, roff, mdep, mimp,
            )
            objects[spec["entity_name"]] = RigidObject(object_cfg.replace(
                spawn=object_cfg.spawn.replace(collision_props=None)
            ))
            _apply_contact_material(
                stage, f"/World/envs/env_0/{spec['entity_name']}", _MAT_OBJ,
                object_cfg.spawn.collision_props,
            )


    # per-env mode: independent copies — reference-clones would compose any
    # post-clone env_0 edit (the per-env object spawns) into EVERY env.
    scene.clone_environments(copy_from_source=per_env_mode)

    # Bimanual pair-slot specs store per-side sub-dicts; single-side specs are
    # scalars whose side is baked into the entity name (object_<side>[_<slot>]).
    def _slot_sides(spec: dict) -> list[str]:
        if isinstance(spec["usd_path"], dict):
            return list(spec["usd_path"])
        return ["left"] if "object_left" in spec.get("entity_name", "") else ["right"]

    def _slot_val(spec: dict, key: str, side: str):
        v = spec[key]
        return v[side] if isinstance(v, dict) else v

    obj_sides = sorted({s for sp in object_slots for s in _slot_sides(sp)}) or ["right"]
    # Index into the per-side gid/hull lists: equals slot_idx when every slot has both
    # sides (per-env mode), and counts within the side for one-sided legacy slots.
    _side_slot: dict[tuple[int, str], int] = {}

    if per_env_mode:
        # Post-clone per-env USD references (heterogeneous objects), then one
        # RigidObject view PER SIDE over the regex path (cfg.spawn=None).
        obj_spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=_slot_val(object_slots[0], "usd_path", obj_sides[0]),
            activate_contact_sensors=not _abl("NOREPORT"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=mdep,
                **({} if mimp is None else {"max_contact_impulse": mimp}),
                linear_damping=0.0,
                angular_damping=0.0,
                max_angular_velocity=200.0,
                solver_position_iteration_count=pos_iters,
                solver_velocity_iteration_count=vel_iters,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=not _abl("NOOBJCOL"),
                contact_offset=coff, rest_offset=roff
            ),
        )
        # Mixed pair pt: a shared slot's left prim is an inert ghost (kinematic, no gravity,
        # parked under the ground) so the per-side view stays uniform; the command aliases it.
        ghost_spawn_cfg = obj_spawn_cfg.replace(
            rigid_props=obj_spawn_cfg.rigid_props.replace(
                disable_gravity=True, kinematic_enabled=True
            )
        )
        slots_np = assigned_obj_slot.cpu().numpy()
        for i in range(num_envs):
            spec = object_slots[int(slots_np[i])]
            for side in obj_sides:
                ghost = side in spec.get("ghost", ())
                c = (ghost_spawn_cfg if ghost else obj_spawn_cfg).replace(
                    usd_path=_slot_val(spec, "usd_path", side),
                    scale=_usd_scale(
                        _slot_val(spec, "scale", side) if "scale" in spec else 1.0
                    ),
                )
                c.func(
                    f"/World/envs/env_{i}/object_{side}",
                    c.replace(collision_props=None),
                    translation=(0.0, 0.0, -3.0)
                    if ghost
                    else (0.0, 0.15 if side == "left" else 0.0, 0.2),
                )
                _apply_contact_material(
                    stage, f"/World/envs/env_{i}/object_{side}", _MAT_OBJ,
                    c.collision_props,
                )
        # Verify per-env spawns landed: a reference clone would compose env_0's
        # object everywhere and the spawner would skip silently.
        check = {i for i in (0, 1, num_envs // 2, num_envs - 1) if i < num_envs}
        for i in check:
            for side in obj_sides:
                prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/object_{side}")
                refs = (prim.GetMetadata("references") or None) if prim else None
                items = list(refs.prependedItems) if refs else []
                want = _slot_val(object_slots[int(slots_np[i])], "usd_path", side)
                got = items[0].assetPath if items else "<none>"
                if got != want:
                    raise RuntimeError(
                        f"per-env object spawn mismatch at env_{i}/{side}: "
                        f"{got} != {want}"
                    )
                # Scale-augmented pts: the prim MUST carry the slot's scale, else
                # PhysX simulates the original geometry against scaled refs/inertia.
                spec_i = object_slots[int(slots_np[i])]
                want_s = float(
                    _slot_val(spec_i, "scale", side) if "scale" in spec_i else 1.0
                )
                attr = prim.GetAttribute("xformOp:scale")
                got_s = float(attr.Get()[0]) if attr and attr.Get() is not None else 1.0
                if abs(got_s - want_s) > 1e-6:
                    raise RuntimeError(
                        f"per-env object scale mismatch at env_{i}/{side}: "
                        f"{got_s} != {want_s}"
                    )

        for side in obj_sides:
            objects[f"object_{side}"] = RigidObject(
                RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/object_{side}",
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
                )
            )

    support_views = _spawn_support_disks(cfg, stage, num_envs, coff, roff, _MAT_DISK)
    for name, obj in support_views.items():
        objects[name] = obj

    scene.filter_collisions(
        global_prim_paths=[] if _abl("NOGROUND") else ["/World/ground"]
    )

    # One ContactSensor per alllink body, net force only: PhysX GPU cannot
    # filter pairs vs COACD multi-shape colliders (attribution is SDF-side).
    alllink = _alllink_bodies_by_side(cfg)
    robot_sides = [
        s for s in ("right", "left")
        if alllink[s] and alllink[s][0] in meta.body_names
    ]
    link_sensors: dict[str, list[ContactSensor]] = {}
    for side in robot_sides:
        link_sensors[side] = []
        for body in [] if _abl("NOREPORT") else alllink[side]:
            urdf_body = robot_asset.body_name(body)
            s = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{urdf_body}",
                    update_period=0.0,
                )
            )
            link_sensors[side].append(s)

    # register into the InteractiveScene so update/write/reset drive them
    scene.articulations["robot"] = robot
    for name, obj in objects.items():
        scene.rigid_objects[name] = obj
    for side in robot_sides:
        for body, s in zip(alllink[side], link_sensors[side]):
            scene.sensors[f"_link_contact_{body}"] = s

    # ── boot physics (binds views, applies actuator cfgs) ─────────────────
    sim_ctx.reset()
    scene.update(0.0)

    if not bool(cfg.get("headless", True)):
        # GUI sanity-check framing: look at env_0's hand/object workspace.
        sim_ctx.set_camera_view(eye=(0.9, -0.9, 0.7), target=(0.0, 0.0, 0.15))

    # ── adapters ──────────────────────────────────────────────────────────
    clock = _FreshnessClock()
    sim_adapter = IsaacSimAdapter(
        sim_ctx, device, render=not bool(cfg.get("headless", True))
    )
    sim_adapter._clock = clock

    robot_adapter = ArticulationAdapter(
        robot, meta, clock, device, body_name_map=robot_asset.body_name_map
    )
    # 0.0 matches every run before the fix (resets cleared it); 0.8 is the MJCF gravcomp.
    sim_adapter.add_gravity_compensator(robot_adapter, float(cfg.sim.get("robot_gravcomp", 0.0)))

    scene_adapter = IsaacSceneAdapter(scene, device, sim_ctx)
    scene_adapter._clock = clock
    scene_adapter.entities[cfg.robot.entity_name] = robot_adapter
    for name in support_views:
        scene_adapter.entities[name] = SupportDiskAdapter(objects[name], device)

    # Optional hull-derived INSTALLATION recipe. Execution/control consumers
    # below read the installed PhysX properties, including the default USD path.
    from pathlib import Path as _P

    from .sim import hull_mass_props

    hull_mass, hull_com, hull_I = {s: [] for s in obj_sides}, {s: [] for s in obj_sides}, {s: [] for s in obj_sides}
    _hull_props = os.environ.get("FDR_HULL_PROPS", "0") == "1"
    if _hull_props:
        for _si, spec in enumerate(object_slots):
            for side in _slot_sides(spec):
                _side_slot[(_si, side)] = len(hull_mass[side])
                m, c, inertia = hull_mass_props(
                    _P(_slot_val(spec, "mesh_path", side)).parent,
                    _slot_val(spec, "scale", side) if "scale" in spec else 1.0,
                    800.0,
                )
                # Measured mass wins over the density estimate, clamp included.
                _tgt = float(_slot_val(spec, "mass", side)) if "mass" in spec else 0.0
                if _tgt > 0.0 and m > 0.0:
                    inertia, m = inertia * (_tgt / m), _tgt
                hull_mass[side].append(m)
                hull_com[side].append(c)
                hull_I[side].append(inertia)

    def _write_hull_props(view, slot_of_row: np.ndarray, side: str) -> None:
        """Overwrite PhysX mass/com/inertia with the hull-parity values (CPU)."""
        n = len(slot_of_row)
        view.set_masses(
            torch.tensor([hull_mass[side][s] for s in slot_of_row]).reshape(n, 1), torch.arange(n)
        )
        coms = view.get_coms().clone().reshape(n, 7)
        coms[:, :3] = torch.tensor(
            np.stack([hull_com[side][s] for s in slot_of_row]), dtype=coms.dtype
        )
        view.set_coms(coms.reshape(*view.get_coms().shape), torch.arange(n))
        view.set_inertias(
            torch.tensor(
                np.stack([hull_I[side][s].T.reshape(9) for s in slot_of_row]),
                dtype=torch.float32,
            ),
            torch.arange(n),
        )

    if per_env_mode:
        scene_adapter.assigned_obj_slot = assigned_obj_slot.to(device)
        for side in obj_sides:
            name = f"object_{side}"
            if _hull_props:
                _write_hull_props(
                    objects[name].root_physx_view,
                    assigned_obj_slot.cpu().numpy(),
                    side,
                )
            default_state = torch.zeros(num_envs, 13, device=device)
            default_state[:, 0:3] = scene.env_origins + torch.tensor(
                (0.0, 0.0, 0.2), dtype=torch.float32, device=device
            )
            default_state[:, 3] = 1.0
            scene_adapter.entities[name] = RigidObjectAdapter(
                objects[name], clock, device, default_state,
                body_name=objects[name].body_names[0],
            )
    else:
        for slot_idx, spec in enumerate(object_slots):
            if _hull_props:
                _write_hull_props(
                    objects[spec["entity_name"]].root_physx_view,
                    np.full(num_envs, _side_slot[(slot_idx, _slot_sides(spec)[0])]),
                    _slot_sides(spec)[0],
                )
            park = object_park_position(slot_idx)
            default_state = torch.zeros(num_envs, 13, device=device)
            default_state[:, 0:3] = scene.env_origins + torch.tensor(
                park, dtype=torch.float32, device=device
            )
            default_state[:, 3] = 1.0
            scene_adapter.entities[spec["entity_name"]] = RigidObjectAdapter(
                objects[spec["entity_name"]],
                clock, device,
                default_state,
                body_name=objects[spec["entity_name"]].body_names[0],
            )

    # Scene-owned views bind after complete command/reference construction.
    # one bank per hand side, each SDF-gated against its own side's object.
    for side in robot_sides:
        names = alllink[side]
        margins = torch.tensor(
            [meta.body_bound.get(b, 0.03) + 0.015 for b in names],
            dtype=torch.float32, device=device,
        )
        bank, views = build_tracking_contact_views(
            link_sensors[side], clock, device, side=side, body_names=names,
            contact_site_names=cfg.robot.site_names.contact[side], metadata=meta,
            robot=robot_adapter, margins=margins,
        )
        scene_adapter.sensors.update(views)
        scene_adapter._tracking_banks[side] = bank

    return scene_adapter, sim_adapter, robot_adapter


def _native_articulation_cfg(binding, meta, cfg) -> ArticulationCfg:
    """Native scalar drives only; no PD drive is created for a passive joint."""
    usd_path = cfg.sim.native_assets.articulations[binding.entity_name]
    if not Path(usd_path).is_file():
        raise FileNotFoundError(usd_path)
    validate_native_usd(usd_path, meta)
    driven = meta.driven_joint_names
    actuators = {}
    if driven:
        actuators["native_position"] = ImplicitActuatorCfg(
            joint_names_expr=driven,
            stiffness={j: meta.kp_kv_for_joint(j)[0] for j in driven},
            damping={
                j: meta.kp_kv_for_joint(j)[1]
                + float(meta.joint_damping[meta.joint_names.index(j)])
                for j in driven
            },
            effort_limit_sim={
                j: float(meta.actuator_effort_limit[meta.actuator_names.index(a)])
                for a, j in meta.actuator_joint.items()
            },
        )
    return ArticulationCfg(
        prim_path=f"/World/envs/env_.*/{binding.entity_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=bool(cfg.robot.get("self_collision", False)),
                solver_position_iteration_count=int(
                    cfg.sim.physx.solver_position_iteration_count
                ),
                solver_velocity_iteration_count=int(
                    cfg.sim.physx.solver_velocity_iteration_count
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(meta.free_roots[0].default_pose[:3]),
            rot=tuple(meta.free_roots[0].default_pose[3:]),
            joint_pos={
                name: float(value)
                for name, value in zip(meta.joint_names, meta.joint_default, strict=True)
            },
        ),
        actuators=actuators,
    )


def _native_rigid_object_cfg(binding, meta, usd_path) -> RigidObjectCfg:
    """The rigid family has one actual FREE body and no scalar articulation."""
    if meta.body_names != [binding.root_body_name]:
        raise ValueError("native rigid object requires its one declared physical body")
    if not Path(usd_path).is_file():
        raise FileNotFoundError(usd_path)
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{binding.entity_name}",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path, activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(meta.free_roots[0].default_pose[:3]),
            rot=tuple(meta.free_roots[0].default_pose[3:]),
        ),
    )


def _install_native_joint_properties(asset, meta, joint_ids, device) -> None:
    """Keep mimic drive gains zero; its intrinsic viscous term is in the USD."""
    scalar = dict(joint_ids=joint_ids)
    damping = meta.joint_damping.copy()
    followers = [meta.joint_names.index(m.joint_name) for m in meta.mimics]
    damping[followers] = 0.0
    if followers:
        asset.write_joint_stiffness_to_sim(
            0.0, joint_ids=[joint_ids[i] for i in followers],
        )
    for joint in meta.driven_joint_names:
        damping[meta.joint_names.index(joint)] += meta.kp_kv_for_joint(joint)[1]
    asset.write_joint_damping_to_sim(
        torch.as_tensor(damping, dtype=asset.data.joint_damping.dtype, device=device),
        **scalar,
    )
    asset.write_joint_armature_to_sim(
        torch.as_tensor(
            meta.joint_armature, dtype=asset.data.joint_armature.dtype, device=device,
        ),
        **scalar,
    )
    asset.write_joint_friction_coefficient_to_sim(
        torch.as_tensor(
            meta.joint_frictionloss, dtype=asset.data.joint_friction_coeff.dtype, device=device,
        ),
        **scalar,
    )


def _native_rigid_body_paths(scene, asset):
    from pxr import UsdPhysics

    entity_path = next(
        path for path in sim_utils.find_matching_prim_paths(asset.cfg.prim_path, stage=scene.stage)
        if path.startswith(f"{scene.env_prim_paths[0]}/")
    )
    return entity_path, {
        prim.GetName(): str(prim.GetPath())
        for prim in sim_utils.get_all_matching_child_prims(
            entity_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI), stage=scene.stage,
        )
    }


def _native_fixed_support_paths(scene):
    from pxr import UsdPhysics

    root = f"{scene.env_prim_paths[0]}/NativeStatic"
    colliders = sim_utils.get_all_matching_child_prims(
        root, predicate=lambda prim: prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
        stage=scene.stage,
    )
    paths = set()
    for collider in colliders:
        actor = collider
        while str(actor.GetPath()).startswith(root + "/") and not actor.HasAPI(UsdPhysics.RigidBodyAPI):
            actor = actor.GetParent()
        if not actor.HasAPI(UsdPhysics.RigidBodyAPI):
            actor = collider
        paths.add(str(actor.GetPath()))
    return sorted(paths)


def _bind_native_contact_readback(cfg, scene, sim_ctx, adapter, sim, body_paths):
    import mujoco

    env_zero = scene.env_prim_paths[0]
    layout = adapter.native_layout
    pairs, unavailable, unavailable_support = {}, {}, {}
    model = mujoco.MjModel.from_xml_path(layout.scene_xml_path)
    tip_sites = cfg.robot.site_names.tip
    finger_roles = cfg.robot.body_names.fingers
    fixed_paths = tuple(path.removeprefix(env_zero) for path in _native_fixed_support_paths(scene))

    def view_for(obj, part, filter_paths):
        sensor = body_paths[obj.entity_name][1][part.body_name].removeprefix(env_zero)
        sensors = [env + sensor for env in scene.env_prim_paths]
        filters = [[env + path for path in filter_paths] for env in scene.env_prim_paths]
        return create_named_contact_view(sim_ctx.physics_sim_view, sensors, filters)

    for hand in layout.hands:
        hand_entity = adapter.entities[hand.entity_name]
        hand_paths = body_paths[hand.entity_name][1]
        hand_bodies = [hand_paths[name].removeprefix(env_zero) for name in hand_entity.body_names]
        groups = digit_body_groups(model, hand.root_body_name, tip_sites[hand.side], finger_roles)
        digit_groups = {group: [model.body(body).name for body in bodies] for group, bodies in groups.items()}
        for obj, part in layout.part_bindings:
            key = f"{hand.side}/{obj.entity_name}/{part.name}"
            view, problem = view_for(obj, part, hand_bodies)
            if problem is not None:
                unavailable[key] = problem
            object_entity = adapter.entities[obj.entity_name]
            pairs[key] = IsaacHandPartContact(
                view, hand_entity, range(len(hand_bodies)), object_entity,
                object_entity.body_names.index(part.body_name),
                dict(hand_side=hand.side, hand_body_name=hand.root_body_name,
                     part_entity_name=obj.entity_name, part_body_name=part.body_name), adapter.device,
                digit_groups=digit_groups,
            )
    if fixed_paths:
        for key, (obj, part) in layout.support_pair_bindings.items():
            view, problem = view_for(obj, part, fixed_paths)
            if problem is not None:
                unavailable_support[key] = problem
            object_entity = adapter.entities[obj.entity_name]
            pairs[key] = IsaacHandPartContact(
                view, None, (), object_entity, object_entity.body_names.index(part.body_name),
                dict(part_entity_name=obj.entity_name, part_body_name=part.body_name,
                     first_fixed_collider_paths=fixed_paths), adapter.device,
            )
    readback = IsaacNativeContactReadback(
        sim_ctx, adapter.num_envs, adapter.device, float(cfg.sim.dt), int(cfg.decimation), pairs,
        object_wrenches={obj.entity_name: adapter.entities[obj.entity_name].native_object_wrench
                         for obj in layout.objects},
        wrench_parts=tuple(
            (obj.entity_name, part.body_name, adapter.entities[obj.entity_name].body_names.index(part.body_name))
            for obj, part in layout.part_bindings
        ),
    )
    adapter.native_contacts = sim.native_contacts = readback
    adapter.backend_identity["execution_clock"] = readback.execution_clock()["metadata"]
    adapter.backend_identity["unavailable_hand_part_reports"] = unavailable
    adapter.backend_identity["unavailable_fixed_support_reports"] = unavailable_support


def _build_native_isaac_scene(cfg, sim_ctx, device, scene_meta):
    """Install the named hand, object and static assets converted from the source scene."""
    layout = NativeSceneCfg.from_mapping(cfg.robot.native_scene)
    scene = InteractiveScene(
        InteractiveSceneCfg(
            num_envs=int(cfg.scene.num_envs),
            env_spacing=float(cfg.scene.env_spacing),
        )
    )
    static_cfg = sim_utils.UsdFileCfg(usd_path=cfg.sim.native_assets.static)
    static_cfg.func("/World/envs/env_0/NativeStatic", static_cfg)
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
    )
    assets, metadata, rigid = {}, {}, set()
    for binding in layout.articulations:
        meta = scene_meta.subtree(binding.root_body_name)
        if (
            len(meta.free_roots) != 1
            or meta.free_roots[0].name != binding.root_joint_name
        ):
            raise ValueError(f"wrong FREE root for {binding.entity_name!r}")
        metadata[binding.entity_name] = meta
        if binding in layout.objects and meta.actuator_names:
            raise ValueError("native object articulation must remain passive")
        if binding in layout.objects and not meta.joint_names:
            assets[binding.entity_name] = RigidObject(_native_rigid_object_cfg(binding, meta, cfg.sim.native_assets.articulations[binding.entity_name]))
            rigid.add(binding.entity_name)
        else:
            assets[binding.entity_name] = Articulation(
                _native_articulation_cfg(binding, meta, cfg)
            )
    body_paths = {name: _native_rigid_body_paths(scene, asset) for name, asset in assets.items()}
    contacts, contact_links = {}, {}
    for hand in layout.hands:
        hand_meta = metadata[hand.entity_name]
        names = list(cfg.robot.contact_links[hand.side])
        if not names:
            tip_bodies = {
                hand_meta.sites[site].body_name for site in cfg.robot.site_names.contact[hand.side]
            }
            names = [name for name in hand_meta.body_names if name in hand_meta.body_bound or name in tip_bodies]
        contact_links[hand.side] = names
        asset_path = assets[hand.entity_name].cfg.prim_path
        entity_path, bodies = body_paths[hand.entity_name]
        for body in names:
            hand_meta.body_names.index(body)
            relative_path = bodies[body].removeprefix(entity_path)
            contacts[body] = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"{asset_path}{relative_path}",
                    update_period=0.0,
                )
            )
    scene.clone_environments(copy_from_source=False)
    scene.filter_collisions(global_prim_paths=[])
    for name, asset in assets.items():
        (scene.rigid_objects if name in rigid else scene.articulations)[name] = asset
    for name, sensor in contacts.items():
        scene.sensors[f"_native_contact_{name}"] = sensor
    sim_ctx.reset()
    scene.update(0.0)
    clock = _FreshnessClock()
    sim = IsaacSimAdapter(sim_ctx, device, render=not bool(cfg.get("headless", True)))
    sim._clock = clock
    adapter = IsaacSceneAdapter(scene, device, sim_ctx)
    adapter._clock = clock
    adapter.native_layout = layout
    adapter.backend_identity.update(
        engine_versions={package: package_metadata.version(package) for package in ("isaacsim", "isaaclab")},
        scene_xml_path=str(layout.scene_xml_path),
        installed_usds={
            "static": str(cfg.sim.native_assets.static),
            "articulations": {name: str(path) for name, path in cfg.sim.native_assets.articulations.items()},
        },
    )
    hands = {}
    for binding in layout.articulations:
        meta = metadata[binding.entity_name]
        asset = assets[binding.entity_name]
        if binding.entity_name in rigid:
            default = torch.zeros(asset.num_instances, 13, device=device)
            default[:, :7] = torch.as_tensor(meta.free_roots[0].default_pose, device=device)
            ent = RigidObjectAdapter(
                asset, clock, device, default, body_name=binding.root_body_name, native_object=True,
            )
            adapter.entities[binding.entity_name] = ent
            continue
        ent = FreeArticulationAdapter(
            asset,
            meta,
            clock,
            device,
            root_body_name=binding.root_body_name,
            native_object=binding in layout.objects,
        )
        _install_native_joint_properties(
            asset, meta, ent._isaac_joint_ids.tolist(), device,
        )
        adapter.entities[binding.entity_name] = ent
        if binding in layout.hands:
            hands[binding.side] = ent
    robot = NativeRobotAdapter(hands, scene_meta=scene_meta, clock=clock, device=device)
    adapter.entities[cfg.robot.entity_name] = robot
    adapter._reset_entities = [
        adapter.entities[b.entity_name] for b in layout.articulations
    ]
    # Exact named parts are available to the public recorder; they are bodies
    # of their actual configured articulations, never duplicated hand aliases.
    adapter.object_parts = {
        f"{obj.entity_name}/{p.name}": {"entity_name": obj.entity_name, "body_name": p.body_name}
        for obj, p in layout.part_bindings
    }
    gravity = torch.tensor(sim.gravity_world_m_per_s2(), device=device, dtype=torch.float32)
    magnitude = gravity.norm()
    adapter.gravity_up_w = -gravity / magnitude.clamp_min(1e-12)
    adapter.native_object_weights_n = magnitude * torch.stack([
        adapter.entities[obj.entity_name].body_inertial_properties().mass_kg[0].sum()
        for obj in layout.objects
    ])
    _bind_native_contact_readback(cfg, scene, sim_ctx, adapter, sim, body_paths)
    for hand in layout.hands:
        names = contact_links[hand.side]
        margins = torch.tensor(
            [scene_meta.body_bound.get(n, 0.0) + 0.015 for n in names],
            device=device,
        )
        bank, views = build_tracking_contact_views(
            [contacts[n] for n in names], clock, device, side=hand.side,
            body_names=names, contact_site_names=cfg.robot.site_names.contact[hand.side],
            metadata=scene_meta, robot=robot, margins=margins,
        )
        adapter.sensors.update(views)
        adapter._tracking_banks[hand.side] = bank
    # TODO(Isaac gate): qualify constraint response and installed collision,
    # filter, material and inertia facts; USD admission does not prove parity.
    return adapter, sim, robot
