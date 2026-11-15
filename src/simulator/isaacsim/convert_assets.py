"""Standalone CLI: convert COACD objects + configured robot URDFs to USD via IsaacLab
UrdfConverter (headless AppLauncher boot)."""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from simulator.isaacsim.assets import load_robot_asset

OBJ_DENSITY = 800.0  # kg/m^3, matches get_object_spec / holosoma taco URDFs
OBJ_MASS_CLAMP = (0.05, 0.5)  # kg, matches get_object_spec

# "collision == visual" debug variant: same physics, visuals replaced by the
# exact shapes PhysX collides with. Runtime switch: FDR_COLVIS=1.
COLVIS_OBJ_DIRNAME = ".isaac_usd_colvis"
COLVIS_RGBA = "0.85 0.45 0.2 1"


def colvis_enabled() -> bool:
    """Runtime switch (FDR_COLVIS=1) for the collision-as-visual assets."""
    return os.environ.get("FDR_COLVIS", "0") == "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objects", nargs="*", default=[], help="object ids under <data-root>/objects/",
    )
    parser.add_argument("--data-root", help="dataset directory holding objects/ (needed with --objects)")
    parser.add_argument(
        "--robot-config",
        help="stock robot YAML or resolved run YAML with an explicit sim.robot_asset binding",
    )
    parser.add_argument("--force", action="store_true", help="reconvert even if USD is up to date")
    parser.add_argument(
        "--native-scene-config",
        help="resolved run yaml with robot.native_scene and sim.native_assets: split its named MJCF assets",
    )
    parser.add_argument(
        "--collision-visual", action="store_true",
        help="write the debug variant whose visual meshes ARE the collision geometry "
        f"(objects -> {COLVIS_OBJ_DIRNAME}/, robot -> configured collision_visual_usd_path); "
        "select at runtime with FDR_COLVIS=1",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def _log(msg: str) -> None:
    print(f"[convert_assets] {msg}", flush=True)


def _up_to_date(usd_path: str, sources: list[str]) -> bool:
    if not os.path.exists(usd_path):
        return False
    usd_mtime = os.path.getmtime(usd_path)
    return all(os.path.getmtime(s) <= usd_mtime for s in sources if os.path.exists(s))


def validate_native_mjcf_import(model) -> None:
    """Admit the mechanics implemented by the pinned native Isaac importer.

    In particular, its MJCF parser ignores joint equalities. They must be
    representable by our explicit affine, limited-revolute mimic installation.
    """
    import mujoco
    import numpy as np

    scalar = {
        j for j in range(model.njnt)
        if int(model.jnt_type[j]) in (
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
    }
    if any(
        j not in scalar and int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)
        for j in range(model.njnt)
    ):
        raise ValueError("native Isaac import requires scalar joints and declared FREE roots")
    if any(model.qpos0[int(model.jnt_qposadr[j])] != 0 for j in scalar):
        raise ValueError("native Isaac import requires zero scalar joint references")
    driven = {
        int(model.actuator_trnid[a, 0]) for a in range(model.nu)
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
    }
    if any(model.jnt_stiffness[j] != 0 for j in scalar - driven):
        raise ValueError("native Isaac passive joint springs are not implemented")
    free_bodies = {
        int(model.jnt_bodyid[j]) for j in range(model.njnt)
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    }

    def free_root(joint):
        body = int(model.jnt_bodyid[joint])
        while body and body not in free_bodies:
            body = int(model.body_parentid[body])
        return body

    followers = set()
    for e in range(model.neq):
        first, second = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if (
            int(model.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT)
            or not model.eq_active0[e] or second < 0
            or first not in scalar or second not in scalar
            or int(model.jnt_type[first]) != int(mujoco.mjtJoint.mjJNT_HINGE)
            or int(model.jnt_type[second]) != int(mujoco.mjtJoint.mjJNT_HINGE)
            or not model.jnt_limited[first] or not model.jnt_limited[second]
            or np.any(model.eq_data[e, 2:5] != 0)
            or model.eq_data[e, 1] == 0
            or first in followers or first in driven
            or not free_root(first) or free_root(first) != free_root(second)
        ):
            raise ValueError(
                f"native Isaac equality {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e)!r} requires one "
                "active affine limited-revolute follower in the same FREE articulation"
            )
        followers.add(first)


@dataclass(frozen=True)
class NativeMimicImport:
    joint_name: str
    reference_joint_name: str
    gearing: float
    offset_degrees: float
    viscous_friction_per_degree: float


def native_mimic_imports(meta) -> tuple[NativeMimicImport, ...]:
    """MuJoCo reference-relative coefficients in the actual USD angular units."""
    result = []
    for mimic in meta.mimics:
        follower = meta.joint_names.index(mimic.joint_name)
        reference = meta.joint_names.index(mimic.reference_joint_name)
        p = mimic.polynomial
        if any(p[2:] != 0) or p[1] == 0:
            raise ValueError("native Isaac mimics require affine joint equalities")
        offset = (
            meta.joint_default[follower] + p[0]
            - p[1] * meta.joint_default[reference]
        )
        result.append(NativeMimicImport(
            mimic.joint_name, mimic.reference_joint_name, -float(p[1]),
            -math.degrees(float(offset)),
            float(meta.joint_damping[follower]) * math.pi / 180.0,
        ))
    return tuple(result)


def _materialize_native_inline_meshes(document, model, directory) -> None:
    """Use compiled triangles AND compiled geom poses in one centered chart."""
    from pathlib import Path

    import mujoco

    inline = {}
    for mesh in document.findall("asset/mesh"):
        if mesh.get("file") or mesh.get("vertex") is None:
            continue
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh.get("name"))
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        if not fn:
            raise ValueError(f"native inline mesh {mesh.get('name')!r} has no compiled triangles")
        path = Path(directory) / (mesh.get("name") + ".obj")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            for v in model.mesh_vert[va:va + vn]:
                output.write("v " + " ".join(format(float(x), ".17g") for x in v) + "\n")
            for face in model.mesh_face[fa:fa + fn]:
                output.write("f " + " ".join(str(int(x) + 1) for x in face) + "\n")
        # Scale/ref/recentering are already in compiled vertices and geom poses.
        mesh.attrib.clear()
        mesh.attrib.update(name=model.mesh(mid).name, file=str(path.resolve()), scale="1 1 1")
        inline[mid] = mesh.get("name")
    world = document.find("worldbody")
    for body in [world, *world.iter("body")]:
        bid = 0 if body is world else model.body(body.get("name")).id
        geoms = body.findall("geom")
        if len(geoms) != int(model.body_geomnum[bid]):
            raise ValueError("native mesh import requires the authored body/geom roster")
        for i, geom in enumerate(geoms):
            gid = int(model.body_geomadr[bid]) + i
            if (
                int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH)
                or int(model.geom_dataid[gid]) not in inline
            ):
                continue
            for key in ("axisangle", "euler", "xyaxes", "zaxis"):
                geom.attrib.pop(key, None)
            for key, value in (("pos", model.geom_pos[gid]), ("quat", model.geom_quat[gid])):
                geom.set(key, " ".join(format(float(x), ".17g") for x in value))


def _native_usd_views(usd_path, meta):
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"cannot open native USD: {usd_path}")
    bodies, joints, roots = {}, {}, []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if prim.GetName() in bodies:
                raise ValueError(f"duplicate native USD body {prim.GetName()!r}")
            bodies[prim.GetName()] = prim
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            if prim.GetName() in joints:
                raise ValueError(f"duplicate native USD joint {prim.GetName()!r}")
            joints[prim.GetName()] = prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(prim)
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            ends = (joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets())
            if not ends[0] or not ends[1]:
                raise ValueError("native FREE asset contains a joint to WORLD")
    if set(bodies) != set(meta.body_names) or set(joints) != set(meta.joint_names):
        raise ValueError("native USD body/scalar-joint roster differs from MJCF")
    root = meta.free_roots[0].body_name
    if len(roots) != 1 or roots[0] != bodies[root]:
        raise ValueError("native USD must have its one declared floating articulation root")
    return stage, bodies, joints


def complete_native_usd(usd_path, meta) -> None:
    """Install the native constraints missing from pinned MJCF import, pre-parse."""
    from omni.physx.bindings._physx import (
        JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR,
    )
    from pxr import PhysxSchema, Sdf, UsdPhysics

    stage, _, joints = _native_usd_views(usd_path, meta)
    for i, name in enumerate(meta.joint_names):
        prim = joints[name]
        joint = (UsdPhysics.RevoluteJoint(prim) if prim.IsA(UsdPhysics.RevoluteJoint)
                 else UsdPhysics.PrismaticJoint(prim))
        bounds = meta.joint_range[i] if meta.joint_limited[i] else (-math.inf, math.inf)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            bounds = tuple(math.degrees(float(x)) for x in bounds)
        joint.CreateLowerLimitAttr().Set(float(bounds[0]))
        joint.CreateUpperLimitAttr().Set(float(bounds[1]))
    for fact in native_mimic_imports(meta):
        prim = joints[fact.joint_name]
        reference = joints[fact.reference_joint_name]
        axis = "rot" + UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
        mimic = PhysxSchema.PhysxMimicJointAPI.Apply(prim, axis)
        mimic.CreateReferenceJointRel().SetTargets([reference.GetPath()])
        mimic.CreateReferenceJointAxisAttr().Set(
            "rot" + UsdPhysics.RevoluteJoint(reference).GetAxisAttr().Get()
        )
        mimic.CreateGearingAttr().Set(fact.gearing)
        mimic.CreateOffsetAttr().Set(fact.offset_degrees)
        drive = UsdPhysics.DriveAPI.Apply(prim, UsdPhysics.Tokens.angular)
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
        PhysxSchema.PhysxJointAxisAPI.Apply(prim, UsdPhysics.Tokens.angular)
        prim.CreateAttribute(
            JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR, Sdf.ValueTypeNames.Float,
        ).Set(fact.viscous_friction_per_degree)
    stage.GetRootLayer().Save()
    validate_native_usd(usd_path, meta)


def validate_native_usd(usd_path, meta) -> None:
    """Read-only admission, also used for an externally supplied native USD."""
    import mujoco
    import numpy as np
    from omni.physx.bindings._physx import (
        JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR,
    )
    from pxr import PhysxSchema, UsdPhysics

    source = mujoco.MjModel.from_xml_path(meta.xml_path)
    validate_native_mjcf_import(source)
    _, bodies, joints = _native_usd_views(usd_path, meta)
    expected_mimics = {f.joint_name: f for f in native_mimic_imports(meta)}
    for i, name in enumerate(meta.joint_names):
        prim = joints[name]
        jid = source.joint(name).id
        angular = int(source.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        joint_type = UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint
        if not prim.IsA(joint_type):
            raise ValueError(f"native USD joint type differs for {name!r}")
        joint = joint_type(prim)
        child = meta.joint_body_names[i]
        parent = meta.body_parent_names[meta.body_names.index(child)]
        if (
            joint.GetBody0Rel().GetTargets() != [bodies[parent].GetPath()]
            or joint.GetBody1Rel().GetTargets() != [bodies[child].GetPath()]
        ):
            raise ValueError(f"native USD joint body binding differs for {name!r}")
        bounds = meta.joint_range[i] if meta.joint_limited[i] else (-math.inf, math.inf)
        if angular:
            bounds = tuple(math.degrees(float(x)) for x in bounds)
        if (
            joint.GetLowerLimitAttr().Get() != float(np.float32(bounds[0]))
            or joint.GetUpperLimitAttr().Get() != float(np.float32(bounds[1]))
        ):
            raise ValueError(f"native USD joint limits differ for {name!r}")
        mimics = PhysxSchema.PhysxMimicJointAPI.GetAll(prim)
        drive = UsdPhysics.DriveAPI.Get(prim, "angular" if angular else "linear")
        if name not in expected_mimics:
            if mimics:
                raise ValueError(f"unexpected native USD mimic at {name!r}")
            if name not in meta.driven_joint_names and drive and drive.GetStiffnessAttr().Get() != 0:
                raise ValueError(f"native passive joint {name!r} has a position drive")
            continue
        fact = expected_mimics[name]
        if len(mimics) != 1:
            raise ValueError(f"native USD mimic is absent or ambiguous at {name!r}")
        mimic = mimics[0]
        reference = joints[fact.reference_joint_name]
        if (
            mimic.GetName() != "rot" + joint.GetAxisAttr().Get()
            or mimic.GetReferenceJointAxisAttr().Get()
            != "rot" + UsdPhysics.RevoluteJoint(reference).GetAxisAttr().Get()
            or mimic.GetReferenceJointRel().GetTargets() != [reference.GetPath()]
            or mimic.GetGearingAttr().Get() != float(np.float32(fact.gearing))
            or mimic.GetOffsetAttr().Get() != float(np.float32(fact.offset_degrees))
            or (drive and (drive.GetStiffnessAttr().Get() != 0 or drive.GetDampingAttr().Get() != 0))
            or prim.GetAttribute(JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR).Get()
            != float(np.float32(fact.viscous_friction_per_degree))
        ):
            raise ValueError(f"native USD mimic/property binding differs for {name!r}")


def write_native_mjcf_assets(layout, assets) -> dict[str, str]:
    """Write importer inputs from ONE authoritative composed scene.

    Explicit compiled inertials prevent importer density estimates from changing
    bodies whose original inertial was derived from multiple collision geoms.
    Inline hulls use compiled triangles and geom poses in one declared chart.
    Body transforms and supported passive equalities are retained.
    Mesh files resolve relative to each generated importer XML, with no meshdir.
    This function compiles metadata only; it does not create MjData or step.
    """
    from pathlib import Path

    import mujoco

    source = Path(layout.scene_xml_path).resolve()
    spec = mujoco.MjSpec.from_file(str(source))
    meshdir = source.parent / (spec.meshdir or "")
    for mesh in spec.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str((meshdir / mesh.file).resolve())
    spec.meshdir = ""
    model = spec.compile()
    validate_native_mjcf_import(model)
    actual_roots = {
        model.body(int(model.jnt_bodyid[j])).name
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    }
    if actual_roots != {binding.root_body_name for binding in layout.articulations}:
        raise ValueError("configured native articulations must cover the source FREE roots")
    original = ET.fromstring(spec.to_xml())
    _materialize_native_inline_meshes(
        original, model, Path(assets["static"]).parent / "native_meshes",
    )
    for body in original.findall(".//body"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.get("name"))
        if model.body_mass[bid] <= 0:
            continue
        inertial = body.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(body, "inertial")
        inertial.attrib.clear()
        for key, value in (
            ("pos", model.body_ipos[bid]),
            ("quat", model.body_iquat[bid]),
            ("diaginertia", model.body_inertia[bid]),
        ):
            inertial.set(key, " ".join(format(float(v), ".17g") for v in value))
        inertial.set("mass", format(float(model.body_mass[bid]), ".17g"))
    roots = {b.root_body_name for b in layout.articulations}
    outputs = {}
    selected = [
        (b.entity_name, b.root_body_name, assets["articulations"][b.entity_name]) for b in layout.articulations
    ]
    selected.append(("static", None, assets["static"]))
    for name, root_name, usd in selected:
        document = copy.deepcopy(original)
        world = document.find("worldbody")
        for child in list(world):
            keep = (
                child.tag == "body" and child.get("name") == root_name
                if root_name is not None
                else child.get("name") not in roots
            )
            if not keep:
                world.remove(child)
        joints = {j.get("name") for j in world.iter("joint")}
        # FREE motor channels are replaced by the configured root-wrench action,
        # not imported as fake scalar position drives.
        for section in ("actuator", "equality"):
            element = document.find(section)
            if element is not None:
                for item in list(element):
                    references = [
                        item.get(k)
                        for k in ("joint", "joint1", "joint2")
                        if item.get(k)
                    ]
                    if not references or not all(j in joints for j in references):
                        element.remove(item)
        geoms = {g.get("name") for g in world.iter("geom")}
        contact = document.find("contact")
        if contact is not None:
            for pair in list(contact):
                if pair.get("geom1") not in geoms or pair.get("geom2") not in geoms:
                    contact.remove(pair)
        meshes = {g.get("mesh") for g in world.iter("geom") if g.get("mesh")}
        asset = document.find("asset")
        if asset is not None:
            for mesh in list(asset.findall("mesh")):
                if mesh.get("name") not in meshes:
                    asset.remove(mesh)
        for section in ("sensor", "keyframe"):
            element = document.find(section)
            if element is not None:
                document.remove(element)
        output = Path(usd).with_suffix(".xml")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Isaac's MJCF parser prefixes XML directory + meshdir even for absolute
        # mesh filenames. The original input and generated hulls are absolute;
        # bind both to this XML after pruning its actual retained mesh roster.
        for mesh in document.findall("asset/mesh[@file]"):
            mesh.set("file", os.path.relpath(mesh.get("file"), output.parent))
        ET.ElementTree(document).write(output, encoding="utf-8", xml_declaration=True)
        outputs[name] = str(output)
    return outputs


def convert_native_scene(config_path: str) -> None:
    """Use the pinned Isaac Lab MJCF converter, never the rigid density path."""
    from pathlib import Path

    from isaacsim.core.utils.extensions import enable_extension

    if not enable_extension("isaacsim.asset.importer.mjcf"):
        raise RuntimeError("Isaac native conversion could not enable isaacsim.asset.importer.mjcf")

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from omegaconf import OmegaConf

    from envs.scene.native import NativeSceneCfg
    from simulator.mjcf_meta import load_robot_mjcf_meta

    config = OmegaConf.load(config_path)
    layout = NativeSceneCfg.from_mapping(config.robot.native_scene)
    assets = config.sim.native_assets
    inputs = write_native_mjcf_assets(layout, assets)
    metadata = load_robot_mjcf_meta(layout.scene_xml_path)
    selected = [(b.entity_name, assets.articulations[b.entity_name], False) for b in layout.articulations]
    selected.append(("static", assets.static, True))
    for name, usd, fixed in selected:
        destination = Path(usd)
        MjcfConverter(
            MjcfConverterCfg(
                asset_path=inputs[name],
                usd_dir=str(destination.parent),
                usd_file_name=destination.name,
                force_usd_conversion=True,
                make_instanceable=False,
                fix_base=fixed,
                import_inertia_tensor=True,
                import_sites=True,
                link_density=0.0,
            )
        )
        if not fixed:
            binding = next(b for b in layout.articulations if b.entity_name == name)
            complete_native_usd(destination, metadata.subtree(binding.root_body_name))
        _log(f"converted configured native asset {name}: {destination}")
    # TODO(Isaac gate): qualify installed inertia/collision/filter/material facts
    # and constraint response. USD admission alone is not physical parity.


def object_mass_properties(obj_dir: str):
    """(mass, raw mass, com, inertia) at density 800 with [0.05, 0.5] kg clamp;
    convex-parts fallback when visual.obj is not watertight."""
    import trimesh

    visual = trimesh.load(os.path.join(obj_dir, "visual.obj"), force="mesh")
    solid = visual
    if not visual.is_watertight or visual.volume <= 0.0:
        parts = []
        convex_dir = os.path.join(obj_dir, "convex")
        for f in sorted(os.listdir(convex_dir), key=lambda s: int(os.path.splitext(s)[0])):
            parts.append(trimesh.load(os.path.join(convex_dir, f), force="mesh"))
        solid = trimesh.util.concatenate(parts)
        _log(f"  visual.obj not watertight -> mass props from {len(parts)} convex parts")
    solid.density = OBJ_DENSITY
    mass_raw = float(solid.mass)
    mass = min(max(mass_raw, OBJ_MASS_CLAMP[0]), OBJ_MASS_CLAMP[1])
    scale = mass / mass_raw
    com = solid.center_mass
    inertia = solid.moment_inertia * scale  # about COM, mesh-frame axes
    return mass, mass_raw, com, inertia


def author_object_urdf(
    obj_dir: str, out_dir: str, collision_visual: bool = False
) -> tuple[str, float, int]:
    """Write the single-link URDF for one COACD object. Returns (path, mass, n_hulls).
    collision_visual: render the COACD hulls instead of visual.obj (physics unchanged)."""
    mass, mass_raw, com, inertia = object_mass_properties(obj_dir)
    convex_dir = os.path.join(obj_dir, "convex")
    hulls = sorted(os.listdir(convex_dir), key=lambda s: int(os.path.splitext(s)[0]))

    ixx, iyy, izz = inertia[0, 0], inertia[1, 1], inertia[2, 2]
    ixy, ixz, iyz = inertia[0, 1], inertia[0, 2], inertia[1, 2]
    lines = [
        '<?xml version="1.0"?>',
        f"<!-- Auto-generated by convert_assets.py from {obj_dir}.",
        f"     Inertial: density {OBJ_DENSITY} kg/m^3, raw mass {mass_raw:.6g} kg clamped to"
        f" {mass:.6g} kg (inertia scaled to match). -->",
        f'<robot name="object_{os.path.basename(obj_dir)}">',
        '  <link name="baseLink">',
        "    <inertial>",
        f'      <origin rpy="0 0 0" xyz="{com[0]:.9g} {com[1]:.9g} {com[2]:.9g}"/>',
        f'      <mass value="{mass:.9g}"/>',
        f'      <inertia ixx="{ixx:.9g}" ixy="{ixy:.9g}" ixz="{ixz:.9g}"'
        f' iyy="{iyy:.9g}" iyz="{iyz:.9g}" izz="{izz:.9g}"/>',
        "    </inertial>",
    ]
    if collision_visual:
        # Unique visual names: the URDF importer builds prim paths from them and
        # errors out ("Used null prim") on multiple unnamed visuals.
        for i, hull in enumerate(hulls):
            lines += [
                f'    <visual name="visual_{i}">',
                '      <origin rpy="0 0 0" xyz="0 0 0"/>',
                f'      <geometry><mesh filename="../convex/{hull}" scale="1.0 1.0 1.0"/></geometry>',
                f'      <material name="mat"><color rgba="{COLVIS_RGBA}"/></material>',
                "    </visual>",
            ]
    else:
        lines += [
            "    <visual>",
            '      <origin rpy="0 0 0" xyz="0 0 0"/>',
            '      <geometry><mesh filename="../visual.obj" scale="1.0 1.0 1.0"/></geometry>',
            '      <material name="mat"><color rgba="0.75 0.7 0.6 1"/></material>',
            "    </visual>",
        ]
    for i, hull in enumerate(hulls):
        lines += [
            f'    <collision name="collision_{i}">',
            '      <origin rpy="0 0 0" xyz="0 0 0"/>',
            f'      <geometry><mesh filename="../convex/{hull}" scale="1.0 1.0 1.0"/></geometry>',
            "    </collision>",
        ]
    lines += ["  </link>", "</robot>", ""]

    os.makedirs(out_dir, exist_ok=True)
    urdf_path = os.path.join(out_dir, "object.urdf")
    with open(urdf_path, "w") as f:
        f.write("\n".join(lines))
    return urdf_path, mass, len(hulls)


def convert_object(obj_dir: str, force: bool, collision_visual: bool = False) -> None:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    out_dir = os.path.join(obj_dir, COLVIS_OBJ_DIRNAME if collision_visual else ".isaac_usd")
    usd_path = os.path.join(out_dir, "object.usd")
    sources = [os.path.join(obj_dir, "visual.obj")]
    convex_dir = os.path.join(obj_dir, "convex")
    sources += [os.path.join(convex_dir, f) for f in os.listdir(convex_dir)]
    if not force and _up_to_date(usd_path, sources):
        _log(f"up to date, skipping: {usd_path}")
        return

    urdf_path, mass, n_hulls = author_object_urdf(obj_dir, out_dir, collision_visual)
    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=out_dir,
        usd_file_name="object.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=False,
        link_density=OBJ_DENSITY,
        joint_drive=None,
        collider_type="convex_hull",
    )
    converter = UrdfConverter(cfg)
    _log(f"converted {obj_dir} -> {converter.usd_path} (mass {mass:.4g} kg, {n_hulls} hulls)")
    verify_usd(converter.usd_path)


def author_robot_colvis_urdf(src_urdf: str, out_dir: str) -> str:
    """Visualize the configured URDF's hulls without moving its collision meshes."""
    import trimesh

    urdf_dir = os.path.dirname(src_urdf)
    hull_dir = os.path.join(out_dir, "hull_meshes")
    os.makedirs(hull_dir, exist_ok=True)
    tree = ET.parse(src_urdf)
    n_hulls = 0
    for link in tree.getroot().findall("link"):
        for vis in link.findall("visual"):
            link.remove(vis)
        for i, col in enumerate(link.findall("collision")):
            mesh_el = col.find("geometry/mesh")
            if mesh_el is None:
                raise RuntimeError(f"non-mesh collision in link {link.get('name')!r}")
            src = os.path.normpath(os.path.join(urdf_dir, mesh_el.get("filename")))
            # The generated URDF lives elsewhere. Retain the original collider,
            # including when different links use the same mesh basename.
            mesh_el.set("filename", src)
            stem = f"{link.attrib['name']}_{i}"
            hull_path = os.path.join(hull_dir, f"{stem}.obj")
            if not _up_to_date(hull_path, [src]):
                trimesh.load(src, force="mesh").convex_hull.export(hull_path)
            n_hulls += 1
            # Unique names: the importer builds prim paths from them (see objects).
            vis = ET.SubElement(link, "visual", name=f"visual_{i}_{stem}")
            origin = col.find("origin")
            if origin is not None:
                vis.append(copy.deepcopy(origin))
            geom = ET.SubElement(vis, "geometry")
            ET.SubElement(
                geom, "mesh",
                filename=f"hull_meshes/{stem}.obj",
                scale=mesh_el.get("scale", "1.0 1.0 1.0"),
            )
            mat = ET.SubElement(vis, "material", name="mat_colvis")
            ET.SubElement(mat, "color", rgba=COLVIS_RGBA)
    urdf_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(src_urdf))[0]}_colvis.urdf")
    tree.write(urdf_path)
    _log(f"  authored collision-visual robot URDF ({n_hulls} hulls) -> {urdf_path}")
    return urdf_path


def convert_robot(config_path: str, force: bool, collision_visual: bool = False) -> None:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    from simulator.mjcf_meta import load_robot_mjcf_meta

    asset = load_robot_asset(config_path, collision_visual=collision_visual)
    src_urdf = str(asset.source_urdf_path)
    usd_path = str(asset.usd_path)
    out_dir = str(asset.usd_path.parent)
    meta = load_robot_mjcf_meta(asset.mjcf_path)
    # The import also contains authored fixed frames (for example fingertip
    # sites materialized as URDF links), beyond the MJCF body view.
    expected_links = list(dict.fromkeys(
        [asset.body_name(name) for name in meta.body_names]
        + [link.attrib["name"] for link in ET.parse(src_urdf).getroot().findall("link")]
    ))
    urdf_src = src_urdf
    if collision_visual:
        os.makedirs(out_dir, exist_ok=True)
        urdf_src = author_robot_colvis_urdf(src_urdf, out_dir)
    if not force and _up_to_date(usd_path, [src_urdf]):
        _log(f"up to date, skipping: {usd_path}")
        _block_empty_robot_visual_references(usd_path, urdf_src)
        verify_usd(usd_path, expect_links=expected_links)
        return
    cfg = UrdfConverterCfg(
        asset_path=urdf_src,
        usd_dir=out_dir,
        usd_file_name=asset.usd_path.name,
        force_usd_conversion=True,
        make_instanceable=False,
        # MJCF anchors the wrist chain at worldbody: the 6 wrist joints ARE the
        # floating base, so the URDF "world" root is welded to the stage.
        fix_base=True,
        # Retain the configured fixed link frames in the imported body roster.
        merge_fixed_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            # Neutral drives; the env's IsaacLab actuator cfgs own kp/kv.
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        collider_type="convex_hull",
        self_collision=False,
    )
    converter = UrdfConverter(cfg)
    _log(f"converted robot -> {converter.usd_path}")
    _block_empty_robot_visual_references(converter.usd_path, urdf_src)
    verify_usd(converter.usd_path, expect_links=expected_links)


def _block_empty_robot_visual_references(usd_path: str, urdf_path: str) -> list[str]:
    """Remove generated importer references for URDF links without visuals.

    Isaac Sim 5.1 creates a reference to ``/visuals/<link>`` even for empty
    wrist-chain and tip frames. Root-layer list edits cannot remove arcs inside
    a referenced layer stack. Edit their introducing generated PrimSpecs instead;
    physical links, joints, tips, collisions and real visual assets stay intact.
    """
    from pathlib import Path

    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(usd_path)
    root_path = stage.GetDefaultPrim().GetPath()
    output_dir = Path(usd_path).resolve().parent
    blocked = []
    edits = {}
    layers = {}
    for link in ET.parse(urdf_path).getroot().findall("link"):
        if link.find("visual") is not None:
            continue
        path = root_path.AppendChild(link.attrib["name"]).AppendChild("visuals")
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        changed = False
        for spec in prim.GetPrimStack():
            layer = spec.layer
            if (layer.realPath and Path(layer.realPath).resolve().is_relative_to(output_dir)
                    and spec.HasInfo("references")):
                edits[(layer.identifier, str(spec.path))] = spec
                layers[layer.identifier] = layer
                changed = True
        if changed:
            blocked.append(str(path))
    # Collect USD opinions first; query no composed prims during Sdf changes.
    with Sdf.ChangeBlock():
        for spec in edits.values():
            spec.ClearInfo("references")
    for identifier in sorted(layers):
        layers[identifier].Save()
    _log(f"removed {len(blocked)} empty URDF visual reference paths in {len(layers)} generated layers")
    return blocked


def verify_usd(usd_path: str, expect_links: list[str] | None = None) -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"cannot open generated USD: {usd_path}")
    names, n_rigid, n_coll, masses = [], 0, 0, []
    # TraverseInstanceProxies: the importer parks meshes behind reference arcs
    # that plain Traverse() skips (they compose as instance proxies).
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        names.append(prim.GetName())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            n_rigid += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            n_coll += 1
        if prim.HasAPI(UsdPhysics.MassAPI):
            attr = UsdPhysics.MassAPI(prim).GetMassAttr()
            if attr and attr.HasAuthoredValue():
                masses.append((prim.GetName(), float(attr.Get())))
    _log(f"  verify {os.path.basename(usd_path)}: rigid-body prims={n_rigid}, "
         f"collision prims={n_coll}, authored masses={masses[:4]}{'...' if len(masses) > 4 else ''}")
    for link in expect_links or []:
        if link not in names:
            raise RuntimeError(f"expected link {link!r} missing from {usd_path} prims")
    if n_coll == 0 and expect_links is None:
        raise RuntimeError(f"no collision prims in {usd_path}")


def main() -> None:
    args = parse_args()
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    _ = app_launcher.app  # boot kit; converters need a live stage context

    if args.objects and not args.data_root:
        raise SystemExit("--objects needs --data-root")
    # A relative root breaks the importer's sublayer paths (@/configuration/...): empty USDs, then a hang.
    args.data_root = args.data_root and os.path.abspath(args.data_root)
    for obj_id in args.objects:
        obj_dir = os.path.join(args.data_root, "objects", obj_id)
        if not os.path.isdir(obj_dir):
            raise FileNotFoundError(f"object dir not found: {obj_dir}")
        convert_object(obj_dir, force=args.force, collision_visual=args.collision_visual)
    if args.robot_config:
        convert_robot(args.robot_config, force=args.force, collision_visual=args.collision_visual)
    if args.native_scene_config:
        convert_native_scene(args.native_scene_config)

    _log("all conversions done")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # simulation_app.close() segfaults (isaacsim 5.1 shutdown bug)


if __name__ == "__main__":
    main()
