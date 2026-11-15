"""USD preparation and launch routing; the real PhysX case needs SDK bootstrap."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from simulator.isaacsim.convert_assets import _block_empty_robot_visual_references
from simulator.runtime import launch_runtime


@pytest.fixture
def native_usd_completion(monkeypatch):
    import pxr

    from simulator.isaacsim import convert_assets

    # Only the unchanged PhysX mimic/friction imports need injection. All
    # stages, composition, schemas, removals and file reloads below are real USD.
    monkeypatch.setattr(pxr, "PhysxSchema", ModuleType("PhysxSchema"), raising=False)
    for name in ("omni", "omni.physx", "omni.physx.bindings"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    plugin = ModuleType("omni.physx.bindings._physx")
    plugin.JOINT_AXIS_API = "PhysxJointAxisAPI"
    plugin.JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR = "unused:angularFriction"
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    # The existing scalar/mimic/inertia validator is not this regression's
    # subject. Keep its actual strict topology admission at the final boundary.
    checked = []
    def validate(path, meta):
        checked.append((str(path), meta))
        convert_assets._native_usd_views(path, meta)
    monkeypatch.setattr(convert_assets, "validate_native_usd", validate)
    return convert_assets, checked


def _native_root_composition(tmp_path):
    imported = Usd.Stage.CreateNew(str(tmp_path / "imported.usda"))
    asset = UsdGeom.Xform.Define(imported, "/Asset").GetPrim()
    imported.SetDefaultPrim(asset)
    UsdGeom.Xform.Define(imported, "/Asset/mechanics")
    base = UsdGeom.Xform.Define(imported, "/Asset/mechanics/base").GetPrim()
    tip = UsdGeom.Xform.Define(imported, "/Asset/mechanics/tip").GetPrim()
    for body in (base, tip):
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(.25)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    base.AddAppliedSchema("PhysxArticulationAPI")
    scaffold = UsdGeom.Xform.Define(imported, "/Asset/scaffold").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(scaffold)
    scaffold.AddAppliedSchema("PhysxArticulationAPI")
    UsdPhysics.FilteredPairsAPI.Apply(scaffold)
    UsdGeom.Xformable(scaffold).AddTranslateOp().Set((.1, .2, .3))
    scaffold.CreateAttribute("physxArticulation:articulationEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    collider = UsdGeom.Cube.Define(imported, "/Asset/mechanics/tip/collision").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    material = UsdShade.Material.Define(imported, "/Asset/material")
    UsdShade.MaterialBindingAPI.Apply(collider).Bind(material)
    joint = UsdPhysics.RevoluteJoint.Define(imported, "/Asset/joints/hinge")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([tip.GetPath()])
    joint.CreateLowerLimitAttr().Set(math.degrees(-.2))
    joint.CreateUpperLimitAttr().Set(math.degrees(.5))
    imported.GetRootLayer().Save()

    path = tmp_path / "native.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = stage.DefinePrim("/Asset")
    root.GetReferences().AddReference(str(tmp_path / "imported.usda"))
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()
    meta = SimpleNamespace(body_names=["base", "tip"], joint_names=["hinge"],
        free_roots=[SimpleNamespace(body_name="base")], joint_limited=[True],
        joint_range=[(-.2, .5)], mimics=[])
    return path, stage, imported, meta


def test_native_completion_removes_only_disconnected_root_APIs_in_composed_USD(
    tmp_path, native_usd_completion,
):
    convert_assets, checked = native_usd_completion
    path, stage, imported, meta = _native_root_composition(tmp_path)
    scaffold_path = "/Asset/scaffold"
    source_before = imported.GetRootLayer().ExportToString()

    def facts():
        result = {}
        for prim in stage.Traverse():
            schemas = list(prim.GetAppliedSchemas())
            if str(prim.GetPath()) == scaffold_path:
                schemas = [name for name in schemas if name not in {
                    "PhysicsArticulationRootAPI", "PhysxArticulationAPI"}]
            result[str(prim.GetPath())] = {
                "type": prim.GetTypeName(), "APIs": schemas,
                "attributes": {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                               if a.HasAuthoredValueOpinion()},
                "relationships": {r.GetName(): [str(p) for p in r.GetTargets()]
                                  for r in prim.GetRelationships()},
            }
        return result

    before = facts()
    with pytest.raises(ValueError, match="one declared floating articulation root"):
        convert_assets._native_usd_views(path, meta)
    convert_assets.complete_native_usd(path, meta)
    assert checked == [(str(path), meta)]
    stage.Reload()
    _, bodies, joints = convert_assets._native_usd_views(path, meta)
    assert set(bodies) == {"base", "tip"} and set(joints) == {"hinge"}
    assert stage.GetPrimAtPath(scaffold_path)
    assert not stage.GetPrimAtPath(scaffold_path).HasAPI(UsdPhysics.ArticulationRootAPI)
    deletion = stage.GetRootLayer().GetPrimAtPath(scaffold_path).GetInfo("apiSchemas")
    assert set(deletion.deletedItems) == {"PhysicsArticulationRootAPI", "PhysxArticulationAPI"}
    assert bodies["base"].HasAPI(UsdPhysics.ArticulationRootAPI)
    assert facts() == before
    assert imported.GetRootLayer().ExportToString() == source_before


@pytest.mark.parametrize("extra", ["rigid-body", "body-container", "joint-container", "joint-endpoint", "missing-root"])
def test_native_completion_never_erases_mechanical_root_authority(
    tmp_path, native_usd_completion, extra,
):
    convert_assets, checked = native_usd_completion
    path, stage, _, meta = _native_root_composition(tmp_path)
    if extra == "missing-root":
        stage.GetPrimAtPath("/Asset/mechanics/base").RemoveAPI(UsdPhysics.ArticulationRootAPI)
    elif extra == "joint-endpoint":
        joint = UsdPhysics.FixedJoint.Define(stage, "/Asset/connected")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/mechanics/base")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/scaffold")])
    else:
        protected = {"rigid-body": "/Asset/mechanics/tip",
                     "body-container": "/Asset/mechanics", "joint-container": "/Asset/joints"}[extra]
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(protected))
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="one declared floating articulation root"):
        convert_assets.complete_native_usd(path, meta)
    assert checked == []
    if extra == "joint-endpoint":
        assert stage.GetPrimAtPath("/Asset/scaffold").HasAPI(UsdPhysics.ArticulationRootAPI)
    elif extra != "missing-root":
        assert stage.GetPrimAtPath(protected).HasAPI(UsdPhysics.ArticulationRootAPI)


def test_native_completion_authors_and_requires_codeless_angular_viscosity(tmp_path):
    # Run this owning case after the ordinary public Isaac runtime bootstrap.
    # No generated PhysxJointAxisAPI class, schema stub, converter or step is used.
    PhysxSchema = pytest.importorskip("pxr.PhysxSchema")
    physx = pytest.importorskip("omni.physx.bindings._physx")

    from simulator.isaacsim.convert_assets import (
        complete_native_usd,
        validate_native_usd,
    )
    from simulator.mjcf_meta import load_robot_mjcf_meta

    xml = tmp_path / "affine.xml"
    xml.write_text("""<mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="base">
          <freejoint name="root"/>
          <geom type="sphere" size=".02" mass=".2"/>
          <body name="driver_link" pos=".04 0 0">
            <joint name="driver" axis="0 0 1" limited="true"
                   range="-1 1" damping=".02"/>
            <geom type="sphere" size=".01" mass=".1"/>
            <body name="follower_link" pos=".04 0 0">
              <joint name="follower" axis="0 0 1" limited="true"
                     range="-1 1" damping=".3"/>
              <geom type="sphere" size=".01" mass=".1"/>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="motor" joint="driver" kp="2"
                  ctrlrange="-1 1" forcerange="-2 2"/>
      </actuator>
      <equality>
        <joint name="affine" joint1="follower" joint2="driver"
               polycoef=".1 .5 0 0 0"/>
      </equality>
    </mujoco>""")
    meta = load_robot_mjcf_meta(xml).subtree("base")
    path = tmp_path / "affine.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    bodies = {name: UsdGeom.Xform.Define(stage, "/Asset/" + name).GetPrim()
              for name in meta.body_names}
    for prim in bodies.values():
        UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.ArticulationRootAPI.Apply(bodies["base"])
    for name, child in zip(meta.joint_names, meta.joint_body_names, strict=True):
        parent = meta.body_parent_names[meta.body_names.index(child)]
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joints/" + name)
        joint.CreateAxisAttr(UsdPhysics.Tokens.z)
        joint.CreateBody0Rel().SetTargets([bodies[parent].GetPath()])
        joint.CreateBody1Rel().SetTargets([bodies[child].GetPath()])
    follower = stage.GetPrimAtPath("/Asset/joints/follower")
    drive = UsdPhysics.DriveAPI.Apply(follower, UsdPhysics.Tokens.angular)
    drive.CreateStiffnessAttr().Set(4.0)
    drive.CreateDampingAttr().Set(5.0)
    stage.GetRootLayer().Save()

    complete_native_usd(path, meta)
    stage.Reload()
    follower = stage.GetPrimAtPath("/Asset/joints/follower")
    assert follower.HasAPI(physx.JOINT_AXIS_API, UsdPhysics.Tokens.angular)
    assert follower.GetAttribute(
        physx.JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR,
    ).Get() == pytest.approx(.3 * math.pi / 180)
    drive = UsdPhysics.DriveAPI.Get(follower, UsdPhysics.Tokens.angular)
    assert drive.GetStiffnessAttr().Get() == 0
    assert drive.GetDampingAttr().Get() == 0
    mimic, = PhysxSchema.PhysxMimicJointAPI.GetAll(follower)
    assert follower.HasAPI(PhysxSchema.PhysxMimicJointAPI, "rotZ")
    assert mimic.GetReferenceJointRel().GetTargets() == [Sdf.Path("/Asset/joints/driver")]
    assert mimic.GetReferenceJointAxisAttr().Get() == "rotZ"
    assert mimic.GetGearingAttr().Get() == -.5
    assert mimic.GetOffsetAttr().Get() == pytest.approx(-math.degrees(.1))
    for attribute in (physx.JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR,
                      physx.JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR):
        assert not follower.GetAttribute(attribute).HasAuthoredValueOpinion()

    # A matching attribute alone must not pass: the pinned parser ignores it
    # unless this specific codeless angular API instance is actually applied.
    assert follower.RemoveAPI(physx.JOINT_AXIS_API, UsdPhysics.Tokens.angular)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="mimic/property binding differs"):
        validate_native_usd(path, meta)


def test_empty_visual_reference_repair_preserves_physical_and_real_visual_prims(tmp_path):
    geometry = Usd.Stage.CreateNew(str(tmp_path / "geometry.usda"))
    UsdGeom.Xform.Define(geometry, "/visuals")
    UsdGeom.Cube.Define(geometry, "/visuals/hand/mesh")
    geometry.GetRootLayer().Save()

    # The bad references live in a weaker importer layer, as in the real USD.
    imported = Usd.Stage.CreateNew(str(tmp_path / "imported.usda"))
    robot = UsdGeom.Xform.Define(imported, "/Robot").GetPrim()
    imported.SetDefaultPrim(robot)
    hand = UsdGeom.Xform.Define(imported, "/Robot/hand").GetPrim()
    tip = UsdGeom.Xform.Define(imported, "/Robot/tip").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(hand)
    UsdPhysics.RigidBodyAPI.Apply(tip)
    UsdPhysics.MassAPI.Apply(hand).CreateMassAttr(0.25)
    collider = UsdGeom.Sphere.Define(imported, "/Robot/hand/collisions/sphere")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    joint = UsdPhysics.FixedJoint.Define(imported, "/Robot/tip_joint")
    joint.CreateBody0Rel().SetTargets([hand.GetPath()])
    joint.CreateBody1Rel().SetTargets([tip.GetPath()])
    visuals = UsdGeom.Xform.Define(imported, "/Robot/hand/visuals").GetPrim()
    visuals.GetReferences().AddReference(str(tmp_path / "geometry.usda"), Sdf.Path("/visuals/hand"))
    empty = UsdGeom.Xform.Define(imported, "/Robot/tip/visuals").GetPrim()
    empty.GetReferences().AddReference(str(tmp_path / "geometry.usda"), Sdf.Path("/visuals/tip"))
    imported.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    root = stage.DefinePrim("/Robot")
    root.GetReferences().AddReference(str(tmp_path / "imported.usda"))
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()
    bad_reference = Sdf.Reference(str(tmp_path / "geometry.usda"), Sdf.Path("/visuals/tip"))
    assert stage.GetPrimAtPath("/Robot/tip/visuals").GetMetadata("references").prependedItems == [
        bad_reference
    ]
    assert not geometry.GetPrimAtPath(bad_reference.primPath)
    def facts():
        return {
            str(prim.GetPath()): {
                "type": prim.GetTypeName(), "APIs": list(prim.GetAppliedSchemas()),
                "attributes": {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                               if a.GetName().startswith(("physics:", "physx:"))},
                "relationships": {r.GetName(): [str(p) for p in r.GetTargets()]
                                  for r in prim.GetRelationships()
                                  if r.GetName().startswith(("physics:", "physx:"))},
            }
            for prim in stage.Traverse()
            if not str(prim.GetPath()).startswith("/Robot/tip/visuals")
        }

    before = facts()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="Robot"><link name="hand"><visual/></link><link name="tip"/></robot>')

    assert _block_empty_robot_visual_references(str(tmp_path / "robot.usda"), str(urdf)) == [
        "/Robot/tip/visuals"
    ]
    stage.Reload()
    assert not imported.GetRootLayer().GetPrimAtPath("/Robot/tip/visuals").HasInfo("references")
    assert not stage.GetPrimAtPath("/Robot/tip/visuals").HasAuthoredMetadata("references")
    assert facts() == before
    assert stage.GetPrimAtPath("/Robot/hand/visuals/mesh").IsA(UsdGeom.Cube)
    assert stage.GetPrimAtPath("/Robot/hand/collisions/sphere").HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetPrimAtPath("/Robot/tip").HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.FixedJoint(stage.GetPrimAtPath("/Robot/tip_joint")).GetBody1Rel().GetTargets() == [
        Sdf.Path("/Robot/tip")
    ]


def test_nested_robot_reference_clones_remove_introducing_arcs_not_just_metadata(tmp_path, capfd):
    output = tmp_path / "converted"
    output.mkdir()
    geometry = Usd.Stage.CreateNew(str(tmp_path / "source-geometry.usda"))
    UsdGeom.Xform.Define(geometry, "/visuals")
    UsdGeom.Cube.Define(geometry, "/visuals/hand/mesh")
    geometry.GetRootLayer().Save()
    imported = Usd.Stage.CreateNew(str(output / "base.usda"))
    root = UsdGeom.Xform.Define(imported, "/Robot").GetPrim()
    imported.SetDefaultPrim(root)
    for name in ("hand", "tip"):
        prim = UsdGeom.Xform.Define(imported, "/Robot/" + name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        visual = UsdGeom.Xform.Define(imported, "/Robot/" + name + "/visuals").GetPrim()
        visual.GetReferences().AddReference(str(tmp_path / "source-geometry.usda"),
                                           Sdf.Path("/visuals/" + name))
    imported.GetRootLayer().Save()
    robot_path = output / "robot.usda"
    robot = Usd.Stage.CreateNew(str(robot_path))
    root = robot.DefinePrim("/Robot")
    root.GetReferences().AddReference(str(output / "base.usda"))
    robot.SetDefaultPrim(root)
    # The superseded repair changes resolved metadata but cannot erase an arc
    # introduced across this reference boundary (OpenUSD encapsulation).
    robot.GetPrimAtPath("/Robot/tip/visuals").GetReferences().SetReferences([])
    robot.GetRootLayer().Save()

    def clone_scene():
        scene = Usd.Stage.CreateInMemory()
        for index in range(2):
            scene.DefinePrim(f"/World/envs/env_{index}/Robot").GetReferences().AddReference(str(robot_path))
        return scene

    capfd.readouterr()
    before = clone_scene()
    assert "Unresolved reference prim path" in capfd.readouterr().err
    assert before.GetPrimAtPath("/World/envs/env_0/Robot/tip/visuals").GetMetadata("references").isExplicit
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="Robot"><link name="hand"><visual/></link><link name="tip"/></robot>')
    source_before = geometry.GetRootLayer().ExportToString()
    _block_empty_robot_visual_references(str(robot_path), str(urdf))
    assert not imported.GetRootLayer().GetPrimAtPath("/Robot/tip/visuals").HasInfo("references")
    assert geometry.GetRootLayer().ExportToString() == source_before
    robot.Reload()
    capfd.readouterr()
    after = clone_scene()
    assert "Unresolved reference prim path" not in capfd.readouterr().err
    for index in range(2):
        prefix = f"/World/envs/env_{index}/Robot"
        assert after.GetPrimAtPath(prefix + "/tip").HasAPI(UsdPhysics.RigidBodyAPI)
        assert after.GetPrimAtPath(prefix + "/hand/visuals/mesh").IsA(UsdGeom.Cube)
    assert _block_empty_robot_visual_references(str(robot_path), str(urdf)) == []


@pytest.mark.parametrize(
    "overrides,device,headless,kit_args",
    [
        ([], "cuda:0", True, ""),
        (["device=cuda:0", "headless=false", 'sim.kit_args="--/renderer/activeGpu=1"'],
         "cuda:0", False, "--/renderer/activeGpu=1"),
    ],
)
def test_isaac_runtime_forwards_the_selected_device_before_boot(
    monkeypatch, overrides, device, headless, kit_args,
):
    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = Mock()
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None,
    ):
        config = compose(config_name="flashdexretarget", overrides=overrides)

    runtime = launch_runtime(config)

    app_module.AppLauncher.assert_called_once_with({
        "headless": headless, "device": device, "multi_gpu": False, "kit_args": kit_args,
    })
    assert runtime.launcher is app_module.AppLauncher.return_value


@pytest.mark.parametrize("accepted", ["YES", None])
def test_kinematic_converter_forwards_configured_kit_args_and_operator_acceptance(
    tmp_path, monkeypatch, accepted,
):
    from retarget import kinematic_retargeting as pipeline

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "motion.pt").touch()
    if accepted is None:
        monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    else:
        monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", accepted)
    monkeypatch.setattr(pipeline.torch, "load", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(pipeline, "object_meshes", lambda *_args: [
        (dataset / "objects" / "S10001" / "visual.obj", 1),
    ])
    run = Mock()
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config" / "retarget"),
        version_base=None,
    ):
        config = compose(config_name="kinematic_retargeting", overrides=[
            "dataset=" + str(dataset), "steps.convex=false", "steps.retarget=false",
            "steps.pack=false", "steps.sdf=false", "usd_gpu=GPU-selected-by-operator",
            'usd_kit_args="--/renderer/activeGpu=1 --/plugins/carb.tasking.plugin/threadCount=8"',
        ])
    pipeline.main.__wrapped__(config)

    run.assert_called_once()
    arguments = run.call_args.args[0]
    assert arguments[1] == str(
        Path(pipeline.__file__).resolve().parents[1] / "simulator/isaacsim/convert_assets.py"
    )
    assert arguments[2:] == [
        "--data-root", str(dataset), "--objects", "S10001",
        "--kit_args=--/renderer/activeGpu=1 --/plugins/carb.tasking.plugin/threadCount=8",
    ]
    assert run.call_args.kwargs["check"] is True
    environment = run.call_args.kwargs["env"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-selected-by-operator"
    if accepted is None:
        assert "OMNI_KIT_ACCEPT_EULA" not in environment
    else:
        assert environment["OMNI_KIT_ACCEPT_EULA"] == accepted
