"""Isaac USD preparation and launch routing without a Kit or CUDA context."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from simulator.isaacsim.convert_assets import _block_empty_robot_visual_references
from simulator.runtime import launch_runtime


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
