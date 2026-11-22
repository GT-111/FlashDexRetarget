"""Stock USD installation ordering through the real pre-physics scene builder."""

from __future__ import annotations

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from test_scene_contact_authority import _isaac_scene_type


class _Config(SimpleNamespace):
    def replace(self, **changes):
        return type(self)(**{**vars(self), **changes})


class _Prim:
    def __init__(self, path, *, instanceable=False, source=None, scale=None):
        self.path = path
        self.instanceable = instanceable
        self.source = source
        self.scale = scale
        self.properties = {"collision_enabled": True, "contact_offset": 0.02, "rest_offset": -0.001}
        self.material = None

    def SetInstanceable(self, value):
        self.instanceable = value

    def IsInstanceable(self):
        return self.instanceable

    def GetPath(self):
        return self.path

    def GetMetadata(self, key):
        assert key == "references"
        return SimpleNamespace(prependedItems=[SimpleNamespace(assetPath=self.source)])

    def GetAttribute(self, key):
        assert key == "xformOp:scale"
        return SimpleNamespace(Get=lambda: self.scale)


class _Stage:
    def __init__(self):
        self.prims = {}
        self.events = []

    def GetPrimAtPath(self, path):
        return self.prims[str(path)]

    def subtree(self, root):
        return [prim for path, prim in self.prims.items()
                if path == root.path or path.startswith(root.path + "/")]

    def spawn(self, path, cfg, **kwargs):
        self.events.append(("spawn", path, cfg.collision_props))
        self.prims[path] = _Prim(path, source=cfg.usd_path, scale=getattr(cfg, "scale", None))
        nested = path + "/body/collisions"
        self.prims[nested] = _Prim(nested, instanceable=True)
        self.prims[nested + "/shape"] = _Prim(nested + "/shape")
        if cfg.collision_props is not None:
            self.apply_properties(path, cfg.collision_props)

    def apply_properties(self, path, cfg):
        subtree = self.subtree(self.prims[path])
        if any(prim.instanceable for prim in subtree):
            self.events.append(("skipped_instance", path, cfg))
            return False
        self.events.append(("properties", path, cfg))
        for prim in subtree:
            if prim.path.endswith("/shape"):
                prim.properties.update({name: value for name, value in vars(cfg).items() if value is not None})
        return True

    def bind_material(self, path, material):
        assert not any(prim.instanceable for prim in self.subtree(self.prims[path]))
        self.events.append(("material", path, material))
        for prim in self.subtree(self.prims[path]):
            if prim.path.endswith("/shape"):
                prim.material = material

    def clone(self, worlds):
        for world in range(1, worlds):
            for path, prim in list(self.prims.items()):
                if path.startswith("/World/envs/env_0/"):
                    clone_path = path.replace("/env_0/", f"/env_{world}/", 1)
                    clone = copy.deepcopy(prim)
                    clone.path = clone_path
                    self.prims[clone_path] = clone


@pytest.fixture
def collision_scene(monkeypatch):
    scene_type = _isaac_scene_type(monkeypatch)
    module = sys.modules[scene_type.__module__]
    stage = _Stage()
    usd = SimpleNamespace(PrimRange=lambda root, predicate: stage.subtree(root),
                          TraverseInstanceProxies=lambda: None)
    pxr = ModuleType("pxr")
    pxr.Usd = usd
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    utils = ModuleType("isaaclab.sim.utils")
    utils.bind_physics_material = stage.bind_material
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", utils)
    sys.modules["isaaclab.sim"].utils = utils
    stage_utils = ModuleType("isaacsim.core.utils.stage")
    stage_utils.get_current_stage = lambda: stage
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.stage", stage_utils)
    sys.modules["isaacsim"].core = sys.modules["isaacsim.core"]
    sys.modules["isaacsim.core"].utils = sys.modules["isaacsim.core.utils"]
    sys.modules["isaacsim.core.utils"].stage = stage_utils

    def usd_config(**kwargs):
        return _Config(collision_props=None, **kwargs) if "collision_props" not in kwargs else _Config(**kwargs)

    sim_utils = SimpleNamespace(
        UsdFileCfg=lambda **kwargs: usd_config(func=stage.spawn, **kwargs),
        RigidBodyPropertiesCfg=lambda **kwargs: _Config(**kwargs),
        CollisionPropertiesCfg=lambda **kwargs: _Config(**kwargs),
        ArticulationRootPropertiesCfg=lambda **kwargs: _Config(**kwargs),
        RigidBodyMaterialCfg=lambda **kwargs: _Config(func=lambda *args: None, **kwargs),
        DomeLightCfg=lambda **kwargs: _Config(func=lambda *args: None, **kwargs),
        modify_collision_properties=stage.apply_properties,
    )
    monkeypatch.setattr(module, "sim_utils", sim_utils)
    for name in ("ArticulationCfg", "RigidObjectCfg"):
        config_type = type(name, (_Config,), {"InitialStateCfg": _Config})
        monkeypatch.setattr(module, name, config_type)
    monkeypatch.setattr(module, "ImplicitActuatorCfg", lambda **kwargs: _Config(**kwargs))

    def create_asset(cfg):
        if cfg.spawn is not None:
            cfg.spawn.func(cfg.prim_path.replace("env_.*", "env_0"), cfg.spawn)
        return _Config(cfg=cfg)

    monkeypatch.setattr(module, "Articulation", create_asset)
    monkeypatch.setattr(module, "RigidObject", create_asset)

    class Scene:
        def __init__(self, cfg):
            self.cfg = cfg
            self.articulations, self.rigid_objects, self.sensors = {}, {}, {}

        def clone_environments(self, **kwargs):
            stage.clone(self.cfg.num_envs)

        def filter_collisions(self, **kwargs):
            pass

    monkeypatch.setattr(module, "InteractiveScene", Scene)
    monkeypatch.setattr(module, "InteractiveSceneCfg", lambda **kwargs: _Config(**kwargs))
    monkeypatch.setenv("FDR_ABL_NOGROUND", "1")
    return module, stage


class _BeforeFirstPhysicsReset(Exception):
    pass


@pytest.mark.parametrize("per_env", [False, True])
@pytest.mark.parametrize("collision_enabled", [False, True])
@pytest.mark.parametrize("offsets", [(0.005, 0.001), (None, None)])
def test_stock_scene_installs_selected_properties_after_deinstance_before_first_reset(
    collision_scene, monkeypatch, tmp_path, per_env, collision_enabled, offsets,
):
    module, stage = collision_scene
    monkeypatch.setenv("FDR_ABL_NOOBJCOL", "0" if collision_enabled else "1")
    asset_path = tmp_path / "selected.usd"
    asset_path.touch()
    asset = _Config(usd_path=asset_path, body_name=lambda name: name)
    monkeypatch.setattr(module, "resolve_robot_asset", lambda *args, **kwargs: asset)
    meta = _Config(
        joint_names=["selected_joint"], actuator_names=["selected_motor"],
        actuator_joint={"selected_motor": "selected_joint"},
        actuator_forcerange=[[-3.0, 3.0]], joint_damping=[0.1],
        joint_armature=[0.01], joint_frictionloss=[0.0],
        kp_kv_for_joint=lambda name: (20.0, 2.0), body_names=[],
    )
    cfg = OmegaConf.create({
        "headless": True,
        "robot": {"self_collision": False, "entity_name": "robot", "contact_links": {}},
        "scene": {"num_envs": 2, "env_spacing": 1.0, "sensors": []},
        "commands": {"object": {"support_disks": {"enable": False}}},
        "sim": {"physx": {"solver_position_iteration_count": 8,
                           "solver_velocity_iteration_count": 0,
                           "contact_offset": offsets[0], "rest_offset": offsets[1]}},
    })
    if per_env:
        slots = [
            {"usd_path": {"left": "left0.usd", "right": "right0.usd"},
             "scale": {"left": 1.0, "right": 1.0}},
            {"usd_path": {"left": "left1.usd", "right": "right1.usd"},
             "scale": {"left": 1.0, "right": 1.0}, "ghost": ("left",)},
        ]
        assigned = torch.tensor([0, 1])
    else:
        slots = [{"entity_name": "object_selected", "usd_path": "parked.usd"}]
        assigned = None

    def reset():
        stage.events.append(("physics_reset",))
        raise _BeforeFirstPhysicsReset

    with pytest.raises(_BeforeFirstPhysicsReset):
        module.build_isaac_scene(cfg, SimpleNamespace(reset=reset), "cpu", meta, slots, assigned)
    assert stage.events[-1] == ("physics_reset",)
    assert not any(event[0] == "skipped_instance" for event in stage.events)
    assert all(event[2] is None for event in stage.events if event[0] == "spawn")
    shapes = [prim for prim in stage.prims.values() if prim.path.endswith("/shape")]
    assert len(shapes) == (6 if per_env else 4)
    for shape in shapes:
        robot = "/Robot/" in shape.path
        assert shape.properties == {"collision_enabled": True if robot else collision_enabled,
                                    "contact_offset": 0.02 if offsets[0] is None else offsets[0],
                                    "rest_offset": -0.001 if offsets[1] is None else offsets[1]}
        expected_material = "/World/Materials/handContact" if robot else "/World/Materials/objContact"
        assert shape.material == expected_material
    for i, event in enumerate(stage.events):
        if event[0] == "material":
            assert stage.events[i - 1][:2] == ("properties", event[1])


def test_unselected_collision_properties_and_none_fields_keep_authored_values(collision_scene):
    module, stage = collision_scene
    for path, props in (("/World/no_override", None),
                        ("/World/partial", _Config(collision_enabled=False, contact_offset=None, rest_offset=None))):
        stage.spawn(path, _Config(usd_path="selected.usd", collision_props=None))
        module._apply_contact_material(stage, path, "/World/Materials/selected", props)
        shape = stage.prims[path + "/body/collisions/shape"]
        assert shape.properties["contact_offset"] == 0.02
        assert shape.properties["rest_offset"] == -0.001
        assert shape.properties["collision_enabled"] is (props is None)
        assert shape.material == "/World/Materials/selected"
    assert len([event for event in stage.events if event[0] == "properties"]) == 1
