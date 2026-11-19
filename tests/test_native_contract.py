"""Native topology/action contracts; no Isaac app, MjData, FK or simulation."""

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from envs.actions.native import NativeResidualAction, NativeResidualActionCfg
from envs.scene.native import NativeSceneCfg
from simulator.isaacsim.convert_assets import (
    native_mimic_imports,
    validate_native_mjcf_import,
    write_native_mjcf_assets,
)
from simulator.isaacsim.entity import FreeArticulationAdapter, _FreshnessClock
from simulator.mjcf_meta import load_robot_mjcf_meta


def _scene(tmp_path):
    source = tmp_path / "source.xml"
    source.write_text("""<mujoco><compiler angle="radian"/>
    <worldbody>
      <geom name="support" type="box" size=".1 .1 .02"/>
      <body name="left_root"><freejoint name="left_free"/>
        <geom type="box" size=".02 .02 .02" mass=".2"/>
        <body name="left_driver"><joint name="left_joint" range="0 1" actuatorfrcrange="-.8 .8"/>
          <geom type="box" size=".01 .01 .01" mass=".05"/>
          <body name="left_follower"><joint name="left_mimic" range="0 .6"/>
            <geom type="box" size=".01 .01 .01" mass=".05"/>
            <site name="left_tip" pos="0 0 .02"/>
          </body>
        </body>
      </body>
      <body name="right_root"><freejoint name="right_free"/>
        <geom type="box" size=".02 .02 .02" mass=".2"/>
        <body name="right_driver"><joint name="right_joint" range="0 1" actuatorfrcrange="-1 1"/>
          <geom type="box" size=".01 .01 .01" mass=".1"/>
          <site name="right_tip"/>
        </body>
        <body name="right_second"><joint name="right_other" range="0 2" actuatorfrcrange="-1 1"/>
          <geom type="box" size=".01 .01 .01" mass=".1"/>
        </body>
      </body>
      <body name="base"><freejoint name="base_free"/>
        <geom type="box" size=".1 .1 .01" mass=".075"/>
        <body name="lid"><joint name="hinge" limited="false" damping=".01" armature=".001"/>
          <geom type="box" size=".1 .1 .01" mass=".025"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <position name="command_A" joint="left_joint" kp="20" ctrllimited="false"/>
      <position name="command_B" joint="right_joint" kp="20" ctrllimited="false"/>
      <position name="command_C" joint="right_other" kp="20" ctrllimited="false"/>
    </actuator>
    <equality><joint name="mimic" joint1="left_mimic" joint2="left_joint" polycoef="0 1.5 0 0 0"/></equality>
    </mujoco>""")
    control = dict(
        position_stiffness=400.0,
        position_damping=40.0,
        rotation_stiffness=400.0,
        rotation_damping=40.0,
        force_limit_n=10.0,
        torque_limit_nm=1.0,
        gravity_compensation=1.0,
    )
    mapping = dict(
        scene_xml_path=str(source),
        hands=[
            dict(
                side=side,
                entity_name=f"hand_{side}",
                root_body_name=f"{side}_root",
                root_joint_name=f"{side}_free",
                command_actuator_names=commands,
                root_control=control,
            )
            for side, commands in (
                ("left", ["command_A"]),
                ("right", ["command_B", "command_C"]),
            )
        ],
        objects=(dict(
            entity_name="object",
            root_body_name="base",
            root_joint_name="base_free",
            parts=[
                dict(name=n, body_name=n, mesh_path=f"{n}.obj") for n in ("base", "lid")
            ],
        ),),
    )
    return NativeSceneCfg.from_mapping(mapping), mapping


def _two_object_scene(tmp_path):
    layout, mapping = _scene(tmp_path)
    tree = ET.parse(layout.scene_xml_path)
    root = ET.SubElement(tree.getroot().find("worldbody"), "body", name="prop_root", pos=".3 0 .2")
    ET.SubElement(root, "freejoint", name="prop_free")
    ET.SubElement(root, "geom", type="box", size=".02 .03 .04", mass=".125")
    tree.write(layout.scene_xml_path)
    mapping["objects"] = (*mapping["objects"], dict(
        entity_name="loose_prop", root_body_name="prop_root", root_joint_name="prop_free",
        parts=[dict(name="base", body_name="prop_root", mesh_path="prop.obj")],
    ))
    return NativeSceneCfg.from_mapping(mapping), mapping



def write_native_scene_manifest(dataset: Path, scene: dict) -> Path:
    path = dataset / "native-reference.json"
    declaration = json.loads(path.read_text()) if path.exists() else {}
    declaration.update(native_scene=scene, recording_binding={
        "fixture": "native scene", "scene_identity": {"native_scene": scene},
    })
    path.write_text(json.dumps(declaration))
    return dataset / "motion.pt"


def test_native_metadata_keeps_free_roots_passive_mimics_and_real_effort_bounds(
    tmp_path,
):
    layout, _ = _scene(tmp_path)
    meta = load_robot_mjcf_meta(layout.scene_xml_path)
    assert [r.name for r in meta.free_roots] == ["left_free", "right_free", "base_free"]
    assert meta.joint_names == [
        "left_joint",
        "left_mimic",
        "right_joint",
        "right_other",
        "hinge",
    ]
    left = meta.subtree("left_root")
    assert left.driven_joint_names == ["left_joint"]
    assert left.mimics[0].reference_joint_name == "left_joint"
    np.testing.assert_allclose(left.command_limits(("command_A",)), [[0.0, 0.4]])
    # The source bounds the joint's actuator force, not actuator_forcerange.
    np.testing.assert_allclose(left.actuator_effort_limit, [0.8])
    obj = meta.subtree("base")
    assert obj.actuator_names == []
    assert obj.body_names == ["base", "lid"]
    assert obj.joint_names == ["hinge"] and not obj.joint_limited[0]
    np.testing.assert_allclose(obj.body_mass, [0.075, 0.025])


def test_native_conversion_inputs_keep_multiple_actual_object_trees(
    tmp_path,
):
    from common.native_reference import load_native_reference, native_usd_paths

    _, mapping = _two_object_scene(tmp_path)
    mapping["scene_xml_path"] = "source.xml"
    motion_file = write_native_scene_manifest(tmp_path, mapping)
    scene = load_native_reference(motion_file)["native_scene"]
    layout = NativeSceneCfg.from_mapping(scene)
    original = load_robot_mjcf_meta(layout.scene_xml_path)
    paths = write_native_mjcf_assets(layout, native_usd_paths(motion_file, scene))
    for binding in layout.articulations:
        saved = load_robot_mjcf_meta(paths[binding.entity_name])
        expected = original.subtree(binding.root_body_name)
        assert saved.body_names == expected.body_names
        assert saved.joint_names == expected.joint_names
        assert len(saved.free_roots) == 1
        np.testing.assert_array_equal(saved.body_mass, expected.body_mass)
        np.testing.assert_array_equal(saved.body_ipos, expected.body_ipos)
        np.testing.assert_allclose(
            saved.body_inertia, expected.body_inertia, rtol=1e-15
        )
    left = load_robot_mjcf_meta(paths["hand_left"])
    assert len(left.mimics) == 1
    obj = ET.parse(paths["object"]).getroot()
    assert len(obj.findall(".//freejoint") + obj.findall(".//joint[@type='free']")) == 1
    assert obj.find(".//joint[@name='hinge']") is not None
    static = ET.parse(paths["static"]).getroot()
    assert static.find(".//geom[@name='support']") is not None
    assert not static.findall(".//freejoint") + static.findall(".//joint[@type='free']")
    assert load_robot_mjcf_meta(paths["loose_prop"]).joint_names == []
    assert layout.object_root_body_names == {"object": "base", "loose_prop": "prop_root"}
    from dataclasses import replace
    from unittest import TestCase

    with TestCase().assertRaisesRegex(ValueError, "cover the source FREE roots"):
        write_native_mjcf_assets(replace(layout, objects=layout.objects[:1]), {
            "static": str(tmp_path / "missing/static.usd"),
            "articulations": {binding.entity_name: str(tmp_path / "missing" / (binding.entity_name + ".usd"))
                              for binding in layout.articulations},
        })


def test_native_file_meshes_resolve_from_each_generated_xml(tmp_path):
    import mujoco

    layout, _ = _scene(tmp_path)
    tree = ET.parse(layout.scene_xml_path)
    tree.getroot().find("compiler").set("meshdir", "input_meshes")
    asset = ET.SubElement(tree.getroot(), "asset")
    expected = {}
    vertices = [
        (-.03, -.02, -.01), (.03, -.02, -.01),
        (.03, .02, -.01), (-.03, .02, -.01),
        (-.03, -.02, .01), (.03, -.02, .01),
        (.03, .02, .01), (-.03, .02, .01),
    ]
    faces = [
        (1, 3, 2), (1, 4, 3), (5, 6, 7), (5, 7, 8),
        (1, 2, 6), (1, 6, 5), (4, 8, 7), (4, 7, 3),
        (1, 5, 8), (1, 8, 4), (2, 3, 7), (2, 7, 6),
    ]
    contents = "".join(
        f"{prefix} {' '.join(map(str, values))}\n"
        for prefix, rows in (("v", vertices), ("f", faces)) for values in rows
    )
    for side, scale in (("left", "1.2 .8 1.1"), ("right", ".9 1.3 .7")):
        mesh_file = tmp_path / "input_meshes" / side / "palm.obj"
        mesh_file.parent.mkdir(parents=True)
        mesh_file.write_text(contents)
        expected[side] = (mesh_file, scale)
        # Both supported source forms reach the same compiled writer. Left and
        # right have the same basename in different original asset directories.
        file = str(mesh_file) if side == "right" else f"{side}/palm.obj"
        ET.SubElement(asset, "mesh", name=f"{side}_palm", file=file, scale=scale)
        geom = tree.getroot().find(f".//body[@name='{side}_root']/geom")
        geom.attrib.pop("size")
        geom.attrib.update(type="mesh", mesh=f"{side}_palm")
    tree.write(layout.scene_xml_path)
    paths = write_native_mjcf_assets(layout, {
        "static": str(tmp_path / "converted/environment/static.usd"),
        "articulations": {b.entity_name: str(tmp_path / "converted" / b.entity_name / "asset.usd")
                          for b in layout.articulations},
    })
    for side, (original, scale) in expected.items():
        xml_path = Path(paths[f"hand_{side}"])
        saved = ET.parse(xml_path).getroot()
        compiler = saved.find("compiler")
        assert compiler.get("meshdir", "") == ""
        meshes = saved.findall("asset/mesh")
        assert len(meshes) == 1 and meshes[0].get("name") == f"{side}_palm"
        mesh = meshes[0]
        assert not Path(mesh.get("file")).is_absolute()
        # Match the pinned parser's string concatenation before normalization;
        # Path(base) / absolute would hide the original duplicated-prefix defect.
        importer_path = f"{xml_path.parent}/{compiler.get('meshdir', '')}/{mesh.get('file')}"
        assert Path(importer_path).resolve() == original
        np.testing.assert_array_equal(np.fromstring(mesh.get("scale"), sep=" "),
                                      np.fromstring(scale, sep=" "))
        assert original.read_text() == contents
        compiled = mujoco.MjModel.from_xml_path(str(xml_path))
        assert compiled.nmesh == 1
        assert compiled.mesh(0).name == f"{side}_palm"


def test_native_inline_mesh_roundtrip_keeps_body_local_convex_envelope(tmp_path):
    import mujoco

    layout, _ = _scene(tmp_path)
    tree = ET.parse(layout.scene_xml_path)
    asset = ET.SubElement(tree.getroot(), "asset")
    points = np.array([
        [.3 + x, -.2 + y, .4 + z]
        for x in (0, .04) for y in (0, .06) for z in (0, .09)
    ])
    ET.SubElement(
        asset, "mesh", name="inline_hull", vertex=" ".join(map(str, points.flat)),
        scale="1.3 .7 1.1", refpos=".2 -.1 .3",
        refquat=".9238795325112867 0 0 .3826834323650898",
    )
    geom = tree.getroot().find(".//body[@name='base']/geom")
    geom.attrib.clear()  # An unnamed real mesh geom also follows its body's order.
    geom.attrib.update(type="mesh", mesh="inline_hull", mass=".075",
                       pos=".07 .02 -.04", quat=".9659258262890683 0 .25881904510252074 0")
    tree.write(layout.scene_xml_path)
    original = mujoco.MjModel.from_xml_path(layout.scene_xml_path)
    paths = write_native_mjcf_assets(layout, {
        "static": str(tmp_path / "converted/static.usd"),
        "articulations": {b.entity_name: str(tmp_path / "converted" / b.entity_name / "asset.usd")
                          for b in layout.articulations},
    })
    saved_xml = ET.parse(paths["object"]).getroot()
    mesh = saved_xml.find("asset/mesh[@name='inline_hull']")
    assert mesh.get("vertex") is None and mesh.get("scale") == "1 1 1"
    assert not Path(mesh.get("file")).is_absolute()
    xml_path = Path(paths["object"])
    compiler = saved_xml.find("compiler")
    assert compiler.get("meshdir", "") == ""
    importer_path = f"{xml_path.parent}/{compiler.get('meshdir', '')}/{mesh.get('file')}"
    obj_path = Path(importer_path).resolve()
    assert obj_path == tmp_path / "converted/native_meshes/inline_hull.obj"
    obj = obj_path.read_text().splitlines()
    assert any(line.startswith("f ") for line in obj)
    saved = mujoco.MjModel.from_xml_path(paths["object"])

    def body_vertices(model):
        gid = int(model.body_geomadr[model.body("base").id])
        mid = int(model.geom_dataid[gid])
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
        return model.mesh_vert[va:va + vn] @ rotation.reshape(3, 3).T + model.geom_pos[gid]

    before, after = body_vertices(original), body_vertices(saved)
    # Support in all source face normals plus oblique directions tests the
    # actual envelope, independently of triangle/vertex ordering and winding.
    from scipy.spatial import ConvexHull

    directions = np.concatenate((ConvexHull(before).equations[:, :3], np.eye(3), -np.eye(3),
                                 np.array([[1., 2., 3.], [-3., 2., -1.]])), axis=0)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    np.testing.assert_allclose((before @ directions.T).max(0),
                               (after @ directions.T).max(0), atol=1e-7, rtol=0)
    for name in ("base", "lid"):
        a, b = original.body(name).id, saved.body(name).id
        np.testing.assert_array_equal(original.body_mass[a], saved.body_mass[b])
        np.testing.assert_array_equal(original.body_ipos[a], saved.body_ipos[b])
        inertias = []
        for model, bid in ((original, a), (saved, b)):
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.body_iquat[bid])
            rotation = rotation.reshape(3, 3)
            inertias.append(rotation @ np.diag(model.body_inertia[bid]) @ rotation.T)
        np.testing.assert_allclose(inertias[0], inertias[1], rtol=1e-14, atol=1e-16)


def test_native_affine_import_units_admission_and_normal_conversion_caller(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    from unittest.mock import Mock

    import mujoco
    import pytest

    from simulator.isaacsim import convert_assets

    layout, mapping = _scene(tmp_path)
    tree = ET.parse(layout.scene_xml_path)
    follower = tree.getroot().find(".//joint[@name='left_mimic']")
    follower.set("damping", ".1")
    follower.set("ref", ".2")
    tree.getroot().find(".//joint[@name='left_joint']").set("ref", ".3")
    tree.getroot().find("equality/joint").set("polycoef", ".1 -1.5 0 0 0")
    tree.write(layout.scene_xml_path)
    meta = load_robot_mjcf_meta(layout.scene_xml_path).subtree("left_root")
    fact, = native_mimic_imports(meta)
    assert (fact.joint_name, fact.reference_joint_name) == ("left_mimic", "left_joint")
    assert fact.gearing == 1.5
    assert fact.offset_degrees == pytest.approx(-np.degrees(.2 + .1 + 1.5 * .3))
    assert fact.viscous_friction_per_degree == pytest.approx(.1 * np.pi / 180)
    with pytest.raises(ValueError, match="zero scalar joint references"):
        validate_native_mjcf_import(mujoco.MjModel.from_xml_path(layout.scene_xml_path))
    # The actual admitted input has a known raw joint chart. Nonlinear and
    # cross-articulation equality requests fail before USD/SDK construction.
    for joint in tree.getroot().findall(".//joint[@ref]"):
        joint.attrib.pop("ref")
    tree.write(layout.scene_xml_path)
    model = mujoco.MjModel.from_xml_path(layout.scene_xml_path)
    validate_native_mjcf_import(model)
    model.eq_data[0, 2] = .01
    with pytest.raises(ValueError, match="active affine limited-revolute"):
        validate_native_mjcf_import(model)
    model.eq_data[0, 2] = 0
    model.eq_obj2id[0] = model.joint("right_joint").id
    with pytest.raises(ValueError, match="same FREE articulation"):
        validate_native_mjcf_import(model)
    motion_file = write_native_scene_manifest(tmp_path, mapping)
    events = []
    extensions = ModuleType("isaacsim.core.utils.extensions")
    extensions.enable_extension = Mock(
        side_effect=lambda name: (events.append(("enable", name)) or True)
    )
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.extensions", extensions)
    converters = ModuleType("isaaclab.sim.converters")
    converters.MjcfConverterCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    converters.MjcfConverter = lambda cfg: events.append(("convert", cfg))
    for name in ("isaaclab", "isaaclab.sim"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "isaaclab.sim.converters", converters)
    complete = Mock(side_effect=lambda path, meta: events.append(("complete", meta)))
    monkeypatch.setattr(convert_assets, "complete_native_usd", complete)
    convert_assets.convert_native_scene(str(motion_file))
    assert events[0] == ("enable", "isaacsim.asset.importer.mjcf")
    assert [kind for kind, _ in events[1:]] == ["convert", "complete"] * 3 + ["convert"]
    assert [meta.free_roots[0].body_name for kind, meta in events if kind == "complete"] == [
        b.root_body_name for b in layout.articulations
    ]
    configs = [cfg for kind, cfg in events if kind == "convert"]
    assert {Path(cfg.usd_dir) for cfg in configs} == {tmp_path / "native-usd"}
    assert [cfg.fix_base for cfg in configs] == [False, False, False, True]
    assert all(cfg.import_inertia_tensor and not cfg.make_instanceable for cfg in configs)
    # Failed activation is a conversion capability failure, never a retry with
    # an absent ImportConfig or another simulator/version.
    events.clear()
    extensions.enable_extension.side_effect = None
    extensions.enable_extension.return_value = False
    with pytest.raises(RuntimeError, match="could not enable isaacsim.asset.importer.mjcf"):
        convert_assets.convert_native_scene(str(motion_file))
    assert events == []
    # This CPU case proves connected dispatch and units, not USD/PhysX response.


def test_native_isaac_scene_keeps_mimic_viscous_and_passive_hinge_authority(monkeypatch, tmp_path):
    import sys
    from unittest.mock import Mock

    from omegaconf import OmegaConf
    from test_scene_contact_authority import _isaac_scene_type

    layout, _ = _scene(tmp_path)
    tree = ET.parse(layout.scene_xml_path)
    tree.getroot().find(".//joint[@name='left_joint']").attrib.update(
        damping=".2", armature=".02", frictionloss=".03", ref=".2",
    )
    tree.getroot().find(".//joint[@name='left_mimic']").attrib.update(
        damping=".1", armature=".04", frictionloss=".05", ref=".1",
    )
    tree.getroot().find(".//joint[@name='hinge']").attrib.update(
        armature=".06", frictionloss=".07",
    )
    tree.getroot().find("actuator/position[@name='command_A']").set("kv", ".3")
    tree.write(layout.scene_xml_path)
    meta = load_robot_mjcf_meta(layout.scene_xml_path)
    scene = sys.modules[_isaac_scene_type(monkeypatch).__module__]
    names = {}
    for binding in layout.articulations:
        path = tmp_path / (binding.entity_name + ".usd")
        path.write_text("this fixture checks public property forwarding, not a USD")
        names[binding.entity_name] = str(path)
    cfg = OmegaConf.create({"robot": {"self_collision": True}, "sim": {
        "native_assets": {"articulations": names},
        "physx": {"solver_position_iteration_count": 16, "solver_velocity_iteration_count": 4},
    }})
    events = []
    monkeypatch.setattr(scene, "validate_native_usd", lambda path, meta: events.append("admit"))
    monkeypatch.setattr(scene, "ImplicitActuatorCfg",
                        lambda **kwargs: (events.append("actuator") or SimpleNamespace(**kwargs)))
    articulation_cfg = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    articulation_cfg.InitialStateCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setattr(scene, "ArticulationCfg", articulation_cfg)
    monkeypatch.setattr(scene.sim_utils, "UsdFileCfg", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)
    monkeypatch.setattr(scene.sim_utils, "ArticulationRootPropertiesCfg",
                        lambda **kwargs: SimpleNamespace(**kwargs), raising=False)
    original_default_dtype = torch.get_default_dtype()
    left = meta.subtree("left_root")
    joint_default = left.joint_default.copy()
    assert joint_default.dtype == np.float64
    selected = scene._native_articulation_cfg(layout.hands[0], left, cfg)
    assert events == ["admit", "actuator"]
    initial = selected.init_state.joint_pos
    assert list(initial) == left.joint_names
    assert all(type(value) is float for value in initial.values())
    # Lab infers this source tensor's dtype before an indexed Float-buffer write.
    defaults = torch.zeros(2, left.num_joints, dtype=torch.float32)
    defaults[:, list(range(left.num_joints))] = torch.tensor(list(initial.values()))
    torch.testing.assert_close(
        defaults, torch.tensor(joint_default, dtype=torch.float32).expand_as(defaults),
    )
    drive = selected.actuators["native_position"]
    assert drive.joint_names_expr == ["left_joint"]
    assert drive.stiffness == {"left_joint": 20.0}
    assert drive.damping == {"left_joint": .5}
    calls = []
    authored_viscous = {"left_mimic": native_mimic_imports(left)[0].viscous_friction_per_degree}

    def property_asset(joint_count):
        data = SimpleNamespace(**{
            name: torch.full((2, joint_count), -7.0, dtype=torch.float32)
            for name in ("joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff")
        })

        def write(name, kind, value, **kwargs):
            # Execute the actual-shaped indexed write; clone-only mocks hide Double sources.
            getattr(data, name)[:, kwargs["joint_ids"]] = value
            saved = value.clone() if isinstance(value, torch.Tensor) else value
            calls.append((kind, saved, kwargs))

        return SimpleNamespace(
            data=data,
            write_joint_stiffness_to_sim=lambda value, **kw: write("joint_stiffness", "stiffness", value, **kw),
            write_joint_damping_to_sim=lambda value, **kw: write("joint_damping", "damping", value, **kw),
            write_joint_armature_to_sim=lambda value, **kw: write("joint_armature", "armature", value, **kw),
            write_joint_friction_coefficient_to_sim=lambda value, **kw: write("joint_friction_coeff", "friction", value, **kw),
            authored_viscous=authored_viscous.copy(),
        )

    asset = property_asset(left.num_joints)
    left_ids = list(reversed(range(left.num_joints)))
    scene._install_native_joint_properties(asset, left, left_ids, "cpu")
    follower = left.joint_names.index(left.mimics[0].joint_name)
    assert calls[0] == ("stiffness", 0.0, {"joint_ids": [left_ids[follower]]})
    assert calls[1][2] == {"joint_ids": left_ids}
    np.testing.assert_array_equal(calls[1][1], [.5, 0])
    for field, expected in (
        ("joint_armature", left.joint_armature),
        ("joint_friction_coeff", left.joint_frictionloss),
    ):
        actual = getattr(asset.data, field)[:, left_ids]
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=torch.float32).expand_as(actual),
        )
    assert asset.authored_viscous == authored_viscous
    np.testing.assert_array_equal(left.joint_default, joint_default)
    assert left.joint_damping.dtype == left.joint_armature.dtype == left.joint_frictionloss.dtype == np.float64
    obj = meta.subtree("base")
    selected = scene._native_articulation_cfg(layout.objects[0], obj, cfg)
    assert selected.actuators == {}
    assert all(type(value) is float for value in selected.init_state.joint_pos.values())
    calls.clear()
    asset = property_asset(obj.num_joints)
    object_ids = list(range(obj.num_joints))
    scene._install_native_joint_properties(asset, obj, object_ids, "cpu")
    assert [kind for kind, _, _ in calls] == ["damping", "armature", "friction"]
    torch.testing.assert_close(calls[0][1], torch.tensor(obj.joint_damping, dtype=torch.float32))
    assert all(kwargs == {"joint_ids": object_ids} for _, _, kwargs in calls)
    for field, expected in (
        ("joint_damping", obj.joint_damping),
        ("joint_armature", obj.joint_armature),
        ("joint_friction_coeff", obj.joint_frictionloss),
    ):
        actual = getattr(asset.data, field)
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=torch.float32).expand_as(actual),
        )
    assert torch.get_default_dtype() == original_default_dtype


def test_native_isaac_contacts_resolve_named_rigid_descendants_before_reset(monkeypatch, tmp_path):
    import sys
    from types import ModuleType

    import pytest
    from omegaconf import OmegaConf
    from test_scene_contact_authority import _isaac_scene_type

    layout, mapping = _scene(tmp_path)
    metadata = load_robot_mjcf_meta(layout.scene_xml_path)
    scene_module = sys.modules[_isaac_scene_type(monkeypatch).__module__]
    rigid_api = object()
    pxr = ModuleType("pxr")
    pxr.UsdPhysics = SimpleNamespace(RigidBodyAPI=rigid_api)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    roots = {
        "hand_left": "/World/envs/env_.*/Imports/receiver_A",
        "hand_right": "/World/envs/env_.*/Ports/receiver_B",
        "object": "/World/envs/env_.*/Scene/passive_object",
    }
    left = roots["hand_left"].replace("env_.*", "env_0")
    right = roots["hand_right"].replace("env_.*", "env_0")
    stage = object()
    events = []
    checked_rigid_paths = []

    class Prim:
        def __init__(self, path, rigid):
            self.path, self.rigid = path, rigid

        def GetName(self):
            return self.path.rsplit("/", 1)[1]

        def GetPath(self):
            return self.path

        def HasAPI(self, api):
            assert api is rigid_api
            checked_rigid_paths.append(self.path)
            return self.rigid

    prims = [
        Prim(left + "/left_root", False),  # Same-name wrapper is not a body.
        Prim(left + "/left_root/left_root", True),
        Prim(left + "/left_root/Digits/left_follower", True),
        Prim(right + "/Rig/right_root", False),
        Prim(right + "/Rig/Roots/right_root", True),
        Prim(right + "/Rig/right_second", True),
        Prim("/World/unrelated/left_root", True),
    ]

    def find_paths(regex, *, stage):
        assert stage is native_scene.stage
        events.append(("find", regex))
        return [regex.replace("env_.*", "env_1"), regex.replace("env_.*", "env_0")]

    def find_children(path, predicate, *, stage):
        assert stage is native_scene.stage and path in (left, right)
        events.append(("bodies", path))
        return [prim for prim in prims
                if prim.path.startswith(path + "/") and predicate(prim)]

    native_scene = SimpleNamespace(
        stage=stage, env_prim_paths=["/World/envs/env_0", "/World/envs/env_1"],
        articulations={}, rigid_objects={}, sensors={},
        clone_environments=lambda **kw: events.append(("clone", kw)),
        filter_collisions=lambda **kw: events.append(("filter", kw)),
    )
    monkeypatch.setattr(scene_module, "InteractiveScene", lambda cfg: native_scene)
    monkeypatch.setattr(scene_module, "InteractiveSceneCfg", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(scene_module, "_native_articulation_cfg",
                        lambda binding, meta, cfg: SimpleNamespace(prim_path=roots[binding.entity_name]))
    monkeypatch.setattr(scene_module, "Articulation", lambda cfg: SimpleNamespace(cfg=cfg))
    monkeypatch.setattr(scene_module, "ContactSensorCfg", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(scene_module, "ContactSensor", lambda cfg: SimpleNamespace(cfg=cfg))
    for name in ("UsdFileCfg", "DomeLightCfg"):
        monkeypatch.setattr(scene_module.sim_utils, name,
                            lambda **kw: SimpleNamespace(func=lambda *args: None), raising=False)
    monkeypatch.setattr(scene_module.sim_utils, "find_matching_prim_paths", find_paths, raising=False)
    monkeypatch.setattr(scene_module.sim_utils, "get_all_matching_child_prims", find_children, raising=False)
    cfg = OmegaConf.create({
        "robot": {"native_scene": mapping, "contact_links": {
            "left": ["left_root", "left_follower"],
            "right": ["right_root", "right_second"],
        }},
        "scene": {"num_envs": 2, "env_spacing": 1.0},
        "sim": {"native_assets": {"static": "unused-static.usd"}},
    })

    class AtReset(RuntimeError):
        pass

    def stop_at_reset():
        events.append(("reset",))
        raise AtReset

    # Exercise the production builder through sensor registration, before SDK reset.
    with pytest.raises(AtReset):
        scene_module._build_native_isaac_scene(
            cfg, SimpleNamespace(reset=stop_at_reset), "cpu", metadata,
        )
    expected = {
        "left_root": roots["hand_left"] + "/left_root/left_root",
        "left_follower": roots["hand_left"] + "/left_root/Digits/left_follower",
        "right_root": roots["hand_right"] + "/Rig/Roots/right_root",
        "right_second": roots["hand_right"] + "/Rig/right_second",
    }
    assert {name: sensor.cfg.prim_path for name, sensor in native_scene.sensors.items()} == {
        f"_native_contact_{name}": path for name, path in expected.items()
    }
    assert all(sensor.cfg.update_period == 0.0 for sensor in native_scene.sensors.values())
    assert checked_rigid_paths == [prim.path for prim in prims[:-1]]
    assert events == [
        ("find", roots["hand_left"]), ("bodies", left),
        ("find", roots["hand_right"]), ("bodies", right),
        ("clone", {"copy_from_source": False}),
        ("filter", {"global_prim_paths": []}), ("reset",),
    ]


def _native_action_case(
    tmp_path, *, wrist_target_twist="source_only", dt=.02, cutoff_hz=None,
    hand_sides=("left", "right"),
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    layout, mapping = _scene(tmp_path)
    mapping["hands"] = [hand for hand in mapping["hands"] if hand["side"] in hand_sides]
    meta = load_robot_mjcf_meta(layout.scene_xml_path)
    names = ["left_joint", "left_mimic", "right_joint", "right_other"]
    accepted = []

    def install(target, support, control):
        accepted.append((target.clone(), support.clone(), asdict(control)))
        return torch.zeros(2, 3), torch.zeros(2, 3)

    class Scene(dict):
        env_origins = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    target = torch.tensor([[0.2, 0.3, 0.4, 0.5]]).repeat(2, 1)
    entity = SimpleNamespace(
        data=SimpleNamespace(joint_pos_target=target),
        hands={
            s: SimpleNamespace(apply_root_pose_target=install)
            for s in ("left", "right")
        },
        native_command_limits=meta.command_limits,
        find_joints_by_actuator_names=lambda aa: (
            [names.index(meta.actuator_joint[a]) for a in aa],
            [meta.actuator_joint[a] for a in aa],
        ),
        set_joint_position_target=lambda value, ids: target.__setitem__(
            (slice(None), ids), value
        ),
    )
    roots = {
        s: torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 1.0, 0.0, 0.0]]
        ).repeat(2, 1)
        for s in ("left", "right")
    }
    ml = SimpleNamespace(
        native_joint_position_support=torch.tensor(
            [[True, True, False, True], [True, True, False, True]]
        ),
        native_root_body_names={s: f"{s}_root" for s in roots},
        native_root_state_w=roots,
        native_root_pose_support={s: torch.ones(2, dtype=torch.bool) for s in roots},
        native_root_linear_velocity_support={
            s: torch.zeros(2, dtype=torch.bool) for s in roots
        },
        native_root_angular_velocity_support={
            s: torch.ones(2, dtype=torch.bool) for s in roots
        },
    )
    command = SimpleNamespace(
        motion_lib=ml,
        _motion_flat_ids=torch.tensor([0, 1]),
        motion_ids=torch.tensor([2, 3]),
        motion_steps=torch.tensor([4, 5]),
        ref_joint_pos=torch.zeros(2, 4),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=dt,
        scene=Scene(robot=entity),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        close=lambda: None,
    )
    action = NativeResidualAction(
        NativeResidualActionCfg(
            entity_name="robot",
            command_name="motion",
            native_scene=mapping,
            translation_residual_m=0.1,
            rotation_residual_rad=np.pi / 2,
            clip={".*": (-1.0, 1.0)},
            wrist_target_twist=wrist_target_twist,
            cutoff_hz={} if cutoff_hz is None else cutoff_hz,
        ),
        env,
    )
    return env, action, command, accepted


def test_actual_native_action_derives_dimensions_and_records_accepted_supported_targets(
    tmp_path,
):
    env, action, _command, accepted = _native_action_case(tmp_path)
    assert action.action_dim == 2 * 6 + 3
    from envs.vec_env import FlatObsVecEnv

    env.action_manager = SimpleNamespace(_terms={"native": action})
    wrapper = object.__new__(FlatObsVecEnv)
    wrapper._env = env
    side_ids = wrapper._action_side_indices(["left", "right"])
    assert side_ids == {
        "left": [0, 1, 2, 6, 7, 8, 12],
        "right": [3, 4, 5, 9, 10, 11, 13, 14],
    }
    from envs.actions.residual import ResidualAction
    from envs.rewards.base import BaseRewards

    env.action_manager.action = torch.arange(action.action_dim).float().repeat(2, 1)
    for side, indices in side_ids.items():
        mask = BaseRewards._side_action_mask(env, side)
        assert torch.nonzero(mask).flatten().tolist() == indices
        torch.testing.assert_close(
            BaseRewards.action_penalty(env, side),
            env.action_manager.action[:, indices].square().mean(-1),
        )
    stock_action = object.__new__(ResidualAction)
    stock_action._entity = SimpleNamespace(joint_names=["R_finger", "L_wrist"])
    stock_action._all_joint_ids = torch.tensor([1, 0])
    assert stock_action.action_names == ("L_wrist", "R_finger")
    assert stock_action.action_sides == ("left", "right")
    raw = torch.ones(2, action.action_dim) * 2
    # Rotate only about WORLD z; the supported source omega must rotate too.
    raw[:, 6:12] = 0.0
    raw[:, 8] = raw[:, 11] = 1.0
    action.process_actions(raw)
    assert not action.installed_control["installed_valid"].any()
    action.apply_actions()
    record = action.installed_control
    np.testing.assert_allclose(
        record["finger_target_rad"], [[0.4, 0.4, 2.0], [0.4, 0.4, 2.0]]
    )
    assert record["finger_joint_names"] == ("left_joint", "right_joint", "right_other")
    np.testing.assert_allclose(record["root_state_w"][..., 7:10], 0.0)
    np.testing.assert_allclose(
        record["root_state_w"][..., 10:13], [[[0.0, 1.0, 0.0]] * 2] * 2, atol=1e-6
    )
    np.testing.assert_allclose(
        record["root_state_w"][:, 0, :3], env.scene.env_origins + 0.1
    )
    assert not record["root_linear_velocity_support"].any()
    np.testing.assert_array_equal(record["reference_motion_steps"], [4, 5])
    assert len(accepted) == 2 and record["installed_valid"].all()
    assert record["root_control"]["main_tracker_numeric_parity"] is False
    recipe = record["action_recipe"]
    assert recipe["translation_residual_per_axis_m"] == 0.1
    assert recipe["rotation_residual_per_axis_rad"] == np.pi / 2
    assert recipe["action_names"] == action.action_names
    assert recipe["action_sides"] == action.action_sides
    import json

    json.dumps(recipe)  # A recorder can retain the actual accepted action recipe.
    action.reset(torch.tensor([0]))
    np.testing.assert_array_equal(record["installed_valid"], [False, True])


def test_native_wrist_source_only_keeps_source_rates_through_filter_updates(tmp_path):
    from scipy.spatial.transform import Rotation

    _env, action, _command, _accepted = _native_action_case(
        tmp_path, cutoff_hz={"wrist_trans": 5., "wrist_rot": 5.}
    )
    for coordinate in (0., .8, -.6):
        action.process_actions(torch.full_like(action.raw_action, coordinate))
        action.apply_actions()
        record = action.installed_control
        correction = action._processed_actions[:, action._rotation_slice].reshape(-1, 3)
        matrices = Rotation.from_rotvec(
            correction.numpy() * action.cfg.rotation_residual_rad
        ).as_matrix()
        expected = matrices[:, :, 0].reshape(action.num_envs, len(action.hand_sides), 3)
        np.testing.assert_allclose(record["root_state_w"][..., 7:10], 0., atol=1e-6)
        np.testing.assert_allclose(record["root_state_w"][..., 10:13], expected, atol=1e-6)
        assert not action._correction_interval_valid.any()
        assert not action._correction_twist.any()
    assert record["action_recipe"]["wrist_target_twist"] == "source_only"
    assert record["action_recipe"]["control_period_s"] == .02


def test_native_wrist_feedforward_preserves_zero_and_static_world_corrections(tmp_path):
    from scipy.spatial.transform import Rotation

    _env, original, old_command, _accepted = _native_action_case(tmp_path / "source")
    _env, candidate, command, _accepted = _native_action_case(
        tmp_path / "candidate", wrist_target_twist="filtered_correction_feedforward"
    )
    reference = Rotation.from_rotvec([.2, -.1, .35]).as_quat()[[3, 0, 1, 2]]
    for owner in (old_command, command):
        for root in owner.motion_lib.native_root_state_w.values():
            root[:, 3:7] = torch.tensor(reference, dtype=root.dtype)
    for coordinate in (0., .3):
        for action in (original, candidate):
            action.reset()
        raw = torch.full_like(original.raw_action, coordinate)
        for _ in range(2):
            for action in (original, candidate):
                action.process_actions(raw)
                action.apply_actions()
            torch.testing.assert_close(candidate.installed_control["root_state_w"],
                                       original.installed_control["root_state_w"], atol=1e-6, rtol=0)
            torch.testing.assert_close(candidate.installed_control["finger_target_rad"],
                                       original.installed_control["finger_target_rad"], atol=0, rtol=0)
            torch.testing.assert_close(candidate._correction_twist,
                                       torch.zeros_like(candidate._correction_twist), atol=1e-6, rtol=0)
    # A nonfirst static packet has zero increment and retains a usable Jacobian.
    differentiable = raw.detach().requires_grad_(True)
    candidate.process_actions(differentiable)
    gradient, = torch.autograd.grad(candidate._correction_twist[0, 0, 3:].sum(), differentiable)
    assert torch.isfinite(gradient).all()
    assert gradient[:, candidate._rotation_slice].abs().sum() > 0
    _env, zero, _command, _accepted = _native_action_case(
        tmp_path / "zero-jacobian", wrist_target_twist="filtered_correction_feedforward"
    )
    zero.reset()
    zero.process_actions(torch.zeros_like(zero.raw_action))
    differentiable_zero = torch.zeros_like(zero.raw_action, requires_grad=True)
    zero.process_actions(differentiable_zero)
    zero_gradient, = torch.autograd.grad(zero._correction_twist[0, 0, 3:].sum(), differentiable_zero)
    assert torch.isfinite(zero_gradient).all()
    expected_gradient = torch.zeros_like(zero_gradient)
    begin = zero._rotation_slice.start
    expected_gradient[0, begin:begin + 3] = zero.cfg.rotation_residual_rad / zero._control_dt
    torch.testing.assert_close(zero_gradient, expected_gradient, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dt,hand_sides", ((.02, ("left", "right")), (.04, ("right",))),
                         ids=("bilateral20ms", "unilateral40ms"))
def test_native_wrist_feedforward_uses_filtered_world_increment_and_control_dt(
    tmp_path, dt, hand_sides,
):
    from scipy.spatial.transform import Rotation

    _env, action, _command, _accepted = _native_action_case(
        tmp_path, wrist_target_twist="filtered_correction_feedforward", dt=dt,
        cutoff_hz={"wrist_trans": 5., "wrist_rot": 5.}, hand_sides=hand_sides,
    )
    first = torch.linspace(-.4, .4, action.action_dim).repeat(action.num_envs, 1)
    requested = torch.linspace(2.4, -2.4, action.action_dim).repeat(action.num_envs, 1)
    action.process_actions(first)
    previous = action._processed_actions.clone()
    assert not action._correction_interval_valid.any()
    action.process_actions(requested)
    expected_filter = previous + action._alpha * (requested.clamp(-1., 1.) - previous)
    torch.testing.assert_close(action._processed_actions, expected_filter)
    action.apply_actions()
    current = action._processed_actions
    previous_r = previous[:, action._rotation_slice].reshape(-1, 3).numpy()
    current_r = current[:, action._rotation_slice].reshape(-1, 3).numpy()
    old_rotation = Rotation.from_rotvec(previous_r * action.cfg.rotation_residual_rad).as_matrix()
    new_rotation = Rotation.from_rotvec(current_r * action.cfg.rotation_residual_rad).as_matrix()
    relative = new_rotation @ old_rotation.transpose(0, 2, 1)
    angular_correction = Rotation.from_matrix(relative).as_rotvec() / dt
    shape = (action.num_envs, len(hand_sides), 3)
    expected_linear = ((current[:, action._translation_slice] - previous[:, action._translation_slice])
                       * action.cfg.translation_residual_m / dt).reshape(shape)
    expected_angular = (angular_correction + new_rotation[:, :, 0]).reshape(shape)
    record = action.installed_control
    np.testing.assert_allclose(record["root_state_w"][..., 7:10], expected_linear, atol=2e-6)
    np.testing.assert_allclose(record["root_state_w"][..., 10:13], expected_angular, atol=2e-5)
    assert not record["root_linear_velocity_support"].any()  # The source component is absent.
    assert record["root_angular_velocity_support"].all()
    assert action._correction_interval_valid.all()
    assert record["action_recipe"]["root_rate_support"] == "independent_source_component_support"


def test_native_wrist_feedforward_reset_and_source_gaps_keep_independent_carry(tmp_path):
    _env, action, command, accepted = _native_action_case(
        tmp_path, wrist_target_twist="filtered_correction_feedforward"
    )
    for value in command.motion_lib.native_root_angular_velocity_support.values():
        value.zero_()
    action.process_actions(torch.zeros_like(action.raw_action))
    action.apply_actions()
    command.motion_lib.native_root_pose_support["left"][0] = False
    for coordinate in (.2, .4):
        action.process_actions(torch.full_like(action.raw_action, coordinate))
        action.apply_actions()
        assert action._correction_interval_valid.all()
        assert action.wrist_control_state["filtered_coordinates"][0].any()
        assert not accepted[-2][1][0]  # Root effort authority remains absent.
        torch.testing.assert_close(action.installed_control["root_state_w"][0, 0, 7:13],
                                   torch.zeros(6), atol=0, rtol=0)
    command.motion_lib.native_root_pose_support["left"][0] = True
    action.process_actions(torch.full_like(action.raw_action, .6))
    action.apply_actions()
    # Reentry compares adjacent controller packets, not the last supported pose.
    torch.testing.assert_close(action.installed_control["root_state_w"][0, 0, 7:10],
                               torch.ones(3), atol=1e-6, rtol=0)
    assert not action.installed_control["root_linear_velocity_support"].any()
    assert not action.installed_control["root_angular_velocity_support"].any()
    action.reset(torch.tensor([0]))
    assert action.wrist_control_state["filter_pending"].tolist() == [True, False]
    assert not action.wrist_control_state["filtered_coordinates"][0].any()
    assert not action._correction_interval_valid[0]
    action.process_actions(torch.full_like(action.raw_action, .8))
    action.apply_actions()
    assert action._correction_interval_valid.tolist() == [False, True]
    torch.testing.assert_close(action._correction_twist[0], torch.zeros_like(action._correction_twist[0]),
                               atol=0, rtol=0)
    assert action._correction_twist[1].abs().sum() > 0


def test_native_wrist_feedforward_holds_one_increment_and_tracks_held_raw_ema(tmp_path):
    _env, action, _command, accepted = _native_action_case(
        tmp_path, wrist_target_twist="filtered_correction_feedforward",
        cutoff_hz={"wrist_trans": 5., "wrist_rot": 5.},
    )
    action.process_actions(torch.zeros_like(action.raw_action))
    raw = torch.full_like(action.raw_action, .8)
    action.process_actions(raw)
    first = action.capture_control_context()
    for _ in range(4):
        action.apply_actions()
    for substep in range(1, 4):
        for hand in range(len(action.hand_sides)):
            torch.testing.assert_close(accepted[substep * len(action.hand_sides) + hand][0],
                                       accepted[hand][0], atol=0, rtol=0)
    torch.testing.assert_close(action._correction_twist, first.correction_twist, atol=0, rtol=0)
    action.process_actions(raw)
    assert not torch.equal(action._filter_state, first.filter_state)
    expected = ((action._filter_state[:, action._translation_slice]
                 - first.filter_state[:, action._translation_slice])
                * action.cfg.translation_residual_m / action._control_dt)
    torch.testing.assert_close(action._correction_twist[..., :3].flatten(1), expected)
    assert action._correction_twist.abs().sum() > 0


def test_native_wrist_target_twist_rejects_unknown_convention(tmp_path):
    with pytest.raises(ValueError, match="target-twist convention"):
        _native_action_case(tmp_path, wrist_target_twist="unknown")


def test_native_free_root_wrench_is_root_only_and_recomputed_from_current_body_state(
    tmp_path,
):
    layout, _ = _scene(tmp_path)
    meta = load_robot_mjcf_meta(layout.scene_xml_path).subtree("left_root")
    transforms = torch.zeros(1, len(meta.body_names), 7)
    transforms[..., 6] = 1.0  # PhysX XYZW
    transforms[:, 1:, 1] = 0.02
    calls = []
    view = SimpleNamespace(
        get_masses=lambda: torch.as_tensor(meta.body_mass).float().unsqueeze(0),
        get_coms=lambda: torch.zeros_like(transforms),
        get_inertias=lambda: torch.diag_embed(torch.as_tensor(meta.body_inertia).float()).reshape(1, len(meta.body_names), 9),
        get_link_transforms=lambda: transforms,
        get_link_velocities=lambda: torch.zeros(1, len(meta.body_names), 6),
    )
    art = SimpleNamespace(
        is_fixed_base=False,
        num_instances=1,
        body_names=meta.body_names,
        find_bodies=lambda names, preserve_order: (list(range(len(names))), names),
        find_joints=lambda names, preserve_order: (list(range(len(names))), names),
        root_physx_view=view,
        set_external_force_and_torque=lambda f, t, **kw: calls.append(
            (f.clone(), t.clone(), kw)
        ),
    )
    clock = _FreshnessClock()
    hand = FreeArticulationAdapter(art, meta, clock, "cpu", root_body_name="left_root")
    state = torch.tensor(
        [[0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    force1, _ = hand.apply_root_pose_target(
        state, torch.tensor([True]), layout.hands[0].root_control
    )
    transforms[..., 0] = 0.04
    clock.bump()
    force2, _ = hand.apply_root_pose_target(
        state, torch.tensor([True]), layout.hands[0].root_control
    )
    assert force1[0, 0] > 0 and force2[0, 0] < 0
    for forces, torques, kwargs in calls:
        assert torch.count_nonzero(forces[:, 1:]) == 0
        assert torch.count_nonzero(torques[:, 1:]) == 0
        assert kwargs["is_global"] is True
        assert float(forces[:, 0].norm()) <= 10.0
        assert float(torques[:, 0].norm()) <= 1.0


def test_native_metadata_admits_position_and_equivalent_general_charts(tmp_path):
    layout, _ = _scene(tmp_path)
    tree = ET.parse(layout.scene_xml_path)
    actuators = tree.getroot().find("actuator")
    a, b, c = list(actuators)
    # XHand's explicit damping, F1's saved general form and Sharpa's dampratio.
    a.set("kv", "1.5")
    b.tag = "general"
    b.attrib.clear()
    b.attrib.update(name="command_B", joint="right_joint", gaintype="fixed",
                    biastype="affine", gainprm="20", biasprm="0 -20")
    c.set("kp", "6.95")
    c.set("dampratio", ".9")
    actuators[:] = [c, a, b]
    ET.SubElement(actuators, "motor", name="world_wrench_x", jointinparent="left_free",
                  gear="1 0 0 0 0 0")
    tree.write(layout.scene_xml_path)

    meta = load_robot_mjcf_meta(layout.scene_xml_path)
    assert meta.actuator_names == ["command_C", "command_A", "command_B"]
    assert meta.actuator_joint == {
        "command_C": "right_other", "command_A": "left_joint", "command_B": "right_joint",
    }
    np.testing.assert_array_equal(meta.actuator_kp, [6.95, 20., 20.])
    assert np.isfinite(meta.actuator_kv[0]) and meta.actuator_kv[0] > 0
    np.testing.assert_array_equal(meta.actuator_kv[1:], [1.5, 0.])
    assert [r.name for r in meta.free_roots] == ["left_free", "right_free", "base_free"]
    assert meta.subtree("base").actuator_names == []
    np.testing.assert_allclose(meta.command_limits(("command_A",)), [[0., .4]])


def test_native_metadata_rejects_motor_and_nonunit_gear_position_targets(tmp_path):
    import pytest

    for name, tag, attrs in (
        ("motor", "motor", {}),
        ("geared", "position", {"kp": "20", "gear": "2"}),
        ("reversed", "position", {"kp": "20", "gear": "-1"}),
    ):
        directory = tmp_path / name
        directory.mkdir()
        layout, _ = _scene(directory)
        tree = ET.parse(layout.scene_xml_path)
        actuator = tree.getroot().find("actuator")[0]
        actuator.tag = tag
        actuator.attrib.clear()
        actuator.attrib.update(name="command_A", joint="left_joint", **attrs)
        tree.write(layout.scene_xml_path)
        with pytest.raises(ValueError, match="actuator 'command_A'.*memoryless unit-gear position servo"):
            load_robot_mjcf_meta(layout.scene_xml_path)


def test_native_metadata_rejects_changed_bias_filtered_and_delayed_targets(tmp_path):
    import pytest

    for name, tag, attrs in (
        ("changed_gain_bias", "general", {"gainprm": "20", "biastype": "affine", "biasprm": "0 -10"}),
        ("target_offset", "general", {"gainprm": "20", "biastype": "affine", "biasprm": "1 -20"}),
        ("filtered", "position", {"kp": "20", "timeconst": ".01"}),
        ("delayed", "position", {"kp": "20", "delay": ".01", "nsample": "8"}),
    ):
        directory = tmp_path / name
        directory.mkdir()
        layout, _ = _scene(directory)
        tree = ET.parse(layout.scene_xml_path)
        actuator = tree.getroot().find("actuator")[0]
        actuator.tag = tag
        actuator.attrib.clear()
        actuator.attrib.update(name="command_A", joint="left_joint", **attrs)
        tree.write(layout.scene_xml_path)
        with pytest.raises(ValueError, match="actuator 'command_A'.*memoryless unit-gear position servo"):
            load_robot_mjcf_meta(layout.scene_xml_path)


def test_native_metadata_rejects_nonbuiltin_roster_before_position_indexing(tmp_path):
    import pytest

    layout, _ = _scene(tmp_path)
    tree = ET.parse(layout.scene_xml_path)
    actuators = tree.getroot().find("actuator")
    ET.SubElement(actuators, "general", name="changed_root_gain", jointinparent="left_free",
                  gaintype="affine", gainprm="1 1 0", gear="1 0 0 0 0 0")
    tree.write(layout.scene_xml_path)
    with pytest.raises(ValueError, match="actuator 'changed_root_gain'.*builtin scalar fixed-gain"):
        load_robot_mjcf_meta(layout.scene_xml_path)


if __name__ == "__main__":
    # The pinned author environment need not install a separate test runner.
    from pathlib import Path
    from tempfile import TemporaryDirectory

    torch.set_num_threads(1)
    cases = (
        test_native_metadata_keeps_free_roots_passive_mimics_and_real_effort_bounds,
        test_native_conversion_inputs_keep_multiple_actual_object_trees,
        test_actual_native_action_derives_dimensions_and_records_accepted_supported_targets,
        test_native_free_root_wrench_is_root_only_and_recomputed_from_current_body_state,
    )
    for case in cases:
        with TemporaryDirectory() as directory:
            case(Path(directory))
        print(f"PASS {case.__name__}", flush=True)
    print(
        f"{len(cases)} passed; no FK, dynamics steps or Isaac application", flush=True
    )


def test_native_scalar_coordinate_metadata_preserves_compiled_type_and_named_order(tmp_path):
    from simulator.isaacsim.entity import RigidObjectAdapter
    from simulator.native.entity import NamedArticulationView

    source = tmp_path / "mixed-coordinates.xml"
    source.write_text("""<mujoco><compiler angle="degree"/>
      <worldbody><body name="assembly"><freejoint name="root"/>
        <geom type="sphere" size=".02" mass=".1"/>
        <body name="carriage"><joint name="translation" type="slide" range="-.1 .2"/>
          <geom type="sphere" size=".01" mass=".02"/>
          <body name="flap"><joint name="rotation" type="hinge" range="-90 90"/>
            <geom type="sphere" size=".01" mass=".01"/>
          </body>
        </body>
      </body></worldbody></mujoco>""")
    meta = load_robot_mjcf_meta(source)
    assert meta.joint_names == ["translation", "rotation"]
    assert meta.joint_types == ["slide", "hinge"]
    np.testing.assert_allclose(meta.joint_range, [[-.1, .2], [-np.pi / 2, np.pi / 2]])
    part = meta.subtree("carriage")
    assert part.joint_names == meta.joint_names and part.joint_types == meta.joint_types
    leaf = meta.subtree("flap")
    assert leaf.joint_names == ["rotation"] and leaf.joint_types == ["hinge"]
    entity = object.__new__(NamedArticulationView)
    entity._meta, entity.joint_names = meta, ["rotation", "translation"]
    assert entity.joint_coordinate_metadata() == {
        "joint_names": ("rotation", "translation"),
        "joint_types": ("hinge", "slide"),
        "position_units": ("rad", "m"),
        "velocity_units": ("rad/s", "m/s"),
    }
    rigid = object.__new__(RigidObjectAdapter)
    rigid.joint_names = ()  # Actual public rigid owner, without constructing PhysX.
    assert rigid.joint_coordinate_metadata() == {
        "joint_names": (), "joint_types": (),
        "position_units": (), "velocity_units": (),
    }
