"""Explicit stock asset/config/body joins on CPU; no Kit or physics advance."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import trimesh
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from simulator.isaacsim import convert_assets
from simulator.isaacsim.assets import load_robot_asset, resolve_robot_asset
from simulator.isaacsim.entity import ArticulationAdapter, _FreshnessClock
from simulator.mjcf_meta import load_robot_mjcf_meta

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "robot,override", [("xhand", False), ("xhand", True), ("sharpa", True)]
)
def test_registered_stock_assets_share_hydra_conversion_and_runtime_binding(
    monkeypatch,
    tmp_path,
    robot,
    override,
):
    monkeypatch.delenv("FDR_ROBOT_USD", raising=False)
    if override:
        monkeypatch.setenv("FDR_ROBOT_USD", str(tmp_path / "operator-xhand.usd"))
    with initialize_config_dir(config_dir=str(_REPO / "config"), version_base=None):
        cfg = compose(
            config_name="flashdexretarget",
            overrides=[f"envs/robot@robot={robot}_bimanual"],
        )
    live = resolve_robot_asset(cfg)
    direct = load_robot_asset(f"config/envs/robot/{robot}_bimanual.yaml")
    assert live == direct
    resolved = tmp_path / "run.yaml"
    # The converter needs only these selected fields, not an unrelated motion
    # file, learner, CUDA context or AppLauncher to interpret this registration.
    OmegaConf.save(
        OmegaConf.create(
            {
                "robot": {"xml_path": cfg.robot.xml_path},
                "sim": {
                    "robot_asset": OmegaConf.to_container(
                        cfg.sim.robot_asset, resolve=True
                    )
                },
            }
        ),
        resolved,
    )
    assert load_robot_asset(resolved) == live
    if robot == "xhand":
        assert (
            live.source_urdf_path
            == _REPO / "assets/robot/xhand/urdf/xhand_bimanual_spec.urdf"
        )
        expected = (
            tmp_path / "operator-xhand.usd"
            if override
            else (
                _REPO / "assets/robot/xhand/urdf/converted_spec/xhand_bimanual_spec.usd"
            )
        )
        assert live.usd_path == expected
        assert live.body_name("R_forearm_rot_y_link") == "right_hand_link"
    else:
        assert (
            live.usd_path
            == _REPO / "assets/robot/sharpa/urdf/converted/sharpa_bimanual.usd"
        )
        assert live.body_name("R_forearm_rot_y_link") == "right_hand_link"
    # Check the registered source-name join, without importing a USD or
    # claiming that the importer preserved its physical graph.
    links = {
        link.attrib["name"]
        for link in ET.parse(live.source_urdf_path).getroot().findall("link")
    }
    bodies = ET.parse(live.mjcf_path).getroot().findall("./worldbody//body")
    assert all(live.body_name(body.attrib["name"]) in links for body in bodies)


def test_collision_visual_and_unregistered_assets_require_explicit_destinations(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("FDR_ROBOT_USD", str(tmp_path / "not-a-global-override.usd"))
    stock = load_robot_asset(
        "config/envs/robot/xhand_bimanual.yaml", collision_visual=True
    )
    assert (
        stock.usd_path
        == _REPO / "assets/robot/xhand/urdf/converted_colvis/xhand_bimanual_spec.usd"
    )
    with pytest.raises(ValueError, match="collision_visual_usd_path"):
        load_robot_asset(
            "config/envs/robot/sharpa_bimanual.yaml", collision_visual=True
        )
    # An unknown MJCF must not acquire a XHAND asset, even with the old env var.
    with pytest.raises(ValueError, match="explicit sim.robot_asset"):
        resolve_robot_asset(
            OmegaConf.create({"robot": {"xml_path": "unknown_left.xml"}, "sim": {}})
        )


@pytest.fixture
def registered_asset(tmp_path):
    mjcf = tmp_path / "different_morphology.xml"
    mjcf.write_text("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="R_forearm_rot_y_link"><joint name="rotation" range="-1 1"/>
      <geom type="box" size=".02 .03 .04" mass=".2"/>
      <body name="tool"><joint name="extension" type="slide" range="0 .1"/>
        <geom type="box" size=".01 .01 .02" mass=".05"/>
        <site name="tool_reference"/>
      </body>
    </body></worldbody>
    <actuator><position name="rotate" joint="rotation" kp="20"/>
      <position name="extend" joint="extension" kp="10"/></actuator></mujoco>""")
    urdf = tmp_path / "original.urdf"
    urdf.write_text(
        '<robot name="different"><link name="imported_palm"/><link name="tool"/>'
        '<link name="extra_tool_frame"/></robot>'
    )
    config = tmp_path / "registered.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "robot": {"xml_path": str(mjcf)},
                "sim": {
                    "robot_asset": {
                        "source_urdf_path": str(urdf),
                        "usd_path": str(tmp_path / "elsewhere/explicit.usd"),
                        "body_name_map": {"R_forearm_rot_y_link": "imported_palm"},
                    }
                },
            }
        ),
        config,
    )
    return config


@pytest.mark.parametrize("renamed", [False, True])
def test_articulation_joins_configured_bodies_and_permuted_joints(
    registered_asset, renamed
):
    binding = load_robot_asset(registered_asset)
    meta = load_robot_mjcf_meta(binding.mjcf_path)
    mapping = binding.body_name_map if renamed else {}
    body_names = [mapping.get(name, name) for name in reversed(meta.body_names)]
    joint_names = list(reversed(meta.joint_names))
    transforms = torch.zeros(1, len(body_names), 7)
    transforms[..., 6] = 1.0  # PhysX XYZW
    transforms[0, :, 0] = torch.arange(len(body_names)) + 4
    q = torch.tensor([[0.7, 0.2]])
    view = SimpleNamespace(
        get_coms=lambda: torch.zeros_like(transforms),
        get_link_transforms=lambda: transforms,
        get_dof_positions=lambda: q,
    )
    requested = []

    def find_bodies(names, preserve_order):
        assert preserve_order
        requested.extend(names)
        return [body_names.index(name) for name in names], names

    art = SimpleNamespace(
        body_names=body_names,
        num_instances=1,
        root_physx_view=view,
        find_bodies=find_bodies,
        find_joints=lambda names, preserve_order: (
            [joint_names.index(n) for n in names],
            names,
        ),
    )
    kwargs = {"body_name_map": mapping} if renamed else {}
    adapter = ArticulationAdapter(art, meta, _FreshnessClock(), "cpu", **kwargs)
    assert requested == [mapping.get(n, n) for n in meta.body_names]
    torch.testing.assert_close(
        adapter.data.body_link_pos_w[0, :, 0], torch.tensor([5.0, 4.0])
    )
    torch.testing.assert_close(adapter.data.joint_pos, torch.tensor([[0.2, 0.7]]))
    assert (
        adapter.body_names == meta.body_names
        and adapter.joint_names == meta.joint_names
    )


@pytest.mark.parametrize("existing", [False, True])
def test_converter_uses_configured_source_destination_and_complete_named_roster(
    monkeypatch,
    registered_asset,
    existing,
):
    binding = load_robot_asset(registered_asset)
    if existing:
        binding.usd_path.parent.mkdir()
        binding.usd_path.write_text("not opened by the stubbed converter")
    module = ModuleType("isaaclab.sim.converters")
    module.UrdfConverterCfg = Mock()
    module.UrdfConverter = Mock(
        return_value=SimpleNamespace(usd_path=str(binding.usd_path))
    )
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.sim", ModuleType("isaaclab.sim"))
    monkeypatch.setitem(sys.modules, "isaaclab.sim.converters", module)
    repair, verify = Mock(), Mock()
    monkeypatch.setattr(convert_assets, "_block_empty_robot_visual_references", repair)
    monkeypatch.setattr(convert_assets, "verify_usd", verify)
    convert_assets.convert_robot(str(registered_asset), force=False)
    repair.assert_called_once_with(str(binding.usd_path), str(binding.source_urdf_path))
    verify.assert_called_once_with(
        str(binding.usd_path),
        expect_links=["imported_palm", "tool", "extra_tool_frame"],
    )
    if existing:
        module.UrdfConverter.assert_not_called()
    else:
        options = module.UrdfConverterCfg.call_args.kwargs
        assert options["asset_path"] == str(binding.source_urdf_path)
        assert options["usd_dir"] == str(binding.usd_path.parent)
        assert options["usd_file_name"] == binding.usd_path.name
        assert options["fix_base"] is True and options["merge_fixed_joints"] is False
        assert options["collider_type"] == "convex_hull"
        assert options["self_collision"] is False


def test_collision_visual_relocation_preserves_original_meshes_and_physics(tmp_path):
    for side, extent in (("near", 0.02), ("far", 0.05)):
        folder = tmp_path / side
        folder.mkdir()
        trimesh.creation.box(extents=(extent, 0.03, 0.04)).export(folder / "same.obj")
    urdf = tmp_path / "source.urdf"
    urdf.write_text("""<robot name="different"><link name="body">
      <inertial><mass value=".2"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".002" iyz="0" izz=".002"/></inertial>
      <visual><geometry><mesh filename="unrelated.obj"/></geometry></visual>
      <collision><origin xyz=".01 0 0" rpy="0 .2 0"/><geometry><mesh filename="near/same.obj" scale="1 2 3"/></geometry></collision>
      <collision><origin xyz="0 .03 0" rpy=".1 0 0"/><geometry><mesh filename="far/same.obj"/></geometry></collision>
    </link></robot>""")
    original = ET.parse(urdf).getroot().find("link")
    output = convert_assets.author_robot_colvis_urdf(
        str(urdf), str(tmp_path / "generated")
    )
    result = ET.parse(output).getroot().find("link")
    assert ET.tostring(result.find("inertial")) == ET.tostring(
        original.find("inertial")
    )
    for before, after, visual in zip(
        original.findall("collision"),
        result.findall("collision"),
        result.findall("visual"),
        strict=True,
    ):
        source = (
            urdf.parent / before.find("geometry/mesh").attrib["filename"]
        ).resolve()
        assert after.find("geometry/mesh").attrib["filename"] == str(source)
        assert after.find("geometry/mesh").get("scale", "1.0 1.0 1.0") == visual.find(
            "geometry/mesh"
        ).get("scale")
        assert ET.tostring(after.find("origin")) == ET.tostring(before.find("origin"))
        assert ET.tostring(visual.find("origin")) == ET.tostring(before.find("origin"))
    hulls = [
        visual.find("geometry/mesh").attrib["filename"]
        for visual in result.findall("visual")
    ]
    assert len(set(hulls)) == 2
    assert all((Path(output).parent / mesh).is_file() for mesh in hulls)
    assert trimesh.load(Path(output).parent / hulls[0], force="mesh").extents[
        0
    ] == pytest.approx(0.02)
    assert trimesh.load(Path(output).parent / hulls[1], force="mesh").extents[
        0
    ] == pytest.approx(0.05)
