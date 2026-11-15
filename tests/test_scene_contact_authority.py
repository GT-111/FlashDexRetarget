"""Scene-owned CPU contracts; no Kit, plant advance or learned update."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

# The public package exposes vendored managers without starting Kit.
import simulator.isaacsim  # noqa: F401

# isort: split
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from omegaconf import OmegaConf

from envs.commands.motion_tracking.motion_tracking import MotionTrackingCommand
from envs.commands.motion_tracking.motion_tracking_cfg import ObjectCfg
from envs.commands.motion_tracking.objects.multi_object import MultiObjectMixin
from envs.commands.motion_tracking.reference.object import ObjectPropertiesMixin
from simulator.env import ManagerBasedRlEnv, ManagerEnvCfg
from simulator.isaacsim.entity import _FreshnessClock
from simulator.isaacsim.sensors import build_tracking_contact_views


def _isaac_scene_type(monkeypatch):
    """Import the real scene adapter with only its unused SDK type names supplied."""
    for name, attributes in {
        "isaaclab": (),
        "isaaclab.sim": (),
        "isaaclab.actuators": ("ImplicitActuatorCfg",),
        "isaaclab.assets": (
            "Articulation",
            "ArticulationCfg",
            "RigidObject",
            "RigidObjectCfg",
        ),
        "isaaclab.scene": ("InteractiveScene", "InteractiveSceneCfg"),
        "isaaclab.sensors": ("ContactSensor", "ContactSensorCfg"),
    }.items():
        module = ModuleType(name)
        for attr in attributes:
            setattr(module, attr, type(attr, (), {}))
        monkeypatch.setitem(sys.modules, name, module)
    return importlib.import_module("simulator.isaacsim.scene").IsaacSceneAdapter


def _bank(*, reports=True):
    body_names = ["anchor", "one", "two", "three"]
    links = ["three", "anchor", "two", "one"]
    positions = torch.zeros(1, len(body_names), 3)
    positions[0, :, 0] = torch.arange(len(body_names)) + 1
    robot = SimpleNamespace(
        data=SimpleNamespace(body_link_pos_w=positions),
        find_bodies=lambda names, preserve_order: (
            [body_names.index(name) for name in names],
            names,
        ),
    )
    metadata = SimpleNamespace(
        sites={
            "noncanonical_A": SimpleNamespace(body_name="one"),
            "noncanonical_B": SimpleNamespace(body_name="three"),
            "noncanonical_C": SimpleNamespace(body_name="two"),
        }
    )
    sensors = (
        [
            SimpleNamespace(
                data=SimpleNamespace(
                    net_forces_w=torch.tensor(
                        [[[float(body_names.index(name) + 1), 0.0, 0.0]]]
                    )
                )
            )
            for name in links
        ]
        if reports
        else []
    )
    return build_tracking_contact_views(
        sensors,
        _FreshnessClock(),
        "cpu",
        side="left",
        body_names=links,
        contact_site_names=["noncanonical_B", "noncanonical_A", "noncanonical_C"],
        metadata=metadata,
        robot=robot,
        margins=torch.full((len(links),), 0.02),
    )


class _AdmittedCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.admitted = True

    @property
    def command(self):
        return torch.full((self.num_envs, 1), self.cfg.marker)

    @property
    def object_distance_sign(self):
        return "winding"

    def sdf_query(self, points, side):
        assert side == "left"
        gradient = torch.zeros_like(points)
        gradient[..., 2] = 1
        return points.new_full(points.shape[:-1], self.cfg.marker), gradient

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        pass


@dataclass(kw_only=True)
class _AdmittedCommandCfg(CommandTermCfg):
    marker: float

    def build(self, env):
        return _AdmittedCommand(self, env)


def _bound_observation(env):
    command = env.command_manager.get_term("motion")
    bank = env.scene._tracking_banks["left"]
    assert command.admitted and bank.command is command
    return bank._link_sdf()[0]


def test_manager_binds_complete_independent_commands_before_observations(monkeypatch):
    adapter_type = _isaac_scene_type(monkeypatch)
    environments = []
    for marker in (0.003, 0.008):
        backend_scene = SimpleNamespace(
            cfg=SimpleNamespace(num_envs=1), env_origins=torch.zeros(1, 3)
        )
        scene = adapter_type(backend_scene, "cpu", SimpleNamespace(current_time=0.0))
        bank, views = _bank()
        scene._tracking_banks["left"] = bank
        scene.sensors.update(views)
        sim = SimpleNamespace(expand_model_fields=lambda fields: None)
        env = ManagerBasedRlEnv(
            ManagerEnvCfg(
                decimation=4,
                physics_dt=0.005,
                commands={
                    "motion": _AdmittedCommandCfg(
                        marker=marker, resampling_time_range=(1e9, 1e9)
                    )
                },
                observations={
                    "actor": ObservationGroupCfg(
                        terms={"bound": ObservationTermCfg(func=_bound_observation)}
                    )
                },
            ),
            scene,
            sim,
            device="cpu",
        )
        assert scene.hand_part_contact_facts() is None
        assert (
            scene.tracking_contact_metadata()["left"]["pair_normal_force_available"]
            is False
        )
        environments.append(env)
    assert (
        environments[0].scene._tracking_banks["left"]
        is not environments[1].scene._tracking_banks["left"]
    )
    assert environments[0].command_manager.get_term("motion") is not environments[
        1
    ].command_manager.get_term("motion")
    for env, marker in zip(environments, (0.003, 0.008), strict=True):
        observed = env.observation_manager.compute(update_history=False)
        torch.testing.assert_close(observed["actor"], torch.full((1, 4), marker))


def test_configured_noncanonical_contact_sites_select_named_force_rows():
    bank, views = _bank()
    command = SimpleNamespace(object_distance_sign="winding")
    command.sdf_query = lambda points, side: (
        torch.full(points.shape[:-1], 0.01),
        torch.zeros_like(points),
    )
    bank.bind_command(command)
    assert bank.tip_rows == (0, 3, 2)
    torch.testing.assert_close(
        views["l_fingertip_contact"].data.force[0, :, 0], torch.tensor([4.0, 2.0, 3.0])
    )
    torch.testing.assert_close(
        bank._link_sdf()[2][0, :, 0], torch.tensor([4.0, 1.0, 3.0, 2.0])
    )
    metadata = views["l_alllink_contact"].data.measurement_metadata
    assert metadata["body_names"] == ["three", "anchor", "two", "one"]
    assert metadata["net_force_view_available"] is True
    assert metadata["signed_distance_query_available"] is True
    assert metadata["pair_normal_force_available"] is False


def test_unsigned_geometry_rejects_signed_penetration_instead_of_zero():
    bank, views = _bank()
    bank.bind_command(SimpleNamespace(object_distance_sign="unsigned"))
    assert bank.measurement_metadata["signed_distance_query_available"] is False
    with pytest.raises(NotImplementedError, match="signed penetration is unavailable"):
        _ = views["l_fingertip_penetration"].data.dist


def test_no_report_observation_zeros_have_no_force_measurement_authority():
    bank, views = _bank(reports=False)
    bank.bind_command(SimpleNamespace(object_distance_sign="winding"))
    assert not bank.force().any()
    metadata = views["l_alllink_contact"].data.measurement_metadata
    assert metadata["net_force_view_available"] is False
    assert metadata["ablation_zero_observation_is_measurement"] is False
    assert metadata["pair_normal_force_available"] is False


@pytest.mark.parametrize("ground", [False, True])
def test_explicit_parking_preserves_selected_worlds_and_repark_recipe(ground):
    writes = []

    class Scene(dict):
        env_origins = torch.zeros(3, 3)

    scene = Scene(
        {
            name: SimpleNamespace(
                data=SimpleNamespace(),
                write_root_state_to_sim=lambda state, env_ids, name=name: writes.append(
                    (name, state.clone(), env_ids.clone())
                ),
            )
            for name in ("slot0", "slot1")
        }
    )

    class ParkedObjects(MultiObjectMixin, ObjectPropertiesMixin):
        def __init__(self):
            self._env = SimpleNamespace(scene=scene, num_envs=3)
            self.device = "cpu"
            self.cfg = SimpleNamespace(
                object=ObjectCfg(
                    entity_names={"left": ["slot0", "slot1"]},
                    park_inactive_on_ground=ground,
                )
            )
            self._park_inactive_on_ground = self.cfg.object.park_inactive_on_ground
            self._side_list = ["left"]
            self.motion_lib = SimpleNamespace(
                obj_trans={"left": torch.zeros(1, 3)},
                _motion_num_frames=torch.tensor([8, 8]),
                traj_obj_slot={"left": torch.tensor([0, 1])},
            )
            self.motion_ids = torch.tensor([0, 1, 0])
            self.motion_steps = torch.zeros(3, dtype=torch.long)
            self._eval_mode = True
            self._xfrc_curr_ctrl = None

        def _apply_object_assist(self):
            pass

    command = ParkedObjects()
    MotionTrackingCommand._update_command(command)
    assert command.motion_steps.tolist() == [1, 1, 1]
    if ground:
        assert writes == []
    else:
        assert [(name, ids.tolist()) for name, _, ids in writes] == [
            ("slot0", [1]),
            ("slot1", [0, 2]),
        ]
    root = Path(__file__).resolve().parents[1]
    assert OmegaConf.load(
        root / "config/envs/commands/motion_tracking.yaml"
    ).commands.object.park_inactive_on_ground
    assert not OmegaConf.load(
        root / "config/envs/native.yaml"
    ).commands.object.park_inactive_on_ground


@pytest.mark.parametrize("exact", [False, True])
def test_normal_recorder_distinguishes_unavailable_pair_force_from_measured_zero(
    tmp_path, exact
):
    from test_native_rollout_record import record_reference_phase_fixture

    source = dict(
        source_unit_id="synthetic",
        prepared_time_ns=[0, 20_000_000, 40_000_000],
        selected_source_frame_index=[0, 1, 2],
        prepared_source_ordinal=[0.0, 1.0, 2.0],
        prepared_elapsed_time_ns=[0, 20_000_000, 40_000_000],
    )
    binding = dict(
        scene_from_source_world=np.eye(4).tolist(),
        hands={
            side: dict(
                body_name=side + "_root",
                physical_joint_names={"driver": side + "_driver"},
            )
            for side in ("left", "right")
        },
        object_parts={
            name: dict(entity_name="object", body_name=body)
            for name, body in (("base", "object_root"), ("lid", "object_lid"))
        },
        object_part_names=["base", "lid"],
        clips={"fixture": source},
    )
    association = {
        "left": {
            "force_authority": "synthetic net-link force/SDF association, not exact pairs",
            "pair_normal_force_available": False,
        }
    }

    def setup(env, command):
        env.scene.tracking_contact_metadata = lambda: association
        if not exact:
            env.scene.hand_part_contact_facts = lambda: None
            return

        def facts():
            row = int(command.reference_phase["motion_steps"].item())
            supported = row > 0
            addresses = {
                f"{side}/{name}": dict(
                    hand_side=side,
                    hand_body_name=side + "_root",
                    part_entity_name="object",
                    part_body_name=part["body_name"],
                )
                for side in ("left", "right")
                for name, part in binding["object_parts"].items()
            }
            return dict(
                addresses=addresses,
                pairs={
                    name: dict(
                        normal_force_n=torch.tensor(
                            [0.0 if supported else float("nan")]
                        ),
                        force_bearing_contact_present=torch.tensor([False]),
                        force_bearing_distance_max_m=torch.tensor([float("nan")]),
                    )
                    for name in addresses
                },
                measurement_support=torch.tensor([supported]),
                sample_time_s=torch.tensor(
                    [row * 0.02 - 0.005 if supported else float("nan")]
                ),
                clock="synthetic solved substep start, not physics qualification",
                normal_force_authority="synthetic exact-pair protocol",
                distance_authority="synthetic no-contact separation unavailable",
            )

        env.scene.hand_part_contact_facts = facts

    manifest_path = record_reference_phase_fixture(
        tmp_path, binding, native=True, setup_scene=setup
    )
    manifest = json.loads(manifest_path.read_text())
    with np.load(
        manifest_path.parent / manifest["saved"][0]["rollout"], allow_pickle=False
    ) as arrays:
        assert (
            json.loads(str(arrays["native_contact_association_metadata_json"]))
            == association
        )
        np.testing.assert_array_equal(arrays["recorded_motion_steps"], [0, 1, 2])
        np.testing.assert_array_equal(arrays["command_motion_steps"], [1, 2, 2])
        if exact:
            np.testing.assert_array_equal(
                arrays["hand_part_measurement_support"][:, 0], [False, True, True]
            )
            assert np.isnan(arrays["hand_part_normal_force_n"][0]).all()
            assert (arrays["hand_part_normal_force_n"][1:] == 0).all()
            assert not arrays["hand_part_force_bearing_contact_present"].any()
            assert np.isnan(arrays["hand_part_force_bearing_distance_max_m"]).all()
        else:
            assert not any(name.startswith("hand_part_") for name in arrays.files)


def test_isaac_scene_clock_reads_global_sdk_time_through_partial_entity_reset(monkeypatch):
    adapter_type = _isaac_scene_type(monkeypatch)
    reset_rows = []
    context = SimpleNamespace(current_time=7.125000000000007)
    backend_scene = SimpleNamespace(
        cfg=SimpleNamespace(num_envs=3), env_origins=torch.zeros(3, 3),
        reset=lambda rows: reset_rows.append(rows.clone()),
    )
    scene = adapter_type(backend_scene, "cpu", context)
    before = scene.physics_time_s()
    assert before.dtype == torch.float64 and before.shape == (3,)
    assert before.tolist() == [context.current_time] * 3
    rows = torch.tensor([2, 0])
    scene.reset(rows)
    assert torch.equal(reset_rows[0], rows)
    assert torch.equal(scene.physics_time_s(), before)
    # This is a stopped SDK protocol value, deliberately unlike nominal manager time.
    context.current_time = 7.148456789012345
    after = scene.physics_time_s()
    assert after.tolist() == [context.current_time] * 3
    assert before.tolist() == [7.125000000000007] * 3
    assert scene.backend_identity["physics_clock"]["scope"] == "scene-global time repeated over cloned environments"
    assert scene.hand_part_contact_facts() is None
    assert scene.hand_part_contact_interval_facts() is None
