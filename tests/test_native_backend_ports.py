"""CPU contracts for real backend seams; no Kit, GPU, learned model or integration."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest
import torch
from envs import create_envs
from simulator.mjcf_meta import load_robot_mjcf_meta
from simulator.native.mjwarp import MjWarpPlant
from simulator.native.newton import NewtonPlant
from simulator.native.plant import (
    NativePlant,
    body_origin_velocity,
    free_root_velocity_to_native,
    root_drive_indices,
    root_origin_wrench,
)
from simulator.native.scene import NativeScene, NativeSim
from simulator.runtime import launch_runtime
from test_native_contract import _scene


def _mechanics(tmp_path):
    layout, mapping = _scene(tmp_path)
    document = ET.parse(layout.scene_xml_path)
    root = document.getroot()
    for ordinal, side in enumerate(("left", "right")):
        body = root.find(f"./worldbody/body[@name='{side}_root']")
        body.set("pos", f"{-.3 + .6 * ordinal} 0 .5")
        for axis, gear in enumerate(np.eye(6, dtype=int)):
            ET.SubElement(root.find("actuator"), "motor", name=f"{side}_drive_{axis}",
                          jointinparent=f"{side}_free", gear=" ".join(map(str, gear)))
    body = root.find("./worldbody/body[@name='left_root']")
    ET.SubElement(body, "inertial", pos=".007 -.011 .014", mass=".2", diaginertia=".001 .002 .0025")
    root.find("./worldbody/body[@name='base']").set("pos", "1 0 .5")
    document.write(layout.scene_xml_path)
    model = mujoco.MjModel.from_xml_path(layout.scene_xml_path)
    return layout, mapping, model


def test_native_config_dispatch_and_runtime_do_not_boot_kit_or_require_usd(tmp_path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    environment = {"FDR_NATIVE_SCENE_XML": str(tmp_path / "not-opened.xml"),
                   "FDR_NATIVE_BOX_BASE_MESH": "base.obj", "FDR_NATIVE_BOX_LID_MESH": "lid.obj"}
    had = set(sys.modules)
    with patch.dict("os.environ", environment, clear=True), initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None,
    ):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
        for backend in ("mjwarp", "newton"):
            cfg = compose(config_name="native", overrides=[f"envs/simulator@simulator={backend}",
                "commands.motion_file=synthetic.pt", "native_recording_binding={}",
                "native_tracking_tolerances={root_position_error_m: 1, root_rotation_error_rad: 1, part_position_error_m: 1, part_rotation_error_rad: 1, joint_position_error_rad: 1, command_position_error_rad: 1}"])
            OmegaConf.resolve(cfg)
            assert cfg.sim.backend == backend and cfg.sim.runtime_launcher is None
            assert "native_assets" not in cfg.sim
            assert launch_runtime(cfg) is None
            with patch("hydra.utils.get_method", return_value=lambda config, device: (config, device)) as factory:
                result = create_envs(cfg, device="cpu")
            assert result == (cfg, "cpu")
            factory.assert_called_once_with("simulator.native.create.create_native_envs")
    assert not any(n.startswith(("isaaclab", "isaacsim", "omni.")) for n in set(sys.modules) - had)


def test_world_root_wrench_and_body_velocity_match_actual_mujoco_charts(tmp_path):
    _layout, _mapping, model = _mechanics(tmp_path)
    data = mujoco.MjData(model)
    root = model.body("left_root").id
    joint = model.joint("left_free").id
    q, d = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    data.qpos[q + 3:q + 7] = (.8, .2, .4, .4)
    data.qvel[d:d + 6] = (.2, -.1, .4, .3, -.7, .9)
    mujoco.mj_forward(model, data)
    quaternion = torch.tensor(data.xquat[root][None])
    force, torque = torch.tensor([[1.3, -.4, .8]]), torch.tensor([[.2, -.1, .3]])
    request = root_origin_wrench(force, torque, quaternion.float(), torch.tensor(model.body_ipos[root]).float())
    data.ctrl[:] = 0
    drives = root_drive_indices(model, "left_free")
    data.ctrl[drives] = request[0].numpy()
    mujoco.mj_fwdActuation(model, data)
    expected = np.zeros(model.nv)
    mujoco.mj_applyFT(model, data, force[0].numpy().astype(float), torque[0].numpy().astype(float),
                     data.xipos[root], root, expected)
    np.testing.assert_allclose(data.qfrc_actuator, expected, atol=2e-7, rtol=2e-7)
    for body in load_robot_mjcf_meta(_layout.scene_xml_path).subtree("left_root").body_names:
        bid = model.body(body).id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, bid, velocity, 0)
        linear, angular = body_origin_velocity(torch.tensor(data.cvel[bid]), torch.tensor(data.xpos[bid]),
                                               torch.tensor(data.subtree_com[model.body_rootid[bid]]))
        np.testing.assert_allclose(linear.numpy(), velocity[3:], atol=1e-12)
        np.testing.assert_allclose(angular.numpy(), velocity[:3], atol=1e-12)
    root_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, root, root_velocity, 0)
    state = torch.tensor(np.r_[data.xpos[root], data.xquat[root], root_velocity[3:], root_velocity[:3]])
    native = free_root_velocity_to_native(state)
    np.testing.assert_allclose(native.numpy(), data.qvel[d:d + 6], atol=1e-12)
    # The authored left passive mimic remains outside the independent command width.
    assert load_robot_mjcf_meta(_layout.scene_xml_path).actuator_names == ["command_A", "command_B", "command_C"]


def test_mjwarp_constructor_initializes_runtime_before_torch_stream_conversion():
    events = []
    initialized = False

    class ReachedModelLoad(Exception):
        pass

    def initialize():
        nonlocal initialized
        events.append("warp_init")
        initialized = True

    stream = SimpleNamespace(cuda_stream=7, device=torch.device("cuda:0"))

    def torch_stream(device):
        assert device == stream.device
        events.append("torch_stream")
        return stream

    def warp_stream(value):
        # Warp's real Torch bridge reads context.runtime directly. This first
        # constructor must initialize it rather than inherit another backend's.
        if not initialized:
            raise AttributeError("uninitialized Warp runtime before stream conversion")
        assert value is stream
        events.append("warp_stream")
        return object()

    def model_load(path):
        assert path == "not-opened.xml"
        events.append("model_load")
        raise ReachedModelLoad

    warp = SimpleNamespace(init=initialize, stream_from_torch=warp_stream)
    with patch.dict(sys.modules, {"warp": warp, "mujoco_warp": SimpleNamespace()}), \
            patch.object(torch.cuda, "current_stream", side_effect=torch_stream), \
            patch.object(mujoco.MjModel, "from_xml_path", side_effect=model_load), \
            pytest.raises(ReachedModelLoad):
        MjWarpPlant(scene_xml_path="not-opened.xml", num_envs=2, sim_dt=.005,
                    device="cuda:0", contact_pairs={}, nconmax=512, njmax=2048)
    assert events == ["warp_init", "torch_stream", "warp_stream", "model_load"]


def test_newton_adapter_uses_public_completed_state_and_actual_solver_clock(tmp_path):
    from dexforge_tracking.envs.backends.newton_device import NewtonDeviceBackend
    from dexforge_tracking.envs.physics import NativeSceneSpec

    layout, _mapping, model = _mechanics(tmp_path)
    backend = MagicMock(spec=NewtonDeviceBackend)
    backend.model = model
    state = SimpleNamespace(qpos=torch.tensor(np.tile(model.qpos0, (2, 1)), dtype=torch.float32),
                            qvel=torch.zeros(2, model.nv))
    clock = torch.tensor([7.125, 3.25], dtype=torch.float64)
    events = []
    backend.public_state_torch.return_value = state

    def physics_time():
        events.append("time")
        return clock.clone()

    def step(control, *, nsteps):
        events.append("step")
        assert control.shape == (2, model.nu) and nsteps == 1
        clock.add_(.005)
        return SimpleNamespace(public_state=state, applied_joint_effort=torch.zeros(2, model.nv))

    facts = {"normal_force_n": torch.tensor([2., 0.]),
             "force_bearing_contact_present": torch.tensor([True, False]),
             "force_bearing_distance_max_m": torch.tensor([.003, float("nan")])}
    backend.get_physics_time_torch.side_effect = physics_time
    backend.step_torch.side_effect = step
    backend.get_body_pair_contact_facts.return_value = facts
    hand, part = [model.body(name).id for name in ("left_root", "base")]
    with patch("dexforge_tracking.envs.backend.create_backend", return_value=backend) as factory:
        plant = NewtonPlant(scene_xml_path=layout.scene_xml_path, num_envs=2, sim_dt=.005,
                            device="cpu", contact_pairs={"left/base": ([hand], [part])})
    args, kwargs = factory.call_args
    assert args[0] == "newton" and isinstance(args[1], NativeSceneSpec)
    assert args[1].model_file == layout.scene_xml_path and args[1].collision_generation is None
    assert args[2:] == (2, .005) and kwargs == {"device": "cpu", "cuda_graph": True}
    plant.advance()
    assert events == ["time", "step"]
    torch.testing.assert_close(plant.contact_sample_time_s, torch.tensor([7.125, 3.25], dtype=torch.float64),
                               rtol=0, atol=0)
    assert plant.contact_support.tolist() == [True, True]
    for key, value in facts.items():
        torch.testing.assert_close(plant.contact_facts["left/base"][key], value, equal_nan=True)
    requested = backend.get_body_pair_contact_facts.call_args.args
    np.testing.assert_array_equal(requested[0], [hand])
    np.testing.assert_array_equal(requested[1], [part])
    qpos, qvel = plant.state()
    assert qpos is state.qpos and qvel is state.qvel


class _ProtocolPlant(NativePlant):
    """Synthetic state owner for reset/installation contracts, never a physical result."""

    def __init__(self, model):
        self._init_buffers(model, 2, "cpu", {})
        self.q = torch.tensor(np.tile(model.qpos0, (2, 1)), dtype=torch.float32)
        self.v = torch.zeros(2, model.nv)
        self.effort = torch.zeros_like(self.v)
        self._time = torch.zeros(2, dtype=torch.float64)
        self.identity = {"backend": "synthetic_protocol_not_physics"}
        self.set_calls = self.advance_calls = 0

    def state(self):
        return self.q, self.v

    def reset(self, env_ids):
        self.q[env_ids] = torch.tensor(self.model.qpos0, dtype=self.q.dtype)
        self.v[env_ids] = 0
        self.control[env_ids] = 0
        self._time[env_ids] = 0
        self.invalidate_contacts(env_ids)

    def set_state(self, env_ids, qpos, qvel):
        self.set_calls += 1
        self.q[env_ids], self.v[env_ids] = qpos, qvel
        self.invalidate_contacts(env_ids)

    def body_state(self, body_ids):
        data = mujoco.MjData(self.model)
        values = []
        for q, v in zip(self.q.numpy(), self.v.numpy(), strict=True):
            data.qpos[:], data.qvel[:] = q, v
            mujoco.mj_forward(self.model, data)
            velocities = []
            for bid in body_ids:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_XBODY, int(bid), velocity, 0)
                velocities.append(velocity)
            values.append((data.xpos[body_ids].copy(), data.xquat[body_ids].copy(),
                           np.asarray(velocities)[:, 3:], np.asarray(velocities)[:, :3]))
        return tuple(torch.tensor(np.stack([x[k] for x in values]), dtype=torch.float32) for k in range(4))

    def physics_time(self):
        return self._time

    def advance(self):
        self.advance_calls += 1
        self._time += .005

    def forward(self):
        pass

    def capture_physics(self):
        return self.q.clone(), self.v.clone(), self._time.clone()

    def restore_physics(self, state):
        for target, source in zip((self.q, self.v, self._time), state, strict=True):
            target.copy_(source)


def test_native_views_stage_one_reset_then_install_without_teleporting(tmp_path):
    layout, _mapping, model = _mechanics(tmp_path)
    meta = load_robot_mjcf_meta(layout.scene_xml_path)
    plant = _ProtocolPlant(model)
    sim = NativeSim(plant)
    scene = NativeScene(layout, meta, sim, "robot")
    before = plant.q[0].clone()
    rows = torch.tensor([1])
    sim.reset(rows)
    robot = scene["robot"]
    position = torch.tensor([[.2, .3, .1, .4]])
    robot.write_joint_state_to_sim(position, torch.zeros_like(position), env_ids=rows)
    root_state = scene["hand_left"]._default_root_state[rows].clone()
    root_state[:, 3:7] = torch.tensor([.8, .2, .4, .4])
    root_state[:, 10:13] = torch.tensor([.3, -.7, .9])
    scene["hand_left"].write_root_state_to_sim(root_state, rows)
    assert plant.set_calls == 0
    scene.write_data_to_sim()
    assert plant.set_calls == 1
    torch.testing.assert_close(plant.q[0], before, rtol=0, atol=0)
    hand = scene["hand_left"]
    d = hand._root.dof_address
    torch.testing.assert_close(plant.v[1, d + 3:d + 6], free_root_velocity_to_native(root_state)[0, 3:])
    assert len(robot.joint_names) == 4 and scene["object"].joint_names == ["hinge"]
    ids, names = robot.find_joints_by_actuator_names(("command_C", "command_A"))
    assert names == ["right_other", "left_joint"] and ids == [3, 0]
    robot.set_joint_position_target(torch.tensor([[.3, .1], [.4, .2]]), ids)
    for _ in range(2):
        hand.apply_root_pose_target(hand._default_root_state, torch.ones(2, dtype=torch.bool), layout.hands[0].root_control)
        scene.write_data_to_sim()
        sim.step()
        scene.update(.005)
    assert plant.advance_calls == 2 and plant.set_calls == 1
    passive = model.joint("hinge").id
    assert not np.any(model.actuator_trnid[:, 0] == passive)
    snapshot = plant.snapshot()
    plant.control[:] = 99
    plant.q[:] += .1
    plant.restore(snapshot)
    torch.testing.assert_close(plant.q, snapshot.physics[0], rtol=0, atol=0)
    torch.testing.assert_close(plant.control, snapshot.control, rtol=0, atol=0)


def test_direct_contact_reduction_uses_actual_body_pairs_and_worst_separation(tmp_path):
    _layout, _mapping, model = _mechanics(tmp_path)
    plant = object.__new__(MjWarpPlant)
    hand, base, lid, other = [model.body(n).id for n in ("left_root", "base", "lid", "right_root")]
    plant._init_buffers(model, 2, "cpu", {"left/base": ([hand], [base]), "left/lid": ([hand], [lid])})
    # Synthetic solved-contact protocol: two contacts on the required part,
    # one different part, and a high-force unrelated hand that must be excluded.
    geom_by_body = {body: int(np.flatnonzero(model.geom_bodyid == body)[0]) for body in (hand, base, lid, other)}
    geoms = torch.tensor([[geom_by_body[a], geom_by_body[b]] for a, b in
                          ((hand, base), (base, hand), (hand, lid), (other, base))] + [[-1, -1]])
    data = SimpleNamespace(contact=SimpleNamespace(geom=geoms, worldid=torch.tensor([0, 0, 1, 1, -1]),
                           dist=torch.tensor([.002, -.026, -.003, 0., 0.])), nacon=torch.tensor([4]))
    plant._wp = SimpleNamespace(to_torch=lambda value: value)
    plant._mjw = SimpleNamespace(contact_force=lambda *args: None)
    plant._model, plant._data = None, data
    plant._geom_body = torch.tensor(model.geom_bodyid)
    plant._contact_ids = torch.arange(5)
    plant._contact_forces = torch.zeros(5, 6)
    plant._contact_forces[:, 0] = torch.tensor([1., 2., .7, 100., 0.])
    assert not plant.contact_support.any() and torch.isnan(plant.contact_facts["left/base"]["normal_force_n"]).all()
    plant._read_solved_contacts()
    torch.testing.assert_close(plant.contact_facts["left/base"]["normal_force_n"], torch.tensor([3., 0.]))
    torch.testing.assert_close(plant.contact_facts["left/lid"]["normal_force_n"], torch.tensor([0., .7]))
    assert plant.contact_facts["left/base"]["force_bearing_distance_max_m"][0] == torch.tensor(.026)
    assert torch.isnan(plant.contact_facts["left/base"]["force_bearing_distance_max_m"][1])
    plant.invalidate_contacts(torch.tensor([0]))
    assert torch.isnan(plant.contact_facts["left/base"]["normal_force_n"][0])
    assert plant.contact_support.tolist() == [False, True]


def test_native_contact_archive_retains_source_phase_and_normal_reader(tmp_path):
    from dexforge_adaptation.flashdex_retarget import (
        flashdex_source_mapping,
        load_flashdex_rollout,
    )
    from test_native_rollout_record import record_reference_phase_fixture

    source = flashdex_source_mapping(
        source_id="synthetic", source_revision="protocol-only", source_unit_id="capture",
        source_episode_index=0, fold="train", source_frame_index=(0, 1, 2),
        source_time_ns=(1_000_000_000, 1_020_000_000, 1_040_000_000),
        selected_source_frame_index=(0, 1, 2), prepared_source_ordinal=(0., 1., 2.),
        prepared_elapsed_time_ns=(0, 20_000_000, 40_000_000),
        clock_authority="synthetic actual-phase protocol", processing={"fixture": True},
    )
    binding = dict(
        producer_identity={"method": "synthetic-not-physical"}, scene_identity={"fixture": True},
        scene_from_source_world=np.eye(4).tolist(),
        assistance_receipt={"evaluation_external_wrenches_cleared": True, "evaluation_object_playback": False},
        hands={s: {"body_name": s + "_root", "command_joint_names": {"driver": s + "_driver"},
                   "physical_joint_names": {"driver": s + "_driver"}} for s in ("left", "right")},
        object_part_names=["base", "lid"],
        object_parts={n: {"entity_name": "object", "body_name": b} for n, b in
                      (("base", "object_root"), ("lid", "object_lid"))},
        clips={"fixture": source},
    )

    def setup(env, command):
        env.scene.backend_identity = {"backend": "synthetic_port_protocol_not_physics"}
        env.scene.physics_time_s = lambda: torch.tensor(
            [7.125 + int(command.reference_phase["motion_steps"].item()) * .02], dtype=torch.float64)
        def contacts():
            row = int(command.reference_phase["motion_steps"].item())
            known = row > 0
            addresses = {f"{s}/{n}": dict(hand_side=s, hand_body_name=s + "_root",
                         part_entity_name="object", part_body_name=p["body_name"])
                         for s in ("left", "right") for n, p in binding["object_parts"].items()}
            return dict(addresses=addresses,
                pairs={name: {"normal_force_n": torch.tensor([float(row) if known else float("nan")]),
                              "force_bearing_contact_present": torch.tensor([known]),
                              "force_bearing_distance_max_m": torch.tensor([.003 if known else float("nan")])}
                       for name in addresses},
                measurement_support=torch.tensor([known]),
                sample_time_s=torch.tensor([(7.125 + row * .02 - .005) if known else float("nan")], dtype=torch.float64),
                clock="synthetic_last_integrated_substep_begin", normal_force_authority="synthetic protocol",
                distance_authority="synthetic protocol")
        env.scene.hand_part_contact_facts = contacts

    manifest = record_reference_phase_fixture(tmp_path, binding, native=True, setup_scene=setup)
    recorded = load_flashdex_rollout(manifest, motion_id=0)
    arrays = recorded.arrays
    np.testing.assert_array_equal(arrays["recorded_motion_steps"], [0, 1, 2])
    np.testing.assert_array_equal(arrays["command_motion_steps"], [1, 2, 2])
    assert arrays["hand_part_normal_force_n"].shape == (3, 4)
    assert np.isnan(arrays["hand_part_normal_force_n"][0]).all()
    np.testing.assert_array_equal(arrays["hand_part_measurement_support"][:, 0], [False, True, True])
    np.testing.assert_allclose(arrays["hand_part_sample_time_s"][1:], [7.14, 7.16])
    np.testing.assert_allclose(arrays["native_physics_time_s"], [7.125, 7.145, 7.165])
    assert arrays["native_physics_reset_time_s"] == 7.125
    assert arrays["native_hand_twist_w"].shape == (3, 2, 6)
    metadata = json.loads(str(arrays["hand_part_contact_metadata_json"]))
    assert len(metadata["pairs"]) == 4 and "not original" in metadata["provenance"]
    assert json.loads(str(arrays["native_backend_metadata_json"]))["backend"] == "synthetic_port_protocol_not_physics"
