"""CPU contracts for real backend seams; no Kit, GPU, learned model or integration."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest
import torch
from test_native_contract import _scene, _two_object_scene, write_native_scene_manifest

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

    from common.native_reference import bind_prepared_native_reference

    _, mapping = _scene(tmp_path)
    motion_file = write_native_scene_manifest(tmp_path, mapping)
    had = set(sys.modules)
    with patch.dict("os.environ", {}, clear=True), initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None,
    ):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
        for backend in ("mjwarp", "newton"):
            cfg = compose(config_name="native", overrides=[f"envs/simulator@simulator={backend}",
                f"commands.motion_file={motion_file}",
                "native_tracking_tolerances={root_position_error_m: 1, root_rotation_error_rad: 1, part_position_error_m: 1, part_rotation_error_rad: 1, joint_position_error_rad: 1, command_position_error_rad: 1}"])
            bind_prepared_native_reference(cfg)
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
    model.opt.timestep = .005
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
    def body_wrench(rows, *, body_ids, forces, torques):
        events.append("body_wrench")
        torch.testing.assert_close(rows, torch.arange(2))
        torch.testing.assert_close(body_ids, torch.arange(1, model.nbody))
        torch.testing.assert_close(forces[:, :, 0], torch.full((2, model.nbody - 1), 3.0))
        torch.testing.assert_close(torques[:, :, 2], torch.full((2, model.nbody - 1), -2.0))

    backend.get_physics_time_torch.side_effect = physics_time
    backend.set_body_wrench_w_torch.side_effect = body_wrench
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
    plant.external_body_wrench_w[:, 1:, 0] = 3.0
    plant.external_body_wrench_w[:, 1:, 5] = -2.0
    plant.advance()
    assert events == ["time", "body_wrench", "step"]
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


@pytest.mark.parametrize("dt", (.005, 1 / 240))
def test_newton_execution_clock_tracks_accepted_calls_partial_reset_and_restore(tmp_path, dt):
    _layout, _mapping, model = _mechanics(tmp_path)
    model.opt.timestep = dt
    plant = object.__new__(NewtonPlant)
    plant._init_buffers(model, 2, "cpu", {"left/base": ([1], [2])})
    plant.identity = {"backend": "newton", "fixture": "public backend protocol, not physical qualification"}
    plant.effort = torch.zeros(2, model.nv)
    plant._state = SimpleNamespace(qpos=torch.tensor(np.tile(model.qpos0, (2, 1)), dtype=torch.float32),
                                   qvel=torch.zeros(2, model.nv))
    plant._world_ids, plant._body_ids = torch.arange(2), torch.arange(1, model.nbody)
    raw_clock = torch.tensor([125., 7.], dtype=torch.float32)
    facts = dict(normal_force_n=torch.ones(2), force_bearing_contact_present=torch.ones(2, dtype=torch.bool),
                 force_bearing_distance_max_m=torch.full((2,), .001))
    calls = []
    def advance(control, *, nsteps):
        calls.append(plant.execution_clock()["steps"].clone())
        assert nsteps == 1
        raw_clock.add_(dt)
        return SimpleNamespace(public_state=plant._state, applied_joint_effort=torch.zeros(2, model.nv))
    def set_state(rows, *, qpos, qvel):
        plant._state.qpos[rows], plant._state.qvel[rows] = qpos, qvel
        raw_clock[rows] = 0
        return plant._state
    def restore(state, *, source_rows):
        raw_clock.copy_(state)
        return plant._state
    plant.backend = SimpleNamespace(model=model, get_physics_time_torch=lambda: raw_clock.clone(),
        set_body_wrench_w_torch=lambda *args, **kwargs: None, step_torch=advance,
        get_body_pair_contact_facts=lambda *args: facts, set_state_torch=set_state,
        get_default_qpos=lambda: model.qpos0.copy(), capture_full_physics_state=lambda: raw_clock.clone(),
        restore_full_physics_state_torch=restore)
    plant._bind_execution_clock(dt)
    plant.configure_control_interval(4)
    plant.contact_interval.begin()
    for _ in range(4):
        plant.advance()
    plant.contact_interval.finish()
    assert plant.execution_clock()["steps"].tolist() == [4, 4]
    assert len(calls) == 4 and calls[0].tolist() == [0, 0]
    interval = plant.contact_interval.facts()
    assert interval["execution_interval_complete"].all()
    torch.testing.assert_close(interval["execution_sample_begin_step"], torch.arange(4).expand(2, -1))
    metadata = plant.execution_clock()["metadata"]
    assert metadata["schema"] == "accepted_newton_fixed_step_clock_v1"
    assert metadata["configured_native_timestep_verified"] is True
    assert "configured_device_cast_verified" not in metadata
    snapshot = plant.snapshot()
    plant.advance()
    plant.restore(snapshot)
    assert plant.execution_clock()["steps"].tolist() == [4, 4]
    torch.testing.assert_close(raw_clock, snapshot.physics[0])
    assert not plant.contact_interval.facts()["execution_interval_complete"].any()
    plant.forward()
    plant.set_state(torch.tensor([1]), plant._state.qpos[1:].clone(), plant._state.qvel[1:].clone())
    assert plant.execution_clock()["steps"].tolist() == [4, 4]
    plant.reset(torch.tensor([1]))
    assert plant.execution_clock()["steps"].tolist() == [4, 0]
    assert raw_clock[0] == snapshot.physics[0][0]
    plant.contact_interval.begin()
    for _ in range(4):
        plant.record_contact_substep()
    plant.contact_interval.finish()
    assert not plant.contact_interval.facts()["execution_interval_complete"].any()
    def failed_step(*args, **kwargs):
        raise RuntimeError("fixture solver did not return")
    plant.backend.step_torch = failed_step
    with pytest.raises(RuntimeError, match="did not return"):
        plant.advance()
    assert plant.execution_clock()["steps"].tolist() == [4, 0]


def test_newton_execution_clock_rejects_a_different_installed_native_timestep():
    plant = object.__new__(NewtonPlant)
    plant.backend = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)))
    with pytest.raises(ValueError, match="installed fixed native timestep"):
        plant._bind_execution_clock(.005)


@pytest.mark.parametrize("dt", (.005, 1 / 240))
def test_mjwarp_execution_clock_tracks_actual_calls_reset_restore_and_missing_step(tmp_path, dt):
    """Real clock/advance owners; only the external solver operations are stubs."""
    _layout, _mapping, model = _mechanics(tmp_path)
    plant = object.__new__(MjWarpPlant)
    plant._init_buffers(model, 2, "cpu", {"left/base": ([1], [2])})
    plant.identity = {"backend": "mjwarp", "fixture": "solver protocol, not physics"}
    plant._scope = nullcontext
    plant._wp = SimpleNamespace(to_torch=lambda value: value, from_torch=lambda value: value)
    plant._model = SimpleNamespace(opt=SimpleNamespace(timestep=torch.tensor([dt], dtype=torch.float32)))
    plant._bind_execution_clock(dt)
    plant._data = SimpleNamespace(time=torch.tensor([25., 7.], dtype=torch.float32), overflow=torch.zeros(2, dtype=torch.int32))
    plant._qpos = torch.tensor(np.tile(model.qpos0, (2, 1)), dtype=torch.float32)
    plant._qvel = torch.zeros(2, model.nv)
    plant._ctrl, plant._xfrc_applied = torch.zeros_like(plant.control), torch.zeros_like(plant.external_body_wrench_w)
    plant._solved_effort = torch.zeros(2, model.nv)
    plant.effort = torch.zeros_like(plant._solved_effort)
    plant._reset_mask = torch.zeros(2, dtype=torch.bool)
    plant._integration = torch.empty(2, 1)
    calls = []

    def step(_model, data):
        calls.append(plant.execution_clock()["steps"].clone())
        data.time.add_(plant._model.opt.timestep[0])

    def reset(_model, data, *, reset):
        data.time[reset] = 0

    def get_state(_model, data, target, _kind):
        target.copy_(data.time[:, None])

    def set_state(_model, data, source, _kind):
        data.time.copy_(source[:, 0])

    plant._mjw = SimpleNamespace(step=step, reset_data=reset, get_state=get_state, set_state=set_state,
                                 State=SimpleNamespace(INTEGRATION=1))
    plant.forward = lambda: None

    def solved_contacts():
        plant.contact_support[:] = True
        facts = plant.contact_facts["left/base"]
        facts["normal_force_n"][:] = 2.
        facts["force_bearing_contact_present"][:] = True
        facts["force_bearing_distance_max_m"][:] = .001

    plant._read_solved_contacts = solved_contacts
    sim = NativeSim(plant)
    plant.configure_control_interval(4)
    sim.begin_control_step()
    raw_before = plant.physics_time()
    for _ in range(4):
        sim.step()
    sim.end_control_step()
    assert len(calls) == 4 and calls[0].tolist() == [0, 0] and calls[-1].tolist() == [3, 3]
    facts = plant.contact_interval.facts()
    torch.testing.assert_close(facts["execution_sample_begin_step"], torch.arange(4).expand(2, -1), rtol=0, atol=0)
    assert facts["execution_interval_complete"].all()
    torch.testing.assert_close(facts["pairs"]["left/base"]["normal_force_n"], torch.full((2, 4), 2.))
    torch.testing.assert_close(facts["sample_begin_time_s"][:, 0], raw_before.double(), rtol=0, atol=0)
    assert plant.physics_time().dtype == torch.float32
    if dt == .005:
        assert int(torch.round((plant.physics_time()[0].double() - raw_before[0].double()) * 1e9)) != 20_000_000
    metadata = plant.execution_clock()["metadata"]
    assert metadata["configured_timestep_s"] == dt and metadata["configured_device_cast_verified"] is True
    assert metadata["device_timestep_s"] == float(torch.tensor(dt, dtype=torch.float32))
    scene = object.__new__(NativeScene)
    scene.sim = sim
    assert scene.physics_execution_clock()["steps"].tolist() == [4, 4]

    snapshot = plant.snapshot()
    sim.step()
    plant.restore(snapshot)
    torch.testing.assert_close(plant.physics_time(), snapshot.physics[0][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(plant.execution_clock()["steps"], snapshot.execution_steps, rtol=0, atol=0)
    torch.testing.assert_close(plant.contact_sample_execution_step, snapshot.contact_sample_execution_step, rtol=0, atol=0)
    assert not plant.contact_interval.facts()["execution_interval_complete"].any()
    sim.forward()
    plant.set_state(torch.tensor([1]), plant._qpos[1:].clone(), plant._qvel[1:].clone())
    assert plant.execution_clock()["steps"].tolist() == [4, 4]
    sim.reset(torch.tensor([1]))
    assert plant.execution_clock()["steps"].tolist() == [4, 0]
    assert plant.physics_time()[0] == snapshot.physics[0][0, 0] and plant.physics_time()[1] == 0

    # Re-reading the last solve four times is not four new accepted advances.
    sim.begin_control_step()
    for _ in range(4):
        plant.record_contact_substep()
    sim.end_control_step()
    assert not plant.contact_interval.facts()["execution_interval_complete"].any()


def test_mjwarp_execution_clock_rejects_an_actual_device_timestep_mismatch():
    plant = object.__new__(MjWarpPlant)
    plant.num_envs, plant.device, plant.identity = 2, "cpu", {}
    plant._wp = SimpleNamespace(to_torch=lambda value: value)
    plant._model = SimpleNamespace(opt=SimpleNamespace(timestep=torch.tensor([.006], dtype=torch.float32)))
    with pytest.raises(ValueError, match="installed fixed timestep cast"):
        plant._bind_execution_clock(.005)


def _mjwarp_overflow_fixture(overflow_rows):
    """Run actual advance/error owners with the SDK's public flag enum only."""
    from mujoco_warp import OverflowType

    plant = object.__new__(MjWarpPlant)
    model = SimpleNamespace(nu=1, nv=1, nbody=2, opt=SimpleNamespace(gravity=(0., 0., -9.81)))
    plant._init_buffers(model, 3, "cpu", {"hand/part": ([0], [1])})
    plant._scope = nullcontext
    plant._wp = SimpleNamespace(to_torch=lambda value: value)
    plant._model = SimpleNamespace(opt=SimpleNamespace(timestep=torch.tensor([.005], dtype=torch.float32)))
    plant.identity = {"backend": "mjwarp", "nconmax": 512, "njmax": 2048}
    plant._bind_execution_clock(.005)
    plant._data = SimpleNamespace(
        time=torch.zeros(3), overflow=torch.zeros(3, dtype=torch.int32),
        naconmax=1536, naccdmax=1536, njmax=2048, njmax_nnz=8192, nvmax=6,
        nacon=torch.tensor([3]), ncollision=torch.tensor([4]),
        nefc=torch.tensor([2, 3, 5]), solver_niter=torch.tensor([1, 3, 5]),
    )
    plant._ctrl = torch.zeros_like(plant.control)
    plant._xfrc_applied = torch.zeros_like(plant.external_body_wrench_w)
    plant._solved_effort, plant.effort = torch.full((3, 1), 9.), torch.full((3, 1), 2.)
    pending = iter(overflow_rows)
    events = []

    def step(_model, data):
        events.append("step")
        data.time.add_(.005)
        data.overflow.copy_(torch.tensor(next(pending), dtype=torch.int32))

    def contacts():
        events.append("contacts")
        plant.contact_support[:] = True
        plant.contact_facts["hand/part"]["normal_force_n"][:] = 2.
        plant.contact_facts["hand/part"]["force_bearing_contact_present"][:] = True
        plant.contact_facts["hand/part"]["force_bearing_distance_max_m"][:] = .001

    plant._mjw = SimpleNamespace(step=step, OverflowType=OverflowType)
    plant._read_solved_contacts = contacts
    plant.forward = lambda: events.append("forward")
    return plant, events


@pytest.mark.parametrize("flag_names", [
    ("EPA_HORIZON",), ("NEFC",), ("NARROWPHASE",),
    ("EPA_HORIZON", "NEFC", "ITERATIONS", "LS_ITERATIONS"),
], ids=["epa", "constraint", "contact", "mixed"])
def test_mjwarp_overflow_preserves_sdk_flags_and_unaccepted_step(flag_names):
    from mujoco_warp import OverflowType

    bits = sum(int(OverflowType[name]) for name in flag_names)
    plant, events = _mjwarp_overflow_fixture([[0, bits, 0]])
    plant._execution_steps.copy_(torch.tensor([4, 8, 12]))
    if "NEFC" in flag_names:
        plant._data.nefc[1] = 2049
    plant.configure_control_interval(2)
    # A direct call outside a manager interval must discard the prior tape.
    plant.contact_interval.sample_recorded[:] = True
    plant.contact_interval.interval_complete[:] = True
    with patch.object(torch, "_assert_async", side_effect=AssertionError("must not poison CUDA")) as destructive:
        with pytest.raises(RuntimeError, match="MJWarp overflow: ") as failure:
            plant.advance()
    destructive.assert_not_called()
    witness = json.loads(str(failure.value).split("MJWarp overflow: ", 1)[1])
    assert witness["raw_bits"] == [0, bits, 0] and witness["affected_worlds"] == [1]
    assert witness["world_flags"] == [dict(world=1, raw_bits=bits,
        flags=[flag.name for flag in OverflowType if bits & int(flag)], unknown_bits=0)]
    assert witness["accepted_steps_before_attempt"] == [4, 8, 12]
    assert witness["attempted_execution_steps"] == [5, 9, 13]
    assert witness["control_interval"] == dict(active=False, accepted_substeps=0,
        attempted_substep=None, configured_substeps=2)
    assert witness["configured_capacities"] == dict(nconmax=512, njmax=2048)
    assert witness["resolved_capacities"] == dict(naconmax=1536, naccdmax=1536,
        njmax=2048, njmax_nnz=8192, nvmax=6)
    assert witness["counts"] == dict(nacon=[3], ncollision=[4],
        nefc=[2, 2049 if "NEFC" in flag_names else 3, 5], solver_niter=[1, 3, 5])
    assert witness["failed_attempt_accepted"] is False
    assert plant._execution_steps.tolist() == [4, 8, 12] and events == ["step"]
    assert torch.equal(plant.effort, torch.full((3, 1), 2.))
    assert not plant.contact_support.any()
    assert torch.isnan(plant.contact_sample_time_s).all()
    assert plant.contact_sample_execution_step.tolist() == [-1, -1, -1]
    assert not plant.contact_interval.sample_recorded.any()
    assert not plant.contact_interval.interval_complete.any()
    # The retained ordinary exception no longer depends on mutable SDK flags.
    plant._data.overflow.zero_()
    assert json.loads(str(failure.value).split("MJWarp overflow: ", 1)[1]) == witness


def test_mjwarp_overflow_preserves_only_the_accepted_manager_prefix():
    from mujoco_warp import OverflowType

    bits = int(OverflowType.EPA_HORIZON)
    plant, events = _mjwarp_overflow_fixture([[0, 0, 0], [bits, 0, bits]])
    plant.configure_control_interval(3)
    sim = NativeSim(plant)
    sim.begin_control_step()
    sim.step()
    with pytest.raises(RuntimeError, match="MJWarp overflow: ") as failure:
        sim.step()
    witness = json.loads(str(failure.value).split("MJWarp overflow: ", 1)[1])
    assert witness["affected_worlds"] == [0, 2]
    assert witness["control_interval"] == dict(active=True, accepted_substeps=1,
        attempted_substep=2, configured_substeps=3)
    assert plant._execution_steps.tolist() == [1, 1, 1]
    assert events == ["step", "contacts", "forward", "step"]
    assert plant.contact_sample_execution_step.tolist() == [0, 0, 0]
    tape = plant.contact_interval.facts()
    assert tape["sample_recorded"].tolist() == [[True, False, False]] * 3
    assert not tape["interval_complete"].any()
    assert not tape["execution_interval_complete"].any()
    # The solver attempted the second step; no invented rollback or accepted count.
    torch.testing.assert_close(plant.physics_time(), torch.full((3,), .01))


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
        self.external_body_wrench_w[env_ids] = 0
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
    layout, _ = _two_object_scene(tmp_path)
    model = mujoco.MjModel.from_xml_path(layout.scene_xml_path)
    plant = object.__new__(MjWarpPlant)
    hand, base, lid, other, prop = [model.body(n).id for n in ("left_root", "base", "lid", "right_root", "prop_root")]
    plant._init_buffers(model, 2, "cpu", {"left/base": ([hand], [base]), "left/lid": ([hand], [lid]),
                                          "object_pair": ([base, lid], [prop])})
    # Synthetic solved-contact protocol: two contacts on the required part,
    # one different part, and a high-force unrelated hand that must be excluded.
    geom_by_body = {body: int(np.flatnonzero(model.geom_bodyid == body)[0]) for body in (hand, base, lid, other, prop)}
    body_pairs = ((hand, base), (base, hand), (hand, lid), (other, base), (prop, base), (lid, prop))
    geoms = torch.tensor([[geom_by_body[a], geom_by_body[b]] for a, b in body_pairs] + [[-1, -1]])
    data = SimpleNamespace(contact=SimpleNamespace(geom=geoms, worldid=torch.tensor([0, 0, 1, 1, 0, 0, -1]),
                           dist=torch.tensor([.002, -.026, -.003, 0., -.008, -.019, 0.])), nacon=torch.tensor([6]))
    plant._wp = SimpleNamespace(to_torch=lambda value: value)
    plant._mjw = SimpleNamespace(contact_force=lambda *args: None)
    plant._model, plant._data = None, data
    plant._geom_body = torch.tensor(model.geom_bodyid)
    plant._contact_ids = torch.arange(len(geoms))
    plant._contact_forces = torch.zeros(len(geoms), 6)
    plant._contact_forces[:, 0] = torch.tensor([1., 2., .7, 100., 0.1, 0.6, 0.])
    assert not plant.contact_support.any() and torch.isnan(plant.contact_facts["left/base"]["normal_force_n"]).all()
    plant._read_solved_contacts()
    torch.testing.assert_close(plant.contact_facts["left/base"]["normal_force_n"], torch.tensor([3., 0.]))
    torch.testing.assert_close(plant.contact_facts["left/lid"]["normal_force_n"], torch.tensor([0., .7]))
    torch.testing.assert_close(plant.contact_facts["object_pair"]["normal_force_n"], torch.tensor([.7, 0.]))
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
    from test_native_task_contract import _add_loose_object

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
        object_part_names=["base", "lid", "prop"],
        object_parts={name: {"entity_name": entity, "body_name": body} for name, entity, body in (
            ("base", "object", "object_root"), ("lid", "object", "object_lid"), ("prop", "loose_prop", "prop_root"))},
        clips={"fixture": source},
    )

    def setup(env, command):
        _add_loose_object(command, env, [])
        env.scene.backend_identity = {"backend": "synthetic_port_protocol_not_physics"}
        env.scene.physics_time_s = lambda: torch.tensor(
            [7.125 + int(command.reference_phase["motion_steps"].item()) * .02], dtype=torch.float64)
        env.scene.physics_execution_clock = lambda: dict(
            steps=torch.tensor([40 + int(command.reference_phase["motion_steps"].item()) * 4], dtype=torch.int64),
            metadata=dict(schema="accepted_mjwarp_fixed_step_clock_v1", configured_timestep_s=.005,
                device_timestep_s=float(torch.tensor(.005, dtype=torch.float32)), device_timestep_dtype="torch.float32",
                configured_device_cast_verified=True, authority="synthetic recording protocol, not physics"))
        def contacts():
            row = int(command.reference_phase["motion_steps"].item())
            known = row > 0
            addresses = {f"{s}/{n}": dict(hand_side=s, hand_body_name=s + "_root",
                         part_entity_name=p["entity_name"], part_body_name=p["body_name"])
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

        def interval():
            last = contacts()
            row = int(command.reference_phase["motion_steps"].item())
            known = row > 0
            start = 40 + (row - 1) * 4 + torch.arange(4)[None]
            raw = 7.125 + (row - 1) * .02 + torch.arange(4, dtype=torch.float64)[None] * .005
            return {**last, "num_substeps": 4, "sampling": "synthetic every-substep protocol",
                "pairs": {name: {key: value[:, None].expand(-1, 4).clone() for key, value in facts.items()}
                          for name, facts in last["pairs"].items()},
                "measurement_support": torch.full((1, 4), known), "sample_recorded": torch.full((1, 4), known),
                "clock_support": torch.full((1, 4), known), "interval_complete": torch.tensor([known]),
                "sample_begin_time_s": raw if known else torch.full_like(raw, float("nan")),
                "sample_end_time_s": raw + .005 if known else torch.full_like(raw, float("nan")),
                "execution_clock": env.scene.physics_execution_clock()["metadata"],
                "execution_sample_begin_step": start if known else torch.full_like(start, -1),
                "execution_sample_end_step": start + 1 if known else torch.full_like(start, -1),
                "execution_clock_support": torch.full((1, 4), known),
                "execution_interval_complete": torch.tensor([known])}

        env.scene.hand_part_contact_interval_facts = interval
        key = next(iter(command.native_layout.object_pair_bindings))

        def object_interval():
            row = int(command.reference_phase["motion_steps"].item())
            forces = (0., 1.5, 0.25, 0.) if row == 1 else (0., 0., 0., 0.)
            return {**interval(), "pairs": {key: {"normal_force_n": torch.tensor([forces])}}}

        receiver = SimpleNamespace(
            sim=SimpleNamespace(plant=SimpleNamespace(contact_interval=SimpleNamespace(facts=object_interval))),
            _object_pair_contact_addresses={key: dict(first_entity_name="object", first_root_body_name="object_root",
                second_entity_name="loose_prop", second_root_body_name="prop_root")})
        env.scene.object_pair_contact_interval_facts = lambda: NativeScene.object_pair_contact_interval_facts(receiver)

    manifest = record_reference_phase_fixture(tmp_path, binding, native=True, setup_scene=setup)
    recorded = load_flashdex_rollout(manifest, motion_id=0)
    arrays = recorded.arrays
    np.testing.assert_array_equal(arrays["recorded_motion_steps"], [0, 1, 2])
    np.testing.assert_array_equal(arrays["command_motion_steps"], [1, 2, 2])
    assert arrays["hand_part_normal_force_n"].shape == (3, 6)
    assert np.isnan(arrays["hand_part_normal_force_n"][0]).all()
    np.testing.assert_array_equal(arrays["hand_part_measurement_support"][:, 0], [False, True, True])
    np.testing.assert_allclose(arrays["hand_part_sample_time_s"][1:], [7.14, 7.16])
    np.testing.assert_allclose(arrays["native_physics_time_s"], [7.125, 7.145, 7.165])
    assert arrays["native_physics_reset_time_s"] == 7.125
    np.testing.assert_array_equal(arrays["native_execution_steps"], [40, 44, 48])
    assert json.loads(str(arrays["native_execution_clock_metadata_json"]))["configured_timestep_s"] == .005
    np.testing.assert_array_equal(arrays["hand_part_interval_execution_sample_begin_step"][1], [40, 41, 42, 43])
    assert not arrays["hand_part_interval_execution_interval_complete"][0]
    assert arrays["hand_part_interval_execution_interval_complete"][1:].all()
    assert arrays["native_hand_twist_w"].shape == (3, 2, 6)
    metadata = json.loads(str(arrays["hand_part_contact_metadata_json"]))
    assert len(metadata["pairs"]) == 6 and "not original" in metadata["provenance"]
    pair = json.loads(str(arrays["object_pair_interval_metadata_json"]))
    assert pair["schema"] == "solver_object_pair_control_interval_v1"
    assert pair["pairs"] == [dict(first_entity_name="object", first_root_body_name="object_root",
        second_entity_name="loose_prop", second_root_body_name="prop_root")]
    np.testing.assert_array_equal(arrays["object_pair_interval_measurement_support"][:, 0], [False, True, True])
    np.testing.assert_allclose(arrays["object_pair_interval_peak_normal_force_n"][:, 0], [np.nan, 1.5, 0.])
    np.testing.assert_allclose(arrays["object_pair_interval_normal_force_n"][:, :, 0],
                               [[np.nan] * 4, [0., 1.5, 0.25, 0.], [0.] * 4])
    assert json.loads(str(arrays["native_backend_metadata_json"]))["backend"] == "synthetic_port_protocol_not_physics"


# Control-context cases use real native/manager owners with a synthetic plant.
# They do not certify an SDK/GPU replay or integrate a physical MuJoCo scene.
def _initial_control_case(tmp_path, *, wrist_target_twist="source_only"):
    from mjlab.managers.action_manager import ActionManager
    from mjlab.managers.command_manager import CommandManager
    from mjlab.managers.event_manager import EventManager, EventTermCfg
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationManager,
        ObservationTermCfg,
    )
    from mjlab.managers.reward_manager import RewardManager, RewardTermCfg

    from envs.actions.native import NativeResidualActionCfg
    from envs.commands.motion_tracking.native import NativeMotionTrackingCommand
    from evaluation.shared import _NullTerminationManager
    from simulator.env import ManagerBasedRlEnv, _NullManager

    layout, mapping, model = _mechanics(tmp_path)

    class ContextPlant(_ProtocolPlant):
        def capture_physics(self):
            return (*super().capture_physics(), self.effort.clone())

        def restore_physics(self, state):
            super().restore_physics(state[:3])
            self.effort.copy_(state[3])

        def body_state(self, body_ids):
            # Explicit synthetic body facts, not FK or physical qualification.
            shape = (2, len(body_ids))
            position = self.q[:, None, :3].expand(*shape, 3).clone()
            quaternion = torch.zeros(*shape, 4)
            quaternion[..., 0] = 1
            return position, quaternion, torch.zeros(*shape, 3), torch.zeros(*shape, 3)

        def advance(self):
            self.prepare_contact_substep()
            self.contact_sample_time_s.copy_(self._time)
            super().advance()
            self.q[:, -1] += self.control[:, 0] * .01
            self.effort[:, -1] = self.control[:, 0]
            self.record_contact_substep()

    plant = ContextPlant(model)
    plant.configure_control_interval(4)
    sim = NativeSim(plant)
    scene = NativeScene(layout, load_robot_mjcf_meta(layout.scene_xml_path), sim, "robot")
    env = object.__new__(ManagerBasedRlEnv)
    env.scene, env.sim, env._device = scene, sim, "cpu"
    env.cfg = SimpleNamespace(decimation=4, physics_dt=.005, auto_reset=False, episode_length_s=5.)
    env._sim_step_counter = env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env._manual_reset_pending = torch.zeros(2, dtype=torch.bool)
    env.extras, env.obs_buf = {}, {}
    env.metrics_manager = env.recorder_manager = env.curriculum_manager = _NullManager()
    env.termination_manager = _NullTerminationManager(2, "cpu")
    command = object.__new__(NativeMotionTrackingCommand)
    command._env, command.native_layout, command.robot = env, layout, scene["robot"]
    command._side_list = [hand.side for hand in layout.hands]
    command._root_body_ids = [command.robot.body_names.index(hand.root_body_name) for hand in layout.hands]
    ids, _ = command.robot.find_joints_by_actuator_names(tuple(a for hand in layout.hands for a in hand.command_actuator_names))
    command._finger_joint_ids = torch.tensor(ids)
    command._refn_enabled = False
    command._eval_mode, command._park_inactive_on_ground = True, False
    command.motion_ids = torch.zeros(2, dtype=torch.long)
    command.motion_steps = torch.zeros(2, dtype=torch.long)
    command.time_left = torch.zeros(2)
    command.command_counter = torch.zeros(2, dtype=torch.long)
    command.metrics = {}
    roots = {hand.side: scene[hand.entity_name]._default_root_state[:1].repeat(3, 1) for hand in layout.hands}
    parts = torch.zeros(3, len(layout.objects[0].parts), 13)
    parts[..., 3] = 1
    command.motion_lib = SimpleNamespace(
        num_trajectories=1, length_starts=torch.tensor([0]), _motion_num_frames=torch.tensor([3]),
        native_time_ns=torch.arange(3, dtype=torch.int64) * 20_000_000,
        robot_joint_pos=torch.zeros(3, command.robot.num_joints), robot_joint_vel=torch.zeros(3, command.robot.num_joints),
        native_joint_position_support=torch.ones(3, command.robot.num_joints, dtype=torch.bool),
        native_joint_velocity_support=torch.ones(3, command.robot.num_joints, dtype=torch.bool),
        native_root_body_names={hand.side: hand.root_body_name for hand in layout.hands},
        native_root_state_w=roots, native_root_pose_support={side: torch.ones(3, dtype=torch.bool) for side in roots},
        native_root_linear_velocity_support={side: torch.ones(3, dtype=torch.bool) for side in roots},
        native_root_angular_velocity_support={side: torch.ones(3, dtype=torch.bool) for side in roots},
        native_object_root_body_names=layout.object_root_body_names,
        native_object_part_body_names=tuple(part.body_name for part in layout.objects[0].parts),
        native_object_joint_names=tuple(scene[layout.objects[0].entity_name].joint_names),
        native_object_joint_pos=torch.zeros(3, scene[layout.objects[0].entity_name].num_joints),
        native_object_joint_vel=torch.zeros(3, scene[layout.objects[0].entity_name].num_joints),
        native_object_joint_position_support=torch.ones(3, scene[layout.objects[0].entity_name].num_joints, dtype=torch.bool),
        native_object_joint_velocity_support=torch.ones(3, scene[layout.objects[0].entity_name].num_joints, dtype=torch.bool),
        native_object_part_state_w=parts, native_object_part_pose_support=torch.ones(3, len(layout.objects[0].parts), dtype=torch.bool),
        native_object_part_linear_velocity_support=torch.ones(3, len(layout.objects[0].parts), dtype=torch.bool),
        native_object_part_angular_velocity_support=torch.ones(3, len(layout.objects[0].parts), dtype=torch.bool),
    )
    command.cfg = SimpleNamespace(sampling=SimpleNamespace(mode="start", start_frame=0),
        resampling_time_range=(1000., 1000.), hand=SimpleNamespace(zero_init_vel=False),
        object=SimpleNamespace(zero_init_vel=False, pin_objects=False,
                               entity_names={side: layout.objects[0].entity_name for side in roots}))
    command._bind_native_reference()
    command._init_metrics()
    command._init_object_assist()
    command._init_native_object_assist()
    env.command_manager = object.__new__(CommandManager)
    env.command_manager._terms = {"tracking": command}
    env.action_manager = ActionManager({"native": NativeResidualActionCfg(
        entity_name="robot", command_name="tracking", native_scene=mapping,
        translation_residual_m=.1, rotation_residual_rad=1.,
        wrist_target_twist=wrist_target_twist,
        cutoff_hz={"wrist_trans": 5., "wrist_rot": 5.}, clip={".*": (-1., 1.)},
    )}, env)
    env.event_manager = EventManager({"reset": EventTermCfg(func=lambda env, env_ids: None, mode="reset")}, env)
    env.reward_manager = RewardManager({"action": RewardTermCfg(
        func=lambda env: env.action_manager.action.square().sum(-1), weight=.3)}, env)
    env.observation_manager = ObservationManager({"actor": ObservationGroupCfg(terms={"state": ObservationTermCfg(
        func=lambda env: torch.cat((env.sim.plant.q, env.action_manager.action), -1))})}, env)
    from mjlab.utils.spaces import Box

    env.action_space = Box(low=-1., high=1., shape=(env.action_manager.total_action_dim,))
    env.reset(seed=39)
    return env, command


def _assert_control_values_equal(actual, expected):
    from dataclasses import fields, is_dataclass

    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(actual):
        for field in fields(actual):
            _assert_control_values_equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_control_values_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_control_values_equal(left, right)
    else:
        assert actual == expected


def test_native_scene_control_context_restores_pending_targets_and_invalidates_reads(tmp_path):
    env, _ = _initial_control_case(tmp_path)
    sim, scene, plant = env.sim, env.scene, env.sim.plant
    hand = scene[scene.native_layout.hands[0].entity_name]
    rows = torch.tensor([1])
    q, v = sim.pending_state(rows)
    q[:, :3] = torch.tensor([[.2, .3, .4]])
    v[:, :3] = .5
    hand._joint_pos_target[1] = .27
    context = scene.capture_control_context()
    snapshot_target = context.joint_position_targets[hand.binding.entity_name].clone()
    cached = hand.data.body_link_pos_w.clone()
    scene.write_data_to_sim()
    hand._joint_pos_target[:] = .8
    assert not torch.equal(hand.data.body_link_pos_w, cached)
    after_write = plant.set_calls
    scene.restore_control_context(context)
    assert plant.set_calls == after_write  # no reset or state-write substitute
    _assert_control_values_equal(sim._pending, context.simulation.pending)
    _assert_control_values_equal(hand._joint_pos_target, snapshot_target)
    _assert_control_values_equal(hand.data.body_link_pos_w, cached)
    scene.write_data_to_sim()
    _assert_control_values_equal(plant.q[rows, :3], torch.tensor([[.2, .3, .4]]))
    # Restore copies, so using the context again cannot inherit a mutated pending write.
    scene.restore_control_context(context)
    sim._pending[1][:] = 77
    assert not bool((context.simulation.pending[1] == 77).any())
    sim._pending = None
    sim.begin_control_step()
    with pytest.raises(RuntimeError, match="closed control interval"):
        scene.capture_control_context()


def test_native_initial_control_context_repeats_real_manager_lpf_phase_and_rng(tmp_path):
    import random

    from common.utils import capture_rng_state

    env, command = _initial_control_case(tmp_path)
    action = env.action_manager.get_term("native")
    assert command.reference_phase["motion_steps"].tolist() == [0, 0]
    assert command.motion_steps.tolist() == [1, 1]
    initial = env.capture_initial_control_context()
    raw = torch.linspace(-.8, .8, action.action_dim).repeat(2, 1)

    def run_two_steps():
        values = []
        for requested in (raw, -raw):
            obs, reward, terminated, time_out, _ = env.step(requested)
            values.append((env.scene.capture_control_context(), action.capture_control_context(),
                           command.capture_control_context(), obs, reward.clone(), terminated.clone(), time_out.clone(),
                           env.action_manager.prev_action.clone(), env.action_manager.prev_prev_action.clone()))
        return values, (random.random(), np.random.rand(), torch.rand(3)), capture_rng_state(None)

    first = run_two_steps()
    assert not torch.equal(action._processed_actions, -raw)  # real second-step LPF carry
    assert command.reference_phase["motion_steps"].tolist() == [2, 2]
    assert env.common_step_counter == 2 and env._sim_step_counter == 8
    with pytest.raises(RuntimeError, match="reset boundary"):
        env.capture_initial_control_context()
    command._source_tip_geometry_cache = {"stale": torch.tensor(99)}
    command._source_tip_geometry_token = (0, initial.commands["tracking"].object_state_version)
    reset_writes = env.sim.plant.set_calls
    env.restore_initial_control_context(initial)
    assert env.sim.plant.set_calls == reset_writes
    assert "_source_tip_geometry_cache" not in vars(command)
    assert command.reference_phase["motion_steps"].tolist() == [0, 0]
    assert command.motion_steps.tolist() == [1, 1]
    assert bool(action._filter_pending.all()) and not bool(action._installed_valid.any())
    assert not hasattr(action, "ref_joint_pos_used")
    assert not hasattr(env, "reward_buf")
    _assert_control_values_equal(env.capture_initial_control_context(), initial)
    repeated = run_two_steps()
    _assert_control_values_equal(repeated, first)
    assert env.common_step_counter == 2 and env._sim_step_counter == 8
    assert env.sim.plant.advance_calls == 16  # only the two ordinary 2×4-step arms


def test_native_wrist_target_twist_repeats_real_manager_substeps_and_context(tmp_path, monkeypatch):
    env, _command = _initial_control_case(
        tmp_path, wrist_target_twist="filtered_correction_feedforward"
    )
    action = env.action_manager.get_term("native")
    installed = {side: [] for side in action.hand_sides}
    for side in action.hand_sides:
        hand = action._entity.hands[side]
        original = hand.apply_root_pose_target

        def record(target, support, control, *, side=side, original=original):
            installed[side].append(target.clone())
            return original(target, support, control)

        monkeypatch.setattr(hand, "apply_root_pose_target", record)
    initial = env.capture_initial_control_context()
    first_action = torch.zeros_like(action.raw_action)
    second_action = torch.linspace(-.6, .6, action.action_dim).repeat(env.num_envs, 1)

    def run():
        for values in installed.values():
            values.clear()
        env.step(first_action)
        assert not action._correction_interval_valid.any()
        env.step(second_action)
        assert action._correction_interval_valid.all()
        assert action._correction_twist.abs().sum() > 0
        for values in installed.values():
            assert len(values) == 2 * env.cfg.decimation
            for begin in (0, env.cfg.decimation):
                for offset in range(1, env.cfg.decimation):
                    torch.testing.assert_close(values[begin + offset], values[begin], atol=0, rtol=0)
        return ({side: torch.stack(values) for side, values in installed.items()},
                action.capture_control_context(), env.scene.capture_control_context())

    first = run()
    held = action.capture_control_context()
    action.process_actions(-second_action)
    action.restore_control_context(held)
    _assert_control_values_equal(action.capture_control_context(), held)
    env.restore_initial_control_context(initial)
    assert not action._correction_interval_valid.any()
    assert not action._correction_twist.any()
    repeated = run()
    _assert_control_values_equal(repeated, first)


def test_native_initial_control_context_refuses_unowned_history_and_backend(tmp_path):
    env, _ = _initial_control_case(tmp_path)
    for field in ("_group_obs_term_history_buffer", "_group_obs_term_delay_buffer"):
        buffers = getattr(env.observation_manager, field)
        buffers["actor"]["state"] = object()
        with pytest.raises(NotImplementedError, match="delay/history"):
            env.capture_initial_control_context()
        buffers["actor"].clear()
    env.sim.begin_control_step()
    with pytest.raises(RuntimeError, match="no executed control interval"):
        env.capture_initial_control_context()
    native_scene = env.scene
    env.scene = SimpleNamespace(num_envs=2)
    with pytest.raises(NotImplementedError, match="NativePlant scenes"):
        env.capture_initial_control_context()
    env.scene = native_scene



def test_native_context_rng_keeps_real_gym_sampler_separate(tmp_path):
    import random

    from gymnasium.spaces import Box

    from common.utils import capture_rng_state, restore_rng_state

    env, _ = _initial_control_case(tmp_path)
    initial = env.capture_initial_control_context()
    assert initial.rng["action_space"] is None
    assert not hasattr(env.action_space, "np_random")  # Real mjlab metadata Box.
    sampler = Box(low=-1., high=1., shape=(3,), dtype=np.float32)
    sampler.seed(17)
    saved = capture_rng_state(sampler)
    expected = (random.random(), np.random.rand(), torch.rand(3), sampler.sample())
    restore_rng_state(saved, sampler)
    actual = (random.random(), np.random.rand(), torch.rand(3), sampler.sample())
    _assert_control_values_equal(actual, expected)
    # The manager restores only its globals; it never fabricates a sampler.
    env.restore_initial_control_context(initial)
    _assert_control_values_equal(capture_rng_state(None), initial.rng)


def test_native_factory_keeps_multiple_objects_and_duplicate_local_part_labels(tmp_path, monkeypatch):
    """Actual factory/scene with a CPU state protocol, no integration or SDK."""
    from omegaconf import OmegaConf

    from simulator.native import create as native_create

    layout, mapping = _two_object_scene(tmp_path)
    document = ET.parse(layout.scene_xml_path)
    for hand in layout.hands:
        for axis, gear in enumerate(np.eye(6, dtype=int)):
            ET.SubElement(document.getroot().find("actuator"), "motor",
                name=f"{hand.side}_wrench_{axis}", jointinparent=hand.root_joint_name,
                gear=" ".join(map(str, gear)))
    document.write(layout.scene_xml_path)

    class FactoryPlant(_ProtocolPlant):
        def __init__(self, *, scene_xml_path, num_envs, sim_dt, device, contact_pairs):
            assert num_envs == 2 and sim_dt == .005 and device == "cpu"
            super().__init__(mujoco.MjModel.from_xml_path(scene_xml_path))
            self._init_buffers(self.model, num_envs, device, contact_pairs)

        def body_state(self, body_ids):
            shape = (self.num_envs, len(body_ids))
            position = torch.zeros(*shape, 3)
            rotation = torch.zeros(*shape, 4)
            rotation[..., 0] = 1
            return position, rotation, position.clone(), position.clone()

    monkeypatch.setattr(native_create, "get_class", lambda name: FactoryPlant)
    monkeypatch.setattr(native_create, "build_manager_cfgs", lambda cfg: {})
    monkeypatch.setattr(native_create, "RslRlVecEnvWrapper", lambda env: env)
    monkeypatch.setattr(native_create, "SharedEvalEnv", lambda env, cfg: None)
    cfg = OmegaConf.create(dict(robot=dict(entity_name="robot", native_scene=mapping),
        commands=dict(object=dict(pin_objects=False, support_disks=dict(enable=False))), curriculum={},
        sim=dict(dt=.005, plant={"_target_": "explicit_cpu_protocol"}), scene=dict(num_envs=2),
        decimation=4, episode_length_s=1., seed=1, eval={}))
    env, _ = native_create.create_native_envs(cfg, "cpu")
    scene, plant = env.scene, env.sim.plant
    key = "objects/object/loose_prop"
    assert len(plant.contact_facts) == 7
    assert plant.contact_pairs[key] == ([plant.model.body("base").id, plant.model.body("lid").id],
                                       [plant.model.body("prop_root").id])
    assert key not in scene.hand_part_contact_facts()["pairs"]
    assert key not in scene.hand_part_contact_interval_facts()["pairs"]
    assert {a["part_entity_name"] for a in scene.hand_part_contact_facts()["addresses"].values()} == {"object", "loose_prop"}
    assert "left/object/base" in plant.contact_facts and "left/loose_prop/base" in plant.contact_facts
    assert set(scene.object_parts) == {"object/base", "object/lid", "loose_prop/base"}
    plant.contact_facts["left/object/base"]["normal_force_n"].zero_()
    facts = scene.hand_part_contact_facts()
    assert facts["pairs"]["left/object/base"]["normal_force_n"].eq(0).all()
    assert facts["pairs"]["left/loose_prop/base"]["normal_force_n"].isnan().all()
    assert scene["loose_prop"].joint_names == [] and plant.advance_calls == plant.set_calls == 0
