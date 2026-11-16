"""Discrete contact evidence software contracts; no solver or physical claims."""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Expose the ordinary vendored managers without launching an SDK.
import simulator.isaacsim  # noqa: F401

# isort: split
from omegaconf import OmegaConf
from test_native_rollout_record import record_reference_phase_fixture
from test_scene_contact_authority import _isaac_scene_type

from evaluation.callbacks.contact_performance import NativeContactPerformance
from evaluation.evaluate import write_current_policy_outcomes
from simulator.env import ManagerBasedRlEnv, ManagerEnvCfg
from simulator.isaacsim.sim import IsaacSimAdapter
from simulator.native.mjwarp import MjWarpPlant
from simulator.native.newton import NewtonPlant
from simulator.native.plant import NativePlant
from simulator.native.scene import NativeScene, NativeSim


def _layout():
    obj = SimpleNamespace(entity_name="assembly", parts=[
        SimpleNamespace(name=name, body_name=name + "_body") for name in ("fixture", "cover", "latch")])
    return SimpleNamespace(
        hands=[SimpleNamespace(side=side, root_body_name=side + "_mount") for side in ("port", "starboard")],
        objects=(obj,), part_bindings=tuple((obj, part) for part in obj.parts),
    )


class _Plant(NativePlant):
    """Explicit CPU contact/clock protocol, without kinematics or integration."""

    def __init__(self, *, substeps=3, requires_grad=False):
        self.device, self.num_envs, self.substeps, self.requires_grad = "cpu", 2, substeps, requires_grad
        self.model = SimpleNamespace(nu=1, nbody=1, opt=SimpleNamespace(gravity=(0, 0, -9.81)))
        self.identity = {"backend": "synthetic_contact_protocol_not_physics"}
        self.layout = _layout()
        self._time = torch.tensor([17.25, 3.5], dtype=torch.float64)
        self.dt, self.slot, self.fail = 0.005, 0, False
        self.q, self.v = torch.zeros(2, 1), torch.zeros(2, 1)
        self._init_buffers(self.model, self.num_envs, self.device, {
            f"{hand.side}/{obj.entity_name}/{part.name}": ([], [])
            for hand in self.layout.hands for obj, part in self.layout.part_bindings
        })

    def advance(self):
        self.prepare_contact_substep()
        if self.fail:
            raise RuntimeError("stopped before next solved sample")
        self.contact_sample_time_s.copy_(self._time)
        self._time += self.dt
        slot = self.slot % self.substeps
        self.contact_support[:] = True
        if slot == 1:
            self.contact_support[1] = False
        force = 0.75 if slot == 0 else 0.0 if slot == 1 else 0.4
        for facts in self.contact_facts.values():
            facts["normal_force_n"] = torch.tensor([force, force], requires_grad=self.requires_grad)
            facts["force_bearing_contact_present"] = torch.tensor([force > 0] * 2)
            distance = 0.01 if slot == 0 else float("nan") if slot == 1 else 0.03
            facts["force_bearing_distance_max_m"] = torch.tensor([distance] * 2)
        self.slot += 1
        self.record_contact_substep()

    def physics_time(self):
        return self._time.clone()

    def state(self):
        return self.q, self.v

    def set_state(self, rows, q, v):
        self.q[rows], self.v[rows] = q, v
        self.invalidate_contacts(rows)

    def reset(self, rows):
        self._time[rows] = 0
        self.q[rows], self.v[rows] = 0, 0
        self.invalidate_contacts(rows)

    def forward(self):
        pass

    def body_state(self, body_ids):
        raise NotImplementedError("this fixture supplies no body-state physics")

    def capture_physics(self):
        return self.q.clone(), self.v.clone(), self._time.clone()

    def restore_physics(self, state):
        for target, value in zip((self.q, self.v, self._time), state, strict=True):
            target.copy_(value)


class _Scene:
    hand_part_contact_facts = NativeScene.hand_part_contact_facts
    hand_part_contact_interval_facts = NativeScene.hand_part_contact_interval_facts

    def __init__(self, sim):
        self.sim, self.native_layout = sim, sim.plant.layout
        self.backend_identity = sim.plant.identity
        self.num_envs, self._clock = sim.plant.num_envs, sim._clock

    def write_data_to_sim(self):
        self.sim.flush_state()

    def reset(self, env_ids):
        del env_ids

    def update(self, dt):
        del dt


def _sim(substeps=3, *, requires_grad=False):
    plant = _Plant(substeps=substeps, requires_grad=requires_grad)
    sim = NativeSim(plant)
    plant.configure_control_interval(substeps)
    return sim, _Scene(sim)


def _advance_control(sim, substeps):
    sim.begin_control_step()
    for _ in range(substeps):
        sim.step()
    sim.end_control_step()


def _tensors(interval):
    facts = interval.facts()
    return [value for value in facts.values() if isinstance(value, torch.Tensor)] + [
        value for pair in facts["pairs"].values() for value in pair.values()]


@pytest.mark.parametrize("substeps", [1, 3, 5])
def test_every_configured_substep_is_stopped_and_storage_is_bounded(substeps):
    sim, scene = _sim(substeps, requires_grad=True)
    interval = sim.plant.contact_interval
    pointers = [value.data_ptr() for value in _tensors(interval)]
    size = sum(value.numel() * value.element_size() for value in _tensors(interval))
    assert scene.hand_part_contact_interval_facts()["num_substeps"] == substeps
    assert not interval.sample_recorded.any() and torch.isnan(interval.sample_begin_time_s).all()
    sim.begin_control_step()
    clocks, forces = [], []
    for _ in range(substeps):
        begin = sim.plant.physics_time()
        sim.step()
        clocks.append((begin, sim.plant.physics_time()))
        forces.append(sim.plant.contact_facts["port/assembly/fixture"]["normal_force_n"].detach().clone())
    sim.end_control_step()
    torch.testing.assert_close(interval.sample_begin_time_s, torch.stack([row[0] for row in clocks], 1), rtol=0, atol=0)
    torch.testing.assert_close(interval.sample_end_time_s, torch.stack([row[1] for row in clocks], 1), rtol=0, atol=0)
    torch.testing.assert_close(interval.pairs["port/assembly/fixture"]["normal_force_n"], torch.stack(forces, 1), rtol=0, atol=0)
    assert interval.interval_complete.all()
    if substeps > 1:
        assert not interval.measurement_support[1, 1] and interval.sample_recorded[1, 1]
    for _ in range(25):
        _advance_control(sim, substeps)
    assert [value.data_ptr() for value in _tensors(interval)] == pointers
    assert sum(value.numel() * value.element_size() for value in _tensors(interval)) == size
    assert all(not value.requires_grad and value.grad_fn is None for value in _tensors(interval))
    assert interval.sample_recorded.shape == (sim.plant.num_envs, substeps)


def test_partial_reset_state_write_and_stopped_interval_discard_old_samples():
    sim, _scene = _sim()
    _advance_control(sim, 3)
    old_other_row = sim.plant.contact_interval.pairs["port/assembly/fixture"]["normal_force_n"][1].clone()
    sim.reset(torch.tensor([0]))
    interval = sim.plant.contact_interval
    assert not interval.sample_recorded[0].any() and not interval.interval_complete[0]
    assert torch.isnan(interval.sample_begin_time_s[0]).all()
    assert torch.isnan(interval.pairs["port/assembly/fixture"]["normal_force_n"][0]).all()
    torch.testing.assert_close(interval.pairs["port/assembly/fixture"]["normal_force_n"][1], old_other_row)
    sim.begin_control_step()
    sim.step()
    sim.plant.fail = True
    with pytest.raises(RuntimeError, match="stopped before"):
        sim.step()
    assert interval.sample_recorded.sum(1).tolist() == [1, 1]
    assert not interval.interval_complete.any() and torch.isnan(interval.sample_begin_time_s[:, 1:]).all()
    sim.plant.fail = False
    _advance_control(sim, 3)
    sim.pending_state(torch.tensor([0]))
    sim.flush_state()
    assert not interval.sample_recorded[0].any() and interval.interval_complete[1]
    saved = sim.plant.snapshot()
    _advance_control(sim, 3)
    sim.plant.restore(saved)
    assert not interval.sample_recorded.any() and not interval.interval_complete.any()
    # Restored last-solve evidence remains distinct from a new control interval.
    torch.testing.assert_close(sim.plant.contact_sample_time_s, saved.contact_sample_time_s, equal_nan=True)
    _advance_control(sim, 3)
    # An unbracketed direct public backend advance also invalidates the old interval.
    sim.plant.advance()
    assert not interval.sample_recorded.any() and not interval.interval_complete.any()


@pytest.mark.parametrize("engine", ["mjwarp", "newton"])
def test_real_backend_advance_methods_feed_the_same_interval_contact_owner(engine):
    """Only the external numerical calls are protocol stubs; run the real adapter method."""
    clock = torch.tensor([8.0, 12.0], dtype=torch.float64)
    model = SimpleNamespace(nu=1, nv=1, nbody=9)
    state = SimpleNamespace(qpos=torch.zeros(2, 1), qvel=torch.zeros(2, 1))
    index = 0

    def step(*args, **kwargs):
        nonlocal index
        index += 1
        clock.add_(0.005)
        return SimpleNamespace(public_state=state, applied_joint_effort=torch.zeros(2, 1))

    def read_contacts(*args):
        return {"normal_force_n": torch.full((2,), float(index)),
                "force_bearing_contact_present": torch.ones(2, dtype=torch.bool),
                "force_bearing_distance_max_m": torch.full((2,), 0.003)}

    class NewtonFixture(NewtonPlant):
        def __init__(self):
            self._init_buffers(model, 2, "cpu", {"renamed/panel": ([3], [8])})
            self.effort = torch.zeros(2, 1)
            self._world_ids, self._body_ids = torch.arange(2), torch.arange(1, model.nbody)
            self.backend = SimpleNamespace(get_physics_time_torch=lambda: clock.clone(),
                step_torch=step, get_body_pair_contact_facts=read_contacts,
                set_body_wrench_w_torch=lambda *args, **kwargs: None)

    class WarpFixture(MjWarpPlant):
        def __init__(self):
            self._init_buffers(model, 2, "cpu", {"renamed/panel": ([3], [8])})
            self._model = SimpleNamespace(opt=SimpleNamespace(timestep=torch.tensor([.005], dtype=torch.float32)))
            self._data = SimpleNamespace(overflow=torch.zeros(1))
            self._ctrl = torch.zeros(2, 1)
            self._xfrc_applied = torch.zeros_like(self.external_body_wrench_w)
            self._solved_effort, self.effort = torch.zeros(2, 1), torch.zeros(2, 1)
            self._mjw, self._wp = SimpleNamespace(step=step), SimpleNamespace(to_torch=lambda value: value)
            self.identity = {"backend": "synthetic actual-MjWarp-method fixture"}
            self._bind_execution_clock(.005)

        def _scope(self):
            return nullcontext()

        def physics_time(self):
            return clock.clone()

        def _read_solved_contacts(self):
            for key, value in read_contacts().items():
                self.contact_facts["renamed/panel"][key].copy_(value)
            self.contact_support[:] = True

        def forward(self):
            pass

    plant = NewtonFixture() if engine == "newton" else WarpFixture()
    plant.configure_control_interval(3)
    plant.contact_interval.begin()
    for _ in range(3):
        plant.advance()
    plant.contact_interval.finish()
    torch.testing.assert_close(plant.contact_interval.pairs["renamed/panel"]["normal_force_n"],
                               torch.tensor([[1.0, 2.0, 3.0]] * 2), rtol=0, atol=0)
    assert plant.contact_interval.interval_complete.all()
    assert plant.contact_interval.sample_recorded.all()
    if engine == "newton":
        assert plant.execution_clock() is None
    else:
        assert plant.execution_clock()["steps"].tolist() == [3, 3]
        assert plant.contact_interval.facts()["execution_interval_complete"].all()


def test_normal_manager_brackets_all_substeps_before_partial_auto_reset():
    sim, scene = _sim(5)
    events = []
    for name in ("begin_control_step", "end_control_step", "reset"):
        method = getattr(sim, name)

        def wrapped(*args, method=method, name=name):
            events.append(name)
            return method(*args)

        setattr(sim, name, wrapped)

    class FixtureManager(ManagerBasedRlEnv):
        def load_managers(self):
            reset = lambda env_ids=None: {}
            self.action_manager = SimpleNamespace(
                process_action=lambda action: events.append("process_action"),
                apply_action=lambda: events.append("apply_action"), reset=reset)
            self.command_manager = SimpleNamespace(compute=lambda dt: events.append("compute"), reset=reset)
            self.observation_manager = SimpleNamespace(compute=lambda **kwargs: {}, reset=reset)
            self.reward_manager = SimpleNamespace(compute=lambda dt: torch.zeros(2), reset=reset)
            self.termination_manager = SimpleNamespace(compute=lambda: torch.tensor([True, False]),
                terminated=torch.tensor([True, False]), time_outs=torch.zeros(2, dtype=torch.bool), reset=reset)
            self.curriculum_manager = SimpleNamespace(compute=lambda **kwargs: None, reset=reset)
            self.event_manager = SimpleNamespace(available_modes=[], reset=reset)

    env = FixtureManager(ManagerEnvCfg(decimation=5, physics_dt=sim.plant.dt), scene, sim, "cpu")
    env.step(torch.zeros(2, 1))
    assert events[0:2] == ["begin_control_step", "process_action"]
    assert events.count("apply_action") == 5
    assert events.index("end_control_step") < events.index("reset") < events.index("compute")
    assert not sim.plant.contact_interval.sample_recorded[0].any()
    assert sim.plant.contact_interval.sample_recorded[1].all() and sim.plant.contact_interval.interval_complete[1]


def _callback_env(sim, scene):
    library = SimpleNamespace(num_trajectories=3, motion_file="protocol.pt",
        motion_files=[f"protocol.pt#motion-{index}" for index in range(3)])
    command = SimpleNamespace(native_layout=sim.plant.layout, motion_lib=library,
        motion_ids=torch.tensor([2, 0]), motion_steps=torch.ones(2, dtype=torch.int64),
        motion_num_frames=torch.tensor([3, 3]))
    return SimpleNamespace(num_envs=2, device="cpu", scene=scene,
        physics_dt=sim.plant.dt, step_dt=sim.plant.dt * sim.plant.contact_interval.num_substeps,
        cfg=SimpleNamespace(decimation=sim.plant.contact_interval.num_substeps),
        command_manager=SimpleNamespace(get_term=lambda name: command))


def test_measurement_callback_and_current_report_use_all_samples_and_real_denominators(tmp_path):
    sim, scene = _sim()
    env = _callback_env(sim, scene)
    callback = NativeContactPerformance("motion", normal_force_threshold_n=0.5, separation_bound_m=0.02)
    callback.on_start(env)
    _advance_control(sim, 3)
    callback.on_step(env)
    state = callback.collect_state()
    assert state["supported_sample_count"][:, 0].tolist() == [3, 2]
    assert state["qualified_sample_count"][:, 0].tolist() == [1, 1]
    assert state["qualified_duration_s"][:, 0].tolist() == pytest.approx([0.005, 0.005], abs=1e-12)
    assert state["supported_duration_s"][:, 0].tolist() == pytest.approx([0.015, 0.01], abs=1e-12)
    assert scene.hand_part_contact_facts()["pairs"]["port/assembly/fixture"]["normal_force_n"].tolist() == pytest.approx([0.4, 0.4])
    output = write_current_policy_outcomes(SimpleNamespace(), env,
        OmegaConf.create({"command_name": "motion", "num_per_motion": 1}),
        [callback], {id(callback): [state]}, callback.reduce_state([state]),
        outcome_dir=tmp_path, env_step=12345)
    report = json.loads(output.read_text())
    key = next(iter(report["callback_metadata"]))
    assert report["motions"][1]["callbacks"][key]["coverage"] == "not_evaluated"
    row = report["motions"][0]["callbacks"][key]["attempts"][0]
    assert row["expected_sample_count"] == [3] * len(callback.pair_names)
    assert row["unsupported_or_unrecorded_sample_count"] == [1] * len(callback.pair_names)
    assert "UNKNOWN" in report["callback_metadata"][key]["source_required_contact_phase"]
    assert "not continuous" in report["callback_metadata"][key]["duration_authority"]


def test_isaac_net_link_and_sdf_views_do_not_create_interval_pair_force_support(monkeypatch):
    sim, _scene = _sim()
    adapter = _isaac_scene_type(monkeypatch)(SimpleNamespace(), "cpu", SimpleNamespace(current_time=0.0))
    env = _callback_env(sim, adapter)
    events = []
    isaac = IsaacSimAdapter(SimpleNamespace(step=lambda render: events.append("step")), "cpu")
    isaac.begin_control_step()
    isaac.end_control_step()
    assert not events and adapter.hand_part_contact_interval_facts() is None
    callback = NativeContactPerformance("motion", 0.5, 0.02)
    callback.on_start(env)
    callback.on_step(env)
    result = callback.current_policy_outcomes([callback.collect_state()])
    assert not result["metadata"]["pair_measurement_available"]
    assert not result["columns"]["scored"].any()
    assert not result["columns"]["supported_sample_count"].any()
    assert (result["columns"]["unsupported_or_unrecorded_sample_count"] == 3).all()


def test_recorder_retains_substep_tape_reset_unknown_and_original_phase(tmp_path):
    from dexforge_adaptation.flashdex_retarget import (
        flashdex_source_mapping,
        load_flashdex_rollout,
    )

    source = flashdex_source_mapping(source_id="arctic", source_revision="fixture-v1",
        source_unit_id="fixture", source_episode_index=174, fold="train",
        source_frame_index=(0, 1, 2), source_time_ns=(0, 20_000_000, 40_000_000),
        selected_source_frame_index=(0, 1, 2), prepared_source_ordinal=(0.0, 1.0, 2.0),
        prepared_elapsed_time_ns=(0, 20_000_000, 40_000_000),
        clock_authority="synthetic phase protocol", processing={"fixture": True})
    binding = dict(producer_identity={"method": "synthetic-not-physical"}, scene_identity={"fixture": True},
        scene_from_source_world=np.eye(4).tolist(),
        assistance_receipt={"evaluation_external_wrenches_cleared": True, "evaluation_object_playback": False},
        hands={side: {"body_name": side + "_root", "command_joint_names": {"driver": side + "_driver"},
                      "physical_joint_names": {"driver": side + "_driver"}} for side in ("left", "right")},
        object_part_names=["base", "lid"],
        object_parts={name: {"entity_name": "object", "body_name": body} for name, body in
                      (("base", "object_root"), ("lid", "object_lid"))}, clips={"fixture": source})

    def setup(env, command):
        def contacts():
            row = int(command.reference_phase["motion_steps"].item())
            known = row > 0
            addresses = {f"{side}/{name}": dict(hand_side=side, hand_body_name=side + "_root",
                part_entity_name="object", part_body_name=part["body_name"])
                for side in ("left", "right") for name, part in binding["object_parts"].items()}
            return dict(addresses=addresses,
                pairs={name: {"normal_force_n": torch.tensor([0.4 if known else float("nan")]),
                    "force_bearing_contact_present": torch.tensor([known]),
                    "force_bearing_distance_max_m": torch.tensor([0.03 if known else float("nan")])} for name in addresses},
                measurement_support=torch.tensor([known]), sample_time_s=torch.tensor([row * 0.02 - 0.02 / 3 if known else float("nan")]),
                clock="synthetic_last_substep_begin", normal_force_authority="synthetic protocol", distance_authority="synthetic protocol")

        def interval():
            facts = contacts()
            row = int(command.reference_phase["motion_steps"].item())
            known = row > 0
            begin = torch.arange(3, dtype=torch.float64)[None] * (0.02 / 3) + (row - 1) * 0.02
            if not known:
                begin[:] = float("nan")
            return {**facts, "num_substeps": 3, "clock": "synthetic solver begin/end", "sampling": "discrete samples, not continuous force",
                "pairs": {name: {"normal_force_n": torch.tensor([[0.75, 0.0, 0.4] if known else [float("nan")] * 3]),
                    "force_bearing_contact_present": torch.tensor([[True, False, True] if known else [False] * 3]),
                    "force_bearing_distance_max_m": torch.tensor([[0.01, float("nan"), 0.03] if known else [float("nan")] * 3])} for name in facts["addresses"]},
                "measurement_support": torch.full((1, 3), known), "sample_recorded": torch.full((1, 3), known),
                "clock_support": torch.full((1, 3), known), "sample_begin_time_s": begin, "sample_end_time_s": begin + 0.02 / 3,
                "interval_complete": torch.tensor([known])}

        env.scene.hand_part_contact_facts = contacts
        env.scene.hand_part_contact_interval_facts = interval

    manifest = record_reference_phase_fixture(tmp_path, binding, native=True, setup_scene=setup)
    arrays = load_flashdex_rollout(manifest, motion_id=0).arrays
    assert arrays["hand_part_interval_normal_force_n"].shape == (3, 3, 4)
    assert np.isnan(arrays["hand_part_interval_normal_force_n"][0]).all()
    assert not arrays["hand_part_interval_sample_recorded"][0].any()
    assert not arrays["hand_part_interval_interval_complete"][0]
    np.testing.assert_allclose(arrays["hand_part_interval_normal_force_n"][1, :, 0], [0.75, 0.0, 0.4])
    np.testing.assert_array_equal(arrays["recorded_motion_steps"], [0, 1, 2])
    np.testing.assert_array_equal(arrays["command_motion_steps"], [1, 2, 2])
    metadata = json.loads(str(arrays["hand_part_interval_metadata_json"]))
    assert "not continuous" in metadata["sampling"] and "UNKNOWN" in metadata["source_required_contact_phase"]
    assert "not original" in metadata["provenance"]
