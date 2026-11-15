"""Installed-property and shared-control CPU contracts; no Kit or physics step."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Expose the actual vendored manager subset before importing its consumers.
import simulator.isaacsim  # noqa: F401

# isort: split
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse
from test_isaac_asset_bindings import registered_asset as _registered_asset
from test_native_backend_ports import _mechanics
from test_scene_contact_authority import _isaac_scene_type

from envs.commands.motion_tracking.motion_tracking_cfg import ObjectCfg
from envs.commands.motion_tracking.objects.assist import ObjectAssistMixin
from envs.commands.motion_tracking.objects.multi_object import MultiObjectMixin
from envs.commands.motion_tracking.reference.object import ObjectPropertiesMixin
from envs.curriculum.base import gravity_curriculum, xfrc_curriculum
from envs.scene.native import NativeRootControlCfg
from simulator.env import ManagerBasedRlEnv, ManagerEnvCfg
from simulator.isaacsim.assets import load_robot_asset
from simulator.isaacsim.entity import ArticulationAdapter, RigidObjectAdapter
from simulator.isaacsim.sim import IsaacSimAdapter
from simulator.mjcf_meta import load_robot_mjcf_meta
from simulator.native.entity import (
    BodyInertialProperties,
    FloatingRootView,
    _FreshnessClock,
)
from simulator.native.scene import NativeArticulation, NativeSim

registered_asset = _registered_asset


def _articulation(registered_asset):
    """Two installed worlds in reverse source-body order, unlike MJCF defaults."""
    binding = load_robot_asset(registered_asset)
    meta = load_robot_mjcf_meta(binding.mjcf_path)
    body_names = list(reversed(meta.body_names))
    mass = torch.tensor([[7.0, 2.0], [11.0, 3.0]], requires_grad=True)
    com = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]], [[0.2, 0.0, 0.1], [0.0, 0.3, 0.1]]]
    )
    principal_pose = torch.zeros(2, 2, 7)
    principal_pose[..., :3] = com
    # A nonidentity principal frame must NOT rotate already body-frame I again.
    principal_pose[..., 3:] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
    inertia = torch.tensor(
        [
            [
                [[3.0, 0.2, 0.1], [0.2, 4.0, 0.3], [0.1, 0.3, 5.0]],
                [[6.0, 0.1, 0.2], [0.1, 8.0, 0.4], [0.2, 0.4, 10.0]],
            ],
            [
                [[12.0, 0.2, 0.3], [0.2, 14.0, 0.4], [0.3, 0.4, 16.0]],
                [[18.0, 0.5, 0.3], [0.5, 20.0, 0.2], [0.3, 0.2, 22.0]],
            ],
        ]
    )
    transforms = torch.zeros(2, 2, 7)
    transforms[..., 3:] = principal_pose[..., 3:]
    velocities = torch.zeros(2, 2, 6)
    velocities[..., 5] = 2.0
    reads = {"mass": 0, "com": 0, "inertia": 0}

    def fetch(name, tensor):
        reads[name] += 1
        return tensor

    view = SimpleNamespace(
        get_masses=lambda: fetch("mass", mass),
        get_coms=lambda: fetch("com", principal_pose),
        get_inertias=lambda: fetch(
            "inertia", inertia.transpose(-1, -2).reshape(2, 2, 9)
        ),
        get_link_transforms=lambda: transforms,
        get_link_velocities=lambda: velocities,
    )
    writes = []
    art = SimpleNamespace(
        num_instances=2,
        root_physx_view=view,
        find_bodies=lambda names, preserve_order: (
            [body_names.index(n) for n in names],
            names,
        ),
        find_joints=lambda names, preserve_order: (
            [meta.joint_names.index(n) for n in names],
            names,
        ),
        set_external_force_and_torque=lambda f, t, **kw: writes.append(
            (f.clone(), t.clone(), kw)
        ),
        reset=lambda **kwargs: None,
    )
    adapter = ArticulationAdapter(art, meta, _FreshnessClock(), "cpu")
    return adapter, mass, com, inertia, reads, writes


def test_articulation_reads_installed_named_world_properties_once_without_double_rotation(
    registered_asset,
):
    adapter, mass, com, inertia, reads, _ = _articulation(registered_asset)
    properties = adapter.body_inertial_properties()
    assert properties.body_names == tuple(adapter.body_names)
    torch.testing.assert_close(properties.mass_kg, mass[:, [1, 0]])
    torch.testing.assert_close(properties.com_position_body_m, com[:, [1, 0]])
    torch.testing.assert_close(properties.inertia_com_body_kg_m2, inertia[:, [1, 0]])
    assert not properties.mass_kg.requires_grad
    expected_offset = quat_apply(adapter.data.body_link_quat_w, com[:, [1, 0]])
    omega = torch.zeros_like(expected_offset)
    omega[..., 2] = 2.0
    torch.testing.assert_close(
        adapter.data.body_link_lin_vel_w, -torch.cross(omega, expected_offset, dim=-1)
    )
    assert adapter.body_inertial_properties() is properties
    assert reads == {"mass": 1, "com": 1, "inertia": 1}
    # A retained property snapshot is independent of mutable view scratch memory.
    with torch.no_grad():
        mass[0, 0] = 999.0
    assert properties.mass_kg[0, 1] == 7.0


def _rigid(mass, *, name="payload", inertia=None, global_wrench=True):
    mass = torch.as_tensor(mass, dtype=torch.float32)
    count = len(mass)
    com = torch.tensor([0.2, 0.1, 0.0]).expand(count, -1).clone()
    transforms = torch.zeros(count, 7)
    transforms[:, :3] = torch.tensor([3.0, 4.0, 5.0])
    transforms[:, 3:] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
    com_pose = torch.zeros(count, 7)
    com_pose[:, :3] = com
    com_pose[:, 3:] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    if inertia is None:
        inertia = torch.diag_embed(torch.stack([mass, mass * 2, mass * 3], -1))
    velocities = torch.zeros(count, 6)
    writes = []

    def global_write(forces, torques, positions=None, env_ids=None, is_global=False):
        writes.append((forces.clone(), torques.clone(), positions, env_ids, is_global))

    def local_write(forces, torques, positions=None, env_ids=None):
        writes.append((forces.clone(), torques.clone(), positions, env_ids, False))

    obj = SimpleNamespace(
        num_instances=count,
        body_names=[name],
        root_physx_view=SimpleNamespace(
            get_masses=lambda: mass[:, None],
            get_coms=lambda: com_pose,
            get_inertias=lambda: inertia.transpose(-1, -2).reshape(count, 9),
            get_transforms=lambda: transforms,
            get_velocities=lambda: velocities,
        ),
        set_external_force_and_torque=global_write if global_wrench else local_write,
        reset=lambda **kwargs: None,
    )
    adapter = RigidObjectAdapter(
        obj, _FreshnessClock(), "cpu", torch.zeros(count, 13), body_name=name
    )
    return adapter, writes, transforms, velocities


@pytest.mark.parametrize("global_wrench", [False, True])
def test_rigid_installed_com_velocity_and_translated_rotated_wrench_point(
    monkeypatch, global_wrench
):
    adapter, writes, _, velocities = _rigid([2, 5], global_wrench=global_wrench)
    properties = adapter.body_inertial_properties()
    assert properties.mass_kg.shape == (2, 1)
    assert properties.com_position_body_m.shape == (2, 1, 3)
    assert properties.inertia_com_body_kg_m2.shape == (2, 1, 3, 3)
    assert not hasattr(adapter, "indexing")
    assert not hasattr(adapter.data, "indexing")
    velocities[:, 5] = 1.0
    torch.testing.assert_close(
        adapter.data.body_link_lin_vel_w[:, 0],
        torch.tensor([[0.2, 0.1, 0.0]]).expand(2, -1),
    )
    force = torch.tensor([[[2.0, 1.0, 0.0]]])
    torque = torch.tensor([[[0.0, 0.0, 3.0]]])
    monkeypatch.setenv("FDR_PIN_AT_COM", "1")
    rows = torch.tensor([1])
    adapter.write_external_wrench_to_sim(force, torque, env_ids=rows)
    issued_force, issued_torque, point, selected, actual_global = writes[-1]
    assert torch.equal(selected, rows) and actual_global is global_wrench
    if global_wrench:
        torch.testing.assert_close(issued_force, force)
        torch.testing.assert_close(issued_torque, torque)
        torch.testing.assert_close(point, torch.tensor([[[2.9, 4.2, 5.0]]]))
    else:
        quat = adapter.data.body_link_quat_w[rows]
        torch.testing.assert_close(issued_force, quat_apply_inverse(quat, force))
        torch.testing.assert_close(issued_torque, quat_apply_inverse(quat, torque))
        torch.testing.assert_close(point, torch.tensor([[[0.2, 0.1, 0.0]]]))
    monkeypatch.delenv("FDR_PIN_AT_COM")
    adapter.write_external_wrench_to_sim(force, torque, env_ids=rows)
    assert writes[-1][2] is None  # Preserve the existing default application request.


class _ObjectCommand(ObjectAssistMixin, ObjectPropertiesMixin):
    def __init__(self, entities, names, slots, *, shared_slots=None):
        self.device, self.num_envs = "cpu", 2
        self._side_list = list(names)
        self.cfg = SimpleNamespace(
            object=ObjectCfg(
                entity_names=names, shared_slots=shared_slots, pin_objects=True
            )
        )
        self._env = SimpleNamespace(
            scene=entities,
            physics_dt=0.1,
            device="cpu",
            num_envs=2,
            common_step_counter=0,
        )
        self._env.scene = _OriginsScene(entities)
        self.motion_ids = torch.tensor([0, 1])
        self._motion_flat_ids = self.motion_ids
        self.motion_lib = SimpleNamespace(
            traj_obj_slot=slots,
            obj_trans={},
            obj_rot={},
            obj_lin_vel={},
            obj_ang_vel={},
        )
        self._init_object_assist()
        self._park_inactive_on_ground = False
        for side in self._side_list:
            position = self._sim_obj_field(side, "body_link_pos_w").clone()
            position[:, 0] += 1.0
            self.motion_lib.obj_trans[side] = position
            self.motion_lib.obj_rot[side] = matrix_from_quat(
                torch.tensor([[np.cos(0.05), 0.0, 0.0, np.sin(0.05)]])
                .float()
                .expand(2, -1)
            )
            self.motion_lib.obj_lin_vel[side] = torch.zeros(2, 3)
            self.motion_lib.obj_ang_vel[side] = torch.zeros(2, 3)


class _OriginsScene(dict):
    def __init__(self, entities):
        super().__init__(entities)
        self.env_origins = torch.zeros(2, 3)


@pytest.mark.parametrize("layout", ["per_world", "slots", "mixed_alias"])
def test_assist_and_mass_curriculum_use_actual_world_slot_and_shared_alias(layout):
    first, first_writes, first_tf, _ = _rigid([2, 3], name="first")
    second, second_writes, second_tf, _ = _rigid([5, 7], name="second")
    first_tf[:, 3:] = second_tf[:, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    if layout == "per_world":
        names, slots = {"left": "first"}, {"left": torch.tensor([1, 0])}
        expected_left, expected_gain_mass = (
            torch.tensor([2.0, 3.0]),
            torch.tensor([2.0, 3.0]),
        )
        shared = None
    elif layout == "slots":
        names, slots = {"left": ["first", "second"]}, {"left": torch.tensor([1, 0])}
        expected_left, expected_gain_mass = (
            torch.tensor([5.0, 3.0]),
            torch.tensor([5.0, 3.0]),
        )
        shared = None
    else:
        names = {"right": "first", "left": "second"}
        slots = {side: torch.tensor([0, 1]) for side in names}
        expected_left, expected_gain_mass = (
            torch.tensor([2.0, 7.0]),
            torch.tensor([2.0, 5.0]),
        )
        shared = [True, False]
    cmd = _ObjectCommand(
        {"first": first, "second": second}, names, slots, shared_slots=shared
    )
    torch.testing.assert_close(cmd._obj_mass_env("left"), expected_left)
    torch.testing.assert_close(
        cmd._obj_inertia_body("left")[:, 2, 2], expected_left * 3
    )
    # The same owner selects achieved pose, not a fabricated body/global ID.
    torch.testing.assert_close(
        cmd.sim_obj_trans_w[:, -1], torch.tensor([[3.0, 4.0, 5.0]]).expand(2, -1)
    )
    env = cmd._env
    env.command_manager = SimpleNamespace(get_term=lambda name: cmd)
    xfrc_curriculum(env, torch.tensor([1]), omega_n_start=2.0, drive_omega_rot=False)
    torch.testing.assert_close(cmd._xfrc_kp_pos_env, expected_gain_mass * 4)
    torch.testing.assert_close(cmd._xfrc_kv_pos_env, expected_gain_mass * 4)
    cmd.cfg.object.xfrc.omega_rot = 1.0
    cmd._apply_object_assist()
    if layout == "slots":
        torch.testing.assert_close(
            first_writes[-1][0][:, 0, 0], torch.tensor([0.0, 12.0])
        )
        torch.testing.assert_close(
            second_writes[-1][0][:, 0, 0], torch.tensor([20.0, 0.0])
        )
    elif layout == "mixed_alias":
        torch.testing.assert_close(
            first_writes[-1][0][:, 0, 0], torch.tensor([8.0, 20.0])
        )
        torch.testing.assert_close(
            second_writes[-1][0][:, 0, 0], torch.tensor([0.0, 20.0])
        )
        torch.testing.assert_close(
            second_writes[-1][1][:, 0, 2], torch.tensor([0.0, 2.1])
        )
    else:
        torch.testing.assert_close(
            first_writes[-1][0][:, 0, 0], torch.tensor([8.0, 12.0])
        )
        torch.testing.assert_close(
            first_writes[-1][1][:, 0, 2], torch.tensor([0.6, 0.9])
        )
    cmd.cfg.object.xfrc.rot_inertia_scale = False
    inertia = cmd._obj_inertia_body("left")
    force, torque = cmd._implicit_damping_wrench(
        "left", torch.ones(2, 3), torch.ones(2, 3), 3.0, 4.0, inertia
    )
    torch.testing.assert_close(force[:, 0], expected_left / (expected_left + 0.3))
    diagonal = inertia.diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(torque, diagonal / (diagonal + 0.4))


def test_reference_object_uses_named_root_instead_of_last_or_global_body_index():
    class Reader(ObjectPropertiesMixin):
        device, num_envs = "cpu", 2

    obj = SimpleNamespace(
        body_names=["base", "lid"],
        root_body_name="base",
        data=SimpleNamespace(
            body_link_pos_w=torch.tensor(
                [[[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]], [[4.0, 5.0, 6.0], [8.0, 8.0, 8.0]]]
            )
        ),
    )
    reader = Reader()
    reader.cfg = SimpleNamespace(object=ObjectCfg(entity_names={"left": "box"}))
    reader._side_list = ["left"]
    reader._env = SimpleNamespace(scene={"box": obj})
    torch.testing.assert_close(
        reader.sim_obj_trans_w[:, 0], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )


def test_native_properties_use_real_compiled_body_rows_and_static_gravity(tmp_path):
    layout, _, model = _mechanics(tmp_path)
    meta = load_robot_mjcf_meta(layout.scene_xml_path).subtree("base")
    ids = [model.body(name).id for name in meta.body_names]
    model.body_mass[ids] = [2.0, 5.0]
    model.body_ipos[ids] = [[0.1, 0.2, 0.0], [0.3, 0.1, 0.0]]
    model.body_inertia[ids] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    model.body_iquat[ids] = [2**-0.5, 0.0, 0.0, 2**-0.5]
    sim = NativeSim(SimpleNamespace(model=model, device="cpu", num_envs=2))
    obj = NativeArticulation(layout.object, meta, sim)
    properties = obj.body_inertial_properties()
    torch.testing.assert_close(
        properties.mass_kg, torch.tensor([[2.0, 5.0]]).expand(2, -1)
    )
    torch.testing.assert_close(
        properties.com_position_body_m[0],
        torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.1, 0.0]]),
    )
    torch.testing.assert_close(
        properties.inertia_com_body_kg_m2[0],
        torch.diag_embed(torch.tensor([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]])),
    )
    # The actual named hand port consumes [:, root_body], not world zero or
    # a principal-frame COM. No engine step or MjData is needed for installation.
    hand_meta = load_robot_mjcf_meta(layout.scene_xml_path).subtree("left_root")
    sim.plant.control = torch.zeros(2, model.nu)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [2**-0.5, 0.0, 0.0, 2**-0.5]])
    sim.plant.body_state = lambda selected: (
        torch.zeros(2, len(selected), 3),
        quaternions[:, None].expand(-1, len(selected), -1),
        torch.zeros(2, len(selected), 3),
        torch.zeros(2, len(selected), 3),
    )
    hand = NativeArticulation(layout.hands[0], hand_meta, sim, hand=True)
    force = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    hand._write_root_wrench(force, torch.zeros_like(force))
    com = hand.body_inertial_properties().com_position_body_m[:, 0]
    expected_torque = torch.cross(quat_apply(quaternions, com), force, dim=-1)
    torch.testing.assert_close(
        sim.plant.control[:, hand._root_drives], torch.cat((force, expected_torque), -1)
    )
    actual = sim.gravity_world_m_per_s2()
    assert sim.set_gravity_world_m_per_s2(actual) == actual
    env = SimpleNamespace(sim=sim, common_step_counter=1000)
    with pytest.raises(NotImplementedError, match="constant constructor"):
        gravity_curriculum(env, torch.tensor([1]), schedule_steps=10)
    with pytest.raises(NotImplementedError, match="native dynamic gravity"):
        gravity_curriculum(env, torch.tensor([1]), schedule_steps=0, full_g=3.0)
    # Mutating CPU metadata cannot claim an executing Warp/Newton gravity update.
    model.opt.gravity[:] = 0
    assert sim.gravity_world_m_per_s2() == actual
    assert not hasattr(sim, "model")


class _Root(FloatingRootView):
    def __init__(self):
        self.device = "cpu"
        self.position = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]).expand(
            2, -1, -1
        )
        self.quaternion = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [2**-0.5, 0.0, 0.0, 2**-0.5]]]
        ).expand(2, -1, -1)
        self.properties = BodyInertialProperties(
            ("root", "child"),
            torch.tensor([[2.0, 1.0], [3.0, 4.0]]),
            torch.tensor([[[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]]).expand(2, -1, -1),
            torch.diag_embed(
                torch.tensor(
                    [
                        [[1.0, 1.0, 2.0], [2.0, 2.0, 5.0]],
                        [[3.0, 3.0, 7.0], [4.0, 4.0, 11.0]],
                    ]
                )
            ),
        )
        self.writes = []

    def body_inertial_properties(self):
        return self.properties

    def _body_pose(self):
        return self.position, self.quaternion

    def _body_vel(self):
        return torch.zeros_like(self.position), torch.zeros_like(self.position)

    def _write_root_wrench(self, force, torque):
        self.writes.append((force.clone(), torque.clone()))


def test_floating_root_uses_per_world_assembly_mass_com_and_inertia():
    root = _Root()
    target = torch.zeros(2, 13)
    target[:, 0] = 0.2
    target[:, 3:7] = torch.tensor([np.cos(0.05), 0.0, 0.0, np.sin(0.05)])
    control = NativeRootControlCfg(
        position_stiffness=10,
        position_damping=0,
        rotation_stiffness=20,
        rotation_damping=0,
        force_limit_n=None,
        torque_limit_nm=None,
    )
    force, torque = root.apply_root_pose_target(
        target, torch.ones(2, dtype=torch.bool), control
    )
    torch.testing.assert_close(force, torch.tensor([[6.0, 0.0, 0.0], [14.0, 0.0, 0.0]]))
    # Analytic planar two-body inertia about the assembly COM, then root-COM lever arm.
    expected = []
    for masses, own_izz in ((np.array([2.0, 1.0]), 7.0), (np.array([3.0, 4.0]), 18.0)):
        points = np.array([[0.1, 0.0], [1.0, 0.1]])
        center = np.average(points, axis=0, weights=masses)
        izz = own_izz + np.sum(masses * np.sum((points - center) ** 2, axis=-1))
        expected.append(2.0 * izz - center[1] * (2.0 * masses.sum()))
    torch.testing.assert_close(
        torque[:, 2], torch.tensor(expected, dtype=torch.float32)
    )
    torch.testing.assert_close(torque[:, :2], torch.zeros(2, 2))
    force, torque = root.apply_root_pose_target(
        target, torch.tensor([True, False]), control
    )
    torch.testing.assert_close(force[1], torch.zeros(3))
    torch.testing.assert_close(torque[1], torch.zeros(3))


def test_actual_global_gravity_curriculum_readback_updates_named_gravcomp(
    monkeypatch, registered_asset
):
    adapter, _, _, _, _, writes = _articulation(registered_asset)
    requests = []
    gravity = [0.0, 0.0, -9.81]

    def set_gravity(value):
        requests.append(value)
        gravity[:] = value

    view = SimpleNamespace(get_gravity=lambda: tuple(gravity), set_gravity=set_gravity)
    monkeypatch.setitem(
        sys.modules, "carb", SimpleNamespace(Float3=lambda *values: tuple(values))
    )
    sim = IsaacSimAdapter(SimpleNamespace(physics_sim_view=view), "cpu")
    sim.add_gravity_compensator(adapter, 0.5)
    mass = adapter.body_inertial_properties().mass_kg
    torch.testing.assert_close(writes[-1][0][..., 2], mass * 9.81 * 0.5)
    env = SimpleNamespace(sim=sim, common_step_counter=5)
    result = gravity_curriculum(env, torch.tensor([1]), schedule_steps=10, full_g=8)
    assert requests == [(0.0, 0.0, -4.0)]  # One global setter despite a subset reset.
    assert result["gravity_z"] == -4
    torch.testing.assert_close(writes[-1][0][..., 2], mass * 2)
    assert writes[-1][2]["body_ids"] == adapter._isaac_body_ids.tolist()
    assert writes[-1][2]["is_global"] is True
    adapter.reset(torch.tensor([1]))
    torch.testing.assert_close(writes[-1][0][..., 2], mass * 2)
    assert not hasattr(sim, "model")


def test_normal_subset_reset_restores_gravcomp_after_sdk_scene_clears_wrenches(
    monkeypatch, registered_asset
):
    adapter, _, _, _, _, _ = _articulation(registered_asset)
    # This protocol reproduces the pinned SDK's selected-row wrench clearing;
    # the manager, scene adapter, articulation adapter and curriculum are real.
    scheduled = torch.zeros(2, len(adapter.body_names), 3)
    events, forwarded = [], []

    def sdk_asset_reset(env_ids=None):
        rows = torch.arange(2) if env_ids is None else env_ids
        events.append(("asset_clear", tuple(rows.tolist())))
        scheduled[rows] = 0

    def sdk_wrench(forces, torques, *, body_ids, is_global):
        assert is_global and not torch.count_nonzero(torques)
        events.append(("scheduled_wrench", None))
        scheduled[:, body_ids] = forces

    def sdk_scene_reset(env_ids=None):
        events.append(("scene_reset", None))
        sdk_asset_reset(env_ids)

    adapter._art.reset = sdk_asset_reset
    adapter._art.set_external_force_and_torque = sdk_wrench
    sdk_scene = SimpleNamespace(
        cfg=SimpleNamespace(num_envs=2),
        env_origins=torch.zeros(2, 3),
        reset=sdk_scene_reset,
        write_data_to_sim=lambda: forwarded.append(scheduled.clone()),
    )
    scene = _isaac_scene_type(monkeypatch)(sdk_scene, "cpu", SimpleNamespace(current_time=0.0))
    scene._clock = adapter._clock
    scene.entities["robot"] = adapter
    gravity = [0.0, 0.0, -9.81]

    def sdk_set_gravity(value):
        gravity[:] = value

    monkeypatch.setitem(
        sys.modules, "carb", SimpleNamespace(Float3=lambda *values: tuple(values))
    )
    sim = IsaacSimAdapter(
        SimpleNamespace(
            physics_sim_view=SimpleNamespace(
                get_gravity=lambda: tuple(gravity),
                set_gravity=sdk_set_gravity,
            )
        ),
        "cpu",
    )
    sim.add_gravity_compensator(adapter, 0.5)
    env = ManagerBasedRlEnv(
        ManagerEnvCfg(
            num_envs=2,
            decimation=4,
            physics_dt=0.005,
            curriculum={
                "gravity": CurriculumTermCfg(
                    func=gravity_curriculum,
                    params={"schedule_steps": 0, "full_g": 4.0},
                )
            },
        ),
        scene,
        sim,
        "cpu",
    )
    events.clear()
    env.reset(env_ids=torch.tensor([1]))
    mass = adapter.body_inertial_properties().mass_kg
    expected = mass[..., None] * mass.new_tensor([0.0, 0.0, 2.0])
    assert len(forwarded) == 1
    torch.testing.assert_close(forwarded[-1][:, adapter._isaac_body_ids], expected)
    assert all(rows == (1,) for name, rows in events if name == "asset_clear")
    last_wrench = max(
        i for i, (name, _) in enumerate(events) if name == "scheduled_wrench"
    )
    assert all(
        i < last_wrench
        for i, (name, _) in enumerate(events)
        if name in ("asset_clear", "scene_reset")
    )
    assert tuple(gravity) == (0.0, 0.0, -4.0)
    # The untouched world keeps the GLOBAL gravity update; the selected world's
    # nonzero request survives the final SDK clear and reaches ordinary write.
    assert bool((forwarded[-1][..., 2] > 0).all())


def test_unsupported_model_swap_rejects_before_harvest_or_fabricated_fields():
    class Command(MultiObjectMixin):
        pass

    command = Command()
    command._env = SimpleNamespace(sim=SimpleNamespace())
    with pytest.raises(NotImplementedError, match="executing per-world MuJoCo"):
        command._init_object_swap(SimpleNamespace())
