"""mjlab-Entity-compatible adapters over Isaac assets, reading PhysX views with
a freshness token. Conventions: wxyz quats, world frame, MJCF ordering."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import (
    quat_apply,
)

from simulator.mjcf_meta import RobotMjcfMeta, site_table
from simulator.native.entity import (
    ArticulationData,
    BodyInertialProperties,
    FloatingRootView,
    NamedArticulationView,
    _Cached,
    _FreshnessClock,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q[..., [3, 0, 1, 2]]


def _installed_body_properties(view, body_names, body_ids, count, device):
    """Read post-install PhysX values once; tensor inertia is body-frame already."""
    mass = view.get_masses().reshape(count, -1).to(device)[:, body_ids]
    com = view.get_coms().reshape(count, -1, 7).to(device)[:, body_ids, :3]
    # PhysX stores the nine body-frame inertia entries in column-major order.
    inertia = view.get_inertias().reshape(count, -1, 3, 3).transpose(-1, -2)
    inertia = inertia.to(device)[:, body_ids]
    return BodyInertialProperties(
        tuple(body_names), mass.detach().clone(), com.detach().clone(), inertia.detach().clone(),
    )


class ArticulationAdapter(NamedArticulationView):
    """MJCF-ordered ``Entity`` view with explicit imported body names."""

    is_fixed_base = True

    def __init__(
        self,
        articulation: "Articulation",
        meta: RobotMjcfMeta,
        clock: _FreshnessClock,
        device: str,
        spec_fn=None,
        *,
        body_name_map: Mapping[str, str] | None = None,
    ) -> None:
        self._art = articulation
        self._meta = meta
        self._clock = clock
        self._cache = _Cached(clock)
        self.device = device

        self.body_names: list[str] = list(meta.body_names)
        self.joint_names: list[str] = list(meta.joint_names)
        self.num_joints = len(self.joint_names)
        self._body_index = {n: i for i, n in enumerate(self.body_names)}

        names = body_name_map or {}
        urdf_names = [names.get(n, n) for n in self.body_names]

        # entity(MJCF) order -> isaac articulation order permutations
        ids, _ = articulation.find_bodies(urdf_names, preserve_order=True)
        self._isaac_body_ids = torch.tensor(ids, dtype=torch.long, device=device)
        jids, _ = articulation.find_joints(self.joint_names, preserve_order=True)
        self._isaac_joint_ids = torch.tensor(jids, dtype=torch.long, device=device)
        self._driven_joint_ids = torch.tensor(
            [self.joint_names.index(n) for n in meta.driven_joint_names],
            dtype=torch.long, device=device,
        )
        # inverse permutation: isaac joint slot -> entity column
        inv = torch.empty_like(self._isaac_joint_ids)
        inv[torch.argsort(self._isaac_joint_ids)] = torch.arange(
            self.num_joints, device=device
        )

        self.site_names, self._site_parent, self._site_lpos, self._site_lquat = (
            site_table(meta, self._body_index, device)
        )

        B = articulation.num_instances
        self._joint_pos_target = torch.zeros(B, self.num_joints, device=device)
        limits = torch.tensor(meta.joint_range, dtype=torch.float32, device=device)
        self._soft_limits = limits.unsqueeze(0).expand(B, -1, -1).contiguous()
        self._default_joint_pos = torch.zeros(B, self.num_joints, device=device)
        self._default_joint_vel = torch.zeros(B, self.num_joints, device=device)

        self._body_properties = _installed_body_properties(
            articulation.root_physx_view, self.body_names, self._isaac_body_ids, B, device,
        )
        self._gravcomp_full: torch.Tensor | None = None  # named WORLD force at link COM
        self._gravcomp_scale = 0.0

        # A standalone MJCF spec factory so the command's CPU-FK reference
        # path (``_compute_robot_ref_state``) works unchanged.
        if spec_fn is None:
            import mujoco

            xml = meta.xml_path
            spec_fn = lambda: mujoco.MjSpec.from_file(xml)  # noqa: E731
        from types import SimpleNamespace

        self.cfg = SimpleNamespace(spec_fn=spec_fn)
        self.data = ArticulationData(self)

    # ── name/index resolution (mjlab API) ──────────────────────────────────

    # ── raw fetches (freshness-cached) ─────────────────────────────────────

    def body_inertial_properties(self):
        """Installed fixed-recipe snapshot in the same order as body state."""
        return self._body_properties

    def _joint_pos(self) -> torch.Tensor:
        def fetch():
            q = self._art.root_physx_view.get_dof_positions().to(self.device)
            return q[:, self._isaac_joint_ids]

        return self._cache.get("joint_pos", fetch)

    def _joint_vel(self) -> torch.Tensor:
        def fetch():
            v = self._art.root_physx_view.get_dof_velocities().to(self.device)
            return v[:, self._isaac_joint_ids]

        return self._cache.get("joint_vel", fetch)

    def _joint_effort(self) -> torch.Tensor:
        return self._art.data.applied_torque[:, self._isaac_joint_ids]

    def _body_pose(self):
        def fetch_pair():
            tf = self._art.root_physx_view.get_link_transforms()
            tf = tf.reshape(self._art.num_instances, -1, 7).to(self.device)
            return tf

        tf = self._cache.get("link_tf", fetch_pair)
        pos = tf[:, self._isaac_body_ids, :3]
        quat = _xyzw_to_wxyz(tf[:, self._isaac_body_ids, 3:7])
        return pos, quat

    def _body_com_velocity(self) -> torch.Tensor:
        def fetch_vel():
            v = self._art.root_physx_view.get_link_velocities()
            return v.reshape(self._art.num_instances, -1, 6).to(self.device)[:, self._isaac_body_ids]

        return self._cache.get("link_vel", fetch_vel)

    def _body_vel(self):
        v = self._body_com_velocity()
        # physx: linear velocity reported at link COM; shift to link origin.
        _, quat = self._body_pose()
        r_com = quat_apply(quat, self._body_properties.com_position_body_m)
        lin_com = v[..., 0:3]
        ang = v[..., 3:6]
        lin_link = lin_com + torch.cross(ang, -r_com, dim=-1)
        return lin_link, ang

    # ── writes (mjlab API) ─────────────────────────────────────────────────

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids=None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        assert joint_ids is None, "full-state writes only"
        ids = self._isaac_joint_ids
        self._art.write_joint_state_to_sim(
            position.to(self.device), velocity.to(self.device),
            joint_ids=ids.tolist(), env_ids=env_ids,
        )
        # keep targets consistent with the teleported pose
        if env_ids is None:
            self._joint_pos_target[:] = position
        else:
            self._joint_pos_target[env_ids] = position
        if self._driven_joint_ids.numel():
            self._art.set_joint_position_target(
                position[:, self._driven_joint_ids].to(self.device),
                joint_ids=ids[self._driven_joint_ids].tolist(), env_ids=env_ids,
            )
        self._clock.bump()

    def set_joint_position_target(
        self, target: torch.Tensor, joint_ids: torch.Tensor | list | None = None
    ) -> None:
        if joint_ids is None:
            ent_ids = torch.arange(self.num_joints, device=self.device)
        elif isinstance(joint_ids, torch.Tensor):
            ent_ids = joint_ids
        else:
            ent_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        if any(self.joint_names[int(i)] not in self._meta.driven_joint_names for i in ent_ids):
            raise ValueError("passive native joints cannot receive position targets")
        self._joint_pos_target[:, ent_ids] = target
        isaac_ids = self._isaac_joint_ids[ent_ids]
        self._art.set_joint_position_target(target, joint_ids=isaac_ids.tolist())

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._art.reset(env_ids=env_ids)
        # Isaac Lab reset zeroes the external wrench; restore the gravity compensation.
        if self._gravcomp_full is not None:
            self._write_gravcomp()

    def init_gravcomp(self, scale: float, gravity_world_m_per_s2) -> None:
        """Per-link world-frame m*g lift at the COM, kept through resets (MJCF gravcomp)."""
        self._gravcomp_scale = float(scale)
        self.update_gravcomp_gravity(gravity_world_m_per_s2)

    def update_gravcomp_gravity(self, gravity_world_m_per_s2) -> None:
        mass = self._body_properties.mass_kg
        self._gravcomp_full = -mass[..., None] * mass.new_tensor(gravity_world_m_per_s2)
        self._write_gravcomp()

    def _write_gravcomp(self) -> None:
        # Always the full buffer: Isaac Lab derives has_external_wrench from the written slice only.
        f = self._gravcomp_full * self._gravcomp_scale
        self._art.set_external_force_and_torque(
            f, torch.zeros_like(f), body_ids=self._isaac_body_ids.tolist(), is_global=True,
        )


class FreeArticulationAdapter(FloatingRootView, ArticulationAdapter):
    """One actual floating articulation; scalar joints remain named DOFs.

    The root is reset through Isaac Lab's public link-state API. During a
    rollout it is driven by an external wrench, never by pose teleportation.
    """

    is_fixed_base = False

    def __init__(self, articulation, meta, clock, device, *, root_body_name):
        super().__init__(articulation, meta, clock, device)
        if articulation.is_fixed_base:
            raise ValueError(f"native articulation {root_body_name!r} has a fixed base")
        if meta.body_names[0] != root_body_name:
            raise ValueError("native root must be the named articulation root body")
        if len(meta.free_roots) != 1 or meta.free_roots[0].body_name != root_body_name:
            raise ValueError("native articulation must contain its declared FREE root")
        self.root_body_name = root_body_name
        b = articulation.num_instances
        self._default_root_state = torch.zeros(b, 13, device=device)
        self._default_root_state[:, :7] = torch.as_tensor(
            meta.free_roots[0].default_pose, dtype=torch.float32, device=device
        )
        self._default_joint_pos[:] = torch.as_tensor(
            meta.joint_default, dtype=torch.float32, device=device
        )
        self._external_force_w = torch.zeros(b, len(self.body_names), 3, device=device)
        self._external_torque_w = torch.zeros_like(self._external_force_w)

    def write_root_state_to_sim(self, root_state, env_ids=None) -> None:
        self._art.write_root_link_state_to_sim(root_state, env_ids=env_ids)
        self._clock.bump()

    def body_com_scalar_jacobians(self) -> torch.Tensor:
        """PhysX WORLD COM Jacobians in named body and scalar-joint order."""
        def fetch():
            view = self._art.root_physx_view
            width = len(self._isaac_joint_ids)
            shape = (self._art.num_instances, len(self._isaac_body_ids), 6, width)
            observed = self._body_com_velocity()
            try:
                jacobians = view.get_jacobians().to(observed)
            except Exception:  # SDK raises Exception when the tensor is unavailable.
                return observed.new_full(shape, torch.nan)
            if jacobians.shape != (*shape[:-1], width + 6):
                return observed.new_full(shape, torch.nan)
            named_bodies = jacobians.index_select(1, self._isaac_body_ids)
            scalar = named_bodies.index_select(-1, self._isaac_joint_ids + 6)
            root_term = named_bodies[..., :6] @ observed[:, :1, :, None]
            scalar_term = scalar @ self._joint_vel()[:, None, :, None]
            predicted = (root_term + scalar_term).squeeze(-1)
            # Keep the chart unavailable if it cannot recover the same endpoint's body rates.
            valid = torch.isclose(predicted, observed, atol=1e-5, rtol=1e-4).all(dim=(1, 2))
            return torch.where(valid[:, None, None, None], scalar, torch.nan)

        return self._cache.get("body_com_scalar_jacobians", fetch)

    def _submit_external_wrenches(self):
        # Lab derives has_external_wrench from the submitted slice. Submit the
        # complete request so clearing one world cannot disable the others.
        self._art.set_external_force_and_torque(
            self._external_force_w, self._external_torque_w,
            body_ids=self._isaac_body_ids.tolist(), is_global=True,
        )

    def write_external_wrench_to_sim(self, forces, torques, env_ids=None, body_ids=None):
        """Replace named WORLD forces/torques at the actual body COMs."""
        rows = torch.arange(self._art.num_instances, device=self.device) if env_ids is None else env_ids
        ids = torch.arange(len(self.body_names), device=self.device) if body_ids is None else torch.as_tensor(
            body_ids, device=self.device, dtype=torch.long,
        )
        self._external_force_w[rows[:, None], ids[None]] = forces.detach()
        self._external_torque_w[rows[:, None], ids[None]] = torques.detach()
        self._submit_external_wrenches()

    def clear_external_wrenches(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        self._external_force_w[rows] = 0
        self._external_torque_w[rows] = 0
        self._submit_external_wrenches()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.clear_external_wrenches(env_ids)

    def _write_root_wrench(self, force, torque):
        self._external_force_w.zero_()
        self._external_torque_w.zero_()
        self._external_force_w[:, 0] = force
        self._external_torque_w[:, 0] = torque
        self._submit_external_wrenches()




class RigidObjectData:
    def __init__(self, adapter: "RigidObjectAdapter") -> None:
        self._a = adapter

    @property
    def default_root_state(self) -> torch.Tensor:
        return self._a._default_root_state

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        return self._a._pose()[0].unsqueeze(1)

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        return self._a._pose()[1].unsqueeze(1)

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        return self._a._vel()[0].unsqueeze(1)

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        return self._a._vel()[1].unsqueeze(1)

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        return self._a._pose()[0]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        return self._a._pose()[1]


class RigidObjectAdapter:
    """mjlab ``Entity`` surface for one object slot (single free body)."""

    is_fixed_base = False

    def joint_coordinate_metadata(self):
        """A physical rigid body has no scalar joint coordinates."""
        return {
            "joint_names": (), "joint_types": (),
            "position_units": (), "velocity_units": (),
        }

    def __init__(
        self,
        obj: "RigidObject",
        clock: _FreshnessClock,
        device: str,
        default_root_state: torch.Tensor,
        *,
        body_name: str,
    ) -> None:
        self._obj = obj
        self._clock = clock
        self._cache = _Cached(clock)
        self.device = device
        self._default_root_state = default_root_state  # (B, 13)
        if list(obj.body_names) != [body_name]:
            raise ValueError("rigid adapter requires its actual single named body")
        self.body_names = [body_name]
        self.root_body_name = body_name
        self.joint_names = ()  # A physical rigid object has no scalar joints.
        view = obj.root_physx_view
        self._body_properties = _installed_body_properties(
            view, self.body_names, [0], obj.num_instances, device,
        )
        self._com_local = self._body_properties.com_position_body_m[:, 0]
        self.data = RigidObjectData(self)
        # world-frame external wrench support probe (isaaclab >= is_global)
        sig = inspect.signature(obj.set_external_force_and_torque)
        self._has_is_global = "is_global" in sig.parameters

    def body_inertial_properties(self):
        return self._body_properties

    def _pose(self):
        def fetch():
            tf = self._obj.root_physx_view.get_transforms().to(self.device)
            return tf

        tf = self._cache.get("tf", fetch)
        return tf[:, :3], _xyzw_to_wxyz(tf[:, 3:7])

    def _vel(self):
        v = self._cache.get(
            "vel", lambda: self._obj.root_physx_view.get_velocities().to(self.device)
        )
        pos, quat = self._pose()
        r_com = quat_apply(quat, self._com_local)
        lin = v[:, 0:3] + torch.cross(v[:, 3:6], -r_com, dim=-1)
        return lin, v[:, 3:6]

    # ── writes ─────────────────────────────────────────────────────────────

    def write_root_state_to_sim(
        self, root_state: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        pose = root_state[:, 0:7].to(self.device)
        lin_link = root_state[:, 7:10].to(self.device)
        ang = root_state[:, 10:13].to(self.device)
        self._obj.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        # convert link-origin linear velocity -> com velocity for physx
        quat = pose[:, 3:7]
        if env_ids is None:
            com_local = self._com_local
        else:
            com_local = self._com_local[env_ids]
        r_com = quat_apply(quat, com_local)
        lin_com = lin_link + torch.cross(ang, r_com, dim=-1)
        vel = torch.cat([lin_com, ang], dim=-1)
        self._obj.write_root_com_velocity_to_sim(vel, env_ids=env_ids)
        self._clock.bump()

    def write_external_wrench_to_sim(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        body_ids=None,
    ) -> None:
        if body_ids is not None and list(body_ids) != [0]:
            raise ValueError("a rigid object wrench selects its sole named body")
        f = forces.reshape(-1, 1, 3)
        t = torques.reshape(-1, 1, 3)
        # Apply at the COM (mujoco xfrc parity); FDR_PIN_AT_COM=0
        # reverts to the default application point for A/B probing.
        import os as _os

        if _os.environ.get("FDR_PIN_AT_COM", "0") == "1":
            com = self._com_local if env_ids is None else self._com_local[env_ids]
            if self._has_is_global:
                link_pos, quat = self._pose()
                if env_ids is not None:
                    link_pos, quat = link_pos[env_ids], quat[env_ids]
                com = link_pos + quat_apply(quat, com)
            pos = com.reshape(-1, 1, 3)
        else:
            pos = None
        if self._has_is_global:
            self._obj.set_external_force_and_torque(
                f, t, positions=pos, env_ids=env_ids, is_global=True
            )
            return
        # world -> body frame (isaaclab applies wrench in body frame)
        _, quat = self._pose()
        if env_ids is not None:
            quat = quat[env_ids]
        from mjlab.utils.lab_api.math import quat_apply_inverse

        q = quat.unsqueeze(1)
        self._obj.set_external_force_and_torque(
            quat_apply_inverse(q, f), quat_apply_inverse(q, t),
            positions=pos, env_ids=env_ids,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._obj.reset(env_ids=env_ids)

    def clear_external_wrenches(self, env_ids=None):
        count = self._obj.num_instances if env_ids is None else len(env_ids)
        zero = torch.zeros(count, 1, 3, device=self.device)
        self.write_external_wrench_to_sim(zero, zero.clone(), env_ids=env_ids)
