"""Named native views and root control shared by PhysX, MJWarp and Newton."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_box_minus,
    quat_mul,
)

from envs.scene.native import NativeRootControlCfg


class _FreshnessClock:
    """Monotonic token shared by all adapters; bump = invalidate caches."""

    def __init__(self) -> None:
        self.token = 0

    def bump(self) -> None:
        self.token += 1


class _Cached:
    """Per-clock-token memo for the engine's named state reads."""

    def __init__(self, clock: _FreshnessClock) -> None:
        self._clock = clock
        self._token = -1
        self._store: dict[str, torch.Tensor] = {}

    def get(self, key: str, fn) -> torch.Tensor:
        if self._token != self._clock.token:
            self._store.clear()
            self._token = self._clock.token
        v = self._store.get(key)
        if v is None:
            v = fn()
            self._store[key] = v
        return v


# ─── Robot (articulation) ────────────────────────────────────────────────────


class ArticulationData:
    """Duck-typed ``EntityData`` view for the robot articulation."""

    def __init__(self, adapter: "NamedArticulationView") -> None:
        self._a = adapter

    # joint space (entity == MJCF order)
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._a._joint_pos()

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._a._joint_vel()

    @property
    def joint_pos_target(self) -> torch.Tensor:
        return self._a._joint_pos_target

    @property
    def soft_joint_pos_limits(self) -> torch.Tensor:
        return self._a._soft_limits

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self._a._default_joint_pos

    @property
    def default_joint_vel(self) -> torch.Tensor:
        return self._a._default_joint_vel

    @property
    def qfrc_actuator(self) -> torch.Tensor:
        # applied joint efforts (implicit PD), entity joint order
        return self._a._joint_effort()

    @property
    def default_root_state(self) -> torch.Tensor:
        return self._a._default_root_state

    # bodies (entity == MJCF order, world excluded)
    @property
    def body_link_pos_w(self) -> torch.Tensor:
        return self._a._body_pose()[0]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        return self._a._body_pose()[1]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        return self._a._body_vel()[0]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        return self._a._body_vel()[1]

    # root == first MJCF body (fixed-base chain anchor), matching mjlab's
    # ``indexing.root_body_id`` semantics for this entity.
    @property
    def root_link_pos_w(self) -> torch.Tensor:
        return self.body_link_pos_w[:, 0]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        return self.body_link_quat_w[:, 0]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        return self.body_link_lin_vel_w[:, 0]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        return self.body_link_ang_vel_w[:, 0]

    # sites (MJCF site frames rigidly attached to parent bodies)
    @property
    def site_pos_w(self) -> torch.Tensor:
        return self._a._site_kin()[0]

    @property
    def site_quat_w(self) -> torch.Tensor:
        return self._a._site_kin()[1]

    @property
    def site_lin_vel_w(self) -> torch.Tensor:
        return self._a._site_kin()[2]

    @property
    def site_ang_vel_w(self) -> torch.Tensor:
        return self._a._site_kin()[3]


class NamedArticulationView:
    """Names, derived sites and data views independent of the simulation engine."""

    def find_sites(self, names, preserve_order: bool = False):
        from mjlab.utils.lab_api.string import resolve_matching_names

        return resolve_matching_names(names, self.site_names, preserve_order)

    def find_joints(self, names, preserve_order: bool = False):
        from mjlab.utils.lab_api.string import resolve_matching_names

        return resolve_matching_names(names, self.joint_names, preserve_order)

    def find_bodies(self, names, preserve_order: bool = False):
        from mjlab.utils.lab_api.string import resolve_matching_names

        return resolve_matching_names(names, self.body_names, preserve_order)

    def find_joints_by_actuator_names(self, names):
        """Actuator names OR joint names -> entity joint ids (input order)."""
        joint_names = []
        for n in names:
            if n in self._meta.actuator_joint:
                joint_names.append(self._meta.actuator_joint[n])
            elif n in self.joint_names:
                joint_names.append(n)
            else:
                raise KeyError(f"unknown actuator/joint name {n!r}")
        return [self.joint_names.index(j) for j in joint_names], joint_names

    def _site_kin(self):
        def compute():
            pos, quat = self._body_pose()
            lin, ang = self._body_vel()
            bp = pos[:, self._site_parent]  # (B, S, 3)
            bq = quat[:, self._site_parent]
            bl = lin[:, self._site_parent]
            ba = ang[:, self._site_parent]
            lpos = self._site_lpos.unsqueeze(0).expand(bp.shape[0], -1, -1)
            lquat = self._site_lquat.unsqueeze(0).expand(bq.shape[0], -1, -1)
            r = quat_apply(bq, lpos)
            spos = bp + r
            squat = quat_mul(bq, lquat)
            slin = bl + torch.cross(ba, r, dim=-1)
            return spos, squat, slin, ba

        key = "site_kin"
        if self._cache._token != self._clock.token or key not in self._cache._store:
            out = compute()
            # stash as tuple via individual keys
            self._cache.get("_sync", lambda: torch.zeros(1))
            self._cache._store[key] = out  # type: ignore[assignment]
        return self._cache._store[key]



class FloatingRootView:
    def apply_root_pose_target(
        self,
        target_state_w: torch.Tensor,
        support: torch.Tensor,
        control: NativeRootControlCfg,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute WORLD PD at every physics substep, about the subtree COM."""
        pos, quat = self._body_pose()
        lin, omega = self._body_vel()
        current = torch.cat([pos[:, 0], quat[:, 0], lin[:, 0], omega[:, 0]], -1)
        target = torch.where(support[:, None], target_state_w, current)
        acceleration = control.position_stiffness * (
            target[:, :3] - current[:, :3]
        ) + control.position_damping * (target[:, 7:10] - current[:, 7:10])
        angular_acceleration = control.rotation_stiffness * quat_box_minus(
            target[:, 3:7], current[:, 3:7]
        ) + control.rotation_damping * (target[:, 10:13] - current[:, 10:13])
        body_com = pos + quat_apply(quat, self._ipos.expand_as(pos))
        total_mass = self._mass.sum()
        center = (body_com * self._mass[None, :, None]).sum(1) / total_mass
        r = body_com - center[:, None]
        rotation = matrix_from_quat(quat_mul(quat, self._inertia_quat.expand_as(quat)))
        inertia = (
            rotation @ torch.diag_embed(self._inertia) @ rotation.transpose(-1, -2)
        )
        eye = torch.eye(3, device=self.device, dtype=pos.dtype)
        parallel = (
            r.square().sum(-1)[..., None, None] * eye
            - r[..., :, None] * r[..., None, :]
        )
        inertia = (inertia + self._mass[None, :, None, None] * parallel).sum(1)
        force = torch.where(support[:, None], total_mass * acceleration, 0.0)
        torque = torch.where(
            support[:, None],
            (inertia @ angular_acceleration[..., None]).squeeze(-1),
            0.0,
        )
        gravity = (
            -total_mass
            * control.gravity_compensation
            * force.new_tensor(control.gravity_world_m_per_s2)
        )
        force = _limit_vector(force + gravity, control.force_limit_n)
        # Report torque about root-link COM; the engine maps its application point.
        torque = _limit_vector(
            torque + torch.cross(center - body_com[:, 0], force, dim=-1),
            control.torque_limit_nm,
        )
        self._write_root_wrench(force, torque)
        return force, torque

    def _write_root_wrench(self, force, torque):
        raise NotImplementedError


def _limit_vector(vector: torch.Tensor, limit: float | None) -> torch.Tensor:
    if limit is None:
        return vector
    return vector * (
        float(limit) / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ).clamp(max=1.0)


class NativeRobotAdapter(NamedArticulationView):
    """The named robot view of independent FREE hand articulations.

    Each child still owns its real articulation and root. Concatenation is only
    the existing robot entity's named scalar/body/site view, not a welded robot.
    """

    is_fixed_base = False

    def __init__(self, hands, *, scene_meta, clock, device):
        from types import SimpleNamespace

        import mujoco

        self.hands: dict[str, NamedArticulationView] = hands
        self.device = device
        self._meta = scene_meta
        self._clock = clock
        self._parts = tuple(hands.values())
        self.joint_names = [n for a in self._parts for n in a.joint_names]
        self.body_names = [n for a in self._parts for n in a.body_names]
        self.site_names = [n for a in self._parts for n in a.site_names]
        self.num_joints = len(self.joint_names)
        self._joint_pos_target = torch.cat(
            [a._joint_pos_target for a in self._parts], 1
        )
        self._soft_limits = torch.cat([a._soft_limits for a in self._parts], 1)
        self._default_joint_pos = torch.cat(
            [a._default_joint_pos for a in self._parts], 1
        )
        self._default_joint_vel = torch.cat(
            [a._default_joint_vel for a in self._parts], 1
        )
        self._default_root_state = self._parts[0]._default_root_state
        self.cfg = SimpleNamespace(
            spec_fn=lambda: mujoco.MjSpec.from_file(scene_meta.xml_path)
        )
        self.data = ArticulationData(self)

    def _joint_pos(self):
        return torch.cat([a._joint_pos() for a in self._parts], 1)

    def native_command_limits(self, actuator_names):
        return self._meta.command_limits(tuple(actuator_names))

    def _joint_vel(self):
        return torch.cat([a._joint_vel() for a in self._parts], 1)

    def _joint_effort(self):
        return torch.cat([a._joint_effort() for a in self._parts], 1)

    def _body_pose(self):
        poses = [a._body_pose() for a in self._parts]
        return tuple(torch.cat([p[i] for p in poses], 1) for i in range(2))

    def _body_vel(self):
        velocities = [a._body_vel() for a in self._parts]
        return tuple(torch.cat([v[i] for v in velocities], 1) for i in range(2))

    def _site_kin(self):
        sites = [a._site_kin() for a in self._parts]
        return tuple(torch.cat([s[i] for s in sites], 1) for i in range(4))

    def write_joint_state_to_sim(
        self, position, velocity, joint_ids=None, env_ids=None
    ):
        if joint_ids is not None:
            raise ValueError(
                "native robot reset requires the complete named scalar state"
            )
        start = 0
        for part in self._parts:
            stop = start + part.num_joints
            part.write_joint_state_to_sim(
                position[:, start:stop], velocity[:, start:stop], env_ids=env_ids
            )
            start = stop
        rows = slice(None) if env_ids is None else env_ids
        self._joint_pos_target[rows] = position

    def set_joint_position_target(self, target, joint_ids=None):
        ids = list(range(self.num_joints)) if joint_ids is None else list(joint_ids)
        start = 0
        for part in self._parts:
            columns = [
                i
                for i, j in enumerate(ids)
                if start <= int(j) < start + part.num_joints
            ]
            if columns:
                local = [int(ids[i]) - start for i in columns]
                part.set_joint_position_target(target[:, columns], joint_ids=local)
            start += part.num_joints
        self._joint_pos_target[:, ids] = target

    def reset(self, env_ids=None):
        for part in self._parts:
            part.reset(env_ids)
        self._clock.bump()


# ─── Object (single free rigid body) ─────────────────────────────────────────


class _ObjIndexing:
    def __init__(self, body_ids: torch.Tensor) -> None:
        self.body_ids = body_ids
