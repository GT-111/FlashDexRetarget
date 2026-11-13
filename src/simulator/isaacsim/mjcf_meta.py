"""Robot metadata from the MJCF via CPU mujoco: joint order (= motion .pt
columns), PD gains, site frames. The MJCF stays the source of truth."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np
import torch


@dataclass
class SiteMeta:
    name: str
    body_name: str
    local_pos: np.ndarray  # (3,)
    local_quat: np.ndarray  # (4,) wxyz


@dataclass(frozen=True)
class FreeRootMeta:
    name: str
    body_name: str
    qpos_address: int
    dof_address: int
    default_pose: np.ndarray  # p + wxyz, not seven scalar articulation joints


@dataclass(frozen=True)
class MimicMeta:
    joint_name: str
    reference_joint_name: str
    polynomial: np.ndarray  # MJCF q = sum(poly[k] * q_reference**k)


@dataclass
class RobotMjcfMeta:
    xml_path: str
    body_names: list[str]  # MJCF order, world excluded
    joint_names: list[str]  # MJCF order == motion .pt column order
    joint_range: np.ndarray  # (n_jnt, 2)
    joint_damping: np.ndarray  # (n_jnt,)
    joint_armature: np.ndarray  # (n_jnt,)
    joint_frictionloss: np.ndarray  # (n_jnt,)
    actuator_names: list[str]  # MJCF order
    actuator_joint: dict[str, str]  # actuator name -> joint name
    actuator_kp: np.ndarray  # (n_act,) position-servo stiffness
    actuator_kv: np.ndarray  # (n_act,) position-servo damping
    actuator_forcerange: np.ndarray  # (n_act, 2)
    sites: dict[str, SiteMeta] = field(default_factory=dict)
    body_bound: dict[str, float] = field(default_factory=dict)
    free_roots: tuple[FreeRootMeta, ...] = ()
    mimics: tuple[MimicMeta, ...] = ()
    joint_body_names: list[str] = field(default_factory=list)
    joint_qpos_address: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    joint_dof_address: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    joint_default: np.ndarray = field(default_factory=lambda: np.empty(0))
    joint_limited: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    actuator_ctrlrange: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    actuator_effort_limit: np.ndarray = field(default_factory=lambda: np.empty(0))
    body_parent_names: list[str] = field(default_factory=list)
    body_mass: np.ndarray = field(default_factory=lambda: np.empty(0))
    body_ipos: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    body_iquat: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    body_inertia: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def driven_joint_names(self) -> list[str]:
        return list(dict.fromkeys(self.actuator_joint.values()))

    def kp_kv_for_joint(self, joint_name: str) -> tuple[float, float]:
        for act, jnt in self.actuator_joint.items():
            if jnt == joint_name:
                i = self.actuator_names.index(act)
                return float(self.actuator_kp[i]), float(self.actuator_kv[i])
        raise KeyError(f"no actuator drives joint {joint_name!r}")

    def subtree(self, root_body_name: str) -> RobotMjcfMeta:
        """One named articulation, preserving the source's scalar/body ordering."""
        self.body_names.index(root_body_name)
        bodies = {root_body_name}
        for name, parent in zip(self.body_names, self.body_parent_names, strict=True):
            if parent in bodies:
                bodies.add(name)
        bi = [i for i, n in enumerate(self.body_names) if n in bodies]
        ji = [i for i, b in enumerate(self.joint_body_names) if b in bodies]
        joint_names = [self.joint_names[i] for i in ji]
        ai = [
            i
            for i, n in enumerate(self.actuator_names)
            if self.actuator_joint[n] in joint_names
        ]
        actuator_names = [self.actuator_names[i] for i in ai]
        return replace(
            self,
            body_names=[self.body_names[i] for i in bi],
            body_parent_names=[self.body_parent_names[i] for i in bi],
            body_mass=self.body_mass[bi],
            body_ipos=self.body_ipos[bi],
            body_iquat=self.body_iquat[bi],
            body_inertia=self.body_inertia[bi],
            body_bound={n: v for n, v in self.body_bound.items() if n in bodies},
            sites={n: s for n, s in self.sites.items() if s.body_name in bodies},
            joint_names=joint_names,
            joint_body_names=[self.joint_body_names[i] for i in ji],
            joint_qpos_address=self.joint_qpos_address[ji],
            joint_dof_address=self.joint_dof_address[ji],
            joint_default=self.joint_default[ji],
            joint_limited=self.joint_limited[ji],
            joint_range=self.joint_range[ji],
            joint_damping=self.joint_damping[ji],
            joint_armature=self.joint_armature[ji],
            joint_frictionloss=self.joint_frictionloss[ji],
            actuator_names=actuator_names,
            actuator_joint={n: self.actuator_joint[n] for n in actuator_names},
            actuator_kp=self.actuator_kp[ai],
            actuator_kv=self.actuator_kv[ai],
            actuator_forcerange=self.actuator_forcerange[ai],
            actuator_ctrlrange=self.actuator_ctrlrange[ai],
            actuator_effort_limit=self.actuator_effort_limit[ai],
            free_roots=tuple(r for r in self.free_roots if r.body_name in bodies),
            mimics=tuple(m for m in self.mimics if m.joint_name in joint_names),
        )

    def command_limits(self, actuator_names: tuple[str, ...]) -> np.ndarray:
        """Independent position commands, including bounds induced by mimics."""
        limits = []
        for actuator in actuator_names:
            a = self.actuator_names.index(actuator)
            joint = self.actuator_joint[actuator]
            j = self.joint_names.index(joint)
            bound = self.actuator_ctrlrange[a].copy()
            if self.joint_limited[j]:
                bound = np.array(
                    [
                        max(bound[0], self.joint_range[j, 0]),
                        min(bound[1], self.joint_range[j, 1]),
                    ]
                )
            for mimic in self.mimics:
                if mimic.reference_joint_name != joint:
                    continue
                follower = self.joint_names.index(mimic.joint_name)
                if not self.joint_limited[follower]:
                    continue
                p = mimic.polynomial
                if np.any(p[2:] != 0) or p[1] == 0:
                    raise ValueError(
                        "native command bounds require linear mimic equalities"
                    )
                induced = np.sort((self.joint_range[follower] - p[0]) / p[1])
                bound = np.array([max(bound[0], induced[0]), min(bound[1], induced[1])])
            if not np.isfinite(bound).all() or bound[0] >= bound[1]:
                raise ValueError(
                    f"position command {actuator!r} has no finite valid range"
                )
            limits.append(bound)
        return np.asarray(limits).reshape(-1, 2)


def load_robot_mjcf_meta(xml_path: str | Path) -> RobotMjcfMeta:
    xml_path = str(xml_path)
    m = mujoco.MjModel.from_xml_path(xml_path)

    def _name(objtype: mujoco.mjtObj, i: int) -> str:
        n = mujoco.mj_id2name(m, objtype, i)
        assert n is not None
        return n

    body_names = [
        _name(mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)
    ]  # skip world
    # Articulation DOFs contain scalar joints only. FREE roots have a separate
    # pose/twist API; treating their first qpos slot as a scalar loses rotation.
    scalar_joints = [
        j
        for j in range(m.njnt)
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    ]
    joint_names = [_name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in scalar_joints]
    dof = [int(m.jnt_dofadr[j]) for j in scalar_joints]
    joint_ids = set(scalar_joints)
    position_actuators = [
        i
        for i in range(m.nu)
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
        and int(m.actuator_trnid[i, 0]) in joint_ids
    ]
    actuator_names = [
        _name(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in position_actuators
    ]
    actuator_joint = {
        _name(mujoco.mjtObj.mjOBJ_ACTUATOR, i): _name(
            mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0])
        )
        for i in position_actuators
    }
    # MuJoCo position servo: gainprm[0] = kp, biasprm = [0, -kp, -kv].
    actuator_kp = m.actuator_gainprm[position_actuators, 0].copy()
    actuator_kv = -m.actuator_biasprm[position_actuators, 2].copy()
    effort_limits = []
    for a in position_actuators:
        joint = int(m.actuator_trnid[a, 0])
        limit = np.inf
        if m.actuator_forcelimited[a]:
            limit = float(np.abs(m.actuator_forcerange[a]).max())
        if m.jnt_actfrclimited[joint]:
            limit = min(limit, float(np.abs(m.jnt_actfrcrange[joint]).max()))
        effort_limits.append(limit)
    free_roots = tuple(
        FreeRootMeta(
            name=_name(mujoco.mjtObj.mjOBJ_JOINT, j),
            body_name=_name(mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])),
            qpos_address=int(m.jnt_qposadr[j]),
            dof_address=int(m.jnt_dofadr[j]),
            default_pose=m.qpos0[
                int(m.jnt_qposadr[j]) : int(m.jnt_qposadr[j]) + 7
            ].copy(),
        )
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    )
    mimics = tuple(
        MimicMeta(
            joint_name=_name(mujoco.mjtObj.mjOBJ_JOINT, int(m.eq_obj1id[e])),
            reference_joint_name=_name(mujoco.mjtObj.mjOBJ_JOINT, int(m.eq_obj2id[e])),
            polynomial=m.eq_data[e, :5].copy(),
        )
        for e in range(m.neq)
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_obj2id[e] >= 0
    )

    sites: dict[str, SiteMeta] = {}
    for s in range(m.nsite):
        sname = _name(mujoco.mjtObj.mjOBJ_SITE, s)
        bid = int(m.site_bodyid[s])
        sites[sname] = SiteMeta(
            name=sname,
            body_name=_name(mujoco.mjtObj.mjOBJ_BODY, bid),
            local_pos=m.site_pos[s].copy(),
            local_quat=m.site_quat[s].copy(),
        )

    # Per-body collision reach: max over collision geoms of |offset| + rbound. Raw
    # MJCF ships collision geoms contype=0, so also match "collision_*" names.
    body_bound: dict[str, float] = {}
    for g in range(m.ngeom):
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        active = m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0
        if not active and not gname.startswith("collision"):
            continue
        bname = _name(mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        reach = float(np.linalg.norm(m.geom_pos[g])) + float(m.geom_rbound[g])
        body_bound[bname] = max(body_bound.get(bname, 0.0), reach)

    return RobotMjcfMeta(
        xml_path=xml_path,
        body_names=body_names,
        joint_names=joint_names,
        joint_range=m.jnt_range[scalar_joints].copy(),
        joint_damping=np.array([m.dof_damping[d] for d in dof]),
        joint_armature=np.array([m.dof_armature[d] for d in dof]),
        joint_frictionloss=np.array([m.dof_frictionloss[d] for d in dof]),
        actuator_names=actuator_names,
        actuator_joint=actuator_joint,
        actuator_kp=actuator_kp,
        actuator_kv=actuator_kv,
        actuator_forcerange=m.actuator_forcerange[position_actuators].copy(),
        sites=sites,
        body_bound=body_bound,
        free_roots=free_roots,
        mimics=mimics,
        joint_body_names=[
            _name(mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])) for j in scalar_joints
        ],
        joint_qpos_address=m.jnt_qposadr[scalar_joints].copy(),
        joint_dof_address=m.jnt_dofadr[scalar_joints].copy(),
        joint_default=m.qpos0[m.jnt_qposadr[scalar_joints]].copy(),
        actuator_ctrlrange=np.where(
            m.actuator_ctrllimited[position_actuators, None],
            m.actuator_ctrlrange[position_actuators],
            np.array([-np.inf, np.inf]),
        ),
        actuator_effort_limit=np.asarray(effort_limits),
        joint_limited=m.jnt_limited[scalar_joints].astype(bool),
        body_parent_names=[
            _name(mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[b]))
            for b in range(1, m.nbody)
        ],
        body_mass=m.body_mass[1:].copy(),
        body_ipos=m.body_ipos[1:].copy(),
        body_iquat=m.body_iquat[1:].copy(),
        body_inertia=m.body_inertia[1:].copy(),
        gravity=m.opt.gravity.copy(),
    )


def site_table(
    meta: RobotMjcfMeta, body_index: dict[str, int], device: str
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """(site_names, parent_body_idx (S,), local_pos (S,3), local_quat (S,4))."""
    names = list(meta.sites.keys())
    parent = torch.tensor(
        [body_index[meta.sites[n].body_name] for n in names],
        dtype=torch.long,
        device=device,
    )
    pos = torch.tensor(
        np.array([meta.sites[n].local_pos for n in names]).reshape(-1, 3),
        dtype=torch.float32,
        device=device,
    )
    quat = torch.tensor(
        np.array([meta.sites[n].local_quat for n in names]).reshape(-1, 4),
        dtype=torch.float32,
        device=device,
    )
    return names, parent, pos, quat
