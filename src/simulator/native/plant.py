"""Small native plant interface; task/control/replay stay with their existing owners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .contacts import ControlIntervalContacts


def free_root_velocity_to_native(state_w):
    """FREE translation rates are WORLD, angular generalized rates are body-local."""
    return torch.cat((state_w[..., 7:10], quat_apply_inverse(state_w[..., 3:7], state_w[..., 10:13])), -1)


def body_origin_velocity(cvel, position, tree_center):
    """MuJoCo mjOBJ_XBODY world velocity from its subtree-COM spatial chart."""
    angular = cvel[..., :3]
    linear = cvel[..., 3:] + torch.linalg.cross(angular, position - tree_center)
    return linear, angular


def root_origin_wrench(force, torque_at_link_com, quaternion, local_com):
    """Move a WORLD link-COM wrench to the FREE body origin, not another frame."""
    arm = quat_apply(quaternion, local_com.expand_as(force))
    return torch.cat((force, torque_at_link_com + torch.linalg.cross(arm, force)), -1)


def root_drive_indices(model, root_joint_name):
    """Resolve actual unit WORLD wrench channels; they never enter finger coordinates."""
    import mujoco

    joint = model.joint(root_joint_name).id
    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("native wrist requires the declared FREE joint")
    selected = np.flatnonzero(
        (model.actuator_trnid[:, 0] == joint)
        & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINTINPARENT)
    )
    axes = []
    for axis in np.eye(6):
        ids = selected[np.all(model.actuator_gear[selected] == axis, axis=1)]
        if len(ids) != 1:
            raise ValueError("native root must retain one authored unit WORLD motor per wrench axis")
        a = int(ids[0])
        if (model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_gainprm[a, 0] != 1
                or model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_NONE
                or model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_ctrllimited[a] or model.actuator_forcelimited[a]):
            raise ValueError("native WORLD root drive changes the declared wrench authority")
        axes.append(a)
    # A different extra root transmission would add a second physical authority.
    attached = np.flatnonzero(
        np.isin(model.actuator_trntype, (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT))
        & (model.actuator_trnid[:, 0] == joint)
    )
    if set(attached) != set(axes):
        raise ValueError("native FREE root has additional undeclared actuator channels")
    return np.asarray(axes, dtype=np.int64)


@dataclass
class NativePlantSnapshot:
    physics: object
    contact_facts: dict[str, dict[str, torch.Tensor]]
    contact_support: torch.Tensor
    control: torch.Tensor
    contact_sample_time_s: torch.Tensor
    external_body_wrench_w: torch.Tensor
    execution_steps: torch.Tensor | None
    contact_sample_execution_step: torch.Tensor | None


class NativePlant(ABC):
    """A coupled bilateral plant in authored MJCF order, independently batched by world."""

    def _init_buffers(self, model, num_envs, device, contact_pairs):
        self.model, self.num_envs = model, int(num_envs)
        self.device = str(device)
        self.contact_pairs = contact_pairs
        self.contact_interval = None
        self._contact_differential_dofs = None
        # Only a backend with an observed fixed-step installation supplies this
        # clock. Raw solver time remains a separate, unchanged engine state.
        self._execution_steps = None
        self.contact_sample_execution_step = None
        self.control = torch.zeros(self.num_envs, model.nu, device=device)
        # Public native-body order, WORLD force then torque at each body COM.
        # This independent external input never creates an actuator/DOF.
        self.external_body_wrench_w = torch.zeros(self.num_envs, model.nbody, 6, device=device)
        self.applied_body_wrench_w = torch.zeros(self.num_envs, model.nbody, 6, device=device)
        self.applied_body_wrench_support = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.has_applied_wrench_readback = False
        self.contact_support = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.contact_sample_time_s = torch.full((self.num_envs,), float("nan"), dtype=torch.float64, device=device)
        self.contact_facts = {
            name: {
                "normal_force_n": torch.full((self.num_envs,), float("nan"), device=device),
                "force_on_second_w_n": torch.full((self.num_envs, 3), float("nan"), device=device),
                "force_bearing_contact_present": torch.zeros(self.num_envs, dtype=torch.bool, device=device),
                "force_bearing_distance_max_m": torch.full((self.num_envs,), float("nan"), device=device),
            }
            for name in contact_pairs
        }
        self.hand_part_feedback_dofs: dict[str, tuple[int, ...]] = {}

    def configure_hand_part_feedback(self, pair_dofs):
        """Bind optional contact feedback before creating the interval buffers."""
        self.hand_part_feedback_dofs = dict(pair_dofs)
        for name, dofs in self.hand_part_feedback_dofs.items():
            facts = self.contact_facts[name]
            facts["force_bearing_tangent_speed_max_m_s"] = torch.full_like(facts["normal_force_n"], torch.nan)
            if dofs:
                facts["joint_effort_on_second"] = facts["normal_force_n"].new_full(
                    (self.num_envs, len(dofs)), torch.nan)

    def configure_control_interval(self, num_substeps):
        self.contact_interval = ControlIntervalContacts(self, num_substeps)

    def prepare_contact_substep(self):
        if self.contact_interval is not None:
            self.contact_interval.before_substep()

    def record_contact_substep(self):
        if self.contact_interval is not None:
            self.contact_interval.append(self)

    def invalidate_contacts(self, rows):
        if self.contact_interval is not None:
            self.contact_interval.discard(rows)
        self.contact_support[rows] = False
        self.contact_sample_time_s[rows] = float("nan")
        if self.contact_sample_execution_step is not None:
            self.contact_sample_execution_step[rows] = -1
        for facts in self.contact_facts.values():
            for value in facts.values():
                value[rows] = False if value.dtype == torch.bool else torch.nan

    @abstractmethod
    def state(self):
        """Return stopped qpos/qvel in the actual source-native chart."""

    @abstractmethod
    def reset(self, env_ids):
        """Clear selected complete integration/controller rows."""

    @abstractmethod
    def set_state(self, env_ids, qpos, qvel):
        """Install an explicitly authored reset; never called by ordinary advance."""

    @abstractmethod
    def advance(self):
        """One actual physical substep, retaining its solved pair-contact facts."""

    @abstractmethod
    def body_state(self, body_ids):
        """Body origins, WXYZ rotation, origin WORLD linear and WORLD angular rate."""

    @abstractmethod
    def physics_time(self):
        """Actual per-world solver clock, not source time or a nominal counter."""

    def execution_clock(self):
        """Accepted fixed-step clock facts, or None when this backend has no port."""
        return None

    def applied_external_body_wrench(self):
        """Post-solve consumed external body-COM wrench, or None when unsupported."""
        if not self.has_applied_wrench_readback:
            return None
        return dict(
            wrench_w=self.applied_body_wrench_w,
            support=self.applied_body_wrench_support,
            authority="post_solve_mjw_xfrc_applied_WORLD_body_COM_wrench",
        )

    @abstractmethod
    def forward(self):
        """Refresh kinematics without integrating or changing retained solved contact facts."""

    @abstractmethod
    def capture_physics(self):
        """Return complete native integration state, not a Q/Qvel substitute."""

    @abstractmethod
    def restore_physics(self, state):
        """Restore all worlds of the exact same plant."""

    def snapshot(self):
        return NativePlantSnapshot(
            self.capture_physics(),
            {n: {k: v.clone() for k, v in f.items()} for n, f in self.contact_facts.items()},
            self.contact_support.clone(), self.control.clone(), self.contact_sample_time_s.clone(),
            self.external_body_wrench_w.clone(),
            None if self._execution_steps is None else self._execution_steps.clone(),
            None if self.contact_sample_execution_step is None else self.contact_sample_execution_step.clone(),
        )

    def restore(self, snapshot):
        self.restore_physics(snapshot.physics)
        self.control.copy_(snapshot.control)
        self.external_body_wrench_w.copy_(snapshot.external_body_wrench_w)
        self.contact_support.copy_(snapshot.contact_support)
        self.contact_sample_time_s.copy_(snapshot.contact_sample_time_s)
        if self._execution_steps is not None:
            self._execution_steps.copy_(snapshot.execution_steps)
            self.contact_sample_execution_step.copy_(snapshot.contact_sample_execution_step)
        for name, facts in snapshot.contact_facts.items():
            for key, value in facts.items():
                self.contact_facts[name][key].copy_(value)
        if self.contact_interval is not None:
            # A restored last-solve sample remains truthful; it is not evidence
            # that a new complete manager control interval has executed.
            self.contact_interval.discard()
