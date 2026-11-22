"""Named PhysX report patches on the shared native control interval."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import quat_apply

from simulator.native.contacts import ControlIntervalContacts, object_pair_interval_facts


def create_named_contact_view(sim_view, sensors, filters):
    """Ask the SDK for the exact per-world list and check its resolved order."""
    try:
        view = sim_view.create_rigid_contact_view(sensors, filters, 128 * len(sensors))
        actual_sensors = list(view.sensor_paths)
        actual_filters = [path for row in view.filter_paths for path in (
            row if isinstance(row, (list, tuple)) else (row,)
        )]
    except Exception as error:  # SDK report setup can fail independently of plant creation.
        return None, str(error)
    if (view.sensor_count != len(sensors) or view.filter_count != len(filters[0])
            or actual_sensors != sensors or actual_filters != [path for row in filters for path in row]
            or view.max_contact_data_count <= 0):
        return None, "PhysX did not resolve the requested ordered sensor and filter prims"
    return view, None


def _patch_owners(counts, starts, slots):
    """Map each fixed SDK buffer slot to its sensor row and filter column."""
    count, begin = counts.long().flatten(), starts.long().flatten()
    end = begin + count
    capacity = len(slots)
    ordered, order = torch.sort(torch.where(count > 0, begin, capacity))
    owner = order[(torch.searchsorted(ordered, slots, right=True) - 1).clamp_min(0)]
    active = (begin[owner] <= slots) & (slots < end[owner])
    # The SDK does not certify completeness when the shared view buffer fills.
    supported = ~((count > 0) & (end >= capacity)).any()
    return owner.div(counts.shape[1], rounding_mode="floor"), owner % counts.shape[1], active, supported


def _point_velocity(data, rows, bodies, point):
    return data.body_link_lin_vel_w[rows, bodies] + torch.linalg.cross(
        data.body_link_ang_vel_w[rows, bodies], point - data.body_link_pos_w[rows, bodies],
    )


class IsaacHandPartContact:
    """One object-side reporter against named links or fixed colliders."""

    def __init__(self, view, hand, hand_body_ids, obj, body_id, address, device, *, digit_groups=None):
        self.view, self.hand, self.obj, self.body_id = view, hand, obj, body_id
        self.address = address
        self._hand_body_ids = torch.as_tensor(hand_body_ids, dtype=torch.long, device=device)
        self.digit_groups = digit_groups or {}
        self._body_properties = obj.body_inertial_properties()
        sample = self._body_properties.mass_kg[:, 0]
        number = sample.new_full(sample.shape, torch.nan)
        vector = sample.new_full((len(sample), 3), torch.nan)
        self.facts = {
            "normal_force_n": number,
            "force_on_second_w_n": vector,
            "force_bearing_contact_present": torch.zeros_like(number, dtype=torch.bool),
            "force_bearing_distance_max_m": number.clone(),
            "force_bearing_tangent_speed_max_m_s": number.clone(),
            "moment_on_second_about_root_w_nm": vector.clone(),
        }
        if len(obj.joint_names):
            self.facts["joint_effort_on_second"] = sample.new_full((len(sample), len(obj.joint_names)), torch.nan)
        if self.digit_groups:
            self._digit_membership = torch.tensor(
                [[hand.body_names[int(body)] in names for names in self.digit_groups.values()]
                 for body in hand_body_ids], dtype=torch.bool, device=device,
            )
            self.facts["digit_group_force_n"] = sample.new_full((len(sample), len(self.digit_groups)), torch.nan)
        if view is not None:
            self._slots = torch.arange(view.max_contact_data_count, device=device)

    def discard(self, rows):
        for value in self.facts.values():
            value[rows] = False if value.dtype == torch.bool else torch.nan

    @torch.no_grad()
    def read(self, dt):
        supported = torch.zeros_like(self.facts["force_bearing_contact_present"])
        if self.view is None:
            return supported
        try:
            magnitude, point, normal, separation, counts, starts = self.view.get_contact_data(dt=dt)
        except Exception:  # Pinned SDK raises Exception when the report is absent.
            self.discard(slice(None))
            return supported
        # Normal and friction calls reuse the SDK start/count allocation.
        rows, columns, active, normal_complete = _patch_owners(counts, starts, self._slots)
        magnitude, separation = magnitude.flatten(), separation.flatten()
        finite = (torch.isfinite(magnitude) & torch.isfinite(separation)
                  & torch.isfinite(point).all(-1) & torch.isfinite(normal).all(-1))
        bearing = active & finite & (magnitude != 0)
        bad = torch.zeros_like(supported, dtype=torch.long).scatter_add_(0, rows, (active & ~finite).long())
        # PhysX can report a negative coefficient for the named sensor/filter order.
        coefficient = torch.where(bearing, magnitude, 0.0)
        force = coefficient[:, None] * normal
        props = self._body_properties
        data = self.obj.data
        com = data.body_link_pos_w + quat_apply(data.body_link_quat_w, props.com_position_body_m)
        center = (com * props.mass_kg[..., None]).sum(1) / props.mass_kg.sum(1, keepdim=True)
        moment = torch.linalg.cross(point - center[rows], force)
        relative = _point_velocity(data, rows, self.body_id, point)
        if self.hand is not None:
            relative = relative - _point_velocity(self.hand.data, rows, self._hand_body_ids[columns], point)
        slip = (relative - (relative * normal).sum(-1, keepdim=True) * normal).norm(dim=-1)

        def sum_vector(values, patch_rows, mask):
            return center.new_zeros(center.shape).scatter_add_(
                0, patch_rows[:, None].expand_as(values), torch.where(mask[:, None], values, 0.0),
            )

        def max_scalar(value):
            return center.new_zeros(len(center)).scatter_reduce_(
                0, rows, torch.where(bearing, value, 0.0), reduce="amax",
            )

        load = center.new_zeros(len(center)).scatter_add_(0, rows, coefficient.abs())
        force_on_part = sum_vector(force, rows, bearing)
        moment_on_part = sum_vector(moment, rows, bearing)
        max_separation, max_slip = max_scalar(separation.abs()), max_scalar(slip)
        try:
            friction, at, counts, starts = self.view.get_friction_data(dt=dt)
        except Exception:  # An absent tangential report does not mean zero friction.
            self.discard(slice(None))
            return supported
        friction_rows, _, used, friction_complete = _patch_owners(counts, starts, self._slots)
        finite = torch.isfinite(friction).all(-1) & torch.isfinite(at).all(-1)
        bad.scatter_add_(0, friction_rows, (used & ~finite).long())
        force_on_part += sum_vector(friction, friction_rows, used)
        moment_on_part += sum_vector(torch.linalg.cross(at - center[friction_rows], friction), friction_rows, used)
        supported = normal_complete & friction_complete & (bad == 0)
        present = supported & (load > 0.0)
        actual = {
            "normal_force_n": (load, supported),
            "force_on_second_w_n": (force_on_part, supported[:, None]),
            "force_bearing_distance_max_m": (max_separation, present),
            "force_bearing_tangent_speed_max_m_s": (max_slip, present),
            "moment_on_second_about_root_w_nm": (moment_on_part, supported[:, None]),
        }
        for name, (value, valid) in actual.items():
            self.facts[name].copy_(torch.where(valid, value, torch.nan))
        self.facts["force_bearing_contact_present"].copy_(present)
        if self.digit_groups:
            digit_force = coefficient.abs()[:, None] * self._digit_membership[columns]
            digit_force = digit_force.new_zeros(len(load), len(self.digit_groups)).scatter_add_(
                0, rows[:, None].expand_as(digit_force), digit_force,
            )
            self.facts["digit_group_force_n"].copy_(torch.where(supported[:, None], digit_force, torch.nan))
        if (effort := self.facts.get("joint_effort_on_second")) is not None:
            jacobian = self.obj.body_com_scalar_jacobians()[:, self.body_id]
            moment_at_part = moment_on_part + torch.linalg.cross(center - com[:, self.body_id], force_on_part)
            wrench_at_part = torch.cat((force_on_part, moment_at_part), -1)
            projected = (jacobian.transpose(1, 2) @ wrench_at_part[..., None]).squeeze(-1)
            effort.copy_(torch.where(supported[:, None], projected, torch.nan))
        return supported


class IsaacObjectPairContact:
    """Accumulate all installed link reports of the second object in a named pair."""

    def __init__(self, address, link_reports):
        self.address = address
        self._link_reports = link_reports
        self.facts = {key: torch.full_like(link_reports[0].facts[key], torch.nan)
                      for key in ("normal_force_n", "force_on_second_w_n")}

    def discard(self, rows):
        for value in self.facts.values():
            value[rows] = torch.nan

    def read(self, dt):
        support = torch.stack([report.read(dt) for report in self._link_reports]).all(0)
        for key, output in self.facts.items():
            summed = torch.stack([report.facts[key] for report in self._link_reports]).sum(0)
            mask = support if output.ndim == 1 else support[:, None]
            output.copy_(torch.where(mask, summed, torch.nan))
        return support


class IsaacNativeContactReadback:
    """Carry observed SDK step indices and exact pair samples through the manager."""

    def __init__(self, context, num_envs, device, timestep_s, num_substeps, pairs, *,
                 object_wrenches=None, wrench_parts=()):
        self.context, self.num_envs, self.device = context, num_envs, device
        self._pairs = pairs
        self.contact_facts = {name: pair.facts for name, pair in pairs.items()}
        self.addresses = {name: pair.address for name, pair in pairs.items() if "hand_side" in pair.address}
        self.object_pair_addresses = {name: pair.address for name, pair in pairs.items()
                                      if "first_entity_name" in pair.address}
        self.fixed_support_addresses = {name: pair.address for name, pair in pairs.items()
                                        if "first_fixed_collider_paths" in pair.address}
        self.contact_support = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._object_wrenches = object_wrenches or {}
        self._wrench_parts = wrench_parts
        self._wrench = torch.zeros(num_envs, len(wrench_parts), 6, dtype=torch.float32, device=device)
        self._wrench_supported = torch.zeros_like(self.contact_support)
        self.contact_sample_time_s = torch.full((num_envs,), torch.nan, dtype=torch.float64, device=device)
        self.contact_sample_execution_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._steps = self.contact_sample_execution_step.clone()
        self._installed_dt = float(context.get_physics_dt())
        if not math.isclose(self._installed_dt, timestep_s, rel_tol=1e-6):
            raise ValueError("native Isaac execution requires its configured fixed physics step")
        self._execution_metadata = {
            "schema": "accepted_isaacsim_fixed_step_clock_v1",
            "authority": "Isaac Sim current_time_step_index from physics-step callbacks",
            "configured_timestep_s": float(timestep_s),
            "installed_timestep_s": self._installed_dt,
            "configured_installed_timestep_verified": True,
            "raw_solver_time": "Isaac Sim current_time since play; shared across cloned worlds",
            "time_ns": "round((SDK physics steps - recorded reset SDK physics steps) * configured_timestep_s * 1e9)",
        }
        self.interval = ControlIntervalContacts(self, num_substeps)

    def execution_clock(self):
        self._steps.fill_(self.context.current_time_step_index)
        return {"steps": self._steps, "metadata": self._execution_metadata}

    def physics_time(self):
        return self.contact_sample_time_s.new_full((self.num_envs,), self.context.current_time)

    def _discard_sample(self, rows):
        self.contact_sample_time_s[rows] = torch.nan
        self.contact_sample_execution_step[rows] = -1
        self.contact_support[rows] = False
        self._wrench[rows] = 0
        self._wrench_supported[rows] = False
        for pair in self._pairs.values():
            pair.discard(rows)

    def reset(self, rows=None):
        self.interval.discard(rows)
        self._discard_sample(slice(None) if rows is None else rows)

    def before_substep(self):
        self.interval.before_substep()
        self._begin = self.context.current_time
        self._begin_step = self.context.current_time_step_index
        self._wrench_supported.zero_()
        if self._wrench_parts:
            committed = {name: port.submit() for name, port in self._object_wrenches.items()}
            for column, (entity, _, body_id) in enumerate(self._wrench_parts):
                self._wrench[:, column].copy_(committed[entity][:, body_id])
            self._wrench_supported.fill_(True)

    def after_substep(self):
        dt = self.context.current_time - self._begin
        if self.context.current_time_step_index == self._begin_step + 1 and math.isclose(
            dt, self._installed_dt, rel_tol=1e-6,
        ):
            self.contact_sample_time_s.fill_(self._begin)
            self.contact_sample_execution_step.fill_(self._begin_step)
            self.contact_support.fill_(True)
            for name, pair in self._pairs.items():
                supported = pair.read(dt)
                if name in self.addresses:
                    self.contact_support &= supported
        else:
            self._discard_sample(slice(None))
        self.interval.append(self)

    def facts(self):
        return {
            "pairs": {name: self.contact_facts[name] for name in self.addresses}, "addresses": self.addresses,
            "measurement_support": self.contact_support, "sample_time_s": self.contact_sample_time_s,
            "clock": "Isaac Sim physics-step callback time at the measured substep begin",
            "normal_force_authority": "actual_solver_body_pair_positive_normal_force",
            "distance_authority": "force_bearing_actual_collider_max_absolute_separation",
        }

    def interval_facts(self):
        result = self.interval.facts()
        facts = self.facts()
        result.update({key: facts[key] for key in ("addresses", "normal_force_authority", "distance_authority")})
        result["pairs"] = {name: self.interval.pairs[name] for name in self.addresses}
        if self.addresses and all(self._pairs[name].digit_groups for name in self.addresses):
            result["digit_groups"] = {
                name: {group: dict(body_names=names, normal_force_n=self.interval.pairs[name]["digit_group_force_n"][..., i])
                       for i, (group, names) in enumerate(self._pairs[name].digit_groups.items())}
                for name in self.addresses
            }
        return result

    def object_pair_interval_facts(self):
        return object_pair_interval_facts(self.interval, self.object_pair_addresses)

    def fixed_support_interval_facts(self):
        return object_pair_interval_facts(self.interval, self.fixed_support_addresses)

    def applied_external_body_wrench(self):
        if not self._wrench_parts:
            return None
        return dict(wrench_w=self._wrench, support=self._wrench_supported,
                    authority="accepted_physx_apply_WORLD_body_COM_wrench")

    def applied_wrench_interval_facts(self):
        readback = self.interval.applied_wrench_facts()
        if readback is None:
            return None
        return {
            **readback,
            "entity_names": tuple(entity for entity, _, _ in self._wrench_parts),
            "part_body_names": tuple(body for _, body, _ in self._wrench_parts),
            "peak_force_n": readback["peaks"][..., 0],
            "peak_torque_nm": readback["peaks"][..., 1],
        }
