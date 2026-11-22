"""Stopped named-pair samples for one configured native control interval."""

from __future__ import annotations

import torch


def digit_body_groups(model, hand_root, tip_sites, finger_roles):
    """Assign only bodies below anatomically unique, physically installed tip paths."""
    if len(tip_sites) != len(finger_roles) or "thumb" not in finger_roles or len(tip_sites) < 2:
        return {}
    root = model.body(hand_root).id
    paths = {"thumb": set(), "fingers": set()}
    for tip, role in zip(tip_sites, finger_roles, strict=True):
        current = int(model.site_bodyid[model.site(tip).id])
        path = paths["thumb" if role == "thumb" else "fingers"]
        while current != root:
            if current == 0:
                raise ValueError("native digit tip is outside its declared physical hand")
            path.add(current)
            current = int(model.body_parentid[current])
    shared = paths["thumb"] & paths["fingers"]
    groups = {name: branch - shared for name, branch in paths.items()}
    if not all(groups.values()):
        return {}
    # Include pad/collider descendants that live on a child body of a digit link.
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        for group in groups.values():
            if parent in group:
                group.add(body)
    return {group: sorted(members) for group, members in groups.items()}


def object_pair_interval_facts(interval, addresses):
    """Read completed named pairs on the shared accepted clock."""
    if interval is None or not addresses:
        return None
    facts = interval.facts()
    if "execution_interval_complete" not in facts:
        return None
    force = torch.stack([facts["pairs"][key]["normal_force_n"] for key in addresses], dim=-1)
    force_on_second = torch.stack([facts["pairs"][key]["force_on_second_w_n"] for key in addresses], dim=-2)
    complete = (facts["interval_complete"] & facts["execution_interval_complete"]
                & facts["measurement_support"].all(-1))
    support = complete[:, None] & torch.isfinite(force).all(dim=1) & torch.isfinite(force_on_second).all(-1).all(1)
    force = torch.where(support[:, None, :], force, torch.nan)
    force_on_second = torch.where(support[:, None, :, None], force_on_second, torch.nan)
    return dict(pairs=list(addresses.values()), num_substeps=facts["num_substeps"],
                normal_force_n=force, force_on_second_w_n=force_on_second,
                peak_normal_force_n=force.amax(dim=1), measurement_support=support)


class ControlIntervalContacts:
    """Fixed storage: worlds × configured substeps × actual named pairs.

    A solver's discrete sample is not a continuous-time force certificate.
    The physics/control loop owns begin/append/finish; readers never advance it.
    """

    def __init__(self, plant, num_substeps: int):
        self.num_substeps = int(num_substeps)
        if self.num_substeps < 1:
            raise ValueError("native contact interval requires positive configured decimation")
        self.num_envs, self.device = plant.num_envs, plant.device
        self.pair_names = tuple(plant.contact_facts)
        shape = (self.num_envs, self.num_substeps)
        self.pairs = {
            name: {
                key: torch.full(shape + value.shape[1:], False if value.dtype == torch.bool else torch.nan,
                                dtype=value.dtype, device=self.device)
                for key, value in plant.contact_facts[name].items()
            }
            for name in self.pair_names
        }
        self.measurement_support = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.sample_recorded = torch.zeros_like(self.measurement_support)
        self.clock_support = torch.zeros_like(self.measurement_support)
        self.sample_begin_time_s = torch.full(shape, float("nan"), dtype=torch.float64, device=self.device)
        self.sample_end_time_s = torch.full_like(self.sample_begin_time_s, float("nan"))
        self.interval_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._discarded = torch.zeros_like(self.interval_complete)
        self._active, self._count = False, 0
        self.substep_wrench_w = None
        wrench_reader = getattr(plant, "applied_external_body_wrench", None)
        readback = None if wrench_reader is None else wrench_reader()
        if readback is not None:
            self.wrench_authority = readback["authority"]
            self.substep_wrench_w = readback["wrench_w"].new_zeros(
                (*shape, *readback["wrench_w"].shape[1:]))
            self.wrench_sample_support = torch.zeros_like(self.sample_recorded)
        clock = plant.execution_clock()
        self.execution_clock_metadata = None if clock is None else clock["metadata"]
        if clock is not None:
            self.execution_sample_begin_step = torch.full(shape, -1, dtype=torch.int64, device=self.device)
            self.execution_sample_end_step = torch.full_like(self.execution_sample_begin_step, -1)
            self.execution_clock_support = torch.zeros_like(self.clock_support)
            self.execution_interval_complete = torch.zeros_like(self.interval_complete)

    def discard(self, rows=None):
        """A reset/state write has no executed contact interval for selected rows."""
        rows = slice(None) if rows is None else rows
        self._discarded[rows] = True
        self.interval_complete[rows] = False
        for field in (self.measurement_support, self.sample_recorded, self.clock_support):
            field[rows] = False
        self.sample_begin_time_s[rows] = float("nan")
        self.sample_end_time_s[rows] = float("nan")
        if self.substep_wrench_w is not None:
            self.substep_wrench_w[rows] = 0.0
            self.wrench_sample_support[rows] = False
        if self.execution_clock_metadata is not None:
            self.execution_sample_begin_step[rows] = -1
            self.execution_sample_end_step[rows] = -1
            self.execution_clock_support[rows] = False
            self.execution_interval_complete[rows] = False
        for pair in self.pairs.values():
            for value in pair.values():
                value[rows] = False if value.dtype == torch.bool else torch.nan

    def begin(self):
        self.discard()
        self._discarded.zero_()
        self._active, self._count = True, 0

    def before_substep(self):
        if not self._active:
            # A public single-substep advance outside a manager interval cannot
            # leave the previous interval attached to its new physical endpoint.
            self.discard()

    @torch.no_grad()
    def append(self, plant):
        if not self._active:
            return
        if self._count == self.num_substeps:
            raise RuntimeError("native interval exceeded configured decimation")
        slot = self._count
        valid = ~self._discarded
        begin = plant.contact_sample_time_s.detach()
        end = plant.physics_time().detach()
        self.sample_recorded[:, slot].copy_(valid)
        self.measurement_support[:, slot].copy_(plant.contact_support.detach() & valid)
        self.sample_begin_time_s[:, slot].copy_(torch.where(valid, begin, float("nan")))
        self.sample_end_time_s[:, slot].copy_(torch.where(valid, end, float("nan")))
        clock = valid & torch.isfinite(begin) & torch.isfinite(end) & (end > begin)
        if slot:
            clock &= begin == self.sample_end_time_s[:, slot - 1]
        self.clock_support[:, slot].copy_(clock)
        if self.substep_wrench_w is not None:
            readback = plant.applied_external_body_wrench()
            if readback is not None:
                wrench = readback["wrench_w"].detach()
                known = valid & readback["support"] & torch.isfinite(wrench).all(dim=(1, 2))
                self.wrench_sample_support[:, slot].copy_(known)
                self.substep_wrench_w[:, slot].copy_(torch.where(known[:, None, None], wrench, 0.0))
        if self.execution_clock_metadata is not None:
            first = plant.contact_sample_execution_step.detach()
            last = plant.execution_clock()["steps"].detach()
            self.execution_sample_begin_step[:, slot].copy_(torch.where(valid, first, -1))
            self.execution_sample_end_step[:, slot].copy_(torch.where(valid, last, -1))
            supported = valid & (first >= 0) & (last == first + 1)
            if slot:
                supported &= first == self.execution_sample_end_step[:, slot - 1]
            self.execution_clock_support[:, slot].copy_(supported)
        for name, target in self.pairs.items():
            for key, values in target.items():
                missing = False if values.dtype == torch.bool else float("nan")
                current = plant.contact_facts[name][key].detach()
                supported = valid[:, None] if current.ndim == 2 else valid
                values[:, slot].copy_(torch.where(supported, current, missing))
        self._count += 1

    def finish(self):
        if not self._active or self._count != self.num_substeps:
            raise RuntimeError("native contact interval did not finish its configured substeps")
        self.interval_complete.copy_(self.sample_recorded.all(-1) & self.clock_support.all(-1))
        if self.execution_clock_metadata is not None:
            self.execution_interval_complete.copy_(
                (self.sample_recorded & self.execution_clock_support).all(-1))
        self._active = False

    def facts(self):
        result = {
            "num_substeps": self.num_substeps, "pairs": self.pairs,
            "sample_recorded": self.sample_recorded, "measurement_support": self.measurement_support,
            "clock_support": self.clock_support, "sample_begin_time_s": self.sample_begin_time_s,
            "sample_end_time_s": self.sample_end_time_s, "interval_complete": self.interval_complete,
            "clock": "actual per-world solver substep begin and achieved endpoint; independent of source time",
            "sampling": "discrete solved samples at every configured physics substep; not continuous force proof",
        }
        if self.execution_clock_metadata is not None:
            result.update(execution_clock=self.execution_clock_metadata,
                execution_sample_begin_step=self.execution_sample_begin_step,
                execution_sample_end_step=self.execution_sample_end_step,
                execution_clock_support=self.execution_clock_support,
                execution_interval_complete=self.execution_interval_complete)
        return result

    def applied_wrench_facts(self):
        if self.substep_wrench_w is None:
            return None
        support = self.wrench_sample_support & self.clock_support
        if self.execution_clock_metadata is not None:
            support &= self.execution_clock_support
        wrench = torch.where(support[..., None, None], self.substep_wrench_w, 0.0)
        return {
            "peaks": torch.linalg.vector_norm(wrench.unflatten(-1, (2, 3)), dim=-1).amax(dim=1),
            "substep_wrench_w": wrench,
            "substep_support": support,
            "measurement_support": self.interval_complete & support.all(-1),
            "num_substeps": self.num_substeps,
            "authority": self.wrench_authority,
        }
