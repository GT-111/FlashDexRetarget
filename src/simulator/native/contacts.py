"""Stopped named-pair samples for one configured native control interval."""

from __future__ import annotations

import torch


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
                "normal_force_n": torch.full(shape, float("nan"), device=self.device),
                "force_bearing_contact_present": torch.zeros(shape, dtype=torch.bool, device=self.device),
                "force_bearing_distance_max_m": torch.full(shape, float("nan"), device=self.device),
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
        if self.execution_clock_metadata is not None:
            self.execution_sample_begin_step[rows] = -1
            self.execution_sample_end_step[rows] = -1
            self.execution_clock_support[rows] = False
            self.execution_interval_complete[rows] = False
        for pair in self.pairs.values():
            pair["normal_force_n"][rows] = float("nan")
            pair["force_bearing_contact_present"][rows] = False
            pair["force_bearing_distance_max_m"][rows] = float("nan")

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
                values[:, slot].copy_(torch.where(valid, plant.contact_facts[name][key].detach(), missing))
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
