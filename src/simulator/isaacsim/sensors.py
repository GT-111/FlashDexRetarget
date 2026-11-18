"""mjlab-ContactSensor views over per-link Isaac sensors: net force + SDF
gating (PhysX GPU can't filter COACD pairs)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor as IsaacContactSensor

FORCE_EPS = 1e-6


class LinkContactBank:
    """Per-step cache: link net forces + SDF-derived contact geometry."""

    def __init__(
        self,
        sensors: list["IsaacContactSensor"],
        clock,
        device: str,
        *,
        side: str,
        body_names: tuple[str, ...],
        tip_rows: tuple[int, ...],
        margins: torch.Tensor,
        link_pos_fn: Callable,
    ) -> None:
        self._sensors = sensors
        self._clock = clock
        self._token = -1
        self._store: dict[str, torch.Tensor] = {}
        self.device = device
        self.side, self.body_names, self.tip_rows = side, body_names, tip_rows
        self.margins, self.link_pos_fn = margins, link_pos_fn
        self.command = None  # bound by the scene after complete command admission

    def bind_command(self, command) -> None:
        self.command = command

    @property
    def measurement_metadata(self) -> dict:
        sign = None if self.command is None else self.command.object_distance_sign
        return {
            "schema": "isaac_net_link_force_sdf_association_v1",
            "hand_side": self.side,
            "body_names": list(self.body_names),
            "tip_rows": list(self.tip_rows),
            "force_authority": "PhysX_WORLD_link_net_force_from_all_contacted_shapes",
            "association_authority": "active_object_SDF_distance_at_link_origin_with_link_margin",
            "net_force_view_available": bool(self._sensors),
            "ablation_zero_observation_is_measurement": False,
            "pair_normal_force_available": False,
            "distance_sign": sign,
            "signed_distance_query_available": sign == "winding",
            "sampling": "ContactSensor net_forces_w refreshed by scene.update; not interval mean",
            "provenance": "derived execution association; not original observed source contact",
        }

    def _fresh(self) -> dict[str, torch.Tensor]:
        if self._token != self._clock.token:
            self._store.clear()
            self._token = self._clock.token
        return self._store

    def force(self) -> torch.Tensor:
        """(B, L, 3) net link force, or explicit NOREPORT observation zeros."""
        s = self._fresh()
        if "force" not in s:
            # NOREPORT ablation drops the PhysX sensors: keep the obs shape valid.
            s["force"] = (
                torch.stack([x.data.net_forces_w[:, 0] for x in self._sensors], dim=1)
                .to(self.device)
                if self._sensors
                else torch.zeros_like(self.link_pos_fn())
            )
        return s["force"]

    def _link_sdf(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Distance/gradient at configured link origins against the active object."""
        s = self._fresh()
        if "sdf" not in s:
            assert self.command is not None
            cmd = self.command
            p = self.link_pos_fn()
            sdf, grad = cmd.sdf_query(p, self.side)
            s["sdf"], s["grad"], s["lpos"] = sdf, grad, p
        return s["sdf"], s["grad"], s["lpos"]

    def found(self) -> torch.Tensor:
        """(B, L) net-force/SDF association, not an exact hand–object pair test."""
        s = self._fresh()
        if "found" not in s:
            sdf, _, _ = self._link_sdf()
            f = self.force().norm(dim=-1)
            s["found"] = ((f > FORCE_EPS) & (sdf < self.margins)).to(torch.float32)
        return s["found"]

    def contact_pos(self) -> torch.Tensor:
        """(B, L, 3) link origin projected onto the object surface."""
        s = self._fresh()
        if "cpos" not in s:
            sdf, grad, lpos = self._link_sdf()
            s["cpos"] = lpos - sdf.unsqueeze(-1) * grad
        return s["cpos"]

    def normal(self) -> torch.Tensor:
        """(B, L, 3) object-surface gradient direction at the associated point."""
        _, grad, _ = self._link_sdf()
        return grad


class _SensorDataView:
    """Duck-typed mjlab ``ContactSensor.data`` namespace."""

    def __init__(self, owner: "ContactAdapterBase") -> None:
        self._o = owner

    @property
    def force(self) -> torch.Tensor:
        return self._o.force()

    @property
    def torque(self) -> torch.Tensor:
        return self._o.torque()

    @property
    def pos(self) -> torch.Tensor:
        return self._o.pos()

    @property
    def found(self) -> torch.Tensor:
        return self._o.found()

    @property
    def dist(self) -> torch.Tensor:
        return self._o.dist()

    @property
    def normal(self) -> torch.Tensor:
        return self._o.normal()

    @property
    def measurement_metadata(self) -> dict:
        return self._o._bank.measurement_metadata


class ContactAdapterBase:
    def __init__(self, bank: LinkContactBank):
        self._bank = bank
        self.data = _SensorDataView(self)

    def force(self) -> torch.Tensor:
        raise NotImplementedError

    def torque(self) -> torch.Tensor:
        raise NotImplementedError

    def pos(self) -> torch.Tensor:
        raise NotImplementedError

    def found(self) -> torch.Tensor:
        raise NotImplementedError

    def dist(self) -> torch.Tensor:
        raise NotImplementedError

    def normal(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, env_ids=None) -> None:  # mjlab Scene.reset parity
        pass


class AlllinkContactAdapter(ContactAdapterBase):
    """``r_alllink_contact``: found-masked net force + torque approximated
    about the link origin from the SDF-projected contact point."""

    def force(self) -> torch.Tensor:
        return self._bank.force() * self._bank.found().unsqueeze(-1)

    def found(self) -> torch.Tensor:
        return self._bank.found()

    def pos(self) -> torch.Tensor:
        return self._bank.contact_pos()

    def torque(self) -> torch.Tensor:
        f = self.force()
        _, _, lpos = self._bank._link_sdf()
        lever = self._bank.contact_pos() - lpos
        return torch.cross(lever, f, dim=-1)


class AlllinkContactPosAdapter(ContactAdapterBase):
    """``r_alllink_contact_pos``: contact point + found (+ outward normal) per link."""

    # mjlab's contact normal is primary->secondary (into the object); ours is the SDF
    # gradient, so consumers flip it (envs.rewards.chord._chord_step).
    normal_is_outward = True

    def found(self) -> torch.Tensor:
        return self._bank.found()

    def pos(self) -> torch.Tensor:
        return self._bank.contact_pos()

    def normal(self) -> torch.Tensor:
        return self._bank.normal()


class FingertipContactAdapter(ContactAdapterBase):
    """``r_fingertip_contact``: tip-link rows of the alllink bank."""

    def force(self) -> torch.Tensor:
        f = self._bank.force() * self._bank.found().unsqueeze(-1)
        return f[:, list(self._bank.tip_rows)]

    def found(self) -> torch.Tensor:
        return self._bank.found()[:, list(self._bank.tip_rows)]


class FingertipPenetrationAdapter(ContactAdapterBase):
    """``r_fingertip_penetration``: signed SDF distance at the contact sites
    (negative inside), close enough to the mjwarp mindist consumers."""

    def __init__(self, bank: LinkContactBank, side: str):
        super().__init__(bank)
        self._side = side

    def dist(self) -> torch.Tensor:
        cmd = self._bank.command
        if cmd.object_distance_sign != "winding":
            raise NotImplementedError(
                "signed penetration is unavailable for the declared object surface distances"
            )
        pts = cmd.robot_contact_trans_w[:, cmd._side_list.index(self._side)]
        sdf, _ = cmd.sdf_query(pts, self._side)  # (B, configured contact sites)
        return sdf

    def found(self) -> torch.Tensor:
        force_found = self._bank.found()[:, list(self._bank.tip_rows)] > 0
        touch = self.dist() < 0.0
        return (force_found | touch).to(torch.float32)


def build_tracking_contact_views(
    sensors, clock, device, *, side, body_names, contact_site_names, metadata,
    robot, margins,
):
    """One configured named bank for stock or native Isaac scene construction."""
    names = tuple(body_names)
    tip_rows = tuple(
        names.index(metadata.sites[name].body_name) for name in contact_site_names
    )
    ids, _ = robot.find_bodies(list(names), preserve_order=True)
    bank = LinkContactBank(
        sensors, clock, device, side=side, body_names=names, tip_rows=tip_rows,
        margins=margins,
        link_pos_fn=lambda: robot.data.body_link_pos_w[:, list(ids)],
    )
    prefix = side[0]
    views = {
        f"{prefix}_alllink_contact": AlllinkContactAdapter(bank),
        f"{prefix}_alllink_contact_pos": AlllinkContactPosAdapter(bank),
        f"{prefix}_fingertip_contact": FingertipContactAdapter(bank),
        f"{prefix}_fingertip_penetration": FingertipPenetrationAdapter(bank, side=side),
    }
    return bank, views
