"""mjlab-Simulation-compatible adapter over Isaac's SimulationContext; mjwarp-
only surfaces no-op or raise so terms take their documented fallbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


def hull_mass_props(obj_dir, scale: float, density: float):
    """(mass, com(3,), I(3,3)) from the COACD hulls, additively composed like
    MuJoCo's per-geom body inertia (mjwarp parity; visual-mesh fallback)."""
    import glob
    import os

    import trimesh

    hulls = sorted(glob.glob(os.path.join(str(obj_dir), "convex", "*.obj")))
    meshes = []
    for h in hulls:
        try:
            m = trimesh.load(h, force="mesh")
            if scale != 1.0:
                m.apply_scale(float(scale))
            if m.volume > 1e-12:
                meshes.append(m)
        except Exception:
            pass
    if not meshes:
        vis = trimesh.load(os.path.join(str(obj_dir), "visual.obj"), force="mesh")
        if scale != 1.0:
            vis.apply_scale(float(scale))
        m = vis if vis.is_watertight else vis.convex_hull
        meshes = [m]
    masses = np.array([m.volume * density for m in meshes])
    total = float(masses.sum())
    coms = np.stack([m.center_mass for m in meshes])
    com = (coms * masses[:, None]).sum(0) / total
    inertia = np.zeros((3, 3))
    for m, mm, c in zip(meshes, masses, coms):
        I_own = np.asarray(m.moment_inertia) * density  # about own com
        r = c - com
        inertia += I_own + mm * ((r @ r) * np.eye(3) - np.outer(r, r))
    mass = float(np.clip(total, 0.05, 0.5))
    inertia *= mass / max(total, 1e-9)
    return mass, com, inertia


class IsaacSimAdapter:
    """PhysX simulator operations for the shared manager loop."""

    supports_dynamic_gravity = True

    def __init__(
        self, sim_ctx: "SimulationContext", device: str, render: bool = False
    ) -> None:
        self._sim = sim_ctx
        self.device = device
        self.render = render
        self._gravity_compensators = []

    def gravity_world_m_per_s2(self):
        """Actual scene-global PhysX acceleration, not a configuration mirror."""
        return tuple(float(value) for value in self._sim.physics_sim_view.get_gravity())

    def set_gravity_world_m_per_s2(self, gravity):
        """Set one global WORLD vector; PhysX exposes no per-environment axis."""
        import carb

        self._sim.physics_sim_view.set_gravity(carb.Float3(*gravity))
        actual = self.gravity_world_m_per_s2()
        for entity in self._gravity_compensators:
            entity.update_gravcomp_gravity(actual)
        return actual

    def add_gravity_compensator(self, entity, scale):
        entity.init_gravcomp(scale, self.gravity_world_m_per_s2())
        self._gravity_compensators.append(entity)

    def begin_control_step(self) -> None:
        # This adapter has no exact hand-part interval force readout.
        pass

    def end_control_step(self) -> None:
        pass

    def step(self) -> None:
        self._sim.step(render=self.render)

    def forward(self) -> None:
        """mjlab sim.forward() parity: PhysX applies no FK on dof writes, so run
        update_articulations_kinematic to refresh link poses after resets."""
        view = getattr(self._sim, "physics_sim_view", None)
        upd = getattr(view, "update_articulations_kinematic", None)
        if upd is not None:
            upd()
        clock = getattr(self, "_clock", None)
        if clock is not None:
            clock.bump()

    def sense(self) -> None:
        pass

    def reset(self, env_ids=None) -> None:
        pass

    def expand_model_fields(self, fields) -> None:
        if fields:
            raise NotImplementedError(
                f"domain-randomization model fields unsupported on isaac: {fields}"
            )
