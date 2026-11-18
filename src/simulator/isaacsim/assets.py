"""Configured stock URDF/USD bindings, available before Isaac or Torch boots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _asset_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else _REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class IsaacRobotAsset:
    """One stock import; native FREE-root scenes keep their per-entity bindings."""

    mjcf_path: Path
    source_urdf_path: Path
    usd_path: Path
    body_name_map: dict[str, str]

    def body_name(self, mjcf_name: str) -> str:
        return self.body_name_map.get(mjcf_name, mjcf_name)


def resolve_robot_asset(cfg, *, collision_visual: bool = False) -> IsaacRobotAsset:
    """Resolve the selected config, without guessing a robot from asset names."""
    asset = cfg.get("sim", {}).get("robot_asset")
    if asset is None:
        raise ValueError(
            "stock Isaac robots require an explicit sim.robot_asset binding"
        )
    output = "collision_visual_usd_path" if collision_visual else "usd_path"
    if not asset.get(output):
        raise ValueError(f"sim.robot_asset.{output} is required for the selected asset")
    return IsaacRobotAsset(
        mjcf_path=_asset_path(cfg.robot.xml_path),
        source_urdf_path=_asset_path(asset.source_urdf_path),
        usd_path=_asset_path(asset[output]),
        body_name_map=dict(asset.get("body_name_map", {})),
    )


def load_robot_asset(
    config_path: str | Path, *, collision_visual: bool = False
) -> IsaacRobotAsset:
    """Read a stock robot YAML or resolved run YAML through the same config seam."""
    return resolve_robot_asset(
        OmegaConf.load(_asset_path(str(config_path))), collision_visual=collision_visual
    )
