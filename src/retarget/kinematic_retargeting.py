from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from retarget.build_motion_pt import build, object_meshes
from retarget.human_demo import clips as C
from retarget.object.obj_convex_decompose import decompose_objects
from retarget.object.obj_sdf_bake import bake


def _log(msg: str) -> None:
    print(f"[kinematic_retargeting] {msg}", flush=True)


def _write_report(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["clip"]))


class MotionInput:
    """A selected input owns preparation/IK; all inputs share the normal packer."""

    def __init__(self, dataset: Path, cfg: DictConfig, workers: int):
        self.dataset, self.cfg, self.workers = dataset, cfg, workers
        self.clip_directory: Path | None = None
        self.clip_names: set[str] = set()
        self.object_ids: list[str] = []
        self.fps = 0.0

    def retarget(self) -> None:
        """Already prepared inputs do not run human-demo IK."""

    def pack(self) -> None:
        if self.clip_directory is not None:
            _log(f"pack: {self.clip_directory} -> {self.dataset / 'motion.pt'}")
            build(self.clip_directory, self.dataset, self.fps, self.cfg.pack, names=self.clip_names)


class HumanDemoInput(MotionInput):
    """The author's preprocessing/IK route, including existing motion-only reuse."""

    def __init__(self, dataset: Path, cfg: DictConfig, workers: int):
        super().__init__(dataset, cfg, workers)
        demo, motion = dataset / "human_demo", dataset / "motion.pt"
        self.report = dataset / "report.csv"
        self.from_clips = demo.is_dir()
        if self.from_clips:
            from retarget.human_demo.preprocess import preprocess_clips

            self.fps = C.clip_fps(C.list_clips(demo))
            _log(f"preprocess: {demo} ({self.fps:g} Hz)")
            self.rows, self.kept = preprocess_clips(demo, dataset, cfg, self.fps, workers)
            _write_report(self.report, self.rows)
            if not self.kept:
                raise RuntimeError(f"every clip was rejected; see {self.report}")
            self.object_ids = C.object_ids(self.kept)
            self.clip_directory = dataset / "retargeted"
            self.clip_names = {p.stem for p in self.kept}
        elif motion.exists():
            _log(f"no {demo}: preparing the objects of {motion}")
            packed = torch.load(motion, map_location="cpu", weights_only=False)
            self.object_ids = sorted({mesh.parent.name for mesh, _ in object_meshes(packed, dataset)})
        else:
            raise FileNotFoundError(f"{dataset} has neither human_demo/ nor motion.pt")

    def retarget(self) -> None:
        if not self.from_clips:
            return
        from retarget.robot_hand.hand_retarget import retarget_clips

        _log(f"retarget: {len(self.kept)} clips -> {self.clip_directory} ({self.cfg.robot})")
        ik = {r["clip"]: r for r in retarget_clips(list(self.kept.items()), self.dataset,
            self.clip_directory, self.cfg.robot, self.cfg.retarget, self.fps, self.workers)}
        for row in self.rows:
            row.update(ik.get(row["clip"], {}))
        _write_report(self.report, self.rows)
        if all(row.get("error") for row in ik.values()):
            raise RuntimeError(f"every clip failed to retarget; see {self.report}")


class PreparedNativeInput(MotionInput):
    """Pack explicitly selected native carriers without changing their source facts."""

    def __init__(self, dataset: Path, cfg: DictConfig, workers: int, manifest: str):
        super().__init__(dataset, cfg, workers)
        declaration = json.loads((dataset / manifest).read_text())
        self.clip_directory = dataset / declaration["clip_directory"]
        self.clip_names = set(declaration["clips"])
        if not self.clip_names:
            raise ValueError("prepared native manifest selects no clips")
        paths = [self.clip_directory / f"{name}.npz" for name in sorted(self.clip_names)]
        self.fps = C.clip_fps(paths)
        # Clip fps is commonly float32; the manifest records its exact control cadence.
        if np.float32(self.fps) != np.float32(declaration["fps"]):
            raise ValueError("prepared native manifest and selected clips have different fps")
        if declaration["fps"] != 1e9 / declaration["control_period_ns"]:
            raise ValueError("prepared native manifest fps and control period disagree")
        for path in paths:
            clip = C.load_clip(path)
            if C.native_reference(clip) is None:
                raise ValueError(f"{path.name}: prepared native input requires native_reference")
            if "support_disks" in clip:
                raise ValueError(f"{path.name}: prepared native support belongs to its authored scene")
        self.object_ids = C.object_ids(paths)
        _log(f"prepared native: {len(paths)} clips from {self.clip_directory} ({self.fps:g} Hz)")


@hydra.main(config_path="../../config/retarget", config_name="kinematic_retargeting", version_base=None)
def main(cfg: DictConfig) -> None:
    workers = cfg.workers or max(1, (os.cpu_count() or 2) // 2)
    steps = cfg.steps
    dataset = Path(cfg.dataset)
    motion = dataset / "motion.pt"
    source = instantiate(cfg.input, dataset=dataset, cfg=cfg, workers=workers, _recursive_=False)
    ids = source.object_ids

    if steps.convex:
        _log(f"convex: {len(ids)} objects (<object>/convex/, {cfg.convex.name})")
        decompose_objects(dataset, ids, cfg.convex, workers)
    if steps.retarget:
        source.retarget()
    if steps.pack:
        source.pack()
    if steps.usd:
        _log("usd: <object>/.isaac_usd/object.usd")
        # CUDA and graphics ordinals can differ; the operator supplies Kit's mapping.
        env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "disabled"),
                   CUDA_VISIBLE_DEVICES=str(cfg.usd_gpu))
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "simulator/isaacsim/convert_assets.py"),
                        "--data-root", str(dataset.resolve()), "--objects", *ids,
                        "--kit_args=" + str(cfg.usd_kit_args)], env=env, check=True)
    if steps.sdf and not motion.exists():
        _log(f"sdf: skipped, no {motion} yet")
    elif steps.sdf:
        _log(f"sdf: <object>/.sdf_cache/, {cfg.sdf.grid_n}^3")
        bake(motion, cfg.sdf, workers)
    _log(f"done: train with MOTION={motion}" if motion.exists() else "done")


if __name__ == "__main__":
    main()
