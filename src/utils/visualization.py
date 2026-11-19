from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh

REPO = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))

SIDES = ("right", "left")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dir with motion.pt or a collection of prepared datasets")
    p.add_argument("--saved", nargs="*", default=[],
                   help="success archives (manifest.json + rollouts/), run dirs holding success/, or their parents")
    p.add_argument("--robot", default="", help="native embodiment to display, or robot for a stock motion.pt")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--geom", default="convex", help="visual, or an objects/<id>/convex* parts dir")
    p.add_argument("--check", action="store_true", help="load every clip and rollout once, then exit")
    return p.parse_args()


def rgb255(c):
    return tuple(int(round(255 * float(x))) for x in c[:3])


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, np.float64).reshape(-1))
    return q


def rot_err_deg(quat_wxyz, rotmat):
    R = np.zeros((len(quat_wxyz), 9))
    for i, q in enumerate(np.asarray(quat_wxyz, np.float64)):
        mujoco.mju_quat2Mat(R[i], q)
    tr = np.einsum("tij,tij->t", R.reshape(-1, 3, 3), rotmat)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def robot_xml(name):
    return REPO / f"assets/robot/{name}/bimanual.xml"


def infer_robot(pt_path):
    ncol = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=False)["joint_pos"].shape[1]
    nq = {d.name: mujoco.MjModel.from_xml_path(str(d / "bimanual.xml")).nq
          for d in sorted((REPO / "assets/robot").iterdir()) if (d / "bimanual.xml").exists()}
    # older packs append the two objects' free joints (2 x 7) after the robot dofs
    for extra in (0, 14):
        hit = [r for r, n in nq.items() if n + extra == ncol]
        if len(hit) == 1:
            return hit[0]
    raise SystemExit(f"{pt_path}: cannot tell the robot from {ncol} joint columns; pass ROBOT=<name>")


def find_robots(root, robot):
    from common.native_reference import load_native_reference

    motion = root / "motion.pt"
    if motion.is_file():
        if motion.with_name("native-reference.json").is_file():
            name = load_native_reference(motion)["target"]["embodiment_id"]
            return {name: motion} if robot in ("", name) else {}
        name = robot or (root.name if robot_xml(root.name).exists() else infer_robot(motion))
        return {name: motion}

    native = {}
    manifests = sorted(p for p in root.rglob("native-reference.json") if p.with_name("motion.pt").is_file())
    for manifest in manifests:
        motion = manifest.with_name("motion.pt")
        name = load_native_reference(motion)["target"]["embodiment_id"]
        if robot not in ("", name):
            continue
        if name in native:
            raise SystemExit(f"{root}: several prepared datasets for {name}; pass --dataset for one motion.pt directory")
        native[name] = motion
    if manifests:
        return native

    pts = sorted(root.glob("*/motion.pt"), key=lambda p: (p.parent.name != "xhand", p.parent.name))
    return {p.parent.name: p for p in pts
            if robot_xml(p.parent.name).exists() and robot in ("", p.parent.name)}


def find_archives(paths):
    out = []
    for p in (Path(x).resolve() for x in paths):
        hit = [c for c in (p, p / "success") if (c / "manifest.json").exists()]
        # Also discover NativeRolloutRecord directories (recordings/current-policy-*/[rollouts/]*.npz)
        if not hit and p.is_dir():
            rec_dirs = sorted({f.parent for f in p.rglob("m*-a*.npz")})
            hit.extend(rec_dirs)
        if not hit and p.is_dir():
            hit = [c for d in sorted(p.iterdir()) if d.is_dir()
                   for c in (d, d / "success") if (c / "manifest.json").exists()]
        if not hit:
            print(f"[viewer] no manifest.json or recordings in {p}", flush=True)
        out += hit
    return out


def vendor_look(spec, vis_dir):
    """Swap the MJCF visuals for the vendor STLs in <robot>/meshes_visual/attachments.json, if present."""
    if not (vis_dir / "attachments.json").exists():
        return False
    look = {"xhand_white": [0.72, 0.72, 0.72, 1.0], "xhand_black": [0.24, 0.24, 0.276, 1.0]}
    bodies = {b.name: b for b in spec.bodies}
    for bname, items in json.load(open(vis_dir / "attachments.json")).items():
        body = bodies.get(bname)
        if body is None:
            continue
        for g in body.geoms:
            if g.name.endswith("_visual"):
                g.group = 4
        for k, it in enumerate(items):
            mesh = f"xv_{bname}_{k}"
            spec.add_mesh(name=mesh, file=str(vis_dir / it["stl"]))
            body.add_geom(name=f"{bname}_xv{k}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
                          pos=it["pos"], quat=it["quat"], rgba=look.get(it.get("mat"), it["rgba"]),
                          contype=0, conaffinity=0, density=0, group=1)
    return True


class Robot:
    def __init__(self, name, pt_path):
        self.name, self.dir = name, Path(pt_path).parent
        self.pt = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=False)
        spec = mujoco.MjSpec.from_file(str(robot_xml(name)))
        self.vendor = vendor_look(spec, robot_xml(name).parent / "meshes_visual")
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        if self.pt["joint_pos"].shape[1] < self.nq:
            raise SystemExit(f"{pt_path}: joint_pos has {self.pt['joint_pos'].shape[1]} dofs, {name} needs {self.nq}")
        self.names = [str(x) for x in self.pt["motion_filename"]]
        self.starts = np.asarray(self.pt["length_starts"]).astype(int)
        self.nframes = np.asarray(self.pt["motion_num_frames"]).astype(int)
        rep = self.dir / "report.csv"
        self.report = {r["clip"]: r for r in csv.DictReader(open(rep))} if rep.exists() else {}
        self.trees = {}

    def add_tree(self, server, key, tint=None, opacity=None):
        m, frames = self.model, {}
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if (m.geom_group[g] >= 3 or m.geom_rgba[g, 3] <= 0 or name.startswith("collision_")
                    or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_bodyid[g] == 0):
                continue
            mat = m.geom_matid[g]
            # MuJoCo uses the material colour unless the geom sets its own (non-default) rgba
            col = m.mat_rgba[mat] if mat >= 0 and np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1]) else m.geom_rgba[g]
            if tint is not None:
                col = 0.35 * np.asarray(col[:3]) + 0.65 * np.asarray(tint)
            b = int(m.geom_bodyid[g])
            if b not in frames:
                frames[b] = server.scene.add_frame(f"/{key}_{self.name}/b{b}", show_axes=False)
            d = m.geom_dataid[g]
            v0, nv, f0, nf = m.mesh_vertadr[d], m.mesh_vertnum[d], m.mesh_faceadr[d], m.mesh_facenum[d]
            server.scene.add_mesh_simple(f"/{key}_{self.name}/b{b}/g{g}", m.mesh_vert[v0:v0 + nv].copy(),
                                         m.mesh_face[f0:f0 + nf].copy(), color=rgb255(col), opacity=opacity,
                                         position=m.geom_pos[g].copy(), wxyz=m.geom_quat[g].copy())
        self.trees[key] = frames

    def pose(self, key, qpos):
        self.data.qpos[:] = qpos[:self.nq]
        mujoco.mj_kinematics(self.model, self.data)
        for b, h in self.trees[key].items():
            h.position = self.data.xpos[b].copy()
            h.wxyz = self.data.xquat[b].copy()

    def show_tree(self, key, on):
        for h in self.trees.get(key, {}).values():
            h.visible = on


class Archive:
    def __init__(self, path, alias, robots):
        man = json.load(open(path / "manifest.json"))
        self.path, self.motion_file = path, man.get("motion_file") or "?"
        self.label = path.parent.name if path.name == "success" else path.name
        self.clips, self.unmatched = {}, []
        for e in man.get("saved", []):
            n = str(e.get("name"))
            m = alias.get(n, alias.get(urllib.parse.unquote(n)))
            if m is None:
                self.unmatched.append(n)
            else:
                self.clips[m] = e
        self.robot = None
        if self.clips:
            ncol = np.load(path / next(iter(self.clips.values()))["rollout"])["joint_pos"].shape[1]
            self.robot = next((r for r in robots.values() if r.nq == ncol), None)
            self.ncol = ncol

    def rollout(self, m):
        e = self.clips.get(m)
        if e is None:
            return None
        z = np.load(self.path / e["rollout"])
        op, oq = z["obj_pos"], z["obj_quat"]
        if op.ndim == 2:
            op, oq = op[:, None], oq[:, None]
        return {"joint_pos": z["joint_pos"], "obj_pos": op, "obj_quat": oq, "entry": e}


def run_native_viewer(args, sets):
    """Display prepared FREE-wrist clips and their declared objects."""
    from common.native_reference import load_native_reference

    class NativeRobot:
        def __init__(self, name, pt_path):
            self.name = name
            self.pt = torch.load(pt_path, map_location="cpu", weights_only=False)
            scene_cfg = load_native_reference(pt_path)["native_scene"]
            self.model = mujoco.MjModel.from_xml_path(scene_cfg["scene_xml_path"])
            self.data = mujoco.MjData(self.model)
            self.hand_free_qadr = {
                h["side"]: int(self.model.joint(h["root_joint_name"]).qposadr[0])
                for h in scene_cfg["hands"]
            }
            self.obj_body_ids = set()
            for obj in scene_cfg["objects"]:
                root_id = self.model.body(obj["root_body_name"]).id
                self.obj_body_ids.add(int(root_id))
                for bid in range(root_id + 1, self.model.nbody):
                    if int(self.model.body_parentid[bid]) in self.obj_body_ids:
                        self.obj_body_ids.add(bid)
            self.names = [
                str(x).rsplit("#", 1)[-1] for x in self.pt["motion_filename"]
            ]
            self.nframes = np.asarray(self.pt["motion_num_frames"]).astype(int)
            self.refs = self.pt["motion_native_reference"]
            self.parts = tuple((obj["entity_name"], part) for obj in scene_cfg["objects"] for part in obj["parts"])
            self.part_addresses = tuple((entity, part["body_name"]) for entity, part in self.parts)
            bodies = tuple(body for _, body in self.part_addresses)
            if any(tuple(clip["object_part_body_names"]) != bodies for clip in self.refs):
                raise ValueError(f"{pt_path}: reference part columns differ from the native scene")
            self.trees = {}

        def add_tree(self, server, key, tint=None, opacity=None):
            m, frames = self.model, {}
            for g in range(m.ngeom):
                gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                b = int(m.geom_bodyid[g])
                if (
                    b == 0
                    or b in self.obj_body_ids
                    or m.geom_rgba[g, 3] <= 0
                    or gname.startswith("collision_")
                    or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                ):
                    continue
                mat = m.geom_matid[g]
                col = (
                    m.mat_rgba[mat]
                    if mat >= 0 and np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1])
                    else m.geom_rgba[g]
                )
                if tint is not None:
                    col = 0.35 * np.asarray(col[:3]) + 0.65 * np.asarray(tint)
                if b not in frames:
                    frames[b] = server.scene.add_frame(f"/{key}_{self.name}/b{b}", show_axes=False)
                d = m.geom_dataid[g]
                v0, nv = m.mesh_vertadr[d], m.mesh_vertnum[d]
                f0, nf = m.mesh_faceadr[d], m.mesh_facenum[d]
                server.scene.add_mesh_simple(
                    f"/{key}_{self.name}/b{b}/g{g}",
                    m.mesh_vert[v0 : v0 + nv].copy(),
                    m.mesh_face[f0 : f0 + nf].copy(),
                    color=rgb255(col),
                    opacity=opacity,
                    position=m.geom_pos[g].copy(),
                    wxyz=m.geom_quat[g].copy(),
                )
            self.trees[key] = frames

        def pose_from_ref(self, key, m_idx, t_idx):
            nr = self.refs[m_idx]
            ti = min(int(t_idx), len(nr["joint_pos"]) - 1)
            self.data.qpos[:] = self.model.qpos0
            for side, qadr in self.hand_free_qadr.items():
                if side in nr["root_state_w"]:
                    self.data.qpos[qadr : qadr + 7] = np.asarray(nr["root_state_w"][side][ti, :7])
            for j_idx, jname in enumerate(nr["joint_names"]):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, str(jname))
                if jid >= 0:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = float(nr["joint_pos"][ti, j_idx])
            mujoco.mj_kinematics(self.model, self.data)
            for b, h in self.trees[key].items():
                h.position = self.data.xpos[b].copy()
                h.wxyz = self.data.xquat[b].copy()

        def pose_from_rollout(self, key, ro, t_idx):
            ti = min(int(t_idx), len(ro["joint_pos"]) - 1)
            self.data.qpos[:] = self.model.qpos0
            for s_idx, side in enumerate(ro["hand_sides"]):
                if side in self.hand_free_qadr:
                    qadr = self.hand_free_qadr[side]
                    self.data.qpos[qadr : qadr + 7] = ro["hand_pose"][ti, s_idx, :7]
            for j_idx, jname in enumerate(ro["joint_names"]):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, str(jname))
                if jid >= 0:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = float(ro["joint_pos"][ti, j_idx])
            mujoco.mj_kinematics(self.model, self.data)
            for b, h in self.trees[key].items():
                h.position = self.data.xpos[b].copy()
                h.wxyz = self.data.xquat[b].copy()

        def show_tree(self, key, on):
            for h in self.trees.get(key, {}).values():
                h.visible = on

    robots = {r: NativeRobot(r, pt) for r, pt in sets.items()}
    ref = next(iter(robots.values()))
    names, nframes = ref.names, ref.nframes
    for r in robots.values():
        if r.names != names or not np.array_equal(r.nframes, nframes) or r.part_addresses != ref.part_addresses:
            raise SystemExit(f"{r.name}: clips or object layout differ from {ref.name}; pass one prepared dataset directory")
    alias = {n: i for i, n in enumerate(names)}
    for i, n in enumerate(names):
        alias.setdefault(urllib.parse.unquote(n), i)

    # Load the declared part-local surface meshes.
    part_meshes = []
    for _, part in ref.parts:
        mesh = trimesh.load(part["mesh_path"], force="mesh", process=False, skip_materials=True)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(part.get("body_from_mesh_quat_wxyz", (1., 0., 0., 0.)), np.float64))
        vertices = mesh.vertices @ rotation.reshape(3, 3).T + part.get("body_from_mesh_position_m", (0., 0., 0.))
        part_meshes.append((np.asarray(vertices, np.float32), np.asarray(mesh.faces, np.int32)))

    # Discover native rollouts (either success/manifest.json or current-policy-*/*.npz)
    archives = {}
    for a in find_archives(args.saved):
        clips = {}
        rob_name = None
        if (a / "manifest.json").exists():
            man = json.loads((a / "manifest.json").read_text())
            npz_list = [(str(e["name"]).rsplit("#", 1)[-1], a / e["rollout"]) for e in man.get("saved", [])]
            label = a.parent.name if a.name == "success" else a.name
        else:
            npz_list = []
            for p in sorted(a.glob("m*-a*.npz")):
                d = np.load(p)
                if "prepared_name" in d:
                    npz_list.append((str(d["prepared_name"]).rsplit("#", 1)[-1], p))
            label = f"{a.parents[1].name}/{a.name}" if len(a.parents) >= 2 else a.name
        for cname, npz_p in npz_list:
            if cname not in alias or not npz_p.exists():
                continue
            d = np.load(npz_p)
            if "native_hand_pose" not in d or "native_binding_json" not in d:
                continue
            nb = json.loads(str(d["native_binding_json"]))
            addresses = tuple((nb["object_parts"][name]["entity_name"], nb["object_parts"][name]["body_name"])
                              for name in nb["object_part_names"])
            if addresses != ref.part_addresses:
                continue
            if rob_name is None:
                emb = nb.get("producer_identity", {}).get("embodiment_id")
                if emb in robots:
                    rob_name = emb
            if rob_name is None:
                jset = tuple(str(x) for x in d["joint_names"])
                rob_name = next((rn for rn, rb in robots.items() if tuple(rb.refs[0]["joint_names"]) == jset), None)
            clips[alias[cname]] = npz_p
        if clips and rob_name in robots:
            key = label if label not in archives else f"{label} ({len(archives)})"
            archives[key] = {"label": key, "robot": robots[rob_name], "clips": clips}
            print(f"[viewer] {key}: {len(clips)} native rollouts ({rob_name})", flush=True)

    import viser

    server = viser.ViserServer(host="0.0.0.0", port=args.port, label=f"FlashDexRetarget Native {Path(args.dataset).name}")
    server.gui.configure_theme(control_width="large", show_share_button=False)
    server.scene.set_up_direction("+z")
    floor = server.scene.add_grid("/floor", width=4.0, height=4.0, cell_size=0.1, section_size=0.5)
    for r in robots.values():
        r.add_tree(server, "robot")
        r.add_tree(server, "ghost", tint=(0.35, 0.6, 1.0), opacity=0.3)

    part_frames = []
    ghost_part_frames = []
    for pidx, (v, f) in enumerate(part_meshes):
        fr = server.scene.add_frame(f"/part_{pidx}", show_axes=False)
        server.scene.add_mesh_simple(f"/part_{pidx}/m", v, f, color=(204, 158, 97))
        part_frames.append(fr)
        gfr = server.scene.add_frame(f"/ghost_part_{pidx}", show_axes=False, visible=False)
        server.scene.add_mesh_simple(f"/ghost_part_{pidx}/m", v, f, color=(130, 175, 235), opacity=0.32)
        ghost_part_frames.append(gfr)

    with server.gui.add_folder("Native clip"):
        g_robot = server.gui.add_dropdown("hand", list(robots) + (["overlay"] if len(robots) > 1 else []), initial_value=next(iter(robots)))
        g_run = server.gui.add_dropdown("saved rollout", ["none"] + list(archives), initial_value=next(iter(archives), "none"))
        g_clip = server.gui.add_dropdown("clip", [f"{i:02d} {n}" for i, n in enumerate(names)], initial_value=f"00 {names[0]}")
        g_frame = server.gui.add_slider("frame", 0, max(1, int(nframes[0]) - 1), 1, 0)
        g_play = server.gui.add_checkbox("play", True)
        g_ghost = server.gui.add_checkbox("reference ghost", True)
        g_info = server.gui.add_markdown("")

    st = {"m": 0, "t": 0.0, "ro": None, "lock": threading.RLock()}

    def load_ro(m):
        a = archives.get(g_run.value)
        if a is None or m not in a["clips"]:
            return None
        d = np.load(a["clips"][m])
        wrench_support = "applied_object_wrench_interval_measurement_support"
        wf = None
        if wrench_support in d and d[wrench_support].any():
            wf = float(d["applied_object_wrench_interval_peak_force_n"][d[wrench_support]].max())

        mean_force = {}
        if "hand_part_interval_metadata_json" in d:
            pairs = json.loads(str(d["hand_part_interval_metadata_json"]))["pairs"]
            samples = d["hand_part_interval_normal_force_n"][d["hand_part_interval_measurement_support"]]
            for side in SIDES:
                columns = [i for i, pair in enumerate(pairs) if pair["hand_side"] == side]
                if columns and len(samples):
                    mean_force[side] = float(samples[:, columns].max(axis=-1).mean())
        return {
            "robot": a["robot"],
            "hand_sides": [str(x) for x in d["native_hand_sides"]],
            "hand_pose": np.asarray(d["native_hand_pose"]),
            "joint_names": [str(x) for x in d["joint_names"]],
            "joint_pos": np.asarray(d["joint_pos"]),
            "part_pose": np.asarray(d["native_object_part_pose"]),
            "mean_contact_force_n": mean_force,
            "max_wrench_n": wf,
        }

    def show_native(t):
        m = st["m"]
        ro = st["ro"]
        nr = ref.refs[m]
        tr = min(int(t), int(nframes[m]) - 1)
        with server.atomic():
            if ro is None:
                shown_r = list(robots) if g_robot.value == "overlay" else [g_robot.value]
                for rn, rb in robots.items():
                    vis = rn in shown_r
                    rb.show_tree("robot", vis)
                    rb.show_tree("ghost", False)
                    if vis:
                        rb.pose_from_ref("robot", m, tr)
                for pidx, fr in enumerate(part_frames):
                    pstate = nr["object_part_state_w"][tr, pidx]
                    fr.position = np.asarray(pstate[:3])
                    fr.wxyz = np.asarray(pstate[3:7])
                    ghost_part_frames[pidx].visible = False
            else:
                rb = ro["robot"]
                t_ro = min(int(t), len(ro["joint_pos"]) - 1)
                for rn, r_other in robots.items():
                    is_sel = rn == rb.name
                    r_other.show_tree("robot", is_sel)
                    r_other.show_tree("ghost", is_sel and g_ghost.value)
                rb.pose_from_rollout("robot", ro, t_ro)
                if g_ghost.value:
                    rb.pose_from_ref("ghost", m, tr)
                for pidx, fr in enumerate(part_frames):
                    pp = ro["part_pose"][t_ro, pidx]
                    fr.position = pp[:3]
                    fr.wxyz = pp[3:7]
                    pstate = nr["object_part_state_w"][tr, pidx]
                    ghost_part_frames[pidx].visible = g_ghost.value
                    ghost_part_frames[pidx].position = np.asarray(pstate[:3])
                    ghost_part_frames[pidx].wxyz = np.asarray(pstate[3:7])

    def load_clip(m):
        with st["lock"]:
            st["m"], st["t"] = int(m), 0.0
            st["ro"] = load_ro(st["m"])
            ro = st["ro"]
            n = int(nframes[m]) if ro is None else max(int(nframes[m]), len(ro["joint_pos"]))
            g_frame.max = max(1, n - 1)
            g_frame.value = 0
            if ro is not None:
                T = min(len(ro["part_pose"]), int(nframes[m]))
                ref_p = ref.refs[m]["object_part_state_w"][:T, :, :3]
                support = ref.refs[m]["object_part_pose_support"][:T]
                errors = np.linalg.norm((ro["part_pose"][:T, :, :3] - ref_p)[support], axis=-1)
                err_cm = f"{errors.mean() * 100.0:.2f} cm" if errors.size else "unmeasured"
                force = ro["mean_contact_force_n"]
                f_l, f_r = (f"{force[side]:.2f} N" if side in force else "unmeasured" for side in ("left", "right"))
                wrench = ro["max_wrench_n"]
                wrench_label = "unmeasured" if wrench is None else f"{wrench:.4f} N"
                g_info.content = (
                    f"**{names[m]}** ({ro['robot'].name})  \n"
                    f"mean part position error: **{err_cm}** · "
                    f"mean peak contact L/R: **{f_l} / {f_r}** · "
                    f"peak applied force: **{wrench_label}**"
                )
            else:
                g_info.content = f"**{names[m]}** · {nframes[m]} frames (kinematic/preloaded-IK reference)"
            show_native(0)

    @g_clip.on_update
    def _(e):
        if e.client is not None:
            load_clip(int(g_clip.value.split()[0]))

    @g_run.on_update
    def _(e):
        if e.client is not None:
            load_clip(st["m"])

    @g_robot.on_update
    def _(e):
        if e.client is not None:
            show_native(st["t"])

    @g_frame.on_update
    def _(e):
        if e.client is not None:
            with st["lock"]:
                st["t"] = float(g_frame.value)
                show_native(st["t"])

    if args.check:
        g_run.value = "none"
        g_robot.value = "overlay" if len(robots) > 1 else g_robot.value
        for m in range(len(names)):
            load_clip(m)
            show_native(nframes[m] - 1)
        print(f"[viewer] native check ok: {len(names)} clips x {list(robots)}", flush=True)
        for label, a in archives.items():
            g_run.value = label
            for m in a["clips"]:
                load_clip(m)
                show_native(len(st["ro"]["joint_pos"]) - 1)
            print(f"[viewer] native rollout check ok: {label}: {len(a['clips'])} rollouts ({a['robot'].name})", flush=True)
        server.stop()
        return

    load_clip(0)
    print(f"[viewer] native ready: {len(names)} clips, hands={list(robots)}, runs={len(archives)} -> http://0.0.0.0:{args.port}", flush=True)
    clock = time.perf_counter()
    while True:
        time.sleep(1 / 30)
        now = time.perf_counter()
        dt, clock = now - clock, now
        if not g_play.value:
            continue
        with st["lock"]:
            n = int(g_frame.max) + 1
            st["t"] = (st["t"] + dt * args.fps) % max(n, 1)
            show_native(st["t"])
            g_frame.value = int(st["t"])


def main():
    args = parse_args()
    root = Path(args.dataset).resolve()
    sets = find_robots(root, args.robot)
    if not sets:
        raise SystemExit(f"no motion.pt (with a robot in assets/robot or native-reference.json) under {root}")
    first_pt = next(iter(sets.values()))
    if (first_pt.parent / "native-reference.json").exists():
        return run_native_viewer(args, sets)

    import viser

    robots = {r: Robot(r, pt) for r, pt in sets.items()}
    ref = next(iter(robots.values()))
    pt, names, starts, nframes = ref.pt, ref.names, ref.starts, ref.nframes
    for r in robots.values():
        if r.names != names or not np.array_equal(r.nframes, nframes):
            raise SystemExit(f"{r.name}: clips differ from {ref.name}; pass one robot dir")
    mesh_dirs = {s: [str(x) for x in pt[f"{s}_object_mesh_dirs"]] for s in SIDES}
    # mesh dirs are relative to the motion.pt dir; a robot subdir may share <root>/objects instead
    pool = next((p for p in (ref.dir, root) if (p / mesh_dirs["right"][0]).exists()), ref.dir)
    scales = {s: [float(x) for x in pt[f"{s}_object_mesh_scales"]] for s in SIDES}
    slots = {s: np.asarray(pt[f"motion_object_slot_{s}"]).astype(int) for s in SIDES}
    disks_all = pt["support_disks"].numpy() if "support_disks" in pt else None
    jn = {s: [str(x) for x in pt[f"mano_{s}_joint_names"]] for s in SIDES} if "mano_right_joint_names" in pt else {}

    info, alias = {}, {}
    for i, n in enumerate(names):
        alias[n] = i
        alias.setdefault(urllib.parse.unquote(n), i)
    for f in (root / "data_info.csv", ref.dir / "data_info.csv", root / "clips.tsv"):
        if f.exists():
            for r in csv.DictReader(open(f), delimiter="\t" if f.suffix == ".tsv" else ","):
                info.setdefault(r["clip"], r["dataset"])
                if r.get("b200_clip") and r["clip"] in alias:  # clips renamed since an older pack
                    alias.setdefault(r["b200_clip"], alias[r["clip"]])
    for n, lineage in zip(names, pt.get("motion_source_lineage", ())):
        if lineage and lineage.get("source_id"):
            info.setdefault(n, lineage["source_id"])
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(info.get(n, "other"), []).append(i)

    archives = {}
    for a in find_archives(args.saved):
        arch = Archive(a, alias, robots)
        if arch.robot is None:
            why = "no clip of this dataset" if not arch.clips else f"{arch.ncol}-dof rollouts, no such robot loaded"
            print(f"[viewer] skip {arch.label}: {why}", flush=True)
            continue
        # a run trained on another pack of the same clips (other world frame / IK) starts away from this reference
        arch.off = sum(np.linalg.norm(np.load(a / e["rollout"])["obj_pos"].reshape(-1, 3)[0]
                                      - pt["obj_right_pos"][starts[m]].numpy()) > 0.05 for m, e in arch.clips.items())
        label = arch.label + (" [other reference]" if arch.off else "")
        label = label if label not in archives else f"{label} ({len(archives)})"
        archives[label] = arch
        print(f"[viewer] {label}: {len(arch.clips)} rollouts ({arch.robot.name})"
              + (f", {len(arch.unmatched)} clips not in this dataset" if arch.unmatched else "")
              + (f", {arch.off} start >5 cm off this dataset (trained on {arch.motion_file})" if arch.off else ""),
              flush=True)

    server = viser.ViserServer(host="0.0.0.0", port=args.port, label=f"FlashDexRetarget {root.name}")
    server.gui.configure_theme(control_width="large", show_share_button=False)
    server.scene.set_up_direction("+z")
    floor = server.scene.add_grid("/floor", width=4.0, height=4.0, cell_size=0.1, section_size=0.5)
    for r in robots.values():
        r.add_tree(server, "robot")
        print(f"[viewer] {r.name}: {'vendor visual meshes' if r.vendor else 'MJCF visuals'}", flush=True)
    for r in {a.robot.name: a.robot for a in archives.values()}.values():
        r.add_tree(server, "ghost", tint=(0.35, 0.6, 1.0), opacity=0.3)

    mesh_cache = {}

    def object_meshes(d, scale, geom):
        key = (d, scale, geom)
        if key not in mesh_cache:
            files = sorted(glob.glob(str(pool / d / geom / "*.obj")),
                           key=lambda f: int(Path(f).stem) if Path(f).stem.isdigit() else 0) if geom != "visual" else []
            mesh_cache[key] = []
            for f in files or [str(pool / d / "visual.obj")]:
                mesh = trimesh.load(f, force="mesh", process=False, skip_materials=True)
                mesh_cache[key].append((np.asarray(mesh.vertices, np.float32) * scale, np.asarray(mesh.faces, np.int32)))
        return mesh_cache[key]

    # every convex* parts dir present (e.g. convex_nomerge next to convex) becomes an option
    variants = sorted({q.name for q in (pool / mesh_dirs["right"][0]).parent.glob("*/convex*") if q.is_dir()},
                      key=lambda x: (x != "convex", x)) + ["visual"]

    with server.gui.add_folder("Clip"):
        g_robot = server.gui.add_dropdown("robot", list(robots) + (["overlay"] if len(robots) > 1 else []),
                                          initial_value=next(iter(robots)))
        g_run = server.gui.add_dropdown("saved run", ["none"] + list(archives), visible=bool(archives),
                                        initial_value=min(archives, key=lambda k: archives[k].off > 0, default="none"))
        g_only = server.gui.add_checkbox("saved clips only", bool(archives), visible=bool(archives))
        g_set = server.gui.add_dropdown("dataset", ["all"] + list(groups), initial_value="all")
        g_clip = server.gui.add_dropdown("clip", ["-"])
        g_prev = server.gui.add_button("prev", icon=viser.Icon.ARROW_LEFT)
        g_next = server.gui.add_button("next", icon=viser.Icon.ARROW_RIGHT)
        g_info = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        g_play = server.gui.add_checkbox("play", True)
        g_speed = server.gui.add_slider("speed", 0.1, 4.0, 0.05, 1.0)
        g_frame = server.gui.add_slider("frame", 0, 1, 1, 0)
        g_center = server.gui.add_button("recenter camera")
    with server.gui.add_folder("Show"):
        g_hand = server.gui.add_checkbox("robot hands", True)
        g_obj = server.gui.add_checkbox("objects", True)
        g_ghost = server.gui.add_checkbox("reference ghost (with a rollout)", True, visible=bool(archives))
        g_geom = server.gui.add_dropdown("object mesh", variants,
                                         initial_value=args.geom if args.geom in variants else variants[0])
        g_mano = server.gui.add_checkbox("MANO joints", bool(jn))
        g_desk = server.gui.add_checkbox("desk disks", True)
        g_floor = server.gui.add_checkbox("floor grid (z=0)", True)

    st = {"m": 0, "t": 0.0, "roll": None, "objs": {}, "ghost_objs": {}, "disks": [], "mano": {},
          "lock": threading.RLock()}

    def arch():
        return archives.get(g_run.value)

    def shown():
        if st["roll"] is not None:
            return [arch().robot.name]
        return list(robots) if g_robot.value == "overlay" else [g_robot.value]

    def clip_order():
        order = list(range(len(names))) if g_set.value == "all" else groups[g_set.value]
        a = arch()
        if a is not None and g_only.value:
            order = [m for m in order if m in a.clips] or order
        return order

    def clip_label(m):
        a = arch()
        return f"{m:02d} {'* ' if a is not None and m in a.clips else ''}{names[m]}"

    def refresh_options():
        g_clip.options = [clip_label(m) for m in clip_order()]

    def recenter(clients=None):
        a, n = starts[st["m"]], nframes[st["m"]]
        ctr = np.mean([pt[f"obj_{s}_pos"][a:a + n].numpy() for s in SIDES], axis=0)
        look = ctr.mean(0)
        dist = 0.6 + 1.2 * float(np.linalg.norm(ctr - look, axis=1).max())
        az, el = np.radians(-125.0), np.radians(-22.0)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        for c in (clients or server.get_clients().values()):
            with c.atomic():
                c.camera.up_direction = (0.0, 0.0, 1.0)
                c.camera.look_at = look
                c.camera.position = look - dist * fwd

    def rollout_err(m, ro):
        a0, T = starts[m], min(len(ro["obj_pos"]), nframes[m])
        errs, start_err = [], 0.0
        for si, s in enumerate(SIDES[:ro["obj_pos"].shape[1]]):
            rp, rr = pt[f"obj_{s}_pos"][a0:a0 + T].numpy(), pt[f"obj_{s}_rotmat"][a0:a0 + T].numpy()
            d = np.linalg.norm(ro["obj_pos"][:T, si] - rp, axis=1)
            start_err = max(start_err, float(d[0]))
            errs.append(f"{s[0]} {d.mean() * 100:.1f} cm / {rot_err_deg(ro['obj_quat'][:T, si], rr).mean():.0f}°")
        return errs, start_err

    def info_md(m, nparts):
        objs = " / ".join(dict.fromkeys(Path(mesh_dirs[s][slots[s][m]]).name
                                        + (f" x{scales[s][slots[s][m]]:g}" if scales[s][slots[s][m]] != 1 else "")
                                        for s in SIDES))
        rows = [f"**{names[m]}**",
                f"{info.get(names[m], 'other')} · {nframes[m]} frames · {nframes[m] / args.fps:.1f} s · {objs}"]
        for r in robots.values():
            rep = r.report.get(names[m], {})
            tip = rep.get("ik_tip_err_mean_m")
            rows.append(f"{r.name} IK: " + (f"tip {float(tip) * 1e3:.1f} mm · " if tip else "")
                        + (rep.get("flags") or "ok") + (f" · reject {rep['reject']}" if rep.get("reject") else ""))
        ro = st["roll"]
        if ro is not None:
            e, (errs, start_err) = ro["entry"], rollout_err(m, ro)
            step = e.get("updated_env_step", e.get("first_saved_env_step"))
            rows.append(f"rollout ({arch().robot.name}): {len(ro['joint_pos'])} frames"
                        + (f" · saved at {step / 1e6:.1f}M steps" if step is not None else "")
                        + " · obj err " + ", ".join(errs))
            if start_err > 0.05:
                rows.append(f"**frame 0 is {start_err * 100:.0f} cm off the reference: this run trained on "
                            f"{arch().motion_file}, not this dataset**")
        elif arch() is not None:
            rows.append("no rollout of this clip in the selected run")
        rows.append(f"{g_geom.value}: " + " / ".join(f"{o} {n} parts" for o, n in nparts.items()))
        return "  \n".join(rows)

    def add_object(path, parts, rgb, opacity, visible):
        fr = server.scene.add_frame(path, show_axes=False, visible=visible)
        for j, (v, f) in enumerate(parts):
            shade = 1.0 if len(parts) == 1 else 0.72 + 0.28 * ((j * 5) % len(parts)) / max(len(parts) - 1, 1)
            server.scene.add_mesh_simple(f"{path}/p{j}", v, f, color=rgb255(np.clip(rgb * shade, 0, 1)),
                                         opacity=opacity)
        return fr

    def refresh_visibility():
        ro = st["roll"]
        for r in robots.values():
            r.show_tree("robot", g_hand.value and r.name in shown())
            r.show_tree("ghost", ro is not None and g_ghost.value and g_hand.value and r.name in shown())
        for h in st["objs"].values():
            h.visible = g_obj.value
        for h in st["ghost_objs"].values():
            h.visible = ro is not None and g_ghost.value and g_obj.value

    def load(m):
        with st["lock"]:
            st["m"], st["t"] = m, 0.0
            a = arch()
            st["roll"] = a.rollout(m) if a is not None else None
            for h in (list(st["objs"].values()) + list(st["ghost_objs"].values())
                      + st["disks"] + list(st["mano"].values())):
                h.remove()
            st["objs"], st["ghost_objs"], st["disks"], st["mano"] = {}, {}, [], {}
            a0 = starts[m]
            # a shared object (both hands on the same body) is drawn once
            same = (mesh_dirs["right"][slots["right"][m]] == mesh_dirs["left"][slots["left"][m]]
                    and np.allclose(pt["obj_right_pos"][a0].numpy(), pt["obj_left_pos"][a0].numpy(), atol=1e-5))
            obj_rgb = {"right": np.array([0.80, 0.62, 0.38]), "left": np.array([0.62, 0.78, 0.52])}
            nparts = {}
            for s in (("right",) if same else SIDES):
                k = slots[s][m]
                parts = object_meshes(mesh_dirs[s][k], scales[s][k], g_geom.value)
                nparts[Path(mesh_dirs[s][k]).name] = len(parts)
                st["objs"][s] = add_object(f"/obj_{s}", parts, obj_rgb[s], None, g_obj.value)
                if st["roll"] is not None:
                    st["ghost_objs"][s] = add_object(f"/ghost_obj_{s}", parts, obj_rgb[s], 0.35, g_obj.value)
            if disks_all is not None:
                for k, (x, y, z_top, r, h) in enumerate(disks_all[m]):
                    if r > 0:
                        cyl = trimesh.creation.cylinder(radius=float(r), height=float(max(h, 1e-3)), sections=48)
                        st["disks"].append(server.scene.add_mesh_simple(
                            f"/desk/d{k}", np.asarray(cyl.vertices, np.float32), np.asarray(cyl.faces, np.int32),
                            color=(158, 153, 143), position=(float(x), float(y), float(z_top - h / 2)),
                            visible=g_desk.value))
            n = int(nframes[m]) if st["roll"] is None else max(int(nframes[m]), len(st["roll"]["joint_pos"]))
            g_frame.max = n - 1
            g_frame.value = 0
            if g_clip.value != clip_label(m):
                if clip_label(m) not in g_clip.options:
                    refresh_options()
                g_clip.value = clip_label(m)
            g_info.content = info_md(m, nparts)
            refresh_visibility()
            show(0)
        recenter()

    def show(t):
        m, ro = st["m"], st["roll"]
        f = int(starts[m]) + min(int(t), int(nframes[m]) - 1)
        with server.atomic():
            if ro is None:
                for r in robots.values():
                    if r.name in shown():
                        r.pose("robot", r.pt["joint_pos"][f].numpy())
                for s, h in st["objs"].items():
                    h.position = pt[f"obj_{s}_pos"][f].numpy()
                    h.wxyz = mat2quat(pt[f"obj_{s}_rotmat"][f].numpy())
            else:
                r, tr = arch().robot, min(int(t), len(ro["joint_pos"]) - 1)
                r.pose("robot", ro["joint_pos"][tr])
                r.pose("ghost", r.pt["joint_pos"][f].numpy())
                for si, s in enumerate(SIDES):
                    if s in st["objs"]:
                        j = min(si, ro["obj_pos"].shape[1] - 1)
                        st["objs"][s].position = ro["obj_pos"][tr, j]
                        st["objs"][s].wxyz = ro["obj_quat"][tr, j]
                        st["ghost_objs"][s].position = pt[f"obj_{s}_pos"][f].numpy()
                        st["ghost_objs"][s].wxyz = mat2quat(pt[f"obj_{s}_rotmat"][f].numpy())
            if jn and g_mano.value:
                for s in SIDES:
                    j = dict(zip(jn[s], pt[f"mano_{s}_joints"][f].numpy()))
                    w = pt[f"mano_{s}_wrist_pos"][f].numpy()
                    seg = []
                    for fi in ("thumb", "index", "middle", "ring", "pinky"):
                        chain = [w] + [j[f"{fi}_{c}"] for c in ("proximal", "intermediate", "distal", "tip")
                                       if f"{fi}_{c}" in j]
                        seg += list(zip(chain[:-1], chain[1:]))
                    seg = np.asarray(seg, np.float32)
                    pts = np.concatenate([[w], np.stack(list(j.values()))]).astype(np.float32)
                    rgb = {"right": (40, 110, 230), "left": (20, 170, 200)}[s]
                    if s in st["mano"]:
                        st["mano"][s].points = seg
                        st["mano"][f"{s}_p"].points = pts
                    else:
                        st["mano"][s] = server.scene.add_line_segments(f"/mano_{s}", seg, rgb, line_width=3)
                        st["mano"][f"{s}_p"] = server.scene.add_point_cloud(
                            f"/mano_{s}_pts", pts, rgb, point_size=0.008, point_shape="circle")

    def step(d):
        order = clip_order()
        pos = order.index(st["m"]) if st["m"] in order else -1
        load(order[(pos + d) % len(order)])

    def select_run():
        st["roll"] = None
        a = arch()
        if a is not None and g_robot.value != a.robot.name:
            g_robot.value = a.robot.name
        g_robot.disabled = a is not None
        refresh_options()
        order = clip_order()
        load(st["m"] if st["m"] in order else order[0])

    # server-side value changes also fire on_update (client None): only user edits act
    @g_clip.on_update
    def _(e):
        if e.client is not None:
            load(int(g_clip.value.split()[0]))

    @g_run.on_update
    def _(e):
        if e.client is not None:
            select_run()

    @g_only.on_update
    def _(_):
        refresh_options()
        if g_only.value and clip_order() and st["m"] not in clip_order():
            load(clip_order()[0])
        else:
            g_clip.value = clip_label(st["m"])

    @g_set.on_update
    def _(_):
        refresh_options()
        order = clip_order()
        if st["m"] in order:
            g_clip.value = clip_label(st["m"])
        else:
            load(order[0])

    g_prev.on_click(lambda _: step(-1))
    g_next.on_click(lambda _: step(1))
    g_center.on_click(lambda _: recenter())

    @g_frame.on_update
    def _(e):
        if e.client is not None:
            with st["lock"]:
                st["t"] = float(g_frame.value)
                show(st["t"])

    @g_robot.on_update
    def _(_):
        refresh_visibility()
        show(st["t"])

    for g in (g_hand, g_obj, g_ghost):
        g.on_update(lambda _: refresh_visibility())

    @g_geom.on_update
    def _(_):
        load(st["m"])

    @g_mano.on_update
    def _(_):
        for h in st["mano"].values():
            h.visible = g_mano.value
        show(st["t"])

    @g_desk.on_update
    def _(_):
        for h in st["disks"]:
            h.visible = g_desk.value

    @g_floor.on_update
    def _(_):
        floor.visible = g_floor.value

    server.on_client_connect(lambda c: recenter([c]))
    if args.check:
        g_run.value = "none"
        g_robot.value = "overlay" if len(robots) > 1 else g_robot.value
        select_run()
        for m in range(len(names)):
            load(m)
            show(nframes[m] - 1)
        print(f"[viewer] check ok: {len(names)} clips x {list(robots)}", flush=True)
        for label, a in archives.items():
            g_run.value = label
            select_run()
            for m in a.clips:
                load(m)
                show(len(st["roll"]["joint_pos"]) - 1)
            print(f"[viewer] check ok: {label}: {len(a.clips)} rollouts", flush=True)
        server.stop()
        return
    select_run()
    print(f"[viewer] {len(names)} clips, robots {list(robots)}, {len(archives)} saved run(s): "
          f"http://0.0.0.0:{args.port}", flush=True)

    clock = time.perf_counter()
    while True:
        time.sleep(1 / 30)
        now = time.perf_counter()
        dt, clock = now - clock, now
        if not g_play.value:
            continue
        with st["lock"]:
            n = int(g_frame.max) + 1
            st["t"] = (st["t"] + dt * args.fps * g_speed.value) % max(n, 1)
            show(st["t"])
            g_frame.value = int(st["t"])


if __name__ == "__main__":
    main()
