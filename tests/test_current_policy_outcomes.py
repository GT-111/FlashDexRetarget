"""Current-policy evaluation evidence using CPU protocol fixtures, without physics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Expose the supported vendored manager subset without starting Kit.
import simulator.isaacsim  # noqa: F401

# isort: split
from omegaconf import OmegaConf
from test_native_task_contract import _command

from evaluation.callbacks.tracking_performance import NativeReferenceTrackingPerformance
from evaluation.evaluate import evaluate, write_current_policy_outcomes
from evaluation.shared import build_eval_plan
from runner.off_policy_runner import OffPolicyRunner


class _Scene(dict):
    def tracking_contact_metadata(self):
        return {"force": "net_link_force", "association": "SDF", "pair_force_supported": False}


class _EvaluationEnv:
    """Small evaluator protocol; no scene constructor or physical-success claim."""

    def __init__(self, *, nonfinite_tip=False):
        self.device, self.num_envs, self.unwrapped = "cpu", 4, self
        self.exit_count = 0
        self.nonfinite_tip = nonfinite_tip
        library = SimpleNamespace(
            num_trajectories=4,
            _motion_num_frames=torch.tensor([4, 3, 5, 2]),
            motion_file="fixture-motion.pt",
            motion_files=[f"fixture-motion.pt#clip-{i}" for i in range(4)],
        )
        self.command = SimpleNamespace(
            motion_lib=library,
            motion_ids=torch.zeros(self.num_envs, dtype=torch.int64),
            motion_steps=torch.zeros(self.num_envs, dtype=torch.int64),
            _side_list=("left", "right"),
            finger_names=("thumb", "index"),
            ref_contact_flags=torch.ones(self.num_envs, 2, 2),
            ref_contact_trans_w=torch.zeros(self.num_envs, 2, 2, 3),
            robot_tip_trans_w=torch.zeros(self.num_envs, 2, 2, 3),
        )
        self.command.ref_contact_flags[1, 0, 1] = 0
        self.command.robot_tip_trans_w[..., 0] = 0.01
        self.command_manager = SimpleNamespace(get_term=lambda _name: self.command)
        self.action_manager = SimpleNamespace(action=torch.zeros(4, 2), prev_action=torch.zeros(4, 2))
        self.scene = _Scene()
        for side in ("l", "r"):
            self.scene[f"{side}_fingertip_penetration"] = SimpleNamespace(
                data=SimpleNamespace(found=torch.tensor([[1, 0]] * 4), dist=torch.tensor([[-0.002, 0.004]] * 4))
            )
            self.scene[f"{side}_fingertip_contact"] = SimpleNamespace(
                data=SimpleNamespace(force=torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]] * 4))
            )
        # Real object-slot planning reorders motion IDs, leaves motion2 absent,
        # and produces a ragged final batch. Each reachable motion has two replicas.
        self._object_eval_plan = build_eval_plan(torch.tensor([1, 0, 1, 0]), torch.tensor([0, 1, 2, 0]), 2)
        self.eval_k, self.eval_num_motions, self.eval_num_batches = 0, 4, len(self._object_eval_plan)

    def reset(self):
        cmd = self.command
        cmd.motion_steps.fill_(1)  # The normal reset consumes the initial reference.
        cmd.motion_num_frames = cmd.motion_lib._motion_num_frames[cmd.motion_ids]
        cmd.ref_obj_trans_w = torch.zeros(4, 2, 3)
        cmd.sim_obj_trans_w = torch.zeros_like(cmd.ref_obj_trans_w)
        self.action_manager.action.zero_()
        self.action_manager.prev_action.zero_()
        errors = torch.tensor([0.01, 0.02, 0.01, 0.01 if self.nonfinite_tip else 0.2])
        cmd.metrics = {}
        for side in ("l", "r"):
            cmd.metrics[f"error_obj_trans_{side}"] = errors
            cmd.metrics[f"error_obj_rot_{side}"] = torch.zeros(4)
            cmd.metrics[f"error_level1_{side}"] = torch.full((4,), 0.002)
            cmd.metrics[f"error_level2_{side}"] = torch.full((4,), 0.002)
            for finger in cmd.finger_names:
                tips = torch.full((4,), 0.001)
                if self.nonfinite_tip:
                    tips[3] = float("nan")
                cmd.metrics[f"error_tip_trans_{side}_{finger}"] = tips
        return self._obs(), {}

    def _obs(self):
        return {"actor": torch.zeros(4, 2), "critic": torch.zeros(4, 4), "window": torch.ones(4, 1)}

    def step(self, actions):
        assert actions.shape == (4, 2)
        cmd = self.command
        cmd.sim_obj_trans_w = cmd.ref_obj_trans_w.clone()
        cmd.sim_obj_trans_w[..., 0] += cmd.metrics["error_obj_trans_l"][:, None]
        cmd.ref_obj_trans_w = cmd.ref_obj_trans_w + 0.01
        cmd.motion_steps += 1
        self.action_manager.prev_action = self.action_manager.action.clone()
        self.action_manager.action += 1
        return self._obs(), torch.zeros(4), torch.zeros(4), {}

    def exit_eval(self):
        self.exit_count += 1
        # The normal evaluator restores TRAIN before reducing collected states.
        self.command.motion_ids.fill_(99)


class _Agent:
    def __init__(self):
        self.calls = 0

    def act(self, obs, deterministic):
        assert deterministic and obs.shape == (4, 3)
        self.calls += 1
        return np.zeros((4, 2), dtype=np.float32)


def _config(grace_steps=1):
    return OmegaConf.create({
        "command_name": "motion", "num_per_motion": 2, "save_success_data": False,
        "success_criterion": "obj", "callbacks": [
            {"_target_": "evaluation.shared.ObjectSlotMotionTrackingEvalSetup", "command_name": "motion", "num_per_motion": 2},
            {"_target_": "evaluation.callbacks.tracking_performance.TrackingPerformance", "command_name": "motion",
             "grace_steps": grace_steps, "threshold_ks": [1.0],
             "thresholds": {"obj_trans": 0.03, "obj_rot_deg": 30.0, "tip_trans": 0.06, "joint_trans": 0.08}},
            {"_target_": "evaluation.callbacks.contact_performance.ContactPerformance", "command_name": "motion", "grace_steps": grace_steps},
        ],
    })


def _runner(tmp_path, env, cfg):
    return SimpleNamespace(
        _agent=_Agent(), _eval_env=env, _eval_cfg=cfg, _device="cpu", _eval_ever_state=None,
        _success_dir=tmp_path / "success", _save_path_base=tmp_path,
    )


def _callback(report, suffix):
    return next(key for key in report["callback_metadata"] if key.endswith(suffix))


def test_normal_runner_eval_persists_all_current_attempts_and_real_support(tmp_path):
    env = _EvaluationEnv()
    runner = _runner(tmp_path, env, _config())
    runner._success_dir.mkdir()
    old_archive = '{"saved": [{"motion_id": 0, "env_step": 1}]}\n'
    (runner._success_dir / "manifest.json").write_text(old_archive)
    metrics = OffPolicyRunner._run_eval(runner, 4096)
    reports = list((tmp_path / "evaluation").glob("current-policy-000000004096-*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert runner._agent.calls == 6 and env.exit_count == 1
    assert report["policy_authority"]["training_env_step"] == 4096
    assert report["policy_authority"]["kind"] == "current_in_memory_agent"
    assert "no checkpoint is inferred" in report["policy_authority"]["checkpoint_mapping"]
    assert report["selected_motion_count"] == 4 and report["configured_replicas_per_motion"] == 2
    assert [motion["motion_name"] for motion in report["motions"]] == [f"clip-{i}" for i in range(4)]
    tracking, contact = _callback(report, ".TrackingPerformance"), _callback(report, ".ContactPerformance")
    motion0, motion1, absent, short = [motion["callbacks"][tracking] for motion in report["motions"]]
    assert [row["evaluation_row"] for row in motion0["attempts"]] == [1, 3]
    assert motion0["full_threshold_counts"] == [{"pass": 1, "fail": 1, "unscored": 0, "nonfinite": 0}]
    assert motion1["scored_attempt_count"] == 2
    assert absent == {"attempts": [], "attempt_count": 0, "scored_attempt_count": 0,
                      "unscored_attempt_count": 0, "coverage": "not_evaluated"}
    assert short["attempt_count"] == short["unscored_attempt_count"] == 2
    assert short["coverage"] == "unscored"
    assert all(row["full_status"] == ["unscored"] and row["observed_steps"] == 0 for row in short["attempts"])
    row = motion0["attempts"][1]
    assert row["motion_start_step"] == 1 and row["remaining_source_steps"] == 3
    assert row["observed_steps"] == 2
    assert row["raw_metric_sums"]["error_obj_trans"] == pytest.approx([0.4, 0.4])
    assert row["per_side_mean_errors"]["obj_trans"] == pytest.approx([0.2, 0.2])
    assert metrics["eval/success_rate_1.0"] == 0.75
    assert metrics["eval/success_rate_obj_1.0"] == 0.75
    assert report["ordinary_aggregate_metrics"]["num_scored_envs"] == 4
    sensor = report["motions"][0]["callbacks"][contact]["attempts"][0]
    assert sensor["observed_steps"] == 2
    assert sensor["reference_contact_count"] == [[2.0, 0.0], [2.0, 2.0]]
    assert sensor["associated_contact_count"] == [[2.0, 0.0], [2.0, 0.0]]
    assert sensor["raw_sum_sensor_force_N"] == [[10.0, 0.0], [10.0, 0.0]]
    assert sensor["max_miss_streak_steps"] == [[0.0, 0.0], [0.0, 2.0]]
    assert not report["callback_metadata"][contact]["sensor_measurement_authority"]["pair_force_supported"]
    assert "UNQUALIFIED" in report["physical_task_or_BOX_success"]
    assert (runner._success_dir / "manifest.json").read_text() == old_archive
    # Periodic and final evaluation can share an env_step; keep both actual sweeps.
    OffPolicyRunner._run_eval(runner, 4096)
    assert len(list((tmp_path / "evaluation").glob("current-policy-000000004096-*.json"))) == 2


def test_nonfinite_tracking_is_explicit_without_changing_object_or_legacy_scores(tmp_path):
    runner = _runner(tmp_path, _EvaluationEnv(nonfinite_tip=True), _config())
    metrics = OffPolicyRunner._run_eval(runner, 4096)
    report = json.loads(next((tmp_path / "evaluation").glob("*.json")).read_text())
    key = _callback(report, ".TrackingPerformance")
    row = report["motions"][0]["callbacks"][key]["attempts"][1]
    assert row["raw_tip_trans_sums"] == [["nan", "nan"], ["nan", "nan"]]
    assert row["full_failed"] == [False]  # Preserve the existing threshold arithmetic.
    assert row["full_status"] == ["nonfinite"] and row["object_status"] == ["pass"]
    assert not row["full_finite_metrics"] and row["object_finite_metrics"]
    assert metrics["eval/success_rate_1.0"] == metrics["eval/success_rate_obj_1.0"] == 1.0


def test_zero_observed_windows_remain_unscored_despite_legacy_aggregate_fallback(tmp_path):
    runner = _runner(tmp_path, _EvaluationEnv(), _config(grace_steps=99))
    metrics = OffPolicyRunner._run_eval(runner, 4096)
    report = json.loads(next((tmp_path / "evaluation").glob("*.json")).read_text())
    key = _callback(report, ".TrackingPerformance")
    rows = [row for motion in report["motions"] for row in motion["callbacks"][key]["attempts"]]
    assert len(rows) == 6
    assert all(row["observed_steps"] == 0 and row["full_status"] == ["unscored"] for row in rows)
    assert metrics["eval/num_scored_envs"] == 6  # Original aggregate fallback is intact.
    assert metrics["eval/success_rate_1.0"] == 1.0


@pytest.mark.parametrize("rigid", [True, False])
def test_native_current_rows_preserve_missing_support_empty_hinges_and_failure(tmp_path, rigid):
    command, env, _ = _command(rigid=rigid)
    thresholds = {"root_position_error_m": 0.03, "root_rotation_error_rad": 0.1,
                  "part_position_error_m": 0.03, "part_rotation_error_rad": 0.1,
                  "joint_position_error_rad": 0.1, "command_position_error_rad": 0.1}
    callback = NativeReferenceTrackingPerformance("motion", thresholds)
    command.motion_steps.fill_(1)
    states = []
    for kind in ("pass", "fail", "missing", "empty"):
        command.robot.data.joint_pos[0, 0] = 0.5 if kind == "fail" else 0
        command.motion_lib.native_object_part_pose_support[1, 0] = kind != "missing"
        if kind == "empty":
            command.motion_steps.fill_(3)
        else:
            command._update_metrics()
        callback.on_start(env)
        callback.on_step(env)
        states.append(callback.collect_state())
    destination = write_current_policy_outcomes(
        _Agent(), env, OmegaConf.create({"command_name": "motion", "num_per_motion": 4}),
        [callback], {id(callback): states}, callback.reduce_state(states),
        outcome_dir=tmp_path, env_step=8192,
    )
    report = json.loads(destination.read_text())
    key = _callback(report, ".NativeReferenceTrackingPerformance")
    result = report["motions"][0]["callbacks"][key]
    assert [row["full_status"] for row in result["attempts"]] == [["pass"], ["fail"], ["unscored"], ["unscored"]]
    assert [row["object_status"] for row in result["attempts"]] == [["pass"], ["pass"], ["unscored"], ["unscored"]]
    missing = result["attempts"][2]
    assert missing["observed_steps"] == 1 and missing["missing_required_support"]
    assert missing["masses"]["part_position_error_m"][0] == 0
    assert result["attempts"][3]["observed_steps"] == 0
    joint_names = report["callback_metadata"][key]["feature_names"]["joint_position_error_rad"]
    assert len(joint_names) == (0 if rigid else 2)
    assert len(result["attempts"][0]["masses"]["joint_position_error_rad"]) == len(joint_names)
    assert "no contact/task" in report["callback_metadata"][key]["meaning"]
    assert "UNQUALIFIED" in report["physical_task_or_BOX_success"]


class _RecordedEvaluationEnv(_EvaluationEnv):
    def __init__(self):
        super().__init__()
        self.scene.env_origins = torch.zeros(4, 3)
        self.command.sim_obj_quat_w = torch.zeros(4, 2, 4)
        self.command.sim_obj_quat_w[..., 0] = 1
        self.command.robot = SimpleNamespace(data=SimpleNamespace(
            joint_pos=torch.zeros(4, 2), joint_vel=torch.zeros(4, 2),
            root_link_pos_w=torch.zeros(4, 3), root_link_quat_w=torch.tensor([[1., 0., 0., 0.]] * 4),
        ))

    def reset(self):
        self.command.robot.data.joint_pos[:] = torch.arange(4)[:, None]
        return super().reset()

    def step(self, actions):
        self.command.robot.data.joint_pos += .01
        return super().step(actions)


@pytest.mark.parametrize("enabled", [True, False])
def test_explicit_frozen_eval_keeps_failed_unscored_absent_replicas_and_cpu_identity(tmp_path, enabled):
    env, cfg, agent = _RecordedEvaluationEnv(), _config(), _Agent()
    cfg.callbacks.append({"_target_": "evaluation.callbacks.rollout_record.RolloutRecord", "enabled": enabled})
    success = tmp_path / "success"
    success.mkdir()
    old = '{"saved": [{"motion_id": 0, "first_saved_env_step": 1}]}\n'
    (success / "manifest.json").write_text(old)
    authority = {"checkpoint": "/fixture/full", "source_revision": "frozen-fixture"}
    outcome_dir = tmp_path / "evaluation"
    evaluate(agent, env, cfg, "cpu", success_dir=success, env_step=8192,
             outcome_dir=outcome_dir, recording_dir=outcome_dir / "recordings", frozen_authority=authority)
    report_path = next(outcome_dir.glob("current-policy-*.json"))
    report = json.loads(report_path.read_text())
    manifest_path = outcome_dir / report["factual_recording_manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "frozen_current_policy_attempts_v1"
    assert manifest["frozen_authority"] == report["frozen_authority"] == authority
    assert (manifest_path.parent / manifest["evaluation_report"]).resolve() == report_path
    assert manifest["selected_motion_count"] == 4 and manifest["configured_replicas_per_motion"] == 2
    assert env.exit_count == 1 and torch.equal(env.command.motion_ids, torch.full((4,), 99))
    tracking = _callback(report, ".TrackingPerformance")
    assert report["motions"][0]["callbacks"][tracking]["full_threshold_counts"] == [
        {"pass": 1, "fail": 1, "unscored": 0, "nonfinite": 0}]
    assert report["motions"][3]["callbacks"][tracking]["unscored_attempt_count"] == 2
    absent = manifest["motions"][2]["attempts"]
    assert len(absent) == 2 and all(row["execution"] == "not_evaluated" for row in absent)
    assert all(row["recording"]["reason"] == "attempt_not_evaluated" and row["recording"]["rollout"] is None for row in absent)
    actual = [row for motion in manifest["motions"] for row in motion["attempts"] if row["execution"] == "executed"]
    assert len(actual) == 6
    assert [row["evaluation_row"] for row in manifest["motions"][0]["attempts"]] == [1, 3]
    assert len({row["attempt_id"] for row in actual}) == 6
    for motion, recorded in zip(report["motions"], manifest["motions"], strict=True):
        assert motion["factual_attempts"] == recorded["attempts"]
        for row in motion["callbacks"][tracking]["attempts"]:
            entry = next(a for a in recorded["attempts"] if a["attempt_id"] == row["attempt_id"])
            assert entry["evaluation_row"] == row["evaluation_row"]
            assert entry["replica_index"] == row["replica_index"]
            recording = entry["recording"]
            if not enabled:
                assert recording["status"] == "unavailable" and recording["reason"] == "recorder_disabled"
                continue
            assert recording["status"] == "recorded" and recording["reason"] is None
            with np.load(manifest_path.parent / recording["rollout"], allow_pickle=False) as saved:
                identity = json.loads(str(saved["evaluation_attempt_json"]))
                assert identity["evaluation_id"] == manifest["evaluation_id"]
                assert identity["attempt_id"] == entry["attempt_id"] and identity["frozen_authority"] == authority
                assert int(saved["motion_id"]) == motion["motion_id"] != 99
                assert identity["evaluation_row"] == row["evaluation_row"]
                assert saved["joint_pos"][0, 0] == pytest.approx(entry["env_index"] + .01)
    assert len(list(manifest_path.parent.glob("rollouts/*.npz"))) == (6 if enabled else 0)
    assert (success / "manifest.json").read_text() == old


def test_frozen_eval_without_selected_recorder_retains_actual_unavailable_attempts(tmp_path):
    env, cfg = _EvaluationEnv(), _config()
    evaluate(_Agent(), env, cfg, "cpu", outcome_dir=tmp_path,
             recording_dir=tmp_path / "recordings",
             frozen_authority={"checkpoint": "/fixture/full", "source_revision": "fixture"})
    report = json.loads(next(tmp_path.glob("current-policy-*.json")).read_text())
    manifest = json.loads((tmp_path / report["factual_recording_manifest"]).read_text())
    actual = [row for motion in manifest["motions"] for row in motion["attempts"] if row["execution"] == "executed"]
    assert len(actual) == 6 and all(row["recording"]["reason"] == "no_rollout_recorder_selected" for row in actual)
    assert not list((tmp_path / "recordings").rglob("*.npz"))


def test_frozen_recording_requires_authority_before_any_rollout(tmp_path):
    env, agent = _EvaluationEnv(), _Agent()
    with pytest.raises(ValueError, match="checkpoint and source_revision"):
        evaluate(agent, env, _config(), "cpu", outcome_dir=tmp_path, recording_dir=tmp_path / "records")
    assert agent.calls == env.exit_count == 0


def test_unmasked_cpu_recorder_identity_survives_restore_train_mutation():
    from evaluation.callbacks.rollout_record import RolloutRecord

    env = _RecordedEvaluationEnv()
    env.command.motion_ids[:] = torch.tensor([1, 0, 1, 0])
    env.reset()
    recorder = RolloutRecord()
    recorder.on_start(env)
    env.step(torch.zeros(4, 2))
    recorder.on_step(env)
    state = recorder.collect_state()  # The supported default keeps every env.
    env.exit_eval()  # Restoring TRAIN changes the same CPU motion_ids in place.
    assert state["motion_ids"].tolist() == [1, 0, 1, 0]
    assert recorder.rollout_arrays([state], 0)["motion_id"] == 1
