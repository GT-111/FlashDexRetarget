"""CPU checks of real replay and learning-state owners; no Kit or physical claims."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from agents.flashsac import FlashSACAgent, FlashSACConfig
from agents.storage.replay_buffer import ReplayBuffer
from common.utils import capture_rng_state, restore_rng_state
from envs.curriculum.assist import XfrcAssistCurriculum
from runner.off_policy_runner import LEARNING_STATE_FILE, OffPolicyRunner


def _input_extension_agent(*, added=False, encoded=True, wrong_sides=False, extra_terms=()):
    """Tiny real native-shaped actor; no asset, reference, or physical constructor."""
    base = _agent()
    cfg = base._cfg
    del base
    if encoded:
        cfg = replace(cfg, traj_encoder={
            "enable": True, "aggregation": "conv1d", "n_future": 10, "n_sides": 1,
            "latent_dim": 4, "hidden_dim": 8, "num_layers": 1,
        })
    flat = ([["actor", "robot_joint_vel", 3], ["actor", "native_source_tip_reference_error", 4],
             ["actor", "robot_joint_pos", 3]] if added else
            [["actor", "robot_joint_pos", 3], ["actor", "robot_joint_vel", 3]])
    flat.extend(["actor", name, width] for name, width in extra_terms)
    windows = [["future_traj", 40]] if encoded else []
    info = {
        "actor_observation_size": (sum(term[2] for term in flat),),
        "obs_terms": flat + [[name, "native_future_reference", width] for name, width in windows],
        "encoded_obs_dims": windows, "reward_heads": ["right", "left"],
        "action_sides": {"right": [0, 1, 2, 6, 7, 8, 12], "left": [3, 4, 5, 9, 10, 11, 13]},
    }
    if wrong_sides:
        info["action_sides"] = {"right": info["action_sides"]["left"], "left": info["action_sides"]["right"]}
    width = sum(term[2] for term in info["obs_terms"])
    agent = FlashSACAgent(
        gym.spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32),
        gym.spaces.Box(-1, 1, shape=(14,), dtype=np.float32), info, cfg,
    )
    return agent, info


def _input_extension_config(agent, info, run_dir, overrides=()):
    cfg = _lifecycle_config([f"run_dir={run_dir}", *overrides])
    payload = OmegaConf.to_container(cfg, resolve=True)
    control = {"position_stiffness": 1., "position_damping": 1., "rotation_stiffness": 1.,
               "rotation_damping": 1., "force_limit_n": None, "torque_limit_nm": None}
    scene = {
        "scene_xml_path": "authored-fixture-not-read.xml",
        "hands": [{"side": side, "entity_name": f"{side}_hand", "root_body_name": f"{side}_wrist",
                   "root_joint_name": f"{side}_free", "command_actuator_names": [f"{side}_finger"],
                   "root_control": dict(control)} for side in ("right", "left")],
        "objects": [{"entity_name": "fixture_object", "root_body_name": "object_base", "root_joint_name": "object_free",
                     "parts": [{"name": "base", "body_name": "object_base", "mesh_path": "not-read.obj"}]}],
    }
    payload["robot"]["native_scene"] = scene
    payload["actions"] = {
        "name": "native", "_target_": "envs.actions.native.NativeResidualActionCfg",
        "entity_name": "robot", "command_name": "motion", "native_scene": scene,
        "action_scale": 1., "action_offset": 0., "clip": {".*": [-1., 1.]},
        "translation_residual_m": .1, "rotation_residual_rad": 1., "finger_residual_rad": None,
        "cutoff_hz": {"wrist_trans": 5., "wrist_rot": 5., "finger": None},
    }
    names = [name for group, name, _ in info["obs_terms"] if group == "actor"]
    payload["obs"] = {
        "_target_": "envs.observations.native.NativeMotionTrackingObs",
        "obs_dict": {"actor_obs": names, "critic_obs": names},
        "encoded_groups": {name: "native_future_reference" for name, _ in info["encoded_obs_dims"]},
        "obs_params": {"native_future_reference": {"n_future": 10, "stride": 1}},
        "obs_scales": {}, "noise_scales": {}, "actor_enable_corruption": False,
        "critic_enable_corruption": False, "nan_policy": "disabled",
    }
    payload["runner"]["agent"] = {"class_name": "agents.flashsac.FlashSACAgent", **asdict(agent._cfg)}
    payload["runner"]["share_actor_critic_obs"] = True
    cfg = OmegaConf.create(payload)
    Path(run_dir).mkdir(parents=True)
    OmegaConf.save(cfg, Path(run_dir) / "config.yaml", resolve=True)
    return cfg


def _feed_input_extension_agent(agent):
    width = agent._layout.full_dim
    initial = np.linspace(-.8, .9, 2 * width, dtype=np.float32).reshape(2, width)
    for frame in range(9):
        observation = np.sin(initial * (frame + 1) + np.arange(width)[None, :] / 10).astype(np.float32)
        agent.observe({
            "observation": observation, "next_observation": observation + .02,
            "action": np.zeros((2, 14), dtype=np.float32),
            "reward": np.full((2, 2), .1, dtype=np.float32),
            "terminated": np.zeros(2, dtype=bool), "truncated": np.zeros(2, dtype=bool),
        })


def test_actor_input_extension_preserves_equal_named_layout_with_fresh_learning_owners(tmp_path, monkeypatch):
    parent, old_info = _input_extension_agent()
    with torch.no_grad():
        for tower in parent._alg.actor.sides:
            norm = tower.embedder.norm
            norm.weight.copy_(torch.linspace(.6, 1.3, norm.weight.numel()))
            norm.bias.copy_(torch.linspace(-.2, .15, norm.bias.numel()))
            norm.running_mean.copy_(torch.linspace(-.3, .4, norm.running_mean.numel()))
            norm.running_var.copy_(torch.linspace(.5, 1.5, norm.running_var.numel()))
    checkpoint = tmp_path / "parent-checkpoint"
    parent.save(str(checkpoint))
    parent_run = tmp_path / "parent-run"
    _input_extension_config(parent, old_info, parent_run)
    child, same_info = _input_extension_agent()
    child_run = tmp_path / "child-run"
    _input_extension_config(child, same_info, child_run)
    fresh = deepcopy(child._alg.save())

    authority = child.initialize_actor_input_extension(
        str(checkpoint), str(parent_run / "config.yaml"), str(child_run / "config.yaml"),
    )
    after = child._alg.save()
    _tree_equal(after["actor_state_dict"], parent._alg.actor.state_dict())
    for key in fresh.keys() - {"actor_state_dict"}:
        _tree_equal(after[key], fresh[key])
    assert authority["added_flat_terms"] == [] and authority["actor_only"]
    assert not authority["renormalized_at_initialization"]
    assert child._alg.replay_buffer._num_in_buffer == 0
    assert not child._alg.replay_buffer._n_step_transitions
    assert not child._alg.actor_optimizer.state and not child._alg.critic_optimizer.state
    assert child._alg._last_obs is child._alg._last_actions is child._alg._cached_noise is None
    observations = torch.linspace(-.9, 1.1, 5 * parent._layout.full_dim).reshape(5, -1)
    for actual, expected in zip(child._alg.actor.get_mean_and_std(observations, training=False),
                                parent._alg.actor.get_mean_and_std(observations, training=False), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy(child.act(observations, deterministic=True)),
                               torch.from_numpy(parent.act(observations, deterministic=True)), rtol=0, atol=0)

    # Equal total width does not admit removal or resizing of an existing named feature.
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("check names before model read"))
    for mismatch in ("removed", "resized"):
        invalid, info = _input_extension_agent()
        info["obs_terms"][0] = ["actor", "replacement", 3] if mismatch == "removed" else ["actor", "robot_joint_pos", 2]
        if mismatch == "resized":
            info["obs_terms"][1] = ["actor", "robot_joint_vel", 4]
        invalid._obs_terms = info["obs_terms"]
        invalid_run = tmp_path / mismatch
        _input_extension_config(invalid, info, invalid_run)
        with pytest.raises(ValueError, match="cannot remove or resize named feature"):
            invalid.initialize_actor_input_extension(
                str(checkpoint), str(parent_run / "config.yaml"), str(invalid_run / "config.yaml"),
            )


def _archive_single_object_scene(config):
    """Write the actual archived parent schema; never used for the live child."""
    payload = OmegaConf.to_container(config, resolve=True)
    for scene in (payload["robot"]["native_scene"], payload["actions"]["native_scene"]):
        scene["object"] = scene.pop("objects")[0]
    return OmegaConf.create(payload)


def test_actor_input_extension_preserves_archived_singleton_actor_in_collection_scene(tmp_path):
    parent, info = _input_extension_agent()
    checkpoint, parent_run = tmp_path / "parent-checkpoint", tmp_path / "parent-run"
    parent.save(str(checkpoint))
    old_config = _archive_single_object_scene(_input_extension_config(parent, info, parent_run))
    OmegaConf.save(old_config, parent_run / "config.yaml", resolve=True)
    child, new_info = _input_extension_agent()
    child_run = tmp_path / "child-run"
    _input_extension_config(child, new_info, child_run)
    fresh = deepcopy(child._alg.save())

    authority = child.initialize_actor_input_extension(
        str(checkpoint), str(parent_run / "config.yaml"), str(child_run / "config.yaml"),
    )
    after = child._alg.save()
    _tree_equal(after["actor_state_dict"], parent._alg.actor.state_dict())
    for key in fresh.keys() - {"actor_state_dict"}:
        _tree_equal(after[key], fresh[key])
    observations = torch.linspace(-.9, 1.1, 5 * parent._layout.full_dim).reshape(5, -1)
    torch.testing.assert_close(torch.from_numpy(child.act(observations, deterministic=True)),
                               torch.from_numpy(parent.act(observations, deterministic=True)), rtol=0, atol=0)
    assert authority["native_scene_validation"]["archived_singleton_parent"]
    assert authority["native_scene_validation"]["object_entities"] == ["fixture_object"]
    assert authority["added_flat_terms"] == []
    assert not child._alg.replay_buffer._num_in_buffer
    assert not child._alg.actor_optimizer.state and not child._alg.critic_optimizer.state


@pytest.mark.parametrize("mismatch", ["root_law", "actuator", "part", "object_count", "action", "parent_action"])
def test_actor_input_extension_rejects_changed_native_semantics_before_model_read(tmp_path, monkeypatch, mismatch):
    parent, info = _input_extension_agent()
    parent_run, child_run = tmp_path / "parent-run", tmp_path / "child-run"
    old = _archive_single_object_scene(_input_extension_config(parent, info, parent_run))
    child, current_info = _input_extension_agent()
    current = OmegaConf.to_container(_input_extension_config(child, current_info, child_run), resolve=True)
    if mismatch == "root_law":
        current["robot"]["native_scene"]["hands"][0]["root_control"]["position_stiffness"] = 2.
    elif mismatch == "actuator":
        current["robot"]["native_scene"]["hands"][0]["command_actuator_names"][0] = "different_actuator"
    elif mismatch == "part":
        current["robot"]["native_scene"]["objects"][0]["parts"][0]["body_name"] = "different_part"
    elif mismatch == "object_count":
        current["robot"]["native_scene"]["objects"].append({
            "entity_name": "second_object", "root_body_name": "second_base", "root_joint_name": "second_free",
            "parts": [{"name": "base", "body_name": "second_base", "mesh_path": "another.obj"}],
        })
    elif mismatch == "action":
        current["actions"]["translation_residual_m"] = .2
    else:
        old.actions.native_scene.object.parts[0].body_name = "different_parent_action_part"
    OmegaConf.save(old, parent_run / "config.yaml", resolve=True)
    OmegaConf.save(OmegaConf.create(current), child_run / "config.yaml", resolve=True)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("native semantics must be checked before model read"))
    with pytest.raises(ValueError, match="native action/scene contract"):
        child.initialize_actor_input_extension(str(tmp_path / "not-opened-checkpoint"),
            str(parent_run / "config.yaml"), str(child_run / "config.yaml"))


@pytest.mark.parametrize("encoded", [False, True])
def test_actor_input_extension_preserves_named_actor_and_learns_new_columns(tmp_path, encoded):
    parent, old_info = _input_extension_agent(encoded=encoded)
    _feed_input_extension_agent(parent)
    parent.update()
    parent.update()
    assert parent._alg.update_step == 2 and parent._alg.actor_optimizer.state
    with torch.no_grad():
        for tower in parent._alg.actor.sides:
            norm = tower.embedder.norm
            norm.weight.copy_(torch.linspace(.6, 1.3, norm.weight.numel()))
            norm.bias.copy_(torch.linspace(-.2, .15, norm.bias.numel()))
    checkpoint = tmp_path / "parent-checkpoint"
    parent.save(str(checkpoint))
    parent_run = tmp_path / "parent-run"
    _input_extension_config(parent, old_info, parent_run)
    child, new_info = _input_extension_agent(added=True, encoded=encoded)
    child_run = tmp_path / "child-run"
    _input_extension_config(child, new_info, child_run)
    fresh = deepcopy(child._alg.save())

    authority = child.initialize_actor_input_extension(
        str(checkpoint), str(parent_run / "config.yaml"), str(child_run / "config.yaml"),
    )
    after = child._alg.save()
    for key in fresh.keys() - {"actor_state_dict"}:
        _tree_equal(after[key], fresh[key])
    assert child._alg.replay_buffer._num_in_buffer == 0
    assert not child._alg.replay_buffer._n_step_transitions
    assert child._alg._last_obs is child._alg._last_actions is child._alg._cached_noise is None
    assert authority["added_flat_terms"] == [["actor", "native_source_tip_reference_error"]]
    assert not authority["renormalized_at_initialization"]

    old = torch.linspace(-.9, 1.1, 5 * parent._layout.full_dim).reshape(5, -1)
    new = torch.cat([old[:, 3:6], torch.linspace(-2., 3., 20).reshape(5, 4), old[:, :3], old[:, 6:]], -1)
    for actual, expected in zip(child._alg.actor.get_mean_and_std(new, training=False),
                                parent._alg.actor.get_mean_and_std(old, training=False), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(torch.from_numpy(child.act(new, deterministic=True)),
                               torch.from_numpy(parent.act(old, deterministic=True)), rtol=1e-6, atol=1e-7)
    future = parent._alg.actor.sides[0].traj_encoder
    future_width = future.latent_dim if future is not None else 0
    mapped = [7, 8, 9, 0, 1, 2] + list(range(10, 10 + future_width))
    before, loaded = parent._alg.actor.state_dict(), child._alg.actor.state_dict()
    expanded = set()
    for index, tower in enumerate(child._alg.actor.sides):
        prefix = f"sides.{index}.embedder."
        for field, initial in (("weight", 1), ("bias", 0), ("running_mean", 0), ("running_var", 1)):
            key = prefix + f"norm.{field}"
            expanded.add(key)
            torch.testing.assert_close(loaded[key][mapped], before[key], rtol=0, atol=0)
            torch.testing.assert_close(loaded[key][3:7], torch.full((4,), initial, dtype=loaded[key].dtype),
                                       rtol=0, atol=0)
        key = prefix + "w.w.weight"
        expanded.add(key)
        torch.testing.assert_close(loaded[key][:, mapped], before[key], rtol=0, atol=0)
        assert not bool(loaded[key][:, 3:7].any())
    for key in before.keys() - expanded:
        torch.testing.assert_close(loaded[key], before[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="only available once"):
        child.initialize_actor_input_extension(str(checkpoint), str(parent_run / "config.yaml"),
                                               str(child_run / "config.yaml"))

    # The old strict loader still rejects the old layout; a child's same-layout
    # checkpoint remains an ordinary complete load, without initialization logic.
    with pytest.raises(ValueError, match="obs layout differs"):
        child.load(str(checkpoint))
    saved_child = tmp_path / "child-checkpoint"
    child.save(str(saved_child))
    restored, _ = _input_extension_agent(added=True, encoded=encoded)
    restored.load(str(saved_child))
    _tree_equal(restored._alg.save(), after)

    _feed_input_extension_agent(child)
    info = child.update()  # actual registered SAC loss, reducer, Adam, and parameter normalization
    assert "actor/loss" in info and all(np.isfinite(value) for value in info.values())
    for tower in child._alg.actor.sides:
        assert bool(tower.embedder.w.w.weight.grad[:, 3:7].abs().sum() > 0)
        assert bool(tower.embedder.w.w.weight[:, 3:7].abs().sum() > 0)
        # Zero input weights give zero first VJP to the new input BN affine.
        assert not bool(tower.embedder.norm.weight.grad[3:7].any())
        assert not bool(tower.embedder.norm.bias.grad[3:7].any())


def test_actor_input_extension_admits_added_term_configuration_and_preserves_carry(tmp_path):
    carry = (("native_wrist_control_state", 13),)
    parent, old_info = _input_extension_agent(extra_terms=carry)
    with torch.no_grad():
        for tower in parent._alg.actor.sides:
            norm = tower.embedder.norm
            norm.weight.copy_(torch.linspace(.6, 1.3, norm.weight.numel()))
            norm.bias.copy_(torch.linspace(-.2, .15, norm.bias.numel()))
            norm.running_mean.copy_(torch.linspace(-.3, .4, norm.running_mean.numel()))
            norm.running_var.copy_(torch.linspace(.5, 1.5, norm.running_var.numel()))
    checkpoint, parent_run = tmp_path / "parent-checkpoint", tmp_path / "parent-run"
    parent.save(str(checkpoint))
    parent_config = _input_extension_config(parent, old_info, parent_run)
    points, parts = 1, len(parent_config.robot.native_scene.objects[0].parts)
    hands = len(parent_config.robot.native_scene.hands)
    surface_width = hands * parts * (3 * points + 1)
    surface = (("native_part_point_cloud_wrist", surface_width),
               ("native_next_part_point_cloud_delta_wrist", surface_width))
    child, new_info = _input_extension_agent(extra_terms=(*carry, *surface))
    child_run = tmp_path / "child-run"
    child_config = _input_extension_config(child, new_info, child_run)
    for name, _width in surface:
        child_config.obs.obs_params[name] = {"n_points": points, "n_obj_verts": 4}
        child_config.obs.obs_scales[name] = 1.0
        child_config.obs.noise_scales[name] = 0.0
    OmegaConf.save(child_config, child_run / "config.yaml", resolve=True)
    fresh = deepcopy(child._alg.save())

    authority = child.initialize_actor_input_extension(
        str(checkpoint), str(parent_run / "config.yaml"), str(child_run / "config.yaml"),
    )
    for key in fresh.keys() - {"actor_state_dict"}:
        _tree_equal(child._alg.save()[key], fresh[key])
    assert authority["added_flat_terms"] == [["actor", name] for name, _width in surface]
    assert ["actor", "native_wrist_control_state"] in authority["retained_flat_terms"]
    old_flat = sum(width for group, _name, width in old_info["obs_terms"] if group == "actor")
    new_flat = sum(width for group, _name, width in new_info["obs_terms"] if group == "actor")
    old = torch.linspace(-.9, 1.1, 5 * parent._layout.full_dim).reshape(5, -1)
    added = torch.linspace(-2., 3., 5 * (new_flat - old_flat)).reshape(5, -1)
    new = torch.cat((old[:, :old_flat], added, old[:, old_flat:]), -1)
    for actual, expected in zip(child._alg.actor.get_mean_and_std(new, training=False),
                                parent._alg.actor.get_mean_and_std(old, training=False), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    for index, tower in enumerate(child._alg.actor.sides):
        before = parent._alg.actor.sides[index]
        future_width = before.traj_encoder.latent_dim
        mapped = list(range(old_flat)) + list(range(new_flat, new_flat + future_width))
        for field, initial in (("weight", 1), ("bias", 0), ("running_mean", 0), ("running_var", 1)):
            actual, expected = getattr(tower.embedder.norm, field), getattr(before.embedder.norm, field)
            torch.testing.assert_close(actual[mapped], expected, rtol=0, atol=0)
            torch.testing.assert_close(actual[old_flat:new_flat],
                torch.full_like(actual[old_flat:new_flat], initial), rtol=0, atol=0)
        assert not bool(tower.embedder.w.w.weight[:, old_flat:new_flat].any())
        _tree_equal(tower.traj_encoder.state_dict(), before.traj_encoder.state_dict())
    _feed_input_extension_agent(child)
    metrics = child.update()
    assert "actor/loss" in metrics and all(np.isfinite(value) for value in metrics.values())
    for tower in child._alg.actor.sides:
        assert bool(tower.embedder.w.w.weight.grad[:, old_flat:new_flat].abs().sum() > 0)
        assert bool(tower.embedder.w.w.weight[:, old_flat:new_flat].abs().sum() > 0)


@pytest.mark.parametrize("mismatch", ["retained_params", "encoded_params", "retained_scale",
                                     "retained_noise", "unselected_params", "global"])
def test_actor_input_extension_rejects_changed_old_semantics_with_added_term_params(
    tmp_path, monkeypatch, mismatch,
):
    parent, old_info = _input_extension_agent()
    checkpoint, parent_run, child_run = (
        tmp_path / name for name in ("parent-checkpoint", "parent-run", "child-run")
    )
    parent.save(str(checkpoint))
    _input_extension_config(parent, old_info, parent_run)
    child, new_info = _input_extension_agent(extra_terms=(("native_part_point_cloud_wrist", 8),))
    config = _input_extension_config(child, new_info, child_run)
    config.obs.obs_params.native_part_point_cloud_wrist = {"n_points": 1, "n_obj_verts": 4}
    if mismatch == "retained_params":
        config.obs.obs_params.robot_joint_pos = {"asset_cfg": {"name": "different_robot"}}
    elif mismatch == "encoded_params":
        config.obs.obs_params.native_future_reference.stride = 2
    elif mismatch == "retained_scale":
        config.obs.obs_scales.robot_joint_pos = .5
    elif mismatch == "retained_noise":
        config.obs.noise_scales.robot_joint_pos = .1
    elif mismatch == "unselected_params":
        config.obs.obs_params.not_an_added_input = {"n_points": 1}
    else:
        config.obs.actor_enable_corruption = True
    OmegaConf.save(config, child_run / "config.yaml", resolve=True)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs:
                        pytest.fail("old semantics must be checked before actor read"))
    with pytest.raises(ValueError, match="observation transforms and future queries"):
        child.initialize_actor_input_extension(str(checkpoint), str(parent_run / "config.yaml"),
                                               str(child_run / "config.yaml"))


@pytest.mark.parametrize("mismatch", ["missing_names", "future_encoder", "future_query", "native_action", "side_scatter"])
def test_actor_input_extension_rejects_unproven_metadata_before_model_read(tmp_path, monkeypatch, mismatch):
    parent, old_info = _input_extension_agent()
    checkpoint = tmp_path / "parent-checkpoint"
    parent.save(str(checkpoint))
    parent_run, child_run = tmp_path / "parent-run", tmp_path / "child-run"
    parent_cfg = _input_extension_config(parent, old_info, parent_run)
    child, new_info = _input_extension_agent(added=True, wrong_sides=mismatch == "side_scatter")
    _input_extension_config(child, new_info, child_run)
    if mismatch == "missing_names":
        (checkpoint / "obs_layout.json").write_text(json.dumps({"encoded_obs_dims": old_info["encoded_obs_dims"]}))
        reason = "saved named obs_terms"
    elif mismatch == "future_encoder":
        parent_cfg.runner.agent.traj_encoder.n_future = 5
        reason = "actor configuration: traj_encoder"
    elif mismatch == "future_query":
        parent_cfg.obs.obs_params.native_future_reference.stride = 2
        reason = "future queries"
    elif mismatch == "native_action":
        parent_cfg.actions.finger_residual_rad = .2
        reason = "native action/scene contract"
    else:
        reason = "action columns disagree"
    OmegaConf.save(parent_cfg, parent_run / "config.yaml", resolve=True)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("metadata must be checked before model read"))
    with pytest.raises(ValueError, match=reason):
        child.initialize_actor_input_extension(str(checkpoint), str(parent_run / "config.yaml"),
                                               str(child_run / "config.yaml"))


def test_runner_routes_actor_input_extension_without_restoring_training_or_ever_state(tmp_path, monkeypatch):
    import runner.off_policy_runner as owner

    parent, old_info = _input_extension_agent()
    checkpoint = tmp_path / "parent-checkpoint"
    parent.save(str(checkpoint))
    (checkpoint / "resume_state.json").write_text(json.dumps({"ever": {"old": [[1]]}}))
    parent_run = tmp_path / "parent-run"
    _input_extension_config(parent, old_info, parent_run)
    child, new_info = _input_extension_agent(added=True)
    child_run = tmp_path / "child-run"
    cfg = _input_extension_config(child, new_info, child_run, [
        f"actor_input_extension_init_path={checkpoint}",
        f"actor_input_extension_init_config_path={parent_run / 'config.yaml'}",
    ])
    assert cfg.runner.actor_input_extension_init_path == str(checkpoint)
    assert cfg.runner.actor_input_extension_init_config_path == str(parent_run / "config.yaml")
    fresh = deepcopy(child._alg.save())
    flat = SimpleNamespace(reset=lambda: (np.zeros((2, child._layout.full_dim), dtype=np.float32), new_info),
        current_aux=lambda: None, action_space=gym.spaces.Box(-1, 1, shape=(2, 14), dtype=np.float32))
    monkeypatch.setattr(owner.FlatObsVecEnv, "from_env", lambda *_args, **_kwargs: flat)
    monkeypatch.setattr(owner, "build_agent", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(owner.wandb, "run", None)
    for name in ("load", "load_buffer"):
        monkeypatch.setattr(child, name, lambda *_args: pytest.fail("actor initialization must not call resume loaders"))
    physical = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: None))
    env = SimpleNamespace(unwrapped=physical)
    runner = OffPolicyRunner(env, cfg.runner, cfg.eval, eval_env=SimpleNamespace())
    for key in fresh.keys() - {"actor_state_dict"}:
        _tree_equal(child._alg.save()[key], fresh[key])
    assert runner._step_offset == 0 and not runner._eval_ever_state and runner._nonfinite_steps == 0
    receipt = json.loads((child_run / "actor-input-extension.json").read_text())
    assert receipt["actor_only"] and receipt["checkpoint"] == str(checkpoint)
    for key, value in (("agent_load_path", str(checkpoint)), ("buffer_load_path", str(checkpoint)),
                       ("learning_state_load_path", str(checkpoint)), ("step_offset", 2), ("ever_state_path", str(checkpoint / "resume_state.json"))):
        invalid = OmegaConf.create(OmegaConf.to_container(cfg.runner, resolve=True))
        invalid[key] = value
        with pytest.raises(ValueError, match="fresh progress"):
            OffPolicyRunner(env, invalid, cfg.eval, eval_env=SimpleNamespace())


def test_log_metrics_preserves_all_scalar_values_in_stdout_and_wandb(monkeypatch, capsys):
    import common.logger as logger

    metrics = {f"episode/reward_{index}": index + .123456789 for index in range(10)}
    metrics.update({"actor/loss": 1.234567890123456e-8, "critic/loss": .9876543210987654,
                    "temperature/value": 2.345678901234567e-7})
    before = metrics.copy()
    calls = []

    def wandb_log(values, *, step):
        assert values is metrics and values == before
        calls.append((values, step))

    monkeypatch.setattr(logger.wandb, "log", wandb_log)
    logger.log_metrics(metrics, step=19_998_720)

    assert len(calls) == 1 and calls[0][0] is metrics and calls[0][1] == 19_998_720
    assert metrics == before
    assert capsys.readouterr().out == f"[train] step=19998720 {before}\n"


def _stop_loop(tmp_path, monkeypatch, *, ready=True):
    """Real loop/checkpoint orchestration with explicit small fake numerical writers."""
    import runner.off_policy_runner as runner_module

    monkeypatch.setattr(runner_module.wandb, "run", None)
    monkeypatch.setattr(runner_module, "log_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.delenv("FDR_ITER_TIMING_FILE", raising=False)
    runner = OffPolicyRunner.__new__(OffPolicyRunner)
    runner._cfg = OmegaConf.create({"save_replay_buffer": False})
    runner._save_path_base = tmp_path
    runner._stop_request_path = tmp_path / "stop.request"
    runner._train_cmd = None
    runner._eval_ever_state = {}
    runner._n_envs, runner._step_offset = 2, 0
    runner._updates_per_step, runner._log_every = 1, 0
    runner._eval_stride, runner._save_stride = 2, 0
    runner._start_clock = lambda: None
    runner._cost_metrics = lambda _step: {}
    runner._finish = lambda: pytest.fail("a requested stop must not run weights-only finish")
    runner._eval_supported = lambda: pytest.fail("stop must precede optional evaluation")
    events = []

    def save(path):
        target = Path(path)
        assert target.name == "stop2.tmp"
        assert runner._stop_request_path.is_file() and not (tmp_path / "stop2").exists()
        target.mkdir()
        (target / "networks.json").write_text("{}")
        events.append("networks")

    def save_buffer(path):
        assert runner._stop_request_path.is_file() and not (tmp_path / "stop2").exists()
        (Path(path) / "replay.json").write_text("{}")
        events.append("replay")

    def save_learning(path, env_step):
        assert runner._stop_request_path.is_file() and not (tmp_path / "stop2").exists()
        (Path(path) / "learning.json").write_text(json.dumps({"env_step": env_step}))
        events.append("learning")

    runner._agent = SimpleNamespace(ready=lambda: ready, update=lambda: {},
                                    save=save, save_buffer=save_buffer)
    runner._save_learning_state = save_learning
    runner._collect = lambda *_args: events.append("collect")
    return runner, events


@pytest.mark.parametrize("request_during", ["update", "evaluation", "final_evaluation"])
def test_stop_request_created_during_update_or_evaluation_commits_full_then_returns(
    monkeypatch, tmp_path, request_during,
):
    runner, events = _stop_loop(tmp_path, monkeypatch)
    timing = tmp_path / "timing.csv"
    monkeypatch.setenv("FDR_ITER_TIMING_FILE", str(timing))

    def update():
        events.append("update")
        if request_during == "update":
            runner._stop_request_path.write_text("operator request\n")
        return {}

    def evaluate(_step):
        events.append("evaluation")
        runner._stop_request_path.write_text("operator request\n")
        return {}

    runner._agent.update = update
    if request_during != "update":
        runner._eval_supported = lambda: True
        runner._eval_env = SimpleNamespace(shares_train_env=False)
        runner._run_eval = evaluate
        runner._all_solved = lambda *_args: pytest.fail("stop must precede all-solved return")
    if request_during == "final_evaluation":
        runner._eval_stride = 0
    runner.learn(1 if request_during == "final_evaluation" else 3)

    checkpoint = tmp_path / "stop2"
    assert not runner._stop_request_path.exists() and not (tmp_path / "stop2.tmp").exists()
    assert (checkpoint / "stop.request").read_text() == "operator request\n"
    assert {path.name for path in checkpoint.iterdir()} == {
        "networks.json", "replay.json", "learning.json", "resume_state.json", "stop.request",
    }
    assert json.loads((checkpoint / "resume_state.json").read_text())["ckpt"] == str(checkpoint)
    assert events.count("collect") == events.count("update") == 1
    assert events[-3:] == ["networks", "replay", "learning"]
    assert len(timing.read_text().splitlines()) == 2


def test_warmup_stop_request_commits_without_update(monkeypatch, tmp_path):
    runner, events = _stop_loop(tmp_path, monkeypatch, ready=False)
    runner._agent.update = lambda: pytest.fail("warmup stop must not manufacture an update")

    def collect(*_args):
        events.append("collect")
        runner._stop_request_path.write_text("stop during warmup\n")

    runner._collect = collect
    runner.learn(3)
    assert events == ["collect", "networks", "replay", "learning"]
    assert (tmp_path / "stop2" / "stop.request").read_text() == "stop during warmup\n"


def test_stop_save_failure_preserves_request_and_no_committed_checkpoint(
    monkeypatch, capsys, tmp_path,
):
    runner, _events = _stop_loop(tmp_path, monkeypatch)
    runner._stop_request_path.write_text("retain this request\n")

    def fail_replay(path):
        (Path(path) / "partial-replay").write_text("incomplete")
        raise OSError("replay disk full")

    runner._agent.save_buffer = fail_replay
    with pytest.raises(OSError, match="replay disk full"):
        runner.learn(3)
    assert runner._stop_request_path.read_text() == "retain this request\n"
    assert not (tmp_path / "stop2").exists() and not (tmp_path / "stop2.tmp").exists()
    assert "graceful stop at" not in capsys.readouterr().out


def _buffer(capacity=64):
    return ReplayBuffer(
        TensorDict({"obs": torch.zeros(1, 2), "aux": torch.zeros(1, 2, dtype=torch.int32)}, batch_size=[1]),
        num_actions=1, n_step=3, gamma=.9, max_length=capacity,
        min_length=8, sample_batch_size=4, device="cpu", obs_storage_dtype=torch.float16,
        aux_groups=("aux",),
    )


def _append(buffer, start, count):
    for frame in range(start, start + count):
        obs = torch.tensor([[frame, 0], [frame, 1]], dtype=torch.float32)
        buffer.add({
            "observation": TensorDict({"obs": obs, "aux": obs.to(torch.int32)}, batch_size=[2]),
            "next_observation": TensorDict({"obs": obs + torch.tensor([1., 0.]),
                "aux": (obs + torch.tensor([1., 0.])).to(torch.int32)}, batch_size=[2]),
            "action": torch.zeros(2, 1), "reward": torch.ones(2),
            "terminated": torch.zeros(2), "truncated": torch.zeros(2),
        })


def test_sparse_save_serializes_occupied_rows_and_keeps_live_capacity(tmp_path):
    buffer = _buffer(capacity=4096)
    _append(buffer, 0, 9)
    path = tmp_path / "replay.pt"
    buffer.save(str(path))
    saved = torch.load(path, map_location="cpu", mmap=True)
    assert saved["capacity"] == buffer._max_length == 4096
    assert saved["num_in_buffer"] == 14
    assert saved["pending_collector_rows"] == 4
    for tensor in [*saved["observation"].values(), *(saved[key] for key in buffer._ROW_ATTRS)]:
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert buffer._observations["obs"].shape == (4096, 2)
    assert saved["observation"]["aux"].dtype == torch.int32
    torch.testing.assert_close(saved["observation"]["obs"], buffer._observations["obs"][:14])


@pytest.mark.parametrize("restart", ["fresh_process", "shared_eval"])
def test_old_frontier_keeps_factual_successors_after_new_physical_fragment(tmp_path, restart):
    buffer = _buffer()
    _append(buffer, 0, 10)
    if restart == "fresh_process":
        path = tmp_path / "replay.pt"
        buffer.save(str(path))
        buffer = _buffer()
        buffer.load(str(path))
    else:
        buffer.end_collection()
    # The old implicit tail is observations5,6,7, with factual successors8,9,10.
    _append(buffer, 100, 9)
    samples = buffer.sample(torch.arange(10, 16))
    expected = torch.tensor([[8, 0], [8, 1], [9, 0], [9, 1], [10, 0], [10, 1]], dtype=torch.float32)
    torch.testing.assert_close(samples["next_observation"]["obs"], expected)
    torch.testing.assert_close(samples["reward"], torch.full((6,), 1 + .9 + .9**2))
    # Ordinary sampling, including the old now-eligible edge, sees the same facts.
    eligible = buffer._num_in_buffer - buffer._n_step * buffer._add_batch_size
    all_rows = buffer.sample(torch.arange(eligible))
    torch.testing.assert_close(all_rows["next_observation"]["obs"][10:16], expected)
    assert all_rows["next_observation"]["obs"][16, 0] == 103
    # Once overwritten by natural wraparound, old edge overrides must disappear.
    _append(buffer, 109, 40)
    overwritten = buffer.sample(torch.arange(10, 16))
    assert bool((overwritten["observation"]["obs"][:, 0] >= 100).all())
    torch.testing.assert_close(overwritten["next_observation"]["obs"][:, 0],
                               overwritten["observation"]["obs"][:, 0] + 3)


def test_timeout_endpoint_survives_full_resume_and_fragment_boundary(tmp_path):
    buffer = _buffer()
    for frame in range(9):
        obs = torch.tensor([[frame if frame <= 6 else frame + 100, 0], [frame, 1]], dtype=torch.float32)
        next_obs = obs + torch.tensor([1., 0.])
        if frame == 6:
            next_obs[0, 0] = 6.25
        buffer.add({
            "observation": TensorDict({"obs": obs, "aux": obs.to(torch.int32)}, batch_size=[2]),
            "next_observation": TensorDict({"obs": next_obs, "aux": next_obs.to(torch.int32)}, batch_size=[2]),
            "action": torch.zeros(2, 1), "reward": torch.ones(2),
            "terminated": torch.zeros(2), "truncated": torch.tensor([frame == 6, False]),
        })
    path = tmp_path / "replay.pt"
    buffer.save(str(path))
    restored = _buffer()
    restored.load(str(path))
    _append(restored, 1000, 9)
    sample = restored.sample(torch.tensor([8, 9]))
    torch.testing.assert_close(sample["next_observation"]["obs"], torch.tensor([[6.25, 0], [7, 1]]))
    torch.testing.assert_close(sample["truncated"], torch.tensor([1., 0.]))


def test_rng_roundtrip_covers_independent_gym_and_learning_generators(tmp_path):
    action_space = gym.spaces.Box(-1, 1, shape=(2, 3), dtype=np.float32)
    action_space.seed(17)
    random.seed(18)
    np.random.seed(19)
    torch.manual_seed(20)
    path = tmp_path / "rng.pt"
    torch.save(capture_rng_state(action_space), path)
    expected = (random.random(), np.random.randn(3), torch.randn(3), action_space.sample())
    random.seed(200)
    np.random.seed(201)
    torch.manual_seed(202)
    action_space.seed(203)
    restore_rng_state(torch.load(path, weights_only=False), action_space)
    assert random.random() == expected[0]
    np.testing.assert_array_equal(np.random.randn(3), expected[1])
    torch.testing.assert_close(torch.randn(3), expected[2], rtol=0, atol=0)
    np.testing.assert_array_equal(action_space.sample(), expected[3])


def _curriculum():
    command = SimpleNamespace()
    curriculum = XfrcAssistCurriculum(
        command=command, num_envs=2, max_episode_length=100, num_steps_per_env=4,
        kp_init=80, kv_init=5, force_range=50, wait_epochs=0, deque_len=2,
        reward_terms={"task": ("task",)}, rew_thresholds={"task": .6},
        upper_ratios={"kp": .9, "kv": .9, "fr": 1},
        lower_ratios={"kp": 1, "kv": 1, "fr": 1},
        dialback_completion_thres=.5, dialback_min_epochs=500,
        dialback_ratios={"kp": .98, "kv": .98, "fr": .98},
        decay_cooldown_epochs=1, kp_zero_thres=.05, kp_lower_zero_thres=.1,
        completion_thres=.65, device="cpu",
    )
    env = SimpleNamespace(
        common_step_counter=0, episode_length_buf=torch.zeros(2, dtype=torch.long),
        reward_manager=SimpleNamespace(active_terms=["task"],
            _term_cfgs=[SimpleNamespace(weight=1)], _step_reward=torch.ones(2, 1)),
        termination_manager=SimpleNamespace(time_outs=torch.zeros(2, dtype=torch.bool),
                                            terminated=torch.zeros(2, dtype=torch.bool)),
    )
    return curriculum, env


def _curriculum_step(curriculum, env, step):
    env.common_step_counter = step
    env.episode_length_buf += 1
    curriculum.on_step(env)


def test_curriculum_restores_histories_and_partial_epoch_on_fresh_control_clock():
    original, env = _curriculum()
    for step in range(1, 7):
        _curriculum_step(original, env, step)
    saved = original.state_dict()
    assert saved["control_step"] == 6 and saved["epoch_steps"] == 2
    restored, fresh_env = _curriculum()
    restored.load_state_dict(saved, fresh_env.common_step_counter)
    assert restored.state_dict() == saved
    assert restored._prev_ep_len is None
    for original_step, fresh_step in [(7, 1), (8, 2), (9, 3)]:
        _curriculum_step(original, env, original_step)
        _curriculum_step(restored, fresh_env, fresh_step)
    # Physical episode lengths differ; global history, phase, gains and decisions agree.
    assert restored.state_dict() == original.state_dict()


def test_shared_env_runner_reset_does_not_count_old_physical_episode_as_completed():
    curriculum, env = _curriculum()
    for step in range(1, 7):
        _curriculum_step(curriculum, env, step)
    global_state = curriculum.state_dict()
    assert curriculum._prev_ep_len.tolist() == [6, 6]
    buffer = _buffer()
    _append(buffer, 0, 10)

    def reset_worlds():
        env.episode_length_buf.zero_()
        return np.zeros((2, 2), dtype=np.float32), {}

    runner = OffPolicyRunner.__new__(OffPolicyRunner)
    runner._agent = SimpleNamespace(reset_collection=buffer.end_collection)
    runner._train_cmd = SimpleNamespace(_xfrc_curr_ctrl=curriculum)
    runner._train_env = SimpleNamespace(reset=reset_worlds)
    runner._obs_aux = None
    runner._reset_train_env()
    assert not buffer._n_step_transitions
    assert buffer._boundary_next_observations
    assert curriculum.state_dict() == global_state
    assert curriculum._prev_ep_len is None
    _curriculum_step(curriculum, env, 7)
    assert not curriculum._epoch_achieved
    assert curriculum._epoch_steps == global_state["epoch_steps"] + 1


def _agent():
    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"),
                               version_base=None):
        cfg = compose(config_name="flashdexretarget", overrides=[
            "commands.motion_file=synthetic-not-read.pt", "num_envs=2", "device=cpu",
            "runner.agent.device_type=cpu", "runner.agent.buffer_device_type=cpu",
            "runner.agent.buffer_max_length=64", "runner.agent.buffer_min_length=8",
            "runner.agent.sample_batch_size=4", "runner.agent.n_step=3",
            "runner.agent.actor_hidden_dim=8", "runner.agent.critic_hidden_dim=16",
            "runner.agent.critic_num_bins=11", "runner.agent.use_amp=false",
            "runner.agent.use_compile=false", "runner.agent.buffer_compact_obs=false",
            "runner.agent.asymmetric_observation=false", "runner.agent.traj_encoder.enable=false",
        ])
        fields = OmegaConf.to_container(cfg.runner.agent, resolve=True)
    fields.pop("class_name")
    return FlashSACAgent(
        gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
        {"actor_observation_size": (6,), "reward_heads": ["right", "left"],
         "action_sides": {"right": [0, 1], "left": [2, 3]}},
        FlashSACConfig(**fields),
    )


def _feed_agent(agent):
    for frame in range(9):
        agent.observe({
            "observation": np.full((2, 6), frame / 100, dtype=np.float32),
            "next_observation": np.full((2, 6), (frame + 1) / 100, dtype=np.float32),
            "action": np.zeros((2, 4), dtype=np.float32),
            "reward": np.full((2, 2), .01, dtype=np.float32),
            "terminated": np.zeros(2, dtype=bool), "truncated": np.zeros(2, dtype=bool),
        })


def _tree_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _tree_equal(a, b)
    else:
        assert left == right


def _runner(agent, action_space, curriculum, path):
    # Real save/restore methods with small CPU owners; no alternate training loop.
    runner = OffPolicyRunner.__new__(OffPolicyRunner)
    runner._agent = agent
    runner._train_env = SimpleNamespace(action_space=action_space,
                                      unwrapped=SimpleNamespace(common_step_counter=curriculum._last_control_step))
    runner._train_cmd = SimpleNamespace(_xfrc_curr_ctrl=curriculum)
    runner._n_envs, runner._step_offset = 2, 100
    runner._nonfinite_steps = 0
    runner._wall_offset, runner._loop_start = 3., time.perf_counter()
    runner._save_path_base = path
    return runner


@pytest.mark.parametrize("resume_mode", ["agent_only", "full", "fresh_replay_learning_state"])
def test_runner_load_restarts_episode_returns_once_without_erasing_learned_state(
    tmp_path, monkeypatch, resume_mode,
):
    import runner.off_policy_runner as runner_module

    monkeypatch.setattr(runner_module.wandb, "run", None)
    original = _agent()
    _feed_agent(original)
    original.update()
    original.update()
    saved_algorithm = original._alg.save()
    assert bool(saved_algorithm["reward_normalizer_state_dict"]["G_r"].any())
    original_curriculum, original_env = _curriculum()
    for step in range(1, 7):
        _curriculum_step(original_curriculum, original_env, step)
    checkpoint = tmp_path / "parent"
    original.save(str(checkpoint))
    original.save_buffer(str(checkpoint))
    parent_space = gym.spaces.Box(-1, 1, shape=(2, 4), dtype=np.float32)
    parent_space.seed(21)
    parent = _runner(original, parent_space, original_curriculum, tmp_path / "first")
    if resume_mode == "fresh_replay_learning_state":
        parent._nonfinite_steps = 7
    parent._save_learning_state(checkpoint, 100)
    saved_learning = torch.load(checkpoint / LEARNING_STATE_FILE, weights_only=False)

    fresh = _agent()
    fresh_curriculum, fresh_env = _curriculum()
    new_curriculum_state = fresh_curriculum.state_dict()
    command = SimpleNamespace(_xfrc_curr_ctrl=fresh_curriculum)
    physical = SimpleNamespace(common_step_counter=fresh_env.common_step_counter,
        command_manager=SimpleNamespace(get_term=lambda _name: command))
    flat = SimpleNamespace(unwrapped=physical,
        reset=lambda: (np.zeros((2, 6), dtype=np.float32), {}), current_aux=lambda: None,
        action_space=gym.spaces.Box(-1, 1, shape=(2, 4), dtype=np.float32),
        step=lambda _actions: (np.ones((2, 6), dtype=np.float32),
            np.full((2, 2), .25, dtype=np.float32), np.zeros(2, dtype=bool),
            np.zeros(2, dtype=bool), {}))
    monkeypatch.setattr(runner_module.FlatObsVecEnv, "from_env", lambda *_args, **_kwargs: flat)
    monkeypatch.setattr(runner_module, "build_agent", lambda *_args, **_kwargs: fresh)
    events = []
    real_load, real_load_buffer = fresh.load, fresh.load_buffer
    real_reset = fresh.reset_collection

    def load(path):
        real_load(path)
        assert bool(fresh._alg.reward_normalizer.state_dict()["G_r"].any())
        events.append("agent")

    def load_buffer(path):
        real_load_buffer(path)
        events.append("replay")

    def reset_collection():
        real_reset()
        events.append("collection")

    monkeypatch.setattr(fresh, "load", load)
    monkeypatch.setattr(fresh, "load_buffer", load_buffer)
    monkeypatch.setattr(fresh, "reset_collection", reset_collection)
    config = _lifecycle_config([f"run_dir={tmp_path / 'child'}"])
    config.runner.agent = OmegaConf.create({"class_name": "agents.flashsac.FlashSACAgent",
        **asdict(fresh._cfg)})
    config.runner.agent_load_path = str(checkpoint)
    config.runner.buffer_load_path = str(checkpoint) if resume_mode == "full" else None
    config.runner.learning_state_load_path = str(checkpoint) if resume_mode == "fresh_replay_learning_state" else None
    config.runner.step_offset = 0 if resume_mode == "agent_only" else 100
    loaded_payloads = []
    real_torch_load = torch.load

    def observed_load(path, *args, **kwargs):
        loaded_payloads.append(Path(path).name)
        if resume_mode == "fresh_replay_learning_state" and Path(path).name == "replay_buffer.pt":
            pytest.fail("fresh-replay learning-state restore must not open old rewards")
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", observed_load)
    torch.manual_seed(312)
    new_torch_rng = torch.get_rng_state().clone()
    runner = OffPolicyRunner(SimpleNamespace(unwrapped=physical), config.runner, config.eval,
        eval_env=SimpleNamespace())

    assert events == (["agent", "replay", "collection"] if resume_mode == "full"
                      else ["agent", "collection"])
    loaded = fresh._alg.save()
    for key in saved_algorithm.keys() - {"reward_normalizer_state_dict"}:
        _tree_equal(loaded[key], saved_algorithm[key])
    for key in saved_algorithm["reward_normalizer_state_dict"].keys() - {"G_r"}:
        _tree_equal(loaded["reward_normalizer_state_dict"][key],
            saved_algorithm["reward_normalizer_state_dict"][key])
    assert not bool(loaded["reward_normalizer_state_dict"]["G_r"].any())
    assert fresh._alg._last_obs is fresh._alg._last_actions is fresh._alg._cached_noise is None
    assert fresh._alg._noise_repeat_n == 1 and fresh._alg._noise_repeat_count == 0
    assert not fresh._alg.replay_buffer._n_step_transitions
    if resume_mode != "agent_only":
        assert fresh_curriculum.state_dict() == saved_learning["curriculum"]
        _tree_equal(capture_rng_state(flat.action_space), saved_learning["rng"])
        assert runner._step_offset == 100 and runner._nonfinite_steps == saved_learning["nonfinite_steps"]
        assert LEARNING_STATE_FILE in loaded_payloads
        if resume_mode == "full":
            assert fresh._alg.replay_buffer._num_in_buffer == original._alg.replay_buffer._num_in_buffer
            assert "replay_buffer.pt" in loaded_payloads
        else:
            assert fresh._alg.replay_buffer._num_in_buffer == 0
            assert "replay_buffer.pt" not in loaded_payloads and runner._nonfinite_steps == 7
            receipt = json.loads((tmp_path / "child/learning-restore.json").read_text())
            assert receipt["checkpoint"] == str(checkpoint) and receipt["env_step"] == 100
            assert receipt["curriculum_restored"]
    else:
        assert fresh_curriculum.state_dict() == new_curriculum_state
        torch.testing.assert_close(torch.get_rng_state(), new_torch_rng, rtol=0, atol=0)
        assert fresh._alg.replay_buffer._num_in_buffer == 0
    # The actual first fresh collection must not include the parent's discounted return.
    runner._collect(1, runner_module.AverageMeter())
    torch.testing.assert_close(fresh._alg.reward_normalizer.state_dict()["G_r"],
        torch.full((2, 2), .25), rtol=0, atol=0)


def test_real_sac_learning_and_full_owner_restore_keep_schedules_rng_and_fresh_episode(tmp_path):
    original = _agent()
    _feed_agent(original)
    for _ in range(2):
        info = original.update()
        assert info and all(np.isfinite(value) for value in info.values())
    assert original._alg.update_step == 2
    curriculum, env = _curriculum()
    for step in range(1, 7):
        _curriculum_step(curriculum, env, step)
    space = gym.spaces.Box(-1, 1, shape=(2, 4), dtype=np.float32)
    space.seed(21)
    checkpoint = tmp_path / "full"
    checkpoint.mkdir()
    original.save(str(checkpoint))
    original.save_buffer(str(checkpoint))
    runner = _runner(original, space, curriculum, tmp_path / "first")
    runner._save_learning_state(checkpoint, 100)
    state = torch.load(checkpoint / LEARNING_STATE_FILE, weights_only=False)
    restore_rng_state(state["rng"], space)
    expected_draws = (torch.randn(3), space.sample())

    restored = _agent()
    restored.load(str(checkpoint))
    restored.load_buffer(str(checkpoint))
    fresh_curriculum, _ = _curriculum()
    fresh_space = gym.spaces.Box(-1, 1, shape=(2, 4), dtype=np.float32)
    fresh = _runner(restored, fresh_space, fresh_curriculum, tmp_path / "fresh")
    saved_algorithm = torch.load(checkpoint / "flashsac.pt")
    before_reset = restored._alg.save()
    _tree_equal(saved_algorithm, before_reset)
    fresh._restore_learning_state(checkpoint)
    actual_restored = torch.load(fresh._save_path_base / "learning-restored.pt", weights_only=False)
    _tree_equal(actual_restored["rng"], state["rng"])
    assert actual_restored["curriculum"] == state["curriculum"]
    torch.testing.assert_close(torch.randn(3), expected_draws[0], rtol=0, atol=0)
    np.testing.assert_array_equal(fresh_space.sample(), expected_draws[1])
    assert fresh_curriculum.state_dict() == state["curriculum"]
    assert restored._alg._cached_noise is None
    assert not restored._alg.replay_buffer._n_step_transitions
    reward_state = restored._alg.reward_normalizer.state_dict()
    assert not bool(reward_state["G_r"].any())
    for key in reward_state.keys() - {"G_r"}:
        _tree_equal(reward_state[key], saved_algorithm["reward_normalizer_state_dict"][key])
    for key in ["actor_optimizer_state_dict", "critic_optimizer_state_dict",
                "temperature_optimizer_state_dict", "actor_scheduler_state_dict",
                "critic_scheduler_state_dict", "temperature_scheduler_state_dict",
                "grad_scaler_state_dict", "update_step"]:
        _tree_equal(restored._alg.save()[key], saved_algorithm[key])
    restored.update()
    assert restored._alg.update_step == 3
    assert restored._alg.critic_scheduler.last_epoch == original._alg.critic_scheduler.last_epoch + 1


def test_stop_full_restore_uses_new_output_and_preserves_existing_state(monkeypatch, tmp_path):
    import runner.off_policy_runner as runner_module

    monkeypatch.setattr(runner_module.wandb, "run", None)
    monkeypatch.delenv("FDR_ITER_TIMING_FILE", raising=False)
    original = _agent()
    _feed_agent(original)
    original.update()
    curriculum, env = _curriculum()
    for step in range(1, 7):
        _curriculum_step(curriculum, env, step)
    space = gym.spaces.Box(-1, 1, shape=(2, 4), dtype=np.float32)
    space.seed(21)
    output = tmp_path / "parent"
    output.mkdir()
    runner = _runner(original, space, curriculum, output)
    runner._cfg = OmegaConf.create({"save_replay_buffer": False})
    runner._stop_request_path = output / "stop.request"
    runner._restored_loop_wall_clock = None
    runner._eval_ever_state = {}
    runner._updates_per_step, runner._log_every = 1, 0
    runner._eval_stride = runner._save_stride = 0
    runner._eval_supported = lambda: pytest.fail("stopped FULL must return before final evaluation")
    runner._finish = lambda: pytest.fail("stopped FULL must not delete or overwrite replay")

    def collect(*_args):
        original.observe({
            "observation": np.full((2, 6), .09, dtype=np.float32),
            "next_observation": np.full((2, 6), .10, dtype=np.float32),
            "action": np.zeros((2, 4), dtype=np.float32),
            "reward": np.full((2, 2), .01, dtype=np.float32),
            "terminated": np.zeros(2, dtype=bool), "truncated": np.zeros(2, dtype=bool),
        })

    update_owner = original.update

    def update_and_request():
        info = update_owner()
        runner._stop_request_path.write_text("full parent\n")
        return info

    runner._collect = collect
    original.update = update_and_request
    runner.learn(3)
    checkpoint = output / "stop102"
    assert (checkpoint / "stop.request").read_text() == "full parent\n"
    assert not (output / "stop.request").exists() and not (output / "final").exists()
    state = torch.load(checkpoint / LEARNING_STATE_FILE, weights_only=False)
    saved_algorithm = torch.load(checkpoint / "flashsac.pt", weights_only=False)
    assert state["env_step"] == 102 and saved_algorithm["update_step"] == 2
    assert json.loads((checkpoint / "resume_state.json").read_text())["ckpt"] == str(checkpoint)
    restore_rng_state(state["rng"], space)
    expected_draws = (torch.randn(3), space.sample())

    restored = _agent()
    restored.load(str(checkpoint))
    restored.load_buffer(str(checkpoint))
    fresh_curriculum, _ = _curriculum()
    fresh_space = gym.spaces.Box(-1, 1, shape=(2, 4), dtype=np.float32)
    fresh = _runner(restored, fresh_space, fresh_curriculum, tmp_path / "child")
    fresh._step_offset = 102
    fresh._restore_learning_state(checkpoint)
    actual = torch.load(fresh._save_path_base / "learning-restored.pt", weights_only=False)
    _tree_equal(actual["rng"], state["rng"])
    assert actual["curriculum"] == state["curriculum"]
    torch.testing.assert_close(torch.randn(3), expected_draws[0], rtol=0, atol=0)
    np.testing.assert_array_equal(fresh_space.sample(), expected_draws[1])
    assert not restored._alg.replay_buffer._n_step_transitions
    assert restored._alg.replay_buffer._boundary_next_observations
    for key in ["actor_optimizer_state_dict", "critic_optimizer_state_dict",
                "temperature_optimizer_state_dict", "actor_scheduler_state_dict",
                "critic_scheduler_state_dict", "temperature_scheduler_state_dict",
                "grad_scaler_state_dict", "update_step"]:
        _tree_equal(restored._alg.save()[key], saved_algorithm[key])
    restored.update()
    assert restored._alg.update_step == 3
    assert (checkpoint / "stop.request").read_text() == "full parent\n"


def _lifecycle_config(overrides=()):
    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"),
                               version_base=None):
        return compose(config_name="flashdexretarget", overrides=[
            "commands.motion_file=synthetic-not-read.pt", "num_envs=2", "device=cpu", *overrides,
        ])


def _training_entry():
    path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("producer_entry_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configured_run_root_routes_outputs_and_rejects_existing_content(monkeypatch, tmp_path):
    import common.logger as logger
    import runner.off_policy_runner as runner_module
    import simulator.runtime as runtime_module

    default = _lifecycle_config()
    assert default.run_dir.startswith("logs/flashsac/") and "${" not in default.run_dir
    output = tmp_path / "outputs"
    output.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = _lifecycle_config([f"run_dir={output}"])
    paths = []
    monkeypatch.setattr(logger.wandb, "run", None)
    monkeypatch.setattr(logger, "_init_wandb", lambda _cfg, path: paths.append(Path(path)))
    logger.setup_logger(config.logger, config.runner)
    assert paths == [output] and Path(config.runner.log_dir) == output
    assert config.logger.wandb.run_name == "flashdexretarget"

    flat = SimpleNamespace(reset=lambda: (np.zeros((2, 1)), {}),
        action_space=gym.spaces.Box(-1, 1, shape=(2, 1), dtype=np.float32), current_aux=lambda: None)
    monkeypatch.setattr(runner_module.FlatObsVecEnv, "from_env", lambda *_args, **_kwargs: flat)

    def save(path):
        target = Path(path)
        target.mkdir()
        (target / "network.json").write_text("{}")

    monkeypatch.setattr(runner_module, "build_agent", lambda *_args, **_kwargs: SimpleNamespace(save=save))
    physical = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: None))
    runner = OffPolicyRunner(SimpleNamespace(unwrapped=physical), config.runner, config.eval,
                             eval_env=SimpleNamespace())
    assert runner._save_path_base == output and runner._success_dir == output / "success"
    assert runner._stop_request_path == output / "stop.request"
    runner._finish()
    assert (output / "final" / "network.json").is_file() and not (elsewhere / "logs").exists()

    entry = _training_entry()
    monkeypatch.setattr(runtime_module, "launch_runtime",
                        lambda _cfg: pytest.fail("stale output must fail before runtime boot"))
    for name in ("stop.request", "config.yaml"):
        stale = tmp_path / name.replace(".", "-")
        stale.mkdir()
        original = stale / name
        original.write_text("preserve old output\n")
        with pytest.raises(FileExistsError, match="new or empty"):
            entry.run(SimpleNamespace(config_path="config", config_name="flashdexretarget", overrides=[
                "commands.motion_file=synthetic-not-read.pt", f"run_dir={stale}",
            ]))
        assert original.read_text() == "preserve old output\n"


@pytest.mark.parametrize("failure", [None, "create_envs", "learn", "train_close", "logger_close", "system_exit"])
def test_training_entry_cleanup_preserves_success_and_failure_status(monkeypatch, capsys, tmp_path, failure):
    import common.logger as logger
    import common.utils as utils
    import envs
    import runner as runner_package
    import simulator.isaacsim.runtime as isaac_runtime
    import simulator.runtime as runtime_module

    events = []
    runtime = isaac_runtime.IsaacRuntime.__new__(isaac_runtime.IsaacRuntime)
    monkeypatch.setattr(isaac_runtime.os, "_exit", lambda code: events.append(("runtime_exit", code)))

    def launch(cfg):
        events.append("runtime_launch")
        assert Path(cfg.run_dir).is_absolute() and (Path(cfg.run_dir) / "config.yaml").is_file()
        return runtime

    def close(name):
        events.append(name)
        if failure == name:
            raise RuntimeError(f"actual {name} failure")

    def create_envs(_cfg, device):
        events.append("create_envs")
        if failure == "create_envs":
            raise RuntimeError("actual create_envs failure")
        return SimpleNamespace(close=lambda: close("train_close")), SimpleNamespace(close=lambda: close("eval_close"))

    def learn(**_kwargs):
        events.append("learn")
        if failure == "learn":
            raise RuntimeError("actual learn failure")
        if failure == "system_exit":
            raise SystemExit(7)

    monkeypatch.setattr(runtime_module, "launch_runtime", launch)
    monkeypatch.setattr(envs, "create_envs", create_envs)
    monkeypatch.setattr(logger, "setup_logger", lambda *_args: events.append("logger_setup"))
    monkeypatch.setattr(logger, "close_logger", lambda: close("logger_close"))
    monkeypatch.setattr(utils, "set_seed", lambda *_args: None)
    monkeypatch.setattr(utils, "configure_torch_backends", lambda: None)
    monkeypatch.setattr(runner_package, "create_runner", lambda *_args, **_kwargs: SimpleNamespace(learn=learn))
    entry = _training_entry()
    args = SimpleNamespace(config_path="config", config_name="flashdexretarget", overrides=[
        "commands.motion_file=synthetic-not-read.pt", f"run_dir={tmp_path / 'run'}",
    ])
    if failure == "system_exit":
        with pytest.raises(SystemExit) as caught:
            entry.run(args)
        assert caught.value.code == 7
    elif failure is not None:
        with pytest.raises(RuntimeError, match=f"actual {failure} failure"):
            entry.run(args)
        assert f"actual {failure} failure" in capsys.readouterr().err
    else:
        entry.run(args)
    assert events[0] == "runtime_launch"
    expected_status = 7 if failure == "system_exit" else int(failure is not None)
    assert events[-1] == ("runtime_exit", expected_status)
    assert events[-2] == "logger_close"
    if failure != "create_envs":
        assert events[-4:-2] == ["train_close", "eval_close"]


def _frozen_policy_fixture(*, encoded_window=False):
    """Small real registered actor, including the stock side/trajectory contracts."""
    base = _agent()
    cfg = base._cfg
    del base
    encoded = [["future_traj", 12]] if encoded_window else []
    width = 6 + sum(dim for _, dim in encoded)
    if encoded_window:
        cfg = replace(cfg, traj_encoder={
            "enable": True, "aggregation": "maxpool", "n_future": 3, "n_sides": 2,
            "latent_dim": 4, "hidden_dim": 8, "num_layers": 1,
        })
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([-2., -.5, .2, -3.], dtype=np.float32),
        high=np.array([1., 2., 1.5, -1.], dtype=np.float32), dtype=np.float32,
    )
    info = {
        "actor_observation_size": (6,), "encoded_obs_dims": encoded,
        "obs_terms": [["actor", "state", 6]] + [[name, "window", dim] for name, dim in encoded],
        "reward_heads": ["right", "left"],
        "action_sides": {"right": [2, 0], "left": [3, 1]},
    }
    agent = FlashSACAgent(observation_space, action_space, info, cfg)
    env = SimpleNamespace(share_obs=True, single_observation_space=observation_space,
                          single_action_space=action_space)
    config = OmegaConf.create({"class_name": "agents.flashsac.FlashSACAgent", **asdict(cfg)})
    return agent, env, info, config


@pytest.mark.parametrize("encoded_window", [False, True])
def test_policy_only_load_matches_normal_actor_and_statistics_without_training_owners(
    tmp_path, monkeypatch, encoded_window,
):
    import agents.flashsac as owner
    from agents import load_policy

    agent, env, info, cfg = _frozen_policy_fixture(encoded_window=encoded_window)
    width = env.single_observation_space.shape[-1]
    observations = torch.linspace(-.7, .8, 5 * width).reshape(5, width)
    # Populate the actual actor's running statistics; a fresh unnormalized actor
    # cannot accidentally match the saved policy on these observations.
    agent._alg.actor.get_mean_and_std(
        agent._alg.actor.flatten_obs(agent._layout.actor_input(observations)), training=True,
    )
    expected = agent.act(observations, deterministic=True)
    checkpoint = tmp_path / "full"
    agent.save(str(checkpoint))
    saved_actor = {name: tensor.clone() for name, tensor in agent._alg.actor.state_dict().items()}
    cfg.buffer_compact_obs = True  # live inference must not allocate replay or a compactor

    def forbidden(*_args, **_kwargs):
        pytest.fail("Policy-only load constructed a training/replay owner")

    for name in ("build_flashsac", "MultiHeadCritic", "ReplayBuffer", "MultiHeadFlashSAC"):
        monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    original_load, loads = torch.load, []

    def load(path, **kwargs):
        loads.append((Path(path).name, kwargs))
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    policy = load_policy(cfg, env, info, "motion", str(checkpoint))
    np.testing.assert_array_equal(policy.act(observations, deterministic=True), expected)
    assert loads == [("flashsac.pt", {"map_location": "cpu", "mmap": True, "weights_only": True})]
    for name, value in policy.actor.state_dict().items():
        torch.testing.assert_close(value, saved_actor[name], rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in policy.actor.parameters())
    assert not policy.checkpoint_authority["replay_loaded"]
    assert not policy.checkpoint_authority["critic_constructed"]
    with pytest.raises(ValueError, match="deterministic"):
        policy.act(observations, deterministic=False)


def test_policy_only_load_auto_compile_matches_eager_actor(tmp_path, monkeypatch):
    from rsl_rl_flashsac.utils import resolve_compile_mode

    from agents import load_policy

    agent, env, info, cfg = _frozen_policy_fixture(encoded_window=True)
    width = env.single_observation_space.shape[-1]
    observations = torch.linspace(-.7, .8, 5 * width).reshape(5, width)
    agent._alg.actor.get_mean_and_std(
        agent._alg.actor.flatten_obs(agent._layout.actor_input(observations)), training=True,
    )
    expected = agent.act(observations, deterministic=True)
    agent.save(str(tmp_path))
    cfg.use_compile, cfg.compile_mode = True, "auto"
    original_compile, compiled_modes = torch.compile, []

    def compile_actor(function, *, mode):
        compiled_modes.append(mode)
        return original_compile(function, mode=mode)

    # Record the actual supported mode while retaining real Inductor compilation
    # and execution. An identity/eager compiler replacement would miss this failure.
    monkeypatch.setattr(torch, "compile", compile_actor)
    policy = load_policy(cfg, env, info, "motion", str(tmp_path))
    assert cfg.use_compile and cfg.compile_mode == "auto"
    assert compiled_modes == [resolve_compile_mode("auto")]
    assert compiled_modes[0] != "auto"
    before = {name: value.clone() for name, value in policy.actor.state_dict().items()}
    for _ in range(2):
        actual = policy.act(observations, deterministic=True)
        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))
    for name, value in policy.actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_policy_only_load_rejects_equal_width_layout_change_before_model_read(tmp_path, monkeypatch):
    from agents import load_policy

    agent, env, info, cfg = _frozen_policy_fixture(encoded_window=True)
    agent.save(str(tmp_path))
    incompatible = {**info, "encoded_obs_dims": [["different_window", 12]]}
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("layout must be checked first"))
    with pytest.raises(ValueError, match="encoded obs"):
        load_policy(cfg, env, incompatible, "motion", str(tmp_path))
    incompatible = {**info, "obs_terms": [["actor", "a", 3], ["actor", "b", 3],
                                         ["future_traj", "window", 12]]}
    with pytest.raises(ValueError, match="obs layout differs"):
        load_policy(cfg, env, incompatible, "motion", str(tmp_path))


def test_policy_only_load_rejects_incomplete_saved_actor(tmp_path):
    from agents import load_policy

    agent, env, info, cfg = _frozen_policy_fixture()
    agent.save(str(tmp_path))
    model = torch.load(tmp_path / "flashsac.pt", weights_only=True)
    model["actor_state_dict"].pop(next(iter(model["actor_state_dict"])))
    torch.save(model, tmp_path / "flashsac.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_policy(cfg, env, info, "motion", str(tmp_path))


@pytest.mark.parametrize("recording", [False, True])
def test_frozen_evaluation_entry_routes_public_policy_and_ordinary_evaluator(tmp_path, monkeypatch, recording):
    import agents
    import common.utils as utils
    import envs
    import simulator.runtime as runtime_owner
    from envs.vec_env import FlatObsVecEnv

    evaluation_owner = importlib.import_module("evaluation.evaluate")

    source = Path(__file__).resolve().parents[1] / "evaluate.py"
    spec = importlib.util.spec_from_file_location("frozen_evaluation_entry", source)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    events = []
    checkpoint = tmp_path / "full"
    checkpoint.mkdir()
    (checkpoint / "resume_state.json").write_text(json.dumps({"env_step": 39_290_880}))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run_dir = tmp_path / "run"
    cfg = OmegaConf.create({
        "run_dir": str(run_dir), "seed": 42, "device": "cpu", "headless": True,
        "runner": {"share_actor_critic_obs": True,
                   "agent": {"class_name": "agents.flashsac.FlashSACAgent"}},
        "eval": {"command_name": "native_reference", "num_per_motion": 1,
                 "env": {"disable_object_pin": True},
                 "command": {"sampling_mode": "start", "noise_to_initial_level": 0., "start_frame": 0}},
    })
    OmegaConf.save(cfg, config_dir / "eval.yaml")
    command = SimpleNamespace(cfg=SimpleNamespace(object=SimpleNamespace(pin_objects=True)))
    def get_command(name):
        assert name == "native_reference"
        return command

    raw = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=get_command), num_envs=3,
        eval_num_motions=3, eval_num_batches=1, physics_dt=1 / 240, step_dt=1 / 60,
    )
    train = SimpleNamespace(unwrapped=raw, close=lambda: events.append("train_close"))

    def enter_eval():
        events.append("ordinary_enter_eval")
        command.cfg.object.pin_objects = False

    evaluated = SimpleNamespace(unwrapped=raw, enter_eval=enter_eval,
                                close=lambda: events.append("eval_close"))
    flat = SimpleNamespace(single_observation_space=SimpleNamespace(shape=(6,)),
                           single_action_space=SimpleNamespace(shape=(4,)),
                           reset=lambda: (events.append("metadata_reset"), {"obs_terms": []}))
    policy = SimpleNamespace(checkpoint_authority={"checkpoint_directory": str(checkpoint),
                                                 "replay_loaded": False})
    monkeypatch.setattr(runtime_owner, "launch_runtime", lambda _cfg: SimpleNamespace(
        close=lambda *, exit_code: events.append(("runtime_close", exit_code))))
    monkeypatch.setattr(utils, "set_seed", lambda seed: events.append(("seed", seed)))
    monkeypatch.setattr(utils, "configure_torch_backends", lambda: None)
    monkeypatch.setattr(envs, "create_envs", lambda _cfg, *, device: (train, evaluated))
    def make_flat(actual_env, *, command: object, share_obs, reward_heads):
        assert actual_env is raw and command is get_command("native_reference")
        assert share_obs is True and reward_heads is None
        return flat

    monkeypatch.setattr(FlatObsVecEnv, "from_env", make_flat)

    def load(_cfg, actual_env, info, name, path):
        assert actual_env is flat and info == {"obs_terms": []} and name == "native_reference"
        assert path == str(checkpoint)
        events.append("public_load_policy")
        return policy

    recording_dir = run_dir / "recordings" if recording else None
    frozen_authority = (None if recording_dir is None else {
        "checkpoint": str(checkpoint), "source_revision": "selected-executable-fixture",
        "training_env_step": 39_290_880, "policy": policy.checkpoint_authority,
    })

    def evaluate(actual_policy, actual_env, _cfg, device, **kwargs):
        assert actual_policy is policy and actual_env is evaluated and device == "cpu"
        assert kwargs == {"ever_state": None, "success_dir": None, "env_step": 39_290_880,
                          "outcome_dir": run_dir / "evaluation", "recording_dir": recording_dir,
                          "frozen_authority": frozen_authority}
        assert not command.cfg.object.pin_objects
        events.append("ordinary_evaluate")
        return {"eval/SR": 0.}

    monkeypatch.setattr(agents, "load_policy", load)
    monkeypatch.setattr(evaluation_owner, "evaluate", evaluate)
    args = argparse.Namespace(config_path=str(config_dir), config_name="eval", overrides=[],
                              checkpoint=str(checkpoint), recording_dir=recording_dir)
    monkeypatch.delenv("FLASH_SOURCE_REF", raising=False)
    if recording:
        with pytest.raises(ValueError, match="requires FLASH_SOURCE_REF"):
            entry.run(args)
        assert not events and not run_dir.exists()
        monkeypatch.setenv("FLASH_SOURCE_REF", "selected-executable-fixture")
    entry.run(args)
    assert events == [("seed", 42), "metadata_reset", "public_load_policy", "ordinary_enter_eval",
                      "ordinary_evaluate", "train_close", "eval_close", ("runtime_close", 0)]
    result = json.loads((run_dir / "evaluation-entry.json").read_text())
    assert result["status"] == "COMPLETED_CURRENT_POLICY_EVALUATION"
    assert result["training_env_step"] == 39_290_880 and result["pin_during_eval"] is False
    assert result["wrench_measurement"].startswith("UNKNOWN")
    assert result["factual_recording_dir"] == (None if recording_dir is None else str(recording_dir))
