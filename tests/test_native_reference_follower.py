"""Analytic actions use the real native affine/filter owner; no plant or model."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import simulator.isaacsim  # noqa: F401

# isort: split
from envs.actions.native import NativeResidualAction
from evaluation.evaluate import validate_recording_authority
from evaluation.policies import NativeReferenceFollower


def _case(widths):
    action = object.__new__(NativeResidualAction)
    action.hand_sides = tuple(side for side, _ in widths)
    action.finger_actuator_names = tuple(f"{side}_a{i}" for side, width in widths for i in range(width))
    action.finger_joint_names = tuple(f"{side}_q{i}" for side, width in widths for i in range(width))
    roots = 6 * len(widths)
    action.action_names = tuple(f"root{i}" for i in range(roots)) + action.finger_actuator_names
    action.finger_command_names = action.finger_joint_names
    action._finger_slice = slice(roots, len(action.action_names))
    action._env = SimpleNamespace(num_envs=2, device="cpu")
    width = len(action.finger_joint_names)
    action._lower = torch.arange(width).float() * -.1 - .4
    action._upper = torch.arange(width).float() * .2 + .6
    action.cfg = SimpleNamespace(action_scale=2., action_offset=.3, finger_residual_rad=None,
                                 cutoff_hz={"finger": None}, wrist_target_twist="source_only")
    action._raw_actions = torch.zeros(2, action.action_dim)
    action._processed_actions = torch.zeros_like(action._raw_actions)
    action._filter_state = torch.zeros_like(action._raw_actions)
    action._filter_pending = torch.ones(2, dtype=torch.bool)
    action._source_contact_guard = None
    action._alpha = torch.ones(action.action_dim)
    action._clip = torch.tensor([[-1., 1.]]).repeat(action.action_dim, 1)
    order = list(reversed(range(width)))
    target = action._lower + .7 * (action._upper - action._lower)
    reference = dict(joint_names=tuple(action.finger_joint_names[i] for i in order),
        position=target[order].repeat(2, 1), position_support=torch.ones(2, width, dtype=torch.bool))
    reference["position_support"][1, 0] = False
    command = SimpleNamespace(command_reference=lambda: reference, next_ref_joint_pos=torch.full((2, width), 999.))
    env = SimpleNamespace(num_envs=2, command_manager=SimpleNamespace(get_term=lambda _: command),
                          action_manager=SimpleNamespace(get_term=lambda _: action, total_action_dim=action.action_dim))
    return env, action, reference, target


@pytest.mark.parametrize("widths", ((("left", 1), ("right", 3)), (("right", 5),)))
def test_native_reference_follower_uses_current_named_commands_and_zero_root_affine(widths):
    env, action, reference, target = _case(widths)
    policy = NativeReferenceFollower(env=env, command_name="configured-motion", action_name="configured-action",
                                     method_id="analytic-fixture")
    raw = policy.act(None, deterministic=True)
    NativeResidualAction.process_actions(action, torch.from_numpy(raw))
    columns = [action.action_names.index(name) for name in action.finger_actuator_names]
    roots = len(action.action_names) - len(columns)
    torch.testing.assert_close(action._processed_actions[:, :roots], torch.zeros(2, roots))
    torch.testing.assert_close(action._processed_actions[0, columns], torch.full_like(target, .4))
    # Missing source command never receives support from a neighbouring command.
    assert action._processed_actions[1, columns[-1]] == 0
    reconstructed = action._lower + (action._processed_actions[0, columns] + 1) * .5 * (action._upper - action._lower)
    torch.testing.assert_close(reconstructed, target)
    reference["position"][:] = action._lower.flip(0)
    assert not np.array_equal(policy.act(None), raw)  # Read the next actual packet, not a cached target.
    assert policy.acting_authority["checkpoint"] is None
    assert policy.acting_authority["has_learned_parameters"] is False
    validate_recording_authority(dict(source_revision="fixture-source", policy=policy.acting_authority))


@pytest.mark.parametrize("changed", ("residual", "filter", "affine"))
def test_native_reference_follower_rejects_a_different_action_contract(changed):
    env, action, _reference, _target = _case((("left", 2), ("right", 1)))
    if changed == "residual":
        action.cfg.finger_residual_rad = .2
    elif changed == "filter":
        action.cfg.cutoff_hz["finger"] = 3.
    else:
        action.cfg.action_scale = 0.
    with pytest.raises(ValueError):
        NativeReferenceFollower(env=env, command_name="motion", action_name="native", method_id="fixture")


def test_native_reference_follower_authority_never_accepts_a_fake_checkpoint_free_actor():
    with pytest.raises(ValueError, match="parameterless"):
        validate_recording_authority(dict(source_revision="fixture", policy=dict(kind="analytic", method_id="x",
            has_learned_parameters=True, checkpoint=None)))


def test_native_reference_follower_profile_instantiates_policy_without_a_learner():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        cfg = compose(config_name="native", overrides=["eval=native_reference"])
    env, action, _reference, _target = _case((("right", 3),))
    policy = instantiate(cfg.eval.policy, env=env)
    assert isinstance(policy, NativeReferenceFollower)
    assert policy.act(None).shape == (2, action.action_dim)
    assert policy.acting_authority["checkpoint"] is None
