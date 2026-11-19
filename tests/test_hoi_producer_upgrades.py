"""Unit tests for 6-hand native HOI producer configs, opposing-digit contact credit, and assist rules."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import torch
from omegaconf import OmegaConf, open_dict
from test_native_contract import _scene, write_native_scene_manifest

from common.native_reference import bind_prepared_native_reference
from envs.rewards.native import NativeMotionTrackingRewards
from envs.scene.native import bind_native_scene_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

SIX_HAND_ROBOT_CONFIGS = (
    "inspire_rh56f1_bimanual",
    "inspire_rh56dftp_bimanual",
    "wuji_hand_bimanual",
    "sharpa_wave_native_bimanual",
    "unitree_dex3_1_bimanual",
    "wonik_allegro_v5_bimanual",
)


def _native_hoi_config(motion_file, robot_cfg="inspire_rh56f1_bimanual"):
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return hydra.compose(
            config_name="native_hoi_surface",
            overrides=[
                f"envs/robot@robot={robot_cfg}",
                "envs/simulator@simulator=newton",
                f"commands.motion_file={motion_file}",
            ],
        )


@pytest.mark.parametrize("robot_cfg", SIX_HAND_ROBOT_CONFIGS)
def test_six_hand_hoi_surface_hydra_configs_compose(robot_cfg: str, tmp_path: Path) -> None:
    _, mapping = _scene(tmp_path)
    cfg = _native_hoi_config(tmp_path / "motion.pt", robot_cfg)
    for hand in mapping["hands"]:
        side = hand["side"]
        hand["tip_site_names"] = [f"{side}_{digit}_pad" for digit in cfg.robot.body_names.fingers]
    write_native_scene_manifest(tmp_path, mapping)
    bind_prepared_native_reference(cfg)
    OmegaConf.resolve(cfg)

    assert cfg.robot.xml_path == mapping["scene_xml_path"]
    assert cfg.actions.native_scene == cfg.robot.native_scene
    assert cfg.native_recording_binding.fixture == "native scene"
    recorder = hydra.utils.instantiate(cfg.eval.callbacks[-1])
    assert recorder.binding["scene_identity"]["native_scene"]["scene_xml_path"] == cfg.robot.xml_path
    assert isinstance(recorder.binding["scene_identity"]["native_scene"]["hands"], list)
    assert not cfg.sim.get("native_assets")
    for hand in mapping["hands"]:
        assert list(cfg.robot.site_names.tip[hand["side"]]) == hand["tip_site_names"]
        assert list(cfg.robot.site_names.contact[hand["side"]]) == hand["tip_site_names"]

    assert cfg.rewards.reward_weights.native_source_hand_part_contact > 0.0
    assert cfg.rewards.reward_params.native_source_hand_part_contact.source_digit_distance_m == 0.042
    assert cfg.rewards.reward_params.native_source_hand_part_proximity.source_digit_distance_m == 0.042
    assert cfg.rewards.reward_params.native_part_surface_displacement.part_reduce == "worst"
    assert 0.0 < float(cfg.commands.object.xfrc.contact_scale) < 1.0
    assert cfg.commands.object.perturb.enabled is True
    assert cfg.commands.object.perturb.max_angular_acceleration_rad_s2 == 1000.
    assert cfg.commands.object.perturb.require_assist_decayed is True
    assert cfg.actions.translation_residual_m == 0.04
    assert cfg.actions.rotation_residual_rad == 0.35
    assert cfg.curriculum.xfrc_curriculum_adaptive.params.upper_ratios.fr == 0.78
    assert cfg.rewards.reward_weights.native_contact_overload < 0.0
    assert cfg.eval.stop_when_all_solved is False
    if robot_cfg == "unitree_dex3_1_bimanual":
        assert list(cfg.robot.body_names.fingers) == ["thumb", "index", "middle"]
        left, right = (tuple(joint.removeprefix(side + "_") for joint in cfg.robot.joint_names.finger
                             if joint.startswith(side + "_")) for side in ("left", "right"))
        assert left == right
    elif robot_cfg == "wonik_allegro_v5_bimanual":
        assert list(cfg.robot.body_names.fingers) == ["thumb", "index", "middle", "ring"]
    elif robot_cfg == "wuji_hand_bimanual":
        body_by_role = {role: names for names, role in cfg.robot.body_mapping.all}
        for finger, role in ((3, "middle"), (4, "ring"), (5, "pinky")):
            assert body_by_role[role + "_proximal"] == {
                side: f"{side}_finger{finger}_link1" for side in ("left", "right")
            }


def test_native_reference_requires_an_embedded_binding(tmp_path: Path) -> None:
    import json

    _, mapping = _scene(tmp_path)
    motion_file = tmp_path / "motion.pt"
    motion_file.with_name("native-reference.json").write_text(json.dumps({"native_scene": mapping}))
    with pytest.raises(ValueError, match="requires a recording_binding mapping"):
        bind_prepared_native_reference(_native_hoi_config(motion_file))


def test_bind_native_scene_config_admits_contact_scale_and_post_decay_perturb(tmp_path: Path) -> None:
    _, mapping = _scene(tmp_path)
    cfg = _native_hoi_config(write_native_scene_manifest(tmp_path, mapping))
    bind_prepared_native_reference(cfg)
    layout = bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=(0.0, 0.0, -9.81))
    assert layout.objects[0].root_body_name == "base"

    with open_dict(cfg.commands.object):
        cfg.commands.object.perturb.require_assist_decayed = False
    with pytest.raises(ValueError, match="require_assist_decayed"):
        bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=(0.0, 0.0, -9.81))


def test_hoi_collection_treats_outside_pose_tolerance_as_a_failed_take(tmp_path: Path) -> None:
    from test_native_task_contract import _command

    from envs.terminations.native import NativeMotionTrackingTerminations

    cfg = _native_hoi_config(tmp_path / "motion.pt")
    bounds = OmegaConf.to_container(cfg.terminations.native_reference_diverged.params, resolve=True)
    command, env, _ = _command(rigid=True)
    object_data = next(iter(command.native_objects.values())).data
    diverged = NativeMotionTrackingTerminations.native_reference_diverged
    object_data.body_link_pos_w[..., 2] = -0.025
    assert not diverged(env, "motion", **bounds).item()
    object_data.body_link_pos_w[..., 2] = -0.075
    assert diverged(env, "motion", **bounds).item()
    object_data.body_link_pos_w.zero_()
    for degrees, failed in ((15, False), (25, True)):
        half_angle = math.radians(degrees / 2)
        object_data.body_link_quat_w[:] = torch.tensor([math.cos(half_angle), 0, 0, math.sin(half_angle)])
        assert diverged(env, "motion", **bounds).item() == failed
    part_pose_reward = cfg.rewards.reward_params.native_object_part_pose
    assert part_pose_reward.position_scale_m == bounds["part_position_m"]
    command.motion_lib.native_object_part_pose_support[:] = False
    assert not diverged(env, "motion", **bounds).item()
    assert bounds["part_rotation_rad"] == cfg.native_tracking_tolerances.part_rotation_error_rad


def test_opposing_thumb_finger_contact_credit_suppresses_single_digit_brushing() -> None:
    # World 0: thumb (5mm) + index (8mm) both near object -> full 1.0x credit
    # World 1: index (5mm) near object, thumb (65mm) far away -> scaled down by non_opposed_scale (0.15x)
    intent_facts = {
        "metadata": {"hand_sides": ("left", "right"), "part_body_names": ("obj_body",)},
        "active": torch.tensor([[[True], [False]], [[True], [False]]]),
        "known": torch.ones((2, 2, 1), dtype=torch.bool),
        "execution_valid": torch.tensor([True, True]),
        "reference_motion_ids": torch.zeros(2, dtype=torch.long),
        "reference_motion_steps": torch.ones(2, dtype=torch.long),
        "reference_time_ns": torch.full((2,), 20_000_000, dtype=torch.long),
    }
    geom_facts = {
        "metadata": {
            "hand_sides": ("left", "right"),
            "hand_offsets": (0, 3, 6),
            "source_point_names": ("thumb_tip", "index_tip", "middle_tip", "thumb_tip", "index_tip", "middle_tip"),
            "part_body_names": ("obj_body",),
        },
        "live_part_distance_m": torch.tensor([
            [[0.005], [0.008], [0.040], [0.050], [0.050], [0.050]],
            [[0.065], [0.005], [0.009], [0.050], [0.050], [0.050]],
        ]),
        "live_part_query_support": torch.ones((2, 6, 1), dtype=torch.bool),
        "desired_part_distance_m": torch.full((2, 6, 1), .01),
        "desired_part_query_support": torch.ones((2, 6, 1), dtype=torch.bool),
        **{key: intent_facts[key] for key in ("execution_valid", "reference_motion_ids",
                                           "reference_motion_steps", "reference_time_ns")},
    }
    interval_facts = {
        "addresses": {
            "left/native_object/object": {
                "hand_side": "left",
                "hand_body_name": "left_wrist",
                "part_entity_name": "native_object",
                "part_body_name": "obj_body",
            },
            "right/native_object/object": {
                "hand_side": "right",
                "hand_body_name": "right_wrist",
                "part_entity_name": "native_object",
                "part_body_name": "obj_body",
            },
        },
        "sample_end_time_s": torch.tensor([[0.02], [0.02]], dtype=torch.float64),
        "sample_begin_time_s": torch.tensor([[0.00], [0.00]], dtype=torch.float64),
        "sample_recorded": torch.ones((2, 1), dtype=torch.bool),
        "clock_support": torch.ones((2, 1), dtype=torch.bool),
        "interval_complete": torch.tensor([True, True]),
        "measurement_support": torch.ones((2, 1), dtype=torch.bool),
        "pairs": {
            "left/native_object/object": {
                "normal_force_n": torch.tensor([[3.0], [3.0]]),
                "force_bearing_contact_present": torch.tensor([[True], [True]]),
                "force_bearing_distance_max_m": torch.tensor([[0.001], [0.001]]),
            },
            "right/native_object/object": {
                "normal_force_n": torch.tensor([[0.0], [0.0]]),
                "force_bearing_contact_present": torch.tensor([[False], [False]]),
                "force_bearing_distance_max_m": torch.tensor([[0.0], [0.0]]),
            },
        },
    }
    cmd = SimpleNamespace(
        source_hand_part_intent_facts=lambda: intent_facts,
        source_tip_geometry_facts=lambda: geom_facts,
        native_layout=SimpleNamespace(
            hands=[
                SimpleNamespace(side="left", root_body_name="left_wrist"),
                SimpleNamespace(side="right", root_body_name="right_wrist"),
            ],
            part_bindings=[(SimpleNamespace(entity_name="native_object"), SimpleNamespace(name="object", body_name="obj_body"))],
        ),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        command_manager=SimpleNamespace(get_term=lambda _n: cmd),
        scene=SimpleNamespace(
            hand_part_contact_interval_facts=lambda: interval_facts,
            physics_time_s=lambda: torch.tensor([0.02, 0.02], dtype=torch.float64),
        ),
    )
    rew = NativeMotionTrackingRewards.native_source_hand_part_contact(
        env,
        "motion",
        force_scale_n=1.5,
        separation_bound_m=0.02,
        source_digit_distance_m=0.042,
        digit_contact_distance_m=0.030,
        unmatched_digit_scale=0.15,
    )
    assert float(rew[0]) > 0.80
    assert pytest.approx(float(rew[1]), rel=1e-5) == 0.15 * float(rew[0])
