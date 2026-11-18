"""Unit tests for 6-hand native HOI producer configs, opposing-digit contact credit, and assist rules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import torch
from omegaconf import OmegaConf, open_dict

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


@pytest.mark.parametrize("robot_cfg", SIX_HAND_ROBOT_CONFIGS)
def test_six_hand_hoi_surface_hydra_configs_compose(robot_cfg: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FDR_NATIVE_SCENE_XML", "/tmp/dummy_scene.xml")
    monkeypatch.setenv("FDR_NATIVE_OBJECT_ROOT_BODY", "oakink_obj_body")
    monkeypatch.setenv("FDR_NATIVE_OBJECT_ROOT_JOINT", "oakink_obj_free")
    monkeypatch.setenv("FDR_NATIVE_OBJECT_MESH", "/tmp/dummy.obj")
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(
            config_name="native_hoi_surface",
            overrides=[
                f"envs/robot@robot={robot_cfg}",
                "envs/simulator@simulator=newton",
                "envs/native_object@native_object=rigid",
                "commands.motion_file=/tmp/dummy_motion.pt",
                "native_recording_binding={}",
            ],
        )
    assert cfg.rewards.reward_weights.native_source_hand_part_contact > 0.0
    assert cfg.rewards.reward_params.native_source_hand_part_contact.require_thumb_opposition is True
    assert 0.0 < float(cfg.commands.object.xfrc.contact_scale) < 1.0
    assert cfg.commands.object.perturb.enabled is True
    assert cfg.commands.object.perturb.require_assist_decayed is True
    hands = cfg.robot.native_scene.hands
    assert len(hands) == 2 and {h.side for h in hands} == {"left", "right"}
    # Strict anatomy check for 3-finger Dex3 and 4-finger Wonik
    if robot_cfg == "unitree_dex3_1_bimanual":
        assert list(cfg.robot.body_names.fingers) == ["thumb", "index", "middle"]
        assert len(cfg.robot.site_names.tip.left) == 3
    elif robot_cfg == "wonik_allegro_v5_bimanual":
        assert list(cfg.robot.body_names.fingers) == ["thumb", "index", "middle", "ring"]
        assert len(cfg.robot.site_names.tip.left) == 4


def test_bind_native_scene_config_admits_contact_scale_and_post_decay_perturb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FDR_NATIVE_SCENE_XML", "/tmp/dummy_scene.xml")
    monkeypatch.setenv("FDR_NATIVE_OBJECT_ROOT_BODY", "oakink_obj_body")
    monkeypatch.setenv("FDR_NATIVE_OBJECT_ROOT_JOINT", "oakink_obj_free")
    monkeypatch.setenv("FDR_NATIVE_OBJECT_MESH", "/tmp/dummy.obj")
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(
            config_name="native_hoi_surface",
            overrides=[
                "envs/robot@robot=inspire_rh56f1_bimanual",
                "envs/simulator@simulator=newton",
                "envs/native_object@native_object=rigid",
                "commands.motion_file=/tmp/dummy_motion.pt",
                "native_recording_binding={}",
            ],
        )
    layout = bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=(0.0, 0.0, -9.81))
    assert layout.objects[0].root_body_name == "oakink_obj_body"

    with open_dict(cfg.commands.object):
        cfg.commands.object.perturb.require_assist_decayed = False
    with pytest.raises(ValueError, match="require_assist_decayed"):
        bind_native_scene_config(cfg, constructor_gravity_world_m_per_s2=(0.0, 0.0, -9.81))


def test_opposing_thumb_finger_contact_credit_suppresses_single_digit_brushing() -> None:
    # World 0: thumb (5mm) + index (8mm) both near object -> full 1.0x credit
    # World 1: index (5mm) near object, thumb (65mm) far away -> scaled down by non_opposed_scale (0.15x)
    intent_facts = {
        "metadata": {"hand_sides": ("left", "right"), "part_body_names": ("obj_body",)},
        "active": torch.tensor([[[True], [False]], [[True], [False]]]),
        "known": torch.ones((2, 2, 1), dtype=torch.bool),
        "execution_valid": torch.tensor([True, True]),
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
        "achieved_position_support": torch.ones((2, 6), dtype=torch.bool),
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
        require_thumb_opposition=True,
        opposition_distance_m=0.030,
        non_opposed_scale=0.15,
    )
    assert float(rew[0]) > 0.80
    assert pytest.approx(float(rew[1]), rel=1e-5) == 0.15 * float(rew[0])
