"""Named native surface geometry, consumed clocks and compact replay; no physics."""

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

# The public manager shim is installed without starting Kit or a backend.
import simulator.isaacsim  # noqa: F401

# isort: split
from mjlab.utils.lab_api.math import matrix_from_quat
from test_native_task_contract import _add_loose_object, _command

from agents.storage.obs_compact import ObsCompactor, flat_obs_layout
from envs.object_points import (
    farthest_point_sample,
    native_part_reference,
    surface_pool,
)
from envs.observations.native import NativeMotionTrackingObs
from envs.rewards.native import NativeMotionTrackingRewards


def _install_part_state(command, state):
    for index, (entity, body) in enumerate(zip(command._part_entities, command._part_body_ids, strict=True)):
        for field, columns in (("body_link_pos_w", slice(0, 3)), ("body_link_quat_w", slice(3, 7)),
                               ("body_link_lin_vel_w", slice(7, 10)), ("body_link_ang_vel_w", slice(10, 13))):
            getattr(entity.data, field)[:, body] = state[:, index, columns]


def _surface_case(tmp_path):
    command, env, calls = _command()
    _add_loose_object(command, env, calls)
    mesh = tmp_path / "surface.obj"
    mesh.write_text("v 0 0 0\nv .1 0 0\nv 0 .2 0\nv 0 0 .3\n"
                    "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    objects, index = [], 0
    for obj in command.native_layout.objects:
        parts = []
        for part in obj.parts:
            parts.append(replace(part, mesh_path=str(mesh), body_from_mesh_position_m=(.04 + .03 * index, -.02, .03),
                body_from_mesh_quat_wxyz=(2**-.5, 0., 0., 2**-.5)))
            index += 1
        objects.append(replace(obj, parts=tuple(parts)))
    command.native_layout = replace(command.native_layout, objects=tuple(objects))
    command._bind_native_reference()
    ml = command.motion_lib
    ml.obj_trans = {}
    ml.length_starts = torch.tensor([0, 3])
    ml._motion_num_frames = torch.tensor([3, 3])
    ml.native_time_ns = torch.tensor([4_000_000_000_000_000_000 + i * 20_000_000 for i in range(3)]
                                   + [5_000_000_000_000_000_000 + i * 20_000_000 for i in range(3)])
    states = torch.zeros(6, 3, 13)
    states[..., 3] = 1.
    states[..., :3] = torch.tensor([[.25, -.1, .4], [.3, .2, .6], [1.1, -.5, .3]])
    states[..., 0] += torch.arange(6)[:, None] * .025
    states[1, 1, 3:7] = torch.tensor([2**-.5, 0., 2**-.5, 0.])
    ml.native_object_part_state_w = states
    ml.native_object_part_pose_support = torch.ones(6, 3, dtype=torch.bool)
    ml.native_object_part_pose_support[4, 1] = False
    env.num_envs = 2
    env.scene.env_origins = torch.tensor([[10., -20., .75], [-5., 8., 1.]])
    for entity in command.native_objects.values():
        for field in ("body_link_pos_w", "body_link_quat_w", "body_link_lin_vel_w", "body_link_ang_vel_w"):
            setattr(entity.data, field, getattr(entity.data, field).expand(2, -1, -1).clone())
    robot = command.robot.data
    robot.body_link_pos_w = torch.tensor([[[.1, .2, .3], [-.2, .5, .15]],
                                          [[-.1, .3, .4], [.25, -.3, .2]]]) + env.scene.env_origins[:, None]
    robot.body_link_quat_w = torch.tensor([[[1., 0., 0., 0.], [2**-.5, 0., 0., 2**-.5]],
                                           [[2**-.5, 2**-.5, 0., 0.], [1., 0., 0., 0.]]])
    robot.body_link_lin_vel_w = torch.zeros(2, 2, 3)
    robot.body_link_ang_vel_w = torch.zeros(2, 2, 3)
    # Actual source hand order is independent of the authored hand list.
    command._side_list, command._root_body_ids = ["right", "left"], [1, 0]
    command.motion_ids, command.motion_steps = torch.tensor([1, 0]), torch.tensor([0, 2])
    live = states[[0, 3]].clone()
    live[..., :3] += env.scene.env_origins[:, None]
    live[1, 2, 3:7] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    _install_part_state(command, live)
    installed = dict(reference_motion_ids=torch.tensor([0, 1]), reference_motion_steps=torch.tensor([0, 0]),
                     installed_valid=torch.ones(2, dtype=torch.bool))
    env.action_manager = SimpleNamespace(_terms={"native": SimpleNamespace(installed_control=installed)})
    return command, env


def _cloud_values(packed, hands=2, parts=3, points=4):
    boundary = hands * parts * points * 3
    return packed[:, :boundary].reshape(-1, hands, parts, points, 3), packed[:, boundary:].reshape(-1, hands, parts)


def _surface_replay(command, env, *, changed_source=None, clip=None, scale=None):
    fields = [("native_object_parts_wrist", NativeMotionTrackingObs.native_object_parts_wrist(env, "motion")),
              ("robot_wrist_pose_w", torch.cat((command.robot_wrist_trans_w - env.scene.env_origins[:, None],
                                               command.robot_wrist_quat_w), -1).flatten(1)),
              ("native_part_point_cloud_wrist", NativeMotionTrackingObs.native_part_point_cloud_wrist(env, "motion", 4, 16)),
              ("native_next_part_point_cloud_delta_wrist",
               NativeMotionTrackingObs.native_next_part_point_cloud_delta_wrist(env, "motion", 4, 16))]
    configs = {name: SimpleNamespace(params=dict(n_points=4, n_obj_verts=16) if "point_cloud" in name else {},
        clip=clip if name == changed_source else None, scale=scale if name == changed_source else None,
        noise=None) for name, _ in fields}
    env.observation_manager = SimpleNamespace(active_terms={"actor": [name for name, _ in fields]},
        group_obs_term_dim={"actor": [(value.shape[-1],) for _, value in fields]},
        get_term_cfg=lambda group, name: configs[name])
    full = torch.cat([value for _, value in fields], -1)
    spans = flat_obs_layout(env, ["actor"])
    return full, ObsCompactor(command, spans, full.shape[-1], "cpu"), spans


def test_native_surface_cache_uses_actual_named_mesh_to_body_transform(tmp_path):
    import trimesh

    command, _env = _surface_case(tmp_path)
    original = trimesh.load
    with patch.object(trimesh, "load", wraps=original) as load:
        points = command.part_surface_points(4, 16)
        assert command.part_surface_points(4, 16) is points
        assert load.call_count == 3
    pool = torch.as_tensor(surface_pool(original(command.native_layout.objects[0].parts[0].mesh_path,
        force="mesh", process=False, skip_materials=True), 16))
    mesh_points = pool[farthest_point_sample(pool, 4)]
    expected = torch.stack((-mesh_points[:, 1], mesh_points[:, 0], mesh_points[:, 2]), -1)
    expected += torch.tensor([.04, -.02, .03])
    expected = expected.expand(3, -1, -1).clone()
    expected[..., 0] += torch.arange(3)[:, None] * .03
    torch.testing.assert_close(points, expected)
    metadata = command.part_surface_geometry_metadata()
    assert len(metadata) == 1 and metadata[0]["hand_sides"] == ["right", "left"]
    assert [(row["entity_name"], row["part_name"], row["body_name"]) for row in metadata[0]["parts"]] == [
        ("object", "body", "object_root"), ("object", "lid", "object_lid"), ("loose_prop", "body", "prop_root")]
    torch.testing.assert_close(torch.tensor([part["points_body_m"] for part in metadata[0]["parts"]]), points)
    assert json.loads(json.dumps(metadata))[0]["parts"][0]["points_body_m"] == points[0].tolist()


def test_native_surface_live_geometry_preserves_physical_size_and_all_actual_parts(tmp_path):
    command, env = _surface_case(tmp_path)
    points = command.part_surface_points(4, 16)
    actual, masks = _cloud_values(NativeMotionTrackingObs.native_part_point_cloud_wrist(env, "motion", 4, 16))
    state = command.sim_object_part_state_w
    world = state[..., None, :3] + torch.einsum("bpij,pnj->bpni", matrix_from_quat(state[..., 3:7]), points)
    expected = torch.einsum("bhij,bhpni->bhpnj", matrix_from_quat(command.robot_wrist_quat_w),
        world[:, None] - command.robot_wrist_trans_w[:, :, None, None])
    torch.testing.assert_close(actual, expected)
    assert masks.bool().all() and actual.shape == (2, 2, 3, 4, 3)
    # Translation/rotation into wrist frames must preserve actual sample spacing.
    torch.testing.assert_close((actual[..., 0, :] - actual[..., 1, :]).norm(dim=-1),
        (points[:, 0] - points[:, 1]).norm(dim=-1).expand(2, 2, -1))


def test_native_surface_named_part_permutation_preserves_geometry(tmp_path):
    command, env = _surface_case(tmp_path)
    original, mask = _cloud_values(NativeMotionTrackingObs.native_part_point_cloud_wrist(env, "motion", 4, 16))
    first, loose = command.native_layout.objects
    command.native_layout = replace(command.native_layout, objects=(loose, replace(first, parts=tuple(reversed(first.parts)))))
    command._part_entities = tuple(command.native_objects[obj.entity_name] for obj, _ in command.native_layout.part_bindings)
    command._part_body_ids = [entity.body_names.index(part.body_name)
        for entity, (_, part) in zip(command._part_entities, command.native_layout.part_bindings, strict=True)]
    del command._native_part_surface_cache
    permuted, permuted_mask = _cloud_values(NativeMotionTrackingObs.native_part_point_cloud_wrist(env, "motion", 4, 16))
    torch.testing.assert_close(permuted, original[:, :, [2, 1, 0]])
    assert torch.equal(permuted_mask, mask[:, :, [2, 1, 0]])
    assert [part["body_name"] for part in command.part_surface_geometry_metadata()[0]["parts"]] == [
        "prop_root", "object_lid", "object_root"]


def test_native_surface_next_query_keeps_int64_clocks_independent_support_and_missing_tail(tmp_path):
    command, env = _surface_case(tmp_path)
    facts = command.part_surface_reference(offset=1)
    assert facts["reference_time_ns"].tolist() == [5_000_000_000_020_000_000, 4_000_000_000_040_000_000]
    assert facts["query_support"].tolist() == [True, False]
    assert facts["pose_support"].tolist() == [[True, False, True], [False, False, False]]
    delta, masks = _cloud_values(NativeMotionTrackingObs.native_next_part_point_cloud_delta_wrist(env, "motion", 4, 16))
    assert torch.equal(masks.bool(), facts["pose_support"][:, None].expand(-1, 2, -1))
    assert not delta[1].any() and not delta[0, :, 1].any()
    points = command.part_surface_points(4, 16)
    live = command.sim_object_part_state_w[0]
    target = facts["state"][0].clone()
    target[:, :3] += env.scene.env_origins[0]
    world_delta = target[:, None, :3] - live[:, None, :3] + torch.einsum("pij,pnj->pni",
        matrix_from_quat(target[:, 3:7]) - matrix_from_quat(live[:, 3:7]), points)
    expected = torch.einsum("hij,pni->hpnj", matrix_from_quat(command.robot_wrist_quat_w[0]), world_delta)
    torch.testing.assert_close(delta[0, :, [0, 2]], expected[:, [0, 2]])


@pytest.mark.parametrize("part_reduce", ["mean", "worst"])
def test_native_surface_reward_uses_consumed_reference_and_supported_part_denominator(tmp_path, part_reduce):
    command, env = _surface_case(tmp_path)
    installed = next(iter(env.action_manager._terms.values())).installed_control
    reference = native_part_reference(command.motion_lib, installed["reference_motion_ids"], installed["reference_motion_steps"])
    current = reference["state"].clone()
    current[..., :3] += env.scene.env_origins[:, None]
    current[0, :, 0] += torch.tensor([.01, .02, 10.])
    command.motion_lib.native_object_part_pose_support[0, 2] = False
    installed["installed_valid"][1] = False
    _install_part_state(command, current)
    values = torch.exp(-10 * torch.tensor([.01, .02]))
    expected = values.mean() if part_reduce == "mean" else values.min()
    kwargs = dict(scale=10., n_points=4, n_obj_verts=16, topk=3, part_reduce=part_reduce)
    reward = NativeMotionTrackingRewards.native_part_surface_displacement(env, "motion", **kwargs)
    torch.testing.assert_close(reward, torch.tensor([expected, 0.]))
    command.motion_steps[:] = 2
    command.motion_ids[:] = command.motion_ids.flip(0)
    torch.testing.assert_close(NativeMotionTrackingRewards.native_part_surface_displacement(env, "motion", **kwargs), reward)
    # Lever-arm normalization does not scale a pure translation error.
    torch.testing.assert_close(NativeMotionTrackingRewards.native_part_surface_displacement(env, "motion", norm_radius=.058, **kwargs), reward)


def test_native_surface_reward_normalizes_only_rotation_and_pools_worst_supported_points(tmp_path):
    command, env = _surface_case(tmp_path)
    installed = next(iter(env.action_manager._terms.values())).installed_control
    command.motion_lib.native_object_part_pose_support[:] = False
    command.motion_lib.native_object_part_pose_support[0, 0] = True
    installed["installed_valid"][1] = False
    current = command.motion_lib.native_object_part_state_w[[0, 3]].clone()
    current[..., :3] += env.scene.env_origins[:, None]
    current[0, 0, 3:7] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    _install_part_state(command, current)
    points = command.part_surface_points(4, 16)[0]
    cloud_before = NativeMotionTrackingObs.native_part_point_cloud_wrist(env, "motion", 4, 16)
    for radius in (0., .058):
        sampled = points if radius == 0 else points * radius / points.norm(dim=-1).max()
        rotated = torch.stack((-sampled[:, 1], sampled[:, 0], sampled[:, 2]), -1)
        worst = (sampled - rotated).norm(dim=-1).topk(3).values.mean()
        expected = math.exp(-10 * max(0., float(worst) - .005))
        reward = NativeMotionTrackingRewards.native_part_surface_displacement(env, "motion", 10., 4, 16,
            topk=3, deadband=.005, norm_radius=radius)
        torch.testing.assert_close(reward, torch.tensor([expected, 0.]))
    torch.testing.assert_close(NativeMotionTrackingObs.native_part_point_cloud_wrist(env, "motion", 4, 16), cloud_before)


def test_native_surface_replay_reconstructs_sampled_rows_without_current_world_reads(tmp_path):
    command, env = _surface_case(tmp_path)
    full, compactor, spans = _surface_replay(command, env)
    compact, aux = compactor.compact(full), compactor.aux_from_command().clone()
    assert compactor.compact_dim == 2 * 3 * 15 + 2 * 7
    assert compactor.full_dim - compactor.compact_dim == 2 * 2 * 3 * (4 * 3 + 1)
    rows = torch.tensor([1, 0, 1])  # Replay count and order differ from the two live worlds.
    for entity in command.native_objects.values():
        entity.data.body_link_pos_w[:] = 1000.
    command.robot.data.body_link_quat_w[:] = torch.tensor([1., 0., 0., 0.])
    command.motion_steps[:] = 0
    command.motion_ids[:] = 0
    reconstructed = compactor.expand(compact[rows], aux[rows])
    torch.testing.assert_close(reconstructed, full[rows])
    assert torch.equal(reconstructed[:, :spans[1].end], full[rows, :spans[1].end])
    for span in spans[2:]:
        assert torch.equal(reconstructed[:, span.end - 6:span.end], full[rows, span.end - 6:span.end])


def test_native_surface_fp16_replay_keeps_exact_addresses_and_support_and_reports_quantization(tmp_path):
    command, env = _surface_case(tmp_path)
    full, compactor, spans = _surface_replay(command, env)
    aux = compactor.aux_from_command().clone()
    compact = compactor.compact(full).half().float()
    restored = compactor.expand(compact, aux)
    assert torch.isfinite(restored).all()
    assert torch.equal(compactor.compact(restored), compact)
    assert aux.tolist() == [[1, 0], [0, 2]]
    for span in spans[2:]:
        assert torch.equal(restored[:, span.end - 6:span.end], full[:, span.end - 6:span.end])
    facts = command.part_surface_reference(offset=1)
    assert facts["reference_time_ns"].tolist() == [5_000_000_000_020_000_000, 4_000_000_000_040_000_000]
    print(json.dumps({"fp16_surface_replay_max_abs_error": {
        span.name: float((restored[:, span.start:span.end] - full[:, span.start:span.end]).abs().max())
        for span in spans}}, sort_keys=True))


@pytest.mark.parametrize("source", ["native_object_parts_wrist", "robot_wrist_pose_w"])
@pytest.mark.parametrize("loss", ["clip", "zero-scale"])
def test_native_surface_replay_rejects_lossy_source_charts(tmp_path, source, loss):
    command, env = _surface_case(tmp_path)
    with pytest.raises(ValueError, match="native surface replay cannot recover"):
        _surface_replay(command, env, changed_source=source,
                        clip=(-.1, .1) if loss == "clip" else None, scale=0. if loss == "zero-scale" else None)


def test_native_surface_checkpoint_metadata_does_not_depend_on_replay_compaction(tmp_path, monkeypatch):
    from agents import flashsac

    command, _env = _surface_case(tmp_path)
    command.part_surface_points(4, 16)
    monkeypatch.setattr(flashsac, "_native_policy_columns", lambda env, name: None)

    class MetadataAgent(flashsac.FlashSACAgent):
        @staticmethod
        def _resolved_config(cfg, share_obs):
            return SimpleNamespace(buffer_compact_obs=False)

        def __init__(self, observations, actions, env_info, cfg):
            self._encoded_layout, self._obs_terms = [], env_info.get("obs_terms", [])
            self._native_policy_columns = env_info.get("native_policy_columns")
            self._native_surface_metadata = env_info.get("native_part_surface_metadata", [])
            self._alg = SimpleNamespace(save=lambda: {}, update_step=0)

    environment = SimpleNamespace(share_obs=True, single_observation_space=None, single_action_space=None,
        unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command)))
    agent = MetadataAgent.build(None, environment,
        {"obs_terms": [["actor", "native_part_point_cloud_wrist", 78]]}, "motion")
    agent.save(str(tmp_path / "checkpoint"))
    layout = json.loads((tmp_path / "checkpoint/obs_layout.json").read_text())
    assert layout["native_part_surface_metadata"][0]["n_points"] == 4
    assert len(layout["native_part_surface_metadata"][0]["parts"]) == 3
    # A supported nontracking/noncompact agent still needs no command manager.
    environment.unwrapped = SimpleNamespace()
    unrelated = MetadataAgent.build(None, environment, {}, "unused")
    assert unrelated._native_surface_metadata == []


def test_native_surface_static_geometry_is_exported_once_through_normal_recorder(tmp_path):
    from test_native_rollout_record import record_reference_phase_fixture

    geometry, _env = _surface_case(tmp_path)
    source = dict(source_unit_id="synthetic", prepared_time_ns=[0, 20_000_000, 40_000_000],
        selected_source_frame_index=[0, 1, 2], prepared_source_ordinal=[0., 1., 2.],
        prepared_elapsed_time_ns=[0, 20_000_000, 40_000_000])
    binding = dict(scene_from_source_world=np.eye(4).tolist(),
        hands={side: dict(body_name=side + "_root", physical_joint_names={"driver": side + "_driver"})
               for side in ("left", "right")}, object_part_names=["body", "lid", "loose"],
        object_parts={name: dict(entity_name=entity, body_name=body) for name, entity, body in (
            ("body", "object", "object_root"), ("lid", "object", "object_lid"), ("loose", "loose_prop", "prop_root"))},
        clips={"synthetic": source})

    def setup(env, command):
        _add_loose_object(command, env, [])
        command.native_layout = geometry.native_layout
        command.part_surface_points(4, 16)

    manifest_path = record_reference_phase_fixture(tmp_path, binding, native=True, setup_scene=setup)
    manifest = json.loads(manifest_path.read_text())
    with np.load(manifest_path.parent / manifest["saved"][0]["rollout"], allow_pickle=False) as arrays:
        assert arrays["native_part_surface_metadata_json"].shape == ()
        assert not any(name.startswith("native_part_surface_point_cloud") for name in arrays.files)
        exported = json.loads(str(arrays["native_part_surface_metadata_json"]))
    assert exported[0]["parts"][2]["entity_name"] == "loose_prop"
    assert len(exported[0]["parts"][2]["points_body_m"]) == 4


def test_native_surface_profile_selects_semantic_terms_and_one_shared_reward(tmp_path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from envs.rewards import build_rewards

    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[1] / "config"), version_base=None):
        cfg = compose(config_name="native_surface", overrides=["commands.motion_file=synthetic.pt",
            "native_surface.n_points=12", "native_surface.n_obj_verts=48"])
        assert cfg.obs.obs_dict.actor_obs == cfg.obs.obs_dict.critic_obs
        for name in ("native_part_point_cloud_wrist", "native_next_part_point_cloud_delta_wrist"):
            assert name in cfg.obs.obs_dict.actor_obs
            assert cfg.obs.obs_params[name].n_points == 12 and cfg.obs.obs_params[name].n_obj_verts == 48
        terms = build_rewards(cfg.rewards)
        surface = terms["native_part_surface_displacement"]
        assert surface.weight == 1. and "side" not in surface.params
        assert terms["native_object_part_pose"].weight == 0.
        assert surface.params["topk"] == 3 and surface.params["norm_radius"] == .058
        assert surface.params["n_points"] == 12 and surface.params["n_obj_verts"] == 48
        assert cfg.runner.agent.buffer_compact_obs
        # This recipe does not silently change the separately configured assistance.
        base = compose(config_name="native", overrides=["commands.motion_file=synthetic.pt"])
        assert OmegaConf.to_container(cfg.commands.object) == OmegaConf.to_container(base.commands.object)
