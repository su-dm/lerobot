# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_rollout_top_level_imports():
    import lerobot.rollout

    for name in lerobot.rollout.__all__:
        assert hasattr(lerobot.rollout, name), f"Missing export: {name}"


def test_inference_submodule_imports():
    import lerobot.rollout.inference

    for name in lerobot.rollout.inference.__all__:
        assert hasattr(lerobot.rollout.inference, name), f"Missing export: {name}"


def test_strategies_submodule_imports():
    import lerobot.rollout.strategies

    for name in lerobot.rollout.strategies.__all__:
        assert hasattr(lerobot.rollout.strategies, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        BaseStrategyConfig,
        DAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    assert BaseStrategyConfig().type == "base"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"
    assert EpisodicStrategyConfig().type == "episodic"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard' or 'pedal'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.input_device == "keyboard"


def test_inference_config_types():
    from lerobot.rollout import RTCInferenceConfig, SyncInferenceConfig

    assert SyncInferenceConfig().type == "sync"

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        BaseStrategy,
        BaseStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        EpisodicStrategy,
        EpisodicStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)
    assert isinstance(create_strategy(EpisodicStrategyConfig()), EpisodicStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


def _make_sync_engine(resize: bool):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.rollout import SyncInferenceConfig, create_inference_engine

    policy = MagicMock()
    policy.config.input_features = {
        "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    return create_inference_engine(
        SyncInferenceConfig(resize_observation_images=resize),
        policy=policy,
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )


def test_sync_engine_resize_disabled_by_default():
    engine = _make_sync_engine(resize=False)
    assert engine._image_resize_shapes is None


def test_sync_engine_resize_shapes_from_policy_visual_features():
    engine = _make_sync_engine(resize=True)
    # Only VISUAL features contribute resize targets; state must be excluded.
    assert engine._image_resize_shapes == {"observation.images.cam": (3, 240, 320)}


# ---------------------------------------------------------------------------
# Chunked action cache (finding D)
# ---------------------------------------------------------------------------


def _make_act_policy_config(n_action_steps=50, temporal_ensemble_coeff=None):
    cfg = MagicMock()
    cfg.type = "act"
    cfg.n_action_steps = n_action_steps
    cfg.temporal_ensemble_coeff = temporal_ensemble_coeff
    return cfg


def _make_chunk_engine(policy_config, *, chunked=True):
    from lerobot.rollout import SyncInferenceConfig, create_inference_engine

    policy = MagicMock()
    policy.config = policy_config
    return create_inference_engine(
        SyncInferenceConfig(chunked_action_cache=chunked),
        policy=policy,
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )


def test_chunked_cache_disabled_by_default():
    engine = _make_chunk_engine(_make_act_policy_config(), chunked=False)
    assert engine._chunk_action_steps is None


def test_chunked_cache_resolves_n_action_steps_for_act():
    engine = _make_chunk_engine(_make_act_policy_config(n_action_steps=42))
    assert engine._chunk_action_steps == 42


def test_chunked_cache_rejects_non_act_policy():
    cfg = MagicMock()
    cfg.type = "diffusion"
    with pytest.raises(ValueError, match="only ACT policies"):
        _make_chunk_engine(cfg)


def test_chunked_cache_rejects_temporal_ensembling():
    cfg = _make_act_policy_config(n_action_steps=1, temporal_ensemble_coeff=0.01)
    with pytest.raises(ValueError, match="temporal ensembling"):
        _make_chunk_engine(cfg)


def _build_chunk_engine_with_stub_policy(n_action_steps, chunk_len, action_dim):
    """Build a SyncInferenceEngine in chunked mode with a deterministic stub policy."""
    from lerobot.rollout import SyncInferenceEngine
    from lerobot.utils.constants import ACTION

    # predict_action_chunk returns row t == [t*10, t*10+1, ...] so order is checkable.
    chunk = torch.stack(
        [torch.arange(action_dim, dtype=torch.float32) + t * 10 for t in range(chunk_len)]
    ).unsqueeze(0)  # (1, chunk_len, action_dim)

    policy = MagicMock()
    policy.predict_action_chunk = MagicMock(return_value=chunk)

    action_names = [f"a{i}" for i in range(action_dim)]
    dataset_features = {ACTION: {"names": action_names, "dtype": "float32", "shape": (action_dim,)}}

    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=MagicMock(side_effect=lambda x: x),
        postprocessor=MagicMock(side_effect=lambda x: x),
        dataset_features=dataset_features,
        ordered_action_keys=action_names,
        task="test",
        device="cpu",
        robot_type="mock",
        chunk_action_steps=n_action_steps,
    )
    return engine, policy


def test_chunked_cache_serves_from_cache_and_refills():
    import numpy as np

    engine, policy = _build_chunk_engine_with_stub_policy(n_action_steps=3, chunk_len=5, action_dim=2)
    obs_frame = {"observation.state": np.zeros(2, dtype=np.float32)}

    # First call runs the policy and caches n_action_steps actions.
    a0 = engine.get_action(obs_frame)
    assert policy.predict_action_chunk.call_count == 1
    torch.testing.assert_close(a0, torch.tensor([0.0, 1.0]))

    # Next two calls are served from the cache (policy NOT re-run).
    a1 = engine.get_action(obs_frame)
    a2 = engine.get_action(obs_frame)
    assert policy.predict_action_chunk.call_count == 1
    torch.testing.assert_close(a1, torch.tensor([10.0, 11.0]))
    torch.testing.assert_close(a2, torch.tensor([20.0, 21.0]))

    # Cache exhausted -> policy runs again, slice restarts at row 0.
    a3 = engine.get_action(obs_frame)
    assert policy.predict_action_chunk.call_count == 2
    torch.testing.assert_close(a3, torch.tensor([0.0, 1.0]))


def test_chunked_cache_slices_to_n_action_steps():
    import numpy as np

    # chunk has 5 rows but n_action_steps=2 -> only 2 cached before refilling.
    engine, policy = _build_chunk_engine_with_stub_policy(n_action_steps=2, chunk_len=5, action_dim=2)
    obs_frame = {"observation.state": np.zeros(2, dtype=np.float32)}

    engine.get_action(obs_frame)
    engine.get_action(obs_frame)
    assert policy.predict_action_chunk.call_count == 1
    engine.get_action(obs_frame)  # cache empty after 2 -> refill
    assert policy.predict_action_chunk.call_count == 2


def test_chunked_cache_none_obs_returns_none_without_running_policy():
    engine, policy = _build_chunk_engine_with_stub_policy(n_action_steps=3, chunk_len=5, action_dim=2)
    assert engine.get_action(None) is None
    assert policy.predict_action_chunk.call_count == 0


def test_reset_clears_action_cache():
    import numpy as np

    engine, policy = _build_chunk_engine_with_stub_policy(n_action_steps=3, chunk_len=5, action_dim=2)
    obs_frame = {"observation.state": np.zeros(2, dtype=np.float32)}
    engine.get_action(obs_frame)  # fills cache (2 left)
    assert len(engine._action_cache) == 2
    engine.reset()
    assert len(engine._action_cache) == 0


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> PAUSED
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.PAUSED)

    # PAUSED -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}
